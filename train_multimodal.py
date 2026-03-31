# -*- coding: utf-8 -*-
import os
import re
import json
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import csv
import pickle
import sys 
import argparse 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# For imbalance problem
# ================= Focal Loss  =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: balance factor, used to reduce the relative weight of negative samples.
            A value of 0.25 means the initial weight for positive samples is three times that of negative samples (0.75 / 0.25).

            gamma: focusing parameter. The larger gamma is, the more the model concentrates on samples that are hard to classify.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # standard BCE Loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # p_t (possibility of correct prediction from the model)
        pt = torch.exp(-bce_loss)
        
        # alpha_t (for the imbalance)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # final Focal Loss
        # alpha * (1-pt)^gamma * log(pt)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ===========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal model training script")

    # --- input ---
    parser.add_argument('--preprocessed_dir', type=str, default="./multimodal_preprocessed", help='.pkl file directory')
    parser.add_argument('--bert_dir', type=str, required=True, help='Path to the BioClinicalBERT model')

    # --- output ---
    parser.add_argument('--output_dir', type=str, default="./multimodal_model_output", help='Directory for output model and metrics')

    # --- training parameter ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum BERT sequence length')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (Epochs)')
    parser.add_argument('--lr_bert', type=float, default=2e-5, help='Learning rate for the BERT layers')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='Learning rate for the MLP head')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Learning rate warmup ratio')
    parser.add_argument('--no_fp16', action='store_true', help='Disable FP16 mixed precision')

    # --- batch size ---
    parser.add_argument('--max_notes', type=int, default=100, help='Maximum number of notes processed per sample to prevent OOM')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size in number of patients (BATCH_SAMPLES)')
    parser.add_argument('--grad_acc', type=int, default=4, help='Gradient accumulation steps (GRAD_ACC_STEPS)')
    parser.add_argument('--note_chunk_size', type=int, default=32, help='Chunk size for internal BERT note processing (NOTE_CHUNK_SIZE)')

    args = parser.parse_args()
    return args
# ===============================================


# ========= Multimodal Dataset  =========

class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, outcome_cols, tabular_cols, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.outcome_cols = outcome_cols
        self.tabular_cols = tabular_cols
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Notes
        notes = row["NOTE_TEXT"]
        if not isinstance(notes, list): notes = [""]
        notes = [str(t) for t in notes if isinstance(t, str) and t.strip() != ""]
        if not notes: notes = [""]
            
        # 2. Labels
        y = row[self.outcome_cols].astype(float).values.astype(np.float32)
        
        # 3. Tabular
        tabular = row[self.tabular_cols].astype(float).values.astype(np.float32)
        
        stay_id = str(row["stay_id"])
        
        return {
            "notes": notes,
            "labels": y,
            "tabular": tabular,
            "stay_id": stay_id 
        }

def collate_fn(batch, tokenizer, max_length=512, limit_notes=None):
    B = len(batch)
    
    labels = [torch.tensor(x["labels"]) for x in batch]
    labels = torch.stack(labels, dim=0)
    
    tabular_data = [torch.tensor(x["tabular"]) for x in batch]
    tabular_data = torch.stack(tabular_data, dim=0)
    
    stay_ids = [x["stay_id"] for x in batch] 

    flat_texts = []
    sample_idx = []
    for i, item in enumerate(batch):
        notes = item["notes"]
        if limit_notes is not None and len(notes) > limit_notes:
            notes = notes[:limit_notes] # in
        for t in notes:
            flat_texts.append(t)
            sample_idx.append(i)

    if not flat_texts:
        flat_texts = [""] * B
        sample_idx = list(range(B))

    tokenized = tokenizer(
        flat_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    sample_idx = torch.tensor(sample_idx, dtype=torch.long)
    
    return tokenized, sample_idx, labels, B, tabular_data, stay_ids 

# ========= Model Architecture =========

# --- Text Encoder (Attention Pooling) ---
class MultiTaskBertAttentionPool(nn.Module):
    def __init__(self, model_dir, max_notes_per_fwd_pass=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        hidden = self.encoder.config.hidden_size
        self.max_notes_per_fwd_pass = max(1, int(max_notes_per_fwd_pass))

        self.attention_project = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.attention_context = nn.Parameter(torch.randn(hidden))
        nn.init.normal_(self.attention_context, std=0.02)
        
    def forward(self, input_ids, attention_mask, sample_idx, B):
        num_notes_total = input_ids.size(0)
        
        # 1. BERT encoding
        if num_notes_total <= self.max_notes_per_fwd_pass:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
        else:
            all_cls_outputs = []
            pbar_internal = trange(0, num_notes_total, self.max_notes_per_fwd_pass, disable=True)
            for i in pbar_internal:
                chunk_out = self.encoder(
                    input_ids=input_ids[i : i + self.max_notes_per_fwd_pass], 
                    attention_mask=attention_mask[i : i + self.max_notes_per_fwd_pass]
                )
                all_cls_outputs.append(chunk_out.last_hidden_state[:, 0, :])
            cls = torch.cat(all_cls_outputs, dim=0)
        
        # 2. Attention Pooling Aggregation
        H = cls.size(-1)
        device = cls.device
        pooled_outputs = torch.zeros((B, H), dtype=cls.dtype, device=device)

        for i in range(B):
            mask = (sample_idx == i)
            if not mask.any(): continue 
            sample_cls_vectors = cls[mask]
            u_i = self.attention_project(sample_cls_vectors)
            scores = torch.matmul(u_i, self.attention_context)
            weights = torch.softmax(scores, dim=0)
            pooled = torch.sum(sample_cls_vectors * weights.unsqueeze(1), dim=0)
            pooled_outputs[i] = pooled
            
        return pooled_outputs 

# --- MultiModalModel ---
class MultiModalModel(nn.Module):
    def __init__(self, model_dir, num_tabular_features, num_labels, max_notes_per_fwd_pass=32):
        super().__init__()
        
        # --- A: Text Encoder ---
        self.text_encoder = MultiTaskBertAttentionPool(
            model_dir, 
            max_notes_per_fwd_pass
        )
        text_hidden_size = self.text_encoder.encoder.config.hidden_size # 768
        
        # --- B: Structure Data Encoder ---
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128) # 输出: 128 维
        )
        tabular_hidden_size = 128
        
        # ---  C: Fusion & classification ---
        fusion_size = text_hidden_size + tabular_hidden_size # 768 + 128 = 896
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels) # final multitask output
        )

    def forward(self, input_ids, attention_mask, sample_idx, B, tabular_input, return_embeds=False): 
        
        text_features = self.text_encoder(input_ids, attention_mask, sample_idx, B) 
        tabular_features = self.tabular_encoder(tabular_input)
        fused_vector = torch.cat([text_features, tabular_features], dim=1) 
        logits = self.fusion_classifier(fused_vector)
        
        if return_embeds:
            return logits, text_features, tabular_features, fused_vector
        else:
            return logits

# ===============================================

# ========= train & eval  =========
def eval_loop(model, dloader, bce, outcome_cols, device, fp16, split="val"):
    model.eval()
    all_logits = []
    all_labels = []
    
    # list for t-sne
    all_text_embeds = []
    all_tabular_embeds = []
    all_fused_embeds = []
    all_stay_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dloader, desc=f"eval:{split}", leave=False)
        for tokenized, sample_idx, labels, B, tabular_batch, stay_ids in pbar:
            tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
            sample_idx = sample_idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            tabular_batch = tabular_batch.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    sample_idx=sample_idx, B=B,
                    tabular_input=tabular_batch,
                    return_embeds=True 
                )
                logits, text_feat, tab_feat, fused_feat = outputs
                
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            all_text_embeds.append(text_feat.detach().cpu())
            all_tabular_embeds.append(tab_feat.detach().cpu())
            all_fused_embeds.append(fused_feat.detach().cpu())
            all_stay_ids.extend(stay_ids) 
            
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    #  concatenate all embeddings
    all_text_embeds = torch.cat(all_text_embeds, dim=0).numpy()
    all_tabular_embeds = torch.cat(all_tabular_embeds, dim=0).numpy()
    all_fused_embeds = torch.cat(all_fused_embeds, dim=0).numpy()

    metrics = {}
    for j, col in enumerate(outcome_cols):
        y_true = all_labels[:, j]
        y_score = 1/(1+np.exp(-all_logits[:, j]))
        if len(np.unique(y_true)) < 2:
            auc, aupr = float("nan"), float("nan")
        else:
            try: auc = roc_auc_score(y_true, y_score)
            except Exception: auc = float("nan")
            try: aupr = average_precision_score(y_true, y_score)
            except Exception: aupr = float("nan")
        metrics[col] = {"AUROC": auc, "AUPRC": aupr}
        
    valid_auroc = [m["AUROC"] for m in metrics.values() if not np.isnan(m["AUROC"])]
    valid_auprc = [m["AUPRC"] for m in metrics.values() if not np.isnan(m["AUPRC"])]
    metrics["macro"] = {
        "AUROC": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
        "AUPRC": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
    }
    
    return metrics, all_logits, all_labels, all_text_embeds, all_tabular_embeds, all_fused_embeds, all_stay_ids

def pick_thresholds_from_val(val_logits, val_labels, outcome_cols):
    thresholds = {}
    probs = 1/(1+np.exp(-val_logits))
    for j, col in enumerate(outcome_cols):
        y_true = val_labels[:, j]
        if len(np.unique(y_true)) < 2:
            thresholds[col] = 0.5; continue
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05, 0.95, 19):
            y_pred = (probs[:, j] >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        thresholds[col] = float(best_t)
    return thresholds

def compute_f1s(logits, labels, thresholds, outcome_cols):
    probs = 1/(1+np.exp(-logits))
    f1s = {}
    for j, col in enumerate(outcome_cols):
        y_true = labels[:, j]
        t = thresholds.get(col, 0.5)
        y_pred = (probs[:, j] >= t).astype(int)
        f1s[col] = f1_score(y_true, y_pred, zero_division=0)
    f1s["macro"] = float(np.mean(list(f1s.values())))
    return f1s

# ========= main =========
def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    FP16 = not args.no_fp16 and DEVICE == "cuda"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- [1/4] Load preprocessed data ---
    print(">> [1/4] Load preprocessed data...")
    try:
        with open(os.path.join(args.preprocessed_dir, "train.pkl"), "rb") as f:
            train_df = pickle.load(f)
        with open(os.path.join(args.preprocessed_dir, "val.pkl"), "rb") as f:
            val_df = pickle.load(f)
        with open(os.path.join(args.preprocessed_dir, "test.pkl"), "rb") as f:
            test_df = pickle.load(f)
    except FileNotFoundError:
        print(f"[FATAL ERROR] No .pkl file found")
        sys.exit()

    all_cols = train_df.columns.tolist()
    OUTCOME_COLS_LIST = [c for c in all_cols if c.startswith('outcome_')]
    TABULAR_COLS_LIST = [c for c in all_cols if c not in OUTCOME_COLS_LIST and c not in ['stay_id', 'NOTE_TEXT']]
    print(f"   Found {len(OUTCOME_COLS_LIST)}  Outcomes.")
    print(f"   Found {len(TABULAR_COLS_LIST)} Tabular features.")
    
    # --- [2/4] Tokenizer and model ---
    print(">> [2/4] Tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
    num_labels = len(OUTCOME_COLS_LIST)
    num_tabular_features = len(TABULAR_COLS_LIST)

    model = MultiModalModel(
        args.bert_dir,
        num_tabular_features=num_tabular_features,
        num_labels=num_labels,
        max_notes_per_fwd_pass=args.note_chunk_size
    ).to(DEVICE)

    # --- [3/4] DataLoader ---
    print(">> [3/4] DataLoader...")
    train_ds = MultiModalDataset(train_df, OUTCOME_COLS_LIST, TABULAR_COLS_LIST, tokenizer, args.max_length)
    val_ds   = MultiModalDataset(val_df,   OUTCOME_COLS_LIST, TABULAR_COLS_LIST, tokenizer, args.max_length)
    test_ds  = MultiModalDataset(test_df,  OUTCOME_COLS_LIST, TABULAR_COLS_LIST, tokenizer, args.max_length)

    collate = lambda b: collate_fn(b, tokenizer, args.max_length, args.max_notes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

    # --- optimizer ---
    no_decay = ["bias", "LayerNorm.weight"]
    encoder_params = [
        {"params": [p for n, p in model.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "lr": args.lr_bert, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],       "lr": args.lr_bert, "weight_decay": 0.0},
    ]
    head_params = [
        {"params": model.tabular_encoder.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": model.fusion_classifier.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(encoder_params + head_params)

    total_steps = max(1, len(train_loader) // max(1, args.grad_acc)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=FP16)

    print(">> [Loss] Use Focal Loss to deal with the imbalance problem...")
    
    bce = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    
    # -------------------------------------------------------------

    best_macro_auprc = -1
    
    # --- [4/4] Start training ---
    print(">> [4/4] Start training ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train:epoch{epoch}", total=len(train_loader))
        optimizer.zero_grad(set_to_none=True)

        for step, (tokenized, sample_idx, labels, B, tabular_batch, _) in enumerate(pbar, start=1):
            tokenized = {k: v.to(DEVICE, non_blocking=True) for k, v in tokenized.items()}
            sample_idx = sample_idx.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            tabular_batch = tabular_batch.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=FP16):
                logits = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    sample_idx=sample_idx, B=B,
                    tabular_input=tabular_batch,
                    return_embeds=False 
                )
                loss = bce(logits, labels)
                if args.grad_acc > 1:
                    loss = loss / args.grad_acc

            scaler.scale(loss).backward()

            if step % args.grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- evaluate ---
        val_metrics, val_logits, val_labels, _, _, _, _ = eval_loop(model, val_loader, bce, OUTCOME_COLS_LIST, DEVICE, FP16, split="val")
        val_macro_auroc = val_metrics['macro']['AUROC']
        val_macro_auprc = val_metrics['macro']['AUPRC']
        print(f"[epoch {epoch}] VAL macro AUROC={val_macro_auroc:.4f}  AUPRC={val_macro_auprc:.4f}")

        thresholds = pick_thresholds_from_val(val_logits, val_labels, OUTCOME_COLS_LIST)
        f1s_val = compute_f1s(val_logits, val_labels, thresholds, OUTCOME_COLS_LIST)
        print(f"[epoch {epoch}] VAL macro F1={f1s_val['macro']:.4f}")


        if not np.isnan(val_macro_auprc) and val_macro_auprc > best_macro_auprc:
            best_macro_auprc = val_macro_auprc
            best_state = {
                "model": model.state_dict(),
                "thresholds": thresholds,
                "outcome_cols": OUTCOME_COLS_LIST,
                "tabular_cols": TABULAR_COLS_LIST
            }
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))
            with open(os.path.join(args.output_dir, "best_thresholds.json"), "w", encoding="utf-8") as f:
                json.dump(thresholds, f, ensure_ascii=False, indent=2)
            print(f"[epoch {epoch}] 保存最佳模型（macro AUPRC={val_macro_auprc:.4f}）")

    # --- test set eval ---
    print(">> Load the best weight and eval in test set ...")
    if not os.path.exists(os.path.join(args.output_dir, "best_model.pt")):
        print(">> 错误：未找到 best_model.pt。")
    else:
        state = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=DEVICE)
        model.load_state_dict(state["model"])
        best_thresholds = state["thresholds"]

        test_metrics, test_logits, test_labels, \
        test_text_embeds, test_tabular_embeds, test_fused_embeds, \
        test_stay_ids = eval_loop(model, test_loader, bce, OUTCOME_COLS_LIST, DEVICE, FP16, split="test")
        
        f1s_test = compute_f1s(test_logits, test_labels, best_thresholds, OUTCOME_COLS_LIST)
        print(f"[TEST] macro AUROC={test_metrics['macro']['AUROC']:.4f}  AUPRC={test_metrics['macro']['AUPRC']:.4f}  F1={f1s_test['macro']:.4f}")

        detail = {"macro": {"AUROC": test_metrics["macro"]["AUROC"], "AUPRC": test_metrics["macro"]["AUPRC"], "F1": f1s_test["macro"]}, "per_task": {}}
        for col in OUTCOME_COLS_LIST:
            detail["per_task"][col] = {
                "AUROC": test_metrics[col]["AUROC"], "AUPRC": test_metrics[col]["AUPRC"],
                "F1": f1s_test[col], "threshold": best_thresholds.get(col, 0.5)
            }
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)

        print(">> 完成。指标已写入：", os.path.join(args.output_dir, "test_metrics.json"))
        print(">> 最佳模型：", os.path.join(args.output_dir, "best_model.pt"))
        
        # Saving t-SNE embeddings
        print(">> Saving t-SNE embeddings...")
        try:
            # 1. Note Embeddings
            np.save(os.path.join(args.output_dir, "test_text_embeddings.npy"), test_text_embeds)
            
            # 2. Other Embeddings
            np.save(os.path.join(args.output_dir, "test_tabular_embeddings.npy"), test_tabular_embeds)
            np.save(os.path.join(args.output_dir, "test_fused_embeddings.npy"), test_fused_embeds)
            
            # 3. Labels
            np.save(os.path.join(args.output_dir, "test_labels.npy"), test_labels)

            # 4. stay_id
            with open(os.path.join(args.output_dir, "test_stay_ids.txt"), "w", encoding="utf-8") as f:
                for sid in test_stay_ids:
                    f.write(f"{sid}\n")
            
            print(f">> t-SNE embeddings: {args.output_dir}")
        except Exception as e:
            print(f"[Error] Saving t-SNE embeddings failed: {e}")


if __name__ == "__main__":
    main()