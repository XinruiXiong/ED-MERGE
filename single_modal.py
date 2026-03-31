# -*- coding: utf-8 -*-
"""
[支持消融实验: 可通过 --mode 选择 'multimodal', 'text_only', 'vitals_only', 'profile_only']
"""

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


# ================= [新增] Focal Loss 类定义 =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子，用来降低负样本权重的比例。
                   0.25 意味着正样本的初始权重是负样本的 3 倍 (0.75/0.25)。
            gamma: 聚焦参数。gamma越高，模型越只关注那些难分类的样本。
                   2.0 是论文推荐的默认值。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算标准的 BCE Loss (不求均值，保留每个样本的 loss)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算 p_t (模型预测正确的概率)
        pt = torch.exp(-bce_loss)
        
        # 计算 alpha_t (处理正负样本不平衡)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算最终 Focal Loss
        # 公式: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ===========================================================

# ========= 1. 参数解析 =========
def parse_args():
    parser = argparse.ArgumentParser(description="多模态模型训练脚本")
    
    # --- 实验模式 (核心修改) ---
    parser.add_argument('--mode', type=str, default='multimodal', 
                        choices=['multimodal', 'text_only', 'vitals_only', 'profile_only'],
                        help='选择训练模式: 全模态、纯文本、纯生命体征、纯静态档案')

    # --- 输入路径 ---
    parser.add_argument('--preprocessed_dir', type=str, default="./multimodal_preprocessed", help='步骤 1 的 .pkl 输出目录')
    parser.add_argument('--bert_dir', type=str, required=True, help='本地 BioClinicalBERT 模型的路径')
    
    # --- 输出路径 ---
    parser.add_argument('--output_dir', type=str, default="./multimodal_model_output", help='输出目录')
    
    # --- 训练参数 ---
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr_bert', type=float, default=2e-5)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--no_fp16', action='store_true')
    
    # --- 批次大小 / OOM 控制 ---
    parser.add_argument('--max_notes', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_acc', type=int, default=4)
    parser.add_argument('--note_chunk_size', type=int, default=32)
    
    args = parser.parse_args()
    return args

# ========= 2. Dataset =========
class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, outcome_cols, tabular_cols, tokenizer, max_length=512, use_text=True):
        self.df = df.reset_index(drop=True)
        self.outcome_cols = outcome_cols
        self.tabular_cols = tabular_cols
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_text = use_text # 是否使用文本

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Notes (如果不用文本，返回空列表即可)
        notes = [""]
        if self.use_text:
            raw_notes = row.get("NOTE_TEXT", [])
            if not isinstance(raw_notes, list): raw_notes = [""]
            processed_notes = [str(t) for t in raw_notes if isinstance(t, str) and t.strip() != ""]
            if processed_notes:
                notes = processed_notes
            
        # 2. Labels
        y = row[self.outcome_cols].astype(float).values.astype(np.float32)
        
        # 3. Tabular (只取选定的列)
        # 如果 tabular_cols 为空 (text_only模式)，这里会返回空数组，没关系
        tabular = np.array([], dtype=np.float32)
        if len(self.tabular_cols) > 0:
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
    
    labels = torch.stack([torch.tensor(x["labels"]) for x in batch], dim=0)
    stay_ids = [x["stay_id"] for x in batch] 
    
    # Tabular stacking
    # 检查是否有表格数据
    if batch[0]["tabular"].shape[0] > 0:
        tabular_data = torch.stack([torch.tensor(x["tabular"]) for x in batch], dim=0)
    else:
        tabular_data = None # 纯文本模式

    # Text stacking
    # 检查是否需要处理文本 (看由 Dataset 返回的内容)
    # 即使是空字符串 [""]，我们也处理它，但在模型 forward 时会跳过 BERT
    flat_texts = []
    sample_idx = []
    for i, item in enumerate(batch):
        notes = item["notes"]
        if limit_notes is not None and len(notes) > limit_notes:
            notes = notes[:limit_notes]
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

# ========= 3. 模型架构 =========

class MultiTaskBertAttentionPool(nn.Module):
    def __init__(self, model_dir, max_notes_per_fwd_pass=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        hidden = self.encoder.config.hidden_size
        self.max_notes_per_fwd_pass = max(1, int(max_notes_per_fwd_pass))
        self.attention_project = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.attention_context = nn.Parameter(torch.randn(hidden))
        nn.init.normal_(self.attention_context, std=0.02)
        
    def forward(self, input_ids, attention_mask, sample_idx, B):
        num_notes_total = input_ids.size(0)
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

class MultiModalModel(nn.Module):
    def __init__(self, model_dir, num_tabular_features, num_labels, max_notes_per_fwd_pass=32, use_text=True, use_tabular=True):
        super().__init__()
        self.use_text = use_text
        self.use_tabular = use_tabular
        
        fusion_input_size = 0
        
        # --- 路径 A: 文本 ---
        if self.use_text:
            self.text_encoder = MultiTaskBertAttentionPool(model_dir, max_notes_per_fwd_pass)
            text_hidden_size = self.text_encoder.encoder.config.hidden_size # 768
            fusion_input_size += text_hidden_size
        
        # --- 路径 B: 表格 ---
        if self.use_tabular:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(num_tabular_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128) # 128 维 embedding
            )
            tabular_hidden_size = 128
            fusion_input_size += tabular_hidden_size
        
        # --- 路径 C: 融合/分类 ---
        # 如果是单模态，"融合"层就是一个简单的分类头
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, sample_idx, B, tabular_input, return_embeds=False):
        vectors_to_fuse = []
        
        text_features = None
        tabular_features = None
        
        if self.use_text:
            text_features = self.text_encoder(input_ids, attention_mask, sample_idx, B) 
            vectors_to_fuse.append(text_features)
            
        if self.use_tabular:
            tabular_features = self.tabular_encoder(tabular_input)
            vectors_to_fuse.append(tabular_features)
            
        fused_vector = torch.cat(vectors_to_fuse, dim=1) 
        logits = self.fusion_classifier(fused_vector)
        
        if return_embeds:
            return logits, text_features, tabular_features, fused_vector
        else:
            return logits

# ===============================================

# ========= 训练 & 评估逻辑 =========
def eval_loop(model, dloader, bce, outcome_cols, device, fp16, split="val"):
    model.eval()
    all_logits, all_labels = [], []
    all_text, all_tab, all_fuse, all_ids = [], [], [], []
    
    with torch.no_grad():
        pbar = tqdm(dloader, desc=f"eval:{split}", leave=False)
        for tokenized, sample_idx, labels, B, tabular_batch, stay_ids in pbar:
            tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
            sample_idx = sample_idx.to(device, non_blocking=True)
            # 注意：如果是 text_only 模式，tabular_batch 是 None，但这没关系，模型里会处理
            if tabular_batch is not None:
                tabular_batch = tabular_batch.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    sample_idx=sample_idx, B=B,
                    tabular_input=tabular_batch,
                    return_embeds=True
                )
                logits, text_f, tab_f, fuse_f = outputs
                
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            if text_f is not None: all_text.append(text_f.detach().cpu())
            if tab_f is not None: all_tab.append(tab_f.detach().cpu())
            if fuse_f is not None: all_fuse.append(fuse_f.detach().cpu())
            all_ids.extend(stay_ids)
            
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # 安全拼接
    res_text = torch.cat(all_text, dim=0).numpy() if all_text else None
    res_tab = torch.cat(all_tab, dim=0).numpy() if all_tab else None
    res_fuse = torch.cat(all_fuse, dim=0).numpy() if all_fuse else None

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
    return metrics, all_logits, all_labels, res_text, res_tab, res_fuse, all_ids

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

# ========= Main =========
def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    FP16 = not args.no_fp16 and DEVICE == "cuda"
    
    # 修改输出目录，避免覆盖
    final_output_dir = os.path.join(args.output_dir, f"mode_{args.mode}")
    os.makedirs(final_output_dir, exist_ok=True)
    print(f">> 实验模式: {args.mode}")
    print(f">> 输出目录: {final_output_dir}")
    
    print(">> [1/4] 加载预处理数据...")
    try:
        with open(os.path.join(args.preprocessed_dir, "train.pkl"), "rb") as f: train_df = pickle.load(f)
        with open(os.path.join(args.preprocessed_dir, "val.pkl"), "rb") as f: val_df = pickle.load(f)
        with open(os.path.join(args.preprocessed_dir, "test.pkl"), "rb") as f: test_df = pickle.load(f)
    except FileNotFoundError:
        print(f"[FATAL ERROR] 未找到 .pkl 文件。")
        sys.exit()

    # --- 确定列 ---
    all_cols = train_df.columns.tolist()
    OUTCOME_COLS_LIST = [c for c in all_cols if c.startswith('outcome_')]
    RAW_TABULAR_COLS = [c for c in all_cols if c not in OUTCOME_COLS_LIST and c not in ['stay_id', 'NOTE_TEXT']]

    # --- [核心] 根据 mode 筛选特征 ---
    USE_TEXT = True
    USE_TABULAR = True
    SELECTED_TABULAR_COLS = []

    if args.mode == 'multimodal':
        # 全选
        USE_TEXT = True
        USE_TABULAR = True
        SELECTED_TABULAR_COLS = RAW_TABULAR_COLS
        
    elif args.mode == 'text_only':
        # 只要文本，不要表格
        USE_TEXT = True
        USE_TABULAR = False
        SELECTED_TABULAR_COLS = []
        
    elif args.mode == 'vitals_only':
        # 只要 Vitals (final_...)，不要文本
        USE_TEXT = False
        USE_TABULAR = True
        # 筛选出以 final_ 开头的列
        SELECTED_TABULAR_COLS = [c for c in RAW_TABULAR_COLS if c.startswith('final_')]
        
    elif args.mode == 'profile_only':
        USE_TEXT = False
        USE_TABULAR = True
        SELECTED_TABULAR_COLS = [
            c for c in RAW_TABULAR_COLS
            if c.startswith('cci_')
            or c.startswith('eci_')
            or c.startswith('chiefcom_')
            or c == 'age'
        ]

    print(f"   Features Info:")
    print(f"     - Use Text: {USE_TEXT}")
    print(f"     - Use Tabular: {USE_TABULAR}")
    print(f"     - Tabular Cols Count: {len(SELECTED_TABULAR_COLS)}")
    if USE_TABULAR and len(SELECTED_TABULAR_COLS) == 0:
        print("[ERROR] 选择了 Tabular 模式但没有找到对应的列！请检查数据列名。")
        sys.exit()

    # --- [2/4] 模型 ---
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
    num_labels = len(OUTCOME_COLS_LIST)
    num_tabular_features = len(SELECTED_TABULAR_COLS)

    model = MultiModalModel(
        args.bert_dir,
        num_tabular_features=num_tabular_features,
        num_labels=num_labels,
        max_notes_per_fwd_pass=args.note_chunk_size,
        use_text=USE_TEXT,
        use_tabular=USE_TABULAR
    ).to(DEVICE)

    # --- [3/4] DataLoader ---
    # 传入筛选后的 SELECTED_TABULAR_COLS
    train_ds = MultiModalDataset(train_df, OUTCOME_COLS_LIST, SELECTED_TABULAR_COLS, tokenizer, args.max_length, use_text=USE_TEXT)
    val_ds   = MultiModalDataset(val_df,   OUTCOME_COLS_LIST, SELECTED_TABULAR_COLS, tokenizer, args.max_length, use_text=USE_TEXT)
    test_ds  = MultiModalDataset(test_df,  OUTCOME_COLS_LIST, SELECTED_TABULAR_COLS, tokenizer, args.max_length, use_text=USE_TEXT)

    collate = lambda b: collate_fn(b, tokenizer, args.max_length, args.max_notes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

    # --- 优化器 ---
    # 根据模式动态添加参数
    params_group = []
    
    if USE_TEXT:
        # BERT 参数
        no_decay = ["bias", "LayerNorm.weight"]
        params_group.append({"params": [p for n, p in model.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "lr": args.lr_bert, "weight_decay": args.weight_decay})
        params_group.append({"params": [p for n, p in model.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],       "lr": args.lr_bert, "weight_decay": 0.0})
    
    # MLP 参数 (Tabular Encoder + Fusion Classifier)
    head_params_list = []
    if USE_TABULAR:
        head_params_list.extend(list(model.tabular_encoder.parameters()))
    
    head_params_list.extend(list(model.fusion_classifier.parameters()))
    
    params_group.append({"params": head_params_list, "lr": args.lr_head, "weight_decay": args.weight_decay})
    
    optimizer = torch.optim.AdamW(params_group)

    total_steps = max(1, len(train_loader) // max(1, args.grad_acc)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=FP16)
    print(">> [Loss] 使用 Focal Loss 处理类别不平衡...")
    
    bce = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    
    # -------------------------------------------------------------
    best_macro_auprc = -1
    
    # --- [4/4] 训练 ---
    print(">> 开始训练 ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train:epoch{epoch}", total=len(train_loader))
        optimizer.zero_grad(set_to_none=True)

        for step, (tokenized, sample_idx, labels, B, tabular_batch, _) in enumerate(pbar, start=1):
            tokenized = {k: v.to(DEVICE, non_blocking=True) for k, v in tokenized.items()}
            sample_idx = sample_idx.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if tabular_batch is not None:
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

        # --- 验证 ---
        val_metrics, val_logits, val_labels, _, _, _, _ = eval_loop(model, val_loader, bce, OUTCOME_COLS_LIST, DEVICE, FP16, split="val")
        val_macro_auroc = val_metrics['macro']['AUROC']
        val_macro_auprc = val_metrics['macro']['AUPRC']
        print(f"[epoch {epoch}] VAL macro AUROC={val_macro_auroc:.4f}  AUPRC={val_macro_auprc:.4f}")

        # 保存
        if not np.isnan(val_macro_auprc) and val_macro_auprc > best_macro_auprc:
            best_macro_auprc = val_macro_auprc
            best_state = {
                "model": model.state_dict(),
                "outcome_cols": OUTCOME_COLS_LIST,
                "config_mode": args.mode
            }
            torch.save(best_state, os.path.join(final_output_dir, "best_model.pt"))
            # 同时计算 thresholds 并保存
            thresh = pick_thresholds_from_val(val_logits, val_labels, OUTCOME_COLS_LIST)
            with open(os.path.join(final_output_dir, "best_thresholds.json"), "w", encoding="utf-8") as f:
                json.dump(thresh, f, indent=2)
            print(f"[epoch {epoch}] 保存最佳模型 ({args.mode})")

    # --- 测试 ---
    print(">> 测试集评估 ...")
    if not os.path.exists(os.path.join(final_output_dir, "best_model.pt")):
        print(">> 错误：未找到 best_model.pt。")
    else:
        state = torch.load(os.path.join(final_output_dir, "best_model.pt"), map_location=DEVICE)
        model.load_state_dict(state["model"])
        with open(os.path.join(final_output_dir, "best_thresholds.json"), 'r') as f:
            best_thresholds = json.load(f)

        test_metrics, test_logits, test_labels, _, _, _, _ = eval_loop(model, test_loader, bce, OUTCOME_COLS_LIST, DEVICE, FP16, split="test")
        f1s_test = compute_f1s(test_logits, test_labels, best_thresholds, OUTCOME_COLS_LIST)
        
        print(f"[TEST] ({args.mode}) macro AUROC={test_metrics['macro']['AUROC']:.4f} AUPRC={test_metrics['macro']['AUPRC']:.4f}")
        
        # 保存结果
        detail = {
            "mode": args.mode,
            "macro": {"AUROC": test_metrics["macro"]["AUROC"], "AUPRC": test_metrics["macro"]["AUPRC"], "F1": f1s_test["macro"]}, 
            "per_task": {}
        }
        for col in OUTCOME_COLS_LIST:
            detail["per_task"][col] = {
                "AUROC": test_metrics[col]["AUROC"], "AUPRC": test_metrics[col]["AUPRC"],
                "F1": f1s_test[col]
            }
        with open(os.path.join(final_output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2)
        print(f">> 结果已写入: {final_output_dir}")

if __name__ == "__main__":
    main()