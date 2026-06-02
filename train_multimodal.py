# -*- coding: utf-8 -*-
import os
import json
import random
import pickle
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# ================= Focal Loss =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha:
                Class balance factor.
                A smaller alpha gives less weight to negative samples and more relative weight to positive samples.

            gamma:
                Focusing parameter.
                A larger gamma makes the model focus more on hard-to-classify samples.

            reduction:
                One of: "mean", "sum", or "none".
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Standard binary cross entropy loss without reduction.
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none"
        )

        # pt is the probability assigned to the correct class.
        pt = torch.exp(-bce_loss)

        # Alpha weighting for class imbalance.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss.
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ================= Argument Parser =================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal model training script with sliding-window note processing"
    )

    # Input paths.
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default="./multimodal_preprocessed",
        help="Directory containing train.pkl, val.pkl, and test.pkl"
    )
    parser.add_argument(
        "--bert_dir",
        type=str,
        required=True,
        help="Path to the pretrained BERT model directory"
    )

    # Output path.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./multimodal_model_output",
        help="Directory for saving model checkpoints, metrics, and embeddings"
    )

    # Training parameters.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum BERT sequence length for each sliding-window chunk"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr_bert",
        type=float,
        default=2e-5,
        help="Learning rate for BERT encoder layers"
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-3,
        help="Learning rate for tabular encoder and classifier head"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for linear learning rate scheduler"
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable FP16 mixed precision training"
    )

    # Batch and note-processing parameters.
    parser.add_argument(
        "--max_notes",
        type=int,
        default=100,
        help="Maximum number of original notes per sample. Use -1 for no limit."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size in number of patients or samples"
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--note_chunk_size",
        type=int,
        default=32,
        help="Maximum number of note windows processed by BERT in one internal forward pass"
    )

    # Sliding-window parameters.
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="Number of overlapping tokens between adjacent sliding-window chunks"
    )
    parser.add_argument(
        "--max_windows_per_sample",
        type=int,
        default=200,
        help="Maximum number of sliding-window chunks per sample. Use -1 for no limit."
    )

    args = parser.parse_args()
    return args


# ================= Dataset =================
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

        # 1. Notes.
        #
        # Expected formats:
        #   - A list of strings: ["note 1", "note 2", ...]
        #   - A single string: "one note"
        #
        # Empty notes are replaced with [""] so that every sample has at least one text input.
        notes = row["NOTE_TEXT"]

        if isinstance(notes, list):
            notes = [
                str(note)
                for note in notes
                if isinstance(note, str) and note.strip() != ""
            ]
        elif isinstance(notes, str) and notes.strip() != "":
            notes = [notes]
        else:
            notes = [""]

        if not notes:
            notes = [""]

        # 2. Labels.
        labels = row[self.outcome_cols].astype(float).values.astype(np.float32)

        # 3. Tabular features.
        tabular = row[self.tabular_cols].astype(float).values.astype(np.float32)

        # 4. Stay ID.
        stay_id = str(row["stay_id"])

        return {
            "notes": notes,
            "labels": labels,
            "tabular": tabular,
            "stay_id": stay_id
        }


# ================= Collate Function with Sliding Window =================
def collate_fn(
    batch,
    tokenizer,
    max_length=512,
    limit_notes=None,
    doc_stride=128,
    max_windows_per_sample=None
):
    """
    Collate function with sliding-window tokenization.

    Old behavior:
        Each note was tokenized with truncation=True and max_length=512.
        Tokens after 512 were discarded.

    New behavior:
        Each long note is split into multiple overlapping windows.
        Each window is encoded independently by BERT.
        The sample_idx tensor maps every window back to its original sample.

    Example:
        Sample 0 has one long note that becomes 3 windows.
        Sample 1 has one short note that becomes 1 window.

        tokenized input rows:
            window 0 -> sample 0
            window 1 -> sample 0
            window 2 -> sample 0
            window 3 -> sample 1

        sample_idx:
            [0, 0, 0, 1]
    """
    batch_size = len(batch)

    labels = [torch.tensor(item["labels"]) for item in batch]
    labels = torch.stack(labels, dim=0)

    tabular_data = [torch.tensor(item["tabular"]) for item in batch]
    tabular_data = torch.stack(tabular_data, dim=0)

    stay_ids = [item["stay_id"] for item in batch]

    # Flatten original notes across the batch.
    #
    # flat_texts[j] is one original note.
    # flat_sample_idx[j] is the sample index that this note belongs to.
    flat_texts = []
    flat_sample_idx = []

    for sample_i, item in enumerate(batch):
        notes = item["notes"]

        if limit_notes is not None and limit_notes > 0 and len(notes) > limit_notes:
            notes = notes[:limit_notes]

        for note in notes:
            if isinstance(note, str) and note.strip() != "":
                flat_texts.append(note)
                flat_sample_idx.append(sample_i)

    # Safety fallback.
    if not flat_texts:
        flat_texts = [""] * batch_size
        flat_sample_idx = list(range(batch_size))

    # Sliding-window tokenization.
    #
    # return_overflowing_tokens=True makes the tokenizer return multiple chunks
    # for notes longer than max_length.
    #
    # overflow_to_sample_mapping maps each returned chunk to the index of the
    # original note inside flat_texts.
    tokenized = tokenizer(
        flat_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )

    overflow_mapping = tokenized.pop("overflow_to_sample_mapping")

    flat_sample_idx = torch.tensor(flat_sample_idx, dtype=torch.long)

    # Map each sliding-window chunk back to the patient/sample index.
    sample_idx = flat_sample_idx[overflow_mapping]

    # Optional safety limit to avoid GPU memory explosion.
    #
    # If max_windows_per_sample is set, each sample keeps only the first N windows.
    # This preserves sequence order but discards later windows if there are too many.
    if max_windows_per_sample is not None and max_windows_per_sample > 0:
        keep_indices = []
        counts = {sample_i: 0 for sample_i in range(batch_size)}

        for window_i, sample_i in enumerate(sample_idx.tolist()):
            if counts[sample_i] < max_windows_per_sample:
                keep_indices.append(window_i)
                counts[sample_i] += 1

        if len(keep_indices) == 0:
            keep_indices = [0]

        keep_indices = torch.tensor(keep_indices, dtype=torch.long)

        tokenized = {
            key: value[keep_indices]
            for key, value in tokenized.items()
        }

        sample_idx = sample_idx[keep_indices]

    return tokenized, sample_idx, labels, batch_size, tabular_data, stay_ids


# ================= Text Encoder with Attention Pooling =================
class MultiTaskBertAttentionPool(nn.Module):
    def __init__(self, model_dir, max_notes_per_fwd_pass=32):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_dir)
        hidden_size = self.encoder.config.hidden_size

        self.max_notes_per_fwd_pass = max(1, int(max_notes_per_fwd_pass))

        self.attention_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        self.attention_context = nn.Parameter(torch.randn(hidden_size))
        nn.init.normal_(self.attention_context, std=0.02)

    def forward(self, input_ids, attention_mask, sample_idx, batch_size):
        num_windows_total = input_ids.size(0)

        # 1. Encode each note window with BERT.
        #
        # Each row in input_ids is now one sliding-window chunk.
        # If there are many chunks, they are processed internally in smaller groups.
        if num_windows_total <= self.max_notes_per_fwd_pass:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_vectors = output.last_hidden_state[:, 0, :]
        else:
            all_cls_vectors = []

            internal_range = trange(
                0,
                num_windows_total,
                self.max_notes_per_fwd_pass,
                disable=True
            )

            for start_i in internal_range:
                end_i = start_i + self.max_notes_per_fwd_pass

                chunk_output = self.encoder(
                    input_ids=input_ids[start_i:end_i],
                    attention_mask=attention_mask[start_i:end_i]
                )

                all_cls_vectors.append(chunk_output.last_hidden_state[:, 0, :])

            cls_vectors = torch.cat(all_cls_vectors, dim=0)

        # 2. Attention pooling.
        #
        # For each sample, gather all its chunks and aggregate them into one text vector.
        hidden_size = cls_vectors.size(-1)
        device = cls_vectors.device

        pooled_outputs = torch.zeros(
            (batch_size, hidden_size),
            dtype=cls_vectors.dtype,
            device=device
        )

        for sample_i in range(batch_size):
            mask = sample_idx == sample_i

            if not mask.any():
                continue

            sample_cls_vectors = cls_vectors[mask]

            projected = self.attention_project(sample_cls_vectors)
            scores = torch.matmul(projected, self.attention_context)
            weights = torch.softmax(scores, dim=0)

            pooled = torch.sum(sample_cls_vectors * weights.unsqueeze(1), dim=0)
            pooled_outputs[sample_i] = pooled

        return pooled_outputs


# ================= Multimodal Model =================
class MultiModalModel(nn.Module):
    def __init__(
        self,
        model_dir,
        num_tabular_features,
        num_labels,
        max_notes_per_fwd_pass=32
    ):
        super().__init__()

        # Text encoder.
        self.text_encoder = MultiTaskBertAttentionPool(
            model_dir=model_dir,
            max_notes_per_fwd_pass=max_notes_per_fwd_pass
        )

        text_hidden_size = self.text_encoder.encoder.config.hidden_size

        # Tabular encoder.
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        tabular_hidden_size = 128

        # Fusion classifier.
        fusion_size = text_hidden_size + tabular_hidden_size

        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        sample_idx,
        batch_size,
        tabular_input,
        return_embeds=False
    ):
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_idx=sample_idx,
            batch_size=batch_size
        )

        tabular_features = self.tabular_encoder(tabular_input)

        fused_vector = torch.cat(
            [text_features, tabular_features],
            dim=1
        )

        logits = self.fusion_classifier(fused_vector)

        if return_embeds:
            return logits, text_features, tabular_features, fused_vector

        return logits


# ================= Evaluation Loop =================
def eval_loop(model, dataloader, loss_fn, outcome_cols, device, fp16, split="val"):
    model.eval()

    all_logits = []
    all_labels = []

    all_text_embeds = []
    all_tabular_embeds = []
    all_fused_embeds = []
    all_stay_ids = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc=f"eval:{split}",
            leave=False
        )

        for tokenized, sample_idx, labels, batch_size, tabular_batch, stay_ids in progress_bar:
            tokenized = {
                key: value.to(device, non_blocking=True)
                for key, value in tokenized.items()
            }

            sample_idx = sample_idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            tabular_batch = tabular_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    sample_idx=sample_idx,
                    batch_size=batch_size,
                    tabular_input=tabular_batch,
                    return_embeds=True
                )

                logits, text_features, tabular_features, fused_features = outputs

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            all_text_embeds.append(text_features.detach().cpu())
            all_tabular_embeds.append(tabular_features.detach().cpu())
            all_fused_embeds.append(fused_features.detach().cpu())
            all_stay_ids.extend(stay_ids)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    all_text_embeds = torch.cat(all_text_embeds, dim=0).numpy()
    all_tabular_embeds = torch.cat(all_tabular_embeds, dim=0).numpy()
    all_fused_embeds = torch.cat(all_fused_embeds, dim=0).numpy()

    metrics = {}

    for label_i, col in enumerate(outcome_cols):
        y_true = all_labels[:, label_i]
        y_score = 1 / (1 + np.exp(-all_logits[:, label_i]))

        if len(np.unique(y_true)) < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            try:
                auroc = roc_auc_score(y_true, y_score)
            except Exception:
                auroc = float("nan")

            try:
                auprc = average_precision_score(y_true, y_score)
            except Exception:
                auprc = float("nan")

        metrics[col] = {
            "AUROC": auroc,
            "AUPRC": auprc
        }

    valid_aurocs = [
        metric["AUROC"]
        for metric in metrics.values()
        if not np.isnan(metric["AUROC"])
    ]

    valid_auprcs = [
        metric["AUPRC"]
        for metric in metrics.values()
        if not np.isnan(metric["AUPRC"])
    ]

    metrics["macro"] = {
        "AUROC": float(np.mean(valid_aurocs)) if valid_aurocs else float("nan"),
        "AUPRC": float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")
    }

    return (
        metrics,
        all_logits,
        all_labels,
        all_text_embeds,
        all_tabular_embeds,
        all_fused_embeds,
        all_stay_ids
    )


# ================= Threshold Selection =================
def pick_thresholds_from_val(val_logits, val_labels, outcome_cols):
    thresholds = {}

    probs = 1 / (1 + np.exp(-val_logits))

    for label_i, col in enumerate(outcome_cols):
        y_true = val_labels[:, label_i]

        if len(np.unique(y_true)) < 2:
            thresholds[col] = 0.5
            continue

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.linspace(0.05, 0.95, 19):
            y_pred = (probs[:, label_i] >= threshold).astype(int)

            current_f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        thresholds[col] = float(best_threshold)

    return thresholds


def compute_f1s(logits, labels, thresholds, outcome_cols):
    probs = 1 / (1 + np.exp(-logits))

    f1s = {}

    for label_i, col in enumerate(outcome_cols):
        y_true = labels[:, label_i]
        threshold = thresholds.get(col, 0.5)

        y_pred = (probs[:, label_i] >= threshold).astype(int)

        f1s[col] = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

    f1s["macro"] = float(np.mean(list(f1s.values())))

    return f1s


# ================= Main Function =================
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    fp16 = not args.no_fp16 and device == "cuda"

    os.makedirs(args.output_dir, exist_ok=True)

    # Interpret -1 as no note limit.
    if args.max_notes is None or args.max_notes < 0:
        max_notes = None
    else:
        max_notes = args.max_notes

    # Interpret -1 as no sliding-window limit.
    if args.max_windows_per_sample is None or args.max_windows_per_sample < 0:
        max_windows_per_sample = None
    else:
        max_windows_per_sample = args.max_windows_per_sample

    # ---------------- Load Data ----------------
    print(">> [1/4] Loading preprocessed data...")

    try:
        with open(os.path.join(args.preprocessed_dir, "train.pkl"), "rb") as f:
            train_df = pickle.load(f)

        with open(os.path.join(args.preprocessed_dir, "val.pkl"), "rb") as f:
            val_df = pickle.load(f)

        with open(os.path.join(args.preprocessed_dir, "test.pkl"), "rb") as f:
            test_df = pickle.load(f)

    except FileNotFoundError:
        print(f"[Fatal Error] Could not find train.pkl, val.pkl, or test.pkl in: {args.preprocessed_dir}")
        sys.exit(1)

    all_cols = train_df.columns.tolist()

    outcome_cols = [
        col for col in all_cols
        if col.startswith("outcome_")
    ]

    tabular_cols = [
        col for col in all_cols
        if col not in outcome_cols and col not in ["stay_id", "NOTE_TEXT"]
    ]

    print(f"   Number of outcome columns: {len(outcome_cols)}")
    print(f"   Number of tabular features: {len(tabular_cols)}")
    print(f"   Device: {device}")
    print(f"   FP16 enabled: {fp16}")
    print(f"   max_length: {args.max_length}")
    print(f"   doc_stride: {args.doc_stride}")
    print(f"   max_notes: {max_notes}")
    print(f"   max_windows_per_sample: {max_windows_per_sample}")

    # ---------------- Tokenizer and Model ----------------
    print(">> [2/4] Initializing tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_dir,
        use_fast=True
    )

    num_labels = len(outcome_cols)
    num_tabular_features = len(tabular_cols)

    model = MultiModalModel(
        model_dir=args.bert_dir,
        num_tabular_features=num_tabular_features,
        num_labels=num_labels,
        max_notes_per_fwd_pass=args.note_chunk_size
    ).to(device)

    # ---------------- Data Loaders ----------------
    print(">> [3/4] Building datasets and dataloaders...")

    train_dataset = MultiModalDataset(
        df=train_df,
        outcome_cols=outcome_cols,
        tabular_cols=tabular_cols,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    val_dataset = MultiModalDataset(
        df=val_df,
        outcome_cols=outcome_cols,
        tabular_cols=tabular_cols,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    test_dataset = MultiModalDataset(
        df=test_df,
        outcome_cols=outcome_cols,
        tabular_cols=tabular_cols,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    collate = lambda batch: collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit_notes=max_notes,
        doc_stride=args.doc_stride,
        max_windows_per_sample=max_windows_per_sample
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True
    )

    # ---------------- Optimizer and Scheduler ----------------
    no_decay = ["bias", "LayerNorm.weight"]

    encoder_params = [
        {
            "params": [
                param
                for name, param in model.text_encoder.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            "lr": args.lr_bert,
            "weight_decay": args.weight_decay
        },
        {
            "params": [
                param
                for name, param in model.text_encoder.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            "lr": args.lr_bert,
            "weight_decay": 0.0
        }
    ]

    head_params = [
        {
            "params": model.tabular_encoder.parameters(),
            "lr": args.lr_head,
            "weight_decay": args.weight_decay
        },
        {
            "params": model.fusion_classifier.parameters(),
            "lr": args.lr_head,
            "weight_decay": args.weight_decay
        }
    ]

    optimizer = torch.optim.AdamW(
        encoder_params + head_params
    )

    total_steps = max(
        1,
        len(train_loader) // max(1, args.grad_acc)
    ) * args.epochs

    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=fp16
    )

    print(">> Using focal loss for class imbalance.")

    loss_fn = FocalLoss(
        alpha=0.25,
        gamma=2.0
    ).to(device)

    best_macro_auprc = -1.0

    # ---------------- Training ----------------
    print(">> [4/4] Starting training...")

    for epoch in range(1, args.epochs + 1):
        model.train()

        progress_bar = tqdm(
            train_loader,
            desc=f"train:epoch{epoch}",
            total=len(train_loader)
        )

        optimizer.zero_grad(set_to_none=True)

        for step, batch_data in enumerate(progress_bar, start=1):
            tokenized, sample_idx, labels, batch_size, tabular_batch, _ = batch_data

            tokenized = {
                key: value.to(device, non_blocking=True)
                for key, value in tokenized.items()
            }

            sample_idx = sample_idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            tabular_batch = tabular_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    sample_idx=sample_idx,
                    batch_size=batch_size,
                    tabular_input=tabular_batch,
                    return_embeds=False
                )

                loss = loss_fn(
                    logits,
                    labels
                )

                if args.grad_acc > 1:
                    loss = loss / args.grad_acc

            scaler.scale(loss).backward()

            if step % args.grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "windows": int(tokenized["input_ids"].size(0))
            })

        # If the last mini-batch did not trigger an optimizer step because of gradient accumulation,
        # apply the remaining gradients.
        if len(train_loader) % args.grad_acc != 0:
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # ---------------- Validation ----------------
        val_metrics, val_logits, val_labels, _, _, _, _ = eval_loop(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            outcome_cols=outcome_cols,
            device=device,
            fp16=fp16,
            split="val"
        )

        val_macro_auroc = val_metrics["macro"]["AUROC"]
        val_macro_auprc = val_metrics["macro"]["AUPRC"]

        print(
            f"[Epoch {epoch}] "
            f"Validation macro AUROC={val_macro_auroc:.4f}, "
            f"AUPRC={val_macro_auprc:.4f}"
        )

        thresholds = pick_thresholds_from_val(
            val_logits=val_logits,
            val_labels=val_labels,
            outcome_cols=outcome_cols
        )

        val_f1s = compute_f1s(
            logits=val_logits,
            labels=val_labels,
            thresholds=thresholds,
            outcome_cols=outcome_cols
        )

        print(
            f"[Epoch {epoch}] "
            f"Validation macro F1={val_f1s['macro']:.4f}"
        )

        # Save best model by validation macro AUPRC.
        if not np.isnan(val_macro_auprc) and val_macro_auprc > best_macro_auprc:
            best_macro_auprc = val_macro_auprc

            best_state = {
                "model": model.state_dict(),
                "thresholds": thresholds,
                "outcome_cols": outcome_cols,
                "tabular_cols": tabular_cols,
                "max_length": args.max_length,
                "doc_stride": args.doc_stride,
                "max_notes": max_notes,
                "max_windows_per_sample": max_windows_per_sample
            }

            best_model_path = os.path.join(
                args.output_dir,
                "best_model.pt"
            )

            torch.save(
                best_state,
                best_model_path
            )

            thresholds_path = os.path.join(
                args.output_dir,
                "best_thresholds.json"
            )

            with open(thresholds_path, "w", encoding="utf-8") as f:
                json.dump(
                    thresholds,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                f"[Epoch {epoch}] Saved best model "
                f"with validation macro AUPRC={val_macro_auprc:.4f}"
            )

    # ---------------- Test Evaluation ----------------
    print(">> Loading the best model and evaluating on the test set...")

    best_model_path = os.path.join(
        args.output_dir,
        "best_model.pt"
    )

    if not os.path.exists(best_model_path):
        print("[Error] best_model.pt was not found. Test evaluation is skipped.")
        return

    state = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        state["model"]
    )

    best_thresholds = state["thresholds"]

    (
        test_metrics,
        test_logits,
        test_labels,
        test_text_embeds,
        test_tabular_embeds,
        test_fused_embeds,
        test_stay_ids
    ) = eval_loop(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        outcome_cols=outcome_cols,
        device=device,
        fp16=fp16,
        split="test"
    )

    test_f1s = compute_f1s(
        logits=test_logits,
        labels=test_labels,
        thresholds=best_thresholds,
        outcome_cols=outcome_cols
    )

    print(
        f"[Test] "
        f"macro AUROC={test_metrics['macro']['AUROC']:.4f}, "
        f"AUPRC={test_metrics['macro']['AUPRC']:.4f}, "
        f"F1={test_f1s['macro']:.4f}"
    )

    detail = {
        "macro": {
            "AUROC": test_metrics["macro"]["AUROC"],
            "AUPRC": test_metrics["macro"]["AUPRC"],
            "F1": test_f1s["macro"]
        },
        "per_task": {}
    }

    for col in outcome_cols:
        detail["per_task"][col] = {
            "AUROC": test_metrics[col]["AUROC"],
            "AUPRC": test_metrics[col]["AUPRC"],
            "F1": test_f1s[col],
            "threshold": best_thresholds.get(col, 0.5)
        }

    test_metrics_path = os.path.join(
        args.output_dir,
        "test_metrics.json"
    )

    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            detail,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f">> Test metrics were saved to: {test_metrics_path}")
    print(f">> Best model path: {best_model_path}")

    # ---------------- Save Embeddings for Visualization ----------------
    print(">> Saving embeddings for downstream visualization...")

    try:
        np.save(
            os.path.join(args.output_dir, "test_text_embeddings.npy"),
            test_text_embeds
        )

        np.save(
            os.path.join(args.output_dir, "test_tabular_embeddings.npy"),
            test_tabular_embeds
        )

        np.save(
            os.path.join(args.output_dir, "test_fused_embeddings.npy"),
            test_fused_embeds
        )

        np.save(
            os.path.join(args.output_dir, "test_labels.npy"),
            test_labels
        )

        stay_id_path = os.path.join(
            args.output_dir,
            "test_stay_ids.txt"
        )

        with open(stay_id_path, "w", encoding="utf-8") as f:
            for stay_id in test_stay_ids:
                f.write(f"{stay_id}\n")

        print(f">> Embeddings were saved to: {args.output_dir}")

    except Exception as error:
        print(f"[Error] Failed to save embeddings: {error}")


if __name__ == "__main__":
    main()