#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1, Stage 1: lightly fine-tune BioClinicalBERT on the new event-stream
data before freezing it for embedding precomputation (Stage 2).

Architecture: BERT + attention-pooling text encoder, tabular MLP, fusion
classifier, focal loss, differential learning rates. The training snapshot
comes from: instead of loading a pre-built fixed-window
train.pkl, each training example is assembled on the fly from the Phase 0
event-stream tables at a RANDOMLY SAMPLED landmark time (resampled every
epoch). This means even this intermediate stage is already training on
varied landmarks rather than one fixed window, even though it is not yet
the sequence model (that's Stage 2 + the sequence model itself).

Only the fine-tuned BERT encoder is consumed downstream (by
precompute_note_embeddings.py) -- the attention-pool/tabular/fusion modules
here exist only to produce a training signal for BERT, they are NOT the
final model.

Two stages instead of joint end-to-end fine-tuning inside the sequence
model: re-running BERT on a changing note subset every epoch under random
landmark sampling is not affordable at this cohort's scale, so BERT is
adapted once here, then frozen and used as a cached feature extractor.
"""

import os
import sys
import json
import random
import pickle
import argparse
import functools

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg
from common.focal_loss import FocalLoss
from common.event_utils import EncounterEventIndex, sample_landmark_minutes


# ================= Dataset =================
class SnapshotAtLandmarkDataset(Dataset):
    """One row per encounter. Landmark time is either resampled every epoch
    (train) or sampled once at construction and held fixed (val/test).
    """

    def __init__(self, static_df, index: EncounterEventIndex, outcome_cols, static_cols,
                 max_landmark_minutes, seed, resample_each_epoch: bool):
        self.static_df = static_df.reset_index(drop=True)
        self.index = index
        self.outcome_cols = outcome_cols
        self.static_cols = static_cols
        self.max_landmark_minutes = max_landmark_minutes
        self.resample_each_epoch = resample_each_epoch
        self.rng = np.random.default_rng(seed)
        self.landmarks = np.zeros(len(self.static_df), dtype=np.float64)
        self.resample_landmarks()

    def resample_landmarks(self):
        data_boundary = self.static_df["data_boundary_minutes"].to_numpy()
        for i in range(len(self.static_df)):
            self.landmarks[i] = sample_landmark_minutes(data_boundary[i], self.max_landmark_minutes, self.rng)

    def __len__(self):
        return len(self.static_df)

    def __getitem__(self, idx):
        row = self.static_df.iloc[idx]
        stay_id = str(row["stay_id"])
        T = float(self.landmarks[idx])

        notes = self.index.notes_upto(stay_id, T)
        notes = [str(n) for n in notes if isinstance(n, str) and n.strip()]
        if not notes:
            notes = [""]

        vitals_state = self.index.vitals_state_at(stay_id, T)
        static_vals = row[self.static_cols].to_numpy(dtype=np.float32)
        tabular = np.concatenate([static_vals, vitals_state]).astype(np.float32)

        labels = row[self.outcome_cols].to_numpy(dtype=np.float32)

        return {"notes": notes, "labels": labels, "tabular": tabular, "stay_id": stay_id}


def collate_fn(batch, tokenizer, max_length=512, limit_notes=100, doc_stride=128, max_windows_per_sample=200):
    """Sliding-window tokenization: notes longer than max_length are split
    into overlapping windows (doc_stride) rather than truncated.
    """
    batch_size = len(batch)
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch], dim=0)
    tabular_data = torch.stack([torch.tensor(item["tabular"]) for item in batch], dim=0)
    stay_ids = [item["stay_id"] for item in batch]

    flat_texts, flat_sample_idx = [], []
    for sample_i, item in enumerate(batch):
        notes = item["notes"]
        if limit_notes is not None and limit_notes > 0 and len(notes) > limit_notes:
            notes = notes[:limit_notes]
        for note in notes:
            if isinstance(note, str) and note.strip() != "":
                flat_texts.append(note)
                flat_sample_idx.append(sample_i)

    if not flat_texts:
        flat_texts = [""] * batch_size
        flat_sample_idx = list(range(batch_size))

    tokenized = tokenizer(
        flat_texts, padding=True, truncation=True, max_length=max_length,
        stride=doc_stride, return_overflowing_tokens=True, return_tensors="pt",
    )
    overflow_mapping = tokenized.pop("overflow_to_sample_mapping")
    flat_sample_idx = torch.tensor(flat_sample_idx, dtype=torch.long)
    sample_idx = flat_sample_idx[overflow_mapping]

    if max_windows_per_sample is not None and max_windows_per_sample > 0:
        keep_indices, counts = [], {i: 0 for i in range(batch_size)}
        for window_i, sample_i in enumerate(sample_idx.tolist()):
            if counts[sample_i] < max_windows_per_sample:
                keep_indices.append(window_i)
                counts[sample_i] += 1
        if not keep_indices:
            keep_indices = [0]
        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        tokenized = {k: v[keep_indices] for k, v in tokenized.items()}
        sample_idx = sample_idx[keep_indices]

    return tokenized, sample_idx, labels, batch_size, tabular_data, stay_ids


# ================= Model =================
class MultiTaskBertAttentionPool(nn.Module):
    def __init__(self, model_dir, max_notes_per_fwd_pass=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        hidden_size = self.encoder.config.hidden_size
        self.max_notes_per_fwd_pass = max(1, int(max_notes_per_fwd_pass))
        self.attention_project = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.attention_context = nn.Parameter(torch.randn(hidden_size))
        nn.init.normal_(self.attention_context, std=0.02)

    def forward(self, input_ids, attention_mask, sample_idx, batch_size):
        num_windows_total = input_ids.size(0)
        if num_windows_total <= self.max_notes_per_fwd_pass:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_vectors = output.last_hidden_state[:, 0, :]
        else:
            chunks = []
            for start_i in range(0, num_windows_total, self.max_notes_per_fwd_pass):
                end_i = start_i + self.max_notes_per_fwd_pass
                out = self.encoder(input_ids=input_ids[start_i:end_i], attention_mask=attention_mask[start_i:end_i])
                chunks.append(out.last_hidden_state[:, 0, :])
            cls_vectors = torch.cat(chunks, dim=0)

        hidden_size = cls_vectors.size(-1)
        pooled = torch.zeros((batch_size, hidden_size), dtype=cls_vectors.dtype, device=cls_vectors.device)
        for sample_i in range(batch_size):
            mask = sample_idx == sample_i
            if not mask.any():
                continue
            sample_cls = cls_vectors[mask]
            projected = self.attention_project(sample_cls)
            scores = torch.matmul(projected, self.attention_context)
            weights = torch.softmax(scores, dim=0)
            pooled[sample_i] = torch.sum(sample_cls * weights.unsqueeze(1), dim=0)
        return pooled


class Stage1Model(nn.Module):
    def __init__(self, model_dir, num_tabular_features, num_labels, max_notes_per_fwd_pass=32):
        super().__init__()
        self.text_encoder = MultiTaskBertAttentionPool(model_dir, max_notes_per_fwd_pass)
        text_hidden = self.text_encoder.encoder.config.hidden_size
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(text_hidden + 128, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask, sample_idx, batch_size, tabular_input):
        text_features = self.text_encoder(input_ids, attention_mask, sample_idx, batch_size)
        tabular_features = self.tabular_encoder(tabular_input)
        fused = torch.cat([text_features, tabular_features], dim=1)
        return self.fusion_classifier(fused)


def eval_loop(model, loader, outcome_cols, device, fp16):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for tokenized, sample_idx, labels, B, tabular_batch, _ in tqdm(loader, desc="eval", leave=False):
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            sample_idx = sample_idx.to(device)
            labels = labels.to(device)
            tabular_batch = tabular_batch.to(device)
            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(tokenized["input_ids"], tokenized["attention_mask"], sample_idx, B, tabular_batch)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    aurocs, auprcs = [], []
    for j in range(len(outcome_cols)):
        y_true = labels[:, j]
        if len(np.unique(y_true)) < 2:
            continue
        y_score = 1 / (1 + np.exp(-logits[:, j]))
        aurocs.append(roc_auc_score(y_true, y_score))
        auprcs.append(average_precision_score(y_true, y_score))
    macro_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    macro_auprc = float(np.mean(auprcs)) if auprcs else float("nan")
    return macro_auroc, macro_auprc


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 1 Stage 1: light BERT fine-tune on random-landmark snapshots.")
    ap.add_argument("--event_dir", type=str, default=cfg.EVENT_STREAM_OUTPUT_DIR_PLACEHOLDER, required=True,
                     help="Output dir from phase0_data/build_event_streams.py")
    ap.add_argument("--bert_dir", type=str, default=cfg.BERT_DIR_PLACEHOLDER, required=True)
    ap.add_argument("--output_dir", type=str, default=cfg.STAGE1_BERT_OUTPUT_DIR_PLACEHOLDER, required=True)

    ap.add_argument("--epochs", type=int, default=2, help="Kept small on purpose -- this stage only adapts BERT to clinical text, it is not the final model.")
    ap.add_argument("--max_landmark_minutes", type=float, default=cfg.MAX_LANDMARK_MINUTES_DEFAULT)
    ap.add_argument("--lr_bert", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_acc", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--max_notes", type=int, default=100)
    ap.add_argument("--max_windows_per_sample", type=int, default=200)
    ap.add_argument("--note_chunk_size", type=int, default=32)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    return ap.parse_args()


def load_split(event_dir, split):
    static_df = pd.read_parquet(os.path.join(event_dir, f"{split}_static.parquet"))
    vitals_df = pd.read_parquet(os.path.join(event_dir, f"{split}_structured_events.parquet"))
    notes_df = pd.read_parquet(os.path.join(event_dir, f"{split}_notes_events.parquet"))
    return static_df, vitals_df, notes_df


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.event_dir, "outcome_cols.json")) as f:
        outcome_cols = json.load(f)
    with open(os.path.join(args.event_dir, "static_feature_cols.json")) as f:
        static_cols = json.load(f)

    num_tabular_features = len(static_cols) + len(cfg.STRUCTURED_CHANNELS)

    print(">> Loading event-stream tables...")
    train_static, train_vitals, train_notes = load_split(args.event_dir, "train")
    val_static, val_vitals, val_notes = load_split(args.event_dir, "val")

    train_index = EncounterEventIndex(train_static, train_vitals, train_notes)
    val_index = EncounterEventIndex(val_static, val_vitals, val_notes)

    train_ds = SnapshotAtLandmarkDataset(
        train_static, train_index, outcome_cols, static_cols, args.max_landmark_minutes,
        seed=args.seed, resample_each_epoch=True,
    )
    val_ds = SnapshotAtLandmarkDataset(
        val_static, val_index, outcome_cols, static_cols, args.max_landmark_minutes,
        seed=args.seed + 1, resample_each_epoch=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
    model = Stage1Model(args.bert_dir, num_tabular_features, len(outcome_cols), args.note_chunk_size).to(device)

    collate = functools.partial(
        collate_fn, tokenizer=tokenizer, max_length=args.max_length, limit_notes=args.max_notes,
        doc_stride=args.doc_stride, max_windows_per_sample=args.max_windows_per_sample,
    )
    # drop_last=True: BatchNorm1d in the tabular encoder (see Stage1Model)
    # requires batch_size > 1 in train mode -- without this, whenever
    # len(train_ds) % batch_size == 1, the final batch of every epoch has
    # exactly 1 sample and crashes several hours into training. Only the
    # train loader needs this; eval uses running statistics (model.eval()),
    # so val_loader is unaffected regardless of its last batch size.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         "lr": args.lr_bert, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],
         "lr": args.lr_bert, "weight_decay": 0.0},
        {"params": list(model.tabular_encoder.parameters()) + list(model.fusion_classifier.parameters()),
         "lr": args.lr_head, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = max(1, len(train_loader) // max(1, args.grad_acc)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)

    best_val_auprc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_ds.resample_landmarks()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"stage1 epoch {epoch}")
        for step, (tokenized, sample_idx, labels, B, tabular_batch, _) in enumerate(pbar, start=1):
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            sample_idx = sample_idx.to(device)
            labels = labels.to(device)
            tabular_batch = tabular_batch.to(device)

            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(tokenized["input_ids"], tokenized["attention_mask"], sample_idx, B, tabular_batch)
                loss = loss_fn(logits, labels)
                if args.grad_acc > 1:
                    loss = loss / args.grad_acc
            scaler.scale(loss).backward()
            if step % args.grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_auroc, val_auprc = eval_loop(model, val_loader, outcome_cols, device, fp16)
        print(f"[Stage1 epoch {epoch}] val macro AUROC={val_auroc:.4f} AUPRC={val_auprc:.4f}")

        if not np.isnan(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            encoder_dir = os.path.join(args.output_dir, "bert_encoder")
            os.makedirs(encoder_dir, exist_ok=True)
            # safe_serialization=False: the attention-pooling wrapper around
            # this encoder can leave some weight tensors non-contiguous
            # (observed in smoke testing), which safetensors refuses to
            # save. The legacy torch.save-based format has no such
            # restriction and transformers' from_pretrained loads it
            # transparently either way.
            model.text_encoder.encoder.save_pretrained(encoder_dir, safe_serialization=False)
            tokenizer.save_pretrained(encoder_dir)
            with open(os.path.join(args.output_dir, "stage1_summary.json"), "w") as f:
                json.dump({"best_epoch": epoch, "val_macro_auroc": val_auroc, "val_macro_auprc": val_auprc}, f, indent=2)
            print(f"[Stage1 epoch {epoch}] saved BERT encoder to {encoder_dir}")

    print(">> Stage 1 done. Use the saved bert_encoder/ dir as --bert_dir for precompute_note_embeddings.py")


if __name__ == "__main__":
    main()
