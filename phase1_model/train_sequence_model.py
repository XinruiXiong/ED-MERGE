#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 core training script: ONE EventSequenceModel trained across randomly
sampled landmark times, so one checkpoint serves every observation time.

Threshold selection: per-outcome F1-maximizing thresholds are computed on the
VALIDATION set only, never on the test set.

Multitask loss: FocalLoss(reduction="mean") averages over every
element of the (batch, num_labels) tensor -- i.e. an unweighted mean over
both samples and tasks. No missing-label handling is needed because every
outcome column is fully labeled 0/1 for every encounter (not sparse/missing
per task).
"""

import os
import sys
import json
import random
import argparse
import functools

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg
from common.focal_loss import FocalLoss
from common.event_utils import EncounterEventIndex, NoteEmbeddingIndex
from sequence_dataset import EventSequenceDataset, collate_sequence_batch
from sequence_model import EventSequenceModel


def load_split_tables(event_dir, embed_cache_dir, split):
    static_df = pd.read_parquet(os.path.join(event_dir, f"{split}_static.parquet"))
    vitals_df = pd.read_parquet(os.path.join(event_dir, f"{split}_structured_events.parquet"))
    vitals_index = EncounterEventIndex(static_df, vitals_df, pd.DataFrame(columns=["stay_id", "t_minutes", "note_text"]))
    note_embed_index = NoteEmbeddingIndex.from_cache_dir(embed_cache_dir, split)
    return static_df, vitals_index, note_embed_index


def eval_loop(model, loader, outcome_cols, device, fp16, return_meta=False):
    model.eval()
    all_logits, all_labels = [], []
    all_stay_ids, all_n_vitals, all_n_notes = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            content = batch["content"].to(device)
            modality = batch["modality"].to(device)
            delta_t = batch["delta_t"].to(device)
            lengths = batch["lengths"]
            static = batch["static"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(content, modality, delta_t, lengths, static)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())
            if return_meta:
                all_stay_ids.extend(batch["stay_ids"])
                all_n_vitals.append(batch["n_vitals_seen"])
                all_n_notes.append(batch["n_notes_seen"])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = {}
    for j, col in enumerate(outcome_cols):
        y_true = labels[:, j]
        if len(np.unique(y_true)) < 2:
            metrics[col] = {"AUROC": float("nan"), "AUPRC": float("nan")}
            continue
        y_score = 1 / (1 + np.exp(-logits[:, j]))
        metrics[col] = {
            "AUROC": roc_auc_score(y_true, y_score),
            "AUPRC": average_precision_score(y_true, y_score),
        }
    valid_auroc = [m["AUROC"] for m in metrics.values() if not np.isnan(m["AUROC"])]
    valid_auprc = [m["AUPRC"] for m in metrics.values() if not np.isnan(m["AUPRC"])]
    metrics["macro"] = {
        "AUROC": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
        "AUPRC": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
    }
    if return_meta:
        meta = {
            "stay_ids": all_stay_ids,
            "n_vitals_seen": torch.cat(all_n_vitals).numpy(),
            "n_notes_seen": torch.cat(all_n_notes).numpy(),
        }
        return metrics, logits, labels, meta
    return metrics, logits, labels


def pick_thresholds_from_val(val_logits, val_labels, outcome_cols):
    thresholds = {}
    probs = 1 / (1 + np.exp(-val_logits))
    for j, col in enumerate(outcome_cols):
        y_true = val_labels[:, j]
        if len(np.unique(y_true)) < 2:
            thresholds[col] = 0.5
            continue
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            y_pred = (probs[:, j] >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds[col] = float(best_thr)
    return thresholds


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 1: train the single sequence model across landmark times.")
    ap.add_argument("--event_dir", type=str, default=cfg.EVENT_STREAM_OUTPUT_DIR_PLACEHOLDER, required=True)
    ap.add_argument("--embed_cache_dir", type=str, default=cfg.NOTE_EMBEDDING_CACHE_DIR_PLACEHOLDER, required=True)
    ap.add_argument("--output_dir", type=str, default=cfg.SEQUENCE_MODEL_OUTPUT_DIR_PLACEHOLDER, required=True)

    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_landmark_minutes", type=float, default=cfg.MAX_LANDMARK_MINUTES_DEFAULT)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=32,
                     help="Can be large -- there is no BERT forward pass "
                          "in this training loop, only cached embeddings + a small GRU.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    return ap.parse_args()


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

    print(">> Loading train/val event-stream + cached note embeddings...")
    train_static, train_vitals_idx, train_note_idx = load_split_tables(args.event_dir, args.embed_cache_dir, "train")
    val_static, val_vitals_idx, val_note_idx = load_split_tables(args.event_dir, args.embed_cache_dir, "val")

    train_ds = EventSequenceDataset(
        train_static, train_vitals_idx, train_note_idx, outcome_cols, static_cols,
        args.max_landmark_minutes, seed=args.seed, resample_each_epoch=True,
    )
    val_ds = EventSequenceDataset(
        val_static, val_vitals_idx, val_note_idx, outcome_cols, static_cols,
        args.max_landmark_minutes, seed=args.seed + 1, resample_each_epoch=False,
    )

    # drop_last=True: same BatchNorm1d-needs-batch_size>1-in-train-mode issue
    # as stage1_finetune_bert.py -- crashed there several hours in when
    # len(train) % batch_size == 1. Only the train loader needs this (eval
    # uses running stats).
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_sequence_batch, num_workers=args.num_workers, pin_memory=True,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_sequence_batch, num_workers=args.num_workers, pin_memory=True)

    note_hidden = train_note_idx.hidden_size or 768
    model = EventSequenceModel(
        num_static_features=len(static_cols), num_labels=len(outcome_cols), note_hidden=note_hidden
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)

    best_val_auprc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_ds.resample_landmarks()
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for batch in pbar:
            content = batch["content"].to(device)
            modality = batch["modality"].to(device)
            delta_t = batch["delta_t"].to(device)
            lengths = batch["lengths"]
            static = batch["static"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(content, modality, delta_t, lengths, static)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_metrics, val_logits, val_labels = eval_loop(model, val_loader, outcome_cols, device, fp16)
        val_auroc = val_metrics["macro"]["AUROC"]
        val_auprc = val_metrics["macro"]["AUPRC"]
        print(f"[Epoch {epoch}] val macro AUROC={val_auroc:.4f} AUPRC={val_auprc:.4f}")

        if not np.isnan(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            thresholds = pick_thresholds_from_val(val_logits, val_labels, outcome_cols)
            best_state = {
                "model": model.state_dict(),
                "thresholds": thresholds,
                "outcome_cols": outcome_cols,
                "static_cols": static_cols,
                "note_hidden": note_hidden,
                "max_landmark_minutes": args.max_landmark_minutes,
                "val_macro_auroc": val_auroc,
                "val_macro_auprc": val_auprc,
                "epoch": epoch,
            }
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))
            with open(os.path.join(args.output_dir, "best_thresholds.json"), "w") as f:
                json.dump(thresholds, f, indent=2)
            print(f"[Epoch {epoch}] saved best_model.pt (val macro AUPRC={val_auprc:.4f})")

    print(">> Training done. Use phase1_model/evaluate_landmarks.py against best_model.pt "
          "to reproduce the Table-2-style landmark grid FROM ONE CHECKPOINT.")


if __name__ == "__main__":
    main()
