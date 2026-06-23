#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# =========================
# Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Finetune on external (MIMIC) then eval (strict ckpt-aligned)")

    p.add_argument("--ckpt_path", required=True, help="best_model.pt (must contain keys like text_encoder.encoder.*)")
    p.add_argument("--thresholds_path", default="", help="best_thresholds.json (optional for F1)")
    p.add_argument("--bert_dir", required=True, help="BERT model dir (BioClinicalBERT)")

    p.add_argument("--train_pkl", required=True)
    p.add_argument("--val_pkl", default="")
    p.add_argument("--test_pkl", required=True)

    p.add_argument("--umn_train_pkl_path", required=True, help="UMN train.pkl to recover strict feature order")
    p.add_argument("--mode", choices=["multimodal", "text_only", "vitals_only", "profile_only"], default="multimodal")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_notes", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--note_chunk_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_fp16", action="store_true")

    # finetune
    p.add_argument("--do_finetune", action="store_true")
    p.add_argument("--finetune_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--freeze_text_encoder", action="store_true")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # outputs
    p.add_argument("--out_dir", default="", help="save metrics + logits/labels")
    p.add_argument("--save_finetuned_ckpt", default="", help="save finetuned ckpt (.pt)")

    return p.parse_args()


# =========================
# Dataset & Collate 
# =========================
class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, outcome_cols, tabular_cols, use_text=True):
        self.df = df.reset_index(drop=True)
        self.outcome_cols = list(outcome_cols)
        self.tabular_cols = list(tabular_cols) if tabular_cols is not None else []
        self.use_text = bool(use_text)

        # ensure outcome cols exist
        for c in self.outcome_cols:
            if c not in self.df.columns:
                self.df[c] = 0

        # ensure numeric
        self.df[self.outcome_cols] = self.df[self.outcome_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

        if self.tabular_cols:
            for c in self.tabular_cols:
                if c not in self.df.columns:
                    self.df[c] = 0.0
            self.df[self.tabular_cols] = self.df[self.tabular_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # notes: expect NOTE_TEXT or note_text (list[str])
        notes = [""]
        if self.use_text:
            raw_notes = None
            if "NOTE_TEXT" in row.index:
                raw_notes = row["NOTE_TEXT"]
            elif "note_text" in row.index:
                raw_notes = row["note_text"]
            else:
                raw_notes = [""]

            if not isinstance(raw_notes, list):
                raw_notes = [""]

            processed = [str(t) for t in raw_notes if isinstance(t, str) and t.strip() != ""]
            if processed:
                notes = processed

        y = row[self.outcome_cols].values.astype(np.float32)

        tabular = np.array([], dtype=np.float32)
        if len(self.tabular_cols) > 0:
            tabular = row[self.tabular_cols].values.astype(np.float32)

        stay_id = str(row["stay_id"]) if "stay_id" in row.index else str(idx)

        return {"notes": notes, "labels": y, "tabular": tabular, "stay_id": stay_id}


def collate_fn(batch, tokenizer, max_length=512, limit_notes=None):
    B = len(batch)
    labels = torch.stack([torch.tensor(x["labels"]) for x in batch], dim=0)
    stay_ids = [x["stay_id"] for x in batch]

    # tabular
    if batch[0]["tabular"].shape[0] > 0:
        tabular_data = torch.stack([torch.tensor(x["tabular"]) for x in batch], dim=0)
    else:
        tabular_data = None

    # flatten notes
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
        flat_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    sample_idx = torch.tensor(sample_idx, dtype=torch.long)

    return tokenized, sample_idx, labels, B, tabular_data, stay_ids


# =========================
# Strict model that matches your checkpoint keys:
# text_encoder.encoder.*  + text_encoder.attention_project.* + text_encoder.attention_context
# =========================
def _max_index_from_prefix(state_dict, prefix):
    mx = -1
    for k in state_dict.keys():
        if not k.startswith(prefix):
            continue
        parts = k.split(".")
        if len(parts) < 3:
            continue
        try:
            i = int(parts[1])
        except Exception:
            continue
        mx = max(mx, i)
    return mx


def _build_seq_with_fills(state_dict, prefix, dropout_p=0.1):
    """
    Reconstruct an nn.Sequential from checkpoint keys matching prefix.

    Parameterized positions are rebuilt from the weight shape:
      - 2D weight -> Linear
      - 1D weight -> BatchNorm1d

    Non-parameterized positions (ReLU, Dropout) have no checkpoint keys and
    are filled in order: first gap -> ReLU, second gap -> Dropout(dropout_p),
    subsequent gaps -> Identity. This matches the training architecture of both
    tabular_encoder and fusion_classifier exactly.

    Returns None when no matching keys are found (triggering the caller's fallback).
    """
    max_idx = _max_index_from_prefix(state_dict, prefix)
    if max_idx < 0:
        return None

    mods = []
    last_fill = None
    for i in range(max_idx + 1):
        w_key = f"{prefix}{i}.weight"
        if w_key in state_dict:
            w = state_dict[w_key]
            if w.ndim == 2:
                out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
                mods.append(nn.Linear(in_dim, out_dim))
            elif w.ndim == 1:
                mods.append(nn.BatchNorm1d(int(w.shape[0])))
            else:
                mods.append(nn.Identity())
            last_fill = None
        else:
            if last_fill is None:
                mods.append(nn.ReLU())
                last_fill = "relu"
            elif last_fill == "relu":
                mods.append(nn.Dropout(p=float(dropout_p)))
                last_fill = "drop"
            else:
                mods.append(nn.Identity())
                last_fill = None
    return nn.Sequential(*mods)


class MultiTaskBertAttentionPool(nn.Module):
    def __init__(self, model_dir, state_dict, max_notes_per_fwd_pass=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        hidden = int(self.encoder.config.hidden_size)
        self.max_notes_per_fwd_pass = max(1, int(max_notes_per_fwd_pass))

        # attention_project: _build_seq_with_fills returns None here because
        # _max_index_from_prefix extracts parts[1] which is "attention_project"
        # (not an integer) for keys like "text_encoder.attention_project.0.weight".
        # The fallback builds the correct Sequential(Linear, Tanh) directly.
        self.attention_project = _build_seq_with_fills(state_dict, "text_encoder.attention_project.")
        if self.attention_project is None:
            self.attention_project = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())

        # attention_context param
        if "text_encoder.attention_context" in state_dict:
            v = state_dict["text_encoder.attention_context"]
            self.attention_context = nn.Parameter(torch.empty_like(v))
        else:
            self.attention_context = nn.Parameter(torch.randn(hidden) * 0.02)

    def forward(self, input_ids, attention_mask, sample_idx, B):
        num_notes_total = input_ids.size(0)

        # chunk forward to avoid OOM (match your old code)
        if num_notes_total <= self.max_notes_per_fwd_pass:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
        else:
            all_cls = []
            for i in range(0, num_notes_total, self.max_notes_per_fwd_pass):
                out = self.encoder(
                    input_ids=input_ids[i:i + self.max_notes_per_fwd_pass],
                    attention_mask=attention_mask[i:i + self.max_notes_per_fwd_pass],
                )
                all_cls.append(out.last_hidden_state[:, 0, :])
            cls = torch.cat(all_cls, dim=0)

        H = cls.size(-1)
        device = cls.device
        pooled = torch.zeros((B, H), dtype=cls.dtype, device=device)

        for i in range(B):
            mask = (sample_idx == i)
            if not mask.any():
                continue
            sample_cls = cls[mask]
            u = self.attention_project(sample_cls)
            scores = torch.matmul(u, self.attention_context)
            weights = torch.softmax(scores, dim=0)
            pooled[i] = torch.sum(sample_cls * weights.unsqueeze(1), dim=0)

        return pooled


class MultiModalModel(nn.Module):
    def __init__(self, bert_dir, state_dict, num_tabular_features, num_labels, max_notes_per_fwd_pass=32, use_text=True, use_tabular=True):
        super().__init__()
        self.use_text = bool(use_text)
        self.use_tabular = bool(use_tabular)

        fusion_in = 0

        if self.use_text:
            self.text_encoder = MultiTaskBertAttentionPool(bert_dir, state_dict, max_notes_per_fwd_pass=max_notes_per_fwd_pass)
            text_hidden = int(self.text_encoder.encoder.config.hidden_size)
            fusion_in += text_hidden

        if self.use_tabular:
            self.tabular_encoder = _build_seq_with_fills(state_dict, "tabular_encoder.", dropout_p=0.2)
            if self.tabular_encoder is None:
                self.tabular_encoder = nn.Identity()
                tab_out = int(num_tabular_features)
            else:
                tab_out = None
                max_idx = _max_index_from_prefix(state_dict, "tabular_encoder.")
                for i in range(max_idx, -1, -1):
                    w_key = f"tabular_encoder.{i}.weight"
                    if w_key in state_dict and state_dict[w_key].ndim == 2:
                        tab_out = int(state_dict[w_key].shape[0])
                        break
                tab_out = int(num_tabular_features) if tab_out is None else int(tab_out)
            fusion_in += tab_out

        self.fusion_classifier = _build_seq_with_fills(state_dict, "fusion_classifier.", dropout_p=0.1)
        if self.fusion_classifier is None:
            self.fusion_classifier = nn.Sequential(nn.Linear(fusion_in, num_labels))

    def forward(self, input_ids, attention_mask, sample_idx, B, tabular_input, return_embeds=False):
        vecs = []
        text_f = None
        tab_f = None

        if self.use_text:
            text_f = self.text_encoder(input_ids, attention_mask, sample_idx, B)
            vecs.append(text_f)

        if self.use_tabular:
            tab_f = self.tabular_encoder(tabular_input) if tabular_input is not None else None
            if tab_f is None:
                tab_f = torch.zeros((B, 0), device=text_f.device if text_f is not None else "cpu")
            vecs.append(tab_f)

        fused = torch.cat(vecs, dim=1) if len(vecs) > 1 else vecs[0]
        logits = self.fusion_classifier(fused)

        if return_embeds:
            return logits, text_f, tab_f, fused
        return logits


# =========================
# Feature order from UMN training data
# =========================
def feature_order_from_train_pkl(train_df: pd.DataFrame, train_outcome_cols):
    base_like = {"stay_id", "subject_id", "is_repeat", "NOTE_TEXT", "note_text"}
    outcome_set = set(train_outcome_cols)

    feats = []
    for c in list(train_df.columns):
        if c in base_like:
            continue
        if c in outcome_set:
            continue
        if isinstance(c, str) and c.startswith("outcome_"):
            continue
        if isinstance(c, str) and c.startswith("Unnamed"):
            continue
        feats.append(c)
    return feats


def mode_filter_features(features, mode):
    feats = list(features)
    if mode == "multimodal":
        return feats
    if mode == "text_only":
        return []
    if mode == "vitals_only":
        return [c for c in feats if isinstance(c, str) and c.startswith("final_")]
    if mode == "profile_only":
        return [c for c in feats if isinstance(c, str) and (c == "age" or c.startswith(("cci_", "eci_", "chiefcom_")))]
    return feats


def ensure_numeric_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    if cols:
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    return df


# =========================
# Eval
# =========================
@torch.no_grad()
def eval_loop(model, dloader, outcome_cols, device, fp16, split="val"):
    model.eval()
    all_logits, all_labels = [], []
    for tokenized, sample_idx, labels, B, tabular_batch, stay_ids in dloader:
        tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
        sample_idx = sample_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if tabular_batch is not None:
            tabular_batch = tabular_batch.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                sample_idx=sample_idx,
                B=B,
                tabular_input=tabular_batch,
                return_embeds=False,
            )

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    metrics = {}
    for j, col in enumerate(outcome_cols):
        y_true = all_labels[:, j]
        y_score = 1.0 / (1.0 + np.exp(-all_logits[:, j]))
        if len(np.unique(y_true)) < 2:
            auc, aupr = float("nan"), float("nan")
        else:
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = float("nan")
            try:
                aupr = average_precision_score(y_true, y_score)
            except Exception:
                aupr = float("nan")
        metrics[col] = {"AUROC": auc, "AUPRC": aupr}

    valid_auroc = [m["AUROC"] for m in metrics.values() if not np.isnan(m["AUROC"])]
    valid_auprc = [m["AUPRC"] for m in metrics.values() if not np.isnan(m["AUPRC"])]
    metrics["macro"] = {
        "AUROC": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
        "AUPRC": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
    }
    return metrics, all_logits, all_labels


def compute_f1s(logits, labels, thresholds, outcome_cols):
    probs = 1.0 / (1.0 + np.exp(-logits))
    f1s = {}
    for j, col in enumerate(outcome_cols):
        y_true = labels[:, j].astype(int)
        t = float(thresholds.get(col, 0.5)) if thresholds else 0.5
        y_pred = (probs[:, j] >= t).astype(int)
        f1s[col] = f1_score(y_true, y_pred, zero_division=0)
    f1s["macro"] = float(np.mean(list(f1s.values()))) if f1s else float("nan")
    return f1s


# =========================
# Finetune
# =========================
def finetune_loop(model, train_loader, val_loader, outcome_cols, device, fp16, args, thresholds):
    if args.freeze_text_encoder:
        for p in model.text_encoder.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="mean")
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    best_val = None
    best_state = None

    for ep in range(1, args.finetune_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        steps = 0

        for it, (tokenized, sample_idx, labels, B, tabular_batch, stay_ids) in enumerate(train_loader, start=1):
            tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
            sample_idx = sample_idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if tabular_batch is not None:
                tabular_batch = tabular_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    sample_idx=sample_idx,
                    B=B,
                    tabular_input=tabular_batch,
                    return_embeds=False,
                )
                loss = loss_fn(logits, labels) / max(1, args.grad_accum)

            scaler.scale(loss).backward()
            total += float(loss.item())
            steps += 1

            if it % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        # Apply remaining accumulated gradients if the last mini-batch did not
        # land on a gradient accumulation boundary.
        if len(train_loader) % args.grad_accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        train_loss = total / max(1, steps)

        if val_loader is not None:
            val_metrics, _, _ = eval_loop(model, val_loader, outcome_cols, device, fp16, split="val")
            val_score = val_metrics["macro"]["AUPRC"]
            if best_val is None or (not np.isnan(val_score) and (np.isnan(best_val) or val_score > best_val)):
                best_val = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[finetune] epoch={ep} train_loss={train_loss:.6f} val_macro_AUPRC={val_score}")
        else:
            print(f"[finetune] epoch={ep} train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)


# =========================
# Main
# =========================
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and (device == "cuda")

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "model" in ckpt and "outcome_cols" in ckpt):
        print("[FATAL] ckpt must be a dict with keys: model, outcome_cols")
        sys.exit(1)

    state_dict = ckpt["model"]
    outcome_cols = list(ckpt["outcome_cols"])

    thresholds = None
    if args.thresholds_path and os.path.exists(args.thresholds_path):
        with open(args.thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    # recover strict tabular order from UMN train.pkl
    with open(args.umn_train_pkl_path, "rb") as f:
        umn_train_df = pickle.load(f)
    if not isinstance(umn_train_df, pd.DataFrame):
        print("[FATAL] umn_train_pkl is not a DataFrame")
        sys.exit(1)

    train_outcomes = [c for c in umn_train_df.columns if isinstance(c, str) and c.startswith("outcome_")]
    all_features = feature_order_from_train_pkl(umn_train_df, train_outcomes)
    all_features = mode_filter_features(all_features, args.mode)

    # expected tabular dim from ckpt (first Linear in tabular_encoder.*)
    expected_dim = None
    for k, v in state_dict.items():
        if k.startswith("tabular_encoder.") and k.endswith(".weight") and v.ndim == 2:
            expected_dim = int(v.shape[1])
            break

    tab_cols = list(all_features)
    if args.mode == "text_only":
        tab_cols = []

    # truncate/pad to expected_dim (match your old strict behavior)
    if expected_dim is not None:
        if len(tab_cols) > expected_dim:
            tab_cols = tab_cols[:expected_dim]
        elif len(tab_cols) < expected_dim:
            tab_cols = tab_cols + [f"_pad_{i}" for i in range(expected_dim - len(tab_cols))]

    # load data
    train_df = pd.read_pickle(args.train_pkl)
    val_df = pd.read_pickle(args.val_pkl) if args.val_pkl else None
    test_df = pd.read_pickle(args.test_pkl)

    # ensure outcomes exist (missing -> 0)
    for c in outcome_cols:
        if c not in train_df.columns:
            train_df[c] = 0
        if val_df is not None and c not in val_df.columns:
            val_df[c] = 0
        if c not in test_df.columns:
            test_df[c] = 0

    # ensure tab cols exist + numeric
    if tab_cols:
        train_df = ensure_numeric_cols(train_df, tab_cols)
        test_df = ensure_numeric_cols(test_df, tab_cols)
        if val_df is not None:
            val_df = ensure_numeric_cols(val_df, tab_cols)

    USE_TEXT = args.mode in ["multimodal", "text_only"]
    USE_TAB = args.mode in ["multimodal", "vitals_only", "profile_only"]

    model = MultiModalModel(
        bert_dir=args.bert_dir,
        state_dict=state_dict,
        num_tabular_features=len(tab_cols),
        num_labels=len(outcome_cols),
        max_notes_per_fwd_pass=args.note_chunk_size,
        use_text=USE_TEXT,
        use_tabular=USE_TAB,
    ).to(device)

    # strict load MUST work now
    model.load_state_dict(state_dict, strict=True)
    print(f"[load] strict OK | mode={args.mode} | tab_dim={len(tab_cols)} | n_outcomes={len(outcome_cols)}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)

    train_ds = MultiModalDataset(train_df, outcome_cols, tab_cols, use_text=USE_TEXT)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, args.max_notes),
    )

    val_loader = None
    if val_df is not None and len(val_df) > 0:
        val_ds = MultiModalDataset(val_df, outcome_cols, tab_cols, use_text=USE_TEXT)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, args.max_notes),
        )

    test_ds = MultiModalDataset(test_df, outcome_cols, tab_cols, use_text=USE_TEXT)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, args.max_notes),
    )

    if args.do_finetune:
        finetune_loop(model, train_loader, val_loader, outcome_cols, device, fp16, args, thresholds)

    test_metrics, test_logits, test_labels = eval_loop(model, test_loader, outcome_cols, device, fp16, split="test")
    f1s = compute_f1s(test_logits, test_labels, thresholds or {}, outcome_cols)

    print("=" * 60)
    print(f"[FINAL TEST RESULTS ({args.mode})]")
    print(f"Macro AUROC: {test_metrics['macro']['AUROC']}")
    print(f"Macro AUPRC: {test_metrics['macro']['AUPRC']}")
    print(f"Macro F1:    {f1s['macro']}")
    print("=" * 60)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = {
            "mode": args.mode,
            "macro": {
                "AUROC": test_metrics["macro"]["AUROC"],
                "AUPRC": test_metrics["macro"]["AUPRC"],
                "F1": f1s["macro"],
            },
            "per_task": {},
        }
        for c in outcome_cols:
            out["per_task"][c] = {
                "AUROC": test_metrics.get(c, {}).get("AUROC", float("nan")),
                "AUPRC": test_metrics.get(c, {}).get("AUPRC", float("nan")),
                "F1": f1s.get(c, float("nan")),
            }

        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        np.save(os.path.join(args.out_dir, "test_logits.npy"), test_logits)
        np.save(os.path.join(args.out_dir, "test_labels.npy"), test_labels)

    if args.save_finetuned_ckpt:
        save_obj = {
            "model": model.state_dict(),
            "outcome_cols": outcome_cols,
            "tabular_cols": tab_cols,
            "args": vars(args),
        }
        torch.save(save_obj, args.save_finetuned_ckpt)
        print(f"[save] {args.save_finetuned_ckpt}")


if __name__ == "__main__":
    main()