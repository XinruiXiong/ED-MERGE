#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse, pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def parse_args():
    p = argparse.ArgumentParser("Eval-only on CONCAT(train+val+test) as one big test set (strict ckpt-aligned)")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--thresholds_path", default="")

    p.add_argument("--train_pkl", required=True)
    p.add_argument("--val_pkl", required=True)
    p.add_argument("--test_pkl", required=True)

    p.add_argument("--umn_train_pkl_path", required=True)
    p.add_argument("--bert_dir", required=True)

    p.add_argument("--mode", choices=["multimodal", "text_only", "vitals_only", "profile_only"], default="multimodal")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_notes", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--note_chunk_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_fp16", action="store_true")

    p.add_argument("--dropout_p", type=float, default=0.1, help="only used for filling missing non-param indices")
    p.add_argument("--dedup_by_stay_id", action="store_true", help="optional: drop duplicated stay_id after concat")
    p.add_argument("--out_dir", default="")
    return p.parse_args()


# -------------------------
# Dataset / Collate
# -------------------------
class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, outcome_cols, tabular_cols, use_text=True):
        self.df = df.reset_index(drop=True)
        self.outcome_cols = list(outcome_cols)
        self.tabular_cols = list(tabular_cols) if tabular_cols is not None else []
        self.use_text = bool(use_text)

        for c in self.outcome_cols:
            if c not in self.df.columns:
                self.df[c] = 0
        self.df[self.outcome_cols] = (
            self.df[self.outcome_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
        )

        if self.tabular_cols:
            for c in self.tabular_cols:
                if c not in self.df.columns:
                    self.df[c] = 0.0
            self.df[self.tabular_cols] = (
                self.df[self.tabular_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

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

            processed = [str(t) for t in raw_notes if isinstance(t, str) and t.strip()]
            if processed:
                notes = processed

        y = row[self.outcome_cols].values.astype(np.float32)

        tabular = np.array([], dtype=np.float32)
        if self.tabular_cols:
            tabular = row[self.tabular_cols].values.astype(np.float32)

        return {"notes": notes, "labels": y, "tabular": tabular}


def collate_fn(batch, tokenizer, max_length=512, limit_notes=None):
    B = len(batch)
    labels = torch.stack([torch.tensor(x["labels"]) for x in batch], dim=0)

    if batch[0]["tabular"].shape[0] > 0:
        tabular_data = torch.stack([torch.tensor(x["tabular"]) for x in batch], dim=0)
    else:
        tabular_data = None

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
        flat_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    sample_idx = torch.tensor(sample_idx, dtype=torch.long)
    return tokenized, sample_idx, labels, B, tabular_data


# -------------------------
# Build modules to MATCH ckpt indices (with fills)
# -------------------------
def _max_index_from_prefix_any(state_dict, prefix):
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
    max_idx = _max_index_from_prefix_any(state_dict, prefix)
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
    def __init__(self, model_dir, state_dict, max_notes_per_fwd_pass=32, dropout_p=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        self.max_notes_per_fwd_pass = max(1, int(max_notes_per_fwd_pass))

        self.attention_project = _build_seq_with_fills(state_dict, "text_encoder.attention_project.", dropout_p=dropout_p)
        if self.attention_project is None:
            h = int(self.encoder.config.hidden_size)
            self.attention_project = nn.Sequential(nn.Linear(h, h), nn.Tanh())
        else:
            if not any(isinstance(m, nn.Tanh) for m in self.attention_project):
                self.attention_project = nn.Sequential(self.attention_project, nn.Tanh())

        if "text_encoder.attention_context" not in state_dict:
            raise RuntimeError("ckpt missing text_encoder.attention_context")
        v = state_dict["text_encoder.attention_context"]
        self.attention_context = nn.Parameter(torch.empty_like(v))

    def forward(self, input_ids, attention_mask, sample_idx, B):
        n = input_ids.size(0)
        if n <= self.max_notes_per_fwd_pass:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
        else:
            chunks = []
            for i in range(0, n, self.max_notes_per_fwd_pass):
                out = self.encoder(
                    input_ids=input_ids[i:i+self.max_notes_per_fwd_pass],
                    attention_mask=attention_mask[i:i+self.max_notes_per_fwd_pass],
                )
                chunks.append(out.last_hidden_state[:, 0, :])
            cls = torch.cat(chunks, dim=0)

        pooled = torch.zeros((B, cls.size(-1)), dtype=cls.dtype, device=cls.device)
        for i in range(B):
            m = (sample_idx == i)
            if not m.any():
                continue
            sample_cls = cls[m]
            u = self.attention_project(sample_cls)
            scores = torch.matmul(u, self.attention_context)
            w = torch.softmax(scores, dim=0)
            pooled[i] = torch.sum(sample_cls * w.unsqueeze(1), dim=0)
        return pooled


class MultiModalModel(nn.Module):
    def __init__(self, bert_dir, state_dict, num_tabular_features, num_labels, max_notes_per_fwd_pass=32,
                 use_text=True, use_tabular=True, dropout_p=0.1):
        super().__init__()
        self.use_text = bool(use_text)
        self.use_tabular = bool(use_tabular)

        fusion_in = 0
        if self.use_text:
            self.text_encoder = MultiTaskBertAttentionPool(bert_dir, state_dict, max_notes_per_fwd_pass, dropout_p)
            fusion_in += int(self.text_encoder.encoder.config.hidden_size)

        if self.use_tabular:
            self.tabular_encoder = _build_seq_with_fills(state_dict, "tabular_encoder.", dropout_p=dropout_p)
            if self.tabular_encoder is None:
                self.tabular_encoder = nn.Identity()
                tab_out = int(num_tabular_features)
            else:
                tab_out = None
                mx = _max_index_from_prefix_any(state_dict, "tabular_encoder.")
                for i in range(mx, -1, -1):
                    w_key = f"tabular_encoder.{i}.weight"
                    if w_key in state_dict and state_dict[w_key].ndim == 2:
                        tab_out = int(state_dict[w_key].shape[0])
                        break
                tab_out = int(num_tabular_features) if tab_out is None else int(tab_out)
            fusion_in += tab_out

        self.fusion_classifier = _build_seq_with_fills(state_dict, "fusion_classifier.", dropout_p=dropout_p)
        if self.fusion_classifier is None:
            self.fusion_classifier = nn.Sequential(nn.Linear(fusion_in, num_labels))

    def forward(self, input_ids, attention_mask, sample_idx, B, tabular_input):
        vecs = []
        if self.use_text:
            vecs.append(self.text_encoder(input_ids, attention_mask, sample_idx, B))
        if self.use_tabular:
            tab_f = self.tabular_encoder(tabular_input) if tabular_input is not None else None
            if tab_f is None:
                tab_f = torch.zeros((B, 0), device=vecs[0].device if vecs else "cpu")
            vecs.append(tab_f)
        fused = torch.cat(vecs, dim=1) if len(vecs) > 1 else vecs[0]
        return self.fusion_classifier(fused)


# -------------------------
# Feature order from UMN train.pkl
# -------------------------
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


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def eval_loop(model, loader, outcome_cols, device, fp16):
    model.eval()
    all_logits, all_labels = [], []

    for it, (tokenized, sample_idx, labels, B, tabular_batch) in enumerate(loader, start=1):
        if it == 1 or it % 50 == 0:
            print(f"[eval] step={it} B={B} n_notes={tokenized['input_ids'].size(0)}", flush=True)

        tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
        sample_idx = sample_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if tabular_batch is not None:
            tabular_batch = tabular_batch.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(tokenized["input_ids"], tokenized["attention_mask"], sample_idx, B, tabular_batch)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # logits = torch.cat(all_logits, dim=0).numpy()
    # labels = torch.cat(all_labels, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).float().cpu().numpy().astype(np.float64)
    labels = torch.cat(all_labels, dim=0).float().cpu().numpy().astype(np.float64)

    per_task = {}
    for j, col in enumerate(outcome_cols):
        y_true = labels[:, j]
        y_score = 1.0 / (1.0 + np.exp(-logits[:, j]))
        if len(np.unique(y_true)) < 2:
            auroc, auprc = float("nan"), float("nan")
        else:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        per_task[col] = {"AUROC": auroc, "AUPRC": auprc}

    valid_auroc = [m["AUROC"] for m in per_task.values() if not np.isnan(m["AUROC"])]
    valid_auprc = [m["AUPRC"] for m in per_task.values() if not np.isnan(m["AUPRC"])]
    macro = {
        "AUROC": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
        "AUPRC": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
    }
    return {"macro": macro, "per_task": per_task}, logits, labels


def compute_f1s(logits, labels, thresholds, outcome_cols):
    logits = logits.astype(np.float64)
    labels = labels.astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    f1s = {}
    for j, col in enumerate(outcome_cols):
        y_true = labels[:, j].astype(int)
        thr = float(thresholds.get(col, 0.5)) if thresholds else 0.5
        y_pred = (probs[:, j] >= thr).astype(int)
        f1s[col] = f1_score(y_true, y_pred, zero_division=0)
    f1s["macro"] = float(np.mean(list(f1s.values()))) if f1s else float("nan")
    return f1s


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and (device == "cuda")

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "model" in ckpt and "outcome_cols" in ckpt):
        print("[FATAL] ckpt must be dict with keys: model, outcome_cols")
        sys.exit(1)
    state_dict = ckpt["model"]
    outcome_cols = list(ckpt["outcome_cols"])

    thresholds = None
    if args.thresholds_path and os.path.exists(args.thresholds_path):
        with open(args.thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    with open(args.umn_train_pkl_path, "rb") as f:
        umn_train_df = pickle.load(f)
    train_outcomes = [c for c in umn_train_df.columns if isinstance(c, str) and c.startswith("outcome_")]
    all_features = mode_filter_features(feature_order_from_train_pkl(umn_train_df, train_outcomes), args.mode)

    expected_dim = None
    for k, v in state_dict.items():
        if k.startswith("tabular_encoder.") and k.endswith(".weight") and v.ndim == 2:
            expected_dim = int(v.shape[1])
            break

    tab_cols = [] if args.mode == "text_only" else list(all_features)
    if expected_dim is not None:
        if len(tab_cols) > expected_dim:
            tab_cols = tab_cols[:expected_dim]
        elif len(tab_cols) < expected_dim:
            tab_cols = tab_cols + [f"_pad_{i}" for i in range(expected_dim - len(tab_cols))]

    # concat all splits as one big test set
    df_train = pd.read_pickle(args.train_pkl)
    df_val = pd.read_pickle(args.val_pkl)
    df_test = pd.read_pickle(args.test_pkl)
    df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    if args.dedup_by_stay_id and "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].astype(str)
        df = df.drop_duplicates(subset=["stay_id"]).reset_index(drop=True)

    # ensure outcomes exist
    for c in outcome_cols:
        if c not in df.columns:
            df[c] = 0

    if tab_cols:
        df = ensure_numeric_cols(df, tab_cols)

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
        dropout_p=args.dropout_p,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    print(f"[load] strict OK | mode={args.mode} | tab_dim={len(tab_cols)} | n_outcomes={len(outcome_cols)}", flush=True)
    print(f"[data] concat_rows={len(df):,}  (dedup={args.dedup_by_stay_id})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
    ds = MultiModalDataset(df, outcome_cols, tab_cols, use_text=USE_TEXT)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, args.max_notes),
    )

    metrics, logits, labels = eval_loop(model, loader, outcome_cols, device, fp16)
    f1s = compute_f1s(logits, labels, thresholds or {}, outcome_cols)

    print("=" * 60)
    print(f"[EVAL ON CONCAT(TRAIN+VAL+TEST) ({args.mode})]")
    print(f"Macro AUROC: {metrics['macro']['AUROC']}")
    print(f"Macro AUPRC: {metrics['macro']['AUPRC']}")
    print(f"Macro F1:    {f1s['macro']}")
    print("=" * 60)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = {
            "mode": args.mode,
            "tab_dim": len(tab_cols),
            "n_outcomes": len(outcome_cols),
            "macro": {**metrics["macro"], "F1": f1s["macro"]},
            "per_task": {c: {**metrics["per_task"][c], "F1": f1s.get(c, float("nan"))} for c in outcome_cols},
        }
        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        np.save(os.path.join(args.out_dir, "logits.npy"), logits)
        np.save(os.path.join(args.out_dir, "labels.npy"), labels)
        print(f"[save] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()