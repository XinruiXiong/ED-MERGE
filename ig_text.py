#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IG visualization (TEXT-ONLY attribution) for YOUR trained MultiModalModel checkpoint.

Key points:
- NO retraining. Loads best_model.pt -> model.state_dict() and runs model.eval().
- Tabular is FIXED CONSTANT (default zeros). We do NOT attribute tabular, only text embeddings.
- Tokenization is aligned with your collate_fn: truncation=True, max_length=..., padding=True.

Outputs:
- sentence_ig.csv
- sentence_ig_topk.png
- ig_highlight_note{i}.html for each note in NOTE_TEXT list
- run_summary.json
"""

import os
import re
import json
import argparse
import pickle
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from captum.attr import IntegratedGradients

import matplotlib.pyplot as plt


# =========================
# Text helpers
# =========================
def simple_sentence_split(text: str) -> List[str]:
    if text is None:
        return []
    t = str(text).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    blocks = [b.strip() for b in t.split("\n") if b.strip()]
    sents: List[str] = []
    for b in blocks:
        parts = re.split(r"(?<=[\.\?\!\;])\s+", b)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            sents.extend(parts)
    return sents


def find_sentence_spans(text: str, sents: List[str]) -> List[Optional[Tuple[int, int]]]:
    spans: List[Optional[Tuple[int, int]]] = []
    cursor = 0
    for s in sents:
        idx = text.find(s, cursor)
        if idx < 0:
            spans.append(None)
            continue
        start, end = idx, idx + len(s)
        spans.append((start, end))
        cursor = end
    return spans


def html_token_highlight(tokens: List[str], norm_vals: np.ndarray) -> str:
    from html import escape

    def rgba(v: float) -> str:
        a = min(1.0, abs(float(v)))
        if v >= 0:
            return f"rgba(255,0,0,{0.12 + 0.60*a})"
        return f"rgba(0,0,255,{0.12 + 0.60*a})"

    parts = []
    for tok, v in zip(tokens, norm_vals.tolist()):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        disp = tok.replace("##", "")
        parts.append(
            f"<span style='background:{rgba(v)}; padding:2px 3px; margin:1px; border-radius:3px;'>"
            f"{escape(disp)}</span>"
        )
    return "<div style='font-family: Arial; line-height: 2.2;'>" + " ".join(parts) + "</div>"


# =========================
# Model (matches your training architecture)
# =========================
class MultiTaskBertAttentionPool(nn.Module):
    def __init__(self, model_dir: str, max_notes_per_fwd_pass: int = 32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        hidden = self.encoder.config.hidden_size
        self.max_notes_per_fwd_pass = max(1, int(max_notes_per_fwd_pass))
        self.attention_project = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.attention_context = nn.Parameter(torch.randn(hidden))
        nn.init.normal_(self.attention_context, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        sample_idx: Optional[torch.Tensor] = None,
        B: Optional[int] = None,
    ) -> torch.Tensor:
        assert (input_ids is not None) or (inputs_embeds is not None)
        assert attention_mask is not None and sample_idx is not None and B is not None

        num_notes_total = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)

        if num_notes_total <= self.max_notes_per_fwd_pass:
            out = self.encoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            cls = out.last_hidden_state[:, 0, :]
        else:
            all_cls = []
            for i in range(0, num_notes_total, self.max_notes_per_fwd_pass):
                chunk_out = self.encoder(
                    input_ids=None if input_ids is None else input_ids[i:i + self.max_notes_per_fwd_pass],
                    inputs_embeds=None if inputs_embeds is None else inputs_embeds[i:i + self.max_notes_per_fwd_pass],
                    attention_mask=attention_mask[i:i + self.max_notes_per_fwd_pass],
                    return_dict=True,
                )
                all_cls.append(chunk_out.last_hidden_state[:, 0, :])
            cls = torch.cat(all_cls, dim=0)

        H = cls.size(-1)
        device = cls.device
        pooled_outputs = torch.zeros((B, H), dtype=cls.dtype, device=device)

        for i in range(B):
            mask = (sample_idx == i)
            if not mask.any():
                continue
            sample_cls = cls[mask]
            u = self.attention_project(sample_cls)
            scores = torch.matmul(u, self.attention_context)
            w = torch.softmax(scores, dim=0)
            pooled_outputs[i] = torch.sum(sample_cls * w.unsqueeze(1), dim=0)

        return pooled_outputs


class MultiModalModel(nn.Module):
    def __init__(self, model_dir: str, num_tabular_features: int, num_labels: int, max_notes_per_fwd_pass: int = 32):
        super().__init__()
        self.text_encoder = MultiTaskBertAttentionPool(model_dir, max_notes_per_fwd_pass=max_notes_per_fwd_pass)
        text_hidden = self.text_encoder.encoder.config.hidden_size

        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        tab_hidden = 128

        fusion_size = text_hidden + tab_hidden
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        sample_idx: Optional[torch.Tensor] = None,
        B: Optional[int] = None,
        tabular_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_feat = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            sample_idx=sample_idx,
            B=B,
        )
        tab_feat = self.tabular_encoder(tabular_input)
        fused = torch.cat([text_feat, tab_feat], dim=1)
        logits = self.fusion_classifier(fused)
        return logits


# =========================
# IG wrapper (TEXT attribution only; tabular is fixed constant)
# =========================
class IGForwardWrapper(nn.Module):
    def __init__(self, model: MultiModalModel, label_idx: int, tabular_const: torch.Tensor):
        super().__init__()
        self.model = model
        self.label_idx = int(label_idx)
        self.tabular_const = tabular_const  # [1, F] fixed

    def forward(self, inputs_embeds, attention_mask, sample_idx, B):
        logits = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            sample_idx=sample_idx,
            B=B,
            tabular_input=self.tabular_const,
        )
        return logits[:, self.label_idx]


# =========================
# IO helpers
# =========================
def load_split_pkls(preprocessed_dir: str):
    def _load(name):
        with open(os.path.join(preprocessed_dir, name), "rb") as f:
            return pickle.load(f)
    return _load("train.pkl"), _load("val.pkl"), _load("test.pkl")


def find_row_by_stay_id(df: pd.DataFrame, stay_id: str) -> pd.Series:
    if "stay_id" not in df.columns:
        raise ValueError("No 'stay_id' in dataframe.")
    sid = str(stay_id).strip()
    idx = df.index[df["stay_id"].astype(str).str.strip() == sid]
    if len(idx) == 0:
        raise ValueError(f"stay_id={sid} not found.")
    return df.loc[idx[0]]


def ensure_notes_list(x) -> List[str]:
    if isinstance(x, list):
        notes = x
    else:
        notes = [""]
    notes = [str(t) for t in notes if isinstance(t, str) and t.strip() != ""]
    if not notes:
        notes = [""]
    return notes


def build_mask_baseline_ids(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no [MASK]; use --baseline pad.")
    base = input_ids.clone()
    special = set(tokenizer.all_special_ids)
    pad_id = tokenizer.pad_token_id
    for i in range(base.size(0)):
        for j in range(base.size(1)):
            tid = int(base[i, j].item())
            if tid in special or (pad_id is not None and tid == pad_id):
                continue
            base[i, j] = mask_id
    return base


def build_pad_baseline_ids(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no PAD; use --baseline mask.")
    base = torch.full_like(input_ids, pad_id)
    if tokenizer.cls_token_id is not None:
        base[:, 0] = tokenizer.cls_token_id
    if tokenizer.sep_token_id is not None:
        base[:, -1] = tokenizer.sep_token_id
    return base


def tokenize_notes(
    tokenizer: AutoTokenizer,
    notes: List[str],
    max_length: int,
    device: str,
):
    # aligned with collate_fn: padding=True, truncation=True, max_length, return_tensors="pt"
    tok = tokenizer(
        notes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    sample_idx = torch.zeros((len(notes),), dtype=torch.long, device=device)
    B = 1

    # offsets/tokens (for sentence aggregation + html)
    offsets_all = []
    tokens_all = []
    for t in notes:
        enc = tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_all.append(enc["offset_mapping"][0].tolist())
        input_ids_one = enc["input_ids"][0].tolist()
        tokens_all.append(tokenizer.convert_ids_to_tokens(input_ids_one))

    return tok, sample_idx, B, offsets_all, tokens_all


def compute_sentence_rows(tokens, offsets, token_attr_1d, raw_text):
    sents = simple_sentence_split(raw_text)
    spans = find_sentence_spans(raw_text, sents)
    sent_scores = np.zeros(len(sents), dtype=float)
    sent_counts = np.zeros(len(sents), dtype=int)

    for ti, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0:
            continue
        if tokens[ti] in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        for si, sp in enumerate(spans):
            if sp is None:
                continue
            ss, se = sp
            if cs < se and ce > ss:
                sent_scores[si] += float(token_attr_1d[ti])
                sent_counts[si] += 1
                break

    sent_mean = np.array([
        sent_scores[i] / sent_counts[i] if sent_counts[i] > 0 else 0.0
        for i in range(len(sents))
    ])

    rows = []
    for si, sent in enumerate(sents):
        rows.append({
            "sent_idx": si,
            "sent_score_sum": float(sent_scores[si]),
            "sent_score_mean": float(sent_mean[si]),
            "sent_token_count": int(sent_counts[si]),
            "sentence": sent,
        })
    return rows


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser("IG text-only attribution for your checkpoint")

    ap.add_argument("--preprocessed_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="best_model.pt")
    ap.add_argument("--bert_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--stay_id", type=str, required=True)
    ap.add_argument("--label", type=str, required=True, help="outcome col name OR integer index")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--note_chunk_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--baseline", type=str, default="mask", choices=["mask", "pad"])
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--out_dir", type=str, default="./ig_out_textonly")

    # tabular constant (fixed, not attributed)
    ap.add_argument("--tabular_const", type=str, default="zeros", choices=["zeros"], help="fixed tabular input")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint (NO training)
    state = torch.load(args.ckpt, map_location=device)
    if "model" not in state:
        raise ValueError("ckpt missing key 'model' (expect best_model.pt from your training script).")

    outcome_cols = state.get("outcome_cols", None)
    tabular_cols = state.get("tabular_cols", None)
    if outcome_cols is None or tabular_cols is None:
        raise ValueError("ckpt missing outcome_cols/tabular_cols.")

    # Resolve label idx
    if re.fullmatch(r"\d+", str(args.label).strip()):
        label_idx = int(args.label)
        if not (0 <= label_idx < len(outcome_cols)):
            raise ValueError("label idx out of range.")
        label_name = outcome_cols[label_idx]
    else:
        label_name = str(args.label).strip()
        if label_name not in outcome_cols:
            raise ValueError(f"label '{label_name}' not found in outcome_cols.")
        label_idx = outcome_cols.index(label_name)

    # Load data row
    train_df, val_df, test_df = load_split_pkls(args.preprocessed_dir)
    df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    row = find_row_by_stay_id(df, args.stay_id)
    notes = ensure_notes_list(row.get("NOTE_TEXT", None))

    # Fixed tabular constant (zeros)
    F = len(tabular_cols)
    tab_const = torch.zeros((1, F), dtype=torch.float32, device=device)

    # Build tokenizer + model and load weights (NO training)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
    model = MultiModalModel(
        model_dir=args.bert_dir,
        num_tabular_features=F,
        num_labels=len(outcome_cols),
        max_notes_per_fwd_pass=args.note_chunk_size,
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    # Tokenize notes
    tok, sample_idx, B, offsets_all, tokens_all = tokenize_notes(
        tokenizer=tokenizer,
        notes=notes,
        max_length=args.max_length,
        device=device,
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    # Embeddings + baselines
    emb_layer = model.text_encoder.encoder.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids)

    if args.baseline == "mask":
        base_ids = build_mask_baseline_ids(tokenizer, input_ids)
    else:
        base_ids = build_pad_baseline_ids(tokenizer, input_ids)
    baseline_embeds = emb_layer(base_ids)

    # IG over text embeddings only
    wrapper = IGForwardWrapper(model, label_idx=label_idx, tabular_const=tab_const).to(device)
    ig = IntegratedGradients(wrapper)

    attributions = ig.attribute(
        inputs=inputs_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(attention_mask, sample_idx, B),
        n_steps=args.steps,
    )  # [N, L, H]
    token_attr = attributions.sum(dim=-1).detach().cpu().numpy()  # [N, L]

    # Prediction reference (tab fixed)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_idx=sample_idx,
            B=B,
            tabular_input=tab_const,
        )
        logit_val = float(logits[0, label_idx].item())
        prob_val = float(torch.sigmoid(logits[0, label_idx]).item())

    # Outputs
    all_rows = []
    for ni in range(len(notes)):
        offsets = offsets_all[ni]
        tokens = tokens_all[ni]
        vals = token_attr[ni].copy()

        # HTML normalize (ignore PAD)
        pad_id = tokenizer.pad_token_id
        input_ids_note = input_ids[ni].detach().cpu().tolist()
        for j, tid in enumerate(input_ids_note):
            if pad_id is not None and tid == pad_id:
                vals[j] = 0.0
        max_abs = float(np.max(np.abs(vals))) if np.max(np.abs(vals)) > 0 else 1.0
        norm_vals = vals / max_abs

        html = html_token_highlight(tokens, norm_vals)
        with open(os.path.join(args.out_dir, f"ig_highlight_note{ni}.html"), "w", encoding="utf-8") as f:
            f.write(html)

        sent_rows = compute_sentence_rows(tokens, offsets, vals, notes[ni])
        for r in sent_rows:
            r.update({
                "stay_id": str(args.stay_id),
                "split": args.split,
                "label_name": label_name,
                "label_idx": label_idx,
                "note_idx": ni,
                "logit": logit_val,
                "prob": prob_val,
            })
            all_rows.append(r)

    out_csv = os.path.join(args.out_dir, "sentence_ig.csv")
    df_out = pd.DataFrame(all_rows).sort_values("sent_score_sum", ascending=False)
    df_out.to_csv(out_csv, index=False)

    topk = df_out.head(args.top_k).copy().iloc[::-1]
    ylabels = []
    for _, rr in topk.iterrows():
        s = re.sub(r"\s+", " ", str(rr["sentence"])).strip()
        if len(s) > 90:
            s = s[:87] + "..."
        ylabels.append(f"note{int(rr['note_idx'])}-s{int(rr['sent_idx'])}: {s}")

    plt.figure(figsize=(12, max(4, 0.35 * len(topk))))
    plt.barh(range(len(topk)), topk["sent_score_sum"].values)
    plt.yticks(range(len(topk)), ylabels, fontsize=9)
    plt.xlabel("IG attribution (sentence sum over tokens) | tabular fixed=0")
    plt.title(f"Top-{args.top_k} sentences by IG | {label_name} | prob={prob_val:.4f} logit={logit_val:.4f}")
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "sentence_ig_topk.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    summary = {
        "stay_id": str(args.stay_id),
        "split": args.split,
        "label_name": label_name,
        "label_idx": label_idx,
        "logit": logit_val,
        "prob": prob_val,
        "max_length": args.max_length,
        "steps": args.steps,
        "baseline": args.baseline,
        "num_notes_used": len(notes),
        "note_chunk_size(max_notes_per_fwd_pass)": args.note_chunk_size,
        "tabular_fixed": "zeros",
        "out_csv": out_csv,
        "out_png": out_png,
        "html_files": [f"ig_highlight_note{ni}.html" for ni in range(len(notes))],
    }
    with open(os.path.join(args.out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] NO-TRAIN | stay_id={args.stay_id} | label={label_name} | prob={prob_val:.6f} logit={logit_val:.6f}")
    print(f"[OK] {out_csv}")
    print(f"[OK] {out_png}")
    print(f"[OK] HTMLs in {args.out_dir}")


if __name__ == "__main__":
    main()
