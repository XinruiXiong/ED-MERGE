#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1, Stage 2: freeze the (lightly fine-tuned, see stage1_finetune_bert.py)
BioClinicalBERT encoder and precompute one embedding per note, ONCE, cached to
disk. The sequence model (sequence_dataset.py / sequence_model.py /
train_sequence_model.py) trains only against these cached embeddings and never
calls BERT again -- this is the whole point of the two-stage design: decouple the
expensive step (done once here) from the cheap iterative step (many epochs
of landmark-sampled sequence-model training).

Per note: sliding-window tokenize (same doc_stride scheme as everywhere else
in this codebase), encode each
window chunk with BERT, then MEAN-POOL the chunk CLS vectors into a single
768-d embedding for that note. Mean pooling across a note's own chunks (not
attention pooling) is a deliberate simplification here -- there is no
trained pooling mechanism at cache-build time, and mean pooling is a
defensible, unbiased default. (This is a different pooling step than the
sequence model's own pooling *across notes*, which happens later, inside the
GRU -- see sequence_model.py.)

Output per split under --output_dir:
  {split}_note_embeddings.npy         float32 array, shape (N_notes, hidden_size)
  {split}_note_embeddings_index.parquet   stay_id, t_minutes, row_idx (join key into the .npy),
                                           plus tos/role AUDIT columns when present in the source
                                           notes_events.parquet (never used by the model -- lets a
                                           note-type/role sensitivity analysis filter
                                           these SAME cached embeddings post-hoc without retraining)
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg


def encode_notes_batch(texts, tokenizer, model, device, max_length, doc_stride, fp16):
    """Returns (n_texts, hidden_size) -- one mean-pooled embedding per input text."""
    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, stride=doc_stride,
        return_overflowing_tokens=True, return_tensors="pt",
    )
    overflow_mapping = tokenized.pop("overflow_to_sample_mapping")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (n_chunks, hidden)

    cls = cls.float().cpu().numpy()
    hidden_size = cls.shape[-1]
    n_texts = len(texts)
    pooled = np.zeros((n_texts, hidden_size), dtype=np.float32)
    counts = np.zeros(n_texts, dtype=np.int32)
    overflow_mapping = overflow_mapping.numpy()
    for chunk_i, text_i in enumerate(overflow_mapping):
        pooled[text_i] += cls[chunk_i]
        counts[text_i] += 1
    counts[counts == 0] = 1
    pooled = pooled / counts[:, None]
    return pooled


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 1 Stage 2: precompute + cache per-note BERT embeddings.")
    ap.add_argument("--event_dir", type=str, default=cfg.EVENT_STREAM_OUTPUT_DIR_PLACEHOLDER, required=True)
    ap.add_argument("--bert_dir", type=str, default=cfg.STAGE1_BERT_OUTPUT_DIR_PLACEHOLDER, required=True,
                     help="The bert_encoder/ dir saved by stage1_finetune_bert.py (or the raw "
                          "BioClinicalBERT dir if you want to skip Stage 1 entirely).")
    ap.add_argument("--output_dir", type=str, default=cfg.NOTE_EMBEDDING_CACHE_DIR_PLACEHOLDER, required=True)
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64, help="Notes per forward pass (before sliding-window expansion).")
    ap.add_argument("--no_fp16", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
    model = AutoModel.from_pretrained(args.bert_dir).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for split in args.splits:
        notes_path = os.path.join(args.event_dir, f"{split}_notes_events.parquet")
        if not os.path.exists(notes_path):
            print(f"[WARN] {notes_path} not found, skipping split={split}")
            continue

        notes_df = pd.read_parquet(notes_path).reset_index(drop=True)
        print(f">> [{split}] encoding {len(notes_df):,} notes...")

        if len(notes_df) == 0:
            np.save(os.path.join(args.output_dir, f"{split}_note_embeddings.npy"),
                    np.zeros((0, model.config.hidden_size), dtype=np.float32))
            notes_df.assign(row_idx=[]).to_parquet(
                os.path.join(args.output_dir, f"{split}_note_embeddings_index.parquet"), index=False
            )
            continue

        all_embeds = []
        texts = notes_df["note_text"].fillna("").astype(str).tolist()
        for start in tqdm(range(0, len(texts), args.batch_size), desc=f"encode:{split}"):
            batch_texts = texts[start:start + args.batch_size]
            batch_texts = [t if t.strip() else " " for t in batch_texts]  # tokenizer needs non-empty-ish input
            pooled = encode_notes_batch(batch_texts, tokenizer, model, device, args.max_length, args.doc_stride, fp16)
            all_embeds.append(pooled)

        embeds = np.concatenate(all_embeds, axis=0)
        assert embeds.shape[0] == len(notes_df), "embedding count must match note count"

        np.save(os.path.join(args.output_dir, f"{split}_note_embeddings.npy"), embeds.astype(np.float32))

        index_cols = ["stay_id", "t_minutes"] + [c for c in ("tos", "role") if c in notes_df.columns]
        index_df = notes_df[index_cols].copy()
        index_df["row_idx"] = np.arange(len(index_df))
        index_df.to_parquet(os.path.join(args.output_dir, f"{split}_note_embeddings_index.parquet"), index=False)

        print(f"[OK] {split}: cached {embeds.shape} to {args.output_dir}")

    print(">> Stage 2 done.")


if __name__ == "__main__":
    main()
