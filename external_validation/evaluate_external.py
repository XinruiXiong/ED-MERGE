#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External validation: score the UMN checkpoint on the MIMIC "external" split
(build_mimic_event_streams.py + precompute_note_embeddings.py --splits external)
at every landmark of the evaluation grid.

Only the outcomes in external_outcome_cols.json are scored (12; the model's
other outputs are ignored). F1 uses the per-outcome thresholds stored in the
checkpoint (selected on the UMN validation set) -- no MIMIC recalibration.

Outputs under --out_dir:
  external_trajectory_by_outcome.csv   landmark x outcome (+ macro): AUROC, AUPRC, F1, prevalence, n
  external_encounter_detail.parquet    per landmark x stay: prob/label per scored outcome, n_notes/n_vitals seen
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1_model"))
from common import config as cfg
from common.event_utils import EncounterEventIndex, NoteEmbeddingIndex
from sequence_dataset import EventSequenceDataset, collate_sequence_batch
from sequence_model import EventSequenceModel
from train_sequence_model import eval_loop

SPLIT = "external"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", required=True, help="output_dir of build_mimic_event_streams.py")
    ap.add_argument("--embed_cache_dir", required=True, help="output_dir of precompute_note_embeddings.py --splits external")
    ap.add_argument("--ckpt_path", required=True, help="UMN best_model.pt")
    ap.add_argument("--landmarks_minutes", type=float, nargs="+", default=cfg.LANDMARK_EVAL_GRID_MINUTES)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def score(y_true, prob, thr):
    if len(np.unique(y_true)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "F1": np.nan}
    return {
        "AUROC": roc_auc_score(y_true, prob),
        "AUPRC": average_precision_score(y_true, prob),
        "F1": f1_score(y_true, (prob >= thr).astype(int), zero_division=0),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    outcome_cols, static_cols = ckpt["outcome_cols"], ckpt["static_cols"]
    thresholds = ckpt.get("thresholds", {})
    model = EventSequenceModel(num_static_features=len(static_cols), num_labels=len(outcome_cols),
                               note_hidden=ckpt.get("note_hidden", 768)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(os.path.join(args.event_dir, "external_outcome_cols.json")) as f:
        scored = json.load(f)
    missing = [c for c in scored if c not in outcome_cols]
    if missing:
        sys.exit(f"[FATAL] scored outcomes not produced by the checkpoint: {missing}")
    scored_idx = [outcome_cols.index(c) for c in scored]

    static_df = pd.read_parquet(os.path.join(args.event_dir, f"{SPLIT}_static.parquet"))
    structured_df = pd.read_parquet(os.path.join(args.event_dir, f"{SPLIT}_structured_events.parquet"))
    vitals_index = EncounterEventIndex(static_df, structured_df, pd.DataFrame(columns=["stay_id", "t_minutes", "note_text"]))
    note_index = NoteEmbeddingIndex.from_cache_dir(args.embed_cache_dir, SPLIT)

    rows, detail = [], []
    for T in args.landmarks_minutes:
        ds = EventSequenceDataset(
            static_df, vitals_index, note_index, outcome_cols, static_cols,
            max_landmark_minutes=ckpt.get("max_landmark_minutes", cfg.MAX_LANDMARK_MINUTES_DEFAULT),
            seed=0, resample_each_epoch=False, fixed_landmark_minutes=T,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_sequence_batch, num_workers=args.num_workers, pin_memory=True)
        _, logits, labels, meta = eval_loop(model, loader, outcome_cols, device, fp16, return_meta=True)
        probs = 1 / (1 + np.exp(-logits))

        per = []
        for col, j in zip(scored, scored_idx):
            m = score(labels[:, j], probs[:, j], thresholds.get(col, 0.5))
            per.append(m)
            rows.append({"landmark_minutes": T, "outcome": col, **m,
                         "prevalence": float(labels[:, j].mean()), "n": int(len(labels))})
        macro = {k: float(np.nanmean([m[k] for m in per])) for k in ("AUROC", "AUPRC", "F1")}
        rows.append({"landmark_minutes": T, "outcome": "macro", **macro, "prevalence": np.nan, "n": int(len(labels))})
        print(f"[T={T:g} min] macro AUROC={macro['AUROC']:.4f} AUPRC={macro['AUPRC']:.4f} F1={macro['F1']:.4f}")

        d = pd.DataFrame({"landmark_minutes": T, "stay_id": meta["stay_ids"],
                          "n_notes_seen": meta["n_notes_seen"], "n_vitals_seen": meta["n_vitals_seen"]})
        for col, j in zip(scored, scored_idx):
            d[f"{col}_prob"] = probs[:, j]
            d[f"{col}_label"] = labels[:, j]
        detail.append(d)

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "external_trajectory_by_outcome.csv"), index=False)
    pd.concat(detail, ignore_index=True).to_parquet(
        os.path.join(args.out_dir, "external_encounter_detail.parquet"), index=False)
    print(f">> Saved external results ({len(scored)} outcomes x {len(args.landmarks_minutes)} landmarks) to {args.out_dir}")


if __name__ == "__main__":
    main()
