#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference-time modality ablation from ONE trained checkpoint (no retraining).
Four components -- static, text, vital, lab -- are removed at inference (see
sequence_dataset.drop_structured_channels / EventSequenceDataset.drop_modalities
for exactly how each is removed), at every landmark of the evaluation grid.

Configurations (--configs, default all 9):
  full                       nothing removed
  only_<c>   (4)             keep component c, remove the other three
  without_<c> (4)            remove component c, keep the other three
  only_structured            vital + lab only (static and text removed)
  without_structured         static + text only (vital and lab removed)

--split external scores the MIMIC cohort (event_dir = build_mimic_event_streams.py
output) on the outcomes in external_outcome_cols.json; val/test score all outcomes.
F1 uses the per-outcome thresholds stored in the checkpoint (UMN validation set).

Output under --out_dir:
  ablation_by_outcome.csv            config x landmark x outcome (+ macro): AUROC, AUPRC, F1
  ablation_detail_<config>.parquet   per-encounter probabilities/labels (for bootstrap_ci.py)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg
from common.event_utils import EncounterEventIndex, NoteEmbeddingIndex
from sequence_dataset import EventSequenceDataset, collate_sequence_batch, ABLATION_COMPONENTS
from sequence_model import EventSequenceModel
from train_sequence_model import eval_loop

CONFIGS = {"full": ()}
CONFIGS.update({f"only_{c}": tuple(x for x in ABLATION_COMPONENTS if x != c) for c in ABLATION_COMPONENTS})
CONFIGS.update({f"without_{c}": (c,) for c in ABLATION_COMPONENTS})
# vital + lab together as one "structured" component
CONFIGS["only_structured"] = ("static", "text")
CONFIGS["without_structured"] = ("vital", "lab")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", required=True)
    ap.add_argument("--embed_cache_dir", required=True)
    ap.add_argument("--ckpt_path", required=True, help="best_model.pt from train_sequence_model.py")
    ap.add_argument("--split", default="test", choices=["val", "test", "external"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
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

    scored = outcome_cols
    if args.split == "external":
        with open(os.path.join(args.event_dir, "external_outcome_cols.json")) as f:
            scored = json.load(f)
    scored_idx = [outcome_cols.index(c) for c in scored]

    static_df = pd.read_parquet(os.path.join(args.event_dir, f"{args.split}_static.parquet"))
    structured_df = pd.read_parquet(os.path.join(args.event_dir, f"{args.split}_structured_events.parquet"))
    vitals_index = EncounterEventIndex(static_df, structured_df, pd.DataFrame(columns=["stay_id", "t_minutes", "note_text"]))
    note_index = NoteEmbeddingIndex.from_cache_dir(args.embed_cache_dir, args.split)

    out_path = os.path.join(args.out_dir, "ablation_by_outcome.csv")
    rows = []
    for name in args.configs:
        detail = []
        for T in args.landmarks_minutes:
            ds = EventSequenceDataset(
                static_df, vitals_index, note_index, outcome_cols, static_cols,
                max_landmark_minutes=ckpt.get("max_landmark_minutes", cfg.MAX_LANDMARK_MINUTES_DEFAULT),
                seed=0, resample_each_epoch=False, fixed_landmark_minutes=T, drop_modalities=CONFIGS[name],
            )
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_sequence_batch, num_workers=args.num_workers, pin_memory=True)
            _, logits, labels, meta = eval_loop(model, loader, outcome_cols, device, fp16, return_meta=True)
            probs = 1 / (1 + np.exp(-logits))
            det = pd.DataFrame({"landmark_minutes": T, "stay_id": meta["stay_ids"]})
            for col, j in zip(scored, scored_idx):
                det[f"{col}_prob"] = probs[:, j].astype(np.float32)
                det[f"{col}_label"] = labels[:, j].astype(np.int8)
            detail.append(det)

            per = []
            for col, j in zip(scored, scored_idx):
                m = score(labels[:, j], probs[:, j], thresholds.get(col, 0.5))
                per.append(m)
                rows.append({"config": name, "landmark_minutes": T, "outcome": col, **m})
            macro = {k: float(np.nanmean([m[k] for m in per])) for k in ("AUROC", "AUPRC", "F1")}
            rows.append({"config": name, "landmark_minutes": T, "outcome": "macro", **macro})
            print(f"[{name:<16} T={T:g}] macro AUROC={macro['AUROC']:.4f} AUPRC={macro['AUPRC']:.4f} F1={macro['F1']:.4f}")
        # written after every config so a long run keeps finished configs
        pd.DataFrame(rows).to_csv(out_path, index=False)
        # per-encounter predictions, input to bootstrap_ci.py
        pd.concat(detail, ignore_index=True).to_parquet(
            os.path.join(args.out_dir, f"ablation_detail_{name}.parquet"), index=False)

    print(f">> Saved {len(args.configs)} configs x {len(args.landmarks_minutes)} landmarks to {out_path}")


if __name__ == "__main__":
    main()
