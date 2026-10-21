#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landmark trajectory -- AUROC/AUPRC at 0, 10, 20, 30, 60, 90, 120, 180, 240,
300, 360 minutes -- from ONE trained checkpoint (best_model.pt from
train_sequence_model.py), evaluated repeatedly at each fixed landmark on the
held-out test split. Every row of the output table comes from the same model
weights; only the amount of observed history changes.
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg
from common.event_utils import EncounterEventIndex, NoteEmbeddingIndex
from sequence_dataset import EventSequenceDataset, collate_sequence_batch
from sequence_model import EventSequenceModel
from train_sequence_model import eval_loop, pick_thresholds_from_val


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", type=str, default=cfg.EVENT_STREAM_OUTPUT_DIR_PLACEHOLDER, required=True)
    ap.add_argument("--embed_cache_dir", type=str, default=cfg.NOTE_EMBEDDING_CACHE_DIR_PLACEHOLDER, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True, help="best_model.pt from train_sequence_model.py")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--landmarks_minutes", type=float, nargs="+", default=cfg.LANDMARK_EVAL_GRID_MINUTES)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    outcome_cols = ckpt["outcome_cols"]
    static_cols = ckpt["static_cols"]
    thresholds = ckpt.get("thresholds", {})

    note_hidden = ckpt.get("note_hidden", 768)
    model = EventSequenceModel(
        num_static_features=len(static_cols), num_labels=len(outcome_cols), note_hidden=note_hidden
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    static_df = pd.read_parquet(os.path.join(args.event_dir, f"{args.split}_static.parquet"))
    vitals_df = pd.read_parquet(os.path.join(args.event_dir, f"{args.split}_structured_events.parquet"))
    vitals_index = EncounterEventIndex(static_df, vitals_df, pd.DataFrame(columns=["stay_id", "t_minutes", "note_text"]))
    note_index = NoteEmbeddingIndex.from_cache_dir(args.embed_cache_dir, args.split)

    triage_lookup = None
    if "triage_acuity" in static_df.columns:
        triage_lookup = static_df.set_index("stay_id")["triage_acuity"]

    rows = []
    per_landmark_detail = {}
    detail_rows = []
    for T in args.landmarks_minutes:
        ds = EventSequenceDataset(
            static_df, vitals_index, note_index, outcome_cols, static_cols,
            max_landmark_minutes=ckpt.get("max_landmark_minutes", cfg.MAX_LANDMARK_MINUTES_DEFAULT),
            seed=0, resample_each_epoch=False, fixed_landmark_minutes=T,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_sequence_batch, num_workers=args.num_workers, pin_memory=True)
        metrics, logits, labels, meta = eval_loop(model, loader, outcome_cols, device, fp16, return_meta=True)

        row = {"landmark_minutes": T, "macro_AUROC": metrics["macro"]["AUROC"], "macro_AUPRC": metrics["macro"]["AUPRC"]}
        rows.append(row)
        per_landmark_detail[str(T)] = metrics
        print(f"[T={T} min] macro AUROC={metrics['macro']['AUROC']:.4f} AUPRC={metrics['macro']['AUPRC']:.4f}")

        # ---- per-encounter detail: stay_id, per-outcome prob/label, n_notes/
        # n_vitals seen so far, ESI -- the raw material for the information-
        # content-axis analysis (reindexing by info count instead of clock
        # time).
        probs = 1 / (1 + np.exp(-logits))
        n_rows = len(meta["stay_ids"])
        for i in range(n_rows):
            stay_id = meta["stay_ids"][i]
            detail_row = {
                "landmark_minutes": T,
                "stay_id": stay_id,
                "n_notes_seen": int(meta["n_notes_seen"][i]),
                "n_vitals_seen": int(meta["n_vitals_seen"][i]),
                "triage_acuity": (triage_lookup.get(stay_id) if triage_lookup is not None else None),
            }
            for j, col in enumerate(outcome_cols):
                detail_row[f"{col}_prob"] = float(probs[i, j])
                detail_row[f"{col}_label"] = float(labels[i, j])
            detail_rows.append(detail_row)

    trajectory_df = pd.DataFrame(rows)
    trajectory_df.to_csv(os.path.join(args.out_dir, "landmark_trajectory.csv"), index=False)
    with open(os.path.join(args.out_dir, "landmark_trajectory_detail.json"), "w") as f:
        json.dump(per_landmark_detail, f, indent=2)

    # Flat (landmark x outcome) CSV -- same numbers as the JSON above, just
    # one row per (landmark, outcome) pair including "macro", so the full
    # per-disease trajectory can be opened directly (Excel, pasted into a
    # doc) instead of reading the nested JSON.
    by_outcome_rows = []
    for landmark_minutes, outcomes in per_landmark_detail.items():
        for outcome, m in outcomes.items():
            by_outcome_rows.append({
                "landmark_minutes": float(landmark_minutes),
                "outcome": outcome,
                "AUROC": m.get("AUROC"),
                "AUPRC": m.get("AUPRC"),
            })
    by_outcome_df = pd.DataFrame(by_outcome_rows).sort_values(
        ["landmark_minutes", "outcome"], key=lambda s: s if s.name != "outcome" else (s != "macro")
    )
    by_outcome_df.to_csv(os.path.join(args.out_dir, "landmark_trajectory_by_outcome.csv"), index=False)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_parquet(os.path.join(args.out_dir, "landmark_encounter_detail.parquet"), index=False)

    print(f">> Saved landmark trajectory (single checkpoint, {len(args.landmarks_minutes)} landmarks) to {args.out_dir}")
    print(f">> Saved per-outcome trajectory ({len(by_outcome_df):,} rows) to landmark_trajectory_by_outcome.csv")
    print(f">> Saved per-encounter detail ({len(detail_df):,} rows) to landmark_encounter_detail.parquet")


if __name__ == "__main__":
    main()
