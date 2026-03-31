#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import pickle


def parse_args():
    ap = argparse.ArgumentParser("Sample n positive stay_id from test.pkl by outcome column")
    ap.add_argument("--preprocessed_dir", type=str, required=True, help="Dir containing test.pkl")
    ap.add_argument("--outcome_col", type=str, required=True, help="Outcome column name, e.g. outcome_ed_death")
    ap.add_argument("--n", type=int, required=True, help="Number of positive stay_ids to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out_txt", type=str, default="", help="Optional output txt (one stay_id per line). If empty, just print.")
    ap.add_argument("--out_csv", type=str, default="", help="Optional output csv with stay_id + outcome value. If empty, no csv.")
    return ap.parse_args()


def main():
    args = parse_args()
    test_pkl = os.path.join(args.preprocessed_dir, "test.pkl")
    if not os.path.exists(test_pkl):
        raise FileNotFoundError(f"Not found: {test_pkl}")

    with open(test_pkl, "rb") as f:
        df = pickle.load(f)

    if "stay_id" not in df.columns:
        raise ValueError("test.pkl dataframe has no 'stay_id' column.")

    if args.outcome_col not in df.columns:
        # helpful hint: list available outcome cols
        cand = [c for c in df.columns if str(c).lower().startswith("outcome_")]
        raise ValueError(
            f"Outcome col '{args.outcome_col}' not found. "
            f"Available outcome_ cols count={len(cand)} (showing first 30): {cand[:30]}"
        )

    y = df[args.outcome_col]

    # Robust positive filter: accept 1/True/"1"/1.0; treat NaN as 0
    y_num = pd.to_numeric(y, errors="coerce").fillna(0.0)
    pos_mask = y_num >= 0.5  # supports 1 or 1.0
    df_pos = df.loc[pos_mask, ["stay_id", args.outcome_col]].copy()

    n_pos = len(df_pos)
    if n_pos == 0:
        raise ValueError(f"No positive rows found for {args.outcome_col} in test.pkl.")

    rng = np.random.default_rng(args.seed)
    if args.n > n_pos:
        print(f"[WARN] Requested n={args.n} but only {n_pos} positives exist. Sampling all positives.")
        sample_df = df_pos.sample(n=n_pos, random_state=args.seed)
    else:
        # pandas sample uses RandomState; that's fine
        sample_df = df_pos.sample(n=args.n, random_state=args.seed)

    # stay_id as clean strings
    stay_ids = sample_df["stay_id"].astype(str).str.strip().tolist()

    # Print
    print(f"[INFO] test.pkl positives for {args.outcome_col}: {n_pos}")
    print(f"[INFO] sampled n={len(stay_ids)} (seed={args.seed})")
    for sid in stay_ids:
        print(sid)

    # Write txt
    if args.out_txt:
        with open(args.out_txt, "w", encoding="utf-8") as f:
            for sid in stay_ids:
                f.write(f"{sid}\n")
        print(f"[OK] wrote: {args.out_txt}")

    # Write csv
    if args.out_csv:
        sample_df_out = sample_df.copy()
        sample_df_out["stay_id"] = sample_df_out["stay_id"].astype(str).str.strip()
        sample_df_out.to_csv(args.out_csv, index=False)
        print(f"[OK] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()