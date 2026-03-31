#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, pickle
import numpy as np
import pandas as pd

VITALS_MAP = {
    "triage_temperature": "final_temperature",
    "triage_heartrate": "final_heartrate",
    "triage_resprate": "final_resprate",
    "triage_o2sat": "final_o2sat",
    "triage_sbp": "final_sbp",
    "triage_dbp": "final_dbp",
}
HIST_PREFIX = ("cci_", "eci_")
BASE_LIKE = {"stay_id", "subject_id", "is_repeat", "NOTE_TEXT", "note_text"}

def parse_args():
    ap = argparse.ArgumentParser("preprocess MIMIC external -> train/val/test pkl (feature order from UMN train.pkl)")
    ap.add_argument("--mimic_csv", required=True)
    ap.add_argument("--umn_scaler_path", required=True)
    ap.add_argument("--umn_train_pkl_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", default="mimic_external")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    return ap.parse_args()

def feature_order_from_train(train_df: pd.DataFrame):
    train_outcomes = {c for c in train_df.columns if isinstance(c, str) and c.startswith("outcome_")}
    feats = []
    for c in train_df.columns:
        if c in BASE_LIKE:
            continue
        if c in train_outcomes:
            continue
        if isinstance(c, str) and (c.startswith("outcome_") or c.startswith("Unnamed")):
            continue
        feats.append(c)
    return feats

def ensure_numeric_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    return df

def split_by_stay(df: pd.DataFrame, seed, r_train, r_val):
    if "stay_id" not in df.columns:
        raise ValueError("missing stay_id")
    df["stay_id"] = df["stay_id"].astype(str)
    uniq = df["stay_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * r_train))
    n_va = int(round(n * r_val))
    tr = set(uniq[:n_tr])
    va = set(uniq[n_tr:n_tr+n_va])
    te = set(uniq[n_tr+n_va:])
    return (df[df["stay_id"].isin(tr)].reset_index(drop=True),
            df[df["stay_id"].isin(va)].reset_index(drop=True),
            df[df["stay_id"].isin(te)].reset_index(drop=True))

def main():
    a = parse_args()
    if abs((a.train_ratio + a.val_ratio + a.test_ratio) - 1.0) > 1e-6:
        print("[FATAL] ratios must sum to 1.0"); sys.exit(1)
    os.makedirs(a.output_dir, exist_ok=True)

    df = pd.read_csv(a.mimic_csv, low_memory=False)
    if "admission_note_text" in df.columns:
        df = df.rename(columns={"admission_note_text": "note_text"})
    elif "NOTE_TEXT" in df.columns and "note_text" not in df.columns:
        df = df.rename(columns={"NOTE_TEXT": "note_text"})
    if "note_text" not in df.columns:
        print("[FATAL] missing text col (need admission_note_text or note_text or NOTE_TEXT)"); sys.exit(1)

    df["note_text"] = df["note_text"].fillna("").astype(str).apply(lambda x: [x] if x.strip() else [""])

    # history numeric (cci_/eci_)
    hist_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(HIST_PREFIX)]
    if hist_cols:
        df[hist_cols] = df[hist_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # vitals -> final_*, missing -> 0, NaN -> mean (if mean NaN -> 0)
    for src, dst in VITALS_MAP.items():
        if src in df.columns:
            s = pd.to_numeric(df[src], errors="coerce")
            m = s.mean()
            if pd.isna(m): m = 0.0
            df[dst] = s.fillna(m)
        else:
            df[dst] = 0.0

    with open(a.umn_train_pkl_path, "rb") as f:
        train_df = pickle.load(f)
    if not isinstance(train_df, pd.DataFrame):
        print("[FATAL] train.pkl is not a DataFrame"); sys.exit(1)
    feats = feature_order_from_train(train_df)
    if not feats:
        print("[FATAL] extracted empty feature list from train.pkl"); sys.exit(1)

    df = ensure_numeric_cols(df, feats)

    with open(a.umn_scaler_path, "rb") as f:
        scaler = pickle.load(f)
    if hasattr(scaler, "mean_") and scaler.mean_.shape[0] != len(feats):
        print(f"[FATAL] scaler expects {scaler.mean_.shape[0]} features, but train.pkl gives {len(feats)}"); sys.exit(1)

    Xs = scaler.transform(df[feats].values.astype(np.float32, copy=False))
    tab_scaled = pd.DataFrame(Xs, columns=feats, index=df.index)

    outcome_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("outcome_")]
    out_df = pd.concat([df[["stay_id"] + outcome_cols + ["note_text"]].copy(), tab_scaled], axis=1)

    tr, va, te = split_by_stay(out_df, a.seed, a.train_ratio, a.val_ratio)
    tr.to_pickle(os.path.join(a.output_dir, f"{a.prefix}_train.pkl"))
    va.to_pickle(os.path.join(a.output_dir, f"{a.prefix}_val.pkl"))
    te.to_pickle(os.path.join(a.output_dir, f"{a.prefix}_test.pkl"))

    print(f"[OK] train={len(tr):,} val={len(va):,} test={len(te):,} | uniq_stay: "
          f"{tr.stay_id.nunique():,}/{va.stay_id.nunique():,}/{te.stay_id.nunique():,}")

if __name__ == "__main__":
    main()