#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
import random
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# =========================
# Outcome
# =========================
OUTCOME_COLS_LIST = [
    'outcome_ed_revisit_3d', 'outcome_hospitalization', 'outcome_critical',
    'outcome_sepsis', 'outcome_copd_exac', 'outcome_acs_mi', 'outcome_stroke',
    'outcome_ards', 'outcome_aki', 'outcome_bac_pne', 'outcome_viral_pne',
    'outcome_all_pne', 'outcome_asthma_exac', 'outcome_ahf', 'outcome_copd_asthma', 'outcome_pe'
]

# =========================
# Vitals 
# =========================
VITALS_MAP = {
    "Temp": "triage_temperature",
    "Heart Rate": "triage_heartrate",
    "Resp": "triage_resprate",
    "SpO2": "triage_o2sat",
    "BP (SYSTOLIC)": "triage_sbp",
    "BP (DIASTOLIC)": "triage_dbp",
}
TRIAGE_COLS = list(VITALS_MAP.values())


# =========================
# Tool functions
# =========================
def clean_id_series_dropna_then_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.astype("object").replace({"nan": np.nan, "None": np.nan})
    s = s.dropna()

    def _fix(x):
        x = str(x).strip()
        if re.fullmatch(r"-?\d+(\.0+)?", x):
            return re.sub(r"\.0+$", "", x)
        return x

    return s.apply(_fix)


def group_split_by_subject(df: pd.DataFrame, subject_col="subject_id", ratios=(0.8, 0.1, 0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    uniq = df[subject_col].dropna().astype(str).unique().tolist()
    random.Random(seed).shuffle(uniq)

    n = len(uniq)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])

    tr_subj = set(uniq[:n_tr])
    va_subj = set(uniq[n_tr:n_tr + n_va])
    te_subj = set(uniq[n_tr + n_va:])

    tr = df[df[subject_col].astype(str).isin(tr_subj)].copy()
    va = df[df[subject_col].astype(str).isin(va_subj)].copy()
    te = df[df[subject_col].astype(str).isin(te_subj)].copy()
    return tr, va, te


def parse_time_window_to_hours(x) -> float:
    """
    support:
      - t0
      - 10min / 20min
      - 0.5h / 1h / 12h
    """
    s = str(x).strip().lower()
    if s in ("t0", "0"):
        return 0.0

    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)

    m = re.fullmatch(r"(\d+(\.\d+)?)\s*(min|m|mins|minute|minutes)", s)
    if m:
        return float(m.group(1)) / 60.0

    m = re.fullmatch(r"(\d+(\.\d+)?)\s*(h|hr|hrs|hour|hours)", s)
    if m:
        return float(m.group(1))

    raise ValueError(f"Unrecognized time_window: {x}")


def apply_dx_boundary(intime: pd.Series, dx_time: pd.Series, window_hours: float) -> pd.Series:
    """
    effective_end = min(intime + window, dx_time) if dx_time notna else intime + window
    """
    time_end = intime + pd.Timedelta(hours=float(window_hours))
    if dx_time is None:
        return time_end

    dx_ok = dx_time.notna()
    out = time_end.copy()
    if dx_ok.any():
        out.loc[dx_ok] = np.minimum(
            time_end.loc[dx_ok].values.astype("datetime64[ns]"),
            dx_time.loc[dx_ok].values.astype("datetime64[ns]"),
        )
    return out


def parse_bp(display_name, value_orig):
    value_orig = str(value_orig)
    try:
        if "SYSTOLIC" in str(display_name):
            return value_orig.split('/')[0]
        elif "DIASTOLIC" in str(display_name):
            parts = value_orig.split('/')
            return parts[1] if len(parts) > 1 else np.nan
        else:
            return value_orig
    except Exception:
        return np.nan


# def load_vitals_auto(vitals_path: str) -> pd.DataFrame:
#     """
#     自动识别 TSV / pipe 分隔：
#       SERVICE_ID, DISPLAY_NAME, RECORDED_DATETIME, VALUE_ORIG
#     """
#     if not os.path.exists(vitals_path):
#         print(f"[FATAL ERROR] vitals_file not found: {vitals_path}", file=sys.stderr)
#         sys.exit(1)

#     # quick detect
#     with open(vitals_path, "r", encoding="utf-8-sig", errors="ignore") as f:
#         header = f.readline()

#     if "\t" in header:
#         seps = ["\t", "|"]
#     elif "|" in header:
#         seps = ["|", "\t"]
#     else:
#         seps = ["\t", "|"]

#     need = ["SERVICE_ID", "DISPLAY_NAME", "RECORDED_DATETIME", "VALUE_ORIG"]
#     last_err = None
#     for sep in seps:
#         try:
#             df = pd.read_csv(
#                 vitals_path,
#                 sep=sep,
#                 engine="python",
#                 dtype=str,
#                 on_bad_lines="skip",
#                 usecols=need
#             )
#             if df is not None and len(df) > 0:
#                 print(f"  [OK] vitals read via sep={repr(sep)} | rows={len(df)}")
#                 return df
#         except Exception as e:
#             last_err = e

#     print(f"[FATAL ERROR] Unable to read vitals with detected separators. Last error: {last_err}", file=sys.stderr)
#     sys.exit(1)


def load_vitals_auto(vitals_path: str) -> pd.DataFrame:
    with open(vitals_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"empty vitals file: {vitals_path}")

    cols = lines[0].rstrip("\n").split("\t")
    need = ["SERVICE_ID", "DISPLAY_NAME", "RECORDED_DATETIME", "VALUE_ORIG"]

    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"header missing columns: {missing}")

    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\\")
        if len(parts) != len(cols):
            continue
        rows.append(parts)

    df = pd.DataFrame(rows, columns=cols)
    df = df[need].copy()
    print(f"[OK] vitals loaded: rows={len(df)}")
    return df


def read_notes_from_dir(notes_dir: str, require_cols) -> pd.DataFrame:
    """
    load multiple notes parquet to concatenate
    """
    files = sorted(glob.glob(os.path.join(notes_dir, "*.parquet")))
    if not files:
        print(f"[FATAL ERROR] No parquet files found under notes_dir: {notes_dir}", file=sys.stderr)
        sys.exit(1)

    parts = []
    for fp in tqdm(files, desc="Reading notes parquet parts"):
        try:
            dfp = pd.read_parquet(fp, columns=list(require_cols))
        except Exception as e:
            print(f"[WARN] Failed reading notes parquet: {fp} ({e})", file=sys.stderr)
            continue
        if dfp is None or len(dfp) == 0:
            continue
        parts.append(dfp)

    if not parts:
        print(f"[FATAL ERROR] All notes parquet parts failed/empty under: {notes_dir}", file=sys.stderr)
        sys.exit(1)

    return pd.concat(parts, ignore_index=True)


# =========================
# arguments
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Build train/val/test from ONE master parquet + vitals file + notes parquet directory; "
                    "time window upper bound uses min(intime+window, dx_time in master)."
    )
    ap.add_argument("--master_parquet", type=str, required=True)
    ap.add_argument("--intime_col", type=str, default="intime")
    ap.add_argument("--subject_col", type=str, default="subject_id")
    ap.add_argument("--stay_col", type=str, default="stay_id")
    ap.add_argument("--dx_time_col", type=str, default="dx_time")

    ap.add_argument("--notes_dir", type=str, required=True)
    ap.add_argument("--notes_stay_col", type=str, default="stay_id")
    ap.add_argument("--notes_time_col", type=str, default="filing_date")
    ap.add_argument("--notes_text_col", type=str, default="note_text")

    ap.add_argument("--vitals_file", type=str, required=True)

    ap.add_argument("--time_window", type=str, default="2h",
                    help="e.g., t0, 10min, 20min, 0.5h, 1h, 2h, 12h, or numeric hours")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_year_le", type=int, default=2022)
    ap.add_argument("--test_year_ge", type=int, default=2023)

    ap.add_argument("--output_dir", type=str, required=True)
    return ap.parse_args()


# =========================
# main pipeline
# =========================
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    window_hours = parse_time_window_to_hours(args.time_window)
    print(f">> time_window={args.time_window} => {window_hours} hours")

    # -------------------------
    # 1) load master
    # -------------------------
    print(">> [1/7] Loading master parquet...")
    master = pd.read_parquet(args.master_parquet)

    # strip columns to avoid dx_time trailing spaces etc.
    master.columns = [str(c).strip() for c in master.columns]

    for c in [args.intime_col, args.subject_col, args.stay_col]:
        if c not in master.columns:
            print(f"[FATAL ERROR] Missing required col in master: {c}", file=sys.stderr)
            print(f"Available columns (first 120): {list(master.columns)[:120]}", file=sys.stderr)
            sys.exit(1)

    master[args.intime_col] = pd.to_datetime(master[args.intime_col], errors="coerce")
    master = master.dropna(subset=[args.intime_col]).copy()

    
    if args.dx_time_col in master.columns:
        master[args.dx_time_col] = pd.to_datetime(master[args.dx_time_col], errors="coerce")
    else:
        master[args.dx_time_col] = pd.NaT  # if missing：use intime+window

    # clean IDs
    print(">> [2/7] Cleaning IDs...")
    master[args.subject_col] = clean_id_series_dropna_then_str(master[args.subject_col]).reindex(master.index)
    master[args.stay_col] = clean_id_series_dropna_then_str(master[args.stay_col]).reindex(master.index)
    master = master.dropna(subset=[args.subject_col, args.stay_col]).copy()

    master[args.subject_col] = master[args.subject_col].astype(str)
    master[args.stay_col] = master[args.stay_col].astype(str)

    master["__year__"] = master[args.intime_col].dt.year

    # -------------------------
    # 2) split train/test by year, and split val from train
    # -------------------------
    print(">> [3/7] Splitting train/test by year and val from train...")
    train_pool = master[master["__year__"] <= int(args.train_year_le)].copy()
    test = master[master["__year__"] >= int(args.test_year_ge)].copy()

    n_train = 1.0 - float(args.val_ratio)
    n_val = float(args.val_ratio)
    
    leftover = max(0.0, 1.0 - n_train - n_val)

    train, val, _unused = group_split_by_subject(
        train_pool,
        subject_col=args.subject_col,
        ratios=(n_train, n_val, leftover),
        seed=args.seed
    )

    print(f"  TrainPool (<= {args.train_year_le}): {len(train_pool)}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test (>= {args.test_year_ge}): {len(test)}")

    # -------------------------
    # 3) 构造 stay_id -> intime, dx_time（从 master 的 split 直接取）
    # -------------------------
    print(">> [4/7] Building master_all_time (stay_id -> intime, dx_time)...")

    def _time_df(df_part: pd.DataFrame) -> pd.DataFrame:
        out = df_part[[args.stay_col, args.intime_col, args.dx_time_col]].copy()
        out = out.rename(columns={args.stay_col: "stay_id", args.intime_col: "intime", args.dx_time_col: "dx_time"})
        out["stay_id"] = out["stay_id"].astype(str)
        out["intime"] = pd.to_datetime(out["intime"], errors="coerce")
        out["dx_time"] = pd.to_datetime(out["dx_time"], errors="coerce")
        out = out.dropna(subset=["stay_id", "intime"]).copy()
        return out

    master_all_time = pd.concat([_time_df(train), _time_df(val), _time_df(test)], ignore_index=True)

    # use the earliest intime + earliest dx_time
    master_all_time = master_all_time.groupby("stay_id", as_index=False).agg(
        intime=("intime", "min"),
        dx_time=("dx_time", "min"),
    )

    # -------------------------
    # 4) Vitals: load + window filter (min(intime+window, dx_time))
    # -------------------------
    print(">> [5/7] Loading & processing vitals...")
    vitals_df = load_vitals_auto(args.vitals_file)

    vitals_df["stay_id"] = clean_id_series_dropna_then_str(vitals_df["SERVICE_ID"]).reindex(vitals_df.index)
    vitals_df = vitals_df.dropna(subset=["stay_id"]).copy()
    vitals_df["stay_id"] = vitals_df["stay_id"].astype(str)

    vitals_df["RECORDED_DATETIME"] = pd.to_datetime(vitals_df["RECORDED_DATETIME"], errors="coerce")
    vitals_df = vitals_df.dropna(subset=["RECORDED_DATETIME"]).copy()

    vitals_df["VALUE_CLEAN"] = vitals_df.apply(
        lambda row: parse_bp(row.get("DISPLAY_NAME", ""), row.get("VALUE_ORIG", "")),
        axis=1
    )
    vitals_df["VALUE_CLEAN"] = pd.to_numeric(vitals_df["VALUE_CLEAN"], errors="coerce")
    vitals_df = vitals_df.dropna(subset=["VALUE_CLEAN"]).copy()

    vitals_df["triage_col_name"] = vitals_df["DISPLAY_NAME"].map(VITALS_MAP)
    vitals_df = vitals_df.dropna(subset=["triage_col_name"]).copy()

    vitals_m = vitals_df.merge(master_all_time, on="stay_id", how="inner")
    eff_end = apply_dx_boundary(vitals_m["intime"], vitals_m["dx_time"], window_hours)

    vitals_win = vitals_m[vitals_m["RECORDED_DATETIME"] <= eff_end].copy()
    vitals_win = vitals_win.sort_values("RECORDED_DATETIME")

    last_vitals = vitals_win.groupby(["stay_id", "triage_col_name"])["VALUE_CLEAN"].agg("last").reset_index()
    vitals_wide = last_vitals.pivot(index="stay_id", columns="triage_col_name", values="VALUE_CLEAN")
    vitals_wide.columns = [f"{c}_vital" for c in vitals_wide.columns]
    print(f"  Vitals wide shape: {vitals_wide.shape}")

    # -------------------------
    # 5) Notes: load parquets + window filter + aggregate to list
    # -------------------------
    print(">> [6/7] Loading & processing notes from parquet directory...")
    notes_df = read_notes_from_dir(
        args.notes_dir,
        require_cols=(args.notes_stay_col, args.notes_time_col, args.notes_text_col)
    )

    notes_df[args.notes_stay_col] = clean_id_series_dropna_then_str(notes_df[args.notes_stay_col]).reindex(notes_df.index)
    notes_df = notes_df.dropna(subset=[args.notes_stay_col]).copy()
    notes_df[args.notes_stay_col] = notes_df[args.notes_stay_col].astype(str)

    notes_df[args.notes_time_col] = pd.to_datetime(notes_df[args.notes_time_col], errors="coerce")
    notes_df = notes_df.dropna(subset=[args.notes_time_col]).copy()

    notes_m = notes_df.merge(
        master_all_time.rename(columns={"stay_id": args.notes_stay_col}),
        on=args.notes_stay_col,
        how="inner"
    )

    notes_eff_end = apply_dx_boundary(notes_m["intime"], notes_m["dx_time"], window_hours)
    notes_win = notes_m[notes_m[args.notes_time_col] <= notes_eff_end].copy()
    notes_win = notes_win.sort_values(args.notes_time_col)

    notes_agg = notes_win.groupby(args.notes_stay_col)[args.notes_text_col].agg(list).reset_index()
    notes_agg.rename(columns={args.notes_stay_col: "stay_id"}, inplace=True)
    print(f"  Notes aggregated stays: {len(notes_agg)}")

    # -------------------------
    # 6) Build tabular features, scale, merge notes, save pkl
    # -------------------------
    print(">> [7/7] Building features, scaling, merging notes, saving...")

    # all_cols = train.columns.tolist()
    # HISTORICAL_COLS = [c for c in all_cols if str(c).startswith("cci_") or str(c).startswith("eci_")]
    all_cols = train.columns.tolist()
    HISTORICAL_COLS = [
        c for c in all_cols
        if str(c).startswith("cci_")
        or str(c).startswith("eci_")
        or str(c).startswith("chiefcom_")
    ]
    if "age" in all_cols:
        HISTORICAL_COLS = HISTORICAL_COLS + ["age"]

    missing_triage_cols = [c for c in TRIAGE_COLS if c not in train.columns]
    if missing_triage_cols:
        print(f"[FATAL ERROR] Master parquet missing these triage cols: {missing_triage_cols}", file=sys.stderr)
        sys.exit(1)

    FINAL_TABULAR_COLS = HISTORICAL_COLS + [f"final_{c.split('_', 1)[1]}" for c in TRIAGE_COLS]

    # triage numeric + fill from train mean
    for col in TRIAGE_COLS:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        val[col] = pd.to_numeric(val[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    triage_means = train[TRIAGE_COLS].mean()
    train[TRIAGE_COLS] = train[TRIAGE_COLS].fillna(triage_means)
    val[TRIAGE_COLS] = val[TRIAGE_COLS].fillna(triage_means)
    test[TRIAGE_COLS] = test[TRIAGE_COLS].fillna(triage_means)

    # historical fill 0
    for df_part in [train, val, test]:
        if HISTORICAL_COLS:
            df_part[HISTORICAL_COLS] = df_part[HISTORICAL_COLS].fillna(0)

    def build_scaled_tabular(df_part: pd.DataFrame, scaler: StandardScaler = None):
        df_part = df_part.copy()
        df_part = df_part.merge(vitals_wide, left_on=args.stay_col, right_index=True, how="left")

        final_cols_data = {}
        for triage_col in TRIAGE_COLS:
            vital_col = f"{triage_col}_vital"
            final_col = f"final_{triage_col.split('_', 1)[1]}"
            if vital_col in df_part.columns:
                final_cols_data[final_col] = df_part[vital_col].fillna(df_part[triage_col])
            else:
                final_cols_data[final_col] = df_part[triage_col]

        final_vitals_df = pd.DataFrame(final_cols_data, index=df_part.index)
        historical_df = df_part[HISTORICAL_COLS] if HISTORICAL_COLS else pd.DataFrame(index=df_part.index)
        tabular_df = pd.concat([historical_df, final_vitals_df], axis=1)
        tabular_df = tabular_df.reindex(columns=FINAL_TABULAR_COLS)

        if scaler is None:
            scaler = StandardScaler()
            tab_scaled = scaler.fit_transform(tabular_df.values)
        else:
            tab_scaled = scaler.transform(tabular_df.values)

        tab_scaled_df = pd.DataFrame(tab_scaled, columns=FINAL_TABULAR_COLS, index=df_part.index)
        return df_part, tab_scaled_df, scaler

    train_df_out, tab_train_scaled, scaler = build_scaled_tabular(train, scaler=None)
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    val_df_out, tab_val_scaled, _ = build_scaled_tabular(val, scaler=scaler)
    test_df_out, tab_test_scaled, _ = build_scaled_tabular(test, scaler=scaler)

    def save_split(name: str, master_part: pd.DataFrame, tab_scaled: pd.DataFrame):
        need_cols = [args.stay_col] + OUTCOME_COLS_LIST
        missing = [c for c in need_cols if c not in master_part.columns]
        if missing:
            print(f"[FATAL ERROR] Split {name} missing outcome cols: {missing}", file=sys.stderr)
            sys.exit(1)

        labels_df = master_part[need_cols].copy()
        final_df = labels_df.join(tab_scaled, how="inner")

        final_df = final_df.merge(notes_agg, left_on=args.stay_col, right_on="stay_id", how="left")

        # unify stay_id
        if "stay_id_y" in final_df.columns:
            final_df.drop(columns=["stay_id_y"], inplace=True)
        if "stay_id_x" in final_df.columns:
            final_df.rename(columns={"stay_id_x": "stay_id"}, inplace=True)
        elif args.stay_col != "stay_id":
            final_df.rename(columns={args.stay_col: "stay_id"}, inplace=True)

        out_path = os.path.join(args.output_dir, f"{name}.pkl")
        final_df.to_pickle(out_path)
        print(f"  [OK] Saved {name}: {out_path} | rows={len(final_df)}")

    save_split("train", train_df_out, tab_train_scaled)
    save_split("val", val_df_out, tab_val_scaled)
    save_split("test", test_df_out, tab_test_scaled)

    print("\n" + "=" * 60)
    print(">> DONE")
    print(f">> Output dir: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
