#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0: build per-encounter EVENT STREAMS (not per-window snapshots).

Design:

  1. Output is ONE dataset covering the whole encounter, not one dataset per
     observation window. Landmark time T is a
     training/eval-time slicing parameter (see phase1_model/sequence_dataset.py),
     not a data-build-time parameter.
  2. Vitals AND labs are stored as ONE forward-filled STRUCTURED STATE STREAM
     (one row per timestamp at which any channel changed), not collapsed into
     a single "final_*" snapshot per window. Labs widen this same state
     vector rather than becoming a separate event modality. Vitals channels are triage-seeded at intime; lab channels have no
     natural seed and stay NaN until their first real result.
  3. Notes are stored as a chronological event stream (raw text; embedding
     happens in Phase 1), not aggregated into an order-agnostic list. TOS
     (note type) and ROLE (author role) are carried through as AUDIT
     metadata only -- never fed to the model -- for a
     note-type/role sensitivity analysis that can filter the cached
     embeddings post-hoc without retraining.
  4. Subject-level exclusivity is enforced GLOBALLY (a subject's encounters
     never span train/val and test), while preserving the forward-in-time
     framing: patients first seen in or after --test_year_ge form the test
     set; everyone else forms the development pool (see build_subject_split).

Outputs under --output_dir:
  {split}_static.parquet            one row per stay_id: static features + outcomes
  {split}_structured_events.parquet long format: stay_id, t_minutes, STRUCTURED_CHANNELS
                                     (6 vital + 16 lab channels, see common/config.py)
  {split}_notes_events.parquet      long format: stay_id, t_minutes, note_text, tos, role
  static_scaler.pkl                 StandardScaler for static features (fit on train)
  vitals_scaler.pkl                 StandardScaler for STRUCTURED_CHANNELS (fit on train)
  subject_splits.csv                subject_id -> split, min_year (audit trail)
  split_diagnostics.json            split counts + overlap a year-only split would have had

  5. All `t_minutes` values are relative to a per-encounter ORIGIN, which is
     always `intime` (arrival), so landmark T=0 means exactly "at arrival".
  6. The truncation boundary is
     data_boundary_minutes = min(dx_boundary, coalesce(edouttime_boundary, admittime_boundary)),
     all relative to the origin above. `outtime` is not used (see
     common/config.py).

All path arguments have placeholder defaults from common/config.py --
override every one of them with real paths.
"""

import os
import re
import sys
import glob
import json
import random
import pickle
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg


# =========================================================================
# ID / text cleaning helpers
# =========================================================================
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


def load_dx_time_from_tables(dx_tables_dir: str) -> pd.DataFrame:
    """The 2026-10 master export dropped the combined dx_time column --
    dx_time now lives in separate per-outcome tables under dx_tables_dir
    (e.g. 09_out_dx_acs_mi.parquet: subject_id, stay_id, dx_time,
    outcome_<name>). Collects dx_time across ALL of them and, per stay_id,
    keeps the EARLIEST one -- the most conservative boundary, since
    documentation after ANY of the 16 outcomes' diagnosis could leak
    backward into "pre-diagnosis" dynamic data.

    dx_time only carries DATE granularity (no real hour/minute), so every value is pushed to 23:59:59 of its own calendar
    date before anything else happens to it. Without this, a same-day
    diagnosis (the common case) would parse as midnight, land BEFORE
    intime (now the origin_time), fail the physical-floor check downstream,
    and silently lose its boundary protection entirely -- worse than being
    imprecise, it would look like no diagnosis-based protection exists at
    all for that encounter.

    Returns a two-column frame: stay_id (cleaned string), dx_time.
    """
    files = sorted(glob.glob(os.path.join(dx_tables_dir, "*.parquet")))
    if not files:
        print(f"[FATAL ERROR] No parquet files found under dx_tables_dir: {dx_tables_dir}", file=sys.stderr)
        sys.exit(1)

    parts = []
    for fp in files:
        dfp = pd.read_parquet(fp, columns=["stay_id", "dx_time"])
        parts.append(dfp)
    all_dx = pd.concat(parts, ignore_index=True)

    all_dx["stay_id"] = clean_id_series_dropna_then_str(all_dx["stay_id"]).reindex(all_dx.index)
    all_dx = all_dx.dropna(subset=["stay_id"]).copy()
    all_dx["stay_id"] = all_dx["stay_id"].astype(str)
    all_dx["dx_time"] = pd.to_datetime(all_dx["dx_time"], errors="coerce")
    all_dx = all_dx.dropna(subset=["dx_time"]).copy()
    # Date-only -> end-of-day: normalize() zeroes the time-of-day, keeping
    # the date, then push to 23:59:59 of that same date.
    all_dx["dx_time"] = all_dx["dx_time"].dt.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

    earliest = all_dx.groupby("stay_id", as_index=False)["dx_time"].min()
    print(f"[OK] dx_time collected from {len(files)} per-outcome tables: "
          f"{len(earliest):,} stays have at least one matched diagnosis")
    return earliest


def read_notes_from_dir(notes_dir: str, require_cols, optional_cols=()) -> pd.DataFrame:
    """notes_dir may be a DIRECTORY of partitioned .parquet files OR a single
    .parquet FILE -- both are supported, detected via os.path.isfile().

    optional_cols (e.g. TOS/ROLE) are included only if present in the
    first part's schema -- checked once via pyarrow (cheap, no data read),
    consistent with the "warn, not fatal" pattern used for triage_acuity.
    """
    if os.path.isfile(notes_dir):
        files = [notes_dir]
    else:
        files = sorted(glob.glob(os.path.join(notes_dir, "*.parquet")))
    if not files:
        print(f"[FATAL ERROR] No parquet file(s) found at notes_dir: {notes_dir}", file=sys.stderr)
        sys.exit(1)

    first_schema_names = set(pq.ParquetFile(files[0]).schema_arrow.names)
    found_optional = [c for c in optional_cols if c in first_schema_names]
    missing_optional = [c for c in optional_cols if c not in first_schema_names]
    if missing_optional:
        print(f"[WARN] notes parquet missing optional cols, skipping: {missing_optional}")
    cols_to_read = list(require_cols) + found_optional

    parts = []
    for fp in tqdm(files, desc="Reading notes parquet parts"):
        try:
            dfp = pd.read_parquet(fp, columns=cols_to_read)
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


def _standardize_temperature_f(temp_series: pd.Series, unit_series: pd.Series) -> pd.Series:
    """Standardize BODY_TEMPERATURE to degrees Fahrenheit. Prints the raw
    unit value distribution so the first real run can sanity-check that no
    unexpected third unit string is silently mishandled."""
    unit_norm = unit_series.astype(str).str.upper()
    print(f"[INFO] BODY_TEMP_UNIT value counts:\n{unit_norm.value_counts(dropna=False).to_string()}")
    is_celsius = unit_norm.str.contains("CELSIUS")
    out = pd.to_numeric(temp_series, errors="coerce").copy()
    out.loc[is_celsius] = out.loc[is_celsius] * 9.0 / 5.0 + 32.0
    return out


def load_vitals_wide(vitals_parquet_path: str) -> pd.DataFrame:
    """SVC_BASIC_VITAL export -- one row per measurement event, wide format
    (all channels as columns)."""
    df = pd.read_parquet(vitals_parquet_path)
    print(f"[OK] vitals loaded: rows={len(df):,}")
    return df


def build_vitals_wide(vitals_raw: pd.DataFrame) -> pd.DataFrame:
    """Renames the new wide vitals source to internal channel names and
    standardizes temperature units. Returns stay_id, event_time, VITAL_CHANNELS.
    """
    v = vitals_raw.copy()
    v["stay_id"] = clean_id_series_dropna_then_str(v[cfg.VITALS_NEW_SERVICE_ID_COL]).reindex(v.index)
    v = v.dropna(subset=["stay_id"]).copy()
    v["stay_id"] = v["stay_id"].astype(str)

    v[cfg.VITALS_NEW_TIME_COL] = pd.to_datetime(v[cfg.VITALS_NEW_TIME_COL], errors="coerce")
    v = v.dropna(subset=[cfg.VITALS_NEW_TIME_COL]).copy()
    v = v.rename(columns={cfg.VITALS_NEW_TIME_COL: "event_time"})

    if "BODY_TEMPERATURE" in v.columns and cfg.VITALS_NEW_TEMP_UNIT_COL in v.columns:
        v["BODY_TEMPERATURE"] = _standardize_temperature_f(v["BODY_TEMPERATURE"], v[cfg.VITALS_NEW_TEMP_UNIT_COL])

    rename_map = {src: dst for src, dst in cfg.VITALS_NEW_COLUMN_MAP.items() if src in v.columns}
    v = v.rename(columns=rename_map)
    for ch in cfg.VITAL_CHANNELS:
        if ch not in v.columns:
            v[ch] = np.nan
        v[ch] = pd.to_numeric(v[ch], errors="coerce")

    return v[["stay_id", "event_time"] + cfg.VITAL_CHANNELS]


def load_lab_long(lab_parquet_path: str) -> pd.DataFrame:
    """lab.parquet -- already extracted/filtered to a fixed 16-LOINC-category
    panel (see common/config.py LAB_NAME_TO_CHANNEL)."""
    df = pd.read_parquet(lab_parquet_path)
    print(f"[OK] lab results loaded: rows={len(df):,}")
    return df


def build_labs_wide(lab_raw: pd.DataFrame) -> pd.DataFrame:
    """Pivots the long-format lab table (one row per test result) into wide
    format (one row per stay_id, event_time), one column per LAB_CHANNELS
    entry. No LOINC filtering here -- the source extraction already scoped
    to exactly this panel; unmapped LAB values are dropped with a warning
    as a safety net, not an expected occurrence.
    """
    l = lab_raw.copy()
    l["stay_id"] = clean_id_series_dropna_then_str(l[cfg.LAB_SERVICE_ID_COL]).reindex(l.index)
    l = l.dropna(subset=["stay_id"]).copy()
    l["stay_id"] = l["stay_id"].astype(str)

    l[cfg.LAB_TIME_COL] = pd.to_datetime(l[cfg.LAB_TIME_COL], errors="coerce")
    l = l.dropna(subset=[cfg.LAB_TIME_COL]).copy()
    l = l.rename(columns={cfg.LAB_TIME_COL: "event_time"})

    l["channel"] = l[cfg.LAB_NAME_COL].map(cfg.LAB_NAME_TO_CHANNEL)
    unmapped = sorted(l.loc[l["channel"].isna(), cfg.LAB_NAME_COL].dropna().unique().tolist())
    if unmapped:
        print(f"[WARN] lab.parquet has LAB values not in LAB_NAME_TO_CHANNEL, dropped: {unmapped}")
    l = l.dropna(subset=["channel"]).copy()

    l[cfg.LAB_VALUE_COL] = pd.to_numeric(l[cfg.LAB_VALUE_COL], errors="coerce")
    l = l.dropna(subset=[cfg.LAB_VALUE_COL]).copy()

    if l.empty:
        return pd.DataFrame(columns=["stay_id", "event_time"] + cfg.LAB_CHANNELS)

    wide = l.pivot_table(
        index=["stay_id", "event_time"], columns="channel", values=cfg.LAB_VALUE_COL, aggfunc="last"
    ).reset_index()
    for ch in cfg.LAB_CHANNELS:
        if ch not in wide.columns:
            wide[ch] = np.nan
    return wide[["stay_id", "event_time"] + cfg.LAB_CHANNELS]


# =========================================================================
# Subject-level split preserving forward-in-time framing
# =========================================================================
def build_subject_split(master: pd.DataFrame, subject_col, year_col,
                         train_year_le, test_year_ge, val_ratio, seed):
    subj_min_year = master.groupby(subject_col)[year_col].min()

    dev_subjects = set(subj_min_year[subj_min_year <= train_year_le].index)
    test_subjects = set(subj_min_year[subj_min_year > train_year_le].index)

    # Diagnostic: encounters a year-only split would have put into test
    # from patients already in the development pool.
    old_scheme_leak_mask = (master[year_col] >= test_year_ge) & (master[subject_col].isin(dev_subjects))
    old_scheme_leaked_encounters = int(old_scheme_leak_mask.sum())
    old_scheme_leaked_subjects = int(master.loc[old_scheme_leak_mask, subject_col].nunique())

    # Dev pool: ALL encounters (any year) belonging to dev subjects.
    dev_mask = master[subject_col].isin(dev_subjects)
    dev_pool_subjects = sorted(dev_subjects)
    rng = random.Random(seed)
    rng.shuffle(dev_pool_subjects)
    n_val = int(len(dev_pool_subjects) * val_ratio)
    val_subjects = set(dev_pool_subjects[:n_val])
    train_subjects = set(dev_pool_subjects[n_val:])

    split = pd.Series(index=master.index, dtype=object)
    split[dev_mask & master[subject_col].isin(train_subjects)] = "train"
    split[dev_mask & master[subject_col].isin(val_subjects)] = "val"

    test_mask = (master[year_col] >= test_year_ge) & (master[subject_col].isin(test_subjects))
    split[test_mask] = "test"
    # Anything left as NaN: a test_subject's encounter that falls in a gap
    # year (i.e. train_year_le < year < test_year_ge, only possible if you
    # pass non-contiguous cutoffs) -- dropped from all splits, not silently
    # merged into one. Report it rather than hide it.
    dropped = int(split.isna().sum())

    diagnostics = {
        "n_dev_subjects": len(dev_subjects),
        "n_test_subjects": len(test_subjects),
        "n_train_subjects": len(train_subjects),
        "n_val_subjects": len(val_subjects),
        "old_scheme_leaked_encounters_into_test": old_scheme_leaked_encounters,
        "old_scheme_leaked_subjects_into_test": old_scheme_leaked_subjects,
        "n_encounters_dropped_gap_year": dropped,
        "n_encounters_train": int((split == "train").sum()),
        "n_encounters_val": int((split == "val").sum()),
        "n_encounters_test": int((split == "test").sum()),
    }
    return split, diagnostics


# =========================================================================
# Structured event stream: forward-filled vitals+labs state (STRUCTURED_
# CHANNELS-wide), one row per timestamp at which any channel changed.
# =========================================================================
def build_structured_event_stream(master_time: pd.DataFrame, vitals_wide: pd.DataFrame,
                                   labs_wide: pd.DataFrame, triage_wide: pd.DataFrame) -> pd.DataFrame:
    """
    master_time: stay_id -> intime, origin_time, data_boundary_minutes
      (origin_time and data_boundary_minutes: see the module docstring)
    vitals_wide: stay_id, event_time, VITAL_CHANNELS (from build_vitals_wide)
    labs_wide: stay_id, event_time, LAB_CHANNELS (from build_labs_wide)
    triage_wide: stay_id -> triage_{channel} columns (vitals seed values, anchored at intime)

    Returns long-format dataframe: stay_id, t_minutes, <STRUCTURED_CHANNELS...>
    with every row fully populated (no NaNs for vitals; lab channels may
    legitimately stay NaN until their first real result) via forward fill,
    truncated at data_boundary_minutes.

    Labs have no triage-style seed -- there is no "triage lactate". A lab
    channel simply stays NaN (later filled with the train-population mean at
    scaling time, same as a vital channel that was never once recorded for a
    given stay) until its first real result appears in the stream.

    t_minutes is relative to origin_time, NOT intime -- see the equivalent
    note in build_notes_event_stream's docstring; identical reasoning here.
    """
    combined = pd.concat([vitals_wide, labs_wide], ignore_index=True, sort=False)
    for ch in cfg.STRUCTURED_CHANNELS:
        if ch not in combined.columns:
            combined[ch] = np.nan

    combined = combined.merge(
        master_time[["stay_id", "intime", "origin_time", "data_boundary_minutes"]], on="stay_id", how="inner"
    )

    # Physical floor: never include a reading from before actual arrival,
    # checked in absolute time against intime (independent of origin_time).
    combined = combined[combined["event_time"] >= combined["intime"]]
    combined["t_minutes"] = (combined["event_time"] - combined["origin_time"]).dt.total_seconds() / 60.0

    has_boundary = combined["data_boundary_minutes"].notna()
    combined = combined[(~has_boundary) | (combined["t_minutes"] <= combined["data_boundary_minutes"])]
    combined = combined[["stay_id", "t_minutes"] + cfg.STRUCTURED_CHANNELS]

    # Seed rows from triage (vitals channels only) -- guarantees every vital
    # channel is initialized even if the vitals source never has a reading
    # for it. Anchored at intime, expressed as t_minutes relative to
    # origin_time -- so this can be <= 0. Lab channels stay NaN here.
    triage_seed_base = master_time[["stay_id", "intime", "origin_time"]].copy()
    triage_seed_base["t_minutes"] = (
        (triage_seed_base["intime"] - triage_seed_base["origin_time"]).dt.total_seconds() / 60.0
    )
    seed_rows = triage_seed_base[["stay_id", "t_minutes"]].merge(triage_wide, on="stay_id", how="left")
    seed_rows = seed_rows.rename(columns={f"triage_{ch}": ch for ch in cfg.VITAL_CHANNELS})
    for ch in cfg.STRUCTURED_CHANNELS:
        if ch not in seed_rows.columns:
            seed_rows[ch] = np.nan
    seed_rows = seed_rows[seed_rows[cfg.VITAL_CHANNELS].notna().any(axis=1)]
    seed_rows = seed_rows[["stay_id", "t_minutes"] + cfg.STRUCTURED_CHANNELS]

    all_events = pd.concat([seed_rows, combined], ignore_index=True, sort=False)
    if all_events.empty:
        return pd.DataFrame(columns=["stay_id", "t_minutes"] + cfg.STRUCTURED_CHANNELS)

    # Collapse rows sharing the same (stay_id, t_minutes) -- e.g. a vitals
    # reading and a lab result landing at the exact same timestamp -- by
    # taking the last non-null value per channel (GroupBy.last() skips NaN).
    all_events = all_events.groupby(["stay_id", "t_minutes"], as_index=False)[cfg.STRUCTURED_CHANNELS].last()

    all_events = all_events.sort_values(["stay_id", "t_minutes"])
    all_events[cfg.STRUCTURED_CHANNELS] = all_events.groupby("stay_id")[cfg.STRUCTURED_CHANNELS].ffill()

    # Any channel still NaN after ffill means this stay had no triage value
    # (vitals) or no result at all (labs) ever -- fill with the training-set
    # population mean at scaling time (Phase 1), not here; here we just flag
    # it by leaving NaN, ensure_numeric-style handling happens downstream.
    return all_events[["stay_id", "t_minutes"] + cfg.STRUCTURED_CHANNELS]


# =========================================================================
# Notes event stream: raw text, chronological, truncated at data_boundary.
# =========================================================================
def build_notes_event_stream(master_time: pd.DataFrame, notes_raw: pd.DataFrame,
                              stay_col, time_col, text_col, tos_col=None, role_col=None) -> pd.DataFrame:
    """t_minutes relative to origin_time; physical floor enforced against
    intime directly. See build_structured_event_stream's docstring --
    identical logic, applied to notes instead of vitals/labs.

    tos_col/role_col (TOS/ROLE), when present in notes_raw, are carried
    through as "tos"/"role" AUDIT columns -- never used by the model. See
    common/config.py's NOTES_TOS_COL_DEFAULT docstring for why.
    """
    n = notes_raw.copy()
    n[stay_col] = clean_id_series_dropna_then_str(n[stay_col]).reindex(n.index)
    n = n.dropna(subset=[stay_col]).copy()
    n[stay_col] = n[stay_col].astype(str)
    n = n.rename(columns={stay_col: "stay_id"})

    n[time_col] = pd.to_datetime(n[time_col], errors="coerce")
    n = n.dropna(subset=[time_col]).copy()

    n = n.merge(master_time[["stay_id", "intime", "origin_time", "data_boundary_minutes"]], on="stay_id", how="inner")

    n = n[n[time_col] >= n["intime"]]  # physical floor, absolute time against intime
    n["t_minutes"] = (n[time_col] - n["origin_time"]).dt.total_seconds() / 60.0

    has_boundary = n["data_boundary_minutes"].notna()
    n = n[(~has_boundary) | (n["t_minutes"] <= n["data_boundary_minutes"])]

    rename_map = {text_col: "note_text"}
    out_cols = ["stay_id", "t_minutes", "note_text"]
    if tos_col and tos_col in n.columns:
        rename_map[tos_col] = "tos"
        out_cols.append("tos")
    if role_col and role_col in n.columns:
        rename_map[role_col] = "role"
        out_cols.append("role")

    n = n.rename(columns=rename_map)
    n = n[out_cols].sort_values(["stay_id", "t_minutes"])
    return n.reset_index(drop=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 0: build per-encounter timestamped event streams.")
    ap.add_argument("--master_parquet", type=str, default=cfg.MASTER_PARQUET_PLACEHOLDER, required=True)
    ap.add_argument("--intime_col", type=str, default=cfg.MASTER_INTIME_COL)
    ap.add_argument("--subject_col", type=str, default=cfg.MASTER_SUBJECT_COL)
    ap.add_argument("--stay_col", type=str, default=cfg.MASTER_STAY_COL)
    ap.add_argument("--dx_time_col", type=str, default=cfg.MASTER_DX_TIME_COL,
                     help="Name to give the per-stay earliest-diagnosis-time column once "
                          "computed from --dx_tables_dir (master_parquet no longer carries "
                          "a combined dx_time column as of the 2026-10 export).")
    ap.add_argument("--dx_tables_dir", type=str, required=True,
                     help="Directory of per-outcome diagnosis tables (e.g. 09_out_dx_acs_mi.parquet), "
                          "each with at least stay_id + dx_time. dx_time is taken as the EARLIEST "
                          "match across all of them per stay_id -- the most conservative choice, "
                          "since documentation after ANY of the 16 outcomes' diagnosis could leak "
                          "backward.")
    ap.add_argument("--edouttime_col", type=str, default=cfg.MASTER_EDOUTTIME_COL,
                     help="True ED-departure timestamp (sparse for hospitalized patients).")
    ap.add_argument("--admittime_col", type=str, default=cfg.MASTER_ADMITTIME_COL,
                     help="Hospital admission timestamp -- complements edouttime's coverage "
                          "gap for hospitalized patients.")
    ap.add_argument("--triage_acuity_col", type=str, default=cfg.MASTER_TRIAGE_ACUITY_COL,
                     help="ESI triage acuity -- carried into {split}_static.parquet as an "
                          "audit/metadata column only (not a model input, not scaled). Used "
                          "downstream for confound-stratified analysis. If missing from the "
                          "master parquet, this is a warning, not a fatal error.")

    ap.add_argument("--notes_dir", type=str, default=cfg.NOTES_DIR_PLACEHOLDER, required=True,
                     help="A directory of partitioned notes_*.parquet files, OR a single "
                          "consolidated notes.parquet file -- both are auto-detected.")
    ap.add_argument("--notes_stay_col", type=str, default=cfg.NOTES_STAY_COL_DEFAULT)
    ap.add_argument("--notes_time_col", type=str, default=cfg.NOTES_TIME_COL_DEFAULT)
    ap.add_argument("--notes_text_col", type=str, default=cfg.NOTES_TEXT_COL_DEFAULT)
    ap.add_argument("--notes_tos_col", type=str, default=cfg.NOTES_TOS_COL_DEFAULT,
                     help="Note type (audit metadata only, not a model input). Missing -> warning, not fatal.")
    ap.add_argument("--notes_role_col", type=str, default=cfg.NOTES_ROLE_COL_DEFAULT,
                     help="Author's clinical role (audit metadata only, not a model input). Missing -> warning, not fatal.")

    ap.add_argument("--vitals_parquet", type=str, default=cfg.VITALS_PARQUET_PLACEHOLDER, required=True,
                     help="SVC_BASIC_VITAL export (wide format).")
    ap.add_argument("--lab_parquet", type=str, default=cfg.LAB_PARQUET_PLACEHOLDER, required=True,
                     help="Pre-filtered lab.parquet (fixed 16-LOINC-category panel). See common/config.py LAB_NAME_TO_CHANNEL.")

    ap.add_argument("--train_year_le", type=int, default=cfg.DEFAULT_TRAIN_YEAR_LE)
    ap.add_argument("--test_year_ge", type=int, default=cfg.DEFAULT_TEST_YEAR_GE)
    ap.add_argument("--val_ratio", type=float, default=cfg.DEFAULT_VAL_RATIO_WITHIN_DEV_POOL)
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)

    ap.add_argument("--output_dir", type=str, default=cfg.EVENT_STREAM_OUTPUT_DIR_PLACEHOLDER, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- 1) Load master ----------------
    print(">> [1/8] Loading master parquet...")
    master = pd.read_parquet(args.master_parquet)
    master.columns = [str(c).strip() for c in master.columns]

    required = [args.intime_col, args.subject_col, args.stay_col]
    missing = [c for c in required if c not in master.columns]
    if missing:
        print(f"[FATAL ERROR] Missing required cols in master: {missing}", file=sys.stderr)
        sys.exit(1)

    master[args.intime_col] = pd.to_datetime(master[args.intime_col], errors="coerce")
    master = master.dropna(subset=[args.intime_col]).copy()

    for col in [args.edouttime_col, args.admittime_col]:
        if col in master.columns:
            master[col] = pd.to_datetime(master[col], errors="coerce")
        else:
            print(f"[WARN] column '{col}' not found in master_parquet -- treated as entirely missing.")
            master[col] = pd.NaT

    print(">> [2/8] Cleaning IDs...")
    master[args.subject_col] = clean_id_series_dropna_then_str(master[args.subject_col]).reindex(master.index)
    master[args.stay_col] = clean_id_series_dropna_then_str(master[args.stay_col]).reindex(master.index)
    master = master.dropna(subset=[args.subject_col, args.stay_col]).copy()
    master[args.subject_col] = master[args.subject_col].astype(str)
    master[args.stay_col] = master[args.stay_col].astype(str)
    master["__year__"] = master[args.intime_col].dt.year

    # dx_time is not in master_parquet -- build it from
    # the per-outcome diagnosis tables and merge in under args.dx_time_col,
    # so every downstream reference to master[args.dx_time_col] (the
    # dx_boundary_minutes computation below) needs no further changes.
    dx_time_table = load_dx_time_from_tables(args.dx_tables_dir)
    dx_time_table = dx_time_table.rename(columns={"stay_id": args.stay_col, "dx_time": args.dx_time_col})
    master = master.merge(dx_time_table, on=args.stay_col, how="left")

    # ---------------- 2) Subject-level split ----------------
    print(">> [3/8] Building subject-level split (forward-in-time framing preserved)...")
    split, split_diag = build_subject_split(
        master, args.subject_col, "__year__", args.train_year_le, args.test_year_ge,
        args.val_ratio, args.seed,
    )
    master["split"] = split
    print(json.dumps(split_diag, indent=2))

    subj_audit = master[[args.subject_col, "__year__", "split"]].drop_duplicates(subset=[args.subject_col])
    subj_audit = subj_audit.rename(columns={args.subject_col: "subject_id", "__year__": "min_year_seen_here"})
    subj_audit.to_csv(os.path.join(args.output_dir, "subject_splits.csv"), index=False)

    master = master.dropna(subset=["split"]).copy()

    # ---------------- 3) origin_time + boundaries + static feature columns ----------------
    print(">> [4/8] Computing origin_time, boundaries, and static feature columns...")

    # origin_time: t0 for all relative t_minutes. Always intime, so T=0 means
    # exactly "at arrival". (edregtime can occur well after intime, so using
    # it would let T=0 include everything recorded in that gap.)
    master["origin_time"] = master[args.intime_col]

    def _boundary_minutes_relative_to_origin(time_col_series):
        """A candidate boundary timestamp -> minutes relative to origin_time,
        NaN if physically impossible (before intime -- checked against intime,
        NOT origin_time, since origin_time can itself be after intime; a
        boundary that falls between intime and origin_time is legitimate and
        may come out negative on purpose, see build_structured_event_stream's
        docstring).
        """
        physically_ok = time_col_series.notna() & (time_col_series >= master[args.intime_col])
        minutes = (time_col_series - master["origin_time"]).dt.total_seconds() / 60.0
        return minutes.where(physically_ok, np.nan)

    master["dx_boundary_minutes"] = _boundary_minutes_relative_to_origin(master[args.dx_time_col])
    master["edouttime_boundary_minutes"] = _boundary_minutes_relative_to_origin(master[args.edouttime_col])
    master["admittime_boundary_minutes"] = _boundary_minutes_relative_to_origin(master[args.admittime_col])

    # Disposition boundary: prefer edouttime (true ED-departure) when present;
    # fall back to admittime (covers the 99% of hospitalized encounters where
    # edouttime is never logged). Coalesce, not min-of-both: edouttime is
    # LATER than admittime 55.26% of the time when both exist (real ED
    # boarding -- patient administratively admitted while still physically in
    # the ED awaiting a bed), and boarding-era documentation is still
    # legitimately ED-phase data.
    master["disposition_boundary_minutes"] = master["edouttime_boundary_minutes"].where(
        master["edouttime_boundary_minutes"].notna(), master["admittime_boundary_minutes"]
    )

    # data_boundary_minutes = min(dx_boundary, disposition_boundary), NaN-safe.
    # This -- not dx_boundary_minutes alone -- is what actually truncates the
    # vitals/notes event streams below. dx_time alone leaves 87.4% of
    # encounters with zero protection (it's null for essentially all
    # operational-outcome-only encounters); the combined boundary above
    # covers 98.6% overall and 100% of hospitalized encounters specifically.
    master["data_boundary_minutes"] = master[["dx_boundary_minutes", "disposition_boundary_minutes"]].min(axis=1, skipna=True)

    all_cols = master.columns.tolist()
    hist_cols = [c for c in all_cols if str(c).startswith(("cci_", "eci_", "chiefcom_"))]
    static_raw_cols = hist_cols + (["age"] if "age" in all_cols else [])

    missing_triage = [c for c in cfg.TRIAGE_COLS if c not in master.columns]
    if missing_triage:
        print(f"[FATAL ERROR] master parquet missing triage cols: {missing_triage}", file=sys.stderr)
        sys.exit(1)

    for c in static_raw_cols:
        if c.startswith(("cci_", "eci_", "chiefcom_")):
            master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0)
    if "age" in master.columns:
        master["age"] = pd.to_numeric(master["age"], errors="coerce")
        master["age"] = master["age"].fillna(master.loc[master["split"] == "train", "age"].mean())

    outcome_cols = [c for c in cfg.OUTCOME_COLS_LIST if c in master.columns]
    missing_outcomes = [c for c in cfg.OUTCOME_COLS_LIST if c not in master.columns]
    if missing_outcomes:
        print(f"[WARN] outcome cols not present in master, skipping: {missing_outcomes}")

    has_triage_acuity = args.triage_acuity_col in master.columns
    if not has_triage_acuity:
        print(f"[WARN] triage acuity col '{args.triage_acuity_col}' not present in master -- "
              f"static.parquet will not have a triage_acuity column (confound-stratified "
              f"analysis downstream will need this).")

    master["stay_id_key"] = master[args.stay_col]

    # ---------------- 4) Fit static scaler on TRAIN only, transform all ----------------
    print(">> [5/8] Fitting static feature scaler on train split only...")
    static_scaler = StandardScaler()
    train_mask = master["split"] == "train"
    static_scaler.fit(master.loc[train_mask, static_raw_cols].values)
    with open(os.path.join(args.output_dir, "static_scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": static_scaler, "columns": static_raw_cols}, f)

    scaled = static_scaler.transform(master[static_raw_cols].values)
    scaled_df = pd.DataFrame(scaled, columns=static_raw_cols, index=master.index)

    static_meta_cols = ["stay_id_key", args.subject_col, "split",
                         "dx_boundary_minutes", "edouttime_boundary_minutes", "admittime_boundary_minutes",
                         "disposition_boundary_minutes", "data_boundary_minutes"]
    if has_triage_acuity:
        static_meta_cols.append(args.triage_acuity_col)
    static_out = master[static_meta_cols + outcome_cols].copy()
    rename_map = {"stay_id_key": "stay_id", args.subject_col: "subject_id"}
    if has_triage_acuity and args.triage_acuity_col != "triage_acuity":
        rename_map[args.triage_acuity_col] = "triage_acuity"
    static_out = static_out.rename(columns=rename_map)
    static_out = pd.concat([static_out, scaled_df.add_suffix("")], axis=1)
    # NOTE: scaled_df already has the plain column names (age, cci_*, ...);
    # concatenating directly is intentional -- these replace the raw values.

    # ---------------- 5) Structured (vitals+labs) event stream ----------------
    print(">> [6/8] Loading & building structured (vitals+labs) event stream...")
    master_time = master[["stay_id_key", args.intime_col, "origin_time", "data_boundary_minutes"]].rename(
        columns={"stay_id_key": "stay_id", args.intime_col: "intime"}
    )
    triage_wide = master[["stay_id_key"] + cfg.TRIAGE_COLS].rename(columns={"stay_id_key": "stay_id"})
    for c in cfg.TRIAGE_COLS:
        triage_wide[c] = pd.to_numeric(triage_wide[c], errors="coerce")

    vitals_raw = load_vitals_wide(args.vitals_parquet)
    vitals_wide = build_vitals_wide(vitals_raw)
    lab_raw = load_lab_long(args.lab_parquet)
    labs_wide = build_labs_wide(lab_raw)

    structured_events = build_structured_event_stream(master_time, vitals_wide, labs_wide, triage_wide)

    print(">> Fitting structured-channel scaler on train split events only...")
    train_stay_ids = set(master.loc[train_mask, "stay_id_key"])
    structured_train_mask = structured_events["stay_id"].isin(train_stay_ids)
    vitals_scaler = StandardScaler()
    # Fill remaining NaNs (channel never observed for that stay) with train
    # population mean BEFORE fitting -- applies uniformly to vitals and lab
    # channels alike.
    train_means = structured_events.loc[structured_train_mask, cfg.STRUCTURED_CHANNELS].mean()
    channels_never_in_train = [c for c in cfg.STRUCTURED_CHANNELS if pd.isna(train_means[c])]
    if channels_never_in_train:
        # A channel with ZERO occurrences anywhere in train (e.g. a rarely-
        # ordered lab that happens to be absent from this cohort/split) has
        # no population mean to fall back on -- fillna(train_means) would
        # leave it all-NaN, which poisons the scaler's mean_/scale_ for that
        # whole column (NaN propagates to every row, every split, not just
        # the missing ones). Fall back to 0 (post-scaling this is exactly
        # what a "no information" value should look like anyway).
        print(f"[WARN] channels never observed in train split, falling back to 0: {channels_never_in_train}")
        train_means = train_means.fillna(0.0)
    structured_events[cfg.STRUCTURED_CHANNELS] = structured_events[cfg.STRUCTURED_CHANNELS].fillna(train_means)
    vitals_scaler.fit(structured_events.loc[structured_train_mask, cfg.STRUCTURED_CHANNELS].values)
    with open(os.path.join(args.output_dir, "vitals_scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": vitals_scaler, "columns": cfg.STRUCTURED_CHANNELS, "train_means": train_means.to_dict()}, f)

    scaled_v = vitals_scaler.transform(structured_events[cfg.STRUCTURED_CHANNELS].values)
    structured_events[cfg.STRUCTURED_CHANNELS] = scaled_v

    # ---------------- 6) Notes event stream ----------------
    print(">> [7/8] Loading & building notes event stream...")
    notes_raw = read_notes_from_dir(
        args.notes_dir,
        require_cols=(args.notes_stay_col, args.notes_time_col, args.notes_text_col),
        optional_cols=(args.notes_tos_col, args.notes_role_col),
    )
    notes_events = build_notes_event_stream(
        master_time, notes_raw, args.notes_stay_col, args.notes_time_col, args.notes_text_col,
        tos_col=args.notes_tos_col, role_col=args.notes_role_col,
    )

    # ---------------- 7) Save per split ----------------
    print(">> [8/8] Saving outputs per split...")
    split_diag["n_static_rows"] = {}
    split_diag["n_structured_events"] = {}
    split_diag["n_notes_events"] = {}

    for sp in ["train", "val", "test"]:
        stay_ids_sp = set(master.loc[master["split"] == sp, "stay_id_key"])

        static_sp = static_out[static_out["split"] == sp].drop(columns=["split"])
        static_sp.to_parquet(os.path.join(args.output_dir, f"{sp}_static.parquet"), index=False)

        structured_sp = structured_events[structured_events["stay_id"].isin(stay_ids_sp)]
        structured_sp.to_parquet(os.path.join(args.output_dir, f"{sp}_structured_events.parquet"), index=False)

        notes_sp = notes_events[notes_events["stay_id"].isin(stay_ids_sp)]
        notes_sp.to_parquet(os.path.join(args.output_dir, f"{sp}_notes_events.parquet"), index=False)

        split_diag["n_static_rows"][sp] = int(len(static_sp))
        split_diag["n_structured_events"][sp] = int(len(structured_sp))
        split_diag["n_notes_events"][sp] = int(len(notes_sp))

        print(f"  [OK] {sp}: static={len(static_sp):,} structured_events={len(structured_sp):,} notes_events={len(notes_sp):,}")

    with open(os.path.join(args.output_dir, "split_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(split_diag, f, indent=2)

    with open(os.path.join(args.output_dir, "outcome_cols.json"), "w", encoding="utf-8") as f:
        json.dump(outcome_cols, f, indent=2)
    with open(os.path.join(args.output_dir, "static_feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(static_raw_cols, f, indent=2)

    print("\n" + "=" * 60)
    print(">> DONE")
    print(f">> Output dir: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
