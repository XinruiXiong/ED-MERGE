#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the MIMIC-IV external-validation cohort in the SAME format Phase 0
writes for UMN (phase0_data/build_event_streams.py), as one extra split named
"external", so the existing Phase 1 tooling runs on it unchanged:

  python phase1_model/precompute_note_embeddings.py --event_dir OUT --splits external ...
  python external_validation/evaluate_external.py --event_dir OUT ...

Differences from the UMN build, all forced by the source data:
  - Truncation boundary is the MIMIC-IV-ED `outtime` (true ED departure).
    MIMIC diagnosis codes carry no timestamp, so there is no dx boundary.
  - Notes: ONE note event at arrival (t=0) = triage chief complaint (free
    text from MIMIC-IV-ED triage, recorded at arrival) + the leakage-
    controlled HPI (hpi_clean.py / hpi_clean_label.py). The discharge
    summary's own chief-complaint field is NOT used -- it is often written
    retrospectively as a diagnosis ("Bacteremia", "Reason for ICU
    admission: ...").
  - Labs come from hosp/labevents, placed at storetime (result
    availability, matching UMN's RESULT_DATETIME), charttime as fallback.
    Joined to stays by subject_id + [intime, outtime] window, because labs
    drawn in the ED often have no hadm_id.
  - Nothing is fit on MIMIC: static and structured inputs are imputed and
    standardized with the UMN scalers (static_scaler.pkl, vitals_scaler.pkl).

Outputs under --output_dir:
  external_static.parquet              one row per stay_id: static features (UMN-scaled) + outcomes
  external_structured_events.parquet   stay_id, t_minutes, STRUCTURED_CHANNELS (UMN-scaled)
  external_notes_events.parquet        stay_id, t_minutes, note_text
  external_outcome_cols.json           the outcomes evaluated externally (EXTERNAL_OUTCOMES)
  external_lab_itemids.json            itemid -> channel mapping actually used (audit)
  external_build_diagnostics.json      cohort / filtering / note counts
"""

import os
import re
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg
from external_validation.hpi_clean import cleaned_text

# Outcomes evaluated externally (12). Hospitalization is constant (every stay
# with a discharge summary was admitted) and 72h revisit is not meaningful
# after admission; bacterial/viral pneumonia have no separate MIMIC code list,
# so pneumonia is scored as all-cause only.
EXTERNAL_OUTCOMES = [
    "outcome_critical", "outcome_sepsis", "outcome_copd_exac", "outcome_acs_mi",
    "outcome_stroke", "outcome_ards", "outcome_aki", "outcome_all_pne",
    "outcome_asthma_exac", "outcome_ahf", "outcome_copd_asthma", "outcome_pe",
]

# MIMIC-IV hosp d_labitems label patterns per channel (case-insensitive,
# full-label match, fluid must contain "blood"). Resolved against the real
# d_labitems at run time and written to external_lab_itemids.json -- check
# that file once; override with --lab_itemid_map if anything is wrong.
LAB_LABEL_PATTERNS = {
    "lab_lactate": r"lactate",
    "lab_wbc": r"white blood cells|wbc count|wbc",
    "lab_creatinine": r"creatinine",
    "lab_bun": r"urea nitrogen",
    "lab_troponin": r"troponin [it]",
    "lab_bnp": r"bnp|b-type natriuretic peptide",
    "lab_ntprobnp": r"nt-?probnp",
    "lab_ddimer": r"d-dimer",
    "lab_glucose": r"glucose",
    "lab_platelet": r"platelet count",
    "lab_sodium": r"sodium|sodium, whole blood",
    "lab_procalcitonin": r"procalcitonin",
    "lab_ph": r"ph",
    "lab_pco2": r"pco2",
    "lab_crp": r"c-reactive protein",
    "lab_inr": r"inr\(pt\)|inr",
}

MIMIC_VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]  # same names as cfg.VITAL_CHANNELS


# =========================================================================
# Readers (csv / csv.gz / parquet), optionally chunked
# =========================================================================
def read_table(path, columns=None):
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def iter_table(path, columns, chunksize=5_000_000):
    if path.endswith(".parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize, columns=columns):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False)


def id_str(s: pd.Series) -> pd.Series:
    """Numeric IDs (possibly float after a csv round-trip with NaN) -> clean string, NaN kept."""
    num = pd.to_numeric(s, errors="coerce").astype("Int64")
    out = num.astype(str).astype(object)
    out[num.isna()] = np.nan
    return out


# =========================================================================
# Cohort
# =========================================================================
def build_cohort(master, a, diag):
    m = master.copy()
    m["stay_id"] = id_str(m[a.stay_col])
    m["subject_id"] = id_str(m[a.subject_col])
    m["hadm_id"] = id_str(m[a.hadm_col])
    m["intime"] = pd.to_datetime(m[a.intime_col], errors="coerce")
    m["outtime"] = pd.to_datetime(m[a.outtime_col], errors="coerce")
    if "age" not in m.columns:
        sys.exit("[FATAL] master has no 'age' column")
    m["age"] = pd.to_numeric(m["age"], errors="coerce")
    diag["n_master_rows"] = int(len(m))

    steps = [
        ("has_ids_and_times", m["stay_id"].notna() & m["subject_id"].notna() & m["intime"].notna() & m["outtime"].notna()),
        ("admitted (hadm_id present)", m["hadm_id"].notna()),
        ("age >= 18", m["age"] >= 18),
        ("ed_los >= 30 min", (m["outtime"] - m["intime"]).dt.total_seconds() >= 30 * 60),
        ("triage vitals recorded", m[cfg.TRIAGE_COLS].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)),
        ("chief complaint recorded", m["__cc__"].fillna("").astype(str).str.strip() != ""),
    ]
    if "__has_note__" in m.columns:
        steps.append(("discharge summary linked", m["__has_note__"]))
    else:
        print(f"[WARN] note column '{a.note_col}' not in master -- discharge-summary linkage filter skipped")

    keep = pd.Series(True, index=m.index)
    diag["cohort_filters"] = {}
    for name, mask in steps:
        keep &= mask.fillna(False)
        diag["cohort_filters"][name] = int(keep.sum())
        print(f"  [cohort] after {name}: {int(keep.sum()):,}")
    m = m[keep].drop_duplicates(subset=["stay_id"]).reset_index(drop=True)
    m["data_boundary_minutes"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 60.0
    return m


# =========================================================================
# Structured stream: ED vitals + hosp labs, forward-filled 22-channel state
# =========================================================================
def load_vitals(path, stays):
    v = read_table(path, ["stay_id", "charttime"] + MIMIC_VITAL_COLS)
    v["stay_id"] = id_str(v["stay_id"])
    v = v[v["stay_id"].isin(set(stays["stay_id"]))].copy()
    v["event_time"] = pd.to_datetime(v["charttime"], errors="coerce")
    for c in MIMIC_VITAL_COLS:
        v[c] = pd.to_numeric(v[c], errors="coerce")  # MIMIC-IV-ED temperature is already deg F, as UMN
    print(f"[OK] ED vitals rows for cohort: {len(v):,}")
    return v.dropna(subset=["event_time"])[["stay_id", "event_time"] + cfg.VITAL_CHANNELS]


def resolve_lab_itemids(d_labitems_path, override_path):
    if override_path:
        with open(override_path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    d = read_table(d_labitems_path)
    d["label_l"] = d["label"].astype(str).str.strip().str.lower()
    d = d[d["fluid"].astype(str).str.lower().str.contains("blood")]
    out = {}
    for ch, pat in LAB_LABEL_PATTERNS.items():
        hit = d[d["label_l"].str.fullmatch(pat)]
        if hit.empty:
            print(f"[WARN] no d_labitems match for {ch} -- channel will be imputed with the UMN train mean")
        for _, r in hit.iterrows():
            out[int(r["itemid"])] = ch
            print(f"  [lab map] {ch:<18} <- {int(r['itemid'])}  {r['label']} | {r['fluid']} | {r.get('category', '')}")
    return out


# Censored results ("<0.01", "LESS THAN 0.01", ">21600*") have an empty valuenum
# in MIMIC-IV, with the text in `comments`. Dropping them discards most negative
# troponins (58% of itemid 51003), leaving mainly elevated values -- so they are
# kept at the detection limit instead.
_CENSORED = re.compile(r"^\s*(?:<|>|less than|greater than)\s*(\d*\.?\d+)", re.I)


# External inputs must be processed exactly as the model's UMN training inputs
# were. UMN lab.parquet carries only a numeric result column: text-valued
# results (censored "<0.010" troponins, all INR and D-dimer) are absent there,
# so censored MIMIC results are dropped too (RECOVER_CENSORED = False) and no
# unit conversion is applied -- the UMN troponin channel mixes ug/L and ng/L
# assays, and MIMIC's ng/mL (= ug/L) matches its conventional-assay part.
# The unit check is expected to flag lab_troponin for that reason.
RECOVER_CENSORED = False
MIMIC_TO_UMN_UNIT_FACTOR = {}


def censored_value(comments: pd.Series) -> pd.Series:
    return pd.to_numeric(comments.astype(str).str.extract(_CENSORED, expand=False), errors="coerce")


def load_labs(path, itemid_map, stays, diag):
    """Lab results whose result time falls inside [intime, outtime] of a cohort stay of the same subject."""
    win = stays[["stay_id", "subject_id", "intime", "outtime"]]
    subjects = set(int(x) for x in win["subject_id"])
    parts = []
    for chunk in tqdm(iter_table(path, ["subject_id", "itemid", "charttime", "storetime", "valuenum", "comments"]),
                      desc="labevents"):
        chunk = chunk[chunk["itemid"].isin(itemid_map.keys()) & chunk["subject_id"].isin(subjects)]
        if chunk.empty:
            continue
        chunk = chunk.assign(subject_id=id_str(chunk["subject_id"]))
        chunk["event_time"] = pd.to_datetime(chunk["storetime"], errors="coerce").fillna(
            pd.to_datetime(chunk["charttime"], errors="coerce"))
        chunk["value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk["censored"] = chunk["value"].isna()
        if RECOVER_CENSORED:
            chunk.loc[chunk["censored"], "value"] = censored_value(chunk.loc[chunk["censored"], "comments"])
        chunk["censored"] &= chunk["value"].notna()
        chunk["channel"] = chunk["itemid"].map(itemid_map)
        chunk["value"] *= chunk["channel"].map(MIMIC_TO_UMN_UNIT_FACTOR).fillna(1.0)
        chunk = chunk.dropna(subset=["event_time", "value"]).merge(win, on="subject_id", how="inner")
        chunk = chunk[(chunk["event_time"] >= chunk["intime"]) & (chunk["event_time"] <= chunk["outtime"])]
        parts.append(chunk[["stay_id", "event_time", "channel", "value", "censored"]])
    if not parts:
        return pd.DataFrame(columns=["stay_id", "event_time"] + cfg.LAB_CHANNELS)
    l = pd.concat(parts, ignore_index=True)
    print(f"[OK] lab results inside ED windows: {len(l):,} (censored, kept at detection limit: {int(l['censored'].sum()):,})")
    diag["lab_censored_frac"] = {ch: float(g["censored"].mean()) for ch, g in l.groupby("channel")}
    diag["lab_raw_mean"] = {ch: float(g["value"].mean()) for ch, g in l.groupby("channel")}
    wide = l.pivot_table(index=["stay_id", "event_time"], columns="channel", values="value", aggfunc="last").reset_index()
    for ch in cfg.LAB_CHANNELS:
        if ch not in wide.columns:
            wide[ch] = np.nan
    return wide[["stay_id", "event_time"] + cfg.LAB_CHANNELS]


def build_structured_stream(stays, vitals, labs):
    """Same construction as build_event_streams.build_structured_event_stream with
    origin = intime: triage seed at t=0, events in [intime, outtime], collapse
    same-timestamp rows, forward fill."""
    ev = pd.concat([vitals, labs], ignore_index=True, sort=False)
    for ch in cfg.STRUCTURED_CHANNELS:
        if ch not in ev.columns:
            ev[ch] = np.nan
    ev = ev.merge(stays[["stay_id", "intime", "data_boundary_minutes"]], on="stay_id", how="inner")
    ev = ev[ev["event_time"] >= ev["intime"]]
    ev["t_minutes"] = (ev["event_time"] - ev["intime"]).dt.total_seconds() / 60.0
    ev = ev[ev["t_minutes"] <= ev["data_boundary_minutes"]][["stay_id", "t_minutes"] + cfg.STRUCTURED_CHANNELS]

    seed = stays[["stay_id"] + cfg.TRIAGE_COLS].copy()
    seed = seed.rename(columns={f"triage_{ch}": ch for ch in cfg.VITAL_CHANNELS})
    for ch in cfg.STRUCTURED_CHANNELS:
        seed[ch] = pd.to_numeric(seed[ch], errors="coerce") if ch in seed.columns else np.nan
    seed["t_minutes"] = 0.0
    seed = seed[seed[cfg.VITAL_CHANNELS].notna().any(axis=1)][["stay_id", "t_minutes"] + cfg.STRUCTURED_CHANNELS]

    allev = pd.concat([seed, ev], ignore_index=True, sort=False)
    allev = allev.groupby(["stay_id", "t_minutes"], as_index=False)[cfg.STRUCTURED_CHANNELS].last()
    allev = allev.sort_values(["stay_id", "t_minutes"])
    allev[cfg.STRUCTURED_CHANNELS] = allev.groupby("stay_id")[cfg.STRUCTURED_CHANNELS].ffill()
    return allev.reset_index(drop=True)


# =========================================================================
# Note event: triage chief complaint + cleaned HPI, at arrival
# =========================================================================
def load_cleaned_hpi(labels_path, diag):
    """hadm_id -> cleaned HPI. Recomputed from the stored sentence labels with the
    CURRENT rule layer (cleaned_text applies rule_veto), not read from the stored
    'cleaned' field. Records whose model output did not parse are unsafe -> no HPI."""
    out, n_lines, n_unparsed = {}, 0, 0
    with open(labels_path) as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            r = json.loads(line)
            if not r.get("labels"):
                n_unparsed += 1
                continue
            out[str(int(r["hadm_id"]))] = cleaned_text(r["sentences"], r["labels"])
    diag["hpi_label_records"] = n_lines
    diag["hpi_unparsed_treated_unsafe"] = n_unparsed
    diag["hpi_empty_after_cleaning"] = int(sum(1 for t in out.values() if not t.strip()))
    print(f"[OK] HPI labels: {n_lines:,} records, {n_unparsed:,} unparsed (dropped as unsafe), "
          f"{diag['hpi_empty_after_cleaning']:,} empty after cleaning")
    return out


def build_note_events(stays, hpi_by_hadm, diag):
    rows = []
    n_hpi = 0
    for stay_id, hadm_id, cc in zip(stays["stay_id"], stays["hadm_id"], stays["__cc__"].fillna("")):
        cc = re.sub(r"\s+", " ", str(cc)).strip()
        hpi = (hpi_by_hadm.get(hadm_id) or "").strip()
        n_hpi += bool(hpi)
        parts = [f"Chief complaint: {cc}"] if cc else []
        if hpi:
            parts.append(f"History of present illness: {hpi}")
        if parts:
            rows.append({"stay_id": stay_id, "t_minutes": 0.0, "note_text": "\n".join(parts)})
    diag["n_note_events"] = len(rows)
    diag["n_stays_with_cleaned_hpi"] = n_hpi
    print(f"[OK] note events: {len(rows):,} (with non-empty cleaned HPI: {n_hpi:,})")
    return pd.DataFrame(rows, columns=["stay_id", "t_minutes", "note_text"])


# =========================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Build the MIMIC-IV external-validation event streams (split 'external').")
    ap.add_argument("--master_path", required=True,
                     help="MIMIC master dataset (csv or parquet): one row per ED stay with triage_*, cci_*/eci_*/"
                          "chiefcom_*, age, outcome_* columns.")
    ap.add_argument("--stay_col", default="stay_id")
    ap.add_argument("--subject_col", default="subject_id")
    ap.add_argument("--hadm_col", default="hadm_id")
    ap.add_argument("--intime_col", default="intime")
    ap.add_argument("--outtime_col", default="outtime", help="MIMIC-IV-ED edstays.outtime (ED departure).")
    ap.add_argument("--cc_col", default="chiefcomplaint", help="Triage free-text chief complaint in master.")
    ap.add_argument("--ed_triage_path", default=None,
                     help="MIMIC-IV-ED triage table; only needed if --cc_col is not in master.")
    ap.add_argument("--note_col", default="admission_note_text",
                     help="Discharge-summary text column in master; only its presence is used (linkage filter).")
    ap.add_argument("--triage_acuity_col", default="triage_acuity")

    ap.add_argument("--hpi_labels", required=True, help="OUTPUT.jsonl of hpi_clean_label.py")
    ap.add_argument("--ed_vitalsign_path", required=True, help="MIMIC-IV-ED vitalsign table")
    ap.add_argument("--labevents_path", required=True, help="MIMIC-IV hosp labevents table")
    ap.add_argument("--d_labitems_path", required=True, help="MIMIC-IV hosp d_labitems table")
    ap.add_argument("--lab_itemid_map", default=None,
                     help="Optional JSON {itemid: channel} that replaces the label-pattern resolution.")
    ap.add_argument("--umn_event_dir", required=True,
                     help="UMN Phase 0 output dir (static_scaler.pkl, vitals_scaler.pkl).")
    ap.add_argument("--output_dir", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    diag = {}

    # ---------------- 1) master + cohort ----------------
    print(">> [1/6] Loading master and building cohort...")
    master = read_table(a.master_path)
    master.columns = [str(c).strip() for c in master.columns]
    if a.note_col in master.columns:
        master["__has_note__"] = master[a.note_col].notna() & (master[a.note_col].astype(str).str.strip() != "")
        master = master.drop(columns=[a.note_col])
    if a.cc_col in master.columns:
        master["__cc__"] = master[a.cc_col]
    elif a.ed_triage_path:
        tri = read_table(a.ed_triage_path, ["stay_id", "chiefcomplaint"])
        tri["stay_id"] = id_str(tri["stay_id"])
        master["__cc__"] = id_str(master[a.stay_col]).map(tri.set_index("stay_id")["chiefcomplaint"])
    else:
        sys.exit(f"[FATAL] '{a.cc_col}' not in master and no --ed_triage_path given")
    missing_triage = [c for c in cfg.TRIAGE_COLS if c not in master.columns]
    if missing_triage:
        sys.exit(f"[FATAL] master missing triage cols: {missing_triage}")
    missing_out = [c for c in EXTERNAL_OUTCOMES if c not in master.columns]
    if missing_out:
        sys.exit(f"[FATAL] master missing external outcome cols: {missing_out}")

    stays = build_cohort(master, a, diag)
    del master

    # ---------------- 2) static features, UMN-scaled ----------------
    print(">> [2/6] Static features (UMN static_scaler)...")
    with open(os.path.join(a.umn_event_dir, "static_scaler.pkl"), "rb") as f:
        s = pickle.load(f)
    static_scaler, static_cols = s["scaler"], s["columns"]
    absent = [c for c in static_cols if c not in stays.columns]
    if absent:
        print(f"[WARN] {len(absent)} UMN static cols absent in MIMIC, set to 0: {absent}")
    X = pd.DataFrame({c: pd.to_numeric(stays[c], errors="coerce") if c in stays.columns else 0.0
                      for c in static_cols}, index=stays.index)
    for i, c in enumerate(static_cols):
        fill = static_scaler.mean_[i] if c == "age" else 0.0  # UMN train mean, never a MIMIC statistic
        X[c] = X[c].fillna(fill)
    X = pd.DataFrame(static_scaler.transform(X.values), columns=static_cols, index=stays.index)

    # Model was trained on all 16 outcomes; the 4 not evaluated externally are
    # 0 placeholders so the Dataset can index ckpt["outcome_cols"]. They are
    # never scored (evaluate_external.py reads external_outcome_cols.json).
    outcomes = pd.DataFrame(index=stays.index)
    for c in cfg.OUTCOME_COLS_LIST:
        outcomes[c] = pd.to_numeric(stays[c], errors="coerce").fillna(0).astype(float) if c in EXTERNAL_OUTCOMES else 0.0
    meta = stays[["stay_id", "subject_id", "hadm_id", "data_boundary_minutes"]].copy()
    if a.triage_acuity_col in stays.columns:
        meta["triage_acuity"] = stays[a.triage_acuity_col]
    static_out = pd.concat([meta, outcomes, X], axis=1)

    # ---------------- 3) structured stream ----------------
    print(">> [3/6] ED vitals + hosp labs...")
    vitals = load_vitals(a.ed_vitalsign_path, stays)
    itemid_map = resolve_lab_itemids(a.d_labitems_path, a.lab_itemid_map)
    with open(os.path.join(a.output_dir, "external_lab_itemids.json"), "w") as f:
        json.dump({str(k): v for k, v in sorted(itemid_map.items())}, f, indent=2)
    labs = load_labs(a.labevents_path, itemid_map, stays, diag)
    structured = build_structured_stream(stays, vitals, labs)

    print(">> [4/6] Scaling structured channels with UMN vitals_scaler...")
    with open(os.path.join(a.umn_event_dir, "vitals_scaler.pkl"), "rb") as f:
        v = pickle.load(f)
    assert list(v["columns"]) == cfg.STRUCTURED_CHANNELS, "UMN vitals_scaler channel order differs from config"
    # Channels never observed in the UMN training split were constant 0 there
    # (build_event_streams.py falls back to mean 0, scale 1), so the model never
    # learned weights for real values of them -- feeding MIMIC's raw values
    # (e.g. D-dimer ~1800) would be far out of range. Keep them unobserved here
    # too, exactly as the model saw them in training.
    sc = v["scaler"]
    umn_unobserved = [ch for i, ch in enumerate(cfg.STRUCTURED_CHANNELS)
                      if sc.mean_[i] == 0 and sc.scale_[i] == 1]
    if umn_unobserved:
        print(f"[WARN] channels never observed in UMN train, set unobserved for MIMIC too: {umn_unobserved}")
        structured[umn_unobserved] = np.nan
    diag["channels_unobserved_in_umn_train"] = umn_unobserved

    # Unit check (after MIMIC_TO_UMN_UNIT_FACTOR): a MIMIC/UMN mean ratio off by
    # an order of magnitude almost always means different units.
    diag["lab_unit_check_mimic_over_umn_mean"] = {}
    for ch, m_mean in diag.get("lab_raw_mean", {}).items():
        if ch in umn_unobserved:
            continue
        u_mean = float(sc.mean_[cfg.STRUCTURED_CHANNELS.index(ch)])
        ratio = m_mean / u_mean if u_mean else float("nan")
        diag["lab_unit_check_mimic_over_umn_mean"][ch] = ratio
        flag = "  <-- likely UNIT MISMATCH" if not (0.1 <= ratio <= 10) else ""
        print(f"  [unit check] {ch:<18} MIMIC mean {m_mean:>10.4g}  UMN mean {u_mean:>10.4g}  ratio {ratio:.3g}{flag}")
    observed = structured[cfg.STRUCTURED_CHANNELS].notna().groupby(structured["stay_id"]).any().mean()
    diag["structured_channel_observed_frac"] = {ch: float(observed[ch]) for ch in cfg.STRUCTURED_CHANNELS}
    structured[cfg.STRUCTURED_CHANNELS] = structured[cfg.STRUCTURED_CHANNELS].fillna(pd.Series(v["train_means"]))
    structured[cfg.STRUCTURED_CHANNELS] = v["scaler"].transform(structured[cfg.STRUCTURED_CHANNELS].values)

    # ---------------- 5) note event ----------------
    print(">> [5/6] Note events (triage chief complaint + cleaned HPI)...")
    hpi = load_cleaned_hpi(a.hpi_labels, diag)
    notes = build_note_events(stays, hpi, diag)

    # ---------------- 6) save ----------------
    print(">> [6/6] Saving...")
    static_out.to_parquet(os.path.join(a.output_dir, "external_static.parquet"), index=False)
    structured.to_parquet(os.path.join(a.output_dir, "external_structured_events.parquet"), index=False)
    notes.to_parquet(os.path.join(a.output_dir, "external_notes_events.parquet"), index=False)
    with open(os.path.join(a.output_dir, "external_outcome_cols.json"), "w") as f:
        json.dump(EXTERNAL_OUTCOMES, f, indent=2)
    diag["n_stays"] = int(len(static_out))
    diag["n_structured_events"] = int(len(structured))
    diag["outcome_prevalence"] = {c: float(static_out[c].mean()) for c in EXTERNAL_OUTCOMES}
    with open(os.path.join(a.output_dir, "external_build_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print(json.dumps({k: diag[k] for k in ("n_stays", "n_structured_events", "n_note_events",
                                           "n_stays_with_cleaned_hpi", "outcome_prevalence")}, indent=2))
    print(f">> DONE -> {a.output_dir}")


if __name__ == "__main__":
    main()
