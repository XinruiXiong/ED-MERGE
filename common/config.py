# -*- coding: utf-8 -*-
"""
Shared constants for ED-MERGE.

All *_PLACEHOLDER paths below are placeholders for site-specific data
locations. Every script also exposes the same values as CLI arguments, so you
do not have to edit this file -- the defaults here document what each path
is *for*.
"""

# =========================================================================
# Placeholder paths -- override via CLI args in each script, do not rely on
# these defaults for anything real.
# =========================================================================
MASTER_PARQUET_PLACEHOLDER = "<PLACEHOLDER_MASTER_PARQUET_PATH>"
NOTES_DIR_PLACEHOLDER = "<PLACEHOLDER_NOTES_PARQUET_DIR>"
VITALS_PARQUET_PLACEHOLDER = "<PLACEHOLDER_VITALS_PARQUET_PATH>"
LAB_PARQUET_PLACEHOLDER = "<PLACEHOLDER_LAB_PARQUET_PATH>"
BERT_DIR_PLACEHOLDER = "<PLACEHOLDER_BIOCLINICALBERT_DIR>"
EVENT_STREAM_OUTPUT_DIR_PLACEHOLDER = "<PLACEHOLDER_EVENT_STREAM_OUTPUT_DIR>"
STAGE1_BERT_OUTPUT_DIR_PLACEHOLDER = "<PLACEHOLDER_STAGE1_BERT_OUTPUT_DIR>"
NOTE_EMBEDDING_CACHE_DIR_PLACEHOLDER = "<PLACEHOLDER_NOTE_EMBEDDING_CACHE_DIR>"
SEQUENCE_MODEL_OUTPUT_DIR_PLACEHOLDER = "<PLACEHOLDER_SEQUENCE_MODEL_OUTPUT_DIR>"

# =========================================================================
# Source-table column names (UMN master table).
# =========================================================================
MASTER_INTIME_COL = "intime"
MASTER_SUBJECT_COL = "subject_id"
MASTER_STAY_COL = "stay_id"
MASTER_DX_TIME_COL = "dx_time"

# ED-departure / admission columns used for the truncation boundary.
# `outtime` is NOT used: its null rate does not track hospitalization status
# the way a real ED-departure signal should. The landmark origin (t=0) is
# always `intime`; `edregtime` is not used because it can fall well after
# arrival, which would let T=0 include data recorded in that gap.
MASTER_EDOUTTIME_COL = "edouttime"    # true ED-departure signal (sparse for hospitalized patients)
MASTER_ADMITTIME_COL = "admittime"    # hospital admission time (complementary coverage to edouttime)

# ESI triage acuity -- carried through to {split}_static.parquet as an
# AUDIT/METADATA column only (not added to static_raw_cols, not scaled, not
# fed to the model). Used downstream for confound-stratified analysis (sicker
# patients get more/faster documentation).
MASTER_TRIAGE_ACUITY_COL = "triage_acuity"

NOTES_STAY_COL_DEFAULT = "SERVICE_ID"
NOTES_TIME_COL_DEFAULT = "FILING_DATE"
NOTES_TEXT_COL_DEFAULT = "NOTE_TEXT"
# TOS (note type, e.g. "9.k.1. Progress Note") and ROLE (author's clinical
# role, e.g. "11. Physician") -- carried through notes_events.parquet and the
# note-embedding index as AUDIT metadata only, never fed to the model. This
# keeps the production model blind to note type/role so a sensitivity
# analysis (excluding diagnostic-note types, comparing physician-authored vs.
# nursing-authored performance) can filter the SAME cached embeddings
# post-hoc without retraining. Missing from the source -> warning, not
# fatal, same pattern as triage_acuity.
NOTES_TOS_COL_DEFAULT = "TOS"
NOTES_ROLE_COL_DEFAULT = "ROLE"

# =========================================================================
# Vitals source: SVC_BASIC_VITAL export, a wide table (one row per
# collection event, all channels as columns).
# =========================================================================
VITALS_NEW_SERVICE_ID_COL = "SERVICE_ID"
VITALS_NEW_TIME_COL = "COLLECTION_DATETIME"
VITALS_NEW_TEMP_UNIT_COL = "BODY_TEMP_UNIT"
# source column -> internal channel name
VITALS_NEW_COLUMN_MAP = {
    "BLOOD_PRESSURE_SYSTOLIC": "sbp",
    "BLOOD_PRESSURE_DIASTOLIC": "dbp",
    "BODY_TEMPERATURE": "temperature",
    "PULSE": "heartrate",
    "PULSE_OXYMETRY": "o2sat",
    "RESPIRATORY_RATE": "resprate",
}
# Fixed channel order -- every vital "state vector" event contributes to a
# vector in exactly this order. Keep this order stable across Phase 0 / Phase
# 1 code. Also used for the triage_{channel} seed columns in master_parquet.
VITAL_CHANNELS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
TRIAGE_COLS = [f"triage_{c}" for c in VITAL_CHANNELS]

# =========================================================================
# Lab source: lab.parquet, extracted upstream to a fixed 16-category LOINC
# panel. The LAB column is the canonical name per category, already folding
# multiple LOINC variants (e.g. Troponin = cTnI serum/whole blood + cTnT
# whole blood + hs-cTnI + hs-cTnT) into one scalar. No further LOINC
# filtering needed in this pipeline.
# =========================================================================
LAB_SERVICE_ID_COL = "SERVICE_ID"
LAB_TIME_COL = "RESULT_DATETIME"
LAB_NAME_COL = "LAB"
LAB_VALUE_COL = "RESULT_VALUE_NUM"

# LAB column value -> internal channel name. Fixed order.
LAB_NAME_TO_CHANNEL = {
    "Lactate": "lab_lactate",
    "WBC": "lab_wbc",
    "Creatinine": "lab_creatinine",
    "BUN": "lab_bun",
    "Troponin": "lab_troponin",
    "BNP": "lab_bnp",
    "NT-proBNP": "lab_ntprobnp",
    "D-dimer": "lab_ddimer",
    "Glucose": "lab_glucose",
    "Platelet": "lab_platelet",
    "Sodium": "lab_sodium",
    "Procalcitonin": "lab_procalcitonin",
    "pH": "lab_ph",
    "pCO2": "lab_pco2",
    "CRP": "lab_crp",
    "INR": "lab_inr",
}
LAB_CHANNELS = list(LAB_NAME_TO_CHANNEL.values())

# The actual state-vector channel order used throughout Phase 0 / Phase 1 --
# vitals first (triage-seeded at intime), then labs (no seed; NaN until the
# first real result, forward-filled from there same as any other channel
# with no seed value). Labs are NOT a separate event-stream modality, they
# widen this one state vector.
# Keep this order stable; sequence_model.py's structured-event projection
# layer width is derived from len(STRUCTURED_CHANNELS).
STRUCTURED_CHANNELS = VITAL_CHANNELS + LAB_CHANNELS

# The 16 outcomes (pneumonia as bacterial, viral, and all-cause).
OUTCOME_COLS_LIST = [
    "outcome_ed_revisit_3d", "outcome_hospitalization", "outcome_critical",
    "outcome_sepsis", "outcome_copd_exac", "outcome_acs_mi", "outcome_stroke",
    "outcome_ards", "outcome_aki",
    "outcome_bac_pne", "outcome_viral_pne", "outcome_all_pne",
    "outcome_asthma_exac", "outcome_ahf", "outcome_copd_asthma", "outcome_pe",
]

# =========================================================================
# Temporal / landmark sampling config
# =========================================================================
# Evaluation grid (0..6h); every point is scored by the same checkpoint.
LANDMARK_EVAL_GRID_MINUTES = [0, 10, 20, 30, 60, 90, 120, 180, 240, 300, 360]
MAX_LANDMARK_MINUTES_DEFAULT = 360

# =========================================================================
# Split config
# =========================================================================
DEFAULT_TRAIN_YEAR_LE = 2022
DEFAULT_TEST_YEAR_GE = 2023
DEFAULT_VAL_RATIO_WITHIN_DEV_POOL = 0.1
DEFAULT_SEED = 42
