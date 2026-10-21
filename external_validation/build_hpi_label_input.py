"""
Build the labelling input jsonl (for hpi_clean_label.py) from the MIMIC master
dataset that already has notes merged in: one row per encounter, note text in
`admission_note_text`, keyed by `hadm_id`.

Usage:
  python -m external_validation.build_hpi_label_input MASTER_PATH OUTPUT.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from external_validation.hpi_clean import label_input


def load_master(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=["hadm_id", "admission_note_text"])
    return pd.read_csv(path, usecols=["hadm_id", "admission_note_text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master_path")
    ap.add_argument("output")
    args = ap.parse_args()

    df = load_master(args.master_path)
    n_total = len(df)
    n_no_note = df["admission_note_text"].isna().sum()

    n_written = 0
    n_no_hpi = 0
    with open(args.output, "w") as f:
        for row in df.itertuples(index=False):
            note = getattr(row, "admission_note_text")
            if pd.isna(note):
                continue
            item = label_input(getattr(row, "hadm_id"), note)
            if item is None:
                n_no_hpi += 1
                continue
            f.write(json.dumps(item) + "\n")
            n_written += 1

    print(f"[OK] rows={n_total} no_note={n_no_note} no_hpi_section={n_no_hpi} written={n_written}")
    print(f"     -> {args.output}  (feed this into hpi_clean_label.py)")


if __name__ == "__main__":
    main()
