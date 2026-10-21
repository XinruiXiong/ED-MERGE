#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case-level risk trajectories for single test-set encounters (Results case figure).

Two modes:
  --mode candidates   rank true-positive encounters of --outcome that start
                      below threshold at arrival and cross it later, from the
                      existing landmark_encounter_detail.parquet (no model run).
                      Prints a de-identified summary (case rank, no stay_id);
                      the stay_id mapping is written to case_candidates_<outcome>.csv.
  --mode trace        one encounter (--case_rank from that csv, or --stay_id):
                        1. risk after every event (step trajectory) and the
                           change each event causes -- exact, the model is
                           simply re-queried with that event included/excluded
                        2. modality contributions on a 10-min grid:
                           p_full - p_without_<component> (same inference-time
                           removal as evaluate_ablation.py)
                      Notes are shown by time / type / author role only; note
                      text is never read.

Output under --out_dir:
  case_candidates_<outcome>.csv
  case_<outcome>_<rank>_events.csv        per-event risk
  case_<outcome>_<rank>_modality.csv      10-min grid, per-config risk
"""

import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config as cfg
from common.event_utils import EncounterEventIndex, NoteEmbeddingIndex, landmark_upper_bound
from sequence_dataset import EventSequenceDataset, collate_sequence_batch
from sequence_model import EventSequenceModel

SHOW_LANDMARKS = [0, 30, 60, 120, 240, 360]
MODALITY_CONFIGS = {
    "full": (),
    "without_text": ("text",),
    "without_structured": ("vital", "lab"),
    "without_static": ("static",),
}
SHORT = {"temperature": "Temp", "heartrate": "HR", "resprate": "RR", "o2sat": "SpO2", "sbp": "SBP", "dbp": "DBP"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["candidates", "trace"])
    ap.add_argument("--outcome", required=True, help="e.g. sepsis, critical (outcome_ prefix optional)")
    ap.add_argument("--event_dir", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test")
    # candidates
    ap.add_argument("--detail", help="landmark_encounter_detail.parquet (candidates mode)")
    ap.add_argument("--min_boundary", type=float, default=120, help="min observation boundary, minutes")
    ap.add_argument("--min_notes", type=int, default=2)
    ap.add_argument("--top", type=int, default=15)
    # trace
    ap.add_argument("--embed_cache_dir", help="note embedding cache (trace mode)")
    ap.add_argument("--case_rank", type=int)
    ap.add_argument("--stay_id")
    ap.add_argument("--grid_step", type=float, default=10)
    return ap.parse_args()


def load_scalers(event_dir):
    with open(os.path.join(event_dir, "static_scaler.pkl"), "rb") as f:
        s = pickle.load(f)
    with open(os.path.join(event_dir, "vitals_scaler.pkl"), "rb") as f:
        v = pickle.load(f)
    return s, v


def describe_static(row, static_scaler):
    sc, cols = static_scaler["scaler"], static_scaler["columns"]
    raw = row[cols].to_numpy(dtype=np.float64) * sc.scale_ + sc.mean_
    age = raw[cols.index("age")] if "age" in cols else np.nan
    on = [c for c, x in zip(cols, raw) if c != "age" and x > 0.5]
    cc = [c.replace("chiefcom_", "") for c in on if c.startswith("chiefcom_")]
    com = [c for c in on if not c.startswith("chiefcom_")]
    return age, cc, com


def positives(row, outcome_cols):
    return [c.replace("outcome_", "") for c in outcome_cols if row[c] == 1]


# --------------------------------------------------------------------------- candidates

def run_candidates(a, col, thr, outcome_cols, static_df, static_scaler):
    det = pd.read_parquet(a.detail, columns=["landmark_minutes", "stay_id", "n_notes_seen", "n_vitals_seen",
                                             f"{col}_prob", f"{col}_label"])
    det = det[det[f"{col}_label"] == 1]
    det["stay_id"] = det["stay_id"].astype(str)
    p = det.pivot(index="stay_id", columns="landmark_minutes", values=f"{col}_prob")
    p.columns = [int(round(c)) for c in p.columns]
    last = det.sort_values("landmark_minutes").groupby("stay_id").last()

    st = static_df.set_index(static_df["stay_id"].astype(str))
    cand = pd.DataFrame({
        "p0": p[0], "p_last": p[max(p.columns)],
        "n_notes": last["n_notes_seen"], "n_struct": last["n_vitals_seen"],
    })
    cand["boundary"] = st.loc[cand.index, "data_boundary_minutes"]
    cand = cand[(cand["p0"] < thr) & (cand["p_last"] >= thr)
                & (cand["boundary"].fillna(np.inf) >= a.min_boundary) & (cand["n_notes"] >= a.min_notes)]
    # first grid landmark at which risk crosses the threshold
    grid = sorted(p.columns)
    cand["cross_min"] = [next((T for T in grid if p.at[s, T] >= thr), np.nan) for s in cand.index]
    cand["rise"] = cand["p_last"] - cand["p0"]
    cand = cand.sort_values("rise", ascending=False).head(a.top).reset_index().rename(columns={"index": "stay_id"})
    cand.insert(0, "case_rank", np.arange(1, len(cand) + 1))
    cand.to_csv(os.path.join(a.out_dir, f"case_candidates_{col.replace('outcome_', '')}.csv"), index=False)

    print(f"\n#### candidates: {col}  threshold={thr:.3f}  (true positives, p0<thr, crosses later)")
    for _, r in cand.iterrows():
        row = st.loc[r["stay_id"]]
        age, cc, _ = describe_static(row, static_scaler)
        traj = "  ".join(f"{T}:{p.at[r['stay_id'], T]:.2f}" for T in SHOW_LANDMARKS if T in p.columns)
        b = "none" if pd.isna(r["boundary"]) else f"{r['boundary']:.0f}"
        print(f"[{int(r['case_rank']):>2}] age {age:.0f}  ESI {row.get('triage_acuity', 'NA')}  boundary {b} min  "
              f"notes {int(r['n_notes'])}  struct {int(r['n_struct'])}  cross@{r['cross_min']:.0f}  | {traj}")
        print(f"      cc: {', '.join(cc) or '-'}   positive: {', '.join(positives(row, outcome_cols))}")


# --------------------------------------------------------------------------- trace

class _OneStayNotes(NoteEmbeddingIndex):
    """Embeddings for one stay only, read from the memory-mapped cache."""

    @classmethod
    def for_stay(cls, cache_dir, split, stay_id):
        idx = pd.read_parquet(os.path.join(cache_dir, f"{split}_note_embeddings_index.parquet"))
        idx = idx[idx["stay_id"].astype(str) == stay_id].sort_values("t_minutes").reset_index(drop=True)
        emb = np.load(os.path.join(cache_dir, f"{split}_note_embeddings.npy"), mmap_mode="r")
        sub = np.asarray(emb[idx["row_idx"].to_numpy()], dtype=np.float32).reshape(len(idx), emb.shape[1])
        meta = idx.drop(columns=["row_idx"]).copy()
        idx["row_idx"] = np.arange(len(idx))
        return cls(sub, idx), meta


def predict(model, device, rows, structured, notes, outcome_cols, static_cols, max_lm, times, drop):
    items = []
    for T in times:
        ds = EventSequenceDataset(rows, structured, notes, outcome_cols, static_cols, max_landmark_minutes=max_lm,
                                  seed=0, resample_each_epoch=False, fixed_landmark_minutes=T, drop_modalities=drop)
        items.append(ds[0])
    out = []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            b = collate_sequence_batch(items[i:i + 64])
            logits = model(b["content"].to(device), b["modality"].to(device), b["delta_t"].to(device),
                           b["lengths"], b["static"].to(device))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def run_trace(a, col, thr, outcome_cols, static_cols, static_df, static_scaler, vitals_scaler, model, device, max_lm):
    name = col.replace("outcome_", "")
    if a.stay_id:
        stay_id, tag = str(a.stay_id), "manual"
    else:
        cand = pd.read_csv(os.path.join(a.out_dir, f"case_candidates_{name}.csv"), dtype={"stay_id": str})
        stay_id, tag = cand.loc[cand["case_rank"] == a.case_rank, "stay_id"].iloc[0], str(a.case_rank)
    j = outcome_cols.index(col)

    rows = static_df[static_df["stay_id"].astype(str) == stay_id].reset_index(drop=True)
    row = rows.iloc[0]
    sdf = pd.read_parquet(os.path.join(a.event_dir, f"{a.split}_structured_events.parquet"))
    sdf = sdf[sdf["stay_id"].astype(str) == stay_id]
    structured = EncounterEventIndex(rows, sdf, pd.DataFrame(columns=["stay_id", "t_minutes", "note_text"]))
    notes, note_meta = _OneStayNotes.for_stay(a.embed_cache_dir, a.split, stay_id)

    upper = landmark_upper_bound(row["data_boundary_minutes"], max_lm)
    # ---- per-event step trajectory
    s_times, s_states = structured.vitals_events_upto(stay_id, upper)
    raw = s_states * vitals_scaler["scaler"].scale_ + vitals_scaler["scaler"].mean_
    ev = []
    for k, t in enumerate(s_times):
        changed = np.arange(len(cfg.STRUCTURED_CHANNELS)) if k == 0 else np.flatnonzero(s_states[k] != s_states[k - 1])
        parts = []
        for c in changed:
            ch = cfg.STRUCTURED_CHANNELS[c]
            if k == 0 and ch in cfg.LAB_CHANNELS:
                continue  # unobserved labs sit at the train mean at arrival
            parts.append(f"{SHORT.get(ch, ch.replace('lab_', ''))} {raw[k, c]:.4g}")
        has_lab = any(cfg.STRUCTURED_CHANNELS[c] in cfg.LAB_CHANNELS for c in changed)
        has_vital = any(cfg.STRUCTURED_CHANNELS[c] in cfg.VITAL_CHANNELS for c in changed)
        kind = "triage" if k == 0 else {(True, True): "vital+lab", (True, False): "lab"}.get((has_lab, has_vital), "vital")
        ev.append({"t_min": t, "type": kind, "detail": ", ".join(parts)})
    for _, n in note_meta[note_meta["t_minutes"] <= upper].iterrows():
        ev.append({"t_min": n["t_minutes"], "type": "note",
                   "detail": " / ".join(str(n[c]) for c in ("tos", "role") if c in note_meta.columns)})
    ev = pd.DataFrame(ev).sort_values("t_min", kind="stable").reset_index(drop=True)

    base = predict(model, device, rows, structured, notes, outcome_cols, static_cols, max_lm,
                   [0.0], ("text", "vital", "lab"))[0, j]
    ut = np.unique(ev["t_min"].to_numpy())
    p_at = dict(zip(ut, predict(model, device, rows, structured, notes, outcome_cols, static_cols, max_lm, ut, ())[:, j]))
    ev["p_after"] = ev["t_min"].map(p_at)
    prev = np.r_[base, [p_at[t] for t in ut[:-1]]]
    ev["delta"] = ev["t_min"].map(dict(zip(ut, [p_at[t] - q for t, q in zip(ut, prev)])))
    # several events at the same minute share one step
    ev.loc[ev.duplicated("t_min", keep="first"), "delta"] = np.nan
    ev.to_csv(os.path.join(a.out_dir, f"case_{name}_{tag}_events.csv"), index=False)

    # ---- modality contributions on a grid
    grid = np.unique(np.r_[np.arange(0, upper + 1e-9, a.grid_step), upper])
    mod = pd.DataFrame({"t_min": grid})
    for cname, drop in MODALITY_CONFIGS.items():
        mod[cname] = predict(model, device, rows, structured, notes, outcome_cols, static_cols, max_lm, grid, drop)[:, j]
    for cname in list(MODALITY_CONFIGS)[1:]:
        mod["contrib_" + cname.replace("without_", "")] = mod["full"] - mod[cname]
    mod.to_csv(os.path.join(a.out_dir, f"case_{name}_{tag}_modality.csv"), index=False)

    # ---- print
    age, cc, com = describe_static(row, static_scaler)
    pd.set_option("display.max_rows", 500)
    print(f"\n#### case {name} #{tag}   threshold={thr:.3f}   observation boundary={upper:.0f} min")
    print(f"age {age:.0f}  ESI {row.get('triage_acuity', 'NA')}  cc: {', '.join(cc) or '-'}")
    print(f"comorbidity flags: {', '.join(com) or '-'}")
    print(f"positive outcomes: {', '.join(positives(row, outcome_cols))}")
    print(f"static-only risk (before any event): {base:.3f}")
    print("\n## events (p_after = risk once this event is seen; delta = change from the previous step)")
    print(f"{'t_min':>7}  {'type':<9} {'p_after':>7} {'delta':>7}  detail")
    for _, e in ev.iterrows():
        d = "" if pd.isna(e["delta"]) else f"{e['delta']:+.3f}"
        print(f"{e['t_min']:7.1f}  {e['type']:<9} {e['p_after']:7.3f} {d:>7}  {e['detail']}")
    print("\n## modality contributions (contrib_X = p_full - p_without_X)")
    cols = ["t_min", "full"] + [c for c in mod.columns if c.startswith("contrib_")]
    print(mod[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt_path, map_location=device)
    outcome_cols, static_cols = ckpt["outcome_cols"], ckpt["static_cols"]
    col = a.outcome if a.outcome.startswith("outcome_") else f"outcome_{a.outcome}"
    thr = ckpt.get("thresholds", {}).get(col, 0.5)
    static_df = pd.read_parquet(os.path.join(a.event_dir, f"{a.split}_static.parquet"))
    static_scaler, vitals_scaler = load_scalers(a.event_dir)

    if a.mode == "candidates":
        run_candidates(a, col, thr, outcome_cols, static_df, static_scaler)
        return

    model = EventSequenceModel(num_static_features=len(static_cols), num_labels=len(outcome_cols),
                               note_hidden=ckpt.get("note_hidden", 768)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    max_lm = ckpt.get("max_landmark_minutes", cfg.MAX_LANDMARK_MINUTES_DEFAULT)
    run_trace(a, col, thr, outcome_cols, static_cols, static_df, static_scaler, vitals_scaler, model, device, max_lm)


if __name__ == "__main__":
    main()
