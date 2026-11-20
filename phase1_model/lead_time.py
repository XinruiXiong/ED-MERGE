#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lead time of the model alert over clinical reference events
(sepsis by default; ACS and PE with --outcomes).

Alert time: the first event (minutes from arrival, within the observation
window) after which the model's risk is >= the validation threshold. The
model is re-queried after every event, so the alert time is exact.

Reference times (minutes from arrival, first non-canceled order, 0-24 h),
all from the order table (SVC_ORDER):
  ACS  ecg_order     first 12-lead ECG order
       trop_order    first troponin order               (main ACS reference)
       trop_result   first troponin result time
  PE   ctpa_order    first CTPA order, IMG1266           (main PE reference)
       ctpa_result   first CTPA result time
  Sepsis bcx_order   first blood culture order      (main sepsis reference)
       lact_order    first lactate order
Result times stamped exactly at midnight are treated as missing.

lead time = reference time - alert time (positive = model earlier).
"Delayed" = reference later than DELAY_MIN after arrival.

Alert rates within the observation window (positives with / without the
work-up order, and negatives as the false-alert burden) come from the
landmark grid in landmark_encounter_detail.parquet.

Outputs under --out_dir:
  lead_time_cases.csv     one row per positive stay (stays on the server)
  lead_time_summary.csv   per reference, all vs delayed
  alert_rates.csv         alert rate by group
  alert_triggers.csv      event type that pushed risk over the threshold
  lead_time_dotplot.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from common import config as cfg
from common.event_utils import (EncounterEventIndex, NoteEmbeddingIndex,
                                landmark_upper_bound)
from sequence_dataset import EventSequenceDataset, collate_sequence_batch
from sequence_model import EventSequenceModel

ECG = ["EKG1", "ECG1", "EKG17", "EKG7", "EKG14", "EKG2", "EKG13",
       "93000", "93000.005", "93000.006", "93005", "93005.006", "93010"]
TROP = ["LAB747", "LAB8259", "LAB8363", "LAB8131",
        "POC52", "POC24", "NUR1147"]
CTPA = ["IMG1266"]
BCX = ["LAB462", "LAB3337", "PLAB900", "NUR613"]
LACT = ["LAB730", "LAB8345", "PLAB322", "POC203", "POC202", "NUR1254"]
MAX_REF_MIN = 24 * 60
DELAY_MIN = 120
ALL_OUTCOMES = ("acs_mi", "pe", "sepsis")
MAIN_REF = {"acs_mi": "trop_order", "pe": "ctpa_order",
            "sepsis": "bcx_order"}
ALL_REFS = [("acs_mi", "ecg_order", "ACS: ECG order"),
            ("acs_mi", "trop_order", "ACS: troponin order"),
            ("acs_mi", "trop_result", "ACS: troponin result"),
            ("pe", "ctpa_order", "PE: CTPA order"),
            ("pe", "ctpa_result", "PE: CTPA result"),
            ("sepsis", "bcx_order", "Sepsis: blood culture order"),
            ("sepsis", "lact_order", "Sepsis: lactate order")]
COLORS = {"acs_mi": "#2b6cb0", "pe": "#c05621", "sepsis": "#2f855a"}
# set in main() from --outcomes
OUTCOMES = ()
REFS = []


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", required=True)
    ap.add_argument("--embed_cache_dir", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--orders", required=True, nargs="+",
                    help="one or more order parquets (same columns)")
    ap.add_argument("--outcomes", nargs="+", default=["sepsis"],
                    choices=ALL_OUTCOMES)
    ap.add_argument("--master", required=True,
                    help="master parquet (stay_id, intime)")
    ap.add_argument("--detail", required=True,
                    help="landmark_encounter_detail.parquet (test split)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    return ap.parse_args()


def clean_id(s):
    s = s.astype(str).str.strip()
    s = s.where(~s.isin(["nan", "None", ""]))
    return s.str.replace(r"\.0+$", "", regex=True)


def first_minutes(df, time_col, intime):
    """Per stay: minutes from arrival of the first time in [0, 24 h]."""
    d = df[["stay_id", time_col]].dropna()
    d = d[d["stay_id"].isin(intime.index)]
    arrival = d["stay_id"].map(intime)
    d = d.assign(m=(d[time_col] - arrival).dt.total_seconds() / 60)
    d = d[(d["m"] >= 0) & (d["m"] <= MAX_REF_MIN)]
    return d.groupby("stay_id")["m"].min()


def reference_times(a, stays, intime):
    cols = ["SERVICE_ID", "PARENT_SERVICE_ID", "PROC_CODE",
            "ORDER_STATUS_ORIG", "ORDER_DATETIME", "RESULT_DATETIME"]
    o = pd.concat([pd.read_parquet(f, columns=cols) for f in a.orders],
                  ignore_index=True)
    o["SERVICE_ID"] = clean_id(o["SERVICE_ID"])
    o["PARENT_SERVICE_ID"] = clean_id(o["PARENT_SERVICE_ID"])
    # link on SERVICE_ID, fall back to PARENT_SERVICE_ID
    direct = o["SERVICE_ID"].isin(stays)
    parent = o["PARENT_SERVICE_ID"].isin(stays)
    o["stay_id"] = np.where(direct, o["SERVICE_ID"],
                            np.where(parent, o["PARENT_SERVICE_ID"], None))
    o = o[o["stay_id"].notna()]
    o = o[o["ORDER_STATUS_ORIG"].fillna("") != "CANCELED"]
    o["PROC_CODE"] = o["PROC_CODE"].astype(str).str.strip()
    for c in ("ORDER_DATETIME", "RESULT_DATETIME"):
        o[c] = pd.to_datetime(o[c], errors="coerce")
    res = o["RESULT_DATETIME"]
    o.loc[res.dt.floor("D") == res, "RESULT_DATETIME"] = pd.NaT

    ecg = o[o["PROC_CODE"].isin(ECG)]
    trop = o[o["PROC_CODE"].isin(TROP)]
    ctpa = o[o["PROC_CODE"].isin(CTPA)]
    bcx = o[o["PROC_CODE"].isin(BCX)]
    lact = o[o["PROC_CODE"].isin(LACT)]
    refs = pd.DataFrame({
        "ecg_order": first_minutes(ecg, "ORDER_DATETIME", intime),
        "trop_order": first_minutes(trop, "ORDER_DATETIME", intime),
        "trop_result": first_minutes(trop, "RESULT_DATETIME", intime),
        "ctpa_order": first_minutes(ctpa, "ORDER_DATETIME", intime),
        "ctpa_result": first_minutes(ctpa, "RESULT_DATETIME", intime),
        "bcx_order": first_minutes(bcx, "ORDER_DATETIME", intime),
        "lact_order": first_minutes(lact, "ORDER_DATETIME", intime),
    })
    return refs.reindex(sorted(stays))


def load_model(ckpt, device):
    model = EventSequenceModel(
        num_static_features=len(ckpt["static_cols"]),
        num_labels=len(ckpt["outcome_cols"]),
        note_hidden=ckpt.get("note_hidden", 768)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def trigger_of(sid, t, structured, note_meta):
    """Event type(s) at minute t that the alert followed."""
    kinds = []
    entry = structured._vitals_by_stay.get(sid)
    if entry is not None:
        times, states = entry
        hits = np.flatnonzero(times == t)
        for k in hits:
            if k == 0:
                kinds.append("triage")
                continue
            changed = np.flatnonzero(states[k] != states[k - 1])
            chans = [cfg.STRUCTURED_CHANNELS[c] for c in changed]
            lab = any(c in cfg.LAB_CHANNELS for c in chans)
            vital = any(c in cfg.VITAL_CHANNELS for c in chans)
            if lab:
                kinds.append("lab")
            if vital or not lab:
                kinds.append("vital")
    notes = note_meta.get(sid)
    if notes is not None:
        for tos, role in notes.get(round(float(t), 3), []):
            kinds.append(f"note: {tos} / {role}")
    if not kinds:
        kinds.append("arrival (static only)")
    return kinds


def alert_times(a, pos, ckpt, device):
    """Per outcome: first query time with risk >= threshold, and the
    event type(s) at that time."""
    outcome_cols = ckpt["outcome_cols"]
    static_cols = ckpt["static_cols"]
    max_lm = ckpt.get("max_landmark_minutes",
                      cfg.MAX_LANDMARK_MINUTES_DEFAULT)
    thr = ckpt.get("thresholds", {})
    stays = set(pos["stay_id"])

    sdf = pd.read_parquet(os.path.join(
        a.event_dir, f"{a.split}_structured_events.parquet"))
    sdf = sdf[sdf["stay_id"].astype(str).isin(stays)]
    empty_notes = pd.DataFrame(columns=["stay_id", "t_minutes", "note_text"])
    structured = EncounterEventIndex(pos, sdf, empty_notes)

    idx = pd.read_parquet(os.path.join(
        a.embed_cache_dir, f"{a.split}_note_embeddings_index.parquet"))
    idx["stay_id"] = idx["stay_id"].astype(str)
    idx = idx[idx["stay_id"].isin(stays)]
    emb = np.load(os.path.join(
        a.embed_cache_dir, f"{a.split}_note_embeddings.npy"), mmap_mode="r")
    notes = NoteEmbeddingIndex(emb, idx)
    note_meta = {}
    for c in ("tos", "role"):
        if c not in idx.columns:
            idx[c] = "NA"
    for sid, t, tos, role in idx[["stay_id", "t_minutes",
                                  "tos", "role"]].itertuples(index=False):
        key = round(float(t), 3)
        note_meta.setdefault(sid, {}).setdefault(key, []).append((tos, role))

    # one query per (stay, event time within the window), plus t=0
    pos_idx, times = [], []
    for i, (sid, b) in enumerate(zip(pos["stay_id"],
                                     pos["data_boundary_minutes"])):
        upper = landmark_upper_bound(b, max_lm)
        vt, _ = structured.vitals_events_upto(sid, upper)
        nt, _ = notes.embeddings_upto(sid, upper)
        ts = np.unique(np.r_[0.0, vt, nt])
        pos_idx.extend([i] * len(ts))
        times.extend(ts.tolist())
    q = pos.iloc[pos_idx].reset_index(drop=True)
    print(f"queries: {len(q):,} for {len(pos):,} stays")

    ds = EventSequenceDataset(
        q, structured, notes, outcome_cols, static_cols,
        max_landmark_minutes=max_lm, seed=0, resample_each_epoch=False,
        fixed_landmark_minutes=0.0)
    ds.landmarks[:] = np.asarray(times)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_sequence_batch)
    model = load_model(ckpt, device)
    probs = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["content"].to(device), b["modality"].to(device),
                           b["delta_t"].to(device), b["lengths"],
                           b["static"].to(device))
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    probs = np.concatenate(probs)

    alerts, triggers = {}, {}
    qt = q[["stay_id"]].assign(t=times)
    for name in OUTCOMES:
        col = f"outcome_{name}"
        th = thr.get(col, 0.5)
        is_pos = pos.loc[pos[col] == 1, "stay_id"]
        hit = qt.assign(p=probs[:, outcome_cols.index(col)])
        hit = hit[(hit["p"] >= th) & hit["stay_id"].isin(is_pos)]
        first = hit.groupby("stay_id")["t"].min()
        alerts[name] = first
        triggers[name] = {sid: trigger_of(sid, t, structured, note_meta)
                          for sid, t in first.items()}
        print(f"{col}: threshold={th:.3f}  alerted={len(first):,}")
    return alerts, triggers


def lead_row(d, ref, label, group):
    has_ref = d[ref].notna()
    lt = d.loc[has_ref, f"lead_{ref}"].dropna()
    q = lt.quantile([0.25, 0.5, 0.75])
    n_ref = int(has_ref.sum())
    return {
        "reference": label, "group": group, "positives": len(d),
        "with_reference": n_ref, "alerted": len(lt),
        "not_alerted_in_window": n_ref - len(lt),
        "median_min": q[0.5], "q1_min": q[0.25], "q3_min": q[0.75],
        "pct_model_earlier": (lt > 0).mean() * 100 if len(lt) else np.nan,
        # never-alerted stays counted as not earlier
        "pct_model_earlier_all": (lt > 0).sum() / max(n_ref, 1) * 100,
        "median_ref_min": d.loc[has_ref, ref].median(),
    }


def summarize(cases):
    rows = []
    for name, ref, label in REFS:
        d = cases[cases[f"pos_{name}"] == 1]
        rows.append(lead_row(d, ref, label, "all"))
        late = d[d[ref] > DELAY_MIN]
        rows.append(lead_row(late, ref, label,
                             f"delayed (>{DELAY_MIN} min)"))
    return pd.DataFrame(rows)


def rate(hit, ids):
    return hit.reindex(ids, fill_value=False).astype(float).mean() * 100


def alert_rates(a, cases, thr):
    """Alert within the window on the landmark grid: any landmark with risk
    >= threshold; also by DELAY_MIN."""
    rows = []
    for name in OUTCOMES:
        col = f"outcome_{name}"
        th = thr.get(col, 0.5)
        det = pd.read_parquet(a.detail, columns=[
            "landmark_minutes", "stay_id", f"{col}_prob", f"{col}_label"])
        det["stay_id"] = det["stay_id"].astype(str)
        det["hit"] = det[f"{col}_prob"] >= th
        any_hit = det.groupby("stay_id")["hit"].any()
        early = det[det["landmark_minutes"] <= DELAY_MIN]
        early_hit = early.groupby("stay_id")["hit"].any()
        label = det.groupby("stay_id")[f"{col}_label"].first()

        ref = MAIN_REF[name]
        pos_ids = label.index[label == 1]
        with_ref = cases.index[cases[ref].notna()]
        groups = {
            "positives, work-up ordered":
                pos_ids.intersection(with_ref),
            "positives, no work-up ordered":
                pos_ids.difference(with_ref),
            "negatives": label.index[label == 0],
        }
        for g, ids in groups.items():
            rows.append({
                "outcome": name, "group": g,
                "work_up": ref, "n": len(ids),
                "alert_in_window_pct": rate(any_hit, ids),
                f"alert_by_{DELAY_MIN}min_pct": rate(early_hit, ids),
            })
    return pd.DataFrame(rows)


def trigger_table(cases, triggers):
    rows = []
    for name in OUTCOMES:
        ref = MAIN_REF[name]
        earlier = cases.index[cases[f"lead_{ref}"] > 0]
        for scope, keep in (("all alerts", None),
                            (f"alerts before {ref}", set(earlier))):
            counts = {}
            for sid, kinds in triggers[name].items():
                if keep is not None and sid not in keep:
                    continue
                for k in kinds:
                    counts[k] = counts.get(k, 0) + 1
            for k, n in sorted(counts.items(), key=lambda x: -x[1]):
                rows.append({"outcome": name, "scope": scope,
                             "trigger": k, "n": n})
    return pd.DataFrame(rows)


def dotplot(cases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    width = max(4, 2.2 * len(REFS))
    fig, ax = plt.subplots(figsize=(width, 5), dpi=200)
    for i, (name, ref, label) in enumerate(REFS):
        d = cases[cases[f"pos_{name}"] == 1]
        color = COLORS[name]
        for j, (sub, alpha) in enumerate(((d, 0.25),
                                          (d[d[ref] > DELAY_MIN], 0.45))):
            lt = sub[f"lead_{ref}"].dropna().to_numpy() / 60
            if not len(lt):
                continue
            x = i + (j - 0.5) * 0.4
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(lt)), lt, s=3,
                       alpha=alpha, color=color, linewidths=0)
            q1, med, q3 = np.percentile(lt, [25, 50, 75])
            ax.plot([x - 0.15, x + 0.15], [med, med], color="black", lw=2)
            ax.plot([x, x], [q1, q3], color="black", lw=1.2)
            for y in (q1, q3):
                ax.plot([x - 0.06, x + 0.06], [y, y], color="black", lw=1)
            ax.annotate(f"n={len(lt)}\n{med:.1f} h", (x, 1.0),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(REFS)))
    ax.set_xticklabels([f"{r[2]}\nall | delayed" for r in REFS], fontsize=7)
    ax.set_ylabel("Lead time (h) = reference time - model alert time")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main():
    global OUTCOMES, REFS
    a = parse_args()
    OUTCOMES = tuple(o for o in ALL_OUTCOMES if o in a.outcomes)
    REFS = [r for r in ALL_REFS if r[0] in OUTCOMES]
    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt_path, map_location=device)
    thr = ckpt.get("thresholds", {})

    st = pd.read_parquet(os.path.join(a.event_dir,
                                      f"{a.split}_static.parquet"))
    st["stay_id"] = st["stay_id"].astype(str)
    out_cols = [f"outcome_{n}" for n in OUTCOMES]
    is_pos = (st[out_cols] == 1).any(axis=1)
    pos = st[is_pos].reset_index(drop=True)
    m = pd.read_parquet(a.master, columns=["stay_id", "intime"])
    m["stay_id"] = clean_id(m["stay_id"])
    m = m.drop_duplicates("stay_id").set_index("stay_id")
    intime = pd.to_datetime(m["intime"], errors="coerce")
    intime = intime.reindex(pos["stay_id"]).dropna()
    counts = "  ".join(f"{n} {int(pos[f'outcome_{n}'].sum()):,}"
                       for n in OUTCOMES)
    print(f"positive stays: {counts}")

    refs = reference_times(a, set(pos["stay_id"]), intime)
    alerts, triggers = alert_times(a, pos, ckpt, device)

    cases = pos[["stay_id", "data_boundary_minutes"] + out_cols].rename(
        columns={f"outcome_{n}": f"pos_{n}" for n in OUTCOMES})
    cases = cases.set_index("stay_id")
    cases = cases.join(refs)
    for name in OUTCOMES:
        cases[f"alert_{name}"] = alerts[name]
    for name, ref, _ in REFS:
        cases[f"lead_{ref}"] = cases[ref] - cases[f"alert_{name}"]
    cases.reset_index().to_csv(
        os.path.join(a.out_dir, "lead_time_cases.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    fmt = {"float_format": lambda x: f"{x:.1f}", "index": False}

    summary = summarize(cases)
    summary.to_csv(os.path.join(a.out_dir, "lead_time_summary.csv"),
                   index=False)
    print("\n=== lead time ===\n" + summary.to_string(**fmt))

    rates = alert_rates(a, cases, thr)
    rates.to_csv(os.path.join(a.out_dir, "alert_rates.csv"), index=False)
    print("\n=== alert rates (landmark grid) ===\n" + rates.to_string(**fmt))

    trig = trigger_table(cases, triggers)
    trig.to_csv(os.path.join(a.out_dir, "alert_triggers.csv"), index=False)
    print("\n=== alert triggers (top 15 per outcome and scope) ===")
    top = trig.groupby(["outcome", "scope"]).head(15)
    print(top.to_string(index=False))

    dotplot(cases, os.path.join(a.out_dir, "lead_time_dotplot.png"))
    print(f"\nsaved to {a.out_dir}")


if __name__ == "__main__":
    main()
