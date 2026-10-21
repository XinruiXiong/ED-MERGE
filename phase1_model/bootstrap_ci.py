#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
95% bootstrap confidence intervals for AUROC / AUPRC (and F1 when thresholds
are given) at every landmark, per outcome and macro-averaged, from the
per-encounter prediction files the evaluation scripts write:

  evaluate_landmarks.py   -> landmark_encounter_detail.parquet
  evaluate_external.py    -> external_encounter_detail.parquet
  evaluate_ablation.py    -> ablation_detail_<config>.parquet

Resampling is by PATIENT (cluster bootstrap): a patient's encounters are
drawn together, since encounters of the same patient are not independent.
Each replicate uses the same resampled patients for every outcome, so the
macro-average CI is the CI of the macro average, not an average of CIs.

Implementation: each replicate is a vector of per-encounter weights (how many
times its patient was drawn); weighted AUROC / average precision are computed
in O(n) on scores sorted once per (landmark, outcome). With unit weights they
equal sklearn's roc_auc_score / average_precision_score exactly.

Output under --out_dir (named after the detail file):
  <name>_ci.csv   landmark_minutes, outcome, metric, estimate, ci_low, ci_high, formatted
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd


class SortedScores:
    """Scores of one (landmark, outcome) sorted descending once; tie groups precomputed."""

    def __init__(self, scores, labels):
        order = np.argsort(-scores, kind="mergesort")
        self.order = order
        self.y = labels[order].astype(np.float64)
        s = scores[order]
        # index of the last element of each tie group
        self.ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
        self.scores_sorted = s

    def auroc_ap(self, w):
        w = w[self.order]
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * (1.0 - self.y))[self.ends]
        P, N = tp[-1], fp[-1]
        if P <= 0 or N <= 0:
            return np.nan, np.nan
        tp0 = np.r_[0.0, tp[:-1]]
        fp0 = np.r_[0.0, fp[:-1]]
        auroc = np.sum((fp - fp0) * (tp + tp0)) / (2.0 * P * N)
        prec = tp / np.maximum(tp + fp, 1e-12)
        ap = np.sum((tp - tp0) / P * prec)
        return auroc, ap

    def f1(self, w, thr):
        w = w[self.order]
        pred = self.scores_sorted >= thr
        tp = np.sum(w * self.y * pred)
        fp = np.sum(w * (1.0 - self.y) * pred)
        fn = np.sum(w * self.y * ~pred)
        if tp + fn <= 0:
            return np.nan
        return 2 * tp / max(2 * tp + fp + fn, 1e-12)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True, help="per-encounter prediction parquet")
    ap.add_argument("--static", required=True,
                    help="{split}_static.parquet of the same split (stay_id -> subject_id for patient-level resampling)")
    ap.add_argument("--thresholds", default=None,
                    help="best_thresholds.json from training; if given, F1 is also reported")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = pd.read_parquet(a.detail)
    d["stay_id"] = d["stay_id"].astype(str)
    st = pd.read_parquet(a.static, columns=["stay_id", "subject_id"])
    subj = st.assign(stay_id=st["stay_id"].astype(str)).set_index("stay_id")["subject_id"].astype(str)
    outcomes = [c[:-5] for c in d.columns if c.endswith("_prob")]
    thresholds = json.load(open(a.thresholds)) if a.thresholds else {}
    metrics = ["AUROC", "AUPRC"] + (["F1"] if thresholds else [])
    rng = np.random.default_rng(a.seed)

    rows = []
    for T, g in d.groupby("landmark_minutes", sort=True):
        g = g.reset_index(drop=True)
        codes, uniq = pd.factorize(g["stay_id"].map(subj).fillna(g["stay_id"]))
        n_pat = len(uniq)
        sorted_by = {o: SortedScores(g[f"{o}_prob"].to_numpy(np.float64), g[f"{o}_label"].to_numpy())
                     for o in outcomes}

        def evaluate(w):
            out = {}
            for o, ss in sorted_by.items():
                auroc, ap = ss.auroc_ap(w)
                out[(o, "AUROC")], out[(o, "AUPRC")] = auroc, ap
                if thresholds:
                    out[(o, "F1")] = ss.f1(w, thresholds.get(o, 0.5))
            for m in metrics:
                vals = [out[(o, m)] for o in outcomes]
                out[("macro", m)] = np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan
            return out

        point = evaluate(np.ones(len(g)))
        boots = {k: np.empty(a.n_boot) for k in point}
        for b in range(a.n_boot):
            w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)[codes].astype(np.float64)
            res = evaluate(w)
            for k, v in res.items():
                boots[k][b] = v
        for (o, m), est in point.items():
            lo, hi = np.nanpercentile(boots[(o, m)], [2.5, 97.5]) if not np.isnan(est) else (np.nan, np.nan)
            rows.append({"landmark_minutes": T, "outcome": o.replace("outcome_", ""), "metric": m,
                         "estimate": est, "ci_low": lo, "ci_high": hi,
                         "formatted": f"{est:.3f} ({lo:.3f}–{hi:.3f})" if not np.isnan(est) else ""})
        mac = point[("macro", "AUROC")]
        print(f"[T={T:g}] {len(g):,} encounters, {n_pat:,} patients, macro AUROC {mac:.4f}", flush=True)

    name = os.path.splitext(os.path.basename(a.detail))[0]
    out_path = os.path.join(a.out_dir, f"{name}_ci.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f">> Saved {out_path}")


if __name__ == "__main__":
    main()
