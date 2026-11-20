#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arrival (t=0) baselines vs ED-MERGE on the internal test set, and 72-hour
revisit evaluated in discharged patients only.

Models (all use only information available at arrival):
  news      modified NEWS2 from triage vitals: respiratory rate, SpO2,
            systolic BP, heart rate, temperature (AVPU and supplemental
            oxygen unavailable, scored 0; a missing vital scores 0).
            One score, used for every outcome.
  lgbm      LightGBM per outcome on the same structured inputs ED-MERGE has
            at t=0 (static features + triage vitals), no notes. Trained on
            the train split, early stopping on the val split.
  edmerge   ED-MERGE at landmark 0 (landmark_encounter_detail.parquet).

72-hour revisit is evaluated in discharged encounters only
(outcome_hospitalization == 0 and outcome_critical == 0); the macro average
uses this definition.

Patient-level paired bootstrap: every replicate resamples patients once and
scores all models on the same sample, so the CI of a difference is valid.

Outputs under --out_dir:
  baseline_predictions.parquet   per test encounter (stays on the server)
  baseline_metrics_ci.csv        model x outcome x metric, estimate + CI
  baseline_delta_ci.csv          ED-MERGE minus baseline, estimate + CI
  revisit_discharged_ci.csv      ED-MERGE revisit, discharged only, per
                                 landmark
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config as cfg
from bootstrap_ci import SortedScores

REVISIT = "outcome_ed_revisit_3d"
MODELS = ("news", "lgbm", "edmerge")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", required=True,
                    help="{train,val,test}_static.parquet")
    ap.add_argument("--master", required=True,
                    help="master parquet with triage_* vitals")
    ap.add_argument("--detail", required=True,
                    help="landmark_encounter_detail.parquet (test)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=8)
    return ap.parse_args()


def clean_id(s):
    s = s.astype(str).str.strip()
    s = s.where(~s.isin(["nan", "None", ""]))
    return s.str.replace(r"\.0+$", "", regex=True)


# --------------------------------------------------------------- NEWS


def band(x, cuts, points):
    """Points for x by upper bounds `cuts` (inclusive); NaN -> 0."""
    out = np.zeros(len(x))
    idx = np.searchsorted(np.asarray(cuts), x, side="left")
    pts = np.asarray(points)
    ok = ~np.isnan(x)
    out[ok] = pts[idx[ok]]
    return out


def modified_news(v):
    """NEWS2 without AVPU and supplemental oxygen. v: DataFrame of raw
    triage vitals."""
    temp = v["triage_temperature"].to_numpy(dtype=float)
    if np.nanmedian(temp) > 50:  # Fahrenheit
        temp = (temp - 32) * 5 / 9
    rr = v["triage_resprate"].to_numpy(dtype=float)
    spo2 = v["triage_o2sat"].to_numpy(dtype=float)
    sbp = v["triage_sbp"].to_numpy(dtype=float)
    hr = v["triage_heartrate"].to_numpy(dtype=float)
    # each band: score for x <= cut[i]; last entry is for x > last cut
    score = (
        band(rr, [8, 11, 20, 24], [3, 1, 0, 2, 3])
        + band(spo2, [91, 93, 95], [3, 2, 1, 0])
        + band(sbp, [90, 100, 110, 219], [3, 2, 1, 0, 3])
        + band(hr, [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
        + band(temp, [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    )
    return score


# --------------------------------------------------------------- LightGBM


def load_split(a, split, triage):
    st = pd.read_parquet(os.path.join(a.event_dir,
                                      f"{split}_static.parquet"))
    st["stay_id"] = st["stay_id"].astype(str)
    return st.merge(triage, on="stay_id", how="left")


def fit_lgbm(train, val, test, feats, outcome_cols, n_jobs):
    import lightgbm as lgb
    preds = {}
    for col in outcome_cols:
        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.1, num_leaves=63,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, n_jobs=n_jobs, verbose=-1)
        clf.fit(train[feats], train[col],
                eval_set=[(val[feats], val[col])], eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        preds[col] = clf.predict_proba(test[feats])[:, 1]
        print(f"  lgbm {col}: best_iter={clf.best_iteration_}", flush=True)
    return preds


# --------------------------------------------------------------- bootstrap


def patient_weights(subject, n_boot, seed):
    codes, uniq = pd.factorize(subject)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draws = rng.integers(0, len(uniq), len(uniq))
        yield np.bincount(draws, minlength=len(uniq))[codes].astype(float)


def eval_population(df, col):
    if col == REVISIT:
        return df["discharged"].to_numpy()
    return np.ones(len(df), dtype=bool)


def bootstrap_models(df, outcome_cols, n_boot, seed):
    """AUROC / AUPRC per model and outcome with paired patient bootstrap."""
    scorers = {}
    for col in outcome_cols:
        m = eval_population(df, col)
        y = df.loc[m, col].to_numpy()
        for model in MODELS:
            s = df.loc[m, f"{model}_{col}"].to_numpy(dtype=float)
            scorers[(model, col)] = (m, SortedScores(s, y))

    def score_all(w):
        out = {}
        for (model, col), (m, sc) in scorers.items():
            out[(model, col)] = sc.auroc_ap(w[m])
        return out

    point = score_all(np.ones(len(df)))
    reps = [score_all(w) for w in
            patient_weights(df["subject_id"], n_boot, seed)]
    return point, reps


def summarize(point, reps, outcome_cols):
    rows, deltas = [], []
    keys = [(m, c) for m in MODELS for c in outcome_cols]
    for k, metric in enumerate(("AUROC", "AUPRC")):
        est = {key: point[key][k] for key in keys}
        boot = {key: np.array([r[key][k] for r in reps]) for key in keys}
        for m in MODELS:
            est[(m, "macro")] = np.mean([est[(m, c)] for c in outcome_cols])
            boot[(m, "macro")] = np.mean(
                [boot[(m, c)] for c in outcome_cols], axis=0)
        for m in MODELS:
            for c in list(outcome_cols) + ["macro"]:
                lo, hi = np.nanpercentile(boot[(m, c)], [2.5, 97.5])
                rows.append({"model": m, "outcome": c, "metric": metric,
                             "estimate": est[(m, c)],
                             "ci_low": lo, "ci_high": hi})
        for base in ("news", "lgbm"):
            for c in list(outcome_cols) + ["macro"]:
                d = boot[("edmerge", c)] - boot[(base, c)]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                deltas.append({
                    "comparison": f"edmerge - {base}", "outcome": c,
                    "metric": metric,
                    "delta": est[("edmerge", c)] - est[(base, c)],
                    "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def revisit_landmarks(det, disc, subj, n_boot, seed):
    """ED-MERGE revisit AUROC/AUPRC in discharged patients per landmark."""
    rows = []
    det = det[det["stay_id"].isin(disc)]
    for T, g in det.groupby("landmark_minutes"):
        g = g.reset_index(drop=True)
        sc = SortedScores(g[f"{REVISIT}_prob"].to_numpy(dtype=float),
                          g[f"{REVISIT}_label"].to_numpy())
        est = sc.auroc_ap(np.ones(len(g)))
        sub = g["stay_id"].map(subj)
        boot = np.array([sc.auroc_ap(w) for w in
                         patient_weights(sub, n_boot, seed)])
        for k, metric in enumerate(("AUROC", "AUPRC")):
            lo, hi = np.nanpercentile(boot[:, k], [2.5, 97.5])
            rows.append({"landmark_minutes": T, "metric": metric,
                         "n": len(g), "estimate": est[k],
                         "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- main


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tri_cols = ["stay_id"] + cfg.TRIAGE_COLS
    m = pd.read_parquet(a.master, columns=tri_cols)
    m["stay_id"] = clean_id(m["stay_id"])
    triage = m.dropna(subset=["stay_id"]).drop_duplicates("stay_id")
    for c in cfg.TRIAGE_COLS:
        triage[c] = pd.to_numeric(triage[c], errors="coerce")

    train = load_split(a, "train", triage)
    val = load_split(a, "val", triage)
    test = load_split(a, "test", triage)
    outcome_cols = [c for c in cfg.OUTCOME_COLS_LIST if c in test.columns]
    static_cols = [c for c in test.columns
                   if c.startswith(("cci_", "eci_", "chiefcom_"))
                   or c == "age"]
    feats = static_cols + cfg.TRIAGE_COLS
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}  "
          f"features {len(feats)}", flush=True)

    # ED-MERGE at arrival
    det = pd.read_parquet(a.detail)
    det["stay_id"] = det["stay_id"].astype(str)
    d0 = det[det["landmark_minutes"] == 0].set_index("stay_id")
    test = test[test["stay_id"].isin(d0.index)].reset_index(drop=True)
    print(f"test encounters with ED-MERGE t=0: {len(test):,}", flush=True)

    df = test[["stay_id", "subject_id"] + outcome_cols].copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["discharged"] = ((test["outcome_hospitalization"] == 0)
                        & (test["outcome_critical"] == 0)).to_numpy()
    news = modified_news(test)
    print("training LightGBM...", flush=True)
    lgbm = fit_lgbm(train, val, test, feats, outcome_cols, a.n_jobs)
    for col in outcome_cols:
        df[f"news_{col}"] = news
        df[f"lgbm_{col}"] = lgbm[col]
        df[f"edmerge_{col}"] = d0.loc[df["stay_id"], f"{col}_prob"].values
    df.to_parquet(os.path.join(a.out_dir, "baseline_predictions.parquet"),
                  index=False)
    print(f"discharged (revisit population): {int(df['discharged'].sum()):,}"
          f"  revisit events: "
          f"{int(df.loc[df['discharged'], REVISIT].sum()):,}", flush=True)

    print(f"bootstrap x{a.n_boot}...", flush=True)
    point, reps = bootstrap_models(df, outcome_cols, a.n_boot, a.seed)
    metrics, deltas = summarize(point, reps, outcome_cols)
    metrics.to_csv(os.path.join(a.out_dir, "baseline_metrics_ci.csv"),
                   index=False)
    deltas.to_csv(os.path.join(a.out_dir, "baseline_delta_ci.csv"),
                  index=False)

    subj = df.set_index("stay_id")["subject_id"]
    disc = set(df.loc[df["discharged"], "stay_id"])
    rev = revisit_landmarks(det, disc, subj, a.n_boot, a.seed)
    rev.to_csv(os.path.join(a.out_dir, "revisit_discharged_ci.csv"),
               index=False)

    pd.set_option("display.width", 200, "display.max_rows", 200)

    def fmt(r):
        return f"{r.estimate:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f})"

    auc = metrics[metrics["metric"] == "AUROC"].copy()
    auc["v"] = auc.apply(fmt, axis=1)
    table = auc.pivot(index="outcome", columns="model", values="v")
    order = outcome_cols + ["macro"]
    print("\n=== AUROC at arrival (95% CI) ===")
    print(table.reindex(order)[list(MODELS)].to_string())

    dl = deltas[deltas["metric"] == "AUROC"].copy()
    dl["v"] = dl.apply(
        lambda r: f"{r.delta:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f})",
        axis=1)
    dtab = dl.pivot(index="outcome", columns="comparison", values="v")
    print("\n=== delta AUROC, ED-MERGE minus baseline (95% CI) ===")
    print(dtab.reindex(order).to_string())

    print("\n=== ED-MERGE revisit, discharged only ===")
    rv = rev.copy()
    rv["v"] = rv.apply(fmt, axis=1)
    print(rv.pivot(index="landmark_minutes", columns="metric",
                   values="v").to_string())
    print(f"\nsaved to {a.out_dir}")


if __name__ == "__main__":
    main()
