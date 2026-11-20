#!/usr/bin/env python3
"""
Arrival (t=0) logistic regression baseline vs ED-MERGE and modified NEWS
on the internal test set (Supplementary Table S6).

Reuses baseline_predictions.parquet written by baselines.py (NEWS and
ED-MERGE scores at t=0, outcomes, discharged flag), so only the logistic
regression is fitted here.

Logistic regression, one per outcome, on the same inputs ED-MERGE has at
t=0 (static features + triage vitals), no notes:
  - continuous inputs (age, triage vitals) clipped to the training
    0.5-99.5th percentiles and imputed with the training median
  - one missing-indicator per triage vital
  - all columns standardized on the training split
  - L2 penalty, C = --C (default 1.0), lbfgs
Fitted on the train split; validation AUROC is printed as a check.

72-hour revisit is evaluated in discharged encounters only, as in
baselines.py. Patient-level paired bootstrap (same resamples for all
models).

  python baselines_lr.py --event_dir <dir> --master <parquet> \
      --pred <baselines_out>/baseline_predictions.parquet \
      --out_dir <baselines_out>/lr

Outputs under --out_dir (aggregates only):
  lr_metrics_ci.csv   model x outcome x metric, estimate + CI
  lr_delta_ci.csv     ED-MERGE minus baseline, estimate + CI
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config as cfg
from bootstrap_ci import SortedScores
from baselines import clean_id, load_split, patient_weights

REVISIT = "outcome_ed_revisit_3d"
MODELS = ("news", "lr", "edmerge")
COMPS = ("lr", "news")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", required=True,
                    help="{train,val,test}_static.parquet")
    ap.add_argument("--master", required=True,
                    help="master parquet with triage_* vitals")
    ap.add_argument("--pred", required=True,
                    help="baseline_predictions.parquet from baselines.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# --------------------------------------------------------------- features


def design(train, val, test, feats):
    """Clip, impute, add missing indicators, standardize (train stats)."""
    from sklearn.preprocessing import StandardScaler
    cont = [c for c in ["age"] + list(cfg.TRIAGE_COLS) if c in feats]
    lo = train[cont].quantile(0.005)
    hi = train[cont].quantile(0.995)
    med = train[cont].median()

    def build(d):
        X = d[feats].apply(pd.to_numeric, errors="coerce")
        X = X.astype("float32")
        for c in cfg.TRIAGE_COLS:
            X["miss_" + c] = X[c].isna().astype("float32")
        X[cont] = X[cont].clip(lo, hi, axis=1).fillna(med)
        return X.fillna(0.0)

    Xtr, Xva, Xte = build(train), build(val), build(test)
    sc = StandardScaler().fit(Xtr)
    return [sc.transform(X).astype("float32") for X in (Xtr, Xva, Xte)]


def fit_lr(Xtr, Xva, Xte, train, val, outcome_cols, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    preds = {}
    for col in outcome_cols:
        t0 = time.time()
        ytr = train[col].to_numpy().astype(int)
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        clf.fit(Xtr, ytr)
        yva = val[col].to_numpy().astype(int)
        auc_va = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        preds[col] = clf.predict_proba(Xte)[:, 1]
        print(f"  lr {col}: iter={int(clf.n_iter_[0])} "
              f"val_auroc={auc_va:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    return preds


# --------------------------------------------------------------- bootstrap


def bootstrap(df, outcome_cols, n_boot, seed):
    scorers = {}
    for col in outcome_cols:
        m = (df["discharged"].to_numpy() if col == REVISIT
             else np.ones(len(df), dtype=bool))
        y = df.loc[m, col].to_numpy()
        for model in MODELS:
            s = df.loc[m, f"{model}_{col}"].to_numpy(dtype=float)
            scorers[(model, col)] = (m, SortedScores(s, y))

    def score_all(w):
        return {k: sc.auroc_ap(w[m]) for k, (m, sc) in scorers.items()}

    point = score_all(np.ones(len(df)))
    reps = [score_all(w) for w in
            patient_weights(df["subject_id"], n_boot, seed)]
    return point, reps


def summarize(point, reps, outcome_cols):
    rows, deltas = [], []
    cols = list(outcome_cols) + ["macro"]
    for k, metric in enumerate(("AUROC", "AUPRC")):
        est, boot = {}, {}
        for m in MODELS:
            for c in outcome_cols:
                est[(m, c)] = point[(m, c)][k]
                boot[(m, c)] = np.array([r[(m, c)][k] for r in reps])
            est[(m, "macro")] = np.mean(
                [est[(m, c)] for c in outcome_cols])
            boot[(m, "macro")] = np.mean(
                [boot[(m, c)] for c in outcome_cols], axis=0)
            for c in cols:
                lo, hi = np.nanpercentile(boot[(m, c)], [2.5, 97.5])
                rows.append({"model": m, "outcome": c, "metric": metric,
                             "estimate": est[(m, c)],
                             "ci_low": lo, "ci_high": hi})
        for base in COMPS:
            for c in cols:
                d = boot[("edmerge", c)] - boot[(base, c)]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                deltas.append({
                    "comparison": f"edmerge - {base}", "outcome": c,
                    "metric": metric,
                    "delta": est[("edmerge", c)] - est[(base, c)],
                    "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def show(metrics, deltas, order):
    def ci(e, lo, hi, sign=""):
        return f"{e:{sign}.3f} ({lo:{sign}.3f}, {hi:{sign}.3f})"

    for metric in ("AUROC", "AUPRC"):
        mt = metrics[metrics["metric"] == metric]
        dt = deltas[deltas["metric"] == metric]
        tab = pd.DataFrame(index=order)
        for m in MODELS:
            g = mt[mt["model"] == m].set_index("outcome")
            tab[m] = [ci(*g.loc[o, ["estimate", "ci_low", "ci_high"]])
                      for o in order]
        for base in COMPS:
            g = dt[dt["comparison"] == f"edmerge - {base}"]
            g = g.set_index("outcome")
            tab[f"edmerge-{base}"] = [
                ci(*g.loc[o, ["delta", "ci_low", "ci_high"]], sign="+")
                for o in order]
        print(f"\n=== {metric} at arrival (95% CI) ===")
        print(tab.to_string())


# --------------------------------------------------------------- main


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tri_cols = ["stay_id"] + list(cfg.TRIAGE_COLS)
    m = pd.read_parquet(a.master, columns=tri_cols)
    m["stay_id"] = clean_id(m["stay_id"])
    triage = m.dropna(subset=["stay_id"]).drop_duplicates("stay_id")
    for c in cfg.TRIAGE_COLS:
        triage[c] = pd.to_numeric(triage[c], errors="coerce")

    train = load_split(a, "train", triage)
    val = load_split(a, "val", triage)
    test = load_split(a, "test", triage)

    df = pd.read_parquet(a.pred)
    df["stay_id"] = df["stay_id"].astype(str)
    test = test.set_index("stay_id").loc[df["stay_id"]].reset_index()

    outcome_cols = [c for c in cfg.OUTCOME_COLS_LIST if c in df.columns]
    static_cols = [c for c in test.columns
                   if c.startswith(("cci_", "eci_", "chiefcom_"))
                   or c == "age"]
    feats = static_cols + list(cfg.TRIAGE_COLS)
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}  "
          f"features {len(feats)} (+{len(cfg.TRIAGE_COLS)} missing "
          f"indicators)", flush=True)

    Xtr, Xva, Xte = design(train, val, test, feats)
    print(f"fitting logistic regression, C={a.C}...", flush=True)
    lr = fit_lr(Xtr, Xva, Xte, train, val, outcome_cols, a.C)
    for col in outcome_cols:
        df[f"lr_{col}"] = lr[col]

    print(f"bootstrap x{a.n_boot}...", flush=True)
    point, reps = bootstrap(df, outcome_cols, a.n_boot, a.seed)
    metrics, deltas = summarize(point, reps, outcome_cols)
    metrics.to_csv(os.path.join(a.out_dir, "lr_metrics_ci.csv"),
                   index=False)
    deltas.to_csv(os.path.join(a.out_dir, "lr_delta_ci.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 20)
    show(metrics, deltas, outcome_cols + ["macro"])
    print(f"\nsaved to {a.out_dir}")


if __name__ == "__main__":
    main()
