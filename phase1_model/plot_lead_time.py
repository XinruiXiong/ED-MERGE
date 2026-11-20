#!/usr/bin/env python3
"""
Supplementary Fig. S4: lead time of the first sepsis alert relative to the
first blood culture order and the first lactate order.

Reads lead_time_cases.csv written by lead_time.py; no model, no GPU.
Lead time = order time - alert time (positive: alert came first).
Each panel: all alerted encounters with the order within 24 h, and those
whose order came more than 2 h after arrival. Bars: median and IQR.

  python plot_lead_time.py --cases <dir>/lead_time_cases.csv \
      --out <dir>/figS4_lead_time.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DELAY_MIN = 120
PANELS = [("bcx_order", "Blood culture order"),
          ("lact_order", "Lactate order")]

INK = "#1B1F24"
MUTED = "#4A5260"
GRID = "#E2E5E9"
DOT_ALL = "#0072B2"
DOT_LATE = "#D55E00"
FONTS = ["Source Sans 3", "Source Sans Pro", "Helvetica Neue",
         "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def groups(cases, ref):
    d = cases[(cases["pos_sepsis"] == 1) & cases[ref].notna()]
    lead = d[f"lead_{ref}"]
    late = d[ref] > DELAY_MIN
    return [("All", lead.dropna().to_numpy(), DOT_ALL),
            ("Ordered >2 h\nafter arrival",
             lead[late].dropna().to_numpy(), DOT_LATE)]


def fmt_min(v):
    return f"{v:.0f}".replace("-", "\u2212")


def strip(ax, x, lead_min, color, rng):
    y = lead_min / 60.0
    # jitter width follows local density so dense bands read as wide
    hist, edges = np.histogram(y, bins=60)
    idx = np.clip(np.digitize(y, edges) - 1, 0, len(hist) - 1)
    dens = hist[idx] / hist.max()
    jit = rng.uniform(-1, 1, len(y)) * (0.06 + 0.26 * dens)
    ax.scatter(x + jit, y, s=2.2, color=color, alpha=0.35,
               linewidths=0, rasterized=True, zorder=2)
    q1, med, q3 = np.percentile(lead_min, [25, 50, 75])
    ax.plot([x, x], [q1 / 60, q3 / 60], color=INK, lw=1.6,
            solid_capstyle="butt", zorder=4)
    for q in (q1, q3):
        ax.plot([x - 0.09, x + 0.09], [q / 60, q / 60], color=INK,
                lw=1.4, zorder=4)
    ax.plot([x - 0.24, x + 0.24], [med / 60, med / 60], color=INK,
            lw=3.0, solid_capstyle="butt", zorder=5)
    earlier = (lead_min > 0).mean() * 100
    return med, q1, q3, earlier


def main():
    a = parse_args()
    cases = pd.read_csv(a.cases, dtype={"stay_id": str})
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONTS,
        "font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "svg.fonttype": "none", "pdf.fonttype": 42})
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for k, (ax, (ref, title)) in enumerate(zip(axes, PANELS)):
        for x, (label, lead, color) in enumerate(groups(cases, ref)):
            if not len(lead):
                continue
            med, q1, q3, earlier = strip(ax, x, lead, color, rng)
            ax.text(x, 25.2,
                    f"n = {len(lead):,}\n"
                    f"median {fmt_min(med)} min\n"
                    f"IQR {fmt_min(q1)} to {fmt_min(q3)}\n"
                    f"{earlier:.0f}% alert first",
                    ha="center", va="bottom", fontsize=7.5, color=INK,
                    linespacing=1.25)
            print(f"{title} | {label.replace(chr(10), chr(32))}: "
                  f"n={len(lead)} median={med:.0f} "
                  f"IQR {q1:.0f} to {q3:.0f} earlier={earlier:.1f}%")
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["All", "Ordered >2 h\nafter arrival"],
                           color=INK, fontsize=8.5)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_title(title, fontsize=10.5, fontweight="bold",
                     color=INK, pad=58)
        ax.text(-0.02, 1.0, "ab"[k], transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=INK,
                ha="right", va="bottom")
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
    ax0 = axes[0]
    ax0.set_ylim(-6.5, 24.5)
    ax0.set_yticks([-6, 0, 6, 12, 18, 24])
    ax0.set_ylabel("Lead time (h)\npositive: alert before order",
                   color=INK, fontsize=9.5, linespacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.12)
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
