"""Figures for the README.

Run from the project root (after 08, 09 and 10):
    python scripts/11_figures.py

Writes PNGs to results/figures/:
  1_failure_rate_by_rain.png   failure rate vs rain in the 48 h before sampling (all 20 sites)
  2_precision_recall_2026.png  precision-recall curves on the 2026 test season
  3_spill_odds_ratios.png      spill effect beyond rain, by distance band and time window

Colours come from a colour-blind-checked palette; every series is also named
in a legend, so nothing relies on colour alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, validated for colour blindness

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
})


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion (well-behaved for small counts)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")


# ------------------------------------------------- 1. failure rate by rain
f = pd.read_csv("data/processed/features.csv")
bins = [-0.01, 0.0, 0.5, 2, 4, 10, np.inf]
labels = ["0", "0-0.5", "0.5-2", "2-4", "4-10", ">10"]
f["rain_bin"] = pd.cut(f["rain_48h"], bins, labels=labels)
g = f.groupby("rain_bin", observed=False)["fail"].agg(["sum", "size"])
g = g[g["size"] > 0]  # drop empty rain groups
labels = list(g.index)
rate = g["sum"] / g["size"]
lo, hi = (np.array(v) for v in zip(*[wilson(k, n) for k, n in zip(g["sum"], g["size"])]))
rate = rate.to_numpy()

fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(len(labels))
ax.bar(x, rate * 100, width=0.6, color=SERIES[0], zorder=2)
ax.errorbar(x, rate * 100, yerr=[(rate - lo) * 100, (hi - rate) * 100], fmt="none",
            ecolor=INK2, elinewidth=1.5, capsize=4, zorder=3)
ax.set_xticks(x, [f"{l} mm\nn={n}" for l, n in zip(labels, g["size"])])
ax.set_ylabel("Samples failing (%)")
ax.set_xlabel("Rain in the 48 h before sampling")
ax.grid(axis="x", visible=False)
ax.set_ylim(0, max(100 * np.nanmax(hi) * 1.1, 10))
ax.set_title("Failure rate rises steeply with recent rain", pad=26)
subtitle(ax, f"All 20 river sites, 2021-2026 ({len(f)} samples). Whiskers: 95% intervals.")
fig.tight_layout()
fig.savefig(FIG / "1_failure_rate_by_rain.png", dpi=200)
plt.close(fig)

# ------------------------------------------- 2. precision-recall, 2026 test
p = pd.read_csv("results/test_predictions.csv")
y = p["fail"].astype(int).to_numpy()
models = [("site_rate", "Site failure rate only"), ("rain_rule", "Rain rule (48 h rain)"),
          ("logreg_all", "Logistic regression, all features"), ("forest_all", "Random forest, all features")]
fig, ax = plt.subplots(figsize=(8, 6))
for (col, name), colour in zip(models, SERIES):
    s = p[f"score_{col}"].to_numpy()
    prec, rec, _ = precision_recall_curve(y, s)
    ax.step(rec, prec, where="post", color=colour, label=f"{name} (PR-AUC {average_precision_score(y, s):.2f})")
ax.axhline(y.mean(), color=MUTED, linestyle=(0, (4, 3)), linewidth=1.5)
ax.text(0.01, y.mean() - 0.015, f"random guessing ({y.mean():.2f})", color=INK2, ha="left", va="top", fontsize=10)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Recall: share of failures caught")
ax.set_ylabel("Precision: share of alerts that were failures")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=10)
ax.set_title("Weather models beat site history; the rain rule does most of the work", pad=26)
subtitle(ax, f"Unseen 2026 season: {int(y.sum())} failures in {len(y)} samples. Higher and further right is better.")
fig.tight_layout()
fig.savefig(FIG / "2_precision_recall_2026.png", dpi=200)
plt.close(fig)

# --------------------------------------------- 3. spill odds ratios by band
e = pd.read_csv("results/spills_experiment.csv")
b = e[(e["analysis"] == "band") & e["spill_feature"].str.match(r"spill_h_\d+h_\d+km")].copy()
b["window"] = b["spill_feature"].str.extract(r"_(\d+)h_").astype(int)[0]
b["band"] = b["spill_feature"].str.extract(r"_(\d+)km").astype(int)[0]
bands, windows = sorted(b["band"].unique()), sorted(b["window"].unique())
fig, ax = plt.subplots(figsize=(8, 5.4))
offsets = np.linspace(0.22, -0.22, len(windows))  # first window on top, matching the legend
for colour, w, off in zip(SERIES, windows, offsets):
    r = b[b["window"] == w].set_index("band").reindex(bands)
    yy = np.arange(len(bands)) + off
    ax.errorbar(r["OR_per_log_unit"], yy, xerr=[r["OR_per_log_unit"] - r["OR_lo"], r["OR_hi"] - r["OR_per_log_unit"]],
                fmt="o", color=colour, ecolor=colour, elinewidth=2, markersize=8, capsize=0,
                markeredgecolor=SURFACE, markeredgewidth=2, label=f"last {w} h", zorder=3)
ax.axvline(1, color=AXIS, linewidth=1.5, zorder=1)
ax.text(1.0, len(bands) - 0.45, " no effect", color=INK2, fontsize=10, va="bottom")
ax.set_xscale("log")
ticks = [0.5, 1, 2, 4, 8]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.set_xlim(0.3, max(10, b["OR_hi"].max() * 1.2))
ax.set_yticks(range(len(bands)), [f"within {k} km\nupstream" for k in bands])
ax.set_ylim(-0.6, len(bands) - 0.2)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Odds ratio per log(1 + spill hours), after accounting for rain (log scale)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(windows), fontsize=10,
          handletextpad=0.3, columnspacing=1.2)
n_fail = int(e.loc[e["analysis"] == "LR test", "failures"].iloc[0]) if (e["analysis"] == "LR test").any() else None
ax.set_title("Spills hint at extra risk beyond rain, but intervals are wide", pad=26)
subtitle(ax, f"Thames sites 2023-2026, {n_fail} failures. Dots: estimates; lines: 95% confidence intervals.")
fig.tight_layout()
fig.savefig(FIG / "3_spill_odds_ratios.png", dpi=200)
plt.close(fig)

print(f"Saved 3 figures to {FIG}/")
