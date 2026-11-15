"""Error analysis and calibration on the 2026 test season.

Run from the project root (after 09_models.py):
    python scripts/12_error_analysis.py

1. Calibration: when a model says "40% chance of failing", do about 40% of
   those samples fail? (The models were trained with balanced class weights,
   which push probabilities up, so expect over-prediction.)
2. Missed failures: for the best model and the rain rule, which failures were
   missed, and what do they have in common (dry weather? one site? only just
   over the limit? which bacterium?)
3. Failures every model missed, listed one by one.
4. False alarms: where and when the best model raised alerts that didn't fail.

Writes results/figures/4_calibration_2026.png, results/missed_failures.csv
and results/false_alarms.csv.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotstyle import AXIS, FIG, INK2, MUTED, SERIES, SURFACE, subtitle, wilson

ECOLI_LIMIT, IE_LIMIT = 1000, 400
BEST = "logreg_all"
MODELS = ["site_rate", "rain_rule", "logreg_rain", "logreg_all", "forest_all"]
PROB_MODELS = [("logreg_all", "Logistic regression, all features"), ("forest_all", "Random forest, all features")]
N_BINS = 5

pred = pd.read_csv("results/test_predictions.csv")
res = pd.read_csv("results/model_results.csv").set_index("model")
feat = pd.read_csv("data/processed/features.csv")
extra = ["rain_6h", "rain_7d", "flow_change_24h", "month", "extra_sample", "discountable"]
pred = pred.merge(feat[["id", "sample_time_utc"] + extra], on=["id", "sample_time_utc"], how="left")
pred["fail"] = pred["fail"].astype(bool)
pred["date"] = pd.to_datetime(pred["sample_time_utc"], utc=True, format="ISO8601").dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
pred["which"] = np.select([(pred["ecoli"] > ECOLI_LIMIT) & (pred["ie"] > IE_LIMIT), pred["ecoli"] > ECOLI_LIMIT,
                           pred["ie"] > IE_LIMIT], ["both", "E. coli only", "enterococci only"], "neither")
# How far over the limit: the larger of count / limit for the two bacteria
pred["times_limit"] = np.maximum(pred["ecoli"] / ECOLI_LIMIT, pred["ie"] / IE_LIMIT).round(1)
pred["rain_group"] = pd.cut(pred["rain_48h"], [-np.inf, 0.5, 4, np.inf], labels=["dry (<= 0.5 mm)", "light (0.5-4 mm)", "wet (> 4 mm)"])
pred["severity"] = pd.cut(pred["times_limit"], [0, 2, 5, np.inf], labels=["1-2x limit", "2-5x limit", "> 5x limit"])
for m in MODELS:
    pred[f"alert_{m}"] = pred[f"score_{m}"] >= res.loc[m, "threshold"]
y = pred["fail"]
print(f"2026 test season: {len(pred)} samples, {int(y.sum())} failures ({y.mean():.1%})\n")

# ------------------------------------------------------------ 1. calibration
print("1. Calibration (predicted probability vs what actually happened)")
fig, ax = plt.subplots(figsize=(6.4, 7))
ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.5, zorder=1, label="perfect calibration")
ratios = []
for (m, name), colour, off in zip(PROB_MODELS, SERIES, (-0.006, 0.006)):
    p = pred[f"score_{m}"]
    bins = pd.qcut(p, N_BINS, duplicates="drop")
    tab = pred.groupby(bins, observed=True).agg(pred_mean=(f"score_{m}", "mean"), obs=("fail", "mean"),
                                                k=("fail", "sum"), n=("fail", "size"))
    lo, hi = (np.array(v) for v in zip(*[wilson(k, n) for k, n in zip(tab["k"], tab["n"])]))
    brier = np.mean((p - y) ** 2)
    ratios.append(p.mean() / y.mean())
    print(f"   {name}: mean predicted {p.mean():.2f} vs actual failure rate {y.mean():.2f}; Brier score {brier:.3f}")
    print(tab[["pred_mean", "obs", "n"]].round(2).to_string(index=False, header=["predicted", "observed", "samples"]))
    x = tab["pred_mean"].to_numpy() + off
    ax.errorbar(x, tab["obs"], yerr=[tab["obs"] - lo, hi - tab["obs"]], fmt="o-", color=colour, ecolor=colour,
                elinewidth=1.5, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Predicted chance of failing (average within each group)")
ax.set_ylabel("Share that actually failed")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=10)
r = float(np.mean(ratios))
title = ("Predicted risk runs high: the models overstate it" if r > 1.25 else
         "Predicted risk runs low: the models understate it" if r < 0.8 else
         "Predicted risk is about right on average")
ax.set_title(title, pad=40)
subtitle(ax, f"2026 test season, {N_BINS} equal groups by predicted risk.\nWhiskers: 95% intervals.")
fig.tight_layout()
fig.savefig(FIG / "4_calibration_2026.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print("   (points below the diagonal = the model predicts more failures than happen)\n")

# ---------------------------------------------------------- 2. missed failures
fails = pred[y]
for m, label in [(BEST, "best model (logistic regression, all features)"), ("rain_rule", "rain rule")]:
    caught = fails[f"alert_{m}"]
    print(f"2. Failures caught vs missed by the {label}: caught {int(caught.sum())}, missed {int((~caught).sum())}")
    for col in ["rain_group", "severity", "which", "seen_site"]:
        t = fails.groupby(col, observed=True)[f"alert_{m}"].agg(failures="size", caught="sum")
        t["missed"] = t["failures"] - t["caught"]
        t["share caught"] = (t["caught"] / t["failures"]).map("{:.0%}".format)
        print(t.to_string())
    print()

missed_best = fails[~fails[f"alert_{BEST}"]]
site_t = missed_best.groupby("site").size().sort_values(ascending=False)
print(f"   Missed failures by site ({BEST}):")
print(site_t.to_string())

# --------------------------------------------------- 3. missed by every model
learned = ["rain_rule", "logreg_rain", "logreg_all", "forest_all"]
all_missed = fails[~fails[[f"alert_{m}" for m in learned]].any(axis=1)]
cols = ["site", "date", "rain_48h", "rain_6h", "rain_7d", "flow_rel", "ecoli", "ie", "which", "times_limit", "extra_sample"]
print(f"\n3. Failures missed by all four weather models: {len(all_missed)} of {len(fails)}")
print(all_missed[cols].sort_values("date").round(2).to_string(index=False))
dry = (all_missed["rain_48h"] <= 0.5).mean() if len(all_missed) else np.nan
print(f"   {dry:.0%} of these came after a dry 48 h: something other than recent rain caused them.")

# --------------------------------------------------------- 4. false alarms
fa = pred[pred[f"alert_{BEST}"] & ~y]
print(f"\n4. False alarms by the best model: {len(fa)} of {int(pred[f'alert_{BEST}'].sum())} alerts")
print(fa.groupby("site").size().sort_values(ascending=False).head(8).to_string())
print(fa.groupby("rain_group", observed=True).size().to_string())
near = (fa["times_limit"] >= 0.5).mean()
print(f"   {near:.0%} of false alarms were at least half the limit (polluted, just not over the line).")

missed_best[cols + [f"score_{BEST}"]].sort_values("date").to_csv("results/missed_failures.csv", index=False)
fa[cols + [f"score_{BEST}"]].sort_values("date").to_csv("results/false_alarms.csv", index=False)
print(f"\nSaved {FIG/'4_calibration_2026.png'}, results/missed_failures.csv, results/false_alarms.csv")
