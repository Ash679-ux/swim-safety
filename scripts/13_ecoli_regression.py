"""Predict log10 E. coli as a number, then use it as a failure score.

Run from the project root (after 09_models.py):
    python scripts/13_ecoli_regression.py

Why: the classifiers in 09 only see pass/fail, so a sample at 900 counts the
same as one at 20, and one at 1,100 the same as one at 50,000. A regression on
log10 E. coli learns from how dirty every sample was. If that extra
information helps, ranking 2026 samples by predicted E. coli should beat
logreg_all, which has the same features but a pass/fail target.

Model: linear regression (ridge, light penalty) on exactly the logreg_all
features with the same preprocessing, so the only difference is the target.
Scored on 2026 like 09: PR-AUC for failures (E. coli OR enterococci), with the
same paired bootstrap against the rain rule and logreg_all. Also reports how
well it predicts the count itself (R2, typical error in log10 units).

Second target: log10 of whichever bacterium is furthest over its limit
(larger of E. coli/1000 and enterococci/400), because about a quarter of 2026
failures were on enterococci alone and an E. coli target can't see them.

Writes results/ecoli_regression.csv.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from funcs import fast_ap, site_rate_by_other_seasons, site_rate_map

SEED = 0
ECOLI_LIMIT, IE_LIMIT = 1000, 400
RAIN = ["rain_6h", "rain_24h", "rain_48h", "rain_72h", "rain_7d"]
FLOW = ["flow_rel", "flow_change_24h"]
FEATS = RAIN + FLOW + ["month", "day_of_year", "site_rate"]  # = logreg_all in 09

df = pd.read_csv("data/processed/features.csv")
train, test = df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()
rate, g = site_rate_map(train)
train["site_rate"] = site_rate_by_other_seasons(train)
test["site_rate"] = test["id"].map(rate).fillna(g)
# ponytail: "<10"-style counts are taken at their reported value; a censored
# (Tobit) regression would treat them as "10 or less". Matters only for very
# clean samples, which don't change the ranking near the limit much.
# Two targets: log10 E. coli (as first planned), and log10 of whichever bacterium is
# furthest over its limit (the prev_exceed formula: > 0 means a fail), which also
# sees the failures on enterococci alone.
for d in (train, test):
    d["log_ecoli"] = np.log10(d["ecoli"].clip(lower=1))
    d["log_exceed"] = np.log10(np.maximum(d["ecoli"] / ECOLI_LIMIT, d["ie"] / IE_LIMIT).clip(lower=1e-3))
TARGETS = {"ecoli_regression": ("log_ecoli", np.log10(ECOLI_LIMIT)), "exceed_regression": ("log_exceed", 0.0)}

prep = ColumnTransformer([
    ("skewed", make_pipeline(SimpleImputer(strategy="median"),
                             FunctionTransformer(lambda X: np.log1p(np.clip(X, 0, None)))), RAIN + FLOW),
    ("other", SimpleImputer(strategy="median"), ["month", "day_of_year", "site_rate"])])
y = test["fail"].astype(int).to_numpy()
print(f"Test 2026: {len(test)} samples, {y.sum()} failures "
      f"({int((test['fail'] & ~test['fail_ecoli']).sum())} of them on enterococci only, invisible to the E. coli target)")
preds = {}
for name, (col, _) in TARGETS.items():
    preds[name] = make_pipeline(prep, StandardScaler(), Ridge(alpha=1.0)).fit(train[FEATS], train[col]).predict(test[FEATS])
    err = np.abs(preds[name] - test[col].to_numpy())
    print(f"{name}: R2 {r2_score(test[col], preds[name]):.2f} (training-mean baseline "
          f"{r2_score(test[col], np.full(len(test), train[col].mean())):.2f}), "
          f"median error {np.median(err):.2f} log10 units (x{10 ** np.median(err):.1f})")

# Same scores and paired bootstrap as 09 (same seed, so the same resamples).
old = pd.read_csv("results/test_predictions.csv")
key = ["id", "sample_time_utc"]
old = test[key].merge(old, on=key, how="left")
assert old["score_rain_rule"].notna().all(), "test rows don't match results/test_predictions.csv: rerun 09"
scores = {**preds, "rain_rule": old["score_rain_rule"].to_numpy(), "logreg_all": old["score_logreg_all"].to_numpy()}
rng = np.random.default_rng(SEED)
boot = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if y[i].min() != y[i].max()]
boots = {k: np.array([fast_ap(y[i], s[i]) for i in boot]) for k, s in scores.items()}

rows = []
print(f"\nPR-AUC for failures on 2026 (random guessing: {y.mean():.2f}):")
for k, s in scores.items():
    lo, hi = np.percentile(boots[k], [2.5, 97.5])
    rows.append({"model": k, "PR_AUC": average_precision_score(y, s), "lo": lo, "hi": hi})
    print(f"  {k:17s} {rows[-1]['PR_AUC']:.2f} ({lo:.2f} to {hi:.2f})")
print("\nPaired differences (95% range over the same resamples):")
for a, b in [(r, o) for r in TARGETS for o in ["rain_rule", "logreg_all"]] + [("exceed_regression", "ecoli_regression")]:
    d = boots[a] - boots[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "can't tell apart"
    print(f"  {a} vs {b:16s} {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f}); better in {(d > 0).mean():.0%} -> {verdict}")
    rows.append({"model": f"{a}_minus_{b}", "PR_AUC": d.mean(), "lo": lo, "hi": hi})

# A plain alert: predicted value over the limit. No threshold tuning at all.
print()
for name, (_, limit) in TARGETS.items():
    flag = preds[name] > limit
    hit = (flag & (y == 1)).sum()
    print(f"Alert if {name} predicts over the limit: {flag.sum()} alerts, {hit} of them real failures "
          f"(catches {hit} of {y.sum()})")

pd.DataFrame(rows).to_csv("results/ecoli_regression.csv", index=False)
print("\nSaved results/ecoli_regression.csv")
