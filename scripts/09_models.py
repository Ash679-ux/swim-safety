"""Main model on all 20 river sites (rain, flow, season, site).

Run from the project root (after 08_features.py):
    python scripts/09_models.py

Split by time: train on 2021-2025, test on the unseen 2026 season.
Six sites were first sampled in 2026, so the test also shows how the models do
on sites they have never seen (reported separately).

Models, from simplest to most complex:
  site_rate      each site's failure rate in the training years (no weather at all)
  rain_rule      "fail if rain in the last 48 h > X mm", X chosen on training data
  logreg_rain    logistic regression on rain + season only
  logreg_all     logistic regression on rain + flow + season + site rate
  forest_all     random forest on the same features
  gboost_all     gradient boosting on the same features, constrained so that more
                 rain (and a dirtier site history) can only raise the predicted risk

Failures are the minority (about 1 in 5), so models are judged on:
  PR-AUC     area under the precision-recall curve (random guessing scores the
             test failure rate, printed with the results)
  ROC-AUC    0.5 = guessing at random
  precision / recall / F1 for failures, at a threshold chosen on the TRAINING
  data, never tuned on the test set.
Thresholds use leave-one-season-out predictions on the training years (each
season predicted by a model that never saw it), matching the test design.

Site failure rate: each training row's rate comes from the OTHER training
seasons only (rebuilt inside every cross-validation fold); test rows use all
training seasons. A leave-one-out rate would leak the label (see
site_rate_by_other_seasons). Models are compared with the rain rule using a
paired bootstrap (same test resamples for both), and the "best" model for
feature importance is chosen by cross-validated training PR-AUC, not test.

Calibration: the models are trained with balanced class weights, which
inflate probabilities. A logistic (Platt) recalibration is fitted on the
leave-one-season-out training predictions and applied to the test scores.
It doesn't change the ranking (so PR-AUC is unchanged), only the probabilities.
95% intervals come from 1000 bootstrap resamples of the test set.

Writes results/model_results.csv, results/model_vs_rain_rule.csv and
results/test_predictions.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tqdm import tqdm

SEED = 0
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
PRIOR_STRENGTH = 10  # smoothing for the site failure rate (in samples)

RAIN = ["rain_6h", "rain_24h", "rain_48h", "rain_72h", "rain_7d"]
FLOW = ["flow_rel", "flow_change_24h"]
SEASON = ["month", "day_of_year"]
SITE = ["site_rate"]

df = pd.read_csv("data/processed/features.csv")
train, test = df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()
y_tr, y_te = train["fail"].astype(int).to_numpy(), test["fail"].astype(int).to_numpy()
print(f"Train {len(train)} samples ({y_tr.mean():.1%} fail), test {len(test)} ({y_te.mean():.1%} fail)")

def site_rate_map(fit):
    """Smoothed failure rate per site, learned from the rows in `fit`.
    Returns (rate per site, overall rate for sites not in `fit`)."""
    g = fit["fail"].mean()
    st = fit.groupby("id")["fail"].agg(["sum", "size"])
    return ((st["sum"] + PRIOR_STRENGTH * g) / (st["size"] + PRIOR_STRENGTH)).to_dict(), g


def site_rate_by_other_seasons(rows):
    """Each row's site rate learned only from OTHER seasons, so no row's own
    season (and so never its own label) feeds its feature. A leave-one-out rate
    would leak: failing rows would always get a slightly lower rate than passing
    rows at the same site, which tree models learn as a fake pattern."""
    out = pd.Series(np.nan, index=rows.index)
    overall = rows["fail"].mean()
    for yr in rows["year"].unique():
        others = rows[rows["year"] != yr]
        mask = rows["year"] == yr
        if others.empty:  # only one season available: no site history to use
            out[mask] = overall
            continue
        rate, g = site_rate_map(others)
        out[mask] = rows.loc[mask, "id"].map(rate).fillna(g)
    return out


rate, g = site_rate_map(train)
train["site_rate"] = site_rate_by_other_seasons(train)
test["site_rate"] = test["id"].map(rate).fillna(g)  # unseen sites get the overall rate
test["seen_site"] = test["id"].isin(train["id"].unique())


def log1p_pos(X):
    return np.log1p(np.clip(X, 0, None))


def logreg(feats):
    # Rain and flow are heavily skewed: log1p keeps big storms from dominating.
    # Season and site rate are left on their own scale.
    skewed = [f for f in feats if f in RAIN + FLOW]
    other = [f for f in feats if f not in skewed]
    prep = ColumnTransformer([
        ("skewed", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(log1p_pos)), skewed),
        ("other", SimpleImputer(strategy="median"), other)])
    return make_pipeline(prep, StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def forest(feats):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))


def gboost(feats):
    # +1 = predicted risk may only rise as the feature rises (rain, site history)
    mono = [1 if f in RAIN + SITE else 0 for f in feats]
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          monotonic_cst=mono, class_weight="balanced", random_state=SEED)


def season_oof(make, feats):
    """Leave-one-season-out predictions on the training years. Inside each fold
    the site rate is rebuilt from the fold's own training seasons only, so the
    held-out season's labels never reach the model or its features."""
    oof = np.zeros(len(train))
    for yr in sorted(train["year"].unique()):
        ho = (train["year"] == yr).to_numpy()
        fit, held = train[~ho].copy(), train[ho].copy()
        if "site_rate" in feats:
            fit["site_rate"] = site_rate_by_other_seasons(fit)
            r, gg = site_rate_map(fit)
            held["site_rate"] = held["id"].map(r).fillna(gg)
        m = make(feats).fit(fit[feats], fit["fail"].astype(int))
        oof[ho] = m.predict_proba(held[feats])[:, 1]
    return oof


def site_rate_oof():
    """Leave-one-season-out version of the site-rate-only baseline."""
    oof = np.zeros(len(train))
    for yr in sorted(train["year"].unique()):
        ho = (train["year"] == yr).to_numpy()
        r, gg = site_rate_map(train[~ho])
        oof[ho] = train.loc[ho, "id"].map(r).fillna(gg).to_numpy()
    return oof


def platt(oof, y):
    """Fit a recalibration map (logistic on the logit of the score)."""
    z = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression().fit(z.reshape(-1, 1), y)

    def apply(p):
        zz = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return cal.predict_proba(zz.reshape(-1, 1))[:, 1]
    return apply


def best_threshold(y, score):
    """Threshold maximising F1 on (out-of-fold) training scores."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return t[np.nanargmax(f1[:-1])]


# The same 1000 resamples of the test set for every model, so differences
# between models can be compared resample by resample (a paired bootstrap).
_rng = np.random.default_rng(SEED)
BOOT = [i for i in (_rng.integers(0, len(y_te), len(y_te)) for _ in range(1000)) if y_te[i].min() != y_te[i].max()]


def boot_scores(s):
    return np.array([average_precision_score(y_te[i], s[i])
                     for i in tqdm(BOOT, desc="  uncertainty ranges", unit="resample", leave=False)])


def evaluate(name, s_tr_oof, s_te, features):
    """Test-set metrics for one model. Returns (results row, bootstrap PR-AUCs)."""
    thr = best_threshold(y_tr, s_tr_oof)
    pred = (s_te >= thr).astype(int)
    boots = boot_scores(s_te)
    pr, pr_ci = average_precision_score(y_te, s_te), np.percentile(boots, [2.5, 97.5])
    row = {"model": name, "features": ", ".join(features), "PR_AUC_train_cv": average_precision_score(y_tr, s_tr_oof),
           "PR_AUC": pr, "PR_AUC_lo": pr_ci[0], "PR_AUC_hi": pr_ci[1],
           "ROC_AUC": roc_auc_score(y_te, s_te), "threshold": thr,
           "precision": precision_score(y_te, pred, zero_division=0),
           "recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
           "flagged": int(pred.sum()), "caught": int((pred & y_te).sum()), "failures": int(y_te.sum())}
    for label, mask in [("seen", test["seen_site"].to_numpy()), ("new", ~test["seen_site"].to_numpy())]:
        if y_te[mask].min() != y_te[mask].max():
            row[f"PR_AUC_{label}_sites"] = average_precision_score(y_te[mask], s_te[mask])
    return row, boots


rows, boots, scores = [], {}, {}


def record(name, oof, s_te, feats):
    row, b = evaluate(name, oof, s_te, feats)
    rows.append(row)
    boots[name], scores[name] = b, s_te


bar = tqdm(total=6, desc="Models", unit="model")
# 1. Site failure rate only
record("site_rate", site_rate_oof(), test["site_rate"].to_numpy(), SITE)
bar.update(1)
# 2. Rain rule: score is rain in the last 48 h; the threshold IS the rule's X mm
r48_tr, r48_te = train["rain_48h"].fillna(0).to_numpy(), test["rain_48h"].fillna(0).to_numpy()
record("rain_rule", r48_tr, r48_te, ["rain_48h"])
bar.update(1)
# 3-5. Learned models
ALL = RAIN + FLOW + SEASON + SITE
models = {"logreg_rain": (logreg, RAIN + SEASON),
          "logreg_all": (logreg, ALL),
          "forest_all": (forest, ALL),
          "gboost_all": (gboost, ALL)}
fitted, calib, probs = {}, [], {}
for name, (make, feats) in models.items():
    bar.set_postfix_str(name)
    oof = season_oof(make, feats)
    m = make(feats).fit(train[feats], y_tr)
    fitted[name] = (m, feats)
    s_te = m.predict_proba(test[feats])[:, 1]
    record(name, oof, s_te, feats)
    probs[name] = platt(oof, y_tr)(s_te)
    calib.append({"model": name, "mean_raw": s_te.mean(), "mean_calibrated": probs[name].mean(),
                  "actual": y_te.mean(), "brier_raw": np.mean((s_te - y_te) ** 2),
                  "brier_calibrated": np.mean((probs[name] - y_te) ** 2)})
    bar.update(1)
bar.close()

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"\nTest season 2026: {int(y_te.sum())} failures in {len(y_te)} samples "
      f"(random guessing gives PR-AUC {y_te.mean():.2f}, ROC-AUC 0.50)\n")
show = res[["model", "PR_AUC_train_cv", "PR_AUC", "PR_AUC_lo", "PR_AUC_hi", "ROC_AUC", "precision", "recall", "F1",
            "flagged", "caught", "PR_AUC_seen_sites", "PR_AUC_new_sites"]]
print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\nCalibration on 2026 (mean predicted probability vs actual failure rate; Brier: lower is better):")
print(pd.DataFrame(calib).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("  Recalibration is fitted on the training years, so it can't correct for 2026 being a cleaner year.")
# Paired bootstrap: is a model really better than the rain rule? Same resamples for both.
print("\nPaired comparison with the rain rule (difference in PR-AUC, 95% range over the same resamples):")
diffs = []
for name in [r["model"] for r in rows if r["model"] != "rain_rule"]:
    d = boots[name] - boots["rain_rule"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    diffs.append({"model": name, "diff_vs_rain_rule": d.mean(), "lo": lo, "hi": hi, "share_resamples_better": (d > 0).mean()})
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "can't tell apart"
    print(f"  {name:12s} {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f}); better in {(d > 0).mean():.0%} of resamples -> {verdict}")
pd.DataFrame(diffs).to_csv(RESULTS / "model_vs_rain_rule.csv", index=False)
rule_mm = res.loc[res["model"] == "rain_rule", "threshold"].iloc[0]
print(f"\nRain rule chosen on training data: fail if more than {rule_mm:.1f} mm fell in the 48 h before sampling")

# What drives the best learned model? Chosen by cross-validated TRAINING PR-AUC,
# so no decision depends on the test set; importance is then measured on 2026.
best = res[res["model"].isin(models)].sort_values("PR_AUC_train_cv", ascending=False)["model"].iloc[0]
m, feats = fitted[best]
print(f"\nWorking out feature importance for {best} (30 shuffles per feature, may take a minute)...")
imp = permutation_importance(m, test[feats], y_te, scoring="average_precision", n_repeats=30, random_state=SEED)
order = np.argsort(-imp.importances_mean)
print(f"\nPermutation importance for {best} (drop in test PR-AUC when the feature is shuffled):")
for i in order:
    print(f"  {feats[i]:16s} {imp.importances_mean[i]:+.3f} ± {imp.importances_std[i]:.3f}")

res.to_csv(RESULTS / "model_results.csv", index=False)
keep = ["id", "site", "sample_time_utc", "fail", "ecoli", "ie", "seen_site", "rain_48h", "flow_rel", "site_rate"]
out = test[keep].copy()
for name, s_te in scores.items():
    out[f"score_{name}"] = s_te
for name, p in probs.items():
    out[f"prob_{name}"] = p
out.to_csv(RESULTS / "test_predictions.csv", index=False)
print(f"\nSaved {RESULTS/'model_results.csv'} and {RESULTS/'test_predictions.csv'}")
