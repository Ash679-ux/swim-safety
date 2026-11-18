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

Calibration: the models are trained with balanced class weights, which
inflate probabilities. A logistic (Platt) recalibration is fitted on the
leave-one-season-out training predictions and applied to the test scores.
It doesn't change the ranking (so PR-AUC is unchanged), only the probabilities.
95% intervals come from 1000 bootstrap resamples of the test set.

Writes results/model_results.csv and results/test_predictions.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
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

# Site failure rate, learned from training labels only. Training rows get a
# leave-one-out version so a row's own label never feeds its own feature.
g = y_tr.mean()
stats = train.groupby("id")["fail"].agg(["sum", "size"])
train["site_rate"] = [(stats.loc[s, "sum"] - yi + PRIOR_STRENGTH * g) / (stats.loc[s, "size"] - 1 + PRIOR_STRENGTH)
                      for s, yi in zip(train["id"], y_tr)]
full_rate = ((stats["sum"] + PRIOR_STRENGTH * g) / (stats["size"] + PRIOR_STRENGTH)).to_dict()
test["site_rate"] = test["id"].map(full_rate).fillna(g)  # unseen sites get the overall rate
test["seen_site"] = test["id"].isin(stats.index)


def log_tf():
    # rain and flow are heavily skewed; log1p keeps big storms from dominating
    return FunctionTransformer(lambda X: np.log1p(np.clip(X, 0, None)), feature_names_out="one-to-one")


def logreg():
    return make_pipeline(SimpleImputer(strategy="median"), log_tf(), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def forest():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))


def gboost(feats):
    # +1 = predicted risk may only rise as the feature rises (rain, site history)
    mono = [1 if f in RAIN + SITE else 0 for f in feats]
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          monotonic_cst=mono, class_weight="balanced", random_state=SEED)


# Leave-one-season-out: each training season is predicted by a model that never saw it
cv = LeaveOneGroupOut()
groups = train["year"].to_numpy()


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


def bootstrap_ci(y, s, metric, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in tqdm(range(n), desc="  uncertainty ranges", unit="resample", leave=False):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def evaluate(name, s_tr_oof, s_te, features):
    thr = best_threshold(y_tr, s_tr_oof)
    pred = (s_te >= thr).astype(int)
    pr, pr_ci = average_precision_score(y_te, s_te), bootstrap_ci(y_te, s_te, average_precision_score)
    row = {"model": name, "features": ", ".join(features),
           "PR_AUC": pr, "PR_AUC_lo": pr_ci[0], "PR_AUC_hi": pr_ci[1],
           "ROC_AUC": roc_auc_score(y_te, s_te), "threshold": thr,
           "precision": precision_score(y_te, pred, zero_division=0),
           "recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
           "flagged": int(pred.sum()), "caught": int((pred & y_te).sum()), "failures": int(y_te.sum())}
    for label, mask in [("seen", test["seen_site"].to_numpy()), ("new", ~test["seen_site"].to_numpy())]:
        if y_te[mask].min() != y_te[mask].max():
            row[f"PR_AUC_{label}_sites"] = average_precision_score(y_te[mask], s_te[mask])
    test[f"score_{name}"] = s_te
    return row


rows = []
bar = tqdm(total=2 + 4, desc="Models", unit="model")
# 1. Site failure rate only
rows.append(evaluate("site_rate", train["site_rate"].to_numpy(), test["site_rate"].to_numpy(), SITE))
# 2. Rain rule: score is rain in the last 48 h; the threshold IS the rule's X mm
r48_tr, r48_te = train["rain_48h"].fillna(0).to_numpy(), test["rain_48h"].fillna(0).to_numpy()
bar.update(1)
rows.append(evaluate("rain_rule", r48_tr, r48_te, ["rain_48h"]))
bar.update(1)
# 3-5. Learned models
ALL = RAIN + FLOW + SEASON + SITE
models = {"logreg_rain": (logreg, RAIN + SEASON),
          "logreg_all": (logreg, ALL),
          "forest_all": (forest, ALL),
          "gboost_all": (lambda: gboost(ALL), ALL)}
fitted, calib = {}, []
for name, (make, feats) in models.items():
    bar.set_postfix_str(name)
    oof = cross_val_predict(make(), train[feats], y_tr, cv=cv, groups=groups, method="predict_proba")[:, 1]
    m = make().fit(train[feats], y_tr)
    fitted[name] = (m, feats)
    s_te = m.predict_proba(test[feats])[:, 1]
    rows.append(evaluate(name, oof, s_te, feats))
    cal = platt(oof, y_tr)
    test[f"prob_{name}"] = cal(s_te)
    calib.append({"model": name, "mean_raw": s_te.mean(), "mean_calibrated": test[f"prob_{name}"].mean(),
                  "actual": y_te.mean(), "brier_raw": np.mean((s_te - y_te) ** 2),
                  "brier_calibrated": np.mean((test[f"prob_{name}"] - y_te) ** 2)})
    bar.update(1)
bar.close()

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"\nTest season 2026: {int(y_te.sum())} failures in {len(y_te)} samples "
      f"(random guessing gives PR-AUC {y_te.mean():.2f}, ROC-AUC 0.50)\n")
show = res[["model", "PR_AUC", "PR_AUC_lo", "PR_AUC_hi", "ROC_AUC", "precision", "recall", "F1",
            "flagged", "caught", "PR_AUC_seen_sites", "PR_AUC_new_sites"]]
print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\nCalibration on 2026 (mean predicted probability vs actual failure rate; Brier: lower is better):")
print(pd.DataFrame(calib).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("  Recalibration is fitted on the training years, so it can't correct for 2026 being a cleaner year.")
rule_mm = res.loc[res["model"] == "rain_rule", "threshold"].iloc[0]
print(f"\nRain rule chosen on training data: fail if more than {rule_mm:.1f} mm fell in the 48 h before sampling")

# What drives the best learned model? Permutation importance on the test set.
best = res[res["model"].isin(models)].sort_values("PR_AUC", ascending=False)["model"].iloc[0]
m, feats = fitted[best]
print(f"\nWorking out feature importance for {best} (30 shuffles per feature, may take a minute)...")
imp = permutation_importance(m, test[feats], y_te, scoring="average_precision", n_repeats=30, random_state=SEED)
order = np.argsort(-imp.importances_mean)
print(f"\nPermutation importance for {best} (drop in test PR-AUC when the feature is shuffled):")
for i in order:
    print(f"  {feats[i]:16s} {imp.importances_mean[i]:+.3f} ± {imp.importances_std[i]:.3f}")

res.to_csv(RESULTS / "model_results.csv", index=False)
keep = ["id", "site", "sample_time_utc", "fail", "ecoli", "ie", "seen_site", "rain_48h", "flow_rel", "site_rate"]
test[keep + [c for c in test.columns if c.startswith(("score_", "prob_"))]].to_csv(RESULTS / "test_predictions.csv", index=False)
print(f"\nSaved {RESULTS/'model_results.csv'} and {RESULTS/'test_predictions.csv'}")
