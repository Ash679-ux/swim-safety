"""Do upstream spills predict failures beyond what rain already explains?

Run from the project root (after 08_features.py):
    python scripts/10_spills_experiment.py

Data: Thames Water sites from 2023 onwards (spill_data_ok), the only samples
with reliable spill history. There are only a handful of failures, so this is
asked as a statistical question rather than by comparing prediction scores.

1. Within similar rain: failure rates with and without a recent upstream
   spill, inside dry / light / wet rain groups.
2. Likelihood-ratio test: does adding spills improve a logistic regression that
   already knows rain? Reported with the spill odds ratio and its 95% CI.
   Repeated with flow added, as a check.
3. Permutation test: the LR statistic again, with spill values shuffled only
   among samples at the same site and in the same rain group. This keeps the
   link between rain and spills, so it asks "do spills add anything beyond
   rain?", and it doesn't rely on large-sample maths that ~10 failures can't
   support.
4. Which distance band and window carry the signal (exploratory: many variants
   on few failures, so read as a pattern, not a set of separate findings).
5. Prediction check: repeated cross-validation PR-AUC, rain-only vs rain + spills.

Writes results/spills_experiment.csv.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Some shuffles can't be fitted (too few failures); they are skipped and counted.
warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 0
N_PERM = 2000
MAIN = "spill_h_48h_25km"  # the pre-chosen main spill feature
RAIN_BINS = [-np.inf, 0.5, 4, np.inf]
RAIN_LABELS = ["dry (<= 0.5 mm)", "light (0.5-4 mm)", "wet (> 4 mm)"]
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv("data/processed/features.csv")
d = df[df["spill_data_ok"]].copy()
d["y"] = d["fail"].astype(int)
d["log_rain"] = np.log1p(d["rain_48h"].clip(lower=0))
d["log_flow"] = np.log(d["flow_rel"].clip(lower=1e-3))
d["rain_group"] = pd.cut(d["rain_48h"], RAIN_BINS, labels=RAIN_LABELS)
spill_cols = [c for c in d.columns if c.startswith("spill_h_")] + ["spill_weighted_72h"]
for c in spill_cols:
    d[f"log_{c}"] = np.log1p(d[c])
print(f"{len(d)} Thames samples (2023+) at {d['id'].nunique()} sites: "
      f"{int(d['y'].sum())} failures ({d['y'].mean():.1%})")
print(d.groupby("site")["y"].agg(samples="size", failures="sum").to_string(), "\n")

# ------------------------------------------------- 1. within similar rain
d["recent_spill"] = d[MAIN] > 0
print(f"1. Failure rate by rain group, with vs without an upstream spill ({MAIN} > 0):")
tab = d.groupby(["rain_group", "recent_spill"], observed=True)["y"].agg(samples="size", failures="sum", rate="mean")
print(tab.round(3).to_string())
both = d.groupby("recent_spill")["y"].agg(["sum", "size"])
if len(both) == 2:
    n_fail = {k: int(both.loc[k, "sum"]) for k in (True, False)}
    n_all = {k: int(both.loc[k, "size"]) for k in (True, False)}
    t = [[n_fail[True], n_all[True] - n_fail[True]],
         [n_fail[False], n_all[False] - n_fail[False]]]
    print(f"   Fisher exact test, spill vs no spill (ignoring rain): p = {stats.fisher_exact(t)[1]:.3f}")


# ---------------------------------------------- 2. likelihood-ratio test
def fit(y, X):
    """Logistic regression via statsmodels; returns None if it can't be fitted."""
    try:
        m = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)
        return m if m.mle_retvals.get("converged", True) else None
    except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError, ValueError):
        return None


def lr_test(data, base, spill):
    sub = data.dropna(subset=base + [spill])
    m0, m1 = fit(sub["y"], sub[base]), fit(sub["y"], sub[base + [spill]])
    if m0 is None or m1 is None:
        return None
    lr = 2 * (m1.llf - m0.llf)
    ci = np.exp(m1.conf_int().loc[spill])
    return {"n": len(sub), "failures": int(sub["y"].sum()), "LR": lr, "p_LR": stats.chi2.sf(lr, 1),
            "OR_per_log_unit": np.exp(m1.params[spill]), "OR_lo": ci.iloc[0], "OR_hi": ci.iloc[1],
            "AIC_base": m0.aic, "AIC_with_spill": m1.aic}


rows = []
print("\n2. Does adding spills improve a model that already knows rain?")
for label, base in [("rain", ["log_rain"]), ("rain + flow", ["log_rain", "log_flow"])]:
    r = lr_test(d, base, f"log_{MAIN}")
    if r is None:
        print(f"   {label}: model could not be fitted (too few failures / separation)")
        continue
    rows.append({"analysis": "LR test", "base": label, "spill_feature": MAIN, **r})
    print(f"   base = {label:12s} n={r['n']}, failures={r['failures']}: LR = {r['LR']:.2f}, p = {r['p_LR']:.3f}; "
          f"odds ratio per log(1 + spill hours) = {r['OR_per_log_unit']:.2f} "
          f"(95% CI {r['OR_lo']:.2f}-{r['OR_hi']:.2f}); AIC {r['AIC_base']:.1f} -> {r['AIC_with_spill']:.1f}")

# ---------------------------------------------------- 3. permutation test
print(f"\n3. Permutation test ({N_PERM} shuffles of spills within site x rain group):")
sub = d.dropna(subset=["log_rain", f"log_{MAIN}"]).reset_index(drop=True)
obs = lr_test(sub, ["log_rain"], f"log_{MAIN}")
if obs is not None:
    rng = np.random.default_rng(SEED)
    strata = sub.groupby(["id", "rain_group"], observed=True).indices
    null, x = [], sub[f"log_{MAIN}"].to_numpy()
    for _ in range(N_PERM):
        xp = x.copy()
        for idx in strata.values():
            xp[idx] = rng.permutation(x[idx])
        perm = sub.assign(**{f"log_{MAIN}": xp})
        r = lr_test(perm, ["log_rain"], f"log_{MAIN}")
        if r is not None:
            null.append(r["LR"])
    null = np.array(null)
    p_perm = (1 + (null >= obs["LR"]).sum()) / (1 + len(null))
    rows.append({"analysis": "permutation", "base": "rain", "spill_feature": MAIN,
                 "n": obs["n"], "failures": obs["failures"], "LR": obs["LR"], "p_perm": p_perm})
    print(f"   observed LR = {obs['LR']:.2f}; permutation p = {p_perm:.3f} ({len(null)} valid shuffles)")

# --------------------------------------------- 4. distance bands / windows
print("\n4. Which spill feature carries the signal? (exploratory, base = rain)")
band = []
for c in spill_cols:
    r = lr_test(d, ["log_rain"], f"log_{c}")
    if r is not None:
        band.append({"spill_feature": c, **r})
band = pd.DataFrame(band).sort_values("p_LR")
if len(band):
    print(band[["spill_feature", "LR", "p_LR", "OR_per_log_unit", "OR_lo", "OR_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rows += [{"analysis": "band", "base": "rain", **r} for r in band.to_dict("records")]

# -------------------------------------------------- 5. prediction check
print("\n5. Prediction check: repeated 5-fold CV PR-AUC (20 repeats)")
X0, X1 = ["log_rain"], ["log_rain", f"log_{MAIN}"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
ap0, ap1 = [], []
yv = sub["y"].to_numpy()
for tr, te in cv.split(sub, yv):
    if yv[te].sum() == 0:
        continue
    for cols, out in [(X0, ap0), (X1, ap1)]:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(sub.loc[tr, cols], yv[tr])
        out.append(average_precision_score(yv[te], m.predict_proba(sub.loc[te, cols])[:, 1]))
ap0, ap1 = np.array(ap0), np.array(ap1)
diff = ap1 - ap0
print(f"   rain only: {ap0.mean():.2f}   rain + spills: {ap1.mean():.2f}   "
      f"(random guessing: {yv.mean():.2f})")
print(f"   rain + spills better in {np.mean(diff > 0):.0%} of {len(diff)} folds; "
      f"mean gain {diff.mean():+.3f} (middle 95% of folds {np.percentile(diff, 2.5):+.3f} to {np.percentile(diff, 97.5):+.3f})")
rows.append({"analysis": "cv PR-AUC", "base": "rain", "spill_feature": MAIN,
             "n": len(sub), "failures": int(yv.sum()), "AP_rain": ap0.mean(), "AP_rain_spill": ap1.mean(),
             "AP_gain": diff.mean(), "share_folds_better": np.mean(diff > 0)})

pd.DataFrame(rows).to_csv(RESULTS / "spills_experiment.csv", index=False)
print(f"\nSaved {RESULTS/'spills_experiment.csv'}")
print("Reminder: with so few failures, report the confidence intervals; a non-significant result means "
      "'not enough data to tell', not 'spills don't matter'.")
