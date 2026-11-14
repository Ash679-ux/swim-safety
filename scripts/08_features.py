"""Day 4: one row of features per sample, using only data from before the sample.

Run from the project root (after scripts 04-07):
    python scripts/08_features.py

Features (all windows END at the sample time t, minus one 15-minute reading
interval, so no reading that might cover time after t is used):
  rain_6h, rain_24h, rain_48h, rain_72h, rain_7d
        mm, averaged over the site's rain gauges; a gauge counts only if at
        least 80% of its readings in the window are present
  flow, flow_rel, flow_change_24h
        river flow (m3/s) at t, flow relative to the station's median flow in
        the training years (so big and small rivers are comparable), and the
        ratio of flow now to flow 24 h earlier
  month, day_of_year
  Thames sites only (spill history exists only for Thames Water):
  spill_h_{24,48,72}h_{10,25,50}km
        hours of spilling, summed over upstream overflows within 10/25/50 km
        along the river, overlapping the window before t
  spilling_now_{10,25,50}km    overflows spilling at time t
  hours_since_spill_{10,25,50}km   capped at 30 days
  spill_weighted_72h   spill hours in the last 72 h, each overflow weighted by
                       exp(-river_km / 10), so nearer spills count more
  offline_frac_72h_25km   share of upstream overflows within 25 km that were
                          offline at some point in the last 72 h
  spill_data_ok        False before 2023 (most monitors weren't connected yet)

A leakage test at the end deletes every reading after t for a random set of
samples, recomputes their features and checks they are identical.

Writes data/processed/features.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

INTERIM, OUT = Path("data/interim"), Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
TEST_YEAR = 2026
STEP = np.timedelta64(15, "m").astype("timedelta64[ns]").astype(np.int64)
HOUR = 3600 * 10**9
RAIN_WINDOWS = {"6h": 6, "24h": 24, "48h": 48, "72h": 72, "7d": 168}
SPILL_WINDOWS, BANDS = (24, 48, 72), (10, 25, 50)
MIN_COVERAGE = 0.8
FLOW_MAX_AGE_H = 2
SINCE_CAP_H = 30 * 24
DECAY_KM = 10
N_LEAK_TESTS = 25


def ns(s):
    """UTC timestamps -> int64 nanoseconds (NaT -> int64 max)."""
    t = pd.to_datetime(pd.Series(s), utc=True, format="ISO8601").dt.as_unit("ns")  # pandas 3 defaults to us
    out = t.astype("int64").to_numpy(np.int64, copy=True)
    out[t.isna().to_numpy()] = np.iinfo(np.int64).max
    return out


def build_series(df, value_col):
    """Per station: sorted times, cumulative sums and cumulative counts of present values."""
    out = {}
    for st, g in df.groupby("station"):
        g = g.sort_values("t")
        v = g[value_col].to_numpy(float)
        ok = ~np.isnan(v)
        out[st] = {"t": g["t"].to_numpy(np.int64), "v": v,
                   "cum": np.concatenate([[0], np.cumsum(np.where(ok, v, 0))]),
                   "cnt": np.concatenate([[0], np.cumsum(ok)])}
    return out


def truncate(series, t):
    """The station series as it would have been known at time t (leakage test)."""
    out = {}
    for st, s in series.items():
        k = np.searchsorted(s["t"], t, "right")
        out[st] = {"t": s["t"][:k], "v": s["v"][:k], "cum": s["cum"][:k + 1], "cnt": s["cnt"][:k + 1]}
    return out


def rain_total(s, t, hours):
    end = t - STEP
    i1 = np.searchsorted(s["t"], end, "right")
    i0 = np.searchsorted(s["t"], end - hours * HOUR, "right")
    if (s["cnt"][i1] - s["cnt"][i0]) < MIN_COVERAGE * hours * 4:
        return np.nan
    return s["cum"][i1] - s["cum"][i0]


def flow_at(s, t):
    i = np.searchsorted(s["t"], t, "right") - 1
    if i < 0 or t - s["t"][i] > FLOW_MAX_AGE_H * HOUR:
        return np.nan
    return s["v"][i]


def site_features(t, gauges, rain, flow_st, flow, flow_median, spills, links_km, offline):
    """Features at time t. Spills/offline periods with no Stop yet have end = int64 max."""
    f = {}
    for name, h in RAIN_WINDOWS.items():
        vals = [rain_total(rain[g], t, h) for g in gauges if g in rain]
        vals = [v for v in vals if not np.isnan(v)]
        f[f"rain_{name}"] = float(np.mean(vals)) if vals else np.nan
    if flow_st is not None and flow_st in flow:
        now, before = flow_at(flow[flow_st], t), flow_at(flow[flow_st], t - 24 * HOUR)
        f["flow"] = now
        f["flow_rel"] = now / flow_median[flow_st] if flow_median.get(flow_st) else np.nan
        f["flow_change_24h"] = now / before if before and before > 0 else np.nan
    else:
        f["flow"] = f["flow_rel"] = f["flow_change_24h"] = np.nan
    if spills is not None:
        known = spills[spills["start"] <= t]
        end = np.minimum(known["end"].to_numpy(), t)  # as known at t
        start = known["start"].to_numpy()
        km = known["river_km"].to_numpy()
        for b in BANDS:
            inb = km <= b
            for w in SPILL_WINDOWS:
                overlap = np.clip(end - np.maximum(start, t - w * HOUR), 0, None)
                f[f"spill_h_{w}h_{b}km"] = overlap[inb].sum() / HOUR
            ongoing = known["end"].to_numpy() > t
            f[f"spilling_now_{b}km"] = int(known.loc[inb & ongoing, "permitNumber"].nunique())
            if inb.any():
                last = 0 if (inb & ongoing).any() else (t - end[inb].max()) / HOUR
                f[f"hours_since_spill_{b}km"] = min(last, SINCE_CAP_H)
            else:
                f[f"hours_since_spill_{b}km"] = SINCE_CAP_H
        overlap72 = np.clip(end - np.maximum(start, t - 72 * HOUR), 0, None)
        f["spill_weighted_72h"] = float((overlap72 * np.exp(-km / DECAY_KM)).sum() / HOUR)
        near = links_km[links_km <= 25].index
        off = offline[(offline["permitNumber"].isin(near)) & (offline["start"] <= t)
                      & (offline["end"] > t - 72 * HOUR)]
        f["offline_frac_72h_25km"] = off["permitNumber"].nunique() / len(near) if len(near) else np.nan
    return f


# ------------------------------------------------------------------- load
samples = pd.read_csv(INTERIM / "samples.csv")
samples["t"] = ns(samples["sample_time_utc"])
sites = pd.read_csv(INTERIM / "sites.csv")
stations = pd.read_csv(INTERIM / "stations.csv")
print("Loading rain and flow readings (a few million rows)...")
rain_df = pd.read_csv(INTERIM / "rain_15min.csv")
rain_df["t"] = ns(rain_df["time_utc"])
flow_df = pd.read_csv(INTERIM / "flow_15min.csv")
flow_df["t"] = ns(flow_df["time_utc"])
rain, flow = build_series(rain_df, "mm"), build_series(flow_df, "m3s")
train_flow = flow_df[pd.to_datetime(flow_df["time_utc"], utc=True, format="ISO8601").dt.year < TEST_YEAR]
flow_median = train_flow.groupby("station")["m3s"].median().to_dict()

spills = pd.read_csv(INTERIM / "spills.csv")
spills["start"], spills["end"] = ns(spills["start_utc"]), ns(spills["end_utc"])
spills = spills[spills["end"] > spills["start"]]  # drop zero-length spills
offline = pd.read_csv(INTERIM / "offline.csv")
offline["start"], offline["end"] = ns(offline["start_utc"]), ns(offline["end_utc"])
links = pd.read_csv(INTERIM / "upstream_links.csv")
thames_ids = set(links["id"])
print(f"{len(samples)} samples at {samples['id'].nunique()} sites; spill features for {len(thames_ids)} Thames sites")

# Per-site lookups
gauges = stations[stations["kind"] == "rain"].groupby("id")["station"].apply(list).to_dict()
flow_station = stations[stations["kind"] == "flow"].set_index("id")["station"].to_dict()
site_spills, site_links = {}, {}
for sid, g in links.groupby("id"):
    km = g.set_index("permitNumber")["river_km"]
    site_links[sid] = km
    sp = spills[spills["permitNumber"].isin(km.index)].copy()
    sp["river_km"] = sp["permitNumber"].map(km)
    site_spills[sid] = sp

# ---------------------------------------------------------------- compute
def features_for(row, rain_, flow_, spills_override=None, offline_override=None):
    sid = row["id"]
    sp = spills_override if spills_override is not None else site_spills.get(sid)
    off = offline_override if offline_override is not None else offline
    return site_features(row["t"], gauges.get(sid, []), rain_, flow_station.get(sid), flow_,
                         flow_median, sp, site_links.get(sid), off)


def as_known_at(events, t):
    """Events as they would have been known at time t: started by t, and any
    Stop that came after t not yet received (end unknown)."""
    ev = events[events["start"] <= t].copy()
    ev.loc[ev["end"] > t, "end"] = np.iinfo(np.int64).max
    return ev


rows = []
for _, r in tqdm(samples.iterrows(), total=len(samples), desc="Building features", unit="sample"):
    rows.append(features_for(r, rain, flow))
feats = pd.DataFrame(rows, index=samples.index)

local = pd.to_datetime(samples["sample_time_utc"], utc=True, format="ISO8601").dt.tz_convert("Europe/London")
out = pd.concat([samples[["id", "site", "sample_time_utc", "year", "ecoli", "ie", "fail",
                          "fail_ecoli", "fail_ie", "discountable", "extra_sample"]], feats], axis=1)
out["month"], out["day_of_year"] = local.dt.month, local.dt.dayofyear
out["thames"] = out["id"].isin(thames_ids)
out["spill_data_ok"] = out["thames"] & (local.dt.year >= 2023)
out["split"] = np.where(out["year"] >= TEST_YEAR, "test", "train")

# ------------------------------------------------------------ leakage test
rng = np.random.default_rng(0)
test_idx = rng.choice(samples.index, size=min(N_LEAK_TESTS, len(samples)), replace=False)
thames_rows = samples.index[samples["id"].isin(thames_ids)]
if len(thames_rows):
    test_idx = np.unique(np.concatenate([test_idx, rng.choice(thames_rows, size=min(10, len(thames_rows)), replace=False)]))
bad = 0
for i in tqdm(test_idx, desc="Leakage test", unit="sample"):
    r = samples.loc[i]
    t = r["t"]
    sp = site_spills.get(r["id"])
    sp = as_known_at(sp, t) if sp is not None else None
    again = features_for(r, truncate(rain, t), truncate(flow, t), sp, as_known_at(offline, t))
    a, b = pd.Series(again), feats.loc[i, list(again)]
    same = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
    if not same.all():
        bad += 1
        print(f"  ! sample {i}: features changed when future data was removed: {list(a.index[~same])}")
print(f"Leakage test: {len(test_idx) - bad}/{len(test_idx)} samples identical with all later data removed")
if bad:
    raise SystemExit("Leakage detected - fix before modelling.")

# ----------------------------------------------------------------- report
fcols = [c for c in feats.columns]
print("\nMissing values per feature (% of rows where it applies):")
for c in fcols:
    base = out[out["thames"]] if c.startswith(("spill", "spilling", "hours_since", "offline")) else out
    print(f"  {c:26s} {base[c].isna().mean():6.1%}")
print("\nFailure rate by 48 h rainfall (sanity check: should rise with rain):")
q = pd.qcut(out["rain_48h"], 4, duplicates="drop")
print(out.groupby(q, observed=True)["fail"].agg(["size", "mean"]).round(3).to_string())
th = out[out["spill_data_ok"]]
if len(th):
    print("\nThames samples (2023+), failure rate by whether any upstream spill within 25 km in the last 48 h:")
    print(th.groupby(th["spill_h_48h_25km"] > 0)["fail"].agg(["size", "mean"]).round(3).to_string())
print(f"\nRows: train {int((out['split'] == 'train').sum())}, test {int((out['split'] == 'test').sum())}; "
      f"Thames with reliable spill data: {int(out['spill_data_ok'].sum())}")
out.to_csv(OUT / "features.csv", index=False)
print(f"Saved {OUT/'features.csv'}")
