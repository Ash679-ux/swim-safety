"""One row of features per sample, using only data from before the sample.

Run from the project root (after scripts 04-07):
    python scripts/08_features.py

Features (all windows END at the sample time t, minus one 15-minute reading
interval, so no reading that might cover time after t is used):
  rain_6h, rain_24h, rain_48h, rain_72h, rain_7d, rain_14d
        mm, averaged over the site's rain gauges; a gauge counts only if at
        least 80% of its readings in the window are present
  flow, flow_rel, flow_change_24h
        river flow (m3/s) at t, flow relative to the station's median flow in
        the training years (so big and small rivers are comparable), and the
        ratio of flow now to flow 24 h earlier
  flow_low, flow_high
        how far flow is below / above normal, on a log scale
        (max(0, -log flow_rel) and max(0, log flow_rel)), so the models can
        give low flow its own effect: less water to dilute steady sources
  temp_48h, temp_7d      mean air temperature (deg C) over the last 48 h / 7 days
  sun_24h                sunlight energy over the last 24 h (MJ/m2)
  sun_hours_24h          hours of bright sunshine over the last 24 h
        (weather from 05b_weather.py: UV in sunlight kills bacteria, and
        temperature affects how long they survive)
  prev_exceed, days_since_prev
        the site's last result that would already be known (taken at least
        2 days earlier, the lab turnaround, in the same season): log10 of how
        far over the limit it was (> 0 = it failed), and how long ago
  month, day_of_year
  Thames sites only (spill history exists only for Thames Water):
  spill_h_{24,48,72}h_{10,25,50}km
        hours of spilling, summed over upstream overflows within 10/25/50 km
        along the river, overlapping the window before t
  spilling_now_{10,25,50}km    overflows spilling at time t
  hours_since_spill_{10,25,50}km   capped at 30 days
  spill_dilution_48h_25km   spill_h_48h_25km divided by flow (hours per m3/s):
                       the same spill should matter less in a bigger river
  spill_weighted_72h   spill hours in the last 72 h, each overflow weighted by
                       exp(-river_km / 10), so nearer spills count more
  offline_frac_72h_25km   share of upstream overflows within 25 km that were
                          offline at some point in the last 72 h
  spill_data_ok        False before 2023 (most monitors weren't connected yet)

A leakage test at the end deletes every reading after t for a random set of
samples, recomputes their features and checks they are identical. It checks
the feature calculations themselves. It does not cover the one quantity worked
out in advance: each flow station's median flow, which uses all training
seasons (so a 2021 sample is scaled partly with 2025 flow). That uses no
labels and no test-season data, so it can't leak the answer, but it is
noted in the README.

Writes data/processed/features.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from funcs import (HOUR, RESULT_LAG_DAYS, as_known_at, build_series, flow_at, hourly_window,
                   previous_result, rain_total, spill_dilution, truncate)

INTERIM, OUT = Path("data/interim"), Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
TEST_YEAR = 2026
RAIN_WINDOWS = {"6h": 6, "24h": 24, "48h": 48, "72h": 72, "7d": 168, "14d": 336}
SPILL_WINDOWS, BANDS = (24, 48, 72), (10, 25, 50)
SINCE_CAP_H = 30 * 24
DECAY_KM = 10
N_LEAK_TESTS = 25


def ns(s):
    """UTC timestamps -> int64 nanoseconds (NaT -> int64 max)."""
    t = pd.to_datetime(pd.Series(s), utc=True, format="ISO8601").dt.as_unit("ns")  # pandas 3 defaults to us
    out = t.astype("int64").to_numpy(np.int64, copy=True)
    out[t.isna().to_numpy()] = np.iinfo(np.int64).max
    return out


def weather_features(t, sid, wx):
    f = {"temp_48h": np.nan, "temp_7d": np.nan, "sun_24h": np.nan, "sun_hours_24h": np.nan}
    if sid in wx["temp_c"]:
        f["temp_48h"] = hourly_window(wx["temp_c"][sid], t, 48, "mean")
        f["temp_7d"] = hourly_window(wx["temp_c"][sid], t, 168, "mean")
    if sid in wx["sw_wm2"]:
        f["sun_24h"] = hourly_window(wx["sw_wm2"][sid], t, 24, "sum") * 3600 / 1e6  # W/m2 per hour -> MJ/m2
    if sid in wx["sun_s"]:
        f["sun_hours_24h"] = hourly_window(wx["sun_s"][sid], t, 24, "sum") / 3600
    return f


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
    lr = np.log(f["flow_rel"]) if f["flow_rel"] and f["flow_rel"] > 0 else np.nan
    f["flow_low"], f["flow_high"] = max(0.0, -lr) if lr == lr else np.nan, max(0.0, lr) if lr == lr else np.nan
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
        f["spill_dilution_48h_25km"] = spill_dilution(f["spill_h_48h_25km"], f["flow"])
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
wx_file = INTERIM / "weather_hourly.csv"
if not wx_file.exists():
    raise SystemExit("data/interim/weather_hourly.csv not found: run scripts/05b_weather.py first")
wx_df = pd.read_csv(wx_file).rename(columns={"id": "station"})
wx_df["t"] = ns(wx_df["time_utc"])
wx = {c: build_series(wx_df, c) for c in ["temp_c", "sw_wm2", "sun_s"]}
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
def features_for(row, rain_, flow_, wx_, spills_override=None, offline_override=None):
    sid = row["id"]
    sp = spills_override if spills_override is not None else site_spills.get(sid)
    off = offline_override if offline_override is not None else offline
    f = site_features(row["t"], gauges.get(sid, []), rain_, flow_station.get(sid), flow_,
                      flow_median, sp, site_links.get(sid), off)
    f.update(weather_features(row["t"], sid, wx_))
    return f


rows = []
for _, r in tqdm(samples.iterrows(), total=len(samples), desc="Building features", unit="sample"):
    rows.append(features_for(r, rain, flow, wx))
feats = pd.DataFrame(rows, index=samples.index)

local = pd.to_datetime(samples["sample_time_utc"], utc=True, format="ISO8601").dt.tz_convert("Europe/London")
out = pd.concat([samples[["id", "site", "sample_time_utc", "year", "ecoli", "ie", "fail",
                          "fail_ecoli", "fail_ie", "discountable", "extra_sample"]], feats], axis=1)
prev = previous_result(samples)
out = pd.concat([out, prev], axis=1)
out["month"], out["day_of_year"] = local.dt.month, local.dt.day_of_year
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
    wx_t = {c: truncate(series, t) for c, series in wx.items()}
    again = features_for(r, truncate(rain, t), truncate(flow, t), wx_t, sp, as_known_at(offline, t))
    a, b = pd.Series(again), feats.loc[i, list(again)]
    same = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
    if not same.all():
        bad += 1
        print(f"  ! sample {i}: features changed when future data was removed: {list(a.index[~same])}")
print(f"Leakage test: {len(test_idx) - bad}/{len(test_idx)} samples identical with all later data removed")
if bad:
    raise SystemExit("Leakage detected - fix before modelling.")
# The previous-result feature uses other samples' labels, so check it separately:
# every result it uses must have been taken at least RESULT_LAG_DAYS earlier.
too_new = (out["days_since_prev"] < RESULT_LAG_DAYS).sum()
print(f"Previous-result check: {too_new} samples use a result less than {RESULT_LAG_DAYS} days old")
if too_new:
    raise SystemExit("Previous-result feature uses results that wouldn't be known yet.")

# ----------------------------------------------------------------- report
fcols = list(feats.columns) + list(prev.columns)
print("\nMissing values per feature (% of rows where it applies):")
for c in fcols:
    base = out[out["thames"]] if c.startswith(("spill", "spilling", "hours_since", "offline")) else out
    print(f"  {c:26s} {base[c].isna().mean():6.1%}")
print("\nFailure rate by 48 h rainfall (sanity check: should rise with rain):")
q = pd.qcut(out["rain_48h"], 4, duplicates="drop")
print(out.groupby(q, observed=True)["fail"].agg(["size", "mean"]).round(3).to_string())
print("\nFailure rate by sunlight in the last 24 h (expect lower risk on sunnier days, all else equal):")
q = pd.qcut(out["sun_24h"], 4, duplicates="drop")
print(out.groupby(q, observed=True)["fail"].agg(["size", "mean"]).round(3).to_string())
print("\nFailure rate by the site's previous known result (persistence):")
grp = pd.cut(out["prev_exceed"], [-np.inf, np.log10(0.5), 0, np.inf],
             labels=["under half the limit", "half to 1x limit", "failed"])
print(out.groupby(grp.cat.add_categories("first of season").fillna("first of season"), observed=True)["fail"]
      .agg(["size", "mean"]).round(3).to_string())
dry = out["rain_48h"] <= 0.5
print("  same, dry 48 h only:")
print(out[dry].groupby(grp[dry], observed=True)["fail"].agg(["size", "mean"]).round(3).to_string())
th = out[out["spill_data_ok"]]
if len(th):
    print("\nThames samples (2023+), failure rate by whether any upstream spill within 25 km in the last 48 h:")
    print(th.groupby(th["spill_h_48h_25km"] > 0)["fail"].agg(["size", "mean"]).round(3).to_string())
print(f"\nRows: train {int((out['split'] == 'train').sum())}, test {int((out['split'] == 'test').sum())}; "
      f"Thames with reliable spill data: {int(out['spill_data_ok'].sum())}")
out.to_csv(OUT / "features.csv", index=False)
print(f"Saved {OUT/'features.csv'}")
