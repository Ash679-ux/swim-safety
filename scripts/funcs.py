"""Small, pure functions used by the pipeline scripts, kept here so they can
be unit-tested (tests/test_funcs.py) without running a whole script.

Times are int64 nanoseconds since 1970 (UTC). A "series" is a dict for one
station: sorted times "t", values "v", and running totals "cum"/"cnt" (with a
leading 0) so any window's sum and count are two lookups.
"""
import numpy as np
import pandas as pd

HOUR = 3600 * 10**9                      # one hour in nanoseconds
STEP = 15 * 60 * 10**9                   # one 15-minute reading interval
MIN_COVERAGE = 0.8                       # share of readings a window needs
FLOW_MAX_AGE_H = 2                       # newest flow reading may be this old
PRIOR_STRENGTH = 10                      # smoothing for site failure rates (in samples)
NO_END = np.iinfo(np.int64).max          # "no Stop received yet"


def fails(count, qual, limit):
    """Label rule: a sample fails if its count is over the limit, or if the lab
    reported it as ">" (above its range) at or above the limit."""
    return (count > limit) | ((qual == ">") & (count >= limit))

def pair(g, start_type, stop_type):
    """Pair each start with the next stop for one monitor."""
    out, open_start, unmatched_stop = [], None, 0
    for row in g.itertuples():
        if row.alertType == start_type:
            if open_start is None:           # a repeated Start keeps the first one
                open_start = row.time_utc
        elif row.alertType == stop_type:
            if open_start is not None:
                out.append((open_start, row.time_utc))
                open_start = None
            else:
                unmatched_stop += 1          # its Start is before the downloaded history
    if open_start is not None:
        out.append((open_start, pd.NaT))     # still ongoing at download time
    return out, unmatched_stop

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

def hourly_window(s, t, hours, how):
    """Sum or mean of hourly values in (t - hours, t]. Each hourly value covers
    the hour up to its timestamp, so nothing after t is used. Needs 80% of
    the hours present, else NaN."""
    i1 = np.searchsorted(s["t"], t, "right")
    i0 = np.searchsorted(s["t"], t - hours * HOUR, "right")
    n = s["cnt"][i1] - s["cnt"][i0]
    if n < MIN_COVERAGE * hours:
        return np.nan
    total = s["cum"][i1] - s["cum"][i0]
    return total / n if how == "mean" else total

def as_known_at(events, t):
    """Events as they would have been known at time t: started by t, and any
    Stop that came after t not yet received (end unknown)."""
    ev = events[events["start"] <= t].copy()
    ev.loc[ev["end"] > t, "end"] = NO_END
    return ev

def site_rate_map(fit, prior=PRIOR_STRENGTH):
    """Smoothed failure rate per site, learned from the rows in `fit`.
    Returns (rate per site, overall rate for sites not in `fit`)."""
    g = fit["fail"].mean()
    st = fit.groupby("id")["fail"].agg(["sum", "size"])
    return ((st["sum"] + prior * g) / (st["size"] + prior)).to_dict(), g

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

def fast_ap(y, s):
    """Same result as sklearn's average_precision_score (checked to 1e-15), but
    much faster: it skips sklearn's input checks, which dominate over 1000 calls."""
    order = np.argsort(-s, kind="mergesort")
    y_s, s_s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s_s)), len(s_s) - 1]  # last row of each tied score
    tps = np.cumsum(y_s)[last]
    precision, recall = tps / (last + 1), tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
