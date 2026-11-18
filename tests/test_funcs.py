"""Unit tests for the pipeline's core logic. Run with:  pytest"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from common import text
from funcs import (HOUR, NO_END, STEP, as_known_at, build_series, fails, fast_ap, flow_at,
                   hourly_window, pair, rain_total, site_rate_by_other_seasons, truncate)

T0 = pd.Timestamp("2024-06-01", tz="UTC").value  # an arbitrary start time, in ns


def series(values, step=STEP, start=T0):
    """One station's series with readings every `step` ns."""
    df = pd.DataFrame({"station": "A", "t": start + step * np.arange(len(values)), "v": values})
    return build_series(df, "v")["A"]


# ----------------------------------------------------------------- labels
def test_label_rule():
    count = pd.Series([999, 1000, 1001, 1000, 1000])
    qual = pd.Series(["=", "=", "=", ">", "<"])
    # over the limit fails; at the limit only fails if the lab said ">" (above its range)
    assert fails(count, qual, 1000).tolist() == [False, False, True, True, False]


def test_text_unwraps_ea_values():
    assert text({"_value": "River Thames"}) == "River Thames"
    assert text([{"_value": "a"}, {"_value": "b"}]) == "a"
    assert text({"inXSDDateTime": {"_value": "2024-06-01T10:00:00"}}) == "2024-06-01T10:00:00"
    assert text(None) == ""


# ------------------------------------------------------------ spill pairing
def alerts(*rows):
    return pd.DataFrame(rows, columns=["alertType", "time_utc"])


def test_pair_start_stop():
    pairs, unmatched = pair(alerts(("Start", 1), ("Stop", 3), ("Start", 5), ("Stop", 6)), "Start", "Stop")
    assert pairs == [(1, 3), (5, 6)] and unmatched == 0


def test_pair_stop_before_history_is_dropped():
    pairs, unmatched = pair(alerts(("Stop", 1), ("Start", 2), ("Stop", 4)), "Start", "Stop")
    assert pairs == [(2, 4)] and unmatched == 1


def test_pair_repeated_start_keeps_first_and_ongoing_has_no_end():
    pairs, _ = pair(alerts(("Start", 1), ("Start", 2), ("Stop", 3), ("Start", 7)), "Start", "Stop")
    assert pairs[0] == (1, 3)
    assert pairs[1][0] == 7 and pd.isna(pairs[1][1])


def test_pair_ignores_other_alert_types():
    pairs, _ = pair(alerts(("Start", 1), ("Offline start", 2), ("Stop", 3)), "Start", "Stop")
    assert pairs == [(1, 3)]


# -------------------------------------------------------------- rain windows
def test_rain_total_sums_window_before_sample():
    s = series(np.full(4 * 48, 0.25))  # 48 h of 0.25 mm per 15 min = 1 mm/h
    t = T0 + 48 * HOUR
    # the window ends one reading before t, so it holds 24 h of readings
    assert rain_total(s, t, 24) == pytest.approx(24.0)


def test_rain_total_needs_80_percent_coverage():
    v = np.full(4 * 24, 1.0)
    v[: int(4 * 24 * 0.3)] = np.nan  # 30% missing
    s = series(v)
    assert np.isnan(rain_total(s, T0 + 24 * HOUR + STEP, 24))


def test_rain_total_ignores_readings_after_sample():
    v = np.r_[np.zeros(4 * 24), np.full(4 * 24, 9.0)]  # heavy rain only AFTER the sample
    s = series(v)
    assert rain_total(s, T0 + 24 * HOUR, 24) == 0.0


def test_truncate_gives_identical_answer():
    """The leakage test relies on this: cutting the data at t can't change a window ending at t."""
    s = series(np.random.default_rng(0).random(4 * 72))
    t = T0 + 40 * HOUR
    cut = truncate({"A": s}, t)["A"]
    assert rain_total(cut, t, 24) == pytest.approx(rain_total(s, t, 24))


def test_hourly_window_mean_and_sum():
    s = series(np.arange(48, dtype=float), step=HOUR)  # values 0..47, one per hour
    t = T0 + 47 * HOUR                                  # last reading is at t
    assert hourly_window(s, t, 24, "sum") == pytest.approx(sum(range(24, 48)))
    assert hourly_window(s, t, 24, "mean") == pytest.approx(np.mean(range(24, 48)))


def test_flow_at_rejects_stale_readings():
    s = series(np.array([5.0, 6.0]), step=HOUR)
    assert flow_at(s, T0 + HOUR + 30 * 60 * 10**9) == 6.0   # 30 min old: fine
    assert np.isnan(flow_at(s, T0 + 4 * HOUR))              # 3 h old: too stale
    assert np.isnan(flow_at(s, T0 - HOUR))                  # before any reading


# -------------------------------------------------------- events as known at t
def test_as_known_at():
    ev = pd.DataFrame({"start": [1, 5, 20], "end": [3, 15, 30]})
    known = as_known_at(ev, 10)
    assert known["start"].tolist() == [1, 5]          # the spill starting at 20 isn't known yet
    assert known["end"].tolist() == [3, NO_END]       # its Stop at 15 hasn't arrived yet


# ------------------------------------------------------------- site history
def test_site_rate_has_no_label_leak():
    """Within one site and season, failing and passing samples must get the SAME
    site rate (a leave-one-out rate would give failures a lower one)."""
    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"id": np.repeat(["a", "b"], 60), "year": np.tile(np.repeat([2023, 2024, 2025], 20), 2),
                         "fail": rng.random(120) < 0.3})
    rate = site_rate_by_other_seasons(rows)
    for (_, _), g in rows.assign(rate=rate).groupby(["id", "year"]):
        assert g["rate"].nunique() == 1


def test_site_rate_uses_only_other_seasons():
    rows = pd.DataFrame({"id": ["a"] * 4, "year": [2024, 2024, 2025, 2025], "fail": [True, True, False, False]})
    rate = site_rate_by_other_seasons(rows)
    # 2024 rows only see 2025 (all passes), so their rate sits below the 2025 rows' rate
    assert rate.iloc[0] < rate.iloc[2]


# ----------------------------------------------------------------- metrics
@pytest.mark.parametrize("seed", range(20))
def test_fast_ap_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(300) < 0.2).astype(int)
    s = np.round(rng.random(300), 2)  # rounding creates ties
    assert fast_ap(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)
