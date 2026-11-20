"""Is the river safe to swim in today? One Streamlit page applying the two-part rule.

Run from the project root (reads sites.csv, written by 04_samples.py and committed):
    streamlit run app.py

A site is flagged if it rained more than RAIN_MM in the last 48 h (live EA rain
gauges within 15 km) OR its most recent published sample failed. On the 2026
test season this caught 48 of 62 failures, with 33% of alerts being real failures.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

sys.path.insert(0, "scripts")
from common import BASE  # noqa: E402
from funcs import build_series, rain_total  # noqa: E402

RAIN_MM = 2.7        # balanced rain-rule threshold, set on 2020-2025 data in 09_models.py
RAIN_KM, N_RAIN = 15, 3
MAX_RESULT_AGE_D = 14  # ponytail: a fixed cut-off for "recent"; the model used same-season results only
DEADLINE_S = 60  # whatever isn't back by then shows as unknown, so the page always loads
RAIN_WORKERS = 8      # 20 sites at once (80 requests) hung on Streamlit Cloud
FM = "https://environment.data.gov.uk/flood-monitoring"


def rain_48h(lat, lon, now_ns):
    """Mean 48 h rain over the nearest gauges, using the same 80%-coverage rule as the pipeline.
    A gauge that doesn't answer within 30 s is skipped, like one with too many gaps.
    ponytail: real-time flood-monitoring gauges, which can differ from the hydrology
    gauges the models were trained on; matters if a site's nearest gauge is different."""
    try:
        r = requests.get(f"{FM}/id/stations", timeout=(5, 20), params={
            "parameter": "rainfall", "lat": lat, "long": lon, "dist": RAIN_KM})
        r.raise_for_status()
        stations = r.json()["items"]
    except requests.RequestException as e:
        print(f"rain stations request failed: {e!r}", flush=True)
        return np.nan
    stations = [s for s in stations if isinstance(s.get("lat"), float)]  # a few list several positions
    stations.sort(key=lambda s: (s["lat"] - lat) ** 2 + ((s["long"] - lon) * np.cos(np.radians(lat))) ** 2)
    since = (datetime.now(timezone.utc) - timedelta(hours=50)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def gauge_total(notation):  # each call takes 5-10 s, so the gauges run at the same time
        try:
            r = requests.get(f"{FM}/id/stations/{notation}/readings", timeout=(5, 20),
                             params={"parameter": "rainfall", "since": since, "_limit": 1000})
            r.raise_for_status()
            items = r.json()["items"]
        except requests.RequestException:
            return np.nan
        if not items:
            return np.nan
        df = pd.DataFrame({"station": notation, "mm": [i.get("value") for i in items],
                           "t": pd.to_datetime([i["dateTime"] for i in items], utc=True).as_unit("ns").asi8})
        return rain_total(build_series(df, "mm")[notation], now_ns, 48)

    with ThreadPoolExecutor(N_RAIN) as pool:
        totals = [x for x in pool.map(gauge_total, [s["notation"] for s in stations[:N_RAIN]]) if not np.isnan(x)]
    return float(np.mean(totals)) if totals else np.nan


def last_results():
    """Newest published sample for every bathing water: {id: (time, failed)}, in ONE request.
    One request per site got HTTP 429 (too many requests) from the EA on Streamlit Cloud.
    Pass/fail uses the same test as the model's previous-result feature: E. coli over 1000
    or enterococci over 400."""
    try:  # no retries: on Streamlit Cloud retries piled up behind HTTP 429s and the page hung
        r = requests.get(f"{BASE}/doc/bathing-water.json", timeout=(5, 30), params={
            "_view": "basic", "_pageSize": 1000, "_properties": (
                "eubwidNotation,latestSampleAssessment.sampleDateTime,"
                "latestSampleAssessment.escherichiaColiCount,latestSampleAssessment.intestinalEnterococciCount")})
        r.raise_for_status()
        items = r.json()["result"]["items"]
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"samples request failed: {e!r}", flush=True)
        return {}
    out = {}
    for it in items:
        s = it.get("latestSampleAssessment")
        if not isinstance(s, dict) or not s.get("sampleDateTime"):
            continue
        # sampleDateTime comes as a URI ending in the local time; an hour out at most, fine for a 14-day cut-off
        when = pd.Timestamp(str(s["sampleDateTime"]).rsplit("/", 1)[-1], tz="UTC")
        ratio = max(float(s.get("escherichiaColiCount") or 0) / 1000,
                    float(s.get("intestinalEnterococciCount") or 0) / 400)
        out[it["eubwidNotation"]] = (when, ratio > 1)
    return out


st.title("Is the river safe to swim in today?")
now = pd.Timestamp.now(tz="UTC")
st.caption(f"Checked {now.tz_convert('Europe/London'):%d %B %Y, %H:%M}. Data refresh every hour.")
if not (5 <= now.month <= 9):
    st.info("Outside the bathing season (May to September) there are no new samples, "
            "so only the rain part of the rule can flag a site.")

def site_row(s, now, rain, latest):
    when, failed = latest.get(s.id, (pd.NaT, False))
    recent = pd.notna(when) and (now - when).days <= MAX_RESULT_AGE_D
    reasons = (["rain"] if rain > RAIN_MM else []) + (["last result failed"] if recent and failed else [])
    risk = "Higher: " + " and ".join(reasons) if reasons else ("Unknown: no rain data" if np.isnan(rain) else "Lower")
    return {"Site": s.site, "lat": s.lat, "lon": s.long, "Risk": risk, "Rain, last 48 h (mm)": round(rain, 1),
            "Last sample": f"{when:%d %b %Y}, {'fail' if failed else 'pass'}" if pd.notna(when) else "none"}


@st.cache_data(ttl=3600, show_spinner=False)  # hourly: the EA rate-limited us at 15 min
def build_rows(now_ns):
    """All sites at once: rain for one site took 20-35 s one by one, so 20 sites took minutes."""
    now = pd.Timestamp(now_ns, tz="UTC")
    sites = pd.read_csv("sites.csv")
    t0 = time.time()
    pool = ThreadPoolExecutor(RAIN_WORKERS + 1)
    samples = pool.submit(last_results)
    futures = {pool.submit(rain_48h, s.lat, s.long, now.value): s for s in sites.itertuples()}
    done, _ = wait([samples, *futures], timeout=DEADLINE_S)
    pool.shutdown(wait=False, cancel_futures=True)  # don't let a stuck request hold up the page
    latest = samples.result() if samples in done else {}
    rain_ok = sum(not np.isnan(f.result()) for f in done & futures.keys())
    print(f"samples: {len(latest)} sites; rain data: {rain_ok}/{len(sites)} sites; "
          f"{time.time() - t0:.0f} s", flush=True)
    # ponytail: missing samples show as "none", not "unavailable"; matters only if the EA is down in season
    return [site_row(s, now, f.result() if f in done else np.nan, latest) for f, s in futures.items()]


with st.spinner("Fetching rain and sample results (about 30 seconds on the first load)..."):
    rows = build_rows(now.floor("1h").value)

COLOURS = {"Higher": ("#d62728", "#fbebeb"), "Lower": ("#2ca02c", "#e5f4e4"), "Unknown": ("#888888", "#f0efec")}
table = pd.DataFrame(rows).sort_values(["Risk", "Site"])
level = table["Risk"].str.split(":").str[0]
flagged, unknown = (level == "Higher").sum(), (level == "Unknown").sum()
st.subheader(f"{flagged} of {len(table)} sites flagged today"
             + (f", {unknown} without rain data" if unknown else ""))

# pydeck (comes with Streamlit) rather than st.map, which can't stop zooming out to the whole world
pts = table.assign(rgb=level.map(lambda k: [int(COLOURS[k][0][i:i + 2], 16) for i in (1, 3, 5)]))[["lat", "lon", "rgb"]]
st.pydeck_chart(pdk.Deck(
    layers=[pdk.Layer("ScatterplotLayer", pts, get_position=["lon", "lat"], get_fill_color="rgb",
                      get_radius=3000, radius_min_pixels=5)],
    initial_view_state=pdk.ViewState(latitude=pts.lat.mean(), longitude=pts.lon.mean(), zoom=5.3, min_zoom=5)))
st.caption("Red: higher risk. Green: lower risk. Grey: no rain data.")

st.dataframe(table.drop(columns=["lat", "lon"]).style.map(
    lambda r: f"background-color: {COLOURS[r.split(':')[0]][1]}", subset=["Risk"]),
    hide_index=True, width="stretch")

st.subheader("Why rain and the last result")
st.markdown(
    f"Rain washes sewage overflows and farm runoff into rivers, so the simplest warning is "
    f"rain: more than {RAIN_MM} mm in the last 48 hours at the nearest gauges. On the 2026 "
    "season, which the rule never saw while it was being set, that alone caught 40 of 62 failed samples. "
    "No machine-learning model I trained did significantly better.\n\n"
    "Most of the failures it missed came after dry weather. Those are hard to predict from "
    "weather, but a site that had just failed was far more likely to fail again: on dry days, "
    "31% of samples failed after a failed result against 5% otherwise. Adding 'or the last result failed' (counting a result only if it is "
    f"under {MAX_RESULT_AGE_D} days old) raised the catch to 48 of 62, at the cost of slightly "
    "more false alarms: about a third of alerts were real failures. It still misses 11 of the "
    "17 failures that followed a dry spell, so lower risk is not a guarantee.\n\n"
    "Data: Environment Agency [bathing-water samples](https://environment.data.gov.uk/bwq/) and "
    "[real-time rain gauges](https://environment.data.gov.uk/flood-monitoring/doc/rainfall). "
    "Code and full results: [github.com/Ash679-ux/swim-safety](https://github.com/Ash679-ux/swim-safety).")
