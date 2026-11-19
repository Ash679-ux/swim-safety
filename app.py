"""Is the river safe to swim in today? One Streamlit page applying the two-part rule.

Run from the project root, after 04_samples.py has written data/interim/sites.csv:
    streamlit run app.py

A site is flagged if it rained more than RAIN_MM in the last 48 h (live EA rain
gauges within 15 km) OR its most recent published sample failed. On the 2026
test season this caught 48 of 62 failures, with 33% of alerts being real failures.
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, "scripts")
from common import BASE, get_all, text  # noqa: E402
from funcs import build_series, rain_total  # noqa: E402

RAIN_MM = 2.7        # balanced rain-rule threshold, set on 2020-2025 data in 09_models.py
RAIN_KM, N_RAIN = 15, 3
MAX_RESULT_AGE_D = 14  # ponytail: a fixed cut-off for "recent"; the model used same-season results only
FM = "https://environment.data.gov.uk/flood-monitoring"


@st.cache_data(ttl=900)
def rain_48h(lat, lon, now_ns):
    """Mean 48 h rain over the nearest gauges, using the same 80%-coverage rule as the pipeline.
    ponytail: real-time flood-monitoring gauges, which can differ from the hydrology
    gauges the models were trained on; matters if a site's nearest gauge is different."""
    stations = requests.get(f"{FM}/id/stations", timeout=30, params={
        "parameter": "rainfall", "lat": lat, "long": lon, "dist": RAIN_KM}).json()["items"]
    stations = [s for s in stations if isinstance(s.get("lat"), float)]  # a few list several positions
    stations.sort(key=lambda s: (s["lat"] - lat) ** 2 + ((s["long"] - lon) * np.cos(np.radians(lat))) ** 2)
    since = (datetime.now(timezone.utc) - timedelta(hours=50)).strftime("%Y-%m-%dT%H:%M:%SZ")
    totals = []
    for s in stations[:N_RAIN]:
        items = requests.get(f"{FM}/id/stations/{s['notation']}/readings", timeout=30,
                             params={"parameter": "rainfall", "since": since, "_limit": 1000}).json()["items"]
        df = pd.DataFrame({"station": s["notation"], "mm": [i.get("value") for i in items],
                           "t": pd.to_datetime([i["dateTime"] for i in items], utc=True).as_unit("ns").asi8})
        if len(df):
            totals.append(rain_total(build_series(df, "mm")[s["notation"]], now_ns, 48))
    totals = [x for x in totals if not np.isnan(x)]
    return float(np.mean(totals)) if totals else np.nan


@st.cache_data(ttl=3600)
def last_result(sid):
    """Newest published sample: (time, failed). Pass/fail uses the same test as the
    model's previous-result feature: E. coli over 1000 or enterococci over 400."""
    items = get_all(f"{BASE}/doc/bathing-water-quality/in-season/sample.json",
                    {"bwq_bathingWater.eubwidNotation": sid})
    if not items:
        return pd.NaT, False
    it = max(items, key=lambda i: text(i.get("sampleDateTime")))
    when = pd.to_datetime(text(it.get("sampleDateTime")), utc=True)  # an hour out at most; fine for a 14-day cut-off
    ratio = max(float(it.get("escherichiaColiCount") or 0) / 1000,
                float(it.get("intestinalEnterococciCount") or 0) / 400)
    return when, ratio > 1


st.title("Is the river safe to swim in today?")
now = pd.Timestamp.now(tz="UTC")
st.caption(f"Checked {now.tz_convert('Europe/London'):%d %B %Y, %H:%M}. Rain data refresh every 15 minutes.")
if not (5 <= now.month <= 9):
    st.info("Outside the bathing season (May to September) there are no new samples, "
            "so only the rain part of the rule can flag a site.")

sites = pd.read_csv("data/interim/sites.csv").dropna(subset=["lat", "long"])
rows = []
with st.spinner("Fetching rain and sample results..."):
    for s in sites.itertuples():
        rain = rain_48h(s.lat, s.long, now.floor("15min").value)
        when, failed = last_result(s.id)
        recent = pd.notna(when) and (now - when).days <= MAX_RESULT_AGE_D
        reasons = (["rain"] if rain > RAIN_MM else []) + (["last result failed"] if recent and failed else [])
        risk = "Higher: " + " and ".join(reasons) if reasons else ("Unknown: no rain data" if np.isnan(rain) else "Lower")
        rows.append({"Site": s.site, "Risk": risk, "Rain, last 48 h (mm)": round(rain, 1),
                     "Last sample": f"{when:%d %b %Y}, {'fail' if failed else 'pass'}" if pd.notna(when) else "none"})

table = pd.DataFrame(rows).sort_values(["Risk", "Site"])
st.dataframe(table, hide_index=True, width="stretch")
st.markdown(
    f"**How this works.** A site is flagged if more than {RAIN_MM} mm of rain fell in the last "
    f"48 hours at the nearest rain gauges, or if its last sample (within {MAX_RESULT_AGE_D} days) "
    "failed the bathing-water limits. On the 2026 test season this caught 48 of 62 failures; "
    "about a third of alerts were real failures. A lower-risk result is not a guarantee: "
    "11 of 17 failures after dry weather were still missed. Data: Environment Agency.")
