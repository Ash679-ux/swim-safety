"""Day 3: Thames Water spill history, rebuilt into spill events.

Run from the project root:
    python scripts/07_spills.py

1. Downloads the full discharge alert history (newest first, 1000 per page)
   back to HISTORY_START, saving every page untouched in data/raw/thames/.
   If the run stops part-way, rerun it: saved pages are reused.
2. Pairs each monitor's alerts into events:
     Start -> Stop                  = a spill (the monitor indicates discharging)
     Offline start -> Offline stop  = the monitor was offline (spill status unknown)
3. Writes
     data/interim/monitors.csv  one row per overflow monitor (location, river, x/y)
     data/interim/spills.csv    one row per spill: start, end, hours
     data/interim/offline.csv   one row per offline period

Times in the API carry no time zone; like the samples they are treated as UK
clock time and converted to UTC. EDM alerts indicate spills, they don't confirm
them. To refresh with newer alerts later, delete data/raw/thames/alerts_*.json.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

URL = "https://api.thameswater.co.uk/opendata/v2/discharge/alerts"
LIMIT = 1000
HISTORY_START = pd.Timestamp("2022-04-01")  # history is valid from here
LOCAL_TZ = "Europe/London"
RAW = Path("data/raw/thames")
OUT = Path("data/interim")
RAW.mkdir(parents=True, exist_ok=True)


def fetch_page(offset):
    """One page of alerts; retries because an empty page can be a hidden rate limit."""
    f = RAW / f"alerts_{offset:07d}.json"
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(5):
        try:
            r = requests.get(URL, params={"limit": LIMIT, "offset": offset}, timeout=(10, 60))
        except requests.RequestException as e:
            r, err = None, str(e)
        if r is not None and r.ok:
            data = r.json()
            items = data.get("items", data if isinstance(data, list) else [])
            if items or attempt == 4:
                f.write_text(json.dumps(items))
                return items
            err = "empty page"
        elif r is not None:
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        wait = 5 * 2 ** attempt
        tqdm.write(f"  offset {offset}: {err}; retrying in {wait}s")
        time.sleep(wait)
    raise SystemExit(f"Gave up at offset {offset}. Rerun to resume from here.")


# ------------------------------------------------------------- 1. download
alerts, offset = [], 0
with tqdm(desc="Downloading alert pages", unit="page") as bar:
    while True:
        items = fetch_page(offset)
        alerts += items
        bar.update(1)
        if not items:
            break
        oldest = pd.to_datetime(items[-1].get("datetime"), errors="coerce")
        bar.set_postfix(oldest=str(oldest.date()) if pd.notna(oldest) else "?")
        if len(items) < LIMIT or (pd.notna(oldest) and oldest < HISTORY_START):
            break
        offset += LIMIT

df = pd.DataFrame(alerts)
print(f"\n{len(df):,} alerts downloaded")
print("Alert types:", df["alertType"].value_counts().to_dict())

# Times -> UTC, drop exact repeats (pages can overlap if new alerts arrived mid-download)
t = pd.to_datetime(df["datetime"], errors="coerce")
df["time_utc"] = t.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
before = len(df)
df = df.dropna(subset=["time_utc"]).drop_duplicates(["permitNumber", "locationName", "time_utc", "alertType"])
print(f"Removed {before - len(df):,} repeated or undatable alerts")

# ------------------------------------------------------------- 2. monitors
monitors = (df.sort_values("time_utc")
            .groupby("permitNumber")
            .agg(location=("locationName", "last"), watercourse=("receivingWaterCourse", "last"),
                 grid_ref=("locationGridRef", "last"), x=("x", "last"), y=("y", "last"),
                 first_alert_utc=("time_utc", "min"), n_alerts=("alertType", "size"))
            .reset_index())
renamed = df.groupby("permitNumber")["locationName"].nunique()
if (renamed > 1).any():
    print(f"Note: {(renamed > 1).sum()} permits appear under more than one location name; "
          "monitors are keyed by permit number.")

# ---------------------------------------------------------------- 3. pairs
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


spills, offline, unmatched = [], [], {"spill": 0, "offline": 0}
df = df.sort_values(["permitNumber", "time_utc"])
for permit, g in tqdm(df.groupby("permitNumber"), desc="Rebuilding events", unit="monitor"):
    for kind, a, b, target in [("spill", "Start", "Stop", spills),
                               ("offline", "Offline start", "Offline stop", offline)]:
        pairs, n_bad = pair(g, a, b)
        unmatched[kind] += n_bad
        target += [{"permitNumber": permit, "start_utc": s, "end_utc": e} for s, e in pairs]

now = pd.Timestamp.now(tz="UTC")
for name, rows in [("spills", spills), ("offline", offline)]:
    ev = pd.DataFrame(rows, columns=["permitNumber", "start_utc", "end_utc"])
    ev["ongoing"] = ev["end_utc"].isna()
    ev["hours"] = ((ev["end_utc"].fillna(now) - ev["start_utc"]).dt.total_seconds() / 3600).round(3)
    ev = ev.merge(monitors[["permitNumber", "location", "watercourse", "x", "y"]], on="permitNumber", how="left")
    ev = ev.sort_values("start_utc")
    ev.to_csv(OUT / f"{name}.csv", index=False)
    if name == "spills":
        sp = ev
    else:
        off = ev

monitors.to_csv(OUT / "monitors.csv", index=False)

# --------------------------------------------------------------- 4. report
print(f"\n{len(monitors):,} monitors, alerts from {df['time_utc'].min():%Y-%m-%d} to {df['time_utc'].max():%Y-%m-%d}")
print(f"{len(sp):,} spills  ({sp['ongoing'].sum()} still ongoing; "
      f"{unmatched['spill']} Stops dropped because their Start was before the history)")
print(f"{len(off):,} offline periods ({unmatched['offline']} unmatched Offline stops dropped)")
print("\nSpill duration (hours):")
print(sp["hours"].describe(percentiles=[.5, .9, .99]).round(2).to_string())
odd = sp[(sp["hours"] <= 0) | (sp["hours"] > 24 * 30)]
if len(odd):
    print(f"  ! {len(odd)} spills last <= 0 h or > 30 days; check these:")
    print(odd[["location", "start_utc", "end_utc", "hours"]].head(10).to_string(index=False))
sp["year"] = sp["start_utc"].dt.year
print("\nSpills and spill hours per year (all monitors):")
print(sp.groupby("year").agg(spills=("hours", "size"), hours=("hours", "sum"),
                             monitors=("permitNumber", "nunique")).round(0).to_string())
print("\nMonitors on the rivers of the Thames bathing sites:")
for river in ["Thames", "Pang", "Evenlode", "Cherwell", "Ock", "Thame", "Kennet", "Windrush", "Mole", "Wey"]:
    n = monitors["watercourse"].str.contains(rf"\b{river}\b", case=False, na=False, regex=True).sum()
    print(f"  {river:10s} {n:4d}")
print(f"\nSaved {OUT/'monitors.csv'}, {OUT/'spills.csv'}, {OUT/'offline.csv'}")
