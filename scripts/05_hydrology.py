"""Day 2: rainfall and river flow near each river bathing water.

Run from the project root (after 04_samples.py):
    python scripts/05_hydrology.py

Uses the EA Hydrology API (long-term, quality-checked gauge records):
    https://environment.data.gov.uk/hydrology/

For each site it
  1. finds rain gauges within RAIN_KM and flow stations within FLOW_KM,
  2. keeps the nearest few that have a 15-minute record,
  3. downloads 15-minute readings for each bathing season the site was sampled
     (15 April to 30 September, so 72 h features work from 1 May),
  4. saves raw JSON untouched in data/raw/hydrology/ (cached; reruns are fast).

Writes
    data/interim/stations.csv   which gauges are linked to which site, and how far away
    data/interim/rain_15min.csv station, time (UTC), rainfall mm, quality
    data/interim/flow_15min.csv station, time (UTC), flow m3/s, quality

Flow stations are matched to the site's river (read from the site name, or
SITE_RIVER_OVERRIDES). Sites with no same-river station are flagged "??".
Force a particular station with FLOW_OVERRIDES.
"""
import json
import math
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API = "https://environment.data.gov.uk/hydrology"
RAW = Path("data/raw/hydrology")
OUT = Path("data/interim")
RAW.mkdir(parents=True, exist_ok=True)

RAIN_KM, N_RAIN = 15, 3      # up to 3 rain gauges within 15 km
FLOW_KM, N_FLOW = 40, 1      # nearest flow station ON THE SAME RIVER within 40 km
SEASON = ("04-15", "09-30")
# Force a flow station for a site: {"site id": "station notation"}.
FLOW_OVERRIDES = {}
# River for sites whose name doesn't say it.
SITE_RIVER_OVERRIDES = {"Wolvercote Mill Stream": "Thames", "Sandy Lane, Chester": "Dee"}


def site_river(name):
    """River a bathing water is on, read from its name ("River Teme in Ludlow" -> "Teme")."""
    if name in SITE_RIVER_OVERRIDES:
        return SITE_RIVER_OVERRIDES[name]
    m = re.search(r"River (\w+)", name) or re.match(r"(\w+) (?:at|above|below|in)\b", name)
    return m.group(1) if m else None


def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=120)
    if not r.ok:
        raise SystemExit(f"HTTP {r.status_code} for {r.url}\n{r.text[:800]}")
    return r.json()


def items_of(data):
    return data.get("items") or data.get("result", {}).get("items", [])


def val(v):
    if isinstance(v, dict):
        return val(v.get("@value", v.get("_value", v.get("label", v.get("@id", "")))))
    if isinstance(v, list):
        return val(v[0]) if v else ""
    return v


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def find_stations(prop, lat, lon, dist):
    """Stations measuring `prop` within `dist` km (cached)."""
    f = RAW / f"stations_{prop}_{lat:.4f}_{lon:.4f}_{dist}.json"
    if f.exists():
        return json.loads(f.read_text())
    data = items_of(get_json(f"{API}/id/stations.json",
                             {"observedProperty": prop, "lat": lat, "long": lon, "dist": dist, "_limit": 200}))
    f.write_text(json.dumps(data))
    return data


def pick_measure(station, pattern):
    """The 15-minute measure id on a station, e.g. ...-rainfall-t-900-mm-qualified."""
    ids = [val(m.get("@id") if isinstance(m, dict) else m) for m in station.get("measures", [])]
    hits = [i for i in ids if re.search(pattern, str(i))]
    return hits[0].rsplit("/", 1)[-1] if hits else None


def readings(measure, start, end):
    """All readings for one measure between two dates (cached per season)."""
    f = RAW / f"readings_{measure}_{start}_{end}.json"
    if f.exists():
        return json.loads(f.read_text())
    data = items_of(get_json(f"{API}/id/measures/{measure}/readings.json",
                             {"mineq-date": start, "maxeq-date": end, "_limit": 100000}))
    f.write_text(json.dumps(data))
    return data


sites = pd.read_csv(OUT / "sites.csv")
samples = pd.read_csv(OUT / "samples.csv")
years = samples.groupby("id")["year"].unique().to_dict()

links, first = [], True
for _, s in sites.iterrows():
    for prop, dist, n, pattern, kind in [("rainfall", RAIN_KM, N_RAIN, r"rainfall-t-900", "rain"),
                                         ("waterFlow", FLOW_KM, N_FLOW, r"flow-i-900", "flow")]:
        found = find_stations(prop, s["lat"], s["long"], dist)
        if first and found:
            print("Example station record fields:", sorted(found[0].keys()))
            print("Example measures:", [val(m.get("@id") if isinstance(m, dict) else m)
                                        for m in found[0].get("measures", [])][:6], "\n")
            first = False
        cands = []
        for st in found:
            measure = pick_measure(st, pattern)
            lat, lon = val(st.get("lat")), val(st.get("long"))
            if not measure or lat in ("", None):
                continue
            notation = str(val(st.get("notation")) or val(st.get("@id")).rsplit("/", 1)[-1])
            cands.append({
                "id": s["id"], "site": s["site"], "kind": kind, "station": notation,
                "station_name": val(st.get("label")), "river": val(st.get("riverName")),
                "distance_km": round(km(s["lat"], s["long"], float(lat), float(lon)), 1),
                "measure": measure, "opened": val(st.get("dateOpened")), "closed": val(st.get("dateClosed")),
            })
        cands.sort(key=lambda c: c["distance_km"])
        if kind == "flow":
            river = site_river(s["site"])
            for c in cands:
                c["site_river"] = river
                c["river_match"] = bool(river) and river.lower() in str(c["river"]).lower()
            # Same-river stations first (nearest first), then the rest as a fallback.
            cands.sort(key=lambda c: (not c["river_match"], c["distance_km"]))
        if kind == "flow" and s["id"] in FLOW_OVERRIDES:
            cands = [c for c in cands if c["station"] == FLOW_OVERRIDES[s["id"]]] or cands
        links += cands[:n]

links = pd.DataFrame(links)
links.to_csv(OUT / "stations.csv", index=False)
print("Linked stations (flow marked OK if on the same river as the site, ?? if not):")
for (sid, site), g in links.groupby(["id", "site"]):
    rain = ", ".join(f"{r.station_name} {r.distance_km} km" for r in g[g.kind == "rain"].itertuples()) or "NONE"
    flow = ", ".join(f"{'OK' if r.river_match else '??'} {r.station_name} ({r.river}) {r.distance_km} km"
                     for r in g[g.kind == "flow"].itertuples()) or "NONE"
    print(f"  {site[:38]:38s} rain: {rain}\n  {'':38s} flow: {flow}")
flow_links = links[links.kind == "flow"]
mismatch = flow_links[~flow_links["river_match"].astype(bool)]
if len(mismatch):
    print(f"  ! {len(mismatch)} site(s) have no flow station on their own river within {FLOW_KM} km;"
          " using the nearest other station:", mismatch["site"].tolist())
no_data = sorted(set(sites["site"]) - set(links[links.kind == "rain"]["site"]))
if no_data:
    print("  ! no rain gauge found for:", no_data, "- raise RAIN_KM")

# Download each station once for every season any linked site needs.
jobs = {}
for r in links.itertuples():
    for y in years.get(r.id, []):
        jobs.setdefault((r.kind, r.station, r.measure), set()).add(int(y))
tables = {"rain": [], "flow": []}
for (kind, station, measure), ys in tqdm(sorted(jobs.items()), desc="Downloading gauges", unit="station"):
    for y in sorted(ys):
        for it in readings(measure, f"{y}-{SEASON[0]}", f"{y}-{SEASON[1]}"):
            tables[kind].append({"station": station, "time_utc": it.get("dateTime"),
                                 "value": it.get("value"), "quality": val(it.get("quality"))})

for kind, unit in [("rain", "mm"), ("flow", "m3s")]:
    df = pd.DataFrame(tables[kind])
    if df.empty:
        print(f"! no {kind} readings downloaded")
        continue
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time_utc"]).drop_duplicates(["station", "time_utc"]).sort_values(["station", "time_utc"])
    df = df.rename(columns={"value": unit})
    df.to_csv(OUT / f"{kind}_15min.csv", index=False)
    q = df["quality"].value_counts(normalize=True).mul(100).round(1).to_dict()
    print(f"\n{kind}: {len(df):,} readings from {df['station'].nunique()} stations, "
          f"{df[unit].isna().mean():.1%} missing values; quality %: {q}")
    cover = df.groupby("station")[unit].apply(lambda v: v.notna().mean()).round(2)
    gaps = cover[cover < 0.9]
    if len(gaps):
        print(f"  ! stations with under 90% of readings present: {gaps.to_dict()}")
print(f"\nSaved {OUT/'stations.csv'}, {OUT/'rain_15min.csv'}, {OUT/'flow_15min.csv'}")
