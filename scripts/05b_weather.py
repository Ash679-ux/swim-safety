"""Air temperature and sunlight at each river bathing site.

Run from the project root (after 04_samples.py):
    python scripts/05b_weather.py

Why: bacteria from sewage and runoff die faster in sunlight (UV damages their
DNA) and survive differently with temperature, so a dry, sunny, warm spell
should lower the risk, and a dull, cool one raise it. This gives the models a
way to explain some failures that rain can't.

Source: Open-Meteo historical weather API (ERA5 reanalysis, hourly, no key
needed, CC BY 4.0): https://open-meteo.com/en/docs/historical-weather-api
For each site and sampled season (15 April to 30 September) it downloads
  temperature_2m        air temperature at 2 m (deg C), at the hour
  shortwave_radiation   sunlight energy (W/m2), mean over the preceding hour
  sunshine_duration     seconds of bright sunshine in the preceding hour
Raw JSON is cached in data/raw/weather/. Writes data/interim/weather_hourly.csv.
"""
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import get_json

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "shortwave_radiation", "sunshine_duration"]
SEASON = ("04-15", "09-30")
RAW = Path("data/raw/weather")
OUT = Path("data/interim")
RAW.mkdir(parents=True, exist_ok=True)

sites = pd.read_csv(OUT / "sites.csv")
samples = pd.read_csv(OUT / "samples.csv")
years = samples.groupby("id")["year"].unique().to_dict()

jobs = [(s, int(y)) for s in sites.itertuples() for y in sorted(years.get(s.id, []))]
rows = []
for s, y in tqdm(jobs, desc="Downloading weather", unit="site-season"):
    f = RAW / f"{s.id}_{y}.json"
    if f.exists():
        data = json.loads(f.read_text())
    else:
        data = get_json(URL, {"latitude": round(s.lat, 4), "longitude": round(s.long, 4),
                              "start_date": f"{y}-{SEASON[0]}", "end_date": f"{y}-{SEASON[1]}",
                              "hourly": ",".join(VARS), "timezone": "GMT"})
        f.write_text(json.dumps(data))
    h = data.get("hourly", {})
    df = pd.DataFrame({"id": s.id, "time_utc": h.get("time", []),
                       **{v: h.get(v, [None] * len(h.get("time", []))) for v in VARS}})
    rows.append(df)

w = pd.concat(rows, ignore_index=True)
w["time_utc"] = pd.to_datetime(w["time_utc"], utc=True, format="ISO8601")
w = w.rename(columns={"temperature_2m": "temp_c", "shortwave_radiation": "sw_wm2", "sunshine_duration": "sun_s"})
w = w.drop_duplicates(["id", "time_utc"]).sort_values(["id", "time_utc"])
w.to_csv(OUT / "weather_hourly.csv", index=False)

print(f"\n{len(w):,} hourly rows for {w['id'].nunique()} sites; missing: "
      + ", ".join(f"{c} {w[c].isna().mean():.1%}" for c in ["temp_c", "sw_wm2", "sun_s"]))
print("Typical bathing-season values (median): "
      f"temperature {w['temp_c'].median():.1f} C, midday sunlight "
      f"{w.loc[w['time_utc'].dt.hour == 12, 'sw_wm2'].median():.0f} W/m2")
print(f"Saved {OUT/'weather_hourly.csv'}")
