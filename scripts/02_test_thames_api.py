"""First request to the Thames Water discharge (spill) history API.

Run from the project root:
    python scripts/02_test_thames_api.py

Uses the v2 open-data endpoint (as used by the POOPy package), which appears
not to need an API key. It returns start/stop "alerts" for each overflow
monitor, newest first, 1000 per page; history is valid from 1 April 2022.
"""
import json
from pathlib import Path

import requests

URL = "https://api.thameswater.co.uk/opendata/v2/discharge/alerts"

r = requests.get(URL, params={"limit": 1000, "offset": 0}, timeout=60)
print("HTTP", r.status_code)
if not r.ok:
    raise SystemExit(r.text[:800])
data = r.json()

out = Path("data/raw/thames/discharge_alerts_page0.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2))

items = data.get("items", data if isinstance(data, list) else [])
print(f"{len(items)} alerts on the first page; saved to {out}")
if items:
    print("Fields:", sorted(items[0].keys()))
    print("Newest:", json.dumps(items[0], indent=2))
    print("Oldest on this page:", json.dumps(items[-1], indent=2))
