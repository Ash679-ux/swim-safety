"""Day 1: first authenticated request to the Thames Water Open Data API.

1. Register at https://data.thameswater.co.uk/s/ and create an application
   to get a client ID and client secret.
2. Copy .env.example to .env and paste them in (.env is git-ignored).
3. Run:  python scripts/02_test_thames_api.py

Check the endpoint URL and parameter names against the portal's API docs
for the "Discharge to Environment - Alert Stream" endpoint; the ones below
are what the portal has used, but confirm them when you log in.
"""
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
URL = "https://prod-tw-opendata-app.uk-e1.cloudhub.io/data/STE/v1/DischargeAlerts"
headers = {
    "client_id": os.environ["TW_CLIENT_ID"],
    "client_secret": os.environ["TW_CLIENT_SECRET"],
}

r = requests.get(URL, headers=headers, params={"limit": 100, "offset": 0}, timeout=60)
print("HTTP", r.status_code)
r.raise_for_status()
data = r.json()

out = Path("data/raw/thames/discharge_alerts_test.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2))

items = data.get("items", data if isinstance(data, list) else [])
print(f"{len(items)} alerts returned; saved to {out}")
if items:
    print("Fields:", sorted(items[0].keys()))
    print("First record:", json.dumps(items[0], indent=2))
