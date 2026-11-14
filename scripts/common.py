"""Shared helpers for the Environment Agency bathing-water scripts."""
import json
from pathlib import Path

import requests

BASE = "https://environment.data.gov.uk"
RAW = Path("data/raw/bwq")
RIVER_TYPE = "RiverBathingWater"


def get_all(url, params=None):
    """Page through an EA linked-data API endpoint and return all items."""
    items, page = [], 0
    while True:
        r = requests.get(url, params={**(params or {}), "_pageSize": 500, "_page": page}, timeout=60)
        if not r.ok:
            raise SystemExit(f"HTTP {r.status_code} for {r.url}\n{r.text[:800]}")
        batch = r.json()["result"]["items"]
        items.extend(batch)
        if len(batch) < 500:
            return items
        page += 1


def text(v):
    """EA JSON wraps many values as {'_value': ...}; unwrap to a string."""
    if isinstance(v, dict):
        return text(v.get("_value", v.get("inXSDDateTime", v.get("name", v.get("label", "")))))
    if isinstance(v, list):
        return text(v[0]) if v else ""
    return "" if v is None else str(v)


def all_text(v):
    """Join every value of a possibly-list field (e.g. several types)."""
    return " | ".join(sorted(text(x) for x in v)) if isinstance(v, list) else text(v)


def load_register():
    """All bathing waters in England (cached in data/raw/bwq/)."""
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / "bathing_waters.json"
    if f.exists():
        return json.loads(f.read_text())
    sites = get_all(f"{BASE}/doc/bathing-water.json")
    f.write_text(json.dumps(sites))
    return sites


def river_sites(register):
    """Register entries the EA labels as river bathing waters."""
    return [s for s in register if RIVER_TYPE in all_text(s.get("type"))]


def load_samples(sid):
    """Every in-season sample for one site (cached in data/raw/bwq/)."""
    f = RAW / f"samples_{sid}.json"
    if f.exists():
        return json.loads(f.read_text())
    samples = get_all(f"{BASE}/doc/bathing-water-quality/in-season/sample.json",
                      {"bwq_bathingWater.eubwidNotation": sid})
    f.write_text(json.dumps(samples))
    return samples
