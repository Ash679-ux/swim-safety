"""Audit: find the candidate bathing waters and count their EA samples.

Run from the project root:
    python scripts/01_audit_bathing_samples.py

What it does
1. Downloads the full list of English bathing waters (saved raw) and finds
   the candidate sites by name.
2. Downloads every in-season sample for each site (saved raw, untouched).
3. Prints samples per site per year, and how many exceed two candidate
   E. coli thresholds, so you can decide whether to add regions.
"""
import json
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://environment.data.gov.uk"
RAW = Path("data/raw/bwq")
RAW.mkdir(parents=True, exist_ok=True)

# Name fragments for the candidate sites (case-insensitive regex).
CANDIDATES = {
    "Wolvercote": r"wolvercote",
    "Wallingford": r"wallingford",
    "Pangbourne": r"pangbourne",
    "Ham & Kingston": r"\bham\b|kingston",
    "Frensham": r"frensham",
}
# Candidate single-sample E. coli thresholds (cfu/100 ml) to compare.
THRESHOLDS = [900, 1000]


def get_all(url, params):
    """Page through an EA linked-data API endpoint and return all items."""
    items, page = [], 0
    while True:
        r = requests.get(url, params={**params, "_pageSize": 500, "_page": page}, timeout=60)
        if not r.ok:
            raise SystemExit(f"HTTP {r.status_code} for {r.url}\n{r.text[:800]}")
        batch = r.json()["result"]["items"]
        items.extend(batch)
        if len(batch) < 500:
            return items
        page += 1


def text(v):
    """EA JSON often wraps values as {'_value': ...}; unwrap to a string."""
    if isinstance(v, dict):
        return text(v.get("_value", v.get("inXSDDateTime", "")))
    if isinstance(v, list):
        return text(v[0]) if v else ""
    return "" if v is None else str(v)


def find_key(item, *needles):
    """First key whose lowercase name contains all needles (field names vary)."""
    for k in item:
        if all(n in k.lower() for n in needles):
            return k
    return None


# 1. All bathing waters
sites = get_all(f"{BASE}/doc/bathing-water.json", {})
(RAW / "bathing_waters.json").write_text(json.dumps(sites))
print(f"{len(sites)} bathing waters in the register\n")

matches = []
for s in sites:
    name = text(s.get("name"))
    for label, pattern in CANDIDATES.items():
        if re.search(pattern, name, re.I):
            matches.append((label, name, text(s.get("eubwidNotation"))))
for m in matches:
    print("match:", m)
print("\nIf a match is wrong (e.g. a coastal 'Kingston'), drop it from CANDIDATES.\n")

# 2. Samples for each matched site
rows = []
for label, name, sid in tqdm(matches, desc="Downloading samples", unit="site"):
    samples = get_all(
        f"{BASE}/doc/bathing-water-quality/in-season/sample.json",
        {"bwq_bathingWater.eubwidNotation": sid},
    )
    (RAW / f"samples_{sid}.json").write_text(json.dumps(samples))
    tqdm.write(f"{name}: {len(samples)} samples downloaded")
    for it in samples:
        qual = it.get("escherichiaColiQualifier") or {}
        rows.append({
            "site": name, "id": sid,
            "date": text(it.get("sampleDateTime")),
            "ecoli": it.get("escherichiaColiCount"),
            # "=" actual count, "<"/">" means below/above the lab's range
            "ecoli_qualifier": qual.get("countQualifierNotation") if isinstance(qual, dict) else None,
            "enterococci": it.get("intestinalEnterococciCount"),
        })

if not rows:
    raise SystemExit("No samples found - check the matched IDs above.")

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df["ecoli"] = pd.to_numeric(df["ecoli"], errors="coerce")
df["year"] = df["date"].dt.year
print("\nE. coli qualifiers:", df["ecoli_qualifier"].value_counts(dropna=False).to_dict(), "\n")

# 3. Summary
summary = df.groupby(["site", "year"]).agg(n=("ecoli", "size"), missing_ecoli=("ecoli", lambda s: s.isna().sum()))
for t in THRESHOLDS:
    summary[f"over_{t}"] = df.groupby(["site", "year"])["ecoli"].apply(lambda s: (s > t).sum())
print(summary.to_string())
print(f"\nTOTAL samples: {len(df)}")
for t in THRESHOLDS:
    print(f"over {t}: {(df['ecoli'] > t).sum()}  ({(df['ecoli'] > t).mean():.1%})")

df.to_csv("data/audit_samples.csv", index=False)
print("\nSaved tidy copy to data/audit_samples.csv")
