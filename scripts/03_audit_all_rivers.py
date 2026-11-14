"""Day 1 audit, part 2: how much data is there across ALL river bathing waters?

Run from the project root (after 01_audit_bathing_samples.py):
    python scripts/03_audit_all_rivers.py

1. Picks out river bathing waters using the register's own type labels.
2. Downloads every in-season sample for each (raw JSON cached in data/raw/bwq/).
3. Counts samples and failures under three candidate labels:
     ecoli  : E. coli > 1000 cfu/100 ml
     ie     : intestinal enterococci > 400 cfu/100 ml
     either : either of the above
   (1000 and 400 are the inland "good" boundaries; record the final choice
   in the README's Data decisions section.)
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
ECOLI_LIMIT, IE_LIMIT = 1000, 400
SINCE = 2020  # spill (EDM) monitoring is patchy before about 2020

# Which register types to include. The EA register labels every site as
# River, Lake, Transitional (estuary) or Coastal. Add "LakeBathingWater"
# here to include lakes and ponds as well.
INCLUDE_TYPES = ["RiverBathingWater"]


def get_all(url, params):
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
    if isinstance(v, dict):
        return text(v.get("_value", v.get("inXSDDateTime", v.get("name", v.get("label", "")))))
    if isinstance(v, list):
        return text(v[0]) if v else ""
    return "" if v is None else str(v)


# 1. Register (reuse the copy saved by script 01 if it exists)
reg_file = RAW / "bathing_waters.json"
if reg_file.exists():
    sites = json.loads(reg_file.read_text())
else:
    sites = get_all(f"{BASE}/doc/bathing-water.json", {})
    reg_file.write_text(json.dumps(sites))

def all_text(v):
    """Join every value of a possibly-list field (e.g. several types)."""
    return " | ".join(sorted(text(x) for x in v)) if isinstance(v, list) else text(v)


# What does the register's "type" field contain? Print it so we can see.
type_counts = pd.Series([all_text(s.get("type")) for s in sites]).value_counts()
print("Register 'type' values:")
print(type_counts.to_string(), "\n")

rivers, info = [], {}
for s in sites:
    name, sid = text(s.get("name")), text(s.get("eubwidNotation"))
    stype = all_text(s.get("type"))
    if any(t in stype for t in INCLUDE_TYPES):
        rivers.append((name, sid))
        info[name] = {"undertaker": text(s.get("appointedSewerageUndertaker"))}

print(f"{len(rivers)} bathing waters of type {INCLUDE_TYPES}:")
for name, sid in rivers:
    print(f"  {sid}  {name:45s} {info[name]['undertaker'][:30]}")

# 2. Samples (cached)
rows = []
for name, sid in tqdm(rivers, desc="Downloading samples", unit="site"):
    f = RAW / f"samples_{sid}.json"
    if f.exists():
        samples = json.loads(f.read_text())
    else:
        samples = get_all(f"{BASE}/doc/bathing-water-quality/in-season/sample.json",
                          {"bwq_bathingWater.eubwidNotation": sid})
        f.write_text(json.dumps(samples))
    for it in samples:
        rows.append({
            "site": name, "id": sid,
            "date": text(it.get("sampleDateTime")),
            "ecoli": it.get("escherichiaColiCount"),
            "ie": it.get("intestinalEnterococciCount"),
        })

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df["year"] = df["date"].dt.year
for c in ["ecoli", "ie"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["year"] >= SINCE].dropna(subset=["ecoli", "ie"], how="all")
df["fail_ecoli"] = df["ecoli"] > ECOLI_LIMIT
df["fail_ie"] = df["ie"] > IE_LIMIT
df["fail_either"] = df["fail_ecoli"] | df["fail_ie"]

# 3. Summary
df["undertaker"] = df["site"].map(lambda n: info[n]["undertaker"])
df["thames"] = df["undertaker"].str.contains("thames", case=False)
by_site = df.groupby(["site", "undertaker"]).agg(
    first_year=("year", "min"), n=("year", "size"),
    fail_ecoli=("fail_ecoli", "sum"), fail_ie=("fail_ie", "sum"), fail_either=("fail_either", "sum"),
).sort_values("n", ascending=False)
print(f"\nSamples from {SINCE} onwards, per site:")
print(by_site.to_string())

by_year = df.groupby("year")[["fail_ecoli", "fail_ie", "fail_either"]].agg(["sum"])
by_year.columns = ["fail_ecoli", "fail_ie", "fail_either"]
by_year.insert(0, "n", df.groupby("year").size())
print("\nPer year (all river sites):")
print(by_year.to_string())

print(f"\nTOTAL: {len(df)} samples at {df['site'].nunique()} river sites since {SINCE}")
for c in ["fail_ecoli", "fail_ie", "fail_either"]:
    print(f"  {c:12s} {int(df[c].sum()):4d}  ({df[c].mean():.1%})")

th = df[df["thames"]]
print(f"\nThames Water sites only (the only company with spill history): "
      f"{len(th)} samples, {int(th['fail_either'].sum())} fail_either")
df.to_csv("data/audit_rivers.csv", index=False)
print("\nSaved data/audit_rivers.csv")
