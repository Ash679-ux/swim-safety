"""Build the clean sample table and the sites table.

Run from the project root:
    python scripts/04_samples.py

Reads the raw downloads in data/raw/bwq/ (fetching anything missing) and writes
    data/interim/sites.csv     one row per river bathing water, with coordinates
    data/interim/samples.csv   one row per sample, cleaned and labelled
    data/raw/bwq/MANIFEST.csv  what was downloaded and when

Cleaning rules (each one prints how many rows it affects):
  1. Keep samples from SINCE onwards.
  2. Reissued records: keep the newest record for each site + sample time.
  3. Drop samples missing either bacteria count.
  4. Label: fail if E. coli > 1000 or intestinal enterococci > 400 cfu/100 ml.
     A ">" count at or above the limit is a fail; a "<" count is a pass.
Nothing else is dropped. Unusual samples are flagged in columns instead.
"""
import json
import re
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm

from common import BASE, RAW, load_register, load_samples, river_sites, text
from funcs import fails

SINCE = 2020
ECOLI_LIMIT, IE_LIMIT = 1000, 400
# Sample times in the API carry no time zone. We treat them as UK clock time.
# If they are really GMT this places each sample up to 1 h early, which is the
# safe direction: features can never include data from after the sample.
LOCAL_TZ = "Europe/London"
SHORT_GAP_DAYS = 4  # routine sampling is weekly; closer than this is an extra sample

OUT = RAW.parents[1] / "interim"
OUT.mkdir(parents=True, exist_ok=True)


def deep_find(obj, key):
    """First simple value stored under `key` anywhere inside nested JSON."""
    if isinstance(obj, dict):
        v = obj.get(key)
        if v is not None and (not isinstance(v, (dict, list)) or (isinstance(v, dict) and "_value" in v)):
            return text(v)
        for child in obj.values():
            found = deep_find(child, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for child in obj:
            found = deep_find(child, key)
            if found is not None:
                return found
    return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def site_detail(sid):
    """Full record for one bathing water (cached), used if the list lacks coordinates."""
    f = RAW / f"site_{sid}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = requests.get(f"{BASE}/doc/bathing-water/{sid}.json", params={"_view": "all"}, timeout=60)
    if not r.ok:
        print(f"  ! could not fetch details for {sid}: HTTP {r.status_code}")
        return {}
    detail = r.json().get("result", {}).get("primaryTopic", {})
    f.write_text(json.dumps(detail))
    return detail


def qualifier(v):
    return v.get("countQualifierNotation") if isinstance(v, dict) else (text(v) or None)


# ---------------------------------------------------------------- 1. sites
rivers = river_sites(load_register())
site_rows = []
for s in rivers:
    sid = text(s.get("eubwidNotation"))
    coords = {}
    for source in (s.get("samplingPoint"), s, None):
        if source is None:  # not in the register list: fetch the full record
            detail = site_detail(sid)
            source = detail.get("samplingPoint") or detail
        for k in ("lat", "long", "easting", "northing"):
            if coords.get(k) is None:
                coords[k] = to_float(deep_find(source, k))
        if coords["lat"] is not None and coords["long"] is not None:
            break
    year = re.search(r"(19|20)\d{2}", text(s.get("yearDesignated")))
    site_rows.append({
        "id": sid, "site": text(s.get("name")),
        "water_company": text(s.get("appointedSewerageUndertaker")),
        "year_designated": int(year.group()) if year else None,
        "sampling_point": deep_find(s.get("samplingPoint"), "samplePointNotation"),
        **coords,
    })
sites = pd.DataFrame(site_rows)
missing_xy = sites[sites["lat"].isna() | sites["long"].isna()]
print(f"{len(sites)} river sites; {len(missing_xy)} without coordinates")
if len(missing_xy):
    print("  ! no coordinates for:", missing_xy["site"].tolist())
    print("  samplingPoint field looks like:", json.dumps(rivers[0].get("samplingPoint"))[:400])

# -------------------------------------------------------------- 2. samples
rows, fields = [], set()
for _, s in tqdm(sites.iterrows(), total=len(sites), desc="Loading samples", unit="site"):
    for it in load_samples(s["id"]):
        fields.update(it)
        about = text(it.get("_about"))
        rec = re.search(r"recordDate/(\d{8})", about)
        rows.append({
            "id": s["id"], "site": s["site"],
            "sample_time_raw": text(it.get("sampleDateTime")),
            "record_date": rec.group(1) if rec else None,
            "ecoli": it.get("escherichiaColiCount"),
            "ecoli_qual": qualifier(it.get("escherichiaColiQualifier")),
            "ie": it.get("intestinalEnterococciCount"),
            "ie_qual": qualifier(it.get("intestinalEnterococciQualifier")),
            "discountable": text(it.get("discountable")).lower() == "true",
            "uri": about,
        })
df = pd.DataFrame(rows)
print("\nFields in the sample records:", sorted(fields))

# Times: parse, then convert to UTC so they line up with rain and spill data.
raw = df["sample_time_raw"]
has_offset = raw.str.contains(r"T.*(?:Z|[+-]\d{2}:?\d{2})$", na=False)
aware = pd.to_datetime(raw.where(has_offset), errors="coerce", utc=True, format="ISO8601")
naive = pd.to_datetime(raw.where(~has_offset), errors="coerce", format="ISO8601")
naive = naive.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
df["sample_time_utc"] = aware.fillna(naive)
print(f"Sample times: {int(has_offset.sum())} carry a time zone, {int((~has_offset).sum())} do not "
      f"(treated as {LOCAL_TZ}); example raw value: {raw.iloc[0]!r}")
for c in ("ecoli", "ie"):
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------------------------------------------------------- 3. cleaning
log = [("all downloaded samples", len(df))]
bad_time = df["sample_time_utc"].isna()
df = df[~bad_time]
log.append(("after dropping unreadable times", len(df)))

df = df[df["sample_time_utc"].dt.year >= SINCE]
log.append((f"from {SINCE} onwards", len(df)))

df = df.sort_values(["id", "sample_time_utc", "record_date"])
dupes = df.duplicated(["id", "sample_time_utc"], keep="last")
if dupes.any():
    print("\nReissued records (older copies dropped):")
    print(df[df.duplicated(["id", "sample_time_utc"], keep=False)]
          [["site", "sample_time_raw", "record_date", "ecoli", "ie"]].to_string(index=False))
df = df[~dupes]
log.append(("after removing reissued duplicates", len(df)))

missing = df["ecoli"].isna() | df["ie"].isna()
df = df[~missing]
log.append(("after dropping missing counts", len(df)))

# --------------------------------------------------------------- 4. labels

df["fail_ecoli"] = fails(df["ecoli"], df["ecoli_qual"], ECOLI_LIMIT)
df["fail_ie"] = fails(df["ie"], df["ie_qual"], IE_LIMIT)
df["fail"] = df["fail_ecoli"] | df["fail_ie"]

# ---------------------------------------------------------------- 5. flags
local = df["sample_time_utc"].dt.tz_convert(LOCAL_TZ)
df["year"] = local.dt.year
md = local.dt.month * 100 + local.dt.day
df["out_of_season"] = (md < 501) | (md > 930)
df["days_since_prev"] = df.groupby("id")["sample_time_utc"].diff().dt.total_seconds() / 86400
same_season = df.groupby("id")["year"].diff() == 0
df["extra_sample"] = same_season & (df["days_since_prev"] < SHORT_GAP_DAYS)
df["censored"] = df["ecoli_qual"].isin(["<", ">"]) | df["ie_qual"].isin(["<", ">"])
odd_low = ((df["ecoli_qual"] == "<") & (df["ecoli"] > ECOLI_LIMIT)) | ((df["ie_qual"] == "<") & (df["ie"] > IE_LIMIT))

# --------------------------------------------------------------- 6. report
print("\nCleaning log:")
for step, n in log:
    print(f"  {n:6d}  {step}")
print("\nQualifiers  E. coli:", df["ecoli_qual"].value_counts(dropna=False).to_dict(),
      " enterococci:", df["ie_qual"].value_counts(dropna=False).to_dict())
print("\nFlags (rows kept, not dropped):")
for name, mask in [("discountable", df["discountable"]), ("extra_sample (< 4 days after previous)", df["extra_sample"]),
                   ("out_of_season (outside 1 May - 30 Sep)", df["out_of_season"]),
                   ('"<" count above a limit (ambiguous)', odd_low)]:
    sub = df[mask]
    rate = f"{sub['fail'].mean():.0%} fail" if len(sub) else "-"
    print(f"  {len(sub):5d}  {name:42s} {rate}")
print(f"  overall failure rate for comparison: {df['fail'].mean():.1%}")

per_site = df.groupby("id").agg(n_samples=("fail", "size"), n_fail=("fail", "sum"),
                                fail_rate=("fail", "mean"), first_year=("year", "min"),
                                samples_per_season=("year", lambda y: round(y.value_counts().mean(), 1)))
sites = sites.merge(per_site, left_on="id", right_index=True, how="left")
print("\nSites:")
print(sites[["site", "water_company", "year_designated", "first_year", "lat", "long",
             "n_samples", "n_fail", "fail_rate", "samples_per_season"]]
      .sort_values("fail_rate", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
early = sites[sites["first_year"] < sites["year_designated"]]
if len(early):
    print("\n  ! samples dated before the designation year at:", early["site"].tolist())

assert not df.duplicated(["id", "sample_time_utc"]).any(), "duplicate samples remain"
print(f"\nFINAL: {len(df)} samples, {int(df['fail'].sum())} fail ({df['fail'].mean():.1%}) "
      f"| train (<2026): {int((df['year'] < 2026).sum())} | test (2026): {int((df['year'] == 2026).sum())}")

# ----------------------------------------------------------------- 7. save
cols = ["id", "site", "sample_time_utc", "year", "ecoli", "ecoli_qual", "ie", "ie_qual",
        "fail_ecoli", "fail_ie", "fail", "discountable", "extra_sample", "out_of_season",
        "censored", "days_since_prev", "sample_time_raw", "record_date", "uri"]
df[cols].to_csv(OUT / "samples.csv", index=False)
sites.to_csv(OUT / "sites.csv", index=False)

manifest = [{"file": f.name, "bytes": f.stat().st_size,
             "downloaded_utc": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}
            for f in sorted(RAW.glob("*.json"))]
pd.DataFrame(manifest).to_csv(RAW / "MANIFEST.csv", index=False)
print(f"Saved {OUT/'samples.csv'}, {OUT/'sites.csv'} and {RAW/'MANIFEST.csv'}")
