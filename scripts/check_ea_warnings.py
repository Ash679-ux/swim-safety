"""One-off comparison with the EA's own daily warnings (not part of run_all.sh).

Run from the project root (after 09_models.py):
    python scripts/check_ea_warnings.py

The EA publishes a daily pollution risk record for every bathing water. At
rain-forecast ("PRF") sites the risk can be raised because of forecast rain.
None of our 20 river sites is a PRF site (every record says NON_PRF_SITE or
has no type), so here the risk is only raised when the EA knows of a
pollution incident. This script counts how many failures fell on such days,
next to the rain rule and the two-part rule on the same samples.

Saves the raw records to data/raw/bwq/prf_<site>.json.
"""
import json

import pandas as pd

from common import BASE, RAW, get_all, text

feat = pd.read_csv("data/processed/features.csv")
rule_mm = pd.read_csv("results/model_results.csv").set_index("model").loc["rain_rule", "threshold"]

rows = []
for sid in feat["id"].unique():
    f = RAW / f"prf_{sid}.json"
    if not f.exists():
        f.write_text(json.dumps(get_all(f"{BASE}/doc/bathing-water-quality/stp-risk-prediction/point/{sid.split('-')[1]}.json")))
    for it in json.loads(f.read_text()):
        rows.append({"id": sid, "date": text(it.get("predictedOn")),
                     "raised": text(it.get("riskLevel")).rsplit("/", 1)[-1] != "normal",
                     "prf_site": text(it.get("prfOriginType")) not in ("", "NON_PRF_SITE")})
prf = pd.DataFrame(rows)
print(f"{len(prf)} daily records for {prf['id'].nunique()} sites; PRF (rain-forecast) records: {int(prf['prf_site'].sum())}")
print("Days with raised risk, by year:")
print(prf[prf["raised"]].groupby(prf["date"].str[:4]).size().to_string())

# A site can have two records on one day (an incident notice and the routine one); raised if either is
days = prf.groupby(["id", "date"])["raised"].any().reset_index()
feat["date"] = pd.to_datetime(feat["sample_time_utc"], utc=True, format="ISO8601").dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
d = feat.merge(days, on=["id", "date"], how="inner")  # only samples on days the EA published a record
f = d["fail"].astype(bool)
rain = d["rain_48h"].fillna(0) >= rule_mm
alerts = {"EA warning": d["raised"], f"rain rule (> {rule_mm:.1f} mm)": rain, "two-part rule": rain | (d["prev_exceed"] > 0)}

print(f"\nSamples taken on a day with an EA record: {len(d)} of {len(feat)} ({int(f.sum())} failures)")
for split, mask in [("all years", d["year"] > 0), ("2026 test", d["year"] == 2026)]:
    print(f"\n{split}: {int(mask.sum())} samples, {int(f[mask].sum())} failures")
    for name, a in alerts.items():
        a = a & mask
        print(f"  {name:22s} flags {int(a.sum()):3d}, catches {int((a & f).sum()):3d} (precision {(a & f).sum() / max(a.sum(), 1):.0%})")
print(f"\nSamples the EA marked discountable: {int(d['discountable'].sum())}, of which on a raised-risk day: {int((d['discountable'] & d['raised']).sum())}")
