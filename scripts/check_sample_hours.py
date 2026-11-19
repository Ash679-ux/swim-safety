"""One-off check of the sample time zone assumption (not part of run_all.sh).

Run from the project root (after 04_samples.py):
    python scripts/check_sample_hours.py

The API's sample times carry no time zone; 04_samples.py treats them as UK
clock time. If that's right, the recorded clock hours should look like a
sampler's working day. Times at exactly midnight would mean date-only records,
whose rain windows would end before the sample was taken.

Writes results/figures/6_sample_hours.png.
"""
import matplotlib.pyplot as plt
import pandas as pd

from plotstyle import FIG, SERIES, subtitle

s = pd.read_csv("data/interim/samples.csv")
clock = pd.to_datetime(s["sample_time_raw"].astype(str).str.slice(0, 19), format="ISO8601")  # the time as recorded, offset ignored

print(f"{len(s)} samples; at exactly 00:00 (date-only records?): {int((clock.dt.hour.eq(0) & clock.dt.minute.eq(0)).sum())}")
print("\nRecorded clock hour, by year:")
print(pd.crosstab(clock.dt.hour.rename("hour"), s["year"], margins=True).to_string())

counts = clock.dt.hour.value_counts().reindex(range(24), fill_value=0)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(counts.index, counts.values, width=0.7, color=SERIES[0], zorder=2)
ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xlabel("Recorded clock hour of the sample")
ax.set_ylabel("Samples")
ax.grid(axis="x", visible=False)
ax.set_title("When samples were taken (recorded clock time)", pad=26)
subtitle(ax, f"All {len(s)} samples, 2021-2026. A working-day spread supports reading them as UK time.")
fig.tight_layout()
fig.savefig(FIG / "6_sample_hours.png", dpi=200)
print(f"\nSaved {FIG / '6_sample_hours.png'}")
