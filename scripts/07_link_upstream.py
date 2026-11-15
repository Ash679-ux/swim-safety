"""Which overflows are upstream of each Thames bathing site?

Run from the project root (after 06_spills.py):
    python scripts/07_link_upstream.py

Method (the "upstream rule" for the README's Data decisions section):
  * Uses a D8 flow-direction grid of the Thames basin (from the POOPy project,
    https://github.com/AlexLipp/POOPy, archived on Zenodo). Each cell holds the
    direction water flows out of it.
  * From each overflow monitor, follow the flow directions downstream.
  * If that path passes within SNAP_M metres of a bathing site, the overflow is
    upstream of the site, and the path length to that point is its distance
    along the river (river_km).
  * Overflows downstream of a site never pass it, so they are excluded
    automatically. The feature step chooses how far upstream to count.

Writes data/interim/upstream_links.csv: one row per (site, upstream overflow).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from tqdm import tqdm

D8_URL = "https://zenodo.org/records/19709169/files/thames_d8.tif?download=1"
D8_MD5 = "c9ec32f43a0b77a8f616dd24a19828df"
D8_FILE = Path("data/raw/d8/thames_d8.tif")
OUT = Path("data/interim")
SNAP_M = 300        # path must pass this close to the site (sites sit on a bank, not mid-channel)
MAX_KM = 300        # stop tracing after this far
# ESRI D8 codes -> (row step, column step)
STEPS = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def download(url, path, md5):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=(10, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        h = hashlib.md5()
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading D8 grid") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
                bar.update(len(chunk))
    if h.hexdigest() != md5:
        tmp.unlink()
        raise SystemExit(f"Checksum mismatch for {path.name}; delete and rerun.")
    tmp.rename(path)


def read_geotiff(path):
    """Grid plus its georeferencing, read with tifffile (no GDAL needed)."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        try:
            arr = page.asarray()
        except ValueError as e:
            raise SystemExit(f"{e}\nThis GeoTIFF needs an extra codec: pip install imagecodecs")
        tags = page.tags
        sx, sy = tags["ModelPixelScaleTag"].value[:2]
        tie = tags["ModelTiepointTag"].value  # (i, j, k, x, y, z)
        x0 = tie[3] - tie[0] * sx
        y0 = tie[4] + tie[1] * sy
    return arr, x0, y0, sx, sy


download(D8_URL, D8_FILE, D8_MD5)
d8, x0, y0, sx, sy = read_geotiff(D8_FILE)
nrows, ncols = d8.shape
print(f"D8 grid: {nrows} x {ncols} cells of {sx:g} m, x {x0:.0f}-{x0 + ncols * sx:.0f}, y {y0 - nrows * sy:.0f}-{y0:.0f}")


def to_cell(x, y):
    return int((y0 - y) // sy), int((x - x0) // sx)


def trace(x, y):
    """Cell-centre coordinates and cumulative distance (m) along the downstream path."""
    r, c = to_cell(x, y)
    xs, ys, dist = [], [], []
    d, max_steps = 0.0, int(MAX_KM * 1000 / sx)
    for _ in range(max_steps):
        if not (0 <= r < nrows and 0 <= c < ncols):
            break
        xs.append(x0 + (c + 0.5) * sx)
        ys.append(y0 - (r + 0.5) * sy)
        dist.append(d)
        step = STEPS.get(int(d8[r, c]))
        if step is None:  # sink, sea or no-data
            break
        r, c = r + step[0], c + step[1]
        d += sx * (2 ** 0.5 if step[0] and step[1] else 1)
    return np.array(xs), np.array(ys), np.array(dist)


# Sites: Thames Water river bathing waters, in British National Grid metres
sites = pd.read_csv(OUT / "sites.csv")
sites = sites[sites["water_company"].str.contains("Thames", case=False, na=False)].copy()
to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
e, n = to_bng.transform(sites["long"].to_numpy(), sites["lat"].to_numpy())
sites["easting"], sites["northing"] = e, n

monitors = pd.read_csv(OUT / "monitors.csv").dropna(subset=["x", "y"])
inside = monitors["x"].between(x0, x0 + ncols * sx) & monitors["y"].between(y0 - nrows * sy, y0)
print(f"{inside.sum()} of {len(monitors)} monitors lie inside the grid")

links = []
for m in tqdm(monitors[inside].itertuples(), total=int(inside.sum()), desc="Tracing overflows", unit="monitor"):
    px, py, pd_ = trace(m.x, m.y)
    if len(px) == 0:
        continue
    for s in sites.itertuples():
        gap = np.hypot(px - s.easting, py - s.northing)
        i = int(gap.argmin())
        if gap[i] <= SNAP_M:
            links.append({"id": s.id, "site": s.site, "permitNumber": m.permitNumber,
                          "location": m.location, "watercourse": m.watercourse,
                          "river_km": round(pd_[i] / 1000, 2), "snap_m": round(float(gap[i]))})

links = pd.DataFrame(links, columns=["id", "site", "permitNumber", "location", "watercourse", "river_km", "snap_m"])
links = links.sort_values(["site", "river_km"])
links.to_csv(OUT / "upstream_links.csv", index=False)

print("\nUpstream overflows per site (by distance along the river):")
for s in sites.itertuples():
    g = links[links["id"] == s.id]
    counts = "  ".join(f"<={k} km: {(g['river_km'] <= k).sum():3d}" for k in (5, 10, 25, 50))
    print(f"  {s.site[:34]:34s} total {len(g):3d}   {counts}")
    for r in g.head(5).itertuples():
        print(f"      {r.river_km:6.1f} km  {r.location[:30]:30s} ({r.watercourse})")
missing = sites[~sites["id"].isin(links["id"])]
if len(missing):
    print(f"\n  ! no upstream overflows found for {missing['site'].tolist()}: "
          f"the site may sit more than {SNAP_M} m from the grid's river line; try a larger SNAP_M")
print(f"\nSaved {OUT/'upstream_links.csv'}")
