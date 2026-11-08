"""Developer tools that rebuild the committed fixtures from the public sources (needs network).

    python -m frost_night_forecaster fetch-met    # MeteoSwiss hourly CSVs -> data/meteoswiss_spring.csv.gz
    python -m frost_night_forecaster fetch-dem    # swissALTI3D 2 m tiles  -> data/dem_visp_10m.npz  (needs rasterio)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BASE_URL, DATA_DIR, FIXTURE, STATIONS, parse_hourly_csv, spring_mask

PERIODS = ("historical_2010-2019", "historical_2020-2029", "recent")
STAC_ALTI = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d"
# DEM window: LV95 km tiles x 2629..2632, y 1126..1130 (Visp, Valais: Rhone valley floor, side-valley mouth and vineyard slopes)
DEM_TILES_X = (2629, 2630, 2631, 2632)
DEM_TILES_Y = (1126, 1127, 1128, 1129, 1130)
DEM_CELL = 10.0


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "frost-night-forecaster/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def fetch_met(cache: Path, out: Path = FIXTURE) -> pd.DataFrame:
    frames = []
    for abbr in STATIONS:
        low = abbr.lower()
        for period in PERIODS:
            dest = cache / f"{low}_{period}.csv"
            _download(f"{BASE_URL}/{low}/ogd-smn_{low}_h_{period}.csv", dest)
            df = parse_hourly_csv(dest.read_text(encoding="utf-8"))
            frames.append(df[spring_mask(df["time_utc"])])
    full = pd.concat(frames, ignore_index=True).sort_values(["station", "time_utc"])
    full = full.drop_duplicates(["station", "time_utc"])
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False, float_format="%.4g", compression="gzip")
    return full


def fetch_dem(cache: Path, out: Path = DATA_DIR / "dem_visp_10m.npz") -> dict:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("fetch-dem needs rasterio: pip install rasterio") from exc
    nx, ny = len(DEM_TILES_X), len(DEM_TILES_Y)
    step = int(DEM_CELL / 2)  # 2 m source -> 10 m by block mean
    tile_px = 500
    big = np.full((ny * tile_px, nx * tile_px), np.nan, dtype="float32")
    for i, x in enumerate(DEM_TILES_X):
        for j, y in enumerate(DEM_TILES_Y):
            name = f"swissalti3d_2019_{x}-{y}"
            dest = cache / f"{name}_2.tif"
            _download(f"{STAC_ALTI}/{name}/{name}_2_2056_5728.tif", dest)
            with rasterio.open(dest) as ds:
                arr = ds.read(1).astype("float32")
            row0 = (ny - 1 - j) * tile_px  # north at the top
            big[row0 : row0 + tile_px, i * tile_px : (i + 1) * tile_px] = arr
    h, w = big.shape
    z = big.reshape(h // step, step, w // step, step).mean(axis=(1, 3))
    meta = dict(
        east0=float(DEM_TILES_X[0] * 1000),
        north_top=float((DEM_TILES_Y[-1] + 1) * 1000),
        cell=DEM_CELL,
    )
    np.savez_compressed(out, z=z.astype("float32"), **meta)
    return meta
