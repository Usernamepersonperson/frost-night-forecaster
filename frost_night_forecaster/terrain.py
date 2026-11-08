"""Terrain handling (numpy only): cold-air-pooling index from a DEM and the downscaled Tmin / frost risk map.

DEM source: swisstopo swissALTI3D (2 m grid, tiles from the federal open-data STAC, block-averaged to 10 m).
Source: Federal Office of Topography swisstopo, free geodata terms of use (source must be cited).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import DATA_DIR
from .model import frost_probability

DEM_FILE = DATA_DIR / "dem_visp_10m.npz"
_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class Dem:
    z: np.ndarray  # elevation m, row 0 = north edge
    east0: float  # LV95 easting of the west edge
    north_top: float  # LV95 northing of the north edge
    cell: float  # cell size m

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    def rowcol(self, e: float, n: float) -> tuple[int, int]:
        r = int((self.north_top - n) / self.cell)
        c = int((e - self.east0) / self.cell)
        if not (0 <= r < self.z.shape[0] and 0 <= c < self.z.shape[1]):
            raise ValueError("point outside the DEM")
        return r, c

    def extent_km(self) -> tuple[float, float, float, float]:
        """(west, east, south, north) in LV95 km, for matplotlib `extent`."""
        h, w = self.z.shape
        return (self.east0 / 1000, (self.east0 + w * self.cell) / 1000,
                (self.north_top - h * self.cell) / 1000, self.north_top / 1000)


def load_dem(path: Path | str | None = None) -> Dem:
    d = np.load(path or DEM_FILE)
    return Dem(z=d["z"].astype("float64"), east0=float(d["east0"]), north_top=float(d["north_top"]),
               cell=float(d["cell"]))


# --- primitives -------------------------------------------------------------------------------------------


def box_mean(a: np.ndarray, radius_cells: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window using an integral image; windows are clipped at the edges."""
    r = int(radius_cells)
    if r < 0:
        raise ValueError("radius must be >= 0")
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    rows = np.arange(h)
    cols = np.arange(w)
    r0, r1 = np.clip(rows - r, 0, h), np.clip(rows + r + 1, 0, h)
    c0, c1 = np.clip(cols - r, 0, w), np.clip(cols + r + 1, 0, w)
    total = ii[r1][:, c1] - ii[r0][:, c1] - ii[r1][:, c0] + ii[r0][:, c0]
    count = (r1 - r0)[:, None] * (c1 - c0)[None, :]
    return total / count


def tpi(z: np.ndarray, cell: float, radius_m: float) -> np.ndarray:
    """Topographic position index: elevation minus the mean elevation within `radius_m` (negative = hollow)."""
    return z - box_mean(z, max(1, int(round(radius_m / cell))))


def slope_deg(z: np.ndarray, cell: float) -> np.ndarray:
    gy, gx = np.gradient(z, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def hillshade(z: np.ndarray, cell: float, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(z, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def fill_depressions(z: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Priority-flood fill: raise sinks to their spill level (+eps per step so flats drain to the border)."""
    h, w = z.shape
    filled = z.astype("float64").copy()
    seen = np.zeros((h, w), dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if not seen[r, c]:
                seen[r, c] = True
                heapq.heappush(heap, (filled[r, c], r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not seen[r, c]:
                seen[r, c] = True
                heapq.heappush(heap, (filled[r, c], r, c))
    while heap:
        level, r, c = heapq.heappop(heap)
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc]:
                seen[rr, cc] = True
                if filled[rr, cc] <= level:
                    filled[rr, cc] = level + eps
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def flow_accumulation(filled: np.ndarray, cell: float) -> np.ndarray:
    """D8 contributing area in cells (each cell counts itself) on a depression-free DEM."""
    h, w = filled.shape
    pad = np.pad(filled, 1, mode="constant", constant_values=np.inf)
    best = np.zeros((h, w))
    rec_r = np.arange(h)[:, None] * np.ones((1, w), dtype=int)
    rec_c = np.ones((h, 1), dtype=int) * np.arange(w)[None, :]
    rec_r, rec_c = rec_r.copy(), rec_c.copy()
    for dr, dc in _D8:
        nb = pad[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        drop = (filled - nb) / (cell * (1.4142 if dr and dc else 1.0))
        better = drop > best
        best = np.where(better, drop, best)
        rec_r = np.where(better, np.arange(h)[:, None] + dr, rec_r)
        rec_c = np.where(better, np.arange(w)[None, :] + dc, rec_c)
    acc = np.ones((h, w))
    order = np.argsort(-filled, axis=None)
    flat_r, flat_c = rec_r.ravel(), rec_c.ravel()
    accf = acc.ravel()
    for idx in order:
        rr, cc = flat_r[idx], flat_c[idx]
        if 0 <= rr < h and 0 <= cc < w and (rr * w + cc) != idx:
            accf[rr * w + cc] += accf[idx]
    return accf.reshape(h, w)


def _norm(a: np.ndarray, lo_q: float = 5, hi_q: float = 95) -> np.ndarray:
    lo, hi = np.percentile(a, [lo_q, hi_q])
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def cold_pool_index(dem: Dem) -> dict[str, np.ndarray]:
    """Heuristic 0..1 index of how strongly a cell collects and holds cold air on calm clear nights.

    Combines low position at 600 m and 150 m scale (TPI), drainage convergence (D8 contributing area on the
    depression-filled DEM, log scale), gentle slope (cold air stalls on flat ground) and hollow depth.
    It is a relative terrain index, not a calibrated temperature model.
    """
    z, cell = dem.z, dem.cell
    t600, t150 = tpi(z, cell, 600), tpi(z, cell, 150)
    filled = fill_depressions(z)
    facc = flow_accumulation(filled, cell)
    depth = filled - z
    slope = slope_deg(z, cell)
    flat = 1.0 / (1.0 + slope / 5.0)
    pool = (0.40 * _norm(-t600) + 0.20 * _norm(-t150) + 0.20 * _norm(np.log10(facc)) + 0.10 * flat
            + 0.10 * _norm(depth, 0, 99))
    return dict(pool=np.clip(pool, 0.0, 1.0), tpi600=t600, tpi150=t150, facc=facc, depth=depth, slope=slope)


# --- downscaling Tmin to the slope ------------------------------------------------------------------------------


def radiative_stability(kc: float, wind: float) -> float:
    """0 (cloudy/windy, well mixed) .. 1 (clear and calm, strong surface inversion)."""
    clear = min(max((kc - 0.3) / 0.5, 0.0), 1.0)
    calm = 1.0 / (1.0 + max(wind, 0.0) / 2.0)
    return float(clear * min(calm * 1.6, 1.0))


def fit_gradient(dz: np.ndarray, dT: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Fit (dT/dz) = g_mixed + s * (g_stable - g_mixed) in K/m from paired stations. Returns (g_mixed, g_stable)."""
    x = np.column_stack([np.ones_like(s), s])
    coef = np.linalg.lstsq(x, dT / dz, rcond=None)[0]
    return float(coef[0]), float(coef[0] + coef[1])


def downscale_tmin(dem: Dem, indices: dict[str, np.ndarray], tmin_ref: float, ref_rc: tuple[int, int],
                   stability: float, g_mixed: float, g_stable: float, pool_amp: float = 1.5) -> np.ndarray:
    """Tmin map: reference-station forecast + elevation gradient (stability-weighted) - cold-pool anomaly."""
    z, pool = dem.z, indices["pool"]
    r, c = ref_rc
    grad = g_mixed + stability * (g_stable - g_mixed)
    return tmin_ref + grad * (z - z[r, c]) - pool_amp * stability * (pool - pool[r, c])


def frost_risk_map(tmin_map: np.ndarray, sigma: float, threshold: float = 0.0) -> np.ndarray:
    return frost_probability(tmin_map, np.full_like(tmin_map, sigma), threshold)
