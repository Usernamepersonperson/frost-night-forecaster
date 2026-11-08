"""Tiny solar-geometry helpers used to turn global radiation into a cloudiness proxy."""

from __future__ import annotations

import numpy as np


def cos_zenith(lat_deg: float, lon_deg: float, times_utc) -> np.ndarray:
    """Cosine of the solar zenith angle for UTC instants (declination/equation of time, +-0.5 deg accuracy)."""
    t = np.asarray(times_utc, dtype="datetime64[s]")
    doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    hour = (t - t.astype("datetime64[D]")).astype("timedelta64[s]").astype(float) / 3600.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    solar_time = hour + lon_deg / 15.0 + eqt / 60.0
    ha = np.deg2rad(15.0 * (solar_time - 12.0))
    lat = np.deg2rad(lat_deg)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)


def clear_sky_ghi(cosz: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance (W/m2)."""
    c = np.clip(cosz, 0.0, None)
    out = np.zeros_like(c, dtype=float)
    pos = c > 0.01
    out[pos] = 1098.0 * c[pos] * np.exp(-0.057 / c[pos])
    return out
