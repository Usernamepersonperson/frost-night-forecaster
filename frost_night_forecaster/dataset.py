"""Turn hourly station data into one row per (station, night) at a given issue hour.

A "night" is identified by the local calendar date D of the evening. The forecast is issued at `issue_hour`
(local clock, default 18:00) and targets the minimum 2 m air temperature (hourly minima, `t2m_min`) from the issue
time until 09:00 local on D+1.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .data import FORECAST_STATIONS, STATIONS, local_to_utc
from .solar import clear_sky_ghi, cos_zenith

END_HOUR_LOCAL = 33  # 09:00 next day
MIN_COVERAGE = 0.8  # share of target-window hours that must have data

FEATURES = [
    "t_now", "td_now", "dep_now", "rh_now", "wind_now", "t5_gap", "d_t3", "d_td6", "t_max_day", "kc", "sunfrac",
    "prev_tmin", "doy", "elev", "d_t_since18",
]


class Station:
    """Hourly arrays on a complete UTC grid with integer-position lookups (fast)."""

    def __init__(self, df: pd.DataFrame, station: str):
        s = df[df["station"] == station].drop_duplicates("time_utc").set_index("time_utc").sort_index()
        self.empty = s.empty
        if self.empty:
            return
        self.start = s.index.min()
        self.index = pd.date_range(s.index.min(), s.index.max(), freq="h")
        s = s.reindex(self.index)
        self.a = {c: s[c].to_numpy(dtype=float) for c in ("t2m", "t2m_min", "t5cm", "rh", "td", "wind", "ghi", "sun")}
        meta = STATIONS[station]
        self.cs = clear_sky_ghi(cos_zenith(meta["lat"], meta["lon"], self.index.values))
        self.n = len(self.index)

    def pos(self, ts: pd.Timestamp) -> int:
        p = int((ts - self.start) / pd.Timedelta(hours=1))
        return p if 0 <= p < self.n else -1

    def val(self, col: str, p: int) -> float:
        return self.a[col][p] if 0 <= p < self.n else np.nan

    def win(self, col: str, lo: int, hi: int) -> np.ndarray:
        """Values at positions lo < p <= hi."""
        return self.a[col][max(lo + 1, 0) : hi + 1]


def night_target(s: Station, d: date, issue_hour: int) -> float:
    lo = s.pos(local_to_utc(d, issue_hour))
    hi = s.pos(local_to_utc(d, END_HOUR_LOCAL))
    if lo < 0 or hi < 0:
        return float("nan")
    w = s.win("t2m_min", lo, hi)
    if len(w) == 0 or np.isnan(w).sum() > (1 - MIN_COVERAGE) * (hi - lo):
        return float("nan")
    return float(np.nanmin(w))


def build_nights(df: pd.DataFrame, issue_hour: int = 18, stations=FORECAST_STATIONS) -> pd.DataFrame:
    """One row per station and night with features at the issue time and the Tmin target."""
    rows = []
    for st in stations:
        meta = STATIONS[st]
        s = Station(df, st)
        if s.empty:
            continue
        for y in sorted(set(s.index.year)):
            d = date(y, 3, 1)
            while d <= date(y, 5, 31):
                p_issue = s.pos(local_to_utc(d, issue_hour))
                p18 = s.pos(local_to_utc(d, 18))
                if p_issue >= 0 and p18 >= 0:
                    rec = _features(s, p_issue, p18)
                    prev = night_target(s, d - timedelta(days=1), issue_hour) if d > date(y, 3, 1) else float("nan")
                    rec.update(station=st, night=pd.Timestamp(d), season=y, tmin=night_target(s, d, issue_hour),
                               prev_tmin=prev, doy=d.timetuple().tm_yday, elev=meta["elev"], issue_hour=issue_hour)
                    rows.append(rec)
                d += timedelta(days=1)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    key = ["t_now", "td_now", "wind_now", "kc", "tmin"]
    return out.dropna(subset=key).reset_index(drop=True)


def _features(s: Station, p: int, p18: int) -> dict:
    t_now, td_now = s.val("t2m", p), s.val("td", p)
    wind3 = s.win("wind", p - 3, p)
    ghi = s.win("ghi", p18 - 10, p18)
    csd = s.cs[max(p18 - 9, 0) : p18 + 1]
    kc = float(ghi.sum() / csd.sum()) if csd.sum() > 50 and not np.isnan(ghi).any() else float("nan")
    sun = s.win("sun", p18 - 10, p18)
    pos = csd > 20
    sunfrac = float(sun[pos].sum() / (60.0 * pos.sum())) if pos.sum() > 0 and not np.isnan(sun[pos]).any() else np.nan
    day_t = s.win("t2m", p18 - 12, p18)
    return dict(
        t_now=t_now, td_now=td_now, dep_now=t_now - td_now, rh_now=s.val("rh", p),
        wind_now=float(np.nanmean(wind3)) if not np.isnan(wind3).all() else float("nan"),
        t5_gap=s.val("t5cm", p) - t_now, d_t3=t_now - s.val("t2m", p - 3), d_td6=td_now - s.val("td", p - 6),
        t_max_day=float(np.nanmax(day_t)) if not np.isnan(day_t).all() else float("nan"),
        kc=min(kc, 1.3) if kc == kc else kc, sunfrac=sunfrac, d_t_since18=t_now - s.val("t2m", p18),
    )
