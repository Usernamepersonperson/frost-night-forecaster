"""Night cooling-curve shape: when will the 2 m air temperature reach the protection trigger?

The Tmin forecast says how cold the night gets; the curve says when. The shape is learned from past
nights: for each hour after 18:00 local, the median share of the total evening-to-minimum drop already
reached. It is applied to the forecast drop of the night at hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .data import local_to_utc
from .dataset import END_HOUR_LOCAL, Station

REF_HOUR = 18  # curve reference (local clock)
N_HOURS = END_HOUR_LOCAL - REF_HOUR + 1  # 0..15 hours after 18:00


@dataclass
class CoolingProfile:
    """frac[k] = share of the 18:00 -> Tmin drop reached k hours after 18:00 local (monotone, 0 .. 1)."""

    frac: np.ndarray

    @classmethod
    def default(cls) -> CoolingProfile:
        return cls(frac=np.clip(1 - np.exp(-np.arange(N_HOURS) / 4.0), 0, 1) / (1 - np.exp(-(N_HOURS - 1) / 4.0)))

    def fraction_at(self, hours_after_18: float) -> float:
        x = np.clip(hours_after_18, 0, N_HOURS - 1)
        return float(np.interp(x, np.arange(N_HOURS), self.frac))

    def time_to_reach(self, t_start: float, tmin_hat: float, target: float, start_hour: float = REF_HOUR) -> float | None:
        """Local clock hour (float, 18..33) when T falls to `target`, or None if the forecast never gets there.

        `t_start` is the temperature at `start_hour`; the curve is renormalised from that point.
        """
        if t_start <= target:
            return float(start_hour)
        if tmin_hat > target:
            return None
        f0 = self.fraction_at(start_hour - REF_HOUR)
        drop_total = t_start - tmin_hat
        if drop_total <= 0 or f0 >= 1:
            return None
        need = (t_start - target) / drop_total  # share of the remaining drop needed
        goal = f0 + need * (1 - f0)
        if goal > self.frac[-1] + 1e-9:
            return None
        k = int(np.searchsorted(self.frac, goal, side="left"))
        if k == 0:
            return float(REF_HOUR)
        lo, hi = self.frac[k - 1], self.frac[k]
        frac_k = (goal - lo) / (hi - lo) if hi > lo else 0.0
        return float(REF_HOUR + k - 1 + frac_k)


def learn_profile(df: pd.DataFrame, nights: pd.DataFrame, min_drop: float = 3.0) -> CoolingProfile:
    """Median normalised cooling curve over the given (training) nights, from hourly 2 m temperatures."""
    curves = []
    for st, g in nights.groupby("station"):
        s = Station(df, st)
        if s.empty:
            continue
        for night, t18, tmin in zip(g["night"], g["t_now"], g["tmin"], strict=True):
            drop = t18 - tmin
            if drop < min_drop or g["issue_hour"].iloc[0] != REF_HOUR:
                continue
            d = night.date()
            base = s.pos(local_to_utc(d, REF_HOUR))
            if base < 0 or base + N_HOURS >= s.n:
                continue
            seg = s.a["t2m"][base : base + N_HOURS]
            if np.isnan(seg).any():
                continue
            curves.append(np.clip((seg[0] - seg) / drop, 0, 1))
    if not curves:
        return CoolingProfile.default()
    prof = np.maximum.accumulate(np.median(np.vstack(curves), axis=0))
    prof[0] = 0.0
    top = prof[-1] if prof[-1] > 0 else 1.0
    return CoolingProfile(frac=prof / top)


def observed_crossing(s: Station, d: date, target: float, start_hour: float = REF_HOUR) -> float | None:
    """First local clock hour (float) after `start_hour` at which the hourly 2 m temperature is <= target."""
    base = s.pos(local_to_utc(d, REF_HOUR))
    if base < 0 or base + N_HOURS >= s.n:
        return None
    seg = s.a["t2m"][base : base + N_HOURS]
    for k in range(int(start_hour - REF_HOUR), N_HOURS):
        if np.isnan(seg[k]):
            continue
        if seg[k] <= target:
            if k == 0 or np.isnan(seg[k - 1]) or seg[k - 1] <= seg[k]:
                return float(REF_HOUR + k)
            return float(REF_HOUR + k - 1 + (seg[k - 1] - target) / (seg[k - 1] - seg[k]))
    return None


def clock_str(hour_float: float, d: date) -> str:
    """ISO-like local timestamp for a local clock hour counted from 00:00 of date `d` (can exceed 24)."""
    mins = int(round(hour_float * 60))
    t = pd.Timestamp(d) + timedelta(minutes=mins)
    return t.strftime("%Y-%m-%dT%H:%M")
