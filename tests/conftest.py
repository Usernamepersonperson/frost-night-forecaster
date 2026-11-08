from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frost_night_forecaster.data import COLUMNS


def synthetic_hourly(station: str = "SIO", year: int = 2020, frost_night: str = "2020-04-10") -> pd.DataFrame:
    """Hourly UTC data 1 Mar .. 2 Jun: constant 5 C, no radiation, except one night at -3 C around 02:00 UTC."""
    idx = pd.date_range(f"{year}-03-01 01:00", f"{year}-06-02 00:00", freq="h")
    df = pd.DataFrame({"station": station, "time_utc": idx})
    df["t2m"] = 5.0
    df["t2m_min"] = 5.0
    df["t5cm"] = 4.0
    df["rh"] = 70.0
    df["td"] = 1.0
    df["wind"] = 2.0
    df["ghi"] = 0.0
    df["sun"] = 0.0
    night = pd.Timestamp(frost_night)
    hit = (df["time_utc"] > night + pd.Timedelta(hours=23)) & (df["time_utc"] <= night + pd.Timedelta(hours=26))
    df.loc[hit, "t2m_min"] = -3.0
    df.loc[hit, "t2m"] = -2.0
    return df[COLUMNS]


@pytest.fixture
def hourly() -> pd.DataFrame:
    return synthetic_hourly()


def synthetic_nights(seed: int = 0, seasons=range(2010, 2016), per_season: int = 250) -> pd.DataFrame:
    """Nights where the true cooling follows the physical terms plus a hidden dependence on T5cm gap."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        n = per_season
        t_now = rng.normal(10, 4, n)
        dep = np.abs(rng.normal(6, 3, n))
        kc = rng.uniform(0.05, 1.1, n)
        wind = rng.uniform(0.3, 8, n)
        gap = rng.normal(-1, 1, n)
        drop = 2 + 0.25 * dep + 3.0 * kc / (1 + wind / 2) + 0.8 * gap
        tmin = t_now - np.clip(drop, 0, None) + rng.normal(0, 0.8, n)
        rows.append(pd.DataFrame({
            "station": rng.choice(["SIO", "VIS"], n), "season": s,
            "night": pd.Timestamp(f"{s}-03-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D"),
            "t_now": t_now, "td_now": t_now - dep, "dep_now": dep, "rh_now": rng.uniform(30, 100, n),
            "wind_now": wind, "t5_gap": gap, "d_t3": rng.normal(-1, 1, n), "d_td6": rng.normal(0, 1, n),
            "t_max_day": t_now + 4, "kc": kc, "sunfrac": kc, "prev_tmin": tmin + rng.normal(0, 2, n),
            "doy": rng.integers(60, 150, n), "elev": np.where(rng.random(n) > 0.5, 482.0, 639.0),
            "d_t_since18": rng.normal(-2, 1, n), "tmin": tmin, "issue_hour": 18,
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def nights() -> pd.DataFrame:
    return synthetic_nights()
