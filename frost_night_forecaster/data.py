"""MeteoSwiss SwissMetNet hourly data (open data) and the cached spring fixture.

Source: MeteoSwiss, https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn (open data, source must be cited).
Timestamps in the raw files are UTC and mark the END of the hourly interval.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
DATA_DIR = Path(__file__).parent / "data"
FIXTURE = DATA_DIR / "meteoswiss_spring.csv.gz"

# MeteoSwiss station metadata (ogd-smn_meta_stations.csv): name, height m a.s.l., WGS84 lat/lon, LV95 E/N
STATIONS: dict[str, dict] = {
    "SIO": dict(name="Sion", elev=482.0, lat=46.21865, lon=7.330203, e=2591633.0, n=1118584.0, region="Valais"),
    "VIS": dict(name="Visp", elev=639.0, lat=46.3029, lon=7.842958, e=2631151.0, n=1128024.0, region="Valais"),
    "AIG": dict(name="Aigle", elev=381.0, lat=46.326647, lon=6.924472, e=2560404.0, n=1130713.0, region="Vaud"),
    "CGI": dict(name="Nyon / Changins", elev=458.0, lat=46.401053, lon=6.227722, e=2506878.0, n=1139574.0,
                region="Vaud"),
    "MAG": dict(name="Magadino / Cadenazzo", elev=203.0, lat=46.160025, lon=8.933672, e=2715480.0, n=1113161.0,
                region="Ticino"),
    "CHU": dict(name="Chur", elev=556.0, lat=46.870572, lon=9.530761, e=2759485.0, n=1193176.0,
                region="Graubuenden"),
    "MVE": dict(name="Montana", elev=1423.0, lat=46.298806, lon=7.460814, e=2601709.0, n=1127489.0,
                region="Valais (slope reference)"),
}
FORECAST_STATIONS = ("SIO", "VIS", "AIG", "CGI", "MAG", "CHU")

# MeteoSwiss hourly parameter id -> our column
PARAMS = {
    "tre200h0": "t2m",  # air temperature 2 m, hourly mean (deg C)
    "tre200hn": "t2m_min",  # air temperature 2 m, hourly minimum
    "tre005h0": "t5cm",  # air temperature 5 cm above ground
    "ure200h0": "rh",  # relative humidity 2 m (%)
    "tde200h0": "td",  # dew point 2 m (deg C)
    "fkl010h0": "wind",  # wind speed scalar 10 m (m/s)
    "gre000h0": "ghi",  # global radiation (W/m2)
    "sre000h0": "sun",  # sunshine duration (min per hour)
}
COLUMNS = ["station", "time_utc", *PARAMS.values()]

SPRING_YEARS = tuple(range(2010, 2027))


def parse_hourly_csv(text: str) -> pd.DataFrame:
    """Parse one MeteoSwiss hourly CSV into the tidy layout `COLUMNS` (time_utc = end of interval)."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    need = {"station_abbr", "reference_timestamp", "tre200h0"}
    if not need <= set(df.columns):
        raise ValueError(f"unexpected MeteoSwiss file, missing columns {sorted(need - set(df.columns))}")
    out = pd.DataFrame(
        {
            "station": df["station_abbr"].str.upper(),
            "time_utc": pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M"),
        }
    )
    for pid, col in PARAMS.items():
        out[col] = pd.to_numeric(df[pid], errors="coerce") if pid in df.columns else np.nan
    return out[COLUMNS]


def spring_mask(times: pd.Series) -> pd.Series:
    """Frost season window kept in the fixture: 1 March 01:00 UTC .. 1 June 09:00 UTC (covers all nights)."""
    doy_ok = (times.dt.month.isin([3, 4, 5])) | ((times.dt.month == 6) & (times.dt.day == 1) & (times.dt.hour <= 9))
    return doy_ok & (times.dt.year.isin(SPRING_YEARS))


def load_fixture(path: Path | str | None = None) -> pd.DataFrame:
    """Load the cached spring fixture (all stations)."""
    df = pd.read_csv(path or FIXTURE, parse_dates=["time_utc"])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"fixture is missing columns {sorted(missing)}")
    return df.sort_values(["station", "time_utc"]).reset_index(drop=True)


# --- Swiss local time (CET/CEST) without a tz database ------------------------------------------------


def _last_sunday(year: int, month: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def zurich_offset_hours(ts: datetime | pd.Timestamp) -> int:
    """UTC offset in hours of Switzerland for a UTC instant (EU rule: last Sunday of March..October, 01:00 UTC)."""
    t = pd.Timestamp(ts)
    start = datetime.combine(_last_sunday(t.year, 3), datetime.min.time()) + timedelta(hours=1)
    end = datetime.combine(_last_sunday(t.year, 10), datetime.min.time()) + timedelta(hours=1)
    return 2 if start <= t.to_pydatetime().replace(tzinfo=None) < end else 1


def offset_for_local_date(d: date) -> int:
    """UTC offset on a calendar date, evaluated at 12:00 local (transitions happen at night)."""
    return zurich_offset_hours(datetime(d.year, d.month, d.day, 12) - timedelta(hours=1))


def local_to_utc(d: date, hour: int) -> pd.Timestamp:
    """Local clock `hour` (0..47 allowed, >=24 means next day) on date `d` -> naive UTC timestamp."""
    base = datetime(d.year, d.month, d.day) + timedelta(hours=hour)
    off = offset_for_local_date(base.date())
    return pd.Timestamp(base - timedelta(hours=off))


def utc_to_local(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.Timedelta(hours=zurich_offset_hours(ts))
