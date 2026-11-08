from datetime import date

import pandas as pd
import pytest

from frost_night_forecaster.data import (
    COLUMNS,
    FIXTURE,
    FORECAST_STATIONS,
    STATIONS,
    load_fixture,
    local_to_utc,
    offset_for_local_date,
    parse_hourly_csv,
    spring_mask,
    utc_to_local,
    zurich_offset_hours,
)

CSV = (
    "station_abbr;reference_timestamp;tre200h0;tre200hn;tre005h0;ure200h0;tde200h0;fkl010h0;gre000h0;sre000h0\n"
    "sio;01.04.2024 17:00;4.5;4.1;3.0;80;1.5;1.2;0;0\n"
    "sio;01.04.2024 18:00;-;3.0;;81;1.0;0.9;0;0\n"
)


def test_parse_hourly_csv_values():
    df = parse_hourly_csv(CSV)
    assert list(df.columns) == COLUMNS
    assert df["station"].tolist() == ["SIO", "SIO"]
    assert df["time_utc"].iloc[0] == pd.Timestamp("2024-04-01 17:00")
    assert df["t2m"].iloc[0] == 4.5


def test_parse_hourly_csv_missing_values_become_nan():
    df = parse_hourly_csv(CSV)
    assert pd.isna(df["t2m"].iloc[1]) and pd.isna(df["t5cm"].iloc[1])


def test_parse_hourly_csv_missing_parameter_column_is_nan():
    text = "station_abbr;reference_timestamp;tre200h0\nSIO;01.04.2024 17:00;4.5\n"
    df = parse_hourly_csv(text)
    assert df["wind"].isna().all() and df["t2m"].iloc[0] == 4.5


def test_parse_hourly_csv_rejects_foreign_file():
    with pytest.raises(ValueError, match="missing columns"):
        parse_hourly_csv("a;b\n1;2\n")


def test_spring_mask_window():
    t = pd.Series(pd.to_datetime(["2024-02-28 12:00", "2024-03-01 01:00", "2024-05-31 23:00", "2024-06-01 09:00",
                                  "2024-06-01 12:00", "2009-04-01 00:00"]))
    assert spring_mask(t).tolist() == [False, True, True, True, False, False]


@pytest.mark.parametrize(
    ("ts", "off"),
    [("2024-03-31 00:59", 1), ("2024-03-31 01:00", 2), ("2024-10-27 00:59", 2), ("2024-10-27 01:00", 1),
     ("2024-01-15 12:00", 1), ("2024-07-15 12:00", 2), ("2021-03-28 01:00", 2), ("2025-03-30 00:00", 1)],
)
def test_zurich_offset(ts, off):
    assert zurich_offset_hours(pd.Timestamp(ts)) == off


def test_offset_for_local_date():
    assert offset_for_local_date(date(2024, 3, 30)) == 1
    assert offset_for_local_date(date(2024, 3, 31)) == 2


def test_local_to_utc_summer_and_winter():
    assert local_to_utc(date(2024, 4, 22), 18) == pd.Timestamp("2024-04-22 16:00")
    assert local_to_utc(date(2024, 3, 1), 18) == pd.Timestamp("2024-03-01 17:00")


def test_local_to_utc_next_day_hour():
    assert local_to_utc(date(2024, 4, 22), 33) == pd.Timestamp("2024-04-23 07:00")


def test_utc_to_local_roundtrip():
    t = local_to_utc(date(2024, 4, 22), 20)
    assert utc_to_local(t) == pd.Timestamp("2024-04-22 20:00")


def test_station_table_complete():
    for st in FORECAST_STATIONS:
        meta = STATIONS[st]
        assert 45.5 < meta["lat"] < 47.5 and 5.9 < meta["lon"] < 10.5 and meta["elev"] > 100


def test_fixture_contents():
    df = load_fixture()
    assert set(STATIONS) <= set(df["station"].unique())
    assert df["time_utc"].dt.year.min() == 2010 and df["time_utc"].dt.year.max() == 2026
    assert df["time_utc"].dt.month.isin([3, 4, 5, 6]).all()
    assert df.duplicated(["station", "time_utc"]).sum() == 0


def test_fixture_april_2024_frost_in_data():
    df = load_fixture()
    a = df[(df["time_utc"] >= "2024-04-15") & (df["time_utc"] < "2024-05-01") & df["station"].isin(FORECAST_STATIONS)]
    assert a["t2m_min"].min() < -1.0


def test_load_fixture_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("station,time_utc,t2m\nSIO,2024-04-01 00:00,3\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_fixture(p)


def test_fixture_file_exists():
    assert FIXTURE.exists()
