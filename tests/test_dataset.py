import numpy as np
import pandas as pd

from frost_night_forecaster.data import load_fixture
from frost_night_forecaster.dataset import MIN_COVERAGE, Station, build_nights, night_target
from frost_night_forecaster.solar import clear_sky_ghi, cos_zenith

from .conftest import synthetic_hourly


def test_target_is_min_of_hourly_minima(hourly):
    n = build_nights(hourly, 18, stations=("SIO",))
    row = n[n["night"] == "2020-04-10"].iloc[0]
    assert row["tmin"] == -3.0
    other = n[n["night"] == "2020-04-12"].iloc[0]
    assert other["tmin"] == 5.0


def test_features_from_constant_series(hourly):
    n = build_nights(hourly, 18, stations=("SIO",))
    row = n[n["night"] == "2020-04-20"].iloc[0]
    assert row["t_now"] == 5.0 and row["dep_now"] == 4.0 and row["wind_now"] == 2.0
    assert row["d_t3"] == 0.0 and row["t_max_day"] == 5.0


def test_prev_tmin_is_previous_night_target(hourly):
    n = build_nights(hourly, 18, stations=("SIO",)).set_index("night")
    assert n.loc["2020-04-11", "prev_tmin"] == -3.0
    assert n.loc["2020-04-12", "prev_tmin"] == 5.0


def test_night_with_too_little_data_is_dropped(hourly):
    h = hourly.copy()
    bad = (h["time_utc"] > "2020-04-14 16:00") & (h["time_utc"] <= "2020-04-15 06:00")
    h.loc[bad, ["t2m_min", "t2m"]] = np.nan
    n = build_nights(h, 18, stations=("SIO",))
    assert pd.Timestamp("2020-04-14") not in set(n["night"])
    assert pd.Timestamp("2020-04-16") in set(n["night"])


def test_missing_station_returns_empty():
    out = build_nights(synthetic_hourly("VIS"), 18, stations=("SIO",))
    assert out.empty


def test_issue_hour_changes_window(hourly):
    n18 = build_nights(hourly, 18, stations=("SIO",)).set_index("night")
    n24 = build_nights(hourly, 24, stations=("SIO",)).set_index("night")
    assert n18.loc["2020-04-10", "tmin"] == -3.0 and n24.loc["2020-04-10", "tmin"] == -3.0
    assert n24.loc["2020-04-10", "issue_hour"] == 24


def test_dst_night_window_is_shorter_but_valid(hourly):
    n = build_nights(hourly, 18, stations=("SIO",)).set_index("night")
    assert pd.Timestamp("2020-03-28") in n.index  # night with the clock change on 29 March 2020


def test_station_helpers(hourly):
    s = Station(hourly, "SIO")
    assert s.pos(pd.Timestamp("2020-03-01 01:00")) == 0
    assert s.pos(pd.Timestamp("2019-01-01")) == -1
    assert np.isnan(s.val("t2m", -5))
    assert np.isnan(night_target(s, pd.Timestamp("2021-04-01").date(), 18))


def test_min_coverage_constant():
    assert 0 < MIN_COVERAGE < 1


def test_real_fixture_sion_2024_has_frost_night():
    df = load_fixture()
    df = df[(df["station"] == "SIO") & (df["time_utc"].dt.year == 2024)]
    n = build_nights(df, 18, stations=("SIO",))
    assert (n["tmin"] <= 0).sum() >= 1
    assert n["kc"].between(0, 1.3).all()


def test_cos_zenith_noon_equinox():
    c = cos_zenith(46.0, 7.5, np.array(["2024-03-20T11:30:00"], dtype="datetime64[s]"))
    assert abs(c[0] - np.cos(np.radians(46.0))) < 0.03


def test_cos_zenith_night_is_negative():
    c = cos_zenith(46.0, 7.5, np.array(["2024-04-20T23:00:00"], dtype="datetime64[s]"))
    assert c[0] < 0


def test_clear_sky_ghi_values():
    assert clear_sky_ghi(np.array([-0.5, 0.0]))[0] == 0.0
    assert abs(clear_sky_ghi(np.array([1.0]))[0] - 1098 * np.exp(-0.057)) < 1e-6
