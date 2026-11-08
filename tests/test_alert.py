import json
from datetime import date, datetime

import numpy as np
import pytest

from frost_night_forecaster.alert import Hysteresis, build_alarm, confidence_label, sms_text, to_json
from frost_night_forecaster.cooling import CoolingProfile, clock_str, learn_profile, observed_crossing
from frost_night_forecaster.dataset import Station

from .conftest import synthetic_hourly

NIGHT = date(2024, 4, 22)
ISSUED = datetime(2024, 4, 22, 18, 0)


def test_hysteresis_switches_on_and_latches():
    h = Hysteresis(p_on=0.3, p_off=0.15, off_updates=2)
    assert [h.update(p) for p in (0.1, 0.35, 0.2, 0.1, 0.4, 0.1, 0.1)] == [False, True, True, True, True, True, False]


def test_hysteresis_resets_counter_when_probability_rises_again():
    h = Hysteresis(p_on=0.3, p_off=0.15, off_updates=2)
    h.update(0.5)
    h.update(0.1)
    h.update(0.2)  # between thresholds resets the "below" counter
    assert h.update(0.1) and not h.update(0.05)


def test_hysteresis_validation():
    with pytest.raises(ValueError):
        Hysteresis(p_on=0.2, p_off=0.5)
    with pytest.raises(ValueError):
        Hysteresis(off_updates=0)


def test_hysteresis_reduces_flapping_on_noisy_series():
    rng = np.random.default_rng(0)
    p = np.clip(0.3 + rng.normal(0, 0.12, 200), 0, 1)
    h = Hysteresis(p_on=0.3, p_off=0.15, off_updates=2)
    latched = [h.update(x) for x in p]
    raw = p >= 0.3
    assert (np.diff(np.array(latched, dtype=int)) != 0).sum() < (np.diff(raw.astype(int)) != 0).sum()


@pytest.mark.parametrize(("t", "expected"), [(-4.0, "high"), (-0.5, "low"), (-1.5, "medium"), (6.0, "high")])
def test_confidence_label(t, expected):
    assert confidence_label(t, 1.5, 0.0) == expected


def test_alarm_message_for_cold_night():
    a = build_alarm("SIO", NIGHT, ISSUED, -3.0, 1.5, 6.0, Hysteresis(p_on=0.3), CoolingProfile.default())
    assert a["schema"] == "frost-alarm/1" and a["alarm"] is True and a["level"] == "alarm"
    assert a["p_frost"]["0"] > a["p_frost"]["-1"] > a["p_frost"]["-2"]
    assert a["tmin_interval_c"][0] < -3.0 < a["tmin_interval_c"][1]
    assert a["start_protection_at"].startswith("2024-04-2")
    assert a["hysteresis"]["state_before"] == "off" and a["hysteresis"]["state_after"] == "on"
    assert "candles" in a["recommendation"]


def test_alarm_message_warm_night_has_no_start_time():
    a = build_alarm("SIO", NIGHT, ISSUED, 6.0, 1.5, 12.0, Hysteresis(), CoolingProfile.default())
    assert a["level"] == "none" and a["start_protection_at"] is None and not a["alarm"]


def test_alarm_watch_level_between_thresholds():
    a = build_alarm("SIO", NIGHT, ISSUED, 1.8, 1.5, 9.0, Hysteresis(p_on=0.6, p_off=0.1), CoolingProfile.default())
    assert a["level"] == "watch" and not a["alarm"]


def test_alarm_latched_stays_on_with_lower_probability():
    state = Hysteresis(p_on=0.3, p_off=0.1, off_updates=2)
    build_alarm("SIO", NIGHT, ISSUED, -2.0, 1.5, 6.0, state, CoolingProfile.default())
    a = build_alarm("SIO", NIGHT, datetime(2024, 4, 22, 21), 0.8, 1.5, 3.0, state, CoolingProfile.default(), update_index=1)
    assert a["alarm"] is True and a["hysteresis"]["state_before"] == "on"


def test_alarm_mqtt_payload_roundtrips_and_topic():
    a = build_alarm("VIS", NIGHT, ISSUED, -3.0, 1.5, 6.0, Hysteresis(), CoolingProfile.default())
    assert a["mqtt"]["topic"] == "frost/vis/alarm" and a["mqtt"]["qos"] == 1
    inner = json.loads(a["mqtt"]["payload"])
    assert inner["station"] == "VIS" and "sms" not in inner and "mqtt" not in inner


def test_sms_text_length_and_content():
    a = build_alarm("SIO", NIGHT, ISSUED, -3.0, 1.5, 6.0, Hysteresis(), CoolingProfile.default())
    assert len(a["sms"]) <= 160 and "FROST ALARM SIO" in a["sms"] and "Start protection" in a["sms"]
    none = {"level": "none", "station": "SIO", "night": "2024-04-22", "tmin_forecast_c": 5.0}
    assert sms_text(none).startswith("Frost SIO")


def test_to_json_serialises_numpy():
    a = build_alarm("SIO", NIGHT, ISSUED, np.float64(-3.0), np.float64(1.5), 6.0, Hysteresis(), CoolingProfile.default())
    assert json.loads(to_json(a))["station"] == "SIO"


# --- cooling curve -----------------------------------------------------------------------------------------


def test_default_profile_is_monotone_unit_interval():
    f = CoolingProfile.default().frac
    assert f[0] == 0.0 and f[-1] == pytest.approx(1.0) and np.all(np.diff(f) >= 0)


def test_time_to_reach_none_when_forecast_stays_above_target():
    assert CoolingProfile.default().time_to_reach(6.0, 2.0, 1.0) is None


def test_time_to_reach_immediate_when_already_below():
    assert CoolingProfile.default().time_to_reach(0.5, -3.0, 1.0, start_hour=21.0) == 21.0


def test_time_to_reach_later_for_lower_target():
    p = CoolingProfile.default()
    t1 = p.time_to_reach(8.0, -4.0, 1.0)
    t2 = p.time_to_reach(8.0, -4.0, -2.0)
    assert 18 <= t1 < t2 <= 33


def test_time_to_reach_linear_profile():
    p = CoolingProfile(frac=np.linspace(0, 1, 16))
    assert p.time_to_reach(10.0, 0.0, 5.0) == pytest.approx(18 + 7.5)


def test_clock_str_wraps_past_midnight():
    assert clock_str(26.5, date(2024, 4, 22)) == "2024-04-23T02:30"
    assert clock_str(19.0, date(2024, 4, 22)) == "2024-04-22T19:00"


def test_learn_profile_recovers_exponential_shape():
    h = synthetic_hourly()
    # cooling nights: T falls linearly 18:00 -> 06:00 local on every night from 5 C to -3 C
    h["t2m"] = 5.0
    idx = h["time_utc"]
    local = idx + __import__("pandas").Timedelta(hours=2)
    hour = local.dt.hour + local.dt.minute / 60
    since18 = np.where(hour >= 18, hour - 18, np.where(hour <= 6, hour + 6, np.nan))
    frac = np.clip(since18 / 12.0, 0, 1)
    h["t2m"] = np.where(np.isnan(since18), 5.0, 5.0 - 8.0 * frac)
    h["t2m_min"] = h["t2m"]
    import pandas as pd

    nights = pd.DataFrame({"station": "SIO", "night": pd.date_range("2020-04-10", periods=5), "t_now": 5.0,
                           "tmin": -3.0, "issue_hour": 18})
    prof = learn_profile(h, nights)
    assert prof.frac[6] == pytest.approx(0.5, abs=0.1)
    assert prof.frac[-1] == pytest.approx(1.0)


def test_observed_crossing_interpolates():
    h = synthetic_hourly()
    s = Station(h, "SIO")
    assert observed_crossing(s, date(2020, 4, 10), 0.0) is not None
    assert observed_crossing(s, date(2020, 4, 20), 0.0) is None


def test_active_alarm_without_start_time_gets_watchful_text():
    # forecast minimum 1.5 C stays above the +1 C trigger but the probability still latches the alarm
    a = build_alarm("SIO", NIGHT, ISSUED, 1.5, 2.5, 8.0, Hysteresis(p_on=0.2, p_off=0.1), CoolingProfile.default())
    assert a["alarm"] and a["start_protection_at"] is None
    assert "stays above" in a["recommendation"]
