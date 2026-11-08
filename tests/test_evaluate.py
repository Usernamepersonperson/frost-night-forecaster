import numpy as np
import pytest

from frost_night_forecaster.cli import main, summary
from frost_night_forecaster.evaluate import (
    brier,
    choose_p_on,
    classification,
    count_toggles,
    mae,
    reliability,
    rmse,
    split,
)


def test_mae_rmse_known_values():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_classification_counts_and_rates():
    obs = np.array([-1.0, -2.0, 1.0, 2.0, 3.0, -0.5])
    alarm = np.array([True, False, True, False, False, True])
    c = classification(obs, alarm, 0.0)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2)
    assert c["hit_rate"] == pytest.approx(2 / 3)
    assert c["false_alarm_ratio"] == pytest.approx(1 / 3)
    assert c["false_positive_rate"] == pytest.approx(1 / 3)


def test_classification_no_events_gives_nan_hit_rate():
    c = classification([5.0, 6.0], [False, False], 0.0)
    assert np.isnan(c["hit_rate"]) and c["n_frost"] == 0


def test_classification_threshold_is_inclusive():
    assert classification([0.0], [True], 0.0)["tp"] == 1


def test_choose_p_on_reaches_target_recall():
    rng = np.random.default_rng(0)
    obs = rng.normal(2, 3, 2000)
    p = np.clip(1 / (1 + np.exp(obs)) + rng.normal(0, 0.05, 2000), 0, 1)
    cut = choose_p_on(obs, p, 0.0, target_recall=0.85)
    ev = obs <= 0
    assert (p[ev] >= cut).mean() >= 0.85
    assert (p[ev] >= cut + 0.02).mean() < 0.9  # near the largest cut that still works


def test_choose_p_on_without_events_default():
    assert choose_p_on([5.0, 6.0], [0.1, 0.2], 0.0) == 0.5


def test_brier_perfect_and_worst():
    assert brier([1, 0], [True, False]) == 0.0
    assert brier([0, 1], [True, False]) == 1.0


def test_reliability_bins_are_consistent():
    p = np.array([0.02, 0.04, 0.2, 0.9, 0.95])
    ev = np.array([False, False, False, True, True])
    r = reliability(p, ev)
    assert sum(b["n"] for b in r) == 5
    assert r[-1]["observed"] == 1.0


def test_count_toggles():
    assert count_toggles([False, True, True, False, True]) == 3
    assert count_toggles([True]) == 0


def test_split_by_season(nights):
    tr, te = split(nights, 2012)
    assert tr["season"].max() == 2012 and te["season"].min() == 2013 and len(tr) + len(te) == len(nights)


def test_cli_rejects_unknown_station(capsys):
    assert main(["forecast", "--station", "XXX", "--night", "2024-04-22"]) == 2
    assert "unknown station" in capsys.readouterr().err


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "0.1.0" in capsys.readouterr().out


def test_summary_shape():
    res = {"test_seasons": [2021, 2026], "tmin_18": {"a": {"mae": 1.234}},
           "frost_18": {"0": {"p_on": 0.2, "hit_rate": 0.8, "false_alarm_ratio": 0.4, "false_positive_rate": 0.1,
                              "n_frost": 10}},
           "lead_time": {"median_lead_h": 5.0},
           "case_april_2024": {"frost_nights_all_stations": 3, "alarmed_all_stations": 3, "false_alarms_all_stations": 1}}
    s = summary(res)
    assert s["tmin_mae_k"] == {"a": 1.23} and s["median_lead_h"] == 5.0
