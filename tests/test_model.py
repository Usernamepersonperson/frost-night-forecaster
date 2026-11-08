import numpy as np
import pandas as pd
import pytest

from frost_night_forecaster.model import (
    Climatology,
    FrostModel,
    LinearModel,
    PhysicsBaseline,
    frost_probability,
    norm_cdf,
    persistence,
    physics_design,
)


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.0) == pytest.approx(0.8413447, abs=1e-6)
    assert norm_cdf(-1.96) == pytest.approx(0.0249979, abs=1e-6)


def test_frost_probability_at_threshold_is_half():
    assert frost_probability(-1.0, 2.0, -1.0) == pytest.approx(0.5)


def test_frost_probability_monotone_in_forecast():
    p = frost_probability(np.array([-4.0, -1.0, 0.0, 3.0]), 2.0, 0.0)
    assert np.all(np.diff(p) < 0) and p[0] > 0.95 and p[-1] < 0.1


def test_frost_probability_lower_threshold_is_rarer():
    p0, p2 = frost_probability(0.5, 2.0, 0.0), frost_probability(0.5, 2.0, -2.0)
    assert p2 < p0


def test_climatology_means_by_station_and_doy():
    df = pd.DataFrame({"station": ["SIO"] * 4 + ["VIS"] * 2, "doy": [100, 100, 101, 140, 100, 100],
                       "tmin": [2.0, 4.0, 3.0, 9.0, -1.0, -3.0]})
    clim = Climatology().fit(df)
    out = clim.predict(pd.DataFrame({"station": ["SIO", "VIS", "SIO"], "doy": [100, 100, 140]}))
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx(-2.0)
    assert out[2] == pytest.approx(9.0)


def test_climatology_unknown_station_is_nan():
    clim = Climatology().fit(pd.DataFrame({"station": ["SIO"], "doy": [100], "tmin": [1.0]}))
    assert np.isnan(clim.predict(pd.DataFrame({"station": ["XXX"], "doy": [100]}))[0])


def test_persistence_falls_back_to_climatology():
    df = pd.DataFrame({"prev_tmin": [1.0, np.nan]})
    assert persistence(df, np.array([5.0, 6.0])).tolist() == [1.0, 6.0]


def test_physics_design_shape_and_wind_effect():
    df = pd.DataFrame({"dep_now": [5.0, 5.0], "kc": [1.0, 1.0], "wind_now": [0.0, 8.0]})
    x = physics_design(df)
    assert x.shape == (2, 6)
    assert x[0, 3] > x[1, 3]  # calm clear sky cools more than windy clear sky


def test_baseline_recovers_exact_linear_drop():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"dep_now": rng.uniform(0, 10, n), "kc": rng.uniform(0, 1, n), "wind_now": rng.uniform(0, 6, n),
                       "t_now": rng.normal(10, 3, n)})
    true = np.array([1.0, 0.2, 0.5, 2.0, 0.1, 0.3])
    df["tmin"] = df["t_now"] - physics_design(df) @ true
    pred = PhysicsBaseline().fit(df).predict(df)
    assert np.max(np.abs(pred - df["tmin"])) < 1e-6


def test_baseline_requires_fit():
    with pytest.raises(RuntimeError):
        PhysicsBaseline().predict(pd.DataFrame())


def test_frost_model_beats_baseline_on_hidden_signal(nights):
    train, test = nights[nights["season"] < 2014], nights[nights["season"] >= 2014]
    m = FrostModel(max_iter=60).fit(train, oof_sigma=False)
    err_model = np.abs(m.predict(test) - test["tmin"]).mean()
    err_base = np.abs(m.baseline.predict(test) - test["tmin"]).mean()
    assert err_model < err_base
    assert err_model < 1.2  # noise floor is 0.8 K


def test_frost_model_requires_fit(nights):
    with pytest.raises(RuntimeError):
        FrostModel().predict(nights.head(3))


def test_frost_model_sigma_regimes(nights):
    train = nights[nights["season"] < 2014]
    m = FrostModel(max_iter=40).fit(train, oof_sigma=True)
    assert 0.3 < m.sigma_clear < 3 and 0.3 < m.sigma_other < 3
    assert m.oof_res is not None and len(m.oof_res) == len(train) and not np.isnan(m.oof_res).any()
    sig = m.sigma(pd.DataFrame({"kc": [0.1, 0.9]}))
    assert sig[0] == m.sigma_other and sig[1] == m.sigma_clear


def test_frost_model_probabilities_in_unit_interval(nights):
    m = FrostModel(max_iter=30).fit(nights[nights["season"] < 2014], oof_sigma=False)
    p = m.p_frost(nights[nights["season"] >= 2014], 0.0)
    assert p.min() >= 0 and p.max() <= 1


def test_frost_model_is_deterministic(nights):
    tr = nights[nights["season"] < 2014]
    a = FrostModel(max_iter=30).fit(tr, oof_sigma=False).predict(nights.head(50))
    b = FrostModel(max_iter=30).fit(tr, oof_sigma=False).predict(nights.head(50))
    assert np.allclose(a, b)


def test_linear_model_fits(nights):
    tr, te = nights[nights["season"] < 2014], nights[nights["season"] >= 2014]
    base = PhysicsBaseline().fit(tr)
    lin = LinearModel().fit(tr, base.predict(tr))
    err = np.abs(lin.predict(te, base.predict(te)) - te["tmin"]).mean()
    assert err < 1.5
