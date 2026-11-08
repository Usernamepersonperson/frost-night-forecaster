"""End-to-end demo: train, evaluate on held-out seasons, build the cold-pool map, write figures and results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .alert import to_json
from .data import FIXTURE, STATIONS, load_fixture
from .dataset import build_nights
from .evaluate import Experiment, example_alarm, run_experiment
from .model import frost_probability
from .terrain import (
    cold_pool_index,
    downscale_tmin,
    fit_gradient,
    frost_risk_map,
    load_dem,
    radiative_stability,
)

POOL_AMPLITUDE_C = 1.5  # assumption: Tmin offset between the least and most pooling-prone terrain on a calm clear night


def regional_gradient(fixture: pd.DataFrame, train_last: int) -> dict:
    """Night-time Tmin gradient with height from Sion (482 m) to Montana (1423 m), by radiative stability."""
    n = build_nights(fixture, 18, stations=("SIO", "MVE"))
    n = n[n["season"] <= train_last]
    a = n[n["station"] == "SIO"].set_index("night")
    b = n[n["station"] == "MVE"].set_index("night")
    j = a.join(b, how="inner", lsuffix="_s", rsuffix="_m")
    dz = STATIONS["MVE"]["elev"] - STATIONS["SIO"]["elev"]
    s = np.array([radiative_stability(k, w) for k, w in zip(j["kc_s"], j["wind_now_s"], strict=True)])
    g_mixed, g_stable = fit_gradient(np.full(len(j), dz), (j["tmin_m"] - j["tmin_s"]).to_numpy(), s)
    return dict(g_mixed_k_per_m=g_mixed, g_stable_k_per_m=g_stable, n_night_pairs=int(len(j)), dz_m=dz)


def build_case_map(exp: Experiment, grad: dict, station: str = "VIS"):
    dem = load_dem()
    idx = cold_pool_index(dem)
    case = exp.case[(exp.case["station"] == station) & (exp.case["tmin"] <= 0)].copy()
    case["stab"] = [radiative_stability(k, w) for k, w in zip(case["kc"], case["wind_now"], strict=True)]
    row = case.loc[case["stab"].idxmax()]  # observed frost night with the most radiative (clear, calm) evening
    rc = dem.rowcol(STATIONS[station]["e"], STATIONS[station]["n"])
    s = radiative_stability(row["kc"], row["wind_now"])
    tmap = downscale_tmin(dem, idx, row["tmin_hat"], rc, s, grad["g_mixed_k_per_m"], grad["g_stable_k_per_m"],
                          POOL_AMPLITUDE_C)
    risk = frost_risk_map(tmap, row["sigma"], 0.0)
    info = dict(night=str(row["night"].date()), station=station, tmin_forecast_c=float(row["tmin_hat"]),
                tmin_observed_c=float(row["tmin"]), stability=s, pool_amplitude_c=POOL_AMPLITUDE_C,
                tmap_p05_c=float(np.percentile(tmap, 5)), tmap_p50_c=float(np.percentile(tmap, 50)),
                tmap_p95_c=float(np.percentile(tmap, 95)),
                share_area_p_frost_ge_50=float((risk >= 0.5).mean()), p_frost_at_station=float(risk[rc]))
    return dem, rc, risk, info


def _say(msg: str) -> None:
    print(msg, flush=True)


def run_demo(out_dir: Path, fixture_path: Path | None = None, progress=_say) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "docs").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    fixture = load_fixture(fixture_path or FIXTURE)
    exp = run_experiment(fixture, progress=progress)
    progress("terrain: cold-pool index and regional gradient")
    grad = regional_gradient(fixture, exp.train_last)
    dem, rc, risk, info = build_case_map(exp, grad)
    res = dict(exp.results)
    res["regional_gradient"] = grad
    res["map_case"] = info
    res["cooling_profile_fraction_by_hour_after_18"] = [round(float(x), 3) for x in exp.profile.frac]
    (out_dir / "results" / "metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    exp.test18.assign(night=exp.test18["night"].dt.date).to_csv(out_dir / "results" / "heldout_predictions_18h.csv",
                                                                 index=False, float_format="%.3f")
    alarm = example_alarm(exp)
    (out_dir / "docs" / "example_alarm.json").write_text(to_json(alarm), encoding="utf-8")
    p_on = res["p_on"]["0"]
    label = info["night"]
    progress("figures")
    plots.main_figure(out_dir / "docs" / "figure.png", dem, risk, rc, res, exp.test18, exp.case, p_on, label)
    plots.post_figure(out_dir / "docs" / "post-figure.png", dem, risk, rc, exp.case, p_on, label)
    return res


def forecast_night(station: str, night: str, fixture_path: Path | None = None, train_last: int = 2020) -> dict:
    """Train the 18:00 model on seasons <= train_last and return the alarm message for one station and night."""
    from datetime import date, datetime

    from .alert import Hysteresis, build_alarm
    from .cooling import learn_profile
    from .evaluate import choose_p_on, split
    from .model import FrostModel

    fixture = load_fixture(fixture_path or FIXTURE)
    nights = build_nights(fixture, 18)
    train, _ = split(nights, train_last)
    model = FrostModel().fit(train)
    d = date.fromisoformat(night)
    row = nights[(nights["station"] == station.upper()) & (nights["night"] == pd.Timestamp(d))]
    if row.empty:
        raise SystemExit(f"no data for {station} on {night} (fixture covers March-May 2010-2026)")
    oof_hat = train["tmin"].to_numpy() - model.oof_res
    p_on = choose_p_on(train["tmin"], frost_probability(oof_hat, model.sigma(train), 0.0), 0.0)
    state = Hysteresis(p_on=p_on, p_off=p_on * 0.5, off_updates=2)
    r = row.iloc[0]
    return build_alarm(station.upper(), d, datetime(d.year, d.month, d.day, 18), float(model.predict(row)[0]),
                       float(model.sigma(row)[0]), float(r["t_now"]), state, learn_profile(fixture, train))
