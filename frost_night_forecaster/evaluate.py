"""Held-out evaluation: Tmin errors, frost hit / false-alarm rates, lead times, hysteresis, April 2024 case study."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .alert import PROTECTION_MARGIN_C, Hysteresis, build_alarm
from .cooling import CoolingProfile, learn_profile, observed_crossing
from .data import FORECAST_STATIONS, STATIONS
from .dataset import Station, build_nights
from .model import THRESHOLDS, Climatology, FrostModel, LinearModel, frost_probability, persistence

TRAIN_LAST_SEASON = 2020
ISSUE_HOURS = (18, 21, 24)  # local clock; 24 = 00:00 after midnight
TARGET_RECALL = 0.85
CASE_START, CASE_END = date(2024, 4, 15), date(2024, 5, 1)


def _say_(msg: str) -> None:
    print(msg, flush=True)


# --- metric primitives ---------------------------------------------------------------------------------------


def mae(pred, obs) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float))))


def rmse(pred, obs) -> float:
    d = np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def classification(tmin_obs, alarm, threshold: float) -> dict:
    """Confusion counts and rates for a boolean alarm vs the event Tmin <= threshold."""
    ev = np.asarray(tmin_obs, dtype=float) <= threshold
    al = np.asarray(alarm, dtype=bool)
    tp, fp = int((al & ev).sum()), int((al & ~ev).sum())
    fn, tn = int((~al & ev).sum()), int((~al & ~ev).sum())
    return dict(
        threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, n_frost=tp + fn, n_nights=tp + fp + fn + tn,
        hit_rate=tp / (tp + fn) if tp + fn else float("nan"),
        false_alarm_ratio=fp / (tp + fp) if tp + fp else float("nan"),
        false_positive_rate=fp / (fp + tn) if fp + tn else float("nan"),
    )


def choose_p_on(tmin_obs, p, threshold: float, target_recall: float = TARGET_RECALL) -> float:
    """Largest alarm probability cut that still reaches `target_recall` on the given (training) nights."""
    ev = np.asarray(tmin_obs, dtype=float) <= threshold
    p = np.asarray(p, dtype=float)
    if ev.sum() == 0:
        return 0.5
    for cut in np.round(np.arange(0.95, 0.04, -0.01), 2):
        if (p[ev] >= cut).mean() >= target_recall:
            return float(cut)
    return 0.05


def brier(p, event) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(event, dtype=float)) ** 2))


def reliability(p, event, edges=(0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0001)) -> list[dict]:
    p, event = np.asarray(p, dtype=float), np.asarray(event, dtype=bool)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if m.any():
            out.append(dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()), mean_p=float(p[m].mean()),
                            observed=float(event[m].mean())))
    return out


def count_toggles(states) -> int:
    s = np.asarray(list(states), dtype=bool)
    return int((s[1:] != s[:-1]).sum()) if len(s) > 1 else 0


# --- the experiment ------------------------------------------------------------------------------------------


@dataclass
class Experiment:
    train_last: int
    fixture: pd.DataFrame
    nights: dict[int, pd.DataFrame] = field(default_factory=dict)
    models: dict[int, FrostModel] = field(default_factory=dict)
    profile: CoolingProfile | None = None
    results: dict = field(default_factory=dict)
    test18: pd.DataFrame | None = None  # per-night predictions at 18:00 (held-out seasons)
    case: pd.DataFrame | None = None
    lead_table: pd.DataFrame | None = None


def split(nights: pd.DataFrame, train_last: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return nights[nights["season"] <= train_last].reset_index(drop=True), \
        nights[nights["season"] > train_last].reset_index(drop=True)


def run_experiment(fixture: pd.DataFrame, train_last: int = TRAIN_LAST_SEASON, issue_hours=ISSUE_HOURS,
                   progress=_say_) -> Experiment:
    exp = Experiment(train_last=train_last, fixture=fixture)
    res: dict = {"train_seasons": [int(fixture["time_utc"].dt.year.min()), train_last],
                 "test_seasons": [train_last + 1, int(fixture["time_utc"].dt.year.max())],
                 "stations": list(FORECAST_STATIONS)}
    preds_by_hour: dict[int, pd.DataFrame] = {}
    p_on: dict[float, float] = {}
    for hour in issue_hours:
        progress(f"issue {hour % 24:02d}:00: building nights")
        nights = build_nights(fixture, issue_hour=hour)
        exp.nights[hour] = nights
        train, test = split(nights, train_last)
        progress(f"issue {hour % 24:02d}:00: fitting on {len(train)} nights, testing on {len(test)}")
        model = FrostModel().fit(train)
        exp.models[hour] = model
        pred = test[["station", "night", "season", "tmin", "t_now", "kc", "wind_now", "prev_tmin", "doy"]].copy()
        pred["tmin_hat"] = model.predict(test)
        pred["sigma"] = model.sigma(test)
        pred["phys"] = model.baseline.predict(test)
        clim = Climatology().fit(train).predict(test)
        pred["clim"] = clim
        pred["pers"] = persistence(test, clim)
        for th in THRESHOLDS:
            pred[f"p{th:g}"] = frost_probability(pred["tmin_hat"], pred["sigma"], th)
        if hour == 18:
            lin = LinearModel().fit(train, model.baseline.predict(train))
            pred["ridge"] = lin.predict(test, pred["phys"].to_numpy())
            oof_hat = train["tmin"].to_numpy() - model.oof_res
            oof_sigma = model.sigma(train)
            for th in THRESHOLDS:
                p_on[th] = choose_p_on(train["tmin"], frost_probability(oof_hat, oof_sigma, th), th)
        preds_by_hour[hour] = pred
    p18 = preds_by_hour[18]
    exp.test18 = p18
    res["p_on"] = {f"{k:g}": v for k, v in p_on.items()}
    res["n_test_nights"] = int(len(p18))
    res["n_train_nights"] = int(len(split(exp.nights[18], train_last)[0]))

    # Tmin errors at 18:00
    methods = {"climatology": "clim", "persistence": "pers", "radiative_baseline": "phys", "ridge": "ridge",
               "baseline_plus_gbm": "tmin_hat"}
    res["tmin_18"] = {name: dict(mae=mae(p18[c], p18["tmin"]), rmse=rmse(p18[c], p18["tmin"]),
                                 bias=float((p18[c] - p18["tmin"]).mean())) for name, c in methods.items()}
    res["tmin_18_by_station"] = {
        st: dict(n=int(len(g)), model_mae=mae(g["tmin_hat"], g["tmin"]), persistence_mae=mae(g["pers"], g["tmin"]),
                 climatology_mae=mae(g["clim"], g["tmin"]))
        for st, g in p18.groupby("station")}
    frost_nights = p18["tmin"] <= 0
    res["tmin_18_frost_nights"] = dict(
        n=int(frost_nights.sum()), model_mae=mae(p18.loc[frost_nights, "tmin_hat"], p18.loc[frost_nights, "tmin"]),
        persistence_mae=mae(p18.loc[frost_nights, "pers"], p18.loc[frost_nights, "tmin"]),
        climatology_mae=mae(p18.loc[frost_nights, "clim"], p18.loc[frost_nights, "tmin"]))
    res["tmin_by_issue_hour"] = {
        f"{h % 24:02d}:00": dict(model_mae=mae(p["tmin_hat"], p["tmin"]), persistence_mae=mae(p["pers"], p["tmin"]),
                                 n=int(len(p))) for h, p in preds_by_hour.items()}

    # frost classification at 18:00 with the cut chosen on training seasons
    res["frost_18"] = {}
    for th in THRESHOLDS:
        key = f"{th:g}"
        alarm = p18[f"p{key}"] >= p_on[th]
        c = classification(p18["tmin"], alarm, th)
        c["p_on"] = p_on[th]
        c["brier"] = brier(p18[f"p{key}"], p18["tmin"] <= th)
        c["brier_climatology_rate"] = brier(np.full(len(p18), float((p18["tmin"] <= th).mean())), p18["tmin"] <= th)
        pers_alarm = p18["pers"] <= th
        c["persistence_alarm"] = classification(p18["tmin"], pers_alarm, th)
        base_alarm = p18["phys"] <= th
        c["baseline_alarm"] = classification(p18["tmin"], base_alarm, th)
        c["reliability"] = reliability(p18[f"p{key}"], p18["tmin"] <= th)
        res["frost_18"][key] = c

    # lead times (alarm at 18:00 -> first observed frost / protection trigger)
    res["lead_time"], lead_table = _lead_times(exp, p18, p_on[0.0])
    res["trigger_timing"] = _trigger_timing(exp, p18)

    # updates at 18/21/00 and hysteresis
    res["updates"] = _update_analysis(preds_by_hour, p_on[0.0])

    # April 2024 case study
    res["case_april_2024"], case_df = _case_study(exp, preds_by_hour, p_on[0.0])
    exp.results = res
    exp.case = case_df
    exp.lead_table = lead_table
    return exp


def _stations(exp: Experiment) -> dict[str, Station]:
    if not hasattr(exp, "_st"):
        exp._st = {st: Station(exp.fixture, st) for st in FORECAST_STATIONS}
    return exp._st


def _lead_times(exp: Experiment, p18: pd.DataFrame, p_on0: float) -> tuple[dict, pd.DataFrame]:
    st = _stations(exp)
    train, _ = split(exp.nights[18], exp.train_last)
    exp.profile = learn_profile(exp.fixture, train)
    rows = []
    for r in p18.itertuples():
        if r.tmin > 0:
            continue
        d = r.night.date()
        first = observed_crossing(st[r.station], d, 0.0)
        rows.append(dict(station=r.station, night=r.night, alarm=r.p0 >= p_on0,
                         lead_h=(first - 18.0) if first is not None else np.nan))
    tab = pd.DataFrame(rows)
    hit = tab[tab["alarm"] & tab["lead_h"].notna()]
    out = dict(n_frost_nights=int(len(tab)), n_with_alarm=int(tab["alarm"].sum()),
               median_lead_h=float(hit["lead_h"].median()) if len(hit) else float("nan"),
               p25_lead_h=float(hit["lead_h"].quantile(0.25)) if len(hit) else float("nan"),
               p75_lead_h=float(hit["lead_h"].quantile(0.75)) if len(hit) else float("nan"),
               share_lead_ge_3h=float((hit["lead_h"] >= 3).mean()) if len(hit) else float("nan"))
    return out, tab


def _trigger_timing(exp: Experiment, p18: pd.DataFrame) -> dict:
    """Predicted vs observed time at which the 2 m temperature reaches the protection trigger (0 C + margin)."""
    st = _stations(exp)
    trigger = 0.0 + PROTECTION_MARGIN_C
    errs, missed, spurious = [], 0, 0
    for r in p18.itertuples():
        obs = observed_crossing(st[r.station], r.night.date(), trigger)
        pred = exp.profile.time_to_reach(r.t_now, r.tmin_hat, trigger)
        if obs is not None and pred is not None:
            errs.append(pred - obs)
        elif obs is not None and r.tmin <= 0:
            missed += 1
        elif pred is not None and obs is None:
            spurious += 1
    e = np.asarray(errs)
    return dict(trigger_c=trigger, n_compared=int(len(e)), mae_h=float(np.abs(e).mean()) if len(e) else float("nan"),
                median_error_h=float(np.median(e)) if len(e) else float("nan"),
                share_within_1h=float((np.abs(e) <= 1).mean()) if len(e) else float("nan"),
                share_within_2h=float((np.abs(e) <= 2).mean()) if len(e) else float("nan"),
                forecast_reaches_but_obs_not=int(spurious), frost_night_no_trigger_time=int(missed))


def _update_analysis(preds: dict[int, pd.DataFrame], p_on0: float, p_off_ratio: float = 0.5) -> dict:
    """Alarm flapping across the 18:00 / 21:00 / 00:00 updates with a plain cut vs hysteresis."""
    keys = ["station", "night"]
    merged = None
    for h, p in preds.items():
        sub = p[keys + ["p0", "tmin"]].rename(columns={"p0": f"p{h}", "tmin": f"tmin{h}"})
        merged = sub if merged is None else merged.merge(sub, on=keys, how="inner")
    cols = [f"p{h}" for h in preds]
    plain, hyst = 0, 0
    final_alarm_frost = final_alarm_nofrost = 0
    for row in merged.itertuples(index=False):
        ps = [getattr(row, c) for c in cols]
        raw = [p >= p_on0 for p in ps]
        st = Hysteresis(p_on=p_on0, p_off=p_on0 * p_off_ratio, off_updates=2)
        latched = [st.update(p) for p in ps]
        plain += count_toggles(raw)
        hyst += count_toggles(latched)
        frost = row.tmin18 <= 0
        if any(latched):
            final_alarm_frost += int(frost)
            final_alarm_nofrost += int(not frost)
    n_frost = int((merged["tmin18"] <= 0).sum())
    return dict(n_nights=int(len(merged)), toggles_plain_cut=plain, toggles_hysteresis=hyst,
                p_off=p_on0 * p_off_ratio, off_updates=2,
                nights_alarmed_by_00_frost=final_alarm_frost, nights_alarmed_by_00_nofrost=final_alarm_nofrost,
                n_frost_nights=n_frost,
                hit_rate_any_update=final_alarm_frost / n_frost if n_frost else float("nan"))


def _case_study(exp: Experiment, preds: dict[int, pd.DataFrame], p_on0: float) -> tuple[dict, pd.DataFrame]:
    p18 = exp.test18
    m = (p18["night"] >= pd.Timestamp(CASE_START)) & (p18["night"] <= pd.Timestamp(CASE_END))
    case = p18[m].copy()
    case["alarm"] = case["p0"] >= p_on0
    case["frost"] = case["tmin"] <= 0
    out = dict(period=[CASE_START.isoformat(), CASE_END.isoformat()])
    per = {}
    for st, g in case.groupby("station"):
        fr = g[g["frost"]]
        per[st] = dict(
            frost_nights=int(len(fr)), alarmed=int(fr["alarm"].sum()), false_alarms=int((g["alarm"] & ~g["frost"]).sum()),
            model_mae=mae(g["tmin_hat"], g["tmin"]), persistence_mae=mae(g["pers"], g["tmin"]),
            coldest_obs=float(g["tmin"].min()), coldest_night=str(g.loc[g["tmin"].idxmin(), "night"].date()))
    out["by_station"] = per
    tot = case[case["frost"]]
    out["frost_nights_all_stations"] = int(len(tot))
    out["alarmed_all_stations"] = int(tot["alarm"].sum())
    out["false_alarms_all_stations"] = int((case["alarm"] & ~case["frost"]).sum())
    out["model_mae"] = mae(case["tmin_hat"], case["tmin"])
    out["persistence_mae"] = mae(case["pers"], case["tmin"])
    return out, case.reset_index(drop=True)


def example_alarm(exp: Experiment, station: str = "VIS", night: date | None = None) -> dict:
    """Alarm JSON for the coldest case-study night at `station` (18:00 update)."""
    case = exp.case[exp.case["station"] == station]
    if night is None:
        night = case.loc[case["tmin"].idxmin(), "night"].date()
    row = case[case["night"] == pd.Timestamp(night)].iloc[0]
    state = Hysteresis(p_on=exp.results["p_on"]["0"], p_off=exp.results["p_on"]["0"] * 0.5, off_updates=2)
    return build_alarm(station, night, datetime(night.year, night.month, night.day, 18, 0), row["tmin_hat"],
                       row["sigma"], row["t_now"], state, exp.profile)


def station_name(code: str) -> str:
    return STATIONS[code]["name"]
