"""Tmin forecast models: climatology, persistence, radiative-cooling baseline, and baseline + gradient boosting."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

THRESHOLDS = (0.0, -1.0, -2.0)
ML_FEATURES = [
    "t_now", "td_now", "dep_now", "rh_now", "wind_now", "t5_gap", "d_t3", "d_td6", "t_max_day", "kc", "sunfrac",
    "prev_tmin", "doy", "elev", "d_t_since18", "phys",
]
CLEAR_KC = 0.6  # clearness index above which the afternoon counts as clear (regime split for the spread)


def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def frost_probability(tmin_hat, sigma, threshold: float):
    """P(Tmin <= threshold) for a Gaussian forecast error."""
    return norm_cdf((threshold - np.asarray(tmin_hat, dtype=float)) / np.asarray(sigma, dtype=float))


# --- reference forecasts ---------------------------------------------------------------------------------


@dataclass
class Climatology:
    """Mean Tmin per station and day of year (+-7 days window) from the training seasons."""

    table: dict = field(default_factory=dict)
    window: int = 7

    def fit(self, train: pd.DataFrame) -> Climatology:
        for st, g in train.groupby("station"):
            doy = g["doy"].to_numpy()
            tm = g["tmin"].to_numpy()
            tab = {}
            for d in range(55, 160):
                sel = np.abs(doy - d) <= self.window
                if sel.any():
                    tab[d] = float(tm[sel].mean())
            self.table[st] = tab
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        for i, (st, d) in enumerate(zip(df["station"], df["doy"], strict=True)):
            tab = self.table.get(st, {})
            if tab:
                key = min(tab, key=lambda k: abs(k - int(d)))
                out[i] = tab[key]
        return out


def persistence(df: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    """Last night's observed Tmin (climatology if missing)."""
    p = df["prev_tmin"].to_numpy(dtype=float)
    return np.where(np.isnan(p), fallback, p)


# --- radiative-cooling baseline ----------------------------------------------------------------------------


def physics_design(df: pd.DataFrame) -> np.ndarray:
    """Terms of the night-cooling drop T_now - Tmin.

    dep   : dew-point depression (dry air radiates more freely; cooling also slows near the dew point)
    clear : clearness of the afternoon sky (cloud cover at sunset persists into the night)
    calm  : turbulent mixing suppresses the surface inversion, scaled as 1/(1+u/u0) with u0 = 2 m/s
    """
    dep = df["dep_now"].to_numpy(dtype=float)
    clear = np.clip(df["kc"].to_numpy(dtype=float), 0, 1.2)
    calm = 1.0 / (1.0 + df["wind_now"].to_numpy(dtype=float) / 2.0)
    return np.column_stack([np.ones(len(df)), dep, clear, clear * calm, dep * clear * calm, calm])


@dataclass
class PhysicsBaseline:
    """Tmin = T_now - drop, drop fitted as a linear combination of physical terms (6 parameters)."""

    coef: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> PhysicsBaseline:
        x = physics_design(train)
        y = (train["t_now"] - train["tmin"]).to_numpy(dtype=float)
        self.coef = np.linalg.lstsq(x, y, rcond=None)[0]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("baseline not fitted")
        drop = np.clip(physics_design(df) @ self.coef, 0.0, None)
        return df["t_now"].to_numpy(dtype=float) - drop


# --- full model ---------------------------------------------------------------------------------------------


def _ml_matrix(df: pd.DataFrame, phys: np.ndarray) -> np.ndarray:
    return df.assign(phys=phys)[ML_FEATURES].to_numpy(dtype=float)


@dataclass
class FrostModel:
    """Physics baseline + HistGradientBoosting on the baseline residual; regime-dependent Gaussian spread."""

    seed: int = 0
    max_iter: int = 120
    baseline: PhysicsBaseline = field(default_factory=PhysicsBaseline)
    gbm: HistGradientBoostingRegressor | None = None
    sigma_clear: float = 2.0
    sigma_other: float = 2.0
    oof_res: np.ndarray | None = None

    def _new_gbm(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=0.05, max_depth=4, min_samples_leaf=30, l2_regularization=1.0,
            random_state=self.seed,
        )

    def fit(self, train: pd.DataFrame, oof_sigma: bool = True) -> FrostModel:
        self.baseline = PhysicsBaseline().fit(train)
        phys = self.baseline.predict(train)
        self.gbm = self._new_gbm().fit(_ml_matrix(train, phys), train["tmin"].to_numpy() - phys)
        if oof_sigma:
            res = self.oof_residuals(train)
            self.oof_res = res
            clear = train["kc"].to_numpy() >= CLEAR_KC
            self.sigma_clear = float(np.std(res[clear])) if clear.sum() > 20 else float(np.std(res))
            self.sigma_other = float(np.std(res[~clear])) if (~clear).sum() > 20 else float(np.std(res))
        return self

    def oof_residuals(self, train: pd.DataFrame, n_folds: int = 4) -> np.ndarray:
        """Out-of-fold residuals with whole seasons held out (honest spread estimate from training seasons only)."""
        res = np.full(len(train), np.nan)
        seasons = sorted(train["season"].unique())
        fold_of = {s: i % n_folds for i, s in enumerate(seasons)}
        fold = train["season"].map(fold_of).to_numpy()
        for k in sorted(set(fold_of.values())):
            m = fold == k
            sub = FrostModel(seed=self.seed, max_iter=self.max_iter).fit(train[~m], oof_sigma=False)
            res[m] = train.loc[m, "tmin"].to_numpy() - sub.predict(train[m])
        return res

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.gbm is None:
            raise RuntimeError("model not fitted")
        phys = self.baseline.predict(df)
        return phys + self.gbm.predict(_ml_matrix(df, phys))

    def sigma(self, df: pd.DataFrame) -> np.ndarray:
        clear = df["kc"].to_numpy(dtype=float) >= CLEAR_KC
        return np.where(clear, self.sigma_clear, self.sigma_other)

    def p_frost(self, df: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
        return frost_probability(self.predict(df), self.sigma(df), threshold)


@dataclass
class LinearModel:
    """Ridge regression on the same inputs (a simpler ML reference)."""

    alpha: float = 3.0
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    ridge: Ridge | None = None

    def fit(self, train: pd.DataFrame, phys: np.ndarray) -> LinearModel:
        x = np.nan_to_num(_ml_matrix(train, phys))
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        self.ridge = Ridge(alpha=self.alpha).fit((x - self.mu) / self.sd, train["tmin"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame, phys: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(_ml_matrix(df, phys))
        return self.ridge.predict((x - self.mu) / self.sd)
