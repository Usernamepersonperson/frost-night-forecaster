"""Figures: README figure (1600x900) and phone-sized post figure (960x700), dark palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .terrain import Dem, hillshade  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
SOURCE = "Source: MeteoSwiss (SwissMetNet, open data); swisstopo swissALTI3D"
RISK_CMAP = LinearSegmentedColormap.from_list("risk", ["#1E3A5F", ACCENT, AMBER, RED])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": TEXT, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.facecolor": BG,
    "figure.facecolor": BG, "savefig.facecolor": BG,
})


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)


def draw_map(ax, dem: Dem, risk: np.ndarray, station_rc: tuple[int, int], label: bool = True, fs: int = 11):
    hs = hillshade(dem.z, dem.cell)
    ext = dem.extent_km()
    ax.imshow(hs, cmap="gray", extent=ext, vmin=0.15, vmax=1.0, alpha=0.9)
    im = ax.imshow(risk, cmap=RISK_CMAP, extent=ext, vmin=0, vmax=1, alpha=0.62)
    ax.contour(np.linspace(ext[0], ext[1], dem.z.shape[1]), np.linspace(ext[3], ext[2], dem.z.shape[0]), dem.z,
               levels=np.arange(500, 1300, 100), colors=[MUTED], linewidths=0.5, alpha=0.6)
    r, c = station_rc
    ax.plot(ext[0] + (c + 0.5) * dem.cell / 1000, ext[3] - (r + 0.5) * dem.cell / 1000, marker="o", ms=7,
            mfc=ACCENT, mec=TEXT, mew=1.2, ls="")
    if label:
        ax.annotate("Visp station", (ext[0] + (c + 0.5) * dem.cell / 1000, ext[3] - (r + 0.5) * dem.cell / 1000),
                    xytext=(8, 6), textcoords="offset points", color=TEXT, fontsize=fs)
    ax.set_xlabel("LV95 east (km)", fontsize=fs)
    ax.set_ylabel("LV95 north (km)", fontsize=fs)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    ax.tick_params(labelsize=fs)
    ax.set_aspect("equal")
    return im


def _title(fig, title: str, subtitle: str, x: float = 0.03, fs=19):
    fig.text(x, 0.955, title, fontsize=fs * 0.75 + 0.0, fontweight="bold", color=TEXT, va="top", ha="left")
    fig.text(x, 0.915, subtitle, fontsize=fs * 0.55, color=MUTED, va="top", ha="left")


def timeseries_april(ax, case: pd.DataFrame, station: str, fs: int = 11, p_on: float | None = None):
    g = case[case["station"] == station].sort_values("night")
    x = g["night"]
    ax.axhline(0, color=RED, lw=1.2, ls="--", alpha=0.8)
    ax.fill_between(x, g["tmin_hat"] - 1.28 * g["sigma"], g["tmin_hat"] + 1.28 * g["sigma"], color=ACCENT, alpha=0.18,
                    lw=0, label="forecast 80% range")
    ax.plot(x, g["tmin_hat"], color=ACCENT, lw=2.4, marker="o", ms=4, label="forecast 18:00")
    ax.plot(x, g["tmin"], color=GREEN, lw=2.4, marker="s", ms=4, label="observed Tmin")
    if p_on is not None:
        al = g[g["p0"] >= p_on]
        ax.scatter(al["night"], np.full(len(al), g["tmin"].min() - 2.2), marker="^", color=AMBER, s=48, zorder=5,
                   label="alarm raised")
    ax.set_ylabel("Tmin 2 m (deg C)", fontsize=fs)
    ax.set_ylim(-6.5, 18)
    ax.tick_params(labelsize=fs)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3))
    _style(ax)


def post_figure(path: Path, dem: Dem, risk: np.ndarray, station_rc, case: pd.DataFrame, p_on: float, night_label: str,
                station: str = "VIS") -> None:
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    _title(fig, "Frost night forecast, Valais (Visp)",
           f"Open data: MeteoSwiss + swisstopo | held-out April 2024 | map = P(frost) {night_label}")
    axm = fig.add_axes([0.085, 0.24, 0.40, 0.58])
    im = draw_map(axm, dem, risk, station_rc, label=False, fs=11)
    axm.set_title("Frost risk map from the 18:00 forecast", fontsize=12.5, color=TEXT, loc="left")
    cax = fig.add_axes([0.085, 0.125, 0.40, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("P(Tmin <= 0 C)", fontsize=11, color=MUTED, labelpad=2)
    cb.ax.tick_params(labelsize=11, colors=MUTED)
    cb.outline.set_edgecolor(GRID)
    axt = fig.add_axes([0.585, 0.24, 0.39, 0.58])
    timeseries_april(axt, case, station, p_on=p_on)
    axt.set_title("Visp: forecast vs observed", fontsize=12.5, color=TEXT, loc="left")
    axt.legend(fontsize=10.5, loc="upper center", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, framealpha=0.9)
    fig.text(0.03, 0.025, "Source: MeteoSwiss (SwissMetNet); swisstopo swissALTI3D. Terrain offsets are assumptions.",
             fontsize=10.5, color=MUTED)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def main_figure(path: Path, dem: Dem, risk: np.ndarray, station_rc, exp_results: dict, test18: pd.DataFrame,
                case: pd.DataFrame, p_on: float, night_label: str, station: str = "VIS") -> None:
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    _title(fig, "frost-night-forecaster: evening Tmin forecast and cold-pool frost risk (Valais and other valleys)",
           f"{SOURCE} | train 2010-2020, held-out 2021-2026 | terrain offset unverified", fs=20)
    # (a) map
    axm = fig.add_axes([0.05, 0.10, 0.27, 0.70])
    im = draw_map(axm, dem, risk, station_rc, fs=11)
    axm.set_title(f"A  Frost risk map, night of {night_label}", fontsize=13, color=TEXT, loc="left")
    cax = fig.add_axes([0.05, 0.055, 0.27, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("P(Tmin <= 0 C)", fontsize=11, color=MUTED)
    cb.ax.tick_params(labelsize=10, colors=MUTED)
    # (b) scatter
    axs = fig.add_axes([0.385, 0.40, 0.26, 0.40])
    _style(axs)
    lim = (-10, 12)
    axs.plot(lim, lim, color=GREY, lw=1)
    axs.axhline(0, color=RED, lw=0.8, ls="--")
    axs.axvline(0, color=RED, lw=0.8, ls="--")
    axs.scatter(test18["tmin"], test18["tmin_hat"], s=5, alpha=0.35, color=ACCENT, lw=0)
    axs.set_xlim(lim)
    axs.set_ylim(lim)
    axs.set_xlabel("observed Tmin (deg C)", fontsize=11)
    axs.set_ylabel("forecast at 18:00 (deg C)", fontsize=11)
    m = exp_results["tmin_18"]
    axs.set_title(f"B  Held-out nights: MAE {m['baseline_plus_gbm']['mae']:.2f} K", fontsize=13, color=TEXT, loc="left")
    axs.tick_params(labelsize=10)
    # (c) bars
    axb = fig.add_axes([0.385, 0.08, 0.26, 0.20])
    _style(axb)
    names = ["climatology", "persistence", "radiative_baseline", "baseline_plus_gbm"]
    labels = ["climatology", "persistence", "radiative", "radiative+GBM"]
    vals = [m[k]["mae"] for k in names]
    bars = axb.barh(labels, vals, color=[GREY, GREY, VIOLET, GREEN])
    for b, v in zip(bars, vals, strict=True):
        axb.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=11, color=TEXT)
    axb.set_xlim(0, max(vals) * 1.2)
    axb.set_xlabel("Tmin MAE (K), held-out seasons", fontsize=11)
    axb.tick_params(labelsize=10.5)
    axb.invert_yaxis()
    # (d) timeseries
    axt = fig.add_axes([0.72, 0.40, 0.26, 0.40])
    timeseries_april(axt, case, station, fs=10.5, p_on=p_on)
    axt.set_title("D  April 2024, Visp", fontsize=13, color=TEXT, loc="left")
    axt.legend(fontsize=9.5, loc="upper center", facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    # (e) frost metrics text
    f0 = exp_results["frost_18"]["0"]
    lt = exp_results["lead_time"]
    axk = fig.add_axes([0.72, 0.08, 0.26, 0.20])
    axk.axis("off")
    lines = [
        ("frost nights hit (Tmin <= 0 C)", f"{f0['hit_rate'] * 100:.0f} %", GREEN),
        ("alarms that were false", f"{f0['false_alarm_ratio'] * 100:.0f} %", AMBER),
        ("median alarm lead time", f"{lt['median_lead_h']:.1f} h", ACCENT),
    ]
    for i, (lab, val, col) in enumerate(lines):
        y = 0.82 - i * 0.30
        axk.text(0.0, y, val, fontsize=22, fontweight="bold", color=col, va="center")
        axk.text(0.33, y, lab, fontsize=12, color=MUTED, va="center")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
