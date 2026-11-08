"""Command line: demo, forecast, fetch-met, fetch-dem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="frost-night", description="Spring frost early warning for vineyards/orchards.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="train, evaluate on held-out seasons, write figures and results")
    d.add_argument("--out", type=Path, default=Path("."), help="repo root where docs/ and results/ are written")

    f = sub.add_parser("forecast", help="alarm JSON for one station and night (18:00 forecast)")
    f.add_argument("--station", required=True, help="SIO VIS AIG CGI MAG CHU")
    f.add_argument("--night", required=True, help="evening date, YYYY-MM-DD (March-May 2010-2026 in the fixture)")

    m = sub.add_parser("fetch-met", help="rebuild the MeteoSwiss fixture from the open-data portal")
    m.add_argument("--cache", type=Path, default=Path("cache"))
    e = sub.add_parser("fetch-dem", help="rebuild the 10 m DEM from swissALTI3D tiles (needs rasterio)")
    e.add_argument("--cache", type=Path, default=Path("cache/dem"))

    a = ap.parse_args(argv)
    if a.cmd == "demo":
        from .pipeline import run_demo

        res = run_demo(a.out)
        print(json.dumps(summary(res), indent=2))
    elif a.cmd == "forecast":
        from .data import FORECAST_STATIONS, STATIONS
        from .pipeline import forecast_night

        if a.station.upper() not in FORECAST_STATIONS or a.station.upper() not in STATIONS:
            print(f"unknown station {a.station}; choose from {', '.join(FORECAST_STATIONS)}", file=sys.stderr)
            return 2
        print(json.dumps(forecast_night(a.station, a.night), indent=2))
    elif a.cmd == "fetch-met":
        from .fetch import fetch_met

        df = fetch_met(a.cache)
        print(f"fixture rows: {len(df)}")
    elif a.cmd == "fetch-dem":
        from .fetch import fetch_dem

        print(fetch_dem(a.cache))
    return 0


def summary(res: dict) -> dict:
    t = res["tmin_18"]
    f0 = res["frost_18"]["0"]
    return {
        "test_seasons": res["test_seasons"],
        "tmin_mae_k": {k: round(v["mae"], 2) for k, v in t.items()},
        "frost_0c": {k: (round(f0[k], 3) if isinstance(f0[k], float) else f0[k])
                     for k in ("p_on", "hit_rate", "false_alarm_ratio", "false_positive_rate", "n_frost")},
        "median_lead_h": round(res["lead_time"]["median_lead_h"], 1),
        "april_2024": {k: res["case_april_2024"][k] for k in
                       ("frost_nights_all_stations", "alarmed_all_stations", "false_alarms_all_stations")},
    }


if __name__ == "__main__":
    sys.exit(main())
