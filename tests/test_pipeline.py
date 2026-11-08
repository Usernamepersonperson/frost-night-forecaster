from types import SimpleNamespace

import numpy as np
import pandas as pd

from frost_night_forecaster.pipeline import POOL_AMPLITUDE_C, build_case_map


def test_build_case_map_picks_most_radiative_frost_night():
    case = pd.DataFrame({
        "station": ["VIS", "VIS", "VIS"], "night": pd.to_datetime(["2024-04-18", "2024-04-22", "2024-04-25"]),
        "tmin": [-2.7, -0.2, 3.0], "tmin_hat": [-1.6, -0.3, 2.0], "sigma": [2.0, 2.0, 2.0],
        "kc": [0.8, 0.4, 1.0], "wind_now": [7.0, 2.0, 0.5],
    })
    dem, rc, risk, info = build_case_map(SimpleNamespace(case=case), {"g_mixed_k_per_m": -0.0058,
                                                                       "g_stable_k_per_m": -0.0021})
    assert info["night"] == "2024-04-18"  # the warm 25 April night is excluded, 18 April is clearer than 22 April
    assert risk.shape == dem.shape and np.nanmin(risk) >= 0 and np.nanmax(risk) <= 1
    assert abs(risk[rc] - info["p_frost_at_station"]) < 1e-12
    assert info["pool_amplitude_c"] == POOL_AMPLITUDE_C
    assert info["tmap_p05_c"] < info["tmap_p95_c"]
