import numpy as np
import pytest

from frost_night_forecaster.terrain import (
    Dem,
    box_mean,
    cold_pool_index,
    downscale_tmin,
    fill_depressions,
    fit_gradient,
    flow_accumulation,
    frost_risk_map,
    hillshade,
    load_dem,
    radiative_stability,
    slope_deg,
    tpi,
)


def bowl(n=61, depth=30.0):
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n // 2, y - n // 2)
    return 500.0 + depth * np.clip(r / (n / 2), 0, 1) ** 2


def plane(n=40, slope=0.1, cell=10.0):
    y, _ = np.mgrid[0:n, 0:n]
    return 1000.0 - slope * cell * y  # descends toward the south (row n-1)


def test_box_mean_constant_and_known_window():
    assert np.allclose(box_mean(np.full((5, 5), 3.0), 2), 3.0)
    a = np.arange(25, dtype=float).reshape(5, 5)
    assert box_mean(a, 1)[2, 2] == pytest.approx(a[1:4, 1:4].mean())
    assert box_mean(a, 1)[0, 0] == pytest.approx(a[0:2, 0:2].mean())  # clipped window at the corner


def test_box_mean_rejects_negative_radius():
    with pytest.raises(ValueError):
        box_mean(np.zeros((3, 3)), -1)


def test_tpi_negative_in_hollow_positive_on_ridge():
    z = bowl()
    t = tpi(z, 10.0, 150)
    assert t[30, 30] < 0
    assert t[0, 30] > t[30, 30]
    zr = 2 * 500.0 - z
    assert tpi(zr, 10.0, 150)[30, 30] > 0


def test_slope_of_plane_matches_gradient():
    s = slope_deg(plane(), 10.0)
    assert s[20, 20] == pytest.approx(np.degrees(np.arctan(0.1)), abs=1e-6)


def test_hillshade_range_and_flat_value():
    h = hillshade(np.full((10, 10), 100.0), 10.0, altitude=45)
    assert np.allclose(h, np.sin(np.radians(45)))
    assert hillshade(bowl(), 10.0).max() <= 1.0


def test_fill_depressions_raises_pit_to_spill_level():
    z = np.full((7, 7), 10.0)
    z[3, 3] = 2.0
    f = fill_depressions(z)
    assert f[3, 3] >= 10.0 and f[3, 3] < 10.1
    assert np.allclose(f[0], z[0])  # border untouched


def test_fill_leaves_draining_terrain_unchanged():
    z = plane()
    assert np.allclose(fill_depressions(z), z)


def test_flow_accumulation_on_plane_collects_column():
    z = plane(n=20)
    acc = flow_accumulation(fill_depressions(z), 10.0)
    assert acc[-1].max() >= 15  # bottom row collects the water of its column
    assert acc[0].max() == 1.0


def test_flow_accumulation_conserves_cells():
    z = bowl(31)
    acc = flow_accumulation(fill_depressions(z), 10.0)
    assert acc.min() >= 1 and acc.max() <= z.size


def test_cold_pool_index_higher_in_hollow_than_on_rim():
    z = bowl(81, 40.0)
    idx = cold_pool_index(Dem(z=z, east0=0, north_top=810, cell=10.0))
    p = idx["pool"]
    assert p.min() >= 0 and p.max() <= 1
    assert p[40, 40] > p[10, 40]


def test_downscale_reference_cell_equals_forecast():
    z = bowl(41)
    dem = Dem(z=z, east0=0, north_top=410, cell=10.0)
    idx = cold_pool_index(dem)
    t = downscale_tmin(dem, idx, -2.0, (20, 20), 1.0, -0.0056, -0.0034, 1.5)
    assert t[20, 20] == pytest.approx(-2.0)


def test_downscale_pool_is_colder_when_stable_and_no_effect_when_mixed():
    z = bowl(41)
    dem = Dem(z=z, east0=0, north_top=410, cell=10.0)
    idx = cold_pool_index(dem)
    ref = (2, 2)
    stable = downscale_tmin(dem, idx, 0.0, ref, 1.0, 0.0, 0.0, 2.0)
    mixed = downscale_tmin(dem, idx, 0.0, ref, 0.0, 0.0, 0.0, 2.0)
    assert np.allclose(mixed, 0.0)
    assert stable[20, 20] < 0 or idx["pool"][20, 20] <= idx["pool"][2, 2]


def test_downscale_applies_elevation_gradient():
    z = np.array([[500.0, 600.0], [700.0, 800.0]])
    dem = Dem(z=z, east0=0, north_top=20, cell=10.0)
    idx = {"pool": np.zeros((2, 2))}
    t = downscale_tmin(dem, idx, 4.0, (0, 0), 0.0, -0.006, -0.003, 1.5)
    assert t[1, 1] == pytest.approx(4.0 - 0.006 * 300)


def test_fit_gradient_recovers_known_values():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 500)
    dz = np.full(500, 940.0)
    g = -0.0056 + s * 0.0022
    gm, gs = fit_gradient(dz, g * dz, s)
    assert gm == pytest.approx(-0.0056, abs=1e-9) and gs == pytest.approx(-0.0034, abs=1e-9)


@pytest.mark.parametrize(("kc", "wind", "lo", "hi"), [(1.0, 0.0, 0.99, 1.0), (0.1, 0.0, 0.0, 0.0), (1.0, 10.0, 0.0, 0.3)])
def test_radiative_stability_bounds(kc, wind, lo, hi):
    assert lo <= radiative_stability(kc, wind) <= hi


def test_frost_risk_map_monotone():
    t = np.array([[-3.0, 0.0, 3.0]])
    p = frost_risk_map(t, 2.0, 0.0)
    assert p[0, 0] > p[0, 1] > p[0, 2] and p[0, 1] == pytest.approx(0.5)


def test_dem_rowcol_and_bounds():
    dem = Dem(z=np.zeros((10, 10)), east0=1000.0, north_top=2000.0, cell=10.0)
    assert dem.rowcol(1015.0, 1985.0) == (1, 1)
    with pytest.raises(ValueError):
        dem.rowcol(0.0, 0.0)


def test_real_dem_covers_visp_station():
    dem = load_dem()
    assert dem.shape == (500, 400) and dem.cell == 10.0
    r, c = dem.rowcol(2631151.0, 1128024.0)
    assert 625 < dem.z[r, c] < 655  # station height 639 m
    assert not np.isnan(dem.z).any()
    assert dem.z.max() > 1200
