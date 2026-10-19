import numpy as np
import pandas as pd
import pytest
import xarray as xr

xe = pytest.importorskip("xesmf")

from downscaling.regrid import build_regridder, regrid, weights_path  # noqa: E402

SOURCE_LAT = np.arange(44.0, 37.75, -0.25)
SOURCE_LON = np.arange(-1.5, 5.0, 0.25)
TIMES = pd.date_range("2003-01-01", periods=2, freq="D")


def source_dataset(**fields) -> xr.Dataset:
    """Regular ERA5-like dataset. Each field is a function f(lon, lat) -> 2D array."""
    lon2d, lat2d = np.meshgrid(SOURCE_LON, SOURCE_LAT)
    data_vars = {
        name: (("valid_time", "latitude", "longitude"), np.stack([f(lon2d, lat2d)] * len(TIMES)))
        for name, f in fields.items()
    }
    return xr.Dataset(
        data_vars,
        coords={"valid_time": TIMES, "latitude": SOURCE_LAT, "longitude": SOURCE_LON},
        attrs={"source": "test"},
    )


def target_grid(lon0: float = 0.0, lat0: float = 39.5, n: int = 6) -> xr.Dataset:
    """Small rotated (curvilinear) grid with 2D latitude/longitude, CERRA-like."""
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lon = lon0 + 0.3 * x + 0.05 * y
    lat = lat0 + 0.3 * y - 0.05 * x
    return xr.Dataset(coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)})


def linear(lon, lat):
    return 2.0 + lon + 0.5 * lat


def test_linear_field_is_reproduced():
    ds = source_dataset(t2m=linear)
    target = target_grid()
    out = regrid(ds, build_regridder(ds, target))

    expected = linear(target["longitude"], target["latitude"])
    for i in range(len(TIMES)):
        np.testing.assert_allclose(out["t2m"].isel(valid_time=i), expected, atol=0.05)


def test_target_outside_source_raises():
    ds = source_dataset(t2m=linear)
    target = target_grid(lat0=43.0)  # reaches north of the source grid (44 N)
    with pytest.raises(ValueError, match="no source data"):
        regrid(ds, build_regridder(ds, target))


def test_sst_land_mask_does_not_spread():
    # "Land" (NaN) west of 0.8 E, i.e. at source longitudes <= 0.75; sea from 1.0 E on.
    ds = source_dataset(t2m=linear, sst=lambda lon, lat: np.where(lon < 0.8, np.nan, 290.0))
    target = target_grid(lon0=-0.47)  # avoids target points exactly on source grid lines
    regridder = build_regridder(ds, target)
    out = regrid(ds, regridder)

    sst = out["sst"].isel(valid_time=0)
    lon = out["longitude"]
    coast = (lon > 0.75) & (lon < 1.0)  # between a land and a sea source column
    assert coast.any()

    # Plain bilinear regridding turns the coastal sea points into NaN
    plain = regridder(ds[["sst"]])["sst"].isel(valid_time=0)
    assert plain.isnull().where(coast, True).all()
    # regrid() keeps a value there, using the valid neighbours only
    assert sst.notnull().where(lon > 0.75, True).all()
    # and points with only land around them stay NaN.
    assert sst.isnull().where(lon < 0.75, True).all()
    # The NaN in sst do not trigger the coverage check, and t2m is complete.
    assert out["t2m"].notnull().all()


def test_static_field_without_time_dimension():
    ds = source_dataset(z_sfc=linear).isel(valid_time=0, drop=True)
    out = regrid(ds, build_regridder(ds, target_grid()))
    assert out["z_sfc"].dims == ("y", "x")


def test_weights_are_saved_and_reused(tmp_path):
    ds = source_dataset(t2m=linear)
    target = target_grid()
    weights = tmp_path / "weights.nc"

    first = regrid(ds, build_regridder(ds, target, weights_file=weights))
    assert weights.exists()
    second = regrid(ds, build_regridder(ds, target, weights_file=weights))  # loads the file
    xr.testing.assert_allclose(first, second)


def test_weights_path_depends_on_both_grids(tmp_path):
    ds = source_dataset(t2m=linear)
    a = weights_path(tmp_path, ds, target_grid())
    assert a == weights_path(tmp_path, ds, target_grid())  # same grids: same file
    assert a != weights_path(tmp_path, ds, target_grid(lon0=0.5))  # other target grid
    assert a != weights_path(tmp_path, ds, target_grid(), method="nearest_s2d")
