import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscaling.geo import lon_to_180, crop_regular_to_box, crop_curvilinear_to_box, box_indices

def test_lon_to_180():
    lon_values = [359.5, 180, 45]
    lon = xr.DataArray(
        lon_values,
        dims="longitude",
        coords={"longitude": lon_values}
    )
    np.testing.assert_allclose(lon_to_180(lon).values, [-0.5, -180, 45])


@pytest.fixture
def box():
    return {"lat_min": 23, "lat_max": 36, "lon_min": -5, "lon_max": 9}

@pytest.fixture
def rectilinear_ds():
    ds = xr.Dataset(
        data_vars=None,
        coords={"valid_time": [pd.Timestamp("2020-01-01")],
                "latitude": [40, 35, 30, 25, 20],
                "longitude": [0, 5, 10, 350, 355]}
    )
    return ds

@pytest.fixture
def curvilinear_ds():
    lat = np.flip(np.linspace(25,45,5))[:,None] + np.zeros((5,5))
    lon = np.array([351, 356, 1, 6, 11])[None,:] + np.arange(5)[:,None]
    ds = xr.Dataset(
        data_vars=None,
        coords={"time": [pd.Timestamp("2020-01-01")],
                "latitude": (("y","x"), lat),
                "longitude": (("y","x"), lon)}
    )
    return ds

def test_crop_regular_to_box(rectilinear_ds, box):
    cropped_1d = crop_regular_to_box(rectilinear_ds, box)
    np.testing.assert_allclose(cropped_1d.latitude.values, [35, 30, 25])
    np.testing.assert_allclose(cropped_1d.longitude.values, [-5, 0, 5])

def test_crop_curvilinear_to_box(curvilinear_ds, box):
    box_ind = box_indices(curvilinear_ds.latitude, curvilinear_ds.longitude, box)
    cropped_2d = crop_curvilinear_to_box(curvilinear_ds, box_ind)
    cropped_lat = np.flip(np.linspace(25,35,3))[:,None] + np.zeros((3,4))
    cropped_lon = np.array([-7, -2, 3, 8])[None,:] + np.arange(3)[:,None]
    np.testing.assert_allclose(cropped_2d.latitude.values, cropped_lat)
    np.testing.assert_allclose(cropped_2d.longitude.values, cropped_lon)

