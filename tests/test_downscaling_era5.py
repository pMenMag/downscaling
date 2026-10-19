import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscaling.era5 import daily_prev_hour_aggr, daily_wind_speed


@pytest.fixture
def era5_ds():
    times = xr.date_range("2020-01-01", "2020-01-03", freq="h")
    values = np.zeros((49, 1, 1))
    values[0, :, :], values[24, :, :], values[48, :, :] = (-8, -5, 100)
    v_values = values.copy()
    values[0, :, :], values[24, :, :] = (15, 12)
    values[1:8, :, :], values[25:36, :, :] = (np.ones((7, 1, 1)), np.ones((11, 1, 1)))
    ds = xr.Dataset(
        data_vars={
            "u10": (("valid_time", "latitude", "longitude"), values),
            "v10": (("valid_time", "latitude", "longitude"), v_values),
            "mx2t": (("valid_time", "latitude", "longitude"), values),
            "tp": (("valid_time", "latitude", "longitude"), values),
        },
        coords={"valid_time": times, "latitude": [0.0], "longitude": [0.0]},
    )
    return ds


def test_daily_prev_hour_aggr(era5_ds):
    start = pd.Timestamp("2020-01-01")
    stop = pd.Timestamp("2020-01-03")
    result = daily_prev_hour_aggr(era5_ds, start, stop, {"mx2t": "max", "tp": "sum"})
    np.testing.assert_allclose(result["mx2t"].values, [[[12]], [[100]]])
    np.testing.assert_allclose(result["tp"].values, [[[19]], [[111]]])


def test_daily_wind_speed(era5_ds):
    start = pd.Timestamp("2020-01-01")
    stop = pd.Timestamp("2020-01-03")
    result = daily_wind_speed(era5_ds, start, stop)
    np.testing.assert_allclose(result.values, [[[1]], [[1]]])
