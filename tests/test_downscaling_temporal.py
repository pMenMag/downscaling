import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscaling.temporal import daily_mean, compute_last_full_day, resolve_date_range

rng = np.random.default_rng(0)

@pytest.fixture
def era5_ds():
    times = xr.date_range("2020-01-01", "2020-01-03", freq="h")
    values = rng.random((49,1,1)) 
    values[:24,:,:] = values[:24,:,:] - values[:24,:,:].mean() + 5
    values[24:48,:,:] = values[24:48,:,:] - values[24:48,:,:].mean() + 10
    ds = xr.Dataset(
        data_vars={"t2m": (("valid_time", "latitude", "longitude"), values)},
        coords={
            "valid_time": times,
            "latitude": [0.0],
            "longitude": [0.0]
        }
    )
    return ds

@pytest.fixture
def cerra_ds():
    times = xr.date_range("2020-01-01", "2020-01-03", freq="3h")
    values = rng.random((17,1,1)) 
    values[:8,:,:] = values[:8,:,:] - values[:8,:,:].mean() + 5
    values[8:16,:,:] = values[8:16,:,:] - values[8:16,:,:].mean() + 10
    ds = xr.Dataset(
        data_vars={"t2m": (("time", "latitude", "longitude"), values)},
        coords={
            "time": times,
            "latitude": [0.0],
            "longitude": [0.0]
        }
    )
    return ds

def test_era5_daily_mean(era5_ds):
    start = pd.Timestamp("2020-01-01")
    stop = pd.Timestamp("2020-01-03")
    result = daily_mean(era5_ds, start, stop)
    np.testing.assert_allclose(result["t2m"].values, [[[5]], [[10]]])

def test_cerra_daily_mean(cerra_ds):
    start = pd.Timestamp("2020-01-01")
    stop = pd.Timestamp("2020-01-03")
    step = pd.Timedelta(hours=3)
    result = daily_mean(cerra_ds, start, stop, step=step, time_dim="time")
    np.testing.assert_allclose(result["t2m"].values, [[[5]], [[10]]])

def test_compute_last_full_day(era5_ds):
    assert compute_last_full_day(era5_ds) == pd.Timestamp("2020-01-02")

def test_resolve_data_range():
    first = pd.Timestamp("2020-01-01")
    last = pd.Timestamp("2020-01-03")
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-01-02")
    assert resolve_date_range(start, end, first, last) == end
    end = pd.Timestamp("2020-01-05")
    assert resolve_date_range(start, end, first, last) == last
    start = pd.Timestamp("2020-01-04")
    with pytest.raises(ValueError):
        resolve_date_range(start, end, first, last)
    start = pd.Timestamp("2019-01-01")
    with pytest.raises(ValueError):
        resolve_date_range(start, end, first, last)