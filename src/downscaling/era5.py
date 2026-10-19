"""Aggregation functions specific to ERA5 data."""

import pandas as pd
import xarray as xr

from downscaling.temporal import ONE_HOUR

# -----------------------------------------
# Daily aggregation functions
# -----------------------------------------

def daily_prev_hour_aggr(
        ds: xr.Dataset, 
        start: pd.Timestamp, 
        stop: pd.Timestamp, 
        prev_hour_vars : dict[str, str]
) -> xr.Dataset:
    """Computes daily aggregates of previous-hour variables over the hours 01:00 of start and 00:00 of stop.
    Expects [start + ONE_HOUR, stop] time range to be contained in the ds.valid_time.
    """
    subset = ds.sel(
        valid_time=slice(start + ONE_HOUR, stop)
    )
    subset = subset.assign_coords(valid_time = subset.valid_time - ONE_HOUR)
    daily = xr.Dataset()
    for var, agg_method in prev_hour_vars.items():
            if agg_method == "sum":
                daily[var] = (
                    subset[var]
                    .resample(valid_time="1D")
                    .sum(keep_attrs=True)
                )
            elif agg_method == "max":
                daily[var] = (
                    subset[var]
                    .resample(valid_time="1D")
                    .max(keep_attrs=True)
                )
            elif agg_method == "min":
                daily[var] = (
                    subset[var]
                    .resample(valid_time="1D")
                    .min(keep_attrs=True)
                )
            else:
                raise ValueError(f"Aggregation method {agg_method} not supported.")
    return daily

def daily_wind_speed(
        ds: xr.Dataset, start: pd.Timestamp, stop: pd.Timestamp
) -> xr.DataArray:
    """Computes daily mean of 10m wind speed.
    Averaging u10 and v10 cancels wind contributions in opposing directions 
    along the day so wind_speed keeps that information.
    """
    subset = ds[["u10","v10"]].sel(valid_time=slice(start, stop - ONE_HOUR))
    wind_speed = (subset["u10"]**2 + subset["v10"]**2)**0.5
    daily_wind_speed = wind_speed.resample(valid_time="1D").mean()
    daily_wind_speed.attrs = {
        "long_name": "Daily mean 10 m wind speed",
        "units": "m s**-1",
    }
    return daily_wind_speed.rename("wind_speed")

