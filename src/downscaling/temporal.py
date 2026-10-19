"""Function that computes daily averages of Xarray datasets. Used both for ERA5 and CERRA.

And functions that check and resolve date ranges.
"""

import logging

import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

ONE_HOUR = pd.Timedelta(hours=1)
ONE_DAY = pd.Timedelta(days=1)

# -----------------------------------------
# Daily mean aggregator
# -----------------------------------------


def daily_mean(
    ds: xr.Dataset,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    step: pd.Timedelta = ONE_HOUR,
    time_dim: str = "valid_time",
) -> xr.Dataset:
    """Computes daily averages of ds for time range specified by start and stop
    and with steps specified by step. Expects [start, stop - step] time range
    to be contained in the ds.valid_time.
    """
    subset = ds.sel({time_dim: slice(start, stop - step)})
    daily = subset.resample({time_dim: "1D"}).mean(keep_attrs=True)
    return daily


# -----------------------------------------
# Date resolver
# -----------------------------------------


def compute_last_full_day(ds: xr.Dataset) -> pd.Timestamp:
    """Computes last day D in dataset ds such that 00:00 of D+1 is in ds."""
    return (pd.Timestamp(ds.valid_time.values[-1]) - ONE_DAY).floor("D")


def resolve_date_range(
    start: pd.Timestamp, end: pd.Timestamp, first: pd.Timestamp, last: pd.Timestamp
) -> pd.Timestamp:
    """Checks wether end date > last available full day in dataset. If so, changes
    end to last day.

    Raises ValueError if then start date > than end date (which is <= last) or if
    start date < first available date.
    """
    if end > last:
        log.warning(
            "End date (%s) is after the last available date (%s). Processing data up to "
            "last available full date.",
            end,
            last,
        )
        end = last
    if start > end:
        raise ValueError(
            f"Start date ({start:%Y-%m-%d}) must be earlier than (or the same as) "
            "end ({end:%Y-%m-%d})"
        )
    if start < first:
        raise ValueError(
            f"Start date ({start:%Y-%m-%d}) is before the first available date ({{first:%Y-%m-%d}})"
        )
    return end
