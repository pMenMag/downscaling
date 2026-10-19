"""Aggregate hourly (in UTC) era5 single level and pressure level data for
specified variables and over a specified region to daily data and save as
zarr file.

Optionally, also stores the ERA5 static variables land-sea mask and
geopotential in another zarr file.

Data is read lazily from the era5 single level and era5 pressure level zarr
stores hosted on the destine.eu hub.

The output zarr files are saved in the data/era5 directory.

Command line inputs: start date and end date in the format YYYY-MM-DD,
--static flag to also store the static variables,
--overwrite flag to overwrite existing output file if it exists.

Example
-------
python scripts/data/era5_store.py 2000-01-01 2000-02-01 --static --overwrite
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from downscaling.era5 import daily_prev_hour_aggr, daily_wind_speed
from downscaling.geo import crop_regular_to_box
from downscaling.io import open_zarr, skipping_if_exists, write_to_zarr
from downscaling.temporal import ONE_DAY, compute_last_full_day, daily_mean, resolve_date_range

log = logging.getLogger(__name__)

# -----------------------------------------
# Configuration
# -----------------------------------------

ERA5_SL_URL = "https://data.earthdatahub.destine.eu/era5/era5-single-levels-atmosphere-v0.zarr"

ERA5_PL_URL = "https://data.earthdatahub.destine.eu/era5/era5-pressure-levels-v0.zarr"

# First available date in the era5 single level and pressure level zarr stores.
FIRST_DATE = pd.Timestamp("1940-01-01")

# Bounding box for the region of interest (in degrees). Longitude is in the range [-180,180]
# and latitude is in the range [-90,90].
BOX = {
    "lat_min": 38,
    "lat_max": 44,
    "lon_min": -1.5,
    "lon_max": 4.75,
}

# ERA5 single level variables whose value is taken at the time stamp (instantaneous variables)
INSTANT_SL_VARS = ["t2m", "d2m", "sst", "msl", "u10", "v10"]

# ERA5 single level variables whose value describes the hour ENDING at the time stamp.
# Variables are keys, values are the aggregation method to use when aggregating to daily values.
PREV_HOUR_VARS = {"mx2t": "max", "mn2t": "min", "tp": "sum", "ssrd": "sum"}

SL_VARIABLES = INSTANT_SL_VARS + list(PREV_HOUR_VARS)

# ERA5 pressure level variables
PL_VARIABLES = ["t", "z"]

# Selected pressure levels
PL_LEVELS = [850, 500]

# ERA5 single level static variables renamed so that the names do not conflict
# with pressure level variable names
STATIC_VARIABLES = {"lsm": "lsm", "z": "z_sfc"}

OUTPUT_CHUNKS = {"valid_time": 31, "isobaricInhPa": -1, "latitude": -1, "longitude": -1}
OUT_DIR = Path(__file__).resolve().parents[2] / "data/era5"

# -----------------------------------------
# Pipelines
# -----------------------------------------


def make_daily_store(
    sl_ds: xr.Dataset,
    pl_ds: xr.Dataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
    box: dict = BOX,
    overwrite: bool = False,
) -> None:
    """Makes daily aggregates of both single level and pressure level ERA5 hourly datasets
    for specified time range and region. Stores the result in a zarr file.

    Starts by lazily selecting a slightly larger time range so that crop_to_box() doesn't
    apply sortby('longitude') for the whole time range of the dataset. Then applies the aggregating
    functions daily_mean(), daily_wind_speed() and daily_prev_hour_aggr().

    Parameters
    ----------
    sl_ds: xarray dataset containing the era5 single level data.
    pl_ds: xarray dataset containing the era5 pressure level data.
    start: start date Timestamp (expected in the form YYYY-MM-DD).
    end: end date Timestamp (expected in the form YYYY-MM-DD).
    box: dictionary with keys "lat_min", "lat_max", "lon_min", "lon_max"
        specifying the bounding box for the region of interest.
    overwrite: if True, overwrite existing output file. If False,
            skip processing if output file already exists.
    """
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"era5_{start:%Y-%m-%d}_{end:%Y-%m-%d}.zarr"

    if skipping_if_exists(out_file, overwrite):
        return

    stop = end + ONE_DAY

    sub_sl_ds = sl_ds[SL_VARIABLES].sel(valid_time=slice(start, stop))
    sub_sl_ds = crop_regular_to_box(sub_sl_ds, box)

    sub_pl_ds = pl_ds[PL_VARIABLES].sel(isobaricInhPa=PL_LEVELS, valid_time=slice(start, stop))
    sub_pl_ds = crop_regular_to_box(sub_pl_ds, box)

    daily_ds = xr.merge(
        [
            daily_mean(sub_sl_ds[INSTANT_SL_VARS], start, stop),
            daily_wind_speed(sub_sl_ds, start, stop).rename("wind_speed"),
            daily_prev_hour_aggr(sub_sl_ds[list(PREV_HOUR_VARS)], start, stop, PREV_HOUR_VARS),
            daily_mean(sub_pl_ds, start, stop),
        ],
        join="exact",
    )
    daily_ds.attrs = {
        "source": f"{ERA5_SL_URL} ; {ERA5_PL_URL}",
        "processing": (
            "Daily UTC aggregates. Instantaneous variables (single and pressure levels) "
            "averaged over 00-23 UTC; previous-hour variables aggregated over the hours "
            "ending 01-24 UTC."
        ),
        "region": str(box),
    }

    write_to_zarr(daily_ds, out_file, OUTPUT_CHUNKS)


def store_static_data(ds: xr.Dataset, box: dict = BOX, overwrite: bool = False) -> None:
    """Store the era5 static data (land-sea mask and surface geopotential) as a zarr file
    in the data/era5 directory."""

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "era5_static.zarr"

    if skipping_if_exists(out_file, overwrite):
        return

    sub_ds = ds[list(STATIC_VARIABLES)].isel(valid_time=0, drop=True)
    sub_ds = crop_regular_to_box(sub_ds, box)
    out_ds = sub_ds.rename(STATIC_VARIABLES)
    out_ds.attrs = {"source": ERA5_SL_URL, "region": str(box)}

    write_to_zarr(out_ds, out_file, OUTPUT_CHUNKS)


# -----------------------------------------
# Command line interface
# -----------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate hourly era5 single level data to daily data and save as zarr file."
    )
    parser.add_argument("start", type=parse_date, help="Start date in the format YYYY-MM-DD.")
    parser.add_argument(
        "end", type=parse_date, help="End date (inclusive) in the format YYYY-MM-DD."
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Also store static variables (land-sea-mask and geopotential).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists.",
    )
    return parser


def parse_date(text: str) -> pd.Timestamp:
    """Parse a YYYY-MM-DD date (used as an argparse type)."""
    try:
        value = pd.to_datetime(text, format="%Y-%m-%d")
    except ValueError:
        value = pd.NaT
    if pd.isna(value):
        raise argparse.ArgumentTypeError(f"invalid date {text}, expected YYYY-MM-DD")
    return value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sl_ds = open_zarr(ERA5_SL_URL)
    pl_ds = open_zarr(ERA5_PL_URL)

    last_full_sl_day = compute_last_full_day(sl_ds)
    last_full_pl_day = compute_last_full_day(pl_ds)
    last_full_day = min(last_full_sl_day, last_full_pl_day)

    try:
        end = resolve_date_range(args.start, args.end, FIRST_DATE, last_full_day)
    except ValueError as e:
        parser.error(str(e))

    make_daily_store(sl_ds, pl_ds, args.start, end, box=BOX, overwrite=args.overwrite)

    if args.static:
        store_static_data(sl_ds, box=BOX, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
