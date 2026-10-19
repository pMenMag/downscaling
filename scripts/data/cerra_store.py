"""Aggregate 3-hourly CERRA single level data for 2 m temperature and over a specified region
to daily data and save as zarr file. 
Optionally, also stores the CERRA static variables land-sea mask and orography in another zarr file.

Data is retrieved from Copernicus Climate Data Store using cdsapi.
Data is stored in steps of 3 years to optimize request time, file size and number of cds requests.
The store is made in two stages:
    crop: data is requested with cdsapi, cropped to the region and stored in zarr in data/cerra/interim
    daily: the stores in data/cerra/interim are agrregated to daily values and stored in data/cerra

Command line inputs: starting year and ending year in the format YYYY, 
--stage with possible choices "crop", "daily" and "all". Defaults to "all".
--static flag to also store the static variables,
--overwrite flag to overwrite existing output file if it exists.

If (ending year - starting year) is not a multiple of 3, then the remaining years after ending year
are downloaded to reach a multiple of three

Example
-------
python scripts/data/cerra_store.py 2000 2004 --stage daily --static --overwrite

stores two files in data/cerra/interim:
cerra_crop_2000_to_2002.zarr
cerra_crop_2003_to_2005.zarr

and 

stores two files in data/cerra:
cerra_daily_sl_2000_to_2005.zarr
cerra_static.zarr
"""

import argparse
import logging
from pathlib import Path
import re

import pandas as pd

from downscaling.io import open_grib, open_mult_zarr, cerra_sl_request, cerra_static_request, skipping_if_exists, write_to_zarr, remove_cerra_grib
from downscaling.geo import box_indices, crop_curvilinear_to_box
from downscaling.temporal import ONE_HOUR, daily_mean, resolve_date_range

log = logging.getLogger(__name__)

# -----------------------------------------
# Configuration
# -----------------------------------------

# First and last possible input years.
FIRST_YEAR = pd.Timestamp("1984")
LAST_YEAR = pd.Timestamp("2024")

# Bounding box for the region of interest (in degrees). Longitude is in the range [-180,180] and latitude is in the range [-90,90].
BOX = {"lat_min": 38.65, 
       "lat_max": 43.35, 
       "lon_min": -0.65, 
       "lon_max": 4.2}

INPUT_CHUNKS = {"time": 56, "y": -1, "x": -1}
OUTPUT_CHUNKS = {"time": 31, "y": -1, "x": -1}

OUT_DIR = Path(__file__).resolve().parents[2] / "data/cerra"

# -----------------------------------------
# Pipelines
# -----------------------------------------

def make_cropped_store(
        start_year: int,
        end_year: int, 
        box: dict[str,float] = BOX,
        overwrite: bool = False,
    ) -> None:
    """Download CERRA 2 m temperature in 3-year batches, crop it and store it as Zarr.

    For each batch starting at start_year and advancing in steps of three
    years up to end_year, the 3-hourly data are requested from the CDS as
    GRIB (unless the GRIB file is already present), cropped by grid index to
    the region in box, and written to 
    data/cerra/interim/cerra_cropped_<year>_to_<year+2>.zarr 
    The GRIB file is deleted once the Zarr store has been written.

    The grid indices are computed once, from the first batch processed, and
    reused for all batches (the CERRA grid is the same in every file).

    Parameters
    ----------
    start_year
        First year of the first batch.
    end_year
        Last year to include. The last batch may extend up to two years beyond it.
    box
        Region with keys lat_min, lat_max, lon_min, lon_max (degrees).
    overwrite
        If True, recompute batches whose Zarr store already exists, otherwise skip them.
    """
    out_dir = OUT_DIR / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)

    box_ind = {}

    for year in range(start_year, end_year + 1, 3):
        out_file = out_dir / f"cerra_cropped_{year}_to_{year+2}.zarr"

        if skipping_if_exists(out_file, overwrite):
            continue

        request_target = out_dir / f"cerra_sl_{year}_to_{year+2}.grib"
    
        if not request_target.exists():
            log.info("requesting %d-%d:", year, year+2)
            cerra_sl_request(request_target, year, year+2)

        ds = open_grib(request_target, INPUT_CHUNKS)

        if not box_ind:
            box_ind = box_indices(ds.latitude, ds.longitude, box)

        ds = crop_curvilinear_to_box(ds, box_ind)
        ds.attrs = {
                    "source": "CERRA CDS API",
                    "region": str(box),
                }

        write_to_zarr(ds, out_file, INPUT_CHUNKS)
        
        remove_cerra_grib(request_target)
                    

def make_daily_store(
        start_year: int, 
        end_year: int, 
        overwrite: bool = False
    ) -> None:
    """Aggregate the cropped 3-hourly CERRA stores to daily means and store them as Zarr.

    Opens the interim stores of all 3-year batches between start_year and end_year 
    (as written by make_cropped_store), computes the daily mean of the eight 3-hourly 
    analyses of each day (00-21 UTC), and writes the result to 
    data/cerra/cerra_daily_sl_<start_year>_to_<last_year>.zarr, 
    where last_year is the last year of the final batch.

    Parameters
    ----------
    start_year
        First year of the first batch.
    end_year
        Last year to include, rounded up to the end of its 3-year batch.
    overwrite
        If True, recompute the store if it already exists; otherwise skip it.

    Raises
    ------
    FileNotFoundError
        If the interim store of any required batch is missing.
    """
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    year_list = list(range(start_year, end_year+1, 3))
    last_year = year_list[-1] + 2

    out_file = OUT_DIR / f"cerra_daily_sl_{start_year}_to_{last_year}.zarr"

    if skipping_if_exists(out_file, overwrite):
        return

    paths = [OUT_DIR / "interim" / f"cerra_cropped_{year}_to_{year+2}.zarr" for year in year_list]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {[p.name for p in missing]}")
    
    ds = open_mult_zarr(paths)
    start = pd.Timestamp(str(start_year))
    stop = pd.Timestamp(str(last_year+1))
    daily_ds = daily_mean(ds, start, stop, step=3*ONE_HOUR, time_dim="time")
    daily_ds.attrs = {
            **ds.attrs,
            "processing": (
                f"Daily mean of 2 m temperature of CERRA from {start_year} to {last_year}."
            ),
        }

    write_to_zarr(daily_ds, out_file, OUTPUT_CHUNKS)


def store_static_data(
        box: dict[str, float],
        overwrite: bool = False
    ) -> None:
    """Download the CERRA static fields, crop them and store them as Zarr.

    Requests the land-sea mask and orography from the CDS as GRIB (unless the
    GRIB file is already present), crops them by grid index to the region in
    box, and writes them to 
    data/cerra/cerra_static.zarr 
    The GRIB file is deleted once the Zarr store has been written.

    Parameters
    ----------
    box
        Region with keys lat_min, lat_max, lon_min, lon_max (degrees).
    overwrite
        If True, recompute the store if it already exists. Otherwise skip it.
    """
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"cerra_static.zarr"

    if skipping_if_exists(out_file, overwrite):
        return

    request_target = out_dir / f"cerra_static.grib"

    if not request_target.exists():
        log.info("requesting static cerra variables.")
        cerra_static_request(request_target)

    ds = open_grib(request_target, INPUT_CHUNKS)
    box_ind = box_indices(ds.latitude, ds.longitude, box)
    ds = crop_curvilinear_to_box(ds, box_ind)
    ds.squeeze(drop=True)
    scalars = [name for name in ds.coords if ds[name].ndim == 0]   
    ds = ds.drop_vars(scalars)    
    ds.attrs = {"source": "CERRA CDS API", "region": str(box)}

    write_to_zarr(ds, out_file, OUTPUT_CHUNKS)
    
    remove_cerra_grib(request_target)


# -----------------------------------------
# Command line interface
# -----------------------------------------

def parse_year(text: str) -> int:
    """Parse a YYYY (used as an argparse type)."""
    if not re.fullmatch(r"\d{4}", text):
        raise argparse.ArgumentTypeError(f"invalid year '{text}', expected YYYY (e.g. 2005)")
    return int(text)

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (years, --stage, --static, --overwrite)."""
    parser = argparse.ArgumentParser(
        description="Request CERRA data in steps of three years, crop it to Catalonia region, "
                    "aggregate it to daily data and store it in Zarr format."
    )
    parser.add_argument(
        "start", type=parse_year, help="Starting year in format YYYY."
    )
    parser.add_argument(
        "end", type=parse_year, help="Ending year in format YYYY. Data is stored in steps of three years. "
                                    "So ending year is rounded up to its 3-year batch. Last possible input " 
                                    "for ending year is 2024, in which case, data is stored from all of 2024 " 
                                    "to last available day of 2026."
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Also store static variables (land-sea-mask and orography)."
    )
    parser.add_argument(
        "--stage",
        choices=["crop", "daily", "all"],
        default="all",
        help="Which stage to run (default: %(default)s)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists."
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    start = args.start
    end = args.end

    try:
        end = resolve_date_range(pd.Timestamp(str(start)), pd.Timestamp(str(end)), FIRST_YEAR, LAST_YEAR).year
    except ValueError as e:
        parser.error(str(e))

    if args.stage in ("crop", "all"):
        make_cropped_store(start, end, box=BOX, overwrite=args.overwrite)
    if args.stage in ("daily", "all"):
        make_daily_store(start, end, overwrite=args.overwrite)

    if args.static:
        store_static_data(BOX, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

