"""Reading, downloading and writing of the project's datasets.
 
Reading: remote and local Zarr stores and GRIB files. 
Downloading: CERRA requests to the Copernicus Climate Data Store (CDS). 
Writing: Zarr stores and clean-up of downloaded GRIB files.
 
Downloads and Zarr stores are written to a temporary name first and renamed
when complete, so an interrupted run never leaves a file that looks finished.
"""

import logging
import shutil
from pathlib import Path

import cdsapi
import xarray as xr
from dask.diagnostics.progress import ProgressBar

log = logging.getLogger(__name__)


# -----------------------------------------
# Reading functions
# -----------------------------------------

def open_zarr(url: str) -> xr.Dataset:
    """Open the zarr store hosted on the specified URL."""
    ds = xr.open_dataset(
        url,
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )
    return ds

def open_mult_zarr(paths: list[Path]) -> xr.Dataset:
    """Open multiple zarr stores in specified list of Paths."""
    ds = xr.open_mfdataset(
        paths, 
        engine="zarr",
        data_vars="minimal", 
        coords="minimal", 
        compat="override", 
        join="exact")
    return ds

def open_grib(path: Path, chunks: dict[str,int]) -> xr.Dataset:
    """Open a GRIB file lazily with cfgrib with specified chunks."""
    ds = xr.open_dataset(
        path,
        chunks=chunks,
        engine="cfgrib")
    return ds.unify_chunks()

def cerra_sl_request(target: Path, start: int, stop: int) -> None:
    """Download 3-hourly CERRA 2 m temperature analyses as GRIB.
 
    The request covers every day of the years start to stop (both included),
    at 00, 03, ..., 21 UTC, on the full CERRA domain. The file is downloaded
    to <target>.tmp and renamed to target only when the download is
    complete.
    """
    client = cdsapi.Client()
    tmp_target = target.with_name(target.name + ".tmp")
    dataset = "reanalysis-cerra-single-levels"
    request = {
            "variable": ["2m_temperature"],
            "level_type": "surface_or_atmosphere",
            "data_type": ["reanalysis"],
            "product_type": "analysis",
            "year": [str(year) for year in range(start, stop + 1)],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "03:00", "06:00",
                "09:00", "12:00", "15:00",
                "18:00", "21:00"
            ],
            "data_format": "grib"
        }
    client.retrieve(dataset, request, str(tmp_target))
    tmp_target.replace(target)

def cerra_static_request(target: Path) -> None:
    """Download the CERRA land-sea mask and orography as GRIB.
 
    These fields do not change in time, so one time step (2025-01-01 00 UTC)
    is enough. The file is downloaded to <target>.tmp and renamed to
    target only when the download is complete.
    """
    client = cdsapi.Client()
    tmp_target = target.with_name(target.name + ".tmp")
    dataset = "reanalysis-cerra-single-levels"
    request = {
        "variable": [
            "land_sea_mask",
            "orography"
        ],
        "level_type": "surface_or_atmosphere",
        "data_type": ["reanalysis"],
        "product_type": "analysis",
        "year": ["2025"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "grib"
    }
    client.retrieve(dataset, request, str(tmp_target))
    tmp_target.replace(target)

# -----------------------------------------
# Writting functions
# -----------------------------------------

def remove_cerra_grib(tmp_file: Path) -> None:
    """Delete a GRIB file and the .idx index files cfgrib created for it."""
    tmp_file.unlink()
    folder = tmp_file.parent
    for idx in folder.glob(tmp_file.name + "*.idx"):
        idx.unlink()

def skipping_if_exists(out_file: Path, overwrite: bool) -> bool:
    """Returns True (and logs it) if and only if the output file exists and must not be overwritten."""
    if out_file.exists() and not overwrite:
        log.info("Output file %s already exists. Skipping.", out_file)
        return True
    return False

def write_to_zarr(ds: xr.Dataset, out_file: Path, out_chunks : dict[str, int]) -> None:
    """Write ds to zarr file on out_file location. 
    First writes to a temporary file and then renames it, in case download is interrupted.
    """
    chunks = {dim: size for dim, size in out_chunks.items() if dim in ds.dims}
    ds = ds.drop_encoding().astype("float32").chunk(chunks)
    tmp_file = out_file.with_name(out_file.name + ".tmp")
    if tmp_file.exists():
        shutil.rmtree(tmp_file)
    with ProgressBar():
        ds.to_zarr(tmp_file, mode="w")
    if out_file.exists():
        shutil.rmtree(out_file)
    tmp_file.rename(out_file)
    log.info("Wrote %s", out_file)


