"""Regridding between regular and curvilinear grids with xESMF."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

LAT = "latitude"
LON = "longitude"


def spatial_grid(ds: xr.Dataset) -> xr.Dataset:
    """Return only the horizontal grid of ds, with coordinates named as xESMF expects.

    Works for 1D coordinates (regular grids such as ERA5) and 2D coordinates
    (curvilinear grids such as CERRA's Lambert conformal grid).
    """
    return xr.Dataset(coords={"lat": ds[LAT].variable, "lon": ds[LON].variable})


def weights_path(
    directory: Path, source: xr.Dataset, target: xr.Dataset, method: str = "bilinear"
) -> Path:
    """Path of the weights file for regridding source to target with method.

    The name contains a fingerprint of both grids, so weights computed for one
    pair of grids are never reused for another (e.g. after changing the region).
    """
    digest = hashlib.sha256(method.encode())
    for ds in (source, target):
        for name in (LAT, LON):
            digest.update(np.ascontiguousarray(ds[name].values, dtype="float64").tobytes())
    return directory / f"{method}_{digest.hexdigest()[:12]}.nc"


def build_regridder(
    source: xr.Dataset,
    target: xr.Dataset,
    method: str = "bilinear",
    weights_file: Path | None = None,
) -> xe.Regridder:
    """Create a regridder from the grid of source to the grid of target.

    Computing the weights is the expensive part and depends only on the two
    grids, so they are saved to weights_file on first use and loaded from it
    afterwards. Target points not covered by the source grid are set to NaN
    (not 0), so that gaps can be detected.
    """
    grid_in, grid_out = spatial_grid(source), spatial_grid(target)
    if weights_file is not None and weights_file.exists():
        return xe.Regridder(
            grid_in, grid_out, method, weights=str(weights_file), unmapped_to_nan=True
        )
    regridder = xe.Regridder(grid_in, grid_out, method, unmapped_to_nan=True)
    if weights_file is not None:
        weights_file.parent.mkdir(parents=True, exist_ok=True)
        regridder.to_netcdf(str(weights_file))
    return regridder


def check_coverage(ds: xr.Dataset, regridder: xe.Regridder) -> None:
    """Raise ValueError if any target point lies outside the source grid.

    Regrids a field of ones on the source grid, so the check depends neither on
    the data (e.g. sst being NaN over land) nor on the name of the time dimension.
    """
    da = ds[next(iter(ds.data_vars))]
    field = da.isel({dim: 0 for dim in da.dims[:-2]}, drop=True)
    n_nan = int(regridder(xr.ones_like(field, dtype=float)).isnull().sum())
    if n_nan:
        raise ValueError(f"{n_nan} target points have no source data. Widen the source region.")


def regrid(
    ds: xr.Dataset,
    regridder: xe.Regridder,
    nan_vars: Sequence[str] = ("sst",),
    na_thres: float = 1.0,
) -> xr.Dataset:
    """Regrid every variable of ds to the regridder's target grid.

    Variables in nan_vars contain NaN by design (e.g. sst over land). They are
    regridded with skipna=True: each target value is computed from the valid
    neighbouring source points only, with their weights renormalised, so the NaN
    area does not grow. na_thres is the largest share of NaN weight still
    accepted (1.0: a value is computed whenever at least one neighbour is valid).
    All other variables are regridded normally.

    Non-spatial dimensions (e.g. valid_time) and all attributes are kept.
    The spatial coordinates of the output are named latitude/longitude.

    Raises
    ------
    ValueError
        If some target points lie outside the source grid.
    """
    check_coverage(ds, regridder)

    with_nan = [v for v in ds.data_vars if v in nan_vars]
    complete = [v for v in ds.data_vars if v not in nan_vars]

    parts = []
    if complete:
        parts.append(regridder(ds[complete], keep_attrs=True))
    if with_nan:
        parts.append(regridder(ds[with_nan], keep_attrs=True, skipna=True, na_thres=na_thres))

    out = xr.merge(parts, join="exact").rename({"lat": LAT, "lon": LON})
    out.attrs = dict(ds.attrs)
    return out
