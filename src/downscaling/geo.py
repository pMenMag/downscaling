"""Functions that crop Xarray data to specified region (both for 
rectilinear and curvilinear grids) and that convert longitude conventions.
"""

import numpy as np
import xarray as xr

def lon_to_180(lon: xr.DataArray) -> xr.DataArray:
    """Convert longitude coordinates from [0,360] to [-180,180]."""
    return ((lon + 180) % 360) - 180

def crop_regular_to_box(ds: xr.Dataset, box: dict[str, float]) -> xr.Dataset:
    """Crop data set for the specified bounding box (box)
    and change longitudes to [-180,180] range. Also sorts dataset by longitude.
    Latitude assumed counting from north to south.
    Recommended to use this function after selecting for a small time interval 
    to avoid using sortby() on a large dataset.
    """
    lon_box = (lon_to_180(ds.longitude) >= box["lon_min"]) & (
        lon_to_180(ds.longitude) <= box["lon_max"]
    )
    ds = ds.sel(latitude=slice(box["lat_max"], box["lat_min"]), longitude=lon_box)
    ds = ds.assign_coords(longitude=lon_to_180(ds.longitude)).sortby("longitude")
    return ds

def box_indices(lat: xr.DataArray, lon: xr.DataArray, box: dict[str, float]) -> dict[str, slice]:
    """Index slices of the smallest grid rectangle containing every point inside box.

    For curvilinear grids, where latitude and longitude are 2D arrays over the
    grid's own dimensions (e.g. y, x).
    """
    lon = lon_to_180(lon)
    inside = (
        (lat >= box["lat_min"]) & (lat <= box["lat_max"]) & 
        (lon >= box["lon_min"]) & (lon <= box["lon_max"])
    ).compute()
    if not inside.any():
        raise ValueError(f"There are no grid points inside {box}")
    y_dim, x_dim = lat.dims
    rows = np.flatnonzero(inside.any(x_dim))
    cols = np.flatnonzero(inside.any(y_dim))
    return {y_dim: slice(rows[0], rows[-1] + 1), x_dim : slice(cols[0], cols[-1] + 1)}

def crop_curvilinear_to_box(ds: xr.Dataset, box_ind: dict[str, slice]) -> xr.Dataset:
    """Crops a curvilinear grid to indices specified by box_ind, indexed by latitude and longitude.
    """
    ds = ds.isel(box_ind)
    return ds.assign_coords(longitude=lon_to_180(ds.longitude))

