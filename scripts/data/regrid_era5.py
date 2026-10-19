"""Regrid the ERA5 stores onto the CERRA grid.

Every ERA5 store matching --pattern in --era5-dir (the daily stores and
the static store) is regridded bilinearly onto the grid of the cropped CERRA
data and written to --out-dir under the same file name. Variables that are
NaN by design (sst over land) are regridded from their valid neighbours
only, so the land mask does not spread over the sea.

The interpolation weights depend only on the two grids: they are computed once,
saved in --weights-dir and reused on later runs. The file name contains a
fingerprint of both grids, so changing either grid creates new weights.

The time dimension is renamed from valid_time to time so that the
output matches the CERRA stores.

Example
-------
    python scripts/data/regrid_era5.py
    python scripts/data/regrid_era5.py --pattern "era5_2003*.zarr" --overwrite
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from downscaling.io import skipping_if_exists, write_to_zarr
from downscaling.regrid import LAT, LON, build_regridder, regrid, weights_path

log = logging.getLogger(__name__)

# ------------------------------------
# Configuration
# ------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ERA5_DIR = ROOT / "data" / "era5"
CERRA_GRID = ROOT / "data" / "cerra" / "cerra_static.zarr"
OUT_DIR = ROOT / "data" / "era5_on_cerra"
WEIGHTS_DIR = ROOT / "data" / "weights"

METHOD = "bilinear"
NAN_VARS = ("sst",)  
OUTPUT_CHUNKS = {"time": 31, "y": -1, "x": -1}


# ------------------------------------
# Pipeline
# ------------------------------------


def same_grid(a: xr.Dataset, b: xr.Dataset) -> bool:
    """True if a and b have identical latitude and longitude coordinates."""
    return np.array_equal(a[LAT].values, b[LAT].values) and np.array_equal(
        a[LON].values, b[LON].values
    )


def regrid_stores(
    paths: list[Path],
    target: xr.Dataset,
    out_dir: Path,
    weights_dir: Path,
    overwrite: bool = False,
) -> None:
    """Regrid each ERA5 store in paths onto the grid of target and write it to out_dir.

    All stores must share the same ERA5 grid; the regridder is built from the
    first store that needs processing and reused for the others.

    Raises
    ------
    ValueError
        If a store is on a different grid than the first one, or if the CERRA
        grid extends beyond the ERA5 region.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    regridder = None
    reference = None
    weights_file = None

    for path in paths:
        out_file = out_dir / path.name
        if skipping_if_exists(out_file, overwrite):
            continue

        ds = xr.open_zarr(path)
        if regridder is None:
            weights_file = weights_path(weights_dir, ds, target, METHOD)
            regridder = build_regridder(ds, target, METHOD, weights_file)
            reference = ds
        elif not same_grid(ds, reference):
            raise ValueError(f"{path.name} is on a different grid than the other ERA5 stores.")

        out = regrid(ds, regridder, nan_vars=NAN_VARS)
        if "valid_time" in out.dims:
            out = out.rename({"valid_time": "time"})
        out.attrs.update(
            regridding=(
                f"{METHOD} onto the CERRA grid with xESMF; "
                f"{', '.join(NAN_VARS)} regridded from valid (non-NaN) neighbours only"
            ),
            regridding_source=path.name,
            regridding_weights=weights_file.name,
        )
        write_to_zarr(out, out_file, OUTPUT_CHUNKS)
        log.info("Regridded %s", path.name)


# ------------------------------------
# Command line interface
# ------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--era5-dir", type=Path, default=ERA5_DIR, help="Folder of ERA5 stores.")
    parser.add_argument(
        "--pattern",
        default="era5_*.zarr",
        help="Glob pattern of the ERA5 stores to regrid (default: %(default)s).",
    )
    parser.add_argument(
        "--cerra-grid",
        type=Path,
        default=CERRA_GRID,
        help="Any cropped CERRA store. Only its latitude/longitude are used.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output folder.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=WEIGHTS_DIR,
        help="Folder where the regridding weights are saved and reused.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs.")
    return parser


def main() -> None:
    """Find the ERA5 stores, load the CERRA grid and regrid every store."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = sorted(args.era5_dir.glob(args.pattern))
    if not paths:
        parser.error(f"no stores matching '{args.pattern}' in {args.era5_dir}")
    if not args.cerra_grid.exists():
        parser.error(f"CERRA grid store not found: {args.cerra_grid}")

    target = xr.open_zarr(args.cerra_grid)
    regrid_stores(paths, target, args.out_dir, args.weights_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
