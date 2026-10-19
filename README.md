# Downscaling ERA5 to CERRA over Catalonia

Statistical and machine-learning downscaling of daily 2 m temperature
(mean, maximum and minimum) from **ERA5** (~31 km) to **CERRA** (5.5 km) over
Catalonia, validated against **Meteocat XEMA** weather stations.
 
The aim is to compare simple baselines with deep-learning models.

> **Work in progress.** The ERA5 and CERRA data pipelines are done. The models and the
> station validation are next. Commands and paths may still change.

## Current data

| Dataset | Role | Resolution | Source |
| --- | --- | --- | --- |
| ERA5 (single and pressure levels) | Input (predictors) | 0.25°, hourly | Earth Data Hub (DestinE), Zarr |
| CERRA | Target | 5.5 km, 3-hourly | Copernicus Climate Data Store, GRIB |

## Installation
 
The environment uses conda-forge.

```bash
git clone https://github.com/pMenMag/downscaling.git
cd downscaling
conda env create -f environment.yml
conda activate downscalingproj
pip install -e . --no-deps
```

### Credentials
 
Two accounts are needed. 
 
- **Copernicus Climate Data Store** (CERRA): put your API key in `~/.cdsapirc`
  (see the [CDS API instructions](https://cds.climate.copernicus.eu/how-to-api)).
- **Earth Data Hub** (ERA5): put your key in `~/.netrc` and restrict it with
  `chmod 600 ~/.netrc`.

## Usage
 
Each step is a script with `--help`. A typical run:
 
```bash
# 1. ERA5: crop to Catalonia and daily stores (single and pressure levels). Optionally also crop and store static variables
python scripts/data/era5_store.py --start 2003-01-01 --end 2005-12-31 --static
 
# 2. CERRA: download, crop to Catalonia, then daily aggregation. Optionally also download and crop static variables.
python scripts/data/cerra_download.py 2003 2005 --stage all --static
 
# 3. Regrid ERA5 onto the CERRA grid
python scripts/data/regrid_era5.py
```

## Development
 
GitHub Actions runs `ruff check`, `ruff format --check` and `pytest` on every
push and pull request.


 