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

### With Docker

Instead of creating the conda environment, you can use the Docker image,
which contains the environment and the code:

```bash
docker pull ghcr.io/pmenmag/downscaling:latest
```

Without a command, the container runs the tests. After having the credential files, you can run a container with your credentials and save the data on your computer by doing the following:
```bash
RUN="docker run --rm \
  -v $PWD/data:/app/data \
  -v $HOME/.netrc:/home/mambauser/.netrc:ro \
  -v $HOME/.cdsapirc:/home/mambauser/.cdsapirc:ro \
  ghcr.io/pmenmag/downscaling:latest"

# Then, for example, to store ERA5 data:
$RUN python scripts/data/era5_store.py 2000-01-01 2000-01-31
```

## Usage
 
Each step is a script with `--help`. A typical run:
 
```bash
# 1. ERA5: crop to Catalonia and daily stores (single and pressure levels). Optionally also crop and store static variables
python scripts/data/era5_store.py 2003-01-01 2005-12-31 --static
 
# 2. CERRA: download, crop to Catalonia, then daily aggregation. Optionally also download and crop static variables.
python scripts/data/cerra_download.py 2003 2005 --stage all --static
 
# 3. Regrid ERA5 onto the CERRA grid
python scripts/data/regrid_era5.py
```

With Docker, prefix each command with $RUN (see above). For example, 
```bash
$RUN python scripts/data/regrid_era5.py
```

## Development
 
GitHub Actions runs 
```bash
pytest
ruff check .
ruff format --check .
```
on every
push and pull request. A new Docker image is built and pushed to `ghcr.io` for 
each version tag (`v*`).
