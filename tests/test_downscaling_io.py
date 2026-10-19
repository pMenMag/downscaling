import numpy as np
import pytest
import xarray as xr

from downscaling.io import write_to_zarr

CHUNKS = {"valid_time": 24, "latitude": 3, "longitude": 4}

 
@pytest.fixture
def small_ds():
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), rng.random((48, 3, 4)))},
        coords={
            "valid_time": xr.date_range("2020-01-01", periods=48, freq="h"),
            "latitude": [40.0, 41.0, 42.0],
            "longitude": [0.0, 1.0, 2.0, 3.0],
        },
    )

def test_failed_write(small_ds, tmp_path, monkeypatch):
    out = tmp_path / "out.zarr"
    write_to_zarr(small_ds, out, CHUNKS)

    def fail(self, store, *args, **kwargs):
        store.mkdir()                                 
        (store / "partial_store").write_text("incomplete")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(xr.Dataset, "to_zarr", fail)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_to_zarr(small_ds + 1, out, CHUNKS)

    assert (tmp_path / "out.zarr.tmp").exists()
    result = xr.open_zarr(out).load()
    xr.testing.assert_allclose(result, small_ds.astype("float32"))