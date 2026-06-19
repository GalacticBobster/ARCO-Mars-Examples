import xarray as xr
import fsspec
import numpy as np


#Loading ARCO-MACDA from Huggingface Repository
fs    = fsspec.filesystem("hf")
store = fs.get_mapper("datasets/ananyo01/ARCO-MACDA/macda_combined.zarr")
ds    = xr.open_zarr(store, decode_times=False)
