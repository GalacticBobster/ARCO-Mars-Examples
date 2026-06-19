import xarray as xr
import fsspec
import numpy as np


#User defined Huggingface Access Token

#Loading ARCO-EMARS from Huggingface Repository
fs    = fsspec.filesystem("hf")

store_m = fs.get_mapper("datasets/ananyo01/ARCO-EMARS/emars_combined.zarr")
store_s = fs.get_mapper("datasets/ananyo01/ARCO-EMARS/emars_sprd.zarr")

de_m = xr.open_zarr(store_m, decode_times=False)
de_s = xr.open_zarr(store_s, decode_times=False)
