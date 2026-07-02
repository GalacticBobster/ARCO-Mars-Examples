import numpy as np
import xarray as xr
import fsspec

def open_hf(repo, zarr):
    fs = fsspec.filesystem("hf")
    return xr.open_zarr(fs.get_mapper(f"datasets/{repo}/{zarr}"),
                        decode_times=False)

datasets = {
    "ARCO-EMARS":    ("ananyo01/ARCO-EMARS",    "emars_combined.zarr"),
    "ARCO-EMARS-SPRD":("ananyo01/ARCO-EMARS",   "emars_sprd.zarr"),
    "ARCO-MACDA":    ("ananyo01/ARCO-MACDA",     "macda_combined.zarr"),
    "ARCO-OpenMARS-24":("ananyo01/ARCO-OpenMars","openmars_MY24-27.zarr"),
    "ARCO-OpenMARS-28":("ananyo01/ARCO-OpenMars","openmars_MY28-35.zarr"),
}

for name, (repo, zarr) in datasets.items():
    ds = open_hf(repo, zarr)
    print(f"\n{name}")
    print(f"  Dims:    {dict(ds.sizes)}")
    print(f"  Vars:    {len(ds.data_vars)} variables")
    print(f"  2D vars: {[v for v in ds.data_vars if len(ds[v].dims)==3]}")
    print(f"  3D vars: {[v for v in ds.data_vars if len(ds[v].dims)==4]}")

for name, (repo, zarr) in datasets.items():
    ds = open_hf(repo, zarr)
    t  = ds["time"].values
    print(f"\n{name}")
    print(f"  time attrs: {ds['time'].attrs}")
    print(f"  first 5 values: {t[:5]}")
    print(f"  median diff: {np.median(np.diff(t[:100])):.4f}")
