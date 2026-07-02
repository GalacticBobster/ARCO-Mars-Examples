import numpy as np
import xarray as xr
from arraylake import Client

datasets = {
    "ARCO-EMARS":    ("arco-planetary/ARCO-EMARS",    "mean"),
    "ARCO-EMARS-SPRD":("arco-planetary/ARCO-EMARS",   "sprd"),
    "ARCO-MACDA":    ("arco-planetary/ARCO-MACDA",     ""),
    "ARCO-OpenMARS-24":("arco-planetary/ARCO-OpenMARS","my24"),
    "ARCO-OpenMARS-28":("arco-planetary/ARCO-OpenMARS","my28"),
}


client = Client()
repo = client.get_repo("arco-planetary/ARCO-EMARS")

for name, (repo, group) in datasets.items():
    repo_name = client.get_repo(repo)
    session = repo_name.readonly_session("main")
    ds = xr.open_zarr(session.store, group = group, consolidated = False, decode_times = False)
    print(f"\n{name}")
    print(f"  Dims:    {dict(ds.sizes)}")
    print(f"  Vars:    {len(ds.data_vars)} variables")
    print(f"  2D vars: {[v for v in ds.data_vars if len(ds[v].dims)==3]}")
    print(f"  3D vars: {[v for v in ds.data_vars if len(ds[v].dims)==4]}")

for name, (repo, group) in datasets.items():
    repo_name = client.get_repo(repo)
    session = repo_name.readonly_session("main")
    ds = xr.open_zarr(session.store, group = group, consolidated = False, decode_times = False)
    t  = ds["time"].values
    print(f"\n{name}")
    print(f"  time attrs: {ds['time'].attrs}")
    print(f"  first 5 values: {t[:5]}")
    print(f"  median diff: {np.median(np.diff(t[:100])):.4f}")
