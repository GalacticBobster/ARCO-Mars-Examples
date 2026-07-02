from arraylake import Client
import xarray as xr

client = Client()
repo = client.get_repo("arco-planetary/ARCO-EMARS")
session = repo.readonly_session("main")

ds_mean = xr.open_zarr(
    session.store, 
    group="mean", 
    consolidated=False, 
    decode_times=False
)

ds_sprd = xr.open_zarr(
    session.store, 
    group="sprd", 
    consolidated=False, 
    decode_times=False
)


print(ds_mean)
