from arraylake import Client
import xarray as xr

client = Client()
repo = client.get_repo("arco-planetary/ARCO-OpenMARS")

session = repo.readonly_session("main")


ds = xr.open_zarr(session.store, group = "my24", consolidated = False, decode_times = False)
print(ds)
