import fsspec
import json

fs = fsspec.filesystem("hf")

for repo, zarr_name in [
    ("ananyo01/ARCO-EMARS",    "emars_combined.zarr"),
    ("ananyo01/ARCO-MACDA",    "macda_combined.zarr"),
    ("ananyo01/ARCO-OpenMars", "openmars_MY24-27.zarr"),
]:
    try:
        with fs.open(f"datasets/{repo}/{zarr_name}/zarr.json") as f:
            meta = json.load(f)
        print(f"{zarr_name}: zarr_format={meta.get('zarr_format','?')}")
    except Exception as e:
        print(f"{zarr_name}: ERROR {e}")
