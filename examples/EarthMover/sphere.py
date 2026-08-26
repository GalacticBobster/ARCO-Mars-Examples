import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from arraylake import Client
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# 1. Connect to Arraylake and fetch the 3D Temperature Field
client = Client()
repo = client.get_repo("arco-planetary/ARCO-MACDA")
session = repo.readonly_session("main")

ds = xr.open_zarr(session.store, consolidated=False, decode_times=False)

num_frames = 48
    
# Extract 'temp' or 'T' (3D Atmospheric Temperature)
# Slicing the lowest vertical layer: isel(lev=0)
air_temp_3d = ds["temp"] if "temp" in ds.variables else ds["T"]
air_temp_surface = air_temp_3d.isel(lev=0, time=slice(0, num_frames))

# Force arrays into memory
lon = ds["lon"].values
lat = ds["lat"].values
temp_frames = air_temp_surface.values



# 2. Setup Figure Canvas (Flat 2D map projection)
fig = plt.figure(figsize=(10, 6), facecolor="#111111")
ax = plt.axes(
    projection=ccrs.PlateCarree(),
    facecolor="#111111",
)
ax.set_global()
ax.axis("off")

# Establish static data boundaries
vmin, vmax = np.min(temp_frames), np.max(temp_frames)

# 3. Create a static colorbar using a 2D Meshgrid
dummy_data, dummy_lon = add_cyclic_point(temp_frames[0], coord=lon)
# FIX: Convert 1D vectors into 2D gridded arrays for transform_first=True
lon_2d_init, lat_2d_init = np.meshgrid(dummy_lon, lat)

initial_setup = ax.contourf(
    lon_2d_init,
    lat_2d_init,
    dummy_data,
    levels=60,
    transform=ccrs.PlateCarree(),
    transform_first=True,  # Keeps grid structure perfectly aligned
    cmap="coolwarm",
    vmin=vmin,
    vmax=vmax,
)

cbar = plt.colorbar(
    initial_setup,
    orientation="horizontal",
    pad=0.05,
    shrink=0.6,
    aspect=40,
    ax=ax,
)
cbar.set_label("Lowest Layer Air Temperature [K]", color="white", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=8, labelcolor="white")
cbar.outline.set_visible(False)


# 4. The Animation Frame Update Function
def update(frame):
    global ax
    # Clear map contour collections from the previous frame
    for coll in ax.collections:
        coll.remove()

    # Wrap the 0-360 longitude seam
    data_cyclic, lon_cyclic = add_cyclic_point(temp_frames[frame], coord=lon)
    
    # FIX: Convert 1D vectors into 2D gridded arrays for this frame
    lon_2d, lat_2d = np.meshgrid(lon_cyclic, lat)

    # Render flat 2D heatmap with 2D coordinate matrices
    contour = ax.contourf(
        lon_2d,
        lat_2d,
        data_cyclic,
        levels=60,
        transform=ccrs.PlateCarree(),
        transform_first=True,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    # Clean text tracking frame indices
    if hasattr(update, "txt"):
        update.txt.remove()
    update.txt = ax.text(
        0.02,
        0.92,
        f"Step: {frame:02d}",
        transform=ax.transAxes,
        color="white",
        fontsize=12,
        fontweight="bold",
        fontfamily="monospace",
    )

    return contour.collections + [update.txt]


# 5. Compile and Save the Flat GIF
print("🎬 Rendering flat 2D air temperature animation...")
ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=150, blit=False
)

ani.save(
    "macda_mars_flat_air_temp.gif",
    writer="pillow",
    savefig_kwargs={"facecolor": "#111111"},
)
plt.close()

print("✓ Success! Saved to macda_mars_flat_air_temp.gif")
