from arraylake import Client
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MY = 29
LS_SNAPS = [0, 90, 180, 270]

client = Client()
print("Opening...")
dm  = xr.open_zarr(
    client.get_repo("arco-planetary/ARCO-MACDA")
    .readonly_session("main").store,
    consolidated=False, decode_times=False)
de  = xr.open_zarr(
    client.get_repo("arco-planetary/ARCO-EMARS")
    .readonly_session("main").store,
    group="mean", consolidated=False,
    decode_times=False)
d28 = xr.open_zarr(
    client.get_repo("arco-planetary/ARCO-OpenMARS")
    .readonly_session("main").store,
    group="my28", consolidated=False,
    decode_times=False)

my_m=dm["MY_Ls"].values.astype(float)
ls_m=dm["Ls"].values.astype(float)
my_e=de["anal_mean_MY"].values.astype(float)
ls_e=de["anal_mean_Ls"].values.astype(float)
my_o=d28["MY"].values.astype(float)
ls_o=d28["Ls"].values.astype(float)

lat_m=dm["lat"].values; lon_m=dm["lon"].values
lat_e=de["lat"].values; lon_e=de["lon"].values
lat_o=d28["lat"].values; lon_o=d28["lon"].values

def fix(lon, *fields):
    f=lon.copy(); f[f>180]-=360; s=np.argsort(f)
    return (f[s],)+tuple(x[...,s] for x in fields)

def snap_idx(my_arr, ls_arr, my_t, ls_t):
    # Handle Ls=0 wrap-around
    ls_t = float(ls_t)
    mask = (my_arr==my_t) & np.isfinite(ls_arr)
    return np.where(mask)[0][
        np.argmin(np.abs(ls_arr[mask]-ls_t))]

# ── Load all snapshots ────────────────────────────────────────
print("Loading snapshots...")
snaps = {ds_name: [] for ds_name in
         ["MACDA","EMARS","OpenMARS"]}
ls_found = {ds_name: [] for ds_name in
            ["MACDA","EMARS","OpenMARS"]}

for ls_t in LS_SNAPS:
    for ds_name,ds,my_arr,ls_arr,ps_var in [
        ("MACDA",   dm,  my_m, ls_m, "psurf"),
        ("EMARS",   de,  my_e, ls_e, "anal_mean_ps"),
        ("OpenMARS",d28, my_o, ls_o, "ps"),
    ]:
        ip = snap_idx(my_arr, ls_arr, MY, ls_t)
        ps = ds[ps_var].isel(
            time=ip).compute().values
        ls_actual = float(ls_arr[ip])
        snaps[ds_name].append(ps)
        ls_found[ds_name].append(ls_actual)
        print(f"  {ds_name} Ls={ls_t}° → "
              f"actual Ls={ls_actual:.1f}°")

# Fix longitudes
lon_m_f = fix(lon_m)[0]
lon_e_f = fix(lon_e)[0]
lon_o_f = fix(lon_o)[0]

snaps["MACDA"]    = [fix(lon_m, p)[1]
                      for p in snaps["MACDA"]]
snaps["EMARS"]    = [fix(lon_e, p)[1]
                      for p in snaps["EMARS"]]
snaps["OpenMARS"] = [fix(lon_o, p)[1]
                      for p in snaps["OpenMARS"]]

# ── Plot — 4 rows × 3 cols ────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(15, 16),
                          sharex=True, sharey=True)
#fig.suptitle(
#    f"Surface Pressure (Pa) — MY{MY}\n"
#    "ARCO-MACDA | ARCO-EMARS | ARCO-OpenMARS",
#    fontsize=12)

norm  = mcolors.Normalize(vmin=350, vmax=950)
cmap  = "RdYlBu_r"
titles= ["MACDA", "EMARS",
          "OpenMARS"]
lats  = [lat_m, lat_e, lat_o]
lons  = [lon_m_f, lon_e_f, lon_o_f]
keys  = ["MACDA", "EMARS", "OpenMARS"]

for row, ls_t in enumerate(LS_SNAPS):
    for col, (key, lat, lon) in enumerate(
            zip(keys, lats, lons)):
        ax   = axes[row, col]
        ps   = snaps[key][row]
        ls_a = ls_found[key][row]

        lon2d, lat2d = np.meshgrid(lon, lat)
        cf = ax.contourf(lon2d, lat2d, ps,
                          levels=np.linspace(350,950,25),
                          norm=norm, cmap=cmap,
                          extend="both")

        # Mean value in corner
        wts  = np.cos(np.radians(lat))
        mean = np.average(ps.mean(axis=1),
                           weights=wts)

        if row == 0:
            ax.set_title(titles[col], fontsize=14)
        if col == 0:
            ax.set_ylabel(
                f"Ls≈{ls_t}°\nLat (°N)",
                fontsize=14)
        if row == 3:
            ax.set_xlabel("Lon (°E)", fontsize=14)
        ax.set_xticks([-180,-90,0,90,180])
        ax.grid(True, alpha=0.2)

# Shared colorbar
fig.subplots_adjust(
    left=0.07, right=0.90,
    top=0.93, bottom=0.05,
    hspace=0.06, wspace=0.06)
cbar_ax = fig.add_axes([0.92,0.05,0.02,0.88])
cb = fig.colorbar(cf, cax=cbar_ax, extend="both")
cb.set_label("Surface Pressure (Pa)", fontsize=13)
cb.set_ticks(np.arange(350,951,100))

plt.savefig("ps_seasonal_snapshots.png",
             dpi=150, bbox_inches="tight")
print("Saved ps_seasonal_snapshots.png")
