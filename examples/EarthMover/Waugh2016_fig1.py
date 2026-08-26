from arraylake import Client
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

# ── Connect to Earthmover ─────────────────────────────────────
client = Client()

def open_lake(repo, group=""):
    r       = client.get_repo(repo)
    session = r.readonly_session("main")
    if group:
        return xr.open_zarr(session.store, group=group,
                            consolidated=False,
                            decode_times=False)
    return xr.open_zarr(session.store,
                        consolidated=False,
                        decode_times=False)

# ── Open datasets ─────────────────────────────────────────────
print("Opening datasets from Earthmover...")
dm  = open_lake("arco-planetary/ARCO-MACDA")
de  = open_lake("arco-planetary/ARCO-EMARS",    "mean")
d24 = open_lake("arco-planetary/ARCO-OpenMARS", "my24")
d28 = open_lake("arco-planetary/ARCO-OpenMARS", "my28")

# ── Parameters ────────────────────────────────────────────────
# Reproduce Waugh et al. 2016 Fig 1
# Monthly mean around northern winter solstice Ls=270°
# ±15° window = 30 sols
LS_CENTER = 270.0
LS_HALF   = 15.0          # ±15° = 30 sol window
LS_MIN    = LS_CENTER - LS_HALF   # 255°
LS_MAX    = LS_CENTER + LS_HALF   # 285°
YEARS     = [24]  # average over 3 clear years

# ── Load indices ──────────────────────────────────────────────
print("Loading indices...")
my_m  = dm["MY_Ls"].values.astype(float)
ls_m  = dm["Ls"].values.astype(float)
my_e  = de["anal_mean_MY"].values.astype(float)
ls_e  = de["anal_mean_Ls"].values.astype(float)
my_24 = d24["MY"].values.astype(float)
ls_24 = d24["Ls"].values.astype(float)
my_28 = d28["MY"].values.astype(float)
ls_28 = d28["Ls"].values.astype(float)

# ── Pressure axes ─────────────────────────────────────────────
# MACDA/OpenMARS sigma → pressure
ps_m_mean = float(dm["psurf"].mean().compute())
ps_o_mean = float(d28["ps"].mean().compute())
lev_m     = dm["lev"].values
lev_o     = d28["lev"].values
P_m       = lev_m * ps_m_mean      # Pa
P_o       = lev_o * ps_o_mean      # Pa

# EMARS hybrid sigma-pressure
ak        = de["anal_mean_ak"].values
bk        = de["anal_mean_bk"].values
ps_e_mean = float(de["anal_mean_ps"].mean().compute())
P_half    = ak + bk * ps_e_mean
P_e       = (P_half[:-1] + P_half[1:]) / 2   # Pa

# Convert Pa → hPa for plot (matching Waugh 2016)
P_m_hpa   = P_m / 100.0
P_o_hpa   = P_o / 100.0
P_e_hpa   = P_e / 100.0

lat_m = dm["lat"].values
lat_o = d28["lat"].values
lat_e = de["lat"].values

print(f"MACDA P:   {P_m_hpa.min():.4f} → {P_m_hpa.max():.2f} hPa")
print(f"OpenMARS P:{P_o_hpa.min():.4f} → {P_o_hpa.max():.2f} hPa")
print(f"EMARS P:   {P_e_hpa.min():.4f} → {P_e_hpa.max():.2f} hPa")

# ── Compute Ls=270° ±15° mean T and U ─────────────────────────
def get_TU_mean(my_arr, ls_arr, ds, T_var, U_var,
                lat_dim, lev_dim, lon_dim,
                years, ls_min, ls_max):
    """
    Compute zonal mean T and U averaged over
    Ls window and multiple years.
    Returns T(lat,lev), U(lat,lev)
    """
    T_all = []
    U_all = []

    for y in years:
        mask = ((my_arr == y) &
                (ls_arr >= ls_min) &
                (ls_arr <= ls_max) &
                np.isfinite(ls_arr))

        if mask.sum() == 0:
            print(f"  MY{y}: no data in Ls={ls_min}-{ls_max}°")
            continue

        print(f"  MY{y}: {mask.sum()} steps")

        # Zonal mean: mean over lon → (time, lev, lat)
        T = ds[T_var].isel(
            time=mask).mean(
            dim=[lon_dim, "time"]).compute().values
        U = ds[U_var].isel(
            time=mask).mean(
            dim=[lon_dim, "time"]).compute().values

        # Transpose to (lat, lev) if needed
        # xarray returns (lev, lat) after mean
        # check dims
        if T.ndim == 2:
            # Check which axis is lat
            if T.shape[0] != len(ds[lat_dim].values):
                T = T.T
                U = U.T

        T_all.append(T)
        U_all.append(U)

    if not T_all:
        return None, None

    return (np.nanmean(T_all, axis=0),
            np.nanmean(U_all, axis=0))

# ── MACDA ─────────────────────────────────────────────────────
print("\nComputing MACDA T, U at Ls=270°±15°...")
T_m, U_m = get_TU_mean(
    my_m, ls_m, dm,
    T_var="temp", U_var="uwind",
    lat_dim="lat", lev_dim="lev", lon_dim="lon",
    years=YEARS, ls_min=LS_MIN, ls_max=LS_MAX)

# ── EMARS ─────────────────────────────────────────────────────
print("\nComputing EMARS T, U at Ls=270°±15°...")
# EMARS only goes to MY33 — use MY29-32
emars_years = [y for y in YEARS if y <= 33]
T_e, U_e = get_TU_mean(
    my_e, ls_e, de,
    T_var="anal_mean_T", U_var="anal_mean_U",
    lat_dim="lat", lev_dim="pfull", lon_dim="lon",
    years=emars_years,
    ls_min=LS_MIN, ls_max=LS_MAX)

# EMARS U uses latu not lat — handle separately
if T_e is None:
    print("  Loading T and U separately for EMARS...")
    T_list = []
    U_list = []
    for y in emars_years:
        mask = ((my_e == y) &
                (ls_e >= LS_MIN) &
                (ls_e <= LS_MAX) &
                np.isfinite(ls_e))
        if mask.sum() == 0:
            continue
        print(f"  MY{y}: {mask.sum()} steps")

        T_y = de["anal_mean_T"].isel(
            time=mask).mean(
            dim=["lon","time"]).compute().values  # (pfull, lat)
        U_y = de["anal_mean_U"].isel(
            time=mask).mean(
            dim=["lon","time"]).compute().values  # (pfull, latu)

        T_list.append(T_y.T)   # → (lat, pfull)
        U_list.append(U_y.T)   # → (latu, pfull)

    T_e = np.nanmean(T_list, axis=0) if T_list else None
    U_e = np.nanmean(U_list, axis=0) if U_list else None

# ── OpenMARS ──────────────────────────────────────────────────
print("\nComputing OpenMARS T, U at Ls=270°±15°...")
T_o_list = []
U_o_list = []
for y in YEARS:
    # Select correct store
    if y <= 27:
        ds_o  = d24
        my_o  = my_24
        ls_o  = ls_24
    else:
        ds_o  = d28
        my_o  = my_28
        ls_o  = ls_28

    mask = ((my_o == y) &
            (ls_o >= LS_MIN) &
            (ls_o <= LS_MAX) &
            np.isfinite(ls_o))

    if mask.sum() == 0:
        print(f"  MY{y}: no data")
        continue

    print(f"  MY{y}: {mask.sum()} steps")

    T_y = ds_o["temp"].isel(
        time=mask).mean(
        dim=["lon","time"]).compute().values  # (lev, lat)
    U_y = ds_o["u"].isel(
        time=mask).mean(
        dim=["lon","time"]).compute().values

    T_o_list.append(T_y.T)  # → (lat, lev)
    U_o_list.append(U_y.T)

T_o = np.nanmean(T_o_list, axis=0) if T_o_list else None
U_o = np.nanmean(U_o_list, axis=0) if U_o_list else None

# ── Interpolate all to common pressure grid ───────────────────
# Match Waugh 2016: 0.001 to 10 hPa range
P_COMMON = np.logspace(
    np.log10(0.001), np.log10(10), 60)  # hPa

def interp_to_common_P(field, P_in, P_out):
    """Interpolate (lat, lev) field to common P grid."""
    n_lat = field.shape[0]
    out   = np.full((n_lat, len(P_out)), np.nan)
    for i in range(n_lat):
        valid = np.isfinite(field[i])
        if valid.sum() < 2:
            continue
        f = interp1d(np.log(P_in[valid]),
                     field[i][valid],
                     bounds_error=False,
                     fill_value=np.nan)
        out[i] = f(np.log(P_out))
    return out

print("\nInterpolating to common pressure grid...")
if T_m is not None:
    T_m_c = interp_to_common_P(T_m, P_m_hpa, P_COMMON)
    U_m_c = interp_to_common_P(U_m, P_m_hpa, P_COMMON)
if T_e is not None:
    T_e_c = interp_to_common_P(T_e, P_e_hpa, P_COMMON)
    U_e_c = interp_to_common_P(
        U_e, P_e_hpa, P_COMMON)
if T_o is not None:
    T_o_c = interp_to_common_P(T_o, P_o_hpa, P_COMMON)
    U_o_c = interp_to_common_P(U_o, P_o_hpa, P_COMMON)

# ── Interpolate to common lat grid ────────────────────────────
# Northern hemisphere only: 0-90° (matching Waugh 2016)
LAT_COMMON = np.linspace(0, 90, 46)

def interp_lat(field, lat_in, lat_out):
    """Interpolate (lat, lev) to common lat grid."""
    n_lev = field.shape[1]
    out   = np.full((len(lat_out), n_lev), np.nan)
    for l in range(n_lev):
        col = field[:, l]
        # Use only NH
        nh  = lat_in >= 0
        if nh.sum() < 2:
            continue
        f   = interp1d(lat_in[nh], col[nh],
                       bounds_error=False,
                       fill_value=np.nan)
        out[:, l] = f(lat_out)
    return out

if T_m is not None:
    T_m_c = interp_lat(T_m_c, lat_m, LAT_COMMON)
    U_m_c = interp_lat(U_m_c, lat_m, LAT_COMMON)
if T_e is not None:
    T_e_c = interp_lat(T_e_c, lat_e, LAT_COMMON)
    U_e_c = interp_lat(U_e_c, lat_e, LAT_COMMON)
if T_o is not None:
    T_o_c = interp_lat(T_o_c, lat_o, LAT_COMMON)
    U_o_c = interp_lat(U_o_c, lat_o, LAT_COMMON)

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                          sharex=True, sharey=True)
#fig.suptitle(
#    "Zonal-Mean Temperature (T) and Zonal Wind (U)\n"
#    f"Northern Winter Solstice Ls={LS_CENTER}° ±{LS_HALF}° "
#    f"(~30 sols)\n"
#    f"MY{YEARS[0]}-{YEARS[-1]} mean | "
#    "Reproducing Waugh et al. (2016) Fig. 1",
#    fontsize=11)

datasets = [
    (T_m_c, U_m_c, "MACDA"),
    (T_e_c, U_e_c, "EMARS"),
    (T_o_c, U_o_c, "OpenMARS"),
]

# Colorscales matching Waugh 2016
T_levels = np.arange(100, 240, 5)
U_levels = np.arange(-170, 200, 8)

T_cmap   = "RdYlBu_r"
U_norm   = mcolors.TwoSlopeNorm(vmin=-150,
                                  vcenter=0,
                                  vmax=150)
U_cmap   = "RdBu_r"

for col, (T, U, label) in enumerate(datasets):
    # ── T panel (top row) ─────────────────────────────────────
    ax = axes[0, col]
    if T is not None:
        cf_T = ax.contourf(LAT_COMMON, P_COMMON, T.T,
                            levels=T_levels,
                            cmap=T_cmap,
                            extend="both")
        ax.contour(LAT_COMMON, P_COMMON, T.T,
                   levels=T_levels[::2],
                   colors="k",
                   linewidths=0.5,
                   alpha=0.6)
        if col == 2:
            plt.colorbar(cf_T, ax=ax,
                          label="T (K)",
                          pad=0.02, shrink=0.9)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(7, 0.001)
    ax.set_title(f"{label} - T")
    ax.set_ylabel("Pressure (hPa)")
    ax.axhline(0.1, color="gray", lw=0.5,
               ls=":", alpha=0.5)   # mesopause
    ax.set_xlim(0, 90)
    ax.grid(True, alpha=0.2)

    # ── U panel (bottom row) ──────────────────────────────────
    ax = axes[1, col]
    if U is not None:
        cf_U = ax.contourf(LAT_COMMON, P_COMMON, U.T,
                            levels=U_levels,
                            cmap=U_cmap,
                            norm=U_norm,
                            extend="both")
        ax.contour(LAT_COMMON, P_COMMON, U.T,
                   levels=U_levels[::2],
                   colors="k",
                   linewidths=0.5,
                   alpha=0.6)
        if col == 2:
            plt.colorbar(cf_U, ax=ax,
                          label="U (m/s)",
                          pad=0.02, shrink=0.9)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(7, 0.001)
    ax.set_title(f"{label} - U")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("Latitude (°N)")
    ax.axhline(0.1, color="gray", lw=0.5,
               ls=":", alpha=0.5)
    ax.axhline(1.0, color="gray", lw=0.8,
               ls="--", alpha=0.3)
    ax.set_xlim(0, 90)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("waugh2016_fig1_ARCO.png",
            dpi=150, bbox_inches="tight")
print("\nSaved waugh2016_fig1_ARCO.png")

# ── Print summary statistics ───────────────────────────────────
print(f"\n{'='*55}")
print(f"Summary — Ls={LS_CENTER}°±{LS_HALF}° NH mean")
print(f"{'='*55}")
for label, T, U, P in [
    ("MACDA",    T_m_c, U_m_c, P_COMMON),
    ("EMARS",    T_e_c, U_e_c, P_COMMON),
    ("OpenMARS", T_o_c, U_o_c, P_COMMON),
]:
    if T is None:
        print(f"{label}: no data")
        continue
    # Find jet maximum (U max in NH)
    u_max = np.nanmax(U)
    u_min = np.nanmin(U)
    t_mean= np.nanmean(T)
    print(f"{label}:")
    print(f"  T mean (NH):    {t_mean:.1f} K")
    print(f"  U max (jet):    {u_max:.1f} m/s")
    print(f"  U min (eastly): {u_min:.1f} m/s")
