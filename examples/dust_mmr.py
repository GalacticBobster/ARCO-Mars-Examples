import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import fsspec, time


def open_hf(repo, zarr):
    fs = fsspec.filesystem("hf")
    return xr.open_zarr(fs.get_mapper(f"datasets/{repo}/{zarr}"),
                        decode_times=False)

def load_retry(da, retries=5, wait=360):
    for i in range(retries):
        try:
            return da.load().values
        except Exception as e:
            if "429" in str(e) and i < retries-1:
                print(f"    429 — waiting {wait}s..."); time.sleep(wait)
            else: raise

# ── Open ──────────────────────────────────────────────────────
print("Opening...")
dm = open_hf("ananyo01/ARCO-MACDA",    "macda_combined.zarr")
de = open_hf("ananyo01/ARCO-EMARS",    "emars_combined.zarr")
d28= open_hf("ananyo01/ARCO-OpenMars", "openmars_MY28-35.zarr")

# ── Indices ───────────────────────────────────────────────────
print("Loading indices...")
my_m = load_retry(dm["MY_Ls"]);  time.sleep(5)
ls_m = load_retry(dm["Ls"]);     time.sleep(5)
my_e = load_retry(de["anal_mean_MY"]); time.sleep(5)
ls_e = load_retry(de["anal_mean_Ls"]); time.sleep(5)
my_28= load_retry(d28["MY"]);    time.sleep(5)
ls_28= load_retry(d28["Ls"]);    time.sleep(5)

# ── Find MY28  ─────────────────────────────
LS_TARGET = 250.0
MY        = 28

mask_m = (my_m==MY) & np.isfinite(ls_m)
mask_e = (my_e==MY) & np.isfinite(ls_e)
mask_o = (my_28==MY) & np.isfinite(ls_28)

# Single timestep closest
def peak_idx(mask, ls):
    ls_sub = ls[mask]
    return np.where(mask)[0][np.argmin(np.abs(ls_sub - LS_TARGET))]

ip_m = peak_idx(mask_m, ls_m)
ip_e = peak_idx(mask_e, ls_e) if mask_e.sum()>0 else None
ip_o = peak_idx(mask_o, ls_28)

# Average over Ls 240-260° for T profiles
def band_idx(mask, ls, lo=240, hi=260):
    ls_sub = ls[mask]
    return np.where(mask)[0][(ls_sub>=lo)&(ls_sub<=hi)]

ib_m = band_idx(mask_m, ls_m)
ib_e = band_idx(mask_e, ls_e) if mask_e.sum()>0 else None
ib_o = band_idx(mask_o, ls_28)

print(f"Peak: MACDA Ls={ls_m[ip_m]:.1f}°  "
      f"OpenMARS Ls={ls_28[ip_o]:.1f}°")
print(f"Band: MACDA {len(ib_m)} steps  OpenMARS {len(ib_o)} steps")

# ── Grid ──────────────────────────────────────────────────────
lat_m = dm["lat"].values
lon_m = dm["lon"].values
lev_m = dm["lev"].values
lat_o = d28["lat"].values
lon_o = d28["lon"].values
lev_o = d28["lev"].values
lat_e = de["lat"].values

ak = de["anal_mean_ak"].values
bk = de["anal_mean_bk"].values
# Load actual mean ps
ps_e_mean = load_retry(de["anal_mean_ps"].mean()).item()
ps_m_mean = load_retry(dm["psurf"].mean()).item()
ps_o_mean = load_retry(d28["ps"].mean()).item()

# Fix three lines:
P_half = ak + bk * ps_e_mean          # was 600.0
P_e    = (P_half[:-1] + P_half[1:]) / 2
P_m    = lev_m * ps_m_mean            # was 600.0
P_o    = lev_o * ps_o_mean            # was 600.0

# ── Load T profiles (Ls 240-260° zonal mean) ─────────────────
print("Loading T profiles...")
T_m = load_retry(dm["temp"].isel(time=ib_m).mean(dim=["time","lon"]))
time.sleep(20)
T_o = load_retry(d28["temp"].isel(time=ib_o).mean(dim=["time","lon"]))
time.sleep(20)
if ib_e is not None and len(ib_e)>0:
    T_e = load_retry(de["anal_mean_T"].isel(time=ib_e)
                      .mean(dim=["time","lon"]))
    time.sleep(20)
else:
    T_e = None

# ── Load dust fields (single timestep) ───────────────────────
print("Loading dust fields...")
# MACDA dustmmr: zonal mean (lev, lat)
dustmmr = load_retry(dm["dustmmr"].isel(time=ip_m).mean(dim="lon"))
time.sleep(20)

# MACDA coldust OD map (lat, lon)
coldust = load_retry(dm["coldust"].isel(time=ip_m))
psurf   = load_retry(dm["psurf"].isel(time=ip_m))
tau_m   = coldust * 610.0 / psurf
time.sleep(20)


print("All data loaded.")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))


# ── Panel 1: T vertical profiles ─────────────────────────────
ax = axes[0]
colors_ds = {"MACDA":"tomato","OpenMARS":"green","EMARS":"steelblue"}

def lat_mean(T, lats, lo, hi):
    idx = (lats>=lo) & (lats<=hi)
    return T[:, idx].mean(axis=1)

for label, T, P, lats in [
    ("MACDA",    T_m, P_m, lat_m),
    ("OpenMARS", T_o, P_o, lat_o),
]:
    T_eq  = lat_mean(T, lats, -10, 10)
    T_pol = lat_mean(T, lats,  70, 90)
    ax.plot(T_eq,  P, color=colors_ds[label], lw=1.5,
            ls="-",  label=f"{label} (-10 to 10 deg)")
    ax.plot(T_pol, P, color=colors_ds[label], lw=1.5,
            ls="--", label=f"{label} (70 to 90 deg)")

if T_e is not None:
    T_eq_e  = lat_mean(T_e, lat_e, -10, 10)
    T_pol_e = lat_mean(T_e, lat_e,  70, 90)
    ax.plot(T_eq_e,  P_e, color="steelblue", lw=1.5,
            ls="-",  label="EMARS (-10 to 10 deg)")
    ax.plot(T_pol_e, P_e, color="steelblue", lw=1.5,
            ls="--", label="EMARS (70 to 90 deg)")

ax.set_xlabel("Temperature (K)", fontsize=10)
ax.set_ylabel("Pressure (Pa)",   fontsize=10)
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_ylim(700, 0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=1)

# ── Panel 2: Dust MMR cross-section ──────────────────────────
ax = axes[1]
vmin, vmax = 1e-8, 1e-4
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cf = ax.contourf(lat_m, P_m, dustmmr,
                  levels=np.logspace(-8, -4, 25),
                  norm=norm, cmap="YlOrRd", extend="both")
cb = plt.colorbar(cf, ax=ax, label="Dust MMR (kg kg⁻¹)",
                   pad=0.02, shrink=0.9)
cb.ax.tick_params(labelsize=7)
ax.set_xlabel("Latitude (°N)", fontsize=10)
ax.set_ylabel("Pressure (Pa)", fontsize=10)
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_ylim(700, 0.5)
ax.set_xlim(-90, 90)
ax.grid(True, alpha=0.2, color="white")


# ── Panel 3: Column Dust OD map ───────────────────────────────
ax = axes[2]
norm_od = mcolors.Normalize(vmin=0, vmax=3.0)
cf = ax.contourf(lon_m, lat_m, tau_m,
                  levels=np.linspace(0, 3.0, 25),
                  norm=norm_od, cmap="hot_r", extend="max")
cb = plt.colorbar(cf, ax=ax, label="Dust OD (610Pa ref)",
                   pad=0.02, shrink=0.9)
cb.ax.tick_params(labelsize=7)



ax.set_xlabel("Longitude (°E)", fontsize=10)
ax.set_ylabel("Latitude (°N)",  fontsize=10)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks([-180,-90,0,90,180])
ax.set_yticks([-60,-30,0,30,60])


plt.tight_layout()
plt.savefig("arco_dust_profiles.png", dpi=150, bbox_inches="tight")
print("Saved arco_dust_profiles.png")
