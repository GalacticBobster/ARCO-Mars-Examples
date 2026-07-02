import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import fsspec, time


def open_hf(repo, zarr):
    fs = fsspec.filesystem("hf")
    return xr.open_zarr(fs.get_mapper(f"datasets/{repo}/{zarr}"),
                        decode_times=False)

def load_retry(da, retries=5, wait=360):
    for i in range(retries):
        try:
            return da.load()
        except Exception as e:
            if "429" in str(e) and i < retries-1:
                print(f"    429 — waiting {wait}s..."); time.sleep(wait)
            else: raise



# ── Open ──────────────────────────────────────────────────────
print("Opening datasets...")
de_m = open_hf("ananyo01/ARCO-EMARS",    "emars_combined.zarr")
dm   = open_hf("ananyo01/ARCO-MACDA",    "macda_combined.zarr")
d24  = open_hf("ananyo01/ARCO-OpenMars", "openmars_MY24-27.zarr")
d28  = open_hf("ananyo01/ARCO-OpenMars", "openmars_MY28-35.zarr")

# ── 1D indices ────────────────────────────────────────────────
print("Loading indices...")
my_e  = load_retry(de_m["anal_mean_MY"]).astype(float); time.sleep(5)
ls_e  = load_retry(de_m["anal_mean_Ls"]).astype(float); time.sleep(5)
my_m  = load_retry(dm["MY_Ls"]).astype(float);          time.sleep(5)
ls_m  = load_retry(dm["Ls"]).astype(float);             time.sleep(5)
my_24 = load_retry(d24["MY"]).astype(float);            time.sleep(5)
ls_24 = load_retry(d24["Ls"]).astype(float);            time.sleep(5)
my_28 = load_retry(d28["MY"]).astype(float);            time.sleep(5)
ls_28 = load_retry(d28["Ls"]).astype(float);            time.sleep(5)

# ── Find pressure levels ───────────────────────────────────────
ak   = de_m["anal_mean_ak"].values
bk   = de_m["anal_mean_bk"].values
P_e  = ak + bk * load_retry(de_m["anal_mean_ps"]
                              .mean()).values.item()

ps_m_mean = load_retry(dm["psurf"].mean()).values.item()
ps_o_mean = load_retry(d24["ps"].mean()).values.item()

TARGET_P = [100.0, 50.0]   # Pa

def find_lev_e(target):
    return int(np.argmin(np.abs(P_e - target)))

def find_lev_sigma(lev_vals, target, ps_actual):
    return int(np.argmin(np.abs(lev_vals * ps_actual - target)))

levs_e = {p: find_lev_e(p) for p in TARGET_P}
levs_m = {p: find_lev_sigma(dm["lev"].values,  p, ps_m_mean)
           for p in TARGET_P}
levs_o = {p: find_lev_sigma(d24["lev"].values, p, ps_o_mean)
           for p in TARGET_P}

print("Pressure levels:")
for p in TARGET_P:
    print(f"  {p:5.0f} Pa — EMARS:{levs_e[p]} "
          f"(P={P_e[levs_e[p]]:.1f}Pa)  "
          f"MACDA:{levs_m[p]} "
          f"(σ={dm['lev'].values[levs_m[p]]:.4f})  "
          f"OpenMARS:{levs_o[p]} "
          f"(σ={d24['lev'].values[levs_o[p]]:.4f})")

# ── Weights ───────────────────────────────────────────────────
wts_e  = np.cos(np.radians(de_m["lat"]))
wts_eu = np.cos(np.radians(de_m["latu"]))
wts_m  = np.cos(np.radians(dm["lat"]))
wts_o  = np.cos(np.radians(d24["lat"]))

# ── Ls bins ───────────────────────────────────────────────────
LS_E = np.arange(0, 361, 5)
LS_C = (LS_E[:-1] + LS_E[1:]) / 2

def bin_ls(ls, val):
    m = np.full(len(LS_C), np.nan)
    for i,(l0,l1) in enumerate(zip(LS_E[:-1],LS_E[1:])):
        idx = (ls>=l0)&(ls<l1)&np.isfinite(val)
        if idx.sum() > 1: m[i] = val[idx].mean()
    return m

# ── Selected years ────────────────────────────────────────────
YEARS  = [25, 28, 32]
LABELS = {25:"MY25 (storm)", 28:"MY28 (GDS)", 32:"MY32 (clear)"}
VARS   = ["T", "U", "V"]
VARS_L = {"T":"Temperature (K)",
           "U":"Zonal Wind (m s⁻¹)",
           "V":"Meridional Wind (m s⁻¹)"}
COLORS = {"EMARS":"steelblue", "MACDA":"tomato", "OpenMARS":"green"}
STYLES = {"EMARS":"-",         "MACDA":"--",     "OpenMARS":":"}

# ── Plot: rows=vars, cols=years, panels per cell = 3 pressures ─
fig, axes = plt.subplots(len(VARS), len(YEARS),
                          figsize=(5*len(YEARS), 4*len(VARS)))
#fig.suptitle("ARCO-Mars — T, U, V at 100/50/1 Pa\n"
#             "EMARS (solid) | MACDA (dashed) | OpenMARS (dotted)",
#             fontsize=12)

#P_COLORS = {100.0:"#333333", 50.0:"#888888", 1.0:"#BBBBBB"}
#P_LW     = {100.0:1.5,       50.0:1.2,       1.0:0.8}

P_COLORS = {100.0:"#333333", 50.0:"#888888"}
P_LW     = {100.0:1.5,       50.0:1.0}

for col, my in enumerate(YEARS):
    print(f"\nMY{my}...")

    ds_o = d24  if my<=27 else d28
    my_o = my_24 if my<=27 else my_28
    ls_o = ls_24 if my<=27 else ls_28

    me = (my_e==my) & np.isfinite(ls_e)
    mm = (my_m==my) & np.isfinite(ls_m)
    mo = (my_o==my) & np.isfinite(ls_o)

    for row, var in enumerate(VARS):
        ax = axes[row, col]

        for p in TARGET_P:
            le = levs_e[p]; lm = levs_m[p]; lo = levs_o[p]
            pc = P_COLORS[p]; lw = P_LW[p]
            plabel = f"{p:.0f} Pa"

            print(f"  {var} @ {p:.0f}Pa...")

            # EMARS
            if me.sum()>0 and my<=33:
                if var=="T":
                    v = load_retry(
                        de_m["anal_mean_T"].isel(time=me,pfull=le)
                        .weighted(wts_e).mean(dim=["lat","lon"])).values
                elif var=="U":
                    v = load_retry(
                        de_m["anal_mean_U"].isel(time=me,pfull=le)
                        .weighted(wts_eu).mean(dim=["latu","lon"])).values
                elif var=="V":
                    v = load_retry(
                        de_m["anal_mean_V"].isel(time=me,pfull=le)
                        .weighted(wts_e).mean(dim=["lat","lonv"])).values
                vm = bin_ls(ls_e[me], v)
                ax.plot(LS_C, vm, color=pc, lw=lw, ls="-",
                        label=f"EMARS {plabel}"
                        if col==0 and var==VARS[0] else "")
                time.sleep(10)

            # MACDA
            if mm.sum()>0:
                vname = {"T":"temp","U":"uwind","V":"vwind"}[var]
                v = load_retry(
                    dm[vname].isel(time=mm,lev=lm)
                    .weighted(wts_m).mean(dim=["lat","lon"])).values
                vm = bin_ls(ls_m[mm], v)
                ax.plot(LS_C, vm, color=pc, lw=lw, ls="--",
                        label=f"MACDA {plabel}"
                        if col==0 and var==VARS[0] else "")
                time.sleep(10)

            # OpenMARS
            if mo.sum()>0:
                vname = {"T":"temp","U":"u","V":"v"}[var]
                v = load_retry(
                    ds_o[vname].isel(time=mo,lev=lo)
                    .weighted(wts_o).mean(dim=["lat","lon"])).values
                vm = bin_ls(ls_o[mo], v)
                ax.plot(LS_C, vm, color=pc, lw=lw, ls=":",
                        label=f"OpenMARS {plabel}"
                        if col==0 and var==VARS[0] else "")
                time.sleep(10)

        #ax.axhline(0, color="gray", lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
        ax.tick_params(labelsize=7)

        if row==0: ax.set_title(LABELS[my], fontsize=10)
        if col==0: ax.set_ylabel(VARS_L[var], fontsize=9)
        if row==len(VARS)-1: ax.set_xlabel("Ls (°)", fontsize=9)

# ── Legends ───────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Dataset style legend
axes[0,0].legend(handles=[
    Line2D([0],[0],color="k",lw=1.5,ls="-", label="EMARS"),
    Line2D([0],[0],color="k",lw=1.5,ls="--",label="MACDA"),
    Line2D([0],[0],color="k",lw=1.5,ls=":", label="OpenMARS"),
], fontsize=7, loc="upper left", framealpha=0.8)

# Pressure level legend
axes[0,1].legend(handles=[
    Line2D([0],[0],color=P_COLORS[p],lw=P_LW[p],ls="-",
           label=f"{p:.0f} Pa")
    for p in TARGET_P
], fontsize=7, loc="upper left", framealpha=0.8,
   title="Pressure level")
plt.tight_layout()
plt.savefig("arco_vertical_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved arco_vertical_comparison.png")
