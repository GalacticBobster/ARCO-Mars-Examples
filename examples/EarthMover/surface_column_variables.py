import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from arraylake import Client
import time

client = Client()

def open_lake(repo, group):
    repo_name = client.get_repo(repo)
    session = repo_name.readonly_session("main")
    return xr.open_zarr(session.store, group = group, consolidated = False, decode_times = False)


def load_retry(da, retries=5, wait=360):
    for i in range(retries):
        try:
            return da.load()
        except Exception as e:
            if "429" in str(e) and i < retries-1:
                print(f"    429 — waiting {wait}s...")
                time.sleep(wait)
            else: raise

# ── Open ──────────────────────────────────────────────────────
print("Opening datasets...")
de_m = open_lake("arco-planetary/ARCO-EMARS",    "mean")
de_s = open_lake("arco-planetary/ARCO-EMARS",    "sprd")
dm   = open_lake("arco-planetary/ARCO-MACDA",    "")
d24  = open_lake("arco-planetary/ARCO-OpenMARS", "my24")
d28  = open_lake("arco-planetary/ARCO-OpenMARS", "my28")

# ── 1D indices ────────────────────────────────────────────────
print("Loading 1D indices...")
my_e  = load_retry(de_m["anal_mean_MY"]).astype(float);       
ls_e  = load_retry(de_m["anal_mean_Ls"]).astype(float);       
hr_e  = load_retry(de_m["anal_mean_mars_hour"]).astype(float);
my_m  = load_retry(dm["MY_Ls"]).astype(float);                
ls_m  = load_retry(dm["Ls"]).astype(float);                   
my_24 = load_retry(d24["MY"]).astype(float);                  
ls_24 = load_retry(d24["Ls"]).astype(float);                  
my_28 = load_retry(d28["MY"]).astype(float);                 
ls_28 = load_retry(d28["Ls"]).astype(float);              
print("Indices done.")

# ── EMARS level ───────────────────────────────────────────────
bk    = de_m["anal_mean_bk"].values
sfc_e = min(int(np.argmax(bk)),
            de_m["anal_mean_T"].sizes["pfull"]-1)

# ── Selected years only ───────────────────────────────────────
YEARS  = [25, 28, 32]
LABELS = {25:"MY25",
          28:"MY28",
          32:"MY32"}

# ── Ls binning ────────────────────────────────────────────────
LS_E = np.arange(0, 361, 5)
LS_C = (LS_E[:-1] + LS_E[1:]) / 2

def bin_ls(ls, val, hr=None):
    m = np.full(len(LS_C), np.nan)
    s = np.full(len(LS_C), np.nan)
    for i,(l0,l1) in enumerate(zip(LS_E[:-1],LS_E[1:])):
        idx = (ls>=l0)&(ls<l1)&np.isfinite(val)
        if idx.sum() < 2: continue
        m[i] = val[idx].mean()
        if hr is not None:
            hm = np.array([val[idx&(hr==h)].mean()
                           for h in range(24)
                           if (idx&(hr==h)).sum()>0])
            s[i] = hm.std() if len(hm)>3 else np.nan
        else:
            s[i] = val[idx].std()
    return m, s

# ── Plot setup ────────────────────────────────────────────────
fig, axes = plt.subplots(3, len(YEARS),
                          figsize=(5*len(YEARS), 11),
                          sharex=True)
fig.suptitle("ARCO-Mars — Surface Pressure, Temperature & Dust\n"
             "EMARS (blue+shading) | MACDA (red dashed) | OpenMARS (green dotted)",
             fontsize=12)

# ── Load and plot per year ────────────────────────────────────
for col, my in enumerate(YEARS):
    print(f"\nMY{my}...")
    ax_ps, ax_T, ax_dust = axes[0,col], axes[1,col], axes[2,col]
    print(de_m["lat"])
    print(dm["lat"])
    print(d24["lat"]) 
    wts_e = np.cos(np.radians(de_m["lat"]))    # EMARS lat
    wts_m = np.cos(np.radians(dm["lat"]))      # MACDA lat
    wts_o = np.cos(np.radians(d24["lat"]))     # OpenMARS lat (same for d28)
    ds_o  = d24  if my<=27 else d28
    my_o  = my_24 if my<=27 else my_28
    ls_o  = ls_24 if my<=27 else ls_28

    # masks
    me = (my_e==my) & np.isfinite(ls_e)
    mm = (my_m==my) & np.isfinite(ls_m)
    mo = (my_o==my) & np.isfinite(ls_o)

    # ── Load 2D: ps ───────────────────────────────────────────
    print("  ps...")
    if me.sum()>0:
        ps_e = load_retry(de_m["anal_mean_ps"].isel(time=me).weighted(wts_e)
                           .mean(dim=["lat","lon"])).values
        ps_s = load_retry(de_s["anal_sprd_ps"].isel(time=me).weighted(wts_e)
                           .mean(dim=["lat","lon"])).values
        time.sleep(15)
    if mm.sum()>0:
        ps_m = load_retry(dm["psurf"].isel(time=mm).weighted(wts_m)
                           .mean(dim=["lat","lon"])).values
        time.sleep(15)
    if mo.sum()>0:
        ps_o = load_retry(ds_o["ps"].isel(time=mo).weighted(wts_o)
                           .mean(dim=["lat","lon"])).values
        time.sleep(15)

    # ── Load 3D: T (single level) ─────────────────────────────
    print("  T...")
    if me.sum()>0:
        T_e = load_retry(de_m["anal_mean_T"].isel(time=me,pfull=sfc_e).weighted(wts_e)
                          .mean(dim=["lat","lon"])).values
        T_s = load_retry(de_s["anal_sprd_T"].isel(time=me,pfull=sfc_e).weighted(wts_e)
                          .mean(dim=["lat","lon"])).values
        time.sleep(15)
    if mm.sum()>0:
        T_m = load_retry(dm["tsurf"].isel(time=mm).weighted(wts_m)
                          .mean(dim=["lat","lon"])).values
        time.sleep(15)
    if mo.sum()>0:
        T_o = load_retry(ds_o["tsurf"].isel(time=mo).weighted(wts_o)
                          .mean(dim=["lat","lon"])).values
        time.sleep(15)

    # ── Load 2D: dust ─────────────────────────────────────────
    print("  dust...")
    if mm.sum()>0:
        du_m = load_retry((dm["coldust"]*610/dm["psurf"]).isel(time=mm).weighted(wts_m)
                           .mean(dim=["lat","lon"])).values
        time.sleep(15)
    if mo.sum()>0:
        du_o = load_retry(ds_o["dustcol"].isel(time=mo).weighted(wts_o)
                           .mean(dim=["lat","lon"])).values
        time.sleep(30)

    # ── Bin and plot ──────────────────────────────────────────
    def plot_var(ax, ls_e_, val_e, val_s,
                     ls_m_, val_m,
                     ls_o_, val_o,
                     hr_e_=None, logy=False):
        # EMARS
        if me.sum()>0:
            vm,vd = bin_ls(ls_e_, val_e, hr_e_)
            vs    = bin_ls(ls_e_, val_s)[0]
            ax.plot(LS_C,vm,color="steelblue",lw=1.2,label="EMARS")
            ax.fill_between(LS_C,vm-vd,vm+vd,
                            color="steelblue",alpha=0.1,
                            label="diurnal ±1σ")
            ax.fill_between(LS_C,vm-vs,vm+vs,
                            color="steelblue",alpha=0.2,
                            label="ensemble ±1σ")
        # MACDA
        if mm.sum()>0:
            vm2,_ = bin_ls(ls_m_, val_m)
            ax.plot(LS_C,vm2,color="tomato",lw=1.0,
                    ls="--",label="MACDA")
        # OpenMARS
        if mo.sum()>0:
            vo,_ = bin_ls(ls_o_, val_o)
            ax.plot(LS_C,vo,color="green",lw=1.0,
                    ls=":",label="OpenMARS")
        ax.grid(True,alpha=0.3)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.01,5)

    if me.sum()>0 and mm.sum()>0 and mo.sum()>0:
        plot_var(ax_ps,
                 ls_e[me], ps_e, ps_s,
                 ls_m[mm], ps_m,
                 ls_o[mo], ps_o,
                 hr_e_=hr_e[me])
        plot_var(ax_T,
                 ls_e[me], T_e, T_s,
                 ls_m[mm], T_m,
                 ls_o[mo], T_o,
                 hr_e_=hr_e[me])

    if mm.sum()>0 and mo.sum()>0:
        vm2,_ = bin_ls(ls_m[mm], du_m)
        vo,_  = bin_ls(ls_o[mo], du_o)
        ax_dust.plot(LS_C,vm2,color="tomato",lw=1.0,ls="--",label="MACDA")
        ax_dust.plot(LS_C,vo, color="green", lw=1.0,ls=":", label="OpenMARS")
        ax_dust.grid(True,alpha=0.3)
        ax_dust.set_yscale("log")
        ax_dust.set_ylim(0.01,5)

    # ── Labels ────────────────────────────────────────────────
    ax_ps.set_title(LABELS[my], fontsize=10)
    ax_dust.set_xlabel("Solar Longitude Ls (°)", fontsize=9)
    if col==0:
        ax_ps.set_ylabel("Surface Pressure (Pa)",  fontsize=9)
        ax_T.set_ylabel("Near-surface T (K)",      fontsize=9)
        ax_dust.set_ylabel("Dust Optical Depth",  fontsize=9)
        ax_ps.legend(fontsize=7, loc="upper left",
                     framealpha=0.8)
        ax_dust.legend(fontsize=7, loc="upper left")

plt.tight_layout()
plt.savefig("arco_3years.png", dpi=150, bbox_inches="tight")
print("\nSaved arco_3years.png")
