
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

MY=30; LS_VALUES=[0,90,180,270]; LS_HALF=15.


def check_sigma_model_top(ds, lev_name, ps_name):
    lev = ds[lev_name].values.astype(float)
    ps  = ds[ps_name]

    sigma_top = np.nanmin(lev)

    ps_min  = float(ps.min().compute())
    ps_mean = float(ps.mean().compute())
    ps_max  = float(ps.max().compute())

    print(f"sigma_top = {sigma_top:.6g}")
    print(f"ps range  = {ps_min:.2f} → {ps_max:.2f} Pa")
    print(f"model top pressure:")
    print(f"  minimum = {sigma_top * ps_min:.5f} Pa")
    print(f"  mean    = {sigma_top * ps_mean:.5f} Pa")
    print(f"  maximum = {sigma_top * ps_max:.5f} Pa")

import fsspec, time
from huggingface_hub import HfFileSystem
from huggingface_hub.errors import HfHubHTTPError

# ── Patch HfFileSystem at chunk level ────────────────────────
import huggingface_hub.hf_file_system as _hffs
if not getattr(_hffs.HfFileSystem, '_patched', False):
    _orig_cat = _hffs.HfFileSystem.cat_file
    def _cat_retry(self, path, *a, **kw):
        for attempt in range(10):
            try:
                return _orig_cat(self, path, *a, **kw)
            except Exception as e:
                if "429" in str(e):
                    print(f"\n[429] Rate limited — "
                          f"waiting 310s "
                          f"(attempt {attempt+1}/10)...",
                          flush=True)
                    time.sleep(310)
                else:
                    raise
        return _orig_cat(self, path, *a, **kw)
    _hffs.HfFileSystem.cat_file = _cat_retry
    _hffs.HfFileSystem._patched = True

# ── Open ──────────────────────────────────────────────────────
def open_hf(repo, zarr):
    fs = HfFileSystem()
    return xr.open_zarr(fs.get_mapper(
        f"datasets/{repo}/{zarr}"),
        consolidated=False,
        decode_times=False)

def load_retry(da, retries=5, wait=360):
    for i in range(retries):
        try:
            return da.load()
        except Exception as e:
            if "429" in str(e) and i < retries-1:
                print(f"    429 — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

# ── Open all datasets ─────────────────────────────────────────
de = open_hf("ananyo01/ARCO-EMARS",
                   "emars_combined.zarr")
de_sprd = open_hf("ananyo01/ARCO-EMARS",
                   "emars_sprd.zarr")
dm      = open_hf("ananyo01/ARCO-MACDA",
                   "macda_combined.zarr")
d24    = open_hf("ananyo01/ARCO-OpenMars",
                   "openmars_MY24-27.zarr")
d28    = open_hf("ananyo01/ARCO-OpenMars",
                   "openmars_MY28-35.zarr")


my_m=dm["MY_Ls"].values.astype(float); ls_m=dm["Ls"].values.astype(float)
my_e=de["anal_mean_MY"].values.astype(float); ls_e=de["anal_mean_Ls"].values.astype(float)
my_o=d28["MY"].values.astype(float); ls_o=d28["Ls"].values.astype(float)

lat_m=dm["lat"].values.astype(float)
lat_e=de["latu"].values.astype(float)  # V uses latu
lat_o=d28["lat"].values.astype(float)
lev_m=dm["lev"].values.astype(float)
lev_o=d28["lev"].values.astype(float)
ak=de["anal_mean_ak"].values.astype(float)
bk=de["anal_mean_bk"].values.astype(float)

#P_OUT = np.logspace(np.log10(1000), np.log10(0.01), 60)
P_OUT = np.logspace(np.log10(700),
                     np.log10(0.1), 60)
# All three datasets reach 0.05 Pa

def to_P(V_z, P_z, P_out):
    out=np.full((len(P_out),V_z.shape[1]),np.nan)
    for i in range(V_z.shape[1]):
        p=P_z[:,i]; f=V_z[:,i]; v=np.isfinite(f)&(p>0)
        if v.sum()<2: continue
        s=np.argsort(p[v])[::-1]
        out[:,i]=interp1d(np.log(p[v][s]),f[v][s],
                           bounds_error=False,
                           fill_value=np.nan)(np.log(P_out))
    return out

def get_mask(my_arr, ls_arr, ls_c):
    ls_min=ls_c-LS_HALF; ls_max=ls_c+LS_HALF
    if ls_min<0:
        return ((my_arr==MY)&
                ((ls_arr>=ls_min+360)|(ls_arr<=ls_max))&
                np.isfinite(ls_arr))
    return ((my_arr==MY)&(ls_arr>=ls_min)&
             (ls_arr<=ls_max)&np.isfinite(ls_arr))

def get_V(ds, V_var, ps_var, my_arr, ls_arr,
           lev, lat, ds_type, ls_c):
    mask = get_mask(my_arr, ls_arr, ls_c)
    if mask.sum()==0: return None
    da      = ds[V_var]
    lon_dim = next(d for d in da.dims if "lon" in d)
    V = da.isel(time=mask).mean(
        dim=[lon_dim,"time"]).compute().values
    nlev=(len(lev) if lev is not None else len(ak)-1)
    nlat=len(lat)
    if V.shape==(nlat,nlev): V=V.T
    ps_mean=float(ds[ps_var].isel(
        time=mask).mean().compute())
    if ds_type=="sigma":
        P_z=lev[:,None]*ps_mean*np.ones((nlev,nlat))
    else:
        Ph=ak[:,None]+bk[:,None]*ps_mean
        P_z=np.tile((Ph[:-1]+Ph[1:])/2,(1,nlat))
        P_z=P_z[:,:nlat]
    return to_P(V, P_z, P_OUT)

# ── Compute ───────────────────────────────────────────────────
configs=[
    ("EMARS",   de, my_e,ls_e,None, lat_e,
     "anal_mean_V","anal_mean_ps","hybrid"),
    ("OpenMARS",d28,my_o,ls_o,lev_o,lat_o,
     "v","ps","sigma"),
    ("MACDA",   dm, my_m,ls_m,lev_m,lat_m,
     "vwind","psurf","sigma"),
]

print("Computing...")
results={}
for ds_name,ds,my_arr,ls_arr,lev,lat,V_var,ps_var,dst in configs:
    print(f"  {ds_name}")
    panels=[]
    for ls_c in LS_VALUES:
        V=get_V(ds,V_var,ps_var,my_arr,ls_arr,
                 lev,lat,dst,ls_c)
        n=get_mask(my_arr,ls_arr,ls_c).sum()
        print(f"    Ls={ls_c:3.0f}°: {n} steps "
              f"V={np.nanmin(V):.1f}→{np.nanmax(V):.1f}"
              if V is not None else
              f"    Ls={ls_c:3.0f}°: no data")
        panels.append(V)
    results[ds_name]={"panels":panels,"lat":lat}

# ── Plot ──────────────────────────────────────────────────────
fig,axes=plt.subplots(4,3,figsize=(20,27),
                       sharex=True,sharey=True)
vmax = 50.
levels = np.arange(-50, 51, 5)

print("MACDA")
check_sigma_model_top(dm, "lev", "psurf")

print("\nOpenMARS")
check_sigma_model_top(d28, "lev", "ps")

norm = mcolors.TwoSlopeNorm(
    vmin=-vmax,
    vcenter=0,
    vmax=vmax
)

cmap = "RdBu_r"

col_order=["EMARS","OpenMARS","MACDA"]
col_titles=["EMARS","OpenMARS","MACDA"]
ls_labels=["Ls=0°","Ls=90°","Ls=180°","Ls=270°"]

for col,ds_name in enumerate(col_order):
    lat=results[ds_name]["lat"]
    for row,V in enumerate(results[ds_name]["panels"]):
        ax=axes[row,col]
        if V is not None and np.isfinite(V).sum()>10:
            cf=ax.contourf(lat,P_OUT,V,levels=levels,
                            norm=norm,cmap=cmap,extend="both")
            ax.contour(lat,P_OUT,V,levels=levels[levels>0],
                        colors="k",linewidths=0.5,
                        linestyles="solid")
            ax.contour(lat,P_OUT,V,levels=levels[levels<0],
                        colors="k",linewidths=0.5,
                        linestyles="dashed")
            ax.contour(lat,P_OUT,V,levels=[0],
                        colors="k",linewidths=0.8)
            ax.contour(lat, P_OUT, V, levels=[0],
              colors="k", linewidths=1.0,
              linestyles="dotted")
            cf_last=cf
        else:
            ax.set_facecolor("lightgray")
            ax.text(0.5,0.5,"no data",
                    transform=ax.transAxes,
                    ha="center",color="red")

        ax.set_yscale("log"); ax.invert_yaxis()
        ax.set_ylim(700,0.1); ax.set_xlim(-90,90)
        ax.set_xticks([-90,-60,-30,0,30,60,90])
        ax.grid(True,alpha=0.2)
        if row==0: ax.set_title(col_titles[col])
        if col==0: ax.set_ylabel("Pressure (Pa)")
        if row==3: ax.set_xlabel("Latitude (°)")
        if col==2:
            ax.text(1.04,0.5,ls_labels[row],
                    transform=ax.transAxes,
                    rotation=270,va="center")

fig.subplots_adjust(
    left=0.08,
    right=0.92,
    top=0.92,
    bottom=0.09,
    hspace=0.06,
    wspace=0.12
)

if cf_last is not None:
    cbar_ax = fig.add_axes([
        0.20,   # left
        0.025,  # bottom
        0.60,   # width
        0.015   # height
    ])

    cb = fig.colorbar(
        cf_last,
        cax=cbar_ax,
        orientation="horizontal",
        extend="both"
    )

    cb.set_label("m/s", fontsize=17)
    cb.set_ticks(np.arange(-vmax, vmax+1, 5))
    cb.ax.tick_params(labelsize=15)

plt.savefig("guzewich2026_fig4_ARCO.png",
             dpi=150,bbox_inches="tight")
print("Saved guzewich2026_fig4_ARCO.png")
