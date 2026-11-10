from arraylake import Client
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.interpolate import interp1d
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})


# ── Config ────────────────────────────────────────────────────
EMARS_NC  = Path("/nuke/ananyo/emars_back_v2/back_mean/"
                  "emars_v1.0_back_mean_MY28_Ls240-270.nc")
MY=28; LS_TARGET=280.0
LAT_EQ=(-10,10); LAT_POL=(70,90)

# ── EMARS NC ──────────────────────────────────────────────────
print("Loading EMARS...")
ds   = xr.open_dataset(EMARS_NC, decode_times=False)
ip_e = int(np.argmin(np.abs(
    ds["Ls"].values.astype(float)-LS_TARGET)))
o1=ds["o1"].values[ip_e]; o2=ds["o2"].values[ip_e]
o3=ds["o3"].values[ip_e]; ps_e=ds["ps"].values[ip_e]
dod_e=ds["dod"].values[ip_e]; t_e=ds["t"].values[ip_e]
dod_e = dod_e * (610.0 / ps_e)
ak=ds["ak"].values; bk=ds["bk"].values
lat_e=ds["lat"].values; lon_e=ds["lon"].values
ds.close()
dust_e=o1+o2+o3
Ph=ak[:,None,None]+bk[:,None,None]*ps_e[None]
P_e=(Ph[:-1]+Ph[1:])/2

def fix_lon(lon,*fields):
    fix=lon.copy(); fix[fix>180]-=360
    s=np.argsort(fix)
    return (fix[s],)+tuple(f[...,s] for f in fields)

lon_e,dod_e,dust_e,t_e,P_e=fix_lon(
    lon_e,dod_e,dust_e,t_e,P_e)

# ── MACDA + OpenMARS from Earthmover ──────────────────────────
print("Loading MACDA & OpenMARS...")
client=Client()
dm=xr.open_zarr(
    client.get_repo("arco-planetary/ARCO-MACDA")
    .readonly_session("main").store,
    consolidated=False,decode_times=False)
d28=xr.open_zarr(
    client.get_repo("arco-planetary/ARCO-OpenMARS")
    .readonly_session("main").store,
    group="my28",consolidated=False,
    decode_times=False)

def get_snap(ds,my_arr,ls_arr,my_t,ls_t,vars_):
    mask=(my_arr==my_t)&np.isfinite(ls_arr)
    ip=np.where(mask)[0][
        np.argmin(np.abs(ls_arr[mask]-ls_t))]
    print(f"  Ls={ls_arr[ip]:.1f}°")
    return {v: ds[v].isel(time=ip).compute().values
            for v in vars_}

my_m=dm["MY_Ls"].values.astype(float)
ls_m=dm["Ls"].values.astype(float)
my_o=d28["MY"].values.astype(float)
ls_o=d28["Ls"].values.astype(float)

vm=get_snap(dm,my_m,ls_m,MY,LS_TARGET,
             ["dustmmr","coldust","psurf","temp"])
vo=get_snap(d28,my_o,ls_o,MY,LS_TARGET,
             ["dustcol","ps","temp"])

lat_m=dm["lat"].values; lon_m=dm["lon"].values
lat_o=d28["lat"].values; lon_o=d28["lon"].values
lev_m=dm["lev"].values; lev_o=d28["lev"].values

P_m=lev_m[:,None,None]*vm["psurf"][None]
P_o=lev_o[:,None,None]*vo["ps"][None]
tau_m=vm["coldust"]*610/vm["psurf"]
tau_o=vo["dustcol"]   # already normalized

lon_m,tau_m,vm["dustmmr"],vm["temp"],P_m=fix_lon(
    lon_m,tau_m,vm["dustmmr"],vm["temp"],P_m)
lon_o,tau_o,vo["temp"],P_o=fix_lon(
    lon_o,tau_o,vo["temp"],P_o)

# ── Common P grid ─────────────────────────────────────────────
P_hpa=np.logspace(np.log10(6),np.log10(0.01),50)

def to_P(field,P_in,P_out):
    nlev,nlat=field.shape
    out=np.full((len(P_out),nlat),np.nan)
    for i in range(nlat):
        p=P_in[:,i]; f=field[:,i]
        v=np.isfinite(f)&(p>0)
        if v.sum()<2: continue
        s=np.argsort(p[v])
        out[:,i]=interp1d(
            np.log(p[v][s]),f[v][s],
            bounds_error=False,
            fill_value=np.nan)(np.log(P_out))
    return out

def lat_avg(T,lat,lmin,lmax):
    idx=(lat>=lmin)&(lat<=lmax)
    w=np.cos(np.radians(lat[idx]))
    return np.average(T[:,idx],weights=w,axis=1)

# Dust zonal means
dm_d=to_P(vm["dustmmr"].mean(axis=2),
           P_m.mean(axis=2)/100*100,P_hpa*100)
de_d=to_P(dust_e.mean(axis=2),
           P_e.mean(axis=2)/100*100,P_hpa*100)

# T profiles all three
dm_T=to_P(vm["temp"].mean(axis=2),
           P_m.mean(axis=2)/100*100,P_hpa*100)
de_T=to_P(t_e.mean(axis=2),
           P_e.mean(axis=2)/100*100,P_hpa*100)
do_T=to_P(vo["temp"].mean(axis=2),
           P_o.mean(axis=2)/100*100,P_hpa*100)

Tm_eq =lat_avg(dm_T,lat_m,*LAT_EQ)
Tm_pol=lat_avg(dm_T,lat_m,*LAT_POL)
Te_eq =lat_avg(de_T,lat_e,*LAT_EQ)
Te_pol=lat_avg(de_T,lat_e,*LAT_POL)
To_eq =lat_avg(do_T,lat_o,*LAT_EQ)
To_pol=lat_avg(do_T,lat_o,*LAT_POL)

# ── Plot — 3 rows × 3 cols ────────────────────────────────────
# Row 1: CDOD maps — MACDA | OpenMARS | EMARS
# Row 2: Dust MMR cross-section — MACDA | EMARS | (diff)
# Row 3: T profiles — equator | polar | both

fig=plt.figure(figsize=(25,19))
#fig.suptitle(
#    f"MY{MY} Ls≈{LS_TARGET:.0f}° (GDS Peak)\n"
#    "Dust Optical Depth, 3D Dust MMR & Temperature\n"
#    "ARCO-MACDA (with DA) | ARCO-OpenMARS | "
#    "ARCO-EMARS back_mean (no DA)",
#    fontsize=11)

norm_od =mcolors.Normalize(vmin=0,vmax=3)
norm_mmr=mcolors.LogNorm(vmin=1e-8,vmax=1e-4)
norm_d  =mcolors.TwoSlopeNorm(
    vmin=-5e-5,vcenter=0,vmax=5e-5)
cmap_d  ="YlOrRd"

# ── Row 1: Column OD maps ─────────────────────────────────────
for col,(lon,lat,tau,title,lbl) in enumerate([
    (lon_m,lat_m,tau_m,
     "MACDA",r"$\tau$"),
    (lon_o,lat_o,tau_o,
     "OpenMARS",r"$\tau$"),
    (lon_e,lat_e,dod_e,
     "EMARS",r"$\tau$"),
]):
    ax=fig.add_subplot(3,3,col+1)
    cf=ax.contourf(*np.meshgrid(lon,lat),tau,
                    levels=np.linspace(0,3,25),
                    norm=norm_od,cmap=cmap_d,
                    extend="max")
    plt.colorbar(cf,ax=ax,label=lbl,pad=0.02)
    ax.set_title(title)
    ax.set_ylabel("Lat (°N)")
    ax.set_xticks([-180,-90,0,90,180])
    ax.grid(True,alpha=0.2)

# ── Row 2: Dust MMR cross-sections ───────────────────────────
ax=fig.add_subplot(3,3,4)
cf=ax.contourf(lat_m,P_hpa,dm_d,
                levels=np.logspace(-8,-4,25),
                norm=norm_mmr,cmap=cmap_d,
                extend="both")
plt.colorbar(cf,ax=ax,label="Dust MMR (kg/kg)",pad=0.02)
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_ylim(6,0.01)
ax.set_ylabel("P (hPa)")
ax.set_title("MACDA Dust")
ax.grid(True,alpha=0.2,color="white")

ax=fig.add_subplot(3,3,5)
cf=ax.contourf(lat_e,P_hpa,de_d,
                levels=np.logspace(-8,-4,25),
                norm=norm_mmr,cmap=cmap_d,
                extend="both")
plt.colorbar(cf,ax=ax,label="Dust MMR (kg/kg)",pad=0.02)
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_ylim(6,0.01)
ax.set_title("EMARS Dust")
ax.grid(True,alpha=0.2,color="white")

# Difference panel
lat_c=np.linspace(-90,90,37)
def ilat(f,lat,lo): 
    out=np.full((f.shape[0],len(lo)),np.nan)
    for k in range(f.shape[0]):
        v=np.isfinite(f[k])
        if v.sum()<2: continue
        out[k]=interp1d(lat[v],f[k][v],
                         bounds_error=False,
                         fill_value=np.nan)(lo)
    return out

dm_c=ilat(dm_d,lat_m,lat_c)
de_c=ilat(de_d,lat_e,lat_c)
diff=(dm_c-de_c)
vmax_d=np.nanpercentile(np.abs(diff),97)
ax=fig.add_subplot(3,3,6)
cf=ax.contourf(lat_c,P_hpa,diff,levels=40,
                norm=mcolors.TwoSlopeNorm(
                    vmin=-vmax_d,vcenter=0,
                    vmax=vmax_d),
                cmap="RdBu_r",extend="both")
plt.colorbar(cf,ax=ax,label=r"$\Delta$ Dust (kg/kg)",
              pad=0.02)
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_ylim(6,0.01)
ax.set_title("MACDA-EMARS")
ax.grid(True,alpha=0.2)

# ── Row 3: T profiles all three ───────────────────────────────
colors={"MACDA":"tomato","EMARS":"steelblue",
         "OpenMARS":"green"}

ax=fig.add_subplot(3,3,7)
ax.plot(Tm_eq, P_hpa,"tomato",   lw=2,label="MACDA")
ax.plot(Te_eq, P_hpa,"steelblue",lw=2,label="EMARS")
ax.plot(To_eq, P_hpa,"green",    lw=2,label="OpenMARS")
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_ylim(6,0.01)
ax.set_xlabel("T (K)")
ax.set_ylabel("P (hPa)")
ax.set_title(f"Equator Temperatures\n"
              f"({LAT_EQ[0]}–{LAT_EQ[1]}°N)")
ax.legend(); ax.grid(True,alpha=0.3)

ax=fig.add_subplot(3,3,8)
ax.plot(Tm_pol,P_hpa,"tomato",   lw=2,label="MACDA")
ax.plot(Te_pol,P_hpa,"steelblue",lw=2,label="EMARS")
ax.plot(To_pol,P_hpa,"green",    lw=2,label="OpenMARS")
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_ylim(6,0.01)
ax.set_xlabel("T (K)")
ax.set_title(f"Polar Temperatures\n"
              f"({LAT_POL[0]}–{LAT_POL[1]}°N)")
ax.legend(); ax.grid(True,alpha=0.3)

ax=fig.add_subplot(3,3,9)
for T_eq,T_pol,label,c in [
    (Tm_eq,Tm_pol,"MACDA","tomato"),
    (Te_eq,Te_pol,"EMARS","steelblue"),
    (To_eq,To_pol,"OpenMARS","green"),
]:
    ax.plot(T_eq, P_hpa,c,lw=2,label=f"{label} eq")
    ax.plot(T_pol,P_hpa,c,lw=2,ls="--",
             label=f"{label} polar")
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_ylim(6,0.01)
ax.set_xlabel("T (K)")
ax.set_title("Mean Temperatures\neq (solid) vs "
              "polar (dashed)")
ax.legend(loc="best")
ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig("dust_T_all3.png",
             dpi=150,bbox_inches="tight")
print("Saved dust_T_all3.png")
