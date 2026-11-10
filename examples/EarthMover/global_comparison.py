"""ARCO-Mars pressure, dust and near-surface air-temperature comparison.

Run: python global_comparison.py
The first run reads the selected years. Later runs reuse the reduced 1D data.
Set REFRESH_CACHE=True after updating the source datasets.

Temperature is interpolated linearly in log(p/ps) at TARGET_SIGMA before the
spatial mean. This is a pressure-fraction benchmark, not a fixed height or T2m.
Temperature has separate caches, so existing pressure/dust caches are reusable.
"""

from functools import lru_cache
from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


# Configuration: compare the storm years only.
ALL_YEARS = [25, 28, 32]
STORMS = {25: "GDS", 28: "GDS", 32: "GDS"}
EMARS_BACKGROUND = Path("")
CACHE_DIR = Path(__file__).resolve().parent / "global_1d_cache"
OUTPUT = Path("arco_peryear.png")
REFERENCE_PRESSURE = 610.0  # Pa
TARGET_SIGMA = 0.99  # Common near-surface pressure fraction p/ps, not metres.
REFRESH_CACHE = False
LATITUDE_WEIGHTED = False  # Same spatial means as the supplied script.
                           # Set True for cosine-latitude weighted means.
DASK_WORKERS = 4
CACHE_VERSION = "v1"

LS_E = np.arange(0, 361, 5)
LS_C = (LS_E[:-1] + LS_E[1:]) / 2


def load_retry(da, retries=5, wait=360):
    for attempt in range(retries):
        try:
            return da.load(num_workers=DASK_WORKERS)
        except Exception as exc:
            if "429" not in str(exc) or attempt == retries - 1:
                raise
            print(f"429: retrying in {wait}s...", flush=True)
            time.sleep(wait)


@lru_cache(maxsize=1)
def lake_client():
    # Arraylake is only contacted when a yearly cache is missing.
    from arraylake import Client
    return Client()


@lru_cache(maxsize=None)
def open_lake(repo, group=""):
    print(f"Opening {repo} / {group or '(root)'}...", flush=True)
    store = lake_client().get_repo(repo).readonly_session("main").store
    return xr.open_zarr(
        store, group=group, consolidated=False, decode_times=False,
        # Zero is a valid hybrid coefficient. Keep masking/scaling enabled for
        # meteorological fields, but do not turn coefficient zeros into NaNs.
        mask_and_scale={"anal_mean_ak": False, "anal_mean_bk": False},
    )


@lru_cache(maxsize=None)
def indices(repo, group, my_name, ls_name, hour_name=None):
    ds = open_lake(repo, group)
    fields = {"my": ds[my_name], "ls": ds[ls_name]}
    if hour_name:
        fields["hour"] = ds[hour_name]
    loaded = load_retry(xr.Dataset(fields))
    arrays = {name: loaded[name].values.astype(float) for name in fields}
    for name, values in arrays.items():
        if values.ndim != 1 or len(values) != ds.sizes["time"]:
            raise ValueError(f"{repo}: {name} must be a 1D time index")
    return arrays


def year_indices(index, my):
    return np.flatnonzero((index["my"] == my) & np.isfinite(index["ls"]))


def time_indexer(idx):
    # Contiguous slices avoid a large fancy-indexing graph.
    if len(idx) and idx[-1] - idx[0] + 1 == len(idx):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def spatial_mean(da):
    if LATITUDE_WEIGHTED:
        weights = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(weights).mean(dim=["lat", "lon"])
    return da.mean(dim=["lat", "lon"])


def empty_series(*names):
    return {name: np.empty(0, dtype=float) for name in ("ls", *names)}


def cache_path(source, my):
    averaging = "coslat" if LATITUDE_WEIGHTED else "unweighted"
    return CACHE_DIR / (
        f"{source}_MY{my}_{averaging}_tau{REFERENCE_PRESSURE:g}_{CACHE_VERSION}.npz"
    )


def validate_series(data):
    length = len(data["ls"])
    for name, values in data.items():
        if values.ndim != 1 or len(values) != length:
            raise ValueError(f"Cached {name} is not aligned with its Ls index")


def cached_series(source, my, loader):
    path = cache_path(source, my)
    if path.exists() and not REFRESH_CACHE:
        with np.load(path, allow_pickle=False) as cached:
            data = {name: cached[name].copy() for name in cached.files}
        validate_series(data)
        print(f"  {source}: cache hit", flush=True)
        return data

    print(f"  {source}: computing MY{my}...", flush=True)
    started = time.perf_counter()
    data = {name: np.asarray(value) for name, value in loader(my).items()}
    validate_series(data)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=".npz", delete=False) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        f"  {source}: saved {len(data['ls'])} steps "
        f"({time.perf_counter() - started:.1f}s)", flush=True
    )
    return data


def emars_series(my):
    repo = "arco-planetary/ARCO-EMARS"
    index = indices(repo, "mean", "anal_mean_MY", "anal_mean_Ls", "anal_mean_mars_hour")
    idx = year_indices(index, my)
    if not len(idx):
        return empty_series("ps", "spread", "hour")
    mean = open_lake(repo, "mean")
    spread = open_lake(repo, "sprd")
    if mean.sizes["time"] != spread.sizes["time"]:
        raise ValueError("EMARS mean/spread time lengths differ; align them before plotting")
    selection = time_indexer(idx)
    if "time" in mean.coords and "time" in spread.coords:
        t_mean = mean["time"].isel(time=selection).values
        t_spread = spread["time"].isel(time=selection).values
        if not np.array_equal(t_mean, t_spread):
            raise ValueError("EMARS mean/spread timestamps differ; align them before plotting")
    fields = load_retry(xr.Dataset({
        "ps": spatial_mean(mean["anal_mean_ps"].isel(time=selection)),
        "spread": spatial_mean(spread["anal_sprd_ps"].isel(time=selection)),
    }))
    return {
        "ls": index["ls"][idx], "hour": index["hour"][idx],
        "ps": fields["ps"].values, "spread": fields["spread"].values,
    }


def macda_series(my):
    repo = "arco-planetary/ARCO-MACDA"
    index = indices(repo, "", "MY_Ls", "Ls")
    idx = year_indices(index, my)
    if not len(idx):
        return empty_series("ps", "dust")
    ds = open_lake(repo, "").isel(time=time_indexer(idx))
    # Reduce these together so pressure reads can be shared.
    fields = load_retry(xr.Dataset({
        "ps": spatial_mean(ds["psurf"]),
        "dust": spatial_mean(ds["coldust"] * REFERENCE_PRESSURE / ds["psurf"]),
    }))
    return {"ls": index["ls"][idx], "ps": fields["ps"].values, "dust": fields["dust"].values}


def openmars_series(my):
    repo = "arco-planetary/ARCO-OpenMARS"
    group = "my24" if my <= 27 else "my28"
    index = indices(repo, group, "MY", "Ls")
    idx = year_indices(index, my)
    if not len(idx):
        return empty_series("ps", "dust")
    ds = open_lake(repo, group).isel(time=time_indexer(idx))
    fields = load_retry(xr.Dataset({
        "ps": spatial_mean(ds["ps"]),
        "dust": spatial_mean(ds["dustcol"]),  # Already normalized in the supplied data.
    }))
    return {"ls": index["ls"][idx], "ps": fields["ps"].values, "dust": fields["dust"].values}


def sigma_bracket(sigma, source):
    """Find native indices bracketing the target, without extrapolating."""
    if not 0 < TARGET_SIGMA < 1:
        raise ValueError("TARGET_SIGMA must be between 0 and 1")
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 1:
        raise ValueError(f"{source}: pressure fractions must be one-dimensional")
    native = np.flatnonzero(np.isfinite(sigma) & (sigma > 0) & (sigma <= 1))
    native = native[np.argsort(sigma[native])]
    values = sigma[native]
    if not len(values):
        raise ValueError(f"{source}: no finite pressure fractions in (0, 1]; check coefficients")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{source}: duplicate atmospheric pressure fractions")
    exact = np.flatnonzero(values == TARGET_SIGMA)
    if len(exact):
        level = int(native[exact[0]])
        return level, level
    if not values[0] < TARGET_SIGMA < values[-1]:
        raise ValueError(
            f"{source}: p/ps={TARGET_SIGMA:g} is outside resolved levels "
            f"[{values[0]:.8g}, {values[-1]:.8g}]; no extrapolation"
        )
    upper = int(np.searchsorted(values, TARGET_SIGMA))
    lower = upper - 1
    i0, i1 = int(native[lower]), int(native[upper])
    return i0, i1


def interpolate_sigma(temperature, sigma, dim, source):
    """Select only the bracketing layers; interpolate in log pressure fraction."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 1 or len(sigma) != temperature.sizes[dim]:
        raise ValueError(f"{source}: pressure fractions must match {dim}")
    i0, i1 = sigma_bracket(sigma, source)
    if i0 == i1:
        print(f"    {source}: native {dim} index {i0}, p/ps={TARGET_SIGMA:g}", flush=True)
        return temperature.isel({dim: i0}, drop=True)
    s0, s1 = sigma[i0], sigma[i1]
    weight = (np.log(TARGET_SIGMA) - np.log(s0)) / (np.log(s1) - np.log(s0))
    print(
        f"    {source}: {dim} indices {i0}/{i1}, "
        f"p/ps={s0:.8g}/{s1:.8g} -> {TARGET_SIGMA:g}", flush=True
    )
    # Scalar vertical coordinates are dropped so xarray cannot misalign them.
    t0 = temperature.isel({dim: i0}, drop=True)
    t1 = temperature.isel({dim: i1}, drop=True)
    return (1 - weight) * t0 + weight * t1


def emars_temperature_at_sigma(ds, temperature, dim, surface_pressure):
    """Interpolate using actual hybrid pressure fractions in every column.

    EMARS: p_half = ak + bk*ps; p_full = (p2-p1)/log(p2/p1).
    Native EMARS ak is nonzero even just above the surface, so bk alone cannot
    supply exact full-level p/ps. Read the year's 2D pressure once, use its range
    to identify all needed layers, then evaluate weights at each local ps.
    See Greybush et al. (2019), doi:10.1002/gdj3.77, Section 3.3.
    """
    nlevels = temperature.sizes[dim]
    ak = np.asarray(ds["anal_mean_ak"].values, dtype=float)
    bk = np.asarray(ds["anal_mean_bk"].values, dtype=float)
    if ak.ndim != 1 or bk.shape != ak.shape or len(ak) != nlevels + 1:
        raise ValueError("EMARS: ak/bk must contain one more interface than T levels")
    if not np.all(np.isfinite(ak)) or not np.all(np.isfinite(bk)):
        raise ValueError("EMARS: missing ak/bk coefficients; check their raw values and fill metadata")
    if np.any(ak < 0) or np.any(bk < 0):
        raise ValueError("EMARS: expected nonnegative native ak/bk coefficients")

    ps = load_retry(surface_pressure.astype(float))
    valid = np.isfinite(ps.values) & (ps.values > 0)
    if not np.any(valid):
        raise ValueError("EMARS: no valid positive surface pressure")
    pmin, pmax = float(ps.values[valid].min()), float(ps.values[valid].max())

    def full_sigma_at(pressure):
        half = ak + bk * pressure
        difference = np.diff(half)
        if not (np.all(difference > 0) or np.all(difference < 0)):
            raise ValueError(f"EMARS: nonmonotonic interface pressures at ps={pressure:g} Pa")
        with np.errstate(divide="ignore", invalid="ignore"):
            return difference / np.log1p(difference / half[:-1]) / pressure

    # For nonnegative ak/bk, each full-level fraction decreases as ps increases.
    # Bracketing at both pressure extremes covers every valid column in the MY.
    smin, smax = full_sigma_at(pmin), full_sigma_at(pmax)
    endpoints = set(sigma_bracket(smin, "emars") + sigma_bracket(smax, "emars"))
    order = np.argsort(full_sigma_at((pmin + pmax) / 2))
    positions = [int(np.flatnonzero(order == level)[0]) for level in endpoints]
    levels = order[min(positions):max(positions) + 1]
    print(
        f"    emars: ps={pmin:.1f}-{pmax:.1f} Pa; native {dim} indices "
        f"{levels.tolist()} -> p/ps={TARGET_SIGMA:g} (full hybrid pressures)", flush=True
    )

    # Preserve chunked arithmetic after this bounded, one-year pressure read.
    if temperature.chunks is not None:
        ps = ps.chunk({name: temperature.chunksizes[name] for name in ps.dims})
    ps = ps.where(np.isfinite(ps) & (ps > 0))

    def fraction(level):
        lower = ak[level] + bk[level] * ps
        upper = ak[level + 1] + bk[level + 1] * ps
        return (upper - lower) / np.log1p((upper - lower) / lower) / ps

    fractions = [fraction(int(level)) for level in levels]
    layers = [temperature.isel({dim: int(level)}, drop=True) for level in levels]
    field = xr.full_like(ps, np.nan)
    for s0, s1, t0, t1 in zip(fractions[:-1], fractions[1:], layers[:-1], layers[1:]):
        bracket = (s0 < TARGET_SIGMA) & (TARGET_SIGMA < s1)
        weight = (np.log(TARGET_SIGMA) - np.log(s0)) / (np.log(s1) - np.log(s0))
        field = xr.where(bracket, (1 - weight) * t0 + weight * t1, field)
    for sigma, layer in zip(fractions, layers):
        field = xr.where(sigma == TARGET_SIGMA, layer, field)
    return field


def temperature_series(source, my):
    """Read the requested MY and two air-temperature layers from Arraylake."""
    specs = {
        "emars": ("arco-planetary/ARCO-EMARS", "mean", "anal_mean_MY", "anal_mean_Ls", "anal_mean_T"),
        "macda": ("arco-planetary/ARCO-MACDA", "", "MY_Ls", "Ls", "temp"),
        "openmars": ("arco-planetary/ARCO-OpenMARS", "my24" if my <= 27 else "my28", "MY", "Ls", "temp"),
    }
    repo, group, my_name, ls_name, t_name = specs[source]
    hour_name = "anal_mean_mars_hour" if source == "emars" else None
    index = indices(repo, group, my_name, ls_name, hour_name)
    idx = year_indices(index, my)
    if not len(idx):
        print(f"    {source}: no MY{my} coverage", flush=True)
        return empty_series("tair", "coverage")
    ds = open_lake(repo, group)
    temperature = ds[t_name].isel(time=time_indexer(idx))
    if source == "emars":
        vertical = [dim for dim in temperature.dims if dim not in ("time", "lat", "lon")]
        if len(vertical) != 1:
            raise ValueError(f"EMARS: unexpected temperature dimensions {temperature.dims}")
        dim = vertical[0]
        field = emars_temperature_at_sigma(
            ds, temperature, dim, ds["anal_mean_ps"].isel(time=time_indexer(idx))
        )
    else:
        dim, sigma = "lev", ds["lev"].values
        field = interpolate_sigma(temperature, sigma, dim, source)
    fields = load_retry(xr.Dataset({
        "tair": spatial_mean(field),
        "coverage": spatial_mean(field.notnull().astype(float)),
    }))
    tair, coverage = fields["tair"].values, fields["coverage"].values
    if not np.any(np.isfinite(tair)):
        raise ValueError(f"{source} MY{my}: no valid interpolated temperature; check native grid/levels")
    print(
        f"    {source}: spatial-mean T {np.nanmin(tair):.1f}-{np.nanmax(tair):.1f} K; "
        f"minimum spatial coverage {100 * np.nanmin(coverage):.1f}%", flush=True
    )
    return {"ls": index["ls"][idx], "tair": tair, "coverage": coverage}


def cached_temperature(source, my):
    # A new target/method gets its own temperature cache without rereading dust.
    method = "hybrid_logp_v2" if source == "emars" else "logp_v1"
    tag = f"{source}_tair_sigma{float(TARGET_SIGMA)}_{method}"
    return cached_series(tag, my, lambda year: temperature_series(source, year))


def background_files(my):
    if not EMARS_BACKGROUND.is_dir():
        raise FileNotFoundError(f"EMARS background directory not found: {EMARS_BACKGROUND}")
    prefix = "emars_v1.0_back_mean"
    files = set(EMARS_BACKGROUND.glob(f"{prefix}_MY{my}_Ls*.nc"))

    def first_ls(path):
        return float(path.stem.rsplit("_Ls", 1)[1].split("-", 1)[0])

    # Check adjacent boundary files for samples whose native MY crosses a
    # filename boundary; select records by their MY variable below.
    for year, choose in [(my - 1, max), (my + 1, min)]:
        adjacent = list(EMARS_BACKGROUND.glob(f"{prefix}_MY{year}_Ls*.nc"))
        if adjacent:
            files.add(choose(adjacent, key=first_ls))
    return sorted(files)


def background_series(my):
    files = background_files(my)
    if not files:
        if next(EMARS_BACKGROUND.glob("emars_v1.0_back_mean_MY*_Ls*.nc"), None) is None:
            raise FileNotFoundError(f"No EMARS background files found in {EMARS_BACKGROUND}")
        return empty_series("dust")
    print(f"Reading {len(files)} EMARS background files for MY{my} (including boundary checks)...", flush=True)
    ls_parts, dust_parts = [], []
    for number, path in enumerate(files, start=1):
        print(f"    [{number}/{len(files)}] {path.name}", flush=True)
        started = time.perf_counter()
        # No multi-file concatenation or Dask graph for unused 3D variables.
        # One file is open at a time; only MY, Ls, dod and ps are accessed.
        with xr.open_dataset(path, decode_times=False, chunks=None, cache=False) as ds:
            native_my = np.asarray(ds["MY"].values, dtype=float)
            ls = np.asarray(ds["Ls"].values, dtype=float)
            if native_my.ndim == 0:
                native_my = np.full(ds.sizes["time"], native_my.item())
            if native_my.shape != (ds.sizes["time"],) or ls.shape != native_my.shape:
                raise ValueError(f"{path.name}: MY/Ls must match the time dimension")
            idx = np.flatnonzero((native_my == my) & np.isfinite(ls))
            if not len(idx):
                print(f"      no MY{my} samples; skipped", flush=True)
                continue
            selected = ds[["dod", "ps"]].isel(time=time_indexer(idx))
            # Normalize each grid point/time before the spatial reduction.
            dust = spatial_mean(
                selected["dod"] * REFERENCE_PRESSURE / selected["ps"]
            ).values
            ls_parts.append(ls[idx].copy())
            dust_parts.append(dust.copy())
        print(f"      {len(idx)} steps ({time.perf_counter() - started:.1f}s)", flush=True)
    if not ls_parts:
        return empty_series("dust")
    return {"ls": np.concatenate(ls_parts), "dust": np.concatenate(dust_parts)}


def bin_ls(ls, val, hr=None):
    means = np.full(len(LS_C), np.nan)
    spreads = np.full(len(LS_C), np.nan)
    for i, (lo, hi) in enumerate(zip(LS_E[:-1], LS_E[1:])):
        keep = (ls >= lo) & (ls < hi) & np.isfinite(val)
        if keep.sum() < 2:
            continue
        means[i] = val[keep].mean()
        if hr is None:
            spreads[i] = val[keep].std()
        else:
            hourly = np.array([
                val[keep & (hr == hour)].mean()
                for hour in range(24) if np.any(keep & (hr == hour))
            ])
            spreads[i] = hourly.std() if len(hourly) > 3 else np.nan
    return means, spreads


def main():
    fig, axes = plt.subplots(
        3, len(ALL_YEARS), figsize=(3.2 * len(ALL_YEARS), 10),
        sharex=True, sharey="row", squeeze=False
    )
    #fig.suptitle(
    #    f"ARCO-Mars - Surface Pressure, Dust OD & Air Temperature (p/ps={TARGET_SIGMA:g})\n"
    #    "EMARS (blue+shading), MACDA (red dashed), OpenMARS (green dotted)", fontsize=11
    #)
    rmse = {}
    for col, my in enumerate(ALL_YEARS):
        print(f"MY{my}...", flush=True)
        e = cached_series("emars", my, emars_series)
        m = cached_series("macda", my, macda_series)
        o = cached_series("openmars", my, openmars_series)
        b = cached_series("emars_background", my, background_series)
        ax_ps, ax_dust, ax_t = axes[:, col]

        if len(e["ls"]):
            pressure, diurnal = bin_ls(e["ls"], e["ps"], e["hour"])
            spread = bin_ls(e["ls"], e["spread"])[0]
            ax_ps.plot(LS_C, pressure, color="steelblue", lw=1, label="EMARS")
            #ax_ps.fill_between(LS_C, pressure-diurnal, pressure+diurnal,
            #                   color="steelblue", alpha=0.1, label="diurnal +/-1 sigma")
            #ax_ps.fill_between(LS_C, pressure-spread, pressure+spread,
            #                   color="steelblue", alpha=0.2, label="mean pointwise ensemble spread")

        for data, label, color, style in [
            (m, "MACDA", "tomato", "--"), (o, "OpenMARS", "green", ":")
        ]:
            if len(data["ls"]):
                pressure = bin_ls(data["ls"], data["ps"])[0]
                dust = bin_ls(data["ls"], data["dust"])[0]
                ax_ps.plot(LS_C, pressure, color=color, lw=1, ls=style, label=label)
                ax_dust.plot(LS_C, dust, color=color, lw=1, ls=style, label=label)

        if len(b["ls"]):
            dust = bin_ls(b["ls"], b["dust"])[0]
            ax_dust.plot(LS_C, dust, color="steelblue", lw=1, label="EMARS background")

        for source, label, color, style in [
            ("emars", "EMARS analysis", "steelblue", "-"),
            ("macda", "MACDA", "tomato", "--"),
            ("openmars", "OpenMARS", "green", ":"),
        ]:
            data = cached_temperature(source, my)
            if len(data["ls"]):
                tair = bin_ls(data["ls"], data["tair"])[0]
                ax_t.plot(LS_C, tair, color=color, lw=1, ls=style, label=label)
            elif source == "emars":
                ax_t.text(0.02, 0.03, "EMARS unavailable", transform=ax_t.transAxes, fontsize=6)

        # Retain the supplied timestep interpolation RMSE calculation.
        if len(m["ls"]) > 1 and len(o["ls"]) > 1:
            v1 = interp1d(m["ls"], m["ps"], bounds_error=False, fill_value=np.nan)(LS_C)
            v2 = interp1d(o["ls"], o["ps"], bounds_error=False, fill_value=np.nan)(LS_C)
            difference = v1 - v2
            valid = np.isfinite(difference)
            if np.any(valid):
                r = np.sqrt(np.mean(difference[valid] ** 2))
                rmse[my] = r
                ax_ps.text(0.97, 0.97, f"RMSE\n{r:.1f}Pa", transform=ax_ps.transAxes,
                           ha="right", va="top", fontsize=6,
                           bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        ax_ps.set_title(f"MY{my}" , fontsize=7)
        ax_ps.grid(True, alpha=0.3)
        ax_dust.grid(True, alpha=0.3)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_xlabel("Ls (deg)", fontsize=7)
        ax_dust.set_yscale("log")
        ax_dust.set_ylim(0.01, 5)
        ax_ps.tick_params(labelsize=6)
        ax_dust.tick_params(labelsize=6)
        ax_t.tick_params(labelsize=6)
        if col == 0:
            ax_ps.set_ylabel("P$_{s}$ (Pa)", fontsize=8)
            ax_dust.set_ylabel(r"$\tau$ (610 Pa)", fontsize=8)
            ax_t.set_ylabel("T$_{a}$ (K) \n P/P$_{s}$ = 0.99", fontsize=8)
            ax_ps.legend(fontsize=5, loc="upper left")
            ax_dust.legend(fontsize=5, loc="upper left")
            ax_t.legend(fontsize=5, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT}", flush=True)
    print("\nRMSE MACDA vs OpenMARS ps:")
    for my, value in rmse.items():
        print(f"  MY{my}: {value:.1f} Pa")
    plt.close(fig)


if __name__ == "__main__":
    main()
