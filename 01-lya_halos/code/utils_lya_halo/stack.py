"""
stack.py -- Stage 2 (STACK).   Fast, re-run freely; reads the galaxy FITS only.

Mirrors the Line Shift notebook's two-step convention:
  1. convert_avg_fiber_bin : flux density -> requested unit. The (1+z)
     rest-density factor and the flux_unit scale are applied HERE (once), and
     the area normalization is per AVERAGE FIBER (each bin spectrum is an
     average-fiber spectrum), not per annulus.
  2. stack_spec_values     : resample to rest frame by dividing the spectral
     axis by (1+z) and interpolating. NO (1+z) flux multiplication here.

Order matters: convert first (applies (1+z)), then resample (axis only), so the
(1+z) factor is never double-counted.

Then coadd across the GALAXY axis per radial bin for EVERY method in
config.galaxy_combine_methods, in one pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import astropy.units as u
from astropy.table import vstack
from scipy.interpolate import interp1d

from .virial import cosmo                 # Planck18, single source
from .config import PipelineConfig
from .extract import combine_fibers, radial_bin_edges   # reused for galaxy-axis combine + Rvir

if TYPE_CHECKING:
    from .io import GalaxyProduct


# ---------------------------------------------------------------------
# 2.1  Unit conversion  (matches convert_avg_fiber_bin)
# ---------------------------------------------------------------------
def convert_avg_fiber_bin(fluxes, errors, z, output="L_kpc2",
                          flux_unit_scale=1e-17, fiber_diam_arcsec=1.5,
                          rest_density=True):
    """
    Convert one radial bin of average-fiber spectra. (ngal, nwave) in, same out.

      'raw'           observed flux density per average fiber
      'flux_arcsec2'  SB density per arcsec^2
      'L_fiber'       luminosity density per average fiber
      'L_kpc2'        luminosity surface density per average-fiber kpc^2

    rest_density=True multiplies flux density by (1+z) so integrating over
    rest-frame Angstrom gives the correct integrated line flux/luminosity.
    """
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    z = np.asarray(z, dtype=float)
    f = fluxes * flux_unit_scale
    e = errors * flux_unit_scale

    if output == "raw":
        return f, e, {"y_unit": "erg/s/cm^2/A/fiber"}

    fiber_area_arcsec2 = np.pi * (fiber_diam_arcsec / 2) ** 2
    kpc_per_arcsec = np.array(
        [cosmo.angular_diameter_distance(zi).to(u.kpc).value / 206265 for zi in z]
    )
    fiber_area_kpc2 = fiber_area_arcsec2 * kpc_per_arcsec ** 2
    zfac = (1 + z)[:, None] if rest_density else 1.0

    if output == "flux_arcsec2":
        y = f * zfac / fiber_area_arcsec2
        ye = e * zfac / fiber_area_arcsec2
        info = {"y_unit": "erg/s/cm^2/A_rest/arcsec^2",
                "fiber_area_arcsec2": fiber_area_arcsec2}
    elif output == "L_fiber":
        DL = cosmo.luminosity_distance(z).to(u.cm).value
        fac = 4 * np.pi * DL[:, None] ** 2 * zfac
        y = f * fac
        ye = e * fac
        info = {"y_unit": "erg/s/A_rest/fiber"}
    elif output == "L_kpc2":
        DL = cosmo.luminosity_distance(z).to(u.cm).value
        fac = 4 * np.pi * DL[:, None] ** 2 * zfac
        y = f * fac / fiber_area_kpc2[:, None]
        ye = e * fac / fiber_area_kpc2[:, None]
        info = {"y_unit": "erg/s/A_rest/kpc^2", "fiber_area_kpc2": fiber_area_kpc2}
    else:
        raise ValueError("output must be 'raw', 'flux_arcsec2', 'L_fiber', or 'L_kpc2'")
    return y, ye, info


# ---------------------------------------------------------------------
# 2.2  Rest-frame resample  (matches stack_spec_values)
# ---------------------------------------------------------------------
def rest_grid(config: "PipelineConfig") -> np.ndarray:
    """Common rest-frame wavelength grid (Angstrom) from config."""
    n = int(round((config.rest_wave_max - config.rest_wave_min) / config.rest_delta)) + 1
    return config.rest_wave_min + config.rest_delta * np.arange(n)


def stack_spec_values(fs, es, zs, wavelength, new_wavelength):
    """
    Rest-frame resample spectra ALREADY in the desired y-units.
    Spectral axis -> /(1+z); linear interpolation onto new_wavelength.
    NO flux_unit multiplication, NO (1+z) flux multiplication (done upstream).

    fs/es : (ngal, nwave). Returns (ngal, nwave_rest).
    """
    out_f = np.full((len(fs), new_wavelength.size), np.nan)
    out_e = np.full((len(es), new_wavelength.size), np.nan)
    for i in range(len(fs)):
        rest_axis = np.asarray(wavelength) / (1 + zs[i])
        good = np.isfinite(fs[i])
        if good.sum() < 2:
            continue
        fi = interp1d(rest_axis[good], fs[i][good], kind="linear",
                      bounds_error=False, fill_value=np.nan)
        ei = interp1d(rest_axis[good], es[i][good], kind="linear",
                      bounds_error=False, fill_value=np.nan)
        out_f[i] = fi(new_wavelength)
        out_e[i] = ei(new_wavelength)
    return out_f, out_e


# ---------------------------------------------------------------------
# 2.3  Galaxy-axis coadd (all methods at once)
# ---------------------------------------------------------------------
def coadd_galaxies(cube_flux, cube_err, methods, weights=None):
    """
    cube_flux/cube_err : (ngal, nrad, nwave). Combine across the galaxy axis per
    radial bin, once per method. `weights` (ngal, nrad), e.g. product.nfib, is
    used ONLY by 'weighted_median' (the per-bin column is the galaxy widths);
    every other method ignores it. Returns {method:{'flux','err'}} (nrad, nwave).
    """
    ngal, nrad, nwave = cube_flux.shape
    w = None if weights is None else np.asarray(weights, dtype=float)
    out = {}
    for method in methods:
        f = np.full((nrad, nwave), np.nan)
        e = np.full((nrad, nwave), np.nan)
        needs_w = method.lower() == "weighted_median"
        for r in range(nrad):
            w_r = w[:, r] if (needs_w and w is not None) else None
            f[r], e[r] = combine_fibers(cube_flux[:, r, :], cube_err[:, r, :],
                                        method=method, weights=w_r)
        out[method] = {"flux": f, "err": e}
    return out


# ---------------------------------------------------------------------
# 2.0  Full assembly
# ---------------------------------------------------------------------
def sample_virial_radius_kpc(config: "PipelineConfig", mass, z):
    """
    Biweight virial radius (kpc) of the sample -- the factor that converts the
    R/Rvir bin edges to kpc for plotting/reporting (VR_biweight_v).

    Virial mode only: derives each galaxy's Rvir from its per-galaxy kpc edges
    (radial_bin_edges, which already calls the virial relation) as
    edge / (R/Rvir) at the first nonzero bin, then biweights over galaxies.
    Returns None for non-virial bin modes or if it cannot be computed, so it is
    purely informational and never affects the stacks themselves.
    """
    from astropy.stats import biweight_location

    if config.bin_mode.lower() not in ("virial", "vr"):
        return None
    bins = np.asarray(config.bins, dtype=float)
    nz = np.nonzero(bins)[0]
    if nz.size == 0:
        return None
    j = int(nz[0])
    mass = np.asarray(mass, dtype=float)
    z = np.asarray(z, dtype=float)

    rvir = np.full(len(z), np.nan)
    for i in range(len(z)):
        try:
            edges = np.asarray(radial_bin_edges(config, float(mass[i]), float(z[i])),
                               dtype=float)
            rvir[i] = edges[j] / bins[j]
        except Exception:
            rvir[i] = np.nan

    if not np.any(np.isfinite(rvir)):
        return None
    val = biweight_location(rvir, ignore_nan=True)
    return float(val) if np.isfinite(val) else None


def build_stacks(config: "PipelineConfig", product: "GalaxyProduct",
                 keep_cube: bool = True, weights=None) -> dict:
    """
    Per radial bin: convert_avg_fiber_bin (unit + (1+z) + per-fiber area), then
    stack_spec_values (rest-frame resample), then coadd across galaxies for
    every method in config.galaxy_combine_methods.

    `keep_cube` retains the per-galaxy rest-frame cube (ngal, nrad, nwave_rest),
    needed for the Stage 3 galaxy bootstrap.

    `weights` (ngal, nrad) are the per-galaxy widths consumed by the
    'weighted_median' combine; every other method ignores them. Defaults to
    product.nfib (fibers per bin) when None -- pass your own array to override
    (e.g. inverse-variance or uniform weights). Stored as cube_weights when
    keep_cube, so Stage 3's bootstrap resamples the same widths.

    Returns dict: rest_wave, r_edges, stacks{method:{flux,err}}, z_median,
    mass_median, unit_info, and (if keep_cube) cube_flux/cube_err/cube_weights.
    """
    spec = np.asarray(product.spec)         # (ngal, nrad, nwave)
    err = np.asarray(product.err)
    obs_wave = np.asarray(product.wave)
    cat = product.catalog
    ngal, nrad, _ = spec.shape

    # default the galaxy-axis widths to nfib; allow a caller override
    weights = product.nfib if weights is None else np.asarray(weights, dtype=float)

    rest_wave = rest_grid(config)
    z = np.asarray(cat[config.z_col], dtype=float)
    m = np.asarray(cat[config.mass_col], dtype=float)

    cube_f = np.full((ngal, nrad, rest_wave.size), np.nan, dtype=np.float64)
    cube_e = np.full((ngal, nrad, rest_wave.size), np.nan, dtype=np.float64)
    # float64 is REQUIRED here: in L_kpc2 units flux ~1e37-1e38 and err^2 ~1e74,
    # which overflow float32 (max ~3.4e38) -> inf -> inv_var/mean collapse to NaN.
    unit_info = None

    # Process per radial bin (matches the notebook's per-bin loop)
    for r in range(nrad):
        y, ye, unit_info = convert_avg_fiber_bin(
            spec[:, r, :], err[:, r, :], z,
            output=config.flux_unit,
            flux_unit_scale=config.flux_unit_scale,
            fiber_diam_arcsec=config.fiber_diam_arcsec,
            rest_density=config.rest_density,
        )
        f_rf, e_rf = stack_spec_values(y, ye, z, obs_wave, rest_wave)
        cube_f[:, r, :] = f_rf
        cube_e[:, r, :] = e_rf

    stacks = coadd_galaxies(cube_f, cube_e, config.galaxy_combine_methods, weights)

    result = {
        "rest_wave": rest_wave,
        "r_edges": np.asarray(product.r_edges),
        "stacks": stacks,
        "bin_mode": config.bin_mode,
        "z_median": float(np.nanmedian(z)),
        "mass_median": float(np.nanmedian(m)),
        "unit_info": unit_info,
    }
    try:
        result["VR_biweight_v"] = sample_virial_radius_kpc(config, m, z)
    except Exception:
        result["VR_biweight_v"] = None   # never let a reporting helper break Stage 2
    if keep_cube:
        result["cube_flux"] = cube_f
        result["cube_err"] = cube_e
        result["cube_weights"] = weights        # (ngal, nrad), aligned to the cube
    from .config import stamp_provenance
    return stamp_provenance(result, config, "build_stacks")


# ---------------------------------------------------------------------
# 2.3.5  SINGLE GALAXY  (build_stacks minus the galaxy-axis coadd)
# ---------------------------------------------------------------------
# GalaxyProduct.spec is already (ngal, nrad, nwave) -- every galaxy carries its
# own radial-binned spectra; build_stacks only collapses the galaxy axis at the
# very end. So one galaxy's "spectra vs radius" is product.spec[i], no new
# extraction needed. This is intended for BRIGHT objects (AGN, strong LAEs,
# double-peakers): a single object's per-bin S/N is roughly the stack's divided
# by sqrt(N_gal) that went into a normal stack, so most galaxies will show
# signal only in the inner 1-2 bins. Use selection.top_sn_galaxies on a CORE
# product first to find good candidates, then inspect their halo profile here.
def resolve_galaxy_index(product: "GalaxyProduct", *, gid=None, index=None,
                         id_col="ID") -> int:
    """Row index of the requested galaxy on the product's galaxy axis.

    Pass exactly one of `gid` (matched against catalog[id_col], as strings so
    123 and '123' both work) or `index` (direct row, negatives allowed).
    Raises on no / multiple ID matches so a silent mismatch can't slip through.
    """
    if (gid is None) == (index is None):
        raise ValueError("pass exactly one of gid= or index=.")
    if index is not None:
        ng = product.ngal
        i = int(index)
        i = i + ng if i < 0 else i
        if not (0 <= i < ng):
            raise IndexError(f"index {index} out of range for ngal={ng}")
        return i
    cat = product.catalog
    if id_col not in cat.colnames:
        raise KeyError(f"id_col {id_col!r} not in catalog columns {list(cat.colnames)}")
    col = np.asarray(cat[id_col]).astype(str)
    hits = np.flatnonzero(col == str(gid))
    if hits.size == 0:
        raise KeyError(f"no galaxy with {id_col}={gid!r}")
    if hits.size > 1:
        raise ValueError(f"{hits.size} galaxies match {id_col}={gid!r} "
                         f"(rows {hits.tolist()}); pass index= to disambiguate.")
    return int(hits[0])


def _single_galaxy_rvir_kpc(config, mass, z):
    """This galaxy's own kpc-per-(R/Rvir) factor, for the kpc top axis.
    Mirrors sample_virial_radius_kpc but for one object (no biweight).
    Returns None for non-virial bin modes or if it can't be derived."""
    if config.bin_mode.lower() not in ("virial", "vr"):
        return None
    bins = np.asarray(config.bins, dtype=float)
    nz = np.nonzero(bins)[0]
    if nz.size == 0 or not np.isfinite(mass) or not np.isfinite(z):
        return None
    j = int(nz[0])
    try:
        edges = np.asarray(radial_bin_edges(config, float(mass), float(z)), dtype=float)
        val = edges[j] / bins[j]
        return float(val) if np.isfinite(val) else None
    except Exception:
        return None


def single_galaxy_spectra(config: "PipelineConfig", product: "GalaxyProduct",
                          *, gid=None, index=None, id_col=None,
                          merge_bins=None) -> dict:
    """
    Radius-resolved rest-frame spectra for ONE galaxy, in the same units and
    frame convention as build_stacks (convert_avg_fiber_bin -> stack_spec_values,
    fed a length-1 galaxy axis).

    Selection: pass gid= (catalog ID) or index= (row). id_col defaults to
    config.id_col.

    merge_bins (optional): a list of index groups to coadd adjacent radial bins
    for this one object (nfib-weighted, via combine_fibers), buying back S/N at
    the cost of radial resolution, e.g. merge_bins=[[0],[1],[2,3],[4,5,6]] makes
    4 output bins from 7. None -> native bins.

    Returns a dict shaped like a stacks_result (rest_wave, r_edges,
    stacks={'galaxy': {'flux','err'}}, unit_info, bin_mode, VR_biweight_v) so
    plotting.plot_stack_panels / plot_radius_velocity_map work on it unchanged
    with method='galaxy'. Also carries nfib and a crude per-bin line S/N (sn)
    in config.line_window, plus z/mass/gid/index, for plotting.plot_single_
    galaxy_panels' annotations.
    """
    from .measure import integrated_line_flux   # lazy: avoid a hard measure<->stack import cycle

    id_col = id_col or config.id_col
    i = resolve_galaxy_index(product, gid=gid, index=index, id_col=id_col)

    spec_i = np.asarray(product.spec[i], dtype=float)     # (nrad, nwave) observed
    err_i  = np.asarray(product.err[i],  dtype=float)
    nfib_i = np.asarray(product.nfib[i], dtype=float)     # (nrad,)
    obs_wave = np.asarray(product.wave, dtype=float)
    r_edges  = np.asarray(product.r_edges, dtype=float)
    z    = float(np.asarray(product.catalog[config.z_col])[i])
    mass = float(np.asarray(product.catalog[config.mass_col])[i])
    nrad = spec_i.shape[0]

    # optional adjacent-bin merge (nfib-weighted), before rest-framing
    if merge_bins is not None:
        groups = [list(np.atleast_1d(g)) for g in merge_bins]
        f_m = np.full((len(groups), spec_i.shape[1]), np.nan)
        e_m = np.full_like(f_m, np.nan)
        nf_m = np.zeros(len(groups))
        new_edges = [r_edges[0]]
        for k, g in enumerate(groups):
            w = nfib_i[g]
            f_m[k], e_m[k] = combine_fibers(spec_i[g], err_i[g],
                                            method="weighted_median", weights=w)
            nf_m[k] = float(np.sum(w))
            new_edges.append(r_edges[max(g) + 1])
        spec_i, err_i, nfib_i = f_m, e_m, nf_m
        r_edges = np.asarray(new_edges, dtype=float)
        nrad = spec_i.shape[0]

    # unit + (1+z), then rest-frame resample -- exactly the stack path, per bin,
    # feeding this one galaxy as a length-1 galaxy axis.
    rest_wave = rest_grid(config)
    flux = np.full((nrad, rest_wave.size), np.nan)
    err  = np.full_like(flux, np.nan)
    unit_info = None
    zc = np.array([z])
    for r in range(nrad):
        y, ye, unit_info = convert_avg_fiber_bin(
            spec_i[r][None, :], err_i[r][None, :], zc,
            output=config.flux_unit, flux_unit_scale=config.flux_unit_scale,
            fiber_diam_arcsec=config.fiber_diam_arcsec,
            rest_density=config.rest_density,
        )
        f_rf, e_rf = stack_spec_values(y, ye, zc, obs_wave, rest_wave)
        flux[r] = f_rf[0]
        err[r]  = e_rf[0]

    # crude per-bin line S/N in the config line window (integrated flux / its err)
    lo, hi = float(config.line_window[0]), float(config.line_window[1])
    sn = np.full(nrad, np.nan)
    for r in range(nrad):
        res = integrated_line_flux(rest_wave, flux[r], err=err[r], bounds=(lo, hi))
        if res["success"] and res["err_sum"] and np.isfinite(res["err_sum"]) and res["err_sum"] > 0:
            sn[r] = res["flux_sum"] / res["err_sum"]

    return {
        "rest_wave": rest_wave,
        "r_edges": r_edges,
        "stacks": {"galaxy": {"flux": flux, "err": err}},
        "nfib": np.asarray(nfib_i, dtype=float),
        "sn": sn,
        "bin_mode": config.bin_mode,
        "z": z, "mass": mass,
        "gid": (None if gid is None else str(gid)), "index": i,
        "unit_info": unit_info,
        "VR_biweight_v": _single_galaxy_rvir_kpc(config, mass, z),
    }


# ---------------------------------------------------------------------
# 2.4  CORE stack  (single-aperture PSF core spectra, no radial axis)
# ---------------------------------------------------------------------
def build_core_stacks(cfg, core_prod, mask=None, output="flux", methods=None):
    """
    Stack PSF-weighted CORE spectra (a core.read_core_fits dict) into one
    rest-frame coadd per combine method. This is build_stacks with the radial
    axis removed: the core extraction is a single aperture, (ngal, nwave), so we
    convert + (1+z) + resample exactly as the binned path does, then coadd across
    galaxies via coadd_galaxies with a length-1 radial axis.

    output : 'flux'    -> erg/s/cm^2/A_rest (flux_unit_scale * (1+z) if rest_density)
             'L_fiber' -> rest-frame luminosity density (4*pi*DL^2*(1+z))
    The area-based outputs (flux_arcsec2, L_kpc2) are intentionally NOT offered
    here: they divide by the FIBER area, which is wrong for a PSF-weighted
    point-source core (it needs the PSF effective aperture instead).

    Returns dict: rest_wave, stacks{method:{flux,err}}, ngal, z_median,
    unit_info, and cube_flux/cube_err (ngal, nwave rest-frame) for the Stage-3
    galaxy bootstrap.
    """
    spec     = np.asarray(core_prod["core_spec"], float)   # (ngal, nwave)
    err      = np.asarray(core_prod["core_err"],  float)
    obs_wave = np.asarray(core_prod["wave_obs"],  float)
    cat      = core_prod["catalog"]
    z        = np.asarray(cat[cfg.z_col], float)

    if mask is not None:
        mask = np.asarray(mask, bool)
        spec, err, z, cat = spec[mask], err[mask], z[mask], cat[mask]
    ngal = spec.shape[0]
    if ngal == 0:
        raise ValueError("no galaxies left after mask")

    # --- 1. unit + (1+z) rest-density ---
    if output == "flux":
        zfac = (1.0 + z)[:, None] if cfg.rest_density else 1.0
        y, ye = spec * cfg.flux_unit_scale * zfac, err * cfg.flux_unit_scale * zfac
        unit_info = {"y_unit": "erg/s/cm^2/A_rest" if cfg.rest_density
                                else "erg/s/cm^2/A_obs"}
    elif output == "L_fiber":
        y, ye, unit_info = convert_avg_fiber_bin(
            spec, err, z, output="L_fiber",
            flux_unit_scale=cfg.flux_unit_scale,
            fiber_diam_arcsec=cfg.fiber_diam_arcsec,
            rest_density=cfg.rest_density,
        )
    else:
        raise ValueError("core stack output must be 'flux' or 'L_fiber' "
                         "(area outputs need the PSF aperture, not the fiber area)")

    # --- 2. rest-frame resample ---
    rest_wave = rest_grid(cfg)
    f_rf, e_rf = stack_spec_values(y, ye, z, obs_wave, rest_wave)

    # --- 3. coadd across galaxies, every method (fake nrad=1 axis) ---
    methods = list(methods or cfg.galaxy_combine_methods)

    # weighted_median needs a per-galaxy width; on the core path that's
    # 'nfib_used' (fibers in the PSF aperture), which read_core_fits does NOT
    # restore, so it's normally absent -> drop the method rather than crash.
    core_w = core_prod.get("results", {}).get("nfib_used")
    if core_w is not None:
        core_w = np.asarray(core_w, float)
        if mask is not None:
            core_w = core_w[mask]
        core_w = core_w[:, None]                       # (ngal, 1) for the nrad=1 axis
    elif any(m.lower() == "weighted_median" for m in methods):
        methods = [m for m in methods if m.lower() != "weighted_median"]
        print("[core] 'weighted_median' skipped: core product has no 'nfib_used' "
              "weights (radial-stack method; not needed for the single-aperture core).")

    co = coadd_galaxies(f_rf[:, None, :], e_rf[:, None, :], methods, weights=core_w)
    stacks = {m: {"flux": co[m]["flux"][0], "err": co[m]["err"][0]} for m in methods}

    return {
        "rest_wave": rest_wave, "stacks": stacks, "ngal": int(ngal),
        "z_median": float(np.nanmedian(z)), "output": output,
        "unit_info": unit_info,
        "cube_flux": f_rf, "cube_err": e_rf,   # (ngal, nwave) rest-frame, for Stage-3 bootstrap
    }


# ---------------------------------------------------------------------
# 2.5  CORE-product assembly (pool fields before stacking)
# ---------------------------------------------------------------------
def concat_core_prods(*prods, wave_atol=1e-3):
    """Concatenate core products (core.read_core_fits dicts) along the galaxy
    axis. All must share the same observed wavelength grid. Catalog row order
    stays aligned to the spectra. Returns a new prod dict usable by
    build_core_stacks / core.select_core_by_index.

    If the inputs land on DIFFERENT observed grids this raises; in that case
    build a stack per product (build_core_stacks already resamples each galaxy
    onto rest_grid(cfg)) and pool the per-galaxy rest-frame cubes instead.
    """
    prods = [p for p in prods if p is not None]
    if not prods:
        raise ValueError("no products given")

    # 1) wave grids must match (same instrument/pipeline -> they should)
    w0 = np.asarray(prods[0]["wave_obs"], float)
    for k, p in enumerate(prods):
        w = np.asarray(p["wave_obs"], float)
        if w.shape != w0.shape or not np.allclose(w, w0, atol=wave_atol, equal_nan=True):
            raise ValueError(
                f"product {k} has a different wave_obs grid (n={w.shape} vs {w0.shape}); "
                "use the rest-frame-concat fallback (pool cube_flux/cube_err from "
                "per-product build_core_stacks) instead of this function")

    # 2) stack spectra + errors
    spec = np.vstack([np.asarray(p["core_spec"], float) for p in prods])
    err  = np.vstack([np.asarray(p["core_err"],  float) for p in prods])

    # 3) stack catalogs (try exact; fall back to outer if columns differ)
    try:
        cat = vstack([p["catalog"] for p in prods], join_type="exact",
                     metadata_conflicts="silent")
    except Exception as exc:
        cols = [set(p["catalog"].colnames) for p in prods]
        diff = set().union(*cols) - set().intersection(*cols)
        print(f"[concat] catalogs differ in columns {sorted(diff)}; "
              f"using outer join (missing entries masked).  ({exc})")
        cat = vstack([p["catalog"] for p in prods], join_type="outer",
                     metadata_conflicts="silent")

    # 4) results: concat per-galaxy arrays, keep/validate scalars
    scalar_keys = ("method", "n_sigma", "n_bootstrap")
    ngals = [len(p["core_spec"]) for p in prods]
    results = {}
    for key in prods[0]["results"]:
        if any(key not in p["results"] for p in prods):
            continue
        if key in scalar_keys:
            vals = [p["results"][key] for p in prods]
            if len({str(v) for v in vals}) > 1:
                print(f"[concat] warning: '{key}' differs across inputs {vals}; keeping first")
            results[key] = vals[0]
        else:
            parts = [np.atleast_1d(np.asarray(p["results"][key])) for p in prods]
            results[key] = (np.concatenate(parts)
                            if all(pt.shape[0] == n for pt, n in zip(parts, ngals))
                            else prods[0]["results"][key])

    return dict(core_spec=spec, core_err=err, wave_obs=w0,
                catalog=cat, results=results, header=prods[0]["header"],
                n_per_input=ngals)
