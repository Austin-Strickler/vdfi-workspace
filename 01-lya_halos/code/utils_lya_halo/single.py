"""
single.py -- radius-resolved Lya spectra for ONE galaxy (no galaxy-axis coadd).

This is build_stacks with the coadd removed. Stage 1 already extracts every
galaxy's binned, background-subtracted spectra into GalaxyProduct.spec
(ngal, nrad, nwave); this module pulls one galaxy, applies the SAME unit + (1+z)
+ rest-frame resample the stack uses (convert_avg_fiber_bin -> stack_spec_values),
and packages the result so the existing stack plotters work on it unchanged.

    prod  = read_galaxy_fits(path)            # or load_for_stack(cfg)
    sg    = single_galaxy_spectra(cfg, prod, gid=12345)
    plot_single_galaxy_panels(sg)             # panels + per-bin nfib / S/N

Because sg has the same keys as a stacks_result (rest_wave, r_edges, unit_info,
and stacks={'galaxy': {'flux','err'}}), it also drops straight into the existing
plotters:

    from .plotting import plot_stack_panels, plot_radius_velocity_map
    plot_stack_panels(sg, method='galaxy')
    plot_radius_velocity_map(sg, method='galaxy', normalize='none')

Intended use is BRIGHT objects (AGN, strong LAEs, double-peakers). For an average
galaxy the outer bins are noise: a single object's per-bin S/N is ~ the stack's
divided by sqrt(N_gal). The per-bin nfib and sn arrays (and the panel
annotations) are there so you can see where the signal dies for each object.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import astropy.units as u

from .stack import convert_avg_fiber_bin, stack_spec_values, rest_grid
from .extract import radial_bin_edges, combine_fibers
from .virial import estimate_M200c_R200c_from_Mstar, cosmo
from .measure import measure_centroid, integrated_line_flux, DEFAULT_CONT_BOUNDS

if TYPE_CHECKING:
    from .config import PipelineConfig, PSFSpec
    from .io import GalaxyProduct

C_KMS    = 299792.458
LYA_REST = 1215.67


# =====================================================================
# 0.  find the galaxy row
# =====================================================================
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


def _single_galaxy_rvir_kpc(config, mass, z, m_min: float = 7.0, m_max: float = 11.0) -> Optional[float]:
    """This galaxy's own R200c [kpc] -- the kpc-per-(R/Rvir=1) factor, for the
    kpc<->R/Rvir comparison axis. Mirrors stack.sample_virial_radius_kpc for
    one object (no biweight/scatter -- a single galaxy has no sample scatter
    to combine). Computed the same way regardless of config.bin_mode (see
    stack.sample_virial_radius_kpc)."""
    if not np.isfinite(mass) or not np.isfinite(z) or z <= 0:
        return None
    try:
        m_used = np.clip(float(mass), m_min, m_max)
        model = getattr(config, "smhm_model", "girelli20")
        _, r200c = estimate_M200c_R200c_from_Mstar(10 ** m_used, float(z), model=model)
        return float(r200c) if np.isfinite(r200c) else None
    except Exception:
        return None


# =====================================================================
# 1.  the data function  (build_stacks minus the coadd)
# =====================================================================
def single_galaxy_spectra(config: "PipelineConfig", product: "GalaxyProduct",
                          *, gid=None, index=None, id_col=None,
                          merge_bins=None) -> dict:
    """
    Radius-resolved rest-frame spectra for ONE galaxy, in the same units and
    frame convention as the stacks.

    Selection: pass gid= (catalog ID) or index= (row). id_col defaults to
    config.id_col.

    merge_bins (optional): a list of index groups to coadd adjacent radial bins
    for this one object, buying back S/N at the cost of radial resolution, e.g.
    merge_bins=[[0], [1], [2, 3], [4, 5, 6]] makes 4 output bins. Coadd is
    nfib-weighted (weighted_median), matching the galaxy-axis convention. r_edges
    are collapsed to the outer edge of each group. None -> native bins.

    Returns a dict shaped like a stacks_result so the existing plotters work:
        rest_wave : (nrad_out,)-independent rest grid
        r_edges   : (nrad_out+1,)
        stacks    : {'galaxy': {'flux': (nrad_out, nwave), 'err': ...}}
        nfib      : (nrad_out,)  fibers per (merged) bin
        sn        : (nrad_out,)  crude per-bin line S/N in the config line window
        bin_mode, z, mass, gid, index, unit_info, VR_biweight_v
    """
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


# =====================================================================
# 2.  the plotter  (panels + per-bin nfib / S/N annotation)
# =====================================================================
def plot_single_galaxy_panels(sg, *, xlim=(1150, 1300),
                              lya_center=LYA_REST, lya_window=4.0,
                              figsize_width=10, panel_height=2.3,
                              smooth=None, show_err=True,
                              title=None, savepath=None, dpi=300):
    """
    One rest-frame spectral panel per radial bin for a SINGLE galaxy, each panel
    annotated with the bin's fiber count and crude line S/N -- so you can see how
    far out this object is trustworthy. Mirrors plotting.plot_stack_panels.

    sg : the dict from single_galaxy_spectra.
    Returns (fig, axs).
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    rest_wave = np.asarray(sg["rest_wave"])
    r_edges   = np.round(np.asarray(sg["r_edges"]), 3)
    flux_all  = np.asarray(sg["stacks"]["galaxy"]["flux"])   # (nrad, nwave)
    err_all   = np.asarray(sg["stacks"]["galaxy"]["err"])
    nfib      = np.asarray(sg["nfib"])
    sn        = np.asarray(sg["sn"])
    nrad      = flux_all.shape[0]
    unit      = (sg.get("unit_info") or {}).get("y_unit", "")
    dist      = {"virial": r"$R/R_{\rm vir}$", "vr": r"$R/R_{\rm vir}$",
                 "kpc": "kpc", "arcsec": '"'}.get(str(sg.get("bin_mode", "")).lower(), "")

    fig, axs = plt.subplots(nrad, 1, figsize=(figsize_width, panel_height * nrad),
                            sharex=True, constrained_layout=True)
    if nrad == 1:
        axs = [axs]
    win_l, win_r = lya_center - lya_window, lya_center + lya_window

    for r, ax in enumerate(axs):
        flux = flux_all[r]
        err  = err_all[r] if show_err else None
        y = gaussian_filter1d(flux, smooth) if smooth is not None else flux
        ax.step(rest_wave, y, where="mid", lw=1, color="k")
        if show_err and err is not None:
            ax.fill_between(rest_wave, flux - err, flux + err, step="mid",
                            color="0.75", alpha=0.35, lw=0)
        ax.axvspan(win_l, win_r, color="magenta", alpha=0.05)
        ax.axvline(lya_center, color="magenta", lw=0.8, alpha=0.6)
        ax.axhline(0, color="tomato", lw=1, alpha=0.8)

        win = (rest_wave >= xlim[0]) & (rest_wave <= xlim[1]) & np.isfinite(flux)
        if np.any(win):
            fmin, fmax = np.nanpercentile(flux[win], [0.5, 99.5])
            pad = 0.3 * (fmax - fmin if fmax > fmin else abs(fmax) + 1)
            ax.set_ylim(fmin - pad, fmax + pad)

        label = (f"{np.round(r_edges[r], 3)}\u2013{np.round(r_edges[r+1], 3)} {dist}"
                 f"   N$_{{\\rm fib}}$={int(nfib[r])}"
                 f"   S/N={sn[r]:.1f}" if np.isfinite(sn[r]) else
                 f"{np.round(r_edges[r], 3)}\u2013{np.round(r_edges[r+1], 3)} {dist}"
                 f"   N$_{{\\rm fib}}$={int(nfib[r])}")
        ax.text(0.02, 0.80, label, transform=ax.transAxes, fontsize=10,
                color="dodgerblue",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.85",
                          boxstyle="round,pad=0.25"))
        ax.tick_params(axis="both", which="major", labelsize=10, direction="in")
        ax.grid(alpha=0.18, lw=0.6)

    axs[-1].set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=14)
    axs[-1].set_xlim(xlim)
    fig.supylabel(f"Flux density [{unit}]" if unit else "Flux density", fontsize=13)
    if title is None:
        gid = sg.get("gid"); idx = sg.get("index")
        who = f"ID {gid}" if gid not in (None, "None") else f"row {idx}"
        title = (rf"Single galaxy {who}   ($z={sg.get('z', float('nan')):.3f}$, "
                 rf"$\log M={sg.get('mass', float('nan')):.2f}$)")
    fig.suptitle(title, fontsize=15, y=1.01)

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axs


# =====================================================================
# 3.  OPTIONAL phase-2 measurement  (centroid / flux vs radius, one galaxy)
# =====================================================================
def measure_single_galaxy(sg, *, method="flux_weighted",
                          bounds=None, cont_bounds=DEFAULT_CONT_BOUNDS):
    """
    Per-bin centroid velocity and integrated flux for the single-galaxy product
    from single_galaxy_spectra. NO bootstrap (one object can't be resampled over
    galaxies); the error bar per bin is the propagated flux error, and sn comes
    from sg['sn']. Good enough to eyeball a bright object's centroid(r), or to
    read the red-peak shift of a double-peaker bin by bin.

    Returns a dict of (nrad,) arrays: r_edges, centroid_v, flux, flux_err, sn,
    plus the method used. Feed centroid_v / r_edges straight into your own
    radius-axis plot (see the example in the module docstring / chat).
    """
    rest_wave = np.asarray(sg["rest_wave"])
    flux_all  = np.asarray(sg["stacks"]["galaxy"]["flux"])
    err_all   = np.asarray(sg["stacks"]["galaxy"]["err"])
    r_edges   = np.asarray(sg["r_edges"])
    nrad      = flux_all.shape[0]
    b = tuple(bounds) if bounds is not None else (LYA_REST - 4.0, LYA_REST + 4.0)

    cen_v = np.full(nrad, np.nan)
    fx    = np.full(nrad, np.nan)
    fxe   = np.full(nrad, np.nan)
    for r in range(nrad):
        c = measure_centroid(rest_wave, flux_all[r], err=err_all[r],
                             method=method, bounds=b, cont_bounds=cont_bounds)
        if c.get("success"):
            cen_v[r] = c["centroid_vel_kms"]
        fres = integrated_line_flux(rest_wave, flux_all[r], err=err_all[r],
                                    bounds=b, cont_bounds=cont_bounds)
        if fres["success"]:
            fx[r]  = fres["flux_integral"]    # x dlambda (2026-09-17; was flux_sum)
            fxe[r] = fres["err_integral"]

    return {"r_edges": r_edges, "centroid_v": cen_v, "flux": fx,
            "flux_err": fxe, "sn": np.asarray(sg["sn"]), "method": method}


# =====================================================================
# 4.  PSF standardization for ONE galaxy (specs/single-galaxy-pipeline.md
#     Part B). Backs Part A's leak-aware core/halo split and Part C's
#     imaging PSF contour -- one PSFSpec, one function that resolves it to a
#     per-galaxy fwhm_kpc, used everywhere in this module that needs a PSF.
# =====================================================================
def _measured_psf_fwhm_kpc(vdfi: dict, ra: float, dec: float, z: float, *,
                           bad_fiber_mask=None, seeing_radius_arcsec: float = 3.5,
                           wave_exponent: float = -1.0 / 5.0):
    """
    This galaxy's own PSF FWHM, from the REAL per-exposure VDFI seeing
    (PSFSpec provenance='measured'; Part B1/B2) -- each galaxy gets its own
    value, determined by which exposures actually cover it, rather than one
    sample-wide constant.

    'Covers this galaxy' = at least one good, finite-coordinate fiber within
    seeing_radius_arcsec of (ra, dec) in that exposure -- the SAME test
    core.extract_core_spectra uses (`dist_all[e] <= core_radius`) to decide
    which exposures feed the PSF-weighted extraction, so pass the same
    core_radius here (default parameter name differs only because this
    function doesn't do extraction) to keep 'measured' and 'psf_weighted'
    agreeing on which exposures this galaxy's PSF model is built from.

    Each covering exposure contributes ONE scalar seeing (nan-median over
    its own fibers -- the same collapse core._exposure_seeing does); this
    galaxy's reference seeing is the median across covering exposures. This
    is a simple, robust combine, NOT a PSF-weighted per-fiber average (that
    would just re-derive core.extract_core_spectra's own machinery) -- it's
    meant to be a cheap per-galaxy summary for the leak model and the
    imaging PSF contour, not a second optimal-extraction pass.

    Returns (fwhm_kpc, fwhm_arcsec_at_lya). Raises RuntimeError if vdfi
    lacks RA/DEC/seeing, or if no exposure covers the galaxy at all (in
    which case use psf=PSFSpec(provenance='fiducial') instead).
    """
    RA = vdfi.get("RA"); DEC = vdfi.get("DEC"); seeing = vdfi.get("seeing")
    if RA is None or DEC is None or seeing is None:
        raise RuntimeError(
            "_measured_psf_fwhm_kpc needs vdfi['RA']/['DEC']/['seeing'] -- "
            "pass the dict from extract.load_field(config) (or "
            "core.load_field_with_psf for the psf_weighted core path).")
    RA = np.asarray(RA); DEC = np.asarray(DEC)
    seeing = np.asarray(seeing, dtype=float)
    nexp = RA.shape[0]

    see_vals = []
    for e in range(nexp):
        dra  = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
        ddec = (DEC[e] - dec) * 3600.
        r = np.sqrt(dra ** 2 + ddec ** 2)
        good = np.isfinite(RA[e]) & np.isfinite(DEC[e])
        if bad_fiber_mask is not None:
            good = good & (~np.asarray(bad_fiber_mask[e]))
        if not np.any(good & (r <= seeing_radius_arcsec)):
            continue
        row = seeing[e] if seeing.ndim > 1 else seeing[e:e + 1]
        val = float(np.nanmedian(row)) if np.ndim(row) else float(row)
        if np.isfinite(val):
            see_vals.append(val)

    if not see_vals:
        raise RuntimeError(
            f"_measured_psf_fwhm_kpc: no exposure has a good fiber within "
            f"{seeing_radius_arcsec}\" of (ra={ra}, dec={dec}) -- can't "
            f"determine a measured seeing for this galaxy. Pass "
            f"psf=PSFSpec(provenance='fiducial') to use the fixed analytic "
            f"Moffat instead.")

    seeing_ref = float(np.nanmedian(see_vals))
    obs_wave_lya = LYA_REST * (1.0 + z)
    seeing_at_lya = seeing_ref * (obs_wave_lya / 4500.) ** wave_exponent
    kpc_per_arcsec = float(cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265)
    return seeing_at_lya * kpc_per_arcsec, seeing_at_lya


def psf_fwhm_kpc_for_galaxy(config: "PipelineConfig", z: float, *,
                            vdfi: dict = None, ra: float = None, dec: float = None,
                            bad_fiber_mask=None, seeing_radius_arcsec: float = 3.5,
                            psf: "PSFSpec" = None) -> dict:
    """
    This galaxy's PSF FWHM in kpc, per config.PSFSpec (Part B) -- the single
    entry point every function in this module uses to resolve a per-galaxy
    PSF, so 'which PSF produced this number' is one code path, not
    re-derived per caller.

    psf : override config.psf for this one call. None (default) -> config.psf.

    provenance == 'measured' (config default) needs vdfi, ra, dec -- raises
    a clear RuntimeError if they're missing (this module's convention: a
    missing PSF ingredient is an error, not a silent NaN -- see the module
    docstring's note on vdfi['PSF']). provenance == 'fiducial' only needs z:
    mirrors measure.measure_psf_corrected_core_luminosity's own fixed-
    literature-Moffat convention exactly, including NOT wavelength-
    correcting the fixed angular FWHM (kept consistent with that existing
    stack-level behavior rather than introducing a new correction there).

    Returns dict: fwhm_kpc, fwhm_arcsec_used, beta, provenance.
    """
    psf = psf or config.psf
    beta = float(psf.beta)

    if psf.provenance == "fiducial":
        fwhm_arcsec = float(psf.fwhm_arcsec)
        kpc_per_arcsec = float(cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265)
        return {"fwhm_kpc": fwhm_arcsec * kpc_per_arcsec,
                "fwhm_arcsec_used": fwhm_arcsec, "beta": beta,
                "provenance": "fiducial"}

    if psf.provenance == "measured":
        if vdfi is None or ra is None or dec is None:
            raise RuntimeError(
                "psf.provenance='measured' needs vdfi, ra, dec -- pass the "
                "loaded field dict (extract.load_field(config), run once by "
                "the caller -- see this module's docstring) and this "
                "galaxy's RA/DEC, or pass psf=PSFSpec(provenance='fiducial') "
                "to use the fixed analytic Moffat instead.")
        fwhm_kpc, fwhm_arcsec_used = _measured_psf_fwhm_kpc(
            vdfi, ra, dec, z, bad_fiber_mask=bad_fiber_mask,
            seeing_radius_arcsec=seeing_radius_arcsec,
            wave_exponent=psf.wave_exponent)
        return {"fwhm_kpc": fwhm_kpc, "fwhm_arcsec_used": fwhm_arcsec_used,
                "beta": beta, "provenance": "measured"}

    raise ValueError(f"unknown psf.provenance {psf.provenance!r}")


# =====================================================================
# 5.  PART A -- per-galaxy core + halo luminosity and the halo fraction,
#     LEAK-AWARE (specs/single-galaxy-pipeline.md Part A).
# =====================================================================
def _psf_weighted_core_luminosity(config: "PipelineConfig", product: "GalaxyProduct",
                                  i: int, *, vdfi: dict, core_radius: float,
                                  bounds, cont_bounds, cont_method, cont_order,
                                  clip_negative: bool, ra_col: str, dec_col: str) -> dict:
    """
    PSF-weighted core flux for ONE galaxy (row `i` of `product.catalog`), via
    a JOIN to core.extract_core_spectra -- not a reimplementation (Part A1).

    Runs the same per-exposure PSF-weighted optimal extraction the core
    track uses (on a 1-row run_table), then converts + rest-frames it
    exactly as stack.build_core_stacks does with output='L_fiber' -- a
    genuine rest-frame LUMINOSITY DENSITY, directly comparable (after line-
    window integration) to the aperture path's area-scaled core luminosity.
    Deliberately NOT 'L_kpc2': that output divides by the FIBER area, which
    is wrong for a PSF-weighted point-source core (build_core_stacks'
    own docstring makes the same point -- it needs the PSF's effective
    aperture, not a fiber's, and 'L_fiber' already gives a total-source
    quantity that needs no further area scaling).

    Requires vdfi loaded with build_psf=True (core.load_field_with_psf);
    raises extract_core_spectra's own clear error otherwise.
    """
    from . import core as core_mod

    row = product.catalog[i:i + 1]
    Spec, Err, wave_obs, cat = core_mod.extract_core_spectra(
        config, run_table=row, core_radius=core_radius, vdfi=vdfi,
        ra_col=ra_col, dec_col=dec_col, save_background=False)

    z = np.array([float(np.asarray(cat[config.z_col])[0])])
    y, ye, unit_info = convert_avg_fiber_bin(
        Spec, Err, z, output="L_fiber",
        flux_unit_scale=config.flux_unit_scale,
        fiber_diam_arcsec=config.fiber_diam_arcsec,
        rest_density=config.rest_density,
    )
    rest_wave = rest_grid(config)
    f_rf, e_rf = stack_spec_values(y, ye, z, wave_obs, rest_wave)

    res = integrated_line_flux(rest_wave, f_rf[0], err=e_rf[0],
                               bounds=bounds, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order,
                               clip_negative=clip_negative)
    core_flux = res["flux_integral"] if res["success"] else np.nan   # x dlambda (2026-09-17)
    core_flux_err = res["err_integral"] if res["success"] else np.nan
    return {"core_flux": core_flux, "core_flux_err": core_flux_err, "unit_info": unit_info}


def measure_single_galaxy_core_halo(
    config: "PipelineConfig", product: "GalaxyProduct",
    *, gid=None, index=None, id_col=None,
    core_method: str = "aperture",
    core_bin_index: int = 0,
    boundary_kpc: float = None,
    twohalo_outer_kpc: float = None,
    vdfi: dict = None,
    core_radius: float = 3.5,
    psf: "PSFSpec" = None,
    bounds=None, cont_bounds=None, cont_method=None, cont_order=None,
    clip_negative: bool = False,
    sn_floor: float = None,
    ra_col: str = "RA", dec_col: str = "DEC",
) -> dict:
    """
    Part A -- per-galaxy core flux, halo flux, and the halo fraction, for
    ONE galaxy, LEAK-AWARE.

    core_method : 'aperture' (default) -- the innermost radial bin
        (`core_bin_index`, native kpc bins from single_galaxy_spectra) is
        "the core", aperture/PSF corrected by rescaling it 1/EE_inner where
        EE_inner is the fraction of a point source's flux the Moffat model
        (this galaxy's own fwhm_kpc, see psf_fwhm_kpc_for_galaxy) predicts
        lands inside that bin's radius. Chosen over a fixed physical
        aperture per Austin's 2026-08-04 call: it matches what the stack
        code already does (measure.measure_psf_corrected_core_luminosity),
        the bins are already fixed-kpc-width so every galaxy's core is the
        same physical size, and it keeps this function's zone definition
        consistent with measure_onehalo_luminosity's.
      | 'psf_weighted' -- the core term is instead core.extract_core_spectra's
        PSF-weighted optimal extraction for this one galaxy (a JOIN, see
        _psf_weighted_core_luminosity), which is already an estimate of the
        point source's TOTAL flux -- no EE_inner rescale applied to it.
        Needs vdfi loaded with build_psf=True.
      | 'both' -- core_flux/total_flux/halo_fraction are still anchored on
        the 'aperture' path (so results are directly comparable across the
        catalog regardless of core_method), and the PSF-weighted estimate is
        ALSO computed purely to populate core_extendedness (the ratio of the
        two independent core estimators) -- a bonus diagnostic, per Part A3.

    The leak problem (Part A2, the correctness-critical part): PSF-correcting
    the core bin asserts "this much flux leaked OUT of the core aperture."
    That leaked flux is still sitting, uncorrected, in the outer bins this
    function calls the halo zone -- so `total = core_corrected + halo_raw`
    double-counts it and biases halo_fraction low. This function instead
    subtracts, from EACH halo-zone bin, `leak_frac_j * F_true`, where
    `leak_frac_j = EE(r_hi,j) - EE(r_lo,j)` is the SAME per-galaxy Moffat
    model's prediction of how much of the point source's flux lands in that
    bin, and F_true is this function's best single estimate of the point
    source's total flux (core_flux under 'aperture'/'both' -- note
    core_flux/EE_inner and raw_core_flux*leak_frac_j/EE_inner are the exact
    same quantity algebraically, so this is exactly measure.measure_psf_
    corrected_core_onehalo_luminosity's own per-bin leak formula, just
    applied to one galaxy instead of a bootstrap-restacked sample; under
    'psf_weighted', F_true is the PSF-weighted core flux directly, since
    that estimator already targets the total). Mirrors measure_psf_
    corrected_core_onehalo_luminosity's model; see its docstring for the
    full derivation.

    boundary_kpc : the core/two-halo crossover radius. None -> config.
        core_boundary_kpc (a FITTED stack-level number -- see that field's
        own docstring). The value actually used is always returned as
        boundary_kpc_used, so a result stays self-describing even if the
        default later drifts from the fit that produced it.

    twohalo_outer_kpc : outer edge of the two-halo zone (bins with r_mid
        between boundary_kpc and this). None -> the outermost bin edge, i.e.
        use everything available. Returned as twohalo_outer_used.

    A NOTE ON FIXED APERTURES (2026-08-07): every zone boundary here is a
    single number applied identically to every galaxy -- boundary_kpc is a
    stack-level fitted constant, the bin edges are fixed kpc, and sn_floor
    only ever sets an advisory flag. That is deliberate and load-bearing for
    the per-galaxy-statistic-into-regression workflow: any S/N-dependent or
    adaptive aperture would make the statistic's bias track brightness ->
    mass -> overdensity, and the regression would then rediscover the
    aperture rule as an "environmental" result. Do not add adaptive sizing
    here without understanding that this is what it breaks. The one genuinely
    per-galaxy quantity in the chain is the PSF (psf_fwhm_used, 'measured'),
    which is an observing condition rather than a galaxy property -- it is
    returned so it can be carried as a nuisance regressor.

    sn_floor : None (default, no flagging) -- pass a value to populate
        flag_low_sn (True when sn_core is below it or non-finite). This is
        advisory only: rows are never dropped here. Per the module's S/N
        reality check, whatever floor an analysis actually uses is a STATED
        sample cut in the paper, not a silent filter -- this function just
        makes the flag available to apply that cut downstream.

    No bootstrap: a single galaxy can't be resampled over galaxies (same
    convention as measure_single_galaxy). Every *_err below is propagated
    photon noise ONLY (in quadrature across the halo-zone bins, plus the
    leak term's own contribution from the core's error) -- it does NOT
    separate real object-to-object scatter from measurement noise. That
    conflation is a STATED CAVEAT (2026-08-04, Austin's call on Open
    Question 4: no injection-recovery deconvolution pass for now) rather
    than something this function tries to fix; revisit with injection.py's
    machinery if the halo_fraction distribution's width ends up mattering.

    Returns dict: core_flux, halo_flux, total_flux (all erg/s, area-scaled
    true luminosities, leak-aware so they never double-count), *_err for
    each, halo_fraction, core_fraction (= 1 - halo_fraction; both bounded
    0-1 by construction since they're fractions of the same total), sn_core,
    sn_halo, core_extendedness (NaN unless core_method != 'aperture'),
    psf_fwhm_used, psf_beta_used, psf_provenance, boundary_kpc_used,
    core_bin_index, R_inner_kpc, EE_inner, zone_mask, raw_core_flux,
    raw_halo_flux (both PSF-uncorrected, for a sanity check against the
    corrected values), twohalo_flux/_err, sn_twohalo, raw_twohalo_flux,
    twohalo_mask, twohalo_outer_used, nfib_core/_halo/_twohalo, flag_low_sn,
    sn_floor, z, gid, index, core_method, unit_info. Plus
    psf_weighted_core_flux/_err when core_method != 'aperture'.
    """
    id_col = id_col or config.id_col
    i = resolve_galaxy_index(product, gid=gid, index=index, id_col=id_col)

    if core_method not in ("aperture", "psf_weighted", "both"):
        raise ValueError("core_method must be 'aperture', 'psf_weighted', or "
                         f"'both' (got {core_method!r})")

    if config.flux_unit != "L_kpc2":
        warnings.warn(
            f"measure_single_galaxy_core_halo: config.flux_unit={config.flux_unit!r}, "
            "not 'L_kpc2' -- the leak-aware area scaling (per-bin surface "
            "brightness times that bin's own kpc^2 area) assumes L_kpc2, "
            "matching measure.measure_twohalo_luminosity's / measure_psf_"
            "corrected_core_luminosity's own convention. Results are only "
            "physically meaningful under flux_unit='L_kpc2'.")

    sg = single_galaxy_spectra(config, product, gid=gid, index=index, id_col=id_col)
    rest_wave = np.asarray(sg["rest_wave"])
    flux_all = np.asarray(sg["stacks"]["galaxy"]["flux"])
    err_all  = np.asarray(sg["stacks"]["galaxy"]["err"])
    r_edges  = np.asarray(sg["r_edges"], dtype=float)
    nfib_bin = np.asarray(sg["nfib"], dtype=float)
    z = float(sg["z"])
    nrad = flux_all.shape[0]

    if str(sg.get("bin_mode", "")).lower() != "kpc":
        warnings.warn(
            f"measure_single_galaxy_core_halo: bin_mode={sg.get('bin_mode')!r}, "
            "not 'kpc' -- r_edges is assumed to already be in kpc for the "
            "leak-aware area scaling and the PSF encircled-energy model. "
            "Results are only physically meaningful under bin_mode='kpc' "
            "(see lya-halos-binning-is-kpc-not-virial: production runs kpc "
            "bins even though config.py's own default is stale).")

    if not (0 <= core_bin_index < nrad - 1):
        raise ValueError(f"core_bin_index={core_bin_index} leaves no bins "
                         f"beyond it for {nrad} radial bins.")

    boundary_kpc_used = float(boundary_kpc if boundary_kpc is not None
                              else config.core_boundary_kpc)

    b_bounds = tuple(bounds) if bounds is not None else tuple(config.line_window)
    cb = cont_bounds if cont_bounds is not None else config.cont_bounds
    cm = cont_method if cont_method is not None else config.cont_method
    co = cont_order if cont_order is not None else config.cont_order

    psf_used = psf or config.psf
    ra = dec = None
    if psf_used.provenance == "measured" or core_method in ("psf_weighted", "both"):
        ra = float(np.asarray(product.catalog[ra_col])[i])
        dec = float(np.asarray(product.catalog[dec_col])[i])

    psf_info = psf_fwhm_kpc_for_galaxy(
        config, z, vdfi=vdfi, ra=ra, dec=dec,
        seeing_radius_arcsec=core_radius, psf=psf_used)
    fwhm_kpc = psf_info["fwhm_kpc"]
    beta = psf_info["beta"]

    # ---- per-bin integrated flux + error: a surface-brightness value
    # (L_kpc2), exactly like total_flux_fid in the stack code -- see
    # integrated_line_flux's own docstring for the blue+red-side equivalence.
    flux_sum = np.full(nrad, np.nan)
    err_sum  = np.full(nrad, np.nan)
    for r in range(nrad):
        res = integrated_line_flux(rest_wave, flux_all[r], err=err_all[r],
                                   bounds=b_bounds, cont_bounds=cb,
                                   cont_method=cm, cont_order=co,
                                   clip_negative=clip_negative)
        if res["success"]:
            flux_sum[r] = res["flux_integral"]   # x dlambda (2026-09-17; was the pixel sum)
            err_sum[r]  = res["err_integral"]

    # per-bin annulus area, kpc^2 (r_lo=0 for the core bin -> full circle --
    # the r_lo=0 special case of the same annulus formula every other bin
    # uses, matching measure._annulus_area_kpc2's convention exactly).
    area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    raw_lum = flux_sum * area           # PSF-uncorrected true luminosity, erg/s
    raw_lum_err = err_sum * area

    R_inner_kpc = float(r_edges[core_bin_index + 1])
    from . import fitting
    EE_inner = float(np.clip(fitting.moffat_encircled_energy_fraction(
        R_inner_kpc, fwhm_kpc, beta=beta), 1e-6, 1.0))

    raw_core_flux = float(raw_lum[core_bin_index])
    raw_core_flux_err = float(raw_lum_err[core_bin_index])

    psf_weighted_core_flux = psf_weighted_core_flux_err = None
    if core_method in ("psf_weighted", "both"):
        if vdfi is None:
            raise RuntimeError(
                f"core_method={core_method!r} needs vdfi (loaded with "
                "build_psf=True via core.load_field_with_psf) for the "
                "PSF-weighted core extraction join.")
        psfw = _psf_weighted_core_luminosity(
            config, product, i, vdfi=vdfi, core_radius=core_radius,
            bounds=b_bounds, cont_bounds=cb, cont_method=cm, cont_order=co,
            clip_negative=clip_negative, ra_col=ra_col, dec_col=dec_col)
        psf_weighted_core_flux = psfw["core_flux"]
        psf_weighted_core_flux_err = psfw["core_flux_err"]

    if core_method == "psf_weighted":
        core_flux = psf_weighted_core_flux
        core_flux_err = psf_weighted_core_flux_err
    else:   # 'aperture' or 'both' -- both anchor total/halo_fraction on the aperture path
        core_flux = raw_core_flux / EE_inner
        core_flux_err = raw_core_flux_err / EE_inner

    core_extendedness = (raw_core_flux / psf_weighted_core_flux
                         if (psf_weighted_core_flux not in (None, 0)
                             and np.isfinite(psf_weighted_core_flux))
                         else np.nan)

    # ---- halo zone: bins strictly beyond the core bin, out to boundary_kpc,
    # leak-corrected against F_true = core_flux (see docstring derivation:
    # this is algebraically identical to measure_psf_corrected_core_onehalo_
    # luminosity's raw_core * leak_frac/EE_inner formula under 'aperture').
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    zone_mask = (np.arange(nrad) > core_bin_index) & (r_mid <= boundary_kpc_used)
    if not np.any(zone_mask):
        raise ValueError(
            f"measure_single_galaxy_core_halo: no radial bins strictly "
            f"between core_bin_index={core_bin_index} (outer edge "
            f"{r_edges[core_bin_index + 1]:.4g} kpc) and "
            f"boundary_kpc={boundary_kpc_used:.4g} -- nothing to call the "
            "halo zone for this binning. Check r_edges / boundary_kpc.")

    def _zone_sum(mask):
        """Leak-aware sum of raw_lum over `mask`, plus its propagated error.

        Quadrature: each bin's own raw error, plus the leak term's error
        propagated from core_flux_err. Ignores the covariance between zone
        bins induced by sharing the same subtracted core term (all leak
        terms move together, fully correlated, when core_flux moves) -- a
        stated approximation consistent with this function's photon-noise-
        only convention (see "No bootstrap" above); the leak correction is
        a small fraction of most zone bins' own flux at r >> R_inner, where
        the Moffat's power-law wings have little power left, so this
        undercounts the true error only marginally for typical bins --
        worth revisiting if the innermost halo bin ever dominates the total
        error budget. (For the two-halo zone the leak term is utterly
        negligible -- ~1e-4 of the point source at r > 70 kpc -- but it is
        applied anyway so both zones use ONE formula.)
        """
        tot = 0.0
        var = 0.0
        for j in np.flatnonzero(mask):
            leak_frac = float(
                fitting.moffat_encircled_energy_fraction(r_edges[j + 1], fwhm_kpc, beta=beta)
                - fitting.moffat_encircled_energy_fraction(r_edges[j], fwhm_kpc, beta=beta))
            tot += raw_lum[j] - leak_frac * core_flux
            var += raw_lum_err[j] ** 2 + (leak_frac * core_flux_err) ** 2
        return float(tot), float(np.sqrt(var))

    halo_flux, halo_flux_err = _zone_sum(zone_mask)

    # ---- two-halo zone: everything beyond boundary_kpc, out to
    # twohalo_outer_kpc (None -> the outermost bin edge available). Measured
    # with the SAME leak-aware machinery as the halo zone so the three zones
    # are one consistent decomposition, not three conventions.
    #
    # INTERPRETATION WARNING (2026-08-07, kept deliberately): per galaxy this
    # is NOT a smooth halo property. The two-halo term is the summed light of
    # whatever discrete neighbours happen to land in the annulus, so its
    # galaxy-to-galaxy scatter is dominated by Poisson companion counts --
    # heavily right-skewed, a few objects carrying most of the signal. Treat
    # twohalo_flux as material for a RANK-level test against overdensity
    # (Spearman / median regression, bootstrap errors), not as an OLS
    # response, and do not read its sigma_int as "halo-to-halo variation".
    twohalo_outer_used = float(twohalo_outer_kpc if twohalo_outer_kpc is not None
                               else r_edges[-1])
    twohalo_mask = (r_mid > boundary_kpc_used) & (r_mid <= twohalo_outer_used)
    if np.any(twohalo_mask):
        twohalo_flux, twohalo_flux_err = _zone_sum(twohalo_mask)
        sn_twohalo = (twohalo_flux / twohalo_flux_err
                      if twohalo_flux_err else np.nan)
        raw_twohalo_flux = float(np.nansum(raw_lum[twohalo_mask]))
    else:
        twohalo_flux = twohalo_flux_err = sn_twohalo = np.nan
        raw_twohalo_flux = np.nan

    total_flux = core_flux + halo_flux
    total_flux_err = float(np.sqrt(core_flux_err ** 2 + halo_flux_err ** 2))
    halo_fraction = halo_flux / total_flux if total_flux else np.nan
    core_fraction = core_flux / total_flux if total_flux else np.nan

    sn_core = core_flux / core_flux_err if core_flux_err else np.nan
    sn_halo = halo_flux / halo_flux_err if halo_flux_err else np.nan
    flag_low_sn = bool(sn_floor is not None
                       and (not np.isfinite(sn_core) or sn_core < sn_floor))

    out = {
        "core_flux": core_flux, "core_flux_err": core_flux_err,
        "halo_flux": halo_flux, "halo_flux_err": halo_flux_err,
        "total_flux": total_flux, "total_flux_err": total_flux_err,
        "halo_fraction": halo_fraction, "core_fraction": core_fraction,
        "sn_core": sn_core, "sn_halo": sn_halo,
        "core_extendedness": core_extendedness,
        "psf_fwhm_used": psf_info["fwhm_arcsec_used"],
        "psf_beta_used": beta, "psf_provenance": psf_info["provenance"],
        "boundary_kpc_used": boundary_kpc_used,
        "core_bin_index": core_bin_index, "R_inner_kpc": R_inner_kpc,
        "EE_inner": EE_inner, "zone_mask": zone_mask,
        "raw_core_flux": raw_core_flux,
        "raw_halo_flux": float(np.nansum(raw_lum[zone_mask])),
        # two-halo zone (see the INTERPRETATION WARNING at its computation)
        "twohalo_flux": twohalo_flux, "twohalo_flux_err": twohalo_flux_err,
        "sn_twohalo": sn_twohalo, "raw_twohalo_flux": raw_twohalo_flux,
        "twohalo_mask": twohalo_mask,
        "twohalo_outer_used": twohalo_outer_used,
        # fiber coverage per zone -- NOT a physical quantity: it tracks where
        # this galaxy landed relative to the dither pattern / field edges, so
        # it sets sigma_i and can smuggle field position into a regression if
        # left out. Carry as a nuisance regressor.
        "nfib_core": float(nfib_bin[core_bin_index]),
        "nfib_halo": float(np.nansum(nfib_bin[zone_mask])),
        "nfib_twohalo": (float(np.nansum(nfib_bin[twohalo_mask]))
                         if np.any(twohalo_mask) else np.nan),
        "flag_low_sn": flag_low_sn, "sn_floor": sn_floor,
        "z": z, "gid": sg.get("gid"), "index": i,
        "core_method": core_method,
        "unit_info": sg.get("unit_info"),
    }
    if core_method in ("psf_weighted", "both"):
        out["psf_weighted_core_flux"] = psf_weighted_core_flux
        out["psf_weighted_core_flux_err"] = psf_weighted_core_flux_err
    return out


# =====================================================================
# 5b.  PART A, BATCHED -- run Part A over a whole catalog into one table,
#      plus the variance-budget diagnostic that says whether the resulting
#      per-galaxy statistic carries any real galaxy-to-galaxy information.
#
# Why this exists (2026-08-07): the full-profile fit (h1, r_c, the 1-halo ->
# 2-halo transition) needs 100-200 galaxies per stack, so it can only ever be
# evaluated on a handful of hand-chosen splits. The alternative is a per-
# galaxy summary statistic that is individually noisy but UNBIASED, regressed
# against every catalog property at once -- the noise averages down over N,
# and the surviving predictors then tell you which 2-3 splits are worth
# spending stacks on. This block produces the table that feeds that
# regression, and the go/no-go test that runs before it.
# =====================================================================
_CORE_HALO_ARRAY_KEYS = ("zone_mask", "twohalo_mask", "unit_info")


def _key_unique(cat, cols) -> bool:
    """Is `cols` a unique row key on `cat`? Thin wrapper on multicat's own key
    builder so the batch runner and the catalog joins agree bit-for-bit on
    what counts as a key (including its NUL-separator collision guard)."""
    from .multicat import _id_key
    k = _id_key(cat, list(cols)).astype(str)
    return np.unique(k).size == len(k)


def measure_catalog_core_halo(
    config: "PipelineConfig", product: "GalaxyProduct", *,
    gids=None, indices=None, id_col=None,
    join_catalog=True, catalog_cols=None,
    save_path: str = None, progress: bool = True,
    **kwargs,
) -> "Table":
    """
    Run measure_single_galaxy_core_halo over a whole catalog -> one Table.

    One row per galaxy, one column per scalar the per-galaxy function
    returns (the array/dict fields -- zone_mask, twohalo_mask, unit_info --
    are dropped; fiber_area_kpc2 is pulled out of unit_info as a scalar).

    gids / indices : restrict to a subset (pass EITHER). None for both ->
        every row of product.catalog.

        `gids` is matched on a UNIQUE key, not on a bare ID. 3D-HST ID
        numbers restart per field, so on a combined COSMOS+EGS product a
        bare ID genuinely collides (ID=6875 is two different galaxies) --
        matching on it would not error, it would quietly pair each AEGIS
        galaxy with its COSMOS namesake. So:
          - id_col=None (default) picks the first of ('FIELD', config.id_col)
            then (config.id_col,) that exists and is UNIQUE, and says which
            it chose. Same candidate order as multicat._DEFAULT_ID_CANDIDATES.
          - an explicit id_col (str or tuple) is honoured, but still refused
            if it is not unique.
        When the key is composite, pass `gids` as the sub-catalog TABLE
        itself (e.g. gids=catalog_cos), not a bare ID list -- the table
        carries FIELD, so the match is unambiguous. A list of tuples also
        works. A flat list of bare IDs is rejected with an explanation
        rather than guessed at.

    join_catalog : True (default) -> horizontally join the matched rows of
        product.catalog onto the result, so mass / z / SFR / overdensity /
        EW / dust / inclination / delta_v sit alongside core_flux and
        halo_flux in one table ready for a regression. catalog_cols
        restricts which columns come across (None -> all of them). Any name
        that would collide with a measurement column is suffixed '_cat'.

    save_path : optional path; the table is written there (FITS by format
        inference) once the loop finishes. A 400-galaxy pass is not free --
        write it once and regress off the file.

    **kwargs are forwarded verbatim to measure_single_galaxy_core_halo
    (core_method, boundary_kpc, twohalo_outer_kpc, vdfi, psf, sn_floor, ...).

    FAILURES ARE NOT DROPPED. A galaxy that raises gets a row with `ok=False`
    and `error` set to the exception text, all measurements NaN. This is
    deliberate: silently skipping the objects that fail is exactly the
    brightness-dependent selection that would poison a downstream
    regression, so the failures stay visible and countable. Check
    `t['ok'].sum()` and look at WHY the failures failed before cutting them.

    Likewise note that no S/N cut is applied here. sn_floor only ever sets
    the advisory flag_low_sn column; keeping low-S/N and NEGATIVE flux rows
    is REQUIRED for the regression to be unbiased (a negative halo_flux is a
    correct noise draw around a small positive truth, not a bad measurement).

    Returns astropy Table with, in addition to the per-galaxy keys:
        ok       : bool, did the measurement succeed
        error    : str, exception text ('' when ok)
        gid      : the bare catalog id (filled here, since the per-galaxy
                   function is called by index= and returns gid=None)
        join_key : the full composite key actually used -- keep this, it is
                   what lets a saved table be re-joined unambiguously later
    """
    from astropy.table import Table, hstack

    from .multicat import _as_key_list, _id_key

    cat = product.catalog
    ngal = len(cat)

    if gids is not None and indices is not None:
        raise ValueError("pass gids= or indices=, not both.")

    # ---- join key. A bare ID is NOT survey-wide unique: 3D-HST numbers
    # restart per field, so a combined COSMOS+EGS product has real ID
    # collisions (e.g. ID=6875 exists in both). Same problem multicat.py
    # already solved -- reuse its key machinery and its default candidate
    # order rather than inventing a second convention here. With id_col=None
    # we escalate to ('FIELD', id) automatically when the bare id is not
    # unique; an explicit id_col is honoured as given (and still guarded).
    if id_col is None:
        for cand in (("FIELD", config.id_col), (config.id_col,)):
            if all(c in cat.colnames for c in cand) and _key_unique(cat, cand):
                key_cols = list(cand)
                break
        else:
            from .multicat import _id_key as _ik
            k = _ik(cat, [config.id_col]).astype(str)
            _, cnt = np.unique(k, return_counts=True)
            raise ValueError(
                f"measure_catalog_core_halo: no unique join key on this "
                f"product. Bare {config.id_col!r} has {int(np.sum(cnt > 1))} "
                f"duplicated value(s) over {len(cat)} rows, and "
                f"('FIELD', {config.id_col!r}) is "
                + ("also not unique" if "FIELD" in cat.colnames
                   else "unavailable (no 'FIELD' column)")
                + ". Pass id_col= explicitly with a column that separates the "
                  f"fields. Catalog columns: {list(cat.colnames)}")
        if len(key_cols) > 1:
            print(f"measure_catalog_core_halo: bare {config.id_col!r} is not "
                  f"unique on this product -- using composite key "
                  f"{tuple(key_cols)}.")
    else:
        key_cols = _as_key_list(id_col)
        missing_cols = [c for c in key_cols if c not in cat.colnames]
        if missing_cols:
            raise KeyError(f"id_col columns {missing_cols} not in catalog "
                           f"{list(cat.colnames)}")
        if not _key_unique(cat, key_cols):
            raise ValueError(
                f"measure_catalog_core_halo: id_col={tuple(key_cols)} is NOT "
                "unique on this product -- rows would be silently mismatched. "
                "Widen the key, e.g. id_col=('FIELD','ID').")

    prod_key = _id_key(cat, key_cols).astype(str)
    scalar_id_col = key_cols[-1]

    if gids is not None:
        lookup = {k: i for i, k in enumerate(prod_key)}
        if hasattr(gids, "colnames"):            # a Table / sub-catalog
            miss = [c for c in key_cols if c not in gids.colnames]
            if miss:
                raise KeyError(
                    f"gids table is missing key column(s) {miss}. It needs "
                    f"{tuple(key_cols)} to be matched unambiguously -- pass "
                    "the sub-catalog itself, not just its ID column.")
            want = _id_key(gids, key_cols).astype(str)
        else:
            arr = list(np.atleast_1d(gids))
            if len(key_cols) == 1:
                want = _id_key({key_cols[0]: np.asarray(arr, dtype=object)},
                               key_cols).astype(str)
            else:
                try:
                    cols = {c: np.asarray([r[j] for r in arr], dtype=object)
                            for j, c in enumerate(key_cols)}
                except (TypeError, IndexError):
                    raise ValueError(
                        f"the join key is composite {tuple(key_cols)}, so gids "
                        "must be a Table carrying those columns, or a list of "
                        f"{len(key_cols)}-tuples -- a flat list of bare IDs is "
                        "ambiguous across fields and is exactly the mismatch "
                        "the key guard exists to prevent.") from None
                want = _id_key(cols, key_cols).astype(str)
        absent = sorted({k for k in want if k not in lookup})
        if absent:
            raise KeyError(
                f"{len(absent)} of {len(want)} requested galaxies are not in "
                f"this product under key {tuple(key_cols)}; first few: "
                f"{absent[:5]}")
        idx = [lookup[k] for k in want]
    elif indices is not None:
        idx = [int(k) for k in np.atleast_1d(indices)]
    else:
        idx = list(range(ngal))

    n = len(idx)
    rows = []
    for k, i in enumerate(idx):
        if progress and (k % 25 == 0 or k == n - 1):
            print(f"\rmeasure_catalog_core_halo: {k + 1}/{n}", end="", flush=True)
        try:
            res = measure_single_galaxy_core_halo(
                config, product, index=i, id_col=scalar_id_col, **kwargs)
            ui = res.get("unit_info") or {}
            fa = ui.get("fiber_area_kpc2", np.nan)
            row = {kk: vv for kk, vv in res.items()
                   if kk not in _CORE_HALO_ARRAY_KEYS}
            row["fiber_area_kpc2"] = float(np.ravel(fa)[0]) if np.size(fa) else np.nan
            row["ok"] = True
            row["error"] = ""
        except Exception as e:                      # noqa: BLE001 -- see docstring
            row = {"index": i, "ok": False,
                   "error": f"{type(e).__name__}: {e}"}
        # measure_single_galaxy_core_halo is called by index=, so its own 'gid'
        # comes back None -- fill both the bare id and the full composite key
        # from the catalog here, so a saved table can always be re-joined
        # unambiguously even after the fields are pooled.
        row["gid"] = str(np.asarray(cat[scalar_id_col])[i])
        row["join_key"] = prod_key[i]
        rows.append(row)
    if progress:
        print()

    # union of keys across rows (failures carry fewer), NaN-filled
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    cols = {}
    for kk in keys:
        vals = [r.get(kk, None) for r in rows]
        if kk in ("gid", "join_key", "error", "core_method", "psf_provenance"):
            cols[kk] = np.array(["" if v is None else str(v) for v in vals])
        elif kk in ("ok", "flag_low_sn"):
            cols[kk] = np.array([bool(v) if v is not None else False for v in vals])
        elif kk in ("index", "core_bin_index"):
            cols[kk] = np.array([-1 if v is None else int(v) for v in vals])
        else:
            def _f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return np.nan
            cols[kk] = np.array([_f(v) for v in vals], dtype=float)
    t = Table(cols)

    if join_catalog:
        sub = cat[np.asarray(idx, dtype=int)]
        if catalog_cols is not None:
            sub = sub[list(catalog_cols)]
        sub = Table(sub, copy=True)
        for c in list(sub.colnames):
            if c in t.colnames:
                sub.rename_column(c, c + "_cat")
        t = hstack([t, sub])

    t.meta["n_requested"] = n
    t.meta["n_ok"] = int(np.sum(t["ok"]))
    for kk, vv in kwargs.items():
        if isinstance(vv, (int, float, str, bool, type(None))):
            t.meta[f"arg_{kk}"] = "" if vv is None else vv

    if save_path:
        t.write(save_path, overwrite=True)
        print(f"wrote {save_path}  ({t.meta['n_ok']}/{n} ok)")
    return t


def variance_budget(y, yerr, *, name: str = "statistic", n_sigma_target: float = 3.0,
                    reference=None, verbose: bool = True) -> dict:
    """
    Does a per-galaxy statistic carry ANY real galaxy-to-galaxy signal?

    This is the go/no-go test that should run before any regression or any
    stack split, because it screens BOTH. The observed scatter decomposes as

        Var(y) = sigma_int^2 + <sigma_i^2>

    where sigma_int is the real object-to-object variation and <sigma_i^2>
    is the mean squared measurement error. If sigma_int^2 comes back
    consistent with zero, every galaxy has -- within your ability to tell --
    the same halo, and then no regression finds a trend AND no split on
    anything shows a difference, since a split just compares means of
    subsets of a population with no real spread. That is the whole value of
    running this first: it costs one line and can retire the entire program
    branch before any stacks get built.

    Also reports what a sample this size can actually detect. A correlation
    coefficient is detectable at n_sigma when r_obs >~ n_sigma/sqrt(N), but
    measurement noise dilutes the correlation you would measure:

        r_obs = r_true * sqrt( sigma_int^2 / (sigma_int^2 + sigma_noise^2) )

    so `r_true_needed` = (n_sigma/sqrt(N)) / dilution is the honest bar the
    underlying astrophysical relation has to clear. If that comes back above
    ~0.7 the statistic is too noisy to regress on, and the answer is a
    better statistic (or a wider zone), not more predictors.

    reference : the SAME statistic measured on a null -- an off-band (line-
        free) window, or a z-scrambled run. Pass a variance_budget dict, or
        a (y, yerr) tuple which is budgeted internally. This is not optional
        in spirit, only in signature.

        WHY (learned the hard way 2026-08-07, on the two-halo zone): Var_int
        measures excess scatter, FULL STOP. It cannot tell real object-to-
        object astrophysics from per-galaxy sky residuals, and a systematic
        that varies galaxy to galaxy sails through every check above -- the
        two-halo zone returned sigma_int/sigma_noise = 12.6 and a confident
        "usable", and then the off-band null returned a LARGER sigma_int
        (6.1e45 vs 4.6e45) in a window with no line in it. All of that
        scatter was systematics. With `reference` set, this function
        subtracts the null's Var_int and reports f_sys, the fraction of the
        excess scatter the null already accounts for; f_sys near or above 1
        means there is nothing left to regress against no matter how
        significant the raw excess looked.

        Note this is specifically about SCATTER. A statistic whose per-galaxy
        variance is systematics-dominated can still have a perfectly good
        STACKED mean, if the mean on-band level sits well above the null's --
        check the two separately and do not let one condemn the other.

    NaNs are dropped pairwise. Pass the statistic in LINEAR flux with
    negatives intact -- a positivity cut or a log makes Var(y) meaningless
    here and biased downstream.

    Returns dict: n, var_obs, var_noise, var_int, sigma_int, sigma_noise,
    ratio (sigma_int/sigma_noise), chi2, dof, chi2_per_dof, excess_sigma
    (significance of the excess scatter, (chi2-dof)/sqrt(2*dof)), dilution,
    r_detectable (n_sigma/sqrt(N)), r_true_needed, weighted_mean. With
    `reference`: also var_int_null, f_sys, var_int_signal, sigma_int_signal,
    and dilution / r_true_needed recomputed against the SIGNAL variance
    (the raw ones are kept as *_raw).
    """
    y = np.asarray(y, dtype=float)
    e = np.asarray(yerr, dtype=float)
    m = np.isfinite(y) & np.isfinite(e) & (e > 0)
    y, e = y[m], e[m]
    n = y.size
    if n < 3:
        raise ValueError(f"variance_budget: only {n} finite rows -- nothing to do.")

    var_obs = float(np.var(y, ddof=1))
    var_noise = float(np.mean(e ** 2))
    var_int = var_obs - var_noise

    w = 1.0 / e ** 2
    ybar = float(np.sum(w * y) / np.sum(w))
    chi2 = float(np.sum(((y - ybar) / e) ** 2))
    dof = n - 1
    excess_sigma = (chi2 - dof) / np.sqrt(2.0 * dof)

    dilution = float(np.sqrt(max(var_int, 0.0) / var_obs)) if var_obs > 0 else 0.0
    r_det = float(n_sigma_target / np.sqrt(n))
    r_true_needed = r_det / dilution if dilution > 0 else np.inf

    out = {
        "name": name, "n": n,
        "var_obs": var_obs, "var_noise": var_noise, "var_int": var_int,
        "sigma_int": float(np.sqrt(max(var_int, 0.0))),
        "sigma_noise": float(np.sqrt(var_noise)),
        "ratio": (float(np.sqrt(max(var_int, 0.0) / var_noise))
                  if var_noise > 0 else np.nan),
        "chi2": chi2, "dof": dof, "chi2_per_dof": chi2 / dof,
        "excess_sigma": float(excess_sigma),
        "dilution": dilution,
        "r_detectable": r_det, "r_true_needed": float(r_true_needed),
        "weighted_mean": ybar,
    }

    # ---- null subtraction (see `reference` in the docstring)
    if reference is not None:
        if isinstance(reference, dict):
            ref = reference
        else:
            ry, re_ = reference
            ref = variance_budget(ry, re_, name=f"{name} [null]",
                                  n_sigma_target=n_sigma_target, verbose=False)
        var_int_null = float(ref["var_int"])
        var_int_signal = var_int - var_int_null
        f_sys = (var_int_null / var_int) if var_int > 0 else np.inf
        dil_sig = (float(np.sqrt(max(var_int_signal, 0.0) / var_obs))
                   if var_obs > 0 else 0.0)
        out.update({
            "var_int_null": var_int_null,
            "sigma_int_null": float(np.sqrt(max(var_int_null, 0.0))),
            "f_sys": float(f_sys),
            "var_int_signal": var_int_signal,
            "sigma_int_signal": float(np.sqrt(max(var_int_signal, 0.0))),
            "dilution_raw": out["dilution"],
            "r_true_needed_raw": out["r_true_needed"],
            "dilution": dil_sig,
            "r_true_needed": (r_det / dil_sig) if dil_sig > 0 else np.inf,
            "null_weighted_mean": float(ref["weighted_mean"]),
        })
        # Stacked-level comparison as a DIFFERENCE in sigma, not a ratio.
        # (mean/null_mean was the first version and it was wrong for exactly
        # the reason ratios are banned elsewhere in this workflow: the null
        # mean sits near zero with its own error, so the ratio blows up and
        # can even come out negative. A difference over the quadrature error
        # is stable whatever the null level is.) Errors use the OBSERVED
        # scatter, sqrt(Var_obs/n), not the photon errors -- when systematics
        # dominate, the photon error badly understates the mean's uncertainty.
        sem = np.sqrt(var_obs / n)
        sem_ref = np.sqrt(ref["var_obs"] / ref["n"])
        d = ybar - ref["weighted_mean"]
        out["mean_excess"] = float(d)
        out["mean_excess_sigma"] = float(d / np.hypot(sem, sem_ref))

    if verbose:
        print(f"--- variance budget: {name}  (N = {n}) ---")
        print(f"  Var(y)            = {var_obs:.4g}")
        print(f"  <sigma_i^2>       = {var_noise:.4g}")
        print(f"  Var_int           = {var_int:.4g}"
              f"   -> sigma_int/sigma_noise = {out['ratio']:.3f}")
        print(f"  chi2/dof          = {out['chi2_per_dof']:.3f}  (dof = {dof})"
              f"   -> excess scatter at {excess_sigma:.1f} sigma")
        if reference is not None:
            print(f"  Var_int (null)    = {out['var_int_null']:.4g}"
                  f"   -> f_sys = {out['f_sys']:.2f}"
                  " (fraction of the excess the null already explains)")
            print(f"  Var_int (signal)  = {out['var_int_signal']:.4g}")
            print(f"  mean - null mean  = {out['mean_excess']:+.4g}"
                  f"  ({out['mean_excess_sigma']:+.1f} sigma)"
                  "   <- the STACKED level, judged separately from the scatter")
        print(f"  dilution factor   = {out['dilution']:.3f}")
        print(f"  detectable r_obs  = {r_det:.3f}  ({n_sigma_target:g} sigma)"
              f"   -> need r_true > {out['r_true_needed']:.3f}")
        # Verdict thresholds, calibrated on synthetic data 2026-08-07. NOTE the
        # asymmetry, which cost one wrong version of this function: the excess-
        # scatter test and the regression are NOT equally sensitive. This test
        # has to find excess variance against a sqrt(2*dof) noise floor with no
        # idea where the signal lives; a regression against a specific x
        # projects onto one axis and is MORE powerful. In a planted case with
        # sigma_int/sigma_noise = 0.46 the excess registered at only 2.9 sigma
        # (i.e. "not significant") while a planted r_true = 0.5 came through at
        # r_obs = 0.20 against a 0.15 threshold -- detectable. So a marginal
        # excess is NOT a stop signal, only <1 sigma is.
        if reference is not None and out["f_sys"] >= 0.7:
            print(f"  VERDICT: SCATTER IS SYSTEMATICS. The null reproduces "
                  f"{out['f_sys']*100:.0f}% of the excess scatter, so there is "
                  "no per-galaxy signal to regress against however significant "
                  "the raw excess looked. Report a limit, not a measurement.")
            if out["mean_excess_sigma"] > 3:
                print(f"           NOTE the STACKED mean still sits "
                      f"{out['mean_excess_sigma']:.1f} sigma above the null "
                      "level -- the coadded signal is unaffected by this. It "
                      "is only the per-galaxy scatter that is dead.")
        elif excess_sigma < 1:
            print("  VERDICT: DEAD. No excess scatter at all -- the observed "
                  "spread is fully explained by measurement noise. Nothing to "
                  "regress against, and no split on any variable will show a "
                  "difference either. (Or the sigma_i are overestimated -- "
                  "check this statistic on the CORE first, where chi2/dof "
                  "should come out >> 1.)")
        elif excess_sigma < 3:
            print("  VERDICT: MARGINAL excess scatter. Do not stop here -- the "
                  "regression is more sensitive than this test (it knows which "
                  "axis to look along). Proceed, but expect only the single "
                  "strongest predictor to clear the bar above.")
        elif r_true_needed > 0.7:
            print("  VERDICT: real scatter, but too diluted -- the underlying "
                  "relation would have to be implausibly tight to show up. "
                  "Widen the zone or improve the statistic before regressing.")
        else:
            print("  VERDICT: usable. Proceed to the multivariate regression.")
    return out


# =====================================================================
# 6.  PART C -- 2D narrowband image, per galaxy (specs/single-galaxy-
#     pipeline.md Part C). Works directly off vdfi (the loaded fiber
#     field), NOT the Stage-1 binned product -- no azimuthal-symmetry
#     assumption anywhere in this chain, which is the whole point (Part C
#     is how an asymmetry, companion, or disturbed morphology would ever
#     be visible; the radial-binned path imposes symmetry by construction).
#
# Two judgment calls made here that the spec left open, documented rather
# than asked about since they extend principles the spec already states:
#   - radius_arcsec (field of view / fiber-selection radius) defaults to
#     config.bg_outer_arcsec: beyond that is, by this pipeline's OWN
#     background-annulus definition, "background" (see lya-halos-bg-
#     annulus-contains-no-signal), so it's an already-justified, free
#     stopping point rather than an arbitrary new number.
#   - grid_method='voronoi' is accepted as a value but raises immediately:
#     Part C3 says it's DISPLAY-only, but building that display helper is
#     genuinely optional scope (Part C's own success criterion only needs
#     the gaussian gridder + PSF contour), so it's deferred rather than
#     built speculatively. Flag if you want it.
# =====================================================================
def _galaxy_fiber_photometry(config: "PipelineConfig", vdfi: dict,
                             ra: float, dec: float, z: float,
                             wave_bounds_rest, *, bad_fiber_mask=None,
                             radius_arcsec: float) -> dict:
    """
    Per-fiber, background-subtracted, line-window-collapsed brightness for
    ONE galaxy, pooled across all exposures -- the raw material for
    galaxy_image's gridder (Part C2).

    wave_bounds_rest : (lo, hi) rest-frame Angstrom -- converted to this
        galaxy's OWN observed window via *(1+z) here (Part C1: never accept
        a fixed observed-frame window, it isn't stackable across z).

    Background: the SAME [bg_inner, bg_outer) annulus used everywhere else
    in this pipeline, per-exposure with a galaxy-pooled fallback for
    exposures that have no annulus fibers of their own -- this block is
    copied verbatim from core.extract_core_spectra's own background step
    (which itself copies extract.extract_fibers'), per Part C1's "no new
    background definition" instruction: core, halo, and image all share
    ONE background definition.

    Per fiber: collapse the background-subtracted spectrum to
    nansum(flux) / nansum(err**2) over the observed window -- a plain sum,
    matching integrated_line_flux's own convention, but with NO separate
    per-fiber continuum fit (the annulus background subtraction already
    removes the smooth sky/continuum floor; a second per-fiber continuum
    model would be pure noise at one-fiber S/N). Values are scaled by
    config.flux_unit_scale (the same first conversion step convert_avg_
    fiber_bin applies) but NOT rest-framed or area-normalized -- Part C is
    a qualitative diagnostic image, not the luminosity number (that's Part
    A/D's job); flag if you want full physical-unit conversion here too.

    Returns dict: dra, ddec (arcsec, per fiber), value, var (per fiber,
    flux_unit_scale-corrected), exp (exposure index), radius_arcsec,
    wave_bounds_obs. Raises if no fiber falls within radius_arcsec in any
    exposure.
    """
    from .extract import combine_fibers
    from .background import smooth_interp_then_gauss, smooth_spectrum_nan_safe_adaptive

    RA = vdfi["RA"]; DEC = vdfi["DEC"]
    wave = np.asarray(vdfi["wave"], dtype=float)
    spectra = vdfi["spectra"]; error = vdfi["error"]
    nexp, nfib_tot = RA.shape

    wlo = float(wave_bounds_rest[0]) * (1.0 + z)
    whi = float(wave_bounds_rest[1]) * (1.0 + z)
    win = (wave >= wlo) & (wave <= whi)
    if not np.any(win):
        raise ValueError(f"observed window [{wlo:.2f},{whi:.2f}] A has no "
                         f"overlap with vdfi['wave'] -- check z / wave_bounds_rest.")

    bg_method = config.resolved_bg_combine_method().lower()
    bg_needs_err = bg_method in ("mean", "inv_var")
    sc_sigma = float(getattr(config, "sigma_clip_sigma", 3.0))
    sc_maxiters = getattr(config, "sigma_clip_maxiters", 5)

    def _maybe_smooth(spec_1d):
        if not config.smooth_bg:
            return spec_1d
        method = getattr(config, "bg_smooth_method", "interp_gauss").lower()
        if method == "interp_gauss":
            return smooth_interp_then_gauss(
                spec_1d, wave, z, lya_rest=config.LYA_REST,
                sigma_A=config.bg_smooth_sigma_A,
                mask_halfwidth_A=config.line_mask_halfwidth_A)
        sv = config.smoothing_values
        return smooth_spectrum_nan_safe_adaptive(
            spec_1d, wave, sigma_A=sv[0], edge_sigma_A=sv[1], edge_width_A=sv[2])

    # ---- pass 1: per-exposure geometry + own annulus bg, if any ----
    dist_all = np.full((nexp, nfib_tot), np.nan)
    dra_all  = np.full((nexp, nfib_tot), np.nan)
    ddec_all = np.full((nexp, nfib_tot), np.nan)
    good_all = np.zeros((nexp, nfib_tot), dtype=bool)
    bg_per_exp = np.full((nexp, wave.size), np.nan)
    has_own_bg = np.zeros(nexp, dtype=bool)
    all_bg_fibers, all_bg_errs = [], []
    for e in range(nexp):
        dra  = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
        ddec = (DEC[e] - dec) * 3600.
        dra_all[e] = dra; ddec_all[e] = ddec
        dist_all[e] = np.sqrt(dra ** 2 + ddec ** 2)
        good = np.isfinite(RA[e]) & np.isfinite(DEC[e])
        if bad_fiber_mask is not None:
            good = good & (~np.asarray(bad_fiber_mask[e]))
        good_all[e] = good

        bg_sel = ((dist_all[e] >= config.bg_inner_arcsec) &
                  (dist_all[e] <  config.bg_outer_arcsec) & good)
        n_bg = int(np.sum(bg_sel))
        if n_bg == 0:
            continue
        all_bg_fibers.append(spectra[e, bg_sel])
        if bg_needs_err:
            all_bg_errs.append(error[e, bg_sel])
        if n_bg >= config.min_bg_fibers:
            er_bg = error[e, bg_sel] if bg_needs_err else None
            bg_e, _ = combine_fibers(spectra[e, bg_sel], er_bg, method=bg_method,
                                     sigma=sc_sigma, maxiters=sc_maxiters)
            bg_per_exp[e] = _maybe_smooth(bg_e)
            has_own_bg[e] = True

    if all_bg_fibers:
        sp_all = np.concatenate(all_bg_fibers, axis=0)
        er_all = np.concatenate(all_bg_errs, axis=0) if bg_needs_err else None
        bg_gal, _ = combine_fibers(sp_all, er_all, method=bg_method,
                                   sigma=sc_sigma, maxiters=sc_maxiters)
        bg_gal = _maybe_smooth(bg_gal)
    else:
        bg_gal = np.full(wave.size, np.nan)
    for e in range(nexp):
        if not has_own_bg[e]:
            bg_per_exp[e] = bg_gal

    # ---- pass 2: science fibers within radius_arcsec, bg-subtract, collapse ----
    dra_l, ddec_l, val_l, var_l, exp_l = [], [], [], [], []
    scale = float(config.flux_unit_scale)
    for e in range(nexp):
        sel = good_all[e] & (dist_all[e] <= radius_arcsec)
        nsel = int(np.sum(sel))
        if nsel == 0:
            continue
        sp = spectra[e, sel].astype(float) - bg_per_exp[e][np.newaxis, :]
        er = error[e, sel].astype(float)
        val = np.nansum(sp[:, win], axis=1) * scale
        var = np.nansum(er[:, win] ** 2, axis=1) * scale ** 2
        dra_l.append(dra_all[e, sel]); ddec_l.append(ddec_all[e, sel])
        val_l.append(val); var_l.append(var)
        exp_l.append(np.full(nsel, e))

    if not dra_l:
        raise RuntimeError(
            f"_galaxy_fiber_photometry: no good fiber within {radius_arcsec}\" "
            f"of (ra={ra}, dec={dec}) in any exposure.")

    return dict(
        dra=np.concatenate(dra_l), ddec=np.concatenate(ddec_l),
        value=np.concatenate(val_l), var=np.concatenate(var_l),
        exp=np.concatenate(exp_l), radius_arcsec=radius_arcsec,
        wave_bounds_obs=(wlo, whi),
    )


def _gaussian_grid(x_kpc, y_kpc, value, var, *, h: float, pixel_size_kpc: float,
                   extent_kpc: float, n_eff_min: float = 1.0) -> dict:
    """
    Fixed-kernel, inverse-variance-weighted Gaussian gridder (Part C3):

        w_jk    = K(|x_j - x_k| ; h) / sigma_k^2      K = Gaussian, FIXED width h
        I_j     = sum_k w_jk f_k / sum_k w_jk
        Var_j   = sum_k w_jk^2 sigma_k^2 / (sum_k w_jk)^2
        N_eff,j = (sum_k w_jk)^2 / sum_k w_jk^2

    Kernel truncated at 3h via a cKDTree ball query (sub-second per galaxy).

    h, pixel_size_kpc : REQUIRED, no default (Part C3) -- start h near one
        fiber radius in kpc and pixel_size_kpc at <~ 1/3 the fiber diameter
        in kpc (VIRUS fibers are 1.5", ~12 kpc at z~2.5, so ~3-4 kpc pixels
        avoids aliasing), then tune by hand. Smaller h -> better resolution,
        noisier, more coverage holes.
    n_eff_min : pixels with fewer effective fibers than this are blanked
        (NaN) -- otherwise an edge pixel supported by a single noisy fiber
        renders as a solid, misleadingly confident patch. Default 1.0 (any
        real coverage counts); raise it to demand more redundancy.

    Returns dict: image, var_map, n_eff_map (each (ny,nx)), x_edges,
    y_edges, x_centers, y_centers (kpc), h, pixel_size_kpc, n_eff_min.
    """
    from scipy.spatial import cKDTree

    x_kpc = np.asarray(x_kpc, dtype=float); y_kpc = np.asarray(y_kpc, dtype=float)
    value = np.asarray(value, dtype=float); var = np.asarray(var, dtype=float)
    good = (np.isfinite(x_kpc) & np.isfinite(y_kpc) & np.isfinite(value)
           & np.isfinite(var) & (var > 0))
    x_kpc, y_kpc, value, var = x_kpc[good], y_kpc[good], value[good], var[good]

    edges = np.arange(-extent_kpc, extent_kpc + pixel_size_kpc, pixel_size_kpc)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = centers.size
    xx, yy = np.meshgrid(centers, centers)
    pix_xy = np.column_stack([xx.ravel(), yy.ravel()])

    image = np.full(pix_xy.shape[0], np.nan)
    var_map = np.full(pix_xy.shape[0], np.nan)
    n_eff_map = np.zeros(pix_xy.shape[0])

    if x_kpc.size:
        tree = cKDTree(np.column_stack([x_kpc, y_kpc]))
        pairs = tree.query_ball_point(pix_xy, r=3.0 * h)
        inv_var = 1.0 / var
        for k, idx in enumerate(pairs):
            if not idx:
                continue
            idx = np.asarray(idx)
            d2 = (x_kpc[idx] - pix_xy[k, 0]) ** 2 + (y_kpc[idx] - pix_xy[k, 1]) ** 2
            w = np.exp(-0.5 * d2 / h ** 2) * inv_var[idx]
            sw = np.sum(w)
            if sw <= 0:
                continue
            image[k] = np.sum(w * value[idx]) / sw
            var_map[k] = np.sum(w ** 2 * var[idx]) / sw ** 2
            n_eff_map[k] = sw ** 2 / np.sum(w ** 2)

    blank = n_eff_map < n_eff_min
    image[blank] = np.nan
    var_map[blank] = np.nan

    return dict(
        image=image.reshape(n, n), var_map=var_map.reshape(n, n),
        n_eff_map=n_eff_map.reshape(n, n),
        x_edges=edges, y_edges=edges, x_centers=centers, y_centers=centers,
        h=float(h), pixel_size_kpc=float(pixel_size_kpc), n_eff_min=float(n_eff_min),
    )


def galaxy_image(config: "PipelineConfig", vdfi: dict, row, *,
                 h: float, pixel_size_kpc: float,
                 radius_arcsec: float = None, extent_kpc: float = None,
                 grid_method: str = "gaussian",
                 n_eff_min: float = 1.0, bad_fiber_mask=None,
                 psf: "PSFSpec" = None, ra_col: str = "RA", dec_col: str = "DEC",
                 line_window=None, cont_bounds=None) -> dict:
    """
    2D narrowband image of ONE galaxy's Lya line, plus its off-band
    companion, on a shared pixel grid (Part C).

        vdfi = extract.load_field(cfg)                # cell 1, run once
        img  = single.galaxy_image(cfg, vdfi, row)     # cell 2+, per galaxy

    row : a single-row astropy Table/Row carrying ra_col/dec_col/
        config.z_col -- e.g. product.catalog[i:i+1]. This does NOT resolve
        a gid/index against a Stage-1 BINNED product (that's Part A's job);
        imaging works off the raw fiber field, so pass the catalog row
        directly.
    vdfi : the loaded field dict (extract.load_field(config), run ONCE by
        the caller -- see this module's docstring). Needed regardless of
        PSFSpec provenance (unlike Part A's 'fiducial' shortcut) since the
        image itself is built from vdfi's fibers.
    h, pixel_size_kpc : REQUIRED (Part C3) -- see _gaussian_grid.
    radius_arcsec : fiber-selection field-of-view radius (which fibers are
        PULLED). None (default) -> config.bg_outer_arcsec -- see this
        section's header comment for why that's a natural, already-
        justified default rather than an arbitrary new number.
    extent_kpc : output GRID half-width, kpc. None (default) -> derived from
        radius_arcsec via THIS galaxy's own kpc_per_arcsec(z) (Part C's
        original single-galaxy behavior). Pass an explicit value when
        stacking multiple galaxies (single.stack_galaxy_images does this for
        you): different galaxies sit at different z, so radius_arcsec alone
        gives each one a slightly different PHYSICAL grid extent/shape --
        an explicit extent_kpc forces every galaxy onto the exact same
        (x_edges, y_edges), which stacking needs to combine pixel-for-pixel
        with no resampling. This does NOT change which fibers are pulled
        (still radius_arcsec-controlled) -- only the output grid's size.
    grid_method : 'gaussian' (default, the only implemented MEASUREMENT
        gridder) | 'voronoi' -- raises immediately; see this section's
        header comment.
    bad_fiber_mask : optional pre-built (nexp, nfib) boolean mask. None
        (default) -> built here via extract.build_bad_fiber_mask, which is
        EXPENSIVE (a segmentation-mask lookup over the whole field) and
        prints a diagnostic block. If you're calling galaxy_image in your
        own loop over several galaxies, build it ONCE yourself and pass it
        in every call -- stack_galaxy_images already does this for you.
    line_window : rest-frame (lo, hi) Angstrom. None -> config.line_window
        (Part C1 -- never a fixed observed-frame window).
    cont_bounds : (blue, red) rest-frame sidebands. None -> config.
        cont_bounds -- the SAME sidebands the line measurement already
        uses. Both sidebands are pooled into ONE off-band image (Part C2's
        built-in null: if the off-band image shows the same blob, the blob
        isn't Lya), width-normalized to match the on-band image's own
        per-Angstrom scale so the shared color scale is meaningful.

    Returns dict: on_band, off_band (each a _gaussian_grid output), psf
    (this galaxy's psf_fwhm_kpc_for_galaxy result -- the contour source for
    plot_galaxy_image), fiber_xy_kpc (this galaxy's on-band fiber positions
    in kpc, for plot_psf_as_image), ra, dec, z, radius_arcsec,
    kpc_per_arcsec, n_fibers_on, line_window, cont_bounds.
    """
    if grid_method != "gaussian":
        raise ValueError(
            f"grid_method={grid_method!r} not supported by galaxy_image -- "
            "'voronoi' is DISPLAY-only (Part C3) and has no measurement "
            "implementation in this module yet; flag if you want it built.")

    ra = float(np.asarray(row[ra_col])[0])
    dec = float(np.asarray(row[dec_col])[0])
    z = float(np.asarray(row[config.z_col])[0])

    radius_arcsec = float(radius_arcsec if radius_arcsec is not None
                          else config.bg_outer_arcsec)
    lw = tuple(line_window if line_window is not None else config.line_window)
    cb = cont_bounds if cont_bounds is not None else config.cont_bounds
    blue_bounds, red_bounds = tuple(cb[0]), tuple(cb[1])

    kpc_per_arcsec = float(cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265)
    extent_kpc = float(extent_kpc if extent_kpc is not None else radius_arcsec * kpc_per_arcsec)

    if bad_fiber_mask is None:
        from .extract import build_bad_fiber_mask
        bad_fiber_mask = build_bad_fiber_mask(config, vdfi)

    on = _galaxy_fiber_photometry(config, vdfi, ra, dec, z, lw,
                                  bad_fiber_mask=bad_fiber_mask,
                                  radius_arcsec=radius_arcsec)
    blue = _galaxy_fiber_photometry(config, vdfi, ra, dec, z, blue_bounds,
                                    bad_fiber_mask=bad_fiber_mask,
                                    radius_arcsec=radius_arcsec)
    red = _galaxy_fiber_photometry(config, vdfi, ra, dec, z, red_bounds,
                                   bad_fiber_mask=bad_fiber_mask,
                                   radius_arcsec=radius_arcsec)

    lw_width = float(lw[1] - lw[0]) * (1.0 + z)
    blue_width = float(blue_bounds[1] - blue_bounds[0]) * (1.0 + z)
    red_width = float(red_bounds[1] - red_bounds[0]) * (1.0 + z)

    x_on = on["dra"] * kpc_per_arcsec
    y_on = on["ddec"] * kpc_per_arcsec
    on_grid = _gaussian_grid(x_on, y_on, on["value"] / lw_width, on["var"] / lw_width ** 2,
                             h=h, pixel_size_kpc=pixel_size_kpc,
                             extent_kpc=extent_kpc, n_eff_min=n_eff_min)

    x_off = np.concatenate([blue["dra"], red["dra"]]) * kpc_per_arcsec
    y_off = np.concatenate([blue["ddec"], red["ddec"]]) * kpc_per_arcsec
    val_off = np.concatenate([blue["value"] / blue_width, red["value"] / red_width])
    var_off = np.concatenate([blue["var"] / blue_width ** 2, red["var"] / red_width ** 2])
    off_grid = _gaussian_grid(x_off, y_off, val_off, var_off,
                              h=h, pixel_size_kpc=pixel_size_kpc,
                              extent_kpc=extent_kpc, n_eff_min=n_eff_min)

    psf_info = psf_fwhm_kpc_for_galaxy(
        config, z, vdfi=vdfi, ra=ra, dec=dec, bad_fiber_mask=bad_fiber_mask,
        seeing_radius_arcsec=min(radius_arcsec, 3.5), psf=psf)

    return dict(
        on_band=on_grid, off_band=off_grid, psf=psf_info,
        fiber_xy_kpc=(x_on, y_on),
        ra=ra, dec=dec, z=z, radius_arcsec=radius_arcsec,
        kpc_per_arcsec=kpc_per_arcsec, n_fibers_on=int(on["dra"].size),
        line_window=lw, cont_bounds=(blue_bounds, red_bounds),
    )


def plot_galaxy_image(img: dict, *, show_psf: bool = True, show_n_eff: bool = False,
                      cmap: str = "inferno", figsize=None, title: str = None):
    """
    Part C5 display: on-band + off-band images side by side on a SHARED
    color scale (the built-in null -- a blob that survives into the
    off-band panel isn't Lya). PSF contour overlaid on BOTH panels by
    default (show_psf=False to disable) -- any claim that something in the
    image is extended is only readable against it. N_eff panel on request.

    img : the dict from galaxy_image. Returns (fig, axs).
    """
    import matplotlib.pyplot as plt
    from .fitting import moffat_1d

    on = img["on_band"]; off = img["off_band"]
    finite = np.concatenate([on["image"][np.isfinite(on["image"])],
                             off["image"][np.isfinite(off["image"])]])
    vmin, vmax = (tuple(np.nanpercentile(finite, [1, 99])) if finite.size else (0.0, 1.0))

    ncols = 2 + int(show_n_eff)
    figsize = figsize or (4.6 * ncols, 4.3)
    fig, axs = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)
    axs = np.atleast_1d(axs)

    extent = [on["x_edges"][0], on["x_edges"][-1], on["y_edges"][0], on["y_edges"][-1]]
    xx, yy = np.meshgrid(on["x_centers"], on["y_centers"])
    psf_img = None
    if show_psf:
        fwhm_kpc = img["psf"]["fwhm_kpc"]; beta = img["psf"]["beta"]
        psf_img = moffat_1d(np.sqrt(xx ** 2 + yy ** 2), fwhm=fwhm_kpc, beta=beta)
        psf_img = psf_img / np.nanmax(psf_img)

    im = None
    for ax, grid, panel_title in ((axs[0], on, "on-band (Lyα)"), (axs[1], off, "off-band")):
        im = ax.imshow(grid["image"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent)
        ax.set_title(panel_title); ax.set_xlabel("kpc"); ax.set_ylabel("kpc")
        if psf_img is not None:
            ax.contour(on["x_centers"], on["y_centers"], psf_img,
                      levels=[0.1, 0.5, 0.9], colors="cyan", linewidths=0.8, alpha=0.85)
    fig.colorbar(im, ax=list(axs[:2]), shrink=0.85,
                label="surface brightness [flux_unit_scale x config.flux_unit / Å]")

    if show_n_eff:
        imN = axs[2].imshow(on["n_eff_map"], origin="lower", cmap="viridis", extent=extent)
        axs[2].set_title(r"N$_{\rm eff}$ (on-band)"); axs[2].set_xlabel("kpc")
        fig.colorbar(imN, ax=axs[2], shrink=0.85, label="effective fiber count")

    if title is None:
        title = (rf"z={img['z']:.3f}   PSF FWHM={img['psf']['fwhm_arcsec_used']:.2f}\" "
                rf"({img['psf']['provenance']})   h={on['h']:.1f} kpc")
    fig.suptitle(title, y=1.06)
    return fig, axs


def plot_psf_as_image(img: dict, *, cmap: str = "inferno", figsize=(5, 4.3), ax=None):
    """
    Part C4's 'useful half' of the empirical-PSF idea -- a DIAGNOSTIC plot,
    not a required pipeline stage. Renders THIS galaxy's own analytic PSF
    (img['psf']) through the SAME gridder and the SAME fiber positions as
    its on-band image, so the coverage gaps / sampling irregularity a
    smooth analytic curve can't show become visible: feed a perfect point
    source (uniform nominal variance -- this is about the SAMPLING pattern,
    not real photon noise) through the identical fiber positions/kernel/
    pixel grid used for img['on_band'].

    Returns (fig, ax, grid) where grid is the _gaussian_grid output.
    """
    import matplotlib.pyplot as plt

    fwhm_kpc = img["psf"]["fwhm_kpc"]; beta = img["psf"]["beta"]
    x_kpc, y_kpc = img["fiber_xy_kpc"]
    r = np.sqrt(x_kpc ** 2 + y_kpc ** 2)

    from .fitting import moffat_1d
    value = moffat_1d(r, fwhm=fwhm_kpc, beta=beta)
    var = np.ones_like(value)

    on = img["on_band"]
    grid = _gaussian_grid(x_kpc, y_kpc, value, var, h=on["h"],
                          pixel_size_kpc=on["pixel_size_kpc"],
                          extent_kpc=img["radius_arcsec"] * img["kpc_per_arcsec"],
                          n_eff_min=on["n_eff_min"])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    im = ax.imshow(grid["image"], origin="lower", cmap=cmap,
                   extent=[grid["x_edges"][0], grid["x_edges"][-1],
                           grid["y_edges"][0], grid["y_edges"][-1]])
    ax.set_xlabel("kpc"); ax.set_ylabel("kpc")
    ax.set_title("Point-source PSF through this galaxy's own fiber sampling")
    fig.colorbar(im, ax=ax, label="normalized PSF (fiber-sampled)")
    return fig, ax, grid


# =====================================================================
# 7.  PART D -- image stacking (specs/single-galaxy-pipeline.md Part D).
#     "Once C works, this is mostly free" -- and it's true: the combine
#     itself is a JOIN to measure.stack_galaxies (a (ngal, ny, nx) image
#     cube combines along axis=0 exactly like its native (ngal, nrad,
#     nwave) case; nothing in that function is wavelength-specific), not a
#     reimplementation. The individual-galaxy driver rule from Part A
#     ("don't wrap the field load, don't hide it inside a batch driver")
#     does NOT forbid a stacking driver here: vdfi is still an explicit
#     argument, loaded once by the caller and passed in -- this loop is the
#     Part-A-style thin per-galaxy loop, just living in this module instead
#     of the notebook because stacking inherently needs the whole sample in
#     hand at once (same precedent as core.extract_core_spectra's own
#     run_table loop).
# =====================================================================
def _rotate_image_pair(image, var_map, angle_deg: float):
    """
    Rotate one galaxy's (image, var_map) by angle_deg about the array
    center, for the random-position-angle null (Part D). NaN-safe: NaN
    pixels are filled with 0 before the spline rotation (which would
    otherwise bleed NaN into every neighboring pixel), and a validity mask
    is rotated separately with nearest-neighbor order and used to re-blank
    the result -- so the rotated image has the same "shape" of missing
    coverage, just reoriented, rather than NaN-smeared edges.
    """
    from scipy.ndimage import rotate as _ndi_rotate

    image = np.asarray(image, dtype=float)
    var_map = np.asarray(var_map, dtype=float)
    valid = np.isfinite(image) & np.isfinite(var_map)

    img_filled = np.where(valid, image, 0.0)
    var_filled = np.where(valid, var_map, 0.0)

    img_rot = _ndi_rotate(img_filled, angle_deg, reshape=False, order=1,
                          mode="constant", cval=0.0)
    var_rot = _ndi_rotate(var_filled, angle_deg, reshape=False, order=1,
                          mode="constant", cval=0.0)
    valid_rot = _ndi_rotate(valid.astype(float), angle_deg, reshape=False,
                            order=0, mode="constant", cval=0.0) > 0.5

    img_rot[~valid_rot] = np.nan
    var_rot[~valid_rot] = np.nan
    return img_rot, var_rot


def stack_galaxy_images(
    config: "PipelineConfig", vdfi: dict, rows, *,
    h: float, pixel_size_kpc: float, extent_kpc: float,
    radius_arcsec: float = None, method="both",
    bad_fiber_mask=None, psf: "PSFSpec" = None,
    random_rotate: bool = False, seed: int = None,
    ra_col: str = "RA", dec_col: str = "DEC",
    line_window=None, cont_bounds=None,
    verbose: bool = True,
) -> dict:
    """
    Part D -- stack per-galaxy kpc-gridded images (median AND inverse-
    variance mean, per spec) across `rows`, on ONE common physical grid.

    rows : any subset of a catalog Table (product.catalog, a run_table, or
        a pre-cut subset from splits.py's recipes -- "subset stacks reusing
        the existing splits recipes" is exactly this: cut `rows` however
        you like before calling, no new plumbing needed here). Iterated as
        rows[k:k+1] internally, one galaxy_image call each.
    h, pixel_size_kpc, extent_kpc : passed straight to every galaxy_image
        call -- extent_kpc is REQUIRED here (unlike galaxy_image's own
        optional override) because every galaxy MUST share the same output
        grid for the combine below to be a plain pixel-for-pixel stack with
        no resampling.
    method : 'median' | 'inv_var' | 'both' (default) -- both are cheap once
        the cube is built, and Part D asks for both.
    random_rotate : False (default, the REAL stack) -- True rotates each
        galaxy's image by an INDEPENDENT random position angle (0-360,
        `seed`-controlled) before stacking, for Part D's null: "anything
        that survives random rotation is not real orientation-dependent
        structure." Call this function twice (False, then True with a few
        different seeds) and compare quadrant_asymmetry/image_ellipticity
        between them -- this function does not loop over null draws itself
        (same "thin loop belongs to the caller" convention as Part A).

    Returns dict: stacks -- {'median': {'on_band': {image, err}, 'off_band':
    {...}}, 'inv_var': {...}} (only the requested method(s)), x_centers,
    y_centers, x_edges, y_edges (kpc, shared by every stack), ngal_used,
    n_failed (galaxies galaxy_image raised on -- e.g. no fiber in radius --
    skipped, not silently zero-filled), h, pixel_size_kpc, extent_kpc,
    random_rotate, rotation_angles_deg (None unless random_rotate), psf (the
    psf_fwhm_kpc_for_galaxy result for the FIRST successfully-imaged galaxy,
    for a representative contour -- the sample's PSF varies galaxy to
    galaxy under provenance='measured', so this is a display convenience,
    not a claim that every galaxy shares one PSF).
    """
    n = len(rows)
    rng = np.random.default_rng(seed) if random_rotate else None
    angles = rng.uniform(0.0, 360.0, size=n) if random_rotate else None

    # Build the bad-fiber mask ONCE, not per galaxy: build_bad_fiber_mask is
    # expensive (segmentation-mask lookup over the whole field) and prints a
    # diagnostic block every call. galaxy_image would otherwise rebuild it
    # from scratch (and re-print those diagnostics) on every single
    # iteration of the loop below -- the actual cause of a "50s/it, spammed
    # every galaxy" run. Same fix core.extract_core_spectra already applies
    # for its own per-galaxy loop.
    if bad_fiber_mask is None:
        from .extract import build_bad_fiber_mask
        bad_fiber_mask = build_bad_fiber_mask(config, vdfi)

    on_imgs, on_vars, off_imgs, off_vars = [], [], [], []
    psf_rep = None
    n_failed = 0
    it = range(n)
    if verbose:
        try:
            from tqdm.auto import tqdm as _tqdm
            it = _tqdm(it, desc="stack_galaxy_images")
        except Exception:
            pass

    for k in it:
        row = rows[k:k + 1]
        try:
            img = galaxy_image(
                config, vdfi, row, h=h, pixel_size_kpc=pixel_size_kpc,
                radius_arcsec=radius_arcsec, extent_kpc=extent_kpc,
                bad_fiber_mask=bad_fiber_mask, psf=psf,
                ra_col=ra_col, dec_col=dec_col,
                line_window=line_window, cont_bounds=cont_bounds,
            )
        except (RuntimeError, ValueError) as exc:
            n_failed += 1
            if verbose:
                print(f"  [skip] row {k}: {exc}")
            continue

        on_im, on_va = img["on_band"]["image"], img["on_band"]["var_map"]
        off_im, off_va = img["off_band"]["image"], img["off_band"]["var_map"]
        if random_rotate:
            on_im, on_va = _rotate_image_pair(on_im, on_va, float(angles[k]))
            off_im, off_va = _rotate_image_pair(off_im, off_va, float(angles[k]))

        on_imgs.append(on_im); on_vars.append(on_va)
        off_imgs.append(off_im); off_vars.append(off_va)
        if psf_rep is None:
            psf_rep = img["psf"]

    if not on_imgs:
        raise RuntimeError("stack_galaxy_images: every galaxy in `rows` failed "
                           "(see the [skip] messages above) -- nothing to stack.")

    on_cube = np.stack(on_imgs, axis=0)
    on_var_cube = np.stack(on_vars, axis=0)
    off_cube = np.stack(off_imgs, axis=0)
    off_var_cube = np.stack(off_vars, axis=0)

    methods = ["median", "inv_var"] if method == "both" else [method]
    for m in methods:
        if m not in ("median", "inv_var"):
            raise ValueError(f"method must be 'median', 'inv_var', or 'both' (got {m!r})")

    from .measure import stack_galaxies as _stack_galaxies
    stacks = {}
    for m in methods:
        on_stack, on_err = _stack_galaxies(on_cube, np.sqrt(on_var_cube), method=m)
        off_stack, off_err = _stack_galaxies(off_cube, np.sqrt(off_var_cube), method=m)
        stacks[m] = {
            "on_band": {"image": on_stack, "err": on_err},
            "off_band": {"image": off_stack, "err": off_err},
        }

    edges = np.arange(-extent_kpc, extent_kpc + pixel_size_kpc, pixel_size_kpc)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        stacks=stacks, x_centers=centers, y_centers=centers,
        x_edges=edges, y_edges=edges,
        ngal_used=int(on_cube.shape[0]), n_failed=int(n_failed),
        h=float(h), pixel_size_kpc=float(pixel_size_kpc), extent_kpc=float(extent_kpc),
        random_rotate=bool(random_rotate),
        rotation_angles_deg=(angles.tolist() if angles is not None else None),
        psf=psf_rep,
    )


def azimuthal_profile(image, x_centers, y_centers, r_edges_kpc, *,
                      var_map=None, center=(0.0, 0.0)) -> dict:
    """
    Azimuthally-averaged radial profile of a 2D (stacked) image, binned into
    the SAME r_edges (kpc) the 1D radial pipeline uses -- this is the
    function validate_stack_vs_1d_profile calls, and Part D's gating check
    (agreement with the existing 1D profile) is only meaningful if the bin
    edges actually match, so pass product.r_edges / stacks['r_edges']
    directly rather than re-deriving them.

    var_map : per-pixel variance -- if given, each bin is the inverse-
        variance weighted mean of its pixels; if None, a plain nanmean.

    Returns dict: r_mid, profile, profile_err, npix (all len(r_edges)-1),
    r_edges.
    """
    x_centers = np.asarray(x_centers, dtype=float)
    y_centers = np.asarray(y_centers, dtype=float)
    image = np.asarray(image, dtype=float)
    r_edges = np.asarray(r_edges_kpc, dtype=float)
    xx, yy = np.meshgrid(x_centers - center[0], y_centers - center[1])
    rr = np.sqrt(xx ** 2 + yy ** 2)

    nbin = r_edges.size - 1
    profile = np.full(nbin, np.nan)
    profile_err = np.full(nbin, np.nan)
    npix = np.zeros(nbin, dtype=int)
    var_map = np.asarray(var_map, dtype=float) if var_map is not None else None

    for b in range(nbin):
        sel = (rr >= r_edges[b]) & (rr < r_edges[b + 1]) & np.isfinite(image)
        if var_map is not None:
            sel = sel & np.isfinite(var_map) & (var_map > 0)
        npix[b] = int(np.sum(sel))
        if npix[b] == 0:
            continue
        if var_map is not None:
            w = 1.0 / var_map[sel]
            sw = np.sum(w)
            profile[b] = np.sum(image[sel] * w) / sw
            profile_err[b] = np.sqrt(1.0 / sw)
        else:
            profile[b] = np.nanmean(image[sel])
            profile_err[b] = np.nanstd(image[sel]) / np.sqrt(npix[b])

    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    return dict(r_mid=r_mid, profile=profile, profile_err=profile_err,
               npix=npix, r_edges=r_edges)


def quadrant_asymmetry(image, x_centers, y_centers, *, center=(0.0, 0.0)) -> dict:
    """
    Part D symmetry diagnostic: total flux (nansum) in each of the 4
    quadrants about `center`, plus a simple asymmetry statistic
    (max-min)/mean across the 4 quadrant totals -- 0 for a perfectly
    symmetric image, growing with genuine lopsidedness. Compare this
    statistic between the real stack and several random-position-angle null
    stacks (stack_galaxy_images(..., random_rotate=True)) to judge whether
    any asymmetry is significant.

    Returns dict: NE, NW, SE, SW (quadrant totals), asymmetry.
    """
    x_centers = np.asarray(x_centers, dtype=float) - center[0]
    y_centers = np.asarray(y_centers, dtype=float) - center[1]
    image = np.asarray(image, dtype=float)
    xx, yy = np.meshgrid(x_centers, y_centers)

    quads = {
        "NE": np.nansum(image[(xx >= 0) & (yy >= 0)]),
        "NW": np.nansum(image[(xx < 0) & (yy >= 0)]),
        "SW": np.nansum(image[(xx < 0) & (yy < 0)]),
        "SE": np.nansum(image[(xx >= 0) & (yy < 0)]),
    }
    vals = np.array(list(quads.values()))
    mean = np.mean(vals)
    asymmetry = float((vals.max() - vals.min()) / mean) if mean else np.nan
    return dict(**quads, asymmetry=asymmetry)


def image_ellipticity(image, x_centers, y_centers, *, center=(0.0, 0.0),
                      clip_negative: bool = True) -> dict:
    """
    Part D symmetry diagnostic: intensity-weighted second moments of the
    image -> axis ratio q (0-1, 1=circular) and position angle (deg, from
    +x axis, standard atan2(2*Ixy, Ixx-Iyy)/2 convention).

    clip_negative : zero negative pixels before weighting (default True --
        negative noise excursions otherwise distort the moments, same
        rationale as measure.integrated_line_flux's own clip_negative).

    Returns dict: q, pa_deg, Ixx, Iyy, Ixy, x_centroid, y_centroid (all kpc/
    dimensionless; centroid is the intensity-weighted center actually used,
    which may differ slightly from `center` if the light isn't perfectly
    centered -- a real diagnostic in itself).
    """
    x_centers = np.asarray(x_centers, dtype=float)
    y_centers = np.asarray(y_centers, dtype=float)
    image = np.asarray(image, dtype=float)
    xx, yy = np.meshgrid(x_centers, y_centers)

    w = np.where(np.isfinite(image), image, 0.0)
    if clip_negative:
        w = np.where(w > 0, w, 0.0)
    tot = np.sum(w)
    if tot <= 0:
        return dict(q=np.nan, pa_deg=np.nan, Ixx=np.nan, Iyy=np.nan, Ixy=np.nan,
                   x_centroid=np.nan, y_centroid=np.nan)

    xc = np.sum(w * xx) / tot
    yc = np.sum(w * yy) / tot
    dx = xx - xc; dy = yy - yc
    Ixx = np.sum(w * dx ** 2) / tot
    Iyy = np.sum(w * dy ** 2) / tot
    Ixy = np.sum(w * dx * dy) / tot

    trace = Ixx + Iyy
    diff = Ixx - Iyy
    disc = np.sqrt(diff ** 2 + 4 * Ixy ** 2)
    lam1 = 0.5 * (trace + disc)   # major-axis variance
    lam2 = 0.5 * (trace - disc)   # minor-axis variance
    q = float(np.sqrt(lam2 / lam1)) if lam1 > 0 else np.nan
    pa_deg = float(np.degrees(0.5 * np.arctan2(2 * Ixy, diff)))

    return dict(q=q, pa_deg=pa_deg, Ixx=float(Ixx), Iyy=float(Iyy), Ixy=float(Ixy),
               x_centroid=float(xc), y_centroid=float(yc))


def validate_stack_vs_1d_profile(
    stack_result: dict, r_edges_kpc, profile_1d, profile_1d_err=None, *,
    method: str = "inv_var", band: str = "on_band", normalize: str = "peak",
) -> dict:
    """
    Part D's GATING check: does the stacked image's azimuthally-averaged
    radial profile reproduce the existing 1D radial profile? Fiber -> grid
    -> azimuthal average (this module) and fiber -> radial bin -> combine
    (stack.py/measure.py) are genuinely independent code paths, so
    agreement is a real cross-check on both; disagreement means something
    in the gridder is wrong and nothing downstream (Part C's images, this
    stack) is trustworthy.

    UNIT CAVEAT, read before trusting an absolute-flux comparison: this
    module's images are in flux_unit_scale-corrected, per-Angstrom OBSERVED
    units (see galaxy_image's docstring) -- they are NOT rest-framed or
    area-normalized into the L_kpc2 surface-brightness convention
    total_flux_fid/the 1D profile uses. This project has a real history of
    exactly this kind of area/normalization mismatch causing silent bugs
    (the 2026-07-22 PSF area-convention bug; see memory). Rather than guess
    at the reconciliation, normalize='peak' (default) divides BOTH profiles
    by their own peak value before comparing -- i.e. this checks the SHAPE
    only, not the absolute flux scale. That is a real, useful check (a
    gridder bug that redistributes flux between radii would still show up),
    but it is NOT the full quantitative "agrees within errors" success
    criterion the spec asks for -- that needs the physical-unit
    reconciliation done first (flagged in the spec's decision log; ask
    before doing that work, given the project's history here).
    normalize='none' skips this and compares raw values (only meaningful
    once/if that reconciliation is done).

    Returns dict: r_mid, stack_profile, stack_profile_err, oned_profile,
    oned_profile_err, ratio (stack/1d, after normalization), normalize.
    """
    if normalize not in ("peak", "none"):
        raise ValueError(f"normalize must be 'peak' or 'none' (got {normalize!r})")

    stk = stack_result["stacks"][method][band]
    az = azimuthal_profile(stk["image"], stack_result["x_centers"], stack_result["y_centers"],
                           r_edges_kpc, var_map=stk["err"] ** 2)

    p1d = np.asarray(profile_1d, dtype=float)
    e1d = (np.asarray(profile_1d_err, dtype=float) if profile_1d_err is not None
          else np.full_like(p1d, np.nan))
    if p1d.shape != az["profile"].shape:
        raise ValueError(f"profile_1d has {p1d.shape} bins, azimuthal_profile "
                         f"produced {az['profile'].shape} -- pass the SAME r_edges_kpc "
                         "used to build profile_1d.")

    stack_p = az["profile"].copy()
    stack_p_err = az["profile_err"].copy()
    p1d_use = p1d
    if normalize == "peak":
        s_peak = np.nanmax(np.abs(stack_p))
        d_peak = np.nanmax(np.abs(p1d))
        if s_peak:
            stack_p = stack_p / s_peak
            stack_p_err = stack_p_err / s_peak
        if d_peak:
            p1d_use = p1d / d_peak

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = stack_p / p1d_use

    return dict(r_mid=az["r_mid"], stack_profile=stack_p, stack_profile_err=stack_p_err,
               oned_profile=p1d_use, oned_profile_err=e1d, ratio=ratio, normalize=normalize)


def plot_stack_diagnostics(stack_result: dict, *, method: str = "inv_var",
                           r_edges_kpc=None, cmap: str = "inferno",
                           validation: dict = None, figsize=None):
    """
    One-stop Part D figure: stacked on-band + off-band images (shared color
    scale, PSF contour if stack_result['psf'] is available), the azimuthal
    profile (if r_edges_kpc given), and the 1D-profile overlay (if a
    validate_stack_vs_1d_profile result is passed as `validation`).

    Returns (fig, axs).
    """
    import matplotlib.pyplot as plt

    stk = stack_result["stacks"][method]
    on = stk["on_band"]["image"]; off = stk["off_band"]["image"]
    finite = np.concatenate([on[np.isfinite(on)], off[np.isfinite(off)]])
    vmin, vmax = (tuple(np.nanpercentile(finite, [1, 99])) if finite.size else (0.0, 1.0))

    ncols = 2 + int(r_edges_kpc is not None)
    figsize = figsize or (4.6 * ncols, 4.3)
    fig, axs = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)
    axs = np.atleast_1d(axs)

    extent = [stack_result["x_edges"][0], stack_result["x_edges"][-1],
             stack_result["y_edges"][0], stack_result["y_edges"][-1]]
    psf_img = None
    if stack_result.get("psf") is not None:
        fwhm_kpc = stack_result["psf"]["fwhm_kpc"]; beta = stack_result["psf"]["beta"]
        xx, yy = np.meshgrid(stack_result["x_centers"], stack_result["y_centers"])
        from .fitting import moffat_1d
        psf_img = moffat_1d(np.sqrt(xx ** 2 + yy ** 2), fwhm=fwhm_kpc, beta=beta)
        psf_img = psf_img / np.nanmax(psf_img)

    im = None
    for ax, grid, title in ((axs[0], on, f"stacked on-band ({method})"),
                            (axs[1], off, f"stacked off-band ({method})")):
        im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title); ax.set_xlabel("kpc"); ax.set_ylabel("kpc")
        if psf_img is not None:
            ax.contour(stack_result["x_centers"], stack_result["y_centers"], psf_img,
                      levels=[0.1, 0.5, 0.9], colors="cyan", linewidths=0.8, alpha=0.85)
    fig.colorbar(im, ax=list(axs[:2]), shrink=0.85, label="stacked surface brightness")

    if r_edges_kpc is not None:
        az = azimuthal_profile(on, stack_result["x_centers"], stack_result["y_centers"],
                               r_edges_kpc, var_map=stk["on_band"]["err"] ** 2)
        ax3 = axs[2]
        ax3.errorbar(az["r_mid"], az["profile"], yerr=az["profile_err"],
                    fmt="o-", color="tab:red", label="stacked image (azimuthal)")
        if validation is not None:
            ax3.errorbar(validation["r_mid"], validation["oned_profile"],
                        yerr=validation["oned_profile_err"], fmt="s--",
                        color="k", label="1D pipeline")
            ax3.legend(fontsize=9)
        ax3.set_xlabel("radius [kpc]"); ax3.set_ylabel("surface brightness")
        ax3.set_title("azimuthal profile" + (" vs 1D" if validation is not None else ""))

    fig.suptitle(f"N$_{{gal}}$={stack_result['ngal_used']} "
               f"({stack_result['n_failed']} skipped)"
               + ("   [random-rotation null]" if stack_result["random_rotate"] else ""),
               y=1.06)
    return fig, axs
