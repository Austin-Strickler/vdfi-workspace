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

from typing import TYPE_CHECKING, Optional

import numpy as np

from .stack import convert_avg_fiber_bin, stack_spec_values, rest_grid
from .extract import radial_bin_edges, combine_fibers
from .virial import estimate_M200c_R200c_from_Mstar
from .measure import measure_centroid, integrated_line_flux, DEFAULT_CONT_BOUNDS

if TYPE_CHECKING:
    from .config import PipelineConfig
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
        _, r200c = estimate_M200c_R200c_from_Mstar(10 ** m_used, float(z))
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
            fx[r]  = fres["flux_sum"]
            fxe[r] = fres["err_sum"]

    return {"r_edges": r_edges, "centroid_v": cen_v, "flux": fx,
            "flux_err": fxe, "sn": np.asarray(sg["sn"]), "method": method}
