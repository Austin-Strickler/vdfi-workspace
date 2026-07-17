"""
analysis.py -- Stage-3 notebook helpers: orchestration, tables, and plot wrappers.

The DISPLAY / ORCHESTRATION companion to measure.py (which does the computation).
Near-identical role-name, opposite job: measure.py COMPUTES, analysis.py runs the
suite and DRAWS. (Formerly named measuring.py.) Run it in a notebook with minimal
setup:

    from utils_lya_halo import analysis

    # Stage 3: run everything
    boot = analysis.run_measurement_suite(cfg, stacks)

    # Inspect results
    analysis.print_centroid_table(boot)
    analysis.compare_centroid_methods(stacks, config=cfg)          # quick point-estimate table
    cmp = analysis.measure_centroid_methods(stacks, config=cfg)    # bootstrapped
    analysis.plot_centroid_methods(cmp, stacks)                    # overlay w/ 16-84 bars

    # Spectral diagnostics
    analysis.plot_line_panels(stacks, boot)
    # (continuum CHECK lives in validation now: validation.plot_continuum_check)

    # Radial profiles
    analysis.plot_centroid_profile(boot, stacks)
    analysis.plot_flux_profile(boot, stacks)        # logy=True by default
    analysis.plot_flux_profile_fit(boot, stacks, method="psf")     # + double-exp fit
    analysis.plot_flux_profile_fit(boot, stacks, method="naive")   # no-PSF fit
    analysis.plot_asymmetry_profile(boot, stacks)
    analysis.plot_moments_profile(stacks)

    # Curve of growth (cumulative flux / flux fraction vs radius; drops the
    # outermost bin by default -- see measure.flux_curve_of_growth)
    from utils_lya_halo import measure
    cog = measure.flux_curve_of_growth(boot, stacks)
    analysis.plot_flux_curve_of_growth(cog, stacks)

    # Bootstrap health checks
    analysis.plot_success_frac(boot, stacks)
    analysis.plot_bootstrap_draws(boot, radial_index=0)

    # Run all plots in one shot
    analysis.run_all_plots(stacks, boot)

Two-stack (half-sample comparison) notebook
--------------------------------------------
When comparing two sub-samples (e.g. low-mass vs high-mass), use the *_two
helpers to put both on one figure instead of running each plot twice. Note the
stack-panel figure itself lives in plotting.py (not analysis.py):

    from utils_lya_halo import plotting
    plotting.plot_stack_panels_two(stacks_low_m, stacks_high_m,
                                   labels=("low-M", "high-M"))   # overlay or side
    analysis.plot_centroid_profile_two(boot_lm, boot_hm, stacks_low_m, stacks_high_m)
    analysis.plot_flux_profile_two(boot_lm, boot_hm, stacks_low_m, stacks_high_m)
    analysis.plot_asymmetry_profile_two(boot_lm, boot_hm, stacks_low_m, stacks_high_m)

    # all four comparison figures in one call (this wraps plotting.plot_stack_panels_two)
    analysis.run_all_plots_two(stacks_low_m, stacks_high_m, boot_lm, boot_hm,
                               labels=("low-M", "high-M"))

What belongs here vs validation.py
-----------------------------------
This file: things you run every time to CHARACTERISE and INSPECT a result --
spectra, profiles, tables, centroid comparison, bootstrap diagnostics.

validation.py: things you run to STRESS-TEST the result -- z-scramble null,
placebo, injection-recovery, jackknife cross-check, continuum/window sweeps.

Centroid-estimator labels (fw_clip, fw_noclip, gauss_weighted, gaussian,
flux_median, peak) and continuum defaults are NOT defined here -- they come from
measure.CENTROID_METHOD_SPECS/LABELS and config.py, so this module never drifts
from the engine or the stress-tests.
"""

from __future__ import annotations

import warnings
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .virial import cosmo   # arcsec->kpc PSF conversion (Lujan Niemeyer FWHM default)
from .measure import (
    measure_all_bins, bootstrap_measurements, bootstrap_stack_error, bootstrap_all,
    measure_centroid, line_moments, get_continuum_model,
    CENTROID_METHOD_SPECS, CENTROID_METHOD_LABELS, CENTROID_METHODS_DEFAULT,
    DEFAULT_CONT_BOUNDS, DEFAULT_CONT_METHOD, DEFAULT_CONT_ORDER,
    LYA_REST, C_KMS,
)
from .plotting import (
    _setup_radius_axis, _resolve_bin_mode, _get_vr_biweight_v, _get_vr_biweight_e,
    _safe_yerr,
    plot_centroid_vs_radius, plot_blue_red_vs_radius,
    _mark_lines,
)
from . import fitting


def _resolve_measure_defaults(config=None, *, bounds=None, cont_bounds=None,
                              cont_method=None, cont_order=None, lya_center=None):
    """
    Fill in measurement kwargs from `config` when given, else from the canonical
    config.py module defaults (DEFAULT_CONT_*). Explicit, non-None arguments
    always win. This is the ONE place the analysis layer decides its continuum /
    window / systemic defaults, so its tables match measure_all_bins instead of
    silently using a different continuum model.

    Returns (bounds, cont_bounds, cont_method, cont_order, lya_center).
    """
    if config is not None:
        if lya_center is None:  lya_center = float(getattr(config, "LYA_REST", LYA_REST))
        if bounds is None:      bounds = tuple(config.line_window)
        if cont_bounds is None: cont_bounds = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
        if cont_method is None: cont_method = getattr(config, "cont_method", DEFAULT_CONT_METHOD)
        if cont_order is None:  cont_order = getattr(config, "cont_order", DEFAULT_CONT_ORDER)
    lya_center  = LYA_REST            if lya_center  is None else float(lya_center)
    cont_bounds = DEFAULT_CONT_BOUNDS if cont_bounds is None else cont_bounds
    cont_method = DEFAULT_CONT_METHOD if cont_method is None else cont_method
    cont_order  = DEFAULT_CONT_ORDER  if cont_order  is None else cont_order
    bounds      = (lya_center - 4, lya_center + 4) if bounds is None else tuple(bounds)
    return bounds, cont_bounds, cont_method, cont_order, lya_center


# sentinel distinguishing "use the auto-generated title" (default) from
# "no title at all" (title=None, an explicit user choice)
_AUTO = object()


def _psf_fwhm_arcsec_to_kpc(fwhm_arcsec: float, z_median, *, sample_label: str) -> float:
    """
    Convert a FIXED angular PSF FWHM (arcsec -- e.g. the Lujan Niemeyer 2022
    literature default, 1.2-1.4" fiducial 1.3") to a kpc width for ONE
    sample, using that sample's own representative redshift (z_median, as
    stored on the Stage-2 stacks dict by stack.build_stacks). Same
    kpc-per-arcsec conversion stack.convert_avg_fiber_bin already uses per
    galaxy; here it's applied ONCE per SAMPLE (a single representative z),
    since the PSF-aware fit in plot_flux_profile_fit/plot_flux_profile_two
    works on the already-stacked profile, not per-galaxy.

    Deliberately per-SAMPLE, not shared across both samples in a two-sample
    comparison: a low-z vs. high-z split is exactly the case where sharing
    one converted kpc value between the two samples would be wrong (the
    same 1.3" PSF is physically WIDER in kpc for the higher-z sample), so
    plot_flux_profile_two converts each sample's PSF separately using its
    own stacks_a/stacks_b z_median.

    Raises a clear error (not a silent fallback) if z_median is unavailable,
    since guessing here would silently build the wrong-width PSF.
    """
    if z_median is None:
        raise ValueError(
            f"psf_fwhm_arcsec needs a redshift to convert to kpc ({sample_label}), "
            f"but no z_median was found/given. Pass the corresponding `stacks` dict "
            f"(build_stacks stores z_median on it automatically), an explicit "
            f"z_median override, or your own psf_r/psf_vals curve already in kpc.")
    kpc_per_arcsec = cosmo.angular_diameter_distance(float(z_median)).to(u.kpc).value / 206265.0
    return float(fwhm_arcsec) * kpc_per_arcsec


def _resolve_savename(save_name: str | None, default: str) -> str:
    """
    Turn a user-supplied `save_name` (with or without an extension, e.g.
    'Flux_Profile' or 'Flux_Profile.png') into a concrete filename for
    plt.savefig, falling back to `default` when save_name is None.
    """
    if not save_name:
        return default
    known_ext = (".png", ".pdf", ".jpg", ".jpeg", ".svg", ".eps", ".tif", ".tiff")
    return save_name if save_name.lower().endswith(known_ext) else save_name + ".png"


# =====================================================================
# 1.  STAGE-3 EXECUTION
# =====================================================================

def run_measurement_suite(
    config,
    stacks: dict,
    stack_method: str | None = None,
    compute_stack_error: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the full Stage-3 measurement suite on a Stage-2 stacks dict and return
    a combined result ready for every downstream function in this module.

    Calls measure_all_bins (bootstrap centroid + blue/red side ratio) then, if
    compute_stack_error=True, bootstrap_stack_error (per-pixel flux error). Also
    computes line_moments on the FIDUCIAL stack (point estimate only -- no
    bootstrap cost) and stores them in boot["moments"]. All results are returned
    in one dict so the rest of the notebook only ever holds `boot`.

    Parameters
    ----------
    config         : PipelineConfig
    stacks         : output of run_stack / build_stacks (keep_cube=True required)
    stack_method   : galaxy-combine to use (None -> biweight or config value)
    compute_stack_error : whether to run bootstrap_stack_error (adds ~1 min)
    verbose        : print progress

    Returns
    -------
    boot : dict  (measure_all_bins summary + "moments" key + axis info)
    """
    boot = measure_all_bins(
        config, stacks,
        stack_method=stack_method,
        compute_stack_error=compute_stack_error,
        verbose=verbose,
    )

    # add point-estimate line moments (free: no bootstrap)
    wave = np.asarray(stacks["rest_wave"])
    sm = stack_method or getattr(config, "measure_stack_method", "biweight")
    fiducial = np.asarray(stacks["stacks"][sm]["flux"])   # (nrad, nwave)
    nrad = fiducial.shape[0]
    cont_bounds = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
    cont_method = getattr(config, "cont_method", "poly")
    cont_order  = getattr(config, "cont_order",  1)
    bounds      = tuple(config.line_window)
    lya         = float(config.LYA_REST)

    mkeys = ("centroid_A", "centroid_vel_kms", "sigma_A", "sigma_kms", "skewness")
    mom = {k: np.full(nrad, np.nan) for k in mkeys}
    mom["success"] = np.zeros(nrad, dtype=bool)
    for r in range(nrad):
        res = line_moments(wave, fiducial[r], bounds=bounds, cont_bounds=cont_bounds,
                           lya_center=lya, cont_method=cont_method,
                           cont_order=cont_order)
        mom["success"][r] = res["success"]
        if res["success"]:
            for k in mkeys:
                mom[k][r] = res[k]
    boot["moments"] = mom

    if verbose:
        print("\nline moments (fiducial stack, no bootstrap):")
        edges = np.round(np.asarray(stacks["r_edges"]), 3)
        for r in range(nrad):
            print(f"  {edges[r]:g}-{edges[r+1]:g}: "
                  f"v={mom['centroid_vel_kms'][r]:+.1f}  "
                  f"σ={mom['sigma_kms'][r]:.1f}  "
                  f"skew={mom['skewness'][r]:+.3f}")

    return boot


# =====================================================================
# 2.  TABLES
# =====================================================================

def print_centroid_table(boot: dict, r_edges=None):
    """
    Print a formatted table of centroid + side-ratio results per radial bin.

    Columns: bin edges | v_fid | [v_lo, v_hi] | success_frac
             | blue/red_fid | sigma_kms | skewness (last two from moments if present)

    boot    : output of run_measurement_suite / measure_all_bins
    r_edges : override for bin edges (default: boot["r_edges"] if present)
    """
    edges = np.round(np.asarray(r_edges if r_edges is not None
                                else boot.get("r_edges", [])), 3)
    nrad  = len(boot["centroid_v_fid"])

    has_ratio   = "blue_over_red_fid" in boot
    has_moments = "moments" in boot

    hdr  = f"{'bin':>10}  {'v_fid':>7}  {'[lo,hi]':>16}  {'ok%':>5}"
    if has_ratio:   hdr += f"  {'B/R':>6}"
    if has_moments: hdr += f"  {'σ_kms':>7}  {'skew':>7}"
    print(hdr)
    print("─" * len(hdr))

    for r in range(nrad):
        lbl = (f"{edges[r]:g}-{edges[r+1]:g}" if len(edges) > r + 1
               else f"bin{r}")
        v   = boot["centroid_v_fid"][r]
        lo  = boot["centroid_v_lo"][r]
        hi  = boot["centroid_v_hi"][r]
        ok  = boot["success_frac"][r] * 100
        row = f"{lbl:>10}  {v:>+7.1f}  [{lo:>+6.1f},{hi:>+6.1f}]  {ok:>4.0f}%"
        if has_ratio:
            bor = boot["blue_over_red_fid"][r]
            row += f"  {bor:>6.3f}"
        if has_moments:
            sig  = boot["moments"]["sigma_kms"][r]
            skew = boot["moments"]["skewness"][r]
            row += f"  {sig:>7.1f}  {skew:>+7.3f}"
        print(row)


def compare_centroid_methods(
    stacks: dict,
    config=None,
    methods=CENTROID_METHODS_DEFAULT,
    stack_method: str = "biweight",
    bounds=None,
    cont_bounds=None,
    cont_method=None,
    cont_order=None,
    lya_center=None,
    verbose: bool = True,
) -> dict:
    """
    Run the standard centroid estimators on the fiducial stack (POINT ESTIMATES
    ONLY, no bootstrap) and print a comparison table. Fastest way to see how the
    clip choice and estimator shape move each radial bin.

    Estimators come from measure.CENTROID_METHOD_SPECS (the same labels the
    bootstrap and the injection test use): by default fw_clip, fw_noclip,
    gauss_weighted, gaussian, flux_median, peak. ('moments' is deliberately not a
    centroid here -- its 1st moment equals fw_clip; for width/skewness use
    plot_moments_profile or the moments columns of print_centroid_table.)

    Continuum / window / systemic default to `config` (so this MATCHES the
    headline measure_all_bins), falling back to the config.py module defaults
    (DEFAULT_CONT_*). Explicit arguments override. NOTE: with no config and no
    overrides this now uses the canonical default continuum (DEFAULT_CONT_METHOD,
    currently 'median'), not the old hard-coded 'poly'/order-1.

    For a bootstrapped comparison (with error bars) use measure_centroid_methods;
    for a stress-test against a synthetic line use validation.injection_recovery.

        analysis.compare_centroid_methods(stacks, config=cfg)

    Returns
    -------
    dict: {"methods": [...], "velocities": {label: (nrad,) km/s},
           "extras": {label: {extra_key: (nrad,)}}, "r_edges": ...}
    """
    wave    = np.asarray(stacks["rest_wave"])
    fid     = np.asarray(stacks["stacks"][stack_method]["flux"])  # (nrad, nwave)
    r_edges = np.round(np.asarray(stacks["r_edges"]), 3)
    nrad    = fid.shape[0]
    bounds, cont_bounds, cont_method, cont_order, lya_center = _resolve_measure_defaults(
        config, bounds=bounds, cont_bounds=cont_bounds, cont_method=cont_method,
        cont_order=cont_order, lya_center=lya_center)

    specs = {lab: dict(CENTROID_METHOD_SPECS[lab]) for lab in methods}
    common = dict(bounds=bounds, cont_bounds=cont_bounds, lya_center=lya_center,
                  cont_method=cont_method, cont_order=cont_order)
    velocities = {}
    extras     = {}

    for lab, spec in specs.items():
        meth = spec["centroid_method"]
        clip = spec.get("clip_negative", True)
        vv  = np.full(nrad, np.nan)
        ext = {}
        for r in range(nrad):
            res = measure_centroid(wave, fid[r], method=meth,
                                   clip_negative=clip, **common)
            vv[r] = res["centroid_vel_kms"]
            # capture line width when the estimator produces one (free)
            if "sigma_A" in res:
                ext.setdefault("sigma_A", np.full(nrad, np.nan))[r] = res.get("sigma_A", np.nan)
        velocities[lab] = vv
        extras[lab]     = ext

    if verbose:
        col_w = max(len(f"{r_edges[r]:g}-{r_edges[r+1]:g}") for r in range(nrad)) + 1
        hdr = f"{'method':>24} | " + " ".join(
            f"{f'{r_edges[r]:g}-{r_edges[r+1]:g}':>{col_w}}" for r in range(nrad))
        print(hdr)
        print("─" * len(hdr))
        for lab in specs:
            row = f"{CENTROID_METHOD_LABELS.get(lab, lab):>24} | " + " ".join(
                f"{velocities[lab][r]:>+{col_w}.1f}" for r in range(nrad))
            print(row)
        print(f"\n  units: km/s  (point estimates on fiducial stack, no bootstrap;"
              f" continuum={cont_method})")

    return {"methods": list(specs), "velocities": velocities,
            "extras": extras, "r_edges": r_edges}


def measure_centroid_methods(
    stacks: dict,
    config=None,
    methods=CENTROID_METHODS_DEFAULT,
    stack_method: str = "biweight",
    nboot: int = 500,
    bounds=None,
    cont_bounds=None,
    cont_method=None,
    cont_order=None,
    lya_center=None,
    seed: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Bootstrap the REAL centroid with several estimators and return one boot dict
    per method, ready to overlay. This is the with-error-bars sibling of
    compare_centroid_methods (which is point-estimate-only and prints a table).

    Unlike validation.injection_recovery this does NOT stress-test against a
    synthetic line -- it simply measures your actual data with each estimator and
    bootstraps the galaxy resampling, so you can decide which estimator to adopt
    by comparing real centroids ± 16/84 bars directly.

    Every estimator is run on the SAME cube with the SAME seed, so the only thing
    that varies between curves is the estimator itself. Needs the per-galaxy cube
    (build the stacks with keep_cube=True).

    Parameters
    ----------
    stacks   : Stage-2 dict with cube_flux / cube_err (keep_cube=True).
    config   : optional PipelineConfig; if given, line_window / continuum settings
               are read from it (explicit args still win).
    methods  : labels from measure.CENTROID_METHOD_SPECS, or an explicit {label: kwargs}
               dict forwarded to bootstrap_measurements.
    nboot    : bootstrap draws per estimator.

    Returns
    -------
    dict: {"boots": {label: boot_dict}, "methods": [...], "r_edges": ...,
           "bin_mode": ..., "VR_biweight_v": ...}
           Each boot_dict is a bootstrap_measurements summary (centroid_v_fid /
           _med / _lo / _hi, success_frac, ...). Pass straight to
           plot_centroid_methods.
    """
    cube = stacks.get("cube_flux")
    if cube is None:
        raise KeyError("measure_centroid_methods needs cube_flux (build stacks "
                       "with keep_cube=True).")
    cube = np.asarray(cube, dtype=float)
    cube_e = stacks.get("cube_err")
    cube_e = np.asarray(cube_e, dtype=float) if cube_e is not None else None
    wave = np.asarray(stacks["rest_wave"], dtype=float)

    # config-aware defaults (explicit args win); falls back to config.py defaults
    bounds, cont_bounds, cont_method, cont_order, lya_center = _resolve_measure_defaults(
        config, bounds=bounds, cont_bounds=cont_bounds, cont_method=cont_method,
        cont_order=cont_order, lya_center=lya_center)

    # resolve estimator specs from the canonical vocabulary (measure.py)
    if isinstance(methods, dict):
        specs = {k: dict(v) for k, v in methods.items()}
    else:
        specs = {}
        for lab in methods:
            if lab not in CENTROID_METHOD_SPECS:
                raise ValueError(f"unknown method label {lab!r}; choose from "
                                 f"{sorted(CENTROID_METHOD_SPECS)} or pass a "
                                 f"{{label: kwargs}} dict.")
            specs[lab] = dict(CENTROID_METHOD_SPECS[lab])

    common = dict(nboot=nboot, bounds=bounds, cont_bounds=cont_bounds,
                  lya_center=lya_center, stack_method=stack_method,
                  cont_method=cont_method, cont_order=cont_order,
                  compute_side_ratio=False, compute_stack_error=False,
                  seed=seed, verbose=verbose)

    boots = {}
    for lab, kw in specs.items():
        # fast merged+sliced bootstrap (bootstrap_all), one call per estimator;
        # the run header + progress bar carry the estimator label so you can see
        # which method is running. compute_stack_error=False: centroids only.
        b = bootstrap_all(cube, cube_e, wave, **kw, **common,
                          label=f"centroid [{CENTROID_METHOD_LABELS.get(lab, lab)}]")
        b["r_edges"] = np.asarray(stacks["r_edges"])
        b["bin_mode"] = stacks.get("bin_mode")
        b["VR_biweight_v"] = stacks.get("VR_biweight_v")
        boots[lab] = b

    if verbose:
        edges = np.round(np.asarray(stacks["r_edges"]), 3)
        nrad = len(next(iter(boots.values()))["centroid_v_fid"])
        labs = list(boots)
        col_w = max(len(f"{edges[r]:g}-{edges[r+1]:g}") for r in range(nrad)) + 1
        hdr = f"{'method':>22} | " + " ".join(
            f"{f'{edges[r]:g}-{edges[r+1]:g}':>{col_w}}" for r in range(nrad))
        print("\n" + hdr)
        print("─" * len(hdr))
        for lab in labs:
            v = boots[lab]["centroid_v_fid"]
            row = f"{CENTROID_METHOD_LABELS.get(lab, lab):>22} | " + " ".join(
                f"{v[r]:>+{col_w}.1f}" for r in range(nrad))
            print(row)
        print("\n  units: km/s  (fiducial centroid; ±16/84 bars shown in the plot)")

    return {"boots": boots, "methods": list(boots),
            "r_edges": np.asarray(stacks["r_edges"]),
            "bin_mode": stacks.get("bin_mode"),
            "VR_biweight_v": stacks.get("VR_biweight_v")}


def plot_centroid_methods(
    result: dict,
    stacks: dict | None = None,
    use_fiducial: bool = True,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    ylims=(-250, 250),
    xlims=None,
    jitter: float = 0.05,
    figsize=(7.8, 5.0),
    title="Centroid vs. radius by estimator (bootstrap 16–84)",
    save_fig: bool = False,
    savename: str = "Figure_centroid_methods.png",
):
    """
    Overlay the bootstrapped centroid-vs-radius curves from measure_centroid_methods,
    one labelled curve per estimator with its 16/84 bars, so you can SEE which
    estimators agree and where the clip choice (clipped vs no-clip) or shape
    (Gaussian vs flux-weighted) starts to matter with radius.

    The plotted point is the fiducial (full-sample) centroid by default
    (use_fiducial=True), matching compare_centroid_methods; set use_fiducial=False
    to plot the bootstrap median instead. Same radius axis as every other figure.

        cmp = analysis.measure_centroid_methods(stacks, config=cfg)
        analysis.plot_centroid_methods(cmp, stacks)

    Returns (fig, ax).
    """
    boots = result["boots"]
    labels = result["methods"]
    radial_bins = np.asarray(result.get("r_edges")
                             if result.get("r_edges") is not None
                             else (stacks["r_edges"] if stacks else []))
    sr = stacks or result
    bm = _resolve_bin_mode(bin_mode, sr)
    vr = _get_vr_biweight_v(VR_biweight_v, sr)

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, _ = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims)
    suff = "_fid" if use_fiducial else "_med"
    K = len(labels)
    for k, lab in enumerate(labels):
        b = boots[lab]
        v = np.asarray(b["centroid_v" + suff])
        lo = np.asarray(b["centroid_v_lo"]); hi = np.asarray(b["centroid_v_hi"])
        jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
        yerr, unstable = _safe_yerr(v, lo, hi)
        eb = ax.errorbar(jit, v, yerr=yerr, fmt="o-",
                         capsize=3, ms=5, lw=1.3,
                         label=CENTROID_METHOD_LABELS.get(lab, lab))
        if np.any(unstable):
            ax.scatter(jit[unstable], v[unstable], s=70, facecolors="none",
                       edgecolors=eb[0].get_color(), linewidths=1.3, zorder=5)
    ax.axhline(0, color="tomato", alpha=0.4, lw=1)
    ax.set_ylim(ylims)
    ax.set_ylabel(r"Ly$\alpha$ centroid velocity [km s$^{-1}$]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, title="estimator")
    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# 3.  SPECTRAL DIAGNOSTIC PLOTS
# =====================================================================

def plot_line_panels(
    stacks: dict,
    boot: dict | None = None,
    stack_method: str = "biweight",
    bounds=None,
    cont_bounds=DEFAULT_CONT_BOUNDS,
    cont_method: str = "poly",
    cont_order: int = 1,
    lya_center: float = LYA_REST,
    xlim: tuple = (1208.0, 1226.0),
    xunit: str = "wavelength",
    panel_height: float = 2.6,
    figwidth: float = 8.0,
    ncols: int = 1,
    fill_order: str = "row",
    show_cont: bool = True,
    show_window: bool = True,
    show_sidebands: bool = True,
    show_centroid: bool = True,
    stack_color: str = "k",
    band_color: str = "0.82",
    band_alpha: float = 1.0,
    cont_color: str = "tomato",
    window_color: str = "magenta",
    window_alpha: float = 0.15,
    sideband_color: str = "0.55",
    sideband_alpha: float = 0.25,
    centroid_color: str = "magenta",
    lya_line_color: str = "0.7",
    zero_color: str = "0.7",
    label_fontsize: float = 9.5,
    legend_fontsize: float = 9,
    tick_fontsize: float = 10,
    xlabel_fontsize: float = 12,
    ylabel_fontsize: float = 11,
    title_fontsize: float = 13,
    xlabel_text: str | None = None,
    ylabel_text: str = "Flux",
    title_text: str | None = None,
    xlabel_y: float = 0.01,
    ylabel_x: float = 0.01,
    title_y: float | None = None,
    legend_y: float = 0.95,
    top_margin: float = 0.87,
    band_label: str = "bootstrap 1$\\sigma$ (fid-centered)",
    stack_label: str | None = None,
    cont_label: str = "continuum",
    save_fig: bool = False,
    savename: str = "Figure_line_panels.png",
):
    """
    The most important per-run diagnostic: one spectral panel per radial bin
    zoomed to the Lyα region, showing:

      • Bootstrap 1-sigma shaded band, fid-centered (grey; needs boot["stack_error"]).
        Width/asymmetry come from the bootstrap p16/p84 spread around the
        bootstrap median, but the band is re-anchored onto the fiducial curve
        below, so it is a per-pixel error ON the fiducial stack, not a raw
        percentile band that can wander off-center for small-N bins.
      • Biweight (or chosen method) fiducial stack (black)
      • Polynomial continuum fit (red dashed; optional)
      • Line window shaded (magenta by default; optional)
      • Sideband regions shaded (grey by default; optional)
      • Centroid marker: vertical dashed line at v_fid, with a short horizontal
        bar at the top spanning the 16/84 interval (optional; needs boot)

    Every color and font size above is a keyword argument (see below), so the
    whole figure is restylable without touching the function body.

    This is the visual check that the centroid is landing on the line peak, that
    the continuum is flat in the sidebands, and that the window captures the
    emission without clipping the wings.

    Parameters
    ----------
    stacks       : Stage-2 dict (keep_cube=True for the boot error band)
    boot         : Stage-3 bootstrap result; None -> no error band / centroid marker
    stack_method : which stacks[method] to display
    bounds       : line window (lo, hi) Å; None -> (lya_center-4, lya_center+4)
    xlim         : x-axis range shown in each panel. ALWAYS given in
                   Angstroms regardless of xunit (same for bounds/cont_bounds/
                   lya_center) -- these stay the physical, science-facing
                   inputs; xunit only changes what's DISPLAYED. e.g. the
                   default xlim=(1208, 1226) Å auto-converts to roughly
                   (-1890, +2550) km/s when xunit="velocity".
    xunit        : "wavelength" (default, Å, rest-frame) or "velocity" (km/s
                   relative to lya_center, via v = c*(wave-lya_center)/lya_center
                   -- the same convention already used for the centroid). The
                   vertical dotted rest-frame marker line sits at v=0 in
                   velocity mode. The x-axis label switches automatically too
                   ("Velocity [km s$^{-1}$]") unless you set xlabel_text.
    ncols        : number of grid columns (default 1 = the original single tall
                   column). For a paper figure with many radial bins, ncols=2
                   (or more) arranges the panels in a grid instead of one very
                   tall column. x-axis tick labels are only drawn on the
                   bottom-most used panel of each column. Try e.g.
                   ncols=2, figwidth=11, panel_height=2.6, show_centroid=False
                   for a compact 2x5 print figure with the centroid marker
                   left out (since that belongs on the separate centroid-vs-
                   radius plot).
    fill_order   : "row" (default) -- panels fill left-to-right, then wrap to
                   the next row (bin 0 top-left, bin 1 top-right, ...).
                   "col" -- panels fill top-to-bottom within a column, then
                   continue into the next column (the original grid ordering).
                   No effect when ncols=1.
    stack_color, band_color, cont_color, window_color, sideband_color,
    centroid_color, lya_line_color, zero_color : per-element colors, all
                   independently overridable. window_color/sideband_color
                   default to magenta/grey; band_alpha, window_alpha,
                   sideband_alpha control their fill opacity.
    label_fontsize, legend_fontsize, tick_fontsize, xlabel_fontsize,
    ylabel_fontsize, title_fontsize : text sizes for the per-panel bin label,
                   the legend, the tick labels, the shared x/y axis labels,
                   and the figure suptitle, respectively.
    xlabel_text, ylabel_text, title_text : override the shared x-axis label,
                   shared y-axis label, and suptitle strings. xlabel_text=None
                   picks the wavelength/velocity default based on xunit;
                   title_text=None keeps the auto-generated "Lyα line panels
                   -- {method} stack". The x-axis label is drawn ONCE for the
                   whole figure (like the y-axis "Flux" label) via
                   fig.supxlabel, not repeated under every column.
    band_label, stack_label, cont_label : legend text for the bootstrap 1σ
                   band, the fiducial stack line, and the continuum line.
                   stack_label=None defaults to f"{stack_method} stack" (so
                   it tracks whichever method you're plotting); the other two
                   default to "bootstrap 1$\\sigma$ (fid-centered)" and
                   "continuum".
    xlabel_y, ylabel_x : figure-fraction position of the shared x-/y-axis
                   label (passed straight to fig.supxlabel(y=...) /
                   fig.supylabel(x=...)); nudge these if the label sits too
                   close to or far from the tick numbers.
    title_y      : figure-fraction y position of the suptitle. None -> auto
                   (near the very top for ncols=1, or above the reserved
                   legend strip for a grid).
    legend_y     : figure-fraction y position (bbox_to_anchor) of the shared
                   legend drawn above the grid when ncols>1. No effect when
                   ncols=1 (legend sits inside the first panel instead).
    top_margin   : fraction of the figure height given to the panel grid
                   when ncols>1 (the rest, above it, is reserved for the
                   legend + title). Lower this (e.g. 0.80) if the legend is
                   crowding the top row -- this is the fix for a squeezed
                   legend with many columns or with axes that show a
                   scientific-notation exponent (e.g. "1e38") above them.
    """
    if xunit not in ("wavelength", "velocity"):
        raise ValueError("xunit must be 'wavelength' or 'velocity'")

    wave   = np.asarray(stacks["rest_wave"])
    fid    = np.asarray(stacks["stacks"][stack_method]["flux"])   # (nrad, nwave)
    edges  = np.round(np.asarray(stacks["r_edges"]), 3)
    nrad   = fid.shape[0]
    bounds = bounds if bounds is not None else (lya_center - 4, lya_center + 4)
    bm     = stacks.get("bin_mode", "")
    stack_label_final = stack_label if stack_label is not None else f"{stack_method} stack"

    # xunit only changes what's DISPLAYED -- bounds/cont_bounds/xlim/lya_center
    # stay in Angstroms as the science-facing inputs (and the continuum fit
    # below is always computed in wavelength space); here we just derive the
    # x-COORDINATES actually handed to the plotting calls, converting to km/s
    # (same convention as the centroid: v = c*(wave-lya_center)/lya_center)
    # when xunit="velocity".
    def _to_velocity(wl):
        return C_KMS * (np.asarray(wl, dtype=float) - lya_center) / lya_center

    if xunit == "velocity":
        x_plot      = _to_velocity(wave)
        bounds_x    = tuple(_to_velocity(bounds))
        cont_bounds_x = [tuple(_to_velocity(cb)) for cb in cont_bounds]
        xlim_x      = tuple(_to_velocity(xlim))
        lya_line_x  = 0.0
    else:
        x_plot      = wave
        bounds_x    = bounds
        cont_bounds_x = cont_bounds
        xlim_x      = xlim
        lya_line_x  = lya_center

    if xlabel_text is None:
        xlabel_text_final = (r"Velocity [km s$^{-1}$]" if xunit == "velocity"
                             else r"Rest-frame wavelength [$\AA$]")
    else:
        xlabel_text_final = xlabel_text

    # bootstrap error band -- anchored on the fiducial curve, not on the
    # bootstrap distribution's own median. Raw p16/p84 are percentiles of the
    # RESAMPLED restacks and are centered on stack_med (the bootstrap
    # distribution's own median), which is a different quantity from `fid`
    # (the biweight of the actual, unresampled sample). The two normally sit
    # close together, but for small-N / low-S/N bins (the outer bins) they
    # can visibly diverge, so a raw p16-p84 fill_between shows a shaded band
    # whose own midline does not track the solid fiducial curve.
    #
    # Fix: keep the bootstrap-derived WIDTH and ASYMMETRY (how far p16/p84
    # sit below/above stack_med) but re-anchor that spread onto fid, so the
    # black fiducial line is always exactly the midline of the shaded region
    # by construction -- band_lo/band_hi are per-pixel 1-sigma errors ON fid,
    # not an independent percentile band.
    band_lo = band_hi = None
    if boot is not None and boot.get("stack_error") is not None:
        p16       = np.asarray(boot["stack_error"]["p16"])         # (nrad, nwave)
        p84       = np.asarray(boot["stack_error"]["p84"])
        stack_med = np.asarray(boot["stack_error"]["stack_med"])
        err_lo    = stack_med - p16     # distance from the bootstrap median down to p16
        err_hi    = p84 - stack_med     # distance from the bootstrap median up to p84
        band_lo   = fid - err_lo
        band_hi   = fid + err_hi

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(nrad / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols,
                                  figsize=(figwidth, panel_height * nrows),
                                  sharex=True, squeeze=False)

    # bin index r -> (row, col). "row" fill_order goes left-to-right then
    # wraps to the next row (bin 0 top-left, bin 1 top-right, ...); "col"
    # fill_order goes top-to-bottom within a column, then continues into the
    # next column. With ncols=1 both reduce to the original single-column order.
    if fill_order == "row":
        grid_pos = [(r // ncols, r % ncols) for r in range(nrad)]
    else:
        grid_pos = [(r % nrows, r // nrows) for r in range(nrad)]
    axes = [axes_grid[row, col] for row, col in grid_pos]

    # hide any grid cells left empty because nrad doesn't evenly fill nrows*ncols
    used = set(grid_pos)
    for row in range(nrows):
        for col in range(ncols):
            if (row, col) not in used:
                axes_grid[row, col].axis("off")

    for r, ax in enumerate(axes):
        # --- bootstrap band (fid-centered 1-sigma) ---
        if band_lo is not None:
            ax.fill_between(x_plot, band_lo[r], band_hi[r], color=band_color, lw=0,
                            alpha=band_alpha, zorder=0,
                            label=band_label if r == 0 else None)

        # --- sideband regions ---
        if show_sidebands:
            for (lo, hi) in cont_bounds_x:
                ax.axvspan(lo, hi, color=sideband_color, alpha=sideband_alpha, lw=0)

        # --- line window ---
        if show_window:
            ax.axvspan(bounds_x[0], bounds_x[1], color=window_color, alpha=window_alpha, lw=0)

        # --- fiducial stack ---
        ax.plot(x_plot, fid[r], color=stack_color, lw=1.1, zorder=3,
                label=stack_label_final if r == 0 else None)

        # --- continuum (always fit in wavelength space; only the DRAWN curve
        # is shown on the x_plot axis, whichever unit that is) ---
        if show_cont:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cont = get_continuum_model(wave, fid[r], cont_bounds=cont_bounds,
                                           cont_method=cont_method, cont_order=cont_order)
            ax.plot(x_plot, cont, color=cont_color, lw=1.0, ls="--", zorder=2,
                    label=cont_label if r == 0 else None)

        # --- centroid marker ---
        if show_centroid and boot is not None:
            v_f  = boot["centroid_v_fid"][r]
            v_lo = boot["centroid_v_lo"][r]
            v_hi = boot["centroid_v_hi"][r]
            if xunit == "velocity":
                # already in km/s -- no need to round-trip through wavelength
                x_c, x_lo, x_hi = v_f, v_lo, v_hi
            else:
                x_c  = lya_center * (1 + v_f  / C_KMS)
                x_lo = lya_center * (1 + v_lo / C_KMS)
                x_hi = lya_center * (1 + v_hi / C_KMS)
            # vertical dashed line
            ax.axvline(x_c, color=centroid_color, lw=1.1, ls="--", zorder=4,
                       label=f"centroid {v_f:+.0f} km/s" if r == 0 else None)
            # 16/84 bar at top of panel (0.92 in axes coords)
            y_bar = ax.get_ylim()[1] * 0.0   # will be set after ylim below; use transform
            ax.annotate("", xy=(x_hi, 1.0), xytext=(x_lo, 1.0),
                        xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="|-|", color=centroid_color,
                                        lw=1.1, mutation_scale=4))

        # axes / labels
        lbl = f"{edges[r]:g}–{edges[r+1]:g} {bm}"
        ax.text(0.02, 0.88, lbl, transform=ax.transAxes, fontsize=label_fontsize,
                color="0.3",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                          boxstyle="round,pad=0.15"))
        ax.axhline(0, color=zero_color, lw=0.6)
        ax.axvline(lya_line_x, color=lya_line_color, lw=0.6, ls=":")
        ax.set_xlim(xlim_x)
        ax.grid(alpha=0.12)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        in_xlim = (x_plot >= xlim_x[0]) & (x_plot <= xlim_x[1])
        yvals = fid[r][in_xlim]
        if band_lo is not None:
            # both edges -- the old version only looked at the lower edge
            # (p16), so the top of the band could silently clip out of view.
            yvals = np.concatenate([yvals, band_lo[r][in_xlim], band_hi[r][in_xlim]])
        fin = yvals[np.isfinite(yvals)]
        if len(fin):
            lo_, hi_ = np.nanpercentile(fin, [1, 99])
            pad = 0.25 * (hi_ - lo_)
            ax.set_ylim(min(lo_ - pad, 0), hi_ + pad)

    if ncols == 1:
        # single narrow column -- a full-width in-panel legend fits fine,
        # matches the original look exactly.
        axes[0].legend(frameon=False, fontsize=legend_fontsize, ncol=3, loc="upper right")
        title_y_final = title_y if title_y is not None else 1.005
        reserve_top = None
    else:
        # grid mode -- panels are narrower, so an in-panel legend collides
        # with the bin-range label in the corner. Pull the handles off the
        # first panel and draw ONE shared legend in a reserved strip above
        # the panel grid instead. Reserving the strip via subplots_adjust
        # AFTER tight_layout guarantees the axes grid is pushed down out of
        # the legend/title's way (tight_layout's own rect= doesn't reliably
        # leave room for manually-placed fig.legend/suptitle artists).
        # top_margin/legend_y control how much room that strip gets -- widen
        # the gap (lower top_margin and/or raise legend_y) if the legend
        # crowds the top row, which happens easily with many columns or with
        # axes that draw a scientific-notation exponent (e.g. "1e38") just
        # above their top spine.
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=legend_fontsize,
                      ncol=len(labels), loc="upper center",
                      bbox_to_anchor=(0.5, legend_y))
        title_y_final = title_y if title_y is not None else min(legend_y + 0.04, 0.995)
        reserve_top = top_margin

    # x-axis tick labels only on the bottom-most USED panel of each column;
    # hide tick labels on the panels above it so a multi-column grid doesn't
    # repeat the wavelength axis tick numbers down every row. The x-axis
    # TITLE itself is drawn once for the whole figure below (fig.supxlabel),
    # not repeated per column.
    last_row_in_col = {}
    for row, col in grid_pos:
        last_row_in_col[col] = max(last_row_in_col.get(col, -1), row)
    for row, col in grid_pos:
        if row != last_row_in_col[col]:
            axes_grid[row, col].tick_params(labelbottom=False)

    fig.supxlabel(xlabel_text_final, fontsize=xlabel_fontsize, y=xlabel_y)
    fig.supylabel(ylabel_text, fontsize=ylabel_fontsize, x=ylabel_x)
    fig.suptitle(title_text if title_text is not None
                else f"Lyα line panels — {stack_method} stack",
                fontsize=title_fontsize, y=title_y_final)
    plt.tight_layout()
    if reserve_top is not None:
        fig.subplots_adjust(top=reserve_top)
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes


# NOTE: plot_continuum_check moved to validation.py (it is a continuum CHECK,
# i.e. a robustness/diagnostic test, not a science measurement). Use:
#     from utils_lya_halo import validation
#     validation.plot_continuum_check(stacks)


# =====================================================================
# 4.  RADIAL PROFILE PLOTS
# =====================================================================

def plot_centroid_profile(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    ylims=(-250, 250),
    xlims=None,
    z_err_kms: float = 0,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    plot_literature: bool = False,
    figsize=(7.2, 4.7),
    title: str | None = r"Bootstrapped Ly$\alpha$ centroid vs. radius",
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    The headline figure: centroid velocity ± bootstrap 16/84 per radial bin.
    Thin wrapper around plotting.plot_centroid_vs_radius with sane defaults.

    Accepts the full `boot` dict from run_measurement_suite (which also carries
    r_edges, VR_biweight_v, etc.), so you just call:

        analysis.plot_centroid_profile(boot, stacks)

    title     : axis title; pass None to omit the title entirely.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_centroid_profile.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present.
    """
    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    savename = _resolve_savename(save_name, "Figure_centroid_profile.png")
    return plot_centroid_vs_radius(
        boot, radial_bins, VR_biweight_v=VR_biweight_v,
        stacks_result=stacks or boot,
        bin_mode=bin_mode, z_err_kms=z_err_kms,
        vr_ticks=vr_ticks, figsize=figsize, title=title,
        ylims=ylims, xlims=xlims,
        show_vr=show_vr, VR_biweight_error=VR_biweight_error,
        plot_literature=plot_literature,
        save_fig=save_fig,
        savename=savename,
    )


def plot_flux_profile(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    logy: bool = True,
    ylims=None,
    xlims=None,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    figsize=(7.2, 4.7),
    title: str | None = "Integrated Lyα flux vs. radius",
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Integrated Lyα flux (total_flux_fid) ± bootstrap 16/84 vs radius.
    Log-y (the default) is strongly recommended for seeing the faint outer bins;
    pass logy=False for a linear axis.

    Requires compute_side_ratio=True in the bootstrap (the default), which
    provides total_flux_fid = blue_flux_fid + red_flux_fid.

    title     : axis title; pass None to omit the title entirely.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_flux_profile.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present.
    """
    if "total_flux_fid" not in boot:
        raise KeyError("boot does not contain total_flux_fid; re-run with "
                       "compute_side_ratio=True (the default).")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks or boot)

    y    = np.asarray(boot["total_flux_fid"])
    y_lo = np.asarray(boot["total_flux_lo"])
    y_hi = np.asarray(boot["total_flux_hi"])

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims,
                                     show_vr=show_vr, VR_biweight_error=vr_e)
    yerr, unstable = _safe_yerr(y, y_lo, y_hi)
    ax.errorbar(r_mid, y, xerr=xerr, yerr=yerr,
                fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:blue",
                label="integrated flux (bootstrap 16–84)")
    if np.any(unstable):
        ax.scatter(r_mid[unstable], y[unstable], s=70, facecolors="none",
                   edgecolors="tab:blue", linewidths=1.3, zorder=5,
                   label="fiducial outside 16–84 band")
    ax.axhline(0, color="0.7", lw=0.7)
    if logy:
        pos = y[y > 0]
        if len(pos):
            ax.set_yscale("log")
            ax.set_ylim(pos.min() * 0.3, y.max() * 3)
    elif ylims is not None:
        ax.set_ylim(ylims)
    unit = (boot.get("unit_info") or {}).get("y_unit", "")
    ax.set_ylabel(f"Integrated Lyα flux [{unit}]" if unit else "Integrated Lyα flux")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_flux_profile.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_flux_profile_fit(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    method: str = "psf",
    fit_skip_inner: int = 1,
    psf_r=None,
    psf_vals=None,
    psf_fwhm_arcsec: float = 1.3,
    psf_beta: float = 3.0,
    z_median: float | None = None,
    p0=None,
    r_fine=None,
    logy: bool = True,
    ylims=None,
    xlims=None,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    show_components: bool = True,
    figsize=(10, 5),
    title: str | None = _AUTO,
    save_fig: bool = False,
    save_name: str | None = None,
    verbose: bool = True,
):
    """
    Same headline figure as plot_flux_profile (integrated/mean Lyα flux ±
    bootstrap 16-84 vs radius) but with a two-component exponential
    I(r) = A1*exp(-r/h1) + A2*exp(-r/h2) fit overlaid, using the fitting.py
    infrastructure (see fitting.py's module docstring -- the same code this
    fits with is validated against known-truth synthetic data in
    ../psf_exponential_recovery.py).

    boot['total_flux_fid'] is treated as a per-bin AVERAGE flux (not a sum;
    see fitting.bin_average_no_psf / bin_average_psf), matching how the real
    stack is biweight-averaged within each radial annulus.

    method : "psf" (default) -- forward-models the PSF (ring-convolution ->
             per-bin mean), fits against ALL bins including the innermost
             one (the PSF model accounts for its core contamination
             directly, so nothing needs to be dropped).
             "naive" -- no PSF correction; fits the bin-averaged intrinsic
             profile directly, after dropping the innermost fit_skip_inner
             bin(s) (default 1, since the innermost bin is the one most
             smeared by the PSF core for a naive/no-PSF fit; pass
             fit_skip_inner=0 to keep every bin).

    psf_r, psf_vals : explicit PSF curve (e.g. from
             starpsf.psf_profiles_for_lines), in the SAME radial unit as
             `bin_mode` (typically kpc -- pass r_edges/bin_mode='kpc' for a
             physically meaningful h1/h2 when fitting with a PSF). Only used
             for method="psf". If not given, falls back to an analytic
             Moffat PSF built from psf_fwhm_arcsec/psf_beta, converted to
             kpc (see next) -- skips that conversion entirely if you pass
             your own curve here.
    psf_fwhm_arcsec, psf_beta : Moffat PSF parameters used ONLY when
             psf_r/psf_vals are not given. Default 1.3"/beta=3, the Lujan
             Niemeyer (2022) literature fiducial (1.2-1.4" range) shown to
             fit VIRUS stellar profiles well -- superseding the earlier
             generic fwhm=3.0-kpc placeholder per halo-flux-fitting.md's
             recommendation to use ONE PSF convention across this pipeline's
             fits rather than a second, divergent default. Converted to kpc
             via z_median (below) using the SAME angular-diameter-distance
             conversion stack.convert_avg_fiber_bin uses per galaxy -- here
             applied ONCE for the whole (already-stacked) sample. NOTE:
             this changes the fitted h1/h2 slightly relative to the
             previous fwhm=3.0-kpc default (real-data check: h1 shifts from
             16.9 to ~15-16 kpc-ish territory depending on the sample's own
             z_median) -- re-quote any already-written h1/h2/chi2 numbers
             after switching to this default.
    z_median : the sample's representative redshift, used ONLY to convert
             psf_fwhm_arcsec to kpc. None (default) -> read from
             stacks['z_median'] (set automatically by stack.build_stacks).
             Raises a clear error if unavailable from either source and
             method="psf" with no explicit psf_r/psf_vals -- guessing a
             redshift here would silently build the wrong-width PSF.
    p0      : optional explicit (A1, h1, A2, h2) starting guess, passed
             through to fitting.fit_naive/fit_psf_aware. None -> a handful
             of automatic seeds (fitting._default_seeds) are tried and the
             lowest-chi2 one kept -- see fitting.estimate_truth_from_profile
             for a quick standalone ballpark if you want to inspect a seed
             first.
    r_fine  : optional fine radial grid (fitting.default_fine_grid used if
             not given) that the fit integrates over internally.
    logy    : log-y axis (default True, recommended for the faint outer
             bins -- matches plot_flux_profile's default).
    title   : axis title; the default (_AUTO sentinel) auto-generates one
             from `method`; pass a string to override, or None for no title.
    save_name : filename (with or without extension) used when save_fig=True.
             None -> "Figure_flux_profile_fit.png".
    show_vr, VR_biweight_error : same Rvir reference-line behavior as
             plot_flux_profile.
    show_components : True (default) -- in addition to the combined
             I(r)=A1*exp(-r/h1)+A2*exp(-r/h2) curve already drawn, overplot
             each exponential term ALONE (A1*exp(-r/h1) dashed, A2*exp(-r/h2)
             dash-dot) from the SAME fitted popt. Purely a visualization
             addition -- the fit itself (this module's validated model) is
             completely unchanged; this only draws two extra lines so it's
             visually obvious which term dominates at a given radius, rather
             than only seeing the sum (where the steep core term dying out
             and the shallow halo term taking over can look like one smooth
             curve instead of two distinct physical components handing off).
    verbose : print the fitted (A1,h1,A2,h2) +/- errors and chi2/dof via
             fitting.describe_fit.

    Returns
    -------
    (fig, ax, fit_result) : fit_result is the dict from fitting.fit_naive /
        fit_psf_aware (success, A1/h1/A2/h2 + _err, popt, pcov, chi2, dof,
        mask, model_binned, ...), plus a few extra keys stashed here for
        later inspection without recomputing anything:
            method, r_edges, r_mid, r_fine, y, y_lo, y_hi, sigma, bin_mode,
            and (method="psf" only) R, psf_r, psf_vals.
        Keep this around to see exactly how the fit did against your real
        data -- e.g. fit_result['chi2'] / fit_result['dof'],
        fit_result['h1'] +/- fit_result['h1_err'], or re-plot
        fitting.intrinsic_profile(r, *fit_result['popt']) yourself.
    """
    if method not in ("psf", "naive"):
        raise ValueError(f"method must be 'psf' or 'naive' (got {method!r})")
    if "total_flux_fid" not in boot:
        raise KeyError("boot does not contain total_flux_fid; re-run with "
                       "compute_side_ratio=True (the default).")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks or boot)
    if method == "psf" and bm != "kpc" and psf_r is None and verbose:
        print(f"plot_flux_profile_fit: bin_mode={bm!r} but psf_fwhm_arcsec/psf_beta "
              f"build a Moffat converted assuming bin_mode is kpc -- "
              f"pass r_edges/bin_mode='kpc' (or your own psf_r/psf_vals in "
              f"{bm} units) for a physically meaningful PSF-aware fit.")

    y    = np.asarray(boot["total_flux_fid"], dtype=float)
    y_lo = np.asarray(boot["total_flux_lo"], dtype=float)
    y_hi = np.asarray(boot["total_flux_hi"], dtype=float)
    # per-bin sigma for the fit: average of the up/down bootstrap 16-84
    # half-widths (a symmetric 1-sigma-equivalent from an asymmetric band).
    sigma = ((y_hi - y) + (y - y_lo)) / 2.0

    r_fine_arr = (np.asarray(r_fine, dtype=float) if r_fine is not None
                 else fitting.default_fine_grid(radial_bins))
    r_mid = 0.5 * (radial_bins[:-1] + radial_bins[1:])

    if method == "psf":
        if psf_r is not None and psf_vals is not None:
            psf_r_use = np.asarray(psf_r, dtype=float)
            psf_vals_use = np.asarray(psf_vals, dtype=float)
        else:
            z_med = z_median if z_median is not None else (stacks or {}).get("z_median")
            fwhm_kpc = _psf_fwhm_arcsec_to_kpc(psf_fwhm_arcsec, z_med, sample_label="this fit")
            if verbose:
                print(f"plot_flux_profile_fit: PSF FWHM = {psf_fwhm_arcsec:.2f}\" -> "
                      f"{fwhm_kpc:.3g} kpc (z_median={z_med:.3g}, beta={psf_beta:g})")
            psf_r_use = np.linspace(0.0, 20.0 * fwhm_kpc, 400)
            psf_vals_use = fitting.moffat_1d(psf_r_use, fwhm=fwhm_kpc, beta=psf_beta)
        R = fitting.ring_convolution_matrix(r_fine_arr, radial_bins, psf_r_use, psf_vals_use)
        fit_result = fitting.fit_psf_aware(r_mid, y, sigma, R, r_fine_arr, radial_bins,
                                           p0=p0, verbose=verbose)
        fit_result["R"] = R
        fit_result["psf_r"] = psf_r_use
        fit_result["psf_vals"] = psf_vals_use
    else:
        fit_result = fitting.fit_naive(r_mid, radial_bins, r_fine_arr, y, sigma,
                                       fit_skip_inner=fit_skip_inner, p0=p0,
                                       verbose=verbose)

    fit_result["method"] = method
    fit_result["r_edges"] = radial_bins
    fit_result["r_mid"] = r_mid
    fit_result["r_fine"] = r_fine_arr
    fit_result["y"] = y
    fit_result["y_lo"] = y_lo
    fit_result["y_hi"] = y_hi
    fit_result["sigma"] = sigma
    fit_result["bin_mode"] = bm

    fig, ax = plt.subplots(figsize=figsize)
    r_mid_native, xerr = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims,
                                            show_vr=show_vr, VR_biweight_error=vr_e)
    yerr, unstable = _safe_yerr(y, y_lo, y_hi)
    ax.errorbar(r_mid_native, y, xerr=xerr, yerr=yerr,
                fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:blue",
                label="observed (bootstrap 16-84)", zorder=5)
    if np.any(unstable):
        ax.scatter(r_mid_native[unstable], y[unstable], s=70, facecolors="none",
                   edgecolors="tab:blue", linewidths=1.3, zorder=6,
                   label="fiducial outside 16-84 band")

    if method == "naive" and fit_result.get("mask") is not None:
        dropped = ~fit_result["mask"]
        if np.any(dropped):
            ax.scatter(r_mid_native[dropped], y[dropped], s=130, facecolors="none",
                       edgecolors="tab:blue", linewidths=1.6, zorder=7,
                       label=f"dropped by naive fit (inner {fit_skip_inner})")

    if fit_result.get("success"):
        popt = fit_result["popt"]
        A1_f, h1_f, A2_f, h2_f = popt
        color = "tab:green" if method == "psf" else "tab:orange"
        chi2_txt = (f", $\\chi^2$/dof={fit_result['chi2']/max(fit_result['dof'],1):.2f}"
                    if "chi2" in fit_result else "")
        ax.plot(r_fine_arr, fitting.intrinsic_profile(r_fine_arr, *popt), "-",
                color=color, lw=1.8, zorder=3,
                label=(f"{'PSF-aware' if method == 'psf' else 'naive'} fit (sum)  "
                       f"(h1={h1_f:.2g}, h2={h2_f:.2g}{chi2_txt})"))
        model_binned = fitting.binned_model_from_result(
            fit_result, r_fine_arr, radial_bins, fit_result.get("R"))
        if model_binned is not None:
            ax.plot(r_mid_native, model_binned,
                    "^" if method == "psf" else "s",
                    color=color, ms=7, mfc="none" if method == "naive" else color,
                    zorder=4, label="fit predicted bin mean")
        if show_components:
            # Each exponential term alone, from the SAME fitted popt -- no
            # new fitting here, just splitting the already-drawn sum back
            # into its two physical pieces so it's visible which one is
            # actually doing the work at a given radius (rather than only
            # seeing one smooth combined curve where the steep term dies
            # out and the shallow term takes over).
            ax.plot(r_fine_arr, A1_f * np.exp(-r_fine_arr / h1_f), "--",
                    color=color, lw=1.3, alpha=0.65, zorder=2,
                    label=f"  core term alone (h1={h1_f:.2g})")
            ax.plot(r_fine_arr, A2_f * np.exp(-r_fine_arr / h2_f), "-.",
                    color=color, lw=1.3, alpha=0.65, zorder=2,
                    label=f"  halo term alone (h2={h2_f:.2g})")
    elif verbose:
        print(f"plot_flux_profile_fit: fit FAILED -- {fit_result.get('reason')}")

    ax.axhline(0, color="0.7", lw=0.7)
    if logy:
        pos = y[y > 0]
        if len(pos):
            ax.set_yscale("log")
            ax.set_ylim(pos.min() * 0.3, y.max() * 3)
    elif ylims is not None:
        ax.set_ylim(ylims)
    unit = (boot.get("unit_info") or {}).get("y_unit", "")
    ax.set_ylabel(f"Integrated Lyα flux [{unit}]" if unit else "Integrated Lyα flux")
    title_final = (f"{'PSF-aware' if method == 'psf' else 'Naive'} double-exponential fit"
                   if title is _AUTO else title)
    if title_final:
        ax.set_title(title_final)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_flux_profile_fit.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()

    if verbose:
        fitting.describe_fit(fit_result, label=f"{method} fit vs real data")

    return fig, ax, fit_result


def plot_flux_curve_of_growth(
    cog: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    logy: bool = True,
    xlims=None,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    figsize=None,
    panels: str = "both",
    title1: str | None = _AUTO,
    title2: str | None = "Lyα flux fraction vs. radius",
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Lyα curve of growth: cumulative luminosity vs radius (center outward), and
    the same normalized to a flux FRACTION of the total summed out to r_max.
    Both panels show the bootstrap 16-84 band.

    Takes `cog`, the dict returned by measure.flux_curve_of_growth -- this
    function only DRAWS it (measure.py computes, analysis.py draws, same split
    as every other profile here). `stacks` is optional and only used as a
    fallback for bin_mode / VR_biweight_v (cog itself does not carry them):

        cog = measure.flux_curve_of_growth(boot, stacks)   # r_max_kpc=None ->
                                                             # drops the last bin
        analysis.plot_flux_curve_of_growth(cog, stacks)

    The top panel's y-unit is derived from cog['unit_info']['y_unit'] with the
    '/kpc^2' stripped (multiplying by the fiber area cancels it) -- same
    fiber-footprint convention as plot_flux_profile's total_flux_fid, not a
    true annulus-integrated luminosity (see measure.flux_curve_of_growth).

    panels    : "both" (default, two stacked panels) | "cumulative" (top panel
                only) | "fraction" (bottom panel only). Use the single-panel
                modes to grab just one of the two images, e.g. to save it on
                its own with a distinct save_name.
    title1    : cumulative-luminosity panel title; omitted (_AUTO, the
                default) auto-generates "Lyα curve of growth (r_max = ... kpc)";
                pass a string to override, or None for no title.
    title2    : flux-fraction panel title; pass None for no title.
    figsize   : None -> (7.2, 8.0) for panels="both", (7.2, 4.5) for a single
                panel.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_flux_curve_of_growth.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present. (cog itself does not carry this -- it comes from
                `stacks`, same fallback as VR_biweight_v.)

    Returns (fig, (ax1, ax2)) for panels="both", else (fig, ax).
    """
    for key in ("flux_cumulative_fid", "flux_fraction_fid"):
        if key not in cog:
            raise KeyError(f"cog missing {key!r}; pass the dict returned by "
                           "measure.flux_curve_of_growth.")
    if panels not in ("both", "cumulative", "fraction"):
        raise ValueError("panels must be 'both', 'cumulative', or 'fraction'")

    radial_bins = np.asarray(r_edges if r_edges is not None else cog["r_edges_used"])
    bm  = _resolve_bin_mode(bin_mode, stacks)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks)

    if figsize is None:
        figsize = (7.2, 8.0) if panels == "both" else (7.2, 4.5)

    if panels == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    elif panels == "cumulative":
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=figsize)
        ax1 = None

    # panel 1: cumulative luminosity
    if ax1 is not None:
        r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims,
                                         show_vr=show_vr, VR_biweight_error=vr_e)
        y    = np.asarray(cog["flux_cumulative_fid"])
        y_lo = np.asarray(cog["flux_cumulative_lo"])
        y_hi = np.asarray(cog["flux_cumulative_hi"])
        yerr, unstable = _safe_yerr(y, y_lo, y_hi)
        ax1.errorbar(r_mid, y, xerr=xerr, yerr=yerr,
                    fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:green",
                    label="cumulative flux (bootstrap 16–84)")
        if np.any(unstable):
            ax1.scatter(r_mid[unstable], y[unstable], s=70, facecolors="none",
                       edgecolors="tab:green", linewidths=1.3, zorder=5,
                       label="fiducial outside 16–84 band")
        ax1.axhline(0, color="0.7", lw=0.7)
        if logy:
            pos = y[y > 0]
            if len(pos):
                ax1.set_yscale("log")
                ax1.set_ylim(pos.min() * 0.3, y.max() * 3)
        base_unit = (cog.get("unit_info") or {}).get("y_unit", "")
        lum_unit = base_unit.replace("/kpc^2", "") if "kpc^2" in base_unit else base_unit
        ax1.set_ylabel(f"Cumulative Lyα luminosity [{lum_unit}]" if lum_unit
                       else "Cumulative Lyα luminosity")
        if title1 is _AUTO:
            ax1.set_title(f"Lyα curve of growth (r_max = {cog['meta']['r_max_kpc']:g} kpc)")
        elif title1:
            ax1.set_title(title1)
        ax1.legend(frameon=False, fontsize=9)
        ax1.grid(alpha=0.15)

    # panel 2: flux fraction
    if ax2 is not None:
        r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims,
                                           show_vr=show_vr, VR_biweight_error=vr_e)
        fy    = np.asarray(cog["flux_fraction_fid"])
        fy_lo = np.asarray(cog["flux_fraction_lo"])
        fy_hi = np.asarray(cog["flux_fraction_hi"])
        fyerr, funstable = _safe_yerr(fy, fy_lo, fy_hi)
        ax2.errorbar(r_mid2, fy, xerr=xerr2, yerr=fyerr,
                    fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:orange",
                    label="flux fraction (bootstrap 16–84)")
        if np.any(funstable):
            ax2.scatter(r_mid2[funstable], fy[funstable], s=70, facecolors="none",
                       edgecolors="tab:orange", linewidths=1.3, zorder=5)
        ax2.axhline(0.5, color="0.5", lw=0.8, ls="--", label="half-light radius")
        ax2.axhline(1.0, color="0.7", lw=0.6)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel(r"Flux fraction  $L(<r) / L(<r_{\rm max})$")
        if title2:
            ax2.set_title(title2)
        ax2.legend(frameon=False, fontsize=9)
        ax2.grid(alpha=0.15)

    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_flux_curve_of_growth.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    if panels == "both":
        return fig, (ax1, ax2)
    return fig, (ax1 if ax1 is not None else ax2)


def plot_asymmetry_profile(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    xlims=None,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    figsize=None,
    panels: str = "both",
    title1: str | None = "Lyα asymmetry: blue flux fraction",
    title2: str | None = "Blue and red side flux vs. radius",
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Asymmetry diagnostic: blue fraction B/(B+R) vs radius with the 16/84
    bootstrap band, and absolute blue and red flux vs radius on the same axis
    so you can see which side is driving the asymmetry.

    fraction = blue_flux / (blue_flux + red_flux), bounded to [0, 1]:
    = 0.5 → symmetric line.
    > 0.5 → more blue flux (blueshifted / absorption-driven?).
    < 0.5 → more red flux (the expected resonant scattering signature).

    Replaces the old blue/red ratio (which ran 0→∞ and put the symmetric point
    at 1). The fraction is computed PER bootstrap draw from blue_flux_all /
    red_flux_all so the 16/84 band is correct (you cannot divide percentiles).

    panels    : "both" (default, two stacked panels) | "fraction" (blue-fraction
                panel only) | "flux" (blue/red flux panel only). Use the
                single-panel modes to grab just one of the two images.
    title1    : blue-fraction panel title; None for no title.
    title2    : blue/red-flux panel title; None for no title.
    figsize   : None -> (7.2, 8.0) for panels="both", (7.2, 4.5) for a single
                panel.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_asymmetry_profile.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present.

    Returns (fig, (ax1, ax2)) for panels="both", else (fig, ax).
    """
    if "blue_over_red_fid" not in boot:
        raise KeyError("boot missing blue/red keys; re-run with compute_side_ratio=True.")
    if panels not in ("both", "fraction", "flux"):
        raise ValueError("panels must be 'both', 'fraction', or 'flux'")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks or boot)

    # --- blue fraction per draw -> correct 16/84 (fall back to fid-only) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        bf_fid = np.asarray(boot["blue_flux_fid"], float)
        rf_fid = np.asarray(boot["red_flux_fid"], float)
        bfrac_fid = bf_fid / (bf_fid + rf_fid)

        if "blue_flux_all" in boot and "red_flux_all" in boot:
            ball = np.asarray(boot["blue_flux_all"], float)   # (nboot, nrad)
            rall = np.asarray(boot["red_flux_all"], float)
            bfrac_all = ball / (ball + rall)
            bfrac_lo = np.nanpercentile(bfrac_all, 16, axis=0)
            bfrac_hi = np.nanpercentile(bfrac_all, 84, axis=0)
        else:
            # no per-draw arrays: approximate band from blue/red lo/hi
            warnings.warn("blue_flux_all/red_flux_all not in boot; approximating "
                          "B/total error bars from blue_flux_lo/hi.")
            bl = np.asarray(boot["blue_flux_lo"], float); bh = np.asarray(boot["blue_flux_hi"], float)
            rl = np.asarray(boot["red_flux_lo"],  float); rh = np.asarray(boot["red_flux_hi"],  float)
            bfrac_lo = bl / (bl + rh)
            bfrac_hi = bh / (bh + rl)

    if figsize is None:
        figsize = (7.2, 8.0) if panels == "both" else (7.2, 4.5)

    if panels == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    elif panels == "fraction":
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=figsize)
        ax1 = None

    # panel 1: blue fraction
    if ax1 is not None:
        r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims,
                                         show_vr=show_vr, VR_biweight_error=vr_e)
        yerr1, unstable1 = _safe_yerr(bfrac_fid, bfrac_lo, bfrac_hi)
        ax1.errorbar(r_mid, bfrac_fid, xerr=xerr, yerr=yerr1,
                     fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:purple")
        if np.any(unstable1):
            ax1.scatter(r_mid[unstable1], bfrac_fid[unstable1], s=70, facecolors="none",
                        edgecolors="tab:purple", linewidths=1.3, zorder=5,
                        label="fiducial outside 16–84 band")
        ax1.axhline(0.5, color="0.5", lw=0.8, ls="--", label="symmetric (B/total = 0.5)")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Blue fraction  B / (B+R)")
        if title1:
            ax1.set_title(title1)
        ax1.legend(frameon=False, fontsize=9)
        ax1.grid(alpha=0.15)

    # panel 2: absolute fluxes
    if ax2 is not None:
        r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims,
                                           show_vr=show_vr, VR_biweight_error=vr_e)
        bf    = np.asarray(boot["blue_flux_fid"])
        rf    = np.asarray(boot["red_flux_fid"])
        bf_lo = np.asarray(boot["blue_flux_lo"]);  bf_hi = np.asarray(boot["blue_flux_hi"])
        rf_lo = np.asarray(boot["red_flux_lo"]);   rf_hi = np.asarray(boot["red_flux_hi"])
        dx = (r_mid2[1] - r_mid2[0]) * 0.05 if len(r_mid2) > 1 else 0
        bf_yerr, bf_unstable = _safe_yerr(bf, bf_lo, bf_hi)
        rf_yerr, rf_unstable = _safe_yerr(rf, rf_lo, rf_hi)
        ax2.errorbar(r_mid2 - dx, bf, xerr=xerr2, yerr=bf_yerr,
                     fmt="o", capsize=3, ms=5, lw=1.3, color="royalblue", label="blue flux")
        ax2.errorbar(r_mid2 + dx, rf, xerr=xerr2, yerr=rf_yerr,
                     fmt="s", capsize=3, ms=5, lw=1.3, color="tomato", label="red flux")
        if np.any(bf_unstable):
            ax2.scatter((r_mid2 - dx)[bf_unstable], bf[bf_unstable], s=60, facecolors="none",
                        edgecolors="royalblue", linewidths=1.3, zorder=5)
        if np.any(rf_unstable):
            ax2.scatter((r_mid2 + dx)[rf_unstable], rf[rf_unstable], s=60, facecolors="none",
                        edgecolors="tomato", linewidths=1.3, zorder=5)
        ax2.axhline(0, color="0.7", lw=0.6)
        ax2.set_ylabel("Flux")
        if title2:
            ax2.set_title(title2)
        ax2.legend(frameon=False, fontsize=9)
        ax2.grid(alpha=0.15)

    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_asymmetry_profile.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    if panels == "both":
        return fig, (ax1, ax2)
    return fig, (ax1 if ax1 is not None else ax2)


# =====================================================================
# 4b.  TWO-STACK (HALF-SAMPLE) OVERLAY PROFILES
# =====================================================================
# These put two boot/stacks pairs (e.g. low-mass vs high-mass) on ONE figure,
# instead of calling the single-stack version twice. Same native-unit radius
# axis as the single-stack figures, so they drop straight into the two-stack
# notebook. Self-contained (no longer thin wrappers around the shared
# plotting.py multi-sample engines) so every color/marker/font-size is a
# keyword argument here, same philosophy as plot_line_panels. NO TITLE is
# drawn by default (paper-figure friendly, drop straight into a manuscript
# grid) -- pass a string to title/title1/title2 to add one back.

def plot_centroid_profile_two(
    boot_a: dict,
    boot_b: dict,
    stacks_a: dict | None = None,
    stacks_b: dict | None = None,
    labels=("A", "B"),
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    VR_biweight_error: float | None = None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    show_vr: bool = True,
    ylims=(-250, 250),
    xlims=None,
    jitter: float = 0.04,
    colors=("tab:blue", "tab:red"),
    fmts=("o-", "s--"),
    markersize: float = 6,
    linewidth: float = 1.5,
    capsize: float = 3.5,
    sample_b_alpha: float = 0.7,
    zero_color: str = "tomato",
    zero_alpha: float = 0.4,
    zero_lw: float = 1.0,
    figsize=(7.6, 4.9),
    ylabel_text: str = r"Ly$\alpha$ centroid velocity [km s$^{-1}$]",
    title: str | None = None,
    legend_title: str | None = "sample",
    legend_fontsize: float = 9,
    legend_loc: str = "best",
    tick_fontsize: float = 10,
    label_fontsize: float = 11,
    title_fontsize: float = 13,
    grid_alpha: float = 0.15,
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Overlay the bootstrapped centroid profiles of TWO samples on one axis
    (half-sample comparison, e.g. low-M vs high-M). Replaces calling
    plot_centroid_profile twice.

    Self-contained (not a plotting.py engine wrapper) so every color, marker,
    and font size below is a keyword argument -- fully restylable without
    touching the function body. Sample A is drawn solid/filled, sample B
    open/dashed (colors, fmts, sample_b_alpha all overridable).

    title     : axis title; None (default) -> no title at all. Pass a string
                to add one back.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_centroid_profile_two.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks_a['VR_biweight_e'] if present.

    Example
    -------
        analysis.plot_centroid_profile_two(
            boot_lm, boot_hm, stacks_low_m, stacks_high_m,
            labels=("low-M", "high-M"),
            xlims=(3, 3000), bin_mode="kpc",
            ylims=(-300, 300), figsize=(9, 5),
            colors=("tab:blue", "tab:red"), jitter=0.05,
            title="Centroid comparison, low-M vs high-M",
        )

    Returns (fig, ax).
    """
    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot_a.get("r_edges",
                             stacks_a["r_edges"] if stacks_a else []))
    bm   = _resolve_bin_mode(bin_mode, stacks_a or boot_a)
    vr   = _get_vr_biweight_v(VR_biweight_v, stacks_a or boot_a)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks_a or boot_a)

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims,
                                     show_vr=show_vr, VR_biweight_error=vr_e)

    K = 2
    for k, (lab, b, col) in enumerate(((labels[0], boot_a, colors[0]),
                                        (labels[1], boot_b, colors[1]))):
        v  = np.asarray(b["centroid_v_med"])
        lo = np.asarray(b["centroid_v_lo"]); hi = np.asarray(b["centroid_v_hi"])
        jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
        yerr, unstable = _safe_yerr(v, lo, hi)
        ax.errorbar(jit, v, xerr=xerr, yerr=yerr, fmt=fmts[k],
                    color=col, capsize=capsize, ms=markersize, lw=linewidth,
                    alpha=1.0 if k == 0 else sample_b_alpha,
                    markerfacecolor=(col if k == 0 else "none"),
                    label=lab)
        if np.any(unstable):
            ax.scatter(jit[unstable], v[unstable], s=70, facecolors="none",
                       edgecolors=col, linewidths=1.3, zorder=5)

    ax.axhline(0, color=zero_color, alpha=zero_alpha, lw=zero_lw)
    ax.set_ylim(ylims)
    ax.set_ylabel(ylabel_text, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.legend(frameon=False, fontsize=legend_fontsize, loc=legend_loc, title=legend_title)
    ax.grid(alpha=grid_alpha)
    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_centroid_profile_two.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_flux_profile_two(
    boot_a: dict,
    boot_b: dict,
    stacks_a: dict | None = None,
    stacks_b: dict | None = None,
    labels=("A", "B"),
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    VR_biweight_error: float | None = None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    show_vr: bool = True,
    logy: bool = True,
    ylims=None,
    xlims=None,
    jitter: float = 0.04,
    colors=("tab:blue", "tab:red"),
    fmts=("o-", "s--"),
    markersize: float = 6,
    linewidth: float = 1.5,
    capsize: float = 3.5,
    sample_b_alpha: float = 0.7,
    zero_color: str = "0.7",
    zero_lw: float = 0.7,
    fit: bool = False,
    fit_method: str = "psf",
    fit_skip_inner: int = 1,
    gamma_fixed: float | None = 0.8,
    psf_r=None,
    psf_vals=None,
    psf_fwhm_arcsec: float = 1.3,
    psf_beta: float = 3.0,
    z_median_a: float | None = None,
    z_median_b: float | None = None,
    fit_p0=None,
    fit_r_fine=None,
    fit_show_components: bool = True,
    fit_verbose: bool = True,
    figsize=(7.6, 4.9),
    ylabel_text: str | None = None,
    title: str | None = None,
    legend_title: str | None = "sample",
    legend_fontsize: float = 9,
    legend_loc: str = "best",
    tick_fontsize: float = 10,
    label_fontsize: float = 11,
    title_fontsize: float = 13,
    grid_alpha: float = 0.15,
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Overlay integrated Lyα flux +/- bootstrap 16/84 for TWO samples on one axis.
    Replaces calling plot_flux_profile twice. logy=True by default (faint outer
    bins). Requires compute_side_ratio=True in the bootstrap (the default).

    Self-contained (not a plotting.py engine wrapper) so every color, marker,
    and font size below is a keyword argument. For a LINEAR axis (logy=False)
    with ylims=None, the y-limits are taken from BOTH samples (union of their
    finite ranges) so neither is clipped; on the log axis the scale already
    spans both.

    title     : axis title; None (default) -> no title at all. Pass a string
                to add one back.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_flux_profile_two.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks_a['VR_biweight_e'] if present.

    Fitting overlay (subsample-derived-properties.md, Part 4)
    -----------------------------------------------------------
    fit : False (default, unchanged behavior) -- pass True to independently
        fit EACH sample's flux profile and overlay the fitted curve on this
        same comparison figure, instead of only drawing the two error-bar
        series. Uses the PSF-aware exponential-core + cored-power-law model
        (fitting.intrinsic_profile_expcore / fit_psf_aware_expcore --
        halo-flux-fitting.md Part 2/"Option C"), NOT the older validated
        two-exponential model plot_flux_profile_fit uses -- deliberately, per
        the explicit ask that drove this extension ("fit with a PSF-aware
        exponential core with a power law"). Each sample is fit completely
        independently (its own popt, its own chi2/dof); nothing about the
        fit is shared between samples except the model form.
    fit_method : "psf" (default) -- full PSF forward model, all bins fit
        (inner bin kept). "naive" -- no PSF correction, drops fit_skip_inner
        innermost bin(s). Same semantics as plot_flux_profile_fit's `method`.
    gamma_fixed : 0.8 (default) -- fitting.projected_slope_from_3d(1.8), the
        Limber-projected z~2-3 clustering slope. r_c and a floating gamma are
        strongly degenerate on real data (even the highest-S/N ALL stack
        fits gamma essentially unconstrained, and compare_models_aic_bic
        prefers the plain two-exponential model over expcore regardless),
        so a free gamma mainly adds noise -- and does so worse in lower-S/N
        per-subsample fits (this is what motivated the change from a
        floating default). Pass None to let gamma float anyway, or another
        literature value (e.g. 1.8, unprojected) to test it directly via
        chi2/AIC -- see fitting.fit_psf_aware_expcore's docstring.
    psf_r, psf_vals : explicit shared (r, vals) PSF curve, in kpc -- if
        given, used for BOTH samples as-is (skips the arcsec->kpc
        conversion below entirely). None (default) -> built per SAMPLE from
        psf_fwhm_arcsec/psf_beta instead (see next).
    psf_fwhm_arcsec, psf_beta : the fixed literature Moffat PSF -- default
        1.3" / beta=3, the Lujan Niemeyer (2022) fiducial (1.2-1.4" range)
        that fits VIRUS stellar profiles well, per subsample-derived-
        properties.md Part 3's recommendation to use ONE literature PSF
        convention everywhere in this pipeline rather than a second,
        divergent default just for this fit. Converted to kpc SEPARATELY
        for each sample using that sample's own z_median (stacks_a/
        stacks_b['z_median'], as stored by build_stacks -- or explicit
        z_median_a/z_median_b if you don't have the stacks dicts handy) --
        deliberately per-sample, not shared: a low-z vs. high-z split is
        exactly the case where the same fixed angular PSF corresponds to a
        physically different kpc width for each sample. Raises a clear
        error (rather than silently guessing) if a sample's z_median isn't
        available from either source. Requires bin_mode='kpc' (a warning is
        printed, same as plot_flux_profile_fit, if bin_mode != 'kpc').
    z_median_a, z_median_b : explicit override for the arcsec->kpc PSF
        conversion above; None (default) -> read from stacks_a['z_median']
        / stacks_b['z_median'].
    fit_p0 : optional explicit (A1, h1, A2, r_c[, gamma]) starting guess per
        sample (SAME seed used for both samples); None (default) -> automatic
        multi-seed search (fitting._default_seeds_expcore) per sample.
    fit_r_fine : optional shared fine radial grid for both fits (
        fitting.default_fine_grid(radial_bins) used if not given).
    fit_show_components : True (default) -- in addition to each sample's
        combined fitted curve, overplot its core term alone (dashed) and halo
        term alone (dash-dot), in that sample's own color -- mirrors
        plot_flux_profile_fit's show_components, applied per sample here.
    fit_verbose : print each sample's fitting.describe_fit_expcore summary.

    Example
    -------
        analysis.plot_flux_profile_two(
            boot_lm, boot_hm, stacks_low_m, stacks_high_m,
            labels=("low-M", "high-M"),
            xlims=(3, 3000), bin_mode="kpc",
            logy=True, figsize=(9, 5),
            colors=("tab:blue", "tab:red"),
            fit=True, gamma_fixed=0.8,   # Limber-projected z~2-3 clustering slope (default)
        )

    Returns
    -------
    (fig, ax) when fit=False (unchanged from before this extension).
    (fig, ax, fit_result_a, fit_result_b) when fit=True -- each fit_result is
    the dict from fitting.fit_psf_aware_expcore / fit_naive_expcore for that
    sample (success, A1/h1/A2/r_c/gamma (+ _err), popt, pcov, chi2, dof,
    model_binned, mask, ...), plus r_edges/r_mid/r_fine/y/y_lo/y_hi/sigma
    stashed for later inspection. This is exactly the dict
    fitting.find_core_halo_boundary / measure.measure_outer_properties /
    measure.measure_psf_corrected_core_luminosity consume for the
    core/halo-boundary and derived-property pipeline in
    subsample-derived-properties.md -- so the SAME fit call backs both this
    figure and that downstream table, never two independent fits.
    """
    for b in (boot_a, boot_b):
        if "total_flux_fid" not in b:
            raise KeyError("boot missing total_flux_fid; re-run with "
                           "compute_side_ratio=True (the default).")
    if fit and fit_method not in ("psf", "naive"):
        raise ValueError(f"fit_method must be 'psf' or 'naive' (got {fit_method!r})")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot_a.get("r_edges",
                             stacks_a["r_edges"] if stacks_a else []))
    bm   = _resolve_bin_mode(bin_mode, stacks_a or boot_a)
    vr   = _get_vr_biweight_v(VR_biweight_v, stacks_a or boot_a)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks_a or boot_a)
    if fit and fit_method == "psf" and bm != "kpc" and psf_r is None and fit_verbose:
        print(f"plot_flux_profile_two: bin_mode={bm!r} but psf_fwhm/psf_beta "
              f"build a Moffat assumed to be in the SAME unit as bin_mode -- "
              f"pass r_edges/bin_mode='kpc' (or your own psf_r/psf_vals in "
              f"{bm} units) for a physically meaningful PSF-aware fit.")

    series = []
    for lab, b, col in ((labels[0], boot_a, colors[0]), (labels[1], boot_b, colors[1])):
        series.append(dict(
            label=lab, color=col,
            y=np.asarray(b["total_flux_fid"], float),
            lo=np.asarray(b["total_flux_lo"], float),
            hi=np.asarray(b["total_flux_hi"], float),
        ))

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims,
                                     show_vr=show_vr, VR_biweight_error=vr_e)

    K = 2
    for k, s in enumerate(series):
        jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
        yerr, unstable = _safe_yerr(s["y"], s["lo"], s["hi"])
        ax.errorbar(jit, s["y"], xerr=xerr, yerr=yerr, fmt=fmts[k],
                    color=s["color"], capsize=capsize, ms=markersize, lw=linewidth,
                    alpha=1.0 if k == 0 else sample_b_alpha,
                    markerfacecolor=(s["color"] if k == 0 else "none"),
                    label=s["label"])
        if np.any(unstable):
            ax.scatter(jit[unstable], s["y"][unstable], s=70, facecolors="none",
                       edgecolors=s["color"], linewidths=1.3, zorder=5)

    fit_results = []
    if fit:
        # LINEAR bin midpoints for the fit itself (matches
        # plot_flux_profile_fit exactly -- fitting.py's model is evaluated/
        # bin-integrated against r_edges, not the geometric r_mid used only
        # for the errorbar x-position above); the fitted curve is then drawn
        # against r_fine and its predicted-bin-mean markers against the same
        # geometric r_mid (jittered) the data points use, so everything lines
        # up visually on one axis.
        r_mid_linear = 0.5 * (radial_bins[:-1] + radial_bins[1:])
        r_fine_arr = (np.asarray(fit_r_fine, dtype=float) if fit_r_fine is not None
                     else fitting.default_fine_grid(radial_bins))
        # Build the PSF-aware forward model's R matrix PER SAMPLE (not
        # shared) when it's derived from the fixed literature FWHM -- see
        # _psf_fwhm_arcsec_to_kpc's docstring for why a low-z/high-z split
        # specifically needs two different kpc widths from the same one
        # angular PSF. An explicit psf_r/psf_vals curve, in contrast, IS
        # shared as-is between both samples (it's already in kpc and the
        # caller presumably built it deliberately that way).
        R_by_sample = {}
        if fit_method == "psf":
            if psf_r is not None and psf_vals is not None:
                psf_r_use = np.asarray(psf_r, dtype=float)
                psf_vals_use = np.asarray(psf_vals, dtype=float)
                R_shared = fitting.ring_convolution_matrix(r_fine_arr, radial_bins, psf_r_use, psf_vals_use)
                R_by_sample[0] = R_shared
                R_by_sample[1] = R_shared
            else:
                z_meds = (
                    z_median_a if z_median_a is not None else (stacks_a or {}).get("z_median"),
                    z_median_b if z_median_b is not None else (stacks_b or {}).get("z_median"),
                )
                for kk, z_med in enumerate(z_meds):
                    fwhm_kpc = _psf_fwhm_arcsec_to_kpc(psf_fwhm_arcsec, z_med, sample_label=labels[kk])
                    psf_r_use = np.linspace(0.0, 20.0 * fwhm_kpc, 400)
                    psf_vals_use = fitting.moffat_1d(psf_r_use, fwhm=fwhm_kpc, beta=psf_beta)
                    R_by_sample[kk] = fitting.ring_convolution_matrix(
                        r_fine_arr, radial_bins, psf_r_use, psf_vals_use)
                    if fit_verbose:
                        print(f"plot_flux_profile_two: {labels[kk]} PSF FWHM = "
                              f"{psf_fwhm_arcsec:.2f}\" -> {fwhm_kpc:.3g} kpc "
                              f"(z_median={z_med:.3g}, beta={psf_beta:g})")

        for k, s in enumerate(series):
            sigma = ((s["hi"] - s["y"]) + (s["y"] - s["lo"])) / 2.0
            if fit_method == "psf":
                R = R_by_sample[k]
                fr = fitting.fit_psf_aware_expcore(
                    r_mid_linear, s["y"], sigma, R, r_fine_arr, radial_bins,
                    gamma_fixed=gamma_fixed, p0=fit_p0, verbose=False)
                fr["R"] = R
            else:
                fr = fitting.fit_naive_expcore(
                    r_mid_linear, radial_bins, r_fine_arr, s["y"], sigma,
                    fit_skip_inner=fit_skip_inner, gamma_fixed=gamma_fixed,
                    p0=fit_p0, verbose=False)
            fr.update({"method": fit_method, "r_edges": radial_bins, "r_mid": r_mid_linear,
                      "r_fine": r_fine_arr, "y": s["y"], "y_lo": s["lo"], "y_hi": s["hi"],
                      "sigma": sigma, "bin_mode": bm, "label": s["label"]})
            fit_results.append(fr)

            if fr.get("success"):
                popt = fr["popt"]
                A1_f, h1_f, A2_f, r_c_f = popt[0], popt[1], popt[2], popt[3]
                gamma_f = fr["gamma"]
                params_for_curve = popt if gamma_fixed is None else np.append(popt, gamma_fixed)
                chi2_txt = (f", $\\chi^2$/dof={fr['chi2']/max(fr['dof'],1):.2f}"
                           if "chi2" in fr else "")
                gtag = f"$\\gamma$={gamma_f:.2g}" + (" (fixed)" if fr["gamma_fixed"] else "")
                alpha_k = 1.0 if k == 0 else sample_b_alpha
                jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
                ax.plot(r_fine_arr, fitting.intrinsic_profile_expcore(r_fine_arr, *params_for_curve),
                        "-", color=s["color"], lw=1.8, alpha=alpha_k, zorder=3,
                        label=f"  {s['label']} fit (h1={h1_f:.2g}, r_c={r_c_f:.2g}, {gtag}{chi2_txt})")
                model_binned = fitting.binned_model_from_result_expcore(
                    fr, r_fine_arr, radial_bins, fr.get("R"))
                if model_binned is not None:
                    ax.plot(jit, model_binned, "D" if fit_method == "psf" else "^",
                            color=s["color"], ms=6, alpha=alpha_k,
                            mfc="none" if k == 1 else s["color"], zorder=4)
                if fit_show_components:
                    ax.plot(r_fine_arr, A1_f * np.exp(-r_fine_arr / h1_f), "--",
                            color=s["color"], lw=1.2, alpha=alpha_k * 0.65, zorder=2)
                    ax.plot(r_fine_arr, A2_f * (1.0 + (r_fine_arr / r_c_f) ** 2) ** (-gamma_f / 2.0),
                            "-.", color=s["color"], lw=1.2, alpha=alpha_k * 0.65, zorder=2)
            elif fit_verbose:
                print(f"plot_flux_profile_two: {s['label']} fit FAILED -- {fr.get('reason')}")

        if fit_verbose:
            for fr in fit_results:
                fitting.describe_fit_expcore(fr, label=f"{fr['label']} ({fit_method})")

    if logy:
        pos = np.concatenate([s["y"][s["y"] > 0] for s in series])
        if pos.size:
            ax.set_yscale("log")
            ymax = max(np.nanmax(s["hi"]) for s in series)
            ax.set_ylim(pos.min() * 0.3, ymax * 3)
    else:
        ax.axhline(0, color=zero_color, lw=zero_lw)
        if ylims is not None:
            ax.set_ylim(ylims)
        else:
            # union y-limits over BOTH samples so neither is clipped
            lows, highs = [], []
            for s in series:
                for arr in (s["y"], s["lo"], s["hi"]):
                    a = arr[np.isfinite(arr)]
                    if a.size:
                        lows.append(a.min()); highs.append(a.max())
            if lows:
                lo, hi = min(lows), max(highs)
                pad = 0.1 * (hi - lo if hi > lo else abs(hi) or 1)
                ax.set_ylim(min(lo - pad, 0), hi + pad)

    unit = (boot_a.get("unit_info") or {}).get("y_unit", "")
    ylabel_final = (ylabel_text if ylabel_text is not None
                    else (f"Integrated Lyα flux [{unit}]" if unit else "Integrated Lyα flux"))
    ax.set_ylabel(ylabel_final, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.legend(frameon=False, fontsize=legend_fontsize, loc=legend_loc, title=legend_title)
    ax.grid(alpha=grid_alpha)
    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_flux_profile_two.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    if fit:
        return fig, ax, fit_results[0], fit_results[1]
    return fig, ax


def plot_asymmetry_profile_two(
    boot_a: dict,
    boot_b: dict,
    stacks_a: dict | None = None,
    stacks_b: dict | None = None,
    labels=("A", "B"),
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    VR_biweight_error: float | None = None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    show_vr: bool = True,
    xlims=None,
    jitter: float = 0.04,
    panels: str = "both",
    fmts=("o-", "s--"),
    markersize: float = 6,
    linewidth: float = 1.5,
    capsize: float = 3.5,
    sample_b_alpha: float = 0.65,
    frac_color: str = "tab:purple",
    symmetric_color: str = "0.5",
    blue_color: str = "royalblue",
    red_color: str = "tomato",
    zero_color: str = "0.7",
    figsize=None,
    title1: str | None = None,
    title2: str | None = None,
    legend_fontsize: float = 9,
    tick_fontsize: float = 10,
    label_fontsize: float = 11,
    title_fontsize: float = 13,
    grid_alpha: float = 0.15,
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Two-sample asymmetry overlay. Same two-panel layout as plot_asymmetry_profile
    (top: blue fraction B/(B+R); bottom: absolute blue & red flux), but both
    samples are drawn on each panel so the comparison is read on one figure.

    Self-contained (not a plotting.py engine wrapper) so every color, marker,
    and font size below is a keyword argument. Sample A is solid/filled
    markers, sample B is open/dashed (sample_b_alpha controls how faded B
    looks); blue and red keep their own colors. Per-panel y-limits use BOTH
    samples so neither is clipped.

    panels    : "both" (default, two stacked panels) | "fraction" (blue-fraction
                panel only) | "flux" (blue/red flux panel only) -- grab just
                one of the two images as its own standalone figure, same as
                the single-sample plot_asymmetry_profile.
    title1    : blue-fraction panel title; None (default) -> no title.
    title2    : blue/red-flux panel title; None (default) -> no title.
    figsize   : None -> (7.6, 8.0) for panels="both", (7.6, 4.5) for a single
                panel.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_asymmetry_profile_two.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks_a['VR_biweight_e'] if present.

    Example
    -------
        analysis.plot_asymmetry_profile_two(
            boot_lm, boot_hm, stacks_low_m, stacks_high_m,
            labels=("low-M", "high-M"),
            xlims=(3, 3000), bin_mode="kpc", figsize=(9, 8),
            panels="both",
        )

    Returns (fig, (ax1, ax2)) for panels="both", else (fig, ax).
    """
    for b in (boot_a, boot_b):
        if "blue_over_red_fid" not in b:
            raise KeyError("boot missing blue/red keys; re-run with "
                           "compute_side_ratio=True.")
    if panels not in ("both", "fraction", "flux"):
        raise ValueError("panels must be 'both', 'fraction', or 'flux'")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot_a.get("r_edges",
                             stacks_a["r_edges"] if stacks_a else []))
    bm   = _resolve_bin_mode(bin_mode, stacks_a or boot_a)
    vr   = _get_vr_biweight_v(VR_biweight_v, stacks_a or boot_a)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks_a or boot_a)

    if figsize is None:
        figsize = (7.6, 8.0) if panels == "both" else (7.6, 4.5)

    if panels == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    elif panels == "fraction":
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=figsize)
        ax1 = None

    K = 2

    # panel 1: blue fraction, both samples (per-draw fraction -> correct 16/84)
    if ax1 is not None:
        r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims,
                                         show_vr=show_vr, VR_biweight_error=vr_e)
        for k, (lab, b) in enumerate(((labels[0], boot_a), (labels[1], boot_b))):
            with np.errstate(divide="ignore", invalid="ignore"):
                bf_fid = np.asarray(b["blue_flux_fid"], float)
                rf_fid = np.asarray(b["red_flux_fid"], float)
                bfrac_fid = bf_fid / (bf_fid + rf_fid)
                if "blue_flux_all" in b and "red_flux_all" in b:
                    ball = np.asarray(b["blue_flux_all"], float)
                    rall = np.asarray(b["red_flux_all"], float)
                    bfrac_all = ball / (ball + rall)
                    bfrac_lo = np.nanpercentile(bfrac_all, 16, axis=0)
                    bfrac_hi = np.nanpercentile(bfrac_all, 84, axis=0)
                else:
                    warnings.warn(f"{lab}: blue_flux_all/red_flux_all not in boot; "
                                  "approximating B/total error bars from blue_flux_lo/hi.")
                    bl = np.asarray(b["blue_flux_lo"], float); bh = np.asarray(b["blue_flux_hi"], float)
                    rl = np.asarray(b["red_flux_lo"],  float); rh = np.asarray(b["red_flux_hi"],  float)
                    bfrac_lo = bl / (bl + rh)
                    bfrac_hi = bh / (bh + rl)
            jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
            yerr, unstable = _safe_yerr(bfrac_fid, bfrac_lo, bfrac_hi)
            ax1.errorbar(jit, bfrac_fid, yerr=yerr, fmt=fmts[k],
                        color=frac_color, capsize=capsize, ms=markersize, lw=linewidth,
                        alpha=1.0 if k == 0 else sample_b_alpha,
                        markerfacecolor=(frac_color if k == 0 else "none"),
                        label=f"{lab}  B/(B+R)")
            if np.any(unstable):
                ax1.scatter(jit[unstable], bfrac_fid[unstable], s=70, facecolors="none",
                            edgecolors=frac_color, linewidths=1.3, zorder=5)
        ax1.axhline(0.5, color=symmetric_color, lw=0.8, ls="--", label="symmetric (B/total = 0.5)")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Blue fraction  B / (B+R)", fontsize=label_fontsize)
        ax1.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        if title1:
            ax1.set_title(title1, fontsize=title_fontsize)
        ax1.legend(frameon=False, fontsize=legend_fontsize)
        ax1.grid(alpha=grid_alpha)

    # panel 2: absolute blue & red flux, both samples
    if ax2 is not None:
        r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims,
                                           show_vr=show_vr, VR_biweight_error=vr_e)
        dx = (r_mid2[1] - r_mid2[0]) * 0.05 if len(r_mid2) > 1 else 0
        for k, (lab, b) in enumerate(((labels[0], boot_a), (labels[1], boot_b))):
            jit = r_mid2 * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid2
            bf = np.asarray(b["blue_flux_fid"]); rf = np.asarray(b["red_flux_fid"])
            bf_lo = np.asarray(b["blue_flux_lo"]); bf_hi = np.asarray(b["blue_flux_hi"])
            rf_lo = np.asarray(b["red_flux_lo"]);  rf_hi = np.asarray(b["red_flux_hi"])
            mfc_b = blue_color if k == 0 else "none"
            mfc_r = red_color  if k == 0 else "none"
            bf_yerr, bf_unstable = _safe_yerr(bf, bf_lo, bf_hi)
            rf_yerr, rf_unstable = _safe_yerr(rf, rf_lo, rf_hi)
            ax2.errorbar(jit - dx, bf, yerr=bf_yerr, fmt=fmts[k],
                        capsize=capsize, ms=max(markersize - 1, 1), lw=max(linewidth - 0.2, 0.5),
                        color=blue_color, markerfacecolor=mfc_b, label=f"{lab} blue")
            ax2.errorbar(jit + dx, rf, yerr=rf_yerr, fmt=fmts[k],
                        capsize=capsize, ms=max(markersize - 1, 1), lw=max(linewidth - 0.2, 0.5),
                        color=red_color, markerfacecolor=mfc_r, label=f"{lab} red")
            if np.any(bf_unstable):
                ax2.scatter((jit - dx)[bf_unstable], bf[bf_unstable], s=60, facecolors="none",
                            edgecolors=blue_color, linewidths=1.3, zorder=5)
            if np.any(rf_unstable):
                ax2.scatter((jit + dx)[rf_unstable], rf[rf_unstable], s=60, facecolors="none",
                            edgecolors=red_color, linewidths=1.3, zorder=5)
        ax2.axhline(0, color=zero_color, lw=0.6)
        ax2.set_ylabel("Flux", fontsize=label_fontsize)
        ax2.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        if title2:
            ax2.set_title(title2, fontsize=title_fontsize)
        ax2.legend(frameon=False, fontsize=max(legend_fontsize - 1, 6), ncol=2)
        ax2.grid(alpha=grid_alpha)

    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_asymmetry_profile_two.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    if panels == "both":
        return fig, (ax1, ax2)
    return fig, (ax1 if ax1 is not None else ax2)


def run_all_plots_two(
    stacks_a: dict,
    stacks_b: dict,
    boot_a: dict,
    boot_b: dict,
    labels=("A", "B"),
    stack_method: str = "biweight",
    layout: str = "overlay",
    logy_flux: bool = True,
    save_figs: bool = False,
):
    """
    Two-stack companion to run_all_plots: every half-sample comparison figure in
    one call (panels + centroid + flux + asymmetry).

        analysis.run_all_plots_two(stacks_low_m, stacks_high_m, boot_lm, boot_hm,
                                    labels=("low-M", "high-M"))
    """
    from .plotting import plot_stack_panels_two

    print("─── stack panels (both samples) ───")
    plot_stack_panels_two(stacks_a, stacks_b, labels=labels,
                          method=stack_method, layout=layout, savepath=None)

    print("\n─── centroid profile (both samples) ───")
    plot_centroid_profile_two(boot_a, boot_b, stacks_a, stacks_b,
                              labels=labels, save_fig=save_figs)

    print("\n─── flux profile (both samples) ───")
    plot_flux_profile_two(boot_a, boot_b, stacks_a, stacks_b,
                          labels=labels, logy=logy_flux, save_fig=save_figs)

    print("\n─── asymmetry profile (both samples) ───")
    plot_asymmetry_profile_two(boot_a, boot_b, stacks_a, stacks_b,
                               labels=labels, save_fig=save_figs)


def plot_moments_profile(
    stacks: dict,
    stack_method: str = "biweight",
    bounds=None,
    cont_bounds=DEFAULT_CONT_BOUNDS,
    cont_method: str = "poly",
    cont_order: int = 1,
    lya_center: float = LYA_REST,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    figsize=(7.2, 8.5),
    save_fig: bool = False,
    savename: str = "Figure_moments_profile.png",
):
    """
    Three-panel radial profile of the Lyα line moments (FIDUCIAL STACK, no
    bootstrap) from top to bottom:

      1. Centroid velocity (km/s)  -- same quantity as the headline figure but
                                      from line_moments (identical for clip=True)
      2. Line width σ (km/s)       -- 2nd moment; proxy for intrinsic/kinematic
                                      broadening + redshift mixing in the stack
      3. Skewness                  -- 3rd standardised moment; > 0 = red tail
                                      (expected resonant Lya sign). This is the
                                      quantity that explains why clip vs. no-clip
                                      and Gaussian vs. flux-weighted disagree.

    These are POINT ESTIMATES. For error bars on σ and skewness, bootstrap them
    separately by calling bootstrap_measurements with method='moments' -- but
    note that is slower and the injection-recovery test is usually sufficient to
    characterise the systematics.
    """
    wave   = np.asarray(stacks["rest_wave"])
    fid    = np.asarray(stacks["stacks"][stack_method]["flux"])
    edges  = np.round(np.asarray(stacks["r_edges"]), 3)
    nrad   = fid.shape[0]
    bounds = bounds if bounds is not None else (lya_center - 4, lya_center + 4)
    bm     = _resolve_bin_mode(bin_mode, stacks)
    vr     = _get_vr_biweight_v(VR_biweight_v, stacks)

    keys = ("centroid_vel_kms", "sigma_kms", "skewness")
    mom  = {k: np.full(nrad, np.nan) for k in keys}
    for r in range(nrad):
        res = line_moments(wave, fid[r], bounds=bounds, cont_bounds=cont_bounds,
                           lya_center=lya_center, cont_method=cont_method,
                           cont_order=cont_order)
        if res["success"]:
            for k in keys:
                mom[k][r] = res[k]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    ylabels = [r"Centroid [km s$^{-1}$]", r"$\sigma$ [km s$^{-1}$]", "Skewness"]
    hlines  = [0.0, None, 0.0]
    colors  = ["tab:blue", "tab:green", "tab:orange"]

    for ax, key, ylabel, hline, col in zip(axes, keys, ylabels, hlines, colors):
        r_mid, xerr = _setup_radius_axis(ax, edges, bm, vr, None, vr_ticks, None)
        ax.plot(r_mid, mom[key], "o-", color=col, ms=6, lw=1.5)
        if hline is not None:
            ax.axhline(hline, color="0.65", lw=0.8, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)

    axes[2].set_title("Skewness > 0 = red tail  (resonant Lyα signature)",
                      fontsize=9, color="0.4")
    fig.suptitle("Lyα line moments vs. radius (fiducial stack)", fontsize=13, y=1.005)
    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes


# =====================================================================
# 5.  BOOTSTRAP HEALTH CHECKS
# =====================================================================

def plot_success_frac(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    threshold: float = 0.8,
    figsize=(7.2, 3.8),
    save_fig: bool = False,
    savename: str = "Figure_success_frac.png",
):
    """
    Bootstrap centroid success fraction per radial bin. A bar below the threshold
    (default 0.8) means the bootstrap is computing a centroid from a truncated
    distribution -- the 16/84 error bars in that bin underestimate the true
    uncertainty and should be treated with caution.

    Common causes of low success: inner bin has many NaN spectra (few real
    fibers), continuum window is too wide and overlaps with the line, or the
    clip_negative=False estimator is blowing up on noise-dominated outer bins.
    """
    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)
    sf  = np.asarray(boot["success_frac"])

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, _ = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, None)
    cols = ["tomato" if s < threshold else "tab:blue" for s in sf]
    ax.bar(r_mid, sf * 100, color=cols, edgecolor="0.3", alpha=0.85,
           width=np.minimum(np.diff(radial_bins) * 0.7,
                            r_mid * 0.4 if r_mid[0] > 0 else np.full(len(r_mid), 0.1)))
    ax.axhline(threshold * 100, color="tomato", lw=1.2, ls="--",
               label=f"threshold {threshold*100:.0f}%")
    for x, s in zip(r_mid, sf):
        ax.text(x, s * 100 + 1.5, f"{s*100:.0f}%",
                ha="center", va="bottom", fontsize=8, color="0.2")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Bootstrap success fraction (%)")
    ax.set_title("Centroid bootstrap success fraction\n"
                 "(red bars = trust the error bars cautiously)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()

    low = np.where(sf < threshold)[0]
    if len(low):
        edges = np.round(radial_bins, 3)
        for r in low:
            print(f"  WARNING bin {edges[r]:g}-{edges[r+1]:g}: "
                  f"success_frac={sf[r]:.2f} < {threshold}")
    return fig, ax


def plot_bootstrap_draws(
    boot: dict,
    radial_index: int = 0,
    stacks: dict | None = None,
    r_edges=None,
    n_bins: int = 40,
    figsize=(6.5, 4.2),
    save_fig: bool = False,
    savename: str = "Figure_bootstrap_draws.png",
):
    """
    Histogram of the centroid bootstrap DRAWS for one radial bin. The shape of
    this distribution tells you more than the 16/84 summary:

    • Roughly Gaussian → error bars are reliable; 16/84 ≈ 1-sigma.
    • Heavy tails or bimodal → bootstrap variance is dominated by rare outlier
      draws; consider jackknife or flagging the bin.
    • Very narrow peak with shoulder → a sub-population of draws has a different
      centroid (possibly two modes in the galaxy sample or a bad sky subtraction
      in a few spectra).

    Parameters
    ----------
    radial_index : which bin to show (-1 = outermost)
    """
    draws  = np.asarray(boot["centroid_v_all"])    # (nboot, nrad)
    r      = radial_index
    sample = draws[:, r]
    finite = sample[np.isfinite(sample)]

    edges  = np.round(np.asarray(r_edges if r_edges is not None
                      else boot.get("r_edges",
                      stacks["r_edges"] if stacks else [])), 3)
    nrad   = draws.shape[1]
    rr     = r if r >= 0 else nrad + r
    lbl    = (f"{edges[rr]:g}–{edges[rr+1]:g}" if len(edges) > rr + 1
              else f"bin {rr}")

    v_fid = boot["centroid_v_fid"][r]
    v_lo  = boot["centroid_v_lo"][r]
    v_hi  = boot["centroid_v_hi"][r]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(finite, bins=n_bins, color="steelblue", alpha=0.75,
            edgecolor="white", lw=0.4, density=True)
    ax.axvline(v_fid, color="k",       lw=1.6, ls="-",  label=f"fiducial {v_fid:+.1f}")
    ax.axvline(v_lo,  color="0.5",     lw=1.1, ls="--", label=f"16th {v_lo:+.1f}")
    ax.axvline(v_hi,  color="0.5",     lw=1.1, ls="--", label=f"84th {v_hi:+.1f}")
    ax.axvline(np.nanmedian(finite), color="tomato", lw=1.0, ls=":",
               label=f"median {np.nanmedian(finite):+.1f}")

    # overlay a Gaussian for comparison
    if len(finite) > 5:
        from scipy.stats import norm
        mu, sig = np.nanmedian(finite), 0.5 * (v_hi - v_lo)
        xg = np.linspace(finite.min(), finite.max(), 200)
        ax.plot(xg, norm.pdf(xg, mu, sig), color="tomato", lw=1.2, ls="-",
                alpha=0.6, label=f"Gaussian (μ={mu:+.0f}, σ={sig:.0f})")

    sf = boot["success_frac"][r]
    ax.set_xlabel(r"Centroid velocity [km s$^{-1}$]")
    ax.set_ylabel("Density")
    ax.set_title(f"Bootstrap draw distribution — bin {lbl}\n"
                 f"n_draws={len(finite)}/{len(sample)}  success_frac={sf:.2f}")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def measurement_confidence(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    flux_sn_min: float = 3.0,
    nsigma_min: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Per-bin "how much do we trust this point" summary, complementary to
    plot_success_frac. Derived entirely from an existing boot dict (no extra
    compute), so it is free to run next to the profiles.

    Why this exists: success_frac is a *did-the-estimator-run* flag, not a
    confidence measure. For the flux-weighted clipped estimator it is ≈1.0 by
    construction (a clipped weighted mean cannot leave the window and a real
    line always has positive net flux over >=min_pixels), so it reads 100% even
    when a point is faint or consistent with zero shift. The real confidence
    lives in the bootstrap interval width and the line detection S/N, which is
    what this function reports.

      • success_frac   -- convergence flag echoed for context (NOT confidence).
      • flux_sn        -- integrated-line detection S/N in the stack,
                          total_flux_fid / (bootstrap flux error). This is a
                          galaxy-resampling stability S/N ("does a line survive
                          dropping/duplicating galaxies"), NOT per-pixel photon
                          S/N. Needs compute_side_ratio=True (the default).
      • v_err_kms      -- centroid precision = half the 16-84 interval.
      • nsigma_offset  -- |v_fid| / (error toward 0): how many sigma the centroid
                          sits from systemic (0 km/s). This is the number you
                          keep wanting when success_frac reads 100%.

    Rigorous significance still belongs in validation.py (null / placebo /
    injection-recovery); nsigma_offset here is a gut-check, not a p-value.

        boot = analysis.run_measurement_suite(cfg, stacks)
        conf = analysis.measurement_confidence(boot, stacks)
        solid = conf["flux_sn"] >= 3        # mask faint outer bins, etc.

    Returns
    -------
    dict: {"flux_sn", "v_err_kms", "nsigma_offset", "success_frac", "r_edges"},
          each a (nrad,) array.
    """
    edges = np.round(np.asarray(r_edges if r_edges is not None
                                else boot.get("r_edges",
                                stacks["r_edges"] if stacks else [])), 3)
    v   = np.asarray(boot["centroid_v_fid"], float)
    lo  = np.asarray(boot["centroid_v_lo"], float)
    hi  = np.asarray(boot["centroid_v_hi"], float)
    sf  = np.asarray(boot["success_frac"], float)
    nrad = len(v)

    v_err = 0.5 * (hi - lo)
    # error on the side facing zero -> honest "distance from systemic"
    err_to_zero = np.where(v >= 0, v - lo, hi - v)
    with np.errstate(divide="ignore", invalid="ignore"):
        nsigma = np.abs(v) / err_to_zero

    if "total_flux_fid" in boot:
        f    = np.asarray(boot["total_flux_fid"], float)
        f_lo = np.asarray(boot["total_flux_lo"], float)
        f_hi = np.asarray(boot["total_flux_hi"], float)
        f_err = 0.5 * (f_hi - f_lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_sn = f / f_err
    else:
        flux_sn = np.full(nrad, np.nan)  # need compute_side_ratio=True

    if verbose:
        hdr = (f"{'bin':>10}  {'v_fid±err':>14}  {'nσ≠0':>6}  "
               f"{'flux S/N':>8}  {'ok%':>5}  notes")
        print(hdr)
        print("─" * len(hdr))
        for r in range(nrad):
            lbl = (f"{edges[r]:g}-{edges[r+1]:g}" if len(edges) > r + 1
                   else f"bin{r}")
            notes = []
            if np.isfinite(flux_sn[r]) and flux_sn[r] < flux_sn_min:
                notes.append("faint")
            if np.isfinite(nsigma[r]) and nsigma[r] < nsigma_min:
                notes.append("consistent w/ 0")
            if sf[r] < 1.0:
                notes.append(f"sf={sf[r]:.2f}")
            print(f"{lbl:>10}  {v[r]:>+7.1f}±{v_err[r]:>5.1f}  "
                  f"{nsigma[r]:>6.1f}  {flux_sn[r]:>8.1f}  {sf[r]*100:>4.0f}%  "
                  f"{', '.join(notes)}")
        print("\n  flux S/N = galaxy-bootstrap stability of integrated flux "
              "(not photon S/N).")
        print("  nσ≠0 is a gut-check only; use validation.py for real "
              "significance.")

    return {"flux_sn": flux_sn, "v_err_kms": v_err, "nsigma_offset": nsigma,
            "success_frac": sf, "r_edges": edges}


# =====================================================================
# 6.  CONVENIENCE RUNNER
# =====================================================================

def run_all_plots(
    stacks: dict,
    boot: dict,
    stack_method: str = "biweight",
    logy_flux: bool = True,
    save_figs: bool = False,
):
    """
    Run every analysis plot in a sensible order with one call. Good for the top
    of a notebook cell after run_measurement_suite to get a full picture before
    diving into specifics.

        boot = analysis.run_measurement_suite(cfg, stacks)
        analysis.run_all_plots(stacks, boot)
    """
    print("─── centroid table ───")
    print_centroid_table(boot)

    print("\n─── measurement confidence (S/N, σ-from-systemic) ───")
    measurement_confidence(boot, stacks)

    print("\n─── centroid method comparison (point estimates) ───")
    compare_centroid_methods(stacks, stack_method=stack_method)

    print("\n─── centroid profile ───")
    plot_centroid_profile(boot, stacks, save_fig=save_figs)

    print("\n─── flux profile ───")
    plot_flux_profile(boot, stacks, logy=logy_flux, save_fig=save_figs)

    print("\n─── asymmetry profile ───")
    plot_asymmetry_profile(boot, stacks, save_fig=save_figs)

    print("\n─── moments profile ───")
    plot_moments_profile(stacks, stack_method=stack_method, save_fig=save_figs)

    print("\n─── line panels (zoom to Lyα window) ───")
    plot_line_panels(stacks, boot, stack_method=stack_method, save_fig=save_figs)

    print("\n─── bootstrap success fraction ───")
    plot_success_frac(boot, stacks, save_fig=save_figs)


# ---------------------------------------------------------------------
# CORE-STACK COMPARISON  (AGN vs galaxy overlay, a diagnostic on
# stack.build_core_stacks results)
# ---------------------------------------------------------------------
def compare_core_stacks(stack_dict, method="inv_var", normalize="lya_peak",
                        agn_zoom=(1225, 1670)):
    """Overlay several core stacks (e.g. {'AGN': res_agn, 'galaxy': res_gal}).

    Each value is a stack.build_core_stacks result. The lower panel zooms on the
    AGN diagnostic region (N V, Si IV, C IV, He II) and is rescaled so the Lya
    peak doesn't flatten it.

    normalize : None | 'lya_peak' (each scaled to unit Lya peak) | 'max'.
    """
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))
    ax[1].set_xlim(*agn_zoom)
    zmax, ylab = 0.0, None

    for k, (label, res) in enumerate(stack_dict.items()):
        w = res["rest_wave"]; S = res["stacks"][method]
        f, e = S["flux"].astype(float), S["err"].astype(float)
        if normalize == "lya_peak":
            sel = (w > 1205) & (w < 1226); norm = np.nanmax(f[sel])
            ylab = "flux (Lya-peak normalized)"
        elif normalize == "max":
            norm = np.nanmax(f); ylab = "flux (peak normalized)"
        else:
            norm = 1.0; ylab = res["unit_info"]["y_unit"]
        norm = norm if (np.isfinite(norm) and norm) else 1.0
        f, e = f/norm, e/norm
        c = cyc[k % len(cyc)]
        for a in ax:
            a.plot(w, f, color=c, lw=1.0, label=f"{label} (N={res['ngal']})")
            a.fill_between(w, f-e, f+e, color=c, alpha=0.12, lw=0)
        inz = (w >= agn_zoom[0]) & (w <= agn_zoom[1])
        zmax = max(zmax, np.nanmax((f+e)[inz]))

    for a in ax:
        a.axhline(0, color="0.6", lw=0.7)
        _mark_lines(a)        # uses _REST_UV / _LINE_C, handles the 'kind' field
        a.set_xlabel("rest wavelength [A]"); a.set_ylabel(ylab)
    ax[1].set_ylim(-0.05*zmax, 1.15*zmax)   # rescale zoom so Lya peak doesn't flatten it
    ax[0].set_title(f"core stacks - {method}"); ax[0].legend(fontsize=9)
    ax[1].set_title("AGN diagnostic region (N V, Si IV, C IV, He II)")
    plt.tight_layout(); plt.show()
