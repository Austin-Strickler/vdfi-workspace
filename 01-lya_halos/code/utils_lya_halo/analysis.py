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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .measure import (
    measure_all_bins, bootstrap_measurements, bootstrap_stack_error, bootstrap_all,
    measure_centroid, line_moments, get_continuum_model,
    CENTROID_METHOD_SPECS, CENTROID_METHOD_LABELS, CENTROID_METHODS_DEFAULT,
    DEFAULT_CONT_BOUNDS, DEFAULT_CONT_METHOD, DEFAULT_CONT_ORDER,
    LYA_REST, C_KMS,
)
from .plotting import (
    _setup_radius_axis, _resolve_bin_mode, _get_vr_biweight_v, _safe_yerr,
    plot_centroid_vs_radius, plot_blue_red_vs_radius,
    plot_centroid_comparison, plot_radial_overlay,
    _mark_lines,
)


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
    panel_height: float = 2.6,
    figwidth: float = 8.0,
    show_cont: bool = True,
    show_window: bool = True,
    show_sidebands: bool = True,
    show_centroid: bool = True,
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
      • Line window shaded (yellow; optional)
      • Sideband regions shaded (light blue; optional)
      • Centroid marker: vertical dashed line at v_fid, with a short horizontal
        bar at the top spanning the 16/84 interval (optional; needs boot)

    This is the visual check that the centroid is landing on the line peak, that
    the continuum is flat in the sidebands, and that the window captures the
    emission without clipping the wings.

    Parameters
    ----------
    stacks       : Stage-2 dict (keep_cube=True for the boot error band)
    boot         : Stage-3 bootstrap result; None -> no error band / centroid marker
    stack_method : which stacks[method] to display
    bounds       : line window (lo, hi) Å; None -> (lya_center-4, lya_center+4)
    xlim         : wavelength range shown in each panel
    """
    wave   = np.asarray(stacks["rest_wave"])
    fid    = np.asarray(stacks["stacks"][stack_method]["flux"])   # (nrad, nwave)
    edges  = np.round(np.asarray(stacks["r_edges"]), 3)
    nrad   = fid.shape[0]
    bounds = bounds if bounds is not None else (lya_center - 4, lya_center + 4)
    bm     = stacks.get("bin_mode", "")

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

    fig, axes = plt.subplots(nrad, 1, figsize=(figwidth, panel_height * nrad),
                             sharex=True)
    if nrad == 1:
        axes = [axes]

    for r, ax in enumerate(axes):
        # --- bootstrap band (fid-centered 1-sigma) ---
        if band_lo is not None:
            ax.fill_between(wave, band_lo[r], band_hi[r], color="0.82", lw=0,
                            zorder=0, label="bootstrap 1$\\sigma$ (fid-centered)" if r == 0 else None)

        # --- sideband regions ---
        if show_sidebands:
            for (lo, hi) in cont_bounds:
                ax.axvspan(lo, hi, color="steelblue", alpha=0.08, lw=0)

        # --- line window ---
        if show_window:
            ax.axvspan(bounds[0], bounds[1], color="gold", alpha=0.13, lw=0)

        # --- fiducial stack ---
        ax.plot(wave, fid[r], color="k", lw=1.1, zorder=3,
                label="biweight stack" if r == 0 else None)

        # --- continuum ---
        if show_cont:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cont = get_continuum_model(wave, fid[r], cont_bounds=cont_bounds,
                                           cont_method=cont_method, cont_order=cont_order)
            ax.plot(wave, cont, color="tomato", lw=1.0, ls="--", zorder=2,
                    label="continuum" if r == 0 else None)

        # --- centroid marker ---
        if show_centroid and boot is not None:
            v_f  = boot["centroid_v_fid"][r]
            v_lo = boot["centroid_v_lo"][r]
            v_hi = boot["centroid_v_hi"][r]
            lam_c  = lya_center * (1 + v_f  / C_KMS)
            lam_lo = lya_center * (1 + v_lo / C_KMS)
            lam_hi = lya_center * (1 + v_hi / C_KMS)
            # vertical dashed line
            ax.axvline(lam_c, color="magenta", lw=1.1, ls="--", zorder=4,
                       label=f"centroid {v_f:+.0f} km/s" if r == 0 else None)
            # 16/84 bar at top of panel (0.92 in axes coords)
            y_bar = ax.get_ylim()[1] * 0.0   # will be set after ylim below; use transform
            ax.annotate("", xy=(lam_hi, 1.0), xytext=(lam_lo, 1.0),
                        xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="|-|", color="magenta",
                                        lw=1.1, mutation_scale=4))

        # axes / labels
        lbl = f"{edges[r]:g}–{edges[r+1]:g} {bm}"
        ax.text(0.02, 0.88, lbl, transform=ax.transAxes, fontsize=9.5,
                color="0.3",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                          boxstyle="round,pad=0.15"))
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(lya_center, color="0.7", lw=0.6, ls=":")
        ax.set_xlim(xlim)
        ax.grid(alpha=0.12)

        in_xlim = (wave >= xlim[0]) & (wave <= xlim[1])
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

    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=12)
    fig.supylabel("Flux", fontsize=11, x=0.01)
    fig.suptitle(f"Lyα line panels — {stack_method} stack", fontsize=13, y=1.005)
    plt.tight_layout()
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
    plot_literature: bool = False,
    figsize=(7.2, 4.7),
    title=r"Bootstrapped Ly$\alpha$ centroid vs. radius",
    save_fig: bool = False,
):
    """
    The headline figure: centroid velocity ± bootstrap 16/84 per radial bin.
    Thin wrapper around plotting.plot_centroid_vs_radius with sane defaults.

    Accepts the full `boot` dict from run_measurement_suite (which also carries
    r_edges, VR_biweight_v, etc.), so you just call:

        analysis.plot_centroid_profile(boot, stacks)
    """
    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    return plot_centroid_vs_radius(
        boot, radial_bins, VR_biweight_v=VR_biweight_v,
        stacks_result=stacks or boot,
        bin_mode=bin_mode, z_err_kms=z_err_kms,
        vr_ticks=vr_ticks, figsize=figsize, title=title,
        ylims=ylims, xlims=xlims,
        plot_literature=plot_literature,
        save_fig=save_fig,
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
    figsize=(7.2, 4.7),
    save_fig: bool = False,
    savename: str = "Figure_flux_profile.png",
):
    """
    Integrated Lyα flux (total_flux_fid) ± bootstrap 16/84 vs radius.
    Log-y (the default) is strongly recommended for seeing the faint outer bins;
    pass logy=False for a linear axis.

    Requires compute_side_ratio=True in the bootstrap (the default), which
    provides total_flux_fid = blue_flux_fid + red_flux_fid.
    """
    if "total_flux_fid" not in boot:
        raise KeyError("boot does not contain total_flux_fid; re-run with "
                       "compute_side_ratio=True (the default).")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)

    y    = np.asarray(boot["total_flux_fid"])
    y_lo = np.asarray(boot["total_flux_lo"])
    y_hi = np.asarray(boot["total_flux_hi"])

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims)
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
    ax.set_title("Integrated Lyα flux vs. radius")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_flux_curve_of_growth(
    cog: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    logy: bool = True,
    xlims=None,
    figsize=(7.2, 8.0),
    save_fig: bool = False,
    savename: str = "Figure_flux_curve_of_growth.png",
):
    """
    Two-panel Lyα curve of growth: (top) cumulative luminosity vs radius,
    center outward; (bottom) the same, normalized to a flux FRACTION of the
    total summed out to r_max. Both panels show the bootstrap 16-84 band.

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
    """
    for key in ("flux_cumulative_fid", "flux_fraction_fid"):
        if key not in cog:
            raise KeyError(f"cog missing {key!r}; pass the dict returned by "
                           "measure.flux_curve_of_growth.")

    radial_bins = np.asarray(r_edges if r_edges is not None else cog["r_edges_used"])
    bm  = _resolve_bin_mode(bin_mode, stacks)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # panel 1: cumulative luminosity
    r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims)
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
    ax1.set_title(f"Lyα curve of growth (r_max = {cog['meta']['r_max_kpc']:g} kpc)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.15)

    # panel 2: flux fraction
    r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims)
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
    ax2.set_title("Lyα flux fraction vs. radius")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2)


def plot_asymmetry_profile(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    xlims=None,
    figsize=(7.2, 8.0),
    save_fig: bool = False,
    savename: str = "Figure_asymmetry_profile.png",
):
    """
    Two-panel asymmetry diagnostic: (top) blue fraction B/(B+R) vs radius with
    the 16/84 bootstrap band; (bottom) absolute blue and red flux vs radius on
    the same axis, so you can see which side is driving the asymmetry.

    fraction = blue_flux / (blue_flux + red_flux), bounded to [0, 1]:
    = 0.5 → symmetric line.
    > 0.5 → more blue flux (blueshifted / absorption-driven?).
    < 0.5 → more red flux (the expected resonant scattering signature).

    Replaces the old blue/red ratio (which ran 0→∞ and put the symmetric point
    at 1). The fraction is computed PER bootstrap draw from blue_flux_all /
    red_flux_all so the 16/84 band is correct (you cannot divide percentiles).
    """
    if "blue_over_red_fid" not in boot:
        raise KeyError("boot missing blue/red keys; re-run with compute_side_ratio=True.")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # panel 1: blue fraction
    r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims)
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
    ax1.set_title("Lyα asymmetry: blue flux fraction")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.15)

    # panel 2: absolute fluxes
    r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims)
    bf    = np.asarray(boot["blue_flux_fid"])
    rf    = np.asarray(boot["red_flux_fid"])
    bf_lo = np.asarray(boot["blue_flux_lo"]);  bf_hi = np.asarray(boot["blue_flux_hi"])
    rf_lo = np.asarray(boot["red_flux_lo"]);   rf_hi = np.asarray(boot["red_flux_hi"])
    dx = (r_mid[1] - r_mid[0]) * 0.05 if len(r_mid) > 1 else 0
    bf_yerr, bf_unstable = _safe_yerr(bf, bf_lo, bf_hi)
    rf_yerr, rf_unstable = _safe_yerr(rf, rf_lo, rf_hi)
    ax2.errorbar(r_mid - dx, bf, xerr=xerr2, yerr=bf_yerr,
                 fmt="o", capsize=3, ms=5, lw=1.3, color="royalblue", label="blue flux")
    ax2.errorbar(r_mid + dx, rf, xerr=xerr2, yerr=rf_yerr,
                 fmt="s", capsize=3, ms=5, lw=1.3, color="tomato", label="red flux")
    if np.any(bf_unstable):
        ax2.scatter((r_mid - dx)[bf_unstable], bf[bf_unstable], s=60, facecolors="none",
                    edgecolors="royalblue", linewidths=1.3, zorder=5)
    if np.any(rf_unstable):
        ax2.scatter((r_mid + dx)[rf_unstable], rf[rf_unstable], s=60, facecolors="none",
                    edgecolors="tomato", linewidths=1.3, zorder=5)
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.set_ylabel("Flux")
    ax2.set_title("Blue and red side flux vs. radius")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2)


# =====================================================================
# 4b.  TWO-STACK (HALF-SAMPLE) OVERLAY PROFILES
# =====================================================================
# These put two boot/stacks pairs (e.g. low-mass vs high-mass) on ONE figure,
# instead of calling the single-stack version twice. Same native-unit radius
# axis as the single-stack figures, so they drop straight into the two-stack
# notebook.

def plot_centroid_profile_two(
    boot_a: dict,
    boot_b: dict,
    stacks_a: dict | None = None,
    stacks_b: dict | None = None,
    labels=("A", "B"),
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    ylims=(-250, 250),
    xlims=None,
    jitter: float = 0.04,
    figsize=(7.6, 4.9),
    title=r"Ly$\alpha$ centroid vs. radius (half-sample comparison)",
    save_fig: bool = False,
):
    """
    Overlay the bootstrapped centroid profiles of TWO samples on one axis.
    Replaces calling plot_centroid_profile twice. Thin wrapper around
    plotting.plot_centroid_comparison (the shared multi-sample engine).

        analysis.plot_centroid_profile_two(boot_lm, boot_hm,
                                            stacks_low_m, stacks_high_m,
                                            labels=("low-M", "high-M"))
    """
    boots = {labels[0]: boot_a, labels[1]: boot_b}
    if r_edges is None:
        r_edges = (boot_a.get("r_edges")
                   if boot_a.get("r_edges") is not None
                   else (stacks_a["r_edges"] if stacks_a else None))
    return plot_centroid_comparison(
        boots, radial_bins=r_edges, VR_biweight_v=VR_biweight_v,
        stacks_result=stacks_a or boot_a, bin_mode=bin_mode,
        vr_ticks=vr_ticks, figsize=figsize, ylims=ylims, xlims=xlims,
        jitter=jitter, title=title, save_fig=save_fig,
    )


def plot_flux_profile_two(
    boot_a: dict,
    boot_b: dict,
    stacks_a: dict | None = None,
    stacks_b: dict | None = None,
    labels=("A", "B"),
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    logy: bool = True,
    ylims=None,
    xlims=None,
    figsize=(7.6, 4.9),
    save_fig: bool = False,
    savename: str = "Figure_flux_profile_two.png",
):
    """
    Overlay integrated Lyα flux ± bootstrap 16/84 for TWO samples on one axis.
    Replaces calling plot_flux_profile twice. logy=True by default (faint outer
    bins). Built on plotting.plot_radial_overlay so the radius axis matches every
    other figure.

    For a LINEAR axis (logy=False) the y-limits are taken from BOTH samples
    (union of their finite ranges) so neither is clipped; on the log axis the
    scale already spans both.

        analysis.plot_flux_profile_two(boot_lm, boot_hm,
                                        stacks_low_m, stacks_high_m,
                                        labels=("low-M", "high-M"))
    """
    for b in (boot_a, boot_b):
        if "total_flux_fid" not in b:
            raise KeyError("boot missing total_flux_fid; re-run with "
                           "compute_side_ratio=True (the default).")
    if r_edges is None:
        r_edges = (boot_a.get("r_edges")
                   if boot_a.get("r_edges") is not None
                   else (stacks_a["r_edges"] if stacks_a else None))

    series = []
    for lab, b in ((labels[0], boot_a), (labels[1], boot_b)):
        series.append(dict(
            label=lab,
            y=np.asarray(b["total_flux_fid"]),
            lo=np.asarray(b["total_flux_lo"]),
            hi=np.asarray(b["total_flux_hi"]),
        ))

    # union y-limits for the LINEAR case (log auto-ranges over both already)
    if ylims is None and not logy:
        lows, highs = [], []
        for s in series:
            for arr in (s["y"], s["lo"], s["hi"]):
                a = np.asarray(arr, float)
                a = a[np.isfinite(a)]
                if a.size:
                    lows.append(a.min()); highs.append(a.max())
        if lows:
            lo, hi = min(lows), max(highs)
            pad = 0.1 * (hi - lo if hi > lo else abs(hi) or 1)
            ylims = (min(lo - pad, 0), hi + pad)

    unit = (boot_a.get("unit_info") or {}).get("y_unit", "")
    ylabel = f"Integrated Lyα flux [{unit}]" if unit else "Integrated Lyα flux"
    return plot_radial_overlay(
        series, radial_bins=r_edges, bin_mode=bin_mode,
        VR_biweight_v=VR_biweight_v, stacks_result=stacks_a or boot_a,
        vr_ticks=vr_ticks, figsize=figsize, ylims=ylims, xlims=xlims,
        ylabel=ylabel, title="Integrated Lyα flux vs. radius (half-sample comparison)",
        zero_line=None if logy else 0.0, legend_title="sample",
        logy=logy, save_fig=save_fig, savename=savename,
    )


def plot_asymmetry_profile_two(
    boot_a: dict,
    boot_b: dict,
    stacks_a: dict | None = None,
    stacks_b: dict | None = None,
    labels=("A", "B"),
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    xlims=None,
    jitter: float = 0.04,
    figsize=(7.6, 8.0),
    save_fig: bool = False,
    savename: str = "Figure_asymmetry_profile_two.png",
):
    """
    Two-sample asymmetry overlay. Same two-panel layout as plot_asymmetry_profile
    (top: blue/red ratio; bottom: absolute blue & red flux), but both samples are
    drawn on each panel so the comparison is read on one figure.

    Sample A is solid markers, sample B is open/dashed; blue and red keep their
    colours. Per-panel y-limits use BOTH samples so neither is clipped.

        analysis.plot_asymmetry_profile_two(boot_lm, boot_hm,
                                             stacks_low_m, stacks_high_m,
                                             labels=("low-M", "high-M"))
    """
    for b in (boot_a, boot_b):
        if "blue_over_red_fid" not in b:
            raise KeyError("boot missing blue/red keys; re-run with "
                           "compute_side_ratio=True.")
    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot_a.get("r_edges",
                             stacks_a["r_edges"] if stacks_a else []))
    bm = _resolve_bin_mode(bin_mode, stacks_a or boot_a)
    vr = _get_vr_biweight_v(VR_biweight_v, stacks_a or boot_a)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # panel 1: blue/red ratio, both samples
    r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims)
    K = 2
    fmts = ("o-", "s--")
    for k, (lab, b) in enumerate(((labels[0], boot_a), (labels[1], boot_b))):
        bor    = np.asarray(b["blue_over_red_fid"])
        bor_lo = np.asarray(b["blue_over_red_lo"])
        bor_hi = np.asarray(b["blue_over_red_hi"])
        jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
        bor_yerr, bor_unstable = _safe_yerr(bor, bor_lo, bor_hi)
        ax1.errorbar(jit, bor, yerr=bor_yerr,
                     fmt=fmts[k], capsize=3.5, ms=6, lw=1.5,
                     color="tab:purple", alpha=1.0 if k == 0 else 0.65,
                     markerfacecolor=("tab:purple" if k == 0 else "none"),
                     label=f"{lab}  (B/R)")
        if np.any(bor_unstable):
            ax1.scatter(jit[bor_unstable], bor[bor_unstable], s=70, facecolors="none",
                        edgecolors="tab:purple", linewidths=1.3, zorder=5)
    ax1.axhline(1.0, color="0.5", lw=0.8, ls="--", label="symmetric (B/R = 1)")
    ax1.set_ylabel("Blue / Red flux ratio")
    ax1.set_title("Lyα asymmetry: blue/red side ratio (half-sample comparison)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.15)

    # panel 2: absolute blue & red flux, both samples
    r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims)
    dx = (r_mid2[1] - r_mid2[0]) * 0.05 if len(r_mid2) > 1 else 0
    for k, (lab, b) in enumerate(((labels[0], boot_a), (labels[1], boot_b))):
        jit = r_mid2 * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid2
        bf = np.asarray(b["blue_flux_fid"]); rf = np.asarray(b["red_flux_fid"])
        bf_lo = np.asarray(b["blue_flux_lo"]); bf_hi = np.asarray(b["blue_flux_hi"])
        rf_lo = np.asarray(b["red_flux_lo"]);  rf_hi = np.asarray(b["red_flux_hi"])
        mfc_b = "royalblue" if k == 0 else "none"
        mfc_r = "tomato"    if k == 0 else "none"
        bf_yerr, bf_unstable = _safe_yerr(bf, bf_lo, bf_hi)
        rf_yerr, rf_unstable = _safe_yerr(rf, rf_lo, rf_hi)
        ax2.errorbar(jit - dx, bf, yerr=bf_yerr,
                     fmt=fmts[k], capsize=3, ms=5, lw=1.3, color="royalblue",
                     markerfacecolor=mfc_b, label=f"{lab} blue")
        ax2.errorbar(jit + dx, rf, yerr=rf_yerr,
                     fmt=fmts[k], capsize=3, ms=5, lw=1.3, color="tomato",
                     markerfacecolor=mfc_r, label=f"{lab} red")
        if np.any(bf_unstable):
            ax2.scatter((jit - dx)[bf_unstable], bf[bf_unstable], s=60, facecolors="none",
                        edgecolors="royalblue", linewidths=1.3, zorder=5)
        if np.any(rf_unstable):
            ax2.scatter((jit + dx)[rf_unstable], rf[rf_unstable], s=60, facecolors="none",
                        edgecolors="tomato", linewidths=1.3, zorder=5)
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.set_ylabel("Flux")
    ax2.set_title("Blue and red side flux vs. radius")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2)


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
