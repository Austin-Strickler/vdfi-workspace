"""
validation.py -- Tier-1 validation: null tests, robustness sweeps, and diagnostics.

Everything here runs on a Stage-1 product (the galaxy FITS) or a Stage-2 stacks
dict, so it costs NOTHING from the 13-19 hr extraction -- it re-stacks and/or
re-measures the cheap derived products to ask "is the signal real, and how much
does it depend on my choices?"

Null tests (is the line real?):
    redshift_scramble_null / plot_scramble_null   -- permute z, re-stack; lines wash out
    placebo_wavelengths / plot_placebo            -- measure off-line "placebo" windows
    zswap_spectra, collect_null_ensembles,
        plot_null_envelope / plot_null_synthesis  -- null-ensemble intuition + verdict

Robustness sweeps (how choice-dependent is the trend?):
    sweep_line_window      / plot_window_sweep    -- centroid vs +/-N A window
    sweep_continuum_method / plot_continuum_sweep -- centroid vs continuum model
    windowed_flux_scan     / plot_windowed_scan

Error / recovery checks:
    error_cross_check      / plot_error_cross_check  -- propagated vs bootstrap errors
    plot_error_calibration
    injection_recovery / injection_bias_fast         -- inject a known line, recover it
        plot_injection_recovery / plot_injection_bias / plot_bias_summary

Spectral diagnostics + summaries:
    signal_to_noise_spectrum / plot_sn_spectrum / plot_sn_map
    plot_continuum_check
    summarize_validation / print_validation_summary / centroid_dossier

(See __all__ for the full public surface.) Centroid-estimator labels are shared
with analysis.py via measure.CENTROID_METHOD_SPECS/LABELS, so a label means the
same thing whether you measure it or stress-test it.

Deliberately NOT imported in utils_lya_halo/__init__.py -- it pulls in matplotlib,
which the package otherwise keeps out of import. Use:

    from utils_lya_halo import validation
    # or
    from utils_lya_halo.validation import redshift_scramble_null, sweep_line_window
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from .io import GalaxyProduct
from .stack import build_stacks, coadd_galaxies
from .measure import (
    flux_weighted_centroid, measure_centroid, bootstrap_measurements,
    bootstrap_stack_error, bootstrap_all, stack_galaxies, get_continuum_model,
    integrated_line_flux, CENTROID_METHOD_SPECS, CENTROID_METHOD_LABELS,
    DEFAULT_CONT_BOUNDS, LYA_REST, run_header,
)
from .config import stamp_provenance
# reuse the EXACT radius axis + figure engines the headline figures use
from .plotting import (
    _setup_radius_axis, _resolve_bin_mode, _get_vr_biweight_v, _radius_points,
    plot_radial_overlay, plot_null_band,
)

if TYPE_CHECKING:
    from .config import PipelineConfig

C_KMS = 299792.458

__all__ = [
    "redshift_scramble_null",
    "plot_scramble_null",
    "sweep_line_window",
    "plot_window_sweep",
    "sweep_continuum_method",
    "plot_continuum_sweep",
    "plot_continuum_check",
    "error_cross_check",
    "plot_error_cross_check",
    "DEFAULT_UV_LINES",
    "placebo_wavelengths",
    "plot_placebo",
    "injection_recovery",
    "injection_bias_fast",
    "plot_injection_recovery",
    "plot_injection_bias",
    "plot_bias_summary",
    "signal_to_noise_spectrum",
    "plot_sn_spectrum",
    "plot_sn_map",
    "summarize_validation",
    "print_validation_summary",
    "centroid_dossier",
    # --- null-ensemble synthesis (intuition + verdict) ---
    "zswap_spectra",
    "plot_null_envelope",
    "collect_null_ensembles",
    "plot_null_synthesis",
    "plot_error_calibration",
    "windowed_flux_scan",
    "plot_windowed_scan",
]


# Rest-frame vacuum UV lines in the stack window (name, lambda_A, kind).
# 'em' = expected in emission, 'abs' = expected in absorption. Doublets are
# listed as separate components. Used to mark, mask (for continuum), and to
# integrate line strength vs radius.
DEFAULT_UV_LINES = [
    ("Lyβ",       1025.72, "em"),
    ("Lyα",       1215.67, "em"),
    ("N V",       1238.82, "abs"),
    ("N V",       1242.80, "abs"),
    ("Si II",     1260.42, "abs"),
    ("Si II*",    1264.74, "em"),
    ("O I+Si II", 1303.27, "abs"),   # 1302.17 / 1304.37 blend
    ("C II",      1334.53, "abs"),
    ("Si IV",     1393.76, "abs"),
    ("Si IV",     1402.77, "abs"),
    ("Si II",     1526.71, "abs"),
    ("C IV",      1548.20, "abs"),
    ("C IV",      1550.77, "abs"),
    ("Fe II",     1608.45, "abs"),
    ("He II",     1640.42, "em"),
    ("Al II",     1670.79, "abs"),
]


def _line_window_mask(wave, lines, hw):
    """Boolean over wave: True inside ±hw of ANY catalog line."""
    w = np.asarray(wave, dtype=float)
    mask = np.zeros_like(w, dtype=bool)
    for _, lam, _ in lines:
        mask |= np.abs(w - lam) <= hw
    return mask


def _robust_poly(wave, flux, fit_mask, order=3, sigma_clip=3.0, niter=3):
    """Sigma-clipped polynomial continuum over the full wave grid, fit only to
    the pixels in fit_mask (lines pre-excluded by the caller). Wavelength is
    centered before the fit for numerical stability."""
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = fit_mask & np.isfinite(w) & np.isfinite(f)
    if m.sum() < order + 2:
        fin = np.isfinite(f)
        return np.full_like(w, np.nanmedian(f[fin]) if np.any(fin) else 0.0)
    x0 = np.nanmedian(w[m])
    xx = w[m] - x0
    y = f[m]
    good = np.ones(m.sum(), dtype=bool)
    for _ in range(niter):
        c = np.polyfit(xx[good], y[good], order)
        resid = y - np.polyval(c, xx)
        s = 1.4826 * np.nanmedian(np.abs(resid[good] - np.nanmedian(resid[good])))
        if not np.isfinite(s) or s == 0:
            break
        good = np.abs(resid) < sigma_clip * s
    c = np.polyfit(xx[good], y[good], order)
    return np.polyval(c, w - x0)


def _robust_std(x):
    """0.5*(p84-p16), NaN-safe; falls back to nanstd if degenerate."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nanstd(x) if x.size else np.nan
    lo, hi = np.nanpercentile(x, [16, 84])
    s = 0.5 * (hi - lo)
    return s if np.isfinite(s) and s > 0 else np.nanstd(x)


# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------
def _continuum_kwargs(config):
    """Read sideband/continuum settings from config if present, else defaults
    (mirrors measure.measure_all_bins, so this tracks whatever the real run uses)."""
    return dict(
        cont_bounds=getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS),
        cont_method=getattr(config, "cont_method", "poly"),
        cont_order=getattr(config, "cont_order", 1),
    )


def _integrated_flux_per_bin(stack_flux, rest_wave, bounds, cont_kw):
    """Continuum-subtracted flux sum in the line window, per radial bin,
    via the canonical measure.integrated_line_flux. Returns flux_SUM (the raw
    pixel sum, no dwave factor) so it is in the same units as the bootstrap's
    total_flux_* = blue_flux + red_flux, which are also raw sums. The null-band
    SNR is scale-invariant so dwave cancels; using flux_sum keeps real_F and
    total_flux_fid directly comparable for the compare_boot overlay."""
    nrad = stack_flux.shape[0]
    F = np.full(nrad, np.nan)
    for r in range(nrad):
        res = integrated_line_flux(rest_wave, stack_flux[r], bounds=bounds, **cont_kw)
        F[r] = res["flux_sum"]
    return F


def _scramble_product(product: GalaxyProduct, z_col: str, rng) -> GalaxyProduct:
    """Copy the product with the catalog z column permuted across galaxies.

    Only the small catalog table is copied; the big spec/err arrays are shared
    (they are never mutated -- build_stacks reads the scrambled z from the
    catalog for BOTH the unit conversion and the rest-frame axis division)."""
    cat = product.catalog.copy()
    z = np.asarray(cat[z_col], dtype=float).copy()
    cat[z_col] = z[rng.permutation(len(z))]
    return GalaxyProduct(
        spec=product.spec, err=product.err, nfib=product.nfib,
        wave=product.wave, r_edges=product.r_edges, catalog=cat,
        header=product.header, background=product.background,
    )


# =====================================================================
# 1. REDSHIFT-SCRAMBLE NULL
# =====================================================================
def redshift_scramble_null(
    config: "PipelineConfig",
    product: GalaxyProduct,
    n_realizations: int = 30,
    method: str | None = None,
    bounds=None,
    real_boot: dict | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Permute z across galaxies, re-stack, and measure the INTEGRATED line flux per
    radial bin; repeat n_realizations times. The scatter across realizations is
    the null distribution of integrated flux for a sample with NO coherent line
    (the redshift mis-mapping scatters each galaxy's real line to a random rest
    wavelength). The real signal's integrated flux is compared against it, and
    the null-calibrated significance is reported as a z-score / S/N.

    Why integrated flux, not the centroid: the flux-weighted centroid is a RATIO
    (Σx·max(y,0)/Σmax(y,0)); under the null a single positive noise pixel near a
    window edge yields a ±large velocity, so the centroid null band is enormous
    and the test is uninformative. Integrated flux is a SUM, stays finite and
    ~Gaussian under the null, so the band is well-defined and the comparison is
    meaningful.

    Relationship to signal_to_noise_spectrum: that function gives the per-pixel
    S/N of the REAL stack using the galaxy bootstrap error; it trusts those
    errors. This test builds an EMPIRICAL null by destroying the signal while
    keeping the fibers/sky/continuum/coadd, so it also catches coherent additive
    structure the bootstrap error would not flag. Use both; they report the same
    statistic (integrated flux and its S/N) so they are directly comparable.

    Parameters
    ----------
    product       : the SAME Stage-1 GalaxyProduct you feed Stage 2.
    n_realizations: number of z-permutations; each is a build_stacks pass.
    method        : galaxy-combine to read from stacks['stacks'] (default 'biweight').
    bounds        : line window (lo, hi) in rest A. None -> config.line_window.
    real_boot     : optional run_measure output; if it carries side fluxes
                    (blue_flux + red_flux) the real integrated flux is also drawn
                    with its bootstrap 16/84 bars for context.

    Returns
    -------
    dict with:
        real_F        (nrad,)               real integrated line flux
        null_F_all    (n_realizations,nrad) per-realization null integrated flux
        null_F_med/lo/hi (nrad,)            null median, 16th, 84th pct
        null_F_std    (nrad,)               null std (the null noise)
        snr           (nrad,)               (real_F - null_med)/null_std  [signed]
        z_score       (nrad,)               |snr|
        success_frac  (nrad,)               fraction of realizations with finite flux
        r_edges, bin_mode, VR_biweight_v, unit_info, real_boot, bounds, method
    """
    bounds = tuple(config.line_window) if bounds is None else tuple(bounds)
    method = method or "biweight"
    cont_kw = _continuum_kwargs(config)

    # --- real (unscrambled) integrated flux ---
    real = build_stacks(config, product, keep_cube=False)
    rest_wave = np.asarray(real["rest_wave"])
    if method not in real["stacks"]:
        raise KeyError(
            f"method {method!r} not in stacks (have {list(real['stacks'])}). "
            f"Add it to config.galaxy_combine_methods or pass an available one."
        )
    real_F = _integrated_flux_per_bin(real["stacks"][method]["flux"], rest_wave, bounds, cont_kw)
    nrad = real_F.size

    # --- scramble loop ---
    rng = np.random.default_rng(seed)
    null_F = np.full((n_realizations, nrad), np.nan)
    it = range(n_realizations)
    if verbose:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="z-scramble")
        except Exception:
            pass
    for k in it:
        sp = _scramble_product(product, config.z_col, rng)
        st = build_stacks(config, sp, keep_cube=False)
        null_F[k] = _integrated_flux_per_bin(
            st["stacks"][method]["flux"], np.asarray(st["rest_wave"]), bounds, cont_kw)

    with warnings.catch_warnings():           # all-NaN bins -> quiet NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        null_med = np.nanmedian(null_F, axis=0)
        null_lo = np.nanpercentile(null_F, 16, axis=0)
        null_hi = np.nanpercentile(null_F, 84, axis=0)
        null_std = np.nanstd(null_F, axis=0)
    snr = (real_F - null_med) / np.where(null_std > 0, null_std, np.nan)
    z_score = np.abs(snr)
    success = np.mean(np.isfinite(null_F), axis=0)

    if verbose:
        edges = np.round(np.asarray(real["r_edges"]), 3)
        print(f"\nz-scramble null ({n_realizations} realizations, method={method}, "
              f"window=±{0.5*(bounds[1]-bounds[0]):.1f} Å):")
        print(f"{'bin':>14}  {'real F':>10}  {'null med':>10}  {'null σ':>10}  {'S/N':>6}")
        for r in range(nrad):
            lbl = f"{edges[r]:g}-{edges[r+1]:g}"
            print(f"{lbl:>14}  {real_F[r]:10.3g}  {null_med[r]:10.3g}  "
                  f"{null_std[r]:10.3g}  {snr[r]:6.1f}")

    return stamp_provenance({
        "real_F": real_F,
        "null_F_all": null_F,
        "null_F_med": null_med,
        "null_F_lo": null_lo,
        "null_F_hi": null_hi,
        "null_F_std": null_std,
        "snr": snr,
        "z_score": z_score,
        "success_frac": success,
        "r_edges": np.asarray(real["r_edges"]),
        "bin_mode": real.get("bin_mode"),
        "VR_biweight_v": real.get("VR_biweight_v"),
        "unit_info": real.get("unit_info"),
        "real_boot": real_boot,
        "bounds": bounds,
        "method": method,
        "n_realizations": n_realizations,
    }, config, "redshift_scramble_null")


def _boot_total_flux(real_boot):
    """Pull (med, lo, hi) integrated flux from a run_measure summary, preferring
    the canonical total_flux_* keys and falling back to blue+red. Returns
    (None, None, None) if the bootstrap flux is unavailable."""
    if not real_boot:
        return None, None, None
    if "total_flux_med" in real_boot:
        return (np.asarray(real_boot["total_flux_med"], float),
                np.asarray(real_boot["total_flux_lo"], float),
                np.asarray(real_boot["total_flux_hi"], float))
    if "blue_flux_med" in real_boot and "red_flux_med" in real_boot:
        med = np.asarray(real_boot["blue_flux_med"], float) + \
              np.asarray(real_boot["red_flux_med"], float)
        return med, None, None       # no combined 16/84 available -> point only
    return None, None, None


def plot_scramble_null(
    result: dict,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    figsize=(7.4, 4.8),
    ylims=None,
    xlims=None,
    logy=False,
    compare_boot=False,
    title=None,
    save_fig=False,
):
    """
    Real integrated line flux vs radius over the z-scramble null band, via the
    shared plot_null_band engine. Real points above the grey 16/84 envelope =
    the line is not a coadd artifact. Each point is annotated with its
    null-calibrated S/N (real - null_med)/null_std.

    logy : log y-axis. Essential for seeing the faint outer bins, whose flux is
           orders of magnitude below the inner bins. The shared engine drops the
           y=0 reference line automatically when logy=True; note that on a log
           axis any negative null/real values are clipped by matplotlib, which is
           fine here because the integrated flux is positive where there is
           signal (negative excursions are pure-noise bins anyway).
    compare_boot : also overlay the GALAXY-BOOTSTRAP 16/84 band on the real flux,
           from result['real_boot']. This contrasts the two error estimates: the
           z-scramble null is the COADD-artifact floor (how big a fake line the
           stacking could make), while the bootstrap band is the SAMPLING error
           on the real line. Where the bootstrap band sits well above the null
           band, the detection is both real and well-measured.
    """
    y_unit = (result.get("unit_info") or {}).get("y_unit", "")
    real_lo = real_hi = None
    if compare_boot:
        _, real_lo, real_hi = _boot_total_flux(result.get("real_boot"))
    return plot_null_band(
        radial_bins=np.asarray(result["r_edges"]),
        null_lo=result["null_F_lo"], null_hi=result["null_F_hi"],
        null_med=result["null_F_med"], real_y=result["real_F"],
        real_lo=real_lo, real_hi=real_hi,
        bin_mode=result.get("bin_mode"), VR_biweight_v=result.get("VR_biweight_v"),
        vr_ticks=vr_ticks, figsize=figsize, ylims=ylims, xlims=xlims, logy=logy,
        ylabel=(f"Integrated line flux [{y_unit}]" if y_unit else "Integrated line flux"),
        title=title or "Integrated flux vs. z-scramble null",
        null_label="z-scramble null (16–84%)",
        real_label=("real flux (bootstrap 16–84)" if compare_boot
                    else "real integrated flux"),
        snr=result.get("snr"), save_fig=save_fig, savename="Figure_scramble_null.png",
    )


# =====================================================================
# 2. LINE-WINDOW ROBUSTNESS SWEEP
# =====================================================================
def sweep_line_window(
    config: "PipelineConfig",
    stacks: dict,
    half_windows=(3, 4, 5, 6),
    nboot: int = 300,
    method: str | None = None,
    seed: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Re-measure the centroid AND integrated flux for several symmetric line
    windows (±half_window Å about Lya), each with its own galaxy bootstrap, on
    the SAME Stage-2 cube. compute_side_ratio is ON so the integrated (windowed)
    flux is available too; pick the statistic in plot_window_sweep.

    Needs stacks built with keep_cube=True (run_stack default).

    Returns {half_window: bootstrap_measurements summary, ..., '_meta': {...}}.
    Feed the result straight to plot_window_sweep.
    """
    if "cube_flux" not in stacks or "cube_err" not in stacks:
        raise KeyError("sweep_line_window needs the per-galaxy cube: "
                       "run_stack(config, ..., keep_cube=True).")
    flux = np.asarray(stacks["cube_flux"], dtype=float)
    err = np.asarray(stacks["cube_err"], dtype=float)
    wave = np.asarray(stacks["rest_wave"], dtype=float)
    lya = float(config.LYA_REST)
    cont_kw = _continuum_kwargs(config)
    sm = method or getattr(config, "measure_stack_method", "biweight")

    run_header("sweep_line_window", verbose=verbose, nboot=nboot, stack=sm,
               half_windows=tuple(half_windows), seed=seed)
    out: dict = {}
    for hw in half_windows:
        bounds = (lya - hw, lya + hw)
        out[hw] = bootstrap_all(
            flux, err, wave, nboot=nboot, bounds=bounds, lya_center=lya,
            stack_method=sm, compute_side_ratio=True, compute_stack_error=False,
            seed=seed, label=f"window ±{hw:g} Å", announce=False,
            verbose=verbose, **cont_kw,
        )
    out["_meta"] = {
        "half_windows": list(half_windows),
        "nboot": nboot,
        "stack_method": sm,
        "r_edges": np.asarray(stacks["r_edges"]),
        "bin_mode": stacks.get("bin_mode"),
        "VR_biweight_v": stacks.get("VR_biweight_v"),
    }
    return stamp_provenance(out, config, "sweep_line_window")


_STAT_KEYS = {
    "centroid": ("centroid_v_med", "centroid_v_lo", "centroid_v_hi",
                 r"Ly$\alpha$ centroid velocity [km s$^{-1}$]", (-200, 200)),
    "flux": ("total_flux_med", "total_flux_lo", "total_flux_hi",
             "Integrated line flux", None),
}


def _sweep_series(sweep, entry_keys, labels, statistic):
    """Build plot_radial_overlay series for the chosen statistic from a sweep."""
    med, lo, hi, _, _ = _STAT_KEYS[statistic]
    series = []
    for key, lab in zip(entry_keys, labels):
        s = sweep[key]
        if med not in s:
            raise KeyError(f"statistic {statistic!r} needs compute_side_ratio "
                           f"(missing {med!r}); re-run the sweep.")
        series.append({"label": lab, "y": s[med], "lo": s[lo], "hi": s[hi]})
    return series


def plot_window_sweep(
    sweep: dict, statistic="centroid",
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.6, 4.9),
    ylims="auto", xlims=None, jitter=0.05, title=None, save_fig=False,
):
    """
    Overlay the chosen statistic vs radius for each line window, via the shared
    plot_radial_overlay engine. statistic='centroid' (default) or 'flux'
    (integrated windowed flux). Each curve carries its bootstrap 16/84 bars.
    """
    meta = sweep["_meta"]
    hws = meta["half_windows"]
    series = _sweep_series(sweep, hws, [fr"$\pm{hw}$ Å" for hw in hws], statistic)
    _, _, _, ylabel, default_ylims = _STAT_KEYS[statistic]
    return plot_radial_overlay(
        series, radial_bins=np.asarray(meta["r_edges"]),
        bin_mode=meta.get("bin_mode"), VR_biweight_v=meta.get("VR_biweight_v"),
        vr_ticks=vr_ticks, figsize=figsize,
        ylims=(default_ylims if ylims == "auto" else ylims), xlims=xlims,
        ylabel=ylabel, jitter=jitter, legend_title="line window",
        title=title or f"{statistic} vs. radius — line-window sweep",
        save_fig=save_fig, savename="Figure_window_sweep.png",
    )


# =====================================================================
# 2b. CONTINUUM-METHOD ROBUSTNESS SWEEP
# =====================================================================
def _default_continuum_combos(config):
    """A sensible default set of continuum settings to sweep. Each is
    (label, cont_method, cont_order, cont_bounds)."""
    nb = DEFAULT_CONT_BOUNDS                      # the new symmetric ~10 A sidebands
    wide = ((nb[0][0] - 5, nb[0][1]), (nb[1][0], nb[1][1] + 5))
    narrow = ((nb[0][0] + 3, nb[0][1]), (nb[1][0], nb[1][1] - 3))
    return [
        ("poly1 (default)", "poly", 1, nb),
        ("poly2", "poly", 2, nb),
        ("median", "median", 1, nb),
        ("poly1 wide bands", "poly", 1, wide),
        ("poly1 narrow bands", "poly", 1, narrow),
    ]


def sweep_continuum_method(
    config: "PipelineConfig", stacks: dict, combos=None, nboot: int = 300,
    method: str | None = None, seed: int = 1, verbose: bool = True,
) -> dict:
    """
    Re-measure centroid AND integrated flux under several CONTINUUM settings
    (method / order / sideband windows), each with its own bootstrap, on the
    SAME Stage-2 cube. The continuum is the other big lever on the centroid (the
    line window is the first; see sweep_line_window), so this shows how the
    radial trend moves as the continuum model varies -- and lets you see how the
    new default sidebands compare to wider/narrower/median choices.

    combos : list of (label, cont_method, cont_order, cont_bounds). None -> a
             default set built from the shared DEFAULT_CONT_BOUNDS.

    Returns {label: bootstrap_measurements summary, ..., '_meta': {...}}; feed to
    plot_continuum_sweep (statistic='centroid' or 'flux').
    """
    if "cube_flux" not in stacks or "cube_err" not in stacks:
        raise KeyError("sweep_continuum_method needs the per-galaxy cube "
                       "(keep_cube=True).")
    flux = np.asarray(stacks["cube_flux"], dtype=float)
    err = np.asarray(stacks["cube_err"], dtype=float)
    wave = np.asarray(stacks["rest_wave"], dtype=float)
    lya = float(config.LYA_REST)
    bounds = tuple(config.line_window)
    sm = method or getattr(config, "measure_stack_method", "biweight")
    combos = combos if combos is not None else _default_continuum_combos(config)

    run_header("sweep_continuum_method", verbose=verbose, nboot=nboot, stack=sm,
               window=bounds, n_combos=len(combos), seed=seed)
    out: dict = {}
    labels = []
    for (lab, cmeth, corder, cbounds) in combos:
        out[lab] = bootstrap_all(
            flux, err, wave, nboot=nboot, bounds=bounds, cont_bounds=cbounds,
            lya_center=lya, stack_method=sm, cont_method=cmeth, cont_order=corder,
            compute_side_ratio=True, compute_stack_error=False, seed=seed,
            verbose=verbose, label=f"continuum: {lab}", announce=False,
        )
        labels.append(lab)
    out["_meta"] = {
        "labels": labels, "nboot": nboot, "stack_method": sm,
        "r_edges": np.asarray(stacks["r_edges"]),
        "bin_mode": stacks.get("bin_mode"),
        "VR_biweight_v": stacks.get("VR_biweight_v"),
    }
    return stamp_provenance(out, config, "sweep_continuum_method")


def plot_continuum_sweep(
    sweep: dict, statistic="centroid",
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.8, 4.9),
    ylims="auto", xlims=None, jitter=0.05, title=None, save_fig=False,
):
    """Overlay the chosen statistic vs radius across continuum settings, via the
    shared engine. statistic='centroid' (default) or 'flux'."""
    meta = sweep["_meta"]
    labels = meta["labels"]
    series = _sweep_series(sweep, labels, labels, statistic)
    _, _, _, ylabel, default_ylims = _STAT_KEYS[statistic]
    return plot_radial_overlay(
        series, radial_bins=np.asarray(meta["r_edges"]),
        bin_mode=meta.get("bin_mode"), VR_biweight_v=meta.get("VR_biweight_v"),
        vr_ticks=vr_ticks, figsize=figsize,
        ylims=(default_ylims if ylims == "auto" else ylims), xlims=xlims,
        ylabel=ylabel, jitter=jitter, legend_title="continuum",
        title=title or f"{statistic} vs. radius — continuum sweep",
        save_fig=save_fig, savename="Figure_continuum_sweep.png",
    )


# =====================================================================
# 2c. BOOTSTRAP vs JACKKNIFE error cross-check
# =====================================================================
def _stat_per_stack(stack, rest_wave, bounds, lya, cont_kw, statistic):
    """(nrad,) statistic on one (nrad, nwave) stack: centroid km/s or integrated flux."""
    nrad = stack.shape[0]
    out = np.full(nrad, np.nan)
    for r in range(nrad):
        if statistic == "centroid":
            res = flux_weighted_centroid(rest_wave, stack[r], bounds=bounds,
                                         lya_center=lya, **cont_kw)
            out[r] = res["centroid_vel_kms"]
        else:
            res = integrated_line_flux(rest_wave, stack[r], bounds=bounds, **cont_kw)
            out[r] = res["flux_sum"]   # flux_sum matches bootstrap total_flux convention
    return out


def error_cross_check(
    config: "PipelineConfig", stacks: dict, product: GalaxyProduct | None = None,
    group_col: str | None = "FIELD", n_blocks: int = 10, statistic: str = "centroid",
    method: str | None = None, bounds=None, nboot: int = 300, seed: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Cross-check the galaxy BOOTSTRAP error against JACKKNIFE errors, per radial
    bin. A jackknife error much larger than the bootstrap is a warning that the
    bootstrap bars are too tight -- most likely on inner bins that rest on few
    galaxies (the cut_radial_bin=-1 nuance).

    Methods compared:
      * bootstrap      : 0.5*(p84-p16) from bootstrap_measurements.
      * block jackknife: partition galaxies into n_blocks, delete each block,
                         re-stack, re-measure; err = sqrt((K-1)/K * Σ(θ_i-θ̄)²).
      * group jackknife: if `product` + `group_col` (e.g. 'FIELD') are given,
                         delete-one-group (leave-one-field-out). Needs the SAME
                         product that built `stacks` (galaxy-axis alignment).

    statistic : 'centroid' (km/s) or 'flux' (integrated windowed flux).

    Returns dict with central, boot_err, jk_block_err, (group_err), ratios, axes.
    """
    if "cube_flux" not in stacks or "cube_err" not in stacks:
        raise KeyError("error_cross_check needs the per-galaxy cube (keep_cube=True).")
    if statistic not in ("centroid", "flux"):
        raise ValueError("statistic must be 'centroid' or 'flux'.")
    flux = np.asarray(stacks["cube_flux"], dtype=float)
    err = np.asarray(stacks["cube_err"], dtype=float)
    wave = np.asarray(stacks["rest_wave"], dtype=float)
    lya = float(config.LYA_REST)
    bounds = tuple(config.line_window) if bounds is None else tuple(bounds)
    cont_kw = _continuum_kwargs(config)
    sm = method or getattr(config, "measure_stack_method", "biweight")
    ngal, nrad, _ = flux.shape

    # --- bootstrap error ---
    boot = bootstrap_measurements(
        flux, err, wave, nboot=nboot, bounds=bounds, lya_center=lya,
        stack_method=sm, compute_side_ratio=(statistic == "flux"),
        seed=seed, verbose=verbose, **cont_kw)
    if statistic == "centroid":
        central = np.asarray(boot["centroid_v_fid"])
        boot_err = 0.5 * (np.asarray(boot["centroid_v_hi"]) - np.asarray(boot["centroid_v_lo"]))
    else:
        central = np.asarray(boot["total_flux_fid"])
        boot_err = 0.5 * (np.asarray(boot["total_flux_hi"]) - np.asarray(boot["total_flux_lo"]))

    # --- block jackknife ---
    rng = np.random.default_rng(seed)
    blocks = np.array_split(rng.permutation(ngal), n_blocks)
    theta = np.full((len(blocks), nrad), np.nan)
    for i, blk in enumerate(blocks):
        keep = np.ones(ngal, dtype=bool); keep[blk] = False
        stack_i, _ = stack_galaxies(flux[keep], err[keep] if err is not None else None, method=sm)
        theta[i] = _stat_per_stack(stack_i, wave, bounds, lya, cont_kw, statistic)
    K = len(blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        jk_mean = np.nanmean(theta, axis=0)
        jk_block_err = np.sqrt((K - 1) / K * np.nansum((theta - jk_mean) ** 2, axis=0))

    # --- group (delete-one) jackknife, optional ---
    group_err = None; groups = None
    if product is not None and group_col is not None and \
       group_col in getattr(product.catalog, "colnames", []):
        labels = np.asarray(product.catalog[group_col]).astype(str)
        if labels.size == ngal:
            groups = sorted(set(labels))
            if len(groups) >= 2:
                tg = np.full((len(groups), nrad), np.nan)
                for i, g in enumerate(groups):
                    keep = labels != g
                    stack_g, _ = stack_galaxies(flux[keep], err[keep] if err is not None else None, method=sm)
                    tg[i] = _stat_per_stack(stack_g, wave, bounds, lya, cont_kw, statistic)
                Kg = len(groups)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    gmean = np.nanmean(tg, axis=0)
                    group_err = np.sqrt((Kg - 1) / Kg * np.nansum((tg - gmean) ** 2, axis=0))
        elif verbose:
            print(f"group_col '{group_col}' length != ngal; skipping group jackknife.")

    ratio = jk_block_err / np.where(boot_err > 0, boot_err, np.nan)
    if verbose:
        edges = np.round(np.asarray(stacks["r_edges"]), 3)
        print(f"\nerror cross-check ({statistic}, method={sm}, "
              f"{n_blocks} blocks{', +group JK' if group_err is not None else ''}):")
        print(f"{'bin':>14}  {'boot':>9}  {'block JK':>9}  {'JK/boot':>8}"
              + ("  {:>9}".format('group JK') if group_err is not None else ""))
        for r in range(nrad):
            line = f"{edges[r]:g}-{edges[r+1]:g}".rjust(14) + \
                   f"  {boot_err[r]:9.3g}  {jk_block_err[r]:9.3g}  {ratio[r]:8.2f}"
            if group_err is not None:
                line += f"  {group_err[r]:9.3g}"
            print(line)

    return stamp_provenance({
        "statistic": statistic, "central": central,
        "boot_err": boot_err, "jk_block_err": jk_block_err,
        "group_err": group_err, "groups": groups, "ratio": ratio,
        "n_blocks": n_blocks,
        "r_edges": np.asarray(stacks["r_edges"]),
        "bin_mode": stacks.get("bin_mode"),
        "VR_biweight_v": stacks.get("VR_biweight_v"),
        "method": sm, "bounds": bounds,
    }, config, "error_cross_check")


def _jackknife_flux_sigma_repeated(flux, err, wave, bounds, cont_kw, stack_method,
                                   mode="block", n_blocks=40, n_reps=5, seed=1):
    """Block jackknife sigma averaged over n_reps independent partitions, to
    remove partition-luck scatter while preserving the genuine 'this bin leans
    on few galaxies' signal (a fragile bin stays high across ALL partitions).
    delete-one is partition-free, so n_reps is ignored there."""
    if mode == "delete-one":
        return _jackknife_flux_sigma(flux, err, wave, bounds, cont_kw,
                                     stack_method, mode=mode, seed=seed)
    central = sig_acc = None
    sigs = []
    for j in range(n_reps):
        c, s, _ = _jackknife_flux_sigma(flux, err, wave, bounds, cont_kw,
                                        stack_method, mode="block",
                                        n_blocks=n_blocks, seed=seed + j,
                                        progress=progress,
                                        desc=f"jackknife rep {j+1}/{n_reps}")
        central = c
        sigs.append(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sigma = np.nanmean(np.vstack(sigs), axis=0)       # average σ over partitions
        sigma_spread = np.nanstd(np.vstack(sigs), axis=0)  # how partition-dependent it is
    return central, sigma, sigma_spread


def plot_error_cross_check(
    result, vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.8, 4.9),
    ylims="auto", xlims=None, jitter=0.05, title=None, save_fig=False,
):
    """
    Overlay the same central values with bootstrap vs (block / group) jackknife
    error bars, via plot_radial_overlay -- so you can see directly whether the
    jackknife inflates the bars relative to the bootstrap.
    """
    central = np.asarray(result["central"])
    series = [
        {"label": "bootstrap", "y": central,
         "lo": central - result["boot_err"], "hi": central + result["boot_err"], "fmt": "o"},
        {"label": f"block JK (×{result['n_blocks']})", "y": central,
         "lo": central - result["jk_block_err"], "hi": central + result["jk_block_err"], "fmt": "s"},
    ]
    if result.get("group_err") is not None:
        series.append({"label": "group JK", "y": central,
                       "lo": central - result["group_err"], "hi": central + result["group_err"], "fmt": "D"})
    _, _, _, ylabel, default_ylims = _STAT_KEYS[result["statistic"]]
    return plot_radial_overlay(
        series, radial_bins=np.asarray(result["r_edges"]),
        bin_mode=result.get("bin_mode"), VR_biweight_v=result.get("VR_biweight_v"),
        vr_ticks=vr_ticks, figsize=figsize,
        ylims=(default_ylims if ylims == "auto" else ylims), xlims=xlims,
        ylabel=ylabel, jitter=jitter, legend_title="error method",
        title=title or f"Bootstrap vs jackknife error — {result['statistic']}",
        save_fig=save_fig, savename="Figure_error_cross_check.png",
    )


# =====================================================================
# 3. PLACEBO WAVELENGTHS  (estimator null on the REAL continuum)
# =====================================================================
def _moving_block_bootstrap_std(y, block_len, nboot, rng):
    """Moving-block bootstrap of _robust_std (and median) for a 1D sample whose
    entries may be autocorrelated (overlapping placebo windows). Resamples
    contiguous blocks of length `block_len` with replacement so the local
    correlation is preserved, then reports the across-replicate spread.

    Returns (std_point, std_lo, std_hi, med_lo, med_hi); the CI fields are the
    16/84 percentiles across replicates, NaN if the sample is too small to block.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    m = y.size
    if m < 4:
        return (_robust_std(y), np.nan, np.nan, np.nan, np.nan)
    L = int(max(1, min(block_len, m)))
    n_blocks = int(np.ceil(m / L))
    max_start = m - L
    stds = np.empty(nboot)
    meds = np.empty(nboot)
    for b in range(nboot):
        starts = (rng.integers(0, max_start + 1, size=n_blocks)
                  if max_start > 0 else np.zeros(n_blocks, dtype=int))
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:m]
        samp = y[idx]
        stds[b] = _robust_std(samp)
        meds[b] = np.nanmedian(samp)
    return (float(np.nanmedian(stds)),
            float(np.nanpercentile(stds, 16)), float(np.nanpercentile(stds, 84)),
            float(np.nanpercentile(meds, 16)), float(np.nanpercentile(meds, 84)))


def placebo_wavelengths(
    config: "PipelineConfig",
    stacks: dict,
    centers=None,
    method: str | None = None,
    half_window: float | None = None,
    step: float = 5.0,
    avoid=DEFAULT_UV_LINES,
    avoid_hw: float = 10.0,
    pad: float = 2.0,
    center_bounds=None,          # (lo, hi) rest-Å: restrict placebos to a clean band
    abs_avoid_nbins: int | None = 2,  # avoid ABSORPTION only in this many inner bins
    nboot_floor: int = 0,        # >0 -> moving-block bootstrap CI on the floor σ
    block_len: int | None = None,     # block size (centers); None -> auto from footprint
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Run the estimator at many LINE-FREE rest wavelengths on the real stacks. At
    each placebo center the line window AND the continuum sidebands track that
    center, so the measurement geometry is identical to the real Lyα measurement
    -- only the location is line-free. The integrated-flux scatter across centers
    is the estimator+continuum noise floor (the stable statistic); the centroid
    null is kept for reference.

    Three controls sharpen the floor:

    center_bounds   : (lo, hi) rest Å. Build the floor only from this band, so it
                      reflects local noise near Lyα and not the noisier blue side,
                      e.g. center_bounds=(1216, 1290). None -> full grid.
    abs_avoid_nbins : EMISSION lines are avoided at every radius (they can be real
                      signal). ABSORPTION lines are real signal only in the inner
                      bins -- past the inner continuum they wash out -- so they are
                      avoided only in the first `abs_avoid_nbins` bins and kept as
                      free noise-sampling wavelengths in the outer bins. This is
                      what fixes the outer-bin floor being starved of centers.
                      None (or >= n_bins) -> avoid absorption everywhere (old behavior).
    nboot_floor     : if >0, also moving-block bootstrap the per-bin floor σ over
                      its (correlated, overlapping) centers, returning a CI on the
                      floor. block_len auto-derives from the window+sideband
                      footprint so adjacent overlapping centers are kept together.

    centers : explicit list of placebo centers (Å). None -> auto grid in `step` Å,
              dropping any center whose footprint (window + sidebands ± pad)
              overlaps an EMISSION avoid line within ±avoid_hw.

    Returns the previous keys, plus:
        center_usable       (ncen, nrad) bool  which centers feed which bin's floor
        touches_abs         (ncen,) bool       center sits on an absorption line
        n_centers_per_bin   (nrad,) int
        center_bounds, abs_avoid_nbins, block_len
        (if nboot_floor>0) flux_null_std_boot/lo/hi (nrad,)  floor σ + 16/84 CI
    """
    sm = method or "biweight"
    lya = float(config.LYA_REST)
    hw = half_window if half_window is not None else 0.5 * (config.line_window[1] - config.line_window[0])
    rest_wave = np.asarray(stacks["rest_wave"])
    if sm not in stacks["stacks"]:
        raise KeyError(f"method {sm!r} not in stacks (have {list(stacks['stacks'])}).")
    flux = np.asarray(stacks["stacks"][sm]["flux"])     # (nrad, nwave)
    nrad = flux.shape[0]

    cont_method = getattr(config, "cont_method", "poly")
    cont_order = getattr(config, "cont_order", 1)
    cb0 = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
    offs = [(lo - lya, hi - lya) for lo, hi in cb0]     # sideband offsets vs Lya

    foot_lo = min([o[0] for o in offs] + [-hw])
    foot_hi = max([o[1] for o in offs] + [hw])

    # split avoid catalog by kind: emission always avoided, absorption radius-aware
    avoid_em = np.array([lam for _, lam, k in avoid if k == "em"], float)
    avoid_abs = np.array([lam for _, lam, k in avoid if k != "em"], float)

    def _touches(lams, c):
        if lams.size == 0:
            return False
        f_lo, f_hi = c + foot_lo - pad, c + foot_hi + pad
        return bool(np.any((lams >= f_lo) & (lams <= f_hi)) or
                    np.any(np.abs(lams - c) <= avoid_hw))

    if centers is None:
        lo = config.rest_wave_min - foot_lo + pad
        hi = config.rest_wave_max - foot_hi - pad
        if center_bounds is not None:
            lo = max(lo, float(center_bounds[0]))
            hi = min(hi, float(center_bounds[1]))
        grid = np.arange(np.ceil(lo), np.floor(hi) + 1, step)
        centers = [float(c) for c in grid if not _touches(avoid_em, c)]
    else:
        centers = list(np.atleast_1d(centers))
        if center_bounds is not None:
            centers = [float(c) for c in centers
                       if center_bounds[0] <= c <= center_bounds[1]]
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        raise ValueError("No line-free placebo centers found; widen the grid, "
                         "relax avoid_hw, or loosen center_bounds.")

    # which centers sit on an absorption line, and the per-bin usability mask
    touches_abs = np.array([_touches(avoid_abs, c) for c in centers], dtype=bool)
    n_inner = nrad if abs_avoid_nbins is None else int(abs_avoid_nbins)
    usable = (~touches_abs)[:, None] | (np.arange(nrad)[None, :] >= n_inner)  # (ncen,nrad)

    # --- measure every kept center in every bin (cheap; mask is applied later) ---
    v_off = np.full((centers.size, nrad), np.nan)
    f_off = np.full((centers.size, nrad), np.nan)
    for ci, c in enumerate(centers):
        bounds_c = (c - hw, c + hw)
        cont_c = tuple((c + o[0], c + o[1]) for o in offs)
        for r in range(nrad):
            res = flux_weighted_centroid(
                rest_wave, flux[r], bounds=bounds_c, cont_bounds=cont_c,
                lya_center=c, cont_method=cont_method, cont_order=cont_order)
            v_off[ci, r] = res["centroid_vel_kms"]
            fr = integrated_line_flux(
                rest_wave, flux[r], bounds=bounds_c, cont_bounds=cont_c,
                lya_center=c, cont_method=cont_method, cont_order=cont_order)
            f_off[ci, r] = fr["flux_sum"]

    if block_len is None:
        block_len = int(max(1, np.ceil((foot_hi - foot_lo) / step)))

    # --- per-bin aggregation over the USABLE centers only ---
    rng = np.random.default_rng(seed)
    f_med = np.full(nrad, np.nan); f_lo = np.full(nrad, np.nan); f_hi = np.full(nrad, np.nan)
    f_std = np.full(nrad, np.nan)
    null_med = np.full(nrad, np.nan); null_lo = np.full(nrad, np.nan); null_hi = np.full(nrad, np.nan)
    null_std = np.full(nrad, np.nan)
    success = np.full(nrad, np.nan)
    n_used = np.zeros(nrad, dtype=int)
    f_std_boot = np.full(nrad, np.nan)
    f_std_lo = np.full(nrad, np.nan); f_std_hi = np.full(nrad, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for r in range(nrad):
            m = usable[:, r]
            fr = f_off[m, r]; vr = v_off[m, r]
            n_used[r] = int(np.sum(np.isfinite(fr)))
            f_med[r] = np.nanmedian(fr)
            f_lo[r] = np.nanpercentile(fr, 16); f_hi[r] = np.nanpercentile(fr, 84)
            f_std[r] = _robust_std(fr)
            null_med[r] = np.nanmedian(vr)
            null_lo[r] = np.nanpercentile(vr, 16); null_hi[r] = np.nanpercentile(vr, 84)
            null_std[r] = _robust_std(vr)
            success[r] = np.mean(np.isfinite(fr)) if fr.size else np.nan
            if nboot_floor > 0:
                sb, slo, shi, _, _ = _moving_block_bootstrap_std(
                    fr, block_len, nboot_floor, rng)
                f_std_boot[r] = sb; f_std_lo[r] = slo; f_std_hi[r] = shi

    if verbose:
        print(f"placebo: {centers.size} centers (method={sm}, window=±{hw:.1f} Å, "
              f"abs avoided in first {n_inner} bin(s))")
        edges = np.round(np.asarray(stacks["r_edges"]), 3)
        for r in range(nrad):
            ci = (f" [σ {f_std_lo[r]:.3g}–{f_std_hi[r]:.3g}]"
                  if nboot_floor > 0 and np.isfinite(f_std_lo[r]) else "")
            print(f"  {edges[r]:g}-{edges[r+1]:g}: n={n_used[r]:3d}  "
                  f"flux null = {f_med[r]:+.3g} ± {f_std[r]:.3g}{ci} | "
                  f"centroid null = {null_med[r]:+.0f} ± {null_std[r]:.0f} km/s")

    return stamp_provenance({
        "centers": centers,
        "flux_offset": f_off, "v_offset": v_off,
        "flux_null_med": f_med, "flux_null_lo": f_lo, "flux_null_hi": f_hi,
        "flux_null_std": f_std,
        "null_med": null_med, "null_lo": null_lo, "null_hi": null_hi,
        "null_std": null_std,
        "success_frac": success,
        "center_usable": usable, "touches_abs": touches_abs,
        "n_centers_per_bin": n_used,
        "flux_null_std_boot": f_std_boot,
        "flux_null_std_lo": f_std_lo, "flux_null_std_hi": f_std_hi,
        "r_edges": np.asarray(stacks["r_edges"]),
        "bin_mode": stacks.get("bin_mode"),
        "VR_biweight_v": stacks.get("VR_biweight_v"),
        "unit_info": stacks.get("unit_info"),
        "half_window": hw, "method": sm,
        "center_bounds": (tuple(center_bounds) if center_bounds is not None else None),
        "abs_avoid_nbins": n_inner, "block_len": block_len,
    }, config, "placebo_wavelengths")

# def placebo_wavelengths(
#     config: "PipelineConfig",
#     stacks: dict,
#     centers=None,
#     method: str | None = None,
#     half_window: float | None = None,
#     step: float = 5.0,
#     avoid=DEFAULT_UV_LINES,
#     avoid_hw: float = 10.0,
#     pad: float = 2.0,
#     verbose: bool = True,
# ) -> dict:
#     """
#     Run the estimator at many LINE-FREE rest wavelengths on the real stacks. At
#     each placebo center the line window AND the continuum sidebands are shifted
#     to track that center, so the measurement geometry is identical to the real
#     Lya measurement -- only the location is line-free.

#     Measures BOTH per center: the integrated flux (flux_offset) and the centroid
#     velocity (v_offset). The FLUX null is the primary product -- it is the
#     estimator's noise floor for the same statistic the scramble null and SN
#     spectrum use, and (as established) is far more stable than the centroid,
#     which on line-free noise scatters wildly because it divides by a vanishing
#     line flux. The centroid null is kept for reference but is intrinsically broad.

#     On featureless continuum the integrated flux should scatter around ~0; its
#     spread is the noise floor you overlay on the real Lya integrated flux.

#     Complementary to redshift_scramble_null: the scramble tests the COADD (does
#     stacking manufacture a line?), the placebo tests the ESTIMATOR+CONTINUUM on
#     fixed real data at line-free wavelengths.

#     centers : explicit list of placebo rest centers (A). None -> auto grid from
#               rest_wave_min..rest_wave_max in `step` A, dropping any center whose
#               full footprint (window + both sidebands, ± pad) overlaps an `avoid`
#               line within ±avoid_hw.

#     Returns dict:
#         centers                (ncen,)
#         flux_offset            (ncen, nrad)   integrated flux about each center
#         v_offset               (ncen, nrad)   centroid vel about each center (km/s)
#         flux_null_med/lo/hi/std (nrad,)        flux null aggregated over centers
#         null_med/lo/hi/std     (nrad,)        centroid null (reference)
#         success_frac           (nrad,)
#         r_edges, bin_mode, VR_biweight_v, unit_info, half_window, method
#     """
#     sm = method or "biweight"
#     lya = float(config.LYA_REST)
#     hw = half_window if half_window is not None else 0.5 * (config.line_window[1] - config.line_window[0])
#     rest_wave = np.asarray(stacks["rest_wave"])
#     if sm not in stacks["stacks"]:
#         raise KeyError(f"method {sm!r} not in stacks (have {list(stacks['stacks'])}).")
#     flux = np.asarray(stacks["stacks"][sm]["flux"])     # (nrad, nwave)
#     nrad = flux.shape[0]

#     cont_method = getattr(config, "cont_method", "poly")
#     cont_order = getattr(config, "cont_order", 1)
#     cb0 = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
#     offs = [(lo - lya, hi - lya) for lo, hi in cb0]     # sideband offsets vs Lya

#     # footprint relative to a center, for the avoid test
#     foot_lo = min([o[0] for o in offs] + [-hw])
#     foot_hi = max([o[1] for o in offs] + [hw])

#     if centers is None:
#         lo = config.rest_wave_min - foot_lo + pad
#         hi = config.rest_wave_max - foot_hi - pad
#         grid = np.arange(np.ceil(lo), np.floor(hi) + 1, step)
#         avoid_lams = np.array([lam for _, lam, _ in avoid])
#         centers = []
#         for c in grid:
#             f_lo, f_hi = c + foot_lo - pad, c + foot_hi + pad
#             if not np.any((avoid_lams >= f_lo) & (avoid_lams <= f_hi)) and \
#                not np.any(np.abs(avoid_lams - c) <= avoid_hw):
#                 centers.append(float(c))
#     centers = np.asarray(centers, dtype=float)
#     if centers.size == 0:
#         raise ValueError("No line-free placebo centers found; widen the grid or "
#                          "relax avoid_hw.")

#     v_off = np.full((centers.size, nrad), np.nan)
#     f_off = np.full((centers.size, nrad), np.nan)
#     for ci, c in enumerate(centers):
#         bounds_c = (c - hw, c + hw)
#         cont_c = tuple((c + o[0], c + o[1]) for o in offs)
#         for r in range(nrad):
#             res = flux_weighted_centroid(
#                 rest_wave, flux[r], bounds=bounds_c, cont_bounds=cont_c,
#                 lya_center=c, cont_method=cont_method, cont_order=cont_order)
#             v_off[ci, r] = res["centroid_vel_kms"]      # already relative to c
#             fr = integrated_line_flux(
#                 rest_wave, flux[r], bounds=bounds_c, cont_bounds=cont_c,
#                 lya_center=c, cont_method=cont_method, cont_order=cont_order)
#             f_off[ci, r] = fr["flux_sum"]   # flux_sum matches bootstrap total_flux convention

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         null_med = np.nanmedian(v_off, axis=0)
#         null_lo = np.nanpercentile(v_off, 16, axis=0)
#         null_hi = np.nanpercentile(v_off, 84, axis=0)
#         null_std = np.array([_robust_std(v_off[:, r]) for r in range(nrad)])
#         f_med = np.nanmedian(f_off, axis=0)
#         f_lo = np.nanpercentile(f_off, 16, axis=0)
#         f_hi = np.nanpercentile(f_off, 84, axis=0)
#         f_std = np.array([_robust_std(f_off[:, r]) for r in range(nrad)])
#     success = np.mean(np.isfinite(f_off), axis=0)

#     if verbose:
#         print(f"placebo: {centers.size} line-free centers, method={sm}, "
#               f"window=±{hw:.1f} Å")
#         edges = np.round(np.asarray(stacks["r_edges"]), 3)
#         for r in range(nrad):
#             print(f"  {edges[r]:g}-{edges[r+1]:g}: flux null = {f_med[r]:+.3g} "
#                   f"± {f_std[r]:.3g} | centroid null = {null_med[r]:+.0f} "
#                   f"± {null_std[r]:.0f} km/s  (ok {success[r]*100:.0f}%)")

#     return stamp_provenance({
#         "centers": centers,
#         "flux_offset": f_off, "v_offset": v_off,
#         "flux_null_med": f_med, "flux_null_lo": f_lo, "flux_null_hi": f_hi,
#         "flux_null_std": f_std,
#         "null_med": null_med, "null_lo": null_lo, "null_hi": null_hi,
#         "null_std": null_std,
#         "success_frac": success,
#         "r_edges": np.asarray(stacks["r_edges"]),
#         "bin_mode": stacks.get("bin_mode"),
#         "VR_biweight_v": stacks.get("VR_biweight_v"),
#         "unit_info": stacks.get("unit_info"),
#         "half_window": hw, "method": sm,
#     }, config, "placebo_wavelengths")


def plot_placebo(result, real_boot=None, which="flux",
                 vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.4, 4.8),
                 ylims=None, xlims=None, logy=False, title=None, save_fig=False):
    """
    Placebo null band with the real Lya measurement overlaid, via plot_null_band.
    which='flux' (default, the stable statistic) plots the integrated-flux null
    and the real integrated flux (real_boot total_flux_* or blue+red); which=
    'centroid' plots the centroid null (intrinsically broad -- mainly for
    reference) and the real centroid.

    Passing real_boot draws the real Lya measurement WITH its galaxy-bootstrap
    16/84 band, so this plot already does the bootstrap-vs-null comparison the
    scramble plot's compare_boot flag adds: the placebo band is the
    estimator+continuum noise floor, the bootstrap band is the sampling error on
    the real line.

    logy : log y-axis (which='flux' only) to see the faint outer bins; ignored
           for the centroid view (velocities are signed and centred on 0).
    """
    if which == "flux":
        y_unit = (result.get("unit_info") or {}).get("y_unit", "")
        real_y, real_lo, real_hi = _boot_total_flux(real_boot)
        return plot_null_band(
            radial_bins=np.asarray(result["r_edges"]),
            null_lo=result["flux_null_lo"], null_hi=result["flux_null_hi"],
            null_med=result["flux_null_med"], real_y=real_y,
            real_lo=real_lo, real_hi=real_hi,
            bin_mode=result.get("bin_mode"), VR_biweight_v=result.get("VR_biweight_v"),
            vr_ticks=vr_ticks, figsize=figsize, ylims=ylims, xlims=xlims, logy=logy,
            ylabel=(f"Integrated flux [{y_unit}]" if y_unit else "Integrated flux"),
            title=title or "Lyα integrated flux vs. placebo-wavelength null",
            null_label="placebo flux null (16–84%)", real_label="real Lyα flux",
            save_fig=save_fig, savename="Figure_placebo.png",
        )
    else:  # centroid (reference)
        real_y = real_lo = real_hi = None
        if real_boot is not None and "centroid_v_med" in real_boot:
            real_y = np.asarray(real_boot["centroid_v_med"])
            real_lo = np.asarray(real_boot["centroid_v_lo"]); real_hi = np.asarray(real_boot["centroid_v_hi"])
        return plot_null_band(
            radial_bins=np.asarray(result["r_edges"]),
            null_lo=result["null_lo"], null_hi=result["null_hi"],
            null_med=result["null_med"], real_y=real_y, real_lo=real_lo, real_hi=real_hi,
            bin_mode=result.get("bin_mode"), VR_biweight_v=result.get("VR_biweight_v"),
            vr_ticks=vr_ticks, figsize=figsize,
            ylims=(-200, 200) if ylims is None else ylims, xlims=xlims,
            ylabel=r"Centroid velocity [km s$^{-1}$]",
            title=title or "Lyα centroid vs. placebo-wavelength null (reference)",
            null_label="placebo centroid null (16–84%)", real_label="real Lyα centroid",
            save_fig=save_fig, savename="Figure_placebo_centroid.png",
        )


# =====================================================================
# 4. INJECTION -- RECOVERY  (into the empty z-scramble null)
# =====================================================================
# Centroid-estimator specs: label -> kwargs forwarded to bootstrap_measurements.
# These are ALIASES of the canonical vocabulary in measure.py, so the same label
# means the same thing in the analysis tables and in these stress-tests. To add
# or rename an estimator, edit measure.CENTROID_METHOD_SPECS (one place), not here.
# =====================================================================
_INJ_METHODS = CENTROID_METHOD_SPECS
_METHOD_LABELS = CENTROID_METHOD_LABELS


def injection_recovery(
    config: "PipelineConfig",
    product: GalaxyProduct,
    v_inputs=(-150, -100, -50, 0, 50, 100, 150),
    snr: float = 5.0,
    sigma_A: float = 1.5,
    nboot: int = 200,
    methods=("fw_clip", "fw_noclip", "gaussian"),
    method: str | None = None,        # back-compat: old arg was the STACK combine
    stack_method: str = "biweight",
    n_noise: int = 1,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Inject a synthetic Gaussian Lya line of KNOWN velocity into the empty
    (z-scrambled, line-free) cube at a fixed peak S/N, then recover it with EACH
    requested centroid estimator. This is what quantifies the max(0,y) clip bias:
    'fw_clip' vs 'fw_noclip' on the same injected cube and the same bootstrap
    draws isolates the clip's effect; 'gaussian' / 'flux_median' / 'peak' are
    structurally-different cross-checks.

    Per estimator it calibrates (a) the centroid BIAS = recovered - input and
    (b) whether the bootstrap 16/84 interval COVERS the truth.

    Because the injected line is a SYMMETRIC Gaussian, this test isolates
    NOISE-INDUCED estimator bias (the clip question). It does NOT probe how an
    estimator responds to Lya's intrinsic red asymmetry -- that only shows up on
    the real line, where the clip choice also changes wing-weighting.

    The substrate is a scrambled-z cube: real per-fiber/per-galaxy noise, no
    coherent line. Per radial bin the injected peak amplitude is set to
    snr × (per-pixel stack noise near line center), so every bin is tested at the
    same input S/N regardless of its brightness.

    Parameters
    ----------
    methods : iterable of labels from _INJ_METHODS, or an explicit
              {label: kwargs} dict to override. Default compares clipped vs
              unclipped flux-weighted vs Gaussian.
    stack_method : galaxy combine used inside every bootstrap (held fixed across
                   estimators so ONLY the centroid estimator varies).
    method  : DEPRECATED. The old signature used `method` as the STACK combine.
              If given (and `methods` is left at its default), it is mapped to
              stack_method and the estimator set collapses to {'fw_clip'} so old
              calls reproduce the old single-curve result.
    v_inputs : input velocities to test (km/s).
    snr      : injected PEAK signal-to-noise per radial bin.
    sigma_A  : injected line sigma in rest A.
    n_noise  : independent scrambled cubes. 1 -> bias/recovery only (cheap).
               >1 -> also estimates coverage over noise realizations.

    Returns
    -------
    dict
        v_input    (n_v,)
        methods    list of labels actually run
        by_method  {label: {recovered, rec_p16, rec_p84, boot_med, rec_lo,
                   rec_hi, bias, coverage, success_frac, empty_centroid,
                   additive_a, slope_alpha}}, each array (n_v, nrad) except
                   empty_centroid/additive_a/slope_alpha which are (nrad,).
                   rec_p16/rec_p84 are the across-noise 16/84 (use these for
                   error bars -- always consistent with `recovered`); rec_lo/
                   rec_hi are the legacy j==0 bootstrap bars. additive_a /
                   slope_alpha come from recovered = a + alpha*v per bin.
        snr, sigma_A, r_edges, nrad, bin_mode, VR_biweight_v, n_noise,
        stack_method, bounds, window_center_kms, noise_blue, noise_red
        # back-compat: top-level mirror of the FIRST method
        recovered, rec_lo, rec_hi, bias, coverage, method
    """
    # ---- resolve estimator set (+ back-compat for the old `method` arg) ------
    default_methods = ("fw_clip", "fw_noclip", "gaussian")
    if isinstance(methods, dict):
        method_specs = {k: dict(v) for k, v in methods.items()}
    else:
        method_specs = {}
        for lab in methods:
            if lab not in _INJ_METHODS:
                raise ValueError(f"unknown method label {lab!r}; "
                                 f"choose from {sorted(_INJ_METHODS)}")
            method_specs[lab] = dict(_INJ_METHODS[lab])
    if method is not None and tuple(methods) == default_methods:
        # old call style: `method` was the stack combine, not a centroid estimator
        stack_method = method
        method_specs = {"fw_clip": dict(_INJ_METHODS["fw_clip"])}

    labels = list(method_specs)
    sm = stack_method
    lya = float(config.LYA_REST)
    bounds = tuple(config.line_window)
    cont_kw = _continuum_kwargs(config)
    v_inputs = np.asarray(v_inputs, dtype=float)
    rng = np.random.default_rng(seed)

    rec_all = {lab: [] for lab in labels}             # per method: list of (n_v, nrad)
    med_all = {lab: [] for lab in labels}             # bootstrap-median per realization
    empty_all = {lab: [] for lab in labels}           # empty-substrate centroid per realization
    lo_first = {lab: None for lab in labels}
    hi_first = {lab: None for lab in labels}
    cover_hits = {lab: None for lab in labels}
    succ_acc = {lab: None for lab in labels}
    blue_noise = red_noise = None
    st = None
    nrad = None

    for j in range(n_noise):
        sp = _scramble_product(product, config.z_col, rng)
        st = build_stacks(config, sp, keep_cube=True)
        cube0 = np.asarray(st["cube_flux"], dtype=float)
        cube0_e = np.asarray(st["cube_err"], dtype=float)
        wave = np.asarray(st["rest_wave"])
        ngal, nrad, nwave = cube0.shape

        # per-bin per-pixel noise near line center -> injected amplitude per bin
        bse = bootstrap_stack_error(cube0, cube0_e, nboot=nboot, stack_method=sm,
                                    seed=seed + 1, verbose=False)
        noise = bse["stack_err"]                 # (nrad, nwave)
        near = np.abs(wave - lya) <= max(2 * sigma_A, 3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sigma_pix = np.nanmedian(noise[:, near], axis=1)   # (nrad,)
        amp = snr * sigma_pix                                  # (nrad,)

        # blue/red half-window noise (asymmetric floor -> additive clip bias);
        # captured once, it is a property of the substrate not the realization.
        if blue_noise is None:
            in_win = (wave >= bounds[0]) & (wave <= bounds[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                blue_noise = np.nanmedian(noise[:, in_win & (wave < lya)], axis=1)
                red_noise = np.nanmedian(noise[:, in_win & (wave >= lya)], axis=1)

        if j == 0:
            for lab in labels:
                lo_first[lab] = np.full((v_inputs.size, nrad), np.nan)
                hi_first[lab] = np.full((v_inputs.size, nrad), np.nan)
                cover_hits[lab] = np.zeros((v_inputs.size, nrad))
                succ_acc[lab] = np.zeros((v_inputs.size, nrad))
        rec_j = {lab: np.full((v_inputs.size, nrad), np.nan) for lab in labels}
        med_j = {lab: np.full((v_inputs.size, nrad), np.nan) for lab in labels}

        # empty substrate (NO line injected) -- the line-independent floor.
        # One coadd, measured per estimator (cheap; no bootstrap needed here).
        empty_stack = coadd_galaxies(cube0, cube0_e, [sm])[sm]["flux"]
        for lab in labels:
            em = np.full(nrad, np.nan)
            for rb in range(nrad):
                er = measure_centroid(
                    wave, empty_stack[rb],
                    method=method_specs[lab]["centroid_method"],
                    clip_negative=method_specs[lab].get("clip_negative", True),
                    bounds=bounds, lya_center=lya, **cont_kw)
                em[rb] = er["centroid_vel_kms"]
            empty_all[lab].append(em)

        for vi, v in enumerate(v_inputs):
            lam_c = lya * (1.0 + v / C_KMS)
            profile = np.exp(-0.5 * ((wave - lam_c) / sigma_A) ** 2)   # (nwave,)
            line = amp[:, None] * profile[None, :]                     # (nrad, nwave)
            cube_inj = cube0 + line[None, :, :]            # broadcast over galaxies

            for lab in labels:
                # SAME seed across estimators -> identical resample draws, so any
                # difference between estimators is purely the estimator.
                res = bootstrap_measurements(
                    cube_inj, cube0_e, wave, nboot=nboot, bounds=bounds,
                    lya_center=lya, stack_method=sm, compute_side_ratio=False,
                    seed=seed + 7, verbose=False,
                    **method_specs[lab], **cont_kw)
                rec_j[lab][vi] = res["centroid_v_fid"]
                med_j[lab][vi] = res["centroid_v_med"]
                if j == 0:
                    lo_first[lab][vi] = res["centroid_v_lo"]
                    hi_first[lab][vi] = res["centroid_v_hi"]
                succ_acc[lab][vi] += res["success_frac"]
                cover_hits[lab][vi] += ((v >= res["centroid_v_lo"]) &
                                        (v <= res["centroid_v_hi"])).astype(float)

        for lab in labels:
            rec_all[lab].append(rec_j[lab])
            med_all[lab].append(med_j[lab])
        if verbose:
            print(f"  noise realization {j + 1}/{n_noise} done "
                  f"({len(labels)} estimators)")

    # ---- aggregate per method ------------------------------------------------
    by_method = {}
    for lab in labels:
        ra = np.array(rec_all[lab])                    # (n_noise, n_v, nrad)
        recovered = np.nanmedian(ra, axis=0)           # (n_v, nrad)
        # across-noise 16/84 -> bars that are ALWAYS consistent with `recovered`
        # (the j==0 bootstrap bars conflate noise fields and can sit off the point)
        p16 = np.nanpercentile(ra, 16, axis=0)
        p84 = np.nanpercentile(ra, 84, axis=0)
        boot_med = np.nanmedian(np.array(med_all[lab]), axis=0)
        empty = np.nanmedian(np.array(empty_all[lab]), axis=0)   # (nrad,)
        # recovered = a + alpha*v per bin (additive offset vs slope flattening)
        a = np.full(nrad, np.nan); alpha = np.full(nrad, np.nan)
        for rb in range(nrad):
            y = recovered[:, rb]; good = np.isfinite(y)
            if good.sum() >= 2:
                alpha[rb], a[rb] = np.polyfit(v_inputs[good], y[good], 1)
        by_method[lab] = {
            "recovered": recovered,
            "rec_p16": p16, "rec_p84": p84,
            "boot_med": boot_med,
            "rec_lo": lo_first[lab],            # j==0 bootstrap bars (back-compat)
            "rec_hi": hi_first[lab],
            "bias": recovered - v_inputs[:, None],
            "coverage": cover_hits[lab] / n_noise,
            "success_frac": succ_acc[lab] / n_noise,
            "empty_centroid": empty,
            "additive_a": a, "slope_alpha": alpha,
        }

    if verbose:
        edges = np.round(np.asarray(st["r_edges"]), 3)
        print(f"\ninjection-recovery (peak S/N={snr}, σ={sigma_A} Å, "
              f"{n_noise} noise real, stack={sm}):")
        for lab in labels:
            mb = np.nanmedian(np.abs(by_method[lab]["bias"]), axis=0)
            print(f"  [{_METHOD_LABELS.get(lab, lab)}]")
            for r in range(nrad):
                cov = (np.nanmean(by_method[lab]["coverage"][:, r])
                       if n_noise > 1 else np.nan)
                cov_s = f", coverage {cov*100:.0f}%" if n_noise > 1 else ""
                print(f"    {edges[r]:g}-{edges[r+1]:g}: median |bias| "
                      f"{mb[r]:.1f} km/s{cov_s}")

    first = labels[0]
    return {
        "v_input": v_inputs,
        "methods": labels,
        "by_method": by_method,
        "snr": snr, "sigma_A": sigma_A,
        "r_edges": np.asarray(st["r_edges"]),
        "nrad": nrad,
        "bin_mode": st.get("bin_mode"),
        "VR_biweight_v": st.get("VR_biweight_v"),
        "n_noise": n_noise, "stack_method": sm,
        "bounds": bounds,
        "window_center_kms": C_KMS * (0.5 * (bounds[0] + bounds[1]) - lya) / lya,
        "noise_blue": blue_noise, "noise_red": red_noise,
        # back-compat top-level mirror of the first estimator
        "recovered": by_method[first]["recovered"],
        "rec_lo": by_method[first]["rec_lo"],
        "rec_hi": by_method[first]["rec_hi"],
        "bias": by_method[first]["bias"],
        "coverage": by_method[first]["coverage"],
        "method": first,
    }


def injection_bias_fast(
    config: "PipelineConfig",
    product: GalaxyProduct,
    v_inputs=(-150, -75, 0, 75, 150),
    snr: float = 5.0, sigma_A: float = 1.2,
    methods=("fw_clip", "gaussian", "flux_median"),
    stack_method: str = "biweight",
    n_noise: int = 100, seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Fast sibling of injection_recovery: same scrambled-substrate injection, but
    with the per-draw galaxy bootstrap STRIPPED OUT. The bias only needs the
    fiducial recovered centroid per noise field, so dropping the bootstrap lets
    n_noise reach ~100 cheaply (cost ~ n_noise x build_stacks, no nboot factor).

    Per noise field it scrambles z (_scramble_product), builds the stack, and
    measures each estimator's centroid on the (full-sample) injected stack -- no
    resampling. The injected amplitude is set from the per-pixel MAD across the
    galaxy axis (1.4826*MAD/sqrt(ngal)), so no bootstrap_stack_error call either.

    Returns the SAME by_method schema as injection_recovery (so the same plotters
    work), minus the bootstrap-only fields (rec_lo/rec_hi/boot_med/coverage):
        recovered, rec_p16, rec_p84   across-noise median + 16/84 (consistent bars)
        bias, empty_centroid, additive_a, slope_alpha, success_frac
    plus v_input, methods, snr, sigma_A, r_edges, nrad, bounds, n_noise,
    stack_method, window_center_kms, noise_blue, noise_red.

    Use this to explore (S/N sweeps, many noise fields); use injection_recovery
    when you also need the bootstrap interval / coverage calibration.
    """
    if isinstance(methods, dict):
        specs = {k: dict(v) for k, v in methods.items()}
    else:
        specs = {lab: dict(_INJ_METHODS[lab]) for lab in methods}
    labels = list(specs)

    lya = float(config.LYA_REST)
    bounds = tuple(config.line_window)
    cont_kw = _continuum_kwargs(config)
    v_inputs = np.asarray(v_inputs, dtype=float)
    rng = np.random.default_rng(seed)

    rec_all = {lab: [] for lab in labels}     # each -> list of (n_v, nrad)
    empty_all = {lab: [] for lab in labels}   # each -> list of (nrad,)
    succ_acc = {lab: None for lab in labels}
    amp = None
    blue_noise = red_noise = None
    nrad = None
    st = None

    for j in range(n_noise):
        sp = _scramble_product(product, config.z_col, rng)
        st = build_stacks(config, sp, keep_cube=True)
        cube0 = np.asarray(st["cube_flux"], dtype=float)
        cube0_e = np.asarray(st["cube_err"], dtype=float)
        wave = np.asarray(st["rest_wave"])
        ngal, nrad, nwave = cube0.shape

        # amplitude via MAD across galaxies (once; no bootstrap)
        if amp is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(cube0, axis=0, keepdims=True)
                noise = 1.4826 * np.nanmedian(np.abs(cube0 - med), axis=0) / np.sqrt(ngal)
                near = np.abs(wave - lya) <= max(2 * sigma_A, 3.0)
                sigma_pix = np.nanmedian(noise[:, near], axis=1)
                amp = snr * sigma_pix
                in_win = (wave >= bounds[0]) & (wave <= bounds[1])
                blue_noise = np.nanmedian(noise[:, in_win & (wave < lya)], axis=1)
                red_noise = np.nanmedian(noise[:, in_win & (wave >= lya)], axis=1)
            for lab in labels:
                succ_acc[lab] = np.zeros((v_inputs.size, nrad))

        # empty substrate (no line)
        empty_stack = coadd_galaxies(cube0, cube0_e, [stack_method])[stack_method]["flux"]
        for lab in labels:
            em = np.full(nrad, np.nan)
            for rb in range(nrad):
                er = measure_centroid(
                    wave, empty_stack[rb], method=specs[lab]["centroid_method"],
                    clip_negative=specs[lab].get("clip_negative", True),
                    bounds=bounds, lya_center=lya, **cont_kw)
                em[rb] = er["centroid_vel_kms"]
            empty_all[lab].append(em)

        rec_j = {lab: np.full((v_inputs.size, nrad), np.nan) for lab in labels}
        for vi, v in enumerate(v_inputs):
            lam_c = lya * (1.0 + v / C_KMS)
            profile = np.exp(-0.5 * ((wave - lam_c) / sigma_A) ** 2)
            line = amp[:, None] * profile[None, :]
            cube_inj = cube0 + line[None, :, :]
            stk = coadd_galaxies(cube_inj, cube0_e, [stack_method])[stack_method]["flux"]
            for lab in labels:
                for rb in range(nrad):
                    er = measure_centroid(
                        wave, stk[rb], method=specs[lab]["centroid_method"],
                        clip_negative=specs[lab].get("clip_negative", True),
                        bounds=bounds, lya_center=lya, **cont_kw)
                    rec_j[lab][vi, rb] = er["centroid_vel_kms"]
                    succ_acc[lab][vi, rb] += float(er["success"])
        for lab in labels:
            rec_all[lab].append(rec_j[lab])

        if verbose and (j + 1) % max(1, n_noise // 10) == 0:
            print(f"  {j + 1}/{n_noise} noise fields done")

    window_center_kms = C_KMS * (0.5 * (bounds[0] + bounds[1]) - lya) / lya
    by_method = {}
    for lab in labels:
        ra = np.array(rec_all[lab])                    # (n_noise, n_v, nrad)
        recovered = np.nanmedian(ra, axis=0)
        p16 = np.nanpercentile(ra, 16, axis=0)
        p84 = np.nanpercentile(ra, 84, axis=0)
        empty = np.nanmedian(np.array(empty_all[lab]), axis=0)
        a = np.full(nrad, np.nan); alpha = np.full(nrad, np.nan)
        for rb in range(nrad):
            y = recovered[:, rb]; good = np.isfinite(y)
            if good.sum() >= 2:
                alpha[rb], a[rb] = np.polyfit(v_inputs[good], y[good], 1)
        by_method[lab] = {
            "recovered": recovered, "rec_p16": p16, "rec_p84": p84,
            "bias": recovered - v_inputs[:, None],
            "empty_centroid": empty, "additive_a": a, "slope_alpha": alpha,
            "success_frac": succ_acc[lab] / n_noise,
        }

    if verbose:
        _print_injection_flags({
            "methods": labels, "by_method": by_method, "nrad": nrad,
            "bounds": bounds, "window_center_kms": window_center_kms,
            "noise_blue": blue_noise, "noise_red": red_noise,
            "snr": snr, "n_noise": n_noise})

    return {
        "v_input": v_inputs, "methods": labels, "by_method": by_method,
        "snr": snr, "sigma_A": sigma_A, "n_noise": n_noise,
        "stack_method": stack_method, "r_edges": np.asarray(st["r_edges"]),
        "bin_mode": st.get("bin_mode"), "VR_biweight_v": st.get("VR_biweight_v"),
        "nrad": nrad, "bounds": bounds,
        "window_center_kms": window_center_kms,
        "noise_blue": blue_noise, "noise_red": red_noise,
    }


def _print_injection_flags(result):
    """Per-bin diagnostic flags shared by injection_bias_fast (and printable for
    injection_recovery): window centering, blue/red noise asymmetry, and the
    empty-substrate / additive / slope decomposition per estimator."""
    bounds = result["bounds"]; wc = result["window_center_kms"]
    print(f"\n=== injection-bias flags (n_noise={result['n_noise']}, "
          f"peak S/N={result['snr']}) ===")
    flag = "  <-- OFF-CENTER" if abs(wc) > 10 else "  (centered OK)"
    print(f"window {bounds[0]:g}-{bounds[1]:g} A  center offset {wc:+.1f} km/s{flag}")
    if result.get("noise_blue") is not None:
        br = np.asarray(result["noise_blue"]) / np.asarray(result["noise_red"])
        asym = np.nanmax(np.abs(br - 1)) > 0.15
        print("blue/red half-window noise ratio per bin: "
              + ", ".join(f"{x:.2f}" for x in br)
              + ("   <-- ASYMMETRIC FLOOR" if asym else ""))
    for lab in result["methods"]:
        d = result["by_method"][lab]
        if "additive_a" not in d:
            continue
        print(f"\n[{_METHOD_LABELS.get(lab, lab)}]   recovered = a + alpha*v")
        for rb in range(result["nrad"]):
            em = d["empty_centroid"][rb]; a = d["additive_a"][rb]; al = d["slope_alpha"][rb]
            note = ""
            if np.isfinite(em) and np.isfinite(a) and abs(a) > 5 and abs(em - a) < 0.4 * abs(a):
                note = "  <-- OFFSET TRACKS EMPTY SUBSTRATE"
            elif np.isfinite(al) and al < 0.7:
                note = "  <-- SLOPE FLATTENING"
            print(f"  bin {rb}: empty={em:+6.1f}  a={a:+6.1f}  alpha={al:5.2f}{note}")


def plot_injection_recovery(result, radial_index=-1, figsize=(6.8, 6.4),
                            methods=None, min_success=0.6, title=None,
                            save_fig=False):
    """
    Recovered vs input centroid for one radial bin, with the 1:1 line.

    The error bars are drawn from the SAME distribution as the plotted point, so
    they cannot sit off it (the old version plotted the fiducial point but the
    j==0 bootstrap bar, which for n_noise>1 could leave the point outside its own
    bar -- even crash on negative yerr). Preference order for the interval:
      1. rec_p16 / rec_p84   across-noise 16/84 (present in both fast + long)
      2. rec_lo  / rec_hi    j==0 bootstrap (legacy long-test fallback)
    The point is `recovered` (across-noise median). Points whose success_frac is
    below `min_success` are drawn faint, so a noise-latched / railed bin is
    flagged rather than passing as a clean measurement. If a fit line (a+alpha*v)
    is available it is overlaid.

    methods : optional subset of labels to plot (default: all in the result).
    """
    v_in = np.asarray(result["v_input"])
    edges = np.round(np.asarray(result["r_edges"]), 3)
    nrad = len(edges) - 1
    rr = radial_index if radial_index >= 0 else nrad + radial_index

    if "by_method" in result:
        labs = list(methods) if methods is not None else list(result["methods"])
        series = [(lab, result["by_method"][lab]) for lab in labs]
    else:  # legacy single-series result
        series = [(result.get("method", "recovered"), result)]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(series), 1)))
    span = (v_in.max() - v_in.min()) or 1.0
    jit = 0.012 * span
    all_y = [v_in]

    for k, (lab, d) in enumerate(series):
        rec = np.asarray(d["recovered"])[:, rr]
        if "rec_p16" in d and "rec_p84" in d:
            lo = np.asarray(d["rec_p16"])[:, rr]; hi = np.asarray(d["rec_p84"])[:, rr]
        else:
            lo = np.asarray(d["rec_lo"])[:, rr]; hi = np.asarray(d["rec_hi"])[:, rr]
        sf = (np.asarray(d["success_frac"])[:, rr] if "success_frac" in d
              else np.ones_like(rec))
        all_y.append(rec[np.isfinite(rec)])
        dx = (k - (len(series) - 1) / 2.0) * jit
        # clip guards against any residual point/interval mismatch (never <0)
        elo = np.clip(rec - lo, 0, None); ehi = np.clip(hi - rec, 0, None)
        ok = sf >= min_success
        lbl = _METHOD_LABELS.get(lab, lab)
        if "slope_alpha" in d and np.isfinite(d["slope_alpha"][rr]):
            lbl = f"{lbl}  (a={d['additive_a'][rr]:+.0f}, α={d['slope_alpha'][rr]:.2f})"
        ax.errorbar((v_in + dx)[ok], rec[ok],
                    yerr=np.vstack([elo[ok], ehi[ok]]),
                    fmt="o", capsize=3.0, ms=5.5, lw=1.4, color=colors[k], label=lbl)
        if np.any(~ok):
            ax.errorbar((v_in + dx)[~ok], rec[~ok],
                        yerr=np.vstack([elo[~ok], ehi[~ok]]),
                        fmt="o", capsize=3.0, ms=5.5, lw=1.2, color=colors[k], alpha=0.25)
        if "slope_alpha" in d and np.isfinite(d["slope_alpha"][rr]):
            xs = np.array([v_in.min(), v_in.max()])
            ax.plot(xs, d["additive_a"][rr] + d["slope_alpha"][rr] * xs,
                    color=colors[k], lw=1.0, ls="--", alpha=0.5)

    ymin = min(v_in.min(), *(np.nanmin(a) for a in all_y if len(a)))
    ymax = max(v_in.max(), *(np.nanmax(a) for a in all_y if len(a)))
    lim = (ymin - 30, ymax + 30)
    ax.plot(lim, lim, color="0.5", ls="--", lw=1, label="1:1 (unbiased)", zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Injected velocity [km s$^{-1}$]")
    ax.set_ylabel(r"Recovered centroid [km s$^{-1}$]")
    ax.set_title(title or f"Injection–recovery, bin {rr} "
                 f"({edges[rr]:g}–{edges[rr+1]:g}), peak S/N={result.get('snr')}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8.0, loc="best")
    if save_fig:
        plt.savefig(f"Figure_injection_recovery_bin{rr}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_bias_summary(result, methods=None, figsize=(8.6, 9.2),
                      untrusted_rvir=2.0, alpha_ylim=(0.55, 1.15),
                      title=None, save_fig=False):
    """
    One figure that makes an injection result legible: the three quantities that
    carry the physics, all vs radial bin. Works for BOTH injection_recovery and
    injection_bias_fast (both populate slope_alpha / additive_a / empty_centroid).

      1. slope alpha    recovery gain; alpha<1 = compression toward zero (robust,
                        stable even at low n_noise).
      2. intercept a    additive offset at v=0 [km/s]; needs high n_noise to trust.
      3. empty centroid centroid of the substrate with NO line; noisy in the inner
                        bins, diverges in the low-S/N outer bins (shaded).
    """
    edges = np.round(np.asarray(result["r_edges"]), 3)
    nrad = result.get("nrad", len(edges) - 1)
    labs = list(methods) if methods is not None else list(result["methods"])
    x = np.arange(nrad)
    binlabels = [f"{edges[i]:g}-{edges[i+1]:g}" for i in range(nrad)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labs), 1)))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw=dict(hspace=0.10, height_ratios=[1, 1, 1.1]))

    lo_edges = edges[:-1]
    untrusted = np.where(lo_edges >= untrusted_rvir)[0]

    def shade(ax):
        if untrusted.size:
            ax.axvspan(untrusted[0] - 0.5, nrad - 0.5, color="0.92", zorder=0)

    for k, lab in enumerate(labs):
        d = result["by_method"][lab]
        c = colors[k]
        ax1.plot(x, d["slope_alpha"], "o-", color=c, lw=1.5, ms=5.5, label=lab)
        ax2.plot(x, d["additive_a"], "o-", color=c, lw=1.5, ms=5.5, label=lab)
        ax3.plot(x, d["empty_centroid"], "o-", color=c, lw=1.5, ms=5.5, label=lab)

    for ax in (ax1, ax2, ax3):
        shade(ax); ax.grid(alpha=0.25)
    ax1.axhline(1.0, color="0.35", ls="--", lw=1)
    ax1.text(0.01, 1.0, " unbiased (α=1)", transform=ax1.get_yaxis_transform(),
             va="bottom", ha="left", color="0.4", fontsize=8.5)
    ax2.axhline(0.0, color="0.35", ls="--", lw=1)
    ax3.axhline(0.0, color="0.35", ls="--", lw=1)

    ax1.set_ylabel("slope α\n(recovery gain)"); ax1.set_ylim(*alpha_ylim)
    ax2.set_ylabel("intercept a\n[km s$^{-1}$]")
    ax3.set_ylabel("empty-substrate\ncentroid [km s$^{-1}$]")
    ax3.set_xticks(x); ax3.set_xticklabels(binlabels, rotation=45, ha="right")
    ax3.set_xlabel(r"radial bin [$R/R_\mathrm{vir}$]")
    ax1.set_xlim(-0.5, nrad - 0.5)
    if untrusted.size:
        ax1.text((untrusted[0] - 0.5 + nrad - 0.5) / 2, alpha_ylim[1],
                 "low S/N – untrusted", ha="center", va="top",
                 color="0.5", fontsize=8.5, style="italic")
    ax1.legend(frameon=False, fontsize=9, ncol=len(labs), loc="lower left")
    ax1.set_title(title or
                  f"Injection-bias summary  (n_noise={result['n_noise']}, "
                  f"peak S/N={result['snr']})", fontsize=12)
    if save_fig:
        plt.savefig("Figure_bias_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2, ax3)


def plot_injection_bias(result, methods=None, figsize=(7.4, 4.8),
                        title=None, save_fig=False):
    """
    Companion 'verdict' view: median |bias| (km/s) per radial bin, one line per
    estimator. Reads off directly which estimator is least biased in the INNER
    bins -- the exact question the clip-vs-no-clip comparison is meant to settle.
    """
    edges = np.round(np.asarray(result["r_edges"]), 3)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if "by_method" in result:
        labs = list(methods) if methods is not None else list(result["methods"])
        series = [(lab, result["by_method"][lab]) for lab in labs]
    else:
        series = [(result.get("method", "bias"), result)]

    fig, ax = plt.subplots(figsize=figsize)
    for lab, d in series:
        mb = np.nanmedian(np.abs(np.asarray(d["bias"])), axis=0)   # (nrad,)
        ax.plot(centers, mb, "o-", lw=1.6, ms=5,
                label=_METHOD_LABELS.get(lab, lab))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel(f"Radial bin center [{result.get('bin_mode', 'radius')}]")
    ax.set_ylabel(r"Median |bias| [km s$^{-1}$]")
    ax.set_title(title or f"Estimator bias vs radius (peak S/N={result.get('snr')})")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8.5)
    if save_fig:
        plt.savefig("Figure_injection_bias.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# 5. SIGNAL / S-N SPECTRUM  (significance everywhere, line strength vs radius)
# =====================================================================
def signal_to_noise_spectrum(
    config: "PipelineConfig",
    stacks: dict,
    method: str | None = None,
    nboot: int = 300,
    line_catalog=DEFAULT_UV_LINES,
    cont_mask_hw: float = 8.0,
    cont_order: int = 3,
    line_hw: float = 3.0,
    robust: bool = True,
    seed: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Build a per-radial-bin S/N spectrum: S/N(λ) = (stack - continuum)/σ, with σ
    the empirical per-pixel bootstrap error (bootstrap_stack_error) and the
    continuum a robust polynomial fit with ALL catalog lines masked out.

    Three products fall out of this:
      * the S/N spectrum itself -> significance of Lya (and any feature) at every
        wavelength, in every radial bin;
      * a continuum-region S/N self-check (std of S/N where there is no line):
        ≈1 means the errors are well-scaled, >1 means they are UNDER-estimated
        (the same diagnostic that catches inv_var over-trusting fiber variances);
      * integrated line strength and significance per catalog line vs radius.

    Parameters
    ----------
    cont_mask_hw : half-width (A) masked around each catalog line before the
                   continuum fit.
    cont_order   : polynomial order of the line-masked continuum over the grid.
    line_hw      : half-width (A) of the integration window for per-line strength.

    Returns dict:
        rest_wave     (nwave,)
        signal        (nrad, nwave)   stack - continuum
        noise         (nrad, nwave)   per-pixel 1σ (bootstrap)
        snr           (nrad, nwave)   signal / noise
        continuum     (nrad, nwave)
        stack         (nrad, nwave)   the input fiducial stack (method)
        snr_cont_std  (nrad,)         std of S/N in line-free pixels (≈1 if errors ok)
        snr_cont_med  (nrad,)         median S/N in line-free pixels (≈0 if continuum ok)
        line_table    dict: names (nl,), waves (nl,), kinds (nl,),
                            flux (nrad, nl)     integrated (signal·dλ) in the window
                            int_snr (nrad, nl)  integrated S/N = Σsig/√Σσ²
                            peak_snr (nrad, nl) max |S/N| in the window
        r_edges, bin_mode, VR_biweight_v, unit_info, method, lines
    """
    if "cube_flux" not in stacks or "cube_err" not in stacks:
        raise KeyError("signal_to_noise_spectrum needs the cube: "
                       "run_stack(config, ..., keep_cube=True).")
    sm = method or "biweight"
    if sm not in stacks["stacks"]:
        raise KeyError(f"method {sm!r} not in stacks (have {list(stacks['stacks'])}).")
    wave = np.asarray(stacks["rest_wave"], dtype=float)
    stack = np.asarray(stacks["stacks"][sm]["flux"], dtype=float)   # (nrad, nwave)
    cube_f = np.asarray(stacks["cube_flux"], dtype=float)
    cube_e = np.asarray(stacks["cube_err"], dtype=float)
    nrad, nwave = stack.shape

    # per-pixel empirical noise (same combine method as the signal stack)
    bse = bootstrap_stack_error(cube_f, cube_e, nboot=nboot, stack_method=sm,
                                robust=robust, seed=seed, verbose=verbose)
    noise = np.asarray(bse["stack_err"], dtype=float)

    # line mask (for continuum fit) and continuum per bin
    line_mask = _line_window_mask(wave, line_catalog, cont_mask_hw)   # True = on a line
    fit_mask = ~line_mask
    cont = np.full_like(stack, np.nan)
    for r in range(nrad):
        cont[r] = _robust_poly(wave, stack[r], fit_mask, order=cont_order)

    signal = stack - cont
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = signal / noise

    # continuum-region self-check
    contreg = fit_mask
    snr_cont_std = np.full(nrad, np.nan)
    snr_cont_med = np.full(nrad, np.nan)
    for r in range(nrad):
        vals = snr[r][contreg & np.isfinite(snr[r])]
        if vals.size:
            snr_cont_std[r] = _robust_std(vals)
            snr_cont_med[r] = np.nanmedian(vals)

    # per-line strength + significance vs radius
    dw = float(np.nanmedian(np.diff(wave)))
    names = [n for n, _, _ in line_catalog]
    waves = np.array([w for _, w, _ in line_catalog])
    kinds = [k for _, _, k in line_catalog]
    nl = len(line_catalog)
    line_flux = np.full((nrad, nl), np.nan)
    int_snr = np.full((nrad, nl), np.nan)
    peak_snr = np.full((nrad, nl), np.nan)
    for li, lam in enumerate(waves):
        win = np.abs(wave - lam) <= line_hw
        if not np.any(win):
            continue
        for r in range(nrad):
            s = signal[r][win]; ns = noise[r][win]
            good = np.isfinite(s) & np.isfinite(ns) & (ns > 0)
            if not np.any(good):
                continue
            line_flux[r, li] = np.nansum(s[good]) * dw
            int_snr[r, li] = np.nansum(s[good]) / np.sqrt(np.nansum(ns[good] ** 2))
            sn = snr[r][win]
            peak_snr[r, li] = np.nanmax(np.abs(sn[np.isfinite(sn)])) if np.any(np.isfinite(sn)) else np.nan

    if verbose:
        edges = np.round(np.asarray(stacks["r_edges"]), 3)
        lya_i = int(np.argmin(np.abs(waves - float(config.LYA_REST))))
        print(f"\nS/N spectrum (method={sm}). Continuum-region S/N std should be ≈1:")
        for r in range(nrad):
            print(f"  {edges[r]:g}-{edges[r+1]:g}: S/N_cont std={snr_cont_std[r]:.2f} "
                  f"med={snr_cont_med[r]:+.2f} | Lyα int S/N={int_snr[r, lya_i]:+.1f}")

    return stamp_provenance({
        "rest_wave": wave,
        "signal": signal, "noise": noise, "snr": snr,
        "continuum": cont, "stack": stack,
        "snr_cont_std": snr_cont_std, "snr_cont_med": snr_cont_med,
        "line_table": {"names": names, "waves": waves, "kinds": kinds,
                       "flux": line_flux, "int_snr": int_snr, "peak_snr": peak_snr},
        "r_edges": np.asarray(stacks["r_edges"]),
        "bin_mode": stacks.get("bin_mode"),
        "VR_biweight_v": stacks.get("VR_biweight_v"),
        "unit_info": stacks.get("unit_info"),
        "method": sm, "lines": line_catalog,
    }, config, "signal_to_noise_spectrum")


_LINE_COLOR = {"em": "tab:green", "abs": "tab:orange"}


def _mark_lines(ax, lines, xlim, label=False, y_for_label=0.92):
    """Vertical guides at catalog lines within xlim; Lya magenta, em green, abs orange."""
    for name, lam, kind in lines:
        if lam < xlim[0] or lam > xlim[1]:
            continue
        col = "magenta" if name.startswith("Lyα") else _LINE_COLOR.get(kind, "0.5")
        ax.axvline(lam, color=col, lw=0.8, ls=":", alpha=0.55)
        if label:
            ax.text(lam, y_for_label, name, transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="right", fontsize=7.5, color=col,
                    alpha=0.9)


def plot_sn_spectrum(
    result, distance_type="virial radii", xlim=None, sn_clip=3.0,
    figsize_width=10, panel_height=2.0, mark=True, title=None,
    savepath=None, dpi=300,
):
    """
    One S/N(λ) panel per radial bin, with ±1σ / ±sn_clip guides and catalog-line
    markers (labelled on the top panel). Shows where Lya -- and anything else --
    is significant, in every bin, on the full rest grid.
    """
    wave = np.asarray(result["rest_wave"])
    snr = np.asarray(result["snr"])
    r_edges = np.round(np.asarray(result["r_edges"]), 3)
    nrad = snr.shape[0]
    if xlim is None:
        xlim = (float(np.nanmin(wave)), float(np.nanmax(wave)))

    fig, axs = plt.subplots(nrad, 1, figsize=(figsize_width, panel_height * nrad),
                            sharex=True, constrained_layout=True)
    if nrad == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.step(wave, snr[i], where="mid", lw=0.9, color="black")
        ax.axhline(0, color="tomato", lw=1, alpha=0.7)
        for s in (1, -1):
            ax.axhline(s, color="0.6", lw=0.7, ls="--", alpha=0.6)
        for s in (sn_clip, -sn_clip):
            ax.axhline(s, color="dodgerblue", lw=0.7, ls=":", alpha=0.6)
        if mark:
            _mark_lines(ax, result["lines"], xlim, label=(i == 0))
        ax.set_ylim(-max(4, sn_clip + 1), max(6, sn_clip + 3))
        lab = f"{r_edges[i]:g}–{r_edges[i+1]:g} {distance_type}"
        ax.text(0.01, 0.86, lab, transform=ax.transAxes, fontsize=10,
                color="dodgerblue",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.85",
                          boxstyle="round,pad=0.25"))
        ax.grid(alpha=0.15, lw=0.6)
    axs[-1].set_xlim(xlim)
    axs[-1].set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=13)
    fig.supylabel("S/N per pixel", fontsize=13)
    fig.suptitle(title or "Per-bin S/N spectrum", fontsize=15, y=1.005)
    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axs


def plot_sn_map(
    result, distance_type="virial radii", xlim=None, sn_max=None,
    cmap="RdBu_r", mark=True, title=None, figsize=(8.5, 5.0),
    savepath=None, dpi=300,
):
    """
    2-D radius–wavelength S/N map: all bins in one image, wavelength on x, radial
    bin on y (equal-height rows), S/N as diverging color. The compact companion
    to plot_sn_spectrum -- read the radial run of every feature at a glance.
    """
    wave = np.asarray(result["rest_wave"])
    snr = np.asarray(result["snr"])
    r_edges = np.round(np.asarray(result["r_edges"]), 3)
    nrad = snr.shape[0]
    if xlim is None:
        xlim = (float(np.nanmin(wave)), float(np.nanmax(wave)))
    if sn_max is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sn_max = np.nanpercentile(np.abs(snr), 98)
        if not np.isfinite(sn_max) or sn_max == 0:
            sn_max = 5.0

    # midpoint edges in wavelength for pcolormesh
    w = wave
    mid = 0.5 * (w[:-1] + w[1:])
    wedges = np.concatenate([[w[0] - (mid[0] - w[0])], mid, [w[-1] + (w[-1] - mid[-1])]])
    row_edges = np.arange(nrad + 1)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    mesh = ax.pcolormesh(wedges, row_edges, snr, cmap=cmap,
                         vmin=-sn_max, vmax=sn_max, shading="flat")
    if mark:
        _mark_lines(ax, result["lines"], xlim, label=True, y_for_label=0.99)
    ax.set_yticks(np.arange(nrad) + 0.5)
    ax.set_yticklabels([f"{r_edges[i]:g}–{r_edges[i+1]:g}" for i in range(nrad)],
                       fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_ylabel(f"Radial bin [{distance_type}]", fontsize=12)
    ax.set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=12)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("S/N per pixel", fontsize=11)
    ax.set_title(title or "Radius–wavelength S/N map", fontsize=14)
    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# CONTINUUM FIT CHECK  (moved here from analysis.py)
# =====================================================================
def plot_continuum_check(
    stacks: dict,
    stack_method: str = "biweight",
    cont_bounds=DEFAULT_CONT_BOUNDS,
    cont_method: str = "poly",
    cont_order: int = 1,
    lya_center: float = LYA_REST,
    xlim: tuple = (1150, 1300),
    panel_height: float = 2.4,
    figwidth: float = 8.5,
    save_fig: bool = False,
    savename: str = "Figure_continuum_check.png",
):
    """
    Full rest-frame spectrum per radial bin with the sideband continuum overlaid.
    Shows whether the continuum fit is anchored on clean regions, whether N V or
    other lines contaminate the right sideband, and whether the window brackets
    the line cleanly. The residual (stack - cont) is drawn in a lighter line so
    you can see the continuum-subtracted shape without switching plots.

    Sideband regions are marked with blue shading; the line window in yellow.

    Lives in validation.py (not analysis.py) because it is a CHECK on the
    continuum model, i.e. a robustness diagnostic, not a science measurement.
    """
    wave   = np.asarray(stacks["rest_wave"])
    fid    = np.asarray(stacks["stacks"][stack_method]["flux"])
    edges  = np.round(np.asarray(stacks["r_edges"]), 3)
    nrad   = fid.shape[0]

    fig, axes = plt.subplots(nrad, 1, figsize=(figwidth, panel_height * nrad),
                             sharex=True)
    if nrad == 1:
        axes = [axes]

    for r, ax in enumerate(axes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cont = get_continuum_model(wave, fid[r], cont_bounds=cont_bounds,
                                       cont_method=cont_method, cont_order=cont_order)
        resid = fid[r] - cont

        # shading
        for (lo, hi) in cont_bounds:
            ax.axvspan(lo, hi, color="steelblue", alpha=0.10, lw=0)
        ax.axvspan(lya_center - 4, lya_center + 4, color="gold", alpha=0.12, lw=0)

        ax.plot(wave, fid[r], color="k", lw=0.9, label="stack" if r == 0 else None)
        ax.plot(wave, cont,   color="tomato", lw=1.0, ls="--",
                label="continuum" if r == 0 else None)
        ax.plot(wave, resid, color="steelblue", lw=0.7, alpha=0.7,
                label="residual" if r == 0 else None)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(lya_center, color="0.6", lw=0.5, ls=":")

        lbl = f"{edges[r]:g}\u2013{edges[r+1]:g} {stacks.get('bin_mode','')}"
        ax.text(0.01, 0.88, lbl, transform=ax.transAxes, fontsize=9,
                color="0.35",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                          boxstyle="round,pad=0.1"))
        ax.set_xlim(xlim)
        ax.grid(alpha=0.10)

        m = (wave >= xlim[0]) & (wave <= xlim[1]) & np.isfinite(fid[r])
        if np.any(m):
            lo_, hi_ = np.nanpercentile(fid[r][m], [2, 98])
            ax.set_ylim(lo_ - 0.2 * abs(hi_ - lo_), hi_ + 0.3 * abs(hi_ - lo_))

    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=12)
    fig.supylabel("Flux", fontsize=11, x=0.01)
    fig.suptitle("Continuum fit diagnostic", fontsize=13, y=1.005)
    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes


# =====================================================================
# 6.  VALIDATION DOSSIER  (fuse the independent tests onto one radius axis)
# =====================================================================
# The individual tests above each answer ONE question about the centroid. The
# dossier collects whatever you have already run and lays them on a shared radius
# axis so "is this bin's centroid real and trustworthy?" is one glance instead of
# four figures. Nothing here recomputes anything -- it consumes existing result
# dicts -- so it is instant and never re-triggers the slow injection test.
#
# The three independent questions, and which test answers each:
#   (1) Is there a line at all?      -> S/N spectrum (Lya integrated S/N) and/or
#                                       placebo flux null. This is the GATE: if it
#                                       fails, the centroid is the centroid of
#                                       noise and nothing else matters.
#   (2) Is the bootstrap bar honest? -> error_cross_check (jackknife vs bootstrap).
#   (3) Is the estimator unbiased    -> injection_recovery. OPTIONAL and treated
#       at this S/N?                    with caution: it is slow, tests only a
#                                       SYMMETRIC injected line (so it cannot probe
#                                       Lya's intrinsic red asymmetry), and is the
#                                       noisiest of the three. A missing injection
#                                       result never blocks a verdict.

# default per-bin thresholds; override via `thresholds=` on either function.
_DOSSIER_DEFAULTS = dict(
    sn_real=3.0,        # Lya integrated S/N at/above this -> line is "real"
    sn_marginal=2.0,    # between marginal and real -> "marginal"
    jk_ratio_warn=1.5,  # jackknife/bootstrap error ratio above this -> bar suspect
    jk_ratio_fail=2.5,  # ... well above this -> bar untrustworthy
    inj_bias_frac=1.0,  # |bias| above this * bootstrap_err -> estimator-bias flag
)


def _bin_centers_edges(r_edges):
    e = np.round(np.asarray(r_edges, dtype=float), 3)
    return e, 0.5 * (e[:-1] + e[1:])


def _lya_int_sn(sn_result, lya_rest):
    """Per-bin Lya integrated S/N from a signal_to_noise_spectrum result."""
    lt = sn_result.get("line_table", {})
    waves = np.asarray(lt.get("waves", []))
    isn = np.asarray(lt.get("int_snr", np.empty((0, 0))))
    if waves.size == 0 or isn.size == 0:
        return None
    li = int(np.argmin(np.abs(waves - lya_rest)))
    return np.abs(isn[:, li])


def summarize_validation(
    real_boot: dict | None = None,
    sn: dict | None = None,
    placebo: dict | None = None,
    error_check: dict | None = None,
    injection: dict | None = None,
    statistic: str = "centroid",
    lya_rest: float = LYA_REST,
    thresholds: dict | None = None,
) -> dict:
    """
    Fuse the independent validation tests into a per-radial-bin PASS / WARN / FAIL
    verdict for the (centroid) measurement. Consumes already-computed result dicts;
    every argument is optional and a missing test is reported as "not tested"
    rather than silently assumed to pass.

    The verdict is built from three independent checks (see module notes):
      gate    : is there a line? (S/N spectrum Lya integrated S/N, else placebo).
                FAIL here forces the bin's overall verdict to FAIL -- a centroid
                with no line under it is meaningless.
      error   : is the bootstrap bar honest? (jackknife/bootstrap ratio).
      bias    : is the estimator unbiased at this S/N? (injection |bias| vs the
                bootstrap error). Treated as advisory: it can only DEMOTE pass->warn,
                never rescue a failing gate, and is skipped (not failed) when absent.

    Parameters
    ----------
    real_boot   : bootstrap_measurements / run_measure summary for the real line
                  (provides centroid_v_fid and the bootstrap 16/84 bar).
    sn          : signal_to_noise_spectrum result (preferred gate).
    placebo     : placebo_wavelengths result (fallback / cross-check gate via the
                  real flux vs the flux null).
    error_check : error_cross_check result (statistic should match).
    injection   : injection_recovery result (optional, advisory only).
    thresholds  : override any of _DOSSIER_DEFAULTS.

    Returns
    -------
    dict with:
        r_edges, centers, bin_mode, statistic
        n_bins
        gate_sn        (nrad,)  Lya integrated S/N per bin (nan if no gate)
        gate_status    list[str] one of {"real","marginal","none","untested"}
        err_ratio      (nrad,)  jackknife/bootstrap (nan if untested)
        err_status     list[str] {"ok","wide","untested"}
        bias_frac      (nrad,)  median|bias| / bootstrap_err (nan if untested)
        bias_status    list[str] {"ok","biased","untested"}
        verdict        list[str] {"pass","warn","fail"} per bin
        notes          list[str] short human-readable reason per bin
        thresholds     the thresholds actually used
    """
    th = dict(_DOSSIER_DEFAULTS)
    if thresholds:
        th.update(thresholds)

    # establish bin geometry / count from whatever we have
    src = real_boot or sn or placebo or error_check
    if src is None or "r_edges" not in src:
        raise ValueError("summarize_validation needs at least one result dict "
                         "carrying 'r_edges' (real_boot / sn / placebo / error_check).")
    r_edges, centers = _bin_centers_edges(src["r_edges"])
    nrad = len(centers)
    bin_mode = src.get("bin_mode")

    # ---- gate: is there a line? ----
    gate_sn = np.full(nrad, np.nan)
    gate_status = ["untested"] * nrad
    sn_vec = _lya_int_sn(sn, lya_rest) if sn is not None else None
    if sn_vec is None and placebo is not None and real_boot is not None:
        # fallback: real flux vs placebo flux-null spread (S/N-like proxy)
        ry, rlo, rhi = _boot_total_flux(real_boot)
        fstd = np.asarray(placebo.get("flux_null_std", np.full(nrad, np.nan)))
        fmed = np.asarray(placebo.get("flux_null_med", np.zeros(nrad)))
        if ry is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                sn_vec = np.abs(np.asarray(ry) - fmed) / np.where(fstd > 0, fstd, np.nan)
    if sn_vec is not None:
        gate_sn = np.asarray(sn_vec, dtype=float)
        for r in range(nrad):
            s = gate_sn[r]
            if not np.isfinite(s):
                gate_status[r] = "untested"
            elif s >= th["sn_real"]:
                gate_status[r] = "real"
            elif s >= th["sn_marginal"]:
                gate_status[r] = "marginal"
            else:
                gate_status[r] = "none"

    # ---- error: is the bootstrap bar honest? ----
    err_ratio = np.full(nrad, np.nan)
    err_status = ["untested"] * nrad
    if error_check is not None and "ratio" in error_check:
        err_ratio = np.asarray(error_check["ratio"], dtype=float)
        for r in range(nrad):
            rr = err_ratio[r]
            if not np.isfinite(rr):
                err_status[r] = "untested"
            elif rr >= th["jk_ratio_fail"]:
                err_status[r] = "wide"      # strongly inflated
            elif rr >= th["jk_ratio_warn"]:
                err_status[r] = "wide"      # mildly inflated (still a warn)
            else:
                err_status[r] = "ok"

    # ---- bias: estimator unbiased at this S/N? (advisory) ----
    bias_frac = np.full(nrad, np.nan)
    bias_status = ["untested"] * nrad
    if injection is not None and real_boot is not None:
        # median |bias| over the tested input velocities, per bin
        if "by_method" in injection:
            lab = injection.get("method", injection["methods"][0])
            bias_arr = np.asarray(injection["by_method"][lab]["bias"])
        else:
            bias_arr = np.asarray(injection.get("bias", np.empty((0, 0))))
        if bias_arr.size:
            med_abs_bias = np.nanmedian(np.abs(bias_arr), axis=0)  # (nrad,)
            boot_err = 0.5 * (np.asarray(real_boot["centroid_v_hi"])
                              - np.asarray(real_boot["centroid_v_lo"]))
            with np.errstate(invalid="ignore", divide="ignore"):
                bias_frac = med_abs_bias / np.where(boot_err > 0, boot_err, np.nan)
            for r in range(nrad):
                bf = bias_frac[r]
                if not np.isfinite(bf):
                    bias_status[r] = "untested"
                elif bf >= th["inj_bias_frac"]:
                    bias_status[r] = "biased"
                else:
                    bias_status[r] = "ok"

    # ---- fuse into a per-bin verdict ----
    verdict, notes = [], []
    for r in range(nrad):
        g, e, b = gate_status[r], err_status[r], bias_status[r]
        reason = []
        # gate dominates
        if g == "none":
            verdict.append("fail"); notes.append("no line (gate S/N below threshold)")
            continue
        if g == "untested":
            reason.append("line presence UNTESTED (no gate)")
            v = "warn"
        else:
            v = "pass"
        if g == "marginal":
            v = "warn"; reason.append("marginal line S/N")
        if e == "wide":
            v = "warn"; reason.append(f"jackknife bar ×{err_ratio[r]:.1f} bootstrap")
        if b == "biased":
            v = "warn"; reason.append(f"estimator bias ≈{bias_frac[r]:.1f}× bar (injection)")
        if not reason:
            reason.append("all available checks pass"
                          if "untested" not in (g, e, b) else "passes available checks")
        verdict.append(v); notes.append("; ".join(reason))

    return {
        "r_edges": r_edges, "centers": centers, "bin_mode": bin_mode,
        "statistic": statistic, "n_bins": nrad,
        "gate_sn": gate_sn, "gate_status": gate_status,
        "err_ratio": err_ratio, "err_status": err_status,
        "bias_frac": bias_frac, "bias_status": bias_status,
        "verdict": verdict, "notes": notes, "thresholds": th,
    }


def print_validation_summary(summary: dict) -> None:
    """Pretty per-bin table of a summarize_validation result."""
    e = summary["r_edges"]
    nrad = summary["n_bins"]
    bm = summary.get("bin_mode", "radius")
    mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    print(f"\nvalidation summary ({summary['statistic']}, bins in {bm}):")
    w = max(len(f"{e[r]:g}-{e[r+1]:g}") for r in range(nrad)) + 1
    print(f"{'bin':>{w}}  {'verdict':>7}  {'gate S/N':>9}  {'JK/boot':>7}  "
          f"{'bias/bar':>8}  notes")
    print("-" * (w + 50))
    for r in range(nrad):
        gsn = summary["gate_sn"][r]
        jk = summary["err_ratio"][r]
        bf = summary["bias_frac"][r]
        gs = f"{gsn:8.1f}" if np.isfinite(gsn) else "      --"
        js = f"{jk:6.2f}" if np.isfinite(jk) else "    --"
        bs = f"{bf:7.2f}" if np.isfinite(bf) else "     --"
        print(f"{f'{e[r]:g}-{e[r+1]:g}':>{w}}  {mark[summary['verdict'][r]]:>7}  "
              f"{gs:>9}  {js:>7}  {bs:>8}  {summary['notes'][r]}")
    nfail = summary["verdict"].count("fail")
    nwarn = summary["verdict"].count("warn")
    print(f"\n  {summary['verdict'].count('pass')} pass, {nwarn} warn, {nfail} fail"
          "   (gate=line presence, JK/boot=error honesty, bias=injection [advisory])")


def centroid_dossier(
    real_boot: dict,
    sn: dict | None = None,
    placebo: dict | None = None,
    error_check: dict | None = None,
    injection: dict | None = None,
    summary: dict | None = None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    centroid_ylims=(-250, 250),
    xlims=None,
    lya_rest: float = LYA_REST,
    thresholds: dict | None = None,
    figsize=(8.4, 9.2),
    title="Centroid validation dossier",
    save_fig: bool = False,
    savename: str = "Figure_centroid_dossier.png",
    verbose: bool = True,
):
    """
    Single figure that fuses the independent centroid checks onto one shared
    radius axis, so "is this bin's centroid real and trustworthy?" is one glance.
    Consumes results you have ALREADY computed -- it does not recompute anything,
    so it is instant and never re-triggers the slow injection-recovery test.

    Three stacked panels (shared radius x-axis) + a verdict strip:
      1. GATE: Lyα integrated S/N vs radius (from `sn`), with the marginal/real
         thresholds drawn. If `sn` is absent but `placebo` is given, a flux-vs-null
         S/N proxy is used. This answers "is there a line under the centroid?".
      2. CENTROID: real centroid_v_fid with its bootstrap 16/84 bar. If
         `error_check` is given, the (wider) jackknife bar is overlaid so you can
         see where the bootstrap under-reports the error. If `injection` is given,
         a shaded ±median|bias| band is drawn around the measured centroid -- shown
         as an ADVISORY caveat (it is the slow, symmetric-line test), never as the
         headline.
      3. VERDICT strip: per-bin pass / warn / fail from summarize_validation.

    Every validation input is optional; the dossier degrades gracefully and labels
    missing tests as "not tested". Only `real_boot` is required.

    Example
    -------
        from utils_lya_halo import validation as val
        boot = analysis.run_measurement_suite(...)          # the real measurement
        sn   = val.signal_to_noise_spectrum(cfg, stacks)     # fast gate
        ec   = val.error_cross_check(cfg, stacks)            # fast error check
        # injection is optional + slow; pass it only if you have it:
        val.centroid_dossier(boot, sn=sn, error_check=ec)    # instant
        val.centroid_dossier(boot, sn=sn, error_check=ec, injection=inj)

    Returns (fig, axes, summary).
    """
    summary = summary or summarize_validation(
        real_boot=real_boot, sn=sn, placebo=placebo, error_check=error_check,
        injection=injection, statistic="centroid", lya_rest=lya_rest,
        thresholds=thresholds)
    th = summary["thresholds"]

    r_edges = np.asarray(summary["r_edges"])
    nrad = summary["n_bins"]
    bm = _resolve_bin_mode(bin_mode, real_boot)
    vr = _get_vr_biweight_v(VR_biweight_v, real_boot)

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 3.2, 0.7], hspace=0.08))
    ax_gate, ax_cen, ax_verdict = axes

    # ---- panel 1: gate (line presence) ----
    # set the radius axis on the BOTTOM panel (verdict) and reuse limits up the stack
    r_mid, _ = _setup_radius_axis(ax_verdict, r_edges, bm, vr, None, vr_ticks, xlims)
    shared_xlim = ax_verdict.get_xlim()
    for ax in (ax_gate, ax_cen):
        ax.set_xscale("log"); ax.set_xlim(shared_xlim)
        ax.grid(True, which="major", lw=0.8, alpha=0.6)
        ax.grid(True, which="minor", lw=0.5, alpha=0.3)

    gate_sn = summary["gate_sn"]
    if np.any(np.isfinite(gate_sn)):
        ax_gate.plot(r_mid, gate_sn, "o-", color="seagreen", lw=1.5, ms=6)
        ax_gate.axhspan(0, th["sn_marginal"], color="0.85", alpha=0.5, lw=0)
        ax_gate.axhline(th["sn_marginal"], color="0.5", ls=":", lw=1)
        ax_gate.axhline(th["sn_real"], color="0.4", ls="--", lw=1,
                        label=f"real (S/N≥{th['sn_real']:g})")
        ax_gate.set_ylabel(r"Ly$\alpha$ integrated S/N")
        ax_gate.legend(frameon=False, fontsize=8.5, loc="upper right")
    else:
        ax_gate.text(0.5, 0.5, "no S/N gate provided\n(pass sn= or placebo=)",
                     ha="center", va="center", transform=ax_gate.transAxes,
                     color="0.5", fontsize=10)
        ax_gate.set_ylabel(r"Ly$\alpha$ S/N")
    ax_gate.set_title(title, fontsize=14)

    # ---- panel 2: centroid (measurement + error honesty + bias caveat) ----
    v = np.asarray(real_boot["centroid_v_fid"])
    vlo = np.asarray(real_boot["centroid_v_lo"]); vhi = np.asarray(real_boot["centroid_v_hi"])

    # advisory injection-bias band first (drawn underneath)
    if np.any(np.isfinite(summary["bias_frac"])):
        boot_err = 0.5 * (vhi - vlo)
        bias_kms = summary["bias_frac"] * boot_err
        ax_cen.fill_between(r_mid, v - bias_kms, v + bias_kms, color="orange",
                            alpha=0.15, lw=0,
                            label="injection |bias| (advisory)")

    # jackknife bar overlaid (wider where bootstrap under-reports)
    if error_check is not None and error_check.get("statistic") == "centroid":
        jk = np.asarray(error_check["jk_block_err"])
        ax_cen.errorbar(r_mid * 1.02, v, yerr=jk, fmt="none", ecolor="0.55",
                        capsize=4, lw=1.1, alpha=0.9, label="jackknife bar")

    # the real measurement with its bootstrap bar (on top)
    ax_cen.errorbar(r_mid, v, yerr=np.vstack([v - vlo, vhi - v]), fmt="o-",
                    color="tab:blue", capsize=3, ms=6, lw=1.4,
                    label="real centroid ± bootstrap")
    ax_cen.axhline(0, color="tomato", alpha=0.5, lw=1)
    ax_cen.set_ylim(centroid_ylims)
    ax_cen.set_ylabel(r"Ly$\alpha$ centroid velocity [km s$^{-1}$]")
    ax_cen.legend(frameon=False, fontsize=8.5, loc="best")

    # ---- panel 3: verdict strip ----
    cmap = {"pass": "#2e7d32", "warn": "#f9a825", "fail": "#c62828"}
    bins, _, _ = _radius_points_safe(r_edges)
    for r in range(nrad):
        lo = bins[r] if bins is not None else r_mid[r] * 0.85
        hi = bins[r + 1] if bins is not None else r_mid[r] * 1.15
        ax_verdict.axvspan(lo, hi, color=cmap[summary["verdict"][r]], alpha=0.75, lw=0)
        ax_verdict.text(r_mid[r], 0.5, summary["verdict"][r][0].upper(),
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white", transform=ax_verdict.get_xaxis_transform())
    ax_verdict.set_yticks([])
    ax_verdict.set_ylabel("verdict", fontsize=10)

    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()

    if verbose:
        print_validation_summary(summary)
    return fig, axes, summary


def _radius_points_safe(r_edges):
    """Edges/mids tolerant of non-monotonic or single-bin inputs (for shading)."""
    try:
        return _radius_points(r_edges)
    except Exception:
        e = np.asarray(r_edges, dtype=float)
        mids = 0.5 * (e[:-1] + e[1:])
        return e, mids, None


# =====================================================================
# 6. NULL-ENSEMBLE SYNTHESIS  (make the nulls less of a black box)
# =====================================================================
# Five tools that turn the existing nulls into VISUAL diagnostics:
#
#   zswap_spectra          -- run the z-scramble loop ONCE, keep the per-bin
#                             stack-spectrum envelope (not just integrated flux),
#                             so you can SEE the null cloud the real line sits in.
#   plot_null_envelope     -- the intuition plot: real rest-frame stack on top of
#                             the z-scramble null envelope, per bin (small mult.).
#   collect_null_ensembles -- gather every null (z-scramble, placebo-λ, bootstrap,
#                             + any extra you pass, e.g. random-position) and the
#                             real Lyα flux into ONE structure, each null carrying
#                             an actual SAMPLE (so σ is derived, never asserted).
#   plot_null_synthesis    -- the verdict plot: integrated Lyα flux vs radius with
#                             each null as a ±Nσ band about its own median, the
#                             real flux + bootstrap bar overplotted, and the
#                             binding null S/N annotated per bin.
#   plot_error_calibration -- the calibration strip: per-bin σ from every method
#                             side by side (bootstrap's optimism made visible).
#   windowed_flux_scan     -- slide the line window across the WHOLE rest grid
#                             (lines NOT avoided), keep the SIGN, so SiII/Lyα/etc.
#                             show up as bumps against the noise floor.
#   plot_windowed_scan     -- draw that scan with catalog lines marked.
#
# Design rule: every null returns a SAMPLE of null fluxes, so the plots show the
# distribution behind each number and you can always go look at the cloud.

_NULL_COLORS = {
    "z-scramble":  "tab:purple",
    "placebo (λ)": "tab:green",
    "random-pos":  "tab:brown",
    "_default":    ["tab:olive", "tab:cyan", "tab:pink", "tab:gray"],
}


def _null_color(label, i):
    return _NULL_COLORS.get(label, _NULL_COLORS["_default"][i % 4])


def _jackknife_flux_sigma(flux, err, wave, bounds, cont_kw, stack_method,
                          mode="block", n_blocks=20, seed=1, progress=True,
                          desc="jackknife"):
    """Per-bin jackknife sigma of the integrated line flux (flux_sum), on the
    per-galaxy cube: delete galaxy subsets, re-stack, re-measure. See module
    notes for mode/scaling. Returns (central, sigma, theta)."""
    flux = np.asarray(flux, float)
    err = None if err is None else np.asarray(err, float)
    ngal, nrad, _ = flux.shape

    full_stack, _ = stack_galaxies(flux, err, method=stack_method)
    central = _stat_per_stack(full_stack, wave, bounds, None, cont_kw, "flux")

    if mode == "delete-one":
        groups = [np.array([i]) for i in range(ngal)]
    else:
        rng = np.random.default_rng(seed)
        groups = np.array_split(rng.permutation(ngal), n_blocks)

    K = len(groups)
    theta = np.full((K, nrad), np.nan)
    it = enumerate(groups)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=K, desc=desc)
        except Exception:
            pass
    for i, g in it:
        keep = np.ones(ngal, dtype=bool); keep[g] = False
        st, _ = stack_galaxies(flux[keep], None if err is None else err[keep],
                               method=stack_method)
        theta[i] = _stat_per_stack(st, wave, bounds, None, cont_kw, "flux")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(theta, axis=0)
        sigma = np.sqrt((K - 1) / K * np.nansum((theta - mean) ** 2, axis=0))
    return central, sigma, theta


def zswap_spectra(
    config: "PipelineConfig",
    product: GalaxyProduct,
    n_realizations: int = 200,
    method: str | None = None,
    bounds=None,
    seed: int = 0,
    keep_all: bool = False,
    verbose: bool = True,
) -> dict:
    """
    z-scramble null, but KEEP the per-bin stack SPECTRUM (not only the integrated
    flux). Runs the same scramble loop as redshift_scramble_null -- permute z,
    re-stack -- n_realizations times, and accumulates, per radial bin, the
    envelope of the null stack spectra (16/84 and 2.5/97.5 across realizations).
    This is what plot_null_envelope draws so you can SEE whether the real Lyα
    bump poke out of the null cloud, and whether comparable bumps live elsewhere
    in the same panel.

    The integrated-flux null it also returns (null_F_all, null_med, null_std,
    snr) uses the SAME convention as redshift_scramble_null, so it drops straight
    into collect_null_ensembles.

    Memory: holds an (n_realizations, nrad, nwave) cube while it runs; for
    n~200, nrad~6, nwave~400 that is a few MB. Set keep_all=True to also return
    the raw cube (off by default).

    Example
    -------
        zs = val.zswap_spectra(cfg, product, n_realizations=200)
        val.plot_null_envelope(zs, bins="all")          # intuition plot
        # outer bins only, lightly smoothed:
        val.plot_null_envelope(zs, bins=[4, 5], smooth=1.0)

    Returns dict: rest_wave, real_stack (nrad,nwave), spec_med/lo/hi/lo2/hi2
        (nrad,nwave), real_F (nrad,), null_F_all (n,nrad) [if keep_all],
        null_med/null_std/snr (nrad,), r_edges, bin_mode, VR_biweight_v,
        unit_info, bounds, method, n_realizations.
    """
    method = method or "biweight"
    bounds = tuple(config.line_window) if bounds is None else tuple(bounds)
    cont_kw = _continuum_kwargs(config)

    real = build_stacks(config, product, keep_cube=False)
    rest_wave = np.asarray(real["rest_wave"], dtype=float)
    if method not in real["stacks"]:
        raise KeyError(f"method {method!r} not in stacks (have {list(real['stacks'])}).")
    real_stack = np.asarray(real["stacks"][method]["flux"], dtype=float)   # (nrad, nwave)
    nrad, nwave = real_stack.shape
    real_F = _integrated_flux_per_bin(real_stack, rest_wave, bounds, cont_kw)

    rng = np.random.default_rng(seed)
    spectra = np.full((n_realizations, nrad, nwave), np.nan)
    null_F = np.full((n_realizations, nrad), np.nan)
    it = range(n_realizations)
    if verbose:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="z-scramble (spectra)")
        except Exception:
            pass
    for k in it:
        sp = _scramble_product(product, config.z_col, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = build_stacks(config, sp, keep_cube=False)
        wv = np.asarray(st["rest_wave"], dtype=float)
        sf = np.asarray(st["stacks"][method]["flux"], dtype=float)
        if sf.shape == real_stack.shape and np.allclose(wv, rest_wave, equal_nan=True):
            spectra[k] = sf
        else:                                   # grid drift safety net (shouldn't fire)
            spectra[k] = np.vstack([np.interp(rest_wave, wv, sf[r]) for r in range(nrad)])
        null_F[k] = _integrated_flux_per_bin(sf, wv, bounds, cont_kw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spec_med = np.nanmedian(spectra, axis=0)
        spec_lo, spec_hi = np.nanpercentile(spectra, [16, 84], axis=0)
        spec_lo2, spec_hi2 = np.nanpercentile(spectra, [2.5, 97.5], axis=0)
        null_med = np.nanmedian(null_F, axis=0)
        null_std = np.array([_robust_std(null_F[:, r]) for r in range(nrad)])
    snr = (real_F - null_med) / np.where(null_std > 0, null_std, np.nan)

    if verbose:
        edges = np.round(np.asarray(real["r_edges"]), 3)
        print(f"\nz-scramble spectra ({n_realizations} realizations, method={method}):")
        for r in range(nrad):
            print(f"  {edges[r]:g}-{edges[r+1]:g}: real F={real_F[r]:+.3g}  "
                  f"null σ={null_std[r]:.3g}  S/N={snr[r]:+.1f}")

    out = {
        "rest_wave": rest_wave, "real_stack": real_stack,
        "spec_med": spec_med, "spec_lo": spec_lo, "spec_hi": spec_hi,
        "spec_lo2": spec_lo2, "spec_hi2": spec_hi2,
        "real_F": real_F, "null_med": null_med, "null_std": null_std, "snr": snr,
        "null_F_all": (null_F if keep_all else None),
        "r_edges": np.asarray(real["r_edges"]),
        "bin_mode": real.get("bin_mode"), "VR_biweight_v": real.get("VR_biweight_v"),
        "unit_info": real.get("unit_info"), "bounds": bounds,
        "method": method, "n_realizations": n_realizations,
    }
    # null_F_all is needed by collect_null_ensembles even when keep_all=False
    out["_null_F_all"] = null_F
    return stamp_provenance(out, config, "zswap_spectra")


def plot_null_envelope(
    zs: dict, bins="all", smooth=None, xlim=None, lya_center: float = LYA_REST,
    mark: bool = True, ncols: int = 2, figsize=None, title=None,
    save_fig: bool = False, savename: str = "Figure_null_envelope.png",
):
    """
    THE INTUITION PLOT. Per radial bin: the real rest-frame stack spectrum (blue
    step) on top of the z-scramble null envelope (grey 16/84 + lighter 2.5/97.5),
    with catalog lines marked. If the real Lyα bump clears the grey cloud and no
    comparable excursion sits elsewhere in the panel, you believe it; if the
    outer-bin bump is just one grey-height wiggle among several, you do not.

    zs : a zswap_spectra result, OR a collect_null_ensembles result (its
         ["zswap_spectra"] is used automatically).
    bins : "all" or a list of bin indices (e.g. [4, 5] for the outer bins).
    smooth : Gaussian sigma (pixels) applied to BOTH the real curve and the
             envelope for display only (None = raw).

    Example
    -------
        zs = val.zswap_spectra(cfg, product, n_realizations=200)
        val.plot_null_envelope(zs)                       # every bin
        val.plot_null_envelope(zs, bins=[4, 5], smooth=1.0, xlim=(1180, 1260))
    """
    if isinstance(zs, dict) and "zswap_spectra" in zs and zs["zswap_spectra"] is not None:
        zs = zs["zswap_spectra"]
    wave = np.asarray(zs["rest_wave"], dtype=float)
    real = np.asarray(zs["real_stack"], dtype=float)
    nrad = real.shape[0]
    bin_list = list(range(nrad)) if (isinstance(bins, str) and bins == "all") \
        else list(np.atleast_1d(bins))
    edges = np.round(np.asarray(zs["r_edges"]), 3)

    sm_fn = None
    if smooth is not None:
        from scipy.ndimage import gaussian_filter1d
        sm_fn = lambda y: gaussian_filter1d(y, smooth)

    nrows = int(np.ceil(len(bin_list) / ncols))
    if figsize is None:
        figsize = (6.6 * ncols, 2.3 * nrows + 0.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             constrained_layout=True)
    axes_flat = axes.ravel()

    for ax_i, rb in enumerate(bin_list):
        ax = axes_flat[ax_i]
        lo2, hi2 = zs["spec_lo2"][rb], zs["spec_hi2"][rb]
        lo, hi = zs["spec_lo"][rb], zs["spec_hi"][rb]
        y = real[rb]
        if sm_fn is not None:
            lo2, hi2, lo, hi, y = map(sm_fn, (lo2, hi2, lo, hi, y))
        ax.fill_between(wave, lo2, hi2, color="0.6", alpha=0.22, lw=0,
                        label="null 2.5–97.5%")
        ax.fill_between(wave, lo, hi, color="0.45", alpha=0.35, lw=0,
                        label="null 16–84%")
        ax.step(wave, y, where="mid", color="tab:blue", lw=1.2, label="real stack")
        ax.axvline(lya_center, color="magenta", lw=0.8, alpha=0.7)
        ax.axhline(0, color="tomato", lw=0.9, alpha=0.7)
        if xlim is not None:
            ax.set_xlim(xlim)
        if mark:
            _mark_lines(ax, DEFAULT_UV_LINES, ax.get_xlim(),
                        label=(ax_i == 0), y_for_label=0.98)
        ax.set_title(f"{edges[rb]:g}–{edges[rb+1]:g}  "
                     f"(S/N={zs['snr'][rb]:+.1f})", fontsize=10)
        ax.grid(alpha=0.15, lw=0.6)
        if ax_i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    for j in range(len(bin_list), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or "Real stack vs. z-scramble null envelope", fontsize=13)
    fig.supxlabel(r"Rest-frame wavelength [$\AA$]", fontsize=11)
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes


def collect_null_ensembles(
    config: "PipelineConfig",
    stacks: dict,
    product: GalaxyProduct,
    *,
    methods=("zswap", "bootstrap", "jackknife"),
    n_zswap: int = 200,
    nboot: int = 300,
    jk_mode: str = "block",
    jk_n_blocks: int = 40,
    jk_n_reps: int = 5,
    placebo_step: float = 5.0,
    method: str | None = None,
    seed: int = 1,
    extra_nulls=None,
    keep_zswap_spectra: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Gather the noise estimators into ONE structure, per radial bin, for the
    integrated Lyα flux. Three DIFFERENT kinds of object (don't conflate them):

      * z-scramble (null)   -> in nulls["nulls"]; a band about ~0. The FLOOR the
                               real flux must clear. Destroys the line, rebuilds
                               the stack, so it carries galaxy-sampling AND coadd
                               structure. This is the only true null here.
      * bootstrap (error)   -> nulls["bootstrap"]; error bar ON the real point
                               (resample galaxies w/ replacement). Needs the cube.
      * jackknife (error)   -> nulls["jackknife"]; error bar ON the real point
                               (delete galaxy subsets, re-stack). Needs the cube.
                               Should agree with bootstrap if the bar is honest.

    methods : subset of {"zswap","bootstrap","jackknife","placebo"} (placebo is
              the old wavelength floor, demoted to opt-in).
    jk_mode : "block" (default; jk_n_blocks deletions) or "delete-one".

    Feed to plot_null_synthesis (real vs floor, both error bars) and
    plot_error_calibration (all three σ side by side). plot_null_envelope reads
    the z-scramble spectra stashed here.

    Returns dict: real_F, nulls {label:{sample,med,sigma}}, bootstrap {central,
        sigma}|None, jackknife {central,sigma,mode,n_blocks}|None,
        zswap_spectra|None, snr {label:(nrad,)}, r_edges, bin_mode,
        VR_biweight_v, unit_info, bounds, method.
    """
    sm = method or "biweight"
    bounds = tuple(config.line_window)
    cont_kw = _continuum_kwargs(config)
    rest_wave = np.asarray(stacks["rest_wave"], dtype=float)
    if sm not in stacks["stacks"]:
        raise KeyError(f"method {sm!r} not in stacks (have {list(stacks['stacks'])}).")
    real_stack = np.asarray(stacks["stacks"][sm]["flux"], dtype=float)
    real_F = _integrated_flux_per_bin(real_stack, rest_wave, bounds, cont_kw)
    nrad = real_F.size
    have_cube = ("cube_flux" in stacks and "cube_err" in stacks)

    nulls: dict = {}
    zs = None
    if "zswap" in methods:
        zs = zswap_spectra(config, product, n_realizations=n_zswap, method=sm,
                           bounds=bounds, seed=seed, keep_all=False, verbose=verbose)
        nulls["z-scramble"] = {"sample": zs["_null_F_all"],
                               "med": zs["null_med"], "sigma": zs["null_std"]}
    if "placebo" in methods:
        pb = placebo_wavelengths(config, stacks, method=sm, step=placebo_step,
                                 verbose=verbose)
        nulls["placebo (λ)"] = {"sample": pb["flux_offset"],
                                "med": pb["flux_null_med"], "sigma": pb["flux_null_std"]}

    boot = None
    if "bootstrap" in methods:
        if have_cube:
            b = bootstrap_measurements(
                np.asarray(stacks["cube_flux"], dtype=float),
                np.asarray(stacks["cube_err"], dtype=float),
                rest_wave, nboot=nboot, bounds=bounds,
                lya_center=float(config.LYA_REST), stack_method=sm,
                compute_side_ratio=True, seed=seed, verbose=verbose, **cont_kw)
            be = 0.5 * (np.asarray(b["total_flux_hi"]) - np.asarray(b["total_flux_lo"]))
            boot = {"central": np.asarray(b["total_flux_fid"]), "sigma": be}
        elif verbose:
            print("bootstrap skipped: stacks has no cube (rebuild keep_cube=True).")

    jack = None
    if "jackknife" in methods:
        if have_cube:
            cf = np.asarray(stacks["cube_flux"], dtype=float)
            ce = np.asarray(stacks["cube_err"], dtype=float)
            ngal = cf.shape[0]
            if jk_mode == "delete-one" and ngal > 800 and verbose:
                print(f"jackknife delete-one on {ngal} galaxies is "
                      f"{ngal} re-stacks; consider jk_mode='block'.")
            central, sigma, sigma_spread = _jackknife_flux_sigma_repeated(
                cf, ce, rest_wave, bounds, cont_kw, sm,
                mode=jk_mode, n_blocks=jk_n_blocks, n_reps=jk_n_reps, seed=seed)
            jack = {"central": central, "sigma": sigma,
                    "sigma_spread": sigma_spread, "mode": jk_mode,
                    "n_blocks": (cf.shape[0] if jk_mode == "delete-one" else jk_n_blocks),
                    "n_reps": (1 if jk_mode == "delete-one" else jk_n_reps)}
        elif verbose:
            print("jackknife skipped: stacks has no cube (rebuild keep_cube=True).")

    if extra_nulls:
        for lab, samp in extra_nulls.items():
            samp = np.asarray(samp, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(samp, axis=0)
            sig = np.array([_robust_std(samp[:, r]) for r in range(nrad)])
            nulls[lab] = {"sample": samp, "med": med, "sigma": sig}

    snr = {lab: (real_F - d["med"]) / np.where(d["sigma"] > 0, d["sigma"], np.nan)
           for lab, d in nulls.items()}

    if verbose:
        edges = np.round(np.asarray(stacks["r_edges"]), 3)
        labs = list(nulls)
        print(f"\nnull/error ensemble (real Lyα flux):")
        head = f"{'bin':>14}  " + "  ".join(f"{l[:11]:>11}" for l in labs)
        if boot is not None: head += f"  {'σ_boot':>10}"
        if jack is not None: head += f"  {'σ_jack':>10}"
        print(head)
        for r in range(nrad):
            row = f"{edges[r]:g}-{edges[r+1]:g}".rjust(14)
            row += "  " + "  ".join(f"{snr[l][r]:>+11.1f}" for l in labs)
            if boot is not None: row += f"  {boot['sigma'][r]:>10.3g}"
            if jack is not None: row += f"  {jack['sigma'][r]:>10.3g}"
            print(row)
        print("  (null S/N = real vs floor; σ_boot/σ_jack are error bars on the "
              "real point and should agree)")

    out = {
        "real_F": real_F, "nulls": nulls, "bootstrap": boot, "jackknife": jack,
        "zswap_spectra": (zs if keep_zswap_spectra else None),
        "snr": snr,
        "r_edges": np.asarray(stacks["r_edges"]),
        "bin_mode": stacks.get("bin_mode"),
        "VR_biweight_v": stacks.get("VR_biweight_v"),
        "unit_info": stacks.get("unit_info"),
        "bounds": bounds, "method": sm,
    }
    return stamp_provenance(out, config, "collect_null_ensembles")


def plot_null_synthesis(
    nulls: dict, nsigma: float = 2.0, symlog: bool = True, linthresh=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(8.2, 5.2), xlims=None,
    ylims=None, title=None, annotate: bool = True,
    save_fig: bool = False, savename: str = "Figure_null_synthesis.png",
    verbose: bool = True,
):
    """
    THE VERDICT PLOT. Integrated Lyα flux vs radius, with EACH null drawn as a
    ±nsigma band about its own median (so you see the spread the real flux must
    clear), the real flux + bootstrap bar overplotted, and the BINDING null S/N
    (the smallest across nulls) annotated per bin. Where the bootstrap bar is
    much tighter than the null bands, the bootstrap's optimism is visible at a
    glance.

    nsigma : band half-width in σ (default 2; 3 not needed).
    symlog : symmetric-log y so the orders-of-magnitude-brighter inner bins AND
             the near-zero outer bins are both visible WITH their sign kept
             (negatives survive, unlike a plain log axis).

    Example
    -------
        nulls = val.collect_null_ensembles(cfg, stacks, product)
        val.plot_null_synthesis(nulls, nsigma=2)
    """
    r_edges = np.asarray(nulls["r_edges"])
    real_F = np.asarray(nulls["real_F"], dtype=float)
    nrad = real_F.size
    bm = _resolve_bin_mode(None, nulls)
    vr = _get_vr_biweight_v(None, nulls)
    y_unit = (nulls.get("unit_info") or {}).get("y_unit", "")

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, _ = _setup_radius_axis(ax, r_edges, bm, vr, None, vr_ticks, xlims)

    # null bands
    for i, (label, d) in enumerate(nulls["nulls"].items()):
        med = np.asarray(d["med"], dtype=float)
        sig = np.asarray(d["sigma"], dtype=float)
        c = _null_color(label, i)
        ax.fill_between(r_mid, med - nsigma * sig, med + nsigma * sig,
                        step="mid", color=c, alpha=0.18, lw=0,
                        label=f"{label} (±{nsigma:g}σ)")
        ax.plot(r_mid, med, color=c, lw=0.8, alpha=0.5, drawstyle="steps-mid")

    # real flux with error bars: bootstrap (primary) + jackknife (overlay).
    # Both are errors ON the real point, not floors -- only the z-scramble band
    # above is a null. Two bars here = the bootstrap-vs-jackknife agreement check.
    boot = nulls.get("bootstrap")
    jack = nulls.get("jackknife")
    if boot is not None:
        ax.errorbar(r_mid, real_F, yerr=nsigma * np.asarray(boot["sigma"], dtype=float),
                    fmt="o-", color="black", capsize=3, ms=6, lw=1.5,
                    label=f"real Lyα ±{nsigma:g}σ$_{{boot}}$", zorder=6)
    else:
        ax.plot(r_mid, real_F, "o-", color="black", ms=6, lw=1.5,
                label="real Lyα flux", zorder=6)
    if jack is not None:
        ax.errorbar(r_mid * 1.02, real_F,
                    yerr=nsigma * np.asarray(jack["sigma"], dtype=float),
                    fmt="none", ecolor="0.45", capsize=4, lw=1.2, alpha=0.9,
                    label=f"±{nsigma:g}σ$_{{jack}}$", zorder=5)

    ax.axhline(0, color="0.5", lw=0.9, ls=":")
    if symlog:
        if linthresh is None:
            maxsig = np.nanmax([np.nanmax(d["sigma"]) for d in nulls["nulls"].values()]) \
                if nulls["nulls"] else 1.0
            linthresh = float(maxsig) if np.isfinite(maxsig) and maxsig > 0 else 1.0
        ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    if ylims is not None:
        ax.set_ylim(ylims)

    # binding-null S/N annotation per bin
    if annotate and nulls["nulls"]:
        snr = nulls["snr"]
        labs = list(nulls["nulls"])
        for r in range(nrad):
            vals = [snr[l][r] for l in labs if np.isfinite(snr[l][r])]
            if not vals:
                continue
            binding = min(vals)
            ax.annotate(f"{binding:+.2f}σ", (r_mid[r], real_F[r]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color="0.25")

    ax.set_ylabel(f"Integrated Lyα flux [{y_unit}]" if y_unit else "Integrated Lyα flux")
    ax.set_title(title or f"Lyα flux vs. all nulls (binding null S/N annotated)")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.grid(alpha=0.15, which="both")
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()

    if verbose and nulls["nulls"]:
        snr = nulls["snr"]; labs = list(nulls["nulls"])
        edges = np.round(r_edges, 3)
        print("binding null per bin (smallest S/N):")
        for r in range(nrad):
            vals = {l: snr[l][r] for l in labs if np.isfinite(snr[l][r])}
            if not vals:
                print(f"  {edges[r]:g}-{edges[r+1]:g}: (no finite null)")
                continue
            l_min = min(vals, key=vals.get)
            print(f"  {edges[r]:g}-{edges[r+1]:g}: {l_min} at {vals[l_min]:+.1f}σ")
    return fig, ax


def plot_error_calibration(
    nulls: dict, vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(8.2, 4.6),
    logy: bool = True, title=None, save_fig: bool = False,
    savename: str = "Figure_error_calibration.png",
):
    """
    THE CALIBRATION STRIP. Per radial bin, the noise σ from every method side by
    side as grouped bars: each null's spread AND the bootstrap error. If the
    bootstrap bar is consistently the shortest, its optimism is right there in
    the figure -- the nulls (which retain coherent sky/continuum/coadd structure)
    sit higher because they measure a fuller noise budget.

    Example
    -------
        nulls = val.collect_null_ensembles(cfg, stacks, product)
        val.plot_error_calibration(nulls)
    """
    r_edges = np.round(np.asarray(nulls["r_edges"]), 3)
    nrad = nulls["real_F"].size
    labels = list(nulls["nulls"].keys())
    sigma_by = {lab: np.asarray(nulls["nulls"][lab]["sigma"], dtype=float) for lab in labels}
    for extra in ("bootstrap", "jackknife"):
        if nulls.get(extra) is not None:
            labels = labels + [extra]
            sigma_by[extra] = np.asarray(nulls[extra]["sigma"], dtype=float)

    x = np.arange(nrad)
    nb = len(labels)
    width = 0.8 / max(nb, 1)
    fig, ax = plt.subplots(figsize=figsize)
    for i, lab in enumerate(labels):
        c = {"bootstrap": "black", "jackknife": "0.4"}.get(lab, _null_color(lab, i))
        ax.bar(x + (i - (nb - 1) / 2) * width, sigma_by[lab], width=width, color=c, alpha=0.85, label=lab)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r_edges[r]:g}–{r_edges[r+1]:g}" for r in range(nrad)],
                       rotation=0, fontsize=9)
    ax.set_xlabel(f"Radial bin [{nulls.get('bin_mode', 'radius')}]")
    ax.set_ylabel("Integrated-flux noise σ (per method)")
    ax.set_title(title or "Per-bin error calibration (bootstrap vs nulls)")
    ax.legend(frameon=False, fontsize=8.5, ncol=min(nb, 4))
    ax.grid(alpha=0.2, axis="y", which="both")
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def windowed_flux_scan(
    config: "PipelineConfig", stacks: dict, method: str | None = None,
    half_window: float | None = None, step: float = 1.0, pad: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Slide the line window across the WHOLE rest grid (lines are NOT avoided) and
    record the SIGNED continuum-subtracted windowed flux per radial bin at every
    center. Unlike placebo_wavelengths (which steps around line-free centers to
    build a null), this is meant to SHOW the lines: in the inner bins SiII, Lyα,
    etc. appear as bumps -- emission positive, absorption negative -- standing
    out of the noise floor; in the outer bins they fade into it. Signs are kept
    deliberately (no abs), so an absorption trough reads as a negative dip.

    The window and its sidebands track each center (identical geometry to the real
    Lyα measurement), so the curve is directly the same statistic you integrate
    for Lyα, scanned in wavelength.

    Example
    -------
        scan = val.windowed_flux_scan(cfg, stacks, step=1.0)
        val.plot_windowed_scan(scan, bins="all")
        val.plot_windowed_scan(scan, bins=[0, 1])     # inner bins: see SiII etc.

    Returns dict: centers (ncen,), flux_scan (ncen,nrad), v_scan (ncen,nrad),
        r_edges, bin_mode, VR_biweight_v, unit_info, half_window, method, lines.
    """
    sm = method or "biweight"
    lya = float(config.LYA_REST)
    hw = half_window if half_window is not None else 0.5 * (config.line_window[1] - config.line_window[0])
    rest_wave = np.asarray(stacks["rest_wave"], dtype=float)
    if sm not in stacks["stacks"]:
        raise KeyError(f"method {sm!r} not in stacks (have {list(stacks['stacks'])}).")
    flux = np.asarray(stacks["stacks"][sm]["flux"], dtype=float)   # (nrad, nwave)
    nrad = flux.shape[0]

    cont_method = getattr(config, "cont_method", "poly")
    cont_order = getattr(config, "cont_order", 1)
    cb0 = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
    offs = [(lo - lya, hi - lya) for lo, hi in cb0]
    foot_lo = min([o[0] for o in offs] + [-hw])
    foot_hi = max([o[1] for o in offs] + [hw])

    lo = float(np.nanmin(rest_wave)) - foot_lo + pad
    hi = float(np.nanmax(rest_wave)) - foot_hi - pad
    centers = np.arange(np.ceil(lo), np.floor(hi) + 1, step)
    if centers.size == 0:
        raise ValueError("scan grid empty; reduce pad or widen the rest grid.")

    f_scan = np.full((centers.size, nrad), np.nan)
    v_scan = np.full((centers.size, nrad), np.nan)
    for ci, c in enumerate(centers):
        bounds_c = (c - hw, c + hw)
        cont_c = tuple((c + o[0], c + o[1]) for o in offs)
        for r in range(nrad):
            fr = integrated_line_flux(rest_wave, flux[r], bounds=bounds_c,
                                      cont_bounds=cont_c, lya_center=c,
                                      cont_method=cont_method, cont_order=cont_order)
            f_scan[ci, r] = fr["flux_sum"]
            cr = flux_weighted_centroid(rest_wave, flux[r], bounds=bounds_c,
                                        cont_bounds=cont_c, lya_center=c,
                                        cont_method=cont_method, cont_order=cont_order)
            v_scan[ci, r] = cr["centroid_vel_kms"]

    if verbose:
        print(f"windowed flux scan: {centers.size} centers "
              f"({centers[0]:.0f}–{centers[-1]:.0f} Å, step {step:g}), "
              f"window ±{hw:.1f} Å, method={sm}")

    return stamp_provenance({
        "centers": centers, "flux_scan": f_scan, "v_scan": v_scan,
        "r_edges": np.asarray(stacks["r_edges"]),
        "bin_mode": stacks.get("bin_mode"),
        "VR_biweight_v": stacks.get("VR_biweight_v"),
        "unit_info": stacks.get("unit_info"),
        "half_window": hw, "method": sm, "lines": DEFAULT_UV_LINES,
    }, config, "windowed_flux_scan")


def plot_windowed_scan(
    scan: dict, bins="all", ncols: int = 1, xlim=None, show_noise_band: bool = True,
    avoid_hw: float = 10.0, lya_center: float = LYA_REST, mark: bool = True,
    figsize=None, title=None, save_fig: bool = False,
    savename: str = "Figure_windowed_scan.png",
):
    """
    Draw a windowed_flux_scan: signed windowed flux vs window-center wavelength,
    one panel per radial bin, catalog lines marked (Lyα magenta, emission green,
    absorption orange). A ±1σ noise band from the line-free part of the scan is
    shaded so real lines (bumps clearing the band) separate from noise. Keeping
    the sign means emission lines point up and absorption lines point down -- so
    the inner-bin SiII/CII absorption troughs are visible as negative dips, which
    is the point.

    Example
    -------
        scan = val.windowed_flux_scan(cfg, stacks)
        val.plot_windowed_scan(scan)                 # all bins stacked
        val.plot_windowed_scan(scan, bins=[0, 1], xlim=(1230, 1360))
    """
    centers = np.asarray(scan["centers"], dtype=float)
    F = np.asarray(scan["flux_scan"], dtype=float)
    nrad = F.shape[1]
    bin_list = list(range(nrad)) if (isinstance(bins, str) and bins == "all") \
        else list(np.atleast_1d(bins))
    edges = np.round(np.asarray(scan["r_edges"]), 3)
    lines = scan.get("lines", DEFAULT_UV_LINES)

    line_free = ~_line_window_mask(centers, lines, avoid_hw)

    nrows = int(np.ceil(len(bin_list) / ncols))
    if figsize is None:
        figsize = (8.4 * ncols, 1.9 * nrows + 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, constrained_layout=True)
    axes_flat = axes.ravel()

    for ax_i, rb in enumerate(bin_list):
        ax = axes_flat[ax_i]
        y = F[:, rb]
        if show_noise_band:
            vals = y[line_free & np.isfinite(y)]
            if vals.size > 3:
                s = _robust_std(vals)
                ax.axhspan(-s, s, color="0.6", alpha=0.18, lw=0,
                           label="±1σ (line-free)")
        ax.axhline(0, color="tomato", lw=0.9, alpha=0.7)
        ax.plot(centers, y, "-", color="tab:blue", lw=1.1)
        if xlim is not None:
            ax.set_xlim(xlim)
        if mark:
            _mark_lines(ax, lines, ax.get_xlim(), label=(ax_i == 0), y_for_label=0.98)
        ax.set_ylabel(f"{edges[rb]:g}–{edges[rb+1]:g}", fontsize=9)
        ax.grid(alpha=0.15, lw=0.6)
        if ax_i == 0 and show_noise_band:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    for j in range(len(bin_list), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or "Signed windowed-flux scan (lines NOT avoided)", fontsize=13)
    fig.supxlabel(r"Window-center rest wavelength [$\AA$]", fontsize=11)
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes


# import the per-spectrum estimators these tools drive (they live in core.py,
# next to run_core_hierarchical which also uses them). No import cycle:
# core.py does not import validation.
from .core import (measure_line, measure_integration,       # noqa: E402
                   measure_gaussian, measure_matched_filter,
                   classify_lya, to_rest_frame)

# progress-bar helper + estimator registry the moved tools expect (mirror core.py)
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:                                            # noqa: BLE001
    def _tqdm(x, **kw):
        return x

_ESTIMATORS = {
    "integration":    measure_integration,
    "gaussian":       measure_gaussian,
    "matched_filter": measure_matched_filter,
}

# =====================================================================
# CORE-SPECTRUM CHARACTERIZATION -- per-galaxy detection credibility
#   Moved here from core.py. These quantify how much to trust a Lya
#   measurement on ONE galaxy's core spectrum (bias, floor, completeness,
#   false-positive rate) -- the per-galaxy analog of the stack nulls above.
#   NOTE: core's injection_recovery was renamed injection_recovery_synthetic
#   here to avoid clashing with validation's cube-based injection_recovery.
# =====================================================================

def compare_estimators(wave, flux, err=None, lya_center=LYA_REST, n_sigma=2.0, **kw):
    """
    Run all three estimators on one rest-frame spectrum and tabulate.

    Returns {method: {flux, flux_err, sn, success, label, reason}}.
    Quick sanity check: do all three agree? If the matched filter reports emitter
    but the Gaussian fails, the line is marginal -- don't trust the Gaussian.
    """
    out = {}
    for name in _ESTIMATORS:
        res = measure_line(wave, flux, err, method=name, lya_center=lya_center, **kw)
        cls = classify_lya(res, n_sigma=n_sigma)
        out[name] = {"flux": res["flux"], "flux_err": res["flux_err"], "sn": res["sn"],
                     "success": res["success"], "label": cls["label"],
                     "reason": res.get("reason")}
    return out


def injection_recovery_synthetic(
    wave, err,
    input_fluxes=(0.0, 0.5, 1.0, 2.0, 5.0),
    lya_center=LYA_REST, sigma_A=1.5, red_offset=0.5,
    methods=("integration", "gaussian", "matched_filter"),
    n_real=300, base_flux=None, seed=0, n_sigma=2.0,
    measure_kwargs=None, verbose=True,
):
    """
    METHOD CHARACTERIZATION -- inject into synthetic noise.

    Injects a unit-area Gaussian of known flux into a clean noise substrate
    (base_flux + N(0, err)), then recovers with each estimator n_real times per
    input level.

    The key diagnostic is input_flux=0: a peak-chasing estimator biases positive
    even with zero signal because it finds the largest noise excursion inside the
    window. The matched filter at a fixed center is linear -> zero-mean under the
    null -> zero bias. The free-center Gaussian shows the largest upward bias.
    This difference sets the practical floor below which you cannot trust a
    detection.

    Returns: {method: {recovered_mean, recovered_std, bias, detect_rate, all},
              input_fluxes, sigma_A, n_real, methods}
    """
    wave  = np.asarray(wave,  dtype=float)
    err   = np.asarray(err,   dtype=float)
    nwave = wave.size
    base  = np.zeros(nwave) if base_flux is None else np.asarray(base_flux, dtype=float)
    inF   = np.asarray(input_fluxes, dtype=float)
    rng   = np.random.default_rng(seed)
    mk    = measure_kwargs or {}
    ctr   = lya_center + red_offset
    g     = (np.exp(-0.5*((wave-ctr)/sigma_A)**2)
             / (sigma_A * np.sqrt(2*np.pi)))

    out = {"input_fluxes": inF, "sigma_A": sigma_A,
           "n_real": n_real, "methods": list(methods)}
    for meth in methods:
        rec = np.full((inF.size, n_real), np.nan)
        det = np.zeros((inF.size, n_real), dtype=bool)
        for fi, Fin in enumerate(inF):
            for k in range(n_real):
                spec  = base + Fin*g + rng.normal(0.0, err)
                res   = measure_line(wave, spec, err, method=meth,
                                     lya_center=lya_center, red_offset=red_offset, **mk)
                rec[fi, k] = res["flux"]
                det[fi, k] = (res["success"] and np.isfinite(res["sn"])
                              and res["sn"] >= n_sigma)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mn  = np.nanmean(rec,  axis=1)
            std = np.nanstd(rec,   axis=1)
        out[meth] = {"recovered_mean": mn, "recovered_std": std,
                     "bias": mn - inF, "detect_rate": det.mean(axis=1), "all": rec}
        if verbose:
            print(f"\n[{meth}]  sigma_A={sigma_A} A, {n_real} reals")
            print(f"{'F_in':>8}  {'F_rec':>9}  {'bias':>9}  {'scatter':>9}  {'det%':>6}")
            for fi, Fin in enumerate(inF):
                print(f"{Fin:8.3g}  {mn[fi]:9.3g}  {mn[fi]-Fin:9.3g}  "
                      f"{std[fi]:9.3g}  {100*out[meth]['detect_rate'][fi]:6.0f}")
    return out


def injection_recovery_sample(
    core_err_obs, wave_obs, catalog,
    input_snrs=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
    method="matched_filter", sigma_A=1.5, red_offset=0.5,
    n_real_per_gal=10, base_spec_obs=None,
    z_col="z", rest_density=True, n_sigma=2.0,
    measure_kwargs=None, seed=0, verbose=True,
):
    """
    REALISTIC COMPLETENESS -- inject into each galaxy's actual noise spectrum.

    For each galaxy × input S/N level (expressed as multiples of that galaxy's
    matched-filter noise floor sigma_F):
        F_in = input_snr * sigma_F(galaxy)     [so every galaxy is tested at the
                                                 same "difficulty", not absolute flux]
        inject F_in * g into noise draws from N(0, core_err[i]) n_real_per_gal times
        recover, record whether S/N >= n_sigma

    Aggregate over galaxies -> completeness curve vs input S/N. This accounts for:
        - z-dependent sky brightness (errors vary with wavelength AND galaxy z)
        - wavelength coverage gaps (some galaxies lose Lya off the detector edge)
        - the natural spread of sigma_F across the sample

    The input_snrs range includes 0.0 -- this measures the false-positive rate
    (should equal ~one-sided tail of Normal for the matched filter, ~2.5% at 2sigma).

    Returns dict: input_snrs (n_snr,), completeness (n_snr,), false_positive_rate,
                  sigma_F_per_gal (ngal,), detect_all (ngal, n_snr, n_real_per_gal),
                  method, per-galaxy bias arrays.
    """
    core_err_obs = np.asarray(core_err_obs, dtype=float)
    wave_obs     = np.asarray(wave_obs,     dtype=float)
    ngal, nwave  = core_err_obs.shape
    z            = np.asarray(catalog[z_col], dtype=float)
    input_snrs   = np.asarray(input_snrs,    dtype=float)
    rng          = np.random.default_rng(seed)
    mk           = measure_kwargs or {}
    base_obs     = (np.zeros((ngal, nwave), dtype=float) if base_spec_obs is None
                    else np.asarray(base_spec_obs, dtype=float))

    sigma_F_all  = np.full(ngal, np.nan)
    detect_all   = np.zeros((ngal, input_snrs.size, n_real_per_gal), dtype=bool)
    bias_all     = np.full( (ngal, input_snrs.size, n_real_per_gal), np.nan)

    it = _tqdm(range(ngal), desc="inj-rec sample") if verbose else range(ngal)
    for i in it:
        if not np.isfinite(z[i]) or z[i] <= 0:
            continue
        # rest-frame
        wr, br, er = to_rest_frame(wave_obs, base_obs[i], core_err_obs[i],
                                   z=z[i], rest_density=rest_density)
        # per-galaxy noise floor from matched filter (analytical)
        mf0 = measure_matched_filter(wr, br, er, red_offset=red_offset,
                                     template_sigma=sigma_A, **mk)
        if not mf0["success"] or not np.isfinite(mf0["flux_err"]):
            continue
        sigma_F_all[i] = mf0["flux_err"]
        lya_c = LYA_REST
        ctr   = lya_c + red_offset
        g     = (np.exp(-0.5*((wr - ctr)/sigma_A)**2)
                 / (sigma_A * np.sqrt(2*np.pi)))

        for si, snr_in in enumerate(input_snrs):
            Fin = snr_in * sigma_F_all[i]
            for k in range(n_real_per_gal):
                spec_k = br + Fin*g + rng.normal(0.0, er)
                res    = measure_line(wr, spec_k, er, method=method,
                                      red_offset=red_offset, **mk)
                detect_all[i, si, k] = (res["success"] and np.isfinite(res["sn"])
                                        and res["sn"] >= n_sigma)
                if np.isfinite(res["flux"]) and np.isfinite(sigma_F_all[i]):
                    bias_all[i, si, k] = (res["flux"] - Fin) / sigma_F_all[i]

    # aggregate over galaxies and realizations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        completeness = np.nanmean(detect_all.reshape(ngal, input_snrs.size, -1),
                                  axis=(0, 2))
        bias_med     = np.nanmedian(bias_all.reshape(ngal, input_snrs.size, -1),
                                    axis=(0, 2))

    if verbose:
        print(f"\ninjection_recovery_sample [{method}] "
              f"({ngal} galaxies, {n_real_per_gal} reals each)")
        print(f"{'SNR_in':>8}  {'completeness':>14}  {'bias/σ_F':>10}")
        for si, snr in enumerate(input_snrs):
            print(f"{snr:8.2f}  {completeness[si]:14.3f}  {bias_med[si]:10.3f}")

    return dict(input_snrs=input_snrs, completeness=completeness,
                false_positive_rate=completeness[0] if input_snrs[0]==0 else np.nan,
                bias_median_sigma=bias_med,
                sigma_F_per_gal=sigma_F_all, detect_all=detect_all,
                method=method, n_real_per_gal=n_real_per_gal)


def detection_completeness(injection_result, method=None):
    """
    Extract completeness vs input flux (or input S/N) from an injection result.

    Works with the output of either injection_recovery_synthetic (absolute flux) or
    injection_recovery_sample (S/N units). Returns (input_axis, detect_rate).
    The value at input=0 is the false-positive rate; the 50% crossing is the
    effective detection threshold.
    """
    if "completeness" in injection_result:
        # from injection_recovery_sample -- S/N-conditioned
        return (np.asarray(injection_result["input_snrs"],  dtype=float),
                np.asarray(injection_result["completeness"], dtype=float))
    # from injection_recovery_synthetic -- per-method absolute-flux
    meth = method or injection_result["methods"][0]
    if meth not in injection_result:
        raise KeyError(f"{meth!r} not in result (have {injection_result['methods']}).")
    return (np.asarray(injection_result["input_fluxes"],               dtype=float),
            np.asarray(injection_result[meth]["detect_rate"],          dtype=float))


def null_redshift_test(
    core_spec_obs, core_err_obs, wave_obs, z_true,
    method="matched_filter", n_permutations=30,
    rest_density=True, measure_kwargs=None, seed=0, n_sigma=2.0, verbose=True,
):
    """
    NULL TEST -- permute redshifts, measure 'Lya' at the wrong z.

    Complements validation.redshift_scramble_null (which acts on the STACKED
    halo profile) but targets the PER-GALAXY core measurement. With scrambled z
    the measurement window lands on a random part of the spectrum that has no
    real line; the recovered flux distribution should be centered at 0. Any
    systematic offset indicates a continuum artifact (e.g. sky-line residual,
    Lya-forest tilt) at the expected line wavelength.

    Returns dict: null_flux (n_perm, ngal), null_sn (n_perm, ngal),
                  null_flux_med (ngal), null_flux_std (ngal), method.
    """
    core_spec_obs = np.asarray(core_spec_obs, dtype=float)
    core_err_obs  = np.asarray(core_err_obs,  dtype=float)
    wave_obs      = np.asarray(wave_obs,       dtype=float)
    z_true        = np.asarray(z_true,         dtype=float)
    ngal          = core_spec_obs.shape[0]
    rng           = np.random.default_rng(seed)
    mk            = measure_kwargs or {}

    null_flux = np.full((n_permutations, ngal), np.nan)
    null_sn   = np.full((n_permutations, ngal), np.nan)

    for p in range(n_permutations):
        z_perm = z_true[rng.permutation(ngal)]
        for i in range(ngal):
            if not np.isfinite(z_perm[i]) or z_perm[i] <= 0:
                continue
            wr, fr, er = to_rest_frame(wave_obs, core_spec_obs[i], core_err_obs[i],
                                       z=z_perm[i], rest_density=rest_density)
            res              = measure_line(wr, fr, er, method=method, **mk)
            null_flux[p, i]  = res["flux"]
            null_sn[p, i]    = res["sn"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nf_med = np.nanmedian(null_flux, axis=0)
        nf_std = np.nanstd(null_flux,   axis=0)

    if verbose:
        print(f"\nnull_redshift_test [{method}] ({n_permutations} permutations, "
              f"{ngal} galaxies)")
        print(f"  null flux median over sample: {np.nanmedian(nf_med):+.4g}")
        print(f"  null flux std   over sample: {np.nanmedian(nf_std):.4g}")

    return dict(null_flux=null_flux, null_sn=null_sn,
                null_flux_med=nf_med, null_flux_std=nf_std,
                n_permutations=n_permutations, method=method)


# =====================================================================
# ONE-CALL DRIVER for the core-characterization suite above.
# =====================================================================

def run_core_characterization(
    core_spec_obs, core_err_obs, wave_obs, catalog,
    # ---- shared / sample ----
    z_col="z", z_true=None, rest_density=True, measure_kwargs=None,
    n_sigma=2.0, seed=0, verbose=True,
    # ---- which pieces to run (all on by default) ----
    run_synthetic=True, run_sample=True, run_null=True, run_compare=True,
    # ---- injection_recovery_synthetic (into synthetic noise) ----
    syn_err_spectrum=None,
    syn_input_fluxes=(0.0, 0.5, 1.0, 2.0, 5.0),
    syn_lya_center=LYA_REST, syn_sigma_A=1.5, syn_red_offset=0.5,
    syn_methods=("integration", "gaussian", "matched_filter"),
    syn_n_real=300, syn_base_flux=None,
    # ---- injection_recovery_sample (into each galaxy's real noise) ----
    samp_input_snrs=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
    samp_method="matched_filter", samp_sigma_A=1.5, samp_red_offset=0.5,
    samp_n_real_per_gal=10, samp_base_spec_obs=None,
    # ---- null_redshift_test (measure Lya at the WRONG z) ----
    null_method="matched_filter", null_n_permutations=30,
    # ---- compare_estimators (one example galaxy, rest-frame) ----
    compare_index=0, compare_lya_center=LYA_REST,
):
    """
    Run the whole per-galaxy core-characterization suite in one call and return
    every result in one dict. Every sub-function's parameters are exposed here
    (prefixed syn_/samp_/null_/compare_) and passed straight through -- nothing
    is silently left on its own default. Toggle any piece off with the run_*
    flags. Meant for the per-galaxy detection-credibility write-up, NOT the
    stacked-profile paper.

    core_spec_obs, core_err_obs : (ngal, nwave) OBSERVED-frame core spectra +
        errors from core.extract_core_spectra.
    wave_obs   : (nwave,) observed wavelength grid.
    catalog    : per-galaxy table (must carry z_col).
    z_true     : per-galaxy true redshifts for the null test; None -> catalog[z_col].
    syn_err_spectrum : (nwave,) error spectrum for the SYNTHETIC injection;
        None -> the per-pixel nanmedian of core_err_obs (a representative noise
        level for the sample).
    compare_index : row to rest-frame and tabulate estimators on for
        compare_estimators; None -> skip that piece even if run_compare.

    Returns dict with keys (present only for the pieces you ran):
        'synthetic'      -> injection_recovery_synthetic result
        'sample'         -> injection_recovery_sample result
        'null'           -> null_redshift_test result
        'compare'        -> compare_estimators table (one galaxy)
        'completeness'   -> {'synthetic': (x, rate)|None, 'sample': (x, rate)}
                            via detection_completeness on the injection results
        'params'         -> the exact settings used (for provenance)
    """
    import numpy as _np
    out = {"params": dict(
        z_col=z_col, rest_density=rest_density, n_sigma=n_sigma, seed=seed,
        syn=dict(input_fluxes=syn_input_fluxes, sigma_A=syn_sigma_A,
                 red_offset=syn_red_offset, methods=syn_methods,
                 n_real=syn_n_real, base_flux=syn_base_flux),
        samp=dict(input_snrs=samp_input_snrs, method=samp_method,
                  sigma_A=samp_sigma_A, red_offset=samp_red_offset,
                  n_real_per_gal=samp_n_real_per_gal),
        null=dict(method=null_method, n_permutations=null_n_permutations),
        compare=dict(index=compare_index, lya_center=compare_lya_center),
    )}
    completeness = {}

    if z_true is None:
        z_true = _np.asarray(catalog[z_col], dtype=float)

    if run_synthetic:
        err_spec = (syn_err_spectrum if syn_err_spectrum is not None
                    else _np.nanmedian(_np.asarray(core_err_obs, float), axis=0))
        out["synthetic"] = injection_recovery_synthetic(
            wave_obs, err_spec, input_fluxes=syn_input_fluxes,
            lya_center=syn_lya_center, sigma_A=syn_sigma_A,
            red_offset=syn_red_offset, methods=syn_methods, n_real=syn_n_real,
            base_flux=syn_base_flux, seed=seed, n_sigma=n_sigma,
            measure_kwargs=measure_kwargs, verbose=verbose)
        try:
            completeness["synthetic"] = detection_completeness(out["synthetic"])
        except Exception:                                        # noqa: BLE001
            completeness["synthetic"] = None

    if run_sample:
        out["sample"] = injection_recovery_sample(
            core_err_obs, wave_obs, catalog, input_snrs=samp_input_snrs,
            method=samp_method, sigma_A=samp_sigma_A, red_offset=samp_red_offset,
            n_real_per_gal=samp_n_real_per_gal, base_spec_obs=samp_base_spec_obs,
            z_col=z_col, rest_density=rest_density, n_sigma=n_sigma,
            measure_kwargs=measure_kwargs, seed=seed, verbose=verbose)
        completeness["sample"] = detection_completeness(out["sample"])

    if run_null:
        out["null"] = null_redshift_test(
            core_spec_obs, core_err_obs, wave_obs, z_true, method=null_method,
            n_permutations=null_n_permutations, rest_density=rest_density,
            measure_kwargs=measure_kwargs, seed=seed, n_sigma=n_sigma,
            verbose=verbose)

    if run_compare and compare_index is not None:
        i = int(compare_index)
        wr, fr, er = to_rest_frame(wave_obs, core_spec_obs[i], core_err_obs[i],
                                   float(z_true[i]), rest_density=rest_density)
        out["compare"] = compare_estimators(
            wr, fr, er, lya_center=compare_lya_center, n_sigma=n_sigma,
            **(measure_kwargs or {}))

    out["completeness"] = completeness
    if verbose:
        ran = [k for k in ("synthetic", "sample", "null", "compare") if k in out]
        print(f"run_core_characterization: ran {', '.join(ran)}")
    return out
