"""
measure.py -- Stage 3 (MEASURE) computation engine. Arrays in, numbers out. Fast.

Pure measurement: no matplotlib, and the per-spectrum helpers need no
PipelineConfig, so everything drops straight into a notebook. The DISPLAY and
orchestration companion is analysis.py (formerly measuring.py): near-identical
role-name, opposite job -- measure.py COMPUTES, analysis.py DRAWS.

This module is also the SINGLE SOURCE OF TRUTH for the centroid-estimator
vocabulary: CENTROID_METHOD_SPECS (label -> measure kwargs) and
CENTROID_METHOD_LABELS (label -> pretty name). analysis.py and validation.py
import these so every table, plot, and stress-test uses the same labels.

Primary observable is the flux-weighted Lya CENTROID per radial bin; its
uncertainty comes from a galaxy-resampling bootstrap (resample galaxies ->
re-stack -> re-measure), which is why Stage 2 preserves the per-galaxy cube.
A double-Gaussian blue/red decomposition is deliberately NOT done here -- it
needs higher S/N and spectral resolution than the binned stacks have. Instead
a cheap, robust "side ratio" (summed continuum-subtracted flux blue vs red of
line center) is provided as a coarse asymmetry proxy.

This module has four layers:

  1. Per-spectrum measurement (operate on one (nwave,) stacked spectrum):
        fit_local_poly_continuum   -- robust sigma-clipped sideband polynomial
        get_continuum_model        -- sideband continuum ('median' or 'poly')
        flux_weighted_centroid     -- the headline centroid (optional neg-clip)
        gaussian_centroid          -- single-Gaussian-fit peak (mu)
        flux_median_centroid       -- 50% cumulative-flux centroid
        peak_parabola_centroid     -- sub-pixel peak (mode)
        gauss_weighted_centroid    -- adaptive Gaussian-weighted first moment
        line_moments               -- centroid + width + skewness (asymmetry)
        measure_centroid           -- dispatcher over the above (CENTROID_METHOD_SPECS)
        integrated_line_flux       -- canonical windowed continuum-subtracted flux
        integrated_line_flux_per_bin -- integrated_line_flux looped over one stack's
                                        radial bins (fiducial stack or one bootstrap draw)
        blue_red_side_ratio        -- coarse asymmetry proxy

  2. Stacking a resampled cube along the galaxy axis:
        stack_galaxies

  3. Galaxy bootstraps over the per-galaxy cube (ngal, nrad, nwave):
        bootstrap_measurements     -- centroid (+ side ratio) per bin, with 16/84
        bootstrap_stack_error      -- per-pixel 1-sigma flux error per bin

  4. Derived radial diagnostics -- operate on a measure_all_bins / bootstrap_all
     SUMMARY dict (not raw arrays), so they run AFTER Stage 3, not inside it:
        flux_curve_of_growth      -- cumulative Lya luminosity and flux fraction
                                      vs radius, built from total_flux_fid/_all
                                      (the already-bootstrapped windowed flux)
                                      times a representative per-galaxy fiber
                                      footprint (fiber_area_kpc2, from the L_kpc2
                                      unit conversion), cumulatively summed out to
                                      a configurable r_max.

All functions take plain arrays (no PipelineConfig required) so they drop
straight into a notebook. LYA_REST defaults to 1215.67 (== PipelineConfig.LYA_REST).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy.stats import biweight_location

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm optional
    def tqdm(x, **kwargs):
        return x

if TYPE_CHECKING:
    from .config import PipelineConfig

# Canonical measurement defaults live in config.py (the single source of truth);
# imported here so the standalone per-spectrum helpers below share ONE default
# with the config-driven pipeline. Change a default in config.py, not here.
from .config import (
    DEFAULT_CONT_BOUNDS, DEFAULT_CONT_METHOD, DEFAULT_CONT_ORDER,
    DEFAULT_CENTROID_METHOD,
)

C_KMS = 299792.458       # speed of light (km/s)
LYA_REST = 1215.67       # vacuum Lya rest wavelength (Angstrom); == PipelineConfig.LYA_REST


# =====================================================================
# Shared "what am I running" header.
#
# Every analysis / measure / validation entry point that resamples or loops
# calls this once at the top so the notebook shows, on one line, WHAT is being
# run and WITH WHICH parameters -- before the progress bar starts. Keeps the
# reporting identical and in one place instead of ad-hoc prints per function.
# =====================================================================
def run_header(action, *, verbose=True, announce=True, **params):
    """
    Print a one-line '[action] key=val key=val ...' banner and return `action`
    (so it can double as the tqdm `desc`).

    action   : short human label, e.g. 'bootstrap centroid [flux-weighted]'.
    verbose  : master switch; False -> print nothing, still return `action`.
    announce : when False, skip the banner but still return `action` (used by
               sweeps that print their OWN header once, then want per-iteration
               progress bars labelled but not re-announced).
    params   : the knobs that change the result (nboot, method, bounds, seed).
               None-valued entries are dropped so the line stays tight.
    """
    if not verbose or not announce:
        return action
    parts = []
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        elif isinstance(v, (tuple, list)) and len(v) == 2 and all(
                isinstance(x, (int, float)) for x in v):
            parts.append(f"{k}=({v[0]:g},{v[1]:g})")
        else:
            parts.append(f"{k}={v}")
    print(f"▶ {action}" + ("  |  " + "  ".join(parts) if parts else ""))
    return action


# =====================================================================
# 1. PER-SPECTRUM MEASUREMENT
# =====================================================================

def fit_local_poly_continuum(wave, flux, cont_bounds=DEFAULT_CONT_BOUNDS,
                             order=1, sigma_clip=3, niter=3):
    """
    Robust local polynomial continuum from the sideband windows only.

    Fits a low-order polynomial to the pixels inside cont_bounds (the line
    region is excluded by construction), with sigma-clipping to reject
    outliers, and evaluates it across the full wave array. Wavelength is
    centered before the fit for numerical stability.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    cmask = np.zeros_like(wave, dtype=bool)
    for lo, hi in cont_bounds:
        cmask |= (wave >= lo) & (wave <= hi)
    cmask &= np.isfinite(wave) & np.isfinite(flux)

    x = wave[cmask]
    y = flux[cmask]
    if len(x) < order + 2:
        return np.full_like(wave, np.nanmedian(flux))

    x0 = np.nanmedian(x)
    xx = x - x0
    good = np.ones(len(x), dtype=bool)

    for _ in range(niter):
        coeff = np.polyfit(xx[good], y[good], order)
        model = np.polyval(coeff, xx)
        resid = y - model
        sig = 1.4826 * np.nanmedian(np.abs(resid[good] - np.nanmedian(resid[good])))
        if not np.isfinite(sig) or sig == 0:
            break
        good = np.abs(resid) < sigma_clip * sig

    coeff = np.polyfit(xx[good], y[good], order)
    return np.polyval(coeff, wave - x0)


def get_continuum_model(wave, flux, cont_bounds=DEFAULT_CONT_BOUNDS,
                        cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER):
    """
    Continuum model over the full wave grid.

    cont_method : 'poly'   -- robust sigma-clipped local polynomial (default)
                  'median' -- single median level of the sideband pixels
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if cont_method == "median":
        cmask = np.zeros_like(wave, dtype=bool)
        for lo, hi in cont_bounds:
            cmask |= (wave >= lo) & (wave <= hi)
        cmask &= np.isfinite(flux)
        cont_val = np.nanmedian(flux[cmask]) if np.any(cmask) else 0.0
        return np.full_like(wave, cont_val)
    elif cont_method == "poly":
        return fit_local_poly_continuum(wave, flux, cont_bounds=cont_bounds,
                                        order=cont_order)
    else:
        raise ValueError("cont_method must be 'poly' or 'median'")


def flux_weighted_centroid(
    wave, flux, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    min_pixels=8,
    clip_negative=True,
):
    """
    Continuum-subtract, then flux-weight to a centroid over the line window.

        centroid_A = sum(x * w) / sum(w)   over the window,
        w = max(y, 0)  if clip_negative   else   w = y,
        y = flux - continuum.

    Returns the centroid in Angstrom and as a velocity offset from lya_center
    (km/s). This is the headline observable: cheap, robust, and not dependent
    on a model line profile.

    clip_negative=True  (default, == historical behaviour): zero the negative
        pixels before weighting. Stable, but biases a near-null signal slightly
        positive (keeps positive noise, discards negative noise).
    clip_negative=False: weight by the raw continuum-subtracted flux. Unbiased
        under symmetric noise, but the weighted mean can fall OUTSIDE the window
        when sum(w) is small/negative -- those draws are flagged success=False
        (out-of-window guard) rather than returning a wild velocity.

    Returns
    -------
    dict with keys: success, reason, centroid_A, centroid_vel_kms, continuum,
                    flux_contsub.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y_all = flux - cont

    m = (wave >= bounds[0]) & (wave <= bounds[1]) & np.isfinite(y_all)
    if err is not None:
        err = np.asarray(err, dtype=float)
        m &= np.isfinite(err) & (err > 0)

    if np.count_nonzero(m) < min_pixels:
        return {"success": False, "reason": "Not enough finite pixels",
                "centroid_A": np.nan, "centroid_vel_kms": np.nan,
                "continuum": cont, "flux_contsub": y_all}

    x = wave[m]
    y = y_all[m]
    weights = np.where(y > 0, y, 0.0) if clip_negative else y
    wsum = np.nansum(weights)

    if np.isfinite(wsum) and wsum > 0:
        centroid = np.nansum(x * weights) / wsum
        if bounds[0] <= centroid <= bounds[1]:
            centroid_vel = C_KMS * (centroid - lya_center) / lya_center
            ok, reason = True, None
        else:
            # unstable (negative-weight) draw -- weighted mean left the window
            centroid = np.nan
            centroid_vel = np.nan
            ok, reason = False, "Centroid outside window (unstable weights)"
    else:
        centroid = np.nan
        centroid_vel = np.nan
        ok, reason = False, "No positive net line flux"

    return {"success": ok, "reason": reason,
            "centroid_A": centroid, "centroid_vel_kms": centroid_vel,
            "continuum": cont, "flux_contsub": y_all}


# =====================================================================
# 1b. ALTERNATIVE CENTROID ESTIMATORS + DISPATCHER
# =====================================================================

def gaussian_centroid(
    wave, flux, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    min_pixels=8,
    max_sigma_A=None,
):
    """
    Fit a single Gaussian (A·exp(-0.5((x-mu)/sigma)^2)) to the continuum-subtracted
    flux in the window; the centroid is the fitted mean mu.

    Unlike the flux-weighted mean this does NOT weight by where the flux sits, so
    it is insensitive to symmetric noise in the wings -- but for an intrinsically
    asymmetric line (Lya red wing) mu tracks the PEAK, a genuinely different
    quantity than the flux-weighted mean. Good as a cross-check; not a drop-in
    replacement on the real asymmetric line.

    max_sigma_A : upper bound on sigma (A). Defaults to half the window width.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y_all = flux - cont

    m = (wave >= bounds[0]) & (wave <= bounds[1]) & np.isfinite(y_all)
    if err is not None:
        err = np.asarray(err, dtype=float)
        m &= np.isfinite(err) & (err > 0)

    base = {"centroid_A": np.nan, "centroid_vel_kms": np.nan,
            "continuum": cont, "flux_contsub": y_all,
            "amp": np.nan, "sigma_A": np.nan}
    if np.count_nonzero(m) < min_pixels:
        return {"success": False, "reason": "Not enough finite pixels", **base}

    x = wave[m]
    y = y_all[m]
    e = err[m] if err is not None else None

    try:
        from scipy.optimize import curve_fit
    except Exception:
        return {"success": False, "reason": "scipy unavailable", **base}

    half_w = 0.5 * (bounds[1] - bounds[0])
    sig_hi = float(max_sigma_A) if max_sigma_A is not None else half_w
    A0 = float(np.nanmax(y))
    p0 = [max(A0, 1e-3), float(lya_center), min(max(half_w / 3.0, 0.3), sig_hi)]
    lower = [0.0, bounds[0], 0.1]
    upper = [np.inf, bounds[1], sig_hi]

    def _gauss(xx, A, mu, sig):
        return A * np.exp(-0.5 * ((xx - mu) / sig) ** 2)

    try:
        popt, _ = curve_fit(
            _gauss, x, y, p0=p0, sigma=e, absolute_sigma=e is not None,
            bounds=(lower, upper), maxfev=5000)
    except Exception as exc:
        return {"success": False, "reason": f"fit failed: {exc}", **base}

    A, mu, sig = (float(v) for v in popt)
    ok = np.isfinite(mu) and (bounds[0] <= mu <= bounds[1]) and (A > 0)
    centroid = mu if ok else np.nan
    vel = (C_KMS * (mu - lya_center) / lya_center) if ok else np.nan
    return {"success": ok,
            "reason": None if ok else "Gaussian fit non-physical",
            "centroid_A": centroid, "centroid_vel_kms": vel,
            "continuum": cont, "flux_contsub": y_all,
            "amp": A, "sigma_A": sig}


def flux_median_centroid(
    wave, flux, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    min_pixels=8,
):
    """
    Wavelength at which the cumulative (positive) line flux reaches 50%.

    A non-mean centroid: the flux MEDIAN rather than the flux MEAN, so it
    down-weights the wings relative to flux_weighted_centroid and cannot blow up
    (the CDF is monotone in [0, 1]). Positive clipping is intrinsic here -- a CDF
    needs non-negative increments -- so this is the "robust to wing noise"
    counterpart of the clipped flux-weighted mean.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y_all = flux - cont

    m = (wave >= bounds[0]) & (wave <= bounds[1]) & np.isfinite(y_all)
    if err is not None:
        err = np.asarray(err, dtype=float)
        m &= np.isfinite(err) & (err > 0)

    base = {"centroid_A": np.nan, "centroid_vel_kms": np.nan,
            "continuum": cont, "flux_contsub": y_all}
    if np.count_nonzero(m) < min_pixels:
        return {"success": False, "reason": "Not enough finite pixels", **base}

    x = wave[m]
    y = np.where(y_all[m] > 0, y_all[m], 0.0)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    tot = np.nansum(ys)
    if not (np.isfinite(tot) and tot > 0):
        return {"success": False, "reason": "No positive line flux", **base}

    # Half-pixel CENTERED cumulative: each pixel's flux is credited at its
    # CENTER, not its right edge. The naive cumsum(ys)/tot over-credits each
    # pixel's own flux at its own wavelength, which pushes the 0.5 crossing
    # ~half a pixel blue -- a fixed, sub-pixel, bin-independent offset (~-24 km/s
    # on a 0.1 A grid) that biased flux_median relative to the flux-weighted
    # mean. Subtracting 0.5*ys removes it.
    cdf = (np.cumsum(ys) - 0.5 * ys) / tot   # centered; nondecreasing, ends <1
    centroid = float(np.interp(0.5, cdf, xs))
    vel = C_KMS * (centroid - lya_center) / lya_center
    return {"success": True, "reason": None,
            "centroid_A": centroid, "centroid_vel_kms": vel,
            "continuum": cont, "flux_contsub": y_all}


def peak_parabola_centroid(
    wave, flux, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    min_pixels=8,
):
    """
    Sub-pixel peak (mode) from a 3-point parabola through the max pixel and its
    two neighbours (general spacing, so it is correct on a nonuniform grid).

    Locates the emission PEAK and ignores the wing shape entirely, so it is the
    most asymmetry-insensitive estimator -- but the noisiest at low S/N, since it
    keys off a single pixel. Useful to bracket how much of the flux-weighted
    trend is "peak moved" vs "wing reweighted".
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y_all = flux - cont

    m = (wave >= bounds[0]) & (wave <= bounds[1]) & np.isfinite(y_all)
    if err is not None:
        err = np.asarray(err, dtype=float)
        m &= np.isfinite(err) & (err > 0)

    base = {"centroid_A": np.nan, "centroid_vel_kms": np.nan,
            "continuum": cont, "flux_contsub": y_all}
    if np.count_nonzero(m) < max(min_pixels, 3):
        return {"success": False, "reason": "Not enough finite pixels", **base}

    x = wave[m]
    y = y_all[m]
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    i = int(np.nanargmax(ys))
    if ys[i] <= 0:
        return {"success": False, "reason": "No positive peak", **base}

    if i == 0 or i == len(xs) - 1:
        # peak pinned at a window edge: take the pixel itself, flag the reason
        centroid = float(xs[i])
        reason = "Peak at window edge (pixel-level)"
    else:
        x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        d1 = (y1 - y0) / (x1 - x0)
        d2 = (y2 - y1) / (x2 - x1)
        a = (d2 - d1) / (x2 - x0)             # parabola curvature
        if not np.isfinite(a) or a >= 0:      # not a concave-down max
            centroid = float(x1)
            reason = "Degenerate parabola (pixel-level)"
        else:
            b = d1 - a * (x0 + x1)
            vertex = -b / (2.0 * a)
            if bounds[0] <= vertex <= bounds[1]:
                centroid = float(vertex)
                reason = None
            else:
                centroid = float(x1)
                reason = "Vertex outside window (pixel-level)"

    vel = C_KMS * (centroid - lya_center) / lya_center
    return {"success": True, "reason": reason,
            "centroid_A": centroid, "centroid_vel_kms": vel,
            "continuum": cont, "flux_contsub": y_all}


def line_moments(
    wave, flux, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    min_pixels=8,
    clip_negative=True,
):
    """
    Flux-weighted MOMENTS of the line over the window, in one pass:

        centroid_A = sum(w x) / sum(w)                         (1st moment)
        sigma_A    = sqrt(sum(w (x-c)^2) / sum(w))             (2nd moment)
        skewness   = [sum(w (x-c)^3) / sum(w)] / sigma_A^3     (3rd, standardized)

    w = max(y, 0) if clip_negative else y;  y = flux - continuum. The 1st moment
    is exactly flux_weighted_centroid (same convention), so this is its natural
    extension: sigma is the line width and skewness is a model-free ASYMMETRY
    statistic. skewness > 0 means a tail to longer wavelength (a RED wing) -- the
    expected sign for resonant Lya scattering. This is the quantity that explains
    WHY the clipped / unclipped / Gaussian centroids disagree: an asymmetric line
    has centroid != peak by an amount set by the skew.

    Returns dict: success, reason, centroid_A, centroid_vel_kms, sigma_A,
                  sigma_kms, skewness, continuum, flux_contsub.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y_all = flux - cont

    m = (wave >= bounds[0]) & (wave <= bounds[1]) & np.isfinite(y_all)
    if err is not None:
        err = np.asarray(err, dtype=float)
        m &= np.isfinite(err) & (err > 0)

    base = {"centroid_A": np.nan, "centroid_vel_kms": np.nan,
            "sigma_A": np.nan, "sigma_kms": np.nan, "skewness": np.nan,
            "continuum": cont, "flux_contsub": y_all}
    if np.count_nonzero(m) < min_pixels:
        return {"success": False, "reason": "Not enough finite pixels", **base}

    x = wave[m]
    y = y_all[m]
    w = np.where(y > 0, y, 0.0) if clip_negative else y
    wsum = np.nansum(w)
    if not (np.isfinite(wsum) and wsum > 0):
        return {"success": False, "reason": "No positive net line flux", **base}

    c = np.nansum(w * x) / wsum
    if not (bounds[0] <= c <= bounds[1]):
        return {"success": False,
                "reason": "Centroid outside window (unstable weights)", **base}

    var = np.nansum(w * (x - c) ** 2) / wsum
    sigma_A = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan
    if np.isfinite(sigma_A) and sigma_A > 0:
        skew = float(np.nansum(w * (x - c) ** 3) / wsum / sigma_A ** 3)
    else:
        skew = np.nan

    return {"success": True, "reason": None,
            "centroid_A": float(c),
            "centroid_vel_kms": C_KMS * (c - lya_center) / lya_center,
            "sigma_A": sigma_A,
            "sigma_kms": (C_KMS * sigma_A / lya_center
                          if np.isfinite(sigma_A) else np.nan),
            "skewness": skew,
            "continuum": cont, "flux_contsub": y_all}


def gauss_weighted_centroid(
    wave, flux, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    min_pixels=8,
    sigma_w=None,
    max_iter=100,
    tol=1e-4,
):
    """
    Gaussian-weighted ("windowed" / adaptive) first-moment centroid.

    Multiply the continuum-subtracted flux by a Gaussian weight centred on the
    current centroid estimate, take the weighted first moment, and iterate the
    centre to its fixed point:

        w_i = exp(-0.5 ((x_i - c)/sigma_w)^2)
        c   = sum(w_i x_i y_i) / sum(w_i y_i),     y = flux - continuum

    The smooth weight suppresses the window edges GRADUALLY (variance reduction,
    like neg-clipping) with NO threshold, so it avoids the rectification bias of
    clip_negative=True while staying far more stable than clip_negative=False;
    because the weight is matched to the line it tracks the Gaussian-fit centroid
    at a fraction of the cost. For a SYMMETRIC line the fixed point is the true
    centre for any sigma_w (each iteration shrinks the residual offset by
    sigma_line^2 / (sigma_line^2 + sigma_w^2) < 1). On an ASYMMETRIC line (the
    Lya red wing) the symmetric weight down-weights the tail, so the result sits
    between the PEAK and the flux-weighted MEAN -- closer to the peak the narrower
    sigma_w. Like gaussian_centroid, that is a genuinely different quantity than
    flux_weighted_centroid, NOT a drop-in lower-noise replacement on the real line.

    Does NOT consult clip_negative (the smooth weight replaces clipping). The
    centre is SEEDED from the clipped flux-weighted moment, and -- unless sigma_w
    is given -- the weight width from the clipped 2nd moment (matched filter,
    floored at ~2 pixels).

    sigma_w : weight width in Angstrom. None -> derive per spectrum from the line.
              Pass a CONSTANT for a controlled radial comparison: a per-bin-derived
              width makes the measured quantity drift slightly with S/N and width.
    max_iter, tol : iteration cap and convergence tolerance on the centre (A).

    Returns dict: success, reason, centroid_A, centroid_vel_kms, sigma_A,
                  sigma_kms, n_iter, continuum, flux_contsub.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y_all = flux - cont

    m = (wave >= bounds[0]) & (wave <= bounds[1]) & np.isfinite(y_all)
    if err is not None:
        err = np.asarray(err, dtype=float)
        m &= np.isfinite(err) & (err > 0)

    base = {"centroid_A": np.nan, "centroid_vel_kms": np.nan,
            "sigma_A": np.nan, "sigma_kms": np.nan, "n_iter": 0,
            "continuum": cont, "flux_contsub": y_all}
    if np.count_nonzero(m) < min_pixels:
        return {"success": False, "reason": "Not enough finite pixels", **base}

    x = wave[m]
    y = y_all[m]
    dpix = float(np.nanmedian(np.diff(x))) if x.size > 1 else 1.0

    # --- seed centre (+ width) from the CLIPPED moment; seeding only ---
    fpos = np.where(y > 0, y, 0.0)
    wsum0 = np.nansum(fpos)
    if not (np.isfinite(wsum0) and wsum0 > 0):
        return {"success": False, "reason": "No positive net line flux", **base}
    c0 = np.nansum(x * fpos) / wsum0
    if sigma_w is None:
        var0 = np.nansum(fpos * (x - c0) ** 2) / wsum0
        sigma_w = np.sqrt(var0) if np.isfinite(var0) and var0 > 0 else 3.0 * dpix
        sigma_w = max(float(sigma_w), 2.0 * dpix)   # floor: don't collapse to a pixel
    sigma_w = float(sigma_w)
    sig_kms = C_KMS * sigma_w / lya_center
    base_sig = {**base, "sigma_A": sigma_w, "sigma_kms": sig_kms}

    # --- fixed-width fixed-point iteration on the centre (RAW flux, not clipped) ---
    c = float(c0)
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        w = np.exp(-0.5 * ((x - c) / sigma_w) ** 2)
        denom = np.nansum(w * y)
        if not (np.isfinite(denom) and denom > 0):
            return {"success": False, "reason": "No positive net line flux",
                    **{**base_sig, "n_iter": n_iter}}
        c_new = np.nansum(w * x * y) / denom
        if not np.isfinite(c_new) or not (bounds[0] <= c_new <= bounds[1]):
            return {"success": False,
                    "reason": "Centroid outside window (unstable weights)",
                    **{**base_sig, "n_iter": n_iter}}
        if abs(c_new - c) < tol:
            c = c_new
            converged = True
            break
        c = c_new

    if not converged:
        return {"success": False, "reason": "No convergence",
                **{**base_sig, "n_iter": n_iter}}

    return {"success": True, "reason": None,
            "centroid_A": float(c),
            "centroid_vel_kms": C_KMS * (c - lya_center) / lya_center,
            "sigma_A": sigma_w, "sigma_kms": sig_kms, "n_iter": n_iter,
            "continuum": cont, "flux_contsub": y_all}


_CENTROID_DISPATCH = {
    "flux_weighted": flux_weighted_centroid,
    "fw": flux_weighted_centroid,
    "gaussian": gaussian_centroid,
    "gauss": gaussian_centroid,
    "flux_median": flux_median_centroid,
    "median50": flux_median_centroid,
    "peak": peak_parabola_centroid,
    "parabola": peak_parabola_centroid,
    "mode": peak_parabola_centroid,
    "moments": line_moments,
    "gauss_weighted": gauss_weighted_centroid,
    "windowed": gauss_weighted_centroid,
    "gwt": gauss_weighted_centroid,
}


# =====================================================================
# CANONICAL CENTROID-ESTIMATOR VOCABULARY (single source of truth)
# =====================================================================
# Map a short stable LABEL -> the kwargs that select that estimator in
# bootstrap_measurements / measure_centroid. Imported by analysis.py and
# validation.py so the same label means the same thing in every table, plot,
# and stress-test. 'moments' is intentionally absent: its 1st moment IS the
# flux-weighted centroid (clipped), so as a CENTROID it is redundant -- use
# line_moments directly when you want the width/skewness, not a centroid.
CENTROID_METHOD_SPECS = {
    "fw_clip":        dict(centroid_method="flux_weighted", clip_negative=True),
    "fw_noclip":      dict(centroid_method="flux_weighted", clip_negative=False),
    "gauss_weighted": dict(centroid_method="gauss_weighted"),
    "gaussian":       dict(centroid_method="gaussian"),
    "flux_median":    dict(centroid_method="flux_median"),
    "peak":           dict(centroid_method="peak"),
}

CENTROID_METHOD_LABELS = {
    "fw_clip":        "flux-weighted (clipped)",
    "fw_noclip":      "flux-weighted (no clip)",
    "gauss_weighted": "Gaussian-weighted moment",
    "gaussian":       "Gaussian fit (µ)",
    "flux_median":    "50% cumulative",
    "peak":           "parabolic peak",
}

# Default subset used wherever a function compares "all the usual estimators".
CENTROID_METHODS_DEFAULT = tuple(CENTROID_METHOD_SPECS)


def measure_centroid(
    wave, flux, err=None,
    method=DEFAULT_CENTROID_METHOD,
    clip_negative=True,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    min_pixels=8,
    **method_kw,
):
    """
    One entry point for every centroid estimator. All share the return contract
    {success, reason, centroid_A, centroid_vel_kms, continuum, flux_contsub}
    (the 'moments' method additionally returns sigma_A, sigma_kms, skewness).

    method : 'flux_weighted' (uses clip_negative) | 'gaussian' | 'flux_median'
             | 'peak' | 'moments' | 'gauss_weighted'
             (aliases: fw, gauss, median50, parabola, mode, windowed, gwt)
    clip_negative : consulted by 'flux_weighted' and 'moments' only.
    method_kw     : forwarded to the chosen estimator (e.g. max_sigma_A for
                    'gaussian'; sigma_w / max_iter / tol for 'gauss_weighted').
    """
    key = str(method).lower()
    if key not in _CENTROID_DISPATCH:
        raise ValueError(f"unknown centroid method {method!r}; "
                         f"choose from {sorted(set(_CENTROID_DISPATCH))}")
    fn = _CENTROID_DISPATCH[key]
    common = dict(bounds=bounds, cont_bounds=cont_bounds, lya_center=lya_center,
                  cont_method=cont_method, cont_order=cont_order,
                  min_pixels=min_pixels)
    if fn in (flux_weighted_centroid, line_moments):
        return fn(wave, flux, err=err, clip_negative=clip_negative, **common)
    return fn(wave, flux, err=err, **common, **method_kw)


def integrated_line_flux(
    wave, flux, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    clip_negative=False,
):
    """
    Canonical integrated continuum-subtracted flux in a wavelength window.

    y = flux - continuum;  flux_sum = sum(y[in window]);  flux_integral = flux_sum * dlambda.
    This is the ONE integrator the rest of the pipeline routes through (the
    z-scramble null, the placebo null, the SN-spectrum line table, and -- by the
    same convention -- the blue/red side flux), so every "integrated flux"
    quantity uses one continuum model and one summation convention. Unlike the
    flux-weighted centroid this is a plain SUM, so it stays finite and roughly
    Gaussian under a null (no division by vanishing line flux).

    Note: with the standard half-window split, blue_red_side_ratio's
    (blue_flux + red_flux) equals this flux_sum (lya_center pixel counted once on
    the red side); flux_integral is that times the pixel width.

    Parameters
    ----------
    err : optional (nwave,) error array; if given, err_integral is the
          quadrature-summed 1-sigma over the window (times dlambda).
    clip_negative : zero negative pixels before summing (default False).

    Returns
    -------
    dict: success, flux_sum, flux_integral, err_sum, err_integral, npix, dwave,
          continuum, flux_contsub.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y = flux - cont
    if clip_negative:
        y = np.where(y > 0, y, 0.0)
    dw = float(np.nanmedian(np.diff(wave)))
    m = (wave >= bounds[0]) & (wave <= bounds[1]) & np.isfinite(y)
    if not np.any(m):
        return {"success": False, "reason": "Empty window",
                "flux_sum": np.nan, "flux_integral": np.nan,
                "err_sum": np.nan, "err_integral": np.nan, "npix": 0, "dwave": dw,
                "continuum": cont, "flux_contsub": y}
    fsum = np.nansum(y[m])
    esum = eint = np.nan
    if err is not None:
        err = np.asarray(err, dtype=float)
        me = m & np.isfinite(err)
        esum = float(np.sqrt(np.nansum(err[me] ** 2)))
        eint = esum * dw
    return {"success": True, "reason": None,
            "flux_sum": float(fsum), "flux_integral": float(fsum * dw),
            "err_sum": esum, "err_integral": eint, "npix": int(m.sum()),
            "dwave": dw, "continuum": cont, "flux_contsub": y}


def integrated_line_flux_per_bin(
    wave, stack, err=None,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    clip_negative=False,
):
    """
    integrated_line_flux, looped over every radial bin of ONE stack.

    Thin per-bin wrapper around the canonical integrator -- bootstrap_all already
    has this exact shape internally (its local `_measure_stack` closure loops
    over nrad calling measure_centroid + blue_red_side_ratio per bin) but that
    closure isn't exported and doesn't call integrated_line_flux. This is the
    reusable version, for any caller that has one restacked (nrad, nwave)
    spectrum -- the fiducial stack, or a single bootstrap draw -- and wants
    per-bin integrated flux without hand-rolling the loop. Used by
    optimize.py's line-S/N diagnostic so line-flux integration goes through the
    one convention (continuum-subtracted, config-driven window) everywhere.

    stack : (nrad, nwave) -- one restacked spectrum.
    err   : (nrad, nwave) or None -- propagated per-pixel error, if available.

    Returns (flux_sum, err_sum), each (nrad,) -- NaN in bins where
    integrated_line_flux reports success=False (e.g. empty window).
    """
    wave = np.asarray(wave, dtype=float)
    stack = np.asarray(stack, dtype=float)
    err = np.asarray(err, dtype=float) if err is not None else None
    nrad = stack.shape[0]
    flux_sum = np.full(nrad, np.nan)
    err_sum = np.full(nrad, np.nan)
    for r in range(nrad):
        e_r = err[r] if err is not None else None
        res = integrated_line_flux(
            wave, stack[r], err=e_r, bounds=bounds, cont_bounds=cont_bounds,
            lya_center=lya_center, cont_method=cont_method, cont_order=cont_order,
            clip_negative=clip_negative,
        )
        if res["success"]:
            flux_sum[r] = res["flux_sum"]
            err_sum[r] = res["err_sum"]
    return flux_sum, err_sum


def blue_red_side_ratio(
    wave, flux,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    clip_negative=False,
):
    """
    Coarse blue/red asymmetry: sum the continuum-subtracted flux on each side
    of lya_center within the line window and take the ratio.

    This is the simple, model-free companion to a (deferred) double-Gaussian
    decomposition -- it does not separate two emission peaks, it just compares
    integrated flux blueward vs redward of systemic.

    Convention: ratio = blue_flux / red_flux  ('blue to red', as requested);
    red_over_blue is also returned for convenience. With clip_negative=True the
    negative pixels are zeroed before summing (avoids sign flips from noisy
    continuum); default False sums the raw continuum-subtracted flux.

    Returns
    -------
    dict: success, blue_flux, red_flux, blue_over_red, red_over_blue, continuum.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y = flux - cont
    if clip_negative:
        y = np.where(y > 0, y, 0.0)

    blue_m = (wave >= bounds[0]) & (wave < lya_center) & np.isfinite(y)
    red_m = (wave >= lya_center) & (wave <= bounds[1]) & np.isfinite(y)

    if not np.any(blue_m) or not np.any(red_m):
        return {"success": False, "reason": "Empty blue or red side",
                "blue_flux": np.nan, "red_flux": np.nan,
                "blue_over_red": np.nan, "red_over_blue": np.nan,
                "continuum": cont}

    blue_flux = np.nansum(y[blue_m])
    red_flux = np.nansum(y[red_m])

    bor = blue_flux / red_flux if red_flux != 0 else np.nan
    rob = red_flux / blue_flux if blue_flux != 0 else np.nan

    return {"success": np.isfinite(bor),
            "reason": None,
            "blue_flux": blue_flux, "red_flux": red_flux,
            "blue_over_red": bor, "red_over_blue": rob,
            "continuum": cont}


# =====================================================================
# 2. STACK A RESAMPLED CUBE ALONG THE GALAXY AXIS
# =====================================================================

def stack_galaxies(flux_bs, err_bs=None, method="biweight", weights=None,
                   sigma=3.0, maxiters=5):
    """
    Combine a (ngal, nrad, nwave) cube along the galaxy axis.

    method in {'biweight', 'median', 'mean', 'inv_var', 'weighted_median',
    'sigma_clip'}. 'weighted_median' needs `weights` (ngal,) -- the per-galaxy
    widths (nfib), already resampled to match flux_bs's galaxy axis by the
    caller. 'sigma_clip' uses `sigma`/`maxiters` (astropy sigma_clip along the
    galaxy axis, median center + mad_std, then the mean of the survivors).
    Returns (stack, stack_err) each (nrad, nwave).
    """
    if method == "median":
        stack = np.nanmedian(flux_bs, axis=0)
    elif method == "mean":
        stack = np.nanmean(flux_bs, axis=0)
    elif method == "biweight":
        stack = biweight_location(flux_bs, axis=0, ignore_nan=True)
    elif method == "sigma_clip":
        from astropy.stats import sigma_clip as _sigma_clip
        clipped = _sigma_clip(flux_bs, sigma=sigma, maxiters=maxiters, axis=0,
                              cenfunc="median", stdfunc="mad_std", masked=True)
        stack = np.ma.mean(clipped, axis=0).filled(np.nan)
    elif method == "inv_var":
        if err_bs is None:
            raise ValueError("err_bs required for inverse-variance stacking")
        var = err_bs ** 2
        good = np.isfinite(flux_bs) & np.isfinite(var) & (var > 0)
        w = np.where(good, 1.0 / var, 0.0)
        denom = np.nansum(w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            stack = np.nansum(flux_bs * w, axis=0) / denom
        stack = np.where(denom > 0, stack, np.nan)
    elif method == "weighted_median":
        if weights is None:
            raise ValueError(...)
        from .extract import combine_fibers
        ngal, nrad, nwave = flux_bs.shape
        w_arr = np.asarray(weights, dtype=float)
        # accept (ngal,) [same width every bin] or (ngal, nrad) [per-bin widths]
        if w_arr.ndim == 1:
            w_arr = np.repeat(w_arr[:, None], nrad, axis=1)
        stack = np.full((nrad, nwave), np.nan)
        for rb in range(nrad):
            stack[rb], _ = combine_fibers(flux_bs[:, rb, :], method="weighted_median",
                                          weights=w_arr[:, rb])
    else:
        raise ValueError("method must be 'biweight', 'median', 'mean', 'inv_var', "
                         "'sigma_clip', or 'weighted_median'")

    n = flux_bs.shape[0]
    if err_bs is not None:
        stack_err = np.sqrt(np.nanmean(err_bs ** 2, axis=0)) / np.sqrt(n)
    else:
        stack_err = np.nanstd(flux_bs, axis=0) / np.sqrt(n)
    return stack, stack_err


# =====================================================================
# 3. GALAXY BOOTSTRAPS
# =====================================================================

def bootstrap_measurements(
    flux_rf, err_rf, wave, nboot=1000,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    stack_method="biweight",
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    centroid_method=DEFAULT_CENTROID_METHOD,
    clip_negative=True,
    compute_side_ratio=True,
    clip_negative_sides=False,
    seed=1, verbose=True,
    weights=None,                      # (ngal,) per-galaxy widths for weighted_median
    label=None, announce=True,
):
    """
    Galaxy-resampling bootstrap of the centroid (and, optionally, the blue/red
    side ratio) per radial bin.

    Each draw resamples galaxies WITH REPLACEMENT, re-stacks, and re-measures.
    Only cheap functionals are evaluated per draw (no curve_fit), so this is
    fast even at nboot=1000. The point estimate is taken on the full-sample
    ('fiducial') stack; the error is the 16/84 spread across draws.

    Parameters
    ----------
    flux_rf : (ngal, nrad, nwave)   per-galaxy rest-frame flux cube (Stage 2, keep_cube=True)
    err_rf  : (ngal, nrad, nwave) or None   matching error cube (needed for inv_var)
    wave    : (nwave,)              rest-frame wavelength grid
    nboot   : int                   number of bootstrap draws
    stack_method : galaxy combine used inside the bootstrap

    Returns
    -------
    summary : dict
        centroid_v_med / _lo / _hi    (nrad,)   bootstrap median, 16th, 84th pct (km/s)
        centroid_v_fid                (nrad,)   centroid on the full-sample stack (km/s)
        centroid_A_med / _lo / _hi    (nrad,)   same, in Angstrom
        centroid_v_all                (nboot,nrad)
        success_frac                  (nrad,)   fraction of draws with a valid centroid
        (if compute_side_ratio) blue_over_red_med/_lo/_hi, _fid, _all and
                                 blue_flux_med, red_flux_med
        meta: nboot, ngal, stack_method, cont_method, cont_order, bounds, cont_bounds
    """
    flux_rf = np.asarray(flux_rf, dtype=float)
    err_rf = np.asarray(err_rf, dtype=float) if err_rf is not None else None
    wave = np.asarray(wave, dtype=float)
    ngal, nrad, nwave = flux_rf.shape
    rng = np.random.default_rng(seed)

    centroid_A = np.full((nboot, nrad), np.nan)
    centroid_v = np.full((nboot, nrad), np.nan)
    success = np.zeros((nboot, nrad), dtype=bool)
    if compute_side_ratio:
        bor = np.full((nboot, nrad), np.nan)
        bflux = np.full((nboot, nrad), np.nan)
        rflux = np.full((nboot, nrad), np.nan)

    def _measure_stack(stack):
        cA = np.full(nrad, np.nan); cV = np.full(nrad, np.nan)
        ok = np.zeros(nrad, dtype=bool)
        bo = np.full(nrad, np.nan); bf = np.full(nrad, np.nan); rf = np.full(nrad, np.nan)
        for rb in range(nrad):
            res = measure_centroid(
                wave, stack[rb], method=centroid_method,
                clip_negative=clip_negative,
                bounds=bounds, cont_bounds=cont_bounds,
                lya_center=lya_center, cont_method=cont_method,
                cont_order=cont_order)
            ok[rb] = res["success"]
            cA[rb] = res["centroid_A"]; cV[rb] = res["centroid_vel_kms"]
            if compute_side_ratio:
                sr = blue_red_side_ratio(
                    wave, stack[rb], bounds=bounds, cont_bounds=cont_bounds,
                    lya_center=lya_center, cont_method=cont_method,
                    cont_order=cont_order, clip_negative=clip_negative_sides)
                bo[rb] = sr["blue_over_red"]; bf[rb] = sr["blue_flux"]; rf[rb] = sr["red_flux"]
        return cA, cV, ok, bo, bf, rf

    # fiducial (full-sample) point estimate
    fid_stack, _ = stack_galaxies(flux_rf, err_rf, method=stack_method, weights=weights)
    fcA, fcV, _, fbo, fbf, frf = _measure_stack(fid_stack)

    desc = run_header(
        label or f"bootstrap [{centroid_method}]", verbose=verbose, announce=announce,
        nboot=nboot, stack=stack_method, centroid=centroid_method,
        window=bounds, cont=cont_method, side_ratio=compute_side_ratio, seed=seed)
    w_all = None if weights is None else np.asarray(weights, dtype=float)
    for b in tqdm(range(nboot), disable=not verbose, desc=desc):
        idx = rng.integers(0, ngal, ngal)
        flux_bs = flux_rf[idx]
        err_bs = err_rf[idx] if err_rf is not None else None
        w_bs = w_all[idx] if w_all is not None else None
        stack_bs, _ = stack_galaxies(flux_bs, err_bs, method=stack_method, weights=w_bs)
        cA, cV, ok, bo, bf, rf = _measure_stack(stack_bs)
        centroid_A[b] = cA; centroid_v[b] = cV; success[b] = ok
        if compute_side_ratio:
            bor[b] = bo; bflux[b] = bf; rflux[b] = rf

    summary = {
        "centroid_v_med": np.nanmedian(centroid_v, axis=0),
        "centroid_v_lo": np.nanpercentile(centroid_v, 16, axis=0),
        "centroid_v_hi": np.nanpercentile(centroid_v, 84, axis=0),
        "centroid_v_fid": fcV,
        "centroid_A_med": np.nanmedian(centroid_A, axis=0),
        "centroid_A_lo": np.nanpercentile(centroid_A, 16, axis=0),
        "centroid_A_hi": np.nanpercentile(centroid_A, 84, axis=0),
        "centroid_A_fid": fcA,
        "centroid_v_all": centroid_v,
        "centroid_A_all": centroid_A,
        "success_frac": np.nanmean(success, axis=0),
        "meta": {"nboot": nboot, "ngal": ngal, "stack_method": stack_method,
                 "cont_method": cont_method, "cont_order": cont_order,
                 "centroid_method": centroid_method, "clip_negative": clip_negative,
                 "bounds": bounds, "cont_bounds": cont_bounds},
    }
    if compute_side_ratio:
        summary.update({
            "blue_over_red_med": np.nanmedian(bor, axis=0),
            "blue_over_red_lo": np.nanpercentile(bor, 16, axis=0),
            "blue_over_red_hi": np.nanpercentile(bor, 84, axis=0),
            "blue_over_red_fid": fbo,
            "blue_over_red_all": bor,
            "blue_flux_med": np.nanmedian(bflux, axis=0),
            "blue_flux_lo": np.nanpercentile(bflux, 16, axis=0),
            "blue_flux_hi": np.nanpercentile(bflux, 84, axis=0),
            "blue_flux_fid": fbf,
            "blue_flux_all": bflux,
            "red_flux_med": np.nanmedian(rflux, axis=0),
            "red_flux_lo": np.nanpercentile(rflux, 16, axis=0),
            "red_flux_hi": np.nanpercentile(rflux, 84, axis=0),
            "red_flux_fid": frf,
            "red_flux_all": rflux,
        })
        # total integrated (windowed) flux per draw = blue + red, so percentiles
        # are correct (cannot add blue/red percentiles). Matches the SUM
        # convention of measure.integrated_line_flux (flux_sum).
        tot = bflux + rflux
        summary.update({
            "total_flux_med": np.nanmedian(tot, axis=0),
            "total_flux_lo": np.nanpercentile(tot, 16, axis=0),
            "total_flux_hi": np.nanpercentile(tot, 84, axis=0),
            "total_flux_fid": fbf + frf,
            "total_flux_all": tot,
        })
    return summary


def bootstrap_stack_error(
    flux_rf, err_rf=None, nboot=1000, stack_method="biweight",
    robust=True, seed=1, verbose=True, return_draws=False, weights=None,
    label=None, announce=True,
):
    """
    Per-pixel 1-sigma flux error of the stack from galaxy resampling.

    Resamples galaxies with replacement, re-stacks each draw, and takes the
    pixel-by-pixel spread across draws. This is an EMPIRICAL stack error that
    does not assume the input (propagated) errors are correct -- so it is the
    natural thing to compare against the propagated stack error, and to use for
    honest shaded bands on the spectrum panels.

    Parameters
    ----------
    flux_rf : (ngal, nrad, nwave)   per-galaxy rest-frame flux cube
    err_rf  : (ngal, nrad, nwave) or None   needed only for stack_method='inv_var'
    robust  : if True, sigma = 0.5*(p84 - p16) (robust to outlier draws);
              if False, sigma = std across draws.

    Returns
    -------
    out : dict
        stack_med   (nrad, nwave)   median stack across draws
        stack_err   (nrad, nwave)   per-pixel 1-sigma (robust or std per `robust`)
        stack_err_std    (nrad, nwave)
        stack_err_p1684  (nrad, nwave)
        p16, p84    (nrad, nwave)   percentile bands
        meta: nboot, ngal, stack_method, robust
        (if return_draws) draws (nboot, nrad, nwave)   -- can be large

    Memory note: holds (nboot, nrad, nwave) float64 internally to take
    percentiles. For nboot=1000, nrad=7, nwave=1600 that is ~90 MB. Lower nboot
    or process bins in batches if memory-bound.
    """
    flux_rf = np.asarray(flux_rf, dtype=float)
    err_rf = np.asarray(err_rf, dtype=float) if err_rf is not None else None
    ngal, nrad, nwave = flux_rf.shape
    rng = np.random.default_rng(seed)

    desc = run_header(
        label or "bootstrap per-pixel stack error", verbose=verbose, announce=announce,
        nboot=nboot, stack=stack_method, robust=robust, seed=seed)
    draws = np.empty((nboot, nrad, nwave), dtype=np.float64)
    w_all = None if weights is None else np.asarray(weights, dtype=float)
    for b in tqdm(range(nboot), disable=not verbose, desc=desc):
        idx = rng.integers(0, ngal, ngal)
        flux_bs = flux_rf[idx]
        err_bs = err_rf[idx] if err_rf is not None else None
        w_bs = w_all[idx] if w_all is not None else None
        stack_bs, _ = stack_galaxies(flux_bs, err_bs, method=stack_method, weights=w_bs)
        draws[b] = stack_bs

    stack_med = np.nanmedian(draws, axis=0)
    p16 = np.nanpercentile(draws, 16, axis=0)
    p84 = np.nanpercentile(draws, 84, axis=0)
    err_p1684 = 0.5 * (p84 - p16)
    err_std = np.nanstd(draws, axis=0)
    stack_err = err_p1684 if robust else err_std

    out = {
        "stack_med": stack_med,
        "stack_err": stack_err,
        "stack_err_std": err_std,
        "stack_err_p1684": err_p1684,
        "p16": p16, "p84": p84,
        "meta": {"nboot": nboot, "ngal": ngal,
                 "stack_method": stack_method, "robust": robust},
    }
    if return_draws:
        out["draws"] = draws
    return out


# =====================================================================
# Legacy stubs (kept so pipeline.run_measure import does not break).
# Use the functions above directly from a notebook for now.
# =====================================================================

def _measurement_window(bounds, cont_bounds, pad=2.0):
    """
    Minimal wavelength span any Stage-3 functional touches: the union of the
    line window and both continuum sidebands, padded by `pad` A. Slicing the
    cube to this before the bootstrap loop is a pure speedup -- centroids and
    the per-pixel error are unchanged because nothing reads outside it.
    """
    (cb_lo, _), (_, cb_hi) = cont_bounds
    lo = min(cb_lo, bounds[0]) - pad
    hi = max(cb_hi, bounds[1]) + pad
    return lo, hi


def bootstrap_all(
    flux_rf, err_rf, wave, nboot=1000,
    bounds=(LYA_REST - 4, LYA_REST + 4),
    cont_bounds=DEFAULT_CONT_BOUNDS,
    lya_center=LYA_REST,
    stack_method="biweight",
    cont_method=DEFAULT_CONT_METHOD, cont_order=DEFAULT_CONT_ORDER,
    centroid_method=DEFAULT_CENTROID_METHOD,
    clip_negative=True,
    compute_side_ratio=True,
    clip_negative_sides=False,
    compute_stack_error=True,
    robust_stack_error=True,
    seed=1, verbose=True,
    weights=None,
    sigma_clip_sigma=3.0, sigma_clip_maxiters=5,
    label=None, announce=True,
):
    """
    ONE-PASS Stage-3 bootstrap: centroid (+ blue/red side ratio) AND the
    per-pixel stack error, from a SINGLE galaxy-resampling loop on a cube
    sliced to the measurement window.

    This is exactly equivalent to calling bootstrap_measurements and
    bootstrap_stack_error separately with the same seed (both consume the same
    draw sequence), but it re-stacks each draw ONCE instead of twice and only
    over the ~200 px measurement window instead of the full rest grid. For the
    biweight combine that window slice alone is the dominant speedup, because
    astropy's biweight scales worse than linearly in nwave.

    Returns the bootstrap_measurements summary dict, plus:
        stack_error : dict like bootstrap_stack_error's output but on the SLICED
                      grid (stack_err, stack_err_std, stack_err_p1684, p16, p84,
                      stack_med, error_wave), or None if compute_stack_error=False.
    """
    flux_rf = np.asarray(flux_rf, dtype=float)
    err_rf = np.asarray(err_rf, dtype=float) if err_rf is not None else None
    wave = np.asarray(wave, dtype=float)
    ngal, nrad, nwave_full = flux_rf.shape

    # --- slice the cube to the measurement window (pure speedup) ---
    w_lo, w_hi = _measurement_window(bounds, cont_bounds)
    sel = (wave >= w_lo) & (wave <= w_hi)
    wv = wave[sel]
    fcube = flux_rf[:, :, sel]
    ecube = err_rf[:, :, sel] if err_rf is not None else None

    rng = np.random.default_rng(seed)

    centroid_A = np.full((nboot, nrad), np.nan)
    centroid_v = np.full((nboot, nrad), np.nan)
    success = np.zeros((nboot, nrad), dtype=bool)
    if compute_side_ratio:
        bor = np.full((nboot, nrad), np.nan)
        bflux = np.full((nboot, nrad), np.nan)
        rflux = np.full((nboot, nrad), np.nan)
    draws = (np.empty((nboot, nrad, wv.size), dtype=np.float64)
             if compute_stack_error else None)

    def _measure_stack(stack):
        cA = np.full(nrad, np.nan); cV = np.full(nrad, np.nan)
        ok = np.zeros(nrad, dtype=bool)
        bo = np.full(nrad, np.nan); bf = np.full(nrad, np.nan); rf = np.full(nrad, np.nan)
        for rb in range(nrad):
            res = measure_centroid(
                wv, stack[rb], method=centroid_method,
                clip_negative=clip_negative,
                bounds=bounds, cont_bounds=cont_bounds,
                lya_center=lya_center, cont_method=cont_method,
                cont_order=cont_order)
            ok[rb] = res["success"]
            cA[rb] = res["centroid_A"]; cV[rb] = res["centroid_vel_kms"]
            if compute_side_ratio:
                sr = blue_red_side_ratio(
                    wv, stack[rb], bounds=bounds, cont_bounds=cont_bounds,
                    lya_center=lya_center, cont_method=cont_method,
                    cont_order=cont_order, clip_negative=clip_negative_sides)
                bo[rb] = sr["blue_over_red"]; bf[rb] = sr["blue_flux"]; rf[rb] = sr["red_flux"]
        return cA, cV, ok, bo, bf, rf

    # fiducial (full-sample) point estimate -- consumes no rng, matching
    # bootstrap_measurements so the per-draw sequence is identical.
    fid_stack, _ = stack_galaxies(fcube, ecube, method=stack_method, weights=weights,
                                  sigma=sigma_clip_sigma, maxiters=sigma_clip_maxiters)
    fcA, fcV, _, fbo, fbf, frf = _measure_stack(fid_stack)

    desc = run_header(
        label or f"bootstrap [{centroid_method}]", verbose=verbose, announce=announce,
        nboot=nboot, stack=stack_method, centroid=centroid_method,
        window=bounds, cont=cont_method, side_ratio=compute_side_ratio,
        stack_err=compute_stack_error, seed=seed)
    w_all = None if weights is None else np.asarray(weights, dtype=float)
    for b in tqdm(range(nboot), disable=not verbose, desc=desc):
        idx = rng.integers(0, ngal, ngal)
        flux_bs = fcube[idx]
        err_bs = ecube[idx] if ecube is not None else None
        w_bs = w_all[idx] if w_all is not None else None
        stack_bs, _ = stack_galaxies(flux_bs, err_bs, method=stack_method, weights=w_bs,
                                     sigma=sigma_clip_sigma, maxiters=sigma_clip_maxiters)
        if draws is not None:
            draws[b] = stack_bs
        cA, cV, ok, bo, bf, rf = _measure_stack(stack_bs)
        centroid_A[b] = cA; centroid_v[b] = cV; success[b] = ok
        if compute_side_ratio:
            bor[b] = bo; bflux[b] = bf; rflux[b] = rf

    summary = {
        "centroid_v_med": np.nanmedian(centroid_v, axis=0),
        "centroid_v_lo": np.nanpercentile(centroid_v, 16, axis=0),
        "centroid_v_hi": np.nanpercentile(centroid_v, 84, axis=0),
        "centroid_v_fid": fcV,
        "centroid_A_med": np.nanmedian(centroid_A, axis=0),
        "centroid_A_lo": np.nanpercentile(centroid_A, 16, axis=0),
        "centroid_A_hi": np.nanpercentile(centroid_A, 84, axis=0),
        "centroid_A_fid": fcA,
        "centroid_v_all": centroid_v,
        "centroid_A_all": centroid_A,
        "success_frac": np.nanmean(success, axis=0),
        "meta": {"nboot": nboot, "ngal": ngal, "stack_method": stack_method,
                 "cont_method": cont_method, "cont_order": cont_order,
                 "centroid_method": centroid_method, "clip_negative": clip_negative,
                 "bounds": bounds, "cont_bounds": cont_bounds,
                 "sliced_window": (float(wv[0]), float(wv[-1]))},
    }
    if compute_side_ratio:
        summary.update({
            "blue_over_red_med": np.nanmedian(bor, axis=0),
            "blue_over_red_lo": np.nanpercentile(bor, 16, axis=0),
            "blue_over_red_hi": np.nanpercentile(bor, 84, axis=0),
            "blue_over_red_fid": fbo,
            "blue_over_red_all": bor,
            "blue_flux_med": np.nanmedian(bflux, axis=0),
            "blue_flux_lo": np.nanpercentile(bflux, 16, axis=0),
            "blue_flux_hi": np.nanpercentile(bflux, 84, axis=0),
            "blue_flux_fid": fbf,
            "blue_flux_all": bflux,
            "red_flux_med": np.nanmedian(rflux, axis=0),
            "red_flux_lo": np.nanpercentile(rflux, 16, axis=0),
            "red_flux_hi": np.nanpercentile(rflux, 84, axis=0),
            "red_flux_fid": frf,
            "red_flux_all": rflux,
        })
        tot = bflux + rflux
        summary.update({
            "total_flux_med": np.nanmedian(tot, axis=0),
            "total_flux_lo": np.nanpercentile(tot, 16, axis=0),
            "total_flux_hi": np.nanpercentile(tot, 84, axis=0),
            "total_flux_fid": fbf + frf,
            "total_flux_all": tot,
        })

    if compute_stack_error:
        # Computed only on the sliced window (that is the only region the error
        # is used), but returned on the FULL rest grid, NaN-padded outside the
        # window, so shapes stay (nrad, nwave_full) for existing plotting code.
        def _pad(a):
            full = np.full((nrad, nwave_full), np.nan, dtype=float)
            full[:, sel] = a
            return full
        p16 = np.nanpercentile(draws, 16, axis=0)
        p84 = np.nanpercentile(draws, 84, axis=0)
        err_p1684 = 0.5 * (p84 - p16)
        err_std = np.nanstd(draws, axis=0)
        summary["stack_error"] = {
            "stack_med": _pad(np.nanmedian(draws, axis=0)),
            "stack_err": _pad(err_p1684 if robust_stack_error else err_std),
            "stack_err_std": _pad(err_std),
            "stack_err_p1684": _pad(err_p1684),
            "p16": _pad(p16), "p84": _pad(p84),
            "error_wave": wave,             # full grid (matches padded arrays)
            "meta": {"nboot": nboot, "ngal": ngal,
                     "stack_method": stack_method, "robust": robust_stack_error,
                     "sliced_window": (float(wv[0]), float(wv[-1]))},
        }
    else:
        summary["stack_error"] = None
    return summary


def measure_all_bins(config: "PipelineConfig", stacks: dict,
                     stack_method: str = None, compute_stack_error: bool = True,
                     verbose: bool = True) -> dict:
    """
    Stage 3, config-driven, on the Stage-2 stacks dict.

    Requires the per-galaxy cube -- run Stage 2 with run_stack(..., keep_cube=True)
    (the default). Bootstraps the centroid (+ blue/red side ratio) and, unless
    compute_stack_error=False, the per-pixel stack error, both from the cube.

    Config drives it: line_window -> centroid bounds, n_bootstrap -> draws,
    LYA_REST -> systemic. Continuum (cont_bounds, cont_method, cont_order),
    centroid estimator (centroid_method, clip_negative), and the measure-stack
    method (measure_stack_method) are all read from config, falling back to the
    module DEFAULT_* (imported from config) for older configs that predate those
    fields. stack_method, if passed, overrides config.measure_stack_method.

    Returns the bootstrap_measurements summary augmented with:
        stack_error  : the bootstrap_stack_error dict (per-pixel), or None
        rest_wave, r_edges, VR_biweight_v, unit_info   (so it is directly
                       plottable: pass it as both boot and stacks_result).
    """
    if "cube_flux" not in stacks or "cube_err" not in stacks:
        raise KeyError(
            "Stage 3 needs the per-galaxy cube. Re-run Stage 2 with "
            "run_stack(config, ..., keep_cube=True) (the default)."
        )
    flux_rf = np.asarray(stacks["cube_flux"], dtype=float)
    err_rf = np.asarray(stacks["cube_err"], dtype=float)
    wave = np.asarray(stacks["rest_wave"], dtype=float)

    bounds = tuple(config.line_window)
    nboot = int(config.n_bootstrap)
    lya = float(config.LYA_REST)
    sm = stack_method or getattr(config, "measure_stack_method", "biweight")
    cont_bounds = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
    cont_method = getattr(config, "cont_method", DEFAULT_CONT_METHOD)
    cont_order = getattr(config, "cont_order", DEFAULT_CONT_ORDER)
    centroid_method = getattr(config, "centroid_method", DEFAULT_CENTROID_METHOD)
    clip_negative = getattr(config, "clip_negative", True)
    sc_sigma = getattr(config, "sigma_clip_sigma", 3.0)
    sc_maxiters = getattr(config, "sigma_clip_maxiters", 5)

    cube_w = stacks.get("cube_weights")            # (ngal, nrad) or None
    use_w = cube_w if sm == "weighted_median" else None

    # Single merged, window-sliced bootstrap: centroid (+ side ratio) AND the
    # per-pixel stack error from ONE loop. Numerically identical to the former
    # two-loop path (bootstrap_measurements + bootstrap_stack_error) at the same
    # seed, but re-stacks each draw once over the ~200 px measurement window
    # instead of twice over the full rest grid -- the dominant Stage-3 speedup.
    results = bootstrap_all(
        flux_rf, err_rf, wave, nboot=nboot, bounds=bounds,
        cont_bounds=cont_bounds, lya_center=lya, stack_method=sm,
        cont_method=cont_method, cont_order=cont_order,
        centroid_method=centroid_method, clip_negative=clip_negative,
        compute_side_ratio=True, compute_stack_error=compute_stack_error,
        verbose=verbose, weights=use_w,
        sigma_clip_sigma=sc_sigma, sigma_clip_maxiters=sc_maxiters,
    )

    # carry the axes/scale so the result is directly plottable
    results["rest_wave"] = wave
    results["r_edges"] = np.asarray(stacks["r_edges"])
    results["VR_biweight_v"] = stacks.get("VR_biweight_v")
    results["unit_info"] = stacks.get("unit_info")
    from .config import stamp_provenance
    return stamp_provenance(results, config, "measure_all_bins")


# =====================================================================
# Single-galaxy companion to measure_all_bins (no bootstrap -- one object
# can't be resampled over galaxies). Per-bin centroid + integrated flux on a
# stack.single_galaxy_spectra product, for eyeballing a bright object's
# centroid(r) or the red-peak shift of a double-peaker, bin by bin.
# =====================================================================
def measure_single_galaxy(sg: dict, *, method=DEFAULT_CENTROID_METHOD,
                          bounds=None, cont_bounds=DEFAULT_CONT_BOUNDS) -> dict:
    """
    Per-bin centroid velocity and integrated flux for the single-galaxy product
    from stack.single_galaxy_spectra. NO bootstrap; the error bar per bin is the
    propagated flux error, and sn comes straight from sg['sn'] (computed at
    build time in config.line_window). Only trust bins with decent sn.

    sg     : the dict from stack.single_galaxy_spectra.
    bounds : line window (rest-A); defaults to LYA_REST +/- 4.

    Returns dict(r_edges, centroid_v, flux, flux_err, sn, method), all (nrad,).
    """
    rest_wave = np.asarray(sg["rest_wave"])
    flux_all = np.asarray(sg["stacks"]["galaxy"]["flux"])
    err_all = np.asarray(sg["stacks"]["galaxy"]["err"])
    r_edges = np.asarray(sg["r_edges"])
    nrad = flux_all.shape[0]
    b = tuple(bounds) if bounds is not None else (LYA_REST - 4.0, LYA_REST + 4.0)

    cen_v = np.full(nrad, np.nan)
    fx = np.full(nrad, np.nan)
    fxe = np.full(nrad, np.nan)
    for r in range(nrad):
        c = measure_centroid(rest_wave, flux_all[r], err=err_all[r],
                             method=method, bounds=b, cont_bounds=cont_bounds)
        if c.get("success"):
            cen_v[r] = c["centroid_vel_kms"]
        fres = integrated_line_flux(rest_wave, flux_all[r], err=err_all[r],
                                    bounds=b, cont_bounds=cont_bounds)
        if fres["success"]:
            fx[r] = fres["flux_sum"]
            fxe[r] = fres["err_sum"]

    return {"r_edges": r_edges, "centroid_v": cen_v, "flux": fx,
           "flux_err": fxe, "sn": np.asarray(sg["sn"]), "method": method}


# =====================================================================
# 4. DERIVED RADIAL DIAGNOSTICS (post-bootstrap)
# =====================================================================

def flux_curve_of_growth(
    boot: dict,
    stacks: dict | None = None,
    r_max_kpc: float | None = None,
    drop_last_bin: bool = True,
    area_combine: str = "biweight",
    verbose: bool = True,
) -> dict:
    """
    Cumulative Lya luminosity and flux fraction vs radius ("curve of growth"),
    built on top of a measure_all_bins / bootstrap_all summary dict.

    Converts the already-bootstrapped windowed flux per bin (total_flux_fid =
    blue_flux_fid + red_flux_fid, in whatever y_unit the stack is in -- L_kpc2
    by default, i.e. a Lya surface-density) back into a luminosity by
    multiplying by a representative fiber footprint area (fiber_area_kpc2, a
    per-galaxy (ngal,) array stashed in unit_info by stack.convert_avg_fiber_bin
    -- it depends only on each galaxy's redshift, not on the radial bin, so ONE
    representative value applies at every radius), then cumulatively sums bins
    from the center outward.

    total_flux_fid * area = the luminosity an average fiber would capture at
    that radius, NOT a true annulus-integrated (2*pi*r*dr) total -- this is a
    fiber-footprint curve of growth, consistent with how every other radial
    quantity in this pipeline is defined (one average-fiber measurement per
    bin), not a full azimuthally-integrated light profile.

    Parameters
    ----------
    boot : dict
        Output of measure_all_bins / bootstrap_all / bootstrap_measurements.
        Needs total_flux_fid (and, for error bands, total_flux_all) plus
        r_edges and unit_info -- measure_all_bins already carries all three
        through from `stacks`, so passing `stacks` again is usually optional.
    stacks : dict, optional
        Falls back here for r_edges / unit_info if not present on `boot`
        (e.g. boot came from the lower-level bootstrap_measurements/bootstrap_all
        instead of measure_all_bins).
    r_max_kpc : float or None
        Outer radius (kpc) to sum out to. Must land exactly on a bin edge in
        r_edges (bins are discrete; a fractional bin isn't well-defined here).
        None (default) -> governed by `drop_last_bin` instead.
    drop_last_bin : bool
        When r_max_kpc is None, True (default) sums out to r_edges[-2],
        i.e. EXCLUDES the outermost bin (its bootstrap errors and area both
        blow up fastest, so it's a poor default contributor to a cumulative
        total). False includes every bin out to r_edges[-1].
    area_combine : {'biweight', 'median', 'mean'}
        How the per-galaxy fiber_area_kpc2 array is reduced to the single
        representative area multiplied into every bin. Default 'biweight'
        matches stack_galaxies' default galaxy-combine method, so the area is
        consistent with how the flux itself was stacked.
    verbose : bool
        Emit warnings (see below). Set False to silence them (the values
        returned are identical either way).

    Bad bins (warn, then propagate as measured -- this IS the measurement, not
    hidden): a NaN total_flux_fid bin makes every cumulative bin from that
    point outward NaN (plain np.cumsum, not nan-safe, by design); a negative
    net-flux bin (noise-driven, most often the faint outer bins) makes the
    cumulative curve locally non-monotonic. Both are warned about but left in
    the output uncorrected.

    Returns
    -------
    dict
        r_edges_used         (n_bins+1,)  bin edges actually summed over
        flux_bin_fid          (n_bins,)    per-bin luminosity (total_flux_fid * area)
        flux_cumulative_fid   (n_bins,)    running total, center -> r_max
        flux_cumulative_lo/_hi (n_bins,)   16/84 bootstrap band (NaN if
                                            total_flux_all unavailable)
        flux_cumulative_all   (nboot, n_bins) or None
        flux_fraction_fid      (n_bins,)   flux_cumulative_fid / its last value
        flux_fraction_lo/_hi   (n_bins,)   16/84 band, computed PER DRAW
                                            (cum_draw / cum_draw[-1]) then
                                            percentiled -- NOT a ratio of the
                                            cumulative percentiles
        flux_fraction_all      (nboot, n_bins) or None
        fiber_area_kpc2_used   float        the representative area applied
        unit_info              dict         passed through, for the y-unit label
        meta                   dict         area_combine, n_bins, r_max_kpc,
                                             drop_last_bin, n_nan_bins,
                                             n_negative_bins
    """
    if "total_flux_fid" not in boot:
        raise KeyError("boot does not contain total_flux_fid; re-run the bootstrap with "
                       "compute_side_ratio=True (the default).")

    r_edges = np.asarray(
        boot.get("r_edges", (stacks or {}).get("r_edges")), dtype=float)
    if r_edges.size == 0:
        raise KeyError("no r_edges found on boot or stacks.")

    unit_info = boot.get("unit_info") or (stacks or {}).get("unit_info") or {}
    fiber_area_kpc2 = unit_info.get("fiber_area_kpc2")
    if fiber_area_kpc2 is None:
        raise KeyError(
            "unit_info has no fiber_area_kpc2 -- flux_curve_of_growth needs the "
            "stack built with flux_unit='L_kpc2' (the default), which is the "
            "only output mode that records the per-galaxy fiber footprint.")
    fiber_area_kpc2 = np.asarray(fiber_area_kpc2, dtype=float)

    if area_combine == "biweight":
        area = float(biweight_location(fiber_area_kpc2, ignore_nan=True))
    elif area_combine == "median":
        area = float(np.nanmedian(fiber_area_kpc2))
    elif area_combine == "mean":
        area = float(np.nanmean(fiber_area_kpc2))
    else:
        raise ValueError("area_combine must be 'biweight', 'median', or 'mean'")

    total_flux_fid = np.asarray(boot["total_flux_fid"], dtype=float)
    nrad = total_flux_fid.size
    flux_bin_fid_full = total_flux_fid * area

    # --- resolve how many bins to sum ---
    if r_max_kpc is not None:
        idx = np.where(np.isclose(r_edges, r_max_kpc, rtol=1e-3, atol=1e-6))[0]
        if idx.size == 0:
            raise ValueError(
                f"r_max_kpc={r_max_kpc!r} does not match any edge in r_edges="
                f"{r_edges.tolist()}; pass an exact bin edge, or leave "
                f"r_max_kpc=None and use drop_last_bin instead.")
        n_bins = int(idx[0])
    else:
        n_bins = (nrad - 1) if drop_last_bin else nrad
    if n_bins < 1:
        raise ValueError(
            f"n_bins resolved to {n_bins} (<1); check r_max_kpc/drop_last_bin "
            f"against r_edges={r_edges.tolist()}.")

    sl = slice(0, n_bins)
    r_edges_used = r_edges[: n_bins + 1]
    flux_bin_fid = flux_bin_fid_full[sl]

    # --- warn (not hide) on bad bins within the summed range ---
    n_nan = int(np.sum(~np.isfinite(flux_bin_fid)))
    n_neg = int(np.sum(np.isfinite(flux_bin_fid) & (flux_bin_fid < 0)))
    if verbose and n_nan:
        warnings.warn(
            f"flux_curve_of_growth: {n_nan} bin(s) within r_max have a NaN "
            f"total_flux_fid; the cumulative curve is NaN from that bin outward "
            f"(propagated, not masked).")
    if verbose and n_neg:
        warnings.warn(
            f"flux_curve_of_growth: {n_neg} bin(s) within r_max have negative "
            f"net flux (noise-driven, typically the faint outer bins); the "
            f"cumulative curve may be locally non-monotonic -- this reflects "
            f"the actual measurement and is left uncorrected.")

    flux_cumulative_fid = np.cumsum(flux_bin_fid)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_fraction_fid = flux_cumulative_fid / flux_cumulative_fid[-1]

    # --- bootstrap bands: cumsum/fraction PER DRAW, then percentile ---
    flux_cumulative_all = flux_fraction_all = None
    flux_cumulative_lo = np.full(n_bins, np.nan)
    flux_cumulative_hi = np.full(n_bins, np.nan)
    flux_fraction_lo = np.full(n_bins, np.nan)
    flux_fraction_hi = np.full(n_bins, np.nan)
    if "total_flux_all" in boot:
        total_flux_all = np.asarray(boot["total_flux_all"], dtype=float)  # (nboot, nrad)
        flux_bin_all = total_flux_all[:, sl] * area
        flux_cumulative_all = np.cumsum(flux_bin_all, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_fraction_all = flux_cumulative_all / flux_cumulative_all[:, -1][:, None]
        flux_cumulative_lo = np.nanpercentile(flux_cumulative_all, 16, axis=0)
        flux_cumulative_hi = np.nanpercentile(flux_cumulative_all, 84, axis=0)
        flux_fraction_lo = np.nanpercentile(flux_fraction_all, 16, axis=0)
        flux_fraction_hi = np.nanpercentile(flux_fraction_all, 84, axis=0)
    elif verbose:
        warnings.warn(
            "flux_curve_of_growth: boot has no total_flux_all; returning the "
            "fiducial curve only (no bootstrap error bands).")

    return {
        "r_edges_used": r_edges_used,
        "flux_bin_fid": flux_bin_fid,
        "flux_cumulative_fid": flux_cumulative_fid,
        "flux_cumulative_lo": flux_cumulative_lo,
        "flux_cumulative_hi": flux_cumulative_hi,
        "flux_cumulative_all": flux_cumulative_all,
        "flux_fraction_fid": flux_fraction_fid,
        "flux_fraction_lo": flux_fraction_lo,
        "flux_fraction_hi": flux_fraction_hi,
        "flux_fraction_all": flux_fraction_all,
        "fiber_area_kpc2_used": area,
        "unit_info": unit_info,
        "meta": {
            "area_combine": area_combine,
            "n_bins": n_bins,
            "r_max_kpc": float(r_edges_used[-1]),
            "drop_last_bin": drop_last_bin,
            "n_nan_bins": n_nan,
            "n_negative_bins": n_neg,
        },
    }
