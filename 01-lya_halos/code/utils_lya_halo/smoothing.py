"""
smoothing.py -- background smoothers for Stage 1.

The per-exposure / galaxy-level background is smoothed before being subtracted
from every science fiber (an unsmoothed background injects annulus shot-noise
into all science spectra). extract.extract_fibers picks the smoother by
config.bg_smooth_method:

  'interp_gauss' (DEFAULT) -> smooth_interp_then_gauss
       Mask the Lya line, LINEARLY interpolate across the gap, then one uniform
       gaussian_filter1d. Order matters: interpolating first removes the line
       from the data, so the smoother sees no gap and no line -- no division-fill
       bowing, no pulse artifact, no wing-anchoring. Fast (C-level gaussian, no
       NaN-safe per-pixel division): ~100x cheaper than the adaptive smoother,
       which is what made background smoothing a multi-hour bottleneck on the
       big extractions.

  'adaptive' (LEGACY)      -> smooth_spectrum_nan_safe_adaptive
       NaN-safe gaussian with a position-dependent kernel that ramps from a
       small edge sigma to a large interior sigma. Kept for A/B comparison.

  smooth_fill_then_gauss  -- DIAGNOSTIC / Stage-2 ONLY, not in the
       config.bg_smooth_method dispatch above. Generalizes smooth_interp_then_
       gauss's fill step to three candidate methods ('linear' == the
       production behavior, 'median', 'median_interp') so they can be A/B'd
       against each other before any of them becomes a production default.
       Driven by background.compare_fill_methods / plot_fill_method_gallery.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_interp_then_gauss(flux, wave, z, lya_rest=1215.67, sigma_A=25.0,
                             mask_halfwidth_A=20.0, mode="reflect"):
    """
    Background smoother: mask the Lya line, LINEARLY interpolate across the gap,
    THEN run a single uniform Gaussian smooth over the (now gap-free) spectrum.

    Interpolating first physically removes the line from the data, so the
    smoother sees no gap and no line -- no division-fill bowing, no pulse
    artifact, no wing-anchoring. A plain fixed-width gaussian_filter1d then does
    the smoothing (fast, C-level, no NaN-safe division needed).

      flux, wave       : (nwave,) background for one galaxy (wave may carry
                         astropy units; .value is used)
      z                : that galaxy's systemic redshift; the line is masked at
                         the OBSERVED Lya, lya_rest*(1+z). Non-finite z -> no
                         line mask (degrades to a plain gaussian smooth).
      sigma_A          : uniform smoothing width (A)
      mask_halfwidth_A : half-width (observed A) of the masked line region,
                         interpolated across
      mode             : gaussian_filter1d boundary mode
    Returns the smoothed background, same shape.
    """
    flux = np.asarray(flux, dtype=float).copy()
    wave = np.asarray(getattr(wave, "value", wave), dtype=float)

    lam = lya_rest * (1 + z)
    line = np.abs(wave - lam) <= mask_halfwidth_A          # all-False if z is NaN
    good = np.isfinite(flux)
    fill = good & ~line                                    # clean anchor pixels
    if fill.sum() >= 2:                                    # linear-interp gap (+ NaNs)
        flux[~fill] = np.interp(wave[~fill], wave[fill], flux[fill])
    elif not np.any(good):
        return flux                                        # nothing to smooth

    dw = np.nanmedian(np.diff(wave))
    sig_pix = sigma_A / dw
    return gaussian_filter1d(flux, sig_pix, mode=mode)


def smooth_spectrum_nan_safe_adaptive(flux, wave, sigma_A=200, edge_sigma_A=20,
                                      edge_width_A=None):
    """
    NaN-safe Gaussian smooth with a position-dependent kernel width.

    The kernel sigma ramps from edge_sigma_A at the spectrum edges up to sigma_A
    in the interior over edge_width_A Angstrom (default 2*sigma_A), so the noisy
    ends aren't over-smoothed across the boundaries. At each pixel the weighted
    average uses only finite neighbours, so NaN gaps don't bleed into the result.

    flux, wave   : (nwave,) arrays (wave may carry astropy units; .value is used)
    sigma_A      : interior kernel sigma (Angstrom)
    edge_sigma_A : edge kernel sigma (Angstrom)
    edge_width_A : ramp length (Angstrom); None -> 2*sigma_A
    Returns the smoothed flux, same shape, NaN where no finite neighbours exist.
    """
    flux = np.asarray(flux, dtype=float)
    wave = np.asarray(wave.value if hasattr(wave, "value") else wave, dtype=float)

    dw = np.nanmedian(np.diff(wave))
    if edge_width_A is None:
        edge_width_A = 2 * sigma_A

    dist_left = wave - np.nanmin(wave)
    dist_right = np.nanmax(wave) - wave
    dist_edge = np.minimum(dist_left, dist_right)

    # ramp from edge_sigma_A at the edges to sigma_A in the interior
    t = np.clip(dist_edge / edge_width_A, 0, 1)
    sigma_local_A = edge_sigma_A + (sigma_A - edge_sigma_A) * t

    smooth_flux = np.full_like(flux, np.nan)
    good = np.isfinite(flux)

    for i in range(len(flux)):
        sig_pix = sigma_local_A[i] / dw
        half_width = int(np.ceil(4 * sig_pix))
        lo = max(0, i - half_width)
        hi = min(len(flux), i + half_width + 1)

        x = np.arange(lo, hi)
        weights = np.exp(-0.5 * ((x - i) / sig_pix) ** 2)
        valid = good[lo:hi]
        if np.any(valid):
            smooth_flux[i] = (np.nansum(flux[lo:hi][valid] * weights[valid]) /
                              np.nansum(weights[valid]))
    return smooth_flux


def _patch_other_gaps(out, wave, protect):
    """
    In-place: linearly interpolate any NaNs OUTSIDE `protect` (the line mask)
    so a downstream gaussian smooth has no gaps left to propagate through.
    Used by the median-based fill methods below, which only ever set values
    INSIDE the line mask and would otherwise leave unrelated bad pixels (e.g.
    a dead fiber contribution) as NaN straight through to the gaussian.
    """
    good = np.isfinite(out)
    gap = ~good & ~protect
    if gap.any() and good.sum() >= 2:
        out[gap] = np.interp(wave[gap], wave[good], out[good])


def smooth_fill_then_gauss(flux, wave, z, lya_rest=1215.67, sigma_A=30.0,
                           mask_halfwidth_A=10.0, fill_method="linear",
                           fill_sideband_width_A=30.0, mode="reflect"):
    """
    DIAGNOSTIC / Stage-2 ONLY -- see the module docstring. Mask the Lya line,
    FILL the gap with one of three methods, then run the same single uniform
    Gaussian smooth production uses (skipped entirely if sigma_A <= 0, which
    is how the comparison tools get the "filled, not yet smoothed" stage).

      fill_method = "linear"        : np.interp between the two pixels
                                       immediately at the mask edges. Identical
                                       to smooth_interp_then_gauss's fill step
                                       (this IS that function, generalized).
                                       Sensitive to those two individual,
                                       possibly-noisy pixels, and assumes
                                       whatever local slope they happen to
                                       imply.
                   = "median"       : the ENTIRE gap filled with ONE flat
                                       value -- the median of BOTH flanking
                                       sidebands pooled together. Most robust
                                       to noise, but flattens any real slope
                                       across the gap.
                   = "median_interp": median of EACH sideband taken
                                       SEPARATELY, then np.interp between
                                       those two medians, anchored at the mask
                                       edges -- same span as "linear", so it
                                       keeps "linear"'s ability to track a
                                       real slope across the gap, but each
                                       endpoint is a robust
                                       ~fill_sideband_width_A-wide median
                                       instead of one noisy pixel. If only one
                                       side has enough points it falls back to
                                       a flat fill from that side.

    fill_sideband_width_A : width (observed A) of each flanking sideband used
        by "median" and "median_interp", starting right at the mask edge.
        mask_halfwidth_A=10 (production default), fill_sideband_width_A=30
        -> sidebands at [-40,-10] and [10,40] A from line center.

    NO CONTINUUM MODEL anywhere in this function. These are flat/linear local
    levels for filling a masked gap in the BACKGROUND itself, not a fit for
    line-flux measurement -- that machinery (measure.get_continuum_model,
    cont_method / cont_order) is a separate, science-measurement-only tool
    with no business here.

      flux, wave : (nwave,) background for one galaxy (wave may carry astropy
                   units; .value is used)
      z          : that galaxy's systemic redshift; the line is masked at the
                   OBSERVED Lya, lya_rest*(1+z). Non-finite z -> no line mask,
                   just the gaussian (if sigma_A > 0) or a no-op.
    Returns the processed background, same shape.
    """
    flux = np.asarray(flux, dtype=float).copy()
    wave = np.asarray(getattr(wave, "value", wave), dtype=float)
    dw = np.nanmedian(np.diff(wave))

    if not np.isfinite(z):
        return flux if sigma_A <= 0 else gaussian_filter1d(flux, sigma_A / dw, mode=mode)

    lam = lya_rest * (1 + z)
    d = wave - lam
    line = np.abs(d) <= mask_halfwidth_A
    good = np.isfinite(flux)
    out = flux.copy()

    left_side = good & (d < -mask_halfwidth_A) & (d >= -(mask_halfwidth_A + fill_sideband_width_A))
    right_side = good & (d > mask_halfwidth_A) & (d <= (mask_halfwidth_A + fill_sideband_width_A))

    if fill_method == "linear":
        fill = good & ~line
        if fill.sum() >= 2:
            out[~fill] = np.interp(wave[~fill], wave[fill], flux[fill])
    elif fill_method == "median":
        sideband = left_side | right_side
        if sideband.sum() >= 3:
            out[line] = np.nanmedian(flux[sideband])
        _patch_other_gaps(out, wave, line)
    elif fill_method == "median_interp":
        if left_side.sum() >= 2 and right_side.sum() >= 2:
            y_lo, y_hi = np.nanmedian(flux[left_side]), np.nanmedian(flux[right_side])
            x_lo, x_hi = lam - mask_halfwidth_A, lam + mask_halfwidth_A
            out[line] = y_lo + (y_hi - y_lo) * (wave[line] - x_lo) / (x_hi - x_lo)
        elif left_side.sum() >= 2 or right_side.sum() >= 2:
            side = left_side if left_side.sum() >= 2 else right_side
            out[line] = np.nanmedian(flux[side])   # only one side available -- flat fallback
        _patch_other_gaps(out, wave, line)
    else:
        raise ValueError("fill_method must be 'linear', 'median', or 'median_interp'")

    return out if sigma_A <= 0 else gaussian_filter1d(out, sigma_A / dw, mode=mode)
