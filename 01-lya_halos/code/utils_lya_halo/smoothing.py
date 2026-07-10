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
