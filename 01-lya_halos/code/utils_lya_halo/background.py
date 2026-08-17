"""
background.py -- offline testing of the background subtraction.

WHAT THIS MODULE IS FOR, in one paragraph. The extraction destroys the fiber
dimension: `GalaxyProduct.spec` is already background-subtracted and the
annulus fibers no longer exist, so the pipeline can never re-derive a
background from a different annulus, a different mask width, or a different
smoothing -- those choices are baked in at extraction time and permanently.
A run with `store_spec_nobg` + `store_raw_bg` breaks that: it stores the
science coadd BEFORE subtraction and the annulus coadds BEFORE the Lya mask
and the gaussian smooth, and every background question becomes a free,
repeatable Stage-2 sweep. This module is the Stage-2 half of that.

    reconstruct_background   raw annulus coadd -> mask, interpolate, smooth
    reconstruct_spec         SPEC_NOBG - reconstruct_background(variant)
    reconstruct_product      the same, as a drop-in GalaxyProduct for build_stacks
    verify_reconstruction    the identity check -- run this FIRST, always

    stack_backgrounds        rest / observed / z-scrambled stack of the RAW
                             backgrounds, with a galaxy bootstrap
    measure_line_residual    is there a Lya feature at 1215.67 A, and at what
                             significance
    adopt_mask_halfwidth     the pre-registered criterion of 2.5.1
    background_half_difference   N-half minus S-half: the background's SPATIAL
                             variance (B10/B11)
    describe_background_test one call that runs and prints the whole 1.1 test

    bg_variant_radii         the annuli in arcsec AND kpc, SORTED BY RADIUS,
                             with the science bins each one overlaps
    plot_background_gallery  look at individual raw backgrounds before
                             computing anything on them
    compare_bg_variants      THE MAIN EVENT: the radial map of the background --
                             line amplitude AND non-line level vs annulus radius
    mask_width_scan          2D (mask half-width x smoothing sigma) residual scan
    background_variance_profile   galaxy-to-galaxy vs spatial scatter (B10/B11)
    profile_impact           what actually reaches the paper: the binned flux
                             profile AND the centroid profile under each
                             background choice

    compare_fill_methods     optimizing the mask/FILL/smooth chain itself (not
                             the annulus choice): raw vs filled vs smoothed,
                             one figure per candidate fill method, each with
                             its own galaxy-bootstrap error band. NO continuum
                             model -- that belongs to line-flux measurement,
                             not background reconstruction.
    plot_fill_method_gallery per-galaxy version of the same comparison, one
                             panel per galaxy, y-axis set from the ZOOMED
                             window only (fixes the huge-ylim-from-edge-spikes
                             problem an un-zoomed autoscale has)

WHAT THIS MODULE IS TRYING TO ACHIEVE, since it is easy to lose. Because the
extraction reaches to several Mpc, the background annulus unavoidably contains
real Lya. The job is NOT to find a signal-free annulus -- there may not be one
at a usable radius. The job is to SURGICALLY REMOVE the Lya that is present
while leaving a background that is still a valid estimate of the systematic
fluctuations (sky residuals, scattered light, instrumental pedestal) it exists
to remove. Both halves matter: a 60 A mask and a huge sigma will drive the line
residual to zero and leave a background that can no longer do its job. Every
knob here is a point on that trade-off.

WHAT THESE DIAGNOSTICS ARE JUDGED ON. Bin values and plots, not fitted
parameters. The paper's primary science is the measured signal and the
spectroscopic centroid out to several Mpc; the profile fit is a secondary
product for literature comparison. So `profile_impact` reports per-bin flux and
per-bin centroid shifts in units of the fiducial's own error bar, and
deliberately does NOT report a change in h1 -- on a coarse bin grid that number
would be noise wearing a result's clothes. See
specs/background-diagnostics.md 0.4.

THE ONE THING TO KNOW ABOUT THE STORED ARRAYS. `BACKGROUND` (always written) is
post-mask and post-smooth: any Lya feature in the annulus has already been
interpolated across and gaussian-smoothed away in it, by construction. Testing
for that feature in `BACKGROUND` is therefore circular and will always return
"nothing there". `BG_VARIANTS` is the same fibers before any of that, and is
the array every function here that tests for a line reads.

WHY THE SUBTRACT-OFFLINE TRICK IS EXACT rather than approximate: the biweight
location estimator is translation-equivariant -- subtracting a per-wavelength
scalar from every fiber shifts the combined result by exactly that scalar --
and (with `min_bg_fibers` larger than the fiber count, which extract.py
asserts) every exposure shares one pooled background. So subtracting after the
fiber combine gives the same numbers as subtracting before it. The residual
disagreement is float32 rounding on the stored background, ~1e-7 relative,
which is what `verify_reconstruction`'s default tolerance allows for.

See specs/systematics-and-extraction.md 1.1, 1.1b, 2.2, 2.5, and
specs/background-diagnostics.md (the Stage-2 toolkit this module implements).
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter1d

from .config import PipelineConfig
from .extract import combine_fibers
from .measure import get_continuum_model
from .virial import cosmo          # Planck18 -- the single cosmology source

# The background SMOOTHERS (smooth_interp_then_gauss, smooth_fill_then_gauss,
# smooth_spectrum_nan_safe_adaptive) were merged into this module from the old
# smoothing.py on 2026-08-03 -- see the "BACKGROUND SMOOTHERS" section at the
# bottom of the file. They live here rather than in extract.py because this
# module already owns every other background choice, and because extract.py is
# imported ABOVE: defining them here means extract.py has to reach for them
# lazily (it does, inside extract_fibers._maybe_smooth) instead of at module
# scope, which would be a circular import.

LYA_REST = PipelineConfig.LYA_REST

# Continuum sidebands for the BACKGROUND stack -- deliberately WIDER and further
# out than the science default ((1195,1205),(1225,1236)). The feature this test
# looks for is the two-halo Lya component, whose velocity extent is set by
# clustering along the line of sight and is expected to be BROAD (2.2: +/-740
# km/s is only +/-10.5 cMpc, well inside the correlation function's reach). The
# science sidebands sit 10-20 A from line center, so a broad feature would leak
# into them and be partly subtracted out as "continuum" -- the test would then
# under-report the very thing it exists to detect. These sidebands start ~25 A
# out on each side instead.
BG_CONT_BOUNDS = ((1150.0, 1190.0), (1245.0, 1290.0))


# =====================================================================
# RECONSTRUCTION  --  swap the background choice after the fact
# =====================================================================

def reconstruct_background(bg_raw, wave, z, config=None, *,
                           mask_halfwidth_A=None, sigma_A=None,
                           smooth_bg=None, bg_smooth_method=None,
                           fill_method=None, fill_sideband_width_A=30.0):
    """
    Apply the background smoother to ONE galaxy's raw annulus coadd.

    This is the operation `extract._maybe_smooth` performs inside the
    extraction, lifted out so it can be re-run at any mask width or smoothing
    sigma without re-extracting. Passing a config reproduces that run's
    settings exactly; the keyword overrides are what a sweep varies.

      bg_raw           : (nwave,) RAW annulus coadd, i.e. BG_VARIANTS[i, v]
      wave             : (nwave,) OBSERVED wavelength grid
      z                : that galaxy's redshift (the line is masked at the
                         observed Lya, LYA_REST*(1+z))
      mask_halfwidth_A : half-width of the masked gap, OBSERVED Angstrom.
                         None -> config.line_mask_halfwidth_A
      sigma_A          : gaussian sigma, Angstrom. None -> config.bg_smooth_sigma_A
      fill_method      : None (DEFAULT) -> the production path exactly,
                         `smooth_interp_then_gauss`. Set to "linear",
                         "median" or "median_interp" to route through the
                         diagnostic `smooth_fill_then_gauss` instead, which is
                         what a fill-method sweep varies.

                         "linear" IS the production fill, so
                         fill_method="linear" and fill_method=None must give
                         the same array to floating-point noise -- that
                         equivalence is worth asserting once (see
                         compare_background_impact) rather than assumed, since
                         everything downstream differences against it.

      fill_sideband_width_A : sideband width used by "median" /
                         "median_interp" only. Ignored otherwise.

    Returns (nwave,), same convention as the stored BACKGROUND HDU.
    """
    if config is None:
        config = PipelineConfig()
    smooth = config.smooth_bg if smooth_bg is None else smooth_bg
    if not smooth:
        return np.asarray(bg_raw, dtype=float)

    if fill_method is not None:
        return smooth_fill_then_gauss(
            bg_raw, wave, z, lya_rest=config.LYA_REST,
            sigma_A=(config.bg_smooth_sigma_A if sigma_A is None else sigma_A),
            mask_halfwidth_A=(config.line_mask_halfwidth_A
                              if mask_halfwidth_A is None else mask_halfwidth_A),
            fill_method=fill_method,
            fill_sideband_width_A=fill_sideband_width_A,
        )

    method = (bg_smooth_method or getattr(config, "bg_smooth_method", "interp_gauss")).lower()
    if method == "interp_gauss":
        return smooth_interp_then_gauss(
            bg_raw, wave, z, lya_rest=config.LYA_REST,
            sigma_A=(config.bg_smooth_sigma_A if sigma_A is None else sigma_A),
            mask_halfwidth_A=(config.line_mask_halfwidth_A
                              if mask_halfwidth_A is None else mask_halfwidth_A),
        )
    sv = config.smoothing_values                      # legacy 'adaptive'
    return smooth_spectrum_nan_safe_adaptive(
        bg_raw, wave, sigma_A=sv[0], edge_sigma_A=sv[1], edge_width_A=sv[2]
    )


def reconstruct_spec(product, config=None, *, variant=0,
                     mask_halfwidth_A=None, sigma_A=None, smooth_bg=None,
                     fill_method=None, fill_sideband_width_A=30.0):
    """
    Rebuild the background-subtracted science cube under a DIFFERENT background
    choice: `SPEC_NOBG - reconstruct_background(BG_VARIANTS[:, variant])`.

      variant : index into the BG_VARIANTS axis. 0 is the fiducial annulus (the
                one the extraction actually subtracted), so variant=0 with no
                overrides must reproduce `product.spec` -- that is exactly what
                verify_reconstruction checks.

    Returns (ngal, nrad, nwave). Requires a product written with both
    store_spec_nobg and store_raw_bg.

    Note on errors: `product.err` folds the background error in quadrature and
    is NOT reconstructible this way. That is acceptable because errors come
    from the galaxy bootstrap (1.3, and err_sum should never reach a table) --
    but it is a real limitation, not an oversight, so the ERROR array is passed
    through unchanged rather than silently recomputed as if it tracked.
    """
    if config is None:
        config = PipelineConfig()
    if product.spec_nobg is None or product.bg_variants is None:
        raise ValueError(
            "reconstruct_spec needs SPEC_NOBG and BG_VARIANTS. Re-extract with "
            "store_spec_nobg=True and store_raw_bg=True (see 2.5); the plain "
            "BACKGROUND HDU cannot substitute -- it is already masked and "
            "smoothed, so subtracting it again would double-count."
        )

    z = np.asarray(product.catalog[config.z_col], dtype=float)
    wave = np.asarray(product.wave, dtype=float)
    out = np.array(product.spec_nobg, dtype=float, copy=True)

    for i in range(out.shape[0]):
        bg = reconstruct_background(
            product.bg_variants[i, variant], wave, z[i], config,
            mask_halfwidth_A=mask_halfwidth_A, sigma_A=sigma_A,
            smooth_bg=smooth_bg, fill_method=fill_method,
            fill_sideband_width_A=fill_sideband_width_A,
        )
        out[i] -= bg[np.newaxis, :]
    return out


def reconstruct_product(product, config=None, **kwargs):
    """
    `reconstruct_spec` wrapped back into a GalaxyProduct, so a background
    variant can be pushed straight through the normal Stage 2:

        p2 = reconstruct_product(p, cfg, variant=1)
        stacks = stack.build_stacks(cfg, p2, keep_cube=True)

    That two-liner is what makes 2.2's joint mask-width x smoothing-sigma sweep
    a loop instead of a re-extraction. Everything except `spec` is shared with
    the original product (no copy), so a sweep over many settings costs one
    science cube each and nothing else.
    """
    from dataclasses import replace
    return replace(product, spec=reconstruct_spec(product, config, **kwargs))


def verify_reconstruction(product, config=None, *, tol=None, tol_factor=64.0,
                          verbose=True):
    """
    THE CHECK TO RUN FIRST, before trusting anything else in this module.

    Asserts `SPEC_NOBG - smooth(mask(BG_VARIANTS[0])) == SPEC`. If this fails,
    one of the two assumptions the whole offline-subtraction scheme rests on is
    broken -- either the estimator is not translation-equivariant as used, or
    some exposure carried its own background (min_bg_fibers too low) -- and
    every background variant computed downstream is wrong. It is a cheap test
    of an expensive assumption.

    ON THE TOLERANCE, because getting it wrong makes this test useless in
    either direction. The disagreement is float32 rounding: the background is
    stored float32 and re-smoothed in float64, so the error is set by the
    magnitude of the BACKGROUND (a large, positive sky level), not by the
    magnitude of SPEC (a small, background-subtracted residual that crosses
    zero). A relative tolerance against SPEC would therefore flag every pixel
    where SPEC happens to be near zero -- which is most of them -- while a
    fixed absolute tolerance would not transfer between flux units. So the
    default is an ABSOLUTE tolerance derived from the data:

        tol = tol_factor * eps_float32 * max|SPEC_NOBG|

    i.e. a small multiple of the representation error of the largest stored
    value. Pass `tol` explicitly to override. A failure at 1e-2 of the
    background level means something structural, not rounding.

    Returns a dict: max_abs_diff, tol, n_bad, ok, nan_mismatch.
    """
    if config is None:
        config = PipelineConfig()
    rebuilt = reconstruct_spec(product, config, variant=0)
    ref = np.asarray(product.spec, dtype=float)

    both = np.isfinite(rebuilt) & np.isfinite(ref)
    diff = np.abs(rebuilt - ref)[both]

    if tol is None:
        scale = float(np.nanmax(np.abs(np.asarray(product.spec_nobg, dtype=float))))
        tol = tol_factor * float(np.finfo(np.float32).eps) * max(scale, 1.0)

    n_bad = int(np.sum(diff > tol))
    res = {
        "max_abs_diff": float(np.nanmax(diff)) if diff.size else np.nan,
        "tol": float(tol),
        "n_compared": int(both.sum()),
        "n_bad": n_bad,
        # NaN patterns must agree too: a reconstruction that quietly turns a
        # finite pixel into NaN (or vice versa) would pass a value comparison
        # over the intersection while changing which pixels enter every stack.
        "nan_mismatch": int(np.sum(np.isfinite(rebuilt) != np.isfinite(ref))),
    }
    res["ok"] = (n_bad == 0) and (res["nan_mismatch"] == 0)
    if verbose:
        print(f"[verify_reconstruction] {'OK' if res['ok'] else 'FAILED'}: "
              f"max|diff| = {res['max_abs_diff']:.3e} "
              f"(tol {res['tol']:.3e}), "
              f"{res['n_bad']}/{res['n_compared']} pixels over tolerance, "
              f"{res['nan_mismatch']} NaN-pattern mismatches")
        if not res["ok"]:
            print("  -> SPEC_NOBG cannot be trusted. Check that min_bg_fibers "
                  "exceeded the fiber count for this run (header MINBGFIB) and "
                  "that BG_VARIANTS[0] is the fiducial annulus (header BGVAR0).")
    return res


# =====================================================================
# STACKING THE RAW BACKGROUNDS  --  the 1.1 test
# =====================================================================

def _resample_to_rest(bg, wave_obs, z, rest_grid):
    """One galaxy's background onto a common rest grid. NaN outside coverage."""
    rest = wave_obs / (1.0 + z)
    good = np.isfinite(bg) & np.isfinite(rest)
    if good.sum() < 2:
        return np.full(rest_grid.shape, np.nan)
    out = np.interp(rest_grid, rest[good], bg[good], left=np.nan, right=np.nan)
    # np.interp clamps rather than extrapolating unless left/right are given;
    # with them it returns NaN outside, which is what we want -- a galaxy whose
    # observed grid does not reach a rest wavelength must not contribute there.
    return out


def stack_backgrounds(product, config=None, *, variant=0, frame="rest",
                      rest_grid=None, combine="biweight", n_boot=1000,
                      continuum=True, cont_bounds=BG_CONT_BOUNDS,
                      cont_method="poly", cont_order=1, seed=0):
    """
    Stack the RAW (pre-mask, pre-smooth) background spectra and look for a line.

    This is 1.1's direct test. If the 80-90" annulus is Lya-free, the rest-frame
    stack shows nothing at 1215.67 A. If it carries a two-halo component, a
    feature appears -- and its WIDTH says directly whether the fiducial +/-10 A
    observed mask is wide enough, which is the number 2.2 needs.

    frame:
      'rest'      -- each galaxy shifted by its OWN z. A real Lya feature adds
                     coherently here.
      'observed'  -- no shift at all. The control: a real Lya feature is spread
                     over the sample's redshift range and washes out, while an
                     instrumental or sky artifact sits at a fixed observed
                     wavelength and survives. A feature in BOTH frames is not
                     astrophysical.
      'scrambled' -- each galaxy shifted by ANOTHER galaxy's z (a derangement).
                     The stronger control: identical grid, identical estimator,
                     identical noise, only the galaxy-redshift association
                     broken. Anything left is not a line at the systemic
                     redshift.

    continuum:  subtract a local continuum from each galaxy's spectrum before
                stacking. Almost always yes -- the raw background is dominated
                by the sky/continuum level, whose galaxy-to-galaxy variation
                would otherwise swamp a small line feature. Note the sidebands
                default to BG_CONT_BOUNDS, which are WIDER than the science
                ones on purpose (see that constant's comment).

    Returns a dict:
      grid        (ngrid,)  wavelength axis (rest A, or observed A for 'observed')
      stack       (ngrid,)  combined background
      err         (ngrid,)  galaxy-bootstrap 1-sigma
      per_galaxy  (ngal, ngrid) the resampled, continuum-subtracted inputs
      n_eff       (ngrid,)  galaxies contributing per pixel
      frame, variant, annulus, z_med, ngal
    """
    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError(
            "stack_backgrounds needs the RAW backgrounds (BG_VARIANTS). The "
            "BACKGROUND HDU is post-mask and post-smooth, so stacking it to "
            "look for a masked line would be circular by construction -- it "
            "would find nothing no matter what is really there. Re-extract "
            "with store_raw_bg=True."
        )

    frame = frame.lower()
    if frame not in ("rest", "observed", "scrambled"):
        raise ValueError("frame must be 'rest', 'observed' or 'scrambled'")

    wave_obs = np.asarray(product.wave, dtype=float)
    z = np.asarray(product.catalog[config.z_col], dtype=float)
    bg = np.asarray(product.bg_variants[:, variant, :], dtype=float)
    ngal = bg.shape[0]
    z_med = float(np.nanmedian(z))

    rng = np.random.default_rng(seed)
    if frame == "scrambled":
        # A derangement, not a plain shuffle: a galaxy keeping its own z would
        # contribute a real, correctly-positioned line to the null.
        order = rng.permutation(ngal)
        for _ in range(100):
            if not np.any(order == np.arange(ngal)):
                break
            order = rng.permutation(ngal)
        z_use = z[order]
    else:
        z_use = z

    if frame == "observed":
        grid = wave_obs
        per_gal = bg.copy()
    else:
        if rest_grid is None:
            # Step = the NATIVE rest sampling (observed pixel / (1+z_med)), not
            # something finer. This matters: measure_line_residual sums the
            # bootstrap errors in quadrature as if pixels were independent, so
            # oversampling the grid would inflate every significance in this
            # module by ~sqrt(oversampling factor) -- the same
            # noise-underestimate fingerprint 1.3 flags for rest_delta=0.2
            # against a ~0.6 A native sampling. Sampling at the native step
            # keeps the pixels approximately independent and the significances
            # honest. 1140-1300 covers the line, both wide sidebands, and
            # enough wing to see a broad feature fall off.
            dw_rest = float(np.nanmedian(np.diff(wave_obs))) / (1.0 + z_med)
            rest_grid = np.arange(1140.0, 1300.0 + 1e-9, dw_rest)
        grid = np.asarray(rest_grid, dtype=float)
        per_gal = np.empty((ngal, grid.size), dtype=float)
        for i in range(ngal):
            per_gal[i] = _resample_to_rest(bg[i], wave_obs, z_use[i], grid)

    if continuum:
        for i in range(ngal):
            if np.all(~np.isfinite(per_gal[i])):
                continue
            cb = cont_bounds
            if frame == "observed":
                # sidebands are defined in rest A; move them to the sample's
                # median observed frame so the control uses the same windows
                cb = tuple((lo * (1 + z_med), hi * (1 + z_med)) for lo, hi in cont_bounds)
            per_gal[i] = per_gal[i] - get_continuum_model(
                grid, per_gal[i], cont_bounds=cb,
                cont_method=cont_method, cont_order=cont_order)

    weights = None
    if combine == "weighted_median":
        if product.bg_nfib is None:
            raise ValueError("combine='weighted_median' needs BG_NFIB "
                             "(store_raw_bg or store_bg_halves).")
        weights = np.asarray(product.bg_nfib[:, variant], dtype=float)

    stack, _ = combine_fibers(per_gal, None, method=combine, weights=weights)

    # Bootstrap over GALAXIES -- the same resampling unit the rest of the
    # pipeline uses for errors, so this significance is comparable to every
    # other error bar in the paper rather than being its own convention.
    boot = np.empty((n_boot, grid.size), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, ngal, ngal)
        w = weights[idx] if weights is not None else None
        boot[b], _ = combine_fibers(per_gal[idx], None, method=combine, weights=w)
    err = np.nanstd(boot, axis=0)

    annuli = product.bg_annuli
    return {
        "grid": grid,
        "stack": stack,
        "err": err,
        "per_galaxy": per_gal,
        "n_eff": np.sum(np.isfinite(per_gal), axis=0),
        "frame": frame,
        "variant": variant,
        "annulus": annuli[variant] if variant < len(annuli) else None,
        "z_med": z_med,
        "ngal": ngal,
        "combine": combine,
        "continuum_subtracted": bool(continuum),
    }


def measure_line_residual(stacked, *, center=LYA_REST, halfwidth_A=3.0,
                          observed_frame_center=None):
    """
    Integrated residual in a window around the line, with its bootstrap error.

    `halfwidth_A` is always REST Angstrom -- the units the science line window
    is quoted in, so a call here and the pipeline's +/-3 A mean the same thing.
    For an 'observed' stack both the center and the half-width are converted
    with the sample's median (1+z), so the control integrates a physically
    equivalent window rather than one (1+z) times too narrow. Getting that
    wrong on a coarse grid silently selects zero pixels and returns NaN.

    The error treats the bootstrap per-pixel sigmas as independent. They are
    approximately so on the default grid (stack_backgrounds samples at the
    native rest step for exactly this reason) but not exactly, so read a
    marginal result (< ~3) as "no detection" rather than as a weak one.

    Returns dict: integral, err, significance, peak, peak_err, n_pix, window.
    """
    grid = np.asarray(stacked["grid"], dtype=float)
    y = np.asarray(stacked["stack"], dtype=float)
    e = np.asarray(stacked["err"], dtype=float)

    c = center if observed_frame_center is None else observed_frame_center
    hw = halfwidth_A
    if stacked["frame"] == "observed":
        if observed_frame_center is None:
            c = center * (1.0 + stacked["z_med"])
        hw = halfwidth_A * (1.0 + stacked["z_med"])

    sel = np.abs(grid - c) <= hw
    sel &= np.isfinite(y) & np.isfinite(e)
    if not np.any(sel):
        return {"integral": np.nan, "err": np.nan, "significance": np.nan,
                "peak": np.nan, "peak_err": np.nan, "n_pix": 0,
                "window": (c - hw, c + hw)}

    dx = float(np.nanmedian(np.diff(grid)))
    integral = float(np.nansum(y[sel]) * dx)
    err = float(np.sqrt(np.nansum(e[sel] ** 2)) * dx)
    k = int(np.nanargmax(np.where(sel, y, -np.inf)))
    return {
        "integral": integral,
        "err": err,
        "significance": integral / err if err > 0 else np.nan,
        "peak": float(y[k]),
        "peak_err": float(e[k]),
        "peak_wave": float(grid[k]),
        "n_pix": int(sel.sum()),
        "window": (c - hw, c + hw),
    }


def adopt_mask_halfwidth(stacked, *, halfwidths_A=(10.0, 15.0, 20.0, 25.0, 30.0),
                         anchor_width_A=20.0, threshold=1.0, verbose=True):
    """
    The PRE-REGISTERED CRITERION of 2.5.1, implemented literally.

        Adopt the smallest line_mask_halfwidth_A for which the rest-frame
        stacked background shows no residual feature at 1215.67 A above its own
        noise.

    Operationally the quantity that matters is not the flux inside the mask --
    that gets interpolated over regardless -- but the flux in the ANCHOR
    WINDOW just outside it, because `np.interp` fills the gap from exactly the
    pixels at the mask edge (1.1b). If those pixels sit on the wing of a broad
    two-halo component, the interpolated level is elevated and the line is
    over-subtracted. So for each candidate half-width this measures the
    residual in [HW, HW + anchor_width] on both sides and asks whether it is
    consistent with zero.

    WHY THIS IS NOT CIRCULAR, and it is worth being able to say so out loud:
    the quantity measured here (a feature in the background ANNULUS) and the
    quantity the paper reports (the radial profile of the SCIENCE bins) are
    different measurements on different fibers, and the criterion makes no
    reference to the profile's shape, amplitude, h1, or crossover radius. The
    one thing that WOULD make it circular is choosing the mask width by which
    value makes h1 come out best -- so do not do that, and do not run this
    after looking at the profiles it would produce.

    `halfwidths_A` are OBSERVED Angstrom (the units of the config knob); the
    stack is in rest Angstrom, so they are converted with the sample's median
    (1+z), which is recorded in the result.

    Returns dict: rows (one per half-width), adopted, threshold.
    """
    if stacked["frame"] == "observed":
        raise ValueError("adopt_mask_halfwidth expects a rest-frame stack; the "
                         "observed-frame stack is the control, not the test.")

    grid = np.asarray(stacked["grid"], dtype=float)
    y = np.asarray(stacked["stack"], dtype=float)
    e = np.asarray(stacked["err"], dtype=float)
    one_plus_z = 1.0 + stacked["z_med"]
    dx = float(np.nanmedian(np.diff(grid)))

    rows = []
    for hw_obs in halfwidths_A:
        hw = hw_obs / one_plus_z                      # observed A -> rest A
        aw = anchor_width_A / one_plus_z
        d = np.abs(grid - LYA_REST)
        sel = (d >= hw) & (d < hw + aw) & np.isfinite(y) & np.isfinite(e)
        if not np.any(sel):
            rows.append({"halfwidth_obs_A": hw_obs, "halfwidth_rest_A": hw,
                         "anchor_flux": np.nan, "anchor_err": np.nan,
                         "significance": np.nan, "n_pix": 0, "clean": False})
            continue
        val = float(np.nansum(y[sel]) * dx)
        err = float(np.sqrt(np.nansum(e[sel] ** 2)) * dx)
        sig = val / err if err > 0 else np.nan
        rows.append({
            "halfwidth_obs_A": float(hw_obs),
            "halfwidth_rest_A": float(hw),
            "anchor_flux": val,
            "anchor_err": err,
            "significance": sig,
            "n_pix": int(sel.sum()),
            "clean": bool(np.isfinite(sig) and abs(sig) < threshold),
        })

    adopted = next((r["halfwidth_obs_A"] for r in rows if r["clean"]), None)

    if verbose:
        print(f"[adopt_mask_halfwidth] anchor window = mask edge -> +{anchor_width_A:g} A "
              f"observed, criterion |residual| < {threshold:g} sigma, "
              f"1+z_med = {one_plus_z:.3f}")
        print(f"  {'HW_obs':>8} {'HW_rest':>8} {'anchor':>12} {'+/-':>10} "
              f"{'sigma':>7}  clean")
        for r in rows:
            print(f"  {r['halfwidth_obs_A']:8.1f} {r['halfwidth_rest_A']:8.2f} "
                  f"{r['anchor_flux']:12.4e} {r['anchor_err']:10.3e} "
                  f"{r['significance']:7.2f}  {'yes' if r['clean'] else 'NO'}")
        if adopted is None:
            print("  -> NO candidate half-width leaves a clean anchor window. "
                  "Either widen the range, or the annulus carries a component "
                  "broader than any mask will exclude -- in which case the mask "
                  "is the wrong tool and 1.1b's window-anchor or "
                  "normalized-convolution fill is the fix.")
        else:
            print(f"  -> adopt line_mask_halfwidth_A = {adopted:g} "
                  "(smallest width with a clean anchor window)")
    return {"rows": rows, "adopted": adopted, "threshold": threshold,
            "anchor_width_A": anchor_width_A, "one_plus_z": one_plus_z}


# =====================================================================
# SPATIAL VARIANCE OF THE BACKGROUND  (B10 / B11)
# =====================================================================

def background_half_difference(product, config=None, *, variant=0, n_boot=1000,
                               seed=0, verbose=True):
    """
    N-half minus S-half of the background annulus, stacked over galaxies.

    The two halves are disjoint fiber sets drawn from the same annulus with the
    same estimator, so their difference isolates how much the background varies
    ACROSS the annulus -- the spatial term. Nothing in Stage 2 can recover this
    from the pooled background, which is why it has to be written at extraction
    time.

    Reads it against the annulus's own noise: if the halves agree within the
    per-galaxy scatter, a single pooled background per galaxy is adequate and
    B10/B11's monotonic rise-and-fall of background level with radius is NOT a
    spatial-gradient effect. If they disagree, the pooled background is an
    average over a real gradient and the subtraction leaves a radius-dependent
    residual -- which is the shape B10/B11 describes.

    THE STATISTIC is the per-pixel z-score `diff / diff_err` summarised as
    chi2/dof and median|z|, NOT a ratio of RMS values. Those are different
    quantities -- the RMS of the stacked difference over wavelength includes
    any real spectral structure, while the bootstrap error is a galaxy-to-galaxy
    scatter -- and dividing one by the other gives a number that looks like a
    significance without being one.

    CAVEAT WORTH READING BEFORE INTERPRETING A LARGE VALUE. The error here is a
    bootstrap over GALAXIES, so it measures how much the half-difference varies
    from galaxy to galaxy. Where galaxies are close enough on the sky that
    their 80-90" annuli overlap, they are not independent draws: the same
    fibers enter both, the galaxy-to-galaxy scatter collapses, and chi2/dof is
    inflated by the correlation rather than by real spatial structure. This is
    the same limitation 7 records for the galaxy bootstrap generally (resampling
    galaxies that share exposures reuses the same correlated sky). Check the
    on-sky separations before reading a large chi2/dof as a gradient.

    Returns dict: diff, diff_err, mean_half, chi2_dof, median_abs_z, frac_3sig.
    """
    if config is None:
        config = PipelineConfig()
    if product.bg_halves is None:
        raise ValueError("background_half_difference needs BG_HALVES "
                         "(re-extract with store_bg_halves=True).")

    h = np.asarray(product.bg_halves[:, variant, :, :], dtype=float)   # (ngal, 2, nwave)
    diff_gal = h[:, 0, :] - h[:, 1, :]
    mean_gal = 0.5 * (h[:, 0, :] + h[:, 1, :])
    ngal = h.shape[0]

    diff, _ = combine_fibers(diff_gal, None, method="biweight")
    mean_half, _ = combine_fibers(mean_gal, None, method="biweight")

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, diff.size), dtype=float)
    for b in range(n_boot):
        boot[b], _ = combine_fibers(diff_gal[rng.integers(0, ngal, ngal)], None,
                                    method="biweight")
    diff_err = np.nanstd(boot, axis=0)

    # Is the stacked difference consistent with zero, pixel by pixel?
    ok = np.isfinite(diff) & np.isfinite(diff_err) & (diff_err > 0)
    zsc = np.full(diff.shape, np.nan)
    zsc[ok] = diff[ok] / diff_err[ok]
    chi2_dof = float(np.nansum(zsc[ok] ** 2) / max(int(ok.sum()), 1))
    median_abs_z = float(np.nanmedian(np.abs(zsc[ok]))) if ok.any() else np.nan
    frac_3sig = float(np.mean(np.abs(zsc[ok]) > 3.0)) if ok.any() else np.nan

    if verbose:
        annuli = product.bg_annuli
        ann = annuli[variant] if variant < len(annuli) else ("?", "?")
        print(f"[background_half_difference] variant {variant} "
              f"annulus {ann}, {ngal} galaxies, {int(ok.sum())} usable pixels")
        print(f"  chi2/dof of (N - S) vs zero = {chi2_dof:.2f}   (1 = consistent)")
        print(f"  median |z|                  = {median_abs_z:.2f}   (0.67 expected)")
        print(f"  fraction |z| > 3            = {frac_3sig:.3f}   (0.003 expected)")
        verdict = ("consistent with noise -- one pooled background per galaxy "
                   "is adequate" if chi2_dof < 2.0 else
                   "EXCESS. Either a real spatial gradient across the annulus, "
                   "or overlapping annuli collapsing the galaxy bootstrap "
                   "(see the docstring caveat) -- check on-sky separations "
                   "before concluding the former")
        print(f"  -> {verdict}")
    return {"diff": diff, "diff_err": diff_err, "mean_half": mean_half,
            "z": zsc, "chi2_dof": chi2_dof, "median_abs_z": median_abs_z,
            "frac_3sig": frac_3sig, "per_galaxy_diff": diff_gal,
            "ngal": ngal, "variant": variant}


# =====================================================================
# ONE-CALL DRIVER
# =====================================================================

def describe_background_test(product, config=None, *, variants=None,
                             n_boot=1000, halfwidth_A=3.0,
                             halfwidths_A=(10.0, 15.0, 20.0, 25.0, 30.0),
                             seed=0):
    """
    Run and print the whole 1.1 / 2.5 background test for a product.

    For each annulus variant: the rest-frame stack (the test), the scrambled-z
    and observed-frame stacks (the controls), the residual at 1215.67 A in
    each, and -- for the fiducial annulus -- the 2.5.1 mask-width criterion.
    Ends with the half-difference check (B10/B11) if BG_HALVES is present.

    Reading it:
      * a residual in REST only              -> real Lya in the annulus; the
                                                mask width matters, use the
                                                adopt_mask_halfwidth row
      * a residual in rest AND the controls  -> instrumental / sky, not a line
      * a residual falling with annulus radius -> consistent with a genuine
                                                two-halo component; a flat one
                                                across annuli is not
      * nothing anywhere                     -> the annulus is Lya-free at this
                                                depth; say so in Methods and
                                                keep the current mask width

    Returns dict keyed by variant index, plus 'halves' and 'mask_criterion'.
    """
    if config is None:
        config = PipelineConfig()
    annuli = product.bg_annuli
    nvar = (product.bg_variants.shape[1] if product.bg_variants is not None else 0)
    if variants is None:
        variants = range(nvar)

    print("=" * 72)
    print("BACKGROUND VARIANT TEST  (specs/systematics-and-extraction.md 1.1, 2.5)")
    print(f"  {product.ngal} galaxies, {nvar} annulus variant(s), "
          f"n_boot = {n_boot}, line window +/-{halfwidth_A:g} A rest")
    print("=" * 72)

    out = {}
    for v in variants:
        ann = annuli[v] if v < len(annuli) else None
        label = f"{ann[0]:g}-{ann[1]:g}\"" if ann else f"variant {v}"
        print(f"\n-- annulus {label} " + "-" * (60 - len(label)))
        res = {}
        for frame in ("rest", "scrambled", "observed"):
            st = stack_backgrounds(product, config, variant=v, frame=frame,
                                   n_boot=n_boot, seed=seed)
            r = measure_line_residual(st, halfwidth_A=halfwidth_A)
            res[frame] = {"stack": st, "residual": r}
            print(f"  {frame:>10}: integral = {r['integral']:11.4e} "
                  f"+/- {r['err']:9.3e}   ({r['significance']:6.2f} sigma)")
        out[v] = res

    if 0 in out:
        print("\n-- pre-registered mask-width criterion (2.5.1) " + "-" * 24)
        out["mask_criterion"] = adopt_mask_halfwidth(
            out[0]["rest"]["stack"], halfwidths_A=halfwidths_A)

    if product.bg_halves is not None:
        print("\n-- background spatial variance (B10/B11) " + "-" * 30)
        out["halves"] = background_half_difference(product, config, n_boot=n_boot,
                                                   seed=seed)
    return out


def plot_background_stack(stacked, ax=None, *, center=LYA_REST, xlim=40.0,
                          show_mask=None, color=None, label=None):
    """
    Plot a background stack with its bootstrap band around the line.

    `show_mask` (observed Angstrom) shades the region the extraction would
    interpolate across at that mask half-width -- the point being to see, by
    eye, whether the feature is contained by the mask or spills into the
    anchors on either side of it.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    grid = np.asarray(stacked["grid"], dtype=float)
    y = np.asarray(stacked["stack"], dtype=float)
    e = np.asarray(stacked["err"], dtype=float)

    c = center
    if stacked["frame"] == "observed":
        c = center * (1.0 + stacked["z_med"])

    ann = stacked.get("annulus")
    if label is None:
        label = (f"{stacked['frame']}"
                 + (f", {ann[0]:g}-{ann[1]:g}\"" if ann else ""))

    ax.plot(grid, y, lw=1.4, color=color, label=label)
    ax.fill_between(grid, y - e, y + e, alpha=0.25, color=color, lw=0)
    ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
    ax.axvline(c, color="crimson", lw=0.9, ls="--")

    if show_mask:
        hw = show_mask if stacked["frame"] == "observed" else show_mask / (1 + stacked["z_med"])
        ax.axvspan(c - hw, c + hw, color="crimson", alpha=0.08, lw=0)

    ax.set_xlim(c - xlim, c + xlim)
    ax.set_xlabel("rest wavelength (A)" if stacked["frame"] != "observed"
                  else "observed wavelength (A)")
    ax.set_ylabel("background (continuum-subtracted)"
                  if stacked["continuum_subtracted"] else "background")
    ax.legend(frameon=False, fontsize=9)
    return ax


# =====================================================================
# VARIANT GEOMETRY  --  the x-axis of the radial map
# =====================================================================

def bg_variant_radii(product, config=None, *, z=None, verbose=False):
    """
    The stored annuli in arcsec AND kpc, SORTED BY RADIUS, with the science
    bins each one overlaps.

    THREE REASONS THIS EXISTS RATHER THAN BEING INLINED.

    (1) THE ORDERING TRAP. `bg_variants` is a config tuple and nothing requires
    it to be monotonic in radius -- the run this module was written for is
    ((80,90), (60,70), (40,50), (110,130)), where variant INDEX order and
    RADIUS order disagree. Plotting or tabulating by index produces a
    nonsense-looking curve that is very easy to mistake for a real
    non-monotonicity in the background. Every function here that reports "vs
    annulus radius" sorts through this helper.

    (2) THE UNIT THE PHYSICS IS IN. The config is in arcsec; the profile, the
    bins and every argument about where the two-halo term lives are in kpc. At
    the sample's median z one arcsec is ~8 kpc, so the conversion changes which
    comparisons look sensible. Done once, here, off the same Planck18
    `cosmo` the extraction and Stage 2 use.

    (3) SELF-SUBTRACTION, which is the one that can silently ruin an analysis.
    Every annulus in a deep run lies INSIDE some science bin. A background drawn
    from 40-50" and subtracted from the 200-500 kpc bin is that bin subtracted
    from itself: the result is driven toward zero regardless of what is there.
    That is fine while the inner variants are only being MEASURED (which is all
    this module does with them), and it is a hard constraint on any future
    radius-dependent background. `science_bins` records the overlap so the
    constraint is visible rather than remembered.

    Returns a list of dicts sorted by mean annulus radius, each with:
        variant, label, inner_arcsec, outer_arcsec, r_mid_arcsec,
        inner_kpc, outer_kpc, r_mid_kpc, area_arcsec2, is_fiducial,
        nfib_med (if BG_NFIB present), science_bins (if bin_mode == 'kpc')
    """
    if config is None:
        config = PipelineConfig()
    annuli = product.bg_annuli
    if not annuli:
        raise ValueError(
            "No BGVAR* header keywords -- this product was not written by a "
            "store_raw_bg run, so there are no annulus variants to describe."
        )

    if z is None:
        z = float(np.nanmedian(np.asarray(product.catalog[config.z_col], dtype=float)))
    kpc_per_arcsec = float(cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265.0)

    # Science bin edges are only interpretable as kpc when the run binned in
    # kpc. For 'virial' or 'arcsec' bin modes the overlap check is skipped
    # rather than guessed at -- a wrong overlap claim is worse than none.
    edges_kpc = None
    if str(getattr(config, "bin_mode", "")).lower() == "kpc":
        edges_kpc = np.asarray(product.r_edges, dtype=float)

    rows = []
    for v, (lo, hi) in enumerate(annuli):
        lo_kpc, hi_kpc = lo * kpc_per_arcsec, hi * kpc_per_arcsec
        row = {
            "variant": v,
            "label": f'{lo:g}-{hi:g}"',
            "inner_arcsec": float(lo), "outer_arcsec": float(hi),
            "r_mid_arcsec": 0.5 * (lo + hi),
            "inner_kpc": lo_kpc, "outer_kpc": hi_kpc,
            "r_mid_kpc": 0.5 * (lo_kpc + hi_kpc),
            "area_arcsec2": float(np.pi * (hi ** 2 - lo ** 2)),
            "is_fiducial": (v == 0),
        }
        if product.bg_nfib is not None and v < product.bg_nfib.shape[1]:
            row["nfib_med"] = float(np.nanmedian(np.asarray(product.bg_nfib[:, v],
                                                            dtype=float)))
        if edges_kpc is not None:
            hit = [(float(edges_kpc[k]), float(edges_kpc[k + 1]))
                   for k in range(edges_kpc.size - 1)
                   if (hi_kpc > edges_kpc[k]) and (lo_kpc < edges_kpc[k + 1])]
            row["science_bins"] = hit
        rows.append(row)

    rows.sort(key=lambda r: r["r_mid_arcsec"])

    if verbose:
        print(f"[bg_variant_radii] z = {z:.3f}, {kpc_per_arcsec:.2f} kpc/arcsec "
              f"(Planck18), sorted by radius")
        for r in rows:
            tag = "  <- FIDUCIAL (the one SPEC was subtracted with)" if r["is_fiducial"] else ""
            nf = f"  nfib~{r['nfib_med']:.0f}" if "nfib_med" in r else ""
            print(f"  v{r['variant']}  {r['label']:>12}  "
                  f"{r['inner_kpc']:7.0f}-{r['outer_kpc']:<7.0f} kpc{nf}{tag}")
            if r.get("science_bins"):
                bins = ", ".join(f"{a:g}-{b:g}" for a, b in r["science_bins"])
                print(f"        overlaps science bin(s): {bins} kpc")
        if any(r.get("science_bins") for r in rows):
            print("  NOTE: annuli overlap science bins. That is fine while they are "
                  "only MEASURED (nothing here subtracts a non-fiducial variant "
                  "from a bin), but a future radius-dependent background must "
                  "never subtract a bin using fibers drawn from inside it.")
    return rows


def _field_labels(product):
    """
    Per-galaxy field label, or None if the catalog doesn't carry one.

    Field matters for the NOISE (EGS is the deeper field, COSMOS the
    larger-area one), so every scatter estimate should be splittable. It is
    optional rather than required because a single-field run legitimately has
    no such column, and failing hard there would block the whole toolkit over a
    reporting convenience.
    """
    cat = product.catalog
    for key in ("FIELD", "field", "Field"):
        if key in cat.colnames:
            return np.asarray([str(s).strip().upper() for s in cat[key]])
    return None


def _line_free_pixels(wave_obs, z, mask_halfwidth_A, bounds=None):
    """Observed pixels excluding this galaxy's own Lya, optionally range-cut."""
    sel = np.isfinite(wave_obs)
    if bounds is not None:
        sel = sel & (wave_obs >= bounds[0]) & (wave_obs <= bounds[1])
    if np.isfinite(z):
        sel = sel & (np.abs(wave_obs - LYA_REST * (1.0 + z)) > mask_halfwidth_A)
    return sel


# =====================================================================
# 2.1  LOOK AT THE INDIVIDUAL BACKGROUNDS FIRST
# =====================================================================

def plot_background_gallery(product, config=None, *, indices=None, variant=0,
                            n=12, ncols=4, rest_frame=False, show_smoothed=True,
                            mask_halfwidth_A=None, sigma_A=None,
                            cont_bounds=BG_CONT_BOUNDS, figsize=None, seed=0):
    """
    A grid of INDIVIDUAL raw backgrounds -- run this before computing anything.

    A stack hides exactly the things that most often go wrong: a galaxy whose
    annulus caught a bright sky residual, one with a pathological fiber count,
    NaN structure at the array edges, a continuum slope unlike every other
    galaxy's. Those galaxies do not announce themselves in the stacked spectrum;
    they just move it. Looking first is cheap and is where the galaxy-rejection
    criterion should come from -- from the panels, not from theory. Whatever
    criterion comes out of it gets WRITTEN DOWN and applied uniformly, not
    applied by eye per galaxy.

      indices     : which galaxies. None -> a random sample of n (seeded, so the
                    same call twice shows the same galaxies).
      rest_frame  : False (default) plots observed wavelength, which is the frame
                    sky residuals and scattered light live in and therefore the
                    frame an instrumental artifact is recognisable in. True
                    shifts each panel by its own z so Lya lines up across panels.
      show_smoothed : overlay what the extraction would actually subtract, i.e.
                    reconstruct_background at the config's (or the overridden)
                    mask width and sigma. The gap between the raw curve and this
                    overlay near the line IS the quantity every other function
                    here measures.

    Shaded: the Lya mask region (red) and the continuum sidebands (grey).
    Annotated per panel: galaxy index, z, nfib for this variant, field.
    """
    import matplotlib.pyplot as plt

    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError("plot_background_gallery needs the RAW backgrounds "
                         "(BG_VARIANTS). Re-extract with store_raw_bg=True.")

    wave = np.asarray(product.wave, dtype=float)
    z = np.asarray(product.catalog[config.z_col], dtype=float)
    bg = np.asarray(product.bg_variants[:, variant, :], dtype=float)
    fields = _field_labels(product)
    ngal = bg.shape[0]

    if indices is None:
        rng = np.random.default_rng(seed)
        n = min(int(n), ngal)
        indices = np.sort(rng.choice(ngal, size=n, replace=False))
    indices = np.atleast_1d(np.asarray(indices, dtype=int))

    mhw = (config.line_mask_halfwidth_A if mask_halfwidth_A is None
           else mask_halfwidth_A)
    nrows = int(np.ceil(len(indices) / ncols))
    if figsize is None:
        figsize = (3.4 * ncols, 2.3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    annuli = product.bg_annuli
    ann = annuli[variant] if variant < len(annuli) else None
    for ax, i in zip(axes.ravel(), indices):
        x = wave / (1.0 + z[i]) if rest_frame else wave
        c = LYA_REST if rest_frame else LYA_REST * (1.0 + z[i])
        hw = mhw / (1.0 + z[i]) if rest_frame else mhw

        ax.plot(x, bg[i], lw=0.8, color="0.25", label="raw")
        if show_smoothed:
            sm = reconstruct_background(bg[i], wave, z[i], config,
                                        mask_halfwidth_A=mhw, sigma_A=sigma_A)
            ax.plot(x, sm, lw=1.2, color="tab:orange", label="subtracted")
        ax.axvspan(c - hw, c + hw, color="crimson", alpha=0.12, lw=0)
        for lo, hi in cont_bounds:
            lo_x = lo if rest_frame else lo * (1.0 + z[i])
            hi_x = hi if rest_frame else hi * (1.0 + z[i])
            ax.axvspan(lo_x, hi_x, color="0.6", alpha=0.15, lw=0)
        ax.axvline(c, color="crimson", lw=0.7, ls="--")

        nf = (f", nfib {int(product.bg_nfib[i, variant])}"
              if product.bg_nfib is not None else "")
        fl = f", {fields[i]}" if fields is not None else ""
        ax.set_title(f"#{i}  z={z[i]:.3f}{nf}{fl}", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(indices):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("rest A" if rest_frame else "observed A", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7)
    ttl = f'raw backgrounds, annulus {ann[0]:g}-{ann[1]:g}"' if ann else "raw backgrounds"
    fig.suptitle(ttl + ("  (rest frame)" if rest_frame else "  (observed frame)"),
                 fontsize=10)
    fig.tight_layout()
    return fig, axes


# =====================================================================
# 2.4  THE RADIAL MAP OF THE BACKGROUND  --  the main event
# =====================================================================

def _restack_subset(stacked, sel, *, n_boot=500, combine="biweight",
                    weights=None, seed=0):
    """
    Re-combine an already-computed stack over a SUBSET of its galaxies.

    `stack_backgrounds` returns `per_galaxy` -- the resampled,
    continuum-subtracted inputs -- so a field split or a paired difference
    needs no re-resampling and no re-continuum-fitting, just a fresh combine.
    That is what keeps the by-field arm and the variant differences inside the
    runtime budget instead of tripling the cost.

    Returns a dict shaped like `stack_backgrounds`' output (enough of it for
    `measure_line_residual` to consume).
    """
    per_gal = np.asarray(stacked["per_galaxy"], dtype=float)[sel]
    w = None if weights is None else np.asarray(weights, dtype=float)[sel]
    n = per_gal.shape[0]
    out, _ = combine_fibers(per_gal, None, method=combine, weights=w)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, out.size), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b], _ = combine_fibers(per_gal[idx], None, method=combine,
                                    weights=(None if w is None else w[idx]))
    return {"grid": stacked["grid"], "stack": out, "err": np.nanstd(boot, axis=0),
            "per_galaxy": per_gal, "frame": stacked["frame"],
            "z_med": stacked["z_med"], "ngal": n,
            "variant": stacked.get("variant"), "annulus": stacked.get("annulus"),
            "combine": combine, "continuum_subtracted": stacked.get("continuum_subtracted")}


def _paired_stack_difference(stack_a, stack_b, *, n_boot=500, combine="biweight",
                             seed=0):
    """
    Stack of (A - B) per galaxy, not (stack of A) - (stack of B).

    The SAME galaxies contribute to every annulus variant, so the two stacks are
    strongly correlated and the difference is far better constrained than
    independent error bars on each would suggest. Differencing per galaxy first
    and bootstrapping the difference keeps that correlation, which is the whole
    reason the variant comparison has any power on ~100 galaxies. Using
    independent errors here would overstate the uncertainty by a large and
    unknown factor and would make a real radial trend look insignificant.
    """
    a = np.asarray(stack_a["per_galaxy"], dtype=float)
    b = np.asarray(stack_b["per_galaxy"], dtype=float)
    diff_gal = a - b
    n = diff_gal.shape[0]
    out, _ = combine_fibers(diff_gal, None, method=combine)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, out.size), dtype=float)
    for bi in range(n_boot):
        boot[bi], _ = combine_fibers(diff_gal[rng.integers(0, n, n)], None,
                                     method=combine)
    return {"grid": stack_a["grid"], "stack": out, "err": np.nanstd(boot, axis=0),
            "per_galaxy": diff_gal, "frame": stack_a["frame"],
            "z_med": stack_a["z_med"], "ngal": n, "combine": combine,
            "continuum_subtracted": True}


def background_levels(product, config=None, *, variants=None,
                      mask_halfwidth_A=None, bounds=None, robust=True):
    """
    The per-galaxy NON-LINE background LEVEL for each annulus variant.

    This is Reading 2 of 2.4 and it is a different measurement from the line
    stack, deliberately:

      * It is computed on the RAW background with NO continuum subtraction,
        because the continuum IS the quantity -- the sky + scattered-light
        pedestal. Subtracting a per-galaxy continuum first would remove exactly
        what this is trying to measure.
      * It is computed in the OBSERVED frame, because sky lines and scattered
        light live at fixed observed wavelengths. Shifting to rest first would
        smear a fixed-observed-wavelength structure across the sample's z range.
      * Each galaxy's own Lya is masked out (at `mask_halfwidth_A`, default the
        config's), so a genuinely line-carrying annulus does not report a higher
        "level" just because it has more line in it.

    WHY IT MATTERS. If this level is flat across annuli, the background is
    spatially uniform over the range the science bins occupy, one global annulus
    is defensible, and the whole variable-background question closes with a
    one-line justification. If it varies systematically with radius, a
    background measured at one radius is a biased estimate of the background
    under a bin at another, and a radius-dependent background becomes worth
    designing (see specs/background-diagnostics.md 0.3 for the two constraints
    any such design has to satisfy).

    Returns (ngal, nvar) levels, in variant-INDEX order (not radius order).
    """
    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError("background_levels needs BG_VARIANTS (store_raw_bg=True).")

    wave = np.asarray(product.wave, dtype=float)
    z = np.asarray(product.catalog[config.z_col], dtype=float)
    mhw = config.line_mask_halfwidth_A if mask_halfwidth_A is None else mask_halfwidth_A
    nvar = product.bg_variants.shape[1]
    variants = range(nvar) if variants is None else list(variants)

    out = np.full((product.ngal, nvar), np.nan)
    for v in variants:
        bg = np.asarray(product.bg_variants[:, v, :], dtype=float)
        for i in range(bg.shape[0]):
            sel = _line_free_pixels(wave, z[i], mhw, bounds) & np.isfinite(bg[i])
            if sel.sum() < 10:
                continue
            out[i, v] = (np.nanmedian(bg[i][sel]) if robust
                         else np.nanmean(bg[i][sel]))
    return out


def compare_bg_variants(product, config=None, *, variants=None, n_boot=500,
                        halfwidth_A=3.0, controls=True, level_mask_halfwidth_A=None,
                        level_bounds=None, by_field=True, combine="biweight",
                        seed=0, verbose=True):
    """
    THE MAIN EVENT: how the background changes with annulus radius.

    Stacks every stored annulus and reports TWO readings off the same
    computation. On the run this was written for the annuli are 40-50", 60-70",
    80-90" (fiducial) and 110-130", i.e. 330-1070 kpc at z ~ 2.3 -- a factor of
    3 in radius, all of it inside the range the science bins occupy.

    READING 1 -- the LINE amplitude vs annulus radius.
        falls with radius  -> real, physically clustered Lya. Expected, since
                              the inner annuli sit where the profile already
                              detects signal at good S/N. This calibrates how
                              much line the fiducial annulus is carrying.
        flat across annuli -> instrumental or sky, not line. A wider mask will
                              not help and the fix is elsewhere.
        rises with radius  -> something is wrong. Stop and investigate before
                              reading anything else here.

    READING 2 -- the NON-LINE level vs annulus radius (see `background_levels`).
        flat -> one global annulus is defensible; failure mode B is small; the
                variable-background idea closes cheaply.
        varies -> the background is not spatially uniform and a background
                measured at one radius is biased under bins at another.

    EVERYTHING IS REPORTED SORTED BY RADIUS, not by variant index. The config
    tuple need not be monotonic (the reference run is not), and an index-ordered
    plot of a perfectly monotonic background looks like a non-monotonic one.

    DIFFERENCES ARE PAIRED. The same galaxies feed every variant, so both the
    line-amplitude differences and the level differences are computed per galaxy
    and then bootstrapped -- see `_paired_stack_difference` for why treating
    them as independent would hide a real trend.

    RUNTIME. Dominated by n_boot x n_variants x (2 if controls) combines.
    n_boot=500 over 4 variants runs in well under a minute; raise it for a
    number that goes in the paper, lower it while exploring.

    Returns dict:
      rows        one per variant, SORTED BY RADIUS, with geometry, line
                  integral/err/significance (rest and observed-frame control),
                  level and its error, and the paired differences vs the fiducial
      stacks      {variant: rest-frame stack dict}  (for plotting)
      controls    {variant: observed-frame stack dict} or {}
      levels      (ngal, nvar) raw per-galaxy levels
      extremes    the innermost-vs-outermost paired comparison, called out
                  explicitly because it is the widest lever the run has
      by_field    per-field rows, or None
      geometry, z_med, kpc_per_arcsec, ngal
    """
    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError("compare_bg_variants needs BG_VARIANTS (store_raw_bg=True).")

    geom = bg_variant_radii(product, config)
    if variants is not None:
        keep = set(int(v) for v in variants)
        geom = [g for g in geom if g["variant"] in keep]
    order = [g["variant"] for g in geom]                 # radius-sorted

    z_med = float(np.nanmedian(np.asarray(product.catalog[config.z_col], dtype=float)))
    kpc_per_arcsec = float(cosmo.angular_diameter_distance(z_med).to(u.kpc).value / 206265.0)

    # --- the stacks -------------------------------------------------
    stacks, ctrl = {}, {}
    for v in order:
        stacks[v] = stack_backgrounds(product, config, variant=v, frame="rest",
                                      n_boot=n_boot, combine=combine, seed=seed)
        if controls:
            ctrl[v] = stack_backgrounds(product, config, variant=v, frame="observed",
                                        n_boot=n_boot, combine=combine, seed=seed)

    # --- the levels -------------------------------------------------
    levels = background_levels(product, config, variants=order,
                               mask_halfwidth_A=level_mask_halfwidth_A,
                               bounds=level_bounds)
    rng = np.random.default_rng(seed)

    def _level_stat(vals):
        good = vals[np.isfinite(vals)]
        if good.size < 3:
            return np.nan, np.nan
        boot = np.array([np.nanmedian(good[rng.integers(0, good.size, good.size)])
                         for _ in range(n_boot)])
        return float(np.nanmedian(good)), float(np.nanstd(boot))

    def _paired_level(v_a, v_b):
        d = levels[:, v_a] - levels[:, v_b]
        good = d[np.isfinite(d)]
        if good.size < 3:
            return np.nan, np.nan
        boot = np.array([np.nanmedian(good[rng.integers(0, good.size, good.size)])
                         for _ in range(n_boot)])
        return float(np.nanmedian(good)), float(np.nanstd(boot))

    fid = 0 if 0 in order else order[0]

    rows = []
    for g in geom:
        v = g["variant"]
        r_rest = measure_line_residual(stacks[v], halfwidth_A=halfwidth_A)
        row = dict(g)
        row.update({
            "line_integral": r_rest["integral"],
            "line_err": r_rest["err"],
            "line_sig": r_rest["significance"],
            "line_peak": r_rest["peak"],
        })
        if controls:
            r_obs = measure_line_residual(ctrl[v], halfwidth_A=halfwidth_A)
            row["line_sig_observed"] = r_obs["significance"]
        lvl, lvl_e = _level_stat(levels[:, v])
        row["level"], row["level_err"] = lvl, lvl_e
        if v != fid:
            d = _paired_stack_difference(stacks[v], stacks[fid], n_boot=n_boot,
                                         combine=combine, seed=seed)
            rd = measure_line_residual(d, halfwidth_A=halfwidth_A)
            row["dline_vs_fid"] = rd["integral"]
            row["dline_vs_fid_err"] = rd["err"]
            row["dline_vs_fid_sig"] = rd["significance"]
            row["dlevel_vs_fid"], row["dlevel_vs_fid_err"] = _paired_level(v, fid)
        rows.append(row)

    # --- the named comparison: innermost vs outermost ----------------
    extremes = None
    if len(order) >= 2:
        v_in, v_out = order[0], order[-1]
        d = _paired_stack_difference(stacks[v_in], stacks[v_out], n_boot=n_boot,
                                     combine=combine, seed=seed)
        rd = measure_line_residual(d, halfwidth_A=halfwidth_A)
        dl, dl_e = _paired_level(v_in, v_out)
        extremes = {
            "inner": geom[0]["label"], "outer": geom[-1]["label"],
            "inner_variant": v_in, "outer_variant": v_out,
            "dline": rd["integral"], "dline_err": rd["err"],
            "dline_sig": rd["significance"],
            "dlevel": dl, "dlevel_err": dl_e,
            "dlevel_sig": (dl / dl_e if (dl_e and np.isfinite(dl_e) and dl_e > 0)
                           else np.nan),
            "stack": d,
        }

    # --- field split -------------------------------------------------
    field_rows = None
    fields = _field_labels(product) if by_field else None
    if fields is not None:
        field_rows = {}
        for f in sorted(set(fields)):
            sel = (fields == f)
            fr = []
            for g in geom:
                v = g["variant"]
                st = _restack_subset(stacks[v], sel, n_boot=n_boot,
                                     combine=combine, seed=seed)
                r = measure_line_residual(st, halfwidth_A=halfwidth_A)
                lvl, lvl_e = _level_stat(levels[sel, v])
                fr.append({**g, "ngal": int(sel.sum()),
                           "line_integral": r["integral"], "line_err": r["err"],
                           "line_sig": r["significance"],
                           "level": lvl, "level_err": lvl_e})
            field_rows[f] = fr

    result = {
        "rows": rows, "stacks": stacks, "controls": ctrl, "levels": levels,
        "extremes": extremes, "by_field": field_rows, "geometry": geom,
        "z_med": z_med, "kpc_per_arcsec": kpc_per_arcsec, "ngal": product.ngal,
        "halfwidth_A": halfwidth_A, "n_boot": n_boot, "fiducial_variant": fid,
    }
    if verbose:
        describe_bg_variants(result)
    return result


def describe_bg_variants(result):
    """Print `compare_bg_variants` as two readable tables. Radius-sorted."""
    print("=" * 78)
    print("BACKGROUND RADIAL MAP  (specs/background-diagnostics.md 2.4)")
    print(f"  {result['ngal']} galaxies, z_med = {result['z_med']:.3f}, "
          f"{result['kpc_per_arcsec']:.2f} kpc/arcsec, n_boot = {result['n_boot']}, "
          f"line window +/-{result['halfwidth_A']:g} A rest")
    print("=" * 78)

    has_obs = any("line_sig_observed" in r for r in result["rows"])
    print("\nREADING 1 -- Lya line amplitude vs annulus radius")
    hdr = (f"  {'annulus':>12} {'r_mid':>9} {'nfib':>7} {'integral':>12} "
           f"{'+/-':>10} {'sigma':>7}")
    if has_obs:
        hdr += f" {'obs ctrl':>9}"
    hdr += f"  {'d(vs fid)':>11} {'sigma':>7}"
    print(hdr)
    for r in result["rows"]:
        nf = f"{r.get('nfib_med', np.nan):7.0f}" if "nfib_med" in r else "      -"
        line = (f"  {r['label']:>12} {r['r_mid_kpc']:8.0f}k {nf} "
                f"{r['line_integral']:12.4e} {r['line_err']:10.3e} "
                f"{r['line_sig']:7.2f}")
        if has_obs:
            line += f" {r.get('line_sig_observed', np.nan):9.2f}"
        if "dline_vs_fid" in r:
            line += f"  {r['dline_vs_fid']:11.3e} {r['dline_vs_fid_sig']:7.2f}"
        else:
            line += f"  {'(fiducial)':>11} {'':>7}"
        print(line)
    print("  reading: falls with radius -> real clustered Lya (expected);  "
          "flat -> instrumental/sky;  rises -> investigate.")
    if has_obs:
        print("  'obs ctrl' is the observed-frame stack: a real Lya feature washes "
              "out there, an artifact at fixed observed wavelength survives.")

    print("\nREADING 2 -- non-line background LEVEL vs annulus radius "
          "(raw, no continuum subtraction, own-Lya masked)")
    print(f"  {'annulus':>12} {'r_mid':>9} {'level':>13} {'+/-':>11} "
          f"{'d(vs fid)':>13} {'sigma':>7}")
    for r in result["rows"]:
        d = r.get("dlevel_vs_fid", np.nan)
        de = r.get("dlevel_vs_fid_err", np.nan)
        ds = (d / de) if (np.isfinite(d) and np.isfinite(de) and de > 0) else np.nan
        tail = (f"{d:13.4e} {ds:7.2f}" if "dlevel_vs_fid" in r
                else f"{'(fiducial)':>13} {'':>7}")
        print(f"  {r['label']:>12} {r['r_mid_kpc']:8.0f}k {r['level']:13.5e} "
              f"{r['level_err']:11.3e} {tail}")
    print("  reading: flat -> one global annulus is defensible, and the "
          "variable-background question closes with a one-line justification;")
    print("           varies -> a background measured at one radius is biased "
          "under bins at another (see spec 0.3 before designing anything).")

    ex = result.get("extremes")
    if ex:
        print(f"\nTHE WIDEST LEVER -- {ex['inner']} vs {ex['outer']} "
              f"(paired, same galaxies)")
        print(f"  line amplitude difference : {ex['dline']:11.4e} "
              f"+/- {ex['dline_err']:9.3e}  ({ex['dline_sig']:6.2f} sigma)")
        print(f"  level difference          : {ex['dlevel']:11.4e} "
              f"+/- {ex['dlevel_err']:9.3e}  ({ex['dlevel_sig']:6.2f} sigma)")
        print("  This is the direct test of 'would a larger, further-out "
              "background aperture be better'.")

    if result.get("by_field"):
        print("\nBY FIELD  (amplitudes should agree; noise levels need not -- "
              "the fields differ in depth)")
        for f, rows in result["by_field"].items():
            print(f"  -- {f} ({rows[0]['ngal']} galaxies)")
            for r in rows:
                print(f"     {r['label']:>12}  line {r['line_integral']:11.3e} "
                      f"({r['line_sig']:6.2f} sig)   level {r['level']:11.4e}")

    if any(r.get("science_bins") for r in result["rows"]):
        print("\nNOTE: some annuli lie inside science bins. Harmless here -- "
              "nothing in this function subtracts a non-fiducial variant from a "
              "bin -- but a radius-dependent background must never subtract a "
              "bin using fibers drawn from inside it (spec 0.3).")
    return result


def plot_bg_variant_comparison(result, *, xlim=40.0, figsize=(13, 4.2),
                               show_level_spectra=False, product=None,
                               config=None):
    """
    The main figure of the variant run: three panels, radius-sorted.

      left   -- every annulus's rest-frame stack overlaid around Lya, with
                bootstrap bands. This is where you SEE whether the feature
                shrinks with radius, and whether its width is contained by the
                mask.
      middle -- Reading 1: line integral vs annulus radius (kpc), with errors.
      right  -- Reading 2: non-line level vs annulus radius (kpc), with errors.

    The x-axis of the two right panels is PHYSICAL radius, not variant index,
    for the reason in `bg_variant_radii`.
    """
    import matplotlib.pyplot as plt

    rows = result["rows"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    cmap = plt.get_cmap("viridis")
    n = max(len(rows), 1)
    for k, r in enumerate(rows):
        st = result["stacks"][r["variant"]]
        color = cmap(0.15 + 0.7 * k / max(n - 1, 1))
        lbl = f'{r["label"]}  ({r["r_mid_kpc"]:.0f} kpc)'
        if r["is_fiducial"]:
            lbl += "  [fid]"
        plot_background_stack(st, ax=axes[0], xlim=xlim, color=color, label=lbl)

    for ax, key, ekey, title in (
            (axes[1], "line_integral", "line_err", "Reading 1: Lya amplitude"),
            (axes[2], "level", "level_err", "Reading 2: non-line level")):
        x = [r["r_mid_kpc"] for r in rows]
        y = [r[key] for r in rows]
        e = [r[ekey] for r in rows]
        ax.errorbar(x, y, yerr=e, marker="o", lw=1.4, capsize=3, color="tab:blue")
        for r in rows:
            if r["is_fiducial"]:
                ax.axvline(r["r_mid_kpc"], color="crimson", lw=0.9, ls="--",
                           label="fiducial")
        ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("annulus radius (kpc)")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_ylabel(f"integral +/-{result['halfwidth_A']:g} A rest")
    axes[2].set_ylabel("background level (raw units)")
    fig.tight_layout()
    return fig, axes


# =====================================================================
# 2.5  MASK HALF-WIDTH  x  SMOOTHING SIGMA  --  the joint knob
# =====================================================================

def mask_width_scan(product, config=None, *, widths=(5., 10., 15., 20., 25., 30., 40.),
                    sigmas=None, variant=0, halfwidth_A=3.0, n_boot=300,
                    rest_grid=None, cont_bounds=BG_CONT_BOUNDS, cont_method="poly",
                    cont_order=1, combine="biweight", threshold=1.0,
                    seed=0, verbose=True):
    """
    2D scan of the residual Lya left in the PROCESSED background, over
    (mask half-width, smoothing sigma).

    WHY 2D AND NOT 1D. Mask width and smoothing sigma are one joint knob, not
    two (systematics 2.2). Widening the mask removes more line; the gaussian
    then smears the surviving wings back INTO the gap, so a width that looks
    clean at sigma = 10 A need not be clean at sigma = 30 A. Scanning one at a
    time finds a false plateau.

    WHAT IS MEASURED. For every grid point: re-mask, re-fill and re-smooth every
    galaxy's RAW background exactly as the extraction would, stack the PROCESSED
    backgrounds in the rest frame, and integrate the residual within
    +/-halfwidth_A of 1215.67 A. That is the flux the extraction would have
    subtracted from the science bins as if it were background.

    HOW THE ERROR IS COMPUTED, because this is the one shortcut here and it
    should be visible. Bootstrapping every grid point would dominate the runtime
    and is unnecessary: the mask width barely changes the NOISE (it changes the
    BIAS), while sigma changes the noise a great deal. So the bootstrap runs
    ONCE PER SIGMA, at the widest mask in the scan -- the configuration where
    the residual should be closest to zero and its scatter is therefore the
    honest noise for that smoothing. All widths in a row share that error. If
    you need a per-cell error for a number that goes in the paper, re-run the
    single adopted cell through `stack_backgrounds` + `measure_line_residual`.

    READING THE RESULT. Adopt from the PLATEAU, not from the edge: the smallest
    width whose |residual/err| is under `threshold` AND stays under it at the
    next width up. A single crossing is noise. And remember what over-masking
    costs -- a 60 A mask with a huge sigma drives the residual to zero by
    turning the background into a featureless line that can no longer remove
    the systematics it exists to remove.

    RUNTIME. len(sigmas) x len(widths) x ngal smooth+resample operations (fast,
    C-level) plus len(sigmas) bootstraps. A 5x7 grid on 100 galaxies at
    n_boot=300 is a couple of minutes. Everything that does not depend on the
    scanned parameters is hoisted out of the loop.

    Returns dict: widths, sigmas, residual (nsig, nwid), err (nsig,),
    ratio (nsig, nwid), adopted (per sigma), variant, annulus.
    """
    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError("mask_width_scan needs BG_VARIANTS (store_raw_bg=True).")

    if sigmas is None:
        s0 = float(config.bg_smooth_sigma_A)
        sigmas = tuple(sorted({10.0, 20.0, s0, 40.0, 50.0}))
    widths = np.asarray(sorted(float(w) for w in widths), dtype=float)
    sigmas = np.asarray(sorted(float(s) for s in sigmas), dtype=float)

    wave = np.asarray(product.wave, dtype=float)
    z = np.asarray(product.catalog[config.z_col], dtype=float)
    bg = np.asarray(product.bg_variants[:, variant, :], dtype=float)
    ngal = bg.shape[0]
    z_med = float(np.nanmedian(z))

    # Hoisted: the rest grid and each galaxy's rest wavelengths do not depend on
    # either scanned parameter, so they are built once instead of 35 times.
    if rest_grid is None:
        dw_rest = float(np.nanmedian(np.diff(wave))) / (1.0 + z_med)
        rest_grid = np.arange(1140.0, 1300.0 + 1e-9, dw_rest)
    grid = np.asarray(rest_grid, dtype=float)
    rest_wave = [wave / (1.0 + z[i]) for i in range(ngal)]
    dx = float(np.nanmedian(np.diff(grid)))
    win = np.abs(grid - LYA_REST) <= halfwidth_A

    def _process(w, s):
        """All galaxies at one grid point -> (ngal, ngrid) continuum-subtracted."""
        per_gal = np.empty((ngal, grid.size), dtype=float)
        for i in range(ngal):
            proc = reconstruct_background(bg[i], wave, z[i], config,
                                          mask_halfwidth_A=w, sigma_A=s)
            good = np.isfinite(proc)
            if good.sum() < 2:
                per_gal[i] = np.nan
                continue
            per_gal[i] = np.interp(grid, rest_wave[i][good], proc[good],
                                   left=np.nan, right=np.nan)
            if np.any(np.isfinite(per_gal[i])):
                per_gal[i] -= get_continuum_model(grid, per_gal[i],
                                                  cont_bounds=cont_bounds,
                                                  cont_method=cont_method,
                                                  cont_order=cont_order)
        return per_gal

    residual = np.full((sigmas.size, widths.size), np.nan)
    err = np.full(sigmas.size, np.nan)
    rng = np.random.default_rng(seed)

    for si, s in enumerate(sigmas):
        for wi, w in enumerate(widths):
            per_gal = _process(w, s)
            stack, _ = combine_fibers(per_gal, None, method=combine)
            residual[si, wi] = float(np.nansum(stack[win]) * dx)
            if wi == widths.size - 1:          # widest mask: the noise reference
                boot = np.empty(n_boot, dtype=float)
                for b in range(n_boot):
                    bs, _ = combine_fibers(per_gal[rng.integers(0, ngal, ngal)],
                                           None, method=combine)
                    boot[b] = float(np.nansum(bs[win]) * dx)
                err[si] = float(np.nanstd(boot))

    ratio = residual / err[:, None]

    adopted = {}
    for si, s in enumerate(sigmas):
        clean = np.abs(ratio[si]) < threshold
        pick = None
        for wi in range(widths.size - 1):
            if clean[wi] and clean[wi + 1]:       # a plateau, not a crossing
                pick = float(widths[wi])
                break
        if pick is None and clean.size and clean[-1]:
            pick = float(widths[-1])              # only the last one is clean
        adopted[float(s)] = pick

    annuli = product.bg_annuli
    result = {
        "widths": widths, "sigmas": sigmas, "residual": residual, "err": err,
        "ratio": ratio, "adopted": adopted, "threshold": threshold,
        "variant": variant,
        "annulus": annuli[variant] if variant < len(annuli) else None,
        "halfwidth_A": halfwidth_A, "n_boot": n_boot, "ngal": ngal,
        "config_width": float(config.line_mask_halfwidth_A),
        "config_sigma": float(config.bg_smooth_sigma_A),
    }
    if verbose:
        ann = result["annulus"]
        print(f"[mask_width_scan] variant {variant}"
              + (f", annulus {ann[0]:g}-{ann[1]:g}\"" if ann else "")
              + f", {ngal} galaxies, residual within +/-{halfwidth_A:g} A rest")
        print(f"  production is mask={result['config_width']:g} A, "
              f"sigma={result['config_sigma']:g} A "
              "(taken from the config, never from function defaults)")
        print("  residual / err   (rows = sigma, cols = mask half-width, "
              "observed A)")
        print("      sigma |" + "".join(f"{w:8.0f}" for w in widths))
        print("      ------+" + "-" * (8 * widths.size))
        for si, s in enumerate(sigmas):
            cells = "".join(f"{ratio[si, wi]:8.2f}" for wi in range(widths.size))
            a = adopted[float(s)]
            print(f"   {s:8.0f} |{cells}   -> adopt "
                  + (f"{a:g} A" if a is not None else "NONE clean"))
        print(f"  criterion: |residual| < {threshold:g} sigma at a width AND at "
              "the next width up (a plateau, not a single crossing)")
        if all(v is None for v in adopted.values()):
            print("  -> NO clean plateau anywhere. Either widen the scan, or the "
                  "annulus carries a component broader than any mask excludes -- "
                  "in which case the mask is the wrong tool and 1.1b's "
                  "window-anchor / normalized-convolution fill is the fix.")
        else:
            print("  -> adopt from the PLATEAU, not the edge. A very wide mask "
                  "with a large sigma reaches zero by destroying the "
                  "background's ability to remove systematics.")
    return result


def plot_mask_width_scan(result, *, figsize=(7.2, 4.4), cmap="RdBu_r", vmax=None):
    """
    Heatmap of residual/err over (mask half-width, smoothing sigma), with the
    production setting marked and the adopted plateau outlined.

    The colour scale is symmetric about zero on purpose -- the question is
    whether a cell is consistent with zero, and a sequential map makes an
    over-subtracted (negative) cell look like a good one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ratio = np.asarray(result["ratio"], dtype=float)
    widths = np.asarray(result["widths"], dtype=float)
    sigmas = np.asarray(result["sigmas"], dtype=float)
    if vmax is None:
        vmax = max(float(np.nanmax(np.abs(ratio))), result["threshold"] * 2.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ratio, origin="lower", aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax,
                   extent=[-0.5, widths.size - 0.5, -0.5, sigmas.size - 0.5])
    ax.set_xticks(range(widths.size), [f"{w:g}" for w in widths])
    ax.set_yticks(range(sigmas.size), [f"{s:g}" for s in sigmas])
    ax.set_xlabel("mask half-width (observed A)")
    ax.set_ylabel("smoothing sigma (A)")

    for si in range(sigmas.size):
        for wi in range(widths.size):
            v = ratio[si, wi]
            if np.isfinite(v):
                ax.text(wi, si, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color=("white" if abs(v) > 0.6 * vmax else "0.15"))

    for si, s in enumerate(sigmas):
        a = result["adopted"].get(float(s))
        if a is not None:
            wi = int(np.argmin(np.abs(widths - a)))
            ax.add_patch(Rectangle((wi - 0.5, si - 0.5), 1, 1, fill=False,
                                   edgecolor="lime", lw=2.0))
    cw, cs = result["config_width"], result["config_sigma"]
    if cw in widths and cs in sigmas:
        ax.plot(int(np.argmin(np.abs(widths - cw))),
                int(np.argmin(np.abs(sigmas - cs))),
                marker="*", ms=14, color="k", label="production")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    ann = result.get("annulus")
    ax.set_title("residual / err  (green = adopted plateau)"
                 + (f'   annulus {ann[0]:g}-{ann[1]:g}"' if ann else ""),
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="residual / err")
    fig.tight_layout()
    return fig, ax


# =====================================================================
# 2.6  THE SPREAD  --  galaxy-to-galaxy vs spatial (B10 / B11)
# =====================================================================

def background_variance_profile(product, config=None, *, variant=0, by_field=True,
                                n_quartiles=4, verbose=True):
    """
    How much the background varies -- and how much of that variation is SPATIAL.

    The galaxy-to-galaxy scatter of the background sets whether a SINGLE
    galaxy's background is trustworthy, which is a different question from
    whether the STACKED background is unbiased. A stack can be perfectly fine
    while every individual subtraction is noisy.

    Everything is computed in the OBSERVED frame: sky residuals and scattered
    light sit at fixed observed wavelengths, and shifting to rest first would
    smear that structure across the sample's redshift range and understate it.

    THREE DECOMPOSITIONS, each answering a different question:

      by field      -- EGS is the deeper field, COSMOS the larger-area one, so
                       they should NOT have the same noise. If they do, or if
                       the ratio is far from the depth ratio, something is off.
      by nfib quartile -- shot noise falls as 1/sqrt(nfib) while real spatial
                       variation does not. If the scatter is flat across
                       quartiles, it is not shot noise, and adding fibers will
                       not fix it.
      azimuthal halves -- the cleanest handle available on B10/B11. The two
                       halves are disjoint fiber sets from the SAME annulus of
                       the SAME galaxy, so their difference has the
                       galaxy-to-galaxy term removed by construction and
                       isolates what varies WITHIN one annulus. `(h0-h1)/2` is
                       normalised so that, under pure shot noise and no spatial
                       structure, its scatter equals the pooled background's own
                       shot noise -- see the comment at the code for why the
                       factor is 2 and not sqrt(2). So `spatial_fraction` near 1
                       means the galaxy-to-galaxy scatter is essentially all
                       shot-noise-like, near 0 means real between-galaxy
                       variation dominates, and comfortably ABOVE 1 means there
                       is structure across the annulus that stacking galaxies
                       will never remove.
                       This is a variance comparison, not a significance test;
                       `background_half_difference` is the test of whether the
                       MEAN half-difference is consistent with zero.

    Returns dict: wave, level, scatter, n_used, by_field, by_nfib_quartile,
    halves {scatter, spatial_fraction}, and scalar summaries.
    """
    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError("background_variance_profile needs BG_VARIANTS "
                         "(store_raw_bg=True).")

    wave = np.asarray(product.wave, dtype=float)
    bg = np.asarray(product.bg_variants[:, variant, :], dtype=float)
    annuli = product.bg_annuli
    ann = annuli[variant] if variant < len(annuli) else None

    def _scatter(a):
        # mad_std, not std: one galaxy whose annulus caught a bright sky
        # residual would otherwise set the "typical" scatter single-handed.
        return mad_std(a, axis=0, ignore_nan=True)

    scatter = _scatter(bg)
    level = np.nanmedian(bg, axis=0)
    n_used = np.sum(np.isfinite(bg), axis=0)

    out = {"wave": wave, "level": level, "scatter": scatter, "n_used": n_used,
           "variant": variant, "annulus": ann, "ngal": bg.shape[0]}

    fields = _field_labels(product) if by_field else None
    if fields is not None:
        out["by_field"] = {f: {"scatter": _scatter(bg[fields == f]),
                               "level": np.nanmedian(bg[fields == f], axis=0),
                               "ngal": int(np.sum(fields == f))}
                           for f in sorted(set(fields))}
    else:
        out["by_field"] = None

    if product.bg_nfib is not None:
        nf = np.asarray(product.bg_nfib[:, variant], dtype=float)
        edges = np.nanpercentile(nf, np.linspace(0, 100, n_quartiles + 1))
        q = {}
        for k in range(n_quartiles):
            # NOTE the parentheses. Written without them,
            #   (nf >= lo) & (nf <= hi if last else nf < hi)
            # Python's conditional expression binds LOOSER than `&`, so the
            # whole `(nf >= lo) & (nf <= hi)` becomes the true-branch and every
            # non-final quartile silently loses its lower bound -- giving
            # nested, overlapping "quartiles". Caught by the smoke test only
            # because a constant nfib made the collapse visible.
            upper = (nf <= edges[k + 1]) if k == n_quartiles - 1 else (nf < edges[k + 1])
            sel = (nf >= edges[k]) & upper
            if sel.sum() >= 3:
                q[f"Q{k + 1}"] = {"nfib_range": (float(edges[k]), float(edges[k + 1])),
                                  "nfib_med": float(np.nanmedian(nf[sel])),
                                  "scatter_med": float(np.nanmedian(_scatter(bg[sel]))),
                                  "ngal": int(sel.sum())}
        out["by_nfib_quartile"] = q
    else:
        out["by_nfib_quartile"] = None

    if product.bg_halves is not None:
        h = np.asarray(product.bg_halves[:, variant, :, :], dtype=float)
        # THE FACTOR OF 2, which is easy to get wrong and changes the answer.
        # Each half is built from ~N/2 fibers against the pooled background's N,
        # so under pure shot noise sigma_half = sqrt(2) * sigma_pooled and
        # var(h0 - h1) = 2 * sigma_half^2 = 4 * sigma_pooled^2. Dividing the
        # difference by 2 (NOT by sqrt(2)) therefore puts it on the same
        # footing as the pooled per-galaxy background: with no spatial
        # structure at all, mad_std of (h0-h1)/2 equals the pooled background's
        # OWN shot noise. Dividing by sqrt(2) instead inflates the ratio below
        # by sqrt(2) and makes a perfectly uniform background look strongly
        # spatially structured.
        half_diff = (h[:, 0, :] - h[:, 1, :]) / 2.0
        sp = _scatter(half_diff)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.nanmedian(sp / scatter)
        out["halves"] = {"scatter": sp, "spatial_fraction": float(frac),
                         "per_galaxy_diff": half_diff}
    else:
        out["halves"] = None

    out["scatter_med"] = float(np.nanmedian(scatter))
    out["level_med"] = float(np.nanmedian(level))

    if verbose:
        print(f"[background_variance_profile] variant {variant}"
              + (f", annulus {ann[0]:g}-{ann[1]:g}\"" if ann else "")
              + f", {out['ngal']} galaxies")
        print(f"  median level across wavelength : {out['level_med']:.5e}")
        print(f"  median galaxy-to-galaxy scatter: {out['scatter_med']:.5e}")
        if out["by_field"]:
            for f, d in out["by_field"].items():
                print(f"    {f:>8}: scatter {float(np.nanmedian(d['scatter'])):.4e} "
                      f"({d['ngal']} galaxies)")
            print("    (the fields differ in depth -- these are EXPECTED to "
                  "differ; equal values would be the surprise)")
        if out["by_nfib_quartile"]:
            print("  by fiber-count quartile (shot noise should fall as "
                  "1/sqrt(nfib); flat means it is not shot noise):")
            for k, d in out["by_nfib_quartile"].items():
                print(f"    {k}: nfib~{d['nfib_med']:8.0f}  "
                      f"scatter {d['scatter_med']:.4e}  ({d['ngal']} gal)")
        if out["halves"]:
            fr = out["halves"]["spatial_fraction"]
            print(f"  WITHIN-ANNULUS fraction (halves/2 over total) = {fr:.2f}"
                  "   [1.0 = all shot-noise-like, 0 = all galaxy-to-galaxy]")
            if fr < 0.4:
                print("    -> the scatter is dominated by variation BETWEEN "
                      "galaxies (depth, sky conditions, exposure count), not "
                      "within an annulus. Stacking more galaxies keeps helping.")
            elif fr > 1.2:
                print("    -> EXCESS above the shot-noise expectation: the "
                      "background varies spatially WITHIN one annulus, which no "
                      "amount of galaxy stacking removes. This is the floor on "
                      "failure mode B.")
            else:
                print("    -> consistent with the two halves differing by shot "
                      "noise alone; no evidence for spatial structure across "
                      "the annulus at this precision.")
            print("    NOTE this is a VARIANCE comparison, not a significance "
                  "test. For 'is the mean N-half minus S-half consistent with "
                  "zero', use background_half_difference -- different question, "
                  "and it carries its own overlapping-annuli caveat.")
        else:
            print("  (no BG_HALVES -- re-extract with store_bg_halves=True to "
                  "separate the spatial term from the galaxy-to-galaxy one)")
    return out


def plot_background_variance(result, *, figsize=(10, 4.0), logy=True):
    """
    Scatter vs observed wavelength: total, per field, and the spatial (halves)
    term, plus the scatter-vs-fiber-count panel.

    Read the left panel for STRUCTURE (sky-line forests show up as spikes and
    are where per-galaxy rejection criteria come from) and the right panel for
    whether the noise behaves like shot noise at all.
    """
    import matplotlib.pyplot as plt

    ncol = 2 if result.get("by_nfib_quartile") else 1
    fig, axes = plt.subplots(1, ncol, figsize=figsize, squeeze=False)
    ax = axes[0, 0]

    ax.plot(result["wave"], result["scatter"], lw=1.0, color="0.2",
            label=f"total (all {result['ngal']} gal)")
    if result.get("by_field"):
        for f, d in result["by_field"].items():
            ax.plot(result["wave"], d["scatter"], lw=0.9, alpha=0.85,
                    label=f"{f} ({d['ngal']})")
    if result.get("halves"):
        ax.plot(result["wave"], result["halves"]["scatter"], lw=1.0,
                color="tab:red", alpha=0.85,
                label=f"spatial (halves), frac={result['halves']['spatial_fraction']:.2f}")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("observed wavelength (A)")
    ax.set_ylabel("scatter across galaxies (mad_std)")
    ax.legend(frameon=False, fontsize=8)
    ann = result.get("annulus")
    ax.set_title("background scatter"
                 + (f'   annulus {ann[0]:g}-{ann[1]:g}"' if ann else ""),
                 fontsize=10)

    if ncol == 2:
        ax2 = axes[0, 1]
        q = result["by_nfib_quartile"]
        x = [d["nfib_med"] for d in q.values()]
        y = [d["scatter_med"] for d in q.values()]
        ax2.plot(x, y, "o-", color="tab:blue", label="measured")
        if len(x) >= 2 and x[0] > 0:
            # pure shot noise reference, anchored on the lowest-nfib quartile
            ref = y[0] * np.sqrt(x[0] / np.asarray(x, dtype=float))
            ax2.plot(x, ref, "k--", lw=1.0, label=r"pure shot noise $\propto N^{-1/2}$")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("fibers in annulus (median per quartile)")
        ax2.set_ylabel("median scatter")
        ax2.set_title("is it shot noise?", fontsize=10)
        ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


# =====================================================================
# 3.4  WHAT ACTUALLY REACHES THE PAPER
#      the binned flux profile AND the centroid profile, under each
#      background choice
# =====================================================================

def _sigma_from_band(lo, hi):
    """Half the 16-84 interval as a 1-sigma proxy, matching repo convention."""
    return 0.5 * (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float))


def profile_impact(product, config=None, *, choices=None, n_boot=200,
                   reference=0, keep_stacks=False, verbose=True):
    """
    THE DELIVERABLE: how much does the science result move when the background
    choice changes?

    For each background choice this rebuilds the science cube offline
    (`reconstruct_product`), re-stacks it, and re-measures, then reports the
    change in the two things the paper actually claims:

      * the BINNED FLUX PROFILE -- the measured signal per radial bin
      * the CENTROID PROFILE -- the spectroscopic velocity centroid per bin

    THE CENTROID IS NOT AN AFTERTHOUGHT. The centroid combined with the systemic
    redshift is a primary product of this paper, and background subtraction can
    move it in a way it does not move the flux: subtracting a background that
    still contains residual Lya removes flux preferentially from one side of the
    line, which shifts the measured POSITION as well as the amplitude. A
    background choice that leaves the flux profile untouched can still move the
    centroid. Both are reported and both must be checked.

    WHAT IS DELIBERATELY NOT REPORTED: any change in h1, r_c or the crossover
    radius. The paper's primary science is the measured signal and centroid out
    to several Mpc; the fit is a secondary product for literature comparison,
    and on a coarse bin grid a Delta-h1 is noise wearing a result's clothes.
    Compare bins, and look at the plot. See specs/background-diagnostics.md 0.4.

    THE UNIT OF COMPARISON is the reference's OWN error bar. "The 500-1000 kpc
    bin moved by 0.3 sigma" is a statement a referee can act on; "the flux
    changed by 4.2e36" is not. `max_dflux_sigma` / `max_dcent_sigma` are the
    one-number summaries, and the Methods sentence this function exists to earn
    is: *the profile and the centroid are insensitive to the background annulus
    and to the mask half-width at the <X> sigma level*.

      choices : list of dicts, each `{"label": str, **reconstruct_spec kwargs}`,
                e.g. {"label": '110-130"', "variant": 3} or
                     {"label": "mask 20 A", "variant": 0, "mask_halfwidth_A": 20.0}.
                None -> one choice per stored annulus, sorted by radius.
      n_boot  : bootstrap draws INSIDE the measurement. Deliberately lower than
                the production default -- this is a differential comparison
                between choices that share galaxies and noise, so the error bar
                only has to be good enough to normalise by. Raise it for a
                number that goes in the paper.
      reference : index into `choices` that everything is differenced against.
                Defaults to the first, which is the fiducial annulus.

    RUNTIME: one build_stacks + one measure_all_bins per choice. Roughly half a
    minute per choice at n_boot=200 on ~100 galaxies and a coarse grid, so 4
    choices is a couple of minutes. It scales with n_boot x n_bins.

    Returns dict: rows (one per choice, with flux/centroid arrays and their
    bands), r_edges, r_mid, reference, deltas, and the scalar summaries.
    """
    from dataclasses import replace
    from .stack import build_stacks
    from .measure import measure_all_bins

    if config is None:
        config = PipelineConfig()
    if product.spec_nobg is None or product.bg_variants is None:
        raise ValueError(
            "profile_impact needs SPEC_NOBG and BG_VARIANTS -- it rebuilds the "
            "science cube under each background choice. Re-extract with "
            "store_spec_nobg=True and store_raw_bg=True (2.5)."
        )

    if choices is None:
        choices = [{"label": g["label"] + (" [fid]" if g["is_fiducial"] else ""),
                    "variant": g["variant"]}
                   for g in bg_variant_radii(product, config)]
        # put the fiducial first so it is the natural reference
        choices.sort(key=lambda c: 0 if c["variant"] == 0 else 1)

    cfg_boot = replace(config, n_bootstrap=int(n_boot))
    rows = []
    for k, ch in enumerate(choices):
        kw = {key: val for key, val in ch.items() if key != "label"}
        label = ch.get("label", str(kw))
        if verbose:
            print(f"[profile_impact] {k + 1}/{len(choices)}: {label} ...")
        p2 = reconstruct_product(product, config, **kw)
        stacks = build_stacks(config, p2, keep_cube=True)
        boot = measure_all_bins(cfg_boot, stacks, verbose=False)
        row = {
            "label": label, "kwargs": kw,
            "flux": np.asarray(boot["total_flux_fid"], dtype=float),
            "flux_lo": np.asarray(boot["total_flux_lo"], dtype=float),
            "flux_hi": np.asarray(boot["total_flux_hi"], dtype=float),
            "centroid": np.asarray(boot["centroid_v_fid"], dtype=float),
            "centroid_lo": np.asarray(boot["centroid_v_lo"], dtype=float),
            "centroid_hi": np.asarray(boot["centroid_v_hi"], dtype=float),
            "success_frac": np.asarray(boot["success_frac"], dtype=float),
        }
        if keep_stacks:
            row["stacks"], row["boot"] = stacks, boot
        rows.append(row)

    r_edges = np.asarray(product.r_edges, dtype=float)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    ref = rows[int(reference)]
    sf = _sigma_from_band(ref["flux_lo"], ref["flux_hi"])
    sc = _sigma_from_band(ref["centroid_lo"], ref["centroid_hi"])

    for row in rows:
        with np.errstate(invalid="ignore", divide="ignore"):
            row["dflux"] = row["flux"] - ref["flux"]
            row["dflux_sigma"] = row["dflux"] / sf
            row["dcentroid"] = row["centroid"] - ref["centroid"]
            row["dcentroid_sigma"] = row["dcentroid"] / sc

    result = {
        "rows": rows, "reference": int(reference), "reference_label": ref["label"],
        "r_edges": r_edges, "r_mid": r_mid,
        "ref_flux_sigma": sf, "ref_centroid_sigma": sc,
        "n_boot": int(n_boot), "ngal": product.ngal,
        "max_dflux_sigma": float(np.nanmax([np.nanmax(np.abs(r["dflux_sigma"]))
                                            for r in rows])),
        "max_dcent_sigma": float(np.nanmax([np.nanmax(np.abs(r["dcentroid_sigma"]))
                                            for r in rows])),
    }
    if verbose:
        describe_profile_impact(result)
    return result


def describe_profile_impact(result):
    """Print `profile_impact` as a per-bin table in units of the reference's error."""
    rows, r_edges = result["rows"], result["r_edges"]
    nrad = len(result["r_mid"])
    print("=" * 78)
    print("PROFILE + CENTROID IMPACT OF THE BACKGROUND CHOICE")
    print(f"  {result['ngal']} galaxies, n_boot = {result['n_boot']}, "
          f"reference = {result['reference_label']}")
    print("  entries are (value - reference) / reference's own 1-sigma")
    print("=" * 78)

    for what, key in (("FLUX", "dflux_sigma"), ("CENTROID", "dcentroid_sigma")):
        print(f"\n{what} -- shift in units of the reference error bar")
        print("  " + f"{'bin (kpc)':>16}" +
              "".join(f"{r['label']:>16}" for r in rows if r is not rows[result['reference']]))
        for b in range(nrad):
            cells = "".join(f"{r[key][b]:16.2f}" for r in rows
                            if r is not rows[result["reference"]])
            print(f"  {f'{r_edges[b]:g}-{r_edges[b+1]:g}':>16}" + cells)

    ref = rows[result["reference"]]
    print(f"\nREFERENCE ({ref['label']}) values, for scale")
    print(f"  {'bin (kpc)':>16} {'flux':>14} {'+/-':>12} "
          f"{'centroid km/s':>15} {'+/-':>9}")
    for b in range(nrad):
        print(f"  {f'{r_edges[b]:g}-{r_edges[b+1]:g}':>16} "
              f"{ref['flux'][b]:14.4e} {result['ref_flux_sigma'][b]:12.3e} "
              f"{ref['centroid'][b]:15.1f} {result['ref_centroid_sigma'][b]:9.1f}")

    mf, mc = result["max_dflux_sigma"], result["max_dcent_sigma"]
    print(f"\n  largest flux shift     : {mf:.2f} sigma")
    print(f"  largest centroid shift : {mc:.2f} sigma")
    worst = max(mf, mc)
    if worst < 1.0:
        print("  -> every choice agrees within the errors. That is a COMPLETE "
              "answer and the Methods sentence writes itself: the profile and "
              "the centroid are insensitive to the background choice.")
    elif worst < 2.0:
        print("  -> marginal. Check WHICH bin moves (a single outer bin is a "
              "different story from a coherent shift across radius) and raise "
              "n_boot before concluding anything.")
    else:
        print("  -> a real shift. Identify the bin and the choice, and quote it "
              "as a systematic rather than picking whichever choice looks best "
              "-- selecting on the science profile is the one thing that would "
              "make this circular (2.5.1).")
    return result


def plot_profile_impact(result, *, figsize=(11.5, 4.4), logx=True, logy_flux=True):
    """
    Two panels: the binned flux profile and the centroid profile, every
    background choice overplotted with the reference's error band shaded.

    This is the figure the Methods sentence is read off. If the curves lie on
    top of each other inside the shaded band, the result is insensitive to the
    background choice and there is nothing further to do.
    """
    import matplotlib.pyplot as plt

    rows, r_mid = result["rows"], result["r_mid"]
    ref = rows[result["reference"]]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap("viridis")
    n = max(len(rows), 1)

    for ax, key, lo, hi, sig, ylab in (
            (axes[0], "flux", "flux_lo", "flux_hi", "ref_flux_sigma",
             "integrated flux per bin"),
            (axes[1], "centroid", "centroid_lo", "centroid_hi", "ref_centroid_sigma",
             "centroid (km/s)")):
        ax.fill_between(r_mid, ref[key] - result[sig], ref[key] + result[sig],
                        color="0.7", alpha=0.45, lw=0,
                        label=f"{ref['label']} +/-1 sigma")
        for k, r in enumerate(rows):
            ax.plot(r_mid, r[key], marker="o", ms=3.5, lw=1.3,
                    color=cmap(0.12 + 0.75 * k / max(n - 1, 1)), label=r["label"])
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("radius (bin centre)")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
    axes[1].axhline(0.0, color="0.5", lw=0.8, ls=":")
    if logy_flux:
        axes[0].set_yscale("symlog",
                           linthresh=max(float(np.nanmedian(result["ref_flux_sigma"])), 1e-30))
    axes[0].set_title("flux profile vs background choice", fontsize=10)
    axes[1].set_title("centroid profile vs background choice", fontsize=10)
    fig.tight_layout()
    return fig, axes


# =====================================================================
# FILL-METHOD COMPARISON  --  optimizing mask/FILL/smooth, not the annulus
# =====================================================================
#
# Everything above this line asks "which ANNULUS, and is there a residual
# line in it." This section asks a different question: given the annulus is
# fixed, does the mask -> FILL -> smooth chain that turns a raw annulus coadd
# into what actually gets subtracted preserve the background's own shape,
# and does it remove the line cleanly? That is purely about
# smoothing.smooth_fill_then_gauss's fill step, and deliberately carries NO
# continuum model of any kind (median or poly) -- get_continuum_model /
# cont_method / cont_order belong to line-flux MEASUREMENT (measure.py) and
# have nothing to do with background reconstruction. Conflating the two here
# would misrepresent what the extraction actually does.


def _resolve_variant(product, *, annulus=None, variant=0):
    """
    Variant index from either an explicit int or an (inner, outer) arcsec
    tuple, matched against product.bg_annuli. Raises rather than silently
    falling back to index 0 on a typo'd annulus -- see bg_variant_radii's
    docstring on why variant index and radius order are not the same thing.
    """
    if annulus is None:
        return variant
    for v, (lo, hi) in enumerate(product.bg_annuli):
        if np.isclose(lo, annulus[0]) and np.isclose(hi, annulus[1]):
            return v
    raise ValueError(f"annulus {annulus} not found among {product.bg_annuli}")


def _stack_observed_array(arr_obs, wave_obs, z_all, *, rest_grid=None,
                          combine="biweight", n_boot=1000, seed=0):
    """
    Resample a (ngal, nwave) OBSERVED-frame array to a common REST grid per
    galaxy (each galaxy shifted by its OWN z), then biweight-combine with a
    galaxy bootstrap for the error.

    This is the frame/resample/bootstrap machinery `stack_backgrounds` runs
    on BG_VARIANTS specifically, factored out here so it can run on ANY
    per-galaxy processed array -- raw, masked-and-filled, or
    filled-and-smoothed -- which is exactly what comparing fill methods
    needs. No continuum subtraction step; see the section docstring above.

    THE PLOTTED LINE IS THE FIDUCIAL FULL-SAMPLE COMBINE, not a bootstrap
    statistic -- `stack` comes from combine_fibers on the real per_gal array,
    `err` comes separately from the bootstrap std. Raising n_boot only
    sharpens `err`; it must not move `stack`. (Same fiducial-vs-bootstrap-
    median distinction the centroid plots were fixed for -- see plotting.py.)

    Returns dict: grid, stack, err, per_galaxy, n_eff, frame='rest', z_med,
    ngal, combine, continuum_subtracted=False.
    """
    arr_obs = np.asarray(arr_obs, dtype=float)
    ngal = arr_obs.shape[0]
    z_med = float(np.nanmedian(z_all))
    if rest_grid is None:
        dw_rest = float(np.nanmedian(np.diff(wave_obs))) / (1.0 + z_med)
        rest_grid = np.arange(1140.0, 1300.0 + 1e-9, dw_rest)
    rest_grid = np.asarray(rest_grid, dtype=float)

    per_gal = np.empty((ngal, rest_grid.size), dtype=float)
    for i in range(ngal):
        per_gal[i] = _resample_to_rest(arr_obs[i], wave_obs, z_all[i], rest_grid)

    stack, _ = combine_fibers(per_gal, None, method=combine)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, rest_grid.size), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, ngal, ngal)
        boot[b], _ = combine_fibers(per_gal[idx], None, method=combine)
    err = np.nanstd(boot, axis=0)

    return {
        "grid": rest_grid, "stack": stack, "err": err, "per_galaxy": per_gal,
        "n_eff": np.sum(np.isfinite(per_gal), axis=0), "frame": "rest",
        "z_med": z_med, "ngal": ngal, "combine": combine,
        "continuum_subtracted": False,
    }


def compare_fill_methods(product, config=None, *, annulus=(80.0, 90.0), variant=None,
                         fill_methods=("linear", "median", "median_interp"),
                         mask_halfwidth_A=None, sigma_A=None,
                         fill_sideband_width_A=30.0, n_boot=1000, seed=0,
                         zoom_A=45.0, ylim_pct=(1.0, 99.0), share_ylim=False,
                         figsize=(7, 4.2), verbose=True):
    """
    THE FIGURE for optimizing the fill step: one plot per candidate
    `fill_method`, each showing the RAW / masked-and-FILLED / filled-and-
    SMOOTHED stack for the given annulus, zoomed on the line, each curve with
    its own galaxy-bootstrap error band.

    This tests whether mask -> fill -> smooth preserves the background's own
    shape and removes the line cleanly -- NOT whether there is a residual
    signal in the annulus (that question is stack_backgrounds's). No
    continuum model anywhere (see the section docstring).

    annulus / variant : select the annulus either way; `annulus=(80.,90.)`
        (default, your current fiducial) is resolved against product.bg_annuli
        rather than assumed to be variant 0.
    mask_halfwidth_A, sigma_A : None -> config defaults (10.0, 30.0 A obs).
    fill_sideband_width_A     : shared across "median" and "median_interp";
        mask_halfwidth_A=10 + fill_sideband_width_A=30 -> sidebands at
        [-40,-10] and [10,40] A, matching what you described.
    zoom_A          : rest A shown on either side of the line.
    ylim_pct        : robust percentile of the plotted curves WITHIN the zoom
        window used to set each y-axis. matplotlib does not auto-scale y to a
        manually-set xlim on its own, and these coadds commonly spike hard
        near the wavelength-array edges -- without this, an unzoomed autoscale
        leaves the line region a flat, unreadable smear dominated by those
        edge spikes.
    share_ylim      : False (default) -> each of the 3 figures scaled
        independently, to maximize visibility of wiggles within each one.
        True -> one shared range across all 3, easier to compare amplitude
        method-to-method at the cost of possibly compressing the smallest one.

    Returns dict keyed by fill_method, each holding the 'raw' / 'filled' /
    'smoothed' stack dicts (from _stack_observed_array) plus that method's
    Figure under 'fig'. 'raw' is identical across all three (fill method
    doesn't touch it) and is computed once.
    """
    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError(
            "compare_fill_methods needs BG_VARIANTS -- re-extract with "
            "store_raw_bg=True. (If this product came from combine_products, "
            "check that it actually carried BG_VARIANTS through -- as of this "
            "writing combine_products does not.)"
        )
    if variant is None:
        variant = _resolve_variant(product, annulus=annulus)

    mhw = config.line_mask_halfwidth_A if mask_halfwidth_A is None else mask_halfwidth_A
    sig = config.bg_smooth_sigma_A if sigma_A is None else sigma_A

    wave_obs = np.asarray(product.wave, dtype=float)
    z_all = np.asarray(product.catalog[config.z_col], dtype=float)
    raw_all = np.asarray(product.bg_variants[:, variant, :], dtype=float)
    ngal = raw_all.shape[0]
    ann = product.bg_annuli[variant] if variant < len(product.bg_annuli) else None
    z_med = float(np.nanmedian(z_all))

    if verbose:
        label = f'{ann[0]:g}-{ann[1]:g}"' if ann else f"variant {variant}"
        print(f"[compare_fill_methods] annulus {label} (variant {variant}), "
              f"{ngal} galaxies, mask +/-{mhw:g} A, sigma {sig:g} A, "
              f"fill sideband {fill_sideband_width_A:g} A, n_boot={n_boot}")

    # raw doesn't depend on fill_method -- compute once, reuse in every figure
    raw_stack = _stack_observed_array(raw_all, wave_obs, z_all, n_boot=n_boot, seed=seed)

    results = {}
    all_window_vals = []
    for method in fill_methods:
        filled_all = np.empty_like(raw_all)
        smoothed_all = np.empty_like(raw_all)
        for i in range(ngal):
            filled_all[i] = smooth_fill_then_gauss(
                raw_all[i], wave_obs, z_all[i], lya_rest=LYA_REST, sigma_A=0.0,
                mask_halfwidth_A=mhw, fill_method=method,
                fill_sideband_width_A=fill_sideband_width_A)
            smoothed_all[i] = smooth_fill_then_gauss(
                raw_all[i], wave_obs, z_all[i], lya_rest=LYA_REST, sigma_A=sig,
                mask_halfwidth_A=mhw, fill_method=method,
                fill_sideband_width_A=fill_sideband_width_A)

        filled_stack = _stack_observed_array(filled_all, wave_obs, z_all,
                                             rest_grid=raw_stack["grid"], n_boot=n_boot, seed=seed)
        smoothed_stack = _stack_observed_array(smoothed_all, wave_obs, z_all,
                                               rest_grid=raw_stack["grid"], n_boot=n_boot, seed=seed)
        results[method] = {"raw": raw_stack, "filled": filled_stack, "smoothed": smoothed_stack}

        if share_ylim:
            for st in (raw_stack, filled_stack, smoothed_stack):
                sel = np.abs(st["grid"] - LYA_REST) <= zoom_A
                all_window_vals.append(st["stack"][sel])

    shared_ylim = None
    if share_ylim and all_window_vals:
        allvals = np.concatenate(all_window_vals)
        allvals = allvals[np.isfinite(allvals)]
        if allvals.size:
            lo, hi = np.nanpercentile(allvals, ylim_pct)
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            shared_ylim = (lo - pad, hi + pad)

    import matplotlib.pyplot as plt
    mhw_rest = mhw / (1 + z_med)
    for method in fill_methods:
        r = results[method]
        fig, ax = plt.subplots(figsize=figsize)
        window_vals = []
        for name, color in (("raw", "0.4"), ("filled", "tab:blue"), ("smoothed", "tab:orange")):
            st = r[name]
            grid, y, e = st["grid"], st["stack"], st["err"]
            ax.plot(grid, y, lw=1.4, color=color, label=name)
            ax.fill_between(grid, y - e, y + e, color=color, alpha=0.18, lw=0)
            sel = np.abs(grid - LYA_REST) <= zoom_A
            window_vals.append(y[sel])

        ax.axvspan(LYA_REST - mhw_rest, LYA_REST + mhw_rest, color="crimson", alpha=0.08, lw=0)
        ax.axvline(LYA_REST, color="crimson", lw=0.8, ls="--")
        ax.axhline(0.0, color="0.5", lw=0.7, ls=":")
        ax.set_xlim(LYA_REST - zoom_A, LYA_REST + zoom_A)

        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
        else:
            vals = np.concatenate(window_vals)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo, hi = np.nanpercentile(vals, ylim_pct)
                pad = 0.1 * (hi - lo) if hi > lo else 1.0
                ax.set_ylim(lo - pad, hi + pad)

        label = f'{ann[0]:g}-{ann[1]:g}"' if ann else f"variant {variant}"
        ax.set_title(f'annulus {label},  fill="{method}",  {ngal} gal,  '
                     f'mask +/-{mhw:g} A,  sigma {sig:g} A,  n_boot={n_boot}', fontsize=9.5)
        ax.set_xlabel("rest wavelength (A)")
        ax.set_ylabel("background")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        results[method]["fig"] = fig

    return results


def plot_fill_method_gallery(product, config=None, *, annulus=(80.0, 90.0), variant=None,
                             indices=None, n=12, ncols=4,
                             fill_methods=("linear", "median", "median_interp"),
                             mask_halfwidth_A=None, sigma_A=None,
                             fill_sideband_width_A=30.0, zoom_A=60.0,
                             ylim_pct=(1.0, 99.0), figsize=None, seed=0):
    """
    Per-galaxy version of compare_fill_methods: one panel per galaxy, the raw
    background plus each fill method's SMOOTHED result, zoomed on the line.

    The y-axis fix this exists for: background coadds commonly spike hard
    near the wavelength-array edges (fewer contributing exposures/fibers out
    there), and matplotlib does NOT auto-scale y to match a manually-set
    xlim. A panel zoomed on the line but left to autoscale y still shows the
    full, edge-spike-dominated range, and the line region collapses to a
    flat smear -- which is the exact complaint this function fixes: the
    y-axis is set explicitly from a robust percentile of ONLY the data
    inside the zoom window, per panel.

    Returns (fig, axes).
    """
    import matplotlib.pyplot as plt

    if config is None:
        config = PipelineConfig()
    if product.bg_variants is None:
        raise ValueError("plot_fill_method_gallery needs BG_VARIANTS "
                         "(re-extract with store_raw_bg=True).")
    if variant is None:
        variant = _resolve_variant(product, annulus=annulus)

    mhw = config.line_mask_halfwidth_A if mask_halfwidth_A is None else mask_halfwidth_A
    sig = config.bg_smooth_sigma_A if sigma_A is None else sigma_A

    wave_obs = np.asarray(product.wave, dtype=float)
    z_all = np.asarray(product.catalog[config.z_col], dtype=float)
    raw_all = np.asarray(product.bg_variants[:, variant, :], dtype=float)
    ngal = raw_all.shape[0]

    if indices is None:
        rng = np.random.default_rng(seed)
        n = min(int(n), ngal)
        indices = np.sort(rng.choice(ngal, size=n, replace=False))
    indices = np.atleast_1d(np.asarray(indices, dtype=int))

    colors = {"linear": "tab:blue", "median": "tab:orange", "median_interp": "tab:green"}
    nrows = int(np.ceil(len(indices) / ncols))
    if figsize is None:
        figsize = (3.6 * ncols, 2.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    ann = product.bg_annuli[variant] if variant < len(product.bg_annuli) else None
    for ax, i in zip(axes.ravel(), indices):
        z, raw = z_all[i], raw_all[i]
        lam = LYA_REST * (1 + z)
        x = wave_obs - lam
        sel = np.abs(x) <= zoom_A

        ax.plot(x, raw, lw=0.7, color="0.6", label="raw")
        window_vals = [raw[sel]]
        for method in fill_methods:
            sm = smooth_fill_then_gauss(raw, wave_obs, z, lya_rest=LYA_REST, sigma_A=sig,
                                        mask_halfwidth_A=mhw, fill_method=method,
                                        fill_sideband_width_A=fill_sideband_width_A)
            ax.plot(x, sm, lw=1.2, color=colors.get(method, "k"), label=method)
            window_vals.append(sm[sel])

        ax.axvspan(-mhw, mhw, color="crimson", alpha=0.10, lw=0)
        vals = np.concatenate(window_vals)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.nanpercentile(vals, ylim_pct)
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-zoom_A, zoom_A)
        ax.set_title(f"#{i}  z={z:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(indices):]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    ttl = f'{ann[0]:g}-{ann[1]:g}"' if ann else f"variant {variant}"
    fig.suptitle(f"annulus {ttl}: raw vs smoothed(fill method), "
                f"mask +/-{mhw:g} A, sigma {sig:g} A", fontsize=10)
    fig.tight_layout()
    return fig, axes


# =====================================================================
# 2.6  BACKGROUND IMPACT ON THE SCIENCE BINS
#
# compare_fill_methods (2.5) answers "does the mask->fill->smooth chain
# preserve the BACKGROUND's own shape". This section answers the question
# that actually reaches the paper: once that background is SUBTRACTED, how
# much does the SCIENCE result move -- and specifically in the bins that
# physically overlap the annulus, where the subtraction is partly the bin
# subtracted from itself.
#
# THE TWO QUESTIONS, KEPT SEPARATE ON PURPOSE.
#
#   Q1  does the METHOD bias anything?  linear vs median vs median_interp.
#       A differential question between choices that share galaxies, noise,
#       and everything else. Expected small. Answering it earns a Methods
#       sentence.
#
#   Q2  does subtracting AT ALL remove real signal?  Beyond the annulus's
#       inner radius the science bins overlap the fibers the background was
#       drawn from, so any real halo Lya out there is partly subtracted from
#       itself and the measured flux is driven toward zero regardless of what
#       is present. Expected larger, and it is the one that constrains how far
#       out the paper can claim anything.
#
# Q2 is why the default sweep carries a "no background at all" row. It is free
# (SPEC_NOBG is already stored) and it BRACKETS the whole argument: it is the
# largest the subtraction's effect can possibly be. If the entire method-to-
# method spread is a small fraction of that bracket, Q1 is settled by
# inspection. Without it, a table of small differences between three fill
# methods looks reassuring while saying nothing about whether the background
# should have been subtracted at all.
#
# THE OFF-LINE PEDESTAL is carried as a third metric because it is the only
# NON-CIRCULAR check in here. Every other number can only be compared to the
# fiducial, which assumes the fiducial is right. But the 80-90" annulus holds
# no real stellar continuum -- at those radii the UV continuum is long gone
# (UV scale ~1.5 kpc vs Lya ~16 kpc) -- so after a correctly-scaled
# subtraction the OUTER science bins must have an off-line level consistent
# with zero. A variant that leaves a nonzero pedestal out there is provably
# mis-scaled, with no reference to any other variant.
#
#   CAVEAT, and it matters: this is a clean pass/fail in the OUTER bins ONLY.
#   The inner bins have genuine UV continuum from the galaxies themselves, so
#   a nonzero pedestal there is signal, not error. `pedestal_is_clean` marks
#   which bins the test is valid in rather than leaving it to be remembered.
#
# WHY THIS IS NOT profile_impact. profile_impact sweeps the ANNULUS (which
# variant) and the mask width, and reports flux + centroid. This sweeps the
# FILL METHOD and the subtract/don't-subtract axis, adds the pedestal, and
# adds the per-bin spectrum-overlay figure. The reporting convention -- deltas
# in units of the reference's OWN error bar -- is deliberately identical, so
# the two tables read the same way and can sit side by side in the Methods.
# =====================================================================

def _pedestal_per_bin(rest_wave, stack_flux, config=None, *, cube=None,
                      n_boot=200, seed=1, stack_method="biweight", weights=None):
    """
    The off-line level of each radial bin's stacked spectrum, and its error.

    Uses the SAME continuum windows and the SAME flat-median estimator
    measure.py subtracts (`cont_bounds`, cont_method='median'), so this is
    literally the pedestal the line-flux integration removes -- not a
    second, parallel definition of "continuum" that could drift away from it.

    THE ERROR IS A GALAXY BOOTSTRAP, and the reason matters. The obvious cheap
    estimator -- mad_std(continuum pixels) / sqrt(npix) -- is WRONG here and
    wrong in the dangerous direction. It assumes the continuum-window pixels
    are independent, but the stack is resampled from observed to rest frame
    per galaxy (np.interp, so neighbours are blended) on top of the
    instrumental LSF, so the effective number of independent points is several
    times smaller than npix. That estimator therefore UNDERSTATES the error by
    roughly the square root of the correlation length and turns a ~2 sigma
    pedestal into a ~10 sigma one. It was the first thing tried here and it
    produced exactly that false alarm.

    It is also the wrong error conceptually: the dominant uncertainty on a
    stacked level is galaxy-to-galaxy variation, not pixel noise within one
    already-stacked spectrum. So the error comes from resampling GALAXIES,
    matching every other error bar in this module.

    Cost is kept low by bootstrapping only the continuum-window pixels
    (~tens of pixels) rather than the full rest grid.

      cube : (ngal, nrad, nwave) rest-frame cube, i.e. stacks['cube_flux'].
             None -> no bootstrap is possible and err is returned as NaN
             rather than silently falling back to the bad estimator.
      seed : PASS THE SAME SEED as the measurement bootstrap and the draws are
             the same galaxy sets, so pedestals from different background
             choices are paired and their difference is meaningful.

    Returns (value, err), each (nrad,).
    """
    from .config import DEFAULT_CONT_BOUNDS
    from .measure import stack_galaxies
    if config is None:
        config = PipelineConfig()
    cont_bounds = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)

    rest_wave = np.asarray(rest_wave, dtype=float)
    stack_flux = np.asarray(stack_flux, dtype=float)

    cmask = np.zeros_like(rest_wave, dtype=bool)
    for lo, hi in cont_bounds:
        cmask |= (rest_wave >= lo) & (rest_wave <= hi)

    nrad = stack_flux.shape[0]
    val = np.full(nrad, np.nan)
    for r in range(nrad):
        y = stack_flux[r][cmask]
        y = y[np.isfinite(y)]
        if y.size >= 3:
            val[r] = float(np.median(y))

    err = np.full(nrad, np.nan)
    if cube is None:
        return val, err

    cube = np.asarray(cube, dtype=float)[:, :, cmask]
    ngal = cube.shape[0]
    w_all = None if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.full((int(n_boot), nrad), np.nan)
    for b in range(int(n_boot)):
        idx = rng.integers(0, ngal, ngal)
        w_bs = w_all[idx] if w_all is not None else None
        st, _ = stack_galaxies(cube[idx], None, method=stack_method, weights=w_bs)
        with np.errstate(invalid="ignore"):
            boot[b] = np.nanmedian(st, axis=1)
    err = np.nanstd(boot, axis=0)
    return val, err


def _annulus_kpc_span(product, config=None, variant=0, pct=(2.5, 97.5)):
    """
    Where the annulus lands in kpc -- as a RANGE, not a single radius.

    The annulus is fixed in ARCSEC; the science bins are in KPC; every galaxy
    has its own z. So the inner edge of the 80-90" annulus does not map to one
    kpc value, it maps to a distribution, and the radius beyond which
    self-subtraction is possible is smeared across several bins rather than
    being a line on the plot. Drawing it as a line would overstate how sharp
    the boundary is, which is exactly the kind of thing a referee notices.

    Returns dict: inner_lo, inner_hi, outer_lo, outer_hi, inner_med, outer_med
    (kpc), from the `pct` percentiles of the per-galaxy conversion.
    """
    if config is None:
        config = PipelineConfig()
    ann = product.bg_annuli[variant]
    z = np.asarray(product.catalog[config.z_col], dtype=float)
    z = z[np.isfinite(z)]
    kpc_per_as = cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265.0
    lo_kpc = ann[0] * kpc_per_as
    hi_kpc = ann[1] * kpc_per_as
    return {
        "annulus_arcsec": (float(ann[0]), float(ann[1])),
        "inner_lo": float(np.percentile(lo_kpc, pct[0])),
        "inner_hi": float(np.percentile(lo_kpc, pct[1])),
        "inner_med": float(np.median(lo_kpc)),
        "outer_lo": float(np.percentile(hi_kpc, pct[0])),
        "outer_hi": float(np.percentile(hi_kpc, pct[1])),
        "outer_med": float(np.median(hi_kpc)),
    }


def default_impact_choices(product, config=None):
    """
    The 5-row default sweep: 3 fill methods + the no-background bracket + the
    outermost stored annulus.

    Deliberately ONE KNOB AT A TIME off the fiducial rather than a grid. A
    grid multiplies runtime and, worse, makes every cell a two-knob change so
    nothing can be attributed. Add mask-width / sigma rows only if the fill
    rows turn out to be non-negligible.

      "fiducial"        variant 0, production smoother, no overrides. The
                        reference everything is differenced against, and the
                        row verify_reconstruction proves equals `product.spec`.
      "fill=median"     flat fill from both sidebands pooled
      "fill=median_interp"  robust endpoints, slope preserved
      "no background"   SPEC_NOBG untouched -- the Q2 bracket
      outermost annulus the least-self-subtracted background available

    Returns a list of choice dicts for compare_background_impact.
    """
    if config is None:
        config = PipelineConfig()
    choices = [
        {"label": "fiducial", "variant": 0},
        {"label": "fill=median", "variant": 0, "fill_method": "median"},
        {"label": "fill=median_interp", "variant": 0, "fill_method": "median_interp"},
        {"label": "no background", "no_background": True},
    ]
    try:
        rows = bg_variant_radii(product, config)
        outer = rows[-1]
        if outer["variant"] != 0:
            choices.append({"label": f'annulus {outer["label"]}',
                            "variant": outer["variant"]})
    except Exception:
        pass                      # single-annulus run: the bracket rows still work
    return choices


def compare_background_impact(product, config=None, *, choices=None, n_boot=200,
                              reference=0, keep_stacks=True, stack_method=None,
                              check_linear_equivalence=True, verbose=True):
    """
    THE DELIVERABLE for 2.6: rebuild the science cube under each background
    choice and report how far the per-bin LINE FLUX, the CENTROID and the
    OFF-LINE PEDESTAL move -- with a no-background row that brackets the whole
    effect. See the section docstring for why those three and not others.

      choices : list of dicts, `{"label": str, **reconstruct_spec kwargs}`.
                The extra key `"no_background": True` means "use SPEC_NOBG as
                is" (no subtraction at all) and takes no other kwargs.
                None -> default_impact_choices (the 5-row sweep).
      n_boot  : bootstrap draws inside the measurement. Low on purpose -- this
                is a DIFFERENTIAL comparison between choices sharing galaxies
                and noise, so the error bar only has to be good enough to
                normalise by. Raise it for a number that goes in the paper.
      reference : index into `choices` differenced against. Default 0 = fiducial.
      check_linear_equivalence : assert fill_method="linear" reproduces the
                production path. Cheap, and it is the assumption every delta
                in the table rests on.

    RUNS verify_reconstruction FIRST and raises if it fails. That is not
    politeness: if SPEC_NOBG - smooth(BG_VARIANTS[0]) != SPEC then the
    "fiducial" row is not the thing the paper was measured with, and every
    delta below is a difference against the wrong reference. It is a cheap
    test of an expensive assumption and there is no reason to make remembering
    to run it the user's job.

    PER FIELD. Production extracts COSMOS and EGS separately and combines the
    finalized galaxies afterward, so a background statistic computed on a
    pooled product is not the thing production does. This function does NOT
    split for you -- it runs on whatever product it is handed -- but if that
    product is a merged one, the background variants it builds are still
    per-galaxy and therefore fine. The place pooling bites is a per-wavelength
    robust COMBINE across a bimodal field sample; see compare_fill_methods.

    RUNTIME: one build_stacks + one measure_all_bins per choice, same as
    profile_impact. ~30 s per choice at n_boot=200 on ~100 galaxies.

    Returns dict: rows, r_edges, r_mid, reference, annulus_span,
    pedestal_is_clean, ref_*_sigma, and the scalar summaries.
    """
    from dataclasses import replace
    from .stack import build_stacks
    from .measure import measure_all_bins

    if config is None:
        config = PipelineConfig()
    if product.spec_nobg is None or product.bg_variants is None:
        raise ValueError(
            "compare_background_impact needs SPEC_NOBG and BG_VARIANTS -- it "
            "rebuilds the science cube under each background choice, and the "
            "no-background row IS SPEC_NOBG. Re-extract with "
            "store_spec_nobg=True and store_raw_bg=True (2.5)."
        )

    ver = verify_reconstruction(product, config, verbose=verbose)
    if not ver["ok"]:
        raise RuntimeError(
            "verify_reconstruction FAILED -- the fiducial reconstruction does "
            "not reproduce product.spec, so the 'fiducial' row is not what the "
            "science was measured with and every delta below would be a "
            f"difference against the wrong reference. max|diff| = "
            f"{ver['max_abs_diff']:.3e} vs tol {ver['tol']:.3e}."
        )

    if check_linear_equivalence:
        wave = np.asarray(product.wave, dtype=float)
        z0 = float(np.asarray(product.catalog[config.z_col], dtype=float)[0])
        a = reconstruct_background(product.bg_variants[0, 0], wave, z0, config)
        b = reconstruct_background(product.bg_variants[0, 0], wave, z0, config,
                                   fill_method="linear")
        d = float(np.nanmax(np.abs(a - b)))
        scale = max(float(np.nanmax(np.abs(a))), 1e-30)
        if d / scale > 1e-10:
            raise RuntimeError(
                f'fill_method="linear" does not reproduce the production '
                f"smoother (max|diff|/scale = {d / scale:.2e}). The fill sweep "
                "would then be comparing two things that differ for a second, "
                "unlabelled reason."
            )
        if verbose:
            print(f'[compare_background_impact] fill="linear" == production '
                  f"path (max|diff|/scale = {d / scale:.1e})  OK")

    if choices is None:
        choices = default_impact_choices(product, config)

    sm = stack_method or getattr(config, "measure_stack_method", "biweight")
    cfg_boot = replace(config, n_bootstrap=int(n_boot))
    boot_seed = int(getattr(config, "seed", 1) or 1)

    rows = []
    for k, ch in enumerate(choices):
        kw = {key: val for key, val in ch.items()
              if key not in ("label", "no_background")}
        label = ch.get("label", str(kw))
        if verbose:
            print(f"[compare_background_impact] {k + 1}/{len(choices)}: {label} ...")

        if ch.get("no_background"):
            p2 = replace(product, spec=np.asarray(product.spec_nobg, dtype=float))
        else:
            p2 = reconstruct_product(product, config, **kw)

        stacks = build_stacks(config, p2, keep_cube=True)
        boot = measure_all_bins(cfg_boot, stacks, seed=boot_seed, verbose=False)

        sf = stacks["stacks"][sm]["flux"] if sm in stacks["stacks"] else \
            stacks["stacks"][next(iter(stacks["stacks"]))]["flux"]
        # same seed as the measurement bootstrap -> the pedestal draws are the
        # same galaxy sets across choices, so pedestal differences are paired too
        ped, ped_err = _pedestal_per_bin(
            stacks["rest_wave"], sf, config, cube=stacks.get("cube_flux"),
            n_boot=int(n_boot), seed=boot_seed, stack_method=sm,
            weights=(stacks.get("cube_weights") if sm == "weighted_median" else None))

        row = {
            "label": label, "kwargs": kw,
            "no_background": bool(ch.get("no_background", False)),
            "flux": np.asarray(boot["total_flux_fid"], dtype=float),
            "flux_lo": np.asarray(boot["total_flux_lo"], dtype=float),
            "flux_hi": np.asarray(boot["total_flux_hi"], dtype=float),
            "centroid": np.asarray(boot["centroid_v_fid"], dtype=float),
            "centroid_lo": np.asarray(boot["centroid_v_lo"], dtype=float),
            "centroid_hi": np.asarray(boot["centroid_v_hi"], dtype=float),
            "success_frac": np.asarray(boot["success_frac"], dtype=float),
            "pedestal": ped, "pedestal_err": ped_err,
            "rest_wave": np.asarray(stacks["rest_wave"], dtype=float),
            "stack_flux": np.asarray(sf, dtype=float),
            # per-DRAW arrays (nboot, nrad) -- the raw material for the paired
            # comparison below. Kept, not summarised, because a difference of
            # summaries is not a summary of differences.
            "flux_all": np.asarray(boot.get("total_flux_all"), dtype=float)
                        if boot.get("total_flux_all") is not None else None,
            "centroid_all": np.asarray(boot.get("centroid_v_all"), dtype=float)
                            if boot.get("centroid_v_all") is not None else None,
            "boot_meta": dict(boot.get("meta", {})),
        }
        if keep_stacks:
            row["stacks"], row["boot"] = stacks, boot
        rows.append(row)

    r_edges = np.asarray(product.r_edges, dtype=float)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    ref = rows[int(reference)]
    sf_ = _sigma_from_band(ref["flux_lo"], ref["flux_hi"])
    sc_ = _sigma_from_band(ref["centroid_lo"], ref["centroid_hi"])

    # ---- PAIRED bootstrap on the deltas ------------------------------------
    # THE POINT: every choice is measured on the SAME galaxies with the SAME
    # noise, so `choice - reference` has far less variance than either one
    # alone. Dividing that difference by the reference's ABSOLUTE error (the
    # dflux_sigma column, kept for continuity with profile_impact) therefore
    # systematically makes real, reproducible systematics look marginal.
    #
    # This costs NOTHING extra. bootstrap_all seeds a fresh
    # np.random.default_rng(seed) and draws rng.integers(0, ngal, ngal) once
    # per iteration, consuming no other randomness -- so with the same seed,
    # nboot and ngal, every choice sees the IDENTICAL sequence of galaxy index
    # sets. The draws were already paired; nothing was differencing them.
    # `paired_ok` records that those three conditions actually held rather than
    # assuming it, because if they ever stop holding the sigmas below become
    # nonsense silently.
    ref_meta = ref.get("boot_meta", {})
    for row in rows:
        m = row.get("boot_meta", {})
        row["paired_ok"] = bool(
            m.get("seed") == ref_meta.get("seed") is not None
            and m.get("nboot") == ref_meta.get("nboot")
            and m.get("ngal") == ref_meta.get("ngal"))

    def _paired_sigma(row, ref_row, key):
        a, b = row.get(key), ref_row.get(key)
        if a is None or b is None or not row.get("paired_ok") or a.shape != b.shape:
            return None
        with np.errstate(invalid="ignore"):
            return np.nanstd(a - b, axis=0)

    for row in rows:
        with np.errstate(invalid="ignore", divide="ignore"):
            row["dflux"] = row["flux"] - ref["flux"]
            row["dflux_sigma"] = row["dflux"] / sf_
            row["dcentroid"] = row["centroid"] - ref["centroid"]
            row["dcentroid_sigma"] = row["dcentroid"] / sc_
            row["pedestal_sigma"] = row["pedestal"] / row["pedestal_err"]

            sdf = _paired_sigma(row, ref, "flux_all")
            sdc = _paired_sigma(row, ref, "centroid_all")
            row["dflux_paired_err"] = sdf
            row["dcentroid_paired_err"] = sdc
            row["dflux_paired_sigma"] = (row["dflux"] / sdf) if sdf is not None else None
            row["dcentroid_paired_sigma"] = ((row["dcentroid"] / sdc)
                                             if sdc is not None else None)
            # how much tighter the paired error is than the absolute one --
            # the factor by which the naive column was diluting significance
            row["flux_gain"] = (sf_ / sdf) if sdf is not None else None
            row["centroid_gain"] = (sc_ / sdc) if sdc is not None else None

    span = _annulus_kpc_span(product, config, variant=0)
    # the pedestal test is only non-circular where there is no real UV
    # continuum -- i.e. outside the annulus's inner edge, conservatively the
    # UPPER end of its kpc spread so no bin is claimed clean on a technicality
    clean = r_edges[:-1] >= span["inner_hi"]

    result = {
        "rows": rows, "reference": int(reference), "reference_label": ref["label"],
        "r_edges": r_edges, "r_mid": r_mid,
        "ref_flux_sigma": sf_, "ref_centroid_sigma": sc_,
        "annulus_span": span, "pedestal_is_clean": clean,
        "n_boot": int(n_boot), "ngal": product.ngal, "stack_method": sm,
        "verify": ver,
        "max_dflux_sigma": float(np.nanmax([np.nanmax(np.abs(r["dflux_sigma"]))
                                            for r in rows if r is not ref])),
        "max_dcent_sigma": float(np.nanmax([np.nanmax(np.abs(r["dcentroid_sigma"]))
                                            for r in rows if r is not ref])),
    }
    # the Q1 vs Q2 separation, as one number each
    meth = [r for r in rows if not r["no_background"] and r is not ref
            and "fill_method" in r["kwargs"]]
    brack = [r for r in rows if r["no_background"]]
    result["method_spread_sigma"] = float(np.nanmax(
        [np.nanmax(np.abs(r["dflux_sigma"])) for r in meth])) if meth else np.nan
    result["bracket_sigma"] = float(np.nanmax(
        [np.nanmax(np.abs(r["dflux_sigma"])) for r in brack])) if brack else np.nan
    result["paired_ok"] = all(r["paired_ok"] for r in rows)
    pf = [np.nanmax(np.abs(r["dflux_paired_sigma"])) for r in rows
          if r is not ref and r.get("dflux_paired_sigma") is not None]
    pc = [np.nanmax(np.abs(r["dcentroid_paired_sigma"])) for r in rows
          if r is not ref and r.get("dcentroid_paired_sigma") is not None]
    result["max_dflux_paired_sigma"] = float(np.nanmax(pf)) if pf else np.nan
    result["max_dcent_paired_sigma"] = float(np.nanmax(pc)) if pc else np.nan

    if verbose:
        describe_background_impact(result)
    return result


def describe_background_impact(result):
    """Print `compare_background_impact` as three per-bin tables plus a verdict."""
    rows, r_edges = result["rows"], result["r_edges"]
    ref = rows[result["reference"]]
    others = [r for r in rows if r is not ref]
    nrad = len(result["r_mid"])
    span = result["annulus_span"]

    print("=" * 84)
    print("BACKGROUND IMPACT ON THE SCIENCE BINS")
    print(f"  {result['ngal']} galaxies, n_boot = {result['n_boot']}, "
          f"stack = {result['stack_method']}, reference = {ref['label']}")
    print(f"  annulus {span['annulus_arcsec'][0]:g}-{span['annulus_arcsec'][1]:g}\" "
          f"lands at {span['inner_lo']:.0f}-{span['inner_hi']:.0f} kpc (inner edge, "
          f"2.5-97.5 pct over z) -- bins beyond that are partly self-subtracted")
    print("=" * 84)

    print("\nTWO COLUMNS PER CHOICE, AND THEY ANSWER DIFFERENT QUESTIONS:")
    print("  abs  = (choice - ref) / ref's own 1-sigma   -> DOES IT MATTER?")
    print("         small means the shift is buried in the statistical error.")
    print("  pair = (choice - ref) / sigma of the PAIRED per-draw difference")
    print("         -> IS IT REAL? The choices share galaxies and noise, so")
    print("         their difference is far better determined than either one.")
    print("  A shift can be pair=20 (unmistakably systematic) and abs=0.1")
    print("  (irrelevant). That combination is the good outcome, not a problem.")
    if not result.get("paired_ok", False):
        print("  !! paired columns UNAVAILABLE -- seed/nboot/ngal differ between "
              "choices, so the bootstrap draws are not the same galaxy sets.")

    for what, ka, kp in (("FLUX", "dflux_sigma", "dflux_paired_sigma"),
                         ("CENTROID", "dcentroid_sigma", "dcentroid_paired_sigma")):
        print(f"\n{what}")
        print("  " + f"{'bin (kpc)':>16}"
              + "".join(f"{r['label'][:17]:>20}" for r in others))
        print("  " + f"{'':>16}" + "".join(f"{'abs':>9}{'pair':>11}" for _ in others))
        for b in range(nrad):
            cells = ""
            for r in others:
                p = r.get(kp)
                cells += f"{r[ka][b]:9.2f}" + (f"{p[b]:11.1f}" if p is not None
                                               else f"{'--':>11}")
            print(f"  {f'{r_edges[b]:g}-{r_edges[b + 1]:g}':>16}" + cells)
        gk = "flux_gain" if what == "FLUX" else "centroid_gain"
        # EXCLUDE the no-background row from the gain summary. Its own
        # bootstrap variance is dominated by the un-subtracted background, so
        # Var(choice - ref) ~ Var(choice) >> Var(ref) and the "paired" error
        # comes out LARGER than the absolute one. That is correct arithmetic,
        # not a bug -- pairing only helps when the two things being differenced
        # are comparably noisy AND correlated -- but averaging it in would hide
        # the real gain on the rows where pairing does its job.
        g = [r[gk] for r in others if r.get(gk) is not None and not r["no_background"]]
        if g:
            print(f"  paired error is {np.nanmedian(np.concatenate(g)):.1f}x tighter "
                  "than the absolute one (median, excluding the no-background row)")

    print("\nOFF-LINE PEDESTAL -- level / its own 1-sigma. Should be ~0 in the")
    print("  bins marked [clean]; the others hold real UV continuum, so a")
    print("  nonzero value there is signal and this test does not apply.")
    print("  " + f"{'bin (kpc)':>16}" + "".join(f"{r['label']:>20}" for r in rows))
    for b in range(nrad):
        tag = "[clean]" if result["pedestal_is_clean"][b] else "       "
        cells = "".join(f"{r['pedestal_sigma'][b]:20.2f}" for r in rows)
        print(f"  {f'{r_edges[b]:g}-{r_edges[b + 1]:g}':>9} {tag}" + cells)

    print(f"\nREFERENCE ({ref['label']}) values, for scale")
    print(f"  {'bin (kpc)':>16} {'flux':>14} {'+/-':>12} "
          f"{'centroid km/s':>15} {'+/-':>9}")
    for b in range(nrad):
        print(f"  {f'{r_edges[b]:g}-{r_edges[b + 1]:g}':>16} "
              f"{ref['flux'][b]:14.4e} {result['ref_flux_sigma'][b]:12.3e} "
              f"{ref['centroid'][b]:15.1f} {result['ref_centroid_sigma'][b]:9.1f}")

    ms, br = result["method_spread_sigma"], result["bracket_sigma"]
    print(f"\n  largest flux shift        : {result['max_dflux_sigma']:.2f} sigma (abs)"
          + (f", {result['max_dflux_paired_sigma']:.1f} (paired)"
             if np.isfinite(result.get("max_dflux_paired_sigma", np.nan)) else ""))
    print(f"  largest centroid shift    : {result['max_dcent_sigma']:.2f} sigma (abs)"
          + (f", {result['max_dcent_paired_sigma']:.1f} (paired)"
             if np.isfinite(result.get("max_dcent_paired_sigma", np.nan)) else ""))
    # the verdict is about the choices that COMPETE, so it excludes the
    # no-background bracket -- nobody is going to run without subtracting
    comp = [r for r in others if not r["no_background"]]
    def _mx(key):
        v = [np.nanmax(np.abs(r[key])) for r in comp if r.get(key) is not None]
        return float(np.nanmax(v)) if v else np.nan
    mp = max(_mx("dflux_paired_sigma"), _mx("dcentroid_paired_sigma"))
    ma = max(_mx("dflux_sigma"), _mx("dcentroid_sigma"))
    # these are MAXIMA over competing choices, so the bar is the
    # multiple-comparisons expectation, not 1 sigma. Bins are correlated (one
    # background is subtracted from all of them) so they are not extra trials.
    n_alt = max(len(comp), 1)
    p_fam = _family_p(mp, n_alt)
    result["family_p"] = p_fam
    if np.isfinite(mp):
        print(f"  family-wise p for the largest paired shift ({mp:.1f} sigma over "
              f"{n_alt} competing choice(s)) = {p_fam:.3f}")
    if np.isfinite(mp):
        if p_fam >= 0.05:
            print("  -> consistent with noise. There is no systematic here to "
                  "quote, and because it IS noise, more galaxies will shrink it "
                  "as 1/sqrt(N). Fold in the other field.")
        elif mp > 3 and ma < 1:
            print("  -> the shifts are REAL (paired) but IRRELEVANT (absolute): a "
                  "reproducible systematic buried well inside the statistical "
                  "error. Quote it as a bounded systematic; do not chase it.")
        elif mp > 3 and ma >= 1:
            print("  -> the shifts are real AND comparable to the statistical "
                  "error. These belong in the error budget, added in quadrature "
                  "to the affected bins. More galaxies will NOT average them away.")
        elif mp <= 3:
            print("  -> not even distinguishable in the paired comparison, which "
                  "is the most sensitive test available. This is noise, and more "
                  "data will shrink it.")
    if np.isfinite(ms) and np.isfinite(br):
        print(f"\n  Q1  fill-method spread    : {ms:.2f} sigma")
        print(f"  Q2  no-background bracket : {br:.2f} sigma")
        if br > 0:
            frac = 100.0 * ms / br
            print(f"      method spread is {frac:.0f}% of the bracket")
            if frac < 10:
                print("  -> the fill method is irrelevant next to the decision to "
                      "subtract at all. Q1 is settled; quote the bracket as the "
                      "systematic that matters and move on.")
            else:
                print("  -> the fill method is a non-trivial fraction of the total "
                      "subtraction effect. Identify WHICH bins and quote both "
                      "numbers as separate systematics.")

    bad = [(b, r) for r in rows for b in range(nrad)
           if result["pedestal_is_clean"][b] and abs(r["pedestal_sigma"][b]) > 3]
    if bad:
        print(f"\n  !! {len(bad)} clean-bin pedestal(s) more than 3 sigma from zero. "
              "The annulus holds no real continuum, so a nonzero off-line level "
              "out there is a mis-scaled subtraction, not physics. Worst: "
              + ", ".join(f"{rr['label']} bin {b}" for b, rr in bad[:3]))
    else:
        print("\n  pedestal: every clean bin consistent with zero -- the "
              "subtraction is correctly scaled where that can be tested.")
    return result


def plot_background_impact(result, *, figsize=(12.5, 8.0), logx=True,
                           show_annulus=True):
    """
    THE SUMMARY FIGURE -- three stacked panels vs radius, the one the Methods
    sentence is read off:

      1  d(line flux) / sigma_ref   -- every choice, with the +/-1 sigma band
      2  d(centroid)  / sigma_ref   -- ditto
      3  off-line pedestal, in units of its OWN error, with the bins where
         that test is valid marked

    THE ANNULUS IS SHADED AS A BAND, NOT DRAWN AS A LINE. The annulus is fixed
    in arcsec and the bins are in kpc, so its inner edge maps to a
    z-dependent RANGE (see _annulus_kpc_span). A single vertical line would
    claim a sharpness the data does not have, and would make a gradual rise
    starting before the boundary look like it started after it.

    Returns (fig, axes).
    """
    import matplotlib.pyplot as plt

    rows, r_mid = result["rows"], result["r_mid"]
    ref = rows[result["reference"]]
    span = result["annulus_span"]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ONE colour per row, assigned once and reused in all three panels. Panels
    # 1-2 skip the reference (it is identically zero there) while panel 3 shows
    # it, so a per-panel colour counter puts the same choice in two different
    # colours -- which is exactly the kind of figure that gets misread.
    colors = {}
    ci = 0
    for r in rows:
        if r is ref:
            colors[r["label"]] = "0.25"
        else:
            colors[r["label"]] = cyc[ci % len(cyc)]
            ci += 1

    for ax in axes:
        if show_annulus:
            ax.axvspan(span["inner_lo"], span["inner_hi"], color="crimson",
                       alpha=0.10, lw=0, zorder=0)
            ax.axvspan(span["outer_lo"], span["outer_hi"], color="crimson",
                       alpha=0.05, lw=0, zorder=0)
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
        if logx:
            ax.set_xscale("log")

    for ax, key, lab in ((axes[0], "dflux_sigma", r"$\Delta$ line flux  [$\sigma_{ref}$]"),
                         (axes[1], "dcentroid_sigma", r"$\Delta$ centroid  [$\sigma_{ref}$]")):
        ax.axhspan(-1, 1, color="0.6", alpha=0.18, lw=0, zorder=0)
        for r in rows:
            if r is ref:
                continue
            style = dict(lw=2.2, ls="--") if r["no_background"] else dict(lw=1.6)
            ax.plot(r_mid, r[key], marker="o", ms=4,
                    color=colors[r["label"]], label=r["label"], **style)
        ax.set_ylabel(lab)

    ax = axes[2]
    for r in rows:
        ax.plot(r_mid, r["pedestal_sigma"], lw=1.4, marker="o", ms=4,
                color=colors[r["label"]], label=r["label"],
                ls="--" if r["no_background"] else "-")
        # overplot the bins where the test does NOT apply with a hollow marker
        bad = ~result["pedestal_is_clean"]
        if bad.any():
            ax.plot(np.asarray(r_mid)[bad], r["pedestal_sigma"][bad], "o", ms=7,
                    mfc="none", mec=colors[r["label"]], mew=1.2, ls="none")
    ax.axhspan(-3, 3, color="0.6", alpha=0.15, lw=0, zorder=0)
    # SYMLOG, linear inside +/-10. A badly-scaled variant (no subtraction at
    # all, or the wrong annulus) leaves a pedestal of order 100 sigma, which on
    # a linear axis squashes the 1-5 sigma differences between the variants
    # that actually compete into a single flat line at the bottom. Symlog keeps
    # the +/-3 sigma decision band readable AND shows the blow-ups.
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_ylabel(r"off-line pedestal  [$\sigma_{\rm own}$]")
    ax.set_xlabel("radius (kpc)")

    axes[0].set_title(
        f"background impact vs radius  |  {result['ngal']} gal, "
        f"n_boot={result['n_boot']}, ref = {ref['label']}  |  shaded band = "
        f'{span["annulus_arcsec"][0]:g}" inner edge across the z range',
        fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2)
    axes[2].text(0.01, 0.04,
                 "hollow markers: real UV continuum present, pedestal test does not apply",
                 transform=axes[2].transAxes, fontsize=7.5, color="0.35")
    fig.tight_layout()
    return fig, axes


def plot_background_impact_bins(result, bins=None, *, zoom_A=45.0, ncols=2,
                                figsize=None, ylim_pct=(1.0, 99.0)):
    """
    PER-BIN DETAIL: two rows per selected radial bin -- the stacked spectrum
    under every background choice on top, and (choice - reference) with the
    reference's own +/-1 sigma band below.

    The BOTTOM row is the one that answers the question; the top row is
    context. A difference that stays inside the shaded band is a difference
    you cannot measure, however large it looks on the top panel.

    WHAT TO LOOK FOR in the bottom row, because "is it small" is not the only
    question. If the difference is a FLAT OFFSET, the background choice moved
    the zero point only and the line shape (hence the centroid) is untouched.
    If it has a LYA-SHAPED component, the choice is eating signal. Those are
    very different problems and the delta panel separates them by eye.

      bins : indices of radial bins to show. None -> the outermost 4, which
             are the ones that overlap the annulus and where this matters.
      zoom_A : rest Angstrom either side of Lya.

    Returns (fig, axes).
    """
    import matplotlib.pyplot as plt

    rows, r_edges = result["rows"], result["r_edges"]
    ref = rows[result["reference"]]
    nrad = len(result["r_mid"])
    if bins is None:
        bins = list(range(max(0, nrad - 4), nrad))
    bins = list(bins)

    nrows = int(np.ceil(len(bins) / ncols))
    if figsize is None:
        figsize = (6.2 * ncols, 4.9 * nrows)
    fig = plt.figure(figsize=figsize)
    # NESTED gridspec: a loose outer grid (one cell per radial bin) with a
    # tight 2-row inner grid inside each cell. A single flat 2*nrows grid with
    # one hspace cannot do this -- the same spacing would apply between a
    # bin's own top/bottom pair (which must be flush) and between one bin and
    # the next (which needs room for a title and tick labels), so the titles
    # collide with the axes above them.
    outer = fig.add_gridspec(nrows, ncols, hspace=0.34, wspace=0.24)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    axes = []

    wave = ref["rest_wave"]
    sel = np.abs(wave - LYA_REST) <= zoom_A

    colors = {}
    ci = 0
    for r in rows:
        if r is ref:
            colors[r["label"]] = "0.25"
        else:
            colors[r["label"]] = cyc[ci % len(cyc)]
            ci += 1

    for k, b in enumerate(bins):
        rr, cc = divmod(k, ncols)
        inner = outer[rr, cc].subgridspec(2, 1, height_ratios=[2.2, 1.0],
                                          hspace=0.05)
        ax_t = fig.add_subplot(inner[0])
        ax_b = fig.add_subplot(inner[1], sharex=ax_t)
        axes.append((ax_t, ax_b))

        # reference's per-pixel 1-sigma, if the bootstrap stack error is kept
        err = None
        st = ref.get("boot", {}).get("stack_error") if isinstance(ref.get("boot"), dict) else None
        if isinstance(st, dict):
            for key in ("err", "stack_err", "sigma"):
                if key in st:
                    e = np.asarray(st[key], dtype=float)
                    if e.ndim == 2 and e.shape[0] > b:
                        err = e[b]
                    break

        vals, dvals = [], []
        for r in rows:
            y = r["stack_flux"][b]
            c = colors[r["label"]]
            ax_t.plot(wave, y, lw=1.5 if r is ref else 1.2, color=c,
                      ls="--" if r["no_background"] else "-", label=r["label"])
            vals.append(y[sel])
            if r is not ref:
                d = y - ref["stack_flux"][b]
                ax_b.plot(wave, d, lw=1.2, color=c,
                          ls="--" if r["no_background"] else "-")
                # the bracket row is huge BY CONSTRUCTION (it is the whole
                # background); letting it set the delta-panel scale hides the
                # differences between the choices that actually compete
                if not r["no_background"]:
                    dvals.append(d[sel])

        if err is not None:
            ax_b.fill_between(wave, -err, err, color="0.5", alpha=0.25, lw=0,
                              label=r"ref $\pm1\sigma$")
        for ax in (ax_t, ax_b):
            ax.axvline(LYA_REST, color="crimson", lw=0.8, ls="--")
            ax.set_xlim(LYA_REST - zoom_A, LYA_REST + zoom_A)
        ax_b.axhline(0.0, color="0.4", lw=0.8, ls=":")

        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        if v.size:
            lo, hi = np.nanpercentile(v, ylim_pct)
            pad = 0.12 * (hi - lo) if hi > lo else 1.0
            ax_t.set_ylim(lo - pad, hi + pad)

        if dvals:
            dv = np.concatenate(dvals + ([err, -err] if err is not None else []))
            dv = dv[np.isfinite(dv)]
            if dv.size:
                m = float(np.nanpercentile(np.abs(dv), 99.0))
                if m > 0:
                    ax_b.set_ylim(-1.35 * m, 1.35 * m)

        ax_t.set_title(f"{r_edges[b]:g}-{r_edges[b + 1]:g} kpc", fontsize=9.5)
        ax_t.tick_params(labelbottom=False)
        ax_b.set_xlabel("rest wavelength (A)")
        if cc == 0:
            ax_t.set_ylabel("stacked flux")
            ax_b.set_ylabel("- reference")
        if k == 0:
            ax_t.legend(frameon=False, fontsize=7.5)

    fig.suptitle(f"background choice, per radial bin  |  ref = {ref['label']}  |  "
                 "lower panels: difference vs the reference's own error "
                 "(y-scale excludes the no-background row)",
                 fontsize=10)
    # NOT tight_layout: the paired top/bottom axes come from a nested GridSpec
    # with explicit height_ratios and a deliberately tight inner hspace, and
    # tight_layout overrides that spacing (and warns) -- which is the whole
    # point of the pairing.
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.09, right=0.98)
    return fig, axes


# =====================================================================
# 2.7  IS THE ANNULUS DEPENDENCE REALLY A CONTINUUM-WINDOW ARTIFACT?
#
# THE COUPLING THIS EXISTS TO TEST. The background reconstruction masks the
# line (+/-10 A observed) and then gaussian-smooths with sigma = 30 A
# OBSERVED. At z ~ 2.3 that is a 9.1 A REST kernel, and the production
# continuum windows start at 1205 and 1225 A -- 10.7 and 9.3 A from Lya, i.e.
# barely ONE kernel sigma away, where a gaussian still carries ~60% of peak.
#
# So the broad residual bump that mask-then-smooth leaves behind sits ON TOP
# OF the pixels measure.py uses to define the continuum. Subtracting a
# background therefore does not just remove a background -- it CHANGES THE
# CONTINUUM ESTIMATE, by an amount set by how much Lya that particular
# annulus contained. Different annulus -> different continuum -> different
# line flux, with no self-subtraction involved and no reason for the effect to
# be monotonic in annulus radius.
#
# That is a specific, falsifiable explanation for an annulus dependence that
# otherwise looks alarming, and it has a cheap test: move the continuum
# windows until they clear the kernel and see whether the annulus spread
# collapses. If it does, the fix is one config line (`cont_bounds`) and the
# background was never the problem. If it does not, the annulus dependence is
# real and belongs in the error budget.
#
# WHY THIS IS NOT CIRCULAR. The windows are being moved AWAY from the line, to
# a region with less line contamination -- not tuned to make the answer come
# out flat. The scan reports every window set, including ones that make things
# worse, and the decision rule (spread vs kernel distance) is fixed in advance.
# Choosing the windows that minimise the annulus spread and then quoting that
# spread WOULD be circular; reporting the whole curve is not.
# =====================================================================

DEFAULT_CONT_WINDOW_SETS = {
    "production":   ((1195.0, 1205.0), (1225.0, 1236.0)),
    "red only":     ((1225.0, 1236.0),),
    "stepped out":  ((1180.0, 1195.0), (1245.0, 1260.0)),
    "far":          ((1160.0, 1180.0), (1265.0, 1290.0)),
    "far red only": ((1265.0, 1290.0),),
}


def _family_p(z, n_trials):
    """
    Family-wise two-sided p for the LARGEST of `n_trials` shifts.

    Every "spread" in this section is a maximum over competing choices, so
    comparing it to 1 sigma overstates significance -- the maximum of several
    noise draws is systematically larger than any single draw. A p-value is
    reported rather than a threshold because these tests land in the awkward
    2-2.5 sigma band where "significant / not significant" is a worse summary
    than the number itself.
    """
    from math import erfc, sqrt
    if not np.isfinite(z):
        return np.nan
    p1 = erfc(abs(float(z)) / sqrt(2.0))          # two-sided, per trial
    return float(1.0 - (1.0 - p1) ** max(int(n_trials), 1))


def _z_for_family_p(p_target, n_trials):
    """Inverse of _family_p: the per-shift sigma needed to reach p_target."""
    zz = np.linspace(0.5, 6.0, 4001)
    pp = np.array([_family_p(z, n_trials) for z in zz])
    return float(zz[int(np.argmin(np.abs(pp - p_target)))])


def _kernel_rest_sigma(product, config=None, sigma_A=None):
    """The background smoothing sigma converted to REST Angstrom at median z."""
    if config is None:
        config = PipelineConfig()
    sig = config.bg_smooth_sigma_A if sigma_A is None else sigma_A
    z = float(np.nanmedian(np.asarray(product.catalog[config.z_col], dtype=float)))
    return float(sig) / (1.0 + z), z


def _window_gap_sigmas(cont_bounds, kernel_rest_sigma, center=LYA_REST):
    """
    How far the NEAREST continuum-window edge sits from Lya, in units of the
    background smoothing kernel's rest sigma. This is the x-axis of the whole
    test: below ~2 the windows are inside the residual bump, above ~3 they are
    effectively clear of it.
    """
    d = min(min(abs(lo - center), abs(hi - center)) for lo, hi in cont_bounds)
    return d / kernel_rest_sigma if kernel_rest_sigma > 0 else np.inf


def continuum_window_scan(product, config=None, *, window_sets=None, choices=None,
                          n_boot=200, reference=0, verbose=True):
    """
    Re-run the background comparison under several `cont_bounds` choices and
    report whether the ANNULUS SPREAD collapses once the continuum windows
    clear the smoothing kernel. See the section docstring for the mechanism.

      window_sets : dict {label: cont_bounds}. None -> DEFAULT_CONT_WINDOW_SETS,
                    which walks the windows from the production position
                    (~1 kernel sigma from the line) out to ~4 sigma, with
                    red-only variants because the blue side is the more
                    contaminated one.
      choices     : passed to compare_background_impact. None -> the ANNULUS
                    variants only (not the fill methods), since it is the
                    annulus dependence this is trying to explain. The
                    no-background row is excluded -- it has its own, much
                    larger continuum offset and would dominate every spread.

    RUNTIME: one full compare_background_impact per window set. With the
    default 5 sets and ~4 annuli that is ~20 stack+measure passes, so run it
    at a lower n_boot first and raise it once the shape is clear.

    Returns dict: rows (one per window set, with the per-choice spreads),
    kernel_rest_sigma, z_med, and the verdict fields.
    """
    from dataclasses import replace
    if config is None:
        config = PipelineConfig()
    if window_sets is None:
        window_sets = DEFAULT_CONT_WINDOW_SETS

    if choices is None:
        ann = bg_variant_radii(product, config)
        choices = [{"label": g["label"] + (" [fid]" if g["is_fiducial"] else ""),
                    "variant": g["variant"]} for g in ann]
        choices.sort(key=lambda c: 0 if c["variant"] == 0 else 1)

    k_sig, z_med = _kernel_rest_sigma(product, config)
    if verbose:
        print(f"[continuum_window_scan] kernel sigma = {config.bg_smooth_sigma_A:g} A "
              f"observed = {k_sig:.1f} A REST at z_med = {z_med:.2f}; "
              f"{len(window_sets)} window set(s) x {len(choices)} choice(s)")

    rows = []
    for label, cb in window_sets.items():
        gap = _window_gap_sigmas(cb, k_sig)
        if verbose:
            edges = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in cb)
            print(f"\n[continuum_window_scan] '{label}': {edges} A  "
                  f"(nearest edge {gap:.1f} kernel sigma from Lya)")
        cfg_w = replace(config, cont_bounds=tuple(tuple(b) for b in cb))
        imp = compare_background_impact(product, cfg_w, choices=choices,
                                        n_boot=n_boot, reference=reference,
                                        keep_stacks=False,
                                        check_linear_equivalence=False,
                                        verbose=False)
        ref = imp["rows"][imp["reference"]]
        others = [r for r in imp["rows"] if r is not ref]

        def _spread(key):
            v = [np.nanmax(np.abs(r[key])) for r in others if r.get(key) is not None]
            return float(np.nanmax(v)) if v else np.nan

        rows.append({
            "label": label, "cont_bounds": cb, "gap_sigmas": gap,
            "flux_spread_abs": _spread("dflux_sigma"),
            "flux_spread_paired": _spread("dflux_paired_sigma"),
            "cent_spread_abs": _spread("dcentroid_sigma"),
            "cent_spread_paired": _spread("dcentroid_paired_sigma"),
            "pedestal_clean": np.array([r["pedestal_sigma"] for r in imp["rows"]]),
            "impact": imp,
        })

    result = {"rows": rows, "kernel_rest_sigma": k_sig, "z_med": z_med,
              "n_boot": int(n_boot), "choices": choices,
              "pedestal_is_clean": rows[0]["impact"]["pedestal_is_clean"],
              "r_edges": rows[0]["impact"]["r_edges"]}
    if verbose:
        describe_continuum_window_scan(result)
    return result


def describe_continuum_window_scan(result):
    """Print the scan as spread-vs-window-distance, with an explicit verdict."""
    rows = sorted(result["rows"], key=lambda r: r["gap_sigmas"])
    print("\n" + "=" * 86)
    print("CONTINUUM-WINDOW SENSITIVITY OF THE ANNULUS DEPENDENCE")
    print(f"  background kernel = {result['kernel_rest_sigma']:.1f} A rest "
          f"(z_med {result['z_med']:.2f}), n_boot = {result['n_boot']}")
    print("  'gap' = distance of the nearest window edge from Lya, in kernel sigmas.")
    print("  Below ~2 the windows sit inside the residual bump the background")
    print("  smoothing leaves behind; above ~3 they are effectively clear of it.")
    print("=" * 86)
    print(f"  {'window set':>14} {'gap':>6} | {'FLUX abs':>9} {'FLUX pair':>10} | "
          f"{'CENT abs':>9} {'CENT pair':>10}")
    for r in rows:
        print(f"  {r['label']:>14} {r['gap_sigmas']:6.1f} | "
              f"{r['flux_spread_abs']:9.2f} {r['flux_spread_paired']:10.1f} | "
              f"{r['cent_spread_abs']:9.2f} {r['cent_spread_paired']:10.1f}")

    # ---- GATE: is there an annulus dependence to explain at all? ----------
    # Asking "do the windows explain the spread" before asking "is the spread
    # significant" is how a null result gets reported as a systematic. The
    # spread is a MAXIMUM over choices, so it must be judged against the
    # multiple-comparisons expectation, not against 1 sigma.
    #
    # THE TRIAL COUNT IS THE NUMBER OF ANNULI, NOT annuli x bins. The SAME
    # background spectrum is subtracted from every radial bin of every galaxy,
    # so one noisy background realization shifts all bins coherently -- the
    # bins are close to a single trial, not nrad independent ones. (This is
    # also why "all 7 bins moved the same direction" is NOT evidence of a
    # systematic: under noise it is the expected behaviour.)
    n_alt = max(len(result.get("choices", [])) - 1, 1)
    exp_max = float(np.sqrt(2.0 * np.log(2.0 * n_alt)))
    obs_max = float(np.nanmax([max(r["flux_spread_paired"], r["cent_spread_paired"])
                               for r in rows]))
    p_fam = _family_p(obs_max, n_alt)
    result["expected_max_noise"] = exp_max
    result["observed_max_paired"] = obs_max
    result["family_p"] = p_fam
    result["is_significant"] = bool(p_fam < 0.05)

    print(f"\n  SIGNIFICANCE GATE: largest PAIRED shift anywhere = {obs_max:.1f} sigma.")
    print(f"  Judged against {n_alt} alternative annuli (bins are correlated and do")
    print("  NOT count as separate trials -- one background draw moves them all")
    print(f"  together): family-wise p = {p_fam:.3f}. Reaching p<0.05 would need "
          f"{_z_for_family_p(0.05, n_alt):.1f} sigma.")

    if not result["is_significant"]:
        print("\n  -> THERE IS NO ANNULUS DEPENDENCE TO EXPLAIN. The spread is at or")
        print("     below what noise alone produces, so the window scan below is a")
        print("     null result on a null effect and says nothing about mechanism.")
        print("     Read this as: background annulus choice does not move the result")
        print("     by more than the noise, and because it IS noise, adding galaxies")
        print("     WILL shrink it as 1/sqrt(N).")
        print("\n  Why pairing bought so little here (abs ~= pair): the annuli are")
        print("  DISJOINT sky regions, so their backgrounds are independent noise")
        print("  realizations. spec_A and spec_B share the science data but not the")
        print("  background, which caps their correlation near rho ~ 0.5 and means")
        print("  this comparison is limited by background noise, not by anything")
        print("  systematic. That is a property of the test, not a defect in it.")
        print("\n  CAUTION: 'consistent with noise' is not 'the coupling is absent'.")
        print(f"  The production windows still sit {rows[0]['gap_sigmas']:.1f} kernel sigma "
              "from Lya,")
        print("  so the mechanism is live -- this data just cannot resolve its effect.")
        return result

    near = [r for r in rows if r["gap_sigmas"] < 2.0]
    far = [r for r in rows if r["gap_sigmas"] >= 3.0]
    if near and far:
        fn = np.nanmax([r["flux_spread_abs"] for r in near])
        ff = np.nanmax([r["flux_spread_abs"] for r in far])
        cn = np.nanmax([r["cent_spread_abs"] for r in near])
        cf = np.nanmax([r["cent_spread_abs"] for r in far])
        print(f"\n  flux spread     near windows {fn:.2f} -> far windows {ff:.2f} sigma")
        print(f"  centroid spread near windows {cn:.2f} -> far windows {cf:.2f} sigma")
        drop = (fn - ff) / fn if fn > 0 else 0.0
        if drop > 0.5:
            print(f"  -> the annulus spread FALLS BY {100 * drop:.0f}% when the windows "
                  "clear the kernel. The 'annulus dependence' is largely a "
                  "CONTINUUM-WINDOW artifact: the smoothed background's residual "
                  "bump was being absorbed into the continuum estimate. Move "
                  "cont_bounds out and re-run the science; this is a one-line fix, "
                  "not a background problem.")
        elif drop > 0.2:
            print(f"  -> partial: {100 * drop:.0f}% of the spread is window-driven, "
                  "the rest is not. Move the windows out AND carry the residual "
                  "annulus spread as a systematic.")
        else:
            print("  -> the spread SURVIVES moving the windows out. The annulus "
                  "dependence is real, not a continuum artifact, and belongs in "
                  "the error budget. The next suspect is self-subtraction: check "
                  "which science bins each annulus lands in (bg_variant_radii).")
    else:
        print("\n  -> need window sets both inside (<2 sigma) and clear of (>3 sigma) "
              "the kernel to run the comparison. Widen `window_sets`.")
    print("\n  CAVEAT: moving the windows out is not free. Further from the line")
    print("  means lower S/N per continuum pixel and more room for real spectral")
    print("  features, so pick the CLOSEST windows that clear the kernel rather")
    print("  than the furthest ones available.")
    return result


def plot_continuum_window_scan(result, *, figsize=(11.0, 4.2)):
    """
    Annulus spread vs how far the continuum windows sit from Lya, in kernel
    sigmas. A curve that falls and flattens is the signature of a
    window-driven artifact; a flat curve means the annulus dependence is real.

    Returns (fig, axes).
    """
    import matplotlib.pyplot as plt
    rows = sorted(result["rows"], key=lambda r: r["gap_sigmas"])
    g = [r["gap_sigmas"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for ax, (ka, kp, name) in zip(axes, (("flux_spread_abs", "flux_spread_paired", "FLUX"),
                                         ("cent_spread_abs", "cent_spread_paired", "CENTROID"))):
        ax.plot(g, [r[ka] for r in rows], "o-", lw=1.8, color="tab:blue",
                label=r"/ $\sigma_{\rm ref}$  (does it matter)")
        ax.plot(g, [r[kp] for r in rows], "s--", lw=1.5, color="tab:orange",
                label=r"/ $\sigma_{\rm paired}$  (is it real)")
        ax.axvspan(0, 2, color="crimson", alpha=0.10, lw=0)
        ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("nearest continuum-window edge  [kernel $\\sigma$ from Ly$\\alpha$]")
        ax.set_title(f"{name}: max annulus spread", fontsize=10)
        for r in rows:
            ax.annotate(r["label"], (r["gap_sigmas"], r[ka]), fontsize=7,
                        textcoords="offset points", xytext=(4, 5), color="0.35")
    axes[0].set_ylabel("max spread across annuli")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"shaded: windows inside the background kernel "
                 f"({result['kernel_rest_sigma']:.1f} A rest)", fontsize=9.5)
    fig.tight_layout()
    return fig, axes


# #####################################################################
# #####################################################################
# BACKGROUND SMOOTHERS  (merged from smoothing.py, 2026-08-03)
#
# The per-exposure / galaxy-level background is smoothed before being
# subtracted from every science fiber (an unsmoothed background injects
# annulus shot-noise into all science spectra). extract.extract_fibers
# picks the smoother by config.bg_smooth_method:
#
#   'interp_gauss' (DEFAULT) -> smooth_interp_then_gauss
#        Mask the Lya line, LINEARLY interpolate across the gap, then one
#        uniform gaussian_filter1d. Order matters: interpolating first
#        removes the line from the data, so the smoother sees no gap and
#        no line -- no division-fill bowing, no pulse artifact, no
#        wing-anchoring. Fast (C-level gaussian, no NaN-safe per-pixel
#        division): ~100x cheaper than the adaptive smoother, which is
#        what made background smoothing a multi-hour bottleneck on the
#        big extractions.
#
#   'adaptive' (LEGACY)      -> smooth_spectrum_nan_safe_adaptive
#        NaN-safe gaussian with a position-dependent kernel that ramps
#        from a small edge sigma to a large interior sigma. Kept for A/B
#        comparison.
#
#   smooth_fill_then_gauss  -- DIAGNOSTIC / Stage-2 ONLY, not in the
#        config.bg_smooth_method dispatch above. Generalizes
#        smooth_interp_then_gauss's fill step to three candidate methods
#        ('linear' == the production behavior, 'median', 'median_interp')
#        so they can be A/B'd against each other before any of them
#        becomes a production default. Driven by compare_fill_methods /
#        plot_fill_method_gallery, both in THIS module.
#
# NOTE the import direction: extract.py imports these lazily, inside
# extract_fibers._maybe_smooth, because this module imports extract.py at
# module scope. Do not add a top-level `from .background import ...` to
# extract.py.
# #####################################################################
# #####################################################################

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
