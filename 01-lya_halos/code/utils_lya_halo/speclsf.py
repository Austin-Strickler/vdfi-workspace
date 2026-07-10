"""
speclsf.py -- spectral line-spread function (LSF) of the Lya stack, and a
forward-model hook for recovering a (very general) INTRINSIC line profile.

Why this is its own module (not part of starpsf.py): the PSF is a SPATIAL
profile vs radius; the LSF is a 1-D SPECTRAL kernel. They are different objects
with different downstream consumers -- the PSF feeds the radial/stacking side,
the LSF feeds measure.py (it sets the width floor your line_moments / Gaussian
sigmas sit on). So this file's natural dependency is the STACK + MEASURE layer,
not the PSF machinery.

The key physics, and why the L/kpc^2 flux scaling does NOT complicate the width:
each galaxy's instrumental LSF lives in OBSERVED wavelength. Stage 2 resamples
the spectral axis to rest frame by lambda/(1+z), so a fixed observed kernel of
FWHM_obs becomes FWHM_obs/(1+z) in REST Angstrom -- narrower for higher-z
galaxies. The per-galaxy L/kpc^2 conversion is a multiplicative AMPLITUDE factor
(4*pi*D_L^2*(1+z)/area); it rescales each galaxy's kernel but cannot change its
width in Angstrom. So the stacked effective LSF is the redshift- and
amplitude-weighted BLEND of per-galaxy kernels of differing rest widths (itself
slightly non-Gaussian and broader than the narrowest), and to reproduce it we
only need the SAME resample + coaddition the science line gets -- which is
exactly what stack_lsf does by routing synthetic kernels through stack.py's own
convert_avg_fiber_bin -> stack_spec_values -> galaxy combine.

Redshift uncertainty: MOSDEF systemic redshifts carry ~60 km/s error, which
randomly offsets each galaxy's rest-frame line center and so adds a further
(redshift-independent in velocity) broadening to the STACK. It is folded in here
per galaxy, in quadrature, via z_err_kms.

Functions
---------
    instrumental_fwhm        -- VIRUS observed-frame FWHM(lambda) (const-FWHM / const-R / curve)
    lsf_line_template        -- one galaxy's observed-frame kernel (unit area)
    stack_lsf                -- push per-galaxy kernels through the SAME stack -> rest-frame LSF
    intrinsic_sigma          -- analytic single-Gaussian width deconvolution
    deconvolve_intrinsic_profile  -- forward-model single/double Gaussian through the LSF,
                                     with automatic single-vs-double selection

Notebook usage
--------------
    from utils_lya_halo import speclsf, analysis

    # 1. the stacked rest-frame LSF (uses the same product Stage 2 stacks)
    lsf = speclsf.stack_lsf(cfg, product, z_err_kms=60.0)
    print(f"stacked LSF: FWHM={lsf['fwhm_A']:.2f} A  "
          f"sigma={lsf['sigma_kms']:.0f} km/s")

    # plot it next to the measured stacked line
    import matplotlib.pyplot as plt
    plt.plot(lsf['rest_wave'], lsf['lsf'], label='stacked LSF')
    plt.axvline(1215.67, color='magenta', lw=0.8)
    plt.xlim(1215.67-8, 1215.67+8); plt.xlabel('rest wavelength [A]'); plt.legend()
    plt.show()

    # 2. deconvolve a measured width (single-Gaussian, analytic+forward check)
    #    boot = analysis.run_measurement_suite(cfg, stacks)
    #    sig_obs_A = boot['moments']['sigma_A'][radial_bin]
    #    sig_int = speclsf.intrinsic_sigma(sig_obs_A, lsf['sigma_A'])

    # 3. forward-model the intrinsic profile per radial bin, auto single-vs-double
    rw   = stacks['rest_wave']
    flux = stacks['stacks']['biweight']['flux'][radial_bin]      # the stacked line
    ferr = boot['stack_error']['stack_err'][radial_bin]          # per-pixel error
    res  = speclsf.deconvolve_intrinsic_profile(rw, flux, ferr, lsf, model='both')
    print(res['selected'], res['comparison']['reason'])

    import matplotlib.pyplot as plt
    plt.step(rw, flux, where='mid', color='k', lw=1, label='stacked line')
    plt.plot(rw, res['single']['model'], label='single (intrinsic ⊗ LSF)')
    if res['double'] and res['double']['success']:
        plt.plot(rw, res['double']['model'], '--', label='double')
    plt.xlim(1215.67-6, 1215.67+6); plt.legend(); plt.show()
"""

from __future__ import annotations

import numpy as np

from .measure import LYA_REST, C_KMS
# stack.py is imported lazily inside stack_lsf so the pure-template /
# deconvolution helpers stay usable without pulling in the Stage-2 chain.

_FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.35482

# VIRUS / HETDEX defaults (verify per data release):
#   Hill et al. 2021: FWHM ~ 4.7 A roughly CONSTANT in wavelength, R 750->950
#   cross-power papers: R ~= 800 -> FWHM 5.6 A at 4500 A.
# Default to a constant observed-frame FWHM (the more physical statement); the
# const-R mode is available for sensitivity tests.
VIRUS_FWHM_A = 5.6      # observed-frame FWHM (Angstrom)
VIRUS_R = 800           # resolving power (used only in mode='R')


# =====================================================================
# 1. INSTRUMENTAL LSF WIDTH + TEMPLATE (observed frame, per galaxy)
# =====================================================================

def instrumental_fwhm(wave_obs, mode="fwhm", fwhm_A=VIRUS_FWHM_A, R=VIRUS_R,
                      curve=None):
    """
    Observed-frame instrumental FWHM (Angstrom) at wavelength(s) wave_obs.

    mode :
      'fwhm'  -- constant FWHM in Angstrom (= fwhm_A). DEFAULT. Matches the
                 Hill+2021 statement that VIRUS's FWHM is ~constant (~4.7-5.6 A)
                 while R varies; here the only rest-frame variation comes from
                 the (1+z) resample.
      'R'     -- constant resolving power: FWHM = wave_obs / R (so FWHM GROWS to
                 the red). Use for a sensitivity test against the const-FWHM model.
      'curve' -- interpolate a measured (wavelength, FWHM) table passed as
                 curve=(wl_array, fwhm_array); linear, edge-held.

    Returns a scalar or array matching wave_obs.
    """
    wave_obs = np.asarray(wave_obs, dtype=float)
    if mode == "fwhm":
        return np.full_like(wave_obs, float(fwhm_A)) if wave_obs.ndim else float(fwhm_A)
    if mode == "R":
        return wave_obs / float(R)
    if mode == "curve":
        if curve is None:
            raise ValueError("mode='curve' needs curve=(wl_array, fwhm_array)")
        wl, fw = np.asarray(curve[0], float), np.asarray(curve[1], float)
        return np.interp(wave_obs, wl, fw)
    raise ValueError("mode must be 'fwhm', 'R', or 'curve'")


def lsf_line_template(wave_obs, lambda_obs, fwhm_obs, z_err_kms=0.0,
                      normalize="area"):
    """
    One galaxy's observed-frame LSF kernel on the grid wave_obs.

    A Gaussian centered at lambda_obs (the galaxy's OBSERVED Lya,
    LYA_REST*(1+z)) whose width is the instrumental FWHM broadened in quadrature
    by the redshift-error velocity:

        sigma_obs = sqrt( (fwhm_obs/2.355)^2 + (z_err_kms/c * lambda_obs)^2 )

    The z_err term is the observed-frame image of the systemic-redshift velocity
    error; after Stage 2 divides the axis by (1+z) it becomes the (1+z)-invariant
    rest velocity spread you wanted to keep track of.

    normalize : 'area' (unit integral, default), 'peak' (unit max), or None.

    Returns the kernel, same shape as wave_obs.
    """
    wave_obs = np.asarray(wave_obs, dtype=float)
    sig_inst = float(fwhm_obs) / _FWHM_OVER_SIGMA
    sig_z = (float(z_err_kms) / C_KMS) * float(lambda_obs)
    sigma = np.hypot(sig_inst, sig_z)
    g = np.exp(-0.5 * ((wave_obs - float(lambda_obs)) / sigma) ** 2)
    if normalize == "area":
        dw = np.gradient(wave_obs)
        area = np.nansum(g * dw)
        if area > 0:
            g = g / area
    elif normalize == "peak":
        gmax = np.nanmax(g)
        if gmax > 0:
            g = g / gmax
    return g


# =====================================================================
# 2. STACKED REST-FRAME LSF  (routes kernels through the SAME stack path)
# =====================================================================

def _profile_width(rest_wave, prof, center=LYA_REST, half_window=8.0):
    """
    sigma (2nd moment, area-weighted) and FWHM (half-max crossings) of a profile
    within +/- half_window of center. Area-normalizes internally; NaN-safe.
    Returns (sigma_A, fwhm_A, peak_A).
    """
    rest_wave = np.asarray(rest_wave, float)
    prof = np.asarray(prof, float)
    m = (rest_wave >= center - half_window) & (rest_wave <= center + half_window) \
        & np.isfinite(prof)
    if np.count_nonzero(m) < 3:
        return np.nan, np.nan, np.nan
    x, y = rest_wave[m], np.clip(prof[m], 0, None)
    tot = np.nansum(y)
    if not (tot > 0):
        return np.nan, np.nan, np.nan
    mu = np.nansum(x * y) / tot
    sigma = np.sqrt(max(np.nansum(y * (x - mu) ** 2) / tot, 0.0))
    # FWHM via half-max crossings around the peak
    ipk = int(np.nanargmax(y))
    half = 0.5 * y[ipk]
    xpk = x[ipk]
    # left crossing
    xl = x[0]
    for i in range(ipk, 0, -1):
        if y[i - 1] <= half <= y[i]:
            xl = np.interp(half, [y[i - 1], y[i]], [x[i - 1], x[i]])
            break
    xr = x[-1]
    for i in range(ipk, len(x) - 1):
        if y[i + 1] <= half <= y[i]:
            xr = np.interp(half, [y[i + 1], y[i]], [x[i + 1], x[i]])
            break
    fwhm = float(xr - xl)
    return float(sigma), fwhm, float(xpk)


def stack_lsf(config, product, lsf_mode="fwhm", fwhm_A=VIRUS_FWHM_A, R=VIRUS_R,
              fwhm_curve=None, z_err_kms=0.0, method=None,
              apply_flux_scaling=True, half_window=8.0):
    """
    The STACKED rest-frame LSF: build each galaxy's observed-frame kernel, then
    route them through the SAME Stage-2 path the science line takes
    (convert_avg_fiber_bin amplitude scaling -> stack_spec_values rest resample
    -> galaxy combine), so the redshift blend and the coadd weighting are
    identical to the data.

    Modeling assumptions (read these):
      * Each galaxy is given a UNIT-area line, i.e. we assume equal intrinsic
        line luminosity across galaxies. With apply_flux_scaling=True the
        per-galaxy L/kpc^2 factor still reweights them exactly as the science
        coadd does; the only thing assumed equal is the intrinsic SHAPE+area.
        (A v2 could weight by each galaxy's measured line flux -- see notes.)
      * For a LINEAR combine ('mean','inv_var') the result is the exact effective
        LSF. For 'biweight'/'median' it is the linear-equivalent approximation
        (a nonlinear combiner has no single exact LSF); default follows
        config.measure_stack_method but you can force method='mean' for the clean
        linear answer.

    Parameters
    ----------
    config   : PipelineConfig (gives z_col, flux unit + rest grid params)
    product  : GalaxyProduct (gives per-galaxy z and the observed WAVE grid)
    lsf_mode, fwhm_A, R, fwhm_curve : forwarded to instrumental_fwhm
    z_err_kms : systemic-redshift velocity error to fold in (MOSDEF ~ 60)
    method    : galaxy combine; None -> config.measure_stack_method or 'biweight'
    apply_flux_scaling : route through convert_avg_fiber_bin (recommended;
                         reproduces the science amplitude weighting)
    half_window : +/- A around Lya used for the width measurement

    Returns
    -------
    dict: rest_wave, lsf (unit-area), sigma_A, fwhm_A, sigma_kms, fwhm_kms,
          peak_A, meta{...}
    """
    from .measure import stack_galaxies   # local: keeps import light
    from .stack import rest_grid, convert_avg_fiber_bin, stack_spec_values

    z = np.asarray(product.catalog[config.z_col], dtype=float)
    obs_wave = np.asarray(product.wave, dtype=float)
    ngal = z.size
    lya_obs = LYA_REST * (1.0 + z)

    fwhm_obs = instrumental_fwhm(lya_obs, mode=lsf_mode, fwhm_A=fwhm_A, R=R,
                                 curve=fwhm_curve)
    fwhm_obs = np.broadcast_to(np.asarray(fwhm_obs, float), (ngal,))

    # per-galaxy observed-frame kernels (unit area), (ngal, nwave_obs)
    kernels = np.empty((ngal, obs_wave.size), dtype=float)
    for i in range(ngal):
        if not np.isfinite(z[i]):
            kernels[i] = np.nan
            continue
        kernels[i] = lsf_line_template(obs_wave, lya_obs[i], fwhm_obs[i],
                                       z_err_kms=z_err_kms, normalize="area")

    # amplitude scaling exactly as the science line (per-galaxy L/kpc^2 factor)
    if apply_flux_scaling:
        y, ye, _ = convert_avg_fiber_bin(
            kernels, np.zeros_like(kernels), z,
            output=config.flux_unit, flux_unit_scale=config.flux_unit_scale,
            fiber_diam_arcsec=config.fiber_diam_arcsec,
            rest_density=config.rest_density)
    else:
        y, ye = kernels, np.zeros_like(kernels)

    # rest-frame resample with the SAME interpolation Stage 2 uses
    rest_wave = rest_grid(config)
    f_rf, e_rf = stack_spec_values(y, ye, z, obs_wave, rest_wave)

    # galaxy combine (reshape to a 1-bin cube to reuse the science combiner)
    sm = method or getattr(config, "measure_stack_method", "biweight")
    stack, _ = stack_galaxies(f_rf[:, None, :], e_rf[:, None, :], method=sm)
    lsf_prof = stack[0]

    # normalize to unit area within the window, then measure width
    m = (rest_wave >= LYA_REST - half_window) & (rest_wave <= LYA_REST + half_window)
    dw = float(np.nanmedian(np.diff(rest_wave)))
    area = np.nansum(np.clip(lsf_prof[m], 0, None)) * dw
    if area > 0:
        lsf_prof = lsf_prof / area

    sigma_A, fwhm_meas, peak_A = _profile_width(rest_wave, lsf_prof,
                                                center=LYA_REST,
                                                half_window=half_window)
    to_kms = C_KMS / LYA_REST
    return {
        "rest_wave": rest_wave,
        "lsf": lsf_prof,
        "sigma_A": sigma_A,
        "fwhm_A": fwhm_meas,
        "sigma_kms": sigma_A * to_kms if np.isfinite(sigma_A) else np.nan,
        "fwhm_kms": fwhm_meas * to_kms if np.isfinite(fwhm_meas) else np.nan,
        "peak_A": peak_A,
        "meta": {"ngal": int(ngal), "lsf_mode": lsf_mode, "fwhm_A_input": fwhm_A,
                 "R": R, "z_err_kms": z_err_kms, "method": sm,
                 "apply_flux_scaling": apply_flux_scaling,
                 "z_median": float(np.nanmedian(z))},
    }


def intrinsic_sigma(sigma_obs_A, sigma_lsf_A):
    """
    Analytic single-Gaussian deconvolution: sqrt(sigma_obs^2 - sigma_lsf^2),
    NaN where the line is unresolved (sigma_obs <= sigma_lsf). Quick check /
    fallback for the forward model below. Inputs/output in Angstrom.
    """
    so = np.asarray(sigma_obs_A, float)
    var = so ** 2 - float(sigma_lsf_A) ** 2
    return np.where(var > 0, np.sqrt(np.clip(var, 0, None)), np.nan)


# =====================================================================
# 3. FORWARD-MODEL DECONVOLUTION  [STUB / OUTLINE -- single works, double TODO]
# =====================================================================

def _lsf_kernel_from_stack(lsf_result, oversample=1):
    """Recenter the stacked LSF to 0, clip to the window, return a unit-area
    kernel on the rest grid spacing (for np.convolve)."""
    rw = np.asarray(lsf_result["rest_wave"], float)
    k = np.clip(np.asarray(lsf_result["lsf"], float), 0, None)
    pk = lsf_result.get("peak_A", LYA_REST)
    m = np.isfinite(k) & (np.abs(rw - pk) <= 8.0)
    kk = k[m]
    s = kk.sum()
    return kk / s if s > 0 else kk


def _gauss(x, amp, mu, sig):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def deconvolve_intrinsic_profile(rest_wave, stack_flux, stack_err, lsf_result,
                                 model="both", bounds_window=4.0, sigma_hi=8.0,
                                 baseline=True, select="ftest", alpha=0.05,
                                 delta_bic_thresh=10.0, delta_aic_thresh=2.0,
                                 p0_single=None, peak_guess=None, verbose=True):
    """
    Forward-model the INTRINSIC Lya profile: convolve a parametric model with the
    empirically stacked LSF and fit to the stacked line.

        model(lambda) = [ intrinsic(lambda; theta) (x) stacked_LSF ] + baseline
        chi^2 = sum( (model - data)^2 / err^2 ) over the +/-bounds_window window

    model :
      'single' -- one Gaussian (amp, mu, sigma_int).
      'double' -- blue+red Gaussians, parametrized mu_r = mu_b + dsep (dsep>=0)
                  so the red peak can't cross the blue one (no label-swap).
      'both'   -- DEFAULT. Fit both and AUTO-SELECT, keeping the double only if it
                  is justified given its 3 extra DOF, by `select`:
                    'ftest' -- nested-model F-test, prefer double if p < alpha
                    'bic'   -- prefer double if BIC_single - BIC_double > delta_bic_thresh
                    'aic'   -- prefer double if AIC_single - AIC_double > delta_aic_thresh
                  Guardrails: double that fails to converge, fails to reduce chi^2,
                  or is degenerate (peaks merged / one amp < 5%) -> single.

    baseline : add a constant continuum term c to the model (default True). Your
               stacked line sits on a ~non-zero pseudo-continuum; without c the
               model is forced through 0 and the wings/trough are mis-fit (which
               also sabotages the single-vs-double decision).

    peak_guess : optional physical prior for the DOUBLE, used as the first
               multi-start seed (the lowest-chi^2 seed still wins). A dict:
                 blue_offset : blue-peak offset from systemic [A]   (e.g. -1.9)
                 red_offset  : red-peak  offset from systemic [A]   (e.g. +1.0)
                 flux_ratio  : blue_amp / red_amp                   (e.g. 0.45)
                 sigma       : intrinsic sigma guess for both [A]
               This is how you feed in a by-eye fit; peaks are otherwise found
               automatically with scipy.signal.find_peaks (prominence-based, so a
               strong peak's wing is not mistaken for the second peak). c is fit, reported
               as result['baseline'], and is NOT part of the returned 'intrinsic'
               line.

    Double-fit robustness: the double is the part that's easy to get wrong. A
    single broad seed slides into a merged blob that fits WORSE than the single
    (chi^2 up, parameters wasted) and the auto-selector then -- correctly but
    unhelpfully -- keeps the single. To avoid that we (a) seed the two components
    at the actual blue-side and red-side peaks of the data, narrow, and (b)
    MULTI-START from several separations and keep the lowest chi^2. This is what
    lets a visibly double-peaked bin actually be recovered.

    Honest expectations: single-Gaussian width recovery is well posed when
    sigma_obs is safely above sigma_LSF; the double split is only as good as the
    stack_err calibration and the LSF width (~1.5-1.9 A rest). A wide, deep trough
    recovers cleanly; a shallow one will stay marginal.

    Returns
    -------
    model='single'/'double' : that fit's result dict.
    model='both' : {"selected", "single", "double", "comparison"}.
    All fits carry: success, reason, chi2, ndof, chi2_red, aic, bic, n, popt,
    pcov, baseline, intrinsic (un-convolved line on the grid), model (full curve).
    """
    from scipy.optimize import curve_fit
    from scipy.stats import f as _fdist

    rest_wave = np.asarray(rest_wave, float)
    data = np.asarray(stack_flux, float)
    err = np.asarray(stack_err, float)
    kernel = _lsf_kernel_from_stack(lsf_result)

    win = (rest_wave >= LYA_REST - bounds_window) & (rest_wave <= LYA_REST + bounds_window)
    yv, ev = data[win], err[win]
    good = np.isfinite(yv) & np.isfinite(ev) & (ev > 0)
    n = int(good.sum())
    xg, yg, eg = rest_wave[win][good], yv[good], ev[good]
    dw = float(np.nanmedian(np.diff(rest_wave)))
    nb = 1 if baseline else 0   # number of baseline params

    # robust continuum seed from the window edges (lowest decile of the window)
    c_seed = float(np.nanpercentile(yg, 10)) if n else 0.0

    def _conv(g):
        return np.convolve(g, kernel, "same")

    # models on the FULL grid (edge pixels convolve correctly), sliced to fit pts.
    # baseline c is added AFTER convolution (a constant (x) unit-area kernel == c).
    def m_single(_x, *p):
        amp, mu, sig = p[:3]; c = p[3] if baseline else 0.0
        return (_conv(_gauss(rest_wave, amp, mu, sig)) + c)[win][good]

    def m_double(_x, *p):
        amp_b, mu_b, sig_b, amp_r, dsep, sig_r = p[:6]; c = p[6] if baseline else 0.0
        g = _gauss(rest_wave, amp_b, mu_b, sig_b) + _gauss(rest_wave, amp_r, mu_b + dsep, sig_r)
        return (_conv(g) + c)[win][good]

    def _full_single(p):
        amp, mu, sig = p[:3]; c = p[3] if baseline else 0.0
        line = _gauss(rest_wave, amp, mu, sig)
        return _conv(line) + c, line, c

    def _full_double(p):
        amp_b, mu_b, sig_b, amp_r, dsep, sig_r = p[:6]; c = p[6] if baseline else 0.0
        line = _gauss(rest_wave, amp_b, mu_b, sig_b) + _gauss(rest_wave, amp_r, mu_b + dsep, sig_r)
        return _conv(line) + c, line, c

    def _metrics(mfunc, popt, k):
        resid = (yg - mfunc(xg, *popt)) / eg
        chi2 = float(np.nansum(resid ** 2))
        ndof = int(n - k)
        return chi2, ndof, chi2 + 2 * k, chi2 + k * np.log(max(n, 1))   # chi2, ndof, AIC, BIC

    def _b(*p):   # append baseline seed if enabled
        return list(p) + ([c_seed] if baseline else [])

    def _bnd_lo(core):   # append baseline lower bound (-inf) if enabled
        return core + ([-np.inf] if baseline else [])

    def _bnd_hi(core):   # append baseline upper bound (+inf) if enabled
        return core + ([np.inf] if baseline else [])

    # ---- single ----
    def _fit_single():
        if n < 5 + nb:
            return {"success": False, "reason": "too few finite pixels", "model_name": "single"}
        amp0 = max(float(np.nanmax(yg) - c_seed), 1e-30)
        p0 = p0_single or _b(amp0, LYA_REST, 1.0)
        lo = _bnd_lo([0.0, LYA_REST - bounds_window, 0.05])
        hi = _bnd_hi([np.inf, LYA_REST + bounds_window, sigma_hi])
        try:
            popt, pcov = curve_fit(m_single, xg, yg, p0=p0, sigma=eg,
                                   absolute_sigma=True, bounds=(lo, hi), maxfev=20000)
        except Exception as exc:
            return {"success": False, "reason": f"single fit failed: {exc}", "model_name": "single"}
        chi2, ndof, aic, bic = _metrics(m_single, popt, 3 + nb)
        amp, mu, sig = (float(v) for v in popt[:3])
        full, line, c = _full_single(popt)
        return {"success": True, "reason": None, "model_name": "single",
                "amp": amp, "mu": mu, "sigma_int_A": sig,
                "sigma_int_kms": C_KMS * sig / LYA_REST, "baseline": float(c),
                "chi2": chi2, "ndof": ndof, "chi2_red": chi2 / max(ndof, 1),
                "aic": aic, "bic": bic, "n": n, "popt": popt, "pcov": pcov,
                "intrinsic": line, "model": full}

    # ---- double (data-driven multi-start) ----
    def _fit_double(seed=None):
        if n < 7 + nb:
            return {"success": False, "reason": "too few finite pixels for double", "model_name": "double"}
        lo = _bnd_lo([0.0, LYA_REST - bounds_window, 0.05, 0.0, 0.0, 0.05])
        hi = _bnd_hi([np.inf, LYA_REST + bounds_window, sigma_hi, np.inf,
                      2 * bounds_window, sigma_hi])

        # ---- locate the two intrinsic peaks to seed from ----
        yc = yg - c_seed
        amax_data = max(float(np.nanmax(yc)), 1e-30)

        # (i) proper LOCAL maxima via find_peaks (prominence + min separation),
        #     so a strong peak's wing is NOT mistaken for the other peak.
        mu_b0 = mu_r0 = None
        try:
            from scipy.signal import find_peaks
            min_dist = max(int(round(0.7 / dw)), 1)        # >=0.7 A apart
            pk, props = find_peaks(yc, prominence=0.08 * amax_data, distance=min_dist)
            if len(pk) >= 2:
                top2 = pk[np.argsort(props["prominences"])[::-1][:2]]
                top2 = sorted(top2)                        # blue (lower lambda) first
                mu_b0, a_b0 = float(xg[top2[0]]), max(float(yc[top2[0]]), 1e-3)
                mu_r0, a_r0 = float(xg[top2[1]]), max(float(yc[top2[1]]), 1e-3)
        except Exception:
            pass

        # (ii) fallback: tallest point on each side of systemic
        if mu_b0 is None:
            bmask = xg < LYA_REST
            def _peak(mask, default_mu):
                if mask.sum() == 0:
                    return default_mu, amax_data
                j = int(np.nanargmax(yc[mask]))
                return float(xg[mask][j]), max(float(yc[mask][j]), 1e-3)
            mu_b0, a_b0 = _peak(bmask, LYA_REST - 1.5)
            mu_r0, a_r0 = _peak(~bmask, LYA_REST + 1.5)

        sep0 = float(np.clip(mu_r0 - mu_b0, 2 * dw, 2 * bounds_window - 1e-3))
        sg0 = float(np.clip(sep0 / 3.0, 0.2, sigma_hi))
        amax0 = max(a_b0, a_r0, 1e-3)

        seeds = [
            _b(a_b0, mu_b0, sg0, a_r0, sep0, sg0),                          # find_peaks
            _b(a_b0, mu_b0, max(sg0 * 0.6, 0.2), a_r0, sep0, max(sg0 * 0.6, 0.2)),  # narrower
            _b(amax0 * 0.7, LYA_REST - 1.5, 1.0, amax0, 3.0, 1.0),          # generic wide
            _b(amax0 * 0.7, LYA_REST - 2.0, 0.8, amax0, 4.0, 0.8),          # wider still
        ]

        # (iii) USER physical prior, tried FIRST (the lowest-chi^2 seed wins anyway).
        #   peak_guess = {blue_offset, red_offset, flux_ratio, sigma}, offsets in A
        #   from systemic, flux_ratio = blue_amp / red_amp.
        if peak_guess:
            bo = float(peak_guess.get("blue_offset", -1.9))
            ro = float(peak_guess.get("red_offset", 1.0))
            fr = float(peak_guess.get("flux_ratio", 0.45))
            sgG = float(peak_guess.get("sigma", max(sg0, 0.6)))
            mu_rG = LYA_REST + ro
            a_rG = max(float(np.interp(mu_rG, xg, yc)), amax0, 1e-3)
            seeds.insert(0, _b(fr * a_rG, LYA_REST + bo, sgG, a_rG, ro - bo, sgG))

        if seed is not None and seed.get("success"):
            a, mu, sg = seed["amp"], seed["mu"], min(seed["sigma_int_A"], 1.5)
            seeds.append(_b(0.6 * a, mu - 1.0, sg, a, 2.0, sg))

        best = None
        for p0 in seeds:
            p0 = list(np.clip(p0, lo, hi))
            try:
                popt, pcov = curve_fit(m_double, xg, yg, p0=p0, sigma=eg,
                                       absolute_sigma=True, bounds=(lo, hi), maxfev=30000)
            except Exception:
                continue
            chi2 = float(np.nansum(((yg - m_double(xg, *popt)) / eg) ** 2))
            if best is None or chi2 < best[0]:
                best = (chi2, popt, pcov)
        if best is None:
            return {"success": False, "reason": "all double seeds failed", "model_name": "double"}

        _, popt, pcov = best
        chi2, ndof, aic, bic = _metrics(m_double, popt, 6 + nb)
        amp_b, mu_b, sig_b, amp_r, dsep, sig_r = (float(v) for v in popt[:6])
        amax = max(amp_b, amp_r, 1e-30)
        degenerate = (dsep < dw) or (min(amp_b, amp_r) / amax < 0.05)
        full, line, c = _full_double(popt)
        return {"success": True, "reason": None, "model_name": "double",
                "amp_b": amp_b, "mu_b": mu_b, "sig_b_A": sig_b,
                "amp_r": amp_r, "mu_r": mu_b + dsep, "sig_r_A": sig_r,
                "peak_sep_A": dsep, "peak_sep_kms": C_KMS * dsep / LYA_REST,
                "degenerate": bool(degenerate), "baseline": float(c),
                "chi2": chi2, "ndof": ndof, "chi2_red": chi2 / max(ndof, 1),
                "aic": aic, "bic": bic, "n": n, "popt": popt, "pcov": pcov,
                "intrinsic": line, "model": full}

    if model == "single":
        res = _fit_single()
        if verbose and res["success"]:
            print(f"[single] sigma_int={res['sigma_int_A']:.3f} A "
                  f"({res['sigma_int_kms']:.0f} km/s)  chi2/ndof={res['chi2_red']:.2f}")
        return res
    if model == "double":
        res = _fit_double()
        if verbose and res["success"]:
            print(f"[double] sep={res['peak_sep_A']:.2f} A "
                  f"({res['peak_sep_kms']:.0f} km/s)  chi2/ndof={res['chi2_red']:.2f}"
                  f"{'  [DEGENERATE -> effectively single]' if res['degenerate'] else ''}")
        return res
    if model != "both":
        raise ValueError("model must be 'single', 'double', or 'both'")

    # ---- both: fit and auto-select ----
    single = _fit_single()
    if not single["success"]:
        return {"selected": None, "single": single, "double": None,
                "comparison": {"reason": "single fit failed: " + str(single["reason"])}}
    double = _fit_double(seed=single)

    def _pick_single(reason):
        return {"selected": "single", "single": single, "double": double,
                "comparison": {"criterion": select, "reason": reason}}

    if not double["success"]:
        if verbose:
            print(f"[both] double did not converge -> single. ({double['reason']})")
        return _pick_single("double fit failed; fell back to single")
    if double["degenerate"]:
        if verbose:
            print("[both] double degenerate (peaks merged) -> single.")
        return _pick_single("double degenerate (peaks merged / one component vanished)")

    delta_chi2 = single["chi2"] - double["chi2"]
    delta_dof = single["ndof"] - double["ndof"]   # = 3
    f_stat = (delta_chi2 / max(delta_dof, 1)) / (double["chi2"] / max(double["ndof"], 1)) \
        if double["chi2"] > 0 else np.inf
    f_pvalue = float(_fdist.sf(f_stat, delta_dof, double["ndof"])) if np.isfinite(f_stat) else 0.0
    delta_aic = single["aic"] - double["aic"]
    delta_bic = single["bic"] - double["bic"]

    if delta_chi2 <= 0:
        sel, reason = "single", "double did not reduce chi^2"
    elif select == "ftest":
        sel = "double" if f_pvalue < alpha else "single"
        reason = f"F-test p={f_pvalue:.3g} {'<' if sel=='double' else '>='} alpha={alpha}"
    elif select == "bic":
        sel = "double" if delta_bic > delta_bic_thresh else "single"
        reason = f"delta_BIC={delta_bic:.2f} {'>' if sel=='double' else '<='} {delta_bic_thresh}"
    elif select == "aic":
        sel = "double" if delta_aic > delta_aic_thresh else "single"
        reason = f"delta_AIC={delta_aic:.2f} {'>' if sel=='double' else '<='} {delta_aic_thresh}"
    else:
        raise ValueError("select must be 'ftest', 'bic', or 'aic'")

    comparison = {"criterion": select, "selected": sel, "reason": reason,
                  "delta_chi2": float(delta_chi2), "delta_dof": int(delta_dof),
                  "f_stat": float(f_stat), "f_pvalue": f_pvalue,
                  "aic_single": single["aic"], "aic_double": double["aic"],
                  "delta_aic": float(delta_aic),
                  "bic_single": single["bic"], "bic_double": double["bic"],
                  "delta_bic": float(delta_bic)}
    if verbose:
        print(f"[both] chi2/ndof single={single['chi2_red']:.2f} double={double['chi2_red']:.2f} "
              f"| F={f_stat:.2f} p={f_pvalue:.3g} dAIC={delta_aic:.1f} dBIC={delta_bic:.1f} "
              f"-> SELECTED: {sel} ({reason})")
    return {"selected": sel, "single": single, "double": double, "comparison": comparison}


# =====================================================================
# 4. GALAXY-RESAMPLING BOOTSTRAP of the intrinsic-profile fit
# =====================================================================

def bootstrap_intrinsic_profile(stacks, lsf_result, radial_bin, stack_err,
                                config=None, model="both", nboot=300,
                                stack_method=None, bounds_window=4.0,
                                baseline=True, select="ftest", peak_guess=None,
                                seed=1, progress=True, verbose=True):
    """
    Galaxy-resampling bootstrap of the intrinsic-profile fit for ONE radial bin,
    giving confidence intervals on sigma_int (single) / peak separation (double),
    and -- for model='both' -- the FRACTION of resamples that prefer a double.

    How it mirrors the rest of the pipeline: resample the galaxy axis of the
    Stage-2 cube WITH REPLACEMENT, re-stack that bin's line with the same combine
    method bootstrap_measurements uses, then re-run deconvolve_intrinsic_profile.
    Two things are held FIXED across draws, deliberately:
      * the per-pixel stack_err (chi^2 weights; standard bootstrap practice), and
      * the stacked LSF (lsf_result). Its sample-to-sample variance is second
        order; rebuild it per draw only if you need that last bit of rigor.

    The headline number for model='both' is double_fraction: how often the data
    support two peaks under resampling. A clean detection is double_fraction near
    1; a marginal bin where single vs double flips is exactly the "consistent
    with single, double not required" regime -- the fraction quantifies it.

    Parameters
    ----------
    stacks     : Stage-2 dict with cube_flux/cube_err (keep_cube=True)
    lsf_result : output of stack_lsf (held fixed)
    radial_bin : which bin to bootstrap
    stack_err  : per-pixel error for this bin (e.g. boot['stack_error']['stack_err'][bin])
    config     : optional PipelineConfig (for the combine method); else 'biweight'
    model      : 'single' | 'double' | 'both' (forwarded to deconvolve)
    nboot      : number of galaxy resamples

    Returns
    -------
    dict:
      n_ok, nboot, double_fraction,
      sigma_int_A : {med, lo, hi, all}   (from draws fit as single)
      peak_sep_A  : {med, lo, hi, all}   (from draws fit as double)
      meta
    """
    from .measure import stack_galaxies

    cube = np.asarray(stacks["cube_flux"], float)         # (ngal, nrad, nwave)
    cube_e = stacks.get("cube_err")
    cube_e = np.asarray(cube_e, float) if cube_e is not None else None
    wave = np.asarray(stacks["rest_wave"], float)
    err = np.asarray(stack_err, float)
    ngal = cube.shape[0]
    sm = stack_method or (getattr(config, "measure_stack_method", "biweight")
                          if config is not None else "biweight")
    rng = np.random.default_rng(seed)

    sig = np.full(nboot, np.nan)
    sep = np.full(nboot, np.nan)
    chose_double = np.zeros(nboot, dtype=bool)
    ok = np.zeros(nboot, dtype=bool)

    try:
        from tqdm.auto import tqdm
        _iter = tqdm(range(nboot), desc=f"bootstrap bin {radial_bin}",
                     disable=not progress, leave=False)
    except Exception:
        _iter = range(nboot)

    for i in _iter:
        idx = rng.integers(0, ngal, ngal)
        sub = cube[idx][:, radial_bin, :][:, None, :]
        sub_e = (cube_e[idx][:, radial_bin, :][:, None, :]
                 if cube_e is not None else None)
        line, _ = stack_galaxies(sub, sub_e, method=sm)
        res = deconvolve_intrinsic_profile(wave, line[0], err, lsf_result,
                                           model=model, bounds_window=bounds_window,
                                           baseline=baseline, select=select,
                                           peak_guess=peak_guess, verbose=False)
        # normalize the (single | double | both) return into one fit
        if isinstance(res, dict) and "selected" in res:
            sel = res["selected"]
            fit = res.get(sel) if sel else None
        else:
            fit = res
            sel = res.get("model_name") if isinstance(res, dict) else None
        if not (fit and fit.get("success")):
            continue
        ok[i] = True
        if sel == "double":
            chose_double[i] = True
            sep[i] = fit.get("peak_sep_A", np.nan)
        else:
            sig[i] = fit.get("sigma_int_A", np.nan)

    def _pct(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"med": np.nan, "lo": np.nan, "hi": np.nan, "all": a}
        return {"med": float(np.nanpercentile(a, 50)),
                "lo": float(np.nanpercentile(a, 16)),
                "hi": float(np.nanpercentile(a, 84)), "all": a}

    n_ok = int(ok.sum())
    dbl_frac = float(chose_double[ok].mean()) if n_ok else np.nan
    out = {"n_ok": n_ok, "nboot": nboot, "double_fraction": dbl_frac,
           "sigma_int_A": _pct(sig), "peak_sep_A": _pct(sep),
           "meta": {"radial_bin": radial_bin, "model": model, "select": select,
                    "stack_method": sm, "seed": seed}}

    if verbose:
        s = out["sigma_int_A"]; p = out["peak_sep_A"]
        print(f"bin {radial_bin}: {n_ok}/{nboot} draws ok; "
              f"double chosen {dbl_frac*100:.0f}%")
        if np.isfinite(s["med"]):
            print(f"  sigma_int = {s['med']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}] A "
                  f"({C_KMS*s['med']/LYA_REST:.0f} km/s)  [from single-fit draws]")
        if np.isfinite(p["med"]):
            print(f"  peak_sep  = {p['med']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}] A "
                  f"({C_KMS*p['med']/LYA_REST:.0f} km/s)  [from double-fit draws]")
    return out
