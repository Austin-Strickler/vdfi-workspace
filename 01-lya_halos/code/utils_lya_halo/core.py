"""
core.py -- per-galaxy CORE Lya extraction + measurement + escape fraction.

This is a SEPARATE sub-pipeline from the binned halo stack (pipeline.py). That
pipeline coadds annular bins ACROSS galaxies and measures a stacked CENTROID;
this module extracts ONE optimally-weighted core spectrum PER GALAXY and
measures that galaxy's integrated Lya FLUX, enabling per-object classification
(emitter / absorber / null) and a Lya escape fraction. The C1 extraction wraps
the PSF-weighted optics.py functions that the binned path explicitly deferred.

----------------------------------------------------------------------
THE OVERALL RUN PATTERN (what "simply run this" looks like)
----------------------------------------------------------------------

    1.  extract_core_spectra(config, ...)
            -> Spec (ngal, nwave), Err (ngal, nwave)   [observed frame]
            -> wave_obs (nwave,), catalog (astropy Table)

    2.  run_core(Spec, Err, wave_obs, catalog, ...)
            Per galaxy, in a simple loop:
              a. to_rest_frame(wave_obs, Spec[i], Err[i], z=z[i])
              b. measure_line(...)          -> flux, flux_err, sn, label
              c. bootstrap_core_flux(...)   -> flux_boot_lo/hi (optional)
              d. lya_escape_fraction(...)   -> fesc, fesc_err
            Pack into a results dict with per-galaxy numpy arrays (all ngal-long).

    3.  write_core_fits(path, Spec, Err, wave_obs, catalog, results)
            One FITS, everything inside -- spectra, errors, measurements,
            bootstrap intervals, escape fractions, catalog, provenance.

That is it. ~500 galaxies, one spectrum each: the extraction runs in minutes
(no 19 GB full radial grid), no per-galaxy cache files, no Stage-2 stacking.

----------------------------------------------------------------------
STAGES
----------------------------------------------------------------------
C1  extract_core_spectra  [SOCKET -- wires optics.py; heavy, run once]
    PSF-weighted optimal extraction per galaxy per exposure; inverse-variance
    combine exposures -> (ngal, nwave) observed-frame core spectra.

C2  run_core              [implemented; fast, re-run freely]
    rest-frame shift + one of three line estimators + classification + bootstrap.

C3  escape fraction       [implemented]
    lya_escape_fraction: the existing Case-B / Hbeta anchor formula.

----------------------------------------------------------------------
ESTIMATORS (measure_line)
----------------------------------------------------------------------
  integration    -- model-free sum over an asymmetric window (+0.5 A red);
                    routes through measure.integrated_line_flux so the continuum
                    and summation convention match the rest of the pipeline.
  gaussian       -- Gaussian + constant fit with explicit failure flags (no
                    convergence, railed width/center, non-finite covariance).
  matched_filter -- OPTIMAL (max-likelihood, Gaussian noise):
                    F = sum(y g/var) / sum(g^2/var),  sigma_F = 1/sqrt(sum(g^2/var))
                    at a FIXED template center -> no peak-chasing bias, minimal
                    scatter, the right choice at low S/N. (see injection_recovery)

----------------------------------------------------------------------
VALIDATION -- injection-recovery and completeness
----------------------------------------------------------------------
Two injection modes, addressing different questions:

  injection_recovery(wave, err, ...)          -- METHOD CHARACTERIZATION
    Injects into pure synthetic noise (flat or shaped, but no real sky).
    Isolates the estimator behavior from confounding real-data effects.
    Include input_flux=0 to measure the upward bias from peak-finding:
    matched_filter should give ~0, free-center Gaussian biases positive.

  injection_recovery_sample(core_err, wave_obs, catalog, ...)  -- REALISM
    Uses each galaxy's ACTUAL error spectrum as the noise model; injects
    at input S/N = k * sigma_F (the per-galaxy matched-filter noise floor),
    so completeness is expressed as a function of true input S/N rather than
    absolute flux. This accounts for z-dependent sky noise, wavelength coverage
    gaps, and fiber-position effects. It is the test the paper needs.

  null_redshift_test(core_spec, core_err, wave_obs, z, ...)    -- NULL
    Permutes redshifts across galaxies and measures "Lya" at the wrong z.
    The recovered flux distribution should be centered at 0; any systematic
    offset signals a continuum artifact at the expected line position.

  completeness_snr_curve(injection_result)
    Pulls detection-rate-vs-input-SNR from either injection function.

----------------------------------------------------------------------
All C2/C3 functions take plain arrays -- no PipelineConfig needed -- so they
drop straight into a notebook. LYA_REST == PipelineConfig.LYA_REST == 1215.67.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .measure import (
    get_continuum_model, integrated_line_flux, DEFAULT_CONT_BOUNDS,
)

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    def _tqdm(x, **kw): return x

if TYPE_CHECKING:
    from .config import PipelineConfig

C_KMS    = 299792.458
LYA_REST = 1215.67


# =====================================================================
# 0.  FRAME + WINDOW HELPERS
# =====================================================================

def to_rest_frame(wave_obs, flux, err=None, z=None, rest_density=True):
    """
    Shift observed-frame core spectrum to rest frame.

    wave_rest = wave_obs / (1+z). With rest_density=True the flux density is
    multiplied by (1+z) so the rest-frame integral preserves the physical line
    flux -- same convention as Stage 2 (config.rest_density).
    Returns (wave_rest, flux_rest, err_rest).
    """
    if z is None:
        raise ValueError("to_rest_frame needs z.")
    wave_obs = np.asarray(wave_obs, dtype=float)
    flux     = np.asarray(flux,     dtype=float)
    opz      = 1.0 + float(z)
    scale    = opz if rest_density else 1.0
    return (
        wave_obs / opz,
        flux * scale,
        np.asarray(err, dtype=float) * scale if err is not None else None,
    )


def lya_window(lya_center=LYA_REST, hw=4.0, red_offset=0.5):
    """
    Lya line window (lo, hi) in rest-A, symmetric about the expected line peak.

    The peak sits at (lya_center + red_offset) -- resonant-scattered Lya is
    redshifted off systemic -- and the window spans +/-hw around THAT center:
        (lya_center + red_offset - hw,  lya_center + red_offset + hw)
    Set red_offset=0 for a window centered on systemic Lya.
    """
    center = lya_center + red_offset
    return (center - hw, center + hw)


def _sn(flux, flux_err):
    """Signed S/N, NaN-safe."""
    if flux_err is None or not np.isfinite(flux_err) or flux_err <= 0:
        return np.nan
    return float(flux) / float(flux_err)


# =====================================================================
# 1.  PER-SPECTRUM ESTIMATORS
#     Common return schema: method, success, reason, flux, flux_err, sn, center_A
# =====================================================================

def measure_integration(
    wave, flux, err=None,
    lya_center=LYA_REST, hw=4.0, red_offset=0.5,
    cont_bounds=DEFAULT_CONT_BOUNDS, cont_method="poly", cont_order=1,
):
    """
    METHOD 1 -- model-free continuum-subtracted integration over the Lya window.

    Delegates to measure.integrated_line_flux so the continuum model and
    summation convention are identical to the stack pipeline (one integrator
    everywhere). The window is lya_window(...): symmetric (+/-hw) about the
    expected peak (lya_center + red_offset).
    """
    bounds = lya_window(lya_center, hw, red_offset)
    res = integrated_line_flux(
        wave, flux, err=err, bounds=bounds, cont_bounds=cont_bounds,
        lya_center=lya_center, cont_method=cont_method, cont_order=cont_order,
    )
    F  = res.get("flux_integral", np.nan)
    eF = res.get("err_integral",  np.nan)
    return {
        "method": "integration", "success": bool(res.get("success", False)),
        "reason": res.get("reason"),
        "flux": F, "flux_err": eF, "sn": _sn(F, eF),
        "center_A": lya_center, "bounds": bounds, "npix": res.get("npix"),
        "continuum": res.get("continuum"), "flux_contsub": res.get("flux_contsub"),
    }


def _gauss_const(w, amp, mu, sigma, c):
    return amp * np.exp(-0.5 * ((w - mu) / sigma) ** 2) + c


def measure_gaussian(
    wave, flux, err=None,
    lya_center=LYA_REST, init_sigma=1.5, red_offset=0.5, fit_hw=6.0,
    sigma_bounds=(0.5, 4.0), center_tol=None,
    cont_bounds=DEFAULT_CONT_BOUNDS, cont_method="poly", cont_order=1,
    min_pixels=8,
):
    """
    METHOD 2 -- Gaussian fit with explicit failure detection.

    Fits y = A*exp(-0.5((w-mu)/s)^2) + c on the continuum-subtracted line.
    Flux = A*s*sqrt(2pi); error propagated from the (A, s) covariance block.
    A < 0 produces a negative flux, feeding directly into classify_lya.

    success=False (with reason) when: scipy missing, too few pixels, fit fails
    to converge, non-finite covariance, width railed on sigma_bounds, or center
    ran past center_tol -- every failure mode the fit can silently hide.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    c0   = lya_center + red_offset
    ctol = fit_hw if center_tol is None else center_tol
    fail = {"method": "gaussian", "success": False, "flux": np.nan,
            "flux_err": np.nan, "sn": np.nan, "center_A": np.nan, "fit": None}

    if curve_fit is None:
        return {**fail, "reason": "scipy not available"}

    cont = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                               cont_method=cont_method, cont_order=cont_order)
    y = flux - cont
    m = (np.abs(wave - c0) <= fit_hw) & np.isfinite(y) & np.isfinite(wave)
    sig = None
    if err is not None:
        err = np.asarray(err, dtype=float)
        m  &= np.isfinite(err) & (err > 0)
        sig = err[m]
    if m.sum() < min_pixels:
        return {**fail, "reason": "Not enough finite pixels", "continuum": cont}

    xw, yw = wave[m], y[m]
    a0 = float(np.nanmax(np.abs(yw)) * np.sign(np.nansum(yw))) or 1.0
    p0 = [a0, c0, init_sigma, 0.0]
    lo = [-np.inf, c0 - ctol, sigma_bounds[0], -np.inf]
    hi = [ np.inf, c0 + ctol, sigma_bounds[1],  np.inf]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(_gauss_const, xw, yw, p0=p0, bounds=(lo, hi),
                                   sigma=sig, absolute_sigma=(sig is not None),
                                   maxfev=5000)
    except Exception as exc:
        return {**fail, "reason": f"fit failed: {type(exc).__name__}", "continuum": cont}

    A, mu, s, c_ = popt
    if not np.all(np.isfinite(pcov)):
        return {**fail, "reason": "non-finite covariance", "continuum": cont}
    if not (sigma_bounds[0]*1.01 < s < sigma_bounds[1]*0.99):
        return {**fail, "reason": "width railed on sigma_bounds", "continuum": cont,
                "fit": dict(A=A, mu=mu, sigma=s, c=c_)}
    if abs(mu - c0) >= ctol * 0.99:
        return {**fail, "reason": "center railed on center_tol", "continuum": cont,
                "fit": dict(A=A, mu=mu, sigma=s, c=c_)}

    sq2pi = np.sqrt(2 * np.pi)
    F    = A * s * sq2pi
    dFdA = s * sq2pi;  dFds = A * sq2pi
    vF   = dFdA**2 * pcov[0,0] + dFds**2 * pcov[2,2] + 2*dFdA*dFds*pcov[0,2]
    eF   = float(np.sqrt(vF)) if (np.isfinite(vF) and vF >= 0) else np.nan
    mu_err = float(np.sqrt(pcov[1,1])) if (np.isfinite(pcov[1,1]) and pcov[1,1] >= 0) else np.nan
    s_err  = float(np.sqrt(pcov[2,2])) if (np.isfinite(pcov[2,2]) and pcov[2,2] >= 0) else np.nan
    return {
        "method": "gaussian", "success": True, "reason": None,
        "flux": float(F), "flux_err": eF, "sn": _sn(F, eF),
        "center_A": float(mu),
        "fit": dict(A=float(A), mu=float(mu), sigma=float(s), c=float(c_),
                    mu_err=mu_err, sigma_err=s_err),
        "continuum": cont, "flux_contsub": y,
    }


def measure_matched_filter(
    wave, flux, err,
    lya_center=LYA_REST, template_sigma=1.5, red_offset=0.5, fit_hw=6.0,
    cont_bounds=DEFAULT_CONT_BOUNDS, cont_method="poly", cont_order=1,
    min_pixels=5,
):
    """
    METHOD 3 -- optimal matched-filter flux for a fixed-center template.

    Solves the WLS problem y_i = F * g_i for the unit-area Gaussian g at a
    FIXED center (lya_center + red_offset), returning the max-likelihood flux:

        F      = sum(y g / sigma^2) / sum(g^2 / sigma^2)
        sigma_F = 1 / sqrt(sum(g^2 / sigma^2))

    Because the template center is fixed (not fitted), the estimator is LINEAR
    in the data and has zero peak-chasing bias on pure noise (injection_recovery
    at F_in=0 returns ~0, unlike the free-center Gaussian). It is the natural
    companion to the PSF optimal extraction in C1 -- both are inverse-variance
    weighted by a known profile shape. Requires per-pixel errors.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if err is None:
        return {"method": "matched_filter", "success": False,
                "reason": "requires a per-pixel error array",
                "flux": np.nan, "flux_err": np.nan, "sn": np.nan, "center_A": np.nan}
    err    = np.asarray(err, dtype=float)
    center = lya_center + red_offset
    cont   = get_continuum_model(wave, flux, cont_bounds=cont_bounds,
                                 cont_method=cont_method, cont_order=cont_order)
    y   = flux - cont
    var = err ** 2
    g   = (np.exp(-0.5 * ((wave - center) / template_sigma) ** 2)
           / (template_sigma * np.sqrt(2 * np.pi)))
    m   = (np.abs(wave - center) <= fit_hw) & np.isfinite(y) & np.isfinite(var) & (var > 0)
    if m.sum() < min_pixels:
        return {"method": "matched_filter", "success": False,
                "reason": "Not enough finite pixels",
                "flux": np.nan, "flux_err": np.nan, "sn": np.nan,
                "center_A": center, "continuum": cont}
    gm, ym, vm = g[m], y[m], var[m]
    denom = float(np.sum(gm*gm / vm))
    if not np.isfinite(denom) or denom <= 0:
        return {"method": "matched_filter", "success": False,
                "reason": "degenerate weights",
                "flux": np.nan, "flux_err": np.nan, "sn": np.nan,
                "center_A": center, "continuum": cont}
    F  = float(np.sum(ym * gm / vm)) / denom
    eF = 1.0 / np.sqrt(denom)
    return {
        "method": "matched_filter", "success": True, "reason": None,
        "flux": F, "flux_err": eF, "sn": F / eF,
        "center_A": center, "template_sigma": template_sigma,
        "continuum": cont, "flux_contsub": y,
    }


_ESTIMATORS = {
    "integration":    measure_integration,
    "gaussian":       measure_gaussian,
    "matched_filter": measure_matched_filter,
}


def measure_line(wave, flux, err=None, method="matched_filter", **kw):
    """Dispatch to one estimator by name."""
    try:
        fn = _ESTIMATORS[method]
    except KeyError:
        raise ValueError(f"method must be one of {list(_ESTIMATORS)}; got {method!r}")
    return fn(wave, flux, err, **kw) if method == "matched_filter" else fn(wave, flux, err=err, **kw)


def _label_from_sn(sn, success=True, reason=None,
                   n_sigma_emit=2.0, n_sigma_absorb=2.0):
    """
    Core emitter / absorber / null decision from a single SIGNED S/N.

    The emission and absorption thresholds move INDEPENDENTLY:
        sn >= +n_sigma_emit    -> 'emitter'
        sn <= -n_sigma_absorb  -> 'absorber'
        otherwise              -> 'null'
    So you can, e.g., keep absorbers at a conservative 2 sigma while pushing the
    emitter cut down to 1 sigma to match LAE-sample definitions in the literature
    (Du+2011, Trainor+2025) -- or vice versa. A failed / non-finite-S/N
    measurement is 'null' with the reason passed through.
    """
    if not success or not np.isfinite(sn):
        return {"label": "null", "sn": sn, "reason": reason or "non-finite S/N"}
    label = ("emitter"  if sn >=  n_sigma_emit   else
             "absorber" if sn <= -n_sigma_absorb else
             "null")
    return {"label": label, "sn": float(sn), "reason": None}


def classify_lya(result, n_sigma=None, *, n_sigma_emit=None, n_sigma_absorb=None):
    """
    Emitter / absorber / null from a measure_* result dict (reads result['sn']).

    Thresholds (independent):
        sn >= +n_sigma_emit    -> 'emitter'
        sn <= -n_sigma_absorb  -> 'absorber'
        otherwise              -> 'null'

    Back-compatible: pass a single ``n_sigma`` to use one symmetric threshold for
    both sides (the old behavior). ``n_sigma_emit`` / ``n_sigma_absorb`` override
    each side individually; anything left unset falls back to ``n_sigma`` and then
    to 2.0. Failed / NaN-S/N measurements are 'null' with the reason passed through.
    """
    base   = 2.0 if n_sigma is None else float(n_sigma)
    n_emit = base if n_sigma_emit   is None else float(n_sigma_emit)
    n_abs  = base if n_sigma_absorb is None else float(n_sigma_absorb)
    return _label_from_sn(
        result.get("sn", np.nan),
        success=bool(result.get("success", False)),
        reason=result.get("reason"),
        n_sigma_emit=n_emit, n_sigma_absorb=n_abs,
    )


# =====================================================================
# 2.  ESCAPE FRACTION  (the existing simple design, unchanged)
# =====================================================================

def lya_escape_fraction(Flya, e_Flya, Hb, e_Hb, ebv, e_ebv,
                        k_Hb=4.6, lya_hb_ratio=24.9):
    """
    Lya escape fraction anchored on dust-corrected Hbeta (Case B).

        log f_esc = log10(F_Lya) - log10(F_Hb) - 0.4*k_Hb*E(B-V) - log10(R_caseB)

    F_Lya and F_Hb must be in the same flux units; the MOSDEF HB_FLUX column is
    stored pre-multiplied by 1e17, so set lya_scale accordingly in run_core.
    Errors propagated in log space. Unchanged from the Escape-Fraction notebook.
    Returns (f_esc, f_esc_err) as float arrays.
    """
    Flya = np.asarray(Flya, dtype=float)
    Hb   = np.asarray(Hb,   dtype=float)
    ln10 = np.log(10)
    log_flya     = np.log10(Flya)
    log_hb       = np.log10(Hb)
    sig_log_flya = (np.asarray(e_Flya, dtype=float) / Flya) / ln10
    sig_log_hb   = (np.asarray(e_Hb,   dtype=float) / Hb)   / ln10
    dust         = 0.4 * k_Hb * np.asarray(ebv,   dtype=float)
    sig_dust     = 0.4 * k_Hb * np.asarray(e_ebv, dtype=float)
    log_fesc     = log_flya - log_hb - dust - np.log10(lya_hb_ratio)
    sig_log_fesc = np.sqrt(sig_log_flya**2 + sig_log_hb**2 + sig_dust**2)
    fesc         = 10 ** log_fesc
    fesc_err     = fesc * ln10 * sig_log_fesc
    return fesc, fesc_err


def lya_ew_rest(line_flux_cgs, mag_ab, z, lya_center=LYA_REST,
                mag_min=10.0, mag_max=40.0):
    """
    Rest-frame Lya equivalent width (Angstrom) from an integrated line flux and a
    broadband continuum AB magnitude (the g- or r-band counterpart).

        f_nu = 10^(-0.4 * (mag_ab + 48.6))            [erg/s/cm^2/Hz]
        EW_0 = F_line * lya_center^2 * (1 + z) / (f_nu * c)

    Derivation: EW_obs = F_line / f_lambda,cont(obs) with f_lambda = f_nu c/lam^2
    evaluated at lam_obs = lya_center*(1+z); EW_0 = EW_obs/(1+z) collapses to the
    line above. F_line must be in CGS erg/s/cm^2 (i.e. core flux already scaled by
    flux_to_cgs). A flat-f_nu continuum is assumed across the band->Lya gap, the
    standard first-order approximation; pick a band that sits REDWARD of observed
    Lya so the forest/break don't bias the continuum. Positive EW = emission.

    Magnitudes outside [mag_min, mag_max] are treated as "no continuum" (-> NaN
    EW) so catalog sentinels (e.g. +99 / -99 non-detections) cannot manufacture a
    spurious EW. Returns NaN where any input is non-finite. Vectorized.
    """
    F = np.asarray(line_flux_cgs, float)
    m = np.asarray(mag_ab, float).copy()
    z = np.asarray(z, float)
    m[~((m > mag_min) & (m < mag_max))] = np.nan      # reject sentinel/unphysical mags
    c_A = 2.99792458e18                       # speed of light, Angstrom / s
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        f_nu = 10.0 ** (-0.4 * (m + 48.6))    # erg/s/cm^2/Hz
        ew = F * (lya_center ** 2) * (1.0 + z) / (f_nu * c_A)
    ew = np.asarray(ew, float)
    ew[~np.isfinite(ew)] = np.nan
    return ew


# =====================================================================
# 3b.  HIERARCHICAL PER-GALAXY MEASUREMENT
#      matched filter (detection statistic, everyone) -> if S/N >= detect_snr,
#      fit a Gaussian for the POSITION (v_offset) and the +/-2sigma flux (A).
#      Two integrated fluxes are produced so you can compare them yourself:
#        flux_mf  -- matched-filter optimal flux                       (everyone)
#        flux_B   -- integral over the fixed Lya window, symmetric +/-hw
#                    about (lya_center + red_offset)                   (everyone)
#        flux_A   -- integral over the Gaussian's mu +/- 2sigma, scaled up by
#                    1/0.9545 to recover the ~5% truncated wings (DETECTIONS only)
#      We never fit a Gaussian to a non-detection, so flux_A / mu / sigma /
#      v_offset are NaN below threshold -- no fluxes reported that weren't measured.
# =====================================================================

# fraction of a Gaussian's area within +/- 2 sigma (the aperture correction for A)
_TWO_SIGMA_FRAC = 0.9544997361036416


def _integrate_bounds(wave, flux, err, bounds,
                      cont_bounds=DEFAULT_CONT_BOUNDS, cont_method="poly",
                      cont_order=1, lya_center=LYA_REST):
    """Continuum-subtracted integrated flux over `bounds`, via the canonical
    integrator (so every integrated flux shares one continuum + summation
    convention). Returns (flux_integral, err_integral, npix)."""
    res = integrated_line_flux(
        wave, flux, err=err, bounds=bounds, cont_bounds=cont_bounds,
        lya_center=lya_center, cont_method=cont_method, cont_order=cont_order,
    )
    return (res.get("flux_integral", np.nan),
            res.get("err_integral", np.nan),
            res.get("npix", 0))


def measure_core_hierarchical(
    wave, flux, err,
    detect_snr=3.0,
    lya_center=LYA_REST, red_offset=0.0,
    template_sigma=1.5,                       # matched-filter template width
    hw=4.0,                                   # half-width of the fixed B window
    n_sigma_emit=2.0, n_sigma_absorb=2.0,     # independent emitter / absorber cuts
    class_stat="B",                           # which signed S/N labels: 'B' or 'mf'
    n_sigma_class=None,                        # deprecated alias: sets both cuts
    cont_bounds=DEFAULT_CONT_BOUNDS, cont_method="poly", cont_order=1,
    gaussian_kwargs=None,
):
    """
    Two-tier measurement of ONE rest-frame core spectrum.

    TIER 1 (everyone): matched filter -> flux_mf, flux_mf_err, sn_mf.
        Detection if sn_mf >= detect_snr (this is the DETECTION statistic and
        the trigger for the Gaussian tier -- it is NOT the labeller).
    Always-on integrated flux (everyone), independent of detection:
        flux_B over the fixed window lya_window(lya_center, hw, red_offset),
        symmetric (+/-hw) about (lya_center + red_offset). With the default
        red_offset=0 this window is centred on systemic Lya, and its signed S/N
        sn_B = flux_B / flux_B_err is the constant-window-integration statistic.
    TIER 2 (detections only): Gaussian fit -> mu, sigma, v_offset, and
        flux_A = integral(mu +/- 2 sigma) / 0.9545.
        If the fit fails/rails on a detection, the galaxy STAYS a detection with
        its matched-filter flux, but mu/sigma/v_offset/flux_A are NaN and
        gauss_reason records why (a failed fit never downgrades a detection).

    CLASSIFICATION (the `label`) is decoupled from detection. The signed S/N that
    labels emitter / absorber / null is chosen by ``class_stat``:
        'B'  -> sn_B  (constant-window integration; the default, peak-free)
        'mf' -> sn_mf (matched filter; the previous behaviour)
    and the two cuts move independently:
        sn >= +n_sigma_emit    -> emitter
        sn <= -n_sigma_absorb  -> absorber
        otherwise              -> null
    Pass ``n_sigma_class`` to set both cuts at once (deprecated convenience).

    Returns a flat dict (all scalars) with keys:
        detected, sn_mf, flux_mf, flux_mf_err,
        flux_B, flux_B_err, sn_B,
        flux_A, flux_A_err, mu, sigma, v_offset, v_offset_err,
        label, gauss_success, gauss_reason
    """
    gk = dict(gaussian_kwargs or {})
    # resolve the two independent cuts (n_sigma_class is the legacy both-sides knob)
    base    = 2.0 if n_sigma_class is None else float(n_sigma_class)
    n_emit  = base if n_sigma_emit   is None else float(n_sigma_emit)
    n_abs   = base if n_sigma_absorb is None else float(n_sigma_absorb)
    out = dict(
        detected=False, sn_mf=np.nan, flux_mf=np.nan, flux_mf_err=np.nan,
        flux_B=np.nan, flux_B_err=np.nan, sn_B=np.nan,
        flux_A=np.nan, flux_A_err=np.nan, mu=np.nan, sigma=np.nan,
        v_offset=np.nan, v_offset_err=np.nan,
        label="null", gauss_success=False, gauss_reason=None,
    )

    # ---- TIER 1: matched filter (DETECTION statistic) ----
    mf = measure_matched_filter(
        wave, flux, err, lya_center=lya_center, template_sigma=template_sigma,
        red_offset=red_offset, cont_bounds=cont_bounds,
        cont_method=cont_method, cont_order=cont_order,
    )
    out["flux_mf"]     = mf["flux"]
    out["flux_mf_err"] = mf["flux_err"]
    out["sn_mf"]       = mf["sn"]
    detected = bool(mf["success"] and np.isfinite(mf["sn"])
                    and mf["sn"] >= detect_snr)
    out["detected"] = detected

    # ---- always-on integrated flux B (everyone) -> constant-window S/N ----
    wB = lya_window(lya_center, hw=hw, red_offset=red_offset)
    fB, eB, _ = _integrate_bounds(wave, flux, err, wB, cont_bounds=cont_bounds,
                                  cont_method=cont_method, cont_order=cont_order,
                                  lya_center=lya_center)
    out["flux_B"], out["flux_B_err"] = fB, eB
    out["sn_B"] = _sn(fB, eB)

    # ---- CLASSIFICATION (decoupled from detection) ----
    if str(class_stat).lower() == "mf":
        out["label"] = classify_lya(
            mf, n_sigma_emit=n_emit, n_sigma_absorb=n_abs)["label"]
    else:  # 'B' -- constant-window integration (default)
        out["label"] = _label_from_sn(
            out["sn_B"], success=np.isfinite(out["sn_B"]),
            n_sigma_emit=n_emit, n_sigma_absorb=n_abs)["label"]

    # ---- TIER 2: Gaussian for position + flux_A (detections only) ----
    if detected:
        g = measure_gaussian(
            wave, flux, err, lya_center=lya_center, red_offset=red_offset,
            cont_bounds=cont_bounds, cont_method=cont_method,
            cont_order=cont_order, **gk,
        )
        out["gauss_success"] = bool(g["success"])
        out["gauss_reason"]  = g.get("reason")
        if g["success"]:
            mu  = g["fit"]["mu"]
            sig = g["fit"]["sigma"]
            out["mu"], out["sigma"] = mu, sig
            # velocity offset of the line center from systemic Lya
            out["v_offset"] = (mu - lya_center) / lya_center * C_KMS
            # propagate mu error if the fit covariance gave one (else NaN)
            mu_err = g.get("fit", {}).get("mu_err", np.nan)
            out["v_offset_err"] = (mu_err / lya_center * C_KMS
                                   if np.isfinite(mu_err) else np.nan)
            # flux_A: integrate data over mu +/- 2 sigma, correct for the wings
            fA, eA, _ = _integrate_bounds(
                wave, flux, err, (mu - 2*sig, mu + 2*sig),
                cont_bounds=cont_bounds, cont_method=cont_method,
                cont_order=cont_order, lya_center=lya_center,
            )
            out["flux_A"]     = fA / _TWO_SIGMA_FRAC
            out["flux_A_err"] = eA / _TWO_SIGMA_FRAC
    return out


# =====================================================================
# 3.  PER-GALAXY BOOTSTRAP  (parametric, on one rest-frame spectrum)
# =====================================================================

def bootstrap_core_flux(
    wave, flux, err, method="matched_filter",
    n_boot=300, seed=0, **measure_kwargs,
):
    """
    Parametric bootstrap of the flux measurement for ONE rest-frame spectrum.

    Draws n_boot noise realizations from N(flux, err), re-measures each with
    the chosen estimator, and returns the 16/84 flux interval.

    For the matched_filter this is a sanity cross-check -- the analytical
    sigma_F should equal the robust 16/84 width (it usually does, since the
    estimator IS linear). For the Gaussian fit it is the only way to get an
    honest uncertainty when the covariance matrix is unreliable.

    Returns dict: flux_boot_all (n_boot,), flux_boot_med, flux_boot_lo,
                  flux_boot_hi, flux_boot_err (= 0.5*(p84-p16)).
    """
    rng   = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        spec_b = flux + rng.normal(0.0, err)
        res    = measure_line(wave, spec_b, err, method=method, **measure_kwargs)
        draws[b] = res["flux"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = float(np.nanmedian(draws))
        lo  = float(np.nanpercentile(draws, 16))
        hi  = float(np.nanpercentile(draws, 84))
    return {
        "flux_boot_all": draws, "flux_boot_med": med,
        "flux_boot_lo": lo, "flux_boot_hi": hi,
        "flux_boot_err": 0.5 * (hi - lo),
    }


# =====================================================================
# 4.  RUN DRIVER  (C2 + C3 in one pass over all galaxies)
# =====================================================================

def run_core(
    core_spec, core_err, wave_obs, catalog,
    method="matched_filter", n_sigma=2.0, n_bootstrap=200,
    z_col="z",
    hb_col="HB_FLUX", hb_err_col="HB_FLUX_ERR", ebv_col="EBV_NEB_50",
    hb_scale=1e17, lya_scale=None,
    escape=True, rest_density=True,
    measure_kwargs=None, seed=1, verbose=True,
):
    """
    The full C2 + C3 pass: measure, bootstrap, and escape fraction for all galaxies.

    This is the top-level call after C1. It does:
        for each galaxy:
            1. to_rest_frame(wave_obs, core_spec[i], core_err[i], z[i])
            2. measure_line(...)          -> flux, flux_err, sn, classify
            3. bootstrap_core_flux(...)   -> flux_boot_lo/hi   [if n_bootstrap>0]
            4. lya_escape_fraction(...)   -> fesc, fesc_err    [if escape=True]

    Returns a results dict of per-galaxy arrays (all length ngal), ready to pass
    directly to write_core_fits.

    hb_scale : scalar multiplied onto the HB_FLUX catalog column before the
               escape fraction (MOSDEF stores HB_FLUX * 1e17).
    lya_scale : multiply core flux before the escape ratio so both fluxes are in
                the same physical units. None = use as-is.
    """
    core_spec = np.asarray(core_spec, dtype=float)
    core_err  = np.asarray(core_err,  dtype=float)
    wave_obs  = np.asarray(wave_obs,  dtype=float)
    ngal      = core_spec.shape[0]
    z         = np.asarray(catalog[z_col], dtype=float)
    mk        = measure_kwargs or {}

    flux      = np.full(ngal, np.nan)
    flux_err  = np.full(ngal, np.nan)
    sn        = np.full(ngal, np.nan)
    center    = np.full(ngal, np.nan)
    success   = np.zeros(ngal, dtype=bool)
    label     = np.empty(ngal, dtype=object)
    boot_lo   = np.full(ngal, np.nan)
    boot_hi   = np.full(ngal, np.nan)
    boot_err  = np.full(ngal, np.nan)

    it = _tqdm(range(ngal), desc="measure_core") if verbose else range(ngal)
    for i in it:
        if not np.isfinite(z[i]) or z[i] <= 0:
            label[i] = "null"; continue
        wr, fr, er = to_rest_frame(wave_obs, core_spec[i], core_err[i],
                                   z=z[i], rest_density=rest_density)
        res        = measure_line(wr, fr, er, method=method, **mk)
        flux[i]    = res["flux"];    flux_err[i] = res["flux_err"]
        sn[i]      = res["sn"];      center[i]   = res.get("center_A", np.nan)
        success[i] = res["success"]
        label[i]   = classify_lya(res, n_sigma=n_sigma)["label"]
        if n_bootstrap > 0 and er is not None:
            bt         = bootstrap_core_flux(wr, fr, er, method=method,
                                             n_boot=n_bootstrap, seed=seed+i, **mk)
            boot_lo[i] = bt["flux_boot_lo"]
            boot_hi[i] = bt["flux_boot_hi"]
            boot_err[i]= bt["flux_boot_err"]

    out = dict(flux=flux, flux_err=flux_err, sn=sn, center_A=center,
               success=success, label=label,
               flux_boot_lo=boot_lo, flux_boot_hi=boot_hi, flux_boot_err=boot_err,
               method=method, n_sigma=n_sigma, n_bootstrap=n_bootstrap)

    if escape:
        Fl = flux if lya_scale is None else flux * lya_scale
        eF = flux_err if lya_scale is None else flux_err * lya_scale
        Hb  = np.asarray(catalog[hb_col],     dtype=float) * hb_scale
        eHb = np.asarray(catalog[hb_err_col], dtype=float) * hb_scale
        ebv = np.asarray(catalog[ebv_col],     dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fesc, fesc_e = lya_escape_fraction(Fl, eF, Hb, eHb, ebv, np.zeros_like(ebv))
        out["fesc"] = fesc;  out["fesc_err"] = fesc_e

    return out


def run_core_hierarchical(
    core_spec, core_err, wave_obs, catalog,
    detect_snr=3.0,
    lya_center=LYA_REST, red_offset=0.0, template_sigma=1.5, hw=4.0,
    n_sigma_emit=2.0, n_sigma_absorb=2.0, class_stat="mf",
    n_sigma_class=None,
    cont_bounds=DEFAULT_CONT_BOUNDS, cont_method="poly", cont_order=1,
    z_col="z", rest_density=True,
    # ebv_col = NEBULAR reddening: the formula dust-corrects observed Hbeta (a
    # nebular recombination line) to intrinsic, so it needs the gas-phase
    # (Balmer-decrement) reddening, NOT the stellar-continuum EBV_STAR (~0.44x
    # smaller, would under-correct). VERIFIED consistent: EBV_NEB_50 was built
    # with the Calzetti curve giving k(Hb)=4.598, matching k_Hb=4.6 below.
    # (Old catalogs called this "E(B-V)_50".)
    # hb_scale=1e17: catalog PREFERREDFLUX is raw cgs (~1e-17); x1e17 -> order
    # unity for the log math. Verified against the new catalog flux magnitudes.
    hb_col="HB_FLUX", hb_err_col="HB_FLUX_ERR", ebv_col="EBV_NEB_50",
    hb_scale=1e17, lya_scale=None, escape=True,
    flux_for_escape="A",
    # --- Lya EW + LAE classification ---
    # EW_0 from an integrated line flux (ew_flux) and a continuum AB mag. The
    # core line flux is order-unity (same scale as HB_FLUX*1e17), so flux_to_cgs
    # =1e-17 puts it in erg/s/cm^2 for the EW. mag_col='auto' picks the band that
    # sits redward of observed Lya: MAG_G for z<auto_z_switch, else MAG_R.
    ew_threshold=20.0, ew_flux="mf", mag_col="auto",
    flux_to_cgs=1e-17, auto_z_switch=2.3,
    mag_g_col="MAG_G", mag_r_col="MAG_R",
    gaussian_kwargs=None, verbose=True,
):
    """
    HIERARCHICAL C2 + C3 pass (the two-tier ladder).

    For each galaxy:
        1. to_rest_frame(...)
        2. measure_core_hierarchical(...):
             - matched filter for EVERYONE -> flux_mf, sn_mf, detection flag
             - flux_B (fixed window, symmetric +/-hw about lya_center+red_offset)
               for EVERYONE
             - if sn_mf >= detect_snr: Gaussian -> mu, sigma, v_offset, flux_A
        3. escape fraction from the chosen flux column [if escape=True]
        4. rest-frame Lya EW from ew_flux + a continuum mag -> LAE classification

    Detection vs non-detection is the ONLY split (no middle tier): sn_mf >=
    detect_snr is a detection (gets a Gaussian); below it is a non-detection
    (matched-filter flux + B only, no position). Lower detect_snr to pull more
    galaxies into the position sample.

    Classification (the `label` column) is a 4-way scheme, DECOUPLED from the
    detection threshold. The labelling S/N is chosen by ``class_stat``:
        'B'  -> sn_B  (constant-window integration, default; peak-free, the cut
                       the paper quotes against Du+2011 / Trainor+2025)
        'mf' -> sn_mf (matched filter; the previous behaviour)
    with the two cuts moving INDEPENDENTLY:
        absorber : sn <= -n_sigma_absorb
        LAE      : emitter (sn >= +n_sigma_emit) AND rest-frame EW >= ew_threshold
        emitter  : sn >= +n_sigma_emit but EW below threshold (or no continuum)
        null     : everything else / unmeasurable
    So you can push the emitter cut to e.g. n_sigma_emit=1.0 (to shrink 'null'
    toward LAE-style samples that have no null class) while holding absorbers at
    n_sigma_absorb=2.0. ``n_sigma_class`` is a deprecated convenience that sets
    BOTH cuts at once (and is what older FITS files inherit). LAE is the high-EW
    subset of emitters; an emitter with no usable continuum magnitude (NaN EW)
    stays 'emitter' -- we never promote to LAE without a measured continuum.
    Lowering the cut trades a smaller 'null' bin against more noise-driven
    misclassification, so check it against the injection-recovery completeness.

    The continuum model (cont_bounds / cont_method / cont_order) is exposed and
    forwarded to every estimator, recorded in the FITS header for reproducibility.

    flux_for_escape : which flux column feeds the escape fraction -- one of
        'A' (Gaussian +/-2sigma, detections only -> NaN escape for non-detections),
        'B' (fixed window, everyone), 'mf' (matched-filter optimal, everyone).
        Default 'A'.
    ew_flux : which flux column feeds the EW ('A'/'B'/'mf'); 'mf' is defined for
        all emitters so it is the default.
    mag_col : continuum AB-mag column, or 'auto' to pick MAG_G (z<auto_z_switch)
        vs MAG_R, so the band used is redward of observed Lya.

    Returns a results dict of per-galaxy arrays (length ngal):
        detected (bool), sn_mf, flux_mf, flux_mf_err,
        flux_B, flux_B_err, sn_B,
        flux_A, flux_A_err, mu, sigma, v_offset, v_offset_err,
        ew_rest, ew_rest_err,
        label, gauss_success,        # label in {emitter, absorber, LAE, null}
        fesc, fesc_err            [if escape]
        + scalar metadata: detect_snr, red_offset, template_sigma, hw,
          n_sigma_emit, n_sigma_absorb, class_stat, n_sigma_class (legacy mirror),
          rest_density, cont_bounds, cont_method, cont_order,
          flux_for_escape, ew_threshold, ew_flux, mag_col, flux_to_cgs,
          auto_z_switch
    Ready to hand to write_core_hier_fits.
    """
    core_spec = np.asarray(core_spec, dtype=float)
    core_err  = np.asarray(core_err,  dtype=float)
    wave_obs  = np.asarray(wave_obs,  dtype=float)
    ngal      = core_spec.shape[0]
    z         = np.asarray(catalog[z_col], dtype=float)

    # resolve the two independent classification cuts. n_sigma_class is the
    # legacy single-threshold knob (also what old FITS headers carry); an
    # explicit n_sigma_emit / n_sigma_absorb always wins over it.
    _base    = 2.0 if n_sigma_class is None else float(n_sigma_class)
    n_emit   = _base if n_sigma_emit   is None else float(n_sigma_emit)
    n_abs    = _base if n_sigma_absorb is None else float(n_sigma_absorb)

    keys_f = ["sn_mf", "flux_mf", "flux_mf_err", "flux_B", "flux_B_err", "sn_B",
              "flux_A", "flux_A_err",
              "mu", "sigma", "v_offset", "v_offset_err"]
    arr = {k: np.full(ngal, np.nan) for k in keys_f}
    detected = np.zeros(ngal, dtype=bool)
    gauss_ok = np.zeros(ngal, dtype=bool)
    label    = np.empty(ngal, dtype=object)

    it = _tqdm(range(ngal), desc="measure_core_hier") if verbose else range(ngal)
    for i in it:
        if not np.isfinite(z[i]) or z[i] <= 0:
            label[i] = "null"; continue
        wr, fr, er = to_rest_frame(wave_obs, core_spec[i], core_err[i],
                                   z=z[i], rest_density=rest_density)
        r = measure_core_hierarchical(
            wr, fr, er, detect_snr=detect_snr, lya_center=lya_center,
            red_offset=red_offset, template_sigma=template_sigma, hw=hw,
            n_sigma_emit=n_emit, n_sigma_absorb=n_abs, class_stat=class_stat,
            cont_bounds=cont_bounds, cont_method=cont_method, cont_order=cont_order,
            gaussian_kwargs=gaussian_kwargs,
        )
        for k in keys_f:
            arr[k][i] = r[k]
        detected[i] = r["detected"]
        gauss_ok[i] = r["gauss_success"]
        label[i]    = r["label"]

    out = dict(detected=detected, gauss_success=gauss_ok, label=label, **arr)
    out.update(detect_snr=detect_snr, red_offset=red_offset,
               template_sigma=template_sigma, hw=hw,
               n_sigma_emit=n_emit, n_sigma_absorb=n_abs, class_stat=class_stat,
               n_sigma_class=n_emit,            # legacy mirror for old readers
               rest_density=rest_density,
               cont_bounds=cont_bounds, cont_method=cont_method,
               cont_order=cont_order, flux_for_escape=flux_for_escape,
               ew_threshold=ew_threshold, ew_flux=ew_flux, mag_col=mag_col,
               flux_to_cgs=flux_to_cgs, auto_z_switch=auto_z_switch)

    # ---- escape fraction (needs HB_FLUX, HB_FLUX_ERR, EBV_NEB_50) ----
    have_escape = escape and all(c in catalog.colnames
                                 for c in (hb_col, hb_err_col, ebv_col))
    if escape and not have_escape:
        missing = [c for c in (hb_col, hb_err_col, ebv_col) if c not in catalog.colnames]
        if verbose:
            print(f"  [escape skipped: missing {missing}]")
    if have_escape:
        flux_map = {"A": "flux_A", "B": "flux_B", "mf": "flux_mf"}
        err_map  = {"A": "flux_A_err", "B": "flux_B_err", "mf": "flux_mf_err"}
        fkey = flux_map.get(flux_for_escape, "flux_A")
        ekey = err_map.get(flux_for_escape, "flux_A_err")
        Fl = arr[fkey] if lya_scale is None else arr[fkey] * lya_scale
        eF = arr[ekey] if lya_scale is None else arr[ekey] * lya_scale
        Hb  = np.asarray(catalog[hb_col],     dtype=float) * hb_scale
        eHb = np.asarray(catalog[hb_err_col], dtype=float) * hb_scale
        ebv = np.asarray(catalog[ebv_col],     dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fesc, fesc_e = lya_escape_fraction(Fl, eF, Hb, eHb, ebv, np.zeros_like(ebv))
        out["fesc"] = fesc;  out["fesc_err"] = fesc_e
    else:
        out["fesc"] = np.full(ngal, np.nan)
        out["fesc_err"] = np.full(ngal, np.nan)

    # ---- Lya rest-frame EW + LAE classification (needs a continuum mag) ----
    ew_flux_key = {"A": "flux_A", "B": "flux_B", "mf": "flux_mf"}.get(ew_flux, "flux_mf")
    ew_err_key  = {"A": "flux_A_err", "B": "flux_B_err",
                   "mf": "flux_mf_err"}.get(ew_flux, "flux_mf_err")

    if str(mag_col).lower() == "auto":
        need = (mag_g_col, mag_r_col)
    else:
        need = (mag_col,)
    have_ew = all(c in catalog.colnames for c in need)

    if have_ew:
        if str(mag_col).lower() == "auto":
            mg = np.asarray(catalog[mag_g_col], dtype=float)
            mr = np.asarray(catalog[mag_r_col], dtype=float)
            mag = np.where(z < auto_z_switch, mg, mr)
        else:
            mag = np.asarray(catalog[mag_col], dtype=float)
        F_line = arr[ew_flux_key] * flux_to_cgs
        ew = lya_ew_rest(F_line, mag, z, lya_center=lya_center)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(np.asarray(arr[ew_err_key]) / np.asarray(arr[ew_flux_key]))
        ew_err = np.abs(ew) * rel
    else:
        if verbose:
            print(f"  [EW/LAE skipped: missing {[c for c in need if c not in catalog.colnames]}]"
                  f" -> no LAE upgrade; emitters stay 'emitter']")
        ew     = np.full(ngal, np.nan)
        ew_err = np.full(ngal, np.nan)

    out["ew_rest"] = ew
    out["ew_rest_err"] = ew_err

    # upgrade high-EW emitters to LAE only where EW is measured
    label = np.asarray(label, dtype=object)
    is_lae = (label == "emitter") & np.isfinite(ew) & (ew >= ew_threshold)
    label[is_lae] = "LAE"
    out["label"] = label

    if verbose:
        nd = int(detected.sum())
        ng = int(gauss_ok.sum())
        nlae = int((label == "LAE").sum())
        nem  = int((label == "emitter").sum())
        nab  = int((label == "absorber").sum())
        nnull = int((label == "null").sum())
        nan_ew = int(np.isnan(ew).sum())
        print(f"\nrun_core_hierarchical: {nd}/{ngal} detections (S/N_mf >= {detect_snr}); "
              f"{ng} with a usable Gaussian position.")
        print(f"  labelled on sn_{class_stat} with emit >= {n_emit:g}sigma, "
              f"absorb <= -{n_abs:g}sigma (red_offset={red_offset:g}):")
        print(f"  labels: LAE={nlae} (EW_0 >= {ew_threshold:g} A)  emitter={nem}  "
              f"absorber={nab}  null={nnull}   [{nan_ew} galaxies had no continuum EW]")

    return out


# =====================================================================
# 5.  FITS I/O  (write + read the all-in-one product)
# =====================================================================

# Strings that an ascii Table.read turns boolean columns into. A column whose
# every non-empty value is one of these (case-insensitive) is a real bool that
# got stringified -> 'False' is truthy under bool('False'), the classic bug.
_TRUE_STRS  = {"true", "t", "1", "1.0", "yes", "y"}
_FALSE_STRS = {"false", "f", "0", "0.0", "no", "n"}
_BOOL_STRS  = _TRUE_STRS | _FALSE_STRS
# Sentinels that mean "missing" in the source catalogs -> NaN on float coercion.
_NULL_STRS  = {"", "--", "nan", "na", "none", "null", "-999", "-99", "-1.0", "-1"}
# Never floatify these (identifiers / text), even if they look numeric.
_PROTECT_NUMERIC = ("ID", "FIELD", "CATALOG", "LABEL", "NAME", "INDEX")


def _is_str_dtype(col):
    return getattr(col, "dtype", np.dtype("O")).kind in ("U", "S", "O")


def coerce_catalog_dtypes(table, numeric=True, verbose=False):
    """
    Repair a catalog Table whose dtypes were mangled by ``Table.read(...,
    format='ascii')`` -- the usual culprit being boolean columns stored as the
    strings 'True'/'False' (so ``bool('False')`` is True downstream).

    Boolean repair (always on): any string column whose non-empty values are all
    in {'true','false','t','f','1','0','yes','no'} becomes a real bool column.

    Numeric repair (``numeric=True``): any remaining string column whose non-empty
    values all parse as floats becomes float (sentinels -> NaN). Columns whose
    name contains an identifier/text token (ID, FIELD, CATALOG, LABEL, NAME,
    INDEX) are left as strings so IDs and field names are never floatified.

    Returns a NEW Table (the input is not modified). Safe to call repeatedly and
    on already-correct tables (numeric/bool columns are passed through untouched).
    """
    out = table.copy()
    changed = {"bool": [], "float": []}
    for name in out.colnames:
        col = out[name]
        if not _is_str_dtype(col):
            continue
        vals = [("" if v is None else str(v)).strip() for v in np.asarray(col)]
        nonempty = [v for v in vals if v.lower() not in _NULL_STRS]
        if not nonempty:
            continue

        # --- boolean? ---
        if all(v.lower() in _BOOL_STRS for v in nonempty):
            out[name] = np.array([v.strip().lower() in _TRUE_STRS for v in vals],
                                 dtype=bool)
            changed["bool"].append(name)
            continue

        # --- numeric? ---
        if numeric and not any(tok in name.upper() for tok in _PROTECT_NUMERIC):
            def _f(v):
                if v.lower() in _NULL_STRS:
                    return np.nan
                try:
                    return float(v)
                except ValueError:
                    return None
            parsed = [_f(v) for v in vals]
            if all(p is not None for p in parsed):
                out[name] = np.array(parsed, dtype=float)
                changed["float"].append(name)

    if verbose and (changed["bool"] or changed["float"]):
        if changed["bool"]:
            print(f"coerce_catalog_dtypes: -> bool  {changed['bool']}")
        if changed["float"]:
            print(f"coerce_catalog_dtypes: -> float {changed['float']}")
    return out


def read_keep_catalog(path, format="ascii", numeric=True, verbose=True):
    """
    Read an ascii (or other) catalog and immediately repair its dtypes.

    Thin wrapper over ``Table.read(path, format=...)`` + ``coerce_catalog_dtypes``
    so the catalog enters the pipeline with real bools/floats instead of the
    stringified flags ascii round-trips produce. Use this in place of a bare
    ``Table.read(..., format='ascii')`` for the KEEP/CLEAN catalog.
    """
    tab = Table.read(path, format=format)
    return coerce_catalog_dtypes(tab, numeric=numeric, verbose=verbose)


def write_core_fits(
    path,
    core_spec, core_err, wave_obs, catalog,
    results,
    config_meta=None,
    overwrite=False,
):
    """
    Write the complete core product to a single FITS file.

    HDU layout:
        PRIMARY     -- provenance keywords (method, n_sigma, n_bootstrap, DATE, ...)
        SPEC        -- (ngal, nwave) float32 observed-frame core spectra
        ERROR       -- (ngal, nwave) float32 per-pixel errors
        WAVE        -- (nwave,)      float32 observed wavelength grid
        CATALOG     -- BinTableHDU   per-galaxy catalog (RA, DEC, z, masses, Hb, ...)
        RESULTS     -- BinTableHDU   per-galaxy measurements:
                         FLUX, FLUX_ERR, SN, CENTER_A, SUCCESS, LABEL
                         FLUX_BOOT_LO, FLUX_BOOT_HI, FLUX_BOOT_ERR
                         FESC, FESC_ERR  (if present in results)

    config_meta : optional dict of extra header keywords (e.g. template_sigma,
                  red_offset, hw, k_Hb, hb_scale) to record for reproducibility.
    """
    import os
    ngal, nwave = np.asarray(core_spec).shape

    # --- PRIMARY ---
    hdr = fits.Header()
    hdr["NGAL"]    = ngal
    hdr["NWAVE"]   = nwave
    hdr["METHOD"]  = results.get("method", "unknown")
    hdr["NSIGMA"]  = results.get("n_sigma", np.nan)
    hdr["NBOOT"]   = results.get("n_bootstrap", 0)
    hdr["DATE"]    = _dt.datetime.now().isoformat(timespec="seconds")
    if config_meta:
        for k, v in config_meta.items():
            hdr[k[:8].upper()] = v
    primary = fits.PrimaryHDU(header=hdr)

    # --- spectra + wave ---
    spec_hdu  = fits.ImageHDU(np.asarray(core_spec, dtype=np.float32), name="SPEC")
    err_hdu   = fits.ImageHDU(np.asarray(core_err,  dtype=np.float32), name="ERROR")
    wave_hdu  = fits.ImageHDU(np.asarray(wave_obs,  dtype=np.float32), name="WAVE")

    # --- catalog table ---
    # store real bool/float dtypes (not stringified ascii flags) so flags
    # round-trip correctly -- bool('False') == True was the old bug.
    cat_hdu      = fits.table_to_hdu(coerce_catalog_dtypes(Table(catalog)))
    cat_hdu.name = "CATALOG"

    # --- results table ---
    cols = [
        fits.Column("FLUX",          "E", array=results["flux"].astype(np.float32)),
        fits.Column("FLUX_ERR",      "E", array=results["flux_err"].astype(np.float32)),
        fits.Column("SN",            "E", array=results["sn"].astype(np.float32)),
        fits.Column("CENTER_A",      "E", array=results["center_A"].astype(np.float32)),
        fits.Column("SUCCESS",       "L", array=results["success"].astype(bool)),
        fits.Column("LABEL",        "8A", array=np.array(results["label"], dtype="U8")),
        fits.Column("FLUX_BOOT_LO",  "E", array=results["flux_boot_lo"].astype(np.float32)),
        fits.Column("FLUX_BOOT_HI",  "E", array=results["flux_boot_hi"].astype(np.float32)),
        fits.Column("FLUX_BOOT_ERR", "E", array=results["flux_boot_err"].astype(np.float32)),
    ]
    if "fesc" in results:
        cols += [
            fits.Column("FESC",     "E", array=results["fesc"].astype(np.float32)),
            fits.Column("FESC_ERR", "E", array=results["fesc_err"].astype(np.float32)),
        ]
    res_hdu      = fits.BinTableHDU.from_columns(cols)
    res_hdu.name = "RESULTS"

    hdul = fits.HDUList([primary, spec_hdu, err_hdu, wave_hdu, cat_hdu, res_hdu])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    hdul.writeto(path, overwrite=overwrite)
    if True:  # always print path
        print(f"wrote core FITS -> {path}  ({ngal} galaxies)")
    return path


# ---------------------------------------------------------------------
# Measurement-config <-> FITS-header mapping (one source of truth)
# ---------------------------------------------------------------------
# Every scalar run_core_hierarchical knob is stamped on write, restored on read,
# and inherited on re-measure from this single spec. cont_bounds (a nested tuple)
# is handled separately as a parseable CONTBND string.
#   (results_key, header_card, kind, default)
_MEAS_PARAM_SPEC = [
    ("detect_snr",      "DETSNR",   float, 3.0),
    ("red_offset",      "REDOFF",   float, 0.5),
    ("template_sigma",  "TMPLSIG",  float, 1.5),
    ("hw",              "HW",       float, 4.0),
    ("n_sigma_class",   "NSIGCLS",  float, 2.0),    # legacy single-threshold mirror
    ("n_sigma_emit",    "NSIGEMIT", float, None),   # absent (old file) -> fall back
    ("n_sigma_absorb",  "NSIGABS",  float, None),   #   to NSIGCLS via the resolver
    ("class_stat",      "CLASSTAT", str,   "mf"),    # old files were matched-filter
    ("rest_density",    "RESTDENS", bool,  True),
    ("cont_method",     "CONTMETH", str,   "poly"),
    ("cont_order",      "CONTORD",  int,   1),
    ("flux_for_escape", "FESCFLX",  str,   "A"),
    ("ew_threshold",    "EWTHRESH", float, 20.0),
    ("ew_flux",         "EWFLUX",   str,   "mf"),
    ("mag_col",         "MAGCOL",   str,   "auto"),
    ("flux_to_cgs",     "FLUX2CGS", float, 1e-17),
    ("auto_z_switch",   "EWZSWTCH", float, 2.3),
]


def _fmt_cont_bounds(cb):
    """((1190,1205),(1225,1240)) -> '1190,1205;1225,1240' (FITS-safe, parseable)."""
    try:
        return ";".join(",".join(f"{float(x):g}" for x in pair) for pair in cb)
    except Exception:
        return str(cb)


def _parse_cont_bounds(s):
    """Inverse of _fmt_cont_bounds -> tuple of (lo, hi) pairs (DEFAULT on failure)."""
    if isinstance(s, (tuple, list)):
        return tuple(tuple(float(x) for x in pair) for pair in s)
    try:
        return tuple(tuple(float(x) for x in chunk.split(","))
                     for chunk in str(s).split(";"))
    except Exception:
        return DEFAULT_CONT_BOUNDS


def _write_meas_params(hdr, results):
    """Stamp the measurement configuration onto a FITS header (one card each,
    plus the continuum sidebands as a parseable CONTBND string)."""
    def _safe(v, default):
        if v is None:
            return default
        if isinstance(v, float) and not np.isfinite(v):
            return default
        return v
    for key, card, kind, default in _MEAS_PARAM_SPEC:
        v = _safe(results.get(key), default)
        if v is None:                     # unset (e.g. emit/absorb fell back) -> skip
            continue
        hdr[card] = (bool(v) if kind is bool else
                     int(v)  if kind is int  else
                     float(v) if kind is float else str(v))
    hdr["CONTBND"] = _fmt_cont_bounds(results.get("cont_bounds", DEFAULT_CONT_BOUNDS))
    return hdr


def _read_meas_params(header):
    """Restore the measurement configuration from a FITS header into a kwargs
    dict for run_core_hierarchical (inverse of _write_meas_params)."""
    out = {}
    for key, card, kind, default in _MEAS_PARAM_SPEC:
        if card in header:
            v = header[card]
            out[key] = (bool(v) if kind is bool else
                        int(v)  if kind is int  else
                        float(v) if kind is float else str(v))
        else:
            out[key] = default
    out["cont_bounds"] = _parse_cont_bounds(header.get("CONTBND", DEFAULT_CONT_BOUNDS))
    return out


def write_core_hier_fits(
    path, core_spec, core_err, wave_obs, catalog, results,
    config_meta=None, overwrite=False, extras=None,
):
    """
    Write a HIERARCHICAL core product (run_core_hierarchical output) to FITS.

    Same HDU layout as write_core_fits. The PRIMARY header records the FULL
    measurement configuration via _write_meas_params (DETSNR, REDOFF, TMPLSIG,
    HW, NSIGCLS, RESTDENS, CONTMETH, CONTORD, CONTBND, FESCFLX) so a re-measure
    can faithfully inherit it; the RESULTS table carries the hierarchical columns:
        SN_MF, FLUX_MF, FLUX_MF_ERR, FLUX_B, FLUX_B_ERR,
        FLUX_A, FLUX_A_ERR, MU, SIGMA, V_OFFSET, V_OFFSET_ERR,
        DETECTED, GAUSS_OK, LABEL,  (+ FESC, FESC_ERR if present)

    The catalog is dtype-repaired on the way out (coerce_catalog_dtypes), so
    boolean flags are stored as real FITS logical columns -- not 'True'/'False'
    strings -- and floats stay floats.
    """
    import os
    ngal, nwave = np.asarray(core_spec).shape

    hdr = fits.Header()
    hdr["NGAL"]  = ngal
    hdr["NWAVE"] = nwave
    hdr["MODE"]  = "hierarchical"
    _write_meas_params(hdr, results)          # full measurement config, one place
    hdr["DATE"]  = _dt.datetime.now().isoformat(timespec="seconds")
    if config_meta:
        for k, v in config_meta.items():
            if v is None:
                v = ""
            if isinstance(v, (tuple, list, np.ndarray)):
                v = str(v)
            if isinstance(v, float) and not np.isfinite(v):
                v = ""
            hdr[k[:8].upper()] = v
    primary = fits.PrimaryHDU(header=hdr)

    spec_hdu = fits.ImageHDU(np.asarray(core_spec, dtype=np.float32), name="SPEC")
    err_hdu  = fits.ImageHDU(np.asarray(core_err,  dtype=np.float32), name="ERROR")
    wave_hdu = fits.ImageHDU(np.asarray(wave_obs,  dtype=np.float32), name="WAVE")

    # store real bool/float dtypes (not stringified ascii flags) so flags
    # round-trip correctly -- bool('False') == True was the old bug.
    cat_hdu      = fits.table_to_hdu(coerce_catalog_dtypes(Table(catalog)))
    cat_hdu.name = "CATALOG"

    float_cols = ["sn_mf", "flux_mf", "flux_mf_err", "flux_B", "flux_B_err", "sn_B",
                  "flux_A", "flux_A_err",
                  "mu", "sigma", "v_offset", "v_offset_err",
                  "ew_rest", "ew_rest_err", "fesc", "fesc_err"]
    cols = []
    for name in float_cols:
        if name in results:
            cols.append(fits.Column(name.upper(), "E",
                        array=np.asarray(results[name], dtype=np.float32)))
    if "detected" in results:
        cols.append(fits.Column("DETECTED", "L",
                    array=np.asarray(results["detected"], dtype=bool)))
    if "gauss_success" in results:
        cols.append(fits.Column("GAUSS_OK", "L",
                    array=np.asarray(results["gauss_success"], dtype=bool)))
    if "label" in results:
        cols.append(fits.Column("LABEL", "8A",
                    array=np.array(results["label"], dtype="U8")))
    res_hdu      = fits.BinTableHDU.from_columns(cols)
    res_hdu.name = "RESULTS"

    hdul = fits.HDUList([primary, spec_hdu, err_hdu, wave_hdu, cat_hdu, res_hdu])

    # --- optional: persist the background so a saved run is self-diagnosing ---
    if extras is not None:
        bg = extras.get("background")
        if bg is not None:
            import json
            bg_hdu = fits.ImageHDU(np.asarray(bg, dtype=np.float32), name="BACKGROUND")
            bgp_params = dict(extras.get("bg_params") or {})
            # full parameter set as JSON (astropy CONTINUE-encodes long strings),
            # so it round-trips with real key names and types -- self-describing.
            bg_hdu.header["BGPARAMS"] = (
                json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v)
                                else v) for k, v in bgp_params.items()}, default=str),
                "bg parameters that produced this HDU (JSON)")
            # a few of the key knobs also as plain cards, for quick header eyeballing
            for card, key in (("BGINNER", "bg_inner_arcsec"),
                              ("BGOUTER", "bg_outer_arcsec"),
                              ("MINBGFIB", "min_bg_fibers"),
                              ("BGSMOOTH", "bg_smooth_method")):
                if key in bgp_params and bgp_params[key] is not None:
                    bg_hdu.header[card] = bgp_params[key]
            hdul.append(bg_hdu)
        bgp = extras.get("background_pooled")
        if bgp is not None:
            hdul.append(fits.ImageHDU(np.asarray(bgp, dtype=np.float32), name="BGPOOLED"))
        # per-galaxy coverage counts -> small companion table
        nbf = extras.get("n_bg_fibers"); neb = extras.get("n_exp_own_bg")
        if nbf is not None and neb is not None:
            bginfo = fits.BinTableHDU.from_columns([
                fits.Column("N_BG_FIBERS",  "J", array=np.asarray(nbf, dtype=np.int32)),
                fits.Column("N_EXP_OWN_BG", "J", array=np.asarray(neb, dtype=np.int32)),
            ])
            bginfo.name = "BGINFO"
            hdul.append(bginfo)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    hdul.writeto(path, overwrite=overwrite)
    print(f"wrote hierarchical core FITS -> {path}  ({ngal} galaxies)"
          + ("  [+background]" if extras is not None and extras.get("background") is not None else ""))
    return path


def read_core_fits(path):
    """
    Read a core FITS back into plain arrays. Works for BOTH the flat
    (write_core_fits) and hierarchical (write_core_hier_fits) schemas -- it keys
    off the PRIMARY MODE card and only restores the columns that are present.

    Returns dict: core_spec, core_err, wave_obs, catalog (Table), results (dict
    of per-galaxy arrays), header.
    """
    with fits.open(path) as hdul:
        header    = hdul[0].header
        core_spec = hdul["SPEC"].data.astype(float)
        core_err  = hdul["ERROR"].data.astype(float)
        wave_obs  = hdul["WAVE"].data.astype(float)
        catalog   = coerce_catalog_dtypes(Table(hdul["CATALOG"].data))
        rt        = hdul["RESULTS"].data
        results   = {c.lower(): np.asarray(rt[c]) for c in rt.names}
        mode = str(header.get("MODE", "flat")).lower()
        # string/bool columns -> proper dtypes if present
        if "label" in results:
            results["label"] = np.array(results["label"], dtype=str)
        for bcol in ("success", "detected", "gauss_ok"):
            if bcol in results:
                results[bcol] = np.array(results[bcol], dtype=bool)
        if "gauss_ok" in results:        # normalize to the run_core_hierarchical key
            results["gauss_success"] = results.pop("gauss_ok")
        if mode == "hierarchical":
            results["mode"] = "hierarchical"
            results.update(_read_meas_params(header))   # full measurement config
        else:
            results["mode"]        = "flat"
            results["method"]      = str(header.get("METHOD", "unknown"))
            results["n_sigma"]     = float(header.get("NSIGMA", np.nan))
            results["n_bootstrap"] = int(header.get("NBOOT", 0))
        # --- optional background block (present if written with extras=) ---
        extras = {}
        if "BACKGROUND" in hdul:
            import json
            extras["background"] = hdul["BACKGROUND"].data.astype(float)
            raw = hdul["BACKGROUND"].header.get("BGPARAMS", "")
            try:
                extras["bg_params"] = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                extras["bg_params"] = {}
        if "BGPOOLED" in hdul:
            extras["background_pooled"] = hdul["BGPOOLED"].data.astype(float)
        if "BGINFO" in hdul:
            bi = hdul["BGINFO"].data
            extras["n_bg_fibers"]  = np.asarray(bi["N_BG_FIBERS"])
            extras["n_exp_own_bg"] = np.asarray(bi["N_EXP_OWN_BG"])
    out = dict(core_spec=core_spec, core_err=core_err, wave_obs=wave_obs,
               catalog=catalog, results=results, header=header)
    if extras:
        out["extras"] = extras
    return out


# ---------------------------------------------------------------------
# 5b.  RE-MEASURE  (refresh measurements on an already-extracted file)
# ---------------------------------------------------------------------
# The extraction (SPEC/ERROR/WAVE/CATALOG) is the expensive part and never
# changes; the measurement (RESULTS) is fast and iterated as measure.py / the
# estimators improve. remeasure_core_fits reads a product, re-runs ONLY the
# measurement, and re-writes -- so every measurement generation is cheap and the
# spectra never drift between versions.

# The full set of run_core_hierarchical kwargs persisted in the file (and thus
# inheritable on re-measure): the scalar spec keys + cont_bounds.
_MEASURE_KEYS = tuple(k for k, *_ in _MEAS_PARAM_SPEC) + ("cont_bounds",)


def measure_params_from_file(prod_or_path):
    """
    Recover the full measurement configuration recorded in a core FITS as a
    kwargs dict for run_core_hierarchical. Accepts a read_core_fits dict or a
    path. Reads straight from the header so it works even on a dict whose
    results were built before _read_meas_params populated them.
    """
    if isinstance(prod_or_path, dict):
        header = prod_or_path.get("header")
        if header is not None:
            return _read_meas_params(header)
        res = prod_or_path["results"]
        return {k: res[k] for k in _MEASURE_KEYS if k in res and res[k] is not None}
    return _read_meas_params(read_core_fits(prod_or_path)["header"])


def remeasure_core_fits(in_path, out_path=None, *, inherit_params=True,
                        overwrite=True, config_meta=None, verbose=True,
                        **measure_kwargs):
    """
    Re-run the measurement pass on an ALREADY-EXTRACTED core FITS and re-write it.

    Reads SPEC/ERROR/WAVE/CATALOG from `in_path` and passes them through
    untouched; re-runs run_core_hierarchical, regenerating the RESULTS HDU and the
    measurement header cards only. Use whenever measure.py / the estimators
    improve and the product needs refreshing without re-extracting.

    inherit_params : start from the params recorded in the file, then apply your
        **measure_kwargs on top (explicit kwargs win) -- a plain re-run reproduces
        the file, and you override only the knob you changed. Set False to start
        from run_core_hierarchical's own defaults.
    out_path : defaults to in_path (in-place). Pass a new (dated) name to keep the
        previous measurement generation alongside the new one.
    config_meta : extra header keywords to stamp (merged with a REMEASUR date).
    **measure_kwargs : forwarded to run_core_hierarchical (detect_snr, red_offset,
        red_offset, cont_bounds, flux_for_escape, template_sigma, hw, escape, ...).

    Returns out_path.
    """
    prod = read_core_fits(in_path)

    params = measure_params_from_file(prod) if inherit_params else {}
    params.update(measure_kwargs)            # explicit overrides win
    params.setdefault("verbose", verbose)

    if verbose:
        shown = {k: params[k] for k in params if k != "verbose"}
        print(f"re-measuring {in_path}  (ngal={len(prod['core_spec'])})")
        print(f"  measurement params: {shown}")

    results = run_core_hierarchical(
        prod["core_spec"], prod["core_err"], prod["wave_obs"], prod["catalog"],
        **params,
    )

    out_path = out_path or in_path
    meta = dict(config_meta or {})
    meta.setdefault("REMEASUR", _dt.datetime.now().isoformat(timespec="seconds"))
    return write_core_hier_fits(
        out_path, prod["core_spec"], prod["core_err"], prod["wave_obs"],
        prod["catalog"], results, config_meta=meta, overwrite=overwrite,
    )


# =====================================================================
# 5c.  CORE STACKING  (relocated)
#      build_core_stacks lives in stack.py (next to build_stacks); its quicklook
#      plot is plotting.plot_core_stack. build_core_stacks / concat_core_prods
#      are re-exported here (no matplotlib dep) so core.build_core_stacks keeps
#      working; call plotting.plot_core_stack directly for the figure.
# =====================================================================

from .stack import build_core_stacks, concat_core_prods   # noqa: E402,F401


# =====================================================================
# 6.  C1 EXTRACTION SOCKET  (wires optics.py; heavy, run once)
# =====================================================================

def extract_core_spectra(config: "PipelineConfig", run_table=None,
                         core_radius=3.5, ra_col="RA", dec_col="DEC",
                         return_extras=False, bad_fiber_mask=None, vdfi=None,
                         save_background=True):
    """
    C1 -- PSF-weighted optimal core extraction.   SOCKET over optics.py.

    Runs the point-source path that the binned extraction explicitly deferred.
    No per-galaxy caching (the sample is small; the whole extraction fits in
    memory): just a loop over galaxies, returning (Spec, Err, wave_obs, catalog).

    Call sequence per galaxy (functions that fill each step):
      0. load_field(config, build_psf=True) -> PSF interpolator in vdfi['PSF']
         (extract.load_vdfi_field with build_psf=True)
      1. build_run_table / build_bad_fiber_mask  (same as Stage 1)
      2. per exposure:
           dra  = (RA_fib - RA_gal) * cos(dec) * 3600   [arcsec]
           ddec = (DEC_fib - DEC_gal) * 3600
           bg from the [bg_inner, bg_outer) annulus, smoothed -- REUSE the
           per-exposure background block from extract.extract_fibers verbatim
           so core and halo share one background definition.
           optics.get_spectrum_exposure(spectra_e, error_e, dra, ddec,
                                        seeing_e, vdfi['PSF'], wave)
           -> spec_e (nwave,), err_e (nwave,), weights_e, goodpix_mask_e
      3. inverse-variance combine across exposures:
           w = 1 / err_e^2  (per pixel, where goodpix)
           Spec[i] = sum(spec_e * w, axis=0) / sum(w, axis=0)
           Err[i]  = 1 / sqrt(sum(w, axis=0))
      4. optics.get_continuum(Spec[i:i+1, :], nbins=25) -> continuum diagnostic
         (store or discard; the C2 estimators re-derive their own continuum in
         rest frame)

    Parameters
    ----------
    config      : PipelineConfig    -- field, paths, bg annulus, masking, z/mass cols.
    run_table   : optional Table    -- pre-cut catalog rows (else build_run_table).
    core_radius : float (arcsec)    -- aperture of fibers fed to the PSF extraction.
                  Fibers beyond this get ~zero PSF weight anyway; the cut just keeps
                  each exposure's working set small and bounded.
    ra_col, dec_col : catalog coordinate columns (default 'RA'/'DEC', as Stage 1).
    return_extras   : if True, also return a dict of per-galaxy diagnostics
                      (nfib_used, n_exp_used, continuum, background, ...) -- handy
                      for the overlay plot AND for background bug-testing.
    save_background : if True (default), PSF-extract the per-exposure background
                      through the SAME optimal path and record, per galaxy, the
                      background actually folded into Spec (extras['background']).
                      Because the extraction is LINEAR with identical weights,
                          Spec + extras['background'] == the UNSUBTRACTED spectrum,
                      i.e. the exact input for a subtracted-vs-unsubtracted
                      diagnosis. Costs a second get_spectrum_exposure call per
                      exposure (~doubles extraction time); set False to skip it
                      (the cheap pooled background + coverage counts are still
                      recorded either way).
    bad_fiber_mask  : optional pre-built (nexp, nfib) boolean mask (True = bad).
                      None (default) -> built from config via build_bad_fiber_mask.
                      Injecting one lets compare_mask_spectra extract the SAME
                      galaxies under several candidate masks (old image mask vs
                      new segmap, or two segmap bakes) without touching config.
    vdfi            : optional pre-loaded field dict from load_field_with_psf.
                      None (default) -> loaded here. Pass it when calling this
                      repeatedly so the field is read from disk only once.

    Returns:
        Spec    (ngal, nwave)  float32 -- observed-frame PSF-weighted core spectra
        Err     (ngal, nwave)  float32 -- propagated per-pixel errors
        wave    (nwave,)       float32 -- observed wavelength grid
        catalog astropy Table            -- aligned to the galaxy axis
      (+ extras dict if return_extras=True)
    """
    # Lazy imports: keep core.py's top-level import cheap and dodge any
    # circular import with extract.py (which imports optics, masking, virial...).
    from . import optics
    from .extract import (
        load_field, build_run_table, build_bad_fiber_mask, combine_fibers,
    )

    # ---- 0. field + PSF + masks + catalog -------------------------------
    # vdfi injectable so repeat calls (e.g. compare_mask_spectra, one call per
    # candidate mask) reuse ONE loaded field instead of re-reading it each time.
    if vdfi is None:
        vdfi = load_field_with_psf(config)    # load_field(config) but build_psf=True
    PSF       = vdfi["PSF"]
    if PSF is None:
        raise RuntimeError(
            "vdfi['PSF'] is None -- the field was loaded without build_psf=True. "
            "extract_core_spectra needs the Moffat PSF grid."
        )
    wave      = np.asarray(vdfi["wave"], dtype=float)
    RA        = vdfi["RA"]; DEC = vdfi["DEC"]
    spectra   = vdfi["spectra"]; error = vdfi["error"]
    seeing    = np.asarray(vdfi["seeing"], dtype=float)   # per-exposure (one scalar each)
    dar_ra    = vdfi.get("dar_ra");  dar_dec = vdfi.get("dar_dec")  # (nexp, nwave) DAR
    nexp      = RA.shape[0]
    nfib_tot  = RA.shape[1]
    nwave     = wave.size

    catalog   = build_run_table(config, run_table)
    bad_fiber = (bad_fiber_mask if bad_fiber_mask is not None
                 else build_bad_fiber_mask(config, vdfi, run_table=catalog))

    ngal = len(catalog) if config.num_gal is None else min(config.num_gal, len(catalog))
    catalog = catalog[:ngal]

    bg_method    = config.resolved_bg_combine_method().lower()
    bg_needs_err = bg_method in ("mean", "inv_var")
    sc_sigma     = float(getattr(config, "sigma_clip_sigma", 3.0))
    sc_maxiters  = getattr(config, "sigma_clip_maxiters", 5)

    def _maybe_smooth(spec_1d, z):
        if not config.smooth_bg:
            return spec_1d
        method = getattr(config, "bg_smooth_method", "interp_gauss").lower()
        if method == "interp_gauss":
            from .smoothing import smooth_interp_then_gauss
            return smooth_interp_then_gauss(
                spec_1d, wave, z, lya_rest=config.LYA_REST,
                sigma_A=config.bg_smooth_sigma_A,
                mask_halfwidth_A=config.line_mask_halfwidth_A,
            )
        from .smoothing import smooth_spectrum_nan_safe_adaptive   # legacy 'adaptive'
        sv = config.smoothing_values
        return smooth_spectrum_nan_safe_adaptive(
            spec_1d, wave, sigma_A=sv[0], edge_sigma_A=sv[1], edge_width_A=sv[2]
        )

    # ---- outputs --------------------------------------------------------
    Spec = np.full((ngal, nwave), np.nan, dtype=np.float32)
    Err  = np.full((ngal, nwave), np.nan, dtype=np.float32)
    Cont = np.full((ngal, nwave), np.nan, dtype=np.float32)
    nfib_used = np.zeros(ngal, dtype=np.int32)
    nexp_used = np.zeros(ngal, dtype=np.int32)
    # --- background bookkeeping (saved so a run is self-diagnosing later) ---
    BG_pooled    = np.full((ngal, nwave), np.nan, dtype=np.float32)  # galaxy-pooled annulus bg
    BG_sub       = np.full((ngal, nwave), np.nan, dtype=np.float32)  # bg actually folded into Spec
    n_bg_fibers  = np.zeros(ngal, dtype=np.int32)                    # annulus fibers pooled
    n_exp_own_bg = np.zeros(ngal, dtype=np.int32)                    # exposures with their own annulus bg

    it = _tqdm(range(ngal), desc="extract_core")
    for cnt in it:
        ra  = float(catalog[ra_col][cnt])
        dec = float(catalog[dec_col][cnt])
        z   = (float(catalog[config.z_col][cnt])
               if config.z_col in catalog.colnames else np.nan)

        # ---- geometry + good-fiber mask (identical to Stage 1) ----
        dist_all  = np.full((nexp, nfib_tot), np.nan)
        dra_all   = np.full((nexp, nfib_tot), np.nan)
        ddec_all  = np.full((nexp, nfib_tot), np.nan)
        good_all  = np.zeros((nexp, nfib_tot), dtype=bool)
        for e in range(nexp):
            dra  = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
            ddec = (DEC[e] - dec) * 3600.
            dra_all[e]  = dra
            ddec_all[e] = ddec
            dist_all[e] = np.sqrt(dra ** 2 + ddec ** 2)
            good_all[e] = (~bad_fiber[e]) & np.isfinite(RA[e]) & np.isfinite(DEC[e])

        # ---- per-exposure background (REUSED verbatim from extract_fibers) ----
        bg_per_exp     = np.full((nexp, nwave), np.nan, dtype=np.float32)
        has_own_bg     = np.zeros(nexp, dtype=bool)
        all_bg_fibers, all_bg_errs = [], []
        for e in range(nexp):
            bg_sel = ((dist_all[e] >= config.bg_inner_arcsec) &
                      (dist_all[e] <  config.bg_outer_arcsec) &
                      good_all[e])
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
                bg_per_exp[e] = _maybe_smooth(bg_e, z)
                has_own_bg[e] = True
        if all_bg_fibers:
            sp_all = np.concatenate(all_bg_fibers, axis=0)
            er_all = np.concatenate(all_bg_errs, axis=0) if bg_needs_err else None
            bg_gal, _ = combine_fibers(sp_all, er_all, method=bg_method,
                                       sigma=sc_sigma, maxiters=sc_maxiters)
            bg_gal = _maybe_smooth(bg_gal, z)
        else:
            bg_gal = np.full(nwave, np.nan, dtype=np.float32)
        for e in range(nexp):
            if not has_own_bg[e]:
                bg_per_exp[e] = bg_gal
        # record the pooled annulus background + coverage for this galaxy
        BG_pooled[cnt]    = bg_gal
        n_bg_fibers[cnt]  = int(sum(a.shape[0] for a in all_bg_fibers))
        n_exp_own_bg[cnt] = int(np.sum(has_own_bg))

        # ---- per-exposure PSF-weighted optimal extraction ----
        # inverse-variance accumulators across exposures
        acc_num = np.zeros(nwave)          # sum(spec_e / var_e)
        acc_w   = np.zeros(nwave)          # sum(1 / var_e)
        acc_bg  = np.zeros(nwave)          # sum(bg_e / var_e) -- bg folded into Spec
        for e in range(nexp):
            sel = good_all[e] & (dist_all[e] <= core_radius)
            nsel = int(np.sum(sel))
            if nsel == 0:
                continue
            # background-subtract the science fibers feeding the extraction
            sp_e = spectra[e, sel].astype(float) - bg_per_exp[e][np.newaxis, :]
            er_e = error[e, sel].astype(float)
            see_e = _exposure_seeing(seeing, e)
            # get_spectrum_exposure expects dra/ddec as 2-D (nfib, nwave): the
            # per-fiber offset AT EACH WAVELENGTH, including the DAR term (the
            # atmosphere shifts every fiber's effective sky position vs lambda).
            dra_w, ddec_w = _offsets_with_dar(
                dra_all[e, sel], ddec_all[e, sel], dar_ra, dar_dec, e, nwave,
            )
            # get_spectrum_exposure does nansum(...)/nansum(weights^2/var); at
            # wavelength pixels where every contributing fiber is NaN/masked the
            # denominator is 0, so it (correctly) emits 0/0 -> NaN and warns. The
            # NaN is exactly what we want (that pixel has no data); the warning is
            # just noise. Suppress only here, around this one call.
            with np.errstate(divide="ignore", invalid="ignore"), \
                 warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                spec_e, err_e, _w, _pf = optics.get_spectrum_exposure(
                    sp_e, er_e, dra_w, ddec_w, see_e, PSF, wave,
                )
            spec_e = np.atleast_1d(np.asarray(spec_e, dtype=float))
            err_e  = np.atleast_1d(np.asarray(err_e,  dtype=float))
            if spec_e.size != nwave:        # get_spectrum_exposure rejected this exp
                continue
            good_pix = np.isfinite(spec_e) & np.isfinite(err_e) & (err_e > 0)
            if not np.any(good_pix):
                continue
            inv_var = np.zeros(nwave)
            inv_var[good_pix] = 1.0 / err_e[good_pix] ** 2
            acc_num[good_pix] += spec_e[good_pix] * inv_var[good_pix]
            acc_w  [good_pix] += inv_var[good_pix]
            # Fold the SAME per-exposure background through the SAME optimal
            # extraction (linear op, identical weights): extracting the fiber-
            # constant bg gives its exact contribution to spec_e. Accumulated
            # with the science inverse-variance, acc_bg/acc_w is the background
            # actually subtracted from Spec -> Spec + BG_sub == unsubtracted.
            if save_background:
                bg_bcast = np.broadcast_to(
                    bg_per_exp[e][np.newaxis, :], sp_e.shape).astype(float)
                with np.errstate(divide="ignore", invalid="ignore"), \
                     warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bgspec_e, _, _, _ = optics.get_spectrum_exposure(
                        bg_bcast, er_e, dra_w, ddec_w, see_e, PSF, wave,
                    )
                bgspec_e = np.atleast_1d(np.asarray(bgspec_e, dtype=float))
                if bgspec_e.size == nwave:
                    gp_bg = good_pix & np.isfinite(bgspec_e)
                    acc_bg[gp_bg] += bgspec_e[gp_bg] * inv_var[gp_bg]
            nexp_used[cnt] += 1
            nfib_used[cnt] += int(np.sum(sel))

        has = acc_w > 0
        Spec[cnt, has] = (acc_num[has] / acc_w[has]).astype(np.float32)
        Err [cnt, has] = (1.0 / np.sqrt(acc_w[has])).astype(np.float32)
        if save_background:
            BG_sub[cnt, has] = (acc_bg[has] / acc_w[has]).astype(np.float32)
        # continuum diagnostic (C2 re-derives its own in rest frame; this is just
        # a quick-look, e.g. for the overlay plot's spectrum inset)
        try:
            Cont[cnt] = optics.get_continuum(Spec[cnt:cnt + 1, :], nbins=25)[0]
        except Exception:
            pass

    if return_extras:
        extras = dict(
            continuum=Cont, nfib_used=nfib_used, n_exp_used=nexp_used,
            core_radius=core_radius, field=config.normalized_field(),
            # --- background: what was subtracted, for later bug-testing ---
            #   background        : (ngal,nwave) bg folded into Spec; Spec+background
            #                       == the UNSUBTRACTED spectrum. None if save_background=False.
            #   background_pooled : (ngal,nwave) galaxy-pooled annulus bg, matching
            #                       inspect_core_background's bg_gal_smooth (cheap cross-check).
            #   n_bg_fibers       : (ngal,) annulus fibers pooled into the background.
            #   n_exp_own_bg      : (ngal,) exposures that had their own annulus bg
            #                       (the rest fell back to the pooled galaxy bg).
            #   bg_params         : the exact parameters that produced this background,
            #                       so a saved run is self-describing and reproducible.
            background=(BG_sub if save_background else None),
            background_pooled=BG_pooled,
            n_bg_fibers=n_bg_fibers,
            n_exp_own_bg=n_exp_own_bg,
            bg_params=dict(
                bg_inner_arcsec=config.bg_inner_arcsec,
                bg_outer_arcsec=config.bg_outer_arcsec,
                min_bg_fibers=config.min_bg_fibers,
                smooth_bg=config.smooth_bg,
                bg_smooth_method=getattr(config, "bg_smooth_method", "interp_gauss"),
                bg_smooth_sigma_A=getattr(config, "bg_smooth_sigma_A", None),
                line_mask_halfwidth_A=getattr(config, "line_mask_halfwidth_A", None),
                bg_combine_method=config.resolved_bg_combine_method(),
                mask_method=getattr(config, "mask_method", None),
                core_radius=core_radius,
            ),
        )
        return Spec, Err, wave.astype(np.float32), catalog, extras
    return Spec, Err, wave.astype(np.float32), catalog


def inspect_core_background(
    config: "PipelineConfig", galaxy_ids, vdfi=None, core_radius=3.5,
    id_col=None, run_table=None, ra_col="RA", dec_col="DEC",
):
    """
    Diagnostic: compute the per-exposure + pooled annulus BACKGROUND for a few
    galaxies, RAW and SMOOTHED, WITHOUT running the full PSF core extraction.
    Mirrors the exact background block inside extract_core_spectra, so what
    you see here is exactly what gets subtracted there -- but returned instead
    of thrown away, and fast enough to use on a handful of galaxies at a time.

    Use this to diagnose a suspicious core spectrum (e.g. a negative continuum
    or absorption lines dipping below zero after subtraction): look at
    bg_gal_raw / bg_gal_smooth directly for the offending galaxies BEFORE
    deciding whether the background scheme itself needs to change.

    galaxy_ids : id_col values (or any list) to diagnose.
    id_col     : catalog column to match galaxy_ids against; None -> config.id_col.
    run_table  : optional pre-built catalog; None -> build_run_table(config).
    vdfi       : optional pre-loaded field (extract.load_field); None -> loaded here
                 (no PSF needed for this -- background only).

    Returns a dict keyed by id_col value, each entry:
        {'bg_gal_raw', 'bg_gal_smooth'   : (nwave,) pooled-annulus background,
                                            before / after config.bg_smooth_method
         'bg_per_exp_smooth'             : (nexp, nwave) per-exposure background
                                            actually used in extraction (own
                                            annulus where available, else the
                                            galaxy fallback -- same as
                                            extract_core_spectra's bg_per_exp)
         'has_own_bg'                    : (nexp,) bool, per-exposure annulus
                                            cleared min_bg_fibers on its own
         'wave', 'n_bg_fibers_total'}
    """
    from .extract import load_field, build_run_table, build_bad_fiber_mask, combine_fibers

    idc = id_col or getattr(config, "id_col", "ID")
    full_table = run_table if run_table is not None else build_run_table(config)
    ids = np.asarray(galaxy_ids)
    sel_table = full_table[np.isin(np.asarray(full_table[idc]), ids)]
    if len(sel_table) == 0:
        raise ValueError(f"None of {list(ids)} found in catalog column {idc!r}")

    if vdfi is None:
        vdfi = load_field(config)
    RA, DEC, wave = vdfi["RA"], vdfi["DEC"], vdfi["wave"]
    spectra, error = vdfi["spectra"], vdfi.get("error")
    nexp, nfib_tot = RA.shape
    nwave = spectra.shape[-1]

    bad_fiber = build_bad_fiber_mask(config, vdfi, run_table=full_table)
    bg_method = config.resolved_bg_combine_method().lower()
    bg_needs_err = bg_method in ("mean", "inv_var")
    sc_sigma = float(getattr(config, "sigma_clip_sigma", 3.0))
    sc_maxiters = getattr(config, "sigma_clip_maxiters", 5)

    def _smooth(spec_1d, z):
        if not config.smooth_bg:
            return spec_1d
        method = getattr(config, "bg_smooth_method", "interp_gauss").lower()
        if method == "interp_gauss":
            from .smoothing import smooth_interp_then_gauss
            return smooth_interp_then_gauss(
                spec_1d, wave, z, lya_rest=config.LYA_REST,
                sigma_A=config.bg_smooth_sigma_A,
                mask_halfwidth_A=config.line_mask_halfwidth_A)
        from .smoothing import smooth_spectrum_nan_safe_adaptive
        sv = config.smoothing_values
        return smooth_spectrum_nan_safe_adaptive(
            spec_1d, wave, sigma_A=sv[0], edge_sigma_A=sv[1], edge_width_A=sv[2])

    out = {}
    for row in sel_table:
        ra, dec = float(row[ra_col]), float(row[dec_col])
        z = float(row[config.z_col]) if config.z_col in sel_table.colnames else np.nan

        dist_all = np.full((nexp, nfib_tot), np.nan)
        good_all = np.zeros((nexp, nfib_tot), dtype=bool)
        for e in range(nexp):
            dra = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
            ddec = (DEC[e] - dec) * 3600.
            dist_all[e] = np.sqrt(dra ** 2 + ddec ** 2)
            good_all[e] = (~bad_fiber[e]) & np.isfinite(RA[e]) & np.isfinite(DEC[e])

        bg_per_exp = np.full((nexp, nwave), np.nan, dtype=np.float32)
        has_own_bg = np.zeros(nexp, dtype=bool)
        all_bg_fibers, all_bg_errs, n_bg_total = [], [], 0
        for e in range(nexp):
            bg_sel = ((dist_all[e] >= config.bg_inner_arcsec) &
                     (dist_all[e] < config.bg_outer_arcsec) & good_all[e])
            n_bg = int(np.sum(bg_sel))
            n_bg_total += n_bg
            if n_bg == 0:
                continue
            all_bg_fibers.append(spectra[e, bg_sel])
            if bg_needs_err:
                all_bg_errs.append(error[e, bg_sel])
            if n_bg >= config.min_bg_fibers:
                er_bg = error[e, bg_sel] if bg_needs_err else None
                bg_e, _ = combine_fibers(spectra[e, bg_sel], er_bg, method=bg_method,
                                         sigma=sc_sigma, maxiters=sc_maxiters)
                bg_per_exp[e] = _smooth(bg_e, z)
                has_own_bg[e] = True

        if all_bg_fibers:
            sp_all = np.concatenate(all_bg_fibers, axis=0)
            er_all = np.concatenate(all_bg_errs, axis=0) if bg_needs_err else None
            bg_gal_raw, _ = combine_fibers(sp_all, er_all, method=bg_method,
                                           sigma=sc_sigma, maxiters=sc_maxiters)
            bg_gal_smooth = _smooth(bg_gal_raw, z)
        else:
            bg_gal_raw = np.full(nwave, np.nan, dtype=np.float32)
            bg_gal_smooth = bg_gal_raw.copy()

        for e in range(nexp):
            if not has_own_bg[e]:
                bg_per_exp[e] = bg_gal_smooth

        out[row[idc]] = dict(
            bg_gal_raw=np.asarray(bg_gal_raw, dtype=np.float32),
            bg_gal_smooth=np.asarray(bg_gal_smooth, dtype=np.float32),
            bg_per_exp_smooth=bg_per_exp, has_own_bg=has_own_bg,
            wave=np.asarray(wave, dtype=np.float32), n_bg_fibers_total=n_bg_total,
            z=z,
        )
    return out


def diagnose_core_fibers(
    config: "PipelineConfig", galaxy_ids, vdfi=None, core_radius=3.5,
    id_col=None, run_table=None, ra_col="RA", dec_col="DEC",
):
    """
    Explain WHY a core spectrum is NaN (or confirm it should extract) for a few
    galaxies, by counting fibers at each stage of extract_core_spectra's
    selection -- WITHOUT running the full PSF extraction. Distinguishes causes
    that look identical on a plotted spectrum (and identical in the appendix
    figure, which shows the whole 15" box, not just the core aperture):

      * 'all core fibers MASKED'  -- fibers land in the core aperture but every
        one is flagged bad. FIXABLE AT BAKE TIME: unmask this galaxy's segment
        in the baked segmap FITS (zero it out) and re-bake -- BUT only if the
        mask is clipping the galaxy's own light, not a real neighbour. In a
        crowded field (many 'near' in the appendix title) the maskers are
        usually genuine bright sources bleeding in, so unmasking pulls that
        contamination straight into the core.
      * 'core fibers unmasked but DATA all-NaN'  -- fibers are kept but their
        spectra are non-finite (dead fibers / no wavelength coverage). NOT a
        masking problem; no override radius will help.
      * 'no fibers in core aperture'  -- nothing lands within core_radius
        (coverage gap / field edge). Also not a masking problem.
      * 'no background fibers'  -- the [bg_inner,bg_outer) annulus has no usable
        fibers, so the bg subtraction turns the core spectrum NaN even though
        the core fibers themselves are fine.
      * 'core looks extractable'  -- finite core fibers AND background; any NaN
        is coming from somewhere other than fiber selection.

    Uses the SAME geometry + bad-fiber mask as extract_core_spectra, with the
    CURRENT config -- run it before and after retweaking the baked segmap
    mask to see the effect. Pass a pre-loaded vdfi (from extract.load_field)
    to reuse it; otherwise one is loaded (no PSF needed).

    Returns an astropy Table, one row per found id:
        id, n_core (finite-coord fibers within core_radius, masked or not),
        n_core_good (of those, unmasked), n_core_data (unmasked fibers with any
        finite spectral pixel), wave_cov (fraction of wavelength pixels that
        have >=1 good core fiber with data -- this is what predicts how much of
        the output spectrum is non-NaN; a low value means data-starved even if
        n_core_good looks healthy), n_bg_good (unmasked annulus fibers), verdict.
    """
    idc = id_col or getattr(config, "id_col", "ID")
    if vdfi is None:
        from .extract import load_field
        vdfi = load_field(config)
    from .extract import build_run_table, build_bad_fiber_mask

    full_table = run_table if run_table is not None else build_run_table(config)
    ids = np.asarray(galaxy_ids)
    sel_table = full_table[np.isin(np.asarray(full_table[idc]), ids)]
    if len(sel_table) == 0:
        raise ValueError(f"None of {list(ids)} found in catalog column {idc!r}")
    missing = set(ids.tolist()) - set(np.asarray(sel_table[idc]).tolist())
    if missing:
        print(f"diagnose_core_fibers: {len(missing)} id(s) not found: {sorted(missing)}")

    bad_fiber = build_bad_fiber_mask(config, vdfi, run_table=full_table)
    RA = vdfi["RA"]; DEC = vdfi["DEC"]
    spectra = vdfi["spectra"]
    nexp = RA.shape[0]
    bg_in, bg_out = config.bg_inner_arcsec, config.bg_outer_arcsec

    nwave = spectra.shape[2]
    rows = {"id": [], "n_core": [], "n_core_good": [], "n_core_data": [],
            "wave_cov": [], "n_bg_good": [], "verdict": []}
    for row in sel_table:
        ra, dec = float(row[ra_col]), float(row[dec_col])
        n_core = n_core_good = n_core_data = n_bg = 0
        wave_cov = np.zeros(nwave, dtype=int)   # per-pixel count of good core fibers with data
        for e in range(nexp):
            dra = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
            ddec = (DEC[e] - dec) * 3600.
            dist = np.sqrt(dra ** 2 + ddec ** 2)
            finite_coord = np.isfinite(RA[e]) & np.isfinite(DEC[e])
            good = (~bad_fiber[e]) & finite_coord
            in_core = (dist <= core_radius) & finite_coord
            core_good = in_core & good
            n_core += int(np.sum(in_core))
            n_core_good += int(np.sum(core_good))
            if np.any(core_good):
                fin = np.isfinite(spectra[e, core_good])          # (n_sel, nwave)
                n_core_data += int(np.sum(np.any(fin, axis=1)))   # fibers with >=1 finite pixel
                wave_cov += fin.sum(axis=0)                        # pixels covered, pooled over exp
            n_bg += int(np.sum((dist >= bg_in) & (dist < bg_out) & good))

        # frac of wavelength pixels that will have ANY good-fiber data -- this is
        # what actually determines how much of the output spectrum is non-NaN.
        n_wave_cov = int(np.sum(wave_cov > 0))
        frac_wave = n_wave_cov / nwave if nwave else 0.0

        if n_core == 0:
            verdict = "no fibers in core aperture (coverage gap / field edge)"
        elif n_core_good == 0:
            verdict = "all core fibers MASKED -> unmask its segment in the baked mask IF self-light (see docstring)"
        elif n_wave_cov == 0:
            verdict = "core fibers unmasked but DATA all-NaN -> not a masking issue"
        elif frac_wave < 0.5:
            verdict = (f"DATA-STARVED core: only {frac_wave:.0%} of wavelengths have a good "
                       f"fiber ({n_core_data}/{n_core_good} fibers carry data) -> spectrum "
                       f"mostly/all NaN. Coverage/data issue, NOT masking -- exclude it.")
        elif n_bg == 0:
            verdict = "no background fibers in annulus -> bg subtraction yields NaN"
        else:
            verdict = "core looks extractable (finite fibers + bg)"

        rows["id"].append(row[idc]); rows["n_core"].append(n_core)
        rows["n_core_good"].append(n_core_good); rows["n_core_data"].append(n_core_data)
        rows["wave_cov"].append(round(frac_wave, 3))
        rows["n_bg_good"].append(n_bg); rows["verdict"].append(verdict)

    t = Table(rows)
    for r in t:
        print(f"{idc}={r['id']}: core={r['n_core']} good={r['n_core_good']} "
              f"data_fibers={r['n_core_data']} wave_cov={r['wave_cov']:.0%} "
              f"bg={r['n_bg_good']}  -> {r['verdict']}")
    return t


def compare_mask_spectra(
    config: "PipelineConfig", galaxy_ids, masks,
    core_radius=3.5, id_col=None, run_table=None, vdfi=None,
    figsize_per_row=(10, 3.0), line_window=None, plot=True,
):
    """
    Spot-check tool: for a HANDFUL of galaxies, extract the PSF-weighted core
    spectrum once per candidate bad-fiber MASK and overlay them, so you can see
    directly how a masking choice changes the core spectrum. This is the A/B
    instrument for validating the baked segmap against the legacy image mask
    (or one segmap bake against a retweaked one). Meant for a small set of
    galaxies, NOT the whole sample -- extract_core_spectra has no per-galaxy
    cache, so this is only cheap because galaxy_ids is short.

    galaxy_ids : list/array of config.id_col catalog values to test.
    masks      : list of (label, bad_fiber_mask) pairs. Each bad_fiber_mask is
        a (nexp, nfib) boolean (True = bad), e.g. from
        extract.build_bad_fiber_mask under different configs, or
        masking.segmentation_mask on different bake files. One spectrum is
        extracted per mask per galaxy, keyed by label.
    core_radius: aperture (arcsec) fed to the PSF extraction. Fibers beyond it
        get ~zero PSF weight; only mask differences INSIDE this radius can
        change the core spectrum -- masks that agree on every core fiber give
        identical spectra (the silent no-op this tool warns about).
    vdfi       : optional pre-loaded field from load_field_with_psf. Build your
        masks and this comparison from ONE loaded field to avoid re-reading it
        per mask (it is forwarded to every extract_core_spectra call).
    line_window: optional (lo, hi) OBSERVED-frame wavelength window; if given,
        the integrated flux in it is printed per galaxy per mask (a quick
        quantitative read to go with the plot). None -> plot only.
    plot       : if True (default) also draw the figure via
        plotting.plot_mask_spectra and return it. Set False to get results only
        and plot later / re-style without re-running the (slow, uncached)
        extraction:  _, res = compare_mask_spectra(..., plot=False);
        fig = plot_mask_spectra(res, ylim=(-1, 5), zoom_hw_A=20).

    Returns (fig, results); fig is None when plot=False. results = {
        'wave', 'labels', 'spectra' (dict label -> (n_gal, nwave)),
        'errors' (same layout), 'catalog', 'z' (per-galaxy, from config.z_col,
        for rest-frame line placement), 'ids', 'id_col'
    } so you can do your own quantitative comparison beyond the plot.
    """
    from .extract import build_run_table

    idc = id_col or getattr(config, "id_col", "ID")
    full_table = run_table if run_table is not None else build_run_table(config)
    ids = np.asarray(galaxy_ids)
    sel_table = full_table[np.isin(np.asarray(full_table[idc]), ids)]
    if len(sel_table) == 0:
        raise ValueError(f"None of {list(ids)} found in catalog column {idc!r}")
    missing = set(ids.tolist()) - set(np.asarray(sel_table[idc]).tolist())
    if missing:
        print(f"compare_mask_spectra: {len(missing)} id(s) not found "
              f"and skipped: {sorted(missing)}")

    masks = list(masks)
    labels = [str(lab) for lab, _ in masks]
    if len(set(labels)) != len(labels):
        raise ValueError(f"mask labels must be unique, got {labels}")

    if vdfi is None:
        vdfi = load_field_with_psf(config)

    # ---- silent-no-op guard -------------------------------------------------
    # Only mask differences within core_radius of a requested galaxy can move
    # its core spectrum. If a pair of masks flags IDENTICAL core fibers for
    # every requested galaxy, their spectra will be identical -- warn up front
    # so a "no difference" plot is read as agreement, not as a broken tool.
    RA, DEC = vdfi["RA"], vdfi["DEC"]
    near_any = np.zeros(RA.shape, dtype=bool)
    for i in range(len(sel_table)):
        gra, gdec = float(sel_table["RA"][i]), float(sel_table["DEC"][i])
        for e in range(RA.shape[0]):
            dra = (RA[e] - gra) * np.cos(np.deg2rad((DEC[e] + gdec) / 2)) * 3600.
            ddec = (DEC[e] - gdec) * 3600.
            near_any[e] |= (dra ** 2 + ddec ** 2) <= core_radius ** 2
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            la, ma = masks[a]
            lb, mb = masks[b]
            if np.array_equal(np.asarray(ma)[near_any], np.asarray(mb)[near_any]):
                print(f"NOTE: masks {la!r} and {lb!r} flag IDENTICAL fibers within "
                      f"{core_radius:g}\" of every requested galaxy -- their core "
                      f"spectra will be identical (differences, if any, lie outside "
                      f"the aperture).")

    # ---- one extraction per mask ---------------------------------------------
    spectra, errors = {}, {}
    wave = None
    for label, mask in masks:
        Spec, Err, wave, _cat = extract_core_spectra(
            config, run_table=sel_table, core_radius=core_radius,
            ra_col="RA", dec_col="DEC",
            bad_fiber_mask=np.asarray(mask, dtype=bool), vdfi=vdfi,
        )
        spectra[label] = Spec
        errors[label] = Err

    # per-galaxy redshift, so the plotter can place rest-frame UV lines in the
    # observed frame. Pulled from config.z_col; NaN where absent.
    z_col = getattr(config, "z_col", "z")
    if z_col in sel_table.colnames:
        zvals = np.asarray(sel_table[z_col], dtype=float)
    else:
        zvals = np.full(len(sel_table), np.nan)

    results = dict(wave=wave, labels=labels, spectra=spectra, errors=errors,
                   catalog=sel_table, z=zvals, ids=np.asarray(sel_table[idc]),
                   id_col=idc)

    if line_window is not None:
        for i in range(len(sel_table)):
            lo, hi = line_window
            wsel = (wave >= lo) & (wave <= hi)
            fluxes = [np.nansum(spectra[lab][i][wsel]) for lab in labels]
            summary = ", ".join(f"{lab}:{f:.3g}" for lab, f in zip(labels, fluxes))
            print(f"{idc}={sel_table[idc][i]}  integrated flux "
                  f"[{lo:g}-{hi:g} A]  {summary}")

    fig = None
    if plot:
        fig = plot_mask_spectra(results, figsize_per_row=figsize_per_row)
    return fig, results


def load_field_with_psf(config: "PipelineConfig") -> dict:
    """
    load_field(config) but forcing build_psf=True so vdfi['PSF'] is populated.

    extract.load_field hardwires build_psf=False (the binned path never needs the
    Moffat grid). The core path does, so we resolve the base path the same way
    and re-load with the PSF grid on.
    """
    from .extract import load_vdfi_field
    base_path = config.resolve_vdfi_base_path()
    return load_vdfi_field(base_path, field=config.normalized_field(),
                           build_psf=True)


def _exposure_seeing(seeing, e, sel=None):
    """
    Return ONE scalar seeing FWHM for exposure `e`.

    This is the contract optics.get_spectrum_exposure expects: it does
    `wavelength_corrected_seeing(seeing, wave)` -> needs `seeing` to broadcast
    against the (nwave,) grid, then `seeing[None, :]` broadcast across fibers.
    That only works if `seeing` is a SCALAR (one value per exposure, identical
    for every fiber), NOT a per-fiber vector -- passing a (nfib,) vector is what
    triggered the (nfib,) vs (nwave,) broadcast error.

    The VDFI seeing HDU is stored per exposure (and is NOT reshaped to per-fiber
    in load_vdfi_field, unlike RA/DEC). Whatever its trailing shape, we collapse
    exposure e to a single representative (nan-median) value. `sel` is accepted
    and ignored, so the call site can stay uniform.
    """
    seeing = np.asarray(seeing, dtype=float)
    if seeing.ndim == 0:
        return float(seeing)
    row = seeing[e]                       # all values belonging to exposure e
    val = np.nanmedian(row) if np.ndim(row) else float(row)
    if not np.isfinite(val):              # fall back to the global typical seeing
        val = float(np.nanmedian(seeing))
    return float(val)


def _dar_axis_for_exposure(dar, e, nwave):
    """
    Return exposure e's DAR offset as a (nwave,) array, or zeros if unavailable.

    The VDFI DAR HDUs are stored per exposure as a function of wavelength. We
    accept a few plausible shapes and normalize to (nwave,):
      * (nexp, nwave)        -> dar[e]
      * (nwave,)             -> shared across exposures
      * (nexp,)  scalar/exp  -> broadcast that scalar over wavelength
      * None / wrong length  -> zeros (no DAR correction)
    """
    if dar is None:
        return np.zeros(nwave)
    dar = np.asarray(dar, dtype=float)
    if dar.ndim == 2 and dar.shape[0] > e and dar.shape[1] == nwave:
        return dar[e]
    if dar.ndim == 1 and dar.size == nwave:
        return dar
    if dar.ndim == 1 and dar.size > e:
        return np.full(nwave, dar[e])
    if dar.ndim == 2 and dar.shape[0] > e:    # (nexp, k) with k != nwave
        row = dar[e]
        return np.full(nwave, float(np.nanmedian(row)))
    return np.zeros(nwave)


def _offsets_with_dar(dra_1d, ddec_1d, dar_ra, dar_dec, e, nwave):
    """
    Build the 2-D (nfib, nwave) sky offsets get_spectrum_exposure expects.

    dra_1d / ddec_1d are the static per-fiber offsets (arcsec, no wavelength
    dependence). The DAR term is added as a function of wavelength, identical for
    every fiber in the exposure, so the result is:

        dra[fiber, wave]  = dra_1d[fiber]  + dar_ra[e, wave]
        ddec[fiber, wave] = ddec_1d[fiber] + dar_dec[e, wave]
    """
    dra_1d  = np.asarray(dra_1d,  dtype=float)
    ddec_1d = np.asarray(ddec_1d, dtype=float)
    dar_r = _dar_axis_for_exposure(dar_ra,  e, nwave)   # (nwave,)
    dar_d = _dar_axis_for_exposure(dar_dec, e, nwave)   # (nwave,)
    dra_w  = dra_1d[:,  None] + dar_r[None, :]          # (nfib, nwave)
    ddec_w = ddec_1d[:, None] + dar_d[None, :]
    return dra_w, ddec_w


# =====================================================================
# 6b.  FIBER-EXTRACTION GEOMETRY  (compute only)
#      Which fibers feed one galaxy's core extraction, their PSF weights, and
#      aperture membership. The figure that draws this is
#      plotting.plot_fiber_overlay; audit_galaxy_mask_overlap and
#      plotting.plot_fiber_mask_appendix both build on this geometry.
# =====================================================================

def fiber_extraction_geometry(
    vdfi, ra, dec, bad_fiber_mask=None,
    core_radius=3.5, wave_ref=None,
):
    """
    Per-fiber geometry + PSF weight for ONE galaxy -- the data the overlay draws.

    Pools all exposures together (each fiber is one dot at its own dra/ddec) and
    evaluates the SAME Moffat PSF weight the extraction uses, collapsed to a
    single number per fiber by taking the weight at wave_ref (default: the middle
    of the wavelength grid). Fibers are returned whether or not they fall inside
    core_radius, with an `in_aperture` flag, so the plot can show the catchment
    AND the wider field faintly.

    vdfi['PSF'] may be None (e.g. vdfi loaded via extract.load_field, whose
    default build_psf=False skips the PSF grid since Stage 1 doesn't need it)
    -- weight is then NaN for every fiber, but dra/ddec/r/good/in_aperture are
    unaffected, so geometry-only callers (e.g. the mask-overlap audit/appendix
    plot) work without paying for the PSF build.

    Returns dict:
        dra, ddec       (nfib_pooled,)  arcsec offsets from the galaxy
        r               (nfib_pooled,)  radial offset, arcsec
        weight          (nfib_pooled,)  PSF weight at wave_ref (0..~1), NaN if none
        in_aperture     (nfib_pooled,)  bool, r <= core_radius and good
        good            (nfib_pooled,)  bool, passes bad-fiber + finite-coord mask
        exp             (nfib_pooled,)  which exposure each fiber came from
        core_radius, wave_ref, n_in_aperture
    """
    from . import optics
    RA = vdfi["RA"]; DEC = vdfi["DEC"]
    wave = np.asarray(vdfi["wave"], dtype=float)
    PSF  = vdfi["PSF"]
    seeing = np.asarray(vdfi["seeing"], dtype=float)
    nexp, nfib = RA.shape
    if wave_ref is None:
        wave_ref = float(np.nanmedian(wave))

    dra_l, ddec_l, r_l, w_l, good_l, exp_l = [], [], [], [], [], []
    for e in range(nexp):
        dra  = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
        ddec = (DEC[e] - dec) * 3600.
        r    = np.sqrt(dra ** 2 + ddec ** 2)
        good = np.isfinite(RA[e]) & np.isfinite(DEC[e])
        if bad_fiber_mask is not None:
            good = good & (~bad_fiber_mask[e])
        # PSF weight at wave_ref for every (finite) fiber. One scalar seeing per
        # exposure (the same value get_spectrum_exposure uses), corrected to the
        # reference wavelength, then evaluated at each fiber's radius. PSF is
        # None when vdfi was loaded without build_psf=True (e.g. extract.load_field's
        # default) -- geometry/good-fiber info is still valid, weight is just NaN.
        w = np.full(nfib, np.nan)
        finite = np.isfinite(r)
        if PSF is not None and np.any(finite):
            see_e = _exposure_seeing(seeing, e)          # scalar
            see_at_ref = float(see_e) * (wave_ref / 4500.) ** (-1. / 5.)
            see_vec = np.full(int(finite.sum()), see_at_ref)
            w[finite] = PSF(r[finite], see_vec)
        dra_l.append(dra); ddec_l.append(ddec); r_l.append(r)
        w_l.append(w); good_l.append(good)
        exp_l.append(np.full(nfib, e))

    dra  = np.concatenate(dra_l);  ddec = np.concatenate(ddec_l)
    r    = np.concatenate(r_l);    w    = np.concatenate(w_l)
    good = np.concatenate(good_l); exp  = np.concatenate(exp_l)
    in_ap = good & np.isfinite(r) & (r <= core_radius)
    return dict(dra=dra, ddec=ddec, r=r, weight=w, in_aperture=in_ap,
                good=good, exp=exp, core_radius=core_radius,
                wave_ref=wave_ref, n_in_aperture=int(in_ap.sum()))


# =====================================================================
# 6c.  MASK-OVERLAP AUDIT
#      Which galaxies have a 'bad' fiber sitting close to their own center
#      (any mask method). The companion tiled-cutout figure lives in
#      plotting.plot_fiber_mask_appendix.
# =====================================================================

def audit_galaxy_mask_overlap(vdfi, catalog, bad_fiber_mask, radius_arcsec=5.0,
                              ra_col="RA", dec_col="DEC", verbose=True):
    """
    Per-galaxy count of masked fibers near its own center -- the diagnostic
    for "does the bad-fiber mask overlap this galaxy".

    For every row in `catalog`, pools fibers over all exposures (via
    fiber_extraction_geometry) and counts how many are flagged bad in
    `bad_fiber_mask` within radius_arcsec of the galaxy's RA/DEC.

    Run this on the legacy image mask to see how many galaxies the
    self-masking problem actually touches; re-run on the baked segmap mask to
    confirm `flagged` drops out for the galaxies whose segments were unmasked
    at bake time (while genuinely contaminated cores stay flagged).

    Returns an astropy Table, one row per galaxy (catalog order):
        index           row index into `catalog`
        n_bad_near      # masked fibers with r <= radius_arcsec
        n_total_near    # finite-coordinate fibers (masked or not) in that radius
        frac_bad_near   n_bad_near / n_total_near (NaN if n_total_near == 0)
        flagged         n_bad_near > 0 -- the selection plot_fiber_mask_appendix uses
    """
    ra = np.asarray(catalog[ra_col], dtype=float)
    dec = np.asarray(catalog[dec_col], dtype=float)
    ngal = len(ra)
    n_bad_near = np.zeros(ngal, dtype=int)
    n_total_near = np.zeros(ngal, dtype=int)

    for i in range(ngal):
        if not (np.isfinite(ra[i]) and np.isfinite(dec[i])):
            continue
        geom = fiber_extraction_geometry(vdfi, float(ra[i]), float(dec[i]),
                                         bad_fiber_mask=bad_fiber_mask,
                                         core_radius=radius_arcsec)
        dra, ddec, r, good = geom["dra"], geom["ddec"], geom["r"], geom["good"]
        finite = np.isfinite(dra) & np.isfinite(ddec) & (r <= radius_arcsec)
        n_total_near[i] = int(np.sum(finite))
        n_bad_near[i] = int(np.sum(finite & ~good))

    frac = np.full(ngal, np.nan)
    nz = n_total_near > 0
    frac[nz] = n_bad_near[nz] / n_total_near[nz]
    flagged = n_bad_near > 0

    t = Table({"index": np.arange(ngal), "n_bad_near": n_bad_near,
              "n_total_near": n_total_near, "frac_bad_near": frac,
              "flagged": flagged})
    if verbose:
        print(f"audit_galaxy_mask_overlap: {int(flagged.sum())}/{ngal} galaxies have "
              f"a masked fiber within {radius_arcsec:g}\"")
    return t


# =====================================================================
# 8.  CORE SELECTION BRIDGE
#     Pull individual galaxies' core spectra + measurements out of a core
#     product, by row index, catalog ID, or RA/DEC cone match. Mirrors
#     selection.py (which works on the binned GalaxyProduct), but the core
#     product is one spectrum per galaxy -- (ngal, nwave), no radial axis --
#     so it gets its own lightweight result types here.
#
#     A "core" here is the dict returned by read_core_fits (or the live dict
#     you assemble from extract_core_spectra + run_core); it must have at least
#     core_spec (ngal, nwave), wave_obs (nwave,), catalog (Table). core_err and
#     results are used when present.
# =====================================================================

from dataclasses import dataclass as _dataclass, field as _dc_field
from typing import Any as _Any, Dict as _Dict, List as _List


def _haversine_arcsec(ra0, dec0, ra, dec):
    """Great-circle separation (arcsec) from one center to many points (deg in)."""
    r0, d0 = np.radians(ra0), np.radians(dec0)
    r, d   = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    a = np.sin((d - d0) / 2) ** 2 + np.cos(d0) * np.cos(d) * np.sin((r - r0) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0


def _core_ngal(core):
    return int(np.asarray(core["core_spec"]).shape[0])


def _result_row(results, i):
    """Per-galaxy measurement row as a plain dict (skips scalar metadata keys)."""
    if not results:
        return {}
    out = {}
    for k, v in results.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == len(np.asarray(results.get("flux", []))):
            out[k] = arr[i]
    return out


@_dataclass
class CoreSpectrum:
    """One galaxy's core spectrum + its catalog row + its measurement row.

    Fields
    ------
    index    : row position in the parent core product
    wave     : (nwave,) observed wavelength grid (shared)
    flux     : (nwave,) this galaxy's core spectrum
    err      : (nwave,) its error spectrum (None if the product had none)
    row      : its catalog row (astropy Row)
    meas     : dict of its measurements (flux, flux_err, sn, label, fesc, ...) if present
    """
    index: int
    wave: np.ndarray
    flux: np.ndarray
    err: "np.ndarray | None"
    row: _Any
    meas: _Dict[str, _Any] = _dc_field(default_factory=dict)

    def __repr__(self):
        lab = self.meas.get("label", "")
        sn  = self.meas.get("sn", np.nan)
        extra = f", {lab} (S/N={sn:.1f})" if lab else ""
        return f"<CoreSpectrum #{self.index}{extra}>"

    def plot(self, ax=None, rest_frame=False, z=None, show_err=True, **kw):
        """Quick-look plot of this galaxy's core spectrum.

        rest_frame=True shifts to rest frame using z (defaults to the catalog
        z column value on this row if present under 'z').
        """
        import matplotlib.pyplot as plt
        w, f, e = self.wave, self.flux, self.err
        if rest_frame:
            zz = z if z is not None else float(self.row["z"])
            w, f, e = to_rest_frame(w, f, e, z=zz)
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.plot(w, f, lw=1.0, **kw)
        if show_err and e is not None:
            ax.fill_between(w, f - e, f + e, alpha=0.25, lw=0)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xlabel("rest wavelength (Å)" if rest_frame else "observed wavelength (Å)")
        ax.set_ylabel("flux")
        ax.set_title(repr(self))
        return ax


@_dataclass
class CoreSelection:
    """A subset of galaxies pulled from a core product, in selection order.

    Fields
    ------
    index      : int indices into the parent (selection order)
    wave       : (nwave,) shared wavelength grid
    spec       : (nsel, nwave) selected core spectra
    err        : (nsel, nwave) selected errors (None if absent)
    catalog    : the selected catalog rows (Table), aligned to `index`
    results    : dict of selected per-galaxy measurement arrays (aligned to `index`)
    parent_ngal: ngal of the product selected from
    sep_arcsec : separations (cone search only; else None)
    info       : short human description
    """
    index: np.ndarray
    wave: np.ndarray
    spec: np.ndarray
    err: "np.ndarray | None"
    catalog: _Any
    results: _Dict[str, _Any]
    parent_ngal: int
    sep_arcsec: "np.ndarray | None" = None
    info: str = ""

    @property
    def n(self):
        return int(len(self.index))

    def __len__(self):
        return self.n

    def __repr__(self):
        extra = ""
        if self.sep_arcsec is not None and self.n:
            extra = f", sep {self.sep_arcsec.min():.1f}-{self.sep_arcsec.max():.1f}\""
        head = f" [{self.info}]" if self.info else ""
        return f"<CoreSelection {self.n}/{self.parent_ngal} galaxies{extra}>{head}"

    def __getitem__(self, k):
        """Integer k -> the k-th CoreSpectrum in this selection."""
        i = int(self.index[k])
        e = None if self.err is None else self.err[k]
        return CoreSpectrum(index=i, wave=self.wave, flux=self.spec[k], err=e,
                            row=self.catalog[k], meas=_result_row(self.results, k))

    def __iter__(self):
        for k in range(self.n):
            yield self[k]

    def to_core(self):
        """Repackage this selection as a standalone core dict (read_core_fits-shaped),
        so it can be re-measured, written, or fed back through the same helpers."""
        return dict(core_spec=self.spec, core_err=self.err, wave_obs=self.wave,
                    catalog=self.catalog, results=self.results, header=None)


def _make_core_selection(core, idx, *, sep=None, info=""):
    idx = np.asarray(idx, dtype=int)
    spec = np.asarray(core["core_spec"])[idx]
    err  = (np.asarray(core["core_err"])[idx]
            if core.get("core_err") is not None else None)
    cat  = core["catalog"][idx]
    if sep is not None:
        cat = cat.copy()
        cat["sep_arcsec"] = np.asarray(sep, float)
    res  = core.get("results") or {}
    flux_len = len(np.asarray(res.get("flux", [])))
    res_sel = {}
    for k, v in res.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == flux_len and flux_len == _core_ngal(core):
            res_sel[k] = arr[idx]
        else:
            res_sel[k] = v          # scalar metadata (method, n_sigma, ...) pass through
    return CoreSelection(index=idx, wave=np.asarray(core["wave_obs"]),
                         spec=spec, err=err, catalog=cat, results=res_sel,
                         parent_ngal=_core_ngal(core), sep_arcsec=sep, info=info)


# ---------------------------------------------------------------------
# 8.1  select by ROW INDEX
# ---------------------------------------------------------------------
def select_core_by_index(core, index, *, single=None):
    """Pull galaxies by row index/indices.

    index : an int (-> one CoreSpectrum) or a sequence of ints (-> CoreSelection).
            Negative indices work (Python convention). Order is preserved.
    single: force the return type. None (default) auto-detects: scalar int ->
            CoreSpectrum, sequence -> CoreSelection.
    """
    scalar = np.isscalar(index) and not isinstance(index, (list, tuple, np.ndarray))
    if single is None:
        single = scalar
    ng = _core_ngal(core)
    idx = np.atleast_1d(np.asarray(index, dtype=int))
    idx = np.where(idx < 0, idx + ng, idx)          # resolve negatives
    if np.any((idx < 0) | (idx >= ng)):
        bad = idx[(idx < 0) | (idx >= ng)]
        raise IndexError(f"index out of range for ngal={ng}: {bad.tolist()}")
    if single:
        i = int(idx[0])
        e = (np.asarray(core["core_err"])[i]
             if core.get("core_err") is not None else None)
        return CoreSpectrum(index=i, wave=np.asarray(core["wave_obs"]),
                            flux=np.asarray(core["core_spec"])[i], err=e,
                            row=core["catalog"][i],
                            meas=_result_row(core.get("results") or {}, i))
    return _make_core_selection(core, idx, info=f"index[{len(idx)}]")


# ---------------------------------------------------------------------
# 8.2  select by CATALOG ID column
# ---------------------------------------------------------------------
def select_core_by_id(core, id_value, *, id_col="ID", single=None, verbose=True):
    """Pull galaxies whose catalog `id_col` matches `id_value`.

    id_value : a single id (-> CoreSpectrum if exactly one match) or a sequence
               of ids (-> CoreSelection, in the order requested). Matching is
               done on the catalog column as strings, so 123 and "123" both work.
    id_col   : the catalog column holding the identifier (e.g. 'ID', 'MOSDEF_ID',
               'v4id'). Raises if the column is absent.
    """
    cat = core["catalog"]
    if id_col not in cat.colnames:
        raise KeyError(f"id_col {id_col!r} not in catalog columns {list(cat.colnames)}")
    col = np.asarray(cat[id_col]).astype(str)
    want = ([str(id_value)]
            if (np.isscalar(id_value) and not isinstance(id_value, (list, tuple, np.ndarray)))
            else [str(v) for v in np.asarray(id_value).ravel()])
    idx = []
    for w in want:
        hits = np.flatnonzero(col == w)
        if hits.size == 0:
            if verbose:
                print(f"select_core_by_id: no match for {id_col}={w!r}")
            continue
        if hits.size > 1 and verbose:
            print(f"select_core_by_id: {hits.size} matches for {id_col}={w!r}; "
                  f"taking all (rows {hits.tolist()})")
        idx.extend(hits.tolist())
    idx = np.asarray(idx, dtype=int)
    if single is None:
        single = (len(want) == 1 and idx.size == 1)
    if single:
        if idx.size == 0:
            raise KeyError(f"no galaxy with {id_col}={id_value!r}")
        return select_core_by_index(core, int(idx[0]), single=True)
    return _make_core_selection(core, idx, info=f"{id_col} in {want}")


# ---------------------------------------------------------------------
# 8.3  select by RA/DEC CONE
# ---------------------------------------------------------------------
def select_core_cone(core, ra, dec, radius_arcsec, *, ra_col="RA", dec_col="DEC",
                     sort=True, single=False, verbose=True):
    """Pull galaxies within `radius_arcsec` of (ra, dec) [deg, arcsec].

    Returns a CoreSelection with a `sep_arcsec` column (nearest first if sort).
    single=True returns just the nearest match as a CoreSpectrum (or raises if
    nothing is within the radius). Non-finite coordinates never match.
    """
    cat = core["catalog"]
    cra  = np.asarray(cat[ra_col],  float)
    cdec = np.asarray(cat[dec_col], float)
    sep  = _haversine_arcsec(ra, dec, cra, cdec)
    mask = np.isfinite(sep) & (sep <= float(radius_arcsec))
    idx  = np.flatnonzero(mask)
    if sort:
        idx = idx[np.argsort(sep[idx])]
    if verbose:
        print(f"select_core_cone: {idx.size} galaxies within {radius_arcsec:g}\" "
              f"of ({ra:.5f}, {dec:.5f})")
    if single:
        if idx.size == 0:
            raise ValueError(f"no galaxy within {radius_arcsec:g}\" of ({ra}, {dec})")
        return select_core_by_index(core, int(idx[0]), single=True)
    return _make_core_selection(core, idx, sep=sep[idx],
                                info=f"cone r<={radius_arcsec:g}\" @({ra:.4f},{dec:.4f})")


# ---------------------------------------------------------------------
# 8.4  unified dispatcher
# ---------------------------------------------------------------------
def select_core(core, *, index=None, id=None, id_col="ID",
                ra=None, dec=None, radius_arcsec=None, **kw):
    """One entry point for all three keys -- pick exactly one.

        select_core(core, index=12)                       # row index
        select_core(core, index=[1, 5, 9])                # several rows
        select_core(core, id="COSMOS-12345", id_col="ID") # catalog ID
        select_core(core, ra=150.12, dec=2.21, radius_arcsec=2.0)  # cone

    Returns a CoreSpectrum (single match) or CoreSelection (several), matching
    the underlying helper.
    """
    given = [("index", index is not None), ("id", id is not None),
             ("cone", ra is not None or dec is not None or radius_arcsec is not None)]
    picked = [name for name, on in given if on]
    if len(picked) != 1:
        raise ValueError("select_core: pass exactly one of index=, id=, "
                         "or (ra=, dec=, radius_arcsec=); got "
                         f"{picked or 'nothing'}.")
    if index is not None:
        return select_core_by_index(core, index, **kw)
    if id is not None:
        return select_core_by_id(core, id, id_col=id_col, **kw)
    if radius_arcsec is None or ra is None or dec is None:
        raise ValueError("cone search needs ra=, dec=, AND radius_arcsec=.")
    return select_core_cone(core, ra, dec, radius_arcsec, **kw)


# ---------------------------------------------------------------------
# 8.5  measurements -> table  (the mask-building path)
# ---------------------------------------------------------------------
def core_results_table(core, *, include_catalog=True, id_col=None):
    """Per-galaxy measurements as an astropy Table -- the currency for masks.

    One row per galaxy (parent order). Columns: the measurement arrays from
    run_core (FLUX, FLUX_ERR, SN, CENTER_A, SUCCESS, LABEL, FESC, ...), plus the
    row index. With include_catalog, the catalog columns ride along so you can
    build masks on BOTH measurements and catalog properties in one place, e.g.

        t = core_results_table(core)
        emitters = t["LABEL"] == "emitter"
        bright   = emitters & (t["SN"] >= 5) & (t["MASS_50"] > 10)
        sel      = select_core_by_index(core, np.flatnonzero(bright))

    id_col, if given, is pulled to the front for readability.
    """
    ng = _core_ngal(core)
    res = core.get("results") or {}
    cols = {"index": np.arange(ng)}
    flux_len = len(np.asarray(res.get("flux", [])))
    for k, v in res.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == ng:
            cols[k.upper()] = arr
    t = Table(cols)
    if include_catalog:
        cat = core["catalog"]
        for name in cat.colnames:
            if name.upper() not in t.colnames:
                t[name] = np.asarray(cat[name])
    if id_col and id_col in t.colnames:
        t = t[[id_col] + [c for c in t.colnames if c != id_col]]
    return t


def core_mask(core, criteria, *, verbose=True):
    """Boolean (length ngal) mask from a criterion over the results+catalog table.

    `criteria` is a callable t -> bool array, or a string expression over the
    table's column names plus np, e.g. "(LABEL == 'emitter') & (SN >= 5)".
    Hand the result straight to select_core_by_index(core, np.flatnonzero(mask)),
    or use it to build a special sample for stacking.
    """
    t = core_results_table(core)
    # expose each column under its own name AND lower/upper aliases, so an
    # expression can say `detected` or `DETECTED`, `v_offset` or `V_OFFSET`.
    ns = {"np": np}
    for name in t.colnames:
        col = np.asarray(t[name])
        ns[name] = col
        ns[name.lower()] = col
        ns[name.upper()] = col
    if callable(criteria):
        mask = np.asarray(criteria(t))
    else:
        mask = np.asarray(eval(criteria, {"__builtins__": {}}, ns))   # noqa: S307
    mask = mask.astype(bool)
    if verbose:
        print(f"core_mask: {int(mask.sum())}/{len(mask)} galaxies"
              + (f"  [{criteria}]" if isinstance(criteria, str) else ""))
    return mask
