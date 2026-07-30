"""
lis.py -- LIS / rest-UV line velocities from a rest-frame core stack.

WHAT THIS IS FOR
    Two independent jobs, one codebase (spec: ../specs/lis-velocity-calibration.md):

    1. CALIBRATION. Measure Delta_v of the low-ionization interstellar (LIS)
       absorption lines against the MOSDEF systemic frame and compare to the
       MOSDEF-LRIS result, Weldon et al. 2022: Delta_v_LIS = -60 +/- 10 km/s
       (149 galaxies, all with genuine MOSFIRE nebular z_sys). Agreement says the
       VIRUS wavelength solution is consistent with LRIS's -- which was itself
       sky-line zeropointed (Topping et al. 2020 sec 2.2) and is stated to be on
       vacuum wavelengths (Weldon sec 1). Disagreement at the ~85 km/s level is
       the air/vacuum signature the parent spec is hunting.
    2. PHYSICS. The same numbers are the sample's outflow velocity, i.e. split
       #15 in ../docs/subsample_splits.md, plus the Si II / Si II* fine-structure
       pairs.

    This module is the MEASUREMENT LAYER ONLY. It does not build stacks and it
    does not slice sub-samples. Feed it the output of
    stack.build_core_stacks (already rest-framed) or bare (wave, flux, err).

WHY THE VELOCITY FORMULA IS EXACTLY WELDON'S EQ. 4
    build_core_stacks shifts each galaxy by its catalog z (= z_sys) before
    coadding, so the stack's abscissa is rest-frame Angstrom in the SYSTEMIC
    frame. A line whose observed wavelength is lam_obs lands at
    lam_meas = lam_obs / (1 + z_sys). Then, with lam_0 the rest wavelength,

        z_LIS = lam_obs / lam_0 - 1
        Delta_v = c (z_LIS - z_sys) / (1 + z_sys)          <- Weldon Eq. 4
                = c (lam_obs / lam_0 - 1 - z_sys) / (1 + z_sys)
                = c (lam_obs / ((1 + z_sys) lam_0) - 1)
                = c (lam_meas - lam_0) / lam_0             <- what we compute

    The identity is exact, not first-order. So measuring a rest-frame line
    position on the stack IS measuring Weldon's Delta_v, with no extra step.

TWO ESTIMATORS, AND THEY ARE NOT THE SAME NUMBER
    Weldon use DIFFERENT estimators for individual galaxies and for composites,
    which is the single easiest way to get this comparison wrong:

      per-object (sec 2.3)  Gaussian centroid + quadratic local continuum,
                            averaged over the good lines ("typically two")
                            -> this is what produces the -60 +/- 10
      composite (sec 3.5)   "the wavelength at the absolute minimum of the
                            detected LIS absorption trough"

    On a blue-winged asymmetric trough the minimum sits REDWARD of the Gaussian
    centroid. This module returns BOTH for every line, always, so the comparison
    can be made like-with-like:

      vel_gauss_kms  -> compare to Weldon's -60 +/- 10 (mean of per-object
                        Gaussian centroids; a composite's Gaussian centroid
                        equals the mean of individual ones, because the galaxy
                        velocity distribution enters as a convolution and moves
                        no centroid)
      vel_min_kms    -> compare to Weldon's composite points (their Fig. 2
                        green squares)

    The difference between them on one stack is an asymmetry measurement, free.

CONTINUUM WINDOWS
    Every window here is Weldon Table 1 verbatim ("chosen to bracket the line and
    free of other spectral features"). NO window in this module is invented --
    the fine-structure lines are declared as PARTNERS of their resonance line and
    share its windows, which also avoids the fact that Si II* 1264.74 and
    Si II* 1533.43 have no clean blue sideband of their own (the resonance trough
    is 4-7 A blueward of each).

RESOLUTION NOTE (matters for the blends)
    LRIS-B used the 400 l/mm grism through 1"2 slits -> FWHM ~7 A ~ 485-515 km/s
    at these lines. VIRUS is 5.6-6.3 A ~ 340-460 km/s. VIRUS is the SHARPER
    instrument by 10-20%, so every blend is more blended for LRIS than for us:
    their -60 already carries the C II* fill-in bias, and more of it. The
    differential comparison therefore partially cancels it. Use ncomp>1 to
    measure the residual rather than assume it.

    5.6 A is speclsf.VIRUS_FWHM_A. Austin's stacked rest-frame Lya FWHM of
    ~1.9 A brackets the true value at 5.6-6.27 A observed and excludes the
    Hill+2021 4.7 A. The LSF is UNMEASURED at 4718-5511 A where Si II 1526 lands.

PUBLIC API
    LINES, line_spec, science_lines
    normalize_local          -- Weldon-linear or polynomial local continuum
    measure_extremum         -- trough minimum / peak maximum, parabola-refined
    fit_line_profile         -- simultaneous poly continuum + N Gaussians, one
                                shared velocity, sign-constrained amplitudes
    measure_line             -- both estimators + Topping's 100x error-spectrum MC
    measure_lines            -- loop a line table
    measure_lis_stack        -- convenience wrapper over build_core_stacks output
    z_lis_average            -- Topping's z_LIS: unweighted mean of the good lines
    describe_lis             -- printer, with the Weldon comparison inline
    as_line_result           -- coerce any of the three result shapes to one line
    plot_lis_fit             -- single-line diagnostic overlay (line=... to pick)
    plot_lis_grid            -- all lines at once
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dcfield
from typing import Sequence

import numpy as np

C_KMS = 299792.458

try:                                     # optional, only for the LSF prior
    from .speclsf import instrumental_fwhm, VIRUS_FWHM_A
except Exception:                        # pragma: no cover
    VIRUS_FWHM_A = 5.6

    def instrumental_fwhm(wave_obs, mode="fwhm", fwhm_A=VIRUS_FWHM_A, **kw):
        return np.full_like(np.asarray(wave_obs, float), float(fwhm_A))


# =====================================================================
# 1.  THE LINE TABLE
#     windows: Weldon et al. 2022 Table 1, rest-frame vacuum (NIST)
#     sign: -1 absorption, +1 emission
#     partners: blended companions sharing the parent's window and velocity
# =====================================================================

@dataclass(frozen=True)
class Component:
    """One Gaussian component of a (possibly blended) line complex."""
    name: str
    lam_rest: float
    sign: int                     # -1 absorption, +1 emission


@dataclass(frozen=True)
class LineSpec:
    name: str
    lam_rest: float
    sign: int
    blue_window: tuple            # (lo, hi) rest A
    red_window: tuple
    partners: tuple = ()          # extra Components, fixed separation from lam_rest
    windows_source: str = "Weldon+2022 Table 1"
    note: str = ""
    ncomp_recommended: int = 1     # components needed for an UNBIASED velocity;
                                   # 1 is always the Topping/Weldon-matching value

    @property
    def components(self) -> tuple:
        """Main component first, then partners. Order fixes the amplitude order."""
        return (Component(self.name, self.lam_rest, self.sign),) + tuple(self.partners)

    @property
    def fit_span(self) -> tuple:
        """Full wavelength span used for the fit / normalization."""
        return (float(self.blue_window[0]), float(self.red_window[1]))

    @property
    def cont_bounds(self) -> tuple:
        return (tuple(self.blue_window), tuple(self.red_window))


#: Weldon Table 1, plus fine-structure partners. No invented windows.
LINES: dict = {
    "Lya": LineSpec(
        "Lya", 1215.67, +1, (1195, 1202), (1225, 1235),
        note="Resonant, aperture-sensitive. Weldon measured dv_Lya = +400+/-23 "
             "km/s (72 gal) through 1\"2 slits; a fiber IFU core captures far "
             "more extended blueshifted halo light, so this is NOT directly "
             "comparable to slit values. Diagnostic only."),
    "SiII1260": LineSpec(
        "SiII1260", 1260.42, -1, (1245, 1252), (1270, 1275),
        partners=(Component("SII1259", 1259.52, -1),
                  Component("SiII*1264", 1264.74, +1)),
        ncomp_recommended=3,
        note="SCIENCE LINE (Topping's 3 best). Lands 3895-4550 A at z=2.09-2.61 "
             "-- essentially on top of where z~2 Lya falls (3756-4389 A), so "
             "this is the line whose calibration applies at Lya's wavelength. "
             "S II 1259.52 is 214 km/s away (~0.5 VIRUS FWHM): blended. "
             "Si II* 1264.74 is +1028 km/s (~2.6 FWHM): resolved, and it sits "
             "between the trough and Weldon's red window, which is therefore "
             "clean."),
    "OI1302": LineSpec(
        "OI1302", 1302.17, -1, (1285, 1293), (1312, 1318),
        partners=(Component("SiII1304", 1304.37, -1),
                  Component("SiII*1309", 1309.28, +1)),
        ncomp_recommended=3,
        note="Weldon's fallback line (6 objects). O I 1302 + Si II 1304 are "
             "506 km/s apart (~1.3 VIRUS FWHM): marginally blended. Use ncomp=2 "
             "or 3; a single Gaussian here is a blend centroid, not a line."),
    "CII1334": LineSpec(
        "CII1334", 1334.53, -1, (1320, 1330), (1342, 1351),
        partners=(Component("CII*1335", 1335.71, +1),),
        ncomp_recommended=2,
        note="SCIENCE LINE (Topping's 3 best). C II* 1335.71 is fine-structure "
             "EMISSION only 265 km/s redward (~0.65 VIRUS FWHM, ~0.55 LRIS "
             "FWHM): it fills in the red side of the trough and biases a "
             "single-Gaussian centroid BLUEWARD -- the same direction as the "
             "artifact being hunted. Always run ncomp=2 alongside ncomp=1 here."),
    "SiII1526": LineSpec(
        "SiII1526", 1526.71, -1, (1512, 1520), (1535, 1540),
        partners=(Component("SiII*1533", 1533.43, +1),),
        ncomp_recommended=2,
        note="SCIENCE LINE (Topping's 3 best) and the CLEANEST at VIRUS "
             "resolution: Si II* 1533.43 is +1320 km/s away (~3.9 FWHM), fully "
             "resolved. Caveat: Weldon's red window starts at 1535 A and "
             "Si II* reaches ~1534.97 at 2 sigma under a 6 A LSF, so the "
             "Weldon-linear continuum is mildly tilted up on the red side "
             "(biases the trough REDWARD, i.e. against a spurious blueshift). "
             "The simultaneous fit with ncomp=2 handles this properly. "
             "Lands 4718-5511 A -- ~1000 A redward of where z~2 Lya falls."),
    "FeII1608": LineSpec(
        "FeII1608", 1608.45, -1, (1590, 1600), (1616, 1620),
        note="Clean singlet, no fine structure from the ground term. But only "
             "in band to z=2.444 (68% of the z~2 bin), so the sub-sample "
             "differs from the other lines -- do not average it in blindly."),
}

#: Topping's three best redshift lines, "not contaminated by nearby features
#: (Shapley et al. 2003)". That designation was made at LOWER resolution than
#: VIRUS -- see the module docstring.
SCIENCE_LINES = ("SiII1260", "CII1334", "SiII1526")

#: Lines whose reported velocity should be treated as a blend centroid unless
#: ncomp > 1 is used.
BLENDED_AT_VIRUS = ("SiII1260", "OI1302", "CII1334")

#: Components needed for an UNBIASED velocity, from injection-recovery
#: (tests/test_lis.py). Pass ncomp="recommended" to measure_lines to use these.
#: NOTE ncomp=1 is what Topping/Weldon actually did, so the ncomp=1 number is
#: the estimator-matched one and the recommended number is the unbiased one.
#: Report BOTH -- the difference is the blend systematic, not a correction to
#: apply silently.
NCOMP_RECOMMENDED = {k: v.ncomp_recommended for k, v in LINES.items()}


def line_spec(line) -> LineSpec:
    """Accept a LineSpec, or a key into LINES."""
    if isinstance(line, LineSpec):
        return line
    try:
        return LINES[line]
    except KeyError:
        raise KeyError(f"unknown line {line!r}; known: {sorted(LINES)}") from None


def science_lines() -> tuple:
    """The three lines to actually trust, as LineSpecs."""
    return tuple(LINES[k] for k in SCIENCE_LINES)


# =====================================================================
# 2.  LOCAL CONTINUUM / NORMALIZATION
# =====================================================================

def normalize_local(wave, flux, err=None, line="SiII1526",
                    mode="weldon_linear", cont_order=2, sigma_clip=3.0,
                    niter=3):
    """
    Local continuum over the line's fit span, and the normalized spectrum.

    mode
      'weldon_linear' -- Weldon sec 2.2 verbatim: "a linear model between the
                         average flux in two spectral windows, bluewards and
                         redwards of the LIS absorption line". A straight line
                         through the two window means. This is THE recipe to
                         match; it is not robust to a contaminated window.
      'poly'          -- robust sigma-clipped polynomial of order `cont_order`
                         fit to the window pixels only (delegates to the same
                         logic as measure.fit_local_poly_continuum). Use as the
                         sensitivity test.

    Returns dict: wave (span-restricted), flux, err, cont, nflux, nerr, mode.
    Normalized so that continuum == 1; nflux < 1 is absorption.
    """
    spec = line_spec(line)
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = None if err is None else np.asarray(err, float)

    lo, hi = spec.fit_span
    span = (wave >= lo) & (wave <= hi)
    w = wave[span]
    f = flux[span]
    e = None if err is None else err[span]

    bw, rw = spec.cont_bounds
    mblue = (w >= bw[0]) & (w <= bw[1]) & np.isfinite(f)
    mred = (w >= rw[0]) & (w <= rw[1]) & np.isfinite(f)

    base = dict(wave=w, flux=f, err=e, mode=mode, line=spec.name,
                n_blue=int(mblue.sum()), n_red=int(mred.sum()))

    if mblue.sum() < 2 or mred.sum() < 2:
        nan = np.full_like(w, np.nan)
        return {**base, "cont": nan, "nflux": nan, "nerr": nan,
                "success": False, "reason": "empty continuum window"}

    if mode == "weldon_linear":
        xb, yb = np.nanmean(w[mblue]), np.nanmean(f[mblue])
        xr, yr = np.nanmean(w[mred]), np.nanmean(f[mred])
        slope = (yr - yb) / (xr - xb) if xr != xb else 0.0
        cont = yb + slope * (w - xb)
    elif mode == "poly":
        cmask = mblue | mred
        x0 = np.nanmedian(w[cmask])
        xx, yy = w[cmask] - x0, f[cmask]
        good = np.ones(xx.size, bool)
        order = int(min(cont_order, max(1, xx.size - 2)))
        for _ in range(int(niter)):
            coeff = np.polyfit(xx[good], yy[good], order)
            resid = yy - np.polyval(coeff, xx)
            sig = 1.4826 * np.nanmedian(np.abs(resid[good] - np.nanmedian(resid[good])))
            if not np.isfinite(sig) or sig == 0:
                break
            good = np.abs(resid) < sigma_clip * sig
            if good.sum() < order + 2:
                good = np.ones(xx.size, bool)
                break
        coeff = np.polyfit(xx[good], yy[good], order)
        cont = np.polyval(coeff, w - x0)
    else:
        raise ValueError("mode must be 'weldon_linear' or 'poly'")

    with np.errstate(invalid="ignore", divide="ignore"):
        nflux = f / cont
        nerr = None if e is None else e / np.abs(cont)
    bad = ~np.isfinite(cont) | (cont == 0)
    nflux = np.where(bad, np.nan, nflux)
    if nerr is not None:
        nerr = np.where(bad, np.nan, nerr)

    return {**base, "cont": cont, "nflux": nflux, "nerr": nerr,
            "success": True, "reason": None}


# =====================================================================
# 3.  ESTIMATOR A -- the trough minimum (Weldon's COMPOSITE estimator)
# =====================================================================

def _parabolic_shift(y_lo, y_mid, y_hi, dx):
    """Sub-pixel offset of a parabola's vertex through three equally spaced
    points. Sign-agnostic: works for a minimum and a maximum. Returns 0.0 if
    the refinement is degenerate or would leave the central pixel."""
    denom = y_lo - 2.0 * y_mid + y_hi
    if not np.isfinite(denom) or denom == 0.0:
        return 0.0
    delta = 0.5 * (y_lo - y_hi) / denom
    if not np.isfinite(delta) or abs(delta) > 1.0:
        return 0.0
    return float(delta * dx)


def measure_extremum(wave, nflux, line="SiII1526", search_halfwidth_A=4.0,
                     refine=True):
    """
    Weldon sec 3.5's composite estimator: "the wavelength at the absolute
    minimum of the detected LIS absorption trough" (or the maximum, for
    emission), searched within +/- search_halfwidth_A of the rest wavelength.

    The raw argmin is a single-pixel statistic. On the default rest grid
    (config.rest_delta = 0.2 A) one pixel is ~49 km/s at 1215 A and ~40 km/s at
    1526 A -- comparable to the whole effect being measured -- so a 3-point
    parabolic refinement is applied by default and the pixel scale is reported
    so the quantization stays visible. Weldon's own error-spectrum Monte Carlo
    smooths this too; measure_line does both.
    """
    spec = line_spec(line)
    wave = np.asarray(wave, float)
    nflux = np.asarray(nflux, float)

    dx = float(np.nanmedian(np.diff(wave))) if wave.size > 1 else np.nan
    pix_kms = C_KMS * dx / spec.lam_rest if np.isfinite(dx) else np.nan

    m = (np.abs(wave - spec.lam_rest) <= float(search_halfwidth_A)) & np.isfinite(nflux)
    out = dict(line=spec.name, lam_A=np.nan, vel_kms=np.nan, refined=False,
               pix_scale_A=dx, pix_scale_kms=pix_kms, depth=np.nan,
               success=False, reason=None)
    if m.sum() < 3:
        out["reason"] = "fewer than 3 finite pixels in the search window"
        return out

    idx_local = np.nanargmin(nflux[m]) if spec.sign < 0 else np.nanargmax(nflux[m])
    idx = np.flatnonzero(m)[idx_local]

    lam = float(wave[idx])
    refined = False
    if refine and 0 < idx < wave.size - 1:
        y_lo, y_mid, y_hi = nflux[idx - 1], nflux[idx], nflux[idx + 1]
        if np.all(np.isfinite([y_lo, y_mid, y_hi])):
            shift = _parabolic_shift(y_lo, y_mid, y_hi, dx)
            if shift != 0.0:
                lam += shift
                refined = True

    out.update(lam_A=lam, vel_kms=C_KMS * (lam - spec.lam_rest) / spec.lam_rest,
               refined=refined, depth=float(1.0 - nflux[idx]),
               success=True, idx=int(idx))
    return out


# =====================================================================
# 4.  ESTIMATOR B -- simultaneous continuum + Gaussian(s)
#     Topping sec 2.3 / Weldon sec 2.3: "fitting lines with Gaussian functions
#     and a quadratic function for the local continuum and calculating the
#     centroids of the Gaussians"
# =====================================================================

def _lsf_sigma_rest_A(spec: LineSpec, z, lsf_mode="fwhm", lsf_fwhm_A=None):
    """Instrumental sigma in REST Angstrom at this line, for a stack at z."""
    if z is None or not np.isfinite(z):
        return None
    lam_obs = spec.lam_rest * (1.0 + float(z))
    kw = {} if lsf_fwhm_A is None else {"fwhm_A": float(lsf_fwhm_A)}
    fwhm_obs = float(np.atleast_1d(instrumental_fwhm(np.array([lam_obs]),
                                                     mode=lsf_mode, **kw))[0])
    return (fwhm_obs / (1.0 + float(z))) / 2.3548200450309493


def fit_line_profile(wave, flux, err=None, line="SiII1526", ncomp=1,
                     cont_order=2, z=None, lsf_fwhm_A=None, lsf_mode="fwhm",
                     dv_bounds_kms=(-1200.0, 1200.0), sigma_bounds_A=None,
                     p0_dv_kms=0.0, min_pixels=10, simultaneous=True,
                     cont_mode="weldon_linear"):
    """
    Fit polynomial continuum + `ncomp` Gaussians SIMULTANEOUSLY, sharing one
    velocity and one width, with amplitudes sign-constrained per component
    (Topping constrain the amplitude <= 0 for absorption, >= 0 for emission).

    Parameters
    ----------
    ncomp : 1 fits the main component only -- the literal Topping/Weldon
        estimator, and the default, so the number stays comparable to -60.
        ncomp > 1 adds the spec's partners in order, at FIXED rest separation
        with free amplitude. Use it to measure how far the blend moves the
        answer (notably C II* 1335.71 for CII1334) instead of assuming it.

        Injection-recovery on synthetic stacks (tests/test_lis.py) shows how
        much this matters at plausible partner strengths:
            CII1334  ncomp=1 -> -83 km/s bias   ncomp=2 -> exact
            SiII1260 ncomp=1 -> -42 km/s bias   ncomp=2 -> -5    ncomp=3 -> exact
            SiII1526 ncomp=1 -> +3 km/s         ncomp=2 -> exact
        Each spec carries `ncomp_recommended` (also in NCOMP_RECOMMENDED), and
        measure_lines accepts ncomp="recommended".
    cont_order : 2 = quadratic, matching Topping. Set 1 for linear. Ignored
        when simultaneous=False.
    simultaneous : True (default) fits the continuum jointly with the line, as
        Topping do. False is the pipeline's usual two-step: normalize first via
        `cont_mode`, then fit the line against a continuum fixed at 1. The spec
        calls for the two-step as a sensitivity test, not the primary.
    z, lsf_fwhm_A : used only to set a physical starting width and sigma bounds.
        Without z the bounds fall back to a generous fraction of the window.

    Returns a dict; `dv_kms` is a direct fit parameter, so `dv_err_kms` is the
    covariance error on the velocity itself (not propagated from a centroid).

    WIDTH CAVEAT: an unmodelled partner makes the fitted sigma too LARGE (the
    single Gaussian widens to swallow the companion). Verified: Si II 1526 with
    Si II* present recovers sigma 1.436 A at ncomp=1 vs the injected 1.248 A,
    and exactly 1.248 at ncomp=2. So any width-based inference -- e.g. the
    sigma_v ~ 124 km/s galaxy-scatter check in the spec -- MUST use ncomp > 1.
    """
    spec = line_spec(line)
    comps = spec.components[:max(1, int(ncomp))]
    if int(ncomp) > len(spec.components):
        comps = spec.components

    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = None if err is None else np.asarray(err, float)

    # two-step mode: normalize first, then fit against a continuum fixed at 1
    norm_failed = None
    if not simultaneous:
        _n = normalize_local(wave, flux, err, spec, mode=cont_mode,
                             cont_order=cont_order)
        if _n["success"]:
            wave, flux, err = _n["wave"], _n["nflux"], _n["nerr"]
        else:
            norm_failed = _n["reason"]

    lo, hi = spec.fit_span
    m = (wave >= lo) & (wave <= hi) & np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        m &= np.isfinite(err) & (err > 0)

    # ---- WIDTH BOUNDS -------------------------------------------------
    # THE LOWER BOUND IS THE INSTRUMENTAL LSF, EXACTLY. Convolution adds in
    # quadrature,
    #     sigma_obs = sqrt(sigma_int^2 + sigma_LSF^2)  >=  sigma_LSF,
    # so a fitted width below the LSF is not "a narrow line", it is an
    # unphysical solution and it means the model is wrong somewhere else --
    # most often a quadratic continuum curving down into the trough, eating the
    # broad wings and leaving a narrow core for the Gaussian to sit on. The
    # previous 0.5*sig_lsf floor let exactly that happen silently: a whole
    # science-line table came back at sigma = 0.5*sig_lsf on all three lines
    # (FWHM 201/190/166 km/s at z_med = 2.31), i.e. every fit railed, and the
    # railing check missed it because its tolerance was absolute.
    # Upper bound stays generous: real outflow troughs ARE intrinsically broad.
    # Computed BEFORE nan_res so a failed fit still reports the LSF it was
    # working against -- otherwise an unrelated failure prints as "no z passed".
    sig_lsf = _lsf_sigma_rest_A(spec, z, lsf_mode=lsf_mode, lsf_fwhm_A=lsf_fwhm_A)
    lsf_floor = sig_lsf is not None
    if sigma_bounds_A is not None:
        sig_lo, sig_hi = (float(sigma_bounds_A[0]), float(sigma_bounds_A[1]))
        sig_0 = float(np.clip(sig_lsf or np.mean(sigma_bounds_A), sig_lo, sig_hi))
        lsf_floor = sig_lsf is not None and sig_lo >= 0.999 * sig_lsf
    elif sig_lsf is not None:
        sig_lo, sig_hi, sig_0 = sig_lsf, 6.0 * sig_lsf, 1.3 * sig_lsf
    else:
        # NO z GIVEN -> no LSF prior at all, and this floor is far below any
        # real instrumental width. Flagged in the result so it cannot pass
        # unnoticed; pass z= (measure_lis_stack does this from z_median).
        sig_lo, sig_hi, sig_0 = 0.15, 0.25 * (hi - lo), 0.05 * (hi - lo)
    _fwhm_lsf = (2.3548200450309493 * C_KMS * sig_lsf / spec.lam_rest
                 if sig_lsf is not None else np.nan)

    ncomp_eff = len(comps)
    nan_res = dict(line=spec.name, ncomp=ncomp_eff, success=False,
                   dv_kms=np.nan, dv_err_kms=np.nan, sigma_A=np.nan,
                   sigma_err_A=np.nan, fwhm_kms=np.nan,
                   sigma_lsf_A=sig_lsf, sigma_bounds_A=(sig_lo, sig_hi),
                   lsf_floor=bool(lsf_floor), fwhm_lsf_kms=_fwhm_lsf,
                   amps=[np.nan] * ncomp_eff,
                   amp_errs=[np.nan] * ncomp_eff, comp_names=[c.name for c in comps],
                   cont_coeff=None, model=None, wave=wave[m] if m.any() else None,
                   flux=flux[m] if m.any() else None, reason=None, snr=np.nan)

    if m.sum() < min_pixels:
        nan_res["reason"] = (f"normalization failed: {norm_failed}" if norm_failed
                             else f"only {int(m.sum())} usable pixels (< {min_pixels})")
        return nan_res

    x, y = wave[m], flux[m]
    e = err[m] if err is not None else None

    # scale/center the continuum basis for conditioning
    x0 = spec.lam_rest
    xs = 0.5 * (hi - lo)
    # order < 0 is the sentinel for "no free continuum, held at 1" (two-step)
    order = -1 if not simultaneous else int(max(0, cont_order))

    def _model(xx, *p):
        dv, sig = p[0], p[1]
        amps = p[2:2 + ncomp_eff]
        cc = p[2 + ncomp_eff:]
        # len(cc)==0 only in two-step mode, where the data is already normalized
        out = np.polyval(cc, (xx - x0) / xs) if len(cc) else np.ones_like(xx)
        shift = 1.0 + dv / C_KMS
        for A, c in zip(amps, comps):
            mu = c.lam_rest * shift
            out = out + A * np.exp(-0.5 * ((xx - mu) / sig) ** 2)
        return out

    # ---- starting guesses ----
    norm = normalize_local(wave, flux, err, spec, mode="weldon_linear")
    if norm["success"]:
        cont_lvl = 1.0 if not simultaneous else float(np.nanmedian(norm["cont"]))
        ext = measure_extremum(norm["wave"], norm["nflux"], spec)
        depth = ext["depth"] if np.isfinite(ext.get("depth", np.nan)) else 0.2
        if np.isfinite(ext.get("vel_kms", np.nan)) and p0_dv_kms == 0.0:
            p0_dv_kms = float(np.clip(ext["vel_kms"], *dv_bounds_kms))
    else:
        cont_lvl = float(np.nanmedian(y))
        depth = 0.2
    amp_main = spec.sign * abs(depth) * abs(cont_lvl)
    if amp_main == 0.0 or not np.isfinite(amp_main):
        amp_main = spec.sign * 0.1 * abs(cont_lvl or 1.0)

    p0 = [float(np.clip(p0_dv_kms, *dv_bounds_kms)), float(sig_0), float(amp_main)]
    lower = [float(dv_bounds_kms[0]), float(sig_lo)]
    upper = [float(dv_bounds_kms[1]), float(sig_hi)]
    lower.append(-np.inf if spec.sign < 0 else 0.0)
    upper.append(0.0 if spec.sign < 0 else np.inf)
    for c in comps[1:]:
        p0.append(0.25 * abs(amp_main) * c.sign)
        lower.append(-np.inf if c.sign < 0 else 0.0)
        upper.append(0.0 if c.sign < 0 else np.inf)
    # continuum: highest power first (np.polyval order); start flat at cont_lvl.
    # order < 0 = two-step mode, continuum held at 1 with no free parameters.
    if order >= 0:
        p0 += [0.0] * order + [cont_lvl]
        lower += [-np.inf] * (order + 1)
        upper += [np.inf] * (order + 1)

    try:
        from scipy.optimize import curve_fit
    except Exception:                                        # pragma: no cover
        nan_res["reason"] = "scipy unavailable"
        return nan_res

    try:
        popt, pcov = curve_fit(_model, x, y, p0=p0, sigma=e,
                               absolute_sigma=e is not None,
                               bounds=(lower, upper), maxfev=20000)
    except Exception as exc:
        nan_res["reason"] = f"fit failed: {type(exc).__name__}: {exc}"
        return nan_res

    perr = (np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
            if pcov is not None and np.all(np.isfinite(pcov))
            else np.full(len(popt), np.nan))

    dv, sig = float(popt[0]), float(popt[1])
    amps = [float(v) for v in popt[2:2 + ncomp_eff]]
    amp_errs = [float(v) for v in perr[2:2 + ncomp_eff]]

    # Railing check. Tolerances must be RELATIVE to the parameter's own scale:
    # an absolute 1e-9 on sigma ~ 1.4 A is far tighter than where a bounded
    # trust-region solver actually parks (~1e-8 relative), so the old absolute
    # test silently passed fits that were sitting on the bound. That is how a
    # whole table of sigma = 0.5*sig_lsf results got reported as successful.
    def _at_bound(val, bound, scale):
        return abs(val - bound) <= 1e-4 * max(abs(scale), 1e-12)

    railed = []
    dv_scale = max(abs(upper[0] - lower[0]), 1.0)
    if _at_bound(dv, lower[0], dv_scale) or _at_bound(dv, upper[0], dv_scale):
        railed.append("dv")
    if _at_bound(sig, sig_lo, sig_lo) or _at_bound(sig, sig_hi, sig_hi):
        railed.append(f"sigma (={sig:.4f} A, bounds {sig_lo:.4f}-{sig_hi:.4f})")
    ok = bool(np.isfinite(dv) and np.isfinite(sig) and amps[0] != 0.0 and not railed)

    snr = abs(amps[0]) / amp_errs[0] if amp_errs[0] and np.isfinite(amp_errs[0]) else np.nan

    return dict(line=spec.name, ncomp=ncomp_eff, success=ok,
                dv_kms=dv if ok else np.nan,
                dv_err_kms=float(perr[0]),
                sigma_A=sig, sigma_err_A=float(perr[1]),
                sigma_lsf_A=sig_lsf, sigma_bounds_A=(sig_lo, sig_hi),
                lsf_floor=bool(lsf_floor), fwhm_lsf_kms=_fwhm_lsf,
                fwhm_kms=2.3548200450309493 * C_KMS * sig / spec.lam_rest,
                amps=amps, amp_errs=amp_errs,
                comp_names=[c.name for c in comps],
                cont_coeff=[float(v) for v in popt[2 + ncomp_eff:]],
                model=_model(x, *popt), wave=x, flux=y, snr=snr,
                reason=None if ok else ("railed on " + "/".join(railed)
                                        if railed else "non-finite fit"))


# =====================================================================
# 5.  BOTH ESTIMATORS + TOPPING'S 100x ERROR-SPECTRUM MONTE CARLO
# =====================================================================

def measure_line(wave, flux, err=None, line="SiII1526", *, ncomp=1,
                 cont_mode="weldon_linear", cont_order=2, z=None,
                 lsf_fwhm_A=None, lsf_mode="fwhm", n_mc=100, seed=0,
                 search_halfwidth_A=4.0, refine=True, dv_bounds_kms=(-1200., 1200.),
                 sigma_bounds_A=None, min_pixels=10, simultaneous=True):
    """
    One line, both estimators, with the reference error model.

    `cont_mode` sets the local continuum for the TROUGH-MINIMUM estimator and
    the equivalent width. With simultaneous=True (the default) the GAUSSIAN
    estimator fits its own polynomial continuum jointly and is therefore
    unaffected by cont_mode -- if you want to test continuum sensitivity on
    v_gauss, vary `cont_order`, or set simultaneous=False to route it through
    cont_mode as well.

    Error model is Topping sec 2.3 verbatim: "We repeated this fitting process
    100 times for each line, and with every iteration we perturbed the spectrum
    by its corresponding error spectrum. The average and standard deviation of
    the centroids from the 100 trials became the measured redshift and
    uncertainty for each spectral line."

    So the reported value is the MC MEAN and the error is the MC STD. The
    unperturbed point estimate is returned alongside (`*_point`) because the
    two differ when the estimator is nonlinear -- a large gap is itself a
    warning that the fit is unstable.

    NOTE this is MEASUREMENT error on one stack only. Sample error (bootstrap
    over galaxies, and the scatter across sub-stacks) is a separate term and
    must not be conflated with it -- see the spec's systematics budget.

    Returns a flat dict; see describe_lis for the field meanings.
    """
    spec = line_spec(line)
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = None if err is None else np.asarray(err, float)

    # ---- point estimates ----
    norm = normalize_local(wave, flux, err, spec, mode=cont_mode,
                           cont_order=cont_order)
    ext = measure_extremum(norm["wave"], norm["nflux"], spec,
                           search_halfwidth_A=search_halfwidth_A, refine=refine)
    fit = fit_line_profile(wave, flux, err, spec, ncomp=ncomp,
                           cont_order=cont_order, z=z, lsf_fwhm_A=lsf_fwhm_A,
                           lsf_mode=lsf_mode, dv_bounds_kms=dv_bounds_kms,
                           sigma_bounds_A=sigma_bounds_A, min_pixels=min_pixels,
                           simultaneous=simultaneous, cont_mode=cont_mode)

    # ---- equivalent width, free and useful for split #15 ----
    ew = np.nan
    if norm["success"]:
        w_n, nf = norm["wave"], norm["nflux"]
        inner = np.abs(w_n - spec.lam_rest) <= search_halfwidth_A
        if inner.sum() > 2 and np.isfinite(nf[inner]).any():
            dl = float(np.nanmedian(np.diff(w_n)))
            ew = float(np.nansum(1.0 - nf[inner]) * dl)   # >0 = absorption

    # ---- Monte Carlo ----
    v_min = np.full(int(n_mc), np.nan)
    v_gauss = np.full(int(n_mc), np.nan)
    sig_mc = np.full(int(n_mc), np.nan)
    if err is not None and n_mc and int(n_mc) > 0:
        rng = np.random.default_rng(seed)
        safe_err = np.where(np.isfinite(err), err, 0.0)
        for i in range(int(n_mc)):
            f_i = flux + rng.standard_normal(flux.shape) * safe_err
            n_i = normalize_local(wave, f_i, err, spec, mode=cont_mode,
                                  cont_order=cont_order)
            if n_i["success"]:
                e_i = measure_extremum(n_i["wave"], n_i["nflux"], spec,
                                       search_halfwidth_A=search_halfwidth_A,
                                       refine=refine)
                if e_i["success"]:
                    v_min[i] = e_i["vel_kms"]
            g_i = fit_line_profile(wave, f_i, err, spec, ncomp=ncomp,
                                   cont_order=cont_order, z=z,
                                   lsf_fwhm_A=lsf_fwhm_A, lsf_mode=lsf_mode,
                                   dv_bounds_kms=dv_bounds_kms,
                                   sigma_bounds_A=sigma_bounds_A,
                                   min_pixels=min_pixels,
                                   simultaneous=simultaneous, cont_mode=cont_mode)
            if g_i["success"]:
                v_gauss[i] = g_i["dv_kms"]
                sig_mc[i] = g_i["sigma_A"]

    def _mc(a):
        good = np.isfinite(a)
        if good.sum() < 2:
            return np.nan, np.nan, int(good.sum())
        return float(np.mean(a[good])), float(np.std(a[good], ddof=1)), int(good.sum())

    vmin_mu, vmin_sd, vmin_n = _mc(v_min)
    vg_mu, vg_sd, vg_n = _mc(v_gauss)
    sg_mu, sg_sd, _ = _mc(sig_mc)

    return dict(
        line=spec.name, lam_rest=spec.lam_rest, sign=spec.sign,
        ncomp=fit["ncomp"], comp_names=fit["comp_names"],
        is_science=spec.name in SCIENCE_LINES,
        blended_at_virus=spec.name in BLENDED_AT_VIRUS,
        # --- estimator A: trough minimum (Weldon composite estimator) ---
        vel_min_kms=vmin_mu if np.isfinite(vmin_mu) else ext["vel_kms"],
        vel_min_err_kms=vmin_sd,
        vel_min_point_kms=ext["vel_kms"],
        min_refined=ext["refined"], depth=ext["depth"],
        # --- estimator B: Gaussian centroid (Weldon per-object estimator) ---
        vel_gauss_kms=vg_mu if np.isfinite(vg_mu) else fit["dv_kms"],
        vel_gauss_err_kms=vg_sd,
        vel_gauss_point_kms=fit["dv_kms"],
        vel_gauss_cov_err_kms=fit["dv_err_kms"],
        sigma_A=sg_mu if np.isfinite(sg_mu) else fit["sigma_A"],
        sigma_err_A=sg_sd, fwhm_kms=fit["fwhm_kms"],
        fwhm_lsf_kms=fit["fwhm_lsf_kms"], lsf_floor=fit["lsf_floor"],
        amps=fit["amps"], amp_errs=fit["amp_errs"], snr=fit["snr"],
        # --- shared ---
        ew_rest_A=ew,
        asymmetry_kms=((vmin_mu - vg_mu) if np.isfinite(vmin_mu) and np.isfinite(vg_mu)
                       else np.nan),
        pix_scale_kms=ext["pix_scale_kms"], pix_scale_A=ext["pix_scale_A"],
        n_mc=int(n_mc), n_mc_min_ok=vmin_n, n_mc_gauss_ok=vg_n,
        cont_mode=cont_mode, cont_order=cont_order, z=z,
        fit_success=fit["success"], fit_reason=fit["reason"],
        norm_success=norm["success"], norm_reason=norm["reason"],
        _norm=norm, _fit=fit, _ext=ext,
    )


def measure_lines(wave, flux, err=None, lines=None, *, ncomp=1, **kw):
    """
    Loop measure_line over a set of lines.

    lines : None -> every entry in LINES. 'science' -> Topping's three.
        Otherwise a sequence of names or LineSpecs.
    ncomp : int applied to all; a dict {line_name: ncomp} for per-line control
        (e.g. {'CII1334': 2} to add the C II* component only there); or the
        string "recommended" to use NCOMP_RECOMMENDED, i.e. enough components
        per line for an unbiased velocity. ncomp=1 (the default) is the
        estimator-matched choice; run both and report the difference.
    """
    if lines is None:
        keys = list(LINES)
    elif isinstance(lines, str):
        keys = list(SCIENCE_LINES) if lines == "science" else [lines]
    else:
        keys = [l.name if isinstance(l, LineSpec) else l for l in lines]

    out = {}
    for k in keys:
        if isinstance(ncomp, str):
            if ncomp != "recommended":
                raise ValueError("ncomp must be an int, a dict, or 'recommended'")
            n = NCOMP_RECOMMENDED.get(k, 1)
        elif isinstance(ncomp, dict):
            n = ncomp.get(k, 1)
        else:
            n = ncomp
        out[k] = measure_line(wave, flux, err, k, ncomp=n, **kw)
    return out


def compare_ncomp(wave, flux, err=None, lines="science", *, n_mc=0, **kw):
    """
    Run each line at ncomp=1 (Topping-matched) and at its recommended ncomp,
    and report the shift. The shift IS the blend systematic for that line --
    quote it, do not silently apply it.
    """
    res1 = measure_lines(wave, flux, err, lines=lines, ncomp=1, n_mc=n_mc, **kw)
    resN = measure_lines(wave, flux, err, lines=lines, ncomp="recommended",
                         n_mc=n_mc, **kw)
    rows = {}
    for k in res1:
        v1 = res1[k]["vel_gauss_kms"]
        vN = resN[k]["vel_gauss_kms"]
        rows[k] = dict(ncomp1_kms=v1, ncomp_rec=resN[k]["ncomp"],
                       ncomp_rec_kms=vN, blend_shift_kms=vN - v1,
                       comp_names=resN[k]["comp_names"])
    return {"rows": rows, "ncomp1": res1, "ncomp_rec": resN}


def compare_cont_order(wave, flux, err=None, lines="science", *,
                       orders=(2, 1), ncomp=1, n_mc=0, **kw):
    """
    Refit each line at several continuum polynomial orders and report the shift.

    WHY THIS EXISTS. The simultaneous fit spans only ~30-45 A (Weldon's blue
    window start to red window end). A QUADRATIC over that span has enough
    freedom to curve down into the absorption trough, absorb the broad wings,
    and leave a narrow residual core -- at which point the Gaussian shrinks
    onto the core, the width drives down to its lower bound, and the amplitude
    inflates to compensate. The velocity then comes from the core pixel rather
    than the line. A fitted FWHM at the LSF floor is the signature.

    orders[0] (2, i.e. quadratic) is PRIMARY because it is what Topping sec 2.3
    actually did and the -60 +/- 10 comparison is estimator-matched only for
    that choice. orders[1:] are the sensitivity test. As with compare_ncomp,
    QUOTE THE SHIFT AS A SYSTEMATIC -- do not silently adopt the linear number
    because it looks better behaved.

    Returns {'rows': {line: {...}}, 'by_order': {order: results}}.
    """
    by_order, rows = {}, {}
    for o in orders:
        by_order[o] = measure_lines(wave, flux, err, lines=lines, ncomp=ncomp,
                                    cont_order=int(o), n_mc=n_mc, **kw)
    ref = by_order[orders[0]]
    for k in ref:
        row = dict(ref_order=int(orders[0]),
                   ref_kms=ref[k]["vel_gauss_kms"],
                   ref_fwhm_kms=ref[k]["fwhm_kms"],
                   fwhm_lsf_kms=ref[k].get("fwhm_lsf_kms", np.nan))
        for o in orders[1:]:
            r = by_order[o][k]
            row[f"order{int(o)}_kms"] = r["vel_gauss_kms"]
            row[f"order{int(o)}_fwhm_kms"] = r["fwhm_kms"]
            row[f"shift_order{int(o)}_kms"] = r["vel_gauss_kms"] - row["ref_kms"]
        rows[k] = row
    return {"rows": rows, "by_order": by_order, "orders": tuple(int(o) for o in orders)}


def describe_cont_order(cmp_result):
    """Print compare_cont_order, flagging any width sitting on the LSF floor."""
    orders = cmp_result["orders"]
    ref = orders[0]
    hdr = (f"{'line':11s} {'ord'+str(ref):>9s} {'FWHM':>7s} {'LSF':>7s}"
           + "".join(f" {'ord'+str(o):>9s} {'FWHM':>7s} {'shift':>7s}"
                     for o in orders[1:]))
    print("=" * len(hdr))
    print(f"continuum-order sensitivity (primary = order {ref}, Topping-matched)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for k, r in cmp_result["rows"].items():
        def _f(v, n=1):
            return f"{v:.{n}f}" if v is not None and np.isfinite(v) else "--"
        line = (f"{k:11s} {_f(r['ref_kms']):>9s} {_f(r['ref_fwhm_kms'],0):>7s} "
                f"{_f(r['fwhm_lsf_kms'],0):>7s}")
        for o in orders[1:]:
            line += (f" {_f(r[f'order{o}_kms']):>9s} "
                     f"{_f(r[f'order{o}_fwhm_kms'],0):>7s} "
                     f"{_f(r[f'shift_order{o}_kms']):>7s}")
        print(line)
        fl = r["fwhm_lsf_kms"]
        for o in orders:
            fw = r["ref_fwhm_kms"] if o == ref else r[f"order{o}_fwhm_kms"]
            if np.isfinite(fw) and np.isfinite(fl) and fw < 1.02 * fl:
                print(f"{'':11s}   ! order {o}: FWHM on the LSF floor -- "
                      f"velocity not trustworthy at this order")
    print("-" * len(hdr))
    print("  shift = (this order) - (primary order). This is a SYSTEMATIC to")
    print("  quote, not a correction to apply. If the shift is large AND the")
    print("  primary order's width sits on the LSF floor, the quadratic is")
    print("  eating the trough and the linear number is the better estimate.")
    print("=" * len(hdr))


# =====================================================================
# 6.  STACK WRAPPER
# =====================================================================

#: Weldon sec 3.5 build composites by "converting to luminosity density ...
#: and taking the unweighted average", so this is the (output, method) pair
#: that makes the number directly comparable to -60 +/- 10.
WELDON_MATCH = dict(output="L_fiber", method="mean")


def measure_lis_stack(core_stack, method=None, lines=None, *, ncomp=1, **kw):
    """
    Convenience wrapper over a stack.build_core_stacks return value.

    core_stack : the dict from build_core_stacks -- needs 'rest_wave',
        'stacks', and (optionally) 'z_median' / 'ngal' / 'output', which are
        carried into the result meta and used for the LSF prior.
    method : which combine method to read out of core_stack['stacks'].
        Defaults to 'mean' when present (Weldon's unweighted average), else the
        first available, and warns when it is not the Weldon-matching choice.

    To reproduce Weldon exactly:
        cs = build_core_stacks(cfg, core_prod, output="L_fiber", methods=["mean"])
        res = measure_lis_stack(cs, method="mean")
    and run the project fiducial alongside it -- the difference between the two
    IS the coadd-weighting systematic.
    """
    stacks = core_stack["stacks"]
    if method is None:
        method = "mean" if "mean" in stacks else next(iter(stacks))
    if method not in stacks:
        raise KeyError(f"method {method!r} not in stack; have {sorted(stacks)}")

    out_kind = core_stack.get("output")
    if not (method == "mean" and out_kind == "L_fiber"):
        print(f"[lis] NOTE: method={method!r}, output={out_kind!r}. Weldon+2022 "
              f"sec 3.5 use luminosity density + unweighted average "
              f"(output='L_fiber', method='mean'). This stack is comparable to "
              f"-60+/-10 only after allowing for the weighting difference.")

    kw.setdefault("z", core_stack.get("z_median"))
    res = measure_lines(core_stack["rest_wave"], stacks[method]["flux"],
                        stacks[method]["err"], lines=lines, ncomp=ncomp, **kw)
    return {"lines": res,
            "meta": {"method": method, "output": out_kind,
                     "ngal": core_stack.get("ngal"),
                     "z_median": core_stack.get("z_median"),
                     "weldon_match": bool(method == "mean" and out_kind == "L_fiber")}}


# =====================================================================
# 7.  z_LIS -- Topping's per-object estimator, applied to a stack
# =====================================================================

def z_lis_average(results, lines=SCIENCE_LINES, estimator="gauss",
                  require_success=True):
    """
    Topping sec 2.3: "We averaged any successful redshift measurement from these
    three lines to obtain z_LIS" -- an UNWEIGHTED mean over Si II 1260,
    C II 1334 and Si II 1526, "typically based on two lines" per galaxy.

    Unweighted is deliberate: it is what they did. Do not inverse-variance
    weight here or the estimator stops matching -60 +/- 10.

    estimator : 'gauss' -> compare the result to Weldon's -60 +/- 10
                'min'   -> compare to their composite (Fig. 2) points instead
    """
    res = results.get("lines", results)
    key = "vel_gauss_kms" if estimator == "gauss" else "vel_min_kms"
    ekey = "vel_gauss_err_kms" if estimator == "gauss" else "vel_min_err_kms"

    vals, errs, used, skipped = [], [], [], []
    for name in lines:
        r = res.get(name)
        if r is None:
            skipped.append((name, "not measured"))
            continue
        v = r.get(key, np.nan)
        if not np.isfinite(v) or (require_success and estimator == "gauss"
                                  and not r.get("fit_success", False)):
            skipped.append((name, r.get("fit_reason") or "non-finite"))
            continue
        vals.append(float(v))
        errs.append(float(r.get(ekey, np.nan)))
        used.append(name)

    if not vals:
        return dict(v_lis_kms=np.nan, v_lis_err_kms=np.nan, n_lines=0,
                    lines_used=[], lines_skipped=skipped, estimator=estimator,
                    line_to_line_spread_kms=np.nan)

    vals_a, errs_a = np.asarray(vals, float), np.asarray(errs, float)
    mean_v = float(np.mean(vals_a))
    # error on the unweighted mean: MC errors in quadrature / n
    mc_term = (float(np.sqrt(np.nansum(errs_a ** 2)) / len(vals_a))
               if np.isfinite(errs_a).any() else np.nan)
    spread = float(np.std(vals_a, ddof=1)) if len(vals_a) > 1 else np.nan
    return dict(v_lis_kms=mean_v, v_lis_err_kms=mc_term, n_lines=len(vals),
                lines_used=used, lines_skipped=skipped, estimator=estimator,
                per_line_kms=dict(zip(used, vals)),
                line_to_line_spread_kms=spread)


# =====================================================================
# 8.  PRINTER
# =====================================================================

#: Weldon+2022 reference values. The FIRST is the only one to compare a
#: Gaussian-centroid measurement against.
WELDON_REF = {
    "dv_LIS_mean":      (-60.0, 10.0, "mean of per-object Gaussian centroids, "
                                      "149 gal, all with MOSDEF nebular z_sys"),
    "dv_Lya_mean":      (400.0, 23.0, "mean of per-object Lya centroids, 72 gal; "
                                      "1\"2 slit -- NOT comparable to a fiber core"),
    "v80_median":       (-428.0, 29.0, "blue-wing measure, NOT a centroid"),
    "vmax_median":      (-574.0, 29.0, "blue-wing measure, NOT a centroid"),
    "sigma_v_sample":   (124.0, np.nan, "galaxy-to-galaxy scatter implied by "
                                        "10*sqrt(155); check against excess "
                                        "composite width"),
}
EPS_LRIS_FLOOR_KMS = 22.0   # Topping sec 2.2: 0.3 A blue-side arc residual


def describe_lis(results, estimator="gauss", compare=True, show_all=True):
    """Print the measurement table with the Weldon comparison inline."""
    meta = results.get("meta", {}) if isinstance(results, dict) else {}
    res = results.get("lines", results)

    print("=" * 78)
    print("LIS / rest-UV line velocities   (Delta_v vs MOSDEF systemic, km/s)")
    if meta:
        print(f"  stack: method={meta.get('method')!r} output={meta.get('output')!r}  "
              f"ngal={meta.get('ngal')}  z_med={meta.get('z_median')}")
        if not meta.get("weldon_match"):
            print("  !! NOT the Weldon-matching coadd (L_fiber + mean); the "
                  "absolute comparison carries a weighting systematic.")
    print("=" * 78)
    hdr = (f"{'line':11s} {'v_min':>9s} {'+/-':>6s} {'v_gauss':>9s} {'+/-':>6s} "
           f"{'asym':>7s} {'FWHM':>7s} {'LSF':>7s} {'EW':>6s} {'S/N':>5s} {'nc':>3s}")
    print(hdr)
    print("-" * len(hdr))

    order = [k for k in SCIENCE_LINES if k in res]
    order += [k for k in res if k not in order] if show_all else []
    for k in order:
        r = res[k]
        tag = "*" if r.get("is_science") else " "
        def _f(v, n=1):
            return f"{v:.{n}f}" if v is not None and np.isfinite(v) else "--"
        print(f"{tag}{k:10s} {_f(r['vel_min_kms']):>9s} {_f(r['vel_min_err_kms']):>6s} "
              f"{_f(r['vel_gauss_kms']):>9s} {_f(r['vel_gauss_err_kms']):>6s} "
              f"{_f(r['asymmetry_kms']):>7s} {_f(r['fwhm_kms'],0):>7s} "
              f"{_f(r.get('fwhm_lsf_kms', np.nan),0):>7s} "
              f"{_f(r['ew_rest_A'],2):>6s} {_f(r['snr'],1):>5s} {r['ncomp']:>3d}")
        if not r.get("fit_success"):
            print(f"{'':11s}   gaussian fit: {r.get('fit_reason')}")
        # a fitted width at or below the instrumental LSF is unphysical --
        # sigma_obs = sqrt(sigma_int^2 + sigma_LSF^2) >= sigma_LSF, always.
        fw, fl = r.get("fwhm_kms", np.nan), r.get("fwhm_lsf_kms", np.nan)
        if np.isfinite(fw) and np.isfinite(fl) and fw < 1.02 * fl:
            print(f"{'':11s}   ! FWHM {fw:.0f} <= LSF {fl:.0f} km/s -- UNPHYSICAL. "
                  f"The fit is on the width floor; the velocity is not "
                  f"trustworthy. Check the continuum (try cont_order=1).")
        if not r.get("lsf_floor", True):
            print(f"{'':11s}   ! no LSF prior (z not passed) -- the width is "
                  f"unbounded below. Pass z=.")
        if r.get("blended_at_virus") and r["ncomp"] == 1:
            print(f"{'':11s}   ! blended at VIRUS resolution and ncomp=1 -- this "
                  f"is a blend centroid. Re-run with ncomp=2 to bound it.")

    px = next((r["pix_scale_kms"] for r in res.values()
               if np.isfinite(r.get("pix_scale_kms", np.nan))), np.nan)
    print("-" * len(hdr))
    print("  * = Topping's three best lines (Si II 1260, C II 1334, Si II 1526)")
    print("  v_min   = wavelength of the absolute trough minimum  -> compare to "
          "Weldon COMPOSITE values")
    print("  v_gauss = Gaussian centroid, simultaneous poly continuum -> compare "
          "to Weldon -60 +/- 10")
    print("  asym    = v_min - v_gauss (line asymmetry; >0 means the minimum "
          "sits redward of the centroid)")
    if np.isfinite(px):
        print(f"  rest-grid pixel ~ {px:.0f} km/s -- v_min is quantized at this "
              f"scale before parabolic refinement")

    if compare:
        avg = z_lis_average(results, estimator=estimator)
        ref, ref_e, ref_note = WELDON_REF["dv_LIS_mean"]
        print("-" * len(hdr))
        print(f"  z_LIS-equivalent (unweighted mean of {avg['n_lines']} line(s) "
              f"{avg['lines_used']}, estimator={estimator!r}):")
        if np.isfinite(avg["v_lis_kms"]):
            d = avg["v_lis_kms"] - ref
            tot = float(np.sqrt(np.nansum([avg["v_lis_err_kms"] ** 2, ref_e ** 2,
                                           EPS_LRIS_FLOOR_KMS ** 2])))
            print(f"     VDFI    = {avg['v_lis_kms']:+7.1f} +/- "
                  f"{avg['v_lis_err_kms']:.1f} km/s  (measurement only)")
            print(f"     Weldon  = {ref:+7.1f} +/- {ref_e:.1f} km/s   ({ref_note})")
            print(f"     diff    = {d:+7.1f} km/s ; combined sigma = {tot:.1f} "
                  f"(incl. {EPS_LRIS_FLOOR_KMS:.0f} km/s LRIS floor) -> "
                  f"{abs(d) / tot if tot else float('nan'):.1f} sigma")
            print(f"     line-to-line spread = {avg['line_to_line_spread_kms']:.1f} "
                  f"km/s  (optical-depth term; not an error on the mean)")
            print("     reminder: this constrains eps_VIRUS - eps_LRIS. It does "
                  "NOT close the MOSDEF tie -- MOSDEF cancels. See the spec sec 0.")
        else:
            print(f"     no usable lines: {avg['lines_skipped']}")
        if avg["lines_skipped"]:
            print(f"     skipped: {avg['lines_skipped']}")
    print("=" * 78)


# =====================================================================
# 9.  DIAGNOSTIC PLOTS
#     Topping "manually inspected the fits to each line, and excluded that line
#     if the fits were poor" -- these are that inspection step.
# =====================================================================

def as_line_result(result, line=None):
    """
    Coerce anything measure_line / measure_lines / measure_lis_stack can return
    into ONE single-line result dict.

    The three measurement entry points return three different shapes and it is
    very easy to hand a plotter the wrong one:
        measure_line       -> {'line': 'SiII1526', 'vel_gauss_kms': ..., ...}
        measure_lines      -> {'SiII1526': {...}, 'CII1334': {...}, ...}
        measure_lis_stack  -> {'lines': {<the above>}, 'meta': {...}}
    describe_lis already unwraps all three, so a caller who printed a table and
    then tried to plot the same object used to hit a bare KeyError: 'line'.

    line : pick a member by name when given a container. If omitted and the
        container holds exactly one line, that one is used; otherwise this
        raises with the available names, because guessing would silently plot
        the wrong line.
    """
    if not isinstance(result, dict):
        raise TypeError(f"expected a measurement dict, got {type(result).__name__}")

    # measure_lis_stack wrapper
    if "lines" in result and "line" not in result:
        return as_line_result(result["lines"], line=line)

    # already a single-line result
    if "line" in result and "vel_gauss_kms" in result:
        if line is not None and result["line"] != line:
            raise KeyError(f"this result is for {result['line']!r}, not {line!r}")
        return result

    # measure_lines mapping
    keys = [k for k, v in result.items() if isinstance(v, dict) and "line" in v]
    if not keys:
        raise KeyError(
            "not a LIS measurement result: no 'line' key and no per-line "
            f"sub-dicts. Got keys {sorted(result)[:8]}"
        )
    if line is not None:
        if line not in result:
            raise KeyError(f"line {line!r} not measured; have {keys}")
        return result[line]
    if len(keys) == 1:
        return result[keys[0]]
    raise KeyError(
        f"this is a multi-line result ({keys}). Pass line=... to plot one, "
        f"e.g. plot_lis_fit(res, line='SiII1526'), or use plot_lis_grid(res) "
        f"for all of them."
    )


def plot_lis_fit(result, ax=None, show_components=True, line=None):
    """
    Normalized spectrum, fitted model, both estimator positions, one line.

    `result` may be a single-line dict from measure_line, or a whole
    measure_lines / measure_lis_stack result plus line='SiII1526'. See
    as_line_result.
    """
    import matplotlib.pyplot as plt

    result = as_line_result(result, line=line)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.0))

    spec = LINES[result["line"]] if result["line"] in LINES else None
    norm, fit = result["_norm"], result["_fit"]

    if norm["success"]:
        ax.step(norm["wave"], norm["nflux"], where="mid", color="0.25", lw=1.1,
                label="normalized stack")
        if norm["nerr"] is not None:
            ax.fill_between(norm["wave"], norm["nflux"] - norm["nerr"],
                            norm["nflux"] + norm["nerr"], step="mid",
                            color="0.75", alpha=0.5, lw=0)
    if fit.get("model") is not None and fit.get("wave") is not None:
        cc = fit["cont_coeff"]
        lo, hi = spec.fit_span if spec else (fit["wave"][0], fit["wave"][-1])
        x0, xs = (spec.lam_rest, 0.5 * (hi - lo)) if spec else (0.0, 1.0)
        cont = np.polyval(cc, (fit["wave"] - x0) / xs) if cc else np.ones_like(fit["wave"])
        with np.errstate(invalid="ignore", divide="ignore"):
            ax.plot(fit["wave"], fit["model"] / cont, color="crimson", lw=1.6,
                    label=f"fit, ncomp={fit['ncomp']}")

    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.axvline(result["lam_rest"], color="k", lw=0.8, ls="--",
               label=f"rest {result['lam_rest']:.2f} A")

    if np.isfinite(result["vel_min_kms"]):
        lam = result["lam_rest"] * (1 + result["vel_min_kms"] / C_KMS)
        ax.axvline(lam, color="tab:blue", lw=1.3,
                   label=f"min {result['vel_min_kms']:+.0f} km/s")
    if np.isfinite(result["vel_gauss_kms"]):
        lam = result["lam_rest"] * (1 + result["vel_gauss_kms"] / C_KMS)
        ax.axvline(lam, color="tab:orange", lw=1.3,
                   label=f"gauss {result['vel_gauss_kms']:+.0f} km/s")

    if show_components and spec is not None:
        for c in spec.components[1:result["ncomp"]]:
            ax.axvline(c.lam_rest, color="tab:green", lw=0.8, ls="-.", alpha=0.8)
            ax.annotate(c.name, (c.lam_rest, ax.get_ylim()[0]), fontsize=7,
                        rotation=90, va="bottom", ha="right", color="tab:green")
    if spec is not None:
        for lo_, hi_ in spec.cont_bounds:
            ax.axvspan(lo_, hi_, color="tab:cyan", alpha=0.13, lw=0)

    ax.set_xlabel("rest wavelength (A)")
    ax.set_ylabel("normalized flux")
    ax.set_title(f"{result['line']}  ({'science' if result['is_science'] else 'aux'})",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    return ax


def plot_lis_grid(results, lines=None, ncols=2, figsize_per=(6.2, 3.4)):
    """plot_lis_fit for every measured line, in a grid."""
    import matplotlib.pyplot as plt

    res = results.get("lines", results) if isinstance(results, dict) else results
    if "line" in res and "vel_gauss_kms" in res:      # a single-line result
        res = {res["line"]: res}
    keys = list(lines) if lines is not None else (
        [k for k in SCIENCE_LINES if k in res] + [k for k in res if k not in SCIENCE_LINES])
    n = len(keys)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        plot_lis_fit(res[k], ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes
