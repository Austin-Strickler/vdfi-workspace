r"""
injection.py -- cube-level injection-recovery: fake sources into the REAL fiber cube.

Spec: `specs/injection-recovery.md` (v4, 2026-08-01). Build-order items 1 and 4.


=====================================================================
NAMING -- read this before using anything here
=====================================================================
There are TWO unrelated things in this codebase called "injection recovery",
and conflating them in a Methods section would be a real error:

  * SPECTRAL-level (pre-existing, elsewhere):
        validation.injection_recovery, validation.injection_bias_fast,
        core.injection_recovery_synthetic, core.injection_recovery_sample
    These inject a synthetic LINE into an error spectrum or a stacked
    product. They characterize the measurement METHOD (completeness, flux
    floor, centroid bias vs input S/N). They are closed loops by design and
    that is fine for what they answer.

  * CUBE-level (this module):
        inject_sources, recover_injections
    This injects a fake SOURCE into the real (nexp, nfib, nwave) fiber cube
    at a fake sky position, BEFORE extraction, and runs the unmodified
    pipeline over it. It validates the extraction -> background -> stacking
    -> measurement -> fitting CHAIN against a known truth, using real noise
    correlations, real sky residuals, real masking, real DAR and real
    per-exposure seeing -- precisely because none of those are modeled.

Spec sec 0.1: a fully synthetic cube is a closed loop -- every component of
the simulator is a component of the pipeline's own forward model, so
recovering the input proves only the absence of coding bugs, and "is the
simulation realistic enough?" is a question a referee can always ask and
that design can never answer. Hence: inject into real data.


=====================================================================
THE TWO GENERATORS, AND WHY THEY MUST STAY INDEPENDENT
=====================================================================
Spec sec 3.5. The single most likely way to produce a wrong "bias" in this
test is to compare the recovered value against the number typed into the
config. Three real effects separate the injected PARAMETERS from the
expected OBSERVABLES, all of them correctly reproduced by a correct
pipeline, and all of which look exactly like pipeline bias if you compare
against the wrong truth:

  (a) Bin truth is not dv(r_center). The PSF and the steep profile weight
      each bin's flux toward its inner edge, so bin i's truth is the
      FLUX-WEIGHTED mean of dv over that bin AFTER PSF convolution.
      Largest where the profile is steepest and the bins are widest -- the
      innermost bins and the [180, 350] straddling bin, i.e. exactly the
      bins carrying the result.

  (b) Injected sigma is not recovered sigma. A fiber near the 200 kpc break
      sees flux from both sides and records a superposition of lines at 0
      and +100 km/s. The recorded line is broader than the injected sigma,
      by an amount that varies with radius.

  (c) A velocity step plus PSF blending manufactures skew from symmetric
      lines. Family H's recovered alpha is NOT the injected alpha, and the
      difference is a prediction, not an error.

      SCOPE, 2026-08-16 (Austin): skew is NOT injected and NOT measured.
      The pipeline reports the median-50%-flux CENTROID and nothing else
      about line shape, so a recovered alpha has no consumer. Checked
      before dropping it: the concern was that a skewed line might bias
      the centroid against the finite +-1500 km/s measurement window, but
      at the observed width (320 km/s, see THE LSF CONVENTION below) that
      window is 4.7 sigma and the recovered median is exact to 0.01 km/s
      at alpha up to 4, at both dv = 0 and dv = +100. There is no
      truncation bias to find, so skew buys nothing a centroid-only
      measurement can use.

      Effect (c) still HAPPENS in family H -- the blend of a 0 and a +100
      line is genuinely asymmetric -- and predict_binned_observables still
      forward-models it correctly, because it builds each bin's line as a
      real superposition over shells. It is simply no longer compared
      against anything. Both families run alpha = 0.

      alpha is retained as a parameter (and handled exactly, see
      `apply_lsf`) so that reviving this is a config change rather than a
      code change.

So there are two implementations here:

    build_source_template(...)          -- the INJECTOR. 2-D Cartesian.
                                           Deposits flux into fibers.
    predict_binned_observables(...)     -- the TRUTH. 1-D radial quadrature.
                                           Predicts what should come back.

They share only the DEFINITION of the source (I(r), S(lambda), dv(r)) --
that sharing is correct, because the definition IS the truth. They share
NO machinery for getting from that definition to binned observables:
the injector convolves on a 2-D Cartesian grid using the production
fiber-response interpolator; the truth generator does 1-D radial quadrature
with its own analytically-integrated aperture response.

That independence is the entire point. Comparing the injector against
itself validates nothing, and two implementations disagreeing is the only
mechanism that would have caught the 2026-07-22 area-convention bug, where
one convention was wrong in one place and nothing existed to contradict it.

Nowhere in the recovery code does a success criterion reference a config
value directly. Everything compares against predict_binned_observables.


=====================================================================
THE THREE NORMALIZATION CONVENTIONS (spec sec 7.1) -- RESOLVED
=====================================================================
Audited 2026-08-04 against the current code, because sec 7.1 flags this as
the thing most likely to go wrong and it is the same three-way question
that produced the 2026-07-22 bug.

1. I(r) is a SURFACE BRIGHTNESS.
   `fitting.intrinsic_profile_expcore` returns I(r), and the post-2026-07-22
   convention (`fitting.bin_average_no_psf_expcore`) integrates it as
   2*pi*r*I(r)*dr and divides by the annulus area. So the total flux is

       L = \int I(r) 2 pi r dr

   The missing 2*pi*r on input was half of the 2026-07-22 bug. We normalize
   I explicitly to unit total flux (`_normalize_profile`) and multiply by L,
   so the convention is asserted rather than inherited.

2. moffat_psf_integration ALREADY FOLDS IN THE FIBER APERTURE.
   Read `extract.moffat_psf_integration`: it builds a Moffat on a Cartesian
   grid, normalizes it with `Z = Z / Z.sum()` (unit TOTAL flux), then for
   each offset r sums the pixels within the 0.75" fiber radius and applies
   `adj = pi*0.75^2 / (npix * scale^2)` -- a correction from the discrete
   pixel count to the true aperture area. So

       V(r, seeing) = fraction of a POINT SOURCE's total flux landing in a
                      fiber whose center is offset r from the source.

   It is NOT a surface-brightness profile and it does NOT need a further
   2*pi or a further aperture integration. Applying either is the
   2026-07-22 bug again. This is asserted numerically in `check_level0`.

3. Therefore extended sources need ONE convolution, not two.
   Convolution is associative, so for an extended source

       flux in fiber at r_f = \int I(r_s) V(|r_f - r_s|) d^2 r_s
                            = (I conv V)(r_f)

   i.e. "convolve with the PSF then integrate over the aperture" collapses
   to "convolve with V", because V is already the aperture-integrated PSF.
   Doing it as two steps is how a stray aperture area gets applied twice.

Ladder level 1 (sec 7.2) is the check on all three at once: a point source
into a zero cube must return V exactly, and a stray 2*pi or a missing
annulus area shows up as a clean multiplicative offset visible at a glance
rather than as a subtle distortion buried in a stacked average.


=====================================================================
WHAT IS DELIBERATELY NOT HERE
=====================================================================
  * The error array is not injected (spec sec 1.5). VDFI is sky-dominated at
    these fluxes, so the source's own shot noise is negligible against the
    background, and noiseless signal on real noise is exactly what isolates
    PIPELINE bias from source Poisson scatter. This is a decision, not an
    oversight. `shot_noise_ratio` quantifies it for Methods.
  * The InjectedSpectra overlay and inject_sources live in build-order item
    2, not here. This module is items 1 and 4: the generators, which are
    pure code and cost no extraction time.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf
from scipy.spatial import Delaunay, cKDTree

# Shared DEFINITIONS only -- see the module docstring on what may be shared.
from .fitting import intrinsic_profile_expcore
from .config import PipelineConfig

LYA_REST = PipelineConfig.LYA_REST      # 1215.67 A, vacuum
C_KMS = 299792.458

# --- THE INSTRUMENT'S LINE-SPREAD FUNCTION -------------------------------
# The measured effective resolution is 1.9 A FWHM in the REST frame (see the
# effective-resolution result; this is sample-dependent, and it is the number
# to quote rather than the raw 110-140 km/s pixel scale). Velocity is
# (1+z)-invariant, so it converts once and for all:
#
#     FWHM = c * 1.9 / 1215.67 = 468.6 km/s
#     sigma = FWHM / (2 sqrt(2 ln 2)) = 199.0 km/s
#
# Adopted as 200.0 (Austin, 2026-08-16) -- 0.5% from the derived value and
# far inside the uncertainty on the 1.9 A itself. This also happens to be
# the number that was already hard-coded in the level-1 notebook's cell 10,
# which was correct but undocumented; it now has a derivation attached.
LSF_SIGMA_KMS = 200.0


# --- THE WAVELENGTH FRAME OF THE DATA -----------------------------------
# VIRUS's observed axis (vdfi['wave'], 3470-5540 A) is AIR. HETDEX's public
# data release says so and star_lines.py's Gaia-anchored stellar test measured
# it directly (Austin, 2026-07-30). config.LYA_REST = 1215.67 is VACUUM, which
# is the only valid convention for a rest-UV line -- air is not even defined
# below ~2000 A.
#
# So the observed wavelength a real Lya line lands at is NOT lya_rest*(1+z):
# that is the VACUUM prediction, and the axis it is being compared against is
# air. The difference is ~1.16 A at z = 2.4, i.e. **~85 km/s**.
#
# WHY THE INJECTOR MUST CARE (found at ladder level 2, 2026-08-17).
# The injector writes onto the raw air axis; Stage 2 (stack.stack_spec_values)
# then does air_to_vac(wave) / (1+z) before anything is measured. Writing the
# line at lya_rest*(1+z) therefore emits it ~85 km/s redward of where a real
# source at that redshift would be, and the whole chain faithfully recovers
# the +85 -- the FIRST level-2 run came back at +73 km/s in every radial bin
# where 0 was injected.
#
# That is not a tolerable known offset. Spec sec 4.3 extracts the additive
# velocity bias `b` from family H's core dv = 0, and criterion 2 reads a
# nonzero `b` as the absolute zero point surviving the air->vacuum fix. An
# injector with its own frame error makes `b` measure ME instead of the
# pipeline, which kills criterion 2 outright.
#
# ONE definition, used by every injection-side site that touches the RAW
# observed axis -- the injector, the centroid estimator, and the pixel-scale
# report. Anything working on the pipeline's REST grid (which is already
# vacuum) keeps using LYA_REST directly. Same discipline as the LSF: set in
# one place, because the two-places version survived a week undetected.
DATA_WAVELENGTHS_ARE_AIR = True


def observed_lya_wavelength(z, lya_rest=None):
    """Observed wavelength of Lya at redshift z, IN THE DATA'S OWN FRAME.

    Returns the AIR wavelength (what vdfi['wave'] is labelled in), so a line
    written here is found by the production chain at exactly dv = 0 after its
    air_to_vac + rest-framing. Set DATA_WAVELENGTHS_ARE_AIR = False if the
    fiber data is ever re-reduced onto a vacuum axis, and every call site
    follows.

    Uses specutils' vac_to_air -- the exact inverse of the air_to_vac that
    stack.stack_spec_values applies -- rather than a second formula. A
    hand-rolled Edlen/Ciddor here would be a SECOND convention in the
    codebase, and the entire point of this module is that two implementations
    of the same quantity are for CHECKING each other, never for quietly
    disagreeing.
    """
    lya_rest = LYA_REST if lya_rest is None else float(lya_rest)
    lam_vac = lya_rest * (1.0 + float(z))
    if not DATA_WAVELENGTHS_ARE_AIR:
        return lam_vac
    from specutils.utils.wcs_utils import vac_to_air
    import astropy.units as _u
    return float(vac_to_air(lam_vac * _u.AA).to(_u.AA).value)

__all__ = [
    "fiber_response_hybrid", "PSF_GRID_VALID_ARCSEC", "LSF_SIGMA_KMS",
    "VelocityField", "ConstantVelocity", "StepVelocity", "apply_lsf",
    "SourceModel", "family_P", "family_H",
    "fiber_response_production", "fiber_response_independent",
    "select_nearby_fibers", "build_source_template",
    "clear_injection_caches", "time_template_build",
    "predict_binned_observables",
    "check_level0", "check_level1",
    "check_background_annulus_negligible", "shot_noise_ratio",
    "DATA_WAVELENGTHS_ARE_AIR", "observed_lya_wavelength",
    "InjectedSpectra", "check_overlay_identity",
    "inject_sources", "validate_inj_table", "INJ_REQUIRED_COLS",
]


# =====================================================================
# 1.  THE SOURCE MODEL
#     Shared by both generators. This is the DEFINITION of truth.
# =====================================================================

class VelocityField:
    """dv(r) in km/s, r in kpc. Subclasses define the radial dependence."""

    def __call__(self, r):
        raise NotImplementedError

    def is_constant(self):
        return False


class ConstantVelocity(VelocityField):
    """Family P: a single dv at all radii.

    Spec sec 4.1: a ZERO-velocity injection cannot distinguish "the pipeline
    measures velocity correctly" from "the pipeline returns zero regardless".
    A known nonzero constant tests the absolute velocity zero point through
    the entire chain -- extraction, background subtraction, rest-framing,
    stacking, centroid estimation -- which is precisely the quantity that was
    silently wrong until the air->vacuum correction.
    """

    def __init__(self, dv):
        self.dv = float(dv)

    def __call__(self, r):
        return np.full_like(np.asarray(r, dtype=float), self.dv)

    def is_constant(self):
        return True

    def __repr__(self):
        return f"ConstantVelocity(dv={self.dv:+.1f} km/s)"


class StepVelocity(VelocityField):
    """Family H: dv_in inside r_break, dv_out outside. Abrupt, not smoothed.

    Spec sec 4.2: a step is the cleanest probe of radial resolution -- the
    recovered curve is a STEP RESPONSE, from which the response to any other
    transition shape follows. Inject something already smooth and the
    smoothness put in cannot be separated from the smearing the pipeline
    added.

    r_break = 200 kpc is chosen deliberately AWAY from R_vir (~70-75 kpc).
    If the injected break sat at R_vir, recovering it would prove nothing --
    the result would be indistinguishable from the binning preferring that
    radius. At ~2.7 R_vir the test is sharp in both directions: recovering
    the break at 200 kpc shows the pipeline localizes a transition where it
    actually is; recovering it pulled toward R_vir, or a spurious feature at
    R_vir, means the binning imposes structure there -- precisely the
    objection a referee would raise against Paper I's headline.
    """

    def __init__(self, dv_in=0.0, dv_out=100.0, r_break=200.0):
        self.dv_in = float(dv_in)
        self.dv_out = float(dv_out)
        self.r_break = float(r_break)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        return np.where(r < self.r_break, self.dv_in, self.dv_out)

    def __repr__(self):
        return (f"StepVelocity({self.dv_in:+.1f} -> {self.dv_out:+.1f} km/s "
                f"at {self.r_break:.0f} kpc)")


class RampVelocity(VelocityField):
    """Family G: dv_in at small r, ramping to dv_out by r_out, flat outside.

    Added 2026-08-20 for the STEP-RESPONSE DECONVOLUTION run. StepVelocity
    answers "would a transition at 200 kpc have been seen?"; this answers the
    different question the real data poses -- "the measured core gradient is
    +196 -> +67 over 0-24 kpc, is the INTRINSIC gradient steeper than that?"

    It has to be a separate field rather than a StepVelocity with a small
    r_break because the two effects that dilute a core gradient (PSF blending
    across the gradient, and flux-weighting inside a wide bin) both act on the
    LOCAL SLOPE, which a step does not have. You cannot infer the response to
    a ramp from the response to a step without assuming linearity of the
    binning+PSF operator, and the whole point of the exercise is to measure
    that operator, not assume it.

    shape:
      "log"    dv linear in log10(r) between r_in and r_out. THE DEFAULT, and
               the right one for this target: the measured real curve is very
               close to linear in log r (bin centres 5/13/20/30 kpc read
               196/132/67/45, i.e. -194 km/s per dex, extrapolating to ~+330
               at 1 kpc and crossing zero near 50 kpc), and the bins are
               log-spaced so a log-linear ramp is resolved evenly across them.
      "linear" dv linear in r. Kept because a physical outflow decelerating
               with radius is closer to this, and the pair of shapes brackets
               the systematic from the assumed form.

    r_in must be > 0 for shape="log". dv is CLAMPED outside [r_in, r_out] --
    dv_in inside, dv_out outside -- so the field is bounded everywhere and
    r=0 is well defined. That flat inner core is deliberate: below ~1 kpc the
    seeing (1.3-2.0 arcsec = 11-16 kpc) means nothing is resolved anyway, and
    an unbounded log ramp would put arbitrary velocity on the r->0 shells that
    build_source_template integrates over.
    """

    def __init__(self, dv_in=300.0, dv_out=0.0, r_in=1.0, r_out=100.0,
                 shape="log", n_steps=8):
        if shape not in ("log", "linear"):
            raise ValueError("shape must be 'log' or 'linear'")
        if r_out <= r_in:
            raise ValueError("r_out must exceed r_in")
        if shape == "log" and r_in <= 0:
            raise ValueError("r_in must be > 0 for shape='log'")
        if n_steps is not None and int(n_steps) < 2:
            raise ValueError("n_steps must be >= 2, or None for continuous")
        self.dv_in = float(dv_in)
        self.dv_out = float(dv_out)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.shape = shape
        self.n_steps = None if n_steps is None else int(n_steps)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        if self.shape == "log":
            x = ((np.log10(np.clip(r, self.r_in, self.r_out))
                  - np.log10(self.r_in))
                 / (np.log10(self.r_out) - np.log10(self.r_in)))
        else:
            x = ((np.clip(r, self.r_in, self.r_out) - self.r_in)
                 / (self.r_out - self.r_in))
        if self.n_steps is not None:
            x = np.round(x * (self.n_steps - 1)) / (self.n_steps - 1)
        return self.dv_in + (self.dv_out - self.dv_in) * x

    def __repr__(self):
        st = "continuous" if self.n_steps is None else f"{self.n_steps} steps"
        return (f"RampVelocity({self.dv_in:+.1f} -> {self.dv_out:+.1f} km/s, "
                f"{self.r_in:g}-{self.r_out:g} kpc, {self.shape}, {st})")


class BinnedVelocity(VelocityField):
    """dv constant within each radial bin. The natural basis for the INVERSE
    problem: the pipeline reports one velocity per bin, so parameterising the
    intrinsic field on the same edges makes "what is the intrinsic dv in bin
    i" a question with exactly as many unknowns as there are measurements.

    Also the CHEAP one. `build_source_template` loops over
    `np.unique(shell_dv)`, so cost is set by the number of DISTINCT shell
    velocities, not by the number of shells -- see the guard there. A binned
    field on the 11 production edges costs 11 groups; a continuous field on a
    0.25" shell grid costs one group per shell.

    edges_kpc : (n+1,) bin edges. dv_kpc : (n,) velocity in each bin.
    Outside the last edge dv is `dv_out` (default: the last bin's value).
    """

    def __init__(self, edges_kpc, dv, dv_out=None):
        self.edges = np.asarray(edges_kpc, dtype=float)
        self.dv = np.asarray(dv, dtype=float)
        if self.dv.size != self.edges.size - 1:
            raise ValueError(f"dv has {self.dv.size} values but edges_kpc "
                             f"defines {self.edges.size - 1} bins")
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges_kpc must be strictly increasing")
        self.dv_out = float(self.dv[-1] if dv_out is None else dv_out)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        i = np.searchsorted(self.edges, r, side="right") - 1
        out = np.full(r.shape, self.dv_out, dtype=float)
        inside = (i >= 0) & (i < self.dv.size)
        out[inside] = self.dv[i[inside]]
        return out

    def __repr__(self):
        return (f"BinnedVelocity({self.dv.size} bins, "
                f"{self.dv.min():+.0f}..{self.dv.max():+.0f} km/s, "
                f"outside {self.dv_out:+.0f})")


DEFAULT_R_TRUNC_KPC = 2000.0        # the outermost production bin edge


def _normalize_profile(prof_fn, r_max=DEFAULT_R_TRUNC_KPC, n=200000):
    """Return the scalar N such that \\int_0^r_max (prof/N) 2 pi r dr = 1.

    Convention 1 in the module docstring, asserted rather than inherited.
    The 2*pi*r measure here is the half of the 2026-07-22 bug that was
    missing on input to the ring convolution.

    r_max IS PART OF THE SOURCE DEFINITION, not a numerical detail -- see
    SourceModel.r_trunc.
    """
    r = np.linspace(0.0, r_max, n)
    return float(np.trapz(prof_fn(r) * 2.0 * np.pi * r, r))


def apply_lsf(sigma_kms, alpha, lsf_kms):
    """Blur an intrinsic line by the instrument, EXACTLY. Returns
    (sigma_observed, alpha_observed).

    An instrument cannot record a line narrower than its own resolution, so
    the line the detector sees is the intrinsic line smeared by the LSF.
    That smearing is a convolution, and for the skewed-Gaussian family it has
    a closed form -- the blurred line is still a skewed Gaussian, with BOTH
    parameters changed:

        sigma_obs = sqrt(sigma^2 + lsf^2)          <- the familiar one
        delta     = alpha / sqrt(1 + alpha^2)
        delta_obs = delta * sigma / sigma_obs
        alpha_obs = delta_obs / sqrt(1 - delta_obs^2)

    Verified numerically against a brute-force FFT convolution: agreement to
    2e-15 of the peak. This is exact, not an approximation.

    WHY alpha HAS TO MOVE TOO, even though we currently run alpha = 0.
    Adding sigma in quadrature while holding alpha fixed is the intuitive
    thing to write and it is WRONG for any skewed line. Blurring makes a
    lopsided line MORE symmetric -- it fills in the steep side -- so the
    observed skew is always smaller than the injected one. At sigma = 250,
    lsf = 200, alpha = 2 the observed skew is 0.98, not 2. And because the
    50%-flux centroid of a skewed line depends on the skew, getting alpha
    wrong ALSO puts the predicted centroid in the wrong place: 209.4 km/s
    where the true blurred centroid is 172.5, a 36.9 km/s error.

    That would be invisible today (alpha = 0 makes delta = 0, so alpha_obs =
    0 and this reduces to the plain quadrature sum) and would appear the
    moment anyone set alpha nonzero -- disguised, like the 2026-08-07
    half-pixel bug, as a clean systemic velocity shift rather than as a line
    -shape error. Written exactly so that trap is not left armed.
    """
    sigma_kms = float(sigma_kms)
    lsf_kms = float(lsf_kms)
    alpha = float(alpha)
    sigma_obs = float(np.hypot(sigma_kms, lsf_kms))
    if alpha == 0.0 or lsf_kms == 0.0:
        return sigma_obs, alpha
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    delta_obs = delta * sigma_kms / sigma_obs
    alpha_obs = float(delta_obs / np.sqrt(1.0 - delta_obs ** 2))
    return sigma_obs, alpha_obs


class SourceModel:
    """f(r, lambda) = L * I(r) * S(lambda | z, dv(r), sigma, alpha).

    Spec sec 3.1: this is NOT separable once dv depends on radius. It is a
    FAMILY of separable products indexed by r, evaluated on an (r, lambda)
    grid, not a single outer product. Cheap either way, but the distinction
    has teeth: because different radii carry different velocities, a fiber
    integrates a superposition of differently-shifted lines, not one shifted
    line. Its recorded spectrum is broader than the injected sigma, and can
    be skewed even when alpha = 0. That is real physics, correctly
    reproduced -- and it is why the truth is the forward-modeled observable,
    never the injected parameter.

    Family P is the exception: dv is constant, so P IS separable, which is
    one more reason it is the right first rung on the ladder.

    Parameters
    ----------
    L : total Lyman-alpha luminosity/flux, in whatever units vdfi['spectra']
        carries. Sets the overall scale; \\int I 2 pi r dr = 1 by construction.
    profile : "point" or "expcore".
    h1, A2_over_A1, r_c, gamma : expcore parameters (ignored for "point").
        h1 is the exponential core scale = the literature's halo scale
        length (Austin's convention); r_c is the outer/two-halo transition.
        [FILL] per spec sec 3.2 -- these must come from the post-2026-07-22
        PSF-fix, post-2026-07-27 SMHM-switch refit, NOT from any existing
        figure. Injecting a stale h1 gives a test whose truth value does not
        correspond to the measurement it validates.
    sigma_kms : INTRINSIC line width (km/s), i.e. the width the galaxy has
        before the instrument touches it.
    lsf_kms : the instrument's resolution, as a sigma in km/s. Default
        LSF_SIGMA_KMS = 200, derived from the measured 1.9 A rest-frame FWHM.

        THE LSF LIVES HERE, IN ONE PLACE, ON PURPOSE (2026-08-16). Both
        generators get the observed line by calling `line_profile`, which
        applies it. Before this, the injector applied NO LSF and the truth
        generator applied its own -- so the injector was emitting a 250 km/s
        line while the truth generator predicted 320 km/s for the same
        source. Harmless for the centroid (symmetric blurring does not move
        a median, measured at <=0.07 km/s) which is why it survived a week,
        but it would have been fatal at ladder level 7: the injector would
        have added a line NARROWER THAN THE INSTRUMENT CAN PHYSICALLY
        PRODUCE into real spectra that are already blurred.

        Setting it in two places is how that comes back. Setting it in one
        place makes double-application impossible rather than unlikely.
    alpha : skew parameter of the skewed Gaussian. 0 = symmetric.
        OUT OF SCOPE as of 2026-08-16 -- both families run 0. The pipeline
        measures only the centroid, and skew was checked and shown not to
        bias it at this width and window. Retained as a parameter, and
        handled exactly by `apply_lsf`, so reviving it is a config change.
    velocity : a VelocityField.
    r_trunc : kpc. Radius beyond which the source is defined to have no
        flux, and therefore the radius L is normalized within.

        THIS IS NOT A NUMERICAL DETAIL, and the spec does not currently
        specify it (gap found at ladder level 0, 2026-08-04). The expcore
        outer term goes as I ~ r^-gamma at large r, so the flux integrand
        goes as r^(1-gamma) and the TOTAL FLUX DIVERGES for gamma < 2. The
        production default is gamma_fixed = 0.8, i.e. squarely divergent:
        the integrand grows as r^0.2 and the integral as r^1.2. Measured for
        expcore(h1=17, f=0.02, r_c=400, gamma=0.8):

            r_trunc (kpc)     total flux     core(<100 kpc) / total
                 1000            40059              0.0600
              >> 2000           103401              0.0232   <- default
                 3000           175087              0.0137
                10000           783227              0.0031
                50000          5486259              0.0004

        So "inject a source of luminosity L" is MEANINGLESS on its own --
        the fraction of L that lands in the core, which is what sets the
        injected stack's S/N and therefore the recovered h1 and the
        deconvolution floor, moves by a factor of ~140 across that range.
        Two people running "the same" injection at different r_trunc would
        get different answers with nothing in the config to explain why.

        Default: 2000 kpc, the outermost production bin edge, so that L is
        "the flux inside the measured aperture" -- the only choice that
        makes the injected L directly comparable to a bootstrapped real
        luminosity (spec sec 2.4). Recorded in the injection manifest.

        CHANGED 3000 -> 2000 on 2026-08-16, when production binning was set
        to 11 bins ending at 2000 kpc (Austin: runtime, and there is little
        S/N in the outer bins anyway). THIS DEFAULT MUST TRACK THE OUTERMOST
        BIN EDGE. If they disagree, L means "flux inside a radius you never
        measure" and the core fraction -- which sets the injected stack's
        S/N and therefore the recovered h1 and the deconvolution floor --
        silently changes. At 2000 the core carries 2.32% of L against 1.37%
        at 3000, i.e. 1.7x more signal where the measurement is best.

        NOTE this also means the divergence is not merely a normalization
        nuisance: with gamma = 0.8 the outer term carries most of the flux
        at any truncation, so the [FILL] values from the refit (spec sec 3.2)
        determine whether family H is a core test or an outskirts test.
    """

    def __init__(self, L=1.0, profile="expcore",
                 h1=17.0, A2_over_A1=0.02, r_c=400.0, gamma=0.8,
                 sigma_kms=250.0, alpha=0.0, lsf_kms=LSF_SIGMA_KMS,
                 velocity=None, label="", r_trunc=DEFAULT_R_TRUNC_KPC):
        if profile not in ("point", "expcore"):
            raise ValueError("profile must be 'point' or 'expcore'")
        self.r_trunc = float(r_trunc)
        self.L = float(L)
        self.profile = profile
        self.h1 = float(h1)
        self.A2_over_A1 = float(A2_over_A1)
        self.r_c = float(r_c)
        self.gamma = float(gamma)
        self.sigma_kms = float(sigma_kms)
        self.alpha = float(alpha)
        self.lsf_kms = float(lsf_kms)
        self.velocity = velocity if velocity is not None else ConstantVelocity(0.0)
        self.label = label
        self._norm = (None if profile == "point"
                      else _normalize_profile(self._raw_I, r_max=self.r_trunc))

    # ---- spatial ----------------------------------------------------

    def _raw_I(self, r):
        """Unnormalized I(r). Uses the PRODUCTION expcore definition."""
        return intrinsic_profile_expcore(r, 1.0, self.h1,
                                         self.A2_over_A1, self.r_c, self.gamma)

    def I(self, r):
        """Normalized surface brightness: \\int I(r) 2 pi r dr = 1.

        For profile="point" this is undefined (the source is a delta
        function) and raises -- a point source has no surface brightness
        profile to evaluate, and silently returning something finite is how
        a "point source" test quietly stops being one.
        """
        if self.profile == "point":
            raise ValueError(
                "I(r) is undefined for a point source -- it is a delta "
                "function. Both generators special-case profile='point' "
                "rather than evaluating a profile; if you reached here, a "
                "code path is treating P as extended, which would silently "
                "destroy the strongest null in the plan (spec sec 4.1)."
            )
        r = np.asarray(r, dtype=float)
        return np.where(r <= self.r_trunc, self._raw_I(r) / self._norm, 0.0)

    def enclosed_fraction(self, r):
        """Fraction of total flux inside projected radius r."""
        if self.profile == "point":
            return np.where(np.asarray(r, dtype=float) > 0, 1.0, 0.0)
        rr = np.linspace(0.0, float(np.max(r)), 20000)
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (self.I(rr[1:]) * 2 * np.pi * rr[1:]
                   + self.I(rr[:-1]) * 2 * np.pi * rr[:-1]) * np.diff(rr))])
        return np.interp(r, rr, cum)

    # ---- spectral ---------------------------------------------------

    @property
    def observed_line(self):
        """(sigma, alpha) of the line as the INSTRUMENT records it.

        The intrinsic line blurred by the LSF. This is what both generators
        emit, and it is the only width any comparison should ever be made
        against -- the intrinsic sigma is not observable.
        """
        return apply_lsf(self.sigma_kms, self.alpha, self.lsf_kms)

    @property
    def sigma_obs(self):
        """Observed line width, km/s: sqrt(intrinsic^2 + LSF^2)."""
        return self.observed_line[0]

    def line_profile(self, v, dv):
        """The line as OBSERVED: skewed Gaussian in velocity space, unit
        integral, 50%-flux centroid located at `dv`, LSF ALREADY APPLIED.

        LSF INCLUDED (2026-08-16). `sigma_kms` is what the galaxy emits;
        what a spectrograph records is that blurred by its own resolution,
        and nothing downstream can see the unblurred version. Applying it
        here -- inside the shared definition, which both generators call --
        means the injector and the truth generator cannot disagree about it,
        and neither can apply it twice.

        WHY THE CENTROID AND NOT THE MODE. The production centroid estimator
        is the median-50%-flux centroid, so "dv" is only an unambiguous truth
        if the profile's 50%-flux point IS dv. For alpha != 0 the skewed
        Gaussian's location parameter xi is NOT its median, so we solve for
        the xi that puts the median at dv. Skipping this solve would inject
        a line whose true centroid differs from the config value by tens of
        km/s at large alpha -- and that offset would be read out as pipeline
        bias. The solve uses the OBSERVED (post-LSF) shape, so `dv` means
        "where the centroid lands in the data", which is the only version of
        it any recovery diagnostic can check. For alpha = 0, which is what
        both families run, the median is the mean and xi = dv exactly.
        """
        v = np.asarray(v, dtype=float)
        sig, al = self.observed_line
        xi = self._xi_for_centroid(dv)
        t = (v - xi) / sig
        pdf = np.exp(-0.5 * t ** 2) / (sig * np.sqrt(2 * np.pi))
        cdf = 0.5 * (1.0 + erf(al * t / np.sqrt(2.0)))
        return 2.0 * pdf * cdf

    def _xi_for_centroid(self, dv):
        """Solve for the location parameter placing the 50%-flux point at dv.

        Works in the OBSERVED (post-LSF) shape -- see `line_profile`.
        """
        sig, al = self.observed_line
        if al == 0.0:
            return float(dv)
        key = round(float(dv), 6)
        cache = getattr(self, "_xi_cache", None)
        if cache is None:
            cache = self._xi_cache = {}
        if key in cache:
            return cache[key]
        # median of the standard skew-normal, found once on a fine grid
        med = getattr(self, "_median_std", None)
        if med is None:
            t = np.linspace(-12.0, 12.0, 400001)
            pdf = 2.0 * np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi) \
                * 0.5 * (1.0 + erf(al * t / np.sqrt(2.0)))
            cdf = (np.cumsum(pdf) - 0.5 * pdf) * (t[1] - t[0])
            cdf /= cdf[-1]
            med = self._median_std = float(np.interp(0.5, cdf, t))
        cache[key] = float(dv) - med * sig
        return cache[key]

    def __repr__(self):
        if self.profile == "point":
            spatial = "point source"
        else:
            spatial = (f"expcore(h1={self.h1:.1f}, f={self.A2_over_A1:.3g}, "
                       f"r_c={self.r_c:.0f}, gamma={self.gamma:.2f})")
        sig, al = self.observed_line
        return (f"SourceModel[{self.label or self.profile}] {spatial}, "
                f"sigma={self.sigma_kms:.0f}->{sig:.0f} km/s "
                f"(LSF {self.lsf_kms:.0f}), alpha={self.alpha:.2f}"
                + (f"->{al:.2f}" if al != self.alpha else "")
                + f", {self.velocity!r}")


def family_P(L=1.0, sigma_kms=250.0, dv=-50.0, lsf_kms=LSF_SIGMA_KMS):
    """Family P -- the point source. The strongest null in the whole plan.

    Spec sec 4.1. A point source convolved with the seeing is, by
    construction, the narrowest thing the instrument can produce. Any
    extended structure recovered from P is MANUFACTURED by the pipeline --
    by the PSF convention, the fiber-aperture integration, the background
    subtraction, or the profile fit. There is no interpretation under which
    recovering h1 > 0 from family P is real.

    This is a much better null than "an exponential with a small h1",
    because it removes the free parameter: you are not asking "is the
    recovered h1 close enough to the injected one", you are asking "is it
    consistent with zero". Given the 2026-07-22 PSF area-convention bug --
    where a wide-PSF h1 collapse turned out to be a normalization error
    back-ported from the UV code -- this family is exactly the regression
    test that bug should have had.

    P also directly measures the PSF-DECONVOLUTION FLOOR: the smallest h1
    the pipeline can distinguish from a point source at this S/N and sample
    size. That number belongs in Methods whether or not anything fails.

    dv = -50 km/s: opposite in sign to family H's +100 so the two can never
    be confused; comparable to the 85 km/s air/vacuum offset and to the
    residual 10-20 km/s now being quoted, so it sits in the regime that
    matters; and comfortably above the ~22 km/s velocity floor.
    """
    return SourceModel(L=L, profile="point", sigma_kms=sigma_kms, alpha=0.0,
                       lsf_kms=lsf_kms,
                       velocity=ConstantVelocity(dv), label="P")


def family_H(L=1.0, h1=17.0, A2_over_A1=0.02, r_c=400.0, gamma=0.8,
             sigma_kms=250.0, alpha=0.0, lsf_kms=LSF_SIGMA_KMS,
             dv_in=0.0, dv_out=100.0, r_break=200.0):
    """Family H -- the double halo with a piecewise dv(r).

    Spec sec 4.2. Validates the model the paper actually fits: if the
    two-component profile is recoverable at this S/N and sample size, the
    fitted h1, r_c and crossover radius mean what they are claimed to mean.
    If it is not, the fit is reporting parameters the data cannot constrain
    -- the outcome that would matter most to a referee.

    A single injection with a flat core and a stepped outer region contains
    both regimes at once: the flat region is the NULL (does the pipeline
    manufacture a gradient), the stepped region is the RECOVERY (does it
    return the right one), and the comparison is between two dv(r) curves
    rather than two scalars.

    Per spec sec 0.4 this is now the primary deliverable, ahead of profile
    recovery: the air->vacuum correction turned the large-radius result into
    a null, and a null cannot be defended with error bars. The referee
    question is "would you have detected an offset at large radius if one
    were there?" -- and the uncertainty on the recovered +100 IS that
    sensitivity statement.

    NOTE h1/sigma default to placeholders. They are [FILL] per spec
    sec 3.2/3.3 and gated on the post-PSF-fix, post-SMHM-switch refit.

    alpha defaults to 0 and is expected to STAY 0 (2026-08-16): skew is out
    of scope, because the pipeline reports only the centroid and skew was
    measured not to bias it here. See the module docstring, effect (c).
    """
    return SourceModel(L=L, profile="expcore", h1=h1, A2_over_A1=A2_over_A1,
                       r_c=r_c, gamma=gamma, sigma_kms=sigma_kms, alpha=alpha,
                       lsf_kms=lsf_kms,
                       velocity=StepVelocity(dv_in, dv_out, r_break),
                       label="H")


def family_G(L=1.0, h1=17.0, A2_over_A1=0.02, r_c=400.0, gamma=0.8,
             sigma_kms=250.0, alpha=0.0, lsf_kms=LSF_SIGMA_KMS,
             dv_in=300.0, dv_out=0.0, r_in=1.0, r_out=100.0, shape="log",
             n_steps=8, velocity=None):
    """Family G -- same expcore spatial profile as H, but a CORE GRADIENT.

    Added 2026-08-20. Not a fourth acceptance-test family: G is the forward
    model used to DECONVOLVE the real core gradient. The loop is

        guess an intrinsic dv(r) -> inject -> extract -> compare the recovered
        table against the REAL table -> adjust the guess

    and the answer is the dv(r) whose recovered table matches. Comparing
    against `predict_binned_observables` still checks the pipeline; comparing
    against the REAL run is what estimates the intrinsic profile.

    `velocity=` overrides the ramp entirely -- pass a `BinnedVelocity` to
    parameterise the intrinsic field on radial bins instead, which is both
    cheaper (cost is set by the number of DISTINCT shell velocities) and the
    natural basis for the inverse problem.

    It doubles as the strongest outer null available: with dv_out = 0 and
    r_out = 100 kpc, EIGHT bins (100-180 outward) have truth exactly 0, so a
    single extraction gives both the gradient response and an 8-bin null with
    known truth -- versus the two H bins that carried the null before.
    """
    return SourceModel(L=L, profile="expcore", h1=h1, A2_over_A1=A2_over_A1,
                       r_c=r_c, gamma=gamma, sigma_kms=sigma_kms, alpha=alpha,
                       lsf_kms=lsf_kms,
                       velocity=(velocity if velocity is not None else
                                 RampVelocity(dv_in, dv_out, r_in, r_out,
                                              shape=shape, n_steps=n_steps)),
                       label="G")


# =====================================================================
# 2.  FIBER RESPONSE -- two implementations, deliberately
# =====================================================================

def fiber_response_production(r_arcsec, seeing_fwhm, **kw):
    """V(r) via the PRODUCTION path: extract.moffat_psf_integration.

    Used by the INJECTOR. Spec sec 3.1: reuse the production optics rather
    than writing a second convolution, so that a PSF-convention error in
    production is REPRODUCED in the injection and therefore detectable as a
    residual. Cf. the 2026-07-22 area-convention bug -- a second,
    independent implementation there would have HIDDEN it.

    Returns the fraction of a point source's total flux entering a fiber
    whose center is offset r_arcsec from the source.

    GOTCHA, found 2026-08-04: `moffat_psf_integration` ends with a
    `LinearNDInterpolator` over the 2-D (r, seeing) point cloud, so it needs
    at least TWO DISTINCT seeing values -- one scalar seeing makes the input
    degenerate and qhull raises "Initial simplex is flat". Since injection is
    per exposure and `core._exposure_seeing` returns exactly one scalar per
    exposure, every call here would hit that. We bracket a scalar seeing with
    two neighbours and evaluate at the true value.

    This is a real property of the production path, not a quirk of ours: any
    caller wanting a single-seeing PSF has to do the same thing.

    COST, 2026-08-07. THIS FUNCTION IS EXPENSIVE PER RADIUS, AND CALLERS MUST
    TREAT IT THAT WAY. `moffat_psf_integration` is a Python loop that rebuilds
    a full (boxsize/scale)^2 = 281x281 Cartesian distance array and a boolean
    aperture mask FOR EVERY (seeing, radius) pair. That is ~1-3 ms each on a
    shared node, so the cost is entirely set by how many radii you ask for --
    asking for 2000 costs minutes, asking for 100 costs a fraction of a
    second, and V(r) inside the 6.25" valid region is a smooth PSF core that
    100 knots resolves to ~1e-4. See `_separation_table`.

    Two changes here for the same reason:
      * the scalar-seeing bracket is now TWO values, not three (a 33% saving
        for nothing -- two distinct values is all qhull needs);
      * V is read straight off the returned grid at the true seeing instead of
        going back through the `LinearNDInterpolator`. The interpolator only
        existed to evaluate the middle of a three-value bracket; if the true
        seeing is one of the rows actually computed, interpolating in the
        seeing direction is both slower and strictly less accurate.
    """
    from .extract import moffat_psf_integration
    r_arcsec = np.atleast_1d(np.asarray(r_arcsec, dtype=float))
    seeing = np.unique(np.atleast_1d(np.asarray(seeing_fwhm, dtype=float)))
    want = seeing.copy()
    if seeing.size < 2:
        s0 = float(seeing[0])
        seeing = np.array([s0, s0 * 1.1])       # s0 itself IS a computed row
    # r must also span >1 distinct value for the same reason
    r_grid = r_arcsec
    if r_grid.size < 2:
        r_grid = np.unique(np.concatenate([r_grid, r_grid + 0.01]))
    _interp, R, S, V = moffat_psf_integration(r_grid, seeing, **kw)

    out = np.empty((want.size, r_arcsec.size), dtype=float)
    for k, s0 in enumerate(want):
        m = np.isclose(S, s0)
        o = np.argsort(R[m])
        out[k] = np.interp(r_arcsec, R[m][o], V[m][o])
    return out


def fiber_response_independent(r_arcsec, seeing_fwhm, alpha=3.0,
                               fiber_radius=0.75, n_rad=None, n_phi=None):
    """V(r) via an INDEPENDENT path: analytic Moffat, polar quadrature.

    alpha default changed 3.5 -> 3.0 on 2026-08-04 to track
    extract.moffat_psf_integration's own default (config.PSFSpec's module
    comment has the full history). This MUST track the production default:
    check_level0's "THE ONE THAT MATTERS" test compares this function
    against fiber_response_production at the SAME alpha, so leaving this one
    at the old value would silently turn that comparison into two different
    PSF profiles instead of two independent implementations of the same one.

    Used by the TRUTH generator. Same quantity as
    `fiber_response_production`, computed a different way on purpose:

        production : Cartesian pixel grid, numeric sum normalized by
                     Z/Z.sum(), discretization corrected by a pixel-count
                     area factor.
        this       : closed-form normalized Moffat, integrated over the
                     fiber aperture in POLAR coordinates centered on the
                     fiber.

    Different discretization, different normalization route, different
    failure modes. If these two agree to a few parts in 1e3 (check_level0),
    the aperture/2*pi/normalization class of bug is excluded -- which is the
    class that produced the 2026-07-22 bug and which spec sec 7.1 names as
    the thing most likely to go wrong here.

    Normalized Moffat:  M(d) = (a-1)/(pi g^2) * (1 + (d/g)^2)^(-a),
    which integrates to 1 over the plane, with
    g = FWHM / (2 sqrt(2^(1/a) - 1)).

    QUADRATURE, changed 2026-08-07 -- midpoint 400x720 -> adaptive
    Gauss-Legendre in u, uniform in phi. This is BOTH more accurate and ~90x
    faster, and it was the single largest remaining cost in template
    building (~8 s per exposure, for the 3000-knot separation table
    `_shell_to_fiber_kernel` needs).

      * phi: the integrand is periodic in phi, so the uniform midpoint rule
        is already spectrally accurate there. Only the node count changes.
      * u: the integrand is smooth and polynomial-like across the aperture,
        which is exactly what Gauss-Legendre is for. Measured against a
        2400x4800 midpoint reference over r = 0 to 740", at seeing
        1.2-2.5": GL 32x64 is accurate to ~3e-8, the old midpoint 400x720
        to ~6e-7. Twenty times better on a thirty-fifth of the work.
      * ADAPTIVITY IS NOT OPTIONAL. Fixed 32 nodes fails badly when the PSF
        is much narrower than the fiber -- at FWHM 0.05" it was 24% wrong,
        because the integrand becomes a spike of width g << fiber_radius
        and no low-order rule resolves it. That is not a hypothetical
        corner: `check_level0`'s limiting check evaluates V(0) at seeing
        0.05" precisely to confirm V -> 1. So the node count scales with
        fiber_radius / g. Measured worst case across 0.05-4.0" FWHM: 1e-5.
    """
    r_arcsec = np.atleast_1d(np.asarray(r_arcsec, dtype=float))
    seeing = np.atleast_1d(np.asarray(seeing_fwhm, dtype=float))
    out = np.empty((seeing.size, r_arcsec.size), dtype=float)

    for j, fwhm in enumerate(seeing):
        g = float(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / alpha) - 1.0))
        nr = n_rad if n_rad else int(np.clip(32.0 * max(1.0, fiber_radius / g),
                                             32, 4096))
        npi = n_phi if n_phi else 2 * nr

        x, w = np.polynomial.legendre.leggauss(nr)
        u = 0.5 * fiber_radius * (x + 1.0)
        w_u = 0.5 * fiber_radius * w * u          # folds in the u du measure
        phi = (np.arange(npi) + 0.5) * (2 * np.pi / npi)
        dphi = 2 * np.pi / npi
        cosphi = np.cos(phi)

        norm = (alpha - 1.0) / (np.pi * g ** 2)
        # chunk over r so the (n_r, n_rad, n_phi) working array stays small
        chunk = max(1, int(4e6 / max(nr * npi, 1)))
        for i0 in range(0, r_arcsec.size, chunk):
            ri = r_arcsec[i0:i0 + chunk][:, None, None]
            # distance from the SOURCE to the point (u, phi) inside the fiber
            d2 = (ri ** 2 + u[None, :, None] ** 2
                  - 2.0 * ri * u[None, :, None] * cosphi[None, None, :])
            M = norm * (1.0 + d2 / g ** 2) ** (-alpha)
            out[j, i0:i0 + chunk] = (M * w_u[None, :, None]).sum(axis=(1, 2)) * dphi
    return out


# The production Moffat grid spans +/- boxsize/2 and each fiber samples a
# 0.75" aperture, so V is only correct out to (boxsize/2 - 0.75).
PSF_GRID_VALID_ARCSEC = 14.0 / 2.0 - 0.75          # 6.25" at the default boxsize


def fiber_response_hybrid(r_arcsec, seeing_fwhm, boxsize=14.0, **kw):
    """Production V inside the grid, analytic V outside. USED BY THE INJECTOR.

    WHY THIS EXISTS -- found at ladder level 0, 2026-08-04.
    `extract.moffat_psf_integration` evaluates the Moffat on a Cartesian grid
    spanning +/- boxsize/2 (default 14" -> +/-7") and, for each offset r,
    sums the pixels within 0.75" of the fiber center. Once r exceeds
    boxsize/2 - 0.75 = 6.25" the fiber aperture starts running off the edge
    of the grid, so the pixel count in `sel` falls and `adj` over-corrects.
    Measured against the analytic response at seeing 1.8":

        r = 6.25"   ratio 0.9963   (fine)
        r = 6.50"   ratio 1.0466   (+4.7%, aperture clipped)
        r = 7.00"   ratio 1.2726   (+27%)
        r >= 8.00"  sel.sum() == 0 -> divide-by-zero -> NaN

    This is NOT a production bug: the only production consumer,
    `get_spectrum_exposure`, is the CORE extraction and works inside
    core_radius = 3.5", comfortably within range. The radial-bin path bins
    fibers rather than PSF-weighting them, so it never calls the
    interpolator at all.

    It IS a trap for the injector, because an injected halo runs to 2700 kpc
    ~ 330" and every shell out there would come back NaN -- and NaN
    propagates through the sum rather than degrading to zero, so an injected
    profile would silently become all-NaN instead of visibly wrong.

    Simply enlarging boxsize is not an option: at r ~ 330" the grid would be
    (663/0.05)^2 ~ 1.8e8 pixels per seeing value per radius.

    So: production inside 6.25", analytic outside. The two agree to 0.4% at
    the boundary and the flux beyond it is < 1e-4 of the total, so the choice
    is numerically irrelevant -- but it is made explicitly and loudly here
    rather than left as a silent NaN.
    """
    r_arcsec = np.atleast_1d(np.asarray(r_arcsec, dtype=float))
    valid = boxsize / 2.0 - 0.75
    out = np.asarray(fiber_response_independent(r_arcsec, seeing_fwhm, **kw),
                     dtype=float)
    inner = r_arcsec <= valid
    if np.any(inner):
        vp = np.asarray(fiber_response_production(
            r_arcsec[inner], seeing_fwhm, boxsize=boxsize, **kw), dtype=float)
        good = np.isfinite(vp)
        out[:, inner] = np.where(good, vp, out[:, inner])
    return out


# =====================================================================
# 3.  THE INJECTOR -- 2-D Cartesian
# =====================================================================

def select_nearby_fibers(dra_1d, ddec_1d, r_max_kpc, kpc_per_arcsec,
                         pad_arcsec=5.0):
    """Boolean mask of the fibers an injected source can possibly reach.

    WHY THIS EXISTS -- found 2026-08-07, after a kernel death.
    `build_source_template` was being handed EVERY fiber in the exposure
    (~3e4), and its cost scales with both the fiber count and, worse, with
    `max(r_fib)` -- which for the full field is the FIELD RADIUS (many
    hundreds of arcsec), not the source radius. Three things blow up at once:

      * the returned template is (nfib, nwave) float64 -- ~280 MB per
        exposure at 3.4e4 x 1036, and the caller keeps one PER EXPOSURE;
      * the shell grid ran to max(r_fib) + 3*seeing at 0.25" steps;
      * the interpolation grid was `unique(round(r_fib, 3))`, i.e. one knot
        per MILLIARCSECOND over that whole range -- ~1e6 knots -- so the
        ring kernel was a (1e6, n_shell) array, tens of GB.

    Fibers outside the outermost production bin are discarded downstream
    anyway: `bin_template_radially` digitizes on the same edges and assigns
    them -1. So cutting them here is EXACT, not an approximation -- provided
    the cut radius is the outermost bin edge plus a margin, because binning
    uses the DAR-shifted, wavelength-median distance while this uses the
    static one. `pad_arcsec = 5"` is >> the DAR excursion (< 1").

    Parameters
    ----------
    dra_1d, ddec_1d : (nfib,) static fiber offsets in arcsec.
    r_max_kpc : the outermost production bin edge (i.e. BINS[-1]).
    pad_arcsec : safety margin on top, in arcsec.
    """
    d = np.hypot(np.asarray(dra_1d, dtype=float),
                 np.asarray(ddec_1d, dtype=float))
    return d <= (float(r_max_kpc) / float(kpc_per_arcsec) + float(pad_arcsec))


def _radius_interp_grid(r_lo, r_hi, seeing_fwhm, n=400, extra=()):
    """Knots for interpolating V_conv(r) onto the fiber radii.

    Replaces `unique(round(r_fib, 3))`. V convolved with the source profile
    is smooth on the seeing scale near the centre and a slowly-varying power
    law outside it, so a milliarcsecond-spaced knot per fiber-radius is
    ~1e6x more resolution than the function has structure -- and the ring
    kernel is (n_knot, n_shell), so that factor lands directly on memory.

    Linear inside ~3 FWHM (where the PSF core lives and the curvature is),
    logarithmic outside (where the profile is scale-free). 400 knots gives
    dr/r < 3% in the log part; the interpolation error on an r^-2 tail is
    then ~5e-4 relative, far below anything this test resolves.

    `extra` pins additional knots. The one that matters is r_trunc: the
    source is defined to have ZERO flux beyond it, so the convolved profile
    falls off a cliff there, and a knot straddling the cliff interpolates
    across it. Measured without the pin, fibers just outside r_trunc came
    back several hundred percent high -- on a value ~1e-9 of the peak, and
    outside the outermost radial bin, so it never reached a result; pinned,
    it is gone, and gone is easier to defend than "negligible".
    """
    r_lo = max(float(r_lo), 0.0)
    r_hi = max(float(r_hi), r_lo + 1e-3)
    knee = float(np.clip(3.0 * float(seeing_fwhm), 1.0, r_hi))
    lin = np.linspace(r_lo, knee, max(n // 2, 2))
    log = np.geomspace(max(knee, 1e-3), r_hi, max(n - n // 2, 2))
    pins = [x for x in np.atleast_1d(np.asarray(extra, dtype=float)).ravel()
            if r_lo < x < r_hi]
    # A hard truncation does not produce a step in the CONVOLVED profile, it
    # produces a knee: measured for expcore(h1=17, gamma=0.8) truncated at
    # 366", dlnV/dlnr runs -3.6 at 340", -10 at 360", -215 at 367". A log
    # grid at dr/r ~ 2% cannot follow slope -200, so pin a dense linear
    # block across the knee rather than a single knot at it.
    pins = np.concatenate(
        [np.linspace(0.92 * p, min(r_hi, 1.03 * p), 120) for p in pins]
    ) if pins else np.empty(0)
    return np.unique(np.concatenate([[r_lo], lin, log, pins, [r_hi]]))


def build_source_template(model, z, dra, ddec, seeing_fwhm, wave,
                          kpc_per_arcsec, grid_max_arcsec=None,
                          grid_step_arcsec=0.25, response=None,
                          n_r_interp=400, dtype=np.float32,
                          max_dv_groups=16):
    """Flux to ADD to each fiber, for one exposure: (nfib, nwave).

    This is the injector. It answers "how much of this source's flux lands
    in fiber f at wavelength lambda", by convolving the source's surface
    brightness with the aperture-integrated PSF V on a 2-D Cartesian grid.

    ONE convolution, not two -- see convention 3 in the module docstring.
    V already folds in the fiber aperture, so an extra aperture integration
    or an extra 2*pi is the 2026-07-22 bug all over again.

    Parameters
    ----------
    model : SourceModel
    z : source redshift. Sets the observed wavelength of the line, and
        therefore the sky-line coincidence rate, the sub-pixel phase against
        the 110-140 km/s grid, and where the line falls in the LSF. Spec
        sec 2.4: bootstrap z from the real run table so the injected stack is
        directly comparable to the real one on all three.
    dra, ddec : (nfib,) or (nfib, nwave) fiber offsets from the source, in
        arcsec. Pass the (nfib, nwave) form from
        `core._offsets_with_dar(...)` -- injecting at a fixed,
        wavelength-independent position would skip the DAR chain and is,
        per spec sec 3.1, the most likely way to build a test that passes
        vacuously.
    seeing_fwhm : scalar FWHM for this exposure, from
        `core._exposure_seeing(seeing, e)`.
    wave : (nwave,) observed wavelength grid, Angstrom.
    kpc_per_arcsec : angular scale at z (angular-diameter distance / 206265).
    n_r_interp : number of radial knots for the V_conv interpolation
        (`_radius_interp_grid`). Cost and memory are LINEAR in this; 400 is
        already ~1e3x more resolution than the function's structure.
    dtype : output dtype. float32 by default -- the array is (nfib, nwave)
        and the caller holds one per exposure, so float64 doubles a number
        that has already killed a kernel once. The template is a noiseless
        model added to float32 fiber data; float32 carries ~7 digits, which
        is ~5 orders below the level anything here is measured at.

    PASS ONLY THE FIBERS THAT MATTER -- see `select_nearby_fibers`. Handing
    this the full-field fiber list makes it both slow and memory-fatal, and
    the extra fibers are discarded by the radial binning anyway.

    Returns
    -------
    (nfib, nwave) float array of flux to add. Never negative, never NaN.
    """
    wave = np.asarray(wave, dtype=float)
    nwave = wave.size
    dra = np.asarray(dra, dtype=float)
    ddec = np.asarray(ddec, dtype=float)
    if dra.ndim == 1:
        dra = dra[:, None] * np.ones((1, nwave))
        ddec = ddec[:, None] * np.ones((1, nwave))
    nfib = dra.shape[0]

    r_fib = np.sqrt(dra ** 2 + ddec ** 2)          # (nfib, nwave), arcsec
    # The line goes where a REAL source at this z would land on this axis --
    # the AIR wavelength, not the vacuum prediction. See
    # observed_lya_wavelength; getting this wrong emits every source ~85
    # km/s redward and poisons the additive-bias term (spec criterion 2).
    lam0_obs = observed_lya_wavelength(z)
    v_grid = C_KMS * (wave - lam0_obs) / lam0_obs

    if response is None:
        response = fiber_response_hybrid   # see its docstring on the 6.25" edge

    # QUANTIZE THE GRID BOUNDS (2026-08-07, third timing pass).
    # The ring kernel below costs (n_r x n_shell x n_phi) interpolations --
    # ~70 million at production sizes -- and it depends ONLY on the radius
    # grid, the shell grid and the seeing. It does NOT depend on which
    # exposure we are on. But `r_uniq` was being built from THIS exposure's
    # nanmin/nanmax fiber radius, which wiggles by a fraction of an arcsec
    # between exposures, so every exposure got a very slightly different grid
    # and paid the full 70 million all over again.
    #
    # Pinning r_lo to 0 and rounding r_hi up to the next 5" makes the grids
    # bit-identical across exposures, which makes the kernel cacheable. With
    # SEEING_MODE="field" that means the kernel is built ONCE for the whole
    # loop instead of once per exposure.
    r_hi = float(np.ceil(float(np.nanmax(r_fib)) / 5.0) * 5.0)
    _pins = () if model.profile == "point" else (
        float(model.r_trunc) / float(kpc_per_arcsec),)
    r_uniq = _radius_interp_grid(0.0, r_hi, seeing_fwhm, n=n_r_interp,
                                 extra=_pins)

    # --- point source: no spatial convolution needed, V IS the answer ---
    if model.profile == "point":
        V_uniq = np.asarray(response(r_uniq, [seeing_fwhm])).ravel()
        V = np.interp(r_fib, r_uniq, V_uniq)            # (nfib, nwave)
        dv0 = float(model.velocity(np.array([0.0]))[0])
        S = model.line_profile(v_grid, dv0)             # (nwave,)
        return (model.L * V * S[None, :]).astype(dtype, copy=False)

    # --- extended: convolve I with V on a 2-D grid, ONE radius shell at a
    #     time so each shell carries its own dv(r). This is the sec 3.1
    #     "family of separable products indexed by r" made concrete: a fiber
    #     ends up with a SUPERPOSITION of differently-shifted lines, which is
    #     what produces effects (b) and (c) in the module docstring.
    #
    # GRID EXTENT. Two independent caps, and the smaller wins:
    #   * the SOURCE has no flux beyond r_trunc by definition (model.I returns
    #     0 there), so shells past it contribute exactly zero -- capping here
    #     is not an approximation;
    #   * no shell more than a few seeing FWHM beyond the outermost fiber can
    #     put light in a fiber.
    # The old default used the fiber cap ALONE, so on a full-field fiber list
    # it ran the shell grid out to the field radius -- thousands of empty
    # shells, each costing an (n_r, n_phi) ring evaluation.
    if grid_max_arcsec is None:
        src_max = float(model.r_trunc) / float(kpc_per_arcsec)
        fib_max = r_hi + 3.0 * float(seeing_fwhm)
        # rounded to the shell step for the same cacheability reason as r_hi
        grid_max_arcsec = float(np.ceil(min(src_max, fib_max)
                                        / grid_step_arcsec) * grid_step_arcsec)

    # source-plane shells, in arcsec
    s_edges = np.arange(0.0, grid_max_arcsec + grid_step_arcsec, grid_step_arcsec)
    s_mid = 0.5 * (s_edges[1:] + s_edges[:-1])
    s_kpc = s_mid * kpc_per_arcsec

    # flux in each shell (annulus area * mean SB), in kpc units
    e_kpc = s_edges * kpc_per_arcsec
    shell_flux = np.array([
        np.trapz(model.I(np.linspace(a, b, 64)) * 2 * np.pi * np.linspace(a, b, 64),
                 np.linspace(a, b, 64))
        for a, b in zip(e_kpc[:-1], e_kpc[1:])
    ])
    shell_dv = model.velocity(s_kpc)

    # COST GUARD -- added 2026-08-20 after a continuous RampVelocity turned a
    # 0.74 s/exp source into 7.15 s/exp and an ETA of 21 hours.
    #
    # The accumulation loop below runs once per DISTINCT shell velocity, and
    # every iteration pays a full np.interp over (nfib, nwave) plus an
    # (nfib, nwave) multiply-accumulate. That design is correct and fast for
    # ConstantVelocity (1 group) and StepVelocity (2 groups). A CONTINUOUS
    # dv(r) gives every shell its own velocity: at the 0.25" default step a
    # ramp out to 100 kpc (12.2" at z=2.4) is ~50 groups, i.e. 25x the work
    # AND 25x the large temporaries per exposure -- which is what turns a
    # linear slowdown into an apparently exponential one once the allocator
    # starts thrashing.
    #
    # Failing loudly beats discovering it from a moving ETA. Use
    # RampVelocity(n_steps=...) or BinnedVelocity to quantize the field, or
    # raise max_dv_groups deliberately if a run really needs it.
    n_groups = int(np.unique(shell_dv).size)
    if n_groups > int(max_dv_groups):
        raise ValueError(
            f"velocity field yields {n_groups} distinct shell velocities "
            f"(max_dv_groups={max_dv_groups}); the template loop costs one "
            f"full (nfib, nwave) interpolation per DISTINCT velocity, so this "
            f"run would be ~{n_groups / 2:.0f}x slower per exposure than a "
            f"StepVelocity one.\n"
            f"  fix: RampVelocity(..., n_steps=8) or BinnedVelocity(edges, dv)\n"
            f"  or:  build_source_template(..., max_dv_groups={n_groups}) to "
            f"accept the cost."
        )

    # For each fiber and each shell, the fraction of that shell's flux
    # entering the fiber = azimuthal average of V over the shell.
    #
    # CACHED. This is the single most expensive thing in the module -- it is
    # (n_r x n_shell x n_phi) interpolations, ~70 million at production sizes.
    # Every argument to it is now quantized above, so it is identical from one
    # exposure to the next and gets built once. The key covers everything the
    # two grids depend on; nothing about the fibers themselves enters, which
    # is the whole point.
    k_key = (id(response), round(float(seeing_fwhm), 4), r_hi,
             int(n_r_interp), round(grid_max_arcsec, 6),
             round(float(grid_step_arcsec), 6),
             None if not _pins else round(_pins[0], 6))
    K = _KERNEL_CACHE.get(k_key)
    if K is None:
        K = _shell_to_fiber_kernel(r_uniq, s_mid, seeing_fwhm, response)
        if len(_KERNEL_CACHE) > 8:      # each is ~4 MB; keep a handful
            _KERNEL_CACHE.clear()
        _KERNEL_CACHE[k_key] = K                                  # (nr, nshell)

    # collapse identical dv shells so the line profile is evaluated once per
    # distinct velocity rather than once per shell
    out = np.zeros((nfib, nwave), dtype=dtype)
    for dv in np.unique(shell_dv):
        w = shell_dv == dv
        if not np.any(w):
            continue
        Vr = K[:, w] @ shell_flux[w]                    # (nr,)
        Vf = np.interp(r_fib, r_uniq, Vr)               # (nfib, nwave)
        out += (model.L * Vf
                * model.line_profile(v_grid, dv)[None, :]).astype(dtype,
                                                                  copy=False)
    return out


_TRUTH_KERNEL_CACHE = {}   # predict_binned_observables' ring kernel
_V_TAB_CACHE = {}       # (response, seeing, d_max, n_phi) -> (d_tab, V_tab)
_KERNEL_CACHE = {}      # (response, seeing, grid bounds) -> K  (nr, nshell)


def clear_injection_caches():
    """Drop the V-table and ring-kernel caches.

    Only needed if you monkeypatch a response function or edit the module in
    place mid-session -- the cache keys use `id(response)`, which does not
    notice that a function object with the same identity now does something
    different. Cheap to call.
    """
    _V_TAB_CACHE.clear()
    _KERNEL_CACHE.clear()
    _TRUTH_KERNEL_CACHE.clear()


def _separation_table(d_max, n_near=100, n_far=500,
                      d_split=PSF_GRID_VALID_ARCSEC):
    """Knots for tabulating V(separation) from 0 to d_max.

    WHY THIS IS NOT JUST A GEOMSPACE -- found 2026-08-07, the second timing
    failure of this cell. The old table was `geomspace(1e-3, d_max, 3000)`.
    A geometric grid spends its points equally per DECADE, and the separations
    run 1e-3" to ~500" -- so about two thirds of the 3000 knots landed BELOW
    6.25". That is exactly the region `fiber_response_hybrid` serves from the
    production path, where each knot means a fresh 281x281 Moffat grid inside
    `moffat_psf_integration`'s Python loop. So the table was silently ordering
    ~2000 expensive evaluations x 3 bracket seeing values = ~6000 grid builds
    PER EXPOSURE, and none of it was cached because the seeing changes every
    exposure. Measured on Austin's node: ~10 minutes per exposure.

    Nothing needed that resolution. Inside 6.25" V is a smooth PSF core, and
    100 knots (dense linear near 0, log outside 1") resolve it to ~1e-4 --
    below the level the far side is accurate to anyway. Outside 6.25" the
    knots are analytic and effectively free, so they stay generous.

    The split point is not arbitrary: `PSF_GRID_VALID_ARCSEC` is the same
    6.25" where `fiber_response_hybrid` hands over from production to
    analytic, so "expensive knot" and "near knot" are the same set by
    construction.
    """
    d_max = float(d_max)
    d_split = min(float(d_split), d_max)
    near = np.concatenate([
        np.linspace(0.0, min(1.0, d_split), max(n_near // 2, 2)),
        np.geomspace(max(min(1.0, d_split), 1e-3), d_split,
                     max(n_near - n_near // 2, 2)),
    ])
    if d_max <= d_split:
        return np.unique(near)
    far = np.geomspace(d_split, d_max, max(int(n_far), 2))
    return np.unique(np.concatenate([near, far]))


def _shell_to_fiber_kernel(r_fiber, s_shell, seeing_fwhm, response,
                           n_phi=180):
    """Fraction of a unit-flux thin ring at radius s entering a fiber at r.

    Azimuthal average of V over the ring:
        K(r, s) = (1/2pi) \\int V(sqrt(r^2 + s^2 - 2 r s cos phi)) dphi

    V is tabulated once on a 1-D separation grid and interpolated, and the
    shell loop is CHUNKED: the full (n_r, n_shell, n_phi) separation array
    is ~10 GB at the grid sizes `predict_binned_observables` uses. Result is
    (n_r, n_shell) and small.
    """
    r_fiber = np.asarray(r_fiber, dtype=float)
    s_shell = np.asarray(s_shell, dtype=float)
    phi = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)      # symmetry: 0..pi
    cosphi = np.cos(phi)

    # V is a smooth 1-D function of separation; tabulate it once on a log
    # grid spanning every separation that can occur, then interpolate.
    #
    # CACHED ACROSS CALLS (2026-08-07). Building this table is the expensive
    # part of a single call: `fiber_response_hybrid` runs `fiber_response_
    # independent`'s Python double loop over all 3000 knots AND, for the
    # ~2/3 of them inside 6.25", `moffat_psf_integration`'s per-radius
    # 281x281 Cartesian aperture sum. Tens of seconds. Nothing in it depends
    # on the source, so per-exposure template building was paying it once per
    # exposure for what is at most a handful of distinct seeing values.
    # QUANTIZE d_max so the cache is actually reusable (2026-08-07). The raw
    # d_max moves by a fraction of an arcsec between exposures and between
    # injected sources -- different fibers, slightly different seeing -- so
    # keying on it exactly meant the cache missed most of the time and the
    # expensive V table got rebuilt anyway. Rounding UP to the next 10" makes
    # every source in a field with the same seeing share one table, at the
    # cost of tabulating a little further out than strictly needed (free: the
    # far knots are analytic).
    d_max = float(np.ceil((float(r_fiber.max() + s_shell.max()) + 1.0) / 10.0)
                  * 10.0)
    key = (id(response), round(float(np.ravel(seeing_fwhm)[0]), 4),
           d_max, n_phi)
    hit = _V_TAB_CACHE.get(key)
    if hit is None:
        d_tab = _separation_table(d_max)
        V_tab = np.asarray(response(d_tab, [seeing_fwhm])).ravel()
        if len(_V_TAB_CACHE) > 64:          # bounded; seeing values are few
            _V_TAB_CACHE.clear()
        _V_TAB_CACHE[key] = (d_tab, V_tab)
    else:
        d_tab, V_tab = hit

    out = np.empty((r_fiber.size, s_shell.size), dtype=float)
    chunk = max(1, int(4e7 / max(r_fiber.size * n_phi, 1)))
    for j0 in range(0, s_shell.size, chunk):
        s = s_shell[j0:j0 + chunk]
        d = np.sqrt(np.clip(
            r_fiber[:, None, None] ** 2 + s[None, :, None] ** 2
            - 2.0 * r_fiber[:, None, None] * s[None, :, None] * cosphi[None, None, :],
            0.0, None))
        out[:, j0:j0 + chunk] = np.interp(d, d_tab, V_tab).mean(axis=2)
    return out


def time_template_build(model, z, dra, ddec, seeing_fwhm, wave,
                        kpc_per_arcsec, verbose=True, **kw):
    """Cost breakdown for ONE exposure's template. Run this BEFORE the loop.

    Written 2026-08-07 after three separate timing failures in the same cell,
    each with a different cause and each found by staring at a traceback after
    30+ minutes of waiting. The useful question is not "is it slow" but
    "is the slow part PER-EXPOSURE work or SHARED work that should be cached",
    and that is one measurement: run it cold, then run it again warm.

        cold >> warm  ->  the cost is the shared PSF table + ring kernel. It
                          is paid once and the loop will be fast. If it is
                          NOT being paid once, something upstream is varying
                          per exposure (seeing is the usual culprit) and
                          breaking the cache key.
        cold ~= warm  ->  the cost is genuinely per-fiber, and the only levers
                          are fewer fibers or fewer wavelength pixels.

    Also prints the size that actually drives the ring kernel:
    n_r x n_shell x n_phi. That number is the interpolation count, and at
    production sizes it is ~7e7 -- if it comes back much larger than that,
    a grid bound has blown up and THAT is the bug, not the machine.
    """
    import time as _time

    clear_injection_caches()
    t0 = _time.time()
    T = build_source_template(model, z, dra, ddec, seeing_fwhm, wave,
                              kpc_per_arcsec, **kw)
    cold = _time.time() - t0

    t0 = _time.time()
    build_source_template(model, z, dra, ddec, seeing_fwhm, wave,
                          kpc_per_arcsec, **kw)
    warm = _time.time() - t0

    shapes = [np.shape(v) for v in _KERNEL_CACHE.values()]
    n_r, n_shell = (shapes[0] if shapes else (0, 0))
    info = {"cold_s": cold, "warm_s": warm, "shape": T.shape,
            "n_r": n_r, "n_shell": n_shell,
            "ring_interpolations": n_r * n_shell * 180}
    if verbose:
        print(f"template {T.shape}  {T.nbytes/1e6:.1f} MB  ({T.dtype})")
        print(f"  cold (empty caches) : {cold:7.2f} s")
        print(f"  warm (caches hot)   : {warm:7.2f} s   <- the per-exposure cost")
        print(f"  shared, paid once   : {cold - warm:7.2f} s")
        print(f"  ring kernel is {n_r} radii x {n_shell} shells x 180 phi "
              f"= {n_r*n_shell*180/1e6:.0f} M interpolations")
        if warm > 0.25 * cold and cold > 2.0:
            print("  [!] warm is not much cheaper than cold -- the cache is "
                  "NOT being reused. Check that seeing (and therefore the "
                  "cache key) is constant across exposures.")
    return info


# =====================================================================
# 4.  THE TRUTH GENERATOR -- 1-D radial quadrature, INDEPENDENT
# =====================================================================

def predict_binned_observables(model, z, bins_kpc, seeing_fwhm,
                               kpc_per_arcsec, lsf_kms=None,
                               n_fine=4000, n_v=1201, v_range=3000.0):
    """Expected per-bin flux, dv, sigma and skew. THE TRUTH.

    Everything in the recovery diagnostics compares against THIS, never
    against a config value (spec sec 3.5). Three effects it exists to
    capture, each of which looks exactly like pipeline bias if you compare
    against the injected parameters instead:

      (a) bin truth is the FLUX-WEIGHTED mean of dv over the bin AFTER PSF
          convolution, not dv(r_center);
      (b) a bin straddling a velocity break records a SUPERPOSITION and is
          therefore broader than the injected sigma;
      (c) that superposition manufactures SKEW from symmetric lines, so the
          recovered alpha is not the injected alpha -- and the difference is
          a prediction, not an error. Still modelled, no longer compared
          against anything: skew is out of scope as of 2026-08-16.

    THE LSF IS NOT APPLIED HERE ANY MORE (2026-08-16). It lives on the
    model, and `model.line_profile` returns the line as observed. This
    function used to broaden the line itself, which meant the LSF was set in
    two places -- and they disagreed: the injector applied none, so it
    emitted 250 km/s while this predicted 320 for the same source. The
    `lsf_kms` argument is kept only as a CONSISTENCY CHECK: pass a value and
    it is asserted against the model's rather than applied, so an old call
    site cannot silently double-blur the line.

    Implementation is deliberately unlike `build_source_template`: 1-D
    radial quadrature with the independent aperture response, rather than a
    2-D Cartesian shell convolution with the production interpolator. See
    the module docstring on why that independence is the whole point.

    The 200 kpc break falls inside the [180, 350] production bin, so that
    bin's truth is a flux-weighted BLEND of 0 and +100, not either pure
    value -- computed here rather than assigned (spec sec 4.2, criterion 1d).

    Returns
    -------
    dict with keys:
      r_edges, r_mid   : (nbin+1,), (nbin,) kpc
      flux             : (nbin,) total flux in the bin, PSF-smeared
      sb               : (nbin,) flux / annulus area
      dv               : (nbin,) 50%-flux centroid velocity, km/s
      sigma            : (nbin,) recovered rms width, km/s
      skew             : (nbin,) third standardized moment
      dv_naive         : (nbin,) dv(r_mid) -- the WRONG truth, returned so
                         the size of effect (a) is visible, never for use
      profile_frac     : (nbin,) fraction of total flux
    """
    # The LSF is the model's, always. A passed value is checked, not used --
    # see the docstring. Tolerance is 1 km/s: tight enough that a genuinely
    # different convention (0 vs 200, or FWHM vs sigma) cannot slip through,
    # loose enough not to trip on 199 vs 200.
    if lsf_kms is not None and abs(float(lsf_kms) - model.lsf_kms) > 1.0:
        raise ValueError(
            f"lsf_kms={float(lsf_kms):.1f} does not match the model's "
            f"{model.lsf_kms:.1f} km/s. The LSF lives on the SourceModel so "
            f"that the injector and the truth generator cannot disagree "
            f"about it. Set it there (SourceModel(lsf_kms=...)) rather than "
            f"here; this argument only checks."
        )

    bins_kpc = np.asarray(bins_kpc, dtype=float)
    r_edges = bins_kpc
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1])
    nbin = r_mid.size

    # --- observed-plane radial grid, in kpc ---
    r_obs = np.linspace(0.0, float(r_edges[-1]), n_fine)
    dr = r_obs[1] - r_obs[0]

    # --- source-plane shells ---
    if model.profile == "point":
        s_kpc = np.array([0.0])
        shell_flux = np.array([1.0])
    else:
        s_edges = np.linspace(0.0, float(r_edges[-1]) * 1.5, n_fine // 2 + 1)
        s_kpc = 0.5 * (s_edges[1:] + s_edges[:-1])
        ds = np.diff(s_edges)
        shell_flux = model.I(s_kpc) * 2 * np.pi * s_kpc * ds
        shell_flux = shell_flux / shell_flux.sum()
    shell_dv = model.velocity(s_kpc)

    # --- ring-spread: independent aperture response, in arcsec ---
    # CACHED 2026-08-20. This is the same ~140M-interpolation ring kernel that
    # build_source_template caches, and it was the ONLY uncached copy -- so
    # every call to the truth generator rebuilt it from scratch. That is
    # tolerable once and intolerable 11 times, which is exactly what building
    # the response matrix for the deconvolution needs. The kernel depends only
    # on the two radius grids, the seeing and the response, all of which are
    # fixed across a basis sweep, so caching turns 11 slow calls into 1.
    _t_key = (id(fiber_response_independent), round(float(seeing_fwhm), 4),
              round(float(kpc_per_arcsec), 6),
              (r_obs / kpc_per_arcsec).tobytes(),
              (s_kpc / kpc_per_arcsec).tobytes())
    K = _TRUTH_KERNEL_CACHE.get(_t_key)
    if K is None:
        K = _shell_to_fiber_kernel(r_obs / kpc_per_arcsec, s_kpc / kpc_per_arcsec,
                                   seeing_fwhm, fiber_response_independent)
        if len(_TRUTH_KERNEL_CACHE) > 4:      # each is large; keep a couple
            _TRUTH_KERNEL_CACHE.clear()
        _TRUTH_KERNEL_CACHE[_t_key] = K

    # --- velocity grid; every bin's line profile is built as a genuine
    #     superposition over the shells contributing to it ---
    v = np.linspace(-v_range, v_range, n_v)
    dvv = v[1] - v[0]
    # model.line_profile already returns the OBSERVED line (LSF applied), so
    # there is nothing to broaden here. The temporary sigma-swap that used to
    # sit at this spot is gone: mutating the model to fake a wider line is
    # exactly how the two generators drifted apart in the first place.
    uniq_dv = np.unique(shell_dv)
    lines = {d: model.line_profile(v, d) for d in uniq_dv}

    flux = np.zeros(nbin)
    dv_out = np.zeros(nbin)
    sig_out = np.zeros(nbin)
    skew_out = np.zeros(nbin)

    for b in range(nbin):
        insel = (r_obs >= r_edges[b]) & (r_obs < r_edges[b + 1])
        if not np.any(insel):
            continue
        # flux from each shell landing in this bin, weighted by 2 pi r dr
        w_ring = (K[insel, :] * (2 * np.pi * r_obs[insel] * dr)[:, None])
        per_shell = w_ring.sum(axis=0) * shell_flux         # (nshell,)
        tot = per_shell.sum()
        flux[b] = tot
        if tot <= 0:
            dv_out[b] = sig_out[b] = skew_out[b] = np.nan
            continue

        # the superposition -- effects (b) and (c) live here
        prof = np.zeros_like(v)
        for d in uniq_dv:
            m = shell_dv == d
            prof += per_shell[m].sum() * lines[d]
        prof /= prof.sum() * dvv

        # HALF-PIXEL CONVENTION -- see centroid_from_spectrum's docstring.
        # FIXED 2026-08-07. This was a bare `np.cumsum(prof) * dvv`, which
        # places the running total at the RIGHT EDGE of each velocity pixel
        # and therefore biases the 50%-flux point by half a pixel. The grid
        # here is 2*v_range/(n_v-1) = 5 km/s wide, so the truth generator was
        # reporting **-2.50 km/s where 0 was injected and 97.50 where 100
        # was**, uniformly in every bin.
        #
        # This is the SAME bug the recovery-side estimator documents at
        # length and fixes. Having it on only one side is the worst case:
        # a constant offset between injector and truth, identical at every
        # radius, so it does NOT look like a gradient artifact -- it looks
        # like a small clean systemic shift and gets waved through as
        # "agreement within errors". Level 0 did not catch it because its
        # line-profile check uses a 0.1 km/s grid (half-pixel = 0.05) and a
        # 1.0 km/s tolerance, so the bias was two orders below the threshold.
        cdf = (np.cumsum(prof) - 0.5 * prof) * dvv
        cdf /= cdf[-1]
        dv_out[b] = float(np.interp(0.5, cdf, v))           # 50%-flux centroid
        mean = float(np.sum(prof * v) * dvv)
        var = float(np.sum(prof * (v - mean) ** 2) * dvv)
        sig_out[b] = np.sqrt(max(var, 0.0))
        skew_out[b] = (float(np.sum(prof * (v - mean) ** 3) * dvv)
                       / max(var ** 1.5, 1e-30))

    areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    total = flux.sum()
    return {
        "r_edges": r_edges,
        "r_mid": r_mid,
        "flux": flux * model.L,
        "sb": flux * model.L / areas,
        "dv": dv_out,
        "sigma": sig_out,
        "skew": skew_out,
        "dv_naive": model.velocity(r_mid),
        "profile_frac": flux / total if total > 0 else flux,
    }


# =====================================================================
# 5.  LADDER LEVEL 0 + the two printed assertions the spec asks for
# =====================================================================

def bin_template_radially(templates, dists_arcsec, bins_kpc, kpc_per_arcsec,
                          method="mean", good=None, wave_block=128):
    """Radially bin a noiseless template the way `extract_fibers` bins fibers.

    This is the round-trip half of ladder levels 1-4: build the template on
    REAL fiber geometry, bin it, and compare against
    `predict_binned_observables`. No cube, no noise, no extraction runtime.

    METHOD MATTERS, AND "mean" IS THE RIGHT DEFAULT HERE.
    Production combines a bin's fibers with a BIWEIGHT, which is a robust
    estimator of the bin's TYPICAL fiber value. On real noisy data that is
    the right choice. But on a noiseless, steeply-falling template the
    biweight converges to the MEDIAN over the annulus, while the truth
    generator predicts the AREA-WEIGHTED MEAN -- and for a steep profile in
    a wide bin those differ by a lot. That difference is the estimator, not
    a pipeline bias, and mixing it into a geometry check would make level 1
    fail for a reason that has nothing to do with geometry.

    So: validate the GEOMETRY with method="mean", then quantify the
    biweight-vs-mean gap separately as its own diagnostic (the mean /
    median / biweight ladder). Do not skip the second step -- it feeds
    directly into what the recovered profile means.

    Parameters
    ----------
    templates : list over exposures of (nfib, nwave) arrays from
        `build_source_template`.
    dists_arcsec : list over exposures of (nfib,) or (nfib, nwave) distances.
        PASS THE 1-D FORM. Production bins on `dist_arcsec_all[e]`, a static
        per-fiber distance with NO DAR term (`extract.py` builds `bin_idx_all`
        once per fiber, not per wavelength), so the 1-D static distance is
        what matches it. A 2-D array is still accepted and reduced with a
        wavelength median, but that median sorts every fiber's 1036-pixel row
        and costs seconds per exposure to reproduce a number that differs
        from the static distance by the sub-arcsec DAR excursion.
    bins_kpc : the production edge list.
    method : "mean" | "median" | "biweight".
    good : optional list of (nfib,) boolean masks (True = usable).
    wave_block : wavelength pixels per gather for median/biweight. Caps peak
        memory; None means "all at once".

    MEMORY -- rewritten 2026-08-07 after this function restarted a kernel.
    The old version built `pools[r]` = a list of fancy-indexed COPIES of every
    exposure's fibers, cast to float64 on the way in. Summed over bins that is
    a complete float64 duplicate of the whole template set -- at 132 exposures
    x 4444 fibers x 1036 pixels that is **4.9 GB**, on top of the 2.4 GB of
    float32 templates already live, and then `np.vstack` copied the largest
    bin again. Over 10 GB peak.

    This now mirrors what `extract_fibers` already does for the real thing
    (`extract.py` section 3, "gather + combine ONE BIN AT A TIME"): assign
    bins once into a compact index array, then handle one bin at a time with
    an exactly-sized float32 pool that is freed before the next bin, in
    wavelength blocks. Peak RAM is set by one bin's block, not by everything
    at once -- ~200 MB instead of 10 GB. `method="mean"` does not build a pool
    at all; it streams a running sum and count.

    Returns
    -------
    (nrad, nwave) binned spectra, and (nrad,) fiber counts.
    """
    from astropy.stats import biweight_location

    bins_kpc = np.asarray(bins_kpc, dtype=float)
    nrad = bins_kpc.size - 1
    nexp = len(templates)
    nwave = np.asarray(templates[0]).shape[1]

    # ---- 1. bin index per fiber, once (production's bin_idx_all) ----
    idx_all = []
    counts = np.zeros(nrad, dtype=np.int64)
    for e in range(nexp):
        d = np.asarray(dists_arcsec[e], dtype=float)
        if d.ndim == 2:
            d = np.nanmedian(d, axis=1)
        idx = np.digitize(d * kpc_per_arcsec, bins_kpc) - 1
        if good is not None:
            idx[~np.asarray(good[e], dtype=bool)] = -1
        idx[(idx < 0) | (idx >= nrad)] = -1
        idx = idx.astype(np.int16)
        idx_all.append(idx)
        v = idx[idx >= 0]
        if v.size:
            counts += np.bincount(v, minlength=nrad).astype(np.int64)

    nfib = counts.astype(int)
    out = np.full((nrad, nwave), np.nan)

    # ---- 2a. mean: stream, never materialize a pool ----
    if method == "mean":
        acc = np.zeros((nrad, nwave))
        cnt = np.zeros((nrad, nwave))
        for e in range(nexp):
            tmpl = np.asarray(templates[e])
            idx = idx_all[e]
            for r in range(nrad):
                sel = idx == r
                if not sel.any():
                    continue
                blk = tmpl[sel]                 # one bin, one exposure only
                acc[r] += np.nansum(blk, axis=0)
                cnt[r] += np.isfinite(blk).sum(axis=0)
                del blk
        with np.errstate(invalid="ignore"):
            out = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), np.nan)
        return out, nfib

    # ---- 2b. median / biweight: one bin at a time, in wavelength blocks ----
    if method not in ("median", "biweight"):
        raise ValueError("method must be mean | median | biweight")
    wb = int(wave_block) if wave_block else nwave
    for r in range(nrad):
        n_r = int(counts[r])
        if n_r == 0:
            continue
        for w0 in range(0, nwave, wb):
            w1 = min(w0 + wb, nwave)
            pool = np.empty((n_r, w1 - w0), dtype=np.float32)
            off = 0
            for e in range(nexp):
                sel = idx_all[e] == r
                k = int(np.count_nonzero(sel))
                if k == 0:
                    continue
                pool[off:off + k] = np.asarray(templates[e])[sel, w0:w1]
                off += k
            if method == "median":
                out[r, w0:w1] = np.nanmedian(pool, axis=0)
            else:
                out[r, w0:w1] = biweight_location(pool, axis=0, ignore_nan=True)
            del pool                     # release before the next block
    return out, nfib


def centroid_from_spectrum(spec, wave, z, window_kms=1500.0):
    """50%-flux centroid velocity of one binned spectrum, in km/s.

    Matches the production convention (median-50%-flux centroid) so that a
    recovered value is directly comparable to `predict_binned_observables`
    output. Returns NaN if the window carries no positive flux.

    HALF-PIXEL CONVENTION -- this bit is not cosmetic.
    A bare `np.cumsum(f)` places the cumulative total at the RIGHT EDGE of
    each pixel, so interpolating it against the pixel-CENTER velocities
    biases the centroid by half a pixel. Measured on a 0.2 A grid at
    z = 2.4, that was a uniform -7.26 km/s across every radial bin -- half
    of the 14.5 km/s pixel. On the real 110-140 km/s VDFI grid the same
    error would be **55-70 km/s**, which is larger than the entire inner
    velocity decline this project is measuring, and it would be uniform
    across radius so it would NOT show up as a gradient artifact -- it would
    look like a clean systemic offset and quietly survive into the absolute
    zero point.

    Fixed by the midpoint convention: cdf_i = (cumsum_i - f_i/2) / total.

    Worth checking the production centroid estimator for the same thing --
    this is exactly the "centroid estimators are not obviously unbiased at
    that pixel scale" concern in spec sec 5.5, and it is cheap to test.
    """
    wave = np.asarray(wave, dtype=float)
    # RAW observed axis -> the line's AIR position, matching the injector.
    lam0 = observed_lya_wavelength(z)
    v = C_KMS * (wave - lam0) / lam0
    sel = np.isfinite(spec) & (np.abs(v) <= window_kms)
    if sel.sum() < 5:
        return np.nan
    f = np.clip(np.asarray(spec, dtype=float)[sel], 0.0, None)
    if f.sum() <= 0:
        return np.nan
    cum = np.cumsum(f)
    cdf = (cum - 0.5 * f) / cum[-1]        # midpoint, not right-edge
    return float(np.interp(0.5, cdf, v[sel]))


def check_level0(model, seeing_fwhm=1.8, kpc_per_arcsec=8.2, verbose=True):
    """Ladder level 0 (spec sec 7.2): no cube, analytic answers only.

    Catches generator unit bugs before anything is built on top. Five
    checks, in increasing order of what they would have caught:

      1. \\int I(r) 2 pi r dr = 1                  -- convention 1
      2. line profile integrates to 1, its 50%-flux centroid is the
         requested dv even for alpha != 0, AND its width is the
         instrument-blurred one                  -- the xi solve, the LSF
      3. the two fiber responses agree            -- convention 2/3, i.e.
         the entire 2*pi / aperture / normalization class (sec 7.1)
      4. V -> 1 as seeing -> 0 and r -> 0, and V is monotone decreasing in r
      5. the LSF closed form matches a brute-force convolution

    Check 3 is the load-bearing one. It is the only place where two
    independent implementations of the same quantity are compared, and per
    sec 3.5 that comparison is the only mechanism that would have caught
    the 2026-07-22 area-convention bug.
    """
    lines = []
    ok = True

    # --- 1. profile normalization ---
    if model.profile != "point":
        tot = _normalize_profile(model.I)
        good = abs(tot - 1.0) < 2e-3
        ok &= good
        lines.append(f"[{'PASS' if good else 'FAIL'}] "
                     f"int I(r) 2 pi r dr = {tot:.6f}  (want 1)")
    else:
        lines.append("[SKIP] profile normalization -- point source")

    # --- 2. line profile: area, centroid, AND observed width ---
    # The width check is new (2026-08-16) and it is what pins the LSF. The
    # line the generators emit must be the intrinsic line blurred by the
    # instrument -- sqrt(250^2 + 200^2) = 320.2, not 250. Tolerance 0.5%,
    # which is far tighter than the difference it exists to catch (a missing
    # LSF is 28% narrow; a FWHM/sigma mix-up is a factor of 2.35). A check
    # that cannot fail is not a check.
    v = np.linspace(-4000, 4000, 80001)
    om_want, al_want = model.observed_line
    # For a skewed line the rms is NOT omega: the skew steals some spread.
    # variance = omega^2 (1 - 2 delta^2 / pi). Reduces to omega at alpha = 0,
    # which is what both families run -- written in full so the check stays
    # valid if skew is ever revived rather than silently failing then.
    _d_want = al_want / np.sqrt(1.0 + al_want ** 2)
    sig_want = om_want * np.sqrt(1.0 - 2.0 * _d_want ** 2 / np.pi)
    for dv_test in (0.0, -50.0, 100.0):
        S = model.line_profile(v, dv_test)
        area = np.trapz(S, v)
        rms = float(np.sqrt(np.trapz(S * (v - np.trapz(S * v, v)) ** 2, v)))
        good_w = abs(rms - sig_want) / sig_want < 5e-3
        ok &= good_w
        lines.append(f"[{'PASS' if good_w else 'FAIL'}] "
                     f"line dv={dv_test:+7.1f}: observed width={rms:7.2f} km/s "
                     f"(want {sig_want:.2f}; omega = sqrt("
                     f"{model.sigma_kms:.0f}^2 + {model.lsf_kms:.0f}^2) = "
                     f"{om_want:.1f})")
        # midpoint cdf, matching centroid_from_spectrum and (as of
        # 2026-08-07) predict_binned_observables. The tolerance below is
        # tightened 1.0 -> 0.05 km/s at the same time: at this grid the
        # half-pixel bias is 0.05 km/s, so a 1.0 km/s tolerance could never
        # have detected the convention error that was live in the truth
        # generator for a week. A check that cannot fail is not a check.
        cdf = (np.cumsum(S) - 0.5 * S) * (v[1] - v[0])
        cdf /= cdf[-1]
        med = float(np.interp(0.5, cdf, v))
        good = abs(area - 1.0) < 1e-4 and abs(med - dv_test) < 0.05
        ok &= good
        lines.append(f"[{'PASS' if good else 'FAIL'}] "
                     f"line dv={dv_test:+7.1f}: area={area:.6f}, "
                     f"50%-flux centroid={med:+8.2f} km/s")

    # --- 3. THE ONE THAT MATTERS: two independent fiber responses ---
    # PROBED ONLY WITHIN PSF_GRID_VALID_ARCSEC (fixed 2026-08-16). The old
    # r_test ran out to 8", past the radius where the production Cartesian
    # grid still has pixels inside the fiber aperture -- there `adj` divides
    # by a zero pixel count and V comes back NaN (the trap logged 2026-08-04).
    # `np.nanmax` then IGNORED that NaN, so the check reported PASS on a
    # comparison it had not actually made. A NaN is now an explicit failure,
    # and the probe stays inside the range production is defined on, which is
    # also the only range `fiber_response_hybrid` ever calls it in.
    r_test = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0])
    r_test = r_test[r_test < PSF_GRID_VALID_ARCSEC]
    Vp = np.asarray(fiber_response_production(r_test, [seeing_fwhm])).ravel()
    Vi = np.asarray(fiber_response_independent(r_test, [seeing_fwhm])).ravel()
    n_bad = int(np.sum(~np.isfinite(Vp)))
    rel = np.abs(Vp - Vi) / np.maximum(np.abs(Vi), 1e-12)
    sel3 = Vi > 1e-6
    worst3 = float(np.nanmax(rel[sel3])) if sel3.any() else np.nan
    good = (n_bad == 0) and np.isfinite(worst3) and worst3 < 0.02
    ok &= good
    lines.append(f"[{'PASS' if good else 'FAIL'}] fiber response, production "
                 f"vs independent: max rel diff = {worst3:.2e}"
                 + (f"   <-- {n_bad} NON-FINITE production value(s)"
                    if n_bad else ""))
    for r, a, b in zip(r_test, Vp, Vi):
        lines.append(f"           r={r:5.2f}\"  production={a:.6f}  "
                     f"independent={b:.6f}  ratio={a / max(b, 1e-30):.5f}")

    # --- 4. limiting behaviour ---
    V0 = float(np.asarray(fiber_response_independent([0.0], [0.05])).ravel()[0])
    mono = np.all(np.diff(Vi) < 0)
    good = V0 > 0.98 and mono
    ok &= good
    lines.append(f"[{'PASS' if good else 'FAIL'}] V(0, seeing->0) = {V0:.5f} "
                 f"(want ~1); V monotone decreasing: {mono}")

    # --- 5. the LSF closed form vs a brute-force convolution ---
    # Same principle as check 3: `apply_lsf` claims that blurring a skewed
    # Gaussian gives another skewed Gaussian with known parameters. That is a
    # claim, so it gets checked against an independent route -- actually
    # convolving the two curves numerically. Run at alpha = 2 as well as the
    # model's own alpha, because at alpha = 0 the check is nearly trivial and
    # would not notice if the skew transformation were dropped.
    if model.lsf_kms > 0:
        # GRID + FFT, NOT np.convolve (fixed 2026-08-16, same day it was added).
        # np.convolve is a DIRECT convolution: cost is n^2, so at the 160001
        # points this first used it was 2.6e10 multiply-adds -- ~5 s per call,
        # six calls, and it turned a cell that ran in seconds into one that ran
        # for minutes. An FFT computes the identical quantity in n log n. 16001
        # points at 1 km/s spacing resolves a 200 km/s kernel ~200x over, which
        # is far finer than the 1e-6 this check is compared at.
        vv = np.linspace(-8000.0, 8000.0, 16001)
        dvv = vv[1] - vv[0]
        worst = 0.0
        for al_in in sorted({model.alpha, 2.0}):
            om_in = model.sigma_kms
            t = vv / om_in
            intrinsic = (2.0 * np.exp(-0.5 * t ** 2) / (om_in * np.sqrt(2 * np.pi))
                         * 0.5 * (1.0 + erf(al_in * t / np.sqrt(2.0))))
            kern = np.exp(-0.5 * (vv / model.lsf_kms) ** 2)
            kern /= kern.sum() * dvv
            brute = np.fft.irfft(np.fft.rfft(intrinsic)
                                 * np.fft.rfft(np.fft.ifftshift(kern)),
                                 n=vv.size) * dvv
            om_o, al_o = apply_lsf(om_in, al_in, model.lsf_kms)
            t2 = vv / om_o
            closed = (2.0 * np.exp(-0.5 * t2 ** 2) / (om_o * np.sqrt(2 * np.pi))
                      * 0.5 * (1.0 + erf(al_o * t2 / np.sqrt(2.0))))
            # both are centred on the same location parameter (xi = 0), so
            # this compares SHAPE, which is what apply_lsf claims about
            worst = max(worst, float(np.max(np.abs(brute - closed))
                                     / np.max(closed)))
        good = worst < 1e-6
        ok &= good
        lines.append(f"[{'PASS' if good else 'FAIL'}] LSF closed form vs "
                     f"brute-force convolution: max rel diff = {worst:.2e}")
    else:
        lines.append("[SKIP] LSF closed form -- lsf_kms = 0")

    if verbose:
        print(f"--- ladder level 0: {model!r}")
        print("\n".join(lines))
        print(f"--- level 0 {'PASSED' if ok else 'FAILED'}")
    return ok, lines


def _pick_covered_position(RA, DEC, rng=None, r_inner_arcsec=4.0):
    """A field position with real fiber coverage in the innermost bins.

    The fill factor is not 100%, and an injection landing in an IFU gap must
    be discarded, not silently extracted as NaN -- so a position has to be
    checked for coverage rather than assumed.

    ONE PASS OVER THE FIBERS, via a 2-D histogram (rewritten 2026-08-16).
    The previous version drew 300 trial positions and, for each, tested every
    fiber: at COSMOS's 132 x 34944 = 4.6 M fibers that is ~1.4e9 comparisons,
    about 30 seconds. It also sat inside the caller's elapsed-time counter,
    so it inflated the reported per-exposure cost by 13x and the ETA with it.

    A histogram answers the same question -- "where are the fibers densest?"
    -- by touching each fiber once instead of 300 times. Cells are sized to
    the search radius, so the densest cell's centre is a well-covered
    position by construction, and the exact count is then computed once for
    that single position rather than for every candidate.
    """
    ra_f = np.asarray(RA).ravel()
    dec_f = np.asarray(DEC).ravel()
    ok = np.isfinite(ra_f) & np.isfinite(dec_f)
    ra_f, dec_f = ra_f[ok], dec_f[ok]
    if ra_f.size == 0:
        raise RuntimeError("no finite fiber positions in the field")

    dec0 = float(np.median(dec_f))
    cs = np.cos(np.radians(dec0))
    cell = r_inner_arcsec / 3600.0            # ~4" cells
    ra_lo, ra_hi = np.nanpercentile(ra_f, [2, 98])
    dec_lo, dec_hi = np.nanpercentile(dec_f, [2, 98])
    nx = max(int((ra_hi - ra_lo) * cs / cell), 1)
    ny = max(int((dec_hi - dec_lo) / cell), 1)

    Hc, xe, ye = np.histogram2d(ra_f, dec_f, bins=(nx, ny),
                                range=((ra_lo, ra_hi), (dec_lo, dec_hi)))
    i, j = np.unravel_index(int(np.argmax(Hc)), Hc.shape)
    ra_i = float(0.5 * (xe[i] + xe[i + 1]))
    dec_i = float(0.5 * (ye[j] + ye[j + 1]))

    # exact count at the chosen position, once
    box = r_inner_arcsec / 3600.0
    m = (np.abs(dec_f - dec_i) < box) & (np.abs(ra_f - ra_i) * cs < box)
    d = np.hypot((ra_f[m] - ra_i) * cs, dec_f[m] - dec_i) * 3600.0
    n_in = int(np.sum(d < r_inner_arcsec))
    if n_in == 0:
        raise RuntimeError(
            "densest histogram cell still has no fibers within "
            f"{r_inner_arcsec}\" -- the field is sparser than expected")
    return ra_i, dec_i, n_in


def l_inj_for_luminosity(L_phys, z, line_is_velocity_density=True):
    """`L_INJ` giving a source of Lya luminosity `L_phys` (erg/s) at redshift z.

    Spec sec 2.4 asks for L to be bootstrapped alongside z, which only means
    anything once L_INJ can be tied to a physical luminosity. This is that
    tie, and it is the ONLY place the conversion should live.

    Why it is needed even for a fixed-brightness test: `total_flux_fid` is a
    LUMINOSITY surface density, so at fixed `L_INJ` a source at z = 2.5 reads
    2.16x brighter than one at z = 2.0. A fixed-L_INJ sample drawn over
    z = 2.0-2.5 is therefore inhomogeneous in luminosity by more than a
    factor of two and its stack is dominated by the high-z end. Fixing the
    PHYSICAL luminosity and deriving L_INJ per source is what makes every
    injected source contribute equally, which in turn is what makes one
    `predict_binned_observables` call at the mean z a valid stack truth.

    line_is_velocity_density : the `c / lambda_obs` term, and the flag exists
        because it corrects a REAL BUG rather than a convention choice.
        `SourceModel.line_profile` has unit integral in VELOCITY (km/s), and
        `build_source_template` multiplies it straight into a cube whose
        values are a flux density per OBSERVED ANGSTROM. So the flux actually
        injected is `lambda_obs / c` times L_INJ -- 72.5x smaller at z = 2.4,
        and z-dependent, which is the part that matters: it puts a
        REDSHIFT-DEPENDENT scale error into any family with a spread in z.

        Traced 2026-08-19 by predicting `total_flux_fid / L_INJ` from first
        principles: the measured constant goes as (1+z)^5 and a genuine
        wavelength density would give (1+z)^4, with the closed form
        5.719e35 * (1+z)^5 matching the data to 0.1% at z = 2.40.

        Leave True while `build_source_template` is unfixed. Once it
        multiplies its template by `C_KMS / lam0_obs`, set this False (or
        change the default) and L_INJ becomes simply "total observed flux in
        cube units", i.e. x1e-17 erg/s/cm^2.

    Ladder levels 0-6 are unaffected by that bug either way -- every rung is
    a shape, a velocity, or a comparison against the same generator, and a
    constant scale on L cancels in all of them. What it breaks is criterion
    1a (the sensitivity statement, which needs physical units) and the sec
    2.4 luminosity bootstrap.
    """
    from astropy.cosmology import Planck18 as _cosmo
    import astropy.units as _u
    z = np.asarray(z, dtype=float)
    DL = _cosmo.luminosity_distance(z).to(_u.cm).value
    L = np.asarray(L_phys, dtype=float) / (4.0 * np.pi * DL ** 2) / 1e-17
    if line_is_velocity_density:
        L = L * C_KMS / (LYA_REST * (1.0 + z))
    return L


# =====================================================================
#  2.3  Placement footprint -- the concave hull of the real catalog
# =====================================================================

class SourceFootprint:
    """The sky region a real catalog actually covers -- an ALPHA SHAPE.

    Spec sec 2.3, v8 (Austin, 2026-08-25). The old footprint was each field's
    RA/Dec min-max box. That box is wrong in a way that matters: MOSDEF is a
    SLIT-MASK survey, so its coverage is a union of a few rectangular mask
    pointings with concave notches between them, and the bounding box
    interpolates straight across every one of them. Randoms drawn in the box
    land in sky MOSDEF never observed, which is precisely the failure the
    randoms test exists to detect -- the null then measures the VDFI field's
    mean rather than the mean under the real sample's selection.

    THE CONSTRUCTION. Roll a probe circle of radius `alpha_arcsec` around the
    outside of the point cloud. It pushes into any gap wider than 2*alpha, so
    real notches in the survey edge get followed; it cannot enter a narrower
    one, so the interior stays solid. Formally: Delaunay-triangulate the
    positions and keep every triangle whose CIRCUMRADIUS is <= alpha. As
    alpha -> infinity every triangle survives and the shape degenerates to the
    convex hull, i.e. back to the big interpolating box; as alpha -> 0 the
    shape evaporates. This is the one parameter, and it has an unambiguous
    physical reading in arcsec.

    THE BUFFER. The alpha shape's boundary passes exactly THROUGH the
    outermost galaxies, so on its own it says MOSDEF observed no sky beyond
    its most peripheral target -- false, and it truncates the outer annuli of
    any random placed near the edge. `buffer_arcsec` fixes that by taking the
    union with a disk of that radius around every source. The union is taken,
    not a global dilation, so the buffer rounds the OUTSIDE without filling
    the interior notches back in.

    WHY NOT A UNION OF DISKS ALONE. Because then every random sits within
    `buffer` of a real galaxy by construction, the random catalog inherits the
    real sample's clustering, and sec 5.4's "far from any real source" slice --
    which is how the `blank` placement mode gets answered for free -- is empty.
    The alpha shape carries the geometry; the buffer only softens its edge.

    Attributes
    ----------
    alpha_arcsec, buffer_arcsec : the two radii, in arcsec.
    area_arcmin2 : Monte-Carlo area of the footprint (buffer included).
    area_alpha_arcmin2 : the same with the buffer switched off. Reported
                   alongside because the buffer hides what alpha is doing --
                   at a small enough alpha the shape is falling apart while
                   the buffered area barely moves.
    fill_frac    : area / bounding-box area, i.e. the acceptance rate a
                   rejection sampler drawing in `bbox` will see.
    n_src        : number of finite catalog positions the shape was built on.
    """

    def __init__(self, ra, dec, *, alpha_arcsec, buffer_arcsec, tri, keep,
                 tree, ra0, dec0, cs, area_arcmin2, fill_frac,
                 area_alpha_arcmin2=np.nan):
        self.ra = ra
        self.dec = dec
        self.alpha_arcsec = float(alpha_arcsec)
        self.buffer_arcsec = float(buffer_arcsec)
        self.area_arcmin2 = float(area_arcmin2)
        self.area_alpha_arcmin2 = float(area_alpha_arcmin2)
        self.fill_frac = float(fill_frac)
        self.n_src = int(ra.size)
        self._tri = tri
        self._keep = keep
        self._tree = tree
        self._ra0 = float(ra0)
        self._dec0 = float(dec0)
        self._cs = float(cs)

    # -- tangent-plane helpers ------------------------------------------
    def _xy(self, ra, dec):
        """(ra, dec) in degrees -> local tangent-plane arcsec about the centre."""
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        x = (ra - self._ra0) * self._cs * 3600.0
        y = (dec - self._dec0) * 3600.0
        return x, y

    @property
    def bbox(self):
        """(ra_lo, ra_hi, dec_lo, dec_hi) in degrees, buffer included.

        The rectangle a rejection sampler should draw in. Buffer is added in
        arcsec on both axes, with the RA one divided by cos(dec) so it is a
        true angular margin rather than a coordinate one.
        """
        pad_dec = self.buffer_arcsec / 3600.0
        pad_ra = pad_dec / self._cs
        return (float(self.ra.min()) - pad_ra, float(self.ra.max()) + pad_ra,
                float(self.dec.min()) - pad_dec, float(self.dec.max()) + pad_dec)

    def contains(self, ra, dec):
        """Boolean mask: is each (ra, dec) inside the footprint?

        Vectorised, and the only membership test anything should use. Scalars
        in give a 0-d array out, so `bool(fp.contains(r, d))` works.
        """
        x, y = self._xy(ra, dec)
        p = np.column_stack([np.atleast_1d(x).ravel(),
                             np.atleast_1d(y).ravel()])
        inside = np.zeros(p.shape[0], dtype=bool)
        s = self._tri.find_simplex(p)
        ok = s >= 0
        if ok.any():
            inside[ok] = self._keep[s[ok]]
        if self.buffer_arcsec > 0:
            d, _ = self._tree.query(p)
            inside |= (d <= self.buffer_arcsec)
        return inside.reshape(np.shape(x))

    def __repr__(self):
        return (f"SourceFootprint(n_src={self.n_src}, "
                f"alpha={self.alpha_arcsec:.1f}\", "
                f"buffer={self.buffer_arcsec:.1f}\", "
                f"area={self.area_arcmin2:.1f} arcmin^2, "
                f"fill={self.fill_frac:.2f})")


def _triangle_circumradii(pts):
    """Circumradius of every triangle in `pts`, shape (nt, 3, 2). R = abc/4A.

    Collinear triangles (A = 0) get inf, which the alpha filter then rejects --
    the right answer, since a sliver's circumcircle is unbounded.
    """
    a = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    b = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    c = np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1)
    u = pts[:, 1] - pts[:, 0]
    v = pts[:, 2] - pts[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(area > 0, a * b * c / (4.0 * np.maximum(area, 1e-300)),
                     np.inf)
    return R


def build_source_footprint(ra, dec, *, alpha_arcsec=None, alpha_scale=3.0,
                           buffer_arcsec=15.0, area_samples=200_000,
                           area_seed=0, verbose=True):
    """Spec sec 2.3. Alpha-shape footprint of a real catalog's positions.

    Pass the RA/Dec of the REAL run table (`extract.build_run_table(cfg)`
    sliced by `cfg.galaxy_slice()`) -- the same rows the real stack uses, so
    the randoms inherit exactly that selection's geometry.

    alpha_arcsec : probe radius. None (default) sets it to `alpha_scale` x the
        MEDIAN NEAREST-NEIGHBOUR separation of the catalog, which is the only
        scale the data itself supplies. 3x is loose enough that MOSDEF's own
        sparse interior patches stay inside the footprint -- they were
        observed, they just held few targets, and punching holes there would
        make the randoms avoid the real sample's low-density regions. Sweep it
        with `plot_source_footprint` before trusting the default on a new
        catalog; it is a geometry choice and it should be eyeballed once.
    buffer_arcsec : outward margin, unioned as a disk per source. 15" is ~4%
        of the 2000 kpc (242") outer aperture -- enough that the boundary does
        not slice through the outermost galaxies, small enough that it cannot
        manufacture coverage.

    Returns a `SourceFootprint`. See that class for what the shape is and why
    it is not a bounding box.
    """
    ra = np.asarray(ra, dtype=float).ravel()
    dec = np.asarray(dec, dtype=float).ravel()
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[ok], dec[ok]
    if ra.size < 3:
        raise ValueError(f"need at least 3 finite positions to build a "
                         f"footprint, got {ra.size}")

    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))
    cs = float(np.cos(np.radians(dec0)))
    x = (ra - ra0) * cs * 3600.0
    y = (dec - dec0) * 3600.0
    pts = np.column_stack([x, y])

    tree = cKDTree(pts)
    nn = tree.query(pts, k=2)[0][:, 1]
    nn_med = float(np.median(nn))
    if alpha_arcsec is None:
        alpha_arcsec = float(alpha_scale) * nn_med

    tri = Delaunay(pts)
    keep = _triangle_circumradii(pts[tri.simplices]) <= float(alpha_arcsec)
    if not keep.any():
        raise RuntimeError(
            f"alpha = {alpha_arcsec:.1f}\" keeps no triangles (median "
            f"nearest-neighbour separation is {nn_med:.1f}\"); the shape "
            f"would be empty. Raise alpha_scale.")

    fp = SourceFootprint(ra, dec, alpha_arcsec=alpha_arcsec,
                         buffer_arcsec=buffer_arcsec, tri=tri, keep=keep,
                         tree=tree, ra0=ra0, dec0=dec0, cs=cs,
                         area_arcmin2=np.nan, fill_frac=np.nan)

    # Monte-Carlo area. The buffer union has no closed form worth writing, and
    # the acceptance rate is the number the sampler actually needs anyway.
    rng = np.random.default_rng(int(area_seed))
    r_lo, r_hi, d_lo, d_hi = fp.bbox
    cand_ra = rng.uniform(r_lo, r_hi, int(area_samples))
    cand_dec = rng.uniform(d_lo, d_hi, int(area_samples))
    fill = float(np.mean(fp.contains(cand_ra, cand_dec)))
    buf, fp.buffer_arcsec = fp.buffer_arcsec, 0.0
    fill_alpha = float(np.mean(fp.contains(cand_ra, cand_dec)))
    fp.buffer_arcsec = buf
    box_arcmin2 = ((r_hi - r_lo) * cs * 60.0) * ((d_hi - d_lo) * 60.0)
    fp.fill_frac = fill
    fp.area_arcmin2 = fill * box_arcmin2
    fp.area_alpha_arcmin2 = fill_alpha * box_arcmin2

    if verbose:
        hull_frac = float(keep.sum()) / float(keep.size)
        print(f"    footprint: {fp.n_src} sources, median NN {nn_med:.1f}\", "
              f"alpha {alpha_arcsec:.1f}\" ({alpha_arcsec / nn_med:.1f}x NN), "
              f"buffer {buffer_arcsec:.1f}\"")
        print(f"    area {fp.area_arcmin2:.1f} arcmin^2 "
              f"({fp.area_arcmin2 / 3600.0:.4f} deg^2, "
              f"{fp.area_alpha_arcmin2:.1f} before the buffer) | "
              f"{fill:.1%} of its bounding box | "
              f"kept {keep.sum()}/{keep.size} Delaunay triangles "
              f"({1 - hull_frac:.1%} of the convex hull carved away)")
    return fp


def plot_source_footprint(ra, dec, *, alpha_scales=(1.0, 1.5, 2.0, 3.0, 5.0),
                          buffer_arcsec=15.0, draws=0, draw_seed=0,
                          figsize=None, verbose=True):
    """Sweep the alpha-shape probe radius and look at what each one keeps.

    The default `alpha_scale = 3` in `build_source_footprint` is a guess about
    a survey's geometry, and geometry guesses should be looked at rather than
    believed. One panel per scale: kept triangles filled, sources on top, area
    and bounding-box fill in the title. Pick the smallest scale whose shape
    still looks like the mask pattern rather than like swiss cheese.

    draws : if > 0, also scatter that many accepted random positions per panel,
        which is the actual question ("where would my randoms land?") rather
        than a proxy for it.

    Returns the list of `SourceFootprint`s, one per scale, so the chosen one
    can be reused instead of rebuilt.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    scales = list(alpha_scales)
    n = len(scales)
    if figsize is None:
        figsize = (4.0 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    fps = []
    for ax, sc in zip(axes, scales):
        fp = build_source_footprint(ra, dec, alpha_scale=sc,
                                    buffer_arcsec=buffer_arcsec,
                                    verbose=False)
        fps.append(fp)
        pts = fp._tri.points
        polys = pts[fp._tri.simplices[fp._keep]]
        ax.add_collection(PolyCollection(polys, facecolors="C0", alpha=0.25,
                                         edgecolors="C0", linewidths=0.2))
        ax.plot(pts[:, 0], pts[:, 1], ".k", ms=2.0, zorder=3)
        if draws > 0:
            rng = np.random.default_rng(int(draw_seed))
            r_lo, r_hi, d_lo, d_hi = fp.bbox
            got_r, got_d = [], []
            while len(got_r) < int(draws):
                cr = rng.uniform(r_lo, r_hi, int(draws) * 3)
                cd = rng.uniform(d_lo, d_hi, int(draws) * 3)
                m = fp.contains(cr, cd)
                got_r.extend(cr[m][: int(draws) - len(got_r)])
                got_d.extend(cd[m][: int(draws) - len(got_d)])
            gx, gy = fp._xy(np.asarray(got_r), np.asarray(got_d))
            ax.plot(gx, gy, ".", color="C3", ms=2.5, zorder=4, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"alpha = {sc:g}x NN = {fp.alpha_arcsec:.0f}\"\n"
                     f"{fp.area_alpha_arcmin2:.0f} arcmin$^2$ "
                     f"(+buffer {fp.area_arcmin2:.0f}), "
                     f"{fp.fill_frac:.0%} of box", fontsize=9)
        ax.set_xlabel('$\\Delta$RA (")')
    axes[0].set_ylabel('$\\Delta$Dec (")')
    fig.tight_layout()
    if verbose:
        print("    shaded = alpha shape only; the buffer is what the +column "
              "adds. Watch the SHADED area, not the buffered one -- the "
              "buffer hides a shape that is coming apart.")
        for sc, fp in zip(scales, fps):
            print(f"    {sc:>4g}x NN -> alpha {fp.alpha_arcsec:6.1f}\"  "
                  f"alpha-only {fp.area_alpha_arcmin2:7.1f}  "
                  f"+buffer {fp.area_arcmin2:7.1f} arcmin^2  "
                  f"fill {fp.fill_frac:5.1%}")
    return fps


# =====================================================================
#  2.4  Redshift sampling -- the real n(z), not a uniform slab
# =====================================================================

def find_redshift_humps(z, *, gap_dz=0.15, min_frac=0.02, pad_dz=0.0,
                        verbose=True):
    """Split a real redshift distribution into contiguous humps at its gaps.

    MOSDEF targets two rest-optical windows -- the strong lines have to land
    between the OH forest and the atmospheric bands -- so its n(z) is two
    clumps near z ~ 2.0-2.6 and z ~ 3.0-3.6 with a hole between them that
    contains essentially no galaxies. A random catalog drawn uniformly across
    the full span puts a third of its sources in that hole, at observed
    wavelengths the real stack never samples. That matters because the
    systematic the null is trying to size -- the sky residual -- is
    WAVELENGTH-dependent: randoms spread over wavelengths the data never sees
    average it away, and the null then reads clean for a reason that has
    nothing to do with the data being clean.

    Humps are found from the data, not hardcoded: sort z, cut wherever
    consecutive redshifts differ by more than `gap_dz`, and keep every run
    holding at least `min_frac` of the sample. Stragglers below that threshold
    are dropped and reported -- a lone galaxy at z = 4.1 is not a hump, and
    stretching a hump out to reach it would reintroduce the empty span.

    Returns a list of (z_lo, z_hi, n, weight), z-ordered.
    """
    z = np.asarray(z, dtype=float).ravel()
    z = np.sort(z[np.isfinite(z)])
    if z.size == 0:
        raise ValueError("no finite redshifts")
    n_min = max(1, int(np.ceil(float(min_frac) * z.size)))
    cuts = np.nonzero(np.diff(z) > float(gap_dz))[0]
    runs = np.split(np.arange(z.size), cuts + 1)

    humps, n_drop = [], 0
    for ii in runs:
        if ii.size < n_min:
            n_drop += int(ii.size)
            continue
        humps.append([float(z[ii[0]]) - pad_dz, float(z[ii[-1]]) + pad_dz,
                      int(ii.size)])
    if not humps:
        raise RuntimeError(
            f"no run of >= {n_min} galaxies survives gap_dz = {gap_dz}; the "
            f"sample is too sparse for hump detection -- lower min_frac, "
            f"raise gap_dz, or use mode='bootstrap'.")

    tot = float(sum(h[2] for h in humps))
    out = [(h[0], h[1], h[2], h[2] / tot) for h in humps]
    if verbose:
        print(f"    n(z): {len(out)} hump(s) from {z.size} real redshifts"
              + (f", {n_drop} straggler(s) dropped" if n_drop else ""))
        for lo, hi, k, w in out:
            print(f"      z {lo:.4f}-{hi:.4f}  n = {k:4d}  weight {w:.3f}")
    return out


def _make_z_sampler(rng, *, z_sample=None, z_range=None, z_mode=None,
                    gap_dz=0.15, min_frac=0.02, pad_dz=0.0, verbose=True):
    """Return (draw_z(), span, label) for `draw_injection_positions`.

    `span` is the TOTAL redshift measure the sampler can reach -- the summed
    hump width, not (z_hi - z_lo) -- because that is what the min_dz exclusion
    has to be compared against. Getting this wrong is how a two-hump sample
    with a wide nominal range silently degenerates into a purely spatial
    exclusion.

    `z_mode=None` resolves to 'hump' when a real sample was handed over and to
    'uniform' when it was not, so a pre-v8 call that passes only `z_range`
    keeps its old behaviour instead of erroring on a mode it never asked for.
    """
    if z_mode is None:
        z_mode = "hump" if z_sample is not None else "uniform"
    mode = str(z_mode).lower()
    if z_sample is None:
        if mode != "uniform":
            raise ValueError(
                f"z_mode = {mode!r} needs z_sample (the real run table's "
                f"redshifts); pass z_mode='uniform' to keep the legacy "
                f"uniform-over-z_range draw.")
        if z_range is None:
            raise ValueError("z_mode='uniform' with no z_sample needs z_range")
        lo, hi = float(z_range[0]), float(z_range[1])
        if verbose:
            print(f"    z: uniform over {lo:.3f}-{hi:.3f} (no real n(z) "
                  f"matched -- randoms will sample wavelengths the data does "
                  f"not)")
        return (lambda: float(rng.uniform(lo, hi))), (hi - lo), "uniform"

    zs = np.asarray(z_sample, dtype=float).ravel()
    zs = zs[np.isfinite(zs)]
    n_all = zs.size
    if z_range is not None:
        zs = zs[(zs >= float(z_range[0])) & (zs <= float(z_range[1]))]
    if zs.size == 0:
        raise ValueError("z_sample has no finite redshifts inside z_range")
    if verbose and zs.size < n_all:
        # Worth saying out loud: passing BOTH z_sample and z_range means the
        # range wins, so the windows are the real sample's *clipped* by it and
        # the hump weights move with the clip. That is a legitimate thing to
        # want (keeping Lya off the ends of the VIRUS grid) but it is not the
        # real n(z) any more, and the difference is invisible in the output.
        print(f"    [!] z_range clips {n_all - zs.size}/{n_all} real "
              f"redshifts; the windows below are the CLIPPED distribution, "
              f"not the sample's own. Pass z_range=None to keep it intact.")

    if mode == "bootstrap":
        if verbose:
            print(f"    z: bootstrapped from {zs.size} real redshifts "
                  f"({zs.min():.3f}-{zs.max():.3f})")
        return (lambda: float(rng.choice(zs))), float(zs.max() - zs.min()), \
               "bootstrap"

    if mode == "uniform":
        lo, hi = (float(zs.min()), float(zs.max())) if z_range is None else \
                 (float(z_range[0]), float(z_range[1]))
        if verbose:
            print(f"    z: uniform over {lo:.3f}-{hi:.3f}")
        return (lambda: float(rng.uniform(lo, hi))), (hi - lo), "uniform"

    if mode != "hump":
        raise ValueError(f"z_mode must be 'hump', 'bootstrap' or 'uniform', "
                         f"got {z_mode!r}")

    humps = find_redshift_humps(zs, gap_dz=gap_dz, min_frac=min_frac,
                                pad_dz=pad_dz, verbose=verbose)
    lo_a = np.array([h[0] for h in humps])
    hi_a = np.array([h[1] for h in humps])
    w_a = np.array([h[3] for h in humps])
    w_a = w_a / w_a.sum()
    span = float(np.sum(hi_a - lo_a))

    def draw():
        k = int(rng.choice(lo_a.size, p=w_a))
        return float(rng.uniform(lo_a[k], hi_a[k]))

    return draw, span, "hump"


def sample_redshifts(n, z_sample=None, *, mode="hump", seed=0, z_range=None,
                     gap_dz=0.15, min_frac=0.02, pad_dz=0.0, verbose=True):
    """Spec sec 2.4. Draw `n` redshifts matched to a real sample's n(z).

    Standalone version of what `draw_injection_positions` does internally, for
    plotting the drawn n(z) against the real one before committing 13 hr of
    extraction to it.

    mode
      'hump'      (default) uniform inside each detected hump, weighted by the
                  real counts. Reproduces WHICH wavelengths the survey reaches
                  and in what proportion, while staying smooth inside each
                  window -- so the randoms do not inherit the real sample's
                  own small-number structure in z.
      'bootstrap' resample the real redshifts with replacement. Matches n(z)
                  exactly, including every spike; what spec sec 2.4 originally
                  asked for.
      'uniform'   legacy flat draw over `z_range`. Records what the pre-v8
                  runs did; do not use for a null.
    """
    rng = np.random.default_rng(int(seed))
    draw, _, _ = _make_z_sampler(rng, z_sample=z_sample, z_range=z_range,
                                 z_mode=mode, gap_dz=gap_dz,
                                 min_frac=min_frac, pad_dz=pad_dz,
                                 verbose=verbose)
    return np.array([draw() for _ in range(int(n))])


def draw_injection_positions(vdfi, n, z_range=None, *, seed=0,
                             min_sep_arcsec=50.0, min_dz=0.01,
                             r_inner_arcsec=4.0, n_fib_inner_min=1,
                             real_ra=None, real_dec=None,
                             footprint=None, z_sample=None, z_mode=None,
                             z_gap_dz=0.15, z_min_frac=0.02,
                             max_tries_per_source=2000, strict=True,
                             verbose=True):
    """Spec sec 2, build-order item 5. Random (ra, dec, z) with an exclusion.

    Rejection sampling, seeded and therefore reproducible from `seed` alone
    -- which is what `inject_sources(placement_seed=...)` records, and the
    reason that argument existed before anything filled it.

    **v8 (Austin, 2026-08-25): draws are now matched to the REAL SAMPLE'S
    SELECTION, not to the VDFI field.** Two changes, one on the sky and one
    in redshift, and they are the same change: a random has to be somewhere a
    real galaxy could have been.

      `footprint` -- an alpha shape (`build_source_footprint`) around the real
        catalog's positions, instead of the field's RA/Dec box. MOSDEF is a
        slit-mask survey and its coverage is a few rectangular pointings with
        concave notches between them; the bounding box interpolates straight
        across every notch and drops randoms on sky nobody observed.
      `z_sample` -- the real run table's redshifts, from which z is drawn
        hump-by-hump (`z_mode`, see `find_redshift_humps`) instead of
        uniformly. MOSDEF's n(z) is two clumps with an empty span between
        them; a uniform draw puts a third of the randoms at observed
        wavelengths the data never reaches, which averages away the
        wavelength-dependent sky residual the null exists to size.

    Both default to None, which reproduces the pre-v8 behaviour exactly --
    every level 0-6 result stands. `footprint='auto'` builds the shape from
    `real_ra`/`real_dec` with `build_source_footprint`'s defaults, which is
    the intended call since those arrays are already being passed for
    `REAL_DIST`.

    THE EXCLUSION IS AN **AND**, AND THAT IS THE WHOLE DESIGN (Austin,
    2026-08-20). A candidate is rejected only if an accepted source is
    BOTH within `min_sep_arcsec` on the sky AND within `min_dz` in redshift.
    Contamination requires spatial *and* spectral overlap: `total_flux_fid`
    and the centroid integrate only `config.line_window` = +/-3 rest A, which
    at z = 2.25 is +/-9.75 observed A = dz 0.0080, so a neighbour further
    than that in redshift contributes NOTHING to either measurement however
    close it sits on the sky. min_dz = 0.01 clears that window with ~25%
    margin.

    This is not a technicality -- it is what makes the experiment fit in the
    field. The COSMOS footprint is ~20' across and a 2000 kpc aperture is
    242" in radius, i.e. 13.1% of the field, so N sources put N x 0.131
    neighbours inside EVERY aperture no matter how they are spaced. Spacing
    alone cannot fix that; only 7-9 sources fit at the 484" true
    zero-overlap separation. With z drawn over a range of 0.5, only ~3% of
    sources are spectral neighbours of any given one, and the effective
    contamination falls ~30x -- the outermost bin at N = 50 goes from ~10x
    the signal to ~0.3x. Measured contamination ratios per bin are in the
    2026-08-20 notes; re-derive them if the profile's gamma changes, because
    gamma = 0.8 is what makes the surface brightness nearly flat from 100 to
    1500 kpc and therefore makes a neighbour anywhere in the aperture
    contribute at every radius.

    **The v8 draws make that exclusion bite harder, in both coordinates.**
    The footprint is smaller than the field, and hump-drawn z's are packed
    into ~1.1 of redshift instead of spread over 1.5 -- so the fraction of
    pairs that are spectral neighbours rises. `min_dz` is validated against
    the SUMMED HUMP WIDTH rather than (z_hi - z_lo) for exactly this reason;
    a two-hump sample with a wide nominal range can otherwise degenerate to a
    purely spatial exclusion without any error being raised. If a run cannot
    place its N, that is the real crowding budget speaking, not a bug --
    lower `min_sep_arcsec` before touching `min_dz`, since the spectral half
    is the part doing the physical work.

    A candidate must also land on real fiber coverage: at least
    `n_fib_inner_min` fibers (summed over exposures, as `N_FIB_INNER` is
    everywhere else) within `r_inner_arcsec`. The fill factor is not 100%
    and a draw in an IFU gap must be discarded, not extracted as NaN.

    Returns
    -------
    dict with 'ra', 'dec', 'z', 'n_fib_inner', 'nn_dist', 'nn_dz',
    'nn_dist_spec', 'real_dist', 'seed', 'n_tries', 'z_mode', 'footprint'.

    `nn_dist` is the nearest projected neighbour at ANY redshift, which is
    what sec 2.2's crowding panel bins on. `nn_dist_spec` is the nearest
    neighbour among those inside `min_dz` -- the one that can actually
    contaminate. Both are returned because they answer different questions
    and conflating them is how a null crowding result gets believed.

    'footprint' is the `SourceFootprint` actually used (None on the legacy
    path), so the shape a run was drawn from can be recovered from its own
    output rather than rebuilt and hoped to match.

    strict=False returns fewer than `n` sources rather than raising. It still
    PRINTS what it got: a silently short sample reads downstream as "the
    field only holds this many", which is a different claim.
    """
    rng = np.random.default_rng(seed)

    if z_range is not None:
        z_lo, z_hi = (float(z_range[0]), float(z_range[1]))
        if not (z_hi > z_lo):
            raise ValueError("z_range must be (z_min, z_max) with z_max > z_min")
    elif z_sample is None:
        raise ValueError("pass z_range, or z_sample (the real run table's "
                         "redshifts) to draw from the real n(z)")

    ra_f = np.asarray(vdfi["RA"]).ravel()
    dec_f = np.asarray(vdfi["DEC"]).ravel()
    ok = np.isfinite(ra_f) & np.isfinite(dec_f)
    ra_f, dec_f = ra_f[ok], dec_f[ok]
    if ra_f.size == 0:
        raise RuntimeError("no finite fiber positions in the field")
    cs = float(np.cos(np.radians(float(np.median(dec_f)))))

    # Coverage prefilter, same trick as _pick_covered_position: one pass over
    # 4.6 M fibers to build a density map, then the exact count only for
    # candidates whose cell is already populated. Testing every fiber against
    # every candidate would be ~1e9 comparisons per accepted source.
    cell = r_inner_arcsec / 3600.0
    ra_lo, ra_hi = np.nanpercentile(ra_f, [1, 99])
    dec_lo, dec_hi = np.nanpercentile(dec_f, [1, 99])
    nx = max(int((ra_hi - ra_lo) * cs / cell), 1)
    ny = max(int((dec_hi - dec_lo) / cell), 1)
    Hc, xe, ye = np.histogram2d(ra_f, dec_f, bins=(nx, ny),
                                range=((ra_lo, ra_hi), (dec_lo, dec_hi)))

    # The candidate rectangle. With a footprint this is the footprint's bbox,
    # NOT the fiber field's -- drawing in the larger box and rejecting would
    # work but wastes most tries and makes max_tries_per_source mean something
    # different. The fiber histogram above still does the coverage prefilter;
    # the footprint and the fiber coverage are independent cuts and a
    # candidate has to pass both.
    fp = footprint
    if isinstance(fp, str):
        if fp != "auto":
            raise ValueError(f"footprint must be a SourceFootprint, 'auto', "
                             f"or None; got {fp!r}")
        if real_ra is None or real_dec is None:
            raise ValueError("footprint='auto' needs real_ra/real_dec (the "
                             "real run table's positions) to build the shape")
        fp = build_source_footprint(real_ra, real_dec, verbose=verbose)
    if fp is not None:
        draw_ra_lo, draw_ra_hi, draw_dec_lo, draw_dec_hi = fp.bbox
    else:
        draw_ra_lo, draw_ra_hi = ra_lo, ra_hi
        draw_dec_lo, draw_dec_hi = dec_lo, dec_hi
        if verbose:
            print("    sky: uniform over the fiber field's 1-99 percentile "
                  "box (no footprint -- randoms will land on sky the real "
                  "sample never covered)")

    draw_z, z_span, z_label = _make_z_sampler(
        rng, z_sample=z_sample, z_range=z_range, z_mode=z_mode,
        gap_dz=z_gap_dz, min_frac=z_min_frac, verbose=verbose)
    if min_dz >= z_span:
        raise ValueError(
            f"min_dz = {min_dz} spans the whole reachable redshift measure "
            f"({z_span:.4f}, summed over {z_label} windows); every pair would "
            f"be a spectral neighbour and the AND exclusion degenerates to a "
            f"pure spatial one.")

    ra_s, dec_s, z_s, nin_s = [], [], [], []
    n_tries = n_rej_cover = n_rej_excl = n_rej_fp = 0
    box = r_inner_arcsec / 3600.0

    while len(ra_s) < int(n):
        tries = 0
        while tries < int(max_tries_per_source):
            tries += 1
            n_tries += 1
            ra_c = float(rng.uniform(draw_ra_lo, draw_ra_hi))
            dec_c = float(rng.uniform(draw_dec_lo, draw_dec_hi))
            z_c = draw_z()

            if fp is not None and not bool(fp.contains(ra_c, dec_c)):
                n_rej_fp += 1
                continue

            i = min(np.searchsorted(xe, ra_c) - 1, Hc.shape[0] - 1)
            j = min(np.searchsorted(ye, dec_c) - 1, Hc.shape[1] - 1)
            if i < 0 or j < 0 or Hc[i, j] <= 0:
                n_rej_cover += 1
                continue

            # THE AND. Spatial proximity alone is not a rejection.
            if ra_s:
                d = np.hypot((np.asarray(ra_s) - ra_c) * cs,
                             np.asarray(dec_s) - dec_c) * 3600.0
                dz = np.abs(np.asarray(z_s) - z_c)
                if np.any((d < min_sep_arcsec) & (dz < min_dz)):
                    n_rej_excl += 1
                    continue

            m = ((np.abs(dec_f - dec_c) < box) & (np.abs(ra_f - ra_c) * cs < box))
            if not m.any():
                n_rej_cover += 1
                continue
            dd = np.hypot((ra_f[m] - ra_c) * cs, dec_f[m] - dec_c) * 3600.0
            n_in = int(np.sum(dd < r_inner_arcsec))
            if n_in < int(n_fib_inner_min):
                n_rej_cover += 1
                continue

            ra_s.append(ra_c); dec_s.append(dec_c)
            z_s.append(z_c); nin_s.append(n_in)
            break
        else:
            msg = (f"placed only {len(ra_s)}/{n} sources: "
                   f"{max_tries_per_source} consecutive draws failed "
                   f"(min_sep={min_sep_arcsec}\", min_dz={min_dz}, "
                   f"z {z_label} span={z_span:.3f}"
                   + (f", footprint {fp.area_arcmin2:.0f} arcmin^2"
                      if fp is not None else "")
                   + "). The exclusion volume is full -- lower min_sep, or "
                     "widen the redshift windows, or ask for fewer sources. "
                     "Do NOT reach for min_dz first: the spectral half of the "
                     "AND is the part doing the physical work.")
            if strict:
                raise RuntimeError(msg)
            print(f"    [!] {msg}")
            break

    ra_s = np.asarray(ra_s); dec_s = np.asarray(dec_s)
    z_s = np.asarray(z_s); nin_s = np.asarray(nin_s, dtype=int)
    k = ra_s.size

    sep = np.hypot((ra_s[:, None] - ra_s[None, :]) * cs,
                   dec_s[:, None] - dec_s[None, :]) * 3600.0
    dzm = np.abs(z_s[:, None] - z_s[None, :])
    np.fill_diagonal(sep, np.inf)
    np.fill_diagonal(dzm, np.inf)
    nn_dist = sep.min(axis=1) if k > 1 else np.full(k, np.inf)
    nn_dz = dzm.min(axis=1) if k > 1 else np.full(k, np.inf)
    spec = np.where(dzm < min_dz, sep, np.inf)
    nn_dist_spec = spec.min(axis=1) if k > 1 else np.full(k, np.inf)

    if real_ra is not None and real_dec is not None:
        rr = np.asarray(real_ra, dtype=float).ravel()
        rd = np.asarray(real_dec, dtype=float).ravel()
        real_dist = np.array([
            float(np.min(np.hypot((rr - a) * cs, rd - d)) * 3600.0)
            for a, d in zip(ra_s, dec_s)]) if rr.size else np.full(k, np.nan)
    else:
        real_dist = np.full(k, np.nan)

    if verbose:
        print(f"    placed {k}/{n} sources in {n_tries} draws "
              f"({n_rej_fp} outside the footprint, {n_rej_cover} off "
              f"coverage, {n_rej_excl} inside the exclusion)")
        print(f"    z {z_s.min():.4f}-{z_s.max():.4f} | "
              f"nn_dist {nn_dist.min():.0f}-{nn_dist.max():.0f}\" | "
              f"nearest SPECTRAL neighbour "
              + ("none within min_dz" if not np.isfinite(nn_dist_spec).any()
                 else f"{np.nanmin(nn_dist_spec[np.isfinite(nn_dist_spec)]):.0f}\"")
              + f" | inner fibers {nin_s.min()}-{nin_s.max()}")
        if np.isfinite(real_dist).any():
            print(f"    distance to nearest REAL source "
                  f"{np.nanmin(real_dist):.0f}-{np.nanmax(real_dist):.0f}\"")

    return {"ra": ra_s, "dec": dec_s, "z": z_s, "n_fib_inner": nin_s,
            "nn_dist": nn_dist, "nn_dz": nn_dz, "nn_dist_spec": nn_dist_spec,
            "real_dist": real_dist, "seed": int(seed), "n_tries": n_tries,
            "z_mode": z_label, "footprint": fp}


def check_level1(model, vdfi, z, bins_kpc, kpc_per_arcsec=None,
                 ra=None, dec=None, seeing_mode="field", plot=True,
                 verbose=True, rng_seed=7, tol_dv_kms=2.0,
                 tol_shape=0.05, min_nfib_shape=20, response=None):
    """Ladder level 1 (spec sec 7.2): the template on REAL fiber geometry,
    against the independent truth generator. No cube, no noise, no extraction.

    Replaces the thirteen-cell level-1 notebook with one call. Everything it
    does was previously a separate cell; the cells that merely INVESTIGATED
    something (the deliberate-break test, the r_trunc table, the
    mean/median/biweight ladder) produced their answers and live in the
    changelog rather than here.

    WHAT THIS IS AND IS NOT. This is the "is the template what I think it is"
    half of the problem -- a question about MY code. It is deliberately NOT
    the "does the pipeline recover it" half, which needs the cube overlay and
    a real extraction. Conflating them gives a test where a failure has two
    causes and no way to tell which, so level 1 is validated to completion
    while the pipeline is held trivially predictable (spec sec 7.2).

    Real fiber positions, real per-exposure seeing, real DAR. The only thing
    asserted is the source; everything it sits in is measured geometry.

    Checks, all against `predict_binned_observables` and never against a
    config value:

      1. templates finite, non-negative, and non-empty
      2. flux conservation -- recovered fraction matches the GEOMETRIC
         prediction (fibers x fiber area / cut-disk area) x (fraction of L
         inside the cut). Catches a stray 2*pi (6.28) or 1/(pi*0.75^2)
         (0.57) cleanly, which is the 2026-07-22 area-convention class.
      3. binned dv matches the truth generator's -- THE PRIMARY RESULT
         (spec sec 0.3). Compared per bin against a tolerance far inside one
         wavelength pixel (110-140 km/s), which is the floor on every
         velocity statement in the project.
      4. binned profile SHAPE matches -- the ratio of recovered to predicted
         surface brightness is constant across bins. Shape only: absolute
         normalization is check 2's job, and conflating them gives a check
         that fails for two unrelated reasons.

    Parameters
    ----------
    model : SourceModel
    vdfi : the dict from `extract.load_field` -- needs wave, RA, DEC, seeing,
        and optionally dar_ra / dar_dec.
    z : test redshift. Sets the observed wavelength, and therefore the
        sky-line coincidence rate, the sub-pixel phase against the pixel
        grid, and where the line falls in the LSF.
    bins_kpc : the PRODUCTION edges. As of 2026-08-16, 11 bins to 2000 kpc.
    kpc_per_arcsec : angular scale. None -> computed from z with Planck18.
    ra, dec : injection position. None -> a covered position is chosen.
    seeing_mode : "field" (one value, the field median) or "exposure".
        "field" is the RIGHT choice here, not a shortcut:
        `predict_binned_observables` accepts only a SCALAR seeing, so a
        per-exposure injection is already being compared against a
        single-seeing truth, and matching the injector to it REMOVES a known
        discrepancy rather than adding one. Measured cost of the choice at
        seeing median 1.97" / sd 0.37": dv differs by at most 0.06 km/s, the
        profile by +3.1% in the innermost bin and <1% outside -- seeing only
        bites where the profile has structure on the seeing scale. Use
        "exposure" for the real cube run, where it is affordable because the
        PSF tables cache per exposure across all sources.
    tol_dv_kms : per-bin velocity tolerance. Default 2.0, against a 110-140
        km/s pixel -- generous by design, because the interpolation grids
        carry ~0.03 km/s of their own and the point is to catch CONVENTION
        errors (half-pixel biases, missing LSF, wrong flux weighting), which
        are tens of km/s, not grid noise. The measured deviation is printed
        so the real margin is visible rather than hidden behind a verdict.
    tol_shape, min_nfib_shape : allowed fractional spread in the
        recovered/predicted surface-brightness ratio, and the minimum fiber
        count for a bin to enter that check. The floor matters: the recovered
        value is a mean over FIBERS and the predicted one is a mean over
        AREA, and those only agree where fibers sample the annulus. A bin
        with three fibers in it is measuring where those three fibers landed.

    Returns
    -------
    (ok, lines, data) -- unlike `check_level0`'s two-tuple, because level 1
    produces per-bin arrays worth keeping. `data` holds r_mid, dv_rec,
    dv_truth, prof_rec, sb_truth, nfib_bin, ra, dec, and the templates.
    """
    import time
    from .core import _exposure_seeing, _offsets_with_dar

    lines, ok = [], True
    t_start = time.time()

    bins_kpc = np.asarray(bins_kpc, dtype=float)
    nbin = bins_kpc.size - 1
    wave = np.asarray(vdfi["wave"], dtype=float)
    RA, DEC = np.asarray(vdfi["RA"]), np.asarray(vdfi["DEC"])
    nexp = RA.shape[0]

    if kpc_per_arcsec is None:
        from astropy.cosmology import Planck18 as _cosmo
        import astropy.units as _u
        kpc_per_arcsec = float(
            _cosmo.angular_diameter_distance(z).to(_u.kpc).value / 206265.0)

    rng = np.random.default_rng(rng_seed)
    if ra is None or dec is None:
        _t = time.time()
        ra, dec, n_in = _pick_covered_position(RA, DEC, rng)
        lines.append(f"           position chosen: RA={ra:.5f} DEC={dec:.5f} "
                     f"({n_in} fibers within 4\", pooled over exposures, "
                     f"{time.time() - _t:.1f} s)")
    else:
        ra, dec = float(ra), float(dec)
        lines.append(f"           position given:  RA={ra:.5f} DEC={dec:.5f}")

    # THE TEMPLATE TIMER STARTS HERE, not at the top of the function.
    # Position picking is one-time setup; including it made exposure 0 look
    # 13x more expensive than it is and multiplied that mistake by nexp to
    # give a 74-minute ETA for a 3-minute job (2026-08-16).
    t_tmpl = time.time()

    lam0 = observed_lya_wavelength(z)
    dv_pix = C_KMS * float(np.mean(np.diff(wave))) / lam0
    lines.append(f"           kpc/arcsec = {kpc_per_arcsec:.3f}; one pixel = "
                 f"{dv_pix:.1f} km/s (the floor on every velocity statement)")

    seeing_field = float(np.nanmedian(vdfi["seeing"]))

    # ---- build the template per exposure, through the real optics ----
    # Per exposure, WITH DAR. Injecting into the coadd or at a fixed
    # wavelength-independent position would skip the optics chain entirely and
    # is the most likely way to build a test that passes vacuously (sec 3.1).
    templates, dists, fib_idx = [], [], []
    t_first = 0.0
    for e in range(nexp):
        cs = np.cos(np.radians(dec))
        dra_1d = (RA[e] - ra) * cs * 3600.0
        ddec_1d = (DEC[e] - dec) * 3600.0
        keep = select_nearby_fibers(dra_1d, ddec_1d, bins_kpc[-1], kpc_per_arcsec)
        if not keep.any():
            templates.append(None); dists.append(None); fib_idx.append(None)
            continue
        dra, ddec = _offsets_with_dar(dra_1d[keep], ddec_1d[keep],
                                      vdfi.get("dar_ra"), vdfi.get("dar_dec"),
                                      e, wave.size)
        s_e = (seeing_field if seeing_mode == "field"
               else _exposure_seeing(vdfi["seeing"], e))
        templates.append(build_source_template(
            model, z=z, dra=dra, ddec=ddec, seeing_fwhm=s_e, wave=wave,
            kpc_per_arcsec=kpc_per_arcsec, response=response))
        # STATIC 1-D distance, no DAR -- this is what production bins on
        # (extract.py builds bin_idx_all per fiber, not per wavelength).
        dists.append(np.hypot(dra_1d[keep], ddec_1d[keep]))
        fib_idx.append(np.flatnonzero(keep))
        # PROGRESS. The first exposure pays shared setup (the PSF table and
        # the ring kernel, cached across every later exposure), so its time is
        # NOT representative -- the ETA is quoted from the warm cost instead.
        # Print the first few and then periodically: 132 exposures of silence
        # is indistinguishable from a hang, which is how a 3-minute cell got
        # reported as broken.
        if verbose and (e < 3 or (e + 1) % 20 == 0 or e == nexp - 1):
            el = time.time() - t_tmpl
            warm = (el - t_first) / max(e, 1) if e > 0 else el
            if e == 0:
                t_first = el
                print(f"           exp {e:3d}: {int(keep.sum())} fibers kept, "
                      f"cold {el:.1f} s (shared setup, paid once)", flush=True)
            else:
                print(f"           exp {e:3d}: {el:6.1f} s elapsed, "
                      f"{warm:.2f} s/exp warm -> ETA "
                      f"{warm * (nexp - e - 1) / 60:.1f} min", flush=True)

    live = [i for i, t in enumerate(templates) if t is not None]

    # ---- 1. templates sane ----
    if not live:
        lines.append("[FAIL] no exposure produced a template -- position has "
                     "no fiber coverage at all")
        if verbose:
            print("\n".join(lines))
        return False, lines, {}
    finite = all(np.isfinite(templates[i]).all() for i in live)
    nonneg = all((templates[i] >= 0).all() for i in live)
    mb = sum(templates[i].nbytes for i in live) / 1e6
    good = finite and nonneg
    ok &= good
    lines.append(f"[{'PASS' if good else 'FAIL'}] {len(live)}/{nexp} exposure "
                 f"templates, {mb:.0f} MB, all finite: {finite}, "
                 f"non-negative: {nonneg}")

    # ---- 2. flux conservation, against a geometric prediction ----
    # The pixel width must be in km/s, NOT Angstrom: line_profile is
    # normalized to unit integral in VELOCITY. At z=2.4 that is c/lam0 = 72.5,
    # and getting it wrong made this check report a phantom 70x normalization
    # error in 2026-08-07's fourth pass -- the check manufacturing exactly the
    # failure it exists to detect.
    r_cut = bins_kpc[-1] / kpc_per_arcsec + 5.0
    frac_L_inside = float(model.enclosed_fraction(np.array([bins_kpc[-1]]))[0])
    n_typ = int(np.median([templates[i].shape[0] for i in live]))
    fill = n_typ * np.pi * 0.75 ** 2 / (np.pi * r_cut ** 2)
    expect = fill * frac_L_inside
    fracs = np.array([float(np.nansum(templates[i])) * dv_pix for i in live])
    ratio = float(np.median(fracs)) / expect if expect > 0 else np.nan
    # Fibers live in IFUs and are not uniform over the disk, so this is a
    # sanity BOUND, not a precision test. What it catches cleanly is a stray
    # 2*pi (6.28), a stray 1/(pi*0.75^2) (0.57), or their inverses.
    good = 0.4 < ratio < 2.5
    ok &= good
    lines.append(f"[{'PASS' if good else 'FAIL'}] flux conservation: recovered "
                 f"{np.median(fracs):.4f} vs geometric prediction {expect:.4f} "
                 f"-> ratio {ratio:.2f}  (2pi would be {ratio * 2 * np.pi:.1f}, "
                 f"aperture-area {ratio / (np.pi * 0.75 ** 2):.2f})")

    # ---- bin exactly the way extract_fibers bins fibers ----
    # method="mean", NOT biweight. On a NOISELESS, steeply-falling template the
    # biweight converges to the median over the annulus while the truth
    # generator predicts the area-weighted MEAN, and for a steep profile in a
    # wide bin those differ a lot. That gap is the ESTIMATOR, not a pipeline
    # bias, and letting it into a geometry check would fail this for a reason
    # that has nothing to do with geometry.
    binned, nfib_bin = bin_template_radially(
        [templates[i] for i in live], [dists[i] for i in live],
        bins_kpc, kpc_per_arcsec, method="mean")

    truth = predict_binned_observables(
        model, z=z, bins_kpc=bins_kpc, seeing_fwhm=seeing_field,
        kpc_per_arcsec=kpc_per_arcsec)

    dv_rec = np.array([centroid_from_spectrum(binned[i], wave, z)
                       for i in range(nbin)])
    prof_rec = np.array([float(np.nansum(binned[i])) for i in range(nbin)])

    # ---- 3. velocity -- the primary result ----
    has_flux = np.isfinite(dv_rec) & (nfib_bin > 0) & (prof_rec > 0)
    dev = np.abs(dv_rec - truth["dv"])
    worst = float(np.nanmax(dev[has_flux])) if has_flux.any() else np.nan
    good = bool(has_flux.any()) and worst < tol_dv_kms
    ok &= good
    lines.append(f"[{'PASS' if good else 'FAIL'}] binned dv vs truth: worst "
                 f"deviation {worst:.3f} km/s over {int(has_flux.sum())} bins "
                 f"(tol {tol_dv_kms}, one pixel = {dv_pix:.0f})")

    # ---- 4. profile SHAPE ----
    # binned[i] is the MEAN over the bin's fibers, i.e. flux per fiber, and a
    # fiber collects SB(r) x (effective fiber area) -- so prof_rec is already
    # proportional to surface brightness. Dividing by annulus area here would
    # double-count it.
    # BINS WITH FEW FIBERS ARE EXCLUDED, and this is a correction to the
    # METRIC, not a loosened tolerance. `binned[i]` is the mean over the
    # bin's FIBERS; the truth generator's `sb` is the AREA-weighted mean over
    # the annulus. Those agree only insofar as the fibers sample the annulus
    # uniformly in area. In a wide bin with a steep profile and a handful of
    # fibers, whichever side of the bin those few fibers happen to fall on
    # sets the answer, and the disagreement is fiber sampling -- real, but
    # nothing to do with whether the template is correct. Measured on a
    # synthetic uniform-grid field: 19% spread including 3-fiber bins, and
    # the offending bins are exactly the sparse ones.
    sel = has_flux & (truth["sb"] > 0) & (nfib_bin >= min_nfib_shape)
    if sel.sum() >= 3:
        rr = prof_rec[sel] / truth["sb"][sel]
        spread = float(np.nanmax(rr) / np.nanmin(rr) - 1.0)
        good = spread < tol_shape
        ok &= good
        lines.append(f"[{'PASS' if good else 'FAIL'}] profile shape: "
                     f"recovered/predicted ratio constant to {100*spread:.2f}% "
                     f"across {int(sel.sum())} bins with >={min_nfib_shape} "
                     f"fibers (tol {100*tol_shape:.0f}%)")
    else:
        lines.append(f"[SKIP] profile shape -- fewer than 3 bins have "
                     f">={min_nfib_shape} fibers")

    # ---- where does the velocity ramp come from? ----
    # The injected step is ABRUPT, but the recovered curve ramps over ~2 bins,
    # and it is worth knowing that almost none of that is the optics. Split it:
    #
    #   BINNING     the bin containing r_break holds gas at BOTH velocities and
    #               must report a blend -- pure arithmetic, computed exactly
    #               from the enclosed-flux profile, no PSF involved.
    #   PSF         whatever the truth generator says on top of that blend.
    #
    # Answers "why doesn't it shoot straight from 0 to 100" with a number
    # instead of an argument.
    if isinstance(model.velocity, StepVelocity) and model.profile != "point":
        rb = model.velocity.r_break
        k = int(np.searchsorted(bins_kpc, rb) - 1)
        if 0 <= k < nbin:
            lo, hi = float(bins_kpc[k]), float(bins_kpc[k + 1])
            F = model.enclosed_fraction(np.array([lo, rb, hi]))
            denom = F[2] - F[0]
            if denom > 0:
                f_beyond = float((F[2] - F[1]) / denom)
                blend = (model.velocity.dv_in
                         + (model.velocity.dv_out - model.velocity.dv_in) * f_beyond)
                lines.append("")
                lines.append(
                    f"           break at {rb:.0f} kpc falls inside bin "
                    f"[{lo:.0f}, {hi:.0f}]: {100*f_beyond:.1f}% of that bin's "
                    f"flux is beyond it")
                lines.append(
                    f"           -> binning alone predicts {blend:.1f} km/s; "
                    f"truth generator says {truth['dv'][k]:.2f}, so the PSF "
                    f"contributes {truth['dv'][k] - blend:+.2f}")
                if k > 0:
                    lines.append(
                        f"           the bin inside it reads "
                        f"{truth['dv'][k-1]:.2f} where the source is exactly "
                        f"{model.velocity.dv_in:.0f} -- THAT is the PSF, and "
                        f"it is "
                        f"{100*abs(truth['dv'][k-1]-model.velocity.dv_in)/max(abs(model.velocity.dv_out-model.velocity.dv_in),1e-9):.1f}%"
                        f" of the step")

    # ---- the per-bin table ----
    # SB ratio normalized to its own median across the checked bins, so the
    # column reads as "how flat is this" rather than needing the absolute
    # scale. A single bin far from 1.000 while the rest sit on it is fiber
    # sampling in that bin, not a template error -- and that distinction took
    # a by-hand printout to make the first time, so it lives here now.
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = prof_rec / np.where(truth["sb"] > 0, truth["sb"], np.nan)
    ref = np.nanmedian(rat[sel]) if sel.any() else np.nan
    rat = rat / ref if np.isfinite(ref) and ref != 0 else rat * np.nan

    lines.append("")
    lines.append(f"{'bin (kpc)':>14} {'nfib':>7} {'dv rec':>9} {'dv TRUTH':>9} "
                 f"{'diff':>8} {'dv naive':>9} {'SB truth':>11} {'SB rat':>8}")
    for i in range(nbin):
        flag = "" if sel[i] else "  (sparse, excluded)"
        lines.append(
            f"{bins_kpc[i]:5.0f}-{bins_kpc[i+1]:<8.0f} {nfib_bin[i]:7d} "
            f"{dv_rec[i]:9.2f} {truth['dv'][i]:9.2f} "
            f"{dv_rec[i]-truth['dv'][i]:8.2f} {truth['dv_naive'][i]:9.1f} "
            f"{truth['sb'][i]:11.4g} {rat[i]:8.4f}{flag}")
    lines.append("")
    lines.append("           'dv naive' is dv(r_center) -- the WRONG truth, shown")
    lines.append("           only so the size of the flux-weighting effect is")
    lines.append("           visible. Never compare against it.")

    data = dict(r_mid=truth["r_mid"], dv_rec=dv_rec, dv_truth=truth["dv"],
                prof_rec=prof_rec, sb_truth=truth["sb"], nfib_bin=nfib_bin,
                ra=ra, dec=dec, kpc_per_arcsec=kpc_per_arcsec,
                templates=templates, dists=dists, fib_idx=fib_idx,
                binned=binned, truth=truth, seeing_field=seeing_field)

    if verbose:
        print(f"--- ladder level 1: {model!r}")
        print("\n".join(lines))
        print(f"--- level 1 {'PASSED' if ok else 'FAILED'} "
              f"({time.time() - t_start:.0f} s)")

    if plot:
        _plot_level1(data, bins_kpc, model=model)

    return ok, lines, data


def _plot_level1(data, bins_kpc, model=None):
    """The level-1 figure. THREE curves, not two, and the distinction between
    them is the whole point of the plot:

      1. INTRINSIC (grey) -- the source as DEFINED. A perfect step in
         velocity; the un-smeared profile in surface brightness. Not
         measurable, and nothing should ever be compared against it.
      2. TRUTH GENERATOR (black) -- what SHOULD come back after the PSF
         smears the source and the light is averaged into radial bins. This
         is a prediction of the MEASUREMENT.
      3. TEMPLATE, BINNED (red) -- what the injector actually produced.

    The test is 3 vs 2. Curve 1 is there so the gap between "what I put in"
    and "what a correct pipeline returns" is visible rather than something
    you have to hold in your head -- and that gap is mostly BINNING, not
    optics. A velocity step at 200 kpc sits inside the [180, 350] bin, so
    that bin contains gas at both velocities and MUST report a blend; ~89% of
    its light lies beyond 200 kpc, which is why the blend lands near 91 and
    not near 50. The genuinely PSF-driven part is the next bin in reading
    0.57 instead of 0, i.e. 0.6% of the step, because 2" of seeing is ~17 kpc
    against bins hundreds of kpc wide out there.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    r_mid, sb = data["r_mid"], data["sb_truth"]
    prof = data["prof_rec"]
    r_fine = np.logspace(np.log10(max(float(bins_kpc[1]) * 0.3, 1.0)),
                         np.log10(float(bins_kpc[-1])), 600)

    # ---- left: profile. SHAPE ONLY -- prof is rescaled to the truth's peak,
    # so this panel CANNOT fail on normalization, by construction.
    # Normalization is the flux-conservation check's job.
    if model is not None and model.profile != "point":
        I_int = model.I(r_fine)
        good = I_int > 0
        # ANCHORED ON THE MEDIAN BIN, NOT THE OUTERMOST ONE (fixed 2026-08-16).
        # The intrinsic profile carries no absolute scale here -- it has to be
        # tied to the truth curve somewhere, and WHERE decides what the eye
        # sees. Anchoring at the outermost bin was the worst possible choice:
        # that is the bin sitting on the r_trunc cut, where truth/intrinsic
        # runs 0.77 while the flat mid-radius bins run 1.00. Tying the two
        # curves together there rescaled the whole grey curve by 1/0.77 = 1.29
        # and manufactured a ~29% gap at mid radius that is not physical.
        # The median ratio is robust to BOTH ends -- the PSF-dominated inner
        # bins and the truncation-affected outer one -- so the curves now
        # coincide where they should and separate only where physics
        # separates them.
        Imid = model.I(r_mid)
        okr = (Imid > 0) & (sb > 0)
        ref = float(np.median(sb[okr] / Imid[okr])) if okr.sum() else np.nan
        if np.isfinite(ref) and ref > 0:
            ax[0].loglog(r_fine[good], I_int[good] * ref,
                         color="0.6", lw=1.5, ls="-",
                         label="intrinsic (no PSF, no binning)")
    ax[0].loglog(r_mid, sb, "k-", lw=2, label="truth generator (PSF + binned)")
    pk = np.nanmax(prof)
    if pk > 0:
        ax[0].loglog(r_mid, prof / pk * np.nanmax(sb), "o--", color="tab:red",
                     label="template, binned (scaled)")
    for edge in bins_kpc[1:]:
        ax[0].axvline(edge, color="0.9", lw=0.6, zorder=0)
    ax[0].set_xlabel("r [kpc]"); ax[0].set_ylabel("surface brightness")
    ax[0].legend(fontsize=8)
    ax[0].set_title("profile — SHAPE only\n"
                    "(grey below black inside ~20 kpc = PSF moving flux out)",
                    fontsize=10)

    # ---- right: velocity, the primary result
    if model is not None:
        ax[1].semilogx(r_fine, model.velocity(r_fine), color="0.6", lw=1.5,
                       label="intrinsic (the step as injected)")
    ax[1].semilogx(r_mid, data["dv_truth"], "k-", lw=2,
                   label="truth generator (PSF + binned)")
    ax[1].semilogx(r_mid, data["dv_rec"], "o--", color="tab:red",
                   label="template, recovered")
    ax[1].axvline(200, color="tab:blue", ls=":", lw=1,
                  label="injected break, 200 kpc")
    ax[1].axvline(75, color="tab:green", ls=":", lw=1,
                  label=r"$R_{vir}\sim75$ kpc")
    for edge in bins_kpc:
        ax[1].axvline(edge, color="0.85", lw=0.6, zorder=0)
    ax[1].set_xlabel("r [kpc]"); ax[1].set_ylabel(r"$\Delta v$ [km s$^{-1}$]")
    ax[1].legend(fontsize=7.5, loc="best")
    ax[1].set_title("velocity — the primary result\n"
                    "(grey = injected step; black = what binning does to it)",
                    fontsize=10)
    plt.tight_layout(); plt.show()


def plot_injection_recovery(result, model, z, seeing_fwhm, kpc_per_arcsec,
                            bins_kpc=None, truth=None, lsf_kms=None,
                            n_fine=4000, real=None, rescale="median",
                            label="recovered", verbose=True):
    """The level-6 figure: WHAT I PUT IN vs WHAT SHOULD COME BACK vs WHAT DID.

    Same three-curve convention as `_plot_level1`, but the third curve is an
    actual EXTRACTION with error bars rather than a binned template. Keeping
    the conventions identical is the point -- a level-1 figure and a level-6
    figure of the same model should be readable side by side.

      1. INTRINSIC (grey, smooth)  -- the source as DEFINED. Not measurable.
         NOTHING should ever be compared against this curve; it is on the plot
         so the gap between "what I injected" and "what a CORRECT pipeline
         returns" is visible instead of held in your head. Most of that gap is
         BINNING, not optics.
      2. TRUTH GENERATOR (black step) -- `predict_binned_observables`: the
         flux-weighted, PSF-convolved, bin-averaged prediction. THE TRUTH.
      3. RECOVERED (red, with errors) -- the measured stack.

    The test is 3 vs 2. The 1-vs-2 gap is the dilution the pipeline is
    ENTITLED to; the 2-vs-3 gap is bias.

    Parameters
    ----------
    result : the `run_measure` dict. Needs `total_flux_fid` (+ `_lo`/`_hi`)
        and `centroid_v_fid` (+ `_lo`/`_hi`), and `r_edges` either here or in
        `bins_kpc`.
    real : optional (v, err) from the REAL run, overplotted in blue on the
        velocity panel. This is the comparison the deconvolution is actually
        about: adjust the injected dv(r) until the red curve lands on the blue
        one, and the INTRINSIC grey curve is then your answer for the real
        galaxies.
    rescale : "median" (default) compares SHAPE ONLY, anchoring the recovered
        profile to the truth by their median ratio -- and PRINTS that ratio.

        SHAPE-ONLY IS NOT LAZINESS, IT IS THE ONLY HONEST DEFAULT RIGHT NOW.
        `line_profile` has unit integral in VELOCITY and is written straight
        into a per-Angstrom cube, so the injected flux is low by c/lambda_obs
        -- 72.5x at z = 2.4, and z-dependent (2026-08-19; fix NOT applied).
        Until that lands, an absolute flux comparison tests the unit bug and
        nothing else.

        WHICH MAKES THE PRINTED RATIO A DIAGNOSTIC, NOT A NUISANCE: if the
        unit bug is the ONLY normalization error, the ratio must come out at
        C_KMS / lambda_obs(z). It is printed next to that prediction. Agreement
        confirms the diagnosis and gives the correction factor for free;
        disagreement means there is a SECOND normalization error, which is
        exactly what the 2026-07-22 PSF area-convention bug looked like.

        Pass rescale=None to plot both on their raw scales instead.
    """
    import matplotlib.pyplot as plt

    if bins_kpc is None:
        bins_kpc = result["r_edges"]
    bins_kpc = np.asarray(bins_kpc, dtype=float)

    if truth is None:
        truth = predict_binned_observables(model, z, bins_kpc, seeing_fwhm,
                                           kpc_per_arcsec, lsf_kms=lsf_kms,
                                           n_fine=n_fine)
    r_mid = truth["r_mid"]

    sb_rec = np.asarray(result["total_flux_fid"], dtype=float)
    sb_err = 0.5 * (np.asarray(result["total_flux_hi"], dtype=float)
                    - np.asarray(result["total_flux_lo"], dtype=float))
    v_rec = np.asarray(result["centroid_v_fid"], dtype=float)
    v_err = 0.5 * (np.asarray(result["centroid_v_hi"], dtype=float)
                   - np.asarray(result["centroid_v_lo"], dtype=float))

    scale = 1.0
    if rescale == "median":
        ok = np.isfinite(sb_rec) & np.isfinite(truth["sb"]) & (sb_rec != 0)
        # MEDIAN RATIO, NOT A RATIO AT ONE BIN. Anchoring on the outermost bin
        # (the r_trunc cut) or the innermost (PSF-dominated) tilts the whole
        # comparison; _plot_level1 hit exactly that and it manufactured a 29%
        # mid-radius gap that was not physical.
        scale = float(np.nanmedian(truth["sb"][ok] / sb_rec[ok]))
        lam_obs = observed_lya_wavelength(z)
        if verbose:
            print(f"  recovered -> truth scale : {scale:.4g}")
            print(f"  c / lambda_obs(z={z:.3f})  : {C_KMS / lam_obs:.4g}"
                  "   <- what the per-km/s-into-per-Angstrom bug predicts")
            print(f"  residual after that      : "
                  f"{scale / (C_KMS / lam_obs):.4g}   <- should be ~1")

    sb_plot, sbe_plot = sb_rec * scale, sb_err * scale

    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.4))
    r_fine = np.logspace(np.log10(max(float(bins_kpc[1]) * 0.3, 1.0)),
                         np.log10(float(bins_kpc[-1])), 600)

    # ---- panel 0: surface brightness ---------------------------------
    if model.profile != "point":
        I_int = model.I(r_fine)
        good = I_int > 0
        anchor = np.nanmedian(truth["sb"] / np.interp(r_mid, r_fine, I_int))
        ax[0].loglog(r_fine[good], I_int[good] * anchor, color="0.6", lw=1.5,
                     label="intrinsic (not measurable)")
    ax[0].loglog(r_mid, truth["sb"], "k-", marker="s", ms=4, lw=1.2,
                 label="truth (PSF + binned)")
    ax[0].errorbar(r_mid, sb_plot, yerr=sbe_plot, fmt="o", color="crimson",
                   ms=4, lw=1.2, capsize=2, label=label)
    ax[0].set_xlabel("r [kpc]")
    ax[0].set_ylabel("surface brightness [arb.]")
    ax[0].set_title("flux profile" + ("  (shape only)" if scale != 1.0 else ""))
    ax[0].legend(fontsize=8)

    # ---- panel 1: fractional residuals -------------------------------
    # TWO curves, and the difference between them is the whole diagnostic:
    # red is recovered/truth (BIAS, should be flat at 0), grey is
    # truth/intrinsic (DILUTION, the pipeline is entitled to this).
    with np.errstate(invalid="ignore", divide="ignore"):
        res = sb_plot / truth["sb"] - 1.0
        rese = sbe_plot / truth["sb"]
    if model.profile != "point":
        I_mid = np.interp(r_mid, r_fine, model.I(r_fine))
        dil = truth["sb"] / (I_mid * np.nanmedian(truth["sb"] / I_mid)) - 1.0
        ax[1].semilogx(r_mid, 100 * dil, color="0.6", ls="--", lw=1.5,
                       label="truth / intrinsic  (dilution)")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].errorbar(r_mid, 100 * res, yerr=100 * rese, fmt="o", color="crimson",
                   ms=4, lw=1.2, capsize=2, label="recovered / truth  (bias)")
    ax[1].set_xlabel("r [kpc]")
    ax[1].set_ylabel("fractional deviation [%]")
    ax[1].set_title("profile residuals")
    ax[1].legend(fontsize=8)

    # ---- panel 2: velocity -------------------------------------------
    ax[2].semilogx(r_fine, model.velocity(r_fine), color="0.6", lw=1.5,
                   label="intrinsic dv(r)")
    ax[2].semilogx(r_mid, truth["dv"], "k-", marker="s", ms=4, lw=1.2,
                   label="truth (blended)")
    ax[2].errorbar(r_mid, v_rec, yerr=v_err, fmt="o", color="crimson", ms=4,
                   lw=1.2, capsize=2, label=label)
    if real is not None:
        rv, re_ = np.asarray(real[0], dtype=float), np.asarray(real[1], dtype=float)
        n = min(rv.size, r_mid.size)
        ax[2].errorbar(r_mid[:n], rv[:n], yerr=re_[:n], fmt="^", color="tab:blue",
                       ms=4, lw=1.2, capsize=2, alpha=0.8, label="REAL data")
    ax[2].axhline(0, color="k", lw=0.8, ls=":")
    ax[2].set_xlabel("r [kpc]")
    ax[2].set_ylabel(r"$\Delta v$ [km s$^{-1}$]")
    ax[2].set_title("centroid velocity")
    ax[2].legend(fontsize=8)

    fig.suptitle(f"{model!r}   z={z:.2f}, seeing={seeing_fwhm:.2f}\"", fontsize=9)
    fig.tight_layout()

    # ---- the numbers, because the eye cannot read a 4-sigma outlier off
    #      a log axis (this is how [60,100] hid at level 2) -------------
    pull = (v_rec - truth["dv"]) / np.where(v_err > 0, v_err, np.nan)
    if verbose:
        print(f"\n  {'bin':>12}  {'truth dv':>9}  {label[:9]:>9}  {'pull':>6}"
              f"  {'SB res':>8}")
        for i in range(r_mid.size):
            print(f"  {bins_kpc[i]:5.0f}-{bins_kpc[i+1]:<6.0f} "
                  f"{truth['dv'][i]:9.1f}  {v_rec[i]:9.1f}  {pull[i]:6.2f}"
                  f"  {100*res[i]:7.1f}%")
        ok = np.isfinite(pull)
        print(f"\n  chi2(dv) = {np.nansum(pull[ok]**2):.1f} / {ok.sum()}"
              f"    rms pull = {np.sqrt(np.nanmean(pull[ok]**2)):.2f}"
              "   <- should be 1.0 if the error bars are honest")
    return fig, {"truth": truth, "scale": scale, "pull": pull,
                 "sb_residual": res}


def plot_injection_recovery(boot, model, z, seeing_fwhm, kpc_per_arcsec,
                            bins_kpc=None, truth=None, real=None,
                            n_fine=4000, scale=None, figsize=(15.0, 4.6)):
    """Recovered vs INTRINSIC vs TRUTH, for a real injected extraction.

    `_plot_level1` does this for a noiseless template. This does it for the
    thing that actually came out of extract -> stack -> measure, with error
    bars, so it is the figure that answers "how well does what I measure
    compare to what I put in".

    THREE CURVES, and the distinction is the whole point (same convention as
    `_plot_level1`, kept deliberately identical so the two figures read the
    same way):

      1. INTRINSIC (grey)  -- the source as DEFINED. Un-smeared, un-binned.
         NOTHING should ever be compared against this, and the gap between it
         and curve 2 is the honest visual answer to "how much does the
         measurement flatten my input".
      2. TRUTH (black)     -- `predict_binned_observables`: what a CORRECT
         pipeline returns after the PSF smears the source and the light is
         averaged into radial bins. This is the prediction of the MEASUREMENT.
      3. RECOVERED (red)   -- what the extraction returned, with bootstrap
         errors.

    The test is 3 vs 2. Curve 1 is context, not a target.

    Parameters
    ----------
    boot : the Stage-3 bootstrap summary -- `run_measure` /
        `measure_all_bins` / `run_measurement_suite` output. Needs
        `centroid_v_fid/_lo/_hi`; the flux panel additionally needs
        `total_flux_fid/_lo/_hi`, which only exist if the bootstrap ran with
        compute_side_ratio=True. `r_edges` is read from here when
        `bins_kpc` is not passed.
    model : the injected `SourceModel`.
    real : optional (v, err) arrays of your REAL measured table, overplotted
        in blue on the velocity panel. Same length as the bins, or shorter
        (extra real bins beyond the injected binning are dropped).
    scale : flux normalization. None = fit one scalar as the median ratio
        recovered/truth. A number applies it. 1.0 compares absolutely.

    ON THE FLUX NORMALIZATION -- read this before believing the left panel.
    Absolute flux is NOT currently testable, for two independent reasons, so
    the default is a one-parameter rescale and the panel tests SHAPE only:

      * `L_INJ` is in the injector's own units against the pipeline's
        `L_kpc2` (noted at level 2, still open), and
      * the per-km/s vs per-Angstrom line-units bug (2026-08-19) is NOT yet
        applied, so the injected flux is low by `C_KMS / lambda_obs` -- 72.5x
        at z = 2.4.

    That second one makes the fitted scale a DIAGNOSTIC rather than a
    nuisance: if the line-units bug is the only normalization error left, the
    fitted scale should come back at `C_KMS / observed_lya_wavelength(z)`.
    The function prints both so you can read the ratio directly. A fitted
    scale that matches predicts the bug is the whole story; one that does not
    means something else is also wrong, and the residual factor is printed.

    Returns
    -------
    (fig, diag) -- diag carries `scale`, `scale_expected`, `scale_residual`,
    `dv_truth`, `dv_rec`, `dv_err`, `pull`, `chi2`, `ndof`, and
    `sb_ratio`, so the numbers behind the figure are usable without
    re-reading it off the axes.
    """
    import matplotlib.pyplot as plt

    if bins_kpc is None:
        if "r_edges" not in boot:
            raise KeyError(
                "no bins_kpc passed and boot has no 'r_edges' -- pass "
                "bins_kpc=BINS explicitly."
            )
        bins_kpc = boot["r_edges"]
    bins_kpc = np.asarray(bins_kpc, dtype=float)

    if truth is None:
        truth = predict_binned_observables(model, z, bins_kpc, seeing_fwhm,
                                           kpc_per_arcsec, n_fine=n_fine)
    r_mid = truth["r_mid"]
    nb = r_mid.size

    def _fid_err(stem):
        if f"{stem}_fid" not in boot:
            return None, None
        fid = np.asarray(boot[f"{stem}_fid"], dtype=float)[:nb]
        lo = np.asarray(boot.get(f"{stem}_lo", np.full(nb, np.nan)), float)[:nb]
        hi = np.asarray(boot.get(f"{stem}_hi", np.full(nb, np.nan)), float)[:nb]
        return fid, 0.5 * (hi - lo)

    v_rec, v_err = _fid_err("centroid_v")
    if v_rec is None:
        raise KeyError("boot has no 'centroid_v_fid' -- is this the bootstrap "
                       "summary, or did you pass the stacks dict?")
    f_rec, f_err = _fid_err("total_flux")

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    r_fine = np.logspace(np.log10(max(float(bins_kpc[1]) * 0.3, 1.0)),
                         np.log10(float(bins_kpc[-1])), 800)
    diag = {"r_mid": r_mid, "dv_truth": truth["dv"],
            "dv_rec": v_rec, "dv_err": v_err}

    # ---------------- panel 1+2: surface brightness -----------------------
    sb_t = np.asarray(truth["sb"], dtype=float)
    if f_rec is None:
        for a in ax[:2]:
            a.text(0.5, 0.5, "no total_flux_fid\n(bootstrap ran with\n"
                             "compute_side_ratio=False)",
                   ha="center", va="center", transform=a.transAxes,
                   fontsize=9, color="0.4")
            a.set_xticks([]); a.set_yticks([])
        diag.update(scale=np.nan, scale_expected=np.nan,
                    scale_residual=np.nan, sb_ratio=None)
    else:
        ok = np.isfinite(f_rec) & np.isfinite(sb_t) & (sb_t > 0)
        if scale is None:
            scale = float(np.median(f_rec[ok] / sb_t[ok])) if ok.any() else 1.0
        lam_obs = observed_lya_wavelength(z)
        scale_expected = C_KMS / float(lam_obs)
        diag.update(scale=float(scale), scale_expected=scale_expected,
                    scale_residual=float(scale) / scale_expected)

        # curve 1: intrinsic, tied to truth by the MEDIAN bin ratio (never the
        # outermost -- that one sits on the r_trunc cut, see _plot_level1)
        if model.profile != "point":
            I_int = model.I(r_fine)
            I_at_bin = model.I(r_mid)
            good = I_at_bin > 0
            anchor_ = (np.median(sb_t[good] / I_at_bin[good])
                       if good.any() else 1.0)
            ax[0].plot(r_fine, I_int * anchor_ * scale, color="0.65", lw=1.6,
                       label="intrinsic (no PSF, no binning)")

        ax[0].step(r_mid, sb_t * scale, where="mid", color="k", lw=1.6,
                   label="truth (PSF + binned)")
        ax[0].errorbar(r_mid, f_rec, yerr=f_err, fmt="o", ms=4.5,
                       color="crimson", lw=1.3, capsize=2.5,
                       label="recovered")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("r [kpc]"); ax[0].set_ylabel("surface brightness")
        ax[0].set_title(f"flux profile  (x{scale:.3g} applied)", fontsize=10)
        ax[0].legend(fontsize=8, frameon=False)

        ratio = np.where(sb_t > 0, f_rec / (sb_t * scale), np.nan)
        rerr = np.where(sb_t > 0, f_err / (sb_t * scale), np.nan)
        diag["sb_ratio"] = ratio
        ax[1].axhline(1.0, color="k", lw=1.2)
        ax[1].errorbar(r_mid, ratio, yerr=rerr, fmt="o", ms=4.5,
                       color="crimson", lw=1.3, capsize=2.5)
        if model.profile != "point":
            I_at_bin = model.I(r_mid)
            good = I_at_bin > 0
            anchor_ = (np.median(sb_t[good] / I_at_bin[good])
                       if good.any() else 1.0)
            ax[1].plot(r_mid[good], sb_t[good] / (I_at_bin[good] * anchor_),
                       color="0.65", lw=1.5, ls="--",
                       label="truth / intrinsic\n(= what the PSF+binning cost)")
            ax[1].legend(fontsize=8, frameon=False, loc="lower right")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("r [kpc]"); ax[1].set_ylabel("recovered / truth")
        ax[1].set_title("flux residual", fontsize=10)

    # ---------------- panel 3: velocity -----------------------------------
    ax[2].axhline(0.0, color="0.8", lw=1.0)
    ax[2].plot(r_fine, model.velocity(r_fine), color="0.65", lw=1.6,
               label="intrinsic dv(r)")
    ax[2].step(r_mid, truth["dv"], where="mid", color="k", lw=1.6,
               label="truth (PSF + binned)")
    ax[2].errorbar(r_mid, v_rec, yerr=v_err, fmt="o", ms=4.5, color="crimson",
                   lw=1.3, capsize=2.5, label="recovered")
    if real is not None:
        rv, re_ = (np.asarray(a, dtype=float) for a in real)
        m = min(rv.size, nb)
        ax[2].errorbar(r_mid[:m], rv[:m], yerr=re_[:m], fmt="s", ms=4.0,
                       color="tab:blue", lw=1.2, capsize=2.5, alpha=0.85,
                       label="REAL data")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("r [kpc]"); ax[2].set_ylabel("dv [km/s]")
    ax[2].set_title("velocity", fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)

    # ---------------- the numbers ----------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        pull = (v_rec - truth["dv"]) / v_err
    good = np.isfinite(pull)
    diag["pull"] = pull
    diag["chi2"] = float(np.nansum(pull[good] ** 2))
    diag["ndof"] = int(good.sum())

    print(f"{'bin':>12} {'truth':>8} {'recovered':>16} {'pull':>7}"
          f"{'   flux rec/truth' if f_rec is not None else ''}")
    print("-" * (46 + (18 if f_rec is not None else 0)))
    for i in range(nb):
        line = (f"{bins_kpc[i]:5.0f}-{bins_kpc[i+1]:<6.0f}"
                f"{truth['dv'][i]:8.1f} {v_rec[i]:9.1f} +/-{v_err[i]:5.1f}"
                f"{pull[i]:7.1f}")
        if f_rec is not None and np.isfinite(diag["sb_ratio"][i]):
            line += f"      {diag['sb_ratio'][i]:8.3f}"
        print(line)
    print(f"\nvelocity chi2 vs truth = {diag['chi2']:.1f} / {diag['ndof']}"
          f"   (well-calibrated errors -> ~1 per bin)")
    if f_rec is not None:
        print(f"flux scale fitted      = {diag['scale']:.4g}")
        print(f"  expected if the per-km/s line-units bug is the only "
              f"normalization error (C_KMS/lambda_obs at z={z}) "
              f"= {diag['scale_expected']:.4g}")
        print(f"  residual factor      = {diag['scale_residual']:.4g}   "
              f"(1.0 => that bug explains all of it)")
    fig.tight_layout()
    return fig, diag


def build_velocity_response_matrix(model, z, bins_kpc, seeing_fwhm,
                                   kpc_per_arcsec, amp=100.0, n_fine=4000,
                                   progress=True, check=True):
    """The PSF+binning operator M, with `truth = M @ dv_intrinsic`.

    Binned truth dv is a FLUX-WEIGHTED AVERAGE of shell velocities, so the
    truth generator is LINEAR in dv(r) at fixed spatial profile. That means
    the deconvolution does not need an injection loop at all: build M once
    from unit basis vectors, invert it against the measured table, done. The
    injection's job is to prove `predict_binned_observables` is right (level 6,
    2026-08-20, chi2 10.3/11); once it is, everything else is linear algebra.

    Linearity is NOT exact -- the estimator is the median-50%-flux centroid of
    a superposition, not the mean -- so `check=True` verifies it on a known
    staircase and prints the worst discrepancy. Measured 2026-08-20: 1.16 km/s
    on a 5 km/s velocity grid, i.e. sub-grid, so the linear treatment is good.

    Rows sum to ~1 by construction. Measured deficits of a few 0.1% (0.994 in
    the mid bins) are flux weight scattered outside the modelled radial range,
    NOT an error -- do not renormalize without deciding what velocity you are
    asserting for the missing light.

    `amp` is the basis amplitude in km/s. It divides out, so it only sets how
    far the (mildly) nonlinear estimator is probed from zero; keep it near the
    amplitude you actually intend to measure.

    Costs one `predict_binned_observables` call per bin, plus one for the
    check. With `_TRUTH_KERNEL_CACHE` live that is ~1 kernel build total; if
    you are running a module that predates the cache it is one build PER CALL
    (measured: 52 min for 11 bins vs ~5 min cached). Reload the module.

    Returns M, shape (nbin, nbin).
    """
    bins_kpc = np.asarray(bins_kpc, dtype=float)
    nbin = bins_kpc.size - 1
    M = np.zeros((nbin, nbin))

    rng = range(nbin)
    if progress:
        try:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc="response matrix")
        except ImportError:
            pass

    def _truth(dv_vec):
        import copy
        m = copy.copy(model)
        m.velocity = BinnedVelocity(bins_kpc, np.asarray(dv_vec, float),
                                    dv_out=0.0)
        return predict_binned_observables(m, z, bins_kpc, seeing_fwhm,
                                          kpc_per_arcsec, n_fine=n_fine)["dv"]

    for j in rng:
        e = np.zeros(nbin); e[j] = float(amp)
        M[:, j] = _truth(e) / float(amp)

    if check:
        chk = np.linspace(float(amp), 0.0, nbin)
        worst = float(np.nanmax(np.abs(M @ chk - _truth(chk))))
        print(f"row sums (want ~1): {np.round(M.sum(1), 3)}")
        print(f"cond(M)           : {np.linalg.cond(M):.4g}")
        print(f"linearity, worst  : {worst:.2f} km/s"
              f"   ({'OK -- linear inversion is valid' if worst < 5 else 'TOO LARGE -- fit numerically instead'})")
    return M


def deconvolve_velocity(M, v_meas, v_err, rcond=1e-10, merge=None):
    """Invert `truth = M @ dv_intrinsic` for the intrinsic velocity profile.

    THE POINT OF THE WHOLE INJECTION EXERCISE. `v_meas` is the measured
    centroid table; the return is what the gas is actually doing, with the
    measurement's PSF and radial binning divided out and the covariance
    propagated properly.

    The dilution is NOT a uniform shrinkage toward zero -- it is a
    REDISTRIBUTION. In the 2026-08-20 level-6 run, intrinsic 300 read as a
    truth of 251.8 in [0,10] while intrinsic 200 read as 232.2 in [10,16]:
    light from the fast core leaks outward and pushes the SECOND bin up while
    pulling the first down. A scalar correction factor gets the sign wrong.

    `merge` : optional list of index lists to combine before inverting, e.g.
        [[0],[1],[2],[3],[4],[5],[6],[7],[8,9,10]] to collapse the outer bins
        into one. Use it when cond(M) is large -- the outer rows of M are
        nearly identical because the PSF cannot separate 600-1000 from
        1000-2000 kpc, and pinv will happily return huge anticorrelated
        velocities there. Merging states the resolution limit honestly instead
        of hiding it in the error bars.

    Returns dict: dv, err, cov, cond, amplification (err_out/err_in per bin),
    and `resid` = M @ dv - v_meas, which should be ~0 and is a bug check.
    """
    M = np.asarray(M, dtype=float)
    v = np.asarray(v_meas, dtype=float)
    e = np.asarray(v_err, dtype=float)
    if v.size != M.shape[0]:
        raise ValueError(f"v_meas has {v.size} bins but M is {M.shape}")

    if merge is not None:
        # average the measured bins in each group (inverse-variance), and sum
        # the corresponding COLUMNS of M (the intrinsic velocities being
        # merged are asserted equal, so their responses add)
        rows, vv, ee = [], [], []
        for g in merge:
            g = list(g)
            w = 1.0 / e[g] ** 2
            vv.append(float((v[g] * w).sum() / w.sum()))
            ee.append(float(1.0 / np.sqrt(w.sum())))
            rows.append((M[g, :] * (w / w.sum())[:, None]).sum(0))
        M = np.array([[r[list(g)].sum() for g in merge] for r in rows])
        v, e = np.array(vv), np.array(ee)

    Minv = np.linalg.pinv(M, rcond=rcond)
    cov = Minv @ np.diag(e ** 2) @ Minv.T
    dv = Minv @ v
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"dv": dv, "err": err, "cov": cov,
            "cond": float(np.linalg.cond(M)),
            "amplification": err / e,
            "resid": M @ dv - v}


def check_background_annulus_negligible(model, kpc_per_arcsec,
                                        bg_inner_arcsec=80.0,
                                        bg_outer_arcsec=90.0,
                                        bg_noise=None, verbose=True):
    """Spec sec 2.2 / 7.7: the injected profile really is negligible at 80-90".

    Unlike with real data we can PROVE this rather than assume it -- so the
    spec asks for a printed assertion at injection setup, not a paragraph of
    reasoning. If it fails, the injected profile is wrong, not the placement.

    This is what licenses the loose 30" minimum separation: the naive
    requirement (no injection inside another's background annulus) caps the
    whole experiment at ~70 objects across both fields, and it is the wrong
    criterion anyway, because the annulus sits at ~660-740 kpc at z ~ 2.3
    where an h1 ~ 17 kpc profile has effectively no flux.

    NOTE: compare against the RAW, pre-mask background (BG_RAW), not the
    smoothed BACKGROUND HDU -- injected line flux lands inside the region
    the background pipeline masks and interpolates across (spec sec 3.6).
    """
    if model.profile == "point":
        if verbose:
            print("[PASS] point source: zero flux at any r > 0 by construction")
        return True, 0.0

    r_in = bg_inner_arcsec * kpc_per_arcsec
    r_out = bg_outer_arcsec * kpc_per_arcsec
    rr = np.linspace(r_in, r_out, 2000)
    flux_in_annulus = float(np.trapz(model.I(rr) * 2 * np.pi * rr, rr)) * model.L
    frac = flux_in_annulus / model.L
    mean_sb = flux_in_annulus / (np.pi * (r_out ** 2 - r_in ** 2))

    if verbose:
        print(f"--- background annulus check ({bg_inner_arcsec:.0f}-"
              f"{bg_outer_arcsec:.0f}\" = {r_in:.0f}-{r_out:.0f} kpc)")
        print(f"    flux in annulus       : {flux_in_annulus:.4g}")
        print(f"    fraction of total L   : {frac:.3e}")
        print(f"    mean SB in annulus    : {mean_sb:.4g} /kpc^2")
        if bg_noise is not None:
            ratio = mean_sb / bg_noise
            print(f"    vs background noise   : {ratio:.3e}  "
                  f"({'PASS' if ratio < 0.05 else 'FAIL'} at the 5% ceiling)")
        else:
            print("    [!] pass bg_noise= (from the RAW pre-mask background, "
                  "not the smoothed BACKGROUND HDU) to turn this into a "
                  "pass/fail rather than a number")
    return (frac < 1e-3), frac


def shot_noise_ratio(L_median, sky_variance, verbose=True):
    """Spec sec 1.5: quantify source shot noise against sky, once, for Methods.

    The error array is deliberately NOT injected -- VDFI is sky-dominated at
    these fluxes, so the source's own shot noise is negligible against the
    background, and noiseless signal on real noise is exactly what isolates
    pipeline bias from source Poisson scatter. That is a decision, recorded
    so it is not later mistaken for an oversight. It does mean the injected
    sample's true per-object noise is very slightly LOWER than a real object
    of the same flux.

    If this ratio is not << 1, the decision must be revisited.
    """
    ratio = float(np.sqrt(max(L_median, 0.0)) / np.sqrt(max(sky_variance, 1e-30)))
    if verbose:
        verdict = "negligible" if ratio < 0.1 else "NOT negligible -- revisit sec 1.5"
        print(f"source shot noise / sky noise = {ratio:.3e}  ({verdict})")
    return ratio


# =====================================================================
# 7.  THE CUBE OVERLAY
#     Spec sec 1.2. Build-order item 2, and the thing that gates ladder
#     levels 2-7.
# =====================================================================
#
# WHY A WRAPPER AT ALL. `load_vdfi_field` opens the flux and error FITS with
# memmap=True and deliberately keeps the HDUList handles alive (_flux_hdul /
# _error_hdul) so Stage 1 can read lazily over many hours. You cannot write
# into vdfi['spectra'] in place, and a materialized copy is ~16 GB.
#
# You do not need one. Only fibers within roughly the outermost radial bin of
# an injected source are modified -- a few hundred per source per exposure at
# the 244" cut. At ~450 sources that is a few hundred MB of patch data against
# a 16 GB cube.
#
# THE INDEXING AUDIT (spec sec 1.2, RE-RUN 2026-08-17 against the current
# extract.py because the spec's line numbers had drifted). Every read of the
# cube in the whole package:
#
#     extract.py:1041   spectra[e, bg_sel]      scalar + bool mask
#     extract.py:1149   spectra[e, sel]         scalar + bool mask
#     extract.py:812    spectra.shape[-1]       attribute
#     extract.py:218    whole array -> make_spectral_continuum_mask
#
# That is the entire surface. core.py, plotting.py, background.py, analysis.py,
# single.py, measure.py, validation.py, splits.py and multicat.py do not index
# vdfi['spectra'] at all -- they receive already-extracted products. The
# spec's v1 audit listed core.py and plotting.py sites; those are gone from
# the current code. The wrapper is therefore SMALLER than the spec provisions
# for, and option B (materialize the touched exposures) stays unbuilt.
#
# THE DESIGN RULE. Any indexing form outside that surface RAISES. A silent
# fallthrough to the raw memmap would return un-injected data and nothing
# downstream would notice -- the extraction would run to completion, the FITS
# would be written, the profile would come back flat, and the only symptom
# would be "the injection didn't work" with 7 hr of runtime already spent. A
# wrapper that supports arbitrary indexing is just a slow, buggy ndarray.
#
# THE FULL-WIDTH STORE WAS THE WRONG SHAPE (2026-08-19, found by a dead
# kernel at ladder level 6, n = 3). The spec's sizing -- "a few hundred
# fibers per source per exposure, a few hundred MB at ~450 sources" -- is
# wrong on BOTH factors. Measured on COSMOS at the 2000 kpc (242") cut:
#
#     4,593 fibers per source per exposure, not "a few hundred"
#     x 132 exposures            =   606,308 fiber-exposures per source
#     x 1036 px x 4 B (float32)  =      2.51 GB per source
#
# and it is exactly linear until it SATURATES at the whole cube:
# 132 x 34944 x 1036 x 4 = 19.1 GB. Three sources already touch 39.4% of all
# fiber-exposures, so ~8 sources reaches the 16 GB copy this class exists to
# avoid -- arrived at from below, one patch at a time, with no error.
#
# The fix is not fewer fibers, it is fewer COLUMNS. An injected patch is a
# pure emission line with no continuum: at z = 2.3 the line sits near
# 4011 A with sigma_obs = 320 km/s = 4.3 A, and the axis is ~2 A/px, so
# sigma is ~2.1 PIXELS out of 1036. A full-width patch is ~97% zeros.
# Storing each patch on its own trimmed, aligned wavelength block cuts the
# store by ~20-30x (7.5 GB -> ~300 MB at n = 3) with a discarded flux bound
# of 1e-9 of the patch peak. It also removes the growth in per-exposure cost
# late in a run, which was allocator pressure from repeatedly rebuilding
# full-width merge buffers, not template work.

# Relative trim threshold on a patch's own peak. 1e-9 is +/-6.4 sigma for a
# Gaussian line -- far below float32's rounding on the base cube.
PATCH_TRIM_REL = 1e-9

# Stored blocks are snapped outward to a multiple of this many pixels, so
# that patches from sources at the same redshift land on the SAME window and
# merge into one block instead of accumulating one block per source. 32 px
# is ~15 sigma at z = 2.3, i.e. wide enough that alignment rarely costs a
# second block and narrow enough to keep the ~30x saving.
PATCH_BLOCK_ALIGN = 32


class InjectedSpectra:
    """Read-only view of the memmapped cube with sparse additive patches.

    Supports exactly the access pattern the extraction uses -- `spectra[e]`
    and `spectra[e, bool_mask]` -- and raises on everything else.

    Parameters
    ----------
    base : the memmapped (nexp, nfib, nwave) array from
        `load_vdfi_field`, i.e. `vdfi["spectra"]`. Never modified.
    zero_base : bool. If True, the base contributes `base * 0` rather than
        `base` -- the ladder-levels-2-to-4 "zero cube" of spec sec 7.2.

        THIS IS `base * 0`, NOT `np.zeros` (decision, Austin, 2026-08-17).
        The distinction is load-bearing. `base * 0` propagates NaN, so the
        real cube's dead- and masked-fiber pattern survives into the zero
        run; `np.zeros` erases it. With a clean zero array the extraction
        would exercise `combine_fibers`' finite-count logic, `_maybe_smooth`
        and every nan-aware reduction in a regime the real run never sees,
        and a masking bug would sail through levels 2-4 and first appear at
        level 6 -- on real data, where it is expensive to find and where two
        things changed at once.

        It also costs nothing: the base slice is read either way, because
        the NaN pattern IS the base.
    dtype : output dtype. None -> the base's, which is float32. Patches are
        cast to this on add.
    align : block alignment in wavelength pixels (PATCH_BLOCK_ALIGN). Set
        larger to force more merging at higher memory, or to 1 for the
        tightest possible blocks and one block per distinct line position.

    MEMORY -- read this before raising the source count
    ---------------------------------------------------
    The store is `sum over blocks of (fibers x block_width x 4 B)`. Patches
    are trimmed to the line's own wavelength support, so block_width is
    ~32-64 px against a 1036 px axis rather than the full width. See the
    block comment above this class for the measurement that forced this and
    for the 19 GB ceiling the full-width version walked into at ~8 sources.

    `repr()` prints `width=<mean block px>/<nwave>`; if that ratio ever
    approaches 1 the trim has stopped working -- most likely because
    something with a continuum, rather than a pure line, is being injected.

    Notes
    -----
    **This class does not enforce spec sec 1.4** (build the bad-fiber mask
    on clean, un-injected data). It cannot: `make_spectral_continuum_mask`
    consumes the cube through `spectra[i]`, which is a legitimate access
    that `plot_fiber_overlay` also needs, so refusing it here would break a
    supported caller to catch a sequencing error. The ordering is enforced
    at the `inject_sources` call site instead, by requiring the mask as an
    argument -- you cannot inject without already holding a mask, and a mask
    you already hold was built before the injection existed.

    `n_exposure_slice_reads` counts whole-exposure reads so that ordering
    can still be audited after the fact: a nonzero count on a run whose mask
    was passed in is fine (plotting), a nonzero count *before* the first
    patch was added is the sec 1.4 violation.
    """

    __slots__ = ("_base", "_patches", "_zero", "_dtype", "_align",
                 "n_exposure_slice_reads", "_frozen")

    def __init__(self, base, zero_base=False, dtype=None,
                 align=PATCH_BLOCK_ALIGN):
        self._base = base
        # {exposure: [ {"lam0": int, "lam1": int,
        #               "idx": int64 (k,), "delta": dtype (k, lam1-lam0)}, ...]}
        #
        # A LIST OF WAVELENGTH BLOCKS, not one full-width array per exposure
        # (2026-08-19). See the MEMORY note in the class docstring: an
        # injected patch is a pure emission line ~2 px wide against a 1036 px
        # axis, so a full-width store is ~97% zeros and the overlay saturates
        # at the 19 GB full-cube copy it exists to avoid, at about 8 sources.
        self._patches = {}
        self._zero = bool(zero_base)
        self._dtype = np.dtype(dtype) if dtype is not None else base.dtype
        self._align = int(align)
        self.n_exposure_slice_reads = 0
        self._frozen = False

    # ---- wavelength trimming -----------------------------------------

    def _window(self, d):
        """The aligned column span [lam0, lam1) worth storing for `delta`.

        Trims at `PATCH_TRIM_REL` of the patch's OWN peak, which for a
        Gaussian line is +/-6.4 sigma -- the discarded flux is bounded at
        ~1e-9 of the peak per pixel, orders below float32's rounding on the
        base cube, so this is lossless in every sense that matters. The
        threshold is RELATIVE on purpose: an absolute one would depend on
        L_INJ and would silently trim harder for a faint source than a
        bright one.

        Then snaps outward to a multiple of `self._align`. Alignment is what
        makes blocks MERGE: two sources at the same redshift produce the
        same aligned window and share one block, so the block count per
        exposure stays at ~(number of distinct redshifts) rather than
        (number of sources). Without it, 400 bootstrapped-z sources would
        leave 400 blocks per exposure and the read path would loop all of
        them.
        """
        nw = d.shape[1]
        peak = float(np.max(d)) if d.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            return 0, 0                     # an all-zero patch stores nothing
        cols = np.flatnonzero(np.max(d, axis=0) > peak * PATCH_TRIM_REL)
        if cols.size == 0:
            return 0, 0
        lo, hi = int(cols[0]), int(cols[-1]) + 1
        a = max(self._align, 1)
        lo = (lo // a) * a
        hi = min(int(np.ceil(hi / a)) * a, nw)
        # A patch that is wide anyway gains nothing from blocking and loses
        # the merge; store it whole rather than in a near-full-width block.
        if (hi - lo) > nw // 2:
            return 0, nw
        return lo, hi

    # ---- construction ------------------------------------------------

    def add_patch(self, e, fiber_idx, delta):
        """Accumulate flux to add to fibers `fiber_idx` of exposure `e`.

        ACCUMULATE, not assign. Two injected sources whose outer bins
        overlap must both deposit into the shared fibers -- overwriting
        would silently delete one of them, and at the 244" cut with
        min_sep = 30" (spec sec 2.2) overlap is the expected case, not the
        edge case.

        `delta` is (len(fiber_idx), nwave), the output of
        `build_source_template` restricted to the fibers `select_nearby_
        fibers` kept. Never negative, never NaN -- asserted, because a NaN
        patch would propagate into the extraction's nan-aware reductions
        and present as missing coverage rather than as a bug.
        """
        if self._frozen:
            raise RuntimeError(
                "this InjectedSpectra is frozen -- patches cannot be added "
                "after an extraction has read from it. Build the full patch "
                "set first, then extract; a mid-extraction add would make "
                "the cube's contents depend on galaxy ordering."
            )
        e = int(e)
        idx = np.ascontiguousarray(fiber_idx, dtype=np.int64).ravel()
        d = np.ascontiguousarray(delta, dtype=self._dtype)
        if d.ndim != 2 or d.shape[0] != idx.size:
            raise ValueError(
                f"delta must be (len(fiber_idx), nwave) = "
                f"({idx.size}, {self._base.shape[-1]}), got {d.shape}")
        if d.shape[1] != self._base.shape[-1]:
            raise ValueError(
                f"delta has {d.shape[1]} wavelength pixels, cube has "
                f"{self._base.shape[-1]}")
        if not np.all(np.isfinite(d)):
            raise ValueError(
                "delta contains non-finite values. An injected template is a "
                "noiseless model and must be finite everywhere; a NaN here "
                "would propagate into the extraction's nan-aware reductions "
                "and look like missing fiber coverage, not like a bug.")
        if np.any(d < 0):
            raise ValueError(
                "delta contains negative values. Lya emission is added, "
                "never subtracted; a negative patch means a sign error in "
                "the template or in the profile normalization.")

        # TRIM IN WAVELENGTH BEFORE STORING. Everything below operates on the
        # trimmed span only; `delta` itself is never retained at full width.
        lam0, lam1 = self._window(d)
        if lam1 <= lam0:
            return                          # nothing above the trim threshold
        d = np.ascontiguousarray(d[:, lam0:lam1])

        # Duplicate fibers within a single add_patch call are summed rather
        # than kept as duplicate rows, so the invariant "idx is sorted and
        # unique" holds per block from the start and the read path can use
        # searchsorted.
        uniq, inv = np.unique(idx, return_inverse=True)
        acc = np.zeros((uniq.size, d.shape[1]), dtype=self._dtype)
        np.add.at(acc, inv, d)

        blocks = self._patches.setdefault(e, [])
        for blk in blocks:
            if blk["lam0"] == lam0 and blk["lam1"] == lam1:
                # Same aligned window -> merge into the existing sorted/unique
                # index set, exactly as the pre-2026-08-19 single-array store
                # did, but at block width instead of full width.
                old_idx, old_d = blk["idx"], blk["delta"]
                new_idx = np.union1d(old_idx, uniq)
                merged = np.zeros((new_idx.size, d.shape[1]), dtype=self._dtype)
                merged[np.searchsorted(new_idx, old_idx)] = old_d
                np.add.at(merged, np.searchsorted(new_idx, uniq), acc)
                blk["idx"], blk["delta"] = new_idx, merged
                return
        blocks.append({"lam0": lam0, "lam1": lam1, "idx": uniq, "delta": acc})

    def freeze(self):
        """Close the patch set. Called once, before extraction."""
        self._frozen = True
        return self

    # ---- the supported read surface ----------------------------------

    def __getitem__(self, key):
        # ---- spectra[e, bool_mask] : extract.py:1041 and :1149 ----
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError(
                    f"InjectedSpectra supports spectra[e] and "
                    f"spectra[e, bool_mask] only; got a {len(key)}-tuple. "
                    f"See spec sec 1.2 -- if a new access pattern is genuinely "
                    f"needed, widen the audit and this class deliberately "
                    f"rather than adding a fallthrough.")
            e, sel = key
            if not np.isscalar(e) or isinstance(e, (bool, np.bool_)):
                raise TypeError(
                    f"the first index must be a scalar exposure number, got "
                    f"{type(e).__name__}. Slicing across exposures is not "
                    f"supported: the patch store is keyed by exposure and a "
                    f"multi-exposure read would have to materialize the cube, "
                    f"which is the 16 GB copy this class exists to avoid.")
            sel = np.asarray(sel)
            if sel.dtype != np.bool_ or sel.ndim != 1:
                raise TypeError(
                    f"the second index must be a 1-D boolean fiber mask, got "
                    f"dtype={sel.dtype} ndim={sel.ndim}. Integer fancy "
                    f"indexing is rejected on purpose -- it is the one form "
                    f"whose row ORDER differs from the mask form, so "
                    f"supporting both silently invites a patch applied to the "
                    f"wrong fiber.")
            if sel.size != self._base.shape[1]:
                raise ValueError(
                    f"fiber mask has {sel.size} entries, cube has "
                    f"{self._base.shape[1]} fibers")

            e = int(e)
            out = np.asarray(self._base[e, sel], dtype=self._dtype)
            if self._zero:
                out = out * 0.0     # NaN survives; see the class docstring
            blocks = self._patches.get(e)
            if not blocks:
                return out
            # Map patched GLOBAL fiber indices onto ROW positions in `out`.
            # out's rows are the selected fibers in ascending global order --
            # np.flatnonzero(sel) -- so a patched fiber's row is its rank
            # among the selected ones. cumsum-1 gives that rank in O(nfib)
            # without materializing flatnonzero for a mask that may select
            # tens of thousands of fibers. Computed ONCE and reused across
            # blocks.
            rank = None
            for blk in blocks:
                hit = sel[blk["idx"]]
                if not hit.any():
                    continue
                if rank is None:
                    rank = np.cumsum(sel) - 1              # (nfib,)
                rows = rank[blk["idx"][hit]]
                # rows is unique within a block (idx is unique), so this
                # fancy-index += cannot double-accumulate.
                out[rows, blk["lam0"]:blk["lam1"]] += blk["delta"][hit]
            return out

        # ---- spectra[e] : make_spectral_continuum_mask, plot_fiber_overlay ----
        if np.isscalar(key) and not isinstance(key, (bool, np.bool_)):
            e = int(key)
            self.n_exposure_slice_reads += 1
            # COPY, NOT asarray. `self._base[e]` is BASIC slicing, so it is a
            # VIEW of the cube -- and `np.asarray(view, dtype=<same dtype>)`
            # hands the view straight back. Adding patches into it therefore
            # wrote INTO THE BASE, so a second read of the same exposure
            # returned the patch twice and the un-injected cube was gone.
            #
            # Latent until 2026-08-19 for two reasons that both stop being
            # true at ladder level 6: on the zero cube `out = out * 0.0`
            # allocates a fresh array before any patch is applied, and the
            # OTHER read form -- spectra[e, bool_mask], which is what the
            # extraction actually calls (extract.py:1041, :1149) -- is
            # ADVANCED indexing and so already copies. This branch is only
            # reached by make_spectral_continuum_mask and plot_fiber_overlay,
            # and sec 1.4 puts the former before injection. On a real
            # read-only memmap it would have raised rather than corrupted;
            # on anything writable it corrupts silently.
            out = np.array(self._base[e], dtype=self._dtype, copy=True)
            if self._zero:
                out = out * 0.0
            for blk in self._patches.get(e, ()):
                out[blk["idx"], blk["lam0"]:blk["lam1"]] += blk["delta"]
            return out

        raise TypeError(
            f"InjectedSpectra does not support indexing with "
            f"{type(key).__name__}. The supported surface is spectra[e] and "
            f"spectra[e, bool_mask] -- see spec sec 1.2 and the audit comment "
            f"above this class. Raising here is the design: a fallthrough to "
            f"the raw memmap would return UN-INJECTED data, the extraction "
            f"would complete normally, and the only symptom would be a flat "
            f"profile after hours of runtime.")

    # ---- the attributes the extraction reads -------------------------

    @property
    def shape(self):
        return self._base.shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return self._base.ndim

    # ---- diagnostics -------------------------------------------------

    @property
    def n_patched_fibers(self):
        """Total patched (exposure, fiber) pairs.

        DISTINCT fibers, deduplicated across blocks -- a fiber carrying two
        sources at different redshifts lives in two blocks but is one
        patched fiber. Kept as a distinct count so the number stays
        comparable to the pre-blocking runs and to nfib in the extraction.
        """
        n = 0
        for blocks in self._patches.values():
            if len(blocks) == 1:
                n += blocks[0]["idx"].size
            elif blocks:
                n += np.unique(np.concatenate([b["idx"] for b in blocks])).size
        return int(n)

    @property
    def n_blocks(self):
        """Wavelength blocks summed over exposures. ~(distinct redshifts) x
        (exposures patched) once alignment has done its merging; a number
        far above that means `align` is too fine for the z spread."""
        return int(sum(len(b) for b in self._patches.values()))

    @property
    def patch_width(self):
        """Mean stored columns per block, against the cube's nwave. This is
        the compression ratio the wavelength trim is buying."""
        widths = [b["lam1"] - b["lam0"]
                  for blocks in self._patches.values() for b in blocks]
        return float(np.mean(widths)) if widths else 0.0

    @property
    def nbytes(self):
        """Patch-store footprint. The number this class exists to keep small."""
        return int(sum(b["delta"].nbytes + b["idx"].nbytes
                       for blocks in self._patches.values() for b in blocks))

    def __repr__(self):
        nw = self._base.shape[-1]
        return (f"InjectedSpectra(shape={self.shape}, "
                f"zero_base={self._zero}, "
                f"exposures_patched={len(self._patches)}, "
                f"fibers_patched={self.n_patched_fibers}, "
                f"blocks={self.n_blocks}, "
                f"width={self.patch_width:.0f}/{nw} px, "
                f"patch_store={self.nbytes / 1e6:.1f} MB)")


def check_overlay_identity(base, n_trials=3, rng_seed=11, verbose=True):
    """Spec sec 7.4: an EMPTY overlay is bit-identical to the raw memmap.

    **PASS A SMALL SLAB, NOT THE CUBE.** This is a test of indexing logic,
    which does not care how big the array is, and every trial fancy-indexes
    ~30% of the fibers across all wavelengths. Handed a real memmapped
    exposure that is hundreds of MB pulled off disk per trial, and it was
    minutes rather than seconds until n_trials dropped from 25 to 3
    (2026-08-17). Do this:

        small = np.asarray(vdfi["spectra"][:2, :3000])   # ~25 MB, in RAM
        inj.check_overlay_identity(small)

    `np.asarray` matters as much as the slice: basic slicing of a FITS
    memmap returns a VIEW, so without it every read still goes to disk.

    Cheap, and it is the only thing standing between "the overlay works" and
    "the overlay silently drops a patch". Run it before any extraction goes
    through the wrapper.

    Four checks:

      1. `spectra[e]` with zero patches -> bit-identical to `base[e]`.
      2. `spectra[e, mask]` with zero patches -> bit-identical, over random
         masks including the two that break naive index arithmetic: all-True
         and all-False.
      3. A patch applied to a KNOWN fiber lands on that fiber and NOWHERE
         else -- checked through the mask path, which is where the global ->
         row-position mapping happens and therefore where an off-by-one
         would live. This is the check that matters: check 1 and 2 pass
         trivially for a class that ignores its patches entirely.
      4. The unsupported indexing forms raise rather than falling through.

    `base` can be the real memmap or any (nexp, nfib, nwave) array; a small
    synthetic one is enough and is what the test suite should use.
    """
    rng = np.random.default_rng(rng_seed)
    nexp, nfib, nwave = base.shape
    lines, ok = [], True

    nbytes = int(np.prod(base.shape)) * 4
    if nbytes > 200e6:
        print(f"    [!] base is ~{nbytes/1e9:.1f} GB. This test only checks "
              f"indexing logic -- pass a slab "
              f"(np.asarray(spectra[:2, :3000])) instead, or it reads "
              f"gigabytes off disk for no extra coverage.")

    # ---- 1 + 2. empty overlay is a no-op --------------------------------
    empty = InjectedSpectra(base)
    same = True
    for _ in range(n_trials):
        e = int(rng.integers(nexp))
        same &= bool(np.array_equal(empty[e], np.asarray(base[e]),
                                    equal_nan=True))
        for mask in (rng.random(nfib) < 0.3,
                     np.ones(nfib, dtype=bool),
                     np.zeros(nfib, dtype=bool)):
            same &= bool(np.array_equal(empty[e, mask],
                                        np.asarray(base[e, mask]),
                                        equal_nan=True))
    ok &= same
    lines.append(f"[{'PASS' if same else 'FAIL'}] empty overlay is "
                 f"bit-identical to the raw array "
                 f"({n_trials} exposures x 3 mask forms, incl. all-True and "
                 f"all-False)")

    # ---- 3. a patch lands on exactly one fiber --------------------------
    # THREE SEPARATE VERDICTS, not one. Collapsing them cost a debugging
    # round on 2026-08-17: the combined line said "FAIL" and gave no way to
    # tell whether the patch missed, leaked, or crossed exposures.
    #
    # The fiber is chosen FINITE. On the real cube a random fiber has a real
    # chance of being dead, and nan + 3.5 = nan -- which fails the "+amp
    # exactly" assertion while the class is behaving perfectly correctly
    # (a dead fiber stays dead whether or not you inject into it, exactly as
    # it does with real data).
    e0 = int(rng.integers(nexp))
    finite_rows = np.flatnonzero(np.isfinite(np.asarray(base[e0])).all(axis=1))
    if finite_rows.size == 0:
        lines.append(f"[SKIP] patch localization -- exposure {e0} has no "
                     f"all-finite fiber to test on")
    else:
        f0 = int(rng.choice(finite_rows))
        # Amplitude scaled to the data, so the check cannot fail on float32
        # resolution when the cube's values are large.
        scale = float(np.nanmedian(np.abs(np.asarray(base[e0, f0])))) or 1.0
        amp = 1000.0 * scale
        ov = InjectedSpectra(base)
        ov.add_patch(e0, [f0], np.full((1, nwave), amp, dtype=np.float32))

        # Through the MASK path, with a mask selecting f0 plus a scatter of
        # others, so the global -> row-position mapping is genuinely
        # exercised rather than being the identity.
        mask = rng.random(nfib) < 0.4
        mask[f0] = True
        got = np.asarray(ov[e0, mask], dtype=np.float64)
        ref = np.asarray(base[e0, mask], dtype=np.float64)
        row = int(np.cumsum(mask)[f0] - 1)
        delta = got - ref

        hit = bool(np.allclose(delta[row], amp, rtol=1e-4))
        ok &= hit
        lines.append(f"[{'PASS' if hit else 'FAIL'}] patch LANDS: (exp {e0}, "
                     f"fiber {f0}) -> row {row}, recovered "
                     f"{np.nanmedian(delta[row]):.6g} vs injected {amp:.6g}")

        # Leakage. Only FINITE rows can be compared numerically; rows that
        # were NaN before must still be NaN, which is a separate assertion
        # rather than something to nan_to_num away.
        others = np.delete(np.arange(delta.shape[0]), row)
        was_nan = ~np.isfinite(ref[others]).all(axis=1)
        num_ok = bool(others.size == 0 or
                      np.all(delta[others][~was_nan] == 0.0))
        nan_ok = bool(others.size == 0 or
                      np.all(~np.isfinite(got[others][was_nan]).all(axis=1)))
        clean = num_ok and nan_ok
        ok &= clean
        lines.append(f"[{'PASS' if clean else 'FAIL'}] no LEAKAGE: "
                     f"{int((~was_nan).sum())} finite rows unchanged "
                     f"({num_ok}), {int(was_nan.sum())} NaN rows still NaN "
                     f"({nan_ok})")

        other_e = (e0 + 1) % nexp
        iso = bool(nexp == 1 or np.array_equal(
            ov[other_e, mask], np.asarray(base[other_e, mask]),
            equal_nan=True))
        ok &= iso
        lines.append(f"[{'PASS' if iso else 'FAIL'}] EXPOSURE-ISOLATED: the "
                     f"patch is absent from exposure {other_e}")

    # ---- 4. unsupported forms raise -------------------------------------
    bad = [
        ("spectra[:]", lambda: empty[:]),
        ("spectra[0:2]", lambda: empty[0:2]),
        ("spectra[e, int_array]", lambda: empty[0, np.array([1, 2, 3])]),
        ("spectra[e, slice]", lambda: empty[0, 0:5]),
        ("spectra[e, mask, wave]", lambda: empty[0, np.ones(nfib, bool), 0]),
    ]
    raised = []
    for name, fn in bad:
        try:
            fn()
        except (TypeError, ValueError, IndexError):
            raised.append(True)
        else:
            raised.append(False)
            lines.append(f"           !! {name} did NOT raise -- it returned "
                         f"data, which for a real run means UN-INJECTED data")
    good = all(raised)
    ok &= good
    lines.append(f"[{'PASS' if good else 'FAIL'}] unsupported indexing raises "
                 f"({sum(raised)}/{len(bad)} forms)")

    if verbose:
        print("--- overlay identity (spec sec 7.4)")
        print("\n".join(lines))
        print(f"--- overlay identity {'PASSED' if ok else 'FAILED'}")
    return ok, lines


# ---------------------------------------------------------------------
# 7b.  inject_sources -- the one function between load_field and
#      extract_fibers. Spec sec 1.1 and sec 1.6.
# ---------------------------------------------------------------------

INJ_REQUIRED_COLS = ("RA", "DEC", "L_INJ", "FIELD", "CATALOG", "FAMILY")


def _mask_checksum(mask):
    """Short content hash of the bad-fiber mask. Goes in the manifest so a
    run whose mask was rebuilt AFTER injection stays identifiable."""
    import hashlib
    a = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    return hashlib.md5(a.tobytes()).hexdigest()[:10]


def validate_inj_table(inj_table, config, verbose=True):
    """Spec sec 1.6: the injection table IS the run table.

    So it must carry everything `extract_fibers` reads (RA, DEC,
    config.z_col, config.mass_col) AND everything `build_run_table`
    filters on (FIELD, CATALOG), or the failure surfaces hundreds of
    galaxies into an extraction as a KeyError with nothing to do with
    injection.

    Also checks the two diagnostic columns that are LOAD-BEARING rather
    than nice-to-have: REAL_DIST, because sec 2.1 replaces the separate
    `blank` extraction with a post-hoc slice on it (a run without it
    cannot answer the placement question at all, and re-deriving it later
    against a catalog that has since been re-cut is not the same number),
    and NN_DIST, because sec 2.2 retires the whole crowding question with
    one panel binned on it.
    """
    cols = set(inj_table.colnames)
    need = set(INJ_REQUIRED_COLS) | {config.z_col, config.mass_col,
                                     config.id_col}
    missing = sorted(need - cols)
    if missing:
        raise ValueError(
            f"inj_table is missing required columns {missing}. It is also the "
            f"run table (spec sec 1.1), so it must carry every column "
            f"extract_fibers reads. See spec sec 1.6 for the full schema.")

    soft = sorted({"NN_DIST", "REAL_DIST", "N_FIB_INNER"} - cols)
    if soft and verbose:
        print(f"    [!] inj_table has no {soft}. These are diagnostic, so the "
              f"extraction will run -- but REAL_DIST is LOAD-BEARING (spec "
              f"sec 2.1 replaces the separate 'blank' pass with a post-hoc "
              f"slice on it) and NN_DIST retires the crowding question "
              f"(sec 2.2). Recompute them at DRAW time, not afterwards.")

    ids = np.asarray(inj_table[config.id_col])
    if np.unique(ids).size != ids.size:
        raise ValueError(
            f"inj_table[{config.id_col!r}] has duplicates. IDs must be unique "
            f"ACROSS FIELDS -- the real prod_combined catalog has genuine "
            f"cross-field ID collisions and they were a real bug once.")

    fields = set(np.asarray(inj_table["FIELD"]).astype(str))
    if fields != {config.normalized_field()}:
        raise ValueError(
            f"inj_table FIELD values {sorted(fields)} do not all match "
            f"config.normalized_field() = {config.normalized_field()!r}. "
            f"Fields are extracted separately and stay separate through the "
            f"diagnostics (standing practice); one table per field.")
    return True


def inject_sources(config, vdfi, inj_table, bad_fiber_mask, model, *,
                   zero_base=False, seeing_mode="field", placement_seed=-1,
                   response=None, verbose=True):
    """Add fake sources to the real fiber spectra. Spec sec 1.1 / 1.6.

    Returns
    -------
    (config_inj, vdfi_inj)

    **Both, and you must use both.** `config_inj` is `config` with
    `inject=<meta>` set, which is what makes `extract_signature()` differ
    from the fiducial one. Returning it rather than mutating the caller's
    config is deliberate: it is not possible to run an injected extraction
    against an un-injected signature, because the config you have to hand
    `extract_fibers` is the one this function gave you. Spec sec 7.5 is the
    single failure mode in the injection plan that would corrupt EXISTING
    science products -- an injected result served from, or written over, a
    fiducial cache -- and this return signature is what forecloses it.

    `vdfi_inj` is a SHALLOW copy: only `spectra` is replaced (with a frozen
    `InjectedSpectra`). Everything else, including the live `_flux_hdul` /
    `_error_hdul` memmap handles, is passed through by reference and must
    not be closed or re-opened.

    Parameters
    ----------
    bad_fiber_mask : REQUIRED, and this is the whole mechanism enforcing
        spec sec 1.4 ("build the fiber mask BEFORE injecting").

        It is not used to place flux -- an injected source deposits into
        masked fibers exactly as a real one does, and the mask is applied
        downstream in the extraction as usual. It is required because
        *holding* a mask proves it was built, and a mask built before this
        call was built on clean data. `build_bad_fiber_mask` with a
        spec_global/spec_local component consumes the WHOLE cube, so on
        injected data the injected sources would enter the percentile
        statistics that define the mask and change it for the REAL fibers
        too -- a global perturbation from a local addition. The alternative
        to a required argument is a comment saying "call this second", which
        is how silent couplings get in.

        Its checksum goes in the manifest, so a run whose mask was rebuilt
        AFTER injection is still identifiable after the fact.
    model : ONE SourceModel for the whole family, not one per row. Spec
        sec 3.3: every injected source in a family gets identical profile
        parameters and only (z, M*, L, position) vary, which is what gives
        the family an unambiguous scalar truth. Per-row models are the
        `profile_scatter` follow-on and are deliberately unsupported here.
    zero_base : the ladder levels 2-4 zero cube. `base * 0`, NOT np.zeros --
        see InjectedSpectra's docstring for why that distinction is
        load-bearing.
    seeing_mode : "field" (the field median, one scalar) or "exposure".
        **"field" for every ladder rung, including level 6.** It is
        tempting to switch to "exposure" once the cube is real, but every
        rung compares against `predict_binned_observables`, which accepts
        only a SCALAR seeing -- so matching the injector to it removes a
        known discrepancy rather than adding realism. Measured cost at
        seeing median 1.97" / sd 0.37": dv differs by <=0.06 km/s, profile
        +3.1% in the innermost bin and <1% outside. "exposure" is a
        production (level 7) setting.
    placement_seed : the RNG seed `draw_injection_positions` used. Recorded
        only; this function does not draw. -1 means "not recorded", which is
        allowed but makes the run non-reproducible from the manifest alone.

    Notes
    -----
    The error array is NOT injected (spec sec 1.5). VDFI is sky-dominated at
    these fluxes, and noiseless signal on real noise is exactly what
    isolates pipeline bias from source Poisson scatter. Quantify it once
    with `shot_noise_ratio`.
    """
    import dataclasses
    import time
    from .core import _exposure_seeing, _offsets_with_dar
    from astropy.cosmology import Planck18 as _cosmo
    import astropy.units as _u

    validate_inj_table(inj_table, config, verbose=verbose)
    if seeing_mode not in ("field", "exposure"):
        raise ValueError("seeing_mode must be 'field' or 'exposure'")
    if config.injects():
        raise ValueError(
            "this config is ALREADY an injected config (config.inject is "
            "set). Injecting twice would stack two source sets into one "
            "signature and the manifest would describe only the second. "
            "Start from the fiducial config.")
    if getattr(config, "store_spec_nobg", False):
        raise ValueError(
            "store_spec_nobg must be OFF for injected runs (spec sec 7.3). "
            "Its guard relies on the biweight being translation-equivariant "
            "under a CONSTANT per-fiber offset, and an injected source is not "
            "constant across fibers -- so the stored SPEC_NOBG would not "
            "reconstruct. Closed by decision, 2026-08-01.")

    wave = np.asarray(vdfi["wave"], dtype=float)
    RA, DEC = np.asarray(vdfi["RA"]), np.asarray(vdfi["DEC"])
    nexp = RA.shape[0]
    seeing_field = float(np.nanmedian(vdfi["seeing"]))
    r_max_kpc = float(np.asarray(config.bins, dtype=float)[-1])

    # THE FIBER CUT IS bins[-1], NOT model.r_trunc. They are the same number
    # today (both 2000 kpc) and the coincidence must not be relied on: the
    # fiber cut is a MEASUREMENT aperture and r_trunc is a SOURCE definition.
    # Warn rather than raise -- a deliberate mismatch is a legitimate
    # diagnostic (it is how the outermost-bin truncation question was
    # settled), but an accidental one means L is "flux inside a radius you
    # never measure" and that is a silent truth-value error.
    if verbose and abs(model.r_trunc - r_max_kpc) > 1e-6:
        print(f"    [!] model.r_trunc = {model.r_trunc:.0f} kpc but bins end "
              f"at {r_max_kpc:.0f}. L is defined inside r_trunc, so the two "
              f"disagreeing means part of the injected luminosity is never "
              f"measured. Intentional for a diagnostic; a bug otherwise "
              f"(spec sec 3.2).")

    overlay = InjectedSpectra(vdfi["spectra"], zero_base=zero_base)

    n_src = len(inj_table)
    n_injected = n_skipped = 0
    t0 = time.time()

    for si in range(n_src):
        L = float(inj_table["L_INJ"][si])
        if L == 0.0:
            # Family Z. No flux, so no template -- building 400 all-zero
            # patches would cost hours and add nothing. The run still goes
            # THROUGH the overlay, with zero patches, which is exactly the
            # sec 7.4 empty-overlay case exercised at production scale.
            n_skipped += 1
            continue

        ra = float(inj_table["RA"][si])
        dec = float(inj_table["DEC"][si])
        z = float(inj_table[config.z_col][si])
        kpc_per_arcsec = float(
            _cosmo.angular_diameter_distance(z).to(_u.kpc).value / 206265.0)
        cs = np.cos(np.radians(dec))

        # PER-EXPOSURE PROGRESS ON THE FIRST SOURCE ONLY.
        # Per-source progress alone is useless at n=1 -- the ladder rungs 2-6
        # all inject ONE source, so the cell printed nothing at all for
        # several minutes and was indistinguishable from a hang (2026-08-17).
        # Reporting the first source per exposure also pays off at n=400: you
        # get a warm rate and a whole-pass ETA within the first source instead
        # of after it.
        #
        # The first EXPOSURE is not representative -- it pays the shared PSF
        # table and ring-kernel setup, which is cached across every later
        # exposure AND every later source. So the ETA is quoted from the warm
        # cost, never from the average including the cold one.
        first_src = (n_injected == 0)
        t_src, t_cold = time.time(), 0.0

        for e in range(nexp):
            dra_1d = (RA[e] - ra) * cs * 3600.0
            ddec_1d = (DEC[e] - dec) * 3600.0
            keep = select_nearby_fibers(dra_1d, ddec_1d, r_max_kpc,
                                        kpc_per_arcsec)
            if not keep.any():
                continue
            dra, ddec = _offsets_with_dar(dra_1d[keep], ddec_1d[keep],
                                          vdfi.get("dar_ra"),
                                          vdfi.get("dar_dec"), e, wave.size)
            s_e = (seeing_field if seeing_mode == "field"
                   else _exposure_seeing(vdfi["seeing"], e))
            # L multiplies the template rather than entering the build. The
            # template is linear in L and build_source_template normalizes
            # against model.L internally, so scaling here is exact and lets
            # the PSF/ring caches be shared across every source in the pass.
            patch = L * build_source_template(
                model, z=z, dra=dra, ddec=ddec, seeing_fwhm=s_e, wave=wave,
                kpc_per_arcsec=kpc_per_arcsec, response=response)
            overlay.add_patch(e, np.flatnonzero(keep), patch)

            if verbose and first_src and (e < 3 or (e + 1) % 20 == 0
                                          or e == nexp - 1):
                el = time.time() - t_src
                if e == 0:
                    t_cold = el
                    print(f"    src 0 exp {e:3d}: {int(keep.sum())} fibers, "
                          f"cold {el:.1f} s (PSF table + ring kernel, paid "
                          f"once for the whole pass)", flush=True)
                else:
                    warm = (el - t_cold) / e
                    left = warm * (nexp - e - 1)
                    print(f"    src 0 exp {e:3d}: {el:6.1f} s elapsed, "
                          f"{warm:.2f} s/exp warm -> {left/60:.1f} min left "
                          f"on this source"
                          + (f", ~{(t_cold + warm*nexp) * (n_src - 1) / 60:.0f} "
                             f"min for the remaining {n_src - 1}"
                             if n_src > 1 and e == nexp - 1 else ""),
                          flush=True)

        n_injected += 1
        if verbose and (n_injected <= 2 or n_injected % 10 == 0
                        or si == n_src - 1):
            el = time.time() - t0
            per = el / max(n_injected, 1)
            print(f"    injected {n_injected}/{n_src - n_skipped}: "
                  f"{el/60:.1f} min elapsed, {per:.1f} s/source, "
                  f"store {overlay.nbytes/1e6:.0f} MB -> ETA "
                  f"{per * (n_src - n_skipped - n_injected)/60:.1f} min",
                  flush=True)

    overlay.freeze()

    meta = {
        "family": str(np.asarray(inj_table["FAMILY"])[0]) if n_src else "",
        "model": repr(model),
        "n_sources": int(n_src),
        "n_with_flux": int(n_injected),
        "zero_base": bool(zero_base),
        "seeing_mode": str(seeing_mode),
        # The bin edges belong here even though they are already a config
        # knob, because r_trunc TRACKS them (spec sec 3.2) -- a manifest that
        # records the truncation radius without recording what it was
        # tracking is not decodable months later.
        "bins": [float(b) for b in config.bins],
        "r_trunc_kpc": float(model.r_trunc),
        "lsf_kms": float(model.lsf_kms),
        "mask_checksum": _mask_checksum(bad_fiber_mask),
        "placement_seed": int(placement_seed),
    }
    config_inj = dataclasses.replace(config, inject=meta)

    if verbose:
        print(f"    injected {n_injected} sources ({n_skipped} zero-flux) "
              f"into {len(overlay._patches)}/{nexp} exposures; "
              f"patch store {overlay.nbytes/1e6:.0f} MB")
        print(f"    signature {config.extract_signature()} (fiducial) -> "
              f"{config_inj.extract_signature()} (injected)")
        if config.extract_signature() == config_inj.extract_signature():
            raise RuntimeError(
                "the injected config hashes to the SAME signature as the "
                "fiducial one. This is spec sec 7.5 -- the injected run would "
                "be served from, or written over, the fiducial cache. Check "
                "that extract_signature_dict includes `inject` when "
                "config.injects().")
        print(f"    {overlay!r}")

    return config_inj, {**vdfi, "spectra": overlay}
