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
                          n_r_interp=400, dtype=np.float32):
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
    v_grid = C_KMS * (wave - LYA_REST * (1.0 + z)) / (LYA_REST * (1.0 + z))

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
    K = _shell_to_fiber_kernel(r_obs / kpc_per_arcsec, s_kpc / kpc_per_arcsec,
                               seeing_fwhm, fiber_response_independent)

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
    lam0 = LYA_REST * (1.0 + z)
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

    lam0 = LYA_REST * (1.0 + z)
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
