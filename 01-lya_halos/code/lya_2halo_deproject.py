"""
Lya 2-halo-term surface density -> volume SFR density, via forward-model
deprojection instead of a velocity-derived line-of-sight depth.

CONTEXT / WHY THIS EXISTS
--------------------------
The original approach backed a line-of-sight depth out of the emission
line's velocity width via Delta_v / H(z). That conflates kinematic line
broadening (RT, outflows, virial/satellite motions) with real spatial
position, and it is unstable: H(z=2.3) ~ 237 km/s/Mpc, so ANY ordinary
peculiar velocity (a few hundred km/s) implies several comoving Mpc of
"depth" under that method, regardless of how the velocity window is
chosen. Worse, the depth was set by Delta_lambda -- widening the
extraction window from +/-3 to +/-6 A would have halved the answer.
No Lya-halo paper in the literature does this (checked against
Steidel+2011, Momose+2016, Wisotzki+2016/2018, Croft+2016/2018, the
HETDEX profile paper Lujan Niemeyer+2022) -- they either fit purely
projected profiles, or (in intensity-mapping work) get their depth from
the real cosmological Delta_z of the survey/instrument bandpass, never
from the target line's own width.

THE FIX implemented here: fit a parametric 3D volume emissivity j(r) to
your ACTUAL projected radial profile Sigma_Lya(R) via the standard
forward-projection integral used for exactly this kind of deprojection
(X-ray cluster SB deprojection, Kriss/Cioffi/Canizares 1983;
Vikhlinin 2006; stellar dynamics, Binney & Tremaine):

    Sigma(R) = integral_{-pi_max}^{+pi_max} j(sqrt(R^2 + pi^2)) dpi

No velocity or linewidth information enters this step. Only the
observed (R, Sigma) pairs do.

HONEST FRAMING OF pi_max -- READ THIS
--------------------------------------
pi_max IS a line-of-sight depth. Claiming this method "assumes no
depth" is wrong. What changes is how tightly the answer is COUPLED to
it, and that coupling is set by the fitted slope gamma, which is
measured from your data rather than assumed:

    gamma -> 0 : Sigma = j * 2*pi_max EXACTLY. pi_max IS the depth and
                 this method degenerates into the old one.
    gamma = 1.0: dln(j0)/dln(pi_max) ~ -0.26 to -0.39
    gamma = 1.8: dln(j0)/dln(pi_max) ~ -0.04 to -0.15
    gamma > 2.5: essentially decoupled (< 0.05)
    OLD METHOD : dln(rho)/dln(depth) = -1.00, exactly, always.

So the robustness of this whole approach is purchased entirely by the
profile being steep. A FLAT stretch of Sigma(R) is the worst case: flat
Sigma <=> gamma ~ 1 <=> nearly-uniform emissivity <=> pi_max is doing
the work again. `pi_max_log_sensitivity` below computes the actual
coupling for YOUR fitted gamma -- run it and quote the number rather
than asserting insensitivity.

Second honest point: pi_max controls only the fraction of Sigma the
model attributes to material along the sightline you never observed
(at gamma=1.8, ~10% of Sigma(R=2 cMpc) comes from |pi| > 10 cMpc; at
gamma=1.0 it's 28%). The old depth was a straight multiplicative
rescaling of the final answer. Different kind of assumption, and this
one is at least falsifiable.

PIVOT SEPARATION (added 2026-08-04)
------------------------------------
The previous version used a single symbol r0 for two physically
unrelated things: the pivot radius of the emissivity power law, and the
correlation length of xi(r). Those are now separate arguments,
`r_pivot_Mpc` and `r0_xi_Mpc`.

This was not cosmetic. With the pivot parked at 4 cMpc -- at or outside
the outer edge of the fitted data -- the projection kernel
K(R, gamma) = Sigma/j0 is NON-MONOTONIC in gamma, because along a
sightline at R < r_pivot the near-LOS material has r < r_pivot (where
raising gamma AMPLIFIES the emissivity) while the far-LOS material has
r > r_pivot (where raising gamma SUPPRESSES it). The two effects trade
off and K has a minimum at a gamma that depends on R. Since j0 = Sigma/K,
j0 rose and then fell as gamma increased -- a pure parameterization
artifact with no physical content, and the source of the
"low slopes give low values, then it rises, then falls" behaviour.

Fix: `r_pivot_Mpc=None` (the new default) places the pivot from the data
rather than by hand. There are actually TWO separate pathologies and
they want different pivots, so the default depends on whether gamma is
free:

  gamma FIXED -> 'geometric': pivot = exp(mean(ln R_fit)). This is only
      about making j0 an INTERPOLATED rather than extrapolated quantity
      and killing the non-monotonicity. There is no correlation to
      remove when gamma isn't a free parameter.

  gamma FREE  -> 'decorrelate': the pivot at which
      Cov(ln j0, gamma) = 0. Because the reparameterization
      ln j0' = ln j0 - gamma*ln(p'/p) is exact for a power law, this has
      a closed form and ONE Newton step from any starting pivot lands on
      it exactly:

          ln(p'/p) = Cov(ln j0, gamma) / Var(gamma)

      (verified numerically: converges to the same pivot to 1e-5 in corr
      from starting pivots of 2.5, 4, 6 and 10 cMpc.)

Worth knowing: the decorrelating pivot lands OUTSIDE the geometric mean
of the fitted radii -- for a test profile spanning 2.47-8.91 cMpc the
geometric mean is 4.70 but the decorrelating pivot is 8.04. That is not
a bug. Sigma(R) at impact parameter R integrates emissivity at 3D radii
r >= R, so the effective 3D radius the projected profile constrains is
systematically LARGER than R. Quoting j0 at the geometric mean of the
impact parameters would be quoting it where the data are weakest.

CONSTANT FLOOR (added 2026-08-04)
----------------------------------
`fit_floor=True` adds a constant pedestal in Sigma. This is the shape
degeneracy that matters most for you: with 2 free parameters and only a
handful of outer points you CANNOT distinguish a power law from a power
law plus a constant, and a constant in Sigma is exactly what residual
background over/under-subtraction looks like. Fit it and check it is
consistent with zero; if it isn't, the fitted gamma is not trustworthy.
Watch the dof -- 4 points and 3 free parameters is 1 dof.

WHAT THIS PHYSICALLY MEASURES (important -- read before quoting a
number): per the literature (Croft+2016/2018; Mas-Ribas & Dijkstra 2016;
Lujan Niemeyer+2022 HETDEX; Byrohl+2021 TNG50 RT sims), the 2-halo
regime of a stacked Lya profile is most often dominated by clustered,
individually-undetected SATELLITE/NEIGHBOR galaxies forming stars via
ordinary case-B emission -- not smooth diffuse cosmic-web glow. UVB
fluorescence (Cantalupo+2005; Gould & Weinberg 1996) and deep cosmic-web
MUSE detections (Nature Astronomy 2023) land at a comparable order of
magnitude, so it can't be ruled out as a contributor, but the
satellite-clustering explanation is the field's default interpretation
for this radial regime, not a smooth IGM-emission claim. Treat this
code as computing:

    rho_local(r)  = SFR density attributable to Lya emission AT that
                    radius in your stack's environment (the defensible,
                    radially-resolved, novel quantity -- report this).

    rho_mean      = rho_local(r) / xi(r), debiased using the REAL 3D
                    radius from the fit (never a velocity-derived
                    radius). Constant in r by construction. Nominally
                    comparable to a Madau & Dickinson number, but really
                    a test of "is the satellite-driven case-B budget
                    consistent with the cosmic SFRD" -- not an
                    independent re-measurement of it.

DEBIAS CONVENTION -- /xi, NOT /(1+xi)  (changed 2026-08-05)
-----------------------------------------------------------
The model is j_total(r) = j_mean * (1 + xi(r)). Background subtraction
removes the constant, so the stack measures the correlated EXCESS only,
j_measured = j_mean * xi, and the correct debias divides by xi. Dividing
by (1+xi) is right only if the cosmic mean is still in your profile,
which it is not. The old `rho_cosmic` key is retained for comparison but
under-reports by (1+xi)/xi = 5.2x at 2700 pkpc. Use `rho_mean`.

THE STRONGER FRAMING: j_mean is exactly the quantity Croft+2016/2018
measure by cross-correlating Lya surface brightness with SDSS/BOSS
quasars. Comparing to Croft is more direct than comparing to Madau &
Dickinson -- clustering-derived emissivity against clustering-derived
emissivity, no case-B factor in between. And because our j_mean assumes
f_esc=1 while Croft measures escaped photons, the RATIO is a direct
estimate of the Lya escape fraction. See croft_fesc_check(); on the
current numbers it returns f_esc <= 0.044, matching the Hayes+2011
global ~5%. That converts the biggest soft spot in the chain (the
f_esc=1 assumption) from a caveat into a measurement.

A2/r_c DEGENERACY: only the product A2 * r_c^gamma_2d enters, so the
r_c_fixed=400-vs-free question is irrelevant here provided the pair
comes from the same fit. See sfr_density_from_expcore's docstring.

WHERE TO EVALUATE rho -- the (1+xi) lever arm
----------------------------------------------
The clustering correction grows fast inward and carries the r0
units ambiguity with it (see UNITS WARNING). At z=2.3, 1" = 8.21 pkpc:

  r [pkpc]   r [cMpc]   1+xi (r0=4 cMpc)   1+xi (r0=4 h^-1 Mpc)   ratio
     100       0.33          90.2                170.5             1.89
     500       1.65           5.92                10.35            1.75
    1000       3.30           2.41                 3.69            1.53
    2000       6.60           1.41                 1.77            1.26
    2700       8.91           1.24                 1.45            1.17

So an UNRESOLVED UNITS CONVENTION alone moves rho_cosmic by 75% at
500 pkpc and 17% at 2700 pkpc. Evaluating inward buys signal-to-noise
and pays for it in systematics that S/N cannot touch. Quote rho_cosmic
in the outer part of the fitted range; use inner radii as a sensitivity
test (see `R_min_sensitivity`), not as the headline.

UNITS WARNING: literature r0 values for galaxy/LAE clustering are
usually quoted in h^-1 Mpc, not literal comoving Mpc. r0=4.0 read as
h^-1 Mpc is a typical LBG value; read as literal comoving Mpc it's on
the high side for LAEs specifically (LAE literature: r0 ~ 2.5-3.2
h^-1 Mpc typically). Confirm which convention you intend and convert
consistently (h ~ 0.7) before trusting r0_xi_Mpc below.

Suggested integration point: fitting.py, alongside the existing
expcore/twoexp profile-fitting functions -- not a new module.
"""

import numpy as np
from scipy import optimize
from scipy.special import gamma as _gammafn

# numpy renamed trapz -> trapezoid in 2.0 (trapz still present but deprecated).
# The TACC/conda stacks are commonly on 1.26, so bind whichever exists.
_trapz = getattr(np, "trapezoid", None) or np.trapz

C_KMS = 299792.458
MADAU_DICKINSON = lambda z: 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)

# ---------------------------------------------------------------
# Clustering correlation length. RESOLVED 2026-08-05.
#
# Literature r0 is quoted in h^-1 Mpc, essentially always. Reading it
# as a literal cMpc is a factor (1/h)^-gamma_xi = 0.526 error on the
# final answer at h=0.7, gamma_xi=1.8 -- enough to flip the conclusion
# from "above" to "below" the Madau & Dickinson SFRD.
#
# USE THE CROSS-CORRELATION VALUE. This measurement is galaxy positions
# x Lya EMISSION, i.e. a cross-correlation. Plugging in an
# autocorrelation r0 silently assumes b_Lya = b_galaxy.
#
# Bielby et al. 2016, MNRAS 456, 4061 (arXiv:1501.01215), abstract:
#   LAE autocorrelation        r0 = 2.86 +/- 0.33  h^-1 Mpc
#   LBG-LAE CROSS-correlation  r0 = 3.29 +/- 0.57  h^-1 Mpc   <-- adopt
#   z~3 LBG auto (Paper III, Bielby+2013)   r0 ~ 4 h^-1 Mpc
# Those fits assume gamma = 1.8, so using r0 together with
# gamma_xi=1.8 is internally consistent rather than an extra assumption.
#
# CAVEAT: Bielby is z=3.1, this sample is z~2.3. Comoving clustering
# lengths evolve slowly for a fixed population but this is still an
# extrapolation -- keep it in the systematics table.
# ---------------------------------------------------------------
H_LITTLE = 0.7
R0_XI_LBG_LAE_CROSS_HINV = 3.29      # Bielby+2016 -- ADOPTED
R0_XI_LBG_LAE_CROSS_ERR_HINV = 0.57
R0_XI_LAE_AUTO_HINV = 2.86
R0_XI_LAE_AUTO_ERR_HINV = 0.33
R0_XI_LBG_AUTO_HINV = 4.0

R0_XI_DEFAULT = R0_XI_LBG_LAE_CROSS_HINV / H_LITTLE   # 4.700 cMpc

# Croft et al. 2018, MNRAS 481, 1320 (arXiv:1806.06050): 95% upper limit
# on the GLOBAL mean Lya surface brightness from the Lya emission x Lya
# forest cross-correlation. See croft_fesc_check() -- this is the
# ESCAPING emissivity, so its ratio to our f_esc=1 intrinsic number is
# a direct measurement of f_esc, not a consistency check.
CROFT2018_MU_LIMIT = 1.2e-22          # erg/s/cm^2/A/arcsec^2, 95% UL
LYA_REST_A = 1215.67
MPC_CM = 3.0856775814913673e24
SR_PER_ARCSEC2 = (np.pi / 180.0 / 3600.0) ** 2        # 2.35045e-11


# ---------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------
def comoving_distance_Mpc(z, H0=70.0, Om=0.3, OL=0.7, zmin=0.0, npts=2000):
    """
    Real comoving distance via direct integration, for anything that needs
    a genuine Delta_z -> Delta_d conversion (e.g. characterizing your
    sample's redshift-selection depth). NEVER use this pattern to convert
    an emission LINE's velocity width into a spatial depth -- that was the
    original bug.
    """
    zz = np.linspace(zmin, z, npts)
    Hz = H0 * np.sqrt(Om * (1 + zz) ** 3 + OL)
    return _trapz(C_KMS / Hz, zz)


def kpc_per_arcsec(z, H0=70.0, Om=0.3, OL=0.7):
    """Physical kpc per arcsec at z. (z=2.3 -> 8.205 pkpc/", so the
    80-90" background annulus sits at 656-738 pkpc.)"""
    D_A = comoving_distance_Mpc(z, H0, Om, OL) / (1 + z)
    return D_A * 1e3 * (np.pi / 180.0 / 3600.0)


# ---------------------------------------------------------------
# Step 1: forward-model projection
# ---------------------------------------------------------------
def project_emissivity(R_Mpc, j0, r_pivot_Mpc, gamma, pi_max_Mpc=25.0,
                       floor=0.0, n_pi=4001):
    """
    Project a power-law volume emissivity j(r) = j0 * (r/r_pivot)^-gamma
    through the line of sight:

        Sigma(R) = integral_{-pi_max}^{+pi_max} j(sqrt(R^2+pi^2)) dpi + floor

    Units: j0 in erg/s/cMpc^3, R and pi_max in comoving Mpc, so Sigma
    comes out in erg/s/cMpc^2. `floor` is a CONSTANT PEDESTAL IN SIGMA
    (erg/s/cMpc^2), representing residual background mis-subtraction --
    it is not part of j and is not converted to an SFR density.

    Uses a fixed sinh-spaced pi grid rather than scipy.quad: the
    integrand is smooth and peaked at pi=0, the grid resolves that, and
    it is ~100x faster and fully deterministic, which matters because
    the sensitivity scans refit dozens of times.

    pi_max_Mpc is a genuine physical assumption -- it IS a depth (see
    module docstring). Use `pi_max_log_sensitivity` to quantify how much
    your particular fit depends on it.
    """
    R_Mpc = np.atleast_1d(R_Mpc).astype(float)
    # sinh spacing: dense near pi=0 where the integrand peaks, sparse in
    # the tail. Symmetric, so integrate 0->pi_max and double.
    u = np.linspace(0.0, np.arcsinh(pi_max_Mpc / 0.05), n_pi)
    pi_grid = 0.05 * np.sinh(u)
    r = np.sqrt(R_Mpc[:, None] ** 2 + pi_grid[None, :] ** 2)
    r = np.maximum(r, 1e-6)
    integrand = j0 * (r / r_pivot_Mpc) ** (-gamma)
    return 2.0 * _trapz(integrand, pi_grid, axis=1) + floor


def projection_kernel(R_Mpc, r_pivot_Mpc, gamma, pi_max_Mpc=25.0):
    """K(R) = Sigma(R)/j0, in comoving Mpc. Diagnostic only."""
    return project_emissivity(R_Mpc, 1.0, r_pivot_Mpc, gamma, pi_max_Mpc)


def effective_los_depth(R_Mpc, r_pivot_Mpc, gamma, pi_max_Mpc=25.0):
    """
    L_eff(R) = Sigma(R) / j(R), comoving Mpc -- "how thick a slab of the
    LOCAL emissivity reproduces the observed column". This is the honest
    analogue of the old code's single `LOS_depth`, and the point is that
    it is a FUNCTION OF R (it grows outward, because at large impact
    parameter the profile is locally flat along the sightline). The old
    code's 20.6 cMpc applied at every radius is wrong in shape, not just
    in normalization.
    """
    R_Mpc = np.atleast_1d(R_Mpc).astype(float)
    K = projection_kernel(R_Mpc, r_pivot_Mpc, gamma, pi_max_Mpc)
    return K / (R_Mpc / r_pivot_Mpc) ** (-gamma)


# ---------------------------------------------------------------
# Step 2: fit to the REAL spatial profile
# ---------------------------------------------------------------
def fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                           r_pivot_Mpc=None, gamma_fixed=None,
                           fit_floor=False, R_min_kpc=750.0,
                           R_max_kpc=None, pi_max_Mpc=25.0,
                           radii_are_comoving=False,
                           sigma_per_comoving_area=False):
    """
    Fit j0 (and optionally gamma, and optionally a constant Sigma floor)
    to your OBSERVED projected profile, by forward-projecting a power-law
    j(r) and comparing to data. Replaces the velocity-derived LOS depth
    entirely -- only (R, Sigma, sigma) data is used.

    R_kpc, Sigma_obs, Sigma_err : your measured radial profile.
        Sigma in erg/s/kpc^2 (your pipeline convention).
        R in PHYSICAL kpc unless radii_are_comoving=True.
    z : redshift of the stack.
    r_pivot_Mpc : pivot radius of the emissivity power law, COMOVING Mpc.
        None (default, RECOMMENDED) -> 'decorrelate' if gamma is free,
        'geometric' if gamma is fixed. May also be given explicitly as
        'decorrelate', 'geometric', or a float. See module docstring:
        parking the pivot outside the data makes j0 non-monotonic in
        gamma and is a pure artifact. Only set a float by hand if you
        have a reason.
        NOTE this is NOT the clustering correlation length -- that is
        r0_xi_Mpc in sfr_density_profile, and they are now separate.
    gamma_fixed : hold the emissivity slope fixed (e.g. at your assumed
        clustering gamma=1.8) and fit only the amplitude. RECOMMENDED as
        the headline when you have few points: 1 free parameter.
        If None, gamma floats -- use that as a DIAGNOSTIC: if best-fit
        gamma lands near your assumed clustering gamma, that supports the
        satellite-clustering interpretation; if it's very different,
        that's evidence of a component with a different radial
        dependence (e.g. smooth IGM/fluorescence glow).
    fit_floor : add a constant pedestal in Sigma (residual background).
        Check it against zero. Costs a dof -- see module docstring.
    R_min_kpc / R_max_kpc : fitted range. Default R_min raised from 300
        to 750 pkpc: 1+xi is 2.4 there vs 5.9 at 500 pkpc, so the
        clustering debias stops dominating the systematic budget.
    pi_max_Mpc : projection-integral cutoff -- see module docstring.
    """
    R_kpc = np.asarray(R_kpc, dtype=float)
    Sigma_obs = np.asarray(Sigma_obs, dtype=float)
    Sigma_err = np.asarray(Sigma_err, dtype=float)

    # UNIT LADDER. Sigma_obs arrives as erg/s per PHYSICAL kpc^2 (your
    # pipeline's convention), but the projection integral runs over a
    # COMOVING line-of-sight coordinate, so Sigma must be expressed per
    # COMOVING Mpc^2 or the result is a physical-area / comoving-length
    # hybrid.
    #   x1e6        : per pkpc^2  -> per pMpc^2
    #   /(1+z)^2    : per pMpc^2  -> per cMpc^2   (1 pMpc^2 = (1+z)^2 cMpc^2)
    # The second factor was MISSING before 2026-08-04 and made j0 too
    # large by exactly (1+z)^2 = 10.89 at z=2.3. It was caught by
    # verify_deprojection() CHECK 4, which compares this numerical path
    # against the independent closed-form Abel route. Both factors appear
    # in the original hand calculation ("Convert kpc^-2 --> physical
    # Mpc^-2" then "Convert physical area --> comoving area").
    area_conv = 1e6 if sigma_per_comoving_area else 1e6 / (1 + z) ** 2
    S = Sigma_obs * area_conv
    E = Sigma_err * area_conv

    mask = R_kpc >= R_min_kpc
    if R_max_kpc is not None:
        mask &= R_kpc <= R_max_kpc
    if mask.sum() < 2:
        raise ValueError(f"only {mask.sum()} points in the fit range")

    Rc = R_kpc[mask] / 1000.0
    if not radii_are_comoving:
        Rc = Rc * (1 + z)          # physical -> comoving Mpc
    Sf, Ef = S[mask], E[mask]

    free_gamma = gamma_fixed is None

    # ---- pivot placement -------------------------------------------
    # 'geometric' : exp(mean(ln R)) -- inside the data, kills the
    #               non-monotonic-j0(gamma) artifact. Right choice when
    #               gamma is fixed (no correlation to remove).
    # 'decorrelate': the pivot where Cov(ln j0, gamma) = 0. Only
    #               meaningful when gamma is free.
    if r_pivot_Mpc is None:
        r_pivot_mode = 'decorrelate' if free_gamma else 'geometric'
    elif isinstance(r_pivot_Mpc, str):
        r_pivot_mode = r_pivot_Mpc
    else:
        r_pivot_mode = 'fixed'
    r_geom = float(np.exp(np.mean(np.log(Rc))))
    if r_pivot_mode == 'fixed':
        pivot = float(r_pivot_Mpc)
    else:
        pivot = r_geom
    if r_pivot_mode == 'decorrelate' and not free_gamma:
        raise ValueError("r_pivot_Mpc='decorrelate' requires gamma to be free "
                         "(gamma_fixed=None); with gamma fixed there is no "
                         "covariance to null. Use 'geometric'.")

    # The floor is fit in units of `floor_scale` so its parameter is O(1).
    # Sigma here is ~1e36 erg/s/cMpc^2, so a raw floor parameter starting at
    # 0.0 makes the Jacobian catastrophically ill-conditioned against
    # log10(j0) and curve_fit returns a singular covariance.
    floor_scale = float(np.median(np.abs(Sf))) or 1.0

    def _run(pivot):
        K_piv = float(projection_kernel(
            pivot, pivot, 1.8 if free_gamma else gamma_fixed, pi_max_Mpc)[0])
        j0_guess = max(np.median(np.abs(Sf)) / K_piv, 1e-30)
        p0 = [np.log10(j0_guess)]
        if free_gamma:
            p0.append(1.8)
        if fit_floor:
            p0.append(0.0)

        def model(R, *p):
            i = 0
            j0 = 10 ** p[i]; i += 1
            g = p[i] if free_gamma else gamma_fixed
            if free_gamma:
                i += 1
            fl = p[i] * floor_scale if fit_floor else 0.0
            return project_emissivity(R, j0, pivot, g, pi_max_Mpc, floor=fl)

        popt, pcov = optimize.curve_fit(model, Rc, Sf, p0=p0, sigma=Ef,
                                        absolute_sigma=True, maxfev=20000)
        return model, popt, pcov

    model, popt, pcov = _run(pivot)

    if r_pivot_mode == 'decorrelate':
        # Exact single Newton step: ln(p'/p) = Cov(ln j0, gamma)/Var(gamma).
        # pcov is in log10(j0), so convert with ln(10).
        var_g = pcov[1, 1]
        if var_g > 0 and np.isfinite(var_g):
            shift = np.log(10) * pcov[0, 1] / var_g
            shift = float(np.clip(shift, -3.0, 3.0))   # guard pathological fits
            pivot = pivot * np.exp(shift)
            model, popt, pcov = _run(pivot)
    r_pivot_Mpc = pivot
    perr = np.sqrt(np.diag(pcov))

    i = 0
    j0 = 10 ** popt[i]
    j0_err = j0 * np.log(10) * perr[i]; i += 1
    if free_gamma:
        gamma, gamma_err = popt[i], perr[i]; i += 1
    else:
        gamma, gamma_err = gamma_fixed, 0.0
    if fit_floor:
        floor, floor_err = popt[i] * floor_scale, perr[i] * floor_scale
    else:
        floor, floor_err = 0.0, 0.0

    resid = (Sf - model(Rc, *popt)) / Ef
    n_free = len(popt)
    dof = len(Rc) - n_free
    chi2 = float(np.sum(resid ** 2))

    # j0/gamma correlation: the diagnostic the pivot change is for.
    if free_gamma:
        corr = pcov[0, 1] / (perr[0] * perr[1])
    else:
        corr = np.nan

    return dict(j0=j0, j0_err=j0_err, gamma=gamma, gamma_err=gamma_err,
                floor=floor, floor_err=floor_err,
                floor_sigma=(floor / floor_err if fit_floor and floor_err > 0
                             else np.nan),
                r_pivot_Mpc=r_pivot_Mpc, r_pivot_mode=r_pivot_mode,
                r_geom_Mpc=r_geom, pi_max_Mpc=pi_max_Mpc,
                R_fit_kpc=R_kpc[mask], R_fit_comoving_Mpc=Rc,
                n_points=int(mask.sum()), n_free=n_free, dof=dof,
                chi2=chi2, chi2_red=(chi2 / dof if dof > 0 else np.nan),
                corr_j0_gamma=corr, z=z, popt=popt, pcov=pcov)


def evaluate_j(fit, r_Mpc):
    """j(r) [erg/s/cMpc^3] from a fit dict, at comoving radii r_Mpc."""
    r_Mpc = np.atleast_1d(r_Mpc).astype(float)
    return fit['j0'] * (r_Mpc / fit['r_pivot_Mpc']) ** (-fit['gamma'])


# ---------------------------------------------------------------
# Step 3: convert fitted j(r) -> SFR density, at a REAL 3D radius
# ---------------------------------------------------------------
def sfr_density_profile(r_Mpc, fit, case_B_sfr=9.1e-43, r0_xi_Mpc=None,
                        gamma_xi=1.8, apply_clustering_correction=True,
                        warn_extrapolation=True):
    """
    Evaluate the fitted j(r) at real 3D COMOVING radii r_Mpc and convert
    to SFR volume density via case-B + Kennicutt (zero escape-fraction
    loss assumed -> LOWER LIMIT; real dust/RT losses push the true SFR
    higher).

    r0_xi_Mpc, gamma_xi are the CLUSTERING correlation-function
    parameters, now fully separate from the emissivity pivot
    (fit['r_pivot_Mpc']). Check the h^-1 convention -- see module
    docstring; it is a 17-75% effect depending on radius.

    Returns BOTH quantities -- they answer different questions:
      rho_local(r)  : SFR density in your stack's own environment.
      rho_mean      : the /xi debias -- THE headline (see
                      sfr_density_from_expcore's RETURNS section for why
                      /xi and not /(1+xi) is right for background-
                      subtracted data).
      rho_cosmic(r) : the /(1+xi) form, kept for comparison only.
    """
    if r0_xi_Mpc is None:
        r0_xi_Mpc = R0_XI_DEFAULT
    r_Mpc = np.atleast_1d(r_Mpc).astype(float)

    if warn_extrapolation:
        lo, hi = fit['R_fit_comoving_Mpc'].min(), fit['R_fit_comoving_Mpc'].max()
        out = (r_Mpc < lo) | (r_Mpc > hi)
        if out.any():
            import warnings
            warnings.warn(
                f"evaluating j at r={r_Mpc[out]} cMpc, outside the fitted "
                f"range [{lo:.2f}, {hi:.2f}] cMpc. Power-law extrapolation "
                f"beyond the data is not constrained -- and note the fitted "
                f"range is in IMPACT PARAMETER while r here is 3D radius.",
                stacklevel=2)

    j_r = evaluate_j(fit, r_Mpc)                      # erg/s/cMpc^3
    rho_local = case_B_sfr * j_r                      # Msun/yr/cMpc^3

    xi = rho_cosmic = rho_mean = j_mean = None
    if apply_clustering_correction:
        xi = (r_Mpc / r0_xi_Mpc) ** (-gamma_xi)
        j_mean = j_r / xi                        # correct debias
        rho_mean = case_B_sfr * j_mean           # THE headline
        rho_cosmic = rho_local / (1 + xi)        # comparison only

    return dict(r_Mpc=r_Mpc, j_r=j_r, rho_local=rho_local,
                j_mean=j_mean, rho_mean=rho_mean,
                rho_cosmic=rho_cosmic, xi=xi,
                r0_xi_Mpc=r0_xi_Mpc, gamma_xi=gamma_xi)


# ---------------------------------------------------------------
# Sensitivity tests -- run all of these before quoting a number
# ---------------------------------------------------------------
def pi_max_sensitivity(R_kpc, Sigma_obs, Sigma_err, z,
                       pi_max_grid=(10, 15, 20, 25, 30, 40), **kw):
    """Refit across a grid of pi_max. Quote the spread as the systematic
    from the projection-integral cutoff."""
    rows = []
    for pmax in pi_max_grid:
        f = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                                   pi_max_Mpc=pmax, **kw)
        rows.append(dict(pi_max=pmax, j0=f['j0'], j0_err=f['j0_err'],
                         gamma=f['gamma'], gamma_err=f['gamma_err'],
                         r_pivot=f['r_pivot_Mpc'], chi2_red=f['chi2_red']))
    return rows


def pi_max_log_sensitivity(R_kpc, Sigma_obs, Sigma_err, z,
                           pi_max_Mpc=25.0, step=1.25, **kw):
    """
    dln(j0)/dln(pi_max) at your fitted gamma -- THE number to quote when
    someone asks "isn't pi_max just the depth again?".

    Reference values: the OLD velocity-depth method had exactly -1.00.
    A result here of, say, -0.08 means a factor-2 error in pi_max moves
    your answer by 6%, versus 100% under the old method. A result
    approaching -0.5 means the profile is too flat for the deprojection
    to be buying you anything and you should say so.
    """
    lo = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                                pi_max_Mpc=pi_max_Mpc / step, **kw)
    hi = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                                pi_max_Mpc=pi_max_Mpc * step, **kw)
    dlnj = np.log(hi['j0'] / lo['j0']) / np.log(step ** 2)
    dlng = ((hi['gamma'] - lo['gamma']) / np.log(step ** 2)
            if kw.get('gamma_fixed') is None else 0.0)
    return dict(pi_max=pi_max_Mpc, dlnj0_dlnpimax=float(dlnj),
                dgamma_dlnpimax=float(dlng),
                old_method_reference=-1.0)


def R_min_sensitivity(R_kpc, Sigma_obs, Sigma_err, z,
                      R_min_grid=(500, 750, 1000, 1250, 1500), **kw):
    """
    Refit while pushing the inner edge of the fit outward. Stability here
    is the check that a single power law describes the fitted range; a
    systematic drift in gamma means the slope is changing across the
    range, which is itself a result (it locates where the satellite
    population takes over) but invalidates the single-power-law headline.

    Also reports 1+xi at the inner edge so you can see the clustering
    lever arm you are taking on by reaching inward.
    """
    rows = []
    r0_xi = kw.pop('r0_xi_Mpc', 4.0)
    gamma_xi = kw.pop('gamma_xi', 1.8)
    for rmin in R_min_grid:
        try:
            f = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                                       R_min_kpc=rmin, **kw)
        except (ValueError, RuntimeError) as e:
            rows.append(dict(R_min_kpc=rmin, error=str(e)))
            continue
        r_in = (rmin / 1000.0) * (1 + z)
        rows.append(dict(R_min_kpc=rmin, n_points=f['n_points'],
                         dof=f['dof'], j0=f['j0'], j0_err=f['j0_err'],
                         gamma=f['gamma'], gamma_err=f['gamma_err'],
                         r_pivot=f['r_pivot_Mpc'], chi2_red=f['chi2_red'],
                         one_plus_xi_at_Rmin=1 + (r_in / r0_xi) ** (-gamma_xi)))
    return rows


R0_XI_VARIANTS = (
    ("Bielby+16 LBG-LAE cross (ADOPTED)", R0_XI_LBG_LAE_CROSS_HINV / H_LITTLE),
    ("  cross -1sig",
     (R0_XI_LBG_LAE_CROSS_HINV - R0_XI_LBG_LAE_CROSS_ERR_HINV) / H_LITTLE),
    ("  cross +1sig",
     (R0_XI_LBG_LAE_CROSS_HINV + R0_XI_LBG_LAE_CROSS_ERR_HINV) / H_LITTLE),
    ("Bielby+16 LAE auto", R0_XI_LAE_AUTO_HINV / H_LITTLE),
    ("Bielby+13 LBG auto ~4/h", R0_XI_LBG_AUTO_HINV / H_LITTLE),
    ("WRONG: 4.0 read as literal cMpc", 4.0),
)


def r0_xi_sensitivity(fit, r_Mpc, r0_variants=None,
                      gamma_xi_variants=(1.6, 1.8, 2.0), **kw):
    """
    rho_mean across the r0 / h-convention / gamma_xi choices. This is
    usually the single largest systematic, and no amount of
    signal-to-noise touches it. Default variants are the labelled
    Bielby+2016 set (R0_XI_VARIANTS), including the "read h^-1 Mpc as
    literal cMpc" mistake so its size stays visible.
    """
    rows = []
    if r0_variants is None:
        variants = list(R0_XI_VARIANTS)
    else:
        variants = [(f"{r0:.3f}", r0) for r0 in r0_variants]
    for lab, r0 in variants:
        for gx in gamma_xi_variants:
            out = sfr_density_profile(r_Mpc, fit, r0_xi_Mpc=r0, gamma_xi=gx,
                                      warn_extrapolation=False, **kw)
            rows.append(dict(label=lab, r0_xi=r0, gamma_xi=gx,
                             rho_local=float(np.atleast_1d(out['rho_local'])[0]),
                             rho_mean=float(np.atleast_1d(out['rho_mean'])[0]),
                             rho_cosmic=float(np.atleast_1d(out['rho_cosmic'])[0]),
                             one_plus_xi=float(1 + np.atleast_1d(out['xi'])[0])))
    return rows


def sensitivity_report(R_kpc, Sigma_obs, Sigma_err, z, r_eval_Mpc,
                       gamma_fixed=1.8, R_min_kpc=750.0, pi_max_Mpc=25.0,
                       fit_floor=True):
    """
    Run the full battery and print it. This is the thing to run before
    quoting rho_cosmic anywhere.
    """
    base = dict(gamma_fixed=gamma_fixed, R_min_kpc=R_min_kpc)

    print("=" * 68)
    print("FIDUCIAL FIT")
    f = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                               pi_max_Mpc=pi_max_Mpc, fit_floor=False, **base)
    print(f"  fit range    = {f['R_fit_comoving_Mpc'].min():.2f} - "
          f"{f['R_fit_comoving_Mpc'].max():.2f} cMpc  ({f['n_points']} pts)")
    print(f"  pivot        = {f['r_pivot_Mpc']:.3f} cMpc "
          f"(mode='{f['r_pivot_mode']}', geom mean {f['r_geom_Mpc']:.3f})")
    print(f"  j0           = {f['j0']:.4e} +/- {f['j0_err']:.2e} erg/s/cMpc^3")
    print(f"  gamma        = {f['gamma']:.3f} +/- {f['gamma_err']:.3f} (FIXED)"
          if gamma_fixed is not None else
          f"  gamma        = {f['gamma']:.3f} +/- {f['gamma_err']:.3f}")
    print(f"  dof          = {f['dof']}   chi2_red = {f['chi2_red']:.2f}")
    Le = effective_los_depth(f['R_fit_comoving_Mpc'], f['r_pivot_Mpc'],
                             f['gamma'], pi_max_Mpc)
    print(f"  L_eff range  = {Le.min():.2f} - {Le.max():.2f} cMpc  "
          f"(old velocity method assumed 20.6 at ALL radii)")

    print("\nFREE-GAMMA DIAGNOSTIC (is the outer slope consistent with "
          "clustered satellites?)")
    try:
        fg = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                                    gamma_fixed=None, R_min_kpc=R_min_kpc,
                                    pi_max_Mpc=pi_max_Mpc)
        print(f"  gamma_free   = {fg['gamma']:.3f} +/- {fg['gamma_err']:.3f}   "
              f"(assumed clustering gamma = 1.8)")
        print(f"  decorr pivot = {fg['r_pivot_Mpc']:.3f} cMpc,  "
              f"corr(j0,gamma) = {fg['corr_j0_gamma']:+.4f} "
              f"(should be ~0 by construction)")
        print(f"  tension      = {abs(fg['gamma'] - 1.8) / max(fg['gamma_err'], 1e-9):.1f} sigma")
    except (ValueError, RuntimeError) as e:
        print(f"  free-gamma fit failed: {e}")

    print("\nIS pi_max JUST THE OLD DEPTH? (dln j0 / dln pi_max)")
    s = pi_max_log_sensitivity(R_kpc, Sigma_obs, Sigma_err, z,
                               pi_max_Mpc=pi_max_Mpc, **base)
    print(f"  {s['dlnj0_dlnpimax']:+.3f}   vs old method {s['old_method_reference']:+.2f}")
    if abs(s['dlnj0_dlnpimax']) > 0.3:
        print("  ** WARNING: profile too flat -- deprojection is buying little. **")

    print("\npi_max SCAN")
    for r in pi_max_sensitivity(R_kpc, Sigma_obs, Sigma_err, z, **base):
        print(f"  pi_max={r['pi_max']:>3} -> j0={r['j0']:.3e}+/-{r['j0_err']:.1e} "
              f"gamma={r['gamma']:.3f}")

    print("\nR_min SCAN (single-power-law check + clustering lever arm)")
    for r in R_min_sensitivity(R_kpc, Sigma_obs, Sigma_err, z,
                               gamma_fixed=gamma_fixed, pi_max_Mpc=pi_max_Mpc):
        if 'error' in r:
            print(f"  R_min={r['R_min_kpc']:>5} -> {r['error']}")
            continue
        print(f"  R_min={r['R_min_kpc']:>5} pkpc  n={r['n_points']:>3} "
              f"j0={r['j0']:.3e} gamma={r['gamma']:.3f} "
              f"chi2r={r['chi2_red']:.2f}  1+xi(Rmin)={r['one_plus_xi_at_Rmin']:.2f}")

    if fit_floor:
        print("\nCONSTANT-FLOOR TEST (residual background degeneracy)")
        try:
            ff = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                                        pi_max_Mpc=pi_max_Mpc, fit_floor=True,
                                        **base)
            print(f"  floor = {ff['floor']:.3e} +/- {ff['floor_err']:.2e} "
                  f"erg/s/cMpc^2  ({ff['floor_sigma']:+.2f} sigma from 0)")
            print(f"  j0 shifts {f['j0']:.3e} -> {ff['j0']:.3e} "
                  f"({100*(ff['j0']/f['j0']-1):+.1f}%),  dof={ff['dof']}")
            if abs(ff['floor_sigma']) > 2:
                print("  ** floor inconsistent with zero -- gamma not trustworthy **")
        except (ValueError, RuntimeError) as e:
            print(f"  floor fit failed: {e}")

    print(f"\nSFR DENSITY at r = {r_eval_Mpc} cMpc  (r0={R0_XI_DEFAULT:.3f} cMpc)")
    out = sfr_density_profile(np.atleast_1d(r_eval_Mpc), f)
    for r, rl, rm, rc in zip(out['r_Mpc'], out['rho_local'],
                             out['rho_mean'], out['rho_cosmic']):
        print(f"  r={r:.2f} cMpc ({r/(1+z)*1000:.0f} pkpc): "
              f"rho_local={rl:.3e}  rho_mean={rm:.3e}  [rho_cosmic={rc:.3e}]")
    print(f"  Madau&Dickinson(z={z}) = {MADAU_DICKINSON(z):.4f} Msun/yr/cMpc^3")
    print("  rho_mean (/xi) is the headline; rho_cosmic (/(1+xi)) is shown "
          "for comparison only.")

    print("\nCROFT+2018 CROSS-CHECK (tension, NOT an f_esc measurement)")
    croft_tension_check(np.atleast_1d(out['j_mean'])[0], z, label="  2-halo")
    print("\n  SFRD vs assumed escape fraction (f_esc=1 is the LOWER LIMIT):")
    sfrd_vs_fesc(np.atleast_1d(out['j_mean'])[0], z)

    print("\nCLUSTERING-CONVENTION SYSTEMATIC on rho_mean")
    rows = r0_xi_sensitivity(f, np.atleast_1d(r_eval_Mpc)[0])
    rm = [r['rho_mean'] for r in rows]
    for r in rows:
        if r['gamma_xi'] != 1.8:
            continue
        print(f"  {r['label']:<34} r0={r['r0_xi']:5.2f} -> "
              f"1+xi={r['one_plus_xi']:6.2f}  rho_mean={r['rho_mean']:.3e}")
    print(f"  SPREAD (incl. gamma_xi 1.6-2.0): {min(rm):.3e} to {max(rm):.3e}  "
          f"(factor {max(rm)/min(rm):.2f})")
    print("=" * 68)
    return f


# ===============================================================
# ANALYTIC ROUTE: deproject the expcore 2-halo term directly.
#
# This is the RECOMMENDED path. It uses A2/r_c/gamma straight from the
# expcore fit, which has already separated the exponential core from the
# 2-halo term. Fitting a bare power law to the TOTAL profile instead
# drags the amplitude up by ~10x because the core (A1 ~ 100*A2) leaks
# into the outer fit.
# ===============================================================
def abel_constant(gamma_3d):
    """
    C(gamma) = sqrt(pi) * Gamma((gamma-1)/2) / Gamma(gamma/2).

    The full line-of-sight integral of j(r) = j0 (r/rp)^-gamma is

        Sigma(R) = C(gamma) * j(R) * R

    so the effective LOS depth is simply C(gamma)*R -- proportional to
    impact parameter, with NO pi_max anywhere. The integral converges on
    its own for gamma > 1. For gamma=1.8, C = 3.679.

    This is why the projected slope is one shallower than the 3D slope:
    Sigma ~ j(R)*R ~ R^(1-gamma).
    """
    if gamma_3d <= 1.0:
        raise ValueError(f"gamma_3d={gamma_3d} <= 1: the LOS integral "
                         "diverges and no truncation-free answer exists. "
                         "Use the numerical fitter with an explicit pi_max.")
    return np.sqrt(np.pi) * _gammafn((gamma_3d - 1) / 2.0) / _gammafn(gamma_3d / 2.0)


def truncation_factor(R_comoving_Mpc, gamma_3d, pi_max_Mpc=25.0, n_pi=200001):
    """
    Ratio (finite |pi|<pi_max integral) / (infinite integral). The analytic
    route is the pi_max -> infinity limit; this says how much you would
    lose by truncating. At gamma=1.8 it is 0.71-0.92 over 1.5-9 cMpc,
    i.e. a modest correction to a converged integral rather than a free
    parameter setting the answer.
    """
    R = np.atleast_1d(R_comoving_Mpc).astype(float)
    u = np.linspace(0, np.arcsinh(pi_max_Mpc / 0.02), n_pi)
    p = 0.02 * np.sinh(u)
    r = np.sqrt(R[:, None] ** 2 + p[None, :] ** 2)
    finite = 2 * _trapz(r ** (-gamma_3d), p, axis=1)
    infinite = abel_constant(gamma_3d) * R ** (1 - gamma_3d)
    return finite / infinite


def sfr_density_from_expcore(A2, r_c_kpc, gamma_2d, z,
                             R_eval_pkpc=(2000, 2400, 2700),
                             A2_err=None, case_B_sfr=9.1e-43,
                             r0_xi_Mpc=None, gamma_xi=1.8,
                             use_asymptotic=True, pi_max_Mpc=None):
    """
    Deproject the expcore 2-halo term analytically.

    A2, r_c_kpc, gamma_2d : straight from your expcore fit. The model is
        A2 * (1 + (R/r_c)^2)^(-gamma_2d/2), which -> A2*(R/r_c)^-gamma_2d
        at R >> r_c. A2 in erg/s/kpc^2, r_c in PHYSICAL kpc.
        A2 is the R->0 plateau of the cored form (at R=r_c the exact form
        gives 0.758*A2 for gamma_2d=0.8), and equivalently the coefficient
        of the asymptote when written A2*(R/r_c)^-gamma_2d.

        A2 AND r_c ARE DEGENERATE and only their product matters here:
        at R >> r_c the model collapses to (A2 * r_c^gamma_2d) * R^-gamma_2d,
        and the whole chain below is LINEAR in that product. Verified:
        holding A2*r_c^0.8 fixed and sliding r_c over 320-500 pkpc gives
        an identical answer to 6 significant figures. So the
        r_c_fixed=400 vs free-r_c question -- which genuinely matters for
        the crossover radius (20-26% across an r_c scan) -- is a NON-ISSUE
        here, provided A2 and r_c come from the SAME fit. The invariant is
        returned as out['A2_rc_invariant']; quote that, not A2 alone.

    gamma_2d : the PROJECTED slope (0.8 in your fits). The 3D slope is
        gamma_2d + 1 = 1.8 -- that relation IS the deprojection, and it
        is already baked into your choice of gamma_2d=0.8.
    R_eval_pkpc : default moved OUT to 2000-2700 pkpc (2026-08-05). Two
        independent reasons, both pushing the same way:
          - the cored form differs from the asymptote by 15% at 500 pkpc
            but only ~1% at >=2000, so r_c drops out of the answer;
          - 1+xi is 5.9 at 500 pkpc vs 1.4 at 2000, so the clustering
            systematic stops dominating.
        Inner radii are a sensitivity test, not the headline.
    use_asymptotic : DEFAULT CHANGED TO True (2026-08-05). Use the pure
        power law rather than the exact cored form. The analytic Abel
        deprojection assumes a power law, so mixing in the cored form is
        internally inconsistent; at R >= 2000 pkpc they agree to ~1%.
        Set False to see the size of that inconsistency.
    r0_xi_Mpc : CLUSTERING correlation length in literal COMOVING Mpc.
        None (default) -> R0_XI_DEFAULT = 3.29/0.7 = 4.700 cMpc, the
        Bielby+2016 LBG-LAE CROSS-correlation length converted out of
        h^-1 units. See the R0_XI_* constants at the top of the module
        for the alternatives and the citation.
    pi_max_Mpc : if set, multiply by truncation_factor to emulate a
        finite LOS. None (default) = the converged infinite-LOS answer.

    RETURNS -- read this, the headline quantity changed 2026-08-05.

    The physical model is  j_total(r) = j_mean * (1 + xi(r))  -- a cosmic
    mean emissivity, boosted by clustering near a galaxy. Your pipeline
    SUBTRACTS A BACKGROUND ANNULUS, which removes the constant, so what
    the stack measures is the correlated EXCESS only:

        j_measured(r) = j_mean * xi(r)   =>   j_mean = j_measured / xi

    so the correct debias is  / xi , NOT  / (1 + xi) . The returned
    `j_mean` / `rho_mean` use /xi and are the numbers to quote.
    `rho_cosmic` (/(1+xi)) is kept for comparison ONLY; it under-reports
    by (1+xi)/xi, which is 5.2x at 2700 pkpc. `rho_cosmic_plateau` is
    retained as a back-compatible alias for `rho_mean`.

    rho_mean is CONSTANT in R by construction (both j and xi go as
    r^-1.8), so it is not evaluated "at" any radius -- it is the pure
    amplitude ratio j0 * r_pivot^gamma * r0^-gamma_xi. That flatness is
    an ASSUMPTION (gamma_xi == gamma_3d), not a result; it constrains the
    SHAPE, not the normalisation. See verify_deprojection CHECK 6.
    """
    if r0_xi_Mpc is None:
        r0_xi_Mpc = R0_XI_DEFAULT
    gamma_3d = gamma_2d + 1.0
    C = abel_constant(gamma_3d)
    R_pkpc = np.atleast_1d(R_eval_pkpc).astype(float)
    R_cMpc = R_pkpc / 1000.0 * (1 + z)

    Sigma_asym = A2 * (R_pkpc / r_c_kpc) ** (-gamma_2d)
    Sigma_core = A2 * (1.0 + (R_pkpc / r_c_kpc) ** 2) ** (-gamma_2d / 2.0)
    Sigma = Sigma_asym if use_asymptotic else Sigma_core

    # --- explicit unit ladder (see verify_deprojection for the audit) ---
    j_phys = Sigma / (C * R_pkpc)              # erg/s/pkpc^3
    j_com = j_phys * 1e9 / (1 + z) ** 3        # -> /pMpc^3 -> /cMpc^3
    if pi_max_Mpc is not None:
        j_com = j_com * truncation_factor(R_cMpc, gamma_3d, pi_max_Mpc)

    rho_local = case_B_sfr * j_com
    xi = (R_cMpc / r0_xi_Mpc) ** (-gamma_xi)

    j_mean = j_com / xi                        # THE headline emissivity
    rho_mean = case_B_sfr * j_mean             # THE headline SFR density
    rho_cosmic = rho_local / (1 + xi)          # comparison only -- see above

    out = dict(R_pkpc=R_pkpc, R_cMpc=R_cMpc, Sigma=Sigma,
               C_gamma=C, gamma_3d=gamma_3d,
               A2_rc_invariant=A2 * r_c_kpc ** gamma_2d,
               cored_vs_asymptotic=Sigma_core / Sigma_asym - 1.0,
               L_eff_cMpc=C * R_pkpc * (1 + z) / 1000.0,
               j_comoving=j_com, rho_local=rho_local, xi=xi,
               j_mean=j_mean, rho_mean=rho_mean,
               rho_cosmic=rho_cosmic,
               rho_cosmic_plateau=rho_mean,    # back-compat alias
               r0_xi_Mpc=r0_xi_Mpc, gamma_xi=gamma_xi,
               case_B_sfr=case_B_sfr, z=z,
               madau_dickinson=MADAU_DICKINSON(z))
    if A2_err is not None:
        out['frac_err'] = A2_err / A2          # rho is LINEAR in A2
        out['j_mean_err'] = j_mean * A2_err / A2
        out['rho_mean_err'] = rho_mean * A2_err / A2
        out['rho_cosmic_err'] = rho_cosmic * A2_err / A2
        out['rho_plateau_err'] = rho_mean * A2_err / A2   # back-compat
    return out


# ---------------------------------------------------------------
# Croft comparison: turns the f_esc=1 caveat into a MEASUREMENT
# ---------------------------------------------------------------
def mean_surface_brightness(j_mean_comoving, z, H0=70.0, Om=0.3, OL=0.7,
                            lambda_rest_A=LYA_REST_A):
    """
    Convert a mean comoving line emissivity [erg/s/cMpc^3] into the mean
    observed surface brightness [erg/s/cm^2/A/arcsec^2], so it can be
    compared directly with intensity-mapping limits.

    Derivation (flat cosmology, no fudge factors):
      a comoving shell of thickness dchi subtending dOmega has comoving
      volume dOmega*chi^2*dchi, hence luminosity j*dOmega*chi^2*dchi;
      the flux received is L/(4*pi*D_L^2) = L/(4*pi*chi^2*(1+z)^2), so

          dF/dOmega = j * dchi / (4*pi*(1+z)^2)

      and for a LINE, dchi maps onto observed wavelength via
      lambda_obs = lambda_rest*(1+z), dchi = (c/H(z)) * dlambda_obs/lambda_rest:

          mu = j * c / (4*pi * (1+z)^2 * H(z) * lambda_rest)

      then cMpc^-2 -> cm^-2 and sr -> arcsec^-2.

    Note chi cancels: this is a per-unit-redshift-path quantity, so no
    distance measurement enters and it is not sensitive to the
    line-of-sight depth.
    """
    Hz = H0 * np.sqrt(Om * (1 + z) ** 3 + OL)          # km/s/Mpc
    mu = (np.asarray(j_mean_comoving, dtype=float) * C_KMS
          / (4 * np.pi * (1 + z) ** 2 * Hz * lambda_rest_A))   # /cMpc^2/A/sr
    return mu / MPC_CM ** 2 * SR_PER_ARCSEC2


def lya_production_budget(z, case_B_sfr=9.1e-43, sfrd=None):
    """
    The HARD CEILING on escaped Lya emissivity from star formation:
    take the known cosmic SFRD and ask how much Lya it makes if EVERY
    photon escapes.

        j_intrinsic_max = SFRD(z) / case_B_sfr        [erg/s/cMpc^3]

    Nothing sourced by star formation can exceed this, ever, because
    f_esc <= 1. No literature escape fraction is required -- which is the
    point: see the note on `required_escape_fraction`.
    """
    if sfrd is None:
        sfrd = MADAU_DICKINSON(z)
    return dict(z=z, sfrd=sfrd, case_B_sfr=case_B_sfr,
                j_intrinsic_max=sfrd / case_B_sfr)


def required_escape_fraction(j_mean_comoving, z, case_B_sfr=9.1e-43,
                             sfrd=None):
    """
    f_esc_required = j_mean / j_intrinsic_max.

    ===================================================================
    WHY THIS REPLACED THE LITERATURE-f_esc COMPARISON (2026-08-05)

    An earlier version compared j_mean against "the SFRD at the
    Hayes+2011 global f_esc ~ 5%". Austin correctly rejected that on
    three grounds, all of which stand:

      1. Hayes+2011 (LARS / the Lyman-alpha Reference Sample) measures
         LOW-REDSHIFT galaxies.
      2. It is an APERTURE, GALAXY-SCALE quantity -- the fraction of Lya
         escaping an individual galaxy, averaged over their sample. It
         is not "global" in any cosmological sense despite the wording
         often used.
      3. z~2 galaxies in this mass range run higher, ~20%, so 5% is not
         even the right galaxy-scale number for this sample.

    And most importantly: the escape fraction of Lya from a galaxy tells
    you NOTHING about the fraction of cosmic Lya production that shows
    up as diffuse emission a Mpc away. Those are different physical
    quantities. No published measurement of the latter exists -- it is
    not directly observable, only model-dependent.

    THE FIX: don't import an escape fraction at all. INVERT the question.
    Report the f_esc this measurement would REQUIRE, and let f_esc <= 1
    do the work. That bound is exact and assumption-free.
    ===================================================================

    Interpretation:
      f_esc_required > 1  -> in-situ case-B star formation CANNOT source
                            this emission. Falsified, no literature
                            value needed.
      f_esc_required ~ realistic values -> consistent with an SF origin,
                            but does not prove one.
    """
    b = lya_production_budget(z, case_B_sfr, sfrd)
    j = np.atleast_1d(j_mean_comoving).astype(float)
    return dict(f_esc_required=j / b['j_intrinsic_max'],
                j_mean=j, **b)


def croft_tension_check(j_mean_comoving, z, mu_limit=CROFT2018_MU_LIMIT,
                        case_B_sfr=9.1e-43,
                        label="2-halo", verbose=True):
    """
    Compare the measured mean Lya emissivity against Croft+2018's 95%
    upper limit on the global mean Lya surface brightness.

    ===================================================================
    CORRECTED 2026-08-05. The previous version (croft_fesc_check) had
    the escape-fraction logic BACKWARDS and reported a spurious
    "f_esc <= 0.044, consistent with Hayes+2011". Austin caught it.
    The error and its correction:

    j_mean is built from the OBSERVED surface brightness. No escape
    fraction enters the deprojection at any point -- it is made entirely
    of photons that reached the telescope. So j_mean is the ESCAPED
    emissivity, and Croft measures the ESCAPED emissivity too. They are
    the SAME quantity and are directly comparable. Exceeding Croft's
    upper limit is a TENSION, not a measurement of f_esc.

    f_esc enters only in the last step, light -> star formation:

        rho_SFR_true = case_B_sfr * j_mean / f_esc  >=  case_B_sfr * j_mean

    so rho_SFR at f_esc=1 is a LOWER LIMIT, and a SMALLER f_esc makes the
    implied SFRD LARGER -- moving further from Madau & Dickinson, not
    closer. The old code had it reducing the discrepancy. It does the
    opposite.
    ===================================================================

    Reports the Croft ratio, and separately expresses Croft's limit as a
    FRACTION OF THE STAR-FORMATION PHOTON BUDGET (see
    lya_production_budget) -- which is where the "~5%" that used to be
    attributed to Hayes+2011 actually comes from. It is Croft's own
    limit restated, self-consistently, with no external escape fraction
    imported. That is the honest version of the coincidence.

    CAVEAT: verify Croft+2018's exact definition (observed-frame vs
    rest-frame per-Angstrom -- a factor (1+z), and in the unhelpful
    direction; total mean vs bias-weighted amplitude) against the paper
    before quoting. The arithmetic is validated; the like-for-like-ness
    needs a human read. arXiv:1806.06050 sections 5-6.
    """
    j = np.atleast_1d(j_mean_comoving).astype(float)
    mu_pred = mean_surface_brightness(j, z)
    ratio_croft = mu_pred / mu_limit
    j_croft_max = float(j[0] * mu_limit / mu_pred[0])     # implied j ceiling
    budget = lya_production_budget(z, case_B_sfr)
    rho_min = case_B_sfr * j                       # f_esc = 1 -> LOWER LIMIT

    if verbose:
        print(f"{label}: j_mean (ESCAPED Lya emissivity, no f_esc in it) "
              f"= {j[0]:.4e} erg/s/cMpc^3")
        print(f"  -> predicted mean SB      = {mu_pred[0]:.4e} "
              f"erg/s/cm^2/A/arcsec^2")
        print(f"  Croft+2018 95% UL         = {mu_limit:.4e}")
        print(f"  => {ratio_croft[0]:.1f}x ABOVE the Croft upper limit")
        print(f"     ({ratio_croft[0]*(1+z):.1f}x if Croft's limit is per REST-frame A)")
        print(f"  Croft's limit as an emissivity  = {j_croft_max:.4e} erg/s/cMpc^3")
        print(f"    = {100*j_croft_max/budget['j_intrinsic_max']:.1f}% of the "
              f"SF photon budget ({budget['j_intrinsic_max']:.3e})")
        print(f"  rho_SFR(f_esc=1) = {rho_min[0]:.4f} Msun/yr/cMpc^3 = LOWER LIMIT "
              f"({rho_min[0]/MADAU_DICKINSON(z):.2f}x M&D)")
        if ratio_croft[0] > 3:
            print("  ** TENSION: the cosmic-mean extrapolation is NOT quotable "
                  "until this is resolved. See interpret_2halo() for the "
                  "hypothesis comparison. **")
    return dict(j_mean=j, mu_predicted=mu_pred, mu_limit=mu_limit,
                ratio_over_croft=ratio_croft,
                j_croft_max=j_croft_max,
                croft_frac_of_budget=j_croft_max / budget['j_intrinsic_max'],
                rho_sfr_lower_limit=rho_min, **budget)


def sfrd_vs_fesc(j_mean_comoving, z, case_B_sfr=9.1e-43,
                 f_esc_grid=(1.0, 0.3, 0.1, 0.05, 0.02), verbose=True):
    """
    rho_SFR = case_B_sfr * j_mean / f_esc, tabulated against f_esc.

    The point of this table: rho_SFR at f_esc=1 is a LOWER LIMIT, and
    every physically plausible f_esc makes the implied SFRD LARGER. Any
    apparent 'agreement with Madau & Dickinson' at f_esc=1 is therefore
    the most optimistic possible case, not the expected one.
    """
    j = float(np.atleast_1d(j_mean_comoving)[0])
    MD = MADAU_DICKINSON(z)
    rows = [dict(f_esc=f, rho_sfr=case_B_sfr * j / f,
                 ratio_md=case_B_sfr * j / f / MD) for f in f_esc_grid]
    if verbose:
        print(f"  {'f_esc':>6}{'SFRD':>10}{'x M&D':>9}    (M&D = {MD:.4f})")
        for r in rows:
            print(f"  {r['f_esc']:>6.2f}{r['rho_sfr']:>10.3f}{r['ratio_md']:>9.1f}")
    return rows


def interpret_2halo(j_mean_comoving, z, case_B_sfr=9.1e-43,
                    L_core_erg_s=None, L_2halo_erg_s=None,
                    mu_limit=CROFT2018_MU_LIMIT, verbose=True):
    """
    ADD-ON, not a replacement (Austin 2026-08-05: "we are testing various
    theories here. Not only one may be true.").

    Takes the measured mean emissivity and asks, for each candidate power
    source, what that hypothesis REQUIRES -- so they can be compared side
    by side rather than one being assumed. Several may contribute at once.

    ---------------------------------------------------------------
    H1  IN-SITU CASE-B STAR FORMATION at that radius
        Prediction: j_mean = f_esc * SFRD/case_B_sfr, with f_esc <= 1.
        Test: required_escape_fraction(). If it exceeds 1 the hypothesis
        is dead on a pure photon-budget argument -- you cannot escape
        more photons than are produced -- and NO literature escape
        fraction is needed to say so.

    H2  SCATTERED LIGHT from central + neighbouring galaxies
        Byrohl+2021 (MNRAS 506, 5129): in TNG50, "the majority of photons
        observed in Lya haloes are re-scattered photons from star-forming
        regions", not in-situ emission or satellites; and the outer-profile
        flattening is dominated by photons originating in OTHER halos.
        Prediction: the 2-halo photons are already counted in the cosmic
        SFRD -- they have merely been moved in space. No extra star
        formation is required, and converting j(r) to an SFR density at
        that radius DOUBLE-COUNTS the producing galaxies.
        Test: pass L_core_erg_s and L_2halo_erg_s. The 2-halo/core
        luminosity ratio must be a plausible scattering redistribution
        (order unity or less), not a factor of many.
        Second test: this hypothesis breaks the /xi debias, because
        scattered halo light is far more concentrated around galaxies
        than the galaxy correlation function is -- so b_Lya >> b_gal and
        dividing by xi_gal over-extrapolates to the cosmic mean.

    H3  COLLISIONAL EXCITATION / COOLING RADIATION in the CGM
        Byrohl & Nelson 2023 (MNRAS 523, 5248; arXiv:2212.08666): diffuse
        Lya filaments "are, in fact, illuminated by photons which
        originate, not from the intergalactic medium itself, but from
        within galaxies and their gaseous halos", "primarily sourced by
        intermediate mass halos (1e10 - 1e11 Msun), principally due to
        COLLISIONAL EXCITATIONS in their circumgalactic media as well as
        central, young stellar populations."
        Prediction: a substantial part of the diffuse budget is powered by
        gravitational/thermal energy, NOT by recombination following star
        formation. For that component the case-B + Kennicutt conversion
        is simply the wrong physics and inflates any inferred SFRD.
        Note also the host mass: 1e10-1e11 Msun is BELOW this sample's
        hosts, so the emitters are not the stacked galaxies themselves.
        Test: no clean closed-form one. The discriminator is the free-gamma
        fit plus the environment split -- collisional CGM emission should
        track halo abundance, not the central's properties.

    H4  UV-BACKGROUND FLUORESCENCE
        Cantalupo+2005; Gould & Weinberg 1996. Bacon+2021 cap fluorescence
        at 10-34% of filament luminosity, so it cannot carry the budget
        alone but is a real contributor.

    H5  THE MEASUREMENT / DEBIAS IS OVER-ESTIMATED
        Not a physical hypothesis but must sit in the same table: A2
        inflated by core leakage or background under-subtraction (rho is
        LINEAR in A2), or the /xi extrapolation invalid (see H2).
        Test: fit_floor=True; refit excluding inner bins; r0 sensitivity.
    ---------------------------------------------------------------
    """
    j = float(np.atleast_1d(j_mean_comoving)[0])
    req = required_escape_fraction(j, z, case_B_sfr)
    f_req = float(req['f_esc_required'][0])
    mu_pred = float(np.atleast_1d(mean_surface_brightness(j, z))[0])
    ratio_croft = mu_pred / mu_limit

    out = dict(j_mean=j, z=z,
               j_intrinsic_max=req['j_intrinsic_max'],
               f_esc_required=f_req,
               ratio_over_croft=ratio_croft,
               H1_viable=f_req <= 1.0)
    if L_core_erg_s is not None and L_2halo_erg_s is not None:
        out['twohalo_over_core'] = L_2halo_erg_s / L_core_erg_s

    if not verbose:
        return out

    print("=" * 70)
    print(f"2-HALO POWER SOURCE: hypothesis comparison   (z = {z})")
    print("=" * 70)
    print(f"  measured mean emissivity j_mean = {j:.4e} erg/s/cMpc^3  (ESCAPED)")
    print(f"  SF photon budget, f_esc=1       = {req['j_intrinsic_max']:.4e}")
    print(f"    (= SFRD {req['sfrd']:.4f} / case_B {case_B_sfr:.2e})")
    print()
    print("H1  in-situ case-B star formation")
    print(f"      requires f_esc = f_SF * B * {f_req:.3f}")
    if f_req > 1:
        print(f"      excluded ONLY in the corner f_SF=1, B=1 (needs f_esc={f_req:.3f}).")
        print(f"      Any f_SF < {1/f_req:.2f} revives it, and B < 1 does so")
        print(f"      independently. NOT falsified in general -- see the f_SF x B")
        print(f"      table below. Do NOT report a single 'Nx too much star")
        print(f"      formation' ratio as a finding; that is this corner, not")
        print(f"      a result.")
    else:
        print(f"      viable on budget grounds (f_esc <= 1), not proven.")
    print(f"      Croft ratio {ratio_croft:.1f}x -- constrains B, NOT the SF budget,")
    print(f"      and only if the quantities are like-for-like (unverified).")
    print()
    print("H2  scattered light from galaxies (Byrohl+2021, TNG50)")
    print("      photons already counted in the cosmic SFRD, moved in space.")
    print("      => no extra SF needed; SFR conversion DOUBLE-COUNTS.")
    print("      => also breaks the /xi debias: b_Lya >> b_gal.")
    if 'twohalo_over_core' in out:
        r = out['twohalo_over_core']
        print(f"      L(2-halo)/L(core) = {r:.3f}  "
              f"{'plausible redistribution' if r < 3 else '** too large for pure scattering **'}")
    else:
        print("      [pass L_core_erg_s and L_2halo_erg_s to test this]")
    print()
    print("H3  collisional excitation / cooling in CGM (Byrohl & Nelson 2023)")
    print("      diffuse filaments sourced by collisional excitation in the")
    print("      CGM of 1e10-1e11 Msun halos + central young stars.")
    print("      => case-B + Kennicutt is the WRONG conversion for this part;")
    print("         any SFRD inferred from it is inflated by an unknown factor.")
    print("      => host masses are BELOW this sample's, so the emitters are")
    print("         not the stacked galaxies.")
    print()
    print("H4  UVB fluorescence -- capped at 10-34% by Bacon+2021. Contributor,")
    print("      not the budget.")
    print()
    print("H5  measurement/debias over-estimate")
    print("      A2 inflated (core leakage / background), or /xi invalid.")
    print("      rho is LINEAR in A2. Test: fit_floor, inner-bin exclusion,")
    print("      r0 sensitivity.")
    print()
    print("  REQUIRED f_esc = f_SF * B * %.3f   (f_SF = escaped-stellar fraction," % f_req)
    print("                                      B = xi_gal/xi_eff spatial bias)")
    hdr = "  " + f"{'':>10}" + "".join(f"{('B=1' if b==1 else 'B=1/%d'%round(1/b)):>9}"
                                       for b in (1, 0.5, 0.2, 0.05))
    print(hdr)
    for fsf in (1.0, 0.5, 0.2, 0.1):
        row = "  " + f"f_SF={fsf:.1f}".rjust(10)
        for b in (1, 0.5, 0.2, 0.05):
            v = fsf * b * f_req
            row += f"{v:>9.3f}" + ("" if v <= 1 else "")
        print(row + ("   <- excluded" if fsf * f_req > 1 else ""))
    print()
    print("  Only cells > 1.0 are excluded. f_SF and B trade off EXACTLY, so")
    print("  neither is constrained by our data alone. These hypotheses are not")
    print("  mutually exclusive and the answer is probably a mixture.")
    print("  Do not quote a single ratio as a result -- quote the table.")
    print("=" * 70)
    return out


# Back-compat shim: the old name reported an inverted, wrong result.
def croft_fesc_check(*a, **kw):
    raise RuntimeError(
        "croft_fesc_check() was REMOVED 2026-08-05: its escape-fraction logic "
        "was inverted and it reported a spurious 'f_esc <= 0.044, consistent "
        "with Hayes+2011'. j_mean is the ESCAPED emissivity, so a smaller "
        "f_esc makes the implied SFRD LARGER, not smaller. "
        "Use croft_tension_check() instead.")


# ===============================================================
# VERIFICATION
# ===============================================================
def verify_deprojection(A2=8.184e37, r_c_kpc=359.6, gamma_2d=0.8, z=2.3,
                        verbose=True):
    """
    Prove the chain is arithmetically correct. Every check is a comparison
    between two INDEPENDENT computations, so agreement is evidence rather
    than a tautology. Returns dict of pass/fail.

    This validates the CODE, not the astrophysics. For the astrophysics
    anchor use case_b_sanity_check() against your own core luminosity.
    """
    ok = {}
    gamma_3d = gamma_2d + 1.0
    p = print if verbose else (lambda *a, **k: None)

    p("=" * 66)
    p("CHECK 1: closed-form Abel constant vs brute-force quadrature")
    p("  (analytic Gamma-function expression vs a 2e6-point integral)")
    for g in [1.4, 1.8, 2.2, 3.0]:
        R = 2.0
        P = 4000.0
        u = np.linspace(0, np.arcsinh(P / 0.01), 2000001)
        pg = 0.01 * np.sinh(u)
        # A finite grid cannot reach the r^-gamma tail for shallow gamma
        # (at gamma=1.4 the material beyond 4000 still carries ~4% of the
        # integral). Add that tail analytically -- for pi >> R,
        # int_P^inf pi^-gamma dpi = P^(1-gamma)/(gamma-1).
        tail = 2 * P ** (1 - g) / (g - 1)
        num = 2 * _trapz((np.sqrt(R ** 2 + pg ** 2)) ** (-g), pg) + tail
        ana = abel_constant(g) * R ** (1 - g)
        rel = abs(num / ana - 1)
        ok[f'abel_g{g}'] = rel < 2e-3
        p(f"    gamma={g:>4.1f}  numeric={num:.6f}  analytic={ana:.6f}  "
          f"rel={rel:.2e}  {'PASS' if rel < 2e-3 else 'FAIL'}")

    p("\nCHECK 2: round trip -- deproject A2 to j(r), re-project, recover Sigma")
    p("  (if the unit ladder or the Abel constant were wrong this breaks)")
    R_pkpc = np.array([750.0, 1000.0, 1500.0, 2000.0, 2700.0])
    res = sfr_density_from_expcore(A2, r_c_kpc, gamma_2d, z,
                                   R_eval_pkpc=R_pkpc, use_asymptotic=True)
    C = res['C_gamma']
    Sigma_in = res['Sigma']
    # independent re-projection: j(r) in erg/s/pkpc^3 from the returned
    # comoving j, then integrate numerically along the LOS in pkpc.
    j_phys = res['j_comoving'] * (1 + z) ** 3 / 1e9
    j0_at = j_phys * (R_pkpc / r_c_kpc) ** (gamma_3d)   # normalise to r_c
    j0 = np.median(j0_at)
    u = np.linspace(0, np.arcsinh(3e6 / 1.0), 400001)
    pg = 1.0 * np.sinh(u)
    Sigma_back = np.array([
        2 * _trapz(j0 * (np.sqrt(R ** 2 + pg ** 2) / r_c_kpc) ** (-gamma_3d), pg)
        for R in R_pkpc])
    rel = np.abs(Sigma_back / Sigma_in - 1)
    ok['roundtrip'] = bool(np.all(rel < 5e-3))
    for R, a, b, r in zip(R_pkpc, Sigma_in, Sigma_back, rel):
        p(f"    R={R:>6.0f} pkpc  in={a:.4e}  back={b:.4e}  rel={r:.2e}")
    p(f"    {'PASS' if ok['roundtrip'] else 'FAIL'}  "
      f"(j0 spread across radii = {np.ptp(j0_at)/np.median(j0_at):.2e}, "
      f"should be ~0 for a pure power law)")

    p("\nCHECK 3: projected slope of the model really is -(gamma_3d - 1)")
    Rt = np.array([2000.0, 2040.0])
    St = A2 * (Rt / r_c_kpc) ** (-gamma_2d)
    slope = np.log(St[1] / St[0]) / np.log(Rt[1] / Rt[0])
    ok['slope'] = abs(slope - (1 - gamma_3d)) < 1e-6
    p(f"    dlnSigma/dlnR = {slope:+.6f}   expected {1-gamma_3d:+.6f}   "
      f"{'PASS' if ok['slope'] else 'FAIL'}")

    p("\nCHECK 4: analytic route vs the NUMERICAL fitter (independent code)")
    p("  feed the fitter synthetic Sigma from the 2-halo term alone;")
    p("  it should recover the same emissivity. Large pi_max so both are")
    p("  in the converged limit.")
    Rg = np.linspace(750, 2700, 40)
    Sig_synth = A2 * (Rg / r_c_kpc) ** (-gamma_2d)        # erg/s/kpc^2
    f = fit_emissivity_profile(Rg, Sig_synth, 0.02 * Sig_synth, z,
                               gamma_fixed=gamma_3d, R_min_kpc=750.0,
                               pi_max_Mpc=4000.0)
    r_test = 6.6
    j_fit = float(evaluate_j(f, r_test)[0])
    j_ana = float(sfr_density_from_expcore(
        A2, r_c_kpc, gamma_2d, z, R_eval_pkpc=[r_test / (1 + z) * 1000],
        use_asymptotic=True)['j_comoving'][0])
    rel = abs(j_fit / j_ana - 1)
    ok['analytic_vs_numeric'] = rel < 0.02
    p(f"    j(6.6 cMpc)  analytic={j_ana:.4e}  numeric_fit={j_fit:.4e}  "
      f"rel={rel:.2e}  {'PASS' if rel < 0.02 else 'FAIL'}")

    p("\nCHECK 5: unit ladder, printed term by term (eyeball this)")
    R0 = 1000.0
    S0 = A2 * (R0 / r_c_kpc) ** (-gamma_2d)
    p(f"    Sigma({R0:.0f} pkpc)          = {S0:.4e} erg/s/pkpc^2")
    p(f"    L_eff = C*R              = {C:.4f} * {R0:.0f} = {C*R0:.1f} pkpc"
      f"  ({C*R0*(1+z)/1000:.2f} cMpc)")
    p(f"    j = Sigma/L_eff          = {S0/(C*R0):.4e} erg/s/pkpc^3")
    p(f"    x1e9 -> per pMpc^3       = {S0/(C*R0)*1e9:.4e} erg/s/pMpc^3")
    p(f"    /(1+z)^3={(1+z)**3:.3f} -> per cMpc^3 = {S0/(C*R0)*1e9/(1+z)**3:.4e} erg/s/cMpc^3")
    p(f"    x{9.1e-43:.2e} (case B)   = {9.1e-43*S0/(C*R0)*1e9/(1+z)**3:.4e} Msun/yr/cMpc^3")

    p("\nCHECK 6: plateau really is radius-independent (the cancellation)")
    res2 = sfr_density_from_expcore(A2, r_c_kpc, gamma_2d, z,
                                    R_eval_pkpc=[400, 700, 1200, 2000, 3000],
                                    use_asymptotic=True, gamma_xi=gamma_3d)
    spread = np.ptp(res2['rho_cosmic_plateau']) / np.median(res2['rho_cosmic_plateau'])
    ok['plateau_flat'] = spread < 1e-9
    p(f"    rho_plateau = {np.array2string(res2['rho_cosmic_plateau'], precision=6)}")
    p(f"    fractional spread = {spread:.2e}  "
      f"{'PASS' if ok['plateau_flat'] else 'FAIL'}")
    p("    (exact only when gamma_xi == gamma_3d; if you set gamma_xi")
    p("     different from gamma_2d+1 this SHOULD fail -- that is the")
    p("     circularity caveat made visible.)")

    p("\nCHECK 7: emissivity -> surface brightness, calibrated against a known")
    p("  answer. Take the Madau&Dickinson SFRD, convert to an ESCAPED Lya")
    p("  emissivity at the Hayes+2011 global f_esc~5%, push it through")
    p("  mean_surface_brightness, and check it lands on Croft+2018's 95% UL.")
    p("  Independent of A2 entirely -- this validates the formula AND pins")
    p("  down what the Croft limit physically corresponds to.")
    j_budget = lya_production_budget(z)['j_intrinsic_max']
    mu_budget = float(np.atleast_1d(mean_surface_brightness(j_budget, z))[0])
    frac = CROFT2018_MU_LIMIT / mu_budget
    ok['sb_conversion'] = 0.005 < frac < 0.5
    p(f"    SF photon budget (f_esc=1) = {j_budget:.4e} erg/s/cMpc^3")
    p(f"    -> mu if ALL escaped       = {mu_budget:.4e} erg/s/cm^2/A/arcsec^2")
    p(f"    Croft+2018 95% UL          = {CROFT2018_MU_LIMIT:.4e}")
    p(f"    => Croft caps the escaped fraction of the SF budget at "
      f"{100*frac:.1f}%  {'PASS' if ok['sb_conversion'] else 'FAIL'}")
    p("    This is a SELF-CONTAINED statement of Croft's limit -- it imports")
    p("    no literature escape fraction. (The '~5%' previously attributed to")
    p("    Hayes+2011 was this number all along; Hayes measures a low-z,")
    p("    aperture, galaxy-scale quantity that does not apply here.)")
    # second, independent check: mu must scale exactly as 1/H(z)/(1+z)^2
    Hz = 70.0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7)
    mu_hand = (j_budget * C_KMS / (4 * np.pi * (1 + z) ** 2 * Hz * LYA_REST_A)
               / MPC_CM ** 2 * SR_PER_ARCSEC2)
    rel = abs(mu_hand / mu_budget - 1)
    ok['sb_formula'] = rel < 1e-12
    p(f"    hand-recomputed mu rel diff = {rel:.2e}  "
      f"{'PASS' if ok['sb_formula'] else 'FAIL'}")

    p("\n" + "=" * 66)
    p(f"RESULT: {sum(ok.values())}/{len(ok)} checks passed")
    p("=" * 66)
    return ok


def case_b_sanity_check(L_lya_core_erg_s, sfr_sed_msun_yr, case_B_sfr=9.1e-43,
                        label="core"):
    """
    THE REAL-DATA ANCHOR. Everything above validates arithmetic; this
    validates the ASTROPHYSICAL conversion against data you already
    trust.

    Apply the identical case-B + Kennicutt factor to the CORE Lyman-alpha
    luminosity of your own stack, and compare with the SED/MOSDEF-derived
    SFR of those same galaxies. The ratio is the implied Lyman-alpha
    escape fraction:

        f_esc = SFR_from_Lya / SFR_from_SED

    Expected: ~0.05-0.3 at z~2 (Hayes+2011 global f_esc ~ 5%; LAE-selected
    samples run higher). If you get f_esc > 1 the conversion chain has a
    bug or the core luminosity includes something that isn't case B. If
    you get 1e-3 there is a unit error somewhere.

    This is the check that would have caught the 10x core contamination
    in the earlier power-law fit immediately.
    """
    sfr_lya = case_B_sfr * np.asarray(L_lya_core_erg_s, dtype=float)
    f_esc = sfr_lya / np.asarray(sfr_sed_msun_yr, dtype=float)
    print(f"{label}: L_Lya = {np.atleast_1d(L_lya_core_erg_s)[0]:.3e} erg/s")
    print(f"  SFR implied by case B (f_esc=1) = "
          f"{np.atleast_1d(sfr_lya)[0]:.3f} Msun/yr")
    print(f"  SFR from SED/MOSDEF             = "
          f"{np.atleast_1d(sfr_sed_msun_yr)[0]:.3f} Msun/yr")
    print(f"  => implied f_esc(Lya)           = {np.atleast_1d(f_esc)[0]:.4f}")
    v = float(np.atleast_1d(f_esc)[0])
    if v > 1:
        print("  ** f_esc > 1: impossible. Conversion chain or L_Lya is wrong. **")
    elif v < 0.01:
        print("  ** f_esc < 1%: suspiciously low, check units on L_Lya. **")
    else:
        print("  OK: within the plausible z~2 range (0.01-0.5).")
    return f_esc


# ---------------------------------------------------------------
# Numerical self-test
# ---------------------------------------------------------------
def _self_test():
    """NOTE: this runs on SYNTHETIC data with an arbitrary amplitude, so the
    Croft f_esc guard inside sensitivity_report() will (correctly) shout
    'f_esc > 1'. That is the guard working, not a failure -- the synthetic
    j0 is not a real emissivity. For the real-data path use
    sfr_density_from_expcore() with your fitted A2/r_c, and
    verify_deprojection() for the arithmetic audit."""
    np.random.seed(0)
    z = 2.3
    r_piv_true, gamma_true, j0_true, pmax_true = 2.0, 1.8, 2.0e37, 25.0

    R_kpc = np.linspace(750, 2700, 12)
    Rc = (R_kpc / 1000.0) * (1 + z)
    # project_emissivity is native erg/s/cMpc^2; /1e6 to generate fake
    # data in the erg/s/kpc^2 convention the fitter expects.
    Sigma_true = project_emissivity(Rc, j0_true, r_piv_true, gamma_true,
                                    pi_max_Mpc=pmax_true) / 1e6
    Sigma_err = 0.05 * Sigma_true
    Sigma_obs = Sigma_true + np.random.normal(0, Sigma_err)

    f = fit_emissivity_profile(R_kpc, Sigma_obs, Sigma_err, z,
                               gamma_fixed=None, R_min_kpc=750.0,
                               pi_max_Mpc=pmax_true)
    j_true_at_piv = j0_true * (f['r_pivot_Mpc'] / r_piv_true) ** (-gamma_true)
    print(f"recovery: gamma {gamma_true:.2f} -> {f['gamma']:.3f}+/-{f['gamma_err']:.3f}")
    print(f"          j(pivot) {j_true_at_piv:.3e} -> {f['j0']:.3e}+/-{f['j0_err']:.1e}")
    print(f"          corr(j0,gamma) = {f['corr_j0_gamma']:+.3f} "
          f"(pivot auto-placed at {f['r_pivot_Mpc']:.2f} cMpc)")

    sensitivity_report(R_kpc, Sigma_obs, Sigma_err, z,
                       r_eval_Mpc=6.6, gamma_fixed=1.8)


if __name__ == "__main__":
    _self_test()
