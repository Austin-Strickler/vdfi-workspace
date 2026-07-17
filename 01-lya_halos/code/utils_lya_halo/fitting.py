"""
fitting.py -- PSF-aware / naive two-component exponential fitting
infrastructure for REAL radial flux profiles (average flux per bin,
i.e. boot['total_flux_fid'] from measure_all_bins / run_measurement_suite).

Split out of ../psf_exponential_recovery.py (the Phase-1 synthetic-recovery
testbed -- see specs/psf-exponential-fit-recovery.md), which validates this
same machinery against known-truth fake data via Monte Carlo. This module
keeps only what's needed to fit ACTUAL data:

  * PSF model (Moffat, or any user-supplied (r, vals) curve)
  * intrinsic two-component exponential model I(r) = A1*exp(-r/h1) +
    A2*exp(-r/h2)
  * ring-convolution PSF smearing -> mixing matrix R
  * bin-AVERAGE (not summed) model prediction, matching the fact that each
    real bin value is a biweight AVERAGE of stacked fiber flux, not a total
  * fit_naive  -- no PSF correction, drops the PSF-correlated inner bin(s)
  * fit_psf_aware -- forward-models the PSF, fits against the full binned
    data including the inner bin

psf_exponential_recovery.py imports this module rather than redefining any
of it, so the synthetic-recovery testbed and the real-data fits in
analysis.py (plot_flux_profile_fit) always run the exact same fitting code.

Section 7 below adds specs/halo-flux-fitting.md's "Part 2" (Option C):
I(r) = A1*exp(-r/h1) + A2*(1+(r/r_c)^2)^(-gamma/2), an exponential core plus
a beta-model-softened (cored) power-law halo -- proposed but NOT shipped.
It lives here, not in analysis.py, deliberately: the two-exponential MODEL
in analysis.py's plot_flux_profile_fit stays exactly as-is (validated)
until/unless this extension earns its way in via compare_models_aic_bic.
Everything in Section 7 is additive -- nothing in Sections 1-6 above is
modified. (plot_flux_profile_fit did later gain an opt-in show_components
visualization flag -- see its docstring -- but that's a plotting addition
only, not a change to the fitted model itself.)

Section 8 below adds the fitting half of specs/halo-flux-fitting.md's
"Part 3": a single-exponential UV-continuum radial-decline model,
I(r) = A*exp(-r/h_UV) (default), plus an optional Sersic form with n
floated or fixed, for the CFHT-LS r-band continuum profile -- a genuinely
different data product (broadband imaging, not VIRUS fiber spectra) from
Sections 1-7's Lya models. Only the MODEL + FIT pieces are implemented
here; Part 3's extraction half (cutouts, centroiding, circular annuli,
background subtraction, sample coaddition) is a separate, not-yet-built
job per the spec's own note that it belongs closer to extract.py's
territory than fitting.py's, even while it's staged here for now. Section
8 reuses ring_convolution_matrix (Section 3) completely unmodified -- it
never depended on the intrinsic profile's functional form -- so
PSF-aware UV fits are just "build R with the CFHT-LS PSF instead of the
VIRUS PSF," nothing else.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import integrate, optimize
from scipy.integrate import IntegrationWarning
from scipy.interpolate import interp1d

# np.trapezoid only exists in NumPy >= 2.0 (it's np.trapz on older installs,
# e.g. HPC/cluster environments that lag behind pip). Use whichever is
# available so this runs unmodified on either.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# =======================================================================
# 1. PSF model (1D)
# =======================================================================
def moffat_1d(r, fwhm, beta=3.0):
    """Unnormalized Moffat profile (1 + (r/alpha)^2)^-beta, alpha derived
    from fwhm and beta. beta=3 default, a reasonable stand-in for the
    VIRUS PSF. Returns an array the same shape as r; NOT flux-normalized
    -- call normalize_psf_flux() before using this as a convolution
    kernel."""
    r = np.asarray(r, dtype=float)
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    return (1.0 + (r / alpha) ** 2) ** (-beta)


def normalize_psf_flux(r, psf_vals):
    """Rescale a 1D radial PSF array so integral(psf(r) * 2*pi*r dr) == 1.
    Works on ANY 1D PSF array (the analytic Moffat above, or a real
    measured PSF curve -- e.g. from starpsf.py's psf_profiles_for_lines) --
    swapping in real data later means passing a different (r, psf_vals)
    pair here, nothing else changes."""
    r = np.asarray(r, dtype=float)
    psf_vals = np.asarray(psf_vals, dtype=float)
    total = _trapz(psf_vals * 2 * np.pi * r, r)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "PSF flux integral is non-positive/non-finite; check r/psf_vals "
            "(r should start at or near 0 and extend well past the PSF core).")
    return psf_vals / total


def _psf_interpolator(psf_r, psf_vals):
    """Normalized PSF as a callable P(r), zero outside the sampled range."""
    psf_vals_n = normalize_psf_flux(psf_r, psf_vals)
    return interp1d(psf_r, psf_vals_n, kind="linear",
                     bounds_error=False, fill_value=0.0)


def moffat_encircled_energy_fraction(r_aperture, fwhm, beta=3.0, *,
                                     r_max_factor=50.0, n=4000):
    """
    Fraction of a POINT SOURCE's total flux landing within a circular
    aperture of radius r_aperture, for a Moffat PSF of the given fwhm/beta
    (SAME radial unit as r_aperture -- e.g. both already in kpc, the PSF
    width converted per-galaxy from a fixed angular FWHM via
    cosmo.angular_diameter_distance). This is `EE_i` in
    subsample-derived-properties.md Part 3's per-galaxy point-source
    aperture correction (measure.measure_psf_corrected_core_luminosity):
    the fraction of an unresolved source's flux the innermost radial bin
    actually captures, so 1/EE_i is the correction factor that rescales a
    galaxy's flux back up to its true total.

        EE(r_aperture) = integral_0^r_aperture Moffat(r)*2*pi*r dr
                         / integral_0^r_max     Moffat(r)*2*pi*r dr

    Built the same way normalize_psf_flux/moffat_1d already normalize a PSF
    elsewhere in this module (sample out to r_max_factor*fwhm, trapz), so
    this is trivially consistent with this module's own "total PSF flux"
    convention rather than a separately-derived closed-form Moffat
    encircled-energy formula.

    r_aperture, fwhm : scalar or array; if both are arrays they must
        broadcast together (the intended real use: ONE fixed r_aperture --
        the innermost bin's fixed kpc aperture, same for every galaxy under
        bin_mode='kpc' -- against a (ngal,) array of per-galaxy fwhm_kpc,
        since the PSF's ANGULAR size is fixed but its kpc size differs per
        galaxy's own redshift).
    r_max_factor : PSF sampled out to r_max_factor*fwhm before normalizing
        (default 50 -- generous; a Moffat's power-law wings mean a small
        r_max_factor measurably undercounts the normalization integral).
    n : radial grid points per galaxy's integral.

    Returns EE, same broadcast shape as r_aperture/fwhm (scalar in, scalar
    out), each clipped to [0, 1].
    """
    r_arr = np.asarray(r_aperture, dtype=float)
    f_arr = np.asarray(fwhm, dtype=float)
    scalar_out = (r_arr.ndim == 0) and (f_arr.ndim == 0)
    r_b, f_b = np.broadcast_arrays(r_arr, f_arr)
    flat_r, flat_f = r_b.ravel(), f_b.ravel()

    ee = np.empty(flat_r.shape, dtype=float)
    for i in range(flat_r.size):
        r_max = r_max_factor * flat_f[i]
        r_grid = np.linspace(0.0, r_max, n)
        psf_vals_n = normalize_psf_flux(r_grid, moffat_1d(r_grid, fwhm=flat_f[i], beta=beta))
        cum = integrate.cumulative_trapezoid(psf_vals_n * 2 * np.pi * r_grid, r_grid, initial=0.0)
        ee[i] = np.interp(flat_r[i], r_grid, cum, left=0.0, right=cum[-1])
    ee = np.clip(ee, 0.0, 1.0).reshape(r_b.shape)
    return float(ee) if scalar_out else ee


# =======================================================================
# 2. Intrinsic two-component exponential model
# =======================================================================
def intrinsic_profile(r, A1, h1, A2, h2):
    """I(r) = A1*exp(-r/h1) + A2*exp(-r/h2). h1 < h2 by convention (steep
    core + shallow halo) but not enforced here -- the fits below bound
    both scale lengths > 0 and let the data sort out which is which."""
    r = np.asarray(r, dtype=float)
    return A1 * np.exp(-r / h1) + A2 * np.exp(-r / h2)


def default_fine_grid(r_edges, *, n_per_bin=30, n_tail=30, tail_factor=1.2):
    """
    Build a sensible r_fine grid: n_per_bin points spread across EACH
    observed bin (so narrow inner bins get resolution scaled to their own
    width, not the same fixed absolute spacing as much wider outer bins),
    plus n_tail points extending out to tail_factor * r_edges[-1] so flux
    isn't clipped before PSF smearing gets a chance to move it back into
    the observed range.

    Why this matters: a plain np.linspace over the FULL [0, r_edges[-1]]
    range needs thousands of points to resolve a profile that's
    simultaneously narrow near the core and enormous at the edge -- checked
    directly against a real 10-bin scheme spanning ~10 to ~1000 kpc: a
    uniform grid still hadn't converged on the innermost bin's flux at 4800
    points, while this per-bin grid is within ~1% of the converged value at
    well under 1000. Scaling resolution to each bin's own width fixes the
    mismatch directly instead of brute-forcing it with more points
    everywhere.
    """
    r_edges = np.asarray(r_edges, dtype=float)
    pieces = [np.linspace(r_edges[i], r_edges[i + 1], n_per_bin)
              for i in range(len(r_edges) - 1)]
    tail = np.linspace(r_edges[-1], r_edges[-1] * tail_factor, n_tail)[1:]
    pieces.append(tail)
    r_fine = np.unique(np.concatenate(pieces))
    r_fine[0] = max(r_fine[0], 1e-3)   # avoid exact r=0 (safe, not physically meaningful)
    return r_fine


# =======================================================================
# 3. Ring convolution -> mixing matrix R[n_bins, n_fine]
# =======================================================================
def _ring_convolved_value(rho, r0, psf_interp, psf_rmax, n_phi=161):
    """
    C(rho; r0) = (1/2pi) * integral_0^2pi psf(sqrt(r0^2+rho^2
    -2*r0*rho*cos(phi))) dphi -- the PSF-smeared surface density at
    observed radius rho from a unit-flux thin ring source at true radius
    r0.

    Integrates over the EXACT phi range where the PSF can be nonzero
    (dist <= psf_rmax), found in closed form via arccos, rather than the
    full [0, 2pi]. This matters: for a compact PSF and r0 large compared
    to psf_rmax (routine for outer bins reaching hundreds to thousands of
    kpc while the PSF is only a few kpc across), the nonzero region shrinks
    to a narrow angular slice near phi=0 (width ~ psf_rmax/r0), which a
    FIXED grid spanning the full [0, 2pi] catastrophically under-samples.
    This was a real, previously-shipped bug: the naive fixed-[0,2pi]
    version gave a flux-conservation check (integrate the PSF-smeared ring
    back out to a total) of ~13 instead of 1 at r0=1000 kpc, growing
    roughly linearly with r0 -- caught by testing a synthetic truth with a
    large h2 (broad halo component reaching into wide outer bins), which
    the original small-r0-only validation never exercised. Restricting to
    the exact support fixes it at no extra cost.
    """
    if r0 <= 0.0 or rho <= 0.0:
        dist = abs(r0 - rho)
        return float(psf_interp(dist)) if dist <= psf_rmax else 0.0

    cos_max = np.clip((r0 ** 2 + rho ** 2 - psf_rmax ** 2) / (2.0 * r0 * rho), -1.0, 1.0)
    if cos_max >= 1.0:
        return 0.0   # PSF cannot reach this (rho, r0) pair at all
    phi_max = np.arccos(cos_max)

    phi = np.linspace(0.0, phi_max, n_phi)
    radicand = np.maximum(r0 ** 2 + rho ** 2 - 2.0 * r0 * rho * np.cos(phi), 0.0)
    dist = np.sqrt(radicand)
    vals = psf_interp(dist)
    half_integral = _trapz(vals, phi)
    return (2.0 * half_integral) / (2 * np.pi)   # double by phi<->2pi-phi symmetry


def ring_convolution_matrix(r_fine, r_edges, psf_r, psf_vals, *,
                            n_phi=161, n_rho=121, verbose=False):
    """
    R[i, j] = fraction of a unit-flux thin ring at r_fine[j] observed
    within annulus [r_edges[i], r_edges[i+1]] after PSF smearing.

    Precompute ONCE per (r_fine, r_edges, PSF) -- it does not depend on
    (A1, h1, A2, h2), since convolution is linear. Every trial parameter
    set during fitting reuses this same matrix:
        observed_per_bin = R @ _fine_ring_flux(r_fine, A1, h1, A2, h2)
    NOTE: multiply by _fine_ring_flux's output (intrinsic_profile times
    the grid spacing dr), not intrinsic_profile(r_fine, ...) directly --
    R[i,j] is a fraction of actual FLUX, not of the raw profile value.

    The rho-integral is likewise restricted to the exact support window
    [max(bin_lo, r0-psf_rmax), min(bin_hi, r0+psf_rmax)] rather than the
    full (possibly very wide) bin range, for the same reason as the phi
    restriction above. Pairs with no overlap at all are skipped entirely
    (R[i,j] left at 0), which is also a real speedup since most (bin,
    r_fine-point) pairs don't overlap once bins are far from r0.
    """
    r_fine = np.asarray(r_fine, dtype=float)
    r_edges = np.asarray(r_edges, dtype=float)
    psf_interp = _psf_interpolator(psf_r, psf_vals)
    psf_rmax = float(np.max(psf_r))

    n_bins = len(r_edges) - 1
    n_fine = len(r_fine)
    R = np.zeros((n_bins, n_fine))

    for j, r0 in enumerate(r_fine):
        support_lo = max(0.0, r0 - psf_rmax)
        support_hi = r0 + psf_rmax
        for i in range(n_bins):
            r_lo = max(r_edges[i], support_lo)
            r_hi = min(r_edges[i + 1], support_hi)
            if r_lo >= r_hi:
                continue   # no overlap -- R[i, j] stays 0
            rho_grid = np.linspace(r_lo, r_hi, n_rho)
            if r_lo < r0 < r_hi:
                # make sure the (possibly sharp) peak at rho=r0 is
                # actually sampled, not just bracketed by nearby points
                rho_grid = np.union1d(rho_grid, [r0])
            c_vals = np.array([_ring_convolved_value(rho, r0, psf_interp, psf_rmax, n_phi=n_phi)
                                for rho in rho_grid])
            R[i, j] = _trapz(c_vals * 2 * np.pi * rho_grid, rho_grid)

    if verbose:
        # Column sum R[:, j] = fraction of a unit-flux ring at r_fine[j] that
        # lands ANYWHERE in the observed range [0, r_edges[-1]] after smearing.
        # It should be ~1 well inside the range and taper toward 0 only for
        # rings near/past the outer edge (whose flux the PSF pushes off-grid).
        col_sums = R.sum(axis=0)
        interior = col_sums[r_fine < 0.9 * r_edges[-1]]
        print("ring_convolution_matrix:")
        print(f"  R shape            : {R.shape}  ({n_bins} bins x {n_fine} fine radii)")
        print(f"  quadrature         : n_phi={n_phi}, n_rho={n_rho}")
        print(f"  PSF support (rmax)  : {float(np.max(psf_r)):.1f} (same units as r)")
        print(f"  flux retained per ring (column sums), interior rings only:")
        print(f"     min={interior.min():.4f}  median={np.median(interior):.4f}"
              f"  max={interior.max():.4f}   (want ~1.0)")
        print(f"  -> interpret: {(1-np.median(interior))*100:.2f}% median flux "
              f"loss off the fine grid; refine psf_r / r_fine if this is large.")

    return R


def _brute_force_ring_value(rho, r0, psf_interp, psf_rmax):
    """Reference implementation using scipy.integrate.quad (restricted to
    the same exact support window) instead of the fast trapz-over-phi
    shortcut. Used ONLY for validation, never in the hot path."""
    if r0 <= 0.0 or rho <= 0.0:
        dist = abs(r0 - rho)
        return float(psf_interp(dist)) if dist <= psf_rmax else 0.0
    cos_max = np.clip((r0 ** 2 + rho ** 2 - psf_rmax ** 2) / (2.0 * r0 * rho), -1.0, 1.0)
    if cos_max >= 1.0:
        return 0.0
    phi_max = np.arccos(cos_max)

    def integrand(phi):
        dist = np.sqrt(max(r0 ** 2 + rho ** 2 - 2.0 * r0 * rho * np.cos(phi), 0.0))
        return psf_interp(dist)
    # For large r0 the PSF support is a sharp spike in a narrow phi window, so
    # quad legitimately hits roundoff before reaching its default tolerance and
    # emits an IntegrationWarning. This path is VALIDATION-ONLY (never in the
    # matrix build) and still agrees with the fast trapz path to <2e-4, so the
    # warning is cosmetic -- silence it here rather than letting it clutter the
    # validation output. A slightly looser epsabs also keeps quad from
    # over-refining a spike it can't resolve to full double precision.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = integrate.quad(integrand, 0, phi_max, limit=200, epsabs=1e-10)
    return (2.0 * val) / (2 * np.pi)


def validate_ring_convolution(psf_r=None, psf_vals=None, verbose=True):
    """
    Regression check: compare the fast trapz-based ring convolution
    against scipy.integrate.quad, AND check that a unit-flux ring's
    PSF-smeared profile integrates back out to ~1 (flux conservation) at
    both small and LARGE r0. The large-r0 check is the one that catches
    the phi-under-sampling bug described in _ring_convolved_value's
    docstring -- don't drop it even though it makes this check slower.
    """
    if psf_r is None:
        psf_r = np.linspace(0, 60, 400)
    if psf_vals is None:
        psf_vals = moffat_1d(psf_r, fwhm=3.0, beta=3.0)
    psf_interp = _psf_interpolator(psf_r, psf_vals)
    psf_rmax = float(np.max(psf_r))

    test_pairs = [(5.0, 8.0), (20.0, 25.0), (60.0, 55.0), (0.0, 3.0)]
    max_rel_err = 0.0
    for r0, rho in test_pairs:
        fast = _ring_convolved_value(rho, r0, psf_interp, psf_rmax)
        brute = _brute_force_ring_value(rho, r0, psf_interp, psf_rmax)
        rel_err = abs(fast - brute) / max(abs(brute), 1e-12)
        max_rel_err = max(max_rel_err, rel_err)
        if verbose:
            print(f"  r0={r0:6.1f} rho={rho:6.1f}  fast={fast:.6e}  "
                  f"brute={brute:.6e}  rel_err={rel_err:.2e}")
    ok = max_rel_err < 1e-3

    max_conservation_err = 0.0
    for r0 in (1.0, 10.0, 100.0, 500.0, 1000.0, 2000.0):
        lo, hi = max(0.0, r0 - psf_rmax), r0 + psf_rmax
        rho_grid = np.linspace(lo, hi, 2001)
        c_vals = np.array([_ring_convolved_value(rho, r0, psf_interp, psf_rmax) for rho in rho_grid])
        total = _trapz(c_vals * 2 * np.pi * rho_grid, rho_grid)
        err = abs(total - 1.0)
        max_conservation_err = max(max_conservation_err, err)
        if verbose:
            print(f"  flux-conservation check: r0={r0:8.1f}  "
                  f"full-support integral={total:.6f} (want 1.0)")
    ok = ok and (max_conservation_err < 0.01)

    if verbose:
        print(f"validate_ring_convolution: max phi-quadrature rel error = "
              f"{max_rel_err:.2e}, max flux-conservation error = "
              f"{max_conservation_err:.2e} -> {'OK' if ok else 'CHECK n_phi/n_rho'}")
    return ok


def _fine_ring_flux(r_fine, A1, h1, A2, h2):
    """
    The actual flux carried by the true radial shell at each r_fine
    sample: intrinsic_profile(r) treated as dF/dr (flux per unit radius --
    the natural reading given the real data product here is already a 1D
    flux(r) curve, not a 2D surface-brightness map), times this grid's
    local (trapezoidal) spacing dr.

    This -- NOT the raw intrinsic_profile values alone -- is what gets
    redistributed via the ring-convolution matrix R. Skipping the dr
    weighting is a real bug, not a style choice: R[i,j] is built as "the
    fraction of a UNIT-FLUX ring at r_fine[j] landing in bin i," so
    multiplying it by anything other than the actual flux quantum at
    r_fine[j] makes the predicted bin flux scale with however many
    r_fine points happen to land in each (very unevenly sized) coarse
    bin -- an artifact of the grid resolution, not a converging physical
    quantity.
    """
    r_fine = np.asarray(r_fine, dtype=float)
    dr = np.gradient(r_fine)
    return intrinsic_profile(r_fine, A1, h1, A2, h2) * dr


def _bin_integrate(r_fine, r_edges, A1, h1, A2, h2):
    """
    Properly bin-integrated, NO-PSF model prediction: sums each fine
    shell's _fine_ring_flux into whichever COARSE bin it geometrically
    falls in -- no redistribution across bins (that's what
    ring_convolution_matrix/R is for), just correct accounting for bin
    width.

    This -- NOT a raw point-evaluation of intrinsic_profile at each bin's
    midpoint -- is what fit_naive fits against. For a bin scheme with a
    narrow innermost bin and a very wide outermost bin, a fast-declining
    exponential's value at one point can differ from its true integral
    over the bin by many orders of magnitude for the widest bins.
    """
    r_fine = np.asarray(r_fine, dtype=float)
    r_edges = np.asarray(r_edges, dtype=float)
    fine_flux = _fine_ring_flux(r_fine, A1, h1, A2, h2)
    n_bins = len(r_edges) - 1
    bin_idx = np.searchsorted(r_edges, r_fine, side="right") - 1
    result = np.zeros(n_bins)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    np.add.at(result, bin_idx[valid], fine_flux[valid])
    return result


def _bin_widths(r_edges):
    """Width of each observed bin, r_edges[i+1]-r_edges[i]."""
    e = np.asarray(r_edges, dtype=float)
    return e[1:] - e[:-1]


# ---------------------------------------------------------------------
# AVERAGE (not summed) per-bin model.
#
# The real data value in each bin is the biweight AVERAGE of the stacked
# fiber fluxes in that annulus -- a representative flux(r), NOT the total
# flux summed over the annulus. So the model prediction for a bin must be
# the bin *mean* of the profile, = (1/width) * integral over the bin, not
# the raw integral. Dividing the integrated quantities above by the bin
# width converts them:
#   * bin_average_no_psf : mean of the intrinsic profile over each bin.
#   * bin_average_psf    : mean of the PSF-SMEARED profile over each bin
#                          (R still carries the ring-convolution geometry;
#                          we only re-normalize its per-bin integral to a
#                          per-bin mean).
# Doing it this way keeps the validated ring-convolution / flux-conservation
# machinery untouched -- R is still "integrated fraction of a unit-flux
# ring"; the averaging is a single, explicit /width at the model level.
# ---------------------------------------------------------------------
def bin_average_no_psf(r_fine, r_edges, A1, h1, A2, h2):
    """Mean intrinsic flux in each bin (NO PSF): _bin_integrate / bin width."""
    return _bin_integrate(r_fine, r_edges, A1, h1, A2, h2) / _bin_widths(r_edges)


def bin_average_psf(R, r_fine, r_edges, A1, h1, A2, h2):
    """Mean PSF-smeared flux in each bin: (R @ fine_flux) / bin width."""
    return (R @ _fine_ring_flux(r_fine, A1, h1, A2, h2)) / _bin_widths(r_edges)


# =======================================================================
# 4. Seeding, bounds, and result packaging
# =======================================================================
def _seed_from_split(r, y, i_mid):
    """One candidate two-exponential seed from splitting the data at index
    i_mid and estimating a scale length from each half's endpoints."""
    def _h(r0, r1, y0, y1, fallback):
        if y0 <= y1 or y1 <= 0:
            return fallback
        return max((r1 - r0) / np.log(y0 / y1), 1e-3)

    i_mid = int(np.clip(i_mid, 1, len(r) - 2))
    h1_0 = _h(r[0], r[i_mid], y[0], y[i_mid], fallback=max(r[i_mid] - r[0], 1.0))
    h2_0 = _h(r[i_mid], r[-1], y[i_mid], y[-1], fallback=max(r[-1] - r[i_mid], 1.0))
    A1_0 = max(float(y[0]), 1e-3)
    A2_0 = max(float(y[i_mid]) * 0.5, 1e-3)
    return [A1_0, h1_0, A2_0, h2_0]


def _default_seeds(r, y):
    """A handful of candidate seeds (different split points, plus an
    h1/h2-swapped variant) rather than a single guess -- two-exponential
    fits are prone to local minima on sparse, widely-spaced radial bins,
    the same instability speclsf.py's double-Gaussian fit already guards
    against with a multi-seed-then-best-chi2 pattern. Reused here."""
    r = np.asarray(r, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 1e-6, None)
    n = len(r)

    seeds = []
    for frac in (0.3, 0.5, 0.7):
        i_mid = max(1, int(round(n * frac)))
        seeds.append(_seed_from_split(r, y, i_mid))
    # swapped variant of the middle seed: try (A2,h2,A1,h1) order in case
    # the "core"/"halo" identity is ambiguous from the split alone
    A1_0, h1_0, A2_0, h2_0 = seeds[len(seeds) // 2]
    seeds.append([A2_0, h2_0, A1_0, h1_0])
    return seeds


def estimate_truth_from_profile(r_mid, y, *, i_mid=None, verbose=False):
    """
    Quick, no-optimizer, two-exponential ballpark estimate (A1, h1, A2, h2)
    from a REAL (or real-like) binned flux profile -- e.g. your actual
    boot['total_flux_fid'] and its bin midpoints. Meant to give a
    "similar, not exact" starting point, calibrated to your data's real
    amplitude scale and overall decline, WITHOUT running curve_fit on
    noisy real bins (some of which may be tiny or slightly negative at
    large radius, exactly where a real nonlinear fit is prone to the
    local-minima issues fit_naive/fit_psf_aware guard against with
    multiple seeds).

    r_mid : bin midpoints, same units you want h1/h2 reported in (e.g. kpc).
    y     : the real flux values at those bins (any amplitude scale --
            this just estimates two exponential segments from a couple of
            endpoint ratios, so it doesn't care whether y ~ 1 or y ~ 1e39).
    i_mid : index splitting "inner" from "outer" bins; defaults to the
            middle bin.

    Returns (A1, h1, A2, h2). Can be passed as p0 to fit_naive/fit_psf_aware.
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    if i_mid is None:
        i_mid = len(r_mid) // 2
    floor = 1e-6 * max(np.max(np.abs(y)), 1.0)   # relative floor -- works at any amplitude scale
    y_clipped = np.clip(y, floor, None)
    A1_0, h1_0, A2_0, h2_0 = _seed_from_split(r_mid, y_clipped, i_mid)
    if verbose:
        print("estimate_truth_from_profile: no-optimizer ballpark seed")
        print(f"  split index i_mid={i_mid} (r_mid={r_mid[i_mid]:.1f})")
        print(f"  -> A1={A1_0:.3g}  h1={h1_0:.3g}  A2={A2_0:.3g}  h2={h2_0:.3g}")
        print("  (this is only a starting guess; the fit calibrates it "
              "against the bin-averaged model next.)")
    return (A1_0, h1_0, A2_0, h2_0)


def _data_driven_bounds(r_fit, y_fit):
    """
    Bounds scaled to the actual data instead of a blanket (0, inf).
    Two-exponential fits on sparse, noisy bins can otherwise wander into
    numerically-degenerate "solutions" (e.g. amplitude -> 1e8 while its
    scale length -> ~0, contributing ~nothing everywhere but technically
    lowering chi2 by a hair) that aren't physically meaningful for THIS
    profile -- no reason a scale length should run to many times the
    radial span actually being fit, or an amplitude to many times the
    largest observed bin. Applied identically to fit_naive and
    fit_psf_aware, so neither is favored by the choice.
    """
    amp_max = max(50.0 * np.max(np.abs(y_fit)), 1.0)
    h_max = max(10.0 * (np.max(r_fit) - np.min(r_fit)), 10.0)
    lower = [0.0, 1e-3, 0.0, 1e-3]
    upper = [amp_max, h_max, amp_max, h_max]
    return lower, upper


def _pack_result(success, popt=None, pcov=None, reason=None, mask=None,
                 extra=None):
    if not success:
        return {"success": False, "reason": reason}
    perr = (np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov))
            else np.full(4, np.nan))
    A1, h1, A2, h2 = popt
    out = {"success": True,
           "A1": A1, "h1": h1, "A2": A2, "h2": h2,
           "A1_err": perr[0], "h1_err": perr[1],
           "A2_err": perr[2], "h2_err": perr[3],
           "popt": popt, "pcov": pcov, "mask": mask}
    if extra:
        out.update(extra)
    return out


def describe_fit(result, *, label="fit", truth=None):
    """Pretty-print a fit result dict: recovered (A1,h1,A2,h2) with 1-sigma
    errors, chi2/dof if present, and -- when a `truth` (A1,h1,A2,h2) is given
    (only meaningful for synthetic recovery tests) -- the signed percentage
    recovery error on each parameter. Purely a reporting helper; returns
    nothing."""
    if not result.get("success"):
        print(f"[{label}] FAILED: {result.get('reason')}")
        return
    names = ("A1", "h1", "A2", "h2")
    print(f"[{label}] converged"
          + (f"  chi2/dof = {result['chi2']:.2f}/{result['dof']} "
             f"= {result['chi2'] / max(result['dof'], 1):.2f}"
             if "chi2" in result else ""))
    for i, k in enumerate(names):
        line = f"    {k:>2} = {result[k]:12.4g} +/- {result[f'{k}_err']:<10.3g}"
        if truth is not None:
            t = truth[i]
            pct = (result[k] - t) / t * 100 if t != 0 else float("nan")
            line += f"   truth={t:11.4g}   err={pct:+7.1f}%"
        print(line)


def _amp_scale(y_fit):
    """A positive flux scale (~the largest bin) used to normalize the fit so
    the amplitude PARAMETERS are order-unity. Real data here is ~1e39, and
    curve_fit's Levenberg-Marquardt step/tolerance logic silently stalls at the
    seed when parameters span ~1e39 against scale lengths ~1-1000 (the Jacobian
    columns differ by ~40 orders of magnitude, so the normal-equations matrix is
    hopelessly ill-conditioned). Fitting y/scale with amplitudes in units of
    `scale` fixes the conditioning; h1/h2 are scale-invariant. Caught directly:
    on raw 1e39 flux the fit returned the seed unchanged (chi2/dof ~ 5e5, model
    6-400x above the data); with this scaling the same data fits at chi2/dof<1."""
    s = float(np.nanmax(np.abs(y_fit))) if len(y_fit) else 1.0
    return s if (np.isfinite(s) and s > 0) else 1.0


def _canonical_order(popt, pcov):
    """Force the steep component to be (A1,h1): the two-exponential model is
    symmetric under swapping (A1,h1)<->(A2,h2), so an unconstrained fit can
    return the core and halo in either slot. Left as-is this makes 'h1 vs truth'
    look wrong half the time and blows up any Monte Carlo scatter (a draw's h1
    randomly being the tiny core or the large halo). Sorting so h1<=h2
    makes h1 always the core, h2 always the halo -- the convention the rest of
    the code assumes."""
    popt = np.asarray(popt, dtype=float)
    if popt[1] > popt[3]:
        idx = [2, 3, 0, 1]
        popt = popt[idx]
        if pcov is not None:
            pcov = pcov[np.ix_(idx, idx)]
    return popt, pcov


def _best_of_seeds(model, r_fit, y_fit, e_fit, seeds, bounds):
    """Try curve_fit from each seed in turn, keep the lowest-chi2 result
    that converges. Returns (popt, pcov) or (None, None) if every seed
    failed."""
    best = None
    for p0 in seeds:
        p0 = list(np.clip(p0, bounds[0], bounds[1]))
        try:
            popt, pcov = optimize.curve_fit(
                model, r_fit, y_fit, p0=p0, sigma=e_fit,
                absolute_sigma=True, bounds=bounds, maxfev=20000)
        except Exception:
            continue
        pred = model(r_fit, *popt)
        chi2 = float(np.nansum(((y_fit - pred) / e_fit) ** 2))
        if not np.isfinite(chi2):
            continue
        if best is None or chi2 < best[0]:
            best = (chi2, popt, pcov)
    if best is None:
        return None, None
    return best[1], best[2]


# =======================================================================
# 5. Two fits: naive (no PSF, drop inner bin) vs PSF-aware (full forward model)
# =======================================================================
def fit_naive(r_mid, r_edges, r_fine, y, yerr, *, fit_skip_inner=1, p0=None,
              verbose=False):
    """
    Two-component exponential fit with NO PSF: fits the bin-AVERAGED
    model (bin_average_no_psf -- correct accounting for bin width, just no
    PSF redistribution between bins) against the data, after dropping
    the innermost fit_skip_inner bin(s) (default 1 -- the innermost bin is
    the one most contaminated by the PSF-smeared core, so it's the standard
    thing to drop for a naive/no-PSF fit; pass fit_skip_inner=0 to keep it).

    NOTE: fits the properly bin-AVERAGED model, not a raw point-evaluation
    of intrinsic_profile at each bin's midpoint -- for a bin scheme with a
    narrow inner bin and a very wide outer bin those can differ by orders
    of magnitude. "naive" here means "ignores the PSF," not "ignores bin
    width/averaging too."
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.arange(len(r_mid)) >= fit_skip_inner
    mask &= np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if mask.sum() < 4:
        return _pack_result(False, reason="fewer than 4 usable bins after fit_skip_inner")

    y_fit, e_fit = y[mask], yerr[mask]

    # Fit in amplitude-normalized units so curve_fit stays well-conditioned on
    # real ~1e39 fluxes (see _amp_scale). The two amplitude params are carried
    # in units of `scale`; h1/h2 are untouched.
    scale = _amp_scale(y_fit)

    def _model_masked(_, a1, h1, a2, h2):
        return bin_average_no_psf(r_fine, r_edges, a1 * scale, h1, a2 * scale, h2)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds(r_mid, y)
    seeds = [[s[0] / scale, s[1], s[2] / scale, s[3]] for s in seeds]
    lo, up = _data_driven_bounds(r_mid[mask], y_fit)
    bounds = ([lo[0] / scale, lo[1], lo[2] / scale, lo[3]],
              [up[0] / scale, up[1], up[2] / scale, up[3]])

    popt, pcov = _best_of_seeds(_model_masked, r_mid[mask], y_fit / scale,
                                e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result(False, reason="all seeds failed to converge")

    # un-scale amplitudes (and their covariance) back to real flux units,
    # then order so h1 is the steep core, h2 the shallow halo.
    D = np.array([scale, 1.0, scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)
    popt, pcov = _canonical_order(popt, pcov)

    # full-bin prediction (mean flux per bin, SAME units as y) + chi2 on the
    # fitted bins -- both handed back so callers/plots don't recompute them.
    model_binned = bin_average_no_psf(r_fine, r_edges, *popt)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - 4
    result = _pack_result(True, popt, pcov, mask=mask,
                          extra={"model_binned": model_binned, "chi2": chi2,
                                 "dof": dof, "n_fit": int(mask.sum())})
    if verbose:
        n_drop = int((~mask).sum())
        print(f"fit_naive: NO PSF, dropped {fit_skip_inner} inner bin(s) "
              f"({int(mask.sum())} bins fit, {n_drop} excluded)")
        describe_fit(result, label="naive")
    return result


def fit_psf_aware(r_mid, y, yerr, R, r_fine, r_edges, *, p0=None, verbose=False):
    """
    Two-component exponential fit WITH the PSF forward model: trial
    (A1,h1,A2,h2) -> intrinsic_profile on r_fine -> R @ (...) -> per-bin
    MEAN flux -> compared against the FULL binned data (no bins dropped --
    the PSF model accounts for the inner-bin correlation directly instead of
    discarding the information).

    r_edges is required so the forward model can be normalized to a per-bin
    average (matching the biweight-averaged data), not a per-bin sum.

    Same multi-seed-then-best-chi2 strategy as fit_naive.
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if mask.sum() < 4:
        return _pack_result(False, reason="fewer than 4 finite bins")

    y_fit, e_fit = y[mask], yerr[mask]

    # amplitude-normalized fit for conditioning (see _amp_scale / fit_naive).
    scale = _amp_scale(y_fit)

    def _model_masked(_, a1, h1, a2, h2):
        return bin_average_psf(R, r_fine, r_edges, a1 * scale, h1, a2 * scale, h2)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds(r_mid, y)
    seeds = [[s[0] / scale, s[1], s[2] / scale, s[3]] for s in seeds]
    lo, up = _data_driven_bounds(r_mid[mask], y_fit)
    bounds = ([lo[0] / scale, lo[1], lo[2] / scale, lo[3]],
              [up[0] / scale, up[1], up[2] / scale, up[3]])

    popt, pcov = _best_of_seeds(_model_masked, r_mid[mask], y_fit / scale,
                                e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result(False, reason="all seeds failed to converge")

    D = np.array([scale, 1.0, scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)
    popt, pcov = _canonical_order(popt, pcov)

    # full-bin PSF-forward prediction (mean flux per bin, like y) + chi2.
    model_binned = bin_average_psf(R, r_fine, r_edges, *popt)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - 4
    result = _pack_result(True, popt, pcov, mask=mask,
                          extra={"model_binned": model_binned, "chi2": chi2,
                                 "dof": dof, "n_fit": int(mask.sum())})
    if verbose:
        print(f"fit_psf_aware: WITH PSF forward model, all {int(mask.sum())} "
              f"bins fit (inner bin kept)")
        describe_fit(result, label="psf")
    return result


# =======================================================================
# 6. Shared diagnostics helper
# =======================================================================
def binned_model_from_result(result, r_fine, r_edges, R=None):
    """The fit's prediction in the SAME per-bin AVERAGE-flux units as the
    data. Prefers the `model_binned` the fit already stashed; otherwise
    rebuilds it from the fitted params (PSF-aware -> R @ fine_flux needs R;
    naive -> bin_average_no_psf needs r_edges)."""
    if result.get("model_binned") is not None:
        return np.asarray(result["model_binned"], dtype=float)
    if not result.get("success"):
        return None
    params = (result["A1"], result["h1"], result["A2"], result["h2"])
    if R is not None and r_edges is not None:
        return bin_average_psf(R, r_fine, r_edges, *params)
    if r_edges is not None:
        return bin_average_no_psf(r_fine, r_edges, *params)
    return None


# =======================================================================
# 7. Part 2 (Option C, proposed/not-shipped) -- exponential core + cored
#    power-law halo. See specs/halo-flux-fitting.md, "Part 2 -- proposed
#    extension". Additive only: nothing above this line is touched, and
#    analysis.py is not touched at all -- this is the testing-only home for
#    this model per that spec's own framing ("not-started extension").
# =======================================================================
def intrinsic_profile_expcore(r, A1, h1, A2, r_c, gamma):
    """
    I(r) = A1*exp(-r/h1) + A2*(1 + (r/r_c)^2)^(-gamma/2).

    A beta-model/Plummer-style softened power law for the outer term
    instead of Part 1's second exponential: as r -> 0 it flattens to A2
    (finite, unlike a bare power law which diverges), and as r >> r_c it
    asymptotes to A2*(r/r_c)^(-gamma), recovering a pure power law -- the
    functional form motivated in halo-flux-fitting.md Part 2 as a stand-in
    for a clustering/two-halo-term outer profile that doesn't just cut off
    the way an exponential does.

    Not symmetric under swapping its two terms the way intrinsic_profile
    (Part 1) is under (A1,h1)<->(A2,h2) -- an exponential core and a cored
    power-law halo are different shapes, so no canonical-ordering step is
    needed anywhere below (see _canonical_order's docstring for the
    contrast; this model has no equivalent).
    """
    r = np.asarray(r, dtype=float)
    return A1 * np.exp(-r / h1) + A2 * (1.0 + (r / r_c) ** 2) ** (-gamma / 2.0)


def _fine_ring_flux_expcore(r_fine, A1, h1, A2, r_c, gamma):
    """Same role as _fine_ring_flux (Part 1): flux per r_fine sample =
    intrinsic_profile_expcore treated as dF/dr, times local grid spacing
    dr. This, not the raw profile values, is what ring_convolution_matrix's
    R redistributes -- see _fine_ring_flux's docstring for why."""
    r_fine = np.asarray(r_fine, dtype=float)
    dr = np.gradient(r_fine)
    return intrinsic_profile_expcore(r_fine, A1, h1, A2, r_c, gamma) * dr


def _bin_integrate_expcore(r_fine, r_edges, A1, h1, A2, r_c, gamma):
    """Same role as _bin_integrate (Part 1), for intrinsic_profile_expcore:
    properly bin-integrated, NO-PSF model prediction."""
    r_fine = np.asarray(r_fine, dtype=float)
    r_edges = np.asarray(r_edges, dtype=float)
    fine_flux = _fine_ring_flux_expcore(r_fine, A1, h1, A2, r_c, gamma)
    n_bins = len(r_edges) - 1
    bin_idx = np.searchsorted(r_edges, r_fine, side="right") - 1
    result = np.zeros(n_bins)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    np.add.at(result, bin_idx[valid], fine_flux[valid])
    return result


def bin_average_no_psf_expcore(r_fine, r_edges, A1, h1, A2, r_c, gamma):
    """Mean intrinsic (expcore) flux in each bin, NO PSF -- same
    divide-by-bin-width convention as Part 1's bin_average_no_psf, per
    halo-flux-fitting.md Part 2's note that the average-vs-summed question
    is already answered by Part 1 and needs no new work here."""
    return _bin_integrate_expcore(r_fine, r_edges, A1, h1, A2, r_c, gamma) / _bin_widths(r_edges)


def bin_average_psf_expcore(R, r_fine, r_edges, A1, h1, A2, r_c, gamma):
    """Mean PSF-smeared (expcore) flux in each bin: (R @ fine_flux) / bin
    width. R is the SAME ring_convolution_matrix used for Part 1's model --
    it depends only on (r_fine, r_edges, psf_r, psf_vals), never on the
    intrinsic profile's functional form, so it's reused completely
    unchanged here (see halo-flux-fitting.md Part 2, 'Reuses the existing
    machinery almost unchanged')."""
    return (R @ _fine_ring_flux_expcore(r_fine, A1, h1, A2, r_c, gamma)) / _bin_widths(r_edges)


def _default_seeds_expcore(r, y, *, gamma_fixed=None, r_c_guess=None):
    """Candidate (A1,h1,A2,r_c[,gamma]) seeds for the expcore model. Reuses
    _seed_from_split for the core term (A1,h1) exactly as Part 1 does; the
    outer term's amplitude is seeded from the same split, and r_c from the
    split radius itself (a genuine free parameter here, not a hand-picked
    cutoff -- see spec).

    gamma_fixed=None (gamma floats): seeds a spread of literature-motivated
    starting slopes -- 1.8 (typical z~2-3 3D galaxy-clustering slope), 0.8
    (that same slope fully projected via Limber, see
    projected_slope_from_3d), and 1.3 splitting the difference -- plus one
    seed with a wider r_c, rather than a single arbitrary guess. Two
    exp-core fits on sparse bins can be just as seed-sensitive as Part 1's
    two-exponential fits (per spec's discussion of r_c/gamma trading off
    non-Gaussian-ly), so this follows the same multi-seed philosophy.

    gamma_fixed=<float> (gamma held fixed, e.g. to directly test 1.8 vs 0.8
    -- open question 3 in the spec): only A1,h1,A2,r_c are free, so seeds
    just vary r_c around the split-radius guess.
    """
    r = np.asarray(r, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 1e-6, None)
    n = len(r)
    i_mid = max(1, n // 2)
    A1_0, h1_0, A2_0, _ = _seed_from_split(r, y, i_mid)
    A2_0 = max(A2_0, 1e-3)
    if r_c_guess is None:
        r_c_guess = max(float(r[i_mid]), 1.0)

    if gamma_fixed is not None:
        return [
            [A1_0, h1_0, A2_0, r_c_guess],
            [A1_0, h1_0, A2_0, r_c_guess * 0.5],
            [A1_0, h1_0, A2_0, r_c_guess * 2.0],
        ]
    return [
        [A1_0, h1_0, A2_0, r_c_guess, 0.8],
        [A1_0, h1_0, A2_0, r_c_guess, 1.8],
        [A1_0, h1_0, A2_0, r_c_guess, 1.3],
        [A1_0, h1_0, A2_0, r_c_guess * 2.0, 1.3],
        # extra r_c variants anchored at gamma=0.8 specifically: real-data
        # testing (see conversation/usage notes) found the free-gamma fit
        # can land in a WORSE local minimum than the gamma=0.8-fixed fit
        # even though the fixed case is a feasible point of the free
        # search -- a genuine r_c/gamma degeneracy on sparse bins, not a
        # hypothetical one (matches the spec's own warning that these two
        # parameters are more prone to trading off non-Gaussian-ly than
        # h1/h2 are). These two extra starting points make it more likely
        # a seed already sits close to that basin.
        [A1_0, h1_0, A2_0, r_c_guess * 0.5, 0.8],
        [A1_0, h1_0, A2_0, r_c_guess * 2.0, 0.8],
    ]


def _data_driven_bounds_expcore(r_fit, y_fit, *, gamma_fixed=None, gamma_bounds=(0.3, 3.5)):
    """Same data-scaled philosophy as Part 1's _data_driven_bounds (see its
    docstring for why blanket (0, inf) bounds let sparse/noisy fits wander
    into numerically-degenerate solutions), extended with r_c > 0 (bounded
    away from zero -- a genuine transition-scale parameter, not a
    hand-picked cutoff per the spec's core objection to a bare power law)
    and gamma constrained to a physically sane range. Default gamma_bounds
    (0.3, 3.5) brackets both literature comparison points discussed in the
    spec -- ~0.8 (fully projected) and ~1.8 (raw 3D) -- with headroom
    either side rather than pinning the box to exactly those two numbers.
    """
    amp_max = max(50.0 * np.max(np.abs(y_fit)), 1.0)
    h_max = max(10.0 * (np.max(r_fit) - np.min(r_fit)), 10.0)
    rc_min = max(1e-3, 0.01 * np.min(r_fit))
    lower = [0.0, 1e-3, 0.0, rc_min]
    upper = [amp_max, h_max, amp_max, h_max]
    if gamma_fixed is None:
        lower.append(gamma_bounds[0])
        upper.append(gamma_bounds[1])
    return lower, upper


def _pack_result_expcore(success, popt=None, pcov=None, reason=None, mask=None,
                          gamma_fixed=None, extra=None):
    """Same role as Part 1's _pack_result, for the (A1,h1,A2,r_c,gamma)
    parameter set. When gamma_fixed is not None, gamma isn't a free
    parameter of the fit -- it's still reported (as the value it was fixed
    to) with gamma_err=0.0 and gamma_fixed=True in the result, so
    describe_fit_expcore / downstream code doesn't need a separate code
    path to know whether gamma floated."""
    if not success:
        return {"success": False, "reason": reason}
    if gamma_fixed is None:
        perr = (np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov))
                else np.full(5, np.nan))
        A1, h1, A2, r_c, gamma = popt
        gamma_err = perr[4]
    else:
        perr = (np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov))
                else np.full(4, np.nan))
        A1, h1, A2, r_c = popt
        gamma = gamma_fixed
        gamma_err = 0.0
    out = {"success": True, "model": "expcore",
           "A1": A1, "h1": h1, "A2": A2, "r_c": r_c, "gamma": gamma,
           "gamma_fixed": gamma_fixed is not None,
           "A1_err": perr[0], "h1_err": perr[1], "A2_err": perr[2],
           "r_c_err": perr[3], "gamma_err": gamma_err,
           "popt": popt, "pcov": pcov, "mask": mask}
    if extra:
        out.update(extra)
    return out


def describe_fit_expcore(result, *, label="fit", truth=None):
    """Pretty-print an expcore fit result -- same role as Part 1's
    describe_fit.

    NOTE: earlier versions of this function auto-printed "if gamma=X is
    read as a 3D slope, projected = X-1" whenever gamma_fixed was set. That
    was wrong in the common case where gamma was deliberately fixed TO the
    already-projected value (e.g. gamma_fixed=projected_slope_from_3d(1.8)
    == 0.8) to test it directly -- printing "as a 3D slope, projected =
    -0.20" under a gamma=0.8 fit is nonsensical and actively misleading
    about what was being tested, caught from real usage. This function no
    longer guesses which convention a fixed gamma was meant in; call
    projected_slope_from_3d/deprojected_slope_bounds yourself if you
    specifically fixed gamma to a 3D value and want its projected
    counterpart."""
    if not result.get("success"):
        print(f"[{label}] FAILED: {result.get('reason')}")
        return
    names = ("A1", "h1", "A2", "r_c", "gamma")
    print(f"[{label}] converged"
          + (f"  chi2/dof = {result['chi2']:.2f}/{result['dof']} "
             f"= {result['chi2'] / max(result['dof'], 1):.2f}"
             if "chi2" in result else ""))
    for k in names:
        tag = " (fixed)" if (k == "gamma" and result.get("gamma_fixed")) else ""
        line = f"    {k:>5} = {result[k]:12.4g} +/- {result[f'{k}_err']:<10.3g}{tag}"
        if truth is not None and k in truth:
            t = truth[k]
            pct = (result[k] - t) / t * 100 if t != 0 else float("nan")
            line += f"   truth={t:11.4g}   err={pct:+7.1f}%"
        print(line)


def _best_of_seeds_expcore(*args, **kwargs):
    """Alias for Part 1's _best_of_seeds -- it's already fully generic
    (any model callable, any parameter count), so nothing new is needed
    here; kept as a named alias only so callers/readers of this section
    don't have to jump back to Section 4 to find it."""
    return _best_of_seeds(*args, **kwargs)


# ---------------------------------------------------------------------
# Two fits, same naive/PSF-aware split as Part 1 (fit_naive/fit_psf_aware),
# for the expcore model.
# ---------------------------------------------------------------------
def fit_naive_expcore(r_mid, r_edges, r_fine, y, yerr, *, fit_skip_inner=1,
                       gamma_fixed=0.8, p0=None, verbose=False):
    """
    Expcore analogue of fit_naive: no PSF correction, drops fit_skip_inner
    innermost bin(s), fits bin_average_no_psf_expcore against the rest.
    See fit_naive's docstring for why the bin-AVERAGE (not point-eval)
    model is used, and fit_psf_aware_expcore's docstring for gamma_fixed
    (default 0.8, same rationale -- pass None to let gamma float).
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.arange(len(r_mid)) >= fit_skip_inner
    mask &= np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    n_free = 4 if gamma_fixed is not None else 5
    if mask.sum() < n_free:
        return _pack_result_expcore(False, reason=f"fewer than {n_free} usable bins after fit_skip_inner")

    y_fit, e_fit = y[mask], yerr[mask]
    scale = _amp_scale(y_fit)

    if gamma_fixed is None:
        def _model_masked(_, a1, h1, a2, r_c, gamma):
            return bin_average_no_psf_expcore(r_fine, r_edges, a1 * scale, h1, a2 * scale, r_c, gamma)[mask] / scale
    else:
        def _model_masked(_, a1, h1, a2, r_c):
            return bin_average_no_psf_expcore(r_fine, r_edges, a1 * scale, h1, a2 * scale, r_c, gamma_fixed)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds_expcore(r_mid, y, gamma_fixed=gamma_fixed)
    if gamma_fixed is None:
        seeds = [[s[0] / scale, s[1], s[2] / scale, s[3], s[4]] for s in seeds]
    else:
        seeds = [[s[0] / scale, s[1], s[2] / scale, s[3]] for s in seeds]

    lo, up = _data_driven_bounds_expcore(r_mid[mask], y_fit, gamma_fixed=gamma_fixed)
    if gamma_fixed is None:
        bounds = ([lo[0] / scale, lo[1], lo[2] / scale, lo[3], lo[4]],
                  [up[0] / scale, up[1], up[2] / scale, up[3], up[4]])
    else:
        bounds = ([lo[0] / scale, lo[1], lo[2] / scale, lo[3]],
                  [up[0] / scale, up[1], up[2] / scale, up[3]])

    popt, pcov = _best_of_seeds_expcore(_model_masked, r_mid[mask], y_fit / scale,
                                        e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result_expcore(False, reason="all seeds failed to converge")

    D = np.array([scale, 1.0, scale, 1.0, 1.0] if gamma_fixed is None else [scale, 1.0, scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)
    # no _canonical_order step -- see intrinsic_profile_expcore's docstring:
    # this model isn't swap-symmetric, so there's no core/halo slot ambiguity.

    if gamma_fixed is None:
        model_binned = bin_average_no_psf_expcore(r_fine, r_edges, *popt)
    else:
        model_binned = bin_average_no_psf_expcore(r_fine, r_edges, popt[0], popt[1], popt[2], popt[3], gamma_fixed)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - n_free
    result = _pack_result_expcore(True, popt, pcov, mask=mask, gamma_fixed=gamma_fixed,
                                  extra={"model_binned": model_binned, "chi2": chi2,
                                         "dof": dof, "n_fit": int(mask.sum()), "k_params": n_free})
    if verbose:
        n_drop = int((~mask).sum())
        print(f"fit_naive_expcore: NO PSF, dropped {fit_skip_inner} inner bin(s) "
              f"({int(mask.sum())} bins fit, {n_drop} excluded), gamma "
              f"{'fixed=' + str(gamma_fixed) if gamma_fixed is not None else 'free'}")
        describe_fit_expcore(result, label="naive-expcore")
    return result


def fit_psf_aware_expcore(r_mid, y, yerr, R, r_fine, r_edges, *, gamma_fixed=0.8,
                          p0=None, verbose=False):
    """
    Expcore analogue of fit_psf_aware: full PSF forward model, all bins fit
    (inner bin kept), same multi-seed-then-lowest-chi2 strategy. r_edges
    required for the same reason as fit_psf_aware (per-bin average
    normalization). R is the SAME ring_convolution_matrix built for the
    Part 1 model -- it's independent of the intrinsic profile's functional
    form, so nothing about building R changes for this model.

    gamma_fixed : 0.8 (default) -- projected_slope_from_3d(1.8), the
        Limber-projected z~2-3 clustering slope (LITERATURE_GAMMA_3D). r_c
        and gamma are strongly degenerate when both float, and on real data
        (even the highest-S/N ALL stack) the free-gamma fit is essentially
        unconstrained (r_c and gamma errors comparable to or larger than
        the values themselves) AND compare_models_aic_bic prefers the
        plain two-exponential model over expcore either way -- so floating
        gamma buys nothing and actively destabilizes r_c, especially in
        lower-S/N subsamples. Pass None to let gamma float anyway (bounded
        per _data_driven_bounds_expcore), or another float (e.g. 1.8, the
        raw unprojected slope) to test a different literature value
        directly via chi2/AIC (open question 3 in halo-flux-fitting.md
        Part 2) -- either way, fewer free parameters than the None case,
        which is the point.
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    n_free = 4 if gamma_fixed is not None else 5
    if mask.sum() < n_free:
        return _pack_result_expcore(False, reason=f"fewer than {n_free} finite bins")

    y_fit, e_fit = y[mask], yerr[mask]
    scale = _amp_scale(y_fit)

    if gamma_fixed is None:
        def _model_masked(_, a1, h1, a2, r_c, gamma):
            return bin_average_psf_expcore(R, r_fine, r_edges, a1 * scale, h1, a2 * scale, r_c, gamma)[mask] / scale
    else:
        def _model_masked(_, a1, h1, a2, r_c):
            return bin_average_psf_expcore(R, r_fine, r_edges, a1 * scale, h1, a2 * scale, r_c, gamma_fixed)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds_expcore(r_mid, y, gamma_fixed=gamma_fixed)
    if gamma_fixed is None:
        seeds = [[s[0] / scale, s[1], s[2] / scale, s[3], s[4]] for s in seeds]
    else:
        seeds = [[s[0] / scale, s[1], s[2] / scale, s[3]] for s in seeds]

    lo, up = _data_driven_bounds_expcore(r_mid[mask], y_fit, gamma_fixed=gamma_fixed)
    if gamma_fixed is None:
        bounds = ([lo[0] / scale, lo[1], lo[2] / scale, lo[3], lo[4]],
                  [up[0] / scale, up[1], up[2] / scale, up[3], up[4]])
    else:
        bounds = ([lo[0] / scale, lo[1], lo[2] / scale, lo[3]],
                  [up[0] / scale, up[1], up[2] / scale, up[3]])

    popt, pcov = _best_of_seeds_expcore(_model_masked, r_mid[mask], y_fit / scale,
                                        e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result_expcore(False, reason="all seeds failed to converge")

    D = np.array([scale, 1.0, scale, 1.0, 1.0] if gamma_fixed is None else [scale, 1.0, scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)

    if gamma_fixed is None:
        model_binned = bin_average_psf_expcore(R, r_fine, r_edges, *popt)
    else:
        model_binned = bin_average_psf_expcore(R, r_fine, r_edges, popt[0], popt[1], popt[2], popt[3], gamma_fixed)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - n_free
    result = _pack_result_expcore(True, popt, pcov, mask=mask, gamma_fixed=gamma_fixed,
                                  extra={"model_binned": model_binned, "chi2": chi2,
                                         "dof": dof, "n_fit": int(mask.sum()), "k_params": n_free})
    if verbose:
        print(f"fit_psf_aware_expcore: WITH PSF forward model, all {int(mask.sum())} "
              f"bins fit (inner bin kept), gamma "
              f"{'fixed=' + str(gamma_fixed) if gamma_fixed is not None else 'free'}")
        describe_fit_expcore(result, label="psf-expcore")
    return result


def binned_model_from_result_expcore(result, r_fine, r_edges, R=None):
    """Expcore analogue of binned_model_from_result: the fit's prediction
    in the same per-bin AVERAGE-flux units as the data."""
    if result.get("model_binned") is not None:
        return np.asarray(result["model_binned"], dtype=float)
    if not result.get("success"):
        return None
    params = (result["A1"], result["h1"], result["A2"], result["r_c"], result["gamma"])
    if R is not None and r_edges is not None:
        return bin_average_psf_expcore(R, r_fine, r_edges, *params)
    if r_edges is not None:
        return bin_average_no_psf_expcore(r_fine, r_edges, *params)
    return None


# ---------------------------------------------------------------------
# Connecting the outer term to a clustering / two-halo-term signal.
#
# halo-flux-fitting.md Part 2's central physical point: a 3D two-point
# correlation xi(r) ~ (r/r0)^(-gamma) does NOT project to a 2D radial
# profile of the same slope. The Limber approximation says a fully
# line-of-sight-integrated power-law correlation is one power SHALLOWER
# than the 3D slope: R^-(gamma-1), not R^-gamma. For a typical z~2-3
# galaxy-clustering gamma=1.8, the fully-projected expectation is closer to
# R^-0.8 -- plausibly why a straight -1.8 fit looked worse than the
# exponential (per this module's Part 2 motivation) even if a
# clustering/two-halo-term outer profile is still the right physical
# picture. How much of the shift applies depends on the line-of-sight
# window depth vs. a fiducial correlation length r0 (open question 4 in the
# spec) -- these two helpers give the two bracketing endpoints, not a
# single verified number.
# ---------------------------------------------------------------------
def projected_slope_from_3d(gamma_3d):
    """Limber approximation: a 3D correlation slope gamma_3d, FULLY
    projected along the line of sight, gives a 2D radial slope of
    gamma_3d - 1. This is one of two bracketing endpoints -- see
    deprojected_slope_bounds and the section docstring above."""
    return gamma_3d - 1.0


def deprojected_slope_bounds(gamma_3d):
    """The two endpoints bracketing the expected 2D outer slope for a 3D
    correlation slope gamma_3d: a narrow line-of-sight window leaves the
    slope close to gamma_3d itself (unprojected limit); a window deep
    compared to the fiducial correlation length r0 pushes it to
    gamma_3d - 1 (fully-projected/Limber limit, projected_slope_from_3d).
    Returns (projected_limit, unprojected_limit) = (gamma_3d - 1, gamma_3d).
    Where the real data's line-of-sight window falls between these is an
    open question (item 4 in halo-flux-fitting.md Part 2) -- treat this as
    a bracket to test both ends of (e.g. via gamma_fixed in
    fit_psf_aware_expcore), not a resolved single value."""
    return (projected_slope_from_3d(gamma_3d), gamma_3d)


# Literature 3D clustering slopes worth testing via gamma_fixed (pass
# through projected_slope_from_3d for the fully-projected 2D comparison
# value). Not exhaustive -- see halo_gas_correlation_literature_review.md
# for the fuller literature context this pipeline's results are checked
# against.
LITERATURE_GAMMA_3D = {
    "z~2-3 galaxy clustering (typical)": 1.8,
}


def compare_models_aic_bic(result_a, result_b, *, label_a="two-exp", label_b="expcore",
                           verbose=True):
    """
    Compare two fit results (e.g. Part 1's fit_psf_aware two-exponential
    result vs. this section's fit_psf_aware_expcore result) via AIC/BIC on
    the SAME bins -- per halo-flux-fitting.md Part 2's explicit guidance
    that once both models exist, this should decide between them rather
    than a visual "exponential looks better" impression, since the outer
    bins (where the two models diverge most) are exactly where noise is
    most likely to fool a by-eye comparison.

    Both AIC = chi2 + 2k and BIC = chi2 + k*ln(n) assume the same chi2
    definition on the same data points -- if result_a/result_b were fit
    with different fit_skip_inner or different `method` (psf vs naive),
    n_fit may differ, which this flags but does not correct for (compare
    like-for-like: both psf-aware, or both naive with the same
    fit_skip_inner).

    k is read from result['k_params'] if present (expcore results, since k
    depends on gamma_fixed), else inferred as len(result['popt']) (Part 1
    two-exponential results: always 4).

    Returns a dict with each model's chi2/dof/k/AIC/BIC, the AIC/BIC
    differences (b - a), and which model each criterion prefers. Rule of
    thumb (not a hard cutoff): |delta AIC| > ~10 is usually decisive, 4-10
    substantial, <2 negligible.
    """
    def _k(result):
        return result.get("k_params", len(result["popt"]))

    n_a, n_b = result_a["n_fit"], result_b["n_fit"]
    if n_a != n_b and verbose:
        print(f"compare_models_aic_bic: WARNING n_fit differs ({label_a}={n_a} vs "
              f"{label_b}={n_b}) -- AIC/BIC only meaningfully compare fits on the "
              f"SAME data points; check fit_skip_inner / method consistency.")

    k_a, k_b = _k(result_a), _k(result_b)
    aic_a, aic_b = result_a["chi2"] + 2 * k_a, result_b["chi2"] + 2 * k_b
    bic_a = result_a["chi2"] + k_a * np.log(max(n_a, 1))
    bic_b = result_b["chi2"] + k_b * np.log(max(n_b, 1))

    out = {
        label_a: {"chi2": result_a["chi2"], "dof": result_a["dof"], "k": k_a, "AIC": aic_a, "BIC": bic_a},
        label_b: {"chi2": result_b["chi2"], "dof": result_b["dof"], "k": k_b, "AIC": aic_b, "BIC": bic_b},
        "delta_AIC (b-a)": aic_b - aic_a,
        "delta_BIC (b-a)": bic_b - bic_a,
        "preferred_by_AIC": label_a if aic_a < aic_b else label_b,
        "preferred_by_BIC": label_a if bic_a < bic_b else label_b,
    }
    if verbose:
        print(f"compare_models_aic_bic: {label_a} vs {label_b}")
        print(f"  {label_a:>10}: chi2/dof={result_a['chi2']:.2f}/{result_a['dof']}  "
              f"k={k_a}  AIC={aic_a:.2f}  BIC={bic_a:.2f}")
        print(f"  {label_b:>10}: chi2/dof={result_b['chi2']:.2f}/{result_b['dof']}  "
              f"k={k_b}  AIC={aic_b:.2f}  BIC={bic_b:.2f}")
        print(f"  delta AIC ({label_b}-{label_a}) = {out['delta_AIC (b-a)']:+.2f}   "
              f"delta BIC = {out['delta_BIC (b-a)']:+.2f}")
        print(f"  preferred: AIC -> {out['preferred_by_AIC']}   BIC -> {out['preferred_by_BIC']}")
    return out


# ---------------------------------------------------------------------
# Literature comparison points, computed from fit results rather than
# hand-read off a plot -- so they can be overlaid on plot_expcore_fit (or
# any future comparison plot) directly, instead of re-derived by eye each
# time. See halo_gas_correlation_literature_review.md Section 7 for the
# real-data numbers these were checked against.
# ---------------------------------------------------------------------
def crossover_radius_twoexp(A1, h1, A2, h2):
    """Radius where the two-exponential model's two terms are equal:
    A1*exp(-r/h1) = A2*exp(-r/h2). Both terms are exponentials, so this
    solves in closed form: r = ln(A1/A2) / (1/h1 - 1/h2).

    This is the physically meaningful "one-halo term stops dominating,
    two-halo term takes over" radius -- NOT h1 or h2 individually, and
    NOT the same quantity as r_c in the expcore model (see
    crossover_radius_expcore's docstring). For the shipped real-data fit
    (A1=6.254e39, h1=16.9, A2=7.318e37, h2=1552) this comes out to ~76
    kpc, essentially exactly R_vir (~75 kpc) -- the qualitative
    "one-halo-dominated inside R_vir, two-halo-dominated outside"
    picture, now as an actual number rather than an eyeballed break point.

    Returns None if h1==h2 (undefined, division by zero) or if the
    implied crossover is <= 0 (terms don't cross for r>0 with these
    amplitudes/scales -- e.g. the "core" term never actually dominates
    the innermost bins at all, which would itself be worth flagging)."""
    if h1 == h2:
        return None
    r = np.log(A1 / A2) / (1.0 / h1 - 1.0 / h2)
    return float(r) if r > 0 else None


def crossover_radius_expcore(A1, h1, A2, r_c, gamma, r_max=None):
    """Radius where the expcore model's two terms are equal:
    A1*exp(-r/h1) = A2*(1+(r/r_c)^2)^(-gamma/2). No closed form (mixed
    exponential/power-law), so root-find with scipy.optimize.brentq.

    This is a DIFFERENT quantity from r_c itself: r_c is where the
    SECOND term alone turns from flat/cored to power-law-declining;
    this crossover is where the FIRST term stops dominating the second.
    Confirmed concretely on real data: a psf-aware fit with gamma fixed
    at the Limber-projected 0.8 gave A1=6.261e39, h1=16.89, A2=7.113e37,
    r_c=488, gamma=0.8 -- r_c itself is ~6.5x R_vir (~75 kpc), but this
    function's crossover for those same parameters comes out to ~76 kpc,
    i.e. essentially the SAME one-halo/two-halo break the two-exponential
    model gives (crossover_radius_twoexp), even though r_c doesn't match
    R_vir at all. Two different, both physically meaningful radii from
    one fit -- see halo_gas_correlation_literature_review.md Section 7.

    r_max : upper bracket for the root search. Default max(50*h1, 5*r_c)
    -- generous enough that if term1 hasn't fallen below term2 by r_max,
    there's effectively no crossover in any physically relevant range.

    Returns None if no sign change is found in (1e-3, r_max), i.e. one
    term dominates throughout the searched range."""
    def diff(r):
        return (A1 * np.exp(-r / h1)
                - A2 * (1.0 + (r / r_c) ** 2) ** (-gamma / 2.0))
    if r_max is None:
        r_max = max(50.0 * h1, 5.0 * r_c)
    lo, hi = 1e-3, r_max
    f_lo, f_hi = diff(lo), diff(hi)
    if f_lo == 0:
        return lo
    if f_lo * f_hi > 0:
        return None
    return float(optimize.brentq(diff, lo, hi))


# ---------------------------------------------------------------------
# subsample-derived-properties.md Part 1 -- the core/halo boundary a
# per-subsample table row uses for BOTH the velocity split (Part 2) and the
# luminosity split (Part 3): "wherever the fit itself says core hands off to
# halo," generalized across whichever model (two-exponential today, expcore
# once/if it ships) was actually fit for that row -- see crossover_radius_
# twoexp / crossover_radius_expcore above for the two model-specific
# closed-form/root-find implementations this just dispatches between.
# ---------------------------------------------------------------------
def find_core_halo_boundary(fit_result, *, fallback_fit_result=None,
                            fallback_radius=None, r_max=None):
    """
    The core/halo boundary for ONE subsample's fit_result: the radius where
    its own fitted core term and halo term are equal. Dispatches purely on
    fit_result's shape -- fit_result.get('model')=='expcore' (has
    (A1,h1,A2,r_c,gamma), e.g. from fit_psf_aware_expcore/fit_naive_expcore
    or plot_flux_profile_two(fit=True)) routes to crossover_radius_expcore;
    anything else (no 'model' key, (A1,h1,A2,h2) instead -- the two-
    exponential fits from fit_psf_aware/fit_naive/plot_flux_profile_fit)
    routes to crossover_radius_twoexp. Callers never need to know or track
    which model was actually fit.

    Accepted risk, per the spec: this is NOT a fixed radius across every
    subsample row -- each row's own fit has its own crossover, so a
    low-mass row's boundary and a high-mass row's boundary can legitimately
    differ. That's the deliberate tradeoff (see the spec's "Accepted risk,
    stated plainly").

    Fallback (real risk on a noisier half-sample split, per the spec):
    when fit_result's own fit doesn't converge, OR it converges but the two
    terms never actually cross anywhere in the searched range (both are
    genuine failure modes, not hypothetical -- a two-exponential/expcore fit
    on ~10 sparse bins is prone to local minima, worse on half the sample),
    fall back in this order and flag boundary_from_own_fit=False:
        1. fallback_fit_result's OWN crossover (typically the full-stack
           fit on the SAME model -- already known to converge and cross),
           if given.
        2. fallback_radius, an explicit number (e.g. a previously computed
           full-stack boundary, or R_vir) if given.
    Never invents a boundary from nothing: if fit_result needs a fallback
    and neither is given, boundary_radius comes back None with
    source="none" -- the caller decides what to do (e.g. skip that row, or
    flag it in the eventual table per the spec's `boundary_from_own_fit`
    column).

    r_max : forwarded to crossover_radius_expcore's own root-search bracket
        (ignored for the two-exponential closed-form case, which needs no
        bracket). None -> that function's own default
        (max(50*h1, 5*r_c)).

    Returns
    -------
    dict: boundary_radius (float or None), boundary_from_own_fit (bool),
        source ("own" | "fallback_fit" | "fallback_radius" | "none"),
        model (the model actually used for the returned boundary_radius,
        "expcore" or "twoexp").
    """
    def _crossover(fr):
        if not fr.get("success"):
            return None, fr.get("model", "twoexp")
        m = fr.get("model", "twoexp")
        if m == "expcore":
            return (crossover_radius_expcore(fr["A1"], fr["h1"], fr["A2"],
                                             fr["r_c"], fr["gamma"], r_max=r_max), m)
        return (crossover_radius_twoexp(fr["A1"], fr["h1"], fr["A2"], fr["h2"]), m)

    own_radius, own_model = _crossover(fit_result)
    if own_radius is not None:
        return {"boundary_radius": own_radius, "boundary_from_own_fit": True,
               "source": "own", "model": own_model}

    if fallback_fit_result is not None:
        fb_radius, fb_model = _crossover(fallback_fit_result)
        if fb_radius is not None:
            return {"boundary_radius": fb_radius, "boundary_from_own_fit": False,
                   "source": "fallback_fit", "model": fb_model}

    if fallback_radius is not None:
        return {"boundary_radius": float(fallback_radius), "boundary_from_own_fit": False,
               "source": "fallback_radius", "model": own_model}

    return {"boundary_radius": None, "boundary_from_own_fit": False,
           "source": "none", "model": own_model}


def sphere_of_influence_kpc(rvir_kpc, factor=7.0):
    """Sorini, Onorbe, Hennawi & Lukic 2018 (ApJ 859, 125)'s
    observationally-calibrated 'sphere of influence' scale: the mean Lya
    absorption profile around a galaxy converges onto the pure LCDM
    (linear-theory) prediction beyond roughly ~7x the halo virial radius
    (their sample: ~700 kpc-1 Mpc, quoted as '~7x virial radius'). This
    is the radius beyond which the two-halo/clustering term has fully
    taken over from any halo-specific astrophysics -- conceptually closer
    to where the expcore model's r_c should land than to
    crossover_radius_expcore/_twoexp (see those docstrings for the
    distinction). Checked directly: for R_vir=75 kpc this predicts ~525
    kpc, and a real psf-aware expcore fit (gamma fixed at 0.8) returned
    r_c=488+/-211 kpc -- consistent within 1sigma, and via a totally
    different observable (Sorini: absorption/tomography; this pipeline:
    emission stacking) -- see halo_gas_correlation_literature_review.md
    Section 7.

    factor=7.0 is an order-of-magnitude literature number, not a
    precisely calibrated one -- Sorini's own quoted range (700 kpc-1 Mpc
    for their sample) corresponds to roughly +/-30% around the central
    ~7x figure once their fiducial R_vir is accounted for, so treat that
    as the reasonable systematic latitude, not the number's precision."""
    return factor * rvir_kpc


# ---------------------------------------------------------------------
# r_c vs. "the radius where this actually IS a power law" -- these are
# NOT the same number, and conflating them is exactly the confusion this
# section exists to resolve.
#
# r_c is where the beta-model SOFTENING term stops being flat -- it is a
# statement about the FUNCTIONAL FORM chosen to avoid a divergence at
# r=0, not a statement about when clustering physics takes over. The
# halo term A2*(1+(r/r_c)^2)^(-gamma/2) only recovers a genuine
# (r/r_c)^(-gamma) power law in the limit r >> r_c; right at r=r_c it is
# still roughly halfway between "flat" and "power law" in log-log slope
# (see effective_slope_expcore below -- exactly -gamma/2 at r=r_c, by
# construction). So quoting r_c as "the correlation length" or "where
# clustering starts" is the same category error the reading material
# above warns about for r0 itself: r_c is a shape parameter of the
# assumed functional form, not a physically-privileged transition scale
# read directly off the data.
# ---------------------------------------------------------------------
def effective_slope_expcore(r, r_c, gamma):
    """
    Local (point-by-point) logarithmic slope of the halo term ALONE,
    d ln[A2*(1+(r/r_c)^2)^(-gamma/2)] / d ln r, evaluated analytically
    (independent of A2 -- the amplitude cancels out of a log-derivative):

        slope(r) = -gamma * (r/r_c)^2 / (1 + (r/r_c)^2)

    This is 0 at r=0 (flat core), passes through exactly -gamma/2 at
    r=r_c (by construction -- this IS what "r_c" means: the halfway
    point in log-log slope, not the onset of power-law behavior), and
    only approaches the asymptotic -gamma once r is several times r_c.
    Use this (or radius_of_slope_fraction below) instead of r_c itself
    when you want to know where the profile actually LOOKS like a
    power law of slope gamma, rather than where the softening kicks in.
    """
    r = np.asarray(r, dtype=float)
    x2 = (r / r_c) ** 2
    return -gamma * x2 / (1.0 + x2)


def radius_of_slope_fraction(r_c, gamma, frac=0.9):
    """
    Closed-form radius where the halo term's local log-log slope
    (effective_slope_expcore) first reaches `frac` of its asymptotic
    value, i.e. slope(r) = -frac*gamma.

    Solving -gamma*x^2/(1+x^2) = -frac*gamma for x=r/r_c gives
    x = sqrt(frac / (1 - frac)), so:

        r_frac = r_c * sqrt(frac / (1 - frac))

    For frac=0.9 (the default -- "90% of the way to the true asymptotic
    slope"): r_frac = r_c * 3.0. For frac=0.5: r_frac = r_c exactly (see
    effective_slope_expcore's docstring -- this is just confirming r_c
    IS the 50%-of-the-way point, never the 90%-or-more point real
    literature power laws are usually compared against).

    `frac` must be in [0, 1); raises ValueError otherwise (frac=1 would
    require r_frac = infinity -- the pure power law is only reached in
    the true asymptotic limit, never at finite r).
    """
    if not (0.0 <= frac < 1.0):
        raise ValueError(f"frac must be in [0, 1) (got {frac})")
    return r_c * np.sqrt(frac / (1.0 - frac))


def local_loglog_slope(r, y):
    """
    Generic NUMERICAL local log-log slope d ln y / d ln r via central
    differences on log-log axes -- works on ANY curve (the full expcore
    model, the full two-exponential model, real binned data, etc.), not
    just the analytic halo term effective_slope_expcore handles in
    closed form. Useful for asking "what slope does the TOTAL fitted
    profile (core+halo added together) actually show at radius r", since
    the core term's contribution can flatten or steepen the total curve
    away from the halo term's own analytic slope, especially near their
    crossover radius (crossover_radius_expcore/_twoexp).

    r, y : arrays of the SAME length, r > 0 and y > 0 (log of a
    non-positive value is undefined -- filter those points out first if
    your profile can go negative, e.g. noisy real outer bins).

    Returns an array the same length as r (NaN at the two endpoints,
    where a centered difference isn't available).
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(r <= 0) or np.any(y <= 0):
        raise ValueError("local_loglog_slope needs strictly positive r and y "
                          "(log is undefined at/below zero) -- mask non-positive "
                          "points out before calling.")
    log_r = np.log(r)
    log_y = np.log(y)
    slope = np.full_like(r, np.nan)
    slope[1:-1] = (log_y[2:] - log_y[:-2]) / (log_r[2:] - log_r[:-2])
    return slope


# ---------------------------------------------------------------------
# Notebook-testing convenience: same headline plot as
# analysis.plot_flux_profile_fit, for the expcore model. Self-contained
# (local matplotlib import, no dependency on analysis.py's private helpers)
# so it doesn't create a circular import (analysis.py already imports this
# module) and doesn't force a hard matplotlib dependency on the rest of
# fitting.py, which is otherwise numpy/scipy only.
# ---------------------------------------------------------------------
def plot_expcore_fit(
    boot: dict,
    r_edges=None,
    method: str = "psf",
    fit_skip_inner: int = 1,
    gamma_fixed=0.8,
    psf_r=None,
    psf_vals=None,
    psf_fwhm: float = 3.0,
    psf_beta: float = 3.0,
    p0=None,
    r_fine=None,
    logy: bool = True,
    logx: bool = False,
    xlims=None,
    rvir_kpc=None,
    show_crossover: bool = True,
    show_sphere_of_influence: bool = True,
    show_components: bool = True,
    slope_frac: float = 0.9,
    show_slope_diagnostic: bool = False,
    compare_result=None,
    figsize=(10, 5),
    title=None,
    verbose=True,
):
    """
    NOTEBOOK-TESTING convenience for halo-flux-fitting.md Part 2 (Option
    C). Deliberately mirrors analysis.plot_flux_profile_fit's call shape
    (boot['total_flux_fid'] + bootstrap 16-84 error bars, method="psf"|
    "naive", psf_r/psf_vals/psf_fwhm/psf_beta, p0, r_fine, logy, xlims,
    figsize) so it's a drop-in swap while testing -- but fits/plots
    intrinsic_profile_expcore instead of the shipped two-exponential, and
    lives HERE rather than in analysis.py on purpose (Part 2 is proposed,
    not validated -- analysis.py's plot_flux_profile_fit is untouched).

    logx : False (default) -- linear radius axis, matching
        analysis.plot_flux_profile_fit (which has no logx option at all).
        Pass True for a log-x axis -- useful here specifically because this
        model's whole point is the core-to-power-law TRANSITION around
        r_c, which is usually squeezed into a couple of pixels on a linear
        axis when bins span ~10 to ~1000+ kpc; log-x spreads the inner
        bins out so the transition shape (and how it compares to the
        two-exponential's h1/h2 break, if compare_result is given) is
        actually visible. Safe here because r_mid is always > 0 (bin
        midpoints of edges starting at r=0), so nothing has to be clipped
        the way r_fine's r=0 point already is.

    gamma_fixed : 0.8 (default) -- projected_slope_from_3d(1.8), the
        Limber-projected z~2-3 clustering slope; see
        fit_psf_aware_expcore's docstring for why floating gamma is not
        the default (r_c/gamma degeneracy, AIC/BIC preferring two-exp
        anyway). Pass None to let gamma float, or another literature value
        -- e.g. LITERATURE_GAMMA_3D["z~2-3 galaxy clustering (typical)"]
        (1.8, raw/unprojected) -- and compare fit_result['chi2'] to see
        which the data actually prefers (open question 3 in the spec).
    rvir_kpc : optional vertical reference line -- lets you see by eye
        whether the fitted r_c sits near R_vir (open question 2: should
        r_c be a free parameter, as it is here, or pinned to R_vir?).
    show_crossover : True (default) -- if the fit(s) succeeded, mark the
        radius where each model's two terms are equal
        (crossover_radius_expcore for this fit, crossover_radius_twoexp
        for compare_result if given). This is the actual "one-halo stops,
        two-halo starts" radius -- a different, usually MUCH smaller
        number than r_c itself; see crossover_radius_expcore's docstring.
    show_sphere_of_influence : True (default) -- if rvir_kpc is given,
        also mark Sorini et al. 2018's ~7x-R_vir "sphere of influence"
        scale (sphere_of_influence_kpc) -- the literature comparison
        point for r_c itself (not for the crossover radius above).
    compare_result : optional fit_result dict from Part 1's fit_psf_aware /
        fit_naive (the shipped two-exponential model, on the same bins) --
        if given, its curve is overlaid and compare_models_aic_bic is
        printed, so you can see whether Option C actually earns its extra
        parameter(s) on your real data rather than just eyeballing two
        separate plots.
    show_components : True (default) -- overplot the core term
        (A1*exp(-r/h1)) and the halo term (A2*(1+(r/r_c)^2)^(-gamma/2))
        EACH ALONE, as dashed/dash-dot curves, in addition to the solid
        combined curve already drawn. This is purely a visualization
        addition -- it draws two more lines from the SAME fitted popt,
        nothing about the fit itself changes. The point: added together,
        it's easy to mistake the transition region for "the core
        smoothly hands off to a genuine power law right around r_c" --
        seeing the two terms separately makes it visually obvious that
        one term is simply dying out while the other (which was already
        present, just subdominant, at small r) takes over, and that r_c
        is NOT the radius where the halo term itself starts looking like
        a clean power law (see effective_slope_expcore's docstring). If
        compare_result is also given, its two exponential terms are
        overlaid the same way.
    slope_frac : 0.9 (default) -- when show_components is True, also
        marks the radius (radius_of_slope_fraction) where the halo
        term's OWN local log-log slope first reaches slope_frac*gamma --
        i.e. where the halo term actually starts looking like a genuine
        power law of the fitted/fixed gamma, as opposed to r_c (which is
        only the 50%-of-the-way point, always -gamma/2 in local slope,
        by construction -- see radius_of_slope_fraction). Concretely
        answers "is r_c the same as where this becomes a REAL (r/r0)^
        -gamma power law?" -- no, this marker is, and it is typically
        several times r_c out.
    show_slope_diagnostic : False (default) -- add a second panel below
        the main plot showing the TOTAL model's local log-log slope
        (local_loglog_slope, numerical, on the combined core+halo curve)
        vs radius, with horizontal reference lines at -0.8 (Limber-
        projected z~2-3 clustering, projected_slope_from_3d(1.8)) and
        -1.8 (raw 3D clustering slope, LITERATURE_GAMMA_3D). This is the
        direct way to see whether/where the ACTUAL fitted profile's local
        slope crosses either literature value, rather than assuming the
        fitted gamma itself is the number to compare -- the core term's
        contribution can shift the total curve's local slope away from
        the halo term's own analytic slope, especially near the
        core/halo crossover radius.

    Returns (fig, ax, fit_result) -- fit_result is the dict from
    fit_psf_aware_expcore / fit_naive_expcore. When show_slope_diagnostic
    is True, ax is the tuple (ax_main, ax_slope) instead of a single Axes.
    """
    import matplotlib.pyplot as plt

    if method not in ("psf", "naive"):
        raise ValueError(f"method must be 'psf' or 'naive' (got {method!r})")
    if "total_flux_fid" not in boot:
        raise KeyError("boot does not contain total_flux_fid; re-run with "
                       "compute_side_ratio=True (the default).")

    radial_bins = np.asarray(r_edges if r_edges is not None else boot["r_edges"])
    y = np.asarray(boot["total_flux_fid"], dtype=float)
    y_lo = np.asarray(boot["total_flux_lo"], dtype=float)
    y_hi = np.asarray(boot["total_flux_hi"], dtype=float)
    sigma = ((y_hi - y) + (y - y_lo)) / 2.0

    r_fine_arr = (np.asarray(r_fine, dtype=float) if r_fine is not None
                 else default_fine_grid(radial_bins))
    r_mid = 0.5 * (radial_bins[:-1] + radial_bins[1:])

    if method == "psf":
        if psf_r is not None and psf_vals is not None:
            psf_r_use = np.asarray(psf_r, dtype=float)
            psf_vals_use = np.asarray(psf_vals, dtype=float)
        else:
            psf_r_use = np.linspace(0.0, 20.0 * psf_fwhm, 400)
            psf_vals_use = moffat_1d(psf_r_use, fwhm=psf_fwhm, beta=psf_beta)
        R = ring_convolution_matrix(r_fine_arr, radial_bins, psf_r_use, psf_vals_use)
        fit_result = fit_psf_aware_expcore(r_mid, y, sigma, R, r_fine_arr, radial_bins,
                                           gamma_fixed=gamma_fixed, p0=p0, verbose=verbose)
        fit_result["R"] = R
    else:
        fit_result = fit_naive_expcore(r_mid, radial_bins, r_fine_arr, y, sigma,
                                       fit_skip_inner=fit_skip_inner,
                                       gamma_fixed=gamma_fixed, p0=p0, verbose=verbose)

    fit_result.update({"method": method, "r_edges": radial_bins, "r_mid": r_mid,
                       "r_fine": r_fine_arr, "y": y, "y_lo": y_lo, "y_hi": y_hi, "sigma": sigma})

    if show_slope_diagnostic:
        fig, (ax, ax_slope) = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 1.7), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_slope = None

    yerr = np.vstack([np.clip(y - y_lo, 0, None), np.clip(y_hi - y, 0, None)])
    ax.errorbar(r_mid, y, yerr=yerr, fmt="o", capsize=3.5, ms=6, lw=1.5,
                color="tab:blue", label="observed (bootstrap 16-84)", zorder=5)

    if fit_result.get("success"):
        popt = fit_result["popt"]
        A1_f, h1_f, A2_f, r_c_f = popt[0], popt[1], popt[2], popt[3]
        gamma_f = fit_result["gamma"]
        params_for_curve = popt if gamma_fixed is None else np.append(popt, gamma_fixed)
        chi2_txt = (f", $\\chi^2$/dof={fit_result['chi2']/max(fit_result['dof'],1):.2f}"
                    if "chi2" in fit_result else "")
        gtag = f"gamma={fit_result['gamma']:.2f}" + (" (fixed)" if fit_result["gamma_fixed"] else "")
        total_curve = intrinsic_profile_expcore(r_fine_arr, *params_for_curve)
        ax.plot(r_fine_arr, total_curve,
                "-", color="tab:purple", lw=1.8, zorder=3,
                label=(f"expcore fit (sum)  (h1={fit_result['h1']:.2g}, "
                       f"r_c={fit_result['r_c']:.2g}, {gtag}{chi2_txt})"))
        model_binned = binned_model_from_result_expcore(fit_result, r_fine_arr, radial_bins,
                                                         fit_result.get("R"))
        if model_binned is not None:
            ax.plot(r_mid, model_binned, "D", color="tab:purple", ms=6, zorder=4,
                    label="expcore predicted bin mean")

        if show_components:
            # Each term alone, from the SAME fitted popt -- no new fitting,
            # purely a visualization of what's already been fit. Dashed for
            # the core, dash-dot for the halo, both lighter/thinner than the
            # solid combined curve so the sum still reads as "the fit."
            core_curve = A1_f * np.exp(-r_fine_arr / h1_f)
            halo_curve = A2_f * (1.0 + (r_fine_arr / r_c_f) ** 2) ** (-gamma_f / 2.0)
            ax.plot(r_fine_arr, core_curve, "--", color="tab:purple", lw=1.3,
                    alpha=0.65, zorder=2, label=f"  core alone (h1={h1_f:.2g})")
            ax.plot(r_fine_arr, halo_curve, "-.", color="tab:purple", lw=1.3,
                    alpha=0.65, zorder=2,
                    label=f"  halo alone (r_c={r_c_f:.2g}, gamma={gamma_f:.2g})")
            # r_c itself is ALWAYS the 50%-of-asymptotic-slope point, by
            # construction (effective_slope_expcore(r_c)=-gamma/2) -- it is
            # not "where this becomes a real power law." Mark the radius
            # where the halo term's OWN local slope actually gets close
            # (slope_frac, default 90%) to the asymptotic -gamma instead.
            r_frac = radius_of_slope_fraction(r_c_f, gamma_f, frac=slope_frac)
            ax.axvline(r_frac, color="tab:purple", ls=(0, (4, 1, 1, 1)), lw=1.2,
                       alpha=0.8,
                       label=(f"halo reaches {slope_frac*100:.0f}% of asymptotic "
                              f"slope (-{gamma_f:.2g}) at r={r_frac:.0f}  "
                              f"[r_c={r_c_f:.0f} itself is only the 50% point]"))
    elif verbose:
        print(f"plot_expcore_fit: fit FAILED -- {fit_result.get('reason')}")

    if compare_result is not None and compare_result.get("success"):
        cA1, ch1 = compare_result["A1"], compare_result["h1"]
        cA2, ch2 = compare_result["A2"], compare_result["h2"]
        ax.plot(r_fine_arr, intrinsic_profile(r_fine_arr, cA1, ch1, cA2, ch2),
                "--", color="tab:green", lw=1.5, zorder=2,
                label=(f"two-exp fit (sum) (h1={ch1:.2g}, h2={ch2:.2g})"))
        if show_components:
            ax.plot(r_fine_arr, cA1 * np.exp(-r_fine_arr / ch1), ":",
                    color="tab:green", lw=1.2, alpha=0.6, zorder=1,
                    label=f"  two-exp core alone (h1={ch1:.2g})")
            ax.plot(r_fine_arr, cA2 * np.exp(-r_fine_arr / ch2), "-.",
                    color="tab:green", lw=1.2, alpha=0.6, zorder=1,
                    label=f"  two-exp halo alone (h2={ch2:.2g})")
        if fit_result.get("success"):
            compare_models_aic_bic(compare_result, fit_result, label_a="two-exp",
                                   label_b="expcore", verbose=verbose)

    if ax_slope is not None and fit_result.get("success"):
        # Numerical local log-log slope of the TOTAL (core+halo) curve --
        # not just the halo term's analytic slope -- since the core term's
        # contribution can shift where the SUM actually reads as slope
        # -0.8/-1.8, especially near the core/halo crossover radius.
        pos = r_fine_arr > 0
        slope_total = local_loglog_slope(r_fine_arr[pos], total_curve[pos])
        ax_slope.plot(r_fine_arr[pos], slope_total, color="tab:purple", lw=1.6,
                      label="total fitted profile, local slope")
        ax_slope.plot(r_fine_arr[pos], effective_slope_expcore(r_fine_arr[pos], r_c_f, gamma_f),
                      color="tab:purple", lw=1.1, ls=":", alpha=0.6,
                      label="halo term alone, analytic slope")
        ax_slope.axhline(-0.8, color="0.35", ls="--", lw=1.1,
                         label="Limber-projected z~2-3 clustering (-0.8)")
        ax_slope.axhline(-1.8, color="0.35", ls=":", lw=1.1,
                         label="raw 3D clustering slope (-1.8)")
        ax_slope.axhline(-gamma_f, color="tab:purple", ls="-.", lw=1.0, alpha=0.5,
                         label=f"this fit's asymptotic slope (-{gamma_f:.2g})")
        ax_slope.set_ylabel("local slope\n" + r"$d\ln I / d\ln r$")
        ax_slope.legend(frameon=False, fontsize=7.5, loc="lower right")
        ax_slope.grid(alpha=0.15)

    if rvir_kpc is not None:
        ax.axvline(rvir_kpc, color="0.4", ls=":", lw=1.2, label=f"R_vir = {rvir_kpc:.0f}")
        if show_sphere_of_influence:
            soi = sphere_of_influence_kpc(rvir_kpc)
            ax.axvline(soi, color="0.4", ls="-.", lw=1.2,
                       label=f"~7x R_vir sphere of influence (Sorini+18) = {soi:.0f}")

    if show_crossover and fit_result.get("success"):
        params_x = fit_result["popt"] if gamma_fixed is None else np.append(fit_result["popt"], gamma_fixed)
        r_x = crossover_radius_expcore(*params_x)
        if r_x is not None:
            ax.axvline(r_x, color="tab:purple", ls=":", lw=1.3, alpha=0.8,
                       label=f"expcore term1=term2 crossover = {r_x:.0f}")
    if show_crossover and compare_result is not None and compare_result.get("success"):
        r_x2 = crossover_radius_twoexp(compare_result["A1"], compare_result["h1"],
                                       compare_result["A2"], compare_result["h2"])
        if r_x2 is not None:
            ax.axvline(r_x2, color="tab:green", ls=":", lw=1.3, alpha=0.8,
                       label=f"two-exp term1=term2 crossover = {r_x2:.0f}")

    ax.axhline(0, color="0.7", lw=0.7)
    if logy:
        pos = y[y > 0]
        if len(pos):
            ax.set_yscale("log")
            ax.set_ylim(pos.min() * 0.3, y.max() * 3)
    if logx:
        ax.set_xscale("log")
        if xlims is None:
            # default log-x lower bound: half the innermost bin's midpoint,
            # not 0 (log scale can't show r=0 anyway) -- avoids matplotlib's
            # own auto-margin pushing the left edge to something silly.
            ax.set_xlim(0.5 * r_mid[0], radial_bins[-1] * 1.2)
    if xlims is not None:
        ax.set_xlim(xlims)
    r_unit = (boot.get("unit_info") or {}).get("r_unit", "")
    unit = (boot.get("unit_info") or {}).get("y_unit", "")
    xlabel_txt = f"radius [{r_unit}]" if r_unit else "radius"
    if ax_slope is not None:
        # shared x-axis: label goes on the bottom (slope) panel only, and
        # the main panel's x tick labels are hidden so they don't repeat.
        plt.setp(ax.get_xticklabels(), visible=False)
        ax_slope.set_xlabel(xlabel_txt)
        if logx:
            ax_slope.set_xscale("log")
    else:
        ax.set_xlabel(xlabel_txt)
    ax.set_ylabel(f"Integrated Lya flux [{unit}]" if unit else "Integrated Lya flux")
    ax.set_title(title or (f"{'PSF-aware' if method == 'psf' else 'Naive'} exp-core + "
                           f"cored-power-law fit (halo-flux-fitting.md Part 2 / Option C)"))
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()

    if verbose:
        describe_fit_expcore(fit_result, label=f"{method} expcore fit vs real data")

    return fig, (ax, ax_slope) if ax_slope is not None else ax, fit_result


# =======================================================================
# 8. Part 3 (fitting half only, proposed/not-shipped) -- UV-continuum
#    radial-decline model. See specs/halo-flux-fitting.md, "Part 3 --
#    proposed extension: UV-continuum radial decline from CFHT-LS r-band
#    imaging", "Model: single exponential (default), with an optional
#    Sersic fit". This section implements ONLY the model + fit machinery;
#    the extraction half (cutouts/centroiding/annuli/background/coaddition)
#    is separate and not built here -- per the spec's own framing, that
#    piece is closer to extract.py's job than fitting.py's, even while it's
#    staged in this file for now. Additive only: nothing in Sections 1-7 is
#    touched, and analysis.py is not touched at all.
# =======================================================================
def sersic_bn(n):
    """
    Ciotti & Bertin (1999) approximation for b_n, the constant in the
    Sersic profile I(r) = I_e*exp(-b_n*[(r/r_e)^(1/n) - 1]) chosen so that
    r_e encloses half the total (2D-integrated) light. Valid to high
    accuracy for n >~ 0.36 (the regime any physically reasonable
    UV-continuum disk/bulge fit here would live in).

    b_n(n=1) = 2 - 1/3 + 4/405 + 46/25515 + 131/1148175 - 2194697/30690717750
             ~= 1.6783, matching the well-known n=1 (pure exponential)
    value -- confirms intrinsic_profile_uv_sersic(r, A, r_e, n=1) reduces
    to the same SHAPE as intrinsic_profile_uv_exp, just reparametrized
    (h_uv = r_e / b_n(1); see intrinsic_profile_uv_sersic's docstring for
    why the two aren't numerically interchangeable parameter-for-parameter).
    """
    n = np.asarray(n, dtype=float)
    return (2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n ** 2)
            + 131.0 / (1148175.0 * n ** 3)
            - 2194697.0 / (30690717750.0 * n ** 4))


def intrinsic_profile_uv_exp(r, A, h_uv):
    """
    I(r) = A*exp(-r/h_uv) -- Part 3's DEFAULT UV-continuum model: a single
    exponential, matching the expected disk-like starlight decline (as
    opposed to Part 1/2's core+halo scattered-Lya-photon models). One
    component, two parameters -- deliberately the simplest thing that
    could work, per the spec's "Physical Picture" note that the UV
    continuum is starlight from the galaxy itself, not a core+extended-halo
    structure.
    """
    r = np.asarray(r, dtype=float)
    return A * np.exp(-r / h_uv)


def intrinsic_profile_uv_sersic(r, A, r_e, n):
    """
    I(r) = A*exp(-b_n*[(r/r_e)^(1/n) - 1]) -- Part 3's OPTIONAL alternative
    to the default single exponential, offered alongside it (not replacing
    it) per the spec. n=1 recovers a pure-exponential SHAPE as a special
    case, but NOT the same parameter VALUES as intrinsic_profile_uv_exp, in
    two ways that both need converting before comparing fit results:
      (1) scale length: r_e/b_n(n), not r_e itself -- e.g. at n=1,
          h_uv-equivalent = r_e / 1.6783, not r_e.
      (2) amplitude anchor point: A here is I(r_e) (the Sersic convention --
          exponent is exactly 0 at r=r_e), whereas intrinsic_profile_uv_exp's
          A is I(0). The two are related by A_exp = A_sersic *
          exp(+b_n(n)) at n=1 (i.e. A_sersic = A_exp * exp(-r_e/h_uv)).
    Confirmed numerically: intrinsic_profile_uv_sersic(r, A*exp(-r_e/h_uv),
    r_e=h_uv*sersic_bn(1), n=1) == intrinsic_profile_uv_exp(r, A, h_uv) to
    machine precision -- the two ARE the same curve at n=1, just addressed
    by different (amplitude, scale) coordinates; don't diff A or r_e/h_uv
    directly across the two models without this conversion.
    """
    r = np.asarray(r, dtype=float)
    b_n = sersic_bn(n)
    return A * np.exp(-b_n * ((r / r_e) ** (1.0 / n) - 1.0))


def _fine_ring_flux_generic(profile_fn, r_fine, params):
    """
    Shared implementation of Part 1/2's "_fine_ring_flux" pattern
    (profile(r) treated as dF/dr, times local grid spacing dr -- the actual
    flux quantum ring_convolution_matrix's R redistributes; see
    _fine_ring_flux's docstring for why skipping the dr weighting is a real
    bug, not a style choice), parameterized over a profile CALLABLE instead
    of duplicated per model. Part 3 adds two more functional forms (plain
    exponential, Sersic); rather than hand-copying the same ~4-line loop a
    third and fourth time (as Sections 1/7 did when there was only one/two
    models to support), it's factored out once here and reused by both.
    """
    r_fine = np.asarray(r_fine, dtype=float)
    dr = np.gradient(r_fine)
    return profile_fn(r_fine, *params) * dr


def _bin_integrate_generic(profile_fn, r_fine, r_edges, params):
    """Shared implementation of Part 1/2's "_bin_integrate" pattern (proper
    bin-integrated, NO-PSF model prediction -- correct accounting for bin
    width, not a point-evaluation at the bin midpoint), parameterized over a
    profile callable for the same reason as _fine_ring_flux_generic."""
    r_fine = np.asarray(r_fine, dtype=float)
    r_edges = np.asarray(r_edges, dtype=float)
    fine_flux = _fine_ring_flux_generic(profile_fn, r_fine, params)
    n_bins = len(r_edges) - 1
    bin_idx = np.searchsorted(r_edges, r_fine, side="right") - 1
    result = np.zeros(n_bins)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    np.add.at(result, bin_idx[valid], fine_flux[valid])
    return result


def _annulus_areas(r_edges):
    """Geometric area pi*(r_out^2 - r_in^2) of each annulus. The PSF forward
    model (bin_average_psf_uv_*) redistributes real 2-D FLUX between annuli,
    so its per-bin flux is converted to a per-bin mean SURFACE BRIGHTNESS by
    dividing by annulus AREA -- NOT by radial bin width (that leaves a stray
    ~2*pi*r factor; see bin_average_psf_uv_exp)."""
    e = np.asarray(r_edges, dtype=float)
    return np.pi * (e[1:] ** 2 - e[:-1] ** 2)


def _fine_ring_flux_uv_2d(profile_fn, r_fine, params):
    """2-D ring flux of the intrinsic profile: profile(r) * 2*pi*r * dr --
    the actual FLUX carried by a thin annulus of surface brightness
    profile(r), width dr, at radius r. This is what ring_convolution_matrix's
    R (a flux-redistribution operator, weighted by the OUTPUT annulus'
    2*pi*rho) must be applied to. Differs from _fine_ring_flux_generic
    (profile(r)*dr, NO 2*pi*r) -- the circumference factor is required, not
    optional; see bin_average_psf_uv_exp's docstring."""
    r_fine = np.asarray(r_fine, dtype=float)
    dr = np.gradient(r_fine)
    return profile_fn(r_fine, *params) * (2.0 * np.pi * r_fine) * dr


def bin_average_no_psf_uv_exp(r_fine, r_edges, A, h_uv):
    """Mean intrinsic (single-exponential) UV flux in each bin, NO PSF --
    same divide-by-bin-width convention as Part 1's bin_average_no_psf, per
    the spec's explicit instruction to report the bin AVERAGE flux per
    annulus, matching Part 1's convention exactly."""
    return _bin_integrate_generic(intrinsic_profile_uv_exp, r_fine, r_edges,
                                  (A, h_uv)) / _bin_widths(r_edges)


def bin_average_psf_uv_exp(R, r_fine, r_edges, A, h_uv):
    """Mean PSF-smeared (single-exponential) UV SURFACE BRIGHTNESS in each
    bin: (R @ 2-D ring flux) / annulus AREA.

    CORRECTED (verified against a brute-force 2-D image convolution +
    azimuthal average -- agreement to <0.1% across all bins). The previous
    version computed (R @ [profile*dr]) / bin_width, which is neither the
    intrinsic 2-D flux (the input ring flux was missing its 2*pi*r
    circumference) NOR a surface brightness (dividing FLUX by radial width
    instead of annulus area leaves a stray ~2*pi*r factor that GROWS with
    radius). That radial ramp suppressed the center and pushed the model's
    peak outward -- the source of the bin1>bin0 inversion and the runaway
    chi^2 when a real (large) PSF FWHM was passed. R itself (the
    ring-convolution geometry) is unchanged; only the input flux weighting
    (now 2*pi*r*dr via _fine_ring_flux_uv_2d) and the output normalization
    (now annulus area via _annulus_areas) are fixed, both at the model level.

    R is built from whatever PSF is passed to ring_convolution_matrix -- the
    CFHT-LS PSF here, VIRUS elsewhere."""
    return (R @ _fine_ring_flux_uv_2d(intrinsic_profile_uv_exp, r_fine, (A, h_uv))
            ) / _annulus_areas(r_edges)


def bin_average_no_psf_uv_sersic(r_fine, r_edges, A, r_e, n):
    """Sersic analogue of bin_average_no_psf_uv_exp."""
    return _bin_integrate_generic(intrinsic_profile_uv_sersic, r_fine, r_edges,
                                  (A, r_e, n)) / _bin_widths(r_edges)


def bin_average_psf_uv_sersic(R, r_fine, r_edges, A, r_e, n):
    """Sersic analogue of bin_average_psf_uv_exp -- same corrected 2-D-flux
    input (2*pi*r*dr) and annulus-area normalization (see that function's
    docstring for why bin width was wrong)."""
    return (R @ _fine_ring_flux_uv_2d(intrinsic_profile_uv_sersic, r_fine, (A, r_e, n))
            ) / _annulus_areas(r_edges)


def _default_seeds_uv_exp(r, y):
    """
    Candidate (A, h_uv) seeds for the single-exponential UV model. Only one
    nonlinear parameter (h_uv), so this is much less local-minima-prone
    than Part 1/2's multi-component fits, but still tries a couple of
    h_uv variants around the endpoint-ratio estimate (same trick
    _seed_from_split uses internally) rather than trusting a single guess
    on noisy real bins.
    """
    r = np.asarray(r, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 1e-9, None)
    A0 = max(float(y[0]), 1e-9)
    if y[0] > y[-1] and y[-1] > 0:
        h0 = max((r[-1] - r[0]) / np.log(y[0] / y[-1]), 1e-3)
    else:
        h0 = max(r[-1] - r[0], 1.0)
    return [[A0, h0], [A0, h0 * 0.5], [A0, h0 * 2.0]]


def _default_seeds_uv_sersic(r, y, *, n_fixed=None, r_e_guess=None):
    """
    Candidate (A, r_e[, n]) seeds for the Sersic UV model.

    r_e_guess defaults to the plain-exponential h_uv estimate (same
    endpoint-ratio trick as _default_seeds_uv_exp) converted to a Sersic
    r_e at n=1 via r_e = h_uv * sersic_bn(1) -- a reasonable starting point
    regardless of which n is actually being seeded, since it's just meant
    to land in the right order of magnitude.

    n_fixed=None (n floats): seeds a spread of common Sersic indices --
    1.0 (pure exponential, the default UV model's own shape), 0.5 (a
    Gaussian-like flatter core), 2.0, and 4.0 (de Vaucouleurs / classical
    bulge profile) -- rather than a single arbitrary guess, since n and r_e
    can trade off against each other on sparse bins the same way Part 2's
    r_c/gamma do.

    n_fixed=<float>: n isn't a free parameter, so seeds just vary r_e
    around the guess.
    """
    r = np.asarray(r, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 1e-9, None)
    if r_e_guess is None:
        if y[0] > y[-1] and y[-1] > 0:
            h0 = max((r[-1] - r[0]) / np.log(y[0] / y[-1]), 1e-3)
        else:
            h0 = max(r[-1] - r[0], 1.0)
        r_e_guess = max(h0 * float(sersic_bn(1.0)), 1.0)
    A0 = max(float(y[0]), 1e-9)
    if n_fixed is not None:
        return [[A0, r_e_guess], [A0, r_e_guess * 0.5], [A0, r_e_guess * 2.0]]
    return [[A0, r_e_guess, 1.0], [A0, r_e_guess, 0.5],
            [A0, r_e_guess, 2.0], [A0, r_e_guess, 4.0]]


def _data_driven_bounds_uv_exp(r_fit, y_fit):
    """Same data-scaled philosophy as Part 1's _data_driven_bounds (see its
    docstring) applied to the two-parameter single-exponential model."""
    amp_max = max(50.0 * np.max(np.abs(y_fit)), 1.0)
    h_max = max(10.0 * (np.max(r_fit) - np.min(r_fit)), 10.0)
    return [0.0, 1e-3], [amp_max, h_max]


def _data_driven_bounds_uv_sersic(r_fit, y_fit, *, n_fixed=None, n_bounds=(0.3, 8.0)):
    """Same data-scaled philosophy as Part 2's _data_driven_bounds_expcore,
    for (A, r_e[, n]). n_bounds default (0.3, 8.0) brackets everything from
    a flattened-core (n<1) through exponential (n=1) to de Vaucouleurs
    (n=4) and a bit beyond, rather than pinning the box to one assumed
    shape."""
    amp_max = max(50.0 * np.max(np.abs(y_fit)), 1.0)
    re_max = max(10.0 * (np.max(r_fit) - np.min(r_fit)), 10.0)
    re_min = max(1e-3, 0.01 * np.min(r_fit))
    lower = [0.0, re_min]
    upper = [amp_max, re_max]
    if n_fixed is None:
        lower.append(n_bounds[0])
        upper.append(n_bounds[1])
    return lower, upper


def _pack_result_uv_exp(success, popt=None, pcov=None, reason=None, mask=None, extra=None):
    """Same role as Part 1's _pack_result, for the (A, h_uv) parameter set."""
    if not success:
        return {"success": False, "reason": reason}
    perr = (np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov))
            else np.full(2, np.nan))
    A, h_uv = popt
    out = {"success": True, "model": "uv_exp",
           "A": A, "h_uv": h_uv, "A_err": perr[0], "h_uv_err": perr[1],
           "popt": popt, "pcov": pcov, "mask": mask}
    if extra:
        out.update(extra)
    return out


def _pack_result_uv_sersic(success, popt=None, pcov=None, reason=None, mask=None,
                            n_fixed=None, extra=None):
    """Same role as Part 2's _pack_result_expcore, for the (A, r_e[, n])
    parameter set. When n_fixed is not None, n is still reported (as the
    value it was fixed to) with n_err=0.0 and n_fixed=True, mirroring
    _pack_result_expcore's gamma_fixed handling."""
    if not success:
        return {"success": False, "reason": reason}
    if n_fixed is None:
        perr = (np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov))
                else np.full(3, np.nan))
        A, r_e, n = popt
        n_err = perr[2]
    else:
        perr = (np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov))
                else np.full(2, np.nan))
        A, r_e = popt
        n = n_fixed
        n_err = 0.0
    out = {"success": True, "model": "uv_sersic",
           "A": A, "r_e": r_e, "n": n, "n_fixed": n_fixed is not None,
           "A_err": perr[0], "r_e_err": perr[1], "n_err": n_err,
           "popt": popt, "pcov": pcov, "mask": mask}
    if extra:
        out.update(extra)
    return out


def describe_fit_uv_exp(result, *, label="fit", truth=None):
    """Pretty-print a single-exponential UV fit result -- same role as
    Part 1's describe_fit."""
    if not result.get("success"):
        print(f"[{label}] FAILED: {result.get('reason')}")
        return
    names = ("A", "h_uv")
    print(f"[{label}] converged"
          + (f"  chi2/dof = {result['chi2']:.2f}/{result['dof']} "
             f"= {result['chi2'] / max(result['dof'], 1):.2f}"
             if "chi2" in result else ""))
    for k in names:
        line = f"    {k:>4} = {result[k]:12.4g} +/- {result[f'{k}_err']:<10.3g}"
        if truth is not None and k in truth:
            t = truth[k]
            pct = (result[k] - t) / t * 100 if t != 0 else float("nan")
            line += f"   truth={t:11.4g}   err={pct:+7.1f}%"
        print(line)


def describe_fit_uv_sersic(result, *, label="fit", truth=None):
    """Pretty-print a Sersic UV fit result -- same role as Part 2's
    describe_fit_expcore."""
    if not result.get("success"):
        print(f"[{label}] FAILED: {result.get('reason')}")
        return
    names = ("A", "r_e", "n")
    print(f"[{label}] converged"
          + (f"  chi2/dof = {result['chi2']:.2f}/{result['dof']} "
             f"= {result['chi2'] / max(result['dof'], 1):.2f}"
             if "chi2" in result else ""))
    for k in names:
        tag = " (fixed)" if (k == "n" and result.get("n_fixed")) else ""
        line = f"    {k:>4} = {result[k]:12.4g} +/- {result[f'{k}_err']:<10.3g}{tag}"
        if truth is not None and k in truth:
            t = truth[k]
            pct = (result[k] - t) / t * 100 if t != 0 else float("nan")
            line += f"   truth={t:11.4g}   err={pct:+7.1f}%"
        print(line)


# ---------------------------------------------------------------------
# Four fits: naive/PSF-aware x exponential/Sersic. Same multi-seed-then-
# lowest-chi2 strategy (_best_of_seeds, already fully generic) and the same
# amplitude-normalization trick (_amp_scale) as Parts 1/2, for the same
# numerical-conditioning reason (see _amp_scale's docstring).
# ---------------------------------------------------------------------
def fit_naive_uv_exp(r_mid, r_edges, r_fine, y, yerr, *, fit_skip_inner=1,
                     p0=None, verbose=False):
    """
    Single-exponential UV-continuum fit with NO PSF: fits the bin-AVERAGED
    model (bin_average_no_psf_uv_exp) against the data, dropping the
    innermost fit_skip_inner bin(s) (default 1, mirroring fit_naive's
    convention that the innermost annulus is the one most contaminated by
    a PSF-smeared centroid/core -- pass fit_skip_inner=0 if that turns out
    not to matter for this imaging's PSF once it's characterized).
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.arange(len(r_mid)) >= fit_skip_inner
    mask &= np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if mask.sum() < 2:
        return _pack_result_uv_exp(False, reason="fewer than 2 usable bins after fit_skip_inner")

    y_fit, e_fit = y[mask], yerr[mask]
    scale = _amp_scale(y_fit)

    def _model_masked(_, a, h_uv):
        return bin_average_no_psf_uv_exp(r_fine, r_edges, a * scale, h_uv)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds_uv_exp(r_mid, y)
    seeds = [[s[0] / scale, s[1]] for s in seeds]
    lo, up = _data_driven_bounds_uv_exp(r_mid[mask], y_fit)
    bounds = ([lo[0] / scale, lo[1]], [up[0] / scale, up[1]])

    popt, pcov = _best_of_seeds(_model_masked, r_mid[mask], y_fit / scale,
                                e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result_uv_exp(False, reason="all seeds failed to converge")

    D = np.array([scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)
    # no canonical-order step -- single component, no swap ambiguity to guard against.

    model_binned = bin_average_no_psf_uv_exp(r_fine, r_edges, *popt)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - 2
    result = _pack_result_uv_exp(True, popt, pcov, mask=mask,
                                 extra={"model_binned": model_binned, "chi2": chi2,
                                        "dof": dof, "n_fit": int(mask.sum()), "k_params": 2})
    if verbose:
        n_drop = int((~mask).sum())
        print(f"fit_naive_uv_exp: NO PSF, dropped {fit_skip_inner} inner bin(s) "
              f"({int(mask.sum())} bins fit, {n_drop} excluded)")
        describe_fit_uv_exp(result, label="naive-uv-exp")
    return result


def fit_psf_aware_uv_exp(r_mid, y, yerr, R, r_fine, r_edges, *, p0=None, verbose=False):
    """
    Single-exponential UV-continuum fit WITH the PSF forward model: trial
    (A, h_uv) -> intrinsic_profile_uv_exp on r_fine -> R @ (...) -> per-bin
    MEAN flux -> compared against the FULL binned data (no bins dropped).
    R is the SAME ring_convolution_matrix machinery as Parts 1/2 -- built
    with whatever PSF (r, vals) pair is passed in, CFHT-LS or otherwise.
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if mask.sum() < 2:
        return _pack_result_uv_exp(False, reason="fewer than 2 finite bins")

    y_fit, e_fit = y[mask], yerr[mask]
    scale = _amp_scale(y_fit)

    def _model_masked(_, a, h_uv):
        return bin_average_psf_uv_exp(R, r_fine, r_edges, a * scale, h_uv)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds_uv_exp(r_mid, y)
    seeds = [[s[0] / scale, s[1]] for s in seeds]
    lo, up = _data_driven_bounds_uv_exp(r_mid[mask], y_fit)
    bounds = ([lo[0] / scale, lo[1]], [up[0] / scale, up[1]])

    popt, pcov = _best_of_seeds(_model_masked, r_mid[mask], y_fit / scale,
                                e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result_uv_exp(False, reason="all seeds failed to converge")

    D = np.array([scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)

    model_binned = bin_average_psf_uv_exp(R, r_fine, r_edges, *popt)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - 2
    result = _pack_result_uv_exp(True, popt, pcov, mask=mask,
                                 extra={"model_binned": model_binned, "chi2": chi2,
                                        "dof": dof, "n_fit": int(mask.sum()), "k_params": 2})
    if verbose:
        print(f"fit_psf_aware_uv_exp: WITH PSF forward model, all {int(mask.sum())} "
              f"bins fit (inner bin kept)")
        describe_fit_uv_exp(result, label="psf-uv-exp")
    return result


def fit_naive_uv_sersic(r_mid, r_edges, r_fine, y, yerr, *, fit_skip_inner=1,
                        n_fixed=None, p0=None, verbose=False):
    """
    Sersic UV-continuum fit with NO PSF -- same naive/drop-inner-bin(s)
    convention as fit_naive_uv_exp, for the (A, r_e[, n]) parameter set.

    n_fixed : None (default) lets n float, bounded per
        _data_driven_bounds_uv_sersic. Pass a float (e.g. 1.0 to directly
        compare against the pure-exponential default, or 4.0 for a
        de Vaucouleurs/bulge-like test) to hold n fixed and fit only
        (A, r_e) -- mirrors fit_naive_expcore's gamma_fixed pattern
        (open question 5 in the spec: n floated vs. tested at fixed
        values).
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.arange(len(r_mid)) >= fit_skip_inner
    mask &= np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    n_free = 2 if n_fixed is not None else 3
    if mask.sum() < n_free:
        return _pack_result_uv_sersic(False, reason=f"fewer than {n_free} usable bins after fit_skip_inner")

    y_fit, e_fit = y[mask], yerr[mask]
    scale = _amp_scale(y_fit)

    if n_fixed is None:
        def _model_masked(_, a, r_e, n):
            return bin_average_no_psf_uv_sersic(r_fine, r_edges, a * scale, r_e, n)[mask] / scale
    else:
        def _model_masked(_, a, r_e):
            return bin_average_no_psf_uv_sersic(r_fine, r_edges, a * scale, r_e, n_fixed)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds_uv_sersic(r_mid, y, n_fixed=n_fixed)
    if n_fixed is None:
        seeds = [[s[0] / scale, s[1], s[2]] for s in seeds]
    else:
        seeds = [[s[0] / scale, s[1]] for s in seeds]

    lo, up = _data_driven_bounds_uv_sersic(r_mid[mask], y_fit, n_fixed=n_fixed)
    if n_fixed is None:
        bounds = ([lo[0] / scale, lo[1], lo[2]], [up[0] / scale, up[1], up[2]])
    else:
        bounds = ([lo[0] / scale, lo[1]], [up[0] / scale, up[1]])

    popt, pcov = _best_of_seeds(_model_masked, r_mid[mask], y_fit / scale,
                                e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result_uv_sersic(False, reason="all seeds failed to converge")

    D = np.array([scale, 1.0, 1.0] if n_fixed is None else [scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)

    if n_fixed is None:
        model_binned = bin_average_no_psf_uv_sersic(r_fine, r_edges, *popt)
    else:
        model_binned = bin_average_no_psf_uv_sersic(r_fine, r_edges, popt[0], popt[1], n_fixed)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - n_free
    result = _pack_result_uv_sersic(True, popt, pcov, mask=mask, n_fixed=n_fixed,
                                    extra={"model_binned": model_binned, "chi2": chi2,
                                           "dof": dof, "n_fit": int(mask.sum()), "k_params": n_free})
    if verbose:
        n_drop = int((~mask).sum())
        print(f"fit_naive_uv_sersic: NO PSF, dropped {fit_skip_inner} inner bin(s) "
              f"({int(mask.sum())} bins fit, {n_drop} excluded), n "
              f"{'fixed=' + str(n_fixed) if n_fixed is not None else 'free'}")
        describe_fit_uv_sersic(result, label="naive-uv-sersic")
    return result


def fit_psf_aware_uv_sersic(r_mid, y, yerr, R, r_fine, r_edges, *, n_fixed=None,
                            p0=None, verbose=False):
    """
    Sersic UV-continuum fit WITH the PSF forward model -- same structure as
    fit_psf_aware_uv_exp, for the (A, r_e[, n]) parameter set. See
    fit_naive_uv_sersic's docstring for n_fixed.
    """
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    mask = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    n_free = 2 if n_fixed is not None else 3
    if mask.sum() < n_free:
        return _pack_result_uv_sersic(False, reason=f"fewer than {n_free} finite bins")

    y_fit, e_fit = y[mask], yerr[mask]
    scale = _amp_scale(y_fit)

    if n_fixed is None:
        def _model_masked(_, a, r_e, n):
            return bin_average_psf_uv_sersic(R, r_fine, r_edges, a * scale, r_e, n)[mask] / scale
    else:
        def _model_masked(_, a, r_e):
            return bin_average_psf_uv_sersic(R, r_fine, r_edges, a * scale, r_e, n_fixed)[mask] / scale

    seeds = [p0] if p0 is not None else _default_seeds_uv_sersic(r_mid, y, n_fixed=n_fixed)
    if n_fixed is None:
        seeds = [[s[0] / scale, s[1], s[2]] for s in seeds]
    else:
        seeds = [[s[0] / scale, s[1]] for s in seeds]

    lo, up = _data_driven_bounds_uv_sersic(r_mid[mask], y_fit, n_fixed=n_fixed)
    if n_fixed is None:
        bounds = ([lo[0] / scale, lo[1], lo[2]], [up[0] / scale, up[1], up[2]])
    else:
        bounds = ([lo[0] / scale, lo[1]], [up[0] / scale, up[1]])

    popt, pcov = _best_of_seeds(_model_masked, r_mid[mask], y_fit / scale,
                                e_fit / scale, seeds, bounds)
    if popt is None:
        return _pack_result_uv_sersic(False, reason="all seeds failed to converge")

    D = np.array([scale, 1.0, 1.0] if n_fixed is None else [scale, 1.0])
    popt = popt * D
    if pcov is not None:
        pcov = pcov * np.outer(D, D)

    if n_fixed is None:
        model_binned = bin_average_psf_uv_sersic(R, r_fine, r_edges, *popt)
    else:
        model_binned = bin_average_psf_uv_sersic(R, r_fine, r_edges, popt[0], popt[1], n_fixed)
    chi2 = float(np.nansum(((y_fit - model_binned[mask]) / e_fit) ** 2))
    dof = int(mask.sum()) - n_free
    result = _pack_result_uv_sersic(True, popt, pcov, mask=mask, n_fixed=n_fixed,
                                    extra={"model_binned": model_binned, "chi2": chi2,
                                           "dof": dof, "n_fit": int(mask.sum()), "k_params": n_free})
    if verbose:
        print(f"fit_psf_aware_uv_sersic: WITH PSF forward model, all {int(mask.sum())} "
              f"bins fit (inner bin kept), n "
              f"{'fixed=' + str(n_fixed) if n_fixed is not None else 'free'}")
        describe_fit_uv_sersic(result, label="psf-uv-sersic")
    return result


def binned_model_from_result_uv_exp(result, r_fine, r_edges, R=None):
    """UV-exponential analogue of binned_model_from_result: the fit's
    prediction in the same per-bin AVERAGE-flux units as the data."""
    if result.get("model_binned") is not None:
        return np.asarray(result["model_binned"], dtype=float)
    if not result.get("success"):
        return None
    params = (result["A"], result["h_uv"])
    if R is not None and r_edges is not None:
        return bin_average_psf_uv_exp(R, r_fine, r_edges, *params)
    if r_edges is not None:
        return bin_average_no_psf_uv_exp(r_fine, r_edges, *params)
    return None


def binned_model_from_result_uv_sersic(result, r_fine, r_edges, R=None):
    """UV-Sersic analogue of binned_model_from_result_expcore."""
    if result.get("model_binned") is not None:
        return np.asarray(result["model_binned"], dtype=float)
    if not result.get("success"):
        return None
    params = (result["A"], result["r_e"], result["n"])
    if R is not None and r_edges is not None:
        return bin_average_psf_uv_sersic(R, r_fine, r_edges, *params)
    if r_edges is not None:
        return bin_average_no_psf_uv_sersic(r_fine, r_edges, *params)
    return None


# ---------------------------------------------------------------------
# Notebook-testing convenience: same headline plot as plot_expcore_fit
# (Section 7), for Part 3's UV-continuum model(s). Takes plain
# (r_edges, y, yerr) arrays rather than a Lya-pipeline `boot` dict, since
# Part 3's extraction half (cutouts/centroiding/annuli/coaddition) isn't
# built yet -- this lets the fit be tested against a synthetic profile or
# a hand-assembled array right now, with no change needed once the real
# coadded UV flux(r) exists.
# ---------------------------------------------------------------------
def plot_uv_fit(
    r_edges,
    y,
    yerr,
    *,
    model: str = "exp",
    method: str = "psf",
    fit_skip_inner: int = 1,
    n_fixed=None,
    psf_r=None,
    psf_vals=None,
    psf_fwhm: float = 1.0,
    psf_beta: float = 3.0,
    p0=None,
    r_fine=None,
    logy: bool = True,
    logx: bool = False,
    xlims=None,
    figsize=(9, 5),
    title=None,
    verbose=True,
):
    """
    NOTEBOOK-TESTING convenience for halo-flux-fitting.md Part 3's
    UV-continuum model.

    r_edges, y, yerr : bin edges (same units you want h_uv/r_e reported
        in) and the per-bin AVERAGE flux + its 1-sigma error -- whatever
        Part 3's eventual coadded profile produces, or a synthetic/
        hand-built stand-in for testing now.
    model : "exp" (default, intrinsic_profile_uv_exp) or "sersic"
        (intrinsic_profile_uv_sersic).
    method : "psf" (default) or "naive" -- see fit_psf_aware_uv_exp /
        fit_naive_uv_exp (or their _sersic counterparts).
    n_fixed : only used when model="sersic" -- see fit_naive_uv_sersic's
        docstring.
    psf_r, psf_vals : explicit PSF curve (e.g. an empirical CFHT-LS star
        measurement). If not given, falls back to an analytic Moffat with
        psf_fwhm/psf_beta -- psf_fwhm=1.0 here is a PLACEHOLDER, not a
        real CFHT-LS seeing value (per the spec's open question 3: try the
        image header / survey docs first, fall back to an empirical
        star-based measurement only if no documented value is
        trustworthy). Replace it once a real number is in hand.

    Returns (fig, ax, fit_result), mirroring plot_expcore_fit's return
    shape.
    """
    import matplotlib.pyplot as plt

    if method not in ("psf", "naive"):
        raise ValueError(f"method must be 'psf' or 'naive' (got {method!r})")
    if model not in ("exp", "sersic"):
        raise ValueError(f"model must be 'exp' or 'sersic' (got {model!r})")

    r_edges = np.asarray(r_edges, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_fine_arr = (np.asarray(r_fine, dtype=float) if r_fine is not None
                 else default_fine_grid(r_edges))

    if method == "psf":
        if psf_r is not None and psf_vals is not None:
            psf_r_use = np.asarray(psf_r, dtype=float)
            psf_vals_use = np.asarray(psf_vals, dtype=float)
        else:
            psf_r_use = np.linspace(0.0, 20.0 * psf_fwhm, 400)
            psf_vals_use = moffat_1d(psf_r_use, fwhm=psf_fwhm, beta=psf_beta)
        R = ring_convolution_matrix(r_fine_arr, r_edges, psf_r_use, psf_vals_use)
        if model == "exp":
            fit_result = fit_psf_aware_uv_exp(r_mid, y, yerr, R, r_fine_arr, r_edges,
                                              p0=p0, verbose=verbose)
        else:
            fit_result = fit_psf_aware_uv_sersic(r_mid, y, yerr, R, r_fine_arr, r_edges,
                                                 n_fixed=n_fixed, p0=p0, verbose=verbose)
        fit_result["R"] = R
    else:
        if model == "exp":
            fit_result = fit_naive_uv_exp(r_mid, r_edges, r_fine_arr, y, yerr,
                                          fit_skip_inner=fit_skip_inner, p0=p0, verbose=verbose)
        else:
            fit_result = fit_naive_uv_sersic(r_mid, r_edges, r_fine_arr, y, yerr,
                                             fit_skip_inner=fit_skip_inner, n_fixed=n_fixed,
                                             p0=p0, verbose=verbose)

    fit_result.update({"method": method, "model_name": model, "r_edges": r_edges,
                       "r_mid": r_mid, "r_fine": r_fine_arr, "y": y, "yerr": yerr})

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(r_mid, y, yerr=yerr, fmt="o", capsize=3.5, ms=6, lw=1.5,
                color="tab:orange", label="UV-continuum profile (input)", zorder=5)

    if fit_result.get("success"):
        popt = fit_result["popt"]
        if model == "exp":
            curve = intrinsic_profile_uv_exp(r_fine_arr, *popt)
            chi2_txt = (f", $\\chi^2$/dof={fit_result['chi2']/max(fit_result['dof'],1):.2f}"
                        if "chi2" in fit_result else "")
            ax.plot(r_fine_arr, curve, "-", color="tab:red", lw=1.8, zorder=3,
                    label=f"exp fit (h_UV={fit_result['h_uv']:.3g}{chi2_txt})")
            model_binned = binned_model_from_result_uv_exp(fit_result, r_fine_arr, r_edges,
                                                            fit_result.get("R"))
        else:
            params_for_curve = popt if n_fixed is None else np.append(popt, n_fixed)
            curve = intrinsic_profile_uv_sersic(r_fine_arr, *params_for_curve)
            chi2_txt = (f", $\\chi^2$/dof={fit_result['chi2']/max(fit_result['dof'],1):.2f}"
                        if "chi2" in fit_result else "")
            ntag = f"n={fit_result['n']:.2f}" + (" (fixed)" if fit_result["n_fixed"] else "")
            ax.plot(r_fine_arr, curve, "-", color="tab:red", lw=1.8, zorder=3,
                    label=f"Sersic fit (r_e={fit_result['r_e']:.3g}, {ntag}{chi2_txt})")
            model_binned = binned_model_from_result_uv_sersic(fit_result, r_fine_arr, r_edges,
                                                               fit_result.get("R"))
        if model_binned is not None:
            ax.plot(r_mid, model_binned, "D", color="tab:red", ms=6, zorder=4,
                    label="predicted bin mean")
    elif verbose:
        print(f"plot_uv_fit: fit FAILED -- {fit_result.get('reason')}")

    ax.axhline(0, color="0.7", lw=0.7)
    if logy:
        pos = y[y > 0]
        if len(pos):
            ax.set_yscale("log")
            ax.set_ylim(pos.min() * 0.3, y.max() * 3)
    if logx:
        ax.set_xscale("log")
        if xlims is None:
            ax.set_xlim(0.5 * r_mid[0], r_edges[-1] * 1.2)
    if xlims is not None:
        ax.set_xlim(xlims)
    ax.set_xlabel("radius")
    ax.set_ylabel("UV-continuum flux (bin mean)")
    ax.set_title(title or (f"{'PSF-aware' if method == 'psf' else 'Naive'} UV-continuum "
                           f"{'single-exponential' if model == 'exp' else 'Sersic'} fit "
                           f"(halo-flux-fitting.md Part 3)"))
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()

    if verbose:
        (describe_fit_uv_exp if model == "exp" else describe_fit_uv_sersic)(
            fit_result, label=f"{method} UV {model} fit")

    return fig, ax, fit_result
