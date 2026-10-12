"""
psf_fiber_vs_naive.py -- is the fiber-aperture PSF model worth the trouble?

THE QUESTION
------------
`starpsf.fit_psf_moffat` fits `fiber_moffat` -- a Moffat pre-averaged over the
1.5"-diameter fiber face -- because every point in the measured star profile is
a fiber, not a point sample. That returns the Moffat ON THE SKY, and the fiber
averaging is then an observation operator that something downstream must apply.

Nothing downstream applies it. `fitting.ring_convolution_matrix` takes a bare
radial kernel, `fitting.moffat_1d` supplies an analytic Moffat, and
`bin_average_psf` area-averages the smeared profile over the ANNULUS. The fiber
never appears. Same for `measure.measure_psf_corrected_core_luminosity`, which
calls `fitting.moffat_encircled_energy_fraction` on an analytic Moffat.

So there are three self-consistent choices, not two:

  (A) SKY + FIBER   fit fiber_moffat -> (fwhm_sky, beta_sky), and add a fiber
                    operator to the forward model. Correct, and costs code.
  (B) EFFECTIVE     fit the ANALYTIC Moffat to the same fiber data ->
                    (fwhm_eff, beta_eff). The fit is biased high ON PURPOSE:
                    the extra width is the fiber blur, absorbed into the kernel.
                    Costs nothing. Correct only insofar as Moffat(x)tophat is
                    itself Moffat-shaped, which it is not, exactly.
  (C) SKY, NO FIBER fit fiber_moffat, then feed (fwhm_sky, beta_sky) to the
                    existing bare-Moffat forward model. This is the one real
                    mistake -- it under-smooths, and it is what you get by
                    doing the careful fit and changing nothing else.

(B) is defensible if and only if its error is small compared to the statistical
error on h1. This module measures that error instead of arguing about it.

WHAT IT DOES
------------
1. `fit_both`      -- fits the composite PSF with the fiber-averaged Moffat and
                      with the analytic Moffat, under an IDENTICAL fit setup
                      (same bins, same log-space chi2, same bounds) so the only
                      difference is the model. Gives fwhm/beta/chi2 for each.

2. `compare_kernels`-- how much the two kernels differ as curves: ratio vs r,
                      and encircled energy vs aperture (the number
                      measure_psf_corrected_core_luminosity actually consumes).

3. `h1_recovery`   -- THE DECIDING TEST. Builds a noiseless exponential halo,
                      forward-models it through the TRUE operator
                      (sky Moffat -> fiber average -> annulus average), then
                      fits it back through the existing bare-kernel pipeline
                      using each candidate kernel, and reports the bias in h1.
                      Run it over a grid of true h1; compare the bias to your
                      statistical error on h1.

   The trick that makes this exact: `starpsf.fiber_moffat` IS the composite
   kernel Moffat (x) (normalized top-hat). The aperture mean over a disk of
   radius R equals convolution with tophat/(pi R^2). So "convolve with the sky
   Moffat, then fiber-average" is a single ring convolution against
   fiber_moffat(r, fwhm_sky, beta_sky) -- no separate operator needed to
   generate truth, and no approximation in the test itself.

USAGE
-----
    import sys; sys.path.insert(0, 'code')          # from 01-lya_halos/
    from tests import psf_fiber_vs_naive as pfn

    fits = pfn.fit_both(pw, r_min=None, r_max=8.0)
    pfn.describe_both(fits)
    pfn.plot_both(pw, fits)

    imp = pfn.h1_recovery(fits, r_edges=cfg.bins_arcsec,
                          h1_true=(0.5, 1.0, 2.0, 4.0),
                          h1_stat_err=0.35)     # your 1-sigma on h1, same unit
    pfn.describe_h1_recovery(imp)

`pw` is whatever `build_piecewise_psf` (or its annulus-normalized successor)
returns: any dict with 'r_mid', 'psf', 'psf_err' in ARCSEC will do.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    from utils_lya_halo.fitting import ring_convolution_matrix
except Exception:                                   # notebook run from code/
    from utils_lya_halo import fitting as _f
    ring_convolution_matrix = _f.ring_convolution_matrix


# ---------------------------------------------------------------------
# PSF model functions
# ---------------------------------------------------------------------
def analytic_moffat(r, fwhm, beta=3.0):
    """Unit-total-flux Moffat surface brightness. Same normalization
    convention as starpsf.fiber_moffat, so the two are directly comparable
    curve-for-curve and the fitted amplitudes mean the same thing."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    fwhm = float(fwhm); beta = float(beta)
    if fwhm <= 0 or beta <= 1.0:
        return np.full(r.shape, np.nan)
    gamma = 0.5 * fwhm / np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    return (beta - 1.0) / (np.pi * gamma ** 2) * (1.0 + (r / gamma) ** 2) ** (-beta)


def fiber_moffat(r, fwhm, beta=3.0, fiber_radius=0.75, n_s=24, n_phi=48):
    """Moffat averaged over a fiber of radius `fiber_radius`, i.e. the Moffat
    convolved with a normalized top-hat. Copied from starpsf so this module
    runs standalone; if starpsf is importable its version is used instead."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    fwhm = float(fwhm); beta = float(beta)
    if fwhm <= 0 or beta <= 1.0:
        return np.full(r.shape, np.nan)
    gamma = 0.5 * fwhm / np.sqrt(2.0 ** (1.0 / beta) - 1.0)

    xs, ws = np.polynomial.legendre.leggauss(int(n_s))
    s = 0.5 * fiber_radius * (xs + 1.0)
    w_s = 0.5 * fiber_radius * ws
    phi = (np.arange(int(n_phi)) + 0.5) * (2.0 * np.pi / int(n_phi))
    w_phi = 2.0 * np.pi / int(n_phi)

    D = r[:, None, None]; S = s[None, :, None]; PH = phi[None, None, :]
    rr2 = D * D + S * S + 2.0 * D * S * np.cos(PH)
    I = (beta - 1.0) / (np.pi * gamma ** 2) * (1.0 + rr2 / gamma ** 2) ** (-beta)
    integ = np.sum(I * S * w_s[None, :, None] * w_phi, axis=(1, 2))
    return integ / (np.pi * fiber_radius ** 2)


try:                                    # prefer the package's own copy
    from utils_lya_halo.starpsf import fiber_moffat as _fm_pkg
    fiber_moffat = _fm_pkg              # noqa: F811
except Exception:
    pass


# ---------------------------------------------------------------------
# 1. Fit the composite PSF both ways, identically
# ---------------------------------------------------------------------
def _fit_one(r, y, e, kind, *, beta_fixed=None, fiber_radius=0.75):
    """Log-space chi2 fit of one model to (r, y, e). Deliberately mirrors
    starpsf.fit_psf_moffat line for line -- same seeds, same bounds, same
    absolute_sigma -- so 'fiber' vs 'analytic' is the ONLY difference and any
    chi2 gap is attributable to the model and not to fit setup."""
    ly = np.log(y)
    lsig = e / y

    model = fiber_moffat if kind == "fiber" else analytic_moffat
    kw = {"fiber_radius": fiber_radius} if kind == "fiber" else {}

    if beta_fixed is None:
        names = ["log_amp", "fwhm", "beta"]
        def f(rr, la, fw, be):
            return la + np.log(model(rr, fw, be, **kw))
        g0 = [ly[0] + 1.0, 1.5, 3.0]
        bounds = ([-np.inf, 0.2, 1.05], [np.inf, 10.0, 12.0])
    else:
        names = ["log_amp", "fwhm"]
        be0 = float(beta_fixed)
        def f(rr, la, fw):
            return la + np.log(model(rr, fw, be0, **kw))
        g0 = [ly[0] + 1.0, 1.5]
        bounds = ([-np.inf, 0.2], [np.inf, 10.0])

    popt, pcov = curve_fit(f, r, ly, p0=g0, sigma=lsig, absolute_sigma=True,
                           bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    resid = (ly - f(r, *popt)) / lsig
    chi2 = float(np.sum(resid ** 2))
    dof = int(r.size - len(popt))

    out = dict(zip(names, popt))
    out.update({
        "kind": kind, "names": names, "params": popt, "errors": perr,
        "cov": pcov, "chi2": chi2, "dof": dof,
        "chi2_dof": chi2 / dof if dof > 0 else np.nan,
        "r": r, "y": y, "yerr": e, "resid": resid,
        "model_y": np.exp(f(r, *popt)),
        "beta": float(popt[2]) if beta_fixed is None else float(beta_fixed),
        "beta_err": float(perr[2]) if beta_fixed is None else 0.0,
        "beta_fixed": beta_fixed, "fiber_radius": fiber_radius,
    })
    return out


def fit_both(pw, *, r_min=None, r_max=None, beta_fixed=None, fiber_radius=0.75):
    """
    Fit the composite PSF with the fiber-averaged Moffat and with the plain
    analytic Moffat, over exactly the same bins.

    Returns {'fiber': fit, 'analytic': fit, 'n_bins': int, 'r_range': (lo, hi)}.

    Read it like this:
      * fwhm_analytic > fwhm_fiber ALWAYS, and the gap is the fiber blur. It is
        not an error in the analytic fit; it is what "effective kernel" means.
      * chi2 comparison is only weakly informative -- both models have the same
        number of parameters and the analytic one can usually mimic the flattened
        core by widening. A LARGE chi2 gap means the fiber shape is detected in
        the data; a small gap means the two are degenerate over the fitted range
        and the choice must be settled downstream (see `h1_recovery`), not here.
    """
    r = np.asarray(pw["r_mid"], dtype=float)
    y = np.asarray(pw["psf"], dtype=float)
    e = np.asarray(pw["psf_err"], dtype=float)
    ok = np.isfinite(r) & np.isfinite(y) & np.isfinite(e) & (y > 0) & (e > 0)
    if r_min is not None:
        ok &= r >= float(r_min)
    if r_max is not None:
        ok &= r <= float(r_max)
    r, y, e = r[ok], y[ok], e[ok]
    if r.size < 3:
        raise ValueError(f"only {r.size} usable bins after the r cuts.")

    return {
        "fiber": _fit_one(r, y, e, "fiber", beta_fixed=beta_fixed,
                          fiber_radius=fiber_radius),
        "analytic": _fit_one(r, y, e, "analytic", beta_fixed=beta_fixed),
        "n_bins": int(r.size), "r_range": (float(r.min()), float(r.max())),
        "fiber_radius": float(fiber_radius),
    }


def describe_both(fits):
    f, a = fits["fiber"], fits["analytic"]
    print(f"fitted {fits['n_bins']} bins over r = "
          f"{fits['r_range'][0]:.2f} - {fits['r_range'][1]:.2f} arcsec")
    print(f"{'':22s}{'FIBER (sky Moffat)':>22s}{'ANALYTIC (effective)':>24s}")
    print(f"{'FWHM [arcsec]':22s}{f['fwhm']:>15.4f} +/-{f['errors'][1]:<5.3f}"
          f"{a['fwhm']:>17.4f} +/-{a['errors'][1]:<5.3f}")
    print(f"{'beta':22s}{f['beta']:>15.4f} +/-{f['beta_err']:<5.3f}"
          f"{a['beta']:>17.4f} +/-{a['beta_err']:<5.3f}")
    print(f"{'chi2/dof':22s}{f['chi2_dof']:>15.2f}      {a['chi2_dof']:>17.2f}")
    d = (a["fwhm"] - f["fwhm"]) / f["fwhm"]
    print(f"\nanalytic FWHM is {100*d:+.1f}% wider -- that excess IS the "
          f"{2*fits['fiber_radius']:.1f}\" fiber blur, absorbed into the kernel.")
    dchi = a["chi2"] - f["chi2"]
    print(f"delta chi2 (analytic - fiber) = {dchi:+.1f} on {f['dof']} dof: "
          + ("the fiber shape is DETECTED; the analytic form is a genuinely "
             "worse description of the star profile."
             if dchi > 2 * np.sqrt(2 * max(f['dof'], 1)) else
             "the two are degenerate over this range -- chi2 cannot decide, "
             "run h1_recovery."))


def plot_both(pw, fits, figsize=(7.6, 6.6), title=None, save_fig=False):
    """Composite PSF with both fits, residuals below. The analytic curve will
    track the data nearly as well -- that is the point, and why the decision
    cannot be made on this figure alone."""
    r = np.asarray(pw["r_mid"], float)
    y = np.asarray(pw["psf"], float)
    e = np.asarray(pw["psf_err"], float)
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1],
                                               "hspace": 0.06})
    ax.errorbar(r[ok], y[ok], yerr=e[ok], fmt="o", ms=4.5, color="k", lw=1.1,
                capsize=0, zorder=3, label="composite PSF")

    rf = np.logspace(np.log10(max(r[ok].min() * 0.7, 1e-3)),
                     np.log10(r[ok].max() * 1.4), 300)
    f, a = fits["fiber"], fits["analytic"]
    ax.plot(rf, np.exp(f["log_amp"]) * fiber_moffat(rf, f["fwhm"], f["beta"],
            fits["fiber_radius"]), "-", color="crimson", lw=1.7,
            label=rf"fiber Moffat: FWHM={f['fwhm']:.3f}$''$, $\beta$={f['beta']:.2f}"
                  rf"  ($\chi^2$/dof={f['chi2_dof']:.1f})")
    ax.plot(rf, np.exp(a["log_amp"]) * analytic_moffat(rf, a["fwhm"], a["beta"]),
            "--", color="tab:blue", lw=1.7,
            label=rf"analytic Moffat: FWHM={a['fwhm']:.3f}$''$, $\beta$={a['beta']:.2f}"
                  rf"  ($\chi^2$/dof={a['chi2_dof']:.1f})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel("normalized surface brightness")
    ax.legend(fontsize=8.5, frameon=False)
    if title:
        ax.set_title(title)

    axr.axhspan(-1, 1, color="0.85", zorder=0)
    axr.axhline(0, color="k", lw=0.8)
    axr.plot(f["r"], f["resid"], "o", ms=4, color="crimson", label="fiber")
    axr.plot(a["r"], a["resid"], "s", ms=4, color="tab:blue", mfc="none",
             label="analytic")
    axr.set_xscale("log")
    axr.set_ylabel(r"resid [$\sigma$]")
    axr.set_xlabel("radius [arcsec]")
    axr.legend(fontsize=8, frameon=False, ncol=2)
    if save_fig:
        fig.savefig(save_fig if isinstance(save_fig, str) else "psf_fiber_vs_naive.png",
                    dpi=150, bbox_inches="tight")
    return fig, (ax, axr)


# ---------------------------------------------------------------------
# 2. Kernel-level comparison
# ---------------------------------------------------------------------
def _encircled_energy(r_ap, kernel_fn, args, r_max=60.0, n=6000):
    g = np.linspace(0.0, r_max, n)
    v = kernel_fn(g, *args)
    v = np.where(np.isfinite(v), v, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * 2 * np.pi * g[1:]
                          + v[:-1] * 2 * np.pi * g[:-1]) * np.diff(g))])
    tot = cum[-1]
    return np.interp(np.atleast_1d(r_ap), g, cum) / tot


def compare_kernels(fits, r=None, apertures=(0.75, 1.0, 2.0, 3.0, 5.0),
                    verbose=True):
    """
    The two kernels as curves, and their encircled energy at a few apertures.

    EE is the number `measure.measure_psf_corrected_core_luminosity` consumes
    (via `fitting.moffat_encircled_energy_fraction`) to build the core aperture
    correction c_i = 1/EE_i. A kernel that is too wide understates EE and so
    OVERCORRECTS the core luminosity. Note that correction is a point-source
    aperture correction on real fiber data, so it wants the EFFECTIVE kernel
    too -- the same convention question, a second place it bites.
    """
    f, a = fits["fiber"], fits["analytic"]
    if r is None:
        r = np.logspace(-1.3, 1.2, 200)
    r = np.asarray(r, float)
    kf = fiber_moffat(r, f["fwhm"], f["beta"], fits["fiber_radius"])
    ka = analytic_moffat(r, a["fwhm"], a["beta"])

    ee_f = _encircled_energy(apertures, fiber_moffat,
                             (f["fwhm"], f["beta"], fits["fiber_radius"]))
    ee_a = _encircled_energy(apertures, analytic_moffat, (a["fwhm"], a["beta"]))

    out = {"r": r, "kernel_fiber": kf, "kernel_analytic": ka,
           "ratio": ka / kf, "apertures": np.asarray(apertures, float),
           "ee_fiber": ee_f, "ee_analytic": ee_a}
    if verbose:
        print("kernel ratio  analytic / true(fiber-averaged):")
        for rr in (0.5, 1.0, 2.0, 4.0, 8.0):
            print(f"   r = {rr:4.1f}\"   {np.interp(rr, r, ka/kf):6.3f}")
        print("\nencircled energy:")
        print(f"   {'aperture':>10s}{'true':>10s}{'analytic':>10s}{'ratio':>9s}")
        for ap, x, z in zip(apertures, ee_f, ee_a):
            print(f"   {ap:>9.2f}\"{x:>10.4f}{z:>10.4f}{z/x:>9.3f}")
    return out


# ---------------------------------------------------------------------
# 3. The deciding test: how much does h1 move?
# ---------------------------------------------------------------------
def _annulus_areas(r_edges):
    r = np.asarray(r_edges, float)
    return np.pi * (r[1:] ** 2 - r[:-1] ** 2)


def _forward(R, r_fine, r_edges, A, h):
    """Binned MEAN surface brightness of a single exponential A*exp(-r/h)
    after the smearing encoded in R. Matches fitting.bin_average_psf's
    convention: R acts on RING FLUX (profile * 2 pi r * dr), and the result is
    divided by the annulus area to give a per-bin average."""
    dr = np.gradient(r_fine)
    ring = A * np.exp(-r_fine / h) * 2.0 * np.pi * r_fine * dr
    return (R @ ring) / _annulus_areas(r_edges)


def _kernel_matrix(r_edges, r_fine, kernel_fn, args, psf_rmax=25.0, n_psf=500):
    psf_r = np.linspace(0.0, psf_rmax, n_psf)
    psf_v = kernel_fn(psf_r, *args)
    psf_v = np.where(np.isfinite(psf_v), psf_v, 0.0)
    return ring_convolution_matrix(r_fine, r_edges, psf_r, psf_v)


def h1_recovery(fits, r_edges, h1_true=(0.5, 1.0, 2.0, 4.0), *,
                h1_stat_err=None, kpc_per_arcsec=None,
                also_assumed=(1.3, 3.0), r_fine=None, psf_rmax=25.0,
                verbose=True):
    """
    THE TEST THAT DECIDES IT.

    Truth is generated with the operator the data actually experience:
    a point source blurred by the SKY Moffat (fwhm_fiber, beta_fiber) and then
    averaged over the fiber -- which is exactly one ring convolution against
    fiber_moffat(., fwhm_sky, beta_sky), since the aperture mean over a disk of
    radius R is convolution with tophat/(pi R^2). The result is then annulus-
    averaged into `r_edges`, matching bin_average_psf.

    That noiseless truth is then fitted back through the EXISTING pipeline
    forward model -- bare kernel, no fiber operator -- with each candidate:

      'effective' : analytic Moffat at (fwhm_analytic, beta_analytic).
                    The cheap option. Bias here is the cost of not adding a
                    fiber operator.
      'sky'       : analytic Moffat at (fwhm_fiber, beta_fiber).
                    The mistake: careful fit, unchanged pipeline. Under-smooths.
      'assumed'   : analytic Moffat at `also_assumed` (default the current
                    1.3", beta=3 fiducial). Where you are today.

    Everything is in ARCSEC. The PSF is a fixed angular blur, so the bias is
    naturally an angular statement; pass `kpc_per_arcsec` only to echo the
    numbers in kpc at one redshift.

    h1_stat_err : your 1-sigma statistical error on h1 in the SAME unit as
                  h1_true. Supply it and the verdict is printed for you.

    Returns dict with per-candidate arrays of recovered h1 and fractional bias.
    """
    r_edges = np.asarray(r_edges, dtype=float)
    if r_fine is None:
        # log-spaced: an exponential with a small h1 needs resolution at small r,
        # and ring_convolution_matrix costs ~15 s per 200 grid points.
        r_fine = np.geomspace(0.02, 1.35 * r_edges[-1], 250)
    r_fine = np.asarray(r_fine, float)

    f, a = fits["fiber"], fits["analytic"]
    R_true = _kernel_matrix(r_edges, r_fine, fiber_moffat,
                            (f["fwhm"], f["beta"], fits["fiber_radius"]),
                            psf_rmax=psf_rmax)
    cands = {
        "effective": (a["fwhm"], a["beta"]),
        "sky":       (f["fwhm"], f["beta"]),
    }
    if also_assumed is not None:
        cands["assumed"] = (float(also_assumed[0]), float(also_assumed[1]))
    R_cand = {k: _kernel_matrix(r_edges, r_fine, analytic_moffat, v,
                                psf_rmax=psf_rmax)
              for k, v in cands.items()}
    # Self-consistency control: refitting truth with the TRUE operator must
    # return h1 exactly. Anything else here is a quadrature/grid artefact, and
    # it sets the floor below which the other rows mean nothing.
    cands["_exact"] = (f["fwhm"], f["beta"])
    R_cand["_exact"] = R_true

    h1_true = np.atleast_1d(np.asarray(h1_true, float))
    rec = {k: np.full(h1_true.size, np.nan) for k in cands}

    for i, h in enumerate(h1_true):
        data = _forward(R_true, r_fine, r_edges, 1.0, h)
        good = np.isfinite(data) & (data > 0)
        for k in cands:
            def m(_, A, hh, _R=R_cand[k]):
                return _forward(_R, r_fine, r_edges, A, hh)[good]
            try:
                p, _ = curve_fit(m, np.arange(good.sum()), data[good],
                                 p0=[1.0, h], bounds=([1e-8, 1e-3], [1e8, 1e3]),
                                 maxfev=20000)
                rec[k][i] = p[1]
            except Exception:
                pass

    out = {"h1_true": h1_true, "recovered": rec, "params": cands,
           "bias": {k: rec[k] - h1_true for k in cands},
           "frac_bias": {k: (rec[k] - h1_true) / h1_true for k in cands},
           "r_edges": r_edges, "h1_stat_err": h1_stat_err,
           "kpc_per_arcsec": kpc_per_arcsec}
    if verbose:
        describe_h1_recovery(out)
    return out


def describe_h1_recovery(imp):
    h = imp["h1_true"]
    ks = [k for k in imp["params"] if k != "_exact"]
    print("h1 recovered through the bare-kernel pipeline, against truth "
          "generated with sky Moffat + fiber aperture")
    print("(all arcsec; noiseless -- this is pure kernel bias, no fit scatter)\n")
    ctl = imp["frac_bias"].get("_exact")
    if ctl is not None:
        print(f"   control (refit with the TRUE operator): "
              f"max |bias| = {100*np.nanmax(np.abs(ctl)):.2f}%  "
              "<- quadrature floor; ignore anything smaller below.\n")
    print("   " + f"{'h1_true':>9s}" + "".join(f"{k:>18s}" for k in ks))
    for i, hv in enumerate(h):
        row = f"   {hv:>9.3f}"
        for k in ks:
            row += (f"{imp['recovered'][k][i]:.3f} "
                    f"({100*imp['frac_bias'][k][i]:+.1f}%)").rjust(18)
        print(row)
    print()
    for k in ks:
        fb = imp["frac_bias"][k]
        fw, be = imp["params"][k]
        print(f"  {k:>10s}  (FWHM={fw:.3f}\", beta={be:.2f}): "
              f"bias {100*np.nanmin(fb):+.1f}% to {100*np.nanmax(fb):+.1f}%")
    se = imp.get("h1_stat_err")
    if se:
        print(f"\n  against a statistical error of {se:g} on h1:")
        for k in ks:
            worst = np.nanmax(np.abs(imp["bias"][k]))
            print(f"    {k:>10s}: worst bias = {worst:.3f} = "
                  f"{worst/se:.2f} sigma  -> "
                  + ("NEGLIGIBLE, skip the fiber operator." if worst / se < 0.3
                     else "MARGINAL, quote it as a systematic."
                     if worst / se < 1.0 else
                     "NOT ACCEPTABLE, the fiber operator has to go in."))
    s = imp.get("kpc_per_arcsec")
    if s:
        print(f"\n  at {s:g} kpc/arcsec: h1_true {h[0]*s:.1f}-{h[-1]*s:.1f} kpc, "
              f"'effective' bias "
              f"{np.nanmin(imp['bias']['effective'])*s:+.2f} to "
              f"{np.nanmax(imp['bias']['effective'])*s:+.2f} kpc")


def plot_h1_recovery(imp, figsize=(7.2, 4.6), save_fig=False):
    fig, ax = plt.subplots(figsize=figsize)
    style = {"effective": ("o-", "tab:blue"), "sky": ("s--", "crimson"),
             "assumed": ("^:", "0.45")}
    for k in imp["params"]:
        if k == "_exact":
            continue
        st, c = style.get(k, ("o-", None))
        fw, be = imp["params"][k]
        ax.plot(imp["h1_true"], 100 * imp["frac_bias"][k], st, color=c, ms=6,
                lw=1.6, label=f"{k}  (FWHM={fw:.2f}\", "r"$\beta$="f"{be:.2f})")
    ax.axhline(0, color="k", lw=0.9)
    se = imp.get("h1_stat_err")
    if se:
        band = 100 * se / imp["h1_true"]
        ax.fill_between(imp["h1_true"], -band, band, color="0.85", zorder=0,
                        label=r"$\pm 1\sigma_{\rm stat}$")
    ax.set_xlabel(r"true $h_1$ [arcsec]")
    ax.set_ylabel(r"$h_1$ bias [%]")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("kernel-convention bias in $h_1$ (noiseless)")
    if save_fig:
        fig.savefig(save_fig if isinstance(save_fig, str) else "h1_kernel_bias.png",
                    dpi=150, bbox_inches="tight")
    return fig, ax
