"""
psf_exponential_recovery.py -- standalone Phase-1 testbed (see
specs/psf-exponential-fit-recovery.md in the specs/ folder).

Generates synthetic two-component exponential flux(r) profiles [a steep
core + a shallow halo, I(r) = A1*exp(-r/h1) + A2*exp(-r/h2)], PSF-smears
them via a closed-form 1D ring-convolution (no pixel grid -- the real
pipeline's data product is already collapsed to one radius axis, so this
stays 1D throughout), bins them into discrete radial bins with realistic
ABSOLUTE per-bin noise, and recovers h1/h2 two ways for comparison:

  * fit_naive     -- no PSF correction, drops the PSF-correlated inner
                     bin(s), fits the same two-exponential functional form
                     directly to the rest.
  * fit_psf_aware -- forward-models the PSF (fine intrinsic profile ->
                     ring-convolution matrix R -> binned prediction),
                     fits against the FULL binned data including the
                     inner bin.

Not part of the utils_lya_halo package -- this is a methods/prototyping
script, meant to be imported from a notebook. Run directly
(`python psf_exponential_recovery.py`) for a quick validation + smoke test.

All of the actual PSF model / intrinsic-profile / ring-convolution / fit
machinery now lives in utils_lya_halo/fitting.py (split out so
analysis.py's real-data fits and this synthetic-recovery testbed always run
the exact same code -- see fitting.py's module docstring). This file keeps
only what's specific to SYNTHETIC recovery testing: generating fake noisy
observations from a known truth, running many-draw Monte Carlo recovery,
and the recovery-diagnostic plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils_lya_halo.fitting import (
    # PSF model
    moffat_1d, normalize_psf_flux,
    # intrinsic profile / grid
    intrinsic_profile, default_fine_grid,
    # ring convolution
    ring_convolution_matrix, validate_ring_convolution,
    # bin-average model
    bin_average_no_psf, bin_average_psf,
    # fits + reporting
    fit_naive, fit_psf_aware, describe_fit, binned_model_from_result,
)

# np.trapezoid only exists in NumPy >= 2.0 (it's np.trapz on older
# installs, e.g. HPC/cluster environments that lag behind pip). Use
# whichever is available so this runs unmodified on either.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------
# Reference values from Austin's real 10-bin flux profile (flux S/N
# column), used as a realistic default noise SHAPE for demos -- not a
# claim that these exact numbers are physically correct for any given
# synthetic truth, just a template so synthetic S/N looks like real data.
# ---------------------------------------------------------------------
EXAMPLE_R_EDGES_KPC = np.array(
    [0, 10, 20, 30, 50, 80, 140, 300, 600, 1000, 2000], dtype=float)
EXAMPLE_FLUX_SN = np.array(
    [10.6, 13.6, 10.4, 8.9, 4.0, 3.4, 4.5, 6.3, 5.7, 4.7])


# =======================================================================
# Fake data: absolute per-bin noise, fed into the intrinsic scatter
# =======================================================================
def sigma_from_target_sn(reference_params, r_edges, r_fine, psf_r, psf_vals,
                         target_sn_per_bin, R=None, verbose=False):
    """
    Convert a target flux-S/N SHAPE (e.g. EXAMPLE_FLUX_SN, Austin's real
    per-bin flux S/N) into an ABSOLUTE noise_sigma_per_bin array, using
    ONE reference truth model to set the flux scale.

    Call this ONCE per (reference truth, PSF, S/N shape) to get a fixed
    sigma array, then reuse that SAME array across every
    make_fake_observation() draw / Monte Carlo run. The noise floor is
    absolute and should not silently rescale itself if you later change
    (A1, h1, A2, h2) to test a different truth against the same noise.

    Returns (sigma_per_bin, R) -- R is handed back so the caller can reuse
    it instead of recomputing.
    """
    if R is None:
        R = ring_convolution_matrix(r_fine, r_edges, psf_r, psf_vals)
    A1, h1, A2, h2 = reference_params
    # AVERAGE flux per bin (matches the biweight-averaged real data), not summed.
    model_binned = bin_average_psf(R, r_fine, r_edges, A1, h1, A2, h2)
    target_sn = np.asarray(target_sn_per_bin, dtype=float)
    sigma = np.abs(model_binned) / target_sn

    if verbose:
        r_mid = 0.5 * (np.asarray(r_edges, float)[:-1] + np.asarray(r_edges, float)[1:])
        print("sigma_from_target_sn: absolute per-bin noise floor "
              "(fixed; reused for every draw)")
        print(f"  reference truth: A1={A1:.3g} h1={h1:.3g} A2={A2:.3g} h2={h2:.3g}")
        print(f"  {'r_mid':>7} {'mean flux (noise-free)':>24} {'target S/N':>11} "
              f"{'sigma':>12}")
        for i in range(len(sigma)):
            print(f"  {r_mid[i]:7.1f} {model_binned[i]:24.4e} {target_sn[i]:11.1f} "
                  f"{sigma[i]:12.4e}")
    return sigma, R


def make_fake_observation(truth_params, r_edges, r_fine, psf_r, psf_vals,
                          *, noise_sigma_per_bin, R=None, seed=None,
                          verbose=False):
    """
    truth_params : (A1, h1, A2, h2)
    noise_sigma_per_bin : ABSOLUTE per-bin flux noise sigma (same units as
        the model amplitude), length == len(r_edges)-1. Not derived from
        S/N at call time -- pass a fixed array (see sigma_from_target_sn)
        so the noise floor doesn't change if the truth does.

    Returns a boot-shaped dict: {"r_edges", "total_flux_fid",
    "total_flux_lo", "total_flux_hi", ...} -- a drop-in stand-in for a
    real `boot` dict (fid -/+ 1 sigma in place of the real bootstrap
    16/84 band).
    """
    rng = np.random.default_rng(seed)
    if R is None:
        R = ring_convolution_matrix(r_fine, r_edges, psf_r, psf_vals)

    A1, h1, A2, h2 = truth_params
    # AVERAGE flux per bin (matches the biweight-averaged real data), not summed.
    model_binned = bin_average_psf(R, r_fine, r_edges, A1, h1, A2, h2)

    sigma = np.asarray(noise_sigma_per_bin, dtype=float)
    if sigma.shape != model_binned.shape:
        raise ValueError(
            f"noise_sigma_per_bin length {sigma.shape} != n_bins "
            f"{model_binned.shape}")

    noise = rng.normal(0.0, sigma)
    noisy = model_binned + noise

    if verbose:
        r_mid = 0.5 * (np.asarray(r_edges, float)[:-1] + np.asarray(r_edges, float)[1:])
        realized_sn = np.abs(noisy) / np.where(sigma > 0, sigma, np.nan)
        print(f"make_fake_observation (seed={seed}): one noisy realization")
        print(f"  {'r_mid':>7} {'noise-free':>13} {'+noise':>12} {'= observed':>13} "
              f"{'sigma':>12} {'realized S/N':>12}")
        for i in range(len(noisy)):
            print(f"  {r_mid[i]:7.1f} {model_binned[i]:13.4e} {noise[i]:+12.3e} "
                  f"{noisy[i]:13.4e} {sigma[i]:12.4e} {realized_sn[i]:12.1f}")
    return {
        "r_edges": np.asarray(r_edges, dtype=float),
        "total_flux_fid": noisy,
        "total_flux_lo": noisy - sigma,
        "total_flux_hi": noisy + sigma,
        "truth_model_binned": model_binned,   # noise-free, for diagnostics
        "R": R,
    }


# =======================================================================
# Diagnostics
# =======================================================================
def plot_recovery_case(r_fine, truth_params, r_mid, y, yerr, naive_result,
                       psf_result, *, r_edges=None, R=None, figsize=(8.5, 5.2),
                       title=None, verbose=True):
    """
    Overlay the observed MEAN-flux profile against each fit -- everything on a
    single axis, all in mean-flux-per-bin units, now that the data is treated
    as a per-bin average (biweight) rather than a sum.

    For each fit we draw two things:
      * the smooth recovered intrinsic profile I(r) (a line on r_fine) -- for
        the PSF-aware fit this is the DEconvolved profile, which correctly rises
        ABOVE the smeared data in the core; for the naive fit it is just the fit;
      * the fit's PREDICTED per-bin mean (markers at bin midpoints) -- this is
        what is actually compared to the data, so residuals are read off these.

    Pass `r_edges` (needed for the naive prediction and the truth curve) and `R`
    (needed for the PSF-aware prediction). verbose=True prints a per-bin table of
    data vs each model prediction and the residual in sigma.
    """
    r_fine = np.asarray(r_fine, dtype=float)
    r_mid = np.asarray(r_mid, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # --- data (per-bin mean flux) ---
    ax.errorbar(r_mid, y, yerr=yerr, fmt="o", ms=6, capsize=3, color="tab:blue",
                label="observed (per-bin mean flux)", zorder=5)

    # --- intrinsic truth profile (same units as the data now) ---
    ax.plot(r_fine, intrinsic_profile(r_fine, *truth_params), "--", color="0.4",
            lw=1.3, label="intrinsic truth I(r)", zorder=2)

    naive_pred = psf_pred = None

    # --- naive fit: recovered profile (line) + predicted per-bin mean (markers) ---
    if naive_result.get("success"):
        mask = naive_result["mask"]
        if np.any(~mask):
            ax.scatter(r_mid[~mask], y[~mask], s=110, facecolors="none",
                       edgecolors="tab:blue", linewidths=1.4, zorder=6,
                       label="dropped by naive fit")
        p = (naive_result["A1"], naive_result["h1"], naive_result["A2"], naive_result["h2"])
        ax.plot(r_fine, intrinsic_profile(r_fine, *p), "-", color="tab:orange",
                lw=1.6, zorder=3,
                label=(f"naive fit I(r)  (h1={p[1]:.1f}, h2={p[3]:.1f}"
                       + (f", chi2/dof={naive_result['chi2']/max(naive_result['dof'],1):.1f}"
                          if 'chi2' in naive_result else "") + ")"))
        naive_pred = binned_model_from_result(naive_result, r_fine, r_edges, R)
        if naive_pred is not None:
            ax.plot(r_mid, naive_pred, "s", color="tab:orange", ms=7, mfc="none",
                    zorder=4, label="naive predicted bin mean")

    # --- PSF-aware fit: recovered (deconvolved) profile + predicted bin mean ---
    if psf_result.get("success"):
        p = (psf_result["A1"], psf_result["h1"], psf_result["A2"], psf_result["h2"])
        ax.plot(r_fine, intrinsic_profile(r_fine, *p), "-", color="tab:green",
                lw=1.6, zorder=3,
                label=(f"PSF-aware fit I(r)  (h1={p[1]:.1f}, h2={p[3]:.1f}"
                       + (f", chi2/dof={psf_result['chi2']/max(psf_result['dof'],1):.1f}"
                          if 'chi2' in psf_result else "") + ")"))
        psf_pred = binned_model_from_result(psf_result, r_fine, r_edges, R)
        if psf_pred is not None:
            ax.plot(r_mid, psf_pred, "^", color="tab:green", ms=7, zorder=4,
                    label="PSF-aware predicted bin mean (smeared)")

    ax.set_yscale("log")
    pos = y[y > 0]
    if len(pos):
        ax.set_ylim(pos.min() * 0.3,
                    max(y.max(), intrinsic_profile(r_mid.min(), *truth_params)) * 3)
    ax.set_xlabel("radius (kpc)")
    ax.set_ylabel("mean flux per bin")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.15)
    plt.tight_layout()

    # --- per-bin readout ---
    if verbose and r_edges is not None:
        print(f"plot_recovery_case: per-bin comparison  (title={title!r})")
        header = f"  {'r_mid':>7} {'observed':>13} {'sigma':>12} {'truth(mean)':>13}"
        if naive_pred is not None:
            header += f" {'naive':>13} {'res/sig':>8}"
        if psf_pred is not None:
            header += f" {'psf':>13} {'res/sig':>8}"
        print(header)
        tb = bin_average_no_psf(r_fine, r_edges, *truth_params)
        for i in range(len(r_mid)):
            row = f"  {r_mid[i]:7.1f} {y[i]:13.4e} {yerr[i]:12.3e} {tb[i]:13.4e}"
            if naive_pred is not None:
                rn = (y[i] - naive_pred[i]) / yerr[i] if yerr[i] > 0 else np.nan
                row += f" {naive_pred[i]:13.4e} {rn:+8.2f}"
            if psf_pred is not None:
                rp = (y[i] - psf_pred[i]) / yerr[i] if yerr[i] > 0 else np.nan
                row += f" {psf_pred[i]:13.4e} {rp:+8.2f}"
            print(row)
        print("  (res/sig = (observed - model)/sigma; |res/sig| <~ 1-2 per bin "
              "is a good fit. Everything is mean flux per bin -- no more sums.)")

    return fig, ax


def _mc_summary(out):
    """Median/std and % bias of recovered params vs truth, per method -- the
    numbers behind the recovery histograms, printed as a small table."""
    truth = dict(zip(("A1", "h1", "A2", "h2"), out["truth_params"]))
    print(f"monte_carlo_recovery summary ({out['n_draws']} draws):")
    print(f"  {'param':>5} {'truth':>10} | {'method':>5} {'n':>4} {'median':>10} "
          f"{'std':>10} {'bias%':>8}")
    for param in ("h1", "h2", "A1", "A2"):
        t = truth[param]
        for method in ("naive", "psf"):
            v = out[method][param]
            if len(v):
                med, sd = np.median(v), np.std(v)
                bias = (med - t) / t * 100 if t != 0 else float("nan")
                print(f"  {param:>5} {t:10.3g} | {method:>5} {len(v):4d} "
                      f"{med:10.3g} {sd:10.3g} {bias:+8.1f}")
    print("  -> the PSF-aware rows should sit closer to truth (smaller |bias%|) "
          "and have smaller std than the naive rows.")


def monte_carlo_recovery(truth_params, r_edges, r_fine, psf_r, psf_vals, *,
                         noise_sigma_per_bin, n_draws=200, seed=None,
                         fit_skip_inner=1, verbose=False):
    """Repeat make_fake_observation + both fits n_draws times at a fixed
    truth/PSF/noise; collect recovered h1/h2 (and A1/A2) arrays per
    method for the recovery histograms below. verbose=True prints a progress
    tick and a median/std/bias summary at the end."""
    rng = np.random.default_rng(seed)
    R = ring_convolution_matrix(r_fine, r_edges, psf_r, psf_vals)
    r_edges = np.asarray(r_edges, dtype=float)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    sigma = np.asarray(noise_sigma_per_bin, dtype=float)

    out = {"naive": {"A1": [], "h1": [], "A2": [], "h2": []},
           "psf": {"A1": [], "h1": [], "A2": [], "h2": []}}

    if verbose:
        print(f"monte_carlo_recovery: running {n_draws} draws "
              f"(truth A1={truth_params[0]:.3g} h1={truth_params[1]:.3g} "
              f"A2={truth_params[2]:.3g} h2={truth_params[3]:.3g}) ...")
    for _draw in range(n_draws):
        if verbose and n_draws >= 20 and (_draw + 1) % max(1, n_draws // 10) == 0:
            print(f"    ... {_draw + 1}/{n_draws} draws done")
        draw_seed = int(rng.integers(0, 2**32 - 1))
        obs = make_fake_observation(truth_params, r_edges, r_fine, psf_r, psf_vals,
                                    noise_sigma_per_bin=sigma, R=R, seed=draw_seed)
        y = obs["total_flux_fid"]

        nres = fit_naive(r_mid, r_edges, r_fine, y, sigma, fit_skip_inner=fit_skip_inner)
        pres = fit_psf_aware(r_mid, y, sigma, R, r_fine, r_edges)

        if nres.get("success"):
            for k in ("A1", "h1", "A2", "h2"):
                out["naive"][k].append(nres[k])
        if pres.get("success"):
            for k in ("A1", "h1", "A2", "h2"):
                out["psf"][k].append(pres[k])

    for method in ("naive", "psf"):
        for k in out[method]:
            out[method][k] = np.array(out[method][k])

    out["truth_params"] = truth_params
    out["n_draws"] = n_draws
    out["R"] = R
    out["r_mid"] = r_mid
    if verbose:
        n_ok_naive = len(out["naive"]["h1"])
        n_ok_psf = len(out["psf"]["h1"])
        print(f"  done. converged: naive {n_ok_naive}/{n_draws}, "
              f"psf {n_ok_psf}/{n_draws}")
        _mc_summary(out)
    return out


def plot_recovery_histograms(mc_result, param="h1", *, figsize=(6.5, 4.5), bins=25):
    """Double histogram (per Austin's ask, not a table): naive (orange)
    vs PSF-aware (blue) recovered values over the Monte Carlo draws, with
    a vertical dashed line at the injected truth. Run once per parameter
    (h1, h2, ...) you want to check."""
    truth_params = mc_result["truth_params"]
    truth_val = dict(zip(("A1", "h1", "A2", "h2"), truth_params))[param]

    fig, ax = plt.subplots(figsize=figsize)
    naive_vals = mc_result["naive"][param]
    psf_vals_ = mc_result["psf"][param]

    def _stats(v):
        return (np.median(v), np.std(v),
                (np.median(v) - truth_val) / truth_val * 100 if truth_val else np.nan)

    if len(naive_vals):
        med, sd, bias = _stats(naive_vals)
        ax.hist(naive_vals, bins=bins, alpha=0.55, color="tab:orange",
               label=f"naive: med={med:.2g}, std={sd:.2g}, bias={bias:+.0f}% (n={len(naive_vals)})")
        ax.axvline(med, color="tab:orange", ls=":", lw=1.2)
    if len(psf_vals_):
        med, sd, bias = _stats(psf_vals_)
        ax.hist(psf_vals_, bins=bins, alpha=0.55, color="tab:blue",
               label=f"PSF-aware: med={med:.2g}, std={sd:.2g}, bias={bias:+.0f}% (n={len(psf_vals_)})")
        ax.axvline(med, color="tab:blue", ls=":", lw=1.2)
    ax.axvline(truth_val, color="k", ls="--", lw=1.5,
              label=f"injected truth ({param}={truth_val:.2f})")
    ax.set_xlabel(f"recovered {param}")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Recovery of {param} over {mc_result['n_draws']} draws "
                 f"(dotted = each method's median)")
    plt.tight_layout()

    print(f"plot_recovery_histograms [{param}]: injected truth = {truth_val:.4g}")
    for name, v in (("naive", naive_vals), ("PSF-aware", psf_vals_)):
        if len(v):
            med, sd, bias = _stats(v)
            print(f"  {name:>9}: median={med:.4g}  std={sd:.4g}  "
                  f"bias={bias:+.1f}%  (n={len(v)})")
    return fig, ax


# =======================================================================
# Smoke test / quick demo (mirrors test_framework.py's `python file.py` convention)
# =======================================================================
if __name__ == "__main__":
    print("Validating ring-convolution quadrature against scipy.integrate.quad...")
    validate_ring_convolution()

    print("\nRunning a quick smoke-test recovery case...")
    r_edges = EXAMPLE_R_EDGES_KPC
    r_fine = default_fine_grid(r_edges)
    psf_r = np.linspace(0, 60, 400)
    psf_vals = moffat_1d(psf_r, fwhm=3.0, beta=3.0)

    truth = (200.0, 8.0, 60.0, 80.0)   # A1, h1, A2, h2
    R = ring_convolution_matrix(r_fine, r_edges, psf_r, psf_vals)
    sigma, R = sigma_from_target_sn(truth, r_edges, r_fine, psf_r, psf_vals,
                                    EXAMPLE_FLUX_SN, R=R)
    obs = make_fake_observation(truth, r_edges, r_fine, psf_r, psf_vals,
                                noise_sigma_per_bin=sigma, R=R, seed=0)

    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    y = obs["total_flux_fid"]
    naive = fit_naive(r_mid, r_edges, r_fine, y, sigma)
    psf_fit = fit_psf_aware(r_mid, y, sigma, R, r_fine, r_edges)

    print("truth:      A1=%.1f h1=%.2f A2=%.1f h2=%.2f" % truth)
    if naive["success"]:
        print("naive fit:  A1=%.1f h1=%.2f A2=%.1f h2=%.2f"
              % (naive["A1"], naive["h1"], naive["A2"], naive["h2"]))
    else:
        print("naive fit failed:", naive["reason"])
    if psf_fit["success"]:
        print("PSF fit:    A1=%.1f h1=%.2f A2=%.1f h2=%.2f"
              % (psf_fit["A1"], psf_fit["h1"], psf_fit["A2"], psf_fit["h2"]))
    else:
        print("PSF fit failed:", psf_fit["reason"])
