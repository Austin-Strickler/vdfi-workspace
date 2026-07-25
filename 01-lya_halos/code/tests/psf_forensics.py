"""
=======================================================================
PSF FORENSICS -- one paste-and-run diagnostic cell
=======================================================================
Purpose: find out, visually and with the simplest possible fits, why the
Lya profile fit collapses when a wide PSF is assumed.

The PRIMARY hypothesis this tests (see notes at the bottom of this file):
the Lya PSF forward model in fitting.py still uses the OLD convention

    model_bin = (R @ [I(r) * dr]) / bin_width          <-- fitting.bin_average_psf_expcore

while the UV path was already CORRECTED (2026-07-xx, uv_profile work) to

    model_bin = (R @ [I(r) * 2*pi*r * dr]) / annulus_area   <-- fitting.bin_average_psf_uv_exp

fitting.bin_average_psf_uv_exp's own docstring says the old version was
"the source of the bin1>bin0 inversion and the runaway chi^2 when a real
(large) PSF FWHM was passed."  That is exactly the symptom here, and the
fix was never back-ported to bin_average_psf / bin_average_psf_expcore.

This file does NOT edit fitting.py.  It reimplements both conventions
locally, checks them against a brute-force 2-D image convolution (ground
truth), and then runs a simple fit ladder under each.

USAGE
-----
    # in a notebook cell, after `boot` and `stacks` exist:
    exec(open("psf_forensics.py").read())      # or paste this whole file
    res = run_psf_forensics(boot, stacks)

    # options:
    res = run_psf_forensics(boot, stacks,
                            fwhm_narrow=1.3, fwhm_wide=2.4,
                            fwhm_sweep=(1.0,1.3,1.6,1.9,2.2,2.4,2.7,3.0),
                            inner_max_kpc=60.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

try:
    from utils_lya_halo import fitting as _fitting
except Exception:                                     # already imported in the notebook
    _fitting = fitting                                # noqa: F821

try:
    from astropy.cosmology import Planck18 as _cosmo
except Exception:
    from astropy.cosmology import Planck15 as _cosmo
import astropy.units as u


# ---------------------------------------------------------------------
# 0. small geometry helpers
# ---------------------------------------------------------------------
def _areas(edges):
    e = np.asarray(edges, float)
    return np.pi * (e[1:] ** 2 - e[:-1] ** 2)


def _widths(edges):
    e = np.asarray(edges, float)
    return e[1:] - e[:-1]


def _kpc_per_arcsec(z):
    return _cosmo.angular_diameter_distance(float(z)).to(u.kpc).value / 206265.0


def make_psf_kpc(fwhm_arcsec, z_median, beta=3.0, n=1200, rmax_factor=25.0):
    """Analytic Moffat PSF on a kpc grid, normalized to unit 2-D flux."""
    fwhm_kpc = float(fwhm_arcsec) * _kpc_per_arcsec(z_median)
    r = np.linspace(0.0, rmax_factor * fwhm_kpc, n)
    v = _fitting.moffat_1d(r, fwhm=fwhm_kpc, beta=beta)
    v = _fitting.normalize_psf_flux(r, v)
    return r, v, fwhm_kpc


def psf_binned_sb(psf_r, psf_vals, r_edges):
    """
    EXACT per-annulus mean SURFACE BRIGHTNESS of a UNIT-FLUX POINT SOURCE.

        SB_i = [EE(r_i+1) - EE(r_i)] / (pi*(r_i+1^2 - r_i^2))

    Computed straight from the normalized PSF curve -- no R matrix, no
    fine grid, no model.  This is the "the fit failed / the exponential
    is zero" curve: what the data WOULD look like if every galaxy were a
    point source.  It is also the ground-truth target that any correct
    forward model must reproduce in the h -> 0 limit.
    """
    psf_r = np.asarray(psf_r, float)
    psf_vals = np.asarray(psf_vals, float)
    cum = cumulative_trapezoid(psf_vals * 2 * np.pi * psf_r, psf_r, initial=0.0)
    enc = np.interp(np.asarray(r_edges, float), psf_r, cum, left=0.0, right=cum[-1])
    return np.diff(enc) / _areas(r_edges)


# ---------------------------------------------------------------------
# 1. the two forward-model conventions
# ---------------------------------------------------------------------
def fwd_area(R, r_fine, r_edges, prof, params):
    """
    CORRECTED convention (matches fitting.bin_average_psf_uv_exp):
        input  ring flux = I(r) * 2*pi*r * dr     (real 2-D flux)
        output normalization = annulus AREA       (-> surface brightness)
    """
    r_fine = np.asarray(r_fine, float)
    f = prof(r_fine, *params) * (2.0 * np.pi * r_fine) * np.gradient(r_fine)
    return (R @ f) / _areas(r_edges)


def fwd_width(R, r_fine, r_edges, prof, params):
    """
    CURRENT Lya convention (fitting.bin_average_psf / _expcore):
        input  ring flux = I(r) * dr              (missing 2*pi*r)
        output normalization = bin WIDTH          (not area)
    """
    r_fine = np.asarray(r_fine, float)
    f = prof(r_fine, *params) * np.gradient(r_fine)
    return (R @ f) / _widths(r_edges)


# ---------------------------------------------------------------------
# 2. brute-force 2-D reference (ground truth, no R matrix involved)
# ---------------------------------------------------------------------
def brute_force_2d(prof, params, psf_r, psf_vals, r_edges_sub, pix=None, pad=2.5):
    """
    Build a 2-D image of I(r), FFT-convolve with the 2-D PSF, then take the
    plain pixel MEAN inside each annulus (= area-weighted mean surface
    brightness).  Slow-but-honest reference for the two conventions above.
    Restricted to the inner bins so the grid stays small.
    """
    r_edges_sub = np.asarray(r_edges_sub, float)
    rmax = r_edges_sub[-1]
    if pix is None:
        pix = float(np.min(np.diff(r_edges_sub))) / 5.0
    box = pad * rmax
    n = int(np.ceil(2 * box / pix))
    if n % 2 == 0:
        n += 1
    x = (np.arange(n) - n // 2) * pix
    X, Y = np.meshgrid(x, x)
    RR = np.hypot(X, Y)

    img = prof(RR, *params)
    ker = np.interp(RR, psf_r, psf_vals, left=psf_vals[0], right=0.0)
    ker = ker / (ker.sum() * pix * pix)          # unit total flux
    conv = fftconvolve(img, ker, mode="same") * pix * pix

    out = np.array([conv[(RR >= lo) & (RR < hi)].mean()
                    for lo, hi in zip(r_edges_sub[:-1], r_edges_sub[1:])])
    return out, dict(pix=pix, n=n)


# ---------------------------------------------------------------------
# 3. generic multi-seed fitter
# ---------------------------------------------------------------------
def _fit(fn, y, sigma, names, lo, hi, seeds, mask=None, label=""):
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma,
                     np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)]))
    m = np.ones(y.size, bool) if mask is None else np.asarray(mask, bool)

    def resid(p):
        return (fn(p)[m] - y[m]) / sigma[m]

    best = None
    for s in seeds:
        s = np.clip(np.asarray(s, float), np.asarray(lo) + 1e-12, np.asarray(hi) - 1e-12)
        try:
            r = least_squares(resid, s, bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return dict(label=label, success=False)

    k = len(names)
    chi2 = 2.0 * best.cost
    dof = int(m.sum()) - k
    try:
        JtJ = best.jac.T @ best.jac
        cov = np.linalg.inv(JtJ)
        perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        perr = np.full(k, np.nan)

    return dict(label=label, success=True, names=list(names),
                popt=best.x, perr=perr, chi2=chi2, dof=max(dof, 1),
                chi2_dof=chi2 / max(dof, 1), aic=chi2 + 2 * k,
                model=fn(best.x), mask=m, resid_bin=(fn(best.x) - y) / sigma)


def _exp(r, A, h):
    return A * np.exp(-np.asarray(r, float) / h)


def build_ladder(R, r_fine, r_edges, psf_sb, fwd, y):
    """
    The fit ladder, simplest first.  Each entry: (fn, names, lo, hi, seeds).

      1. psf_only   : A * PSF                    -- "the exponential is zero"
      2. psf_const  : A * PSF + C                -- is the 'halo' just a pedestal?
      3. exp        : PSF (x) A*exp(-r/h)        -- one scale length, nothing else
      4. exp_const  : PSF (x) A*exp(-r/h) + C
    """
    A0 = float(np.max(y) / max(np.max(psf_sb), 1e-30))
    Amax = 1e6 * max(A0, 1.0)
    C0 = float(np.min(np.abs(y[y != 0]))) if np.any(y != 0) else 0.0
    ymax = float(np.max(np.abs(y)))
    hmax = float(10.0 * (r_edges[-1] - r_edges[0]))

    lad = {}
    lad["psf_only"] = dict(
        fn=lambda p: p[0] * psf_sb, names=["A_ps"],
        lo=[0.0], hi=[Amax], seeds=[[A0], [0.5 * A0], [2 * A0]])
    lad["psf_const"] = dict(
        fn=lambda p: p[0] * psf_sb + p[1], names=["A_ps", "C"],
        lo=[0.0, -ymax], hi=[Amax, ymax],
        seeds=[[A0, C0], [A0, 0.0], [0.5 * A0, C0]])

    def _f_exp(p):
        return fwd(R, r_fine, r_edges, _exp, (p[0], p[1]))

    def _f_expc(p):
        return fwd(R, r_fine, r_edges, _exp, (p[0], p[1])) + p[2]

    A0e = float(np.max(y))
    lad["exp"] = dict(
        fn=_f_exp, names=["A", "h"], lo=[0.0, 1e-3], hi=[Amax, hmax],
        seeds=[[A0e, h] for h in (0.5, 2.0, 5.0, 12.0, 25.0, 50.0)])
    lad["exp_const"] = dict(
        fn=_f_expc, names=["A", "h", "C"],
        lo=[0.0, 1e-3, -ymax], hi=[Amax, hmax, ymax],
        seeds=[[A0e, h, C0] for h in (2.0, 12.0, 30.0)])
    return lad


def run_ladder(lad, y, sigma, mask=None):
    out = {}
    for name, spec in lad.items():
        out[name] = _fit(spec["fn"], y, sigma, spec["names"],
                         spec["lo"], spec["hi"], spec["seeds"],
                         mask=mask, label=name)
    return out


def print_ladder(res, header=""):
    print(f"\n  {header}")
    print(f"  {'model':<12} {'chi2':>9} {'dof':>4} {'chi2/dof':>9} {'AIC':>9}   params")
    print("  " + "-" * 78)
    for name, r in res.items():
        if not r.get("success"):
            print(f"  {name:<12}   FAILED")
            continue
        ps = "  ".join(f"{n}={v:.4g}+/-{e:.2g}"
                       for n, v, e in zip(r["names"], r["popt"], r["perr"]))
        print(f"  {name:<12} {r['chi2']:9.2f} {r['dof']:4d} {r['chi2_dof']:9.2f} "
              f"{r['aic']:9.2f}   {ps}")


# ---------------------------------------------------------------------
# 4. the orchestrator
# ---------------------------------------------------------------------
def run_psf_forensics(boot, stacks=None, *,
                      r_edges=None, z_median=None,
                      fwhm_narrow=1.3, fwhm_wide=2.4, psf_beta=3.0,
                      fwhm_sweep=(1.0, 1.3, 1.6, 1.9, 2.2, 2.4, 2.7, 3.0),
                      inner_max_kpc=60.0,
                      h_overlay=(0.1, 2.0, 5.0, 12.0, 25.0, 50.0),
                      n_phi=101, n_rho=81, r_fine=None,
                      brute_force=True, make_plots=True, verbose=True):

    # ---------------- data ----------------
    r_edges = np.asarray(r_edges if r_edges is not None
                         else boot.get("r_edges",
                                       (stacks or {}).get("r_edges")), float)
    z_med = z_median if z_median is not None else (stacks or {}).get("z_median")
    if z_med is None:
        raise ValueError("need z_median (pass stacks= or z_median=)")

    y = np.asarray(boot["total_flux_fid"], float)
    y_lo = np.asarray(boot["total_flux_lo"], float)
    y_hi = np.asarray(boot["total_flux_hi"], float)
    sigma = ((y_hi - y) + (y - y_lo)) / 2.0
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    r_fine = (np.asarray(r_fine, float) if r_fine is not None
              else _fitting.default_fine_grid(r_edges))

    inner = r_mid <= inner_max_kpc
    kpa = _kpc_per_arcsec(z_med)

    print("=" * 74)
    print("PSF FORENSICS")
    print("=" * 74)
    print(f"  z_median = {z_med:.4g}   ->  1\" = {kpa:.3f} kpc")
    print(f"  {len(r_mid)} radial bins, r_mid = "
          f"{np.array2string(r_mid, precision=1, max_line_width=100)}")
    print(f"  inner subset for the simple ladder: r <= {inner_max_kpc:g} kpc "
          f"({inner.sum()} bins)")

    # ---------------- PSFs & R matrices ----------------
    cache = {}

    def get(fwhm_as):
        if fwhm_as not in cache:
            pr, pv, fk = make_psf_kpc(fwhm_as, z_med, beta=psf_beta)
            R = _fitting.ring_convolution_matrix(r_fine, r_edges, pr, pv,
                                                 n_phi=n_phi, n_rho=n_rho)
            cache[fwhm_as] = dict(psf_r=pr, psf_vals=pv, fwhm_kpc=fk, R=R,
                                  psf_sb=psf_binned_sb(pr, pv, r_edges))
        return cache[fwhm_as]

    N = get(fwhm_narrow)
    W = get(fwhm_wide)
    print(f"\n  narrow PSF: {fwhm_narrow:.2f}\" = {N['fwhm_kpc']:.2f} kpc")
    print(f"  wide   PSF: {fwhm_wide:.2f}\" = {W['fwhm_kpc']:.2f} kpc")

    # =================================================================
    # TEST 1 -- WHICH FORWARD-MODEL CONVENTION IS CORRECT?
    # =================================================================
    print("\n" + "=" * 74)
    print("TEST 1  convention check vs brute-force 2-D convolution")
    print("=" * 74)
    conv_test = None
    if brute_force:
        sub = r_edges[r_edges <= max(inner_max_kpc, r_edges[1] * 3)]
        if sub.size < 3:
            sub = r_edges[:4]
        nsub = sub.size - 1
        h_test = 12.0
        A_test = 1.0
        for tag, D in (("narrow", N), ("wide", W)):
            ref, meta = brute_force_2d(_exp, (A_test, h_test),
                                       D["psf_r"], D["psf_vals"], sub)
            m_area = fwd_area(D["R"], r_fine, r_edges, _exp, (A_test, h_test))[:nsub]
            m_width = fwd_width(D["R"], r_fine, r_edges, _exp, (A_test, h_test))[:nsub]
            print(f"\n  --- {tag} PSF ({D['fwhm_kpc']:.1f} kpc), "
                  f"truth = exp(h={h_test} kpc), 2-D grid {meta['n']}^2 "
                  f"@ {meta['pix']:.2f} kpc/pix ---")
            print(f"  {'r_mid':>8} {'brute2D':>12} {'fwd_area':>12} {'ratio':>8}"
                  f" {'fwd_width':>12} {'ratio':>8}")
            for i in range(nsub):
                print(f"  {0.5*(sub[i]+sub[i+1]):8.1f} {ref[i]:12.5g} "
                      f"{m_area[i]:12.5g} {m_area[i]/ref[i]:8.3f} "
                      f"{m_width[i]:12.5g} {m_width[i]/ref[i]:8.3f}")
            ra = np.abs(m_area / ref - 1).max()
            rw = np.abs(m_width / ref - 1).max()
            print(f"  max |ratio-1| :   fwd_area = {ra:.3%}   "
                  f"fwd_width = {rw:.3%}   <-- fwd_width is fitting.py's current Lya model")
            conv_test = dict(ref=ref, area=m_area, width=m_width, sub=sub)

    # =================================================================
    # TEST 2 -- POINT-SOURCE / DECONVOLVABILITY CHECK (no fitting)
    # =================================================================
    print("\n" + "=" * 74)
    print("TEST 2  is the data even consistent with these PSFs?")
    print("=" * 74)
    print("  ratio = data / (point-source PSF profile, matched at bin 0).")
    print("  Convolution can only ADD width, so for ANY non-negative intrinsic")
    print("  profile this ratio must be NON-DECREASING with radius.")
    print("  A turnover => that PSF cannot have produced this data.\n")
    ratios = {}
    for tag, D in (("narrow", N), ("wide", W)):
        ps = D["psf_sb"] * (y[0] / D["psf_sb"][0])
        ratio = y / ps
        ratios[tag] = ratio
        bad = np.diff(ratio[inner]) < 0
        print(f"  {tag:>7} PSF ({D['fwhm_kpc']:5.1f} kpc): ratio = "
              + " ".join(f"{v:7.3f}" for v in ratio[inner])
              + ("   <-- NON-MONOTONIC (inconsistent)" if bad.any() else "   ok"))

    # =================================================================
    # TEST 3 -- THE SIMPLE FIT LADDER
    # =================================================================
    print("\n" + "=" * 74)
    print("TEST 3  simple fit ladder (no halo term, no fixed gamma/r_c)")
    print("=" * 74)
    ladders = {}
    for tag, D in (("narrow", N), ("wide", W)):
        for cname, fwd in (("area(FIXED)", fwd_area), ("width(current)", fwd_width)):
            lad = build_ladder(D["R"], r_fine, r_edges, D["psf_sb"], fwd, y)
            for scope, msk in (("inner-only", inner), ("all-bins", None)):
                res = run_ladder(lad, y, sigma, mask=msk)
                ladders[(tag, cname, scope)] = res
                print_ladder(res, f"[{tag} PSF {D['fwhm_kpc']:.1f} kpc | "
                                  f"{cname} | {scope}]")

    # =================================================================
    # TEST 4 -- FWHM SWEEP: h and chi2 vs assumed PSF width
    # =================================================================
    print("\n" + "=" * 74)
    print("TEST 4  sweep the assumed PSF FWHM, single-exponential fit only")
    print("=" * 74)
    sweep = {"fwhm_arcsec": [], "fwhm_kpc": []}
    for cname in ("area", "width"):
        for k in ("h", "h_err", "chi2_dof", "A"):
            sweep[f"{cname}_{k}"] = []
    for f_as in fwhm_sweep:
        D = get(f_as)
        sweep["fwhm_arcsec"].append(f_as)
        sweep["fwhm_kpc"].append(D["fwhm_kpc"])
        for cname, fwd in (("area", fwd_area), ("width", fwd_width)):
            lad = build_ladder(D["R"], r_fine, r_edges, D["psf_sb"], fwd, y)
            r = _fit(lad["exp"]["fn"], y, sigma, lad["exp"]["names"],
                     lad["exp"]["lo"], lad["exp"]["hi"], lad["exp"]["seeds"],
                     mask=inner, label="exp")
            sweep[f"{cname}_h"].append(r["popt"][1] if r["success"] else np.nan)
            sweep[f"{cname}_h_err"].append(r["perr"][1] if r["success"] else np.nan)
            sweep[f"{cname}_A"].append(r["popt"][0] if r["success"] else np.nan)
            sweep[f"{cname}_chi2_dof"].append(r["chi2_dof"] if r["success"] else np.nan)
    sweep = {k: np.asarray(v, float) for k, v in sweep.items()}

    print(f"  {'FWHM\"':>7} {'kpc':>7} | {'h[area]':>9} {'+/-':>8} {'c2/dof':>8}"
          f" | {'h[width]':>9} {'+/-':>8} {'c2/dof':>8}")
    print("  " + "-" * 76)
    for i in range(len(sweep["fwhm_arcsec"])):
        print(f"  {sweep['fwhm_arcsec'][i]:7.2f} {sweep['fwhm_kpc'][i]:7.2f} |"
              f" {sweep['area_h'][i]:9.3f} {sweep['area_h_err'][i]:8.3f}"
              f" {sweep['area_chi2_dof'][i]:8.2f} |"
              f" {sweep['width_h'][i]:9.3f} {sweep['width_h_err'][i]:8.3f}"
              f" {sweep['width_chi2_dof'][i]:8.2f}")
    for cname in ("area", "width"):
        c2 = sweep[f"{cname}_chi2_dof"]
        if np.isfinite(c2).any():
            j = int(np.nanargmin(c2))
            print(f"  best-fit PSF width under '{cname}' convention: "
                  f"FWHM = {sweep['fwhm_arcsec'][j]:.2f}\" "
                  f"({sweep['fwhm_kpc'][j]:.1f} kpc), chi2/dof = {c2[j]:.2f}, "
                  f"h = {sweep[f'{cname}_h'][j]:.2f} kpc")

    # =================================================================
    # PLOTS
    # =================================================================
    figs = {}
    if make_plots:
        # ---- Fig 1: the 4 diagnostic panels -------------------------
        fig1, axs = plt.subplots(2, 2, figsize=(13, 9))
        fig1.suptitle("PSF forensics: is the wide PSF compatible with the data?",
                      fontsize=13)

        # (a) convention check
        ax = axs[0, 0]
        if conv_test is not None:
            s = conv_test["sub"]
            rm = 0.5 * (s[:-1] + s[1:])
            ax.plot(rm, conv_test["ref"], "ko-", lw=2, ms=7,
                    label="brute-force 2-D convolution (truth)")
            ax.plot(rm, conv_test["area"], "s--", color="tab:green", ms=6,
                    label="fwd_area  (2*pi*r in, /area)  = UV fix")
            ax.plot(rm, conv_test["width"], "^--", color="tab:red", ms=6,
                    label="fwd_width (dr in, /width)  = current Lya code")
            ax.set_yscale("log")
            ax.set_xlabel("r [kpc]")
            ax.set_ylabel("mean surface brightness")
            ax.set_title(f"(a) forward-model convention, wide PSF, exp(h=12 kpc)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # (b),(c) shape overlays
        for ax, (tag, D) in zip((axs[0, 1], axs[1, 0]),
                                (("narrow", N), ("wide", W))):
            ax.errorbar(r_mid, y / y[0], yerr=sigma / y[0], fmt="o",
                        color="tab:blue", ms=6, capsize=3, zorder=6,
                        label="data (peak-normalized)")
            ps = D["psf_sb"] / D["psf_sb"][0]
            ax.plot(r_mid, ps, "k-", lw=2.5, zorder=5,
                    label="pure point source (exp = 0)")
            cols = plt.cm.viridis(np.linspace(0, 0.9, len(h_overlay)))
            for h, c in zip(h_overlay, cols):
                m = fwd_area(D["R"], r_fine, r_edges, _exp, (1.0, h))
                ax.plot(r_mid, m / m[0], "-", color=c, lw=1.4, alpha=0.9,
                        label=f"PSF (x) exp, h={h:g} kpc")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("r [kpc]")
            ax.set_ylabel("normalized to innermost bin")
            ax.set_title(f"({'b' if tag=='narrow' else 'c'}) {tag} PSF: "
                         f"{D['fwhm_kpc']:.1f} kpc -- can ANY h match the shape?")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3)

        # (d) deconvolvability
        ax = axs[1, 1]
        for tag, c in (("narrow", "tab:green"), ("wide", "tab:red")):
            ax.plot(r_mid, ratios[tag], "o-", color=c, label=f"{tag} PSF")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("data / point-source PSF")
        ax.set_title("(d) deconvolvability: must be NON-DECREASING")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig1.tight_layout()
        figs["diagnostics"] = fig1

        # ---- Fig 2: FWHM sweep --------------------------------------
        fig2, (a1, a2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        for cname, c, lab in (("area", "tab:green", "fwd_area (fixed)"),
                              ("width", "tab:red", "fwd_width (current)")):
            a1.errorbar(sweep["fwhm_arcsec"], sweep[f"{cname}_h"],
                        yerr=sweep[f"{cname}_h_err"], fmt="o-", color=c, label=lab)
            a2.plot(sweep["fwhm_arcsec"], sweep[f"{cname}_chi2_dof"],
                    "o-", color=c, label=lab)
        a1.set_yscale("log")
        a1.set_ylabel("fitted h [kpc]")
        a1.set_title("single-exponential fit vs assumed PSF FWHM (inner bins only)")
        a1.legend(fontsize=9)
        a1.grid(alpha=0.3)
        a2.axhline(1.0, color="k", ls=":", lw=1)
        a2.set_ylabel(r"$\chi^2$/dof")
        a2.set_xlabel("assumed PSF FWHM [arcsec]")
        a2.legend(fontsize=9)
        a2.grid(alpha=0.3)
        fig2.tight_layout()
        figs["sweep"] = fig2

        # ---- Fig 3: ladder fits on the data -------------------------
        fig3, axs3 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, (tag, D) in zip(axs3, (("narrow", N), ("wide", W))):
            ax.errorbar(r_mid, y, yerr=sigma, fmt="o", color="tab:blue",
                        ms=6, capsize=3, zorder=6, label="data")
            res = ladders[(tag, "area(FIXED)", "inner-only")]
            styles = dict(psf_only=("k-", "PSF only"),
                          psf_const=("k--", "PSF + const"),
                          exp=("-", "PSF (x) exp"),
                          exp_const=("--", "PSF (x) exp + const"))
            colmap = dict(psf_only="k", psf_const="0.45",
                          exp="tab:purple", exp_const="tab:orange")
            for name, r in res.items():
                if not r.get("success"):
                    continue
                ls = "--" if "const" in name else "-"
                extra = ""
                if "h" in r["names"]:
                    extra = f", h={r['popt'][r['names'].index('h')]:.2f} kpc"
                ax.plot(r_mid, r["model"], ls, color=colmap[name], lw=1.8,
                        label=f"{styles[name][1]} ($\\chi^2$/dof="
                              f"{r['chi2_dof']:.2f}{extra})")
            ax.axvline(inner_max_kpc, color="0.7", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("r [kpc]")
            ax.set_title(f"{tag} PSF ({D['fwhm_kpc']:.1f} kpc), fwd_area, inner-only fit")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        axs3[0].set_ylabel("mean Lya flux")
        fig3.tight_layout()
        figs["ladder"] = fig3

        plt.show()

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("""
  TEST 1  If fwd_area matches the brute-force 2-D convolution and fwd_width
          does NOT -- and the mismatch GROWS with PSF width -- then
          fitting.bin_average_psf / bin_average_psf_expcore are wrong in
          exactly the way fitting.bin_average_psf_uv_exp's docstring already
          documents ("runaway chi^2 when a real (large) PSF FWHM was passed").
          That alone would explain: fine at 1.3", catastrophic at 2.4",
          h1 -> 0, chi2/dof stuck near 2 even at the h1 -> 0 limit.

  TEST 2  If the ratio TURNS OVER for the wide PSF but not the narrow one,
          the 2.4" PSF is mathematically incompatible with the data
          regardless of model -- i.e. the PSF measurement is wrong.
          If it is monotonic for BOTH, the PSF is fine and the problem is
          in the forward model (Test 1) or the model family.

  TEST 3  If 'exp' under fwd_area gives a sane, stable h at BOTH PSF widths
          while fwd_width collapses only at the wide one, that is the
          smoking gun.  If 'psf_only' is not decisively worse than 'exp',
          the inner bins genuinely do not resolve anything.

  TEST 4  h(FWHM) under the corrected convention should vary smoothly and
          stay finite.  A cliff means degeneracy; the chi2/dof minimum
          tells you what PSF width the DATA prefer -- compare that to 2.4".
""")

    return dict(r_edges=r_edges, r_mid=r_mid, y=y, sigma=sigma, inner=inner,
                z_median=z_med, kpc_per_arcsec=kpa, r_fine=r_fine,
                psf=dict(narrow=N, wide=W), cache=cache,
                convention_test=conv_test, ratios=ratios,
                ladders=ladders, sweep=sweep, figs=figs)


# ---------------------------------------------------------------------
# auto-run if boot/stacks are already in the notebook namespace
# ---------------------------------------------------------------------
if "boot" in globals() and "stacks" in globals():
    psf_forensics_result = run_psf_forensics(globals()["boot"], globals()["stacks"])
