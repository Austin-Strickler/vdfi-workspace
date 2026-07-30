"""Injection-recovery test for utils_lya_halo/lis.py.

Builds a synthetic rest-frame core stack with KNOWN injected velocities and
checks that lis.py recovers them, then probes the three quantitative claims the
spec makes:
  A. C II* 1335.71 emission biases a single-Gaussian centroid BLUEWARD, and
     ncomp=2 recovers the truth.
  B. On a blue-winged asymmetric trough the trough MINIMUM sits REDWARD of the
     Gaussian centroid (so v_min - v_gauss > 0).
  C. Si II 1526 is clean: single-Gaussian recovers truth without help.
"""
import sys
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_lya_halo import lis  # noqa: E402

C = 299792.458
Z = 2.30
LSF_OBS_A = 5.6
SIG_LSF = (LSF_OBS_A / (1 + Z)) / 2.35482          # rest-frame sigma, A
GRID = np.arange(1100.0, 1700.0 + 1e-9, 0.2)       # config rest grid


def gauss(w, mu, sig, amp):
    return amp * np.exp(-0.5 * ((w - mu) / sig) ** 2)


def build_stack(dv_true, *, sig_intr_kms=200.0, cont_slope=3e-4, cont_level=1.0,
                noise=0.0, seed=1, lines=None, blue_wing=None, fine_struct=True):
    """Synthetic normalized-ish stack in arbitrary flux units."""
    rng = np.random.default_rng(seed)
    flux = cont_level * (1.0 + cont_slope * (GRID - 1400.0))
    shift = 1.0 + dv_true / C

    spec_defs = lines if lines is not None else {
        "SiII1260": (1260.42, -0.35, [(1259.52, -0.08)], [(1264.74, +0.10)]),
        "CII1334":  (1334.53, -0.35, [],                 [(1335.71, +0.14)]),
        "SiII1526": (1526.71, -0.32, [],                 [(1533.43, +0.11)]),
        "FeII1608": (1608.45, -0.20, [],                 []),
    }
    for name, (lam0, depth, abs_partners, em_partners) in spec_defs.items():
        sig = np.hypot(SIG_LSF, sig_intr_kms / C * lam0)
        flux += gauss(GRID, lam0 * shift, sig, depth * cont_level)
        if blue_wing is not None:
            wamp, wdv, wsig_kms = blue_wing
            flux += gauss(GRID, lam0 * (1 + (dv_true + wdv) / C),
                          np.hypot(SIG_LSF, wsig_kms / C * lam0), wamp * cont_level)
        for lam_p, dep_p in abs_partners:
            sig_p = np.hypot(SIG_LSF, sig_intr_kms / C * lam_p)
            flux += gauss(GRID, lam_p * shift, sig_p, dep_p * cont_level)
        if fine_struct:
            for lam_p, dep_p in em_partners:
                sig_p = np.hypot(SIG_LSF, sig_intr_kms / C * lam_p)
                flux += gauss(GRID, lam_p * shift, sig_p, dep_p * cont_level)

    err = np.full_like(GRID, max(noise, 1e-6) * cont_level)
    if noise > 0:
        flux = flux + rng.standard_normal(GRID.size) * err
    return GRID, flux, err


def hdr(s):
    print("\n" + "=" * 76)
    print(s)
    print("=" * 76)


fails = []


def check(label, got, want, tol, unit="km/s"):
    ok = np.isfinite(got) and abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:52s} got {got:+8.1f} "
          f"want {want:+8.1f} +/-{tol:.0f} {unit}")
    if not ok:
        fails.append(label)
    return ok


# ---------------------------------------------------------------------
hdr("TEST 1  clean recovery, no noise, fine structure ON, ncomp=1")
for dv_true in (0.0, -60.0, -145.0, +80.0):
    w, f, e = build_stack(dv_true, noise=0.0)
    res = lis.measure_lines(w, f, e, lines="science", ncomp=1, n_mc=0, z=Z)
    print(f"\n injected dv = {dv_true:+.0f} km/s")
    for k, r in res.items():
        print(f"   {k:10s} v_gauss={r['vel_gauss_point_kms']:+8.1f}  "
              f"v_min={r['vel_min_point_kms']:+8.1f}  "
              f"fwhm={r['fwhm_kms']:6.0f}  ok={r['fit_success']}")
    # Si II 1526 is the clean one -> must recover truth tightly
    check(f"SiII1526 gauss, dv_true={dv_true:+.0f}",
          res["SiII1526"]["vel_gauss_point_kms"], dv_true, 12.0)

# ---------------------------------------------------------------------
hdr("TEST 2  CLAIM A: C II* 1335.71 emission biases single-Gaussian BLUEWARD")
dv_true = -60.0
w_on, f_on, e_on = build_stack(dv_true, noise=0.0, fine_struct=True)
w_off, f_off, e_off = build_stack(dv_true, noise=0.0, fine_struct=False)

r_on1 = lis.measure_line(w_on, f_on, e_on, "CII1334", ncomp=1, n_mc=0, z=Z)
r_off1 = lis.measure_line(w_off, f_off, e_off, "CII1334", ncomp=1, n_mc=0, z=Z)
r_on2 = lis.measure_line(w_on, f_on, e_on, "CII1334", ncomp=2, n_mc=0, z=Z)

bias = r_on1["vel_gauss_point_kms"] - r_off1["vel_gauss_point_kms"]
print(f"  ncomp=1, C II* ON  : v_gauss = {r_on1['vel_gauss_point_kms']:+8.1f}")
print(f"  ncomp=1, C II* OFF : v_gauss = {r_off1['vel_gauss_point_kms']:+8.1f}")
print(f"  ncomp=2, C II* ON  : v_gauss = {r_on2['vel_gauss_point_kms']:+8.1f}  "
      f"(amps={[f'{a:+.3f}' for a in r_on2['amps']]})")
print(f"  --> single-Gaussian bias from the fine-structure fill-in = {bias:+.1f} km/s")
print(f"      (spec claims blueward, i.e. NEGATIVE)")
if bias < 0:
    print(f"  [PASS] bias is blueward as the spec claims, magnitude {abs(bias):.0f} km/s")
else:
    print(f"  [FAIL] bias is NOT blueward -- spec claim sec 1.2 is wrong")
    fails.append("CII* bias direction")
check("CII1334 ncomp=2 recovers truth with C II* present",
      r_on2["vel_gauss_point_kms"], dv_true, 15.0)
check("CII1334 ncomp=1 recovers truth with C II* ABSENT",
      r_off1["vel_gauss_point_kms"], dv_true, 12.0)

# ---------------------------------------------------------------------
hdr("TEST 3  CLAIM B: on a blue-winged trough, v_min sits REDWARD of v_gauss")
w, f, e = build_stack(-60.0, noise=0.0, fine_struct=False,
                      blue_wing=(-0.18, -350.0, 300.0))
for k in ("SiII1526", "SiII1260"):
    r = lis.measure_line(w, f, e, k, ncomp=1, n_mc=0, z=Z)
    asym = r["vel_min_point_kms"] - r["vel_gauss_point_kms"]
    print(f"  {k:10s} v_min={r['vel_min_point_kms']:+8.1f}  "
          f"v_gauss={r['vel_gauss_point_kms']:+8.1f}  asym={asym:+7.1f}")
    if asym > 0:
        print(f"    [PASS] minimum redward of centroid, as the spec claims")
    else:
        print(f"    [FAIL] spec's asymmetry claim (sec 'TWO ESTIMATORS') is wrong")
        fails.append(f"{k} asymmetry direction")

# ---------------------------------------------------------------------
hdr("TEST 4  Monte-Carlo error model + noise realism")
w, f, e = build_stack(-60.0, noise=0.02, seed=7)
res = lis.measure_lis_stack(
    {"rest_wave": w, "stacks": {"mean": {"flux": f, "err": e}},
     "ngal": 200, "z_median": Z, "output": "L_fiber"},
    method="mean", lines="science", ncomp={"CII1334": 2}, n_mc=100, seed=3)
lis.describe_lis(res)
avg = lis.z_lis_average(res, estimator="gauss")
print(f"\n  z_LIS-equivalent = {avg['v_lis_kms']:+.1f} +/- {avg['v_lis_err_kms']:.1f} "
      f"km/s from {avg['lines_used']}")
check("noisy 3-line z_LIS mean recovers injected -60", avg["v_lis_kms"], -60.0, 25.0)
for k, r in res["lines"].items():
    if not (0 < r["vel_gauss_err_kms"] < 500):
        print(f"  [FAIL] {k}: implausible MC error {r['vel_gauss_err_kms']}")
        fails.append(f"{k} MC error")
    if r["n_mc_gauss_ok"] < 90:
        print(f"  [WARN] {k}: only {r['n_mc_gauss_ok']}/100 MC gauss fits succeeded")

# ---------------------------------------------------------------------
hdr("TEST 5  pixel quantization of the trough minimum")
w, f, e = build_stack(-60.0, noise=0.0)
r_ref = lis.measure_line(w, f, e, "SiII1526", n_mc=0, z=Z, refine=True)
r_raw = lis.measure_line(w, f, e, "SiII1526", n_mc=0, z=Z, refine=False)
print(f"  pixel scale = {r_ref['pix_scale_kms']:.1f} km/s "
      f"({r_ref['pix_scale_A']:.2f} A)")
print(f"  v_min refined = {r_ref['vel_min_point_kms']:+8.2f} "
      f"(refined={r_ref['min_refined']})")
print(f"  v_min raw     = {r_raw['vel_min_point_kms']:+8.2f}")
check("parabolic refinement improves the trough minimum",
      r_ref["vel_min_point_kms"], -60.0, 15.0)
if abs(r_raw["vel_min_point_kms"] + 60.0) < abs(r_ref["vel_min_point_kms"] + 60.0):
    print("  [WARN] refinement made it worse on this realization")

# ---------------------------------------------------------------------
hdr("TEST 6  degenerate / failure paths must not raise")
w, f, e = build_stack(-60.0, noise=0.0)
cases = {
    "all-NaN flux": (w, np.full_like(f, np.nan), e),
    "all-zero err": (w, f, np.zeros_like(e)),
    "grid too short": (w[:5], f[:5], e[:5]),
    "no err at all": (w, f, None),
    "line out of grid": (np.arange(1600.0, 1700.0, 0.2),
                         np.ones(500), np.full(500, 0.01)),
}
for label, (ww, ff, ee) in cases.items():
    try:
        r = lis.measure_line(ww, ff, ee, "SiII1526", n_mc=5, z=Z)
        print(f"  [PASS] {label:20s} -> no raise; v_gauss="
              f"{r['vel_gauss_kms']}, reason={r['fit_reason']}")
    except Exception as exc:
        print(f"  [FAIL] {label:20s} -> RAISED {type(exc).__name__}: {exc}")
        fails.append(f"raise on {label}")

# ---------------------------------------------------------------------
hdr("TEST 7  continuum mode sensitivity (weldon_linear vs poly)")
w, f, e = build_stack(-60.0, noise=0.01, seed=11)
for mode in ("weldon_linear", "poly"):
    r = lis.measure_line(w, f, e, "SiII1526", cont_mode=mode, n_mc=30, z=Z)
    print(f"  {mode:15s} v_gauss={r['vel_gauss_kms']:+8.1f} +/- "
          f"{r['vel_gauss_err_kms']:5.1f}   v_min={r['vel_min_kms']:+8.1f}")

# ---------------------------------------------------------------------
hdr("TEST 8  Lya emission path (sign=+1) and the all-lines loop")
w, f, e = build_stack(-60.0, noise=0.01, seed=5)
f = f + gauss(GRID, 1215.67 * (1 + 400.0 / C), 2.0, +1.4)   # +400 km/s emission
allres = lis.measure_lines(w, f, e, lines=None, ncomp=1, n_mc=20, z=Z)
print(f"  Lya v_gauss = {allres['Lya']['vel_gauss_kms']:+.1f} km/s (injected +400)")
check("Lya emission centroid", allres["Lya"]["vel_gauss_kms"], 400.0, 40.0)
print(f"  measured {len(allres)} lines: {sorted(allres)}")

# ---------------------------------------------------------------------
hdr("TEST 9  blend systematic table (compare_ncomp) recovers truth at ncomp_rec")
w, f, e = build_stack(-60.0, noise=0.0)
cmp_ = lis.compare_ncomp(w, f, e, lines="science", z=Z)
print(f"  {'line':10s} {'ncomp=1':>9s} {'rec':>4s} {'ncomp=rec':>10s} {'shift':>9s}")
for k, v in cmp_["rows"].items():
    print(f"  {k:10s} {v['ncomp1_kms']:+9.2f} {v['ncomp_rec']:>4d} "
          f"{v['ncomp_rec_kms']:+10.2f} {v['blend_shift_kms']:+9.2f}")
    check(f"{k} at ncomp_recommended recovers truth", v["ncomp_rec_kms"], -60.0, 8.0)

# ---------------------------------------------------------------------
hdr("TEST 10  two-step (simultaneous=False) agrees with the simultaneous fit")
w, f, e = build_stack(-60.0, noise=0.0, fine_struct=False)
for sim in (True, False):
    r = lis.measure_line(w, f, e, "SiII1526", ncomp=1, n_mc=0, z=Z, simultaneous=sim)
    print(f"  simultaneous={str(sim):5s}: dv={r['vel_gauss_point_kms']:+8.2f} "
          f"sigma={r['sigma_A']:.4f} ok={r['fit_success']}")
    check(f"two-step/simultaneous={sim} recovers truth",
          r["vel_gauss_point_kms"], -60.0, 8.0)

# ---------------------------------------------------------------------
hdr("TEST 11  fitted width is unbiased ONLY at ncomp>1 (spec's sigma_v check)")
sig_true = np.hypot(SIG_LSF, 200.0 / C * 1526.71)
fwhm_true = 2.35482 * C * sig_true / 1526.71
w, f, e = build_stack(-60.0, noise=0.0)
r1 = lis.measure_line(w, f, e, "SiII1526", ncomp=1, n_mc=0, z=Z)
r2 = lis.measure_line(w, f, e, "SiII1526", ncomp=2, n_mc=0, z=Z)
print(f"  injected FWHM = {fwhm_true:.1f} km/s")
print(f"  ncomp=1 -> {r1['fwhm_kms']:.1f}   ncomp=2 -> {r2['fwhm_kms']:.1f}")
check("width unbiased at ncomp=2", r2["fwhm_kms"], fwhm_true, 15.0)
if abs(r1["fwhm_kms"] - fwhm_true) < 15.0:
    print("  [WARN] ncomp=1 width was NOT biased here -- the docstring caveat "
          "may be overstated for this partner strength")
else:
    print(f"  [ok] ncomp=1 width inflated by {r1['fwhm_kms'] - fwhm_true:+.0f} km/s, "
          f"as the docstring warns")

# ---------------------------------------------------------------------
hdr("FINAL SUMMARY")
if fails:
    print(f"  {len(fails)} FAILURE(S):")
    for x in fails:
        print(f"    - {x}")
    sys.exit(1)
print("  all checks passed")
