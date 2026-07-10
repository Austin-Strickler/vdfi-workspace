"""
mstar_to_rvir.py
================
Stellar mass -> halo mass -> virial radius (R200c) at z = 2-3.

Chain
-----
  M_star  --[SMHM]-->  M_vir  --[c(M) + NFW]-->  M_200c  -->  R_200c

Why two halo-mass definitions appear:
  * The empirical SMHM relations here (Behroozi+2019 UniverseMachine, and the
    COSMOS/Shuntov HOD work) report *virial* masses (Bryan & Norman 1998
    overdensity, as implemented in Colossus).
  * You asked for R_200c, so we convert M_vir -> M_200c using a concentration-
    mass model (Diemer & Joyce 2019) before computing the radius.
  * R_200c is returned in PHYSICAL kpc. Multiply by (1+z) for comoving kpc.

The pure geometric step R = [3 M / (4 pi * 200 * rho_crit(z))]^(1/3) is exact;
all model dependence lives in the SMHM (M_star -> M_vir) step.

Validation (run as __main__): reproduces the canonical Behroozi+2019 peak
baryon-conversion efficiency M*/Mh ~ 0.023 at logMh ~ 12.1 (z~0), moving to
logMh ~ 12.4 at z=2.

Requires: numpy, colossus  (pip install colossus)
Cosmology: Planck18 (matches the papers, which all adopt Planck).

References
----------
Behroozi, Wechsler, Hearin & Conroy 2019, MNRAS 488, 3143 (UniverseMachine)
Bryan & Norman 1998, ApJ 495, 80              (virial overdensity)
Diemer & Joyce 2019, ApJ 871, 168             (concentration-mass)
Paquereau, Shuntov et al. 2025, A&A (COSMOS-Web) arXiv:2501.11674
Shuntov et al. 2022, A&A 664, A61 (COSMOS2020)  -- SHMR functional form (Eq. 9)
"""
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import mass_defs, concentration

cosmo = cosmology.setCosmology("planck18")
_h = cosmo.Hz(0) / 100.0

# ---------------------------------------------------------------------------
# Behroozi+2019 median SMHM  (Appendix J / released gen_smhm.py, "true" params)
#   log10 M* = log10 M1 + eps - log10(10^{-a x}+10^{-b x}) + g*exp(-0.5 (x/d)^2)
#   x = log10(Mh_vir / M1)
# ---------------------------------------------------------------------------
_B19 = dict(e0=-1.435, ea=1.831, ela=1.368, ez=-0.217,
            m0=12.035, ma=4.556, mla=4.417, mz=-0.731,
            a0=1.963,  aa=-2.316, ala=-1.732, az=0.178,
            b0=0.482,  ba=-0.841, bz=-0.471,
            d0=0.411,
            g0=-1.034, ga=-3.100, gz=-1.055)


def behroozi19_logMstar(logMh_vir, z):
    """log10 M*(M_sun) given log10 M_vir(M_sun) and redshift z (Behroozi+2019)."""
    p = _B19
    a = 1.0 / (1.0 + z); a1 = a - 1.0; lna = np.log(a)
    logM1 = p["m0"] + p["ma"] * a1 - p["mla"] * lna + p["mz"] * z
    eps   = p["e0"] + p["ea"] * a1 - p["ela"] * lna + p["ez"] * z
    al    = p["a0"] + p["aa"] * a1 - p["ala"] * lna + p["az"] * z
    be    = p["b0"] + p["ba"] * a1 + p["bz"] * z
    de    = p["d0"]
    ga    = 10 ** (p["g0"] + p["ga"] * a1 + p["gz"] * z)
    x = np.asarray(logMh_vir) - logM1
    return logM1 + eps - np.log10(10 ** (-al * x) + 10 ** (-be * x)) \
        + ga * np.exp(-0.5 * (x / de) ** 2)


def mstar_to_logMvir(logMstar, z, mh_lo=9.5, mh_hi=13.0, n=4000):
    """Invert the SMHM on its rising branch (below the high-mass turnover).

    Returns np.nan where logMstar exceeds the maximum M* the relation produces
    at that z -- i.e. the regime that would require (unpopulated) cluster-scale
    halos. At z=2-3 the Behroozi turnover is near logM* ~ 10.8-11.0.
    """
    g = np.linspace(mh_lo, mh_hi, n)
    ms = behroozi19_logMstar(g, z)
    logMstar = np.asarray(logMstar, dtype=float)
    out = np.interp(logMstar, ms, g, left=np.nan, right=np.nan)
    out = np.where(logMstar > ms.max(), np.nan, out)
    return out


def logMvir_to_R200c(logMvir, z, cmodel="diemer19"):
    """(log10 M200c[M_sun], R200c[physical kpc]) from log10 M_vir[M_sun]."""
    Mvir_hinv = 10 ** np.asarray(logMvir) * _h            # M_sun / h
    c = concentration.concentration(Mvir_hinv, "vir", z, model=cmodel)
    M200c_hinv, R200c_kpch, _ = mass_defs.changeMassDefinition(
        Mvir_hinv, c, z, "vir", "200c")
    return np.log10(M200c_hinv / _h), R200c_kpch / _h      # kpc physical


def mstar_to_R200c(logMstar, z):
    """Convenience: log10 M*  ->  R200c [physical kpc] (nan past turnover)."""
    lmv = mstar_to_logMvir(logMstar, z)
    r = np.full(np.shape(lmv), np.nan)
    good = np.isfinite(lmv)
    if np.any(good):
        _, rr = logMvir_to_R200c(np.atleast_1d(lmv)[good], z)
        r = np.atleast_1d(r); r[good] = rr
    return r if np.ndim(logMstar) else float(r)


# ---------------------------------------------------------------------------
# NOTE on the COSMOS-Web / Shuntov 2025 column
# ---------------------------------------------------------------------------
# Paquereau, Shuntov et al. 2025 measure the SHMR via an HOD (Zheng+2005) and
# report it as discrete characteristic-halo-mass points per (z, M*-threshold)
# bin, using the Bryan-Norman virial mass definition (same as above). At z=2-3
# their samples cover ~logM* 9-11, and they calibrate to UniverseMachine
# (Behroozi+2019), so their R200c(M*) tracks the Behroozi column closely
# (peak halo mass logMh ~ 12.3 at z~2, vs 12.4 here). To reproduce their exact
# curve, plug their per-bin best-fit parameters into the Shuntov+2022 Eq. 9:
#     log Mh = log M1 + beta*log(M*/M*0)
#              + (M*/M*0)^delta / (1 + (M*/M*0)^-gamma) - 1/2
# (M1, M*0, beta, delta, gamma tabulated per redshift bin in that work).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Validation - Behroozi+2019 peak efficiency:")
    for z in (0.1, 2.0, 3.0):
        g = np.linspace(10.5, 14, 400); r = behroozi19_logMstar(g, z) - g
        i = np.argmax(r)
        print(f"  z={z}: peak M*/Mh={10**r[i]:.3f} at logMh={g[i]:.2f}")

    for z in (2.0, 2.5, 3.0):
        print(f"\n=== z={z}  (Behroozi+2019, R200c physical kpc) ===")
        print(f"{'logM*':>6}{'logM200c':>10}{'R200c/kpc':>11}")
        for x in np.arange(8.0, 12.001, 0.25):
            lmv = mstar_to_logMvir(x, z)
            if not np.isfinite(lmv) or lmv > 12.8:
                print(f"{x:>6.2f}{'--':>10}{'-- (past turnover)':>22}")
            else:
                lm2, rr = logMvir_to_R200c(lmv, z)
                print(f"{x:>6.2f}{lm2:>10.2f}{rr:>11.1f}")
