"""
smhm_models.py -- stellar-mass -> halo-mass (SMHM) relations, several models.

All functions take/return LINEAR Msun (not log), except where noted.
Halo mass convention notes are given per-model (this MATTERS for R200c).

Models:
  - moster13   : Moster, Naab & White 2013 (your current pipeline)   [M200c-ish]
  - girelli20  : Girelli+2020, COSMOS-calibrated, z<=4                [M200c]
  - behroozi19 : Behroozi+2019 UniverseMachine DR1 (Appendix J)       [Mpeak, B&N vir]
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

# =====================================================================
# 1) MOSTER+2013  (your current relation) -- forward: Mhalo -> Mstar
# =====================================================================
def moster13_mstar_from_mhalo(Mh, z):
    a = z / (1 + z)
    logM1 = 11.590 + 1.195 * a
    N     = 0.0351 - 0.0247 * a
    beta  = 1.376  - 0.826 * a
    gamma = 0.608  + 0.329 * a
    M1 = 10 ** logM1
    ratio = 2 * N / ((Mh / M1) ** (-beta) + (Mh / M1) ** gamma)
    return ratio * Mh

# =====================================================================
# 2) GIRELLI+2020  (COSMOS, M200c, z<=4)  -- forward: Mhalo -> Mstar
#    M*/Mh(z) = 2 A(z) [ (Mh/MA)^-beta + (Mh/MA)^gamma ]^-1
#    log MA = B + z*mu ;  A = C*(1+z)^nu ;  gamma = D*(1+z)^eta ;  beta = F*z + E
#    Table 3, reference case (no relative scatter).
# =====================================================================
_GIR = dict(B=11.79, mu=0.20, C=0.046, nu=-0.38, D=0.709, eta=-0.18, F=0.043, E=0.96)

def girelli20_mstar_from_mhalo(Mh, z, p=_GIR):
    logMA = p['B'] + z * p['mu']
    A     = p['C'] * (1 + z) ** p['nu']
    gamma = p['D'] * (1 + z) ** p['eta']
    beta  = p['F'] * z + p['E']
    MA = 10 ** logMA
    ratio = 2 * A / ((Mh / MA) ** (-beta) + (Mh / MA) ** gamma)
    return ratio * Mh

# =====================================================================
# 3) BEHROOZI+2019 UniverseMachine DR1 (Appendix J)  -- forward: Mpeak -> Mstar
#    log10(M*/M1) = eps - log10(10^(-a x)+10^(-b x)) + g*exp(-0.5 (x/d)^2)
#    x = log10(Mpeak/M1).  Params: DR1 gen_smhm.py median fit (all galaxies).
#    NOTE: Mpeak = peak historical Bryan&Norman virial mass, NOT instantaneous M200c.
# =====================================================================
_BEH = dict(
    e0=-1.435, ea=+1.831, ela=+1.368, ez=-0.217,
    m0=+12.035, ma=+4.556, mla=+4.417, mz=-0.731,
    a0=+1.963, aa=-2.316, ala=-1.732, az=+0.178,
    b0=+0.482, ba=-0.841, bz=-0.471,
    d0=+0.411,
    g0=-1.034, ga=-3.100, gz=-1.055,
)

def behroozi19_mstar_from_mhalo(Mh, z, p=_BEH):
    a = 1.0 / (1.0 + z)
    lna = np.log(a)
    logM1 = p['m0'] + p['ma']*(a-1) - p['mla']*lna + p['mz']*z
    eps   = p['e0'] + p['ea']*(a-1) - p['ela']*lna + p['ez']*z
    alpha = p['a0'] + p['aa']*(a-1) - p['ala']*lna + p['az']*z
    beta  = p['b0'] + p['ba']*(a-1) + p['bz']*z
    delta = p['d0']
    loggamma = p['g0'] + p['ga']*(a-1) + p['gz']*z
    gamma = 10 ** loggamma
    x = np.log10(Mh) - logM1
    logMstar = (logM1 + eps
                - np.log10(10 ** (-alpha * x) + 10 ** (-beta * x))
                + gamma * np.exp(-0.5 * (x / delta) ** 2))
    return 10 ** logMstar

# =====================================================================
# Generic inverter: Mstar -> Mhalo  for any forward model
# =====================================================================
_FORWARD = {
    'moster13':   moster13_mstar_from_mhalo,
    'girelli20':  girelli20_mstar_from_mhalo,
    'behroozi19': behroozi19_mstar_from_mhalo,
}

def mhalo_from_mstar(Mstar, z, model='girelli20', logMh_lo=9.0, logMh_hi=15.5):
    """
    Invert any forward SMHM model. Mstar [Msun] -> Mhalo [Msun].

    Robust to the high-z massive-end TURNOVER: some models (notably
    Behroozi+2019) have a median M*(Mh) that saturates and turns over at
    high z, so very massive galaxies have NO median halo. In that case this
    returns np.nan (the galaxy is above the median relation -- treat R_vir
    for it as undefined/highly uncertain rather than extrapolating).
    """
    f = _FORWARD[model]
    target = np.log10(Mstar)
    grid = np.linspace(logMh_lo, logMh_hi, 400)
    logMs = np.log10(f(10 ** grid, z))
    imax = int(np.argmax(logMs))                 # turnover of the rising branch
    if target > logMs[imax]:                     # off the top of the relation
        return np.nan
    def g(logMh):
        return np.log10(f(10 ** logMh, z)) - target
    return 10 ** brentq(g, logMh_lo, grid[imax]) # search rising branch only
