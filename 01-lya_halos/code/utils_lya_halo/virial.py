"""
virial.py -- halo mass / virial radius and radial-bin conversions.

Chain: log Mstar --(SMHM relation)--> Mhalo --> R200c --> bin edges.

Default relation is Girelli+2020 (switched from Moster+2013 on 2026-07-27 --
see docs/research-notes.md "SMHM values and methods" for the divergence test
and literature check that motivated it: this project's own external
centroid-method check, Lujan Niemeyer et al. 2022 [arXiv:2207.11098], uses
Behroozi+2019; Guo et al. 2024 [A&A 691, A66] uses Girelli+2020; neither uses
Moster+2013). The model is selected via PipelineConfig.smhm_model (a real,
provenance-tracked, cache-signature field -- changing it invalidates Stage 1
caches, as it should, since it changes the physical bin edges). Every
extraction-facing function below also defaults its own `model` kwarg to
DEFAULT_SMHM_MODEL, so direct/notebook calls outside the cfg-driven pipeline
get the new default too.

The actual extraction converts PER GALAXY (virial_to_kpc_bins with that
galaxy's mass and z, model=cfg.smhm_model). median_virial_kpc_bins is a
SEPARATE reporting helper for the notebook -- it uses the sample median and
is NOT what bins the data.

Merged 2026-07-27 from the former smhm.py (itself merged from
smhm_models.py + smhm_compare.py): the multi-model machinery (girelli20,
behroozi19, the generic mhalo_from_mstar inverter, compare_smhm,
rvir_divergence_table, and the two comparison plots) belongs here now that
Girelli+2020 is the pipeline default rather than a side investigation.

*** MASS-DEFINITION CAVEAT -- read before trusting the behroozi19 column ***
girelli20 (the default) is calibrated on M200c, so R200c_from_M200c() is a
same-definition, apples-to-apples conversion for it. behroozi19's Mh is
Mpeak, the peak historical Bryan & Norman (1998) VIRIAL mass -- a different,
larger mass definition than M200c at fixed halo. Running it through
R200c_from_M200c() here is an approximation (~+1.8% vs. the properly
converted value, per the 2026-07-27 divergence test against
tests/mstar_to_rvir.py -- small in this sample's mass/z regime, but not
exact). moster13's own halo-mass convention (M200c vs. M200-mean) has not
been independently re-verified either. Lujan Niemeyer et al. 2022 use
behroozi19 and take R_vir directly under Bryan & Norman -- they never
convert to M200c; tests/mstar_to_rvir.py (workspace root) is the module that
does that conversion properly (Diemer & Joyce 2019 concentration-mass via
colossus) if behroozi19 is ever wanted for real, not just comparison.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.constants import G
from scipy.optimize import brentq

# =====================================================================
# COSMOLOGY  --  Planck18, matching the analysis notebook
# (cell 0: `from astropy.cosmology import Planck18 as cosmo`).
# This scales every R200c AND every luminosity distance in Stage 2, so it
# must match the rest of the analysis. Change here if your convention differs.
# =====================================================================
from astropy.cosmology import Planck18 as cosmo


# =====================================================================
# SMHM FORWARD MODELS  (Mhalo -> Mstar) -- pick one via PipelineConfig.smhm_model
# =====================================================================

# ---- 1) MOSTER+2013  (legacy pipeline default through 2026-07-27) ---------
def moster13_mstar_from_mhalo(Mh, z):
    """Moster+2013 SHMR. Mh [Msun] -> Mstar [Msun]."""
    a = z / (1 + z)
    logM1 = 11.590 + 1.195 * a
    N = 0.0351 - 0.0247 * a
    beta = 1.376 - 0.826 * a
    gamma = 0.608 + 0.329 * a
    M1 = 10 ** logM1
    ratio = 2 * N / ((Mh / M1) ** (-beta) + (Mh / M1) ** gamma)
    return ratio * Mh


def mhalo_from_mstar_moster13(Mstar, z):
    """Invert Moster+2013 directly. Mstar [Msun] -> Mhalo [Msun]. Kept as a
    back-compat thin entry point (guide.py/older notebooks call it by name);
    new code should use mhalo_from_mstar(..., model='moster13'), which adds
    turnover-NaN handling (a no-op for moster13 in this sample's mass range --
    both forms agree bit-for-bit, verified 2026-07-27)."""
    def f(logMh):
        Mh = 10 ** logMh
        return np.log10(moster13_mstar_from_mhalo(Mh, z)) - np.log10(Mstar)
    logMh = brentq(f, 9, 15)
    return 10 ** logMh


# ---- 2) GIRELLI+2020  (COSMOS, M200c, z<=4)  -- DEFAULT since 2026-07-27 --
#    M*/Mh(z) = 2 A(z) [ (Mh/MA)^-beta + (Mh/MA)^gamma ]^-1
#    log MA = B + z*mu ;  A = C*(1+z)^nu ;  gamma = D*(1+z)^eta ;  beta = F*z + E
#    Table 3, reference case (no relative scatter). Used by Guo et al. 2024
#    (A&A 691, A66) and Erb et al. 2018/2023 to estimate LAE-sample r_vir.
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


# ---- 3) BEHROOZI+2019 UniverseMachine DR1 (Appendix J)  -- Mpeak -> Mstar --
#    log10(M*/M1) = eps - log10(10^(-a x)+10^(-b x)) + g*exp(-0.5 (x/d)^2)
#    x = log10(Mpeak/M1).  Params: DR1 gen_smhm.py median fit (all galaxies).
#    NOTE: Mpeak = peak historical Bryan&Norman virial mass, NOT instantaneous
#    M200c -- see the module-level mass-definition caveat above. This is the
#    relation Lujan Niemeyer et al. 2022 use for their r_vir estimate.
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
# Generic inverter: Mstar -> Mhalo  for any forward model above
# =====================================================================
_FORWARD = {
    'moster13':   moster13_mstar_from_mhalo,
    'girelli20':  girelli20_mstar_from_mhalo,
    'behroozi19': behroozi19_mstar_from_mhalo,
}
_LABELS = {'moster13': 'Moster+2013 (legacy)',
           'girelli20': 'Girelli+2020 (COSMOS, default since 2026-07-27)',
           'behroozi19': 'Behroozi+2019 (UM DR1)'}

DEFAULT_SMHM_MODEL = 'girelli20'


def mhalo_from_mstar(Mstar, z, model=DEFAULT_SMHM_MODEL, logMh_lo=9.0, logMh_hi=15.5):
    """
    Invert any forward SMHM model. Mstar [Msun] -> Mhalo [Msun].

    Robust to the high-z massive-end TURNOVER: some models (notably
    Behroozi+2019) have a median M*(Mh) that saturates and turns over at
    high z, so very massive galaxies have NO median halo. In that case this
    returns np.nan (the galaxy is above the median relation -- treat R_vir
    for it as undefined/highly uncertain rather than extrapolating).
    girelli20 and moster13 do not turn over in this sample's mass range
    (verified 2026-07-27), so this only bites in practice for behroozi19.
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


# ---- M200c -> R200c -------------------------------------------------------
def R200c_from_M200c(M200c, z):
    """R200c = [G M200c / (100 H(z)^2)]^(1/3), returned in kpc.

    See the module-level mass-definition caveat: only girelli20's Mh is
    actually M200c. moster13 and (especially) behroozi19 feed this function
    a differently-defined halo mass -- see estimate_M200c_R200c_from_Mstar.
    """
    M200c = np.atleast_1d(M200c) * u.Msun
    Hz = cosmo.H(z)
    R200c = ((G * M200c / (100 * Hz ** 2)) ** (1 / 3)).to(u.kpc).value
    return R200c if R200c.size > 1 else float(R200c[0])


def estimate_M200c_R200c_from_Mstar(Mstar, z, model=DEFAULT_SMHM_MODEL):
    """Mstar [Msun] -> (Mhalo [Msun], R200c [kpc]) via the chosen SMHM
    relation + R200c(z). Default is Girelli+2020 (switched from Moster+2013
    2026-07-27); pass model='moster13' or 'behroozi19' for the alternates.
    Returns (nan, nan) if Mstar is above that model's median relation (only
    possible in practice for behroozi19 -- see mhalo_from_mstar)."""
    Mhalo = mhalo_from_mstar(Mstar, z, model=model)
    if not np.isfinite(Mhalo):
        return np.nan, np.nan
    R200c = R200c_from_M200c(Mhalo, z)
    return Mhalo, R200c


def physical_kpc_to_arcsec(R_kpc, z):
    """Physical size R_kpc at redshift z -> angular size in arcsec (Planck18)."""
    scale = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)
    return (R_kpc * u.kpc / scale).value


# ---- virial (R/Rvir) edges -> physical units ------------------------------
def virial_to_kpc_bins(v_bins, m, z, m_min=7.0, m_max=11.0, verbose=False,
                       model=DEFAULT_SMHM_MODEL):
    """R/Rvir edges -> kpc edges, using THIS galaxy's log-mass m and z, under
    the given SMHM `model` (default Girelli+2020). Returns all-NaN if the
    galaxy's mass/z are invalid OR if it's above the model's median relation
    (r200c is then nan and propagates through the multiply)."""
    m = float(m)
    z = float(z)
    if not np.isfinite(m) or not np.isfinite(z) or z <= 0:
        return np.full_like(np.asarray(v_bins, dtype=float), np.nan)
    m_used = np.clip(m, m_min, m_max)
    if verbose and m_used != m:
        print(f"Capping logMstar {m:.2f} -> {m_used:.2f}, z={z:.3f}")
    _, r200c = estimate_M200c_R200c_from_Mstar(10 ** m_used, z, model=model)
    return np.asarray(v_bins, dtype=float) * r200c


def virial_to_angular_bins(v_bins, m, z, m_min=7.0, m_max=11.0,
                           max_arcsec=None, verbose=False, model=DEFAULT_SMHM_MODEL):
    """R/Rvir edges -> arcsec edges, using THIS galaxy's log-mass m and z,
    under the given SMHM `model` (default Girelli+2020)."""
    v_bins = np.asarray(v_bins, dtype=float)
    m = float(m)
    z = float(z)
    if not np.isfinite(m) or not np.isfinite(z) or z <= 0:
        return np.full_like(v_bins, np.nan)
    m_used = np.clip(m, m_min, m_max)
    if verbose and m_used != m:
        print(f"Capping logMstar {m:.2f} -> {m_used:.2f}, z={z:.3f}")
    _, r200c = estimate_M200c_R200c_from_Mstar(10 ** m_used, z, model=model)
    theta_r = physical_kpc_to_arcsec(r200c, z)
    angular_bins = v_bins * theta_r
    if max_arcsec is not None:
        angular_bins = np.minimum(angular_bins, max_arcsec)
    return angular_bins


def compute_annuli_from_virial(v_bins, m, z, kpc_bins=True, model=DEFAULT_SMHM_MODEL):
    """Convenience dispatch: virial edges -> kpc (default) or arcsec."""
    if kpc_bins:
        return virial_to_kpc_bins(v_bins, m, z, model=model)
    return virial_to_angular_bins(v_bins, m, z, model=model)


# ---- reporting only (NOT used for extraction) ------------------------------
def median_virial_kpc_bins(v_bins, run_table, mass_col='MASS_50', z_col='z',
                           model=DEFAULT_SMHM_MODEL):
    """
    EXPLICIT median-galaxy mapping: R/Rvir edges -> kpc using the sample
    median mass and z. For plots/tables only -- the pipeline bins each galaxy
    with its own mass and z, not this.

    Returns (kpc_edges, m_median, z_median).
    """
    m_med = np.nanmedian(np.asarray(run_table[mass_col], dtype=float))
    z_med = np.nanmedian(np.asarray(run_table[z_col], dtype=float))
    return virial_to_kpc_bins(v_bins, m_med, z_med, verbose=False, model=model), m_med, z_med


# =====================================================================
# SMHM MODEL COMPARISON  (merged from smhm.py, 2026-07-27)
# =====================================================================
_MODELS = ['moster13', 'girelli20', 'behroozi19']


def compare_smhm(logMstar, z, models=_MODELS):
    """
    For one galaxy (logMstar, z), return a dict per model with logMhalo and
    R200c [kpc]. NaN where the galaxy is above a model's median relation.
    Behroozi's R200c is approximate -- see the module-level caveat.

    >>> compare_smhm(10.5, 2.3)
    """
    out = {}
    for m in models:
        Mh = mhalo_from_mstar(10 ** logMstar, z, model=m)
        if np.isnan(Mh):
            out[m] = dict(logMhalo=np.nan, R200c_kpc=np.nan)
        else:
            out[m] = dict(logMhalo=float(np.log10(Mh)),
                          R200c_kpc=float(R200c_from_M200c(Mh, z)))
    return out


def rvir_divergence_table(logMstar_grid, z, ref='moster13', models=_MODELS):
    """
    Tabulate R200c(kpc) for each model across a grid of logMstar at fixed z,
    plus the % difference of each model vs `ref`. Returns a plain list of rows
    (dicts) so you can print it or drop it into a DataFrame.

    >>> import pandas as pd
    >>> df = pd.DataFrame(rvir_divergence_table(np.arange(9.0,11.01,0.5), 2.3))
    """
    rows = []
    for lm in logMstar_grid:
        row = {'logMstar': round(float(lm), 3)}
        Rref = None
        for m in models:
            Mh = mhalo_from_mstar(10 ** lm, z, model=m)
            R = np.nan if np.isnan(Mh) else R200c_from_M200c(Mh, z)
            row[f'R200c_{m}'] = None if np.isnan(R) else round(R, 1)
            if m == ref:
                Rref = R
        for m in models:
            R = row[f'R200c_{m}']
            row[f'dpct_{m}'] = (None if (R is None or Rref is None or np.isnan(Rref))
                                else round(100 * (R - Rref) / Rref, 1))
        rows.append(row)
    return rows


def plot_smhm_models(z, logMh=None, models=_MODELS, ratio=False, ax=None):
    """
    Plot logM* vs logMhalo (or M*/Mh ratio) for several models at redshift z.

    >>> plot_smhm_models(2.3)                 # M* vs Mh
    >>> plot_smhm_models(2.3, ratio=True)     # M*/Mh ratio vs Mh
    """
    import matplotlib.pyplot as plt   # lazy: virial.py is imported hot-path-wide
                                       # for `cosmo` alone; keep matplotlib out
                                       # of that unless a plot is actually made.
    if logMh is None:
        logMh = np.linspace(10.5, 14.5, 200)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5))
    Mh = 10 ** logMh
    for m in models:
        Ms = _FORWARD[m](Mh, z)
        y = np.log10(Ms / Mh) if ratio else np.log10(Ms)
        ax.plot(logMh, y, lw=2, label=_LABELS[m])
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_*/M_{\rm halo})$' if ratio
                  else r'$\log_{10}(M_*/M_\odot)$')
    ax.set_title(f'SMHM models at z = {z:g}')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    return ax


def plot_rvir_vs_mstar(z, logMstar=None, models=_MODELS, ax=None):
    """
    Plot R200c [kpc] vs logM* for several models at redshift z -- i.e. exactly
    the quantity that sets your virial bins. Gaps = galaxy above median relation.

    >>> plot_rvir_vs_mstar(2.3)
    """
    import matplotlib.pyplot as plt   # lazy -- see plot_smhm_models
    if logMstar is None:
        logMstar = np.linspace(9.0, 11.0, 100)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5))
    for m in models:
        R = []
        for lm in logMstar:
            Mh = mhalo_from_mstar(10 ** lm, z, model=m)
            R.append(np.nan if np.isnan(Mh) else R200c_from_M200c(Mh, z))
        ax.plot(logMstar, R, lw=2, label=_LABELS[m])
    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$')
    ax.set_ylabel(r'$R_{200c}$ [kpc]')
    ax.set_title(f'Virial radius vs stellar mass at z = {z:g}')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    return ax
