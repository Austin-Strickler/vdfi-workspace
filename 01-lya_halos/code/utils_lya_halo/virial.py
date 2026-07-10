"""
virial.py -- halo mass / virial radius and radial-bin conversions.

Chain: log Mstar --(Moster+2013)--> Mhalo (M200c) --> R200c --> bin edges.

The actual extraction converts PER GALAXY (virial_to_kpc_bins with that
galaxy's mass and z). median_virial_kpc_bins is a SEPARATE reporting helper
for the notebook -- it uses the sample median and is NOT what bins the data.
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


# ---- Moster+2013 stellar<->halo mass relation -----------------------
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
    """Invert Moster+2013. Mstar [Msun] -> Mhalo [Msun]."""
    def f(logMh):
        Mh = 10 ** logMh
        return np.log10(moster13_mstar_from_mhalo(Mh, z)) - np.log10(Mstar)
    logMh = brentq(f, 9, 15)
    return 10 ** logMh


# ---- M200c -> R200c -------------------------------------------------
def R200c_from_M200c(M200c, z):
    """R200c = [G M200c / (100 H(z)^2)]^(1/3), returned in kpc."""
    M200c = M200c * u.Msun
    Hz = cosmo.H(z)
    R200c = (G * M200c / (100 * Hz ** 2)) ** (1 / 3)
    return R200c.to(u.kpc).value


def estimate_M200c_R200c_from_Mstar(Mstar, z):
    """Mstar [Msun] -> (M200c [Msun], R200c [kpc]) via Moster+2013 + R200c(z)."""
    M200c = mhalo_from_mstar_moster13(Mstar, z)
    R200c = R200c_from_M200c(M200c, z)
    return M200c, R200c


def physical_kpc_to_arcsec(R_kpc, z):
    """Physical size R_kpc at redshift z -> angular size in arcsec (Planck18)."""    
    scale = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)
    return (R_kpc * u.kpc / scale).value


# ---- virial (R/Rvir) edges -> physical units ------------------------
def virial_to_kpc_bins(v_bins, m, z, m_min=7.0, m_max=11.0, verbose=False):
    """R/Rvir edges -> kpc edges, using THIS galaxy's log-mass m and z."""
    m = float(m)
    z = float(z)
    if not np.isfinite(m) or not np.isfinite(z) or z <= 0:
        return np.full_like(np.asarray(v_bins, dtype=float), np.nan)
    m_used = np.clip(m, m_min, m_max)
    if verbose and m_used != m:
        print(f"Capping logMstar {m:.2f} -> {m_used:.2f}, z={z:.3f}")
    _, r200c = estimate_M200c_R200c_from_Mstar(10 ** m_used, z)
    return np.asarray(v_bins, dtype=float) * r200c


def virial_to_angular_bins(v_bins, m, z, m_min=7.0, m_max=11.0,
                           max_arcsec=None, verbose=False):
    """R/Rvir edges -> arcsec edges, using THIS galaxy's log-mass m and z."""
    v_bins = np.asarray(v_bins, dtype=float)
    m = float(m)
    z = float(z)
    if not np.isfinite(m) or not np.isfinite(z) or z <= 0:
        return np.full_like(v_bins, np.nan)
    m_used = np.clip(m, m_min, m_max)
    if verbose and m_used != m:
        print(f"Capping logMstar {m:.2f} -> {m_used:.2f}, z={z:.3f}")
    _, r200c = estimate_M200c_R200c_from_Mstar(10 ** m_used, z)
    theta_r = physical_kpc_to_arcsec(r200c, z)
    angular_bins = v_bins * theta_r
    if max_arcsec is not None:
        angular_bins = np.minimum(angular_bins, max_arcsec)
    return angular_bins


def compute_annuli_from_virial(v_bins, m, z, kpc_bins=True):
    """Convenience dispatch: virial edges -> kpc (default) or arcsec."""
    if kpc_bins:
        return virial_to_kpc_bins(v_bins, m, z)
    return virial_to_angular_bins(v_bins, m, z)


# ---- reporting only (NOT used for extraction) -----------------------
def median_virial_kpc_bins(v_bins, run_table, mass_col='MASS_50', z_col='z'):
    """
    EXPLICIT median-galaxy mapping: R/Rvir edges -> kpc using the sample
    median mass and z. For plots/tables only -- the pipeline bins each galaxy
    with its own mass and z, not this.

    Returns (kpc_edges, m_median, z_median).
    """
    m_med = np.nanmedian(np.asarray(run_table[mass_col], dtype=float))
    z_med = np.nanmedian(np.asarray(run_table[z_col], dtype=float))
    return virial_to_kpc_bins(v_bins, m_med, z_med, verbose=False), m_med, z_med
