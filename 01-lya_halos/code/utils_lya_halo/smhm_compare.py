"""
smhm_compare.py -- plotting & measuring helpers to compare SMHM models and see
what changing the relation does to Mhalo and R200c (and hence your R/Rvir bins).

Depends on: smhm_models.py, numpy, matplotlib, astropy.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G
from astropy.cosmology import Planck18 as cosmo

import smhm_models as sm

_MODELS = ['moster13', 'girelli20', 'behroozi19']
_LABELS = {'moster13': 'Moster+2013 (current)',
           'girelli20': 'Girelli+2020 (COSMOS)',
           'behroozi19': 'Behroozi+2019 (UM DR1)'}


# ---- shared physics: M200c -> R200c (matches your virial.py) ----------
def R200c_from_M200c(M200c, z):
    """R200c [kpc] from M200c [Msun] at redshift z (Planck18)."""
    M = np.atleast_1d(M200c) * u.Msun
    Hz = cosmo.H(z)
    R = ((G * M / (100 * Hz ** 2)) ** (1 / 3)).to(u.kpc).value
    return R if R.size > 1 else float(R[0])


# ---- MEASURING: table of Mhalo & R200c per model -----------------------
def compare_smhm(logMstar, z, models=_MODELS):
    """
    For one galaxy (logMstar, z), return a dict per model with logMhalo and
    R200c [kpc]. NaN where the galaxy is above a model's median relation.

    >>> compare_smhm(10.5, 2.3)
    """
    out = {}
    for m in models:
        Mh = sm.mhalo_from_mstar(10 ** logMstar, z, model=m)
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
            Mh = sm.mhalo_from_mstar(10 ** lm, z, model=m)
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


# ---- PLOTTING ---------------------------------------------------------
def plot_smhm_models(z, logMh=np.linspace(10.5, 14.5, 200), models=_MODELS,
                     ratio=False, ax=None):
    """
    Plot logM* vs logMhalo (or M*/Mh ratio) for several models at redshift z.

    >>> plot_smhm_models(2.3)                 # M* vs Mh
    >>> plot_smhm_models(2.3, ratio=True)     # M*/Mh ratio vs Mh
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5))
    Mh = 10 ** logMh
    for m in models:
        Ms = sm._FORWARD[m](Mh, z)
        y = np.log10(Ms / Mh) if ratio else np.log10(Ms)
        ax.plot(logMh, y, lw=2, label=_LABELS[m])
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_*/M_{\rm halo})$' if ratio
                  else r'$\log_{10}(M_*/M_\odot)$')
    ax.set_title(f'SMHM models at z = {z:g}')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    return ax


def plot_rvir_vs_mstar(z, logMstar=np.linspace(9.0, 11.0, 100),
                       models=_MODELS, ax=None):
    """
    Plot R200c [kpc] vs logM* for several models at redshift z -- i.e. exactly
    the quantity that sets your virial bins. Gaps = galaxy above median relation.

    >>> plot_rvir_vs_mstar(2.3)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5))
    for m in models:
        R = []
        for lm in logMstar:
            Mh = sm.mhalo_from_mstar(10 ** lm, z, model=m)
            R.append(np.nan if np.isnan(Mh) else R200c_from_M200c(Mh, z))
        ax.plot(logMstar, R, lw=2, label=_LABELS[m])
    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$')
    ax.set_ylabel(r'$R_{200c}$ [kpc]')
    ax.set_title(f'Virial radius vs stellar mass at z = {z:g}')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    return ax
