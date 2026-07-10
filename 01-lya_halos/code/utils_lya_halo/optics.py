"""
optics.py -- Greg's PSF / DAR / continuum helpers.

SCOPE NOTE: per pipeline_plan_v2.md, the *binned* extraction (current paper)
does NOT use Moffat point-source weighting or the DAR terms. These functions
are the PSF-weighted optimal-extraction path, kept here as a toolbox for the
deferred core-extraction / escape-fraction work. The only one the binned
loader touches is moffat_psf_integration, and only when build_psf=True.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from astropy.modeling.models import Moffat2D
from astropy.stats import biweight_location as biweight


def manual_convolution(a, G, error=False):
    """
    Convolve 1-D array `a` with kernel `G` (object exposing `G.array`) by explicit
    shifted-copy accumulation, zero-filling at the edges. error=False returns
    sum(a_shift * G); error=True propagates in quadrature, sqrt(sum(a_shift^2 *
    G^2)), for combining error spectra.
    """
    b = np.zeros((len(a), len(G.array)))
    N = int(len(G.array) / 2. - 0.5)
    M = len(G.array)
    for i in np.arange(0, N):
        b[(N - i):, i] = a[:-(N - i)]
    b[:, N] = a
    for i in np.arange(N + 1, M):
        b[:-(i - N), i] = a[(i - N):]
    if error:
        return np.sqrt(np.nansum(b ** 2 * G.array ** 2, axis=1))
    return np.nansum(b * G.array, axis=1)


def pca_fit(H, data):
    """
    Least-squares projection of `data` onto the basis rows of `H` (e.g. PCA sky
    eigenvectors), ignoring non-finite samples. Returns the reconstruction
    H.T @ coeffs on the full grid.
    """
    sel = np.isfinite(data)
    sol = np.linalg.lstsq(H.T[sel], data[sel])[0]
    res = np.dot(H.T, sol)
    return res


def get_continuum(spectra, nbins=25):
    """Get continuum from sky-subtracted spectra (biweight in nbins chunks)."""
    a = np.array([biweight(f, axis=1, ignore_nan=True)
                  for f in np.array_split(spectra, nbins, axis=1)]).swapaxes(0, 1)
    x = np.array([np.mean(xi)
                  for xi in np.array_split(np.arange(spectra.shape[1]), nbins)])
    cont = np.zeros(spectra.shape)
    X = np.arange(spectra.shape[1])
    for i, ai in enumerate(a):
        sel = np.isfinite(ai)
        if np.sum(sel) > nbins / 2.:
            I = interp1d(x[sel], ai[sel], kind='quadratic',
                         fill_value=np.nan, bounds_error=False)
            cont[i] = I(X)
        else:
            cont[i] = 0.0
    return cont


def wavelength_corrected_seeing(seeing, wave):
    """Correct the seeing FWHM as a function of wavelength."""
    return seeing * (wave / 4500.) ** (-1. / 5.)


def moffat_psf_integration(r, seeing, boxsize=14., scale=0.05, alpha=3.5):
    """Moffat PSF profile integrated onto fibers -> (interpolator, R, S, V)."""
    xl, xh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
    yl, yh = (0. - boxsize / 2., 0. + boxsize / 2. + scale)
    x, y = (np.arange(xl, xh, scale), np.arange(yl, yh, scale))
    xgrid, ygrid = np.meshgrid(x, y)
    V = np.zeros((len(seeing), len(r)))
    for j, fwhm in enumerate(seeing):
        M = Moffat2D()
        M.alpha.value = alpha
        M.gamma.value = 0.5 * fwhm / np.sqrt(2 ** (1. / M.alpha.value) - 1.)
        Z = M(xgrid, ygrid)
        Z = Z / Z.sum()
        for i, ri in enumerate(r):
            d = np.sqrt((xgrid - ri) ** 2 + (ygrid - 0.) ** 2)
            sel = d <= 0.75
            adj = np.pi * 0.75 ** 2 / (sel.sum() * scale ** 2)
            V[j, i] = np.sum(Z[sel]) * adj
    R, S = np.meshgrid(r, seeing)
    R, S, V = [arr.ravel() for arr in [R, S, V]]
    interp = LinearNDInterpolator(list(zip(R, S)), V)
    return interp, R, S, V


def get_spectrum_exposure(spectra, error, dra, ddec, seeing, PSF, wave,
                          goodpix_thresh=0.4):
    """PSF-weighted optimal extraction for a single exposure (core path)."""
    r = np.sqrt(dra ** 2 + ddec ** 2)
    seeing = wavelength_corrected_seeing(seeing, wave)
    Seeing = np.ones((r.shape[0],))[:, np.newaxis] * seeing[np.newaxis, :]
    weights = PSF(r, Seeing)
    var = error ** 2
    spectrum = (np.nansum(spectra / var * weights, axis=0) /
                np.nansum(weights ** 2 / var, axis=0))
    spectrum_error = np.sqrt(np.nansum(weights, axis=0) /
                             np.nansum(weights ** 2 / var, axis=0))

    mask = np.isfinite(spectra)
    summed_weights = np.nansum(weights * mask, axis=0)
    xi = [(np.mean(wi) - 4500.) / 1000. for wi in np.array_split(wave, 21)]
    yi = [np.nanmedian(wi) for wi in np.array_split(summed_weights, 21)]
    p0 = np.polyfit(xi, yi, 3)
    pfit = np.polyval(p0, (wave - 4500.) / 1000.)
    mask = summed_weights / pfit < goodpix_thresh
    spectrum[mask] = np.nan
    spectrum_error[mask] = np.nan
    if np.median(pfit) < 0.1:
        spectrum = np.nan
        spectrum_error = np.nan
    return spectrum, spectrum_error, summed_weights, pfit
