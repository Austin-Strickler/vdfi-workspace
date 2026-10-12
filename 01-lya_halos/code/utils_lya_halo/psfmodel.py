"""
psfmodel.py -- the VIRUS PSF as a model you can evaluate at any redshift,
with honest errors.

    from utils_lya_halo import psfmodel as pm

    cfg = pm.PSFConfig()                       # every knob, defaults = settled choices
    cal = pm.calibrate(star_both, cfg)         # stars -> FWHM(lambda), beta
    res = pm.apply(cal, z_gal)                 # -> arcsec + kpc kernels, incl. splits
    err = pm.bootstrap(cal, z_gal)             # -> error bars on all of the above

`calibrate` caches the expensive part, so `apply` and `bootstrap` are cheap and
can be re-run with different redshift samples, splits, or beta without touching
the stars again.

WHY THE ERRORS WORK THIS WAY
----------------------------
Every stage shares the same stars, so no stage's error is independent of any
other's. The only honest answer is to resample the stars ONCE, at the top, and
push each draw through the ENTIRE chain -- composite, beta, FWHM(lambda), the
DAR fit, the kernels -- then take the spread of the final answers.

Two things make that affordable and correct:

  * `radial_continuum_profile` returns per_obj (n_star x n_rad). Resampling
    stars does not change that array, only which ROWS are selected. So the
    spectral work runs once per slice, and a draw is index selection plus a
    biweight. That is the difference between a 30-second bootstrap and an
    overnight one.

  * The per-bin errors are only the WEIGHTS in the Moffat fit. They decide
    which radii the fit listens to; they are not the source of the final error
    bars. So they are computed once from the full sample and FROZEN. Weights
    that are not statistically optimal make the estimator slightly less
    efficient; they do not bias it, and the spread across draws is still the
    honest uncertainty on the estimator you actually ran. (This is why the
    known bin-to-bin covariance problem is a DIAGNOSTIC -- it governs chi2/dof
    and hence whether a single Moffat is the right model -- and not a blocker
    on the error bars.)

The same star indices are used for all nine slices in a draw, because it is the
same stars observed at every wavelength; that is what propagates slice-to-slice
correlation. And every kernel a draw produces -- full sample, low-z, high-z --
comes from that one draw, so the low/high kernels share their PSF error almost
entirely and the error on their DIFFERENCE is far smaller than on either alone.
Computing them from separate bootstraps would badly overstate it.

WHAT THE BOOTSTRAP CANNOT SEE
-----------------------------
Resampling stars reveals variation BETWEEN stars. Anything common to all of
them is invisible to it: background-subtraction systematics, the Moffat form
being wrong, the effective-kernel convention, the quadrature approximation in
the DAR model. Quote those separately -- `systematics()` runs the standard set
(vary r_max, the normalization annulus, the magnitude split, beta). Report a
statistical error AND a systematic range, never one combined bar that hides
which is which.

CONVENTIONS THAT MATTER
-----------------------
* This is the EFFECTIVE PSF: an analytic Moffat fitted to fiber data, so the
  1.5" fiber aperture is already folded in. Do NOT add a fiber operator
  downstream. The quoted FWHM is not the delivered seeing.
* The PSF is a fixed ANGULAR blur; only the arcsec->kpc scale is per-galaxy.
* Kernels are built directly in the target unit, so the Moffat's prefactor
  supplies the 1/s^2 Jacobian. Every kernel is still asserted to integrate to 1.
* Galaxies combine UNWEIGHTED: stack.coadd_galaxies' biweight ignores its
  `weights` argument (only 'weighted_median' uses it).

See specs/psf-model.md for the full method and the open items.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import biweight_location

import astropy.units as u
from utils_lya_halo.virial import cosmo            # Planck18, as stack.py uses

_trapz = getattr(np, "trapezoid", None) or np.trapz      # np.trapezoid is NumPy >= 2

Q = chr(34)          # a double quote; f-strings pre-3.12 reject backslashes
LYA_REST = 1215.67
DEFAULT_BANDS = ((3500, 3790), (3790, 4000), (4000, 4200), (4200, 4400),
                 (4400, 4600), (4600, 4800), (4800, 5000), (5000, 5200),
                 (5200, 5500))

__all__ = ["PSFConfig", "calibrate", "DEFAULT_OUTPUT_DIR", "apply", "bootstrap", "systematics",
           "adopt_bootstrap_errors", "recalibrate", "plot_kernels",
           "beta_sensitivity", "run_psf", "PSFCalibration",
           "analytic_moffat", "edlen_refractivity", "fwhm_model_dar",
           "kpc_per_arcsec", "DEFAULT_BANDS"]



# ---------------------------------------------------------------------
# Where things are written. Relative to the working directory, which in
# Jupyter is the notebook's own folder.
# ---------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "outputs_starpsf"
CALIBRATION_JSON = "psf_calibration.json"

# PSFConfig fields that change the cached slices or the frozen inner errors.
# Anything else can change without invalidating the cache (see recalibrate).
_CACHE_FIELDS = ("bands", "clip", "norm_annulus", "norm_weight", "mag_split",
                 "faint_max", "bright_min", "n_boot_inner", "seed")


def _out_path(path, default_name, output_dir):
    """None -> output_dir/default_name; a bare filename -> output_dir/filename;
    anything with a directory component is used as given."""
    if path is None:
        path = default_name
    if not os.path.dirname(str(path)):
        path = os.path.join(output_dir, str(path))
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return str(path)


def _migrate_legacy_json(output_dir, verbose=True):
    """Older runs wrote psf_calibration.json next to the notebook. Move it
    into output_dir. If output_dir already has one, keep both: the stray copy
    is moved in as psf_calibration_legacy.json rather than overwriting."""
    src = CALIBRATION_JSON
    if not os.path.isfile(src):
        return None
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, CALIBRATION_JSON)
    if os.path.exists(dst):
        dst = os.path.join(output_dir, "psf_calibration_legacy.json")
        k = 1
        while os.path.exists(dst):
            dst = os.path.join(output_dir, f"psf_calibration_legacy{k}.json")
            k += 1
    shutil.move(src, dst)
    if verbose:
        print(f"moved ./{src} -> {dst}")
    return dst


def _star_fingerprint(star_product, gmag, radial_profile):
    """Cheap identity check for the star sample: shapes, the wavelength and
    radial grids, the G magnitudes, and a subsample of the spectra/fiber
    counts. Enough to notice a different or re-extracted star file."""
    h = hashlib.sha1()
    spec = star_product.spec
    h.update(repr(tuple(spec.shape)).encode())
    h.update(np.ascontiguousarray(np.asarray(spec[:, :, ::50], float)).tobytes())
    nfib = getattr(star_product, "nfib", None)
    if nfib is not None:
        h.update(np.ascontiguousarray(np.asarray(nfib, float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(star_product.wave, float)).tobytes())
    r_edges = getattr(star_product, "r_edges", None)
    if r_edges is not None:
        h.update(np.ascontiguousarray(np.asarray(r_edges, float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(gmag, float)).tobytes())
    h.update(f"{getattr(radial_profile, '__module__', '')}."
             f"{getattr(radial_profile, '__qualname__', repr(radial_profile))}".encode())
    return h.hexdigest()


def _cache_key(star_product, gmag, cfg, radial_profile):
    d = cfg.to_dict()
    knobs = json.dumps({k: d[k] for k in _CACHE_FIELDS}, sort_keys=True)
    h = hashlib.sha1()
    h.update(knobs.encode())
    h.update(_star_fingerprint(star_product, gmag, radial_profile).encode())
    return h.hexdigest()[:12], knobs


def _save_slice_cache(path, key, knobs, caches, errs, faint, bright):
    arrs = {"faint": faint, "bright": bright, "n_bands": np.array(len(caches)),
            "key": np.array(key), "knobs": np.array(knobs)}
    for i, (c, e) in enumerate(zip(caches, errs)):
        arrs[f"band_{i}"] = np.asarray(c.band, float)
        arrs[f"r_mid_{i}"] = c.r_mid
        arrs[f"p_{i}"] = c.p
        arrs[f"faint_idx_{i}"] = c.faint_idx
        arrs[f"bright_idx_{i}"] = c.bright_idx
        arrs[f"err_{i}"] = e
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)                  # never leave a half-written cache


def _load_slice_cache(path, key):
    with np.load(path, allow_pickle=False) as z:
        if str(z["key"]) != key:
            raise ValueError(f"{path} was written for a different key")
        caches, errs = [], []
        for i in range(int(z["n_bands"])):
            band = tuple(float(x) for x in z[f"band_{i}"])
            band = tuple(int(x) if x.is_integer() else x for x in band)
            caches.append(_SliceCache(band, z[f"r_mid_{i}"], z[f"p_{i}"],
                                      z[f"faint_idx_{i}"], z[f"bright_idx_{i}"]))
            errs.append(z[f"err_{i}"])
        return caches, errs, z["faint"], z["bright"]


# =====================================================================
# 0. Configuration -- every knob in one place
# =====================================================================
@dataclass
class PSFConfig:
    """
    Everything that is a CHOICE rather than data. Changing any of these changes
    the answer, so they travel together, get stored in the result, and get
    written to JSON alongside it.

    Vary them with `replace(cfg, mag_split=16.0)` -- and note that their effect
    is a SYSTEMATIC, measured by re-running, not something the bootstrap sees.
    `systematics()` runs the standard set for you.
    """
    # --- extraction / slicing ---
    bands: tuple = DEFAULT_BANDS          # observed-frame windows, one fit each
    clip: float = 3.0                     # sigma-clip in the per-star band mean

    # --- per-star normalization (see spec 2) ---
    norm_annulus: tuple = (2.0, 4.0)      # arcsec; wide, and outside saturation
    norm_weight: str = "nfib"             # 'nfib' (area-like) or 'none'

    # --- magnitude classes (see spec 3) ---
    mag_split: float = 17.0               # faint/bright boundary in Gaia G
    faint_max: float = 21.0
    bright_min: float | None = None
    min_stars: int = 3                    # per bin, for a class to be valid
    apply_stitch: bool = False            # measure the cross-scale, do NOT apply

    # --- the Moffat fit (see spec 4) ---
    beta: object = "measure"              # "measure" (two-pass) or a float
    fit_r_min: float | None = None
    fit_r_max: float = 12.0               # outer limit; vary it for the systematic

    # --- FWHM(lambda) model (see spec 5.2) ---
    kolm: float = -0.2                    # Kolmogorov seeing exponent, held
    lam_norm: float = 4600.0

    # --- errors ---
    n_boot_inner: int = 300               # per-bin errors -> fit WEIGHTS (frozen)
    n_boot_outer: int = 300               # the actual error bars
    seed: int = 1234

    # --- kernels ---
    lya: float = LYA_REST
    r_arcsec: tuple = (0.05, 40.0, 400)   # (lo, hi, n) geomspace
    r_kpc: tuple = (0.5, 400.0, 500)

    def grid(self, unit):
        lo, hi, n = self.r_arcsec if unit == "arcsec" else self.r_kpc
        return np.geomspace(lo, hi, int(n))

    @property
    def beta_is_measured(self):
        return isinstance(self.beta, str) and self.beta.lower() == "measure"

    def to_dict(self):
        d = asdict(self)
        d["bands"] = [list(b) for b in self.bands]
        d["norm_annulus"] = list(self.norm_annulus)
        d["r_arcsec"] = list(self.r_arcsec); d["r_kpc"] = list(self.r_kpc)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["bands"] = tuple(tuple(b) for b in d["bands"])
        d["norm_annulus"] = tuple(d["norm_annulus"])
        d["r_arcsec"] = tuple(d["r_arcsec"]); d["r_kpc"] = tuple(d["r_kpc"])
        return cls(**d)


# =====================================================================
# 1. Profile models
# =====================================================================
def analytic_moffat(r, fwhm, beta=3.0):
    """Moffat surface brightness, unit TOTAL flux in whatever unit r and fwhm
    share -- so a kernel built directly in kpc needs no explicit Jacobian."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    fwhm = float(fwhm); beta = float(beta)
    if fwhm <= 0 or beta <= 1.0:
        return np.full(r.shape, np.nan)
    gamma = 0.5*fwhm/np.sqrt(2.0**(1.0/beta) - 1.0)
    return (beta - 1.0)/(np.pi*gamma**2)*(1.0 + (r/gamma)**2)**(-beta)


def edlen_refractivity(lam_A):
    """(n-1)*1e8 for standard air (Edlen). Steep blueward -- the shape the
    measured FWHM(lambda) has, and why it is not a power law."""
    s = 1e4/np.asarray(lam_A, dtype=float)
    return 8342.13 + 2406030.0/(130.0 - s**2) + 15997.0/(38.9 - s**2)


def fwhm_model_dar(lam, F0, tanz, lam_ref, kolm=-0.2, lam_norm=4600.0):
    """Seeing and differential atmospheric refraction, in quadrature.
    tanz absorbs the zenith distance AND the order-unity factor between a
    parallactic-smeared arc and an isotropic blur -- a consistency check
    against HET's fixed 35 deg, never a measurement of it."""
    seeing = float(F0)*(np.asarray(lam, float)/lam_norm)**float(kolm)
    dar = 206265.0*(edlen_refractivity(lam)
                    - edlen_refractivity(lam_ref))*1e-8*float(tanz)
    return np.sqrt(seeing**2 + dar**2)


def _powerlaw(lam, a, q, lam_norm=4600.0):
    return a*(np.asarray(lam, float)/lam_norm)**q


def kpc_per_arcsec(z):
    """Proper kpc per arcsec, matching stack.py's convention exactly."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.array([cosmo.angular_diameter_distance(zi).to(u.kpc).value/206265
                     for zi in z])


def _fit_moffat(r, y, e, beta_fixed=None):
    """Log-space weighted Moffat fit. Returns (fwhm, beta, errs, chi2_dof) or
    None if it will not converge. Log space because the profile spans decades;
    a linear fit is a fit to the inner two bins."""
    ok = np.isfinite(r) & np.isfinite(y) & np.isfinite(e) & (y > 0) & (e > 0)
    if ok.sum() < (3 if beta_fixed is None else 2) + 1:
        return None
    r, y, e = r[ok], y[ok], e[ok]
    ly, ls = np.log(y), e/y
    try:
        if beta_fixed is None:
            def f(rr, la, fw, be):
                return la + np.log(analytic_moffat(rr, fw, be))
            p, c = curve_fit(f, r, ly, p0=[ly[0]+1.0, 1.5, 3.0], sigma=ls,
                             absolute_sigma=True, maxfev=20000,
                             bounds=([-np.inf, 0.2, 1.05], [np.inf, 10.0, 12.0]))
            fwhm, beta = float(p[1]), float(p[2])
            errs = np.sqrt(np.diag(c))
            fe, be_ = float(errs[1]), float(errs[2])
            npar = 3
        else:
            b0 = float(beta_fixed)
            def f(rr, la, fw):
                return la + np.log(analytic_moffat(rr, fw, b0))
            p, c = curve_fit(f, r, ly, p0=[ly[0]+1.0, 1.5], sigma=ls,
                             absolute_sigma=True, maxfev=20000,
                             bounds=([-np.inf, 0.2], [np.inf, 10.0]))
            fwhm, beta = float(p[1]), b0
            fe, be_ = float(np.sqrt(np.diag(c))[1]), 0.0
            npar = 2
        resid = (ly - f(r, *p))/ls
        dof = max(r.size - npar, 1)
        return {"fwhm": fwhm, "beta": beta, "fwhm_err": fe, "beta_err": be_,
                "chi2_dof": float((resid**2).sum()/dof), "n_bins": int(r.size),
                "log_amp": float(p[0])}
    except Exception:
        return None


# =====================================================================
# 2. The slice cache -- the expensive part, computed once
# =====================================================================
class _SliceCache:
    """Per-star normalized profiles for one wavelength slice, plus the class
    memberships. Resampling stars selects ROWS of `p`; nothing here is
    recomputed, which is what makes the outer bootstrap affordable."""

    def __init__(self, band, r_mid, p, faint_idx, bright_idx):
        self.band = tuple(band)
        self.lam = 0.5*(band[0] + band[1])
        self.r_mid = r_mid
        self.p = p                       # (n_star, n_rad), already normalized
        self.faint_idx = faint_idx
        self.bright_idx = bright_idx

    def class_curves(self, sel_faint=None, sel_bright=None, min_stars=3):
        """Biweight per class over the selected star rows. `sel_*` are index
        arrays into the FULL star list (a bootstrap draw); None = use all."""
        out = []
        for idx, sel in ((self.faint_idx, sel_faint), (self.bright_idx, sel_bright)):
            rows = idx if sel is None else sel
            blk = self.p[rows]
            with np.errstate(invalid="ignore"):
                cen = biweight_location(blk, axis=0, ignore_nan=True)
            n = np.isfinite(blk).sum(axis=0)
            cen = np.where(n >= min_stars, cen, np.nan)
            out.append(cen)
        return np.vstack(out)            # (2, n_rad): [faint, bright]


def _build_cache(star_product, gmag, cfg, radial_profile, verbose):
    """Run the extraction once per slice and normalize each star on the
    annulus. Everything the bootstrap needs afterwards lives in here."""
    gmag = np.asarray(gmag, float)
    nstar = star_product.spec.shape[0]
    if gmag.size != nstar:
        raise ValueError(f"gmag has {gmag.size} entries, product has {nstar} stars.")

    faint = np.flatnonzero(np.isfinite(gmag) & (gmag >= cfg.mag_split)
                           & (gmag <= cfg.faint_max))
    bright_m = np.isfinite(gmag) & (gmag < cfg.mag_split)
    if cfg.bright_min is not None:
        bright_m &= gmag >= float(cfg.bright_min)
    bright = np.flatnonzero(bright_m)
    if faint.size < cfg.min_stars or bright.size < cfg.min_stars:
        raise ValueError(f"need >= {cfg.min_stars} stars per class; got "
                         f"faint={faint.size}, bright={bright.size} at "
                         f"mag_split={cfg.mag_split}.")

    caches = []
    for band in cfg.bands:
        prof = radial_profile(star_product, band=tuple(band), combine="biweight",
                              clip=cfg.clip)
        per_obj = np.asarray(prof["per_obj"], float)
        r_mid = np.asarray(prof["r_mid"], float)

        lo, hi = cfg.norm_annulus
        in_ann = (r_mid >= lo) & (r_mid <= hi)
        if not np.any(in_ann):
            raise ValueError(f"no radial bin in the {lo}-{hi}\" normalization "
                             f"annulus for band {tuple(band)}.")
        if cfg.norm_weight == "nfib":
            w = np.asarray(prof["nfib_med"], float)[in_ann]
        elif cfg.norm_weight == "none":
            w = np.ones(int(in_ann.sum()))
        else:
            raise ValueError("norm_weight must be 'nfib' or 'none'.")

        blk = per_obj[:, in_ann]
        good = np.isfinite(blk)
        den = np.where(good, w[None, :], 0.0).sum(axis=1)
        num = np.where(good, blk*w[None, :], 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.where(den > 0, num/den, np.nan)
        bad = ~(np.isfinite(norm) & (norm > 0))
        p = per_obj/np.where(bad, np.nan, norm)[:, None]

        caches.append(_SliceCache(band, r_mid, p,
                                  faint[~bad[faint]], bright[~bad[bright]]))
        if verbose:
            print(f"  cached {tuple(band)}  faint={caches[-1].faint_idx.size} "
                  f"bright={caches[-1].bright_idx.size}")
    return caches, faint, bright


def _inner_errors(caches, cfg, rng, verbose):
    """Per-class, per-bin errors by bootstrapping the star sample. Computed
    ONCE from the full sample and frozen as fit weights.

    Not scatter/sqrt(N): the per-star normalization forces every star to agree
    inside the annulus by construction, so the raw scatter there is spuriously
    tiny and would anchor every fit on the normalization annulus."""
    errs = []
    for c in caches:
        e_cls = []
        for idx in (c.faint_idx, c.bright_idx):
            draws = np.empty((cfg.n_boot_inner, c.r_mid.size))
            for b in range(cfg.n_boot_inner):
                rows = idx[rng.integers(0, idx.size, idx.size)]
                with np.errstate(invalid="ignore"):
                    draws[b] = biweight_location(c.p[rows], axis=0, ignore_nan=True)
            with np.errstate(invalid="ignore"):
                e = np.nanstd(draws, axis=0)
            e_cls.append(np.where(np.isfinite(e) & (e > 0), e, np.nan))
        errs.append(np.vstack(e_cls))
        if verbose:
            print(f"  inner errors {c.band}: done")
    return errs


def _composite(curves, errs, cfg):
    """Two class curves -> one composite. The cross-scale between classes is
    MEASURED (both share the normalization annulus, so it is a check) and by
    default NOT applied -- a value away from 1 is a result, not a nuisance.

    Inverse-variance combine keeping EVERY finite bin. No S/N gate: gating on
    signal is a selection on noise and manufactures a positive floor exactly
    where the wings are.
    """
    valid = np.isfinite(curves) & np.isfinite(errs) & (errs > 0) & (curves > 0)
    ov = valid.all(axis=0)
    stitch = np.nan
    if ov.sum() >= 2:
        lr = np.log(curves[1][ov]/curves[0][ov])          # bright / faint
        ls = np.sqrt((errs[1][ov]/curves[1][ov])**2 + (errs[0][ov]/curves[0][ov])**2)
        w = 1.0/ls**2
        stitch = float(np.exp(np.sum(w*lr)/np.sum(w)))

    cur = curves.copy()
    if cfg.apply_stitch and np.isfinite(stitch):
        cur[1] = cur[1]/stitch

    n = curves.shape[1]
    psf = np.full(n, np.nan); psf_err = np.full(n, np.nan)
    for i in range(n):
        m = valid[:, i]
        if not m.any():
            continue
        w = 1.0/errs[m, i]**2
        psf[i] = np.sum(w*cur[m, i])/np.sum(w)
        psf_err[i] = 1.0/np.sqrt(np.sum(w))
    return psf, psf_err, stitch, int(ov.sum())


def _one_pass(caches, errs, cfg, sel_f=None, sel_b=None, beta_fixed=None):
    """One full pass over the slices at a given beta. Returns per-slice dicts."""
    rows = []
    for c, e in zip(caches, errs):
        cur = c.class_curves(sel_f, sel_b, min_stars=cfg.min_stars)
        psf, psf_err, stitch, n_ov = _composite(cur, e, cfg)
        r = c.r_mid
        m = np.ones(r.size, bool)
        if cfg.fit_r_min is not None:
            m &= r >= cfg.fit_r_min
        if cfg.fit_r_max is not None:
            m &= r <= cfg.fit_r_max
        f = _fit_moffat(r[m], psf[m], psf_err[m], beta_fixed=beta_fixed)
        if f is None:
            continue
        f.update({"band": c.band, "lam": c.lam, "stitch": stitch,
                  "n_overlap": n_ov, "r_mid": r, "psf": psf, "psf_err": psf_err})
        rows.append(f)
    return rows


def _beta_star(rows):
    """Inverse-variance mean of the per-slice beta, plus the flatness chi2.
    Uses all slices rather than one window: more S/N, and the scatter about the
    mean IS the evidence that fixing beta is legitimate."""
    b = np.array([d["beta"] for d in rows], float)
    e = np.array([d["beta_err"] for d in rows], float)
    ok = np.isfinite(b) & np.isfinite(e) & (e > 0)
    if ok.sum() < 2:
        return float(np.nanmedian(b)), np.nan
    b, e = b[ok], e[ok]
    m = float(np.sum(b/e**2)/np.sum(1/e**2))
    return m, float(np.sum(((b - m)/e)**2)/(b.size - 1))


def _dar_fit(lam, fw, fe, cfg):
    def f(l, F0, tanz, lam_ref):
        return fwhm_model_dar(l, F0, tanz, lam_ref, kolm=cfg.kolm,
                              lam_norm=cfg.lam_norm)
    p, c = curve_fit(f, lam, fw, p0=[float(np.median(fw)), 0.7, 4800.0],
                     sigma=fe, absolute_sigma=True,
                     bounds=([0.3, 0.0, 3500.0], [8.0, 5.0, 7500.0]), maxfev=40000)
    resid = (fw - f(lam, *p))/fe
    dof = max(lam.size - 3, 1)
    return np.asarray(p, float), np.sqrt(np.diag(c)), float((resid**2).sum()/dof)


# =====================================================================
# 3. Calibration
# =====================================================================
class PSFCalibration:
    """The calibrated PSF plus the cache that produced it. `apply` and
    `bootstrap` both run off this; neither touches the star spectra again."""

    def __init__(self, *, cfg, caches, errs, rows, beta, beta_chi2_dof,
                 beta_rows, dar, dar_err, dar_chi2_dof, powerlaw, pl_chi2_dof,
                 faint_idx, bright_idx):
        self.cfg = cfg
        self._caches = caches
        self._errs = errs
        self.rows = rows
        self.beta = float(beta)
        self.beta_chi2_dof = float(beta_chi2_dof)
        self.beta_rows = beta_rows
        self.dar = np.asarray(dar, float)
        self.dar_err = np.asarray(dar_err, float)
        self.dar_chi2_dof = float(dar_chi2_dof)
        self.powerlaw = np.asarray(powerlaw, float)
        self.pl_chi2_dof = float(pl_chi2_dof)
        self.faint_idx = faint_idx
        self.bright_idx = bright_idx
        self.lam_range = (float(min(d["lam"] for d in rows)),
                          float(max(d["lam"] for d in rows)))
        # Set by adopt_bootstrap_errors(): the measured per-slice FWHM scatter,
        # which replaces the provisional inflated formal errors.
        self.fwhm_err = np.array([d["fwhm_err"]*np.sqrt(d["chi2_dof"])
                                  for d in rows], float)
        self.fwhm_err_source = "formal x sqrt(chi2/dof) -- PROVISIONAL"

    # ---- the calibration curve ----
    def fwhm(self, lam, dar=None):
        p = self.dar if dar is None else dar
        return fwhm_model_dar(lam, *p, kolm=self.cfg.kolm, lam_norm=self.cfg.lam_norm)

    def fwhm_at_z(self, z, dar=None):
        z = np.atleast_1d(np.asarray(z, float))
        return self.fwhm(self.cfg.lya*(1.0 + z), dar=dar)

    def n_extrapolated(self, z):
        lam = self.cfg.lya*(1.0 + np.atleast_1d(np.asarray(z, float)))
        lo, hi = self.lam_range
        return int(np.sum((lam < lo) | (lam > hi)))

    # ---- reporting ----
    def describe(self):
        p, e = self.dar, self.dar_err
        c = self.cfg
        print("PSF calibration  (EFFECTIVE PSF -- the 1.5\" fiber aperture is "
              "already folded in;\n  this is NOT the delivered seeing)")
        print(f"  classes         : faint G {c.mag_split:g}-{c.faint_max:g} "
              f"(N={self.faint_idx.size}), bright G < {c.mag_split:g} "
              f"(N={self.bright_idx.size})")
        print(f"  normalization   : {c.norm_annulus[0]:g}-{c.norm_annulus[1]:g}" + Q + " "
              f"annulus, weight={c.norm_weight}")
        print(f"  fit             : analytic Moffat, log space, r < {c.fit_r_max:g}" + Q)
        if c.beta_is_measured:
            print(f"  beta            = {self.beta:.3f}   [measured: inverse-variance "
                  f"mean of {len(self.beta_rows)} slices, flatness chi2/dof = "
                  f"{self.beta_chi2_dof:.2f}]")
        else:
            print(f"  beta            = {self.beta:.3f}   [HELD at the configured value]")
        print(f"  FWHM(lambda)    = sqrt( seeing^2 + DAR^2 )")
        print(f"     F0           = {p[0]:.3f} +/- {e[0]:.3f} arcsec at {c.lam_norm:.0f} A")
        print(f"     tan z        = {p[1]:.3f} +/- {e[1]:.3f}  "
              f"(z = {np.degrees(np.arctan(p[1])):.1f} deg; HET is fixed at 35 deg "
              f"-- consistency, not a measurement)")
        print(f"     lam_ref      = {p[2]:.0f} +/- {e[2]:.0f} A")
        print(f"     chi2/dof     = {self.dar_chi2_dof:.2f}   "
              f"(single power law: {self.pl_chi2_dof:.2f})")
        print(f"  calibrated over {self.lam_range[0]:.0f}-{self.lam_range[1]:.0f} A")
        print(f"  slice FWHM errors: {self.fwhm_err_source}")
        if "PROVISIONAL" in self.fwhm_err_source:
            print("  -> run bootstrap() then adopt_bootstrap_errors(cal, boot); "
                  "until then the\n     +/- above are NOT citable and chi2/dof is "
                  "meaningless.")

    def table(self):
        print(f"{'band':>14}{'lam':>7}{'FWHM':>10}{'beta':>9}{'chi2/dof':>10}"
              f"{'stitch':>9}{'bins':>6}")
        for d in self.rows:
            print(f"{str(d['band']):>14}{d['lam']:7.0f}{d['fwhm']:10.3f}"
                  f"{d['beta']:9.3f}{d['chi2_dof']:10.2f}{d['stitch']:9.3f}"
                  f"{d['n_bins']:6d}")
        if self.cfg.beta_is_measured and self.beta_rows:
            print(f"\n  beta free per slice: " + "  ".join(
                f"{d['beta']:.3f}+/-{d['beta_err']:.3f}" for d in self.beta_rows))
            print(f"  -> weighted mean {self.beta:.3f}, flatness chi2/dof "
                  f"{self.beta_chi2_dof:.2f} "
                  + ("(flat: fixing beta is justified)" if self.beta_chi2_dof < 2
                     else "(NOT flat: beta varies with wavelength)"))

    def plot_lambda(self, figsize=(7.4, 5.8), lam_mark=(3647, 4863), save_fig=False):
        lam = np.array([d["lam"] for d in self.rows])
        fw = np.array([d["fwhm"] for d in self.rows])
        fe = self.fwhm_err
        fig, (ax, axr) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
        lf = np.linspace(min(3400, lam.min()*0.97), lam.max()*1.03, 400)
        ax.errorbar(lam, fw, fe, fmt="ko", capsize=3, ms=5, zorder=5, label="measured slices")
        ax.plot(lf, self.fwhm(lf), "r-", lw=1.7,
                label=f"seeing + DAR ($\\chi^2$/dof={self.dar_chi2_dof:.2f})")
        ax.plot(lf, self.dar[0]*(lf/self.cfg.lam_norm)**self.cfg.kolm, "r:", lw=1.1,
                label="seeing term alone")
        ax.plot(lf, _powerlaw(lf, *self.powerlaw, lam_norm=self.cfg.lam_norm), "b--",
                lw=1.2, label=f"power law ($\\chi^2$/dof={self.pl_chi2_dof:.2f})")
        ax.axvspan(lf.min(), self.lam_range[0], color="0.93", zorder=0)
        for L in (lam_mark or ()):
            ax.axvline(L, color="green", ls="-.", lw=1.0)
        ax.set_ylabel('FWHM ["]'); ax.legend(fontsize=8.5, frameon=False)
        axr.axhspan(-1, 1, color="0.88"); axr.axhline(0, color="k", lw=0.8)
        axr.plot(lam, (fw - self.fwhm(lam))/fe, "ro", ms=5, label="DAR")
        axr.plot(lam, (fw - _powerlaw(lam, *self.powerlaw,
                                      lam_norm=self.cfg.lam_norm))/fe,
                 "bs", mfc="none", ms=5, label="power law")
        axr.set_xlabel("observed wavelength [Å]"); axr.set_ylabel(r"resid [$\sigma$]")
        axr.legend(fontsize=8, frameon=False, ncol=2)
        if save_fig:
            fig.savefig(save_fig if isinstance(save_fig, str) else "psf_lambda.png",
                        dpi=150, bbox_inches="tight")
        return fig, (ax, axr)

    def save(self, path=None, output_dir=DEFAULT_OUTPUT_DIR):
        """Write the calibration SUMMARY (config, beta, FWHM(lambda) fit, slice
        table) as JSON. This is a record of the result, not a cache -- it
        cannot rebuild the calibration. The slice cache is the separate
        psf_slices_<key>.npz that calibrate() writes.

        path : None -> output_dir/psf_calibration.json; a bare filename goes
            into output_dir; a path with a directory is used as given."""
        _migrate_legacy_json(output_dir)
        path = _out_path(path, CALIBRATION_JSON, output_dir)
        d = {"cfg": self.cfg.to_dict(), "beta": self.beta,
             "beta_chi2_dof": self.beta_chi2_dof, "dar": self.dar.tolist(),
             "dar_err": self.dar_err.tolist(), "dar_chi2_dof": self.dar_chi2_dof,
             "powerlaw": self.powerlaw.tolist(), "pl_chi2_dof": self.pl_chi2_dof,
             "lam_range": list(self.lam_range),
             "rows": [{k: (list(d_[k]) if k == "band" else d_[k])
                       for k in ("band", "lam", "fwhm", "fwhm_err", "beta",
                                 "beta_err", "chi2_dof", "stitch", "n_bins")
                       if k in d_}
                      for d_ in self.rows]}
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)
        print(f"wrote {path}  (summary only; the slice cache is the .npz beside it)")
        return path


def calibrate(star_product, cfg=None, *, gmag=None, radial_profile=None,
              cache=True, refresh=False, output_dir=DEFAULT_OUTPUT_DIR,
              verbose=True):
    """
    Stars -> FWHM(lambda) and beta. The slow step; everything else runs off it.

    gmag defaults to star_product.catalog['G'].

    When cfg.beta == "measure" this is a TWO-PASS estimator:
      Pass A  fit every slice with beta free -> inverse-variance mean -> beta*
      Pass B  refit every slice at beta*      -> FWHM(lambda) -> the DAR fit
    Pass B is necessary, not redundant: a beta-free fit returns the FWHM that
    goes with THAT slice's beta, and since FWHM and beta trade off, those values
    are not on a common footing and cannot be compared across slices.

    CACHE. The slow part -- the per-slice star profiles and the frozen inner
    bootstrap errors -- is saved to output_dir/psf_slices_<key>.npz and reused
    on the next call. <key> hashes the config fields that change those arrays
    (bands, clip, norm_annulus, norm_weight, mag_split, faint_max, bright_min,
    n_boot_inner, seed) plus a fingerprint of the star sample, so a changed
    config or a re-extracted star file gets its own file instead of a stale
    hit. Fit-level knobs (beta, fit_r_max, kolm, ...) do not change the key;
    the fits always re-run, and they are fast.

    cache      : False -> neither read nor write the cache.
    refresh    : True  -> rebuild and overwrite even if a cache file exists.
    output_dir : default "outputs_starpsf", relative to the working directory
                 (the notebook's folder in Jupyter). A stray
                 ./psf_calibration.json from older runs is moved in here.
    """
    cfg = cfg or PSFConfig()
    if radial_profile is None:
        from utils_lya_halo.starpsf import radial_continuum_profile as radial_profile
    if gmag is None:
        cat = star_product.catalog
        col = next((c for c in ("G", "g", "GMAG", "gmag", "phot_g_mean_mag")
                    if c in cat.colnames), None)
        if col is None:
            raise KeyError("no Gaia G column in star_product.catalog -- pass gmag=.")
        gmag = np.asarray(cat[col], float)

    _migrate_legacy_json(output_dir, verbose=verbose)
    key, knobs = _cache_key(star_product, gmag, cfg, radial_profile)
    cache_path = os.path.join(output_dir, f"psf_slices_{key}.npz")

    loaded = False
    if cache and not refresh and os.path.isfile(cache_path):
        try:
            caches, errs, faint, bright = _load_slice_cache(cache_path, key)
            loaded = True
            if verbose:
                print(f"loaded slice cache {cache_path}  "
                      f"({len(caches)} bands; pass refresh=True to rebuild)")
        except Exception as exc:              # corrupt/old file -> rebuild
            if verbose:
                print(f"could not read {cache_path} ({exc}) -- rebuilding")

    if not loaded:
        rng = np.random.default_rng(cfg.seed)
        if verbose:
            print("caching slices (the slow step -- once per band):")
        caches, faint, bright = _build_cache(star_product, gmag, cfg, radial_profile, verbose)
        if verbose:
            print("inner bootstrap -> per-bin errors (frozen as fit weights):")
        errs = _inner_errors(caches, cfg, rng, verbose)
        if cache:
            os.makedirs(output_dir, exist_ok=True)
            _save_slice_cache(cache_path, key, knobs, caches, errs, faint, bright)
            if verbose:
                print(f"saved slice cache {cache_path}")

    if cfg.beta_is_measured:
        beta_rows = _one_pass(caches, errs, cfg, beta_fixed=None)
        beta, beta_chi2 = _beta_star(beta_rows)
    else:
        beta_rows, beta, beta_chi2 = None, float(cfg.beta), np.nan

    rows = _one_pass(caches, errs, cfg, beta_fixed=beta)
    lam = np.array([d["lam"] for d in rows])
    fw = np.array([d["fwhm"] for d in rows])
    fe = np.array([d["fwhm_err"]*np.sqrt(d["chi2_dof"]) for d in rows])
    dar, dar_e, dar_c = _dar_fit(lam, fw, fe, cfg)
    pl, pc = curve_fit(lambda l, a, q: _powerlaw(l, a, q, cfg.lam_norm), lam, fw,
                       p0=[float(fw.mean()), -0.3], sigma=fe, absolute_sigma=True)
    rpl = (fw - _powerlaw(lam, *pl, lam_norm=cfg.lam_norm))/fe

    cal = PSFCalibration(cfg=cfg, caches=caches, errs=errs, rows=rows, beta=beta,
                         beta_chi2_dof=beta_chi2, beta_rows=beta_rows, dar=dar,
                         dar_err=dar_e, dar_chi2_dof=dar_c, powerlaw=pl,
                         pl_chi2_dof=float((rpl**2).sum()/max(lam.size-2, 1)),
                         faint_idx=faint, bright_idx=bright)
    if verbose:
        print(); cal.describe(); print(); cal.table()
    return cal


# =====================================================================
# 4. Apply -- kernels for a redshift distribution
# =====================================================================
def _kernel(fwhm_each, r, beta, label):
    """Mean of the per-galaxy Moffats, then one Moffat refitted to it for
    reporting. Unweighted: stack.coadd_galaxies' biweight ignores weights."""
    K = np.array([analytic_moffat(r, f, beta) for f in fwhm_each])
    rr = np.concatenate([[0.0], r])
    for i, f in enumerate(fwhm_each):                 # the Jacobian assertion
        tot = _trapz(analytic_moffat(rr, f, beta)*2*np.pi*rr, rr)
        assert abs(tot - 1) < 0.02, (
            f"kernel {i} integrates to {tot:.4f}, not 1 (fwhm={f:.3f}) -- "
            "normalization bug, or r does not span the profile")
    K_eff = K.mean(axis=0)
    p, _ = curve_fit(lambda x, la, fw, be: la + np.log(analytic_moffat(x, fw, be)),
                     r, np.log(K_eff), p0=[0.0, float(np.median(fwhm_each)), beta])
    dev = 100*np.max(np.abs(np.exp(p[0])*analytic_moffat(r, p[1], p[2])/K_eff - 1))
    return {"r": r, "K_eff": K_eff, "fwhm": float(p[1]), "beta": float(p[2]),
            "dev_pct": float(dev), "unit": label, "n": len(fwhm_each),
            "fwhm_min": float(np.min(fwhm_each)), "fwhm_max": float(np.max(fwhm_each))}


def _kernels_for(cal, z, dar=None, beta=None, unit="kpc"):
    beta = cal.beta if beta is None else float(beta)
    r = cal.cfg.grid(unit)
    fw_as = cal.fwhm_at_z(z, dar=dar)
    each = fw_as*kpc_per_arcsec(z) if unit == "kpc" else fw_as
    return _kernel(each, r, beta, unit)


def apply(cal, z_gal, *, split_at="median", beta=None, verbose=True):
    """
    A redshift distribution -> the kernels. Instant; re-run it for any
    subsample. `beta=` overrides the calibration's beta without recalibrating
    (see `beta_sensitivity`).

    split_at : 'median', a float, or None to skip the split.
    """
    z = np.asarray(z_gal, float); z = z[np.isfinite(z)]
    out = {"z": z, "beta": cal.beta if beta is None else float(beta),
           "n_extrapolated": cal.n_extrapolated(z)}
    out["arcsec"] = _kernels_for(cal, z, beta=beta, unit="arcsec")
    out["kpc"] = _kernels_for(cal, z, beta=beta, unit="kpc")

    if split_at is not None:
        zc = float(np.median(z)) if split_at == "median" else float(split_at)
        lo_z, hi_z = z[z <= zc], z[z > zc]
        out["z_cut"] = zc
        out["low"] = _kernels_for(cal, lo_z, beta=beta, unit="kpc")
        out["high"] = _kernels_for(cal, hi_z, beta=beta, unit="kpc")
        out["split_pct"] = 100*(out["low"]["fwhm"]/out["high"]["fwhm"] - 1)

    if verbose:
        n = out["n_extrapolated"]
        print(f"N = {z.size} galaxies, z {z.min():.2f}-{z.max():.2f}, beta = {out['beta']:.3f}")
        if n:
            print(f"  WARNING: {n} galaxies blueward/redward of the calibrated "
                  f"{cal.lam_range[0]:.0f}-{cal.lam_range[1]:.0f} A -- extrapolating.")
        for k in ("arcsec", "kpc"):
            d = out[k]
            print(f"  {k:6s}: per-galaxy {d['fwhm_min']:.2f}-{d['fwhm_max']:.2f} "
                  f"({100*(d['fwhm_max']/d['fwhm_min']-1):.0f}% spread)  ->  "
                  f"effective FWHM {d['fwhm']:.3f}, beta {d['beta']:.3f} "
                  f"(single Moffat good to {d['dev_pct']:.1f}%)")
        if split_at is not None:
            print(f"  split at z = {out['z_cut']:.2f}: {out['low']['fwhm']:.2f} vs "
                  f"{out['high']['fwhm']:.2f} kpc ({out['split_pct']:+.0f}%)")
            print(f"    -> any h1 difference between the halves smaller than this "
                  f"is PSF, not evolution.")
    return out


def _curve(k, weighted):
    """Peak-normalized kernel, or r^2*K -- flux per unit log radius, which peaks
    at the radius contributing most of the light and spans ~1 decade instead of 8."""
    if weighted:
        w = k["r"]**2 * k["K_eff"]
        return w/np.nanmax(w)
    return k["K_eff"]/k["K_eff"][0]


def plot_kernels(res, figsize=(7.2, 4.8), xlim=None, ylim=None, weighted=False,
                 save_fig=False):
    fig, ax = plt.subplots(figsize=figsize)
    for key, col, lab in (("low", "tab:blue", "low-z"), ("high", "crimson", "high-z")):
        if key not in res:
            continue
        k = res[key]
        ax.plot(k["r"], _curve(k, weighted), "-", color=col, lw=1.8,
                label=f"{lab} (N={k['n']}): FWHM = {k['fwhm']:.2f} {k['unit']}")
    k = res["kpc"]
    ax.plot(k["r"], _curve(k, weighted), "k--", lw=1.2,
            label=f"all (N={k['n']}): {k['fwhm']:.2f} kpc")
    ax.set_xscale("log")
    if not weighted:
        ax.set_yscale("log")
    ax.set_xlabel(f"radius [{k['unit']}]")
    ax.set_ylabel(r"$r^2 K$  (flux per log radius)" if weighted
                  else "normalized surface brightness")
    # The kernel GRID must span the whole profile (normalization + the Jacobian
    # assertion need it), so trim the VIEW, never cfg.r_kpc.
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if "z_cut" in res:
        ax.set_title(f"effective PSF, split at z = {res['z_cut']:.2f} "
                     f"({res['split_pct']:+.0f}%)")
    ax.legend(fontsize=9, frameon=False)
    if save_fig:
        fig.savefig(save_fig if isinstance(save_fig, str) else "psf_kernels.png",
                    dpi=150, bbox_inches="tight")
    return fig, ax


# =====================================================================
# 5. Bootstrap -- the citable errors
# =====================================================================
def bootstrap(cal, z_gal, *, n_boot=None, split_at="median", verbose=True):
    """
    Resample the STARS, push each draw through the WHOLE chain, take the spread.

    Stratified within each magnitude class -- draw n_faint from the faint pool
    and n_bright from the bright pool separately. Pooling would let a draw come
    back with a different class balance, which measures a different experiment.

    The SAME star indices are used for every wavelength slice in a draw (it is
    the same stars at every wavelength), which propagates slice-to-slice
    correlation. Both passes of the beta estimator run inside the draw, so the
    beta/FWHM correlation propagates too and beta* is never treated as a known
    constant.

    Every kernel a draw produces -- all, low-z, high-z -- comes from that one
    draw, so the split halves share their PSF error and the error on their
    DIFFERENCE is much smaller than on either alone. That is what makes the
    split's significance testable.

    Galaxy redshifts are held FIXED: they are spec-z, and their sample variance
    is the stack's uncertainty, not the PSF's.
    """
    cfg = cal.cfg
    n_boot = int(cfg.n_boot_outer if n_boot is None else n_boot)
    rng = np.random.default_rng(cfg.seed + 1)
    z = np.asarray(z_gal, float); z = z[np.isfinite(z)]
    zc = (float(np.median(z)) if split_at == "median"
          else (None if split_at is None else float(split_at)))

    keys = ["beta", "F0", "tanz", "lam_ref", "fwhm_arcsec", "fwhm_kpc"]
    if zc is not None:
        keys += ["fwhm_low", "fwhm_high", "split_pct"]
    draws = {k: np.full(n_boot, np.nan) for k in keys}
    draws["fwhm_lam"] = np.full((n_boot, len(cal.rows)), np.nan)
    lam_rows = np.array([d["lam"] for d in cal.rows])

    try:
        from tqdm.auto import tqdm
        it = tqdm(range(n_boot), desc="PSF bootstrap", disable=not verbose)
    except Exception:
        it = range(n_boot)

    nf, nb = cal.faint_idx.size, cal.bright_idx.size
    for b in it:
        sel_f = cal.faint_idx[rng.integers(0, nf, nf)]
        sel_b = cal.bright_idx[rng.integers(0, nb, nb)]
        try:
            if cfg.beta_is_measured:
                ra = _one_pass(cal._caches, cal._errs, cfg, sel_f, sel_b, beta_fixed=None)
                if len(ra) < 2:
                    continue
                beta_b, _ = _beta_star(ra)
            else:
                beta_b = float(cfg.beta)
            rb = _one_pass(cal._caches, cal._errs, cfg, sel_f, sel_b, beta_fixed=beta_b)
            if len(rb) < 4:
                continue
            lam = np.array([d["lam"] for d in rb])
            fw = np.array([d["fwhm"] for d in rb])
            fe = np.array([d["fwhm_err"]*np.sqrt(d["chi2_dof"]) for d in rb])
            dar_b, _, _ = _dar_fit(lam, fw, fe, cfg)

            draws["beta"][b] = beta_b
            draws["F0"][b], draws["tanz"][b], draws["lam_ref"][b] = dar_b
            for i, L in enumerate(lam_rows):
                j = np.flatnonzero(lam == L)
                if j.size:
                    draws["fwhm_lam"][b, i] = fw[j[0]]

            draws["fwhm_arcsec"][b] = _kernels_for(cal, z, dar=dar_b, beta=beta_b,
                                                   unit="arcsec")["fwhm"]
            draws["fwhm_kpc"][b] = _kernels_for(cal, z, dar=dar_b, beta=beta_b,
                                                unit="kpc")["fwhm"]
            if zc is not None:
                lo = _kernels_for(cal, z[z <= zc], dar=dar_b, beta=beta_b, unit="kpc")["fwhm"]
                hi = _kernels_for(cal, z[z > zc], dar=dar_b, beta=beta_b, unit="kpc")["fwhm"]
                draws["fwhm_low"][b], draws["fwhm_high"][b] = lo, hi
                draws["split_pct"][b] = 100*(lo/hi - 1)
        except Exception:
            continue

    out = {"draws": draws, "n_boot": n_boot, "z_cut": zc, "lam": lam_rows,
           "n_ok": int(np.isfinite(draws["F0"]).sum())}
    for k in keys:
        v = draws[k][np.isfinite(draws[k])]
        out[k] = ((float(np.median(v)), float(np.std(v)),
                   float(np.percentile(v, 16)), float(np.percentile(v, 84)))
                  if v.size else (np.nan,)*4)
    if verbose:
        _describe_bootstrap(cal, out)
    return out


def _describe_bootstrap(cal, bs):
    print(f"\nstar bootstrap: {bs['n_ok']}/{bs['n_boot']} draws converged")
    print(f"{'quantity':>14}{'median':>11}{'+/- (std)':>12}{'16-84%':>22}")
    lab = {"beta": "beta", "F0": "F0 [" + Q + "]", "tanz": "tan z",
           "lam_ref": "lam_ref [A]", "fwhm_arcsec": "FWHM eff [" + Q + "]",
           "fwhm_kpc": "FWHM eff [kpc]", "fwhm_low": "  low-z [kpc]",
           "fwhm_high": "  high-z [kpc]", "split_pct": "split [%]"}
    for k, name in lab.items():
        if k not in bs:
            continue
        m, s, lo, hi = bs[k]
        print(f"{name:>14}{m:11.3f}{s:12.3f}   [{lo:8.3f}, {hi:8.3f}]")
    if "split_pct" in bs:
        m, s, _, _ = bs["split_pct"]
        print(f"\n  the split is {abs(m)/s:.1f} sigma from zero. Note this error is "
              f"SMALL by construction:\n  both halves come from the same draw, so "
              f"their shared PSF error cancels in the ratio.")
    print("\n  These are STATISTICAL (star-to-star) only. Anything common to all "
          "stars is\n  invisible here -- background systematics, the Moffat form, "
          "the effective-kernel\n  convention, the DAR quadrature approximation. "
          "Run systematics() and quote both.")


# =====================================================================
# 6. Systematics -- what the bootstrap cannot see
# =====================================================================
def adopt_bootstrap_errors(cal, boot, verbose=True):
    """
    Replace the provisional per-slice FWHM errors with the MEASURED ones from
    the bootstrap, and refit FWHM(lambda) with them.

    Why this is a separate step, not a config flag: the measured errors do not
    exist until the bootstrap has run, and the bootstrap needs a calibration to
    run on. Pretending otherwise would hide a circular dependency.

    Why it is safe to do after the fact: inside each bootstrap draw, the
    per-slice errors act only as the RELATIVE weights of the nine points in the
    DAR fit. Both the provisional and the measured errors are nearly flat across
    wavelength (each spans ~10-15% end to end), so the relative weighting is
    almost identical either way and the fitted F0/tan z/lam_ref per draw barely
    move. What changes is the reported chi2/dof and the formal error bars --
    which is exactly what was wrong before.

    The provisional errors were formal errors inflated by sqrt(chi2/dof) of the
    Moffat profile fit. That was a mistake: a high profile chi2 reflects
    structure WITHIN a slice (correlated radial bins, or an imperfect Moffat
    shape), which is not evidence that the recovered FWHM for that slice is
    uncertain by that factor. Inflating for it made the DAR fit far too loose --
    chi2/dof ~ 0.06, i.e. errors about 4x too large.
    """
    d = np.asarray(boot["draws"]["fwhm_lam"], float)
    lam_b = np.asarray(boot["lam"], float)
    lam_r = np.array([r["lam"] for r in cal.rows], float)
    emp = np.full(lam_r.size, np.nan)
    for i, L in enumerate(lam_r):
        j = np.flatnonzero(np.isclose(lam_b, L))
        if j.size:
            col = d[:, j[0]]
            col = col[np.isfinite(col)]
            if col.size > 2:
                emp[i] = float(np.std(col))
    if not np.all(np.isfinite(emp) & (emp > 0)):
        raise ValueError("bootstrap produced no usable per-slice FWHM scatter -- "
                         "check that boot came from this calibration.")

    old, old_chi2 = cal.fwhm_err.copy(), cal.dar_chi2_dof
    fw = np.array([r["fwhm"] for r in cal.rows], float)
    dar, dar_e, dar_c = _dar_fit(lam_r, fw, emp, cal.cfg)
    cal.fwhm_err = emp
    cal.fwhm_err_source = f"bootstrap ({boot['n_ok']} draws) -- MEASURED"
    cal.dar, cal.dar_err, cal.dar_chi2_dof = dar, dar_e, dar_c
    pl, _ = curve_fit(lambda l, a, q: _powerlaw(l, a, q, cal.cfg.lam_norm),
                      lam_r, fw, p0=[float(fw.mean()), -0.3], sigma=emp,
                      absolute_sigma=True)
    rpl = (fw - _powerlaw(lam_r, *pl, lam_norm=cal.cfg.lam_norm))/emp
    cal.powerlaw = np.asarray(pl, float)
    cal.pl_chi2_dof = float((rpl**2).sum()/max(lam_r.size - 2, 1))

    if verbose:
        print("adopted bootstrap errors on FWHM(lambda)")
        print(f"{'lam':>7}{'FWHM':>9}{'provisional':>13}{'measured':>11}{'ratio':>8}")
        for L, f, o, e in zip(lam_r, fw, old, emp):
            print(f"{L:7.0f}{f:9.3f}{o:13.4f}{e:11.4f}{o/e:8.2f}")
        print(f"\n  DAR chi2/dof: {old_chi2:.2f} -> {cal.dar_chi2_dof:.2f}   "
              f"(power law: {cal.pl_chi2_dof:.2f})")
        print(f"  F0      = {dar[0]:.3f} +/- {dar_e[0]:.3f}")
        print(f"  tan z   = {dar[1]:.3f} +/- {dar_e[1]:.3f}  (diagnostic only)")
        print(f"  lam_ref = {dar[2]:.0f} +/- {dar_e[2]:.0f}")
        if cal.dar_chi2_dof > 3:
            print("  chi2/dof is now HIGH -- the model is missing something real, "
                  "which the\n  over-large errors were hiding. Worth a look at the "
                  "residual panel.")
        elif cal.dar_chi2_dof < 0.3:
            print("  chi2/dof still very low -- the measured errors may themselves "
                  "be over-large,\n  or the model has more freedom than nine points "
                  "can constrain.")
    return cal


def recalibrate(cal, **overrides):
    """
    Re-run the calibration with different CHOICES, reusing the cached slices.

    Only knobs downstream of the cache may be changed here -- beta, fit_r_min,
    fit_r_max, kolm, lam_norm, apply_stitch, min_stars. Anything that changes
    the cache itself (bands, norm_annulus, norm_weight, clip, mag_split,
    faint_max, bright_min) needs a full `calibrate` and is rejected, rather
    than silently returning a calibration whose cache does not match its config.
    """
    cache_knobs = {"bands", "norm_annulus", "norm_weight", "clip", "mag_split",
                   "faint_max", "bright_min"}
    bad = cache_knobs & set(overrides)
    if bad:
        raise ValueError(f"{sorted(bad)} change the cached slices -- call "
                         "calibrate() again with a new PSFConfig instead.")
    cfg = replace(cal.cfg, **overrides)

    if cfg.beta_is_measured:
        beta_rows = _one_pass(cal._caches, cal._errs, cfg, beta_fixed=None)
        beta, beta_chi2 = _beta_star(beta_rows)
    else:
        beta_rows, beta, beta_chi2 = None, float(cfg.beta), np.nan

    rows = _one_pass(cal._caches, cal._errs, cfg, beta_fixed=beta)
    lam = np.array([d["lam"] for d in rows])
    fw = np.array([d["fwhm"] for d in rows])
    fe = np.array([d["fwhm_err"]*np.sqrt(d["chi2_dof"]) for d in rows])
    dar, dar_e, dar_c = _dar_fit(lam, fw, fe, cfg)
    pl, _ = curve_fit(lambda l, a, q: _powerlaw(l, a, q, cfg.lam_norm), lam, fw,
                      p0=[float(fw.mean()), -0.3], sigma=fe, absolute_sigma=True)
    rpl = (fw - _powerlaw(lam, *pl, lam_norm=cfg.lam_norm))/fe
    return PSFCalibration(cfg=cfg, caches=cal._caches, errs=cal._errs, rows=rows,
                          beta=beta, beta_chi2_dof=beta_chi2, beta_rows=beta_rows,
                          dar=dar, dar_err=dar_e, dar_chi2_dof=dar_c, powerlaw=pl,
                          pl_chi2_dof=float((rpl**2).sum()/max(lam.size-2, 1)),
                          faint_idx=cal.faint_idx, bright_idx=cal.bright_idx)


def beta_sensitivity(cal, z_gal, betas=(2.7, 2.8, 2.9, 3.0), split_at="median"):
    """
    How the answer follows if beta is simply DECLARED rather than measured.

    This REFITS FWHM(lambda) at each beta, which is the only version of the
    question that means anything: FWHM and beta trade off in a Moffat fit, so
    declaring a beta moves every FWHM, and the effect only shows up if the
    slices are refitted. (Changing beta at the kernel stage alone moves
    nothing, because the kernel is built from an already-fitted FWHM.)

    If the kpc FWHM barely moves across the row, the insensitivity claim is
    earned and can go in the paper. If it moves, beta must be carried as
    measured -- beta sets the wing slope, and the wings are where h1 lives.
    """
    print("beta sensitivity -- FWHM(lambda) REFIT at each beta")
    print(f"{'beta':>10}{'F0':>9}{'z deg':>8}{'FWHM arcsec':>13}"
          f"{'FWHM kpc':>11}{'split %':>10}")
    rows = []
    for b in (list(betas) + ["measure"]):
        try:
            c2 = recalibrate(cal, beta=b)
            r = apply(c2, z_gal, split_at=split_at, verbose=False)
            tag = f"  <- measured ({c2.beta:.3f})" if b == "measure" else ""
            lab = f"{c2.beta:10.3f}" if b != "measure" else f"{'measure':>10}"
            print(f"{lab}{c2.dar[0]:9.3f}{np.degrees(np.arctan(c2.dar[1])):8.1f}"
                  f"{r['arcsec']['fwhm']:13.3f}{r['kpc']['fwhm']:11.2f}"
                  f"{r.get('split_pct', np.nan):10.1f}{tag}")
            rows.append((b, c2, r))
        except Exception as ex:
            print(f"{str(b):>10}   FAILED -- {ex}")
    v = [r["kpc"]["fwhm"] for _, _, r in rows]
    if v:
        print(f"\n  kpc FWHM spans {min(v):.2f}-{max(v):.2f} kpc "
              f"({100*(max(v)-min(v))/np.median(v):.1f}%) across these beta.")
    return rows


def systematics(star_product, z_gal, base_cfg=None, *, gmag=None,
                radial_profile=None, variants=None, split_at="median", cal=None):
    """
    Re-run the FULL calibration under each variant. This is the honest
    systematic -- these are choices, not noise, so the bootstrap cannot see them.

    Default variants probe the four decisions with real freedom in them:
    the fit radius, the normalization annulus, the magnitude split, and beta.
    """
    base = base_cfg or PSFConfig()
    if variants is None:
        variants = {
            "baseline":          {},
            "r_max = 8":         {"fit_r_max": 8.0},
            "r_max = 20":        {"fit_r_max": 20.0},
            "annulus 1.5-3":     {"norm_annulus": (1.5, 3.0)},
            "annulus 3-6":       {"norm_annulus": (3.0, 6.0)},
            "mag split 16":      {"mag_split": 16.0},
            "mag split 18":      {"mag_split": 18.0},
            "beta = 2.8 fixed":  {"beta": 2.8},
            "beta = 3.0 fixed":  {"beta": 3.0},
        }
    print(f"{'variant':>18}{'beta':>8}{'F0':>8}{'z deg':>8}"
          + ("FWHM " + Q).rjust(9) + "FWHM kpc".rjust(10) + "split %".rjust(9))
    out = {}
    cache_knobs = {"bands", "norm_annulus", "norm_weight", "clip", "mag_split",
                   "faint_max", "bright_min"}
    for name, kw in variants.items():
        try:
            # Reuse the cached slices whenever the variant does not touch them;
            # only the annulus and magnitude-split rows need a full re-extraction.
            if cal is not None and not (cache_knobs & set(kw)):
                c = recalibrate(cal, **kw) if kw else cal
            else:
                c = calibrate(star_product, replace(base, **kw), gmag=gmag,
                              radial_profile=radial_profile, verbose=False)
            res = apply(c, z_gal, split_at=split_at, verbose=False)
            print(f"{name:>18}{c.beta:8.3f}{c.dar[0]:8.3f}"
                  f"{np.degrees(np.arctan(c.dar[1])):8.1f}"
                  f"{res['arcsec']['fwhm']:9.3f}{res['kpc']['fwhm']:10.2f}"
                  f"{res.get('split_pct', np.nan):9.1f}")
            out[name] = (c, res)
        except Exception as ex:
            print(f"{name:>18}   FAILED -- {ex}")
    if "baseline" in out:
        b = out["baseline"][1]["kpc"]["fwhm"]
        vals = [r["kpc"]["fwhm"] for _, r in out.values()]
        print(f"\n  kpc FWHM spans {min(vals):.2f}-{max(vals):.2f} "
              f"({100*(max(vals)-min(vals))/b:.1f}% of baseline).")
        print("  Quote that range as the systematic, alongside the bootstrap's "
              "statistical error.")
    return out


# =====================================================================
# 7. One-call convenience
# =====================================================================
def run_psf(star_product, z_gal, cfg=None, *, gmag=None, radial_profile=None,
            n_boot=None, split_at="median", plots=True, save=None,
            cache=True, refresh=False, output_dir=DEFAULT_OUTPUT_DIR):
    """Calibrate -> apply -> bootstrap, printing at each stage."""
    cfg = cfg or PSFConfig()
    print("=" * 70); print("1/3  CALIBRATE"); print("=" * 70)
    cal = calibrate(star_product, cfg, gmag=gmag, radial_profile=radial_profile,
                    cache=cache, refresh=refresh, output_dir=output_dir)
    print("\n" + "=" * 70); print("2/3  APPLY"); print("=" * 70)
    res = apply(cal, z_gal, split_at=split_at)
    print("\n" + "=" * 70); print("3/3  BOOTSTRAP"); print("=" * 70)
    bs = bootstrap(cal, z_gal, n_boot=n_boot, split_at=split_at)
    if plots:
        cal.plot_lambda(); plot_kernels(res)
    if save:
        cal.save(None if save is True else save, output_dir=output_dir)
    return {"cal": cal, "res": res, "boot": bs}
