"""
extract.py -- Stage 1 (EXTRACT) helpers.   I/O-bound, run once.

The per-galaxy fiber-extraction stage: load a VDFI field, build the per-galaxy
run table and bad-fiber mask, bin fibers into radial annuli, and coadd them into
per-galaxy spectra. pipeline.run_extract() orchestrates these into the galaxy
FITS that Stage 2 consumes.

Main entry points:
    load_vdfi_field / load_field            -- read the field's fiber data
    build_run_table                         -- per-galaxy catalog rows for this field
    build_bad_fiber_mask                    -- dispatch to the masking.py method
    radial_bin_edges                        -- per-galaxy annulus edges (bin_mode units)
    combine_fibers / combine_fibers_chunked -- fiber coadd (memory-bounded variant)
    extract_fibers                          -- the core per-galaxy extraction loop
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import os
import gc
import json
import hashlib
import warnings
import datetime as _dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Moffat2D
from astropy.stats import biweight_location as biweight, mad_std, sigma_clip
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.interpolate import interp1d, LinearNDInterpolator
from tqdm import tqdm

from .virial import virial_to_kpc_bins, cosmo   # single cosmology source

# NOTE: the background smoothers (smooth_interp_then_gauss /
# smooth_spectrum_nan_safe_adaptive) now live in background.py, which imports
# THIS module at its top level. Importing them here at module scope would close
# that loop into a circular import, so extract_fibers imports them lazily inside
# _maybe_smooth instead -- the same deferred-import pattern core.py already uses
# against this file. Do not hoist that import back up here.

if TYPE_CHECKING:
    from .config import PipelineConfig


# ---------------------------------------------------------------------
# 1.0  Load fiber data
# ---------------------------------------------------------------------
def load_vdfi_field(base_path, field="",
                    wave_min=3470, wave_max=5540, nwave=1036,
                    radius=3.75, n_r=101, n_seeing=51,
                    seeing_hdu=5, build_psf=False):
    """
    Load VDFI field data: RA/DEC, spectra, errors, DAR, seeing (+ optional PSF).

    base_path is INJECTED (no field->path hardcoding here; that lives in
    PipelineConfig.resolve_vdfi_base_path).

    The flux/error files are opened memmapped and their HDUList handles are
    STORED in the returned dict ('_flux_hdul' / '_error_hdul'). This is
    deliberate: closing them (e.g. via `with`) would invalidate the memmap that
    Stage 1 reads from lazily over many hours.

    build_psf is False by default: the binned extraction does not use the Moffat
    PSF grid (out of scope this paper). Set True only for the core-extraction path.
    """
    wave = np.linspace(wave_min, wave_max, nwave)

    # ---- INFO (RA/DEC/seeing) ----
    info_path = f'{base_path}/all_info.fits'
    with fits.open(info_path) as info:
        RA = info[1].data * 1.0
        DEC = info[2].data * 1.0
        seeing = info[seeing_hdu].data * 1.0
    RA, DEC = [x.reshape((x.shape[0], x.shape[1] * x.shape[2])) for x in [RA, DEC]]
    nexp = RA.shape[0]

    # ---- FLUX + ERROR (memmap; handles kept OPEN on purpose) ----
    flux_hdul = fits.open(f'{base_path}/all_flux_final.fits', memmap=True)
    error_hdul = fits.open(f'{base_path}/all_error.fits', memmap=True)
    spectra = flux_hdul[0].data.reshape(
        flux_hdul[0].data.shape[0],
        flux_hdul[0].data.shape[1] * flux_hdul[0].data.shape[2],
        flux_hdul[0].data.shape[3],
    )
    error = error_hdul[0].data.reshape(
        error_hdul[0].data.shape[0],
        error_hdul[0].data.shape[1] * error_hdul[0].data.shape[2],
        error_hdul[0].data.shape[3],
    )

    # ---- DAR (used only by the PSF/core path) ----
    with fits.open(f'{base_path}/all_initial_dar.fits') as dar:
        dar_ra = dar[0].data * 1.0
        dar_dec = dar[1].data * 1.0

    # ---- SEEING + PSF (optional) ----
    if build_psf:
        r = np.linspace(0, radius, n_r)
        seeing_grid = np.linspace(np.nanmin(seeing) - 0.05,
                                  np.nanmax(seeing) + 0.05, n_seeing)
        PSF, R, S, V = moffat_psf_integration(r, seeing_grid)
    else:
        r = seeing_grid = PSF = R = S = V = None

    print(f'Loaded {field}: {nexp} exposures, {RA.shape[1]} fibers, {nwave} wavelength pixels')

    return {
        'field': field, 'base_path': base_path, 'wave': wave,
        'RA': RA, 'DEC': DEC, 'nexp': nexp,
        'spectra': spectra, 'error': error,
        'dar_ra': dar_ra, 'dar_dec': dar_dec,
        'seeing': seeing, 'seeing_hdu': seeing_hdu,
        'radius': radius, 'r': r, 'seeing_grid': seeing_grid,
        'PSF': PSF, 'R': R, 'S': S, 'V': V,
        '_flux_hdul': flux_hdul, '_error_hdul': error_hdul,  # keep memmap alive
    }


def load_field(config: "PipelineConfig") -> dict:
    """Resolve the base path from config and load the field's fiber data."""
    return load_vdfi_field(
        base_path=config.resolve_vdfi_base_path(),
        field=config.normalized_field(),
        wave_min=config.wave_min,
        wave_max=config.wave_max,
        nwave=config.nwave,
    )


# ---------------------------------------------------------------------
# 1.0b  Barycentric correction  (opt-in: config.barycentric)
# ---------------------------------------------------------------------
# HET site (McDonald Observatory). Explicit numbers so no site-registry download.
HET_LON_DEG, HET_LAT_DEG, HET_HEIGHT_M = -104.014722, 30.681444, 2026.0
C_KMS = 299792.458


def compute_barycentric_velocities(base_path, RA, DEC, ut_hour=5.0,
                                   verify_order=True):
    """
    Barycentric velocity correction v_bc (km/s) for every exposure of a field.

    Where the inputs come from (all_info.fits carries NO timestamps):
      * The field directory holds one `YYYYMMDD_NNNNNNN.h5` per observation
        (Remedy quick_reduction output), 3 dithered exposures each. Sorted
        filenames are chronological, and exposure-axis row i of all_info /
        all_flux_final is (file i // 3, dither i % 3).
      * verify_order=True PROVES that mapping before using it: all_info HDU 3
        is Remedy's per-exposure seeing, identical to the h5 Survey['fwhm']
        column, so the two must agree row by row (exactly 0.0 for COSMOS).
      * Time = night date (from the filename) + 1 day + `ut_hour` UT. HET
        observes at fixed altitude, so the true time is within ~1.5 h of
        transit; any time within the night moves v_bc by < 0.5 km/s, and the
        3-dither set spans ~20 min (< 0.1 km/s).
      * Pointing = mean fiber RA/DEC of the exposure (v_bc varies < 0.1 km/s
        across the field).

    astropy's SkyCoord.radial_velocity_correction does the physics: Earth's
    orbital velocity about the barycenter (~30 km/s) + HET's rotational
    velocity (~0.4 km/s), projected onto the line of sight. Sign: v_bc > 0
    when HET moves TOWARD the target, and lambda_bary = lambda_obs*(1+v_bc/c).

    Returns {'v_bc': (nexp,) km/s, 'obs_ids': (nexp,) str, 't_obs': Time}.
    """
    import glob
    import h5py
    from astropy.time import Time
    from astropy.coordinates import EarthLocation

    nexp = RA.shape[0]
    h5s = sorted(glob.glob(os.path.join(str(base_path), "2*_*.h5")))
    if len(h5s) * 3 != nexp:
        raise ValueError(
            f"barycentric: found {len(h5s)} observation h5 files in {base_path} "
            f"but the field has {nexp} exposures (expected 3 per observation). "
            "Cannot map exposures to observing nights.")

    if verify_order:
        hdu3 = np.asarray(fits.getdata(os.path.join(str(base_path), "all_info.fits"), 3),
                          dtype=float)
        fwhm = []
        for fn in h5s:
            with h5py.File(fn, "r") as f:
                s = np.asarray(f["Survey"]["fwhm"][:], dtype=float)
            if s.size != 3:
                raise ValueError(f"barycentric: {os.path.basename(fn)} has {s.size} "
                                 "exposures in Survey, expected 3.")
            fwhm.extend(s)
        dev = np.nanmax(np.abs(hdu3 - np.asarray(fwhm)))
        if not np.isfinite(dev) or dev > 1e-4:
            raise ValueError(
                f"barycentric: exposure order check FAILED (max |all_info HDU3 - "
                f"h5 Survey fwhm| = {dev:.3g}). Sorted h5 files do not match the "
                "exposure axis; refusing to assign nights.")

    stems = [os.path.basename(fn)[:16] for fn in h5s]          # YYYYMMDD_NNNNNNN
    nights = [s[:8] for s in stems]
    t_night = Time([f"{d[:4]}-{d[4:6]}-{d[6:]}T00:00:00" for d in nights], scale="utc")
    t_night = t_night + (1.0 + float(ut_hour) / 24.0) * u.day
    t_obs = Time(np.repeat(t_night.jd, 3), format="jd", scale="utc")
    obs_ids = np.repeat(np.array(stems), 3)

    het = EarthLocation.from_geodetic(lon=HET_LON_DEG * u.deg, lat=HET_LAT_DEG * u.deg,
                                      height=HET_HEIGHT_M * u.m)
    coords = SkyCoord(np.nanmean(RA, axis=1) * u.deg, np.nanmean(DEC, axis=1) * u.deg)
    v_bc = coords.radial_velocity_correction(
        "barycentric", obstime=t_obs, location=het).to_value(u.km / u.s)
    return {"v_bc": np.asarray(v_bc, dtype=float), "obs_ids": obs_ids, "t_obs": t_obs}


def barycentric_resample_weights(wave, v_bc):
    """
    Per-exposure linear-interpolation weights that move a spectrum from the
    observed frame onto the barycentric frame, ON THE SAME wavelength grid.

    The barycentric spectrum at grid point wave[j] is the observed spectrum at
    wave[j] / (1 + v/c). That position falls between pixels j0 and j0+1 at
    fraction alpha:  f_bary[j] = (1-alpha) f[j0] + alpha f[j0+1].
    Every fiber of one exposure shares (j0, alpha), so this is computed nexp
    times, not once per fiber. |v| <= 30 km/s is a shift of <= ~0.27 pixel on
    the 2 A VIRUS grid (alpha near 0 for v < 0, near 1 for v > 0). Positions
    that fall off the grid end are flagged invalid.

    Returns j0 (nexp, nwave) int32, alpha (nexp, nwave) float32,
            valid (nexp, nwave) bool.
    """
    wave = np.asarray(wave, dtype=float)
    v_bc = np.atleast_1d(np.asarray(v_bc, dtype=float))
    nexp, nwave = v_bc.size, wave.size
    pix = np.arange(nwave, dtype=float)
    j0 = np.zeros((nexp, nwave), dtype=np.int32)
    alpha = np.zeros((nexp, nwave), dtype=np.float32)
    valid = np.zeros((nexp, nwave), dtype=bool)
    for e in range(nexp):
        x = np.interp(wave / (1.0 + v_bc[e] / C_KMS), wave, pix,
                      left=np.nan, right=np.nan)
        ok = np.isfinite(x)
        jj = np.zeros(nwave, dtype=np.int64)
        jj[ok] = np.minimum(np.floor(x[ok]).astype(np.int64), nwave - 2)
        j0[e] = jj
        alpha[e, ok] = (x[ok] - jj[ok]).astype(np.float32)
        valid[e] = ok
    return j0, alpha, valid


def apply_barycentric_shift(block, j0, alpha, valid, is_error=False):
    """
    Apply one exposure's (j0, alpha, valid) to a (nfib, nwave) block.
    Flux:  (1-a) f[j0] + a f[j0+1].
    Error: sqrt(((1-a) e[j0])^2 + (a e[j0+1])^2)  (independent pixels).
    alpha == 0 returns the j0 pixel exactly (no NaN leakage from j0+1).
    Off-grid pixels -> NaN. Keeps float32 in, float32 out.
    """
    lo = block[:, j0]
    hi = block[:, j0 + 1]
    a = alpha[np.newaxis, :]
    if is_error:
        mix = np.sqrt(((1.0 - a) * lo) ** 2 + (a * hi) ** 2)
    else:
        mix = (1.0 - a) * lo + a * hi
    out = np.where(a > 0, mix, lo)
    out[:, ~valid] = np.nan
    return out.astype(block.dtype, copy=False)


def _detect_catalog_format(path) -> str:
    """'fits' for a .fits/.fit/.fz extension (case-insensitive), else 'ascii'."""
    suffix = os.path.splitext(str(path))[1].lower()
    return "fits" if suffix in (".fits", ".fit", ".fz") else "ascii"


def read_catalog_table(path, fmt: str = "auto") -> "Table":
    """
    Read a catalog file as an astropy Table, accepting either FITS or ASCII
    (anything astropy's ascii reader handles -- .txt, .dat, .csv, etc).

    fmt='auto' (default) picks the reader from `path`'s extension
    (.fits/.fit/.fz -> 'fits', anything else -> 'ascii'); pass 'fits' or
    'ascii' explicitly to override when the extension is misleading (or set
    config.catalog_format instead of calling this directly).
    """
    resolved = _detect_catalog_format(path) if fmt == "auto" else fmt
    return Table.read(path, format=resolved)


def build_run_table(config: "PipelineConfig", table: "Table | None" = None) -> "Table":
    """
    Per-galaxy run table: select FIELD == normalized_field and
    CATALOG == config.catalog.

    `table` lets you pass an already-cut catalog (e.g. your z < 2.75 selection)
    during manual/stage-by-stage runs. If None, the catalog is read from
    config.catalog_path via read_catalog_table -- FITS or ASCII, picked
    automatically from the extension (override with config.catalog_format) --
    in which case any sample cuts must already live in that file (the plan
    assumes cuts are pre-applied).
    """
    t = table if table is not None else read_catalog_table(
        config.catalog_path, fmt=getattr(config, "catalog_format", "auto")
    )
    field = config.normalized_field()
    sel_field = np.array(t["FIELD"]).astype(str) == field
    sel_catalog = np.array(t["CATALOG"]).astype(str) == config.catalog
    return t[sel_field & sel_catalog]


# ---------------------------------------------------------------------
# 1.1  Fiber masking
# ---------------------------------------------------------------------
def _dispatch_single_mask_method(method: str, config: "PipelineConfig",
                                 vdfi_data: dict) -> np.ndarray:
    """
    Single-component dispatch body, factored out of build_bad_fiber_mask so a
    composite mask_method ('segmap+spec_local') can call it once per '+'-joined
    piece and OR the results together. Unchanged logic from the pre-composite
    single-method dispatch; no method consults the catalog here (protection
    against self-masking is layered on afterward, in build_bad_fiber_mask, for
    spec_* components only).
    """
    field = config.normalized_field()
    RA, DEC = vdfi_data["RA"], vdfi_data["DEC"]

    if method == "none":
        # No fiber masking at all (Niemeyer-2022-style comparison runs). Every
        # fiber is kept; continuum is removed downstream by the stack-level
        # continuum subtraction in measure.integrated_line_flux.
        return np.zeros(np.shape(RA), dtype=bool)

    if method == "segmap":
        seg_path = config.resolve_seg_mask_path()
        if seg_path is None:
            raise ValueError("seg_mask_path is required when mask_method includes "
                             "'segmap' (set an explicit path or 'auto')")
        return segmentation_mask(seg_path, RA, DEC)

    if method == "image":
        cont_path = config.resolve_cont_image_path()
        if cont_path is None:
            raise ValueError("cont_image_path is required when mask_method includes "
                             "'image' (set an explicit path or 'auto')")
        return continuum_image_mask(
            cont_path, field, config.mask_percentile_image, RA, DEC,
        )

    if method in ("spec_global", "spec_local"):
        mode = "global" if method == "spec_global" else "local"
        return make_spectral_continuum_mask(
            vdfi_data["spectra"],
            mask_percentile_high=config.mask_percentile_spec_high,
            mask_percentile_low=config.mask_percentile_spec_low,
            step=config.spec_mask_step,
            mode=mode,
        )

    raise ValueError(f"unknown mask_method component {method!r}; must be one of "
                      "'none', 'segmap', 'image', 'spec_global', 'spec_local'")


def build_bad_fiber_mask(config: "PipelineConfig", vdfi_data: dict,
                         run_table: Optional["Table"] = None,
                         ra_col: str = "RA", dec_col: str = "DEC") -> np.ndarray:
    """
    Dispatch on config.mask_method, which may be a single method ('segmap') or
    a '+'-joined composite ('segmap+spec_local'); components are OR-combined
    (bad in ANY component -> bad in the result):
      'segmap'       -> segmentation_mask (needs config.seg_mask_path)
      'image'        -> continuum_image_mask (needs config.cont_image_path)  [legacy]
      'spec_global'  -> make_spectral_continuum_mask (mode='global')
      'spec_local'   -> make_spectral_continuum_mask (mode='local')
    Returns a (nexp, nfib) boolean bad-fiber mask.

    run_table : the EXACT catalog this call's run is using -- whatever cuts or
        num_gal slicing the caller has already applied. Not re-derived
        internally. Required ONLY if config.mask_protect_radius_arcsec is not
        None AND mask_method includes a spec_* component; in that case,
        fibers within mask_protect_radius_arcsec of ANY galaxy in run_table
        are force-unmasked in the spec_* component mask(s) (via
        masking.apply_protect_radius) before OR-combining with the other
        components. If mask_protect_radius_arcsec is set but run_table is
        None, this raises -- protection needs galaxy positions, so a silent
        no-protect fallback would be a silent correctness change. If
        mask_protect_radius_arcsec is set but mask_method has no spec_*
        component, it is silently unused (segmap/image already avoid
        self-masking their own way).
    ra_col, dec_col : run_table coordinate columns (default 'RA'/'DEC', as
        elsewhere in the pipeline, e.g. core.extract_core_spectra).
    """
    components = config.mask_components()
    if not components:
        raise ValueError("mask_method must not be empty")

    protect_radius = config.mask_protect_radius_arcsec
    has_spec = config.mask_has_spec()
    if protect_radius is not None and has_spec and run_table is None:
        raise ValueError(
            "config.mask_protect_radius_arcsec is set but no run_table was "
            "passed to build_bad_fiber_mask -- the protect radius needs galaxy "
            "positions to know what to protect. Pass the run_table this call "
            "is using (or set mask_protect_radius_arcsec=None to skip protection)."
        )

    RA, DEC = vdfi_data["RA"], vdfi_data["DEC"]
    masks = []
    for m in components:
        component_mask = _dispatch_single_mask_method(m, config, vdfi_data)
        if m in ("spec_global", "spec_local") and protect_radius is not None \
                and run_table is not None:
            gal_RA = np.asarray(run_table[ra_col], dtype=float)
            gal_DEC = np.asarray(run_table[dec_col], dtype=float)
            component_mask = apply_protect_radius(
                component_mask, RA, DEC, gal_RA, gal_DEC, protect_radius,
            )
        masks.append(component_mask)

    combined = masks[0]
    for m in masks[1:]:
        if m.shape != combined.shape:
            raise ValueError(
                f"mask component shape mismatch: {components[0]!r} gave "
                f"{combined.shape}, a later component gave {m.shape}"
            )
        combined = combined | m
    return combined


# ---------------------------------------------------------------------
# 1.2  Geometry + binning  (recomputed on demand; 0.1% of runtime)
# ---------------------------------------------------------------------
def radial_bin_edges(config: "PipelineConfig", mass: float, z: float) -> np.ndarray:
    """
    Per-galaxy radial bin edges, in the unit extract_fibers bins in:
      'virial' -> convert R/Rvir edges to kpc with THIS galaxy's mass & z
      'kpc'    -> config.bins used directly (already kpc)
      'arcsec' -> config.bins used directly (already arcsec)

    No median-galaxy conversion: each galaxy gets its own edges. (Use
    virial.median_virial_kpc_bins in the notebook for a reporting mapping.)
    """
    mode = config.bin_mode.lower()
    bins = np.asarray(config.bins, dtype=float)
    if mode in ("virial", "vr"):
        return virial_to_kpc_bins(bins, mass, z,
                                  model=getattr(config, "smhm_model", "girelli20"))
    if mode in ("kpc", "arcsec"):
        return bins
    raise ValueError("bin_mode must be 'virial', 'kpc', or 'arcsec'")


# ---------------------------------------------------------------------
# 1.3 + 1.4  Background subtraction, binning, coaddition
# ---------------------------------------------------------------------
def combine_fibers(spec: np.ndarray, err: "np.ndarray | None" = None,
                   method: str = "biweight", sigma: float = 3.0, maxiters: int = 5,
                   weights: "np.ndarray | None" = None):
    """
    Combine a 2-D stack of spectra (n, nwave) -> (nwave,).

    method in {'biweight', 'median', 'sigma_clip', 'mean', 'inv_var',
               'weighted_median'}.

    Error/weight usage splits the methods into three groups:
      * 'biweight', 'median', 'sigma_clip' -- robust, deviation-based estimators.
        They IGNORE `err` and `weights` (location/scale come from the data), so
        both may be None. The returned error is the sample scatter / sqrt(N).
      * 'mean', 'inv_var' -- use the reported per-row errors. They REQUIRE
        `err` and raise if it is None.
      * 'weighted_median' -- robust like 'median', but each ROW carries a weight
        in `weights` (a "pseudo-width"): rows are laid along the value axis, each
        occupying width w_i, and the value at half the TOTAL width is returned.
        REQUIRES `weights` (shape (n,)); `err` is ignored. Intended for the
        GALAXY-axis coadd with w_i = fibers-per-bin, so a fully-covered galaxy
        outweighs a half-covered one WITHOUT the all-or-nothing equality of a
        plain median. Per wavelength column a row that is NaN there contributes
        zero width, so coverage gaps drop out cleanly.

    `sigma`/`maxiters` apply to 'sigma_clip' only (astropy sigma_clip, with a
    median centre and a mad_std scale; clipped, unweighted mean for the location).

    Returns (out_spec, out_err), both (nwave,), NaN where no rows contribute.
    """
    method = method.lower()
    n_eff = np.sum(np.isfinite(spec), axis=0).astype(float)

    if method in ("mean", "inv_var") and err is None:
        raise ValueError(f"method '{method}' requires per-fiber errors, but err is None. "
                         f"Use 'biweight', 'median', or 'sigma_clip' for an err-free combine.")
    if method == "weighted_median":
        if weights is None:
            raise ValueError("method 'weighted_median' requires per-row weights "
                             "(e.g. fibers-per-bin for the galaxy-axis stack), "
                             "but weights is None.")
        weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if weights.shape[0] != spec.shape[0]:
            raise ValueError(f"weights has length {weights.shape[0]} but spec has "
                             f"{spec.shape[0]} rows")

    # All-NaN wavelength columns (a bin empty in some block, masked regions,
    # spectrum edges) are expected and correctly return NaN -- but numpy's and
    # astropy's nan-reductions emit a RuntimeWarning ("All-NaN slice", "Mean of
    # empty slice", "Degrees of freedom <= 0") for each one. With wavelength
    # chunking these fire per block and flood the log. The np.errstate below
    # silences the floating-point flags; this catch_warnings silences the
    # paired Python warnings. Both are scoped to the combine only, so genuine
    # warnings elsewhere are unaffected.
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "biweight":
            out = biweight(spec, axis=0, ignore_nan=True)
            scatter = mad_std(spec, axis=0, ignore_nan=True)
            out_err = scatter / np.sqrt(np.maximum(n_eff, 1))
        elif method == "median":
            out = np.nanmedian(spec, axis=0)
            scatter = mad_std(spec, axis=0, ignore_nan=True)
            out_err = 1.2533 * scatter / np.sqrt(np.maximum(n_eff, 1))  # vs-mean penalty
        elif method == "sigma_clip":
            # Iterative deviation-based rejection, then an UNWEIGHTED mean of the
            # survivors. Like biweight, it never weights by reported errors, so it
            # cannot reintroduce the inv_var outer-continuum residual. The mask
            # also flags non-finite samples, so NaNs are excluded automatically.
            clipped = sigma_clip(spec, sigma=sigma, maxiters=maxiters, axis=0,
                                 cenfunc="median", stdfunc="mad_std", masked=True)
            keep = ~np.ma.getmaskarray(clipped)
            n_keep = keep.sum(axis=0).astype(float)
            out = np.ma.mean(clipped, axis=0).filled(np.nan)
            scatter = np.ma.std(clipped, axis=0).filled(np.nan)
            out_err = scatter / np.sqrt(np.maximum(n_keep, 1))
            empty = n_keep == 0
            out[empty] = np.nan
            out_err[empty] = np.nan
        elif method == "mean":
            out = np.nanmean(spec, axis=0)
            e2 = err ** 2
            e2[~np.isfinite(e2)] = np.nan          # mask overflow/inf before summing
            out_err = np.sqrt(np.nansum(e2, axis=0)) / np.maximum(n_eff, 1)
        elif method == "inv_var":
            w = 1.0 / err ** 2
            w[~np.isfinite(w)] = 0                  # mask inf/nan weights
            wsum = np.nansum(w, axis=0)
            out = np.nansum(spec * w, axis=0) / wsum
            out_err = 1.0 / np.sqrt(wsum)
            out[wsum == 0] = np.nan
            out_err[wsum == 0] = np.nan
        elif method == "weighted_median":
            # Per column: give each row a width (its weight) where it is finite,
            # zero where NaN, sort by value, and read off the value at half the
            # total width. argsort sends NaNs to the trailing rows; their width is
            # already 0, so they never affect the crossing.
            w_col = np.where(np.isfinite(spec), weights[:, None], 0.0)   # (n, nwave)
            order = np.argsort(spec, axis=0)                            # NaNs -> top
            s_sorted = np.take_along_axis(spec,  order, axis=0)
            w_sorted = np.take_along_axis(w_col, order, axis=0)
            cw   = np.cumsum(w_sorted, axis=0)
            wsum = cw[-1]                                               # (nwave,)
            cdf  = (cw - 0.5 * w_sorted) / wsum                         # Hazen position
            hi   = np.argmax(cdf >= 0.5, axis=0)                        # first past half
            cols = np.arange(spec.shape[1])
            lo   = np.maximum(hi - 1, 0)
            cdf_lo, cdf_hi = cdf[lo, cols], cdf[hi, cols]
            s_lo,   s_hi   = s_sorted[lo, cols], s_sorted[hi, cols]
            denom = cdf_hi - cdf_lo
            t = np.where(denom > 0, (0.5 - cdf_lo) / denom, 0.0)        # interp 0.5
            out = s_lo + t * (s_hi - s_lo)
            # Kish effective N (equal weights -> N, one dominant row -> ~1) sets
            # the sqrt(N); scatter is the unweighted mad_std as in 'median', with
            # the same 1.2533 vs-mean penalty. Analytic proxy only -- the Stage-3
            # galaxy bootstrap is the real error.
            w2 = np.sum(w_sorted ** 2, axis=0)
            n_eff_w = np.where(w2 > 0, wsum ** 2 / w2, 0.0)
            scatter = mad_std(spec, axis=0, ignore_nan=True)
            out_err = 1.2533 * scatter / np.sqrt(np.maximum(n_eff_w, 1.0))
            zero_w = wsum <= 0
            out[zero_w] = np.nan
            out_err[zero_w] = np.nan
        else:
            raise ValueError("method must be 'biweight', 'median', 'sigma_clip', "
                             "'mean', 'inv_var', or 'weighted_median'")

    bad = n_eff == 0
    out[bad] = np.nan
    out_err[bad] = np.nan
    return out, out_err


def combine_fibers_chunked(spec, err=None, method="biweight", sigma=3.0,
                           maxiters=5, wave_block=64, weights=None):
    # ... docstring unchanged; weighted_median is per-column like the rest ...
    spec = np.asarray(spec)
    _, nwave = spec.shape
    if wave_block is None or wave_block >= nwave:
        return combine_fibers(spec, err, method=method, sigma=sigma,
                              maxiters=maxiters, weights=weights)

    out = np.full(nwave, np.nan)
    out_err = np.full(nwave, np.nan)
    for j0 in range(0, nwave, wave_block):
        j1 = min(j0 + wave_block, nwave)
        e_slice = err[:, j0:j1] if err is not None else None
        o, oe = combine_fibers(spec[:, j0:j1], e_slice, method=method,
                               sigma=sigma, maxiters=maxiters, weights=weights)
        out[j0:j1] = o
        out_err[j0:j1] = oe
    return out, out_err


# ---------------------------------------------------------------------
# 1.4b  Fast paths for the per-bin combine.  PERFORMANCE ONLY.
#
# combine_fibers / combine_fibers_chunked above remain the REFERENCE
# implementation and are never modified. Everything here is opt-in via
# PipelineConfig and falls back to the reference for anything it does not
# handle. See specs/extraction-speedup.md 1e-ter and 9 for the measurements.
#
# Two independent accelerations, deliberately separable:
#
#   THREADING (combine_fibers_threaded, config.combine_threads)
#       Wavelength blocks are independent -- each reads spec[:, j0:j1] and
#       writes out[j0:j1], with no shared state and no cross-block
#       accumulation -- so evaluating them concurrently is bit-identical BY
#       CONSTRUCTION, not merely to tolerance. numpy releases the GIL inside
#       partition/sort/ufuncs, so the nan-medians really do run in parallel.
#       Measured 4.63x on 8 threads; 12 threads is SLOWER than 8 (memory
#       bandwidth). Works for every method.
#
#   FUSION (biweight_fused, config.combine_fused)  -- biweight only
#       astropy takes FIVE nan-median passes to produce two numbers:
#       biweight_location computes the median, then calls
#       median_absolute_deviation which recomputes that same median and then
#       the MAD; mad_std then repeats both. Fusing computes each once.
#       Fusing itself is exact (see combine_fast.validate with strict=True,
#       which reproduces both outputs bit-for-bit). The default's ~1e-7 shift
#       in Err comes from astropy alone: biweight_location upcasts to float64
#       and mad_std does not, so astropy holds TWO values of the same MAD and
#       one shared value cannot equal both.
# ---------------------------------------------------------------------

# astropy's MAD -> sigma factor. mad_std MULTIPLIES by this literal; dividing
# by 0.6744897501960817 instead rounds differently and breaks bit-identity by
# ~6e-8 in float32. Do not "simplify" this.
_MAD_TO_STD = 1.482602218505602


def biweight_fused(data, axis=0, c=6.0, strict=False):
    """
    (biweight_location, mad_std) from ONE nan-median and ONE nan-MAD.

    The location is bit-identical to
        astropy.stats.biweight_location(data, c=c, axis=axis, ignore_nan=True)
    -- every step mirrors astropy's, including the float64 upcast and the
    "MAD == 0 -> return the median" fallback.

    The scatter matches astropy.stats.mad_std(data, axis=axis, ignore_nan=True)
    to ~1e-7 relative by default; strict=True reproduces it exactly at the cost
    of one extra MAD pass in the input dtype.
    """
    a = np.asanyarray(data)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore",
                                                over="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)

        if strict:                      # mad_std's MAD, in the INPUT dtype
            M32 = np.nanmedian(a, axis=axis, keepdims=True)
            mad32 = np.nanmedian(np.abs(a - M32), axis=axis, keepdims=True)
            scatter = np.squeeze(mad32, axis=axis) * _MAD_TO_STD
            del M32, mad32

        d64 = a.astype(np.float64)      # astropy biweight_location, verbatim
        M = np.nanmedian(d64, axis=axis, keepdims=True)
        d = d64 - M
        mad = np.nanmedian(np.abs(d), axis=axis, keepdims=True)

        if not strict:
            scatter = np.squeeze(mad, axis=axis) * _MAD_TO_STD

        u = d / (c * mad)
        mask = np.abs(u) >= 1
        u = (1 - u ** 2) ** 2
        u[mask] = 0
        M_flat = np.squeeze(M, axis=axis)
        value = M_flat + (np.nansum(d * u, axis=axis) / np.nansum(u, axis=axis))
        loc = np.where(np.squeeze(mad, axis=axis) == 0, M_flat, value)

    return loc, scatter


def combine_fibers_fused(spec, axis=0, strict=False):
    """combine_fibers(spec, None, method='biweight') via the fused estimator."""
    spec = np.asarray(spec)
    n_eff = np.sum(np.isfinite(spec), axis=axis).astype(float)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore",
                                                over="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        out, scatter = biweight_fused(spec, axis=axis, strict=strict)
        out_err = scatter / np.sqrt(np.maximum(n_eff, 1))
    bad = n_eff == 0
    out = np.asarray(out, dtype=float).copy()
    out_err = np.asarray(out_err, dtype=float).copy()
    out[bad] = np.nan
    out_err[bad] = np.nan
    return out, out_err


def _combine_one_block(spec, err, j0, j1, method, sigma, maxiters, weights,
                       fused, fused_strict, fused_transpose):
    """One wavelength block. Shared by the serial and threaded paths so both
    execute identical code -- the only difference is who calls it."""
    if fused and method == "biweight":
        if fused_transpose:
            blk = np.ascontiguousarray(spec[:, j0:j1].T)      # (w, nfib)
            return combine_fibers_fused(blk, axis=1, strict=fused_strict)
        blk = np.ascontiguousarray(spec[:, j0:j1])            # (nfib, w)
        return combine_fibers_fused(blk, axis=0, strict=fused_strict)
    e_slice = err[:, j0:j1] if err is not None else None
    return combine_fibers(spec[:, j0:j1], e_slice, method=method,
                          sigma=sigma, maxiters=maxiters, weights=weights)


_EXECUTOR_CACHE: dict = {}


def get_combine_executor(n_threads):
    """
    Process-wide ThreadPoolExecutor for the per-bin combine, one per thread
    count, created on first use.

    Deliberately a long-lived singleton rather than a per-call `with` block.
    278 galaxies x 12 bins is 3,336 pool creations, and in a notebook a
    per-call pool would also be destroyed and rebuilt on every interrupt. Idle
    threads cost a few MB of stack and no CPU, so caching is the cheaper
    trade. Call shutdown_combine_threads() to release them explicitly.
    """
    if n_threads is None or int(n_threads) <= 1:
        return None
    n = int(n_threads)
    ex = _EXECUTOR_CACHE.get(n)
    if ex is None:
        ex = ThreadPoolExecutor(max_workers=n, thread_name_prefix="vdfi_combine")
        _EXECUTOR_CACHE[n] = ex
    return ex


def shutdown_combine_threads():
    """Release every pooled combine executor. Safe to call at any time; the
    next extract_fibers run simply creates a fresh pool."""
    for ex in _EXECUTOR_CACHE.values():
        ex.shutdown(wait=False)
    _EXECUTOR_CACHE.clear()


def combine_fibers_threaded(spec, err=None, method="biweight", sigma=3.0,
                            maxiters=5, wave_block=16, weights=None,
                            n_threads=8, fused=False, fused_strict=False,
                            fused_transpose=False, min_fibers=50_000,
                            executor=None):
    """
    combine_fibers_chunked with the wavelength blocks evaluated concurrently.

    n_threads       : 8 measured best on the 12-vCPU JupyterHub VM (4.63x);
                      12 is SLOWER than 8. Re-measure on a new machine with
                      combine_fast.benchmark_threads.
    fused           : use biweight_fused instead of astropy (biweight only;
                      silently ignored for other methods).
    min_fibers      : below this pool size, run serially -- dispatching 65
                      thread tasks for a 12-fiber inner bin costs more than it
                      saves. Bins 0-8 are <5 s/galaxy combined.
    executor        : reuse a caller-owned ThreadPoolExecutor instead of
                      creating one per call (extract_fibers does this: 278
                      galaxies x 12 bins would otherwise be 3,336 pools).

    MEMORY: each thread holds its own float64 temporaries for one block,
    ~n_fib * wave_block * 8 B * ~5 arrays. For a 480k-fiber bin at
    wave_block=16 that is ~307 MB/thread, ~2.5 GB across 8 -- on top of the
    1.85 GB pool. Raising wave_block scales this linearly; don't.
    """
    spec = np.asarray(spec)
    n_r, nwave = spec.shape

    serial = (wave_block is None or wave_block >= nwave
              or n_threads is None or n_threads <= 1 or n_r < min_fibers)

    blocks = [(j0, min(j0 + wave_block, nwave))
              for j0 in range(0, nwave, wave_block)] if not serial else None

    if serial:
        if not (fused and method == "biweight"):
            return combine_fibers_chunked(spec, err, method=method, sigma=sigma,
                                          maxiters=maxiters, wave_block=wave_block,
                                          weights=weights)
        if wave_block is None or wave_block >= nwave:
            blocks = [(0, nwave)]
        else:
            blocks = [(j0, min(j0 + wave_block, nwave))
                      for j0 in range(0, nwave, wave_block)]

    out = np.full(nwave, np.nan)
    out_err = np.full(nwave, np.nan)

    def _work(jj):
        j0, j1 = jj
        o, oe = _combine_one_block(spec, err, j0, j1, method, sigma, maxiters,
                                   weights, fused, fused_strict, fused_transpose)
        out[j0:j1] = o          # disjoint slice per block -> no race
        out_err[j0:j1] = oe

    if serial:
        for jj in blocks:
            _work(jj)
    elif executor is not None:
        list(executor.map(_work, blocks))
    else:
        with ThreadPoolExecutor(max_workers=int(n_threads)) as ex:
            list(ex.map(_work, blocks))
    return out, out_err


# ---------------------------------------------------------------------
# Per-galaxy cache (resumability). The signature hash invalidates stale
# caches automatically whenever any extraction parameter changes -- the
# per-galaxy analogue of the FITS SRCFILE staleness check.
# ---------------------------------------------------------------------
def _extract_signature(config: "PipelineConfig") -> str:
    """
    Thin wrapper -- the definition now lives on the config as
    `PipelineConfig.extract_signature`, because the same hash names BOTH the
    cache directory and the galaxy FITS (`descriptive_filename`). Two copies
    would reintroduce exactly the drift the change exists to remove. Kept as a
    module-level name so `combine_fast.py`, the tests and older notebooks that
    import it keep working.
    """
    return config.extract_signature()


def _write_cache_manifest(run_cache_dir: str, config: "PipelineConfig") -> None:
    """
    Drop `_manifest.json` beside a run's cached galaxies, recording the knob
    values behind the directory's 10-hex name. Written once (skipped if present)
    and never read by the pipeline -- it exists purely so a human can answer
    'which run was 4f9e21a0bc?' without re-deriving the hash. Failures are
    swallowed: a missing manifest must never break an extraction.
    """
    path = os.path.join(run_cache_dir, "_manifest.json")
    if os.path.exists(path):
        return
    try:
        payload = {
            "signature": config.extract_signature(),
            "galaxy_fits": config.descriptive_filename(),
            "written": _dt.datetime.now().isoformat(timespec="seconds"),
            # Which arrays a gal_*.npz in this directory carries. Redundant with
            # `knobs` (the background-variant flags are hashed when set, so a
            # variant run already lands in its own directory) but written
            # explicitly because "what is IN this cache" is the question someone
            # resuming a half-finished run actually needs answered, and reading
            # it off a 10-hex hash is not answering it.
            "products": (["spec", "err", "nfib", "bg", "vr"]
                         + (["bg_raw"] if config.store_raw_bg else [])
                         + (["bg_nfib"] if (config.store_raw_bg
                                            or config.store_bg_halves) else [])
                         + (["bg_halves", "bg_half_nfib"]
                            if config.store_bg_halves else [])
                         + (["spec_nobg"] if config.store_spec_nobg else [])
                         + (["vbc_mean"] if getattr(config, "barycentric", False)
                            else [])),
            "bg_variants": [list(p) for p in config.resolved_bg_variants()],
            "knobs": config.extract_signature_dict(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except Exception:
        pass


def _save_cache_atomic(cache_path: str, **arrays) -> None:
    """
    Write a per-galaxy cache entry so it can never be observed half-written:
    save to `<name>.tmp.npz`, then `os.replace` onto the final name, which is
    atomic on POSIX. Matters as soon as more than one process writes into a
    shared `cache/<sig>/` -- a Slurm task killed at walltime mid-`savez` would
    otherwise leave a truncated .npz that the next run finds, treats as a valid
    cache hit, and either crashes on or silently loads short.
    """
    tmp = cache_path + ".tmp.npz"          # savez appends .npz if absent
    np.savez(tmp, **arrays)
    os.replace(tmp, cache_path)


def extract_fibers(
    config: "PipelineConfig",
    vdfi_data: dict,
    run_table: "Table",
    bad_fiber_mask: np.ndarray,
    use_cache: bool = True,
    return_extras: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 1 core loop. For each galaxy:
      1. per-exposure background from the [bg_inner, bg_outer) arcsec annulus
         (needs >= min_bg_fibers, else galaxy-level fallback pooled over exposures);
      2. assign every usable fiber to a radial bin once (small int16 index array);
      3. gather and combine ONE radial bin at a time: each bin's pool is
         preallocated to its exact fiber count, kept in float32, background
         subtracted, combined with config.fiber_combine_method, then freed before
         the next bin. The per-fiber error pool is built only when the estimator
         needs it (mean / inv_var); the robust estimators (biweight / median /
         sigma_clip) skip it. Peak RAM is therefore set by the single largest
         bin, not by all bins at once.

    All parameters come from `config`. Bin edges come from radial_bin_edges
    (per galaxy). kpc/arcsec conversion uses the SAME cosmology as the virial
    edges (imported `cosmo`), so binning is internally consistent.

    Per-galaxy results cache to <cache_dir>/<sig>/gal_<i>.npz; with
    use_cache, cached galaxies are reloaded instead of recomputed.

    Returns (Spec, Err, NFIB, Background, vr_edges_used):
      Spec/Err   : (ngal, nrad, nwave)
      NFIB       : (ngal, nrad)   fibers pooled per bin   <- contract HDU
      Background : (ngal, nwave)  galaxy-level background (diagnostic)
      vr_edges   : (ngal, nrad+1) per-galaxy edges actually used

    With `return_extras=True` a SIXTH element is appended: a dict of the
    background-variant products (2.5), holding only the keys this config asked
    for -- 'bg_raw', 'bg_nfib', 'spec_nobg', 'bg_halves', 'bg_half_nfib'. The
    5-tuple is the default so every existing caller (pipeline.run_extract,
    starpsf.extract_star_profiles) keeps working untouched; the extras are an
    opt-in addition, never a change to the contract.

    BACKGROUND VARIANTS, in one paragraph, because the property everything else
    rests on is easy to lose: variant 0 is the fiducial annulus and the ONLY one
    subtracted from science fibers. Variants 1+ are measured, stored, and never
    used. So `bg_variants` can be lengthened freely and Spec/Err/NFIB/Background
    are byte-identical to a run without it -- there is no version of this flag
    that quietly re-defines a science number.
    """
    wave = vdfi_data["wave"]
    RA = vdfi_data["RA"]
    DEC = vdfi_data["DEC"]
    spectra = vdfi_data["spectra"]
    error = vdfi_data["error"]

    nexp = RA.shape[0]
    nfib_total = RA.shape[1]
    nwave = spectra.shape[-1]

    # Window, not prefix: (0, num_gal) by default, but a Slurm array task gets
    # its own [start, stop). `gal_start` is what makes the cache key GLOBAL --
    # see the cache_path below and config.galaxy_window.
    gal_start, gal_stop = config.galaxy_window(len(run_table))
    ngal = gal_stop - gal_start
    run_table_use = run_table[gal_start:gal_stop]

    fiber_method = config.fiber_combine_method.lower()
    bg_method = config.resolved_bg_combine_method().lower()
    bin_mode = config.bin_mode.lower()
    bins = np.asarray(config.bins, dtype=float)
    nrad = len(bins) - 1

    # Which methods consume per-fiber errors. The robust estimators
    # (biweight / median / sigma_clip) do not, so for those we never build the
    # error pool at all -- that alone halves the per-bin footprint.
    needs_err = fiber_method in ("mean", "inv_var")
    bg_needs_err = bg_method in ("mean", "inv_var")

    # sigma_clip knobs (optional on config; sensible defaults otherwise).
    sc_sigma = float(getattr(config, "sigma_clip_sigma", 3.0))
    sc_maxiters = getattr(config, "sigma_clip_maxiters", 5)

    # Wavelength-block size for the per-bin combine. Caps the robust-estimator
    # temporaries so one oversized outer bin can't spike past the memory cap.
    # Purely a memory knob -- the result is identical for any block size, so it
    # is deliberately NOT part of the cache signature. None -> single-shot.
    wave_block = getattr(config, "combine_wave_block", 64)

    # Combine acceleration (specs/extraction-speedup.md 9). All optional and
    # all defaulting to the current behaviour, so an unchanged config runs the
    # unchanged serial reference path.
    n_threads = int(getattr(config, "combine_threads", 1) or 1)
    min_fibers = int(getattr(config, "combine_min_fibers", 50_000))
    fused = bool(getattr(config, "combine_fused", False))
    fused_strict = bool(getattr(config, "combine_fused_strict", False))
    fused_transpose = bool(getattr(config, "combine_fused_transpose", False))
    if fused and fiber_method != "biweight":
        warnings.warn(
            f"combine_fused=True is implemented for 'biweight' only; "
            f"fiber_combine_method is '{fiber_method}'. Falling back to the "
            f"reference combine (threading, if enabled, still applies).",
            RuntimeWarning, stacklevel=2)
        fused = False
    use_fast = (n_threads > 1) or fused
    _executor = get_combine_executor(n_threads)

    Spec = np.full((ngal, nrad, nwave), np.nan)
    Err = np.full((ngal, nrad, nwave), np.nan)
    NFIB = np.zeros((ngal, nrad), dtype=np.int32)
    Background = np.full((ngal, nwave), np.nan)
    VR = np.full((ngal, nrad + 1), np.nan)

    # ---- background-variant products (specs/systematics-and-extraction.md 2.5) --
    # Variant 0 is the fiducial annulus and the only one ever subtracted; see the
    # docstring. Everything here is off by default and allocates nothing.
    bg_variants = config.resolved_bg_variants()
    nvar = len(bg_variants)
    store_raw_bg = bool(getattr(config, "store_raw_bg", False))
    store_spec_nobg = bool(getattr(config, "store_spec_nobg", False))
    store_bg_halves = bool(getattr(config, "store_bg_halves", False))

    # THE GUARD THAT MAKES SPEC_NOBG MEANINGFUL. Reconstructing an unsubtracted
    # coadd as `Spec + bg` is exact ONLY because the biweight is
    # translation-equivariant AND every exposure shares one background. The
    # second half holds only while the per-exposure branch below never fires,
    # i.e. min_bg_fibers exceeds the total fiber count. If min_bg_fibers is ever
    # lowered, different exposures carry different offsets, those do not commute
    # with a single post-hoc addition, and SPEC_NOBG becomes SILENTLY wrong --
    # wrong in a way no plot would show. Hence a hard failure, not a warning.
    if store_spec_nobg and int(config.min_bg_fibers) <= nfib_total:
        raise ValueError(
            f"store_spec_nobg=True requires min_bg_fibers > nfib_total "
            f"({config.min_bg_fibers} <= {nfib_total}), so that every exposure "
            "shares one pooled background. With per-exposure backgrounds the "
            "offsets differ between exposures and cannot be re-added after the "
            "fiber combine -- SPEC_NOBG would be wrong with no visible symptom."
        )

    BG_RAW = (np.full((ngal, nvar, nwave), np.nan, dtype=np.float32)
              if store_raw_bg else None)
    BG_NFIB = (np.zeros((ngal, nvar), dtype=np.int32)
               if (store_raw_bg or store_bg_halves) else None)
    BG_HALVES = (np.full((ngal, nvar, 2, nwave), np.nan, dtype=np.float32)
                 if store_bg_halves else None)
    BG_HALF_NFIB = (np.zeros((ngal, nvar, 2), dtype=np.int32)
                    if store_bg_halves else None)
    SPEC_NOBG = (np.full((ngal, nrad, nwave), np.nan)
                 if store_spec_nobg else None)

    # ---- barycentric correction (opt-in; see config.barycentric) ----
    # Precompute, once per call, the per-exposure resample weights. Applied in
    # the gather loop below: AFTER the per-exposure background subtraction (sky
    # and instrument residuals live in the telescope frame, as does the annulus
    # background, so they subtract cleanly there) and BEFORE fibers from
    # different nights are pooled. VBC_MEAN[gal, r] records the fiber-weighted
    # mean v_bc of each bin -- the centroid shift the correction should produce,
    # i.e. the prediction an on-vs-off comparison is checked against.
    BARY = bool(getattr(config, "barycentric", False))
    VBC_MEAN = None
    if BARY:
        if store_spec_nobg:
            raise ValueError(
                "barycentric=True is incompatible with store_spec_nobg=True: SPEC "
                "is resampled to the barycentric frame but the re-added background "
                "is in the telescope frame, so SPEC_NOBG = SPEC + bg would no "
                "longer be exact.")
        if getattr(config, "inject", None) is not None:
            raise ValueError(
                "barycentric=True is not supported on injection runs: the injector "
                "writes sources in the telescope frame with no Earth motion, so the "
                "correction would ADD a spurious per-exposure shift to them.")
        v_bc = vdfi_data.get("v_bc")
        if v_bc is None:
            v_bc = compute_barycentric_velocities(
                vdfi_data["base_path"], RA, DEC,
                ut_hour=float(getattr(config, "barycentric_ut_hour", 5.0)))["v_bc"]
        v_bc = np.asarray(v_bc, dtype=float)
        if v_bc.shape != (nexp,):
            raise ValueError(f"v_bc has shape {v_bc.shape}, expected ({nexp},)")
        bary_j0, bary_alpha, bary_valid = barycentric_resample_weights(wave, v_bc)
        VBC_MEAN = np.full((ngal, nrad), np.nan)
        print(f"Barycentric correction ON: v_bc {v_bc.min():.1f} .. {v_bc.max():.1f} "
              f"km/s (mean {v_bc.mean():.1f}) over {nexp} exposures")

    # Which arrays a cache entry for THIS config must contain. Checked on every
    # cache hit: a hit that predates one of these products is recomputed rather
    # than loaded, so turning a product on can never leave a silent NaN row in
    # the stack (see the cache-hit branch below).
    _cache_keys = ["spec", "err", "nfib", "bg", "vr"]
    if store_raw_bg:
        _cache_keys += ["bg_raw"]
    if store_raw_bg or store_bg_halves:
        _cache_keys += ["bg_nfib"]
    if store_bg_halves:
        _cache_keys += ["bg_halves", "bg_half_nfib"]
    if store_spec_nobg:
        _cache_keys += ["spec_nobg"]
    if BARY:
        _cache_keys += ["vbc_mean"]
    _stale_cache_warned = False

    # One directory per signature: cache/<sig>/gal_<i>.npz. The old layout put
    # every galaxy of every config ever run flat into cache/ as
    # gal_<sig>_<i>.npz, which grew until listing it took minutes; scoping a
    # listing to one config also makes retiring a dead config an `rm -rf` of one
    # directory. Indices are zero-padded so `ls` sorts them numerically.
    sig = _extract_signature(config)
    cache_dir = config.resolved_run_cache_dir()
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        _write_cache_manifest(cache_dir, config)

    def _maybe_smooth(spec_1d, z):
        # DEFERRED IMPORT, not a style choice: the smoothers live in
        # background.py, which imports this module at its top level. Importing
        # them at extract.py's module scope would close that into a circular
        # import and break `import utils_lya_halo` outright. By call time
        # background.py is fully loaded, so this resolves fine. (Python caches
        # the module, so the per-galaxy cost here is a dict lookup.)
        from .background import (smooth_interp_then_gauss,
                                 smooth_spectrum_nan_safe_adaptive)
        if not config.smooth_bg:
            return spec_1d
        method = getattr(config, "bg_smooth_method", "interp_gauss").lower()
        if method == "interp_gauss":
            return smooth_interp_then_gauss(
                spec_1d, wave, z, lya_rest=config.LYA_REST,
                sigma_A=config.bg_smooth_sigma_A,
                mask_halfwidth_A=config.line_mask_halfwidth_A,
            )
        sv = config.smoothing_values          # legacy 'adaptive'
        return smooth_spectrum_nan_safe_adaptive(
            spec_1d, wave, sigma_A=sv[0], edge_sigma_A=sv[1], edge_width_A=sv[2]
        )

    for cnt in tqdm(range(ngal), total=ngal):
        ra = run_table_use["RA"][cnt]
        dec = run_table_use["DEC"][cnt]
        z = float(run_table_use[config.z_col][cnt])
        m = float(run_table_use[config.mass_col][cnt])

        # ---- per-galaxy cache check ----
        # Keyed on the GLOBAL run-table row (gal_start + cnt), never the loop
        # counter, so the same galaxy maps to the same file regardless of which
        # array task computed it -- and so array tasks sharing cache/<sig>/
        # cannot overwrite each other.
        cache_path = os.path.join(cache_dir, f"gal_{gal_start + cnt:05d}.npz")
        if use_cache and os.path.exists(cache_path):
            g = np.load(cache_path)
            # A cache entry written before this config's background products
            # existed is INCOMPLETE, not merely old. Loading it would put NaN
            # rows into BG_RAW / SPEC_NOBG for that galaxy and quietly drop it
            # from every downstream stack -- the failure mode 2.5 Change 4 warns
            # about, and the one that surfaces on the first RESUME rather than
            # on the first run. Recompute instead: costs one galaxy, and is
            # strictly safer than trusting a partial entry.
            if all(k in g.files for k in _cache_keys):
                Spec[cnt], Err[cnt], NFIB[cnt] = g["spec"], g["err"], g["nfib"]
                Background[cnt], VR[cnt] = g["bg"], g["vr"]
                if BG_RAW is not None:
                    BG_RAW[cnt] = g["bg_raw"]
                if BG_NFIB is not None:
                    BG_NFIB[cnt] = g["bg_nfib"]
                if BG_HALVES is not None:
                    BG_HALVES[cnt] = g["bg_halves"]
                    BG_HALF_NFIB[cnt] = g["bg_half_nfib"]
                if SPEC_NOBG is not None:
                    SPEC_NOBG[cnt] = g["spec_nobg"]
                if VBC_MEAN is not None:
                    VBC_MEAN[cnt] = g["vbc_mean"]
                continue
            if not _stale_cache_warned:
                missing = [k for k in _cache_keys if k not in g.files]
                warnings.warn(
                    f"cache entries in {cache_dir} predate the background-variant "
                    f"products (missing {missing}); recomputing those galaxies "
                    "instead of loading them short.",
                    RuntimeWarning, stacklevel=2)
                _stale_cache_warned = True

        kpc_per_arcsec = cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265
        radial_bins = radial_bin_edges(config, m, z)
        VR[cnt] = radial_bins

        # ---- precompute distances + good-fiber masks ----
        dist_arcsec_all = np.full((nexp, nfib_total), np.nan)
        good_base_all = np.zeros((nexp, nfib_total), dtype=bool)
        # North/south half-plane flag, built only when the azimuthal split is
        # requested. A FIXED split line (the E-W axis through the galaxy), not a
        # per-galaxy or random one: the two halves must mean the same thing for
        # every galaxy or their difference measures the split, not the sky.
        north_all = (np.zeros((nexp, nfib_total), dtype=bool)
                     if store_bg_halves else None)
        for e in range(nexp):
            dra = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
            ddec = (DEC[e] - dec) * 3600.
            dist_arcsec_all[e] = np.sqrt(dra ** 2 + ddec ** 2)
            good_base_all[e] = (~bad_fiber_mask[e]) & np.isfinite(RA[e]) & np.isfinite(DEC[e])
            if north_all is not None:
                north_all[e] = ddec >= 0.0

        # ---- 1. per-exposure backgrounds + galaxy fallback ----
        # float32 backgrounds: the science subtraction below is float32 - bg, so
        # a float64 bg here would silently upcast every science fiber to float64.
        bg_per_exp = np.full((nexp, nwave), np.nan, dtype=np.float32)
        bg_err_per_exp = np.full((nexp, nwave), np.nan, dtype=np.float32)
        has_own_bg = np.zeros(nexp, dtype=bool)
        all_bg_fibers, all_bg_errs = [], []

        # One pass per annulus variant. VARIANT 0 IS THE FIDUCIAL and the only
        # one that reaches `bg_per_exp` / the science subtraction below; its path
        # through this block is byte-for-byte what it was before variants
        # existed. Variants 1+ skip the per-exposure branch entirely (they are
        # never subtracted, so a per-exposure version of them would mean
        # nothing) and exist only to be stored.
        for v, (bg_lo, bg_hi) in enumerate(bg_variants):
            all_bg_fibers, all_bg_errs, all_bg_north = [], [], []

            for e in range(nexp):
                bg_sel = ((dist_arcsec_all[e] >= bg_lo) &
                          (dist_arcsec_all[e] < bg_hi) &
                          good_base_all[e])
                n_bg = int(np.sum(bg_sel))
                if n_bg == 0:
                    continue
                sp_bg = spectra[e, bg_sel]
                all_bg_fibers.append(sp_bg)
                if bg_needs_err:
                    all_bg_errs.append(error[e, bg_sel])
                if north_all is not None:
                    all_bg_north.append(north_all[e][bg_sel])
                if v == 0 and n_bg >= config.min_bg_fibers:
                    er_bg = error[e, bg_sel] if bg_needs_err else None
                    bg_e, bg_ee = combine_fibers(sp_bg, er_bg, method=bg_method,
                                                 sigma=sc_sigma, maxiters=sc_maxiters)
                    bg_per_exp[e] = _maybe_smooth(bg_e, z)
                    bg_err_per_exp[e] = bg_ee
                    has_own_bg[e] = True

            if all_bg_fibers:
                sp_all = np.concatenate(all_bg_fibers, axis=0)
                er_all = np.concatenate(all_bg_errs, axis=0) if bg_needs_err else None
                bg_raw_v, bg_err_v = combine_fibers(sp_all, er_all, method=bg_method,
                                                    sigma=sc_sigma, maxiters=sc_maxiters)
            else:
                sp_all = er_all = None
                bg_raw_v = np.full(nwave, np.nan)
                bg_err_v = np.full(nwave, np.nan)

            # THE LINE THIS WHOLE RUN EXISTS FOR. `bg_raw_v` is the annulus
            # coadd BEFORE _maybe_smooth, i.e. before the Lya mask, the
            # interpolation across the gap, and the 30 A gaussian. The stored
            # BACKGROUND HDU is all three of those applied (B9), so any Lya
            # feature in the annulus has already been interpolated away in it --
            # which is exactly why 1.1's rest-frame stack test cannot be run on
            # what is currently on disk. This array is that test's input.
            if BG_RAW is not None:
                BG_RAW[cnt, v] = bg_raw_v
            if BG_NFIB is not None:
                BG_NFIB[cnt, v] = int(sum(a.shape[0] for a in all_bg_fibers))

            # Azimuthal halves: same annulus, same estimator, disjoint fibers.
            # Their difference is a direct measurement of how much the
            # background varies ACROSS the annulus (B10/B11) -- something no
            # amount of Stage-2 work on the pooled spectrum can recover.
            if BG_HALVES is not None and sp_all is not None:
                north = np.concatenate(all_bg_north)
                for h, sel_h in enumerate((north, ~north)):
                    n_h = int(np.count_nonzero(sel_h))
                    BG_HALF_NFIB[cnt, v, h] = n_h
                    if n_h == 0:
                        continue
                    er_h = er_all[sel_h] if er_all is not None else None
                    bg_h, _ = combine_fibers(sp_all[sel_h], er_h, method=bg_method,
                                             sigma=sc_sigma, maxiters=sc_maxiters)
                    BG_HALVES[cnt, v, h] = bg_h
                del north

            if v == 0:
                # Fiducial: this is the background the science fibers see.
                bg_gal_err = bg_err_v
                bg_gal = _maybe_smooth(bg_raw_v, z) if all_bg_fibers else bg_raw_v
                Background[cnt] = bg_gal

            del sp_all, er_all, all_bg_fibers, all_bg_errs, all_bg_north

        for e in range(nexp):
            if not has_own_bg[e]:
                bg_per_exp[e] = bg_gal
                bg_err_per_exp[e] = bg_gal_err

        # ---- 2. assign every usable fiber to a radial bin (once) ----
        # bin_idx_all[e, f] = radial-bin index of fiber f in exposure e, or -1
        # for masked / out-of-range fibers. int16 is plenty (nrad is tiny) and
        # this whole array is a few MB; it lets the per-bin gather below avoid
        # re-running digitize, and lets us size each bin's pool exactly.
        bin_idx_all = np.full((nexp, nfib_total), -1, dtype=np.int16)
        for e in range(nexp):
            dist = (dist_arcsec_all[e] if bin_mode == "arcsec"
                    else dist_arcsec_all[e] * kpc_per_arcsec)
            idx = np.digitize(dist, radial_bins) - 1
            idx[~good_base_all[e]] = -1
            idx[(idx < 0) | (idx >= nrad)] = -1
            bin_idx_all[e] = idx

        # fiber count per bin, in one pass over the (in-RAM) index array -- no
        # fiber-data reads, so this is essentially free and gives the exact
        # pool size to preallocate.
        vbc_sum = np.zeros(nrad) if VBC_MEAN is not None else None
        for e in range(nexp):
            v = bin_idx_all[e]
            v = v[v >= 0]
            if v.size:
                counts_e = np.bincount(v, minlength=nrad)
                NFIB[cnt] += counts_e.astype(np.int32)
                if vbc_sum is not None:
                    vbc_sum += counts_e * v_bc[e]
        if vbc_sum is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                VBC_MEAN[cnt] = np.where(NFIB[cnt] > 0, vbc_sum / NFIB[cnt], np.nan)

        # ---- 3. gather + combine ONE BIN AT A TIME ----
        # Each bin's pool is preallocated to its exact size (no list+concatenate
        # doubling), kept in float32, and freed before the next bin -- so peak
        # RAM is set by the single largest bin, not by all bins at once. The
        # error pool is built only when the chosen estimator actually uses it.
        for r in range(nrad):
            n_r = int(NFIB[cnt, r])
            if n_r == 0:
                continue
            pool_spec = np.empty((n_r, nwave), dtype=np.float32)
            pool_err = np.empty((n_r, nwave), dtype=np.float32) if needs_err else None

            off = 0
            for e in range(nexp):
                sel = bin_idx_all[e] == r
                k = int(np.count_nonzero(sel))
                if k == 0:
                    continue
                # float32 - float32 stays float32 (bg_per_exp is float32)
                d = spectra[e, sel] - bg_per_exp[e][np.newaxis, :]
                if BARY:
                    # background removed in the telescope frame; now move this
                    # exposure's fibers to the barycentric frame before pooling
                    d = apply_barycentric_shift(d, bary_j0[e], bary_alpha[e],
                                                bary_valid[e])
                pool_spec[off:off + k] = d
                if needs_err:
                    de = np.sqrt(
                        error[e, sel].astype(np.float32) ** 2
                        + bg_err_per_exp[e][np.newaxis, :] ** 2
                    )
                    if BARY:
                        de = apply_barycentric_shift(de, bary_j0[e], bary_alpha[e],
                                                     bary_valid[e], is_error=True)
                    pool_err[off:off + k] = de
                off += k

            if use_fast:
                Spec[cnt, r], Err[cnt, r] = combine_fibers_threaded(
                    pool_spec, pool_err, method=fiber_method,
                    sigma=sc_sigma, maxiters=sc_maxiters, wave_block=wave_block,
                    n_threads=n_threads, fused=fused, fused_strict=fused_strict,
                    fused_transpose=fused_transpose, min_fibers=min_fibers,
                    executor=_executor,
                )
            else:
                Spec[cnt, r], Err[cnt, r] = combine_fibers_chunked(
                    pool_spec, pool_err, method=fiber_method,
                    sigma=sc_sigma, maxiters=sc_maxiters, wave_block=wave_block,
                )

            # SPEC_NOBG by addition, NOT by a second unsubtracted combine.
            # The biweight location estimator is translation-equivariant:
            # subtracting a per-wavelength scalar from every fiber shifts the
            # result by exactly that scalar (sigma-clipping likewise works on
            # deviations from the center, so it is unaffected too). Every
            # exposure shares one background -- asserted at entry via
            # min_bg_fibers -- so adding it back after the combine is exact,
            # and costs one array add instead of doubling the run's dominant
            # operation. `bg_per_exp[0]` rather than `bg_gal` deliberately: it
            # is the float32 value actually subtracted, so the round trip is
            # exact rather than exact-to-a-cast.
            if SPEC_NOBG is not None:
                SPEC_NOBG[cnt, r] = Spec[cnt, r] + bg_per_exp[0]

            del pool_spec, pool_err   # release this bin before gathering the next

        del bin_idx_all
        gc.collect()                  # keep peak flat across the long per-galaxy run

        if use_cache:
            extra_arrays = {}
            if BG_RAW is not None:
                extra_arrays["bg_raw"] = BG_RAW[cnt]
            if BG_NFIB is not None:
                extra_arrays["bg_nfib"] = BG_NFIB[cnt]
            if BG_HALVES is not None:
                extra_arrays["bg_halves"] = BG_HALVES[cnt]
                extra_arrays["bg_half_nfib"] = BG_HALF_NFIB[cnt]
            if SPEC_NOBG is not None:
                extra_arrays["spec_nobg"] = SPEC_NOBG[cnt]
            if VBC_MEAN is not None:
                extra_arrays["vbc_mean"] = VBC_MEAN[cnt]
            _save_cache_atomic(cache_path, spec=Spec[cnt], err=Err[cnt],
                               nfib=NFIB[cnt], bg=Background[cnt], vr=VR[cnt],
                               **extra_arrays)

    if not return_extras:
        return Spec, Err, NFIB, Background, VR

    extras = {}
    if BG_RAW is not None:
        extras["bg_raw"] = BG_RAW
    if BG_NFIB is not None:
        extras["bg_nfib"] = BG_NFIB
    if BG_HALVES is not None:
        extras["bg_halves"] = BG_HALVES
        extras["bg_half_nfib"] = BG_HALF_NFIB
    if SPEC_NOBG is not None:
        extras["spec_nobg"] = SPEC_NOBG
    if VBC_MEAN is not None:
        extras["vbc_mean"] = VBC_MEAN
    return Spec, Err, NFIB, Background, VR, extras


# #####################################################################
# #####################################################################
# FIBER MASKING  (merged from masking.py, 2026-08-03)
#
# Turns fiber sky positions into a (nexp, nfib) boolean bad-fiber mask
# (True = bad). build_bad_fiber_mask (above) dispatches on
# config.mask_method, which may be a single method ('segmap') or a
# '+'-joined composite ('segmap+spec_local') that OR-combines each
# component's mask (bad in ANY component -> bad overall). All component
# functions end in the same shared world->pixel lookup where applicable.
#
#   METHOD                        function                      config.mask_method
#   ----------------------------  ----------------------------  ------------------
#   baked segmentation (default)  segmentation_mask             'segmap'
#   per-fiber continuum level     make_spectral_continuum_mask  'spec_global' /
#                                                               'spec_local'
#   continuum image (legacy A/B)  continuum_image_mask          'image'
#
#   shared lookup used by segmap + image:  flag_fibers_on_mask
#   spec_* only, applied post-threshold:   apply_protect_radius
#                                          (config.mask_protect_radius_arcsec)
#
# Default is 'segmap+spec_global': segmap catches faint real sources
# image-plane percentile masks miss; spec_global catches instrumental bad
# fibers no image-plane mask can see (they aren't on real sources) --
# disjoint failure modes, so OR-combining both is the default rather than
# a tuning choice.
#
# The 'image' method and its three private-by-convention helpers
# (make_circular_structure, get_pixscale_arcsec, nan_safe_gaussian_smooth)
# are kept ONLY as the A/B baseline for validating the baked segmap mask;
# nothing in the current pipeline path needs them. Self-mask rescues that
# the old code did at runtime are now a bake-time edit to the segmap FITS.
# See core.compare_mask_spectra and plotting.plot_fiber_mask_appendix for
# the checking/validation tools.
# #####################################################################
# #####################################################################

# =====================================================================
# SHARED LOOKUP  (used by the segmap and image methods)
# =====================================================================
def flag_fibers_on_mask(RA, DEC, wcs_full, mask_cutout, xmin=0, ymin=0):
    """
    Flag fibers (RA/DEC in deg, any matching shape) that land on/near masked
    continuum. Returns a boolean array shaped like RA/DEC; True = bad.
    """
    ra = np.array(RA, dtype=float)
    dec = np.array(DEC, dtype=float)
    original_shape = ra.shape

    ra_flat = ra.ravel()
    dec_flat = dec.ravel()
    good_coord = np.isfinite(ra_flat) & np.isfinite(dec_flat)
    bad_flat = np.zeros_like(ra_flat, dtype=bool)

    coords = SkyCoord(ra_flat[good_coord] * u.deg, dec_flat[good_coord] * u.deg)
    x_full, y_full = wcs_full.world_to_pixel(coords)
    x_cut = x_full - xmin
    y_cut = y_full - ymin

    xi = np.rint(x_cut).astype(int)
    yi = np.rint(y_cut).astype(int)

    inside = (
        np.isfinite(x_cut) & np.isfinite(y_cut) &
        (xi >= 0) & (xi < mask_cutout.shape[1]) &
        (yi >= 0) & (yi < mask_cutout.shape[0])
    )
    temp = np.zeros_like(xi, dtype=bool)
    temp[inside] = mask_cutout[yi[inside], xi[inside]]
    bad_flat[good_coord] = temp
    return bad_flat.reshape(original_shape)


# =====================================================================
# METHOD: SEGMAP  (default) -- baked binary mask, pure WCS lookup
# =====================================================================
def segmentation_mask(seg_mask_path, RA_fibers, DEC_fibers, confirmation=True):
    """
    Build a (nexp, nfib) bad-fiber mask from a BAKED binary segmentation mask.

    This is the 'segmap' method: all masking INTELLIGENCE (SExtractor detection,
    magnitude cuts, self-mask removals, saturated-star supplements, AND dilation)
    is frozen into one binary FITS per field upstream in the notebook. The
    pipeline's job shrinks to a dumb world->pixel lookup: is each fiber centre
    sitting on a masked pixel?

    seg_mask_path : path to a binary mask FITS (uint8 / bool; nonzero = masked),
        pixel-identical to the field cutout and carrying that cutout's full WCS
        in its header (non-negotiable -- the whole scheme is world->pixel).

    Unlike continuum_image_mask this does NO smoothing, NO percentile threshold,
    NO dilation, NO galaxy-protection raster, and NO field-specific crop: the
    baked mask already IS the cutout with the correct WCS, so the pixel offsets
    are always zero (xmin = ymin = 0). The old hardcoded AEGIS crop lived here
    only because the previous code cropped the full groth.fits at runtime; that
    fragility now moves to bake time, where it is checkable.

    Returns a boolean array shaped like RA_fibers (True = bad / masked).
    """
    with fits.open(seg_mask_path) as hdul:
        # the mask lives in whichever HDU actually carries 2-D image data
        hdu = next((h for h in hdul if getattr(h, "data", None) is not None
                    and np.ndim(h.data) == 2), hdul[0])
        mask = np.asarray(hdu.data).astype(bool)     # nonzero -> masked
        wcs = WCS(hdu.header)

    bad_fiber_mask = flag_fibers_on_mask(
        RA_fibers, DEC_fibers, wcs_full=wcs, mask_cutout=mask, xmin=0, ymin=0,
    )

    if confirmation:
        ra = np.asarray(RA_fibers, dtype=float)
        dec = np.asarray(DEC_fibers, dtype=float)
        print("Same RA and bad_fiber_mask shape?", bad_fiber_mask.shape == ra.shape)
        print("Bad fibers:", int(np.sum(bad_fiber_mask)))
        print("Good fibers:", int(np.sum(~bad_fiber_mask)))
        print("Fraction bad:", np.round(np.mean(bad_fiber_mask), 3))

        # Coverage guard. flag_fibers_on_mask silently treats any fiber that
        # projects OFF the mask footprint as GOOD. The cutouts were cropped to
        # fit the field, but fibers span the whole VDFI footprint including the
        # 57-63" background annuli around edge galaxies -- if the crop is even
        # slightly tight, edge fibers pass unmasked with no error anywhere
        # downstream. This is the one failure mode that produces no traceback,
        # so report it loudly. (One extra world->pixel pass, confirmation-only.)
        finite = np.isfinite(ra.ravel()) & np.isfinite(dec.ravel())
        if np.any(finite):
            x, y = wcs.world_to_pixel(
                SkyCoord(ra.ravel()[finite] * u.deg, dec.ravel()[finite] * u.deg))
            xi, yi = np.rint(x), np.rint(y)
            inside = ((xi >= 0) & (xi < mask.shape[1]) &
                      (yi >= 0) & (yi < mask.shape[0]))
            frac_out = float(1.0 - np.mean(inside))
        else:
            frac_out = 1.0
        print(f"Fibers off mask footprint: {frac_out:.3%}")
        if frac_out > 0:
            print(f"  WARNING: {frac_out:.3%} of finite fibers project OUTSIDE the "
                  f"mask cutout and are therefore treated as GOOD. If this is more "
                  f"than the far background annulus, the cutout crop is too tight.")

    return bad_fiber_mask


# =====================================================================
# METHOD: SPECTRAL  -- threshold each fiber's own continuum level
# =====================================================================
def make_spectral_continuum_mask(spectra, mask_percentile_high=99.5,
                                 mask_percentile_low=0.5, step=5, mode='global'):
    """
    Bad-fiber mask from per-fiber continuum level in the spectra (nexp, nfib, nwave).
    mode='global' uses one threshold across all exposures; 'local' is per-exposure.

    Two INDEPENDENT tails, OR-combined into one mask:
      mask_percentile_high : fibers ABOVE this percentile are masked (bright /
          continuum-leakage tail -- the original behavior of this function).
          None disables this tail.
      mask_percentile_low  : fibers BELOW this percentile are masked (dim /
          noisy tail). None disables this tail.
    At least one of the two must be non-None. Defaults mask the top AND bottom
    0.5% of fiber continuum levels.

    Returns (nexp, nfib) boolean; True = bad.
    """
    mode = mode.lower()
    if mode not in ('global', 'local'):
        raise ValueError("mode must be 'global' or 'local'")
    if mask_percentile_high is None and mask_percentile_low is None:
        raise ValueError(
            "at least one of mask_percentile_high / mask_percentile_low must be set"
        )

    fiber_cont = np.full(spectra.shape[:2], np.nan, dtype=np.float32)
    for i in tqdm(range(spectra.shape[0])):
        fiber_cont[i] = np.nanmedian(spectra[i][:, ::step], axis=1)
    gc.collect()

    def _tail_mask(cont_1d):
        bad = np.zeros(cont_1d.shape, dtype=bool)
        if mask_percentile_high is not None:
            thresh_hi = np.nanpercentile(cont_1d, mask_percentile_high)
            bad |= (cont_1d > thresh_hi)
        if mask_percentile_low is not None:
            thresh_lo = np.nanpercentile(cont_1d, mask_percentile_low)
            bad |= (cont_1d < thresh_lo)
        return bad

    if mode == 'global':
        return _tail_mask(fiber_cont)

    bad_fiber_mask = np.zeros(spectra.shape[:2], dtype=bool)
    for i in range(spectra.shape[0]):
        bad_fiber_mask[i] = _tail_mask(fiber_cont[i])
    return bad_fiber_mask


# =====================================================================
# GALAXY-PROTECTION RADIUS  (spec_* components only)
# =====================================================================
def apply_protect_radius(bad_fiber_mask, RA_fibers, DEC_fibers, gal_RA, gal_DEC,
                         radius_arcsec):
    """
    Force-UNMASK any fiber within radius_arcsec of ANY galaxy position
    (gal_RA/gal_DEC, arrays -- typically an entire run_table, not just one
    target). Returns a NEW boolean array; does not mutate bad_fiber_mask.

    Intended only for the spec_* mask components: segmap/image already avoid
    self-masking real sources their own way (segmap via bake-time segment
    zeroing; image was the failure mode that motivated segmap in the first
    place), so this is spec-specific plumbing, not a general-purpose mask
    combinator.

    RA_fibers/DEC_fibers : any shape matching bad_fiber_mask (typically
        (nexp, nfib)); gal_RA/gal_DEC : 1-D arrays of galaxy positions (deg).
    """
    from scipy.spatial import cKDTree

    bad_fiber_mask = np.asarray(bad_fiber_mask, dtype=bool)
    ra = np.asarray(RA_fibers, dtype=float)
    dec = np.asarray(DEC_fibers, dtype=float)
    if ra.shape != bad_fiber_mask.shape:
        raise ValueError(
            f"RA_fibers shape {ra.shape} does not match bad_fiber_mask shape "
            f"{bad_fiber_mask.shape}"
        )
    gal_ra = np.asarray(gal_RA, dtype=float).ravel()
    gal_dec = np.asarray(gal_DEC, dtype=float).ravel()
    gal_finite = np.isfinite(gal_ra) & np.isfinite(gal_dec)
    gal_ra, gal_dec = gal_ra[gal_finite], gal_dec[gal_finite]

    original_shape = ra.shape
    ra_flat, dec_flat = ra.ravel(), dec.ravel()
    protect = np.zeros(ra_flat.shape, dtype=bool)

    fib_finite = np.isfinite(ra_flat) & np.isfinite(dec_flat)
    if np.any(fib_finite) and gal_ra.size:
        cosd = np.cos(np.radians(np.nanmedian(dec_flat[fib_finite])))
        gal_xy = np.column_stack([gal_ra * cosd, gal_dec])
        fib_xy = np.column_stack([ra_flat[fib_finite] * cosd, dec_flat[fib_finite]])
        tree = cKDTree(gal_xy)
        d, _ = tree.query(fib_xy, k=1)
        protect[fib_finite] = d <= (radius_arcsec / 3600.0)

    protect = protect.reshape(original_shape)
    return bad_fiber_mask & ~protect


# =====================================================================
# METHOD: IMAGE  (LEGACY, A/B validation only)
#   continuum_image_mask + the three helpers below serve ONLY this
#   method. The baked segmap path needs none of them.
# =====================================================================
def continuum_image_mask(cont_image_filepath, field, percentile,
                         RA_fibers, DEC_fibers,
                         confirmation=True, sigma_arcsec=3, radius_pix=np.sqrt(2)):
    """
    LEGACY: build a (nexp, nfib) bad-fiber mask from a continuum image by
    smooth + percentile-threshold + dilate, then flag fibers on masked pixels.

    Kept as the A/B comparison baseline for validating the baked 'segmap'
    method. Note this masks EVERYTHING above the brightness threshold --
    including each catalog galaxy's own bright core -- with no exemptions;
    self-masked cores are exactly the failure mode the baked mask fixes at
    bake time (by zeroing those segments in the mask FITS).
    """
    with fits.open(cont_image_filepath) as hdul:
        header = hdul[0].header.copy()
        image = np.array(hdul[0].data, dtype=np.float32)
        WCS_image = WCS(header)

    structure = make_circular_structure(radius_pix)

    if field == 'AEGIS':
        xmin, xmax, ymin, ymax = 6000, -6800, 10900, -3200   # groth.fits-specific
        cutout = image[ymin:ymax, xmin:xmax]
    elif field == 'COSMOS':
        ny, nx = image.shape
        xmin, ymin = 0, 0
        cutout = image
    else:
        raise ValueError(f"Unknown field {field!r}")

    smooth = nan_safe_gaussian_smooth(cutout, WCS_image, sigma_arcsec)
    threshold = np.nanpercentile(smooth, percentile)
    cont_mask = smooth > threshold
    cont_mask_dilated = binary_dilation(cont_mask.astype(bool), structure=structure)

    bad_fiber_mask = flag_fibers_on_mask(
        RA_fibers, DEC_fibers, wcs_full=WCS_image,
        mask_cutout=cont_mask_dilated, xmin=xmin, ymin=ymin,
    )

    if confirmation:
        print("Same RA and bad_fiber_mask shape?", bad_fiber_mask.shape == RA_fibers.shape)
        print("Bad fibers:", np.sum(bad_fiber_mask))
        print("Good fibers:", np.sum(~bad_fiber_mask))
        print("Fraction bad:", np.round(np.mean(bad_fiber_mask), 3))

    return bad_fiber_mask


# ---- legacy-image helpers (used only by continuum_image_mask) --------
def make_circular_structure(radius_pix):
    r = int(np.ceil(radius_pix))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x ** 2 + y ** 2) <= radius_pix ** 2


def get_pixscale_arcsec(wcs):
    """Approximate pixel scale in arcsec/pixel for a WCS."""
    pixscale_deg = proj_plane_pixel_scales(wcs.celestial) * u.deg
    return pixscale_deg.to(u.arcsec).value


def nan_safe_gaussian_smooth(image, wcs, sigma_arcsec):
    """Smooth an image with a Gaussian whose sigma is fixed in arcsec."""
    img = np.array(image, dtype=np.float32)
    pixscale = get_pixscale_arcsec(wcs)
    sigma_pix_x = sigma_arcsec / pixscale[0]
    sigma_pix_y = sigma_arcsec / pixscale[1]

    valid = np.isfinite(img)
    img_filled = np.where(valid, img, 0)

    smooth_img = gaussian_filter(img_filled, sigma=(sigma_pix_y, sigma_pix_x))
    smooth_weight = gaussian_filter(valid.astype(float), sigma=(sigma_pix_y, sigma_pix_x))

    smooth = smooth_img / smooth_weight
    smooth[smooth_weight == 0] = np.nan
    return smooth


# #####################################################################
# #####################################################################
# OPTICS -- PSF / DAR / continuum helpers  (merged from optics.py, 2026-08-03)
#
# SCOPE NOTE: per pipeline_plan_v2.md, the *binned* extraction (current
# paper) does NOT use Moffat point-source weighting or the DAR terms.
# These functions are the PSF-weighted optimal-extraction path, kept as a
# toolbox for the deferred core-extraction / escape-fraction work. The
# only one the binned loader (load_vdfi_field) touches is
# moffat_psf_integration, and only when build_psf=True. core.py's C1
# socket (extract_core_spectra) is the other consumer.
# #####################################################################
# #####################################################################

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


def moffat_psf_integration(r, seeing, boxsize=14., scale=0.05, alpha=3.0):
    """Moffat PSF profile integrated onto fibers -> (interpolator, R, S, V).

    alpha (the Moffat power-law index; astropy's Moffat2D.alpha, i.e. what
    fitting.moffat_1d/config.PSFSpec call `beta`) defaults to 3.0 as of
    2026-08-04 -- see config.PSFSpec's module comment for the full history.
    It used to be 3.5 here while every stack-level fit/measure function used
    3.0, a genuine disagreement (single-galaxy-pipeline.md Part B3/Open
    Question 1, resolved to option (a): one index everywhere). This changes
    the PSF weights the core extraction (core.extract_core_spectra) builds
    its point-source model from -- ANY EXISTING core-extraction PRODUCT was
    built under the old alpha=3.5 and must be RE-EXTRACTED before being
    trusted against anything reading config.PSFSpec.beta or
    fitting.moffat_encircled_energy_fraction.
    """
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
