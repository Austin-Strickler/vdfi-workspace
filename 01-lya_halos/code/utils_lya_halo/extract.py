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

import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.stats import biweight_location as biweight, mad_std, sigma_clip
from tqdm import tqdm

from .optics import moffat_psf_integration
from .masking import (segmentation_mask, continuum_image_mask,
                      make_spectral_continuum_mask, apply_protect_radius)
from .virial import virial_to_kpc_bins, cosmo   # single cosmology source
from .smoothing import (smooth_interp_then_gauss,
                        smooth_spectrum_nan_safe_adaptive)

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
                      "'segmap', 'image', 'spec_global', 'spec_local'")


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
        return virial_to_kpc_bins(bins, mass, z)
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
# Per-galaxy cache (resumability). The signature hash invalidates stale
# caches automatically whenever any extraction parameter changes -- the
# per-galaxy analogue of the FITS SRCFILE staleness check.
# ---------------------------------------------------------------------
def _extract_signature(config: "PipelineConfig") -> str:
    keys = ("field", "catalog", "catalog_path", "bins", "bin_mode", "mask_method",
            "mask_percentile_image", "mask_percentile_spec_high",
            "mask_percentile_spec_low", "mask_protect_radius_arcsec",
            "spec_mask_step", "bg_inner_arcsec",
            "bg_outer_arcsec", "min_bg_fibers", "smooth_bg", "bg_smooth_method",
            "bg_smooth_sigma_A", "line_mask_halfwidth_A", "smoothing_values",
            "bg_combine_method", "fiber_combine_method", "mass_col", "z_col",
            "wave_min", "wave_max", "nwave")
    d = {k: getattr(config, k) for k in keys}
    d["field"] = config.normalized_field()
    d["bg_combine_method"] = config.resolved_bg_combine_method()  # hash resolved value
    # sigma_clip knobs (optional on config): included so changing them busts the
    # cache, but read defensively so older configs without the fields still hash.
    d["sigma_clip_sigma"] = getattr(config, "sigma_clip_sigma", 3.0)
    d["sigma_clip_maxiters"] = getattr(config, "sigma_clip_maxiters", 5)
    # Baked segmentation mask identity: CONTENT hash, so retweaking the mask
    # FITS in place busts the cache (path/name alone would not change). Cheap
    # 'noseg' for non-segmap runs since resolve_seg_mask_path() returns None.
    d["seg_mask_hash"] = config.seg_mask_hash()
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def extract_fibers(
    config: "PipelineConfig",
    vdfi_data: dict,
    run_table: "Table",
    bad_fiber_mask: np.ndarray,
    use_cache: bool = True,
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

    Per-galaxy results cache to config.cache_dir as gal_<sig>_<i>.npz; with
    use_cache, cached galaxies are reloaded instead of recomputed.

    Returns (Spec, Err, NFIB, Background, vr_edges_used):
      Spec/Err   : (ngal, nrad, nwave)
      NFIB       : (ngal, nrad)   fibers pooled per bin   <- contract HDU
      Background : (ngal, nwave)  galaxy-level background (diagnostic)
      vr_edges   : (ngal, nrad+1) per-galaxy edges actually used
    """
    wave = vdfi_data["wave"]
    RA = vdfi_data["RA"]
    DEC = vdfi_data["DEC"]
    spectra = vdfi_data["spectra"]
    error = vdfi_data["error"]

    nexp = RA.shape[0]
    nfib_total = RA.shape[1]
    nwave = spectra.shape[-1]

    ngal = len(run_table) if config.num_gal is None else min(config.num_gal, len(run_table))
    run_table_use = run_table[:ngal]

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

    Spec = np.full((ngal, nrad, nwave), np.nan)
    Err = np.full((ngal, nrad, nwave), np.nan)
    NFIB = np.zeros((ngal, nrad), dtype=np.int32)
    Background = np.full((ngal, nwave), np.nan)
    VR = np.full((ngal, nrad + 1), np.nan)

    sig = _extract_signature(config)
    cache_dir = config.resolved_cache_dir()
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)

    def _maybe_smooth(spec_1d, z):
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
        cache_path = os.path.join(cache_dir, f"gal_{sig}_{cnt}.npz")
        if use_cache and os.path.exists(cache_path):
            g = np.load(cache_path)
            Spec[cnt], Err[cnt], NFIB[cnt] = g["spec"], g["err"], g["nfib"]
            Background[cnt], VR[cnt] = g["bg"], g["vr"]
            continue

        kpc_per_arcsec = cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265
        radial_bins = radial_bin_edges(config, m, z)
        VR[cnt] = radial_bins

        # ---- precompute distances + good-fiber masks ----
        dist_arcsec_all = np.full((nexp, nfib_total), np.nan)
        good_base_all = np.zeros((nexp, nfib_total), dtype=bool)
        for e in range(nexp):
            dra = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
            ddec = (DEC[e] - dec) * 3600.
            dist_arcsec_all[e] = np.sqrt(dra ** 2 + ddec ** 2)
            good_base_all[e] = (~bad_fiber_mask[e]) & np.isfinite(RA[e]) & np.isfinite(DEC[e])

        # ---- 1. per-exposure backgrounds + galaxy fallback ----
        # float32 backgrounds: the science subtraction below is float32 - bg, so
        # a float64 bg here would silently upcast every science fiber to float64.
        bg_per_exp = np.full((nexp, nwave), np.nan, dtype=np.float32)
        bg_err_per_exp = np.full((nexp, nwave), np.nan, dtype=np.float32)
        has_own_bg = np.zeros(nexp, dtype=bool)
        all_bg_fibers, all_bg_errs = [], []

        for e in range(nexp):
            bg_sel = ((dist_arcsec_all[e] >= config.bg_inner_arcsec) &
                      (dist_arcsec_all[e] < config.bg_outer_arcsec) &
                      good_base_all[e])
            n_bg = int(np.sum(bg_sel))
            if n_bg == 0:
                continue
            sp_bg = spectra[e, bg_sel]
            all_bg_fibers.append(sp_bg)
            if bg_needs_err:
                all_bg_errs.append(error[e, bg_sel])
            if n_bg >= config.min_bg_fibers:
                er_bg = error[e, bg_sel] if bg_needs_err else None
                bg_e, bg_ee = combine_fibers(sp_bg, er_bg, method=bg_method,
                                             sigma=sc_sigma, maxiters=sc_maxiters)
                bg_per_exp[e] = _maybe_smooth(bg_e, z)
                bg_err_per_exp[e] = bg_ee
                has_own_bg[e] = True

        if all_bg_fibers:
            sp_all = np.concatenate(all_bg_fibers, axis=0)
            er_all = np.concatenate(all_bg_errs, axis=0) if bg_needs_err else None
            bg_gal_raw, bg_gal_err = combine_fibers(sp_all, er_all, method=bg_method,
                                                    sigma=sc_sigma, maxiters=sc_maxiters)
            bg_gal = _maybe_smooth(bg_gal_raw, z)
        else:
            bg_gal = np.full(nwave, np.nan)
            bg_gal_err = np.full(nwave, np.nan)
        Background[cnt] = bg_gal

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
        for e in range(nexp):
            v = bin_idx_all[e]
            v = v[v >= 0]
            if v.size:
                NFIB[cnt] += np.bincount(v, minlength=nrad).astype(np.int32)

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
                pool_spec[off:off + k] = spectra[e, sel] - bg_per_exp[e][np.newaxis, :]
                if needs_err:
                    pool_err[off:off + k] = np.sqrt(
                        error[e, sel].astype(np.float32) ** 2
                        + bg_err_per_exp[e][np.newaxis, :] ** 2
                    )
                off += k

            Spec[cnt, r], Err[cnt, r] = combine_fibers_chunked(
                pool_spec, pool_err, method=fiber_method,
                sigma=sc_sigma, maxiters=sc_maxiters, wave_block=wave_block,
            )
            del pool_spec, pool_err   # release this bin before gathering the next

        del bin_idx_all
        gc.collect()                  # keep peak flat across the long per-galaxy run

        if use_cache:
            np.savez(cache_path, spec=Spec[cnt], err=Err[cnt], nfib=NFIB[cnt],
                     bg=Background[cnt], vr=VR[cnt])

    return Spec, Err, NFIB, Background, VR
