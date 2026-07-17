"""
uv_profile.py -- TESTBED for specs/halo-flux-fitting.md Part 3's
UV-continuum extraction pipeline: cutouts, centroiding, circular-annulus
photometry, and background/neighbor masking, from CFHT-LS r-band imaging.

STATUS: standalone tester, not yet merged. Lives here (not extract.py)
deliberately, per Part 3's own "Where this lives during development" note:
this pulls in astropy.nddata/photutils image-I/O machinery that's a
different job from extract.py's Stage-1 fiber-extraction loop, so it's kept
separate until validated against real CFHT-LS mosaics, then merged into
extract.py as a new UV subsection. Nothing in extract.py / config.py /
masking.py is imported-and-modified or monkeypatched here -- only imported
read-only for the pieces that generalize cleanly (see below).

NOT run against real data yet: no CFHT-LS mosaic or per-imaging-pixel
SExtractor segmap is present in this workspace/session, so nothing here has
been exercised end-to-end. Treat this as a from-the-spec first draft to
smoke-test against your real files, not validated code.

This file now covers extraction THROUGH coaddition AND bootstrap: per-galaxy
cutout -> centroid -> per-annulus flux (Sections 1-7), then Section 8
combines those profiles across the galaxy sample into one stacked flux(r),
bootstraps a 16/84 error band by resampling the galaxy axis with
replacement (mirroring measure.bootstrap_all's pattern), and hands the
result straight to fitting.py's Section 8 (fit_naive_uv_exp /
fit_psf_aware_uv_exp / the Sersic variants) via fitting.plot_uv_fit -- that
module is untouched by this one, just imported and called.

Section 6's per-annulus photometry now ALSO masks segmap-flagged pixels
inside each annulus (mask_annulus_neighbors, default True), fixing a real
inconsistency: Section 5's background already excludes every segmap-flagged
object, but until now Section 6 excluded none of them, so a field galaxy
sitting near a bright neighbor was measured against a sky-only background
while its own annuli silently absorbed the neighbor's flux. This masks the
field segmap directly, with NO attempt to distinguish "the target's own
light" from "a neighbor" -- the segmap is manually vetted at creation time
(visually inspected in per-object mini cutouts) to never cover the host
galaxy being extracted, so anything it flags inside an annulus is safe to
treat as a contaminant. See UVExtractConfig.mask_annulus_neighbors.

Reuses, unmodified, from the existing pipeline:
  * masking.get_pixscale_arcsec   -- WCS pixel-scale helper (same one Stage
                                      1.1's fiber masking already uses)
  * virial.physical_kpc_to_arcsec -- per-galaxy kpc -> arcsec, Planck18
                                      (the SAME cosmology source as the Lya
                                      virial-bin machinery)
  * io.load_for_stack             -- THE 450-galaxy RA/Dec/z/mass sample:
                                      same catalog cuts + finite-spectrum cut
                                      already applied to the Lya spectral
                                      stack, per spec's explicit instruction
                                      not to re-query catalog_path fresh.
  * extract.combine_fibers        -- galaxy-axis coadd (Section 8) -- the
                                      SAME function stack.py already reuses
                                      for ITS galaxy-axis coadd (not just
                                      the fiber axis), applied here to a
                                      radial-bin axis instead of wavelength.
  * fitting.plot_uv_fit           -- fit + plot the coadded profile
                                      (Section 8's fit_and_plot_uv_coadd).

Section map:
  0. UVExtractConfig                 -- Part-3-specific knobs, kept separate
                                         from PipelineConfig (config.py is
                                         untouched)
  1. Image / segmap path resolution  -- the Part-3-local equivalent of
                                         config.py's resolve_cont_image_path
                                         (spec's DECIDED rewire target),
                                         reproduced here since this file
                                         doesn't edit config.py yet
  2. Sample catalog                  -- pull RA/Dec/z from the already-built
                                         Lya GalaxyProduct
  3. Field cutout                    -- bounding-box-from-sample + margin;
                                         builds a fresh cutout for EACH field
                                         (neither AEGIS nor COSMOS has one on
                                         disk in this setup); validate_cutout_
                                         covers_sample is available for later
                                         if you ever inherit a pre-cut image
  4. Per-galaxy cutout + centroid    -- 2D-Gaussian centroid, kept-but-
                                         flagged (never dropped)
  5. Background                      -- masked-median-of-field (primary) /
                                         flat-median-of-field (fallback)
  6. Per-galaxy annulus photometry   -- bin-AVERAGE (not summed) flux(r),
                                         background-subtracted, with the
                                         (pre-vetted) segmap masked out of
                                         each annulus
  7. Field-level orchestration       -- single-field-at-a-time loop (per
                                         spec, mirrors normalized_field()'s
                                         treatment of 'BOTH' as stacking-only)
  8. Coaddition + bootstrap + fit/plot -- combine per-galaxy profiles across
                                         the sample (extract.combine_fibers,
                                         stacked in physical kpc, not
                                         arcsec), bootstrap a 16/84 error
                                         band over the galaxy axis, then fit
                                         + plot via fitting.plot_uv_fit
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field as _field
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, biweight_location
import astropy.units as u
from tqdm.auto import tqdm   # auto-picks the Jupyter widget bar over plain-text
                              # when running in a notebook, console bar otherwise

# photutils reorganized centroids/apertures into submodules at v1.0; try the
# current layout first, fall back to the old flat namespace so this runs
# unmodified on either (same compat posture as fitting.py's np.trapezoid /
# np.trapz fallback).
try:
    from photutils.centroids import centroid_2dg
except ImportError:  # photutils < 1.0
    from photutils import centroid_2dg
try:
    from photutils.aperture import CircularAperture, CircularAnnulus
except ImportError:  # photutils < 1.0
    from photutils import CircularAperture, CircularAnnulus

from .masking import get_pixscale_arcsec
from .virial import physical_kpc_to_arcsec
from .io import load_for_stack
from .extract import combine_fibers
from .fitting import plot_uv_fit, moffat_1d, normalize_psf_flux


# =======================================================================
# 0. Part-3-specific config
# =======================================================================
@dataclass
class UVExtractConfig:
    """
    Settings specific to the UV-continuum extraction testbed, kept SEPARATE
    from PipelineConfig (config.py) -- this is new imaging machinery, not a
    change to the Lya spectral pipeline's own config. Shared inputs
    (catalog_path, catalog, mass_col, z_col, id_col, output_dir, ...) are
    read directly off the PipelineConfig you pass alongside this one -- see
    get_uv_sample_catalog / run_uv_extraction_testbed.
    """
    # --- image paths: the Part-3-local equivalent of config.py's
    #     resolve_cont_image_path, per the spec's DECIDED rewire target.
    #     config.py itself is not edited by this file (see module docstring);
    #     when this extraction earns its way into extract.py, this dict is
    #     what config.resolve_cont_image_path's per-field map should become. ---
    image_paths: dict = _field(default_factory=lambda: {
        "AEGIS": "catalogs/images/groth.fits",
        "COSMOS": "catalogs/images/cosmos.fits",
    })

    # --- background segmentation mask, used for bg_method='masked_median'.
    #     Confirmed location: same folder as the CFHT-LS images
    #     (catalogs/images/), named fibermask_EGS.fits / fibermask_COSMOS.fits
    #     -- the identical filenames config.py's resolve_seg_mask_path already
    #     registers for the FIBER-scale mask. CONFIRMED (from a real shape
    #     mismatch) these are NOT pixel-identical to a freshly-built UV field
    #     cutout -- they were baked for a different crop/registration
    #     entirely. run_uv_extraction_testbed handles this by reprojecting
    #     the segmap onto the science image's own WCS grid before use (see
    #     align_segmap_to_grid) -- pixel-identity is NOT assumed anywhere
    #     downstream. Leaving a field's entry out (or the whole dict empty)
    #     falls back to bg_method='flat_median' automatically -- see
    #     compute_field_background. ---
    bg_segmap_paths: dict = _field(default_factory=lambda: {
        "AEGIS": "catalogs/images/fibermask_EGS.fits",
        "COSMOS": "catalogs/images/fibermask_COSMOS.fits",
    })

    # --- per-galaxy cutout stamp sizing ---
    # stamp half-width = cutout_margin_factor * outermost annulus radius, so
    # the stamp comfortably contains the outermost annulus (per spec step 1:
    # "sized to comfortably contain the outermost annulus + background ring").
    cutout_margin_factor: float = 1.5

    # --- centroid (per-galaxy pipeline step 2) ---
    centroid_window_arcsec: float = 3.0        # full width of the 2D-Gaussian fit window
    centroid_offset_flag_arcsec: float = 0.5   # spec's starting threshold; "loosen if it
                                                #   flags an implausibly large fraction"

    # --- background (per-galaxy pipeline step 3) ---
    bg_method: str = "masked_median"    # 'masked_median' (needs bg_segmap_paths) |
                                        #   'flat_median' (simpler fallback, per spec)

    # --- radial bins, in kpc. SMALL/fine bins tailored to this imaging's much
    #     better native resolution (~0.167"/px) -- NOT Part 1's Lya bin
    #     scheme. Per spec you hand-set these per notebook; this default is
    #     just a sane starting point for smoke-testing. ---
    bins_kpc: Sequence[float] = _field(
        default_factory=lambda: [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    )

    # --- field-cutout margin (build_field_cutout / validate_cutout_covers_sample) ---
    # spec: "padded by a buffer of roughly 2x the largest annulus radius
    # (angular size at the sample's lowest z)... a deliberately generous
    # margin 'just to be sure', not a tight minimum."
    field_cutout_margin_factor: float = 2.0

    # --- per-annulus pixel combine method (per-galaxy pipeline step 3,
    #     measure_uv_annuli / _combine_annulus_pixels) ---
    # How the individual pixel values inside one annulus get reduced to a
    # single flux value for that bin. Default is 'biweight' -- NOT 'mean' --
    # matching the biweight combine already used elsewhere in this pipeline
    # (config.fiber_combine_method / galaxy_combine_methods default to
    # 'biweight' too): same robust-average philosophy, applied here to
    # per-annulus PIXEL combines instead of per-fiber/per-galaxy combines.
    #   'biweight' (default) : astropy.stats.biweight_location -- robust to
    #       outlier pixels (cosmic rays, an unmasked faint neighbor at the
    #       annulus edge). Pixels are included via a >50%-overlap threshold,
    #       not fractional weighting (order statistics have no standard
    #       definition for a fractional pixel weight).
    #   'median'              : same thresholded-inclusion convention as
    #       biweight, plain median instead -- simpler, less efficient than
    #       biweight but sometimes useful as a cross-check.
    #   'mean'                : aperture-weighted mean using the EXACT
    #       partial-pixel overlap fraction as a continuous weight -- what
    #       this function used before this option existed, and the
    #       estimator fitting.py's bin_average_* machinery is written
    #       against (mean flux per bin).
    #   'sum'                 : aperture-weighted TOTAL flux in the annulus
    #       (also exact partial-pixel weighting) -- NOT a per-pixel average.
    #       Not directly comparable to fitting.py's bin-average convention
    #       (see fitting.py Part 1's "average, not summed" note) -- use only
    #       if you specifically want a raw aperture sum.
    flux_combine_method: str = "biweight"

    # --- per-annulus neighbor/contaminant masking (measure_uv_annuli) ---
    # Fixes an inconsistency: compute_field_background already excludes
    # every segmap-flagged object from the sky level, but per-annulus
    # photometry previously excluded none of them -- a galaxy sitting near
    # a bright neighbor had that neighbor's flux silently included in its
    # own annuli even though the background estimate assumed it was clean
    # sky. When True (default), each annulus additionally excludes any
    # segmap-flagged pixel it overlaps, with NO attempt to distinguish the
    # target's own light from a neighbor's -- the segmap is manually vetted
    # at creation time (visually inspected in per-object mini cutouts) to
    # never cover the host galaxy being extracted, so masking it outright is
    # safe and needs no connected-component/self-protection logic. Requires
    # a segmap (bg_segmap_paths) to be configured for the field; with none
    # configured this silently has no effect (same fallback posture as
    # bg_method='masked_median' with no segmap).
    mask_annulus_neighbors: bool = True

    # --- PSF (Part-3 forward model) ---
    # The survey seeing is a fixed ANGULAR FWHM per field (CFHT-LS T0007
    # image quality), but becomes a DIFFERENT physical (kpc) FWHM for every
    # galaxy via its own z. Since the profile is stacked on shared kpc bins,
    # the fit uses an AVERAGED effective PSF curve built from the per-galaxy
    # kpc Moffats (build_effective_psf_uv) -- NOT a single scalar psf_fwhm.
    #   psf_fwhm_arcsec : per-field FALLBACK/override seeing in arcsec, used
    #       only when a field's image header carries NONE of the seeing
    #       keywords below. read_seeing_fwhm_arcsec tries the HEADER first
    #       (per spec open question 3), so both fields are normally read
    #       straight from their own stacks -- leave this empty ({}) unless a
    #       field's header genuinely lacks a seeing keyword.
    #   psf_seeing_header_keys : CANDIDATE FITS keywords (arcsec), tried in
    #       order; the CFHT-LS T0007 stacks are not consistent about the
    #       name -- COSMOS/D2 uses IQIMAGE (=0.751"), AEGIS/D3 uses FINALIQ
    #       (=0.83"). Both are the delivered image-quality FWHM of the stack.
    #       Add more names here if another field's header uses yet another.
    #   psf_beta : Moffat wing index. The header gives only the core FWHM,
    #       so beta stays an ASSUMED value until a star-based empirical PSF
    #       is measured (starpsf.py). Shared by every per-galaxy Moffat.
    #   psf_moffat_params : per-field FITTED Moffat PSF, as
    #       {field: {"fwhm_arcsec": .., "beta": ..}} -- the star-based
    #       (FWHM, beta) from fit_moffat_psf run on that field's stacked
    #       stellar PSF (the citable numbers). When a field has an entry
    #       here it OVERRIDES both the header seeing and the global psf_beta
    #       for that field: extraction uses this fwhm_arcsec (not the header)
    #       and stores this beta per galaxy, so build_effective_psf_uv builds
    #       each galaxy's Moffat with its own field's measured core AND wing
    #       shape. Fields absent here fall back to header seeing + psf_beta.
    #   psf_empirical : per-field MEASURED stellar-PSF curve, as
    #       {field: {"r_arcsec": [...], "value": [...]}} -- the stacked-star
    #       radial profile itself (from measure_stellar_psf), saved verbatim
    #       instead of fitting a Moffat to it. This is the "measure once,
    #       paste, done" path: a single Moffat can't match both a sharp core
    #       and heavy wings, so the honest PSF is just the curve. When a field
    #       has an entry here it takes PRECEDENCE over psf_moffat_params /
    #       header seeing in build_effective_psf_uv, which INTERPOLATES this
    #       curve (scaled to each galaxy's kpc via its own z) rather than
    #       evaluating an analytic Moffat. Any normalization is fine (each
    #       galaxy's curve is flux-normalized before averaging). Generate a
    #       paste-ready block with psf_empirical_entry(field, r_mid, stack).
    #   psf_sat_level / psf_sat_frac : the CORE-saturation reject for
    #       measure_stellar_psf, as a FIXED constant instead of the header
    #       SATURATE (inconsistent between fields: AEGIS ~30000, COSMOS ~3000).
    #       A star is rejected if its core max reaches psf_sat_frac x
    #       psf_sat_level. With shape-based clipping now doing the real cleaning
    #       (log-residual vs the biweight stack), this cut is just a cheap guard
    #       against a truly railed core, so it can be set loose/high here.
    psf_fwhm_arcsec: dict = _field(default_factory=dict)
    psf_seeing_header_keys: tuple = ("IQIMAGE", "FINALIQ")
    psf_beta: float = 3.0
    psf_moffat_params: dict = _field(default_factory=dict)
    psf_empirical: dict = _field(default_factory=dict)
    psf_sat_level: float = 5000.0
    psf_sat_frac: float = 0.8


# =======================================================================
# 1. Image / segmap path resolution
# =======================================================================
def resolve_uv_image_path(field: str, uvcfg: UVExtractConfig) -> str:
    """Per-field CFHT-LS r-band image path. field must be the NORMALIZED
    name ('AEGIS' | 'COSMOS') -- pass config.normalized_field(), not
    config.field, the same convention resolve_cont_image_path uses."""
    if field not in uvcfg.image_paths:
        raise ValueError(f"No UV image path configured for field={field!r}; "
                         f"set uvcfg.image_paths[{field!r}] = <path>.")
    return uvcfg.image_paths[field]


def resolve_uv_segmap_path(field: str, uvcfg: UVExtractConfig) -> Optional[str]:
    """Per-field pixel-scale background segmap path, or None if not
    configured (falls back to bg_method='flat_median' -- see
    compute_field_background)."""
    return uvcfg.bg_segmap_paths.get(field)


def _open_image_hdu(path: str):
    """Return the first HDU carrying 2-D image data -- same convention
    masking.segmentation_mask uses for the baked fiber-scale mask, applied
    here to the science image / segmap FITS."""
    hdul = fits.open(path)
    hdu = next((h for h in hdul if getattr(h, "data", None) is not None
               and np.ndim(h.data) == 2), hdul[0])
    return hdul, hdu


def read_seeing_fwhm_arcsec(image_path: str, field: str, uvcfg: UVExtractConfig,
                            *, verbose: bool = True) -> float:
    """
    Seeing FWHM (ARCSEC) for `field`: try each of uvcfg.psf_seeing_header_keys
    in order (default ('IQIMAGE', 'FINALIQ') -- the CFHT-LS T0007 stacks are
    inconsistent: COSMOS/D2 uses IQIMAGE=0.751", AEGIS/D3 uses FINALIQ=0.83")
    FIRST, per halo-flux-fitting.md Part 3's open question 3 ("try the image
    header / survey docs for the seeing FWHM first; fall back to an empirical
    star-based measurement only if no trustworthy documented value exists"),
    then fall back to uvcfg.psf_fwhm_arcsec[field]. Raises if none is found.

    Each candidate keyword is searched across ALL HDUs (T0007 stacks carry
    the image-quality keyword on the primary HDU even when the image data
    lives in an extension). Returned in arcsec -- convert to per-galaxy kpc
    with psf_fwhm_kpc_for_z, never passed to the fit as arcsec directly.
    """
    keys = uvcfg.psf_seeing_header_keys
    if isinstance(keys, str):   # tolerate a single string
        keys = (keys,)
    val = None
    matched_key = None
    try:
        hdul, _ = _open_image_hdu(image_path)
        try:
            for k in keys:
                for h in hdul:
                    if k in h.header:
                        v = float(h.header[k])
                        if np.isfinite(v) and v > 0:
                            val, matched_key = v, k
                            break
                if val is not None:
                    break
        finally:
            hdul.close()
    except Exception:
        val = None

    if val is not None:
        if verbose:
            print(f"read_seeing_fwhm_arcsec [{field}]: header {matched_key}={val:.4f}\" "
                  f"(from {image_path}).")
        return float(val)

    fallback = uvcfg.psf_fwhm_arcsec.get(field)
    if fallback is None or not np.isfinite(fallback) or fallback <= 0:
        raise ValueError(
            f"No seeing FWHM for field={field!r}: none of {tuple(keys)} present in "
            f"{image_path} and no usable uvcfg.psf_fwhm_arcsec[{field!r}] set.")
    if verbose:
        print(f"read_seeing_fwhm_arcsec [{field}]: none of {tuple(keys)} found in "
              f"{image_path}; using uvcfg.psf_fwhm_arcsec[{field!r}]={float(fallback):.4f}\".")
    return float(fallback)


def psf_fwhm_kpc_for_z(fwhm_arcsec, z):
    """
    Convert an ANGULAR seeing FWHM (arcsec) to a PHYSICAL FWHM (kpc) at
    redshift z, using the SAME Planck18 machinery the radial bins use
    (virial.physical_kpc_to_arcsec): arcsec-per-kpc = physical_kpc_to_arcsec(
    1 kpc, z), so kpc = arcsec / that. Vectorized over z. This is the
    per-galaxy conversion that makes a single fixed angular seeing a
    DIFFERENT kpc FWHM for each galaxy -- the reason the stacked fit needs an
    averaged effective PSF (build_effective_psf_uv) rather than one Moffat.
    """
    arcsec_per_kpc = np.asarray(physical_kpc_to_arcsec(1.0, z), dtype=float)
    return np.asarray(fwhm_arcsec, dtype=float) / arcsec_per_kpc


# =======================================================================
# 2. Sample catalog
# =======================================================================
def get_uv_sample_catalog(config, verbose: bool = True):
    """
    Pull the RA/Dec/z/mass sample straight from the already-built Lya
    GalaxyProduct (io.load_for_stack), per halo-flux-fitting.md Part 3's
    explicit instruction: reuse the combined product rather than
    re-querying catalog_path fresh, so this UV-continuum sample is
    pixel-for-pixel the same galaxies already in the Lya profile (same
    catalog cuts AND the same Stage-2 finite-spectrum cut), not a
    similar-but-possibly-drifted list re-derived from the raw catalog.

    config : a PipelineConfig (the SAME one used for the Lya Stage 1/2 run
        you want this UV sample to match -- field, catalog, cut_radial_bin,
        min_good_wave, output_dir all matter here).

    Returns the astropy Table product.catalog (RA, DEC, z, mass, FIELD, ID, ...).
    """
    product = load_for_stack(config, verbose=verbose)
    if verbose:
        print(f"get_uv_sample_catalog: {product.ngal} galaxies "
              f"(same sample as the Lya spectral stack for this config).")
    return product.catalog


# =======================================================================
# 3. Field cutout: bounding-box-from-sample + margin
# =======================================================================
def compute_field_bbox(catalog, field: str, *, z_col: str, bins_kpc,
                       margin_factor: float = 2.0,
                       ra_col: str = "RA", dec_col: str = "DEC",
                       field_col: str = "FIELD"):
    """
    RA/Dec bounding box for `field`'s subsample of `catalog`, padded by
    margin_factor * (largest annulus radius, converted to angular size)
    using WHICHEVER GALAXY IN THE SAMPLE actually needs the largest angular
    margin -- evaluated per galaxy and maxed, not assumed from a single
    "lowest z" shortcut.

    CORRECTED from an earlier version that used the sample's lowest z on
    the assumption that lower z always means larger angular size for a
    fixed physical radius (the spec's own "angular size at the sample's
    lowest z" framing). That assumption only holds below z~1.6: Planck18's
    angular-diameter distance PEAKS around z~1.6 and DECREASES beyond it,
    so for a z~2-3 sample, angular size for a fixed physical (kpc) radius
    actually GROWS with z -- the opposite of the low-z intuition. Confirmed
    directly: the z_min-based version left the highest-z few galaxies in
    both AEGIS and COSMOS just outside the built cutout's footprint
    (2/233 and 2/277) per validate_cutout_covers_sample, which was already
    evaluating per-galaxy and caught what this function's shortcut missed.

    Padding is applied in RA/Dec directly (not re-derived per galaxy at
    cutout time) since this defines ONE shared cutout footprint for the
    whole field, not a per-galaxy stamp (see make_galaxy_cutout for that).
    By construction, using the sample-wide MAX per-galaxy margin here means
    every individual galaxy's own required margin (exactly what
    validate_cutout_covers_sample checks) is guaranteed to be met.
    """
    sel = np.asarray(catalog[field_col]).astype(str) == field
    sub = catalog[sel]
    if len(sub) == 0:
        raise ValueError(f"No galaxies for field={field!r} in this catalog.")

    ra = np.asarray(sub[ra_col], dtype=float)
    dec = np.asarray(sub[dec_col], dtype=float)
    z = np.asarray(sub[z_col], dtype=float)
    z_finite = z[np.isfinite(z) & (z > 0)]
    if len(z_finite) == 0:
        raise ValueError(f"No finite/positive z values for field={field!r}.")

    r_max_kpc = float(np.max(bins_kpc))
    per_gal_arcsec = np.asarray(physical_kpc_to_arcsec(r_max_kpc, z_finite), dtype=float)
    i_max = int(np.argmax(per_gal_arcsec))
    z_at_max_pad = float(z_finite[i_max])
    pad_arcsec = margin_factor * float(per_gal_arcsec[i_max])
    pad_deg = pad_arcsec / 3600.0

    # RA padding widens with declination (angular RA distance shrinks by
    # cos(dec) at fixed sky separation). Use the sample's MOST POLEWARD dec
    # (largest |dec|), not the median -- cos(dec) is smallest (so 1/cos(dec)
    # is largest) there, so this is the conservative choice that keeps every
    # galaxy's RA padding sufficient, not just a "typical" one.
    dec_extreme = float(dec[np.argmax(np.abs(dec))])
    ra_pad = pad_deg / max(np.cos(np.radians(dec_extreme)), 1e-3)
    dec_pad = pad_deg

    return {
        "ra_min": float(np.nanmin(ra) - ra_pad), "ra_max": float(np.nanmax(ra) + ra_pad),
        "dec_min": float(np.nanmin(dec) - dec_pad), "dec_max": float(np.nanmax(dec) + dec_pad),
        "pad_arcsec": pad_arcsec, "n_gal": int(len(sub)), "z_at_max_pad": z_at_max_pad,
    }


def validate_cutout_covers_sample(cutout_path: str, catalog, field: str, *,
                                  z_col: str, bins_kpc, margin_factor: float = 2.0,
                                  ra_col: str = "RA", dec_col: str = "DEC",
                                  field_col: str = "FIELD", verbose: bool = True) -> bool:
    """
    Check whether an EXISTING cutout FITS's WCS footprint covers every
    galaxy's outermost annulus + margin, per spec's AEGIS instruction:
    "confirm its footprint/margin still covers every AEGIS galaxy's
    outermost annulus + background ring... regenerate if not."

    Margin is evaluated PER GALAXY (each galaxy's own z sets its own
    angular scale for the same physical bins_kpc), unlike
    compute_field_bbox's single lowest-z margin used when BUILDING a new
    cutout from scratch -- this is a stricter per-galaxy check, appropriate
    since we're validating an already-fixed footprint.

    Returns True if every galaxy is fully covered with margin; prints a
    summary either way.
    """
    hdul, hdu = _open_image_hdu(cutout_path)
    try:
        wcs = WCS(hdu.header)
        ny, nx = hdu.data.shape
    finally:
        hdul.close()

    sel = np.asarray(catalog[field_col]).astype(str) == field
    sub = catalog[sel]
    ra = np.asarray(sub[ra_col], dtype=float)
    dec = np.asarray(sub[dec_col], dtype=float)
    z = np.asarray(sub[z_col], dtype=float)

    x, y = wcs.celestial.all_world2pix(ra, dec, 0)
    pixscale = get_pixscale_arcsec(wcs)[0]   # arcsec/pixel

    r_max_kpc = float(np.max(bins_kpc))
    req_arcsec = margin_factor * physical_kpc_to_arcsec(r_max_kpc, np.clip(z, 1e-3, None))
    req_pix = np.asarray(req_arcsec, dtype=float) / pixscale

    inside = ((x - req_pix >= 0) & (x + req_pix < nx)
             & (y - req_pix >= 0) & (y + req_pix < ny))
    ok = bool(np.all(inside))

    if verbose:
        n_bad = int(np.sum(~inside))
        print(f"validate_cutout_covers_sample [{field}]: "
              f"{len(sub) - n_bad}/{len(sub)} galaxies fully covered "
              f"(margin={margin_factor}x outermost bin); {n_bad} too close "
              f"to the cutout edge or outside it entirely.")
        if n_bad:
            print("  -> regenerate this field's cutout with build_field_cutout().")
    return ok


def build_field_cutout(full_image_path: str, catalog, field: str, out_path: str, *,
                       z_col: str, bins_kpc, margin_factor: float = 2.0,
                       ra_col: str = "RA", dec_col: str = "DEC",
                       field_col: str = "FIELD", overwrite: bool = False,
                       verbose: bool = True) -> str:
    """
    Build a WCS-aware cutout FITS from the full ~1 deg^2 mosaic, sized to
    the field subsample's RA/Dec bounding box + margin (compute_field_bbox).

    Needed for BOTH fields in the current setup: neither the AEGIS nor the
    COSMOS raw mosaic has a pre-built cutout on disk (the spec assumed
    AEGIS already had one -- confirmed not the case here), so both are cut
    fresh from their full raw images the same way. validate_cutout_covers_
    sample is still useful later if you ever inherit an already-cut image
    for a field (confirm it covers this bin scheme before trusting it);
    it's just not part of the default path for either field right now.

    Uses astropy.nddata.Cutout2D on the RA/Dec bbox's angular extent
    (mode='trim': the cutout is clipped to whatever of the requested
    footprint actually overlaps the full mosaic, rather than padding with
    NaN/erroring -- appropriate since the bbox is derived FROM this same
    mosaic's galaxies, so it should already be fully contained short of a
    catalog/WCS mismatch, which this would surface as a smaller-than-
    -requested cutout instead of silently padding).
    """
    bbox = compute_field_bbox(catalog, field, z_col=z_col, bins_kpc=bins_kpc,
                              margin_factor=margin_factor, ra_col=ra_col,
                              dec_col=dec_col, field_col=field_col)
    ra_c = 0.5 * (bbox["ra_min"] + bbox["ra_max"])
    dec_c = 0.5 * (bbox["dec_min"] + bbox["dec_max"])
    width_deg = bbox["ra_max"] - bbox["ra_min"]
    height_deg = bbox["dec_max"] - bbox["dec_min"]

    hdul, hdu = _open_image_hdu(full_image_path)
    try:
        wcs_full = WCS(hdu.header)
        center = SkyCoord(ra_c * u.deg, dec_c * u.deg)
        size = (height_deg * u.deg, width_deg * u.deg)   # Cutout2D wants (ny, nx) angular size
        cut = Cutout2D(hdu.data, center, size, wcs=wcs_full, mode="trim")

        new_header = hdu.header.copy()
        new_header.update(cut.wcs.to_header())
        out_hdu = fits.PrimaryHDU(data=cut.data, header=new_header)
    finally:
        hdul.close()

    if verbose:
        print(f"build_field_cutout [{field}]: {bbox['n_gal']} galaxies, requested "
              f"footprint {width_deg*3600:.0f}\"x{height_deg*3600:.0f}\" "
              f"(margin {bbox['pad_arcsec']:.1f}\" driven by z={bbox['z_at_max_pad']:.2f}) "
              f"-> wrote {cut.data.shape} px")

    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass overwrite=True to replace it.")
    out_hdu.writeto(out_path, overwrite=overwrite)
    if verbose:
        print(f"  wrote {out_path}")
    return out_path


# =======================================================================
# 4. Per-galaxy cutout + centroid
# =======================================================================
def make_galaxy_cutout(image_data, wcs, ra: float, dec: float, size_arcsec: float):
    """
    Small WCS-aware stamp around (ra, dec), full width size_arcsec, sized by
    the caller to comfortably contain the outermost annulus + background
    ring (see extract_uv_profiles_for_field). Reuses the SAME WCS/pixel-
    -scale pattern masking.py's fiber masking already uses (WCS(header),
    get_pixscale_arcsec, SkyCoord), applied per-galaxy instead of per-fiber.

    mode='partial' + fill_value=NaN: a stamp near the mosaic edge is kept
    (not dropped), with the out-of-image portion filled NaN -- annulus
    photometry below already treats NaN pixels as excluded from the mean,
    not as zero flux.

    Returns the astropy.nddata.Cutout2D object (cutout.data, cutout.wcs).
    """
    pixscale = get_pixscale_arcsec(wcs)[0]
    size_pix = max(size_arcsec / pixscale, 3.0)
    center = SkyCoord(ra * u.deg, dec * u.deg)
    return Cutout2D(image_data, center, size_pix, wcs=wcs, mode="partial", fill_value=np.nan)


def centroid_galaxy(cutout, ra: float, dec: float, *, window_arcsec: float = 3.0,
                    offset_flag_arcsec: float = 0.5, verbose: bool = False) -> dict:
    """
    2D-Gaussian centroid (photutils.centroid_2dg) on `cutout`, windowed to
    +/- window_arcsec/2 around the catalog position (cutout is centered on
    that position by construction) to avoid pulling onto a neighbor.

    DECIDED per spec: kept-but-flagged, never dropped -- mirrors
    success_frac/mask-based QC already used elsewhere in this pipeline
    (measure.bootstrap_measurements) so the flagged fraction can be
    inspected before ever deciding to cut anything. Offsets above
    offset_flag_arcsec (default 0.5", per spec's starting instinct) are
    flagged.

    DECIDED fallback when the fit fails to converge (low S/N, blended
    source): catalog position + flag=True. Simplest option, consistent with
    kept-but-flagged; a multi-seed retry (mirroring
    fitting._default_seeds/_best_of_seeds' philosophy) is only worth adding
    if this simple fallback flags an implausible fraction of the sample in
    practice.

    Returns dict: ra, dec (the ADOPTED position -- fitted or fallback),
    offset_arcsec, fit_ok, flag.
    """
    pixscale = get_pixscale_arcsec(cutout.wcs)[0]
    ny, nx = cutout.data.shape
    yc0, xc0 = ny / 2.0, nx / 2.0   # catalog position, by construction of make_galaxy_cutout

    half_win = max(int(round(0.5 * window_arcsec / pixscale)), 3)
    y0, y1 = int(max(0, yc0 - half_win)), int(min(ny, yc0 + half_win))
    x0, x1 = int(max(0, xc0 - half_win)), int(min(nx, xc0 + half_win))
    window = np.array(cutout.data[y0:y1, x0:x1], dtype=float)

    fit_ok = False
    xcen = ycen = np.nan
    if np.isfinite(window).sum() >= 9:   # need enough finite pixels for a 2D Gaussian fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xc_win, yc_win = centroid_2dg(np.nan_to_num(window, nan=0.0))
            if np.isfinite(xc_win) and np.isfinite(yc_win):
                xcen, ycen = x0 + xc_win, y0 + yc_win
                fit_ok = True
        except Exception:
            fit_ok = False

    if fit_ok:
        ra_fit, dec_fit = cutout.wcs.celestial.all_pix2world(xcen, ycen, 0)
        ra_fit, dec_fit = float(ra_fit), float(dec_fit)
    else:
        ra_fit, dec_fit = float(ra), float(dec)   # DECIDED fallback: catalog position

    offset_arcsec = float(
        SkyCoord(ra * u.deg, dec * u.deg)
        .separation(SkyCoord(ra_fit * u.deg, dec_fit * u.deg))
        .to(u.arcsec).value
    )
    flag = (not fit_ok) or (offset_arcsec > offset_flag_arcsec)

    if verbose:
        print(f"centroid_galaxy: fit_ok={fit_ok}  offset={offset_arcsec:.3f}\"  flag={flag}")

    return {"ra": ra_fit, "dec": dec_fit, "offset_arcsec": offset_arcsec,
           "fit_ok": fit_ok, "flag": bool(flag)}


# =======================================================================
# 5. Background
# =======================================================================
def align_segmap_to_grid(seg_path: str, target_wcs, target_shape, *, verbose: bool = True):
    """
    Reproject a segmentation-mask FITS onto (target_wcs, target_shape)'s
    pixel grid via NEAREST-NEIGHBOR lookup -- a segmap is a categorical/
    binary map (masked vs. not), so interpolating it like a continuous
    image would invent fractional "half-masked" pixels that mean nothing.

    Needed because the fiber-scale segmap (fibermask_EGS.fits /
    fibermask_COSMOS.fits) is NOT pixel-identical to a freshly-built UV
    field cutout -- confirmed directly from a real shape mismatch
    (segmap (7453,7758) vs. a build_field_cutout output (5925,8432)). It
    was baked for a different crop/registration entirely (most likely the
    one masking.segmentation_mask's FIBER-scale masking already uses,
    which explicitly assumes pixel-identity with ITS OWN cutout -- a
    different cutout from this one). Since both FITS carry their own WCS,
    the fix is to walk the target grid's own WCS out to RA/Dec, then look
    up each point in the segmap's WCS -- not to assume the two arrays
    share a pixel grid just because they're the "same kind of file."

    Pixels in the target grid that fall OUTSIDE the segmap's own footprint
    are marked MASKED (conservative: "no segmap coverage there" is not the
    same as "confirmed clean sky," and compute_field_background should not
    treat unknown pixels as safe background).

    Returns a boolean array shaped like target_shape: True = masked
    (source, dilated buffer, or no segmap coverage), False = clean sky --
    pass this straight into compute_field_background's segmap= argument
    (its `seg == 0` unmasked-pixel check works identically on this boolean
    array as on a raw nonzero-means-masked integer segmap).
    """
    seg_hdul, seg_hdu = _open_image_hdu(seg_path)
    try:
        seg_data = np.asarray(seg_hdu.data)
        seg_wcs = WCS(seg_hdu.header)
    finally:
        seg_hdul.close()
    seg_ny, seg_nx = seg_data.shape

    ny, nx = target_shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = target_wcs.celestial.all_pix2world(xx, yy, 0)
    seg_x, seg_y = seg_wcs.celestial.all_world2pix(ra, dec, 0)
    seg_xi = np.round(seg_x).astype(int)
    seg_yi = np.round(seg_y).astype(int)

    in_footprint = ((seg_xi >= 0) & (seg_xi < seg_nx)
                   & (seg_yi >= 0) & (seg_yi < seg_ny))
    aligned = np.ones((ny, nx), dtype=bool)   # default: masked (no segmap coverage)
    aligned[in_footprint] = seg_data[seg_yi[in_footprint], seg_xi[in_footprint]] != 0

    if verbose:
        cov = float(np.mean(in_footprint))
        n_masked = int(np.sum(aligned))
        print(f"align_segmap_to_grid: {seg_path} -> {cov*100:.1f}% of the target "
              f"grid falls inside the segmap's footprint (the rest is treated as "
              f"masked, not confirmed clean sky); {n_masked}/{aligned.size} "
              f"({n_masked/aligned.size*100:.1f}%) pixels masked overall.")
    return aligned


def _stamp_from_field_array(cutout, field_array):
    """
    Cut a per-galaxy boolean stamp out of a full field-grid array (e.g. the
    aligned segmap), aligned pixel-for-pixel with `cutout` by reusing the
    SAME slices Cutout2D itself used to populate cutout.data from the full
    field array (cutout.slices_original / cutout.slices_cutout) -- no fresh
    per-galaxy WCS lookup, just the mapping already computed once when the
    science cutout (make_galaxy_cutout) was made.

    Any part of the stamp outside that mapped region (e.g. a galaxy cutout
    spilling past the field cutout's own edge) defaults to True (masked) --
    the same "unknown = masked, not confirmed clean" posture
    align_segmap_to_grid already uses.
    """
    stamp = np.ones(cutout.data.shape, dtype=bool)
    stamp[cutout.slices_cutout] = np.asarray(field_array, dtype=bool)[cutout.slices_original]
    return stamp


def compute_field_background(image_data, *, segmap=None, sigma_clip_iters: int = 5,
                             verbose: bool = True) -> dict:
    """
    Global background level for one field's cutout: median of unmasked sky
    pixels. Two modes per spec:

      segmap given  : 'masked_median' -- median of pixels NOT flagged in the
          (already lightly-dilated) pixel-scale SExtractor segmentation
          mask (nonzero = masked/source). This is the PRIMARY plan. `segmap`
          must ALREADY be on the same pixel grid as `image_data` -- build it
          with align_segmap_to_grid(seg_path, image_wcs, image_data.shape)
          first; this function does not do any reprojection itself, only
          the shape check below.
      segmap=None   : 'flat_median' fallback -- median of the WHOLE field,
          unmasked. Simpler; per spec, "accept the small risk of slight
          oversubtraction from any bright-object flux that leaks through,
          and revisit only if it turns out to matter."

    A sigma-clip (astropy.stats.sigma_clipped_stats, not explicitly in the
    spec) is applied on top of either mode as a standard robustness step
    against cosmic rays / hot pixels; sigma_clip_iters=0 disables it and
    falls back to a plain median/std.
    """
    data = np.asarray(image_data, dtype=float)
    if segmap is not None:
        seg = np.asarray(segmap)
        if seg.shape != data.shape:
            raise ValueError(f"segmap shape {seg.shape} != image shape {data.shape}; "
                             f"the segmap must be pixel-identical to this cutout.")
        unmasked = data[(seg == 0) & np.isfinite(data)]
        method = "masked_median"
    else:
        unmasked = data[np.isfinite(data)]
        method = "flat_median"

    if sigma_clip_iters > 0 and len(unmasked):
        _, median, std = sigma_clipped_stats(unmasked, sigma=3.0, maxiters=sigma_clip_iters)
    else:
        median = float(np.median(unmasked)) if len(unmasked) else np.nan
        std = float(np.std(unmasked)) if len(unmasked) else np.nan

    if verbose:
        print(f"compute_field_background ({method}): {len(unmasked)} unmasked px, "
              f"median={median:.4g}, std={std:.4g}")
    return {"background": float(median), "std": float(std), "method": method,
           "n_unmasked": int(len(unmasked))}


# =======================================================================
# 6. Per-galaxy annulus photometry
# =======================================================================
def radial_bin_edges_uv(bins_kpc, z: float):
    """Per-galaxy kpc bin edges -> arcsec, using THIS galaxy's z. Reuses
    virial.physical_kpc_to_arcsec directly -- the same Planck18 cosmology
    source the Lya virial-bin machinery uses, just applied to fixed kpc
    edges here instead of R/Rvir edges (Part 3's bins are hand-set kpc
    values, not virial-scaled)."""
    return np.asarray(physical_kpc_to_arcsec(np.asarray(bins_kpc, dtype=float), z), dtype=float)


_FLUX_COMBINE_METHODS = ("biweight", "mean", "median", "sum")


def _combine_annulus_pixels(w, v, method: str) -> float:
    """
    Reduce one annulus's (weight, value) pixel arrays to a single flux
    statistic, per `method` -- see UVExtractConfig.flux_combine_method for
    the full description of each option and why 'biweight' is the default.

    'mean' and 'sum' use the EXACT partial-pixel overlap fraction (`w`,
    from photutils' method="exact" aperture mask) as a continuous weight --
    the physically correct way to handle a pixel that's only fractionally
    inside the annulus for a linear statistic. 'median' and 'biweight' are
    order statistics with no standard definition for a fractional weight,
    so they instead include a pixel fully if more than half of it overlaps
    the annulus (w > 0.5) and exclude it otherwise -- a thresholded mask,
    not a continuous weight -- mirroring how aperture-statistics tools
    generally handle non-sum statistics.

    Returns NaN if no pixels qualify (nothing to combine), never raises for
    that case -- the caller (measure_uv_annuli) checks for NaN and skips
    the bin, same as it always did for an empty/off-image annulus.
    """
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)

    if method == "mean":
        wsum = float(np.sum(w))
        return float(np.sum(w * v) / wsum) if wsum > 0 else np.nan
    if method == "sum":
        return float(np.sum(w * v))
    if method in ("median", "biweight"):
        included = v[w > 0.5]
        if len(included) == 0:
            return np.nan
        return float(np.median(included) if method == "median"
                     else biweight_location(included))
    raise ValueError(f"Unknown flux_combine_method {method!r}; expected one of "
                     f"{_FLUX_COMBINE_METHODS}.")


def measure_uv_annuli(cutout, center_xy, r_edges_arcsec, *, background: float = 0.0,
                      combine_method: str = "biweight", neighbor_mask=None,
                      verbose: bool = False) -> dict:
    """
    Per-annulus flux, background-subtracted, reduced across pixels via
    `combine_method` (see UVExtractConfig.flux_combine_method for the full
    description of 'biweight' (default) / 'mean' / 'median' / 'sum').

    For any combine_method EXCEPT 'sum', this stays a per-pixel AVERAGE
    (mean/median/biweight), matching Part 1's "average, not summed"
    convention (halo-flux-fitting.md Part 3, per-galaxy pipeline step 3),
    so the profile is fit with the SAME bin_average_*-style machinery in
    fitting.py's Section 8 without renormalizing. 'sum' is a genuine
    exception to that convention -- see _combine_annulus_pixels /
    UVExtractConfig's flux_combine_method docstring.

    center_xy : (x, y) pixel coordinates on `cutout` (the ADOPTED centroid,
        from centroid_galaxy -- NOT necessarily the cutout's geometric
        center, since the centroid can shift off the catalog position).
    r_edges_arcsec : radial bin edges in ARCSEC (this galaxy's own,
        from radial_bin_edges_uv) -- converted to pixels internally via
        this cutout's own WCS pixel scale.

    neighbor_mask : optional boolean array shaped like cutout.data, True =
        exclude this pixel (segmap-flagged) -- typically this galaxy's own
        stamp cut from the field segmap via _stamp_from_field_array. REVISED
        from the spec's original DECIDED choice of no per-annulus masking:
        that choice was inconsistent with compute_field_background's own
        masked-median, which already excludes every segmap-flagged object
        from the sky level, while annulus photometry excluded none of them
        -- silently letting a bright neighbor's flux leak into a target
        galaxy's own annuli even though the background assumed clean sky
        there. No attempt is made here to distinguish "the target's own
        light" from "a neighbor" -- the segmap is manually vetted at
        creation time to never cover the host galaxy being extracted, so
        masking every flagged pixel outright is safe. None (default)
        reproduces the old fully-unmasked behavior, e.g. if no segmap is
        configured for this field.

    Returns dict: r_mid_arcsec, flux_mean (background-subtracted, per
    combine_method -- key name kept as flux_mean for continuity even when
    combine_method != 'mean'), flux_mean_raw (not subtracted, for
    diagnostics), npix, neighbor_frac (fraction of otherwise-valid pixels
    in that bin excluded as a neighbor, NaN where neighbor_mask=None or the
    bin has no valid pixels at all) -- all length len(r_edges_arcsec)-1.
    """
    if combine_method not in _FLUX_COMBINE_METHODS:
        raise ValueError(f"combine_method must be one of {_FLUX_COMBINE_METHODS}, "
                         f"got {combine_method!r}.")

    pixscale = get_pixscale_arcsec(cutout.wcs)[0]
    r_edges_pix = np.asarray(r_edges_arcsec, dtype=float) / pixscale
    data = np.asarray(cutout.data, dtype=float)

    if neighbor_mask is not None:
        neighbor_mask = np.asarray(neighbor_mask, dtype=bool)
        if neighbor_mask.shape != data.shape:
            raise ValueError(f"neighbor_mask shape {neighbor_mask.shape} != cutout "
                             f"data shape {data.shape}; build it from THIS cutout via "
                             f"_stamp_from_field_array, not a different galaxy's.")
        neighbor_mask_f = neighbor_mask.astype(float)

    n_bins = len(r_edges_pix) - 1
    flux_mean = np.full(n_bins, np.nan)
    flux_mean_raw = np.full(n_bins, np.nan)
    npix = np.zeros(n_bins, dtype=int)
    neighbor_frac = np.full(n_bins, np.nan)

    for i in range(n_bins):
        r_in, r_out = float(r_edges_pix[i]), float(r_edges_pix[i + 1])
        ap = (CircularAperture(center_xy, r=r_out) if r_in <= 0
             else CircularAnnulus(center_xy, r_in=r_in, r_out=r_out))
        mask = ap.to_mask(method="exact")
        cut = mask.cutout(data, fill_value=np.nan, copy=True)
        if cut is None:
            continue   # aperture entirely off the cutout
        wgt = mask.data
        valid = np.isfinite(cut)
        if not np.any(valid):
            continue

        if neighbor_mask is not None:
            # fill_value=1.0 (excluded) for any sliver outside neighbor_mask's
            # own bounds -- mirrors align_segmap_to_grid's own "unknown =
            # masked, not confirmed clean" posture; in practice neighbor_mask
            # is sized to the full cutout so this only matters at the very
            # edge of the aperture's bounding box.
            nmask_cut = mask.cutout(neighbor_mask_f, fill_value=1.0, copy=True)
            is_neighbor = nmask_cut >= 0.5
            n_pre = int(np.sum(valid))
            valid = valid & ~is_neighbor
            n_excl = n_pre - int(np.sum(valid))
            neighbor_frac[i] = (n_excl / n_pre) if n_pre else np.nan
            if not np.any(valid):
                continue

        w = wgt[valid]
        v = cut[valid]
        combined = _combine_annulus_pixels(w, v, combine_method)
        if not np.isfinite(combined):
            continue
        flux_mean_raw[i] = combined
        flux_mean[i] = combined - background
        npix[i] = int(np.sum(valid))

    r_edges_arr = np.asarray(r_edges_arcsec, dtype=float)
    r_mid_arcsec = 0.5 * (r_edges_arr[:-1] + r_edges_arr[1:])
    if verbose:
        print(f"measure_uv_annuli ({combine_method}): bin values (bg-subtracted):",
              np.round(flux_mean, 4))
        if neighbor_mask is not None:
            print(f"  neighbor-masked fraction per bin: {np.round(neighbor_frac, 3)}")
    return {"r_mid_arcsec": r_mid_arcsec, "flux_mean": flux_mean,
           "flux_mean_raw": flux_mean_raw, "npix": npix, "neighbor_frac": neighbor_frac}


# =======================================================================
# 7. Field-level orchestration: single-field-at-a-time (per spec)
# =======================================================================
def extract_uv_profiles_for_field(field: str, catalog, image_path: str,
                                  uvcfg: UVExtractConfig, *,
                                  ra_col: str = "RA", dec_col: str = "DEC",
                                  z_col: str = "z", field_col: str = "FIELD",
                                  id_col: str = "ID", background: Optional[dict] = None,
                                  field_segmap=None,
                                  verbose: bool = True) -> list:
    """
    Single-field-at-a-time loop over every galaxy in `catalog` with
    FIELD==field: cutout -> centroid -> annuli. Per spec, this mirrors
    PipelineConfig.normalized_field()'s treatment of 'BOTH' as a
    stacking-ONLY concept -- Stage-1-style extraction is always single-
    field, and combination happens after. ONE mosaic is opened and kept
    open for the whole loop (real cost avoided: repeatedly opening/seeking
    a ~1 deg^2 FITS, or holding both mosaics memory-resident at once).

    background : precomputed dict from compute_field_background, or None to
        skip subtraction entirely (e.g. while still validating cutouts/
        centroids, before the background plan is finalized).

    field_segmap : the aligned boolean segmap (align_segmap_to_grid), same
        pixel grid as image_path's data, or None to skip neighbor masking
        entirely (reproduces the old fully-unmasked annulus behavior). Only
        used when uvcfg.mask_annulus_neighbors is also True. Every flagged
        pixel is masked outright, with no self/neighbor distinction -- see
        measure_uv_annuli's docstring for why that's safe here. Built ONCE
        by the caller (run_uv_extraction_testbed) and reused for every
        galaxy here via _stamp_from_field_array's cutout-slicing.

    Returns a list of per-galaxy dicts (one per row of the field subsample,
    in catalog order): id, ra, dec, z, centroid_*, r_mid_arcsec, flux_mean,
    npix, neighbor_frac. NOT yet coadded across galaxies -- see this
    module's docstring for what's next (Section 8: coaddition + bootstrap).
    """
    sel = np.asarray(catalog[field_col]).astype(str) == field
    sub = catalog[sel]
    if verbose:
        print(f"extract_uv_profiles_for_field [{field}]: {len(sub)} galaxies, "
              f"image={image_path}")

    hdul, hdu = _open_image_hdu(image_path)
    try:
        image_data = np.asarray(hdu.data, dtype=float)
        wcs_full = WCS(hdu.header)
        bg_level = background["background"] if background is not None else 0.0

        # PSF for this field. If a star-fitted Moffat is registered
        # (uvcfg.psf_moffat_params[field], from fit_moffat_psf) use its
        # measured FWHM + beta -- the citable, star-based PSF -- overriding
        # both the header seeing and the global psf_beta. Otherwise fall back
        # to the header seeing (read once, per spec) + global psf_beta. Either
        # way the ANGULAR FWHM is fixed for the field and becomes a per-galaxy
        # kpc FWHM inside the loop via each galaxy's z; both FWHM and beta are
        # stored per galaxy so build_effective_psf_uv rebuilds each Moffat with
        # this field's own shape, no re-reading.
        mp = uvcfg.psf_moffat_params.get(field) if uvcfg.psf_moffat_params else None
        if mp is not None:
            seeing_arcsec = float(mp["fwhm_arcsec"])
            psf_beta_field = float(mp.get("beta", uvcfg.psf_beta))
            if verbose:
                print(f"extract_uv_profiles_for_field [{field}]: star-fitted Moffat "
                      f"FWHM={seeing_arcsec:.4f}\", beta={psf_beta_field:.3f} "
                      f"(uvcfg.psf_moffat_params[{field!r}]).")
        else:
            seeing_arcsec = read_seeing_fwhm_arcsec(image_path, field, uvcfg, verbose=verbose)
            psf_beta_field = float(uvcfg.psf_beta)

        results = []
        for i in tqdm(range(len(sub)), desc=f"UV extract {field}"):
            row = sub[i]
            ra, dec, z = float(row[ra_col]), float(row[dec_col]), float(row[z_col])

            r_edges_arcsec = radial_bin_edges_uv(uvcfg.bins_kpc, z)
            stamp_arcsec = 2.0 * uvcfg.cutout_margin_factor * float(r_edges_arcsec[-1])

            cutout = make_galaxy_cutout(image_data, wcs_full, ra, dec, stamp_arcsec)
            cen = centroid_galaxy(cutout, ra, dec,
                                  window_arcsec=uvcfg.centroid_window_arcsec,
                                  offset_flag_arcsec=uvcfg.centroid_offset_flag_arcsec)

            cx, cy = cutout.wcs.celestial.all_world2pix(cen["ra"], cen["dec"], 0)
            center_xy = (float(cx), float(cy))

            neighbor_mask = None
            if uvcfg.mask_annulus_neighbors and field_segmap is not None:
                neighbor_mask = _stamp_from_field_array(cutout, field_segmap)

            prof = measure_uv_annuli(cutout, center_xy, r_edges_arcsec, background=bg_level,
                                     combine_method=uvcfg.flux_combine_method,
                                     neighbor_mask=neighbor_mask)

            results.append({
                "id": row[id_col] if id_col in sub.colnames else i,
                "ra": ra, "dec": dec, "z": z, "field": field,
                "centroid_ra": cen["ra"], "centroid_dec": cen["dec"],
                "centroid_offset_arcsec": cen["offset_arcsec"],
                "centroid_fit_ok": cen["fit_ok"], "centroid_flag": cen["flag"],
                "r_mid_arcsec": prof["r_mid_arcsec"],
                "flux_mean": prof["flux_mean"], "flux_mean_raw": prof["flux_mean_raw"],
                "npix": prof["npix"], "neighbor_frac": prof["neighbor_frac"],
                "psf_fwhm_arcsec": float(seeing_arcsec),
                "psf_fwhm_kpc": float(psf_fwhm_kpc_for_z(seeing_arcsec, z)),
                "psf_beta": float(psf_beta_field),
            })
    finally:
        hdul.close()

    if verbose:
        summarize_uv_extraction(results, label=field)
    return results


def summarize_uv_extraction(results: list, *, label: str = "") -> dict:
    """
    QC summary over a list of per-galaxy dicts from
    extract_uv_profiles_for_field -- mirrors the success_frac-style
    reporting measure.bootstrap_measurements already uses, applied to the
    centroid-offset flag here. Purely a reporting helper; also returns the
    summary dict so it can be logged/asserted on in a notebook.
    """
    n = len(results)
    n_flag = sum(1 for r in results if r["centroid_flag"])
    n_fit_fail = sum(1 for r in results if not r["centroid_fit_ok"])
    offsets = np.array([r["centroid_offset_arcsec"] for r in results], dtype=float)
    summary = {
        "n_galaxies": n,
        "n_flagged": n_flag,
        "frac_flagged": (n_flag / n) if n else np.nan,
        "n_centroid_fit_failed": n_fit_fail,
        "median_offset_arcsec": float(np.nanmedian(offsets)) if n else np.nan,
        "max_offset_arcsec": float(np.nanmax(offsets)) if n else np.nan,
    }
    tag = f" [{label}]" if label else ""
    print(f"summarize_uv_extraction{tag}: {n} galaxies, "
          f"{n_flag} flagged ({summary['frac_flagged']*100:.1f}%, "
          f"{n_fit_fail} of those from a failed centroid fit) -- KEPT, not "
          f"dropped, per spec. Median/max offset = "
          f"{summary['median_offset_arcsec']:.3f}\" / {summary['max_offset_arcsec']:.3f}\".")

    if n and "neighbor_frac" in results[0]:
        all_frac = np.concatenate([np.asarray(r["neighbor_frac"], dtype=float) for r in results])
        finite = all_frac[np.isfinite(all_frac)]
        n_bins_contam = int(np.sum(finite > 0))
        summary["frac_bins_neighbor_contaminated"] = (n_bins_contam / len(finite)) if len(finite) else np.nan
        summary["median_neighbor_frac_when_contaminated"] = (
            float(np.median(finite[finite > 0])) if n_bins_contam else 0.0)
        print(f"  neighbor masking: {n_bins_contam}/{len(finite)} annulus-bins "
              f"({summary['frac_bins_neighbor_contaminated']*100:.1f}%) had >=1 pixel "
              f"masked from the segmap (median excluded fraction where contaminated: "
              f"{summary['median_neighbor_frac_when_contaminated']*100:.1f}%).")
    return summary


# =======================================================================
# Notebook-testing convenience: ties Sections 2/5/7 together for one call.
# =======================================================================
def run_uv_extraction_testbed(config, uvcfg: UVExtractConfig, field: str, *,
                              cutout_image_path: Optional[str] = None,
                              ra_col: str = "RA", dec_col: str = "DEC",
                              id_col: Optional[str] = None,
                              verbose: bool = True) -> list:
    """
    One-call smoke test: get_uv_sample_catalog -> resolve the image path
    (or use cutout_image_path if you've already built/validated a field
    cutout via build_field_cutout / validate_cutout_covers_sample) ->
    compute_field_background -> extract_uv_profiles_for_field.

    config : PipelineConfig for the Lya run whose sample you want to match
        (field is read from config.normalized_field() unless overridden by
        the `field` argument here -- pass field explicitly since you may
        want to test one field while config.field is set to 'BOTH' for the
        Lya side).
    uvcfg  : UVExtractConfig (Section 0).
    cutout_image_path : path to an already-built/validated field cutout
        (see build_field_cutout / validate_cutout_covers_sample). If None,
        falls back to uvcfg.image_paths[field] (the FULL mosaic) -- fine
        for a first smoke test, but slow/memory-heavy on the real ~1 deg^2
        images; build a cutout first for anything beyond a quick check.
    id_col : catalog ID column name. None -> config.id_col.

    Returns the same per-galaxy list extract_uv_profiles_for_field does.
    """
    catalog = get_uv_sample_catalog(config, verbose=verbose)
    image_path = cutout_image_path or resolve_uv_image_path(field, uvcfg)

    hdul, hdu = _open_image_hdu(image_path)
    try:
        image_data = np.asarray(hdu.data, dtype=float)
        image_wcs = WCS(hdu.header)
    finally:
        hdul.close()

    # The segmap is NOT assumed pixel-identical to this image -- it's a
    # separately-cropped FITS with its own WCS (see align_segmap_to_grid's
    # docstring for the real shape mismatch this was built to fix).
    # Reprojected once here, per field, not per galaxy.
    seg_path = resolve_uv_segmap_path(field, uvcfg)
    need_segmap = (uvcfg.bg_method == "masked_median") or uvcfg.mask_annulus_neighbors
    segmap = None
    if need_segmap and seg_path is not None:
        segmap = align_segmap_to_grid(seg_path, image_wcs, image_data.shape, verbose=verbose)
    elif need_segmap and seg_path is None and verbose:
        print(f"run_uv_extraction_testbed [{field}]: bg_method='masked_median' and/or "
              f"mask_annulus_neighbors=True but no segmap configured "
              f"(uvcfg.bg_segmap_paths[{field!r}]) -- background falls back to "
              f"'flat_median' and neighbor masking is skipped entirely.")

    # Same aligned boolean segmap feeds BOTH steps below: compute_field_
    # background excludes every flagged pixel from the sky level, and
    # extract_uv_profiles_for_field excludes every flagged pixel from each
    # annulus (measure_uv_annuli's neighbor_mask) -- one segmap, one
    # masking convention, used consistently everywhere it matters. No
    # labeling/self-protection step: the segmap is manually vetted to never
    # cover the host galaxy being extracted (see UVExtractConfig.
    # mask_annulus_neighbors), so masking every flagged pixel outright is
    # safe for both uses.
    background = compute_field_background(
        image_data, segmap=(segmap if uvcfg.bg_method == "masked_median" else None),
        verbose=verbose)

    return extract_uv_profiles_for_field(
        field, catalog, image_path, uvcfg,
        ra_col=ra_col, dec_col=dec_col, z_col=config.z_col, field_col="FIELD",
        id_col=id_col or config.id_col, background=background,
        field_segmap=(segmap if uvcfg.mask_annulus_neighbors else None), verbose=verbose,
    )


# =======================================================================
# 8. Coaddition across the galaxy sample, bootstrap error bars, then fit +
#    plot.
#
# Reuses extract.combine_fibers COMPLETELY UNCHANGED. It's already
# axis-agnostic -- (n, ncol) -> (ncol,), nothing in its implementation
# actually cares whether the column axis is wavelength or radius -- and
# stack.coadd_galaxies already reuses it for the GALAXY-axis coadd (not
# just the fiber axis it was originally named for). Applying it here to a
# radial-bin axis instead of a wavelength axis is the SAME reuse, not a
# new one: per halo-flux-fitting.md Part 3, "combine across the galaxy
# axis using the SAME config.galaxy_combine_methods... one coadd
# methodology across the whole pipeline, not a second bespoke one for
# imaging."
#
# bootstrap_uv_coadd mirrors measure.bootstrap_all's own pattern exactly:
# resample the galaxy axis WITH replacement (rng.integers(0, ngal, ngal)),
# recombine, collect nboot draws, report a 16/84 percentile band around a
# SEPARATELY-computed fiducial point estimate (the real, unresampled
# coadd) -- not a per-pixel noise model, a galaxy-to-galaxy scatter
# estimate, same quantity/method as the Lya stacks' own bootstrap, just
# applied to the UV radial-bin axis instead of wavelength.
# =======================================================================
def coadd_uv_profiles(results: list, bins_kpc, *, methods: Optional[Sequence[str]] = None,
                      flux_col: str = "flux_mean", weight_col: str = "npix",
                      verbose: bool = True) -> dict:
    """
    Coadd per-galaxy UV profiles (the list extract_uv_profiles_for_field /
    run_uv_extraction_testbed returns) across the galaxy axis into one
    stacked flux(r) profile, once per requested combine method.

    STACKED BY BIN INDEX, IN PHYSICAL KPC (bins_kpc) -- NOT by literal
    arcsec value. Every galaxy was measured on the SAME fixed bins_kpc
    edges (UVExtractConfig.bins_kpc), so bin i always means the same
    physical annulus [bins_kpc[i], bins_kpc[i+1]] for every galaxy -- but
    each galaxy's own r_mid_arcsec differs (radial_bin_edges_uv converts
    kpc -> arcsec using THAT galaxy's own z), so arcsec values are NOT
    comparable across galaxies and are never used as the stacking axis
    here. This mirrors bin_mode='kpc' in the Lya pipeline (fixed physical
    edges shared by every galaxy) rather than bin_mode='virial' (where the
    physical edges themselves vary per galaxy) -- Part 3's bins_kpc was
    always the 'kpc' case, so this is the correct match, not a choice.

    bins_kpc : the SAME UVExtractConfig.bins_kpc every galaxy in `results`
        was extracted with. Passed explicitly (not read off the results)
        so a length mismatch is caught as an error rather than silently
        mis-binned -- see the per-galaxy check below.

    methods : subset of PipelineConfig.galaxy_combine_methods to compute.
        Defaults to ('biweight', 'median', 'sigma_clip', 'weighted_median')
        -- NOT 'mean'/'inv_var', which extract.combine_fibers requires a
        per-row ERROR for. There's still no PER-GALAXY flux uncertainty
        (that would need e.g. a per-galaxy bootstrap over pixels, not
        implemented) to feed those two; the error bars this module DOES
        produce (bootstrap_uv_coadd, below) are galaxy-to-galaxy scatter in
        the STACKED profile, not a per-galaxy input to this function. Add
        'mean'/'inv_var' here only if a per-galaxy error shows up later.

    weight_col : per-bin weight for 'weighted_median' ONLY (every other
        method ignores it) -- 'npix' (valid pixel count in that annulus)
        by default, the direct UV-imaging analogue of "fibers-per-bin"
        (combine_fibers' own docstring: weights are a "pseudo-width" so a
        fully-covered galaxy outweighs a half-covered one). Matches
        stack.coadd_galaxies' convention of letting this weight vary PER
        BIN, not fixed per galaxy, since coverage can vary annulus-to-
        annulus (e.g. a galaxy near the field edge with a clipped outer
        annulus) -- see the per-bin loop below.

    Returns {method: {"r_mid_kpc", "r_edges_kpc", "flux", "err", "n_gal"}},
    where n_gal is the per-bin COUNT of galaxies with a finite value in
    that bin (a galaxy with a NaN in one bin, e.g. an off-image annulus,
    simply doesn't count there but still contributes in bins where it has
    a value -- the same "coverage gaps drop out cleanly" behavior
    combine_fibers already documents).
    """
    if methods is None:
        methods = ("biweight", "median", "sigma_clip", "weighted_median")

    bins_kpc = np.asarray(bins_kpc, dtype=float)
    n_bins = len(bins_kpc) - 1
    r_mid_kpc = 0.5 * (bins_kpc[:-1] + bins_kpc[1:])

    flux_rows, npix_rows = [], []
    for i, r in enumerate(results):
        flux_i = np.asarray(r[flux_col], dtype=float)
        npix_i = np.asarray(r[weight_col], dtype=float)
        if len(flux_i) != n_bins:
            raise ValueError(
                f"results[{i}] (id={r.get('id', i)}) has {len(flux_i)} bins, "
                f"expected {n_bins} from bins_kpc -- was it extracted with a "
                f"different UVExtractConfig.bins_kpc than what you passed here?")
        flux_rows.append(flux_i)
        npix_rows.append(npix_i)
    flux_stack = np.array(flux_rows)   # (n_gal, n_bins)
    npix_stack = np.array(npix_rows)   # (n_gal, n_bins)
    n_gal_per_bin = np.sum(np.isfinite(flux_stack), axis=0).astype(int)

    out = {}
    for method in methods:
        if method == "weighted_median":
            # weight varies PER BIN (npix) here, unlike every other method
            # -- combine_fibers' weighted_median takes exactly one weight
            # per galaxy PER CALL, so loop over bins and call it once per
            # bin with that bin's own npix column, matching how
            # stack.coadd_galaxies loops for its own per-bin-varying nfib.
            flux_out = np.full(n_bins, np.nan)
            err_out = np.full(n_bins, np.nan)
            for i in range(n_bins):
                f_i, e_i = combine_fibers(flux_stack[:, i:i + 1], method=method,
                                          weights=npix_stack[:, i])
                flux_out[i], err_out[i] = f_i[0], e_i[0]
        else:
            flux_out, err_out = combine_fibers(flux_stack, method=method)
        out[method] = {"r_mid_kpc": r_mid_kpc, "r_edges_kpc": bins_kpc,
                       "flux": flux_out, "err": err_out, "n_gal": n_gal_per_bin}
        if verbose:
            med_n = float(np.median(n_gal_per_bin)) if n_bins else float("nan")
            print(f"coadd_uv_profiles ({method}): {n_bins} bins, "
                  f"{med_n:.0f} galaxies/bin (median), {len(results)} galaxies total")
    return out


def bootstrap_uv_coadd(results: list, bins_kpc, *, method: str = "biweight",
                       nboot: int = 500, ci=(16, 84), flux_col: str = "flux_mean",
                       weight_col: str = "npix", seed: Optional[int] = None,
                       verbose: bool = True) -> dict:
    """
    Bootstrap error band for ONE coadd_uv_profiles() method, resampling the
    GALAXY axis with replacement -- mirrors measure.bootstrap_all's own
    pattern exactly: rng.integers(0, ngal, ngal) per draw, recombine,
    collect nboot draws, report ci-percentiles around a fiducial point
    estimate that's computed once from the REAL (unresampled) sample and
    consumes no rng draws. This is galaxy-to-galaxy scatter in the stacked
    profile, not a per-pixel noise model -- the same quantity/method the
    Lya spectral stacks already use, just applied to the UV radial-bin axis
    instead of wavelength.

    results : the per-galaxy list from extract_uv_profiles_for_field /
        run_uv_extraction_testbed -- the SAME list you'd pass to
        coadd_uv_profiles.
    bins_kpc : the SAME UVExtractConfig.bins_kpc every galaxy in `results`
        was extracted with (passed through to coadd_uv_profiles each draw).
    method : which SINGLE combine method to bootstrap (unlike
        coadd_uv_profiles, one method per call -- resampling nboot times
        for every method in one call would multiply the cost for methods
        you may not need error bars on; call this once per method you do).
    nboot : number of bootstrap draws. 500 is a reasonable default for a
        ~200-300 galaxy sample (measure.bootstrap_all's own ballpark); raise
        it if the percentile band looks noisy draw-to-draw, lower it while
        iterating on everything upstream of this call.
    ci : (lower, upper) percentiles for the error band. (16, 84) is the
        usual +/-1sigma-equivalent convention this pipeline uses elsewhere.
    seed : passed straight to np.random.default_rng -- set it for a
        reproducible bootstrap (e.g. when comparing two otherwise-identical
        runs), leave None for a fresh draw each call.

    Returns dict: r_mid_kpc, r_edges_kpc, flux (the fiducial, unresampled
    coadd), err_lo, err_hi (fiducial minus the ci[0] percentile / the
    ci[1] percentile minus fiducial -- both >=0 in the typical case), lo,
    hi (the raw percentile values, in case asymmetric error bars matter to
    you directly), n_gal (fiducial per-bin galaxy count), boot_draws (the
    full (nboot, n_bins) array of every draw's flux, if you want your own
    percentiles or a draw-by-draw diagnostic plot).
    """
    ngal = len(results)
    if ngal == 0:
        raise ValueError("bootstrap_uv_coadd: empty results list.")

    # Fiducial point estimate: the REAL coadd, no resampling, computed once
    # and consuming no rng draws -- same posture as measure.bootstrap_all's
    # own fid_stack.
    fid = coadd_uv_profiles(results, bins_kpc, methods=(method,),
                            flux_col=flux_col, weight_col=weight_col,
                            verbose=False)[method]
    n_bins = len(fid["r_mid_kpc"])

    rng = np.random.default_rng(seed)
    draws = np.full((nboot, n_bins), np.nan)
    for b in tqdm(range(nboot), desc=f"bootstrap ({method})"):
        idx = rng.integers(0, ngal, ngal)
        resampled = [results[i] for i in idx]
        draw = coadd_uv_profiles(resampled, bins_kpc, methods=(method,),
                                 flux_col=flux_col, weight_col=weight_col,
                                 verbose=False)[method]
        draws[b] = draw["flux"]

    lo = np.nanpercentile(draws, ci[0], axis=0)
    hi = np.nanpercentile(draws, ci[1], axis=0)

    if verbose:
        n_bad = int(np.sum(~np.isfinite(lo) | ~np.isfinite(hi)))
        print(f"bootstrap_uv_coadd ({method}): {nboot} draws over {ngal} galaxies, "
              f"{ci[0]}/{ci[1]} percentile band"
              + (f" -- {n_bad}/{n_bins} bins had <2 galaxies in enough draws to "
                 f"produce a finite percentile there." if n_bad else "."))

    return {
        "r_mid_kpc": fid["r_mid_kpc"], "r_edges_kpc": fid["r_edges_kpc"],
        "flux": fid["flux"], "err_lo": fid["flux"] - lo, "err_hi": hi - fid["flux"],
        "lo": lo, "hi": hi, "n_gal": fid["n_gal"], "boot_draws": draws,
    }


def fit_and_plot_uv_coadd(coadd: dict, method: str = "biweight", *, boot: Optional[dict] = None,
                          model: str = "exp", fit_method: str = "naive", n_fixed=None,
                          psf_r=None, psf_vals=None, psf_fwhm: float = 1.0,
                          psf_beta: float = 3.0, fit_skip_inner: int = 1,
                          logy: bool = True, logx: bool = False, xlims=None,
                          figsize=(9, 5), title=None, verbose: bool = True):
    """
    Fit + plot one coadd_uv_profiles() method's stacked profile, via
    fitting.plot_uv_fit -- the function fitting.py's Section 8 was built
    with specifically for this handoff, so nothing about the fit/plot
    machinery is duplicated here; this is just unit/argument plumbing
    from a coadd_uv_profiles() dict to plot_uv_fit's (r_edges, y, yerr)
    call shape.

    coadd  : the dict returned by coadd_uv_profiles.
    method : which combine method's stack to fit -- 'biweight' (default),
        matching UVExtractConfig.flux_combine_method's own default and
        philosophy (robust average, not a raw mean).
    boot   : optional dict from bootstrap_uv_coadd(..., method=method) --
        when given, y-errors come from the REAL bootstrap 16/84 band
        (averaged into a single symmetric sigma for curve_fit) instead of
        combine_fibers' own per-bin scatter. Leave None to fit/plot fast
        off the coadd alone (e.g. while still iterating on extraction),
        pass a bootstrap once you want the "full pipeline" error bars --
        see bootstrap_and_fit_uv for a one-call version of both together.
    model  : 'exp' (default, single exponential, fitting.intrinsic_profile_
        uv_exp) or 'sersic' (fitting.intrinsic_profile_uv_sersic).
    fit_method : 'naive' (default) -- no PSF correction, drops
        fit_skip_inner inner bin(s) (default 1). 'psf' forward-models the
        PSF and keeps every bin. For 'psf', pass the sample's EFFECTIVE PSF
        via psf_r/psf_vals from build_effective_psf_uv(results, uvcfg) --
        the galaxy-average of the per-galaxy kpc Moffats (a fixed angular
        seeing is a different kpc FWHM per galaxy). The scalar psf_fwhm/
        psf_beta path (single Moffat) still works but is a fallback; its
        1.0 default is only a placeholder. 'naive' is the reasonable first
        fit to run before the PSF curve is built.

    The coadd's r_mid_kpc / r_edges_kpc feed plot_uv_fit's r_edges
    directly (already in physical kpc, per coadd_uv_profiles' docstring on
    why arcsec is never the stacking axis) -- h_uv / r_e come back in kpc.

    Any bin with a non-finite or non-positive error (e.g. a bin with 0-1
    contributing galaxies) is floored to a small fraction of the profile's
    peak so curve_fit has a usable sigma there, rather than silently
    dropping the bin or raising -- flagged via a printed count when it
    happens.

    Returns (fig, ax, fit_result) from fitting.plot_uv_fit.
    """
    if method not in coadd:
        raise KeyError(f"method={method!r} not in this coadd (have: {list(coadd)}). "
                       f"Pass a method you actually computed in coadd_uv_profiles.")
    stack = coadd[method]
    r_edges = stack["r_edges_kpc"]
    y = np.asarray(stack["flux"], dtype=float)

    if boot is not None:
        yerr = 0.5 * (np.asarray(boot["err_lo"], dtype=float)
                      + np.asarray(boot["err_hi"], dtype=float))
        err_source = f"bootstrap_uv_coadd's {len(boot.get('boot_draws', []))}-draw 16/84 band"
    else:
        yerr = np.asarray(stack["err"], dtype=float)
        err_source = "combine_fibers' own per-bin scatter (no bootstrap passed)"

    bad_err = ~np.isfinite(yerr) | (yerr <= 0)
    if np.any(bad_err):
        finite_y = y[np.isfinite(y)]
        floor = 0.01 * float(np.max(np.abs(finite_y))) if len(finite_y) else 1.0
        floor = max(floor, 1e-12)
        yerr = np.where(bad_err, floor, yerr)
        if verbose:
            print(f"fit_and_plot_uv_coadd: {int(np.sum(bad_err))} bin(s) had a "
                 f"non-finite/zero error (likely <2 galaxies contributing there -- "
                 f"see n_gal) -- floored to {floor:.3g} so curve_fit has a usable sigma.")

    if verbose:
        print(f"fit_and_plot_uv_coadd: y-errors from {err_source}.")

    return plot_uv_fit(
        r_edges, y, yerr, model=model, method=fit_method, fit_skip_inner=fit_skip_inner,
        n_fixed=n_fixed, psf_r=psf_r, psf_vals=psf_vals, psf_fwhm=psf_fwhm,
        psf_beta=psf_beta, logy=logy, logx=logx, xlims=xlims, figsize=figsize,
        title=title or f"UV-continuum coadd ({method}, {len(y)} bins)",
        verbose=verbose,
    )


def measure_stellar_psf(stars, image_path, uvcfg: UVExtractConfig, field, *,
                        r_edges_arcsec=None, sat_core_arcsec=None,
                        sky_r_in_arcsec=None, sky_r_out_arcsec=None,
                        sat_frac=None, sat_level=None,
                        combine_method: str = "biweight",
                        core_norm_arcsec=None, clip_sigma: float = 4.0,
                        clip_iters: int = 3, min_keep: int = 5,
                        n_boot: int = 2000, rng_seed: int = 0,
                        min_finite_frac: float = 0.7, stamp_pad_factor: float = 1.3,
                        ra_col: str = "ra", dec_col: str = "dec", verbose: bool = True):
    """
    Measure the empirical stellar PSF for `field`: each star's radial profile
    (in ARCSEC -- stars have no z) with source masking OFF (a star IS the
    PSF), combined into one core-normalized BIWEIGHT stack -- the measured PSF
    to save in uvcfg.psf_empirical (via psf_empirical_entry) or hand to
    fit_moffat_psf.

    Robustness (three layers, all attacking a bright/saturated survivor):
      * Each star is normalized by its CORE INTEGRAL (area-weighted mean
        surface brightness within core_norm_arcsec), not by its single peak
        pixel. A saturated core has a clipped, flat top, so np.nanmax reads too
        low and dividing by it inflates the whole curve upward -- the "stays
        high with a bright tail" signature. A small-core integral is stable
        against one clipped/noisy central pixel.
      * Stars are combined with biweight_location (per bin), not a plain mean,
        so a surviving outlier is downweighted rather than dragging the stack.
      * SHAPE-BASED rejection: iterate clip_iters times -- build the biweight
        stack, score each star by the RMS log-residual to it (over bins where
        the stack is above 1e-3 of peak), and drop stars beyond clip_sigma x
        MAD of that score. This catches saturated survivors an absolute-counts
        cut misses (flat core / bright extended tail), which is exactly what is
        visible by eye. Never drops below min_keep stars.

    The per-bin ERROR is the BOOTSTRAP of the biweight stack: resample the kept
    stars with replacement n_boot times, re-stack, take the per-bin std. This
    is the uncertainty ON THE STACK (with the sqrt(N) shrinkage built in) for
    the actual biweight estimator, not the outlier-driven np.nanstd population
    spread. Returned as qc['stack_err'] -- feed THAT as fit_moffat_psf's sigma.

    Fixes the two stamp-coupled artifacts of the earlier notebook loop, where
    stamp_arcsec silently leaked into the result:

      * SATURATION is checked ONLY in the core -- pixels within
        sat_core_arcsec of the fitted centroid -- not np.nanmax over the whole
        stamp. A saturated NEIGHBOR far out in a large stamp no longer rejects
        an otherwise-clean star, so the surviving-star count stops depending
        on stamp size (the cause of "51 stars at 5x, 12 at 30x").
      * BACKGROUND is the sigma-clipped median of a fixed SKY ANNULUS
        [sky_r_in, sky_r_out] around the centroid, not the whole-stamp median.
        The subtracted sky no longer drifts with stamp size or with how much
        of the stamp the star's own wings fill (the cause of the profile
        SHAPE changing with stamp).

    The stamp is sized INTERNALLY to just hold the sky annulus + pad, so it is
    no longer a free knob. As long as it clears sky_r_out it cannot affect the
    profile -- which is the whole point.

    Defaults (all overridable): r_edges_arcsec = 0..4.5" in ~1.3-pixel bins
    (>= 1 pixel so the core isn't sub-sampled); sat_core = 1.5 x header seeing
    FWHM; sky annulus = [r_edges[-1], 1.5 x r_edges[-1]] -- just outside the
    PSF (< ~1e-4 of peak there, i.e. clean sky).

    sat_frac / sat_level : the core-saturation cut, defaulting to
    uvcfg.psf_sat_frac / uvcfg.psf_sat_level (a FIXED constant, NOT the header
    SATURATE, which is inconsistent between fields -- AEGIS ~30000, COSMOS
    ~3000). A star is rejected if its core max reaches sat_frac x sat_level.
    With the shape clip doing the real cleaning this is just a loose guard, so
    it can be set high in the config. Pass either here to override per call.

    Returns (r_mid_arcsec, stack, profiles, qc):
      r_mid_arcsec : bin midpoints (len n_bins)
      stack        : core-normalized BIWEIGHT PSF (len n_bins)
      profiles     : (n_kept, n_bins) individual core-normalized profiles,
                     AFTER shape-based clipping
      qc           : counts (incl. shape_clipped) + resolved sat_core/sky/stamp
                     + stack_err (bootstrap per-bin 1-sigma on the stack)
    Feed (r_mid_arcsec, stack, qc['stack_err']) to fit_moffat_psf, or save
    (r_mid_arcsec, stack) with psf_empirical_entry.
    """
    if sat_level is None:
        sat_level = float(getattr(uvcfg, "psf_sat_level", 5000.0))
    if sat_frac is None:
        sat_frac = float(getattr(uvcfg, "psf_sat_frac", 0.8))
    if r_edges_arcsec is None:
        r_edges_arcsec = np.arange(0.0, 4.501, 0.15)
    r_edges_arcsec = np.asarray(r_edges_arcsec, dtype=float)
    r_mid = 0.5 * (r_edges_arcsec[:-1] + r_edges_arcsec[1:])
    r_out = float(r_edges_arcsec[-1])

    seeing_hdr = read_seeing_fwhm_arcsec(image_path, field, uvcfg, verbose=False)
    if sat_core_arcsec is None:
        sat_core_arcsec = 1.5 * seeing_hdr
    if sky_r_in_arcsec is None:
        sky_r_in_arcsec = r_out
    if sky_r_out_arcsec is None:
        sky_r_out_arcsec = 1.5 * r_out
    stamp_arcsec = 2.0 * stamp_pad_factor * float(sky_r_out_arcsec)

    # core-normalization radius (half the seeing FWHM), always >= the first
    # bin so at least one bin is averaged; annulus areas (2*pi*r*dr) weight it.
    if core_norm_arcsec is None:
        core_norm_arcsec = max(0.5 * seeing_hdr, float(r_mid[0]))
    dr = np.diff(r_edges_arcsec)
    area_w = 2.0 * np.pi * r_mid * dr
    core_bins = r_mid <= core_norm_arcsec
    if not np.any(core_bins):
        core_bins = np.zeros(len(r_mid), dtype=bool)
        core_bins[0] = True

    hdul, hdu = _open_image_hdu(image_path)
    try:
        image_data = np.asarray(hdu.data, dtype=float)
        wcs = WCS(hdu.header)
    finally:
        hdul.close()

    profiles = []
    kept = n_sat = n_offimg = n_badfit = 0
    for row in stars:
        ra, dec = float(row[ra_col]), float(row[dec_col])
        cutout = make_galaxy_cutout(image_data, wcs, ra, dec, stamp_arcsec)
        data = np.asarray(cutout.data, dtype=float)
        pixscale = get_pixscale_arcsec(cutout.wcs)[0]

        cen = centroid_galaxy(cutout, ra, dec,
                              window_arcsec=uvcfg.centroid_window_arcsec,
                              offset_flag_arcsec=uvcfg.centroid_offset_flag_arcsec)
        if not cen["fit_ok"]:
            n_badfit += 1
            continue
        cx, cy = cutout.wcs.celestial.all_world2pix(cen["ra"], cen["dec"], 0)

        ny, nx = data.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        rr = np.hypot(xx - float(cx), yy - float(cy)) * pixscale

        # saturation: CORE only
        core = data[(rr <= sat_core_arcsec) & np.isfinite(data)]
        if len(core) == 0 or np.max(core) >= sat_frac * sat_level:
            n_sat += 1
            continue

        # coverage: the measured region must be mostly on-image
        meas = rr <= r_out
        if not np.any(meas) or np.isfinite(data[meas]).mean() < min_finite_frac:
            n_offimg += 1
            continue

        # background: sigma-clipped median of a FIXED sky annulus
        sky_pix = data[(rr >= sky_r_in_arcsec) & (rr <= sky_r_out_arcsec) & np.isfinite(data)]
        if len(sky_pix) < 10:
            n_offimg += 1
            continue
        _, bg, _ = sigma_clipped_stats(sky_pix, sigma=3.0, maxiters=5)

        prof = measure_uv_annuli(cutout, (float(cx), float(cy)), r_edges_arcsec,
                                 background=float(bg), combine_method=combine_method,
                                 neighbor_mask=None)
        f = np.asarray(prof["flux_mean"], dtype=float)
        # normalize by the CORE INTEGRAL (area-weighted mean surface brightness
        # within core_norm_arcsec), robust to a clipped/noisy single peak pixel
        norm = np.nansum((f * area_w)[core_bins]) / np.nansum(area_w[core_bins])
        if not np.isfinite(norm) or norm <= 0:
            n_badfit += 1
            continue
        profiles.append(f / norm)
        kept += 1

    profiles = np.array(profiles) if kept else np.zeros((0, len(r_mid)))

    def _biw_stack(P):
        """Per-bin biweight_location over rows (nan-safe; median if < 3 pts)."""
        out = np.full(P.shape[1], np.nan)
        for j in range(P.shape[1]):
            col = P[:, j]
            col = col[np.isfinite(col)]
            if col.size >= 3:
                out[j] = biweight_location(col)
            elif col.size:
                out[j] = np.median(col)
        return out

    # SHAPE-BASED rejection: iteratively drop stars whose profile is an outlier
    # relative to the biweight stack. Score = RMS of log10(profile) - log10(stack)
    # over bins where the stack is above 1e-3 of peak (the mid/wing bins that
    # actually discriminate; the core is ~1 for everyone by construction). A
    # saturated survivor -- flat core inflated by normalization, bright tail --
    # is a multi-bin high outlier and clips out; robust MAD threshold, and never
    # below min_keep stars.
    keep = np.ones(len(profiles), dtype=bool)
    n_shape = 0
    for _ in range(int(clip_iters)):
        if keep.sum() < max(min_keep, 3):
            break
        S = _biw_stack(profiles[keep])
        good_bins = np.isfinite(S) & (S > 1e-3 * np.nanmax(S))
        with np.errstate(divide="ignore", invalid="ignore"):
            logres = np.log10(profiles) - np.log10(S)
        metric = np.full(len(profiles), np.nan)
        for j in range(len(profiles)):
            v = logres[j][good_bins]
            v = v[np.isfinite(v)]
            if v.size:
                metric[j] = np.sqrt(np.mean(v ** 2))
        med = np.nanmedian(metric[keep])
        mad = 1.4826 * np.nanmedian(np.abs(metric[keep] - med))
        if not np.isfinite(mad) or mad <= 0:
            break
        new_keep = keep & np.isfinite(metric) & (metric <= med + float(clip_sigma) * mad)
        if new_keep.sum() < min_keep or new_keep.sum() == keep.sum():
            break
        keep = new_keep
    n_shape = int((~keep).sum())
    profiles = profiles[keep]
    kept = len(profiles)

    stack = _biw_stack(profiles) if kept else np.full(len(r_mid), np.nan)

    # per-bin error = BOOTSTRAP of the biweight stack over STARS (resample with
    # replacement, re-stack, take per-bin std). Robust estimator -> a surviving
    # bright star can't blow this up the way np.nanstd did.
    if kept >= 2 and int(n_boot) > 0:
        rng = np.random.default_rng(rng_seed)
        boot = np.empty((int(n_boot), len(r_mid)), dtype=float)
        for b in range(int(n_boot)):
            boot[b] = _biw_stack(profiles[rng.integers(0, kept, kept)])
        stack_err = np.nanstd(boot, axis=0)
    else:
        stack_err = np.full(len(r_mid), np.nan)

    qc = {"kept": kept, "saturated": n_sat, "off_image": n_offimg,
          "bad_centroid": n_badfit, "shape_clipped": n_shape,
          "seeing_hdr_arcsec": float(seeing_hdr),
          "sat_core_arcsec": float(sat_core_arcsec), "sat_level": float(sat_level),
          "sat_frac": float(sat_frac), "core_norm_arcsec": float(core_norm_arcsec),
          "sky_annulus_arcsec": (float(sky_r_in_arcsec), float(sky_r_out_arcsec)),
          "stamp_arcsec": float(stamp_arcsec), "stack_err": stack_err}
    if verbose:
        print(f"measure_stellar_psf [{field}]: kept {kept} stars (rejected "
              f"{n_sat} saturated-core, {n_offimg} off-image/low-coverage, "
              f"{n_badfit} bad centroid, {n_shape} shape-outlier). biweight "
              f"stack, bootstrap err (n_boot={int(n_boot)}). "
              f"sat_core={sat_core_arcsec:.2f}\" (>= {sat_frac * sat_level:.0f} "
              f"cts rejects), core_norm={core_norm_arcsec:.2f}\", "
              f"sky=[{sky_r_in_arcsec:.2f},{sky_r_out_arcsec:.2f}]\", "
              f"stamp={stamp_arcsec:.1f}\" (internal).")
    return r_mid, stack, profiles, qc


def fit_moffat_psf(r_arcsec, psf_profile, *, sigma=None, fwhm0: float = 0.8,
                   beta0: float = 3.0, seeds=None, fit_log: bool = True,
                   verbose: bool = True):
    """
    Fit an analytic Moffat -- BOTH FWHM and beta FREE -- to a measured PSF
    profile (e.g. a stacked-star curve), giving the citable per-field PSF
    parameters to register in UVExtractConfig.psf_moffat_params.

    Model: A * moffat_1d(r, fwhm, beta), with amplitude A a free nuisance
    parameter so the input profile need not be perfectly peak-normalized.

    fit_log (default True): fit log10(profile) instead of the linear profile,
    so every radius carries ~equal weight across the PSF's several decades.
    This is what makes the fit follow the WINGS -- the whole reason to measure
    a stellar PSF -- rather than being dominated by the near-unity core (a
    plain linear least-squares essentially ignores everything below ~1% of
    peak, i.e. exactly the wing excess that pins down beta). Only finite,
    POSITIVE bins are used (log needs > 0).

    sigma : optional per-bin 1-sigma (same length as psf_profile), e.g. the
    star-to-star scatter of the stellar stack. In log-fit mode it is mapped
    to a log-space error (sigma / profile / ln10). None -> unweighted. Bins
    with zero or non-finite sigma are DROPPED from the fit (a peak-normalized
    stack has zero scatter in its central bin -- that bin is simply ignored
    rather than given infinite weight), so a raw np.nanstd weight is safe to
    pass as-is.

    seeds : optional list of (fwhm0, beta0) starting points. The fit is run
    from each, plus the explicit (fwhm0, beta0), and the LOWEST-cost converged
    result is kept -- guards against settling in a shallow local minimum along
    the FWHM<->beta degeneracy. None -> a diverse default grid (narrow..broad
    cores x light..heavy wings).

    Returns dict: fwhm, beta (best fit), fwhm_err, beta_err (1-sigma from the
    covariance), A, r_arcsec, model (fitted curve sampled on r_arcsec),
    n_points, cost, n_seeds_ok, success. Drop fwhm/beta straight into
    uvcfg.psf_moffat_params[field].
    """
    from scipy.optimize import curve_fit
    r = np.asarray(r_arcsec, dtype=float)
    y = np.asarray(psf_profile, dtype=float)
    good = np.isfinite(r) & np.isfinite(y) & (y > 0)
    # Drop bins with no usable weight. A peak-normalized stack has EXACTLY
    # zero star-to-star scatter in its central bin (every star = 1 there),
    # and often in sparse outer bins; a 0 (or NaN) sigma becomes an infinite
    # 1/sigma weight, which makes curve_fit's residuals non-finite at the
    # initial point ("Residuals are not finite ..."). Excluding those bins
    # here -- rather than flooring them -- is the "just ignore that bin" fix:
    # the free amplitude + log-space fit still pin FWHM/beta from the rest.
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        good &= np.isfinite(sigma) & (sigma > 0)
    if good.sum() < 3:
        return {"success": False,
                "reason": "fewer than 3 finite positive bins with usable sigma"}
    rg, yg = r[good], y[good]
    sg = sigma[good] if sigma is not None else None

    if fit_log:
        def model(rr, logA, fwhm, beta):
            return logA + np.log10(moffat_1d(rr, fwhm=fwhm, beta=beta))
        ydata = np.log10(yg)
        s = (sg / (yg * np.log(10.0))) if sg is not None else None
        amp0 = float(np.log10(np.nanmax(yg)))
        bounds = ([-np.inf, 0.05, 1.05], [np.inf, 5.0, 15.0])
    else:
        def model(rr, A, fwhm, beta):
            return A * moffat_1d(rr, fwhm=fwhm, beta=beta)
        ydata = yg
        s = sg
        amp0 = float(np.nanmax(yg))
        bounds = ([0.0, 0.05, 1.05], [np.inf, 5.0, 15.0])

    # Multi-seed: try a diverse grid of (FWHM, beta) starts and keep the
    # lowest-cost converged fit. A single Moffat's log-space cost surface can
    # have shallow local minima along the FWHM<->beta degeneracy (a sharper
    # core with heavier wings trades off against a broader core with lighter
    # wings), so one start can settle short of the best compromise -- exactly
    # the "I could clearly do better by hand" feeling. Seeds span narrow/broad
    # cores and light/heavy wings; the explicit (fwhm0, beta0) is always tried.
    if seeds is None:
        seeds = [(fw, be) for fw in (0.5, 0.7, 0.9, 1.2) for be in (1.5, 2.5, 3.5, 5.0)]
    seeds = list(seeds) + [(fwhm0, beta0)]

    best = None
    n_ok = 0
    for fw0, be0 in seeds:
        fw0 = float(np.clip(fw0, bounds[0][1] * 1.01, bounds[1][1] * 0.99))
        be0 = float(np.clip(be0, bounds[0][2] * 1.01, bounds[1][2] * 0.99))
        try:
            popt, pcov = curve_fit(model, rg, ydata, p0=[amp0, fw0, be0], sigma=s,
                                   absolute_sigma=False, bounds=bounds, maxfev=20000)
        except Exception:
            continue
        resid = model(rg, *popt) - ydata
        cost = float(np.sum((resid / s) ** 2)) if s is not None else float(np.sum(resid ** 2))
        n_ok += 1
        if best is None or cost < best["cost"]:
            best = {"popt": popt, "pcov": pcov, "cost": cost}

    if best is None:
        return {"success": False, "reason": "all seeds failed to converge"}
    popt, pcov = best["popt"], best["pcov"]
    perr = np.sqrt(np.diag(pcov))
    A = 10.0 ** popt[0] if fit_log else float(popt[0])
    fwhm, beta = float(popt[1]), float(popt[2])
    fwhm_err, beta_err = float(perr[1]), float(perr[2])
    if verbose:
        print(f"fit_moffat_psf: FWHM = {fwhm:.4f} +/- {fwhm_err:.4f}\"   "
              f"beta = {beta:.3f} +/- {beta_err:.3f}   "
              f"({'log' if fit_log else 'linear'} fit over {int(good.sum())} bins, "
              f"best of {n_ok}/{len(seeds)} seeds)")
    return {"success": True, "fwhm": fwhm, "beta": beta,
            "fwhm_err": fwhm_err, "beta_err": beta_err, "A": float(A),
            "r_arcsec": r, "model": A * moffat_1d(r, fwhm=fwhm, beta=beta),
            "n_points": int(good.sum()), "cost": best["cost"], "n_seeds_ok": n_ok}


def psf_empirical_entry(field, r_arcsec, value, *, printout: bool = True):
    """
    Turn a MEASURED stellar-PSF curve into a paste-ready entry for
    uvcfg.psf_empirical[field] -- the "measure once, paste, done" path.

    Run measure_stellar_psf to get (r_mid, stack_stars), pass them here, and
    paste the printed block into the config. No Moffat is fitted: the saved
    curve IS the PSF, and build_effective_psf_uv interpolates it (scaled to
    each galaxy's kpc) instead of evaluating an analytic model. Store it as
    measured -- normalization is irrelevant because each galaxy's curve is
    flux-normalized before averaging.

    Only finite bins are kept. Returns {"r_arcsec": [...], "value": [...]};
    by default also prints the config block.
    """
    r = np.asarray(r_arcsec, dtype=float)
    v = np.asarray(value, dtype=float)
    good = np.isfinite(r) & np.isfinite(v)
    r, v = r[good], v[good]
    entry = {"r_arcsec": [float(x) for x in r], "value": [float(x) for x in v]}
    if printout:
        rr = ", ".join(f"{x:.4f}" for x in r)
        vv = ", ".join(f"{x:.6e}" for x in v)
        print(f'# {field}: {len(r)} bins -- paste into uvcfg.psf_empirical')
        print(f'psf_empirical["{field}"] = {{')
        print(f'    "r_arcsec": [{rr}],')
        print(f'    "value":    [{vv}],')
        print('}')
    return entry


def build_effective_psf_uv(results: list, uvcfg: UVExtractConfig, *,
                           r_max_kpc: Optional[float] = None, n_grid: int = 400,
                           psf_key: str = "psf_fwhm_kpc", verbose: bool = True):
    """
    Build the sample's EFFECTIVE PSF curve (psf_r, psf_vals), in kpc, for the
    stacked-profile fit -- the DECIDED averaged-effective-PSF approach.

    Each galaxy's seeing is a fixed ANGULAR FWHM but a different PHYSICAL
    (kpc) FWHM via its own z (stored as result['psf_fwhm_kpc'] by
    extract_uv_profiles_for_field). The stacked profile -- combining
    per-galaxy profiles on shared kpc bins -- therefore sees not one Moffat
    but the galaxy-AVERAGE of their individual kpc Moffats. This returns that
    average as an explicit (psf_r, psf_vals) curve to hand straight to
    fit_and_plot_uv_coadd(..., fit_method='psf', psf_r=?, psf_vals=?),
    bypassing the single-scalar psf_fwhm path entirely.

    Each per-galaxy Moffat is FLUX-normalized (normalize_psf_flux, unit
    2*pi*r integral) BEFORE averaging, so every galaxy contributes equal
    total flux regardless of its kpc width -- a narrow and a broad PSF weigh
    the same, which is what "average PSF of the sample" should mean. The
    result is itself a unit-flux PSF (ring_convolution_matrix re-normalizes
    anyway, so absolute scale is irrelevant; the SHAPE is the point).

    This is the first-order effective PSF of a LINEAR stack. The actual
    galaxy combine is biweight (mildly non-linear), so treat this as
    correct-to-first-order, not exact. Each galaxy's Moffat uses its OWN
    field's (kpc FWHM, beta) -- from result['psf_fwhm_kpc'] and
    result['psf_beta'], set by extraction from uvcfg.psf_moffat_params (the
    star-fitted per-field values) or falling back to the header FWHM +
    uvcfg.psf_beta -- so different-seeing / different-wing fields average
    correctly.

    r_max_kpc : outer radius of the PSF grid (kpc). Default 20 * (max
        per-galaxy kpc FWHM), covering even the broadest galaxy's wings
        (matches plot_uv_fit's own 20*fwhm convention).

    Returns (psf_r, psf_vals) -- both length n_grid, psf_r in kpc.
    """
    usable = [r for r in results if np.isfinite(r.get(psf_key, np.nan))]
    if len(usable) == 0:
        raise ValueError(
            f"No finite '{psf_key}' in results -- run extract_uv_profiles_for_field "
            f"with a seeing value first (see read_seeing_fwhm_arcsec).")
    fwhm_kpc = np.array([float(r[psf_key]) for r in usable], dtype=float)

    if r_max_kpc is None:
        r_max_kpc = 20.0 * float(np.max(fwhm_kpc))
    psf_r = np.linspace(0.0, float(r_max_kpc), n_grid)

    # Per galaxy: if its FIELD has a measured empirical PSF curve
    # (uvcfg.psf_empirical[field]), interpolate THAT curve -- the honest
    # stacked-star profile -- instead of an analytic Moffat. The curve is in
    # arcsec, so it is scaled to this galaxy's kpc by its own arcsec->kpc
    # factor (psf_fwhm_kpc / psf_fwhm_arcsec, i.e. its z) before interpolating
    # onto the shared kpc grid; beyond the measured outer radius the PSF is
    # taken as 0. Fields WITHOUT an empirical entry fall back to their own
    # (kpc FWHM, beta) Moffat, so a mixed sample averages correctly. Every
    # galaxy's curve is flux-normalized (unit 2*pi*r integral) before
    # averaging, so narrow and broad PSFs weigh equally.
    acc = np.zeros_like(psf_r)
    betas, n_emp, n_moffat = [], 0, 0
    for r in usable:
        emp = uvcfg.psf_empirical.get(r.get("field")) if uvcfg.psf_empirical else None
        if emp is not None:
            fwhm_arcsec = float(r.get("psf_fwhm_arcsec", np.nan))
            kpc_per_arcsec = float(r["psf_fwhm_kpc"]) / fwhm_arcsec
            r_emp = np.asarray(emp["r_arcsec"], dtype=float) * kpc_per_arcsec
            v_emp = np.asarray(emp["value"], dtype=float)
            prof = np.interp(psf_r, r_emp, v_emp, left=float(v_emp[0]), right=0.0)
            acc += normalize_psf_flux(psf_r, prof)
            n_emp += 1
        else:
            b = float(r.get("psf_beta", uvcfg.psf_beta))
            acc += normalize_psf_flux(
                psf_r, moffat_1d(psf_r, fwhm=float(r[psf_key]), beta=b))
            betas.append(b)
            n_moffat += 1
    psf_vals = acc / len(usable)

    if verbose:
        ub = np.unique(np.round(np.array(betas), 3)) if betas else np.array([])
        emp_fields = sorted({r.get("field") for r in usable
                             if uvcfg.psf_empirical and r.get("field") in uvcfg.psf_empirical})
        print(f"build_effective_psf_uv: averaged {len(usable)} flux-normalized "
              f"per-galaxy PSFs -- {n_emp} empirical (fields {emp_fields}), "
              f"{n_moffat} Moffat (beta value(s) {ub}); kpc FWHM "
              f"min/median/max = {fwhm_kpc.min():.2f}/{np.median(fwhm_kpc):.2f}/"
              f"{fwhm_kpc.max():.2f}; grid 0..{r_max_kpc:.1f} kpc, {n_grid} pts.")
    return psf_r, psf_vals


def apply_psf_to_results(results: list, uvcfg: UVExtractConfig, *, verbose: bool = True):
    """
    (Re)assign each galaxy's PSF metadata -- psf_fwhm_arcsec, psf_fwhm_kpc,
    psf_beta -- from uvcfg.psf_moffat_params WITHOUT re-running extraction.

    The PSF never enters the measured fluxes (it's used only later by
    build_effective_psf_uv and the fit), so once you've fit the stellar PSF
    (fit_moffat_psf) and registered the numbers in uvcfg.psf_moffat_params,
    call this to push them onto an EXISTING results list and go straight to
    build_effective_psf_uv -- no cutouts/photometry recomputed. This is the
    fast iterate-on-beta loop: change psf_moffat_params, re-apply, refit.

    Per galaxy: if its field is in uvcfg.psf_moffat_params, adopt that field's
    fitted (fwhm_arcsec, beta); otherwise keep the field's already-stored
    header psf_fwhm_arcsec with uvcfg.psf_beta. psf_fwhm_kpc is recomputed
    from the (possibly updated) arcsec FWHM and the galaxy's z. Requires each
    result to carry 'field' and 'z' (extraction stores both) -- raises if
    'field' is absent (results predate the field tag: re-extract once).

    Mutates results in place AND returns it.
    """
    n_fit, n_hdr = 0, 0
    for r in results:
        if "field" not in r:
            raise KeyError("apply_psf_to_results: a result has no 'field' -- re-extract "
                           "once with the current extract_uv_profiles_for_field to add it.")
        fld, z = r["field"], float(r["z"])
        mp = uvcfg.psf_moffat_params.get(fld) if uvcfg.psf_moffat_params else None
        if mp is not None:
            fw = float(mp["fwhm_arcsec"])
            be = float(mp.get("beta", uvcfg.psf_beta))
            n_fit += 1
        else:
            fw = float(r.get("psf_fwhm_arcsec", np.nan))
            be = float(uvcfg.psf_beta)
            n_hdr += 1
        r["psf_fwhm_arcsec"] = fw
        r["psf_beta"] = be
        r["psf_fwhm_kpc"] = float(psf_fwhm_kpc_for_z(fw, z))
    if verbose:
        print(f"apply_psf_to_results: {n_fit} galaxies set from star-fitted "
              f"psf_moffat_params, {n_hdr} kept header/global fallback.")
    return results


def stack_and_bootstrap_uv(results: list, bins_kpc, *, method: str = "biweight",
                           nboot: int = 500, ci=(16, 84), seed: Optional[int] = None,
                           flux_col: str = "flux_mean", weight_col: str = "npix",
                           verbose: bool = True):
    """
    Coadd + bootstrap ONCE, DECOUPLED from fitting -- run this a single time
    (e.g. nboot=5000) and then fit the SAME (coadd, boot) pair as many times
    as you like (naive/psf, exp/sersic, different psf_fwhm, ...) via
    fit_and_plot_uv_coadd, without ever paying for the bootstrap again.

    The bootstrap resamples the GALAXY axis -- it's a property of the DATA
    (this sample, this method, these bins_kpc), NOT of any particular fit
    model. So it only needs computing once; every downstream fit is a cheap
    curve_fit against the already-bootstrapped error band. This is the whole
    runtime win over bootstrap_and_fit_uv, which recomputes coadd+bootstrap
    on every call: the 5000-draw resample is by far the expensive step.

    Typical use:
        coadd, boot = stack_and_bootstrap_uv(results_all, uvcfg.bins_kpc,
                                             method="biweight", nboot=5000)
        # now fit as many ways as you want, reusing the SAME boot:
        fit_and_plot_uv_coadd(coadd, boot=boot, method="biweight",
                              model="exp", fit_method="naive")
        fit_and_plot_uv_coadd(coadd, boot=boot, method="biweight",
                              model="exp", fit_method="psf", psf_fwhm=...)

    Returns (coadd, boot):
      coadd : coadd_uv_profiles(...) output containing `method` (pass as
              fit_and_plot_uv_coadd's first positional arg).
      boot  : bootstrap_uv_coadd(...) output for `method` (pass as
              fit_and_plot_uv_coadd's boot= arg).
    See coadd_uv_profiles / bootstrap_uv_coadd for every argument's meaning.
    """
    coadd = coadd_uv_profiles(results, bins_kpc, methods=(method,),
                              flux_col=flux_col, weight_col=weight_col, verbose=verbose)
    boot = bootstrap_uv_coadd(results, bins_kpc, method=method, nboot=nboot, ci=ci,
                              seed=seed, flux_col=flux_col, weight_col=weight_col,
                              verbose=verbose)
    return coadd, boot


def bootstrap_and_fit_uv(results: list, bins_kpc, *, method: str = "biweight",
                         nboot: int = 500, ci=(16, 84), seed: Optional[int] = None,
                         model: str = "exp", fit_method: str = "naive", n_fixed=None,
                         psf_r=None, psf_vals=None, psf_fwhm: float = 1.0,
                         psf_beta: float = 3.0, fit_skip_inner: int = 1,
                         logy: bool = True, logx: bool = False, xlims=None,
                         figsize=(9, 5), title=None, verbose: bool = True):
    """
    One-call coadd -> bootstrap -> fit -> plot for the UV-continuum sample.

    CONVENIENCE WRAPPER, kept for backward compatibility and one-shot use.
    It is now a thin call to stack_and_bootstrap_uv (once) + fit_and_plot_uv_
    coadd (once). If you plan to run MORE THAN ONE fit off the same sample
    (e.g. naive AND psf, or several psf_fwhm values), do NOT call this
    repeatedly -- it re-bootstraps every time. Instead call
    stack_and_bootstrap_uv ONCE (with your big nboot) and then
    fit_and_plot_uv_coadd for each fit, reusing the one (coadd, boot).

    Returns (fig, ax, fit_result, coadd, boot) -- the fit/plot outputs plus
    both intermediate dicts, so you can inspect n_gal-per-bin, the raw
    bootstrap draws, etc. without recomputing anything.
    """
    coadd, boot = stack_and_bootstrap_uv(results, bins_kpc, method=method, nboot=nboot,
                                         ci=ci, seed=seed, verbose=verbose)
    fig, ax, fit_result = fit_and_plot_uv_coadd(
        coadd, method=method, boot=boot, model=model, fit_method=fit_method,
        n_fixed=n_fixed, psf_r=psf_r, psf_vals=psf_vals, psf_fwhm=psf_fwhm,
        psf_beta=psf_beta, fit_skip_inner=fit_skip_inner, logy=logy, logx=logx,
        xlims=xlims, figsize=figsize, title=title, verbose=verbose,
    )
    return fig, ax, fit_result, coadd, boot
