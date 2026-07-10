"""
masking.py -- fiber masking (Stage 1.1).

Turns fiber sky positions into a (nexp, nfib) boolean bad-fiber mask (True =
bad). extract.build_bad_fiber_mask dispatches on config.mask_method, which may
be a single method ('segmap') or a '+'-joined composite ('segmap+spec_local')
that OR-combines each component's mask (bad in ANY component -> bad overall).
All four component functions end in the same shared world->pixel lookup where
applicable.

  METHOD                       function                       config.mask_method
  ---------------------------  -----------------------------  ------------------
  baked segmentation (default) segmentation_mask              'segmap'
  per-fiber continuum level     make_spectral_continuum_mask   'spec_global' /
                                                               'spec_local'
  continuum image (legacy A/B)  continuum_image_mask           'image'

  shared lookup used by segmap + image:  flag_fibers_on_mask
  spec_* only, applied post-threshold:   apply_protect_radius (config.mask_protect_radius_arcsec)

Default is 'segmap+spec_global': segmap catches faint real sources image-plane
percentile masks miss; spec_global catches instrumental bad fibers no
image-plane mask can see (they aren't on real sources) -- disjoint failure
modes, so OR-combining both is the default rather than a tuning choice.

The 'image' method and its three private-by-convention helpers
(make_circular_structure, get_pixscale_arcsec, nan_safe_gaussian_smooth) are
kept ONLY as the A/B baseline for validating the baked segmap mask; nothing in
the current pipeline path needs them. Self-mask rescues that the old code did at
runtime are now a bake-time edit to the segmap FITS. See core.compare_mask_spectra
and plotting.plot_fiber_mask_appendix for the checking/validation tools.
"""
from __future__ import annotations

import gc

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import binary_dilation, gaussian_filter
from tqdm import tqdm


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

