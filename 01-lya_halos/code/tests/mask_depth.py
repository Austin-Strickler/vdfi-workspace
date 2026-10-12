"""
mask_depth.py -- what magnitude does the baked continuum mask reach?

TEST BENCH. Lives in code/tests/, outside utils_lya_halo/, so
guide.check_guide() never sees it. Run from the code/ directory (the same cwd
the notebooks use), so the relative 'catalogs/...' paths resolve.

----------------------------------------------------------------------
THE QUESTION
----------------------------------------------------------------------
The baked masks (catalogs/images/fibermask_{EGS,COSMOS}.fits) were made
semi-shallowly on purpose, and MOSDEF targets were un-masked by hand. So there
is no detection threshold to quote. Instead we MEASURE one:

    take an independent, deeper list of real galaxies (CFHTLS T0007 Deep,
    VizieR II/317/cfhtls_d -- the same survey/band the masks were made on),
    ask for each one "is the mask pixel at its position set?",
    and plot the masked fraction vs g magnitude.

Bright galaxies -> fraction ~1, faint ones -> ~0. We quote the RAW fraction
f(m) = N_masked / N and read m90 / m50 where it falls through 0.9 / 0.5.

Why raw and not "corrected for chance masking": a faint galaxy could in
principle land on a bright neighbour's mask by accident. If galaxies were
placed at random, that rate would equal the masked sky fraction (~12% in
COSMOS). The data say otherwise -- at g > 26.5 only ~0.1% are masked, because
CFHTLS blends faint sources into bright ones and flags star halos, so faint
catalog sources are missing from exactly the masked areas. The accidental
floor is therefore measured directly from the faintest bins (empirical_floor)
and is ~0, so no correction is applied. The masked sky fraction is still
reported, as its own result: the share of the field (and fibers) given up.

----------------------------------------------------------------------
TWO PARTS
----------------------------------------------------------------------
PART 1 -- size / depth requirements (does the reference catalog qualify?)
    1a. the mask is a clean crop of the image (integer pixel shift, same scale);
        the mask grid is the reference frame, the image only defines "valid data"
    1b. footprint: centre, pixel scale, area, the cone radius we must query
    1c. the catalog COVERS the footprint (source density per ~1' cell)
    1d. the catalog is DEEP enough: its g number counts must keep rising well
        past the expected mask limit (~24-25); the turnover sits near where
        the catalog itself becomes incomplete

PART 2 -- the counterpart analysis
    2a. clean the catalog (valid g, drop g-band masked regions)
    2b. drop the KEEP-catalog targets (their mask state was edited by hand)
    2c. mask lookup at each source position (same rint lookup as the pipeline's
        extract.flag_fibers_on_mask, which is reused directly)
    2d. masked fraction per mag bin, floor, m50 / m90, figure

PART 3 -- do the TARGETS mask themselves? (e.g. the 3D-HST [O III] sample)
    Only MOSDEF segments were switched off by hand, so any other target
    brighter than the mask limit sits on its OWN segment and loses its
    central fibers. For every target: is the centre pixel masked, and what
    fraction of pixels (= fibers) is masked in radial annuli, stacked and
    compared with the masked sky fraction. Plus postage stamps.

        out = md.run_targets(products)            # the notebook's {'AEGIS':..,'COSMOS':..}
        r = out["COSMOS"]                          # per field: fd, targets, result, figs
        r["result"]["profile"]                     # masked fraction + real NFIB per bin
        md.plot_target_stamps(r["fd"], r["targets"], r["result"], only="masked")
    Nothing is written to disk.

----------------------------------------------------------------------
USAGE
----------------------------------------------------------------------
    import sys; sys.path.insert(0, "tests")
    import mask_depth as md

    res = md.run("COSMOS")                  # everything, prints + figure
    res = md.run("AEGIS", mag_col="gmag")   # 3" aperture mag instead of AUTO

or step by step:

    fd   = md.load_field("COSMOS")
    geo  = md.check_geometry(fd)                         # PART 1a/1b
    cat  = md.fetch_cfhtls_deep(geo["center"], geo["radius_deg"], "COSMOS")
    md.describe_catalog(cat)                             # schema + flags
    md.check_coverage(cat, fd)                           # PART 1c
    md.check_depth(cat)                                  # PART 1d
    cl   = md.clean_catalog(cat)                         # PART 2a
    cl   = md.drop_targets(cl, "COSMOS")                 # PART 2b
    look = md.mask_lookup(cl, fd)                        # PART 2c
    cur  = md.masked_fraction_curve(look["mag"], look["masked"], geo["f_rand"])
    lim  = md.mag_limits(cur)
    md.plot_curve(cur, lim, geo["f_rand"], "COSMOS")

The catalog download needs network (astroquery -> VizieR). It is cached to
catalogs/cfhtls/cfhtls_d_<FIELD>.fits, so only the first run per field is
online.

----------------------------------------------------------------------
REFERENCE CATALOG: CFHTLS T0007 Deep (Hudelot+2012), VizieR II/317/cfhtls_d
----------------------------------------------------------------------
    D2 = COSMOS, D3 = EGS. ~2.3 M sources over the 4 Deep fields.
    Detection on a g+r+i chi2 image; 80% complete to g ~ 26.0 (point sources).
    AB mags, ZP 30. Columns used (labels per the VizieR ReadMe):
        RA, DEC          J2000 position (copied from VizieR's computed _RAJ2000/_DEJ2000)
        gmagA, e_gmagA   g Kron-like (MAG_AUTO) magnitude  <- default
        gmag,  e_gmag    g 3" aperture magnitude
        gcl              g-band CLASS_STAR
        Sfl              SExtractor FLAGS
        Tfl              TERAPIX FLAG_MASK: 1=star, 2=saturated star,
                         4/8/16/32/64/128 = masked region in z/y/i/r/g/u
    The ReadMe was read through a summariser, so describe_catalog() prints the
    real schema and Tfl bit counts -- confirm them on the first run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# pipeline helpers: reuse the exact fiber lookup and the house tick style
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
from utils_lya_halo.extract import flag_fibers_on_mask          # noqa: E402

try:
    from utils_lya_halo.formatting import apply_tick_style      # noqa: E402
except ImportError:                                              # pragma: no cover
    apply_tick_style = None


# =====================================================================
# CONFIG
# =====================================================================
FIELD_ALIASES = {"EGS": "AEGIS", "AEGIS": "AEGIS", "COS": "COSMOS", "COSMOS": "COSMOS"}

IMAGE_PATHS = {                      # same map as config.resolve_cont_image_path
    "AEGIS":  "catalogs/images/groth.fits",
    "COSMOS": "catalogs/images/cosmos.fits",
}
MASK_PATHS = {                       # same map as config.resolve_seg_mask_path
    "AEGIS":  "catalogs/images/fibermask_EGS.fits",
    "COSMOS": "catalogs/images/fibermask_COSMOS.fits",
}
TARGET_CATALOG = "catalogs/lya_halo_catalog_cuts/lya_mosdef_catalog_KEEP.fits"
CACHE_DIR = "catalogs/cfhtls"

VIZIER_TABLE = "II/317/cfhtls_d"

TFL_BITS = {1: "star", 2: "saturated star", 4: "masked z", 8: "masked y",
            16: "masked i", 32: "masked r", 64: "masked g", 128: "masked u"}

EXPECTED_LIMIT = 25.0     # your prior on the mask limit; depth check compares to it


def _field(field):
    try:
        return FIELD_ALIASES[str(field).upper()]
    except KeyError:
        raise ValueError(f"field must be AEGIS/EGS or COSMOS, got {field!r}")


def _pixscale_arcsec(wcs):
    """Mean pixel scale in arcsec (version-proof: the utils function returns deg floats)."""
    from astropy.wcs.utils import proj_plane_pixel_scales
    return float(np.mean(np.abs(proj_plane_pixel_scales(wcs)))) * 3600.0


def _image_hdu(hdul):
    """First HDU carrying 2-D image data (same rule as extract.segmentation_mask)."""
    return next((h for h in hdul if getattr(h, "data", None) is not None
                 and np.ndim(h.data) == 2), hdul[0])


# =====================================================================
# PART 1 -- size / depth requirements
# =====================================================================
def _image_on_mask_grid(img_data, wcs_img, wcs_mask, shape, tol=0.05):
    """
    Resample the (large) image onto the (cropped) mask's pixel grid.

    The mask is a crop of the same CFHTLS stack, so the two grids should differ
    only by an integer pixel SHIFT. Test that at the four mask corners: map
    mask pixel -> sky -> image pixel and require every corner to give the same
    offset, and that offset to be an integer (within tol pixels). If so, the
    resample is just an array slice (fast, exact, memmap-friendly). If not
    (rotated / rescaled grid), fall back to nearest-pixel lookup row block by
    row block. Mask pixels that fall off the image come back as 0 (= no data).

    Returns (image_on_mask_grid, info_dict).
    """
    ny, nx = shape
    px = np.array([0, nx - 1, 0, nx - 1], float)
    py = np.array([0, 0, ny - 1, ny - 1], float)
    ix, iy = wcs_img.world_to_pixel(wcs_mask.pixel_to_world(px, py))
    dx, dy = ix - px, iy - py
    spread = max(np.ptp(dx), np.ptp(dy))
    ox, oy = int(np.rint(dx[0])), int(np.rint(dy[0]))
    frac = max(abs(dx[0] - ox), abs(dy[0] - oy))
    pure_shift = spread < tol and frac < tol

    NY, NX = img_data.shape
    out = np.zeros(shape, np.float32)
    if pure_shift:
        # overlap of mask grid [0,ny)x[0,nx) shifted by (oy,ox) with image [0,NY)x[0,NX)
        y0, y1 = max(0, -oy), min(ny, NY - oy)
        x0, x1 = max(0, -ox), min(nx, NX - ox)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = img_data[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
    else:
        xs = np.arange(nx, dtype=float)
        for r0 in range(0, ny, 256):
            rows = np.arange(r0, min(r0 + 256, ny), dtype=float)
            gx, gy = np.meshgrid(xs, rows)
            jx, jy = wcs_img.world_to_pixel(wcs_mask.pixel_to_world(gx, gy))
            jx, jy = np.rint(jx).astype(int), np.rint(jy).astype(int)
            ok = (jx >= 0) & (jx < NX) & (jy >= 0) & (jy < NY)
            blk = np.zeros(gx.shape, np.float32)
            blk[ok] = img_data[jy[ok], jx[ok]]
            out[r0:r0 + len(rows)] = blk
    info = dict(pure_shift=pure_shift, offset_xy=(ox, oy),
                corner_spread_pix=float(spread), offset_frac_pix=float(frac))
    return out, info


def load_field(field, image_path=None, mask_path=None):
    """
    Load the baked mask and the continuum image for one field, with the image
    resampled onto the MASK's grid. The mask is the reference frame for
    everything (it is what the pipeline uses); the image only supplies the
    'valid data' footprint. The image may be larger than the mask (e.g. the
    full 1 deg^2 CFHTLS stack vs a cropped mask) -- that is fine.
    """
    field = _field(field)
    image_path = image_path or IMAGE_PATHS[field]
    mask_path = mask_path or MASK_PATHS[field]

    with fits.open(mask_path) as hdul:
        h = _image_hdu(hdul)
        mask = np.asarray(h.data).astype(bool)                   # nonzero -> masked
        wcs_mask = WCS(h.header).celestial
    with fits.open(image_path, memmap=True) as hdul:
        h = _image_hdu(hdul)
        wcs_img = WCS(h.header).celestial
        img_shape = h.data.shape
        image, grid = _image_on_mask_grid(h.data, wcs_img, wcs_mask, mask.shape)

    # "valid" = mask pixels with real image data under them. Off-image and
    # zero/NaN padding must not count toward the random floor.
    valid = np.isfinite(image) & (image != 0)

    return dict(field=field, image=image, mask=mask, valid=valid,
                wcs_img=wcs_img, wcs_mask=wcs_mask, img_shape=img_shape,
                grid=grid, image_path=image_path, mask_path=mask_path)


def check_geometry(fd, pad=1.05):
    """
    PART 1a/1b. Is the mask a clean crop of the image? Footprint, query cone,
    random floor.

    1a passes when the mask grid is an integer-pixel shift of the image grid
    (same scale, no rotation), which is what cropping produces, and the mask
    lies on image data.
    """
    mask, valid, wm, grid = fd["mask"], fd["valid"], fd["wcs_mask"], fd["grid"]
    ny, nx = mask.shape

    print(f"== PART 1a/1b geometry: {fd['field']}")
    print(f"  image  {fd['image_path']}  shape {fd['img_shape']}")
    print(f"  mask   {fd['mask_path']}  shape {mask.shape}")
    s_img, s_mask = _pixscale_arcsec(fd["wcs_img"]), _pixscale_arcsec(wm)
    print(f"  pixel scale image / mask:   {s_img:.4f} / {s_mask:.4f} arcsec/pix")
    print(f"  mask = crop of image?       {grid['pure_shift']}  "
          f"(offset x,y = {grid['offset_xy']} pix; corner spread "
          f"{grid['corner_spread_pix']:.3f}, non-integer part {grid['offset_frac_pix']:.3f})")
    if not grid["pure_shift"]:
        print("    -> grids differ by more than a shift; used nearest-pixel resampling")
    f_on = float(valid.mean())
    print(f"  mask pixels with image data: {f_on:.1%}"
          f"{'' if f_on > 0.95 else '  <- mask extends off the image / onto padding'}")

    px = np.array([0, nx - 1, 0, nx - 1, (nx - 1) / 2])
    py = np.array([0, 0, ny - 1, ny - 1, (ny - 1) / 2])
    corners = wm.pixel_to_world(px, py)
    center = corners[4]
    radius = float(np.max(center.separation(corners[:4]).deg)) * pad
    area_all = nx * ny * (s_mask / 60.0) ** 2
    area_valid = valid.sum() * (s_mask / 60.0) ** 2
    print(f"  mask centre (RA, Dec):      {center.ra.deg:.5f}, {center.dec.deg:+.5f}")
    print(f"  mask area (full / on data): {area_all:.1f} / {area_valid:.1f} arcmin^2")
    print(f"  query cone radius:          {radius * 60:.2f} arcmin  (corners x {pad})")

    f_rand = float(mask[valid].mean()) if valid.any() else np.nan
    print(f"  masked sky fraction (share of valid pixels on the mask): {f_rand:.4f}")
    if (mask & ~valid).any():
        print(f"  note: {int((mask & ~valid).sum())} masked pixels sit OFF valid data")

    return dict(crop_ok=grid["pure_shift"], frac_on_data=f_on,
                pixscale=s_mask, center=center, radius_deg=radius,
                area_valid_arcmin2=area_valid, f_rand=f_rand)


def _col(cat, name):
    """Column as a float array; masked/blank entries -> NaN.
    Cast to float BEFORE filling: an integer column (e.g. the int16 flags)
    cannot hold NaN, so filling it first raises."""
    c = cat[name]
    if hasattr(c, "mask"):
        return np.ma.asarray(c).astype(float).filled(np.nan)
    return np.asarray(c, dtype=float)


def validate_catalog(cat, center, radius_deg, mag_col="gmagA"):
    """
    Physical sanity checks that catch a mislabeled table. Column NAMES are only
    metadata -- these test the VALUES:
      * RA/DEC must land inside the query cone (they are the sky positions
        VizieR itself computed, so this also checks the cone was on target)
      * the g magnitudes must look like faint-galaxy magnitudes (median 20-30)
    Raises on failure, so a bad download is never cached.
    """
    sc = SkyCoord(_col(cat, "RA") * u.deg, _col(cat, "DEC") * u.deg)
    sep = sc.separation(center).deg
    worst = float(np.nanmax(sep))
    m = _col(cat, mag_col)
    med = float(np.nanmedian(m[(m > 0) & (m < 50)])) if np.any((m > 0) & (m < 50)) else np.nan
    print(f"  validate: max distance from cone centre {worst * 60:.2f}' "
          f"(cone {radius_deg * 60:.2f}');  median {mag_col} = {med:.2f}")
    if worst > radius_deg * 1.01:
        raise RuntimeError("RA/DEC fall outside the query cone -- positions mislabeled")
    if not (20 < med < 30):
        raise RuntimeError(f"{mag_col} median {med:.2f} is not a g magnitude -- "
                           f"columns mislabeled; inspect cat.colnames")


def fetch_cfhtls_deep(center, radius_deg, field, cache_dir=CACHE_DIR,
                      refresh=False, mag_col="gmagA"):
    """
    Cone query of II/317/cfhtls_d, cached to <cache_dir>/cfhtls_d_<FIELD>.fits.
    Needs astroquery + network on first call only.

    We ask VizieR for ALL columns ("**") rather than a hand-picked list: the
    first attempt requested names that do not exist in the served table
    (RAdeg/DEdeg), and the returned VOTable came back with names attached to
    the wrong data (RA values sitting under 'gmag', etc.). "_RAJ2000" and
    "_DEJ2000" are positions VizieR computes itself for every catalog, so they
    always exist; we copy them to plain RA/DEC columns.

    The cache records the query centre/radius; a cache made for a different
    cone is refused (re-query with refresh=True).
    """
    field = _field(field)
    path = os.path.join(cache_dir, f"cfhtls_d_{field}.fits")
    if os.path.exists(path) and not refresh:
        cat = Table.read(path)
        q = SkyCoord(cat.meta.get("QRA", np.nan) * u.deg, cat.meta.get("QDEC", np.nan) * u.deg)
        same = (np.isfinite(q.ra.deg) and q.separation(center).arcsec < 1.0
                and abs(cat.meta.get("QRAD", -1) - radius_deg) * 3600 < 1.0)
        if not same:
            raise RuntimeError(f"cache {path} was made for a different cone (or by the old "
                               f"code) -- call again with refresh=True")
        print(f"== catalog: read cache {path}  ({len(cat)} rows)")
        return cat

    from astroquery.vizier import Vizier
    viz = Vizier(columns=["**", "_RAJ2000", "_DEJ2000"], row_limit=-1, timeout=600)
    print(f"== catalog: querying {VIZIER_TABLE} around "
          f"({center.ra.deg:.4f}, {center.dec.deg:+.4f}), r = {radius_deg * 60:.1f}' ...")
    res = viz.query_region(center, radius=radius_deg * u.deg, catalog=VIZIER_TABLE)
    if len(res) == 0:
        raise RuntimeError("VizieR returned nothing -- check the cone is on D2/D3")
    cat = res[0]
    print(f"  {len(cat)} rows, {len(cat.colnames)} columns: {cat.colnames}")
    cat["RA"] = _col(cat, "_RAJ2000")
    cat["DEC"] = _col(cat, "_DEJ2000")
    validate_catalog(cat, center, radius_deg, mag_col=mag_col)

    # replace VizieR's long free-text meta (which triggers FITS HIERARCH
    # warnings) with just the query parameters the cache check needs
    cat.meta = dict(QRA=float(center.ra.deg), QDEC=float(center.dec.deg),
                    QRAD=float(radius_deg), SOURCE=VIZIER_TABLE)
    os.makedirs(cache_dir, exist_ok=True)
    cat.write(path, overwrite=True)
    print(f"  -> cached {path}")
    return cat


def _mag(cat, name):
    m = _col(cat, name)
    m[(m <= 0) | (m > 50)] = np.nan          # 99 / -99 non-detection sentinels
    return m


def describe_catalog(cat):
    """Print the schema, value ranges, and flag bit counts. Run this first."""
    print(f"== catalog schema: {len(cat)} rows")
    for name in cat.colnames:
        col = cat[name]
        unit = f" [{col.unit}]" if getattr(col, "unit", None) else ""
        try:
            a = _col(cat, name)
            fin = np.isfinite(a)
            rng = (f"min {np.min(a[fin]):.3f}  median {np.median(a[fin]):.3f}  "
                   f"max {np.max(a[fin]):.3f}  finite {fin.mean():.1%}") if fin.any() else "no finite"
        except (TypeError, ValueError):
            rng = f"non-numeric, e.g. {col[0]!r}"
        print(f"  {name:10s}{unit:12s} {col.dtype!s:8s} {rng}")

    for mc in ("gmagA", "gmag"):
        if mc in cat.colnames:
            m = _col(cat, mc)
            print(f"  {mc}: sentinel/invalid (<=0 or >50 or blank): "
                  f"{np.mean(~np.isfinite(_mag(cat, mc))):.1%}   raw values >50: "
                  f"{np.sum(m > 50)}")

    if "Tfl" in cat.colnames:
        t = np.nan_to_num(_col(cat, "Tfl"), nan=0).astype(int)
        print("  Tfl bits (confirm against the ReadMe meanings):")
        for b, meaning in TFL_BITS.items():
            print(f"    {b:4d} {meaning:15s} {np.sum((t & b) > 0):8d}")
    if "Sfl" in cat.colnames:
        s = np.nan_to_num(_col(cat, "Sfl"), nan=0).astype(int)
        print(f"  Sfl: 0 -> {np.sum(s == 0)}, blended(1|2) -> {np.sum((s & 3) > 0)}, "
              f"saturated(4) -> {np.sum((s & 4) > 0)}")


def check_coverage(cat, fd, cell_arcsec=60.0, min_per_cell=1):
    """
    PART 1c. Does the catalog cover the footprint? Grid the image into
    ~cell_arcsec cells, keep cells that are mostly valid data, and count
    catalog sources per cell. Empty cells => the catalog stops short there.
    """
    ra, dec = _col(cat, "RA"), _col(cat, "DEC")
    ok = np.isfinite(ra) & np.isfinite(dec)
    x, y = fd["wcs_mask"].world_to_pixel(SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg))
    ny, nx = fd["mask"].shape
    scale = _pixscale_arcsec(fd["wcs_mask"])
    step = max(int(round(cell_arcsec / scale)), 1)
    ex, ey = np.arange(0, nx + step, step), np.arange(0, ny + step, step)

    counts, _, _ = np.histogram2d(y, x, bins=[ey, ex])
    vfrac = np.zeros_like(counts)
    for i in range(len(ey) - 1):
        for j in range(len(ex) - 1):
            v = fd["valid"][ey[i]:ey[i + 1], ex[j]:ex[j + 1]]
            vfrac[i, j] = v.mean() if v.size else 0
    use = vfrac > 0.9
    n = counts[use]
    empty = np.mean(n < min_per_cell) if use.any() else np.nan
    print(f"== PART 1c coverage: {use.sum()} cells of {cell_arcsec:.0f}\" on valid data")
    print(f"  sources/cell: median {np.median(n):.0f}, 5th pct {np.percentile(n, 5):.0f}, "
          f"min {n.min():.0f}")
    print(f"  cells with < {min_per_cell} source: {empty:.1%}  "
          f"({'OK' if empty == 0 else 'catalog does NOT cover the whole footprint'})")
    return dict(counts=counts, valid_frac=vfrac, used=use, frac_empty=empty)


def check_depth(cat, mag_col="gmagA", expected_limit=EXPECTED_LIMIT, dm=0.25):
    """
    PART 1d. Number counts. Real galaxy counts rise steadily with magnitude;
    they turn over only where the CATALOG goes incomplete. The turnover must
    sit clearly fainter than the mask limit we want to measure (want >= ~1 mag).
    """
    m = _mag(cat, mag_col)
    m = m[np.isfinite(m)]
    edges = np.arange(16, 30 + dm, dm)
    n, _ = np.histogram(m, edges)
    cen = 0.5 * (edges[1:] + edges[:-1])
    m_peak = float(cen[np.argmax(n)])
    margin = m_peak - expected_limit
    print(f"== PART 1d depth ({mag_col}): counts peak at {m_peak:.2f}; "
          f"expected mask limit ~{expected_limit:.1f} -> margin {margin:+.2f} mag "
          f"({'OK' if margin >= 1.0 else 'TOO SHALLOW to measure the limit cleanly'})")
    return dict(edges=edges, counts=n, m_peak=m_peak, margin=margin)


# =====================================================================
# PART 2 -- counterpart analysis
# =====================================================================
def clean_catalog(cat, mag_col="gmagA", drop_g_masked=True, drop_stars=False):
    """
    Keep sources with a valid g mag. By default drop Tfl bit 64 (source in a
    CFHTLS g-band masked region: bad photometry, usually star halos).
    Stars are KEPT by default -- the mask treats them like any other
    continuum source; set drop_stars=True for a galaxy-only curve.
    """
    m = _mag(cat, mag_col)
    keep = np.isfinite(m) & np.isfinite(_col(cat, "RA")) & np.isfinite(_col(cat, "DEC"))
    n0 = keep.sum()
    if "Tfl" in cat.colnames:
        t = np.nan_to_num(_col(cat, "Tfl"), nan=0).astype(int)
        if drop_g_masked:
            keep &= (t & 64) == 0
        if drop_stars:
            keep &= (t & 3) == 0
    out = cat[keep]
    out["MAG"] = m[keep]
    print(f"== PART 2a clean ({mag_col}): {len(cat)} -> valid mag {n0} -> kept {len(out)}"
          f"  (drop_g_masked={drop_g_masked}, drop_stars={drop_stars})")
    return out


def drop_targets(cat, field, target_path=TARGET_CATALOG, which="MOSDEF",
                 radius_arcsec=1.0, return_mags=False):
    """
    PART 2b. Remove sources matched to the KEEP catalog targets in this field.
    Their mask state was edited by hand, so they are not a fair test of the
    automatic masking. which=None drops every KEEP row in the field.
    return_mags=True also returns the matched targets' catalog magnitudes
    (for predict_target_fraction).
    """
    field = _field(field)
    t = Table.read(target_path)
    sel = np.array(t["FIELD"]).astype(str) == field
    if which is not None and "CATALOG" in t.colnames:
        sel &= np.array(t["CATALOG"]).astype(str) == which
    tg = SkyCoord(np.asarray(t["RA"][sel], float) * u.deg,
                  np.asarray(t["DEC"][sel], float) * u.deg)
    sc = SkyCoord(_col(cat, "RA") * u.deg, _col(cat, "DEC") * u.deg)
    idx, sep, _ = tg.match_to_catalog_sky(sc)
    hit = sep.arcsec < radius_arcsec
    drop = np.zeros(len(cat), bool)
    drop[idx[hit]] = True
    print(f"== PART 2b targets: {sel.sum()} {which or 'KEEP'} targets in {field}, "
          f"{hit.sum()} matched within {radius_arcsec}\" -> dropped {drop.sum()}")
    mt = np.asarray(cat["MAG"][idx[hit]]) if hit.any() else np.array([])
    if hit.any():
        print(f"  matched target mags: median {np.nanmedian(mt):.2f}, "
              f"range {np.nanmin(mt):.2f}-{np.nanmax(mt):.2f}")
    return (cat[~drop], mt) if return_mags else cat[~drop]


def mask_lookup(cat, fd):
    """
    PART 2c. Is the mask set at each source's position? Uses the pipeline's own
    flag_fibers_on_mask (nearest-pixel rint lookup), so "masked" here means
    exactly what it means for a fiber. Sources off the footprint or on
    invalid (no-data) pixels are removed.
    """
    ra, dec = _col(cat, "RA"), _col(cat, "DEC")
    masked = flag_fibers_on_mask(ra, dec, wcs_full=fd["wcs_mask"],
                                 mask_cutout=fd["mask"], xmin=0, ymin=0)
    x, y = fd["wcs_mask"].world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    ny, nx = fd["mask"].shape
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    on_data = np.zeros(len(ra), bool)
    on_data[inside] = fd["valid"][yi[inside], xi[inside]]
    print(f"== PART 2c lookup: {len(ra)} sources, {inside.sum()} inside footprint, "
          f"{on_data.sum()} on valid data, {masked[on_data].sum()} masked "
          f"({masked[on_data].mean():.1%})")
    return dict(mag=np.asarray(cat["MAG"])[on_data], masked=masked[on_data],
                cat=cat[on_data])


def _wilson(k, n, z=1.0):
    """Binomial interval (z=1 -> 16th/84th percentile equivalent)."""
    n = np.asarray(n, float)
    p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def masked_fraction_curve(mag, masked, f_rand=None, lo=17.0, hi=28.0, dm=0.5):
    """
    PART 2d. Raw masked fraction per bin, f = N_masked / N, with its 68%
    binomial interval. f_rand is accepted for backward compatibility and
    ignored (see the module docstring for why no floor correction is applied).
    """
    edges = np.arange(lo, hi + dm, dm)
    n, _ = np.histogram(mag, edges)
    k, _ = np.histogram(mag[masked], edges)
    f = np.divide(k, n, out=np.full(len(n), np.nan), where=n > 0)
    f_lo, f_hi = _wilson(k, n)
    cur = Table(dict(m_lo=edges[:-1], m_hi=edges[1:], m=0.5 * (edges[1:] + edges[:-1]),
                     N=n, N_masked=k, f=f, f_p16=f_lo, f_p84=f_hi))
    for c in ("f", "f_p16", "f_p84"):
        cur[c].format = ".3f"
    return cur


def empirical_floor(cur, m_min=26.5, min_n=100):
    """
    The accidental-masking rate MEASURED from the data: pooled masked fraction
    of all sources in bins centred at m >= m_min (too faint to be masked for
    their own brightness). Compare it to the masked sky fraction.
    """
    sel = (np.asarray(cur["m"]) >= m_min) & (np.asarray(cur["N"]) >= min_n)
    k, n = int(np.sum(cur["N_masked"][sel])), int(np.sum(cur["N"][sel]))
    fl = k / n if n else np.nan
    lo, hi = _wilson(np.array([k]), np.array([n]))
    print(f"== empirical floor (g >= {m_min}): {k}/{n} = {fl:.4f}  "
          f"[{lo[0]:.4f}, {hi[0]:.4f}]")
    return fl


def predict_target_fraction(cur, target_mags):
    """
    Cross-check: the curve's predicted masked fraction for the MOSDEF targets,
    = mean of f(m) over their magnitudes (linear interpolation between bin
    centres; beyond the curve's ends f is held at the end value). Compare with
    the fraction you actually had to un-mask by hand.
    """
    m = np.asarray(target_mags, float)
    m = m[np.isfinite(m)]
    ok = np.isfinite(np.asarray(cur["f"], float))
    p = np.interp(m, np.asarray(cur["m"])[ok], np.asarray(cur["f"], float)[ok])
    print(f"== MOSDEF cross-check: {len(m)} targets, predicted masked fraction "
          f"{p.mean():.2f} (expected count {p.sum():.0f})")
    return float(p.mean())


def mag_limits(cur, levels=(0.9, 0.5), min_n=20, column="f"):
    """
    Where the masked fraction (raw `f` by default) falls through each level.
    Scanned from the FAINT end toward bright: the limit is the faintest bin
    pair that straddles the level, with linear interpolation between bin
    centres, so a noisy dip at the sparse bright end cannot fake an early
    crossing. Bins with N < min_n are ignored.
    """
    ok = np.asarray(cur["N"]) >= min_n
    m, f = np.asarray(cur["m"])[ok], np.asarray(cur[column], float)[ok]
    out = {}
    for lev in levels:
        out[lev] = np.nan
        for i in range(len(m) - 1, 0, -1):
            if f[i - 1] >= lev > f[i]:
                out[lev] = float(m[i - 1] + (lev - f[i - 1]) * (m[i] - m[i - 1]) / (f[i] - f[i - 1]))
                break
    print("== limits (raw fraction): " + ", ".join(f"m{int(l * 100)} = {v:.2f}" for l, v in out.items()))
    return out


def plot_curve(cur, limits, f_rand, field, depth=None, mag_label="g (AUTO, AB)",
               floor=None, tick_direction="in", tick_right=True, savepath=None):
    """
    Top: raw masked fraction f(m) with 68% binomial errors, m90/m50 lines, the
    empirical floor (faint-bin masked fraction), and the masked sky fraction
    quoted in text (it is NOT a floor for this curve). Bottom: counts.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    ax = axes[0]
    ok = cur["N"] > 0
    m = cur["m"][ok]
    ax.errorbar(m, cur["f"][ok],
                yerr=np.clip([cur["f"][ok] - cur["f_p16"][ok],
                              cur["f_p84"][ok] - cur["f"][ok]], 0, None),
                fmt="o", ms=4, color="k", capsize=0, label="masked fraction")
    if floor is None:
        floor = empirical_floor(cur)
    if np.isfinite(floor):
        ax.axhline(floor, ls="--", color="0.5", label=f"empirical floor = {floor:.3f}")
    if f_rand is not None and np.isfinite(f_rand):
        ax.text(0.98, 0.60, f"masked sky fraction = {f_rand:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="0.3",
                bbox=dict(fc="white", ec="none", alpha=0.8))
    for lev, v in limits.items():
        if np.isfinite(v):
            ax.axvline(v, ls=":", color="C3" if lev == 0.5 else "C1",
                       label=f"m{int(lev * 100)} = {v:.2f}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("masked fraction  $N_{\\rm masked}/N$")
    ax.set_title(f"{field}: baked mask vs CFHTLS Deep")
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    ax = axes[1]
    ax.step(cur["m"], cur["N"], where="mid", color="k", label="all (on data)")
    ax.step(cur["m"], cur["N_masked"], where="mid", color="C3", label="masked")
    ax.set_yscale("log")
    ax.set_ylabel("N per bin")
    ax.set_xlabel(mag_label)
    ax.legend(frameon=False, fontsize=9)

    if apply_tick_style is not None:
        apply_tick_style(axes, direction=tick_direction, right=tick_right)
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, axes


# =====================================================================
# VISUAL CHECK -- image + catalog overlay
# =====================================================================
def plot_overlay(fd, cat, mag_col="gmagA", masked=None, region=None,
                 show="image", cmap="binary", stretch=(1.0, 99.5), stride=1,
                 mask_color="tab:orange", mask_alpha=0.35,
                 mask_contour=False, contour_color="tab:green", contour_lw=0.8,
                 mag_range=None, s=15, size_by_mag=False, m_ref=24.0,
                 marker="o", filled=False, lw=0.8, alpha=0.9,
                 color="tab:blue", color_masked="tab:red", color_unmasked="tab:blue",
                 figsize=(9, 9), ax=None, title=None,
                 tick_direction="in", tick_right=True):
    """
    Show the continuum image (on the mask grid) and mark catalog sources.

    WHAT YOU SEE
      show="image"  the CFHTLS cutout, grey scale ('binary' = white sky, dark sources)
      show="mask"   the baked mask itself (black = masked)
      show="both"   the image, with masked pixels tinted mask_color
      mask_contour=True  outline the mask edges on top of any of the above --
                    the cleanest way to see WHICH objects the mask covers

    CHOOSING THE AREA
      region=None                     the whole mask (use stride=4..8 for speed)
      region=(ra, dec, size_arcmin)   a square zoom; use this to judge
                                      individual sources (stride=1)

    MARKERS
      masked=None       every source drawn in `color`
      masked=bool array (e.g. look["masked"] with cat=look["cat"]):
                        red = sits on the mask, blue = not masked
      mag_range=(lo,hi) only draw sources in that magnitude range --
                        e.g. (24, 25) to eyeball what happens near the limit
      filled=False      open circles, so you can see the source underneath
      size_by_mag=True  brighter -> bigger (s at m_ref, x2.5 per 2 mag brighter)

    stretch = (lo, hi) percentiles of the displayed pixels for the grey scale.
    """
    import matplotlib.pyplot as plt

    wcs, img, mask = fd["wcs_mask"], fd["image"], fd["mask"]
    ny, nx = mask.shape
    if region is None:
        x0, x1, y0, y1 = 0, nx, 0, ny
    else:
        ra0, dec0, size = region
        cx, cy = wcs.world_to_pixel(SkyCoord(ra0 * u.deg, dec0 * u.deg))
        half = size * 60.0 / _pixscale_arcsec(wcs) / 2.0
        x0, x1 = int(max(0, cx - half)), int(min(nx, cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(ny, cy + half))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("region does not overlap the mask footprint")
    sub = wcs[y0:y1, x0:x1]                       # WCS of the cut-out, for RA/Dec axes
    k = max(int(stride), 1)
    im_s = img[y0:y1:k, x0:x1:k]
    mk_s = mask[y0:y1:k, x0:x1:k]
    ext = (-0.5, (x1 - x0) - 0.5, -0.5, (y1 - y0) - 0.5)   # pixel extent of `sub`

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=sub)
    else:
        fig = ax.figure

    if show in ("image", "both"):
        good = np.isfinite(im_s) & (im_s != 0)
        vmin, vmax = np.percentile(im_s[good], stretch) if good.any() else (0, 1)
        ax.imshow(im_s, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                  extent=ext, interpolation="nearest")
        if show == "both":
            tint = np.zeros(mk_s.shape + (4,))
            tint[mk_s] = plt.matplotlib.colors.to_rgba(mask_color, mask_alpha)
            ax.imshow(tint, origin="lower", extent=ext, interpolation="nearest")
    elif show == "mask":
        ax.imshow(mk_s, origin="lower", cmap="binary", vmin=0, vmax=1,
                  extent=ext, interpolation="nearest")
    else:
        raise ValueError("show must be 'image', 'mask' or 'both'")
    if mask_contour and mk_s.any():
        ax.contour(mk_s.astype(float), levels=[0.5], colors=contour_color,
                   linewidths=contour_lw, extent=ext, origin="lower")

    # sources -> pixel coords of the cut-out
    ra, dec = _col(cat, "RA"), _col(cat, "DEC")
    mag = _mag(cat, mag_col) if mag_col in cat.colnames else np.asarray(cat["MAG"], float)
    x, y = sub.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    sel = (x >= -0.5) & (x < x1 - x0 - 0.5) & (y >= -0.5) & (y < y1 - y0 - 0.5)
    if mag_range is not None:
        sel &= (mag >= mag_range[0]) & (mag < mag_range[1])
    size = np.full(len(ra), float(s))
    if size_by_mag:
        size = np.clip(s * 10 ** (-0.2 * (mag - m_ref)), s / 5, s * 8)

    def _draw(idx, c, label):
        if not idx.any():
            return
        kw = dict(s=size[idx], marker=marker, alpha=alpha, label=f"{label} ({idx.sum()})")
        if filled:
            kw.update(c=c, edgecolors="none")
        else:
            kw.update(facecolors="none", edgecolors=c, linewidths=lw)
        ax.scatter(x[idx], y[idx], **kw)

    if masked is None:
        _draw(sel, color, "catalog")
    else:
        masked = np.asarray(masked, bool)
        _draw(sel & masked, color_masked, "on mask")
        _draw(sel & ~masked, color_unmasked, "not masked")

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("RA"); ax.set_ylabel("Dec")
    rng = "" if mag_range is None else f", {mag_range[0]}<={mag_col}<{mag_range[1]}"
    ax.set_title(title or f"{fd['field']} {show}{rng}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    if apply_tick_style is not None:
        try:
            apply_tick_style(ax, direction=tick_direction, right=tick_right)
        except Exception:                       # WCSAxes may not accept every option
            pass
    return fig, ax


# =====================================================================
# ONE-CALL DRIVER
# =====================================================================
def run(field, mag_col="gmagA", drop_g_masked=True, drop_stars=False,
        drop_which="MOSDEF", refresh=False, dm=0.5, plot=True):
    fd = load_field(field)
    geo = check_geometry(fd)
    cat = fetch_cfhtls_deep(geo["center"], geo["radius_deg"], fd["field"], refresh=refresh,
                            mag_col=mag_col)
    describe_catalog(cat)
    cov = check_coverage(cat, fd)
    dep = check_depth(cat, mag_col=mag_col)

    cl = clean_catalog(cat, mag_col=mag_col, drop_g_masked=drop_g_masked,
                       drop_stars=drop_stars)
    cl, tmags = drop_targets(cl, fd["field"], which=drop_which, return_mags=True)
    look = mask_lookup(cl, fd)
    cur = masked_fraction_curve(look["mag"], look["masked"], dm=dm)
    cur.pprint(max_lines=-1, max_width=-1)
    lim = mag_limits(cur)
    floor = empirical_floor(cur)
    print(f"== masked sky fraction: {geo['f_rand']:.4f}")
    pred = predict_target_fraction(cur, tmags) if len(tmags) else np.nan
    fig = plot_curve(cur, lim, geo["f_rand"], fd["field"], floor=floor,
                     mag_label=f"g ({'AUTO' if mag_col == 'gmagA' else '3in aper'}, AB)") if plot else None
    return dict(fd=fd, geometry=geo, catalog=cat, coverage=cov, depth=dep,
                lookup=look, curve=cur, limits=lim, floor=floor,
                target_mags=tmags, target_pred=pred, fig=fig)


# =====================================================================
# PART 3 -- do the TARGETS mask themselves?  (3D-HST [O III] sample)
# =====================================================================
# THE QUESTION
#   The baked mask had MOSDEF segments switched off by hand. Any other target
#   (e.g. the 3D-HST [O III] galaxies used for the Niemeyer comparison) that is
#   brighter than the mask limit (m50 ~ 25.2 in COSMOS) has its OWN segment
#   still in the mask -> its central fibers are thrown away. That can only
#   lower (or delete) the galaxy's contribution to the innermost bins.
#
# HOW WE MEASURE IT (pixels stand in for fibers)
#   A fiber is dropped when its centre lands on a masked pixel. Fiber centres
#   fall at random sub-arcsec positions around a galaxy, so the fraction of
#   MASKED PIXELS in an annulus = the expected fraction of FIBERS dropped there.
#
#   for each target:
#       centre_masked = mask[rint(y), rint(x)]          (same lookup as a fiber)
#       for each annulus [r_i, r_i+1):
#           f_i = N(masked & valid pixels) / N(valid pixels)
#   stacked over targets: mean f_i  vs  masked sky fraction f_sky
#
#   Far from the target, f_i -> f_sky (neighbours only). An EXCESS over f_sky
#   at small r is self-masking. Its size and radial reach tell you which kpc
#   bins are affected and by how much.
#
# WHO IS EXEMPT
#   Targets that are also MOSDEF galaxies had their segment zeroed, so they are
#   flagged `in_mosdef` and reported separately.

def _field_values(t, field_col):
    """Normalised field name per row (AEGIS/COSMOS), '' where unknown."""
    for c in (field_col, "FIELD", "SUBSET"):
        if c and c in t.colnames:
            fv = np.char.upper(np.char.strip(np.array(t[c]).astype(str)))
            return np.array([FIELD_ALIASES.get(v, v) for v in fv]), c
    return None, None


def load_targets(field, targets=None, which="3DHST", target_path=TARGET_CATALOG,
                 ra_col="RA", dec_col="DEC", id_col="ID", z_col="z",
                 field_col="FIELD", mosdef_radius_arcsec=1.0):
    """
    The target list for PART 3, as a Table with standard columns
    RA, DEC, ID, z, in_mosdef (+ nfib, r_edges when built from a product).

    targets may be
      * a GalaxyProduct (e.g. products['COSMOS'] or prod_combined from the
        notebook) -> its .catalog: exactly the galaxies that survived
        extraction + apply_finite_cut. Its NFIB (fibers actually used per
        galaxy per radial bin) and R_EDGES are carried along, so PART 3 can
        put the pixel-level prediction next to the real fiber counts.
      * an astropy Table or a FITS path -> your own catalog (name its columns
        with ra_col/dec_col/id_col/z_col)
      * None -> KEEP catalog rows with CATALOG == `which` (fallback only)
    Rows are filtered to `field` using field_col, else FIELD, else SUBSET (the
    label combine_products adds). No field column -> every row kept; rows off
    the mask footprint are dropped later.

    in_mosdef = True when a KEEP-catalog MOSDEF galaxy lies within
    mosdef_radius_arcsec (its segment was un-masked by hand).
    """
    field = _field(field)
    nfib = r_edges = None
    if targets is None:
        t = Table.read(target_path)
        sel = np.array(t["FIELD"]).astype(str) == field
        if which is not None:
            sel &= np.array(t["CATALOG"]).astype(str) == which
        t, src = t[sel], f"KEEP {which}"
    else:
        if hasattr(targets, "catalog") and hasattr(targets, "nfib"):       # GalaxyProduct
            t = Table(targets.catalog)
            nfib = np.asarray(targets.nfib, float)
            re_ = np.asarray(targets.r_edges, float)
            r_edges = np.broadcast_to(re_, (len(t), re_.shape[-1])) if re_.ndim == 1 else re_
            src = "galaxy product"
        elif isinstance(targets, (str, os.PathLike)):
            t, src = Table.read(targets), "catalog file"
        else:
            t, src = Table(targets), "user table"
        fv, used = _field_values(t, field_col)
        if fv is not None:
            keep = fv == field
            t = t[keep]
            if nfib is not None:
                nfib, r_edges = nfib[keep], r_edges[keep]

    out = Table()
    out["RA"] = np.asarray(t[ra_col], float)
    out["DEC"] = np.asarray(t[dec_col], float)
    out["ID"] = np.asarray(t[id_col]) if id_col in t.colnames else np.arange(len(t))
    out["z"] = np.asarray(t[z_col], float) if z_col in t.colnames else np.full(len(t), np.nan)
    if nfib is not None:
        out["nfib"] = nfib
        out["r_edges"] = np.array(r_edges)

    # MOSDEF exemption flag
    k = Table.read(target_path)
    sel = ((np.array(k["FIELD"]).astype(str) == field)
           & (np.array(k["CATALOG"]).astype(str) == "MOSDEF"))
    out["in_mosdef"] = False
    if sel.any() and len(out):
        cm = SkyCoord(np.asarray(k["RA"][sel], float) * u.deg,
                      np.asarray(k["DEC"][sel], float) * u.deg)
        ct = SkyCoord(out["RA"] * u.deg, out["DEC"] * u.deg)
        _, sep, _ = ct.match_to_catalog_sky(cm)
        out["in_mosdef"] = sep.arcsec < mosdef_radius_arcsec
    print(f"== PART 3 targets: {len(out)} in {field} ({src}); "
          f"{int(np.sum(out['in_mosdef']))} are also MOSDEF (hand-unmasked)")
    return out


def _kpc_per_arcsec(z):
    """Planck18, the pipeline's single cosmology (virial.cosmo)."""
    try:
        from utils_lya_halo.virial import cosmo
    except ImportError:                                          # pragma: no cover
        from astropy.cosmology import Planck18 as cosmo
    z = np.asarray(z, float)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0)
    out[ok] = cosmo.angular_diameter_distance(z[ok]).to(u.kpc).value / 206264.806
    return out


# Niemeyer 2022 radial bins (kpc) -- the ones the comparison notebook uses
N22_KPC_EDGES = (0, 5, 10, 15, 20, 30, 40, 60, 80, 160)
DEFAULT_ARCSEC_EDGES = (0, 0.75, 1.5, 2.5, 3.5, 5, 7.5, 10, 15, 20)


def target_self_mask(fd, targets, edges=None, unit="kpc", max_r=180.0,
                     core_radius=3.5, fiber_radius=0.75):
    """
    Per-target masked fraction in radial annuli, plus the stacked profile.

    edges : annulus edges. None (default) -> the product's own radial bins
            (targets['r_edges'], i.e. cfg.bins) up to max_r, so every annulus
            lines up with an NFIB column; without a product, None falls back
            to DEFAULT_ARCSEC_EDGES in arcsec.
            unit='arcsec' uses edges as-is; unit='kpc' converts PER TARGET with
            its own z (Planck18). N22_KPC_EDGES gives Niemeyer's bins.
    max_r : outermost edge kept when edges=None (in `unit`). The mask only
            matters on segment scales, so the big outer bins add nothing.
    core_radius  : the core extraction aperture (core.extract_core_spectra
                   default 3.5"); f_core = masked fraction inside it.
    fiber_radius : VIRUS fiber radius (0.75"); f_fiber = masked fraction of
                   the central fiber footprint.

    Returns dict(per_target=Table, profile=Table, f_sky=float).
      per_target columns: ID, z, in_mosdef, on_data, centre_masked,
                          f_fiber, f_core, f_ann (n_ann vector)
      profile columns   : r_lo, r_hi, f_mean (all non-MOSDEF targets on data),
                          f_p16, f_p84 (bootstrap over targets),
                          f_mosdef (mean for MOSDEF-matched), excess = f_mean - f_sky
    """
    mask, valid, wcs = fd["mask"], fd["valid"], fd["wcs_mask"]
    ny, nx = mask.shape
    scale = _pixscale_arcsec(wcs)
    from_product = edges is None and "r_edges" in targets.colnames
    if from_product:
        re0 = np.asarray(targets["r_edges"][0], float)
        n_ann = int(np.sum(re0[1:] <= max_r)) or 1
        per_row_edges = np.asarray(targets["r_edges"], float)[:, :n_ann + 1]
        edges = per_row_edges[0]
    else:
        if edges is None:
            edges, unit = DEFAULT_ARCSEC_EDGES, "arcsec"
        edges = np.asarray(edges, float)
        n_ann = len(edges) - 1
        per_row_edges = np.broadcast_to(edges, (len(targets), n_ann + 1))
    f_sky = float(mask[valid].mean())

    kpa = _kpc_per_arcsec(targets["z"]) if unit == "kpc" else np.ones(len(targets))
    if unit == "kpc" and not np.all(np.isfinite(kpa)):
        print(f"  warning: {np.sum(~np.isfinite(kpa))} targets have no z -> "
              f"skipped for kpc annuli")

    x, y = wcs.world_to_pixel(SkyCoord(np.asarray(targets["RA"]) * u.deg,
                                       np.asarray(targets["DEC"]) * u.deg))
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)

    N = len(targets)
    centre = np.zeros(N, bool)
    on_data = np.zeros(N, bool)
    f_fiber = np.full(N, np.nan)
    f_core = np.full(N, np.nan)
    f_ann = np.full((N, n_ann), np.nan)

    for i in range(N):
        if not inside[i] or not valid[yi[i], xi[i]]:
            continue
        on_data[i] = True
        centre[i] = mask[yi[i], xi[i]]
        e_i = per_row_edges[i]
        r_edges_as = e_i / kpa[i] if unit == "kpc" else e_i            # -> arcsec
        r_max = max(np.nanmax(r_edges_as), core_radius, fiber_radius)
        if not np.isfinite(r_max):
            continue
        h = int(np.ceil(r_max / scale)) + 1
        y0, y1 = max(0, yi[i] - h), min(ny, yi[i] + h + 1)
        x0, x1 = max(0, xi[i] - h), min(nx, xi[i] + h + 1)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(gx - x[i], gy - y[i]) * scale                     # arcsec, from true centre
        m, v = mask[y0:y1, x0:x1], valid[y0:y1, x0:x1]

        def _frac(sel):
            sel = sel & v
            return m[sel].mean() if sel.any() else np.nan

        f_fiber[i] = _frac(r <= fiber_radius)
        f_core[i] = _frac(r <= core_radius)
        for j in range(n_ann):
            f_ann[i, j] = _frac((r >= r_edges_as[j]) & (r < r_edges_as[j + 1]))

    per = Table(dict(ID=targets["ID"], z=targets["z"], in_mosdef=targets["in_mosdef"],
                     on_data=on_data, centre_masked=centre,
                     f_fiber=f_fiber, f_core=f_core, f_ann=f_ann))
    if from_product:
        per["nfib"] = np.asarray(targets["nfib"], float)[:, :n_ann]
    per.meta.update(edges=list(edges), unit=unit, core_radius=core_radius,
                    fiber_radius=fiber_radius, f_sky=f_sky)

    # stacked profile over non-MOSDEF targets (the ones not hand-edited)
    use = on_data & ~np.asarray(targets["in_mosdef"], bool)
    usem = on_data & np.asarray(targets["in_mosdef"], bool)
    A = f_ann[use]
    fmean = np.nanmean(A, axis=0) if len(A) else np.full(n_ann, np.nan)
    rng = np.random.default_rng(0)
    if len(A) > 1:
        boots = np.array([np.nanmean(A[rng.integers(0, len(A), len(A))], axis=0)
                          for _ in range(500)])
        p16, p84 = np.nanpercentile(boots, [16, 84], axis=0)
    else:
        p16 = p84 = np.full(n_ann, np.nan)
    fm = np.nanmean(f_ann[usem], axis=0) if usem.any() else np.full(n_ann, np.nan)
    prof = Table(dict(r_lo=edges[:-1], r_hi=edges[1:], f_mean=fmean, f_p16=p16,
                      f_p84=p84, f_mosdef=fm, excess=fmean - f_sky))
    for c in ("f_mean", "f_p16", "f_p84", "f_mosdef", "excess"):
        prof[c].format = ".3f"
    prof.meta.update(unit=unit, f_sky=f_sky, n_targets=int(use.sum()),
                     n_mosdef=int(usem.sum()))

    # the REAL fibers: NFIB per bin, centre-masked vs clean (non-MOSDEF) targets.
    # If self-masking bites, the masked group has fewer fibers in the inner
    # bins, by roughly the fraction f_ann predicts, and ~the same further out.
    if from_product:
        nf = per["nfib"]
        clean = use & ~centre
        bad = use & centre
        prof["nfib_clean"] = np.nanmedian(nf[clean], axis=0) if clean.any() else np.nan
        prof["nfib_masked"] = np.nanmedian(nf[bad], axis=0) if bad.any() else np.nan
        prof["nfib_zero_masked"] = (np.mean(nf[bad] == 0, axis=0) if bad.any()
                                    else np.nan)
        prof["nfib_zero_masked"].format = ".2f"

    # summary
    nu = int(use.sum())
    print(f"== PART 3 self-masking: {fd['field']}, {on_data.sum()} targets on data "
          f"({nu} not MOSDEF, {usem.sum()} MOSDEF-matched); masked sky fraction {f_sky:.3f}")
    print(f"  annuli in {unit}" + (" = the product's radial bins" if from_product else ""))
    if nu:
        cm, ff, fc = centre[use], f_fiber[use], f_core[use]
        print(f"  centre pixel masked  : {cm.sum():4d} / {nu}  ({cm.mean():.1%})"
              f"   <- these galaxies' central fibers are dropped")
        print(f"  f(<{fiber_radius}\") > 0.5    : {np.sum(ff > 0.5):4d} / {nu}  ({np.mean(ff > 0.5):.1%})")
        print(f"  f(<{core_radius}\") > 0.5     : {np.sum(fc > 0.5):4d} / {nu}  ({np.mean(fc > 0.5):.1%})")
        print(f"  mean f(<{core_radius}\")      : {np.nanmean(fc):.3f}   (vs f_sky {f_sky:.3f})")
    if usem.any():
        print(f"  MOSDEF-matched check : centre masked {centre[usem].sum()} / {usem.sum()} "
              f"(expect ~0 -- they were un-masked by hand)")
    prof.pprint(max_lines=-1, max_width=-1)
    return dict(per_target=per, profile=prof, f_sky=f_sky)


def plot_self_mask_profile(res, field="", tick_direction="in", tick_right=True):
    """
    Stacked masked fraction around the targets vs radius, with the masked sky
    fraction as the 'neighbours only' reference. Excess above the line at small
    r = the targets masking themselves.
    """
    import matplotlib.pyplot as plt
    prof, f_sky = res["profile"], res["f_sky"]
    unit = prof.meta.get("unit", "arcsec")
    rc = 0.5 * (prof["r_lo"] + prof["r_hi"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(rc, prof["f_mean"],
                yerr=np.clip([prof["f_mean"] - prof["f_p16"], prof["f_p84"] - prof["f_mean"]], 0, None),
                xerr=[rc - prof["r_lo"], prof["r_hi"] - rc], fmt="o", color="k", ms=4,
                capsize=0, label=f"targets, not MOSDEF (N={prof.meta['n_targets']})")
    if prof.meta.get("n_mosdef", 0):
        ax.plot(rc, prof["f_mosdef"], "s", mfc="none", color="C0", ms=5,
                label=f"MOSDEF-matched (N={prof.meta['n_mosdef']})")
    ax.axhline(f_sky, ls="--", color="0.5", label=f"masked sky fraction = {f_sky:.3f}")
    ax.set_xscale("symlog", linthresh=1.0 if unit == "arcsec" else 5.0)
    ax.set_xlabel(f"radius [{unit}]")
    ax.set_ylabel("masked fraction of pixels (= fibers lost)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{field} self-masking profile")
    ax.legend(frameon=False, fontsize=9)
    if apply_tick_style is not None:
        apply_tick_style(ax, direction=tick_direction, right=tick_right)
    return fig, ax


def plot_target_stamps(fd, targets, res=None, size_arcsec=20.0, ncol=8, per_page=48,
                       sort_by="f_core", only=None, circles=(0.75, 3.5),
                       stretch=(5.0, 99.7), cmap="binary",
                       tint=True, mask_color="tab:orange", mask_alpha=0.35,
                       contour_color="tab:green", contour_lw=0.8,
                       stamp_inches=1.6):
    """
    Postage stamps: CFHTLS image around each target, the baked mask tinted and
    outlined, a cross at the target, and circles at the fiber radius (0.75")
    and the core aperture (3.5").

    res     : output of target_self_mask (for labels and sorting). If None it
              is computed with the default annuli.
    tint    : False -> mask shown as an outline only (galaxy fully visible)
    sort_by : 'f_core' (worst first), 'f_fiber', 'z', or None (catalog order).
    only    : None (all on data) | 'masked' (centre masked or f_core > 0) |
              'clean' | 'mosdef' -- to page through just one group.
    Label per stamp:  ID  z
                      c=Y/N (centre pixel masked)  f3.5=<masked frac in core>
              with 'M' when the target is MOSDEF-matched (hand-unmasked).
    Nothing is written to disk. Returns the list of figures (one per page).
    """
    import matplotlib.pyplot as plt

    if res is None:
        res = target_self_mask(fd, targets)
    per = res["per_target"]
    idx = np.where(np.asarray(per["on_data"]))[0]
    cm = np.asarray(per["centre_masked"])
    fc = np.asarray(per["f_core"], float)
    ism = np.asarray(per["in_mosdef"], bool)
    if only == "masked":
        idx = idx[(cm[idx]) | (fc[idx] > 0)]
    elif only == "clean":
        idx = idx[~cm[idx] & (fc[idx] == 0)]
    elif only == "mosdef":
        idx = idx[ism[idx]]
    elif only is not None:
        raise ValueError("only must be None, 'masked', 'clean' or 'mosdef'")
    if sort_by is not None:
        key = np.asarray(per[sort_by], float)[idx]
        idx = idx[np.argsort(-key if sort_by.startswith("f_") else key, kind="stable")]
    if not len(idx):
        print("no targets to show")
        return []

    img, mask, wcs = fd["image"], fd["mask"], fd["wcs_mask"]
    ny, nx = mask.shape
    scale = _pixscale_arcsec(wcs)
    h = int(round(size_arcsec / scale / 2))
    x, y = wcs.world_to_pixel(SkyCoord(np.asarray(targets["RA"]) * u.deg,
                                       np.asarray(targets["DEC"]) * u.deg))
    tint_rgba = plt.matplotlib.colors.to_rgba(mask_color, mask_alpha)

    figs = []
    pages = [idx[k:k + per_page] for k in range(0, len(idx), per_page)]
    for p, page in enumerate(pages):
        nrow = int(np.ceil(len(page) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * stamp_inches, nrow * stamp_inches * 1.12),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        for ax, i in zip(axes.ravel(), page):
            cx, cy = int(np.rint(x[i])), int(np.rint(y[i]))
            # fixed-size stamp, padded where it runs off the mask grid
            st_im = np.full((2 * h + 1, 2 * h + 1), np.nan, np.float32)
            st_mk = np.zeros((2 * h + 1, 2 * h + 1), bool)
            ys0, ys1 = max(0, cy - h), min(ny, cy + h + 1)
            xs0, xs1 = max(0, cx - h), min(nx, cx + h + 1)
            sy, sx = ys0 - (cy - h), xs0 - (cx - h)
            st_im[sy:sy + ys1 - ys0, sx:sx + xs1 - xs0] = img[ys0:ys1, xs0:xs1]
            st_mk[sy:sy + ys1 - ys0, sx:sx + xs1 - xs0] = mask[ys0:ys1, xs0:xs1]
            good = np.isfinite(st_im) & (st_im != 0)
            vmin, vmax = np.percentile(st_im[good], stretch) if good.any() else (0, 1)
            ext = (-h * scale - scale / 2, h * scale + scale / 2,
                   -h * scale - scale / 2, h * scale + scale / 2)    # arcsec, centred on target pixel
            ax.imshow(st_im, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=ext, interpolation="nearest")
            if st_mk.any():
                if tint:
                    tint_img = np.zeros(st_mk.shape + (4,))
                    tint_img[st_mk] = tint_rgba
                    ax.imshow(tint_img, origin="lower", extent=ext, interpolation="nearest")
                ax.contour(st_mk.astype(float), levels=[0.5], colors=contour_color,
                           linewidths=contour_lw, extent=ext, origin="lower")
            # target at its sub-pixel position relative to the stamp centre pixel
            ox, oy = (x[i] - cx) * scale, (y[i] - cy) * scale
            ax.plot(ox, oy, "+", color="red", ms=7, mew=1.0)
            for rr, ls in zip(circles, ("-", "--")):
                ax.add_patch(plt.Circle((ox, oy), rr, fill=False, ec="C0", lw=0.8, ls=ls))
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            zi = per["z"][i]
            ztxt = f"z={zi:.2f}" if np.isfinite(zi) else ""
            flag = " M" if ism[i] else ""
            col = "C3" if cm[i] else ("C1" if fc[i] > 0 else "k")
            ax.set_title(f"{per['ID'][i]}  {ztxt}{flag}\n"
                         f"c={'Y' if cm[i] else 'N'}  f3.5={fc[i]:.2f}",
                         fontsize=7, color=col, pad=2)
        fig.suptitle(f"{fd['field']}: targets + baked mask (orange/green)  "
                     f"{size_arcsec:.0f}\" stamps, circles {circles[0]}\" / {circles[-1]}\"  "
                     f"page {p + 1}/{len(pages)}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        figs.append(fig)

    return figs


def run_targets(source=None, field=None, edges=None, unit="kpc", max_r=180.0,
                stamps=True, stamp_only=None, **cat_kw):
    """
    PART 3 driver, straight from the notebook's products. Nothing is saved.

        out = md.run_targets(products)        # dict {'AEGIS': prod, 'COSMOS': prod}
        out = md.run_targets(prod_combined)   # split by FIELD / SUBSET column
        out = md.run_targets(products["COSMOS"], field="COSMOS")

    source : dict of products (keys = fields), one GalaxyProduct, a Table/path,
             or None (KEEP 3DHST rows; then `field` is required).
    field  : restrict to one field; None -> every field present in `source`.
    edges/unit/max_r : see target_self_mask (default: the product's own bins
             out to 180 kpc).
    cat_kw : -> load_targets (ra_col, dec_col, id_col, z_col, field_col, which).

    Returns {field: dict(fd, targets, result, profile_fig, stamp_figs)}.
    """
    if isinstance(source, dict):
        jobs = [(k, v) for k, v in source.items()]
    elif source is None:
        if field is None:
            raise ValueError("source=None needs field=")
        jobs = [(field, None)]
    else:
        t = source.catalog if hasattr(source, "catalog") else (
            Table.read(source) if isinstance(source, (str, os.PathLike)) else Table(source))
        fv, _ = _field_values(Table(t), cat_kw.get("field_col", "FIELD"))
        fields = [field] if field is not None else (
            sorted(set(fv)) if fv is not None else [])
        if not fields:
            raise ValueError("no field column in source -- pass field=")
        jobs = [(f, source) for f in fields]
    if field is not None:
        jobs = [(f, s_) for f, s_ in jobs if _field(f) == _field(field)]

    results = {}
    for f, src in jobs:
        fd = load_field(f)
        tg = load_targets(fd["field"], targets=src, **cat_kw)
        res = target_self_mask(fd, tg, edges=edges, unit=unit, max_r=max_r)
        prof_fig = plot_self_mask_profile(res, fd["field"])
        figs = plot_target_stamps(fd, tg, res, only=stamp_only) if stamps else []
        results[fd["field"]] = dict(fd=fd, targets=tg, result=res,
                                    profile_fig=prof_fig, stamp_figs=figs)
    return results
