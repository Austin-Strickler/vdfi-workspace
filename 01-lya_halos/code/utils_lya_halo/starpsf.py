"""
starpsf.py -- empirical star PSF + blank-sky null, on the EXISTING Stage-1 path.

Both tools here work by synthesizing a run-table of POSITIONS and pushing it
through the same extract.extract_fibers loop the galaxies use -- no new
extraction code, just new inputs. They differ in exactly one place, the fiber
mask:

  * STAR PSF   -> mask='none'. The continuum-image mask flags BRIGHT fibers as
                  bad (to find blank sky for the background); a star is the
                  brightest source, so the normal mask would delete its core.
                  We must keep every fiber for the star.
  * BLANK SKY  -> mask='auto'. Here we WANT real sources rejected, so we build
                  the normal config mask. Random positions on rejected-source
                  sky should yield a flat continuum and a null centroid -- this
                  is the additive sky/background-residual null (complements the
                  z-scramble coadd null and the placebo estimator null).

Two binning modes for stars, both via the real pipeline:
  * arcsec mode   -> stars and galaxies in absolute angular bins (give fine
                     sub-arcsec inner edges). z/mass are dummies (unused for
                     arcsec binning). The PSF curve-of-growth in arcsec.
  * matched mode  -> assign each star a (mass, z) PAIR drawn from the real
                     galaxy sample, so virial_to_kpc_bins puts the point source
                     into the SAME R/Rvir bins the galaxy analysis uses. This is
                     the "blur": how a point source spreads across the virial
                     bins, i.e. the PSF contribution to each galaxy bin.

The PSF profile is measured OBSERVED-FRAME, straight off the Stage-1 product
(continuum band coadd per bin). Stars have no rest frame, so build_stacks'
(1+z) resample is meaningless for them and is deliberately skipped. The blank-
sky null DOES go through build_stacks (real z assigned; the point is that the
line vanishes).

Not in utils_lya_halo/__init__ (imports matplotlib / triggers field load). Use:
    from utils_lya_halo import starpsf
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.stats import biweight_location, mad_std

from .io import GalaxyProduct, apply_finite_cut
from .stack import build_stacks
from .measure import (
    flux_weighted_centroid, integrated_line_flux, stack_galaxies,
    DEFAULT_CONT_BOUNDS, DEFAULT_CONT_METHOD, DEFAULT_CONT_ORDER,
)
from .plotting import _setup_radius_axis, _resolve_bin_mode, _get_vr_biweight_v

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm optional
    def tqdm(x, **kwargs):
        return x

if TYPE_CHECKING:
    from .config import PipelineConfig

C_KMS = 299792.458

__all__ = [
    "make_position_run_table",
    "make_star_run_table",
    "make_random_sky_table",
    "run_positions_extract",
    "radial_continuum_profile",
    "curve_of_growth",
    "build_piecewise_psf",
    "describe_piecewise_psf",
    "plot_piecewise_psf",
    "fiber_moffat",
    "fit_psf_moffat",
    "describe_psf_fit",
    "plot_psf_composite",
    "flux_scale_check",
    "plot_radial_profile",
    "plot_curve_of_growth",
    "blank_sky_centroid_null",
    "plot_blank_sky_null",
    "LIS_LINES",
    "measure_line_profiles",
    "bootstrap_line_profiles",
    "plot_line_sn",
    "plot_line_sn_radial",
    "psf_profiles_for_lines",
    "plot_line_profiles",
]


# =====================================================================
# 1. SYNTHETIC RUN-TABLES  (shared by stars and blank sky)
# =====================================================================
def _sample_mass_z(n, source_table, z_col, mass_col, rng):
    """Draw n PAIRED (z, mass) rows from a real catalog/run-table (with
    replacement), preserving their joint distribution."""
    z = np.asarray(source_table[z_col], dtype=float)
    m = np.asarray(source_table[mass_col], dtype=float)
    good = np.isfinite(z) & np.isfinite(m) & (z > 0)
    if not np.any(good):
        raise ValueError("source_table has no finite (z, mass) rows to sample.")
    idx = rng.integers(0, int(good.sum()), size=n)
    return z[good][idx], m[good][idx]


def make_position_run_table(ra, dec, z, mass, *, field, catalog, z_col, mass_col):
    """
    Minimal run-table extract_fibers can consume: RA, DEC, the z/mass columns
    named by config, plus FIELD/CATALOG for provenance. Scalars broadcast.

    NB: this table is passed STRAIGHT to extract.extract_fibers (we do NOT round-
    trip through build_run_table), so its rows are used as-is -- no FIELD/CATALOG
    filtering surprises.
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    n = ra.size
    z = np.full(n, z, dtype=float) if np.isscalar(z) else np.asarray(z, dtype=float)
    mass = np.full(n, mass, dtype=float) if np.isscalar(mass) else np.asarray(mass, dtype=float)
    if not (dec.size == n == z.size == mass.size):
        raise ValueError("ra/dec/z/mass length mismatch.")
    t = Table()
    t["RA"] = ra
    t["DEC"] = dec
    t[z_col] = z
    t[mass_col] = mass
    t["FIELD"] = np.full(n, str(field))
    t["CATALOG"] = np.full(n, str(catalog))
    return t


def make_star_run_table(config, ra, dec, *, mode="arcsec", source_table=None,
                        dummy_z=1.0, dummy_mass=10.0, seed=0):
    """
    Build a star run-table.

    mode='arcsec'  : z/mass are dummies (NOT used for arcsec binning). Set
                     config.bin_mode='arcsec' with fine inner edges.
    mode='matched' : draw (z, mass) PAIRS from source_table (the real galaxy
                     run-table) so virial binning matches the galaxy sample.
                     Requires source_table; set config.bin_mode='virial'.

    IMPORTANT: run stars with a config whose `catalog` (and ideally output_dir or
    catalog_path) DIFFERS from the galaxy run, or the star galaxy-FITS / cache
    will collide with the real one (descriptive_filename keys off catalog +
    catalog_path hash). e.g. replace(galaxy_cfg, catalog='STAR_PSF',
    output_dir='./outputs_starpsf', bin_mode='arcsec', bins=[0,.75,1.5,...]).
    """
    rng = np.random.default_rng(seed)
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    n = ra.size
    if mode == "arcsec":
        z = np.full(n, float(dummy_z)); m = np.full(n, float(dummy_mass))
    elif mode == "matched":
        if source_table is None:
            raise ValueError("mode='matched' needs source_table (the real galaxy "
                             "run-table) to draw (z, mass) pairs from.")
        z, m = _sample_mass_z(n, source_table, config.z_col, config.mass_col, rng)
    else:
        raise ValueError("mode must be 'arcsec' or 'matched'.")
    return make_position_run_table(
        ra, dec, z, m, field=config.normalized_field(), catalog=config.catalog,
        z_col=config.z_col, mass_col=config.mass_col,
    )


def make_random_sky_table(config, n, *, source_table, ra_bounds=None,
                          dec_bounds=None, avoid_radius_arcsec=10.0,
                          margin_arcsec=5.0, mass_z_from="source",
                          seed=0, max_tries=50):
    """
    Random in-footprint positions for the blank-sky null. Draws uniform RA/DEC
    inside the footprint, REJECTS any draw within avoid_radius_arcsec of a real
    source, and assigns (z, mass) paired from source_table so the rest-frame
    mapping (and virial binning) is realistic.

    source_table : the real galaxy run-table; supplies the source positions to
                   avoid, the (z, mass) pairs, AND (if ra_bounds/dec_bounds are
                   None) the footprint via its RA/DEC min-max plus a margin.

    Returns a table of n accepted positions (fewer if max_tries is exhausted; a
    warning is printed). Feed to run_positions_extract(..., mask='auto').
    """
    rng = np.random.default_rng(seed)
    src_ra = np.asarray(source_table["RA"], dtype=float)
    src_dec = np.asarray(source_table["DEC"], dtype=float)
    finite = np.isfinite(src_ra) & np.isfinite(src_dec)
    src_ra, src_dec = src_ra[finite], src_dec[finite]

    if ra_bounds is None:
        pad = margin_arcsec / 3600.0
        ra_bounds = (src_ra.min() - pad, src_ra.max() + pad)
    if dec_bounds is None:
        pad = margin_arcsec / 3600.0
        dec_bounds = (src_dec.min() - pad, src_dec.max() + pad)

    cosd = np.cos(np.deg2rad(np.mean(dec_bounds)))
    avoid_deg = avoid_radius_arcsec / 3600.0
    acc_ra, acc_dec = [], []
    tries = 0
    while len(acc_ra) < n and tries < n * max_tries:
        k = (n - len(acc_ra)) * 4
        cand_ra = rng.uniform(ra_bounds[0], ra_bounds[1], size=k)
        cand_dec = rng.uniform(dec_bounds[0], dec_bounds[1], size=k)
        for r, d in zip(cand_ra, cand_dec):
            dra = (src_ra - r) * cosd
            ddec = (src_dec - d)
            if np.min(dra * dra + ddec * ddec) > avoid_deg * avoid_deg:
                acc_ra.append(r); acc_dec.append(d)
                if len(acc_ra) >= n:
                    break
        tries += k
    if len(acc_ra) < n:
        warnings.warn(f"random sky: only placed {len(acc_ra)}/{n} positions "
                      f"clear of sources; loosen avoid_radius_arcsec or footprint.")
    acc_ra = np.array(acc_ra); acc_dec = np.array(acc_dec)

    if mass_z_from == "source":
        z, m = _sample_mass_z(acc_ra.size, source_table, config.z_col,
                              config.mass_col, rng)
    else:
        z = np.full(acc_ra.size, 1.0); m = np.full(acc_ra.size, 10.0)
    return make_position_run_table(
        acc_ra, acc_dec, z, m, field=config.normalized_field(),
        catalog=config.catalog, z_col=config.z_col, mass_col=config.mass_col,
    )


# =====================================================================
# 2. EXTRACTION DRIVER  (mirrors run_extract; injectable fiber mask)
# =====================================================================
def run_positions_extract(config, table, *, mask="none", custom_mask=None,
                          use_cache=False, write=False, descriptive_name=True,
                          verbose=True) -> GalaxyProduct:
    """
    Run Stage-1 extraction on a synthesized position table, returning an
    in-memory GalaxyProduct (header omitted unless written).

    mask : 'none'   -> all fibers kept (REQUIRED for the star PSF: the normal
                       mask would delete the bright star core).
           'auto'   -> build the normal config mask (for blank sky: rejects real
                       sources; needs cont_image_path if mask_method='image').
           'custom' -> pass custom_mask, shape (nexp, nfib) bool, True=bad.

    Cost: this loads the full VDFI field (memmapped) and loops fibers, like the
    real Stage 1. Use a SMALL table (a few hundred stars / sky positions) and
    optionally use_cache. This is the one expensive call here; the profile and
    null functions downstream are cheap and re-runnable on the returned product.
    """
    from . import extract                       # heavy (field load); import lazily
    from .io import write_galaxy_fits
    import os

    vdfi = extract.load_field(config)
    nexp = vdfi["RA"].shape[0]
    nfib = vdfi["RA"].shape[1]

    if mask == "none":
        bad = np.zeros((nexp, nfib), dtype=bool)
    elif mask == "auto":
        # `table` here is the synthesized star/sky position table, not a real
        # galaxy catalog -- passed through anyway so mask_protect_radius_arcsec
        # works if you ever explicitly want it (e.g. protecting a star's own
        # core in a PSF validation run). Leave mask_protect_radius_arcsec=None
        # in config for a normal blank-sky/star run where that protection
        # isn't meaningful.
        bad = extract.build_bad_fiber_mask(config, vdfi, run_table=table)
    elif mask == "custom":
        if custom_mask is None:
            raise ValueError("mask='custom' needs custom_mask (nexp, nfib).")
        bad = np.asarray(custom_mask, dtype=bool)
    else:
        raise ValueError("mask must be 'none', 'auto', or 'custom'.")

    # return_extras is driven by the config's own store_* / bg_variants flags
    # (config.stores_bg_extras()). Without this the flags are honoured inside
    # extract_fibers and then SILENTLY DISCARDED here -- store_spec_nobg=True
    # would cost the compute and produce nothing.
    want_extras = bool(getattr(config, "stores_bg_products", lambda: False)())
    if want_extras:
        spec, err, nfib_arr, background, vr, extras = extract.extract_fibers(
            config, vdfi, table, bad, use_cache=use_cache, return_extras=True)
    else:
        spec, err, nfib_arr, background, vr = extract.extract_fibers(
            config, vdfi, table, bad, use_cache=use_cache)
        extras = None
    ngal = spec.shape[0]
    catalog = table[:ngal]
    r_edges = np.asarray(config.bins, dtype=float)

    if write:
        os.makedirs(config.output_dir, exist_ok=True)
        fname = (config.descriptive_filename() if descriptive_name
                 else f"positions_{config.normalized_field()}_{config.catalog}.fits")
        path = os.path.join(config.output_dir, fname)
        source_id = os.path.join(config.resolve_vdfi_base_path(), "all_flux_final.fits")
        write_galaxy_fits(path, spec=spec, err=err, nfib=nfib_arr,
                          wave=vdfi["wave"], r_edges=r_edges, catalog=catalog,
                          config=config, source_id=source_id,
                          background=background, extras=extras, overwrite=True)
        if verbose:
            print(f"wrote {path}")

    if verbose:
        print(f"extracted {ngal} positions, mask='{mask}', bin_mode={config.bin_mode}")
    ex = extras or {}
    return GalaxyProduct(
        spec=spec, err=err, nfib=nfib_arr, wave=vdfi["wave"],
        r_edges=r_edges, catalog=catalog, header=None, background=background,
        spec_nobg=ex.get("spec_nobg"), bg_variants=ex.get("bg_raw"),
        bg_nfib=ex.get("bg_nfib"), bg_halves=ex.get("bg_halves"),
        bg_half_nfib=ex.get("bg_half_nfib"),
    )


# =====================================================================
# 3. RADIAL PROFILE  (observed-frame continuum; PSF curve-of-growth)
# =====================================================================
def _bin_midpoints(r_edges):
    """Geometric midpoint per bin; the 0-inner edge falls back to r_out/2 so it
    sits on a log axis."""
    r = np.asarray(r_edges, dtype=float)
    lo, hi = r[:-1], r[1:]
    return np.where(lo > 0, np.sqrt(lo * hi), hi / 2.0)


def radial_continuum_profile(product: GalaxyProduct, band=None, combine="biweight",
                             clip=3.0) -> dict:
    """
    Observed-frame continuum surface brightness per radial bin, coadded across
    objects (the star/galaxy axis). Per object/bin, the continuum is the sigma-
    clipped robust mean over `band`; objects are then combined per bin.

    band    : (lo, hi) observed-A window. None -> full finite wave range. Pick a
              clean band away from chip gaps / sky lines for real data.
    combine : 'biweight' | 'median' | 'mean' across objects.

    Returns dict: r_edges, r_mid, bin_mode-agnostic;
        sb        (nrad,)   coadded continuum per bin (per fiber, arbitrary flux)
        sb_err    (nrad,)   scatter-across-objects / sqrt(N_finite)
        per_obj   (nobj, nrad) per-object continuum (for diagnostics)
        nfib_med  (nrad,)   median fibers per bin (area proxy for curve-of-growth)
        n_obj_fin (nrad,)   objects contributing per bin
    """
    spec = np.asarray(product.spec, dtype=float)          # (nobj, nrad, nwave)
    wave = np.asarray(product.wave, dtype=float)
    nobj, nrad, _ = spec.shape

    if band is None:
        band_mask = np.isfinite(wave)
    else:
        band_mask = (wave >= band[0]) & (wave <= band[1])
    if not np.any(band_mask):
        raise ValueError("band selects no wavelength pixels.")

    # per object/bin continuum: sigma-clipped mean over the band
    sub = spec[:, :, band_mask]                            # (nobj, nrad, nb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(sub, axis=2, keepdims=True)
        sd = mad_std(sub, axis=2, ignore_nan=True)[:, :, None]
        keep = np.abs(sub - med) <= clip * np.where(sd > 0, sd, np.inf)
        sub_c = np.where(keep, sub, np.nan)
        per_obj = np.nanmean(sub_c, axis=2)                # (nobj, nrad)

    sb = np.full(nrad, np.nan); sb_err = np.full(nrad, np.nan)
    n_obj_fin = np.zeros(nrad, dtype=int)
    for r in range(nrad):
        col = per_obj[:, r]
        fin = np.isfinite(col)
        n_obj_fin[r] = int(fin.sum())
        if n_obj_fin[r] == 0:
            continue
        if combine == "biweight":
            sb[r] = biweight_location(col[fin])
        elif combine == "median":
            sb[r] = np.nanmedian(col[fin])
        elif combine == "mean":
            sb[r] = np.nanmean(col[fin])
        else:
            raise ValueError("combine must be 'biweight', 'median', or 'mean'.")
        sb_err[r] = mad_std(col[fin]) / np.sqrt(max(n_obj_fin[r], 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nfib_med = np.nanmedian(np.asarray(product.nfib, dtype=float), axis=0)

    return {
        "r_edges": np.asarray(product.r_edges, dtype=float),
        "r_mid": _bin_midpoints(product.r_edges),
        "sb": sb, "sb_err": sb_err, "per_obj": per_obj,
        "nfib_med": nfib_med, "n_obj_fin": n_obj_fin,
    }


def curve_of_growth(profile: dict, area_weight="nfib") -> dict:
    """
    Cumulative enclosed flux vs radius. The per-bin coadd `sb` is a per-fiber
    surface brightness; total flux in a bin ∝ sb × (fibers in bin), and fibers
    ∝ area, so enclosed(≤r) = cumsum(sb × weight). Normalized to 1 at the outer
    edge. The PSF curve rises fast and saturates; an extended galaxy profile
    keeps climbing -- the gap is the resolved (non-PSF) emission.

    area_weight : 'nfib' (median fibers/bin, default) | 'none' (equal weight).
    """
    sb = np.asarray(profile["sb"], dtype=float)
    if area_weight == "nfib":
        w = np.asarray(profile["nfib_med"], dtype=float)
    elif area_weight == "none":
        w = np.ones_like(sb)
    else:
        raise ValueError("area_weight must be 'nfib' or 'none'.")
    contrib = sb * w
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)
    cog = np.cumsum(contrib)
    total = cog[-1] if cog[-1] != 0 else np.nan
    return {
        "r_edges": profile["r_edges"], "r_mid": profile["r_mid"],
        "cog": cog, "cog_norm": cog / total, "weight": w,
    }


# =====================================================================
# 3b. PIECEWISE PSF  (magnitude classes -> one composite curve)
# =====================================================================
def build_piecewise_psf(profile, gmag, *, mag_edges=(8, 14, 17, 19, 21),
                        valid_radius=None, min_stars=3, min_sn=2.0,
                        ref_class=None, scale_to_ref=False):
    """
    Build one PSF curve out of several magnitude classes of star.

    THE NORMALIZATION IS THE WHOLE POINT. `radial_continuum_profile` combines
    stars in RAW flux, so with a wide magnitude range its cross-star biweight
    is dominated by (and then rejects) whichever stars happen to be brightest --
    that is not a PSF. Here every star is divided by its own Gaia G flux first:

        p_i(r) = per_obj[i, r] / 10**(-0.4 * G_i)

    For an unsaturated, uncontaminated point source p_i(r) is the SAME function
    for every star regardless of magnitude. So:

      * where the classes lie on top of each other, the PSF is measured and the
        photometry is linear;
      * where a BRIGHT class falls below the fainter ones at small r, that is
        saturation, read straight off the plot;
      * where a FAINT class scatters or turns up at large r, that is its noise /
        contamination floor.

    The class curves therefore ARE the linearity test -- there is no separate
    test to run. You read the valid radius range for each class off
    `plot_piecewise_psf`, pass it back in as `valid_radius`, and the composite
    is the inverse-variance combine of the classes that are valid in each bin.

    profile      : dict from radial_continuum_profile (needs 'per_obj', 'r_mid').
                   BACKGROUND-SUBTRACT UPSTREAM -- a sky error is additive and
                   does not divide out with the Gaia flux, so it breaks the
                   class agreement at large r and would be absorbed into `scale`.
    gmag         : (nobj,) Gaia G, aligned to profile['per_obj'] rows, i.e. to
                   product.catalog order. Carry it through by setting
                   star_tbl['G'] before extraction.
    mag_edges    : class boundaries in G (len = nclass + 1).
    valid_radius : None -> every bin with >= min_stars is used. Otherwise a dict
                   {class_index: (r_min, r_max)} in the profile's radius unit;
                   classes not listed keep their full range.
    min_sn       : a class/bin is only valid at curve/err >= this. A bin a class
                   cannot actually detect must not enter the composite (its error
                   alone would down-weight it, but it would still widen `valid`
                   and pollute the overlap fit). 0 disables.
    ref_class    : index of the class the composite is tied to. None -> the class
                   valid in the most bins, ties broken toward the FAINTER class
                   (the one least likely to be saturated).
    scale_to_ref : if True, fit one multiplicative factor per class on its
                   overlap with ref_class before combining. Leave False first:
                   the UNSCALED agreement is the measurement. Turn it on only to
                   absorb a known throughput/color offset, and check `scale` is
                   near 1 and `overlap_chi2_dof` near 1 before believing it.

    Returns dict:
        r_mid, labels, curves (nclass, nrad), errs, counts, valid (bool),
        scale (nclass,), overlap_chi2_dof (nclass,), n_overlap (nclass,),
        psf (nrad,), psf_err (nrad,), n_class_used (nrad,), ref_class
    """
    per_obj = np.asarray(profile["per_obj"], dtype=float)
    r_mid = np.asarray(profile["r_mid"], dtype=float)
    g = np.asarray(gmag, dtype=float)
    nobj, nrad = per_obj.shape
    if g.size != nobj:
        raise ValueError(f"gmag has {g.size} entries but per_obj has {nobj} rows "
                         "-- gmag must be aligned to product.catalog order.")

    # normalize each star by its own Gaia flux (arbitrary zero point; only the
    # RELATIVE scaling between stars matters)
    p = per_obj / (10.0 ** (-0.4 * g))[:, None]

    edges = np.asarray(mag_edges, dtype=float)
    ncls = edges.size - 1
    cls = np.digitize(g, edges) - 1

    labels = [f"G {edges[c]:.0f}-{edges[c+1]:.0f}" for c in range(ncls)]
    curves = np.full((ncls, nrad), np.nan)
    errs = np.full((ncls, nrad), np.nan)
    counts = np.zeros((ncls, nrad), dtype=int)

    for c in range(ncls):
        sel = (cls == c) & np.isfinite(g)
        if not np.any(sel):
            continue
        blk = p[sel]
        for r in range(nrad):
            col = blk[:, r]
            fin = np.isfinite(col)
            counts[c, r] = int(fin.sum())
            if counts[c, r] >= min_stars:
                curves[c, r] = np.median(col[fin])
                sd = mad_std(col[fin])
                errs[c, r] = sd / np.sqrt(counts[c, r]) if np.isfinite(sd) else np.nan

    valid = (counts >= min_stars) & np.isfinite(curves) & np.isfinite(errs) & (errs > 0)
    if min_sn and min_sn > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            valid &= (curves / errs) >= float(min_sn)
    if valid_radius:
        for c, rng in valid_radius.items():
            ci = int(c)
            lo, hi = float(rng[0]), float(rng[1])
            valid[ci] &= (r_mid >= lo) & (r_mid <= hi)

    if ref_class is None:
        nv = valid.sum(axis=1)
        # The FAINTEST class with a workable number of valid bins. Faintness is
        # the criterion, not bin count: the faintest class is the one that
        # CANNOT be saturated, so it is the only safe anchor. Picking whichever
        # class has the most valid bins would often anchor on the BRIGHTEST
        # class -- exactly the one whose photometry is suspect -- and then the
        # reported `scale` for every other class is measured against a possibly
        # saturated reference and reads backwards.
        good = np.flatnonzero(nv >= max(3, int(0.25 * nv.max()))) if nv.max() else np.array([], int)
        ref = int(good[-1]) if good.size else int(np.argmax(nv))
    else:
        ref = int(ref_class)

    scale = np.ones(ncls)
    chi2_dof = np.full(ncls, np.nan)
    n_ov = np.zeros(ncls, dtype=int)
    for c in range(ncls):
        ov = valid[c] & valid[ref] & (curves[c] > 0) & (curves[ref] > 0)
        n_ov[c] = int(ov.sum())
        if c == ref or n_ov[c] == 0:
            continue
        # one factor, fitted in log space over ALL overlap bins (not a ratio at
        # one radius -- that has no leverage and no goodness of fit)
        lr = np.log(curves[ref][ov] / curves[c][ov])
        lsig = np.sqrt((errs[ref][ov] / curves[ref][ov]) ** 2
                       + (errs[c][ov] / curves[c][ov]) ** 2)
        w = 1.0 / np.where(lsig > 0, lsig ** 2, np.inf)
        k_log = np.sum(w * lr) / np.sum(w)
        scale[c] = float(np.exp(k_log))
        if n_ov[c] > 1:
            chi2_dof[c] = float(np.sum(w * (lr - k_log) ** 2) / (n_ov[c] - 1))

    use_scale = scale if scale_to_ref else np.ones(ncls)
    cur_s = curves * use_scale[:, None]
    err_s = errs * use_scale[:, None]

    psf = np.full(nrad, np.nan)
    psf_err = np.full(nrad, np.nan)
    n_used = np.zeros(nrad, dtype=int)
    for r in range(nrad):
        m = valid[:, r]
        n_used[r] = int(m.sum())
        if n_used[r] == 0:
            continue
        w = 1.0 / err_s[m, r] ** 2
        psf[r] = np.sum(w * cur_s[m, r]) / np.sum(w)
        psf_err[r] = 1.0 / np.sqrt(np.sum(w))

    return {
        "r_edges": np.asarray(profile["r_edges"], dtype=float),
        "r_mid": r_mid, "labels": labels,
        "curves": curves, "errs": errs, "counts": counts, "valid": valid,
        "scale": scale, "overlap_chi2_dof": chi2_dof, "n_overlap": n_ov,
        "psf": psf, "psf_err": psf_err, "n_class_used": n_used,
        "ref_class": ref, "mag_edges": edges, "scaled": bool(scale_to_ref),
    }


def describe_piecewise_psf(pw):
    """One-screen summary: per class, how many stars, over what radius range it
    is valid, and how well it agrees with the reference class."""
    r = pw["r_mid"]
    ref = pw["ref_class"]
    print(f"reference class: {pw['labels'][ref]}"
          + ("   [class scaling APPLIED]" if pw["scaled"] else "   [unscaled]"))
    print(f"{'class':>12s} {'Nmax':>5s} {'valid r':>18s} {'scale':>7s} "
          f"{'chi2/dof':>9s} {'Nov':>4s}")
    for c, lab in enumerate(pw["labels"]):
        v = pw["valid"][c]
        rng = f"{r[v].min():.2f} - {r[v].max():.2f}" if np.any(v) else "none"
        print(f"{lab:>12s} {pw['counts'][c].max():5d} {rng:>18s} "
              f"{pw['scale'][c]:7.3f} {pw['overlap_chi2_dof'][c]:9.2f} "
              f"{pw['n_overlap'][c]:4d}")
    nb = int(np.isfinite(pw["psf"]).sum())
    print(f"\ncomposite: {nb}/{len(r)} bins filled, "
          f"{pw['n_class_used'].max()} classes deep at best")


def plot_piecewise_psf(pw, bin_mode="arcsec", figsize=(11.0, 4.6),
                       title=None, save_fig=False):
    """Left: the Gaia-normalized class curves (agreement = linearity; a bright
    class dropping below the others at small r = saturation). Right: each class
    divided by the composite, which is where you read off the valid ranges."""
    r = pw["r_mid"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(pw["labels"])))

    for c, lab in enumerate(pw["labels"]):
        cur = pw["curves"][c] * (pw["scale"][c] if pw["scaled"] else 1.0)
        err = pw["errs"][c] * (pw["scale"][c] if pw["scaled"] else 1.0)
        ok = np.isfinite(cur)
        if not np.any(ok):
            continue
        ax1.errorbar(r[ok], cur[ok], yerr=err[ok], color=colors[c], marker="o",
                     ms=3.5, lw=1.2, capsize=0, label=f"{lab} (N={pw['counts'][c].max()})")
        v = pw["valid"][c]
        if np.any(v):
            ax1.plot(r[v], cur[v], color=colors[c], marker="o", ms=6,
                     lw=0, mfc="none", mew=1.4)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = cur / pw["psf"]
            rerr = err / pw["psf"]
        ax2.errorbar(r[ok], ratio[ok], yerr=rerr[ok], color=colors[c],
                     marker="o", ms=3.5, lw=1.2, capsize=0, label=lab)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(_radius_label(bin_mode))
    ax1.set_ylabel("SB / Gaia G flux  [arb.]")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("Gaia-normalized class profiles" if title is None else title)

    ok = np.isfinite(pw["psf"])
    ax2.plot(r[ok], pw["psf"][ok] / pw["psf"][ok], "k-", lw=1.0, zorder=0)
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.set_xscale("log")
    ax2.set_ylim(0.0, 2.0)
    ax2.set_xlabel(_radius_label(bin_mode))
    ax2.set_ylabel("class / composite")
    ax2.set_title("residual (open markers on left = bins used)")
    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig if isinstance(save_fig, str) else "piecewise_psf.png",
                    dpi=150, bbox_inches="tight")
    return fig, (ax1, ax2)


# =====================================================================
# 3c. MOFFAT FIT TO THE COMPOSITE PSF
# =====================================================================
def fiber_moffat(r, fwhm, beta=3.0, fiber_radius=0.75, n_s=24, n_phi=48):
    """
    A unit-total-flux Moffat, AVERAGED OVER A FIBER APERTURE of radius
    `fiber_radius`, evaluated at fiber-centre offsets `r`. Units are surface
    brightness (flux per arcsec^2), which is what a fiber measures and what
    radial_continuum_profile returns.

    YOU MUST FIT THIS, NOT THE ANALYTIC MOFFAT. Every point in the measured
    profile is a 1.5"-diameter fiber, so inside ~2" the aperture average is
    substantially flatter than the analytic profile. Fitting the analytic form
    to fiber data biases the recovered FWHM high.

    Same operator as extract.moffat_psf_integration (which the galaxy core
    extraction uses), but vectorised over r and cheap enough for a fit loop:
    polar Gauss-Legendre quadrature over the aperture instead of a 2-D pixel
    grid + LinearNDInterpolator. That returns enclosed FLUX; this returns the
    aperture MEAN, so the two differ by pi*fiber_radius^2 -- a constant that
    the fitted amplitude absorbs.

    Moffat: I(r) = (beta-1)/(pi gamma^2) * [1 + (r/gamma)^2]^(-beta),
            gamma = 0.5*fwhm / sqrt(2^(1/beta) - 1).
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    fwhm = float(fwhm); beta = float(beta)
    if fwhm <= 0 or beta <= 1.0:
        return np.full(r.shape, np.nan)
    gamma = 0.5 * fwhm / np.sqrt(2.0 ** (1.0 / beta) - 1.0)

    xs, ws = np.polynomial.legendre.leggauss(int(n_s))
    s = 0.5 * fiber_radius * (xs + 1.0)
    w_s = 0.5 * fiber_radius * ws
    phi = (np.arange(int(n_phi)) + 0.5) * (2.0 * np.pi / int(n_phi))
    w_phi = 2.0 * np.pi / int(n_phi)

    D = r[:, None, None]; S = s[None, :, None]; PH = phi[None, None, :]
    rr2 = D * D + S * S + 2.0 * D * S * np.cos(PH)
    I = (beta - 1.0) / (np.pi * gamma ** 2) * (1.0 + rr2 / gamma ** 2) ** (-beta)
    integ = np.sum(I * S * w_s[None, :, None] * w_phi, axis=(1, 2))
    return integ / (np.pi * fiber_radius ** 2)


def fit_psf_moffat(pw, *, beta_fixed=None, r_max=None, r_min=None,
                   model="moffat", fiber_radius=0.75, p0=None):
    """
    Fit the composite PSF from build_piecewise_psf with a fiber-integrated
    Moffat. THE FIT IS DONE IN LOG SPACE -- the profile spans decades, so a
    linear-space fit is a fit to the innermost two bins and nothing else.

    model : 'moffat'   -> A * fiber_moffat(r, fwhm, beta).  3 params (2 if
                          beta_fixed). This is the pipeline's own PSF form; the
                          fitted FWHM/beta are directly comparable to what
                          extract.moffat_psf_integration assumes.
            'moffat+pl'-> the above PLUS an aureole A_w * (r/10)^(-p), the
                          standard extended-PSF form (a Moffat falls as
                          r^(-2 beta), far too steep for real scattered light).
                          5 params. DO NOT USE THIS until the wings are actually
                          detected -- with only a handful of significant bins the
                          wing term is unconstrained and will happily fit noise.

    beta_fixed : hold the Moffat index. Pass 3.0 to compare like-for-like with
                 the extraction's assumption; leave None to measure it.
    r_min/r_max: restrict the fitted range (composite bins outside are ignored).

    Returns dict: names, params, errors, cov, chi2, dof, chi2_dof, r, y, yerr,
    model (at the fitted r), r_fine, model_fine, resid (in sigma).
    """
    from scipy.optimize import curve_fit

    r = np.asarray(pw["r_mid"], dtype=float)
    y = np.asarray(pw["psf"], dtype=float)
    e = np.asarray(pw["psf_err"], dtype=float)
    ok = np.isfinite(r) & np.isfinite(y) & np.isfinite(e) & (y > 0) & (e > 0)
    if r_min is not None:
        ok &= r >= float(r_min)
    if r_max is not None:
        ok &= r <= float(r_max)
    r, y, e = r[ok], y[ok], e[ok]
    if r.size < 3:
        raise ValueError(f"only {r.size} usable composite bins -- nothing to fit. "
                         "Check pw['psf'] / widen r_min,r_max.")

    ly = np.log(y)
    lsig = e / y                       # d(log y) = dy / y

    if model == "moffat":
        if beta_fixed is None:
            names = ["log_amp", "fwhm", "beta"]
            def f(rr, la, fw, be):
                return la + np.log(fiber_moffat(rr, fw, be, fiber_radius))
            g0 = [ly[0] + 1.0, 1.5, 3.0]
            bounds = ([-np.inf, 0.2, 1.05], [np.inf, 10.0, 12.0])
        else:
            names = ["log_amp", "fwhm"]
            be0 = float(beta_fixed)
            def f(rr, la, fw):
                return la + np.log(fiber_moffat(rr, fw, be0, fiber_radius))
            g0 = [ly[0] + 1.0, 1.5]
            bounds = ([-np.inf, 0.2], [np.inf, 10.0])
    elif model == "moffat+pl":
        if beta_fixed is None:
            names = ["log_amp", "fwhm", "beta", "log_amp_wing", "p"]
            def f(rr, la, fw, be, lw, p):
                return np.log(np.exp(la) * fiber_moffat(rr, fw, be, fiber_radius)
                              + np.exp(lw) * (rr / 10.0) ** (-p))
            g0 = [ly[0] + 1.0, 1.5, 3.0, ly[-1], 2.0]
            bounds = ([-np.inf, 0.2, 1.05, -np.inf, 0.5],
                      [np.inf, 10.0, 12.0, np.inf, 6.0])
        else:
            names = ["log_amp", "fwhm", "log_amp_wing", "p"]
            be0 = float(beta_fixed)
            def f(rr, la, fw, lw, p):
                return np.log(np.exp(la) * fiber_moffat(rr, fw, be0, fiber_radius)
                              + np.exp(lw) * (rr / 10.0) ** (-p))
            g0 = [ly[0] + 1.0, 1.5, ly[-1], 2.0]
            bounds = ([-np.inf, 0.2, -np.inf, 0.5], [np.inf, 10.0, np.inf, 6.0])
    else:
        raise ValueError("model must be 'moffat' or 'moffat+pl'.")

    if p0 is not None:
        g0 = list(p0)
    popt, pcov = curve_fit(f, r, ly, p0=g0, sigma=lsig, absolute_sigma=True,
                           bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    resid = (ly - f(r, *popt)) / lsig
    chi2 = float(np.sum(resid ** 2))
    dof = int(r.size - len(popt))
    r_fine = np.logspace(np.log10(max(r.min() * 0.7, 1e-3)),
                         np.log10(r.max() * 1.4), 200)

    return {
        "model": model, "names": names, "params": popt, "errors": perr,
        "cov": pcov, "chi2": chi2, "dof": dof,
        "chi2_dof": chi2 / dof if dof > 0 else np.nan,
        "r": r, "y": y, "yerr": e, "resid": resid,
        "model_y": np.exp(f(r, *popt)),
        "r_fine": r_fine, "model_fine": np.exp(f(r_fine, *popt)),
        "beta_fixed": beta_fixed, "fiber_radius": fiber_radius,
    }


def describe_psf_fit(fit):
    print(f"model = {fit['model']}"
          + (f"   (beta fixed at {fit['beta_fixed']})" if fit["beta_fixed"] else ""))
    for n, p, s in zip(fit["names"], fit["params"], fit["errors"]):
        print(f"  {n:>14s} = {p:10.4f} +/- {s:.4f}")
    print(f"  {'chi2/dof':>14s} = {fit['chi2_dof']:10.2f}   "
          f"({fit['chi2']:.1f} / {fit['dof']})")
    w = np.argmax(np.abs(fit["resid"]))
    print(f"  worst bin: r = {fit['r'][w]:.2f}  ({fit['resid'][w]:+.1f} sigma)")


def plot_psf_composite(pw, fit=None, reference=None, bin_mode="arcsec",
                       figsize=(7.6, 6.4), title=None, save_fig=False):
    """
    The composite PSF on its own, optionally with a fit_psf_moffat model and a
    `reference` curve (r, y) -- e.g. the pipeline's assumed Moffat at the
    measured seeing, tied to the composite in the core. Lower panel is the
    residual in sigma.
    """
    r = np.asarray(pw["r_mid"], float)
    y = np.asarray(pw["psf"], float)
    e = np.asarray(pw["psf_err"], float)
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

    ax.errorbar(r[ok], y[ok], yerr=e[ok], fmt="o", ms=4.5, color="k",
                lw=1.1, capsize=0, zorder=3, label="composite PSF")
    if fit is not None:
        lab = fit["model"] + (f", beta={fit['beta_fixed']:.1f} fixed"
                              if fit["beta_fixed"] else "")
        ax.plot(fit["r_fine"], fit["model_fine"], "-", color="crimson", lw=1.6,
                zorder=2, label=f"{lab}  ($\chi^2$/dof={fit['chi2_dof']:.2f})")
    if reference is not None:
        rr, yy = reference
        ax.plot(rr, yy, "--", color="tab:blue", lw=1.5, zorder=1,
                label="pipeline Moffat (measured seeing)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel("SB / Gaia G flux  [arb.]")
    ax.legend(fontsize=9, frameon=False)
    if title:
        ax.set_title(title)

    if fit is not None:
        axr.errorbar(fit["r"], fit["resid"], yerr=1.0, fmt="o", ms=4,
                     color="crimson", lw=1.0, capsize=0)
    if reference is not None:
        rr, yy = reference
        yi = np.interp(r[ok], rr, yy)
        axr.plot(r[ok], (y[ok] - yi) / e[ok], "s", ms=4, color="tab:blue",
                 mfc="none")
    axr.axhline(0, color="k", lw=0.8)
    axr.axhspan(-1, 1, color="0.85", zorder=0)
    axr.set_xscale("log")
    axr.set_ylabel(r"residual [$\sigma$]")
    axr.set_xlabel(_radius_label(bin_mode))
    if save_fig:
        fig.savefig(save_fig if isinstance(save_fig, str) else "psf_composite.png",
                    dpi=150, bbox_inches="tight")
    return fig, (ax, axr)


# =====================================================================
# 3d. ABSOLUTE FLUX-SCALE CHECK  (VDFI star total flux vs Pan-STARRS1 g)
# =====================================================================
# Question this answers: is the stored spectrum per Angstrom or per 2-A pixel
# (i.e. is config.flux_unit_scale = 1e-17 right, or should it be 0.5e-17)?
# More generally: does the WHOLE chain that sets the height of the surface-
# brightness profile -- flux_unit_scale, the average-fiber convention, the
# fiber-area bookkeeping, the background subtraction -- reproduce an
# independently known stellar flux?
#
# Recipe, per star:
#   1. total flux inside R_ap, rebuilt the way the galaxy SB is built:
#          F(lam) = sum_{bins < R_ap}  spec[bin, lam] * scale * A_ann/A_fib
#      spec is the average-fiber value; A_ann/A_fib turns one fiber into the
#      whole annulus (A_fib = pi*(1.5/2)^2, the same area flux_arcsec2 uses).
#   2. aperture correction: divide by the fraction of a point source's light
#      inside R_ap, from the calibrated EFFECTIVE Moffat (psfmodel). Using the
#      effective kernel is right here: it is the fiber-sampled profile, and
#      convolving with the fiber does not change the total flux.
#   3. synthetic PS1 g magnitude of F(lam) (photon-counting AB):
#          m = -2.5 log10[ int F S lam dlam / (c int S/lam dlam) ] - 48.60
#   4. R = 10**(-0.4*(m_syn - g_PS1)).  R ~ 1: per-A, pipeline right.
#      R ~ 2: per-2A pixel, use flux_unit_scale = 0.5e-17.
#
# PS1 g bandpass: Tonry et al. 2012, ApJ 750, 99, Table 3 (photon capture
# cross-section incl. 1.2 airmass), 10-A sampling, copied from the
# astro-sedpy file panstarrs_g.par. Only its SHAPE enters. 0.2% of the
# band's photon weight lies beyond VIRUS's 5540 A red edge; that sliver is
# filled with a flat extrapolation and reported as band_extrap_frac.
_PS1_G_WAVE = np.arange(3890.0, 5671.0, 10.0)
_PS1_G_S = np.array([
    0, 0.001, 0.001, 0.001, 0.002, 0.004, 0.007, 0.012, 0.021, 0.035, 0.053,
    0.073, 0.096, 0.118, 0.135, 0.147, 0.157, 0.163, 0.167, 0.171, 0.175,
    0.181, 0.186, 0.192, 0.199, 0.204, 0.206, 0.208, 0.211, 0.217, 0.225,
    0.233, 0.24, 0.246, 0.249, 0.25, 0.25, 0.253, 0.259, 0.268, 0.277,
    0.286, 0.295, 0.3, 0.302, 0.302, 0.304, 0.309, 0.316, 0.323, 0.33,
    0.336, 0.34, 0.341, 0.342, 0.344, 0.348, 0.354, 0.361, 0.368, 0.374,
    0.38, 0.386, 0.392, 0.396, 0.398, 0.398, 0.394, 0.388, 0.384, 0.382,
    0.384, 0.388, 0.393, 0.396, 0.396, 0.396, 0.397, 0.397, 0.399, 0.4,
    0.401, 0.402, 0.403, 0.405, 0.408, 0.413, 0.418, 0.424, 0.431, 0.437,
    0.441, 0.444, 0.446, 0.448, 0.451, 0.453, 0.456, 0.458, 0.46, 0.46,
    0.459, 0.455, 0.451, 0.45, 0.453, 0.459, 0.463, 0.467, 0.471, 0.472,
    0.471, 0.468, 0.466, 0.463, 0.459, 0.457, 0.456, 0.455, 0.456, 0.456,
    0.457, 0.458, 0.459, 0.461, 0.465, 0.469, 0.473, 0.475, 0.476, 0.475,
    0.473, 0.472, 0.473, 0.473, 0.474, 0.473, 0.472, 0.474, 0.477, 0.483,
    0.489, 0.495, 0.498, 0.5, 0.501, 0.502, 0.502, 0.502, 0.502, 0.502,
    0.501, 0.5, 0.496, 0.493, 0.489, 0.486, 0.481, 0.467, 0.438, 0.39,
    0.323, 0.248, 0.175, 0.113, 0.068, 0.041, 0.025, 0.015, 0.009, 0.006,
    0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0
])
_C_AA = 2.99792458e18          # speed of light [A/s]
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x / 2.x


def _moffat_enclosed(R, fwhm, beta):
    """Fraction of a unit-flux analytic Moffat's light inside radius R (same
    units as fwhm). Closed form: 1 - (1 + (R/gamma)^2)^(1-beta)."""
    fwhm = np.asarray(fwhm, dtype=float)
    gamma = 0.5 * fwhm / np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    return 1.0 - (1.0 + (R / gamma) ** 2) ** (1.0 - beta)


def _ps1_crossmatch(ra, dec, *, match_radius=1.0, neighbour_radius=10.0,
                    neighbour_dmag=3.0, chunk=200, verbose=True):
    """
    Pan-STARRS1 DR2 mean-PSF photometry for each position (VizieR II/349/ps1,
    one cone of neighbour_radius per star, batched). Needs network + astroquery.

    Returns dict of arrays aligned with ra/dec:
      gmag, e_gmag, rmag   nearest PS1 object within match_radius (NaN if none)
      sep                  its separation [arcsec]
      n_bright_neighbour   OTHER PS1 objects between max(match_radius, 1.5")
                           and neighbour_radius with g < g_star + neighbour_dmag
    """
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra = np.atleast_1d(np.asarray(ra, float))
    dec = np.atleast_1d(np.asarray(dec, float))
    n = ra.size
    out = {k: np.full(n, np.nan) for k in ("gmag", "e_gmag", "rmag", "sep")}
    out["n_bright_neighbour"] = np.zeros(n, dtype=int)

    viz = Vizier(columns=["RAJ2000", "DEJ2000", "gmag", "e_gmag", "rmag"],
                 row_limit=-1)
    inner = max(float(match_radius), 1.5)
    for s in tqdm(range(0, n, chunk), disable=not verbose, desc="PS1 query"):
        idx = np.arange(s, min(s + chunk, n))
        tc = SkyCoord(ra[idx] * u.deg, dec[idx] * u.deg)
        res = viz.query_region(tc, radius=neighbour_radius * u.arcsec,
                               catalog="II/349/ps1")
        if len(res) == 0:
            continue
        t = res[0]
        q = (np.asarray(t["_q"], int) - 1) if len(idx) > 1             else np.zeros(len(t), int)
        pc = SkyCoord(np.asarray(t["RAJ2000"], float) * u.deg,
                      np.asarray(t["DEJ2000"], float) * u.deg)
        g = np.ma.filled(np.ma.asarray(t["gmag"], float), np.nan)
        eg = np.ma.filled(np.ma.asarray(t["e_gmag"], float), np.nan)
        rr = np.ma.filled(np.ma.asarray(t["rmag"], float), np.nan)
        for j, i in enumerate(idx):
            m = q == j
            if not np.any(m):
                continue
            sep = tc[j].separation(pc[m]).arcsec
            k = int(np.argmin(sep))
            if sep[k] > match_radius:
                continue
            gm = g[m]
            out["gmag"][i], out["e_gmag"][i] = gm[k], eg[m][k]
            out["rmag"][i], out["sep"][i] = rr[m][k], sep[k]
            if np.isfinite(gm[k]):
                nb = (sep > inner) & np.isfinite(gm) & (gm < gm[k] + neighbour_dmag)
                out["n_bright_neighbour"][i] = int(nb.sum())
    return out


def _synthetic_ps1_g(wave, flam, ferr):
    """
    Synthetic PS1 g AB magnitude of each row of flam (nobj, nwave) [erg/s/cm2/A].

    NaN pixels inside the band are interpolated over; their share of the
    band's photon weight is returned as bad_frac so the caller can reject.
    Beyond the last finite pixel the spectrum is held flat at the median of
    its last 100 A (extrap_frac says how much of the band that covers).
    Returns m, m_err (noise only, pixels treated as independent), bad_frac,
    extrap_frac.
    """
    wave = np.asarray(wave, float)
    grid = np.arange(_PS1_G_WAVE[0], _PS1_G_WAVE[-1] + 0.5, 1.0)
    S = np.interp(grid, _PS1_G_WAVE, _PS1_G_S)
    wgt = S * grid                                     # photon weight / A
    denom = _C_AA * _trapz(S / grid, grid)
    dlam = np.gradient(wave)
    S_nat = np.interp(wave, _PS1_G_WAVE, _PS1_G_S, left=0.0, right=0.0)

    nobj = flam.shape[0]
    m = np.full(nobj, np.nan); me = np.full(nobj, np.nan)
    bad = np.ones(nobj); ext = np.full(nobj, np.nan)
    for i in range(nobj):
        f, e = flam[i], ferr[i]
        fin = np.isfinite(f)
        if fin.sum() < 10:
            continue
        in_band = S_nat > 0
        w_nat = S_nat * wave * dlam
        bad[i] = w_nat[in_band & ~fin].sum() / w_nat[in_band].sum()
        wmax = wave[fin].max()
        tail = fin & (wave > wmax - 100.0)
        f_ext = np.nanmedian(f[tail])
        fg = np.interp(grid, wave[fin], f[fin], right=f_ext)
        num = _trapz(fg * wgt, grid)
        ext[i] = wgt[grid > wmax].sum() / wgt.sum()
        if not num > 0:
            continue
        m[i] = -2.5 * np.log10(num / denom) - 48.60
        ev = np.where(np.isfinite(e) & fin, e, 0.0) * w_nat
        me[i] = 1.0857 * np.sqrt(np.sum(ev ** 2)) / num
    return m, me, bad, ext


def flux_scale_check(star_product, cal=None, *, fwhm=None, beta=None,
                     r_ap=6.0, r_ap_list=(3.0, 4.5, 6.0, 9.0), ref=None,
                     mag_col="G", mag_range=(15.0, 19.0),
                     match_radius=1.0, neighbour_radius=10.0,
                     neighbour_dmag=3.0, flux_unit_scale=1e-17,
                     fiber_diam_arcsec=1.5, use_nobg=False, max_bad_frac=0.10,
                     field_col="FIELD", n_boot=2000, seed=0, plot=True,
                     figsize=(10.5, 8.0), tick_direction="in", tick_right=True,
                     save_fig=False, verbose=True):
    """
    Absolute flux-scale check: VDFI star total flux vs Pan-STARRS1 g, star by
    star. Tells per-A (R~1) from per-2A-pixel (R~2) and, more generally,
    tests every step that sets the height of the SB profile. Method in the
    section header above.

    star_product : the arcsec-binned, mask='none' star GalaxyProduct (bins
                   must start at 0 and reach r_ap). catalog needs RA, DEC.
    cal          : psfmodel.PSFCalibration (FWHM(lam) and beta for the
                   aperture correction). Or pass cal=None with scalar
                   fwhm= [arcsec, EFFECTIVE] and beta=.
    r_ap         : fiducial aperture [arcsec]; snapped to the nearest bin
                   edge. r_ap_list: apertures for the stability diagnostic --
                   if R drifts with aperture, the aperture correction or the
                   area bookkeeping is wrong, not the unit.
    ref          : None -> query PS1 (VizieR, needs network + astroquery).
                   Else a dict/Table aligned with the catalog rows holding
                   'gmag', 'e_gmag', and optionally 'rmag',
                   'n_bright_neighbour' (so you can query once and rerun
                   offline: pass res['ref'] back in).
    mag_col      : catalog column for the brightness cut (Gaia G). Falls back
                   to PS1 g if absent. mag_range: (bright, faint) limits.
    flux_unit_scale : the value under test -- leave at the pipeline's 1e-17.
    use_nobg     : use spec_nobg (no background subtraction) instead of spec.

    Returns dict: R, R_err, m_syn, g_ps1, mag, gr, field, sel, reasons,
    R_med, R_med_err (bootstrap over stars), R_biweight, R_by_aperture,
    R_by_field, slope_dlogR_dmag, r_ap_used, cog_fid (per wavelength),
    band_extrap_frac, dlam, ref, fig, axes.
    """

    cat = star_product.catalog
    spec = star_product.spec_nobg if use_nobg else star_product.spec
    if spec is None:
        raise ValueError("use_nobg=True but star_product.spec_nobg is None "
                         "(extract with store_spec_nobg=True).")
    spec = np.asarray(spec, float)
    err = np.asarray(star_product.err, float)
    wave = np.asarray(star_product.wave, float)
    edges = np.asarray(star_product.r_edges, float)
    nobj = spec.shape[0]
    if edges[0] != 0:
        raise ValueError("r_edges must start at 0 (need the full aperture).")
    if edges[-1] < r_ap:
        raise ValueError(f"bins stop at {edges[-1]:g}, inside r_ap={r_ap:g}.")

    # --- PSF for the aperture correction ---
    if cal is not None:
        fwhm_lam = np.asarray(cal.fwhm(wave), float)
        beta_use = float(cal.beta if beta is None else beta)
    else:
        if fwhm is None or beta is None:
            raise ValueError("pass cal=PSFCalibration, or both fwhm= and beta=.")
        fwhm_lam = np.full(wave.shape, float(fwhm))
        beta_use = float(beta)

    a_fib = np.pi * (0.5 * fiber_diam_arcsec) ** 2
    a_ann = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    dlam = float(np.nanmedian(np.diff(wave)))

    def _total(R_req):
        k = int(np.argmin(np.abs(edges - R_req)))
        k = max(k, 1)
        fac = (a_ann[:k] / a_fib)[None, :, None]
        f = np.sum(spec[:, :k, :] * fac, axis=1) * flux_unit_scale   # NaN-propagating
        e = np.sqrt(np.sum((err[:, :k, :] * fac) ** 2, axis=1)) * flux_unit_scale
        cog = _moffat_enclosed(edges[k], fwhm_lam, beta_use)
        return edges[k], f / cog, e / cog, cog

    # --- reference photometry ---
    if ref is None:
        if verbose:
            print(f"querying PS1 DR2 (VizieR II/349/ps1) for {nobj} stars ...")
        ref = _ps1_crossmatch(cat["RA"], cat["DEC"], match_radius=match_radius,
                              neighbour_radius=neighbour_radius,
                              neighbour_dmag=neighbour_dmag, verbose=verbose)
    g_ps1 = np.asarray(ref["gmag"], float)
    e_ps1 = np.asarray(ref["e_gmag"], float)
    r_ps1 = np.asarray(ref["rmag"], float) if "rmag" in ref else np.full(nobj, np.nan)
    nnb = (np.asarray(ref["n_bright_neighbour"], int)
           if "n_bright_neighbour" in ref else np.zeros(nobj, int))
    if g_ps1.size != nobj:
        raise ValueError("ref is not aligned with the star catalog.")

    if mag_col in cat.colnames:
        mag = np.asarray(cat[mag_col], float); mag_name = mag_col
    else:
        mag = g_ps1; mag_name = "PS1 g"
    field = (np.asarray(cat[field_col]).astype(str) if field_col in cat.colnames
             else np.full(nobj, "all"))

    # --- fiducial aperture ---
    r_used, F, Fe, cog_fid = _total(r_ap)
    m_syn, m_err, bad_frac, ext_frac = _synthetic_ps1_g(wave, F, Fe)
    R = 10 ** (-0.4 * (m_syn - g_ps1))
    R_err = R * 0.9210 * np.hypot(m_err, np.where(np.isfinite(e_ps1), e_ps1, 0.0))

    # --- selection, with a reason for every rejection (first failing cut) ---
    cuts = OrderedDict([
        ("no PS1 g match", ~np.isfinite(g_ps1)),
        (f"{mag_name} outside {mag_range}", ~((mag >= mag_range[0]) & (mag <= mag_range[1]))),
        (f"PS1 neighbour < +{neighbour_dmag:g} mag within {neighbour_radius:g}\"", nnb > 0),
        (f"> {max_bad_frac:.0%} of band missing inside {r_used:g}\"",
         ~(bad_frac <= max_bad_frac)),
        ("non-positive synthetic flux", ~np.isfinite(R)),
    ])
    sel = np.ones(nobj, bool); reasons = OrderedDict()
    for name, fail in cuts.items():
        reasons[name] = int(np.sum(sel & fail))
        sel &= ~fail

    rng = np.random.default_rng(seed)

    def _med_boot(x):
        x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan, np.nan
        bs = np.median(rng.choice(x, size=(n_boot, x.size)), axis=1)
        return float(np.median(x)), float(np.std(bs))

    R_med, R_med_err = _med_boot(R[sel])
    R_bw = float(biweight_location(R[sel])) if sel.sum() >= 3 else np.nan

    R_by_ap = []
    for Rq in r_ap_list:
        ru, Fq, Feq, _ = _total(Rq)
        mq, _, bq, _ = _synthetic_ps1_g(wave, Fq, Feq)
        Rq_star = 10 ** (-0.4 * (mq - g_ps1))
        ok = sel & (bq <= max_bad_frac)
        med, me = _med_boot(Rq_star[ok])
        R_by_ap.append({"r_ap": ru, "R_med": med, "R_med_err": me,
                        "n": int(ok.sum()),
                        "cog": float(np.nanmedian(_moffat_enclosed(ru, fwhm_lam, beta_use)))})

    R_by_field = OrderedDict()
    for fld in np.unique(field[sel]):
        med, me = _med_boot(R[sel & (field == fld)])
        R_by_field[fld] = {"R_med": med, "R_med_err": me,
                           "n": int(np.sum(sel & (field == fld)))}

    slope = (np.nan, np.nan)
    ok = sel & np.isfinite(mag) & (R > 0)
    if ok.sum() >= 5:
        p, cov = np.polyfit(mag[ok] - np.median(mag[ok]), np.log10(R[ok]), 1, cov=True)
        slope = (float(p[0]), float(np.sqrt(cov[0, 0])))
    gr = g_ps1 - r_ps1

    res = {"R": R, "R_err": R_err, "m_syn": m_syn, "g_ps1": g_ps1, "mag": mag,
           "mag_name": mag_name, "gr": gr, "field": field, "sel": sel,
           "reasons": reasons, "R_med": R_med, "R_med_err": R_med_err,
           "R_biweight": R_bw, "R_by_aperture": R_by_ap,
           "R_by_field": R_by_field, "slope_dlogR_dmag": slope,
           "r_ap_used": r_used, "cog_fid": cog_fid, "beta": beta_use,
           "band_extrap_frac": float(np.nanmedian(ext_frac)), "dlam": dlam,
           "flux_unit_scale": flux_unit_scale, "ref": ref,
           "fig": None, "axes": None}

    if verbose:
        _describe_flux_scale(res, nobj)
    if plot:
        res["fig"], res["axes"] = _plot_flux_scale(
            res, figsize=figsize, tick_direction=tick_direction,
            tick_right=tick_right, save_fig=save_fig)
    return res


def _describe_flux_scale(res, nobj):
    print(f"flux-scale check  (flux_unit_scale = {res['flux_unit_scale']:g}, "
          f"wavelength step = {res['dlam']:g} A)")
    print(f"  aperture {res['r_ap_used']:g}\"  -> median CoG = "
          f"{np.nanmedian(res['cog_fid']):.4f}  (beta = {res['beta']:.3f})")
    print(f"  PS1 g band beyond last VIRUS pixel: {res['band_extrap_frac']:.2%} "
          f"of photon weight (flat extrapolation)")
    print(f"  stars: {nobj} in, {int(res['sel'].sum())} kept")
    for k, v in res["reasons"].items():
        if v:
            print(f"     - {v:4d}  {k}")
    print(f"  R = F_VDFI / F_PS1 :  median {res['R_med']:.3f} +/- "
          f"{res['R_med_err']:.3f}   (biweight {res['R_biweight']:.3f})")
    if np.isfinite(res["R_med"]):
        l2 = np.log2(res["R_med"])
        near = "1  (per A: pipeline scale is right)" if abs(l2) < abs(l2 - 1)             else "2  (per 2-A pixel: flux_unit_scale should be 0.5e-17)"
        print(f"  -> closer to {near}")
    print("  by aperture   : " + "   ".join(
        f"{d['r_ap']:g}\": {d['R_med']:.3f}+/-{d['R_med_err']:.3f}"
        for d in res["R_by_aperture"]))
    print("  by field      : " + "   ".join(
        f"{k}: {v['R_med']:.3f}+/-{v['R_med_err']:.3f} (N={v['n']})"
        for k, v in res["R_by_field"].items()))
    s, se = res["slope_dlogR_dmag"]
    print(f"  linearity     : dlog10(R)/d({res['mag_name']}) = {s:+.4f} +/- {se:.4f}"
          " per mag  (0 = linear)")


def _plot_flux_scale(res, figsize, tick_direction, tick_right, save_fig):
    from .formatting import apply_tick_style

    sel = res["sel"]; R = res["R"]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (a0, a1), (a2, a3) = axes

    lr = np.log2(R[sel & (R > 0)])
    if lr.size:
        lo, hi = min(-0.6, np.nanmin(lr)), max(1.6, np.nanmax(lr))
        a0.hist(lr, bins=np.linspace(lo, hi, 30), color="0.55", edgecolor="k", lw=0.5)
    for x, lab, c in ((0, "per Å", "tab:blue"), (1, "per 2 Å", "tab:red")):
        a0.axvline(x, color=c, ls="--", lw=1.4, label=lab)
    if np.isfinite(res["R_med"]):
        a0.axvline(np.log2(res["R_med"]), color="k", lw=1.6, label="median")
    a0.set_xlabel(r"$\log_2 R$"); a0.set_ylabel("N stars")
    a0.legend(frameon=False, fontsize=9)

    fields = list(res["R_by_field"].keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, fld in enumerate(fields):
        m = sel & (res["field"] == fld)
        kw = dict(fmt="o", ms=3.5, lw=0.6, alpha=0.8, color=colors[j % len(colors)],
                  label=f"{fld} (N={m.sum()})")
        a1.errorbar(res["mag"][m], R[m], yerr=res["R_err"][m], **kw)
        a2.errorbar(res["gr"][m], R[m], yerr=res["R_err"][m], **kw)
    for ax, xl in ((a1, res["mag_name"]), (a2, "PS1 g - r")):
        ax.axhline(1, color="tab:blue", ls="--", lw=1.2)
        ax.axhline(2, color="tab:red", ls="--", lw=1.2)
        ax.set_xlabel(xl); ax.set_ylabel(r"$R = F_{\rm VDFI}/F_{\rm PS1}$")
        ax.legend(frameon=False, fontsize=8)

    ap = res["R_by_aperture"]
    x = [d["r_ap"] for d in ap]
    a3.errorbar(x, [d["R_med"] for d in ap], yerr=[d["R_med_err"] for d in ap],
                fmt="ko-", ms=5, capsize=3, label="median R")
    a3.axhline(1, color="tab:blue", ls="--", lw=1.2)
    a3.axhline(2, color="tab:red", ls="--", lw=1.2)
    a3.set_xlabel("aperture radius [arcsec]"); a3.set_ylabel("median R")
    a3b = a3.twinx()
    a3b.plot(x, [d["cog"] for d in ap], "s:", color="0.5", ms=4)
    a3b.set_ylabel("aperture correction (CoG)", color="0.4")
    a3.legend(frameon=False, fontsize=9, loc="upper left")

    if tick_direction:
        apply_tick_style([a0, a1, a2], direction=tick_direction, right=tick_right)
        apply_tick_style(a3, direction=tick_direction, right=False)
        a3b.tick_params(which="both", direction=tick_direction)
    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig if isinstance(save_fig, str) else "flux_scale_check.png",
                    dpi=150, bbox_inches="tight")
    return fig, axes


# =====================================================================
# 4. PLOTS
# =====================================================================
def _radius_label(bin_mode):
    bm = (bin_mode or "").lower()
    if bm in ("virial", "vr"):
        return r"$R / R_{\rm vir}$"
    if bm == "kpc":
        return "Radius [kpc]"
    if bm == "arcsec":
        return "Radius [arcsec]"
    return "Radius"


def plot_radial_profile(profiles, bin_mode="arcsec", logy=True, normalize=False,
                        figsize=(7.2, 4.8), title=None, save_fig=False):
    """
    Overlay continuum SB vs radius (log-log) for one or more labelled profiles,
    e.g. {'star PSF': p_star, 'galaxy continuum': p_gal}. normalize=True scales
    each profile to its innermost finite bin (compare SHAPES / PSF wings).
    """
    if isinstance(profiles, dict) and "sb" in profiles:
        profiles = {"profile": profiles}
    fig, ax = plt.subplots(figsize=figsize)
    for label, p in profiles.items():
        r_mid = np.asarray(p["r_mid"], dtype=float)
        sb = np.asarray(p["sb"], dtype=float).copy()
        err = np.asarray(p["sb_err"], dtype=float).copy()
        if normalize:
            fin = np.isfinite(sb)
            norm = sb[fin][0] if np.any(fin) else np.nan
            sb = sb / norm; err = err / norm
        ax.errorbar(r_mid, sb, yerr=err, fmt="o-", capsize=3, ms=5, lw=1.3, label=label)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(_radius_label(bin_mode))
    ax.set_ylabel("Continuum SB" + (" (norm.)" if normalize else " [per fiber]"))
    ax.set_title(title or "Radial continuum profile")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2, which="both")
    if save_fig:
        plt.savefig("Star_profile_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_curve_of_growth(cogs, bin_mode="arcsec", figsize=(7.2, 4.8),
                         title=None, save_fig=False):
    """Overlay normalized curve(s) of growth, {'star PSF': cog_star, ...}."""
    if isinstance(cogs, dict) and "cog_norm" in cogs:
        cogs = {"curve": cogs}
    fig, ax = plt.subplots(figsize=figsize)
    for label, c in cogs.items():
        ax.plot(np.asarray(c["r_mid"], dtype=float), np.asarray(c["cog_norm"]),
                "o-", ms=5, lw=1.4, label=label)
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(_radius_label(bin_mode))
    ax.set_ylabel("Enclosed flux (normalized)")
    ax.set_ylim(0, 1.08)
    ax.set_title(title or "Curve of growth")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    if save_fig:
        plt.savefig("Figure_curve_of_growth.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# 5. BLANK-SKY CENTROID NULL  (the new validation function)
# =====================================================================
def _centroids_per_bin(stack_flux, rest_wave, bounds, lya, cont_kw):
    nrad = stack_flux.shape[0]
    cv = np.full(nrad, np.nan)
    for r in range(nrad):
        res = flux_weighted_centroid(rest_wave, stack_flux[r], bounds=bounds,
                                     lya_center=lya, **cont_kw)
        cv[r] = res["centroid_vel_kms"]
    return cv


def blank_sky_centroid_null(config, sky_product: GalaxyProduct, method=None,
                            bounds=None, real_boot=None, verbose=True) -> dict:
    """
    Stack a blank-sky product (from run_positions_extract(..., mask='auto') on
    make_random_sky_table) and measure the Lya centroid per bin. On source-free
    sky the continuum should be flat and the centroid undefined/scattered about
    zero -- any coherent, radius-dependent centroid here is an ADDITIVE
    sky/background-residual artifact, not CGM emission.

    Cheap and re-runnable: the expensive part was the extraction. Pass real_boot
    (run_measure on the real sample) to overlay the real centroid for context.

    Returns dict: sky_v (nrad,), z_median, r_edges, bin_mode, VR_biweight_v,
                  real_boot, bounds, method.
    """
    sm = method or "biweight"
    lya = float(config.LYA_REST)
    bounds = tuple(config.line_window) if bounds is None else tuple(bounds)
    cont_kw = dict(
        cont_bounds=getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS),
        cont_method=getattr(config, "cont_method", "poly"),
        cont_order=getattr(config, "cont_order", 1),
    )
    prod = apply_finite_cut(sky_product, config.cut_radial_bin,
                            config.min_good_wave, verbose=verbose)
    st = build_stacks(config, prod, keep_cube=False)
    if sm not in st["stacks"]:
        raise KeyError(f"method {sm!r} not in stacks (have {list(st['stacks'])}).")
    sky_v = _centroids_per_bin(st["stacks"][sm]["flux"],
                               np.asarray(st["rest_wave"]), bounds, lya, cont_kw)
    if verbose:
        edges = np.round(np.asarray(st["r_edges"]), 3)
        print(f"\nblank-sky null (method={sm}, {prod.ngal} positions):")
        for r in range(sky_v.size):
            print(f"  {edges[r]:g}-{edges[r+1]:g}: sky centroid = {sky_v[r]:+.1f} km/s")
    return {
        "sky_v": sky_v,
        "z_median": st.get("z_median"),
        "r_edges": np.asarray(st["r_edges"]),
        "bin_mode": st.get("bin_mode"),
        "VR_biweight_v": st.get("VR_biweight_v"),
        "real_boot": real_boot, "bounds": bounds, "method": sm,
    }


def plot_blank_sky_null(result, vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
                        figsize=(7.4, 4.8), ylims=(-200, 200), xlims=None,
                        title=None, save_fig=False):
    """Blank-sky centroid (should hug zero) with the real centroid overlaid."""
    bin_mode = _resolve_bin_mode(result.get("bin_mode"), None)
    VR_biweight_v = _get_vr_biweight_v(result.get("VR_biweight_v"), None)
    r_edges = np.asarray(result["r_edges"])
    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, r_edges, bin_mode, VR_biweight_v,
                                     None, vr_ticks, xlims)
    ax.axhline(0, color="tomato", alpha=0.4, lw=1)
    ax.plot(r_mid, result["sky_v"], "s--", color="0.4", ms=6, lw=1.3,
            label="blank-sky centroid")
    rb = result.get("real_boot")
    if rb is not None and "centroid_v_med" in rb:
        v = np.asarray(rb["centroid_v_med"])
        lo = np.asarray(rb["centroid_v_lo"]); hi = np.asarray(rb["centroid_v_hi"])
        ax.errorbar(r_mid, v, xerr=xerr, yerr=np.vstack([v - lo, hi - v]),
                    fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:blue",
                    label="real Lyα centroid")
    ax.set_ylim(ylims)
    ax.set_ylabel(r"Centroid velocity [km s$^{-1}$]")
    ax.set_title(title or "Blank-sky null vs. real Lyα centroid")
    ax.legend(frameon=False, fontsize=9, loc="best")
    if save_fig:
        plt.savefig("Figure_blank_sky_null.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# 6. LINE-PROFILE TRACER TEST  (LIS / UV lines measured EXACTLY as Lyα)
# =====================================================================
# ---------------------------------------------------------------------
# WHAT THIS IS FOR
# ---------------------------------------------------------------------
# The headline worry about an extended Lyα halo is that the *pipeline*
# (extraction → galaxy-axis coadd → (1+z) rest-framing → sideband continuum →
# windowed integration) might manufacture spatial extension for ANY feature,
# so the "halo" would be an artifact rather than CGM emission.
#
# LIS (low-ionization-state) interstellar lines are the control. They arise in
# the ISM of the COMPACT central galaxy (absorption against the stellar
# continuum, plus the Si II* fluorescent companions), so they are spatially
# unresolved — a point source. Measured with the IDENTICAL estimator as Lyα,
# their radial profile must therefore trace the stellar PSF. If it does, the
# pipeline provably does not invent extension, so whatever EXTRA extension Lyα
# shows beyond the PSF is physical.
#
# "Identical estimator" is enforced literally: per line, per radial bin, we call
# the SAME measure.integrated_line_flux / flux_weighted_centroid that the Lyα
# analysis uses, with the line window (half-width = ½·config.line_window) and the
# continuum sidebands (config.cont_bounds, stored as offsets from Lyα) simply
# RE-CENTRED on the line. This is the placebo_wavelengths geometry, only the
# centres are real lines instead of line-free continuum.
#
# The same function runs on GALAXY stacks (real lines) and on STAR stacks
# (where every "line" should come back flat → an estimator null).

# Canonical LIS interstellar lines — the low-ionization subset of
# validation.DEFAULT_UV_LINES. (name, lambda_rest_A, kind). 'abs' lines absorb
# against the central continuum; the Si II* lines are fluorescent emission.
LIS_LINES = [
    ("Si II",     1260.42, "abs"),
    ("Si II*",    1264.74, "em"),    # fluorescent companion of Si II 1260
    ("O I+Si II", 1303.27, "abs"),   # 1302.17 / 1304.37 blend
    ("C II",      1334.53, "abs"),
    ("Si II*",    1335.71, "em"),    # fluorescent companion of C II 1334 (optional)
    ("Si II",     1526.71, "abs"),
    ("Fe II",     1608.45, "abs"),
    ("Al II",     1670.79, "abs"),
]


def _cont_kwargs(config):
    """Continuum model settings read from config (falls back to the shared
    DEFAULT_* so this tracks exactly what the real Lyα run uses)."""
    return dict(
        cont_method=getattr(config, "cont_method", DEFAULT_CONT_METHOD),
        cont_order=getattr(config, "cont_order", DEFAULT_CONT_ORDER),
    )


def _recentre_geometry(config, line_rest):
    """Re-centre the Lyα measurement geometry on `line_rest`.

    Returns (bounds, cont_bounds, hw):
      bounds      : (line-hw, line+hw), hw = ½·config.line_window width.
      cont_bounds : config.cont_bounds shifted by (line_rest - LYA_REST), so the
                    blue/red sidebands keep the SAME offsets they have for Lyα.
    """
    lya = float(config.LYA_REST)
    lw = config.line_window
    hw = 0.5 * (float(lw[1]) - float(lw[0]))
    cb0 = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
    offs = [(lo - lya, hi - lya) for lo, hi in cb0]
    bounds = (line_rest - hw, line_rest + hw)
    cont_bounds = tuple((line_rest + o0, line_rest + o1) for (o0, o1) in offs)
    return bounds, cont_bounds, hw


def _footprint_in_grid(rest_wave, bounds, cont_bounds, pad=1.0):
    """True iff the line window AND both sidebands fall inside the rest grid."""
    lo = min([bounds[0]] + [b[0] for b in cont_bounds]) - pad
    hi = max([bounds[1]] + [b[1] for b in cont_bounds]) + pad
    w = np.asarray(rest_wave, dtype=float)
    return (lo >= np.nanmin(w)) and (hi <= np.nanmax(w))


def _normalize_to_inner(profile):
    """Divide a (nrad,) profile by its first finite, nonzero bin (sign kept), so
    an absorption line (negative) and an emission line (positive) both come back
    as 1.0 at the centre and fall toward 0 outward if they trace a point source.
    Returns (norm_profile, norm_value)."""
    p = np.asarray(profile, dtype=float)
    fin = np.isfinite(p) & (p != 0)
    if not np.any(fin):
        return np.full_like(p, np.nan), np.nan
    v0 = p[fin][0]
    return p / v0, float(v0)


# ---------------------------------------------------------------------
# 6.1  MEASURE  (galaxies OR stars; identical to the Lyα estimator)
# ---------------------------------------------------------------------
def measure_line_profiles(config, stacks, lines=None, method=None,
                          product=None, clip_negative_flux=False,
                          drop_out_of_grid=True, verbose=True):
    """
    Per-line radial profile of the integrated line flux (and centroid), measured
    on a Stage-2 `stacks` dict EXACTLY as Lyα — only the window/sidebands are
    re-centred on each line (see _recentre_geometry).

    Works unchanged on galaxy stacks (real lines) and on star stacks (null).

    Parameters
    ----------
    stacks   : build_stacks(...) output (needs 'rest_wave', 'stacks', 'r_edges').
    lines    : list of (name, lambda_rest_A, kind). None → LIS_LINES.
    method   : galaxy-combine in stacks['stacks'] to read (default 'biweight').
    product  : the SAME GalaxyProduct the stacks were built from. If given, its
               median fibers/bin (∝ annulus area) weight a proper curve of growth
               per line; if None, only the normalized differential profile is
               returned (still fine for the shape/PSF comparison).
    drop_out_of_grid : silently skip (with a collected warning) any line whose
               window+sidebands fall outside config.rest_wave_min..max. With the
               DEFAULT 1140–1300 Å grid only Si II 1260 / Si II* 1264 survive —
               widen rest_wave_max (≳1700) and rebuild the stacks for the full
               LIS suite.

    Returns dict:
        lines_used   list[(name, lam, kind)]
        per_line     {key: {name, lam, kind, flux (nrad,), centroid_v (nrad,),
                            norm_profile (nrad,), norm_value, cog_norm (nrad,)|None}}
        r_edges, r_mid, bin_mode
        comp_norm_mean / comp_norm_med (nrad,)   composite over lines
        comp_norm_lo / comp_norm_hi    (nrad,)   16/84 line-to-line scatter
        method, hw, dropped
    """
    sm = method or "biweight"
    if sm not in stacks["stacks"]:
        raise KeyError(f"method {sm!r} not in stacks (have {list(stacks['stacks'])}).")
    lines = list(LIS_LINES if lines is None else lines)
    rest_wave = np.asarray(stacks["rest_wave"], dtype=float)
    flux_cube = np.asarray(stacks["stacks"][sm]["flux"], dtype=float)   # (nrad, nwave)
    r_edges = np.asarray(stacks["r_edges"], dtype=float)
    nrad = flux_cube.shape[0]
    cont_kw = _cont_kwargs(config)

    # nfib weight (∝ area) for an honest curve of growth, if the product is given
    nfib_med = None
    if product is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nfib_med = np.nanmedian(np.asarray(product.nfib, dtype=float), axis=0)

    per_line, dropped = OrderedDict(), []
    for (name, lam, kind) in lines:
        bounds, cont_bounds, hw = _recentre_geometry(config, lam)
        if drop_out_of_grid and not _footprint_in_grid(rest_wave, bounds, cont_bounds):
            dropped.append((name, lam))
            continue

        flux = np.full(nrad, np.nan)
        cen = np.full(nrad, np.nan)
        for r in range(nrad):
            fr = integrated_line_flux(rest_wave, flux_cube[r], bounds=bounds,
                                      cont_bounds=cont_bounds, lya_center=lam,
                                      clip_negative=clip_negative_flux, **cont_kw)
            flux[r] = fr["flux_sum"]
            cr = flux_weighted_centroid(rest_wave, flux_cube[r], bounds=bounds,
                                        cont_bounds=cont_bounds, lya_center=lam,
                                        **cont_kw)
            cen[r] = cr["centroid_vel_kms"]

        norm_profile, norm_value = _normalize_to_inner(flux)

        cog_norm = None
        if nfib_med is not None:
            contrib = flux * nfib_med
            contrib = np.where(np.isfinite(contrib), contrib, 0.0)
            cog = np.cumsum(contrib)
            tot = cog[-1] if cog[-1] != 0 else np.nan
            cog_norm = cog / tot

        key = f"{name}_{lam:.0f}"
        per_line[key] = {"name": name, "lam": float(lam), "kind": kind,
                         "flux": flux, "centroid_v": cen,
                         "norm_profile": norm_profile, "norm_value": norm_value,
                         "cog_norm": cog_norm}

    # composite across lines (of the normalized differential profiles)
    if per_line:
        stack_norm = np.vstack([d["norm_profile"] for d in per_line.values()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            comp_mean = np.nanmean(stack_norm, axis=0)
            comp_med = np.nanmedian(stack_norm, axis=0)
            comp_lo = np.nanpercentile(stack_norm, 16, axis=0)
            comp_hi = np.nanpercentile(stack_norm, 84, axis=0)
    else:
        comp_mean = comp_med = comp_lo = comp_hi = np.full(nrad, np.nan)

    if verbose:
        print(f"line profiles (method={sm}, ±{hw:.1f} Å windows, "
              f"{len(per_line)} lines):")
        for key, d in per_line.items():
            with np.printoptions(precision=2, suppress=True):
                print(f"  {d['name']:<9} {d['lam']:7.2f}  norm profile = "
                      f"{np.array2string(d['norm_profile'], precision=2)}")
        if dropped:
            names = ", ".join(f"{n} {l:.0f}" for n, l in dropped)
            print(f"  [skipped, off the rest grid: {names}; widen "
                  f"rest_wave_min/max and rebuild stacks]")

    return {
        "lines_used": [(d["name"], d["lam"], d["kind"]) for d in per_line.values()],
        "per_line": per_line,
        "r_edges": r_edges,
        "r_mid": _bin_midpoints(r_edges),
        "bin_mode": stacks.get("bin_mode"),
        "comp_norm_mean": comp_mean, "comp_norm_med": comp_med,
        "comp_norm_lo": comp_lo, "comp_norm_hi": comp_hi,
        "method": sm, "hw": hw, "dropped": dropped,
    }


# =====================================================================
# 6.1b  BOOTSTRAPPED line profiles  (galaxy resampling; per-line S/N)
# =====================================================================
def bootstrap_line_profiles(config, stacks, lines=None, method="biweight",
                            product=None, nboot=500, seed=1,
                            clip_negative_flux=False, drop_out_of_grid=True,
                            measure_centroid_too=True,
                            rank_by="central", select_top=None, min_sn=None,
                            verbose=True):
    """
    Galaxy-resampling bootstrap of the LIS line profiles -- the measurement-error
    version of `measure_line_profiles`, with a per-line signal-to-noise.

    Each draw resamples galaxies WITH REPLACEMENT, re-stacks the whole cube once,
    then re-measures every kept line in every radial bin with the identical
    estimator the Lya analysis uses. The point estimate ('fiducial') is taken on
    the full-sample stack; the error per line per bin is the 16/84 spread across
    draws. This is `measure.bootstrap_measurements` applied to the LIS suite.

    REQUIRES the per-galaxy cube: build the stacks with
        stacks = build_stacks(config, product, keep_cube=True)
    so that stacks['cube_flux'] (and 'cube_err') are present.

    Parameters
    ----------
    stacks    : build_stacks(..., keep_cube=True) output. Needs 'cube_flux',
                'rest_wave', 'r_edges' (and 'cube_err' for method='inv_var').
    lines     : list of (name, lambda_rest_A, kind). None -> LIS_LINES.
    method    : galaxy-combine used inside the bootstrap (default 'biweight').
    product   : the SAME GalaxyProduct, for the area-weighted curve of growth
                (optional; the shape/S-N work without it).
    nboot     : bootstrap draws. Cost ~ nboot x nlines x nrad continuum fits;
                500 is a good default, 1000 for publication.
    rank_by   : 'central' (|inner-bin flux| / its bootstrap error) or 'profile'
                (quadrature S/N summed over all radial bins). Sets the ranking
                used for selection and the printed table.
    select_top: keep only the N highest-S/N lines in the composite (e.g. 5).
                None -> keep all that pass `min_sn`.
    min_sn    : drop lines whose `rank_by` S/N is below this before selecting.
                None -> no threshold.

    Returns
    -------
    dict  (superset of measure_line_profiles, so plot_line_profiles works on it):
        per_line  {key: {name, lam, kind,
                         flux, flux_err (nrad,)          fiducial flux + boot error
                         flux_lo, flux_hi (nrad,)        16/84 flux band
                         norm_profile (nrad,)            fiducial, normed to inner bin
                         norm_lo, norm_hi (nrad,)        bootstrap band on the norm profile
                         norm_value, norm_value_err      inner-bin amplitude + its error
                         sn_central, sn_profile          the two S/N definitions
                         centroid_v, centroid_v_err (nrad,)   (if measure_centroid_too)
                         cog_norm (nrad,)|None
                         flux_all (nboot,nrad)}           raw draws (for custom stats)
        ranking      list of (key, sn) sorted high->low by `rank_by`
        selected     list[key]   lines kept for the composite
        comp_norm_med/_lo/_hi (nrad,)   composite over SELECTED lines, bootstrap band
        comp_norm_mean                  (kept for API parity; == median here)
        lines_used, r_edges, r_mid, bin_mode, method, hw, dropped, meta
    """
    if "cube_flux" not in stacks:
        raise KeyError("bootstrap_line_profiles needs the per-galaxy cube. "
                       "Rebuild with build_stacks(config, product, keep_cube=True).")
    cube_flux = np.asarray(stacks["cube_flux"], dtype=float)      # (ngal, nrad, nwave)
    cube_err = stacks.get("cube_err")
    cube_err = np.asarray(cube_err, dtype=float) if cube_err is not None else None
    rest_wave = np.asarray(stacks["rest_wave"], dtype=float)
    r_edges = np.asarray(stacks["r_edges"], dtype=float)
    ngal, nrad, _ = cube_flux.shape
    lines = list(LIS_LINES if lines is None else lines)
    cont_kw = _cont_kwargs(config)
    rng = np.random.default_rng(seed)

    # --- geometry per line; drop any line off the rest grid (as measure_line_profiles) ---
    kept, dropped = [], []
    for (name, lam, kind) in lines:
        bounds, cont_bounds, hw = _recentre_geometry(config, lam)
        if drop_out_of_grid and not _footprint_in_grid(rest_wave, bounds, cont_bounds):
            dropped.append((name, lam))
            continue
        kept.append({"name": name, "lam": float(lam), "kind": kind,
                     "bounds": bounds, "cont_bounds": cont_bounds, "hw": hw})
    nline = len(kept)
    if nline == 0:
        raise ValueError("No lines fall inside the rest grid; widen rest_wave_max "
                         "and rebuild stacks.")

    # area weight for the curve of growth (optional, same as measure_line_profiles)
    nfib_med = None
    if product is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nfib_med = np.nanmedian(np.asarray(product.nfib, dtype=float), axis=0)

    def _measure_stack(stack2d):
        """Measure all kept lines on a (nrad, nwave) stack. -> (flux, cen) each (nline, nrad)."""
        flux = np.full((nline, nrad), np.nan)
        cen = np.full((nline, nrad), np.nan)
        for li, L in enumerate(kept):
            for r in range(nrad):
                fr = integrated_line_flux(rest_wave, stack2d[r], bounds=L["bounds"],
                                          cont_bounds=L["cont_bounds"], lya_center=L["lam"],
                                          clip_negative=clip_negative_flux, **cont_kw)
                flux[li, r] = fr["flux_sum"]
                if measure_centroid_too:
                    cr = flux_weighted_centroid(rest_wave, stack2d[r], bounds=L["bounds"],
                                                cont_bounds=L["cont_bounds"],
                                                lya_center=L["lam"], **cont_kw)
                    cen[li, r] = cr["centroid_vel_kms"]
        return flux, cen

    # --- fiducial point estimate on the full sample ---
    fid_stack, _ = stack_galaxies(cube_flux, cube_err, method=method)
    flux_fid, cen_fid = _measure_stack(fid_stack)

    # --- bootstrap draws ---
    flux_all = np.full((nboot, nline, nrad), np.nan)
    cen_all = np.full((nboot, nline, nrad), np.nan) if measure_centroid_too else None
    for b in tqdm(range(nboot), disable=not verbose, desc="LIS line bootstrap"):
        idx = rng.integers(0, ngal, ngal)
        f_bs = cube_flux[idx]
        e_bs = cube_err[idx] if cube_err is not None else None
        stack_bs, _ = stack_galaxies(f_bs, e_bs, method=method)
        fl, ce = _measure_stack(stack_bs)
        flux_all[b] = fl
        if measure_centroid_too:
            cen_all[b] = ce

    # --- per-line products ---
    per_line = OrderedDict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for li, L in enumerate(kept):
            ff = flux_fid[li]                              # (nrad,) fiducial flux
            fa = flux_all[:, li, :]                        # (nboot, nrad)
            f_lo = np.nanpercentile(fa, 16, axis=0)
            f_hi = np.nanpercentile(fa, 84, axis=0)
            f_err = 0.5 * (f_hi - f_lo)                    # robust per-bin 1-sigma

            # inner-bin amplitude (the normalization denominator) + its bootstrap error
            fin = np.isfinite(ff) & (ff != 0)
            inner = int(np.flatnonzero(fin)[0]) if np.any(fin) else 0
            amp_fid = ff[inner]
            amp_all = fa[:, inner]
            amp_err = 0.5 * (np.nanpercentile(amp_all, 84) - np.nanpercentile(amp_all, 16))
            sn_central = np.abs(amp_fid) / amp_err if amp_err > 0 else np.nan

            # whole-profile detection significance (uses all bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_bin_sn = ff / f_err
            sn_profile = float(np.sqrt(np.nansum(per_bin_sn ** 2)))

            # normalized profile + its bootstrap band (normalize each draw to ITS inner bin)
            norm_profile, _ = _normalize_to_inner(ff)
            norm_draws = np.full_like(fa, np.nan)
            for b in range(fa.shape[0]):
                norm_draws[b], _ = _normalize_to_inner(fa[b])
            norm_lo = np.nanpercentile(norm_draws, 16, axis=0)
            norm_hi = np.nanpercentile(norm_draws, 84, axis=0)

            # curve of growth (optional)
            cog_norm = None
            if nfib_med is not None:
                contrib = np.where(np.isfinite(ff * nfib_med), ff * nfib_med, 0.0)
                cog = np.cumsum(contrib)
                tot = cog[-1] if cog[-1] != 0 else np.nan
                cog_norm = cog / tot

            cen_v = cen_fid[li] if measure_centroid_too else None
            cen_err = None
            if measure_centroid_too:
                c_lo = np.nanpercentile(cen_all[:, li, :], 16, axis=0)
                c_hi = np.nanpercentile(cen_all[:, li, :], 84, axis=0)
                cen_err = 0.5 * (c_hi - c_lo)

            key = f"{L['name']}_{L['lam']:.0f}"
            per_line[key] = {
                "name": L["name"], "lam": L["lam"], "kind": L["kind"],
                "flux": ff, "flux_err": f_err, "flux_lo": f_lo, "flux_hi": f_hi,
                "norm_profile": norm_profile, "norm_lo": norm_lo, "norm_hi": norm_hi,
                "norm_value": float(amp_fid), "norm_value_err": float(amp_err),
                "sn_central": float(sn_central), "sn_profile": sn_profile,
                "centroid_v": cen_v, "centroid_v_err": cen_err,
                "cog_norm": cog_norm, "flux_all": fa,
            }

    # --- rank + select ---
    sn_key = "sn_central" if rank_by == "central" else "sn_profile"
    ranking = sorted(((k, per_line[k][sn_key]) for k in per_line),
                     key=lambda kv: (-(kv[1] if np.isfinite(kv[1]) else -np.inf)))
    selected = [k for k, sn in ranking if (min_sn is None or (np.isfinite(sn) and sn >= min_sn))]
    if select_top is not None:
        selected = selected[:int(select_top)]
    if not selected:                      # never return an empty composite
        selected = [ranking[0][0]]

    # --- composite over SELECTED lines, with a bootstrap band ---
    sel_idx = [list(per_line).index(k) for k in selected]
    comp_draws = np.full((nboot, nrad), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for b in range(nboot):
            nd = np.vstack([_normalize_to_inner(flux_all[b, j])[0] for j in sel_idx])
            comp_draws[b] = np.nanmedian(nd, axis=0)
        comp_med = np.nanpercentile(comp_draws, 50, axis=0)
        comp_lo = np.nanpercentile(comp_draws, 16, axis=0)
        comp_hi = np.nanpercentile(comp_draws, 84, axis=0)
        # fiducial composite = median over selected lines' fiducial norm profiles
        comp_fid = np.nanmedian(
            np.vstack([per_line[k]["norm_profile"] for k in selected]), axis=0)

    if verbose:
        print(f"\nbootstrapped LIS line profiles (method={method}, nboot={nboot}, "
              f"ranked by {sn_key}):")
        print(f"  {'line':<12}{'central S/N':>12}{'profile S/N':>13}{'kept?':>8}")
        for k, sn in ranking:
            d = per_line[k]
            mark = "keep" if k in selected else "drop"
            label = "{} {:.0f}".format(d["name"], d["lam"])
            print(f"  {label:<12}{d['sn_central']:>12.2f}"
                  f"{d['sn_profile']:>13.2f}{mark:>8}")
        if dropped:
            names = ", ".join(f"{n} {l:.0f}" for n, l in dropped)
            print(f"  [off rest grid, skipped: {names}]")

    return {
        "per_line": per_line,
        "ranking": ranking, "selected": selected,
        "lines_used": [(per_line[k]["name"], per_line[k]["lam"], per_line[k]["kind"])
                       for k in per_line],
        "r_edges": r_edges, "r_mid": _bin_midpoints(r_edges),
        "bin_mode": stacks.get("bin_mode"),
        "comp_norm_med": comp_med, "comp_norm_mean": comp_med,
        "comp_norm_lo": comp_lo, "comp_norm_hi": comp_hi,
        "comp_norm_fid": comp_fid,
        "method": method, "hw": kept[0]["hw"], "dropped": dropped,
        "meta": {"nboot": nboot, "ngal": ngal, "seed": seed, "rank_by": rank_by,
                 "select_top": select_top, "min_sn": min_sn},
    }


# ---------------------------------------------------------------------
# 6.1c  PLOT  the per-line S/N ranking (the new quantity)
# ---------------------------------------------------------------------
def plot_line_sn(boot, rank_by=None, figsize=(7.4, 4.6), title=None, save_fig=False):
    """
    Horizontal bar chart of each LIS line's signal-to-noise from
    `bootstrap_line_profiles`, sorted strongest at top. Selected (kept) lines are
    drawn solid; dropped lines are hatched/greyed, so you can see exactly where
    your select_top / min_sn cut landed.

    rank_by : 'central' or 'profile' to choose which S/N to bar (default: the one
              the bootstrap ranked by, from boot['meta']).
    """
    rb = rank_by or boot["meta"].get("rank_by", "central")
    sn_key = "sn_central" if rb == "central" else "sn_profile"
    selected = set(boot["selected"])

    items = sorted(boot["per_line"].items(),
                   key=lambda kv: (kv[1][sn_key] if np.isfinite(kv[1][sn_key]) else -np.inf))
    labels = [f"{d['name']} {d['lam']:.0f}" for _, d in items]
    vals = [d[sn_key] for _, d in items]
    keep = [k in selected for k, _ in items]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(items))
    for yi, (v, kp) in enumerate(zip(vals, keep)):
        ax.barh(yi, v, color=("tab:blue" if kp else "0.8"),
                edgecolor="black", hatch=(None if kp else "///"),
                label=("kept" if kp else "dropped"))
    if boot["meta"].get("min_sn") is not None:
        ax.axvline(boot["meta"]["min_sn"], color="crimson", ls="--", lw=1.2,
                   label=f"min S/N = {boot['meta']['min_sn']:g}")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(f"{'central' if rb == 'central' else 'profile'} S/N (bootstrap)")
    ax.set_title(title or f"LIS line detection S/N (kept {len(selected)} of {len(items)})")
    # de-duplicate legend
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    if save_fig:
        plt.savefig("Figure_line_sn.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------
# 6.1d  PLOT  oriented S/N vs radius  (sign flipped to expected sign)
# ---------------------------------------------------------------------
def _expected_sign(kind):
    """+1 for emission (flux expected positive), -1 for absorption (flux expected
    negative). Multiplying flux/flux_err by this orients EVERY line so that a real
    detection is POSITIVE, and a value going negative means the line has washed
    into the noise (or flipped to the wrong sign)."""
    return -1.0 if str(kind).lower().startswith("abs") else 1.0


def plot_line_sn_radial(lis_bs, lya_bs=None, only_selected=False,
                        guides=(1, 2, 3), figsize=(8.2, 5.2), title=None,
                        save_fig=False):
    """
    Oriented S/N (= sign_expected · flux/flux_err) per radial bin, one curve per
    line. Each line is flipped by the EXPECTED sign of its signal (absorption
    ·(-1), emission ·(+1)), so a real detection is POSITIVE for every line
    regardless of whether it absorbs or emits. A point dropping BELOW zero means
    that line has gone into the noise (or flipped to the wrong sign) in that bin —
    i.e. the signal is gone. This makes "where does each line die?" a single
    horizontal line (y=0) instead of having to track absorption down and emission
    up separately.

    lis_bs        : a bootstrap_line_profiles result.
    lya_bs        : optional bootstrap_line_profiles result on Lyα (emission), the
                    apples-to-apples overlay — does Lyα hold oriented S/N out where
                    the compact-ISM lines die?
    only_selected : restrict to lis_bs['selected'] (the kept lines).
    guides        : horizontal ±Nσ reference lines.
    """
    r = np.asarray(lis_bs["r_mid"], float)
    sel = set(lis_bs["selected"])
    items = list(lis_bs["per_line"].items())
    if only_selected:
        items = [(k, d) for k, d in items if k in sel]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(items), 1)))

    fig, ax = plt.subplots(figsize=figsize)
    for (k, d), c in zip(items, colors):
        s = _expected_sign(d.get("kind", "em"))
        sn = s * np.asarray(d["flux"], float) / np.asarray(d["flux_err"], float)
        kept = k in sel
        tag = "abs" if s < 0 else "em"
        ax.plot(r, sn, "o-" if kept else "o--", color=c, lw=1.8 if kept else 1.0,
                ms=5, alpha=1.0 if kept else 0.45,
                label=f"{d['name']} {d['lam']:.0f} ({tag})"
                      + ("" if kept else " — dropped"))

    # optional Lyα overlay (emission; oriented the same way)
    if lya_bs is not None:
        d = next(iter(lya_bs["per_line"].values()))
        s = _expected_sign(d.get("kind", "em"))
        sn = s * np.asarray(d["flux"], float) / np.asarray(d["flux_err"], float)
        ax.plot(r, sn, "D-", color="black", lw=2.4, ms=7, zorder=10, label="Lyα")

    ax.axhline(0, color="0.4", lw=1.2)               # signal-gone threshold
    for g in guides:
        for sgn in (+g, -g):
            ax.axhline(sgn, color="0.7", ls=":", lw=0.8)
        ax.text(r[-1], g, f"+{g}σ", fontsize=7.5, color="0.5", va="bottom", ha="right")
        ax.text(r[-1], -g, f"−{g}σ", fontsize=7.5, color="0.5", va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Radius [arcsec]")
    ax.set_ylabel("Oriented S/N  (expected-sign · flux / bootstrap error)")
    ax.set_title(title or "Per-line significance vs radius "
                          "(oriented: real signal positive)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.2, which="both")
    if save_fig:
        plt.savefig("Figure_line_sn_radial.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------
# 6.2  WAVELENGTH-MATCHED STELLAR PSF  ("their own PSF")
# ---------------------------------------------------------------------
def psf_profiles_for_lines(star_product, lines=None, z_obs=0.0,
                           band_halfwidth_A=30.0, combine="biweight",
                           clip=3.0, verbose=True):
    """
    The stellar PSF each line actually sees, sampled OBSERVED-frame at the
    wavelength where that rest-frame line lands for the galaxy sample.

    A rest-frame line at λ_rest appears at λ_obs = λ_rest·(1 + z_obs). The PSF at
    that wavelength is just the star's continuum surface-brightness profile in a
    band around λ_obs — measured with the EXISTING radial_continuum_profile on
    the star product (mask='none' extraction; observed frame; no (1+z) resample).
    Normalized to the inner bin, it is directly comparable to a galaxy line's
    norm_profile from measure_line_profiles.

    >>> NOTE ON z <<<  In pure arcsec mode the star PSF curve-of-growth itself
    does not use redshift at all (dummy z; the bins are absolute arcsec). The
    galaxy z-distribution enters ONLY here, through z_obs (use the galaxy
    stacks['z_median']), which tells each rest-frame line where to sample the
    observed-frame stellar PSF. Pass z_obs=0 to get the PSF at the rest
    wavelengths directly (a stellar-frame check).

    Returns dict:
        per_line  {key: {name, lam_rest, lam_obs, norm_profile (nrad,),
                         sb (nrad,), profile (the radial_continuum_profile dict)}}
        r_edges, r_mid, bin_mode
        comp_norm_med / comp_norm_lo / comp_norm_hi (nrad,)  composite PSF band
    """
    lines = list(LIS_LINES if lines is None else lines)
    per_line = OrderedDict()
    r_edges = np.asarray(star_product.r_edges, dtype=float)


    for (name, lam, kind) in lines:
        lam_obs = float(lam) * (1.0 + float(z_obs))
        band = (lam_obs - band_halfwidth_A, lam_obs + band_halfwidth_A)
        prof = radial_continuum_profile(star_product, band=band,
                                        combine=combine, clip=clip)
        norm_profile, _ = _normalize_to_inner(prof["sb"])
        key = f"{name}_{lam:.0f}"
        per_line[key] = {"name": name, "lam_rest": float(lam), "lam_obs": lam_obs,
                         "norm_profile": norm_profile, "sb": prof["sb"],
                         "profile": prof}

    if per_line:
        stack_norm = np.vstack([d["norm_profile"] for d in per_line.values()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            comp_med = np.nanmedian(stack_norm, axis=0)
            comp_lo = np.nanpercentile(stack_norm, 16, axis=0)
            comp_hi = np.nanpercentile(stack_norm, 84, axis=0)
        r_mid = next(iter(per_line.values()))["profile"]["r_mid"]
    else:
        comp_med = comp_lo = comp_hi = np.full(len(r_edges) - 1, np.nan)
        r_mid = _bin_midpoints(r_edges)

    if verbose:
        print(f"PSF sampled at z_obs={z_obs:g} for {len(per_line)} lines "
              f"(±{band_halfwidth_A:g} Å observed-frame bands).")

    return {
        "per_line": per_line, "r_edges": r_edges, "r_mid": r_mid,
        "bin_mode": "arcsec",   # PSF is always measured in observed-frame arcsec bins
        "comp_norm_med": comp_med, "comp_norm_lo": comp_lo, "comp_norm_hi": comp_hi,
    }


# ---------------------------------------------------------------------
# 6.3  PLOT  (lines + composite + PSF + Lyα, all normalized to the inner bin)
# ---------------------------------------------------------------------
def plot_line_profiles(result, psf=None, lya_result=None, bin_mode="arcsec",
                       statistic="norm_profile", logy=False, show_lines=True,
                       figsize=(7.6, 5.0), ylims=(-0.25, 1.25), title=None,
                       save_fig=False):
    """
    Overlay, vs radius, the normalized profile of every measured line (thin
    grey), the LIS composite median with its line-to-line 16/84 band (black),
    the stellar PSF (dashed red), and — if given — Lyα (blue).

    Everything is normalized to its inner finite bin, so a point source (any LIS
    line, and the PSF) collapses onto one falling curve; Lyα sitting ABOVE that
    curve at large radius is the resolved CGM excess.

    psf         : either a psf_profiles_for_lines result (uses its composite
                  band) OR a single radial_continuum_profile dict (one PSF curve,
                  normalized here). None → no PSF drawn.
    lya_result  : a measure_line_profiles result whose composite is the Lyα
                  profile, e.g. measure_line_profiles(cfg, gal_stacks,
                  lines=[("Lyα", cfg.LYA_REST, "em")]).
    statistic   : 'norm_profile' (default) or 'cog_norm' (curve of growth; needs
                  product= when measuring).
    """
    r_mid = np.asarray(result["r_mid"], dtype=float)
    fig, ax = plt.subplots(figsize=figsize)

    # individual LIS lines
    if show_lines:
        for d in result["per_line"].values():
            y = np.asarray(d.get(statistic), dtype=float)
            if y is None or not np.any(np.isfinite(y)):
                continue
            ax.plot(r_mid, y, "-", color="0.7", lw=1.0, alpha=0.8, zorder=1,
                    label="_nolegend_")

    # LIS composite (median + line-to-line scatter band) — only for norm_profile
    if statistic == "norm_profile":
        ax.fill_between(r_mid, result["comp_norm_lo"], result["comp_norm_hi"],
                        color="0.5", alpha=0.25, lw=0, zorder=2,
                        label="LIS line-to-line 16–84%")
        ax.plot(r_mid, result["comp_norm_med"], "o-", color="black", lw=2.0,
                ms=6, zorder=4, label="LIS composite (median)")

    # stellar PSF
    if psf is not None:
        if "comp_norm_med" in psf:                        # per-line PSF result
            ax.fill_between(psf["r_mid"], psf["comp_norm_lo"], psf["comp_norm_hi"],
                            color="tomato", alpha=0.12, lw=0, zorder=2)
            ax.plot(psf["r_mid"], psf["comp_norm_med"], "s--", color="tomato",
                    lw=2.0, ms=5, zorder=5, label="stellar PSF (composite)")
        else:                                             # radial_continuum_profile dict
            pnorm, _ = _normalize_to_inner(psf["sb"])
            ax.plot(np.asarray(psf["r_mid"]), pnorm, "s--", color="tomato",
                    lw=2.0, ms=5, zorder=5, label="stellar PSF")

    # Lyα
    if lya_result is not None:
        y = np.asarray(lya_result.get("comp_norm_med"), dtype=float)
        if not np.any(np.isfinite(y)):                    # single-line result fallback
            d0 = next(iter(lya_result["per_line"].values()), None)
            y = np.asarray(d0["norm_profile"]) if d0 else y
        ax.plot(np.asarray(lya_result["r_mid"]), y, "D-", color="tab:blue",
                lw=2.0, ms=6, zorder=6, label="Lyα")

    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    elif ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlabel(_radius_label(bin_mode))
    ax.set_ylabel("Enclosed flux (norm.)" if statistic == "cog_norm"
                  else "Line flux / inner-bin flux")
    ax.set_title(title or "LIS lines vs. stellar PSF (do the lines trace the PSF?)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(alpha=0.2, which="both")
    if save_fig:
        plt.savefig("Figure_line_profile_psf.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax
