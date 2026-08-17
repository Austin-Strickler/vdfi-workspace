"""
multicat.py -- connect and compare galaxy products across catalogs.

Stage 2 already loads ONE field/catalog (io.load_for_stack) and can concatenate
AEGIS+COSMOS (io._concat_products) for a single catalog. This module generalizes
that to the questions "what do the low-z vs high-z galaxies look like?" and "let
me stack several catalogs together", in two complementary ways:

  * SPLIT one product by a catalog column (e.g. z into low/high) -> sub-products
    you stack and overlay. Cheap, Stage-2 only, no re-extraction. This is the
    fast path for low-z vs high-z WITHIN a catalog.
  * LOAD several catalog FITS (each its own extraction) -> a labelled dict you
    compare side-by-side, or COMBINE into one product (with a provenance column
    so a joint stack can still be split by origin later). This is the path for
    genuinely DIFFERENT catalogs.

Comparisons are most meaningful in virial (R/Rvir) bins, which are dimensionless
and already fold in each galaxy's mass and z -- so low-z and high-z samples land
on a common radial axis.

Compatibility: combining/comparing requires the SAME wave grid, nrad, and
R_EDGES (i.e. same bin_mode + bins + wavelength axis at extraction). The loaders
check this and refuse or warn, mirroring io._concat_products.

Also carries the SPLIT CATALOG (section 4b) -- Layer A of the subsample-split
pipeline (docs/workstream-roadmap.md Sec 3.1). One row per galaxy, one column per
splittable quantity, keyed on the galaxy ID rather than on any product's galaxy
axis, joining the halo-side master catalog to the core extraction (exact ID join)
and to foreign catalogs (positional). Build it once with build_split_catalog,
audit it with split_catalog_report, put it on any product with
attach_split_catalog, then split with the existing split_product_by.

Also carries the Chartab et al. (2020) CANDELS overdensity catalog helpers
(load_overdensity/attach_overdensity/split_by_overdensity, folded in from the
former standalone environment.py on 2026-07-21 -- see that section below for
why) -- a catalog-specific SPLIT built on this module's own generic
add_matched_column/split_product_by.

Not in utils_lya_halo/__init__ (imports matplotlib). Use:
    from utils_lya_halo import multicat
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import TYPE_CHECKING
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, vstack

from .io import (
    GalaxyProduct, read_galaxy_fits, apply_finite_cut, load_for_stack,
    write_galaxy_fits,
)
from .stack import build_stacks
from .plotting import _setup_radius_axis, _resolve_bin_mode, _get_vr_biweight_v

if TYPE_CHECKING:
    from .config import PipelineConfig

__all__ = [
    "load_products",
    "combine_products",
    "split_product_by",
    "split_by_redshift",
    "match_catalog",
    "add_matched_column",
    "SPLIT_DERIVED",
    "build_split_catalog",
    "attach_split_catalog",
    "write_split_catalog",
    "read_split_catalog",
    "split_catalog_report",
    "stack_each",
    "stack_id_list",
    "measure_each",
    "compare_centroids",
    "load_overdensity",
    "attach_overdensity",
    "split_by_overdensity",
]


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _slice_product(product: GalaxyProduct, sel: np.ndarray) -> GalaxyProduct:
    """Slice every galaxy-axis array + the catalog by a boolean mask."""
    bg = product.background[sel] if product.background is not None else None
    return GalaxyProduct(
        spec=product.spec[sel], err=product.err[sel], nfib=product.nfib[sel],
        wave=product.wave, r_edges=product.r_edges, catalog=product.catalog[sel],
        header=product.header, background=bg,
    )


def _check_compatible(products, labels):
    """Ensure wave / nrad / r_edges match across products (so they share a
    radial + wavelength axis). Raises on mismatch."""
    ref = products[0]
    for lab, p in zip(labels[1:], products[1:]):
        if not np.allclose(ref.wave, p.wave, equal_nan=True):
            raise ValueError(f"'{lab}': WAVE grid differs from '{labels[0]}'.")
        if ref.spec.shape[1] != p.spec.shape[1]:
            raise ValueError(f"'{lab}': nrad {p.spec.shape[1]} != "
                             f"{ref.spec.shape[1]} ('{labels[0]}').")
        if not np.allclose(ref.r_edges, p.r_edges, equal_nan=True):
            raise ValueError(f"'{lab}': R_EDGES differ from '{labels[0]}'.")


# ---------------------------------------------------------------------
# 1. LOAD several catalogs
# ---------------------------------------------------------------------
def load_products(specs, base_config=None, apply_cut=True, verbose=True):
    """
    Load several galaxy products into a labelled dict.

    specs : list of dicts, each with a 'label' and ONE source of:
        'path'      -> read that galaxy-FITS path directly, or
        'config'    -> a full PipelineConfig (resolved via load_for_stack), or
        'overrides' -> dict applied to base_config via dataclasses.replace,
                       then load_for_stack (e.g. {'catalog':'3DHST',
                       'catalog_path': '.../3dhst.txt'}).

    apply_cut : for 'path' specs, apply the finite cut using base_config's
                cut_radial_bin / min_good_wave (config/overrides specs cut inside
                load_for_stack already). Set False to skip for 'path' specs.

    Returns OrderedDict{label: GalaxyProduct}.
    """
    out = OrderedDict()
    for s in specs:
        label = s["label"]
        if "path" in s:
            p = read_galaxy_fits(s["path"])
            if apply_cut:
                if base_config is None:
                    raise ValueError("apply_cut=True with a 'path' spec needs "
                                     "base_config for the cut parameters.")
                p = apply_finite_cut(p, base_config.cut_radial_bin,
                                     base_config.min_good_wave, verbose=verbose)
        elif "config" in s:
            p = load_for_stack(s["config"], verbose=verbose)
        elif "overrides" in s:
            if base_config is None:
                raise ValueError("'overrides' spec needs base_config.")
            cfg = replace(base_config, **s["overrides"])
            p = load_for_stack(cfg, verbose=verbose)
        else:
            raise ValueError(f"spec {label!r} needs 'path', 'config', or 'overrides'.")
        if verbose:
            print(f"loaded '{label}': {p.ngal} galaxies")
        out[label] = p
    return out


# ---------------------------------------------------------------------
# 2. COMBINE several catalogs (with provenance)
# ---------------------------------------------------------------------
def combine_products(products, label_col="SUBSET", verbose=True) -> GalaxyProduct:
    """
    Concatenate a labelled dict of products along the galaxy axis into one
    product, tagging each galaxy with its origin in catalog column `label_col`
    so a joint stack can be split by origin afterwards (split_product_by).

    Catalog columns may differ across catalogs; they are unioned (vstack
    join_type='outer'). z_col / mass_col must be present in every catalog for
    Stage 2 -- a warning is printed if a catalog lacks common columns.
    """
    labels = list(products.keys())
    plist = list(products.values())
    _check_compatible(plist, labels)

    cats = []
    colsets = []
    for lab, p in zip(labels, plist):
        c = p.catalog.copy()
        c[label_col] = np.full(len(c), str(lab))
        cats.append(c)
        colsets.append(set(c.colnames))
    common = set.intersection(*colsets) if colsets else set()
    if verbose and len(colsets) > 1:
        for lab, cs in zip(labels, colsets):
            missing = common.symmetric_difference(cs) - common
            if missing:
                print(f"note: '{lab}' has extra/missing columns vs the common set: "
                      f"{sorted(missing)[:6]}{'...' if len(missing) > 6 else ''}")

    catalog = vstack(cats, join_type="outer", metadata_conflicts="silent")
    bgs = [p.background for p in plist]
    bg = np.concatenate(bgs, axis=0) if all(b is not None for b in bgs) else None
    combined = GalaxyProduct(
        spec=np.concatenate([p.spec for p in plist], axis=0),
        err=np.concatenate([p.err for p in plist], axis=0),
        nfib=np.concatenate([p.nfib for p in plist], axis=0),
        wave=plist[0].wave, r_edges=plist[0].r_edges, catalog=catalog,
        header=plist[0].header, background=bg,
    )
    if verbose:
        print(f"combined {len(labels)} products -> {combined.ngal} galaxies "
              f"(origin in column '{label_col}')")
    return combined


# ---------------------------------------------------------------------
# 3. SPLIT one product (low-z vs high-z, or any column / group)
# ---------------------------------------------------------------------
def _write_subproduct(sub, config, output_dir, tag):
    """Write one sub-product as a galaxy FITS to OUTPUTS (never cache)."""
    od = output_dir if output_dir is not None else config.output_dir
    os.makedirs(od, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z._-]+", "", str(tag).replace("%", "pct").replace("/", "-"))
    base = config.descriptive_filename()
    name = (base[:-5] if base.endswith(".fits") else base) + f"_split-{safe}.fits"
    path = os.path.join(od, name)
    src = None
    if sub.header is not None:
        src = sub.header.get("SOURCE") or sub.header.get("SOURCEID")
    write_galaxy_fits(
        path, spec=sub.spec, err=sub.err, nfib=sub.nfib, wave=sub.wave,
        r_edges=sub.r_edges, catalog=sub.catalog, config=config,
        source_id=src or f"split:{tag}", background=sub.background, overwrite=True,
    )
    return path


def split_product_by(product: GalaxyProduct, column, edges=None, percentiles=None,
                     groups=None, labels=None, write=False, config=None,
                     output_dir=None, verbose=True):
    """
    Split ONE product into sub-products by a catalog column. Pure Stage-2, no
    re-extraction; useful for low-z vs high-z and any other property split.

    Choose exactly ONE rule:
      edges       : value bounds. N edges -> N-1 samples. e.g. [a, b, c] -> 2
                    samples [a,b) and [b,c]; [0, m, hi] for a value split at m.
      percentiles : percentile bound(s) in (0,100) of the column's own
                    distribution. A scalar p -> two samples 0-p% and p-100%
                    (e.g. 50 = median split, 20 = a 0-20 / 20-100 split). A list
                    e.g. [20, 80] -> three samples 0-20 / 20-80 / 80-100%.
      groups      : categorical, e.g. [['MOSDEF'], ['3DHST','OTHER']].
    The LAST bin is inclusive of its upper edge so the maximum value is kept.

    write : if True, also write each sub-product as a galaxy FITS to OUTPUTS
            (output_dir or config.output_dir -- NOT the cache), filename tagged
            by the split label. Requires `config` (for the filename + provenance
            header). The in-memory dict is returned regardless.

    Returns OrderedDict{label: GalaxyProduct}. Non-finite / out-of-range rows are
    dropped from all sub-products.
    """
    if sum(x is not None for x in (edges, percentiles, groups)) != 1:
        raise ValueError("pass exactly one of `edges`, `percentiles`, or `groups`.")
    if write and config is None:
        raise ValueError("write=True needs `config` (for filename + provenance).")

    col = np.asarray(product.catalog[column])
    out = OrderedDict()

    # percentiles -> value edges + percentile-range labels
    if percentiles is not None:
        vals = col.astype(float)
        pcts = np.atleast_1d(np.asarray(percentiles, dtype=float))
        if np.any((pcts <= 0) | (pcts >= 100)):
            raise ValueError("percentiles must be strictly inside (0, 100).")
        bounds_pct = np.concatenate([[0.0], np.sort(pcts), [100.0]])
        value_edges = np.nanpercentile(vals[np.isfinite(vals)], bounds_pct)
        edges = value_edges
        auto_labels = [f"{bounds_pct[i]:g}-{bounds_pct[i+1]:g}%"
                       for i in range(len(bounds_pct) - 1)]
    else:
        auto_labels = None

    if edges is not None:
        edges = np.asarray(edges, dtype=float)
        vals = col.astype(float)
        nbin = len(edges) - 1
        for i in range(nbin):
            lo, hi = edges[i], edges[i + 1]
            last = (i == nbin - 1)
            sel = (np.isfinite(vals) & (vals >= lo) &
                   ((vals <= hi) if last else (vals < hi)))
            lab = (labels[i] if labels is not None
                   else (auto_labels[i] if auto_labels is not None
                         else f"[{lo:g},{hi:g}{']' if last else ')'}"))
            if verbose:
                print(f"  {column} {lab}: {int(sel.sum())} galaxies")
            sub = _slice_product(product, sel)
            if write:
                path = _write_subproduct(sub, config, output_dir, f"{column}-{lab}")
                if verbose:
                    print(f"    -> wrote {path}")
            out[lab] = sub
    else:  # groups
        sval = col.astype(str)
        for i, g in enumerate(groups):
            allowed = set(str(x) for x in (g if isinstance(g, (list, tuple, set)) else [g]))
            sel = np.array([v in allowed for v in sval])
            lab = labels[i] if labels is not None else "+".join(sorted(allowed))
            if verbose:
                print(f"  {column} {lab}: {int(sel.sum())} galaxies")
            sub = _slice_product(product, sel)
            if write:
                path = _write_subproduct(sub, config, output_dir, f"{column}-{lab}")
                if verbose:
                    print(f"    -> wrote {path}")
            out[lab] = sub
    return out


def split_by_redshift(product: GalaxyProduct, edges=None, percentiles=None,
                      z_col="z", labels=None, write=False, config=None,
                      output_dir=None, verbose=True):
    """Convenience: split by the redshift column, by value `edges` or
    `percentiles` (one of them). All other options forward to split_product_by."""
    return split_product_by(product, z_col, edges=edges, percentiles=percentiles,
                            labels=labels, write=write, config=config,
                            output_dir=output_dir, verbose=verbose)


# ---------------------------------------------------------------------
# 4. MATCH a SEPARATE catalog by sky position (cross-catalog link)
# ---------------------------------------------------------------------
# combine_products concatenates products from the SAME extraction (shared wave /
# nrad / r_edges, IDs already aligned). The other way two catalogs relate is when
# they are wholly independent measurements of overlapping galaxies -- different
# source, different IDs, different row order, possibly different ngal -- so the
# only reliable join is sky position. match_catalog provides that positional key;
# once a foreign column is on the product's galaxy axis you split on it with
# split_product_by exactly like a native column, so all grouping logic stays in
# one place (no per-class slice_product boilerplate).

def _match_radec(ra_p, dec_p, ra_e, dec_e, radius_arcsec, chunk=2000):
    """Nearest external row per product galaxy: vectorized haversine (same
    great-circle formula as the cone searches), NaN-safe, chunked over the galaxy
    axis so memory stays O(chunk * n_ext). Returns (ext_index, sep_arcsec), with
    -1 / NaN where no external row lies within `radius_arcsec`."""
    ra_p = np.asarray(ra_p, float); dec_p = np.asarray(dec_p, float)
    ra_e = np.asarray(ra_e, float); dec_e = np.asarray(dec_e, float)
    n_p = ra_p.size
    fin_e = np.isfinite(ra_e) & np.isfinite(dec_e)
    re_, de_ = np.radians(ra_e), np.radians(dec_e)
    idx = np.full(n_p, -1, dtype=int)
    sep = np.full(n_p, np.nan, dtype=float)
    for s in range(0, n_p, chunk):
        sl = slice(s, min(s + chunk, n_p))
        rp = np.radians(ra_p[sl])[:, None]; dp = np.radians(dec_p[sl])[:, None]
        a = (np.sin((de_[None, :] - dp) / 2) ** 2
             + np.cos(dp) * np.cos(de_[None, :])
             * np.sin((re_[None, :] - rp) / 2) ** 2)
        d = np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0
        d[:, ~fin_e] = np.inf
        d[~(np.isfinite(ra_p[sl]) & np.isfinite(dec_p[sl])), :] = np.inf
        j = np.argmin(d, axis=1)
        dj = d[np.arange(d.shape[0]), j]
        ok = dj <= float(radius_arcsec)
        rows = np.arange(sl.start, sl.stop)
        idx[rows[ok]] = j[ok]
        sep[rows[ok]] = dj[ok]
    return idx, sep


def _gather(col, idx, matched, ngal, fill):
    """An external column gathered onto the galaxy axis; unmatched -> fill.
    Numeric -> float/NaN, string -> object/"", bool -> bool/False, unless `fill`
    overrides. The string default puts unmatched rows in NO group, so a following
    split_product_by(..., groups=...) drops them for free."""
    col = np.asarray(col)
    if col.dtype == bool:
        out = np.full(ngal, bool(fill) if fill is not None else False, dtype=bool)
    elif np.issubdtype(col.dtype, np.number):
        out = np.full(ngal, np.nan if fill is None else float(fill), dtype=float)
    else:
        out = np.full(ngal, "" if fill is None else fill, dtype=object)
    out[matched] = col[idx[matched]]
    return out


def match_catalog(product: GalaxyProduct, external, *, radius_arcsec=1.0,
                  ra_col="RA", dec_col="DEC", ext_ra_col=None, ext_dec_col=None,
                  verbose=True):
    """Positionally link a SEPARATE catalog to a product's galaxy axis.

    For each galaxy in `product`, find the nearest row in `external` (any astropy
    Table -- or dict of arrays -- with RA/DEC) within `radius_arcsec`. This is the
    cross-catalog key for two independent measurements of overlapping galaxies.

    Returns (ext_index, sep_arcsec, matched), each length product.ngal and aligned
    to its galaxy axis:
        ext_index  : row in `external` matched (-1 where none within radius)
        sep_arcsec : separation to that row, arcsec (NaN where unmatched)
        matched    : boolean -- has a match. Use directly as a mask
                     (selection.split_by_mask / slice the product), or pass
                     through add_matched_column to carry a column over.

    Takes the SINGLE nearest external row per galaxy (it does not deduplicate the
    external side -- two galaxies may share a nearest row). Coordinate columns
    default to RA/DEC on both sides; set ext_ra_col / ext_dec_col if the external
    catalog names them differently.
    """
    cat = product.catalog
    ext = external if hasattr(external, "colnames") else Table(external)
    era, edc = ext_ra_col or ra_col, ext_dec_col or dec_col
    idx, sep = _match_radec(
        np.asarray(cat[ra_col], float), np.asarray(cat[dec_col], float),
        np.asarray(ext[era], float), np.asarray(ext[edc], float), radius_arcsec)
    matched = idx >= 0
    if verbose:
        n, ng = int(matched.sum()), len(cat)
        print(f"match_catalog: {n}/{ng} galaxies matched ({n/ng:.0%}) within "
              f"{radius_arcsec:g}\" of {len(ext)} external rows")
    return idx, sep, matched


def add_matched_column(product: GalaxyProduct, external, column, *,
                       new_name=None, radius_arcsec=1.0, ra_col="RA",
                       dec_col="DEC", ext_ra_col=None, ext_dec_col=None,
                       fill=None, verbose=True) -> GalaxyProduct:
    """Return a COPY of `product` whose catalog carries `external[column]`,
    matched onto the galaxy axis by sky position (match_catalog).

    new_name : catalog column to write (default: `column`).
    fill     : value for unmatched galaxies (default NaN for numeric, "" for
               string, False for boolean). With the "" default, unmatched rows
               fall into NO group, so a later split_product_by(..., groups=[...])
               drops them automatically.

    Only the catalog is copied (galaxy-axis arrays are shared), like
    combine_products -- so this is cheap and the result is a normal product. The
    whole "carry a foreign classification over and split on it" workflow is then
    two calls:

        prod2  = add_matched_column(mega, ext, "CLASS", radius_arcsec=0.5)
        groups = split_product_by(prod2, "CLASS",
                     groups=[["AGN"], ["LAE"], ["emitter"], ["null"],
                             ["absorber"], ["LAE", "emitter"], ["null", "absorber"]],
                     labels=["AGN","LAE","emitter","null","absorber","bright","dim"])
    """
    ext = external if hasattr(external, "colnames") else Table(external)
    idx, sep, matched = match_catalog(
        product, ext, radius_arcsec=radius_arcsec, ra_col=ra_col, dec_col=dec_col,
        ext_ra_col=ext_ra_col, ext_dec_col=ext_dec_col, verbose=verbose)
    name = new_name or column
    cat = product.catalog.copy()
    cat[name] = _gather(ext[column], idx, matched, len(cat), fill)
    if verbose:
        print(f"  -> wrote catalog column '{name}' "
              f"({int(matched.sum())} matched values, rest = {fill!r})")
    return replace(product, catalog=cat)


# ---------------------------------------------------------------------
# 4b. THE SPLIT CATALOG  --  Layer A of the subsample-split pipeline
# ---------------------------------------------------------------------
# One row per galaxy, one column per splittable quantity, NaN/False where the
# quantity was never measured. See docs/workstream-roadmap.md Sec 3.1.
#
# WHY THIS IS A TABLE AND NOT A PRODUCT. The obvious design is "carry the core
# columns onto the halo product's galaxy axis and be done", which is what the
# notebook cells did one column at a time. But the galaxy axis belongs to ONE
# product: re-extract, re-cut, or swap mega for a per-field product and every
# such attachment has to be rebuilt, and a mask built against the old axis is
# now silently wrong (selection.split_by_mask exists to catch exactly that). The
# split catalog is instead keyed on the galaxy ID and is independent of any
# product's row order or length. Build it once; attach_split_catalog reindexes
# it onto whatever product you happen to be holding.
#
# WHY AN ID JOIN AND NOT match_catalog. The halo product and the core product
# descend from the SAME master catalog, so ID is an exact key. Positional
# matching is the right tool for a foreign catalog (Chartab -- section 6 below),
# but against your own core product it is strictly worse: _match_radec takes the
# single nearest external row without deduplicating the external side, so two
# galaxies can silently claim the same core row, and a galaxy whose coordinates
# were revised between catalog versions silently drops to "unmatched" -- which
# then looks identical to "off the core field". The ID join fails loudly instead
# (duplicate IDs raise) and every unmatched row is real missing coverage.
#
# THREE FACTS, THREE COLUMNS. The notebook version folded spectral class, the
# AGN flag and the KEEP cut into a single CLASS string, using "" for both "cut"
# and "no core coverage". Roadmap Sec 3.3b turns on being able to tell those
# apart -- a galaxy off the core field drops out of the L_Lya and dv splits but
# stays in the mass and z splits, so the two splits run on different samples and
# any cross-split comparison is confounded. So coverage (CORE_COVERED), the cuts
# (CLEAN / CORE_KEEP / CORE_AGN_FLAG) and the spectral class (CORE_LABEL) are
# kept separate here and combined per split, by the split spec, in Layer B.

_LYA_FLUX_TO_CGS = 1e-17   # core flux columns are in units of 1e-17 erg/s/cm^2

# Core-product CATALOG columns to carry alongside the RESULTS arrays.
#
# EMPTY BY DEFAULT, deliberately. The core product's catalog is a SUBSET OF THE
# SAME master catalog the halo product carries -- KEEP, AGN_FLAG, CLEAN,
# MASS_50, SFR_50 and ~200 others are already on the halo side, so carrying them
# again under a CORE_ prefix produces two copies of one column that are equal by
# construction and invite a split spec to reference the wrong one. Only the
# RESULTS arrays (the core MEASUREMENT: flux, v_offset, EW, fesc, label) are
# genuinely core-side and are always carried.
#
# Set this if a future core product ever carries per-galaxy catalog columns the
# halo side does not have.
_CORE_CATALOG_COLS = ()


# The join key is a COMPOSITE, and it has to be. MOSDEF IDs are scoped to their
# field, not to the survey: AEGIS 5814 and COSMOS 5814 are different galaxies at
# completely different coordinates (214.82,+52.79 vs 150.17,+2.24). The master
# catalog carries three ID columns -- ID (field-scoped), ID_V21 and ID_V41 (the
# two 3D-HST catalog versions, and note ID equals ID_V21 in AEGIS but ID_V41 in
# COSMOS, so none of the three is a survey-wide key on its own). So the default
# key is FIELD+ID, and a bare ID is used only if it is provably unique.
#
# This is exactly what the duplicate guard in _index_by_id exists to catch: a
# bare-ID join here would not have errored, it would have quietly mapped every
# AEGIS galaxy onto its COSMOS namesake's core measurement.
_DEFAULT_ID_CANDIDATES = (("FIELD", "ID"), ("ID",))


def _as_key_list(id_col):
    """Normalize an id_col argument to a list of column names. Accepts a single
    name ('ID') or a composite ('FIELD', 'ID')."""
    if id_col is None:
        return None
    if isinstance(id_col, str):
        return [id_col]
    return list(id_col)


def _norm_id_col(col) -> np.ndarray:
    """Normalize one key column to comparable strings, so 123, 123.0, '123' and
    b'123' are all the same value (core.select_core_by_id's convention).
    Strings are stripped; case is preserved deliberately -- silently
    case-folding 'AEGIS'/'aegis' would hide a real catalog inconsistency."""
    arr = np.asarray(col).ravel()

    def _one(v):
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8")
        if isinstance(v, (float, np.floating)):
            if np.isfinite(v) and float(v) == int(v):
                return str(int(v))
            return repr(float(v))
        return str(v).strip()

    return np.array([_one(v) for v in arr], dtype=object)


def _id_key(cat, cols) -> np.ndarray:
    """The composite join key of a catalog as one string array. Parts are joined
    with an ASCII unit separator, which cannot occur inside an ID or a field
    name, so 'AEGIS'+'5814' can never collide with 'AEGIS5'+'814'."""
    parts = [_norm_id_col(cat[c]) for c in cols]
    if len(parts) == 1:
        return parts[0]
    return np.array(["\x1f".join(t) for t in zip(*parts)], dtype=object)


def _key_is_unique(cat, cols):
    k = _id_key(cat, cols).astype(str)
    return np.unique(k).size == len(k)


def _resolve_id_cols(id_col, cats, *, verbose=True):
    """Pick the join key. An explicit `id_col` is honoured (and only checked for
    presence); otherwise the first of _DEFAULT_ID_CANDIDATES that exists on
    EVERY catalog and is unique on every one of them wins.

    Raises with a worked example if no candidate is unique -- an ambiguous key
    is not something to warn about and continue past, because the resulting
    join would be silently wrong rather than absent.
    """
    cats = [c for c in cats if c is not None]
    if not cats:
        raise ValueError("no catalogs given")

    if id_col is not None:
        cols = _as_key_list(id_col)
        for i, c in enumerate(cats):
            miss = [k for k in cols if k not in c.colnames]
            if miss:
                raise KeyError(f"join key column(s) {miss} not in catalog {i} "
                               f"(columns: {list(c.colnames)})")
        return cols

    for cand in _DEFAULT_ID_CANDIDATES:
        cols = list(cand)
        if not all(all(k in c.colnames for k in cols) for c in cats):
            continue
        if all(_key_is_unique(c, cols) for c in cats):
            if verbose:
                print(f"  join key: {'+'.join(cols)}"
                      + ("" if len(cols) == 1 else
                         "  (a bare ID is NOT unique -- IDs are scoped to their field)"))
            return cols

    # nothing worked: show the actual collision rather than a generic message
    for cand in _DEFAULT_ID_CANDIDATES:
        cols = list(cand)
        for i, c in enumerate(cats):
            if not all(k in c.colnames for k in cols):
                continue
            k = _id_key(c, cols).astype(str)
            u, n = np.unique(k, return_counts=True)
            if (n > 1).any():
                ex = u[n > 1][0]
                raise ValueError(
                    f"no usable join key. {'+'.join(cols)} is duplicated in "
                    f"catalog {i} (e.g. {ex!r} appears {int(n[n>1][0])} times). "
                    f"Pass an explicit key, e.g. id_col=('FIELD','ID') or a "
                    f"genuinely survey-wide column.")
    raise ValueError(
        f"no usable join key: none of {_DEFAULT_ID_CANDIDATES} is present on "
        f"every catalog. Pass id_col= explicitly.")


def _index_by_id(dst_cat, src_cat, cols, *, what="source"):
    """Row in `src_cat` for each row of `dst_cat` (-1 where absent), matched on
    the composite key `cols`. Raises if the SOURCE side has duplicate keys -- a
    duplicate there means the join is ambiguous and would silently pick one row,
    which is the failure mode this whole join exists to avoid."""
    src, dst = _id_key(src_cat, cols).astype(str), _id_key(dst_cat, cols).astype(str)
    u, counts = np.unique(src, return_counts=True)
    dup = u[counts > 1]
    if dup.size:
        raise ValueError(
            f"{what} has {dup.size} duplicated key(s) under "
            f"{'+'.join(cols)}, e.g. {list(dup[:5])}. The join is ambiguous -- "
            f"widen the key (id_col=('FIELD','ID')) or de-duplicate {what}.")
    pos = {k: i for i, k in enumerate(src)}
    idx = np.array([pos.get(k, -1) for k in dst], dtype=int)
    return idx, idx >= 0


def _check_key_value_overlap(dst_cat, src_cat, cols, *, dst_name="halo",
                             src_name="core", max_categorical=20):
    """Warn when a CATEGORICAL part of the join key uses different value
    conventions on the two sides -- 'AEGIS' vs 'EGS', 'cosmos' vs 'COSMOS'.

    A match-rate threshold cannot detect this. With FIELD+ID and a field-name
    mismatch on one field, coverage merely drops by roughly half, which is
    indistinguishable from a genuinely half-covered sample -- so the join looks
    plausible and is quietly wrong for every galaxy in that field. Comparing the
    value SETS is exact and has no false positives: a key part with few distinct
    values (a field name, a catalog name) must use the same vocabulary on both
    sides, whereas an ID part legitimately overlaps only partially and is
    skipped by the max_categorical test.
    """
    for c in cols:
        a = {str(v) for v in _norm_id_col(dst_cat[c])}
        b = {str(v) for v in _norm_id_col(src_cat[c])}
        if len(a) > max_categorical or len(b) > max_categorical:
            continue                       # an ID-like part, partial overlap is normal
        only_b = b - a
        if only_b:
            print(f"  ** WARNING: key column {c!r} uses values on the {src_name} "
                  f"side that never appear on the {dst_name} side: "
                  f"{sorted(only_b)}")
            print(f"     {dst_name}={sorted(a)}  {src_name}={sorted(b)}")
            print(f"     Those rows CANNOT match. If these are the same fields "
                  f"under different names, rename one side before joining -- the "
                  f"coverage drop otherwise looks like ordinary missing data.")


def _fits_safe(col):
    """Cast an object-dtype array to fixed-width unicode so the split catalog
    can be written to FITS. _gather returns object arrays for string columns
    (its "" default is what makes unmatched rows fall into NO group in a later
    split_product_by) and astropy refuses to write those; the cast preserves ""
    so that convention is unaffected."""
    arr = np.asarray(col)
    return arr.astype(str) if arr.dtype == object else arr


def _report_cut_columns(sc, cols=("KEEP", "AGN_FLAG", "CLEAN", "VALID")):
    """Report how the sample-cut booleans relate to each other.

    These all live on the master catalog and are NOT independent -- CLEAN is
    normally KEEP AND NOT AGN_FLAG. A split spec that says 'require CLEAN' and
    one that says 'require KEEP and not AGN_FLAG' should therefore select the
    same galaxies, and if they don't, two splits in the same table are running
    on different samples for a reason nobody wrote down. Printing the relation
    once, here, is cheaper than discovering it from a table that doesn't add up.
    """
    present = [c for c in cols if c in sc.colnames
               and np.asarray(sc[c]).dtype == bool]
    if not present:
        return
    n = len(sc)
    print("  sample-cut columns: "
          + ",  ".join(f"{c}={int(np.asarray(sc[c]).sum())}" for c in present)
          + f"   (of {n})")
    if all(c in present for c in ("KEEP", "AGN_FLAG", "CLEAN")):
        keep = np.asarray(sc["KEEP"], bool)
        agn = np.asarray(sc["AGN_FLAG"], bool)
        clean = np.asarray(sc["CLEAN"], bool)
        expect = keep & ~agn
        bad = int((clean != expect).sum())
        leak = int((clean & agn).sum())
        if bad == 0:
            print("    CLEAN == KEEP & ~AGN_FLAG exactly -- the cuts are "
                  "consistent, use whichever reads better in a split spec.")
        else:
            print(f"    ** CLEAN differs from KEEP & ~AGN_FLAG for {bad} "
                  f"galaxies. The two are NOT interchangeable in a split spec; "
                  f"pick one and state which.")
        if leak:
            print(f"    ** {leak} galaxies are flagged AGN_FLAG yet marked "
                  f"CLEAN -- an AGN-excluded stack built on CLEAN would still "
                  f"contain them.")


def _log_luminosity(flux, z):
    """log10 of a line luminosity [erg/s] from a core flux column (units of
    1e-17 cgs) and Planck18 luminosity distance. NaN where z or flux is
    non-finite or the flux is non-positive (absorbers have no log luminosity)."""
    from .virial import cosmo            # the single cosmology source
    import astropy.units as u
    f = np.asarray(flux, dtype=float) * _LYA_FLUX_TO_CGS
    zz = np.asarray(z, dtype=float)
    out = np.full(f.shape, np.nan)
    ok = np.isfinite(zz) & (zz > 0) & np.isfinite(f) & (f > 0)
    if not ok.any():
        return out
    dl_cm = cosmo.luminosity_distance(zz[ok]).to(u.cm).value
    out[ok] = np.log10(4.0 * np.pi * dl_cm ** 2 * f[ok])
    return out


def _arcsec_to_kpc(size_arcsec, z):
    """Angular size (arcsec) -> proper kpc at each galaxy's own z, off the same
    Planck18 `cosmo` the extraction and virial binning use."""
    from .virial import cosmo
    import astropy.units as u
    s = np.asarray(size_arcsec, dtype=float)
    zz = np.asarray(z, dtype=float)
    out = np.full(s.shape, np.nan)
    ok = np.isfinite(zz) & (zz > 0) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return out
    kpc_per_as = cosmo.angular_diameter_distance(zz[ok]).to(u.kpc).value / 206265.0
    out[ok] = s[ok] * kpc_per_as
    return out


def _log_ratio(a, b):
    """a - b, for two columns ALREADY in log10. NaN-safe."""
    return np.asarray(a, float) - np.asarray(b, float)


# The star-forming main sequence, used by DELTA_MS below. Kept as a named,
# documented function rather than a lambda because WHICH calibration you pick is
# a real choice with consequences, not an implementation detail -- see the
# DELTA_MS entry.
def _ms_logsfr_speagle2014(logM, z):
    """
    log10 SFR [Msun/yr] on the Speagle+2014 main sequence, for log10(M*/Msun)
    `logM` at redshift `z`.

    Speagle+2014 eq. 28, their preferred "time" parameterisation:

        log SFR(M*, t) = (0.84 - 0.026 t) log M*  -  (6.51 - 0.11 t)

    where t is the AGE OF THE UNIVERSE in Gyr at that galaxy's redshift (not
    lookback time), computed off the same Planck18 `cosmo` the extraction and
    virial binning use. Scatter about the relation is ~0.2 dex, which is the
    number to compare any DELTA_MS split against.

    ** IMF CAVEAT, and it is not applied here. ** Speagle+2014 calibrate their
    compilation to a Kroupa (2001) IMF. MOSDEF masses and SFRs are Chabrier
    (2003). The two differ by ~0.03-0.04 dex in BOTH log M* and log SFR, in the
    same direction, so the offset largely cancels in Delta(MS) and is ~5x smaller
    than the relation's own scatter. It is left uncorrected DELIBERATELY -- a
    silent 0.04 dex fudge is worse than a documented one -- but if you ever quote
    an ABSOLUTE Delta(MS) rather than a low/high split, say which IMF.

    Returns NaN wherever logM or z is non-finite.
    """
    from .virial import cosmo            # the single cosmology source
    lm = np.asarray(logM, dtype=float)
    zz = np.asarray(z, dtype=float)
    out = np.full(lm.shape, np.nan)
    ok = np.isfinite(lm) & np.isfinite(zz) & (zz > 0)
    if not ok.any():
        return out
    t = cosmo.age(zz[ok]).to_value("Gyr")
    out[ok] = (0.84 - 0.026 * t) * lm[ok] - (6.51 - 0.11 * t)
    return out


# Derived split variables. Each entry is
#     {"needs": (column names it reads,), "doc": ..., "func": callable(Table)}
# and is SKIPPED (with a note) when any needed column is absent -- so the
# defaults below name the columns this pipeline is known to carry, and the ones
# that depend on catalog columns whose names vary (SFR, sizes) simply do not
# appear until you point them at the right column via a `derived` override.
#
# LOG CONVENTION: MASS_50 is log10(M*/Msun) throughout this pipeline (see
# stack._single_galaxy_rvir_kpc's m_min/m_max=7..11). Every entry below that
# combines mass with an SFR assumes the SFR column is ALSO log10 -- if yours is
# linear, override the entry rather than letting it produce nonsense silently.
SPLIT_DERIVED = OrderedDict([
    ("CORE_LOGL_LYA", {
        "needs": ("CORE_FLUX_MF", "z"),
        "doc": "log10 Lya luminosity [erg/s] from the matched-filter core flux. "
               "Matched-filter, not Gaussian, so it is defined for "
               "non-detections too; NaN for absorbers (non-positive flux).",
        "func": lambda t: _log_luminosity(t["CORE_FLUX_MF"], t["z"]),
    }),
    ("RE_KPC", {
        "needs": ("RE_CIRC_BEST", "z"),
        "doc": "Circularized half-light radius in PROPER KPC, from the "
               "van der Wel+2014 GALFIT size (RE_CIRC_BEST, ARCSEC) and the "
               "Planck18 angular diameter distance. Feeds SIGMA_SFR and the "
               "size split (#11).",
        "func": lambda t: _arcsec_to_kpc(t["RE_CIRC_BEST"], t["z"]),
    }),
    ("SIGMA_SFR", {
        "needs": ("SFR_50", "RE_KPC"),
        "doc": "log10 SFR surface density = log SFR - log(2 pi Re^2) "
               "[Msun/yr/kpc^2]. Split #2. ASSUMES SFR_50 is log10 -- verify "
               "before trusting (a linear SFR_50 makes this silently wrong, not "
               "obviously wrong).",
        "func": lambda t: (np.asarray(t["SFR_50"], float)
                           - np.log10(2.0 * np.pi
                                      * np.asarray(t["RE_KPC"], float) ** 2)),
    }),
    ("BURSTINESS", {
        "needs": ("SFR_HA", "SFR_50"),
        "doc": "log(SFR_Ha / SFR_SED) -- instantaneous (~10 Myr) vs "
               "time-averaged SED SFR, the standard burstiness proxy "
               "(split #10). SKIPS until an SFR_HA column exists: the catalog "
               "carries HA_FLUX, not a dust-corrected Halpha SFR, and turning "
               "one into the other is a real calculation (Balmer decrement, "
               "slit-loss, Kennicutt calibration) that belongs in its own "
               "reviewed step, not in a lambda here. See roadmap Sec 4.2a -- "
               "the aperture correction is the first thing a referee checks.",
        "func": lambda t: _log_ratio(t["SFR_HA"], t["SFR_50"]),
    }),
    ("DELTA_MS", {
        "needs": ("SFR_50", "MASS_50", "z"),
        "doc": "log SFR - log SFR_MS(M*, z): vertical offset from the "
               "Speagle+2014 star-forming main sequence, in dex. Split #1/#2 "
               "variant. The relation's own scatter is ~0.2 dex -- compare any "
               "split against that, not against zero. ASSUMES SFR_50 and MASS_50 "
               "are log10 (same assumption SIGMA_SFR makes). "
               "CALIBRATION IS A CHOICE: Speagle+2014 and Whitaker+2014 disagree "
               "by ~0.2 dex in normalization at z~2 and have different mass "
               "slopes, so they move which galaxies land above vs below the line, "
               "most at the low-mass end. Speagle is used here because it is a "
               "redshift-continuous fit to a 25-study compilation rather than a "
               "binned single-survey fit; swap it by overriding this entry via "
               "build_split_catalog(derived_registry=...). "
               "NOT INDEPENDENT: at fixed mass Delta(MS) is sSFR up to the "
               "relation's mass slope, so delta_ms, ssfr and sfr_massctrl are "
               "close to one axis, not three. See _ms_logsfr_speagle2014 for the "
               "IMF caveat.",
        "func": lambda t: (np.asarray(t["SFR_50"], float)
                           - _ms_logsfr_speagle2014(t["MASS_50"], t["z"])),
    }),
])
# Deliberately NOT here: SSFR. The catalog already carries SSFR_50 (with 16/84
# percentiles); recomputing it as SFR_50 - MASS_50 would produce a second,
# subtly different column and no way to know which a table was built on.


def build_split_catalog(product, core=None, *, id_col=None,
                        core_prefix="CORE_", core_catalog_cols=_CORE_CATALOG_COLS,
                        external=(), derived=True, derived_registry=None,
                        carry=None, verbose=True) -> Table:
    """
    Build the SPLIT CATALOG: one row per galaxy of `product`, carrying every
    quantity you might split on, keyed on the galaxy ID.

    This is Layer A of the subsample-split pipeline (roadmap Sec 3.1) -- the
    piece both the splits program and the photon-budget work consume. It does no
    measurement; it joins already-measured things onto one honest table and
    records where each one is missing.

    Sources, in the order they are merged:
      1. `product.catalog`  -- the halo-side master catalog (z, MASS_50, RA/DEC,
         CLEAN, magnitudes, ...). Carried as-is, unprefixed.
      2. `core`             -- a core.read_core_fits dict (or a stack.
         concat_core_prods of several). EVERY per-galaxy array in its 'results'
         is carried with `core_prefix`, plus `core_catalog_cols` from its
         catalog, plus a CORE_COVERED boolean. Joined on `id_col` EXACTLY.
      3. `external`         -- foreign catalogs with no shared ID, matched by
         sky position (this module's own _match_radec). See below.
      4. `derived`          -- computed columns from SPLIT_DERIVED.

    product : GalaxyProduct (or a bare astropy Table -- its catalog).
    core    : core product dict, or None to build a catalog-only split catalog.
    id_col  : the exact join key, present on BOTH sides -- a single column name
              or a COMPOSITE like ('FIELD', 'ID'). None (default) auto-selects
              the first of ('FIELD','ID'), ('ID',) that exists on both sides and
              is unique on both. Duplicates under the chosen key raise.

              MOSDEF IDs are scoped to their field: AEGIS 5814 and COSMOS 5814
              are different galaxies at unrelated coordinates, and the catalog's
              ID column equals ID_V21 in AEGIS but ID_V41 in COSMOS -- so no
              single ID column is a survey-wide key and the auto-selection will
              normally land on FIELD+ID. It says which key it chose.
    external : iterable of specs, each a dict
                 {"table": Table, "columns": [...], "radius_arcsec": 0.5,
                  "prefix": "", "names": {src: dest}, "ra_col": "RA",
                  "dec_col": "DEC", "ext_ra_col": None, "ext_dec_col": None}
               'table' and 'columns' are required; the rest have the shown
               defaults. Positional matching is correct HERE (no shared ID) and
               wrong for `core` (shared ID) -- that asymmetry is the point.

               `names` renames on the way in, and you will need it: foreign
               catalogs reuse generic column names, and the Chartab density
               tables carry their own FIELD, which is part of the join key here.
               A column that would overwrite an existing one RAISES rather than
               clobbering. For Chartab:

                 external=[{"table": ext, "columns": ["DELTA", "FIELD"],
                            "names": {"FIELD": "DENS_FIELD"},
                            "radius_arcsec": 0.5}]

               which produces the same DELTA / DENS_FIELD columns
               attach_overdensity would, so split_by_overdensity can consume
               them directly (pass ext=None to make it reuse these rather than
               re-matching).
    derived : True to apply SPLIT_DERIVED, False to skip, or a list of entry
              names to apply only those.
    derived_registry : override/extend SPLIT_DERIVED, e.g.
              dict(multicat.SPLIT_DERIVED, SSFR={...}) to point SSFR at your own
              SFR column name or fix the log convention.
    carry   : restrict the halo-side columns carried to this list (plus id_col).
              None -> carry all of them.

    Returns an astropy Table with provenance in .meta (source shapes, id_col,
    which derived entries ran, which were skipped and why). Write it with
    write_split_catalog; attach it to any product with attach_split_catalog.

    Coverage bookkeeping, and why it matters
    -----------------------------------------
    CORE_COVERED is True only where the ID matched a core row. Galaxies without
    core coverage keep their catalog quantities (mass, z, environment) and get
    NaN for every core-derived quantity. That asymmetry is REAL -- it means an
    L_Lya split and a mass split run on different samples -- and roadmap Sec
    3.3b requires it be visible rather than absorbed into an empty string.
    split_catalog_report() quantifies it, including pairwise intersections.
    """
    cat_in = product.catalog if hasattr(product, "catalog") else product
    core_cat = core["catalog"] if core is not None else None

    if verbose:
        print("build_split_catalog:")
    id_cols = _resolve_id_cols(id_col, [cat_in, core_cat], verbose=verbose)

    keep_cols = (list(dict.fromkeys(id_cols + list(carry)))
                 if carry else list(cat_in.colnames))
    missing = [c for c in keep_cols if c not in cat_in.colnames]
    if missing:
        raise KeyError(f"carry= requested columns not in the halo catalog: {missing}")

    sc = Table()
    for c in keep_cols:
        sc[c] = np.asarray(cat_in[c])
    ngal = len(sc)

    # the halo side's own keys must be unique or nothing downstream is well posed
    _index_by_id(sc, sc, id_cols, what="the halo catalog")

    meta = {
        "SPLITCAT": "Layer A split catalog (multicat.build_split_catalog)",
        "IDCOL": ",".join(id_cols),
        "NGAL": int(ngal),
        "NCORE": 0,
        "DERIVED": "",
        "SKIPPED": "",
    }
    if getattr(product, "header", None) is not None:
        for k in ("EXTRSIG", "SOURCE", "SOURCEID", "CATALOG", "FIELD"):
            if k in product.header:
                meta[f"H_{k}"[:8]] = str(product.header[k])

    if verbose:
        print(f"build_split_catalog: {ngal} galaxies, {len(keep_cols)} halo-side columns")

    # ---- 2. the core product, joined on ID ----------------------------------
    if core is not None:
        idx, matched = _index_by_id(sc, core_cat, id_cols,
                                    what="the core catalog")
        n_core = len(core_cat)
        if verbose:
            _check_key_value_overlap(sc, core_cat, id_cols,
                                     dst_name="halo", src_name="core")
        sc[f"{core_prefix}COVERED"] = matched

        results = core.get("results", {}) or {}
        n_carried = 0
        for key, val in results.items():
            arr = np.asarray(val)
            if arr.ndim != 1 or arr.shape[0] != n_core:
                continue                      # scalar metadata, not a galaxy array
            name = f"{core_prefix}{key.upper()}"
            sc[name] = _fits_safe(_gather(arr, idx, matched, ngal,
                                          fill=False if arr.dtype == bool else None))
            n_carried += 1
        for c in core_catalog_cols:
            if c in core_cat.colnames:
                col = np.asarray(core_cat[c])
                sc[f"{core_prefix}{c}"] = _fits_safe(_gather(
                    col, idx, matched, ngal,
                    fill=False if col.dtype == bool else None))
                n_carried += 1

        meta["NCORE"] = int(matched.sum())
        if verbose:
            unused = n_core - int(np.unique(idx[matched]).size)
            print(f"  core join on {'+'.join(id_cols)}: "
                  f"{int(matched.sum())}/{ngal} galaxies "
                  f"covered ({int(matched.sum())/max(ngal,1):.0%}); "
                  f"{n_carried} core columns carried with prefix {core_prefix!r}")
            if unused:
                print(f"  note: {unused} of the {n_core} core rows have no halo "
                      f"counterpart (core coverage outside the halo sample)")

    # ---- 3. external catalogs, matched by sky position ----------------------
    for spec in external:
        ext = spec["table"] if hasattr(spec["table"], "colnames") else Table(spec["table"])
        cols = list(spec["columns"])
        # `or`-style fallbacks, not dict.get defaults: an explicit
        # {"ext_ra_col": None} means "use the default", but .get(k, default)
        # returns None whenever the KEY EXISTS -- so a spec that spells out its
        # optional fields (which is the readable way to write one) would pass
        # None straight through to the table lookup.
        rad = float(spec.get("radius_arcsec") or 0.5)
        pre = spec.get("prefix") or ""
        names = spec.get("names") or {}
        ra_c = spec.get("ra_col") or "RA"
        dec_c = spec.get("dec_col") or "DEC"
        e_ra = spec.get("ext_ra_col") or ra_c
        e_dec = spec.get("ext_dec_col") or dec_c

        # Destination names first, and REFUSE to clobber. Foreign catalogs
        # routinely reuse generic column names -- the Chartab density tables
        # carry their own FIELD, and FIELD is part of the join key here, so a
        # silent overwrite would corrupt the key itself and every later
        # attach_split_catalog with it. attach_overdensity renames that column
        # to DENS_FIELD for exactly this reason; do the same via `names`.
        dest = {c: f"{pre}{names.get(c, c)}" for c in cols}
        clash = {c: d for c, d in dest.items() if d in sc.colnames}
        if clash:
            key_clash = {c: d for c, d in clash.items() if d in id_cols}
            raise ValueError(
                f"external column(s) would overwrite existing split-catalog "
                f"columns: {clash}."
                + (f"  {list(key_clash.values())} are part of the JOIN KEY -- "
                   f"overwriting them would corrupt every later join."
                   if key_clash else "")
                + f"  Rename them in the spec, e.g. "
                  f"names={{{list(clash)[0]!r}: 'DENS_{list(clash)[0]}'}}, "
                  f"or set a prefix=.")

        idx_e, sep_e = _match_radec(
            np.asarray(sc[ra_c], float), np.asarray(sc[dec_c], float),
            np.asarray(ext[e_ra], float), np.asarray(ext[e_dec], float), rad)
        m_e = idx_e >= 0
        for c in cols:
            sc[dest[c]] = _fits_safe(_gather(ext[c], idx_e, m_e, ngal, None))
        if verbose:
            print(f"  external -> {', '.join(dest.values())}: "
                  f"{int(m_e.sum())}/{ngal} matched within {rad:g}\" "
                  f"(median sep {np.nanmedian(sep_e[m_e]) if m_e.any() else np.nan:.2f}\")")

    # ---- 3b. how the sample cuts relate to each other -----------------------
    if verbose:
        _report_cut_columns(sc)

    # ---- 4. derived columns -------------------------------------------------
    reg = derived_registry or SPLIT_DERIVED
    if derived is not False:
        wanted = list(reg) if derived is True else list(derived)
        ran, skipped = [], []
        for name in wanted:
            entry = reg.get(name)
            if entry is None:
                skipped.append(f"{name} (not in registry)")
                continue
            absent = [c for c in entry["needs"] if c not in sc.colnames]
            if absent:
                skipped.append(f"{name} (needs {'+'.join(absent)})")
                continue
            sc[name] = np.asarray(entry["func"](sc), dtype=float)
            ran.append(name)
        meta["DERIVED"] = ",".join(ran)
        meta["SKIPPED"] = "; ".join(skipped)
        if verbose:
            print(f"  derived: {', '.join(ran) if ran else '(none)'}")
            if skipped:
                print(f"  derived skipped: {'; '.join(skipped)}")

    sc.meta.update(meta)
    return sc


def attach_split_catalog(product: GalaxyProduct, splitcat, *, columns=None,
                         id_col=None, overwrite=False, verbose=True) -> GalaxyProduct:
    """Return a COPY of `product` whose catalog carries the split catalog's
    columns, reindexed onto its galaxy axis by ID.

    This is the counterpart to build_split_catalog: the table is built once and
    keyed on ID, and this puts it on whatever product you are holding -- mega,
    a single field, or a product from a later re-extraction with a different row
    order. Only the catalog is copied (galaxy-axis arrays are shared), like
    add_matched_column, so it is cheap and the result is a normal product you
    can hand straight to split_product_by.

    columns   : which split-catalog columns to attach (default: all of them that
                aren't already on the product's catalog).
    id_col    : join key, single or composite. Defaults to the key the split
                catalog was BUILT with (splitcat.meta['IDCOL']), so the two ends
                of the join cannot drift apart.
    overwrite : allow attached columns to replace same-named catalog columns.
                Off by default -- a silent overwrite of MASS_50 or z with a
                stale copy is exactly the bug this is trying to prevent.
    """
    id_cols = _as_key_list(id_col) or [
        c for c in str(splitcat.meta.get("IDCOL", "ID")).split(",") if c]
    cat = product.catalog
    for side, tab in (("product", cat), ("split", splitcat)):
        miss = [k for k in id_cols if k not in tab.colnames]
        if miss:
            raise KeyError(f"join key column(s) {miss} not in the {side} catalog.")

    idx, matched = _index_by_id(cat, splitcat, id_cols, what="the split catalog")

    # columns=None means "everything NEW". Most of the split catalog was copied
    # FROM this product's catalog in the first place (z, MASS_50, FIELD, ...), so
    # attaching all of it would clash on nearly every column and refuse to do
    # anything -- the useful default is the columns the product does not already
    # have: the CORE_* block, the externals, the derived ones. An EXPLICIT
    # columns= list still raises on a clash, because there the caller named the
    # column and a silent overwrite would be a surprise.
    skipped = []
    if columns is None:
        want = [c for c in splitcat.colnames
                if c not in id_cols and c not in cat.colnames]
        skipped = [c for c in splitcat.colnames
                   if c not in id_cols and c in cat.colnames]
    else:
        want = list(columns)
        clash = [c for c in want if c in cat.colnames]
        if clash and not overwrite:
            raise ValueError(
                f"these split-catalog columns already exist on the product "
                f"catalog: {clash}. Drop them from columns=, or pass "
                f"overwrite=True if replacing them is what you mean.")

    out = cat.copy()
    for c in want:
        col = np.asarray(splitcat[c])
        out[c] = _fits_safe(_gather(col, idx, matched, len(cat),
                                    fill=False if col.dtype == bool else None))
    if verbose:
        n = int(matched.sum())
        print(f"attach_split_catalog: {n}/{len(cat)} galaxies matched on "
              f"{'+'.join(id_cols)}; attached {len(want)} columns"
              + (f" ({len(skipped)} already on the product catalog, left alone)"
                 if skipped else ""))
        _check_key_value_overlap(cat, splitcat, id_cols,
                                 dst_name="product", src_name="split catalog")
        if n < len(cat):
            print(f"  ** {len(cat)-n} product galaxies are NOT in the split "
                  f"catalog -- they get fill values and will drop out of every "
                  f"split that requires one of these columns.")
    return replace(product, catalog=out)


def write_split_catalog(path, splitcat, overwrite=True, verbose=True):
    """Write a split catalog to FITS, provenance and all. Goes to OUTPUTS, not
    the extraction cache -- it is cheap to rebuild and is not signature-keyed to
    an extraction (that keying belongs on the split RESULTS, Layer C, roadmap
    Sec 3.3a)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    splitcat.write(path, format="fits", overwrite=overwrite)
    if verbose:
        print(f"wrote split catalog: {path}  ({len(splitcat)} rows, "
              f"{len(splitcat.colnames)} columns)")
    return path


def read_split_catalog(path, verbose=True) -> Table:
    """Read a split catalog written by write_split_catalog.

    FITS stores string columns as bytes ('S8'), so every text column is decoded
    back to unicode here. Without that, CORE_LABEL comes back as b'LAE' and a
    later split_product_by(..., groups=[['LAE']]) is comparing strings to bytes
    -- it would return empty groups with no error, which is the worst kind of
    failure. Same reason _to_str exists for the Chartab tables below.
    """
    t = Table.read(path, format="fits")
    for c in t.colnames:
        if np.asarray(t[c]).dtype.kind == "S":
            t[c] = _to_str(t[c])
    if verbose:
        print(f"read split catalog: {path}  ({len(t)} rows, "
              f"{len(t.colnames)} columns, id_col={t.meta.get('IDCOL','ID')})")
    return t


def split_catalog_report(splitcat, columns=None, *, intersect=True, verbose=True):
    """
    Coverage audit of a split catalog: how many galaxies actually have each
    quantity, and how much sample is left when several are required at once.

    This is the number roadmap Sec 3.3b turns on. Splitting on mass uses one
    sample; splitting on core Lya luminosity uses a smaller one; a table whose
    rows come from different samples is comparing composition as much as
    physics. Run this BEFORE scoping a batch of splits, not after.

    columns   : the split variables you plan to use (default: every non-ID,
                non-boolean column).
    intersect : also report the size of the common sample -- galaxies with ALL
                of `columns` available. That is the sample a directly comparable
                table should be built on.

    Returns dict with 'per_column' {name: n_available} and, if intersect,
    'n_common' and 'common_mask'.
    """
    id_cols = [c for c in str(splitcat.meta.get("IDCOL", "ID")).split(",") if c]
    if columns is None:
        columns = [c for c in splitcat.colnames
                   if c not in id_cols and np.asarray(splitcat[c]).dtype != bool]
    n = len(splitcat)

    def _avail(c):
        col = np.asarray(splitcat[c])
        if col.dtype == bool:
            return col
        if np.issubdtype(col.dtype, np.number):
            return np.isfinite(col.astype(float))
        return np.array([str(v).strip() != "" for v in col])

    masks = {c: _avail(c) for c in columns}
    per_column = {c: int(m.sum()) for c, m in masks.items()}

    out = {"n_total": n, "per_column": per_column}
    if intersect and masks:
        common = np.logical_and.reduce(list(masks.values()))
        out["n_common"] = int(common.sum())
        out["common_mask"] = common

    if verbose:
        print(f"split catalog coverage  ({n} galaxies)")
        print(f"  {'column':<28} {'N':>6}  {'frac':>6}")
        for c in sorted(per_column, key=lambda k: -per_column[k]):
            k = per_column[c]
            print(f"  {c:<28} {k:>6}  {k/max(n,1):>5.0%}")
        if intersect and masks:
            print(f"  {'-- ALL of the above':<28} {out['n_common']:>6}  "
                  f"{out['n_common']/max(n,1):>5.0%}")
            print("  (that last row is the sample a directly comparable table "
                  "should use;\n   see workstream-roadmap.md Sec 3.3b)")
    return out


# ---------------------------------------------------------------------
# 5. STACK / MEASURE each, then COMPARE
# ---------------------------------------------------------------------
def stack_each(config, products, keep_cube=True, verbose=False):
    """build_stacks on each product in a labelled dict -> {label: stacks}.

    One config drives all of them (z_col / mass_col / rest params / bin_mode), so
    the products must share that geometry -- which they do if they came from
    split_product_by or compatible extractions."""
    return OrderedDict(
        (label, build_stacks(config, p, keep_cube=keep_cube))
        for label, p in products.items()
    )


def stack_id_list(config, product, ids, *, id_col=None, keep_cube=True,
                  weights=None, verbose=True):
    """
    Coadd a hand-picked list of galaxy IDs into one mini halo profile.

    Thin wrapper: _slice_product (pick the rows) -> build_stacks (the same
    Stage-2 coadd stack_each/run_stack use) -- the ID-list front door for
    turning a selection.top_sn_galaxies shortlist (via selection.id_list) into
    a stack you can run plot_stack_panels / plot_radius_velocity_map on right
    away.

        ids  = selection.id_list(sub)[:5]     # from a core-product S/N triage
        mini = stack_id_list(cfg, prod, ids)  # prod = the usual BINNED product
        plot_stack_panels(mini, method="biweight")

    ids    : list/array of catalog IDs to include (matched as strings, so 123
             and '123' both work -- same convention as core.select_core_by_id).
    id_col : catalog column to match on; defaults to config.id_col.

    Returns the same dict shape as run_stack/build_stacks (rest_wave, r_edges,
    stacks{method:{flux,err}}, z_median, mass_median, unit_info, VR_biweight_v,
    and cube_flux/cube_err if keep_cube) -- so every existing stack plotter/
    measurer works on it unchanged. Raises if none of the requested IDs are
    found; warns (not fails) on partial misses.
    """
    id_col = id_col or config.id_col
    cat = product.catalog
    if id_col not in cat.colnames:
        raise KeyError(f"id_col {id_col!r} not in catalog columns {list(cat.colnames)}")

    want = [str(v) for v in np.atleast_1d(ids)]
    col = np.asarray(cat[id_col]).astype(str)
    mask = np.isin(col, want)
    found = set(col[mask])
    missing = [w for w in want if w not in found]

    if not mask.any():
        raise ValueError(f"none of the requested IDs were found in column {id_col!r}.")
    if verbose:
        print(f"stack_id_list: matched {int(mask.sum())}/{len(want)} requested IDs"
              + (f"; missing {missing}" if missing else ""))

    sub = _slice_product(product, mask)
    return build_stacks(config, sub, keep_cube=keep_cube, weights=weights)


def measure_each(config, stacks_dict):
    """run_measure on each stacks dict -> {label: boot}. Requires keep_cube=True
    stacks. Imported lazily so importing multicat doesn't require measure."""
    from .measure import measure_all_bins
    return OrderedDict(
        (label, measure_all_bins(config, st)) for label, st in stacks_dict.items()
    )


def compare_centroids(boots, bin_mode=None, VR_biweight_v=None,
                      vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.6, 4.9),
                      ylims=(-200, 200), xlims=None, jitter=0.04, title=None,
                      save_fig=False):
    """
    Overlay centroid-vs-radius for several labelled samples (e.g. low-z vs
    high-z). Thin wrapper around plotting.plot_centroid_comparison -- the figure
    itself lives in plotting.py so all radial plots share one implementation and
    layout. Kept here for API convenience from a multi-catalog workflow.
    """
    from .plotting import plot_centroid_comparison
    return plot_centroid_comparison(
        boots, radial_bins=None, VR_biweight_v=VR_biweight_v, bin_mode=bin_mode,
        vr_ticks=vr_ticks, figsize=figsize, ylims=ylims, xlims=xlims,
        jitter=jitter, title=title or "Centroid vs. radius by sample",
        save_fig=save_fig,
    )


# ---------------------------------------------------------------------
# Chartab et al. (2020) CANDELS overdensity catalog -- attach + split.
# Folded in from the standalone environment.py (2026-07-21; that module
# had grown to only 4 functions, all of them thin, catalog-specific
# wrappers around add_matched_column/split_product_by directly above, so
# it didn't earn its keep as a separate file). Everything below is
# Chartab-specific (column names, per-field/mass-matched cell splitting);
# the generic cross-match/split machinery it calls lives in this module.
#
# Typical use (notebook):
#   from utils_lya_halo import multicat
#   ext = multicat.load_overdensity("catalogs/overdensity_cosmos.fit",
#                                   "catalogs/overdensity_egs.fit")
#   groups = multicat.split_by_overdensity(mega, ext)   # per-field median split
#   stacks_lo = stack.build_stacks(cfg_prod, groups["low_density"],  keep_cube=True)
#   stacks_hi = stack.build_stacks(cfg_prod, groups["high_density"], keep_cube=True)
# ---------------------------------------------------------------------
def _to_str(col) -> np.ndarray:
    """Normalize a possibly-bytes column (FITS 'bytes6') to unicode str, and
    STRIP surrounding whitespace.

    The stripping is load-bearing, not cosmetic: FITS fixed-width character
    columns pad to the column width, so the Chartab EGS table's field name comes
    back as 'EGS   ' rather than 'EGS'. Any grouping or equality test on that
    value then fails against every unpadded 'EGS' in the pipeline, silently
    putting those galaxies in their own group or in none at all.
    """
    arr = np.asarray(col)
    if arr.dtype.kind == "S":
        arr = np.char.decode(arr, "utf-8")
    return np.char.strip(arr.astype(str))


def load_overdensity(*paths, ra_col="RAJ2000", dec_col="DEJ2000",
                     delta_col="delta", field_col="Field",
                     z_col="z", id_col="ID", verbose=True) -> Table:
    """Read one or more Chartab+2020 density `.fit` tables and return a single
    astropy Table with columns renamed to the pipeline convention:

        RA, DEC, DELTA, FIELD, ZCAT, IDCAT

    Pass the COSMOS and EGS files in either order; they're stacked. RA/DEC are
    named to match this module's positional-match defaults, so the result
    drops straight into add_matched_column / match_catalog.
    """
    tabs = []
    for p in paths:
        t = Table.read(p, format="fits") if isinstance(p, str) else Table(p)
        out = Table()
        out["RA"]    = np.asarray(t[ra_col], float)
        out["DEC"]   = np.asarray(t[dec_col], float)
        out["DELTA"] = np.asarray(t[delta_col], float)
        out["FIELD"] = _to_str(t[field_col]) if field_col in t.colnames else ""
        if z_col in t.colnames:
            out["ZCAT"] = np.asarray(t[z_col], float)
        if id_col in t.colnames:
            out["IDCAT"] = _to_str(t[id_col])
        tabs.append(out)
        if verbose:
            fields = sorted(set(_to_str(out["FIELD"]))) if len(out) else []
            print(f"  loaded {len(out):>6d} rows from {p}  fields={fields}")
    ext = vstack(tabs) if len(tabs) > 1 else tabs[0]
    if verbose:
        print(f"  -> {len(ext)} total tracer rows")
    return ext


def attach_overdensity(product: GalaxyProduct, ext: Table, *,
                       radius_arcsec=0.5, verbose=True) -> GalaxyProduct:
    """Return a COPY of `product` whose catalog carries DELTA and DENS_FIELD,
    matched from `ext` by sky position (add_matched_column). Unmatched
    galaxies get DELTA=NaN and DENS_FIELD="".
    """
    prod = add_matched_column(product, ext, "DELTA",
                              radius_arcsec=radius_arcsec, verbose=verbose)
    prod = add_matched_column(prod, ext, "FIELD", new_name="DENS_FIELD",
                              radius_arcsec=radius_arcsec, verbose=False)
    return prod


def split_by_overdensity(product: GalaxyProduct, ext: Table = None, *,
                         per_field=True, mass_match=False, mass_col="MASS_50",
                         n_mass_bins=5, radius_arcsec=0.5,
                         labels=("low_density", "high_density"),
                         verbose=True):
    """Attach overdensity and split the product into low- vs. high-density
    sub-products at the median, controlling for field and (optionally) mass.

    The sample is divided into CELLS; within each cell the matched galaxies are
    median-split on delta into "low"/"high"; then all low cells are pooled and
    all high cells are pooled. Because each cell contributes ~half to each side,
    the pooled halves share the cells' field/mass composition by construction.

    per_field=True  (default): one cell per field -> each field split at ITS OWN
        median delta, so the two fields' differing depth/normalization aren't
        mixed. per_field=False: a single pooled cell (global median).

    mass_match=True: further subdivide each field into `n_mass_bins` bins of
        `mass_col`, using quantile edges (equal count per bin), and median-split
        delta within each (field x mass-bin) cell. This equalizes the stellar-
        mass distribution of the low and high halves, breaking the
        mass-environment degeneracy (see docs/research-notes.md). Galaxies with
        non-finite mass are dropped. If a field has too few matched galaxies to
        bin `n_mass_bins` ways (< 2 per bin), it falls back to a single mass bin
        for that field.

    ext=None: reuse DELTA / DENS_FIELD already on the product's catalog instead
        of re-matching -- which is what you want once the split catalog carries
        them (build_split_catalog(external=[...]) -> attach_split_catalog).
        Re-matching there would be worse than redundant: a second positional
        match at a different radius_arcsec silently produces a DIFFERENT DELTA
        for the same galaxy than the one recorded in the split catalog, so the
        environment split and the environment column in the results table would
        disagree with no way to tell which is which. Passing ext re-matches and
        warns if the columns were already present.

    Returns OrderedDict{label: GalaxyProduct}; unmatched / non-finite / <2-per-
    cell galaxies fall into no group and are dropped.
    """
    have = all(c in product.catalog.colnames for c in ("DELTA", "DENS_FIELD"))
    if ext is None:
        if not have:
            raise ValueError(
                "ext=None means 'reuse the DELTA / DENS_FIELD already on the "
                "catalog', but they are not there. Either pass the Chartab "
                "table as `ext`, or attach them first "
                "(build_split_catalog(..., external=[...]) then "
                "attach_split_catalog).")
        prod = product
        if verbose:
            print("  using DELTA / DENS_FIELD already on the catalog (no re-match)")
    else:
        if have and verbose:
            print("  note: DELTA / DENS_FIELD were ALREADY on the catalog and are "
                  "being overwritten by a fresh positional match. If they came "
                  "from a split catalog, pass ext=None instead -- two matches at "
                  "different radii give two different DELTAs for the same galaxy.")
        prod = attach_overdensity(product, ext, radius_arcsec=radius_arcsec,
                                  verbose=verbose)
    delta = np.asarray(prod.catalog["DELTA"], float)
    field = _to_str(prod.catalog["DENS_FIELD"])
    dens_class = np.full(len(delta), "", dtype=object)   # "" -> dropped by split

    # outer grouping: one cell per field, or a single pooled cell
    if per_field:
        outer = [(f, field == f) for f in sorted(set(field) - {""})]
    else:
        outer = [("all", field != "")]

    mass = np.asarray(prod.catalog[mass_col], float) if mass_match else None

    def _tag(cell_mask, label):
        in_c = cell_mask & np.isfinite(delta)
        if int(in_c.sum()) < 2:
            if verbose and in_c.any():
                print(f"  {label}: <2 galaxies, skipped")
            return
        med = np.median(delta[in_c])
        lo, hi = in_c & (delta <= med), in_c & (delta > med)
        dens_class[lo] = "low"
        dens_class[hi] = "high"
        if verbose:
            print(f"  {label}: n={int(in_c.sum())} med(delta)={med:+.3f} "
                  f"-> {int(lo.sum())} low / {int(hi.sum())} high")

    for fname, fmask in outer:
        if not mass_match:
            _tag(fmask, fname)
            continue
        good = fmask & np.isfinite(delta) & np.isfinite(mass)
        if int(good.sum()) < 2 * n_mass_bins:          # too few to bin finely
            _tag(fmask, f"{fname} (mass unbinned)")
            continue
        edges = np.unique(np.nanquantile(mass[good],
                                         np.linspace(0.0, 1.0, n_mass_bins + 1)))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        for i in range(len(edges) - 1):
            cell = fmask & (mass >= edges[i]) & (mass < edges[i + 1])
            _tag(cell, f"{fname} logM[{edges[i]:.2f},{edges[i + 1]:.2f})")

    cat = prod.catalog.copy()
    cat["DENS_CLASS"] = np.asarray(dens_class, dtype=str)
    prod = replace(prod, catalog=cat)

    if verbose:
        dc = np.asarray(dens_class)
        print(f"  pooled: {int((dc == 'low').sum())} low / "
              f"{int((dc == 'high').sum())} high "
              f"({int((dc == '').sum())} dropped)")

    return split_product_by(
        prod, "DENS_CLASS",
        groups=[["low"], ["high"]], labels=list(labels), verbose=verbose)
