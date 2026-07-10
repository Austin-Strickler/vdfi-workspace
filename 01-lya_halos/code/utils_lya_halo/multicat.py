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
    "stack_each",
    "stack_id_list",
    "measure_each",
    "compare_centroids",
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
