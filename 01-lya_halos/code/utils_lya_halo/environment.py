"""
environment.py -- attach a local overdensity to a galaxy product and split
low- vs. high-density, using the published Chartab et al. (2020) CANDELS
density-field catalog (VizieR J/ApJ/890/7).

The catalog is adopted as-is -- no estimator is built here. Its `delta` column
is the per-galaxy overdensity, already normalized within each redshift slice.
We link it onto our galaxy axis positionally (RA/Dec), NOT by ID: the catalog's
`ID` is the CANDELS photometric ID (Nayyeri+2017 COSMOS / Stefanon+2017 EGS),
which is a different numbering than MOSDEF's 3D-HST/Skelton+2014 IDs.

Typical use (notebook):

    from utils_lya_halo import environment as env, stack
    ext = env.load_overdensity("catalogs/overdensity_cosmos.fit",
                               "catalogs/overdensity_egs.fit")
    groups = env.split_by_overdensity(mega, ext)          # per-field median split
    stacks_lo = stack.build_stacks(cfg_prod, groups["low_density"],  keep_cube=True)
    stacks_hi = stack.build_stacks(cfg_prod, groups["high_density"], keep_cube=True)

Design mirrors multicat: the split is pure Stage-2 (no re-extraction), unmatched
galaxies fall into no group and are dropped, and the result is a plain
OrderedDict{label: GalaxyProduct} ready for build_stacks / measure.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from astropy.table import Table, vstack

from .io import GalaxyProduct
from . import multicat


def _to_str(col) -> np.ndarray:
    """Normalize a possibly-bytes column (FITS 'bytes6') to unicode str."""
    arr = np.asarray(col)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    return arr.astype(str)


def load_overdensity(*paths, ra_col="RAJ2000", dec_col="DEJ2000",
                     delta_col="delta", field_col="Field",
                     z_col="z", id_col="ID", verbose=True) -> Table:
    """Read one or more Chartab+2020 density `.fit` tables and return a single
    astropy Table with columns renamed to the pipeline convention:

        RA, DEC, DELTA, FIELD, ZCAT, IDCAT

    Pass the COSMOS and EGS files in either order; they're stacked. RA/DEC are
    named to match multicat's positional-match defaults, so the result drops
    straight into add_matched_column / match_catalog.
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
    matched from `ext` by sky position (multicat.add_matched_column). Unmatched
    galaxies get DELTA=NaN and DENS_FIELD="".
    """
    prod = multicat.add_matched_column(product, ext, "DELTA",
                                       radius_arcsec=radius_arcsec, verbose=verbose)
    prod = multicat.add_matched_column(prod, ext, "FIELD", new_name="DENS_FIELD",
                                       radius_arcsec=radius_arcsec, verbose=False)
    return prod


def split_by_overdensity(product: GalaxyProduct, ext: Table, *,
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

    Returns OrderedDict{label: GalaxyProduct}; unmatched / non-finite / <2-per-
    cell galaxies fall into no group and are dropped.
    """
    from dataclasses import replace

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

    return multicat.split_product_by(
        prod, "DENS_CLASS",
        groups=[["low"], ["high"]], labels=list(labels), verbose=verbose)
