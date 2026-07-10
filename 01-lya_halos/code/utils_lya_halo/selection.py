"""
selection.py -- interactively pick galaxies out of a product, cleanly.

Stage 2 deals in GalaxyProducts (io.GalaxyProduct): a galaxy axis on
spec/err/nfib/background and a CATALOG table in the SAME row order. multicat
already SPLITS a product by a catalog COLUMN (low-z vs high-z, etc.) and
COMBINES several catalogs into one product. This module is the companion you
reach for when the thing you want to select on is not (yet) a plain column:

  * CONE / CRITERIA search -- "give me the galaxies within R of (RA, DEC)", or
    "the ones matching this expression / boolean mask". Returns a small
    Selection object you can inspect (which galaxies, how far, the matched
    catalog rows) AND whose `.product` is a ready-to-stack sub-product.
  * DERIVED quantities -- a property that is a FUNCTION of columns rather than a
    column itself (the canonical case: sSFR = SFR / M*). `with_derived` attaches
    it as a temporary catalog column so you can hand the result straight to
    multicat.split_product_by; `split_by_derived` does both in one call.
  * SPLIT BY YOUR OWN MASK -- you built a boolean (or categorical) array in the
    notebook, possibly using a second catalog you cross-matched yourself. As long
    as it is aligned to THIS product's galaxy axis (same length, same order),
    split_by_mask turns it into the same {label: GalaxyProduct} dict the rest of
    the multi-catalog workflow consumes (stack_each / measure_each / compare_*).

Everything here returns the same currency as multicat -- a GalaxyProduct, or an
OrderedDict{label: GalaxyProduct} -- so selections drop straight into
multicat.stack_each / measure_each / compare_centroids with no glue code.

Pure numpy + the galaxy product at module load (matplotlib and multicat are
imported lazily, only inside the helpers that need them), so
`from utils_lya_halo import selection` stays cheap. Not added to
utils_lya_halo/__init__ for the same reason multicat isn't.

    from utils_lya_halo import selection
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace as _dc_replace
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import numpy as np

from .io import GalaxyProduct

if TYPE_CHECKING:
    from astropy.table import Table
    from .config import PipelineConfig

__all__ = [
    "Selection",
    "slice_product",
    "cone_search",
    "select_by",
    "derive",
    "with_column",
    "with_derived",
    "split_by_derived",
    "split_by_mask",
    "plot_sky",
    "plot_split_hist",
    "resolve_core_sn_fields",
    "top_sn_galaxies",
    "id_list",
]


# =====================================================================
# low-level helpers
# =====================================================================
def _as_float(col) -> np.ndarray:
    """A catalog column as a plain float ndarray, masked entries -> NaN.

    Robust to astropy MaskedColumn (e.g. columns created by an outer-join
    combine when one catalog lacks the column) and to plain ndarrays/lists.
    """
    if hasattr(col, "filled"):                 # astropy MaskedColumn
        col = col.filled(np.nan)
    return np.asarray(col, dtype=float)


def _col_value(col):
    """A catalog column for use in an expression namespace.

    Numeric columns become float arrays (masked entries -> NaN); NON-numeric
    columns (e.g. the string 'SUBSET'/'FIELD'/'CATALOG' columns a combine leaves
    behind) are returned as-is rather than forced to float. That keeps a string
    column from breaking the whole namespace AND lets expressions compare them,
    e.g. "(SUBSET == 'cat1') & (z < 2.5)".
    """
    is_masked = hasattr(col, "filled")
    raw = np.asarray(col)                       # underlying data (mask ignored)
    if np.issubdtype(raw.dtype, np.number):
        if is_masked:
            col = col.filled(np.nan)
        return np.asarray(col, dtype=float)
    if is_masked:
        col = col.filled()                      # dtype-appropriate fill (strings ok)
    return np.asarray(col)


def _expr_namespace(catalog) -> dict:
    """Build the eval namespace: every column by name (numeric->float, else
    raw) plus ``np``. Shared by derive() and select_by()."""
    ns = {name: _col_value(catalog[name]) for name in catalog.colnames}
    ns["np"] = np
    return ns


def _sep_arcsec(ra0: float, dec0: float, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Great-circle separation (arcsec) from one center to many points.

    Haversine, so it is exact at any separation (not a small-angle / flat-sky
    approximation). RA/DEC in degrees.
    """
    r0, d0 = np.radians(ra0), np.radians(dec0)
    r, d = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    a = np.sin((d - d0) / 2) ** 2 + np.cos(d0) * np.cos(d) * np.sin((r - r0) / 2) ** 2
    ang = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))      # radians
    return np.degrees(ang) * 3600.0


def _coerce_mask(product: GalaxyProduct, mask) -> np.ndarray:
    """Validate/normalize a user mask against THIS product's galaxy axis.

    Accepts a boolean array of length ngal, or an integer index array. Returns a
    boolean array of length ngal. Raises a clear error on a length mismatch --
    the usual cause is building the mask against a product BEFORE a cut/combine
    changed ngal; rebuild it against the exact product you are slicing.
    """
    m = np.asarray(mask)
    ng = product.ngal
    if m.dtype == bool:
        if m.shape[0] != ng:
            raise ValueError(
                f"boolean mask length {m.shape[0]} != product.ngal {ng}. The mask "
                f"must be aligned to THIS product's galaxy axis -- rebuild it against "
                f"the same (possibly cut/combined) product you are slicing.")
        return m
    if np.issubdtype(m.dtype, np.integer):
        out = np.zeros(ng, dtype=bool)
        out[m] = True                          # IndexError here = out-of-range index
        return out
    raise TypeError("mask must be a boolean array (len ngal) or an integer index array.")


def slice_product(product: GalaxyProduct,
                  mask: Union[np.ndarray, Sequence]) -> GalaxyProduct:
    """Slice every galaxy-axis array AND the catalog by `mask`, consistently.

    `mask` may be a boolean array (length ngal) or an integer index array. With
    integer indices the output row order follows the indices (so you can sort a
    selection by, e.g., separation). Mirrors io.apply_finite_cut /
    multicat._slice_product -- the one place selection actually cuts the product.
    """
    m = np.asarray(mask)
    sel = m if (m.dtype == bool) else np.asarray(m)         # bool mask or int index
    bg = product.background[sel] if product.background is not None else None
    return GalaxyProduct(
        spec=product.spec[sel], err=product.err[sel], nfib=product.nfib[sel],
        wave=product.wave, r_edges=product.r_edges, catalog=product.catalog[sel],
        header=product.header, background=bg,
    )


# =====================================================================
# the Selection result
# =====================================================================
@dataclass
class Selection:
    """Result of a cone/criteria search -- inspect it, then stack `.product`.

    Attributes
    ----------
    mask        : bool array, length = PARENT ngal (order-independent membership)
    index       : int indices into the parent, in the selection's own order
                  (sorted by separation for a cone search)
    product     : GalaxyProduct sliced to the selection, rows in `index` order
    catalog     : product.catalog (convenience); for a cone search it carries an
                  extra 'sep_arcsec' column
    sep_arcsec  : separations aligned to `index` (cone search only; else None)
    parent_ngal : ngal of the product the selection was made from
    info        : short human description of the query
    """
    mask: np.ndarray
    index: np.ndarray
    product: GalaxyProduct
    catalog: "Table"
    parent_ngal: int
    sep_arcsec: Optional[np.ndarray] = None
    info: str = ""

    @property
    def n(self) -> int:
        return int(self.mask.sum())

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        extra = ""
        if self.sep_arcsec is not None and self.n:
            extra = f", sep {self.sep_arcsec.min():.1f}-{self.sep_arcsec.max():.1f}\""
        head = f" [{self.info}]" if self.info else ""
        return f"<Selection {self.n}/{self.parent_ngal} galaxies{extra}>{head}"


def _make_selection(product, mask, *, order=None, sep=None, info="") -> Selection:
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask) if order is None else np.asarray(order, int)
    sub = slice_product(product, idx)
    cat = sub.catalog
    if sep is not None:
        cat = cat.copy()
        cat["sep_arcsec"] = np.asarray(sep, float)
        sub = _dc_replace(sub, catalog=cat)
    return Selection(mask=mask, index=idx, product=sub, catalog=cat,
                     parent_ngal=product.ngal, sep_arcsec=sep, info=info)


# =====================================================================
# 1. CONE search
# =====================================================================
def cone_search(product: GalaxyProduct, ra: float, dec: float,
                radius_arcsec: float, *, ra_col: str = "RA", dec_col: str = "DEC",
                sort: bool = True, verbose: bool = True) -> Selection:
    """Galaxies within `radius_arcsec` of (ra, dec) [degrees, arcsec].

    Returns a Selection: `.catalog` lists the matched rows (with a 'sep_arcsec'
    column, nearest first if sort=True) for you to eyeball; `.product` is the
    matching sub-product ready for multicat.stack_each. Non-finite coordinates
    never match.
    """
    cra = _as_float(product.catalog[ra_col])
    cdec = _as_float(product.catalog[dec_col])
    sep = _sep_arcsec(ra, dec, cra, cdec)
    mask = np.isfinite(sep) & (sep <= float(radius_arcsec))
    idx = np.flatnonzero(mask)
    if sort:
        idx = idx[np.argsort(sep[idx])]
    if verbose:
        print(f"cone_search: {idx.size} galaxies within {radius_arcsec:g}\" "
              f"of ({ra:.5f}, {dec:.5f})")
    return _make_selection(
        product, mask, order=idx, sep=sep[idx],
        info=f"cone r<={radius_arcsec:g}\" @({ra:.4f},{dec:.4f})")


# =====================================================================
# 2. CRITERIA search (mask / callable / expression)
# =====================================================================
def select_by(product: GalaxyProduct,
              criteria: Union[np.ndarray, Callable[["Table"], np.ndarray], str],
              *, verbose: bool = True, info: str = "") -> Selection:
    """Select by an arbitrary criterion. `criteria` may be:

      * a boolean array (length ngal),
      * a callable cat -> bool array, e.g. ``lambda c: (c['MASS_50'] > 10)``,
      * a string expression over catalog column names plus ``np``, e.g.
        ``"(MASS_50 > 10) & (z < 2.5)"`` (column names must be valid Python
        identifiers; otherwise use the callable form).

    Returns a Selection in natural catalog order. Non-finite comparisons that
    yield False simply don't match.
    """
    cat = product.catalog
    if callable(criteria):
        mask = np.asarray(criteria(cat))
    elif isinstance(criteria, str):
        mask = np.asarray(eval(criteria, {"__builtins__": {}}, _expr_namespace(cat)))   # noqa: S307
        info = info or criteria
    else:
        mask = np.asarray(criteria)
    mask = _coerce_mask(product, mask.astype(bool) if mask.dtype != bool else mask)
    if verbose:
        print(f"select_by: {int(mask.sum())}/{product.ngal} galaxies"
              + (f"  [{info}]" if info else ""))
    return _make_selection(product, mask, info=info)


# =====================================================================
# 3. DERIVED quantities (function of columns -> attach as a column)
# =====================================================================
def derive(product: GalaxyProduct,
           value: Union[Callable[["Table"], np.ndarray], str]) -> np.ndarray:
    """Compute a per-galaxy quantity that is a FUNCTION of catalog columns.

    `value` is a callable cat -> array or a string expression (same rules as
    select_by). Returns a float array aligned to the galaxy axis. This is the
    raw quantity (e.g. for a histogram); use with_derived / split_by_derived to
    attach it and split on it.
    """
    cat = product.catalog
    if callable(value):
        return np.asarray(value(cat), dtype=float)
    return np.asarray(eval(value, {"__builtins__": {}}, _expr_namespace(cat)), dtype=float)  # noqa: S307


def with_column(product: GalaxyProduct, name: str,
                values: Sequence) -> GalaxyProduct:
    """Return a COPY of `product` with a new catalog column `name`.

    Galaxy-axis arrays are shared (not copied); only the catalog is copied, so
    this is cheap. The new column makes a derived quantity addressable by name --
    e.g. so multicat.split_product_by(prod2, name, ...) can split on it.
    """
    vals = np.asarray(values)
    if vals.shape[0] != product.ngal:
        raise ValueError(f"values length {vals.shape[0]} != product.ngal {product.ngal}.")
    cat = product.catalog.copy()
    cat[name] = vals
    return _dc_replace(product, catalog=cat)


def with_derived(product: GalaxyProduct, name: str,
                 value: Union[Callable[["Table"], np.ndarray], str]) -> GalaxyProduct:
    """derive(...) then with_column(...): a product whose catalog has `name`.

    Hand the result to multicat.split_product_by(prod2, name, percentiles=50)
    to split on a derived quantity exactly as you would on a real column.
    """
    return with_column(product, name, derive(product, value))


# =====================================================================
# 4. SPLIT helpers (hook straight into multicat)
# =====================================================================
def split_by_derived(product: GalaxyProduct,
                     value: Union[Callable[["Table"], np.ndarray], str], *,
                     name: str = "DERIVED", percentiles=None, edges=None,
                     labels=None, write: bool = False,
                     config: "PipelineConfig" = None, output_dir=None,
                     verbose: bool = True) -> "OrderedDict":
    """Split a product into sub-products by a DERIVED quantity (low/high sSFR...).

    Computes `value` (callable or expression over columns), attaches it as column
    `name`, then defers to multicat.split_product_by -- so `percentiles`, `edges`,
    `labels`, and `write` behave identically to a normal column split (e.g.
    percentiles=50 -> median split; non-finite / out-of-range rows are dropped).

    Returns OrderedDict{label: GalaxyProduct}, ready for multicat.stack_each.
    """
    from . import multicat                       # lazy: pulls in matplotlib
    prod2 = with_derived(product, name, value)
    if verbose:
        print(f"split_by_derived on '{name}':")
    return multicat.split_product_by(
        prod2, name, percentiles=percentiles, edges=edges, labels=labels,
        write=write, config=config, output_dir=output_dir, verbose=verbose)


def split_by_mask(product: GalaxyProduct, mask, *, labels=None,
                  verbose: bool = True) -> "OrderedDict":
    """Split a product into sub-products by YOUR OWN mask.

    `mask` is aligned to this product's galaxy axis. Two ways to use it:
      * boolean array  -> two groups (the False group, then the True group).
        Default labels ("other", "selected"); pass labels=["low","high"] to taste.
      * integer/string array (categories) -> one group per unique value, in
        sorted order; pass `labels` to rename (must match the number of groups).

    Returns OrderedDict{label: GalaxyProduct}, ready for multicat.stack_each.
    Raises if len(mask) != product.ngal (the common foot-gun: a mask built before
    a cut/combine changed ngal -- rebuild it against THIS product).
    """
    m = np.asarray(mask)
    if m.shape[0] != product.ngal:
        raise ValueError(
            f"mask length {m.shape[0]} != product.ngal {product.ngal}. Build the "
            f"mask against the SAME product you are splitting (after any cut/combine).")

    out = OrderedDict()
    if m.dtype == bool:
        labs = labels if labels is not None else ["other", "selected"]
        if len(labs) != 2:
            raise ValueError("boolean mask needs exactly 2 labels.")
        for lab, sel in zip(labs, (~m, m)):
            if verbose:
                print(f"  {lab}: {int(sel.sum())} galaxies")
            out[lab] = slice_product(product, sel)
    else:
        vals = np.asarray(m)
        uniq = sorted(np.unique(vals).tolist())
        if labels is not None and len(labels) != len(uniq):
            raise ValueError(f"got {len(labels)} labels for {len(uniq)} groups {uniq}.")
        for i, u in enumerate(uniq):
            sel = (vals == u)
            lab = labels[i] if labels is not None else str(u)
            if verbose:
                print(f"  {lab}: {int(sel.sum())} galaxies")
            out[lab] = slice_product(product, sel)
    return out


# =====================================================================
# 5. quick-look plots (lazy matplotlib)
# =====================================================================
def plot_sky(product: GalaxyProduct, selection: "Selection" = None, *,
             center=None, radius_arcsec=None, ra_col: str = "RA",
             dec_col: str = "DEC", ax=None, s: float = 12, title: str = None):
    """Scatter the sample on the sky; highlight a Selection and draw the cone.

    `center=(ra, dec)` and `radius_arcsec` draw the search circle. Returns the
    Axes. (RA axis is drawn increasing leftwards, the usual sky convention.)
    """
    import matplotlib.pyplot as plt

    ra = _as_float(product.catalog[ra_col])
    dec = _as_float(product.catalog[dec_col])
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.4))

    ax.scatter(ra, dec, s=s, c="0.7", lw=0, label=f"all ({product.ngal})")
    if selection is not None:
        sra = _as_float(selection.catalog[ra_col])
        sdec = _as_float(selection.catalog[dec_col])
        ax.scatter(sra, sdec, s=s * 2.0, c="crimson", lw=0,
                   label=f"selected ({selection.n})")
    if center is not None and radius_arcsec is not None:
        cra, cdec = center
        rdeg = radius_arcsec / 3600.0
        th = np.linspace(0, 2 * np.pi, 240)
        # circle of constant angular radius; widen in RA by 1/cos(dec)
        ax.plot(cra + rdeg * np.cos(th) / np.cos(np.radians(cdec)),
                cdec + rdeg * np.sin(th), color="navy", lw=1.3, ls="--")
        ax.plot([cra], [cdec], "+", color="navy", ms=10, mew=1.6)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_aspect(1.0 / np.cos(np.radians(np.nanmedian(dec))))
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.set_title(title or "Sky positions")
    return ax


def plot_split_hist(product_or_values, value=None, *, bins=30, edges=None,
                    log: bool = False, ax=None, xlabel: str = None,
                    title: str = None):
    """Histogram of a (derived) quantity, with split boundaries drawn.

    Call EITHER with an array of values, OR with (product, value) where `value`
    is a callable/expression to derive (same rules as derive()). `edges` draws
    vertical split boundaries (e.g. the percentile/value edges you split at).
    Returns the Axes.
    """
    import matplotlib.pyplot as plt

    if value is not None:
        vals = derive(product_or_values, value)
    else:
        vals = _as_float(product_or_values)
    v = vals[np.isfinite(vals)]
    if log:
        v = v[v > 0]
        v = np.log10(v)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(v, bins=bins, color="0.6", edgecolor="white", lw=0.5)
    if edges is not None:
        ed = np.log10(edges) if log else np.asarray(edges, float)
        for e in np.atleast_1d(ed):
            if np.isfinite(e):
                ax.axvline(e, color="crimson", lw=1.4, ls="--")
    ax.set_xlabel(xlabel or ("log10(quantity)" if log else "quantity"))
    ax.set_ylabel("galaxies")
    ax.set_title(title or "Split quantity distribution")
    return ax


def split_controlled(catalog, split_col, control_col,
                     control_bins=None, n_control_bins=4, method="median",
                     min_per_bin=6, balance=True, verbose=True):
    """
    Low/high split of `split_col` while holding `control_col` constant: within
    each bin of the control variable, galaxies are split below/above, then pooled.

    method  : 'median'  -> two halves   |  'tercile' -> bottom vs top third (drops middle)
    balance : (median only) assign by RANK within each bin so the halves are
              equal-sized even with ties at the median; the odd middle galaxy in
              odd-count bins alternates sides bin-to-bin so the pooled totals stay
              balanced. balance=False reverts to the value threshold y<median.

    Returns (low_mask, high_mask) aligned to the catalog.
    """
    x = np.asarray(catalog[control_col], float)
    y = np.asarray(catalog[split_col],   float)
    ok = np.isfinite(x) & np.isfinite(y)

    if control_bins is None:
        control_bins = np.nanpercentile(x[ok], np.linspace(0, 100, n_control_bins + 1))
    control_bins = np.asarray(control_bins, float)

    low  = np.zeros(len(x), bool)
    high = np.zeros(len(x), bool)
    flip = False                                   # alternates the odd middle galaxy
    for i in range(len(control_bins) - 1):
        lo, hi = control_bins[i], control_bins[i + 1]
        last = (i == len(control_bins) - 2)
        in_bin = ok & (x >= lo) & ((x <= hi) if last else (x < hi))
        nb = int(in_bin.sum())
        if nb < min_per_bin:
            if verbose:
                print(f"  {control_col} [{lo:.3g},{hi:.3g}): {nb} gal -> skipped (too few)")
            continue
        idx = np.flatnonzero(in_bin)
        yv  = y[idx]

        if method == "tercile":
            t1, t2 = np.percentile(yv, [33.333, 66.667])
            low[idx[yv <= t1]]  = True
            high[idx[yv >= t2]] = True
        elif method == "median" and balance:
            order = idx[np.argsort(yv, kind="mergesort")]   # stable; ties keep input order
            half  = nb // 2
            if nb % 2 == 0:
                low[order[:half]]  = True
                high[order[half:]] = True
            else:                                  # odd: middle galaxy alternates side
                if flip:
                    low[order[:half + 1]]  = True
                    high[order[half + 1:]] = True
                else:
                    low[order[:half]]   = True
                    high[order[half:]]  = True
                flip = not flip
        elif method == "median":
            thr = np.median(yv)
            low[idx[yv <  thr]] = True
            high[idx[yv >= thr]] = True
        else:
            raise ValueError("method must be 'median' or 'tercile'")

        if verbose:
            print(f"  {control_col} [{lo:.3g},{hi:.3g}): {nb:3d} gal  "
                  f"low={int(low[idx].sum())}  high={int(high[idx].sum())}")

    if verbose:
        xl, xh = np.median(x[low]), np.median(x[high])
        print(f"\n{split_col} split at fixed {control_col} ({method}):")
        print(f"  low : {low.sum():4d} gal,  median {control_col}={xl:.3g},  "
              f"median {split_col}={np.median(y[low]):.3g}")
        print(f"  high: {high.sum():4d} gal,  median {control_col}={xh:.3g},  "
              f"median {split_col}={np.median(y[high]):.3g}")
        print(f"  {control_col} offset between halves: {abs(xh-xl):.3g}  (want ~0)")
    return low, high

# =====================================================================
# CORE-PRODUCT S/N TRIAGE
#   Rank + shortlist bright galaxies for single-object halo work. Operates on
#   a core.read_core_fits dict (core_spec/core_err/wave_obs/catalog/results),
#   NOT a GalaxyProduct -- everything above this point in the file selects out
#   of the binned product; this section is the equivalent front door for the
#   per-galaxy CORE product.
# =====================================================================
#
# core.run_core_hierarchical's results carry TWO S/N columns defined for every
# galaxy -- sn_mf (matched filter) and sn_b (fixed constant-window integration,
# literally "S/N inside the integration window") -- plus a third, sn_a, that
# only exists for detections (from the Gaussian fit). top_sn_galaxies ranks on
# sn_b by default, matching "high S/N inside the integration window." The
# older flat schema (core.run_core) has a single 'sn' column; both are handled.
#
# NOTE ON WHY THIS SLICES BY HAND: core.select_core_by_index / select_core_by_id
# (and the CoreSpectrum/CoreSelection.meas they build via core._result_row)
# infer the per-galaxy array length from a hardcoded 'flux' key. That key only
# exists in the FLAT schema; the HIERARCHICAL schema has flux_mf/flux_b/flux_a
# instead, so the inferred length comes back 0 and every per-galaxy results
# column silently falls through UNINDEXED (full parent length, wrong order) --
# the root cause of an S/N column reading back all-NaN downstream. The
# one-line upstream fix is to compare against _core_ngal(core) (==
# core_spec.shape[0]) instead of len(results.get('flux', [])) in
# core._make_core_selection / core._result_row. Until that lands, the
# functions below never route through them -- they index core_spec/core_err/
# catalog/results directly against ngal.

_SN_FIELD_ALIASES = {
    "b":  dict(sn="sn_b",  flux="flux_b",  flux_err="flux_b_err"),
    "mf": dict(sn="sn_mf", flux="flux_mf", flux_err="flux_mf_err"),
}


def resolve_core_sn_fields(results: dict, rank_by: Optional[str] = None) -> dict:
    """
    Figure out which results keys hold S/N, flux, flux_err, label, and a
    success/detected flag -- for EITHER core schema (hierarchical or flat).

    rank_by : 'b' (fixed integration-window S/N, sn_b -- the default match for
              "S/N inside the integration window") or 'mf' (matched-filter,
              sn_mf). Ignored for a flat-schema product (only 'sn' exists).

    Returns dict(sn, flux, flux_err, label, success, mode, rank_key).
    Raises KeyError if neither schema's S/N column is present.
    """
    keys = set(results.keys())
    if "sn_b" in keys or "sn_mf" in keys:                      # hierarchical
        which = rank_by or "b"
        if which not in _SN_FIELD_ALIASES:
            raise ValueError("rank_by must be 'b' or 'mf' for a hierarchical "
                             f"core product (got {rank_by!r}).")
        f = _SN_FIELD_ALIASES[which]
        return dict(sn=f["sn"], flux=f["flux"], flux_err=f["flux_err"],
                    label="label", success="detected", mode="hierarchical",
                    rank_key=f["sn"])
    if "sn" in keys:                                            # flat
        return dict(sn="sn", flux="flux", flux_err="flux_err",
                    label="label", success="success", mode="flat",
                    rank_key="sn")
    raise KeyError("core['results'] has neither 'sn_b'/'sn_mf' (hierarchical) "
                  "nor 'sn' (flat) -- run core.run_core or "
                  "core.run_core_hierarchical first.")


def _print_core_rank_table(core, results, order, f, *, id_col, extra_cols):
    cat = core["catalog"]
    have_id = id_col in cat.colnames
    have_z = "z" in cat.colnames
    flux = np.asarray(results.get(f["flux"], np.full(len(order), np.nan)), dtype=float)
    label = (np.asarray(results[f["label"]]) if f["label"] in results
            else np.full(len(order), "", dtype="<U1"))
    sn = np.asarray(results[f["sn"]], dtype=float)

    header = ["rank", "row", "ID" if have_id else "-", f["sn"].upper(), "flux", "label"]
    if have_z:
        header.append("z")
    header += list(extra_cols)
    print("  ".join(f"{c:>10}" for c in header))
    for rank, i in enumerate(order, 1):
        row = [f"{rank:>10d}", f"{int(i):>10d}"]
        row.append(f"{str(cat[id_col][i]):>10}" if have_id else f"{'-':>10}")
        row.append(f"{sn[i]:>10.1f}")
        row.append(f"{flux[i]:>10.2e}")
        row.append(f"{str(label[i]):>10}")
        if have_z:
            row.append(f"{float(cat['z'][i]):>10.3f}")
        for c in extra_cols:
            if c in cat.colnames:
                v = cat[c][i]
                row.append(f"{float(v):>10.3g}"
                          if np.issubdtype(np.asarray(v).dtype, np.number)
                          else f"{str(v):>10}")
        print("  ".join(row))


def top_sn_galaxies(core, *, n=20, rank_by=None, sn_min=None,
                    require_detection=False, labels=None,
                    id_col="ID", extra_cols=(), verbose=True) -> dict:
    """
    Rank galaxies in a core product by their Lya S/N and return the top n as a
    standalone core-shaped dict -- ready for plotting.plot_top_sn_grid, or any
    core.py function that takes a core dict (e.g. core.build_core_stacks).

        core = core.read_core_fits("core_....fits")
        sub  = top_sn_galaxies(core, n=20)     # prints a ranked table
        ids  = id_list(sub)[:5]                # -> multicat.stack_id_list

    n                 : how many to keep (None -> all that pass the filters).
    rank_by           : 'b' (default -- fixed integration-window S/N, sn_b) or
                        'mf' (matched-filter, sn_mf). Ignored (auto -> 'sn')
                        for a flat-schema (core.run_core) product.
    sn_min            : optional hard floor on the ranking S/N.
    require_detection : hierarchical: filter on the 'detected' flag (sn_mf >=
                        detect_snr -- a DIFFERENT, stricter gate than sn_b);
                        flat: filter on 'success'. Default False, since sn_b/
                        sn_mf are already the selection criterion and gating on
                        'detected' as well can drop good sn_b candidates that
                        never crossed the (separate) matched-filter threshold.
    labels            : keep only these classify_lya labels, e.g. ('emitter',
                        'LAE'); None keeps every label.
    id_col            : catalog column with the galaxy ID.
    extra_cols        : extra catalog columns to show in the printed table.

    Returns dict(core_spec, core_err, wave_obs, catalog, results, header,
    sn_ranked, rank_key) sliced to the top n, S/N-descending. sn_ranked is the
    resolved per-galaxy S/N actually used to rank (same order); rank_key names
    which results column that was. Raises if nothing passes the filters.
    """
    results = core.get("results") or {}
    f = resolve_core_sn_fields(results, rank_by=rank_by)
    ngal = int(np.asarray(core["core_spec"]).shape[0])

    sn = np.asarray(results[f["sn"]], dtype=float)
    if sn.shape[0] != ngal:
        raise ValueError(f"results[{f['sn']!r}] has length {sn.shape[0]}, "
                         f"expected ngal={ngal} -- core product looks corrupt.")

    keep = np.isfinite(sn)
    if require_detection and f["success"] in results:
        keep &= np.asarray(results[f["success"]], dtype=bool)
    if sn_min is not None:
        keep &= sn >= float(sn_min)
    if labels is not None and f["label"] in results:
        want = {str(x) for x in np.atleast_1d(labels)}
        lab = np.asarray(results[f["label"]])
        keep &= np.array([str(x) in want for x in lab])

    idx_all = np.flatnonzero(keep)
    if idx_all.size == 0:
        raise ValueError("no galaxies pass the requested filters "
                         f"(rank_by={f['rank_key']!r}, sn_min={sn_min}, labels={labels}).")
    order = idx_all[np.argsort(sn[idx_all])[::-1]]
    if n is not None:
        order = order[:int(n)]

    if verbose:
        _print_core_rank_table(core, results, order, f, id_col=id_col,
                               extra_cols=extra_cols)

    # slice by hand -- see the module note above on why we don't route through
    # core.select_core_by_index for a hierarchical-schema product
    spec = np.asarray(core["core_spec"])[order]
    err = (np.asarray(core["core_err"])[order]
          if core.get("core_err") is not None else None)
    catalog = core["catalog"][order]
    res_sub = {}
    for k, v in results.items():
        arr = np.asarray(v)
        res_sub[k] = arr[order] if (arr.ndim >= 1 and arr.shape[0] == ngal) else v

    return dict(core_spec=spec, core_err=err, wave_obs=core["wave_obs"],
               catalog=catalog, results=res_sub, header=core.get("header"),
               sn_ranked=sn[order], rank_key=f["rank_key"])


def id_list(core_or_sub, *, id_col="ID"):
    """Plain Python list of catalog IDs, in the product's current row order --
    e.g. the top-N core dict from top_sn_galaxies, ready to hand to
    multicat.stack_id_list."""
    col = np.asarray(core_or_sub["catalog"][id_col])
    out = col.tolist()
    return [int(v) for v in out] if np.issubdtype(col.dtype, np.integer) else out
