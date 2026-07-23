"""
formatting.py -- paper-writing helper, NOT part of the utils_lya_halo
measurement/fitting pipeline. Lives here (sibling to the numbered
notebooks, alongside run_config.py/test_framework.py) rather than inside
utils_lya_halo/ because its job is LaTeX table formatting for the draft,
not data reduction -- it does no new computation, only reads
ALREADY-COMPUTED result dicts from measure.py/fitting.py/analysis.py
(the same objects you already hand to
analysis.describe_subsample_properties) and turns them into a
paste-into-the-.tex-draft table.

Import from a notebook in this same code/ directory with:
    import formatting

Quick start (current house style, no config needed)
-----------------------------------------------------
    subsamples = [
        {"label": "ALL Galaxy Stack", "boot_fit": boot_fit,
         "boundary_info": properties["boundary_info"], "vel": properties["vel"],
         "core_lum": properties["core_lum"], "onehalo_lum": properties["onehalo_lum"],
         "twohalo_lum": properties["twohalo_lum"], "ratios": properties["ratios"]},
        # ... one dict per additional subsample/row, same keys
    ]
    print(formatting.make_subsample_latex_table(
        subsamples,
        caption="Derived properties by subsample.",
        label="tab:subsample-properties"))

Each subsample dict uses the SAME kwarg names as
analysis.describe_subsample_properties, so you can build it straight from
whatever you already pass into that function per subsample -- "label" is
the only required key, everything else is optional (a column with a
missing input prints "n/a" for that row, same convention
describe_subsample_properties uses).

Design choices you'll actually want to tweak, and how
------------------------------------------------------
- Which columns, and in what order: `columns=[...]` -- list of
  COLUMN_REGISTRY keys, e.g. ["h1", "L_core", "L_1halo"]. None -> DEFAULT_COLUMNS.
- One-off header text for a column without touching the registry:
  pass (header, key) tuples instead of bare keys inside `columns=`.
- Per-column unit scaling (e.g. report luminosities as X, with the paper
  told "in units of 1e42 erg/s" in the caption instead of every cell):
  `scales={"L_2halo": 1e43}` -- overrides COLUMN_REGISTRY's default scale
  for just that column, for just this call.
- Sig figs on the errors (defaults to 2): `sig=2`, or per-column via
  `registry` override (see ADD A NEW COLUMN below).
- Table width fixes (see the "doesn't fit on the line" conversation):
  `fontsize=r"\\footnotesize"`, `tabcolsep="4pt"`, `table_env="table*"`
  (default, spans both columns in aastex702's preprint2 mode), or
  `rotate=True` for AASTeX's `rotatetable*` landscape environment if it's
  still too wide even at table* + footnotesize.

Add a new column without editing this file's row-building logic
------------------------------------------------------------------
Write a tiny extractor `(fid, lo, hi) = get_thing(inputs)` (inputs is the
per-subsample dict you passed in -- pull whatever keys you need out of
its boot_fit/vel/core_lum/etc. sub-dicts) and register it:

    my_registry = dict(formatting.COLUMN_REGISTRY)
    my_registry["gamma"] = {
        "header": r"$\gamma$", "unit": "", "scale": 1,
        "get": lambda inputs: (
            (inputs["boot_fit"]["gamma_fid"], inputs["boot_fit"]["gamma_lo"],
             inputs["boot_fit"]["gamma_hi"]) if inputs.get("boot_fit") else (None, None, None)),
    }
    formatting.make_subsample_latex_table(subsamples, columns=[..., "gamma"],
                                           registry=my_registry, ...)

No lo/hi available for some quantity (point estimate only, e.g.
boundary_radius when you haven't run bootstrap_fit_profile)? Return
`(value, None, None)` from your `get` function -- _fmt_asym prints
`$value$ [point estimate]` instead of raising.
"""
import string

import numpy as np


# ---------------------------------------------------------------------
# Value+error formatting
# ---------------------------------------------------------------------
def _fmt_asym(fid, lo=None, hi=None, sig=2, scale=1, no_error_tag="[point estimate]"):
    """fid/lo/hi -> '$V^{+a}_{-b}$', pre-divided by `scale`. No unit, no
    embedded power-of-ten -- both get reported once per COLUMN in the
    table's auto-generated Note. line (see _lettered_note), not per cell.

    lo/hi are absolute 16/84 bootstrap bounds, not half-widths. Errors
    round to `sig` sig figs; the central value rounds to match. Missing
    lo/hi (None, or non-finite) -> just the scaled value + `no_error_tag`,
    rather than raising -- lets a column be point-estimate-only for some
    rows (e.g. boundary_radius with no bootstrap refit) without special-
    casing every caller.
    """
    if fid is None or not np.isfinite(fid):
        return "n/a"
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return f"${fid / scale:.3g}$ {no_error_tag}"

    fid_s, hi_s, lo_s = fid / scale, hi / scale, lo / scale
    err_hi, err_lo = hi_s - fid_s, fid_s - lo_s

    def _round_to_sig(x, sig):
        if x == 0 or not np.isfinite(x):
            return x, 0
        d = sig - 1 - int(np.floor(np.log10(abs(x))))
        return round(x, d), d

    err_hi_r, d_hi = _round_to_sig(err_hi, sig)
    err_lo_r, d_lo = _round_to_sig(err_lo, sig)
    d = max(d_hi, d_lo, 0)   # floors negative d (round-to-nearest-10) to 0 -- fine for v1
    return f"${fid_s:.{d}f}^{{+{err_hi_r:.{d}f}}}_{{-{err_lo_r:.{d}f}}}$"


def _invert_ratio(ratios, key="onehalo_over_core"):
    """core/1-halo from the 1-halo/core ratio, inverted so '4.4x too much
    1-halo flux' becomes 'core is 23% of the 1-halo zone' -- easier to
    compare against literature numbers reported the other way round.
    Uses the per-draw _all array (draw-aligned, per
    measure.measure_three_zone_ratios' own docstring) so percentiles come
    from re-percentiling the inverted draws, not just flipping fid's
    lo/hi (equivalent for a monotonic 1/x transform on positive values,
    but more direct since the draws are already sitting there)."""
    fid = ratios.get(f"{key}_fid")
    if fid is None or not np.isfinite(fid) or fid == 0:
        return float("nan"), float("nan"), float("nan")
    inv_fid = 1.0 / fid
    all_ = ratios.get(f"{key}_all")
    if all_ is not None:
        inv_all = 1.0 / np.asarray(all_, dtype=float)
        inv_lo, inv_hi = np.nanpercentile(inv_all, [16, 84])
    else:
        lo, hi = ratios.get(f"{key}_lo"), ratios.get(f"{key}_hi")
        inv_lo, inv_hi = (1.0 / hi, 1.0 / lo) if (lo and hi) else (float("nan"), float("nan"))
    return inv_fid, inv_lo, inv_hi


# ---------------------------------------------------------------------
# Column registry -- one entry per available column. `get(inputs)` pulls
# (fid, lo, hi) out of the per-subsample input dict; inputs is whatever
# dict you passed in `subsamples` (same keys as describe_subsample_properties).
# Add new columns by adding entries here (or in a copy passed via the
# `registry=` kwarg) -- nothing else in this file needs to change.
# ---------------------------------------------------------------------
def _get_h1(inputs):
    bf = inputs.get("boot_fit")
    return (bf["h1_fid"], bf["h1_lo"], bf["h1_hi"]) if bf else (None, None, None)


def _get_r_x(inputs):
    bf = inputs.get("boot_fit")
    if bf and bf.get("crossover_radius_fid") is not None:
        return (bf["crossover_radius_fid"], bf["crossover_radius_lo"], bf["crossover_radius_hi"])
    bi = inputs.get("boundary_info")
    if bi is not None:
        b = bi["boundary_radius"] if isinstance(bi, dict) else float(bi)
        return (b, None, None)   # no bootstrap refit -- point estimate only
    return (None, None, None)


def _get_v_core(inputs):
    v = inputs.get("vel")
    return (v["core_v_fid"], v["core_v_lo"], v["core_v_hi"]) if v else (None, None, None)


def _get_v_2halo(inputs):
    v = inputs.get("vel")
    return (v["twohalo_v_fid"], v["twohalo_v_lo"], v["twohalo_v_hi"]) if v else (None, None, None)


def _get_v_diff(inputs):
    """core - 2halo velocity offset. Not in DEFAULT_COLUMNS (dropped
    2026-07-22 -- v_core/v_2halo side by side already show the offset),
    kept registered so it's a one-line add-back via columns=[..., "v_diff"]."""
    v = inputs.get("vel")
    return (v["diff_fid"], v["diff_lo"], v["diff_hi"]) if v else (None, None, None)


def _get_L_core(inputs):
    cl = inputs.get("core_lum")
    return (cl["core_lum_fid"], cl["core_lum_lo"], cl["core_lum_hi"]) if cl else (None, None, None)


def _get_L_1halo(inputs):
    oh = inputs.get("onehalo_lum")
    return (oh["onehalo_lum_fid"], oh["onehalo_lum_lo"], oh["onehalo_lum_hi"]) if oh else (None, None, None)


def _get_L_2halo(inputs):
    th = inputs.get("twohalo_lum")
    return (th["twohalo_lum_fid"], th["twohalo_lum_lo"], th["twohalo_lum_hi"]) if th else (None, None, None)


def _get_core_over_1halo_pct(inputs):
    r = inputs.get("ratios")
    if not r:
        return (None, None, None)
    fid, lo, hi = _invert_ratio(r, "onehalo_over_core")
    return (fid * 100, lo * 100, hi * 100)


COLUMN_REGISTRY = {
    "h1":       {"header": r"$h_1$",                        "unit": r"\kpc", "scale": 1,    "get": _get_h1},
    "r_x":      {"header": r"$r_{\rm x}$",                  "unit": r"\kpc", "scale": 1,    "get": _get_r_x},
    "v_core":   {"header": r"$v_{\rm core}$",               "unit": r"\kms", "scale": 1,    "get": _get_v_core},
    "v_2halo":  {"header": r"$v_{\rm 2halo}$",              "unit": r"\kms", "scale": 1,    "get": _get_v_2halo},
    "v_diff":   {"header": r"$v_{\rm core}-v_{\rm 2halo}$", "unit": r"\kms", "scale": 1,    "get": _get_v_diff},
    "L_core":   {"header": r"$L_{\rm core}$",               "unit": r"\ergs", "scale": 1e42, "get": _get_L_core},
    "L_1halo":  {"header": r"$L_{\rm 1halo}$",              "unit": r"\ergs", "scale": 1e42, "get": _get_L_1halo},
    "L_2halo":  {"header": r"$L_{\rm 2halo}$",              "unit": r"\ergs", "scale": 1e42, "get": _get_L_2halo},
    "core_over_1halo_pct": {"header": r"$L_{\rm core}/L_{\rm 1halo}$", "unit": r"\%", "scale": 1, "get": _get_core_over_1halo_pct},
}

# Current house style. v_diff is registered above but not included by
# default -- add it back via columns=[..., "v_diff"] if you want it.
DEFAULT_COLUMNS = ["h1", "r_x", "v_core", "v_2halo", "L_core", "L_1halo", "L_2halo", "core_over_1halo_pct"]


# ---------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------
def _build_row(label, inputs, columns, registry, scales, sig):
    row = {"label": label}
    for key in columns:
        spec = registry[key]
        fid, lo, hi = spec["get"](inputs)
        scale = scales.get(key, spec["scale"])
        col_sig = spec.get("sig", sig)
        row[key] = _fmt_asym(fid, lo, hi, sig=col_sig, scale=scale)
    return row


def _normalize_columns(columns, registry):
    """columns: list of registry keys, or (header, key) tuples for a
    one-off header override without touching the registry. Returns a
    list of (header, key) pairs."""
    out = []
    for c in columns:
        if isinstance(c, tuple):
            out.append(c)
        else:
            out.append((registry[c]["header"], c))
    return out


def _add_superscript(header, letter):
    """Append a superscript footnote letter after a column header, e.g.
    r"$h_1$" -> r"$h_1$\textsuperscript{a}". Deliberately appended OUTSIDE
    the header's math mode (rather than inserted as an in-math exponent
    like $h_1^{\rm a}$) -- for multi-symbol headers such as the
    core_over_1halo_pct ratio column (r"$L_{\rm core}/L_{\rm 1halo}$"), an
    in-math exponent would visually stack onto just the last symbol
    (L_1halo^d), misreading as if the letter modified that symbol alone
    rather than marking the whole column."""
    return header + rf"\textsuperscript{{{letter}}}"


def _group_columns(colnames, registry, scales):
    """Groups columns sharing the same (scale, unit), in order of first
    appearance -- e.g. h1/r_x both in \\kpc become one group, the three
    luminosity columns (all scale=1e42, unit=\\ergs) become another."""
    groups = {}
    order = []
    for _, key in colnames:
        if key == "label" or key not in registry:
            continue
        spec = registry[key]
        gkey = (scales.get(key, spec["scale"]), spec["unit"])
        if gkey not in groups:
            groups[gkey] = []
            order.append(gkey)
        groups[gkey].append(key)
    return [(scale, unit, groups[(scale, unit)]) for scale, unit in order]


def _lettered_note(colnames, registry, scales):
    """Literature-standard version of the old _column_note: instead of a
    footnote clause that repeats each header name ('$h_1$, $r_x$ in \\kpc'),
    assign a superscript letter (a, b, c, ...) to each (scale, unit) column
    group and have the footnote define what each letter means. The letters
    themselves get stamped onto the headers by the caller (rows_to_latex_table)
    via _add_superscript. Returns (key -> letter dict, footnote string) so
    the same grouping drives both the headers and the note text.

    Runs out of letters past 26 columns sharing a (scale, unit) grouping --
    not a real limit given DEFAULT_COLUMNS has 8 columns in ~4 groups."""
    groups = _group_columns(colnames, registry, scales)
    key_to_letter = {}
    parts = []
    for letter, (scale, unit, keys) in zip(string.ascii_lowercase, groups):
        for k in keys:
            key_to_letter[k] = letter
        if scale == 1:
            desc = f"in {unit}" if unit else "dimensionless"
        else:
            exp = int(round(np.log10(scale)))
            desc = rf"in units of $10^{{{exp}}}$\,{unit}"
        parts.append(rf"$^{{\rm {letter}}}$ {desc}")
    return key_to_letter, "Note. " + "; ".join(parts) + "."


# ---------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------
def rows_to_latex_table(rows, colnames, caption, label, registry, scales,
                         table_env="table*", fontsize=None, tabcolsep=None, rotate=False):
    """colnames: list of (header, key) pairs, "label" first. Design knobs
    for the "doesn't fit on the line" problem: fontsize (e.g.
    r"\\footnotesize"), tabcolsep (e.g. "4pt"), table_env ("table" or
    "table*"), rotate=True for AASTeX's rotatetable* landscape env
    (overrides table_env when set)."""
    env = "rotatetable*" if rotate else table_env
    key_to_letter, note = _lettered_note(colnames, registry, scales)
    header_row = [_add_superscript(h, key_to_letter[k]) if k in key_to_letter else h
                  for h, k in colnames]
    lines = [rf"\begin{{{env}}}", r"\centering"]
    if fontsize:
        lines.append(fontsize)
    if tabcolsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              r"\begin{tabular}{" + "l" * len(colnames) + "}", r"\hline",
              " & ".join(header_row) + r" \\", r"\hline"]
    for row in rows:
        lines.append(" & ".join(str(row.get(k, "n/a")) for _, k in colnames) + r" \\")
    # \par forces the footnote onto its own paragraph -- without it, LaTeX
    # can run the \footnotesize note onto the same line as the tabular's
    # trailing horizontal space when the table is narrow (few columns),
    # making the note appear to the right of the table instead of below it.
    lines += [r"\hline", r"\end{tabular}",
              r"\par", r"\vspace{2pt}", r"{\footnotesize " + note + "}",
              rf"\end{{{env}}}"]
    return "\n".join(lines)


def make_subsample_latex_table(subsamples, columns=None, scales=None, sig=2,
                                caption="TODO: caption", label="tab:subsample-properties",
                                registry=None, table_env="table*", fontsize=None,
                                tabcolsep=None, rotate=False):
    """
    Top-level entry point -- one call, one table.

    subsamples : list of dicts, one per row. Each dict uses the SAME
        kwarg names as analysis.describe_subsample_properties:
        {"label": ..., "boot_fit": ..., "boundary_info": ..., "vel": ...,
         "core_lum": ..., "onehalo_lum": ..., "twohalo_lum": ..., "ratios": ...}
        Only "label" is required; anything else missing prints "n/a" for
        that row's cells in that section, same convention
        describe_subsample_properties uses.
    columns : list of COLUMN_REGISTRY keys (or (header, key) tuples for a
        one-off header override), in display order. None -> DEFAULT_COLUMNS.
    scales  : {column_key: scale_override}, e.g. {"L_2halo": 1e43}.
    sig     : sig figs on errors, global default (per-column override:
        add "sig": N to that column's registry entry).
    registry: override/extend COLUMN_REGISTRY, e.g.
        dict(COLUMN_REGISTRY, my_new_col={...}) -- see module docstring's
        "Add a new column" section for the shape of an entry.
    table_env/fontsize/tabcolsep/rotate : layout knobs, see
        rows_to_latex_table.
    """
    registry = registry or COLUMN_REGISTRY
    columns = columns or DEFAULT_COLUMNS
    scales = scales or {}
    colnames = [("Subsample", "label")] + _normalize_columns(columns, registry)
    rows = [_build_row(s.get("label", "?"), s, columns, registry, scales, sig) for s in subsamples]
    return rows_to_latex_table(rows, colnames, caption, label, registry, scales,
                                table_env=table_env, fontsize=fontsize,
                                tabcolsep=tabcolsep, rotate=rotate)
