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
def _asym_parts(fid, lo=None, hi=None, sig=2, scale=1):
    """The ARITHMETIC behind _fmt_asym, split out so the LaTeX and Markdown
    renderers cannot drift apart. Returns one of:

        None                                 -- no value at all
        ("point", value)                     -- value, but no usable lo/hi
        ("asym", fid_s, err_hi_r, err_lo_r, d)  -- value + rounded errors, and
                                                the decimal count `d` all three
                                                must be printed at

    lo/hi are absolute 16/84 bootstrap bounds, not half-widths. Errors round to
    `sig` sig figs and the central value rounds to match -- quoting a central
    value to more precision than its own error bar is the classic way to make a
    table look more certain than the measurement is.
    """
    if fid is None or not np.isfinite(fid):
        return None
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return ("point", fid / scale)

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
    return ("asym", fid_s, err_hi_r, err_lo_r, d)


def _fmt_asym(fid, lo=None, hi=None, sig=2, scale=1, no_error_tag="[point estimate]"):
    """fid/lo/hi -> '$V^{+a}_{-b}$', pre-divided by `scale`. No unit, no
    embedded power-of-ten -- both get reported once per COLUMN in the
    table's auto-generated Note. line (see _lettered_note), not per cell.

    Missing lo/hi (None, or non-finite) -> just the scaled value +
    `no_error_tag`, rather than raising -- lets a column be point-estimate-only
    for some rows (e.g. boundary_radius with no bootstrap refit) without
    special-casing every caller.
    """
    parts = _asym_parts(fid, lo, hi, sig=sig, scale=scale)
    if parts is None:
        return "n/a"
    if parts[0] == "point":
        return f"${parts[1]:.3g}$ {no_error_tag}"
    _, fid_s, err_hi_r, err_lo_r, d = parts
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


# =====================================================================
# MARKDOWN TABLE -- the READING version
# =====================================================================
# The LaTeX table above is built for a journal: superscript footnote letters,
# units banished to a Note line, no sample sizes, no flags. That is correct for
# a paper and hostile for reading 30 rows on a screen while you decide what is
# real.
#
# This renders the SAME rows, from the SAME registry and the SAME _asym_parts
# arithmetic, optimised for looking at:
#
#   * units in the header, not in a footnote you have to cross-reference
#   * recipe and side as two sortable columns instead of "recipe: side"
#   * an N column, because "lya_class: absorber" reading n/a everywhere means
#     nothing until you know it is N=42
#   * a flags column, so a degenerate fit or a fallback boundary is visible in
#     the row rather than buried in a warnings list you never printed
#
# math=True emits `$11.43^{+0.88}_{-1.10}$`, which renders in Obsidian, in a
# Jupyter markdown cell, and on GitHub. math=False emits `11.43 +0.88/-1.10`,
# which is plain text and lines up in a terminal.

# LaTeX unit macros -> what they mean in plain text. The macros (\kpc, \kms,
# \ergs) are defined in the paper's preamble and are meaningless anywhere else,
# so they cannot go straight into a markdown header.
UNIT_TEXT = {
    r"\kpc": "kpc",
    r"\kms": "km/s",
    r"\ergs": "erg/s",
    r"\%": "%",
    "": "",
}

# Plain-ASCII header per column, for math=False. Keys are registry keys; a
# column missing from here falls back to its registry header with the LaTeX
# stripped out.
PLAIN_HEADER = {
    "h1": "h1", "r_x": "r_x", "v_core": "v_core", "v_2halo": "v_2halo",
    "v_diff": "v_core-v_2halo", "L_core": "L_core", "L_1halo": "L_1halo",
    "L_2halo": "L_2halo", "core_over_1halo_pct": "L_core/L_1halo",
}


def _unit_text(unit):
    if unit in UNIT_TEXT:
        return UNIT_TEXT[unit]
    return str(unit).lstrip("\\")          # unknown macro -> drop the backslash


def _strip_math(s):
    """'$h_1$' -> 'h1', r'$v_{\\rm core}$' -> 'v_core'. Crude on purpose: it is
    a fallback for columns not in PLAIN_HEADER, not a LaTeX parser."""
    out = str(s).replace("$", "").replace(r"\rm", "").replace("{", "").replace("}", "")
    out = out.replace("\\", "").replace("_", "_").replace(" ", "")
    return out or "?"


def _fmt_asym_md(fid, lo=None, hi=None, sig=2, scale=1, math=True,
                 no_error_tag="(point est.)"):
    """Same numbers as _fmt_asym, rendered for reading.

        math=True   $11.43^{+0.88}_{-1.10}$      (renders in Obsidian/Jupyter)
        math=False  11.43 +0.88/-1.10            (plain text, terminal-safe)
    """
    parts = _asym_parts(fid, lo, hi, sig=sig, scale=scale)
    if parts is None:
        return "--"                       # not "n/a": shorter, and scans as empty
    if parts[0] == "point":
        v = f"${parts[1]:.3g}$" if math else f"{parts[1]:.3g}"
        return f"{v} {no_error_tag}"
    _, fid_s, err_hi_r, err_lo_r, d = parts
    if math:
        return f"${fid_s:.{d}f}^{{+{err_hi_r:.{d}f}}}_{{-{err_lo_r:.{d}f}}}$"
    return f"{fid_s:.{d}f} +{err_hi_r:.{d}f}/-{err_lo_r:.{d}f}"


# Same units again, but as math-mode TeX. Needed because the unit and the
# power-of-ten have to live INSIDE the header's $...$ -- putting "10^{42}"
# outside math mode renders it literally, braces and all, which is the one thing
# that would make these headers worse than the footnote letters they replace.
UNIT_MATH = {
    r"\kpc": r"{\rm kpc}",
    r"\kms": r"{\rm km\,s^{-1}}",
    r"\ergs": r"{\rm erg\,s^{-1}}",
    r"\%": r"\%",
    "": "",
}


def _unit_math(unit):
    return UNIT_MATH.get(unit, r"{\rm " + str(unit).lstrip("\\") + "}" if unit else "")


def _md_header(key, header, registry, scales, math=True, unit_in_header=True):
    """Column header with its unit and power-of-ten folded in -- the whole
    reason the markdown table needs no footnote letters.

        math=True   $L_{\\rm 1halo}\\,[10^{42}\\,{\\rm erg\\,s^{-1}}]$
        math=False  L_1halo (10^42 erg/s)
    """
    spec = registry.get(key)
    if not math:
        name = PLAIN_HEADER.get(key, _strip_math(header))
        if not unit_in_header or spec is None:
            return name
        scale = scales.get(key, spec["scale"])
        unit = _unit_text(spec["unit"])
        if scale == 1:
            return f"{name} ({unit})" if unit else name
        exp = int(round(np.log10(scale)))
        return f"{name} (10^{exp} {unit})" if unit else f"{name} (10^{exp})"

    # math mode: build ONE $...$ so the exponent actually renders as an exponent
    core = str(header).strip()
    inner = core[1:-1] if core.startswith("$") and core.endswith("$") else core
    if not unit_in_header or spec is None:
        return f"${inner}$"
    scale = scales.get(key, spec["scale"])
    unit = _unit_math(spec["unit"])
    if scale == 1:
        return f"${inner}$" if not unit else rf"${inner}\,[{unit}]$"
    exp = int(round(np.log10(scale)))
    mag = rf"10^{{{exp}}}"
    body = rf"{mag}\,{unit}" if unit else mag
    return rf"${inner}\,[{body}]$"


# Warning text -> the short tag that goes in the flags column. Matched as
# substrings, first hit wins, so the ORDER matters: the specific causes come
# before the generic "no boundary" that they also happen to print.
_FLAG_PATTERNS = [
    ("SIDE FAILED",                    "FAILED"),
    ("DEGENERATE boundary",            "degenerate fit"),
    ("no core/two-halo boundary",      "no boundary"),
    ("boundary came from the FULL",    "fallback bdry"),
    ("did not converge on this side",  "fit !converged"),
    ("refit draws failed",             "refits failed"),
    ("< min_n",                        "N<min_n"),
    ("not leak-corrected",             ""),   # level-wide; already in the note
]


def _row_flags(inputs):
    """Short, de-duplicated flags for one row, read off the side dict that
    splits.run() already produces. Returns '' for a clean row."""
    tags = []
    if inputs.get("failed"):
        tags.append("FAILED")
    for w in inputs.get("warnings", []) or []:
        for pat, tag in _FLAG_PATTERNS:
            if pat in str(w):
                if tag and tag not in tags:
                    tags.append(tag)
                break
    return ", ".join(tags)


def make_subsample_markdown_table(subsamples, columns=None, scales=None, sig=2, *,
                                  caption=None, registry=None, math=True,
                                  unit_in_header=True, split_label=True,
                                  show_n=True, show_flags=True, align="right",
                                  note=None):
    """
    The same table as make_subsample_latex_table, rendered as a GitHub-flavoured
    markdown pipe table you can read in Obsidian, a Jupyter markdown cell, or a
    terminal.

    Takes exactly the same `subsamples` list -- one dict per row, using the same
    kwarg names as analysis.describe_subsample_properties -- so anything you can
    pass to the LaTeX version works here unchanged.

    subsamples : list of row dicts. Only "label" is required. If a row also
        carries "n_gal", "warnings" or "failed" (which every splits.run() side
        dict does), the N and flags columns pick them up automatically.
    columns    : registry keys in display order. None -> DEFAULT_COLUMNS.
    scales     : {column_key: scale_override}, e.g. {"L_2halo": 1e43}. Shown in
        the header rather than a footnote.
    math       : True  -> `$11.43^{+0.88}_{-1.10}$`, renders in Obsidian/Jupyter
                 False -> `11.43 +0.88/-1.10`, plain text, terminal-safe
    unit_in_header : fold units and powers of ten into the header. Turn off only
        if you are pasting into something that already states them.
    split_label : a label of the form "recipe: side" becomes two columns, so you
        can sort or scan by recipe. Labels without ": " are left alone.
    show_n     : include the galaxy count. ON BY DEFAULT AND WORTH KEEPING -- a
        row of "--" means one thing at N=226 and something else entirely at
        N=42, and the LaTeX table shows neither.
    show_flags : include the flags column (degenerate fit, fallback boundary,
        failed side, N<min_n). Also on by default: a degenerate fit that is only
        visible as blank cells is exactly how a bad row survives into a draft.
    align      : "right" (default) | "left". Numbers right-aligned scan far
        better down a column.
    note       : trailing italic note line. None -> auto (a caption if given,
        plus a legend for whatever flags actually appear). "" -> no note.

    Returns the markdown string. print() it, or write it to a .md file.
    """
    registry = registry or COLUMN_REGISTRY
    columns = columns or DEFAULT_COLUMNS
    scales = scales or {}
    colnames = _normalize_columns(columns, registry)

    # ---- build rows ------------------------------------------------------
    rows = []
    for s in subsamples:
        label = str(s.get("label", "?"))
        if split_label and ": " in label:
            recipe, side = label.split(": ", 1)
        else:
            recipe, side = "", label
        cells = {}
        for header, key in colnames:
            spec = registry[key]
            fid, lo, hi = spec["get"](s)
            cells[key] = _fmt_asym_md(fid, lo, hi,
                                      sig=spec.get("sig", sig),
                                      scale=scales.get(key, spec["scale"]),
                                      math=math)
        rows.append({"recipe": recipe, "side": side,
                     "n": ("" if s.get("n_gal") is None else str(int(s["n_gal"]))),
                     "flags": _row_flags(s), "cells": cells})

    any_recipe = any(r["recipe"] for r in rows)
    any_flags = show_flags and any(r["flags"] for r in rows)

    # ---- header ----------------------------------------------------------
    head, keys, numeric = [], [], []
    if split_label and any_recipe:
        head.append("Recipe"); keys.append("recipe"); numeric.append(False)
    head.append("Subsample"); keys.append("side"); numeric.append(False)
    if show_n:
        head.append("N"); keys.append("n"); numeric.append(True)
    for header, key in colnames:
        head.append(_md_header(key, header, registry, scales, math, unit_in_header))
        keys.append(key); numeric.append(True)
    if any_flags:
        head.append("flags"); keys.append("flags"); numeric.append(False)

    sep = [("---:" if (n and align == "right") else ":---") for n in numeric]

    # ---- assemble --------------------------------------------------------
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(sep) + "|"]
    for r in rows:
        vals = []
        for k in keys:
            vals.append(r[k] if k in ("recipe", "side", "n", "flags") else r["cells"][k])
        lines.append("| " + " | ".join(vals) + " |")

    # ---- note ------------------------------------------------------------
    if note is None:
        bits = []
        if caption:
            bits.append(caption)
        seen = sorted({t for r in rows for t in r["flags"].split(", ") if t})
        if seen:
            bits.append("flags: " + ", ".join(seen))
        bits.append("errors are bootstrap 16/84 bounds; `--` = not measured")
        note = " — ".join(bits)
    if note:
        lines += ["", f"*{note}*"]
    return "\n".join(lines)


# =====================================================================
# FIGURE STYLE -- house tick style (plot style lives here, not plotting.py)
# =====================================================================

def apply_tick_style(ax, direction="in", right=True, top=False, minor=False,
                     labelsize=None):
    """
    Apply the house tick style to one Axes or an array of Axes.

    Default is the journal convention: ticks point INTO the plot and the
    y-axis ticks are mirrored on the right spine (ticks only -- no right-side
    tick labels). Call it on every new figure's axes:

        from utils_lya_halo.formatting import apply_tick_style
        fig, ax = plt.subplots()
        ...
        apply_tick_style(ax)                    # or apply_tick_style(axes)

    Parameters
    ----------
    ax        : Axes, or any (nested) list/array of Axes; None entries skipped.
    direction : "in" (default), "out", or "inout".
    right     : mirror y ticks on the right spine (default True).
    top       : mirror x ticks on the top spine (default False -- several of
                our figures put a secondary R/R_vir axis up there).
    minor     : turn minor ticks on (default False). Existing minor ticks
                (e.g. log axes) are styled either way.
    labelsize : optional major tick-label font size.

    No matplotlib import needed here: it only calls methods on the Axes.
    """
    stack = [ax]
    flat = []
    while stack:
        a = stack.pop()
        if a is None:
            continue
        if hasattr(a, "tick_params"):
            flat.append(a)
        else:
            stack.extend(np.asarray(a, dtype=object).ravel().tolist())
    for a in flat:
        if minor:
            a.minorticks_on()
        a.tick_params(axis="both", which="both", direction=direction,
                      right=right, top=top)
        if labelsize is not None:
            a.tick_params(axis="both", which="major", labelsize=labelsize)
    return ax
