"""
utils_lya_halo/splits.py -- Layers B and C of the subsample-split pipeline.

Implements ../../specs/subsample-split-pipeline.md (v2.1, 2026-07-31). Layer A
(the split catalog) lives in multicat.py and is NOT duplicated here.

    Layer A  multicat.build_split_catalog     one row per galaxy, every
                                              splittable column, provenance
    Layer B  LIBRARY / a recipe dict          WHAT to split on, declaratively
    Layer C  run()                            slice -> stack -> bootstrap ->
                                              fit -> derived numbers -> cache

WHAT THIS MODULE DOES NOT DO: any new measurement. Every number it reports comes
from stack.py / measure.py / fitting.py / analysis.py exactly as a hand-run
notebook split would call them. This is orchestration, caching and bookkeeping,
and the acceptance test for it (spec Sec 10, step 6) is that a hand-run split and
a runner-run split agree to the last digit -- see check_cut_equivalence() for the
cutting half of that check.

THE ONE THING TO NOT BREAK: the cache key includes config.extract_signature().
The SMHM default changed 2026-07-27 and the PSF area convention changed
2026-07-22; a results cache that does not invalidate on the extraction signature
will serve numbers from two different pipelines in the same table, and both will
look individually plausible. See _cache_key / _side_paths.

Quick start
-----------
    from utils_lya_halo import splits

    splits.show()                                   # what's in the library
    splits.preview(cfg, prod, splitcat=sc)          # seconds: N/side + warnings
    splits.independence(sc)                         # how redundant are they
    res = splits.run(cfg, prod, "mass_half", splitcat=sc)
    splits.describe(res["mass_half"])
    print(splits.table(res))
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import traceback
import warnings
from collections import OrderedDict

import numpy as np

from . import analysis, fitting, formatting, measure, multicat, selection, stack
from .analysis import _psf_fwhm_arcsec_to_kpc

__all__ = [
    "LIBRARY", "RECIPE_DEFAULTS", "QUANTITIES", "LEVELS", "SPLITS_DIR",
    "TWO_HALO_QUANTITIES",
    "normalize_recipe", "recipe_hash", "resolve_recipes", "show",
    "is_unreviewed", "UNREVIEWED_BY",
    "save_library", "load_library", "use_recipes", "edit_recipe",
    "cache_status", "why_miss",
    "resolve_eligibility", "cut", "check_cut_equivalence",
    "preview", "independence", "plot_independence",
    "fit_profile", "run", "intersect",
    "describe", "table", "load", "clear_cache",
]


# =====================================================================
# 1.  LAYER B -- THE RECIPE
# =====================================================================
# A recipe is a plain dict: serializable, hashable, no code. It describes one
# split and does nothing by itself. Only name/column/mode are required.
#
# The defaults below ARE the schema -- normalize_recipe rejects any key not
# listed here, because a typo'd key ("controls", "n_bin") that silently did
# nothing would produce a perfectly plausible wrong answer.

RECIPE_DEFAULTS = OrderedDict([
    ("name",           None),   # unique key; also the cache filename stem
    ("column",         None),   # split-catalog column to cut on
    ("mode",           "half"), # half | trend | bounds | groups
    ("n_bins",         None),   # trend only; None -> 4
    ("bounds",         None),   # bounds only; explicit value edges
    ("groups",         None),   # groups only; list of categories/category lists
    ("labels",         None),   # display labels per side (None -> auto)
    ("control",        None),   # column held fixed while splitting
    ("n_control_bins", 4),
    ("base",           "CLEAN"),# CLEAN | KEEP | ALL, or any boolean column
    ("require",        ()),     # extra per-recipe cut expressions, e.g. "FLAG == 0"
    ("needs_core",     None),   # None | "covered" | "detected"
    ("min_n",          80),     # WARN below this. Never refuses.
    ("per_field",      False),  # cut within each field, then pool
    ("field_col",      "FIELD"),
    ("predicted",      ""),     # directional pre-specification (recorded, not enforced)
    ("predicted_by",   None),   # WHO wrote `predicted`. None or "claude-draft"
                                # means nobody has reviewed it -- see UNREVIEWED_BY
    ("confirmatory",   False),  # True = pre-registered; False = exploratory
    ("family",         None),   # 1 (mass sequence) | 2 (external) | 3 (Lya-internal)
    ("decomposition",  False),  # True = a decomposition of a headline result,
                                #        not an independent test (spec Sec 7.5a)
    ("doc",            ""),     # pointer back to docs/subsample_splits.md
])

_MODES = ("half", "trend", "bounds", "groups")
_CORE_TIERS = (None, "covered", "detected")
LEVELS = ("quick", "standard", "full")


def _n_sides(rec) -> int:
    if rec["mode"] == "groups":
        return len(rec["groups"])
    if rec["mode"] == "bounds":
        return len(rec["bounds"]) - 1
    if rec["mode"] == "trend":
        return int(rec["n_bins"])
    return 2


def normalize_recipe(recipe, **overrides) -> OrderedDict:
    """Validate a recipe dict and fill its defaults. Returns a NEW OrderedDict
    with keys in RECIPE_DEFAULTS order, so two recipes that differ only in key
    order hash identically.

    Raises on: unknown keys, a missing name/column, an unknown mode, a mode
    missing its required argument, an unknown needs_core tier. All of these are
    failures that would otherwise show up as a silently-wrong sample.
    """
    if isinstance(recipe, str):
        recipe = LIBRARY[recipe]
    rec = dict(recipe)
    rec.update(overrides)

    unknown = [k for k in rec if k not in RECIPE_DEFAULTS]
    if unknown:
        raise KeyError(
            f"unknown recipe key(s) {unknown}. Valid keys: "
            f"{list(RECIPE_DEFAULTS)}. (A typo'd key would otherwise be "
            f"silently ignored and you would get the DEFAULT behaviour under a "
            f"name that says otherwise.)")

    out = OrderedDict((k, rec.get(k, v)) for k, v in RECIPE_DEFAULTS.items())

    if not out["name"]:
        raise ValueError("recipe needs a `name` (it is the cache filename).")
    if not out["column"]:
        raise ValueError(f"recipe {out['name']!r} needs a `column`.")
    if out["mode"] not in _MODES:
        raise ValueError(f"recipe {out['name']!r}: mode must be one of {_MODES} "
                         f"(got {out['mode']!r}).")
    if out["mode"] == "trend":
        out["n_bins"] = int(out["n_bins"] or 4)
        if out["n_bins"] < 2:
            raise ValueError(f"recipe {out['name']!r}: trend needs n_bins >= 2.")
    if out["mode"] == "bounds" and out["bounds"] is None:
        raise ValueError(f"recipe {out['name']!r}: mode='bounds' needs `bounds` "
                         f"(explicit value edges, in the column's own units).")
    if out["mode"] == "groups" and out["groups"] is None:
        raise ValueError(f"recipe {out['name']!r}: mode='groups' needs `groups`.")
    if out["mode"] == "groups" and out["control"]:
        raise ValueError(f"recipe {out['name']!r}: a categorical split cannot be "
                         f"rank-controlled; drop `control` or use a matched "
                         f"`require` instead.")
    if out["needs_core"] not in _CORE_TIERS:
        raise ValueError(f"recipe {out['name']!r}: needs_core must be one of "
                         f"{_CORE_TIERS} (got {out['needs_core']!r}).")

    out["require"] = tuple(out["require"] or ())
    if out["bounds"] is not None:
        out["bounds"] = [float(b) for b in out["bounds"]]
    if out["labels"] is not None:
        out["labels"] = [str(s) for s in out["labels"]]

    # LABEL COUNT vs SIDE COUNT. Checked here, not at cut time, because the
    # usual way to get this wrong is to change `bounds` or `n_bins` and forget
    # that `labels` was written for the old number of sides -- and you want to
    # hear about that the moment you make the edit, not after the eligibility
    # step has already run.
    n_sides = _n_sides(out)
    if out["labels"] is not None and len(out["labels"]) != n_sides:
        raise ValueError(
            f"recipe {out['name']!r}: {len(out['labels'])} labels "
            f"{out['labels']} but mode={out['mode']!r} makes {n_sides} side(s)"
            + (f" (bounds={out['bounds']} -> len(bounds)-1 = {n_sides})"
               if out["mode"] == "bounds" else
               (f" (n_bins={out['n_bins']})" if out["mode"] == "trend" else ""))
            + ". Update labels= to match, or pass labels=None for automatic ones.")
    if out["mode"] == "bounds" and n_sides < 2:
        raise ValueError(
            f"recipe {out['name']!r}: bounds={out['bounds']} defines only "
            f"{n_sides} bin. N edges make N-1 bins, so you need at least 3 "
            f"values -- e.g. [0, 9.4, 99] for a two-way cut at 9.4.")
    return out


def _canonical(rec) -> str:
    """JSON with sorted keys -- the exact string the recipe hash is taken of."""
    return json.dumps({k: v for k, v in normalize_recipe(rec).items()
                       if k not in ("doc", "predicted", "predicted_by")},
                      sort_keys=True, default=str)
    # doc/predicted/predicted_by are prose and provenance. Fixing a typo in a
    # docstring, or signing your name to a prediction, must NOT invalidate a
    # 6-hour batch -- none of them can change a single number.


def recipe_hash(recipe) -> str:
    """12-hex-char hash of everything in a recipe that can change a number."""
    return hashlib.sha1(_canonical(recipe).encode()).hexdigest()[:12]


def resolve_recipes(recipes=None) -> "OrderedDict[str, OrderedDict]":
    """Accept a name, a recipe dict, or any list/dict/None of either, and return
    OrderedDict{name: normalized recipe}. This is what lets run() take
    "all of them" and "just this one" as the same call."""
    if recipes is None:
        recipes = LIBRARY
    if isinstance(recipes, (str, dict)) and not _looks_like_library(recipes):
        recipes = [recipes]
    if isinstance(recipes, dict):
        recipes = list(recipes.values())
    out = OrderedDict()
    for r in recipes:
        rec = normalize_recipe(r)
        if rec["name"] in out:
            raise ValueError(f"duplicate recipe name {rec['name']!r} -- names "
                             f"are cache filenames and must be unique.")
        out[rec["name"]] = rec
    return out


def _looks_like_library(obj) -> bool:
    """A dict of recipes (name -> dict) vs a single recipe dict."""
    return (isinstance(obj, dict) and bool(obj)
            and all(isinstance(v, dict) for v in obj.values()))


# =====================================================================
# 2.  LAYER B2 -- THE RECIPE LIBRARY
# =====================================================================
# The 15 candidate splits of ../../docs/subsample_splits.md, written down once.
# Adding the sixteenth split should be one entry here, not a notebook cell.
#
# `family` follows spec Sec 7.2:
#   1 = the mass sequence (mass, SFR, sSFR, SigmaSFR, size, dust) -- effective
#       dimensionality ~2 no matter how many you run
#   2 = externally-set (inclination, environment, z, field) -- ~full count
#   3 = Lya-internal (dv, FWHM, EW, L_Lya, f_esc) -- ~2, plus circularity
# Pick across families, not within one.

LIBRARY = OrderedDict()
# EMPTY HERE ON PURPOSE. The recipes live in utils_lya_halo/splits_recipes.py --
# one file, all of them, with real comments explaining every edge and every
# `require` cut. They are loaded into LIBRARY at the bottom of this section.
#
# They used to be defined right here, with a JSON file layered on top as an
# override. The argument for that indirection was "don't edit the shipped library
# in place or your copy diverges from future versions" -- which is the right
# argument for a third-party package and the wrong one here, because there is no
# upstream. What it actually bought was a recipe list split between a file that
# looked empty and 160 lines buried in the middle of this module. See
# splits_recipes.py's docstring.

# WHO WROTE THE PREDICTIONS. Every `predicted` string that ships with
# splits_recipes.py was drafted by Claude from ../docs/subsample_splits.md, and a
# prediction you did not write is not a pre-registration -- it is a guess with
# your name on it. So they are all stamped, and describe()/show() flag any recipe
# still carrying an unreviewed one. Clear it by rewriting the prediction in
# splits_recipes.py and setting predicted_by="austin" on that recipe.
UNREVIEWED_BY = (None, "", "claude-draft")


def _load_shipped_recipes():
    """Populate LIBRARY from utils_lya_halo/splits_recipes.py at import time.

    Done at import so `splits.show()` / `preview()` / `run()` work immediately in
    a fresh kernel with nothing to remember to call first. use_recipes() is still
    how you push EDITS in after an importlib.reload -- this is just the initial
    load, and it is deliberately non-fatal: if splits_recipes.py has a syntax
    error or a bad recipe, importing splits.py should tell you that clearly
    rather than failing somewhere unrelated three imports up the chain.
    """
    try:
        from . import splits_recipes
    except Exception as exc:                                      # noqa: BLE001
        warnings.warn(f"splits: could not import splits_recipes.py "
                      f"({type(exc).__name__}: {exc}). LIBRARY is EMPTY -- "
                      f"preview()/run() will have nothing to run.", stacklevel=2)
        return
    try:
        LIBRARY.update(resolve_recipes(splits_recipes.RECIPES))
    except Exception as exc:                                      # noqa: BLE001
        warnings.warn(f"splits: splits_recipes.RECIPES failed validation "
                      f"({type(exc).__name__}: {exc}). LIBRARY is EMPTY.",
                      stacklevel=2)
        return
    for _n in LIBRARY:
        if LIBRARY[_n]["predicted"] and LIBRARY[_n]["predicted_by"] is None:
            LIBRARY[_n]["predicted_by"] = "claude-draft"


_load_shipped_recipes()


def is_unreviewed(recipe) -> bool:
    """True if this recipe carries a prediction nobody has signed off on --
    i.e. it still has whatever Claude drafted. Not an error; a flag."""
    rec = normalize_recipe(recipe)
    return bool(rec["predicted"]) and rec["predicted_by"] in UNREVIEWED_BY


# ---------------------------------------------------------------------
# EDITING RECIPES
# ---------------------------------------------------------------------
# Edit utils_lya_halo/splits_recipes.py. That is the whole library, and it is
# already loaded into LIBRARY above by the time you import this module.
#
#     importlib.reload(splits_recipes)
#     splits.use_recipes(splits_recipes.RECIPES)     # push edits into LIBRARY
#
# The JSON functions below (save_library / load_library / edit_recipe) still
# work and are still useful for ONE thing: shipping a recipe set to someone else,
# or freezing the exact set a paper's numbers came from alongside the table.
# They are no longer part of the normal editing loop -- a round trip through JSON
# means LIBRARY-in-memory and the file on disk can disagree with nothing telling
# you which one produced your numbers.
#
# Merging is by NAME in all three cases: an entry whose name matches an existing
# one REPLACES it, any other name is ADDED.

def save_library(path, recipes=None, *, verbose=True) -> str:
    """Write recipes to JSON so you can edit them outside the code.

    Default writes the whole shipped LIBRARY, which gives you a
    correctly-shaped starting point with every field and every `predicted`
    string already filled in. Trim it down to just what you change.
    """
    recs = resolve_recipes(recipes)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"recipes": [dict(r) for r in recs.values()]}, fh, indent=2,
                  default=str)
    if verbose:
        print(f"wrote {len(recs)} recipe(s) -> {path}\n"
              f"  edit that file, then splits.load_library({path!r}) each session.")
    return path


def load_library(path, *, merge=True, apply=True, verbose=True):
    """
    Read recipes from JSON and (by default) merge them into LIBRARY in place,
    so every later `splits.run(...)` / `preview(...)` call sees them.

    merge=True  : matching names REPLACE the shipped entry, new names are added,
                  shipped entries you did not mention are left alone.
    merge=False : the file becomes the WHOLE library -- anything not in it is
                  dropped. Use this when you want the notebook's recipe set to
                  be exactly what the file says and nothing else.
    apply=False : validate and return without touching LIBRARY (a dry run).

    Every recipe is put through normalize_recipe on the way in, so a typo'd key
    or a bad mode raises HERE, naming the file, rather than three hours into a
    batch. Returns OrderedDict{name: recipe}.
    """
    with open(path) as fh:
        raw = json.load(fh)
    items = raw["recipes"] if isinstance(raw, dict) and "recipes" in raw else raw
    if isinstance(items, dict):
        items = [dict(v, name=v.get("name", k)) for k, v in items.items()]
    try:
        loaded = resolve_recipes(items)
    except Exception as exc:                                   # noqa: BLE001
        raise ValueError(f"{path}: {exc}") from exc

    replaced = [n for n in loaded if n in LIBRARY]        # BEFORE the update
    if apply:
        if not merge:
            LIBRARY.clear()
        LIBRARY.update(loaded)
    if verbose:
        print(f"loaded {len(loaded)} recipe(s) from {path}"
              + (f"  ({len(replaced)} replaced a shipped entry, "
                 f"{len(loaded) - len(replaced)} new)" if apply and merge else ""))
        if apply:
            print(f"  LIBRARY now holds {len(LIBRARY)} recipes.")
            if replaced and merge:
                print(f"  replaced: {replaced}")
    return loaded


def use_recipes(recipes, *, merge=True, apply=True, verbose=True):
    """
    load_library(), but from a PYTHON OBJECT instead of a JSON file.

    This is the entry point for keeping your overrides in a module
    (utils_lya_halo/splits_recipes.py) rather than a JSON blob. Same merge
    semantics as load_library -- matching names replace the shipped entry, new
    names are added -- and the same normalize_recipe validation on the way in,
    so a typo'd key raises here rather than three hours into a batch.

        from utils_lya_halo import splits, splits_recipes
        splits.use_recipes(splits_recipes.RECIPES)

    Why a module beats the JSON file: JSON cannot hold comments (so the reasoning
    behind an edge or a `require` has nowhere to live), and the save/load round
    trip means LIBRARY-in-memory and the file on disk can drift with nothing
    telling you which one you are running. A module is imported, so there is only
    ever one version of it. The property the JSON was protecting -- that a
    `predicted` string is committed BEFORE you see the numbers, which is what
    makes it a pre-registration rather than a post-hoc story -- is preserved by
    git, not by the file format.

    `recipes` takes anything resolve_recipes takes: a dict of name->recipe, a
    list of recipe dicts, or a single recipe.
    """
    loaded = resolve_recipes(recipes)
    replaced = [n for n in loaded if n in LIBRARY]        # BEFORE the update
    if apply:
        if not merge:
            LIBRARY.clear()
        LIBRARY.update(loaded)
    if verbose:
        print(f"applied {len(loaded)} recipe(s) from a Python object"
              + (f"  ({len(replaced)} replaced a shipped entry, "
                 f"{len(loaded) - len(replaced)} new)" if apply and merge else ""))
        if apply:
            print(f"  LIBRARY now holds {len(LIBRARY)} recipes.")
            if replaced and merge:
                print(f"  replaced: {replaced}")
            unrev = [n for n, r in loaded.items() if is_unreviewed(r)]
            if unrev:
                print(f"  still carrying Claude's draft prediction: {unrev}\n"
                      f"    (set predicted_by='austin' once you have rewritten them)")
    return loaded


def edit_recipe(path, name, *, verbose=True, **changes):
    """
    Change one recipe and save it, from a notebook cell -- no text editor.

        splits.edit_recipe(RECIPES_PATH, "dust_massctrl", column="EBV")
        splits.edit_recipe(RECIPES_PATH, "inclination", require=[],
                           predicted="face-on -> more escape",
                           predicted_by="austin")

    Reads `path` (creating it from the current LIBRARY if it does not exist yet),
    applies `changes` to the named recipe, validates the result, writes the file
    back, and reloads it into LIBRARY so the very next preview()/run() sees it.
    One call, no restart.

    `name` may be a recipe that only exists in the shipped LIBRARY -- it gets
    copied into your file on first edit, which is exactly the behaviour you want:
    your file grows to contain only the recipes you have actually touched.

    To create a brand-new recipe, pass every required field:
        splits.edit_recipe(PATH, "my_split", column="AGE_50", mode="half")
    """
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        items = raw["recipes"] if isinstance(raw, dict) and "recipes" in raw else raw
        current = OrderedDict((r["name"], r) for r in items)
    else:
        current = OrderedDict()

    base = current.get(name) or LIBRARY.get(name) or {"name": name}
    updated = normalize_recipe(dict(base, name=name), **changes)   # validates here
    was_new = name not in current
    current[name] = dict(updated)

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"recipes": list(current.values())}, fh, indent=2, default=str)

    LIBRARY[name] = updated
    if verbose:
        what = ", ".join(f"{k}={v!r}" for k, v in changes.items())
        print(f"{'added' if was_new else 'updated'} {name!r} in {path}")
        print(f"  {what}")
        if is_unreviewed(updated):
            print("  (prediction is still Claude's draft -- pass "
                  "predicted_by='austin' when you rewrite it)")
    return updated


def show(recipes=None, *, verbose=True):
    """Print the library (or any recipe set): name, column, mode, family,
    eligibility, tier, and the pre-specified prediction. No data needed --
    this is the "what could I run" view; preview() is the "what would it cost"
    view. Returns the resolved OrderedDict."""
    recs = resolve_recipes(recipes)
    if not verbose:
        return recs
    fam_name = {1: "mass-seq", 2: "external", 3: "Lya-int", None: "-"}
    print(f"{'name':<20} {'column':<16} {'mode':<7} {'ctrl':<9} {'base':<6} "
          f"{'core':<9} {'family':<9} tier")
    print("-" * 104)
    for r in recs.values():
        tier = ("confirmatory" if r["confirmatory"] else
                ("decomposition" if r["decomposition"] else "exploratory"))
        print(f"{r['name']:<20} {r['column']:<16} {r['mode']:<7} "
              f"{str(r['control'] or '-'):<9} {r['base']:<6} "
              f"{str(r['needs_core'] or '-'):<9} "
              f"{fam_name.get(r['family'], '?'):<9} {tier}")
        if r["predicted"]:
            mark = "  [UNREVIEWED DRAFT]" if is_unreviewed(r) else ""
            print(f"    predicted: {r['predicted']}{mark}")
    n_unrev = sum(is_unreviewed(r) for r in recs.values())
    if n_unrev:
        print(f"\n** {n_unrev} prediction(s) are still Claude's draft. A prediction "
              f"you did not write is not a pre-registration.\n"
              f"   Rewrite the ones you care about in splits_recipes.py and set "
              f"predicted_by=\"austin\" on those recipes to clear the flag.")
    print(f"\n{len(recs)} recipes. Families present: "
          + ", ".join(f"{fam_name.get(f, '?')}="
                      f"{sum(1 for r in recs.values() if r['family'] == f)}"
                      for f in (1, 2, 3)))
    print("Spec Sec 7.2: ~15 recipes buy ~7 effective independent axes. Pick "
          "ACROSS families, not within one.")
    return recs


# =====================================================================
# 3.  ELIGIBILITY  (spec Sec 3.2 / 3.3)
# =====================================================================

def resolve_eligibility(catalog, recipe, *, verbose=False) -> dict:
    """
    Which galaxies this recipe is allowed to use, and what removed the rest.

    Applied in this fixed order (spec Sec 3.2):
      1. the named base mask  (CLEAN by default; KEEP; ALL; or any bool column)
      2. CORE_COVERED  -- ONLY if needs_core is set, or the column is a CORE_ one
      3. every expression in `require`
      4. finite `column` (and finite `control`, if set)

    Step 2 is the load-bearing one and the rule is deliberately strict: a
    catalog split must NOT inherit a core cut. A galaxy with a good mass and no
    detected Lya core belongs in the mass split; dropping it would shrink the
    sample for no reason AND silently make the mass split run on the
    Lya-DETECTED subpopulation -- a selection on Lya brightness imposed on a
    test that has nothing to do with it.

    Returns {'mask', 'steps' (list of (what, n_left)), 'n', 'blocked'}.
    `blocked` is a string when the recipe cannot run at all on this catalog
    (a missing column) -- preview() reports it instead of raising, so one
    unbuilt column does not stop a 15-recipe preview.
    """
    rec = normalize_recipe(recipe)
    cols = set(catalog.colnames)
    n = len(catalog)
    mask = np.ones(n, dtype=bool)
    steps = [("start", n)]

    def _blocked(msg):
        return {"mask": np.zeros(n, bool), "steps": steps, "n": 0, "blocked": msg,
                "recipe": rec}

    # --- 1. base sample -------------------------------------------------
    base = str(rec["base"] or "ALL")
    if base.upper() != "ALL":
        if base not in cols:
            return _blocked(f"base column {base!r} not on the catalog")
        mask &= np.asarray(catalog[base]).astype(bool)
        steps.append((f"base={base}", int(mask.sum())))

    # --- 2. core dependence --------------------------------------------
    need = rec["needs_core"]
    if need is None and str(rec["column"]).startswith("CORE_"):
        need = "covered"          # a CORE_ column is meaningless without coverage
    if need is not None:
        if "CORE_COVERED" not in cols:
            return _blocked(
                f"needs_core={need!r} but CORE_COVERED is not on the catalog -- "
                f"the split catalog was built with core=None. Rebuild it with a "
                f"core product (multicat.build_split_catalog(prod, core=...)).")
        mask &= np.asarray(catalog["CORE_COVERED"]).astype(bool)
        steps.append(("CORE_COVERED", int(mask.sum())))
        if need == "detected":
            if "CORE_DETECTED" not in cols:
                return _blocked("needs_core='detected' but CORE_DETECTED is not "
                                "on the catalog")
            mask &= np.asarray(catalog["CORE_DETECTED"]).astype(bool)
            steps.append(("CORE_DETECTED", int(mask.sum())))

    # --- 3. per-recipe require expressions -------------------------------
    for expr in rec["require"]:
        ns = selection._expr_namespace(catalog)
        try:
            val = eval(expr, {"__builtins__": {}}, ns)          # noqa: S307
        except Exception as exc:                                # noqa: BLE001
            return _blocked(f"require {expr!r} failed to evaluate: {exc}")
        v = np.asarray(val)
        if v.dtype != bool or v.shape[0] != n:
            return _blocked(f"require {expr!r} did not produce a length-{n} "
                            f"boolean array")
        mask &= v
        steps.append((f"require {expr}", int(mask.sum())))

    # --- 4. finite split (and control) variable --------------------------
    for what, colname in (("column", rec["column"]), ("control", rec["control"])):
        if colname is None:
            continue
        if colname not in cols:
            return _blocked(f"{what} {colname!r} not on the catalog")
        mask &= _available(catalog[colname])
        steps.append((f"finite {colname}", int(mask.sum())))

    out = {"mask": mask, "steps": steps, "n": int(mask.sum()),
           "blocked": None, "recipe": rec}
    if verbose:
        print(f"[{rec['name']}] eligibility")
        for what, k in steps:
            print(f"    {what:<34} {k:>5}")
    return out


def _available(col) -> np.ndarray:
    """'Has a usable value': finite for numbers, True/False for booleans (both
    count -- False is a value), non-empty for strings. Mirrors
    multicat.split_catalog_report's own availability rule so a coverage number
    and an eligibility number never disagree."""
    a = np.asarray(col)
    if a.dtype == bool:
        return np.ones(a.shape[0], bool)
    if np.issubdtype(a.dtype, np.number):
        return np.isfinite(a.astype(float))
    return np.array([str(v).strip() != "" for v in a])


# =====================================================================
# 4.  THE CUT  (spec Sec 5.2)
# =====================================================================

def _value_masks(vals, edges):
    """Masks for value bins [e0,e1), [e1,e2), ..., [e_{n-1}, e_n] -- LAST BIN
    INCLUSIVE of its upper edge, non-finite dropped.

    This replicates multicat.split_product_by's rule EXACTLY, character for
    character, because the runner's acceptance test (spec Sec 10 step 6) is that
    a runner-run split reproduces a hand-run one to the last digit, and a
    hand-run split goes through split_product_by. check_cut_equivalence() runs
    both and asserts the ID sets match -- run it after touching either.
    """
    v = np.asarray(vals, dtype=float)
    e = np.asarray(edges, dtype=float)
    out = []
    for i in range(len(e) - 1):
        last = (i == len(e) - 2)
        out.append(np.isfinite(v) & (v >= e[i]) &
                   ((v <= e[i + 1]) if last else (v < e[i + 1])))
    return out


def _percentile_edges(vals, n_bins):
    """Equal-population value edges. Same nanpercentile call split_product_by
    makes for percentiles=, so `half` here == percentiles=50 there."""
    v = np.asarray(vals, dtype=float)
    return np.nanpercentile(v[np.isfinite(v)], np.linspace(0, 100, n_bins + 1))


def _group_masks(vals, groups):
    """Categorical membership, string-compared. Same rule as split_product_by --
    which matters because FITS stores strings as bytes and a bytes-vs-str
    comparison returns empty groups with NO error (see
    multicat.read_split_catalog)."""
    s = np.asarray(vals).astype(str)
    out = []
    for g in groups:
        allowed = {str(x) for x in (g if isinstance(g, (list, tuple, set)) else [g])}
        out.append(np.array([v in allowed for v in s]))
    return out


def cut(product, recipe, *, eligible=None, verbose=True) -> "OrderedDict[str, dict]":
    """
    Turn one recipe into labelled sides, on a product whose catalog already
    carries the split columns (multicat.attach_split_catalog).

    Returns OrderedDict{label: {
        'mask'      : bool array on the PARENT product's galaxy axis,
        'product'   : the sliced GalaxyProduct,
        'n'         : galaxy count,
        'cut_lo'/'cut_hi'/'cut_value' : realized cut edges in the column's OWN
                      PHYSICAL UNITS -- recorded ALWAYS, including for percentile
                      modes, so a percentile cut can be reported in physical
                      units and compared across reruns (spec Sec 3.1),
        'median'    : realized median of the split variable on this side,
        'control_median' : realized median of the control variable (None if
                      uncontrolled) -- the residual control offset between sides
                      is the number that says whether the control worked,
    }}
    """
    rec = normalize_recipe(recipe)
    cat = product.catalog
    if eligible is None:
        el = resolve_eligibility(cat, rec)
        if el["blocked"]:
            raise KeyError(f"recipe {rec['name']!r} cannot run: {el['blocked']}")
        eligible = el["mask"]
    eligible = np.asarray(eligible, bool)

    vals = np.asarray(cat[rec["column"]])
    ctrl = rec["control"]
    fieldcol = rec["field_col"]
    per_field = bool(rec["per_field"])
    if per_field and fieldcol not in cat.colnames:
        raise KeyError(f"recipe {rec['name']!r} has per_field=True but "
                       f"{fieldcol!r} is not on the catalog.")

    # which "blocks" the cut is computed within: one block (pooled) normally,
    # one per field when per_field=True.
    if per_field:
        fvals = np.asarray(cat[fieldcol]).astype(str)
        blocks = [(f, eligible & (fvals == f))
                  for f in sorted(set(fvals[eligible]))]
    else:
        blocks = [(None, eligible)]

    n = len(cat)
    n_sides = _n_sides(rec)
    side_masks = [np.zeros(n, bool) for _ in range(n_sides)]
    per_block_edges = OrderedDict()

    for fname, bmask in blocks:
        if not bmask.any():
            continue
        edges = None
        if rec["mode"] == "groups":
            gm = _group_masks(vals, rec["groups"])
            masks = [g & bmask for g in gm]
        elif ctrl:
            sub_idx = np.flatnonzero(bmask)
            subcat = cat[sub_idx]
            if rec["mode"] == "trend":
                mm = selection.split_controlled_bins(
                    subcat, rec["column"], ctrl, n_bins=rec["n_bins"],
                    n_control_bins=rec["n_control_bins"], verbose=False)
            else:
                lo, hi = selection.split_controlled(
                    subcat, rec["column"], ctrl,
                    n_control_bins=rec["n_control_bins"], method="median",
                    balance=True, verbose=False)
                mm = [lo, hi]
            masks = []
            for m in mm:
                full = np.zeros(n, bool)
                full[sub_idx[np.asarray(m, bool)]] = True
                masks.append(full)
        else:
            if rec["mode"] == "bounds":
                edges = np.asarray(rec["bounds"], float)
            else:
                nb = rec["n_bins"] if rec["mode"] == "trend" else 2
                edges = _percentile_edges(vals[bmask], nb)
            masks = [m & bmask for m in _value_masks(vals, edges)]
        if len(masks) != n_sides:
            raise RuntimeError(f"internal: {len(masks)} masks for {n_sides} sides")
        for i, m in enumerate(masks):
            side_masks[i] |= m
        per_block_edges[fname] = (None if edges is None
                                  else [float(e) for e in edges])

    labels = _labels_for(rec, per_block_edges)
    out = OrderedDict()
    for i, (lab, m) in enumerate(zip(labels, side_masks)):
        e = per_block_edges.get(None)
        lo = hi = None
        if e is not None:
            lo, hi = float(e[i]), float(e[i + 1])
        v = vals[m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = (float(np.nanmedian(v.astype(float)))
                   if m.any() and np.issubdtype(vals.dtype, np.number) else None)
            cmed = (float(np.nanmedian(np.asarray(cat[ctrl], float)[m]))
                    if (ctrl and m.any()) else None)
        out[lab] = {
            "mask": m, "product": selection.slice_product(product, m),
            "n": int(m.sum()), "cut_lo": lo, "cut_hi": hi,
            "cut_value": hi if (i == 0 and hi is not None) else lo,
            "median": med, "control_median": cmed,
            "per_field_edges": (per_block_edges if per_field else None),
        }
        if verbose:
            edge_s = (f"  [{lo:.4g}, {hi:.4g}]" if lo is not None else "")
            print(f"  {rec['name']}/{lab}: {int(m.sum())} galaxies{edge_s}")
    return out


def _labels_for(rec, per_block_edges) -> list:
    if rec["labels"]:
        if len(rec["labels"]) != _n_sides(rec):
            raise ValueError(f"recipe {rec['name']!r}: {len(rec['labels'])} labels "
                             f"for {_n_sides(rec)} sides.")
        return list(rec["labels"])
    n = _n_sides(rec)
    if rec["mode"] == "groups":
        return ["+".join(str(x) for x in (g if isinstance(g, (list, tuple)) else [g]))
                for g in rec["groups"]]
    e = per_block_edges.get(None)
    if rec["mode"] == "bounds" and e:
        return [f"[{e[i]:g},{e[i+1]:g}{']' if i == n-1 else ')'}" for i in range(n)]
    pcts = np.linspace(0, 100, n + 1)
    return [f"{pcts[i]:g}-{pcts[i+1]:g}%" for i in range(n)]


def check_cut_equivalence(product, recipe, *, id_col="ID", verbose=True) -> bool:
    """
    Acceptance test for spec Sec 10 step 6, cutting half: run this module's cut()
    and multicat.split_product_by on the SAME eligible sample and assert the two
    produce identical ID sets per side.

    Only applies to the uncontrolled, single-block cases split_product_by covers
    (control=None, per_field=False); returns True with a note otherwise. If this
    ever fails, every number the runner produced diverged from the hand-run
    equivalent at the very first step, and nothing downstream is trustworthy.
    """
    rec = normalize_recipe(recipe)
    if rec["control"] or rec["per_field"]:
        if verbose:
            print(f"[{rec['name']}] controlled/per-field cut -- "
                  f"split_product_by has no equivalent, nothing to compare.")
        return True
    el = resolve_eligibility(product.catalog, rec)
    if el["blocked"]:
        raise KeyError(el["blocked"])
    sub = selection.slice_product(product, el["mask"])
    mine = cut(product, rec, eligible=el["mask"], verbose=False)

    kw = {}
    if rec["mode"] == "groups":
        kw["groups"] = rec["groups"]
    elif rec["mode"] == "bounds":
        kw["edges"] = rec["bounds"]
    else:
        nb = rec["n_bins"] if rec["mode"] == "trend" else 2
        kw["percentiles"] = list(np.linspace(0, 100, nb + 1)[1:-1])
    theirs = multicat.split_product_by(sub, rec["column"], verbose=False, **kw)

    ok = True
    for (la, a), (lb, b) in zip(mine.items(), theirs.items()):
        ia = set(np.asarray(a["product"].catalog[id_col]).astype(str))
        ib = set(np.asarray(b.catalog[id_col]).astype(str))
        same = (ia == ib)
        ok &= same
        if verbose:
            print(f"  {la!r} vs {lb!r}: {len(ia)} vs {len(ib)} galaxies -- "
                  f"{'IDENTICAL' if same else '** DIFFERENT **'}")
    if verbose:
        print(f"[{rec['name']}] cut equivalence: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# =====================================================================
# 5.  PREVIEW  (spec Sec 5.5)  -- eligibility + cut + warnings, no stacking
# =====================================================================

def preview(config, product, recipes=None, *, splitcat=None, min_n=None,
            imbalance=0.60, control_tol=0.10, detail="summary",
            verbose=True) -> "OrderedDict[str, dict]":
    """
    Steps 1-3 only: eligibility, cut, warnings. NO stacking, NO bootstrap.
    Seconds for the whole library, versus ~20 minutes per recipe for run().

    This is how you decide which splits are worth running, and it should be the
    FIRST thing run after any change to the split catalog.

    Reports per recipe: N per side, the realized cut values in physical units,
    the separation in the split variable between the extreme sides, and for
    controlled recipes the residual control offset (want ~0). Warns on:
      * N below min_n (the recipe's own, or the global override)
      * side imbalance worse than `imbalance` (default 60/40)
      * control offset above `control_tol` x the split variable's own spread
      * eligibility dropping more than half the base sample
      * needs_core unsatisfiable / any missing column (reported, never raised)

    detail : "summary" (default) -- one line per recipe, warnings collapsed to
             short tags with each tag explained ONCE at the bottom. This is the
             readable view; the full text of a warning is written to be read the
             first time you hit it, not eleven times in one printout.
             "full" -- every warning in full, per recipe.

    Returns OrderedDict{name: {'n_per_side', 'cut', 'warnings', 'blocked', ...}}.
    The returned dict is identical either way; `detail` only changes printing.
    """
    recs = resolve_recipes(recipes)
    if splitcat is not None:
        product = multicat.attach_split_catalog(product, splitcat, verbose=False)
    cat = product.catalog
    out = OrderedDict()

    for name, rec in recs.items():
        floor = int(min_n if min_n is not None else rec["min_n"])
        el = resolve_eligibility(cat, rec)
        rowa = {"recipe": rec, "blocked": el["blocked"], "steps": el["steps"],
                "n_eligible": el["n"], "warnings": [], "sides": OrderedDict()}
        if el["blocked"]:
            out[name] = rowa
            if verbose:
                print(f"[{name}]  BLOCKED: {el['blocked']}")
            continue

        n_base = el["steps"][1][1] if len(el["steps"]) > 1 else el["steps"][0][1]
        if n_base and el["n"] < 0.5 * n_base:
            rowa["warnings"].append(
                f"eligibility keeps only {el['n']}/{n_base} of the base sample "
                f"({el['n']/n_base:.0%}) -- this recipe runs on a materially "
                f"different sample from the catalog-only splits; compare it via "
                f"intersect(), not directly")

        try:
            sides = cut(product, rec, eligible=el["mask"], verbose=False)
        except Exception as exc:                                # noqa: BLE001
            rowa["blocked"] = f"cut failed: {exc}"
            out[name] = rowa
            if verbose:
                print(f"[{name}]  BLOCKED: {rowa['blocked']}")
            continue

        ns = [s["n"] for s in sides.values()]
        for lab, s in sides.items():
            rowa["sides"][lab] = {k: s[k] for k in
                                  ("n", "cut_lo", "cut_hi", "median",
                                   "control_median")}
            if s["n"] < floor:
                rowa["warnings"].append(
                    f"{lab}: N={s['n']} < min_n={floor}. RUNNING ANYWAY -- "
                    f"whether a thin bin is usable depends on where the signal "
                    f"is (60 galaxies is useless for the two-halo term and fine "
                    f"for a core measurement); the power flag on each quantity "
                    f"is what keeps an underpowered number out of the paper")
        if ns and min(ns) > 0 and max(ns) / sum(ns) > imbalance:
            rowa["warnings"].append(
                f"side imbalance {max(ns)}/{sum(ns)} = {max(ns)/sum(ns):.0%} "
                f"(> {imbalance:.0%}) -- the S/N floor is set by the small side")
        if any(n == 0 for n in ns):
            rowa["warnings"].append("at least one side is EMPTY -- check the "
                                    "group labels/edges against the column's "
                                    "actual values")

        # separation in the split variable, extremes
        meds = [s["median"] for s in sides.values() if s["median"] is not None]
        rowa["separation"] = (float(meds[-1] - meds[0]) if len(meds) >= 2 else None)

        # controlled: residual offset in the CONTROL variable between extremes
        if rec["control"]:
            cm = [s["control_median"] for s in sides.values()
                  if s["control_median"] is not None]
            if len(cm) >= 2:
                off = float(max(cm) - min(cm))
                rowa["control_offset"] = off
                cvals = np.asarray(cat[rec["control"]], float)[el["mask"]]
                spread = float(np.nanpercentile(cvals, 84)
                               - np.nanpercentile(cvals, 16)) or 1.0
                if abs(off) > control_tol * spread:
                    rowa["warnings"].append(
                        f"control {rec['control']} still differs by {off:.3g} "
                        f"between the extreme sides ({off/spread:.0%} of its own "
                        f"16-84 spread) -- the control is not holding; try more "
                        f"n_control_bins, or a genuinely orthogonal variable")
        out[name] = rowa

    if verbose:
        _print_preview(out, detail=detail)
    return out


# The full warning text is written to be read ONCE, when you hit that situation
# for the first time. Printed 11 times in a single preview it is wallpaper. So
# each warning also gets a two-word tag, and the compact view prints the tags
# in a column and the full text once at the bottom.
_WARN_TAGS = (
    ("min_n",        "thin side"),
    ("side imbalance", "imbalanced"),
    ("eligibility keeps", "small sample"),
    ("control",       "control leak"),
    ("EMPTY",         "EMPTY SIDE"),
)


def _med_str(v) -> str:
    """' @9.47' -- the side's median split value. Blank for a categorical split,
    which has no median (and where the label already IS the value)."""
    if v is None or not np.isfinite(v):
        return ""
    a = abs(v)
    if a and (a < 1e-3 or a >= 1e5):
        return f" @{v:.2e}"
    return f" @{v:.4g}"


def _short(label, n=13) -> str:
    """Shorten a LABEL for the compact table. Labels are decoration; counts are
    data, so if something has to be clipped it is this."""
    s = str(label)
    return s if len(s) <= n else s[:n - 1] + "…"


def _warn_tag(w) -> str:
    for needle, tag in _WARN_TAGS:
        if needle in w:
            return tag
    return "warning"


def _print_preview(res, detail="summary"):
    """detail='summary' (default) -- one line per recipe, warnings as tags.
       detail='full'              -- every warning in full, the old behaviour."""
    full = (detail == "full")
    # NEVER truncate the side counts. A previous version clipped this column to
    # 34 chars, which turned "emitter=226" into "emitter=2" and silently hid the
    # third bin of every 3-way split -- a wrong number is far worse than a wide
    # table. The column is sized to the widest row instead, and the LABELS get
    # shortened if anything has to give.
    rows = OrderedDict()
    for name, r in res.items():
        if r["blocked"]:
            rows[name] = None
            continue
        # Each side as  label=N @median  -- the REALIZED MEDIAN of the split
        # variable on that side, in its own units. More useful than the
        # difference between the extremes: a median tells you where each bin
        # actually sits (and whether a bin is somewhere physically sensible),
        # which a single separation number cannot.
        bits = " | ".join(
            f"{_short(lab)}={s['n']}" + _med_str(s.get("median"))
            for lab, s in r["sides"].items())
        tags = []
        for w in r["warnings"]:
            t = _warn_tag(w)
            if t not in tags:
                tags.append(t)
        rows[name] = (r["n_eligible"], bits, tags)
    w_bits = max([len(v[1]) for v in rows.values() if v] + [22])

    print(f"{'recipe':<20} {'N':>5} {'sides:  label=N @median':<{w_bits}}  flags")
    print("-" * (30 + w_bits))
    n_warn = n_block = 0
    seen_tags = OrderedDict()
    for name, r in res.items():
        if r["blocked"]:
            n_block += 1
            print(f"{name:<20} {'--':>5} BLOCKED: {r['blocked']}")
            continue
        n_el, bits, tags = rows[name]
        for w in r["warnings"]:
            n_warn += 1
            seen_tags.setdefault(_warn_tag(w), w)
        print(f"{name:<20} {n_el:>5} {bits:<{w_bits}}  " + ", ".join(tags))
        if full:
            if r.get("separation") is not None:
                extra = (f"    separation in {r['recipe']['column']}: "
                         f"{r['separation']:.3g}")
                if r.get("control_offset") is not None:
                    extra += (f"   residual {r['recipe']['control']} offset: "
                              f"{r['control_offset']:.3g} (want ~0)")
                print(extra)
            for w in r["warnings"]:
                print(f"    ! {w}")

    print(f"\n{len(res)} recipes: {n_block} blocked, {n_warn} warning(s) "
          f"in {len(seen_tags)} kind(s).")
    if seen_tags and not full:
        print("\nwhat the flags mean (each explained once):")
        for t, w in seen_tags.items():
            body = " ".join(w.split())
            print(f"  {t:<14} {body[:150]}{'...' if len(body) > 150 else ''}")
        print("\n  preview(..., detail='full') for the per-recipe detail.")
    print("\nNothing here is a refusal -- min_n warns, it never skips.")


# =====================================================================
# 6.  INDEPENDENCE  (spec Sec 7.2b)
# =====================================================================

def independence(splitcat, recipes=None, *, base="CLEAN", graph=False,
                 verbose=True, **plot_kw) -> dict:
    """
    How many of these splits are actually different questions?

    Three things, in increasing order of how much they tell you:
      1. Spearman correlation matrix of the split VARIABLES over the eligible
         sample, grouped by family.
      2. Overlap matrix of the resulting SIDES -- for each pair, the fraction of
         galaxies assigned to the same side. THE directly interpretable number:
         two splits with 85% overlap are one split done twice. 0.5 is chance.
      3. M_eff, the effective number of independent tests, from the eigenvalue
         spectrum of the correlation matrix (Li & Ji 2005). One number, printable.

    Independence is a property of the VARIABLES, not of the sample size -- two
    variables uncorrelated in the population stay uncorrelated at N=500 or
    N=5000. Sample size buys power, not orthogonality. What makes family 1
    redundant is the galaxy scaling relations, i.e. astrophysics, and no amount
    of data changes it.

    The family taxonomy in the library is a PREDICTION; this is the measurement.
    Acceptance test (spec Sec 10 step 4): sfr_massctrl vs a hypothetical
    ssfr_massctrl comes back at overlap ~1.0, and `inclination` comes back ~0.5
    against everything.

    graph : True -> also draw both matrices as lower-triangle heatmaps
        (plot_independence). At 15x15 the printed tables are hard to scan and
        the figure is not; extra kwargs (figsize, cmap, save_fig, save_name)
        forward to it. The figure comes back as res["fig"].

    Operates on the SPLIT CATALOG (a Table), not a product -- no extraction
    needed, so this is a seconds-long call you can run before scoping a batch.
    """
    recs = resolve_recipes(recipes)
    usable, vals, sides = [], {}, {}
    for name, rec in recs.items():
        el = resolve_eligibility(splitcat, dict(rec, base=base))
        if el["blocked"] or el["n"] == 0:
            continue
        col = np.asarray(splitcat[rec["column"]])
        if not np.issubdtype(col.dtype, np.number):
            continue                    # categorical: no rank correlation
        v = np.full(len(splitcat), np.nan)
        v[el["mask"]] = col[el["mask"]].astype(float)
        vals[name] = v
        # binary side assignment at the eligible-sample median, for the overlap
        # matrix. Deliberately the SIMPLE assignment for every recipe (not each
        # recipe's own mode) -- the question here is "do these two variables
        # sort the sample the same way", and comparing a 4-bin trend to a binary
        # split on their own terms would answer a different one.
        thr = np.nanmedian(v[el["mask"]])
        s = np.full(len(splitcat), np.nan)
        s[el["mask"]] = (col[el["mask"]].astype(float) >= thr).astype(float)
        sides[name] = s
        usable.append(name)

    k = len(usable)
    rho = np.full((k, k), np.nan)
    ovl = np.full((k, k), np.nan)
    for i, a in enumerate(usable):
        for j, b in enumerate(usable):
            ok = np.isfinite(vals[a]) & np.isfinite(vals[b])
            if ok.sum() >= 10:
                rho[i, j] = _spearman(vals[a][ok], vals[b][ok])
            ok2 = np.isfinite(sides[a]) & np.isfinite(sides[b])
            if ok2.sum() >= 10:
                ovl[i, j] = float(np.mean(sides[a][ok2] == sides[b][ok2]))

    m_eff = _m_eff(rho)
    out = {"names": usable, "spearman": rho, "overlap": ovl, "m_eff": m_eff,
           "families": {n: recs[n]["family"] for n in usable}}

    if verbose:
        print(f"independence over {k} numeric recipes (base={base})\n")
        _print_matrix("Spearman rho (split VARIABLES)", usable, rho, out["families"])
        print()
        _print_matrix("overlap fraction (same SIDE; 0.5 = chance, 1.0 = the "
                      "same split twice)", usable, ovl, out["families"])
        print(f"\nM_eff out of {k} numeric recipes:")
        print(f"    Cheverud-Nyholt = {m_eff['cheverud_nyholt']:.2f}   <- quote this one")
        print(f"    Li & Ji (2005)  = {m_eff['li_ji']:.2f}   (conservative upper "
              f"bound; nearly flat for a small matrix -- see _m_eff)")
        if m_eff["li_ji"] - m_eff["cheverud_nyholt"] > 1.0:
            print("    the two disagree by >1 -- read the overlap matrix above "
                  "rather than either formula.")
        print("Spec Sec 7.2: ~7 effective axes from ~15 splits is enough for a "
              "real paper and not enough for a 20-parameter decomposition; "
              "frame it as the former.")
        print("Sec 7.2c: the real N~500 limit is INTERACTIONS, not independence "
              "-- a mass x inclination split is 4 cells of ~110. Name two-way "
              "questions as future work rather than reporting them as nulls.")
    if graph and len(usable) >= 2:
        out["fig"], out["axes"] = plot_independence(out, **plot_kw)
    return out


def plot_independence(res, *, figsize=(13, 6), cmap="RdBu_r", label_size=8,
                      grid=True, sort_by_family=True, save_fig=False,
                      save_name="split_independence.png", dpi=150):
    """
    Both independence matrices as lower-triangle heatmaps -- colour only, no
    numbers. At 15x15 the numbers are unreadable anyway, and the thing you
    actually want from this figure is "which cells are dark", not their values.

    LEFT  -- Spearman rho of the split VARIABLES. Blue/red = anti/correlated,
             white = independent. Scale is fixed to [-1, 1] so colour means the
             same thing every time you look at it.
    RIGHT -- overlap fraction of the resulting SIDES. Scale is fixed to [0, 1]
             with the SAME diverging colormap, so 0.5 (chance -- two splits that
             sort the sample independently) lands on white, and deep red is a
             pair of recipes that are one split done twice. This is the panel to
             read; it needs no assumption about linearity or about which M_eff
             formula to believe.

    The diagonal is masked -- it is 1.0 by construction and carries no
    information, and leaving it in wastes the top of the colour scale.

    Recipe names are prefixed with their family (f1 mass sequence, f2 external,
    f3 Lya-internal), so a block of dark cells inside one family is the expected
    redundancy and a dark cell ACROSS families is the interesting one.

    `res` is independence()'s return dict. Returns (fig, axes).
    """
    import matplotlib.pyplot as plt

    names = list(res["names"])
    n = len(names)
    if n < 2:
        raise ValueError("need at least 2 numeric recipes to plot.")
    fams = res.get("families", {})

    # Group family 1 / 2 / 3 together. The whole claim of spec Sec 7.2 is that
    # redundancy is a FAMILY property -- so with the families adjacent, that
    # claim is a block you can see, and a dark cell OUTSIDE a block is the
    # surprise. Scattered ordering hides both.
    order = list(range(n))
    if sort_by_family:
        order = sorted(order, key=lambda i: (fams.get(names[i]) or 9, names[i]))
    names = [names[i] for i in order]
    idx = np.ix_(order, order)
    ticks = [f"f{fams.get(x) or '-'} {x}" for x in names]

    mask = np.tril(np.ones((n, n), bool), k=-1)      # lower triangle, no diagonal
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [("Spearman $\\rho$  (split variables)", res["spearman"], -1.0, 1.0),
              ("overlap fraction  (same side)", res["overlap"], 0.0, 1.0)]

    # family boundaries, for the separator lines
    fam_seq = [fams.get(x) or 9 for x in names]
    breaks = [i for i in range(1, n) if fam_seq[i] != fam_seq[i - 1]]

    for ax, (title, mat, vmin, vmax) in zip(axes, panels):
        m = np.array(mat, dtype=float)[idx]
        m[~mask] = np.nan
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                       interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(ticks, rotation=90, fontsize=label_size)
        ax.set_yticklabels(ticks, fontsize=label_size)
        if grid:
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which="minor", color="0.85", linewidth=0.5)
            ax.tick_params(which="minor", length=0)
        for b in breaks:                       # family separators
            ax.axhline(b - 0.5, color="0.35", lw=1.1)
            ax.axvline(b - 0.5, color="0.35", lw=1.1)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=label_size)
        if vmin == 0.0:
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cb.ax.set_yticklabels(["0", "", "0.5\nchance", "", "1.0\nsame split"],
                                  fontsize=label_size)

    me = res.get("m_eff") or {}
    fig.suptitle(
        f"split independence  --  {n} numeric recipes"
        + (f",  M_eff {me.get('cheverud_nyholt', float('nan')):.1f} "
           f"(Li&Ji {me.get('li_ji', float('nan')):.1f})" if me else ""),
        fontsize=11)
    fig.tight_layout()
    if save_fig:
        fig.savefig(save_name, dpi=dpi, bbox_inches="tight")
        print(f"saved {save_name}")
    return fig, axes


def _spearman(a, b) -> float:
    """Rank correlation, ties averaged. Local so the module has no scipy
    dependency (scipy is not imported anywhere else in this package's core
    path) and so the tie handling is visible rather than assumed."""
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float(ra @ rb / den) if den > 0 else np.nan


def _rankdata(x) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return r


def _m_eff(rho) -> dict:
    """
    Effective number of independent tests from the correlation matrix's
    eigenvalue spectrum. TWO estimators, deliberately, because they disagree
    badly at the size of matrix this program actually has:

      Li & Ji (2005):  M_eff = sum_i [I(l_i>=1) + (l_i - floor(l_i))]
      Cheverud (2001) / Nyholt (2004):  M_eff = 1 + (M-1)(1 - Var(l)/M)

    Li & Ji is the more commonly cited one and it is the WRONG one to quote
    alone here. Its fractional term makes it nearly flat for a small matrix: two
    variables with rho=0.84 have eigenvalues (1.84, 0.16), and Li & Ji returns
    1.84 + 0.16 = 2.00 -- no reduction at all, right up until rho hits exactly
    1.0. It was built for the hundreds-of-SNPs case where eigenvalues run well
    above 2 and the floor() term does real work. Cheverud-Nyholt returns 1.65
    for that same pair, which is the behaviour you want.

    So: report Cheverud-Nyholt as the number, Li & Ji as the (conservative,
    almost always larger) upper bound, and treat any large gap between them as
    a signal to go read the overlap matrix instead -- which is the directly
    interpretable quantity and does not depend on either formula.

    NaNs (pairs with no common galaxies) are treated as zero correlation. That
    can only make M_eff look LARGER, i.e. claim more independence than there is,
    so a SMALL value here is trustworthy and a large one needs the overlap
    matrix to back it up.
    """
    r = np.nan_to_num(np.asarray(rho, float), nan=0.0)
    np.fill_diagonal(r, 1.0)
    r = 0.5 * (r + r.T)
    m = r.shape[0]
    lam = np.abs(np.linalg.eigvalsh(r))
    li_ji = float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))
    var = float(np.var(lam))
    cheverud = float(1.0 + (m - 1) * (1.0 - var / m)) if m > 1 else 1.0
    return {"cheverud_nyholt": cheverud, "li_ji": li_ji, "n_variables": m}


def _print_matrix(title, names, mat, families):
    print(title)
    w = max(len(n) for n in names) + 1
    print(" " * (w + 3) + " ".join(f"{i:>5d}" for i in range(len(names))))
    for i, n in enumerate(names):
        fam = families.get(n) or "-"
        row = " ".join("   . " if not np.isfinite(v) else f"{v:5.2f}"
                       for v in mat[i])
        print(f"{n:<{w}} f{fam} {i:>2d} {row}")


# =====================================================================
# 7.  THE HEADLESS FIT
# =====================================================================
# analysis.plot_flux_profile_fit is the pipeline's fit entry point, but it
# builds a figure. A 30-side batch does not want 30 figures, matplotlib state,
# or a display. This is EXACTLY its fit block, extracted verbatim -- same y,
# same sigma convention, same fine grid, same PSF construction, same fitting.py
# calls in the same order -- so the two return numerically identical
# fit_results. Do not "improve" it independently of that function; if one
# changes, change both.

FIT_DEFAULTS = dict(model="expcore", method="psf", fit_skip_inner=1,
                    gamma_fixed=0.8, r_c_fixed=400.0,
                    psf_fwhm_arcsec=1.3, psf_beta=3.0)


def fit_profile(boot, stacks=None, *, r_edges=None, model="expcore", method="psf",
                fit_skip_inner=1, gamma_fixed=0.8, r_c_fixed=400.0,
                psf_r=None, psf_vals=None, psf_fwhm_arcsec=1.3, psf_beta=3.0,
                z_median=None, r_fine=None, p0=None, verbose=False) -> dict:
    """
    Fit the radial flux profile with NO figure. The headless twin of
    analysis.plot_flux_profile_fit; every argument means exactly what it means
    there, and the returned fit_result carries the same extra keys (method,
    r_edges, r_mid, r_fine, y, y_lo, y_hi, sigma, and for method='psf' the R
    matrix, psf_r, psf_vals) so it feeds straight into
    fitting.find_core_halo_boundary and analysis.bootstrap_fit_profile.

    Defaults are the pipeline's current headline convention: expcore, PSF
    forward-modelled, gamma pinned at 0.8 and r_c pinned at 400 kpc (see
    fitting.fit_psf_aware_expcore for the 2026-07-21 investigation behind both).
    """
    if "total_flux_fid" not in boot:
        raise KeyError("boot has no total_flux_fid -- run Stage 3 with "
                       "compute_side_ratio=True (the default).")
    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                                           (stacks or {}).get("r_edges")), float)

    y = np.asarray(boot["total_flux_fid"], float)
    y_lo = np.asarray(boot["total_flux_lo"], float)
    y_hi = np.asarray(boot["total_flux_hi"], float)
    sigma = ((y_hi - y) + (y - y_lo)) / 2.0       # symmetric 1-sigma from 16-84

    r_fine_arr = (np.asarray(r_fine, float) if r_fine is not None
                  else fitting.default_fine_grid(radial_bins))
    r_mid = 0.5 * (radial_bins[:-1] + radial_bins[1:])

    if method == "psf":
        if psf_r is not None and psf_vals is not None:
            psf_r_use = np.asarray(psf_r, float)
            psf_vals_use = np.asarray(psf_vals, float)
        else:
            z_med = z_median if z_median is not None else (stacks or {}).get("z_median")
            fwhm_kpc = _psf_fwhm_arcsec_to_kpc(psf_fwhm_arcsec, z_med,
                                               sample_label="split side")
            psf_r_use = np.linspace(0.0, 20.0 * fwhm_kpc, 400)
            psf_vals_use = fitting.moffat_1d(psf_r_use, fwhm=fwhm_kpc, beta=psf_beta)
        R = fitting.ring_convolution_matrix(r_fine_arr, radial_bins,
                                            psf_r_use, psf_vals_use)
        if model == "expcore":
            fr = fitting.fit_psf_aware_expcore(r_mid, y, sigma, R, r_fine_arr,
                                               radial_bins, gamma_fixed=gamma_fixed,
                                               r_c_fixed=r_c_fixed, p0=p0,
                                               verbose=verbose)
        else:
            fr = fitting.fit_psf_aware(r_mid, y, sigma, R, r_fine_arr, radial_bins,
                                       p0=p0, verbose=verbose)
        fr["R"], fr["psf_r"], fr["psf_vals"] = R, psf_r_use, psf_vals_use
    elif method == "naive":
        if model == "expcore":
            fr = fitting.fit_naive_expcore(r_mid, radial_bins, r_fine_arr, y, sigma,
                                           fit_skip_inner=fit_skip_inner,
                                           gamma_fixed=gamma_fixed,
                                           r_c_fixed=r_c_fixed, p0=p0, verbose=verbose)
        else:
            fr = fitting.fit_naive(r_mid, radial_bins, r_fine_arr, y, sigma,
                                   fit_skip_inner=fit_skip_inner, p0=p0,
                                   verbose=verbose)
    else:
        raise ValueError(f"method must be 'psf' or 'naive' (got {method!r})")

    fr.update(method=method, r_edges=radial_bins, r_mid=r_mid, r_fine=r_fine_arr,
              y=y, y_lo=y_lo, y_hi=y_hi, sigma=sigma)
    return fr


# =====================================================================
# 8.  POWER FLAGS  (spec Sec 7.3)
# =====================================================================
# Full-stack outer-3-bin bootstrap S/N is 6 / 5 / 2.8 at N~452. A two-way split
# (N~226) gives 4.2 / 3.5 / 2.0; a three-way (N~151) gives 3.5 / 2.9 / 1.6.
# NO SPLIT SUPPORTS A >2 SIGMA OUTERMOST-BIN CLAIM WITH CURRENT DATA.
#
# The runner does not refuse to compute two-halo quantities -- it computes them
# and marks them, so the flag TRAVELS WITH THE NUMBER into the table instead of
# living in someone's head.

# PUBLIC and mutable on purpose -- this is a policy call, not a fact about the
# code. Every quantity named here is flagged power="insufficient" on every side,
# always, because no split supports a >2sigma outermost-bin claim with current
# data. When the ~2x data lands, drop names from this set rather than editing
# _power_flags:
#     splits.TWO_HALO_QUANTITIES.discard("twohalo_lum")
TWO_HALO_QUANTITIES = {
    "twohalo_lum", "twohalo_v", "twohalo_over_onehalo", "twohalo_over_total",
    "r_c", "v_diff",
}


def _power_flags(n_gal, n_sides, min_n) -> dict:
    """{quantity: 'ok' | 'marginal' | 'insufficient'} for one side."""
    if n_gal >= min_n:
        inner = "ok"
    elif n_gal >= 0.5 * min_n:
        inner = "marginal"
    else:
        inner = "insufficient"
    flags = {}
    for q in QUANTITIES:
        if q in TWO_HALO_QUANTITIES:
            flags[q] = "insufficient"      # without exception, per Sec 7.3
        elif n_sides > 2:
            # trend / multi-bin: still fine for core and one-halo quantities at
            # ~110-150 per bin, but nothing past the one-halo term.
            flags[q] = inner if inner != "ok" else "ok"
        else:
            flags[q] = inner
    return flags


# =====================================================================
# 9.  QUANTITY REGISTRY
# =====================================================================
# name -> (which result dict it lives in, its _fid key, where its per-draw array
# is). The per-draw array is what makes an honest DIFFERENCE interval possible:
# two sides' bootstraps are independent (different galaxies, same seed), so
# element-wise differencing of the two draw arrays is a valid sample from the
# difference distribution -- much better than adding half-widths in quadrature,
# which assumes symmetry these distributions are known not to have.

QUANTITIES = OrderedDict([
    ("h1",                   ("boot_fit",    "h1_fid",                   ("draws", "h1"))),
    ("r_c",                  ("boot_fit",    "r_c_fid",                  ("draws", "r_c"))),
    ("crossover_radius",     ("boot_fit",    "crossover_radius_fid",     ("draws", "crossover_radius"))),
    ("boundary_radius",      ("boundary_info", "boundary_radius",        None)),
    ("core_v",               ("vel",         "core_v_fid",               None)),
    ("twohalo_v",            ("vel",         "twohalo_v_fid",            "twohalo_v_all")),
    ("v_diff",               ("vel",         "diff_fid",                 "diff_all")),
    ("core_lum",             ("core_lum",    "core_lum_fid",             "core_lum_all")),
    ("onehalo_lum",          ("onehalo_lum", "onehalo_lum_fid",          "onehalo_lum_all")),
    ("twohalo_lum",          ("twohalo_lum", "twohalo_lum_fid",          "twohalo_lum_all")),
    ("onehalo_over_core",    ("ratios",      "onehalo_over_core_fid",    "onehalo_over_core_all")),
    ("twohalo_over_onehalo", ("ratios",      "twohalo_over_onehalo_fid", "twohalo_over_onehalo_all")),
    ("twohalo_over_total",   ("ratios",      "twohalo_over_total_fid",   "twohalo_over_total_all")),
])


# FALLBACK for the fit parameters. h1 and r_c live in boot_fit, which only
# exists at level="standard"/"full" -- so at level="quick" the single most
# important number in the whole program (h1, the halo scale length) was missing
# from the comparison table entirely, while sitting in plain view in the printed
# fit section. It IS available at quick level, just with curve_fit's
# covariance-based error instead of the refit-per-draw one. Report it, and mark
# the error as the weaker kind rather than showing nothing.
_FIT_FALLBACK = {"h1": "h1", "r_c": "r_c"}


def _quantity(side, qname):
    """(fid, lo, hi, draws) for one quantity of one side; all-None if absent."""
    where, fid_key, draw_key = QUANTITIES[qname]
    d = side.get(where)
    if not isinstance(d, dict) and qname in _FIT_FALLBACK:
        fr = side.get("fit_result")
        p = _FIT_FALLBACK[qname]
        if isinstance(fr, dict) and fr.get(p) is not None:
            v, e = fr.get(p), fr.get(p + "_err")
            v = float(v)
            if e is None or not np.isfinite(e) or e == 0:
                return (v, None, None, None)
            return (v, v - float(e), v + float(e), None)
    if not isinstance(d, dict):
        return (None, None, None, None)
    fid = d.get(fid_key)
    stem = fid_key[:-4] if fid_key.endswith("_fid") else fid_key
    lo, hi = d.get(stem + "_lo"), d.get(stem + "_hi")
    draws = None
    if isinstance(draw_key, tuple):
        draws = (d.get(draw_key[0]) or {}).get(draw_key[1])
    elif draw_key:
        draws = d.get(draw_key)
    f = lambda v: (float(v) if v is not None and np.isfinite(np.asarray(v, float)).all()
                   else None)
    return (f(fid), f(lo), f(hi),
            None if draws is None else np.asarray(draws, float))


# =====================================================================
# 10.  CACHE  (spec Sec 6)
# =====================================================================
# ONE FILE PER SIDE, not per batch -- an interrupted 6-hour batch resumes.
#
# The cache key hashes FOUR things, and item 2 is the one that must never be
# dropped: config.extract_signature(). The SMHM default changed 2026-07-27 and
# the PSF area convention changed 2026-07-22. A results cache that does not
# invalidate on the extraction signature will silently serve numbers from two
# different pipelines IN THE SAME TABLE, and the failure is invisible because
# both numbers are individually plausible. Highest-severity item in the spec.

def _measure_hash(config, level, seed, fit_kw, nboot_fit) -> str:
    blob = json.dumps({
        "level": level, "seed": seed, "nboot_fit": nboot_fit,
        "nboot": int(config.n_bootstrap),
        "stack_method": getattr(config, "measure_stack_method", "biweight"),
        "centroid_method": getattr(config, "centroid_method", None),
        "line_window": list(config.line_window),
        "cont_bounds": [list(np.atleast_1d(b)) for b in config.cont_bounds],
        "cont_method": getattr(config, "cont_method", None),
        "cont_order": getattr(config, "cont_order", None),
        "fit": {k: fit_kw.get(k) for k in sorted(fit_kw)},
    }, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _splitcat_hash(splitcat) -> str:
    """CONTENT hash of the split catalog, not just its provenance metadata.

    Metadata alone is not enough and the failure is silent: re-measure the core
    product, rebuild the split catalog, and DERIVED / NGAL / IDCOL all come back
    identical while every CORE_* column has changed. A metadata-keyed cache would
    then serve pre-remeasurement core splits under the new catalog. Hashing the
    actual bytes costs milliseconds on a ~500-row table.
    """
    if splitcat is None:
        return "nosplitcat"
    meta = getattr(splitcat, "meta", {}) or {}
    h = hashlib.sha1()
    h.update(json.dumps({"DERIVED": str(meta.get("DERIVED", "")),
                         "IDCOL": str(meta.get("IDCOL", "ID")),
                         "ngal": len(splitcat)}, sort_keys=True).encode())
    for c in sorted(splitcat.colnames):
        h.update(c.encode())
        col = np.asarray(splitcat[c])
        if col.dtype == object or col.dtype.kind in "SU":
            h.update(col.astype("U").tobytes())
        else:
            h.update(np.ascontiguousarray(col).tobytes())
    return h.hexdigest()[:12]


def _product_hash(product) -> str:
    """WHICH GALAXIES this run was on.

    config.extract_signature() covers every Stage-1 knob, but NOT which product
    you handed the runner. With field='BOTH' in the config, `prod_combined` and
    `products['COSMOS']` produce the SAME extraction signature -- so without
    this, running a recipe on the combined product and then on one field alone
    would serve the combined numbers for the single field, silently, and both
    would look plausible. That is the exact failure mode spec Sec 6 calls the
    highest-severity item, so the galaxy identity has to be in the key too.

    Hashes the ID list (order-independent), ngal, and the radial bin edges --
    everything that determines which spectra get stacked, without hashing the
    spectra themselves (which would cost real time on every call).
    """
    cat = product.catalog
    idc = next((c for c in ("ID", "id") if c in cat.colnames), cat.colnames[0])
    ids = np.sort(np.asarray(cat[idc]).astype("U"))
    h = hashlib.sha1()
    h.update(f"{idc}|{product.ngal}|".encode())
    h.update(ids.tobytes())
    h.update(np.ascontiguousarray(np.asarray(product.r_edges, float)).tobytes())
    return h.hexdigest()[:12]


def _cache_key(config, recipe, level, seed, fit_kw, nboot_fit, splitcat,
               product=None) -> dict:
    return {
        "recipe_hash": recipe_hash(recipe),
        "extract_signature": config.extract_signature(),
        "measure_hash": _measure_hash(config, level, seed, fit_kw, nboot_fit),
        "splitcat_hash": _splitcat_hash(splitcat),
        "product_hash": "noproduct" if product is None else _product_hash(product),
    }


# Where split RESULTS go. Set this once in a notebook and every function below
# follows it:
#     splits.SPLITS_DIR = "./outputs_measurements"
#
# It is a separate knob from config.output_dir ON PURPOSE. config.output_dir
# also determines resolved_cache_dir() (<output_dir>/cache), which is the
# per-galaxy STAGE-1 cache -- so repointing output_dir to move your split
# results would orphan a 20-hour extraction cache at the same time. This is the
# one thing you should never have to think about, so it is its own variable.
SPLITS_DIR = None


def cache_dir(config, splits_dir=None) -> str:
    """
    Where this config's split results live:  <base>/splits/<extract_signature>/

    `base` is resolved in this order, first hit wins:
      1. the `splits_dir` argument (one-off override)
      2. `config.splits_dir`, if your config carries one
      3. the module-level `splits.SPLITS_DIR`, if you set it
      4. `config.output_dir`  (the default, so nothing breaks if you set nothing)

    The extraction signature is a DIRECTORY level, not part of the filename, so
    a re-extraction's results sit beside the old ones rather than on top of them
    -- and you can delete one whole run by removing one folder.
    """
    base = (splits_dir
            or getattr(config, "splits_dir", None)
            or SPLITS_DIR
            or config.output_dir)
    return os.path.join(base, "splits", config.extract_signature())


def _safe(s) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("_") or "x"


def _side_paths(config, recipe, label, key, splits_dir=None):
    d = cache_dir(config, splits_dir)
    stem = (f"{_safe(recipe['name'])}__{_safe(label)}__"
            f"{key['recipe_hash']}_{key['measure_hash']}")
    return os.path.join(d, stem + ".json"), os.path.join(d, stem + ".npz")


def _encode(obj, arrays, prefix=""):
    """Split a nested result into JSON-able scalars and an array sidecar. Arrays
    become {'__npz__': key} placeholders; the .json therefore stays readable and
    diffable, which is the point -- every hash input is meant to be legible in
    plain text without loading numpy."""
    if isinstance(obj, dict):
        return {str(k): _encode(v, arrays, f"{prefix}{k}/") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v, arrays, f"{prefix}{i}/") for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        key = prefix.rstrip("/") or f"a{len(arrays)}"
        arrays[key] = obj
        return {"__npz__": key}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)                      # last resort: never fail a 6-hour batch


def _decode(obj, npz):
    if isinstance(obj, dict):
        if "__npz__" in obj and len(obj) == 1:
            k = obj["__npz__"]
            return npz[k] if npz is not None and k in npz else None
        return {k: _decode(v, npz) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, npz) for v in obj]
    return obj


# fit_result['R'] is (n_fine x n_bins) and is a pure function of the PSF and the
# radial grid -- identical for every side of every recipe. Storing it 30 times
# is ~30x waste for something fit_profile rebuilds in milliseconds.
_DROP_FROM_CACHE = (("fit_result", "R"), ("fit_result", "psf_r"),
                    ("fit_result", "psf_vals"), ("boot", "stack_error"))


def _strip(side):
    out = dict(side)
    for parent, child in _DROP_FROM_CACHE:
        if isinstance(out.get(parent), dict) and child in out[parent]:
            out[parent] = {k: v for k, v in out[parent].items() if k != child}
    return out


def _save_side(config, recipe, label, side, key, *, side_index=0,
               splits_dir=None, verbose=True):
    d = cache_dir(config, splits_dir)
    os.makedirs(d, exist_ok=True)
    jpath, npath = _side_paths(config, recipe, label, key, splits_dir)
    arrays = {}
    payload = {"recipe": dict(recipe), "label": label, "key": key,
               "side_index": int(side_index),
               "side": _encode(_strip(side), arrays)}
    if arrays:
        np.savez_compressed(npath, **arrays)
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    if verbose:
        print(f"    cached -> {os.path.basename(jpath)}")
    return jpath


def _load_side(config, recipe, label, key, splits_dir=None):
    jpath, npath = _side_paths(config, recipe, label, key, splits_dir)
    if not os.path.exists(jpath):
        return None
    with open(jpath) as fh:
        payload = json.load(fh)
    if payload.get("key") != key:
        return None                      # belt and braces: the stem encodes it
    npz = np.load(npath, allow_pickle=False) if os.path.exists(npath) else None
    return _decode(payload["side"], npz)


def clear_cache(config, recipes=None, *, splits_dir=None, verbose=True) -> int:
    """Delete cached side files for these recipes (default: all of them) under
    the CURRENT extraction signature. Returns the number of files removed.
    force=True on run() is usually what you want instead -- this is for when the
    cache itself is suspect."""
    d = cache_dir(config, splits_dir)
    if not os.path.isdir(d):
        return 0
    names = None if recipes is None else set(resolve_recipes(recipes))
    n = 0
    for f in sorted(os.listdir(d)):
        stem = f.split("__")[0]
        if names is not None and stem not in names:
            continue
        os.remove(os.path.join(d, f))
        n += 1
    if verbose:
        print(f"removed {n} cached file(s) from {d}")
    return n


# ---------------------------------------------------------------------
# CACHE DIAGNOSTICS -- "why is it recomputing?"
# ---------------------------------------------------------------------
# THE problem with a hash-keyed cache: a miss and a first run look identical.
# You change one fit_kw entry, or pass seed=None to load() having run with
# seed=10, and the cache silently recomputes (or silently returns nothing) with
# no indication that anything was different. That is not a hypothetical -- it is
# exactly the seed=None / fit_kw=None mismatch that made splits.load() come back
# empty after a successful 40-minute run.
#
# cache_status() answers "would this call hit?" without running anything, and
# when it misses it says WHICH of the five key components differs from the file
# on disk. Run it before a batch, not after.

_KEY_MEANING = {
    "recipe_hash":       "the recipe dict itself (column/mode/bounds/require/...)",
    "extract_signature": "Stage-1 extraction settings (bins, mask, background, field)",
    "measure_hash":      "level / seed / nboot_fit / n_bootstrap / line window / "
                         "continuum / every fit_kw entry",
    "splitcat_hash":     "the CONTENT of the split catalog table",
    "product_hash":      "which galaxies (sorted ID list, ngal, r_edges)",
}


def cache_status(config, recipes=None, *, product=None, splitcat=None,
                 level="standard", seed=None, nboot_fit=200, fit_kw=None,
                 splits_dir=None, verbose=True) -> "OrderedDict[str, dict]":
    """
    Would run()/load() HIT the cache with these arguments? Costs milliseconds.

    Pass EXACTLY the arguments you intend to give run() -- including `seed` and
    `fit_kw`, which are the two people habitually leave at their defaults when
    re-loading. Anything you leave out is resolved the same way run() resolves
    it, so `cache_status(cfg, BATCH, product=prod, splitcat=sc, seed=10,
    fit_kw=FIT_KW)` and `run(...)` with the same arguments agree by construction.

    For each recipe it reports, per side file it can find on disk:

        HIT      a file exists whose full 5-part key matches -> run() will reuse
        MISS     no file at all for this recipe/side stem
        STALE    a file exists for this recipe under a DIFFERENT key, and the
                 named components are what differ. This is the useful case: it
                 tells you whether you are about to recompute because you
                 genuinely changed the physics (extract_signature, fit_kw) or
                 merely because you typed seed=None.

    Returns OrderedDict{recipe_name: {'dir', 'sides': {label: {...}}}}.
    """
    recs = resolve_recipes(recipes)
    seed = int(seed if seed is not None else getattr(config, "seed", 1))
    fit_kw = dict(FIT_DEFAULTS, **(fit_kw or {}))
    d = cache_dir(config, splits_dir)
    out = OrderedDict()

    # Every key file that exists on disk, so we can diff against near-misses.
    on_disk = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, f)) as fh:
                    payload = json.load(fh)
            except Exception:                                     # noqa: BLE001
                continue
            on_disk.append((f, payload.get("key") or {},
                            (payload.get("side") or {}).get("label"),
                            f.split("__")[0]))

    if verbose:
        print(f"cache dir: {d}")
        print(f"  {'(does not exist yet)' if not os.path.isdir(d) else str(len(on_disk)) + ' cached side file(s) present'}")
        print()

    for name, rec in recs.items():
        key = _cache_key(config, rec, level, seed, fit_kw, nboot_fit, splitcat,
                         product=product)
        mine = [(f, k, lab) for f, k, lab, stem in on_disk if stem == _safe(name)]
        sides = OrderedDict()
        for f, k, lab in mine:
            diff = [c for c in _KEY_MEANING if k.get(c) != key.get(c)]
            sides[lab or f] = {
                "file": f,
                "status": "HIT" if not diff else "STALE",
                "differs": diff,
                "on_disk": {c: k.get(c) for c in diff},
                "wanted": {c: key.get(c) for c in diff},
            }
        if not mine:
            sides["(none)"] = {"file": None, "status": "MISS", "differs": [],
                               "on_disk": {}, "wanted": {}}
        out[name] = {"dir": d, "key": key, "sides": sides}

        if verbose:
            worst = ("MISS" if all(s["status"] == "MISS" for s in sides.values())
                     else ("HIT" if all(s["status"] == "HIT" for s in sides.values())
                           else "PARTIAL/STALE"))
            print(f"{name:<22} {worst}")
            for lab, s in sides.items():
                if s["status"] == "HIT":
                    print(f"    {lab:<20} HIT   (will be reused)")
                elif s["status"] == "MISS":
                    print(f"    {lab:<20} MISS  (never computed under this key)")
                else:
                    print(f"    {lab:<20} STALE -- differs in: "
                          f"{', '.join(s['differs'])}")
                    for c in s["differs"]:
                        print(f"        {c}: on disk {s['on_disk'][c]!r} "
                              f"!= wanted {s['wanted'][c]!r}")
                        print(f"          ^ {_KEY_MEANING[c]}")
    if verbose:
        print("\nSTALE on measure_hash alone is almost always an argument "
              "mismatch (seed / fit_kw / nboot_fit / level), not a real change.\n"
              "STALE on extract_signature or splitcat_hash means the inputs "
              "really did move -- recomputing is correct.")
    return out


def why_miss(config, recipes=None, **kw):
    """Alias for cache_status() -- the name you will actually think of at 2am."""
    return cache_status(config, recipes, **kw)


# =====================================================================
# 11.  LAYER C -- THE RUNNER  (spec Sec 5.1)
# =====================================================================

def _vlevel(verbose) -> int:
    """verbose= accepts True/False or 0/1/2.

      0 / False : silent
      1 / True  : ONE LINE PER SIDE (default). No bootstrap progress bars, no
                  per-call headers -- a 14-recipe batch is ~30 lines, not ~2000.
      2         : everything, including every bootstrap bar. Use when one side
                  is behaving oddly and you want to watch it.
    """
    if verbose is True:
        return 1
    if verbose is False or verbose is None:
        return 0
    return int(verbose)


def _measure_side(config, sub, recipe, label, *, level, seed, min_n, n_sides,
                  fallback_fit=None, fit_kw=None, nboot_fit=200, verbose=True):
    """
    The per-side chain. Every call here is one this pipeline already makes; the
    ONLY thing this function contributes is the order and the bookkeeping.

        build_stacks(keep_cube=True)
          -> measure_all_bins(seed=<shared>)
          -> fit_profile
          -> find_core_halo_boundary (with the full-stack fit as fallback)
          -> measure_core_twohalo_velocity / _twohalo_luminosity
          -> one-halo + (level='full') PSF-corrected core, leak-aware
          -> measure_three_zone_ratios
          -> bootstrap_fit_profile (level >= 'standard')
          -> power flags

    level:
        'quick'    stack + bootstrap + point-estimate fit + boundary + velocity
                   + zone luminosities. Minutes. No h1 error bars.
        'standard' (default) + bootstrap_fit_profile at nboot_fit draws -- the
                   refit-per-draw h1/r_c/crossover errors this pipeline actually
                   trusts (covariance errors are known unreliable for r_c/gamma).
        'full'     + measure_psf_corrected_core_onehalo_luminosity, a second
                   per-galaxy-rescaled restack + bootstrap pass, at the config's
                   full nboot. This is the expensive one.

    Returns a dict whose keys are EXACTLY the kwargs
    analysis.describe_subsample_properties and
    formatting.make_subsample_latex_table already accept, plus bookkeeping. No
    adapter layer, no third schema -- that is deliberate and worth preserving.
    """
    fit_kw = dict(FIT_DEFAULTS, **(fit_kw or {}))
    v = _vlevel(verbose)
    loud = (v >= 2)          # 2 = show every bootstrap bar and sub-call header
    warns = []
    if sub.ngal < min_n:
        warns.append(f"N={sub.ngal} < min_n={min_n} (ran anyway; see power flags)")
    if v:
        import time as _t
        _t0 = _t.time()
        print(f"  {recipe['name']}/{label:<18} N={sub.ngal:<5} {level:<9}",
              end="", flush=True)

    stacks = stack.build_stacks(config, sub, keep_cube=True)
    boot = measure.measure_all_bins(config, stacks, seed=seed, verbose=loud)

    fit_result = fit_profile(boot, stacks, verbose=False, **fit_kw)
    if not fit_result.get("success", True):
        warns.append("profile fit did not converge on this side")

    boundary_info = fitting.find_core_halo_boundary(
        fit_result, fallback_fit_result=fallback_fit)
    b = boundary_info.get("boundary_radius")
    if boundary_info.get("boundary_from_own_fit") is False:
        warns.append(f"boundary came from the FULL-STACK fallback fit, not this "
                     f"side's own fit (source={boundary_info.get('source')}) -- "
                     f"flag it in the table")
    # A boundary is unusable in two ways, and only ONE of them used to be caught
    # here. The first is the obvious one: no crossing at all -> None/nan. The
    # second is a DEGENERATE crossing -- the fit returns a finite crossover that
    # sits at or inside the core bin's own outer edge (e.g. 0.09 kpc). That
    # happens when the exp-core term never actually dominates, which is exactly
    # what a net-absorption or very-low-N stack produces: the amplitude collapses
    # toward zero, the power law dominates at every radius, and the "crossing"
    # the solver reports is a numerical artifact, not a radius.
    #
    # 0.09 kpc is finite, so it sailed past the old guard and blew up two calls
    # later inside measure_onehalo_luminosity -- killing an entire batch. The
    # test below is the SAME condition measure_onehalo_luminosity applies
    # internally (bins strictly outside the core bin whose midpoint is <= b), so
    # we fail the side here, gracefully, instead of there, fatally.
    _bad_boundary = (b is None or not np.isfinite(b))
    if not _bad_boundary:
        _re = np.asarray(fit_result.get("r_edges", boot["r_edges"]), dtype=float)
        _rmid = 0.5 * (_re[:-1] + _re[1:])
        if not np.any((np.arange(len(_rmid)) > 0) & (_rmid <= float(b))):
            _bad_boundary = True
            warns.append(
                f"DEGENERATE boundary: fitted crossover r_x={float(b):.4g} kpc "
                f"falls at or inside the core bin's outer edge "
                f"(r_edges[1]={_re[1]:.4g}) -- there is no separately-measurable "
                f"one-halo zone. Usually means the exp-core term never dominates "
                f"(net-absorption or very-low-N stack), i.e. the expcore model "
                f"does not describe this side. Zone quantities skipped.")
    if _bad_boundary:
        if "DEGENERATE boundary" not in "".join(warns):
            warns.append("no core/two-halo boundary -- every zone quantity is skipped")
        if v:
            print(f"  {_t.time()-_t0:5.0f}s   NO BOUNDARY -- zone quantities skipped")
        return _side_dict(label, sub, boot, stacks, fit_result, boundary_info,
                          None, None, None, None, None, warns,
                          _power_flags(sub.ngal, n_sides, min_n), level)

    vel = measure.measure_core_twohalo_velocity(boot, b)
    twohalo_lum = measure.measure_twohalo_luminosity(boot, b)

    if level == "full":
        core_lum, onehalo_lum = measure.measure_psf_corrected_core_onehalo_luminosity(
            config, sub, stacks, boot, b,
            psf_fwhm_arcsec=fit_kw["psf_fwhm_arcsec"], psf_beta=fit_kw["psf_beta"],
            seed=seed, verbose=loud)
        # leak-aware: the flux the PSF correction implies leaked OUT of the core
        # is removed from the one-halo zone rather than double-counted there
        # (measure.py's 2026-07-22 note).
    else:
        core_lum = None
        onehalo_lum = measure.measure_onehalo_luminosity(boot, b)
        warns.append(f"level={level!r}: core luminosity not computed and the "
                     f"one-halo zone is NOT leak-corrected; re-run at "
                     f"level='full' before quoting a core/one-halo ratio")

    ratios = (measure.measure_three_zone_ratios(core_lum, onehalo_lum, twohalo_lum)
              if core_lum is not None else None)

    boot_fit = None
    if level in ("standard", "full"):
        boot_fit = analysis.bootstrap_fit_profile(
            boot, fit_result["r_edges"], fit_result["r_fine"],
            model=fit_kw["model"], method=fit_kw["method"],
            R=fit_result.get("R"), gamma_fixed=fit_kw["gamma_fixed"],
            r_c_fixed=fit_kw["r_c_fixed"], fit_skip_inner=fit_kw["fit_skip_inner"],
            nboot=nboot_fit, seed=seed, verbose=loud)
        nf = boot_fit.get("n_failed") or 0
        nu = (boot_fit.get("meta") or {}).get("nboot_used") or 1
        if nf > 0.1 * nu:
            warns.append(f"{nf}/{nu} refit draws failed to converge -- the "
                         f"16/84 spread on h1/r_c is built on the survivors and "
                         f"is optimistic")

    if v:
        h1 = (fit_result or {}).get("h1")
        print(f"  {_t.time()-_t0:5.0f}s"
              + (f"   h1={h1:6.2f}" if h1 is not None else "")
              + (f"   r_x={b:6.1f}" if b is not None and np.isfinite(b) else "")
              + (f"   [{len(warns)} note{'s' if len(warns) != 1 else ''}]" if warns else ""))
    return _side_dict(label, sub, boot, stacks, fit_result, boundary_info, vel,
                      core_lum, onehalo_lum, twohalo_lum, ratios, warns,
                      _power_flags(sub.ngal, n_sides, min_n), level, boot_fit)


def _side_dict(label, sub, boot, stacks, fit_result, boundary_info, vel,
               core_lum, onehalo_lum, twohalo_lum, ratios, warns, power, level,
               boot_fit=None):
    id_col = "ID" if "ID" in sub.catalog.colnames else sub.catalog.colnames[0]
    return {
        # --- describe_subsample_properties / make_subsample_latex_table kwargs
        "label": label, "n_gal": int(sub.ngal),
        "fit_result": fit_result, "boundary_info": boundary_info, "vel": vel,
        "core_lum": core_lum, "onehalo_lum": onehalo_lum,
        "twohalo_lum": twohalo_lum, "ratios": ratios, "boot_fit": boot_fit,
        # --- bookkeeping
        "ids": np.asarray(sub.catalog[id_col]).astype(str),
        "z_median": stacks.get("z_median"), "mass_median": stacks.get("mass_median"),
        "power": power, "warnings": warns, "level": level,
        "boot": boot,
    }


def run(config, product, recipes=None, *, splitcat=None, level="standard",
        force=False, min_n=None, seed=None, nboot_fit=200, fit_kw=None,
        cache=True, splits_dir=None, base_sample=None, verbose=True) -> "OrderedDict[str, dict]":
    """
    Run one recipe, several, or the whole library. Resumable.

    `recipes` takes a name, a recipe dict, or any list/dict of either -- so
    "all of them" and "just this one" are the same call:

        splits.run(cfg, prod, "mass_half", splitcat=sc)
        splits.run(cfg, prod, ["mass_half", "inclination"], splitcat=sc)
        splits.run(cfg, prod, splitcat=sc)                    # everything
        splits.run(cfg, prod, dict(LIBRARY["mass_half"], mode="trend",
                                   n_bins=3, name="mass_trend3"), splitcat=sc)

    Parameters
    ----------
    splitcat : the multicat.build_split_catalog table. Attached to the product
        here (attach_split_catalog) so the caller does not have to remember to.
        Pass None if the product's catalog already carries the columns.
    level : 'quick' | 'standard' (default) | 'full'. See _measure_side.
    force : recompute even if a cached side exists. Default False, which is what
        makes an interrupted batch resume rather than restart.
    min_n : global override of every recipe's own min_n. WARNS, never skips.
    seed : the galaxy-resampling seed, shared across EVERY recipe and side (spec
        Sec 5.4). This is not just reproducibility: sharing the draws makes the
        splits correlated in a KNOWN way, which is what makes the overlap
        numbers in independence() interpretable as a correlation structure
        rather than noise. Defaults to config.seed, else 1.
    nboot_fit : draws for the refit-per-draw bootstrap_fit_profile. 200 is a
        deliberate reduction from the config's full nboot -- each draw is a
        separate nonlinear fit, and at 2000 a 15-recipe batch is dominated by it.
        Raise it for the sides that end up in the paper.
    fit_kw : overrides for fit_profile (model, method, gamma_fixed, r_c_fixed,
        psf_fwhm_arcsec, ...). Part of the cache key, so changing one
        invalidates cleanly instead of mixing conventions in one table.

    Returns OrderedDict{recipe_name: result}, where a result is
        {'recipe', 'sides' {label: side}, 'compare', 'provenance', 'warnings'}.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS} (got {level!r}).")
    recs = resolve_recipes(recipes)
    if splitcat is not None:
        product = multicat.attach_split_catalog(product, splitcat,
                                                verbose=(_vlevel(verbose) >= 2))
    seed = int(seed if seed is not None else getattr(config, "seed", 1))
    fit_kw = dict(FIT_DEFAULTS, **(fit_kw or {}))
    v = _vlevel(verbose)

    base_fits = {}          # base name -> full-stack fit_result (the fallback)
    out = OrderedDict()

    for name, rec in recs.items():
        if v:
            print(f"\n=== {name} ===")
        el = resolve_eligibility(product.catalog, rec, verbose=(v >= 2))
        if el["blocked"]:
            out[name] = {"recipe": rec, "sides": OrderedDict(), "compare": None,
                         "provenance": None, "warnings": [f"BLOCKED: {el['blocked']}"]}
            if v:
                print(f"  BLOCKED: {el['blocked']}")
            continue

        floor = int(min_n if min_n is not None else rec["min_n"])
        sides_cut = cut(product, rec, eligible=el["mask"], verbose=(v >= 2))
        n_sides = len(sides_cut)
        key = _cache_key(config, rec, level, seed, fit_kw, nboot_fit, splitcat,
                         product=product)

        # LAZY. The fallback fit is a full stack + bootstrap + fit, and a batch
        # whose sides are all cached must not pay for it -- that would make
        # "resume" cost a fixed several minutes per recipe for a result it then
        # throws away.
        fb = lambda: _base_fallback_fit(config, product, rec, base_fits,
                                        seed=seed, fit_kw=fit_kw, verbose=v)

        sides = OrderedDict()
        for side_index, (label, info) in enumerate(sides_cut.items()):
            cached = (None if (force or not cache)
                      else _load_side(config, rec, label, key, splits_dir))
            if cached is not None:
                if v:
                    print(f"  {name}/{label:<18} N={cached.get('n_gal'):<5} "
                          f"{'cached':<9}      --")
                sides[label] = cached
                continue
            if info["n"] == 0:
                sides[label] = {"label": label, "n_gal": 0,
                                "warnings": ["empty side -- nothing measured"],
                                "power": {}, "level": level}
                continue
            # ONE SIDE MUST NEVER KILL A BATCH. A 15-recipe run at level=
            # 'standard' is ~40 minutes; before this, any exception anywhere in
            # the per-side chain propagated straight out of run() and the whole
            # thing died. (The completed sides were still on disk -- the cache is
            # written per side, immediately -- but the in-memory `res` was lost
            # and there was no way to see what had failed without re-running.)
            #
            # So: catch, record the traceback ON THE SIDE, keep going. A failed
            # side is a first-class result with n_gal, a 'failed' flag and its
            # error text; describe()/table() see the warnings and compare() skips
            # it for want of quantities. It is deliberately NOT cached -- a
            # failure is usually a bug or a bad model choice, and caching it
            # would make the fix invisible on the next run.
            try:
                side = _measure_side(config, info["product"], rec, label, level=level,
                                     seed=seed, min_n=floor, n_sides=n_sides,
                                     fallback_fit=fb(), fit_kw=fit_kw,
                                     nboot_fit=nboot_fit, verbose=verbose)
            except Exception as exc:                              # noqa: BLE001
                tb = traceback.format_exc(limit=6)
                if v:
                    print(f"\n  !! {name}/{label}: FAILED -- "
                          f"{type(exc).__name__}: {exc}\n"
                          f"     continuing with the rest of the batch; this side "
                          f"is NOT cached.")
                sides[label] = {
                    "label": label, "n_gal": int(info["n"]), "failed": True,
                    "error": f"{type(exc).__name__}: {exc}", "traceback": tb,
                    "warnings": [f"SIDE FAILED ({type(exc).__name__}): {exc}"],
                    "power": {}, "level": level,
                    "base_sample": base_sample or rec["base"],
                }
                continue
            side.update(base_sample=base_sample or rec["base"],
                        cut_lo=info["cut_lo"], cut_hi=info["cut_hi"],
                        cut_value=info["cut_value"],
                        split_median=info["median"],
                        control_median=info["control_median"])
            if cache:
                _save_side(config, rec, label, side, key, side_index=side_index,
                           splits_dir=splits_dir, verbose=(v >= 2))
            sides[label] = side

        out[name] = {
            "recipe": rec,
            "sides": sides,
            "compare": compare(sides, rec),
            "provenance": {
                **key, "seed": seed, "level": level, "nboot_fit": nboot_fit,
                "nboot": int(config.n_bootstrap), "fit": fit_kw,
                "base": rec["base"], "min_n": floor,
                "eligibility_steps": el["steps"],
                "cache_dir": cache_dir(config, splits_dir),
            },
            "warnings": [w for s in sides.values() for w in s.get("warnings", [])],
        }
        if v:
            _print_compare(out[name])

    # END-OF-BATCH ROLL-UP. Anything that failed or degenerated scrolled past 40
    # minutes ago; without this you would have to page back through the whole log
    # to find out whether the table you are about to write is complete.
    if v:
        bad = [(n, l, s) for n, r in out.items()
               for l, s in (r.get("sides") or {}).items()
               if s.get("failed") or any("DEGENERATE" in w or "NO BOUNDARY" in w
                                         or "no core/two-halo boundary" in w
                                         for w in s.get("warnings", []))]
        blocked = [n for n, r in out.items()
                   if any(str(w).startswith("BLOCKED") for w in r.get("warnings", []))]
        print("\n" + "=" * 66)
        print(f"BATCH SUMMARY: {len(out)} recipe(s), "
              f"{sum(len(r.get('sides') or {}) for r in out.values())} side(s)")
        if blocked:
            print(f"  BLOCKED recipes ({len(blocked)}): {', '.join(blocked)}")
        if bad:
            print(f"  sides with no usable zone split ({len(bad)}):")
            for n, l, s in bad:
                why = (s.get("error") if s.get("failed")
                       else next((w for w in s.get("warnings", [])
                                  if "DEGENERATE" in w or "boundary" in w), "?"))
                print(f"    {n}/{l}: {str(why)[:96]}")
            print("  (these are recorded, not cached -- fix the cause and re-run "
                  "just those recipes)")
        if not bad and not blocked:
            print("  no failed or degenerate sides.")
        print("=" * 66)
    return out


def _base_fallback_fit(config, product, recipe, cache_dict, *, seed, fit_kw,
                       verbose=True):
    """The full-stack fit on THIS recipe's base sample, used as
    find_core_halo_boundary's fallback when a side's own fit doesn't cross.

    Per base, not per recipe: a KEEP-based recipe (the AGN split) must fall back
    to the KEEP full stack, not the CLEAN one, or its boundary comes from a
    sample it isn't drawn from. Computed once per base per session and reused.

    Failures are MEMOIZED AS None, not re-raised. This fit is shared by every side
    of every recipe on the same base, so a crash here would otherwise be blamed on
    whichever side happened to trigger it -- and then retried, at full cost, for
    the next side, and the next. Falling back to no-fallback is the right
    degradation: find_core_halo_boundary just has no second opinion, which is the
    same position it is in for the very first recipe of any run."""
    base = str(recipe["base"] or "ALL")
    if base in cache_dict:
        return cache_dict[base]
    cat = product.catalog
    mask = (np.ones(len(cat), bool) if base.upper() == "ALL"
            else np.asarray(cat[base]).astype(bool))
    if not mask.any():
        cache_dict[base] = None
        return None
    v = _vlevel(verbose)
    if v:
        print(f"  [fallback fit on base={base}, N={int(mask.sum())} "
              f"-- once, reused by every side]")
    try:
        sub = selection.slice_product(product, mask)
        st = stack.build_stacks(config, sub, keep_cube=True)
        bt = measure.measure_all_bins(config, st, seed=seed, verbose=(v >= 2))
        fr = fit_profile(bt, st, verbose=False, **fit_kw)
    except Exception as exc:                                      # noqa: BLE001
        if v:
            print(f"  !! fallback fit on base={base} FAILED "
                  f"({type(exc).__name__}: {exc}) -- every side on this base will "
                  f"run WITHOUT a fallback boundary. Not retried.")
        cache_dict[base] = None
        return None
    cache_dict[base] = fr
    return fr


# =====================================================================
# 12.  COMPARISON  (spec Sec 5.3 / 7.4)
# =====================================================================

def compare(sides, recipe) -> dict:
    """
    Per-quantity differences between sides, plus trend statistics for
    mode='trend'.

    BINARY: reports the difference and its 16/84 interval and lets the reader
    threshold it -- no p-value. The interval comes from element-wise
    differencing of the two sides' per-draw arrays where both exist. That is
    legitimate BECAUSE the two sides contain different galaxies and their
    bootstraps are therefore independent: pairing draw b of one with draw b of
    the other samples the difference distribution correctly, and it preserves
    the skew that quadrature would flatten. Where a draw array is missing it
    falls back to quadrature on the half-widths and says so in 'method'.

    TREND: a Spearman rank correlation between bin index and value with a
    permutation p-value, plus a straight-line slope in the split variable's own
    units against each bin's REALIZED median. A monotonic trend across four bins
    is one test WITH an ordering constraint -- 1 in 12 under the null, versus
    1 in 2 for a binary split -- which is the whole reason trend mode exists.
    """
    rec = normalize_recipe(recipe)
    # A side that raised is excluded outright rather than carried as a column of
    # Nones: with it in, `a, b = vals[0], vals[-1]` would compare against a
    # missing extreme and quietly report no difference for every quantity.
    labels = [l for l, s in sides.items() if s.get("n_gal") and not s.get("failed")]
    out = {"labels": labels, "mode": rec["mode"], "quantities": OrderedDict()}
    if len(labels) < 2:
        return out

    for q in QUANTITIES:
        vals = [_quantity(sides[l], q) for l in labels]
        fids = [v[0] for v in vals]
        if all(f is None for f in fids):
            continue
        entry = {"per_side": OrderedDict(
            (l, {"fid": v[0], "lo": v[1], "hi": v[2]}) for l, v in zip(labels, vals))}
        entry["power"] = {l: (sides[l].get("power") or {}).get(q) for l in labels}

        a, b = vals[0], vals[-1]         # extremes
        if a[0] is not None and b[0] is not None:
            entry["diff_fid"] = float(b[0] - a[0])
            if a[3] is not None and b[3] is not None:
                n = min(len(a[3]), len(b[3]))
                d = np.asarray(b[3][:n], float) - np.asarray(a[3][:n], float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    entry.update(diff_lo=float(np.nanpercentile(d, 16)),
                                 diff_hi=float(np.nanpercentile(d, 84)),
                                 diff_med=float(np.nanmedian(d)),
                                 method="paired independent draws")
            elif None not in (a[1], a[2], b[1], b[2]):
                sa = ((a[2] - a[0]) + (a[0] - a[1])) / 2.0
                sb = ((b[2] - b[0]) + (b[0] - b[1])) / 2.0
                s = float(np.hypot(sa, sb))
                entry.update(diff_lo=entry["diff_fid"] - s,
                             diff_hi=entry["diff_fid"] + s,
                             method="quadrature (no draw arrays)")
            else:
                entry["method"] = "point estimate only"

        if rec["mode"] == "trend" and sum(f is not None for f in fids) >= 3:
            entry["trend"] = _trend_stats(
                fids, [sides[l].get("split_median") for l in labels])
        out["quantities"][q] = entry
    return out


def _trend_stats(fids, medians, n_perm=10000, seed=12345) -> dict:
    """
    Spearman(bin index, value) with a label-permutation p, plus an OLS slope
    against the realized median of the split variable in each bin.

    ** READ p_floor BEFORE CHOOSING n_bins. ** The permutation null here has only
    n_bins! distinct orderings, so the SMALLEST two-sided p this test can ever
    return is 2/n_bins! -- attained when the bins come out perfectly monotonic:

        n_bins = 3  ->  p_floor = 0.333     can never be "significant" at all
        n_bins = 4  ->  p_floor = 0.083     cannot reach p < 0.05 EITHER
        n_bins = 5  ->  p_floor = 0.017     first n_bins that can clear 0.05
        n_bins = 6  ->  p_floor = 0.0028

    This is a property of ranking 4 numbers, not of the data or the sample size,
    and no amount of extra galaxies changes it. It matters because the spec's
    trend-mode default is n_bins=4: a perfect four-bin monotonic trend is
    genuinely 1-in-12 under the null (spec Sec 5.3, and that IS a real gain over
    a binary split's 1-in-2), but it will never print p < 0.05, and reporting it
    as "not significant" would misread the test. Either quote rho and the
    1-in-12 directly, or use n_bins=5 -- at the cost of ~88 galaxies per bin,
    which spec Sec 7.3 already restricts to core/inner quantities.
    """
    y = np.asarray([np.nan if f is None else f for f in fids], float)
    ok = np.isfinite(y)
    idx = np.arange(len(y), dtype=float)[ok]
    yv = y[ok]
    n = len(yv)
    if n < 3:
        return {"rho": None, "p": None, "slope": None}
    rho = _spearman(idx, yv)
    rng = np.random.default_rng(seed)
    null = np.array([_spearman(idx, rng.permutation(yv)) for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(rho)).mean())
    x = np.asarray([np.nan if m is None else m for m in medians], float)[ok]
    slope = None
    if np.isfinite(x).all() and np.ptp(x) > 0:
        slope = float(np.polyfit(x, yv, 1)[0])
    p_floor = 2.0 / float(math.factorial(n))   # math, not np.math (gone in numpy 2)
    return {"rho": float(rho), "p": p, "n_perm": n_perm, "slope": slope,
            "p_floor": p_floor, "n_bins_used": n,
            "abscissa": "realized median of the split variable per bin"}


def _print_compare(res):
    cmp_ = res.get("compare") or {}
    qs = cmp_.get("quantities") or {}
    if not qs:
        return
    labels = cmp_["labels"]
    print(f"\n  {res['recipe']['name']}: {' vs '.join(labels)}")
    print(f"    {'quantity':<22} " + " ".join(f"{l[:14]:>16}" for l in labels)
          + f" {'difference':>26}  power")
    for q, e in qs.items():
        cells = []
        for l in labels:
            v = e["per_side"][l]["fid"]
            cells.append(f"{'n/a':>16}" if v is None else f"{v:>16.4g}")
        if "diff_fid" in e:
            if "diff_lo" in e:
                d = (f"{e['diff_fid']:+.4g} "
                     f"[{e['diff_lo']:+.3g},{e['diff_hi']:+.3g}]")
            else:
                d = f"{e['diff_fid']:+.4g}"
        else:
            d = "n/a"
        pw = e["power"].get(labels[0]) or "?"
        print(f"    {q:<22} " + " ".join(cells) + f" {d:>26}  {pw}")
        if "trend" in e and e["trend"]["rho"] is not None:
            t = e["trend"]
            sl = "" if t["slope"] is None else f", slope={t['slope']:.4g}"
            print(f"      trend: Spearman rho={t['rho']:+.3f}, "
                  f"permutation p={t['p']:.4f}{sl}   "
                  f"[p_floor={t['p_floor']:.3f} at {t['n_bins_used']} bins]")
    tr = next((e["trend"] for e in qs.values()
               if e.get("trend", {}).get("p_floor")), None)
    if tr and tr["p_floor"] > 0.05:
        print(f"    ** with {tr['n_bins_used']} bins the permutation test CANNOT "
              f"return p < 0.05 -- its floor is {tr['p_floor']:.3f} "
              f"(= 2/{tr['n_bins_used']}!). A perfect monotonic trend is still "
              f"1-in-{int(round(1/tr['p_floor']))} under the null and worth "
              f"reporting AS THAT; do not report it as 'not significant'. Use "
              f"n_bins=5 if you need a p below 0.05.")
    if any(v == "insufficient" for e in qs.values() for v in e["power"].values()):
        print("    ** quantities flagged 'insufficient' are the two-halo ones: "
              "NO split supports a >2sigma outermost-bin claim with current "
              "data (spec Sec 7.3). The number is computed; the claim is not "
              "available.")


# =====================================================================
# 13.  COMPARING SPLITS TO EACH OTHER  (spec Sec 3.2, roadmap Sec 3.3b)
# =====================================================================

def intersect(config, product, recipes, *, splitcat=None, id_col="ID",
              name_suffix="_isect", verbose=True, **run_kw):
    """
    Re-run a set of recipes on their COMMON eligible sample.

    Catalog splits and core splits run on different samples: a mass split uses
    every galaxy with a mass, a Delta_v split only those with a detected Lya
    core. Putting the two in one table and reading a difference between them
    compares SAMPLE COMPOSITION as much as physics. This re-runs both on the
    galaxies that satisfy every recipe's eligibility at once, which is the only
    version of that comparison that means anything.

    Costs sample -- often a lot of it. Use it when two splits are being compared
    TO EACH OTHER; the headline number for any single split should still come
    from its own full eligible sample.

    Recipes are re-run under new names (`<name><name_suffix>`) with an added
    `require`, so the intersection results cache separately and can never be
    confused with the full-sample ones.
    """
    recs = resolve_recipes(recipes)
    if splitcat is not None:
        product = multicat.attach_split_catalog(product, splitcat, verbose=False)
    cat = product.catalog

    common = np.ones(len(cat), bool)
    for name, rec in recs.items():
        el = resolve_eligibility(cat, rec)
        if el["blocked"]:
            raise KeyError(f"{name}: {el['blocked']}")
        common &= el["mask"]
    if verbose:
        print(f"intersect: {int(common.sum())} galaxies satisfy all "
              f"{len(recs)} recipes' eligibility simultaneously.")
        for name, rec in recs.items():
            el = resolve_eligibility(cat, rec)
            print(f"  {name:<20} {el['n']:>5} alone  ->  {int(common.sum()):>5} "
                  f"in common  ({int(common.sum())/max(el['n'],1):.0%} kept)")
    if not common.any():
        raise ValueError("the common sample is empty -- these recipes cannot be "
                         "compared to each other on one sample.")

    ids = set(np.asarray(cat[id_col]).astype(str)[common])
    sub = selection.slice_product(product, common)
    new = []
    for name, rec in recs.items():
        new.append(dict(rec, name=f"{name}{name_suffix}",
                        base="ALL",           # the intersection IS the base now
                        doc=(rec["doc"] + f"  [INTERSECTION RERUN: common sample "
                                          f"of {sorted(recs)}, N={len(ids)}]")))
    return run(config, sub, new, splitcat=None, verbose=verbose, **run_kw)


# =====================================================================
# 14.  REPORTING
# =====================================================================

def describe(result, *, sides=None, verbose=True):
    """
    Print one full summary per side, via analysis.describe_subsample_properties
    -- to which every side dict is already shaped, so this is a loop and a
    header, not an adapter.

    Also prints the recipe's pre-specification (the `predicted` string and the
    confirmatory/exploratory/decomposition tier) BEFORE the numbers. That
    ordering is the point of Sec 7.1: at writing time the difference between
    "we predicted this and found it" and "we found this" should be a fact on
    record, not a memory.
    """
    rec = result["recipe"]
    tier = ("CONFIRMATORY (pre-registered)" if rec["confirmatory"] else
            ("DECOMPOSITION of a headline result -- not an independent test"
             if rec["decomposition"] else "EXPLORATORY"))
    if verbose:
        print("=" * 72)
        print(f"{rec['name']}   [{tier}]")
        print(f"  column={rec['column']}  mode={rec['mode']}  "
              f"control={rec['control']}  base={rec['base']}  "
              f"needs_core={rec['needs_core']}")
        if rec["predicted"]:
            who = rec["predicted_by"] or "unattributed"
            print(f"  PREDICTED (recorded before running, by {who}): "
                  f"{rec['predicted']}")
            if is_unreviewed(rec):
                print("  ** UNREVIEWED PREDICTION -- this is Claude's draft, not "
                      "yours. It cannot be\n     cited as a pre-registration "
                      "until you rewrite it and set predicted_by.")
        if rec["doc"]:
            print(f"  {rec['doc']}")
        print("=" * 72)

    for label, side in result["sides"].items():
        if sides is not None and label not in sides:
            continue
        if not side.get("n_gal"):
            print(f"\n[{label}] EMPTY")
            continue
        analysis.describe_subsample_properties(
            label=label, n_gal=side.get("n_gal"),
            base_sample=side.get("base_sample"),
            fit_result=side.get("fit_result"),
            boundary_info=side.get("boundary_info"), vel=side.get("vel"),
            core_lum=side.get("core_lum"), onehalo_lum=side.get("onehalo_lum"),
            twohalo_lum=side.get("twohalo_lum"), ratios=side.get("ratios"),
            boot_fit=side.get("boot_fit"))
        cv = side.get("cut_value")
        if cv is not None:
            print(f"\n  realized cut in {rec['column']}: {cv:.4g}  "
                  f"(median on this side: {side.get('split_median')})")
        for w in side.get("warnings", []):
            print(f"  ! {w}")
    if verbose:
        _print_compare(result)


def table(results, columns=None, *, fmt="latex", caption=None, label=None,
          level_note=True, **kw) -> str:
    """
    A table straight from cached results, no adapter code (spec Sec 10, step 8).
    Each side becomes one row, consuming the same kwarg names the side dicts
    already use.

    fmt : "latex"    -> formatting.make_subsample_latex_table. The PAPER table:
                        footnote letters, units in a Note line, journal
                        environments. Paste into the .tex draft.
          "markdown" -> formatting.make_subsample_markdown_table. The READING
                        table: units in the headers, recipe and side as separate
                        sortable columns, an N column, and a flags column that
                        surfaces degenerate fits and fallback boundaries.
                        Renders in Obsidian, a Jupyter markdown cell, or GitHub.
          "md" is accepted as an alias.

    `results` is run()'s OrderedDict, a single result, or a list of side dicts.
    Extra kwargs forward to whichever formatter you picked -- scales/sig/registry
    are common to both; table_env/fontsize/rotate are LaTeX-only; math/show_n/
    show_flags/split_label are markdown-only.

    Rows are labelled '<recipe>: <side>'; the markdown renderer splits that back
    into two columns. A note records the measurement level, because a
    level='quick'/'standard' row's one-halo luminosity is NOT leak-corrected and
    must not sit unmarked next to a level='full' one.

    FAILED SIDES ARE INCLUDED, not silently dropped. A side that raised has no
    numbers, so every cell reads '--' (markdown) or 'n/a' (LaTeX) -- but in
    markdown it carries a FAILED flag and its N, which is the difference between
    "we measured nothing there" and "this row quietly vanished from the table".
    """
    fmt = {"md": "markdown"}.get(str(fmt).lower(), str(fmt).lower())
    if fmt not in ("latex", "markdown"):
        raise ValueError(f"fmt must be 'latex' or 'markdown' (got {fmt!r}).")

    rows, levels = [], set()
    for rname, res in _iter_results(results):
        for slabel, side in res["sides"].items():
            if not side.get("n_gal"):
                continue
            rows.append(dict(side, label=(f"{rname}: {slabel}" if rname else slabel)))
            levels.add(side.get("level"))
    if not rows:
        raise ValueError("no non-empty sides to tabulate.")

    lvl = ""
    if level_note and levels:
        lvl = (f" Measurement level: {'/'.join(sorted(str(l) for l in levels))}."
               + (" Rows at level `quick'/`standard' have a one-halo luminosity "
                  "that is NOT PSF-leak corrected." if levels - {"full"} else ""))

    if fmt == "markdown":
        cap = (caption or "Derived Lya halo properties by subsample.") + lvl
        return formatting.make_subsample_markdown_table(
            rows, columns=columns, caption=cap.strip(), **kw)

    cap = (caption or "Derived Ly$\\alpha$ halo properties by subsample.") + lvl
    return formatting.make_subsample_latex_table(
        rows, columns=columns, caption=cap,
        label=label or "tab:subsample-properties", **kw)


def _iter_results(results):
    """(recipe_name, result) pairs from any of the shapes run() can hand back."""
    if isinstance(results, dict) and "sides" in results:
        return [(results.get("recipe", {}).get("name", ""), results)]
    if isinstance(results, dict):
        return list(results.items())
    return [("", r) for r in results]


def load(config, recipes=None, *, product=None, level="standard", seed=None,
         nboot_fit=200, fit_kw=None, splitcat=None, splits_dir=None,
         verbose=True) -> "OrderedDict[str, dict]":
    """
    Rehydrate every cached side for these recipes under the CURRENT extraction
    signature, without computing anything. The arguments that enter the cache
    key (product, level, seed, nboot_fit, fit_kw, splitcat) must match the run
    that wrote them -- if they do not, nothing is found, which is the correct
    behaviour rather than a near-miss.

    `product` is part of the key (see _product_hash) -- pass the SAME product
    the batch ran on. Passing None looks for sides written by a run that also
    had no product, i.e. nothing run() ever wrote, so it will come back empty.

    Use this to build a table in a fresh session, or to see what a killed batch
    got through before it died.
    """
    recs = resolve_recipes(recipes)
    seed = int(seed if seed is not None else getattr(config, "seed", 1))
    fit_kw = dict(FIT_DEFAULTS, **(fit_kw or {}))
    out = OrderedDict()
    for name, rec in recs.items():
        key = _cache_key(config, rec, level, seed, fit_kw, nboot_fit, splitcat,
                         product=product)
        d = cache_dir(config, splits_dir)
        if not os.path.isdir(d):
            break
        found = []
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or not f.startswith(f"{_safe(name)}__"):
                continue
            with open(os.path.join(d, f)) as fh:
                payload = json.load(fh)
            if payload.get("key") != key:
                continue
            side = _load_side(config, rec, payload["label"], key, splits_dir)
            if side is not None:
                found.append((payload.get("side_index", 0), payload["label"], side))
        # Sort by the ORIGINAL side index, not the filename. Side order is
        # meaningful -- compare() takes the difference between the FIRST and
        # LAST side, so a low/high pair loaded alphabetically would report the
        # difference with the sign flipped, and nothing would look wrong.
        sides = OrderedDict((lab, s) for _, lab, s in sorted(found, key=lambda t: t[0]))
        if sides:
            out[name] = {"recipe": rec, "sides": sides,
                         "compare": compare(sides, rec),
                         "provenance": {**key, "seed": seed, "level": level,
                                        "loaded_from": d},
                         "warnings": [w for s in sides.values()
                                      for w in s.get("warnings", [])]}
    if verbose:
        n = sum(len(r["sides"]) for r in out.values())
        print(f"loaded {n} cached side(s) across {len(out)} recipe(s) from "
              f"{cache_dir(config, splits_dir)}")
        missing = [n for n in recs if n not in out]
        if missing:
            print(f"  not cached at these settings: {missing}")
    return out
