# Splits pipeline — runnable notebook cells

Companion to `../specs/subsample-split-pipeline.md`. Every cell below assumes
`1_Measure_Splits.ipynb`'s cell 0/1 have already run, i.e. you have `cfg2`,
`products` (`{'AEGIS':…, 'COSMOS':…}`) and `prod_combined` in scope.

Run order: **A → B → C → D → E → F → G**. A and B are one-time-per-catalog;
C–E are cheap and you should re-run them freely; F is the expensive one.

---

## Cell 0 — paths and your own recipes (put this at the top, once)

```python
from utils_lya_halo import splits, multicat, core, stack

# WHERE SPLIT RESULTS GO. Separate from cfg.output_dir on purpose: output_dir
# also sets the Stage-1 per-galaxy cache (<output_dir>/cache), so repointing it
# would orphan a 20-hour extraction.
splits.SPLITS_DIR = "./outputs_measurements"
#   -> ./outputs_measurements/splits/<extract_signature>/<recipe>__<side>__*.json
# Per-call override if you ever want one: splits.run(..., splits_dir="...")

SPLITCAT_PATH = "../catalogs/split_catalogs/split_catalog.fits"
RECIPES_PATH  = "../catalogs/split_catalogs/my_recipes.json"

# YOUR recipes, from a JSON file — never by editing splits.py.
# First time only, to get a correctly-shaped starting point:
#     splits.save_library(RECIPES_PATH)      # dumps all 20, then you trim it
splits.load_library(RECIPES_PATH)            # every session after that
```

Merging is by name: an entry matching a shipped one **replaces** it, any other
name is **added**, and shipped recipes you didn't mention are left alone. So a
file holding only the three recipes you actually changed is complete and valid —
you never carry a copy of all 20, and my future edits to the shipped library
merge cleanly with yours.

---

## Cell A — imports + build the split catalog (Layer A)

One-time per catalog version. `build_split_catalog` does no measurement; it joins
already-measured things onto one ID-keyed table and records where each is missing.

```python
from utils_lya_halo import splits, multicat, core, stack

# ------------------------------------------------------------------ INPUTS
CORE_PATHS   = ["6-23-26-core_extraction_egs_MF.fits",
                "6-23-26-core_extraction_cosmos_MF.fits"]
CHARTAB_PATHS = []          # e.g. ["../catalogs/chartab2020_cosmos.fit",
                            #       "../catalogs/chartab2020_egs.fit"]
SPLITCAT_PATH = "./outputs/split_catalog.fits"
ID_COL        = None        # None -> auto (normally FIELD+ID). Pass "ID" or
                            # ("FIELD","ID") to pin it explicitly.

# ------------------------------------------------------------------ CORE PRODUCT
# Needed for every needs_core recipe (delta_v_lya, fwhm_lya, ew_lya, lya_lum,
# fesc_hactrl, lya_class). Skip it and those recipes report BLOCKED in preview()
# rather than silently running on the wrong sample.
core_prods = [core.read_core_fits(p) for p in CORE_PATHS]
prod_core  = stack.concat_core_prods(*core_prods) if len(core_prods) > 1 else core_prods[0]

# ------------------------------------------------------------------ EXTERNAL (Chartab overdensity)
external = []
if CHARTAB_PATHS:
    ext = multicat.load_overdensity(*CHARTAB_PATHS)
    external = [{"table": ext,
                 "columns": ["DELTA", "FIELD"],
                 "names": {"FIELD": "DENS_FIELD"},   # FIELD is part of the join key
                 "radius_arcsec": 0.5}]

# ------------------------------------------------------------------ BUILD
sc = multicat.build_split_catalog(
    prod_combined,
    core=prod_core,
    id_col=ID_COL,
    core_prefix="CORE_",
    external=external,
    derived=True,               # applies multicat.SPLIT_DERIVED
    derived_registry=None,      # dict(multicat.SPLIT_DERIVED, ...) to override
    carry=None,                 # None -> carry every halo-side column
    verbose=True,
)
multicat.write_split_catalog(SPLITCAT_PATH, sc, overwrite=True)

# Coverage: how many galaxies actually have each quantity, and what is left when
# several are required at once. Run this BEFORE scoping a batch, not after.
multicat.split_catalog_report(sc, intersect=True, verbose=True)
```

**Watch for:** the `CLEAN == KEEP & ~AGN_FLAG` line in the output. If it does not
say "exactly", two splits in the same table are running on different samples and
the AGN-definition TODO is now blocking.

---

## Cell B — see the library, and how redundant it is

No data needed for `show`; `independence` needs only the split catalog. Both are
seconds.

```python
splits.show()                        # every recipe: column, mode, control, base,
                                     # core tier, family, tier, prediction

splits.independence(
    sc,
    recipes=None,                    # None -> the whole library
    base="CLEAN",                    # sample the correlations are measured on
    verbose=True,
)
```

Read the **overlap matrix**, not M_eff. Two splits at 0.85 overlap are one split
done twice; 0.50 is chance. Pick recipes from different `family` values —
mass + inclination + overdensity + Δv_Lyα is four nearly-independent axes;
mass + SFR + sSFR + ΣSFR is one axis reported four times.

---

## Cell C — preview (do this before every batch)

Eligibility + cut + warnings only. No stacking, no bootstrap. Seconds for the
whole library, versus ~20 min per recipe to actually run.

```python
pv = splits.preview(
    cfg2, prod_combined,
    recipes=None,          # None -> whole library; or ["mass_half","inclination"]
    splitcat=sc,           # attached here so you don't have to remember to
    min_n=None,            # None -> each recipe's own min_n (default 80)
    imbalance=0.60,        # warn if one side holds >60% of the pair
    control_tol=0.10,      # warn if the control still differs by >10% of its own
                           # 16-84 spread between the extreme sides
    verbose=True,
)
```

Expect `mass_bounds` to run on placeholder edges, and `delta_ms` /
`dust_massctrl` / `size_massctrl` to come back **BLOCKED** with the missing
column named. That is the intended behaviour — fix the column, don't delete the
recipe.

---

## Cell D — the acceptance test (once, after any change to cutting)

Asserts this module's cut and `multicat.split_product_by` select **identical ID
sets**. If it fails, the runner diverged from a hand-run split at the very first
step and nothing downstream is trustworthy.

```python
for name in ["mass_half", "mass_bounds", "mass_trend4", "lya_class", "agn"]:
    assert splits.check_cut_equivalence(prod_combined_sc, name, id_col="ID",
                                        verbose=True)
```

`prod_combined_sc = multicat.attach_split_catalog(prod_combined, sc)` first —
this one function takes an already-attached product, unlike `run`/`preview`.

---

## Cell E — one recipe, cheap, to sanity-check the chain

Start here before a batch. `level="quick"` skips both expensive steps.

```python
res = splits.run(
    cfg2, prod_combined, "mass_half",
    splitcat=sc,
    level="quick",         # quick | standard | full
    force=False,           # True -> ignore cache
    min_n=None,
    seed=None,             # None -> cfg2.seed, else 1. SHARED across every recipe.
    nboot_fit=200,         # ignored at level="quick"
    fit_kw=None,
    cache=True,
    base_sample=None,      # label for the table; None -> the recipe's `base`
    verbose=True,
)
splits.describe(res["mass_half"])
```

---

## Cell F — the batch (the expensive one)

Resumable: one cache file per side, keyed on the extraction signature. Kill it
and re-run — it recomputes only what is missing.

```python
BATCH = ["mass_half", "mass_bounds", "sfr_massctrl", "sigma_sfr_massctrl",
         "inclination", "environment", "redshift_massctrl", "field",
         "delta_v_lya", "fwhm_lya", "ew_lya", "lya_lum", "lya_class", "agn"]

res = splits.run(
    cfg2, prod_combined, BATCH,
    splitcat=sc,
    level="standard",      # + refit-per-draw h1/r_c/crossover errors
    force=False,
    min_n=None,
    seed=None,
    nboot_fit=200,         # 200, NOT cfg2.n_bootstrap — each draw is a separate
                           # nonlinear fit. Raise it for the sides that end up
                           # in the paper.
    fit_kw=dict(
        model="expcore",       # expcore | twoexp
        method="psf",          # psf | naive
        fit_skip_inner=1,      # naive only
        gamma_fixed=0.8,
        r_c_fixed=400.0,       # None -> let r_c float
        psf_fwhm_arcsec=1.3,
        psf_beta=3.0,
    ),
    cache=True,
    verbose=True,
)
```

Then promote the ones that matter to `level="full"` — this adds the leak-aware
PSF-corrected core luminosity (a second per-galaxy-rescaled restack + bootstrap).
**Below `level="full"` the one-halo luminosity is NOT leak-corrected**, and every
side carries a warning saying so.

```python
res_full = splits.run(cfg2, prod_combined, ["mass_half", "inclination"],
                      splitcat=sc, level="full", nboot_fit=500, verbose=True)
```

---

## Cell G — read it, tabulate it, come back to it

```python
# per-side numbers, with the pre-specified prediction printed BEFORE them
for name in res:
    splits.describe(res[name], sides=None, verbose=True)

# LaTeX, straight from the results — no adapter code
print(splits.table(
    res,
    columns=None,          # None -> formatting.DEFAULT_COLUMNS
    caption=None,
    label="tab:subsample-properties",
    level_note=True,       # footnote the measurement level
    scales={"L_2halo": 1e43},
    sig=2,
    rotate=False,
))

# a fresh session, no recomputation. EVERY cache-key argument must match the
# run that wrote them -- including `product`, or it finds nothing.
res = splits.load(cfg2, BATCH, product=prod_combined, level="standard", seed=None,
                  nboot_fit=200, fit_kw=None, splitcat=sc, verbose=True)
```

---

## Cell H — comparing two splits **to each other**

Only when the comparison is between splits, not within one. Costs sample, often
a lot of it.

```python
res_i = splits.intersect(
    cfg2, prod_combined,
    ["mass_half", "lya_lum"],     # catalog split vs core split
    splitcat=sc,
    id_col="ID",
    name_suffix="_isect",         # results cache separately; cannot be confused
                                  # with the full-sample ones
    level="standard",
    verbose=True,
)
```

---

## Changing things without touching the code

Everything below is data or an argument. None of it needs an edit to `splits.py`.

| what you want to change | how |
|---|---|
| a recipe's column, mode, control, `require`, `min_n`, labels | edit `my_recipes.json`, `load_library` |
| a **prediction** or `confirmatory`/`decomposition` tier | same — they're recipe fields |
| add a brand-new split | add an entry to `my_recipes.json` |
| one-off tweak, no file | `dict(splits.LIBRARY["mass_half"], name="…", **kw)` |
| where results are written | `splits.SPLITS_DIR` (or `splits_dir=` per call) |
| where the split catalog lives | it's just the path you pass to `write_split_catalog` |
| how the fit is done | `fit_kw=` on `run` — model, method, `gamma_fixed`, `r_c_fixed`, PSF |
| warn thresholds | `min_n=`, `imbalance=`, `control_tol=` on `preview` |
| bootstrap cost | `level=`, `nboot_fit=`, `cfg2.n_bootstrap` |
| which quantities are power-flagged | `splits.TWO_HALO_QUANTITIES` — a public mutable set. When the ~2× data lands, `.discard("twohalo_lum")` rather than editing `_power_flags` |
| derived columns (`SIGMA_SFR`, `RE_KPC`, …) | `multicat.SPLIT_DERIVED`, via `derived_registry=` in cell A |

A recipe file is validated on load, so a typo'd key raises immediately and names
the file — not three hours into a batch:

```
ValueError: my_recipes.json: unknown recipe key(s) ['moed']. Valid keys: [...]
```

---

## Cell I — a custom recipe

Recipes are plain dicts. Amend a library entry inline; unknown keys raise.

```python
rec = dict(splits.LIBRARY["mass_half"],
           name="mass_trend5",          # must be unique — it is the cache filename
           mode="trend", n_bins=5,      # 5, not 4 — see the p_floor note below
           min_n=60,
           predicted="monotonic h1(M*)",
           confirmatory=False)

res = splits.run(cfg2, prod_combined, rec, splitcat=sc, level="standard")
```

Brand-new split, from scratch:

```python
rec = dict(
    name="my_split", column="SOME_COLUMN", mode="half",
    control=None, n_control_bins=4,
    base="CLEAN", require=("SOME_FLAG == 0",),
    needs_core=None,            # None | "covered" | "detected"
    min_n=80, per_field=False,
    labels=("low", "high"),
    family=2, confirmatory=True, decomposition=False,
    predicted="…", doc="…",
)
splits.preview(cfg2, prod_combined, rec, splitcat=sc)    # always preview first
```

---

## Two numbers to know before you write any of this up

**`n_bins=4` cannot produce p < 0.05.** The trend permutation null has only
`n_bins!` orderings, so the smallest attainable two-sided p is `2/n_bins!`:

| `n_bins` | 3 | 4 | 5 | 6 |
|---|---|---|---|---|
| p floor | 0.333 | **0.083** | 0.017 | 0.003 |

A perfect four-bin monotonic trend is genuinely 1-in-12 under the null, and that
is a real gain over a binary split's 1-in-2 — but it will never print p < 0.05.
Quote ρ and the 1-in-12 directly, or use `n_bins=5`. Never write "not
significant" about a four-bin trend. The runner prints `p_floor` alongside every p.

**Every two-halo quantity comes back `power="insufficient"`, always.** Full-stack
outer-bin S/N is 6 / 5 / 2.8; a two-way split gives 4.2 / 3.5 / **2.0**. The
number is computed and the flag travels with it into the table; the *claim* is
not available with current data.
