# The splits pipeline — how it works

Read this once, top to bottom. It assumes nothing.

For copy-paste cells see `splits-quickstart.md`. For the design reasoning see
`../specs/subsample-split-pipeline.md`. This file is the middle thing: what each
piece *is* and what actually happens when you run it.

---

## 1. The problem it solves

You want to ask, fifteen or twenty times: **"does the stacked Lyα halo look
different for high-mass galaxies than low-mass ones?"** — and then the same
question for SFR, inclination, environment, Lyα velocity offset, and a dozen more.

Each one is the same six steps: pick which galaxies go on each side, stack each
side, bootstrap it, fit the profile, pull out the derived numbers, put them in a
table. Doing that by hand in a notebook is twenty near-identical cells that
slowly drift apart from each other.

This pipeline makes each of those questions **one line of data** and the whole
batch **one command**.

---

## 2. The four things you hold

Everything runs on four objects. Three you already have from the existing
pipeline; only the fourth is new.

### The config — `cfg2`

A `PipelineConfig`. It says how the data was extracted and how it should be
measured: which field, the radial bin edges, the background annulus, the line
window, how many bootstrap draws. You already build this in cell 1 of
`1_Measure_Splits.ipynb` with `replace(cfg, ...)`.

The splits pipeline does not change your config or ask for a new one. It reads it.

```python
cfg2.bins            # radial bin edges
cfg2.n_bootstrap     # 1000
cfg2.output_dir      # where things get written
cfg2.extract_signature()   # '65a007' -- a hash of every extraction knob
```

That last one matters later. It's a short hash that changes whenever anything
about the extraction changes. It's how results from two different pipeline
versions are kept from landing in the same table.

### The product — `prod_combined`

A `GalaxyProduct`. The actual data: for every galaxy, its spectrum in every
radial bin, plus a catalog table with one row per galaxy (mass, redshift, RA/DEC,
flags). You already build this by loading the galaxy FITS per field and combining.

**A split is just a mask on this table.** "High mass" means "the rows where
`MASS_50` is above the median". Nothing more exotic than that.

### The split catalog — `sc`

**This is the new one.** One row per galaxy, holding *every quantity you might
ever split on*, gathered from four places:

| source | example columns |
|---|---|
| the halo catalog (already on `prod_combined`) | `MASS_50`, `z`, `SFR_50`, `CLEAN` |
| the core product (your Lyα core extraction) | `CORE_V_OFFSET`, `CORE_SIGMA`, `CORE_LABEL` |
| external catalogs (Chartab overdensity) | `DELTA` |
| computed on the fly | `RE_KPC`, `SIGMA_SFR`, `CORE_LOGL_LYA` |

It does **no measurement**. It joins things that were already measured onto one
table, keyed on galaxy ID, and records `NaN` wherever a galaxy doesn't have that
quantity. You build it once per catalog version and write it to disk.

Why it exists: without it, "split by Lyα velocity offset" means remembering which
FITS file that lives in, matching it by position, and hoping the IDs line up.
With it, it's a column name.

```python
sc = multicat.build_split_catalog(prod_combined, core=prod_core, external=[...])
multicat.write_split_catalog("../catalogs/split_catalogs/split_catalog.fits", sc)
sc.colnames          # everything you can split on
```

To see what's actually usable:

```python
multicat.split_catalog_report(sc)
```

That prints how many galaxies have each column. Some columns cover all 510
galaxies; `CORE_*` columns only cover the ones with core spectra. That difference
is real and the pipeline tracks it rather than hiding it.

### A recipe — one dictionary

A recipe is **a plain Python dict describing one split**. It's data. It holds no
code and does nothing on its own.

```python
{
  "name":       "mass_half",
  "column":     "MASS_50",       # which split-catalog column to cut on
  "mode":       "half",          # cut it at the median
  "base":       "CLEAN",         # start from the clean, non-AGN sample
  "labels":     ["low M*", "high M*"],
  "predicted":  "high M* -> larger h1 and larger crossover radius",
}
```

That's a whole split. Twenty of these is the whole program.

---

## 3. The four ways to cut

`mode` is the only field that changes *how* the sample gets divided.

| mode | what it does | when |
|---|---|---|
| `half` | two equal-size halves at the median | the default. "Does it differ?" |
| `trend` | `n_bins` equal-size ordered bins | "Does it vary steadily?" |
| `bounds` | you give the actual values to cut at | when the physics happens at specific values, e.g. a particular mass |
| `groups` | categories | AGN vs not; emitter / null / absorber |

`half` and `trend` are equal-*population* cuts — the boundary lands wherever it
has to so both sides have the same number of galaxies. `bounds` is an equal-*value*
cut — you say `[0, 9.4, 10.2, 99]` and live with whatever counts come out.

Two optional modifiers:

- **`control`** — hold something fixed while you split. `control: "MASS_50"` on an
  SFR split means: bin the sample by mass first, split by SFR *within each mass
  bin*, then pool. Both sides end up with the same mass distribution, so a
  difference between them isn't just mass in disguise.
- **`per_field`** — cut separately in COSMOS and EGS, then pool. Needed for
  overdensity, because that number is normalized within its own field.

---

## 4. Who decides which galaxies are eligible

Before any cutting, the recipe decides who's allowed in. Four filters, in order:

1. **`base`** — the starting sample. `CLEAN` (443 galaxies, non-AGN) by default,
   `KEEP` (510, includes AGN) for the AGN split, `ALL` for diagnostics.
2. **core coverage** — added *only* if the recipe asks for it via `needs_core`.
3. **`require`** — any extra conditions you write, e.g. `"GALFIT_FLAG_F160W == 0"`.
4. **finite value** — the galaxy actually has a number in that column.

Step 2 is the one worth understanding. `needs_core` has three settings:

- `None` — the split has nothing to do with the Lyα core. **A mass split must be
  this.** A galaxy with a good mass and no Lyα detection still belongs in a mass
  split; dropping it would quietly turn your mass split into a mass split *of
  Lyα-bright galaxies*, which is a different question.
- `"covered"` — the galaxy was observed in the core, whether or not Lyα was
  detected. The emitter/null/absorber split uses this, because nulls and
  absorbers are the point.
- `"detected"` — Lyα was actually detected. Only for quantities that don't exist
  otherwise, like the Lyα velocity offset.

---

## 5. Who writes the predictions

**Right now, I do — and that's a placeholder, not a feature.**

Every recipe carries a `predicted` string: what you expect to see, written down
*before* you run it. At writing-up time that's the difference between "we
predicted this and found it" and "we found this". It's recorded, never enforced —
nothing is blocked by it.

I drafted all twenty from your own `docs/subsample_splits.md`. Each is stamped
`predicted_by: "claude-draft"`, and anything still carrying that stamp gets
flagged:

```
predicted: high M* -> larger h1 ...   [UNREVIEWED DRAFT]

** 20 prediction(s) are still Claude's draft. A prediction you did not write
   is not a pre-registration.
```

To make one yours, rewrite it in your recipe file and sign it:

```json
{"name": "mass_half",
 "predicted": "your own words here",
 "predicted_by": "austin"}
```

Signing a prediction does **not** invalidate any cached results — it's prose, not
a number.

Alongside it, `confirmatory: true/false` records whether the split was
pre-registered or is exploratory, and `decomposition: true` marks splits that
are a re-cut of a result you already have rather than an independent test (the
Lyα velocity-offset split is one of these).

---

## 6. Where the recipes live, and how to see them

The twenty shipped recipes are a **starting point in the code**. Your versions
live in a JSON file in your workspace:

```python
splits.save_library("../catalogs/split_catalogs/my_recipes.json")   # once
# ... edit that file in any text editor ...
splits.load_library("../catalogs/split_catalogs/my_recipes.json")   # each session
```

Loading merges **by name**: an entry matching a shipped recipe replaces it, a new
name is added, anything you didn't mention is untouched. So a file with just the
three recipes you changed is complete — you never keep a copy of all twenty, and
you never edit `splits.py`.

To view them:

```python
splits.show()                    # all of them, one line each + predictions
splits.show(["mass_half"])       # just one
splits.LIBRARY["mass_half"]      # the raw dict
```

### Every field a recipe can have

These are all of them. `splits.RECIPE_DEFAULTS` is the same list in code, and any
key not on it is rejected.

**Required**

| field | what it is |
|---|---|
| `name` | unique string. It's also the cache filename, so changing it starts a fresh set of results. |
| `column` | which split-catalog column to cut on. Must exist, or the recipe reports BLOCKED. |

**How to cut**

| field | default | notes |
|---|---|---|
| `mode` | `"half"` | `half` \| `trend` \| `bounds` \| `groups` |
| `n_bins` | `4` | `trend` only. Number of ordered bins. |
| `bounds` | `None` | `bounds` only. **N edges make N−1 bins** — `[0, 9.4, 10.2, 99]` is *three* bins. Two bins needs three numbers. |
| `groups` | `None` | `groups` only. List of lists: `[["COSMOS"], ["AEGIS","EGS"]]` puts AEGIS and EGS together. |
| `labels` | `None` | Display names, one per side. **Must match the side count** — change `bounds` or `n_bins` and you must change this too, or set it to `None` for automatic labels. |

**Who's eligible**

| field | default | notes |
|---|---|---|
| `base` | `"CLEAN"` | `CLEAN` (443, non-AGN) \| `KEEP` (510) \| `ALL` \| any boolean column |
| `require` | `[]` | Extra conditions as strings: `["GALFIT_FLAG_F160W == 0"]`. Any column name works, plus `np`. |
| `needs_core` | `None` | `None` \| `"covered"` \| `"detected"`. Leave `None` unless the split is about the Lyα core itself. |
| `min_n` | `80` | Warns below this. Never skips. |

**Fairness and pooling**

| field | default | notes |
|---|---|---|
| `control` | `None` | Column to hold fixed while splitting. Not available for `mode="groups"`. |
| `n_control_bins` | `4` | How finely to bin the control variable. More bins = tighter control, fewer galaxies per bin. |
| `per_field` | `False` | Cut inside COSMOS and EGS separately, then pool. |
| `field_col` | `"FIELD"` | Which column names the field. |

**Bookkeeping — no effect on any number**

| field | default | notes |
|---|---|---|
| `predicted` | `""` | What you expect, written before running. |
| `predicted_by` | `None` | Your name. `"claude-draft"` or unset flags it as unreviewed. |
| `confirmatory` | `False` | Was this pre-registered? |
| `decomposition` | `False` | Is this a re-cut of a result you already have, rather than an independent test? |
| `family` | `None` | `1` mass sequence, `2` externally set, `3` Lyα-internal. Used by `independence()`. |
| `doc` | `""` | Free notes. |

### Worked example: retargeting `mass_bounds`

`mass_bounds` ships as a three-bin mass cut. Say you want two bins split at 10.0:

```python
splits.edit_recipe(RECIPES_PATH, "mass_bounds",
                   bounds=[0, 10.0, 99],           # 3 edges -> 2 bins
                   labels=["logM<10", "logM>10"])  # 2 labels, matching
```

Change `bounds` and forget `labels` and it stops you immediately:

```
ValueError: recipe 'mass_bounds': 3 labels ['logM<9.4', '9.4-10.2', 'logM>10.2']
but mode='bounds' makes 1 side(s) (bounds=[0.0, 9.0] -> len(bounds)-1 = 1).
Update labels= to match, or pass labels=None for automatic ones.
```

To point it at a different variable entirely, change `column` **and** `bounds`
together — the edges are in that column's own units, so mass edges are
meaningless for E(B−V):

```python
splits.edit_recipe(RECIPES_PATH, "mass_bounds",
                   name="ebv_bounds", column="EBV",
                   bounds=[0, 0.15, 0.3, 9], labels=["low", "mid", "high"])
```

### Editing without a text editor

You don't have to open the JSON at all. One call changes it, validates it, saves
it, and reloads it so the next `preview`/`run` sees the change immediately:

```python
splits.edit_recipe(RECIPES_PATH, "dust_massctrl", column="EBV")

splits.edit_recipe(RECIPES_PATH, "inclination",
                   require=[],                       # drop the GALFIT flag cut
                   predicted="face-on -> more direct escape down the cone",
                   predicted_by="austin")            # now it's your prediction

splits.edit_recipe(RECIPES_PATH, "my_split",         # brand new one
                   column="AGE_50", mode="half", control="MASS_50",
                   labels=["young", "old"])
```

A recipe that only exists in the shipped library gets copied into your file on
first edit — so your file ends up holding exactly the recipes you've touched and
nothing else. A typo'd field raises immediately and lists the valid ones.

---

## 7. What happens when you run a split

```python
res = splits.run(cfg2, prod_combined, "mass_half", splitcat=sc)
```

Behind the scenes, for each side (`low M*` and `high M*`):

1. **Attach** the split catalog's columns onto the product's table.
2. **Filter** to the eligible galaxies (§4).
3. **Cut** into sides — for `half`, find the median of `MASS_50`, take rows below
   and rows above.
4. **Slice** the product down to that side's galaxies.
5. **Stack** them (`build_stacks`) — the same function your notebook already calls.
6. **Bootstrap** (`measure_all_bins`) — resample galaxies 1000 times to get error
   bars on the centroid and the flux profile.
7. **Fit** the radial profile — the same expcore + PSF fit as your headline figure,
   just without drawing it.
8. **Find the boundary** where the core term and the two-halo term cross.
9. **Measure** the derived numbers: core/one-halo/two-halo luminosity, core and
   two-halo velocity, the ratios.
10. **Save** it to disk.

Steps 5–9 are exactly what you'd type by hand. The runner's contribution is doing
it in the same order every time and writing down what it did.

Then it compares the two sides: for each quantity, side A's value, side B's value,
and the difference with an error bar.

### The three cost levels

Steps 6–9 aren't equally expensive, so:

| `level` | what it adds | rough cost per side |
|---|---|---|
| `"quick"` | steps 1–9 as above | minutes |
| `"standard"` *(default)* | proper error bars on the fit parameters (h1, r_c, crossover) by refitting every bootstrap draw | ~2–3× quick |
| `"full"` | the PSF-corrected core luminosity — a second stacking pass that rescales each galaxy individually | ~2× standard |

Start at `"quick"` to see if a split is interesting. Promote the ones that matter
to `"full"`.

---

## 8. What you get back

`run` returns a dictionary keyed by recipe name. For each recipe:

- **`sides`** — one entry per side, holding every measured number, the list of
  galaxy IDs that went into it, the realized cut value, and a power flag per
  quantity.
- **`compare`** — side A vs side B for each quantity, with the difference and its
  error bar.
- **`provenance`** — every hash, the seed, the level, the fit settings.

To read it:

```python
splits.describe(res["mass_half"])    # printed summary, prediction shown first
print(splits.table(res))             # LaTeX table, ready for the paper
```

**Power flags.** Each number carries `"ok"` or `"insufficient"`. Every two-halo
(outermost-bin) quantity is flagged `"insufficient"` on every split — the current
data doesn't support a >2σ claim out there once you've halved the sample. The
number is still computed; the flag rides along with it so it can't quietly become
a claim.

---

## 9. The cache — why re-running is free

A full batch is hours. So every side is written to its own file the moment it's
finished:

```
outputs_measurements/splits/<extract_signature>/mass_half__low_M__<hashes>.json
```

Re-run the same command and it loads from disk instead of recomputing. Kill it
halfway through and re-run — it picks up only what's missing. `force=True` if you
want it to recompute anyway.

The filename carries hashes of five things: **the recipe, the extraction, the
measurement settings, the split catalog, and which galaxies.** Change any of them
and you get a different file rather than a stale answer. That's the whole point —
it's how numbers from before and after a pipeline change are stopped from sitting
in the same table looking equally plausible.

Where it goes:

```python
splits.SPLITS_DIR = "./outputs_measurements"
```

This is deliberately separate from `cfg2.output_dir`, because that one also
controls your Stage-1 extraction cache and you don't want to move that by accident.

---

## 10. The order you actually work in

```python
# --- once per catalog version -------------------------------------------
sc = multicat.build_split_catalog(prod_combined, core=prod_core, external=[...])
multicat.write_split_catalog(SPLITCAT_PATH, sc)
multicat.split_catalog_report(sc)          # what's actually covered

# --- once, then whenever you change a recipe ----------------------------
splits.load_library(RECIPES_PATH)
splits.show()                              # what am I about to run

# --- seconds. do this before every batch --------------------------------
splits.independence(sc, graph=True)        # how redundant are they, as a figure
splits.preview(cfg2, prod_combined, splitcat=sc)
#   -> N per side, where the cut landed, which recipes are missing a column

# --- minutes. one split, cheap, to check the chain ----------------------
res = splits.run(cfg2, prod_combined, "mass_half", splitcat=sc, level="quick")
splits.describe(res["mass_half"])

# --- hours. the batch ---------------------------------------------------
res = splits.run(cfg2, prod_combined, BATCH, splitcat=sc, level="standard")

# --- the output ---------------------------------------------------------
print(splits.table(res))
```

`preview` is the one to build a habit around. It runs steps 1–3 only — no
stacking, no bootstrap — so the whole library previews in seconds and tells you
what each split would actually be before you spend the afternoon on it.

### Reading `preview`

Each side prints as `label=N @median` — the count, and the **realized median of
the split variable on that side**, in its own units:

```
mass_bounds   444  logM<9.4=122 @9.124 | 9.4-10.2=221 @9.801 | logM>10.2=101 @10.49
```

So you can see not just how the sample divided but *where each bin actually sits*
— whether the high bin is somewhere physically meaningful, whether two bins are
nearly on top of each other. Categorical splits have no median and show counts
only.

### Reading `independence` as a figure

At 15×15 the printed correlation tables are hard to scan:

```python
splits.independence(sc, graph=True, save_fig=True,
                    save_name="./outputs_measurements/split_independence.png")
```

Both matrices, lower triangle, colour only. Recipes are grouped by family with
separator lines drawn between them, so the redundancy the design *predicts* is a
visible block and anything dark **outside** a block is the surprise worth chasing.
Both panels share one diverging colormap on a fixed scale: white always means
independent (ρ = 0, overlap = 0.5), deep red always means the same split twice.
`sort_by_family=False` keeps library order.

### Controlling how much it prints

Both of the noisy commands have a quiet default and a loud option:

```python
splits.preview(...)                      # one line per recipe, warnings as tags
splits.preview(..., detail="full")       # every warning spelled out, per recipe

splits.run(..., verbose=1)               # one line per side (default)
splits.run(..., verbose=2)               # every bootstrap progress bar
splits.run(..., verbose=0)               # silent
```

At `verbose=1` a side prints one line when it finishes, with its time, its `h1`,
and its crossover radius:

```
  mass_half/low M*        N=218   quick      121s   h1= 11.67   r_x=  59.0   [1 note]
  mass_half/high M*       N=225   quick      122s   h1= 18.35   r_x=  81.8   [1 note]
```

A 14-recipe batch is then ~30 lines instead of ~2000.

---

## 11. Vocabulary

| term | means |
|---|---|
| **recipe** | one dict = one split |
| **library** | the collection of recipes, `splits.LIBRARY` |
| **side** | one output group of a split (`low M*`) |
| **base** | the starting sample before any recipe-specific filtering |
| **eligible** | the galaxies that survive all four filters |
| **level** | how much of the expensive measurement to do |
| **power flag** | whether a number has the S/N to support a claim |
| **h1** | the halo scale length — usually the headline number |
| **crossover / boundary radius** | where the galaxy's own halo gives way to the large-scale term |
| **one-halo / two-halo** | this galaxy's own gas / everything clustered around it |
| **extract signature** | short hash of the extraction settings; names the results folder |
