# Subsample split pipeline — Layers B and C

**Status: v2.1 spec, 2026-07-31. IMPLEMENTED 2026-07-31 as `utils_lya_halo/splits.py`
(~1500 lines) + `selection.split_controlled_bins` + a new `guide.py` "splits" section.
`check_guide()` clean (0 missing, 0 unregistered).**

**Implementation decisions taken at build time (these refine the spec, they do not
contradict it):**

1. **Three cost levels, default `standard`.** §5.1 lists a chain whose last two steps
   dominate runtime. `level="quick"` = stack + bootstrap + point-estimate fit + boundary
   + velocity + zone luminosities. `standard` adds `bootstrap_fit_profile` at a reduced
   `nboot_fit` (default 200, not the config's 2000 — each draw is a separate nonlinear
   fit). `full` adds the PSF-corrected core luminosity, a second restack+bootstrap pass.
   The level is part of the cache key, so levels never mix in one table, and `table()`
   footnotes it.
2. **Leak-aware core/one-halo.** Uses `measure_psf_corrected_core_onehalo_luminosity`
   (one pass, corrects the core AND removes the leaked flux from the one-halo zone)
   rather than the cheaper pair, per measure.py's 2026-07-22 double-count note. Below
   `level="full"` the one-halo number is NOT leak-corrected and every side carries an
   explicit warning saying so.
3. **Full-stack fallback boundary.** `find_core_halo_boundary` gets the full-stack fit on
   *that recipe's own base sample* as `fallback_fit_result` — per base, not per recipe, so
   the AGN split (`base="KEEP"`) falls back to the KEEP stack rather than the CLEAN one.
   Computed once per base per session, and **lazily**, so a fully-cached resume costs
   nothing. `boundary_from_own_fit=False` is recorded and warned on.
4. **Headless.** `splits.fit_profile` is `analysis.plot_flux_profile_fit`'s fit block
   extracted verbatim — same `y`, same `sigma` convention, same PSF construction, same
   `fitting.*` calls in the same order. No figures in a 30-side batch. If one changes,
   change both.

**Two findings from building it, both worth acting on:**

- **§5.3's Spearman permutation test cannot reach p < 0.05 at the default `n_bins=4`.**
  The null has only `n_bins!` orderings, so the smallest attainable two-sided p is
  `2/n_bins!` — 0.333 at 3 bins, **0.083 at 4**, 0.017 at 5. The spec's "1 in 12 under the
  null for four bins" is correct and is a real gain over a binary split's 1 in 2; what
  does not follow is that a four-bin trend can ever be *reported* as p < 0.05. Either
  quote ρ and the 1-in-12 directly, or use `n_bins=5`. `_trend_stats` now returns
  `p_floor` and `_print_compare` says this out loud.
- **§7.2b's M_eff (Li & Ji 2005) is the wrong estimator at this matrix size.** Its
  fractional term makes it nearly flat for small M: two variables at ρ = 0.84 have
  eigenvalues (1.84, 0.16) and Li & Ji returns exactly 2.00 — *no* reduction, right up
  until ρ = 1.0. `independence()` now reports Cheverud–Nyholt (1.65 for that pair) as the
  quotable number and Li & Ji as a conservative upper bound, and says to read the overlap
  matrix when they disagree. The overlap matrix — which is formula-independent — remains
  the number the spec's own acceptance test is written against, and it passes.

**Still open, unchanged:** `mass_bounds` edges are the `[0, 9.4, 10.2, MAX]` placeholder;
`DELTA_MS` is in the library but blocked on its calibration and `preview()` reports it as
blocked rather than silently skipping it; `dust_massctrl` points at `AV` and may need
repointing at whatever the catalog actually carries.

Turns "run a split" from a research task into a lookup. Layer A (the split catalog)
already exists in `multicat.build_split_catalog`. This spec covers the two layers
above it: a **declarative recipe library** and a **runner** that executes any list of
recipes into one paper-ready table.

**Changes in v2 (2026-07-31):** three named cut modes (`half` / `trend` / `bounds`)
instead of five overlapping ones; `min_n` warns instead of refusing; base sample is
explicit per recipe with `CLEAN` (= clean non-AGN) as the default; §7.2 rewritten around
the split-family taxonomy.

**v2.1 (same day), correcting v2:** core dependence is per-recipe eligibility, not a
sequencing prerequisite — the extraction is done and joined (§3.3); `DELTA_MS` pinned
and deferred (§4.1); mass `bounds` set manually; `INCL_DEG` / `Q_BEST` / `BA` and the
nebular line widths confirmed present on the catalog, so no new carried columns are
needed; **§7.5's regression-to-the-mean claim about `Δv_Lyα` and `FWHM_Lyα` retracted**
and replaced with the one coupling channel that actually survives.

**Companions**

- `../docs/subsample_splits.md` — the 15 candidate splits and their physical predictions (Layer B's content)
- `../../docs/workstream-roadmap.md` §3 — the three-layer design this implements, plus the power/multiplicity constraints
- `subsample-derived-properties.md` — the per-subsample derived quantities the runner collects (already implemented in `measure.py`/`analysis.py`)
- `../code/utils_lya_halo/multicat.py` — Layer A, `build_split_catalog` / `attach_split_catalog` / `split_product_by`
- `../code/utils_lya_halo/selection.py` — `split_controlled`, the matched-control machinery Layer C reuses

---

## Contents

1. [Goal](#1-goal)
2. [What already exists](#2-what-already-exists)
3. [Layer B — the recipe](#3-layer-b--the-recipe)
4. [Layer B2 — the recipe library](#4-layer-b2--the-recipe-library)
5. [Layer C — the runner](#5-layer-c--the-runner)
6. [Caching and resume](#6-caching-and-resume)
7. [Statistical guardrails](#7-statistical-guardrails)
8. [Public API](#8-public-api)
9. [Module layout](#9-module-layout)
10. [Build order and success criteria](#10-build-order-and-success-criteria)
11. [Open questions](#11-open-questions)

---

## 1. Goal

**One question.** For each of ~15 galaxy properties, does the stacked Lyα halo differ
between galaxies high and low in that property — in scale length, luminosity budget,
velocity, or shape?

**The requirement.** Adding the sixteenth split should be one dictionary entry, not a
new notebook cell. Running all sixteen should be one call. Running one should be the
same call with one argument. An interrupted 6-hour batch should resume, not restart.
Every number in the final table should be traceable to the exact extraction, cut, and
measurement parameters that produced it — with no possibility of two pipeline versions
landing in the same table.

**What this is not.** No new *measurement* code. Every quantity the runner reports is
already implemented (§2). This is orchestration, caching, and bookkeeping.

---

## 2. What already exists

Nothing below needs to be written. The runner is a composition of these.

| Piece | Where | What it does |
|---|---|---|
| Split catalog (Layer A) | `multicat.build_split_catalog` | one row/galaxy, every splittable column, `NaN` where unavailable, provenance in `.meta` |
| Attach to a product | `multicat.attach_split_catalog` | reindexes the split catalog onto any product by ID |
| Coverage report | `multicat.split_catalog_report` | per-column availability + pairwise intersections |
| Cut-column audit | `multicat._report_cut_columns` | verifies `CLEAN == KEEP & ~AGN_FLAG` and reports leakage |
| Uncontrolled split | `multicat.split_product_by` | `edges` / `percentiles` / `groups` → `OrderedDict{label: product}` |
| Controlled split | `selection.split_controlled` | median or tercile split at fixed control variable, rank-balanced |
| Stack | `stack.build_stacks` | per-radial-bin rest-frame stack, `keep_cube=True` for the bootstrap |
| Bootstrap | `measure.bootstrap_all` | one-pass galaxy resample: centroid, side ratio, per-pixel stack error |
| Profile fit | `analysis.bootstrap_fit_profile` | refit-per-draw `h1`, `r_c`, crossover radius, with 16/84 |
| Derived quantities | `measure.measure_*` | core / one-halo / two-halo luminosity, velocities, three-zone ratios |
| Core classification | `core.classify_lya` | emitter / absorber / null from a per-galaxy core S/N |
| Printer | `analysis.describe_subsample_properties` | one formatted summary per subsample, every argument optional |
| LaTeX table | `formatting.make_subsample_latex_table` | list of subsample dicts → paper table |
| Cache signature | `config.extract_signature()` | hash of every knob that affects the extracted product |

The runner's per-subsample output dict is built to be **exactly** the kwargs
`describe_subsample_properties` and `make_subsample_latex_table` already accept. No
adapter layer, no third schema.

---

## 3. Layer B — the recipe

A recipe is a plain dict (serializable, hashable, no code). It describes one split
and does nothing by itself.

```python
{
    "name":       "sigma_sfr_massctrl",     # unique key; the cache filename
    "column":     "SIGMA_SFR",              # split-catalog column
    "mode":       "half",                   # half | trend | bounds | groups  (Sec 3.1)
    "n_bins":     None,                     # trend mode only; default 4
    "bounds":     None,                     # bounds mode only; value edges
    "groups":     None,                     # groups mode only; categories
    "control":    "MASS_50",                # None, or a column held fixed
    "n_control_bins": 4,
    "base":       "CLEAN",                  # eligibility mask (Sec 3.2)
    "require":    ["GALFIT_FLAG_F160W == 0"],   # extra per-recipe cuts
    "needs_core": False,                    # hard prerequisite (Sec 3.3)
    "min_n":      80,                       # WARN below this; never refuses (Sec 3.4)
    "per_field":  False,                    # split within each field, then pool
    "labels":     ("low ΣSFR", "high ΣSFR"),
    "predicted":  "high ΣSFR → larger h1, more redshifted core centroid",
    "confirmatory": True,                   # Sec 7.1
    "doc":        "Split #2. Feedback-driven outflows open escape channels.",
}
```

Only `name`, `column`, and `mode` are required; everything else defaults.

### 3.1 Three cut modes, plus categorical

A 50/50 split is not a different *kind* of thing from a 4-bin split — it is a
percentile split with `n_bins=2`, chosen because equal halves maximize the S/N floor
across the two stacks. So percentile-based cutting is **one** mechanism with two
presets, and value-based cutting is the genuinely different one.

| `mode` | Meaning | When |
|---|---|---|
| `half` | percentile split at 50 — two equal-population sides | default. Maximizes the minimum S/N of the two stacks; the right choice whenever the question is "does it differ" and you have no reason to prefer particular values |
| `trend` | `n_bins` equal-population ordered bins, **default 4** | when the question is "does it vary monotonically." `n_bins=3` for a thin sample, 4–5 when S/N allows. Triggers the trend statistics in §5.3 |
| `bounds` | explicit value edges, in the column's own units | when the science lives at particular *values*, not particular percentiles |
| `groups` | categorical | AGN vs non-AGN; Lyα emitter / absorber / null |

**Why `bounds` is not a convenience but a requirement.** Percentile cuts are
sample-relative: the boundary moves when the eligible sample changes, so the "high-mass
half" of the main run and of an intersection rerun are different galaxies. For most
splits that is acceptable. For halo mass it is not — virial shock heating, the peak of
halo-mass efficiency, and the cold/hot accretion transition all happen at *specific
masses*, and a median cut at whatever value your sample happens to have will straddle
them rather than bracket them. `bounds=[8.5, 9.5, 10.5, 11.5]` centred on log M* ≈ 9,
10, 11 puts the bins where the physics changes and makes the result directly comparable
to a theory curve. Cost: bins are unequal in population, so their S/N differs and the
extreme bins may be thin — which `preview` reports and §3.4 warns about rather than
blocking.

Both percentile modes record the **realized cut value** in the result row, always, so a
percentile cut can be reported in physical units and compared across reruns.

### 3.2 Eligibility (`base`)

Resolved against the split catalog in this order:

1. the named base mask (below);
2. `CORE_COVERED` — added automatically iff the recipe sets `needs_core=True` or
   `column` starts with `CORE_`;
3. every expression in `require`;
4. finite `column` (and finite `control`, if set).

| `base` | N | Notes |
|---|---|---|
| `CLEAN` | 443 | **the default.** `_report_cut_columns` verifies `CLEAN == KEEP & ~AGN_FLAG` exactly — so `CLEAN` already *is* the clean non-AGN sample |
| `KEEP` | 510 | clean + AGN. Required by the AGN-vs-galaxy split, which needs the AGN back |
| `ALL` | — | no cut; diagnostic use only |

Default `CLEAN`, because the first thing a referee reaches for is AGN contamination and
the cheapest answer is "they were never in the sample." If the AGN split later shows the
two populations are indistinguishable in the halo, promoting the default to `KEEP` buys
~15% more galaxies in every split — which is worth having, but only *after* that result
exists, not on the assumption of it.

Roadmap §3.3b: catalog splits and core splits run on **different samples**, so a
cross-split comparison is confounded by sample composition rather than physics. The
runner therefore always records the realized galaxy ID list per side, and
`splits.intersect()` (§8) re-runs any set of recipes on their common intersection when
they are to be compared to each other.

### 3.3 Core dependence is per-recipe eligibility, not sequencing

The core extraction is **already done and already joined** — the split catalog carries
`CORE_COVERED`, `CORE_DETECTED`, `CORE_LABEL`, `CORE_V_OFFSET`, `CORE_SIGMA`,
`CORE_EW_REST`, `CORE_FESC`, `CORE_FLUX_MF`, `CORE_SN_MF` and their errors. So this is
not a build-order question. It is an eligibility question, and the important part is
what it must **not** do.

**A catalog split must not inherit a core cut.** A galaxy with a good mass and no
detected Lyα core belongs in the mass split. Dropping it would shrink the sample for no
reason and, worse, would silently make the mass split run on the Lyα-detected
subpopulation — a selection on Lyα brightness imposed on a test that has nothing to do
with it. Hence the strict rule in §3.2: `CORE_COVERED` is added **only** when the recipe
asks for it.

Three tiers of core dependence, and recipes must state which they need:

| Tier | Requires | Splits |
|---|---|---|
| none | nothing | mass, SFR, ΣSFR, size, inclination, environment, z, dust, AGN |
| covered | `CORE_COVERED` | anything reported per-subsample about the core region |
| detected | finite split column **and** `CORE_DETECTED` where the quantity is only defined for detections | `Δv_Lyα`, `FWHM_Lyα`, EW, `f_esc` |

The distinction between *covered* and *detected* matters: `CORE_COVERED` is a footprint
fact, `CORE_DETECTED` is a S/N fact, and conditioning on the second is conditioning on
Lyα brightness. A `CORE_LABEL` split (emitter / absorber / null) deliberately requires
only **covered** — the null and absorber classes are the point of that split and
requiring detection would delete them.

`needs_core` on a recipe takes `None | "covered" | "detected"`. If the split catalog was
built with `core=None`, a recipe needing either raises at resolution time naming the
missing column, rather than silently producing an empty sample.

### 3.4 `min_n` warns, never refuses

Below `min_n` (default 80, adjustable per recipe and globally) the runner prints a
prominent warning naming the recipe, the side, and the realized N — and then runs it.
Nothing is skipped, nothing is silently dropped.

The reasoning: whether a thin bin is usable depends on where the signal is, and the
person running it knows that better than a threshold does. A 60-galaxy bin is useless
for the two-halo term and perfectly fine for a core-region measurement. The power flag
(§7.3) already travels with each quantity in the result row, which is the mechanism
that actually prevents an underpowered number from reaching the paper. `min_n` is a
tripwire for attention, not a gate.

---

## 4. Layer B2 — the recipe library

`splits.LIBRARY` is an `OrderedDict{name: recipe}` holding the splits from
`../docs/subsample_splits.md`, written down once. Each entry carries its column, mode,
control, eligibility, `predicted` string, and a `doc` pointing back at the numbered
entry in that file.

Notes on specific entries, from what the split catalog actually contains:

- **`mass_bounds`** (#1) uses `mode="bounds"`, edges set **manually** — placeholder
  `[0, 9.4, 10.2, MAX]` until Austin sets them. `mass_half` is kept alongside it as the
  S/N-optimal companion. These are the same physical question asked two ways and should
  be reported as one row pair, not two independent findings.
- **`agn`** (#13a) is `mode="groups"` on `AGN_FLAG`, `base="KEEP"` — the only library
  entry that does not default to `CLEAN`.
- **`lya_class`** (#13b) is `mode="groups"` on `CORE_LABEL`, `needs_core="covered"`
  (not `"detected"` — see §3.3).
- **`size`** (#11) requires a clean GALFIT flag. `RE_KPC` tails to 10–15 kpc for a
  handful of galaxies; at z~2 that is not a plausible half-light radius and is almost
  certainly a blended or failed fit. `require: ["GALFIT_FLAG_F160W == 0"]`, plus a cut
  on `RE_ERR_F160W`.
- **`inclination`** (#8) splits on **`INCL_DEG`**, already in the catalog (alongside
  `BA`, `Q_BEST`, `Q_F160W`, `PA_F160W`). Same GALFIT-flag requirement as `size`. This
  is the most nearly mass-independent axis in the library (§7.2) and the cheapest real
  addition to the program.
- **`environment`** (#6) splits on `DELTA` with `per_field=True`. Chartab overdensity is
  normalized within its own field; pooling COSMOS and EGS before splitting mixes two
  normalizations. Consistent with the existing rule that fields are extracted and
  diagnosed separately.
- **`fwhm_lya`** (#9 variant) splits on `CORE_SIGMA`, `needs_core="detected"`. It is a
  decomposition of the Lyα profile rather than an external variable — see §7.5 for what
  that does and does not imply.
- **`delta_v_lya`** (#3) splits on `CORE_V_OFFSET`, `needs_core="detected"`. §7.5.
- **`fwhm_nebular`** — the catalog carries per-line nebular widths (`HA_FWHM`,
  `OIII5008_FWHM`, `HB_FWHM`, …) with errors, S/N and flags. These are a
  dynamical-mass proxy, so they sit in family 1 (§7.2) rather than being an independent
  axis — but they are measured entirely independently of Lyα, so they carry none of
  §7.5's concerns. Worth having as the clean counterpart to `fwhm_lya`. Needs an S/N
  and flag cut per line, and a decision on which line (Hα where available, [OIII] 5008
  otherwise) before it enters the library.
- **`delta_ms`** (#1/#2 variant) splits on the residual from the star-forming main
  sequence. **Deferred** — the calibration choice is an open decision (§4.1); the recipe
  is specified but not built in the first pass.
- **`ssfr`** — see §7.2. At fixed mass, sSFR **is** SFR. The library carries `ssfr` only
  as an *uncontrolled* recipe; the controlled version is `sfr_massctrl`, and
  `ssfr_massctrl` deliberately does not exist. Making the redundancy explicit in the
  library is cheaper than rediscovering it in the results table.
- **`lya_scale`** (#14) is **not** in the library as a per-galaxy split — roadmap §3.3
  and `subsample_splits.md` both flag it as circular. It appears instead as a `trend`
  recipe: sort by a cleanly measured property, `n_bins=5`, fit `h1` per group.

Recipes are ordinary dicts, so the notebook can build one from scratch or amend a
library entry inline:

```python
rec = dict(splits.LIBRARY["mass_half"], mode="trend", n_bins=3, name="mass_trend3")
```

### 4.1 `DELTA_MS` — specified, deferred

**PINNED 2026-07-31.** The calibration decision is open and is being taken later. The
recipe and the derived column are specified here so the design is settled; neither is
built in the first pass, and the first pass uses `control="MASS_50"` wherever
`DELTA_MS` would have been used.

`DELTA_MS` = `SFR_50 − SFR_MS(MASS_50, z)`, the log offset from the star-forming main
sequence, using a **literature** calibration (Speagle+2014 or Whitaker+2014), not a
self-fit — so the values are comparable to published ΔMS. Belongs in
`multicat.SPLIT_DERIVED` with the other derived columns, not in `splits.py`. The chosen
calibration goes in the entry's `doc` and in every result row's provenance.

Why this rather than `control="MASS_50"`: matched-bin control makes the two sides have
the same *mass distribution*, which is close to orthogonality but not the same thing —
the residual still correlates with everything else in the mass family, and the matching
costs sample. `DELTA_MS` is orthogonal to mass by construction, is a single number per
galaxy, and is a physically meaningful quantity ("above/below the main sequence") that
maps onto the literature. Both approaches stay available; `DELTA_MS` is the stronger
one where a main-sequence calibration is defensible.

Caveat to state in the paper: `DELTA_MS` inherits the calibration's own mass and
redshift dependence, so a systematic error in the main-sequence slope leaks into the
residual as a residual mass trend. Cheap check — regress `DELTA_MS` on `MASS_50` in the
sample and confirm the slope is consistent with zero. If it is not, the calibration does
not describe this sample and that fact is itself worth reporting.

**No new carried columns are needed.** `INCL_DEG`, `BA`, `Q_BEST`, `Q_F160W`,
`GALFIT_FLAG_F160W`, `RE_ERR_F160W`, `DELTA`, and the full `CORE_*` block are all
already on the catalog. The earlier claim in v1 that the axis ratio was discarded was
wrong — `RE_CIRC_BEST` is circularized, but `Q_BEST` and `INCL_DEG` are carried
alongside it.

---

## 5. Layer C — the runner

### 5.1 The per-recipe sequence

```
recipe
  → resolve eligibility mask against the split catalog       (Sec 3.2, 3.3)
  → cut into sides/bins                                       (Sec 5.2)
  → warn on any side below min_n                              (Sec 3.4)
  → for each side/bin:
        slice product
        stack.build_stacks(keep_cube=True)
        measure.bootstrap_all(seed=<shared>)                  (Sec 5.4)
        analysis.bootstrap_fit_profile
        measure.measure_* derived quantities
        power-flag each quantity                              (Sec 7.3)
  → assemble one result row per side + one comparison record
  → write to cache, keyed on (recipe, extraction, measurement) (Sec 6)
```

### 5.2 Cut

| Case | Delegates to |
|---|---|
| `half` / `trend`, `control=None` | `multicat.split_product_by(percentiles=...)` |
| `bounds`, `control=None` | `multicat.split_product_by(edges=...)` |
| `groups` | `multicat.split_product_by(groups=...)` |
| `half` / `trend`, `control` set | `selection.split_controlled` → masks → slice |

**One code path per case, all already written and tested.** The runner supplies
arguments; it does not reimplement cutting. `split_controlled` currently offers
`median` and `tercile` only, so a controlled `trend` with `n_bins ≥ 4` needs a small
generalization there (rank-within-control-bin into `n_bins` groups) — the one piece of
new cutting logic in this spec, and it belongs in `selection.py` next to its sibling,
not in the runner.

`per_field=True` runs the cut separately within each field and pools the sides
afterwards.

### 5.3 Trend mode

`mode="trend"` produces `n_bins` ordered bins (default 4). Each bin is stacked and
measured exactly as a side would be; the comparison record then carries, per derived
quantity:

- the value in each ordered bin, with bootstrap errors;
- a **Spearman rank correlation** between bin index and quantity value, with a p-value
  from permuting bin labels;
- a straight-line slope in the split variable's own units, using the realized median of
  the split variable in each bin as the abscissa.

Why this matters: a monotonic trend across four bins is **one test with an ordering
constraint**, where a binary split is one test with no constraint. A spurious binary
difference needs one bin pair to fluctuate; a spurious monotonic trend needs all four to
fluctuate in rank order — 1 in 12 under the null for four bins, 1 in 60 for five,
versus 1 in 2 for a binary split. Per-bin S/N is lower (~110/bin at `n_bins=4`, ~88 at
5), so trend mode is for **inner/core quantities only** (§7.3). Binary remains the
default for anything reaching past the one-halo term.

### 5.4 Shared bootstrap draws

Roadmap §3.4: generate the galaxy-resampling index array **once per base product** and
reuse it across every recipe (same galaxy pool). Two consequences, one practical and
one statistical:

- **Practical.** Removes a per-split RNG setup and makes reruns byte-reproducible.
- **Statistical.** Splits become correlated in a *known* way rather than an unknown one
  — the same draws underlie every split, so a galaxy that happens to be upweighted is
  upweighted everywhere. That is what makes the empirical overlap numbers in §7.2
  interpretable as a correlation structure rather than noise.

Implementation: `seed` fixed per base product and stored in the cache key. If the base
product changes, the seed's meaning changes, and the extraction signature already
invalidates the cache.

### 5.5 Preview mode

`splits.preview()` runs **only** steps 1–3 — eligibility, cut, warnings. No stacking,
no bootstrap. Seconds for the whole library. Reports per recipe:

- N per side/bin, and the realized cut value(s) in physical units;
- the separation in the split variable between the extreme sides;
- for controlled recipes, the **residual offset in the control variable** between sides
  (want ≈ 0 — `split_controlled` already prints this);
- warnings: `N < min_n`, side imbalance > 60/40, control offset above tolerance,
  eligibility dropping more than half the base sample, `needs_core` unsatisfiable.

This is how you decide which splits are worth ~20 minutes each. It should be the first
thing run after any change to the split catalog.

---

## 6. Caching and resume

**One file per side**, not per batch — an interrupted batch resumes.

Cache key = hash of:

1. the recipe dict (canonicalized, sorted keys);
2. `config.extract_signature()` — the Stage-1/2 signature;
3. a measurement-parameter hash (bootstrap `nboot`/`seed`/`stack_method`, fit `model`/
   `gamma_fixed`/`r_c_fixed`/`fit_skip_inner`, `r_edges`);
4. the split catalog's own provenance (`splitcat.meta['DERIVED']`, `NGAL`, `IDCOL`).

**Why item 2 is non-negotiable.** The SMHM default changed 2026-07-27 and the PSF area
convention changed 2026-07-22. A results cache that does not invalidate on the
extraction signature will silently serve numbers from two different pipelines **in the
same table** — and the failure is invisible, because both numbers are individually
plausible. This is the single highest-severity item in the spec.

Cached payload per side: the result row, the realized galaxy ID list, the realized cut
value, the seed, and every hash input in plain text. `splits.run(..., force=True)`
recomputes; default skips anything already cached.

Layout: `<config.output_dir>/splits/<extract_signature>/<recipe_name>__<side>.json`
(plus a `.npz` sidecar for the bootstrap arrays, which are too large for JSON).

---

## 7. Statistical guardrails

### 7.1 Pre-specification

Every recipe carries `predicted` (a directional statement) and `confirmatory` (bool).
Both are **recorded, not enforced** — nothing is gated. The point is that at writing
time the distinction between "we predicted this and found it" and "we found this" is a
fact on record rather than a memory.

Cost: one string per recipe. This is the cheapest insurance in the document.

### 7.2 Independence: what it is, what it isn't, and the family taxonomy

**Independence is a property of the variables, not of the sample size.** Two variables
uncorrelated in the population stay uncorrelated at N = 500 or N = 5000. Sample size
buys *power*, not orthogonality. So the honest statement is not "500 galaxies can't
have independent splits" — it is "the *galaxy scaling relations* make one particular
family of splits redundant, and that family happens to be the one that is easiest to
measure."

The splits fall into three families with very different behaviour:

**Family 1 — the mass sequence.** Mass, SFR, sSFR, ΣSFR, size, dust, `DELTA_MS`. These
are manifestations of the same underlying scaling relations. Pairwise |ρ| typically
0.5–0.9, and two pairs are *exactly* degenerate (§7.2a). Effective dimensionality of
the whole family: **~2** (a mass axis and a star-formation-efficiency axis). No sample
size changes this — it is astrophysics, not statistics. Running six splits here yields
about two independent results.

**Family 2 — externally-set properties.** Inclination, environment/overdensity,
redshift bin, field. These are **not structurally tied to galaxy mass**, and inclination
is the cleanest case in the whole program: viewing angle is set by random orientation on
the sky and is independent of every intrinsic property by construction. Overdensity has
a weak link to mass through halo bias (massive galaxies mildly prefer dense regions) —
weak enough to treat as a separate axis, strong enough to report the residual mass
offset. Redshift correlates with mass only through the magnitude cut, which is a
selection effect you can measure. Effective dimensionality: **~3, essentially the full
count.**

**Family 3 — Lyα-internal properties.** `Δv_Lyα`, `FWHM_Lyα`, peak separation, EW,
`L_Lyα`, `f_esc`. Strongly correlated *with each other* (all driven by HI column
density, kinematics, and dust), only moderately tied to mass. Effective dimensionality:
**~2**. But correlation is not their main problem — circularity is (§7.5).

**Total: ~7 effective independent axes from ~15 splits, not 15 and not 3.** That is
enough for a real paper. It is not enough for a 20-parameter decomposition, and the
program should be framed and written as the former.

**What this means for design, and it is the actionable part:** pick splits from
*different families* rather than piling up family 1 because its columns are the easiest
to get. Mass + inclination + overdensity + `Δv_Lyα` is four nearly-independent axes.
Mass + SFR + sSFR + ΣSFR is one axis reported four times.

#### 7.2a Two exact degeneracies to name rather than hide

- **sSFR at fixed mass ≡ SFR at fixed mass.** `sSFR = SFR − M` in log; hold `M` fixed
  and these are the same variable, correlation exactly 1.0. Two controlled recipes would
  produce two rows of the same measurement.
- **ΣSFR at fixed mass and fixed size ≡ SFR at fixed mass.** `ΣSFR = SFR − log(2πR_e²)`.
  ΣSFR is a distinct axis *only* if size is allowed to vary — which means the ΣSFR split
  is partly a size split and must be reported that way.

#### 7.2b The diagnostic

`splits.independence(splitcat, recipes)` reports:

1. **Correlation matrix** of the split *variables* over the eligible sample (Spearman,
   NaN-pairwise), grouped by family.
2. **Overlap matrix** of the resulting *sides* — for each pair of recipes, the fraction
   of galaxies assigned to the same side. The directly interpretable number: two splits
   with 85% overlap are one split done twice.
3. **M_eff**, the effective number of independent tests, from the eigenvalue spectrum
   of the correlation matrix (Li & Ji 2005). One number, printable in the paper.

The family taxonomy above is a prediction; the overlap matrix is the measurement. The
acceptance test is that §7.2a's degeneracies come back as overlap ≈ 1.0 and that
inclination comes back near 0.5 (i.e. no overlap beyond chance) against everything else.

#### 7.2c The real N ≈ 500 limit is interactions, not independence

You can test ~7 axes one at a time with adequate core-region power. What you **cannot**
do is test them two at a time: a mass × inclination split is four cells of ~110, and a
three-way is ~55. So questions of the form "does the inclination dependence differ
between high- and low-mass galaxies" — which is where a lot of the radiative-transfer
literature actually lives — are out of reach with this sample and should be named as
future work rather than attempted and reported as null. That is the honest statement of
the finite-sample limit, and it is a much narrower claim than "the splits aren't
independent."

### 7.3 Power flags

Roadmap §3.3c, with N ≈ 452 after AGN cuts and full-stack outer-bin bootstrap S/N of
6 / 5 / 2.8:

| Split | Outer-3 bins S/N |
|---|---|
| Full stack | 6.0, 5.0, **2.8** |
| Two-way (N ≈ 226) | 4.2, 3.5, **2.0** |
| Three-way (N ≈ 151) | 3.5, 2.9, **1.6** |

**No split supports a >2σ outermost-bin claim with current data.** The runner does not
refuse to compute two-halo quantities; it computes them and marks them
`power="insufficient"` in the result row, so the flag travels with the number into the
table rather than living in someone's head. Inner / one-halo / core quantities (`h1`,
core luminosity, core velocity, crossover radius) are `power="ok"` for binary splits.

Trend mode and `bounds` mode with thin extreme bins are flagged `power="insufficient"`
for **everything past the one-halo term**, without exception.

### 7.4 What the runner reports instead of p-values

Effect sizes with bootstrap 16/84 intervals, per quantity, per side — plus the
difference and its interval. A p-value is reported for trend mode (the rank permutation
test) because a trend has a natural null; binary comparisons report the difference and
its interval and let the reader threshold it.

### 7.5 Circularity — what applies where, and what does not

**Two mechanisms, and they have different reach. Getting this wrong in either direction
costs something: over-applying it kills a good split, under-applying it lets an artifact
through.**

**(a) Structural circularity — same measurement, same radii.** Splitting on a fitted
quantity and then reporting that same fitted quantity from the same radial bins is
circular, full stop. This is split #14 (per-galaxy Lyα scale length → measure scale
length), and it is why #14 is excluded from the library as a per-galaxy split.
`subsample_splits.md` #3 makes the weaker, correct version of the same point about the
Δv self-split: it is a *decomposition* of the centroid-vs-radius result, not an
independent test of it. Decompositions are worth running; they must be labelled as
such, and `confirmatory` is the wrong field for them. `f_esc` is a third case —
∝ L_Lyα/L_Hα,corr, therefore partly circular with an `L_Lyα` split specifically, and
the recipe must control for `L_Hα` and say so.

**(b) Noise-driven regression to the mean — narrower than it first appears.** Splitting
on a low-S/N per-galaxy quantity does preferentially select galaxies whose *noise*
scattered one way. But that only propagates into the stacked result **if the split
measurement and the stacked measurement share a noise source.**

> ⚠️ **v1 of this spec claimed `Δv_Lyα` and `FWHM_Lyα` inherit #14's regression-to-the-mean
> problem "wholesale." That was wrong** (Austin, 2026-07-31), and the reason is worth
> stating because it also tells us what the split is actually testing.
>
> The core Δv is measured from the innermost extraction; the halo centroid is measured
> from outer radial bins, different fibers, different sky. **The photon noise is
> essentially independent between them**, so there is no photon-noise pathway from a
> core split to a large-radius result. And the physical prior points the same way: the
> literature treats large-scale halo/IGM kinematics as substantially **disjoint** from
> the galaxy's internal Lyα radiative transfer. A nearby-region correlation is expected;
> a large-radius one is not.
>
> That makes the null well-motivated, which makes the split *more* valuable rather than
> less. If a core-Δv split does move the large-radius halo centroid, there are exactly
> three readings — a shared systematic (below), a genuinely interesting physical result,
> or an error — and the first is testable, which is what makes the other two claimable.

**The one channel that does survive, and it is specific.** Any per-galaxy error that is
**common-mode across radius** couples the two measurements even though the photon noise
does not. The dominant such term is the **systemic redshift error**: `Δv_Lyα` is
measured relative to the MOSDEF systemic velocity, and so is the halo centroid. A
per-galaxy error in the systemic z enters **both with the same sign**. Split on `Δv_Lyα`
and you are partly splitting on systemic-redshift error — which the halo centroid then
reproduces, at all radii, for free. Per-galaxy wavelength-solution and sky-subtraction
residuals act the same way, more weakly.

This is not hypothetical given that the MOSDEF↔VIRUS velocity zero point is the current
top-priority open systematic (`wavelength-zero-point.md`). It is also a *rigid* offset
per galaxy, so it produces a flat velocity shift with radius, not a gradient — the same
bounding argument that protects the headline result applies here.

**Tests, specific to this channel rather than generic:**

1. **Redshift-quality stratification.** Recompute the split restricted to high `Z_QUAL`
   / high `BEST_LINE_SN` galaxies, where the systemic-z error is smallest. A real effect
   is unchanged or stronger; a shared-z-error artifact weakens.
2. **Radial signature.** A shared-z artifact is flat in radius by construction. Anything
   with radial structure is not it.
3. **Independent-variable cross-check.** Does the same halo difference appear when
   splitting on a family-2 variable that correlates with `Δv_Lyα`? If nothing external
   reproduces it, that is evidence it lives in the shared term.
4. **S/N threshold sweep** on the split variable — kept from v1, but as a general
   robustness check rather than as the primary defence.

Each family-3 recipe records which of these it has passed. §7.2 is about how many
independent things you have tested; this section is about whether any single one of them
means what it appears to.

---

## 8. Public API

```python
from utils_lya_halo import splits

# --- inspect -----------------------------------------------------------
splits.show(splits.LIBRARY)                    # names, columns, mode, predictions, cost
splits.preview(cfg, prod, splits.LIBRARY)      # seconds: N/side, cut values, warnings
splits.independence(splitcat, splits.LIBRARY)  # correlation + overlap + M_eff, by family

# --- run ---------------------------------------------------------------
res = splits.run(cfg, prod, "mass_half")                      # one
res = splits.run(cfg, prod, ["mass_half", "environment"])     # several
res = splits.run(cfg, prod, splits.LIBRARY)                   # all, resumable
res = splits.run(cfg, prod, rec, force=True)                  # ignore cache
res = splits.run(cfg, prod, rec, min_n=120)                   # override the warn threshold

# --- compare / report --------------------------------------------------
res = splits.intersect(cfg, prod, ["mass_half", "lya_lum"])   # common-sample rerun
splits.describe(res["mass_half"])              # -> analysis.describe_subsample_properties
splits.table(res, columns=[...])               # -> formatting.make_subsample_latex_table
splits.load(cfg)                               # rehydrate everything cached
```

`run` accepts a name, a recipe dict, or any list/dict of either — so "all of them" and
"just this one" are the same call. Returns `OrderedDict{recipe_name: result}`.

A result is:

```python
{
    "recipe":  {...},                  # the recipe as run
    "sides":   OrderedDict{label: {    # kwargs for describe_subsample_properties
        "label", "n_gal", "base_sample", "fit_result", "boundary_info",
        "vel", "core_lum", "onehalo_lum", "twohalo_lum", "ratios", "boot_fit",
        "ids", "cut_value", "power", "warnings",
    }},
    "compare": {...},                  # per-quantity differences (+ trend stats)
    "provenance": {...},               # every hash input, in plain text
}
```

---

## 9. Module layout

**Layer A stays in `multicat.py`.** `build_split_catalog` and friends are catalog
joining and belong with the other catalog machinery; moving them would break the guide
entries and every existing importer for no gain. The two new derived columns (§4.1) go
into `SPLIT_DERIVED` there.

**The controlled `trend` generalization goes in `selection.py`**, next to
`split_controlled`, which it extends.

**Layers B and C go in a new `splits.py`** — the recipe library, the resolver, the
runner, preview, the independence diagnostic, the cache layer, and table assembly.
Estimated ~900–1200 lines, which earns its own file; `multicat.py` is already 1488 lines
and adding a runner would push it past 2500 while mixing catalog joining with
orchestration.

Register a new `guide.py` section (`"splits"`) and every new public function;
`check_guide()` must stay clean. Registration is not optional — `fitting.py` and
`uv_profile.py` sat unregistered for months with zero audit coverage.

---

## 10. Build order and success criteria

| Step | Deliverable | Done when |
|---|---|---|
| 1 | Recipe schema + resolver + `show` | a recipe round-trips to a mask and back to a hash; a `needs_core=None` recipe provably does **not** inherit a core cut (§3.3) |
| 2 | `LIBRARY` | every entry names a column that exists, or is explicitly marked blocked-on-data |
| 3 | `preview` | whole library previews in < 30 s; N/side matches a hand-count for two recipes |
| 4 | `independence` | reproduces §7.2a as overlap ≈ 1.0 and inclination as ≈ 0.5 vs everything |
| 5 | Cache layer | changing `extract_signature` invalidates; killing a batch mid-run resumes without recompute |
| 6 | `run` — `half` / `bounds` / `groups` | one recipe reproduces a hand-run split's `h1` and centroid **exactly** |
| 7 | `run` — `trend`, incl. controlled | a 4-bin recipe returns per-bin values + Spearman + permutation p |
| 8 | `table` / `describe` | LaTeX table builds from cached results with no adapter code |

**The acceptance test for step 6 is the important one:** take a split already run by
hand in a notebook, run it through the runner, and require the numbers to match to the
last digit. If they do not, the runner has introduced a difference somewhere in the
composition, and every subsequent number is suspect.

---

## 11. Open questions

1. **`DELTA_MS` calibration — PINNED**, to be decided later (§4.1). Literature-based
   (Speagle+2014 or Whitaker+2014), not self-fit. Not built in the first pass.
2. **The `bounds` edges for the mass split** are set manually by Austin. Placeholder
   `[0, 9.4, 10.2, MAX]` until then.
3. **Trend-mode default bin count.** 4 is the default. Whether 5 is usable for core
   quantities is a `preview` question on real data.
4. **`fwhm_nebular` line choice and quality cuts** — Hα where available, [OIII] 5008
   otherwise, with what S/N and `*_FLAG` thresholds.
5. **GALFIT flag cut for `inclination`** — same as `size`, or looser? Axis ratio is
   better constrained than `R_e` for marginal fits, so the same cut may be needlessly
   strict.
6. **Which §7.5 tests each family-3 recipe records.** Test 1 (redshift-quality
   stratification) is the cheapest and most directly targeted at the one surviving
   channel; it should probably be mandatory rather than optional.
