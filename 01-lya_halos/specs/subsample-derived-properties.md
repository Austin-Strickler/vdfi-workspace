# Subsample-derived properties — fitting-based comparison tables

Companion to `halo-flux-fitting.md` (the fitting model, Parts 1–2) and
`subsample_splits.md` (the candidate splits this applies to). Those specs cover
fitting one profile and enumerating which splits to run; this one covers turning
every subsample-comparison PLOT into a row of derived NUMBERS — core/halo
luminosity, core/halo velocity, and their ratios — collected into one reference
table, not just a figure per split. Design finalized 2026-07-17 after working
through the PSF/boundary tradeoffs in detail (see decision log at the bottom);
ready to implement.

**Status: Parts 1–3 (core/halo, two-way split) implemented in `measure.py`
(`measure_core_halo_velocity`, `measure_halo_luminosity`,
`measure_psf_corrected_core_luminosity`, `measure_outer_properties`) and
`fitting.py` (`find_core_halo_boundary`) as of 2026-07-17. Part 3b (three-way
core/halo/2-halo split) implemented 2026-07-18 as
`measure_onehalo_luminosity` + `measure_three_zone_ratios` -- ADDITIVE only;
see Part 3b's "as implemented" note for the one naming deviation from the
design below. Part 4 (fitting overlay in the two-sample comparison plot) is
also implemented -- `analysis.plot_flux_profile_two` carries `fit`/
`fit_model`/`fit_method`/`fit_skip_inner`/`gamma_fixed` kwargs, defaulting to
`fit_model="expcore"` now that `halo-flux-fitting.md` Part 2 has shipped (see
Part 4 below and Open Question 3, now resolved). Part 5 (the
`describe_subsample_properties` printer) implemented 2026-07-18 -- see
Part 5 below.**

## Goal

Today, `1_Measure_Splits.ipynb` calls `analysis.plot_centroid_profile_two` /
`analysis.plot_flux_profile_two` to visually compare two subsamples' centroid-
vs-radius and flux-vs-radius profiles across however many splits are defined
(`subsample_splits.md` #1–11 for Paper 1, plus low-z vs high-z, plus the full
stack). That's the figure half — it works well. What's missing is the
measurement half: for each subsample, derive core luminosity, halo luminosity,
core velocity, halo velocity, and their ratios/differences, then emit those
numbers in a form that drops straight into a paper table instead of being
hand-copied off each figure.

**Framing, per Austin:** none of these derived numbers are meant to be
important as absolute physical quantities on their own — the actual payoff is
the COMPARISONS (low-z core luminosity vs. high-z core luminosity; a given
subsample's core vs. its own halo). That reframes the design goal: every
number needs to be computed IDENTICALLY across every subsample row (so
cross-row comparisons are apples-to-apples) and needs a standard, citable
methodology (so it isn't hand-wavy in front of a referee) — it does not need
to be a maximally "true" physical decomposition.

## Units — luminosity/kpc² (`L_kpc2`), not observed surface brightness — CONFIRMED

Keep `L_kpc2` (the current `config.flux_unit` default) for every comparison in
this spec, especially low-z vs. high-z: observed surface brightness folds in
redshift-dependent cosmological dimming that would confound a real physical
difference between the two z-bins with a cosmological one. No new config knob
needed — this is a recommendation on which existing one to use.

## Part 1 — the core/halo boundary: the fitted crossover radius

**Decided:** the halo boundary, for both the velocity split (Part 2) and the
luminosity split (Part 3), is the radius at which the fitted CORE term and the
fitted HALO term of the two-component flux fit (`halo-flux-fitting.md`) are
equal — generalizing "where does the core stop dominating and the halo take
over" directly from the model already fit to the data, per subsample.

- **Today (Part 1's two-exponential model):** solve `A1*exp(-r/h1) =
  A2*exp(-r/h2)` for `r` — a straightforward root-find/interpolation on the
  already-evaluated `r_fine` grid, no new fitting.
- **Once `halo-flux-fitting.md` Part 2 ships (exponential core + cored power
  law):** the same rule generalizes with no change in spirit — solve
  `A1*exp(-r/h1) = A2*(1+(r/r_c)^2)^(-gamma/2)` instead. Everything below is
  written against "the core term" / "the halo term" generically so it doesn't
  need rewriting when the model swaps.

**Why this and not a fixed radius:** direct precedent from the full-stack
result already in the Discussion draft — the fitted crossover there (≈76 kpc)
lands almost exactly on the independently-derived R_vir (≈75 kpc), and that
agreement (plus the centroid flattening at the same radius) is already one of
the paper's three Discussion pillars. Reusing "wherever the fit itself says
core hands off to halo" for every subsample is the same logic applied
consistently, not a new assumption.

**Accepted risk, stated plainly:** this boundary is NOT the same fixed radius
across every row — each subsample's own fit has its own crossover, so a
low-mass subsample's boundary and a high-mass subsample's boundary can differ.
That's a real tradeoff (no single "core radius" you can quote once for the
whole table) traded for a boundary that's actually justified by each
subsample's own data rather than picked externally.

**Fallback for a subsample whose two-component fit doesn't converge cleanly**
(a real risk — `halo-flux-fitting.md` already notes two-exponential fits on
~10 sparse bins are prone to local minima at FULL sample size; a half-sample
split is noisier still): fall back to the full-stack's own crossover radius
for that row, and flag the row (e.g. a boolean `boundary_from_own_fit` table
column) rather than silently presenting a degenerate subsample-specific number
as if it were trustworthy.

## Part 2 — velocity: core = innermost bin (unmodified), halo = combined outer bins

**Core velocity = the innermost radial bin's centroid, unmodified — not an
average over several inner bins.** The centroid genuinely evolves with radius
(redward near the center, decreasing, crossing systemic near the boundary) —
averaging several inner bins into one "core velocity" would smear exactly the
gradient that's physically interesting, the same way averaging inner FLUX bins
would smear PSF structure. The innermost bin is the smallest core region
actually measured, so it's used as-is.

No PSF correction needed here, unlike for flux: the PSF blurs a photon's
*spatial* position, not its *wavelength*, so a photon landing in the innermost
bin still carries an accurate velocity regardless of exactly which true radius
it scattered from. This is why velocity and flux get different treatment below
even though both use the same core/halo split.

**Halo velocity = inverse-variance-weighted (or biweight, matching the
pipeline's existing centroid-combine convention) average of the per-bin
centroids in every bin beyond the Part 1 boundary.** Combining several outer
bins is fine here — unlike the core side — because the profile is already
established to be comparatively flat past the boundary (Discussion pillar 1:
"crosses systemic near R_vir → mildly negative beyond"), so combining bins out
there doesn't erase real structure the way it would near the center.

**Core–halo velocity difference** = core velocity − halo velocity: the actual
inflow/outflow/systemic diagnostic Austin described — halo redshifted relative
to core (outflow-consistent), blueshifted (inflow-consistent), or ~0 relative
to core (systemic halo, core alone carries the kinematic signal).

**Errors — no new machinery needed.** Core velocity's uncertainty is just the
innermost bin's existing bootstrap 16/84 (already computed). Halo velocity's
uncertainty comes from combining the existing per-bin bootstrap centroid draws
across the outer bins (inverse-variance-weighted per draw), giving a proper
combined distribution — reuses the bootstrap draws already sitting in `boot`,
no re-stacking required. This directly gives the "combined error significance"
result for the full-stack row too, computed by the exact same procedure as
every subsample row.

## Part 3 — luminosity: PSF-corrected core (per-galaxy rescale + re-coadd), raw-summed halo

**Halo luminosity = raw summed flux across the bins beyond the Part 1
boundary, uncorrected.** No PSF correction needed: the PSF's angular support
shrinks fast at large radius (this is exactly why `ring_convolution_matrix`
needed the exact-support-window fix in `fitting.py`), so contamination out
there is a near-field problem that doesn't apply. Error bars: sum the existing
per-bin bootstrap draws across the same outer bins, take 16/84 of the
resulting distribution — again no re-stacking, reuses what's already computed.

**Core luminosity is the harder piece — a genuine point-source aperture
correction, applied PER GALAXY before combining, not as one blanket factor
slapped onto the final stacked number.** The reasoning and the procedure:

*Why per-galaxy, not one aggregate factor.* Each galaxy has its own kpc-per-
arcsec scaling (via its own z), so a PSF that's fixed in angular size (the
VIRUS/fiber PSF) corresponds to a different PSF width in kpc for every galaxy
— not by a huge amount, but enough that applying one "effective"/averaged
correction factor to the whole stacked innermost bin is a rough approximation,
not an intrinsic treatment. Scaling each galaxy individually and THEN
re-coadding accounts for the real per-galaxy differences instead of
approximating them away.

*Procedure, per subsample:*
1. **PSF model — a fixed literature Moffat, not a re-measured empirical
   curve.** Use `fitting.moffat_1d` with `beta=3`, `FWHM` in the range
   1.2″–1.4″ (fiducial: 1.3″, the midpoint) — the analytic form Lujan Niemeyer
   (2022) showed fits VIRUS stellar profiles well. Deliberate simplification:
   this correction only needs to be approximately right, since extended
   PSF-wing residuals shift the resulting core-luminosity number by only a
   fraction of a percent — not something that matters for a quantity whose
   whole point is RELATIVE comparison across subsamples, not an absolute
   physical value. No per-galaxy empirical PSF measurement or
   `uv_profile.py`-style effective-PSF-building machinery is needed here. The
   only per-galaxy step is the ordinary arcsec→kpc conversion of this one
   fixed FWHM via each galaxy's own z (`cosmo.angular_diameter_distance`, the
   same conversion already used throughout the pipeline for `bin_mode`
   handling) — not a second PSF-measurement pipeline.
2. Compute galaxy `i`'s encircled-energy fraction `EE_i` — the fraction of a
   point source's total flux that would land inside galaxy `i`'s innermost
   bin, using the step-1 Moffat curve converted to that galaxy's own kpc
   scale. **Worth using this same Lujan Niemeyer (β=3, 1.2″–1.4″) Moffat as
   the default PSF for `halo-flux-fitting.md`'s two-component PSF-aware fit
   too** (which currently defaults to a generic, less-specific
   `moffat_1d(fwhm=3.0, beta=3.0)` placeholder per `plot_flux_profile_fit`'s
   docstring) — one literature-grounded PSF convention used everywhere in
   this pipeline that fits a PSF, rather than a second, divergent one just
   for this correction. Correction factor `c_i = 1 / EE_i`.
3. Rescale galaxy `i`'s ENTIRE flux array by the single scalar `c_i` (not
   radius-dependent — the correction is only valid at the innermost bin's
   aperture; the rest of the rescaled array is intentionally not physically
   meaningful and is discarded in step 5).
4. Re-run the same combine method already used for the subsample's real stack
   (`config.galaxy_combine_methods`, e.g. biweight) across all of the
   subsample's now individually-rescaled galaxy arrays — i.e., an actual
   second stacking pass reusing `stack.build_stacks`'s existing machinery,
   fed rescaled inputs instead of raw ones.
5. **Core luminosity = this new stack's innermost-bin value only.** Every
   other radius in this rescaled re-coadd is an artifact of applying an
   innermost-bin-only correction across a whole array, and is not used or
   reported.

*Errors:* bootstrap this rescaled re-coadd the normal way (resample galaxies,
recombine, same machinery as `bootstrap_measurements`/`bootstrap_all`) for
16/84 error bars on core luminosity. **Important implementation detail:** use
the SAME bootstrap resampling draws (same galaxy-index draws / RNG seed) for
this rescaled-stack bootstrap as for the original (uncorrected) stack's
bootstrap, so the core/halo luminosity RATIO's error can be computed
per-draw (`ratio_b = core_b / halo_b`) rather than by naively combining two
independently-drawn error bars — the two quantities are not independent
(same galaxies, same noise realizations), and treating them as if they were
would understate or misstate the ratio's real uncertainty.

**Core/halo luminosity ratio** = core luminosity / halo luminosity, with the
per-draw error propagation above.

**Where this lives.** Unlike Part 2 (which only reuses already-computed
bootstrap draws), this is real new pipeline work — a second per-subsample
stacking + bootstrap pass, not just an analysis.py-level computation. Natural
home is alongside `stack.py`/`measure.py`'s existing combine/bootstrap
functions (e.g. a new `measure_psf_corrected_core_luminosity`-style function
that wraps `build_stacks`/`bootstrap_all` with a pre-scaling step injected
before the combine), not `analysis.py` — mirrors how `halo-flux-fitting.md`
Part 3 flagged its own extraction code as closer to `extract.py`'s job than
`fitting.py`'s.

## Part 3b — three-way split: core / halo / 2-halo (design settled 2026-07-18, implemented 2026-07-18)

**As implemented — one deviation from the design below.** Austin's
implementation instruction was explicit: don't touch the two already-shipped
measurements, just fit the new one in alongside them. So
`measure_halo_luminosity` was NOT renamed — it keeps its original name and
behavior completely unchanged. The new middle zone is
`measure_onehalo_luminosity` (not `measure_halo_luminosity`/"halo" as
originally proposed below) — "halo" was already taken, and "one-halo term" /
"two-halo term" is itself standard halo-model terminology for this exact
same-halo-vs-different-halo split, so it reads as precisely as the original
naming would have. A new `measure_three_zone_ratios(core_lum, onehalo_lum,
halo_lum)` combiner was added for the ratio numbers (`onehalo/core`,
`twohalo/onehalo`, `twohalo/total`) described below — pure arithmetic on
already-computed results, no new stacking/bootstrap. Everything else below
(the zone definitions, the empirical-zone-sum method, the expcore-boundary
recommendation, error alignment, edge cases) was implemented as designed.

**The problem this fixes.** Part 3's "halo luminosity" (everything beyond the
Part 1 crossover) is dominated by the fit's OUTER term — which
`halo-flux-fitting.md` Part 2 and `fitting.py`'s own docstrings describe as "a
stand-in for a clustering/two-halo-term outer profile," not the galaxy's own
CGM emission. Meanwhile `h1`'s term (the inner exponential) is what Austin
actually calls "the halo scale length" — the field's normal usage of "halo."
So as shipped, Part 3 quietly measures a **core + two-halo** split and calls
the second piece "halo," while the radial zone that's actually one-halo-term-
dominated (between the PSF-corrected core bin and the crossover) is never its
own reported number — it only ever feeds the fit.

**Rename (originally proposed here, SUPERSEDED — see "As implemented" note
above):** the original plan was `measure_halo_luminosity` →
`measure_two_halo_luminosity`, freeing up "halo" for the new middle zone.
Austin's implementation instruction overrode this: don't touch the existing
function at all. `measure_halo_luminosity` keeps its name; the new zone is
`measure_onehalo_luminosity` instead. The physical point stands regardless
of which name won — `measure_halo_luminosity`'s zone is two-halo/clustering-
dominated, not this galaxy's own CGM — just remember it going forward as "the
function called `measure_halo_luminosity` measures the two-halo term," not
literally as its own name suggests.

**Three zones, all keyed off quantities that already exist:**

- **Core** = bin `core_bin_index` (default 0), PSF-aperture-corrected.
  Unchanged from Part 3 — `measure_psf_corrected_core_luminosity`.
- **Halo** = bins with `core_bin_index < bin` and `r_mid ≤ boundary_radius`
  (from `find_core_halo_boundary`) — the newly-measured middle zone,
  one-halo/CGM-term-dominated.
- **2-halo** = bins with `r_mid > boundary_radius` — Part 3's existing
  "halo luminosity," renamed. Clustering-term-dominated.

**Measurement method for the halo (middle) zone — empirical zone-sum,
decided over the model-decomposed alternative.** Two ways to compute it were
weighed:

1. *Empirical zone-sum* (chosen): sum `total_flux_fid × own annulus area`
   over the zone's bins, exactly `measure_halo_luminosity`'s existing recipe
   applied to a different radius range. No new model dependence, no new
   stacking pass — bootstrap error falls straight out of `total_flux_all` the
   same way. **As implemented:** rather than refactoring
   `measure_halo_luminosity` into a shared primitive (which would touch the
   existing function, contrary to Austin's non-intrusive instruction), the
   recipe is duplicated once, standalone, in the new
   `measure_onehalo_luminosity` — a few lines of repetition traded for zero
   risk to the already-shipped function.
2. *Model-decomposed* (rejected as primary, kept as an optional diagnostic):
   integrate the fit's `A1*exp(-r/h1)` term alone over the zone, netting out
   the outer term's leakage. Physically cleaner attribution (near the
   boundary the two terms are by construction comparable, so option 1 mixes
   a bit of each into the "wrong" zone), but pins the number to the fit being
   right — and `h1`/`r_c`/`gamma` are already documented
   ([[lya-halos-bootstrap-fit-expcore]]) as seed- and `gamma_fixed`-sensitive
   on sparse bins. Adding a second layer of fit uncertainty on top of a
   boundary that's already a fitted (not measured) quantity was judged not
   worth it for a number whose whole point, per this spec's Framing section,
   is a clean cross-subsample comparison, not maximal physical purity.
   Worth computing later as a secondary "estimated 2-halo leakage fraction
   into the halo zone" diagnostic column, not as a correction to the primary
   number.

**Which model's boundary to use — expcore, not two-exponential, once
`halo-flux-fitting.md` Part 2 ships.** `find_core_halo_boundary` already
dispatches on either fit's `model` key, so both work mechanically. But only
the expcore model's outer term (`A2*(1+(r/r_c)^2)^(-gamma/2)`) has an actual
clustering/2-halo physical motivation in the docs — the two-exponential
model's outer term is just a second exponential that happens to land its
crossover near R_vir. A table column literally named "2-halo luminosity"
should be built on the fit that motivates calling it that.

**Errors and ratios — already aligned, no new bookkeeping.** `core_lum_all`
is already built replaying `boot`'s exact bootstrap draw sequence
(guarantee documented in `measure_psf_corrected_core_luminosity`).
`measure_onehalo_luminosity`, built from the same `total_flux_all`, inherits
that same per-draw alignment automatically. So `onehalo/core`,
`twohalo/onehalo`, and `twohalo/total` ratios (implemented in
`measure_three_zone_ratios`) all get correct per-draw error propagation
(`ratio_b = zone_a_all[b] / zone_b_all[b]`, then 16/84-percentile the ratio
distribution) essentially for free, the same
pattern `measure_core_halo_velocity`'s `diff_all` already uses.

**Verification.** `core_lum + onehalo_lum + halo_lum` (`total_lum_fid`,
returned directly by `measure_three_zone_ratios`) should reconcile against
the existing `flux_curve_of_growth` total, MINUS one expected, explicable
offset: because `core_lum` uses the PSF-rescaled re-coadd (always ≥ the raw
core bin), the reconciled three-zone total will be systematically *higher*
than the raw curve-of-growth total by exactly the aperture-correction's added
flux. Worth asserting this as an automated check
(`total_lum_fid - curve_of_growth_total ≈ core_lum_fid - core_lum_raw`)
rather than treating a mismatch as a bug the first time someone runs it. Not
yet implemented as an automated check (still open, see item 4 below).

**Edge cases to guard (same defensive pattern as Part 3's `n_outer < 1`
check):** `boundary_radius` falling inside the core bin itself (empty
one-halo zone — the fit says the core dominates all the way to the
crossover, which would itself be worth flagging rather than silently
returning zero, hence `measure_onehalo_luminosity` raises `ValueError`
instead) or beyond the outermost bin edge (empty two-halo zone — no measured
signal is two-halo-dominated for this subsample; `measure_halo_luminosity`
already raised for this case).

**Table columns this adds** to Part 5's `describe_subsample_properties`
(now implemented — see Part 5): `onehalo_lum` (+16/84, from
`measure_onehalo_luminosity`), `halo_lum` (+16/84, existing/unchanged,
physically the two-halo term), `onehalo/core`, `twohalo/onehalo`,
`twohalo/total` (all three from `measure_three_zone_ratios`) — the last one
is the likely headline number for a Discussion paragraph ("X% of stacked
Lyα luminosity at low-z vs. high-z arises from the 2-halo/clustering term
beyond the fitted boundary, vs. Y% at low-z").

## Part 4 — optional fitting overlay in the two-sample comparison plots

Independent of Parts 1–3 (this is about the FIGURES, not the table): extend
`plot_centroid_profile_two` and `plot_flux_profile_two` in `analysis.py` with
a `fit: bool = False` argument (off by default). When `fit=True`,
`plot_flux_profile_two` independently fits each subsample's flux profile using
the same machinery as the existing single-sample `plot_flux_profile_fit`
(PSF-aware by default), overlays each subsample's fitted curve and
core/halo-term components on the comparison plot, and returns the two
`fit_result` dicts — which are also exactly what Part 1's boundary-finding
step consumes, so the plotting path and the table-building path share one
fit call, not two. Same model-dependency note as Part 1: ships against
today's two-exponential model, generalizes to Part 2's exp+power-law model
with no interface change once that lands.

**Implemented.** `plot_flux_profile_two` carries `fit: bool = False`,
`fit_model: str = "expcore"`, `fit_method: str = "psf"`,
`fit_skip_inner: int = 1`, `gamma_fixed: float | None = 0.8` (matching
`halo-flux-fitting.md` Part 2's shipped default) plus a `psf_r`-style PSF
passthrough — the model-dependency note above resolved itself in
`analysis.py`'s favor once Part 2 landed, so this ships against expcore by
default rather than the originally-assumed two-exponential fallback.
`plot_centroid_profile_two` was NOT given a `fit` argument — there is no
centroid-vs-radius model to overlay, only the flux profile is fit.

## Part 5 — per-run deliverable: printed summary + Part 4 figures (NOT an auto-built cross-split table)

**Clarification:** a single notebook run computes ONE subsample split (or the
full stack) at a time, exactly like today's `plot_centroid_profile_two`/
`plot_flux_profile_two` calls — this spec doesn't change that. Each run's
actual deliverable is (a) the Part 4 comparison figures with the fit overlaid,
and (b) a printed summary of that run's derived numbers, clean enough to read
off and drop straight into the paper — mirroring the existing house
convention `fitting.describe_fit` already uses for a single fit.

**Implemented 2026-07-18** as `analysis.describe_subsample_properties(*,
label, n_gal=None, base_sample=None, fit_result=None, boundary_info=None,
vel=None, core_lum=None, onehalo_lum=None, halo_lum=None, ratios=None,
truth=None)`. Every result argument is an already-computed dict (this
function does no computation, purely prints) and independently optional —
whatever's `None` prints a "(not computed)" placeholder instead of raising,
since an exploratory run rarely has all of `core_lum` (the expensive
PSF-rescale-recoadd piece), `onehalo_lum`/`halo_lum`/`ratios` (Part 3b), and
`vel` computed at once. Prints: header (label, N, base sample), the fit —
delegated to `fitting.describe_fit_expcore`/`describe_fit` (dispatched on
`fit_result['model']`) rather than reimplemented, so it never drifts from
those — the boundary radius + `model`/`source`/`boundary_from_own_fit`,
core/halo velocity + difference, core/halo(one-halo)/2-halo luminosity, and
the three Part 3b ratios + reconciled total, each with its 16/84 band via a
shared `_fmt_val_lohi` helper (`value  [lo, hi]`, not `+/-`, since every
number here is a bootstrap percentile). The `halo_lum` section is explicitly
labeled "2-halo" and annotated "NOT this galaxy's own CGM" in the printed
output itself, not just in the docstring — the whole point of Part 3b was
that this distinction stops being implicit.

**The full cross-split table (one row per `subsample_splits.md` #1–11 entry,
plus low-z/high-z, plus the full stack) is the eventual goal, not something
this spec assembles in one automated pass.** It's built by running each split
once, capturing that run's printed/returned numbers, and only then collecting
them — either by hand (copy each run's printout into the paper's table
directly, the simplest path), or, once every split has actually been run and
its numbers exist, a lightweight `build_property_table(results: dict)` that
just reformats an already-assembled `{label: props}` dict into one exportable
table (`.write(path, format="latex")` / `.to_pandas().to_markdown()`). That
aggregator is real but secondary — it has nothing to compute; it only
reformats numbers that already came out of individual runs.

## Open questions (remaining, minor)

1. Table format for the eventual Part 5 aggregator — astropy Table vs. pandas
   DataFrame.
2. Function homes — `describe_subsample_properties` / the eventual
   `build_property_table` in `analysis.py` vs. a new small module.
3. ~~Part 1/4's model dependency on `halo-flux-fitting.md` Part 2 — ship now
   against the two-exponential model (recommended, per the earlier
   discussion) or wait.~~ **Resolved 2026-07-18:** Part 2 shipped and was
   promoted to the default model in `analysis.py`; Part 4 and
   `describe_subsample_properties` (Part 5) both dispatch on
   `fit_result['model']` and work against either model, defaulting to
   expcore in practice.
4. Part 3b's model-decomposed halo-zone diagnostic (estimated 2-halo leakage
   fraction) — worth building alongside the primary empirical zone-sum, or
   deferred until there's a specific referee/Discussion need for it.

## See also

- `halo-flux-fitting.md` — the two-component fit (Parts 1–2) that defines the
  core/halo boundary this whole spec is built on.
- `uv-flux-fitting.md` — source of the per-galaxy kpc-PSF pattern Part 3's
  core-correction procedure mirrors.
- `../docs/subsample_splits.md` — the full candidate-split list this spec's
  table is meant to eventually cover, one row per split.
