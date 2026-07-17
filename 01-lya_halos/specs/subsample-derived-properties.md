# Subsample-derived properties — fitting-based comparison tables

Companion to `halo-flux-fitting.md` (the fitting model, Parts 1–2) and
`subsample_splits.md` (the candidate splits this applies to). Those specs cover
fitting one profile and enumerating which splits to run; this one covers turning
every subsample-comparison PLOT into a row of derived NUMBERS — core/halo
luminosity, core/halo velocity, and their ratios — collected into one reference
table, not just a figure per split. Design finalized 2026-07-17 after working
through the PSF/boundary tradeoffs in detail (see decision log at the bottom);
ready to implement.

**Status: not started (spec only), design settled.**

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

## Part 5 — per-run deliverable: printed summary + Part 4 figures (NOT an auto-built cross-split table)

**Clarification:** a single notebook run computes ONE subsample split (or the
full stack) at a time, exactly like today's `plot_centroid_profile_two`/
`plot_flux_profile_two` calls — this spec doesn't change that. Each run's
actual deliverable is (a) the Part 4 comparison figures with the fit overlaid,
and (b) a printed summary of that run's derived numbers, clean enough to read
off and drop straight into the paper — mirroring the existing house
convention `fitting.describe_fit` already uses for a single fit. New function,
e.g. `describe_subsample_properties(props)`, printing: split label, N
galaxies (+ which base sample — AGN-excluded ~450 vs. AGN-included ~500, not
always the same one), h1/h2 (or r_c/gamma) + chi2/dof, the fitted boundary
radius (+ `boundary_from_own_fit` flag), core/halo luminosity + ratio,
core/halo velocity + difference, each with its 16/84 error.

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
3. Part 1/4's model dependency on `halo-flux-fitting.md` Part 2 — ship now
   against the two-exponential model (recommended, per the earlier
   discussion) or wait.

## See also

- `halo-flux-fitting.md` — the two-component fit (Parts 1–2) that defines the
  core/halo boundary this whole spec is built on.
- `uv-flux-fitting.md` — source of the per-galaxy kpc-PSF pattern Part 3's
  core-correction procedure mirrors.
- `../docs/subsample_splits.md` — the full candidate-split list this spec's
  table is meant to eventually cover, one row per split.
