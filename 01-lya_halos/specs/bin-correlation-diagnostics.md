# Bin correlation diagnostics — `validation.compute_bin_correlation`

Companion to `combined_bin_significance` (`validation.py`). That function
already builds a bootstrap-draw covariance/correlation matrix across a
CHOSEN subset of radial bins as a means to an end (combining them into one
significance number). This spec promotes that correlation matrix to a
first-class diagnostic in its own right, always computed over EVERY bin, and
adds two more correlation matrices built from the actual spectra rather than
the bootstrap draws — together answering two separate questions the existing
tests never directly targeted: how much do neighboring bins' PSF-driven
cross-talk and large-scale measurement systematics actually correlate,
end to end.

**Status: design finalized 2026-07-24, ready to implement.**

## Goal

Produce one function, `validation.compute_bin_correlation`, that returns
THREE bin-by-bin correlation matrices from existing pipeline products (no new
bootstrap loop), plus a companion plotting function that renders all three as
a shared 1×3 heatmap figure, plus a third standalone tool
(`compare_bin_correlation_resolution`) that answers a separate, longer-
standing question — does the `rest_delta=0.2` vs. `0.6` choice actually
change anything, for the correlation matrices AND the measurement itself.

1. **Bootstrap centroid correlation** — how much do bins' *measured centroids*
   co-vary across galaxy-resampling bootstrap draws (the mechanism
   `combined_bin_significance` already exploits, generalized to all bins).
2. **Lyα-complex spectral correlation** — do bins' continuum-subtracted
   residual spectra, in the window actually used for centroiding, look alike.
3. **Outside-Lyα spectral correlation** — same, but in the rest of the
   spectrum, outside that window.

The inner-bin block of all three is the PSF-cross-talk readout (PSF blur
mixes flux between adjacent small annuli — same galaxies, same photons,
scattered across bin boundaries). The outer-bin block (Austin's target range:
100–3000 kpc) is the large-scale measurement-systematic-floor readout. One
figure, two questions, read off different corners of it.

## Why three matrices, not one

These are genuinely different things, not three views of the same number:

- Matrix 1 is a **statistic-level** covariance — how much the *centroid*
  co-varies across resamples, driven by shared galaxies across annuli (every
  bootstrap draw resamples galaxies once and reuses that resample for every
  radial bin, so one unusual galaxy pulls several bins the same way on the
  same draw).
- Matrices 2/3 are **spectral-shape** correlations — do two bins' leftover
  residual (after continuum subtraction) look like the same shape, pixel for
  pixel. This is a systematics/instrumental fingerprint check (shared sky
  residual, PSF wing leakage, continuum-fit ringing), independent of the
  bootstrap machinery entirely.

Matrix 1 can be near-zero while 2/3 show a real shared spectral bump (e.g. a
sky line contaminating two adjacent bins identically), and vice versa. A
systematic-floor read that only used one would miss what the other catches.

## Matrix 1 — bootstrap centroid correlation

Exactly `combined_bin_significance`'s existing `Corr` (built from
`real_boot['centroid_v_all']`, `(nboot, nrad)`), just always computed over
every bin instead of a caller-selected subset. No new math — this is the
existing `np.cov` / normalize-to-correlation block, lifted out and made
unconditional. Input: `real_boot`, the STANDARD production bootstrap
(`bootstrap_measurements`/`bootstrap_all`/`measure_all_bins` output, run at
the pipeline's normal `rest_delta = 0.2` — see "Resolution" below for why
this one stays fine-grained).

## Matrices 2 & 3 — spectral Pearson correlation

For each bin, continuum-subtract the FIDUCIAL (full-sample) stacked spectrum
using the same continuum model the real measurement uses (`get_continuum_model`,
same `cont_method`/`cont_order` as production — deliberately not a stronger
or different subtraction; see "Why not full/median subtraction" below), then
pairwise-Pearson-correlate the residual vectors across bins, restricted to a
wavelength mask:

- **Matrix 2 (Lyα complex):** mask = `[cont_bounds[0][0], cont_bounds[1][1]]`
  — blue sideband through the line window through red sideband, i.e. the
  exact span the real centroid measurement's own continuum fit + line window
  look at.
- **Matrix 3 (outside Lyα):** mask = everything in `rest_wave` OUTSIDE that
  span. **Exclusive, not the full spectrum** — if Matrix 3 also included the
  Lyα-window pixels, it wouldn't add independent information over Matrix 2,
  just dilute it. Exposed as `outer_mode="exclusive"|"full"` for a later A/B,
  defaulting to `"exclusive"`.

No bootstrap loop needed for either: `bootstrap_all`/`measure_all_bins`
internally slice the cube to the Lyα-complex measurement window for speed
(`_measurement_window`) and never retain the per-draw spectra in their return
value anyway (confirmed by reading `measure.py` — `draws` is built internally
to compute `stack_error`, then discarded). So even a bootstrap product run at
matching resolution cannot supply Matrix 3's outer-region pixels, at any
resolution — that data was thrown away before the loop even started. Both
spectral matrices instead read directly from a `build_stacks()` product's
`stacks['stacks'][method]['flux']`, over the FULL `rest_wave` grid, which
never gets sliced.

## Resolution — two different grids for two different purposes

**Matrix 1 stays at `rest_delta = 0.2`** (the pipeline default) — finer
sampling is what's actually optimal for centroiding, and it's what every
production number in this pipeline already uses, so Matrix 1 should reflect
the real measurement's own resolution, not a decorrelated one.

**Matrices 2/3 should be built from a separately reconfigured `rest_delta =
0.6` pass, not rebinned after the fact inside this function.** Reasoning:

- **The problem is real, not hypothetical.** Adjacent pixels in the residual
  spectrum are autocorrelated (LSF/resampling), so a Pearson r computed
  pixel-by-pixel between two bins' residuals has a much wider sampling
  distribution than `N_pixels` would suggest (smooth/autocorrelated series
  show inflated spurious cross-correlation — Yule 1926's "nonsense
  correlation" problem). Since Matrices 2/3 are single point estimates (no
  per-draw averaging to smooth this out — see previous section), an
  individual off-diagonal cell showing a large |r| could just be two
  independently-noisy-but-smooth curves wiggling together by chance, not a
  real shared systematic.
- **0.6 Å is not a guessed fix — it's the pipeline's own documented native
  sampling.** `config.py`'s `rest_delta` field comment states `0.2` is
  "oversampling of the ~0.6 A native sampling." Independently, the raw
  instrument grid (`wave_min=3470`, `wave_max=5540`, `nwave=1036`) gives
  ~2.0 Å/pixel observed, and at the sample's average redshift (~2.3, per
  Austin) that maps to `2.0 / (1+2.3) ≈ 0.61 Å` rest-frame — matching the
  doc comment. So 0.6 Å is the number to decorrelate against, not an
  arbitrary round choice.
- **Reconfiguring and re-running Stage 2/3 at `rest_delta=0.6` is cheaper
  than rebinning post hoc**, and is ~3x faster (fewer wavelength pixels
  throughout the rest-frame resample, stacking, and any bootstrap on top of
  it) — so there's no real cost to doing it properly upstream instead of
  patching it inside `compute_bin_correlation`. This also keeps the function
  itself simple: it just reads whatever grid spacing the `stacks` object it's
  given actually has, with no internal rebin/decorrelation-parameter logic
  to maintain.
- **This also enables a direct A/B check.** Because the function takes an
  arbitrary `stacks_spec` input rather than hardcoding a resolution, the same
  machinery can be pointed at a `rest_delta=0.2` build and a `rest_delta=0.6`
  build and compared directly — formalized as its own standalone tool,
  `compare_bin_correlation_resolution`, below, rather than a mode of this
  function.

## Why continuum-subtract at all, and why "not fully"

Raw flux across bins is trivially correlated — every bin is a declining
surface-brightness profile, which is real physics, not a systematic, so a
Pearson r on raw flux would just measure "both bins decline with radius" and
say nothing diagnostic. Subtracting the production continuum model removes
that shared trend so whatever correlation survives in the residual is the
actual signal being tested for — leftover noise structure, sky residuals,
PSF-wing shape, anything the continuum fit didn't capture. The subtraction
being imperfect is the point: if it were somehow perfect there would be
nothing left to correlate and the test would be vacuous by construction. Use
the SAME `cont_method`/`cont_order` as production (poly order 1 by default,
read from `stacks_spec`'s config or `real_boot['meta']`), not a separate
"median" or otherwise more aggressive pass — a stronger subtraction would
scrub out the very residual structure this test exists to find, biasing both
spectral matrices toward zero and giving false confidence.

## Inputs

- `real_boot` — production bootstrap output (`rest_delta=0.2`). Supplies
  `centroid_v_all`, `centroid_v_fid`, `r_edges` for Matrix 1 only.
- `stacks_spec` — a `build_stacks()` output, any resolution (recommended:
  a fresh `rest_delta=0.6` run; also usable at `0.2` for the A/B check
  above). Supplies the fiducial per-bin flux + `rest_wave` for Matrices 2/3.
  Must share the same `r_edges` as `real_boot` (same radial binning scheme,
  independent of wavelength grid) — assert this rather than silently
  misaligning the three matrices' bin axes.
- `r_min`, `r_max`, `bins` — subselect which bins are shown in the figure
  and summary print (e.g. Austin's 100–3000 kpc large-scale range). Matrices
  are always computed over ALL bins regardless; this only slices the view.
- `outer_mode` — `"exclusive"` (default) or `"full"`, see Matrix 3 above.
- `stack_method` — which galaxy-combine method's stack to read from
  `stacks_spec['stacks']` (default matches production, e.g. `"biweight"`).
- `verbose` — print a summary table, matching `combined_bin_significance`'s
  existing convention.

## Output

`dict` with bin geometry (`r_edges`, `centers`), the three matrices
(`boot_correlation`, `lya_correlation`, `outer_correlation`) plus their
underlying vectors/covariance where relevant, resolution metadata for each
matrix (`rest_delta` used, the Lyα-complex window bounds, `outer_mode`), and
a provenance stamp (`stamp_provenance`, matching every other function in this
module).

## Figure — `validation.plot_bin_correlation`

1×3 lower-triangle heatmaps (diagonal is trivially 1, matrix is symmetric —
only one triangle needs rendering), shared radius-bin tick labels across all
three panels, one shared diverging colormap centered at 0 (all three
matrices are bounded [-1, 1], so one colorbar works for all panels). Panel
titles: "bootstrap (centroid)", "Lyα complex (spectral)", "outside Lyα
(spectral)". Takes the `compute_bin_correlation` result dict directly, plus
the usual `r_min`/`r_max`/`bins` view-slicing and `save_fig`/`savename`
kwargs matching this module's other plotting functions.

## Cross-resolution comparison — `validation.compare_bin_correlation_resolution`

Answers a standing question independent of the systematics-floor motivation
above: does `rest_delta=0.6` (vs. the production `0.2`) actually change
anything that matters — not just the correlation matrices, but the
measurement itself. Deliberately a separate function from
`compute_bin_correlation`, not a mode of it: this is a one-off
curiosity/sanity check Austin runs when he wants to look, not part of the
standard diagnostic pipeline, and `compute_bin_correlation` stays simpler for
not needing to know about it.

**Core comparison — needs only `stacks_02`, `stacks_06` (no bootstrap at
all).** Runs `compute_bin_correlation`'s Matrix 2/3 machinery at each
resolution off their respective fiducial per-bin stacks, then reports per
matrix: `corr_02`, `corr_06`, the difference (`corr_06 - corr_02`), and one
summary number — flatten each matrix's upper triangle and correlate those
two lists against each other, a single "how similar is the overall
correlation PATTERN across resolutions" score. Nothing beyond two
`build_stacks()` runs at different `rest_delta` is required, since Matrices
2/3 were already established to be pure fiducial-spectrum computations with
zero bootstrap dependency.

**Optional centroid comparison — needs `boot_02`/`boot_06`, or falls back to
a point estimate.** If bootstrap products exist at both resolutions
(`real_boot` at 0.2, a `measure_all_bins`-style run at 0.6 — cheap, ~3x
faster than production), their `centroid_v_fid` values are already sitting
there: just diff them per bin, plus diff their 16/84 bootstrap widths. This
is the piece that actually answers "does going coarser change my measurement
precision," closer to the real long-standing question than the correlation
matrices alone. If no bootstrap was run at one or both resolutions, fall
back to one `measure_centroid` call per bin on each fiducial stack — no
error bar, just the point-estimate shift, always available and cheap.

**Left as opt-in, not default:** a `success_frac`/valid-bin-count diff
(`check_success_frac=False`) and a spectral overlay plot near Lyα for a
couple of representative bins (`plot_spectra_overlay=False`), so a first run
stays fast and focused on the two questions that actually matter: does the
correlation pattern change, does the centroid measurement itself change.

## Open questions

1. ~~Whether to expose a block-bootstrap CI on the Matrix 2/3 Pearson r's.~~
   **Resolved 2026-07-24:** moving Matrices 2/3 to `rest_delta=0.6` (the
   documented native sampling) IS the fix for the autocorrelation problem
   that would have motivated a block-bootstrap CI in the first place — once
   neighboring pixels are close to independent, the ordinary pixel-count-
   based uncertainty on r stops being badly wrong, so the elaborate
   `placebo_wavelengths`-style block-bootstrap isn't needed. Any residual,
   ordinary finite-sample uncertainty on r (ever present in a correlation
   from a limited number of points, autocorrelation aside) has a standard
   closed-form answer (Fisher z-transform interval) if it's ever wanted —
   cheap, analytic, no resampling loop — but left as a deferred nice-to-have,
   not part of the default output.
2. ~~Whether the `rest_delta=0.6` A/B check belongs as a permanent feature of
   `compute_bin_correlation` or a one-off.~~ **Resolved 2026-07-24:** built as
   its own function, `compare_bin_correlation_resolution` (above) — keeps
   `compute_bin_correlation` simple and treats the resolution question as the
   standalone check it actually is.
3. Whether to extend `compare_bin_correlation_resolution` to Matrix 1
   (bootstrap centroid correlation) if a full `rest_delta=0.6` bootstrap
   product becomes routinely available — not required for the initial
   version, which only compares the deterministic spectral matrices.

## See also

- `combined_bin_significance` — the existing function this generalizes
  Matrix 1 out of; still the right tool when the goal is an actual combined
  significance number for a chosen bin range, not a correlation diagnostic.
- `error_cross_check` — the other existing cross-check of bootstrap error
  honesty (jackknife vs. bootstrap), same spirit of validating the error
  bars rather than just trusting them.
- `placebo_wavelengths` — source of the moving-block-bootstrap pattern for
  autocorrelation-aware uncertainty, relevant if Open Question 1 is picked
  up later.
