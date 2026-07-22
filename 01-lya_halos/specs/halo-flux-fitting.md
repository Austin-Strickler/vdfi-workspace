# Halo flux fitting — implemented state + proposed extension

General home for the radial flux-profile fitting work, superseding the
narrower `exponential-plus-powerlaw-fit.md` name (renamed to make room for
documenting what's already shipped, not just the next proposed model).
Sibling/successor to `specs/flux-profile-exponential-fit.md` (the original
Option A/B spec, written before any of this existed) and
`specs/psf-exponential-fit-recovery.md` (the Phase-1 synthetic testbed that
validated the approach before it touched real data).

**Status: DONE (real-data validated).** Part 1 (below) describes shipped code
(`utils_lya_halo/fitting.py`), now run against real data — the PSF-aware
two-exponential fit gives **χ²/dof = 1.17, inner scale h1 ≈ 16–17 kpc**,
outer h2 ≈ 1552 kpc; the inner scale is finalized and in the pipeline. Part 2
(Option C — exponential core + cored power-law halo, "expcore") is now
**implemented, fit against real data, and promoted to the default model** —
see the "As implemented" note at the top of Part 2 below for the real-data
result and how each open question was actually resolved. Part 3
(UV-continuum radial decline from CFHT-LS r-band imaging) is now
**implemented and validated** — the measured **UV scale length
≈ 1.55 kpc** gives a **halo/UV scale ratio ≈ 10:1** — with its own settled
workflow documented in the companion `uv-flux-fitting.md`.

## Part 1 — what's implemented (`fitting.py`)

Split out of `psf_exponential_recovery.py` (the Phase-1 testbed) into its
own module so the exact same fitting code runs against both known-truth
synthetic profiles and real data (`boot['total_flux_fid']` from
`measure_all_bins`/`run_measurement_suite`) — no drift between "the code
that was validated" and "the code that fits real bins."

**Model.** Two-component exponential,
`I(r) = A1*exp(-r/h1) + A2*exp(-r/h2)`, h1/h2 not ordered by construction —
canonicalized after the fact (see below) so h1 is always the steeper
(core) term.

**PSF.** `moffat_1d(r, fwhm, beta=3)` as the default analytic stand-in for
the VIRUS PSF, `normalize_psf_flux` rescales any 1D `(r, vals)` PSF array
(analytic or empirical — e.g. eventually `starpsf.py`'s measured curve) so
its flux integrates to 1. The fundamental PSF input everywhere downstream
is just that `(r, vals)` pair, so swapping in a measured PSF later touches
nothing else.

**Ring convolution (`ring_convolution_matrix` → `R`).** For a circularly
symmetric PSF and a thin unit-flux ring source at true radius `r0`, the
PSF-smeared surface density at observed radius `rho` is a closed-form
angular integral (`_ring_convolved_value`); integrating that over each
observed bin gives `R[i,j]` = fraction of a unit-flux ring at
`r_fine[j]` landing in bin `i`. `R` depends only on `(r_fine, r_edges,
psf_r, psf_vals)` — never on `(A1,h1,A2,h2)`, since convolution is linear,
so it's built once and reused for every trial parameter set during
fitting. Two real bugs were caught and fixed building this (not just
precision nitpicks):
1. The phi-quadrature originally used a fixed grid over the full
   `[0, 2pi]`; for a compact PSF and `r0` large compared to the PSF width
   (routine once bins reach hundreds–thousands of kpc), the PSF's angular
   support shrinks to a sliver near `phi=0` that a fixed grid misses
   entirely. Fixed by computing the exact support window via `arccos`
   rather than integrating blindly — caught by a flux-conservation check
   (`validate_ring_convolution`, a unit-flux ring must integrate back out
   to 1 at every `r0`, including out to 2000 kpc).
2. An early version compared the model at a bin's midpoint against the
   bin's true integral — fine for narrow, similar-width bins, wrong by up
   to ~10^9x for this pipeline's widest bins (10 kpc innermost, 1000+ kpc
   outermost). Fixed via proper bin-integration (`_bin_integrate`) before
   any PSF step is applied.

**Average, not summed — this is load-bearing, not a style choice.** The
real per-bin data value is a **biweight average** of stacked fiber flux in
that annulus, not a total. The model has to match that exactly:
`bin_average_no_psf`/`bin_average_psf` both divide the bin-integrated
quantity by bin width, so the fit compares mean-to-mean rather than
mean-to-sum. This is the same convention Steidel/Wisotzki/Leclercq use
when they fit a *surface brightness* (flux per unit area) rather than a
raw per-pixel sum — this pipeline's convention already matches the
literature's on that point; nothing to fix there. `_fine_ring_flux`
(intrinsic profile × local grid spacing `dr`) is the actual flux quantum
that gets redistributed by `R` — skipping the `dr` weighting was flagged
explicitly in the code as a real bug class (bin-flux artifacts from grid
resolution), not a hypothetical one.

**Two fits, same interface, one difference.**
- `fit_naive` — no PSF correction, drops `fit_skip_inner` innermost bin(s)
  (default 1), fits `bin_average_no_psf` against the rest.
- `fit_psf_aware` — full forward model, `R @ fine_flux` → per-bin mean,
  fit against *all* bins including the inner one (the PSF model absorbs
  the inner-bin correlation instead of discarding it).
- Both: multi-seed-then-lowest-chi2 (`_default_seeds` tries several
  core/halo split points plus an h1/h2-swapped variant — two-exponential
  fits on ~10 sparse bins are prone to local minima on a single seed, the
  same instability `speclsf.py`'s double-Gaussian fit already guards
  against). Amplitude-normalized internally (`_amp_scale`) so `curve_fit`
  stays numerically well-conditioned when amplitudes are ~1e39 and scale
  lengths are ~1–1000 — fitting raw units was checked directly to stall
  at the seed (chi2/dof ~5e5); normalized, the same data converges to
  chi2/dof <1. Bounds are data-driven (`_data_driven_bounds`: amplitude up
  to 50× the largest bin, scale length up to 10× the fitted radial span)
  rather than a blanket `(0, inf)`, so the optimizer can't wander into
  numerically-degenerate, physically meaningless solutions. Result
  ordering is canonicalized (`_canonical_order`) so `h1 <= h2` always —
  the model is symmetric under swapping `(A1,h1)<->(A2,h2)`, so an
  unconstrained fit can otherwise return core/halo in either slot, which
  would make "h1 vs. truth" comparisons and Monte Carlo scatter look wrong
  half the time for no physical reason.

**Current real-data result (this exact code, PSF-aware fit, all 10 bins,
inner bin kept):** χ²/dof = 7.04/6 = 1.17; A1 = 6.254e39 ± 5.04e38,
**h1 = 16.9 ± 1.1 kpc**; A2 = 7.318e37 ± 1.28e37, **h2 = 1552 ± 548 kpc**.
h1 sits inside the Steidel et al. 2011 range for stacked KBSS-like Lyα
halos (20.8–28.4 kpc across subsamples); h2 is well beyond the
one-halo/CGM regime (R_vir marked ≈75 kpc on the same plot, and the break
between the two fitted terms falls almost exactly there) and, converted to
comoving units at a typical KBSS redshift, lands close to the only two
direct-imaging detections of the cosmic web in Lyα emission currently in
the literature (Bacon et al. 2021, 2.5–4 comoving Mpc; the 2024/2025 MUSE
Ultra Deep Field 5 Mpc filament result). See
`halo_gas_correlation_literature_review.md` for the full literature
comparison this result was checked against.

## Part 2 — exponential core + cored power-law halo ("expcore")

**As implemented (2026-07-18), superseding "proposed extension" below.**
The beta-model form sketched in this section (`I(r) = A1*exp(-r/h1) +
A2*(1+(r/r_c)^2)^(-gamma/2)`) shipped as `fitting.py`'s
`intrinsic_profile_expcore` / `fit_naive_expcore` / `fit_psf_aware_expcore`
family, and was then promoted into `analysis.py` as the primary interface:
`plot_flux_profile_fit`/`_two` take `model="expcore"` (now the **default**)
or `model="twoexp"`, and the old `fitting.plot_expcore_fit`/
`bootstrap_fit_expcore` were removed and generalized into
`analysis.bootstrap_fit_profile` (works for either model, and is also the
tool for refit-per-draw parameter uncertainty on `r_c`/`gamma`, which have
no closed-form error otherwise). A real PSF-width bug (fits running ~3–4x
too narrow) was caught and fixed during the promotion.

How the open questions below actually resolved:
- **Q1 (beta-model vs. multiplicative window):** beta-model, as drafted —
  no need for the multiplicative alternative.
- **Q2 (is `r_c` pinned to R_vir?):** no — `r_c` floats freely as a fit
  parameter; `find_core_halo_boundary` (`fitting.py`) reads the *fitted*
  crossover back out afterward and compares it to R_vir post hoc, rather
  than the fit assuming they're equal.
- **Q3 (does `gamma` float, or get fixed at a literature value?):** fixed,
  not floating, and the default landed on the **projected** slope, not the
  raw one — `gamma_fixed=0.8` (`projected_slope_from_3d(1.8)`, the
  Limber-projected z~2–3 clustering slope), not the raw 3D `gamma=1.8`.
  On real data, `r_c` and `gamma` are strongly degenerate when both float —
  even on the highest-S/N full stack the free-`gamma` fit is essentially
  unconstrained — so `gamma` is fixed by default (pass `gamma_fixed=None`
  to float it anyway, or another literature value to test directly).
- **Model comparison, done:** `compare_models_aic_bic` was run —
  **AIC/BIC prefer the plain two-exponential model over expcore**, either
  gamma choice. expcore is kept as the default in `analysis.py` for its
  physical motivation (the outer term reads as an actual clustering/2-halo
  profile, not just a second exponential that happens to cross over near
  R_vir) rather than because it's the statistically preferred fit — worth
  stating explicitly if this goes in the paper, since "default plotting
  model" and "best-fitting model" are not the same claim here.
- **Q5 (`_canonical_order`-equivalent needed?):** not revisited during the
  promotion; the two expcore terms aren't swap-symmetric by construction
  (per the original reasoning below), so this is presumed still moot, not
  re-confirmed in code.
- **Q4 (line-window depth vs. fiducial r0):** still open — not addressed by
  the implementation.

Original proposal text (Q1–Q5 numbering above refers to this) kept below
for the reasoning trail:

Motivated by a direct test: fitting a bare power law (slope pinned to
−1.8, a literature 3D-correlation-style value) to the outer component fit
noticeably worse than the exponential currently in `fitting.py`. Two
things are worth separating here — whether the *functional form* should
change, and whether the *comparison slope* being tested was even the right
number to test against.

### Why not a bare power law dropped into the existing machinery

`A2 * (r/r0)^(-gamma)` diverges as `r -> 0`. Inside the same joint-fit
structure `fitting.py` already uses, that divergence means the outer term
is free to dump arbitrary flux into the innermost bins during the fit —
it either drags `A1`/`h1` off their true values to compensate, or forces
an artificial inner cutoff radius chosen by hand, which creates a
derivative discontinuity at the boundary with no physical motivation (and
makes the fit's goodness-of-fit near the transition sensitive to exactly
where that arbitrary boundary was drawn — the "interference" problem this
extension is meant to avoid).

### Candidate functional form

Give the power law a core so it's finite and smooth everywhere, and only
becomes power-law-like once `r` exceeds some transition scale — a
beta-model/Plummer-style softening, standard in surface-brightness work
(globular clusters, X-ray beta-model cluster profiles) for exactly this
reason:

```
I(r) = A1 * exp(-r / h1)  +  A2 * (1 + (r / r_c)^2)^(-gamma / 2)
```

As `r -> 0` the second term flattens to `A2` (finite); as `r >> r_c` it
asymptotes to `A2 * (r/r_c)^(-gamma)`, recovering the pure power law. `r_c`
is a genuine free parameter (bounded away from 0), not a hand-picked
cutoff. **[DECIDE]** alternative, not recommended as the default: a
multiplicative smooth window instead of an additive core
(`A2 * (r/r_c)^(-gamma) * (r/r_c)^n / (1+(r/r_c)^n)`) — more flexible, one
more parameter, more seed-sensitive; only worth trying if the beta-model
core visibly mismatches the transition region.

Note this new form is **not** symmetric under swapping its two terms the
way the current two-exponential model is — an exponential core and a
cored power-law halo are different shapes, so there's no `(A1,h1)<->(A2,
r_c,gamma)` ambiguity and `_canonical_order`'s swap-and-sort logic isn't
needed for this model (one less thing to guard against, not a gap to
fill).

### Reuses the existing machinery almost unchanged

`R` (`ring_convolution_matrix`) depends only on `(r_fine, r_edges, psf_r,
psf_vals)`, never on the intrinsic profile's functional form — so
everything downstream of "define a new `intrinsic_profile` variant" (`R`
itself, `_fine_ring_flux`'s role, `bin_average_no_psf`/`bin_average_psf`'s
divide-by-bin-width step, the `fit_naive`/`fit_psf_aware` two-fit
comparison structure, `_amp_scale` normalization, `_data_driven_bounds`,
`_best_of_seeds`) carries over with no changes. What's new: a
`intrinsic_profile_expcore` (or similar) function, a matching
`_default_seeds`-style seeding routine for the new parameter set
(`A1, h1, A2, r_c, gamma`, or `A1, h1, A2, r_c` with `gamma` fixed — see
open questions), and bounds that keep `r_c > 0` and `gamma` in a sane
physical range (order 1–3, matching typical galaxy-correlation slopes)
rather than the blanket scaling `_data_driven_bounds` uses today.

### The "average vs. summed" question — already answered by Part 1

Worth restating in this more general doc since it applies identically to
Option C: the real data value being fit is already a **bin average**
(biweight mean flux), matching the surface-brightness convention Steidel/
Wisotzki/Leclercq already use — not a raw per-bin sum. `fitting.py`'s
`bin_average_no_psf`/`bin_average_psf` machinery does this generically
(divides whatever `_bin_integrate`/`R @ fine_flux` produces by bin width),
so it needs no new work for Option C — just confirm, once a new
`intrinsic_profile` variant exists, that it's being fed through the same
`bin_average_*` path rather than accidentally compared as a raw
integral/sum.

### The 2D-vs-3D slope question — check before re-testing any literature γ

A 3D two-point correlation `xi(r) ~ (r/r0)^(-gamma)` does not project to a
2D radial profile of the same slope. The Limber approximation (the same
relation connecting 3D galaxy clustering to the observed angular
correlation function) says integrating a power-law correlation along the
line of sight reduces the slope by one power: fully projected goes as
`R^-(gamma-1)`, not `R^-gamma`. For `gamma=1.8` (a typical z~2-3 galaxy
correlation slope), the fully-projected expectation is closer to `R^-0.8`
— noticeably shallower, and plausibly why a straight `-1.8` fit looked
wrong even if a clustering-driven outer term is still the right physical
picture. Whether the full shift applies depends on how deep a
line-of-sight window this pipeline's `flux(r)` integrates over (the "line
window" already named in `psf-exponential-fit-recovery.md`) relative to a
fiducial correlation length `r0` — a narrow window gives something between
the 3D and fully-projected slope, not automatically either endpoint. This
needs an actual number (line-window depth vs. r0) before any literature
slope comparison means anything, and is flagged as a prerequisite check,
not something resolved by picking a functional form.

### Error propagation and model comparison — deferred, same posture as Part 1

Fiducial-only fitting (current state) is a reasonable starting point,
consistent with how this pipeline staged the rest of the work (Monte
Carlo validation on synthetic data came before real-data bootstrap
propagation). Once Option C exists: bootstrap the *fit* (refit on each
bootstrap draw of the underlying spectra, not just read errors off one
central fit's covariance) is the correct long-run approach, especially
since `r_c`/`gamma` are more likely to trade off non-Gaussian-ly than
`h1`/`h2` do when outer bins are noisy. And once both models exist,
compare Option B (two exponentials) vs. Option C (exponential + cored
power law) via AIC/BIC on the same bins, rather than resting on the visual
"exponential looks better" impression alone — the outer bins are exactly
where noise is most likely to fool visual comparison.

## Open questions before implementing Part 2

1. Beta-model additive core (default) vs. multiplicative smooth-window
   power law — start with the beta-model.
2. Is `r_c` free, or pinned to a physical scale (e.g. R_vir, mirroring how
   `flux-profile-exponential-fit.md`'s Option A optionally pins
   `r_break`)? Pinning is cheaper/more falsifiable but assumes the
   transition *is* at R_vir, which the current h1/h2 result suggests but
   doesn't guarantee for a differently-shaped outer term.
3. Does `gamma` float, or get fit at two literature-motivated fixed values
   (1.8 raw vs. ~0.8 fully-projected) compared by chi2/AIC, to directly
   test which the data prefers rather than one floating-gamma fit?
4. Line-window depth vs. fiducial r0 — prerequisite number needed before
   the 2D/3D slope section above is actionable.
5. Once implemented: does a `_canonical_order`-equivalent check need to
   exist for Option C at all, given the two terms aren't swap-symmetric —
   likely no, but worth a one-line confirmation in the eventual code
   rather than silently assuming it.

## Part 3 — proposed extension: UV-continuum radial decline from CFHT-LS r-band imaging

**Goal.** Measure the sample's rest-frame UV-continuum flux as a function of
radius directly from CFHT-LS r-band imaging (not VIRUS spectra), coadd it
across the sample, and fit its radial decline — a genuinely different data
product from Part 1/2 (broadband imaging vs. fiber spectroscopy), sized to
eventually be fit and compared alongside the Lyα halo profile using the same
`fitting.py` conventions once it exists as a real per-bin flux(r) array.

**Why r-band is the right filter.** Over the sample's z=2–3 range, observed
r-band samples rest-frame ≈1500–3000 Å — squarely the UV continuum, not a
strong emission line — so this is a clean stellar-continuum probe, physically
distinct from the scattered-Lyα-photon halo Part 1/2 model.

**Physical picture — why this is a different model, not a Part-1 reuse.**
Part 1/2's two-exponential and cored-power-law models describe *scattered
Lyα photons* (a genuine core+extended-halo structure). The UV continuum is
starlight from the galaxy itself — expected to look like an ordinary
disk/Sersic light profile, not a core+halo structure. Per the Profile Model
decision below, this is spec'd as its own single-exponential (Sersic-capable)
model in `fitting.py`, not a repurposing of `intrinsic_profile`.

### Data

Two CFHT-LS r-band mosaics, one per field (AEGIS, COSMOS), each ≈1 deg²
at 1/6″/pixel (≈0.167″/px). Full-mosaic images are too large to work with
directly at per-galaxy speed, so each field needs its own **cutout** around
the actual sample footprint before any per-galaxy extraction runs.

**Image paths — DECIDED, rewire `cont_image_path` (config.py code TODO).**
The current `resolve_cont_image_path()` per-field map
(`CFHT_COSMOS_image.fits` for COSMOS, `groth.fits` for AEGIS) is stale.
There's a newer, working pair already sitting in `catalogs/images/`:
`catalogs/images/groth.fits` (AEGIS) and `catalogs/images/cosmos.fits`
(COSMOS). **When coding this up: update `resolve_cont_image_path`'s
per-field dict to these two paths** — no new config field needed (no
`uv_image_path`, superseding the earlier draft of this spec); this
extension and the existing `mask_method='image'` fiber-masking path both
end up reading the same, now-current, per-field image through
`cont_image_path`.

- AEGIS already has a cutout defined from earlier work — **[DECIDE]:
  confirm its footprint/margin (see below) still covers every AEGIS
  galaxy's outermost annulus + background ring**, since it wasn't
  necessarily built for this specific bin scheme; regenerate if not.
- COSMOS has no cutout yet — build one the same way: bounding box from
  the actual RA/Dec **object distribution** of the COSMOS subsample (from
  the catalog, see RA/Dec source below), padded by a buffer of roughly
  **2× the largest annulus radius** (angular size at the sample's lowest
  z, i.e. the largest angular extent in the sample) — a deliberately
  generous margin "just to be sure" per your call, not a tight minimum.

**RA/Dec source: reuse the combined product, don't re-query.** The 450
galaxies to extract are already the exact sample that made it into the
spectral stack (same catalog cuts, same finite-spectrum cut). Pull RA/Dec
from the already-combined `GalaxyProduct`/`stacks`/`boot`'s attached
catalog table (`product.catalog`, carried through `_concat_products`,
`load_for_stack`, and Stage 2) rather than re-querying
`catalog_path` fresh — guarantees this UV-continuum sample is pixel-for-
pixel the same 450 galaxies already in the Lyα profile, not a
similar-but-possibly-drifted list.

### Extraction: single-field-at-a-time, not interleaved

Your instinct is right, and it matches an existing convention:
`PipelineConfig.normalized_field()` already treats `field='BOTH'` as a
**stacking-only** concept — Stage 1 extraction is always single-field, and
combination happens after, specifically to avoid the extraction step
having to reason about two fields at once. Do the same here: run every
AEGIS galaxy's cutout+centroid+annuli against the AEGIS mosaic in one pass,
then every COSMOS galaxy against the COSMOS mosaic, rather than switching
field-by-field per galaxy. This avoids repeatedly opening/seeking a ~1 deg²
FITS (or holding both mosaics memory-resident at once) — real cost, not a
hypothetical one, at this pixel scale.

### Per-galaxy pipeline

1. **Cutout.** From the galaxy's catalog RA/Dec, cut a small WCS-aware
   stamp sized to comfortably contain the outermost annulus + background
   ring. Reuse the WCS/pixel-scale pattern already built for fiber masking
   (`masking.py`'s `WCS(header)`, `get_pixscale_arcsec`, `SkyCoord`) rather
   than reinventing RA/Dec-to-pixel handling — this is the same conversion,
   just applied per-galaxy instead of per-fiber.
2. **Centroid.** 2D-Gaussian fit (`photutils.centroid_2dg`) on the cutout
   to locate the true continuum peak, windowed tightly enough to avoid
   pulling onto a neighbor. Compute the angular offset from the catalog
   position via `SkyCoord` separation; flag (`centroid_offset_flag`)
   galaxies with offset > 0.5–1″ (exact threshold tunable — start at 0.5″
   per your instinct, loosen if it flags an implausibly large fraction of
   the sample). **DECIDED: kept-but-flagged**, not dropped — mirrors
   `success_frac`/mask-based QC already used elsewhere in this pipeline
   (e.g. `bootstrap_measurements`'s success tracking), so you can inspect
   the flagged fraction before ever deciding to cut anything.
   **DECIDED: fallback when the 2D-Gaussian fit fails to converge**
   (low S/N, blended source) is the catalog position + flag — simplest
   option, consistent with the kept-but-flagged policy above. Revisit
   toward a multi-seed retry (same multi-seed-then-best-fit philosophy
   `_default_seeds`/`_best_of_seeds` already use for the flux-profile
   fits) only if this simple fallback ends up flagging an implausibly
   large fraction of the sample in practice.
3. **Circularized annuli / per-galaxy flux profile.** Once centroid is
   fixed, build circular annuli (`photutils.CircularAnnulus`) at the same
   radii convention as the rest of the pipeline (`config.bins`/
   `bin_mode` — see Radial Bins below), converted to this image's native
   pixel scale via the same kpc↔arcsec conversion machinery `virial.py`/
   `stack.py` already use (`cosmo.angular_diameter_distance(z)` at each
   galaxy's own z). Report the **bin AVERAGE** flux per annulus (mean
   per-pixel value within the ring), not a raw sum — matches Part 1's
   "average, not summed" convention exactly (see Part 1 above), so this
   profile is fit with the same `bin_average_*`-style machinery once it
   exists, not a differently-normalized quantity that needs reconciling
   later.
   - **Background/sky subtraction — plan sketched, exact form still
     open.** Working approach: mask bright objects using the existing
     full-field segmentation mask (see below — it's built from these
     exact images), then take the **median of the remaining unmasked
     sky pixels across the whole field** as one global background level
     per field, subtracted from every galaxy's annuli. Simpler fallback
     if that's not runtime-feasible right now: a flat median of the whole
     (unmasked) field — accept the small risk of slight oversubtraction
     from any bright-object flux that leaks through, and revisit only if
     it turns out to matter.
   - **Neighbor/contaminant masking — DECIDED, don't actively mask
     per-annulus.** A SExtractor segmentation map already exists, built
     directly from these CFHT-LS images (not the fiber-scale one — it's
     pixel-compatible by construction) and lightly dilated to be
     conservative; reuse it for the background-median masking above. For
     the per-galaxy annulus photometry itself, don't reject or mask
     contaminated bins — the final biweight coadd across the sample
     (Coaddition below) is expected to be robust to the occasional
     neighbor-contaminated annulus in one galaxy. **Note to revisit**: if
     the stacked profile looks contaminated (e.g. an unexpected bump at a
     particular radius), this is the first assumption to test — rerun
     with per-annulus contaminant masking using the same segmap and
     compare.

### Radial bins

Use `config.bin_mode`'s existing scheme (virial/kpc/arcsec), per your
normal workflow of hand-defining a fresh `PipelineConfig` per notebook —
so this reuses the *mechanism* (`radial_bin_edges`, the per-galaxy
kpc↔arcsec conversion, `bin_mode` semantics) without reusing the Lyα
halo's actual bin *values*. You'll pass your own ultra-fine, small-radius
bin edges tailored to this imaging's much better native resolution
(1/6″/px vs. the fiber-limited spectral bins) — nothing in this spec
requires matching Part 1's `[0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]`
default.

### Coaddition and bootstrap

Once every galaxy has a `(r, mean_flux)` profile: combine across the
galaxy axis using the **same** `config.galaxy_combine_methods`
(biweight/inv_var/mean/median/sigma_clip/weighted_median) Stage 2
(`stack.py`) already computes for the spectral stack — one coadd
methodology across the whole pipeline, not a second bespoke one for
imaging. Once the fiducial coadd is validated, bootstrap the galaxy
resampling (same style as `measure.py`'s `bootstrap_measurements`/
`bootstrap_all` — resample *which galaxies* enter the coadd, not a
per-pixel noise model) for 16/84 error bars on the stacked r-band
flux(r) profile. Deferred until the fiducial pipeline is validated —
same staged posture Part 1/2 already use (synthetic/Monte Carlo
validation, or here fiducial-only validation, before bootstrap
propagation).

### Model: single exponential (default), with an optional Sersic fit

Default model, a new addition to `fitting.py` (not a reuse of Part 1's
`intrinsic_profile`, per the Physical Picture note above):

```
I(r) = A * exp(-r / h_UV)
```

one component, matching the expected disk-like UV-continuum decline. PSF
correction reuses Part 1's ring-convolution machinery unchanged in
structure — `ring_convolution_matrix` depends only on `(r_fine, r_edges,
psf_r, psf_vals)`, never on the intrinsic profile's functional form, so
swapping in the CFHT-LS PSF is the only new input needed — same "swap the
PSF, touch nothing else" property Part 1's docstring already advertises.
**Not needed to get the first profiles built**, but will be needed once
real flux(r) curves exist and a PSF-aware fit is run: first choice is
whatever seeing FWHM is documented in the image header or the CFHT-LS
survey release notes (cheap, no new measurement); fall back to an
empirical measurement from stars in the mosaic (mirroring how
`starpsf.py` measures the VIRUS PSF) only if no documented value is
trustworthy enough.

Optional alternative form, offered alongside (not replacing) the default:

```
I(r) = A * exp(-b_n * [(r / r_e)^(1/n) - 1])          (Sersic)
```

with `n` fixed at 1 recovering the pure-exponential default above as a
special case. **[DECIDE]: `n` floated vs. tested at a small set of fixed
values** — same open-question shape as Part 2's `gamma`-floated-vs-fixed
question; likely worth deferring to once real profiles are in hand rather
than deciding blind.

### Where this lives during development

Per your direction, the extraction + fitting testbed for this starts in
`fitting.py` (fast iteration in one file, same as Option C's Section 7),
not `analysis.py`, until it's validated. One architectural note worth
flagging now rather than discovering later: Part 1/2's existing code in
`fitting.py` is pure numpy/scipy with no image I/O — this extension's
cutout/WCS/centroiding/aperture-photometry pieces pull in astropy/photutils
and are, in spirit, closer to `extract.py`'s Stage-1 job (turn raw
data + a catalog position into a per-galaxy array) than to `fitting.py`'s
job (turn an existing array into fitted parameters). Recommend keeping
that seam explicit even while everything sits in `fitting.py` for testing
— e.g. a clearly-separated "extraction" section vs. "fitting" section
within the file — so that when this *does* mature, the extraction half has
an obvious future home (`extract.py` or a new dedicated module) separate
from the fitting half, which stays in `fitting.py` permanently alongside
Part 1/2's models. Mirrors how Option C's fitting-only code was written to
never touch `analysis.py` until it earns its way in via
`compare_models_aic_bic` — same posture, applied to a wider slice of the
pipeline this time since extraction itself is new, not just a new model.

### Success criteria

- Centroid offsets are small and consistent with astrometric + centroiding
  noise across the sample — a handful of flagged outliers, not a
  systematic offset suggesting a WCS or catalog-matching bug.
- The fiducial coadded profile declines ~monotonically with radius,
  consistent with a UV-continuum disk profile (not dominated by
  neighbor contamination or an unsubtracted background gradient).
- The default single-exponential fit converges with a reasonable
  chi2/dof, reported with the same diagnostic conventions Part 1 already
  established for the real-data Lyα fit.
- Bootstrapped `h_UV` has a believable uncertainty and, once compared,
  a sane relationship to the Lyα halo's own `h1`/`h2` (e.g. is the UV
  continuum's scale length inside, comparable to, or well inside the
  Lyα core term) — the actual physical payoff of building this at all.

## Open questions before implementing Part 3

1. **Code TODO**: rewire `config.py`'s `resolve_cont_image_path()`
   per-field map to `catalogs/images/groth.fits` (AEGIS) and
   `catalogs/images/cosmos.fits` (COSMOS); confirm AEGIS's existing
   cutout still covers this bin scheme's radius + 2× margin, regenerate
   if not; build the COSMOS cutout fresh the same way.
2. Background subtraction: masked-median-of-field (primary plan) vs.
   flat-median-of-field (simpler fallback) — start with flat-median if
   the masked version isn't runtime-feasible yet; check against real
   data whether the oversubtraction risk actually matters.
3. PSF characterization — try the image header / CFHT-LS survey
   documentation for seeing FWHM first; only build an empirical
   star-based measurement (`starpsf.py`-style) if no trustworthy
   documented value exists. Not needed until real profiles are in hand.
4. Offset-flag threshold — start at 0.5″, revisit once the real flagged
   fraction is known.
5. Sersic `n`: floated vs. tested at fixed values — defer until real
   profiles are in hand, same posture as Part 2's `gamma` question.
6. If the stacked profile looks contaminated at some radius: revisit the
   decision not to mask per-annulus neighbors, using the existing
   SExtractor segmap.
