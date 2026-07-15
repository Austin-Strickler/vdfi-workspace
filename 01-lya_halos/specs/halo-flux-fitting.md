# Halo flux fitting — implemented state + proposed extension

General home for the radial flux-profile fitting work, superseding the
narrower `exponential-plus-powerlaw-fit.md` name (renamed to make room for
documenting what's already shipped, not just the next proposed model).
Sibling/successor to `specs/flux-profile-exponential-fit.md` (the original
Option A/B spec, written before any of this existed) and
`specs/psf-exponential-fit-recovery.md` (the Phase-1 synthetic testbed that
validated the approach before it touched real data).

**Status: Part 1 (below) describes shipped code
(`utils_lya_halo/fitting.py`). Part 2 describes a proposed, not-started
extension (Option C).**

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

## Part 2 — proposed extension: exponential core + cored power-law halo

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
