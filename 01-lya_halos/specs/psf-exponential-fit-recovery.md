# PSF-aware two-component exponential recovery — standalone testbed

Spec for a new, **standalone** Python file (not part of the `utils_lya_halo`
package, not touching `analysis.py`) that builds synthetic two-component
exponential flux profiles, PSF-convolves them, bins them the way real data
is binned, and tests whether a PSF-aware fit can recover the injected scale
lengths. Austin drives it from his own Jupyter notebook. This is Phase 1 of
the flux-profile-fit work — Phase 2 (folding a fit method back into
`analysis.py`'s `plot_flux_profile`) is `specs/flux-profile-exponential-fit.md`
and is deliberately blocked on this.

**Status: implemented (2026-07-14).** `01-lya_halos/code/psf_exponential_recovery.py`
exists and has been run/validated (ring-convolution checked against
`scipy.integrate.quad`, a Monte Carlo recovery sweep confirms the PSF-aware
fit recovers both scale lengths with less bias and less scatter than the
naive drop-the-inner-bin fit). Kept as the reference doc per the specs
README convention — update this file if the approach changes again.

## Goal

Leclercq et al. 2017 fit MUSE Lyα narrow-band images with a sum of two
circular 2D exponentials (compact core + extended halo), convolving the
model with the MUSE PSF and comparing against per-pixel variance — rather
than fitting the PSF-*smeared* data with a naive un-smeared exponential and
hoping the bias is small. Austin wants the same idea applied to this
pipeline's radial flux profile: generate a known-truth two-component
profile, PSF-convolve it, discretize it into the same kind of radial bins
the real pipeline produces (with realistic per-bin noise), and check whether
a PSF-aware fit recovers the injected `h1`/`h2` — and, just as importantly,
how much a *naive* fit (no PSF, drop the first bin) is biased by comparison.
That bias comparison is the actual deliverable: it's what justifies (or
disproves) doing the PSF-aware fit for real in Phase 2.

This is a methods/prototyping file, not production pipeline code — it does
not need to live inside `utils_lya_halo`, and nothing in `analysis.py`
changes as part of this spec.

### Everything stays 1D — no pixel grid, ever

**This was a correction from Austin partway through, not a minor detail.**
The real pipeline's data product is already a stack: individual galaxy
spectra get extracted per radial (fiber-annulus) bin, stacked, and
collapsed — the thing that comes out is `SPECTRA(r)`, and integrating that
over the line window gives `flux(r)`, i.e. **one radius axis, full stop.**
There is no 2D image anywhere in the real pipeline for this measurement —
unlike Leclercq's MUSE narrow-band image, which genuinely is a 2D map. So
building a synthetic 2D `(x, y)` pixel image here, even just as an
implementation convenience, would model data that doesn't correspond to
anything the real pipeline actually produces. The 1D-ness is the base data
step, not a simplification applied on top of a "real" 2D problem: the
source model, the PSF, and the mixing between bins are all represented as
1D functions/arrays of radius, never as `(x, y)` arrays.

The one place this needs care: PSF smearing between concentric annuli is
still fundamentally a circular/angular effect (an annulus of true radius
`r0`, PSF-smeared, spreads flux to other radii in a way that depends on the
2D geometry of a ring, not just a 1D radius-to-radius blur). The fix is a
closed-form **ring-convolution integral** (below) that captures that
geometry using only 1D inputs (a radius array and a PSF-vs-radius array)
plus a 1D angular quadrature — it never materializes an `(x, y)` grid.

## Data

None real — this stage is entirely synthetic, on purpose (known truth is
the point). It borrows real conventions from the pipeline rather than
inventing its own, so results transfer cleanly to Phase 2:

- Radial bin edges: `EXAMPLE_R_EDGES_KPC = [0, 10, 20, 30, 50, 80, 140,
  300, 600, 1000, 2000]` (kpc) — Austin's actual 10-bin scheme (finer near
  the core, much coarser at large radius, extending out to a background/
  null-check regime at very large radius), not an arbitrary synthetic
  choice.
- Noise shape: `EXAMPLE_FLUX_SN = [10.6, 13.6, 10.4, 8.9, 4.0, 3.4, 4.5,
  6.3, 5.7, 4.7]` — the "flux S/N" column from Austin's real centroid
  table, used as a realistic per-bin noise SHAPE (see Method 4) rather than
  a flat fractional error.
- The PSF: a **Moffat profile with β=3** (a reasonable analytic stand-in
  for the VIRUS PSF), generated as a 1D array. When Austin has measured the
  real VIRUS PSF (likely the same kind of 1D radial curve `starpsf.py`'s
  `radial_continuum_profile`/`curve_of_growth` already produce for the
  empirical stellar PSF), that measured 1D array drops straight into the
  same interface — every function below takes a `(psf_r, psf_vals)` pair as
  the fundamental PSF input, with the Moffat generator being just one way
  to produce that pair for now.

## Method

### 1. Intrinsic model (1D, fine radius grid)

```
I(r) = A1 * exp(-r / h1) + A2 * exp(-r / h2)      # h1 < h2: steep core, shallow halo
```

evaluated on a fine 1D radius array `r_fine` — finer than the real
observation bins, extending somewhat past the outermost real bin edge so
flux isn't artificially clipped before the PSF has a chance to smear it
further out.

### 2. PSF model — Moffat β=3, as a 1D array

```python
def moffat_1d(r, fwhm, beta=3.0) -> np.ndarray:
    """Unnormalized Moffat profile: (1 + (r/alpha)^2)^-beta, alpha derived
    from fwhm and beta (alpha = fwhm / (2*sqrt(2**(1/beta) - 1))).
    beta=3 fixed default, matching the VIRUS PSF."""

def normalize_psf_flux(r, psf_vals) -> np.ndarray:
    """Rescale a 1D radial PSF array so integral(psf(r) * 2*pi*r dr) == 1
    (numerical trapezoid over r, 2*pi*r weighting) -- required so the
    ring-convolution below conserves flux. Works on ANY input 1D PSF array,
    not just the analytic Moffat, so the same call handles Austin's
    eventually-measured empirical PSF array too."""
```

The interface's fundamental PSF input is the pair `(psf_r, psf_vals)` — a
1D array, exactly as requested. `moffat_1d` + `normalize_psf_flux` is just
today's way of producing that pair for testing; swapping in a real measured
PSF later means passing a different `(psf_r, psf_vals)`, no other code
changes.

### 3. Ring-convolution: mixing true radius into observed radius, in 1D

For a circularly symmetric PSF `P(r)` and a thin unit-flux ring source at
true radius `r0`, the PSF-convolved surface density at observed radius `ρ`
is the closed-form ring integral:

```
C(ρ; r0) = (1 / 2π) ∫₀^2π P( sqrt(r0² + ρ² − 2·r0·ρ·cos φ) ) dφ
```

— a standard result for smearing circularly-symmetric profiles (used e.g.
in PSF-deprojection of globular-cluster/elliptical surface-brightness
profiles). Both inputs (`r0`, `ρ`) and the PSF itself are 1D; the only
"2D-ness" is the `φ` integral, which is a plain 1D numerical quadrature
(a `trapz`-over-a-`φ`-grid fast path, validated against
`scipy.integrate.quad`) — no pixel array involved.

Integrating `C(ρ; r0)` over each observed annulus `[r_edges[i], r_edges[i+1]]`
(again a 1D integral, `2πρ dρ` weighting) gives the fraction of a unit-flux
ring at `r_fine[j]` landing in observed bin `i` — i.e. one entry of a mixing
matrix `R[n_bins, n_fine]`. Precomputed **once** per `(r_fine, r_edges,
psf_r, psf_vals)` — it doesn't depend on `(A1, h1, A2, h2)`, since
convolution is linear. After that, for any trial parameters:

```
fine_flux  = intrinsic_profile(r_fine, A1, h1, A2, h2)   # cheap, analytic, 1D
observed   = R @ fine_flux                                # one matrix-vector product
```

Both the fake-data generator (Method 4) and the fit's forward model (Method
5) call this same `R` and the same `intrinsic_profile`, so generation and
fitting are provably using the identical PSF operator. Validated in
`validate_ring_convolution()` against `scipy.integrate.quad` at several
`(r0, ρ)` points (max relative error ~5e-4 at the default resolution).

### 4. Discretize into synthetic "observations" — ABSOLUTE per-bin noise

`R @ fine_flux` gives the noise-free binned flux. Noise is an **absolute
per-bin sigma array** (`noise_sigma_per_bin`, same units as the model
amplitude, same length as the number of bins) — not a signal-to-noise
RATIO computed on the fly from whatever the current model happens to
predict. Keeping it absolute and fixed means the noise floor doesn't
silently rescale itself if a later cell changes `(A1, h1, A2, h2)` to test
a different truth against the same noise realism.

To get a realistic absolute array in the first place, `sigma_from_target_sn`
converts a target flux-S/N SHAPE (`EXAMPLE_FLUX_SN`, or any other array
matching Austin's real per-bin S/N pattern) into an absolute sigma array,
using ONE reference truth model to set the flux scale — called once, then
that fixed array is reused across every `make_fake_observation` draw /
Monte Carlo run:

```python
def sigma_from_target_sn(reference_params, r_edges, r_fine, psf_r, psf_vals,
                         target_sn_per_bin, R=None) -> tuple[np.ndarray, np.ndarray]:
    """(A1,h1,A2,h2) truth + PSF -> binned model flux -> sigma = |model|/target_sn.
    Returns (sigma_per_bin, R). Call ONCE per (truth, PSF, S/N shape)."""

def make_fake_observation(truth_params, r_edges, r_fine, psf_r, psf_vals,
                          *, noise_sigma_per_bin, R=None, seed=None) -> dict:
    """truth -> intrinsic_profile -> R -> binned flux -> + Gaussian noise
    with the given ABSOLUTE noise_sigma_per_bin -> {"r_edges",
    "total_flux_fid", "total_flux_lo", "total_flux_hi"} -- a boot-shaped
    dict, directly usable by plot_flux_profile once Phase 2 exists."""
```

### 5. Two fits, compared head-to-head (no PSF-fixing tricks)

- **Naive fit** (`fit_naive`): no PSF correction, drops the innermost bin
  (`fit_skip_inner=1`) — the approach specced in
  `flux-profile-exponential-fit.md`. Fits the SAME two-exponential
  functional form directly to the remaining bins, so the only difference
  from the PSF-aware fit is the PSF step + the dropped bin — an
  apples-to-apples comparison, not a different functional form.
- **PSF-aware fit** (`fit_psf_aware`): forward model via `R` (Method 3)
  inside `curve_fit`, fitting `(A1, h1, A2, h2)` against the *full*
  binned+noised data, **including** the innermost bin.
- Both try several seeds and keep the lowest-χ² result (same
  multi-seed-then-best-χ² pattern `speclsf.py`'s double-Gaussian fit
  already uses) — a two-exponential fit on ~9 sparse, non-uniformly spaced
  bins is prone to local minima with only one seed. Bounds are scaled to
  the data itself (amplitude up to 50× the largest bin, scale length up to
  10× the fitted radial span) rather than left at a blanket `(0, inf)`, so
  the optimizer doesn't wander into numerically-degenerate, physically
  meaningless "solutions" — applied identically to both fits.

**Not building `fix_h1`** (Leclercq's trick of fixing the core component's
scale length to an independently-measured continuum value). Only relevant
if the PSF-aware fit turns out to need it to break a real `A1`/`h1`
degeneracy — add it later if that shows up, not preemptively.

**Confirmed result from a first Monte Carlo sweep** (truth `A1=200, h1=8,
A2=60, h2=80` kpc, PSF Moffat β=3 FWHM=3, noise from `EXAMPLE_FLUX_SN`,
40 draws): PSF-aware recovers `h1` (median ≈11 kpc vs. truth 8, std ≈35)
and `h2` (median ≈79 kpc vs. truth 80, std ≈36) with much less bias and
scatter than the naive fit (`h1` median ≈2.6, std ≈89; `h2` median ≈189,
std ≈56) — the naive fit is not just biased but genuinely unstable
draw-to-draw once the correlated inner bin is simply discarded. This is
exactly the comparison the testbed was built to make.

### 6. Diagnostics

**Per-case plot** (`plot_recovery_case`): intrinsic truth curve (thin
dashed line, on `r_fine`), the PSF-convolved+binned "observed" points with
errorbars, the naive-fit overlay, the PSF-aware-fit overlay — log-y, so the
core-bin bias from ignoring the PSF is visible by eye. Bins dropped by the
naive fit are marked with a hollow marker (data still shown, just flagged
as excluded from that fit).

**Injection-recovery comparison, as a double histogram**
(`plot_recovery_histograms`, per Austin's ask, not a table): run
`monte_carlo_recovery` (`make_fake_observation` + both fits, `n_draws`
independent noise realizations at a fixed truth/PSF/`noise_sigma_per_bin`)
and plot the recovered-`h1` distribution from the naive fit (orange)
overlaid with the recovered-`h1` distribution from the PSF-aware fit
(blue), with a vertical dashed line at the injected truth. Same call for
`h2`. Makes both bias (histogram center offset from the truth line) and
precision (spread) visible in one glance, per method, per parameter.

## Implementation (as built)

`01-lya_halos/code/psf_exponential_recovery.py`, standalone (not in
`utils_lya_halo`), functions: `moffat_1d`, `normalize_psf_flux`,
`ring_convolution_matrix` (+ `validate_ring_convolution` regression check),
`intrinsic_profile`, `sigma_from_target_sn`, `make_fake_observation`,
`fit_naive`, `fit_psf_aware`, `plot_recovery_case`, `monte_carlo_recovery`,
`plot_recovery_histograms`. Runs standalone (`python
psf_exponential_recovery.py`) for a quick validation + smoke test, same
convention as `test_framework.py`.

## Success criteria

- `ring_convolution_matrix` agrees with `scipy.integrate.quad` to within
  numerical tolerance — **met** (max relative error ~5e-4 at default
  resolution).
- Given a known truth and `noise_sigma_per_bin`, the PSF-aware fit recovers
  `h1`/`h2` closer to truth, with less scatter, than the naive fit,
  especially for `h1` — **met** in the first Monte Carlo sweep (Method 5).
- The interface is clean enough that Austin's own notebook can vary
  `h1_true`/`h2_true`/PSF FWHM/`noise_sigma_per_bin`/`n_draws` and rerun the
  whole comparison without touching the file — met via the documented
  function signatures above; see the notebook cells provided alongside
  this spec.
- Once satisfied, the winning fit approach (naive vs PSF-aware) feeds back
  into `analysis.py`'s `plot_flux_profile` per
  `flux-profile-exponential-fit.md` (Phase 2, still deferred).

## Post-implementation fixes (2026-07-14)

Two real bugs turned up while testing against a broader synthetic truth and
against Austin's actual `total_flux_med` array — both are numerical/design
bugs, not just precision nitpicks, and both are now fixed in the deployed
`psf_exponential_recovery.py`:

**1. Ring-convolution flux-conservation bug.** `_ring_convolved_value`'s
phi-integral originally used a fixed 121-point grid over the full
`[0, 2pi]`. For a compact PSF and `r0` large compared to the PSF width
(routine here — bins reach ~2000 kpc, PSF is a few kpc), the PSF's angular
support shrinks to a sliver near `phi=0` (width ~`psf_width/r0`) that a
fixed grid completely misses. Caught by a flux-conservation check
(integrate a unit-flux ring's PSF-smeared profile back out to a total,
which must equal 1): it gave ~13 at `r0=1000` kpc, growing roughly linearly
with `r0`. Fixed by computing the exact angular (and radial) support window
in closed form via `arccos`, rather than integrating blindly over the full
range — `validate_ring_convolution` now checks flux conservation at `r0`
up to 2000 kpc, not just small values, so this can't silently regress.

**2. `fit_naive` was comparing the wrong quantity.** It fit
`intrinsic_profile` evaluated directly at each bin's midpoint against the
bin's *integrated* flux — a fine approximation for narrow, similarly-sized
bins, but checked directly against this pipeline's own bin scheme (10 kpc
innermost, 1000 kpc outermost) to be off by up to ~10^9x for the widest
bins: a fast-declining exponential's value at one point badly misrepresents
its integral over a bin many times wider than its own scale length. Fixed
with a new `_bin_integrate` helper (sums the fine-grid flux into whichever
coarse bin it falls in, no PSF redistribution) that `fit_naive` now fits
against instead. `fit_naive`'s signature changed to
`fit_naive(r_mid, r_edges, r_fine, y, yerr, ...)` to support this.

**New helpers added along the way:**
- `default_fine_grid(r_edges, ...)` — builds `r_fine` with resolution
  scaled to each bin's own width (not a single uniform spacing across a
  100x range of bin widths), converging far faster than a plain
  `np.linspace`.
- `estimate_truth_from_profile(r_mid, y)` — a quick, no-optimizer seed
  (A1, h1, A2, h2) from a real profile, meant as a `p0` for `fit_naive`
  (which now does the actual bin-integration-aware calibration), not as a
  final truth on its own.

**Noise input, per Austin's clarification:** `noise_sigma_per_bin` is an
ABSOLUTE per-bin sigma array (not a ratio recomputed from whatever the
model currently predicts) — `sigma_from_target_sn` converts a target
flux-S/N *shape* (e.g. `EXAMPLE_FLUX_SN`, Austin's real per-bin S/N) into
that fixed absolute array once, using one reference truth to set the scale.

**Open, not a bug:** calibrating a clean two-exponential truth against
Austin's real, noisy `total_flux_med` (which spans ~2 orders of magnitude
and may include a non-declining noise floor at large radius) doesn't
converge to a visually perfect match automatically — real data this noisy
may not be perfectly described by a clean two-exponential model. This is
expected and is why `plot_recovery_case` exists: eyeball the comparison and
nudge `truth` by hand if the auto-calibration undershoots, rather than
treating an imperfect automatic fit as a bug.
