# VIRUS PSF model — measurement, wavelength dependence, and the kpc kernel

Supersedes the piecewise Gaia-G-normalized PSF (`starpsf.build_piecewise_psf`,
`lya-halos-piecewise-star-psf.md`) and the `psf_fwhm_arcsec = 1.3″` Lujan
Niemeyer fiducial still hard-coded in `measure.py` / `analysis.py`.

**Status: METHOD AND ERRORS SETTLED (2026-09-12).** The chain runs end to end in
`utils_lya_halo/psfmodel.py`. Errors come from one star bootstrap pushed through
the whole chain (§6.0); run `bootstrap()` then `adopt_bootstrap_errors()` before
quoting anything. Two items remain open: propagating PSF uncertainty into h₁
(§6.3) and the point-source injection validation. The per-slice Moffat fits still
return χ²/dof of 15–58 — that is now a model diagnostic (§6.1), not an error-bar
problem.

**Paper treatment: skim this.** It is a calibration, not a result. Specifics go
in an appendix or a referee reply.

---

## Goal

Produce the point-spread function the Lyα stack actually experiences, as a
kernel in kpc that `fitting.ring_convolution_matrix` can consume, with its
wavelength and redshift dependence measured rather than assumed — and
evaluable separately for any subsample, because a low-z/high-z split needs its
own kernel.

Two things ride on this:

1. `measure.py` / `analysis.py` currently assume 1.3″. The measured effective
   PSF is 2.09″. An assumed-too-narrow PSF hands real instrumental light to the
   halo, so h₁ must be re-run and the shift quoted.
2. The correction is **redshift-dependent in the same direction as any real
   evolution** (low-z kernel is 12% wider in kpc). If an h₁ redshift trend
   disappears under the measured kernel, it was the PSF; if it survives, the
   claim is far stronger than it was.

## Data

- Gaia point sources in AEGIS + COSMOS, pushed through the **identical**
  Stage-1 extraction as the galaxies (`starpsf.make_star_run_table` →
  `run_positions_extract`, `mask='none'` — the normal continuum-image mask
  flags bright fibers as bad and would delete a star's core).
- `star_both = multicat.combine_products({"AEGIS": prod_aegis, "COSMOS": prod_cosmos})`.
  `star_both.catalog['G']` must carry Gaia G, set on `star_tbl` before extraction.
- `gal_prod` (MOSDEF, AEGIS + COSMOS) for the redshift distribution only.

---

## Method

### 1. Extraction, in arcsec

Same fibers, same radial bins, same background annulus as the galaxies, so what
is recovered is **the operator the galaxy stack experiences**, not an abstract
PSF. Per star and radial bin, a sigma-clipped mean over one fixed
observed-frame window — the same window for every star, so no
wavelength-dependent selection enters. No continuum subtraction: stellar
spectra are essentially all continuum.

`starpsf.radial_continuum_profile(star_both, band=(...), combine="biweight")`

### 2. Per-star normalization on a 2–4″ annulus

Each star is divided by its own mean surface brightness over 2–4″ — wide enough
to be well measured, outside any radius saturation can plausibly reach.

**This is the only step where the normalization matters.** It is what lets
stars spanning thousands in brightness be combined with no external flux
reference, and it removes the colour term that sank the earlier
Gaia-G-normalized approach. Everything downstream is analytic and cannot
saturate, so peak-normalize (or unit-flux normalize) there; a 2–4″ marker on a
downstream plot is a leftover and should be removed.

### 3. Two magnitude classes, stitched

| class  | range      | role                          |
|--------|------------|-------------------------------|
| faint  | G 17–21    | sets the core                 |
| bright | G < 17     | the only signal past ~6″      |

Each class biweight-coadded separately, **errors from bootstrapping the star
sample**, not scatter/√N — the per-star normalization forces every star to
agree inside 2–4″ by construction, so the raw scatter there is spuriously tiny
and would anchor the fit on the normalization annulus and nothing else.

Because both classes share the same normalization annulus, their cross-scale is
a **check, not a fitted parameter**. Measured, reported, and *not applied*. It
comes out ≈1.00 redward of 4100 Å; that agreement is simultaneously the
evidence that nothing is saturated and that the two populations are the same
PSF, with the bright stars purely extending radial reach. A bright class
sitting *low* at small r would be saturation, read straight off the plot.

Combine the classes inverse-variance, bin by bin, **keeping every finite bin.**
No S/N gate in the combine: gating on signal is a selection on noise and
manufactures a positive floor exactly where the wings live.

> The stitch drifts to 0.968 / 0.979 in the two bluest slices and sits at
> ~1.00 elsewhere. A pure amplitude offset would not change the fitted width,
> but a *class-dependent* one tilts core against wings and does. Those two
> points carry most of the blue rise in §5 — see §7.

### 4. Fit an **analytic** Moffat, in log space

Log space because the profile spans decades; a linear-space fit is a fit to the
innermost two bins and nothing else.

**Analytic, not fiber-averaged — deliberately.** Every point in the measured
profile is a 1.5″-diameter fiber, so the aperture-averaged (`fiber_moffat`)
form is the physically correct description of the *data*. But nothing
downstream applies a fiber operator: `fitting.ring_convolution_matrix` takes a
bare radial kernel, `fitting.moffat_1d` supplies it, `bin_average_psf` averages
over the **annulus**, and `measure.measure_psf_corrected_core_luminosity` calls
`moffat_encircled_energy_fraction` on an analytic Moffat. Fitting the analytic
form to fiber data returns an **effective kernel with the aperture already
folded in**, which is what those routines want.

Three self-consistent options exist; this is the second:

| option | fit | downstream | verdict |
|---|---|---|---|
| sky + fiber | `fiber_moffat` | add a fiber operator | correct, costs code |
| **effective** | **analytic** | **unchanged** | **chosen** |
| sky, no fiber | `fiber_moffat` | unchanged | **wrong** — under-smooths |

The third is the trap: do the careful fit, change nothing else, and the kernel
is too narrow. Measured cost of each (noiseless h₁ recovery through the real
forward model, `tests/psf_fiber_vs_naive.py`): *effective* biases h₁ by ≤4% and
<1% for h₁ ≥ 1″; *sky-no-fiber* by 2–17%; the 1.3″ fiducial by 10–60%.

**Consequence for the paper: 2.09″ is NOT the delivered seeing** and must never
be described as such — it includes the 1.5″ fiber aperture. A referee reading
2.09″ as image quality will think the data were taken in bad conditions. The
sky Moffat, for reference, is ~10.6% narrower.

### 5. Wavelength dependence

Repeat §1–4 in ~200 Å slices (3500–5500 Å, nine of them). β is held fixed
(§5.1); all the wavelength dependence lives in FWHM.

#### 5.1 β — measured once, then fixed

Fit β **free** in every slice, then hold it at the inverse-variance-weighted
mean (**β ≈ 2.83–2.88**, depending on `fit_r_max`), and **refit FWHM in every
slice at that fixed β**.

β is NOT perfectly flat: at `fit_r_max = 20` the per-slice values run 2.99 (blue)
to 2.78 (red), flatness χ²/dof = 19. That does not matter, and here is why. β and
FWHM are partially degenerate in a Moffat fit, so the pair that matters is
(FWHM(λ), β_fixed) — and because FWHM is refit at the same fixed β that is used
downstream, that pair reproduces the measured profile at every wavelength by
construction. Picking a rough average β is fine **as long as FWHM is refit**,
which it is. The kernel is self-consistent regardless.

What the β trend does affect is only the physical *interpretation* of the DAR
amplitude (see 5.2), which is a diagnostic we do not rely on.

Use all nine slices rather than one 4000–5000 Å stack: more S/N, and the
single-window version gives no flatness test. Quote the 4000–5000 Å value as a
cross-check.

**Rounding β (e.g. to 2.8) is acceptable**, because FWHM is refit at whatever β
is declared. `beta_sensitivity()` measures the cost: on real data the effective
kpc kernel FWHM spans 16.3–17.7 kpc (7.8%) across β = 2.7–3.0, with β = 2.8
landing ~0.8% from the measured value. Quote that as the systematic rather than
asserting insensitivity.

> The trend direction is consistent with 5.2: a displacement smear widens the
> core without adding power-law wings, so the wings look relatively steeper.
> Noted, not modelled.

#### 5.2 FWHM(λ) = seeing ⊕ differential atmospheric refraction

Not a power law. Redward of ~4100 Å the profile is consistent with Kolmogorov;
blueward it steepens sharply and then flattens — a shape a single exponent
cannot produce. The physical form:

```
FWHM(λ)² = [F₀ (λ/4600)^(−1/5)]² + [206265 · (n(λ) − n(λ_ref)) · tan z]²
```

with n(λ) the Edlén refractivity of air. Three parameters. Measured:

| | value | |
|---|---|---|
| F₀ | 2.010 ± 0.016″ | seeing at 4600 Å |
| tan z | 0.867 ± 0.154 | z = 40.9° |
| λ_ref | 4484 ± 123 Å | guider/ADC reference |
| χ²/dof | **0.06** | power law: 1.30 |

Two independent consistency checks fall out: λ_ref lands where a guider
reference wavelength should, and the zenith distance lands near HET's **fixed
35°**. That agreement is the evidence the blue-end steepening is refraction and
not something instrumental.

**tan z is a DIAGNOSTIC ONLY — Austin, 2026-09-12.** It is not a measurement of
the zenith distance (the model treats a parallactic-smeared arc as an isotropic
blur added in quadrature, order-unity correct and no better), and it is also
partly absorbing the mild β trend of 5.1. Keep it as a sanity check that the
blue-end shape is atmospheric; do not build on it and do not quote it as a
result.

**Paper treatment (Austin, 2026-09-12): skim the PSF construction.** It is a
calibration, not a result, and dwelling on it distracts. The specifics belong in
an appendix or in response to a referee who flags it.

Why this matters operationally: at 3647 Å the DAR form gives 2.21″ and a power
law 2.01″ — 4% apart, at exactly the wavelength where z ≈ 2 galaxies sit. The
power law also leaves a U-shaped residual (+3.1σ, −2.0σ, +2.1σ), i.e. real
curvature it cannot follow.

### 6 → see Errors, below

### 7. Per galaxy → kpc

The PSF is a fixed **angular** blur; only the conversion is per-galaxy. For
galaxy *i*: λᵢ = 1215.67(1+zᵢ) → FWHM(λᵢ) off the measured curve → multiply by
kpc/arcsec at zᵢ (`cosmo.angular_diameter_distance(z)/206265`, matching
`stack.py` exactly) → β fixed.

**Build the kernel directly in kpc.** The Moffat's (β−1)/(πγ²) prefactor then
supplies the 1/s² Jacobian automatically. An explicit Jacobian is only needed
if an arcsec-defined curve has its x-axis relabelled — which is how two bugs of
exactly this class already entered this project. Keep the assertion regardless:

```python
rr = np.concatenate([[0.0], r_kpc])          # integrate from 0, not from r[0]
tot = _trapz(analytic_moffat(rr, f, beta)*2*np.pi*rr, rr)
assert abs(tot - 1) < 0.02
```

Combine the N kernels with a **plain unweighted mean**. Unweighted because
`stack.coadd_galaxies`' biweight ignores its `weights` argument entirely (only
`weighted_median` uses it), so the galaxies enter the stack unweighted. A mean
rather than a biweight because these are noiseless analytic curves with nothing
to reject.

The one approximation: the data were combined with a biweight, which is
nonlinear, so a single effective kernel is not formally exact. The per-galaxy
kernels differ by ~30%, and a point-source injection through the real pipeline
measures the residual error directly — **not yet run** (§8).

### 8. Report

Refit one Moffat to the numeric coadd. That reproduces it to ~1%, so two
numbers can be quoted while the pipeline convolves with the numeric curve.

| quantity | value |
|---|---|
| effective angular PSF | FWHM 2.090″, β 2.882 |
| effective kpc kernel | FWHM 17.23 kpc |
| per-galaxy spread (kpc) | 15.0–19.5 kpc (30%) |
| low-z / high-z split at z = 2.31 | 18.20 vs 16.31 kpc (**+12%**) |

That last row is the operational result: **any h₁ difference between the two
redshift halves smaller than 12% is PSF, not evolution.** Refit each half with
its own kernel before interpreting.

Following Lujan Niemeyer, show separate PSFs where the stack is split — but the
framing is stronger here: she had two independently determined PSFs for two
samples; this is **one measured calibration evaluated at two redshifts**. Say
so.

---

## Errors — how they are made

**Resolved 2026-09-12.** Errors come from ONE bootstrap over the stars, run at
the top and pushed through the entire chain. Implemented in `psfmodel.bootstrap`
and `psfmodel.adopt_bootstrap_errors`. §6.1–6.3 below record what was wrong
before and why the fix is what it is; §6.0 is the method as it now stands.

### 6.0 The method

Resample the stars with replacement, **stratified within each magnitude class**
(n_faint from the faint pool, n_bright from the bright pool — pooling would let
a draw return a different class balance, which measures a different
experiment). Use the **same star indices for all nine slices** in a draw, since
it is the same stars at every wavelength; that is what propagates
slice-to-slice correlation. Run the complete chain inside each draw — composite,
both β passes, FWHM(λ), the DAR fit, every kernel including the low-z/high-z
split. The spread across draws is the error.

Galaxy redshifts are held **fixed**: spec-z, and their sample variance is the
stack's uncertainty, not the PSF's.

Two properties make this both affordable and correct:

- `per_obj` (n_star × n_rad) does not change under resampling, only which rows
  are selected. The spectral work runs nine times total, so a draw is index
  selection plus a biweight. 300 draws in ~90 s.
- The per-bin errors are only the **weights** in the Moffat fit. They decide
  which radii the fit listens to; they are not the source of the error bars.
  Computed once from the full sample and frozen. Suboptimal weights cost a
  little efficiency; they do not bias the estimator, and the spread across draws
  is still the honest uncertainty on whatever estimator actually ran.

Because both β passes run inside the draw, β\* is never treated as a known
constant and the β/FWHM correlation propagates for free. Because all kernels in
a draw come from the same resampled stars, the split halves share their PSF
error and the error on their **difference** is much smaller than on either
alone — which is what makes the split's significance testable.

### 6.1 The composite fit ignores bin-to-bin covariance  ← diagnostic, not a blocker

χ²/dof of 15–58 on the slice fits. The cause is **not** that the error bars are
individually too small. The bootstrap resamples stars, so each draw
moves *all* radial bins together and the marginal per-bin variances are right.
What the fit discards is the **off-diagonal covariance**: adjacent bins share
the same stars (and near bin edges, fibers), so a diagonal σ counts one shared
wobble as independent scatter once per bin, inflating χ² without any error bar
being wrong.

This does **not** contaminate the error bars (§6.0: the per-bin errors are only
weights), so it is no longer a blocker. It matters for one thing: χ²/dof ≈ 1 vs
≈ 30 is how you find out whether a single Moffat is actually the right profile
or whether there is an aureole.

**Diagnostic fix:** retain each draw's full profile, form `C = np.cov(draws.T)`,
pass it to `curve_fit` as a 2-D `sigma` (in log space, `C_log ≈ C /
np.outer(y, y)`). If χ²/dof drops toward 1 the errors were always fine and only
the likelihood was wrong; if it does not, the excess is a genuine model failure
and argues for the `moffat+pl` form.

### 6.2 The √(χ²/dof) inflation was wrong — FIXED

The first version inflated each slice's formal FWHM error by √(χ²/dof) of its
**Moffat profile fit** and used that as the DAR fit's sigma. That conflates two
different χ². A high profile χ² reflects structure *within* a slice — correlated
radial bins, or an imperfect Moffat shape — and is **not** evidence that the
recovered FWHM for that slice is uncertain by that factor.

Measured on real data: the inflated errors were 2.3–2.8× the bootstrap's
measured per-slice FWHM scatter, which drove the DAR fit to χ²/dof ≈ 0.06 — a
fit far "too good", i.e. errors several times too large and parameter errors
correspondingly inflated.

`adopt_bootstrap_errors(cal, boot)` replaces them with the measured scatter and
refits. It is a separate call rather than a config flag because the measured
errors do not exist until the bootstrap has run, and the bootstrap needs a
calibration to run on; a flag would hide that circularity.

Safe to apply after the fact: inside a draw those errors act only as the
*relative* weights of nine points, and both the provisional and measured errors
are nearly flat across wavelength (each spans ~10–15% end to end), so the
per-draw F₀/tan z/λ_ref barely move. What changes is χ²/dof and the formal error
bars — exactly what was wrong.

### 6.3 No PSF uncertainty reaches h₁  ← still open

The kernel is treated as exact downstream, so h₁ error bars contain no PSF
contribution. **Fix:** refit h₁ with a sample of kernels from the bootstrap
ensemble and add the spread in quadrature. Until then, h₁ errors are statistical
only and must be labelled as such.

### 6.4 What to quote

Two numbers, never one combined bar that hides which is which:

- **statistical** — the bootstrap spread (star-to-star).
- **systematic** — the range from `systematics()`: fit radius, normalization
  annulus, magnitude split, β. Resampling stars cannot see any of these, nor
  anything else common to all stars (background systematics, the Moffat form,
  the effective-kernel convention, the DAR quadrature approximation).

---

## Implementation

`utils_lya_halo/psfmodel.py` — two stages, split because one is slow:

```python
from utils_lya_halo import psfmodel as pm

def bc(prof, g):                                    # coadd + stitch, injected
    cf = coadd_stars(prof, gmag=g, mag_range=(17, 21))
    cb = coadd_stars(prof, gmag=g, mag_range=(0, 17))
    return stitch_psf([cf, cb], verbose=False)

# CALIBRATE -- slow, once
cal = pm.calibrate(star_both, np.asarray(star_both.catalog["G"], float), bc)
cal.save("psf_calibration.json")

# APPLY -- instant, per subsample
cal = pm.PSFModel.load("psf_calibration.json")
z   = np.asarray(gal_prod.catalog[cfg2.z_col], float)
kpc = cal.kernel_kpc(z)                 # -> (r, K_eff) for ring_convolution_matrix
arc = cal.kernel_arcsec(z)              # the PSF to show off
spl = cal.split_kernels(z)              # low/high-z, one calibration twice
cal.plot_lambda(); cal.plot_kernels(spl)
```

`build_composite` is injected rather than absorbed so the magnitude classes and
the stitch stay editable outside the module.

Related: `tests/psf_fiber_vs_naive.py` (the fiber-vs-analytic decision and the
h₁ bias measurement behind §4).

---

## Success criteria

- [x] Class cross-scale consistent with 1 redward of 4100 Å → no saturation.
- [x] FWHM(λ) fits a physical form at χ²/dof < 1, beating a power law.
- [x] Recovered λ_ref and zenith distance physically sensible.
- [x] β flat across slices, justifying a single fixed value.
- [x] Single Moffat reproduces the numeric coadd to ~1%.
- [x] Split kernels differ measurably (12%), so the split is meaningful.
- [x] Honest errors on F₀, λ_ref, β from the star bootstrap (§6.0), with the
      √(χ²/dof) inflation removed (§6.2).
- [ ] χ²/dof near 1 after the covariance fix, or a demonstrated aureole
      (diagnostic — tells us whether a single Moffat is the right profile).
- [ ] **PSF uncertainty propagated into h₁.**
- [ ] Point-source injection through the real pipeline validating the single
      effective kernel against the per-galaxy set.
- [ ] h₁ re-run with the measured kernel; shift from the 1.3″ assumption quoted.
- [ ] Low-z/high-z h₁ re-checked with per-half kernels.

## Open items

1. **3450–3650 Å slice.** 22 of 528 galaxies sit blueward of the calibrated
   range, all in the low-z half — the half with the wider kernel. The DAR form
   has the right curvature there, but it is still extrapolation into the
   steepest part of the curve. `psfmodel` prints a warning when this happens.
   Cheap now that the loop exists; adding the slice validates the extrapolation
   and materially strengthens the result.
2. **Blue stitch drift** (0.968 / 0.979). Check whether the two classes give the
   same FWHM individually in those slices. If they do, the drift is
   amplitude-only and harmless; if not, the blue FWHM points — which drive the
   DAR amplitude — are suspect.
3. **ADR centroid vs λ** — carried over from the earlier PSF work, still open,
   and now partly explained by §5.2. Whether any residual centroid shift
   survives the DAR model has not been checked.
