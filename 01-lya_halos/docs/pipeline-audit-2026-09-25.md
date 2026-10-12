# Pipeline audit — `utils_lya_halo`

*2026-09-25, revision 3. What changed from revision 2 is listed at the very end.*

---

## Summary

The seven findings that matter most, in the order I'd act on them:

1. **The Niemeyer comparison may be off by ~3×.** The literature notebook built the comparison
   with `rest_density=False`, which now measures SB/(1+z). Before the 09-17 fix, the same setting
   gave 1.5–1.7× too high. → §4.1, issue I2.
2. **`validation.py` is 5× off from the main pipeline.** It still uses the plain pixel sum
   (`flux_sum`). That includes the wavelength-shift tool needed to match Niemeyer's error bars.
   → I1.
3. **New reductions will be served from the old cache.** The cache key doesn't include the fiber
   data or even its folder. Pointing the config at new files changes nothing. → I14.
4. **The production bin edges changed on 09-23.** The injection tests were built for the old edges.
   → I15.
5. **The centroid shrinks toward 0 km/s at low S/N.** A true +150 km/s reads +85 to +125 at S/N 1–3.
   This affects the outer null and the radial gradient. → I4.
6. **Per-galaxy errors are ~2× too small.** This makes `variance_budget` find scatter that isn't
   there. → I3.
7. **The three-zone numbers depend on choices still to be made.** The point-source core, the
   two-halo luminosity reaching to 2 Mpc, and biweight vs mean stacking. → I6, I7, I11.

---

## 1. How the pipeline works

Each stage has four parts: what it does in plain words, pseudocode, what Methods must say, and what
it depends on. "Production" means `code/run_config.py`.

### 1.1 Sample

**In plain words.** The code reads a catalog that is already cut (AGN removed, footprint,
magnitudes) and keeps the rows for one field. After extraction it drops only the galaxies that have
no data at all in the outermost annulus.

```python
catalog   = read("lya_mosdef_catalog_KEEP.fits")        # all sample cuts already applied
run_table = catalog[(FIELD == field) & (CATALOG == "MOSDEF")]

# Stage 2 coverage cut: is there ANY usable spectrum in 1000-2000 kpc?
keep = count(isfinite(SPEC[g, -1, :])) > 100            # cut_radial_bin=-1, min_good_wave=100
```

**Say in Methods.**
- Where `_KEEP.fits` comes from: the AGN cut (X-ray + BPT + IRAC, ~70 of ~510) and the footprint.
  The code applies none of these.
- Redshifts are MOSDEF systemic, used as-is.
- **The coverage cut is loose on purpose.** A galaxy near, or slightly beyond, the footprint edge
  still has fibers in part of its 2000 kpc annulus, so it is kept. Only galaxies with no fibers at all
  in that annulus are dropped. (With `cut_radial_bin=0` instead, galaxies with no fibers at their own
  position would be dropped.)
- **So each radial bin is stacked over a slightly different set of galaxies.** A kept galaxy with no
  fibers within 10 kpc contributes nothing to the inner bins; its NaNs are ignored. Give the number
  of galaxies contributing to each bin, as Niemeyer does ("at least 355 galaxies per bin").

**Depends on.** Footprint edges being unrelated to galaxy properties, which is reasonable. R_vir and
the PSF kernel are averaged over *all* kept galaxies, including ones that never reach the inner bins.
That is a small mismatch, not worth fixing unless the per-bin counts differ a lot.

### 1.2 Fibers and wavelengths

**In plain words.** The fiber spectra are read as they are, sky-subtracted by the VDFI reduction.
The wavelength axis is assumed, not read from the file. It is in air.

```python
spectra = all_flux_final       # (exposures, fibers, 1036), 1e-17 erg/s/cm^2/A per fiber (per A, confirmed)
wave    = linspace(3470, 5540, 1036)     # 2 A pixels, AIR; from config, not from the file
```

**Say in Methods.**
- Air → vacuum happens before dividing by (1+z) (the 08-01 fix, worth 85 km/s).
- No barycentric correction yet. Bounded at ≲30 km/s per exposure, ≲10 km/s radius-dependent after
  stacking.
- No correction for atmospheric dispersion (DAR). Its blur goes into the measured PSF instead (1.9).

### 1.3 Masking

**In plain words.** A fiber is thrown out if its centre lands on a masked pixel of a baked CFHTLS
continuum mask. The mask is deliberately shallow.

```python
bad  = segmentation_mask(fibermask_FIELD.fits, RA, DEC)   # production: "segmap" only
good = ~bad & finite(RA, DEC)
```

**Say in Methods.**
- Mask depth, measured on COSMOS on 09-23: ~99% of sources brighter than g ≈ 24.5 masked, ~50% at
  g ≈ 25.2.
- About a third of MOSDEF targets were bright enough to be masked and were un-masked by hand.

**Depends on.** The mask removes bright *neighbours*. If the outer light is other galaxies' Lyα, the
two-halo level depends on how deep the mask is (I12).

### 1.4 Stage 1: extraction

**In plain words.** For each galaxy, every fiber from every exposure is placed in a radial bin by its
distance. One background spectrum is built from a far annulus and subtracted. Then the fibers in each
bin are combined into one "average fiber" spectrum.

```python
for galaxy in run_table:
    d_kpc = distance(fiber centres, galaxy) * D_A(z)          # proper kpc, Planck18

    # background: one annulus, all exposures pooled
    ann = good & (80" <= d < 90")
    B   = biweight(spectra[ann])                               # per wavelength pixel
    B   = mask_and_interpolate(B, around=1215.67*(1+z), halfwidth=10 A observed)
    B   = gaussian_smooth(B, sigma=30 A)

    # science bins
    for bin in [0, 10, 20, 30, 50, 80, 140, 300, 600, 1000, 2000] kpc:
        fibers_in_bin = good & (d_kpc in bin)
        S[galaxy, bin] = biweight(spectra[fibers_in_bin] - B)  # c = 6
        E[galaxy, bin] = mad_std(...) / sqrt(N)                # a proxy; errors come from 1.7
```

**Say in Methods.**
- Bins are proper kpc, using each galaxy's own scale. A fiber belongs to a bin by its centre.
- Each bin value is a biweight over all fibers, i.e. a surface brightness per fiber.
- The background is one pooled 80–90″ annulus (~650–750 kpc at z = 2.3). The galaxy's own Lyα is
  masked at ±10 Å observed, bridged, then smoothed at σ = 30 Å.
- **What the background removes:** anything smooth over ~30 Å and uniform over ~80″.
- **What it leaves:** real Lyα at the annulus (masked by design, settled) and narrow sky-line
  residuals (they average out across redshifts).
- **The mean diffuse Lyα is already gone.** The sky subtraction and this annulus remove it by
  construction (your 08-31 note).

**Depends on.** The ±10 Å mask being wide enough at every redshift (I9). The upstream sky model not
removing large-scale structure (§4, open question).

### 1.5 Stage 2: units, rest frame, stack

**In plain words.** Each spectrum becomes a luminosity per kpc² in the galaxy's rest frame, is
placed on one shared rest-wavelength grid, and the galaxies are combined pixel by pixel.

```python
y   = S * 1e-17 * 4*pi*D_L**2 * (1+z) / fiber_area_kpc2     # erg/s/A_rest/kpc^2
lam = air_to_vac(wave) / (1+z)
Y   = interpolate(y, lam -> 1100..1700 A in 0.2 A steps)     # errors interpolated too
stack[bin] = biweight over galaxies of Y[:, bin]              # unweighted
```

**Say in Methods.**
- The (1+z) makes a sum over rest Å equal the observed line flux. The luminosity-distance factor
  removes cosmological dimming, so the stack is in intrinsic units.
- The 0.2 Å grid oversamples the native pixels 2.3–3.5×, so neighbouring pixels are correlated. The
  bootstrap handles this for the stack; single-spectrum error sums do not (I3).
- The stack is a biweight: a robust "typical galaxy", not a mean (I11).
- In these units, noise grows as (1+z)⁵. A z = 3.2 galaxy is ~6× noisier than a z = 1.9 one, so the
  high-z half sets most of the stack's noise.

### 1.6 Stage 3: measuring the line

**In plain words.** On each stacked spectrum: subtract a flat continuum, add up the flux within
±3 Å of Lyα, and find the wavelength that splits the positive flux in half.

```python
cont = median(stack[1195-1205 A or 1225-1236 A])
y    = stack - cont
win  = |lam - 1215.67| <= 3 A                   # +/- 740 km/s about systemic
flux = sum(y[win]) * 0.2 A                      # erg/s/kpc^2
yp   = max(y[win], 0)                           # negatives set to zero
v    = velocity where cumulative(yp) crosses 50%     # the "median50" centroid
```

**Say in Methods.**
- Continuum: flat median of two line-free sidebands. The method sweep is closed; one sentence.
- Window: fixed and symmetric; its compression is forward-modelled by injection.
- Centroid: median of the positive-clipped flux. With no line it returns the window centre, 0 km/s.

**Depends on.** The line sitting well inside the window (I8), and enough S/N (I4).

### 1.7 Errors

**In plain words.** Resample the galaxies 2000 times, re-stack, re-measure. The same resample is used
for every bin, so bins stay correlated as they really are.

```python
fid = measure(stack of all galaxies)
for b in range(2000):
    pick = random galaxies with replacement            # seed 1
    draws[b] = measure(stack of picked galaxies)
plot fid with the 16th-84th percentile of draws
```

**Say in Methods.**
- This captures galaxy-to-galaxy variation.
- It cannot see anything shared by all galaxies: a common background or sky error, and the fact that
  outer annuli of different galaxies reuse the same fibers.
- Outer-bin correlation is measured at ~30% (settled 08-25).

### 1.8 Virial radius

**In plain words.** Each galaxy's stellar mass gives a halo mass through Girelli+2020, which gives
R200c. The sample value is the biweight across galaxies.

```python
Mh[g]  = invert Girelli+2020(M*[g], z[g])        # M200c; log M* clipped to 7-11
R[g]   = (G * Mh / (100 * H(z)^2))^(1/3)          # proper kpc
R_vir  = biweight(R)
band   = bootstrap with mass errors (MASS_16/84) + SMHM scatter (0.2 dex placeholder)
```

**Say in Methods.**
- R_vir means R200c (Niemeyer uses the Bryan–Norman virial radius).
- Quote the SMHM-relation band (±8–12%).
- Set a real value for the scatter (I13).
- Check: log M* = 10 at z = 2.3 gives R200c = 79 kpc.

### 1.9 PSF

**In plain words.** Gaia stars go through the same extraction. Their radial profiles are fitted with
a Moffat, band by band. The width's wavelength dependence is fitted as seeing plus atmospheric
dispersion. Each galaxy gets the PSF at its own Lyα wavelength, and the kernels are averaged.

```python
for band in 9 wavelength windows:
    profile = normalise each star at 2-4", biweight faint and bright stars, combine
    fit Moffat(FWHM, beta) to profile at bin midpoints
beta = average over bands; refit FWHM with beta fixed
FWHM(lam) = seeing(lam) (+) dispersion(lam)
kernel = mean over galaxies of Moffat(FWHM(1215.67(1+z_g)) in kpc, beta)
```

**Say in Methods.**
- Effective FWHM 2.090″, β = 2.882, 17.23 kpc.
- It is an *effective* PSF: the fiber size and the dispersion blur are included. It is not the seeing.

**Depends on.**
- One Moffat shape fitting out to large radius (§6.1 in the PSF spec is still open).
- Averaging kernels standing in for a biweight stack. The point-source injection that would test this
  has not been run.
- Stars and galaxies being masked the same way. They are not (I10).

### 1.10 Profile fit

**In plain words.** The model is an exponential halo plus a cored power law. It is blurred by the
PSF and averaged over each annulus, then fitted to the flux profile. The crossover is where the two
terms are equal.

```python
I(r) = A1 exp(-r/h1) + A2 (1 + (r/r_c)^2)^(-gamma/2)     # A2 < A1
model[bin] = annulus average of (I convolved with the PSF)
fit A1, h1, A2 with gamma = 0.8 and r_c = 400 kpc FIXED
r_x = radius where the two terms are equal
errors: refit every bootstrap draw, or MCMC with r_c and gamma free
```

**Say in Methods.**
- The model, and which parameters are fixed.
- That r_x depends on the fixed outer shape (I5).
- That h₁ is the halo scale length in the literature sense.

### 1.11 Zone luminosities

**In plain words.** Each bin's surface brightness times its annulus area gives a luminosity. Bins
are assigned to zones by their midpoint. The core gets a point-source aperture correction.

```python
L[bin]   = flux[bin] * annulus_area[bin]
one_halo = sum of L over bins with midpoint <= r_x (excluding the core bin)
two_halo = sum of L over bins with midpoint >  r_x (up to 2000 kpc)
core     = inner-bin flux * area / (fraction of a point source inside 10 kpc)
```

**Say in Methods.**
- The zone edge is a bin edge, not r_x itself. For r_x = 70 kpc it is 80 kpc with the current edges
  and 60 kpc with the older ones.
- Two-halo luminosity means L(< 2000 kpc).

**Depends on.** The core being unresolved (I6).

---

## 2. Where a model or a choice sets a number

| Assumption | Affects | Status |
|---|---|---|
| Girelli+2020 SMHM, M200c | R_vir, the headline comparison | ±8–12% band known; scatter placeholder (I13) |
| One Moffat PSF, fixed β | h₁, r_x, core | aureole question and point-source injection open |
| Biweight stack | every amplitude, any budget | compare with mean (I11) |
| Exponential + cored power law, γ and r_c fixed | r_x, A₂ | MCMC exists; add r_x to it (I5) |
| Core is a point source | core luminosity, halo fraction | your call (I6) |
| Line inside ±740 km/s | profile shape, centroid | I8 |
| Centroid unaffected by S/N | outer null, gradient | I4 |
| ±10 Å mask wide enough | outer level | I9 |
| Mask depth g ≈ 25 | two-halo level | I12 |
| Bootstrap captures outer errors | outer error bars, A₂ | 30% covariance known |
| Upstream sky keeps large-scale structure | outer profile | question for Greg (§4) |

---

## 3. Issues

Every issue follows the same four parts: **Problem**, **Why it matters**, **Evidence**, **What to do**.
*(measured)* means I ran the real module code to get the number.

### Verified bugs

#### I1. `validation.py` is 5× off from the main pipeline

- **Problem.** Since 09-17 the main pipeline integrates flux as Σ y × 0.2 Å. `validation.py` still
  uses `flux_sum` (no × 0.2), and its comments say this "matches the bootstrap convention."
- **Why it matters.**
  - `error_cross_check` will report a jackknife/bootstrap ratio of ≈5 when the two agree.
  - The z-scramble plot puts the bootstrap band 5× below the data point.
  - `bootstrap_flux_profile`, `sweep_flux_window` and `gaussian_flux_window` give amplitudes and
    luminosities 5× the pipeline's.
  - The wavelength-shift tool (`placebo_wavelengths`) is what reproduces Niemeyer's error bars.
- **Evidence.** Code reading; the ratio is exactly 1/0.2.
- **What to do.** Switch those calls to `flux_integral` and fix the comments.

#### I2. The Niemeyer comparison measures SB/(1+z)

- **Problem.** The 08-03 copy of `2_Validate_Literature` uses `flux_unit="flux_arcsec2"` with
  `rest_density=False`. The spectrum stays per *observed* Å but is summed over *rest* Å, which
  divides the flux by (1+z).
- **Why it matters.** Any agreement found before 09-17 was with a curve 1.5–1.7× too high. After the
  fix, the same setting gives a curve ~3× too low.
- **Evidence** *(measured, a line of known flux through the real `stack.py` + `measure.py`)*:

  | z | `rest_density=True` | `False`, now | `False`, before 09-17 |
  |---|---|---|---|
  | 1.9 | 1.00 | 0.35 | 1.72 |
  | 2.1 | 1.00 | 0.32 | 1.61 |
  | 2.35 | 1.00 | 0.30 | 1.49 |

- **What to do.** Check the live notebook (the copy on disk may be old). Use `rest_density=True`.

#### I3. Per-galaxy errors are ~2× too small

- **Problem.** `integrated_line_flux` adds up errors on the 0.2 Å grid as if each pixel were
  independent. Each native pixel is spread over 2–3.5 grid points, so it is counted several times.
- **Why it matters.** `variance_budget` subtracts the noise from the observed scatter to find the
  "real" scatter. Noise that is too small leaves fake real scatter behind. This feeds the
  per-galaxy regression screening for Paper I. The stacked error bars are not affected.
- **Evidence** *(measured)*: true noise ÷ reported noise = 2.18 at z = 1.9, 1.96 at z = 2.3, 1.77
  at z = 3.2.
- **What to do.** Sum errors on the native pixels. Re-run any `variance_budget` used to rank splits.

### Things that move headline numbers

#### I4. The centroid shrinks toward 0 at low S/N

- **Problem.** With no line, the estimator returns the window centre (0 km/s). With a weak line, it
  lands partway there. The systematics spec (§1.5) already knew the direction and asked for the size.
- **Why it matters.**
  - The outer bins read "≈ 0" partly because of the estimator.
  - S/N falls outward, so a constant velocity would appear to decline outward.
  - The injection response matrix was built from bright sources, so it likely misses this.
- **Evidence** *(measured; Gaussian line, width ≈ resolution plus redshift error; median of 3000
  noise realizations)*:

  | true velocity | S/N 1 | 2 | 3 | 5 | 10 | 30 |
  |---|---|---|---|---|---|---|
  | +150 km/s | +85 | +112 | +125 | +134 | +140 | +143 |
  | +50 km/s | +36 | +40 | +45 | +44 | +46 | +48 |
  | 0 km/s | +4 | −3 | +3 | +2 | +2 | 0 |

- **What to do.** Measure each bin's S/N, read off its compression, and quote the outer bins
  corrected. Show the z-scramble centroid spread next to them.

#### I5. The crossover depends on the fixed outer shape

- **Problem.** With r_c = 400 kpc, the outer term is almost flat inside 100 kpc. So r_x is set by h₁
  and the amplitude ratio. The code's own scan shows r_x moving 20–26% for r_c from 200 to 700 kpc.
- **Why it matters.** That is larger than the ±8–12% SMHM band. The MCMC frees r_c and γ, but does
  not report r_x.
- **What to do.** Add r_x to the MCMC's derived quantities (one root-find per sample). Report the
  prior bounds with it.

#### I6. The core is corrected as a point source, but the model has none

- **Problem.** The core luminosity divides the inner bin by the point-source fraction inside 10 kpc.
  The fitted model says the inner light is a resolved exponential.
- **Why it matters.** The core/halo split (the halo fraction) is a Paper I result. A pure exponential
  with no compact part would still get a large "core." The correction factor is 1.42 at the old 1.3″
  default and 2.31 at the measured 2.09″ *(measured)*.
- **What to do (your call).** If there is a compact Lyα component, add it to the model and take the
  core from that term. If not, take the core from the fitted model.

#### I7. The two-halo luminosity is dominated by the last bin

- **Problem.** It sums every bin out to 2000 kpc. The 1000–2000 kpc bin has 33–43× the area of the
  bins near 200 kpc. The curve of growth leaves that bin out; the zone luminosity doesn't.
- **Why it matters.** The noisiest bin, which is also the most sensitive to any zero-point error,
  sets L₂ₕ and every ratio built on it.
- **What to do.** Quote L(< R) at a stated R, with the zero-point systematic.
- **Question.** In the photon budget, is the beyond-R_vir light counted as this galaxy's own? The
  two-halo budget chain found the galaxy can't supply it. If it is neighbours' light, subtracting it
  double-counts.

#### I8. The flux window loses more of the core line

- **Problem.** The core line sits at +265 km/s; the outer line near 0. A fixed ±740 km/s window
  catches less of the first.
- **Evidence** *(measured, Gaussian with σ 250–300 km/s)*: 4–7% lost at +300 km/s, 0.3–1.4% at 0.
  With grism redshifts: 7–21% (§4).
- **What to do.** Include it in the injection flux criteria, or state it as a correction.

#### I9. The background Lyα mask narrows with redshift

- **Problem.** ±10 Å observed is ±850 km/s at z = 1.9 but ±560 km/s at z = 3.4.
- **Also:** the 08-17 changelog says the mask was widened to 15 Å, but every run file still has
  10 Å. The referee register says "≈ 6.6 Å rest", which is the full width, not the half-width.
- **What to do.** Pick one value (a fixed rest-frame width removes the z dependence), use it for both
  injection and data, and fix the register.

#### I10. Stars and galaxies are not extracted the same way; old PSF constants remain

- **Stars use no mask; galaxies use the segmap.** Neighbour light enters the stars' outer bins and
  their background annulus, which affects the wings.
- **1.3″ is still the default in:**
  - `measure.py` 2268, 2711
  - `analysis.py` 703, 1432, 1904
  - `splits.py` 1360, 1365
  - `validation.py` 4384
- **The core functions take one FWHM number,** so they can't use each galaxy's own PSF.
- **Checked and fine** *(measured)*: the biweight in a bright star's steep inner bins moves the fitted
  FWHM by less than 1%.

### Things to settle before later work

#### I11. The stack is a biweight, not a mean

- **Problem.** A biweight down-weights unusually bright galaxies. In the core that probably means the
  LAEs.
- **Why it matters.** A photon budget compares a *mean* intrinsic luminosity with what is recovered.
  A biweight recovered side would lean toward "destroyed."
- **What to do.** The mean stack already exists. Compare h₁ and r_x between the two.

#### I12. The two-halo level depends on the mask depth

- **Problem.** Fibers on sources brighter than g ≈ 25 are removed everywhere. Some of those are the
  neighbours whose Lyα is the two-halo term.
- **What to do.** Either state it as part of the definition, or measure it with one no-mask run. It
  is also a key difference from Niemeyer (§4).

#### I13. The SMHM scatter placeholder should be about half its value

- **Problem.** The code scatters halo mass at fixed stellar mass. Literature quotes the reverse. The
  two differ by the slope of the relation.
- **Evidence** *(measured, Girelli at z = 2.3)*:

  | log M* | slope | 0.2 dex in M* becomes |
  |---|---|---|
  | 9.5 | 1.96 | 0.10 dex in M_h |
  | 10.0 | 1.81 | 0.11 dex |
  | 10.5 | 1.48 | 0.14 dex |

- **What to do.** If the literature value is 0.2 dex in M*, use ~0.11.

#### I14. New reductions will be served from the old cache

- **Problem.** The cache key includes neither the fiber data nor its folder (`vdfi_base_path` is not
  hashed). `SRCFILE` is written to the header but never checked. New reductions at the same path, or
  at a new path, produce the same key and the same galaxy FITS name, so the old spectra come back.
- **Also:**
  - The wavelength grid is assumed, not read.
  - `load_for_stack(field="BOTH")` silently falls back to one field if the other file is missing,
    which is exactly what happens after a key change.
- **What to do before the first re-extraction on new data.**
  - Add the flux file's path, size and date (or a header hash) to the key.
  - Read the grid from the file.
  - Make the BOTH fallback an error.

#### I15. The production bin edges changed on 09-23

- **Problem.** `run_config.py` now uses `[0, 10, 20, 30, 50, 80, 140, 300, 600, 1000, 2000]`. The
  injection tests, including criterion 1c (the bin holding R_vir) and 1d, were built for
  `[0, 10, 16, 24, 36, 60, 100, 180, 350, 600, 1000, 2000]`. There is no changelog entry.
- **Question.** Is the new list production or a test? If production, the bin-specific injection
  criteria need redoing. If a test, revert it before the next extraction.

---

## 4. The Niemeyer 2022 comparison

### 4.1 What is being compared (confirmed)

**The anchor:** Lujan Niemeyer et al. 2022b (arXiv:2207.11098). 1034 [O III]-selected 3D-HST
galaxies, 1.9 < z < 2.35, median log M* 9.3. Median-stacked, background-subtracted, observed Lyα
surface brightness out to 800 kpc (their Fig. 1).

**Shared:** most of the galaxies (ours is the same sample without GOODS-N), and the instrument.

**Tested:** independent data, reduction, background subtraction and pipeline code.

It anchors the pipeline, not the MOSDEF result directly; MOSDEF is a different, more massive sample
with systemic redshifts. The cleanest way to bridge the two: stack the 81 galaxies in both MOSDEF
and Bowman twice, once with each redshift. The difference is the redshift-source effect, measured.

### 4.2 How to judge "agreement"

- Each curve's error mixes two things: **which galaxies** are in the sample, and **measurement
  noise**.
- For shared galaxies, the first is identical in both curves and cancels in their difference.
- So adding the two error bars in quadrature makes agreement too easy, especially in the core, where
  galaxies differ most.
- **The unshared parts are real differences:**
  - GOODS-N (roughly a quarter of her sample).
  - Her weighting by number of observations (1–15 per galaxy).
  - Any difference in redshift cuts.
- **Cheap fix:** bootstrap over our exposures with the galaxy list fixed. That measures only our
  noise, which is the right bar for the difference.
- **Best fix:** get her galaxy list (and per-galaxy profiles if possible) and compare the same
  galaxies.

### 4.3 The differences, sorted

**Conventions — match these to hers, or they blur the test:**

| Step | Hers | Ours | Effect |
|---|---|---|---|
| Units | observed SB | `rest_density=False` in the notebook | ×0.30–0.35 now (I2) |
| Bins | 0, 5, 10, 15, 20, 30, 40, 60, 80, 160, 320, 800 kpc | same in the notebook ✓ | keep |
| Window | ±10 Å observed (±736–850 km/s) | ±3 Å rest (±740 km/s) | small; ±3.2 Å rest matches at z = 2.1 |
| Fiber combine | median, per observation | biweight, all exposures pooled | small, mostly in the core |
| Galaxy combine | median, per galaxy-observation | biweight, per galaxy | small, mostly in the core |
| Errors | larger of bootstrap and wavelength-shift scatter (≈1.4× bootstrap) | bootstrap only | ours up to 1.4× tighter |

**Under test — leave as ours:**

| Step | Hers | Ours | Effect |
|---|---|---|---|
| Upstream sky | full-frame sky from half the fibers across 18′ | VDFI reduction, **undocumented** | if VDFI's sky is local, large-scale emission is removed before we see it. Ask Greg. |
| Seeing | exposures with FWHM < 1.7″; PSF 1.47″ Moffat + fiber ≈ **1.92″** effective, β = 2.2 | all exposures; **2.09″**, β = 2.88 | inner 40 kpc differs by 5–30% (below) |
| Data and code | HETDEX DR3 | VDFI | what we want to test |

PSF alone, ours ÷ hers per bin *(measured, ring convolution)*:

| bin (kpc) | 0–5 | 5–10 | 10–15 | 15–20 | 20–30 | 30–40 | 40–60 | 60–80 | 80–160 |
|---|---|---|---|---|---|---|---|---|---|
| halo (h = 10 kpc) | 0.92 | 0.94 | 0.96 | 0.99 | 1.02 | 1.05 | 1.05 | 0.99 | 0.96 |
| point source | 0.82 | 0.88 | 1.04 | 1.22 | 1.30 | 1.21 | 0.98 | 0.69 | 0.80 |

**Needs your call:**

| Step | Hers | Ours | Why it's hard |
|---|---|---|---|
| Zero point | subtract one constant: median SB beyond 800 kpc = 1.51×10⁻¹⁹ | per-galaxy background at 80–90″ with Lyα bridged | Her constant is 3.5–11× her signal beyond 100 kpc, so it dominates the outer comparison. Running both ways gives the systematic. |
| Masking | none; continuum subtracted fiber by fiber | segmap (g ≈ 25) | Only MOSDEF targets were un-masked, so some [O III] galaxies may have their own cores masked. Neighbour masking also lowers our outer level. |

### 4.4 What could make the curves agree for the wrong reason

1. **Units** — a factor of ~3 either way, depending on when the comparison was run.
2. **Zero point** — two curves with different zeros can agree at one radius by accident.
3. **Masked cores** — could hide or imitate the PSF difference.
4. **PSF** — 5–30% in the inner 40 kpc.
5. **Sky model and neighbour masking** — both lower our outer level by amounts not yet known.

### 4.5 Recipe

```python
cfg_n22 = replace(cfg,
    catalog="3DHST", catalog_path=<Bowman, 1.9 < z < 2.35>,
    bin_mode="kpc",
    bins=[0, 5, 10, 15, 20, 30, 40, 60, 80, 160, 320, 800, 1500],   # extra bin for her zero point
    flux_unit="flux_arcsec2", rest_density=True,                    # observed SB
    line_window=[1215.67 - 3.2, 1215.67 + 3.2],                     # = +/-10 A observed at z = 2.1
    fiber_combine_method="median",
    galaxy_combine_methods=["median", "biweight"], measure_stack_method="median",
    mask_method=<segmap with 3D-HST targets un-masked; plus one no-mask run>,
)
boot = run_measure(cfg_n22, run_stack(cfg_n22))

# her zero point, applied to our data (report both versions)
SB_hers_style = boot["total_flux_fid"][:-1] - boot["total_flux_fid"][-1]

# her error convention (after I1 is fixed)
sig = maximum(bootstrap half-width, placebo_wavelengths(cfg_n22, stacks)["flux_null_std"])

# PSF: compare r >= 40 kpc directly; inside that, blur her fitted model with our PSF first
```

Only the no-mask run and the 800–1500 kpc bin need a new Stage-1 extraction of the 3D-HST catalog.
Everything else is Stage 2/3.

### 4.6 What a clean comparison can claim

- **Can:** two independent datasets, reductions and codes give the same profile for (mostly) the same
  galaxies, within a stated systematic.
- **Cannot:** check our zero point beyond ~100 kpc, because hers is a convention too.
- **Cannot:** stand in for the MOSDEF result. That needs the 81-galaxy two-redshift test.

---

## 5. Suggested order

1. **I1** — fix `flux_sum` in `validation.py`.
2. **I2** — rerun the Niemeyer comparison with `rest_density=True`.
3. **I15, I14** — settle the bin edges; guard the cache before new data.
4. **I4, I5** — per-bin centroid compression; r_x from the MCMC.
5. **I3** — native-pixel errors.
6. **I6, I7, I11, I12** — three-zone and budget definitions.

## Sources

- Lujan Niemeyer et al. 2022b, Lyα halos around [O III]-selected galaxies in HETDEX —
  https://arxiv.org/abs/2207.11098
- Lujan Niemeyer et al. 2022a, Lyα halos out to 320 kpc in HETDEX (method details) —
  https://arxiv.org/abs/2203.04826

## What changed in revision 3

- **Coverage cut corrected (1.1).** It drops only galaxies with no data in the outermost annulus;
  edge galaxies are kept with partial coverage. Revision 2 had this backwards.
- **Cache issue corrected and upgraded (I14).** Revision 2 said the fiber-data *path* was in the
  cache key. It isn't, so even pointing the config at a new folder reuses the old cache.
- **Rewritten for readability:** a summary up front, one template per stage and per issue, shorter
  tables, and the Niemeyer differences sorted into match / test / decide.
