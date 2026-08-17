# Single-galaxy measurement pipeline — `single.py` expansion

Design doc for turning `utils_lya_halo/single.py` from a per-galaxy
*visualization* helper into a per-galaxy *measurement* module: core + halo flux
and the halo fraction for one object at a time, plus 2D narrowband imaging of
individual galaxies and stacks of those images.

**Status: DESIGN, not implemented.** Written 2026-08-04 from the discussion of
the same date. Parts A–C are the build. Part D is a natural extension once C
works. Part E is explicitly exploratory and deliberately small.

**Everything here goes in `single.py`.** No new modules. If `single.py` outgrows
its name we rename it, we do not spin off a sibling.

---

## Goal

The stack tells us what the average galaxy's Lyα halo looks like. It cannot tell
us how much *object-to-object* variation is hiding inside that average, and it
cannot show us anything that isn't azimuthally symmetric — because azimuthal
symmetry is imposed by the radial binning, not measured.

Two things follow, and they are the two halves of this spec:

1. **A per-galaxy number.** For each galaxy: core flux, halo flux out to the
   stack's boundary radius, and the **halo fraction** = halo / (core + halo).
   That converts "the average galaxy has an extended halo" into a distribution,
   which is a distribution we can then split on (mass, z, environment) using the
   existing `splits` recipes.

2. **A per-galaxy image.** Collapse each fiber to one brightness value in a
   narrow window around Lyα at that galaxy's redshift, grid the fibers onto a
   regular kpc grid, and look at it. No radial-symmetry assumption anywhere in
   that chain. This is how we would ever see an asymmetry, a companion, a
   disturbed morphology, or a bridge between two nearby objects.

Neither is expected to be high-S/N per object. That is stated up front and
handled explicitly (see "S/N reality check" below), not discovered later.

---

## Data

- VDFI field fiber data via `extract.load_field(config)` — RA/DEC per exposure
  per fiber, flux, error, per-exposure seeing, DAR.
- The Stage-1 binned product (`io.GalaxyProduct`) for the annular path.
- The catalog row per galaxy: position, z, mass.
- Existing background definition: the `[bg_inner, bg_outer)` annulus, unchanged.

### Field loading is the caller's job — NOT wrapped

**Every function in this module takes an already-loaded `vdfi` dict as an
argument.** The field load is slow (memmap open + info read) and is run *once*,
by Austin, in the first notebook cell:

```python
vdfi = extract.load_field(cfg)                 # cell 1, run once
img  = single.galaxy_image(cfg, vdfi, row)     # cell 2+, run per galaxy, fast
```

Do **not** write a convenience function that loads the field internally, and do
not write an "do everything for all galaxies" driver that hides the load. Per-galaxy
work after the load is sub-second to seconds; batching over the catalog is a thin
loop the notebook can write, and keeping it in the notebook is what makes it
possible to inspect one galaxy at a time.

`vdfi['PSF']` is `None` unless the field was loaded with `build_psf=True`
(`core.load_field_with_psf`). Functions that need PSF weights must check and
raise a clear error, not silently produce NaN.

---

## S/N reality check — read this before designing around the numbers

A single galaxy's per-bin S/N is roughly the stack's divided by √N_gal. That is
already stated in `single.py`'s own module docstring and it is the governing
constraint on this entire spec.

Consequences, all of which are designed for below rather than discovered later:

- The halo fraction will be a usable number for the **brighter part of the
  sample only**. Every function returns per-galaxy S/N alongside the ratio, and
  there is an explicit S/N floor below which the row is flagged. That floor is a
  parameter, and whatever value the analysis ends up using is a *stated sample
  cut* in the paper, not a silent filter.
- Individual 2D images are a **diagnostic and figure product** for bright objects
  (AGN, strong LAEs, double-peakers) — not a measurement product for typical
  galaxies. The stacked image (Part D) is where the science number lives.
- PSF correction on a single galaxy must be applied as a **forward** operation
  (convolve a model, compare to data) rather than an **inverse** one
  (deconvolve the data) wherever there's a choice. Inverse correction at
  S/N ≈ 1–2 per bin turns noise into apparent structure.

---

## PART A — per-galaxy core + halo luminosity and the halo fraction

### A1. Two flux paths, halo is the default

- **Halo/annular path (DEFAULT).** Reuse `single_galaxy_spectra`, which already
  pulls one galaxy's binned spectra and applies the same unit + (1+z) +
  rest-frame resample the stack uses. Integrate the line in each bin.
- **Core path (OPT-IN).** The PSF-weighted optimal extraction in
  `core.extract_core_spectra`. Already implemented — this is a **join, not a
  reimplementation**.

Exposed as a parameter on one function, not as two parallel functions:
`core_method='aperture'` (default) | `'psf_weighted'` | `'both'`.

### A2. The leak problem — the single most important correctness issue here

PSF-correcting the core bin means asserting "this much flux leaked *out* of the
core aperture." That leaked flux is currently **still sitting in the outer
bins**. So the naive computation

```
total = core_corrected + halo_as_measured        # WRONG
```

double-counts the leaked light, inflating the denominator and biasing the halo
fraction low. `measure.py:2271` documents exactly this failure for the stack-level
version, and `measure.measure_psf_corrected_core_onehalo_luminosity` exists
specifically to do both zones in one leak-aware pass.

**The single-galaxy version must be leak-aware in the same way.** Flux corrected
out of the core is subtracted from the outer zone before summing. This is not
optional and is not a refinement — the ratio is meaningless without it.

### A3. The numbers produced, per galaxy

| quantity | definition |
|---|---|
| `core_flux` | inner region, PSF/aperture corrected |
| `halo_flux` | boundary bin → `core_boundary_kpc`, leak-corrected |
| `total_flux` | `core_flux + halo_flux` (leak-aware, so no double count) |
| `halo_fraction` | `halo_flux / total_flux` |
| `core_fraction` | `core_flux / total_flux` (= 1 − halo_fraction) |
| `core_extendedness` | `F_aperture / F_psf_weighted` for the core |
| `sn_core`, `sn_halo` | per-zone S/N |
| `psf_fwhm_used`, `psf_beta_used`, `psf_provenance` | what PSF this row used |
| `boundary_kpc_used` | the boundary radius this row used |

Fractions over the sum (rather than halo/core) because they're bounded 0–1, sum
to 1, and don't blow up when the core is faint — Austin's call and the right one.

`core_extendedness` is a bonus: PSF-weighted optimal extraction is unbiased only
for a *point* source, so if Lyα is genuinely extended in the core it undercounts.
The ratio of the two estimators is therefore a model-light indicator of how
extended the core is. Costs nothing extra when `core_method='both'`.

### A4. The boundary radius must not get buried

`core_boundary_kpc = 70.0` becomes a **named field in `config.py`**, with a
docstring stating it is the fitted `expcore` core/two-halo boundary from the
stack and **must be re-derived whenever that fit changes**. Functions take
`boundary_kpc=None` → falls back to config. The value actually used is written
into every returned dict and every output FITS header, so it travels with the
product instead of living in one function body.

Registered in `guide.py` like everything else.

---

## PART B — PSF standardization

### B1. The decision (per Austin, 2026-08-04)

- **Default: the measured VDFI per-exposure seeing.** This is real data and it's
  already loaded (`vdfi['seeing']`, HDU 5), wavelength-corrected via
  `extract.wavelength_corrected_seeing` (∝ λ^(-1/5)).
- **Global override: an analytic Moffat, β = 3.0, FWHM = 1.8″.** Deliberately a
  slightly *poorer* seeing than the Niemeyer (2022) 1.2–1.4″ fiducial, on the
  grounds that 1.8″ is more realistic for this data. Used when we want one fixed
  PSF across the sample rather than per-galaxy variation.

For single-galaxy work the measured path means each galaxy gets **its own
fiber-weighted seeing**, which is strictly better than a sample-wide constant —
the galaxy's seeing is determined by which exposures actually cover it.

### B2. What "standardizing the PSF" actually requires

It cannot be a bare module constant like `LYA_REST`, because seeing is measured
per exposure and varies with wavelength. The standardizable pieces are:

```python
@dataclass
class PSFSpec:
    provenance: str      # 'measured' (default) | 'fiducial'
    beta: float          # Moffat index, 3.0
    fwhm_arcsec: float   # 1.8, used only when provenance == 'fiducial'
    wave_exponent: float # -1/5, the seeing-vs-lambda correction
```

One object on `config`, threaded through every call site, recorded in output
headers. The `provenance` flag is the important part: it makes "which PSF
produced this number" a property of the product rather than something to
reconstruct from git history.

### B3. Current state of the code — four disagreeing definitions

Audited 2026-08-04:

| location | Moffat index | FWHM |
|---|---|---|
| `extract.moffat_psf_integration` (core extraction) | **3.5** | measured per-exposure seeing, λ-corrected |
| `measure.py` / `analysis.py` (core lum, profile fits) | **3.0** | 1.3″ fixed (Niemeyer fiducial) |
| `fitting.py:2544` | 3.0 | **1.0 — its own docstring says PLACEHOLDER** |
| `injection.check_level0` | — | 1.8 |

**Clarification, because there are two different 3.5s in this codebase and they
are unrelated:**

- `core.extract_core_spectra(core_radius=3.5)` — 3.5 **arcsec**, the aperture
  radius of fibers fed to the extraction. This is the one you were thinking of.
- `extract.moffat_psf_integration(..., alpha=3.5)` — this sets
  `astropy.modeling.models.Moffat2D.alpha`, and in astropy's parameterization
  `alpha` **is the Moffat power-law index**, i.e. exactly what `fitting.moffat_1d`
  calls `beta`. The function then derives `gamma = 0.5·FWHM/√(2^(1/alpha) − 1)`,
  which is the standard FWHM↔gamma relation and confirms the reading.

So the core extraction really is using index 3.5 while everything else uses 3.0.
It is a genuine disagreement, not a naming coincidence. → **Open Question 1.**

### B4. Work items

- Kill the `psf_fwhm=1.0` placeholder at `fitting.py:2544`. **Agreed.**
- Resolve the index disagreement (OQ1).
- Move `1.3″` → `1.8″` as the fiducial default in `measure.py` / `analysis.py`,
  or make those read `config.psf` instead of carrying their own default. Prefer
  the latter — one source of truth.
- Add a test that changing the fiducial FWHM propagates everywhere it should.

---

## PART C — 2D narrowband image, per galaxy

### C1. Inputs and the wavelength window

Input is the catalog row (position + z) plus the loaded `vdfi`.

The window is specified in **rest-frame Å**, pulled from `config.line_window`,
which is already `LYA_REST ± 3.0`. Do not hardcode, do not accept an observed-frame
window — a fixed observed Å window corresponds to a different physical width at
every redshift and would make the images unstackable across the sample.

> Minor doc bug to fix in passing: the inline comment on `config.line_window`
> says `# +/-4 A about LYA_REST (1211.67-1219.67)` while the code is `± 3.0`.
> The code is right; the comment is stale.

**No sky-line check.** Per Austin: sky subtraction is handled upstream in the
VDFI reductions, residuals are only ever detectable in very low-brightness
stacks, and they will not be visible in an individual fiber cell. Adding a
flagging step here would be machinery that never fires.

### C2. Fiber → single brightness value

Per fiber, pooled across all exposures (`core.fiber_extraction_geometry` already
returns exactly this geometry — `dra`, `ddec` in arcsec, good-fiber flag,
exposure index):

1. Collapse the spectrum over the observed window `LYA_REST·(1+z) ± 3·(1+z)` Å
   to one value + one variance.
2. Subtract the same `[bg_inner, bg_outer)` annulus background used everywhere
   else. No new background definition.
3. Convert `dra, ddec` → kpc using that galaxy's z.

Also produce an **off-band image** from sidebands either side of the line, using
the identical pipeline and plotted on the identical color scale. This is the
built-in null: if the off-band image shows the same blob, the blob isn't Lyα.

### C3. The gridder

Default `grid_method='gaussian'`: fixed-kernel, inverse-variance weighted.

For output pixel *j* and fiber *k*:

```
w_jk    = K(|x_j − x_k| ; h) / σ_k²          K = Gaussian, FIXED width h
I_j     = Σ_k w_jk f_k / Σ_k w_jk
Var_j   = Σ_k w_jk² σ_k² / (Σ_k w_jk)²
N_eff,j = (Σ_k w_jk)² / Σ_k w_jk²
```

Implementation: truncate the kernel at ~3h, `scipy.spatial.cKDTree` ball query.
Sub-second per galaxy.

`N_eff` is the effective number of fibers contributing to each pixel — the 2D
analogue of the per-bin `nfib` the radial code already carries. Pixels below an
`n_eff_min` threshold are blanked. Without this, edge pixels supported by a single
fiber render as solid patches that look like real structure.

**Kernel width `h` is a required, explicit parameter on every call — no hidden
default buried in a function body.** Starting value `h ≈ fiber radius in kpc`,
but this is expected to be tuned by hand to get the resolution we want, so it must
be visible and easy to sweep. Same for pixel size; start at ≲ ⅓ the fiber diameter
in kpc (VIRUS fibers are 1.5″, ≈ 12 kpc at z ≈ 2.5, so ~3–4 kpc pixels) to avoid
aliasing.

The tradeoff, stated for the docstring: smaller `h` → better resolution, noisier,
more coverage holes.

`grid_method='voronoi'` is available as a **display-only** option. Measurement
functions must reject it with a clear error. Reason, for the docstring: a Voronoi
cell in the sparse outskirts is one noisy fiber painted across a large area, and
it has no statable PSF — which matters because every extendedness claim is read
*against* the PSF.

### C4. The PSF of the image

Default: the analytic Moffat from Part B — measured seeing, or the β = 3.0 /
1.8″ global override — as the PSF overlaid on and compared against the images.

Additionally, and as a **plotting function rather than a required step**:
`plot_psf_as_image()`, which renders the PSF through the same gridder with the
same fiber positions, so we can see what a pure point source would look like as
an image. Austin's suggestion, and it's the useful half of the empirical-PSF idea
without making it a mandatory pipeline stage — it shows the sampling irregularity
and coverage gaps that the analytic curve can't.

### C5. Display

- PSF contour overlaid on every image **by default**, with `show_psf=False` to
  disable. The extendedness of anything in the image is only readable against it.
- Image + variance map + `N_eff` map all returned; `N_eff` plotted on request.
- Off-band companion on a shared color scale.

---

## PART D — image stacking

Once C works, this is mostly free, and this is where the actual science number is
(individual images are noise-dominated for typical galaxies; the stack is not).

- Stack per-galaxy kpc-gridded images: median and inverse-variance mean.
- Symmetry diagnostics: azimuthal profile, quadrant comparison, ellipticity.
- **Null: re-stack at random position angles.** Anything that survives random
  rotation is not real orientation-dependent structure.
- Subset stacks reusing the existing `splits` recipes (mass, z, environment).
- **Validation, and this one matters:** the stacked image's azimuthally averaged
  radial profile should reproduce the existing 1D radial profile. They come from
  genuinely independent code paths — fiber → grid → azimuthal average vs.
  fiber → radial bin → combine — so agreement is a real cross-check on both, and
  disagreement localizes a bug fast.

---

## PART E — pairs (exploratory, deliberately small)

Scope check first: there is a lot to build in A–C and this part is speculative.
It is scoped as **a pair finder plus one figure**, not a stacking sub-pipeline.

- **Pair finder.** Galaxies within a projected separation range and a Δv
  threshold. Both fields. Returns a small table.
- **Rotation.** Rotate each pair to a common orientation with the companion
  placed **left–right** (horizontal), not vertical — screens are landscape and it
  reads better.
- **One figure.** With ~10 pairs, plot all of them: each galaxy on the left or
  right side, colored by its velocity **relative to that pair's mean velocity**.
  Expected appearance is a "peanut" — two lobes with scatter, each lobe carrying
  a mix of blues and reds. That plot is the deliverable: it shows the geometry is
  being handled correctly and shows whether there's anything worth pursuing.

**Expectation setting, deliberately:** with ~500 galaxies, pairs meeting both a
small-Δv and a 0.5–2 Mpc separation cut will number in the tens. Expected Lyα
filament surface brightness is ~10⁻²⁰ erg/s/cm²/arcsec² — the SSA22 detection
(Umehata 2019) needed deep MUSE. If this goes anywhere in the paper it goes in as
**a sensitivity demonstration with an upper limit**, framed the same way the
outer-bin null is now framed. Not as a detection attempt.

Anything beyond the pair finder + the peanut figure (separation-rescaled
stacking, inter-pair surface brightness measurement, matched nulls) is a
**separate spec**, written only if the figure looks encouraging.

---

## Success criteria

- **A:** `halo_fraction` computed for the full catalog, leak-aware, with per-galaxy
  S/N attached; distribution is sensible (bounded 0–1, no pile-up at the edges
  beyond what S/N explains); the number is reproducible from a stated `PSFSpec`
  and `boundary_kpc`.
- **B:** one PSF definition, one source of truth, `provenance` recorded in every
  product; the `fitting.py:2544` placeholder gone; a test that catches a
  non-propagating change.
- **C:** a bright LAE produces a visibly extended image against the PSF contour,
  and its off-band companion does not.
- **D:** the stacked image's radial profile agrees with the existing 1D profile
  within errors. **This is the gating check for the whole imaging half** — if
  they disagree, something in the gridder is wrong and nothing downstream is
  trustworthy.
- **E:** the peanut figure renders correctly for ~10 pairs.

---

## Open questions for Austin

1. **Moffat index — 3.0 or 3.5, and does the core extraction get re-run?**
   Now that β = 3.0 is the declared project value, the consistent move is to set
   `extract.moffat_psf_integration(alpha=...)` to 3.0 as well. But that function
   generates the PSF weights used by the actual PSF-weighted core extraction, so
   changing it **changes every core spectrum** and invalidates existing core
   products — they'd need re-extraction. Options:
   (a) change it to 3.0, accept the re-run, get one PSF everywhere;
   (b) leave extraction at 3.5 and document that the extraction PSF and the
   modeling PSF differ deliberately (needs a justification — is 3.5 a measured
   VIRUS value from the reduction team, or just inherited?);
   (c) parameterize it and run both once to see how much the core flux actually
   moves before deciding.
   I lean (c) → then (a), because "how much does it matter" is cheap to answer and
   makes the choice defensible rather than arbitrary. But if 3.5 came from an
   actual VIRUS PSF fit upstream, that's a real argument for (b) and I don't know
   its provenance.

2. **Which zone does "core" mean in the aperture path?** The innermost radial bin
   (22 kpc wide in production), or a fixed physical aperture? The bins are kpc
   and fixed-width, so the innermost bin is the same physical size for every
   galaxy — which is convenient — but it isn't matched to the PSF, whose kpc
   width varies with z across the sample. Fixed-aperture is more defensible;
   innermost-bin is what the existing stack code does. Consistency with the stack
   argues for the bin.

3. **Fiducial FWHM 1.8″ vs. the existing 1.3″ in `measure.py`/`analysis.py`** —
   do those stack-level defaults change to 1.8″ too, or does 1.8″ apply only to
   the new single-galaxy path? If they change, previously computed stack core
   luminosities shift and any numbers already in the presentations become stale.
   Flagging because it's the kind of change that quietly invalidates a table.

4. **Does the halo fraction need a bootstrap?** A single galaxy can't be resampled
   over galaxies, so the error bar is propagated photon noise only. That's
   defensible, but it means the halo-fraction *distribution* will be broadened by
   measurement noise in a way that isn't separated from real object-to-object
   scatter. Deconvolving the two needs either an injection-recovery pass
   (machinery exists in `injection.py`) or a stated caveat. Which?

---

## Decision log

- **2026-08-04** — everything lives in `single.py`; no new modules.
- **2026-08-04** — halo/annular extraction is the default flux path; PSF-weighted
  core extraction is opt-in via a parameter, not a second default.
- **2026-08-04** — fractions are computed over the sum (`halo/total`,
  `core/total`), not `halo/core`. Bounded, stable, sums to 1.
- **2026-08-04** — PSF defaults to measured VDFI seeing; global override is an
  analytic Moffat with β = 3.0, FWHM = 1.8″.
- **2026-08-04** — the field load is **not** wrapped in a driver function; `vdfi`
  is loaded once in a notebook cell and passed in.
- **2026-08-04** — no sky-line flagging in the image path (handled upstream;
  undetectable per-fiber).
- **2026-08-04** — empirical point-source-injection PSF **rejected** as a required
  step; the analytic Moffat is the default, with `plot_psf_as_image` kept as an
  optional diagnostic plot.
- **2026-08-04** — Voronoi gridding available but display-only, rejected by
  measurement functions.
- **2026-08-04** — pair work scoped to a finder + one horizontally-aligned
  "peanut" figure; anything further is a separate spec.
- **2026-08-04** — kernel width `h` and pixel size are explicit required
  parameters, expected to be tuned by hand.
- **2026-08-04** — Open Question 2 resolved: "core" in the default aperture
  path means the innermost radial bin (native kpc bins), not a fixed
  physical aperture -- consistency with the stack code wins.
- **2026-08-04** — Open Question 1 resolved to option (a): Moffat index set
  to 3.0 EVERYWHERE (extract.moffat_psf_integration's default, and
  injection.fiber_response_independent's default so its own cross-check
  against fiber_response_production stays meaningful). This INVALIDATES any
  existing core.extract_core_spectra product built under the old alpha=3.5;
  re-extraction is needed before trusting it against config.PSFSpec.beta.
- **2026-08-04** — Open Question 3 resolved: the new 1.8" fiducial FWHM
  applies to the single-galaxy path (config.PSFSpec) ONLY. The existing
  stack-level 1.3" defaults in measure.py/analysis.py are UNCHANGED for now
  (those numbers are already in presentations); revisit together before
  ever changing both.
- **2026-08-04** — Open Question 4 resolved: no injection-recovery
  deconvolution pass for now. halo_fraction's error is propagated photon
  noise only (single object, no galaxy bootstrap) -- a stated caveat, not a
  gap this build tries to close. Revisit with injection.py's machinery if
  the halo_fraction distribution's width ends up mattering for the paper.
- **2026-08-04** — Part A + the PSFSpec half of Part B implemented in
  single.py/config.py: `measure_single_galaxy_core_halo` (core_flux,
  halo_flux, total_flux, halo_fraction, core_fraction, sn_core, sn_halo,
  core_extendedness, psf/boundary provenance, leak-aware per A2) and
  `psf_fwhm_kpc_for_galaxy` (Part B's per-galaxy PSF resolver). `fitting.py:
  2544`'s `psf_fwhm=1.0` placeholder is killed (now a required arg, raises
  instead of silently using a fake CFHT-LS seeing) -- note this placeholder
  turned out to be the UV-continuum (CFHT-LS imaging) PSF, NOT a Lya/VIRUS
  PSF disagreement; it does not read PSFSpec and should not. Part C
  (imaging) is not started.
- **2026-08-04** — found during Part A's build, not part of this spec:
  `resolve_galaxy_index`, `single_galaxy_spectra`, `_single_galaxy_rvir_kpc`
  are duplicated near-verbatim in `stack.py` (single.py is NOT a thin
  wrapper around them); `measure_single_galaxy` is duplicated in
  `measure.py`; `plot_single_galaxy_panels` is duplicated in `plotting.py`.
  Given "everything goes in single.py, no new modules" (this spec's own
  header), the stack.py/measure.py/plotting.py copies look like leftovers
  from before single.py was carved out that were never deleted. Flagged to
  Austin in chat; not touched here since deleting either copy could break a
  caller of the other -- needs a deliberate decision, not a drive-by fix.
- **2026-08-04** — Part C implemented in single.py: `galaxy_image` (on-band +
  off-band, shared grid/color scale), `_gaussian_grid` (the C3 fixed-kernel
  inverse-variance gridder, h/pixel_size_kpc required with no default, per
  spec), `plot_galaxy_image` (C5 display, PSF contour on by default),
  `plot_psf_as_image` (C4's fiber-sampled PSF diagnostic). Two judgment
  calls made without blocking on Austin, since each extends a principle the
  spec already states rather than inventing a new one:
    - `radius_arcsec` (field of view) defaults to `config.bg_outer_arcsec`
      (90" fiducial) -- beyond that is, by this pipeline's OWN annulus
      definition, "background," so it's a free, already-justified stopping
      point, not an arbitrary new number.
    - `grid_method='voronoi'` is accepted as a value but raises immediately
      with a clear message; building an actual Voronoi DISPLAY helper was
      left out of this pass as genuinely optional scope (Part C's own
      success criterion only needs the gaussian gridder + PSF contour).
  Background subtraction in the imaging path is copied verbatim from
  core.extract_core_spectra's own per-exposure/pooled-fallback block (same
  annulus, same smoothing, no new background definition, per C1). Images
  are in flux_unit_scale-corrected, per-Angstrom observed-frame units --
  NOT rest-framed or area-normalized to a luminosity like Part A's numbers,
  since Part C is a qualitative diagnostic (Part D's stack is where the
  quantitative number lives) -- flag if full physical-unit conversion is
  wanted here too. Verified with a synthetic vdfi (injected declining
  on-band signal recovers a centrally-peaked image >>off-band; voronoi
  raises; plots render) -- not yet run against real VDFI data.
  Part D (stacking the images -- the actual gating validation, since it
  cross-checks this whole gridder chain against the existing 1D radial
  profile) is NOT started.
- **2026-08-04** — Part D implemented in single.py: `stack_galaxy_images`
  (median + inv_var, a JOIN to measure.stack_galaxies -- an image cube
  combines along the galaxy axis exactly like the native (ngal,nrad,nwave)
  case, nothing there is wavelength-specific; `random_rotate=True` gives the
  position-angle null), `azimuthal_profile`, `quadrant_asymmetry`,
  `image_ellipticity`, `validate_stack_vs_1d_profile`, `plot_stack_
  diagnostics`. `galaxy_image` gained one new optional parameter,
  `extent_kpc`, so every galaxy can be forced onto the SAME physical output
  grid regardless of z (needed for a pixel-for-pixel stack with no
  resampling) -- backward compatible, existing single-galaxy calls are
  unaffected. "Subset stacks reusing the existing splits recipes" needed no
  new code: `stack_galaxy_images` takes any catalog subset as `rows`.

  **NOT resolved, and flagged rather than guessed at:** Part C's images are
  in flux_unit_scale-corrected, per-Angstrom OBSERVED units (a deliberate
  scope decision from Part C, documented above) -- NOT the rest-framed,
  area-normalized L_kpc2 surface-brightness convention total_flux_fid/the
  1D profile uses. `validate_stack_vs_1d_profile` defaults to
  `normalize='peak'`, which compares the stacked-image profile and the 1D
  profile as SHAPES only (both divided by their own peak) -- a real, useful
  check (a gridder bug that redistributes flux between radii still shows
  up), but NOT the full quantitative "agrees within errors" success
  criterion Part D actually asks for. Given this exact class of bug (area/
  normalization convention mismatches between the fiber-area and PSF/
  luminosity-distance pictures) has bitten this project for real at least
  twice before (the 2026-07-22 UV and Lya PSF area-convention bugs), doing
  that reconciliation needs a deliberate pass, not a guess folded into this
  build. Verified structurally with a synthetic multi-galaxy vdfi (5
  galaxies, independent noise/fiber sampling each, injected symmetric
  declining signal): the stacked azimuthal profile declines monotonically
  as expected, ellipticity comes back circular (q~1) with a centroid near
  the injection center, quadrant asymmetry is small, and the random-
  rotation null runs. Not yet run against real VDFI data.
