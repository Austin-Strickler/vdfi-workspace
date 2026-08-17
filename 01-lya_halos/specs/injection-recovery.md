# Injection–recovery — testing the pipeline against known truth

**Status: design v6, 2026-08-16.**

- **v6** — **Production binning is 11 bins ending at 2000 kpc**, not 12 ending at 3000 (Austin: runtime, and there is little S/N in the outer bins). Four consequences, all now folded in: (1) `SourceModel.r_trunc` default tracks it, **3000 → 2000** — it has to, since `r_trunc` is the radius the injected `L` is defined *inside*, and a mismatch means `L` is flux inside a radius never measured; the core fraction rises 1.37% → **2.32%**, i.e. 1.7× more injected signal where the measurement is best. (2) `select_nearby_fibers` now cuts at 244″ rather than 366″ — **~55% fewer fibers per source per exposure**, so templates get materially faster and smaller. (3) The `[2000, 3000]` truncation-knee caveat (08-07) **disappears with the bin**. (4) Hartlap goes p = 12 → 11, which moves the correction at N = 400 from 3.3% to 3.0% — **N = 400/field stands unchanged** (§4.4). The `[180, 350]` blend bin is unaffected, so criterion 1d and the 90.62 km s⁻¹ truth are unchanged. ⚠ `run_extract_batch.py`'s `RUN` block still carries the old 13-edge list; unused today (extraction is run from a notebook cell) but it must be updated before it is used, or Stage 1 will produce products the injection truth does not match.

**Superseded status line — design v5, 2026-08-16. Implementation is at ladder level 1 (passing); build-order items 1 and 4 are shipped in `injection.py`, items 2/3/5/6 are not started.** Sections marked **[FILL]** need numbers Austin has and this document does not.

- **v5** — Two changes, both about the line profile. **(1) The LSF convention is settled and now lives in exactly one place.** The instrument's resolution is the measured **1.9 Å rest-frame FWHM = 468.6 km s⁻¹ FWHM = 199.0 km s⁻¹ σ, adopted as 200** (§3.3). Until now the injector applied no LSF and the truth generator applied its own, so the same source was emitted at 250 km s⁻¹ and predicted at 320. Harmless for the centroid — symmetric broadening does not move a median — and fatal at level 7, where it would have injected a line narrower than the instrument can physically produce into spectra that are already convolved. `lsf_kms` is now a `SourceModel` field, `line_profile` returns the line **as observed**, and `predict_binned_observables`'s `lsf_kms` argument is demoted to a consistency assertion. **(2) Skew is out of scope** (Austin, 2026-08-16): the pipeline reports only the median-50%-flux centroid, and the concern that a skewed line might bias that centroid against the finite ±1500 km s⁻¹ window was tested and dismissed — at the observed 320 km s⁻¹ width the window is 4.7σ and the recovered median is exact to 0.01 km s⁻¹ at α up to 4. Both families run α = 0. See §3.3, §4.2, §5.1, §7.2.

- **v4** — Austin's third review, and it fixes a factual error of mine plus fills in the velocity values. **(1) Binning is `bin_mode="kpc"`, not `"virial"`** — the `config.py` dataclass default is stale, and v3's argument that a fixed-kpc break would smear across the R_vir distribution was **wrong and is withdrawn** (§2.4, §4.2). Production edges: `[0, 10, 16, 24, 36, 60, 100, 180, 350, 600, 1000, 2000, 3000]`, **12 bins**; the 22-bin list seen elsewhere was an S/N test for post-merge optimal binning, not production. N = 400/field stands (§4.4). **(2) Velocities set:** family P carries a **constant −50 km s⁻¹**, family H a **step from 0 to +100 km s⁻¹ at 200 kpc** (§4.1, §4.2). **(3) The second-amplitude sensitivity run is dropped** — unnecessary, because P and H together already give a two-point velocity calibration in the core (§4.3). **(4)** New §3.5 (truth generator) and rebuilt §7.2 (the validation ladder) answer "how do we prove the template is real". Context: the air→vacuum correction turned the large-radius result into a **null**, which is why sensitivity now matters at all (§0.4).
- **v3** — Austin's second review, and it overturns v2's main recommendation. **v2 argued family H must carry a zero gradient because a single nonzero amplitude cannot separate additive from multiplicative bias. That argument was too strong.** You do not need `true = 0` to measure an additive offset — you need `true` to be *known*, and zero is only the cleanest known value. Austin's counter-proposal is better than either v2 option: **a piecewise Δv(r) — flat core, red-sloped wings — which contains the null and the recovery in one injection** and separates the two bias terms by their radial *shape* rather than by a second run (§4.2, §4.3). The deferred gradient-recovery pass is removed; it is folded in. One addition: **the break radius is placed deliberately away from R_vir** (§4.2), which turns the design into a test of whether the pipeline localizes transitions where they are rather than where the binning invites them — aimed at the most likely referee objection to Paper I's headline. `store_spec_nobg` question closed by decision: off for injected runs, not worth further analysis (§7.3).
- **v2** — Austin's first review, four changes. **(1)** The two injected families are re-cast as **P (PSF point source)** and **H (double halo)** rather than "symmetric vs. skewed" (§4). This is a better pair: P is a null for *spatial extension itself*, and H validates the model the paper actually fits. **(2)** A consequence Austin should see explicitly: a point source has no outer-bin flux, so **P cannot carry the zero-gradient velocity null** — that null must live in H, which means H carries **zero** Δv gradient and gradient *recovery* becomes a later third pass (§4.3). **(3)** Family Z's N is now computed from the Hartlap requirement at the real bin count (10–12), not assumed: **N ≈ 400 per field**, not 1000 (§4.4). **(4)** The `blank` vs `random` placement question is dissolved rather than answered — run `random` and slice post-hoc by distance to the nearest real source, which costs no extra extraction (§2.1). AGN analogues dropped from scope. New §7 records what must be validated *before* implementation starts.

Companion to `specs/systematics-and-extraction.md` (Stage 1 contract), `specs/background-diagnostics.md` (which this test can accept or reject several knobs of), and `specs/halo-flux-fitting.md` (which supplies the profile model being injected). Closes, or provides the machinery to close, three open TODO items — see §9.

---

## 0. What this test is, and the version of it we are deliberately not building

### 0.1 The closed loop

The tempting design is a fully synthetic cube: generate fibers, apply a PSF, apply an LSF, add Gaussian noise, extract, recover. It is a closed loop. Every component of the simulator is a component of the pipeline's own forward model, so recovering the input demonstrates that the code contains no *bugs* — worth something — while saying nothing at all about whether the *model* is right. Worse, it can never be argued to completion: "is the simulation realistic enough?" is a question a referee can always ask and this design can never answer.

**So: we inject into the real cube.** Fake sources are added to the actual VDFI fiber spectra at fake sky positions, before extraction, and the unmodified pipeline runs over them. Real noise correlations, real sky residuals, real fiber-to-fiber systematics, real masking geometry, real DAR, real per-exposure seeing all come along for free, precisely because we never modeled them. The only thing we assert is the source; everything the source sits in is measured data.

This is also the cheaper build. There is no cube to construct — there is one function between `load_field` and `extract_fibers`.

### 0.2 What the test can and cannot establish

**Can:** whether the extraction → background-subtraction → stacking → measurement → fitting chain returns the profile and the velocity structure that went in, at the S/N and sample size of the real measurement; whether the quoted uncertainties have correct coverage; whether the chain manufactures structure that was never injected.

**Cannot:** whether the injected source model resembles a real Lyα halo. That is not what is under test. The truth value is *defined* by the injection, and the question is only whether the pipeline returns it.

**Scope discipline.** The deliverable for most of this is a short Methods subsection and one or two figures. Size the toolkit to produce those and stop.

### 0.3 The one result that matters most

**Family H (§4.2) carries a piecewise Δv(r): flat in the core, sloped in the wings.** The flat region is the null — if the pipeline returns a gradient where none was injected, that is the single most consequential number this project can produce, because a radial gradient in Lyα centroid velocity is Paper I's headline and it is measured at tens of km s⁻¹ on a 110–140 km s⁻¹ pixel grid whose absolute zero point is itself unverified (`specs/wavelength-zero-point.md`).

Everything else in this document is secondary to recovering that Δv(r) *shape*. If the schedule collapses, run family H and nothing else.

### 0.4 What the air/vacuum correction changed — why this test is now load-bearing

**2026-08-01.** The VDFI spectra were in air wavelengths, never corrected to vacuum Lyα (≈1.1 Å ≈ 85 km s⁻¹ blueward). After correction, the large-radius blueshift **disappears**: what remains beyond ~50 kpc is flat, ~0–70 km s⁻¹ per bin, consistent with zero out to 2700 kpc. The headline is now the **inner decline** (~265 km s⁻¹ at 5 kpc → ~35 by 30–50 kpc) and the transition, not an outer blueshift.

**This flips the statistical burden, and it is the reason this spec matters more than it did a week ago.** A detection is defended with error bars. **A null cannot be** — the failure mode is a false negative. The referee question is no longer "is your outer blueshift real?" but:

> **Would you have detected a velocity offset at large radius if one were there?**

Nothing in the data answers that. An injection does: put a known offset in the outer bins and see what comes back. The recovered value *and its uncertainty* are the sensitivity statement the null result structurally requires. That is now the primary deliverable of family H, ahead of profile recovery.

---

## 1. Architecture

### 1.1 The insertion point

```
load_field(config)                     # unmodified
    ↓
build_bad_fiber_mask(config, vdfi)     # unmodified, FROM CLEAN DATA — see 1.4
    ↓
inject_sources(config, vdfi, inj_table)   # NEW — returns a shallow copy of vdfi
    ↓
build_run_table  ←  inj_table          # run_table IS the injection catalog
    ↓
extract_fibers(config, vdfi_inj, run_table, mask)   # unmodified
    ↓
everything downstream                  # unmodified
```

Nothing after `inject_sources` is aware that injection happened. That is the design constraint: if any downstream module needs an `if injected:` branch, the test is no longer testing the production pipeline.

### 1.2 `spectra` is a read-only memmap — the sparse overlay

`load_vdfi_field` opens the flux and error FITS with `memmap=True` and deliberately keeps the `HDUList` handles alive (stored as `_flux_hdul` / `_error_hdul`) so Stage 1 reads lazily over many hours. You cannot write into `vdfi['spectra']` in place, and a materialized copy is ~16 GB.

You do not need one. Only fibers within roughly the outermost radial bin of an injected source are modified — a few hundred fibers per source per exposure. At ~450 sources that is a few hundred MB of patch data.

**The indexing surface is narrow enough to wrap.** Every read of `spectra` in the extraction path is the same pattern:

| site | expression | form |
|---|---|---|
| `extract.py:1025` | `spectra[e, bg_sel]` | scalar + bool mask |
| `extract.py:1133` | `spectra[e, sel]` | scalar + bool mask |
| `core.py:1670, 1675, 1706, 1899, 1904, 2014` | `spectra[e, <bool mask>]` | scalar + bool mask |
| `core.py:1996`, `extract.py:804` | `spectra.shape[2]`, `.shape[-1]` | attribute |
| `plotting.py:2233` | `spectra[e]`, `spectra.shape[0]` | **scalar only** |
| `extract.py:210` | whole array → `make_spectral_continuum_mask` | **whole array** |

The last two are the exceptions. `plotting.py:2233` (`plot_fiber_overlay`'s dead-fiber marking) takes a bare exposure slice; the overlay must support it. `extract.py:210` passes the entire cube to the mask builder — which per §1.4 runs *before* injection, so it never sees the overlay, but the overlay should raise rather than silently return the raw memmap if it ever does.

So the surface is: `__getitem__((e, bool_mask))`, `__getitem__(e)`, and `.shape` / `.dtype` / `.ndim`. Everything else raises.

```python
class InjectedSpectra:
    """Read-only view of the memmapped cube with sparse additive patches.

    Supports exactly the access pattern the extraction uses: spectra[e, mask].
    Any other indexing raises, loudly — a silent fallthrough to the raw memmap
    would return un-injected data and nothing downstream would notice.
    """
    def __init__(self, base, patches, shape, dtype):
        self._base = base          # the memmap
        self._patches = patches    # {(e, fiber_idx): (nwave,) float32}, or a
                                   # per-exposure CSR-ish (idx array, delta array)
    def __getitem__(self, key): ...
    @property
    def shape(self): return self._base.shape
```

**Decision gate before implementing:** the audit above covers `utils_lya_halo/`. Re-run it against the notebooks that will consume injected products before building, since those are synced snapshots and may index differently. If a non-conforming access appears in a path we need, fall back to option B rather than growing the wrapper's surface — a wrapper that supports arbitrary indexing is just a slow, buggy ndarray.

**Option B (fallback):** materialize a real `np.ndarray` copy of only the *touched exposures*, or of the full cube in float32 if disk and RAM allow (~8 GB at float32). Simplest and safest; costs a write. Keep it in the back pocket, do not build it first.

### 1.3 Cache and output isolation — already solved

No new mechanism is needed. `PipelineConfig.extract_signature_dict()` hashes config knobs into both the cache directory name (`cache/<sig>/`) and `descriptive_filename()`. Add the injection parameters to that dict, **hashed only when injection is active** — exactly the pattern `bg_variants` already uses, where new flags contribute to the hash only when set so that an un-injected config's fine-grid cache stays valid and byte-identical.

Consequences, all free:
- Injected runs land in their own cache directory with their own `_manifest.json`.
- The manifest records the full injection parameter set, so "which run was `<hash>`?" stays answerable.
- A fiducial run can never collide with, or be silently served from, an injected cache.

Add an `INJECT` header keyword to the galaxy FITS alongside `BGINNER`/`BGOUTER`/`MINBGFIB` so an injected product is identifiable from the file alone, not just its path.

### 1.4 Ordering: build the fiber mask *before* injecting

`build_bad_fiber_mask` with a `spec_global`/`spec_local` component calls `make_spectral_continuum_mask(vdfi_data["spectra"], ...)`, i.e. it consumes the whole cube. It must run on **clean, un-injected data**.

Two reasons. First, injected sources would otherwise participate in the percentile statistics that define the mask, changing the mask for the *real* fibers too — a global perturbation from a local addition. Second, an injected source bright enough to trip the continuum percentile would mask itself, reproducing the real sample's self-masking pathology *by accident* rather than by design.

(Our injected sources carry line flux and no continuum, so in practice the mask would barely move. Specify the order anyway; "in practice it barely matters" is how silent couplings get in.)

### 1.5 The error array is not injected

`error[e, sel]` is left untouched. VDFI is sky-dominated at these fluxes, so the source's own shot noise is negligible against the background, and a noiseless signal on real noise is exactly what isolates *pipeline* bias from *source* Poisson scatter.

This is a decision, recorded here so it is not later mistaken for an oversight. It means the injected sample's true per-object noise is very slightly lower than a real object of the same flux. Quantify the shot-noise-to-sky ratio once at the median injected luminosity and put the number in Methods; if it is not ≪1, revisit.

---

## 2. Placement

### 2.1 One mode, sliced afterwards

**v2 (Austin, 2026-08-01).** The knob stays in the code:

```python
placement = "random" | "blank"      # default "random"
```

but **only `random` is run**, and the `blank` question is answered by slicing rather than by a second extraction.

**Why the dilemma dissolves.** Austin's instinct — "real galaxies *do* cluster, artificially avoiding sources is unnatural, but real signal interfering with the test is bad, and both sides are valid" — is right that both sides are valid, but the framing hides the actual asymmetry:

> An injection dropped at a random position has **uncorrelated** neighbours. A real galaxy has **correlated** ones. These are not the same contamination.

So `random` does *not* reproduce real clustering; it reproduces the *mean* neighbour density while destroying the correlation, which is exactly the 2-halo signal the real analysis is trying to measure. And `blank` does not reproduce it either. Neither mode is "realistic" in the sense that matters, so realism is the wrong axis to choose on.

What uncorrelated neighbours actually do is add a random flux offset, most of which the 80–90″ background annulus also sees and subtracts. Expect them to inflate *scatter* rather than shift the *mean* — a second-order effect, which supports Austin's instinct that it doesn't matter much. But "expect" is not "measured".

**So measure it for free.** §5.4 already bins residuals against distance to the nearest real catalog source. Run `random`, then slice that run by that distance: the far-from-anything subsample *is* the `blank` measurement, selected post-hoc from the same extraction. If the residuals are flat in that variable, the question is closed at zero extra cost. If they are not, you have quantified real-source contamination as a function of separation, which is more informative than either mode alone.

The redshift spread (§2.4) does additional work here, as Austin notes: injections at a spread of z see any given real neighbour at a different observed wavelength, so a real LAE only contaminates the small fraction of injections that happen to sit near it *in redshift as well as on sky*. That further suppresses the effect and is worth stating in Methods.

`blank` remains implemented and unexercised, so the fallback is a config change rather than a code change.

### 2.2 Minimum separation between injections — loose, plus a recorded diagnostic

The naive requirement — no injection inside another's 80–90″ background annulus — is unaffordable. Field areas are small:

| Field | Area (deg²) | arcsec² | hex-packing ceiling @180″ | realistic random yield |
|---|---|---|---|---|
| COSMOS | 0.20 | 2.6 × 10⁶ | ~92 | ~50 |
| AEGIS | 0.09 | 1.2 × 10⁶ | ~42 | ~23 |

That criterion caps the whole experiment at ~70 objects. It is also the wrong criterion. The background annulus sits at 80–90″ ≈ 660–740 kpc at z ≈ 2.3, and the injected profile (h1 ~ 17 kpc) has effectively no flux there. Unlike with real data we can *prove* this rather than assume it: **evaluate the injected profile at 80–90″ and confirm it lies below the background noise floor.** Make that a printed assertion in the injection setup, not a paragraph of reasoning. If it fails, the injected profile is wrong, not the placement.

**Adopted (Austin, 2026-08-01): loose separation, random placement.** A hard floor of `min_sep = 30″` (~1 R_vir, ~250 kpc) — enough to prevent pathological near-coincident blends that are neither source, cheap enough to be invisible in the packing budget.

**And record `nn_dist`, the distance to the nearest other injection, per object.** This converts crowding from an assumption into a measured diagnostic: bin the recovery residuals by `nn_dist` and show they are flat. One panel, and it retires the entire question. If they are *not* flat, that is a real result about aperture contamination in the fiducial analysis too.

### 2.3 Footprint

Positions are drawn uniformly over each field's RA/Dec range (COSMOS `[149.90, 150.40] × [2.10, 2.50]`, AEGIS `[214.65, 215.15] × [52.73, 53.03]`) and then rejected unless they have adequate fiber coverage. Define adequate as: at least `n_min_fibers` **[FILL — suggest matching the 5th percentile of the real sample's innermost-bin fiber count]** good fibers within the innermost radial bin, pooled over exposures. The fill factor is not 100% and injections falling into IFU gaps must be discarded, not silently extracted as NaN.

### 2.4 Redshift and mass

Bootstrap **(z, M\*) pairs jointly from the real run table**, resampling with replacement.

**Correction (v4): M\* does not affect the bin edges.** Under `bin_mode="kpc"`, `radial_bin_edges` returns `config.bins` unchanged — no per-galaxy conversion, no SMHM call. So the v3 justification ("R_vir sets the per-galaxy bin edges") is void, and the girelli20/behroozi19 choice does not touch the fiducial binning at all.

M\* is still bootstrapped jointly with z, for three surviving reasons: the split diagnostics key on it, R_vir is still *reported* per galaxy (and the 200 kpc break is quoted in R_vir units), and preserving the real z–M\* correlation costs nothing. But it is now bookkeeping rather than a driver.

**z is the one that matters.** It sets the observed wavelength of the injected line, and therefore the sky-line coincidence rate, the sub-pixel phase against the 110–140 km s⁻¹ grid, and where the line falls in the LSF. Bootstrapping z from the real table makes the injected stack directly comparable to the real one on all three.

Also bootstrap the **Lyα luminosity** from the same rows, as part of the same triple, so the injected stack's S/N matches the real stack's.

---

## 3. The injected source model

### 3.1 Not separable once Δv depends on radius

**Correction (v4).** Earlier drafts called the source "spatial profile × spectral line profile". That is only true when Δv is constant. With a radially varying Δv the source is

```
f(r, λ)  =  I(r) · S(λ | z, Δv(r), σ, α)
```

— a *family* of separable products indexed by r, evaluated on an (r, λ) grid, not a single outer product. Cheap either way, but the distinction has consequences that §3.5 turns into hard requirements: because different radii carry different velocities, **a fiber integrates a superposition of differently-shifted lines**, not one shifted line. Its recorded spectrum is broader than the injected σ, and can be skewed even when α = 0. That is real physics, correctly reproduced — and it is why the truth is the forward-modeled observable, never the injected parameter (§3.5).

Family P is the exception: Δv is constant, so P *is* separable, which is one more reason it is the right first rung on the ladder (§7.2).

Injection is per exposure, so the real seeing and DAR chain is exercised rather than bypassed:

```
flux[e, f, λ] += L · Σ(r_eff[e, f, λ]) · S(λ | z, Δv(r), σ, α)
```

where `r_eff` comes from the same geometry the pipeline uses — `_exposure_seeing(seeing, e)` for the per-exposure Moffat width, `_offsets_with_dar(...)` for the wavelength-dependent sky offsets. **Injecting into the coadd, or at a fixed wavelength-independent position, would skip the optics chain entirely and is the most likely way to build a test that passes vacuously.**

The spatial profile must be convolved with the PSF before being sampled onto fibers, and the fiber must integrate it over its aperture — reuse `moffat_psf_integration` / the `optics` path rather than writing a second convolution, so that a PSF-convention error in production is *reproduced* in the injection and therefore detectable as a residual. (Cf. the 2026-07-22 area-convention bug: a second, independent implementation would have hidden it.)

### 3.2 Spatial profile — both families

The production `expcore` model: exponential core with scale `h1`, plus the outer power-law/second-exponential term with amplitude ratio `f` and transition `r_c`.

**[FILL]** `h1`, `f`, `r_c`, `gamma`. **These must come from the post-2026-07-22-PSF-fix, post-2026-07-27-SMHM-switch refit, not from any existing figure.** TODO item 2 ("re-run everything downstream") is therefore a hard dependency of this spec — injecting the stale h1 would produce a test whose truth value does not correspond to the measurement it is validating.

### 3.3 Spectral profile

Skewed Gaussian, parameterized by centroid velocity `Δv`, width `σ`, and skew `α`.

- **σ** — **[FILL]**, the *intrinsic* width, matched to the observed stacked line width deconvolved for the LSF below.
- **LSF** — **SET (v5): σ_LSF = 200 km s⁻¹.** From the measured **1.9 Å rest-frame FWHM**: velocity is (1+z)-invariant, so `c · 1.9 / 1215.67` = 468.6 km s⁻¹ FWHM, and `/2√(2 ln 2)` = 199.0 km s⁻¹ σ, adopted as 200. The line the generators emit is the intrinsic line **blurred by this**, i.e. √(σ² + 200²) — at the placeholder σ = 250 that is 320.2 km s⁻¹.

  **This is set in one place, on the `SourceModel`, and that is the point.** The injector and the truth generator both read it from there. Setting it independently in each is what produced the 250-vs-320 mismatch that survived a week undetected, because it is invisible in the centroid and only shows up in widths, which nothing was comparing. `check_level0` check 2 now asserts the emitted width, so a recurrence fails loudly.
- **α** — **0, both families. Out of scope (v5, Austin).** The pipeline reports only the median-50%-flux centroid, so a recovered α has no consumer. The one argument for injecting it anyway — that a skewed line might bias the centroid through the finite ±1500 km s⁻¹ measurement window — was tested and does not hold: at the observed 320 km s⁻¹ width the window is 4.7σ and the recovered median is exact to 0.01 km s⁻¹ at α up to 4, at both Δv = 0 and Δv = +100.

  α is retained as a parameter, and handled exactly (the skew-normal family is closed under Gaussian convolution, with **both** ω and α transforming — see `apply_lsf`), so reviving it is a config change rather than a code change. Note that broadening σ in quadrature while holding α fixed, which is the intuitive thing to write, is **wrong**: at σ = 250, LSF = 200, α = 2 it misplaces the predicted centroid by 36.9 km s⁻¹, disguised as a clean systemic shift.
- **Δv(r)** — the discriminating parameter, §4.

Every injected source in a family gets **identical** profile parameters (Austin, 2026-08-01). Only (z, M\*, L, position) vary. This gives an unambiguous scalar truth per family. The separate and genuinely interesting question — whether stacking a *population* of scale lengths returns the population mean h1 or something luminosity-weighted and biased — is a follow-on, gated on this test passing, and is a config knob (`profile_scatter`) specified but not exercised in this pass.

### 3.4 Lyα line masking

### 3.5 The truth generator — and why the truth is not the injected parameters

**The single most likely way to produce a wrong "bias" in this test is to compare the recovered value against the number typed into the config.** The injected parameters are not the expected observables. Three effects separate them, all real, all correctly reproduced by a correct pipeline, and all of which look exactly like pipeline bias if you compare against the wrong truth.

**(a) Bin truth is not Δv(r_center).** The PSF and the steep profile weight each bin's flux toward its inner edge, so the truth for bin *i* is the **flux-weighted mean** of Δv over that bin *after* PSF convolution. This is largest where the profile is steepest and where bins are widest — i.e. the innermost bins and the `[180, 350]` straddling bin, which are exactly the bins carrying the result.

**(b) Injected σ is not recovered σ.** A fiber near the 200 kpc break sees flux from both sides of it and records a superposition of lines at 0 and +100 km s⁻¹. The recorded line is broader than the injected σ, by an amount that varies with radius.

**(c) A velocity step plus PSF blending manufactures skew from symmetric lines.** Family H's recovered α is therefore *not* the injected α, and the difference is a prediction, not an error.

**So the spec requires a truth generator**, `predict_binned_observables(model, z, bins, psf, lsf)`, that takes the injected 3-D model and returns the expected per-bin flux, Δv, σ, and α — the same quantities §5.1 compares against.

Two hard requirements on it:

1. **It must be an independent implementation, not a call into the injector.** Comparing the injector against itself validates nothing. Two implementations that disagree is the only mechanism that would have caught the 2026-07-22 area-convention bug, where one convention was wrong in one place and nothing existed to contradict it.
2. **It must be validated on cases with analytic answers before being trusted on cases without** — see the ladder in §7.2.

Everything in §5 and §6 compares against this generator's output. Nowhere does a success criterion reference a config value directly.

### 3.6 Lyα line masking

Injected line flux lands inside the wavelength region the background pipeline masks and interpolates across (`_maybe_smooth`). That is intended — it is the same treatment the real line receives — but it means the injected background variants are *also* contaminated in the same way, and §2.2's "no flux at 80–90″" assertion should be checked against the raw, pre-mask background (`BG_RAW`, if `store_raw_bg` is on) rather than the smoothed `BACKGROUND` HDU. Cf. `background-diagnostics.md` §0.

---

## 4. The families and the run plan

**Two injected families (Austin, 2026-08-01), plus the zero-amplitude null.** The pair is chosen so that one is a null for spatial structure and the other validates the model the paper fits.

### 4.1 Family P — the point source

| | |
|---|---|
| Spatial | **Pure PSF point source.** No intrinsic extent at all. |
| Line | Symmetric Gaussian, α = 0 |
| **Δv** | **Constant −50 km s⁻¹** (blueshifted), all radii |
| Truth for `h1` | **0** |

**The strongest null in the whole plan.** A point source convolved with the seeing is, by construction, the narrowest thing the instrument can produce. Any extended structure recovered from family P is manufactured by the pipeline — by the PSF convention, the fiber-aperture integration, the background subtraction, or the profile fit. There is no interpretation under which recovering `h1 > 0` from family P is real.

This is a much better null than "an exponential with a small h1", because it removes the free parameter: you are not asking "is the recovered h1 close enough to the injected one", you are asking "is it consistent with zero". Given the 2026-07-22 PSF area-convention bug — where a wide-PSF `h1` collapse turned out to be a normalization error back-ported from the UV code — this family is exactly the regression test that bug should have had.

Family P also directly measures the **PSF-deconvolution floor**: the smallest `h1` the pipeline can distinguish from a point source at this S/N and sample size. That number belongs in Methods whether or not anything fails.

**Why Δv = −50 and not 0 (Austin, 2026-08-01).** A zero-velocity injection cannot distinguish *"the pipeline measures velocity correctly"* from *"the pipeline returns zero regardless."* A known nonzero constant tests the **absolute velocity zero point through the entire chain** — extraction, background subtraction, rest-framing, stacking, centroid estimation — which is precisely the quantity that was silently wrong until the air→vacuum correction (§0.4). Since P's flux is entirely in the core bins, this is a high-S/N test of the absolute scale where the measurement is best constrained.

−50 is chosen well: opposite in sign to family H's +100 so the two families can never be confused; comparable to the 85 km s⁻¹ air/vacuum offset and to the residual 10–20 km s⁻¹ now being quoted, so it sits in the regime that matters; and comfortably above the ~22 km s⁻¹ velocity floor.

### 4.2 Family H — the double halo

| | |
|---|---|
| Spatial | `expcore` two-component, parameters per §3.2 **[FILL]** |
| Line | **Symmetric, α = 0** (v5 — skew dropped, §3.3), σ_obs = √(σ²+200²) |
| **Δv(r)** | **Step: 0 km s⁻¹ inside 200 kpc, +100 km s⁻¹ outside.** Abrupt, not smoothed. |

Validates the model the paper actually fits: if the two-component profile is recoverable at this S/N and sample size, the fitted `h1`, `r_c`, and crossover radius mean what they are claimed to mean. If it is not, the fit is reporting parameters the data cannot constrain — which is the outcome that would matter most to a referee.

**Why piecewise (Austin, 2026-08-01).** A single injection with a flat core and sloped wings contains both regimes at once: the flat region is the null (does the pipeline *manufacture* a gradient), the sloped region is the recovery (does it *return* the right one), and the comparison is between two Δv(r) curves rather than two scalars. See §4.3 for why this is better than either v2 option.

**Why 200 kpc, and why *not* at R_vir.** Paper I's headline is a transition near the virial radius (~70–75 kpc). If the injected kinematic break also sat at R_vir, recovering it would prove nothing — the result would be indistinguishable from the binning preferring that radius. 200 kpc is ~2.7 R_vir, well clear of it, which makes the test sharp in both directions:

- Recovering the break **at 200 kpc** demonstrates the pipeline localizes a kinematic transition where it actually is.
- Recovering it **pulled toward R_vir**, or finding a spurious feature at R_vir when none was injected there, would mean the radial binning imposes structure at R_vir — precisely the objection a referee would raise against the real result, found here rather than in review.

**v3's objection to a kpc-defined break is withdrawn.** v3 argued a fixed-kpc radius would smear across the R_vir distribution in the stack. That assumed `bin_mode="virial"`, which is the stale dataclass default, **not what production runs**. Binning is in physical kpc, so a kpc-defined break is sharp in the stack by construction and no normalization argument applies.

**200 kpc straddles a bin.** The production edges are `[0, 10, 16, 24, 36, 60, 100, 180, 350, 600, 1000, 2000]`, so 200 falls inside the `[180, 350]` bin: that bin's truth is a **flux-weighted blend** of 0 and +100, not one or the other, and it must be computed as such by the §3.5 truth generator rather than assigned.

**Keep it there rather than moving to the 180 kpc edge.** A step sitting exactly on a bin boundary is the easy case — the binning cannot smear what it does not straddle. A real physical transition does not respect bin edges either, so a break inside a bin is closer to the actual inference problem and tests sub-bin behaviour the aligned version would skip. The blend is exactly computable, so nothing is lost by making it harder. (180 kpc is the fallback if the straddling bin turns out to dominate the result.)

**Abrupt, not smoothed.** A step is the cleanest probe of radial resolution: the recovered curve is a **step response**, from which the response to any other transition shape follows. Inject something already smooth and the smoothness put in cannot be separated from the smearing the pipeline added. The step response is also directly the number Paper I needs — it says how sharp a real transition at R_vir would have to be to look the way the measured one does.

~~Skew *is* injected here~~ — **withdrawn in v5.** Skew is not injected and not measured (§3.3): the pipeline reports only the centroid, and skew was tested and shown not to bias it at this width and window. So H tests three things at once — two-component profile recovery, the zero-velocity null (core), and outer-offset **sensitivity** plus transition localization (wings). Note the blending-induced asymmetry of §3.5(c) still physically occurs and is still forward-modeled correctly by the truth generator; it is simply not compared against anything.

### 4.3 The velocity calibration falls out of the P/H pair — no extra runs

**A single injected amplitude does not, on its own, separate additive from multiplicative bias** in `recovered = a·true + b`. v2 concluded from this that one family must be injected at exactly zero; v3 concluded a piecewise profile was needed. **Both are superseded: P and H together already supply two velocity amplitudes at high S/N in the same region.**

| | core Δv truth | what it gives |
|---|---|---|
| Family H | **0** | `b` directly — any recovered core velocity is pure additive bias |
| Family P | **−50** | with H, `a = [recovered(P) − recovered(H)] / (−50 − 0)` |

Both are core measurements where S/N is highest and where the profile is best constrained, so both terms are measured where they can be measured best. **Two families, two amplitudes, zero extra extraction.**

This is also why **the second-amplitude sensitivity run is dropped** (Austin, 2026-08-01). The sensitivity floor at large radius is not obtained by injecting a smaller offset — it is the **uncertainty on the recovered +100**, which the H run reports directly. A second amplitude would only be needed if the response might be nonlinear, and linearity is already checked by the P/H core pair above.

Austin's framing of why this is safe is worth preserving: *if the pipeline correctly measures a real offset, the assumption that it will not also manufacture one from nothing is plausible — otherwise the observed signal would have to be ridiculous.* With the P/H pair, that assumption does not have to be made at all.

**Consequence for the three-zone velocity products.** A step Δv(r) at 200 kpc does not align with the core / one-halo / two-halo zone boundaries used by the fiducial analysis, so the truth values for `measure_core_twohalo_velocity` must be computed by integrating the *step* model over each zone (§3.5), not read off a single number. Bookkeeping, not a problem — but it must come from the model.

### 4.3b What family P gives up

**A point source has no outer-bin flux.** Beyond a few arcsec it contributes nothing, so Δv is undefined in the outer bins of family P. P is a null for *spatial extension*, not for kinematics; all of the velocity work lives in H. This is a division of labour, not a loss.

### 4.4 Family Z — the zero-amplitude null

Same machinery, `L = 0`. No injected flux; only positions and a run table. Delivers:

1. **Random-position nulls** and a Hartlap-corrected **bin–bin covariance matrix** (TODO: "close the correlated outer-bin noise" — flagged as the only item load-bearing for all three papers).
2. **The double-Gaussian null test** (deferred TODO): fit single- and double-Gaussian core templates to pure noise at high N and ask whether the template manufactures structure by construction.

No separation constraint beyond fiber coverage.

**N is set by the covariance requirement, not by symmetry with P and H.** The production edge list is

```python
bins = [0, 10, 16, 24, 36, 60, 100, 180, 350, 600, 1000, 2000]   # kpc
bin_mode = "kpc"
```

— 12 edges, so **`n_bins = p = 11`** (v6; the 3000 edge was dropped 2026-08-16). (A 22-bin list appears in some run cells; that was an S/N test to find the optimal binning after the data merge, **not production**. If it is ever promoted, re-derive this table — at p = 22, N = 400 drops to N/p = 18 and the correction nearly doubles.)

| N per field | Hartlap α = (N−p−2)/(N−1) | correction | element noise ≈ √(2/(N−1)) | extraction (2 min/gal) |
|---|---|---|---|---|
| 150 | 0.913 | 8.7% | 11.6% | 5 hr |
| 200 | 0.935 | 6.5% | 10.0% | 6.7 hr |
| 300 | 0.957 | 4.3% | 8.2% | 10 hr |
| **400** | **0.967** | **3.3%** | **7.1%** | **13 hr** |
| 500 | 0.974 | 2.6% | 6.3% | 17 hr |
| 1000 | 0.987 | 1.3% | 4.5% | 33 hr |

**Adopted: N = 400 per field.** The Hartlap correction is then 3.3% — a footnote in Methods, not a systematic anyone can argue with — and N/p ≈ 33, comfortably past the N ≳ 10–20 p rule of thumb for a stable inverse. Going to 1000 costs **20 additional hours per field** to move covariance-element noise from 7.1% to 4.5%, which is not a trade worth making; that noise is already well below the ~10% level at which it would start to matter for parameter errors. N = 300 is defensible if schedule pressure appears.

Re-derive this table if `n_bins` changes. The whole point of computing it rather than assuming it is that the answer moves with p.

**Run family Z first.** It is by far the cheapest to interpret, and it exercises placement, run-table construction, cache signatures, and mask ordering end to end. Any bug in that plumbing shows up here, before ~30 hr of extraction is spent on P and H.

### 4.5 Passes

One family per extraction pass. Extraction runtime is linear in galaxy count (≈2 min/galaxy; 8.5 hr for ~250 COSMOS galaxies with `combine_threads=8`), so splitting families into separate passes **costs nothing** and buys two things: the families cannot contaminate each other, and each pass only has to support half the placement density.

| Pass | Field | Family | N | Est. runtime | Order |
|---|---|---|---|---|---|
| Z-C | COSMOS | Z (L = 0) | 400 | 13 hr | 1 |
| Z-E | AEGIS | Z (L = 0) | 400 | 13 hr | 2 |
| P-C | COSMOS | P (point source) | 225 | ~7.5 hr | 3 |
| P-E | AEGIS | P (point source) | 225 | ~7.5 hr | 4 |
| H-C | COSMOS | H (double halo, piecewise Δv) | 225 | ~7.5 hr | 5 |
| H-E | AEGIS | H (double halo, piecewise Δv) | 225 | ~7.5 hr | 6 |

**Total ≈ 56 hr, six passes, nothing deferred.** v2's separate gradient-recovery run is gone — the piecewise Δv(r) folds it into H (§4.3), saving ~15 hr and, more importantly, removing a pass that would otherwise have been sequenced behind H's result. Passes are independent and can be run as the machine frees up; nothing here needs to run contiguously.

225 per family per field gives ~450 per family combined, matching the ~450-galaxy real sample, so injection uncertainties are no larger than the uncertainties they are validating. Anything much smaller and the strongest available statement becomes "no bias detected, at a precision worse than the measurement" — which a referee can reasonably shrug at.

**Fields are extracted separately and stay separate through the diagnostics**, per standing practice. Pool only after per-field recovery is shown to be consistent.

---

## 5. Recovery and diagnostics

Every quantity below is measured on the injected stack with the **production** measurement code, no special-casing.

### 5.1 Per-parameter recovery

For each family, report **bias and scatter separately** — a small bias with 2× underestimated errors is worse than a modest bias with honest ones, and a single "RMS error" number hides which one you have.

| Quantity | Truth in P | Truth in H | Why |
|---|---|---|---|
| `h1` | **0** | injected | headline profile scale; P gives the deconvolution floor |
| outer-term amplitude / `r_c` | **absent** | injected | does the fit **invent** (P) or **destroy** (H) the outer component |
| Δv per radial bin | **−50** (core bins) | §3.5 forward model of the step | **the primary result — compare the whole curve** |
| Δv, core bins (r < 200 kpc) | −50 | **0** | the P/H pair gives `a` and `b` (§4.3) |
| Δv, outer bins (r > 200 kpc) | n/a — no flux | **+100**, *and its uncertainty* | **the sensitivity statement the null now requires (§0.4)** |
| recovered break location | n/a | **200 kpc** | transition localization — must not migrate to R_vir (§4.2) |
| line σ per radial bin | √(σ²+LSF²) | §3.5 forward model | baseline for the Hubble-flow broadening prediction |
| ~~skew α per radial bin~~ | — | — | **REMOVED v5** — out of scope, §3.3 |
| three-zone luminosities & ratios | PSF-only | integrate injected profile | `measure.py` Part 3b |
| crossover radius | **undefined / should fail** | analytic from injected params | Paper I headline quantity |

The P column is where most of the teeth are: five of those rows have truth values of exactly zero or "absent", which are the easiest claims to test and the hardest to explain away.

### 5.2 Coverage — the diagnostic that carries the most weight

Does the bootstrap 68% interval contain the truth 68% of the time? Run per parameter, per radial bin, over the injected sample via the existing bootstrap/jackknife/z-scramble machinery.

Most pipelines fail this, and passing it is the most defensible single sentence available for the Methods section. It is also the only diagnostic here that validates the *error bars* rather than the *central values*, and the error bars are what every downstream significance claim rests on.

Report the empirical coverage of the nominal 68% and 95% intervals as two numbers per parameter. If coverage is, say, 55% at nominal 68%, the honest response is an inflation factor quoted in Methods — not a silent one.

### 5.3 Structure that was never injected

This is what family P is for, and it absorbs what v1 proposed as a separate zero-outer-term family — a point source *is* the zero-outer-term case, taken to its limit, so no third family is needed.

- **Fit the full two-component model to family P.** The outer/2-halo term's injected amplitude is exactly zero. How often, and at what amplitude, does the fit report one anyway? This is the direct answer to "is the 2-halo term real", and it is a stronger test than the small-`h1` version because the truth is identically zero rather than merely small.
- **Fit `expcore` to family P and read `h1`.** Its distribution is the PSF-deconvolution floor: the smallest scale length distinguishable from a point source. Any real `h1` measurement below that floor is not a measurement.
- **Attempt a crossover radius on family P.** There is no crossover. If the machinery returns one anyway rather than failing, that is a `diagnose_crossover_failures`-class problem in the fit, not a physical result.
- **Fit a double-Gaussian core to family Z** and count how often it prefers two peaks over one (§4.4).
- **Recover Δv(r) from family H's core bins and fit a slope.** The distribution of that slope over bootstrap realizations is the null distribution against which the real measured gradient's significance should be quoted — a stronger statement than a bootstrap alone, because it is calibrated against a region where the true answer is known to be zero.
- **Overlay recovered Δv(r) on injected Δv(r), per field.** This is the figure. Austin's framing: the test is whether the recovered *full profile* looks like the intrinsic one we manufactured, and that is a visual judgement before it is a numerical one. The summary statistics in §5.1 exist to back up that plot, not to replace it.

### 5.4 Nuisance dependence

Residuals binned against, and shown flat in:

- `nn_dist` (§2.2) — crowding
- number of good fibers in the innermost bin — coverage/masking
- z — sky-line coincidence, wavelength solution, pixel phase
- M\* — R_vir binning path
- **distance to nearest *real* catalog source** — contamination. **Load-bearing per §2.1:** this slice is what replaces a separate `blank` extraction, so it is not optional here.

Any of these showing a trend is a real systematic in the fiducial analysis, not merely an artifact of the test.

### 5.5 Nuisances worth injecting once the baseline passes

Deliberately deferred out of v1 so the first pass has one truth. Each is a knob, each is a separate short run:

- **Systemic-z error.** Perturb the injected z relative to the "catalog" z by the MOSDEF uncertainty. Directly measures the smearing that has so far only been reasoned about analytically.
- **Astrometric jitter.** 0.2–0.5″ offset between truth and catalog position. Smears the inner bin and biases `h1` high.
- **Sub-pixel phase.** Force the injected line center to specific phases relative to the 110–140 km s⁻¹ grid. Centroid estimators are not obviously unbiased at that pixel scale.

---

## 6. Pre-registered success criteria

Written before the data exists, on purpose. The failure mode to avoid is landing the run and then deciding what counts as passing.

All truth values below come from the §3.5 generator, never from a config value.

1. **Family H's recovered Δv(r) reproduces the forward-modeled step.** Four sub-criteria, all required:
   - **1a. Outer bins (r > 200 kpc) recover +100 km s⁻¹ within 1σ, and the uncertainty on that recovery is quoted.** *This is the sensitivity statement the large-radius null structurally requires (§0.4), and it is now the single most important output of this spec.* The uncertainty **is** the detection floor — the smallest offset that could have been seen — and it goes in the paper beside the null.
   - **1b.** Core bins (r < 200 kpc) are consistent with zero at the precision of the real measurement's quoted uncertainty. *The false-positive null.*
   - **1c.** The recovered transition sits at 200 kpc, **and no spurious feature appears at R_vir**. A break that migrates to R_vir means the binning imposes structure there — a direct threat to Paper I's headline (§4.2).
   - **1d.** The `[180, 350]` straddling bin matches the flux-weighted blend, not either pure value (§4.2).
2. **The P/H core pair gives `a` ≈ 1 and `b` ≈ 0** (§4.3). A nonzero `b` is an absolute velocity zero-point offset surviving the air→vacuum fix and must be reported; `a` ≠ 1 is velocity dilution and calibrates every quoted velocity amplitude.
3. **Family P's recovered `h1` is consistent with zero**, and the resulting deconvolution floor lies below the real measured `h1`. If the floor is *above* the measured `h1`, the measurement is not resolved and the paper's central scale length has to be re-framed as an upper limit.
4. **Family P's outer/2-halo term is not detected**, or is detected at an amplitude far below the real measurement's.
5. **Family H's recovered `h1`, `r_c`, and crossover radius are unbiased** at better than the real measurement's uncertainty on each.
6. **Coverage of the nominal 68% interval falls in [0.60, 0.76]** for `h1` and for per-bin Δv. Outside that, quote an inflation factor.
7. **Residuals are flat in `nn_dist`, fiber count, z, M\*, and distance to nearest real source** (§5.4).

**1a is the criterion the paper cannot ship without**, because it is the only thing that converts "we see nothing at large radius" into "we would have seen X." 1b, 2, 3, 4, and 6 have real teeth as null and calibration tests. 5 and 7 are hygiene.

---

## 7. Pre-implementation validation

**Things to check before writing `inject_sources`, in rough order of how badly each would poison the result.**

### 7.1 Flux normalization — the one most likely to go wrong

The question "how do I convert an injected Lyα luminosity into a number added to fiber *f* in exposure *e*?" requires three conventions to be right simultaneously:

1. the physical units of `vdfi['spectra']` (per fiber, per Å, per what solid angle);
2. the normalization of the Moffat PSF returned by `moffat_psf_integration` — does it integrate to unity over 2π r dr, or over the fiber aperture, or neither;
3. whether the fiber's finite aperture is already folded into (2) or must be applied separately.

**This is exactly the failure that produced the 2026-07-22 area-convention bug**, where a missing `2π r` and annulus-area normalization collapsed the wide-PSF `h1` and invalidated every prior PSF fit. The same three conventions are involved. Do not infer them from the code's structure — **validate them numerically** with the round-trip in §7.2 before building anything on top.

### 7.2 The validation ladder — proving the template is real

**Split one problem into two, because they fail differently and debug differently:**

1. **Is the template what I think it is?** — a question about the generator code.
2. **Does the pipeline recover the template?** — a question about the pipeline.

Conflating them gives a test where a failure has two causes and no way to tell which. Validate (1) to completion first, with the pipeline held fixed and trivially predictable.

**The trick that makes (1) cheap: inject into a zero cube.** Real fiber positions, real seeing, real DAR, real masking — but `spectra` set to zero. Extract. The recovered profile *is* the template, with no noise and no real signal, and it can be compared against an independently computed prediction (§3.5).

**This is not the closed loop of §0.1, and the distinction is load-bearing.** A noiseless synthetic cube is illegitimate for validating the *pipeline*, because it assumes the instrument model the pipeline also assumes. It is entirely legitimate for validating the *injector*, because the only question being asked is "did flux land where I asked it to" — a question about my code, not about the instrument.

Then climb one dimension at a time. **Nothing at level *n* is built until *n−1* passes.**

| Level | Cube | Source | Known answer | What it catches |
|---|---|---|---|---|
| 0 | none | template on a fine grid | ∫ = injected L; analytic asymptotics; observed width = √(σ²+LSF²) | generator unit bugs, **LSF convention** |
| 1 | zero | point source, Δv = 0 | recovered profile = the PSF, **exactly** | **the entire 2π / aperture / normalization class (§7.1)** |
| 2 | zero | `expcore` + power law, Δv = 0 | profile ⊛ PSF, computed independently | profile evaluation, bin assignment |
| 3 | zero | + constant Δv = −50 (family P) | −50 in every bin with flux | velocity zero point through the chain |
| 4 | zero | + step Δv at 200 kpc (family H) | §3.5 forward-modeled Δv per bin | flux-weighted bin truth (a), blending (b) |
| ~~5~~ | — | ~~+ skew α~~ | **REMOVED v5** — skew is out of scope (§3.3) | — |
| 6 | **real** | one bright source | levels 2–4 within a few percent | overlay indexing, real-data interaction |
| 7 | **real** | full family | production | — |

**LEVEL 1 PASSES ON REAL COSMOS DATA, 2026-08-16.** 132/132 exposures, worst binned-Δv deviation **0.356 km s⁻¹ against a 145 km s⁻¹ pixel** (0.25% of a pixel), flux conservation 0.92 of the geometric prediction, profile shape flat to 4.87%. Levels 0 and 1 are now closed and `check_level0` / `check_level1` are the two calls that re-run them. **The next step is build-order item 2 — `inject_sources` + the `InjectedSpectra` overlay — which gates every remaining rung.**

Earlier record, for context —

**Levels 0 and 1 PASS as of 2026-08-07** (two independent numerical paths agreeing to 0.64%, with three deliberate convention breaks confirmed to fail the check). The LSF checks were added at level 0 on 2026-08-16 and pass, including three fresh break tests: the LSF omitted entirely (−22% narrow), α held fixed through the convolution (18.5% shape error), and the LSF supplied as a FWHM rather than a σ (533 vs 320 km s⁻¹). Levels 2–4 are next and cost no extraction time.

**Level 1 alone catches the bug most likely to occur.** A point source into a zero cube must return the PSF and nothing else; a stray 2π or a missing annulus area is a clean multiplicative offset visible at a glance, rather than a subtle distortion buried in a stacked average.

**Levels 3–5 are where the §3.5 traps live.** Level 3's answer is trivially checkable (−50 everywhere), which is what makes it the right place to certify the velocity machinery before adding radial structure at level 4. If level 4's recovered Δv(r) disagrees with §3.5's prediction while level 3 passed, the fault is in the truth generator's flux-weighting, not in the injector.

**Level 6 is the acceptance test for the whole build.** One bright source in the real cube, recovering levels 2–5 to a few percent, validates §7.1's conventions, the per-exposure seeing and DAR path, the fiber-aperture integration, and the overlay indexing simultaneously.

Levels 0–5 cost minutes each and need no extraction time worth counting; they are pure code. The ~56 hr in §4.5 buys level 7 only.

### 7.3 The biweight is not linear — you cannot validate by differencing

`combine_fibers` defaults to a biweight, which is not a linear operator, so

```
extract(real + injected)  ≠  extract(real) + extract(injected)
```

This is fine — it is the whole point of injecting into real data rather than simulating separately — but two consequences must be recorded:

- **You cannot verify injection by differencing an injected and an un-injected extraction.** Use the §7.2 round-trip instead.
- **`store_spec_nobg` is off for all injected runs.** Its guard relies on the biweight being translation-equivariant under a *constant* per-fiber offset, and an injected source is not constant across fibers. Closed by decision rather than analysis (Austin, 2026-08-01): this test is about whether measurements match injections, not about background-product internals.

### 7.4 Regression: empty injection is bit-identical

`InjectedSpectra` with zero patches must return bit-identical arrays to the raw memmap, and an extraction through it must produce a byte-identical galaxy FITS to one without it. This is a cheap test and it is the only thing standing between "the overlay works" and "the overlay silently drops a patch".

### 7.5 Cache signature actually changes

Same config with and without injection → different `extract_signature()`. Assert it in a test. The consequence of getting this wrong is an injected result silently served from, or written over, a fiducial cache — the one failure mode in this whole spec that would corrupt existing science products rather than merely producing a wrong test.

### 7.6 The bootstrapped (z, M\*) pairs are all solvable

`radial_bin_edges` goes through the SMHM relation, which has unsolvable regions — the switch to `behroozi19` left 14 galaxies unsolvable at the turnover. Bootstrapping pairs from the *real* run table should inherit only solvable values, but assert it rather than assume it, because a silent NaN in the bin edges would propagate to a NaN profile that looks like missing data rather than like a bug.

### 7.7 The injected profile really is negligible in the background annulus

§2.2 asserts this. Make it a printed check at injection setup: evaluate the injected profile at 80–90″ and compare to the measured background noise. One line, and it converts the central assumption of the placement scheme into a number.

---

## 8. Build order

1. **Template generator + §3.5 truth generator, as two independent implementations.** Ladder levels 0–2 (§7.2). Nothing downstream is meaningful until these agree.
2. `inject_sources` + `InjectedSpectra` overlay (§1.2), with the decision gate on indexing patterns, plus the §7.4 regression test.
3. Injection-parameter hashing into `extract_signature_dict` (§1.3) + `INJECT` FITS keyword, plus the §7.5 test.
4. Kinematics + skew in both generators. **Ladder levels 3–5.**
5. Placement: `draw_injection_positions`, both modes, `nn_dist` and distance-to-nearest-real-source recorded (§2).
6. **Ladder level 6** — one bright source into the real cube. The acceptance gate.
7. **Family Z run.** Shakes out 2–6 end to end and delivers the covariance matrix.
8. **Family P**, then family H.
9. Recovery/diagnostics module (§5) — buildable in parallel with 7–8, and should be, so the analysis exists before the data does.
10. `guide.py` registration + `check_guide()` clean, per standing practice.

Estimated implementation: ~1 week to a working injection, most of it in items 1, 2, and 4. Ladder levels 0–5 are pure code and cost no extraction time; the ~56 hr in §4.5 buys level 7 only.

---

## 9. What this closes

| TODO item | How |
|---|---|
| "Close the correlated outer-bin noise" (flagged load-bearing for all three papers) | Family Z → bin–bin covariance, Hartlap-corrected, N = 400/field |
| "Double-Gaussian null test" (deferred) | Family Z → template-on-noise fits at high N |
| "Systematics: solidify and defend the pipeline choices" | §5, §6 — the general answer to the general item |
| `background-diagnostics.md` v3 `anchor_standoff_A` | Retained there explicitly "as a swept parameter for the injection test to accept or reject" — this is that test |
| PSF-fix regression coverage (2026-07-22) | Family P is the regression test the area-convention bug never had |

---

## 10. Open questions

1. **[FILL] blockers.** §3.2 profile parameters (`h1`, `f`, `r_c`, `gamma`), §3.3 intrinsic line σ, §2.3 `n_min_fibers`. **α and the LSF are closed in v5** (α = 0, out of scope; σ_LSF = 200 km s⁻¹ from the 1.9 Å rest-frame FWHM). Velocities are now set: P = −50, H = 0 → +100 at 200 kpc. The profile parameters are gated on the post-PSF-fix, post-SMHM-switch refit (TODO item 2) — injecting stale values gives a test whose truth does not correspond to the measurement it validates.
2. **Family P's injected luminosity.** Bootstrapping L from the real catalog (§2.4) keeps H comparable to the real stack, but for P the relevant scale is different: P exists to measure a *floor*, and the floor depends on S/N. Consider running P at the median real luminosity *and* at the faint quartile, since the deconvolution floor at the faint end is the number that actually limits the sample.
3. **Does family P need a different `n_min_fibers`?** A point source deposits all its flux in the innermost bin, so P is far more sensitive to inner-bin fiber coverage than H is. The §2.3 threshold may need to be stricter for P.
4. ~~**Skew estimator on family P.**~~ **CLOSED v5** — skew is out of scope entirely (§3.3), so there is no P skew row to adjudicate.

5. **⚠ NEW (v6, 2026-08-16): the outermost bin is anomalous, and `r_trunc` sitting exactly on its outer edge is the suspect.** Two independent symptoms in the same bin: recovered/predicted surface brightness reads **1.0135** where every other bin is within 0.3% of 1.000 (and it is what pushed the shape check to 4.87% against a 5% tolerance), and truth/intrinsic reads **0.774** where the flat mid-radius bins read 1.000. Both appeared when the 3000 kpc edge was dropped, which moved the hard truncation knee (`dlnV/dlnr` = −215 at the cut) from the unused `[2000, 3000]` bin onto the outer edge of the last *measured* bin. **The naive explanation does not cover it:** a 2″ PSF at a 240″ boundary should cost ~2% by rim area, not 23%. Both generators see the same knee so the comparison stays self-consistent and nothing downstream is wrong, but the size is unexplained. Options if it matters: a softer taper, or `r_trunc` beyond the aperture — and the latter reopens what `L` means (§3.2). **Decide before quoting anything from the outermost bin.**

6. **Does the `[180, 350]` straddling bin dominate the transition-localization result?** If most of the constraint on the recovered break location comes from that one blended bin, move the break to the 180 kpc edge and re-run — §4.2 keeps 200 on the argument that sub-bin structure is the more realistic test, but that argument fails if the blended bin is also the only informative one.

**Closed in v4:** kpc-vs-virial binning (kpc; v3's smearing objection withdrawn, §4.2); N per field (400 stands at p = 12, §4.4); injected velocities (P = −50, H = 0 → +100 at 200 kpc); the second-amplitude sensitivity run (dropped — the uncertainty on the recovered +100 *is* the floor, and linearity comes from the P/H core pair, §4.3); "how do we prove the template works" (§3.5 + §7.2 ladder).

**Closed in v3:** zero-gradient vs. gradient-recovery (piecewise Δv(r) delivers both in one pass, §4.3); the deferred G-C/G-E passes (removed); `store_spec_nobg` on injected runs (off, by decision, §7.3).

**Closed in v2:** third-family question (absorbed into family P, §5.3); family Z's N (computed, §4.4); `blank` vs `random` (dissolved by post-hoc slicing, §2.1); AGN analogue (out of scope per Austin, 2026-08-01 — the real sample's AGN exclusion is a selection question, not a pipeline-recovery question, and injection cannot speak to it either way).
