# Systematics & Extraction — the full register and what to do about it

**Status: register v2, 2026-07-25.** v1 (same day) contained a wrong claim about background self-subtraction — corrected in §1.1. Supersedes `docs/systematics-audit.md` and the scope of `GOALS.md`'s five-item validation list.

## Goal

Two questions, answered together because they're the same question:

1. Which systematics can actually change the paper's numbers, and which are decoration?
2. Which can be tested without a 13–19 hr re-extraction, and what would have to change in `extract_fibers` to move more of them into the free column?

## Success criteria

- Every number the paper reports has either (a) a second, structurally independent way to compute it, or (b) an explicit, quantified systematic band.
- Every [HIGH] item has a verdict written into `docs/research-notes.md`.
- The Methods section can state *why* each pipeline choice was made, not just what it was.

---

## 0. Corrections and hard constraints

### 0.1 Assumptions corrected during the 2026-07-25 review

| Was assumed | Actually true | Consequence |
|---|---|---|
| Virial bins to 20 R_vir | **kpc bins to 3000 kpc**, solid to ~2000 | Outer-radius systematics matter far more |
| Offset-position null is cheap | `GalaxyProduct` stores `spec (ngal, nrad, nwave)` — **fibers destroyed at write** | All spatial re-tests need full re-extraction |
| Moffat is the fiducial Lyα PSF | Empirical stellar curve available; `build_effective_psf_uv` is the **UV/CFHT-LS** path, unrelated to Lyα | Lyα PSF provenance still unconfirmed |
| Fixed window clips *outer* line wings | Line is **widest in the core**; outer bins narrower | Bias direction inverted — §1.2 |
| Two-halo params are nuisance-only | `A2` feeds the crossover radius → 1-halo/2-halo luminosity and kinematics | Projection formalism matters — §4 |
| **Background annulus self-subtracts the Lyα signal** | **Wrong.** The background has Lyα masked, interpolated across, and smoothed *before* subtraction. The line is not subtracted. | §1.1 rewritten; the residual concern is a much narrower and more tractable one |
| Sub-binning wide annuli is statistically risky | **Measured: it isn't.** Equal-area sub-bins merged with area weights reproduce the wide-bin biweight to ~0.1% at every noise level tested | §2.3, §5 |

### 0.2 The extraction boundary

Everything at or before `extract_fibers` costs 13–19 hr. Everything after is free.

**Before (expensive):** fiber masking, background annulus, background mask/interp/smooth, radial bin edges, fiber coadd statistic, per-annulus collapse.
**After (free):** rest-frame regridding, galaxy coadd, continuum model, line window, centroid estimator, all fitting, bootstrapping, model comparison.

Because the fiber dimension is destroyed, the pipeline can never re-derive a background from a different annulus, re-mask, or re-bin. Any systematic living in that collapse is permanently baked in unless the extraction re-runs. This is the single most important planning fact in the document.

### 0.3 B8 resolved

Confirmed from `extract.py` §1: the galaxy-level background is the biweight of **only the fibers in `[bg_inner, bg_outer)` = 80–90″**, concatenated across that galaxy's exposures. Not science-bin fibers. With `min_bg_fibers = 999999` the per-exposure branch never fires, so every exposure gets this pooled spectrum. This is what was intended and what the code does.

Scale note: the 80–90″ annulus is ≈5,300 arcsec², the outermost science bin (3000 kpc ≈ 357″) is ≈400,000 arcsec² — **75× larger**. The background is built from far fewer fibers than the bins it's subtracted from, which is why the 30 Å smoothing is load-bearing rather than cosmetic.

### 0.4 On synthetic data

Broadly agreed: synthetic data rarely carries what real data carries, and the register prefers internal consistency tests on the real data. **But there is one class where synthetic data is exactly right — questions about an *estimator* rather than about the sky.** "How does biweight-of-biweights compare to biweight-of-all" has nothing to do with astrophysics; it's a property of the estimator and can be settled cleanly in simulation. §5 is that case, and it's now settled.

---

## 1. Free tests (Stage 2 only)

### 1.1 [HIGH] Lyα leakage into the background — the corrected version

**The v1 claim was wrong.** The background is not raw annulus flux: `smooth_interp_then_gauss` masks ±`line_mask_halfwidth_A` (observed) around the galaxy's Lyα, linearly interpolates across the gap, then Gaussian-smooths. So the Lyα *line* in the annulus is removed before subtraction, the integrated line flux is not self-subtracted at leading order, and a positive signal at 3 Mpc is not contradictory. Retracted.

**What is actually at risk is narrower, and worth pinning down.**

The mask is `line_mask_halfwidth_A = 10` **observed** Å = ±3.0 Å rest = **±740 km s⁻¹** at z = 2.3. Two leaks survive that:

1. **Anchor contamination.** Linear interpolation is anchored on the pixels immediately *outside* the mask. If the annulus carries a two-halo Lyα component, that component is broad — its velocity extent is set by the clustering along the line of sight (±740 km s⁻¹ is only ±10.5 cMpc; the correlation function has power well beyond that), so it does not stop at the mask edge. Elevated anchors → interpolated level sits *above* the true background → over-subtraction under the line.
2. **Smoothing re-injection.** The 30 Å Gaussian runs *after* interpolation, on an array that still contains the unmasked wings. It spreads those wings back into the gap.

Both leaks push the same way (over-subtraction), and both are reduced by a wider mask — which supports the ±20–30 Å instinct in §2.2.

**Now the good news, which is a genuine defense of the method and belongs in the paper.** Whatever leaks through is then partially removed a second time by the science-side continuum fit. If the background error were a *smooth* function on ≳80 Å scales, the sideband continuum fit (1180–1205 / 1225–1260 rest) would absorb it entirely and it would cancel. So the design is doubly protected.

**What survives that double protection** is specifically a *localized* deviation centered on Lyα with a width set by the mask + smoothing scale — roughly 30–40 Å observed ≈ 10–12 Å rest. That is narrower than the ~80 Å sideband span, so a linear continuum fit will *not* remove it, and it sits exactly on the line. Sign: a dip (under-measured flux). Magnitude: same absolute amount in every radial bin, so the *fractional* effect grows with radius — the outer profile is biased down.

**The direct test.** Stack the per-galaxy background spectra in the rest frame (each shifted by its own z) and look at 1215.67 Å. Lyα-free annulus → nothing there. Two-halo component → a feature, whose width tells you directly whether ±10 Å observed is wide enough. Control: the same stack in the **observed** frame, where a real Lyα feature would *not* appear but an instrumental/sky artifact would.

**Blocked today.** B9 resolved: the stored `Background` HDU is written **post-mask and post-smoothing** (`Background[cnt] = bg_gal`, assigned after `_maybe_smooth`). So the feature is already interpolated over and smoothed away in what's on disk — the test needs the raw annulus coadd, which means it waits for §2.1. **This is the strongest single argument for doing §2.1, and for storing `BG_VARIANTS` raw, unmasked, and unsmoothed.**

This remains the leading candidate explanation for the unexplained monotonic rise-and-fall of background level with radius (research-notes 2026-07-13); check it against the leak amplitude once §2.1 makes it measurable.

### 1.1b [HIGH] The interpolation is anchored on exactly two noisy pixels

Confirmed by reading `smooth_interp_then_gauss`:

```python
flux[~fill] = np.interp(wave[~fill], wave[fill], flux[fill])
```

`np.interp` is piecewise-linear between the supplied anchor points, so **every pixel inside the masked gap is determined by exactly two values: the last good pixel before the gap and the first good pixel after it.** At ~2 Å/pixel and a ±10 Å mask, a ~10-pixel gap is being filled from 2 samples. Your instinct — "it could be interpolating off corner noise, which could be a problem" — is exactly right, and it's a property of the estimator, not a suspicion.

Two consequences, with different severities:

- **Noise.** The interpolated segment inherits the full single-pixel noise of two endpoints (no √N averaging) plus a random tilt between them. This is *partially* mitigated downstream: the subsequent σ = 30 Å (≈15 px) Gaussian means only ~26% of the final smoothed value at line center comes from the interpolated region, the rest from real data. So the noise penalty is real but ~4× diluted.
- **Bias.** *Not* mitigated. If the anchors sit on the wings of a broad two-halo Lyα component (§1.1), the interpolation carries that elevation across the gap at full strength, and the smoothing pulls in more of the same wings directly. Both leaks add.

**Two fixes, both fast and vectorized — speed was the reason for the current design, and neither gives it up:**

1. **Window-based anchors.** Instead of two endpoint pixels, take a median (or fit a line) over a window of ~40 Å on each side of the mask and evaluate that across the gap. Cost: two array slices and a 2-point line — negligible. Cuts endpoint noise by ~√n_pix (4–5×), and lets you place the anchor windows *outside* the line wings, which addresses the bias too. This is the "general slope" idea, and it's the minimal change.
2. **Normalized convolution.** Since you're smoothing anyway, replace mask+interp+smooth with `gaussian(flux·m) / gaussian(m)`, where `m` is the good-pixel mask. That fills the gap with a σ-weighted average of *all* nearby good pixels — no two-point dependence at all. Cost: one extra `gaussian_filter1d`, i.e. ~2× the current smoothing, nowhere near the 100× of the legacy adaptive smoother.

The docstring records that division-fill was tried and rejected for "bowing / pulse artifacts." That's a real failure mode when the gap is *wide relative to σ* — but the current parameters have a 20 Å gap inside a 30 Å kernel, where the gap is *narrower* than σ and bowing should be minimal. Worth a re-test at current parameters rather than inheriting a verdict formed under the old 200 Å adaptive scheme.

**This is testable offline today, and it's the one background test that doesn't wait for §2.1:** take the stored (already smooth, line-free) `Background` spectra as ground truth, add realistic noise, then run each interpolation scheme and see which recovers the truth best. Real background *shapes*, synthetic noise — the shape is what matters and it's real, so this sidesteps the usual synthetic-data objection.

### 1.2 [HIGH] Flux window vs. centroid window — run it as a *test*, not a commitment

You're right that decoupling them is inelegant. So don't commit to it — **measure whether it matters, then decide.**

The concern: you noted the core line is wider in FWHM, so a fixed ±3 Å window under-measures core brightness. The consequence is that inner flux is under-measured *relative to* outer flux, the measured profile is shallower than truth, and therefore **`h1` is biased high** — the halo looks more extended than it is. That's a non-conservative bias sitting on the headline number.

The centroid genuinely does need a fixed window — your reasoning is correct, a radius-varying window makes `v(r)` inconsistent bin to bin and a wide window in a narrow-line outer bin only adds noise. Nothing here proposes changing that.

The flux profile has no such requirement. So the test is:

1. Measure the flux profile at ±3 Å (fiducial) and at ±6, ±8, ±10 Å.
2. Refit `h1` on each.
3. If `h1` is stable → **you keep the elegant single-window version and get to say it's window-insensitive.** That's a better outcome than the current silence.
4. If `h1` moves significantly → you needed to know, and the fix is a per-bin curve of growth (`windowed_flux_scan` already computes most of it) or an enclosed-flux-fraction window.

Framed this way it costs one Stage-2 run and probably ends with you keeping what you have, with a defensible sentence attached.

### 1.3 [HIGH] Confirm the noise-underestimate fingerprint

`integrated_line_flux` quadrature-sums per-pixel errors, assuming independence. `rest_delta = 0.2 Å` against a native rest sampling of ~2.0/(1+z) ≈ 0.61 Å at z = 2.3 is ~3× oversampling, so those errors can be low by up to √3 ≈ 1.7×. You bootstrap for real errors, so this is contained, but:

- Run `error_cross_check`; confirm propagated < bootstrap by roughly that factor in the line window. One sentence in the paper if confirmed.
- No `err_sum`-derived number should reach a table. Same for `describe_fit`'s `pcov`-derived `h1_err` (diagonal σ, correlated bins). Bootstrap only.
- Run `compare_bin_correlation_resolution` (0.2 vs 0.6) to confirm the headline doesn't move.

### 1.4 [MED] Bin radius assignment in wide outer bins

`_radius_points` assigns the **geometric** midpoint √(r₁r₂). For a 1000–2000 kpc bin that's 1414 kpc, but the area/flux-weighted mean radius of the fibers in that annulus is ≈1550 kpc — **~10% further out**. The fit is unaffected (the ring-convolution matrix integrates over the true annulus), but every plotted point and every `v(r)` point sits ~10% too far in, in the outer bins only. Free fix, removes an easy referee question.

### 1.5 [MED] The negative-clipping bias — a free argument that probably closes it

You use median50 with negatives clipped and have deprioritized injection-bias work as expensive and hard to do right. Reasonable, and there's an argument that mostly closes it for free:

In a pure-noise bin with negatives clipped, the cumulative-50% point converges to the **center of the measurement window**. If that window is symmetric about rest Lyα, the center is 0 km s⁻¹. So the noise-induced bias pulls the centroid **toward the null**, making any nonzero measurement conservative by construction.

Two free checks make it airtight: confirm the window is exactly symmetric about 1215.67 Å (the notebook already prints an `OFF-CENTER` flag — read it), and identify from `plot_sn_spectrum` where the bins stop having line S/N, since the argument says those bins report ≈0 regardless of truth and should be shaded or dropped rather than interpreted.

### 1.6 [MED] Continuum *method* sweep, not just bounds

The existing sweep varies bounds across five `median`/order-1 settings and never compares `median` against `poly` orders 1/2/3. In outer bins where the continuum is near zero and noise-dominated, the two have different biases and the difference should grow with radius. Free — add the method axis and check for radius dependence.

### 1.7 [MED] Point-source rejection as a number

Fit the PSF-convolved model with the core forced compact (`h1 → 0`) and run `compare_models_aic_bic` against the fiducial. Report ΔAIC. This is the quantitative replacement for "the star PSF falls below the Lyα profile in log-log" and belongs in the abstract. See §3.1.

---

## 2. Extraction-gated work

### 2.1 [HIGH] The multi-variant extraction — highest-leverage change available

Make `extract_fibers` write, in **one pass**, the products that let background choices be tested offline forever:

1. **`SPEC_NOBG`** — per-annulus fiber coadd with **no background subtracted**.
2. **`BG_VARIANTS`** — `(nvar, nwave)` candidate backgrounds per galaxy (80–90″, 120–150″, 150–200″), stored **raw and unsmoothed, unmasked**.
3. **`BG_NFIB`** — fiber count per variant.
4. **`BG_HALVES`** *(if cheap)* — the annulus split by azimuth into two independent halves. Their difference directly measures the background's spatial variance, which speaks to B10/B11.

Because background subtraction is a per-wavelength scalar subtraction, `SPEC_NOBG − smooth(mask(BG_VARIANTS[k]))` reproduces any variant in Stage 2 at zero cost. That turns *every* background systematic — annulus radius, smoothing σ, mask half-width, smoothing method, and the §1.1 leak — into a free, permanent sweep. Storing the variants **unmasked and unsmoothed** is what makes §1.1's rest-frame stack test possible directly.

Cost: `SPEC_NOBG` doubles the science array; the variants are `(ngal, nvar, nwave)`, negligible.

**Check first:** confirm the annulus fibers are still in scope when the science coadd is written, and add a header keyword recording which variant (if any) was pre-subtracted, so nothing downstream silently assumes `spec` is background-subtracted.

### 2.2 [HIGH] Mask width and smoothing σ are one joint knob

Per §1.1, the two surviving leaks are anchor contamination and smoothing re-injection. Both shrink with a wider mask; the smoothing σ controls how far the wings reach back in. They are **not independent** and should be swept jointly, not one at a time.

Setting the target correctly matters: the mask should be set against **the expected width of the two-halo component, not a single galaxy's line width.** Since ±740 km s⁻¹ = ±10.5 cMpc (§4) is already inside the clustering's velocity extent, ±20–30 Å observed is the range that actually brackets it. Your ±10–20 Å instinct is in the right direction; the upper end is probably where it needs to land.

Free after §2.1. Until then it's one full extraction per setting — which is the argument for doing §2.1 first rather than testing this separately.

### 2.3 [MED→ now cheap] Sub-binning the outer annuli — statistically validated, see §5

Your objection was right for the inner bins and is now measured to be wrong for the outer ones (§5.2). Splitting a high-fiber-count bin into equal-area sub-bins and merging with area weights reproduces the single wide-bin biweight to ~0.1% at all noise levels tested, with essentially identical variance.

**Recommendation:** keep inner edges exactly as they are; subdivide only bins with fiber counts above ~50k. The outer range is where the background leak (§1.1), the projection break (§4), and the two-halo measurement all live, and going from ~2 to ~8 bins between 500 and 3000 kpc converts several arguments in this document into measurements.

The "chunks vs. a continuous radial wheel" concern is handled correctly in the fit already — `ring_convolution_matrix` integrates the model over each true annulus with `2πr` weighting and never treats a bin as a point. It affects only the reported bin radius (§1.4).

### 2.4 [HIGH] Extraction architecture and speed — the enabler

Read end to end 2026-07-25: `load_vdfi_field` → `extract_fibers` → `combine_fibers_chunked` → `combine_fibers`.

**How it works today.** `spectra` and `error` are **memmapped** FITS arrays of shape `(nexp, nfib, nwave)`, handles deliberately kept open. `extract_fibers` loops galaxies as the outer loop; per galaxy it (1) computes distances for every fiber in every exposure, (2) reads the 80–90″ annulus fibers via `spectra[e, bg_sel]`, biweights and smooths them into a background, (3) digitizes every fiber into a radial bin, (4) for each radial bin, gathers `spectra[e, sel]` across exposures into a preallocated float32 pool, subtracts the background, and calls `combine_fibers_chunked` with `wave_block = 64`. There's a per-galaxy `.npz` cache keyed on a config signature.

**Reassurance first:** the wavelength chunking is over the *wavelength* axis, so each wavelength still sees all fibers. It is **statistically exactly lossless** — pure memory management, no estimator consequence. (Different operation from the fiber-chunking question in §5.)

#### Where the time probably goes — two hypotheses, one profiling run to decide

**Hypothesis A: I/O-bound.** The galaxy loop is outer and the memmap read is inner, so **the flux file is traversed roughly once per galaxy** — ~450 partial passes. If the file exceeds RAM (24 GB on JupyterHub), the OS page cache thrashes: consecutive galaxies touch overlapping but non-identical fiber sets, so pages are evicted before reuse. On a shared filesystem this dominates everything else.

**Hypothesis B: CPU-bound on the biweight.** If the flux file actually fits in page cache (plausible — ~1–2 M fibers × 1036 × 4 B ≈ 4–8 GB), the first galaxy warms the cache and later reads are RAM-speed. Then the cost is the robust combine over the largest bins.

These call for completely different fixes, so **profile first.** Instrument one galaxy with `time.perf_counter()` around the four stages (distance/digitize, background read+combine, per-bin gather, per-bin combine) and report the split. Also record `psutil` RSS and compare the flux file size to available RAM. An afternoon, and it decides everything below.

#### If I/O-bound: invert the loops (large win, identical results)

The fix is to make **wavelength the outer loop and galaxies the inner loop**:

```
Pass A  (backgrounds; small — annulus fibers are ~1–2% of the array)
    for wave_block:
        read the block once; biweight the bg-annulus fibers for every galaxy
    assemble full-wavelength raw backgrounds, THEN mask+interp+smooth each
    (smoothing is non-local in λ, so it cannot run inside a block)

Pass B  (science)
    for wave_block:
        read spectra[:, :, j0:j1] once            # one sequential read
        for galaxy, for bin:
            gather → subtract precomputed bg slice → biweight
```

This reads the flux file **twice total instead of ~450 times**. Peak memory is still governed by `wave_block` (a 64-pixel block is `nexp × nfib × 64 × 4 B` ≈ 0.5–2 GB), and the largest per-bin pool drops to 500k × 64 × 4 = 128 MB. Because the biweight is independent per wavelength, results are **bit-identical**.

Two costs to plan for: the per-galaxy `.npz` cache becomes a per-wave-block checkpoint instead, and the fiber→bin assignment either gets recomputed per block (pure CPU on RA/DEC, no I/O — cheap) or cached as compact per-bin index arrays (~1 GB total). Neither is hard; both need doing deliberately rather than incidentally.

#### If CPU-bound: three specific wins

1. **`biweight` and `mad_std` each recompute the median and MAD of the same array.** In `combine_fibers`, `biweight(spec, axis=0, ignore_nan=True)` internally computes a nan-median and nan-MAD, then `mad_std(spec, axis=0, ignore_nan=True)` computes both again. On the dominant operation that's roughly **2× redundant work**. Fusing them into one pass is a contained change to `combine_fibers` and should give ~30–40% on the combine.
2. **Is `Err` even used?** The Stage-1 `Err` for robust methods is just `scatter/√N`, it feeds `err_sum`, and errors come from the bootstrap anyway (§1.3). If nothing downstream consumes it, dropping `mad_std` entirely is an immediate ~2× on the combine — **the cheapest speedup available.** Needs confirming: does any galaxy-axis coadd (`inv_var`, `weighted_median`) or plotting path read it? (Open question.)
3. **Parallelize over wave blocks.** Fully independent; `joblib.Parallel` over the block loop gives near-linear scaling with cores and *no* increase in peak memory, since each worker holds one block. This composes with the loop inversion above rather than competing with it.

#### Regardless of which hypothesis wins

- **Request a real TACC allocation.** A compute node has far more RAM and cores than a JupyterHub session; if memory stops binding, `wave_block` goes back up and the chunking overhead disappears. Ask before the data doubles, not after.
- **Measure the `wave_block` knee.** Time one galaxy at 8/16/32/64/128 recording peak RSS. It's a 1D curve with a knee; the current 64 was chosen to survive, not to be optimal.
- **`gc.collect()` runs once per galaxy** (~450 full collections over a heap holding multi-GB arrays). Cheap to test whether it's costing real time; `del` plus a generational collect may be enough.
- **Any restructuring must preserve bit-identical output.** Keep the existing per-galaxy cache from one full run as a regression fixture and assert equality before trusting a rewritten extractor. This is exactly the "smoothly wrong" failure mode that output inspection misses.

---

## 3. The PSF chain

### 3.1 What the log-log comparison proves, and what it doesn't

Necessary but weak: it depends on where you normalize (normalizing in the innermost bin, which already contains halo flux, inflates the PSF curve and makes the halo look *less* extended — conservative, but say so); it produces no number; and it constrains the PSF only where the PSF is measured.

The strong version you can already state: `h1 ≈ 16–17 kpc` against a 1.3″ Moffat with α = 10.7 kpc, HWHM ≈ 5.5 kpc at z = 2.3 — the fitted scale is **~3× the PSF half-width** — plus the FWHM sweep showing `h1` stable at 15.1 kpc (1.3″) → 12.0 kpc (2.4″). With bootstrap intervals attached, that's a real result. §1.7 adds the ΔAIC.

### 3.2 [HIGH] The wing slope is the open risk

The stellar profile dies at 10–20″ (84–168 kpc at z = 2.3). Science extends to 3000 kpc. **The PSF is unconstrained over ~95% of the measured radial range.**

Moffat β = 3 falls as r⁻⁶. Real ground-based PSFs carry aureoles going as r⁻² to r^(−2.5) (King 1971; Racine 1996; Bernstein 2007), typically 1–10% of total flux. Flux fraction beyond R:

| R | Moffat β=3 | 2% r⁻² aureole (25→840 kpc) | ratio |
|---|---|---|---|
| 30 kpc | 1.3% | 1.9% | 1.5× |
| 75 kpc | 0.040% | 1.4% | **35×** |
| 150 kpc | 0.0026% | 0.98% | **380×** |

Two mitigations specific to your setup:

- **10–20″ is 10–15× the FWHM — a decent lever arm.** If the measured stellar profile still tracks β = 3 out there, an aureole carrying ≥2% would likely already show. **So don't extrapolate: fit a free power law to the outer part of the measured stellar profile and report the slope with its error.** That converts "unknown" into "constrained steeper than X" and is the cheapest high-value PSF work available.
- **Fibers are not a CCD.** Much of the aureole literature concerns wide-field imaging with long glass paths. VIRUS may differ. Check whether HETDEX has published a measured VIRUS PSF wing profile — that would close this outright.

If the stellar profile can't pin the slope, propagate a pessimistic aureole as an explicit one-sided systematic: refit with PSF = (measured core) + (r⁻², ε = 2%) and quote the shift in `h1`. The fitting path takes an arbitrary `(psf_r, psf_vals)` array, so no new fitting code is needed.

### 3.3 [HIGH] Astrometry — a free correction

Star centers come from **Gaia** (~mas). Galaxy centers from **MOSDEF/CANDELS** (~0.1–0.3″). The stars are effectively perfectly centered; the galaxies are not.

A random per-galaxy positional error σ_pos convolves into the stacked galaxy profile as an extra Gaussian kernel: 0.2″ = 1.7 kpc, 0.3″ = 2.5 kpc, 0.5″ = 4.2 kpc at z = 2.3, against `h1 ≈ 16 kpc`. **It biases toward your claimed result** — galaxies look more extended than stars, for free.

Your own fix is right and costs nothing: **convolve the measured stellar PSF with a Gaussian of width σ_pos before using it as the galaxy PSF.** It's a 1D array convolution. Get σ_pos from the catalog's positional uncertainties or the astrometric residuals; convolve; refit; report the shift as a correction or a systematic. Do this before any PSF claim goes in the paper.

### 3.4 [MED] Which PSF does the Lyα fit actually use?

`build_effective_psf_uv` is the **UV/CFHT-LS** path — different dataset, different optics, nothing transfers. Confirm for Lyα specifically:

- Empirical stellar curve or analytic Moffat as fiducial?
- Is the kpc conversion done over the **full redshift distribution** (flux-normalize each galaxy's kpc-scaled PSF, then average) or a single median z? A median-z PSF is narrower than the true sample-averaged one, under-corrects, and inflates apparent extent. You believe it's the distribution — worth confirming in code, since the output looks fine either way.
- Is `optics.wavelength_corrected_seeing` applied? Seeing ∝ λ^(−1/5); small, free.
- Are the ~30 (EGS) / ~90 (COSMOS) stars stacked per field with each field's own PSF?

### 3.5 [MED] Are the stars background-subtracted the same way?

If star extraction subtracts an 80–90″ annulus background and stars have wings there, you'd be subtracting the aureole you're trying to measure — making the PSF look artificially compact and biasing toward "the halo is more extended." Given the profile dies at 10–20″ this is probably moot, but the failure mode is silent and points the wrong way, so confirm rather than assume.

Saturation: you avoid saturated stars, but a partially nonlinear core under-normalizes total flux and makes wings look *relatively* stronger — conservative. One linearity check on the brightest stars lets you state the direction.

---

## 4. [HIGH] The projection breaks at your outer radii — and it explains the `r_c` degeneracy

### 4.1 The numbers

The ±3 Å rest window at z = 2.3 is ±740 km s⁻¹. With H(2.3) ≈ 233 km s⁻¹ Mpc⁻¹:

- **π_max ≈ ±10.5 cMpc = ±3.2 pMpc**

Maximum transverse separation: 3000 pkpc = 3 pMpc ≈ **9.9 cMpc**. The projection depth and the maximum transverse scale are **essentially equal**. Your instinct that it might fail out there was right.

Limber-type projection — and the `γ = 0.8` that comes from projecting `γ_3D = 1.8` — is valid only for `r_p ≪ π_max`. When `r_p ≳ π_max` the line-of-sight integral saturates and the projected slope **steepens from −0.8 toward −1.8**.

### 4.2 What I'm actually proposing — concretely

**Current model, second term:**
```
Σ₂(r) = A2 · (1 + (r/r_c)²)^(−γ/2)          γ = 0.8 fixed, r_c = 400 kpc fixed
```
Three shape parameters (A2, r_c, γ), two of them fixed by hand, one (`r_c`) notoriously degenerate.

**Proposed replacement — the actual finite-depth projection of a 3D power law:**
```
Σ₂(r_p) = A2 · ∫₀^{π_max} [ max(√(r_p² + π²), r_soft) / 1000 ]^(−γ_3D) dπ
```
- `γ_3D = 1.8` — literature (Adelberger 2005 z~2–3 galaxy autocorrelation), same source `γ = 0.8` already came from.
- `π_max = 3200 kpc` — **set by your own ±3 Å window**, not free.
- `r_soft ≈ 75 kpc` — regularizes the r → 0 divergence of a bare power law (this is what the Plummer core was doing). Unlike `r_c` it can be fixed at a physically motivated value (the virial radius, where one-halo ends), and the exponential core dominates there anyway so the fit is insensitive to it.

**Net: one free parameter (A2) instead of three, and `r_c` disappears from the model entirely.**

It's a 1D numerical integral on the existing fine grid — a change to `intrinsic_profile_expcore`'s second term, nothing else in the fitting machinery moves.

### 4.3 The two shapes are not remotely the same function

Computed, both normalized at 300 kpc:

| r_p (kpc) | finite-depth | cored γ=0.8 | ratio | slope (fd) | slope (cored) |
|---|---|---|---|---|---|
| 50 | 3.73 | 1.19 | 3.14 | −0.19 | −0.01 |
| 116 | 2.27 | 1.16 | 1.96 | −0.84 | −0.06 |
| 268 | 1.10 | 1.03 | 1.07 | −0.88 | −0.25 |
| 621 | 0.51 | 0.73 | 0.70 | −0.97 | −0.57 |
| 1439 | 0.21 | 0.42 | 0.50 | −1.17 | −0.74 |
| 2189 | 0.12 | 0.30 | 0.41 | −1.32 | −0.77 |

**(a) The current form over-predicts the outer profile** relative to the projection: at 2200 kpc it's 2.4× higher, with slope −0.77 where the finite-depth projection gives −1.32 (heading to −1.8).

**(b) The current form is nearly flat inside ~300 kpc** (slope −0.01 to −0.25), where the projected form is already at −0.85.

### 4.4 Scope — deliberately limited

**Austin's position, adopted:** the fit works well in practice; `A2` is a real flattening that appears right after the crossover, not a subtraction residual; and the outer power-law slope can be more or less whatever it needs to be given the data. The goal is to extract science, not to maximize formal correctness of a nuisance term. So the following is scoped down accordingly.

**What is *not* being claimed.** An earlier draft argued that the near-flat inner behaviour of the cored term makes `A2` degenerate with a background pedestal, and floated that as the cause of the `r_c` spread. That was inferred from the functional form alone, not from the actual fits, and Austin — who has seen them — reports the flattening is a real, reproducible feature. Retained below only as a cheap falsifiable check, not as a diagnosis.

**What is worth doing, in priority order:**

1. **[HIGH, free, no model change] Interpretation caveat.** Whatever functional form is used, the projection depth (±10.5 cMpc) is comparable to the largest transverse separation (9.9 cMpc). **So the fitted outer slope must not be interpreted as a clustering slope**, and `γ = 0.8` should not be described as "the Limber-projected clustering slope" in the paper if the fit range extends past ~1 Mpc. One sentence, no code. This is the part that actually has to happen.
2. **[MED, cheap] Pedestal check.** Refit with a free additive constant alongside the current model. If `A2` and the crossover radius are stable, the pedestal concern is dead and you can say so in one line. Expected outcome given Austin's read: stable. Worth the ten minutes precisely because it converts an argument into a result.
3. **[MED, optional] Projected form as an A/B.** Implement the finite-depth term as an *alternative*, not a replacement, and run `compare_models_aic_bic`. Its selling point is that it removes `r_c` entirely (one free parameter instead of three, with `π_max` set by your own window). If AIC/BIC prefers it and the crossover radius tightens, adopt it; if not, the exercise still yields a defensible "we tested the physically-motivated projection and the empirical form fits at least as well," which is a stronger Methods sentence than silence.
4. **[LOW] π_max varies ~20% from z = 2 to z = 3** (fixed rest window, changing H(z)), so the effective kernel is a z-weighted mixture, structurally like the PSF. Fold in only if (3) is adopted and shows sensitivity.

The crossover radius is defined as where the exponential and the two-halo term become equal, so it survives any of these unchanged — no downstream measurement breaks.

### 4.5 Literature framing

The amplitude is degenerate with the softening and slope, and not reporting it is the right call. But the **shape** argument is not degenerate — π_max is fixed by your own window. The references are already in `docs/literature/`:

- **Rakic 2012**, **Rudie 2012**, **Chen 2020** — HI absorption clustering around z ≈ 2–3 galaxies, with explicit 3D→2D projection and finite-π_max treatment. Direct methodological analogues.
- **Mas-Ribas 2017** — Lyα intensity mapping; the emissivity-side version.
- **Byrohl 2023** — the simulation side.
- **Adelberger 2005** — the z~2–3 autocorrelation `γ_3D ≈ 1.8` that `γ = 0.8` derives from.

Target sentence: *"our ±3 Å window corresponds to a line-of-sight depth of ±10.5 cMpc, comparable to our largest transverse separation; we therefore adopt the finite-depth projection of a γ_3D = 1.8 correlation function rather than the Limber-limit slope, following [refs]."* Cites the slope instead of assuming it, and reports no degenerate parameters.

---

## 5. Biweight statistics — chunking, merging, and what a wide bin measures

Both questions settled by simulation. This is the case where synthetic data is the *right* tool, because the question is about the estimator, not the sky.

### 5.1 What does chunking a biweight cost?

"If I have 100 points and biweight them in chunks of 10, then biweight the 10 results, what happens versus biweighting all 100?" Monte Carlo, 600 trials per cell, scatter is the trial-to-trial standard deviation:

| distribution | N | chunks | 1-stage bw | 2-stage bw | 2-stage, **mean**-merge | plain mean |
|---|---|---|---|---|---|---|
| Gaussian (truth 0) | 100 | 10 | −0.011 ± 0.102 | −0.008 ± 0.110 | −0.011 ± **0.105** | −0.009 ± 0.099 |
| Gaussian | 1000 | 10 | +0.000 ± 0.034 | −0.000 ± 0.037 | +0.000 ± **0.035** | −0.000 ± 0.033 |
| Gaussian | 1000 | 100 | +0.000 ± 0.034 | +0.000 ± 0.036 | −0.000 ± 0.036 | +0.000 ± 0.033 |
| +5% outliers @ +10σ | 1000 | 10 | +0.002 ± 0.033 | +0.001 ± 0.036 | +0.002 ± 0.033 | **+0.505** ± 0.072 |
| lognormal (skewed) | 1000 | 10 | +1.045 ± 0.045 | +1.043 ± 0.048 | +1.049 ± 0.044 | +1.649 ± 0.069 |
| student-t₃ | 1000 | 10 | +0.004 ± 0.037 | +0.004 ± 0.040 | +0.004 ± 0.038 | +0.001 ± 0.052 |

**Conclusions:**

1. **No detectable bias from chunking.** One-stage and two-stage agree to within Monte Carlo error in every case, including skewed and heavy-tailed.
2. **Variance cost is small: ~7–10%** for biweight-of-biweights.
3. **Merging with an n-weighted *mean* instead of a second biweight is better** — cost drops to ~2–3%. This makes sense: the sub-biweights have already rejected outliers, so a second robust stage rejects nothing and only loses efficiency. **If you ever chunk over fibers, merge with a weighted mean, not another biweight.**
4. **The biweight is doing real work.** With 5% outliers the plain mean is biased by +0.5σ while every biweight variant is unbiased; on heavy tails the biweight beats the mean on variance (0.037 vs 0.052).
5. **The biweight is ~4.5% biased on skewed data** (1.045 vs a true median of 1.0) — but chunking does *not* make that worse. It's a property of the estimator, and if fiber flux distributions are positively skewed it applies to the current pipeline as-is. Worth quantifying on real fiber distributions at some point.
6. **The wavelength-chunking memory patch is exactly lossless** — it chunks the *wavelength* axis, so each wavelength still sees all fibers. No estimator consequence at all. Separate question from the above.

### 5.2 Does a wide-bin biweight measure the area-weighted mean?

The more important version, because the forward model (`bin_average_psf`) predicts an **area-weighted mean surface brightness**, and if the data statistic measures something else there's a model–data mismatch. Simulated: fibers drawn area-uniform in a 1000–2000 kpc annulus over a declining `r^−0.8` profile, Gaussian noise, compared against the analytic area-weighted mean (0.72078):

| n_fib | S/N per fiber | one wide biweight | 4 equal-area sub-bins, area-weighted merge |
|---|---|---|---|
| 2,000 | 0.02 | 0.7378 ± 0.858 | 0.7415 ± 0.863 |
| 2,000 | 0.20 | 0.7221 ± 0.083 | 0.7219 ± 0.083 |
| 2,000 | 2.00 | 0.7197 ± 0.009 | 0.7204 ± 0.009 |
| 200,000 | 0.02 | 0.7215 ± 0.095 | 0.7215 ± 0.095 |
| 200,000 | 0.20 | 0.7203 ± 0.009 | 0.7204 ± 0.009 |
| 200,000 | 2.00 | 0.7200 ± 0.0009 | 0.7208 ± 0.0009 |

**Conclusions:**

1. **The wide-bin biweight does recover the area-weighted mean** — 0.7200–0.7215 against a truth of 0.72078, at every noise level. So the forward model's area-averaging convention is correct and there is no model–data mismatch. Good news, and worth one sentence in the Methods.
2. **Sub-binning and merging is statistically free** — agreement to ~0.1%, with identical variance. Your A3 objection was right about the inner bins (where sub-bins would have single-digit fiber counts) and does not apply to the outer bins.
3. This is what makes §2.3 safe to do.

**Caveats:** tested with ≥500 fibers per sub-bin; do not extrapolate to sub-bins with tens of fibers. Tested with Gaussian noise — worth one re-run with a realistically skewed/outlier-contaminated fiber distribution before committing, which is cheap.

**Scripts:** `bw2.py`, `bin_test.py` (see §8 — should be moved into the repo as a reproducible test).

---

## 6. Compact register of everything else

| # | Systematic | Direction | Cost | Priority |
|---|---|---|---|---|
| 6.1 | Percentile masking (99.5/0.5) clips bright fibers beyond the 10″ protect radius | Outer profile biased **down**; two-halo is a lower limit | Extraction | MED |
| 6.2 | Segmap masks detected neighbours, which *are* part of the two-halo signal | Two-halo lower limit, depth-dependent, differs by field | Extraction, or state as caveat | MED |
| 6.3 | Per-exposure vs. pooled background | Settled by investigation; pooled is less noisy | — | closed |
| 6.4 | Sky residuals correlated across amp/IFU | Source of correlated outer-bin noise | Only measurable via random-position null | MED |
| 6.5 | Wavelength zero-point | 0.1 Å at 4000 Å = 7.5 km s⁻¹; common-mode → shape safe | Free (read reduction residuals) | MED |
| 6.6 | Field heterogeneity (COSMOS ~4× depth) | Different effective PSF, noise, segmap depth | Free (report per-field band) | MED |
| 6.7 | Cosmic variance on the two-halo term | Two fields = one degree of freedom | Free (state) | LOW |
| 6.8 | Redshift provenance | Hα/[OIII] nebular — **not circular**; σ≈60 km s⁻¹ random | — | closed |
| 6.9 | AGN removal (X-ray + BPT + IRAC, 70/510) | Standard; residual undetected AGN brighten inner bins | Free (state completeness) | LOW |
| 6.10 | Biweight nonlinearity → stack PSF ≠ average PSF | First-order only | Free-ish | LOW |
| 6.11 | Biweight bias on skewed fiber distributions (~4.5% in test) | Unknown sign on real data | Free (measure real fiber skew) | MED |
| 6.12 | Mass–environment degeneracy on subsample splits | Can silently re-derive the mass trend | Free (mass-match) | MED |
| 6.13 | Double-Gaussian template circularity (split #9) | Fitting-induced structure possible | Needs a null construction | MED (blocks #9 only) |

---

## 7. The random-position null, revised

Offset positions are dead (§0.2), so this is a single expensive decision.

**Unique value:** every existing null breaks the *spectral* association with real fibers at real positions. Only this breaks the *spatial* association, and it is the only route to the empirical outer-bin covariance — which a galaxy bootstrap cannot give, because resampling galaxies that share exposures reuses the same correlated sky.

**How many**, with N ≈ 452 after AGN cuts:

- `N_r = N` → blank-sky errors **equal** to the data's; supports only "consistent with zero at the level of my own error bar." Weak.
- Constraining the floor to fraction *f* of the measurement error needs `N_r ≈ N/f²`: f = 0.5 → ~1800; f = 0.3 → ~5000.
- Empirical covariance needs k ≥ 20 independent stacked realizations → ~9000. Unaffordable.

**Recommendation: N_r ≈ 3–5N (1400–2300)**, and only after §2.4 makes it affordable. Below 3N it doesn't earn its runtime.

**Design requirements:** draw positions weighted by **exposure coverage** rather than uniformly in an RA/DEC box (the current `make_random_sky_table` oversamples low-coverage edges, precisely where the suspected anomaly lives); pair `(z, mass)` one-to-one with real galaxies rather than resampling; match the distribution of real galaxies falling inside each random position's 80–90″ annulus; and run it in the **same pass as §2.1**.

---

## 8. Prioritized plan

**Now — free, and can change the paper**

1. §2.4 **profile one galaxy's extraction** (I/O vs. distance/digitize vs. background vs. combine) and compare flux-file size to RAM. Decides the whole extraction strategy; everything expensive downstream depends on it. *Agreed next action.*
2. §1.1b **interpolation-anchor test** — window anchors and normalized convolution vs. the current two-pixel `np.interp`, evaluated on real stored background shapes with synthetic noise. The one background test that runs today.
3. §1.2 flux-window test at ±3/6/8/10 Å; refit `h1`; keep the single-window version if stable.
4. §3.3 convolve the stellar PSF with σ_pos; refit; quote the shift.
5. §3.2 fit a free power law to the outer stellar profile (10–20″); report the wing slope with error.
6. §4.4 item 1 — the interpretation caveat (one sentence, no code); then item 2, the pedestal check.
7. §1.7 point-source rejection ΔAIC.
8. §1.3 confirm propagated-vs-bootstrap ratio; purge `err_sum` and `pcov` errors from all output.
9. §3.4 confirm what PSF the Lyα path uses (empirical vs. Moffat; z-distribution vs. median z).
10. §1.4, §1.5, §1.6 — cheap closures.
11. Fill in TBD verdicts in `research-notes.md` from one `summarize_validation` run.

**Next — one afternoon each, unlocks the rest**

12. §2.4 implement whichever branch the profile selects (loop inversion if I/O-bound; fuse `biweight`/`mad_std`, drop unused `Err`, parallelize blocks if CPU-bound). Keep a full cached run as a bit-identical regression fixture.
13. §2.4 request a TACC allocation; measure the `wave_block` knee.
14. §5.2 re-run the binning simulation with a realistically skewed fiber distribution before committing to §2.3.

**Then — one extraction, designed to be the last for a while**

14. §2.1 multi-variant extraction (`SPEC_NOBG` + raw unmasked `BG_VARIANTS` + `BG_NFIB`, ideally + `BG_HALVES`).
15. §2.3 subdivide only the >50k-fiber outer bins in the same pass.
16. §2.2 wider Lyα mask — free *after* §2.1, so fold it in rather than testing separately.
17. §6.1 masking-percentile variants, if they can ride along.
18. §7 random-position null at 3–5N, once §2.4 makes it affordable.

---

## 9. Answers recorded 2026-07-25

- **B9** — the stored `Background` HDU is **post-mask and post-smoothing**. §1.1's direct test therefore waits on §2.1.
- **Flux and centroid use the identical ±3 Å window**; they do not currently differ. §1.2 is a genuine change, not a documentation fix.
- **Outer-bin bootstrap S/N: ≈6, 5, 2.8** in the outer three bins. So the outermost bin is ~3σ — real but not comfortable, and the first thing a systematic would eat.
- **Crossover radius** = where the exponential and the two-halo term become equal. Survives any §4 change.
- **`A2` is a real post-crossover flattening**, per the actual fits — not a subtraction residual. §4.4 scoped down accordingly.
- **Stars:** ~50 per field or fewer; a stacked stellar radial profile already exists; profile dies at 10–20″.
- **AGN cut:** X-ray + BPT + IRAC combined, removes ~70 of ~510.
- **Redshifts:** MOSDEF nebular, Hα or [O III]. Not circular.
- **Continuum sidebands:** line-free windows, unchanged; exact values in `config`.

## 10. Still open

- **A1** — what dominates the 13–19 hr. Blocks the §2.4 branch decision. **Next action.**
- Is Stage-1 `Err` consumed by anything downstream? If not, dropping `mad_std` is a free ~2× on the combine (§2.4).
- **B10/B11** — the monotonic rise-and-fall of background level with radius. Check against the §1.1 leak once §2.1 makes it measurable.
- **C12** — how star extraction is configured (mask, background, bin grid, centering). Needed before §3.2/§3.5 can be trusted.
- Whether HETDEX has published a measured VIRUS PSF wing profile — would close §3.2 outright.
- Real fiber-flux skewness within an annulus, for §5.1/§6.11.
- Where `bw2.py` / `bin_test.py` should live in the repo as reproducible tests.

## See also

- `docs/psf-aware-fitting-explained.md` — the forward model §3 stress-tests
- `docs/research-notes.md` — reasoning trail; §"Systematics tests" holds the TBD verdict list
- `specs/bin-correlation-diagnostics.md` — covariance machinery §1.3 wants in the fit
- `specs/halo-flux-fitting.md` — the fit §4 proposes changing
- `validation.py`, `starpsf.py`, `psf_forensics.py`, `optimize.py`, `extract.py`
