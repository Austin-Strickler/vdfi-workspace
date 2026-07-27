# Systematics & Extraction — the full register and what to do about it

**Status: register v4, 2026-07-26.** v1 (2026-07-25) contained a wrong claim about background self-subtraction — corrected in §1.1. v2 added the extraction-boundary framing and B8/B9. v3 closed §1.2 (flux-window test run; verdict: keep the fixed ±3 Å window). **v4 (2026-07-26)** records Austin's decisions on the v3 plan: §1.6 closed (already done, method-independent), §1.1b deferred until real raw backgrounds exist, §3.2+§3.3 merged into one star re-extraction (§3.6), §1.7 scoped as quantification-not-evidence, §2.4 closed (extraction speedup shipped), and the **background-variant custom extraction (§2.5) promoted to the next action** — it is affordable *now* and unblocks §1.1/§1.1b/§2.2/B10/B11 together. Supersedes `docs/systematics-audit.md` and the scope of `GOALS.md`'s five-item validation list.

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
| **COSMOS is ~4× deeper than EGS** (v1–v3, §6.6) | **Wrong, and backwards. Corrected by Austin 2026-07-26: EGS is the *deeper* field, roughly 2×. COSMOS covers a much larger area; EGS roughly half the area. The two FITS files come out similar in size for that reason** — which is probably the observation that got misread into "equal depth over unequal area" and then inverted. | Any per-field reasoning in v1–v3 that leaned on COSMOS being deeper is suspect and should be re-derived. §6.6 corrected; §2.5.4 and §3.6 corrected. **COSMOS's advantage is *area*, so it dominates the outer/two-halo bins; EGS's advantage is *depth*, so it dominates inner-bin S/N.** That is the opposite of the v3 framing and it matters for §2.5, §6.6, §6.7, and the star stacks |
| Sub-binning wide annuli is statistically risky | **Measured: it isn't.** Equal-area sub-bins merged with area weights reproduce the wide-bin biweight to ~0.1% at every noise level tested | §2.3, §5 |

### 0.2 The extraction boundary

Everything at or before `extract_fibers` costs ~8.5 hr for the full sample (was 13–19 hr before the 2026-07-25 parallel-combine work, §2.4). Everything after is free.

**v4 amendment — the boundary is softer than v3 assumed.** `config.galaxy_window()` (`gal_offset` + `num_gal`) already slices the run table, and the per-galaxy cache is keyed on the *global* row index, so a partial run is a config change and nothing else. At ~68 s/galaxy, **100 galaxies ≈ 1.9 hr**. That reclassifies a whole tier of work: any systematic whose test needs a *stacked, spatially-averaged* quantity rather than the final profile amplitude can be settled on a ~100-galaxy subset today, at decision-grade precision, without waiting on the full run. §2.5 is the first and most valuable instance.

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

### 1.1b [RESOLVED 2026-07-27] The interpolation is anchored on exactly two noisy pixels — real, and measurably harmless

> **Verdict: keep `linear`. The concern below is correct as a description of the estimator and wrong as a prediction about the result.** Tooling: `smoothing.smooth_fill_then_gauss` (three fill methods), `background.compare_fill_methods`, `background.compare_background_impact`. Full numbers in **§2.6**.
>
> Measured on 278 COSMOS galaxies with real raw `BG_VARIANTS`, exactly the non-circular test this section deferred for:
>
> - `fill=median` and `fill=median_interp` move the binned flux profile by **≤0.73σ** and the centroid by less, in every bin. That is ~2% of the no-background bracket.
> - The **bias** half of the concern — anchors sitting on a broad two-halo wing — did not materialise at a detectable level. §1.1's rest-frame background stack is the direct test of whether that wing exists; the fill comparison is the test of whether the *anchor* mechanism converts it into a profile error, and it does not.
> - **Why, mechanically:** the fill only touches ±10 Å observed (±3.0 Å rest), and that region is then convolved with a 30 Å kernel. Whatever the fill does is diluted roughly 10:1 before it reaches the data — consistent with the ~26% / ~4× dilution estimated below, which was right.
> - **`min_bg_fibers=999999` already defends the noise half.** Pooling every fiber across every exposure makes the two mask-edge anchor pixels well-determined, which is precisely the failure mode window-based anchors were proposed to fix. The fix was already in place for a different reason.
>
> **Where `median_interp` still earns its place:** per-galaxy work (`single.py`, bright AGN/LAE inspection), where there is no 278-galaxy average to wash out one bad edge pixel. Recommend `linear` for the stack, `median_interp` as the default for single-object analysis.
>
> **Normalized convolution (option 2 below) was not tested and is not needed.** With `linear` and `median_interp` agreeing to ≤0.73σ, a third scheme cannot plausibly disagree with both by more.
>
> `fill_method="linear"` is asserted **bit-identical** to the production `smooth_interp_then_gauss` path on every `compare_background_impact` run (measured max|diff| = 0.0), so the sweep is not silently comparing two things that differ for a second reason.

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

**[v4: DEFERRED to §2.5 — Austin's call, and it is the right one.]** v3 proposed testing the interpolation schemes today against the stored `Background` spectra with synthetic noise added. The objection: the stored backgrounds are *already* masked, interpolated, and smoothed (B9), so that test would evaluate each scheme's ability to recover a spectrum that the current scheme itself produced. The failure mode it most needs to detect — anchors sitting on a real broad two-halo wing — **cannot appear in that ground truth by construction**, because the wing was interpolated away before the array was written. It would measure the noise term (which §1.1b already argues is ~4× diluted and not the severe half) while being structurally blind to the bias term (which is the severe half).

So this test runs *after* §2.5 delivers raw, unmasked, unsmoothed annulus coadds, at which point it becomes the far stronger version: compare each scheme against the **real** background including whatever line feature is actually there, with no synthetic component at all. Cost of waiting: ~2 hr. Value gained: the test stops being circular.

### 1.2 [RESOLVED 2026-07-25] Flux window vs. centroid window — tested, fixed ±3 Å window kept

**Verdict: keep the fixed ±3 Å window for both centroid and flux. The variable/Gaussian-sigma window was tested and rejected.** Tooling: `validation.sweep_flux_window`, `validation.gaussian_sigma_per_bin`, `validation.gaussian_flux_window`, `validation.h1_vs_flux_window`, `validation.plot_flux_window_diagnostic` (all Stage-2-only, no re-extraction).

The original concern: the core line is wider in FWHM, so a fixed ±3 Å window under-measures core brightness relative to outer bins, making the measured profile shallower than truth and biasing `h1` high. The centroid genuinely needs a fixed window regardless (a radius-varying window makes `v(r)` inconsistent bin to bin); only the flux integrator was ever a candidate for decoupling.

**What was run.** (1) `sweep_flux_window` at ±3/5/6/8/10 Å, refitting `h1` (PSF-aware expcore) on each via `h1_vs_flux_window`, with refit-per-draw errors (`analysis.bootstrap_fit_profile`, per §1.3 — no pcov-derived errors). (2) `gaussian_flux_window` at `k_sigma` swept from 0.6 to 2.7, same refit.

**What it showed.** The fixed-width sweep left `h1` consistent within errors across ±3–±10 Å — widening the window mainly *grows the error bars* (more continuum/sky noise enters), it does not shift the central value in a way that clears its own uncertainty. That alone would satisfy this section's original success criterion (§1.2 point 3: stable `h1` → keep the single-window version). The Gaussian-window sweep, by contrast, gave a *smooth, monotonic* `h1` trend across the entire `k_sigma` range with no plateau at either end — the signature of an inconsistently-defined window, not a real core/halo effect (a fixed `k·σ` only holds the same enclosed-flux *fraction* bin to bin if the line is a pure Gaussian in every bin).

**Why, from `plot_flux_window_diagnostic` inspection.** Two compounding problems, worse in the inner bins:

1. **The inner bins are low-fiber-count**, so despite carrying the highest aggregate line signal, the per-pixel S/N is poor enough that the Gaussian sigma fit is not reliably constrained — the fit is unstable exactly where it matters most for setting the window.
2. **Lyα-forest absorption blueward of line center is commonly present and is real**, not noise. A wider window (fixed *or* Gaussian-derived) increasingly integrates over that absorption trough, which *removes* flux rather than capturing more of it. This confounds "wider window → more captured emission" with "wider window → more forest absorption subtracted off," and it hits hardest in the same low-fiber inner bins where the line is intrinsically strongest.

**Conclusion (Austin's read, confirmed by the above):** the small integrated flux the ±3 Å window reports in the inner two bins is not an under-measurement artifact — it is the more trustworthy number, because it stays inside the region least contaminated by forest absorption while the wider/Gaussian windows are demonstrably less stable, not more complete. The ±3 Å window **acquires more of the true line flux than the wider fits recover**, once absorption contamination and fit instability are accounted for, and it does so with the simplicity of a single fixed window. No extraction-side change follows from this (it was always free/Stage-2); no further action needed on this item.

### 1.3 [HIGH] Confirm the noise-underestimate fingerprint

`integrated_line_flux` quadrature-sums per-pixel errors, assuming independence. `rest_delta = 0.2 Å` against a native rest sampling of ~2.0/(1+z) ≈ 0.61 Å at z = 2.3 is ~3× oversampling, so those errors can be low by up to √3 ≈ 1.7×. You bootstrap for real errors, so this is contained, but:

- Run `error_cross_check`; confirm propagated < bootstrap by roughly that factor in the line window. One sentence in the paper if confirmed.
- No `err_sum`-derived number should reach a table. Same for `describe_fit`'s `pcov`-derived `h1_err` (diagonal σ, correlated bins). Bootstrap only.
- Run `compare_bin_correlation_resolution` (0.2 vs 0.6) to confirm the headline doesn't move.

### 1.4 [MED] Bin radius assignment in wide outer bins

`_radius_points` assigns the **geometric** midpoint √(r₁r₂). For a 1000–2000 kpc bin that's 1414 kpc, but the area/flux-weighted mean radius of the fibers in that annulus is ≈1550 kpc — **~10% further out**. The fit is unaffected (the ring-convolution matrix integrates over the true annulus), but every plotted point and every `v(r)` point sits ~10% too far in, in the outer bins only. Free fix, removes an easy referee question.

**[v4] Open sub-question, per Austin: what convention does the literature use?** Worth settling before changing anything, because the answer determines whether this is a *fix* or a *deviation*. Candidates in play: geometric midpoint (current), arithmetic midpoint, area-weighted mean radius `⟨r⟩ = (2/3)(r₂³−r₁³)/(r₂²−r₁²)`, and the flux-weighted (profile-dependent, self-referential) radius. Check Leclercq 2017, Wisotzki 2018, Niemeyer 2022, Guo 2024 — Niemeyer especially, since the external-replication comparison (GOALS.md) is only apples-to-apples if the radius convention matches hers. **Note that the new fine bin grid shrinks this effect substantially** (bin widths in the outer range drop from ~1000 kpc to ~200 kpc), so this may fall below the noise entirely — check the size of the shift on the new grid before spending effort on it.

### 1.5 [MED] The negative-clipping bias — a free argument that probably closes it

You use median50 with negatives clipped and have deprioritized injection-bias work as expensive and hard to do right. Reasonable, and there's an argument that mostly closes it for free:

In a pure-noise bin with negatives clipped, the cumulative-50% point converges to the **center of the measurement window**. If that window is symmetric about rest Lyα, the center is 0 km s⁻¹. So the noise-induced bias pulls the centroid **toward the null**, making any nonzero measurement conservative by construction.

Two free checks make it airtight: confirm the window is exactly symmetric about 1215.67 Å (the notebook already prints an `OFF-CENTER` flag — read it), and identify from `plot_sn_spectrum` where the bins stop having line S/N, since the argument says those bins report ≈0 regardless of truth and should be shaded or dropped rather than interpreted.

**[v4 update] Symmetry confirmed by Austin — the window is exactly symmetric about 1215.67 Å.** So the analytic argument holds and the bias is toward the null. What Austin wants beyond that, and it's a fair ask: the analytic argument tells you the *direction* and the asymptotic limit (pure noise → exactly 0 km/s), but not the **transfer function** — given a true centroid of +150 km/s at the S/N of bin *k*, what does median50-with-clipping actually report? "Biased toward zero" could mean 5% attenuation or 60%.

**Injection–recovery is the right tool and it is not a black box.** The construction is deliberately transparent: take a real measured bin spectrum, shift it by a known Δv, add noise at the measured per-bin level, run the *actual* `centroid` code path, and plot recovered vs. injected. Output is a per-bin attenuation curve `Δv_rec(Δv_true)`; its slope near zero is the attenuation factor, and it is directly invertible as a correction or quotable as a one-sided systematic. The only synthetic ingredient is the noise realization; the line shape and the estimator are both real. That places it in exactly the §0.4 category where simulation is legitimate — a question about an estimator, not about the sky.

Priority is set by where the centroid is measured relative to where line S/N dies. Do this on the *new fine grid*, since more bins at lower per-bin S/N is precisely the regime where the attenuation grows.

### 1.6 [RESOLVED 2026-07-26] Continuum *method* sweep — done, method is irrelevant

**Verdict: closed.** Austin has run the method axis (`median` vs. `poly` orders 1/2/3), not just the bounds sweep. The methods produce **essentially identical results**, with no radius-dependent divergence in the outer bins where v3 predicted the biases would separate. The continuum estimator is not a systematic. One sentence in Methods ("the continuum model was varied across median and polynomial orders 1–3 with no significant change"), no further work.

### 1.7 [MED] Point-source rejection as a number

Fit the PSF-convolved model with the core forced compact (`h1 → 0`) and run `compare_models_aic_bic` against the fiducial. Report ΔAIC. See §3.1.

**[v4] Austin's pushback — "what does this add over the log-log Lyα-vs-stellar-PSF comparison?" — is correct, and the honest answer scopes this item down.**

**It is not independent evidence.** It uses the same PSF, the same data, the same bins. If the PSF is wrong, both the log-log comparison and the ΔAIC are wrong in the same direction. Anyone treating ΔAIC as a second, corroborating line of evidence is double-counting. So: it should be described as a *quantification* of the log-log statement, not as a separate test, and it should not be given its own bullet in a list of independent checks.

**What it does add, and why it's still worth doing — three things the eyeball comparison structurally cannot give:**

1. **It uses the error bars.** The log-log plot shows two curves separating. It does not say whether the separation is large compared to the uncertainty on the data points, which is the only question a referee actually cares about. ΔAIC folds in the per-bin errors and answers it.
2. **It removes the normalization choice.** §3.1 already flags that the log-log comparison depends on where you normalize, and that normalizing in the innermost bin (which contains halo flux) inflates the PSF curve. That is an *arbitrary analyst choice sitting inside the headline argument*. The ΔAIC fit has no such choice — the point-source amplitude is a fitted free parameter, so the comparison is normalization-independent by construction. This is the strongest single reason to do it.
3. **It compares like with like.** The log-log plot overlays a stellar profile on a binned galaxy profile. The ΔAIC version passes the point-source hypothesis through the *same* forward model — same ring convolution, same bin averaging, same annulus integration — so any residual difference is genuinely about extent rather than about binning treatment.

**Recommended framing:** one sentence in Results — *"a PSF-convolved point source is disfavoured at ΔAIC = X"* — presented as the quantitative form of the profile-vs-PSF comparison, in the same paragraph, not as a separate result. Abstract-worthy only if X is large; demote from v3's "belongs in the abstract" to "belongs in the sentence where the log-log comparison is stated."

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

**[v4: SUPERSEDED BY ACTION — the fine grid is running as of 2026-07-26.]** Austin launched a full extraction on the finer grid:

```
[0, 10, 16, 24, 36, 55, 110, 200, 300, 400, 500, 600, 750, 900,
 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700]   # 22 bins
```

This does §2.3 and goes further — 15 bins beyond 500 kpc where the old grid had ~2, plus a finer inner sampling (10/16/24/36/55) that the §5.2 caveat did *not* validate. Two notes:

- **Outer-bin fiber counts confirmed adequate:** Austin reports ~1,000–50,000 per bin on the new grid, comfortably above the ≥500-per-sub-bin floor the §5.2 simulation was tested at. The statistical concern is closed for the outer range.
- **The inner bins are the ones to watch, not the outer ones.** 0–10, 10–16, and 16–24 kpc are sub-fiber-scale at HET's 1.5″ fiber (12.6 kpc at z=2.3) and will carry single-digit-to-tens fiber counts. §5.2 explicitly says *do not extrapolate to sub-bins with tens of fibers*, and §1.2 already found the inner bins are where low fiber counts made the Gaussian-sigma fits unstable. Expect the innermost 2–3 bins to be dominated by the PSF and by their own noise; plan to report them but not to interpret them independently.
- Note also that the top bin moved 3000 → 2700 kpc, which slightly *improves* the §4 projection situation (max transverse 8.9 cMpc vs. π_max 10.5 cMpc) without changing the conclusion.

The "chunks vs. a continuous radial wheel" concern is handled correctly in the fit already — `ring_convolution_matrix` integrates the model over each true annulus with `2πr` weighting and never treats a bin as a point. It affects only the reported bin radius (§1.4).

### 2.4 [SHIPPED 2026-07-25] Extraction architecture and speed — the enabler

**Resolved. A1 is closed: the extraction was CPU-bound on the combine (Hypothesis B), not I/O-bound.** The fix was threading the per-wavelength-block combine — `config.combine_threads = 8` — giving **20.3 hr → 8.5 hr on COSMOS, bit-identical output**. 12 threads measured *slower* than 8 (memory-bandwidth saturation / hypervisor vCPU contention, see `specs/extraction-speedup.md` §287). Ideas tested and rejected: dropping `Err`, spatial sort, resident array, block gather. `combine_threads` is deliberately excluded from `extract_signature`, so changing it does not invalidate the cache.

Loop inversion was **not** needed and should not be pursued — it was the fix for Hypothesis A, which lost. Remaining live items from this section: the per-run cache restructure and a galaxy-slicing parameter for SLURM arrays (the latter now exists as `gal_offset`/`num_gal` → `config.galaxy_window()`, which is what makes §2.5 possible).

The architectural description below is retained as reference, since §2.5 and §2.1 both modify this code path.

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

### 2.5 [IMPLEMENTED 2026-07-26 — ready to launch] The custom background-variant extraction

> **Code shipped 2026-07-26.** All five changes below are in the repo and covered by `test_framework.test_background_variants`. What exists now:
>
> | Piece | Where |
> |---|---|
> | `bg_variants`, `store_raw_bg`, `store_spec_nobg`, `store_bg_halves` | `config.py`, all default off/None |
> | `resolved_bg_variants()` — validates variant 0 == fiducial | `config.py` |
> | variant loop, raw BG capture, azimuthal halves, `SPEC_NOBG` | `extract.py` |
> | `SPEC_NOBG`/`BG_VARIANTS`/`BG_NFIB`/`BG_HALVES` HDUs + header keys | `io.py` |
> | Stage-2 reconstruction, background stacks, mask criterion, halves | **`background.py` (new module)**, registered in `guide.py` |
>
> **Decisions taken where the recipe left a choice:**
>
> - **Change 0 resolved by omit-when-default.** The new keys are absent from `extract_signature_dict()` at their defaults, so every existing signature is bit-identical and the in-flight fine-grid cache was never at risk. When the keys *are* set they *do* enter the hash — not because they change `SPEC`/`ERROR`/`NFIB`/`BACKGROUND` (they cannot) but because they change what a cache *entry contains*. So a variant run gets its own `cache/<sig>/` and its own FITS name automatically, and §2.5.5's isolation requirement is satisfied by construction rather than by remembering to set `cache_dir`.
> - **`bg_variants` defaults to `None`, not `((80,90),)`.** A hard default of the fiducial pair would make any config with a non-default `bg_inner_arcsec` fail the validator on construction. `None` → `((bg_inner, bg_outer),)` keeps every existing config valid.
> - **Cache safety is recompute, not fail-loud.** A cache entry missing one of the new arrays is recomputed (with one warning), not loaded short. Strictly safer than the alternative — the failure mode being guarded against is a silent NaN row, and recomputing one galaxy costs ~68 s.
> - **`extract_fibers` keeps its 5-tuple return**; the products come back via `return_extras=True` as a 6th element. `starpsf.extract_star_profiles` and every notebook that unpacks 5 values are untouched.
>
> **Two corrections to the estimates in this section:**
>
> - **Runtime is more than the "3–5%" §2.5.6 predicted.** The variant annuli are much larger than the fiducial: 80–90″ is 5,340 arcsec², but 120–150″ is 25,400 and 150–200″ is 55,000 — the two extra variants together are **15× the fiducial annulus**, or ~20% of the total science area out to 3000 kpc. `store_bg_halves` doubles the background work again (two combines per variant). Expect **~1.9 hr → ~2.5–2.8 hr** for 100 galaxies with all three variants and halves on; drop halves to save ~20%. Still cheap, but budget it correctly.
> - **§1.1's test needs `measure_line_residual`'s significance read with care.** `background.stack_backgrounds` samples the rest grid at the *native* rest step rather than something finer, deliberately: the residual error sums per-pixel bootstrap sigmas in quadrature, so an oversampled grid would inflate every significance in the module by ~√(oversampling) — the same noise-underestimate fingerprint §1.3 flags for `rest_delta = 0.2`. Verified on injected-line tests: a real line recovers at 53σ in the rest stack while the z-scrambled and observed-frame controls sit at 1.2σ and 0.1σ.
>
> **Still to do before launching (§2.5.4, unchanged):** check the run-table ordering. `galaxy_window()` is still contiguous — Austin's call — so rows `[0:100]` must be verified not to be sorted by field, mass, or z.



**This is §2.1 executed early on a ~100-galaxy subset, and it is the highest-value thing available right now.** Runtime ~1.9 hr at the measured ~68 s/galaxy. It converts §1.1 (Lyα leakage), §1.1b (interpolation anchors), §2.2 (mask width × smoothing σ), and B10/B11 (background level vs. radius) from *blocked* to *free and permanent*, all at once.

#### 2.5.1 Is a test on real data defensible? Yes — with one framing requirement

Austin asked directly. The answer is yes, and the concern people raise about real-data tests does not apply here, provided the procedure is stated correctly.

**The objection people actually have is circularity, not "real data."** Nobody objects to measuring a contaminant on real data — that is what control regions are *for*. What draws fire is *tuning a parameter on the same data whose signal you then report*. So the question is whether choosing `line_mask_halfwidth_A` from this test is circular.

**It is not, and the reason is structural:** the quantity being measured (a Lyα feature in the rest-frame stack of the **80–90″ background annulus**) and the quantity being reported (the radial surface-brightness profile of the science bins) are different measurements on different fibers. The mask width is selected by a criterion — *the background must be line-free* — that makes no reference to the profile's shape, amplitude, `h1`, or the crossover radius. That is the same logic as choosing a sky-subtraction aperture, a continuum sideband, or a masking threshold, all of which are routinely set from the data.

**The one way to make it circular, which must be avoided explicitly:** do **not** select the mask width by which value makes `h1` or the outer profile come out best. Fix the criterion *before* looking at the profiles, and write it into the doc now:

> **Pre-registered criterion.** Adopt the smallest `line_mask_halfwidth_A` for which the rest-frame stacked background shows no residual feature at 1215.67 Å above its own noise, with the observed-frame stack as the null control. Report the criterion, the residual amplitude at the adopted width, and the `h1` shift between the old (±10 Å) and adopted widths as a systematic.

Stated that way, the resulting Methods sentence is a strength rather than a hedge:

> *"We verified directly that the background annulus is free of Lyα emission at the adopted mask width by stacking the raw, unmasked background spectra of N galaxies in the rest frame; the residual at 1215.67 Å is consistent with zero at the X level, while the observed-frame control stack shows no feature, confirming the residual is not instrumental. We additionally re-derived the profile under background annuli at 80–90″, 120–150″ and 150–200″ and under mask half-widths of 10, 20 and 30 Å, finding shifts of ..."*

That is a much stronger paragraph than any simulation could produce, and it is the direct, specific answer to a referee asking whether the halo signal is an artifact of over-subtracting the line. **Recommendation: run it.**

#### 2.5.2 Why the subtract-offline trick is exact — and the one condition it depends on

The plan stores the science coadd *without* background subtraction (`SPEC_NOBG`) plus the raw backgrounds, and reconstructs any variant in Stage 2 as `SPEC_NOBG − smooth(mask(BG_VARIANTS[k]))`. This is **exact, not approximate**, for a reason worth recording: the biweight location estimator is *translation-equivariant* — subtracting the same scalar from every fiber at a given wavelength shifts the biweight by exactly that scalar. Sigma-clipping is likewise defined on deviations from the center, so it too is unaffected. Since the current code subtracts a per-wavelength scalar (`bg_per_exp[e][np.newaxis, :]`) from every fiber before combining, subtracting it *after* the combine gives bit-identical results.

**The condition: this holds only while every exposure shares the same background.** With `min_bg_fibers = 999999` the per-exposure branch never fires and all exposures get the pooled `bg_gal`, so a single per-galaxy background is subtracted everywhere and the reconstruction is exact. **If `min_bg_fibers` is ever lowered, the trick silently breaks** (different exposures would carry different offsets, which do not commute with a single post-hoc subtraction). Assert `min_bg_fibers` is large at write time and record it in the header.

Secondary caveat: `Err` currently folds `bg_err_per_exp` in quadrature, so the error array is *not* reconstructible the same way. Given §1.3 (errors come from the bootstrap, `err_sum` should never reach a table), this does not matter — but note it rather than discovering it later.

#### 2.5.3 What to write

| Product | Shape | Purpose | Notes |
|---|---|---|---|
| `SPEC_NOBG` | `(ngal, nrad, nwave)` | offline background swapping | ~18 MB at 100 gal × 22 bins × 1036 λ, float64 |
| `BG_VARIANTS` | `(ngal, nvar, nwave)` | the actual test data | **raw — pre-mask, pre-smooth**. This is the whole point |
| `BG_NFIB` | `(ngal, nvar)` | weighting the stack; sanity | |
| `BG_HALVES` | `(ngal, nvar, 2, nwave)` | annulus split by azimuth → spatial variance of the background (B10/B11) | cheap; include if it doesn't complicate the write |

Variants: `(80, 90)` (fiducial), `(120, 150)`, `(150, 200)`. Three is enough to establish whether the feature amplitude falls with annulus radius — which is the signature that distinguishes a real two-halo component from an instrumental artifact.

The single non-negotiable item is **`BG_VARIANTS` stored raw**. Everything else is convenience; that one array is what makes §1.1's test exist at all.

#### 2.5.4 Sample selection — the one real gotcha

`galaxy_window()` returns a **contiguous** slice `run_table[start:stop]`. If the run table is ordered by field, by catalog, by mass, or by redshift, then rows `[0:100]` are a *biased* subsample and the stacked background will not represent the full-sample background. **Check the ordering before launching.** If it is ordered, either take a strided/shuffled subset (safe — this run gets its own `extract_signature` directory, so it cannot collide with the fine-grid cache) or run two windows, one per field, and stack them together. The fields differ in depth (§6.6 — **EGS is the deeper field, ~2×**), so a subset that is accidentally single-field will misstate the noise level even if the feature amplitude is right. Sample sizes are comparable (278 COSMOS + 244 EGS pre-AGN-cut), so a ~50/50 split is both representative and easy.

#### 2.5.5 What this run does *not* need

Deliberately keep it cheap and orthogonal to the fine-grid run in flight:

- **Bins do not need to be fine.** The test quantity is the *background spectrum*, which does not depend on the science bin grid at all. Use the old coarse grid; it is faster and the `SPEC_NOBG` product is a bonus rather than the goal.
- **It does not need to be the same config as the production run.** It is a systematics probe, not a science product. Its numbers answer "how wide should the mask be," and the adopted answer then gets applied in the next full extraction.
- **Do not run it concurrently with the fine-grid extraction if they share a machine** — §2.4 found the combine is memory-bandwidth-bound at 8 threads, so two jobs will contend and both will slow down. Queue it after, or drop `combine_threads` for whichever runs second.

#### 2.5.6 Implementation recipe — self-contained, five changes

Written to be executable without further context. Line numbers are as of 2026-07-26.

**Change 0 (do this first): the signature hazard.**

`PipelineConfig.extract_signature_dict()` (`config.py:441`) hashes a fixed key list (`config.py:434`). **Adding a new key to that list changes every signature and orphans the in-flight fine-grid cache.** So new config fields must hash *identically to today when left at their defaults*. Two acceptable ways: omit the keys from the dict when they hold default values, or don't add them to the hashed list at all and give this run an explicit separate `cache_dir`. Verify before doing anything else — `test_framework.py:121` already has the "knob does not change signature" test pattern to copy:

```python
sig_before = cfg.extract_signature()
# ... after adding the fields, with defaults ...
assert replace(cfg).extract_signature() == sig_before
```

**Change 1 — `config.py`, new fields.**

```python
bg_variants: tuple = ((80.0, 90.0),)   # (inner, outer) arcsec; [0] must equal
                                       # (bg_inner_arcsec, bg_outer_arcsec)
store_raw_bg: bool = False             # write pre-mask, pre-smooth backgrounds
store_spec_nobg: bool = False          # write the un-background-subtracted coadd
store_bg_halves: bool = False          # azimuthal split of each annulus
```

Add a validator asserting `bg_variants[0] == (bg_inner_arcsec, bg_outer_arcsec)`, so variant 0 is always the fiducial and the existing path cannot silently change meaning.

**Change 2 — `extract.py:894–922`, the background block.**

Currently one annulus. Wrap in a variant loop; variant 0 must stay byte-identical to today.

- Allocate `BG_RAW = np.full((ngal, nvar, nwave), np.nan, dtype=np.float32)` alongside `Background` (line 826).
- Inside the loop over variants, the `bg_sel` mask at 895–897 takes `(lo, hi)` from `config.bg_variants[v]` instead of `config.bg_inner_arcsec` / `bg_outer_arcsec`.
- **The one line that is the entire point of this run.** `bg_gal_raw` already exists at line 916 and is currently discarded:

```python
BG_RAW[cnt, v] = bg_gal_raw            # RAW: before _maybe_smooth
if v == 0:
    bg_gal = _maybe_smooth(bg_gal_raw, z)   # unchanged fiducial path
    Background[cnt] = bg_gal
```

- Only variant 0 feeds `bg_per_exp` and the science subtraction. Variants 1+ are measured and stored, never subtracted.
- Cost: annulus fibers are ~1–2% of the array, so three variants add roughly 3–5% runtime, not 3×.
- `BG_NFIB[cnt, v] = n_bg_total` for the variant (sum over exposures) — needed to weight the stack.

**Change 3 — `SPEC_NOBG` without a second combine.**

Do **not** re-run the combine unsubtracted. The biweight location estimator is translation-equivariant and (with `min_bg_fibers` huge) every exposure shares the same `bg_gal`, so:

```python
if config.store_spec_nobg:
    SPEC_NOBG[cnt, r] = Spec[cnt, r] + bg_gal
```

**Guard it.** Assert at entry that `config.min_bg_fibers > nfib_total`, i.e. the per-exposure background branch (`extract.py:905`) never fires. If it ever does, different exposures carry different offsets, they do not commute with a single post-hoc subtraction, and `SPEC_NOBG` becomes silently wrong. Record `min_bg_fibers` in the FITS header. Note also that `Err` folds `bg_err_per_exp` in quadrature (lines 973–976) and is therefore *not* reconstructible this way — acceptable per §1.3 (bootstrap-only errors), but do not pretend otherwise.

**Change 4 — cache round-trip.** This is the easiest place to introduce a silent bug.

- `_save_cache_atomic` (line 998) gains `bg_raw=`, `bg_nfib=`, and `spec_nobg=`.
- **The cache-hit branch at lines 867–871 must read them back.** If it doesn't, any galaxy already cached returns NaN for the new arrays and the stack quietly loses those galaxies. Since this run uses a fresh signature dir it starts empty — but the bug appears on the first resume, which is exactly when it's hardest to notice.
- Bump the cache manifest (`_write_cache_manifest`, line 693) to record which products the dir contains, so a resume against a dir written by the old code fails loudly instead of silently.

**Change 5 — writer and run config.**

New HDUs `BG_VARIANTS` `(ngal, nvar, nwave)`, `BG_NFIB` `(ngal, nvar)`, `SPEC_NOBG` `(ngal, nrad, nwave)`. Header keywords: `BGRAW=T`, `NOBG=T`, `BGVARN`, `BGVAR0`..`BGVARn` (the annulus pairs), `MINBGFIB`, and — critically — a keyword stating **that `SPEC` is background-subtracted and with which variant**, so nothing downstream ever has to guess which of `SPEC` / `SPEC_NOBG` it holds.

Run config: coarse bins, `num_gal ≈ 100` split ~50/50 across fields (§2.5.4), explicit separate `cache_dir`, `combine_threads` lowered if anything else is running.

**Regression requirement.** Before trusting any of it: run one galaxy with `bg_variants=((80,90),)`, `store_*=False` and assert `Spec`/`Err`/`NFIB`/`Background` are **bit-identical** to the existing cached output for that galaxy. §2.4's lesson — this is the "smoothly wrong" failure mode that output inspection misses.

**Stage-2 verification, in the same session:** confirm `SPEC_NOBG − smooth(mask(BG_VARIANTS[0])) == SPEC` to floating-point tolerance. If that identity fails, the equivariance assumption or the `min_bg_fibers` guard is broken and nothing else in §2.5 can be trusted.

---

### 2.6 [RESOLVED 2026-07-27] What the subtraction actually does to the science bins

**The headline: background subtraction does not threaten the result. Every choice tested agrees within the noise, and the two alarms raised along the way were both artifacts of a bad error estimator that has since been fixed.**

Tooling: `background.compare_background_impact`, `describe_background_impact`, `plot_background_impact`, `plot_background_impact_bins`, `continuum_window_scan`. All registered in `guide.py`. Run: 278 COSMOS galaxies, `n_boot=1000`, fiducial 80–90″, alternatives 40–50 / 60–70 / 110–130″, plus a no-background bracket.

#### 2.6.1 The numbers

| choice | max |Δflux| | max |Δcentroid| |
|---|---|---|
| `fill=median`, `fill=median_interp` | ≤0.73 σ_abs | smaller |
| annulus 40–50 / 60–70 / 110–130″ | 2.41 σ_abs, **2.1 σ_paired** | 1.45 σ_abs, **1.6 σ_paired** |
| no background at all | ~2.6 σ_abs flux | **4.0 σ_abs** centroid, outermost bin |

Family-wise p across the 3 alternative annuli: **flux p = 0.10, centroid p = 0.29.** Consistent with noise. Reaching p<0.05 would take 2.4σ.

`verify_reconstruction` passes (max|diff| = 2.1e-7 vs tol 3.5e-5, 0/1968241 pixels over tolerance), so the offline reconstruction is the thing the science was measured with.

#### 2.6.2 Two claims retracted — the pedestal error was wrong

The off-line pedestal error was first estimated as `mad_std(continuum pixels)/sqrt(npix)`. **That is wrong in the dangerous direction.** It assumes independent pixels, but the stack is rest-frame resampled per galaxy (`np.interp` blends neighbours) on top of the LSF, so the effective N is several times smaller than `npix`. It understated the error by roughly the square root of the correlation length. Replaced with a galaxy bootstrap over the continuum-window pixels only (cheap — tens of pixels, not the full grid).

| | broken estimator | galaxy bootstrap |
|---|---|---|
| 1000–2000 kpc pedestal (clean bin) | +10.80σ | **+1.53σ** |
| 500–1000 kpc: fiducial vs other annuli | 1.21 vs 6.8–9.9 | 0.21 vs 0.65–1.67 |

- **"The clean-bin pedestal fails at 10.8σ"** — retracted. It is 1.53σ. The subtraction is correctly scaled everywhere it can be tested.
- **"Self-subtraction caught at 500–1000 kpc"** — mostly retracted. The *ordering* survives (fiducial lowest, as self-subtraction predicts, and 80–90″ lands at 601–756 kpc, inside that bin) but at 0.21 vs 1.67 it is not a detection.
- The +7.02σ pedestal at 0–20 kpc **is** real UV continuum from the galaxies, correctly excluded from the clean set by `pedestal_is_clean`.

**Caveat on `pedestal_is_clean` itself.** Its justification — "the 80–90″ annulus holds no real continuum" — is true of the *annulus*, which averages over near-random sky. A *science bin* at 1000–2000 kpc is centred on the galaxies and contains their clustered neighbours, which do carry UV continuum. The test may therefore be less clean than the label implies; treat a positive outer-bin pedestal as ambiguous rather than as proof of mis-scaling.

#### 2.6.3 The continuum windows sit ~1 kernel σ from the line

Real coupling, not currently resolvable:

- background smoothing σ = 30 Å **observed** = **8.7 Å rest** at z_med = 2.45
- `DEFAULT_CONT_BOUNDS` start at 1205 / 1225 Å = 9.3 / 10.7 Å from Lyα ≈ **1.1 kernel σ**, where a Gaussian still carries ~60% of peak

So the residual bump left by mask-then-smooth lands on the very pixels that define the continuum, and subtracting a background changes the continuum estimate by an amount set by that annulus's Lyα content. `continuum_window_scan` walks `cont_bounds` from 1.1 to 5.7 kernel σ and finds **no significant change** in the annulus spread. There is a faint hint in the expected direction (flux spread 2.35 → 1.61 at the furthest windows; dropping the blue window alone gives 2.35 → 1.96) but both sit inside the noise.

**Read this as "cannot resolve," not "absent."** The mechanism is live; this data cannot see its effect above the background noise. Worth re-running once EGS is folded in.

#### 2.6.4 Why the comparison is background-noise-limited — and what that costs

The paired bootstrap reports `σ_ref/σ_Δ ≈ 2.2`. Since the annuli are **disjoint sky** their backgrounds are independent, and the science data cancels in the difference:

$$\sigma_\Delta^2 = 2W,\qquad \sigma_{\rm ref}^2 = V+W,\qquad g^2 = \frac{V+W}{2W}$$

g = 2.2 → **the background contributes 10.3% of the total variance.** Two consequences:

- A *perfect, noiseless* background would shrink the error bars by only **~5%** (up to ~11% if the annuli are partially correlated, which raises W). That is the ceiling on any background-noise-reduction scheme, including a global background.
- The annulus-to-annulus scatter is dominated by the background's own sampling noise, not by anything systematic — which is why it is consistent with zero and why **more galaxies will shrink it as 1/√N**.

#### 2.6.5 Global constant background — probably not worth it, and here is the decision criterion

The per-exposure → per-galaxy change worked because it pooled across exposures *at the same sky position*: pure noise averaging, nothing else varying. **A global background crosses to different sky positions**, so it removes only the *mean* of any spatial structure in sky residuals, scattered light, and focal-plane pedestal and leaves the rest in. Locality is the product of the annulus design, not a side effect. Ceiling on the gain is the ~5% above.

**The decision is empirical and the tools already exist:** `background_variance_profile` splits the scatter into galaxy-to-galaxy vs **spatial**, and `background_half_difference` (N-half − S-half) isolates the spatial term directly. That is B10/B11. If the spatial term is negligible, global is safe and buys ~5%; if not, global converts a removed systematic into an unremoved one. **Run this before considering global.**

**Better middle path — widen the annulus.** Background noise scales as 1/√area:

| annulus | area | bg variance | error shrink | inner edge (kpc) |
|---|---|---|---|---|
| 80–90″ (current) | ×1.0 | ×1.00 | — | 601–756 |
| 110–130″ | ×2.8 | ×0.35 | 3.4% | ~924 |
| 80–110″ | ×3.4 | ×0.30 | 3.7% | ~672–924 |
| 80–130″ | ×6.2 | ×0.16 | 4.4% | ~672–1092 |

80–110″ captures most of the available gain with no spatial risk and **stays clear of the 1000–2000 kpc bin**. 80–130″ buys 0.7% more and pushes the annulus into that bin, adding self-subtraction reach to the weakest point in the profile. Recommend 80–110″ if this is pursued; 110–130″ already agrees with the fiducial within noise, so a wider annulus is partially validated.

#### 2.6.6 Effect on the reported error budget

Since the annulus differences are consistent with zero, quoting the observed shifts as a systematic would double-count noise already in σ_stat. What can be said: **the comparison's own precision is ~1× the statistical error, so a background-annulus systematic of that order cannot be excluded.** Whether to add that in quadrature (≈40% inflation on the outer bins) is a judgement call — conservative and defensible, but arguably over-conservative off a null result. **Set the bound from the EGS-included numbers rather than from COSMOS alone.**

The inner bins are untouched under every choice (≤0.04σ at 0–20 kpc), so **the R_vir transition at 50–200 kpc is unaffected.** The no-background bracket's 4.0σ centroid swing in the outermost bin is the honest upper bound on how much that one point depends on background subtraction — worth stating, even though nobody would run without subtracting.

#### 2.6.7 Methodological lessons worth reusing

1. **A nonlinear robust combine does not commute with anything.** `biweight(smooth(x)) ≠ smooth(biweight(x))`: astropy's tuning radius is 6×MAD, so any operation that changes the noise level changes which galaxies get downweighted. A pooled COSMOS+EGS background stack (bimodal: AEGIS ~0, COSMOS ~−0.0029) produced a spurious 5.4e-4 offset this way. Diagnostic: **compute the effect under mean, median and biweight.** Monotonic in estimator robustness and ~zero for the mean ⇒ it is the estimator, not the operation. Production is unaffected — fields are extracted separately — but every offline diagnostic must be run per field.
2. **Difference errors need paired draws.** `bootstrap_all` seeds a fresh RNG and draws `rng.integers(0, ngal, ngal)` consuming nothing else, so equal seed/nboot/ngal already gives identical galaxy sets across choices. Differencing the per-draw arrays costs nothing and gives σ_Δ ~2.2× tighter than σ_ref here. `abs` answers *does it matter*; `paired` answers *is it real*. Both are needed.
3. **Correlated pixels break `/sqrt(npix)`.** See §2.6.2. Any error on a quantity derived from a resampled spectrum should come from a galaxy bootstrap.
4. **A "spread" is a maximum, so judge it against the multiple-comparisons expectation.** And **radial bins are not independent trials** — the same background spectrum is subtracted from every bin of every galaxy, so one noisy background draw moves them all together. "All 7 bins shifted the same direction" is what noise looks like here, not evidence of a systematic.
5. **Ask "is there an effect?" before "what explains it?"** The first version of `continuum_window_scan` compared near-window to far-window spread and announced a real annulus dependence without ever checking that the spread was significant. Significance gate added.

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

### 3.6 [v4, HIGH] The star re-extraction — §3.2 + §3.3 + §3.5 as one job

**Austin's framing, adopted: these are not three separate tasks, they are one star extraction done properly.** §3.2 (outer wing slope), §3.3 (astrometric convolution), §3.5 (are stars background-subtracted the same way), and the saturation check all read the same stacked stellar profile, so they should be produced together from a single re-extraction whose defining requirement is that it **matches the Lyα extraction path exactly**.

**Existing machinery to build on, not replace:**

- `starpsf.make_star_run_table(config, ra, dec, mode="arcsec", source_table=...)` already puts stars through the *same* `extract_fibers` code path with an arcsec bin mode. That is the mechanism that makes "matches the Lyα method exactly" achievable rather than aspirational.
- **[CORRECTED 2026-07-27 — see `lya-halos-uv-vs-fiber-psf-tools` in memory]** `measure_stellar_psf` lives in `uv_profile.py` and is the **UV/CFHT-LS imaging** PSF tool — a completely different telescope, doing pixel-cutout photometry, structurally unrelated to the Lyα fiber pipeline. Its saturated-core rejection and the quoted log line (*"kept 57 stars (rejected 8 saturated-core, 0 off-image/low-coverage, 0 bad centroid, 5 shape-outlier), biweight stack, bootstrap err (n_boot=2000), sat_core=1.13″, core_norm=0.38″, sky=[4.40,6.60]″, stamp=17.2″"*) describe that imaging measurement only and say nothing about the fiber-based star extraction below.
- **The fiber-based star extraction has actually been run.** `0_Extractor_Stars.ipynb` calls `starpsf.make_star_run_table` + `run_positions_extract` (`mask='none'`) on the AEGIS field, with bins `[0,1,2,3,4,5,7,10,15,20,30,40,50,70,100]″` — reaching 100″, not stopping at ~17″. The output filename records `bg57-63`: a 57–63″ background annulus. That matches the pipeline as described in the 6-16-26 presentation, but the background annulus later moved to the current **80–90″** (7-28-26 presentation, "Pipeline Update" — pooled over all exposures, more fibers, replaced smoothing with mask+interpolate+Gaussian). So this run is not missing, and it is not using a mismatched tool — it is a real fiber-based star (and a parallel blank-sky, via `make_random_sky_table`) extraction that **predates the current background config** and needs re-running under it.

**Requirements for the re-extraction:**

1. **Radial grid extended well past the current stamp.** The headline problem (§3.2) is that the stellar profile is reported to die at 10–20″ while science runs to 2700 kpc ≈ 320″. **[CORRECTED 2026-07-27] Unverified which measurement this claim actually describes.** The UV-imaging `measure_stellar_psf`'s stamp (17.2″, a different telescope, imaging-based) is not the same thing as the fiber-based star extraction's radial grid — the latter (`0_Extractor_Stars.ipynb`) already reaches 100″. Whether the *fiber-side* profile also dies at 10–20″ has not been checked against that run's actual `curve_of_growth` output — do that before treating "dies at 10–20″" as established for the Lyα-relevant measurement. Re-running the fiber-based star extraction under the current 80–90″ background config (grid already reaches past it) is what makes §3.5 answerable — and, if there is measurable star flux out there, it is simultaneously the aureole detection §3.2 wants. **These are the same measurement.**
2. **§3.5 resolved by construction.** Extract the stars with the *Lyα* background configuration (80–90″ annulus) *and* with no background subtraction, and compare. The difference is exactly the aureole flux being subtracted away. This is the same `SPEC_NOBG` trick as §2.5 and is exact for the same equivariance reason.
3. **Free power-law fit to the outer profile**, slope reported with bootstrap error (§3.2). Converts "unconstrained over 95% of the radial range" into "constrained steeper than X."
4. **σ_pos convolution applied to the result** (§3.3) before the profile is used as the galaxy PSF. Get σ_pos from MOSDEF/CANDELS positional uncertainties or astrometric residuals; Gaia star positions are effectively exact, so the correction is one-sided and it currently biases *toward* the claimed result.
5. **Per field.** ~57 (COSMOS) and ~30 (EGS) surviving stars, each field with its own PSF and depth (§6.6 — EGS deeper, COSMOS larger area, so COSMOS supplies more stars but EGS's are better measured; weight accordingly rather than pooling naively).

**The hard part Austin identified: stars are continuum, galaxies are a line.** The Lyα path measures `integrated_line_flux` in a ±3 Å rest window on a continuum-subtracted spectrum. Stars have no line; their signal *is* continuum, so the continuum-subtraction step that defines the galaxy measurement would zero them out. The star profile therefore cannot be produced by literally running the galaxy Stage-2 code. Options, in preference order:

- **Broadband collapse over the same wavelengths.** Integrate the star spectrum over the observed wavelength range that the galaxy sample's Lyα falls into, so the PSF is measured at the right λ (seeing ∝ λ^(−1/5), §3.4). Everything *spatial* — fiber masking, radial binning, biweight coadd, bin averaging — stays byte-for-byte identical to the Lyα path, which is where the systematics actually live. The wavelength treatment differs because it must.
- **Per-galaxy λ matching.** Rather than one broadband collapse, weight each star's spectrum by the sample's redshift distribution, producing a PSF matched to the *sample-averaged* observed wavelength. This is the same construction §3.4 asks about for the kpc conversion, and it should be done consistently with whatever that turns out to be.
- **Check Niemeyer 2022 pt. 1 for precedent.** Austin's instinct is right that this is the obvious reference — it is already the source of the fiducial Moffat (β=3, 1.2–1.4″, `analysis.py:133`, `measure.py:2270`) and the external-replication target in `GOALS.md`. If she constructed a continuum PSF for a line measurement, matching her construction is worth more than optimizing our own, because it makes the profile comparison apples-to-apples.

**On the N-star limitation.** ~57 + ~30 is small, and it caps the wing measurement more than anything else does. Two things that buy real S/N without new data: (a) the wing constraint integrates over a huge area — at 80″ the annulus area is ~1000× the core's, so even a faint aureole is being averaged over enormous fiber counts, and the per-star S/N matters much less than the star count suggests; (b) stack all stars from both fields *for the wing slope only* (a shape measurement) while keeping per-field stacks for the core, since the wing is set by optics common to both fields and the core is set by per-field seeing. Worth stating the assumption explicitly rather than assuming it.

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
| 6.6 | Field heterogeneity — **EGS is ~2× deeper than COSMOS**; COSMOS covers a much larger area, EGS roughly half the area, so the two FITS files are similar in size (corrected 2026-07-26, see §0.1) | Different effective PSF, noise, segmap depth; **outer bins are better sampled in COSMOS, inner S/N better in EGS** | Free (report per-field band) | MED |
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

## 8. Prioritized plan — v4, 2026-07-26

**Closed since v3:** §1.2 (fixed ±3 Å window kept), §1.6 (continuum method irrelevant), §2.4 / A1 (CPU-bound, `combine_threads=8`, 20.3 → 8.5 hr), §2.3 (fine grid running), §1.5 window symmetry (confirmed).

**Closed 2026-07-27 by the §2.5 extraction landing:** **§1.1b** (fill method irrelevant, keep `linear`), **§2.6** (background subtraction does not threaten the result; annulus choice consistent with noise at p≈0.10 flux / 0.29 centroid). Plan items 3 and 4 below are done. See §2.6 for the numbers and for two retracted claims.

**In flight right now**

0. Fine-grid full extraction (22 bins, 0 → 2700 kpc). Lands overnight 2026-07-26. Everything Stage-2 waits on it.

**Next — the custom extraction, launch as soon as the machine is free**

1. **§2.5 background-variant extraction, ~100 galaxies, ~2.5–2.8 hr.** `SPEC_NOBG` + **raw unmasked** `BG_VARIANTS` at 80–90 / 120–150 / 150–200″ + `BG_NFIB` + `BG_HALVES`. Coarse bins, own signature dir (automatic — the flags are hashed when set). **CODE SHIPPED 2026-07-26** (see §2.5's status box); the only remaining pre-launch item is checking the run-table ordering per §2.5.4.
2. §1.1 rest-frame stack of the raw backgrounds at 1215.67 Å, with the observed-frame control stack. Unblocked by (1).
3. ~~§2.2 joint mask-width × smoothing-σ sweep~~ **PARTIALLY DONE 2026-07-27.** mask ±5/±20 Å and σ 8/60 Å run through `compare_background_impact`; all inside the noise. The full 2D `mask_width_scan` against §2.5.1's pre-registered criterion is still worth running, but it is no longer on the critical path — §2.6 shows the knob does not move the science.
4. ~~§1.1b interpolation-anchor comparison~~ **DONE 2026-07-27 — see §1.1b's resolution box and §2.6.** Keep `linear`. Normalized convolution not tested and not needed.
5. **B10/B11 — now the highest-value background item left, and it has a second job.** Beyond checking the background rise-and-fall against the §1.1 leak, `background_variance_profile` + `background_half_difference` are **the decision criterion for whether a global background is admissible** (§2.6.5). Run before any global-background work.
5b. **Fold in EGS and re-run §2.6.** The annulus scatter is background-*noise*-limited (§2.6.4), so it shrinks as 1/√N. This sets the final error-budget bound (§2.6.6) and re-tests the continuum-window coupling (§2.6.3) at better precision. Highest single gain available.
5c. **Optional: test 80–110″ as a wider fiducial annulus** (§2.6.5). ~4% error reduction, no spatial risk, stays clear of the 1000–2000 kpc bin. Do *not* go to 80–130″ — it pushes the annulus into that bin. Low priority; the gain is small.

**Also next — the star re-extraction, independent of the above**

6. **§3.6** — one job covering §3.2 (outer wing slope, free power law with error), §3.3 (σ_pos convolution), §3.5 (Lyα-configured background vs. none, exposing any aureole self-subtraction), and the saturation linearity check. Grid extended to 80–90″. Resolve the continuum-vs-line construction, checking Niemeyer 2022 pt. 1 for precedent first. Blocked on C12 (how star extraction is currently configured).
7. §3.4 confirm what PSF the Lyα path actually uses (empirical vs. Moffat; z-distribution vs. median z; `wavelength_corrected_seeing`). Should be answered *before* §3.6 so the re-extraction targets the right thing.

**Free, once the fine grid lands**

8. §4.4 item 1 — the projection interpretation caveat (one sentence, no code). Then item 2, the pedestal check.
9. §1.3 confirm propagated-vs-bootstrap ratio; purge `err_sum` and `pcov` errors from all output.
10. §1.5 **injection–recovery transfer function** for the centroid, per bin, on the new grid — recovered vs. injected Δv, slope near zero quoted as attenuation.
11. §1.7 point-source ΔAIC, framed per §1.7's v4 note as quantification of the log-log comparison, not as independent evidence.
12. §1.4 — first settle what the literature convention is (Niemeyer especially), then check whether the shift is even significant on the fine grid.
13. Sanity-check inner-bin fiber counts on the new grid (§2.3 v4 note); decide which inner bins get reported-but-not-interpreted.
14. Fill in TBD verdicts in `research-notes.md` from one `summarize_validation` run.

**Then — one full extraction, designed to be the last for a while**

15. §2.1 full-sample multi-variant extraction at the mask width adopted in (3), fine grid, all background products retained permanently.
16. §6.1 masking-percentile variants, if they can ride along.
17. §7 random-position null at 3–5N.
18. §5.2 re-run the binning simulation with a realistically skewed fiber distribution (also §6.11) — now lower priority, since the fine grid's outer fiber counts (1k–50k) are comfortably inside the validated regime.

---

## 9. Answers recorded 2026-07-25

- **B9** — the stored `Background` HDU is **post-mask and post-smoothing**. §1.1's direct test therefore waits on §2.1.
- **Flux and centroid use the identical ±3 Å window, and stay that way.** §1.2 was tested (not just a documentation question) and resolved 2026-07-25: fixed ±3 Å kept for both — the wider/Gaussian-derived flux windows tried were less stable (inner-bin low-fiber-count Gaussian fits, Lyα-forest absorption entering wider windows), not more complete.
- **Outer-bin bootstrap S/N: ≈6, 5, 2.8** in the outer three bins. So the outermost bin is ~3σ — real but not comfortable, and the first thing a systematic would eat.
- **Crossover radius** = where the exponential and the two-halo term become equal. Survives any §4 change.
- **`A2` is a real post-crossover flattening**, per the actual fits — not a subtraction residual. §4.4 scoped down accordingly.
- **Stars:** ~50 per field or fewer; a stacked stellar radial profile already exists; profile dies at 10–20″.
- **AGN cut:** X-ray + BPT + IRAC combined, removes ~70 of ~510.
- **Redshifts:** MOSDEF nebular, Hα or [O III]. Not circular.
- **Continuum sidebands:** line-free windows, unchanged; exact values in `config`.

## 10. Still open

- ~~**A1** — what dominates the 13–19 hr.~~ **CLOSED 2026-07-25: CPU-bound on the combine. `combine_threads=8` → 8.5 hr, bit-identical. See §2.4.**
- ~~Is Stage-1 `Err` consumed by anything downstream?~~ **CLOSED: dropping it was tested and rejected as a speedup (§2.4).**
- **B10/B11** — the monotonic rise-and-fall of background level with radius. Check against the §1.1 leak once **§2.5** makes it measurable (was: §2.1). **[2026-07-27: now also gates the global-background question, §2.6.5. The spatial term is the decision criterion. Highest-value background item remaining.]**
- **Is there real UV continuum in the outer science bins from clustered neighbours?** Raised by §2.6.2: `pedestal_is_clean` assumes no continuum beyond the annulus radius, which is defensible for the *annulus* (near-random sky) but questionable for a *science bin* centred on a galaxy with correlated neighbours. Relevant to the ~2R_vir flattening interpretation. Would make the outer-bin pedestal test ambiguous rather than clean.
- **Does the continuum-window / smoothing-kernel coupling matter at EGS-included precision?** §2.6.3: production windows sit ~1.1 kernel σ from Lyα, so the coupling is real; COSMOS alone cannot resolve its effect. If it turns out to matter, the fix is one config line (`cont_bounds`), and the rule is to pick the *closest* windows that clear the kernel — further out costs S/N per continuum pixel and admits more real features.
- **C12 [CORRECTED 2026-07-27]** — how star extraction is configured (mask, background, bin grid, centering). Now blocks **§3.6**, which merges §3.2/§3.3/§3.5. **The `measure_stellar_psf` log line does not answer this** — that function (`uv_profile.py`) is the UV/CFHT-LS imaging PSF tool, a different telescope entirely, unrelated to the Lyα fiber pipeline; the previous framing here conflated the two. The real answer: the fiber-based star extraction (`starpsf.py`, run in `0_Extractor_Stars.ipynb`) exists and used a 57–63″ background annulus, matching the pipeline as of the 6-16-26 presentation but predating the current 80–90″ config (adopted per the 7-28-26 presentation). So the fiber-side extraction is **stale**, not mismatched with a different tool, and needs re-running under the current config — exactly what §3.6 already proposes. §3.5 remains a live question; just for the corrected reason.
- **How does Niemeyer 2022 pt. 1 build a PSF profile for a line measurement from continuum stars?** Blocks the construction choice in §3.6, and matching her is worth more than optimizing ours given the planned external-replication comparison.
- What radius convention does the literature use for wide annular bins (§1.4)?
- Whether HETDEX has published a measured VIRUS PSF wing profile — would close §3.2 outright.
- Real fiber-flux skewness within an annulus, for §5.1/§6.11.
- Where `bw2.py` / `bin_test.py` should live in the repo as reproducible tests.
- **Run-table ordering** — is it sorted by field/mass/z? Determines whether §2.5's contiguous 100-galaxy window is representative. Check before launching.
- **Where did "COSMOS ~4× depth" come from?** It was asserted in §6.6 from v1 and propagated unchallenged through v3 until Austin corrected it 2026-07-26 (§0.1: EGS is deeper, ~2×; COSMOS is larger area). Worth a quick pass over `research-notes.md` and `GOALS.md` for any other per-field claim that inherited it — the inversion changes which field dominates which radial range.

## See also

- `specs/background-diagnostics.md` — **the Stage-2 toolkit that consumes §2.5's output.** Specifies the stacking/control/filler-testing machinery for §1.1, §1.1b, §2.2 and B10/B11. Being built in parallel with the §2.5 extraction.
- `code/example_cells_background_impact.md` — runnable cells for §2.6's toolkit (`compare_background_impact`, `continuum_window_scan`), plus the list of changes made to existing code (`reconstruct_background` gained `fill_method`; `measure._measurement_window` now accepts any number of continuum sidebands, so a one-sided continuum no longer crashes Stage 3).
- `docs/psf-aware-fitting-explained.md` — the forward model §3 stress-tests
- `docs/research-notes.md` — reasoning trail; §"Systematics tests" holds the TBD verdict list
- `specs/bin-correlation-diagnostics.md` — covariance machinery §1.3 wants in the fit
- `specs/halo-flux-fitting.md` — the fit §4 proposes changing
- `validation.py`, `starpsf.py`, `psf_forensics.py`, `optimize.py`, `extract.py`
