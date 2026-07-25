# Session handoff — 2026-07-25 (systematics audit + extraction profiling)

Written to carry context into a fresh session. Read this, then the two specs it points at.

## What this session produced

| File | What it is |
|---|---|
| `specs/systematics-and-extraction.md` | **The systematics register.** Every systematic, tagged by severity, cost, and whether it needs a re-extraction. The main deliverable. |
| `specs/extraction-speedup.md` | **The extraction optimization spec**, including the measured profile and the verdict. |
| `code/extract_profile.py` | Standalone, read-only profiler for `extract_fibers`. Modifies nothing. |
| `docs/systematics-audit.md` | Stub — superseded, points at the spec. Ignore. |

Nothing in `utils_lya_halo/` was changed. No analysis was re-run. This session was diagnosis and planning only.

---

## The one structural fact that governs everything

**`GalaxyProduct` stores `spec (ngal, nrad, nwave)` — the fiber dimension is destroyed at extraction time.**

So: masking, background annulus, background smoothing, radial bin edges, and the fiber coadd are all *before* the 13–19 hr boundary and can never be changed post-hoc. Rest-frame regridding, galaxy coadd, continuum, line window, centroid, fitting, and bootstrapping are all *after* and are free.

This killed several cheap-looking tests (notably the offset-position null) and is why making the extraction fast is a precondition for the systematics program rather than a side quest.

---

## Extraction: measured, diagnosed, fix identified

First profile run (COSMOS): `all_flux_final.fits` = 17.8 GB, shape `(132, 34944, 1036)`, dtype **`>f4` (big-endian)**. Node RAM 94.3 GB, 35.6 available. Per galaxy: **338–545 s, 16–26 GB read, ~400–560k major page faults.**

**Verdict: I/O-bound.** Roughly the entire file is read once per galaxy → ~35 hr and ~5.8 TB for COSMOS's 278.

Three findings:

- **F1** — the file nominally fits in RAM but isn't cached, because the process's own pools evict it. Page cache and process memory share one budget.
- **F2** — `pool = np.empty((n_r, 1036), float32)` is allocated full-wavelength but `combine_fibers_chunked(wave_block=16)` only ever uses 16 wavelengths at a time. A 2M-fiber bin allocates **7.7 GB** where **0.12 GB** would do — **65× oversized**. This is both the memory problem and the cause of the I/O problem.
- **F3** — big-endian array on a little-endian machine: every `spectra[e, sel]` byte-swaps to native, allocating a temporary, on the hottest path.

**The fix (much simpler than the wavelength-major transposition originally proposed, which is now superseded):**

1. Gather per wavelength block → pools drop ~8 GB → ~0.2 GB.
2. That frees the page cache to hold all 17.8 GB → reads become RAM-speed after galaxy 1.
3. Write `all_flux_final_native.npy` as `<f4` once and memmap that → byte swap never happens again.

Then the residual cost is the biweight, attacked by: **dropping the unused `mad_std`** (Stage-1 `Err` is validation-only, no config knob exists to disable it — likely a straight ~2×), and parallelizing over wavelength blocks.

**Caveat:** the numbers above were contaminated by a profiler bug (a retained `pool.copy()` of up to 20 GB, which starved the cache and OOM-killed the kernel). Fixed — retained pool is now a subsampled ≤0.8 GB slice, `max_pool_gb` defaults to 4.0. **The A/B re-run has not been done yet.** That is the immediate next action:

```python
importlib.reload(ep)
a = ep.profile_galaxies(cfg_profile, s["vdfi_data"], s["run_table"], s["bad_fiber_mask"],
                        n_gal=3, max_pool_gb=4.0, block_gather=False)   # current
b = ep.profile_galaxies(cfg_profile, s["vdfi_data"], s["run_table"], s["bad_fiber_mask"],
                        n_gal=3, max_pool_gb=4.0, block_gather=True)    # proposed
ep.print_profile(a); ep.print_profile(b)
```

With `block_gather=True` the outermost bins stop being skipped, so this also produces the first real numbers for them.

---

## Systematics: the findings that matter

Organizing principle adopted: **the headline result is a differential (centroid and profile shape vs. radius), so common-mode systematics cancel and radius-dependent ones don't.** Everything in the register is tagged that way.

### Confirmed / high value

- **Fixed ±3 Å window biases `h1` HIGH.** Core lines are wider in FWHM, so a fixed window under-measures *inner* flux relative to outer → measured profile shallower than truth → halo looks more extended than it is. Non-conservative, sitting on the headline number. Fix is free and doesn't require giving up the elegant single-window design: keep ±3 Å for the centroid (which genuinely needs consistency), test the *flux* profile at ±6/8/10 Å or via per-bin curve of growth. If `h1` is stable you keep everything and gain a sentence; if it moves, you needed to know.
- **The background interpolation is anchored on exactly two pixels.** `np.interp` is piecewise-linear between anchors, so every pixel in the masked Lyα gap is set by the last good pixel before and the first after — a ~10-pixel gap from 2 samples. Noise is ~4× diluted by the subsequent 30 Å smoothing; **bias is not diluted at all**. Fixes, both fast and vectorized: window-based anchors (median/line fit over ~40 Å each side), or normalized convolution `gaussian(flux·m)/gaussian(m)`. The latter was rejected once for "bowing," but that needs a gap *wider* than σ and the current gap (20 Å) is narrower than σ (30 Å) — worth re-testing at current parameters. **Testable today** on the stored `Background` shapes with synthetic noise.
- **Astrometric asymmetry, free fix.** Stars are centered from Gaia (~mas), galaxies from MOSDEF/CANDELS (~0.1–0.3″ = 1.7–2.5 kpc at z=2.3, vs `h1`≈16). Makes galaxies look more extended than stars — biases toward the claimed result. Fix: convolve the measured stellar PSF with a Gaussian of width σ_pos before using it. One array operation.
- **PSF wing slope is unconstrained over ~95% of the radial range.** Stellar profile dies at 10–20″ (84–168 kpc); science runs to 3000 kpc. Moffat β=3 falls as r⁻⁶; real aureoles go as r⁻² and carry 1–10% of flux — **35× more light beyond R_vir than the assumed Moffat**. But 10–20″ is 10–15× FWHM, a decent lever arm: **fit a free power law to the outer stellar profile and report the measured slope with its error** rather than extrapolating. Also check whether HETDEX has published a measured VIRUS wing profile.
- **Noise is underestimated in one specific place.** `integrated_line_flux` quadrature-sums per-pixel errors assuming independence; `rest_delta=0.2` oversamples the native rest sampling (~0.61 Å at z=2.3) by ~3×, so errors can be low by up to √3. Contained because bootstrap is used for real errors — but `err_sum` and `describe_fit`'s `pcov`-derived `h1_err` must never reach a table.

### Projection — scoped down after pushback

The ±3 Å window is ±10.5 cMpc deep; the largest transverse separation is 9.9 cMpc. They're comparable, so Limber breaks and the projected slope steepens from −0.8 toward −1.8 past ~1 Mpc.

**What has to happen:** a one-sentence interpretation caveat — the fitted outer slope must not be described as a clustering slope. No code change.

**What was retracted:** an argument that the cored γ=0.8 term's near-flat inner behaviour makes `A2` degenerate with a background pedestal. That was inferred from the functional form, not from the fits. Austin reports `A2` is a real, reproducible post-crossover flattening. Kept only as a cheap falsifiable check (refit with a free additive constant; expected outcome: stable).

**Optional:** a finite-depth projected two-halo term as an A/B via `compare_models_aic_bic`. Selling point is that it removes `r_c` entirely (one free parameter, π_max set by the window). Not a mandate.

### Settled by simulation (`bw2.py`, `bin_test.py` — should be moved into the repo)

- **Chunking a biweight costs ~7–10% variance and no detectable bias** across Gaussian, 5%-outlier, lognormal, and student-t₃. **Merging with an n-weighted mean instead of a second biweight cuts the cost to ~2–3%** — the sub-biweights already rejected the outliers, so a second robust stage only loses efficiency.
- **A wide-bin biweight does recover the area-weighted mean** (0.7203 vs analytic 0.72078 on a declining `r^−0.8` profile) — so the forward model's area-averaging convention matches what the data statistic measures. Worth a Methods sentence.
- **Sub-binning outer annuli is statistically free** (0.1% agreement, identical variance). The original objection was right for inner bins (single-digit fiber counts) and doesn't apply to bins with 500k+ fibers. Recommendation: subdivide only bins above ~50k fibers.
- The existing 16-wave-pixel memory patch chunks the *wavelength* axis, so every wavelength still sees all fibers — **statistically exactly lossless.**
- Caveat: tested with Gaussian noise. Re-run with a realistically skewed fiber distribution before committing.

### Corrected mid-session (don't re-derive these)

- **The background does NOT self-subtract the Lyα signal.** Lyα is masked, interpolated across, and smoothed before subtraction. An early claim that the profile must cross zero near 700 kpc was wrong and is retracted.
- The residual concern is narrower: broad two-halo Lyα elevating the interpolation anchors, plus smoothing re-injecting the wings. Both are further suppressed by the sideband continuum fit, which removes anything smooth — **only a localized ~10–12 Å rest deviation centered on Lyα survives both.** That double protection is a genuine defense of the design and belongs in the Methods.
- **The direct test is blocked.** The stored `Background` HDU is written post-mask and post-smoothing, so the line is already gone from what's on disk. Answering "does the annulus carry Lyα" requires the raw annulus coadd → the multi-variant extraction below.

---

## The one extraction to design carefully

Once the extraction is fast, do **one** run that writes everything needed to make background systematics free forever:

- `SPEC_NOBG` — per-annulus coadd with no background subtracted
- `BG_VARIANTS` — candidate backgrounds (80–90″, 120–150″, 150–200″) stored **raw, unmasked, unsmoothed**
- `BG_NFIB`, and ideally `BG_HALVES` (annulus split by azimuth → direct spatial-variance measurement)
- outer bins subdivided (>50k fibers only)

Since subtraction is a per-wavelength scalar operation, `SPEC_NOBG − smooth(mask(BG_VARIANTS[k]))` reproduces any variant in Stage 2 at zero cost.

---

## Facts established this session (don't re-ask)

- Bins: kpc, `[0, 10, 16, 24, 36, 60, 100, 180, 350, 600, 1000, 2000, 3000]`. Solid to ~2000 kpc.
- Outer-bin bootstrap S/N ≈ **6, 5, 2.8**.
- N: 278 COSMOS + 244 EGS; AGN cut (X-ray + BPT + IRAC) removes ~70 of ~510.
- Redshifts: MOSDEF nebular (Hα / [O III]). Not circular.
- Flux and centroid currently use the **identical** ±3 Å window.
- Centroid estimator: `median50` (`flux_median`) with negatives clipped.
- Crossover radius = where the exponential and two-halo terms become equal.
- Stars: ~50 per field or fewer; stacked radial profile exists; dies at 10–20″.
- Background = biweight of **only** the 80–90″ annulus fibers, pooled across that galaxy's exposures (`min_bg_fibers=999999` forces pooling). Annulus is 75× smaller in area than the outermost science bin.
- Continuum sidebands: line-free, `((1195, 1205), (1225, 1236))`.
- Stage-1 `Err` is validation-only; no config knob exists to disable it.
- Notebooks in this workspace are stale snapshots — Austin runs the real ones elsewhere.

## Still open

- ~~Does anything downstream consume Stage-1 `Err`?~~ **ANSWERED: yes** —
  `stack.coadd_galaxies` feeds it to `inv_var`/`mean`, `stack_one_galaxy` uses
  it for line S/N. The "drop Err" speedup is dead. See
  `session-handoff-2026-07-25b-extraction-parallel.md`.
- ~~The A/B profile re-run with the fixed profiler.~~ **DONE, and the extraction
  is now 2.38× faster with bit-identical output.** `block_gather` turned out to
  be a 35% regression; threading the combine was the actual fix. Full account in
  `session-handoff-2026-07-25b-extraction-parallel.md` and
  `specs/extraction-speedup.md` §1d–§1g, §9.
- How star extraction is configured (mask, background, bin grid, centering) — needed before any PSF wing claim.
- Real fiber-flux skewness within an annulus.
- Whether HETDEX published a measured VIRUS PSF wing profile.

## Working notes on method

Two habits that came out of this session and are worth keeping:

- **Output inspection is a legitimate validation strategy but is blind to one bug class** — anything wrong by a smooth, slowly-varying factor. The 2πr area bug is the local example: plausible outputs for months, caught only when the PSF was widened. The generalization isn't "read more code," it's: **for each number the paper reports, is there a second, structurally different way to compute it?** Where the answer is no, that's the exposure. `psf_forensics.py` is the model.
- **Synthetic data is the right tool when the question is about an estimator rather than about the sky.** "How does biweight-of-biweights compare to biweight-of-all" has no astrophysics in it and was settled cleanly in simulation. Questions about the sky still want internal consistency tests on the real data.
