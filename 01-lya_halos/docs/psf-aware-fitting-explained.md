# How the PSF-aware two-component fit actually works

This walks through `fitting.py`'s `fit_psf_aware_expcore` (and its sibling `fit_psf_aware` for the older two-exponential model) from the physics down to the exact lines of code, so you can defend every step of it without treating it as a black box. It's written to be read in order — each section leans on the one before it.

## 1. The problem it's solving

Every fiber points at the sky through a PSF. A galaxy's true Lyα surface-brightness profile, whatever it actually is, gets smeared radially before it ever reaches your stacked, biweight-averaged flux-vs-radius bins. So the profile you *measure* is not the profile that's *there* — light from the bright inner region leaks outward into your middle and outer annuli, and (more subtly) light from far out gets pulled slightly inward too. The innermost bin is hit hardest, since that's where the surface brightness gradient is steepest.

There are two honest ways to handle this. One is to just throw away the contaminated inner bin(s) and fit the rest — that's `fit_naive`/`fit_naive_expcore`, and it's still in the pipeline as a deliberate sanity check. The other is to accept that you can't cleanly deconvolve noisy real data (deconvolution amplifies noise catastrophically), and instead go the other direction: propose a true intrinsic profile, predict *exactly* what that profile would look like after passing through the known PSF and being binned the same way the real data was binned, and fit that prediction against the data. That's the PSF-aware / forward-modeling approach, and it's the one you actually want to defend, because it's the one that keeps the inner bin — your highest-S/N, most physically informative bin — in the fit instead of discarding it.

## 2. The model: what's actually being fit

The intrinsic (pre-PSF) profile is two additive terms:

```
I(r) = A1 * exp(-r/h1)                      (core)
     + A2 * (1 + (r/r_c)^2)^(-gamma/2)       (halo / 2-halo term)
```

This is `intrinsic_profile_expcore` in `fitting.py`. The first term is a plain exponential — the compact, resonantly-scattered Lyα halo immediately around the galaxy, with `h1` its scale length. That's your "core," and per your own convention, `h1` *is* the halo scale length in the literature's sense — don't let the code's internal naming confuse you when you're writing this up.

The second term is a cored (Plummer/beta-model-style) power law rather than a second exponential: it's flat at small `r` (approaches `A2`) and falls off as a power law `(r/r_c)^(-gamma)` once `r >> r_c`. This is standing in for the large-scale "2-halo" clustering signal — Lyα photons scattered by gas well outside the galaxy's own halo, tracing the surrounding large-scale structure rather than the galaxy itself. A power law is the physically motivated shape for that (it's what you'd get from a Limber-projected clustering correlation function), and the "cored" softening just keeps it finite as `r → 0` instead of diverging, which a bare power law would do. The older model in the codebase, `intrinsic_profile` (two exponentials, `model="twoexp"`), is kept around specifically as the null/comparison case — if the extra shape flexibility of the cored power law isn't earning its keep on a given fit, `compare_models_aic_bic` will tell you so via AIC/BIC, which is the legitimate way to defend using the more complex model rather than just asserting it looks better.

Three of this model's five parameters are fixed by default rather than floated: `gamma_fixed=0.8` and `r_c_fixed=400.0`. This isn't arbitrary — see §6.

## 3. Characterizing the PSF

The PSF is a 1D radial curve, either an analytic Moffat (`moffat_1d`, default 1.3″ FWHM, β=3 — the Lujan & Niemeyer 2022 literature fiducial for VIRUS) or an empirically measured curve from `starpsf.py`'s stellar profile fits, converted to kpc via the sample's median redshift. Either way, it gets flux-normalized (`normalize_psf_flux`) so that `∫ PSF(r) · 2πr dr = 1` — i.e., it's normalized as a genuine 2D point-spread function (unit total flux spread over a disk), not just normalized so its peak or its 1D integral is 1. That `2πr` is the first appearance of the circumference weighting that matters everywhere in this pipeline — a PSF value at radius `r` represents a surface density, and turning surface densities into actual flux always costs you a factor of the annulus's circumference.

## 4. The fine radial grid

Fitting can't work directly on the coarse observed bins — the PSF needs to be convolved against a profile sampled finely enough to resolve both a narrow inner bin and a very wide outer bin without over- or under-resolving either. `default_fine_grid` builds this by giving each observed bin its own linspace (so bin width, not a single global spacing, sets local resolution) plus a tail extending 20% past the outermost edge, because flux sitting just beyond your last measured bin still gets smeared back *into* the observed range by the PSF and would otherwise vanish from the accounting entirely.

The quadrature weights on this grid (`_segment_weights`) are a genuine numerical detail worth knowing about because it was a real, fixed bug: computing `dr` via `np.gradient` across the whole grid treats the resolution jump at each bin edge as a smooth transition (a central-difference average of the coarse and fine spacing on either side), when the truth is a hard jump — the grid was deliberately built with a spacing discontinuity there. That silently misallocated weight between adjacent bins, worst right at the boundary where the annulus's `2πr` weighting is largest. Building the trapezoid rule bin-by-bin instead dropped the innermost bin's error against exact quadrature from 2.77% to 0.06% at identical grid density.

## 5. The ring-convolution matrix — the actual "2D annulus binning" step

This is `ring_convolution_matrix`, and it's the geometric heart of the whole method. Conceptually: take an infinitesimally thin ring of unit total flux sitting at true radius `r_fine[j]`. Smear it with the PSF. Ask what fraction of that ring's flux lands inside each observed annulus `[r_edges[i], r_edges[i+1]]`. That fraction is `R[i,j]`.

Two integrals are nested here. The inner one (`_ring_convolved_value`) sweeps the azimuthal angle φ around the ring, computing the PSF-smeared surface density at an observed radius ρ due to a unit-flux ring at r0 — this is a standard "PSF applied to a thin annular source" convolution, done in closed form as an angular integral rather than a brute 2D image convolution, because the source has circular symmetry. The outer integral (inside `ring_convolution_matrix`'s loop) then sweeps the observed radius ρ across each annulus and integrates `C(ρ; r0) · 2πρ dρ` over it — that outer `2πρ` is exactly the same circumference-to-flux conversion as in §3, just applied on the *receiving* end this time, because you're asking how much actual flux (not surface density) lands in that annulus.

Both integrals are restricted to the exact angular/radial window where the PSF can possibly be nonzero, worked out in closed form via `arccos`, rather than integrated over the full range and hoping the sampling is fine enough. This matters a lot at large radius: for a compact PSF and an outer bin sitting hundreds of kpc out, the nonzero region shrinks to a narrow sliver near φ=0 of width ~`psf_rmax/r0`. This was a real, previously shipped bug — a fixed-width grid over the full `[0, 2π]` badly undersampled that sliver, and a flux-conservation check (does a unit-flux ring's PSF-smeared light still integrate to 1 across all radii?) came back at ~13 instead of 1 at `r0 = 1000` kpc, growing roughly linearly with radius. Restricting to the exact support window fixed it. `validate_ring_convolution` is the standing regression test for this: it cross-checks the fast trapezoid method against `scipy.integrate.quad` and checks flux conservation at `r0` from 1 to 2000 kpc, both currently passing to <1e-3 relative error.

`R` is precomputed exactly once per (grid, bin edges, PSF) — critically, it does **not** depend on the trial amplitudes or scale lengths being fit, because convolution is linear. Every candidate `(A1, h1, A2, r_c, gamma)` during the fit just needs one matrix-vector multiply against `R`, not a fresh convolution. That's both a performance necessity (nonlinear least-squares tries many trial parameter sets) and worth stating plainly when defending the method: the *geometry* of PSF smearing is computed once and is independent of what shape of galaxy profile you're proposing; only the flux being redistributed through that fixed geometry changes from one trial to the next.

## 6. The part that actually broke once: sum vs. average, and the `2πr` bookkeeping

This is worth being precise about because it's the exact bug that had to be found and fixed (2026-07-22), and understanding why it was wrong is probably the best way to make the whole averaging step defensible rather than mysterious.

Your real per-bin data value is a **biweight average surface brightness** in that annulus — a representative flux density, not a summed total. So the model's prediction for a bin has to be an area-weighted mean, not a raw sum. The correct chain is:

1. Convert the intrinsic profile to actual 2D ring flux at each fine-grid point: `I(r) · 2πr · dr` (`_fine_ring_flux_2d`) — you cannot redistribute "flux" through `R` unless what you're handing it is actually flux, and a surface-brightness profile only becomes flux once you multiply by its ring's circumference and width.
2. Redistribute that flux through the PSF geometry: `R @ (I(r)·2πr·dr)` — this gives the *total* PSF-smeared flux landing in each observed annulus.
3. Convert back to a mean surface brightness by dividing by that annulus's actual geometric area, `π(r_out² − r_in²)` (`_annulus_areas`) — **not** by the bin's radial width. Dividing by width instead of area leaves a stray, radius-growing `~2πr` factor uncorrected.

That's `bin_average_psf` / `bin_average_psf_expcore`:
```
model_i = ( R @ [ I(r) · 2πr · dr ] )_i / [ π(r_out,i² − r_in,i²) ]
```

The earlier, buggy version skipped the `2πr` on the input side *and* divided by radial width instead of area on the output side — two separate errors that happened to partially cancel for a narrow PSF (where `R` barely mixes radii, so within one bin the profile is nearly constant and the error is small) but compounded badly for a realistically wide PSF, where `R` genuinely redistributes flux across radii and the errors no longer cancel. It was caught and quantified directly, not just suspected: checked against a brute-force 2D image convolution + azimuthal average (`psf_forensics.py`), the corrected form agrees to <1% at both 1.3″ and 2.4″ PSF FWHM, while the old form was off by 51% and 96% respectively — growing with PSF width exactly as the diagnosis predicts. Under the old, buggy accounting, a single-exponential fit at 2.4″ collapsed to `h → 0` (a numerically-degenerate non-answer); under the corrected accounting `h` is stable (15.1 kpc at 1.3″, 12.0 kpc at 2.4″, χ²/dof ~1.1–1.4) across a FWHM sweep from 1.0″ to 3.0″. That stability-under-a-PSF-sweep is a genuinely useful thing to be able to say when defending the fit: the recovered scale length isn't an artifact of exactly which PSF width you assumed.

One consequence worth flagging explicitly: because the amplitude convention changed (central *surface brightness* now, not the old ill-defined per-radius quantity), absolute `A1`/`A2` values from any fit run before 2026-07-22 are not comparable to current ones. Scale lengths, the crossover radius, and amplitude *ratios* are unaffected and remain comparable across the fix.

## 7. Fitting mechanics

With `R` and the fine grid built, `fit_psf_aware_expcore` is nonlinear least-squares (`scipy.optimize.curve_fit`) against `bin_average_psf_expcore`, using **every** observed bin including the innermost one — the PSF model is exactly what makes that legitimate, since the inner bin's contamination by smeared-in flux from further out is now part of the model's own prediction rather than an unaccounted-for systematic.

A few conditioning/robustness choices, all standard practice but worth knowing why they're there:

- **Amplitude scaling.** Real fluxes are ~1e39 while scale lengths are ~1–1000. Handing `curve_fit` parameters spanning ~40 orders of magnitude in scale leaves the Jacobian catastrophically ill-conditioned — checked directly, the unscaled fit just returned the seed unchanged (χ²/dof ~5e5). Fitting `y/scale` with amplitudes carried in units of `scale` (the largest bin value) fixes this; `h1`/`r_c` are untouched since they're already order-unity.
- **Multi-seed fitting.** Two-component profiles on sparse, noisy radial bins are prone to local minima. Several candidate starting points (different core/halo split points, plus a swapped variant) are each fit independently and the lowest-χ² convergent result is kept, rather than trusting a single seed.
- **Data-driven bounds**, not a blanket `(0, ∞)`: amplitude and scale-length bounds are set relative to the actual data being fit (e.g. no scale length allowed to run to many times the fitted radial span), which forecloses numerically-degenerate "fits" like an amplitude that blows up while its scale length collapses to contribute nothing.

## 8. Why `A2 = f·A1`, and why `r_c`/`gamma` are fixed by default

Both of these were forced by diagnosing real failures on real bootstrap refits, not chosen a priori.

**Amplitude ordering.** `A2` is parametrized as `f · A1` with `f` bounded in `[0, 0.999]`, rather than fit as an independent amplitude. This structurally forbids the halo term from exceeding the core term near `r ≈ 0` — a genuine, observed failure mode: a dedicated investigation (`diagnose_crossover_failures`) found that on a real low/high-density subsample bootstrap, 0 of 3000 refit failures were due to the crossover search bracket being too narrow, but 85/3000 (2.8%, low-density) and 610/3000 (20.3%, high-density — i.e. the noisier subsample) were this amplitude inversion. Since it's worse on the noisier data, in exactly the direction you'd expect for a fit wandering into a degenerate region, widening a search bracket can't fix it — the fix has to be structural, which is what the reparametrization is.

**`r_c_fixed=400.0`.** Even after fixing the amplitude ordering, `r_c` and `A2` remained badly degenerate on realistic subsample S/N — a same-ordering-only refit on the noisier subsample still returned `r_c` anywhere from ~2 to ~2400 kpc, i.e. essentially unconstrained. A sensitivity scan fixing `r_c` across [200, 700] kpc found that `h1` and the *ratio* of crossover radius between the low- and high-density subsamples were stable to within ~5–15% across that entire range, while the *absolute* crossover radius is genuinely not `r_c`-independent (it scales up ~20–26% across that same range, which makes physical sense — a larger `r_c` keeps the halo term near its plateau longer before falling off, pushing the core/halo crossing outward). AIC/BIC on real fits favored, or were statistically indifferent to, the fixed-`r_c` model over letting it float — and notably *improved* on the noisier subsample, where the extra free parameter wasn't earning its keep. 400 kpc sits inside the scan's stable range. The honest way to report a crossover radius from this fit is therefore "conditional on `r_c = 400` kpc," not as a free-standing measurement — and `r_c_fixed=None` is still available any time you want to run that comparison explicitly via `compare_models_aic_bic`.

**`gamma_fixed=0.8`.** This is the Limber-projected z~2–3 clustering slope (the projection of a raw 3D slope of 1.8). Letting `gamma` float alongside `r_c` makes the degeneracy above worse, not better — on real data, even your highest-S/N full-sample fit leaves `r_c` and `gamma` with uncertainties comparable to or larger than the values themselves, and `compare_models_aic_bic` prefers the plain two-exponential model over expcore in that free-`gamma` regime anyway. Fixing it at the literature value removes a parameter that the data can't actually constrain, without discarding the physically-motivated functional form itself.

## 9. Validation, in one place

Three independent checks back this up, at three different levels of the pipeline:
1. `validate_ring_convolution` — the fast trapezoid-based ring convolution agrees with brute-force `scipy.integrate.quad` to <1e-3 relative error, and a unit-flux ring's PSF-smeared light integrates back out to 1.0 (flux conservation) to <1% from `r0 = 1` to `r0 = 2000` kpc.
2. `psf_forensics.py`, Test 1 — the full forward model (ring convolution + area-averaging) agrees with an independent brute-force 2D image convolution + azimuthal average to <1% at both 1.3″ and 2.4″ PSF FWHM.
3. `compare_models_aic_bic` — the choice between `expcore` and `twoexp`, and between fixed and floating `r_c`/`gamma`, is decided by AIC/BIC on the actual fit rather than by preference.

## 10. The short version, for when you need to say it out loud

The method never tries to undo the PSF by deconvolving noisy data — it forward-models the PSF instead: propose a true profile, predict exactly what that profile would look like after PSF smearing and biweight-binning, and fit that prediction to the data. The PSF-smearing step is a precomputed linear operator (the ring-convolution matrix, independent of the fitted parameters) built from exact-support angular/radial integrals rather than fixed-grid ones, because the naive fixed-grid version silently failed at large radius. Converting a radial profile into flux, and back into a per-bin mean surface brightness, always carries a `2πr`/annulus-area bookkeeping step — getting that wrong is a real, previously-shipped bug, not a hypothetical one, and its fix is independently validated against a brute-force 2D image convolution. Because the PSF is modeled explicitly, the fit uses every bin including the innermost one, rather than discarding your best-S/N data the way a naive PSF-blind fit has to. And the model's parametrization (`A2=f·A1`, fixed `r_c` and `gamma`) reflects specific, diagnosed degeneracies found via bootstrap refits on real subsample data, not arbitrary simplifications — each one is a response to a documented failure mode, with the alternative (fully free fit) still available for direct A/B comparison whenever you want to check that the fixed version isn't hiding anything.
