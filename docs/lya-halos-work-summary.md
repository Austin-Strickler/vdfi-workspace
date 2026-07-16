# Lyα Halos (VIRAL) — Comprehensive Work Summary

*Compiled 2026-07-16, combining the "VIRAL-Halos Overview" deck (6-16-26) with everything worked on since.*

---

## Part 1 — Foundation (established through mid-June 2026)

### 1.1 Project & motivating questions

**VIRAL** = VIRUS Investigation of Resonant Lyman-Alpha Light (Halos). Goal: quantify the extent and kinematics of Lyα halos around a sample of Cosmic Noon galaxies to learn about galaxy formation, accretion, and environment.

Motivating questions the project was framed around:
- How can we observe galaxy evolution through extended Lyα emission?
- What do Lyα halos tell us about a galaxy and its immediate environment?
- How, where, and when do galaxies primarily accrete mass?
- How does halo extent/kinematics change with galaxy properties (mass, SFR, dust, etc.)?
- What can a *stack* of galaxies reveal that individual measurements can't?
- How and where do satellite galaxies contribute to an observed Lyα halo?

### 1.2 Dataset

**MOSDEF survey** (galaxy sample):
- H-band selected, H≲24.5 (z~2 bin) / H≲25 (z~3 bin)
- Systemic (rest-frame optical) redshifts from Hα or [OIII] — this is the project's core methodological advantage: an *independent* systemic anchor, not a shift measured relative to the Lyα line itself (the limitation in comparable studies like Guo et al. 2024)
- Spans a range of SFR, mass, and Lyα detection status (LAEs, LAAs, AGN)
- Current sample (no AGN emission lines detected in stacks):
  - 1.9 < z < 2.75 — AEGIS 232, COSMOS 219, GOODS-N 176
  - 2.75 < z < 3.5 — AEGIS 48, COSMOS 107, GOODS-N 93
  - (Later narrowed to the ~450-500 galaxy AEGIS/COSMOS-only sample once VDFI footprint coverage was applied; see below)
- Benchmarked against 3D-HST: MOSDEF median stellar mass 9.94 vs. 3D-HST 9.40 (log M☉), plus a redshift-distribution comparison

**VDFI data** (the actual spectroscopic data being stacked):
- First data release, EGS and COSMOS fields
- ~100–150 exposures, each with ~35,000 fibers distributed across the sky in individual IFUs, each fiber giving one spectrum

### 1.3 Pre-pipeline setup

- **Data cuts**: quality cut requiring [OIII] or Hα S/N > 5, removing ~10% of the MOSDEF sample
- **Fiber masking**: mask fibers in the upper 90th percentile of brightness in CFHT g-band imaging
- **Config-driven pipeline**: a `config.py` "instructions" object defines field, catalog, fiber-masking method (spectral vs. image-based), bin scheme/type, background annulus extent, minimum background fibers, background smoothing size, fiber-combine method, and more — every run is fully parameterized rather than hardcoded

### 1.4 Full extraction & stacking pipeline

Per galaxy, per exposure:
1. Compute a background from a 57″–63″ annulus, smoothed with a ~200 Å filter
2. Subtract background per fiber, bin fibers into annuli defined in angular, kpc, or virial-radius units
   - kpc bins: redshift-dependent only (~9 kpc ≈ 1″)
   - virial bins: scale with mass, shrink with redshift (1 R_vir ≈ 90 kpc ≈ 10″ historically — later refined, see Part 2); capped at log(M)=11 to avoid runaway bin sizes/memory use
3. Coadd fibers per bin (inv_var vs. biweight — inv_var produced a spurious blue-end continuum rise, likely mis-estimated errors; biweight became the standard)
4. Write each galaxy's per-bin spectra to a FITS product (caching against kernel death at large radii, where hundreds of thousands of fibers get pulled in)
5. Read galaxies back in, de-redshift, convert to physical units (L/kpc², the deprojected analog of F/arcsec²), and coadd across the galaxy sample (biweight standard) — produces one spectrum per radial bin representing the "average galaxy" at that radius
   - Innermost bins (<5″) are PSF-dominated and highly correlated — VIRUS cannot resolve a clean down-the-barrel ISM spectrum

### 1.5 Measurement methods

- **Continuum**: median continuum fit from bounds flanking the line, subtracted; a 1D linear fit is available but risks biasing outer-bin centroids by fitting a slope to noise
- **Line flux**: integrated within ±4 Å of rest-frame Lyα (1215.67 Å), window set by the MOSDEF redshift precision (~60 km/s)
- **Centroid**: several consistent estimators; the adopted method is a median-flux centroid (cumulative flux from the left edge to 50%), more noise-robust than a standard first-moment centroid
- **Errors**: bootstrap resampling, cross-checked against propagated error spectra and found consistent

### 1.6 Core results (as of the June deck)

- **Flux (brightness) profile** vs. radius, shown in both virial and kpc binning
- **Centroid velocity vs. radius**, shown in both virial and kpc binning — red-dominated near center, moving toward/through systemic with radius
- **Other radial diagnostics**: biweight-normalized radius–velocity map, blue-side vs. red-side flux vs. radius, blue/red flux-ratio asymmetry vs. radius (ratio ≈1 near ~4 kpc)

### 1.7 Validation & systematics (established)

- **PSF comparison**: radial continuum profiles of field stars (AEGIS/COSMOS) vs. the galaxy signal — confirms the halo is genuinely more extended than the instrumental PSF; used finer 0–100″ binning and a tighter ±3 Å window (thinner peaks at large radii)
- **LSF comparison**: synthetic LSF built from VIRUS FWHM=5.6 Å convolved with the 60 km/s MOSDEF redshift uncertainty — the LSF sets the width of the red Lyα peak but centroids are recoverable despite the blurring
- **Independent error methods**: z-scramble (randomize redshifts, restack), bootstrap (galaxy resampling), and jackknife (delete-one) — cross-checked against each other across radial bins
- **Centroid-method comparison**: multiple estimators (flux-weighted moment, Gaussian-weighted moment, Gaussian fit, median-flux) agree
- **External cross-check**: compared against an independent 3D-HST sample (Niemeyer et al. 2022) — consistent, a good sign the result isn't dataset-specific

### 1.8 Preliminary subsample looks (June deck, "Extras")

Early, uncontrolled low-vs-high splits shown for: redshift, stellar mass, E(B−V), and SFR100 — flagged as preliminary, not yet controlled or statistically vetted (that rigor is exactly what Part 2's subsample-split framework below was built to formalize).

### 1.9 Known data artifacts (June deck)

- Small spurious flux "spikes" resembling previously-identified fake LAEs in VDFI — traced to miscalibrated pixel errors that show up in inverse-variance galaxy stacks but not in biweight fiber stacks
- A similar mean-vs-biweight outlier issue in far-radius annuli

### 1.10 Insights / interpretation (as of the June deck)

- An observed blueshift trend moving away from a red-dominant central peak, read as consistent with either accretion or a decreasing line-of-sight outflow velocity
- No strong blue-shifted peak detected in the middle bins — argued against gravitational accretion being the dominant signal in this sample
- At far radii (systematics-dependent), a possible net outflow/red-peak signature consistent with clustering with satellite LAEs — supported by a flattening of the flux profile at a similar radius, ~2 R_vir (at the time)

### 1.11 Next steps identified in the June deck

Re-run with additional VDFI data; compare halo extent/kinematics across subsamples grounded in physical hypotheses; run theoretical tests; build/improve a core-spectrum measurement function (escape fraction, core-vs-halo flux, LAE/LAA status — scoped as future/Paper 2 work); define the scope of and begin writing the Lyα halo paper.

---

## Part 2 — Recent work (July 2026)

This is where the June deck's "Next Steps" turned into actual results — a real fit, a formal statistical-significance treatment, a literature-grounded interpretation, and the scaffolding for the subsample-split and UV-continuum work still ahead.

### 2.1 Statistical rigor: combined significance beyond R_vir

Formalized *how confident* the centroid signal beyond R_vir actually is, via two independent methods:
- **`combined_bin_significance`** — combines several already-bootstrapped per-bin centroids into one test, using the full covariance matrix induced by resampling the *same* galaxies across all radial bins in each bootstrap draw (bins aren't independent — treating them as such overstates the evidence). Reports a covariance-aware combined offset/z/p, a naive-independent comparison for contrast, and a whole-vector Mahalanobis test.
- **`pool_bins_and_bootstrap`** — an independent cross-check that instead pools several radial bins at the *flux* level (before any centroid is measured), using the same weighted-combine machinery (`combine_fibers`) the rest of the pipeline already uses, then bootstraps the merged bin directly.

**Result:** individual outer bins (~110–220 kpc) sit at ~1.2–1.35σ each; the covariance-combined significance across all bins beyond R_vir is **~1.7σ** (two-sided, the primary number); the independent flux-pooled cross-check reproduces a consistent central value (~−47 to −48 km/s) at **~1.5σ**. Agreement across two independent combination methods is the strongest part of the case — stated honestly as "a consistent, modest trend," not a confident detection.

Also fixed while building this: `plot_line_panels`'s bootstrap error band was re-anchored onto the fiducial stack (it had been centered on the bootstrap distribution's own median, which can visibly diverge from the fiducial curve in low-S/N outer bins), and a `seed` parameter was threaded through `measure_all_bins`/`bootstrap_all` (previously silently hardcoded, so different runs weren't drawing genuinely independent resamples).

### 2.2 PSF-aware two-component flux-profile fitting

The headline new *result* of the last month: a proper forward-modeled fit to the flux profile, built and validated from scratch.

- **Model:** two-component exponential, `I(r) = A1·exp(-r/h1) + A2·exp(-r/h2)`
- **PSF treatment:** closed-form 1D ring-convolution (`ring_convolution_matrix`) — for a circularly symmetric PSF and a thin unit-flux ring source, computes the exact PSF-smeared contribution to each observed radial bin. Validated via a flux-conservation check (a unit-flux ring must integrate back to 1 at every source radius, out to 2000 kpc). Two real bugs were caught and fixed in the process: a fixed angular-quadrature grid that missed the PSF's shrinking angular support at large radii, and a bin-midpoint-vs-bin-integral mismatch that was wrong by up to ~10⁹× for the widest bins.
- **Two fits, same interface:** `fit_naive` (no PSF correction, drops the innermost bin) and `fit_psf_aware` (full forward model, all bins including the inner one) — both multi-seed with lowest-χ² selection, amplitude-normalized and data-driven bounds for numerical stability.
- **Validated first on synthetic data** (`psf_exponential_recovery.py`, a Phase-1 testbed with known ground truth) before being pointed at the real stack — the exact same fitting code runs on both, so there's no drift between "validated code" and "code that fits real bins."

**Current real-data result (PSF-aware, all 10 bins):** χ²/dof = 7.04/6 = 1.17; **h1 = 16.9 ± 1.1 kpc** (core term), **h2 = 1552 ± 548 kpc** (outer term). h1 sits inside the Steidel et al. 2011 range (20.8–28.4 kpc) for stacked KBSS-like halos; h2 is well beyond the one-halo/CGM regime — the break between the two terms falls almost exactly at R_vir (≈75 kpc), and h2 itself, converted to comoving units, lands close to the only direct-imaging detections of the cosmic web in Lyα emission in the literature (Bacon et al. 2021, 2.5–4 comoving Mpc; a 2024/2025 MUSE Ultra Deep Field 5 Mpc filament result).

**Proposed next step (spec'd, not yet fit — "Option C"):** an exponential core + *cored* power-law halo, `I(r) = A1·exp(-r/h1) + A2·(1+(r/r_c)²)^(-γ/2)`, motivated by a bare power law fitting the outer term noticeably worse and diverging unphysically as r→0. Flags an important prerequisite: a 3D galaxy-clustering correlation slope (γ≈1.8) does *not* project onto a 2D radial profile at the same slope — the Limber approximation predicts a fully-projected slope closer to γ−1≈0.8 — so any literature-slope comparison needs the line-of-sight window depth checked first.

### 2.3 Literature deep-dive: one-halo / two-halo framing

A full literature review (`halo_gas_correlation_literature_review.md`) was written to properly ground the h1/h2 result, organized around the field's own halo-model vocabulary:
- **One-halo term** = galaxy-to-own-CGM correlation, dominates from a few kpc out to ~R_vir
- **Two-halo term** = galaxy-to-neighboring-halo correlation (large-scale structure/clustering bias), dominates beyond R_vir

Surveyed both emission-side exponential fits (Steidel 2011, Wisotzki 2016, Leclercq 2017, and critically **Byrohl et al. 2021**, whose IllustrisTNG50 radiative-transfer simulations show the outer-profile flattening is dominated by photons from *other* halos, not the galaxy's own diffuse gas — the physical interpretation the h2 term is now organized around) and absorption-side tomography (Rakic 2012's FoG/Kaiser two-scale anisotropy split, and **Sorini et al. 2018**, which stitches together small- and large-scale absorption measurements and finds a galaxy's "sphere of influence" extends to ~7×R_vir — a literature-calibrated number that loosely echoes the fitted r_c ≈ 488 ± 211 kpc, ~6.5×R_vir). Also surveyed direct cosmic-web-in-emission detections (Umehata 2019, Bacon 2021) as the comparison set if h2 is confirmed to be tracing genuinely diffuse/large-scale structure.

### 2.4 Discussion-section draft: three organizing pillars

The paper's Discussion section (`docs/paper-writing/discussion-draft-notes.md`) was drafted around three pillars, confirmed as "the heart of the paper." (Originally scoped as four — the R_vir cutoff and the flux-profile fit were merged into one pillar, since the fit turned out to be the quantitative confirmation of the cutoff rather than a separate topic.)

1. **Centroid vs. radius** — the systemic-redshift anchor as the core methodological advance over Guo et al. 2024's self-referenced approach; overall shape (red-dominated center → crosses systemic near R_vir → mildly negative beyond) and candidate physical readings (inflow scattering vs. outflow vs. satellite clustering); the ~1.7σ / ~1.5σ combined-significance result from §2.1, stated as a modest trend.
2. **~1 R_vir cutoff, quantified by the fit** — three independent qualitative diagnostics (flux profile bend, centroid sign flip, blue/red flip) converge on one radius, and the PSF-aware two-component fit (§2.2 above) turns that agreement into a number: the fitted crossover radius between the core and outer terms lands at ≈76 kpc, essentially exactly on the independently-derived **R_vir = 75 kpc** (corrected, AGN-excluded, ~450-galaxy reference value — a ~90 kpc figure elsewhere in the draft is the older full ~500-galaxy AGN-included sample and is superseded). Four independent routes converging on the same radius — three qualitative diagnostics plus one quantitative fit — is the strongest version of this claim. The core term (h1=16.9 kpc) is compared to Steidel/Wisotzki/Leclercq; the outer term's interpretation is left deliberately open between a galaxy-clustering-slope story and Chen et al. 2020's own IGM optical-depth slope (~−0.5), which may be the more physically apt comparison for diffuse emission.
3. **Origins via subsample splits** (§2.5 below) — organized around Byrohl 2021's inner/outer photon-origin decomposition.

### 2.5 Subsample-split framework

A full candidate list (`docs/subsample_splits.md`) was built and split across the two papers:

- **Paper 1 (kinematics, prioritized for discovery):** mass/SFR-normalized mass, ΣSFR, Lyα velocity self-split, z~2 vs. z~3 (mass-normalized), dust attenuation (Balmer decrement/UV slope β), environment/satellite clustering, EW, orientation/inclination, double-Gaussian peak separation & height ratio, burstiness (Hα vs. UV-timescale SFR — flagged as needing Paper 2's larger dataset), half-light radius (mass-normalized).
- **Paper 2 (flux/photon budget):** Hβ-normalized escape fraction, core classification type (AGN/LAE/emitter/null/absorber), PSF-aware exponential scale length of Lyα itself.

Not yet started as actual measurements — this is the scoped-and-prioritized plan, still the top open item in `TODO.md`.

### 2.6 UV-continuum extraction pipeline (Part 3 of the fitting spec — new, untested)

A full extraction pipeline for the sample's rest-frame UV continuum (`uv_profile.py`, 1400+ lines) was designed and written, intended to eventually be fit and compared directly against the Lyα h1/h2 result via the same fitting machinery. **Not yet run against real data** — no CFHT-LS mosaic is present in this workspace/session yet. Pipeline: per-galaxy cutout from CFHT-LS r-band imaging → 2D-Gaussian centroid (kept-but-flagged, never dropped, mirroring the pipeline's existing QC philosophy) → circular-annulus photometry with masked-median background and segmap-based neighbor masking → galaxy-axis coaddition (reusing the same `combine_fibers` machinery as the spectral side) → bootstrap. Default model is a single exponential (`I(r) = A·exp(-r/h_UV)`), with an optional Sersic generalization. The physical payoff once run: a clean, same-sample UV continuum scale length to compare h1 against, replacing the currently-borrowed Steidel continuum figure (3–4 kpc).

### 2.7 Figures & plotting infrastructure

A pass across `analysis.py`/`plotting.py`/`stack.py`/`single.py` added a per-sample/per-galaxy virial-radius-in-kpc convenience (to draw the R_vir reference line directly on flux/centroid plots) and standardized figure-saving filenames — groundwork for the "bring figures up to publication quality" TODO item, though this pass wasn't documented slide-by-slide and may be worth a closer look before finalizing paper figures.

### 2.8 Open systematics / unresolved questions

Carried forward as the honest caveats section, several of which now thread through multiple Discussion pillars at once:
- **Correlated (not random) outer-bin background noise** — still an open systematic that could bias the exact cutoff radius; the background annulus radius and smoothing window both measurably affect outer-bin behavior (findings logged, root cause not yet closed).
- **R_vir mass calibration** — currently assumes a KBSS/LBG-like ~10¹² M☉ halo; LAE clustering (ODIN/DESI) suggests lower masses (~10¹¹), which could shift what "1 R_vir" even means.
- **Double-Gaussian noise-prefabrication question** — open null test: does fitting a double-Gaussian template to pure noise and stacking by the fit's own derived quantities (peak separation, flux ratio) produce a template-shaped stack purely as a selection artifact? Needed before the peak-separation subsample split (§2.5, item 9) can be trusted.
- **r_c/γ degeneracy** in the proposed Option C model — current radial coverage (~1000 kpc) isn't wide enough to break it; would need ~1.5 Mpc.
- **SMHM relation choice** (Moster et al. 2013) — not yet re-validated against the two alternative relations already coded up (Girelli+2020, Behroozi+2019).

---

## Where this leaves things

The June deck was pipeline-and-methods-forward: dataset, extraction, validation, a first look at the headline centroid result, and speculative next steps. The work since has turned two of those "next steps" into real, literature-grounded results (a validated PSF-aware fit with actual numbers, and a formal combined-significance treatment of the R_vir-beyond signal), organized the paper's Discussion section around three pillars built on top of them — with the R_vir cutoff and the flux-profile fit now merged into one, since the fit is the quantitative backbone of the cutoff claim rather than a separate topic — and scoped out, but not yet executed, the two biggest remaining pieces of work: the subsample-split analysis and the UV-continuum comparison.
