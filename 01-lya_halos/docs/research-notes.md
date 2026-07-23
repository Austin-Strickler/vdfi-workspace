# Research Notes — 01 Lya Halos

A continually-updating log of non-headline research and investigations from along the way — things worth retaining that don't belong in `GOALS.md` (settled scope/status) or the paper itself (results only), but that we'd otherwise lose track of. Replaces ad-hoc tracking of this kind of thing in the old workspace-root `docs/VDFI Notes.txt`, which stays as-is as a general dated log for the AGN/LAE-catalog side of the workspace — this file is specific to the `01-lya_halos` pipeline.

**Format:** each topic below gets dated sub-entries as investigations happen. Add a new `### YYYY-MM-DD` entry under the relevant topic rather than editing old ones — we want the history of what was tried, not just the current belief. If a topic reaches a settled, paper-ready conclusion, promote the one-line summary into `GOALS.md` and/or the paper draft, but leave the entry here for the reasoning trail.

---

## Virial radius ↔ kpc conversion

Placeholder — no findings logged yet.

Relevant code: `virial.py` — `estimate_M200c_R200c_from_Mstar`, `virial_to_kpc_bins` / `virial_to_angular_bins` (per-galaxy, what the pipeline actually bins with), `median_virial_kpc_bins` (sample-median reporting helper only, not used for binning), `physical_kpc_to_arcsec`. Open questions to log findings against: whether the `m_min=7.0`/`m_max=11.0` log-mass clipping range is still the right guard, whether per-galaxy binning vs. the median-galaxy reporting bins ever diverge enough to matter, and how sensitive R200c is to the assumed cosmology (Planck18, matching the rest of Stage 2).

Tracked in `TODO.md` under Pipeline ("Revisit the virial-radius conversions").

### YYYY-MM-DD
- *(add findings here)*

---

## SMHM values and methods

Placeholder — no findings logged yet.

Relevant code: `smhm_models.py` implements three stellar-mass↔halo-mass relations — `moster13` (current pipeline default), `girelli20` (Girelli+2020, COSMOS-calibrated, z≤4), `behroozi19` (Behroozi+2019 UniverseMachine DR1, note it's Mpeak/Bryan-Norman virial mass, not instantaneous M200c — a real convention mismatch to watch for). `smhm_compare.py` already has `compare_smhm` (per-galaxy logMhalo/R200c across all three models) and `rvir_divergence_table` (grid of logMstar → % R200c difference vs. a reference model) built to quantify exactly this. Open question: has `rvir_divergence_table` actually been run and inspected, and if so, at what stellar masses/redshifts does the choice of SMHM relation change R200c (and hence the R/Rvir bin edges) enough to matter for the centroid-vs-radius result?

Tracked in `TODO.md` under Pipeline ("Confirm the Moster+2013 SMHM mapping is still the right call").

### YYYY-MM-DD
- *(add findings here)*

---

## Background subtraction techniques and field-based anomalies

Relevant code: `smoothing.py` — `smooth_interp_then_gauss` (default: mask the Lyα line, linearly interpolate across the gap, then one uniform Gaussian filter — chosen because it avoids division-fill bowing/pulse artifacts near the mask and is ~100x cheaper) vs. `smooth_spectrum_nan_safe_adaptive` (legacy, NaN-safe adaptive-kernel smoother, kept only for A/B comparison). `optimize.py` — `continuum_noise_metric` scores background/smoothing variants; `noise_from_stacks`, `run_combine_sweep`, `rank_scores`/`score_table` rank them.

"Field-based anomalies" — open thread from the AGN/LAE side (`docs/VDFI Notes.txt`, 2026-02-12): a "trough" problem in the HET data reduction pipeline, flagged as something to chase down at UT Austin. Worth checking whether it has any bearing on background subtraction here (e.g. COSMOS vs. EGS field differences in the correlated outer-bin noise noted in `GOALS.md`), or whether it's unrelated and stays scoped to the AGN/LAE work.

### 2026-07-13
- The annulus radius used to define the background (`config.bg_inner_arcsec`/`bg_outer_arcsec` in `extract.py`) affects the height of the annular bins — the sky background itself changes as a function of radius, which it shouldn't: stars, and foreground objects generally, are not physically related to the galaxies we're stacking, so a real, physical background shouldn't have this kind of radial dependence. Suspect this is at least partly a boundary effect — some objects sit near the edges of the dataset/field — but the continuum level behaving like a function of distance needs to be chased down rather than assumed benign.
- The smoothing window width (`config.bg_smooth_sigma_A`, fed to `smooth_interp_then_gauss` in `smoothing.py`) affects the noise in the outer radial bins. Smaller is generally better, up to roughly ~30 Å — haven't yet mapped out where it stops helping or starts hurting below/above that.
- A full galaxy-level background (pooled across all exposures for a galaxy — `bg_gal` in `extract.py`) is superior to a per-exposure background. Mechanism: each exposure only gets its own background (`bg_per_exp`) if it has `>= config.min_bg_fibers` fibers in the annulus; otherwise it falls back to the galaxy-pooled `bg_gal`. Setting `min_bg_fibers` to something impossibly large (e.g. 99999999) forces every exposure to fail that check and fall back to the pooled galaxy background, which is less noisy than the individual per-exposure ones — effectively a deliberate way to force full galaxy-level pooling instead of exposure-level.

---

## Systematics tests

Everything here is a robustness/null check that would not get its own paragraph in the paper — at most a single sentence ("we tried X across a range of values and got a consistent answer") with the full sweep/plot pushed to an appendix. This section exists so we don't have to re-derive *what we already checked* every time we write that sentence. Pull from `GOALS.md`'s "Validation / systematics" section for the checks that already have a verdict, and from `validation.py`'s actual inventory for checks that exist in code but need their one-line verdict filled in here.

**Already have a verdict (carried over from `GOALS.md`):**
- Bootstrap errors — validated as consistent/reproducible; the standard error method for this pipeline.
- Centroid method comparison — consistent across estimators; cross-checked against an independent 3D-HST sample.
- Line spread function (LSF) check — the ±4 Å measurement window isn't dominated by instrumental LSF width.
- Star PSF comparison — confirms the halo is genuinely more extended than the instrumental PSF.
- Background noise structure — outer-bin noise is correlated, not random; actively being investigated (see Background subtraction section above), not yet a closed item.

**In `validation.py`, verdict TBD (fill in one-liner as each is actually run/checked):**
- `redshift_scramble_null` / `plot_scramble_null` — permute redshift, re-stack; the line should wash out if it's real. Verdict: *(TBD)*
- `placebo_wavelengths` / `plot_placebo` — measure off-line "placebo" windows as a null baseline. Verdict: *(TBD)*
- `zswap_spectra`, `collect_null_ensembles`, `plot_null_envelope` / `plot_null_synthesis` — null-ensemble intuition + verdict. Verdict: *(TBD)*
- `sweep_line_window` / `plot_window_sweep` — centroid vs. ±N Å measurement window. Verdict: *(TBD)*
- `sweep_continuum_method` / `plot_continuum_sweep` — centroid vs. continuum model choice. Verdict: *(TBD)*
- `windowed_flux_scan` / `plot_windowed_scan` — flux measurement vs. window choice. Verdict: *(TBD)*
- `error_cross_check` / `plot_error_cross_check` — propagated errors vs. bootstrap errors. Verdict: *(TBD)*
- `injection_recovery` / `injection_bias_fast` (+ `plot_injection_recovery` / `plot_injection_bias` / `plot_bias_summary`) — inject a known line, recover it; checks for measurement bias. Verdict: *(TBD)*
- `signal_to_noise_spectrum` / `plot_sn_spectrum` / `plot_sn_map` — S/N diagnostics, not a null test per se but part of the same dossier. Verdict: *(TBD)*

`summarize_validation` / `print_validation_summary` / `centroid_dossier` in `validation.py` already exist to roll a run of these up into one summary — worth checking whether that dossier output can just be pasted/linked here directly instead of hand-copying verdicts.

### YYYY-MM-DD
- *(add findings here)*

---

## Core measurement templates (single vs. double Gaussian) & noise-prefabrication

Placeholder — open question, not yet investigated. **Tracked in `TODO.md`.**

The question: core Lyα profiles get fit with single- or double-Gaussian templates (`subsample_splits.md` #9 — peak separation and relative peak height as physical proxies for HI column density and outflow/inflow asymmetry), and those fits feed stacks split by separation (blue- vs. red-shifted), flux ratio, etc. Are we prefabricating the noise into looking like the template we're fitting? I.e., in pure noise (no real signal), does fitting a double-Gaussian template and then stacking by the fit's own derived quantities (separation, flux ratio) converge toward a clean-looking, template-shaped stack purely as an artifact of the fitting + selection procedure, especially at high N? If so, how much of any observed double-peaked structure in the real data is genuine signal vs. this kind of fitting-induced circularity?

This needs a dedicated check — e.g. run the same template-fit-then-stack-by-fit-parameter pipeline on pure-noise (or scrambled/null) spectra at realistic N and see whether a template-shaped stack "emerges" anyway. Depends on things not yet worked out (how to construct a fair null for this specific procedure), hence placeholder + TODO rather than a conclusion here.

### YYYY-MM-DD
- *(add findings here)*

---

## Local overdensity as an environment parameter (MOSDEF, COSMOS+EGS)

Goal: attach a per-galaxy local overdensity to the MOSDEF sample (COSMOS + EGS, z~2-3) and test whether Lyα halo scale depends on environment, as **one binary low/high split** in the existing subsample-split machinery — a single paper paragraph, not a standalone LSS analysis.

Relevant code: no new estimator needed — we adopt a published density catalog and cross-match it in via `multicat.add_matched_column` (see recipe below).

### Plan (settled as of 2026-07-21)

**Data — adopt Chartab et al. 2020, don't build our own.** Chartab et al. (2020, ApJ 890, 7) publicly release a per-galaxy density catalog covering **all five CANDELS fields, including both EGS and COSMOS**, for 86,716 galaxies at H(F160W)<26 AB, 0.4<z<5. It's built with a single uniform pipeline: weighted von Mises kernel-density estimation on the **full photo-z PDFs** (Kodra+2019 CANDELS photo-z, σ_NMAD≈0.02, folding in spec/3D-HST-grism z where available), boundary-corrected, LCV-optimized bandwidth. Density values extend to z=5, so our z~2-3 targets are covered. This is the standard, correct way to measure ~Mpc-scale projected environment — **photo-z-PDF-weighting is the right tool**, and is what lets the two small CANDELS fields work at all. (An earlier version of this note assumed photo-z was unusable and leaned COSMOS-only via spec-z compilations; that was wrong. Photo-z can't do sub-Mpc LOS-resolved density, but that isn't what we need. Spec-z at z~2-3 is far too sparse — ~13 Mpc mean tracer separation in EGS — to resolve small scales in either field, which is exactly why the whole cosmic-noon LSS literature uses photo-z-PDF density fields.)

**Catalog contents (VizieR `J/ApJ/890/7`, table2):** `ID` (CANDELS photometric-catalog ID), `RAJ2000`, `DEJ2000`, `z` (z_best), comoving density, physical density, and **density contrast δ** — δ is the overdensity, already normalized within its z-slice, i.e. the field-relative quantity we want, no post-processing.

**Cross-match by sky position, not ID.** The `ID` column is the CANDELS photometric ID (COSMOS→Nayyeri+2017, EGS→Stefanon+2017), *not* the 3D-HST/Skelton+2014 ID MOSDEF uses — a blind ID join is wrong. But every row has RA/Dec, so a positional match at ~0.3-0.5″ is trivial and robust (MOSDEF observed the CANDELS footprints, so overlap is high). Two checks before trusting a galaxy's δ: (a) it falls inside the WFC3 footprint and clear of the boundary/edge region; (b) it clears Chartab's mass-completeness floor M_min(z) (~10^9.7 at 1.2<z<2.2, higher at 2.2<z<3.5) — most MOSDEF galaxies do, but verify per-galaxy.

**Structure — one binary axis, mass/z as controls.** With ~278 COSMOS + ~244 EGS, a single low/high median split leaves ~120-140 per bin per field (~250/bin pooled) — enough for a halo-scale stack. A mass×z×density grid would leave ~30-60 per cell and just measure noise, so **do not** cross-split; fold mass and redshift in as matched-subsample controls instead. Tag low/high vs. **each field's own median δ**, then the tags may be pooled across fields for the stack — but never compare raw δ values field-to-field (different effective depth/resolution). Note the catalog is a *single* smoothing scale (their LCV bandwidth); a second (sub-Mpc) scale would require re-running their estimator and isn't worth it for the paragraph.

**Headline caveat — the mass–environment degeneracy.** Mass correlates with environment *and* with h1 (our own result), so a naive density split can silently re-derive the mass trend. **Mass-matching the low/high subsamples is what makes the result mean anything.** Second-order: UV/emission-line selection preferentially misses dusty/quiescent galaxies in the densest regions (Chartab 2025), which can dilute a real trend.

### How to run it (uses existing `multicat` + `selection` + `stack`)

Where to get the catalog: **VizieR `J/ApJ/890/7`** (also on the ApJ article page as the machine-readable table, and mirrored via arXiv). Download table2 for the EGS + COSMOS fields as FITS or CSV, or pull it with `astroquery`:

```python
# one-time: fetch Chartab+2020 density catalog, keep EGS + COSMOS rows
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
tab = Vizier.get_catalogs("J/ApJ/890/7")[0]      # table2: density measurements
# inspect tab.colnames once; expect ~ ['Field','ID','RAJ2000','DEJ2000','z','ComD','PhysD','Dcont']
ext = tab[[str(f) in ("EGS", "COSMOS") for f in tab["Field"]]]   # if a Field col exists
ext.rename_columns(["RAJ2000", "DEJ2000", "Dcont"], ["RA", "DEC", "DELTA"])
ext.write("chartab2020_egs_cosmos.fits", overwrite=True)

# in the notebook: carry DELTA onto the galaxy product by position, then split
from utils_lya_halo import multicat, selection, stack
from astropy.table import Table
ext = Table.read("chartab2020_egs_cosmos.fits")

mega2 = multicat.add_matched_column(mega, ext, "DELTA", radius_arcsec=0.5)   # RA/DEC on both sides
groups = multicat.split_product_by(mega2, "DELTA", percentiles=50,
                                    labels=["low_density", "high_density"])   # median split; NaN/unmatched auto-dropped
g_lo, g_hi = groups["low_density"], groups["high_density"]
stacks_lo = stack.build_stacks(cfg_prod, g_lo, keep_cube=True)
stacks_hi = stack.build_stacks(cfg_prod, g_hi, keep_cube=True)
```

`add_matched_column` already does the positional cross-match (defaults to RA/DEC on both sides, unmatched → NaN); `split_product_by(..., percentiles=50)` is a median split that drops non-finite/unmatched rows automatically. For a mass-matched version, pre-filter `mega` to a mass/z-matched subset (or use `selection.split_by_mask` on the mass column) before the density split.

**Open follow-ups:**
1. Pull table2 and confirm the exact EGS/COSMOS column names (Field/RA/Dec/δ) load as expected.
2. Run the positional cross-match; report match rate and how many matched galaxies clear M_min(z) and the boundary.
3. Read Chartab et al. 2025 for the selection-function/mass-matching details before finalizing.

### Draft paragraph (for the paper — pending the actual cross-match + stack)
> To test whether Lyα halo scale depends on large-scale environment, we assign each galaxy in our MOSDEF COSMOS and EGS subsamples a local overdensity from the publicly released CANDELS density-field catalog of Chartab et al. (2020), which reconstructs the galaxy number-density field in both fields with a single, uniform weighted von Mises kernel-density estimator applied to full photometric-redshift probability distributions (σ_NMAD ≈ 0.02), with boundary correction and a mass-completeness limit following Pozzetti et al. (2010). We cross-match our targets to that catalog by sky position and adopt its per-galaxy density contrast, restricting to galaxies above the redshift-dependent stellar-mass completeness limit. Within each field we split the sample at the median overdensity into low- and high-density subsamples — matched in stellar mass and redshift to break the mass–environment degeneracy — and re-measure the stacked Lyα halo scale length in each. Because the density field is normalized within each field, the two fields are combined only at the level of the low/high tag, not by comparing absolute densities.

### See also (this section)
- **VizieR `J/ApJ/890/7` (table2)** — the Chartab+2020 density catalog itself, the primary data product here: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/890/7
- Chartab 2020 - LSS Catalog in CANDELS Fields.pdf (`docs/literature/`) — the paper; method + mass-completeness (Pozzetti 2010) + Table 2 column definitions
- Hatamnia 2026 - COSMOS-Web Density Maps.pdf (`docs/literature/`) — alternative photo-z density field, COSMOS only (cross-check/fallback)
- Chartab et al. 2025, arXiv:2510.07445 (LATIS environment methods) — selection-function pitfalls; not yet in `docs/literature/`

---

## Expcore fit stability: amplitude ordering & r_c degeneracy

Settled as of 2026-07-21 — now the pipeline default (`fitting.fit_psf_aware_expcore`/`fit_naive_expcore`, `analysis.bootstrap_fit_profile`, `plot_flux_profile_fit`/`_two`). Left here for the full reasoning trail per this doc's own instructions; see `CHANGELOG.md` (2026-07-21) for the shipped-code summary and `specs/halo-flux-fitting.md` Part 2's addendum for the spec-level version.

Relevant code: `fitting.py` — `fit_psf_aware_expcore`/`fit_naive_expcore` (the fits themselves), `diagnose_crossover_failures` (new, classifies bootstrap-refit failures), `summarize_diagnosed_params` (new, re-aggregates a diagnosis's physically-ordered subset). `analysis.py` — `bootstrap_fit_profile` (refit-per-draw uncertainty).

### 2026-07-21

**The trigger.** A low- vs. high-density environment subsample split (see "Local overdensity" section below) produced two expcore fits whose h1 and everything else looked similar but whose r_c differed by ~100 kpc in a way that flipped the visual "one-halo/two-halo crossover" picture between the two samples. Question: is that difference physical, or fit noise?

**Diagnosis, step 1 — is r_c actually constrained?** `bootstrap_fit_profile` (nboot=1000, gamma_fixed=0.8, r_c free) gave: low h1=11.15 [9.86,12.2], r_c=162 [84.2,324]; high h1=22.46 [17,24.3], r_c=802 [0.865,2130]. r_c's own uncertainty is larger than or comparable to its value in both samples — not a measurement, closer to unconstrained.

**Diagnosis, step 2 — why does `bootstrap_fit_profile` report ~20-28% "no crossover found"?** Two competing hypotheses going in: (a) the crossover search bracket (`r_max = max(50*h1, 5*r_c)` in `crossover_radius_expcore`) is too small for some draws (censoring — the reported crossover_radius would be biased low), or (b) the fit puts more amplitude in the halo/power-law term (A2) than the core term (A1) near r~0 for some draws (an "inverted" solution — a different degeneracy censoring can't fix). Built `diagnose_crossover_failures` to distinguish them directly: evaluate `diff(r) = A1*exp(-r/h1) - A2*(1+(r/r_c)^2)^(-gamma/2)` at both ends of the search bracket per draw, then for draws where `diff` doesn't change sign, retest with the bracket widened 20x.

Result (nboot=3000): **0 of either sample's failures were censoring.** Low: 2915 crossed, 0 censored (either flavor), 85 inverted (2.8%). High: 2390 crossed, 0 censored, 610 inverted (20.3%). All "no crossover" draws were amplitude inversion, worse on the noisier (high-density) subsample as expected — confirms hypothesis (b), rules out (a) cleanly.

**Diagnosis, step 3 — does filtering the inverted draws out of h1/r_c (not just crossover_radius, which `bootstrap_fit_profile` already filtered) change anything?** `summarize_diagnosed_params` re-aggregated using only the "crossed" (physically-ordered) subset: low h1=11 [9.81,12.2], r_c=179 [93.2,329]; high h1=20.9 [16.6,24.3], r_c=866 [35.2,2370]. h1 barely moved — robust to the inverted-draw contamination. **r_c did not clean up** — even the physically-ordered 79.7% of high's draws span a factor of ~68x (35 to 2370 kpc), confirming the wide r_c isn't just an artifact of blending in inverted solutions; it's a real information-limit on a half-sample split.

**Fix attempt 1 — enforce A1>A2 structurally.** Reparametrized A2 = f*A1, f bounded in [0, f_max≈0.999] (rather than fitting A2 as an independent, unconstrained amplitude) — forbids the inverted branch by construction. Tested standalone (r_c still free, same k=4 parameter budget as before): 500/500 draws converged for both samples (0 failures), but r_c for high still ranged from 2.13 to 2390 kpc and A2 up to 3.72e39 — forcing an ordered solution doesn't cure the underlying non-identifiability, it just relocates where the optimizer lands. **"100% converged" is not itself evidence of a good fit** — worth remembering generally.

**Fix attempt 2 — also fix r_c (borrowed from the full/combined-sample fit).** Same ordering fix, plus r_c held fixed (tested at 400 kpc) instead of floated — down to a 3-parameter fit (A1, h1, f). Result (nboot=500): low h1=11.5 [10.4,12.8], A2=6.08e37 [4.53e37,7.39e37], crossover_radius=57 [51.6,63.3]; high h1=20.9 [18.2,23.5], A2=7.8e37 [6.46e37,9.12e37], crossover_radius=88.7 [78.7,98.8]. Central values barely moved from the free-r_c case; uncertainty on A2 and crossover_radius tightened substantially (crossover_radius's 16/84 width for high: 57.2 kpc free-but-filtered → 20.1 kpc fixed).

**Sensitivity scan — is the fixed r_c value doing the work, or is 400 special?** Reran the fixed-r_c fit (ordering enforced) across r_c_fixed ∈ {200,300,400,500,700} kpc, nboot=1000 each:

| r_c_fixed | low h1 | low crossover_radius | high h1 | high crossover_radius | high/low ratio |
|---|---|---|---|---|---|
| 200 | 11.1 [10,12.4] | 51.1 [46.1,57.2] | 19.4 [17.1,21.9] | 76.8 [68.3,86] | 1.503 |
| 300 | 11.4 [10.2,12.7] | 54.7 [49.4,61.1] | 20.4 [17.9,23] | 84 [74.6,94.1] | 1.536 |
| 400 | 11.6 [10.3,12.9] | 57.2 [51.5,63.7] | 21 [18.3,23.6] | 89 [79.1,99.7] | 1.556 |
| 500 | 11.7 [10.4,13] | 59.1 [53.1,65.8] | 21.4 [18.6,24] | 92.4 [82.3,104] | 1.563 |
| 700 | 11.8 [10.5,13.2] | 61.7 [55.4,68.8] | 21.9 [19,24.6] | 97 [86.4,109] | 1.572 |

n_fail = 0/1000 in all 10 runs (5 r_c values × 2 samples) — the ordering fix holds up at this bootstrap size too.

**Conclusions from the scan:** h1 drifts only mildly (~6-13%) across a 3.5x span in the assumed r_c — robust. The **absolute** crossover_radius is NOT r_c-independent (scales up ~20-26% across the same span, as expected physically: a larger r_c keeps the halo term near its plateau longer, pushing the core/halo crossing outward) — report it as conditional on r_c_fixed, with the scan range as an explicit systematic band, not as a free-standing measurement. The high/low **ratio**, however, sits in a tight 1.50-1.57 band across the whole scan (only ~5% relative movement despite the 3.5x span in the underlying assumption) — because both samples share the same fixed r_c in each pass, the systematic largely cancels in the ratio. **"The two-halo crossover sits ~50-57% further out in the high-density environment" is the robust, r_c-choice-independent claim; "the crossover is at 89 kpc" is not**, without stating which r_c_fixed it's conditional on.

**AIC/BIC verdict (fiducial fit, r_c_fixed=400 vs. free, both with the ordering fix):** low — free (k=4): chi2=6.23, dof=8, chi2/dof=0.778, AIC=14.23, BIC=16.17; fixed (k=3): chi2=8.50, dof=9, chi2/dof=0.945, AIC=14.50, BIC=15.96 (a wash, |ΔAIC|/|ΔBIC| < 2). High — free (k=4): chi2=12.17, dof=8, chi2/dof=1.52, AIC=20.17, BIC=22.11; fixed (k=3): chi2=12.34, dof=9, chi2/dof=1.37, AIC=18.34, BIC=19.79 (fixing r_c *improves* both criteria; ΔBIC≈2.3 is real, if modest, positive evidence for the simpler model) — the free r_c parameter wasn't earning its keep in the noisier subsample, exactly where you'd want that to be true.

**Decision:** adopt A2=f*A1 ordering (f_max=0.999) + r_c_fixed=400.0 as the new pipeline default (mirrors `gamma_fixed`'s existing None-means-free convention exactly; pass `r_c_fixed=None` for the old floating behavior, still available for explicit A/B comparison via `compare_models_aic_bic`). Report h1 and the crossover_radius *ratio* between subsamples as the robust numbers; report absolute crossover_radius as conditional on r_c_fixed with the scan range as its systematic uncertainty; continue not reporting r_c/A2/gamma as independently physical numbers (the zone-luminosity and velocity measurements in `specs/subsample-derived-properties.md` were already unaffected by any of this, since they depend only on the derived boundary_radius, not on r_c/A2 point estimates).

**Open follow-up:** the AIC/BIC and sensitivity-scan numbers above are from one specific real low/high-density subsample pair — worth re-running the same scan on at least one other split (e.g. the total/full-sample fit, or a different subsample-split candidate from `subsample_splits.md`) before treating r_c_fixed=400 as universal rather than "validated for the density split, plausible elsewhere."

---

## Pipeline organization pass: environment.py merge + halo→two-halo rename

### 2026-07-21

Two housekeeping changes requested alongside the r_c_fixed work above, both code-complete.

**`environment.py` folded into `multicat.py`.** The Chartab et al. (2020) CANDELS overdensity helpers (`load_overdensity`, `attach_overdensity`, `split_by_overdensity` — see "Local overdensity as an environment parameter" above) lived in their own 4-function module. Too small to earn a separate file, so it was merged into `multicat.py` (the module whose `add_matched_column`/`split_product_by` it's built on) and the standalone file deleted. No call-site changes needed — nothing else imported `utils_lya_halo.environment` as a module.

**"halo" → "two-halo" renamed wherever it meant the large-scale/clustering zone.** The pipeline had accumulated a real naming ambiguity: `measure_halo_luminosity` and `measure_core_halo_velocity` both used bare "halo" for the zone beyond the fitted core/halo crossover — which is physically the **two-halo**/clustering-term contribution (random density correlations + cosmic-web filaments), not a galaxy's own CGM halo. Meanwhile `h1` (the fit's inner exponential scale) is what's conventionally called "the halo scale length" (see the h1-is-the-halo-scale convention note), and `measure_onehalo_luminosity` (added 2026-07-18, see `specs/subsample-derived-properties.md` Part 3b) was already correctly named for the true one-halo/CGM zone. Austin's instruction: rename the ambiguous ones, leave `measure_onehalo_luminosity` and `find_core_halo_boundary`/`boundary_radius` (the crossover concept itself, not either zone) alone.

Renamed: `measure_halo_luminosity` → `measure_twohalo_luminosity`; `measure_core_halo_velocity` → `measure_core_twohalo_velocity` (its `halo_combine` param and `halo_v_*` return keys → `twohalo_combine`/`twohalo_v_*`); `measure_three_zone_ratios`'s `halo_lum` parameter → `twohalo_lum`; `describe_subsample_properties`'s `halo_lum` parameter → `twohalo_lum`. `measure_outer_properties` (the one-call bundle) updated to match. All of `measure.py`, `analysis.py` (including `plot_flux_profile_fit`/`_two`'s "halo term alone" plot-legend labels → "two-halo term alone"), and `guide.py`'s front-door registry updated to match; `specs/subsample-derived-properties.md` got a dated addendum rather than a rewrite of its 2026-07-17/18 design narrative (preserved as the historical record).

**A genuine pre-existing bug surfaced during the rename, not just a cosmetic rename:** `describe_subsample_properties`'s velocity section printed under the header `"-- velocity: core vs. halo (one-halo/CGM zone) --"`, but the value it prints (`vel['halo_v_fid']`, now `vel['twohalo_v_fid']`) comes from the COMBINED OUTER bins beyond the boundary — the two-halo zone, not one-halo/CGM. This directly validates Austin's stated reasoning for the rename ("the two-halo is the one that has a negative average velocity in those bins" — an infall signature at large radius). Fixed to `"-- velocity: core vs. two-halo --"` alongside the rename.

While in `guide.py`, ran `check_guide()` and found (independent of the rename) 6 public functions missing front-door entries: `measure_onehalo_luminosity`, `measure_three_zone_ratios`, `analysis.describe_subsample_properties` (all added 2026-07-18, never registered), plus the 3 newly-merged `multicat` overdensity functions. Added entries for all 6 so `check_guide()` now reports clean (`{'missing': [], 'new': []}`).

Full function-level diff: `CHANGELOG.md` (2026-07-21).

**Follow-up, same day: `fitting.py`/`uv_profile.py` registered in `guide.py`, two investigative functions retired.** The pipeline audit above (in the prior chat turn) flagged that `fitting.py` (the PSF/profile-fitting engine) and `uv_profile.py` (the UV-continuum pipeline) had zero entries in `guide.py`'s front-end registry — and since `check_guide()` only scans modules that already have at least one entry, neither file was ever covered by the drift check at all. Added a `fitting` section (47 entries: PSF & convolution, Two-exponential fit, Expcore fit, Model comparison & boundary, UV-continuum fit) and a `uv_continuum` section (27 entries: Field & catalog I/O, Per-galaxy extraction, Coadd & fit, Stellar PSF) to `guide.py`. `check_guide()` now reports clean across all 8 sections.

At the same time, removed `diagnose_crossover_failures`/`summarize_diagnosed_params` from `fitting.py` — both were one-off diagnostic tooling built specifically for the r_c/amplitude-ordering investigation above, and Austin asked for them to be dropped now that the investigation is closed and its fix has shipped as the pipeline standard. Confirmed no other code called them; the removal is noted in-line in `fitting.py` and in `CHANGELOG.md`, with this section remaining the full historical write-up of what they found.

---

## See also

- `GOALS.md` — project scope, headline measurements, and the settled/in-progress validation list
- `subsample_splits.md` — the double-Gaussian peak-separation split (#9) this doc's last section stress-tests
- `TODO.md` (workspace root) — near-term working items, including the ones this doc marks as placeholders
- `validation.py` — the actual code behind the Systematics section above
- `docs/VDFI Notes.txt` (workspace root) — general dated log for the AGN/LAE-catalog side of the workspace; not superseded by this file
