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

**RESOLVED 2026-07-27 — pipeline default switched to Girelli+2020.** Relevant code: all three stellar-mass↔halo-mass relations (`moster13` legacy, `girelli20` default, `behroozi19`) plus the comparison tools (`compare_smhm`, `rvir_divergence_table`, `plot_smhm_models`, `plot_rvir_vs_mstar`) now live in `virial.py` directly — `smhm_models.py`/`smhm_compare.py` were merged into a standalone `smhm.py` earlier the same day, which was then merged into `virial.py` once Girelli+2020 became the actual pipeline default rather than a side comparison. Model choice is `PipelineConfig.smhm_model` (default `"girelli20"`), cache-signature- and provenance-tracked.

Tracked in `TODO.md` under Pipeline (was "Confirm the Moster+2013 SMHM mapping is still the right call", now closed with a re-run follow-up).

### 2026-07-27

Ran `rvir_divergence_table` over logM* = 9.0–11.0 (step 0.5) at z=2.3 and z=3.0 (the project's MOSDEF z~2/z~3 bins), `ref='moster13'`. Findings:

- **girelli20 vs moster13**: R200c consistently *smaller*, by −4% (logM*=9.0) widening to −27% (logM*=11.0) at z=2.3; similar shape at z=3.0. Diverges most at the high-mass end.
- **behroozi19 vs moster13**: R200c consistently *larger* at these masses, +12% at logM*=9.0 shrinking to +6% at logM*=10.5 (z=2.3); no median-relation solution (NaN) above ~logM*=10.5–11 at either z — the known high-z turnover (already noted in `GOALS.md`).
- **Sanity check**: `smhm.py`'s `moster13` column reproduces `virial.py`'s production `mhalo_from_mstar_moster13` bit-for-bit at every grid point tested — the merge introduced no regression.
- **Mass-definition caveat, quantified**: `behroozi19`'s naive M200c treatment in `smhm.py` (Mpeak run straight through `R200c_from_M200c`) vs. the proper Mvir→M200c conversion (Diemer & Joyce 2019 concentration-mass via colossus) in `tests/mstar_to_rvir.py` differ by only **+1.7 to +1.9%** in R200c across logM*=9.0–10.5 at both z. The mismatch is real in principle but numerically small in this halo-mass/z regime — if `behroozi19` is adopted, `mstar_to_rvir.py`'s conversion is still the one to wire into the pipeline (it's the mass-definition-correct version), but the naive column here isn't badly wrong for this sample.
- **Literature (what the field actually does)**: Guo et al. 2024 (A&A 691, A66, Yucheng Guo et al., MXDF) estimate r_vir via **Girelli+2020**, cross-checked against the Garel+2015 r_vir–UV-magnitude semi-analytic relation (~20 vs ~23 kpc for their sample — consistent). Lujan Niemeyer et al. 2022 (arXiv:2207.11098, "Lyα Halos around [OIII]-Selected Galaxies in HETDEX" — this is VDFI's actual external centroid-method check, LN22 in the text) estimate r_vir via **Behroozi+2019**, taken directly under the **Bryan & Norman (1998) virial** definition — they do *not* convert to M200c. For their [O III] sample (logM* 8.8–9.9, 68% of the sample) they get rvir ≈ 59–105 kpc.
- **Flag for the crossover-radius claim**: the headline "≈76 kpc measured crossover vs. 75 kpc independently-derived R_vir" coincidence (`GOALS.md` item 2) uses Moster+2013. A 6–12% R_vir shift from switching to Behroozi+2019 (the same relation the project's own external-check paper, LN22, uses) is ≈4.5–9 kpc at R_vir≈75 kpc — comparable to or larger than the ~1 kpc gap the coincidence rests on. This doesn't mean the transition result is wrong, but the "75 kpc" number should be reported with which SMHM relation and stellar mass it assumes, and the crossover-vs-Rvir comparison probably needs a systematic-uncertainty band from SMHM choice alongside the fit-parameter uncertainty already tracked.
- **Still open**: Moster+2013's own halo-mass convention (M200c vs. M200-mean) hasn't been independently re-verified against the original paper — flagged inline in `virial.py`'s module docstring, not yet resolved.

**Decision (same day, after running the divergence table on the real run_table, AGN excluded)**: girelli20 median R_vir = 71.3 kpc [16,84]=[56.9,87.3], 0 turnover failures; moster13 median = 77.1 kpc [58.7,97.6], 0 failures; behroozi19 median = 82.8 kpc [63.3,101.5], **14 galaxies with no median-relation solution** even after the AGN cut (was 34 before — AGN-inflated stellar mass estimates were pushing galaxies past the turnover, consistent with the existing AGN-excluded R_vir convention). Chose **girelli20 as the new pipeline default**: full coverage across the sample's mass range, M200c-native (no mass-definition conversion assumption, unlike behroozi19), and matches Guo et al. 2024's choice for the same kind of measurement. Explicitly did NOT choose based on which relation lands closest to the 75 kpc crossover-coincidence number — that would be picking the answer after seeing it. The crossover-vs-R_vir comparison should report the ±8-12% SMHM-choice spread as a stated systematic rather than a bare 75 kpc.

Implemented 2026-07-27: `PipelineConfig.smhm_model` (default `"girelli20"`), `smhm.py` merged into `virial.py`, all call sites (`extract.py`, `single.py`, `stack.py`) updated, cache signature/provenance/FITS header all track the choice. See `CHANGELOG.md` 2026-07-27 for the full file list. Follow-up (`TODO.md`): re-run everything downstream so figures/tables reflect the new default rather than a stale Moster+2013 cache.

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
- Flux-integration window (fixed vs. variable) — **resolved 2026-07-25, see dated entry below.** Keep the fixed ±3 Å window; a Gaussian-sigma-scaled variable window was tested and rejected.

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

- `sweep_flux_window` / `gaussian_sigma_per_bin` / `gaussian_flux_window` / `h1_vs_flux_window` / `plot_flux_window_diagnostic` — flux-integration window, fixed-width vs. Gaussian-sigma-scaled, decoupled from the (always-fixed) centroid window. Verdict: **keep the fixed ±3 Å window**, see below.

### 2026-07-25
`specs/systematics-and-extraction.md` §1.2, closed. Ran the fixed-width sweep (±3/5/6/8/10 Å) and a Gaussian-sigma-scaled variable window (`k_sigma` from 0.6 to 2.7), refitting `h1` (PSF-aware expcore, refit-per-draw errors) on each.

- Fixed-width sweep: `h1` consistent within errors across ±3–±10 Å. Widening the window mainly grows the error bars (more continuum/sky noise enters the sum); it does not move the central value outside its own uncertainty. Window-insensitive by the original success criterion.
- Gaussian-sigma sweep: `h1` drifted smoothly and monotonically across the *entire* `k_sigma` range with no plateau at either end — the signature of a window that isn't holding the same enclosed-flux fraction bin to bin, not a real core/halo effect.
- Visual inspection (`plot_flux_window_diagnostic`) explains why: the inner (highest-signal) bins have the fewest fibers, so the per-pixel S/N is too poor to constrain the Gaussian sigma fit reliably there — the fit is least trustworthy exactly where it matters most. Separately, Lyα-forest absorption blueward of line center is commonly present in these spectra and is real, not noise; any window wider than ±3 Å increasingly integrates over that absorption trough, which removes flux rather than capturing more of it, and this effect is worst in the same low-fiber inner bins.
- **Verdict:** the small integrated flux the ±3 Å window reports in the inner two bins is not an under-measurement artifact — it is the more trustworthy number. Keep the fixed ±3 Å window for flux (as already used for the centroid). No extraction change follows (this was Stage-2/free throughout).

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

## HST-Hyperion grism catalog: evaluation, redshift calibration, and the protosupercluster sample

**Source:** Forrest et al. 2025, ApJ 985, 61 ([arXiv:2503.04884](https://arxiv.org/abs/2503.04884)); public data release at [MAST HLSP HST-HYPERION](https://archive.stsci.edu/hlsp/hst-hyperion), DOI 10.17909/5kka-ms10.

### 2026-07-28 — what the survey is, and the verdict on using it to grow the galaxy sample

**VERDICT: not adopted as a 3D-HST/MOSDEF catalog augmentation. Adopted (provisionally) as a protosupercluster environment sample — see the next sub-entry.**

**What it actually is.** 50 orbits of WFC3/F160W + G141 over 25 new pointings targeting the Hyperion protosupercluster (z~2.45) in COSMOS, *plus* a from-scratch `grizli` v1.9.5 re-reduction of the adjacent 3D-HST COSMOS G141 data, fit jointly against COSMOS2020 CLASSIC photometry. **COSMOS only — nothing for EGS.** So the honest framing is "3D-HST COSMOS v2, with ~2× the area," not "a new field to bolt on." Selection is *not* [O III]-selected: it is magnitude-limited (all sources with m_HST ≤ 25 in the F140W+F160W mosaic get a spectrum). The [O III] selection we associate with 3D-HST is Bowman et al. 2019's cut *applied to* 3D-HST, and is the parent of Lujan Niemeyer 2022's sample.

**Line coverage vs. our redshift range.** G141 usable range 1.10–1.65 µm; VIRUS Lyα window 3500–5500 Å → 1.879 < z < 3.524.

| Line | in G141 at | overlaps our Lyα range? |
|---|---|---|
| Hα 6563 | 0.68 < z < 1.51 | **never** |
| [O III] 5007 | 1.20 < z < 2.30 | 1.88 – 2.30 |
| Hβ 4861 | 1.26 < z < 2.39 | 1.88 – 2.39 |
| [O II] 3727 | 1.95 < z < 3.43 | 1.95 – 3.43 |

Consequences: **no Hα anywhere in our range** (no Balmer decrement, no line-based dust correction); z > 2.4 rests on a single, weak, unresolved-doublet [O II] feature; 1.95 < z < 2.30 is the only window where O32 is available.

**Why it was rejected as an augmentation.** Bowman+2019 is the comparison, and it wins on every axis: deeper in continuum (m_JH < 26 vs. m_HST ≤ 25), 759 galaxies across *both* AEGIS (470) and COSMOS (289) rather than COSMOS-only, and it carries the full derived-property set (`stmass`, SFR at four timescales with percentiles, `E(B-V)`, `Log Z`, β, sizes, O32, R23) that HST-Hyperion does not release. Chained through footprint and quality cuts, the genuinely-new usable increment is **~15–30 galaxies** against a base of ~500 MOSDEF + ~110 strong-line Bowman — a 2–3% error-bar improvement, in exchange for a bespoke reader, a doublet-convention fix, a missing SED catalog, and a mass zero-point reconciliation. Not a good trade.

**Catalog mechanics (recorded so we don't rediscover them):**

- **The release columns are not the paper's Table 2.** Delivered: `RA DEC mag ID_C20 bf zp_lp zp_ez ID_g zg qf_g qf_gz spec_com` (12). **Missing: `zg_16`, `zg_84` (the p(z) percentiles — the *only* per-object redshift uncertainty), `zs`, `qf_s`.** Added: `bf` (blending flag). Without `zg_16/84` there is no per-galaxy velocity error to propagate — only a global σ_NMAD. Worth an email to Ben Forrest (bforrest@ucdavis.edu) asking for the percentiles.
- **No stellar-population catalog in MAST v1.0**, despite §4.3 promising one ("stellar mass, star-formation rate, and stellar age... in a separate, line-matched catalog"). Fallback if ever needed: `ID_C20` → COSMOS2020 CLASSIC, and where `|z_phot − z_grism|/(1+z) < 0.05` the published `lp_mass` is good to ~0.05 dex without refitting.
- **z-cat has 12812 rows, el-cat has 12814. They are NOT line-matched as delivered — join on `ID_g`.**
- **Both files break standard ASCII readers.** The column-name line lives inside the leading `#` block, and z-cat's trailing `spec_com` is free text containing spaces, so the per-row token count varies (11–16). Fix: split each row with `maxsplit = n_cols − 1` so the remainder lands in the comment column. Numeric coercion needs an explicit null-token list; several el-cat flux columns otherwise stay `str`.
- **`F_OIII` is the [O III] 4959+5007 doublet** (grizli convention), while Bowman's `OIII_FLUX` is 5007 alone. Measured `F_OIII / OIII_FLUX = 1.47`, decomposing as 1.335 (doublet/5007) × 1.12 (the paper's own +0.05 dex reduction offset vs. 3D-HST) = 1.50. Fluxes are in **raw cgs**, not 10⁻¹⁷. **This means Hyperion O32 and Bowman O32 differ by 0.125 dex — do not pool raw.**
- **`ID_g` prefixes (`hyperion_peak1`, `peak2n3n7`, `peak4n5n6`, `peak4extra`) are mosaic-region names, not survey names.** The re-reduced 3D-HST pointings were merged into the `peak1` mosaic (all 65 Bowman cross-matches land there, none elsewhere). To separate footprints, use a convex hull of Bowman's COSMOS positions, not the ID string.
- **`spec_com` is more informative than `qf_gz` for our purposes** — it names *which feature* produced the redshift, and `wide_` (feature broader than the LSF, ~>100 Å) is an explicit **type-1 AGN flag**, which is a free spectroscopic AGN rejection given our AGN-excluded R_vir = 75 kpc convention. `co` ("contamination offset from window") is benign — do not cut on it.

**Footprint accounting, qf_gz ≥ 4, inside the Lyα window (N = 208):** 65 already in Bowman · ~61 inside the 3D-HST footprint but rejected by Bowman's vetting · **82 genuinely new sky**. At qf_gz = 5 (N = 83): 35 / 15 / 33.

**Quality-flag calibration from the paper** (fraction within |Δz|/(1+z) < 0.004 vs. ground-based spec-z): qf=5 → 93.2%, qf=4 → 80.6%, qf=3 → **67.4%**. Catalog shares for qf 0–5: 6.19 / 8.67 / 41.21 / 31.49 / 7.34 / 5.10%. Note the flag system is *anti-correlated* with our redshift range — qf=5 effectively means "Hα in band," i.e. z < 1.5 — so the published calibration is dominated by z < 1.5 and does not transfer to z ~ 2.

### 2026-07-28 — grism redshift precision against MOSDEF systemic redshifts (**the real result of this investigation**)

Cross-matched Bowman+2019 and the HST-Hyperion qf≥4 sample against the MOSDEF catalog (1″ radius) and measured Δz/(1+z) directly. **This is the first direct calibration of grism-redshift precision against true rest-optical systemic redshifts at cosmic noon that we have, and it decides the Bowman question (759 galaxies), not just the Hyperion one.**

| sample / cut | N | median | σ_NMAD | at Lyα |
|---|---|---|---|---|
| Bowman × MOSDEF, all | 81 | +86 km/s | **391 km/s** | 5.0 Å |
| ... [O III] S/N > 5 | 63 | +62 | 315 | 4.1 Å |
| ... S/N > 10 | 38 | +61 | 315 | 4.1 Å |
| ... S/N > 20 | 14 | +85 | **128** | **1.7 Å** |
| ... OIII_EW < 100 Å | 31 | +133 | 571 | 7.4 Å |
| ... EW 100–250 Å | 39 | +60 | 366 | 4.7 Å |
| ... EW 250–500 Å | 8 | +68 | **107** | **1.4 Å** |
| ... z 2.0–2.1 | 10 | +33 | 340 | 4.3 Å |
| ... z 2.1–2.2 | 35 | +86 | 310 | 4.0 Å |
| ... z 2.2–2.35 | 35 | +97 | **485** | 6.4 Å |
| HST-Hyperion × MOSDEF, qf≥4 | 18 | +45 | 456 | 5.9 Å |

Also measured Hyperion × Bowman (same objects, both grism): σ = 330 km/s, +28 km/s. That comparison shares G141 photons, so it measures only reduction/photometry differences and is a **floor**, not the error — which is exactly why the MOSDEF comparison was needed.

**Interpretation.**

1. **The scatter is a line-strength effect, not a floor.** Strong-line objects reach ~110–130 km/s — within a factor of two of MOSDEF's ~60 km/s, and 1.4–1.7 Å at Lyα, comfortably inside our ±4 Å measurement window with **no window widening needed**. Weak-line objects (EW < 100 Å) are at 571 km/s and are unusable.
2. **The z 2.2–2.35 degradation is [O III] λ5007 hitting the red edge of G141** (it exits at z = 2.295), where throughput falls and the wavelength solution is least constrained. A `z < 2.2` cut is cheaper in galaxies than an S/N cut for similar benefit.
3. **The +86 km/s median offset is small and correctable** — its uncertainty is ~1.25σ/√N ≈ 54 km/s, so it is only ~1.6σ from zero. Same sign as the paper's uncorrected +210 km/s against ground-based [O II], consistent with a G141 wavelength zero-point. Residual after correction ≈ 0.7 Å.
4. **The whole-sample merge fails on arithmetic, and this is the reasoning to keep.** At 391 km/s (5.0 Å) against a ±4 Å window, ~35% of the line flux falls outside the aperture — and because the lost fraction depends on the local line width, which varies with radius, it is a **radius-dependent shape bias on h1 and the crossover radius**, not just noise. Widening to ±15 Å restores flux conservation but costs √(30/8) = 1.94× more background noise, against a √(1050/500) = 1.45× sample gain — **net 1.34× worse**, and the penalty lands squarely on the correlated outer-bin background noise that is already the program's #1 blocker.
5. **Therefore: if Bowman is used at all, use it strong-line-selected** (`OIII_EW > 250` or S/N > 20) **and `z < 2.2`**, at the existing ±4 Å window — ~110 galaxies, ~+10% precision, no bias and no noise penalty. **But state the confound:** a high-EW cut selects low-mass, high-sSFR, low-metallicity galaxies, so that arm is an extreme-emission-line population, not a random extension of MOSDEF. It is a good **split** ("EELGs vs. normal SFGs") and a bad **filler** for the mean profile. Tier it — MOSDEF-only for anything velocity-referenced, Bowman normalized and *compared*, never co-added into the fiducial stack.
6. **These numbers belong in the Methods section** regardless of what we do with the catalogs. They are the quantitative justification for why VDFI's MOSDEF anchoring matters — every competitor (Niemeyer 2022, Davis 2023, McKay 2026, Cooper 2026) is working without it, and this table says what that costs.

### 2026-07-28 — the protosupercluster sample (the one thing worth extracting)

**Why this and not the catalog merge.** The two-halo interpretation rests on Byrohl 2021 — photons originating in *other* halos — which predicts the two-halo term scales with environment. A median-density split on Chartab+2020 is a weak version of that test; a stack on a spectroscopically confirmed protosupercluster is the extreme end of the axis, and nothing else in our data provides it. This is core measurement #3 territory, approached from the opposite end.

**The sample, reproduced against the paper.** `qf_gz ≥ 3` and `2.35 < zg < 2.65`, no other cuts → **196 galaxies**, against the paper's §7 figure of 196 (Fig. 18's caption says 192; the paper is mildly inconsistent with itself). Exact agreement validates the catalog parsing end-to-end. Flag breakdown: qf 3 / 4 / 5 = 166 / 24 / 6.

Assigning to the seven Cucciati+2018 ≥5σ peaks (Forrest+2025 Table 4 centres, 3′ radius, |Δz| < 0.025 — **our parameters, not the paper's**): Selene 11, peak5 11, Theia 9, Eos 7, Helios 6, peak7 2, peak6 2 = **48 peak-associated, 148 'field'**. The 3′ aperture is almost certainly too small and too circular — Cucciati's overdensity contours are irregular and filamentary — so treat 48 as a lower bound and tune against Table 4's per-peak counts (which are the *combined* grism + ground-based census, so grism-only should come out lower).

Note the paper's "125 confirmed members at 2.40 < z < 2.53" is that **combined** census, not 125 grism redshifts. Since the release dropped `zs`/`qf_s`, the ground-based half is not recoverable from these files — it would need Forrest+2024's C3VO compilation, which is not public.

**Two properties that make this work despite qf≥3:**

- **Outliers dilute rather than bias.** A 33% catastrophic-outlier rate at qf=3 scatters *field* galaxies into the structure sample, which weakens a two-halo enhancement rather than manufacturing one. Seeing enhancement despite that contamination is therefore a conservative result.
- **The 148 'field' galaxies are an internal matched control** — same redshift range, same instrument, same selection, same sky, same reduction, just not in a peak. That is a far cleaner comparison than anything cross-field.

Velocity-frame work is off the table here: at z ~ 2.5 the redshifts are [O II]-only. This is a **flux-profile measurement** (wide window, shape/amplitude of the two-halo term), not a centroid measurement.

**Implementation plan (per Austin, 2026-07-28).** Manufacture a MOSDEF-shaped catalog rather than touching the pipeline: `field = 'COSMOS'`, `CATALOG = 'HYPERION'`, `z = zg`, `mass = np.nan`, and extract with `bin_mode = 'kpc'` so the virial machinery is never invoked. Then `combine_products` / `split_product_by` handle peak-vs-field as an ordinary categorical split.

Blockers to clear first, in order:

1. **Does VDFI cover the new pointings?** They span RA 149.79–150.35, Dec 2.05–2.50; our COSMOS field is ~0.25 deg². Peaks 4/5/6 sit at the eastern edge (RA ~150.23–150.33) and Peak 2 at the southern (Dec ~2.05–2.10). `plot_field_coverage` answers this in one call. **If coverage misses the eastern peaks this whole section is moot** — and note Selene + peak5 together are 22 of the 48 peak-associated galaxies, both at RA ~150.23–150.26.
2. **Segmentation-mask coverage.** `mask_method='segmap'` uses a baked per-field mask; galaxies in the new pointings sit outside the CANDELS COSMOS strip. Confirm the mask actually covers them — undefined behaviour there means either no masking (bright-neighbour contamination) or everything masked.
3. **`mass = np.nan` guard.** Verify Stage 1 doesn't compute R_vir unconditionally even under `bin_mode='kpc'`; `virial.py` clips log-mass to [7, 11] and NaN may not propagate cleanly.
4. **Background annulus at the field edge.** The 80–90″ annulus will be truncated for galaxies near the coverage boundary — check fiber counts per galaxy before trusting per-galaxy backgrounds out there.
5. **`id_col`.** `ID_g` is a string (`hyperion_peak1_00013`); MOSDEF's `ID` is likely integer. Check `select_core_by_id` / `plot_fiber_mask_appendix` tolerate it.

**Not doing:** adding any of these galaxies to the MOSDEF or Bowman stacks. This is a standalone environment measurement with its own internal control.

---

## Hubble-flow line broadening at large radii — a parameter-free prediction and a flux systematic

*Added 2026-07-30, from Austin's geometric argument. Not yet tested against data.*

**The idea.** The Lyα emission around a galaxy is (statistically) spherically symmetric in 3D. A
measurement at projected radius R integrates along the line of sight, and every LOS offset s carries
a Hubble velocity H(z)·s. So the observed line at projected R is the intrinsic line convolved with
the LOS *depth* distribution of the emission, scaled by H(z). At small R the emitting depth is a few
tens of kpc and the effect is invisible; at Mpc radii the depth is comparable to R itself and the
contribution becomes larger than the instrumental LSF.

**Why it isn't just "the outer bins are broad."** The effect's size depends on the *shape* of the 3D
emissivity, and the two candidate outer profiles predict wildly different broadening. Numbers below
at z = 2.3, where H(z) = 237 km s⁻¹ per proper Mpc and the effective resolution is 1.9 Å rest-frame
= 469 km s⁻¹ FWHM. `j(r)` is the deprojected 3D emissivity; a projected SB slope α corresponds to
j ∝ r^−(α+1), so the expcore outer term's α ≈ 0.8 gives j ∝ r^−1.8.

| R (kpc) | FWHM from exponential halo (h = 17 kpc) | FWHM from j ∝ r^−1.8 | total FWHM (LSF ⊕ power law) | flux inside ±3 Å window |
|---|---|---|---|---|
| 50 | 17 | 26 | 469 | 100% |
| 100 | 24 | 51 | 471 | 100% |
| 200 | 33 | 102 | 480 | 100% |
| 500 | 52 | 255 | 534 | 99.9% |
| 1000 | 73 | 511 | 693 | 98.8% |
| 2000 | 103 | 1022 | 1124 | **87.9%** |

Two things fall out:

**1. It is a discriminator, and a strong one.** At R = 1 Mpc a genuinely 3D-extended clustered
distribution predicts **511 km s⁻¹ of Hubble broadening — larger than the instrumental LSF itself**,
for a total 48% wider line than at small radii. A compact, local, or instrumental origin for the
outer signal predicts **no broadening at all**, and an exponential halo predicts only 73 km s⁻¹.
Measuring the stacked line *width* vs. radius therefore tests whether the outer flux really sits at
the 3D distance the two-halo interpretation requires — independent of, and orthogonal to, every
existing systematics check, none of which use the line width. This is the emission-side analog of
Rakic et al. 2012's absorption-side FoG/Kaiser anisotropy, which is a good citation hook and an
argument that it belongs in Paper I rather than a later paper.

**2. It implies a flux systematic nobody has flagged.** `line_window` is rest-frame ±3 Å about
`LYA_REST` (= ±740 km s⁻¹), applied on the rest-frame resampled grid. A line that broadens with
radius therefore loses an increasing fraction of its flux out of a *fixed* window — negligible
inside 500 kpc, ~1% at 1 Mpc, but **~12% at 2 Mpc.** That biases the outermost bins low and steepens
the measured outer profile, in the same direction as (and confusable with) the correlated-background
question. Cheap to check: re-measure the outer bins with a widened window and see whether the outer
flux rises by the predicted amount. If it does, that is *also* evidence for the broadening.

**Caveats before this is quoted.**
- Peculiar velocities of clustered neighbours (~200–300 km s⁻¹ pairwise) add in quadrature and are
  not separable from the Hubble term without a model. The *scaling with R* is the discriminating
  signature, not the absolute width.
- Random per-galaxy redshift error (~60 km s⁻¹, unverified) also broadens but is radius-independent,
  so it sets a floor rather than a trend — the same argument already made in
  `specs/wavelength-zero-point.md` §5 for why random error doesn't shift the centroid.
- Requires the line width to be measurable at low S/N in the outer bins. Check the achievable width
  precision before scoping this as a result rather than as a systematic.
- The table assumes a single representative redshift. H(z) runs 208 → 312 km s⁻¹/Mpc across
  z = 2 → 3, so the predicted broadening is ~50% stronger in the z~3 bin — **which makes the
  existing z~2 vs. z~3 comparison a free test of the whole picture.**

**Correction to an earlier claim made in conversation (2026-07-30).** I described the S/N minimum
near R_vir as the profile being "genuinely faintest where the two components cross." That is wrong.
Surface brightness declines monotonically — the *signal* at 60–200 kpc is higher than anywhere
further out. The S/N dip is purely a fiber-count effect: the outer bins coadd 10+× more fibers. This
matters for binning strategy, because the mid-radius bins are noise-limited by counting statistics
rather than by the large-scale correlated systematic that limits the outer bins. **They therefore
improve as √N with the incoming data, where the outer bins largely will not** — an argument for
keeping fine radial resolution across the transition now even at low per-bin S/N, rather than
merging it away permanently.

## See also

- `GOALS.md` — project scope, headline measurements, and the settled/in-progress validation list
- `subsample_splits.md` — the double-Gaussian peak-separation split (#9) this doc's last section stress-tests
- `TODO.md` (workspace root) — near-term working items, including the ones this doc marks as placeholders
- `validation.py` — the actual code behind the Systematics section above
- `docs/VDFI Notes.txt` (workspace root) — general dated log for the AGN/LAE-catalog side of the workspace; not superseded by this file
