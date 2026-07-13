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

### YYYY-MM-DD
- *(add findings here)*

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

## See also

- `GOALS.md` — project scope, headline measurements, and the settled/in-progress validation list
- `subsample_splits.md` — the double-Gaussian peak-separation split (#9) this doc's last section stress-tests
- `TODO.md` (workspace root) — near-term working items, including the ones this doc marks as placeholders
- `validation.py` — the actual code behind the Systematics section above
- `docs/VDFI Notes.txt` (workspace root) — general dated log for the AGN/LAE-catalog side of the workspace; not superseded by this file
