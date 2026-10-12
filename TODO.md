# VDFI To-Do

Working list across the workspace. Check items off as they land; log the actual change in `CHANGELOG.md` when they do. Newest items go at the bottom of their section unless priority says otherwise.

## Pipeline

- [ ] **[TOP PRIORITY] Verify the MOSDEF ↔ VDFI velocity zero point.** The absolute frame is the project's headline methodological claim and the tie between the MOSDEF near-IR and VIRUS optical wavelength scales has never been checked. Leading hypothesis is air/vacuum: λ_vac − λ_air ≈ 1.1 Å ≈ **85 km s⁻¹ blueward**, nearly flat across the band — right sign, right size for the observed outer blueshift.
  **Route (revised 2026-07-30):** Tier 0 convention audit (trace the wavelength solution through `utils_python/`, confirm MOSDEF/HETDEX conventions, check barycentric) **+ Gaia-anchored stellar RVs** (Tier 2, promoted — never stack the stars). The z~1.5 core extraction is **deferred**, see below. Sky lines are blocked.
  Full plan: `specs/wavelength-zero-point.md`. Bounded damage, and write it into Methods regardless of outcome: a rigid offset cannot produce the radial *gradient*, only the absolute outer-bin sign.

- [ ] **Re-run everything downstream of the 2026-07-22 PSF fix and the 2026-07-27 SMHM switch.** Old caches are invalidated by signature, but every existing figure/table still reflects the old values — h1, crossover radii, zone luminosities, three-zone ratios, all splits. Report crossover-vs-R_vir with the ±8–12% SMHM systematic, not a bare number.

- [ ] **Two follow-ups from the splits build, both methodological.** (a) The trend-mode Spearman permutation test has a floor of `2/n_bins!` — **it cannot return p < 0.05 at the spec's default `n_bins=4`** (floor 0.083); use `n_bins=5` or quote ρ and the 1-in-12 directly, and never write "not significant" about a 4-bin trend. (b) Li & Ji's M_eff is nearly flat at this matrix size (ρ=0.84 → M_eff exactly 2.00, no reduction); `independence()` now quotes Cheverud–Nyholt instead, with Li & Ji as an upper bound. Decide which goes in the paper's methods section.

- [ ] **Point the split library's placeholder columns at real ones before the first batch.** `mass_bounds` still carries the `[0, 9.4, 10.2, MAX]` placeholder edges (spec §11 item 2, yours to set); `delta_ms` is specified but blocked until the `DELTA_MS` calibration is chosen; `dust_massctrl` guesses `AV`. Run `splits.preview(cfg, prod, splitcat=sc)` — it reports each of these as BLOCKED with the missing column named, rather than silently skipping them.

- [x] **Build `inject_sources` + the `InjectedSpectra` overlay.** Shipped 2026-08-17. `InjectedSpectra` (the read-only wrapper with the `zero_base` flag), `inject_sources`, `validate_inj_table`, `draw_injection_positions` and `check_overlay_identity` are all in `injection.py`. The §1.2 decision gate was answered as expected — the simple wrapper is sufficient, the materialize-the-cube fallback was not needed.

- [ ] **[INJECTION — NEXT] Apply the per-km/s → per-Å line-units fix at the injector.** `line_profile` has unit integral in **velocity** and is written into a **per-Å** cube, so injected flux is low by `c/λ_obs` — **72.5× at z = 2.4, and z-dependent**. Diagnosed 2026-08-19 and still marked "fix NOT applied" in `injection.py`'s own comments. `l_inj_for_luminosity(..., line_is_velocity_density=True)` compensates at the luminosity-sizing end, but any template built without going through that hook is still low by the same factor.
  **The fix is a multiplication by `C_KMS / lam0_obs`.** It is on the list ahead of the remaining runs because it invalidates every flux- or luminosity-keyed criterion measured before it, **criterion 1a included** — a pass measured through a known 72.5× flux error is not a pass. It does **not** touch the velocity criteria: the bug is in flux normalisation and a symmetric rescaling does not move a median centroid, so the velocity work can proceed in parallel.

- [ ] **[INJECTION] Run family P, and close criterion 1c.** Two independent things, both open.
  (a) **Family P has never been run**, so criteria 2–7 are untested — the injection suite has validated one source family (G) out of three. The value of pre-registering seven criteria is lost if one is tested and the paper written anyway.
  (b) **Criterion 1c is failing in the `[60,100]` bin, which is the bin containing R_vir** — i.e. exactly where Paper I's headline transition lives. `build_velocity_response_matrix` + `deconvolve_velocity` (shipped 08-20) now make this decidable: run the response matrix on that bin and ask whether the offset **survives deconvolution** (a real radius-dependent bias, and Results cannot be written until it is understood) or **is** the PSF+binning operator (a known forward-modelling effect you correct and report). Do not attribute it to the level-1 fiber-sampling effect by analogy — that was demonstrated for the outermost bin, not this one.
  **Carry into the runs:** `check_level1` holds every exposure's template (3.1 GB, and per-exposure cost degraded 4× over the run) — templates must be consumed per exposure at level 7, not accumulated. Continuous-velocity fields are ~25× slower and have a guard. Spec: `specs/injection-recovery.md` v8.

- [ ] **Rename one of the two `plot_injection_recovery` definitions in `injection.py`** (lines ~2757 and ~2925). Python keeps the second, so the level-6 three-curve figure is currently unreachable. The name is also taken by `validation.py`'s estimator-bias close-up, so there are three functions and one name.

- [ ] **Close the correlated outer-bin noise.** The only item load-bearing for all three papers. Extract random-position nulls (N_r ≈ 500–1000) in the same pass as the next re-extraction — roadmap §1.4 argues that returns a Hartlap-corrected bin–bin covariance matrix, not just an error floor, against §7's "unaffordable ~9000" claim. Unresolved; resolve it while the machine is already running.

- [ ] **Re-extract the stars under the 80–90″ background config.** The `0_Extractor_Stars` runs are still on the old 57–63″ setup, so the star-PSF systematic rests on a stale config.

- [ ] **Codebase organization pass.** ~34k lines across 28 modules; functions aren't where you'd look. First target: background logic is split across `background.py` (Stage-2 testing), `smoothing.py` (smoothers), `extract.py` (annulus selection/combine), `core.py` (`inspect_core_background`) — consolidate into `background.py`. Then the `specs/`/`docs/`/notebook-snapshot layout. Deliberate pass, `guide.py` + `check_guide()` clean at every step. Precedent: the `environment.py` → `multicat.py` merge.

- [x] **Register `injection.py` in `guide.py`.** Done 2026-08-23: 34 entries across four new "Source injection" subsections of the validation section, covering all 27 public functions and all 7 classes. `_BACKEND["injection"]` is deliberately empty — nothing in the module is exempt. The section blurb now distinguishes the two injection facilities: `validation.py`'s "Injection-recovery" (a line into an existing stacked spectrum, for estimator bias) vs `injection.py`'s "Source injection" (a synthetic galaxy into the raw fiber cube, through the whole chain).

- [ ] **Register `test_framework.py` (08-03) and `lya_2halo_deproject.py` (08-05) in `guide.py`.** Still zero entries each, so still zero audit coverage. `lya_2halo_deproject.py` also still sits at `code/` root outside `utils_lya_halo/` — it holds a falsification result, so it is not scratch and wants a home.

- [ ] **Update `run_extract_batch.py`'s `RUN` block to the 11-bin list.** It still carries the old 13-edge list ending at 3000 kpc. Unused today (extraction runs from a notebook cell), but the first time it *is* used it will produce Stage-1 products the injection truth does not match, and the symptom is one extra bin with no truth rather than anything obviously wrong.

- [ ] Systematics: solidify and defend the pipeline choices and techniques.

- [ ] **Restructure the core: take it from a fitted compact term, not a point-source aperture correction.** (2026-09-25, pipeline audit I6.) `measure_psf_corrected_core_luminosity` and `measure_psf_corrected_core_onehalo_luminosity` divide the 0–10 kpc bin by a point-source encircled-energy fraction (x1.42 at the old 1.3" default, x2.31 at the measured 2.09"), but the fitted profile has no point-source term: its inner light is the resolved exponential (h1). A pure exponential with no compact part would still get a large "core", so the core / one-halo split (halo fraction) is not model-consistent.
  **Plan:** (1) add a compact component to the profile model (PSF-shaped, or a UV-like scale per the ~10:1 halo/UV result) next to A1*exp(-r/h1) + the cored power law; (2) take the core luminosity from that term and the one-halo from the exponential, so the halo fraction comes out of one model; (3) if the data don't want a compact term (Delta-AIC, cf. systematics spec §1.7), take the core from the fitted model inside a stated radius instead. Touches `fitting.py` (model + forward model), `analysis.bootstrap_fit_profile`, `mcmc_fit_expcore`, the `measure.*core*` functions and `splits`. Details: `01-lya_halos/docs/pipeline-audit-2026-09-25.md` I6.

## Paper

- [ ] Edit the Introduction, Data, Methods, and Results sections
- [ ] Improve and generate proper paper figures — nail down what each should show, bring to publication quality. Continual.

## Science

- [ ] **Literature claims register.** Work through the field systematically, extract each paper's specific testable claims, and test them against VDFI. Doubles as model-building and as Introduction/Discussion material. See `docs/paper-writing/` for where it lands.
- [ ] **Hubble-flow line broadening at large radii.** σ_line should grow ~linearly with projected radius with slope set by H(z) — a parameter-free prediction, and a direct emission-side analog of Rakic 2012's absorption FoG/Kaiser result. Also flags a possible flux systematic: a fixed measurement window loses an increasing fraction of a broadening line with radius. See the note in `research-notes.md`.


## Deferred

- [ ] **z~1.5 core extraction (as a zero-point driver).** *(Deferred 2026-07-30: tested, the signal isn't there.)* Would have given four systemic anchors spanning 3888–5154 Å in one pass. Revisit with the incoming ~2× data. If it is ever revived: the pipeline has never extracted a non-Lyα sample, so the line mask, `cont_bounds`, and anything keyed to `LYA_REST` need re-pointing first — and do not report a C[III] centroid from a single-Gaussian fit (doublet-ratio bias up to 62 km s⁻¹, `wavelength-zero-point.md` §4.5).
- [ ] **SLURM galaxy-slicing / batch submission.** Not needed at current data volume. Revisit when the incoming data pushes a run past ~3 days or out of memory.
- [ ] MCMC on the flux fit — degeneracies between gamma and the power-law scale radius.
- [ ] Lyα EW and SFR vs. predicted maximum, to constrain emission sources at large radii (Paper III). Connect SF-powered emission to what gravitational cooling could supply.
- [ ] **UV-continuum exponential scale + halo/UV ratio.** CFHT-LS PSF is the hard part; fallback is redoing the UV profile on 3D-HST imaging.
- [ ] **Double-Gaussian null test.** Do single/double-Gaussian core templates fit noise that resembles the template by construction — does stacking by the fit's own derived quantities manufacture template-shaped structure at high N? Gates the peak-separation and peak-height splits and part of the classification scheme. See `research-notes.md`.
