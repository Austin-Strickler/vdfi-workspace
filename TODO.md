# VDFI To-Do

Working list across the workspace. Check items off as they land; log the actual change in `CHANGELOG.md` when they do. Newest items go at the bottom of their section unless priority says otherwise.

## Pipeline

- [ ] **[TOP PRIORITY] Verify the MOSDEF ↔ VDFI velocity zero point.** The absolute frame is the project's headline methodological claim and the tie between the MOSDEF near-IR and VIRUS optical wavelength scales has never been checked. Leading hypothesis is air/vacuum: λ_vac − λ_air ≈ 1.1 Å ≈ **85 km s⁻¹ blueward**, nearly flat across the band — right sign, right size for the observed outer blueshift.
  **Route (revised 2026-07-30):** Tier 0 convention audit (trace the wavelength solution through `utils_python/`, confirm MOSDEF/HETDEX conventions, check barycentric) **+ Gaia-anchored stellar RVs** (Tier 2, promoted — never stack the stars). The z~1.5 core extraction is **deferred**, see below. Sky lines are blocked.
  Full plan: `specs/wavelength-zero-point.md`. Bounded damage, and write it into Methods regardless of outcome: a rigid offset cannot produce the radial *gradient*, only the absolute outer-bin sign.

- [ ] **Re-run everything downstream of the 2026-07-22 PSF fix and the 2026-07-27 SMHM switch.** Old caches are invalidated by signature, but every existing figure/table still reflects the old values — h1, crossover radii, zone luminosities, three-zone ratios, all splits. Report crossover-vs-R_vir with the ±8–12% SMHM systematic, not a bare number.

- [ ] **Two follow-ups from the splits build, both methodological.** (a) The trend-mode Spearman permutation test has a floor of `2/n_bins!` — **it cannot return p < 0.05 at the spec's default `n_bins=4`** (floor 0.083); use `n_bins=5` or quote ρ and the 1-in-12 directly, and never write "not significant" about a 4-bin trend. (b) Li & Ji's M_eff is nearly flat at this matrix size (ρ=0.84 → M_eff exactly 2.00, no reduction); `independence()` now quotes Cheverud–Nyholt instead, with Li & Ji as an upper bound. Decide which goes in the paper's methods section.

- [ ] **Point the split library's placeholder columns at real ones before the first batch.** `mass_bounds` still carries the `[0, 9.4, 10.2, MAX]` placeholder edges (spec §11 item 2, yours to set); `delta_ms` is specified but blocked until the `DELTA_MS` calibration is chosen; `dust_massctrl` guesses `AV`. Run `splits.preview(cfg, prod, splitcat=sc)` — it reports each of these as BLOCKED with the missing column named, rather than silently skipping them.

- [ ] **[INJECTION — NEXT] Build `inject_sources` + the `InjectedSpectra` overlay.** Ladder levels 0 and 1 both **PASS on real COSMOS data** as of 2026-08-16 (`inj.check_level0` / `inj.check_level1`, four notebook cells total). The template is validated; nothing downstream of it exists. This one build gates every remaining rung — levels 2–4 (zero background), 6 (one bright source in real data) and 7 (the full run) are the *same* code path with different settings, not separate builds.
  **Four pieces, in order, each testable before the next:** (1) the read-only wrapper that impersonates the flux memmap and adds the source on the fly, carrying a `zero_base` flag so the zero-background run is a config switch rather than a 16 GB array; (2) `inject_sources`; (3) injection parameters into `extract_signature_dict` + an `INJECT` FITS keyword; (4) two tests — **empty injection → byte-identical output**, and **signature changes when injection is on** (that second one is the only failure here that could corrupt existing science products rather than just produce a wrong test).
  **Audit already done (2026-08-16):** every `spectra[...]` read in `utils_lya_halo/` conforms to the one pattern the wrapper supports, including the three `single.py` sites added 08-07; `extract.py:1396` is non-conforming but runs *before* injection (§1.4), so the wrapper never sees it. The §1.2 decision gate is answered — the simple wrapper is sufficient, the materialize-the-cube fallback is not needed.
  **Agreed run order after the build:** zeros (levels 2–4 as ONE step) → one very bright source, full extraction, eyeball the spectra vs radius → point-vs-extended and the velocity step → 3–5 sources (first stacking test) → timing → full run. Do not skip the 3–5 source step; one source never exercises the stacking path.
  **Carry into it:** `check_level1` holds every exposure's template (3.1 GB, and per-exposure cost degraded 4× over the run) — templates must be consumed per exposure at level 7, not accumulated. Spec: `specs/injection-recovery.md` v6.

- [ ] **Close the correlated outer-bin noise.** The only item load-bearing for all three papers. Extract random-position nulls (N_r ≈ 500–1000) in the same pass as the next re-extraction — roadmap §1.4 argues that returns a Hartlap-corrected bin–bin covariance matrix, not just an error floor, against §7's "unaffordable ~9000" claim. Unresolved; resolve it while the machine is already running.

- [ ] **Re-extract the stars under the 80–90″ background config.** The `0_Extractor_Stars` runs are still on the old 57–63″ setup, so the star-PSF systematic rests on a stale config.

- [ ] **Codebase organization pass.** ~34k lines across 28 modules; functions aren't where you'd look. First target: background logic is split across `background.py` (Stage-2 testing), `smoothing.py` (smoothers), `extract.py` (annulus selection/combine), `core.py` (`inspect_core_background`) — consolidate into `background.py`. Then the `specs/`/`docs/`/notebook-snapshot layout. Deliberate pass, `guide.py` + `check_guide()` clean at every step. Precedent: the `environment.py` → `multicat.py` merge.

- [ ] **Register `injection.py` in `guide.py`.** The module has **zero entries** — no `build_source_template`, `predict_binned_observables`, `check_level0`, `check_level1`, `SourceModel`, nothing — so `check_guide()` gives the project's most active module zero audit coverage, silently. Flagged in the 08-10 briefing and still open. Also unregistered: `test_framework.py` (08-03), `lya_2halo_deproject.py` (08-05).

- [ ] **Update `run_extract_batch.py`'s `RUN` block to the 11-bin list.** It still carries the old 13-edge list ending at 3000 kpc. Unused today (extraction runs from a notebook cell), but the first time it *is* used it will produce Stage-1 products the injection truth does not match, and the symptom is one extra bin with no truth rather than anything obviously wrong.

- [ ] Systematics: solidify and defend the pipeline choices and techniques.

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
