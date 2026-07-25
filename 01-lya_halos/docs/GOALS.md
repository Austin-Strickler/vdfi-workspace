# Goals — 01 Lya Halos (Paper I)

See `docs/vdfi-background.md` (workspace root) for the HETDEX/VDFI dataset context this project builds on, and `docs/paper-writing/three-paper-program.md` (workspace root) for the full rationale behind the three-paper structure this file now follows. That document is the source of truth for scope; this file is the working summary for this project specifically.

## The three-paper program, in one paragraph

VDFI answers one question: where does the Lyα light around a cosmic-noon galaxy come from? Three papers attack it with three different capabilities. **Paper I** (this project) establishes that the light has a boundary at the virial radius and a different kinematic character on each side, measured in an absolute velocity frame. **Paper II** (splits — currently also tracked in this project's `subsample_splits.md`, pending a folder decision) asks whether the light outside that boundary responds to the galaxy or to its environment. **Paper III** (`02-lya_escape`) asks whether the galaxy could have supplied the light at all, from an independent photon budget.

## Overarching goal (Paper I)

Establish that Lyα emission around cosmic-noon galaxies has a physical transition at the virial radius — a different kinematic and profile-shape character inside vs. outside R_vir — measured in an absolute velocity frame anchored to MOSDEF systemic redshifts. This is the headline: **the transition, not the kinematics alone.** The kinematic turnover is corroborating evidence for a boundary established by the two-component profile fit, not a standalone result. Open questions this speaks to: how do galaxies accrete mass; what does a galaxy's mean Lyα halo (not yet split by galaxy property — that's Paper II) tell you about the one-halo/two-halo boundary; what can a *stack* of galaxies tell you that individual measurements can't (and what do we lose by stacking, given our medium-resolution spectrograph)?

## Dataset

- MOSDEF galaxies, magnitude-cut roughly H<24.5 (z~2 bin) and H<25 (z~3 bin) — precise cuts TBD/to confirm.
- ~1500 MOSDEF galaxies total across all MOSDEF fields; ~500 after cutting to the COSMOS/EGS footprint we have VDFI coverage for. Working number, not final.
- Precise rest-frame optical **systemic** redshifts from MOSDEF, reportedly ~60 km/s precision (unverified — came from one source, needs a real citation check before it goes in the paper).

**Why this dataset matters:** the systemic redshift is the project's central methodological advantage over every serious competitor (Niemeyer 2022, Davis 2023, McKay 2026, Cooper 2026 — all VIRUS/HETDEX, none with independent rest-optical systemic redshifts). Guo et al. 2024 measure Lyα centroid shifts *relative to the Lyα line itself* (self-referenced); VDFI can state where the gas actually sits in absolute velocity. That is not a marginal improvement — it changes what kind of statement the measurement can make.

## Core measurements

1. **Centroid velocity vs radius** — the primary result. Is the line at systemic velocity, blueshifted, redshifted, and how does that change with radius? Radial bins are in units of R/Rvir (stellar-mass → halo-mass → virial-radius mapping, currently Moster et al. 2013) — implemented and in active use, including per-galaxy mini-extractions within a galaxy's own R/Rvir. `TODO.md` tracks revisiting whether Moster+2013 is still the right SMHM relation to use.
2. **Light (flux) profile vs radius**, fit as a two-component (one-halo + two-halo) model with a fitted crossover radius. A crossover at ≈76 kpc against an independently derived R_vir = 75 kpc would be the first emission-side localization of the one-halo/two-halo boundary — the field currently has this in absorption (Rakic 2012) and in simulation (Byrohl 2021), not cleanly in emission.
3. **Environment / overdensity split** (Chartab 2020 CANDELS density catalog, via `multicat.py`) — the two-halo interpretation rests on Byrohl 2021's "photons originate in other halos"; this is only testable with an environment split, so it belongs in Paper I rather than Paper II despite being a split, because Paper I's own interpretation can't close without it.
4. **UV continuum comparison (h_Lyα / h_UV)** — the same-sample analog of Steidel et al. 2011: Lyα ~5–10× more extended than the UV continuum. Characterizes the mean profile; not a split.
5. **z~2 vs. z~3 comparison** — doubles as the magnitude-cut systematics check Paper I needs anyway, and connects to Guo et al. 2023's finding of no size evolution across cosmic time.

None of items 3–5 require the PSF-extracted core spectrum. That is the actual boundary between this project and Paper II: **Paper I measures the average galaxy (core photometry); Paper II measures how the population varies around it (core spectroscopy).**

### The Leclercq/Guo tension — a discussion target, not just a result

Leclercq et al. 2020 report the Lyα line reddening at larger halo radii; Guo et al. 2024 report a red→blue trend outward. VDFI's absolute frame can discriminate among the candidate reconciliations (reference-frame artifact, radial-range difference either side of the transition, IGM blue-wing attenuation, sample selection) in a way neither original study could. Worth a dedicated discussion paragraph once the centroid-vs-radius result is finalized.

## Splits that live elsewhere (Paper II scope — not this project's core measurements)

The fourteen-split subsample program (mass, SFR, ΣSFR, dust, EW, size, orientation, core kinematics/classification splits, etc.) is **Paper II's**, organized around a single pre-registered thesis: *the one-halo term tracks the galaxy; the two-halo term tracks the environment.* It is currently tracked in `subsample_splits.md`, which still carries the old two-paper (Paper 1/Paper 2) assignment from before this restructure and needs reconciling with the new Paper I/II/III split (see `docs/paper-writing/three-paper-program.md` §8, item 10 — a housekeeping task, not yet done). Don't treat that file's paper labels as current until that reconciliation happens.

This project's folder (`01-lya_halos`) is where Paper II's splits are also currently being computed and prototyped, since the underlying `measure_*`/`fitting.py` machinery is shared — a folder split (e.g. into paper-1/paper-2 subfolders) is a live open question, not yet decided.

## Validation / systematics

These need to land in a final, publication-ready figure/summary before the transition/centroid result can be trusted. Ordered by how load-bearing each is, per the delivery requirements in `three-paper-program.md` §8:

- **[Critical, blocks all three papers] Correlated outer-bin background noise.** Outer radial bins are noisy, and the noise is *correlated* rather than random — plausibly tied to fiber coaddition, annulus definition, background smoothing choice, and masking method. If underestimated: Paper I loses its outer-halo claims and the transition radius shifts, Paper II loses the two-halo half of its thesis, Paper III loses the beyond-R_vir budget term. The single highest-value open task in the program.
- **[High priority] Quantify the PSF-width systematic on the crossover radius.** h1 moves ~15.1 → 12.0 kpc across 1.3″–2.4″ FWHM — roughly 3× the statistical error — and the data only mildly prefer one width (Δχ²≈1 on 3 dof). `psf_forensics.py` can already sweep this; report h1 as a value with a systematic band, not a bare uncertainty.
- **[High priority] Re-run everything downstream of the 2026-07-22 PSF flux-convention fix.** h1, crossover radii, zone luminosities, three-zone ratios, all subsample splits. Nothing needs building — the pipeline is correct now — but no number means anything until this pass is done.
- **Bootstrap errors** — validated as consistent and reproducible; the standard error method for this pipeline.
- **Centroid method comparison** — consistent across estimators, and cross-checked against an independent paper's 3D-HST sample (Niemeyer 2022 — the strongest single systematics check in the suite: if VDFI's curve tracks hers with smaller error bars, that's inherited credibility).
- **Line spread function (LSF) check** — confirms the ±4 Å measurement window isn't itself dominated by instrumental LSF width.
- **Star PSF comparison** — confirms the halo is genuinely more extended than the instrumental PSF.
- **Two independent routes, not four** — centroid and B/T are functionals of the same flux distribution about the same zero point, so they aren't independent checks of each other; state the honest count (a kinematic turnover and a profile-shape turnover) rather than implying four confirmations.

## Definition of done (for Paper I)

- A validated centroid-vs-radius profile *and* a validated two-component flux-profile fit with a localized crossover radius, backed by the full validation/systematics suite above (correlated-noise closure and PSF-width systematic first), plus the environment split, UV comparison, and z~2/z~3 check.
- A literature comparison, with the Leclercq/Guo tension addressed explicitly as a discussion item, and the Niemeyer 2022 external-replication check in the main text (not an appendix).
- Pipeline choices solidified and defensible — able to justify *why* a given masking percentile or background method was chosen, not just that it was. Prerequisite for the Methods section.
- The paper itself written — Introduction, Data, Methods, Results (tracked in `TODO.md`). Framed as the survey-and-transition paper — the mandatory citation for Papers II and III, so the Data section is not a place to trim.

## Related work

- **Paper II (splits)** — thesis and split list in `subsample_splits.md` (paper-label reconciliation pending), derived-property tooling in `specs/subsample-derived-properties.md`. Currently computed in this project's folder; may move to its own subproject folder later.
- **Paper III (escape fraction / photon budget)** — `02-lya_escape`. Uses this project's `measure_three_zone_ratios`/`measure_psf_corrected_core_onehalo_luminosity` output as an input to the photon-budget accounting.
- **`03-lae_catalog`** — undecided: either a fourth paper or infrastructure for Paper II's spectral classification scheme. Not yet priced into this program (`three-paper-program.md` §8, item 11).

## See also

- `docs/vdfi-background.md` (workspace root) — HETDEX/VDFI dataset context
- `docs/paper-writing/three-paper-program.md` (workspace root) — the full three-paper rationale, sequencing, systematics tiering, and delivery requirements this file summarizes
- `01-lya_halos/README.md` — pipeline stages and usage
- `01-lya_halos/specs/` — per-analysis design docs (write one before a new stacking/measurement approach)
- `01-lya_halos/docs/research-notes.md` — continually-updating log of non-headline research along the way (virial/kpc, SMHM, background subtraction, systematics inventory, core-template noise question)
- `01-lya_halos/docs/subsample_splits.md` — Paper II's split list (paper-label reconciliation pending)
- `TODO.md` (workspace root) — near-term working items
- `presentations/Lya_halos/6-16-26 - VIRAL-Halos Overview.pptx` — source for much of this doc; note per Austin (2026-07-10) that some of its content is outdated relative to what's written here
