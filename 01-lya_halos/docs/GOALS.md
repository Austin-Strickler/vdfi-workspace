# Goals — the VDFI Lyα program

**Scope of this file (changed 2026-07-30).** This used to be "Goals — 01 Lya Halos (Paper I)."
It is now the **program-level** goals document: one shared dataset section, then a section per
candidate paper. `01-lya_halos` is the umbrella folder for all of it while we're exploring.

See `docs/VDFI-background.md` (workspace root) for the HETDEX/VDFI dataset context, and
`docs/paper-writing/three-paper-program.md` (workspace root) for the long-form rationale behind
the paper structure. That document remains the argument; **this file is the current state.**
Where they disagree, this file wins (see "What changed" at the bottom).

---

## 0. The paper count is provisional — read this first

**We do not yet know whether this is one, two, or three papers.** The three-paper structure is a
working hypothesis, not a commitment. It could collapse to two if the splits and the escape work
turn out to share a thesis, or if either lacks the results to stand alone. That is a decision to
make *after* the measurements exist, not before.

Two consequences, and they are the reason for this restructure:

- **Everything lives in `01-lya_halos` for now.** No folder-per-paper. The measurement machinery
  (`measure_*`, `fitting.py`, `background.py`, the three-zone decomposition) is shared across all
  three candidate papers, and splitting the folder before the papers are real just creates stale
  cross-references — which is exactly what happened with the retired `03-lya_escape`.
- **Paper labels in this file are addresses, not contracts.** They exist so a given measurement has
  one obvious home in conversation. When the structure firms up, this section gets deleted and the
  labels become real.

---

## 1. The program in one paragraph

VDFI answers one question: **where does the Lyα light around a cosmic-noon galaxy come from?**
Three capabilities attack it, and no other dataset combines them. **Paper I** establishes that the
light has a boundary at the virial radius and a different kinematic character on each side,
measured in an **absolute** velocity frame. **Paper II** asks whether the light outside that
boundary responds to the galaxy or to its environment. **Paper III** asks whether the galaxy could
have supplied the light at all, and — under the framework adopted 2026-07-30 — whether the missing
photons were **destroyed or merely relocated**. II and III test the same conclusion by independent
routes (environmental correlation vs. energy conservation); if they agree, the combined claim is far
stronger than either alone.

| | The instrument nobody else has |
|---|---|
| **Paper I** | Independent systemic redshifts (MOSDEF) on a deep IFU stack → an **absolute** velocity frame |
| **Paper II** | Per-galaxy core Lyα spectra **and** CANDELS-depth multiwavelength properties on the same objects |
| **Paper III** | Balmer-line intrinsic photon budget, JWST morphology, **and** a spatially resolved halo on the same objects |

---

## 2. Shared dataset

### Primary sample

- MOSDEF galaxies, magnitude-cut roughly H<24.5 (z~2 bin) and H<25 (z~3 bin) — precise cuts
  TBD/to confirm.
- ~1500 MOSDEF galaxies total across all MOSDEF fields; **~510 in the COSMOS/EGS footprint we have
  VDFI coverage for (278 COSMOS + 244 EGS), and ~450 after the AGN cut.** The AGN-excluded ~450 is
  the number that matters — it is the sample the fits, the R_vir convention, and the split power
  analysis are all computed on. Working numbers, not final.
- **AGN cut: X-ray + BPT + IRAC combined**, removing ~70 of ~510. This is load-bearing, not
  hygiene: **the project's R_vir convention is AGN-excluded**, AGN-inflated stellar masses were
  pushing galaxies past the SMHM turnover (34 → 14 unsolvable galaxies under behroozi19 once the
  cut was applied), and residual undetected AGN would brighten the inner bins. State the
  completeness of the cut in Methods rather than just its existence.
  Reference catalogs — *verify exact versions before citing*:
  - **X-ray:** Chandra. COSMOS and AEGIS use different surveys (COSMOS Legacy vs. AEGIS-X Deep);
    confirm which release each field's cut was run against.
  - **IRAC colour selection:** Donley et al. 2012 wedge (`docs/literature/`).
  - **BPT:** Melendez et al. 2014 / MOSDEF's own AGN identification, Azadi et al. 2017.
- Precise rest-frame optical **systemic** redshifts from MOSDEF (nebular, Hα or [O III] — **not**
  circular with Lyα), reportedly ~60 km/s precision (unverified — came from one source, needs a real
  citation check before it goes in the paper). See §7: a *random* per-galaxy error broadens the
  stacked line but does not shift it; only a *systematic* offset moves the centroid.

### Ancillary catalogs

- **3D-HST photometry (Skelton et al. 2014).** The CANDELS-depth multiwavelength base — SED-derived
  stellar masses, dust, and photometric redshifts — and the parent catalog MOSDEF was selected from.
  Note the ID caveat already logged in `research-notes.md`: Chartab's `ID` is the CANDELS
  photometric ID (COSMOS→Nayyeri+2017, EGS→Stefanon+2017), *not* the 3D-HST/Skelton ID MOSDEF uses,
  so cross-matching is positional (~0.3–0.5″), never by ID.
- **3D-HST / CANDELS GALFIT structural parameters (van der Wel et al. 2014).** Sérsic index,
  effective radius r_e, axis ratio, position angle. Feeds the **size split** (Paper II #11), the
  **orientation/inclination split** (#8), and — importantly — is the **HST fallback for the r_e that
  Greg's framework wants from JWST**, on a footprint that covers the whole sample rather than the
  JWST overlap. Given the κ·r_e PSF-degeneracy question in §6, having a full-sample r_e from HST is
  worth more than a partial-footprint one from JWST for the *split*; JWST's advantage is resolution
  for the forward model, not coverage.
- **3D-HST [O III] emitters — potential future sample extension.** Not currently used and not
  scoped. Flagged here because it is the parent population of both external anchors: Lujan Niemeyer
  et al. 2022 (the Tier 1 replication check) and Weiss et al. 2021 (the f_esc technique validation)
  are both [O III]-selected 3D-HST samples. If the sample ever needs extending beyond MOSDEF, this
  is the obvious direction — **with the caveat already worked out for the Bowman/Hyperion
  assessment** (`research-notes.md`): a strong-line [O III] cut selects low-mass, high-sSFR,
  low-metallicity galaxies, which makes it a good *split* (EELGs vs. normal SFGs) and a bad *filler*
  for the mean profile. Anything velocity-referenced stays MOSDEF-only.
- **JWST rest-frame optical morphology** (COSMOS-Web / PRIMER / CEERS) for intrinsic r_e and
  SED-derived SFH phase. *[Added 2026-07-30 with the Greg framework — a genuinely new data axis for
  the program, not previously in any goals doc. `1_Measure_COSMOS-Web.ipynb` exists; the cross-match
  footprint and completeness against the MOSDEF sample have not been characterized, and that
  characterization gates how much of Greg's E1/E3 is actually runnable.]*
- **Environment: Chartab et al. 2020 CANDELS density catalog**, via `multicat.py`. Both fields,
  positional cross-match, photo-z-PDF-weighted KDE.

**Why this dataset matters:** the systemic redshift is the program's central methodological advantage
over every serious competitor (Niemeyer 2022, Davis 2023, McKay 2026, Cooper 2026 — all VIRUS/HETDEX,
none with independent rest-optical systemic redshifts). Guo et al. 2024 measure Lyα centroid shifts
*relative to the Lyα line itself* (self-referenced); VDFI can state where the gas actually sits in
absolute velocity. That is not a marginal improvement — it changes what kind of statement the
measurement can make.

---

## 3. Glossary — one vocabulary, two notations

The **three-zone** language is primary throughout the project, because it carries the R_vir
transition that Paper I establishes. Greg's framework notation maps onto it as aliases; use the
three-zone terms in code, specs, and figures, and define Greg's once where the framework is
discussed.

| Three-zone (primary) | Greg's notation | Implemented as |
|---|---|---|
| Core zone | compact, F_comp | `measure_psf_corrected_core_onehalo_luminosity` |
| One-halo zone (core → R_vir) | part of the halo term | `measure_onehalo_luminosity` |
| Two-halo zone (beyond R_vir) | part of the halo term | `measure_twohalo_luminosity` |
| Core / total flux ratio | **R_comp** = F_comp/F_tot | `measure_three_zone_ratios` |
| 1 − (core/total) | **HF** (halo fraction) = 1 − R_comp | " |
| Fitted scale length h1 | r_halo | `fitting.py` (expcore) |

**Note the one real mismatch:** Greg's HF is a *single* halo term, while the three-zone scheme
splits the halo at R_vir into one-halo and two-halo. That split is Paper I's headline result, so we
keep it — meaning HF is a coarser quantity than what we actually measure, not a different one.
When comparing to Greg's framework or to literature that uses a single halo fraction, state the
mapping explicitly.

---

## 4. Paper I — the virial-radius transition, in an absolute frame

### Overarching goal

Establish that Lyα emission around cosmic-noon galaxies has a physical transition at the virial
radius — a different kinematic and profile-shape character inside vs. outside R_vir — measured in an
absolute velocity frame anchored to MOSDEF systemic redshifts. **The headline is the transition, not
the kinematics alone.** The kinematic turnover is corroborating evidence for a boundary established
by the two-component profile fit, not a standalone result.

### Core measurements

1. **Centroid velocity vs radius** — the primary result. Radial bins in units of R/Rvir
   (stellar-mass → halo-mass → virial-radius mapping, currently Girelli+2020 — switched from
   Moster+2013 on 2026-07-27).
2. **Light (flux) profile vs radius**, fit as a two-component (one-halo + two-halo) model with a
   fitted crossover radius. A crossover landing on an independently derived R_vir would be the first
   emission-side localization of the one-halo/two-halo boundary — the field has this in absorption
   (Rakic 2012) and in simulation (Byrohl 2021), not cleanly in emission.

   > ⚠️ **The "≈76 kpc vs. 75 kpc" numbers are stale — do not quote them.** That pairing predates
   > *two* changes: the **2026-07-22 PSF flux-convention fix** and the **2026-07-27 SMHM default
   > switch** (Moster+2013 → Girelli+2020, which moves median R_vir 77.1 → 71.3 kpc, −7.8%).
   > The result is a *refinement*, not a discrepancy — but the specific coincidence, as written,
   > rested on a ~1 kpc gap while the SMHM choice alone spans ±8–12% (≈4.5–9 kpc). **Report the
   > crossover against R_vir with the SMHM-choice systematic band attached, never as a bare
   > number**, and do not let the choice of relation be made by which one makes the coincidence look
   > best — that decision was explicitly made on other grounds (see `research-notes.md`, "SMHM
   > values and methods"). Numbers get re-quoted here only after the §7 re-run pass.
3. **Environment / overdensity split** — the two-halo interpretation rests on Byrohl 2021's
   "photons originate in other halos," which is only testable with an environment split. It belongs
   in Paper I despite being a split, because Paper I's own interpretation can't close without it.
4. **UV continuum comparison (h_Lyα / h_UV)** — the same-sample analog of Steidel et al. 2011.
   Characterizes the mean profile; not a split.
5. **z~2 vs. z~3 comparison** — doubles as the magnitude-cut systematics check, and connects to
   Guo et al. 2023's finding of no size evolution across cosmic time.

None of items 3–5 require the PSF-extracted core spectrum. That is the actual boundary with
Paper II: **Paper I measures the average galaxy (core photometry); Paper II measures how the
population varies around it (core spectroscopy).**

### The Leclercq/Guo tension — a discussion target

Leclercq et al. 2020 report the Lyα line reddening at larger halo radii; Guo et al. 2024 report a
red→blue trend outward. VDFI's absolute frame can discriminate among the candidate reconciliations
(reference-frame artifact, radial-range difference either side of the transition, IGM blue-wing
attenuation, sample selection) in a way neither original study could. The most likely reconciliation
— that the line reddens inside the one-halo zone and turns blue beyond R_vir — would make Paper I's
own transition the explanation for a standing disagreement in the literature.

### Definition of done (Paper I)

- A validated centroid-vs-radius profile *and* a validated two-component flux-profile fit with a
  localized crossover radius, backed by the systematics suite in §7 (correlated-noise closure and
  PSF-width systematic first), plus the environment split, UV comparison, and z~2/z~3 check.
- A literature comparison, with the Leclercq/Guo tension addressed explicitly, and the Niemeyer 2022
  external-replication check in the main text (not an appendix).
- Pipeline choices solidified and defensible — able to justify *why* a given masking percentile or
  background method was chosen, not just that it was. Prerequisite for the Methods section.
- The paper itself written (tracked in `TODO.md`). Framed as the survey-and-transition paper — the
  mandatory citation for anything downstream, so the Data section is not a place to trim.

---

## 5. Paper II — what the halo responds to

### The thesis

Not "here are fourteen subsamples." One pre-registered, falsifiable prediction that all fourteen
splits jointly test:

> **The one-halo term tracks the galaxy. The two-halo term tracks the environment.**

That is Byrohl 2021's physical picture converted into an observational test. Under this thesis every
split becomes evidence for or against one claim, null results are as informative as detections, and
the table stops being the paper and becomes the evidence for it.

It also resolves a live tension: **Steidel et al. 2011 found halo *size* does not correlate with
overdensity**, while Byrohl predicts the outer term *should* track environment. If h1 and core
luminosity move with mass/SFR/EW/size while ignoring environment, and the two-halo amplitude does the
reverse, that confirms Byrohl from the emission side and explains Steidel's null in one stroke.

### Scope and status

- Split list and directional predictions: `subsample_splits.md`. **That file still carries the old
  two-paper labels and needs reconciling** — it is on the housekeeping list, not done.
- Requires the **core extraction + classification + measure pass** and the **split catalog (Layer A)**
  — one row per galaxy, one column per splittable quantity, cached and signature-tracked. See
  `docs/workstream-roadmap.md` §3. This is the prerequisite for both the splits and the photon-budget
  work.
- **Power:** well-powered for scale lengths, core kinematics, and all flux/luminosity quantities
  including the outer zone. The single tight measurement is the **outer-zone centroid**
  (σ_difference ≈ 56 km/s against a full-stack signal of ~47 km/s), which improves directly with the
  incoming ~2× data. Split outer bins should be treated as capping around 2.0σ.
  *(The h1 = 16.9 ± 1.1 kpc figure this power analysis was scaled from is **pre-2026-07-22 PSF fix**
  — see §7. The 2σ-needs-~26% conclusion is robust to a modest shift in the central value, but
  re-derive before it goes in a paper.)*
- **Gate:** the double-Gaussian null test. Does fitting a double-Gaussian template to pure noise and
  stacking by the fit's own derived quantities manufacture template-shaped structure? Peak-separation
  and peak-height splits, plus part of the classification scheme, sit downstream. Treat as a gate on
  scope, not a caveat in the discussion.

### Reporting format

- A **forest plot** as the signature figure: fourteen splits, Δ(one-halo) and Δ(two-halo) against the
  predicted direction. Full numeric table to the appendix.
- A **correlation matrix of the splitting parameters** — mass, SFR, ΣSFR, size, dust, EW are mutually
  correlated, so several splits select overlapping halves. Weiss et al. 2021 hit this on a nearly
  identical sample; publishing the correlation structure pre-answers the first question that
  community will ask.
- A **global sign test** for the outer zone: do fourteen pre-registered directional predictions agree
  with observation more often than chance? Eleven of fourteen is p≈0.03 from a single test. This
  converts the one genuine statistical limitation into a methodological contribution.

---

## 6. Paper III — redistribution, not loss

**Framework adopted 2026-07-30: `Greg-Initial-Driving-Model.md`.** Manuscript skeleton:
`Greg-Origins-and-Structure-Outline.md`. This section summarizes; those are the sources.

### The thesis

Lyα that doesn't escape compactly isn't destroyed — it's **relocated**. The framework is four layers:
an **engine** (young stars produce the photons), a **valve** (ISM geometry, HI column, dust — sets
*how* they leave), a **distributor** (CGM resonant scattering — sets *where* they end up), and an
**ecosystem** (inflows, outflows, environment — sets what sustains it). The core/total partition
probes the instantaneous valve state; the halo terms and scale length probe redistribution and the
ecosystem behind it.

### Why this framing and not "photon budget"

`three-paper-program.md` §8 item 7 warned that an f_esc-split framing reads as an increment on
Weiss et al. 2021, and that only a closed-budget framing survives that. **Redistribution-vs-loss is
the sharper version of the same defense** — it is falsifiable, it makes the geometry and the
throughput two halves of one measurement rather than two separate results, and it turns the
long-standing Steidel prediction (f_esc anticorrelates with halo scale length) into the natural
test rather than an add-on.

### The measurements

- **Geometry:** core/one-halo/two-halo partition (i.e. R_comp, HF) and scale length h1 (r_halo),
  from the PSF-convolved forward model. Already implemented.
- **Throughput:** **absolute** f_esc per zone, normalizing the observed luminosity by the Case-B
  intrinsic Lyα inferred from dust-corrected Balmer lines (L_Lyα,int = 8.7 L_Hα,int or
  24.9 L_Hβ,int). *[New relative to the old goals doc — the existing three-zone ratios are
  relative; this makes them absolute.]*
- **The Steidel test:** f_esc vs. halo scale length. Redistribution predicts an anticorrelation;
  destruction predicts none. **Verify the exact form of the claim in Steidel 2011 before an abstract
  is built on it.**
- **The stack-level budget, which needs no split at all:** of the Lyα a galaxy intrinsically produces,
  what fraction is recovered in the core, in the one-halo zone, beyond R_vir, and what fraction is
  genuinely destroyed? Directly comparable to the LARS/LzLCS accounting at z~0 (Hayes 2015,
  Melinder 2023). This means Paper III has a result regardless of how the f_esc correlation turns out.
- **SFH phase ordering (burst → post-burst → steady):** a falsifiable *sequence*, not a vague
  burstiness split — burst gives high R_comp / low HF, post-burst gives low R_comp / high HF /
  larger r_halo, steady gives both low. *[New: the old goals doc had "burstiness" only as an
  unstructured item.]*

### De-risked, and the timing pressure

Weiss et al. 2021 measured Lyα escape fractions for 935 [O III] emitters at 1.9<z<2.35 by stacking
HST/WFC3 grism against HETDEX IDR2 — the same measurement, essentially the same population, far worse
Lyα data. The technique is validated. That paper also supplies the number that shapes this one:
**f_esc declines from ≳16% at log M/M⊙ ≲ 9 to ≲1% at log M/M⊙ ≳ 10.** MOSDEF is H-band selected, so
this sample sits at the massive, low-escape end — meaning **95–99% of intrinsically produced Lyα is
unaccounted for**, and VDFI is the only dataset that can ask where it went with a resolved halo
attached. Weiss 2021 and Niemeyer 2022 are the same collaboration, instrument, and parent sample;
this paper is conceptually their product, and of the three it has the most timing pressure.

### Two open feasibility questions — these gate the scope

1. **Per-object vs. stack.** Greg's data table and the E2 hierarchical regressions assume one row per
   galaxy. Everything built so far is stacked, and at H ≲ 24.5 with ~2″ seeing a single-galaxy
   core+halo decomposition is very likely out of reach — Niemeyer and Weiss both stacked, on this
   population, for this reason. **Decision (2026-07-30): pursue both.** Stack-level is the
   deliverable now; per-object stays in scope as a low-signal measurement that may become real with
   the incoming ~2× data and better reductions. Report per-object as limits until it isn't.
2. **κ·r_e,conv may be PSF-degenerate.** With r_e ~ 1–4 kpc ≈ 0.12–0.5″ against a ~2″ PSF, the
   convolved compact aperture is close to the PSF itself for *every* galaxy, so R_comp may barely
   respond to r_e — which would make Greg's "r_e↓ ⇒ R_comp↑" prediction untestable with VIRUS.
   **Needs an injection–recovery check at the real seeing before that prediction is quoted.** The
   JWST-anchored version of this test is natural for a higher-resolution instrument, not for VIRUS.
   *Note the r_e source is a separate question from the degeneracy:* the van der Wel 2014 GALFIT
   catalog covers the whole sample, JWST covers a subset at better resolution. If the degeneracy is
   real, better resolution on r_e doesn't rescue it — the limit is the VIRUS PSF, not the input size.

---

## 7. Systematics — select, don't accumulate

**Listing many checks that agree is not many checks.** Most are correlated; a small set of genuinely
orthogonal ones reads as rigor where a large correlated suite reads as padding. Ordered by what each
actually rules out, and by how load-bearing it is:

- **[TOP PRIORITY] The MOSDEF ↔ VIRUS wavelength zero point.** The absolute velocity frame is the
  program's headline methodological claim and the tie between the two wavelength scales has never
  been checked. Leading hypothesis is an air/vacuum convention error: ~1.1 Å ≈ **85 km/s in the blue
  direction**, nearly flat across the VIRUS band — the right sign and roughly the right size for the
  observed large-radius blueshift. Full test plan in `specs/wavelength-zero-point.md`.
  **Bound the damage in the Methods regardless of outcome:** a rigid offset moves every bin equally,
  so it cannot produce the radial *gradient* — the transition survives, only the absolute outer-bin
  sign is at risk.
- **[CRITICAL, blocks everything] Correlated outer-bin background noise.** The noise is correlated
  rather than random. If underestimated: Paper I loses its outer-halo claims and the transition radius
  shifts, Paper II loses the two-halo half of its thesis, Paper III loses the beyond-R_vir budget
  term. **The only item load-bearing for all three papers simultaneously.**
- **[HIGH] PSF-width systematic on the crossover radius.** h1 moves ~15.1 → 12.0 kpc across 1.3″–2.4″
  FWHM — roughly 3× the statistical error — and the data only mildly prefer one width (Δχ²≈1 on
  3 dof). Report h1 with a systematic band, not a bare ±.
- **[HIGH] Re-run everything downstream of the 2026-07-22 PSF flux-convention fix and the 2026-07-27
  SMHM default switch.** Nothing needs building; no number means anything until this pass is done.
- **Tier 1 — external replication.** Niemeyer et al. 2022 on the 3D-HST subsample: same parent sample,
  independent reduction, background treatment, and analysis. **The single most valuable check in the
  suite; main text, not appendix.** Frame it as an inherited-credibility argument.
- **Tier 2 — instrumental.** Star-PSF extent (the halo is genuinely broader than the instrumental
  PSF) and the LSF check (the ±4 Å window isn't instrumentally dominated). Independent of Tier 1 and
  of each other.
- **Tier 3 — estimator.** Multiple centroid estimators agreeing. One figure.
- **Tier 4 — error estimation.** Bootstrap, jackknife, redshift-scramble are **not independent of
  each other**. Quote one as primary, note the others agree, appendix.
- **Tier 5 — field stability.** COSMOS vs. EGS directly tests whether the transition radius is
  field-stable. Cheap; worth including.
- **Say "two independent routes," not four.** Centroid and B/T are functionals of the same flux
  distribution about the same zero point. The honest count is a kinematic turnover and a
  profile-shape turnover. Make the reduction yourself in the text.

---

## 8. Open structural questions

1. **Is this two papers or three?** Undecided by design — see §0.
2. **The RT / Byrohl forward-fitting thread is unassigned.** Tracked as its own collaboration
   workstream (`docs/workstream-roadmap.md` §2), which takes the position that we should read RT to
   sharpen the question, not build a model — the shell-model parameter space is unconstrainable from
   a stacked, medium-resolution, single-line measurement (Gronke, Bull & Dijkstra 2015).
3. **`02-lae_catalog`** — still undecided: a fourth paper, or infrastructure for Paper II's spectral
   classification scheme. Unpriced.
4. **`subsample_splits.md` paper labels** are stale (old two-paper scheme). Reconciliation pending.
5. **Divide the literature review across the introductions** — three papers on one dataset will each
   want to say the same things about Lyα halos, cosmic noon, and the CGM. Decide who owns what, or
   II and III read as padding.
6. **Pin the AGN-cut catalog versions.** The cut (X-ray + BPT + IRAC, ~70 of ~510) is applied and
   load-bearing, but §2 records it by *method* rather than by *citation* — in particular which
   Chandra release each field used (COSMOS and AEGIS are different surveys). Methods needs the
   exact catalogs and the resulting completeness, and it is much cheaper to write down now than to
   reconstruct later. Referee register entry 6.9 rates residual undetected AGN as LOW risk but
   "state completeness" — that statement doesn't exist yet.
7. **Characterize the JWST cross-match** (COSMOS-Web / PRIMER / CEERS against the ~450). How many
   galaxies actually have JWST morphology, and in which field? This gates how much of Greg's E1/E3
   is runnable, and the answer may be that van der Wel 2014 GALFIT is the practical r_e source with
   JWST as a resolution cross-check on a subset.

---

## What changed on 2026-07-30

- This file went from Paper-I-only to program-level.
- **`03-lya_escape` retired.** It held no code, only framework docs, which now live in this folder.
  Escape work shares the `01` pipeline entirely.
- **`Greg-Initial-Driving-Model.md` adopted as Paper III's framework**, replacing the "photon budget"
  framing (the budget survives inside it as the throughput half).
- **`Paper2_Outline_Lya_Halo_Origins_Structure.md` → `Greg-Origins-and-Structure-Outline.md`**, and
  is **no longer Paper II** — its contents are Paper III material. The splits are Paper II.
- **JWST morphology added as a data axis** (§2) — genuinely new to this document.
- **Absolute f_esc and the SFH-phase ordering added** to Paper III (§6) — also new.
- **Three-zone vocabulary declared primary**, with Greg's R_comp/HF mapped onto it (§3).
- Two feasibility blockers written in explicitly (§6) rather than left implicit in the framework doc.

**Second pass, same day — the dataset section (§2) was rebuilt and three stale numbers flagged:**

- **Ancillary catalogs now recorded**, none of which this doc previously named: 3D-HST photometry
  (Skelton 2014), 3D-HST/CANDELS GALFIT structural parameters (van der Wel 2014) — which is the
  full-footprint fallback for the r_e Greg's framework wants from JWST — Chandra X-ray + IRAC
  (Donley 2012 wedge) + BPT for the AGN cut, and 3D-HST [O III] emitters as a flagged
  *potential-future* sample extension, not currently scoped.
- **The AGN cut is now in the dataset section at all.** It was applied, documented in
  `specs/systematics-and-extraction.md`, and absent from GOALS — despite the project's whole R_vir
  convention being AGN-excluded. Sample numbers corrected to ~510 in footprint → **~450 after the
  AGN cut**, which reconciles §2 with the N≈450 the §5 power analysis already assumed.
- **⚠️ "≈76 kpc crossover vs. R_vir = 75 kpc" flagged as stale and no longer quotable** (§4 item 2).
  It predates *both* the 2026-07-22 PSF fix and the 2026-07-27 SMHM switch, and the coincidence
  rested on a ~1 kpc gap while SMHM choice alone spans ±8–12%. This was the single most load-bearing
  stale number in the document — it is Paper I's headline.
- **h1 = 16.9 ± 1.1 kpc flagged as pre-PSF-fix** where the §5 split power analysis leans on it.

## See also

- `Greg-Initial-Driving-Model.md` — Paper III's physical framework
- `Greg-Origins-and-Structure-Outline.md` — Paper III's manuscript skeleton
- `subsample_splits.md` — Paper II's split list (paper-label reconciliation pending)
- `research-notes.md` — continually-updating log of non-headline research
- `referee-objection-register.md` — objections and their status
- `specs/` — per-analysis design docs (write one before a new stacking/measurement approach)
- `docs/paper-writing/three-paper-program.md` (root) — long-form rationale
- `docs/workstream-roadmap.md` (root) — near-term plan across systematics / RT / splits / budget
- `docs/VDFI-background.md` (root) — HETDEX/VDFI dataset context
- `TODO.md` (root) — near-term working items
- `01-lya_halos/README.md` — pipeline stages and usage
