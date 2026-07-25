# The VDFI Lyα Trilogy — Program Rationale

*Written 2026-07-24. Supersedes and merges `three-paper-plan-assessment.md` and
`three-paper-plan-second-assessment.md`, which were adversarial stress-tests of the plan. Those two
documents did their job: the objections that survived scrutiny are collected in §8 as delivery
requirements, and the ones that didn't are gone. This document is the constructive version — what
the three papers are, why three is the right number, and what each one contributes to the field.*

---

## 1. The program in one paragraph

VDFI answers one question: **where does the Lyα light around a cosmic-noon galaxy come from?**
Three papers attack it with three different capabilities that no other dataset combines. Paper I
establishes that the light has a boundary at the virial radius and a different kinematic character
on each side, measured in an absolute velocity frame. Paper II asks whether the light outside that
boundary responds to the galaxy or to its environment. Paper III asks whether the galaxy could have
supplied it at all, from an independent photon budget. Papers II and III test the same conclusion by
entirely independent routes — environmental correlation and energy conservation — and if they agree,
the combined claim about the physical origin of diffuse Lyα emission is far stronger than either
makes alone.

That is a program, not a dataset with three chapters. It should be stated as such in all three
abstracts.

---

## 2. Why three, and not one or two

### The capability argument

The cleanest test of whether a multi-paper split is legitimate: does each paper's novelty rest on a
*different* piece of the data? Here it does.

| | The instrument nobody else has |
|---|---|
| **Paper I** | Independent systemic redshifts (MOSDEF) on a deep IFU stack → an **absolute** velocity frame |
| **Paper II** | Per-galaxy core Lyα spectra **and** CANDELS-depth multiwavelength properties on the same objects |
| **Paper III** | Balmer-line intrinsic photon budget **and** a spatially resolved halo on the same objects |

Three papers deriving novelty from three different data combinations are three papers. Three papers
deriving novelty from one combination sliced three ways would be salami-slicing. This is the former.

### The boundary that makes I and II separable

The earlier two-paper plan was incoherent for a specific reason worth recording, because it explains
why this restructure was necessary rather than cosmetic. Paper I was assigned every subsample split
with a kinematic prediction — but the velocity-offset self-split, the LAE/LAA/absorber/AGN
classification, the double-Gaussian peak-separation and peak-height splits *all require the
PSF-extracted core spectrum*. The old Paper I was therefore defined as "the splits we can do without
the tool that makes splits interesting," which is not a scope, it's an accident.

The new boundary is a genuine methodological line:

- **Paper I needs core *photometry*** — the PSF-corrected core luminosity that
  `measure_psf_corrected_core_onehalo_luminosity` provides, which is load-bearing for the three-zone
  decomposition. Paper I is not "the paper without a core."
- **Paper II adds core *spectroscopy*** — per-galaxy line classification, FWHM, double-Gaussian
  decomposition, peak separation.

Equivalently, and better for abstracts: **Paper I measures the average galaxy; Paper II measures how
the population varies around it.** Mean and variance. That is a standard, immediately legible
structure.

### The precedent

Guo et al. 2024 is the apt precedent and maps almost exactly: *Lyα Halos in the MUSE Extremely Deep
Field* (profiles) followed by *Spectroscopic analysis of Lyα haloes* (spectroscopy). Profiles first,
spectroscopy second, same team, same data.

Niemeyer et al. 2022 is a **weaker** precedent than it looks — those two papers split by *sample*
(LAEs vs. 3D-HST galaxies), not methods-then-science. Cite Guo when defending the structure; cite
Niemeyer for validation (see §7), which is what it's actually good for.

### The length argument

Paper I already carries a survey description, a catalog construction, a full methods chain
(masking, background subtraction, extraction, stacking, measurement), four results sections, a
systematics suite, and a two-component profile fit. Paper II adds a core-extraction methodology plus
fourteen splits, each with its own literature prediction and its own discussion paragraph. These do
not fit in one paper. They barely fit in two. The split is not a choice about credit; it's a
constraint from page count.

---

## 3. Paper I — the virial-radius transition, in an absolute frame

### What it establishes

The headline should be **the transition**, not the kinematics. Here is why that ordering matters.

The field's current evidence for two physically distinct gas regimes around cosmic-noon galaxies:

- **In absorption:** Rakic et al. 2012 — an order-of-magnitude optical-depth drop at ~100 kpc
  (≈R_vir), with distinct redshift-space anisotropies on either side (FoG inside ~1 Mpc, Kaiser at
  1.4–2.0 Mpc). The field's cleanest "two physically distinct scales" result, but it is absorption.
- **In simulation:** Byrohl et al. 2021 (IllustrisTNG50) — outer-profile flattening in *emission*
  comes from photons originating in **other halos**, not the galaxy's own diffuse gas. A prediction.
- **In emission, observationally:** halos are detected far out (Niemeyer et al. 2022 to 320 kpc;
  Wisotzki et al. 2018 finding nearly the whole sky covered) — but a *transition* localized at R_vir
  has not been cleanly isolated.

A fitted crossover at ≈76 kpc against an independently derived R_vir = 75 kpc is **the first
emission-side localization of the one-halo/two-halo boundary.** That is the claim that enters the
introduction of every subsequent Lyα halo paper, and it is more durable than a velocity trend.

Structurally this is also the stronger rhetorical position. Leading with the transition means the
kinematic turnover becomes *corroborating evidence* for a boundary established by the profile fit,
rather than a marginal result the reader is asked to accept on its own. Same content, much better
load-bearing.

### The systemic-redshift moat

Every serious competitor is VIRUS/HETDEX — Niemeyer 2022, Davis 2023, McKay 2026, Cooper 2026 — and
none has independent rest-optical systemic redshifts across the stacked sample. Guo et al. 2024 can
only report a *relative* red→blue trend because their reference is the Lyα line itself. VDFI can
state where the gas actually sits in velocity. That is not a marginal improvement; it changes what
kind of statement the measurement can make, and it cannot be replicated by depth alone.

### The Leclercq/Guo tension — a highlight, not a hazard

There is an apparent contradiction in the literature that Paper I is uniquely positioned to
adjudicate, and it should be a featured discussion item rather than a footnote:

- **Leclercq et al. 2020** (MUSE-HUDF XIII) report the Lyα line becoming **redder** at larger halo
  radii.
- **Guo et al. 2024** report a red→blue trend outward.

These may not actually conflict, and there are at least four candidate reconciliations worth
laying out — VDFI can discriminate among them where neither original study could:

1. **Reference frame.** If the "shift" is measured relative to the source's own Lyα rather than
   systemic velocity, the apparent trend depends on what the core line looks like, which depends on
   selection. VDFI's absolute frame removes this degeneracy entirely — this is the single clearest
   demonstration of why the systemic anchor matters, and it should be presented that way.
2. **Radial range.** Leclercq's individual halos probe tens of kpc; VDFI reaches hundreds. **If the
   line reddens within the one-halo zone and turns blue beyond R_vir, both results are correct and
   were measured on opposite sides of the transition.** This is the most likely reconciliation and
   it is a beautiful one, because it makes Paper I's own transition the explanation for a standing
   disagreement in the literature.
3. **IGM blue-wing attenuation.** Leclercq's sample extends to z~6, where IGM absorption
   preferentially suppresses the blue peak and systematically reddens outer emission. VDFI at
   z~2–3 is far less affected.
4. **Selection.** LAE-selected samples pick objects with the radiative-transfer conditions (low
   N_HI, strong red peak) that set line shape, which is exactly the quantity being compared.

Resolving a live disagreement between two well-regarded groups, using a capability neither had, is
the kind of discussion section that gets a paper read.

### Splits that belong in Paper I

Three items sit more naturally in Paper I than Paper II, because Paper I's own interpretation cannot
be closed without them and none requires the core spectrum:

- **Environment / overdensity (Chartab 2020, via `multicat.py`).** Paper I's two-halo interpretation
  rests on Byrohl 2021's "photons from other halos." That is only testable with an environment
  split. Without it, the interpretation is asserted rather than tested.
- **UV continuum comparison (h_Lyα / h_UV).** The same-sample analog of Steidel et al. 2011's
  classic result — Lyα ~5–10× more extended than the UV continuum, with UV scale lengths of 3–4 kpc
  against Lyα's 20–30 kpc. This characterizes the mean profile; it is not a split at all.
- **z~2 vs. z~3.** Doubles as the magnitude-cut systematics check Paper I needs anyway, and connects
  to Guo et al. 2023's finding of no evolution in halo size across cosmic time.

---

## 4. Paper II — what the halo responds to

### The thesis

Paper II should not be titled around subsamples. It should be titled around a single
pre-registered, falsifiable prediction that all fourteen splits jointly test:

> **The one-halo term tracks the galaxy. The two-halo term tracks the environment.**

That is Byrohl 2021's physical picture, converted into an observational test. Under this thesis
every split becomes evidence for or against one claim, null results are as informative as
detections, and the table stops being the paper and becomes the evidence for the paper.

There is also a live literature tension it resolves. **Steidel et al. 2011 found halo *size* does
not correlate with overdensity.** Byrohl predicts the outer term *should* track environment. These
are reconcilable — size and outer amplitude are different quantities, and Steidel's overdensity
dynamic range was limited — but they set up a sharp prediction: h1 and core luminosity should move
with mass, SFR, EW and size while ignoring environment; the two-halo amplitude and outer luminosity
fraction should do the reverse. If that pattern separates cleanly, it confirms Byrohl from the
emission side and explains Steidel's null in one stroke. If it fails — if the two-halo term tracks
stellar mass — that is arguably more interesting, because the external-origin interpretation needs
revisiting.

### Why the splits work

Fourteen independent 50/50 median splits, ~225 galaxies per side, each with a literature-motivated
directional prediction already recorded in `subsample_splits.md`. Running the power numbers:

- **Scale length h1** (16.9 ± 1.1 kpc at N≈450): per-half σ ≈ 1.6 kpc; difference σ ≈ 2.2 kpc, so 2σ
  needs a **~26% change** — dropping to **~18%** with the incoming ~2× data. If h1 tracks R_vir even
  loosely (h ∝ M_h^⅓), a median mass split spanning ~0.5 dex in halo mass predicts ~40%. Comfortably
  powered. Simulation work already finds the stacked profile is steeper for higher-mass halos with
  both characteristic scales increasing with halo mass, so this is a prediction with a known sign.
- **Core kinematics:** per-galaxy Lyα offsets scatter at ~150 km/s; stacked at N≈225 that's ~10 km/s.
  Powered by an order of magnitude. Every core-spectroscopy split is comfortable.
- **Outer-zone flux and luminosity ratios:** measured at high S/N. Fine.
- **Outer-zone centroid:** the one genuinely tight measurement — σ_difference ≈ 56 km/s against a
  full-stack signal of ~47 km/s. This is where the incoming data and improved reductions matter
  most, and where the ensemble reporting below earns its place.

So the correct statement is: **the splits are well-powered for scale lengths, core kinematics, and
all flux/luminosity quantities including in the outer zone. The single tight measurement is the
outer-zone centroid**, which improves directly with the incoming data volume and reduction quality.

### The mass split as a contest between mechanisms

Worth framing explicitly, because it makes the result more interesting than a monotonic trend. High
mass predicts *two competing things*: a deeper potential well driving stronger large-scale accretion
(→ more blueshifted outer centroid), and more dust, stronger outflows, and higher N_HI (→ redder).
A measured blueshift in the high-mass half is therefore not just a detection — it's evidence that
**gravity wins over feedback at large radii**, which is a physical statement worth a paragraph.
The same logic applies to several other splits and is a good organizing device for the discussion:
name both mechanisms, then report which won.

### Splits with direct literature hooks

Beyond the standard mass/SFR/dust/size set, three splits test named published relations directly —
these should be highlighted because a split that adjudicates a specific published claim reads very
differently from a split that explores a parameter:

- **Core FWHM (Leclercq et al. 2020).** Leclercq report a relation between line width and halo
  properties. Splitting on core FWHM and reporting the resulting flux and centroid profiles is a
  direct test, and it pairs naturally with Paper I's treatment of the Leclercq/Guo tension —
  Paper I addresses it at the stack level, Paper II tests the underlying relation with per-galaxy
  spectra. Good demonstration that the I/II boundary is working.
- **Peak separation and peak-height ratio.** Separation traces N_HI, height ratio traces
  outflow/inflow asymmetry. Cite the actual scaling relations (Verhamme+2015, Gronke & Dijkstra
  2016, Orlitová+2018), not just the qualitative orientation papers.
- **Spectral classification (LAE / LAA / absorber / null / AGN).** Steidel et al. 2011 defined
  comparable subtypes; Trainor et al. 2025 found even Lyα absorbers are net emitters. Reproducing
  those categories on a sample with resolved halos and multiwavelength properties is a direct,
  citable comparison — and keeping "absorber" as its own category rather than a non-emitting control
  is the right call given Trainor.

### Reporting format

Two recommendations that make the paper substantially more readable and defuse the obvious
criticisms in advance:

- **A forest plot as the signature figure.** Fourteen splits, each showing Δ(one-halo quantity) and
  Δ(two-halo quantity) with error bars against the predicted direction. The eye immediately sees
  whether the one-halo/two-halo pattern separates. Keep the full numeric table as an appendix and
  machine-readable table — it will be the thing people actually reuse, but it should not be what the
  reader has to parse to get the point.
- **A correlation matrix of the splitting parameters.** Mass, SFR, ΣSFR, size, dust, and EW are
  mutually correlated by the main sequence and the mass–size relation, so several splits will select
  overlapping halves. Weiss et al. 2021 hit exactly this on a nearly identical sample and said so
  plainly — f_esc appeared to correlate with size, SFR, and excitation, but *"most of these
  relations are indirect,"* with mass and extinction the primary drivers. Publishing the correlation
  structure, and reporting splits at fixed mass where possible (as `subsample_splits.md` already
  specifies for several), pre-answers the first question that community will ask.
- **A global sign test for the outer zone.** Fourteen pre-registered directional predictions is
  textbook setup: do the observed directions agree with prediction more often than chance? Eleven of
  fourteen is p≈0.03 from a single test. This is the right way to report two-halo-zone results where
  individual splits are tight, and it converts the one genuine statistical limitation into a
  methodological contribution.

### Why this paper will be cited

Every existing "how do halo properties depend on galaxy properties" measurement comes from a sample
that is Lyα-selected or continuum-selected with thin ancillary data — Steidel 2011 (LBG,
continuum-based), Momose 2014 (narrowband LAE, ~5–6 kpc scale lengths), Leclercq 2017 (MUSE LAE,
~4.5 kpc), Trainor 2025 (KBSS, small sample, good ancillary). Nobody has a large sample with
*simultaneously* a deep IFU halo measurement, per-galaxy core spectra, and CANDELS-depth
multiwavelength characterization from 3D-HST and MOSDEF.

That gap is the paper's justification, and the consequence is that Paper II likely becomes the
**most-cited of the three** — anyone building or fitting a Lyα halo model needs numbers to compare
against, and this becomes the comparison set. That is Momose 2014's citation profile, and Momose
2014 is cited constantly.

---

## 5. Paper III — the photon budget, and a fifteen-year-old prediction

### The measurement is already de-risked

Weiss et al. 2021 (ApJ 912, 100) measured Lyα escape fractions for 935 [O III] emitters at
1.9 < z < 2.35 by stacking HST/WFC3 grism against HETDEX IDR2 and taking the Hβ/Lyα ratio — the same
measurement, on essentially the same 3D-HST population, with far worse Lyα data. With MOSDEF Hα and
3D-HST SED-derived dust on top, the photon-budget side of Paper III is not a research risk. It is a
validated technique on a validated sample.

That paper also supplies a hard number that shapes the whole paper: **f_esc declines from ≳16% at
log M/M_⊙ ≲ 9 to ≲1% at log M/M_⊙ ≳ 10.** MOSDEF is H-band selected at H ≲ 24.5, so this sample sits
at the massive, low-escape end. Which means **95–99% of intrinsically produced Lyα is unaccounted
for** — and VDFI is the only dataset that can ask where it went with a resolved halo attached.

### The headline: Steidel's prediction, finally testable

The physical question — *are the missing photons destroyed, or relocated?* — has a specific
predicted signature, and it is one you have already flagged from Steidel et al. 2011: **escape
fraction should be inversely correlated with halo scale length.** The logic is clean. Low f_esc
means photons scatter more before escaping, which means they random-walk further out, which means a
larger halo. Redistribution predicts an anticorrelation; destruction predicts none.

*(Worth verifying the exact form of the claim in Steidel 2011 before building on it — the physical
argument stands regardless, but the citation should be precise.)*

That makes Paper III a **direct test of a long-standing prediction with the first dataset capable of
performing it**, rather than an exploratory split. That is a much stronger framing and it should be
the abstract's centerpiece.

### The accompanying stack-level result

Pair it with a closed budget that needs no split at all, using the three-zone decomposition already
implemented (`measure_three_zone_ratios`, `measure_psf_corrected_core_onehalo_luminosity`):

> Of the Lyα photons a galaxy intrinsically produces, what fraction is recovered in the core, what
> fraction in the one-halo zone out to R_vir, what fraction beyond R_vir, and what fraction is
> genuinely destroyed?

This uses the entire sample, is directly comparable to the LARS/LzLCS accounting at z~0 (Hayes 2015,
Melinder 2023), and folds in the star-formation ceiling test already scoped in
`Lya-maximum-emission-summary.md`. "The cosmic-noon counterpart to the LARS photon budget, with
200 kpc of spatial information LARS never had" is a paper on its own, and it means Paper III has a
result regardless of how the f_esc correlation turns out.

### How it closes the loop

If the core-plus-halo emission exceeds what the galaxy's own star formation can power, the excess
must come from elsewhere — cooling, fluorescence, or undetected neighbors. **That is the same
conclusion Paper II reaches from environmental correlation, arrived at by energy conservation
instead.** Two independent routes to one claim about the origin of diffuse Lyα is the strongest
thing this program produces, and it is the reason the trilogy is more than the sum of its parts.

### Positioning note

Weiss 2021 (f_esc from HETDEX+3D-HST) and Niemeyer 2022 (halos from HETDEX+3D-HST) are the same
collaboration, instrument, and parent sample. Paper III is, in a sense, the product of those two
papers. That it hasn't been written is plausibly because HETDEX's per-object depth doesn't support a
zone-by-zone luminosity decomposition — VDFI's ~100–150 exposures per galaxy is the moat. It's a
real moat, but a conceptually obvious next step for a group that owns both halves. Of the three,
this is the one with the most timing pressure.

---

## 6. Sequencing

Paper II cannot cite Paper I's methods until Paper I is at least submitted, or it has to re-describe
them and the split's rationale erodes. Standard practice: submit I, draft II during review, cite as
"Paper I, submitted."

This makes **Paper I's timeline the gate for the entire program**, and Paper I's numbers all need
regenerating after the 2026-07-22 PSF flux-convention fix. That re-run is the critical path for
three papers, not one.

On the incoming ~2× data: it argues for getting Paper I out on DR1 rather than waiting. The moat is
the systemic redshifts, not the depth, and Paper I framed as the survey-and-transition anchor is not
embarrassed by its own sequels using more data — that's what a Paper I is for. The relationship with
VDFI's originator makes this cleaner still: Paper I can be written as the canonical survey
description with the originator's blessing, which makes it the mandatory citation for every future
VDFI paper by anyone. That's a durable citation base and a good reason not to trim the Data section.

---

## 7. Systematics strategy — select, don't accumulate

The instinct here is right and worth formalizing: **listing many systematics checks that agree is
not many checks.** Most are correlated, and a suite of correlated confirmations reads as padding to
a careful referee while a small set of genuinely orthogonal ones reads as rigor. Tier them by what
each actually rules out:

**Tier 1 — external replication (strongest).** The Niemeyer et al. 2022 comparison on the 3D-HST
subsample. Same parent sample, independent reduction, independent background treatment, independent
analysis, same instrument. If VDFI's curve tracks hers with smaller error bars, then anyone who
trusts Niemeyer must trust VDFI, and anyone who doesn't has a problem that predates this paper.
**This is the single most valuable check in the suite and belongs in the main text, not an
appendix.** Frame it exactly that way — as an inherited-credibility argument, not just an agreement
plot.

**Tier 2 — instrumental.** Star-PSF extent comparison (the halo is genuinely broader than the
instrumental PSF) and the LSF simulation (the ±4 Å window isn't instrumentally dominated). These
rule out the instrument manufacturing the signal. Independent of Tier 1 and of each other.

**Tier 3 — estimator.** Multiple centroid estimators (flux-weighted moment, Gaussian-weighted
moment, Gaussian fit, median-flux) agreeing rules out estimator choice. One figure.

**Tier 4 — error estimation.** Bootstrap, jackknife, and redshift-scramble. These are **not
independent of each other** — they are three ways of asking the same question about the same
resampling structure. Quote one as primary, note the others agree, appendix. Presenting three
agreeing error methods as three checks invites the same reduction a referee will apply to
"centroid and B/T both flip at R_vir" (see §8).

**Tier 5 — field stability.** A COSMOS vs. EGS split directly tests whether the transition radius is
field-stable, which matters given the correlated-noise question. Cheap and worth including.

---

## 8. Delivery requirements — what this is contingent on

These are the surviving objections from the adversarial passes. Each is a thing a referee will
reach for, and each has a known answer that needs to be executed rather than argued.

### Critical — the whole program is load-bearing on this

**1. The correlated outer-bin background noise must be closed.** Outer-bin backgrounds are
correlated rather than random and this is still open. If those errors are underestimated: Paper I
loses its outer-halo claims and its transition radius shifts, Paper II loses the two-halo half of
its central thesis, and Paper III loses the beyond-R_vir term in its budget. This is **the only item
in the project that is load-bearing for all three papers simultaneously**, which makes it the
highest-value single task in the program — higher than any of the science, because all of the
science sits downstream of it.

### High priority

**2. Quantify the PSF-width systematic on the crossover radius.** h moves ~15.1 → 12.0 kpc across
1.3″–2.4″ FWHM — roughly 3× the statistical error — and the data only mildly prefer one width
(Δχ²≈1 on 3 dof). The crossover radius is Paper I's headline and its PSF sensitivity is currently
unquantified. `psf_forensics.py` can already sweep this. It may well prove stable, since a crossover
is set by where two terms cross rather than by either scale individually — but that needs
demonstrating, not assuming. Report h1 as a value with a systematic band, not a bare ±1.1.

**3. Say "two independent routes," not four.** Centroid and B/T are functionals of the same flux
distribution about the same zero point — B/T crossing 0.5 where the centroid crosses zero is close
to a tautology unless the *line shape* changes independently, which is a separate claim requiring
separate demonstration. Likewise the flux bend and the fitted crossover come from the same fit. The
honest count is two independent routes: a kinematic turnover and a profile-shape turnover. Two
independent observables agreeing at one radius is still the paper's best argument. Making this
reduction yourself, in the text, reads as rigor; having it made in report does not.

**4. Run the double-Gaussian null test before scoping Paper II.** Does fitting a double-Gaussian
template to pure noise and stacking by the fit's own derived quantities manufacture template-shaped
structure? Splits on peak separation and height ratio, plus part of the classification scheme, sit
downstream. Treat it as a gate on scope, not a caveat in the discussion.

**5. Re-run everything downstream of the 2026-07-22 PSF fix.** h1, crossover radii, zone
luminosities, three-zone ratios, and all subsample splits. Nothing needs building — the pipeline is
correct now — but no number means anything until this pass is done, and it gates all three papers.

### Framing requirements

**6. Paper II must have a thesis, not a table.** The one-halo/two-halo separation prediction is the
thesis and it is already latent in the discussion notes. A paper organized as "here are fourteen
splits" gets read as an appendix; the same content organized as "here is a prediction and here are
fourteen tests of it" gets read as a result. This is entirely a writing decision and it is the
highest-leverage one in the plan.

**7. Paper III must be a budget, not a split.** Framed as an f_esc subsample split it reads as an
increment on Weiss 2021 and a referee from that community will say so. Framed as a closed photon
budget testing Steidel's f_esc–scale-length prediction, it's a new measurement. There is no middle
outcome. Check the f_esc *distribution* early — not the sample count, which Weiss settled — because
if dynamic range is compressed at the massive end, the budget framing becomes mandatory rather than
preferred.

**8. Verify the Steidel 2011 f_esc–scale-length claim before building Paper III's abstract on it.**
The physical argument is sound regardless, but if the prediction is being attributed to a specific
paper it needs to be the right attribution.

**9. Publish the splitting-parameter correlation matrix.** Cheap, one small figure, and it
pre-answers the Weiss 2021 critique — that apparent correlations across many galaxy properties are
mostly indirect — before it gets raised.

### Housekeeping

**10. Reconcile `subsample_splits.md` with the new plan.** It still carries the old two-paper
assignment (kinematic splits → Paper 1, flux splits → Paper 2), which the restructure supersedes.
Burstiness and classification should live wherever the core spectrum lives; Paper III keeps only
what requires Balmer-line photon accounting.

**11. Decide what `03-lae_catalog` is.** It's a scaffolded subproject with no slot in this program —
either a fourth paper, or infrastructure for Paper II's classification scheme. Right now it's an
unpriced item on the schedule.

**12. Divide the literature review across the three introductions.** Three papers on one dataset
will each want to say the same things about Lyα halos, cosmic noon, and the CGM. Decide now which
paper owns which piece, or papers II and III read as padding.

---

## Sources

- Weiss et al. 2021, *The HETDEX Survey: The Lyα Escape Fraction from 3D-HST Emission-Line Galaxies
  at z ~ 2*, ApJ 912, 100 — https://ui.adsabs.harvard.edu/abs/2021ApJ...912..100W
- Guo et al. 2023, *On the evolution of the size of Lyman alpha haloes across cosmic time*,
  MNRAS 522, 4275 — https://academic.oup.com/mnras/article/522/3/4275/7146216
- Steidel et al. 2011, *Diffuse Lyα emitting halos* — https://arxiv.org/pdf/1010.3017
- Rakic et al. 2012, Byrohl et al. 2021, Niemeyer et al. 2022, Guo et al. 2024, Leclercq et al.
  2017 & 2020, Momose et al. 2014, Trainor et al. 2025, Wisotzki et al. 2018, Hayes et al. 2015,
  Melinder et al. 2023, Verhamme et al. 2015, Chartab et al. 2020 — local copies in
  `docs/literature/`
