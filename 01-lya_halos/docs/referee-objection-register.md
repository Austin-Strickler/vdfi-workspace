# Referee objection register — Paper I (Lyα halos)

**Status: v1, 2026-07-28.** Built from an adversarial review session whose explicit
goal was to null the result. Every entry below was written to be *believed*, then
attacked, then either closed by evidence or kept.

## What this document is for

The systematics work is spread across `specs/systematics-and-extraction.md`,
`specs/background-diagnostics.md`, `specs/bin-correlation-diagnostics.md`,
`research-notes.md`, and a long tail of notebook sessions. That is the right place
for *how* each test works. It is the wrong place to answer the question this
document exists for:

> If a referee sets out to prove this result is an artifact, what are their best
> shots, which ones have already been answered, and what number answers them?

Each entry is: **the objection**, **why it is plausible** (stated as strongly as
possible — an objection written weakly cannot be closed), **status**, and **the
test or number that settles it**. Several entries are closed by work already done
where the *value was never recorded*; those are marked BOUNDED and the deliverable
is a number in Methods, not new analysis.

### Status vocabulary

| Status | Meaning |
|---|---|
| **CLOSED** | Answered by evidence in hand. Not expected to reopen. |
| **BOUNDED** | Mechanism is real but measured to be small. Needs a number quoted, not new work. |
| **OPEN** | Live. Needs a test that has not been run. |
| **RETIRED** | Raised, examined, and shown not to apply. Recorded so it does not come back. |

---

## 1. CLOSED

### 1.1 "You chose the estimator that produced a signal"

**The objection.** Production combines with a biweight. The unweighted mean shows
a null in the intermediate bins (60–180 kpc) and a flat plateau in the outer bins
where the biweight shows a declining profile. A result that appears under a robust
estimator and vanishes under a mean is the classic signature of estimator shopping,
and no amount of downstream validation repairs it. This was the single most
dangerous objection in the review.

**Status: CLOSED, 2026-07-28.**

**What settles it.** A **3σ-clipped mean** (`stack_galaxies(method="sigma_clip")`
— astropy `sigma_clip`, median centre, `mad_std` scale, mean of survivors)
reproduces the biweight result *essentially exactly*: same flux profile, same
centroid trend, same everything. A clipped mean is still a mean. Therefore the
biweight/mean discrepancy is **outlier contamination of the unweighted mean**, not
a property of the estimator family, and the result is reproducible with a
mean-based statistic.

**What to put in the paper.** State that the profile is reproduced by a σ-clipped
mean and quote both. This converts the most dangerous line of attack into a
strength. Do not bury it in an appendix — a referee who finds the mean/biweight
difference on their own will treat it very differently than one who is handed the
resolution.

**Machinery note.** `galaxy_combine_methods` computes all six galaxy-axis
statistics (`biweight`, `inv_var`, `mean`, `median`, `sigma_clip`,
`weighted_median`) in a single Stage-2 pass, because the expensive step is the
rest-frame resample and the combines themselves are nearly free. Every Stage-2
product on disk already carries all six. This is deliberate and is why the
estimator question could be answered without re-running anything.

### 1.2 Instrumental systematics out to ~800 kpc

**The objection.** The comparison to Lujan Niemeyer et al. is not an independent
check, because it is the same telescope and the same instrument (VIRUS on HET).
A shared instrumental systematic — fiber-profile wings, scattered light, sky
subtraction — would produce agreement without either measurement being right.

**Status: CLOSED for instrumental systematics, out to ~800 kpc.**

**What settles it.** LN22 Paper II reaches **800 kpc** with [O III]-selected
galaxies. When the same sample is measured through this pipeline (excluding
GOODS-N, which VDFI does not cover), the resulting curve agrees. Critically the
two measurements differ in **fiber format**, **data reduction**, and **background
subtraction** — VDFI is a full-fill-factor mosaic of IFUs, not the HETDEX
observing mode. The only element genuinely common to both is the instrument's
optical behaviour, and that is precisely what the independent empirical stellar
PSF constrains.

**The residual, stated honestly.** Beyond ~800 kpc there is no external check
from anyone, because nobody has published a resolved Lyα profile there. That is
not a weakness of this analysis; it is the reason the measurement is worth making.
It should be labelled as unvalidated territory in the text rather than left for a
referee to point out.

### 1.3 Line-spread function and redshift precision

**The objection.** The COSMOS double peak and the EGS single peak may be a
resolution difference rather than a column-density difference. A 400–500 km/s
peak separation is 5.5–6.7 Å observed at 4000 Å, comparable to the VIRUS LSF,
so redshift jitter or a wavelength offset between the fields could wash out a
double peak without any change in physics.

**Status: CLOSED.**

**What settles it.** The **combined kernel** — instrumental LSF convolved with
MOSDEF redshift precision, shifted to rest frame and measured over the actual
redshift mixture of the sample — is **~1.9 Å rest**. The observed peak widths in
the core spectrum match that kernel, i.e. the peaks are *resolved features*, not
noise structure. Even at a pessimistic 2 Å the separation is resolved. Redshift
jitter is already folded into the measured number, so it cannot be invoked
separately.

### 1.4 Can a robust estimator manufacture a line from noise?

**Status: CLOSED**, and worth stating explicitly because it is a question referees
ask in the abstract rather than testing.

The `redshift_scramble_null` and `placebo_wavelengths` tests run through the
**identical** estimator chain — same fibers, same fiber-level combine, same
galaxy-level combine, same rejection — and return flat nulls. If the combine
could conjure a line profile out of noise, those tests would show it. Label those
figures as estimator tests as well as detection tests; that is the work they are
doing.

The known biases all run the *wrong* way for manufacturing signal: a positively
skewed contaminated distribution has median *below* mean, and biweight's 6·MAD
rejection scale down-weights genuinely bright galaxies. Both suppress signal.

---

## 2. BOUNDED — the work is done, the number is not recorded

These are real mechanisms, already measured to be small. The deliverable is a
quantity in Methods, not new analysis. "We checked and it was fine" is not a
defence a referee can evaluate; a number is.

### 2.1 Background-annulus Lyα leakage → additive offset

**The mechanism.** The background annulus (80–90″ ≈ 660–743 kpc at z = 2.3) sits
*inside* the detected signal range, so it unavoidably contains real Lyα. The Lyα
mask exists so the line is not subtracted, but any flux leaking past the mask is
smoothed into the pedestal and subtracted identically from every science bin —
a constant $C$, coherent in the rest frame because it tracks each galaxy's
redshift by construction, and therefore surviving the stack at full strength.

**Why it matters more than its size suggests.** Subtracting a constant from a
declining profile steepens the measured log-slope by a factor $S/(S-C)$, which
grows without bound as $S \to C$. A small $C$ is negligible at 100 kpc and
decisive at 2 Mpc. This is a *shape* systematic, not a detection systematic —
note that $C > 0$ biases the profile **down**, so it cannot manufacture a
detection, only truncate one.

**Why it is bounded.** `line_mask_halfwidth_A = 10.0` (observed) ≈ 6.6 Å rest at
the sample's redshifts, which fully covers the line. Widening the mask further
changes nothing beyond noise; 5 Å is mildly biased, 2.5 Å still retains the
signal. The mask fully covering the line is what bounds the leak.

**Numbers to quote in Methods:**

- The change in the outer profile between mask = 10 Å and mask = 20 Å. This is a
  direct empirical upper limit on $C$ and it is what licenses the outermost bins.
- The outer profile under all four `bg_variants` — (80,90), (60,70), (40,50),
  (110,130) arcsec. $C$ scales with the Lyα surface brightness at the annulus
  radius, so it should fall substantially from 40–50″ to 110–130″. "The outer
  profile is unchanged across four annuli spanning 330–1073 kpc" is a far stronger
  sentence than any argument for why it should be.

**Optional strengthening.** If the variants show any residual trend, fit the
profile with a free additive constant and marginalize over it — standard practice
in galaxy–galaxy lensing for exactly this problem. Don't pick an annulus; solve
for the offset.

### 2.2 Point-spread function wings

**The mechanism.** The empirical stellar PSF is only measured out to the radius
where a stellar profile stays above sky. Beyond that, "the PSF is too small to do
this" is an extrapolation, and the atmospheric aureole plus instrumental scattered
light have shallow far-wing power.

**Why it is bounded.** Three independent constraints, and they are strong:

1. The empirical stellar PSF is measured, not modelled, and falls ~4 orders of
   magnitude before becoming noise-limited.
2. The continuum absorption features seen in the core bins are **not** extended,
   which directly limits scattered light.
3. LN22 Paper II agreement out to 800 kpc on a different fiber format.

**Numbers to quote:** the radius to which the empirical PSF is actually measured,
the upper limit on wing power beyond it, and that limit converted into a maximum
contribution to the 600–1000 kpc bin. Same for the extended-continuum
non-detection — turn it into a quantitative exclusion rather than a qualitative
statement.

### 2.3 Inter-bin correlation and combined significance

**The mechanism.** Outer bins correlate at ρ ≈ 0.3–0.4. A coherent component
carrying ~√0.35 ≈ 60% of the per-bin noise amplitude caps the achievable combined
significance at roughly $1/\sqrt{\rho} \approx 1.7\times$ the single-bin S/N, no
matter how many bins are stacked.

**Status: acknowledged, needs to propagate.** Any quoted combined outer-bin
significance must use the full covariance from `compute_bin_correlation` /
`combined_bin_significance`, not an independent-bin combination. If it already
does, say so explicitly.

**Note the blind spot.** A correlation matrix measures how bins' *fluctuations*
co-vary. It is structurally incapable of detecting a **constant additive offset**
common to all outer bins, which is exactly what a spurious extended detection
would look like. Only §2.1 (annulus variants) and §3.4 (random-position null) can
see that. Do not cite the correlation matrices as evidence against an additive
floor.

### 2.4 Effective sample size in the outer bins

**The mechanism.** At 1–2 Mpc (≈120–240″) essentially every galaxy's annulus
overlaps every other galaxy's in the same field, so the same physical fibers enter
many galaxies' outer bins. Galaxy-bootstrap resampling does not reduce the
variance of a component common to the field, and leave-one-galaxy-out jackknife
shares the identical blind spot — which is why the two agree.

**Why it is bounded (and this is largely conceded).** Redshift diversity does real
work here: a fixed observed-frame residual lands at different rest wavelengths for
different galaxies, so resampling galaxies genuinely resamples that systematic.
And the arithmetic is reassuring — at z ≈ 2–2.6 over a CANDELS-sized field, a
±1000 km/s slice is Δz ≈ 0.01, so typical overlap groups are 2–5 galaxies, not 20.

**The residual.** The correction is modest on average but dominated by the tail: a
genuine protocluster puts many galaxies in a single (position, redshift) cell. One
**delete-one-structure jackknife** — group galaxies by position overlap *and*
Δv ≲ 1000 km/s, then delete groups — bounds it. Worth doing once; not worth
restructuring the error model around.

---

## 3. OPEN — needs a test that has not been run

Ranked by how much trouble each can still cause.

### 3.1 Line width versus radius in the outermost bins — **highest priority**

**The objection.** In the mean stack, the 1000–2000 and 2000–3000 kpc panels do
not show a Lyα line. They show a smooth arc spanning several thousand km/s, on a
strongly tilted baseline (near zero on the blue side, collapsing to −2.5e36 and
−4e36 by +4000 km/s). The 350–600 and 600–1000 kpc panels, by contrast, show a
well-formed line — ±700–1000 km/s, red-asymmetric, flat baseline, recognisably the
same profile as the 24–36 kpc bin scaled down.

**Why this is serious.** The arc's width matches `bg_smooth_sigma_A = 30` Å
observed ≈ **2250 km/s at 4000 Å** — i.e. the width of the background smoothing
kernel, not the width of the physical line. A feature whose width tracks an
analysis kernel rather than the astrophysics is not a detection. Compounding it,
in the 2000–3000 panel the continuum level sits at ≈ −0.85e36 while the local
baseline near line centre is ≈ 0 and the peak is ≈ +0.9e36, so roughly **half the
measured flux in that bin comes from where the continuum was placed**, and the
continuum was placed low because the red sideband collapsed.

**Important: this is not an estimator question.** It was raised against the mean
stack, but the concern is spectral shape, and §1.1's resolution does not touch it.
It must be checked in the production (biweight / σ-clipped) stack.

**The test.** Fit the line width per radial bin and plot **width vs. radius**
alongside the flux profile. If width is roughly constant from 10 kpc to ~1 Mpc and
then jumps by ~3× in the outer two bins, those bins are measuring something else
and should be reported as **upper limits**, not detections. This is also the
single best figure to hand a referee, because it demonstrates the check was made.

**Companion test.** Fit a line to the sidebands per bin and report the **baseline
tilt vs. radius**. If the tilt grows monotonically and only becomes comparable to
the signal past ~1 Mpc, that sets the radius beyond which the profile is
untrustworthy — derived from systematics rather than from where the data looks
bad, which is the difference between a defensible cut and a post-hoc one.

### 3.2 Azimuthal incompleteness and radius-dependent galaxy sample

**The objection.** At 2 Mpc ≈ 242″ ≈ 4′, annuli run off the field edge for many
galaxies, and the surviving azimuth points preferentially toward field centre.
Since the background is known *not* to be spatially flat (per B10/B11), the outer
bins sample a spatially biased subset of sky, and the bias grows with radius —
which is indistinguishable from a profile.

**Compounding effect.** The galaxy-axis combine weights galaxies equally. A
field-edge galaxy contributes an outer-bin value built from a handful of fibers;
a field-centre galaxy from many. So outer-bin behaviour is dominated by the
worst-measured galaxies, which are exactly the azimuthally-biased ones. Related:
a robust estimator across galaxies of heterogeneous precision behaves like an
implicit inverse-variance weighting, so the production profile is preferentially
measuring well-observed, field-central galaxies. That is arguably the *better*
sample — but it is currently implicit, and implicit is what gets flagged.

**The tests.**

1. Compute median azimuthal completeness vs. radius per field (`plot_field_coverage`,
   `list_exposures`) and **overplot it on the flux profile**. If completeness starts
   dropping at the radius where the profile changes character, the shape is geometry.
   If the change begins well inside full coverage, the shape is real — and that is a
   headline-grade result, not a caveat.
2. Report **per-bin contributing-galaxy counts** and re-derive the profile restricted
   to galaxies contributing to *every* bin. If unchanged, both halves close permanently.

### 3.3 Identity of the outliers rejected by the clipping

**The objection.** §1.1 closed the estimator question by showing that clipping
outliers reproduces the biweight result. That establishes the mean was
contaminated — it does **not** establish that the rejected galaxies were
artifacts. A minority of galaxies carrying enough flux to flatten the mean profile
from 180 kpc to 3 Mpc is a great deal of flux concentrated in few objects.

Two readings, with very different consequences:

- **Artifacts** (bad fibers, unmasked sources, badly-subtracted backgrounds) →
  rejection is correct, nothing further needed.
- **Real flux** (genuinely bright neighbours, protocluster members) → the median
  and clipped mean measure the *typical sightline* while the unweighted mean
  measures *total flux*. Both are legitimate, but they are different quantities,
  and **luminosity integrals and literature comparisons require the total.**

**The test.** `sigma_clip(..., masked=True)` already returns which galaxies were
rejected in each bin. Check whether the rejected set is systematically anything —
field position, local overdensity (Chartab), redshift, proximity to known bright
emitters. A random handful across bins → artifacts, closed. Clustering in
overdense regions → real flux, and the language in the paper has to change.

### 3.4 Random-position null

**The objection.** Every existing null (`redshift_scramble_null`,
`placebo_wavelengths`, `zswap_spectra`, `injection_recovery`) randomizes
*wavelength* or *sample membership*. **None randomizes position.** These test
orthogonal failure modes, and only one has been exercised.

**The test.** Stack N random sky positions in the same fields, drawing redshifts
from the real z-distribution, through the identical pipeline including the 80–90″
background annulus. Because random positions at 1–2 Mpc sample nearly the same
fibers as real positions, this is the correct control for a coherent additive
floor — the one thing §2.3's correlation matrices structurally cannot see.

**Status: gated on extraction time.** High cost, high value. A referee from the
cross-correlation or intensity-mapping community will ask for this by name.

### 3.5 Unmasked line-emitting neighbours in the background annulus

**The objection.** The Lyα mask removes the *central galaxy's* line at the
central's redshift. A neighbouring LAE at ±1000 km/s sits at ±13 Å observed —
**outside the ±10 Å mask** — and a continuum-derived segmentation map will not
catch a faint emitter with little continuum. Its flux enters the annulus coadd
and is subtracted from that galaxy's science bins.

**Why this specifically threatens the environment analysis.** The effect **scales
with local galaxy density**: galaxies in overdense regions get more neighbour flux
in their annulus, hence more over-subtraction. That biases the Chartab overdensity
split in the direction that suppresses the high-density bin — and the overdensity
split is the test needed to address the protocluster interpretation of the
large-scale blueshift (§4.3). It could flatten a real environmental signal or
invert one.

**The tests.** Correlate the raw annulus Lyα amplitude (`store_raw_bg`) against
Chartab density. If correlated, mask known catalog emitters from the annulus
before running the environment split. The mask-width sweep in §2.1 partially bounds
this channel already — another reason to have that number.

### 3.6 Cross-fiber CCD leakage

**The objection.** Light from a bright source bleeds into spectrally adjacent
columns on a fiber-fed spectrograph CCD. If the fiber-to-column mapping correlates
at all with sky position, core light appears at specific sky offsets as a false
extended halo.

**Why this is not hypothetical.** This is precisely the systematic that reduced
the [Croft et al. 2016](https://academic.oup.com/mnras/article/457/4/3541/2588984)
BOSS Lyα intensity-mapping detection **by 50%** in
[Croft et al. 2018](https://arxiv.org/abs/1806.06050) — cross-fibre light that
correlated with the signal *because the sources were clustered*. Their fix was to
eliminate fiber pairs separated by five columns or fewer. It is the closest
published analogue to this measurement and referees in this space are primed for it.

**Why it is probably bounded here.** Two things Croft et al. did not have: an
empirical stellar PSF measured through the same instrumental path (so it captures
crosstalk automatically), and a z-scramble that decoheres anything not tracking the
target redshift — leaked light sits at the *source's* observed wavelength, so
assigning a wrong redshift moves it to the wrong rest wavelength.

**The deliverable.** State the VIRUS fiber-to-column mapping, confirm the stellar
PSF was measured through the same path, and cite Croft et al. Say it explicitly
rather than leaving a referee to wonder.

### 3.7 Centroid ratio-estimator dilution

**The objection.** A flux-weighted or median-50% centroid is a ratio estimator; as
the denominator approaches zero its sampling distribution becomes heavy-tailed
while bootstrap 16/84 intervals stay deceptively finite and symmetric.

**Why it cannot fake the result.** If the centroid window is centred on systemic,
low S/N pulls the estimator *toward zero*. It can dilute a blueshift, never create
one. If anything the true outer blueshift is larger than measured.

**Why it still matters.** It is the mundane explanation for COSMOS −100 km/s vs.
EGS 0 km/s being **one measurement at two S/N levels** rather than a field
difference. That is a simpler story and it removes a wrinkle from the paper.

**The test.** Inject a zero-offset line at each field's measured outer-bin
amplitude into real outer-bin noise (`injection_recovery`, applied to the centroid
rather than the flux) and confirm recovery is centred on zero with the quoted width.

### 3.8 A repeated feature at ≈ +2300 km/s

**The objection.** A narrow spike appears at approximately the same velocity in
180–350, 350–600, 600–1000, 1000–2000, and 2000–3000 kpc. Five nominally
independent radial bins showing the same narrow feature at the same velocity is a
shared systematic, and it sits at the edge of the red continuum sideband, so it
pulls the continuum fit in every one of those bins.

**The test.** Identify it (sky line at the sample's modal redshift? window
boundary? detector feature?), then mask it or move the sideband. Also check whether
it is what drives the 0.3–0.4 outer-bin correlations in §2.3 — the bin-correlation
matrices were built for exactly this.

### 3.9 Two splits not yet run

- **By MOSDEF redshift window**, not by median z. MOSDEF's windows are discrete,
  set by where Hα/[O III] fall in the atmospheric bands, so a median-z cut splits
  *within* a window rather than between them and cannot detect an observed-frame
  systematic localised to one window.
- **By observing epoch / exposure set.** Orthogonal to the field split. If a subset
  of exposures taken in poor conditions drives the outer signal, only a date split
  catches it. Standard, cheap, and conspicuous by absence in a systematics section.

---

## 4. RETIRED — raised, examined, does not apply

Recorded so they are not re-litigated.

### 4.1 Continuum-fit asymmetry faking the inner red offset

A polynomial continuum fit spanning both sidebands cannot represent the one-sided
Lyα-forest decrement blueward of 1215.67 Å, which would leave a negative blue
residual and push the measured centroid red — manufacturing the red end of the
red→blue trend.

**Retired.** `cont_method = "median"` was chosen for this reason, and continuum is
only present inside ~50 kpc. Note also that the residual bias from a median over
asymmetric sidebands runs *blue*, which would understate the inner red offset
rather than create it.

### 4.2 S/N as the explanation for the COSMOS/EGS double-peak difference

**Retired.** EGS has more fibers in the core bins and slightly higher core S/N, so
"COSMOS resolves it because COSMOS is deeper" is backwards. Combined with §1.3
(resolution and redshift jitter both excluded), the field difference in line shape
survives as a physical result.

### 4.3 Clustering as the explanation for the large-scale blueshift

**Retired as an artifact explanation, retained as an interpretation caveat.**

Ordinary clustered Lyα from neighbours should appear **red**: each neighbour's own
Lyα is redshifted from its own systemic, and neighbour peculiar velocities are
symmetric about the central. So the two-halo term cannot produce a net blueshift.

What *can*, and should be recorded: **large-scale infall onto an overdensity
produces a blue-dominant Lyα profile** (infall → blue peak; outflow → red peak),
and at z ≈ 2 the turnaround radius for a protocluster is Mpc-scale. COSMOS is
known to intersect multiple superclusters; EGS less so. That predicts exactly the
observed pattern — COSMOS reaching ≈ −100 km/s while EGS stays flat — and it
predicts the sign, which "smaller sample, weaker flux" does not.

This does not invalidate anything. It changes the claim from *"z ≈ 2 galaxies show
large-scale blue Lyα"* to *"protocluster environments do"*, which is at least as
interesting. **The Chartab overdensity split is the test**, and §3.5 must be
cleared first or the split will be biased.

### 4.4 Purely kinematic or geometric explanations for the blueshift

**Retired, and this is an argument to use rather than merely a dismissal.**

Coherent infall (Kaiser) compresses structure along the line of sight
*symmetrically*: far-side gas approaches, near-side gas recedes, mean shift zero.
Redshift-space distortion changes line **width**, not centroid. No purely kinematic
or projection effect produces a mean velocity offset — **only radiative transfer
does.**

### 4.5 Per-field line-spread kernel

**Retired.** Raised as a narrow survival of §1.3 — the ~1.9 Å kernel is pooled over
all redshifts, so COSMOS and EGS might differ. Not credible: it is not plausible for
either field's kernel to exceed ~2 Å, and at 2 Å the peaks are still resolved and
comfortably inside the measurement window.

### 4.6 Sky-line contamination surviving the redshift scramble

**Retired.** The MOSDEF redshift range places Lyα anywhere from ~3525–4380 Å, wide
enough that scrambling genuinely decoheres narrow sky features. The z-scramble is
doing real work.

---

## 5. Two systematics that run *against* the result

Worth a dedicated paragraph in the paper rather than being scattered through
defensive replies. Both of the systematics a referee reaches for first push the
wrong way:

1. **PSF wings and scattered light** carry the bright core's **red** centroid
   outward. Any leakage biases the outer centroid *red*. The measured trend runs
   **blue** — contamination and signal have opposite signs.
2. **Background over-subtraction** (§2.1) biases the outer flux **down**. The
   detection survives despite it, and the true profile is higher than measured.

Stated together: the measurement is conservative in both quantities.

---

## 6. Standing assessment

**Relative to the field.** The standard systematics package in the Lyα halo
literature is a PSF check, one stacking null, and a continuum/background
sensitivity test — and most of that work sits at ≤100 kpc, where systematics are
far more forgiving. This analysis carries injection-recovery, placebo wavelengths,
a bin covariance matrix, jackknife-vs-bootstrap cross-checking, multiple SMHM
relations with a divergence budget, four background annuli, mask-width and window
sweeps, six galaxy-combine estimators, and two-field replication with independent
reductions. That is well above the norm.

**Two places behind the norm**, both cheap: the random-position null (§3.4),
standard in the cross-correlation and intensity-mapping literature; and, until
2026-07-28, estimator transparency (§1.1, now closed).

**The honest radial standing:**

| Range | Standing |
|---|---|
| ≤ 800 kpc | Well defended. External corroboration from LN22 Paper II on different fiber format, reduction, and background subtraction, plus an independent empirical stellar PSF. Better validated than most published work in this area. |
| 800 kpc – 1 Mpc | Solid. Line shape in 350–600 and 600–1000 kpc is recognisably the same profile as the inner bins. |
| 1 – 3 Mpc | **Unresolved.** §3.1 (line width and baseline tilt) must be run before these bins are claimed as detections. Currently they may be upper limits. |

Novelty attracts scrutiny disproportionate to systematics effort: a referee will
concentrate on the one range where nobody can check the result, regardless of how
good the rest is. §3.1 and §3.2 are what decide that range, and neither needs new
data.

---

## 7. Test queue

Ordered by value per unit effort. Items 1–4 need **no new extraction**.

| # | Test | Closes | Cost |
|---|---|---|---|
| 1 | Line width vs. radius, per bin, production stack | §3.1 | Low |
| 2 | Baseline tilt vs. radius (sideband slope per bin) | §3.1 | Low |
| 3 | Azimuthal completeness vs. radius, overplotted on profile; per-bin galaxy counts; common-subset profile | §3.2 | Low |
| 4 | Identify clipped-outlier galaxies; test against Chartab density | §3.3 | Low |
| 5 | Quote mask 10 Å vs. 20 Å outer-profile change; quote outer profile under all four `bg_variants` | §2.1 | None (recorded, not re-run) |
| 6 | Identify the +2300 km/s feature; mask or move sideband | §3.8 | Low |
| 7 | Annulus raw Lyα amplitude vs. Chartab density | §3.5 | Low |
| 8 | Centroid injection-recovery at each field's outer amplitude | §3.7 | Medium |
| 9 | Delete-one-structure jackknife | §2.4 | Medium |
| 10 | Splits by MOSDEF redshift window and by observing epoch | §3.9 | Medium |
| 11 | Random-position null | §3.4 | High — gated on extraction |

---

## See also

- `specs/systematics-and-extraction.md` — extraction-level systematics, §2.5 background-variant products
- `specs/background-diagnostics.md` — §0 on where the annuli actually sit; the controlling geometry for §2.1
- `specs/bin-correlation-diagnostics.md` — machinery behind §2.3, and the tool for §3.8
- `docs/GOALS.md` — project framing and headline claims
- `docs/research-notes.md` — SMHM divergence test, biweight/smoothing artifact history
- Croft et al. 2018, MNRAS 481, 1320 ([arXiv:1806.06050](https://arxiv.org/abs/1806.06050)) — the cautionary precedent for §3.6
