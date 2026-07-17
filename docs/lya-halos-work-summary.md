# Lyα Halos (VIRAL) — Project Overview

*Planning document for the next presentation/paper pass, revised 2026-07-16. Organized around the
science rather than the chronology of the work — dataset/pipeline/methods are kept brief;
motivation, results, validation, and interpretation carry the weight. Title TBD.*

---

## 1. Motivation

The project quantifies the extent and kinematics of Lyα halos around a sample of Cosmic Noon
(z~2–3) galaxies. Five questions frame why this matters:

- **How can we observe galaxy evolution through extended Lyα emission?** Resonantly scattered
  Lyα traces neutral hydrogen well beyond what continuum imaging reaches, offering a direct probe
  of gas that continuum-only studies can't see.
- **What do Lyα halos tell us about galaxies and their surrounding environments** — spanning the
  ISM, CGM, and IGM/cosmic web? A single radial profile potentially carries information across all
  three regimes, from the galaxy's own interstellar gas out to structure shared with neighbors.
- **How are gas distribution and kinematics related to galaxy properties?** Does halo extent or
  velocity structure track mass, star formation, dust, or environment — and which of those
  relationships is causal versus incidental?
- **How does the Lyα line behave kinematically relative to the systemic velocity?** Is the gas
  net-inflowing, net-outflowing, or something more radius-dependent — and can that be pinned to
  true zero velocity rather than just a relative line shape?
- **How do satellite galaxies contribute to Lyα halos, and how much of the halo is produced by
  scattering** (as opposed to in-situ emission — cooling radiation, fluorescence, or truly diffuse
  intergalactic gas)?

This project's specific methodological edge for answering the fourth question: MOSDEF supplies an
**independent, systemic (rest-frame optical) redshift** for every galaxy, from Hα or [OIII].
Comparable studies (e.g. Guo et al. 2024) can only measure a Lyα centroid shift *relative to the
line itself* — a genuinely different, less direct quantity. Anchoring to systemic velocity means
this analysis can state directly where the gas sits kinematically (blueshifted / redshifted / at
rest), not just how the line shape changes.

---

## 2. Dataset, Pipeline & Methods (brief)

**Dataset.** MOSDEF galaxies (H≲24.5 at z~2, H≲25 at z~3) cross-matched to VDFI/VIRUS
spectroscopy in the COSMOS and EGS fields — roughly 500 galaxies total, ~450 after excluding AGN
(the reference sample for all R_vir-relative numbers below). Each galaxy has ~100–150 exposures,
each contributing spectra from ~35,000 individually-positioned fibers.

**Pipeline.** Per galaxy, per exposure: subtract a local sky background, bin fibers into radial
annuli (angular, kpc, or virial-radius units), and coadd. Bins are then coadded again across the
full galaxy sample (biweight) to produce one composite spectrum per radial bin — the spectrum of
the "average galaxy" at that radius. The innermost bins are PSF-dominated and highly correlated;
VIRUS's fiber size means this pipeline cannot recover a clean, resolved down-the-barrel spectrum
of any single galaxy's ISM, which is why the composite radial stack — not individual objects — is
the unit of measurement throughout.

**Measurement.** A local continuum is fit and subtracted; Lyα line flux is integrated in a
±4 Å window around rest-frame 1215.67 Å (set by the ~60 km/s precision of the MOSDEF redshifts);
the centroid is measured as the flux-weighted 50% point (median-flux centroid), which is more
noise-robust than a first-moment centroid. Errors are bootstrapped (galaxy resampling),
cross-checked against independent methods in §4.

---

## 3. Core Results

### 3.1 Radial flux (surface-brightness) profile

The stacked Lyα flux as a function of radius, in both kpc and virial-radius units. Flux declines
steeply near the galaxy and flattens at large radius rather than dropping to zero — the first
qualitative sign of a two-component (core + extended) structure, quantified in §7.

### 3.2 Centroid velocity vs. radius — the headline kinematic result

The Lyα centroid, referenced to the true systemic velocity (not a relative line shift), as a
function of radius: **red-dominated near the galaxy center, crossing systemic velocity near
R_vir, then mildly negative (blueshifted) beyond it.** Candidate physical readings: a blueshift
consistent with scattering off inflowing/accreting gas, a redshift consistent with outflows or
satellite-clustering contamination.

The signal beyond R_vir is real but modest, not a confident detection — quantified two
independent ways: individual outer bins sit at ~1.2–1.35σ each; a covariance-aware combination
across all bins beyond R_vir (accounting for the fact that bootstrap draws share galaxies across
radii, so bins aren't statistically independent) gives **~1.7σ** (two-sided); an independent
flux-level pooling cross-check reproduces the same central value (~−47 to −48 km/s) at **~1.5σ**.
Agreement between two independently-constructed statistics is the strongest part of the case.

### 3.3 Central & comparison spectral profiles — blue vs. total emission

Beyond the single centroid number, the actual line shape at each radius is compared directly:
blue-side flux and red-side flux tracked separately as a function of radius, alongside a
biweight-normalized radius–velocity map showing how the whole line profile (not just its
midpoint) evolves outward. This is the qualitative view that the centroid number in §3.2
summarizes into one figure per bin.

### 3.4 Blue-to-total flux ratio

Not a headline result on its own, but a useful, literature-comparable quantitative companion to
the centroid: the fraction of line flux on the blue side of systemic, B/(B+R), at each radius.
**0.5 means symmetric (systemic); above 0.5 is blue-dominated, below is red-dominated.** This
gives an independent, differently-constructed number that tracks the same physical transition as
the centroid — both flip sign near the same radius (§7) — and is a more direct quantity to compare
against studies that report flux ratios rather than velocity centroids.

---

## 4. Validation & Systematics (summary)

A standard battery of checks was run to establish that the centroid and flux-profile results are
real signal, not a pipeline artifact:

- **Error consistency** — bootstrap, jackknife, and redshift-scramble error estimates agree with
  each other and with propagated error spectra.
- **Centroid-method consistency** — multiple centroid estimators (flux-weighted moment,
  Gaussian-weighted moment, Gaussian fit, median-flux) agree.
- **Instrumental checks** — a star-PSF comparison confirms the halo is genuinely more extended
  than the instrumental PSF; an LSF simulation confirms the ±4 Å measurement window is not itself
  dominated by instrumental line-spread broadening.
- **Known open item** — outer-bin background noise is correlated rather than random; still under
  investigation as a possible source of bias in the exact cutoff radius (see §7 caveats).

**Headline validation: independent agreement with Niemeyer et al. 2022.** Both analyses use
VIRUS/HETDEX data, but Niemeyer et al. 2022 is an independently reduced, independently analyzed
3D-HST-based sample. The fact that this pipeline's centroid and flux measurements closely
reproduce theirs — despite different samples, reduction, and methodology built independently — is
the strongest available evidence that the result isn't an artifact of this specific pipeline.

---

## 5. Core Spectrum Extraction

Before the subsample analysis, each galaxy's own core (innermost, PSF-dominated) Lyα spectrum is
measured directly — line flux, velocity offset, and (where S/N allows) a single- vs. double-Gaussian
decomposition of the line profile, giving peak separation and relative peak height as additional
per-galaxy quantities. These per-galaxy numbers are what several of the subsample bins in §6 are
actually built from: the Lyα velocity-offset self-split, the classification scheme (LAE / LAA /
absorber / null / AGN), and the double-Gaussian peak-separation and peak-height splits all use a
quantity measured here, not an external catalog value. One open methodological question flagged
before trusting the peak-separation split specifically: whether fitting a double-Gaussian template
to pure noise and stacking by the fit's own derived quantities could produce a template-shaped
result purely as a selection artifact — a null test planned but not yet run.

---

## 6. Subsample Analysis

This is the section expected to carry the most new weight going forward, and — once the updated
splits land — could plausibly be reframed directly into a paper's Results/Discussion structure
without much restructuring; the pieces already largely exist.

**Framework.** Splits cost statistical power (~√N per split), so each candidate split earns its
place by testing a specific, literature-motivated physical prediction rather than being explored
opportunistically. Splits are divided by which physical question they answer:

- **Kinematics-focused** (prioritized for the discovery paper): mass and SFR-normalized mass,
  ΣSFR (feedback/outflow proxy), the Lyα velocity-offset self-split, redshift bin (z~2 vs. z~3,
  mass-normalized — also a systematics check), dust attenuation, environment/satellite clustering,
  equivalent width, orientation/inclination, double-Gaussian peak separation & height ratio,
  burstiness, and mass-normalized half-light radius.
- **Flux/photon-budget-focused** (second paper): Hβ-normalized escape fraction, core
  classification type, and a PSF-aware exponential scale length fit to the Lyα emission itself.

**Status.** *[PENDING — to be updated with the new subsample splits once ready.]* Even without
the additional incoming dataset, the current sample already supports enough of these splits to
give the presentation (and eventually a paper) real subsample-level results, not just a plan.

**Presentation format for each split, once results are in:** for every subsample quantity, state
(1) what the literature predicts, (2) what is actually observed, and (3) what that agreement or
disagreement means — never present a figure without that three-part framing.

---

## 7. Profile Fitting Analysis

The flux profile (§3.1) and the centroid turnover (§3.2) are two views of the same underlying
transition; this section is where that transition gets a number attached to it.

**Model.** A two-component exponential is fit to the radial flux profile,
`I(r) = A1·exp(-r/h1) + A2·exp(-r/h2)`, using a PSF-aware forward model (ring-convolution) so the
innermost, most PSF-affected bin can be included rather than discarded. Fit quality: χ²/dof = 1.17
across all 10 bins.

**Results:**
- **Core scale length: h1 = 16.9 ± 1.1 kpc** — consistent with the range from comparable stacked
  Lyα halo studies (Steidel et al. 2011: 20.8–28.4 kpc; Wisotzki et al. 2016: 1–7 kpc; Leclercq
  et al. 2017: ~4.5 kpc).
- **Outer scale length: h2 = 1552 ± 548 kpc** — well beyond the single-halo/CGM regime, and
  (converted to comoving units) close to the only direct-imaging detections of the cosmic web in
  Lyα emission published to date.
- **Turnover radius: the fitted crossover between the core and outer terms sits at ≈76 kpc —
  essentially exactly at the independently-derived virial radius, R_vir = 75 kpc** (AGN-excluded
  sample). This is the fit converting the qualitative "profile bends here" observation into an
  actual, physically anchored number.

**The key connection to emphasize:** this turnover radius is not just where the flux profile
flattens — it is *also*, independently, almost exactly where the centroid (§3.2) and blue/total
ratio (§3.4) both flip sign. Three qualitative diagnostics and one quantitative fit are all
converging on the same physical radius. That convergence — not any single measurement on its own —
is the strongest evidence this is a real transition rather than a fitting or pipeline artifact,
and it is the direct observational link to theoretical halo-scale gas-transition models, making it
a central discussion point rather than a methods footnote.

**Caveats to carry into discussion:** the outer term's physical origin (galaxy clustering vs.
diffuse IGM optical depth) is genuinely open, not resolved (§8); the correlated outer-bin noise
flagged in §4 could shift the exact turnover radius; fit errors are currently single-fit
covariance, not yet bootstrapped.

---

## 8. Literature Context: One-Halo / Two-Halo Framework (brief)

Just enough framing to connect the two-component fit to the field's existing vocabulary, not a
full review (the full literature survey lives in `halo_gas_correlation_literature_review.md`).

Standard halo-model language splits any galaxy–gas correlation into a **one-halo term** (gas
belonging to the galaxy's own dark-matter halo — CGM proper, dominant from a few kpc out to
~R_vir) and a **two-halo term** (gas correlated with *neighboring* halos — large-scale structure,
dominant beyond R_vir). This maps directly onto the fitted h1/h2 decomposition. Key anchors:
**Rakic et al. 2012** found an order-of-magnitude opacity drop at ~100 kpc (≈R_vir) in absorption,
with distinct redshift-space anisotropies on either side — the field's cleanest precedent for "two
physically distinct scales." **Byrohl et al. 2021**'s radiative-transfer simulations predict that
outer-profile flattening in emission specifically comes from photons originating in *other* halos,
not the galaxy's own diffuse gas — the physical interpretation this project's h2 term is now
organized around. **Sorini et al. 2018** independently finds a galaxy's "sphere of influence"
extends to ~7×R_vir in absorption, loosely consistent with this fit's r_c ≈ 6.5×R_vir.

---

## 9. UV Continuum Comparison

A direct comparison of the Lyα halo's core scale length against the sample's own rest-frame UV
continuum scale length — the classic way this kind of result is quantified in the literature
(e.g. Steidel et al. 2011 report Lyα halos ~5–10× more extended than the UV continuum for the
same galaxies).

**Status: *[PENDING]*.** The extraction pipeline (CFHT-LS r-band cutout → centroid → annular
photometry → coaddition → single-exponential PSF-aware fit, mirroring the same fitting machinery
used for the Lyα profile in §7) is fully built but has not yet been run against real imaging data.
Once run, the payoff is a single clean number: **how many times larger the Lyα-emitting region is
than the galaxies' own starlight**, using the exact same sample and the exact same fitting
convention as §7 — a same-sample comparison rather than borrowing another study's continuum figure.

---

## 10. Status & Synthesis

What's ready now, in roughly the order it should appear in a presentation: the motivation (§1),
a brief methods overview (§2), the core radial results with real numbers behind them (§3), a
validated methodology backed by independent agreement with prior HETDEX/VIRUS work (§4), and a
quantitative fit that ties the flux, centroid, and B/T turnover together at a single physically
meaningful radius (§7), with just enough literature framing to place that result in context (§8).

What's still pending: the updated subsample-split results (§6) and the UV-continuum comparison
(§9). Both are scoped and largely built, not blocked on new ideas — just execution. Once those
two land, this outline is close to a full paper's Results/Discussion structure, not just a talk.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             