# VIRAL-Halos Overview 2 — Slide Text

*Section titles + slide-ready content, written for you to paste into your own deck and trim/edit
as needed. Fully written where the content is already known; marked [PENDING] where it depends on
work not yet done (subsample splits, UV continuum). Numbers are current as of 2026-07-16.*

---

## Overview of Project

V — VIRUS deep field
I — Investigation of
R — Resonant Lyman
A — Alpha
L — Light (Halos)

Goal: Quantify the extent and kinematics of Lyα halos around Cosmic Noon (z~2–3) galaxies —
connecting the ISM, CGM, and IGM/cosmic web — to learn about galaxy formation, gas kinematics, and
environment.

---

## Motivating Questions

*Kicker line:* MOSDEF's independent systemic redshift lets us anchor these questions to true zero
velocity, not just a relative line shift — the project's core methodological advantage over
studies like Guo et al. 2024.

- How can we observe galaxy evolution through extended Lyα emission?
- What do Lyα halos tell us about galaxies and their surrounding environments — spanning the ISM,
  CGM, and IGM/cosmic web?
- How are gas distribution and kinematics related to galaxy properties?
- How does the Lyα line behave kinematically relative to the systemic velocity?
- How do satellite galaxies contribute to Lyα halos, and how much of the halo is produced by
  scattering?

---

## Dataset

- MOSDEF survey: H-band selected galaxies, H ≲ 24.5 (z~2 bin) / H ≲ 25 (z~3 bin); systemic
  (rest-frame optical) redshifts from Hα or [OIII]
- Sample: ~500 galaxies across COSMOS + EGS; ~450 after excluding AGN — this AGN-excluded sample
  is the reference for all R_vir-relative measurements in this talk
- VDFI/VIRUS spectroscopy: first data release, ~100–150 exposures per galaxy, each exposure
  contributing ~35,000 individually positioned fibers

---

## Pipeline & Methods

- Per exposure: subtract a local sky background, bin fibers into radial annuli (angular / kpc /
  virial-radius units)
- Coadd fibers per bin (biweight), then coadd across the full galaxy sample (biweight) — produces
  one composite spectrum per radial bin, the "average galaxy" at that radius
- Continuum subtracted via a local fit; Lyα line flux integrated in a ±4 Å window around
  rest-frame 1215.67 Å; centroid measured as the flux-weighted 50% point (median-flux centroid)
- Errors from bootstrap galaxy resampling
- Innermost bins are PSF-dominated — VIRUS's fiber size means we cannot recover a resolved,
  down-the-barrel spectrum of any single galaxy; the stacked radial profile, not individual
  objects, is the unit of measurement throughout

---

## Radial Flux Profile

- Stacked Lyα surface brightness declines steeply near the galaxy center and flattens at large
  radius rather than dropping to zero
- First qualitative sign of a two-component (core + extended) structure — quantified directly by
  the profile fit later in the talk
- Shown in both kpc and virial-radius (R/R_vir) binning

---

## Centroid vs. Radius — headline result

- The Lyα centroid, measured relative to the true systemic velocity (not a relative line shift),
  as a function of radius
- Shape: red-dominated near the galaxy center → crosses systemic velocity near R_vir → mildly
  negative (blueshifted) beyond R_vir
- Candidate physical readings: blueshift consistent with scattering off inflowing/accreting gas;
  redshift consistent with outflows or satellite-clustering contamination
- Significance beyond R_vir: individual outer bins ~1.2–1.35σ each; a covariance-aware combination
  across all bins beyond R_vir gives **~1.7σ** (two-sided); an independent flux-pooled cross-check
  reproduces the same central value (~−47 to −48 km/s) at **~1.5σ**
- Framed honestly as a consistent, modest trend — not a confident detection on its own

---

## Central & Comparison Spectral Profiles

- Blue-side and red-side Lyα flux tracked separately as a function of radius — the line-shape view
  behind the single centroid number
- Biweight-normalized radius–velocity map shows how the whole line profile evolves outward, not
  just its midpoint
- Sets up the Blue-to-Total ratio (next slide) as a second, independently-constructed quantity
  that tracks the same transition

---

## Blue-to-Total Ratio

- B/(B+R): the fraction of Lyα line flux on the blue side of systemic velocity, at each radius
- 0.5 = symmetric/systemic; above 0.5 = blue-dominated; below 0.5 = red-dominated
- Not a headline result on its own, but a literature-comparable quantitative companion to the
  centroid — many published studies report flux ratios rather than velocity centroids
- Flips across 0.5 near the same radius the centroid crosses systemic — an independent
  confirmation of the same turnover

---

## Validation & Systematics

- Error consistency: bootstrap, jackknife, and redshift-scramble error estimates agree with each
  other and with propagated error spectra
- Centroid-method consistency: multiple estimators (flux-weighted moment, Gaussian-weighted
  moment, Gaussian fit, median-flux) agree
- Instrumental checks: star-PSF comparison confirms the halo is genuinely more extended than the
  instrumental PSF; LSF simulation confirms the ±4 Å window isn't itself dominated by instrumental
  broadening
- **Headline: independent agreement with Niemeyer et al. 2022.** Both analyses use VIRUS/HETDEX
  data, but Niemeyer et al. 2022 is an independently reduced, independently analyzed 3D-HST-based
  sample — close agreement between the two is the strongest available evidence this result isn't a
  pipeline artifact
- Open item: outer-bin background noise is correlated rather than random — still under
  investigation as a possible source of bias in the exact turnover radius

---

## Core Spectrum Extraction

- Each galaxy's own core (innermost, PSF-dominated) Lyα spectrum is measured directly: line flux,
  velocity offset, and (where S/N allows) a single- vs. double-Gaussian decomposition giving peak
  separation and relative peak height
- These per-galaxy quantities feed several of the subsample bins directly: the Lyα velocity-offset
  self-split, the spectral classification scheme (LAE / LAA / absorber / null / AGN), and the
  double-Gaussian peak-separation and peak-height splits
- Open methodological check before trusting the peak-separation split: whether fitting a
  double-Gaussian template to pure noise and stacking by the fit's own derived quantities could
  produce a template-shaped result purely as a selection artifact (planned, not yet run)

---

## Subsample Analysis

- Splitting the stack costs statistical power (~√N per split) — each split earns its place by
  testing a specific, literature-motivated physical prediction
- Kinematics-focused splits: mass and SFR-normalized mass, ΣSFR, Lyα velocity-offset self-split,
  redshift bin (z~2 vs. z~3, mass-normalized), dust attenuation, environment/satellite clustering,
  equivalent width, orientation/inclination, double-Gaussian peak separation & height ratio,
  burstiness, mass-normalized half-light radius
- Flux/photon-budget-focused splits (second paper): Hβ-normalized escape fraction, core
  classification type, PSF-aware exponential scale length of Lyα itself
- **[PENDING — results to be added once the updated splits are ready; framework and candidate
  list are ready now]**
- Format for each split once results land: what does the literature predict? what do we observe?
  what does the agreement or disagreement mean?

---

## Profile Fitting Analysis — key result

- Two-component exponential fit to the radial flux profile: `I(r) = A1·exp(−r/h1) +
  A2·exp(−r/h2)`, PSF-aware (ring-convolution forward model), fit to all 10 radial bins
- Fit quality: χ²/dof = 7.04/6 = **1.17**
- **Core scale length: h1 = 16.9 ± 1.1 kpc** — compares to Steidel et al. 2011 (20.8–28.4 kpc),
  Wisotzki et al. 2016 (1–7 kpc), Leclercq et al. 2017 (~4.5 kpc)
- **Outer scale length: h2 = 1552 ± 548 kpc** — well beyond the single-halo/CGM regime, comparable
  (in comoving units) to the only direct-imaging detections of the cosmic web in Lyα emission
  published to date
- **Turnover radius: the fitted crossover between the two terms sits at ≈76 kpc** — essentially
  exactly at the independently-derived virial radius, **R_vir = 75 kpc** (AGN-excluded sample)
- **Key connection:** this turnover radius is also, independently, almost exactly where the
  centroid and blue-to-total ratio both flip sign — three qualitative diagnostics and one
  quantitative fit converge on the same physical radius. This convergence is the strongest
  evidence for a real physical transition, and the direct observational link to theoretical
  halo-scale gas-transition models — the central discussion point of the whole talk

---

## Literature Context: One-Halo / Two-Halo

- Standard halo-model language splits any galaxy–gas correlation into a **one-halo term** (gas in
  the galaxy's own dark-matter halo — CGM proper, dominant from a few kpc out to ~R_vir) and a
  **two-halo term** (gas correlated with neighboring halos — large-scale structure, dominant
  beyond R_vir)
- Maps directly onto the fitted h1/h2 decomposition
- Rakic et al. 2012: order-of-magnitude opacity drop at ~100 kpc (≈R_vir) in absorption, with
  distinct redshift-space anisotropies on either side — the field's cleanest precedent for "two
  physically distinct scales"
- Byrohl et al. 2021 (IllustrisTNG50 radiative transfer): outer-profile flattening in emission
  specifically comes from photons originating in *other* halos, not the galaxy's own diffuse gas —
  the physical interpretation our h2 term is organized around
- Sorini et al. 2018: a galaxy's "sphere of influence" extends to ~7×R_vir in absorption — loosely
  consistent with our fitted r_c ≈ 6.5×R_vir

---

## UV Continuum Comparison

- Direct comparison of the Lyα halo's core scale length (h1) against the sample's own rest-frame
  UV-continuum scale length (h_UV) — the same-sample analog of Steidel et al. 2011's classic
  result (Lyα halos ~5–10× more extended than the UV continuum for the same galaxies)
- Extraction pipeline built: CFHT-LS r-band cutout → centroid → annular photometry → coaddition →
  single-exponential, PSF-aware fit, using the same fitting machinery as the Lyα profile
- **[PENDING — not yet run against real imaging data]**
- Payoff once measured: a single clean number — how many times larger the Lyα-emitting region is
  than the galaxies' own starlight, on the exact same sample and fitting convention as the halo
  fit above

---

## Synthesis & Next Steps

- What's solid now: motivation, dataset/methods, core radial results (flux, centroid, spectral
  shape, B/T), validated methodology (independent Niemeyer et al. 2022 agreement), and a
  quantitative fit tying flux/centroid/B-T together at one physical radius
- What's pending: updated subsample-split results, UV-continuum comparison — both scoped and
  largely built, not blocked on new ideas, just execution
- Once those land, this outline is close to a full paper's Results/Discussion structure, not just
  a talk
