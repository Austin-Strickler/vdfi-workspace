
---

## 0. Abstract?

???

---

## 1. Introduction

- Resonant scattering traces HI past where continuum imaging reaches; one profile spans ISM → CGM → IGM.
- Halos are ubiquitous: Wisotzki 2018, Steidel 2011, Momose 2014/2016, Leclercq 2017, Guo 2024.
- Stacked profiles are common; absolute kinematics aren't. Other HETDEX work (Niemeyer, Davis, McKay, Cooper) has no independent systemic z. Guo 2024 is self-referenced to the line.
- One-halo/two-halo boundary exists in absorption (Rakic 2012) and simulation (Byrohl 2021, Zheng 2011), not in emission.
- Leclercq 2020 and Guo 2024 disagree on which way the centroid line goes outward.
- What we add: absolute frame, fitted crossover radius, two observables on one radius, per-galaxy properties on the same objects.
- Conventions to define once: cosmology, R_vir, proper vs. comoving, air vs. vacuum, and that h1 / r_c / r_0 / crossover are different quantities.

---
## 2. Data

- **VIRUS/VDFI** — COSMOS + EGS, ~100–150 exposures per galaxy, R≈800, ~2 Å effective rest-frame.  
- **MOSDEF** — H-selected, ~500 in footprint, ~450 after AGN cut. LAEs, emitters, absorbers, nulls, generally a mixed LBG+MELG population.
- AGN cut: X-ray + IRAC wedge + BPT.
- Systemic z precision ~60 km/s (citation unverified).
- Ancillary: 3D-HST photometry, van der Wel structure, Chartab density, JWST morphology, CFHT-LS imaging.
- Additional samples: 3D-HST (independent repeat + compare to Niemeyer 2022), COSMOS LAE catalog (still in progress).
- COSMOS intersects real over-densities (COSTCO) — field differences are plausibly physical.

---
## 3. Methods

- Mask foreground contaminants with Source Extractor.
- Extract fibers into annular kpc bins out to ~2 Mpc.
- Background from ~800kpc annulus (80-90 arcsec), Lyα region masked, smoothed with 30A filter.
- Galaxy: biweight individual fibers into spectra per annulus per galaxy
- Stack: de-redshift, proper units, biweight coadd → ~10 spectra, one per
  radius. Fields extracted separately.
- Continuum-subtract; flux in a narrow rest window; centroid = median-flux point.
- Centroid window compresses velocity slightly — forward-modeled, not widened.
- Velocity zero point: air/vacuum offset (~85 km/s) found and corrected. Barycentric small. A rigid
  offset can't produce the gradient, only flip the outer sign.
- PSF: ring-convolution forward model, innermost bin kept. Star PSF shows the halo exceeds it.
  Open — PSF constants differ across modules.
- Errors: bootstrap primary; jackknife, z-scramble, and random wavelength generally agree. Bins correlated, covariance measured.
- Injection–recovery through the full chain. Acceptance passed, floor ~10 km/s, velocity response
  invertible.

---

## 4. Results

Observation:
- **Radial spectra** — the stack panels itself.
- **Brightness profile** — PSF rounding, then steep decline, then flattening.
- **Centroid vs. radius** — red near center, becomes consistent with zero near R_vir. Only measurable for MOSDEF with z_sys, and with LAEs with relative shift.
- **Two-component fit** — PSF-aware exponential + power law. Scale length ~15 kpc, crossover ~65–70 kpc, outer scale ~1 Mpc. 3D-HST: smaller both, with no flattening at intermediate radii and is straight into power law. LAEs even smaller, seeing-sensitive.
- **Validation** — Niemeyer 2022 agrees with 3D-HST; Guo 2024 agrees with LAEs; centroid estimators agree; error methods generally agree; injection recovers input.

Advanced:
- **Individual Galaxy Halo Investigation** — measure core vs 1-halo light individually for each galaxy (rough numbers), and find trends that best describe and explain the halo vs no halo phenomena based on morphology, emission lines, and SED measures.
* **UV comparison** — same-sample Lyα-to-UV scale ratio, roughly 10×. PSF is the hard part. Compare ratio to others (Steidel...).
- **Splits** — halo vs. galaxy property. Regression ranks, quintile stacks are possible. No
  outer-bin claims are possible however with N < 100. Correlation matrix separates direct from indirect.
- **Measured two-halo slope** — measure on MOSDEF, 3D-HST, and LAEs to obtain population-wide numbers. Use MCMC to disentangle possible degeneracy with scale radius (primary problem for MOSDEF)
- **Environment Comparison** — answer why the MOSDEF sample looks different than 3D-HST and others. Tie in Chartab 2020 density maps, or create our own scale-dependent overdensity to independently predict and measure overdensity-Lyα relation.
- **AGN/LAE Physics from MOSDEF** — look into how the AGN and LAE/emitter sample compares with the others in their shape (might be less informative compared to individual investigation)

Plausible:
- **Photon budget** — intrinsic Lyα from dust-corrected Balmer × Case B compared to observed Lya inside 1-halo. Anything less than 100% is due to dust extinction, though is a lower limit as other Lyα production channels may contribute.
- **Two-halo amplitude** — use literature cited bias scale length to predict global mean Lyα. Measuring bias with the stack is not possible (to our knowledge), but going from distance to flux measures is.
- **Peak Separation** — with enough signal, measure the double-peak separation of semi-resolved double peak Lyα emission as a proxy for neutral hydrogen column density.
- **Galaxy-galaxy pair stacking** — do we detect a Lyα bridge connecting the galaxy pairs?
- **Galaxy-LAE Correlations** — measure correlation functions, mask neighbors, use LAEs for overdensity, all to learn about the general galaxy environments
- **More Catalogs** — obtain more COSMOS or AEGIS spectroscopic catalogs and look at those samples to get broader selection
- **Proto-clusters** — look at galaxies embedded inside proto-clusters and extract those profiles

---

## 5. Discussion

- **The boundary** — emission-side localization of the one-halo/two-halo split. Rakic 2012 in
  absorption; Chen 2020 kinematic transition at similar radii; Sorini 2018 sphere of influence.
- **Inside** — halo is redistributed light; redistributed-vs-destroyed set by galaxy properties.
  Absorbed fraction and scale length should track? (longer scattering path → more dust *and* bigger halo) — Steidel's f_esc–size prediction.
- **Outside** — in-situ SF (falsified on budget), photons from other halos (Byrohl), filamentary
  emission. Discriminator: density field at several smoothing scales, and which one the outer
  amplitude tracks. MUSE (Bacon, Kozlova, Tornotti) argues clustering with other emitters; Bacon
  caps fluorescence, needing faint LAEs our cuts may exclude.
- **Mpc-scale kinematics** — outer bins carry hundreds of km/s of Hubble flow, but also peculiar velocities. IGM eats blue photons. What would an inflow look like?
- **Leclercq / Guo** — reddening inside and blue outside, different scales, and different samples would reconcile them.
- **Literature** — scale length vs. Steidel, Wisotzki, Leclercq, Guo, Trainor, Kozlova.
- **Simulation** — look into TNG or other simulations to predict both the flux and kinematics near R_vir to see the break.

---

## 6. Conclusions



---

## 7. Future work

- Two-halo splits with more data.
- MCMC on the flux fit — outer slope as a result, not an input.
- TNG50 comparison for an environment split and a core-luminosity split.
- JWST morphology.
- z~1.5 zero-point driver.

---

## 8. Figures

- Radial spectra stack
- Brightness profile + PSF
- Two-component fit + crossover
- Centroid vs. radius
- B/T vs. radius
- Convergence figure — all diagnostics on one radius axis (doesn't exist yet)
- Niemeyer comparison
- Injection recovery
- Split panel grid
- Photon budget / f_esc vs. radius
- Outer amplitude vs. density at several scales

---

## 9. Parking lot

- LIS-line stacked outflow velocity.
- Our own LAEs with systemic z as an internal zero point for the LAE catalog.
- Literature claims register — pull testable claims, test against VDFI.
- Fixed-window flux systematic: a broadening line loses more of itself with radius.
- Cosmic-web imaging comparisons not unit-matched to our flux convention.

---

## See also

- `../../01-lya_halos/docs/GOALS.md`, `referee-objection-register.md`
- `../workstream-roadmap.md`
- `discussion-draft-notes.md`
- `lyahalos_draft_7-27-26.tex`
