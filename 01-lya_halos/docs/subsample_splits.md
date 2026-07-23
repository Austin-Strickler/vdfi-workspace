# Subsample Splits — Lyα Halos

Candidate subsample splits for the stacked Lyα halo measurement, with the physical motivation behind each.

Splitting the stack loses statistical power (roughly √N per split), so each candidate here earns its place by testing a specific kinematic prediction or isolating a physical mechanism a full-sample stack would wash out.

**Paper assignment.** Paper 1 (`01-lya_halos`) takes the splits with a clear kinematic prediction, prioritized toward whichever are most likely to yield a detectable signal — this is the discovery paper. Paper 2 (`02-lya_escape`) takes the splits that are fundamentally about flux curves and total photon budget rather than kinematics.

---

## Paper 1 (`01-lya_halos`) — kinematics, prioritized for discovery

### 1. Mass and/or SFR-normalized mass

- High-mass should result in larger dark-matter halos
- Plausibly more gravitational cooling (unless shock-heating inefficiencies?)
- Comparing a presumably intrinsic photon budget to large-scale flux distributions
- Kinematic comparisons with accretion

### 2. ΣSFR (SFR surface density) and/or mass-normalized ΣSFR

- ΣSFR drives feedback-launched outflows, which open the dust-clearing channels Lyα photons escape through — directly measurable with current data
- Intense, bursty star formation vs. not

### 3. Lyα velocity offset — self-split, core-measured

- Split the sample by each galaxy's own core-measured Lyα centroid offset (blueshifted / redshifted / systemic)
- Direct test of the accretion (blueshift) vs. outflow (redshift) interpretation of the centroid-vs-radius result
- Not statistically independent of that headline result — this is a decomposition of it, not a separate discovery

### 4. Redshift bin, mass-normalized — z~2 vs. z~3

- Doubles as a systematics check, since the two bins carry different magnitude cuts (H<24.5 vs. H<25)
- Tests whether halo kinematics and extent evolve over the ~1 Gyr separating the bins

### 5. Mass-normalized dust attenuation — e.g. Balmer decrement, UV slope β

- Lighter-weight, kinematics-focused proxy for the same photon-budget question the escape-fraction split (#10) answers more fully
- Stays in Paper 1's kinematics territory without requiring Paper 2's escape-fraction apparatus

### 6. Environment / satellite clustering

- Tests the satellite-clustering explanation for a redshifted centroid
- Pull from Chartab+2020 CANDELS catalogs for overdensity numbers per galaxy

### 7. Lyα equivalent width (EW)

- Halo strength and extent scale with EW largely independent of broadband SFR/mass (Steidel+2011, Momose+2014)
- Continuous-variable complement to the discrete classification scheme in #11

### 8. Orientation / inclination

- Radiative-transfer models predict the observed kinematic profile shape depends on viewing angle relative to outflow/scattering geometry (Verhamme+2012, Zheng & Wallace 2014)
- Requires a morphological inclination proxy from imaging

### 9. Peak separation & relative peak height (double-Gaussian decomposition)

- Fit the core Lyα profile with a double Gaussian, fitting peak height ratio and peak separation as free parameters
- Split on peak separation (proxy for HI column density / scattering path length) and on relative peak height (proxy for outflow vs. inflow asymmetry, red-peak dominance)
- More free parameters per fit means more noise per bin — needs better core data, but the payoff is a direct kinematic handle on HI column density and outflow/inflow asymmetry

### 10. Burstiness — Hα vs. UV/SED-timescale SFR (or τ from SFH fitting)

- Bursty star formation drives episodic feedback; instantaneous (Hα, ~10 Myr) vs.
  time-averaged (UV/SED, ~100 Myr–1 Gyr) SFR mismatch is the standard burstiness proxy
- Prediction: bursty subsample shows a larger/more redshifted centroid offset (fresh
  outflow launch) vs. steady star-formers at fixed total SFR/mass — a direct test of
  whether kinematics track instantaneous or time-averaged energy injection
- Caveat: matched radii are currently PSF-dominated and burstiness timescales only reach
  ~0.3 R_vir — too shallow to speak to the R_vir-scale transition that's Paper 1's
  headline result with the current data; Paper 2's incoming ~2× data is what unlocks this
  (already scoped there as "Burstiness ↔ kinematics")

### 11. Half-light radius, mass-normalized

- Tests whether the halo scale length (h1) tracks a purely gravitational/kinematic scale
  (R_vir, mass) or a baryonic/morphological one — do more extended galaxies at fixed mass
  produce a more extended halo, independent of mass itself?
- Also a direct handle on the "population mixing (LBG+MELG)" blindspot from the Discussion
  draft: could show whether h1 undershooting Steidel 2011 is a morphology effect rather
  than a sample-composition one
- Prediction: larger R_half at fixed mass → larger h1, more diffuse core term; compact
  galaxies → tighter profile, cleaner test of the pure gravitational prediction
- Lightweight, similar effort to the dust-attenuation split (#5) — needs an imaging-based
  size measurement, not a new apparatus

---

## Paper 2 (`02-lya_escape`) — flux / photon budget

### 12. Hβ-normalized escape fraction

- Intrinsic photon budget normalized
- Test differences between high and low escaping systems in kinematics and extent

### 13. Core classification type

- AGN, LAE, emitter, null, absorber → if too low signal, split into AGN, emitter, no emission
- Test the Trainor+2025 findings that even LAA are net-emitters
- Determine kinematics of each sample and possible emission mechanics (AGN = fluorescence...)

### 14. Exponential scale length of Lyα, PSF-aware

- PSF-deconvolved fit for the Lyα-emitting region's own scale length, rather than
  borrowing Steidel's continuum-based 3–4 kpc figure — the same "fit our own continuum
  scale length from CFHT-LS imaging" upgrade flagged as future work in the Discussion
  draft (§3), but for the Lyα profile itself rather than continuum
- Note: the known PSF forward-model area bug in `fitting.py` (returns 2πr·SB, not
  surface brightness — needs dividing by annulus area, not bin width) sits directly in
  this measurement's path; fix that first
- Prediction: a clean PSF-corrected h1 is the input the core-to-halo escape-fraction
  machinery (§3 Methods, Paper 2) actually needs — this is infrastructure Paper 2's
  headline result depends on, not an optional split

---

## See also

- `GOALS.md` — the broader (less detailed) subsample-split candidate list this doc expands on
- `TODO.md` (workspace root) — subsample splits and their physical predictions in the literature is tracked there as the top-priority open item
- `../specs/subsample-derived-properties.md` — turns each split's comparison plot into a row of derived numbers (core/halo flux, core/halo velocity) collected into one paper-ready table, not just a figure
