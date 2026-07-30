# LIS-line velocity measurement — cross-instrument calibration + outflow physics

**Status: v3, 2026-07-29. `lis.py` IMPLEMENTED and tested.**

**v3 changes.** `utils_lya_halo/lis.py` written, registered in `guide.py` (new `"lis"` section, `check_guide()` clean), regression test at `code/tests/test_lis.py` (11 blocks, all passing). Injection–recovery corrected two of this spec's own numbers — see the new **§1.2a**: the C II\* bias is **83 km s⁻¹** (v2 said 10–20) and the Si II 1260 blend is **42 km s⁻¹** needing **three** components (v1 and v2 both called it "mild, partly cancelling"). Also: an unmodelled partner inflates the fitted width by ~16%, so §3.1's σ_v check is only valid at ncomp > 1.

---

**Status of v2, 2026-07-29 (spec only, no code):** Companion to `wavelength-zero-point.md`, which this promotes from a demoted idea (its §6) to a live tier.

**v2 changes — three of my v1 arguments were wrong, one badly.** All from reading Weldon et al. 2022 (arXiv:2203.09539) in full rather than from search summaries:

1. **⚠️ I had the resolution comparison BACKWARDS.** Weldon §2.2: LRIS-B used the **400 lines/mm grism through 1″2 slits** → FWHM ≈ 7 Å, i.e. **~485–515 km s⁻¹** at the LIS lines. VIRUS is **5.6–6.3 Å → ~340–460 km s⁻¹**. **VIRUS is the sharper instrument by 10–20%.** My v1 §1 argument — "Topping's 'uncontaminated' designation doesn't transfer because they had better resolution" — is the reverse of the truth. See §1.
2. **The coadd recipe I recommended does not match theirs.** Weldon §3.5 build composites in **luminosity density with an unweighted average**, then normalize the composite. My v1 §2.4 said normalize *per galaxy first*. That is the right way to get an unbiased sample mean but the **wrong** way to match their number. See §2.4.
3. **The circularity risk is narrower than I made it.** Weldon §2.4: the LRIS-flow sample was selected to *require* z_sys from MOSDEF. All 155 have genuine MOSFIRE nebular redshifts. **The −60 ± 10 comparison is clean.** The circularity concern applies only to Topping's +32/+89 fill-in relations. See §2.1.
4. Also corrected: MOSDEF's third bin is **z = 2.95–3.80**, not 3.13–3.80 (Weldon §2.1). Coverage table redone — the z~3 bin delivers more than v1 claimed.
5. **The internal drift test is demoted** per Austin's objection, which was right. See §4.

**Net effect: the test is in better shape than v1 implied.** Comparable resolution, identical estimator family, identical systemic redshifts, and — per Weldon's own result — outflow velocity is only weakly coupled to galaxy properties, which largely defuses the sample-matching worry. Austin's read that "it is generally a valid comparison to do" is well supported.

**Reference method:** Topping et al. 2020 §2.2–2.3. **Reference result:** Weldon et al. 2022.

---

## 0. What this test measures

Write ε_V for a VIRUS wavelength-scale error, ε_L for LRIS, ε_M for the MOSDEF redshift scale, v_out for the true mean LIS outflow velocity:

- **VDFI measures:** v_out + ε_V − ε_M
- **Weldon measured:** v_out + ε_L − ε_M = **−60 ± 10 km s⁻¹**

The difference is **ε_V − ε_L**; the MOSDEF term cancels algebraically. In v1 I treated that cancellation as a scope limit that stopped the test from answering the question. **That was too strict.** The quantity Austin needs is ε_V − ε_M, and it is recoverable because the other two terms are documented rather than unknown:

| Term | Evidence | Status |
|---|---|---|
| **ε_L air/vacuum** | Weldon §1, verbatim: *"All wavelengths are presented in the vacuum frame."* Table 1 note: rest wavelengths *"taken from the Atomic Spectra Database website of NIST."* | **closed** |
| **ε_L zero point** | Topping §2.2: re-reduced without background subtraction, measured sky-line centroids, *"shifted the wavelength solution zeropoint so that the sky lines appear at their correct wavelength values"* | **sky-line anchored** |
| **ε_L residual** | Topping §2.2 arc residuals: ~0.3 Å blue-side, ~0.035 Å red-side. At 4000–4950 Å the blue figure applies | **±18–22 km s⁻¹ floor** |
| **ε_M air/vacuum** | Weldon §2.1: systemic z from *"the strongest emission line, usually Hα or [O III] λ5008."* **λ5008 is the vacuum wavelength** — air is 5006.84. MOSFIRE R ~ 3000–3600 | **strongly indicated vacuum**; confirm against Kriek et al. 2015 |

So ε_L ≈ 0 ± 22 km s⁻¹ and ε_M ≈ 0, both from published statements. **ε_V is then the only free term, and this test measures it.** Austin's argument — "they use the same MOSDEF redshifts, so if we agree with them, either both are right or both are wrong identically, and identically-wrong requires MOSDEF itself to be broken" — is correct, and the citations above are what close the "both wrong identically" branch.

> **The 22 km s⁻¹ floor is real and should be quoted.** Detecting an 85 km s⁻¹ offset at >3σ is comfortable. Detecting 30 km s⁻¹ is not achievable by this route.

Note that the ~4 Å median sky-line shift Topping applied is ≈300 km s⁻¹ at 4000 Å. That is the size of a correction they *made*, not an error bar — but it does say LRIS's raw solution drifted badly mask-to-mask, which is why the published one is empirically anchored.

---

## 1. Resolution and blends — the v1 argument, corrected

### 1.1 The instruments are close, and VIRUS is slightly sharper

**LRIS-B:** 400 lines/mm grism, 1″2 slits, dichroic at ~5000 Å (Weldon §2.2). The 400/3400 grism gives ~1.09 Å/pix and **FWHM ≈ 6.5–7 Å** at a 1″ slit; the 1″2 slit pushes it to ~7–8 Å. All three redshift lines land on the **blue** side for a z ≈ 2.24 sample (Si II 1260 → 4084 Å, C II 1334 → 4324 Å, Si II 1526 → 4947 Å).

**VIRUS:** `speclsf.py::VIRUS_FWHM_A = 5.6` Å observed, constant. Austin's measurement independently brackets this: the stacked rest-frame Lyα FWHM is **≈1.9 Å**, which at z = 2.3 is 6.27 Å observed. Back out the LSF:

| Assumed LSF (obs) | Implied residual z-scatter + intrinsic width |
|---|---|
| 4.70 Å (Hill+2021) | 310 km s⁻¹ — **implausible**, rules this value out |
| 5.60 Å (`speclsf.py`) | 211 km s⁻¹ |
| 6.00 Å | 136 km s⁻¹ |
| 6.27 Å | 0 — the hard ceiling |

**So the measured 1.9 Å brackets the VIRUS LSF between ~5.6 and 6.27 Å observed, and excludes 4.7 Å.** The config value is right, possibly ~10% low. Good — but note this constrains the LSF **only at Lyα's observed wavelength (3756–4389 Å)**. Si II 1526 sits at 4718–5511 Å, where nothing measures it, and constant-FWHM vs constant-R differ by ~20% there. **That uncertainty propagates directly into the blend models (§1.2) and is a named systematic.**

| Line | λ_obs (z=2.24) | LRIS-B (7 Å) | VIRUS (5.6–6.3 Å) |
|---|---|---|---|
| Si II 1260 | 4084 | 514 km s⁻¹ | **411–462** |
| C II 1334 | 4324 | 485 km s⁻¹ | **388–437** |
| Si II 1526 | 4947 | 424 km s⁻¹ | **339–382** |

### 1.2 Which means the blend argument partly cancels

| Pair | Δv | / LRIS FWHM | / VIRUS FWHM | |
|---|---|---|---|---|
| Si II 1260.42 / S II 1259.52 | 214 | 0.42 | 0.46–0.52 | blended in **both** |
| C II 1334.53 / **C II\* 1335.71** | 265 | 0.55 | 0.61–0.68 | blended in **both** |
| Si II 1526.71 / Si II\* 1533.43 | 1320 | 3.11 | 3.46–3.89 | **clean in both** |

**Every blend is *more* blended for LRIS than for VIRUS.** Two consequences, both favourable:

- Topping's "uncontaminated" designation was made at *lower* resolution than VDFI has. Their −60 ± 10 **already carries whatever C II\* fill-in bias exists, and more of it.** In a differential comparison the bias **partially cancels**, and the residual runs in the direction that VDFI is *less* biased — i.e. toward VDFI measuring a slightly *less* blueshifted Δv_LIS than they did, for purely instrumental reasons.
- My v1 §2.3 "resolution-dependent centroid correction" **flips sign**: VIRUS is the sharper instrument, so the correction goes the other way, and it is a ~10–20% resolution difference rather than the factor of two I implied.

**What still stands:** the C II\* contamination is *not zero* and is *not fully cancelling*, because the resolutions differ by 10–20% and the fill-in scales nonlinearly with the blend ratio. So: **still model the C II/C II\* pair rather than fitting a single Gaussian.** Si II 1526 remains the cleanest anchor, and the reason to lead with it is now robustness rather than necessity.

### 1.2a ⚠️ Measured blend biases — and v1/v2's "mild" call on Si II 1260 was wrong

Injection–recovery on synthetic stacks at the VIRUS LSF (`code/tests/test_lis.py`, dv injected = −60 km s⁻¹, partner depths 23–40% of the main line):

| Line | ncomp=1 | at `ncomp_recommended` | **blend shift** | components |
|---|---|---|---|---|
| C II 1334 | −143.2 | **−59.9** (2) | **+83.3** | C II 1334, C II\* 1335 |
| Si II 1260 | −101.9 | **−60.1** (3) | **+41.8** | Si II 1260, S II 1259, Si II\* 1264 |
| Si II 1526 | −56.7 | **−60.0** (2) | **−3.3** | Si II 1526, Si II\* 1533 |

Three corrections to earlier versions of this spec:

1. **The C II\* bias is ~83 km s⁻¹, not "~10–20 km s⁻¹."** It is the same size as the air/vacuum effect being hunted and in the same direction. v2's downgrade of this from "fatal" to "a small residual" went too far — what cancels against LRIS's coarser resolution is only *part* of it.
2. **Si II 1260 is NOT a mild blend.** Both v1 and v2 called the S II 1259.52 contamination "smaller, partly cancelling." It is **−42 km s⁻¹** at a plausible S II strength, and it needs **three** components (S II 1259 *and* Si II\* 1264), not two — a 2-component fit still leaves ~5 km s⁻¹. Since Si II 1260 is the line that sits at Lyα's observed wavelength (§1.3), this matters more than the C II case.
3. **Si II 1526 is confirmed clean** at −3.3 km s⁻¹, vindicating it as the anchor.

**How to report this.** `ncomp=1` is the *estimator-matched* number, because that is what Topping and Weldon actually fit; `ncomp_recommended` is the *unbiased* number. **Quote both** — `lis.compare_ncomp` produces exactly this table — and treat the shift as a systematic, not a correction to apply silently. The partner depths are assumed, so the shifts scale with the true fine-structure strength in the real stack, which the fits themselves measure.

**Also verified: an unmodelled partner inflates the fitted WIDTH.** Si II 1526 with Si II\* present recovers FWHM 668 km s⁻¹ at ncomp=1 versus an injected 577 (+91 km s⁻¹, +16%), and 577.2 exactly at ncomp=2. So **§3.1's σ_v ≈ 124 km s⁻¹ galaxy-scatter check is only valid at ncomp > 1.** At ncomp=1 the excess width is partly swallowed fine structure, not sample scatter.

**Minor, worth knowing:** Weldon's Table 1 red continuum window for Si II 1526 is **1535–1540 Å**, and Si II\* λ1533.43 reaches 1534.97 Å at 2σ under a 6 Å VIRUS LSF. The fine-structure emission just clips the blue edge of their red window, tilting the local continuum up and biasing the fitted trough centroid slightly **redward** — i.e. *against* a spurious blueshift. Worse for LRIS than VIRUS, so again partly cancelling. Low severity; note it, don't chase it.

### 1.3 Coverage — corrected bins

VIRUS band 3470–5540 Å. MOSDEF bins z = 1.37–1.70, **2.09–2.61**, **2.95–3.80** (Weldon §2.1).

| Line | z~2 (2.09–2.61) | λ_obs | z~3 (2.95–3.80) | λ_obs |
|---|---|---|---|---|
| Si II 1260.42 | **100%** | 3895–4550 | z ≤ 3.395 (**52%**) | 4979–5540 |
| C II 1334.53 | **100%** | 4124–4818 | z ≤ 3.151 (**24%**) | 5271–5540 |
| Si II 1526.71 | **100%** | 4718–5511 | out of band | — |
| *Lyα 1215.67* | *100%* | ***3756–4389*** | *z ≤ 3.557 (71%)* | ***4802–5540*** |

Three things this table says:

1. **All three of Topping's redshift lines are fully in band for the whole z~2 bin.** No coverage compromise at all there.
2. **The z~3 bin contributes more than v1 said** — Si II 1260 for 52% and C II 1334 for 24%, landing at 4979–5540 Å. That is exactly where z~3 Lyα falls (4802–5540), and it is the **only** handle above 5154 Å anywhere in the calibration program (cf. `wavelength-zero-point.md` §4.1, where no emission anchor reached that far). Do this part.
3. **Si II 1260 at z~2 lands on top of where z~2 Lyα lands.** 3895–4550 vs 3756–4389. That is the line to trust for the correction Austin actually wants to apply, even though Si II 1526 is intrinsically cleaner. This is the whole reason to measure more than one line — see §4.

**Lines blueward of Lyα** (Austin noted these are visible): Si II 1190.42/1193.28 sit at −6227/−5522 km s⁻¹ from Lyα, well outside the emission profile, and land at 3678–4308 Å at z~2 — good wavelength position. But they are a **pair separated by 719 km s⁻¹ = 1.5–1.7 VIRUS FWHM** (marginally resolved), they sit **inside the Lyα forest**, and Topping/Weldon do not measure them, so there is **no external comparison value**. Si III 1206.50 is at −2261 km s⁻¹, also forest-affected. **Verdict: useful as internal consistency and as forest-independent-ness checks, not as calibrators — no reference number exists for them.**

---

## 2. The comparison target — there are six numbers in these two papers and only one is right

This is the largest remaining risk, and it is bookkeeping rather than physics.

| Quantity | Value | What it is |
|---|---|---|
| **Δv_LIS, mean of per-object Gaussian centroids** | **−60 ± 10** | **← the target.** 149 objects with Δv_LIS, all with MOSDEF z_sys |
| Δv_LIS range | −510 to +380 | ⇒ σ_v ≈ 124 km s⁻¹ |
| Δv_LIS, **composite** | Fig. 2 green squares | **different estimator** — see below |
| v_80 | −428 ± 29 (median) / −468 ± 29 | blue-wing measure, **not a centroid** |
| v_max | −574 ± 29 (median) / −591 ± 29 | blue-wing measure, **not a centroid** |
| Δv_Lyα, mean | **+400 ± 23** (72 obj, range −190 to +950) | Lyα centroid, not LIS |
| Topping fill-in relations | −32 (z_LIS only), −89 (z_LIS + z_Lyα) | **regressions for objects lacking MOSDEF z** |

**Δv_LIS is defined explicitly** (Weldon Eq. 4): Δv_LIS = c(z_LIS − z_sys)/(1 + z_sys). Use exactly this.

### 2.1 Which number, and the corrected circularity picture

**Compare against −60 ± 10.** Weldon §2.4 states the LRIS-flow sample required *"z_sys determined from MOSDEF observations"* — so all 155 (149 with Δv_LIS) have genuine MOSFIRE nebular redshifts, and the mean is not contaminated by their own fill-in relations. **My v1 claim that this comparison is circularity-exposed was wrong.**

The circularity risk survives only in a narrow form:

- **Topping's +32 / +89 relations were built to assign z_sys to objects *without* MOSDEF.** Comparing against those, not against −60, would be partly circular. **v1 recommended −89 as the likely target on the grounds that Austin's sample is Lyα-detected. Withdraw that** — the −60 is the measured quantity on a clean subsample and the +32/+89 are fill-in regressions. Use −60.
- **On the VDFI side the risk is real and unchanged:** if any galaxy's "MOSDEF redshift" is not a nebular MOSFIRE detection (Weldon footnote 2 notes 45 of the 260 MOSDEF-LRIS targets had only a pre-MOSDEF spectroscopic or a photometric z), Δv_LIS for that object is partly measuring catalog assumptions. **Filter on nebular-redshift quality and record the surviving count.**
- **Match the AGN cut.** Weldon removed 12 galaxies for possible AGN via IR emission, X-ray, and [N II]/Hα. VDFI already has an AGN-excluded convention (R_vir = 75 kpc); align the criteria explicitly.

### 2.2 ⚠️ Their composite estimator is NOT their per-object estimator

This is the trap that matters most for a sub-stack measurement, and it is stated plainly in Weldon §3.5:

> *"Centroid velocities (Δv_LIS and Δv_Lyα) for the composite were measured in a similar way as v_80 and v_max (Section 3.4.1), using the normalised spectra to measure the **wavelength at the absolute minimum** of the detected LIS absorption trough or at the maximum of the Lyα peak."*

Versus per-object (Weldon §2.3, after Topping §2.3):

> *"these redshifts were obtained by **fitting lines with Gaussian functions and a quadratic function for the local continuum** and calculating the centroids of the Gaussians. Redshift uncertainties were determined by perturbing the LRIS spectra by the corresponding error spectra, refitting lines, and recalculating the centroids. Any LIS absorption lines with poor fits were excluded in the calculation of the average z_LIS, **typically based on two lines**, for a given galaxy."*

**Absolute trough minimum ≠ Gaussian centroid** on an asymmetric blue-winged profile — the minimum sits redward of the centroid. Austin is measuring sub-stacks, i.e. composites. So:

- **Do not compare a composite trough-minimum against the −60 mean-of-Gaussian-centroids.** Different statistics.
- **Measure both on every sub-stack.** Report the Gaussian centroid against −60 ± 10 (justified because for a fixed intrinsic profile the composite's Gaussian centroid equals the mean of individual Gaussian centroids — the velocity distribution enters as a convolution and moves no centroid), and the trough minimum against Weldon's Fig. 2 composite points.
- **The difference between the two estimators on the same sub-stack is itself the asymmetry measurement**, and it is free.

Note also **"typically based on two lines"** — z_LIS is usually a 2-line average, not 3. Match the multiplicity, and record which lines contributed.

### 2.3 Their continuum windows, verbatim — use these

Weldon Table 1, rest-frame vacuum, *"chosen to bracket the line and free of other spectral features"*:

| Line | λ (Å) | Blue window | Red window |
|---|---|---|---|
| Lyα | 1215.67 | 1195–1202 | 1225–1235 |
| Si II | 1260.42 | 1245–1252 | 1270–1275 |
| O I | 1302.17 | 1285–1293 | 1312–1318 |
| C II | 1334.53 | 1320–1330 | 1342–1351 |
| Si II | 1526.71 | 1512–1520 | 1535–1540 |
| Fe II | 1608.45 | 1590–1600 | 1616–1620 |

Two different continuum treatments appear in these papers and both are needed:

- **Normalization** (Weldon §2.2): *"a linear model between the average flux in two spectral windows"* — the Table 1 windows above. **Linear, two-window average.**
- **Redshift fitting** (§2.3): **quadratic** local continuum, fit **simultaneously** with the Gaussian.

The current pipeline does neither exactly: `measure.get_continuum_model` defaults to `cont_order=1` and is a **two-step** (fit continuum, then fit line). Replicate the simultaneous quadratic version as primary and keep the two-step as a sensitivity test. Also **confirm `gaussian_centroid` permits a negative amplitude** — Topping constrain it ≤ 0 for absorption.

**Error model — two distinct terms, do not conflate:**

- **Measurement error:** perturb by the error spectrum, refit, 100 iterations, take mean and standard deviation. This is Topping's recipe and `bootstrap_measurements` does *not* do it.
- **Sample error:** bootstrap over galaxies, plus the empirical scatter across sub-stacks.

### 2.4 Coadd recipe — corrected, and the correction reverses my v1 advice

Weldon §3.5, verbatim:

> *"The composite spectrum is computed by **shifting each galaxy's blue and red spectra into the rest-frame, converting to luminosity density, interpolating onto a new wavelength grid, and taking the unweighted average** of the spectra for all galaxies contributing to the composite."*

That is **luminosity-density space, unweighted average**, with normalization applied to the finished composite. **v1 told Austin to normalize each galaxy's continuum to unity *before* coadding. That is the wrong prescription for matching their number** — it produces an equal-weight sample mean, whereas theirs is effectively luminosity-weighted.

**Resolution:** primary = replicate their recipe exactly (luminosity density, unweighted average, normalize the composite). Secondary = the per-galaxy-normalized version as a sensitivity test. **The difference between them is the weighting systematic**, and it is small but measurable.

And it is likely *small*, because of a Weldon result that also defuses the whole sample-matching objection — see §2.5.

### 2.5 Sample matching is a smaller problem than v1 implied

Weldon's headline: outflow velocities are *"found to be marginally correlated or independent of galaxy properties, such as star-formation rate (SFR) and star-formation rate surface density (Σ_SFR),"* with **v_out ∝ SFR^0.24** and *no* significant Σ_SFR or Σ_sSFR correlation.

**If Δv_LIS barely depends on galaxy properties, a sample mismatch barely shifts the mean.** Austin's instinct that the VDFI sample might be usable as a whole, without replicating theirs object by object, is supported by their own analysis. Keep the property-matched sub-stack comparison as a *check* that this holds, but demote it from a prerequisite.

Their sample for reference: 8.61 < log(M\*/M☉) < 10.88, median 9.89; 0.32 < log(SFR) < 1.97, median 0.9; z = 1.42–3.48, median 2.24; ~7.5 hr LRIS integrations.

### 2.6 The Lyα numbers do not reconcile — stop citing them as confirmation

- **Weldon measured** Δv_Lyα = **+400 ± 23 km s⁻¹** on 72 galaxies with MOSDEF z_sys, range −190 to +950.
- **Topping's prescription** for z_Lyα-only objects implies **+153** (z ≤ 2.7) / +317 (z > 2.7).
- **VDFI core centroid:** ~+200.

**+153 and +400 cannot both be the mean Lyα offset of the same population.** One of them is a different quantity (plausibly peak-vs-centroid, or a subset selected on lacking MOSDEF z) and this spec cannot resolve which without reading Topping §2.3 in full. **Consequence: do not use either as confirmation of the +200 core centroid.** Concretely:

- The v1 objection stands and strengthens: averaging 153 and 317 to get 235 was not defensible, and now there is a *measured* number (+400) that neither supports nor is supported by +200.
- +400 is consistent with the classic LBG literature (Steidel 2010 found Lyα ≈ +360 to +450). So **+400 is the credible population value, and VDFI's +200 is well below it.**
- **The most likely reason is aperture, and it is interesting rather than alarming.** LRIS used 1″2 slits; VDFI uses a fiber IFU whose "core" captures far more of the spatially extended, preferentially blueshifted halo emission. That pulls the core centroid blueward of a slit measurement. **This is a real, statable result — but it also means the VDFI core centroid is aperture-dependent and not directly comparable to slit-based Lyα offsets at all.** Say so in Methods rather than inviting the comparison.

---

## 3. Architecture — sub-stacks of 10–30

Set by the S/N regime Austin reports: LIS troughs are clear in sub-stacks of ~10–30, not per-galaxy, not full-stack-only. Si II 1260 and Si II 1526 are both described as clearly visible, with fine-structure emission also present.

**What sub-stacks buy:**

1. Empirical error on the mean from scatter across sub-stacks, independent of any error model.
2. Cells in (line × redshift) for the §4 line-agreement check.
3. Field (COSMOS vs EGS) and epoch splits → the barycentric test (§3.2).
4. Property-matched comparison to Weldon as a check on §2.5.

**Reusable infrastructure — more than expected.** `config.rest_wave_min = 1100`, `rest_wave_max = 1700`, `rest_delta = 0.2` (`config.py` 208–213): **the rest-frame stacking grid already spans the entire LIS region.** `stack.py::rest_grid`, `stack_spec_values`, `build_core_stacks`, `concat_core_prods` should work as-is. Both z~2 and z~3 core products are already extracted — **no extraction latency**. New code: simultaneous quadratic+Gaussian fitter with the ≤0 amplitude constraint, the trough-minimum estimator, blend models, the 100× error-spectrum MC, and luminosity-density coadd.

**Where it lives.** New module `lis.py`, registered in `guide.py` with its own section, per the `fitting.py` / `uv_profile.py` precedent — both were unregistered for a long time and had zero `check_guide()` coverage as a result. Serves this test and split #15 in `../docs/subsample_splits.md` from one codebase.

### 3.1 Free bonus: the composite width

Composite width² ≈ LSF² + intrinsic profile² + (galaxy-to-galaxy velocity scatter)². Weldon's range implies **σ_v ≈ 124 km s⁻¹**. Recovering that from the excess width independently reproduces their distribution and validates the sample match **without using the centroid at all**.

> ⚠️ **Only valid at ncomp > 1.** Verified in `code/tests/test_lis.py`: an unmodelled fine-structure partner inflates the fitted FWHM by ~16% (668 vs 577 km s⁻¹ injected, for Si II 1526 with Si II\* present), because the single Gaussian widens to swallow the companion. At ncomp=1 the "excess width" is partly blend, not sample scatter, and this check would return a spuriously large σ_v. Run it on the multi-component fits.

### 3.2 Barycentric

Excess scatter beyond σ_v ≈ 124 km s⁻¹ is unmodelled systematic, and the leading candidate is a missing barycentric correction (`wavelength-zero-point.md` §1.4): up to ±30 km s⁻¹, varying with season and sky position, **non-rigid**. Keck reductions apply it as standard; whether VDFI's does is still unverified. COSMOS and EGS sit at very different RA, so sub-stacking by field and by epoch is a direct test, free with the same fits.

---

## 4. Drift — demoted, per Austin's objection, which was right

**v1 called the internal drift test "the half worth protecting." Austin disagreed:** the question is whether the full z~2–3 stack is offset from the MOSDEF scale by a *constant*, motivated by the outer-bin sign flip from redshifted core to blueshifted beyond R_vir. That is a constant-offset question, and drift is not what is being asked.

**He is right, and v1's prioritisation was unjustified for a reason I should have checked.** A rigid offset cannot produce the radial gradient (`wavelength-zero-point.md` §0); a drift can only do so **if the radial bins carry different effective observed-wavelength weighting.** I asserted drift mattered without checking whether they do.

**So check it directly — it is a ten-minute calculation on stacks already in hand.** Compute the effective redshift (equivalently, effective observed-λ) distribution per radial bin from the existing coadds:

- **Distributions identical across radius** → drift is *provably* irrelevant to the gradient, and only the constant matters. Drift work stops here.
- **Distributions differ** → quantify the drift, because a wavelength-dependent error then couples to radius.

**This check replaces the drift test as a deliverable**, and it should run before any drift modelling is scoped.

**What to keep, at zero cost.** Austin is measuring Si II 1260 (λ_obs 3895–4550) and Si II 1526 (4718–5511) anyway. Those sample the wavelength solution ~1000 Å apart, so **reporting both numbers gives the drift check for free** — no separate test, no extra cells, no dedicated design. Two reasons to report both rather than just the cleaner line:

1. **Si II 1260 is where Lyα is.** Its 3895–4550 Å range sits essentially on top of z~2 Lyα's 3756–4389 Å. Si II 1526 is clean but 1000 Å redward. Applying a Si II 1526-derived correction at Lyα's wavelength assumes no drift; quoting Si II 1260 alongside it tests that assumption instead of assuming it.
2. Adding the z~3 Si II 1260 cells (4979–5540 Å) extends the lever arm to the region where z~3 Lyα falls.

If the two lines agree within errors, quote a single constant and note that it is wavelength-independent across 3900–5540 Å. That is a stronger Methods sentence than a single-line result, and it costs nothing extra.

**Residual caveat if they disagree:** line-to-line differences also arise from optical depth — more saturated lines trace higher-column, faster gas — so disagreement is not automatically drift. Weldon's per-line values (if published per object) would separate the two. **Unverified: whether they publish per-line velocities.**

---

## 5. Systematics budget

| Term | Size | Status |
|---|---|---|
| ε_L — LRIS blue-side residual (0.3 Å arc) | **±18–22 km s⁻¹** | **hard floor**, sourced (Topping §2.2) |
| ε_L air/vacuum | 0 | **closed** — Weldon §1 states vacuum, NIST wavelengths |
| ε_M air/vacuum | ≈0 | strongly indicated ([O III] λ5008 = vacuum); confirm vs Kriek 2015 |
| Resolution difference LRIS↔VIRUS | 10–20%, **VIRUS sharper** | direction now known; compute the centroid consequence |
| C II 1334 / C II\* 1335 fill-in | **83 km s⁻¹** at ncomp=1 (measured, §1.2a) | ncomp=2 removes it; partial cancellation vs LRIS is only part of it |
| Si II 1260 / S II 1259 + Si II\* 1264 | **42 km s⁻¹** at ncomp=1 (measured, §1.2a) | needs **ncomp=3**; "mild" was wrong |
| Si II 1526 / Si II\* 1533 | **3 km s⁻¹** (measured, §1.2a) | confirmed clean — the anchor |
| VIRUS LSF at 4718–5511 Å | ±20% (const-FWHM vs const-R) | **unmeasured** — feeds the blend models |
| Estimator: trough-minimum vs Gaussian centroid | tens of km s⁻¹ | §2.2 — measure both, compare like with like |
| Coadd weighting: luminosity-density vs per-galaxy-normalized | small (v_out ∝ SFR^0.24) | §2.4 — run both |
| Line-to-line optical depth | ? | measure; also a physics result |
| Barycentric, if unapplied | up to ±30 km s⁻¹, non-rigid | field/epoch sub-stacks — §3.2 |
| Sample property mismatch | small (§2.5) | check, don't assume; **demoted from v1** |
| Statistical (sub-stack scatter) | ~25 galaxies gives ±25 km s⁻¹ at σ_v = 124 | **not the limiting term** |

**Sampling statistics are the least important term.** Do not grow the sample; spend the effort on the estimator match (§2.2) and the coadd recipe (§2.4).

---

## 6. What would completely ruin it

Reordered from v1 given what the paper says.

1. **Comparing the wrong pair of numbers** (§2, §2.2). Six candidate values, including v_80/v_max at −428/−574 and Δv_Lyα at +400. Comparing a composite trough-minimum to the −60 mean-of-Gaussian-centroids is the specific, easy, invisible version of this error.
2. **Non-nebular redshifts on the VDFI side** (§2.1). Filter on quality flags; record the count.
3. **Coadding in the wrong space** (§2.4). Luminosity-density unweighted average is their recipe; anything else is a different statistic.
4. **Fitting C II 1334 as a single line** (§1.2). Now a ~10–20 km s⁻¹ residual rather than fatal, but it biases blueward — *toward* a false positive.
5. **Quoting a constant from Si II 1526 alone** (§4). It sits 1000 Å from where z~2 Lyα falls. Report Si II 1260 alongside it.
6. **Treating the result as closing everything** (§0). It constrains ε_V given the documented ε_L and ε_M. If either document check fails, the interpretation changes.

---

## 7. Task order

| # | Task | Cost | Note |
|---|---|---|---|
| 1 | **Effective observed-λ distribution per radial bin** | **~10 min, existing stacks** | **Do this first.** Decides whether drift can matter at all — §4 |
| 2 | MOSDEF nebular-redshift quality audit; count usable galaxies per bin | Hours | Gates the comparison — §2.1 |
| 3 | Read Topping §2.1–2.3 in full: confirm LRIS grating/slit resolution, reconcile +153 vs +400, check for per-line and per-object published values | Hours | Resolves three open items at once |
| ~~4~~ | ~~`lis.py`~~ | — | ✅ **DONE (v3)** — `utils_lya_halo/lis.py`, registered in `guide.py`, tested |
| ~~5~~ | ~~Blend models~~ | — | ✅ **DONE (v3)** — `ncomp`/`compare_ncomp`; biases measured in §1.2a |
| 6 | Δv_LIS per line for z~2; 2-line average matched to Topping; **compare to −60 ± 10** | — | The answer |
| 7 | z~3 Si II 1260 (z ≤ 3.395, 52%) at 4979–5540 Å | Low | Only handle above 5154 Å — §1.3 |
| 8 | Field/epoch sub-stacks; composite width → σ_v vs 124 km s⁻¹ | Free with 6 | Barycentric + sample-match check — §3.1, §3.2 |
| 9 | Property-matched comparison to Weldon | Low | Confirms §2.5 rather than assuming it |

Tasks 1–3 need no new code and should run in parallel with 4.

---

## 8. Unverified claims in this spec

- LRIS-B 400/3400 FWHM ≈ 6.5–7 Å at 1″ slit is from instrument documentation via search, not from Topping §2.1 directly. The 1″2 slit scaling to ~7–8 Å is my estimate. **Confirm from Topping §2.1** — this now underpins the whole §1.2 cancellation argument.
- Whether MOSDEF's published redshifts are vacuum-derived (strongly indicated by [O III] λ5008; confirm vs Kriek et al. 2015).
- Whether the MOSDEF catalog carries a usable nebular-redshift quality flag.
- Whether Topping/Weldon publish per-object or per-line z_LIS, or a machine-readable composite.
- Why Topping's +153 and Weldon's +400 differ (§2.6). Unresolved and it matters for any Lyα comparison.
- Whether VDFI's reduction applies a barycentric correction (also open in `wavelength-zero-point.md` §1.2c).
- The VIRUS LSF at 4718–5511 Å (const-FWHM vs const-R), which feeds the blend models.

---

## See also

- `wavelength-zero-point.md` — parent spec. §0 (rigidity), §1 (Tier 0 audit, barycentric), §4 (the z~1.5 anchor test), §6 (the demoted-LIS argument this supersedes)
- `../docs/subsample_splits.md` — split #15, the physics half
- Weldon et al. 2022, MNRAS 515, 841 (arXiv:2203.09539) — §2.1–2.4, §3.4.1, §3.5, Table 1
- Topping et al. 2020, MNRAS 495, 4430 — §2.1–2.3, the method of record
- `../code/utils_lya_halo/` — `config.py` (`rest_wave_min/max`), `stack.py`, `measure.py`, `speclsf.py`
