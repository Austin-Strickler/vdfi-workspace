# Wavelength & velocity zero point — MOSDEF ↔ VDFI

**Status: v2, 2026-07-29.** Test plan, not results. Nothing here has been executed. Written to be worked top-to-bottom; each tier is designed to be abandonable the moment an earlier tier returns a verdict.

**v3 changes (2026-07-29, later same day).** Three corrections from Austin, two of which change the plan:

1. **Tier 1 (sky lines) is not executable.** VDFI is not HETDEX and the pre-reduced frames this spec assumed are not available. §2 is **blocked**, not merely deprioritised, and it was the free-and-better alternative that justified demoting the LIS test in §6.
2. **The z~3 MOSDEF catalog is in scope**, not just z~2. §4.1's coverage discussion was written as if z~2 only.
3. **§6's central factual claim was wrong.** LIS lines do *not* sit 150–200 km s⁻¹ blueward of systemic for this comparison — that is the LBG-composite figure (Shapley 2003, Steidel 2010). On MOSDEF galaxies against MOSFIRE nebular redshifts it is a few tens of km s⁻¹ (Topping 2020: −32 / −89; Weldon 2022: −60 ± 10). The demotion in §6 rested partly on that wrong number and partly on the availability of Tier 1. **Both supports are gone.**

**v3.1 addenda (2026-07-29, after reading Weldon 2022 in full).** Two corrections that affect *this* spec, not just the LIS one:

- **MOSDEF's third redshift bin is z = 2.95–3.80**, not 3.13–3.80 as §4.1 states (Weldon §2.1: bins are 1.37–1.70, 2.09–2.61, 2.95–3.80). §4.1's z~3 discussion should be re-derived.
- **Tier 0(b) is largely closed from the literature.** Weldon §1, verbatim: *"All wavelengths are presented in the vacuum frame,"* with Table 1 rest wavelengths *"taken from the Atomic Spectra Database website of NIST."* And §2.1 gives MOSDEF systemic redshifts from *"the strongest emission line, usually Hα or [O III] λ5008"* — **λ5008 is the vacuum wavelength** (air is 5006.84). So both the LRIS and MOSDEF sides are on vacuum by published statement. **The remaining air/vacuum risk is entirely on the VIRUS side, §1.2(c)** — which is where this spec already said the real risk was.

**Consequence:** the LIS test is promoted from "removed from the calibration path" to a live tier with its own spec — **`lis-velocity-calibration.md`**. It is *not* a replacement for §4: it measures ε_VIRUS − ε_LRIS, not the MOSDEF tie (see that spec's §0). It does, however, recover most of Tier 1's information content, and its internal drift test (that spec's §4) needs no external reference at all.

**v2 changes (same day).** §4 rewritten. v1 recommended He II λ1640 on the z~2 science sample as the preferred cross-instrument test; **that is superseded** — He II is too weak, and a single faint line cannot test for wavelength drift. Austin confirmed **291 MOSDEF galaxies in the z~1.5 window across COSMOS + AEGIS, ~200 expected to survive coverage cuts.** §4 now specifies a **four-anchor joint fit** on that sample (He II 1640, O III] 1661/1666, Si III] 1883/1892, C III] 1907/1909), which returns the offset *and* the drift from one extraction. New §4.5 adds the **C III] doublet-ratio systematic** — a ≤62 km s⁻¹ centroid error that v1 did not identify and that would have been invisible in a single-Gaussian fit. New §4.6 covers **Si II\* fine-structure emission** as a same-sample secondary check.

**Related:** `systematics-and-extraction.md` (the general register; this is a systematic that register does not currently carry), `../docs/GOALS.md` (the absolute-frame claim this protects), `../../docs/workstream-roadmap.md` (where this sits in the wider plan).

---

## Goal

Establish, with a quotable number and a stated uncertainty, whether there is a **systematic velocity offset between the MOSDEF systemic-redshift scale and the VDFI/VIRUS wavelength solution**, and if so, correct for it.

This exists because the absolute velocity frame is the project's entire methodological advantage. Every competitor (Niemeyer 2022, Davis 2023, McKay 2026, Cooper 2026, Guo 2024) measures Lyα velocities self-referenced to Lyα. VDFI claims to say where the gas sits *in absolute velocity*. A referee is entitled to ask how the two wavelength scales were tied together, and right now the answer is "they weren't — we assumed they agree."

### Success criteria

- A number: Δv_sys (MOSDEF frame − VDFI frame) with an error bar, or an upper limit.
- A statement of whether it is **constant** or **wavelength-dependent**, since those have different consequences (§1.3).
- One paragraph of Methods text that a referee reads and stops asking.

---

## 0. What is and isn't at risk

**Read this before deciding how much to panic.**

A wavelength-solution or redshift-convention error is **rigid**: it moves every radial bin by the same Δv. It therefore:

- **Cannot** manufacture the headline result. The headline is a *gradient* in centroid across R_vir — a differential between bins. A constant offset cancels out of it entirely.
- **Can** invalidate the absolute statement ("the outer halo is blueshifted *relative to systemic*"), which is the specific claim the absolute frame exists to support, and which drives the whole accretion-vs-outflow discussion.

So the damage is bounded, and the bound is worth stating explicitly in the paper: *the transition survives any rigid zero-point error; the sign of the outer-bin velocity does not.* Say it before a referee says it for you.

**One exception to "rigid."** If the wavelength solution has a *drift* with λ rather than a constant offset, it is not rigid across the sample, because galaxies at different z put Lyα at different observed λ. That would smear rather than shift, and could in principle interact with radius if the radial bins have different effective z-weighting (they might: outer bins draw on more fibers and, after AGN/coverage cuts, possibly a different galaxy mix). §1.3 and §4.3 test for this. Do not assume rigidity — measure it.

---

## 1. Tier 0 — the convention audit (do this first, it costs hours)

**Hypothesis: someone, somewhere, is mixing air and vacuum wavelengths.**

### 1.1 The magnitude, computed

Standard-air refractive index (Edlén 1966), evaluated across the VIRUS band (`config.wave_min = 3470`, `wave_max = 5540`):

| λ_air (Å) | n_air | λ_vac − λ_air (Å) | Equivalent Δv (km s⁻¹) |
|---|---|---|---|
| 3500 | 1.0002861 | 1.001 | **85.8** |
| 4000 | 1.0002827 | 1.131 | **84.8** |
| 4400 | 1.0002809 | 1.236 | **84.2** |
| 5500 | 1.0002778 | 1.528 | **83.3** |

**Three things to notice:**

1. The effect is **≈ 84–86 km s⁻¹** and, in velocity, is nearly **flat across the whole VIRUS band** — a 2.5 km s⁻¹ variation from 3470 to 5540 Å. So an air/vacuum error is very close to a pure rigid offset, which makes it hard to detect from internal consistency alone and easy to detect from an external comparison.
2. It is **the same size as the effect being investigated**, and comparable to the entire claimed MOSDEF redshift precision (~60 km s⁻¹, itself unverified — see §5).
3. **The sign is the sign of the anomaly.** λ_vac > λ_air. If a VIRUS wavelength axis is in air and is treated as vacuum, every observed wavelength is under-stated by ~1.1 Å, the line appears at a bluer wavelength than truth, and the inferred velocity is **blueshifted by ~85 km s⁻¹ uniformly**. That is the direction and roughly the magnitude of the large-radius blueshift.

> ⚠️ **This is not evidence that the blueshift is an artifact.** It is evidence that the hypothesis is *live* and cheap to kill. Kill it before building any physical interpretation on the outer-bin sign.

### 1.2 What to actually check, in order

**(a) The rest wavelength — probably fine, check anyway.** `config.py` line 46 declares `LYA_REST = 1215.67  # vacuum`. That is correct and not the risky end: rest-UV wavelengths are always quoted in vacuum because the air-index formulae are not defined below ~2000 Å. Confirm nothing downstream re-derives it.

**(b) The MOSDEF side.** MOSDEF redshifts are MOSFIRE near-IR nebular (Hα or [O III], per register §9). Confirm from the MOSDEF survey/data-release papers whether the published redshifts are derived against **vacuum** rest wavelengths (near-IR spectroscopy conventionally is, and Hα 6564.61 vs. 6562.80 is exactly the vacuum/air pair — if a catalog quotes 6562.8 it is using air). Getting this backwards is a 82 km s⁻¹ error at Hα.

**(c) The VIRUS side — the actual risk.** Trace the wavelength solution through the reduction (`utils_python/`: `do_reductions.py`, `virusraw.py`, `get_wavelength_shifts.py`, `virusobs.py`) and answer:

- What line list was the arc/wavelength solution fit against, and is that list in air or vacuum?
- Is a vacuum conversion applied anywhere between the arc solution and the `WAVE` HDU?
- Is a **barycentric / heliocentric** correction applied? (See §1.4 — this is a second, independent ~±30 km s⁻¹ term that is easy to forget and is *not* rigid.)
- Does the HETDEX/VIRUS published data-release documentation state the convention? If HETDEX states it explicitly, that is citable and closes (c) without reading code.

**(d) Anything in `utils_lya_halo` that converts wavelength to velocity.** `measure.py::to_rest_frame` and the centroid family. Confirm a single convention is used and that no module silently re-applies one.

### 1.3 Pass / fail

- **All-vacuum, consistently, with barycentric applied** → Tier 0 passes. Record it in Methods in one sentence and move to Tier 1 to *verify empirically* rather than to *find a bug*.
- **Any air/vacuum inconsistency found** → apply the correction, re-run everything downstream of the centroid, and note in `CHANGELOG.md`. Tiers 1–3 then become the confirmation that the fix worked, which is a much happier reason to run them.
- **Cannot determine from code or documentation** → escalate straight to Tier 1, which measures the answer empirically regardless of what the code says. This is the likely outcome and it is fine.

### 1.4 The barycentric term — a second, non-rigid offset

Separate from air/vacuum, and easy to overlook. Earth's orbital motion projected onto a target is up to **±30 km s⁻¹**, varies with time of year, and varies with sky position. So:

- If **no** barycentric correction is applied, exposures taken in different seasons carry different offsets. Across a multi-year survey, this averages down toward a small residual for a large stack — but it *broadens* the stacked line, and it does not average to zero for any individual galaxy or for a subsample split by observing epoch.
- COSMOS and EGS sit at very different RA. If the correction is missing, the two fields carry **systematically different** offsets, which would show up as a COSMOS-vs-EGS velocity discrepancy. **That is a free internal test** and it is worth running the moment EGS is folded in (roadmap item 5b in the register): if the two fields disagree in centroid by tens of km s⁻¹ with no astrophysical reason, suspect barycentric before suspecting the sky.

**Action:** add "per-field centroid comparison" to the EGS fold-in, framed as a barycentric/wavelength-solution check rather than only as a depth check.

---

## 2. Tier 1 — sky lines (⛔ BLOCKED — data not available)

> ⛔ **Blocked as of 2026-07-29.** This tier assumed access to pre-reduced VDFI frames from which sky-line centroids could be measured. **VDFI is not HETDEX and that data is not in hand.** Everything below is retained as the design to execute if and when the frames become available — it is still the best version of this test — but it cannot currently be run, and nothing downstream should be planned as if it can.
>
> **Substitute:** `lis-velocity-calibration.md` §4 recovers the Δv-vs-λ drift curve across ~3900–5540 Å from galaxy absorption lines instead of sky lines, with a worse absolute zero point but a real lever arm on drift. Notably, it also inherits a *sky-line-anchored* zero point indirectly, because LRIS's published wavelength solution was itself zeropointed on sky lines (Topping 2020 §2.2).

**This is the test that should have been in the register from the start.** It uses data already on disk and answers "is the VIRUS wavelength solution where it says it is" without touching astrophysics.

### 2.1 Method

1. From reduced frames, measure centroids of bright, isolated night-sky emission lines across 3470–5540 Å.
2. Compare against a **published vacuum sky-line list** (e.g. the UVES/Hanuschik or Osterbrock atlases — note which convention the atlas is published in; several are air, which is exactly the trap being tested for).
3. Plot residual Δv vs. λ.

### 2.2 What the result looks like

| Residual pattern | Interpretation |
|---|---|
| Flat, consistent with zero | VIRUS solution is where it claims. Tier 0(c) closed empirically. |
| Flat at **≈ −85 km s⁻¹** (or +85) | Air/vacuum error, confirmed, with the sign telling you which way. |
| Sloped with λ | Wavelength-solution drift. **Not rigid** — see §0. Must be modelled, not just subtracted, because galaxies at different z sample different λ. |
| Scattered, no pattern | Solution noise; quantify the scatter as the per-exposure wavelength systematic and carry it in the error budget. |

### 2.3 Why this is not sufficient on its own

Sky lines validate VIRUS against a laboratory scale. They say **nothing** about whether MOSDEF's redshift scale agrees with that laboratory scale. If both instruments are internally correct but MOSDEF's redshifts carry their own offset, sky lines will look perfect and the problem will remain. That is what Tier 3 is for.

### 2.4 Bonus: sky lines are also a resolution check

While the lines are being fit, record their widths. That is a direct, per-wavelength LSF measurement, which feeds the existing LSF validation item (`speclsf.py`) and the ±3 Å window justification (register §1.2). Cheap to collect in the same pass; do not run this twice.

---

## 3. Tier 2 — stellar absorption on the field stars

The star extraction is already being re-run for the PSF work (register §3.6). Ride along.

**Method.** Stack the extracted stellar spectra and fit the centroids of strong z ≈ 0 features in band: Ca II H & K (3968.47 / 3933.66 air; 3969.59 / 3934.78 vacuum — note the pair differs by ~1.1 Å, i.e. the effect itself), the Balmer series (Hδ 4101.7, Hγ 4340.5, Hβ 4861.3 air), Mg b ~5175, and the G band ~4300.

**What it gives.** A second absolute wavelength check, independent of the sky-line list and of the sky's own emission physics, at ~5 points spread across the band — so it also constrains drift.

**What it does not give.** Same limitation as Tier 1: it validates VIRUS, not the MOSDEF tie-in.

**Caveats, in order of severity:**

- Stellar radial velocities are not zero. Galactic field stars have peculiar velocities of tens of km s⁻¹, so a *single* star is not a calibrator. A stack of ~50 per field averages toward the Galactic mean along that sightline, but that mean is not zero either and depends on the line of sight. Treat this as a **consistency check against Tier 1**, not as an independent absolute anchor.
- Stellar lines are broad and blended (especially Ca II H with Hε). Centroid systematics from blending can reach tens of km s⁻¹. Fit with a template, not a single Gaussian, or restrict to the cleanest features.
- Register §9 notes only ~50 stars per field or fewer, so statistical precision is modest.

**Verdict on priority: low.** It is nearly free because §3.6 is happening anyway, but it is the weakest of the four tiers and should not be allowed to consume time.

---

## 4. Tier 3 — the cross-instrument test (the one that actually answers the question)

**Tiers 1 and 2 validate VIRUS against the lab. Only this tier validates MOSDEF against VIRUS on the same galaxies.** If time is short, this is the tier to protect.

**Principle:** find a rest-frame feature that (a) falls in the VIRUS band, (b) sits at systemic velocity — i.e. is nebular and **non-resonant** — and (c) exists in galaxies that also have a MOSDEF near-IR redshift. Measure its centroid in VIRUS, compare to the MOSDEF redshift. The difference *is* Δv_sys.

### 4.1 The z~1.5 sample is a four-anchor calibrator, not a one-line one

MOSDEF's redshift bins are approximately **z ≈ 1.37–1.70**, **2.09–2.61**, **3.13–3.80**. Sample size confirmed by Austin 2026-07-29: **291 galaxies in the z~1.5 window across COSMOS + AEGIS, ~200 expected after VDFI coverage cuts.**

Across that redshift range, **four** independent systemic-velocity anchors fall in the VIRUS band:

| Line | λ_rest (vac) | Observed at z = 1.37–1.70 | Character |
|---|---|---|---|
| **He II** | 1640.42 | 3888 – 4429 Å | Recombination, non-resonant. Weak, but free in the same spectrum. |
| **O III]** | 1660.81, 1666.15 | 3936 – 4499 Å | Semi-forbidden. Separation 5.34 Å rest ≈ 960 km s⁻¹ → **resolved**. |
| **Si III]** | 1883.00, 1892.03 | 4463 – 5108 Å | Semi-forbidden. Separation 9.03 Å ≈ **1430 km s⁻¹ = 3.6× LSF FWHM → fully resolved.** λ1892 is a clean single line. |
| **C III]** | 1906.68, 1908.73 | 4519 – 5154 Å | Brightest rest-UV nebular feature after Lyα. Separation 2.05 Å ≈ 322 km s⁻¹ → **blended**; see §4.5. |

Excluded: ~~Mg II 2796/2803~~ (resonant, outflow-shifted — same disqualification as the LIS lines, §6); ~~[O II] 3727~~ (out of band for any MOSDEF bin); ~~C II] 2326~~ (only the lowest-z edge is in band).

**Why four anchors beats one bright one.** They span **3888 – 5154 Å** on a *single sample with a single extraction*. That measures the offset **and its wavelength dependence simultaneously** — which is the §0 rigidity assumption that otherwise has to be imported from Tier 1.

> **The transfer objection largely closes itself.** Lyα for the MOSDEF z~2 science bin (2.09–2.61) lands at **3757 – 4388 Å**. He II (3888–4429) and O III] (3936–4499) at z~1.5 land *in that same range*. So the calibration is measured **at the wavelength where the science Lyα actually falls**, not extrapolated to it. This was v1's main objection to the z~1.5 option and it is much weaker than v1 assumed.
>
> **Caveat for the z~3 bin.** Lyα at z = 3.13–3.80 lands at 5021 – 5837 Å, running off the top of the band above z = 3.556. Only Si III] and C III] (4463–5154 Å) reach into that range, and none of the anchors cover beyond 5154 Å. **The z~3 bin's red end is not directly calibrated** — state this rather than letting a reader assume otherwise.

### 4.2 The measurement: a joint fit of the 1880–1915 Å complex

**Fit Si III] 1883, 1892 + C III] 1907, 1909 together, with a single common velocity and a free C III] doublet ratio.**

- **Si III] λ1892 anchors the velocity** — it is resolved from λ1883 at 3.6 LSF widths, so its centroid carries no blend systematic.
- **C III] contributes the S/N** — it is the brightest feature in the region, typically 2–5× Si III].
- **The C III] ratio floats harmlessly**, because the velocity is not relying on it. This is the whole point of fitting them jointly rather than fitting C III] alone (§4.5).

Then **He II 1640 and O III] 1661/1666 as independent blue-end checks**, fit separately. Agreement between the blue pair and the red complex *is* the drift measurement.

### 4.3 Precision — you are not statistics-limited at N ≈ 200

Stacked centroid error scales roughly as σ_v ≈ σ_line / (S/N)_integrated, with σ_line set by the LSF (substitute the measured value from `speclsf.py`; ~5.6 Å FWHM at 4800 Å → **σ_line ≈ 149 km s⁻¹**).

| Target | Required σ_v | Required **stack** S/N | Implied per-galaxy S/N at N = 200 |
|---|---|---|---|
| 3σ detection of an 85 km s⁻¹ offset | 28 km s⁻¹ | **5.2** | 0.37 |
| 5σ detection | 17 km s⁻¹ | **8.7** | 0.62 |

And what better data buys:

| Stack S/N | σ_v |
|---|---|
| 5 | 30 km s⁻¹ |
| 10 | 15 km s⁻¹ |
| 20 | 7.5 km s⁻¹ |

> **A 5–6σ stacked detection of C III] in ~200 galaxies is a low bar.** Do not spend effort growing the sample — at N = 200 the statistical requirement is already met with margin. Spend the effort on §4.5 instead, because **the doublet systematic, not N, is what limits this measurement.**

Numbers above are analytic scalings for scoping. The reported error must come from the same bootstrap machinery used everywhere else, not from this table.

### 4.4 What to extract

These are **core** measurements (galaxy-centred, no radial bins), so this rides on `core.py::extract_core_spectra` rather than the full radial extraction — cheap by comparison. The z~1.5 sample has **never been extracted by this pipeline**, so scope before launching:

- **Footprint cross-match.** 291 candidates → confirm how many have usable VDFI exposure coverage. Austin's estimate is ~200.
- **Can the core path be pointed at a non-Lyα rest wavelength?** ⚠️ **This is the real scoping question and it is not cosmetic.** Several things in the Lyα configuration are wrong or meaningless for a z~1.5 target:
  - **Lyα is not in band at all** at z = 1.37–1.70 (1215.67 × 2.37 = 2881 Å, well below `wave_min = 3470`). Anything keyed to the galaxy's Lyα position is pointing off the array.
  - **The background annulus line mask** (`line_mask_halfwidth_A`, applied around the galaxy's observed Lyα) must be re-pointed at the C III]/Si III] complex or disabled.
  - **The continuum sidebands** (`cont_bounds`, defined in rest-frame Å around Lyα) are meaningless here. New sidebands are needed, chosen line-free around 1880–1915 Å and around 1640–1670 Å.
  - Audit anything else keyed to `LYA_REST` in the core path.
- **Consider a minimal purpose-built path.** This measurement needs only a galaxy-centred coadd and a stacked rest-frame spectrum in two narrow windows — no radial bins, no curve of growth, no PSF deconvolution. If re-pointing `extract_core_spectra` turns out to touch many Lyα assumptions, a small dedicated extraction may be faster and lower-risk than generalizing the production path. Decide this during scoping, not halfway through.

### 4.5 ⚠️ The C III] doublet ratio — a systematic that would not have been visible

**This is the thing that would have quietly compromised the measurement.** It was not in v1.

C III] λ1906.68 / λ1908.73 are separated by **2.05 Å rest = 322 km s⁻¹**, against a VIRUS LSF of roughly 400 km s⁻¹ FWHM. The doublet is therefore **blended but not negligibly so** — it presents as a single feature ~15% broader than the LSF. And the ratio **R = I(1907)/I(1909) is electron-density dependent**, running from ≈1.53 in the low-density limit downward as n_e rises.

The blend centroid moves with R:

| R = I(1907)/I(1909) | Blend centroid | Shift vs. low-density limit |
|---|---|---|
| 1.53 (low-n_e limit) | 1907.490 Å | reference |
| 1.40 | 1907.534 Å | **+7 km s⁻¹** |
| 1.20 | 1907.612 Å | **+19 km s⁻¹** |
| 1.00 | 1907.705 Å | **+34 km s⁻¹** |
| 0.80 | 1907.819 Å | **+52 km s⁻¹** |
| 0.70 | 1907.886 Å | **+62 km s⁻¹** |

**Fit a single Gaussian assuming the low-density limit when the true ratio is 1.0, and you inject +34 km s⁻¹** — 40% of the effect being searched for, in a form indistinguishable from it, with no diagnostic in the residuals.

**Why fitting R free does not fully solve it.** At 322 km s⁻¹ separation under a ~400 km s⁻¹ LSF, R is strongly degenerate with the centroid. Floating it converts a systematic into a larger statistical error rather than eliminating it. Given the §4.3 margin that trade is probably affordable — **but measure the inflation rather than assuming it.**

**Three mitigations, in order of preference:**

1. **Si III] λ1892 (§4.2).** Resolved, single, in the same fit. The velocity anchor never depends on R. This is why the joint fit is specified rather than a C III]-only fit.
2. **Constrain n_e externally.** [S II] λ6716/6731 lands in H band at z = 1.37–1.70 and MOSDEF covers it. A stacked [S II] ratio for the same ~200 galaxies gives a mean n_e → R to ≈±0.1 → residual ≈±10 km s⁻¹. Faint, but it is the clean closure if the free-ratio fit costs too much precision.
3. **Report the R-sensitivity as a systematic band** if neither of the above is decisive. It is a known, computable function — §4.5's table *is* the systematic, and quoting it is defensible.

**Do not report a C III] centroid from a single-Gaussian fit under any circumstances.**

### 4.6 Si II\* fine-structure emission — a same-sample secondary check

Austin notes clear fine-structure emission in the existing data (2026-07-29). Worth using, with its limits stated.

**Why these lines are in emission.** Si II absorbs from the ground state (λ1190.42, 1193.28, 1260.42, 1304.37, 1526.71); the excited ion can decay instead to the *upper* fine-structure level of the ground term, emitting **Si II\* λ1194.50, 1197.39, 1264.74, 1309.28, 1533.43**. That lower level is essentially unpopulated, so these photons are not resonantly re-absorbed — they escape after one scattering. Hence net emission where the resonance lines show net absorption.

**Where the emission sits.** In an idealized spherically symmetric, dust-free, fully-captured outflow, re-emission arrives from gas at all projected velocities and the profile is **symmetric about systemic**. That is a real improvement over the resonance lines. Three effects break the symmetry:

- **Occultation** — the far side of the outflow is behind the continuum source. The far side recedes, so *red* emission is preferentially lost → net blueshift.
- **Dust** — far-side photons traverse the galaxy. Same direction: → net blueshift.
- **Aperture** — the emission is spatially extended. This one *favours* VDFI: a fiber IFU captures far more of it than the 0.7″ slits behind most of the literature.

**Verdict: secondary, not an anchor.** The expected residual is *tens of km s⁻¹*, theoretically blueward, and the empirical literature does not converge tightly on a value. An intrinsic offset of "tens of km s⁻¹, sign not firmly settled" is the same order as the 85 km s⁻¹ being searched for. It is several times better than resonant LIS (150–200 km s⁻¹, §6) but it cannot set the zero point.

**What it is genuinely good for — three things:**

1. **It is in-band for the science sample itself.** All five lines fall in 3470–5540 Å across essentially the whole MOSDEF z~2 bin (λ1533.43 in band to z = 2.613, right at the bin edge; λ1264.74 to z = 3.380). Same galaxies, same exposures, no transfer assumption at all.
2. **Five lines spanning 3942 – 5060 Å at z = 2.3** — an internal drift test within one sample.
3. **All redward of Lyα**, so no forest contamination. Si II λ1260 resonance sits at the forest edge; Si II\* λ1264.74 does not.

**How to use it.** Run it *after* the z~1.5 four-anchor measurement, as the check on whether that calibration transfers to the science sample. Agreement within errors closes the transfer objection empirically. Disagreement is bounded by the Si II\* model uncertainty and should be read as a limit, not a correction.

**Free bonus, and it belongs to a different workstream.** Measure Si II λ1526.71 absorption against Si II\* λ1533.43 emission in the same spectrum: **the wavelength calibration cancels exactly**, and the difference is a clean outflow-velocity measurement of a single ion. That is split #15's actual measurement (`../docs/subsample_splits.md`), obtained for free once the region is being fit.

---

## 5. Related: the MOSDEF redshift *precision* claim

`GOALS.md` carries "~60 km s⁻¹ precision (unverified — came from one source, needs a real citation check before it goes in the paper)." Close it in the same pass, since it means opening the same papers.

**Distinguish two things, because they behave completely differently:**

- **Random** per-galaxy redshift error (which is what "60 km s⁻¹ precision" means): does **not** shift the stacked centroid. It **broadens** the stacked line by adding in quadrature to the intrinsic width. Consequence: it inflates measured line widths, and it should be deconvolved before any width is interpreted physically. It does not threaten the centroid.
- **Systematic** offset (this document): shifts the stacked centroid rigidly and does not broaden it.

They are separable in the data: a random error changes width without moving the centroid; a systematic moves the centroid without changing width. Worth stating in Methods, because it makes clear that the 60 km s⁻¹ number — whatever it turns out to be — is not the thing that threatens the result.

---

## 6. Flagged pushback — the LIS-line test, demoted

> **Recorded per the register's convention of preserving contested reasoning rather than silently folding it in.**

**Proposed (Austin, 2026-07-29):** measure low-ionization interstellar (LIS) absorption lines in the VDFI spectra; if they appear blueshifted by ~50 km s⁻¹, that explains the large-scale Lyα blueshift as a calibration artifact.

> ### ⚠️ SUPERSEDED 2026-07-29 (v3) — objections 1 and 2 below are WRONG. See `lis-velocity-calibration.md`.
>
> **Objection 1 was factually wrong.** The 150–200 km s⁻¹ figure is the LBG-composite value (Shapley 2003, Steidel 2010) — a different sample, aperture, and systemic reference. Measured on **MOSDEF galaxies against MOSFIRE nebular redshifts**, Δv_LIS is **−60 ± 10 km s⁻¹** (Weldon 2022, N = 155), and Topping 2020's own systemic relations imply **−32** (z_LIS only) and **−89 km s⁻¹** (z_LIS + z_Lyα). The intrinsic offset is therefore *smaller* than the ~85 km s⁻¹ systematic being searched for, not larger. Austin's instinct that a ~−50 km s⁻¹ measurement is informative was right; only the sign of the inference flipped — **near −50 to −90 is a null result; ≈ −145 to −175 is the smoking gun.**
>
> **Objection 2 was wrong in the way Austin identified.** The degeneracy *is* breakable — by an external measurement of the same quantity on a matched sample. Topping/Weldon supply exactly that, on MOSDEF galaxies, using MOSDEF systemic redshifts. What survives is a narrower point: because the MOSDEF term appears in both measurements it **cancels**, so the test constrains ε_VIRUS − ε_LRIS and does **not** close the MOSDEF tie. That is a scope limit, not a fatal degeneracy, and LRIS is a legitimate anchor because its zero point was itself sky-line-corrected.
>
> **Objection 3 (S/N) stands as written but is no longer binding**, per Austin's direct read of the data: LIS troughs are measurable in sub-stacks of ~10–30 galaxies. Sampling statistics turn out to be the *least* important term in the error budget.

**Objection — LIS lines are not wavelength calibrators.**

1. **They have a large intrinsic astrophysical offset.** LIS absorption (Si II 1260, O I + Si II 1303, C II 1334, Si II 1526, Fe II 1608, Al II 1670) traces outflowing gas and is systematically blueshifted relative to systemic by typically **150–200 km s⁻¹** at these redshifts (Shapley et al. 2003; Steidel et al. 2010), with comparable galaxy-to-galaxy scatter. That intrinsic offset is **larger than the ~85 km s⁻¹ systematic being searched for**. A measured −50 km s⁻¹ would not confirm a calibration error — it would be an anomalously *small* outflow signature, and would more likely indicate a problem with the measurement than with the wavelength solution.
2. **The degeneracy is not breakable.** Any measured LIS velocity is (calibration offset) + (outflow velocity), and there is no independent handle on the second term for this sample. The test cannot return a clean answer regardless of what it finds.
3. **S/N.** These are continuum absorption features on H ≈ 24.5–25 galaxies observed with a fiber IFU at R ~ 750. The per-galaxy continuum S/N is essentially zero; a deep stack is required, and even then the lines are shallow and blended (O I + Si II 1303 is a blend; Si II 1260 sits near the Lyα forest edge).

**Retained, reframed.** LIS lines are still worth measuring — as **physics, not calibration**. A stacked LIS velocity gives a direct outflow-velocity measurement for the sample, which is:

- a genuine constraint on the scattering medium, feeding the RT discussion;
- a candidate **split variable** (`subsample_splits.md`) with a real kinematic prediction — outflow velocity should correlate with Lyα peak asymmetry and with halo extent if the halo is scattered core light;
- a literature comparison point against Steidel 2010, on a sample with independent systemic redshifts, which is itself a small result.

**Move to:** `subsample_splits.md` as a candidate split and the roadmap's RT workstream. ~~**Remove from:** the calibration path.~~

> **v3 verdict (2026-07-29): restored to the calibration path.** Full design in **`lis-velocity-calibration.md`**. Slots as **Tier 1.5** — the substitute for the blocked Tier 1, sitting between the convention audit (Tier 0) and the z~1.5 four-anchor test (Tier 3). Its internal drift test needs no external reference and is the half worth protecting; the physics use above remains valid and rides on the same codebase.

---

## 7. Prioritized plan

**Driver: the z~1.5 core extraction (task 2 → 3).** It is the only task with real latency, it is the only tier that actually answers the question, and everything else on this list can be done while it is being scoped or while it runs. Tasks 1 and 4b are cheap and should proceed in parallel, not in series.

> **v3 revision.** Task 4 is blocked, so **task 4b is the near-term driver** — it uses data already extracted and can start immediately, whereas task 2→3 still needs a footprint cross-match and a Lyα-keyed config audit. Task 4b does **not** demote task 3: 4b constrains ε_VIRUS − ε_LRIS, task 3 constrains ε_VIRUS − ε_MOSDEF, and only the latter is the quantity the science depends on. Run 4b first because it is available, not because it is sufficient.

| # | Task | Tier | Cost | Notes |
|---|---|---|---|---|
| 1 | Air/vacuum + barycentric convention audit (code + MOSDEF/HETDEX docs) | 0 | Hours | Do in parallel with task 2. Cannot make task 3 unnecessary — if a bug is found, the extraction becomes the confirmation that the fix worked |
| 2 | **Scope the z~1.5 core extraction** (§4.4): footprint cross-match on the 291; audit the Lyα-keyed config (line mask, `cont_bounds`, `LYA_REST`); decide re-point vs. minimal dedicated path | 3 | ~Half a day | **The real gate.** Do not launch before this |
| 3 | **Launch the z~1.5 core extraction; joint fit of the 1880–1915 Å complex + He II/O III]** | 3 | Core extraction, small | The actual answer. Four anchors, 3888–5154 Å |
| ~~4~~ | ~~Sky-line centroids vs. a vacuum list, Δv vs. λ~~ | 1 | — | ⛔ **BLOCKED** — data unavailable (§2). Retained as a design only |
| **4b** | **LIS-line velocity vs. Topping/Weldon + internal drift test** — `lis-velocity-calibration.md` | **1.5** | 1–2 days, **data already on disk** | **Tier 1's substitute, and the near-term driver.** Both z~2 and z~3 core products already extracted, so no extraction latency. Delivers the Δv-vs-λ drift curve Tier 1 was for, plus split #15 physics from the same codebase |
| 5 | Si II\* secondary check on the z~2 science sample (§4.6) | 3 | Likely free, Stage 2 | Run *after* task 3, as the transfer check |
| 6 | Stellar absorption centroids, riding on the §3.6 star re-extraction | 2 | ~Free | Lowest value; do not let it consume time |
| 7 | Per-field (COSMOS vs. EGS) centroid comparison as a barycentric check, at EGS fold-in | 0 | Free | — |

**Two outcomes worth pre-committing to.** If tasks 1 and 4 both come back clean and flat *and* task 3 returns a null, the result is a quotable upper limit and the absolute-frame claim is fully defended. If any of them returns a signal, everything downstream of the centroid re-runs — so do not build interpretation on the outer-bin sign until task 3 lands.

---

## 8. What goes in the paper regardless of outcome

One Methods paragraph, roughly:

> Lyα velocities are measured in the rest frame defined by MOSDEF systemic redshifts. Because this is an absolute rather than self-referenced frame, the tie between the MOSDEF near-IR wavelength scale and the VIRUS optical wavelength scale is a potential systematic. We verified that both scales use vacuum wavelengths and that barycentric corrections are applied [or: we measured and corrected an offset of X ± Y km s⁻¹]. We independently checked the VIRUS wavelength solution against night-sky emission lines, finding residuals of [X] km s⁻¹ with no significant trend across 3470–5540 Å. We then measured the MOSDEF–VIRUS tie directly, using a stack of N ≈ 200 MOSDEF galaxies at z = 1.37–1.70 for which four non-resonant systemic-velocity indicators — He II λ1640, O III] λλ1661,1666, Si III] λλ1883,1892 and C III] λλ1907,1909 — fall within the VIRUS band, spanning 3888–5154 Å and therefore bracketing the observed wavelength of Lyα in our z ≈ 2 sample. C III] was fit jointly with Si III] using a common velocity and a free doublet ratio, since the density-dependent C III] λ1907/λ1909 ratio biases a single-component centroid by up to 62 km s⁻¹; Si III] λ1892 is resolved at our resolution and carries no such bias. We obtain Δv = [X ± Y] km s⁻¹, with no significant wavelength dependence across the four indicators. We note that a rigid velocity offset cannot produce the radial gradient that constitutes our primary result; it affects only the absolute sign of the outer-halo velocity, which we report with this systematic included.

That last sentence is the one that matters. Write it whatever the outcome.

---

## See also

- **`lis-velocity-calibration.md` — Tier 1.5, the substitute for the blocked Tier 1 and the near-term driver (v3)**
- `systematics-and-extraction.md` — the general register; this systematic is currently **not** in it, and §10 should gain a pointer
- `../docs/GOALS.md` — the absolute-frame claim, and the unverified 60 km s⁻¹ precision note (§5)
- `../docs/subsample_splits.md` — where the LIS-line measurement goes instead (§6)
- `../../docs/workstream-roadmap.md` — this spec's place in the wider plan
- `../code/utils_python/` — the reduction path to audit in §1.2(c)
- `../code/utils_lya_halo/config.py` — `wave_min`/`wave_max`, `LYA_REST`
- `../code/utils_lya_halo/core.py` — `extract_core_spectra`, the Tier 3 machinery
