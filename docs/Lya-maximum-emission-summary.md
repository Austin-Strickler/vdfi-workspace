# Theoretical Maximum Lyα Emission — Literature Summary

Purpose: a grounded reference for "how much Lyα is too much?" — the theoretical ceilings on Lyα production and escape as a function of galaxy properties (SFR, stellar mass, metallicity, age, dust), so that observed flux (e.g., integrated as a function of radius) can be tested against a physically motivated maximum before invoking a non-stellar photon source. This is written for the VDFI/HETDEX context — see [`VDFI-background.md`](VDFI-background.md) and [`LARS-summary.md`](LARS-summary.md) for related project background.

The short version: there are two separate ceilings in the literature, and they answer different questions. One is a **luminosity ceiling** (how much Lyα can Case B recombination make, given a measured SFR or Hα/UV luminosity). The other is an **equivalent-width ceiling** (how large can W(Lyα) get relative to the non-ionizing UV continuum, given a stellar population's age/metallicity/IMF). Neither ceiling is specific to "LAEs" as a selected class — both are set by stellar/nebular physics that applies to any star-forming galaxy, including massive, dusty ones. What differs between a canonical LAE and a massive dusty star-forming galaxy is not the ceiling itself but how far below it the galaxy is expected to sit, and dust is expected to push massive galaxies *down* from the ceiling, not up. That asymmetry is what makes an observed excess in a massive/dusty system a stronger anomaly than the same excess in a young, low-mass system.

---

## 1. The luminosity ceiling: Case B recombination and the SFR–L(Lyα) relation

This is the most directly applicable "maximum" for comparing to a measured SFR or halo mass, and it is **not restricted to LAEs** — it is a direct rewriting of the Kennicutt (1998) Hα–SFR calibration through case B recombination theory, and applies to any star-forming galaxy.

Under case B recombination (T_e ≈ 10⁴ K, n_e in the low-density limit), the intrinsic ratio of Lyα to Hα emission from photoionized gas is fixed at

**L(Lyα)/L(Hα) ≈ 8.7**

(the standard case B value; see e.g. discussion in Hayes 2015, *Lyman-alpha Emitting Galaxies in the Nearby Universe* review, and used throughout the Lyα-SFR literature). Combining this with the Kennicutt (1998) calibration SFR = 7.9×10⁻⁴² L(Hα) [M☉/yr, L in erg/s; Salpeter IMF 0.1–100 M☉, solar abundances, constant SFR over ~10⁸ yr] gives the **dust-free, no-loss maximum Lyα luminosity for a given SFR**:

**L(Lyα)_max ≈ 1.1 × 10⁴² × SFR(M☉/yr) erg/s**

This is the number to beat: it is the absolute photon-production ceiling from case B recombination alone, assuming every ionizing photon produced by the young stellar population is reprocessed into recombination lines *in situ* (i.e., ionization-bounded HII regions, zero escape of ionizing photons as LyC, zero dust destruction of Lyα photons). Any measured/integrated Lyα flux converted to a luminosity that exceeds this value for the galaxy's independently-measured SFR (from UV continuum, IR, or Hα itself) requires either (a) an SFR that is underestimated, (b) a stellar population with above-canonical ionizing photon output per unit SFR (see §2), or (c) a non-recombination or non-stellar photon source.

Because this ceiling derives from Hα→SFR, if you have a direct Hα measurement for the same aperture, using L(Lyα)_max = 8.7 × L(Hα)_obs is a tighter, calibration-independent version of the same test — it removes the SFR-indicator conversion uncertainty entirely and is only a statement about recombination physics.

**Where this can slip:** the 8.7 ratio and the 7.9×10⁻⁴² coefficient both assume solar metallicity, a Salpeter IMF, and continuous star formation over ~100 Myr. Sub-solar metallicity, top-heavy IMFs, and especially young/bursty star formation histories all raise the ionizing photon output per unit UV/Hα luminosity above these canonical values (see §2), which raises this ceiling. So this is a *typical* maximum, not an absolute one — it is the number appropriate for a "normal," IMF-averaged, equilibrium stellar population, and should be treated as a soft ceiling unless the stellar population's age/metallicity/IMF have been independently constrained (e.g., via SED fitting) to rule out an elevated ionizing efficiency.

## 2. Ionizing photon production efficiency (ξ_ion): what sets the ceiling in §1

The §1 ceiling is really the product of (i) the UV continuum luminosity and (ii) the ionizing photon production efficiency ξ_ion ≡ Q(H⁰)/L_UV(1500 Å), converted through case B. Canonical SFR calibrations imply log₁₀(ξ_ion/Hz erg⁻¹) ≈ 25.1–25.2. Measured values for high-z star-forming galaxies span roughly log ξ_ion ≈ 25.1–25.8, with compact, low-mass, low-metallicity, and high-sSFR (bursty) systems systematically elevated relative to typical/massive galaxies; in the high-mass regime, ξ_ion shows little to no dependence on stellar mass once other properties are controlled for. This means the §1 ceiling can legitimately be ~2–4× higher than the canonical 1.1×10⁴² SFR relation for a young/metal-poor/bursty stellar population, but for a typical massive star-forming galaxy with a fairly evolved stellar population, ξ_ion (and hence the ceiling) should sit close to the canonical value rather than the elevated end.

## 3. The equivalent-width ceiling: what stellar population synthesis models allow

This is the ceiling most often discussed in the "is this EW too high to be normal star formation" literature, and it is a **statement about stellar population physics (age, metallicity, IMF), not about LAE selection**. Any star-forming galaxy with a young stellar component is bound by the same models; a massive galaxy is not exempt from the ceiling, it's just expected to sit far beneath it because its light is dominated by an older, more chemically evolved population and because dust preferentially removes Lyα relative to continuum (§4).

Key reference values from population-synthesis calculations of the intrinsic (dust-free, 100%-escape) rest-frame Lyα equivalent width:

| Regime | W(Lyα)_max (rest-frame) | Conditions | Source |
|---|---|---|---|
| Normal metallicity, Salpeter-like IMF, constant SFR, age ≲ 10 Myr | ~50–240 Å (SSP ceiling ~210–240 Å depending on IMF upper mass cutoff) | Solar-ish Z, standard IMF, near-instantaneous/young burst | Charlot & Fall 1993 |
| "Normal" young starburst upper bound often quoted | ~300 Å | 1 Myr population; falls to ~150 Å by 10 Myr, ~100 Å by ≳10⁸ yr | Malhotra & Rhoads 2002 |
| Very young, very low metallicity (Z ≲ 4×10⁻⁴ Z☉) | up to ~1000–1500 Å | Extreme metal-poor/near-Pop III, top-heavy/hard ionizing spectrum, age ≲ 2 Myr | Schaerer 2003 |
| Metal-free Population III | >1000 Å (declining to ~500 Å for older bursts) | Zero metallicity, T_eff ~10⁵ K massive stars, ~60% more H I-ionizing photons and up to 10⁵× more He II-ionizing photons than Pop II | Schaerer 2003; Raiter, Schaerer & Fosbury 2010 |

The commonly cited "hard ceiling" for normal star formation is **W(Lyα) ≈ 200–240 Å**: above this, a normal-IMF, solar-ish-metallicity stellar population plus case B recombination cannot produce enough Lyα relative to non-ionizing UV continuum, even dust-free and with 100% escape. Observed EWs above this threshold (a substantial fraction of high-z narrowband-selected LAE samples do exceed it) require *some* combination of: extreme youth/burstiness, sub-percent-solar metallicity, a top-heavy IMF, binary-star effects that harden the ionizing spectrum at fixed age, radiative-transfer/geometric enhancement (§4), or a non-stellar contribution (AGN, shocks, cooling radiation — see §5). AGN photoionization is a real alternative but is usually disfavored for individual very-high-EW emitters unless accompanied by broad lines or high-ionization species (e.g., N V λ1240), which are typically absent.

**Applicability beyond LAEs — the key point for a mass-selected sample:** because this ceiling is a property of the young stellar population's ionizing-to-non-ionizing photon ratio, it applies identically to a massive, mass-selected star-forming galaxy that happens to be forming stars vigorously. What changes for massive galaxies is the *expected* (not the maximum) EW: MOSDEF and KBSS-type samples show a clear anti-correlation between W(Lyα) and stellar mass, driven by older/more mixed stellar populations and higher dust content in massive galaxies — both effects that push emergent EW down, not up. So a massive galaxy sitting near or above the 200–300 Å ceiling is, if anything, a *more* surprising result than a young low-mass LAE doing so, precisely because everything about being massive (age, metallicity, dust) should suppress EW relative to the ceiling.

## 4. Dust: how it enters, and why it isn't a simple screen for Lyα

Dust is the obvious first suspect for suppressing Lyα below the §1/§3 ceilings, but because Lyα is a resonant line, dust does not act as a simple foreground screen the way it does for a UV/optical continuum measurement. This matters directly for interpreting your dust SED-fit parameters.

**Differential attenuation, nebular vs. stellar.** Nebular emission lines (Balmer decrement-derived E(B–V)_gas) are empirically found to be more attenuated than the stellar UV/optical continuum (E(B–V)_star), by a factor of roughly 2 in the classic Calzetti (1997, 2000) formulation, because nebular gas traces the youngest, most dust-embedded star-forming regions while the continuum includes an older, less obscured population. If you're using an SED-fit continuum attenuation to predict the *nebular/recombination* Lyα budget, that continuum E(B–V) will generally underestimate the attenuation actually experienced by the ionizing photon "engine," unless you've fit the nebular lines directly.

**Attenuation curve shape near Lyα/LyC.** Reddy et al. (2016a), from ~1000 z~3 LBGs with deep rest-UV spectroscopy (850–1300 Å), find that attenuation near the Lyman continuum (~900 Å) is roughly 2× lower than naive polynomial extrapolations of standard curves (e.g., Calzetti et al. 2000) would predict. This means dust models calibrated at longer UV wavelengths (~1500–2000 Å, i.e., typical SED-fit E(B–V)) can overpredict attenuation right at/near Lyα if extrapolated blindly — worth checking which curve your SED fits used and over what wavelength baseline it was calibrated.

**Radiative transfer decouples f_esc(Lyα) from a simple dust-screen prediction.** Because Lyα resonantly scatters, its effective path length through dust is far longer and more geometry-dependent than a continuum photon's. Two effects matter here:

- *Neufeld (1991) clumpy/multiphase enhancement.* If dust resides in cold clumps and Lyα photons scatter preferentially off clump surfaces (large scattering cross-section) while continuum photons are more likely to enter and be absorbed within clumps, Lyα can escape *more* efficiently than the continuum for the same dust mass — potentially inflating emergent EW above what a homogeneous dust-screen model would predict, and even above naive expectations from the same dust column. However, this is a geometry-dependent, non-guaranteed enhancement, not a generic boost: Laursen, Duval & Östlin (2013) show the "Neufeld effect" requires specific conditions (very low ICM opacity, high covering factor, small clump volume filling factor) and largely vanishes or reverses under more realistic multiphase ISM structures — i.e., don't assume it as a free pass to explain excess flux without checking whether the required geometry is plausible for your galaxies.
- *Expanding-shell (Verhamme et al. 2006, 2008) modeling.* The standard semi-empirical approach fits observed Lyα line profiles with a central source surrounded by an expanding, dusty, HI shell (parameterized by column density, expansion velocity, dust opacity, Doppler parameter). This is the tool of choice for translating a Lyα line profile (not just integrated flux) into an effective HI column and dust opacity, and pairs naturally with your SED dust parameters if you want to test whether a single shell geometry can self-consistently reproduce both continuum reddening and Lyα escape.

**Empirical f_esc(Lyα)–E(B–V) relation.** As a purely empirical cross-check (bypassing radiative transfer modeling), Atek et al. (2009, 2014) and Hayes et al. (2011) find f_esc(Lyα) anti-correlates with E(B–V), roughly of the form f_esc ≈ 10^(−0.4 × E(B–V) × k_Lyα), calibrated on local starbursts/LARS-like samples. This gives you a quick, model-independent "expected f_esc given your measured dust" to multiply onto the §1 ceiling, as a sanity check against the more detailed shell/clumpy modeling.

**Practical implication for your SED dust parameters:** the single most useful thing you can pull from your dust SED fits is the *nebular* E(B–V) (from Balmer decrement if available) rather than the stellar continuum E(B–V), since it's the nebular attenuation that actually gates the case B budget in §1. If only continuum E(B–V) is available, treat the dust-attenuated ceiling as a lower bound on the true ceiling (real nebular attenuation is probably higher, i.e., the true dust-attenuated max may be even lower than what continuum-only dust correction implies) — which makes an observed excess over that ceiling even harder to explain away with "dust modeling was wrong."

## 5. Radial flux and extended halos: what caps the *total*, integrated-to-large-radius flux

This is the section most relevant to a flux-vs-radius/curve-of-growth test, since resonant scattering — not just dust — actively redistributes photons in radius without creating or destroying them.

**Scattering redistributes, it doesn't add photons.** Because Lyα is resonantly scattered by any residual HI in the circumgalactic medium, photons produced centrally (HII regions) can random-walk outward in both space and frequency before escaping, building an extended halo even with no additional photon production. Steidel et al. (2011), Momose et al. (2014, 2016), Matsuda et al. (2012), Wisotzki et al. (2016), and radiative-transfer work by Gronke & Bird (2017) and Byrohl et al. (2021) show that scattering of centrally-produced (and, importantly, sometimes satellite-produced) Lyα photons can account for much or most of the diffuse halo signal seen in stacked and individual MUSE/HETDEX observations. If your total flux, integrated out to large radius, is consistent with pure scattering of the central §1/§3 budget, no extra photon source is needed even if the surface brightness profile looks "too extended" compared to a naive point-source expectation — extension in radius alone is not evidence of excess *total* flux.

**But the photon budget itself can legitimately include more than one central galaxy's recombination output.** Two additional physical (not central-galaxy-stellar) sources are established in the literature as real contributors to the integrated halo flux budget, and both would appear as a genuine excess over a single-galaxy §1 ceiling:

- *Satellite/clustered star formation.* Mitchell et al. (2021), using cosmological simulations, find satellite galaxies are the dominant Lyα source at intermediate halo radii (~10–40 kpc), with cooling radiation also contributing; Momose et al. (2016) similarly proposed halo emission is powered by multiple satellites clustered around the central galaxy rather than the central galaxy alone. If your "galaxy" flux aperture is drawing in satellite contributions at larger radii, the effective ceiling should be the *sum* of the central and satellite case B budgets, not just the central one — this is a legitimate way total flux can exceed the single-galaxy §1 ceiling without any exotic physics.
- *Gravitational cooling radiation from cold accretion.* Cold gas streaming into a dark matter halo's potential well can shock/cool and radiate significantly in Lyα, independent of in-situ star formation. Predicted luminosities are highly model-dependent: Dijkstra & Loeb (2009) and Rosdahl & Blaizot (2012) find L_cooling ≲ 5×10⁴¹ erg/s for a ~10¹¹ M☉ halo (i.e., often sub-dominant to star-formation-powered Lyα for typical SFRs), while other numerical treatments push predicted cooling luminosities as high as ~10⁴⁴ erg/s for massive halos — a genuinely unsettled range in the literature (see summary discussion and references in Dijkstra 2017, Saas-Fee lecture notes review, arXiv:1704.03416). Rosdahl & Blaizot (2012) further find that dense gas with collisionally-excited emission (n > 0.3 cm⁻³) can contribute up to ~40% of total simulated Lyα luminosity in some halos. This mechanism scales with halo mass and gas accretion rate, not SFR, so it is a plausible route to a genuine SFR-normalized excess, but the theoretical uncertainty is large enough that it's a "maybe, check the halo mass regime" rather than a clean quantitative ceiling. Historically, individual giant Lyα blobs originally proposed as cooling-powered have in some cases (e.g., the LAB1 prototype) had that interpretation later challenged/overturned in favor of embedded star formation/AGN — worth treating cooling-radiation claims for any single object skeptically without kinematic/multi-line corroboration.

**Recommended framework for your radius-vs-flux test:**

1. Compute L(Lyα)_max from case B via §1, using the best available SFR indicator (or directly via 8.7×L(Hα) if you have spatially matched Hα).
2. Apply a dust correction using nebular (Balmer-decrement-preferred) E(B–V) via an appropriate curve (§4) — or, as a cross-check, the empirical f_esc(Lyα)–E(B–V) relation — to get a dust-attenuated *expected* (not maximum) Lyα luminosity; treat the dust-free §1 number as the true outer ceiling and the dust-attenuated number as the "expected" value, with real objects populating the space between plus scatter from geometry/ξ_ion effects (§2, §4).
3. Build your curve of growth (cumulative flux vs. aperture radius), and check it against a convergence radius — Lyα curves of growth from scattering alone typically flatten by a few tens of kpc at these redshifts (per the MUSE/HETDEX halo literature above); a curve that keeps rising well past where dust/scattering models predict convergence is itself a flag.
4. Compare the converged total to the dust-free ceiling from step 1 (plus, if relevant, a plausible satellite contribution — check for companions in your imaging/catalog within the aperture). If the converged flux exceeds even the dust-free, satellite-inclusive ceiling, that is a real excess requiring either (a) elevated ξ_ion/non-canonical stellar population (§2 — check age/metallicity from your SED fit), (b) a Neufeld-type geometric enhancement (§4 — check if the required ISM geometry is physically plausible, don't assume it), or (c) a genuinely non-stellar source (cooling radiation, AGN, shocks — §5), which is the scientifically interesting conclusion your framing is aiming at.
5. Because your sample is high-mass/dusty, remember §3's asymmetry: dust and stellar age should be pushing these galaxies *below* the ceiling, not toward it. An excess in this regime is harder to wave away with "maybe the dust correction is off in the generous direction" than it would be for a young, low-mass LAE.

---

## Summary table: what's an "LAE-only" result vs. what's general

| Result | Applies to LAEs specifically? | Applies to your (high-mass, mass-selected) sample? |
|---|---|---|
| L(Lyα)_max ≈ 1.1×10⁴² × SFR (case B ceiling) | No — general recombination physics | Yes, directly |
| ξ_ion-driven elevation of that ceiling | No — depends on stellar pop, not selection | Yes, if young/low-Z sub-populations present; check SED fit |
| W(Lyα)_max ≈ 200–300 Å (normal IMF/Z) | No — stellar population physics | Yes; your galaxies are simply expected to sit well below it |
| W(Lyα) > 1000 Å (Pop III / extreme metal-poor) | Applies to any galaxy with that stellar population, but empirically only seen/plausible in extreme high-z, low-mass, low-Z systems | Unlikely to be relevant unless you have evidence for a genuinely primitive sub-component |
| EW–mass anti-correlation | Describes the LAE-selected population's mass distribution, but the underlying cause (age, dust) is general | Yes — this is *why* your sample sits below the ceiling by default |
| Neufeld clumpy-dust EW enhancement | General radiative transfer effect, geometry-dependent | Yes, if applicable geometry — not automatic |
| Satellite/cooling-radiation contributions to extended flux | General halo physics, mass/environment dependent | Yes, arguably more relevant at high mass (more satellites, deeper potential wells) |

---

## References

- Charlot, S. & Fall, S. M. 1993, ApJ, 415, 580 — stellar population synthesis predictions for W(Lyα), 50–240 Å ceiling for normal IMF/metallicity young populations.
- Malhotra, S. & Rhoads, J. E. 2002, ApJL, 617, L5 — [arXiv:astro-ph/0111126](https://arxiv.org/abs/astro-ph/0111126) — ~300 Å ceiling for 1 Myr populations, declining with age; discussion of z=4.5 LAEs exceeding normal-population predictions.
- Schaerer, D. 2003, A&A, 397, 527 — [arXiv:astro-ph/0210462](https://arxiv.org/abs/astro-ph/0210462) — population synthesis incl. Population III/low-metallicity extreme EW predictions (>1000 Å).
- Raiter, A., Schaerer, D. & Fosbury, R. A. E. 2010, A&A, 523, A64 — Population III/extreme low-metallicity colors and Lyα EW predictions.
- Kennicutt, R. C. 1998, ARA&A, 36, 189 — SFR calibration from Hα (and other) luminosity, basis of the case-B-derived L(Lyα)–SFR relation.
- Bouwens et al. and others using case B L(Lyα) = 8.7 L(Hα) → SFR(Lyα) ≈ 9.1×10⁻⁴³ L(Lyα) [see review discussion in Hayes, M. 2015, PASA, 32, e027, "Lyman Alpha Emitting Galaxies in the Nearby Universe"].
- Reddy, N. A. et al. 2016a, ApJ, 828, 107 — far-UV dust attenuation curve from z~3 LBG spectroscopy, attenuation near Lyman continuum ~2× lower than extrapolated Calzetti-type curves.
- Calzetti, D. et al. 2000, ApJ, 533, 682; Calzetti, D. 1997 — differential nebular vs. stellar continuum reddening (factor ~2).
- Neufeld, D. A. 1991, ApJL, 370, L85 — clumpy/multiphase ISM model for enhanced Lyα escape relative to continuum ("Neufeld effect").
- Laursen, P., Duval, F. & Östlin, G. 2013, ApJ, 766, 124 — [arXiv:1211.2833](https://arxiv.org/abs/1211.2833) — critical reassessment/non-enhancement result for the Neufeld effect under realistic multiphase conditions.
- Verhamme, A., Schaerer, D. & Maselli, A. 2006, A&A, 460, 397 — expanding shell radiative transfer model for Lyα line profiles.
- Verhamme, A. et al. 2008, A&A, 491, 89 — shell model applied to observed Lyα spectra, dust/HI column/velocity fitting.
- Atek, H. et al. 2009, A&A, 506, L1; Atek, H. et al. 2014, ApJ, 800, 18 — empirical f_esc(Lyα) vs. E(B–V) local calibration.
- Hayes, M. et al. 2011, ApJ, 730, 8 — f_esc(Lyα)–dust relation, redshift evolution.
- Steidel, C. C. et al. 2011, ApJ, 736, 160 — stacked extended Lyα emission around z~2–3 LBGs.
- Momose, R. et al. 2014, MNRAS, 442, 110; Momose, R. et al. 2016, MNRAS, 457, 2318 — stacked Lyα halo profiles, satellite-powered halo scenario.
- Wisotzki, L. et al. 2016, A&A, 587, A98 — [arXiv:1509.05143-family MUSE HUDF Lyα halo paper] — individual-object extended Lyα halo detections with MUSE.
- Leclercq, F. et al. 2017, A&A, 608, A8 — [arXiv:1710.10271](https://arxiv.org/abs/1710.10271) — MUSE HUDF Survey VIII, extended Lyα haloes around high-z star-forming galaxies.
- Gronke, M. & Bird, S. 2017, ApJ, 835, 207 — radiative transfer origin of extended Lyα halos from central + satellite sources.
- Mitchell, P. D. et al. 2021, MNRAS, 501, 5757 — simulated origin of extended Lyα emission, satellite dominance at intermediate radii.
- Dijkstra, M. & Loeb, A. 2009, MNRAS, 400, 1109 — cooling radiation predictions for Lyα blobs.
- Faucher-Giguère, C.-A. et al. 2010, ApJ, 725, 633 — cold accretion/cooling radiation Lyα luminosity predictions.
- Rosdahl, J. & Blaizot, J. 2012, MNRAS, 423, 344 — simulated cooling-radiation and collisional-excitation contributions to Lyα blobs.
- Dijkstra, M. 2014, PASA, 31, e040 (review); Dijkstra, M. 2017, Saas-Fee Lecture Notes, "Physics of Lyman Alpha Radiative Transfer" — [arXiv:1704.03416](https://arxiv.org/abs/1704.03416) — comprehensive review tying together escape mechanisms, halo origins, and outstanding uncertainties (including the cooling-radiation luminosity range spanning ~10⁴¹–10⁴⁴ erg/s across different treatments).
- MOSDEF survey EW–mass results: Reddy, N. A. et al. 2018, ApJ, 869, 92 (MOSDEF rest-optical EW evolution paper; referenced for the general EW–mass/age/dust trend framework) — cross-check against the specific Lyα–mass papers in your own literature folder before citing a number.
- Ionizing photon production efficiency (ξ_ion): review/compilation values discussed in Nakajima, K. et al. 2020, ApJ, 889, 161; Simmonds, C. et al. 2023/2025 (A&A) ξ_ion at z~4–10; Emami, N. et al. 2020 low-mass ξ_ion — cited above for the log ξ_ion ≈ 25.1–25.8 range and mass/burstiness dependence.

*Note: this summary was compiled from web search of paper abstracts/secondary discussion, not full-text read-through of every primary source (particularly the ξ_ion compilation papers and some MOSDEF/KBSS EW–mass citations, which are given as representative rather than verified page-and-table level). Treat this as a starting map of the literature and spot-check the specific numbers (especially the EW ceiling values and the cooling-radiation luminosity range) against the primary papers before quoting them in a paper draft — same caveat as noted in `LARS-summary.md`.*
