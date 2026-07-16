# Discussion Section — Draft Notes

*Draft ideas, not final prose. Trimmed for skimming — expand into full paragraphs later.*

---

## 1. Centroid as a function of radius

- **Systemic anchor solves the Guo 2024 problem.** MOSDEF's independent systemic redshift lets us
  report an intrinsic velocity offset (blue/redshift relative to true zero), not a shift relative
  to the Lyα line itself. Guo et al. 2024 (self-referenced) can only describe a relative red→blue
  trend; we can say where the gas actually sits kinematically. This is the paper's core
  methodological advantage — state it plainly here (§2 just references back to this).
- **Overall shape / connection to prior propositions:** red-dominated near center → crosses
  systemic near R_vir → mildly negative (blueshifted) beyond. Candidate readings: blueshift ~
  scattering off inflowing/accreting gas (not necessarily cooling emission itself); redshift ~
  outflows; a redshifted signal could also be environment/satellite-clustering driven (test via
  the environment split in §4). Centroid-method comparison is already cross-checked against an
  independent 3D-HST sample — consistent, good sign this isn't dataset-specific. Compare shape
  directly against Guo 2024's red-to-blue trend as the nearest literature analog.
- **Combined significance of points beyond R_vir:** individual outer bins (~110–220 kpc) sit
  around 1.2–1.35σ each; a covariance-combined significance across the >R_vir bins lands near
  ~1.7σ (two-sided) — the number worth quoting as primary. An independent flux-pooling cross-check
  (weighted-median merge, not a linear combine) reproduces the same central value (~−47 to
  −48 km/s) at a somewhat lower but still consistent ~1.5σ. Convergence across independent
  combination methods is the strongest part of the case, even with no single number clearing a
  formal detection threshold — state honestly as "a consistent, modest trend," not a confident
  detection.

---

## 2. The ~1 R_vir cutoff, quantified by the fit

R_vir reference = **75 kpc** (AGN-excluded, ~450 galaxies — the correct way to compute it). The
90 kpc in the current draft tex is the full ~500-galaxy sample (AGN included); supersede it.

**Three independent qualitative diagnostics turn over near this radius:** flux curve bends,
centroid flips negative (§1), B/T flips negative. Agreement across independent diagnostics is the
actual evidence this is a real physical transition, not a fitting artifact — and the two-component
fit below is what turns that agreement into an actual number, so the cutoff and the fit are one
story, not two.

**The fit.** Two-component exponential, `I(r) = A1*exp(-r/h1) + A2*exp(-r/h2)`, PSF-aware
(ring-convolution forward model), all 10 bins: χ²/dof = 7.04/6 = 1.17.
- **Core: h1 = 16.9 ± 1.1 kpc.** vs. Steidel 2011 (20.8–28.4 kpc range), Wisotzki 2016 (1–7 kpc),
  Leclercq 2017 (~4.5 kpc). Sits below Steidel, plausibly from our mixed LBG+MELG composition.
  Continuum comparison currently borrows Steidel's 3–4 kpc; **plan to fit our own continuum scale
  length from CFHT-LS imaging** for a same-sample comparison instead — future work, flag it here.
- **Crossover radius (term1=term2) ≈ 76 kpc ≈ R_vir, robust.** This is the fit's own quantitative
  echo of the cutoff — the same turnover the flux/centroid/B-T diagnostics point to qualitatively,
  landing almost exactly on the independently-derived R_vir = 75 kpc. Four independent routes to
  the same radius (three qualitative diagnostics + one quantitative fit) is the strongest version
  of the cutoff claim and worth stating as such, not as four separate coincidences.
- **Outer term (h2 = 1552 ± 548 kpc):** flux persists to very large radii — the qualitative result
  is solid, the interpretation isn't. Keep this modest; don't over-invest in the γ=1.8→0.8 story
  below, there's a real chance it's not even the right comparison.
  - raw galaxy-clustering slope (γ~1.8, 3D) vs. Limber-projected (~0.8) tested; projected fits
    marginally better but not decisively (small ΔAIC, 6 dof) — plausibility note, not a result.
  - **counter-claim worth featuring:** Chen 2020's own optical-depth slope at large radii is
    ~−0.5, from diffuse IGM HI — arguably the more apt comparison, since our outer signal is
    diffuse emission, not a galaxy-clustering statistic. Genuinely open, not resolved either way.
  - r_c ≈488±211 kpc (~6.5×R_vir, from the proposed Option C exponential + cored-power-law
    extension) loosely echoes Sorini 2018's ~7×R_vir "sphere of influence" (different observable —
    absorption vs. our emission) — one sentence, don't lean on it; r_c/γ are degenerate at current
    radial coverage.
  - Byrohl 2021: outer flattening = photons from *other* halos, not own CGM, onset ~30 pkpc in
    TNG50 (below our ~76 kpc) — a testable prediction for §3, not an assumed explanation.
- Flux curve-of-growth: worth quoting as one clean number — "X% of total flux lies beyond R_vir."

**Supporting checks:** Niemeyer O[III] comparison (independent reduction/background/method)
agrees — good corroboration, worth pulling out of the Appendix. A COSMOS-vs-EGS split would
directly test whether the cutoff radius is field-stable (matters given the correlated-noise
caveat below).

**Lit:** **Rakic et al. 2012** (ApJ 751,94) — optical depth drops an order of magnitude at ~100 kpc
(≈R_vir); FoG (<1 Mpc) vs. Kaiser (1.4–2.0 Mpc) anisotropy split is the field's cleanest
precedent for "two physically distinct scales." **Chen et al. 2020** — independent kinematic
transition (outflow→infall) at 80–200 pkpc, pure Hubble flow beyond ~300 pkpc — a supporting
analog, not the same measurement.

**Caveats:** correlated (not random) outer-bin background noise is still an open systematic —
could bias the exact cutoff location. R_vir is calibrated assuming a KBSS/LBG-like ~10^12 M_sun
halo; LAE clustering (ODIN/DESI) suggests lower masses (~10^11) — could shift "1 R_vir" itself.
r_c/γ degeneracy (bins only reach ~1000 kpc, need ~1.5 Mpc to clear it); fit errors are single-fit
covariance, not bootstrapped; still undecided whether the shipped model is a broken exponential
pinned to R_vir vs. a smooth two-component sum — changes the rhetorical claim.

---

## 3. Origins of emission — subsample splits

Widest-scoped section — will include AGN vs. galaxy, low-z vs. high-z, and the rest below,
organized around one question: which term (core vs. outer) tracks which physical variable.

- Splits in play: mass, ΣSFR, dust (Balmer decrement/UV slope β), z bin (z~2 vs z~3), Lyα velocity
  offset (self-split), environment, EW, orientation, AGN vs. galaxy, double-Gaussian peak
  separation/height.
- Organizing frame — **Byrohl 2021**: inner term ≈ own star-forming-region photons, outer term ≈
  photons from other halos. Splits test this directly: does h1 track mass/SFR while the
  outer/crossover term tracks environment instead of the individual galaxy?
- **AGN vs. galaxy** — test for a fluorescence signature (disproportionate outer flux in
  AGN-flagged subsample). Trainor 2025: even Lyα absorbers are net emitters — keep "absorber" as
  its own category, not a non-emitting control.
- **z~2 vs. z~3** — doubles as a systematics check (different mag cuts) but is also a real result:
  ~1 Gyr evolutionary baseline on halo extent/kinematics.
- **Peak separation/ratio** (double-Gaussian core fit) — separation ~ HI column density, peak
  height ratio ~ outflow/inflow asymmetry. Only physically meaningful because the LSF check already
  confirms the ±4 Å window isn't instrumentally dominated — state that dependency explicitly. Cite
  actual N_HI scaling relations (Verhamme+2015, Gronke & Dijkstra 2016, Orlitová+2018), not just
  the qualitative orientation papers already in the draft.
- **SF ceiling test** (`Lya-maximum-emission-summary.md`, Case B/EW ceiling): does flux/EW beyond
  R_vir exceed what star formation alone can power? A direct, falsifiable test for a non-stellar
  source (cooling, fluorescence, undetected satellites) — good anchor for the whole section.
- **Bacon 2021**: fluorescence caps at 10–34% of filament luminosity → needs an undetected
  ultra-faint LAE population to explain the rest. Worth asking whether our own mag cuts
  (H<24.5/25) exclude exactly that population — ties to the open GOALS.md question of what's lost
  by stacking with a medium-resolution spectrograph.
- Caveats: double-Gaussian splits need a noise-prefabrication null test first (open item,
  `research-notes.md`); √N loss + marginal whole-sample signal (~1.5–1.7σ) limits how many crossed
  splits are meaningful; classification scheme (AGN/LAE/absorber/null) still being finalized;
  environment/overdensity catalog — the field's most diagnostic test — not yet in hand.

---

## Blindspots (quick hits)

1. Scale-length "category error" — h1/h2, r_c, r0 aren't interchangeable; one clarifying line in §2.
2. R_vir-anchor mass mismatch touches every R_vir-relative number, not just §2's cutoff.
3. Correlated outer-bin noise is one systematic threading through §1 and §2 especially — say it once, reference it later, rather than repeating.
4. Population mixing (LBG+MELG) is itself a candidate explanation for why h1 undershoots Steidel.
5. Cosmic-web imaging comparisons (Bacon SB, UVB fluorescence line) are still qualitative — not unit-matched to our flux convention yet.
6. Fit functional form (broken exponential @ R_vir vs. smooth sum) still undecided — changes what the "break" claim actually means.
7. Redshift-space anisotropy (Rakic's FoG/Kaiser split) isn't implemented here — the Rakic comparison is currently an analogy, not a matched test.

---

*Context note: leans on recent working sessions (fitting/PSF infra, clustering lit review,
plotting/validation work) + GOALS.md/research-notes.md/subsample_splits.md/specs. One pulled
session was unrelated personal content — a reminder "last N chats" isn't a full proxy for
pipeline state; the standing docs are.*
