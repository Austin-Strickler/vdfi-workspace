# Discussion Section — Draft Notes

*Prepared 2026-07-15. Source material: `01-lya_halos/docs/GOALS.md`, `research-notes.md`,
`subsample_splits.md`, `specs/halo-flux-fitting.md`, `halo_gas_correlation_literature_review.md`,
`docs/paper-writing/lyahalos_draft_7-13-26.tex`, and recent working-session context. Numbers below
are pulled from the pipeline's own results, not re-derived here — verify each against the live
`boot`/`fitting.py` output before it goes in the paper.*

---

## 1. The ~1 R_vir cutoff / turnover

**What we have.** Three independent diagnostics turn over near the same radius:
- Flux profile: the two-component fit's term1=term2 crossover radius is ~76 kpc, both in the
  two-exponential model and the exponential+cored-power-law (expcore) model.
- Centroid: per the current draft text, centroid shifts from red-dominated through systemic at
  ≈90 kpc ("one virial radius"), staying slightly negative (blueward) beyond that, though
  consistent with zero given outer-bin noise.
- R_vir itself is independently estimated at ≈75 kpc (55–90 kpc band) from the Moster+2013
  SMHM-derived biweight virial radius.

Note the ~76/90/75 kpc figures aren't identical — worth reconciling (or explicitly noting the
spread) before asserting a single "1 R_vir" number in the Discussion. The B/T (presumably
blue/total flux ratio) turning negative should be checked against this same radius range and
folded into the same paragraph as a third independent line of evidence, alongside flux and
centroid.

**Literature framing.** This is the observational signature of a one-halo → two-halo (or CGM →
IGM/large-scale-structure) transition, which is well precedented:
- **Rakic et al. 2012** (ApJ 751, 94 — the KBSS paper linked) — median HI optical depth around
  z~2.4 galaxies drops by an order of magnitude at ~100 kpc, close to the halo virial radius,
  then stays significantly enhanced out to 2.8 proper Mpc. They also split by redshift-space
  anisotropy: finger-of-god (halo-internal kinematics, <1 Mpc) vs. Kaiser-effect compression
  (large-scale coherent infall, 1.4–2.0 Mpc) — the cleanest observational precedent for "small-scale
  term + large-scale term, different physical origins" in this literature, and a template for how
  to argue this isn't just a fitting artifact.
- **Chen et al. 2020** (MNRAS 499, 1721 + erratum) — KBSS galaxy pairs, independent kinematic
  tracer: outflow-dominated gas inside ~50 pkpc, a flattening/minimum in both LOS velocity range
  and Lyα EW over 80–200 pkpc marking the outflow→infall transition, pure Hubble flow (no longer
  halo-bound) beyond ~300 pkpc. The 80–200 kpc flattening sits almost exactly where our own
  crossover/turnover lands — a kinematic transition, not a density or emission transition, so it's
  a supporting analog rather than the same measurement, but worth citing as convergent evidence.

**Caveats to state explicitly, not bury.**
- Background noise in the outer radial bins is *correlated*, not random (per `research-notes.md`),
  and the cause is still being chased down (annulus definition, field boundary effects, smoothing
  choice). Until that's resolved, some fraction of the apparent turnover's exact location could be
  systematic rather than purely physical — flag this rather than let the ~1 R_vir coincidence stand
  unqualified.
- R_vir = 75 kpc is anchored to a KBSS/LBG-like ~10^12 M_sun halo mass via Moster+2013. If the
  sample is more LAE-dominated than that calibration assumes, halo masses (and R_vir) could run
  lower — recent LAE clustering work (ODIN/DESI, Anand et al. 2024, Ramuta et al. 2025) infers
  log(M_h/M_sun) ~10.9–11.4, noticeably below the ~10^12 KBSS/LBG value. Worth a sentence on how
  sensitive the "cutoff sits at 1 R_vir" claim is to this mass calibration.

---

## 2. Fitting: core exponential + outer decline

**Core term.** h1 = 16.9 ± 1.1 kpc (PSF-aware, all 10 bins, χ²/dof = 1.17 for the shipped
two-exponential model). Comparison set:
- **Steidel et al. 2011** (ApJ 736, 160) — 25.2 kpc (full KBSS sample), 25.6 kpc (emission-line
  subset), 20.8 kpc (absorption-line subset), 28.4 kpc (LAE-like). Our h1 sits below this range but
  is bracketed by it on the low side; the mixed LBG + massive-emission-line-galaxy composition of
  our MOSDEF-selected sample is a plausible reason it doesn't land exactly on Steidel's numbers —
  worth stating directly rather than leaving the reader to guess why we're not at 25 kpc.
  UV continuum for comparison: 3–4 kpc (i.e., halo is ~5–10x more extended than starlight).
- **Wisotzki et al. 2016** (A&A 587, A98) — 1–7 kpc, fainter/lower-mass MUSE sample.
- **Leclercq et al. 2017** (A&A 608, A8) — median ~4.5 kpc, individual (not stacked) galaxies.
- Framing: h1 scales with sample selection (mass/luminosity), and our number's position relative to
  Steidel vs. Wisotzki/Leclercq is itself informative about where our sample sits in that space.

**Outer/power-law term — the more interesting methodological story.**
- A bare power law pinned to the literature's raw 3D clustering slope (γ=1.8) fits *worse* than the
  currently-shipped two-exponential outer term. Reason, worked out via the Limber approximation: a
  3D two-point correlation ξ(r) ~ r^-γ does not project straight onto a 2D radial emission profile —
  fully projected goes as R^-(γ-1), i.e. γ=1.8 → an expected projected slope of ~0.8. Testing both
  fixed values on the exponential-core + cored-power-law (expcore) model:
  - two-exponential (shipped): χ²/dof = 1.17
  - expcore, γ=0.8 (Limber-projected): χ²/dof = 1.11 — best of the four tried
  - expcore, γ=1.8 (raw 3D): χ²/dof = 1.24 — worst
  - The *ordering* (projected beats raw) matches the prediction, but **the differences are not
    statistically decisive** — ΔAIC is below any real threshold with only 6 dof. State this as
    "consistent with, not proof of" the projected slope, not as a settled result.
- **Crossover radius** (term1 = term2): ~76 kpc, robust across both functional forms and
  essentially identical to R_vir — the one clean, model-independent number here. This is the
  quantity that actually maps onto "where the one-halo term stops dominating," not h2 or r_c
  themselves.
- **r_c** (expcore's internal softening/turnover scale) = 488 ± 211 kpc, ≈6.5× R_vir. Independently
  matches **Sorini, Oñorbe, Hennawi & Lukić 2018** (ApJ 859, 125), who calibrate a galaxy's "sphere
  of influence" — where absorption-based Lyα profiles converge onto the pure ΛCDM prediction — at
  ~7× R_vir (~700 kpc–1 Mpc for their sample; 7×75 kpc = 525 kpc here, within 1σ of our r_c). This
  is a genuinely strong result: two totally different observables (their absorption/tomography vs.
  our emission stacking) landing on the same physical scale independently. Worth being a headline
  sentence in the Discussion, not just a table entry.
- **Byrohl et al. 2021** (MNRAS 506, 5129, TNG50 radiative transfer) — outer-profile flattening in
  their simulated stacks is dominated by photons scattered from *other* halos (companions/satellites),
  not diffuse emission from the galaxy's own CGM, with onset around ≥30 pkpc. That's roughly a
  factor of 2–3 below our ~76 kpc crossover — plausibly explained by TNG50's lower-mass simulated
  population vs. our more massive KBSS-like sample, but worth stating as a testable prediction (see
  §3) rather than an assumed explanation.

**Caveats to state explicitly.**
- r_c and γ are degenerate given the current radial coverage: bins only reach ~600–1000 kpc, but
  reaching 90% of the asymptotic slope requires ~3× r_c (~1.5 Mpc). The data sit in the *middle* of
  the turnover, not the clean power-law regime — this is why the free-γ fit wanders (γ≈0.64,
  likely a local minimum) while the fixed-γ comparisons are more stable. State this as a limitation
  of the current radial coverage/S-N, not a flaw in the model.
- The Limber/projection argument has only been checked at the two bracketing slope values (0.8 vs.
  1.8), not computed exactly for this survey's actual line-of-sight window depth vs. a fiducial
  correlation length r0 (Bielby et al. 2016 gives r0 ≈ 2.5–4.5 h⁻¹ Mpc for LBG/LAE-like
  populations). That real number is what would let the projected-slope argument be stated as a
  derivation rather than a plausibility check — flag as a needed follow-up if not done before
  submission.
- Reported uncertainties come from a single central fit's covariance, not a bootstrap over the fit
  itself (refit per bootstrap draw of the underlying spectra). Since r_c/γ trade off non-Gaussian-ly
  when outer bins are noisy, this could matter more for the outer-term uncertainties than for h1.
- On "does our γ connect to the literature 1.8 → 0.8 story": frame it as "our fit prefers the
  projected slope over the raw 3D slope, consistent with the depth of our spectral window relative
  to typical LBG/LAE correlation lengths" — not as "we measure γ" — the current data can't
  distinguish 0.8 from a range of nearby values with any confidence.

---

## 3. Origins of emission — subsample splits

**Candidate splits already scoped** (`subsample_splits.md`): mass, ΣSFR (mass-normalized),
Lyα velocity offset (self-split), redshift bin (z~2 vs z~3), dust attenuation (Balmer
decrement/UV slope β), environment/satellite clustering, Lyα EW, orientation/inclination, and
double-Gaussian peak separation/relative height (core spectroscopic decomposition).

**Direct connection to Byrohl et al. 2021.** Their radiative-transfer decomposition splits the
profile into (a) photons rescattered out of the galaxy's own star-forming region — dominant inner
profile — and (b) photons from other halos — dominant outer flattening. Our subsample splits are an
empirical way to test that picture directly: if the inner term (h1) tracks mass/SFR (an
own-galaxy photon-budget proxy) while the outer term/crossover tracks environment or local
density rather than the individual galaxy's properties, that's direct observational support for
Byrohl's simulation result, not just a consistency check on scale lengths. This should be framed as
the central organizing idea of this Discussion subsection — splits aren't just "does the profile
change," they're a test of *which* term responds to *which* physical variable.

**Supporting/predecessor literature to cite alongside Byrohl:**
- **Steidel et al. 2011 / Momose et al. 2014** — halo strength and extent scale with Lyα EW largely
  independent of broadband SFR/mass; motivates EW as a continuous-variable complement to any
  discrete classification split.
- **Verhamme et al. 2012, Zheng & Wallace 2014** — radiative-transfer prediction that the observed
  kinematic profile shape depends on viewing angle relative to outflow/scattering geometry; relevant
  if an inclination/orientation proxy is available.
- **Bacon et al. 2021** — UV-background fluorescence alone can supply at most ~10–34% of observed
  diffuse filament luminosity; if our outer/diffuse term turns out large even in low-mass/low-SFR
  subsamples, that argues against a simple "more massive galaxies → bigger own-halo diffuse
  emission" story and toward a clustering/other-halo origin (i.e., supports Byrohl's picture over a
  pure own-CGM story) — this is a genuine discriminating test, not just a plausibility note.
- **Trainor et al. 2025** — even Lyα absorbers (LAAs) are net emitters; motivates keeping "absorber"
  in the core classification scheme rather than treating it as a non-emitting control class.

**Tie flux splits to spectroscopic/kinematic splits explicitly** (per your ask — not just flux
amplitude differences): the velocity-offset self-split is a decomposition of the centroid-vs-radius
headline result (accretion/blueshift vs. outflow/redshift), and the double-Gaussian peak
separation/height split gives a direct kinematic handle on HI column density and outflow/inflow
asymmetry. Report these alongside the flux-split amplitudes rather than as a separate section —
the goal is one coherent origins story (flux budget + kinematics together), not two parallel lists.

**Caveats to state explicitly.**
- The double-Gaussian-derived splits (peak separation, flux ratio) haven't yet been checked against
  a noise-prefabrication null test — i.e., whether fitting a two-peak template to pure noise and
  then stacking by the fit's own derived quantities can produce a template-shaped result purely as
  an artifact, especially at high N (open item in `research-notes.md`). Either resolve this before
  asserting a kinematic-origin conclusion from that particular split, or caveat it explicitly in the
  Discussion.
- Splitting costs ~√N in statistical power. The whole-sample centroid signal is itself only
  marginal (recent pooled cross-checks land around 1.5–1.7σ). Crossed splits (e.g., mass ×
  environment) may not clear any meaningful significance threshold — worth stating up front which
  splits are being treated as exploratory/qualitative vs. which are expected to be statistically
  load-bearing.
- The discrete spectroscopic classification scheme (AGN/LAE/absorber/null/emitter) is still being
  pinned down, not finalized. This paper's origins discussion may need to lean on continuous splits
  (mass, SFR, EW, dust) rather than discrete class splits — save the AGN/fluorescence-mechanism
  angle for when the classification scheme is settled (possibly Paper 2), rather than implying it's
  covered here.
- The environment/overdensity split — the single most diagnostic test the field actually uses to
  distinguish two-halo/clustering origin from diffuse/instrumental origin (per the literature
  review) — needs an overdensity or pair catalog not currently in hand. State this as the natural
  next step / limitation, not as something this paper's splits already resolve.

---

## Blindspots — not in your original list, worth considering

1. **Reconcile the three "1 R_vir" numbers.** The draft text says centroid turns over at ≈90 kpc,
   the fit crossover is ~76 kpc, and R_vir itself is quoted as ≈75 kpc (band 55–90). These aren't
   the same number. Either show they're consistent within error, or explicitly discuss the spread —
   right now the Discussion could look internally inconsistent if a reader checks the actual figures
   against each other.
2. **The scale-length "category error."** h1/h2 (exponential scale lengths), r_c (beta-model core
   radius), and r0 (2-point correlation length) are three different kinds of quantities that
   sometimes get compared as if interchangeable in casual language. Worth a short explicit paragraph
   in the Discussion clarifying what each number actually means, so reviewers don't misread "r_c ≈
   7×R_vir matches Sorini's number" as "our clustering length matches theirs" (it's not a
   clustering length at all).
3. **Sample-mass mismatch with the R_vir anchor.** Already noted above under §1, but it's broader
   than just the cutoff radius — it also affects every R_vir-relative number in the paper (crossover
   radius, r_c/R_vir, the "7× R_vir" comparison). Worth a single explicit paragraph addressing this
   rather than letting it silently qualify every R_vir-relative claim.
4. **Correlated (not random) outer-bin noise is still an open systematic**, actively under
   investigation. Any Discussion claim resting on outer-bin behavior (the flux turnover, B/T sign
   change, r_c/γ) inherits this open question. A short "systematics" paragraph up front in the
   Discussion, referenced by the later subsections, would be cleaner than repeating the caveat three
   times.
5. **Population mixing within the stack.** The sample mixes LBGs and massive emission-line galaxies
   (MOSDEF-selected), which Steidel et al. 2011 already show shifts scale lengths by ~8 kpc between
   subsamples for exactly this reason. This is itself a candidate explanation for *why* our h1 isn't
   at the Steidel or Wisotzki value — could be worth a sentence tying population mixing directly to
   the "where does h1 sit" discussion in §2, rather than treating sample composition and scale-length
   comparison as separate topics.
6. **Model comparison honesty.** The χ²/AIC differences between the two-exponential, γ=0.8, and
   γ=1.8 models are small and not statistically decisive with only 6 dof. It would be easy to overstate
   "our data prefers the Limber-projected slope" — worth deciding now how strongly the paper wants to
   lean on this, since a careful referee will check the AIC numbers.
7. **Comparison to direct cosmic-web imaging is currently unconstrained by data**, not by choice —
   Bacon et al. 2021 and the MUSE Ultra Deep Field filament results are cited as scale comparisons in
   the literature review, but nothing in the pipeline currently computes a UVB-fluorescence null line
   or unit-matches to Bacon's measured filament surface brightness. If the Discussion wants to invoke
   these papers quantitatively (not just as "our h2/r_c is roughly this scale"), that unit-matching
   work needs to happen first — otherwise keep the comparison qualitative.
8. **The paper hasn't yet decided between Option A (piecewise/broken exponential, break pinned to
   R_vir) and Option B/C (smooth sum, no explicit break) for the flux-profile fit that actually ships**
   in the Methods/Results section (`specs/flux-profile-exponential-fit.md` still has this as an open
   [DECIDE]). The Discussion's rhetorical framing ("the data show a break at R_vir" vs. "the data show
   a smooth two-component profile whose crossover happens to sit near R_vir") depends on which one is
   chosen — worth resolving before the Discussion prose is finalized, since the two framings support
   are subtly different claims.
9. **Redshift-space anisotropy** (Rakic 2012's finger-of-god vs. Kaiser split) is cited as the
   cleanest observational precedent for the two-term structure, but this pipeline doesn't currently
   have an equivalent velocity-space split implemented. If it's feasible with existing data, it would
   be a strong addition; if not, the paper should be careful about how much rhetorical weight the
   Rakic comparison carries, since it's currently an analogy rather than a matched test.

---

## Notes on this session's context

Per your request, this draft leans on the last several working sessions in this project (fitting
infrastructure, PSF-aware two-exponential/expcore models, the halo-gas clustering literature review,
plotting/validation utility work) plus the standing docs (`GOALS.md`, `research-notes.md`,
`subsample_splits.md`, specs). These sessions are a recent slice of a much longer pipeline — treat
this document as a starting point for your own drafting pass, not a settled account of the science.
One session in the pulled history was unrelated personal content (a running-injury conversation) and
contributed nothing here, which is a useful reminder that "last N sessions" isn't a clean proxy for
"all relevant prior work" — GOALS.md/TODO.md/specs remain the more reliable source of truth for scope
and status.
