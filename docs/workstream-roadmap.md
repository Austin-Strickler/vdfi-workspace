# Workstream roadmap

**Status: v1, 2026-07-29.** Planning document. Organized by **workstream**, not by paper — paper assignment is being revised separately and this doc deliberately does not commit to it. Where a workstream obviously feeds a particular paper, that's noted in passing and should not be treated as authoritative.

Captures a planning conversation of 2026-07-29 covering four threads: **systematics**, **radiative transfer / models**, **the subsample-split pipeline**, and **Lyα budget & escape fraction**. Contested points are preserved in flagged blocks rather than folded silently into the plan.

**Companion:** `../01-lya_halos/specs/wavelength-zero-point.md` — the full test plan for the highest-priority item below.

---

## Contents

1. [Systematics](#1-systematics)
2. [Radiative transfer and models](#2-radiative-transfer-and-models)
3. [The subsample-split pipeline](#3-the-subsample-split-pipeline)
4. [Lyα budget and escape fraction](#4-lyα-budget-and-escape-fraction)
5. [Sequencing](#5-sequencing)
6. [Flagged pushback — index](#6-flagged-pushback--index)

---

## 1. Systematics

### 1.1 The MOSDEF ↔ VDFI velocity zero point — highest priority

**Full treatment in `../01-lya_halos/specs/wavelength-zero-point.md`.** Summary of why it leads the list:

The absolute velocity frame is the project's entire methodological advantage over every competitor. Nobody has verified that the MOSDEF near-IR redshift scale and the VIRUS optical wavelength scale are tied together. Right now the answer to "how did you tie them?" is "we assumed they agree."

**The specific, cheap, live hypothesis: air vs. vacuum.** At the observed wavelength of Lyα for this sample, λ_vac − λ_air ≈ 1.1 Å ≈ **85 km s⁻¹**, and — critically — this is nearly **flat in velocity across the whole VIRUS band** (85.8 km s⁻¹ at 3500 Å, 83.3 at 5500 Å). If a VIRUS wavelength axis is in air and treated as vacuum, everything appears **blueshifted by ~85 km s⁻¹, uniformly**. That is the right sign and roughly the right size for the observed large-radius blueshift. It is a code-and-documentation question answerable in an afternoon, and it must be answered before any physical interpretation is built on the outer-bin sign.

**Bounding the damage — state this in the paper regardless of outcome.** A rigid zero-point error moves every radial bin equally, so it **cannot** produce the headline radial *gradient* across R_vir. It can only invalidate the *absolute* statement about the outer-halo velocity sign. The transition survives; the sign does not. Say it before a referee does.

**Second, independent term: barycentric correction.** Up to ±30 km s⁻¹, varying with season and sky position. COSMOS and EGS sit at very different RA, so a missing barycentric correction produces a **systematic COSMOS-vs-EGS velocity difference** — a free internal test to run at the EGS fold-in.

**The driver is a core extraction on the MOSDEF z~1.5 sample.** 291 candidates across COSMOS + AEGIS, ~200 expected after coverage cuts (Austin, 2026-07-29). That sample carries **four** non-resonant systemic anchors in the VIRUS band — He II 1640, O III] 1661/1666, Si III] 1883/1892, C III] 1907/1909 — spanning **3888–5154 Å**, which brackets where Lyα from the z~2 science sample actually lands (3757–4388 Å). So it returns the offset *and* the wavelength drift from one extraction, and the "does a z~1.5 calibration transfer?" objection largely closes itself. At N = 200 the measurement is **not statistics-limited**: a 3σ detection of an 85 km s⁻¹ offset needs only a stack S/N of ~5. It is limited instead by the **C III] doublet-ratio systematic** (≤62 km s⁻¹ if fit as a single Gaussian) — which is why the specified measurement is a joint fit anchored on the *resolved* Si III] λ1892 rather than a C III] centroid. See `wavelength-zero-point.md` §4.5.

### 1.2 A systematic nobody had flagged: Hubble flow and redshift-space distortion at Mpc radii

> ⚠️ **New concern, not previously in the register.**

At z = 2.3 with the standard cosmology, H(z) ≈ **237 km s⁻¹ per proper Mpc**. The radial bins run to **3000 kpc**. So the outermost bin spans roughly **700 km s⁻¹ of pure Hubble flow**, on top of ~200–300 km s⁻¹ of peculiar velocity from clustered neighbours. (At z = 3 it is worse: 312 km s⁻¹/Mpc, ~940 km s⁻¹ across the bin.)

**Consequence for interpretation.** In the outer bins the measurement is not "gas kinematics." It is the redshift-space distribution of clustered Lyα emitters convolved with real velocity. Any statement of the form "the gas at 2 Mpc is moving at X km s⁻¹" needs to survive that.

**The saving argument, and it is a good one.** Symmetric Hubble flow and symmetric clustering produce line **broadening**, not a **shift**. A centroid shift requires an asymmetry. So the honest framing is: *what could break the symmetry?* Only two candidates — net radial flow (the physical interpretation), or asymmetric attenuation. Which leads directly to:

### 1.3 IGM attenuation pushes the wrong way for the artifact story — use this

At z ~ 2–3 the IGM preferentially removes **blue**-side Lyα photons (Laursen et al. 2011). Therefore an *observed* blueshift is a **lower limit** on the intrinsic blueshift, and the leading "it's an attenuation artifact" objection predicts the opposite sign. This is a Discussion paragraph and a strong one. It also means the effect should strengthen, not weaken, with increasing z — a testable prediction for the z~2 vs. z~3 comparison already in the plan.

### 1.4 The random-position null — the covariance argument may be over-pessimistic

> ⚠️ **Flagged disagreement with `systematics-and-extraction.md` §7.**

§7 concludes: an empirical bin–bin covariance matrix "needs k ≥ 20 independent stacked realizations → ~9000. Unaffordable," and recommends N_r ≈ 3–5N (1400–2300) as an error floor only.

**The objection.** An empirical covariance does not require 20 independent *stacks*. With N_r random positions each yielding a radial profile **vector**, the bin–bin covariance can be estimated directly from those N_r vectors, Hartlap-corrected. That requires N_r ≫ n_bins — order **500–1000** for a 22-bin grid, not 9000. The stated 3–5N would already be enough.

**The real complication, and it is not fatal.** Positions sharing exposures are not independent, so the naive sample covariance is biased for the *stack*. But the cross-position term can be **measured from the same sample** as a function of angular separation, and then propagated. That converts the shared-sky correlation from a blocker into a measured quantity — which is arguably the more valuable output, since it is exactly the correlation a galaxy bootstrap cannot capture (§7's own stated motivation).

**Status: unresolved.** Worth re-deriving before accepting "unaffordable." If it holds, the already-planned 3–5N run returns a covariance matrix instead of just an error floor, at zero extra cost. **Action: revisit §7 before the next full extraction pass; do not pull the run forward on the strength of this alone.**

### 1.5 Already tracked elsewhere

Not repeated here — see `systematics-and-extraction.md` §8 for the live plan: B10/B11 background rise-and-fall, EGS fold-in and §2.6 re-run, the §3.6 star re-extraction, the §4 projection formalism, and the injection–recovery transfer function.

---

## 2. Radiative transfer and models

### 2.1 The strategic position: do not build an RT model

**Read to sharpen the question, not to fit anything.** With a stacked, medium-resolution, single-line measurement, the shell-model parameter space (N_HI, v_exp, b, τ_dust, intrinsic line width) is unconstrainable. Gronke, Bull & Dijkstra (2015) showed that even statistically *good* shell-model fits return parameters that do not correspond to the underlying physics. The contribution here is the measurement; the model belongs to a collaborator.

The productive output of the reading is a **Discussion section on why the degeneracy exists**, which is more defensible than a fit and is what makes the collaboration pitch credible.

### 2.2 Sharpening the degeneracy statement

> ⚠️ **Correction to the framing used in the planning conversation.**

The framing was: inflowing HI wall (scattering → blueshift) *vs.* outflowing satellites (emission → blueshift). **Both of those predict blue.** Scattering off *infalling* gas preferentially transmits blue photons; in-situ emission from gas moving toward the observer is also blue. So the two hypotheses are **not** distinguished by the centroid — that *is* the degeneracy, stated precisely.

**What does distinguish them:**

| Discriminant | Scattered core light | In-situ emission |
|---|---|---|
| Blue-to-red flux ratio / line shape | Set by RT through the medium | Set by the emitters' own kinematics |
| **Environment / overdensity dependence** | Weak — tracks the central galaxy | Strong — tracks satellite density |
| **Core Lyα luminosity dependence** | Strong — halo flux tracks core flux | Weak — halo flux decouples from core |

The last two are already planned measurements (the Chartab overdensity split, and the core↔halo flux relation). `blue_red_side_ratio` already exists in `measure.py`.

**The unifying conclusion:** *the subsample-split infrastructure is the RT-degeneracy-breaking tool.* Workstream 3 is not merely bookkeeping — it is the instrument that answers the question workstream 2 poses.

### 2.3 The collaboration pitch

Reframe the approach to Byrohl / Li. Not "help me model this" (open-ended, low commitment, easy to decline) but:

> *"We have a Lyα halo measurement in an absolute velocity frame — the only one of its kind, anchored to independent rest-optical systemic redshifts — split by environment and by core Lyα luminosity. What does TNG50 predict for those two splits specifically?"*

That is a defined deliverable, it maps directly onto the origin-decomposition work Byrohl has already published, and it is a co-authorship proposal rather than a favour.

**Send this now, in parallel with everything else.** Other people's calendars are the long pole; the reading is not.

### 2.4 Reading list, ordered by usefulness

1. **Dijkstra 2017 lecture notes** (arXiv:1704.03416) — the physics, and the clearest available statement of what is and isn't degenerate.
2. **Byrohl et al. 2021** and **Byrohl & Nelson 2023** — photon-origin decomposition in TNG50. The direct quantitative target, *and* the collaborator's own work. Read before sending the email.
3. **Gronke, Bull & Dijkstra 2015** — why shell-model fits don't recover truth. The citation for declining to fit a model.
4. **Laursen, Sommer-Larsen & Razoumov 2011** — IGM transmission at z ~ 2–3; underwrites §1.3.
5. **Mitchell et al. 2021** — RT in zoom simulations; closest to "what physically makes a halo."
6. **Verhamme et al. 2006, 2015** — the shell formalism, for shared vocabulary in the collaboration conversation.

### 2.5 LIS lines land here, not in calibration

Per `wavelength-zero-point.md` §6, the LIS-line measurement is demoted as a wavelength calibrator (they carry an intrinsic 150–200 km s⁻¹ outflow blueshift, larger than the systematic being searched for) but **retained as physics**: a stacked LIS velocity is a direct outflow-velocity measurement for the sample, a real constraint on the scattering medium, a candidate split variable, and a literature comparison against Steidel 2010 on a sample with independent systemic redshifts.

---

## 3. The subsample-split pipeline

### 3.1 Three layers, not one

The requirement as described bundles three things that should be separated. The separation is the point — it is what turns "run a split" from a research task into a lookup.

**Layer A — the split catalog.** One row per galaxy, one column per splittable quantity, `NaN` where unavailable, provenance-tracked, cached, versioned. Sources: `multicat.py` (mass, SFR, dust, z, Chartab overdensity), core measurements (L_Lyα, Δv_Lyα, EW, classification), imaging (size, inclination).

> **This is the load-bearing piece.** It is what both the splits work and the photon-budget work consume, and it is the highest leverage per unit effort available in the repo right now.

**Layer B — the split spec.** Declarative and serializable: `{column, method (median | tertile | fixed edges | controlled-match), control_column, min_N, label}`. A list of these *is* the "list of splits to run."

**Layer C — the runner.** Per spec: slice → stack → bootstrap → fit → `describe_subsample_properties` → one table row. Cache keyed on `(split_spec_hash, stage1_signature, measure_params_hash)`.

### 3.2 The core-extraction join

Splits on mass, SFR, dust, size, and redshift come from catalogs. Splits on **L_Lyα, Δv_Lyα, EW, and spectral class** require the PSF-extracted core spectrum, and some galaxies sit off the field — halo coverage without core coverage.

**Design:** run the core extraction + classification + measurement as one pass, write to a keyed table, left-join onto the existing catalog, leave `NaN` where there is no core coverage. `core.py` already has the pieces (`extract_core_spectra`, `run_core_hierarchical`, `classify_lya`, `write_core_hier_fits`, `core_results_table`); this is an orchestration and join problem, not new measurement code.

### 3.3 Four things that will bite, in order of severity

**(a) The cache key must include the Stage-1 signature.** The SMHM default changed 2026-07-27 and the PSF area convention changed 2026-07-22. A split-results cache that does not invalidate on those will silently serve numbers from two different pipelines **in the same table**. Reuse the extraction cache's signature pattern.

**(b) `NaN` handling is not neutral.** Galaxies off the core field drop out of L_Lyα and Δv splits but remain in mass and z splits. The two splits are therefore run on **different samples**, and any cross-split comparison ("the L_Lyα split shows a bigger effect than the mass split") is confounded by sample composition rather than physics.

- *Minimum:* record the sample mask per split and report N.
- *Better:* run every split twice — once on all available, once on the **common intersection** — and use the intersection version whenever splits are compared to each other.

**(c) Statistical power should decide which splits run now.** ⚠️ **This is a hard limit and it is easy to discover too late.**

N ≈ 452 after AGN cuts. Register §9 gives full-stack outer-bin bootstrap S/N of **6, 5, 2.8**. Splitting:

| Split | Outer-3 bins S/N |
|---|---|
| Full stack | 6.0, 5.0, **2.8** |
| Two-way (N ≈ 226) | 4.2, 3.5, **2.0** |
| Three-way (N ≈ 151) | 3.5, 2.9, **1.6** |

**No split supports a >2σ outermost-bin claim with current data.** Consequence: splits are worth running *now* for **inner / one-halo / core-region** quantities (h1, core luminosity, core velocity, crossover radius), and every **two-halo** claim from a split should be explicitly deferred to the incoming ~2× data. Bake this into the output table as a per-quantity power flag rather than rediscovering it per split.

**(d) Pre-register, or multiple comparisons will eat the paper.** Fourteen splits × several derived quantities is 40–70 tests. Something will come back at p < 0.05 by chance — near-certainly. Before running the batch, write down which splits are **confirmatory** (a specific prediction stated in advance — the Lyα velocity self-split, environment, f_esc) and which are **exploratory**, and report exploratory results with a look-elsewhere correction or as "suggestive, requires the 2× data." Costs nothing now; saves the paper later.

### 3.4 Mechanics

- Bootstrap is the ~20 min/split cost. **Precompute the bootstrap index array once per product and reuse it across every split** — same galaxy pool. Bonus: it makes splits correlated in a trackable way rather than an unknown one.
- Parallelize **across** splits, not within. Splits are embarrassingly parallel; the bootstrap inner loop is already the optimized path.
- Resume/append: cache per split row, not per batch, so an interrupted batch resumes rather than restarts.

---

## 4. Lyα budget and escape fraction

### 4.1 f_esc(<R) — the strongest idea on the list

**Rank this above the splits for novelty.** It is the only proposed test that yields a *falsifiable, model-independent* statement.

**Construction.** Take L_Lyα,intrinsic = R_CaseB × L_Hα,intrinsic; dust-correct Hα using the MOSDEF Balmer decrement; compare against the observed Lyα curve of growth (`measure.py::flux_curve_of_growth_annulus`, already built). This yields **f_esc(< R)** as a function of radius, which must be monotonically increasing and bounded above by 1.

**Why it is strong:**

- If f_esc(<R) **crosses 1** at any radius, in-situ production beyond that radius is *required* — by photon conservation alone. No RT model, no scattering-geometry assumption, nothing degenerate.
- If it **saturates below 1**, the in-situ fraction is bounded from above.

Either outcome is a result, and neither depends on anything that a referee can argue about on modelling grounds. It is the natural headline for the escape-fraction paper.

### 4.2 Three gotchas, ordered by how badly they hurt

**(a) Aperture mismatch — the big one.** MOSDEF Hα comes from a **0.7″ MOSFIRE slit**. The Lyα is a total from a fiber IFU integrated to Mpc scales. A slit-loss-affected numerator over an aperture-corrected denominator (or vice versa) is where this measurement most commonly goes wrong in the literature, and it is the first thing a referee will check. Get the slit-loss correction explicit, stated, and propagated — do not inherit it silently from a catalog column.

**(b) The Case B ratio is a band, not a number.** L_Lyα/L_Hα ≈ 8.7 is the T = 10⁴ K, Case B value; it ranges roughly **8.1–11.4** over T = 5,000–20,000 K and depends on density. Quote and propagate the band. Also confirm the AGN exclusion is applied consistently on both sides of the ratio.

**(c) Hβ detections are sparser than Hα in MOSDEF.** Check N *before* designing the analysis around the Balmer decrement; the sample with both may be materially smaller than the Lyα sample.

### 4.3 Per-galaxy halo scale lengths — pushed back on

> ⚠️ **Two independent objections. The second is a methodological failure, not a nitpick.**

**Objection 1 — the S/N is not there.** `single.py`'s own docstring states it: a single galaxy's per-bin S/N is the stack's divided by √N ≈ 21. With full-stack outer bins at S/N 2.8, a typical individual galaxy is at **≈ 0.13**. Per-object h1 fits are possible only for the brightest handful — AGN and strong LAEs — which are precisely the objects being cut.

**Objection 2 — splitting on a noisy fit of the quantity you then measure is circular.** Selecting galaxies into "large h1" and "small h1" bins using a noisy fitted h1, then measuring h1 (or profile shape) per bin, produces **regression to the mean and Eddington bias**: the "large h1" bin is preferentially populated by objects that scattered high, and it will regress downward on remeasurement. The resulting "relation" is partly manufactured by the selection.

**What works instead.** Sort galaxies by a **cleanly measured** property (mass, SFR, EW, environment), stack in 4–5 groups, fit h1 per group. This yields **h1 vs. property** — the actual scientific content — with no per-object fit anywhere in the chain. The follow-on question ("with a five-parameter SED, what best distinguishes a large from a small halo?") then becomes a regression of stacked-group h1 on group-mean properties, or a hierarchical model. Not a per-object split.

### 4.4 O32 and burstiness

**O32** is physically well-motivated — ionization parameter and density-bounded escape correlate with both LyC and Lyα escape. **But** at z ~ 2.3, [O II] λ3727 and [O III] λ5007 land in different MOSFIRE bands, so the subsample with both is smaller than the Hα sample. **Check N before scoping.**

**Burstiness** already carries its own caveat in `subsample_splits.md` #10: matched radii are PSF-dominated and the timescales reach only ~0.3 R_vir, so it cannot speak to the R_vir-scale transition with current data. Unchanged by this conversation — it waits for the 2× data.

### 4.5 The f_esc split is partly circular with an L_Lyα split

> ⚠️ **Flag, not a disqualification.**

f_esc ∝ L_Lyα / L_Hα,corr. Splitting by f_esc is therefore close to splitting by L_Lyα *at fixed Hα*. That is not fatal — it is still a meaningful axis — but it must be stated, and L_Hα should be controlled for so the split is not silently reproducing an L_Lyα split under a different name.

---

## 5. Sequencing

**Ordered by (a) what can change what the paper says, and (b) what makes everything after it cheaper.**

| # | Item | Workstream | Why here |
|---|---|---|---|
| 1 | **Wavelength zero-point audit** — Tier 0 + Tier 1 (`wavelength-zero-point.md` §7) | Systematics | Days of work; can invalidate the sign of a headline number; blocks nothing else from starting in parallel |
| 2 | **Email Byrohl and Li** with the two-split pitch (§2.3) | RT | Zero marginal cost, longest latency. Do it the same day |
| 3 | **Split catalog (Layer A)** | Splits | Highest leverage per unit effort; consumed by both the splits work and the photon budget |
| 4 | **f_esc(<R) budget curve** | Budget | Highest novelty; machinery largely exists; independent of 1–3 |
| 5 | **Splits runner (Layers B/C)** | Splits | After Layer A, with the power flags (§3.3c) and pre-registration (§3.3d) built in from the start |
| 6 | **Random-position null** | Systematics | Expensive; already scheduled as register §8 item 17. Do **not** pull forward — but do revisit the covariance argument (§1.4) before then |
| 7 | **RT reading** | RT | Continuous, in parallel with everything |

**The distinction worth holding onto:** items 1 and 4 can change *what the paper says*. Items 3 and 5 are infrastructure that make everything downstream cheaper. Item 5 is more enjoyable than item 3 and should still not be done first.

---

## 6. Flagged pushback — index

Contested points preserved deliberately, following the register's convention. Each is a place where the plan as originally described has a problem.

| Where | The claim | The objection |
|---|---|---|
| `wavelength-zero-point.md` §6 | LIS lines can calibrate the redshift scale | They carry an intrinsic 150–200 km s⁻¹ outflow blueshift — larger than the systematic being searched for — with no independent handle on it. Retained as physics, removed from the calibration path |
| §1.4 | Register §7: empirical covariance needs ~9000 randoms, unaffordable | Covariance is estimable from N_r profile *vectors* (order 500–1000 for 22 bins), not 20 independent stacks. Shared-sky correlation is measurable, not blocking. **Unresolved** |
| §2.2 | Inflow-scattering vs. outflow-emission are competing explanations for the blueshift | Both predict blue. Centroid cannot distinguish them; the splits can |
| §3.3c | Splits can be run and interpreted as-is | Every split's outermost bin falls to ≤2σ. Two-halo claims from splits must wait for the 2× data |
| §3.3d | Run the fourteen splits and see what shows up | 40–70 tests guarantees a false positive. Pre-register confirmatory vs. exploratory before running |
| §4.3 | Fit per-galaxy halo scale lengths and split on large vs. small | Per-object S/N ≈ 0.13 in outer bins; and splitting on a noisy fit of the measured quantity is circular (regression to the mean). Use stacked property-ordered groups instead |
| §4.5 | f_esc is an independent split axis | Partly circular with L_Lyα at fixed Hα. Control for L_Hα and state it |
| §1.2 | Outer-bin velocities are gas kinematics | At 3000 kpc, ~700 km s⁻¹ of Hubble flow plus peculiar velocities dominate. Only an *asymmetry* produces a centroid shift — which is the argument to make, not to avoid |

---

## See also

- `../01-lya_halos/specs/wavelength-zero-point.md` — the full test plan for §1.1
- `../01-lya_halos/specs/systematics-and-extraction.md` — the systematics register (§7 covariance, §8 live plan)
- `../01-lya_halos/docs/GOALS.md` — Paper I scope and the absolute-frame claim
- `../01-lya_halos/docs/subsample_splits.md` — the split list §3 operationalizes
- `../01-lya_halos/specs/subsample-derived-properties.md` — the per-split derived-quantity machinery
- `../01-lya_halos/docs/Greg-Origins-and-Structure-Outline.md` — where §2 and §4 land
- `paper-writing/three-paper-program.md` — paper assignment (under revision; not authoritative here)
- `../TODO.md` — near-term working items
