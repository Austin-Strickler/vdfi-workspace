# Background diagnostics — testing the raw background spectra

**Status: design v4, 2026-07-26.** Written *before* the data exists, deliberately.

- **v2** — Austin's first review: §0 added (the fiducial annulus sits inside the detected-signal range, reframing the test from detection to prediction-checking); redshift-scramble control dropped as answering the wrong question; `rest_delta` kept at 0.2 to match production; the existing `(1150, 1300)` line-free window adopted; anchor standoff spelled out.
- **v3** — Austin's second review, two substantive corrections. **(1)** v2's §0 treated "the annulus contains signal" as straightforwardly bad. It isn't: the mask exists so the line is not subtracted, and the competing failure — an annulus too far from the science bins to represent their background — pushes annulus radius the *other* way. §0.1 rewritten around both failure modes and the architectural question underneath them; §2.4 gained a second reading that measures it. **(2)** The anchor standoff was over-claimed. Excluding the mask and widening outward (Austin's simpler scheme) is now the default, `anchor_standoff_A = 0`, retained only as a swept parameter for the injection test to accept or reject.
- **v4** — Austin's third review, taken while the run is in flight, and it changes the shape of the whole document. Five things. **(1)** The variants that are actually running go *inward*, not outward — `((80,90), (60,70), (40,50), (110,130))` — which makes this a map of the background across the science range rather than a search for a clean outer radius (§0). **(2)** The stated purpose is now explicit and narrower: **surgically remove the Lyα that is genuinely present in the annulus while leaving a valid background that still removes the systematics it exists to remove** (§0.1). **(3)** The variable-radius background is a *concept under consideration*, deliberately not implemented, with two hard constraints recorded so a future design does not rediscover them (§0.3). **(4)** **The fitting emphasis throughout v1–v3 was wrong and is removed.** The primary science of this paper is a measured, spectroscopically-resolved signal and centroid out to several Mpc; the profile fit is a secondary product for literature comparison. Diagnostics are therefore judged at the level of *bin values and visual inspection*, not Δh1 (§0.4, §3.4). **(5)** Scope discipline: the deliverable for most of this is one sentence in Methods, and the toolkit should be sized to produce that sentence and stop (§0.4). Several v3 open questions are closed inline.

Companion to `specs/systematics-and-extraction.md` §1.1, §1.1b, §2.2, §2.5. This document specifies the Stage-2 toolkit that consumes `BG_VARIANTS` / `BG_NFIB` / `SPEC_NOBG` from the §2.5 custom extraction.

## Why write this now

§2.5 is a ~1.9 hr run that produces arrays nothing can currently read. The failure mode to avoid is landing the data and then improvising analysis against it — because the first plot that shows *something* at 1215.67 Å will be very persuasive, and by then there is no pre-registered standard to judge it against. §2.5.1 already commits to a criterion; this document specifies the machinery that evaluates it, and it should be built while the answer is still unknown.

Everything here is Stage 2 and free to re-run. Nothing here blocks the §2.5 extraction — the two can be built in parallel, which is the point.

---

## 0. The controlling fact: where the background annuli actually sit

**Austin, 2026-07-26 — this reframes the whole test and must be understood before any of it is interpreted.**

**v4: the annuli that are actually running are not the ones v1–v3 assumed.** The launched config is

```python
cfg_bg = replace(cfg3,
    bins=[0, 20, 50, 100, 200, 500, 1000, 2000],   # coarse: the test quantity is the BACKGROUND
    num_gal=100,
    bg_variants=((80., 90.), (60., 70.), (40., 50.), (110., 130.)),
    store_raw_bg=True, store_spec_nobg=True, store_bg_halves=True,
    combine_threads=8)
```

At z = 2.3, 1″ ≈ 8.25 kpc, so:

| variant | arcsec | kpc | which coarse science bin it sits in |
|---|---|---|---|
| 0 (fiducial) | 80–90″ | 660–743 kpc | inside 500–1000 |
| 1 | 60–70″ | 495–578 kpc | inside 500–1000 |
| 2 | 40–50″ | 330–413 kpc | **inside 200–500** |
| 3 | 110–130″ | 908–1073 kpc | straddles 500–1000 and 1000–2000 |

**This is a different experiment from the one v1–v3 specified, and a better-aimed one.** v3 assumed variants marching *outward* (120–150″, 150–200″) to find a radius where Lyα is definitely gone. Three of the four running variants instead sit *inward* of the fiducial, and all four lie inside the radial range the science bins occupy. Two consequences:

1. **The primary reading is now the radial map, not the outer null.** v3's §2.4 "Reading 2" — how the background level itself behaves between 330 and 1073 kpc — was already argued to be the more valuable of the two readings. With this variant set it is the *only* reading with a real lever arm, and §2.4 is rewritten accordingly.
2. **The weak-null problem of §0.2 is partly solved.** At 40–50″ and 60–70″ the profile has real, well-detected Lyα. Those variants are not marginal nulls to be over-read — they are measurements, and the *trend* from 330 → 1073 kpc is the quantity with signal in it.

The existing measurement detects Lyα at **S/N ≈ 5 over 600–1000 kpc**. Every one of these annuli is at or inside that range.

**The 110–130″ variant carries a specific question of its own.** It is the "go bigger / go further out" test: does moving the background annulus outward to ~900–1070 kpc give a cleaner background, or does it just trade Lyα contamination for a worse spatial match to the bins? Comparing 40–50″ against 110–130″ is the widest lever this run has and is where the answer lives.

### 0.1 What background subtraction is actually for — and why "the annulus contains signal" is not by itself the problem

**Austin's correction to the first draft of this section, and it is the right frame.** An earlier version read "the annulus is inside the signal" as straightforwardly bad and concluded the annulus should move outward. That is one-sided.

**The purpose, stated plainly (Austin, v4).** Because the extraction reaches to several Mpc, the background annulus unavoidably contains real Lyα. The job is therefore not "find a signal-free annulus" — there may not be one at a usable radius. The job is:

> **Surgically remove the Lyα that is present in the annulus, while leaving a background that is still a valid estimate of the systematic fluctuations — sky residuals, scattered light, instrumental pedestal — that would otherwise plague the measurement.**

Both halves of that sentence are load-bearing. Removing the line too aggressively (a very wide mask, a filler that drags the continuum) damages the background's ability to do its actual job. Removing it too timidly leaves Lyα in the background and over-subtracts the science bins. **Every knob in this document is a point on that trade-off, and every diagnostic exists to locate it.**

**The goal, expressed as a property of the output.** Background subtraction should leave a science bin containing *the line and a flat zero*. For bins of ~100 kpc and larger the intrinsic non-line background is zero, so a correct subtraction produces line + flat zero, and nothing else.

Given that goal, **Lyα emission being present in the annulus is not the failure** — the mask exists precisely so the line is not subtracted. What remains after masking is sky, scattered light, and instrumental pedestal, which is the thing we actually want to remove. The residual worry from systematics §1.1 is narrow: the *leak* around the mask, not the line itself.

**There are two failure modes, and they push in opposite directions on annulus radius.**

| | Failure mode A — signal contamination | Failure mode B — spatial mismatch |
|---|---|---|
| **What goes wrong** | Lyα in the annulus leaks past the mask and gets subtracted from the science bins | The annulus is too far from the bin to represent that bin's actual background |
| **Direction of the bias** | Over-subtraction; outer profile biased down | Wrong pedestal; injects annulus noise into every bin |
| **Fixed by** | moving the annulus **outward**, wider mask | moving the annulus **inward**, closer to the bins |
| **Handled today by** | `line_mask_halfwidth_A`, the 30 Å smooth | nothing — the annulus is fixed at 80–90″ |

**Failure mode B is the one this document previously ignored, and Austin's point is that it may be the larger of the two.** A background measured at 200″ and subtracted from a bin at 10″ is spatially out of touch: it is a different part of the IFU, different fibers, different amp/IFU sky residuals (§6.4). Using it degrades rather than improves the subtraction, adding noise that propagates straight into the line measurement.

**The architectural problem underneath both.** A single per-galaxy background spectrum is subtracted from *every* bin, from 0–10 kpc (≈1.2″) out to 2700 kpc (≈327″). **No annulus is local to all of them.** At 80–90″ the background is far outside the inner bins and well inside the outer ones. So "which annulus is best" has no single answer unless the background is spatially flat — and **B10/B11 is direct evidence that it is not**, since the background level shows a monotonic rise-and-fall with radius.

That elevates a question that was previously implicit:

> **Is a single global annulus the right architecture at all**, or should the background be a smooth function of radius, evaluated per bin?

`BG_VARIANTS` at three radii is exactly the measurement that informs this, because it samples the background's own radial dependence. That reading is at least as valuable as the mask-width answer and should be produced in the same pass (§2.4).

### 0.2 The S/N caveat — weaker in v4, but not gone

The S/N ≈ 5 detection pools *600–1000 kpc*. The fiducial annulus spans only 660–743 kpc — a far narrower radial slice with correspondingly fewer fibers. Per-annulus S/N will be **below 5**, and on a ~100-galaxy subset lower still.

**v4 softens this for the inner variants and keeps it for the outer one.** At 40–50″ and 60–70″ the surface brightness is substantially higher and the annuli are inside the well-detected range, so a feature should be *seen*, not merely bounded. At 110–130″ the original caveat stands in full.

What survives from v3, unchanged:

1. **A null at 110–130″ must not be written as "the annulus is line-free."** It is an upper limit, and should be quoted as one.
2. **The trend across variants carries more weight than any single annulus.** 330 → 1073 kpc is a factor of 3 in radius; the *shape* of the amplitude-vs-radius curve is measurable even where individual points are marginal, and it is the quantity that distinguishes real clustered Lyα (falls with radius) from an instrumental pedestal (flat).
3. **Visual inspection is a legitimate first pass and often the last one.** See §0.4 — several of these questions are answered by looking at four stacked spectra side by side, and formalising them further buys nothing that reaches the paper.

What v4 **drops**: v3 made §1.2's sensitivity estimate *mandatory* and made "compare the measured amplitude against the profile-predicted amplitude" the headline deliverable. Both were built on the assumption that every annulus was a marginal null. With signal expected in two of the four, the prediction comparison is demoted to a nice-to-have (§1.2) and the headline deliverable is the variant trend (§2.4).

### 0.3 The variable-radius background — a concept, deliberately not implemented

**Austin, v4: this is being pushed off, on purpose.** It is the thing this run might *motivate*, not the thing this run builds. Recording it here so the constraints are not rediscovered later.

**The idea.** Rather than one annulus subtracted from every bin, use a background that depends on radius — either **two independent backgrounds** (an inner one and an outer one, with a switch), or **a smooth function** interpolated between two or more measured annuli and evaluated per bin. §0.1's failure mode B is the motivation: a single background cannot be local to bins spanning 1.2″ to 327″.

**Constraint 1 — extraction cost, and where it actually bites.** Each additional annulus is another fiber selection and another robust combine at extraction time. The cost is *not* uniform: it scales with annulus area, so adding an inner annulus is nearly free while adding a far-out one is expensive. §2.5's measured example — 80–90″ is 5,340 arcsec² but 150–200″ is 55,000 — is the illustration. **So a variable-background scheme with several closely-spaced inner annuli is cheap; one that reaches far out is not.** Any future design should spend its annuli where they are cheap.

**Constraint 2 — self-subtraction, which is the harder one.** Every annulus in this run lies *inside* a science bin (see §0's table). A background measured at 40–50″ and subtracted from the 200–500 kpc bin is that bin subtracted from itself: the result is forced toward zero regardless of what is there. This is not a flaw in the current run — the inner variants are being *measured*, not subtracted — but it is a hard constraint on any variable scheme:

> **A bin must never be subtracted using a background drawn from fibers inside that bin.** Whatever the eventual architecture, the mapping from bin to annulus needs an explicit non-overlap rule, and it needs stating in Methods.

This is the reason a naive "use the nearest annulus to each bin" scheme is wrong even though it sounds obviously right.

### 0.4 Scope discipline — what the deliverable actually is

**Austin, v4, and this correction applies to v1–v3 throughout.** The specs have drifted toward treating the profile fit as the object of the exercise. It is not.

**The primary science of this paper** is: a consistently measured Lyα halo signal out to several Mpc, measured *spectroscopically*, with a centroid that — combined with the systemic redshift — probes velocity structure at radii nobody has reached before. **The fit is secondary**: it exists so the numbers can be compared against the literature. It is not the science driver and it is not what these diagnostics should be optimising or judged against.

Two operational consequences, and they should be applied to every function in this document:

1. **Judge diagnostics at the level of bin values and visual inspection, not fitted parameters.** "Does the outer profile change when I move the background annulus" is answered by overplotting the binned profiles and looking. Δh1 is a worse instrument for the same question — it compresses a whole profile into one number, and on the coarse grid this run uses it is not even well determined (§3.4).
2. **Size the machinery to the sentence it has to produce.** Most of this ends up in the paper as one line: *"We tested X and it does not change Y,"* or *"We selected Z by optimising W."* Being exact and precise matters; going deeper than the sentence requires is the failure mode to avoid, because it is the thing that consumes the project without improving it. **When a diagnostic can be settled by eye on a plot, settle it by eye and write the sentence.**

**Runtime budget follows from this:** a few minutes maximum per call, so that tuning is a loop you actually run rather than a job you schedule.

---

### 0.5 The two questions, kept separate

They use the same data and get conflated constantly. They are not the same question and they have different success criteria.

**Q1 — Is there Lyα in the background annulus, and how wide is it?**
A measurement. Answered by stacking. Determines `line_mask_halfwidth_A`. Success = a number with an uncertainty and a null control.

**Q2 — Given a gap of that width, what is the best way to fill it?**
An estimator question. Answered per-galaxy, and — critically — answerable *without knowing the answer to Q1*, using line-free regions of the same spectra as ground truth (§3). Success = one filler demonstrably lower-bias and lower-variance than `np.interp`-on-two-points.

Q2 does not depend on Q1, which is useful: **§3 can be built and validated on the existing stored backgrounds' wavelength grid the moment raw data lands, and its verdict is independent of whatever Q1 finds.**

**v4 sequencing decision: Q1 first, Q2 deferred.** Q2 (the filler comparison, §3) needs none of the variant data and can be run at any time. Q1 is what the in-flight run exists to answer. Build Q1's toolkit now; revisit Q2 once the architecture question in §0.3 has an answer, since a variable-radius background would change what the filler has to do.

---

## 1. Data access layer

### 1.1 ~~`BackgroundProduct`~~ — **superseded, this already exists**

**v4: do not build this.** The §2.5 implementation put every one of these arrays on `GalaxyProduct` directly (`io.py`): `spec_nobg`, `bg_variants`, `bg_nfib`, `bg_halves`, plus a `bg_annuli` property that reads the `BGVAR*` header keys. A second dataclass would be a parallel representation of data that already round-trips correctly. **Everything below reads `GalaxyProduct`.**

What the v3 requirements list was right about, retained as checks to perform rather than a class to write:

- **`field` must be reachable.** It comes off `product.catalog`, not off a dedicated attribute. Confirm the column is present and usable for splitting before writing anything that assumes it.
- **`bg_nfib` travels with `bg_raw`.** It is the stack weight and the first thing to check on an outlier galaxy.
- **`bg_annuli[0]` must equal the config's fiducial pair.** `extract.py`'s validator enforces this at write time; assert it again on read, since the whole toolkit assumes variant 0 is the one that was actually subtracted.

The original shape reference, kept for orientation:

```
bg_raw     (ngal, nvar, nwave)   raw, unmasked, unsmoothed annulus coadds
bg_nfib    (ngal, nvar)          fibers per variant per galaxy
bg_halves  (ngal, nvar, 2, nwave)  optional; azimuthal split
annuli     (nvar, 2)             the (inner, outer) arcsec pairs
wave       (nwave,)              observed frame
z          (ngal,)               systemic redshifts, from the run table
field      (ngal,)               'COSMOS' | 'AEGIS' — needed throughout (§6.6)
spec_nobg  (ngal, nrad, nwave)   optional
```

### 1.2 A pre-flight S/N estimate — **demoted in v4, optional**

v3 made this the mandatory first build and the gate on whether the test could work at all. **Austin's call: not a gate.** The reasoning is §0 — with two variants sitting inside the well-detected range, "is there enough signal to see anything" is answered by stacking and looking at the error band, which the existing `stack_backgrounds` already returns. A separate pre-flight estimator that predicts what the stack will show, before the stack is run, is machinery for a marginal experiment this no longer is.

**What stays useful, if it is cheap:** a way to ask *how much residual Lyα is in each annulus, and is it aligned at the systemic redshift* — i.e. an amplitude with an error bar per variant. That is `measure_line_residual` applied across variants, which §2.4 does anyway. If a standalone sensitivity number turns out to be wanted later it can be added then.

```
estimate_stack_sensitivity(product, profile=None, variant=0, width_A=10.0) -> dict   # OPTIONAL
```

The v3 design, kept for reference if it is ever built: per-galaxy pixel-to-pixel scatter in the line-free window (§1.3), propagated through the stack weights, giving `sigma_stack`; optionally compared against `predicted`, the fitted profile's surface brightness integrated over the annulus, with `ratio = predicted / sigma_stack`. Reported per variant and per field.

**Do not make anything else depend on it.**

### 1.3 The line-free window — reuse the existing one

Austin: a line-free window was already used in the earlier background testing, and reusing it keeps everything consistent. It is `optimize.continuum_noise_metric`:

```
window = (1150.0, 1300.0)          # rest Å
lines  = DEFAULT_UV_LINES
line_hw = 6.0                      # rest Å masked around each
```

**Adopt this verbatim** as the shared definition for: the noise estimate in §1.2, the placebo-gap locations in §3.1, and the per-galaxy continuum fit in §2.2. Define it once in the new module (or import `DEFAULT_UV_LINES` directly) so the three uses cannot drift apart.

One caveat to check rather than assume: that window was chosen for *science stacks*. The background is sky + scattered light, so the sky-residual structure inside 1150–1300 Å rest differs galaxy to galaxy because each galaxy's z maps that window to a different observed range. Verify on real backgrounds that the window is still clean in practice; if particular galaxies land it on a strong sky residual, that is a per-galaxy rejection criterion (§2.1), not a reason to change the window.

---

## 2. Q1 — What is in the background?

### 2.1 Look at individual galaxies first

Austin's instinct, and it's the right order. Before any stack:

```
plot_background_gallery(product, config, indices=None, variant=0, n=12, rest_frame=False)
```

A grid of individual raw backgrounds, with the Lyα mask region shaded, the continuum sidebands shaded differently, and `nfib` + `z` + `field` annotated per panel. Options to overlay the smoothed version and the current production filler's output.

What this is for: catching the things a stack hides. Galaxies with pathological `nfib`, backgrounds dominated by a bright sky residual, NaN structure, continuum slopes that differ wildly between fields. **Expect to throw galaxies out, and expect the rejection criterion to come from looking at these panels rather than from theory.** Whatever that criterion turns out to be, it gets written down and applied uniformly, not per-galaxy by eye.

### 2.2 The stack — **already implemented, with two deliberate departures**

```
stack_backgrounds(bgprod, variant=0, frame="rest", continuum="sideband",
                  weight="nfib", rest_delta=0.6, n_boot=2000) -> dict
```

**v4: this shipped in `background.py` on 2026-07-26, before this section was reconciled against it.** The real signature is `stack_backgrounds(product, config, *, variant, frame, rest_grid, combine="biweight", n_boot=1000, continuum=True, cont_bounds=BG_CONT_BOUNDS, ...)`. Two departures from the design below, both intentional in the implementation, and the spec is the thing that was out of date:

- **Wider continuum sidebands.** The design (a) below says use the *science* sidebands. The implementation uses `BG_CONT_BOUNDS = ((1150, 1190), (1245, 1290))`, ~25 Å further out on each side, on the grounds that the two-halo feature being looked for is broad and would leak into the science sidebands and be partly subtracted away as "continuum" — the test would then under-report the thing it exists to detect. **That reasoning is better than (a)'s and stands.** The cost is that (a)'s "what survives here survives there" argument no longer holds exactly, which should be stated wherever the comparison is made.
- **Three frames, not two.** The implementation kept the z-scrambled control that §2.3 argued for dropping. It is already written and cheap at `n_boot=1000`, so it stays; §2.3's argument stands as a reason not to *lean* on it, not a reason to remove working code.

Four design decisions, each with a reason:

**(a) Continuum-subtract per galaxy *before* stacking.** Each background carries its own continuum level and slope — different sky, different exposure count, different field. Stacking raw sums those offsets and buries a faint line under their spread. Fit and subtract using the *same* sidebands as the science path (1180–1205 / 1225–1260 rest) so that any leak the science continuum fit would also remove is removed here too, and what survives in this stack is exactly what survives in the science path. This is the single most important step and it is easy to omit.

**(b) Merge with an `nfib`-weighted mean, not a second biweight.** This is not a preference, it follows from systematics §5.1 conclusion 3: each per-galaxy background is *already* a biweight over its annulus fibers, so a second robust stage rejects nothing and only loses efficiency (~7–10% variance vs. ~2–3%). Offer `weight="biweight"` as a cross-check, expect it to agree, and say so.

**(c) `rest_delta = 0.2 Å` — match production.** An earlier draft of this spec proposed 0.6 Å to sidestep the oversampling issue in systematics §1.3. **Austin's call, adopted: stay at 0.2.** It works well, it is what the rest of the pipeline uses, and a diagnostic sampled differently from the thing it is diagnosing invites its own class of mismatch. The §1.3 concern is about *quadrature-summed per-pixel errors* being low by ~√3 under oversampling — which is handled here the same way it is handled everywhere else in the repo: **errors come from the bootstrap over galaxies, never from propagated per-pixel errors.** With that rule enforced, oversampling costs nothing but redundant pixels.

**(d) Errors from bootstrap over galaxies, `n_boot ≥ 2000`.** Never from propagated per-pixel errors. Returns the 16/84 band alongside the median, matching the convention used everywhere else in this repo.

Returns the stacked spectrum, its bootstrap band, the per-galaxy contribution matrix (for jackknife and for finding which galaxy drives a feature), and the effective N.

### 2.3 The controls — two, not three

An earlier draft proposed a redshift-scramble null as the primary control. **Austin's call, adopted: drop it.** The reasoning is §0 — the scramble null is machinery for establishing that a marginal bump is real, i.e. for a *blind detection*. That is not this test. The profile already tells us there is Lyα at 660–743 kpc; we are checking an amplitude against a prediction, and a p-value against "no line at all" answers a question nobody is asking. It would also be the most expensive piece here (~500 restacks) for the least return.

Two controls, both cheap, both retained:

1. **Observed-frame stack.** Same data, no per-galaxy shift. A real Lyα feature smears out; a sky line or instrumental artifact sharpens up. This is the discriminator that matters — it separates "real emission" from "artifact at a fixed observed wavelength" — and it is the panel that goes in the paper figure.
2. **Placebo wavelengths.** The same statistic measured at line-free rest wavelengths in the §1.3 window (`validation.placebo_wavelengths` exists). Supplies the empirical distribution of "feature amplitude where nothing should be," which is the honest noise denominator and doubles as a check that the continuum subtraction isn't leaving structure.

```
background_line_amplitude(product, config, profile=None, variant=0) -> dict
```

returning the measured amplitude and width, the observed-frame control, and the placebo distribution. **v4 drops the "predicted amplitude / measured-to-predicted ratio" headline** — that was §0.2's answer to a marginal-null problem the running variant set largely removes, and it makes the diagnostic depend on the fitted profile, which §0.4 says these diagnostics should not do. `profile=` may stay as an optional argument; nothing should require it.

### 2.4 **The main event** — the radial map of the background

**v4: this is the headline of the whole document, not one section among many.** With four annuli spanning 330–1073 kpc (§0), this is the measurement the run exists for.

```
compare_bg_variants(product, config, variants=(0,1,2,3)) -> dict
```

Stack each annulus separately — 80–90″ (fiducial), 60–70″, 40–50″, 110–130″ — and compare. **Two readings come out of the same computation.**

**Reading 1 — the line feature vs. annulus radius.** *How much Lyα is in each annulus, and does it behave like Lyα?*

- **Amplitude falls with annulus radius** → real, physical, clustered Lyα. This is the expected and probably the actual answer, given three of the four annuli sit where the profile already detects signal. It tells us how much line the fiducial annulus is carrying relative to annuli where we know there is signal, which is the calibration the mask width needs.
- **Amplitude flat across annuli** → instrumental or sky, not line; a wider mask will not help and the fix is elsewhere.
- **Amplitude rises with radius** → something is wrong; stop and investigate.

Sort the variants by radius when reporting (40–50, 60–70, 80–90, 110–130), not by variant index — the index order is the config's, which is not monotonic and will produce a nonsense-looking plot if used directly. **This is a real trap given the running config.**

**Reading 2 — the *non-line* background level and shape vs. annulus radius.** Strip the line and ask what the background does between 330 and 1073 kpc. This is B10/B11 measured directly, and it is the input to the architecture question in §0.3:

- **Flat across annuli** → the background is spatially uniform over the range the science bins occupy, a single global annulus is defensible, failure mode B is small, and the whole variable-background idea can be dropped with a one-line Methods justification. **That is a completely good outcome and the cheapest one to write up.**
- **Varies systematically with radius** → the background is *not* uniform, a background measured at 80–90″ is a biased estimate under bins at other radii, and §0.3's variable scheme becomes worth designing.

Report the level per variant with its bootstrap band, and the *paired difference* between variants — the same galaxies contribute to all four, so differences are far better constrained than the individual levels. Do not use independent errors on the differences.

**The specific comparison Austin named: 40–50″ vs. 110–130″.** Widest lever in the run, inner-most against outer-most, and the direct test of "would a larger, further-out background aperture be better." Report it explicitly rather than leaving it to be read off a four-point plot.

**Confounds to check.** Outer annuli have more fibers and more area, so both their noise and their contamination change; report `nfib` per variant and check whether either trend survives normalising by it. Split by field. **Neighbouring galaxies are *not* on this list** — Austin, v4: every annulus and every science bin picks up fibers belonging to other sources, at all radii, so it is a property of the survey rather than a confound specific to the inner variants. Worth noting in Methods, not worth a masking exercise here.

### 2.5 Feeding the §2.5.1 criterion

The pre-registered criterion needs a specific number: the residual at line center as a function of mask half-width.

```
mask_width_scan(product, config, widths=(5,10,15,20,25,30,40), sigmas=(...), variant=0) -> dict
plot_mask_width_scan(result)
```

For each width: apply mask + fill + smooth, then measure the residual at 1215.67 Å in the stack of the *processed* backgrounds. The adopted width is the smallest where that residual is consistent with zero *and* stays consistent at the next width up (a single crossing is noise; a plateau is a result).

This is also where §2.2's joint knob gets exercised: the scan must be **2D over (mask half-width, smoothing σ)**, not 1D, because the smoothing kernel is what re-injects the wings. A 7×5 grid is cheap. Report the residual as a heatmap and pick from the plateau region, not the edge.

**v4 — runtime budget, and it binds here more than anywhere else.** A 7×5 grid re-smooths every galaxy's background 35 times and re-stacks each time. Target **a few minutes maximum** for the whole scan (§0.4), which is the constraint that makes it a knob Austin actually turns rather than a job he schedules. If the naive implementation misses that, the fix is to hoist the per-galaxy resample-to-rest out of the grid loop — it does not depend on either scanned parameter — before reducing the grid.

**And note what the scan is for**, per §0.1: it locates the point where the Lyα is removed but the background is still doing its job. A residual driven to zero by a 60 Å mask and a huge σ is not a success; it is a background that has been smoothed into a featureless line and can no longer remove the systematics it exists to remove. The plateau, not the extreme.

### 2.6 The spread, which is a result in its own right

Austin flagged this and it deserves its own output rather than being a byproduct of the bootstrap. The galaxy-to-galaxy scatter of the background is what sets whether a *single* galaxy's background is trustworthy, and it speaks directly to B10/B11:

```
background_variance_profile(product, config, variant=0) -> dict
```

Report: pixel-wise scatter across galaxies, split by field; the same split by `nfib` quartile (to separate shot noise from real spatial variation); and — if `BG_HALVES` was written — the difference between azimuthal halves of the *same* annulus, which measures spatial variance with the galaxy-to-galaxy term removed. That last one is the cleanest handle on B10/B11 available and is the reason to write `BG_HALVES`.

---

## 3. Q2 — Which filler is best, tested on real data with real truth

> **v4 — DEFERRED. Do not build this yet.** Austin's call (§0.5). None of §3 needs the variant data, so none of it is urgent, and a variable-radius background (§0.3) would change what the filler is being asked to do. Q1 (§2) is what the in-flight run exists to answer; build that, get the architecture answer, then come back here. The design below is kept intact and is believed sound — it is scheduling, not doubt.
>
> **Exception: §3.4 is NOT deferred.** It sits in this section for historical reasons but it is a Q1 deliverable — "how does the binned profile and the centroid change under each background choice" — and it is item 4 in the build order (§5). Read §3.4 as belonging to §2. It should probably be renumbered into §2 the next time this document is touched.
>
> One piece of §3 is worth pulling forward if it turns out to be nearly free: `fill_line_gap`'s alternative methods are pure functions on arrays, and `median_window` in particular is a candidate for the mask-width scan in §2.5 to sweep alongside σ. If it falls out naturally, take it; do not build the placebo/injection harness (§3.3) for it.

### 3.1 The key design move: line-free placebo gaps give real ground truth

The blocking problem in systematics §1.1b was that the stored backgrounds are already interpolated, so testing fillers against them is circular. The fix is not synthetic data — it is to **test the filler where the truth is known because it was never masked**:

> Take a real raw background. Mask a window of the same width at a **line-free** wavelength. Fill it with each candidate scheme. Compare to the pixels that were actually there.

Real spectrum, real noise, real continuum structure, real sky residuals, and exact ground truth. No synthetic component at all. Repeat over many placebo wavelengths × all ~100 galaxies for good statistics.

This measures **noise performance and generic bias** and it is available the moment raw backgrounds exist.

**What it cannot measure** is the specific failure mode §1.1b cares about: anchors sitting on the wings of a broad feature. Line-free regions have no wings by construction. That needs §3.3.

### 3.2 A pluggable filler API

Put the implementations in `smoothing.py` next to `smooth_interp_then_gauss`, because one of them is a candidate to *become* the production Stage-1 filler and it should not have to migrate modules later.

```
fill_line_gap(flux, wave, z, *, method="interp2", mask_halfwidth_A=10.0, **kw)
```

Methods:

| method | description | key kwarg |
|---|---|---|
| `interp2` | current behaviour — `np.interp` off the two edge pixels | — |
| `median_window` | **Austin's proposal.** Median over a window on each side of the mask; linear between the two medians | `anchor_window_A` (~40) |
| `linfit_window` | robust line fit to both anchor windows jointly, evaluated across the gap | `anchor_window_A` |
| `normconv` | normalized convolution `gaussian(f·m)/gaussian(m)` — no two-point dependence at all | `sigma_A` |
| `none` | leave the gap; for A/B against no-fill | — |

#### 3.2.1 What "anchor standoff" means, and why it exists

Austin asked for this to be spelled out. It is one extra parameter and it is the parameter that does the actual work.

**The current scheme (`interp2`).** `np.interp` fills the gap using the single pixel on each side of the mask:

```
                   masked gap (±10 Å)
              ┌──────────────────────────┐
  ~~~~~~~~~~~~█                          █~~~~~~~~~~~~
              ▲                          ▲
           anchor                     anchor          ← 1 pixel each
```

Every filled pixel is a linear blend of exactly those two values. Two problems: they carry full single-pixel noise (no averaging), and if a broad line wing is present, both sit *on* it.

**Austin's proposal (`median_window`).** Replace each single pixel with a median over a window, which fixes the noise:

```
                   masked gap (±10 Å)
              ┌──────────────────────────┐
  ~~~~~~[▓▓▓▓▓]                          [▓▓▓▓▓]~~~~~~
         ▲                                 ▲
    median of ~40 Å                   median of ~40 Å
```

Noise drops by ~√n_pix. **But the bias does not.** The windows are immediately adjacent to the mask, which is exactly where a broad line's wings are strongest — so a wider window averages in *more* wing, not less. It fixes half the problem and can slightly worsen the other half.

**With a standoff.** Insert a dead zone between the mask edge and the start of the anchor window:

```
                   masked gap (±10 Å)
              ┌──────────────────────────┐
  ~~[▓▓▓▓▓]···┤                          ├···[▓▓▓▓▓]~~
      ▲       ↑                          ↑      ▲
  anchor    standoff                 standoff  anchor
            (skipped)                (skipped)
```

**Austin's pushback, and the corrected position.** Austin read the concern as "the window bleeds into the masked region," and answered — correctly — that if the mask region is simply excluded from the median, widening the window is straightforward and needs no extra parameter.

The concern was actually different: **line wings extending *past* the mask edge**, so that the anchor window is contaminated by the line even though it never overlaps the mask. But Austin's position is still largely right, for a reason the first draft glossed over: **the estimator is a median.** A median over a 40 Å window in which the wing contaminates only the inner portion is substantially robust to that contamination — which is much of the reason to prefer `median_window` over a mean in the first place. Standoff only earns its place if the wings extend far enough past the mask to contaminate a *large fraction* of the window, and the magnitude of that effect is **unmeasured**.

So the corrected design:

- **`anchor_standoff_A` defaults to 0**, i.e. exactly Austin's scheme: exclude the mask, widen outward from its edge. Simpler, and the presumptive answer.
- It stays in the API as a **swept parameter, not a recommendation**, because the sweep is being run anyway and costs nothing extra.
- **§3.3 Test B decides it.** The placebo test (§3.1) structurally cannot — line-free regions have no wings — so only the injection test can measure whether standoff > 0 buys anything. If it doesn't, remove the parameter and say so; a simpler filler with a measured justification is a better outcome than a more complex one with an argued justification.

Concretely: `anchor_window_A ≈ 40` (swept), `anchor_standoff_A = 0` default (swept 0–40 in Test B only). Note `normconv` has no standoff by construction — it weights all nearby good pixels by a Gaussian, so no small set dominates, though it does pull in wings at the σ scale.

#### 3.2.2 Parameter trap — confirmed 2026-07-26

`smooth_interp_then_gauss` has signature defaults `sigma_A=25.0, mask_halfwidth_A=20.0`. Production uses neither. Austin confirmed both `config.py` and `run_config.py` default to:

```
bg_smooth_sigma_A     = 30.0     # interp_gauss uniform smoothing width (A)
line_mask_halfwidth_A = 10.0     # interp_gauss half-width (observed A) of the masked gap
smoothing_values      = [200, 20, 300]   # LEGACY 'adaptive' only: [sigma_A, edge_sigma_A, edge_width_A]
```

So a test that calls the smoother directly gets σ=25/mask=20 while production runs σ=30/mask=10 — a silent, plausible-looking mismatch in both parameters at once. **Every diagnostic in this document must take its parameters from a config object and never from function defaults.** Worth an assertion at the top of the test entry points that the config values are what's actually in play.

### 3.3 The two tests

**Test A — placebo-gap recovery (real truth).**

```
filler_placebo_test(bgprod, methods=(...), n_placebo=40, variant=0) -> dict
```

Per galaxy × placebo wavelength × method, report:

- **bias** — mean residual at gap center, and averaged across the gap
- **noise** — RMS residual across the gap, which is where `interp2`'s two-pixel dependence should lose by ~√n
- **tilt** — slope of the residual across the gap, the signature of the random endpoint tilt §1.1b describes
- **post-smoothing residual** — the same metrics *after* the Gaussian smooth, since §1.1b argues the smoothing dilutes the noise penalty ~4×. If it does, that's a real and quotable finding: it means the noise argument for changing fillers is weak and the case rests on bias alone.

**Test B — broad-feature injection (the specific failure mode).**

```
filler_injection_test(bgprod, methods=(...), amp_grid=..., width_grid=...) -> dict
```

Inject a broad line of known amplitude and velocity width at 1215.67 Å into each *line-free-verified* background, then mask + fill + smooth and measure how much of the injected feature survives into the filled gap. This is the direct measurement of "does the filler carry a two-halo wing across the gap."

Sweep width across the range that matters — ±740 km/s (the current mask) out to the ±10.5 cMpc clustering extent — so the output is a curve of *leak fraction vs. feature width* for each filler at each mask width. That curve, crossed with the measured width from §2.3, is what actually sets the production configuration.

**Note the ordering dependency:** Test B's injected width should be informed by Q1's measured width, but Test B's *method ranking* should not be. Run Test B across a wide width grid so the ranking is established independently, then read off the relevant width once Q1 lands.

### 3.4 The metric that actually matters — **rewritten in v4, and this correction applies far beyond §3**

Bias and RMS in the gap are diagnostics, not the deliverable. The deliverable is:

```
profile_impact(product, config, *, variants=..., mask_halfwidths=..., sigmas=...,
               methods=...) -> dict
```

For each background choice: rebuild the science profile via `reconstruct_product(...)` → `build_stacks(...)`, and report **how the binned profile changes**.

**v3 said "report Δh1, Δcrossover." That was wrong and is removed.** Per §0.4:

- **The comparison is at the level of bin values, not fitted parameters.** Overplot the binned surface-brightness profiles under each background choice, with their bootstrap errors, and report the per-bin differences. That is the thing the paper's primary science rests on and it is directly readable.
- **The centroid profile gets the same treatment.** The spectroscopic centroid vs. radius is a primary product of this paper, and background subtraction can move it — arguably more subtly than it moves the surface brightness, since a background with residual Lyα in it is being subtracted from a line whose *position* is the measurement. **Any background choice must be checked against the centroid profile, not only the flux profile.** v1–v3 never mentioned this and it may be the more important of the two.
- **Do not fit on this run's grid.** The coarse bins `[0, 20, 50, 100, 200, 500, 1000, 2000]` put four bins inside 200 kpc, which is where `h1` is determined. A fit there is not well constrained and Δh1 from it would be noise dressed as a result. If a fitted number is ever wanted for this question, it waits for the fine grid.

**The good outcome, and the likely one:** every background choice gives the same binned profile and the same centroid within errors. That is a *complete answer* and exactly the Methods sentence §0.4 describes — *"the profile and centroid are insensitive to the background annulus over 330–1073 kpc and to mask half-widths of 10–30 Å."* The toolkit should make that sentence easy to earn, rather than being built on the assumption that a change is needed.

---

## 4. Plots

**v4: `plot_*` functions go in `background.py`, beside their computation.** v3 pointed at `validation.py` by analogy with `plot_scramble_load`/`plot_window_sweep`; the convention is *beside the computation*, and the computation is in `background.py` — where `plot_background_stack` already lives. **Austin's call: new code goes in `background.py`.** See §7 on the consolidation this implies.

**Per §0.4, the plots carry more of the argument here than the numbers do.** Several of these questions are settled by looking, and the plot is then the artifact that justifies the Methods sentence.

| plot | shows | status |
|---|---|---|
| `plot_background_gallery` | individual raw backgrounds, mask + sidebands shaded | to build |
| `plot_background_stack` | rest-frame stack + bootstrap band, controls overlaid — **the paper figure** | **exists** |
| `plot_bg_variant_comparison` | **the main figure of this run** — four annuli stacked and overlaid, plus amplitude and level vs. annulus radius, sorted by radius not by index | to build |
| `plot_profile_impact` | binned flux profile *and* centroid profile under each background choice, overplotted — the §3.4 deliverable | to build |
| `plot_mask_width_scan` | 2D heatmap of residual vs. (mask half-width, smoothing σ), plateau marked | to build |
| `plot_background_variance` | scatter vs. wavelength, split by field and `nfib` quartile | to build |
| ~~`plot_filler_comparison`~~ / ~~`plot_filler_demo`~~ | placebo-test metrics; interactive filler demo | **deferred with §3** |

### 4.1 On "playing with the values"

Austin wants to tune parameters and see the effect. Per the standing preference for notebook cells over throwaway scripts, the design is: **every function takes explicit kwargs and returns a plain dict; no hidden state; no interactive widgets required.** Tuning is then a two-line notebook cell looping over the parameter of interest.

**Runtime is part of the interface.** §0.4's budget — a few minutes maximum per call — is what makes the loop real. A function that takes half an hour is a function whose parameters never get tuned.

If an `ipywidgets` slider wrapper is wanted later it can sit on top of these functions without changing them. Don't build it first.

---

## 5. Build order — **rewritten for v4**

Ordered so each step is testable when it lands. **Everything below is Q1; §3's Q2 items are deferred (§3 preamble) and have been removed from this list.**

**Already shipped 2026-07-26** (`background.py`, registered in `guide.py`): `reconstruct_background`, `reconstruct_spec`, `reconstruct_product`, `verify_reconstruction`, `stack_backgrounds`, `measure_line_residual`, `adopt_mask_halfwidth`, `background_half_difference`, `describe_background_test`, `plot_background_stack`.

**The moment the run lands — in this order:**

1. **`verify_reconstruction`.** Non-negotiable first call. If `SPEC_NOBG − smooth(mask(BG_VARIANTS[0])) ≠ SPEC`, nothing else in the module means anything. Already written; just run it.
2. **`plot_background_gallery`** — look at the data before computing on it. Set the galaxy-rejection criterion here, write it down, apply it uniformly.
3. **`compare_bg_variants` + `plot_bg_variant_comparison`** — the main event (§2.4), both readings. This is the measurement the run exists for and it should come before any tuning, because what it shows determines whether tuning is even the right response.
4. **`profile_impact` + `plot_profile_impact`** (§3.4) — binned flux profile *and* centroid profile under each variant and each mask width. This is what reaches the paper.
5. **`mask_width_scan` (2D) + `plot_mask_width_scan`** → adopt a width per the §2.5.1 criterion, from the plateau.
6. **`background_variance_profile`** → B10/B11, using `BG_HALVES`.

**Deferred:** everything in §3 (`fill_line_gap`, `filler_placebo_test`, `filler_injection_test`), and `estimate_stack_sensitivity` (§1.2, optional).

**Housekeeping, not optional:** register every new public function in `guide.py` and run `check_guide()` clean. `fitting.py`/`uv_profile.py` sat unregistered with zero audit coverage until 2026-07-21; don't repeat it.

### 5.1 [IMPLEMENTED 2026-07-26] The run cell

All six steps shipped in `background.py`, registered in `guide.py`, `check_guide()` clean and `test_framework.py` green. Two real bugs were caught by a synthetic smoke test before shipping — an operator-precedence error that made the nfib "quartiles" nested and overlapping, and a √2-instead-of-2 normalisation that inflated the reported spatial fraction. Both are documented at the code.

Drop this in a notebook once the run lands. Each block is independent; run them in order the first time.

```python
from utils_lya_halo import read_galaxy_fits
from utils_lya_halo import background as B

p = read_galaxy_fits(path)          # path = run_extract(cfg_bg)

# 1. NON-NEGOTIABLE FIRST. If this fails, stop -- nothing below means anything.
B.verify_reconstruction(p, cfg_bg)

# 2. Look at the data. Set the galaxy-rejection criterion HERE, write it down.
B.bg_variant_radii(p, cfg_bg, verbose=True)      # geometry + kpc + bin overlap
B.plot_background_gallery(p, cfg_bg, n=12)

# 3. THE MAIN EVENT -- the radial map. Both readings, radius-sorted.
res = B.compare_bg_variants(p, cfg_bg, n_boot=500)
B.plot_bg_variant_comparison(res)

# 4. What reaches the paper: flux AND centroid per bin, per background choice.
imp = B.profile_impact(p, cfg_bg, n_boot=200)
B.plot_profile_impact(imp)

# 5. Mask width x smoothing sigma. Adopt from the plateau, not the edge.
scan = B.mask_width_scan(p, cfg_bg, sigmas=(10, 20, 30, 40, 50))
B.plot_mask_width_scan(scan)

# 6. B10/B11: how much of the scatter is spatial rather than galaxy-to-galaxy.
var = B.background_variance_profile(p, cfg_bg)
B.plot_background_variance(var)
```

Tuning is a loop over any of these — every function takes explicit kwargs and returns a plain dict, no hidden state. To sweep a mask width against the *profile* rather than against the background residual:

```python
imp = B.profile_impact(p, cfg_bg, n_boot=200, choices=[
    {"label": "mask 10 A (production)", "variant": 0, "mask_halfwidth_A": 10.},
    {"label": "mask 20 A",              "variant": 0, "mask_halfwidth_A": 20.},
    {"label": "mask 30 A",              "variant": 0, "mask_halfwidth_A": 30.},
])
```

**One ordering discipline, from §2.5.1:** run step 5 and adopt a mask width *before* looking at step 4's profiles under different widths. Selecting the width by which value makes the profile look best is the one thing that would make this circular.

---

## 6. Open design questions

**Closed in v4:**

- ~~Run-table ordering / is the 100-galaxy slice biased?~~ **Austin, v4: checked, the ordering is clean.** No action.
- ~~Neighbouring galaxies contaminating the inner annuli.~~ **Not a concern (§2.4).** Every annulus and every science bin contains fibers belonging to other sources, at every radius. It is a survey property, worth a Methods sentence at most.
- ~~`BackgroundProduct` design.~~ Already on `GalaxyProduct` (§1.1).
- ~~Is `estimate_stack_sensitivity` a gate?~~ No (§1.2).
- ~~Where does new code live?~~ `background.py` (§4, §7).
- ~~Should the profile-level check report Δh1?~~ No (§0.4, §3.4).

**Still open:**

- **Galaxy rejection criterion for the stack.** Will be set from §2.1's gallery. Needs writing down and applying uniformly once chosen.
- **Is 100 galaxies enough?** Weakened but not closed by §0.2 — likely yes for 40–50″ and 60–70″, marginal for 110–130″. If the outer variant is the one that matters and it is too noisy, extending the run is cheap and is the right move over over-reading a null.
- **Should the production background annulus move — and in which direction?** Genuinely two-sided (§0.1) and must not be prejudged. Outward reduces Lyα contamination (mode A); inward keeps the background spatially closer to the bins (mode B). 80–90″ may already be a reasonable compromise. Decide from §2.4, and specifically from the 40–50″ vs. 110–130″ comparison.
- **Is a single global annulus the right architecture at all?** (§0.3.) Deferred deliberately. If §2.4's Reading 2 shows the background is flat with radius, this closes with a one-line justification. If it varies, §0.3's two constraints apply and it gets its own spec. **Do not scope until Reading 2 exists.**
- **What sets the floor on failure mode B?** Even a line-free annulus at 110–130″ is ~1000 kpc from the galaxy, subtracted from a bin at 10 kpc. If sky/scattered light varies on IFU scales (systematics §6.4), there is an irreducible mismatch. `BG_HALVES` bounds it: the azimuthal split measures spatial variance at fixed radius.
- **Do the two fields get stacked together or separately?** EGS is deeper, COSMOS is larger-area. For the *amplitude* they should agree and combining is right; for the *noise level* they must be separate. Default to separate plus combined, check consistency before quoting the combined number.
- **Continuum treatment on the background.** The implementation already departed from the science sidebands deliberately (§2.2, `BG_CONT_BOUNDS`). Still worth checking the fit residuals on real backgrounds — the background's continuum is sky and scattered light, not starlight, and may have structure neither sideband choice anticipates.
- **Does the mask-width scan interact with the variant choice?** The scan is specified at variant 0. If the annuli carry very different amounts of Lyα (which §2.4 may well show), the adopted width could differ between them. Run the scan at the adopted annulus, not blindly at variant 0.

---

## 7. Module placement and the consolidation this implies

**Austin, v4: new code goes in `background.py`** — same realm, and `plot_background_stack` and the reconstruction helpers are already there.

That raises a broader point worth recording rather than acting on immediately. **Background logic is currently spread across four modules:**

| where | what |
|---|---|
| `background.py` | Stage-2 offline testing, reconstruction, stacks, mask criterion, halves |
| `smoothing.py` | `smooth_interp_then_gauss`, `smooth_spectrum_nan_safe_adaptive` — the actual smoothers |
| `extract.py` | annulus fiber selection, the per-galaxy combine, `_maybe_smooth` |
| `core.py` | `inspect_core_background` |

**Candidate change: consolidate background computation into `background.py`.** Not scoped here and not urgent — but it is the natural home, and the split is already causing the kind of "where does this live" friction that the `environment.py` → `multicat.py` merge (2026-07-21) fixed for the overdensity helpers. **Logged in the workspace `TODO.md` under the broader codebase-organization item.** Whenever it happens: keep `guide.py` registration current and `check_guide()` clean through the move.

## See also

- `specs/systematics-and-extraction.md` — §1.1 (the leak), §1.1b (the anchors), §2.2 (mask × σ), §2.5 (the extraction that produces this data), §5.1 (why n-weighted mean merging)
- `validation.py` — `redshift_scramble_null`, `zswap_spectra`, `placebo_wavelengths`, `injection_recovery` — all patterns to reuse rather than reinvent
- `smoothing.py` — `smooth_interp_then_gauss`, the current filler
