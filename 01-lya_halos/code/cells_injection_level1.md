# Notebook cells — injection ladder levels 0 and 1

**Consolidated 2026-08-16: thirteen cells → four.** The old version was thirteen
cells because it was *diagnosing* the generators. They now pass, so the
diagnosis collapses into two function calls (`check_level0`, `check_level1`) and
the investigations that produced one-time answers live in `CHANGELOG.md` rather
than in a notebook you re-run.

Dropped, with their answers recorded in the changelog: the deliberate-break test
(now folded into `check_level0`), flux conservation (now inside `check_level1`),
the mean/median/biweight ladder, and the `r_trunc` truncation table.

**Scope.** This is the *"is the template what I think it is"* half of the problem
— a question about our code — run on **real fiber geometry** but with **no cube,
no noise, and no extraction**. Spec §7.2 splits it deliberately from *"does the
pipeline recover it"*, because they fail differently and debug differently.
Nothing here costs extraction time.

**Organizing rule for this notebook.** Anything shared lives in the setup cells
at the top. Every validation cell is a **dead end**: it prints a verdict and
produces nothing the sections below it need. That is what lets you skip cell 4
once it has passed without breaking anything downstream.

---

### Cell 1 — setup  *(seconds, always run)*

```python
# Production binning, SET 2026-08-16: 11 bins ending at 2000 kpc. The 3000 edge
# was dropped for runtime and because there is little S/N out there. bin_mode is
# "kpc" -- the config.py dataclass default of "virial" is stale and is NOT what
# production runs.
#
# TWO KNOCK-ONS OF THE 2000 EDGE, both handled, both worth knowing:
#   * SourceModel's r_trunc default tracks it (3000 -> 2000). It has to: r_trunc
#     is the radius the injected luminosity L is defined INSIDE, so a mismatch
#     means L is "flux inside a radius you never measure". The core fraction
#     moves 1.37% -> 2.32% of L, i.e. 1.7x more injected signal where the
#     measurement is best constrained.
#   * select_nearby_fibers now cuts at 244" instead of 366" -- ~55% fewer fibers
#     per source per exposure, so templates are faster and smaller.
#
# NOTE run_extract_batch.py's RUN block still carries the OLD 13-edge list. Not
# in use yet, but it must be updated before it is, or Stage 1 will produce
# products the injection truth does not match.
#
# Nothing in this notebook reads a truth value off a config. Every comparison is
# against predict_binned_observables (spec §3.5).

import numpy as np
import matplotlib.pyplot as plt

from utils_lya_halo import injection as inj
from utils_lya_halo.config import PipelineConfig
from utils_lya_halo import extract

BINS   = [0, 10, 16, 24, 36, 60, 100, 180, 350, 600, 1000, 2000]   # kpc, 11 bins
Z_TEST = 2.4            # replace with a bootstrapped z from the real run table later
FIELD  = "COSMOS"       # ONE field. Fields are extracted separately and stay
                        # separate through the diagnostics (standing practice);
                        # 'BOTH' is a Stage-2 setting and does not belong here.

# [FILL] h1/sigma are placeholders, gated on the post-PSF-fix + post-SMHM-switch
# refit (spec §3.2). alpha stays 0 -- skew is out of scope, the pipeline reports
# only the centroid.
P = inj.family_P()
H = inj.family_H(h1=17.0, A2_over_A1=0.02, r_c=400.0, gamma=0.8,
                 sigma_kms=250.0, alpha=0.0,
                 dv_in=0.0, dv_out=100.0, r_break=200.0)

print(f"bins: {BINS}  ({len(BINS)-1} bins)")
print(f"LSF: {inj.LSF_SIGMA_KMS:.0f} km/s sigma  (from the measured 1.9 A rest-frame FWHM)")
print(P)
print(H)
```

---

### Cell 2 — ladder level 0  *(~0.3 s)*

```python
# Level 0. No cube, no geometry -- just: do the generators mean what they say?
# Five checks:
#   1. int I(r) 2*pi*r dr = 1     the surface-brightness convention (the missing
#                                 2*pi*r was half the 2026-07-22 area bug)
#   2. the line integrates to 1, its 50%-flux centroid is the requested dv, AND
#      its width is the INSTRUMENT-BLURRED one -- sqrt(250^2 + 200^2) = 320.2,
#      not 250. That third part pins the LSF convention, which until 08-16 was
#      set in two places that disagreed.
#   3. THE ONE THAT MATTERS: the production fiber response (moffat_psf_
#      integration, Cartesian pixel grid) vs an independent one (closed-form
#      Moffat, polar quadrature). The only place two independent implementations
#      of the same quantity are compared, and per §3.5 that comparison is the
#      only mechanism that would have caught the 07-22 bug.
#   4. V -> 1 as seeing -> 0 and r -> 0, and V monotone decreasing.
#   5. the LSF closed form vs a brute-force convolution.
#
# The alpha=2 model is NOT something we inject -- both families run alpha=0. It
# is here because at alpha=0 the skew half of the LSF transformation is
# trivially zero, so check 5 would not notice if it were dropped. One skewed
# case is what keeps that check honest.
#
# EXPECT: all PASS; response ratio within a few parts in 1e3; observed width
# 320.2 km/s; LSF closed form vs brute force ~1e-15.

for m in (P, H, inj.family_H(h1=17.0, alpha=2.0)):
    inj.check_level0(m)
    print()
```

---

### Cell 3 — load the field  *(always run; the mask scan is the slow part)*

```python
# Real geometry from here on: real fiber positions, real per-exposure seeing,
# real DAR. Everything the source sits in is measured data; the only thing
# asserted is the source itself.

cfg = PipelineConfig(field=FIELD, bin_mode="kpc", bins=BINS)
vdfi = extract.load_field(cfg)

wave = vdfi["wave"]
nexp, nfib = vdfi["RA"].shape
print(f"{nexp} exposures x {nfib} fibers, {wave.size} wavelength pixels")
print(f"wavelength grid: {wave[0]:.1f}-{wave[-1]:.1f} A, "
      f"{np.diff(wave).mean():.3f} A/pix")
print(f"seeing: median {np.nanmedian(vdfi['seeing']):.2f}\", "
      f"sd {np.nanstd(vdfi['seeing']):.2f}\"")
```

---

### Cell 4 — ladder level 1  *(minutes — the expensive cell)*

```python
# THE CELL THIS NOTEBOOK EXISTS FOR. Builds the template on real fiber geometry,
# per exposure, through the real optics (_exposure_seeing for the per-exposure
# Moffat width, _offsets_with_dar for the wavelength-dependent sky offsets), then
# compares the binned result against the INDEPENDENT truth generator.
#
# Injecting into the coadd, or at a fixed wavelength-independent position, would
# skip the optics chain entirely and is the most likely way to build a test that
# passes vacuously (spec §3.1). So: per exposure, with DAR.
#
# Two implementations that share only the DEFINITION of the source and no
# machinery for turning it into binned observables. If they agree, the injector
# does what it claims. If they disagree, ONE of them is wrong and which quantity
# disagrees tells you which.
#
# SEEING_MODE="field" is the RIGHT choice here, not a shortcut:
# predict_binned_observables accepts only a SCALAR seeing, so a per-exposure
# injection is already being compared against a single-seeing truth, and
# matching the injector to it REMOVES a known discrepancy. Measured cost at
# seeing median 1.97"/sd 0.37": dv differs by <=0.06 km/s, profile +3.1% in the
# innermost bin and <1% outside. Switch to "exposure" for the real cube run --
# affordable there because the PSF tables cache per exposure across all sources.
#
# Read the velocity column first: it is the primary result (spec §0.3). Note the
# [180, 350] bin -- 200 kpc falls INSIDE it, so its truth is a flux-weighted
# BLEND of 0 and +100 (91.00 km/s), not either pure value. That is criterion 1d,
# and it is computed rather than assigned. Comparing that bin against a flat 100
# would report ~9 km/s of "bias" that does not exist.

ok, lines, data = inj.check_level1(H, vdfi, z=Z_TEST, bins_kpc=BINS,
                                   seeing_mode="field", plot=True)
```

---

## What "pass" looks like

| Cell | Pass |
|---|---|
| 2 | all PASS for all three models; observed width **320.2 km/s** (not 250); response ratio a few parts in 1e3; LSF closed form ~1e-15 |
| 4 | `dv rec` matches `dv TRUTH` to well inside one pixel; core bins ≈ 0, outer ≈ +100, `[180,350]` lands on **91.00**, *not* on 100; `SB rat` column flat at 1.000; both curves on top of each other in the figure |

**Reading a level-1 failure.** The per-bin table is built to localise it:

- **`SB rat` off in one bin only, and that bin has few fibers** — fiber sampling,
  not a template error. The recovered value is a mean over *fibers*; the truth is
  a mean over *area*; they agree only where fibers sample the annulus. Measured on
  a synthetic uniform-grid field: bins with ≥205 fibers agreed to **0.15%**, the
  63-fiber bin was off by **6.1%**. Raise `min_nfib_shape` if this is the only
  failure — but check the fiber count first rather than assuming.
- **`SB rat` sloping smoothly across all bins** — a real shape error. Suspect the
  profile evaluation or the bin assignment.
- **`dv` off by a constant at every radius** — the centroid convention or the
  velocity zero point. This is the disguise the 2026-08-07 half-pixel bug wore:
  a clean offset at every radius looks like a small systemic shift, not a bug.
- **`dv` off only in `[180, 350]`** — the truth generator's flux weighting.
- **Profile wrong by a smooth multiplicative factor but velocity fine** — aperture
  normalization. `check_level0`'s check 3 would have caught that already, so a
  failure here with cell 2 passing points at the geometry, not the optics.

**Only after both cells pass** is it worth building `inject_sources` /
`InjectedSpectra` and spending extraction time. Levels 0–1 are pure code; the
extraction budget buys level 7 only.
