# plot_flux_profile — two-component exponential fit

Spec for a new fit overlay on `01-lya_halos/code/utils_lya_halo/analysis.py`'s
`plot_flux_profile`. Written *forward* (spec first, code second) — this is a
work-in-progress design, not a record of something already built. Several
choices below are marked **[DECIDE]** where the spec deliberately leaves a
fill-in-the-blank rather than presuming Austin's answer.

**Status: Phase 2, deferred. Not started.** Phase 1 — proving out a
PSF-aware two-component fit on synthetic data in a standalone script — comes
first; see `specs/psf-exponential-fit-recovery.md`. This document describes
what eventually lands in `analysis.py` once Phase 1 settles the fit method,
including whether "ignore the PSF, drop the first bin" (the naive approach
this doc originally specced) is good enough or whether the PSF-forward-model
fit from Phase 1 needs to replace it. Kept as a record of the naive-fit
design in case that turns out to be the one that ships, but expect this file
to be revised once Phase 1 has results.

## Goal

`plot_flux_profile` currently just plots integrated Lyα flux vs. radius
(`boot["total_flux_fid"]` ± bootstrap 16/84) on a log-y axis. Visually the
profile looks two-component — a steep inner slope and a shallower outer
slope, with the break falling roughly at R_vir, which is a physically
interesting result on its own. The goal here is to make that visible break
*quantitative*: fit exponential slope(s) to the profile and report the scale
length(s), so "it looks two-component" becomes a number (or two) that can go
in the paper.

Scope is deliberately narrow: **one new plotting capability inside
`analysis.py`**, activated by new keyword arguments on the existing
`plot_flux_profile` function. No new file, no new module. The fit machinery
itself lives as a private helper section within `analysis.py` (a "submodule"
in the sense of a clearly-delimited block of helper functions, not a
separate `.py` file) — mirroring how `_setup_radius_axis` / `_safe_yerr` /
etc. already live in `plotting.py` as helpers behind the public plot
functions.

**Explicitly out of scope for this spec:** the fake-data / PSF-convolution
testing harness Austin wants (synthetic two-component profiles with known
scale lengths, PSF-convolved, refit, iterate on recovery). That's real and
wanted, but it's a *separate* piece of work — a standalone test script plus
a notebook, not a change to `analysis.py`. Flagged here as follow-on work so
the fit function's interface (below) is designed to be testable against
synthetic input from day one, even though the harness itself isn't being
spec'd yet.

## Data

Same inputs `plot_flux_profile` already takes — no new data source:

- `boot["total_flux_fid"]`, `total_flux_lo`, `total_flux_hi` — the y values
  and bootstrap 16/84 band already plotted today.
- `r_mid` — bin midpoints in native units, already computed today via
  `plotting._setup_radius_axis` → `_radius_points(radial_bins)`. Native
  units are `R/R_vir` (`bin_mode="virial"`, the default), kpc, or arcsec —
  see `plotting._resolve_bin_mode`.
- `vr` (`VR_biweight_v`) — needed if the break radius defaults to R_vir
  (see Method).

No PSF measurement is consumed here. `starpsf.py` already has the empirical
stellar-PSF curve-of-growth machinery (`radial_continuum_profile`,
`curve_of_growth`, `psf_profiles_for_lines`) that the *future* fake-data
harness should reuse rather than re-derive a PSF model from scratch — noted
here so that follow-on work doesn't reinvent it, but nothing in this spec
calls into `starpsf.py`.

## Method

### Excluding the PSF-correlated inner point

The first (innermost) radial bin is systematically correlated with the
second because of PSF spread — Austin's read is that it shouldn't anchor an
exponential fit. Implemented as a parameter, not a hardcoded skip, so it's
visible and adjustable rather than a silent magic number:

```python
fit_skip_inner: int = 1   # number of innermost bins dropped from the FIT only
```

`fit_skip_inner` bins are excluded from the fit's x/y/weight arrays but
**stay on the plot** as normal data points (with their errorbars) — only the
fit ignores them. Any additional non-finite or non-positive bins (can't take
log of a non-positive flux) are also dropped from the fit, same as the
existing `logy` masking (`pos = y[y > 0]`) already does for the axis limits.

### Fit model — **[DECIDE]** which two-component form

Two candidate forms for "two-component exponential." Both are standard in
radial-profile literature; the difference is whether the break radius is an
explicit fit knob or an emergent property of two summed terms. Recommend
**Option A** as the default given Austin's framing ("splits roughly around
the virial radius" — a specific radius, not just an emergent knee), but this
is the one real fill-in-the-blank in this spec and should be confirmed
before writing code.

**Option A — piecewise/broken exponential (recommended default).**
Two independent single-exponential segments, `I(r) = A * exp(-r / h)`,
joined at a break radius `r_break`:

- inner segment: bins with `r < r_break` (after `fit_skip_inner` exclusion)
- outer segment: bins with `r >= r_break`
- each segment fit independently in log space: `ln(I) = ln(A) - r/h`, a
  weighted linear regression (or `curve_fit` on the log-linear form for
  consistency with the project's existing `curve_fit`-based fits — see
  below) → slope directly gives `-1/h`.
- `r_break` default: `vr` (R_vir in native units, converted via
  `plotting._vr_to_native` when `bin_mode != "virial"`) — i.e. literally the
  break Austin already sees by eye. `fit_break_r` param overrides it.
- `fit_free_break: bool = False` — if `True`, `r_break` itself becomes a
  free parameter (grid search or `curve_fit` over a small window around the
  default) instead of being pinned to R_vir. Off by default: pinning to
  R_vir is the more falsifiable claim ("the break IS at R_vir") and is
  cheaper/more stable than a 5-parameter simultaneous fit.
- continuity at the break: **[DECIDE]** allow a jump (two fully independent
  fits, simplest) vs. force `A_in * exp(-r_break/h_in) == A_out *
  exp(-r_break/h_out)` (physically tidier, one fewer free parameter, more
  fit-code complexity). Default to the simple independent-fits version
  first; a visible jump at the break is itself diagnostic (if it's large,
  that's worth knowing).

**Option B — sum of two exponentials** (alternative, not the default):
`I(r) = A1*exp(-r/h1) + A2*exp(-r/h2)`, fit as one nonlinear `curve_fit` over
all bins at once. No explicit break radius — a "knee" emerges wherever the
steep term dies below the shallow term. Standard for disk photometry
(thin+thick disk decomposition). Downside for this use case: doesn't let
Austin assert "the break is at R_vir" directly, and with only ~8-10 radial
bins a 4-parameter nonlinear fit is more seed-sensitive (see the
multi-seed-then-best-chi2 pattern `speclsf.py`'s `_fit_double` already uses
for exactly this stability problem).

`n_components` selects between the two component-counts, not between A/B —
whichever of A/B is chosen becomes what `n_components=2` means:

```python
fit_exponential: bool = False   # master toggle, off by default
n_components: int = 2           # 1 -> single global exponential (no break)
                                 # 2 -> two-component (default; see Option A/B above)
```

`n_components=1` is the plain single-exponential fit over all (non-excluded)
bins — useful as the "does two-component actually fit better" baseline,
since it's the same underlying single-segment fit code Option A already
needs for each half.

### Fitting mechanics (house convention)

Follow the pattern already used for every other `curve_fit` call in this
codebase (`measure.py`'s Gaussian centroid fit, `core.py`'s line fit,
`speclsf.py`'s single/double LSF fit): wrap in `try/except`, return a result
dict with `success`/`reason` rather than raising, weight by the bootstrap
error (`sigma=`, `absolute_sigma=True`), and — for the two-component case —
try multiple seeds and keep the lowest-chi2 result, the same stabilization
`speclsf.py`'s `_fit_double` uses for its own single-vs-double model
comparison.

- weights: reuse the same `yerr` already computed via `plotting._safe_yerr`
  (average of the asymmetric lo/hi lengths, or the fit works in log-space so
  propagate to `sigma_ln = yerr / y`).
- fit in **log space** (`ln(I)` vs. `r`), not linear space — matches the
  log-y display and is what makes "slope = -1/h" a direct linear-regression
  quantity for Option A.
- `p0` seed: rough scale length from the two endpoints of each segment
  (`h0 = (r[-1]-r[0]) / ln(y[0]/y[-1])`), same spirit as the amplitude/sigma
  seeding already done in `measure.py`'s `_gauss` fit.

### Private helper section in `analysis.py`

New block, placed just above `plot_flux_profile` (or in a clearly marked
`# ---- exponential-fit helpers (flux profile) ----` section), all
underscore-prefixed (internal, not part of the public API surface unless
Austin later wants to call the fit standalone):

```python
def _exp_model(r, A, h):
    """I(r) = A * exp(-r / h)."""

def _fit_single_exponential(r, y, yerr, *, p0=None) -> dict:
    """One curve_fit call in log space. Returns {"success", "reason",
    "A", "h", "h_err", "chi2", "ndof", ...} -- same shape convention as
    measure.py/speclsf.py's fit results."""

def _fit_broken_exponential(r, y, yerr, *, r_break, free_break=False) -> dict:
    """Option A: two _fit_single_exponential calls, one per side of
    r_break. Returns {"success", "inner": {...}, "outer": {...},
    "r_break", ...}."""

def _fit_flux_profile_exponential(r, y, yerr, *, n_components, fit_skip_inner,
                                   r_break, free_break) -> dict:
    """Dispatches to _fit_single_exponential (n_components=1) or
    _fit_broken_exponential (n_components=2) after applying fit_skip_inner
    and the positive-flux mask. This is the function plot_flux_profile
    calls; also the entry point the future fake-data test harness calls
    directly (bypassing the plot) to check recovered h against injected
    truth."""
```

### `plot_flux_profile` signature additions

```python
def plot_flux_profile(
    boot, stacks=None, r_edges=None, bin_mode=None, VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), logy=True, ylims=None, xlims=None,
    show_vr=True, VR_biweight_error=None, figsize=(7.2, 4.7),
    title="Integrated Lyα flux vs. radius", save_fig=False, save_name=None,
    # -- new --
    fit_exponential: bool = False,
    n_components: int = 2,
    fit_skip_inner: int = 1,
    fit_break_r: float | None = None,
    fit_free_break: bool = False,
    return_fit: bool = False,
):
```

`fit_exponential=False` is fully backward compatible — every existing call
site (`analysis.plot_flux_profile(boot, stacks)` per the module docstring,
`run_all_plots`, etc.) is untouched. `return_fit=False` by default keeps the
existing `return fig, ax` signature; only when explicitly requested does the
function return `fig, ax, fit` (needed by the future test harness to check
recovered scale lengths against injected truth — a plot alone can't be
asserted against in a test).

### Display

- Overlay the fit curve(s) as a dashed line on top of the existing errorbar
  points (`ls="--"`, distinct color from the data, e.g. `tab:orange`).
  `n_components=2` draws two dashed segments meeting (or not, per the
  continuity **[DECIDE]** above) at `r_break`; a thin vertical guide line at
  `r_break` if it doesn't already coincide with the R_vir line already drawn
  by `show_vr`.
- The `fit_skip_inner`-excluded point(s) stay plotted normally but get a
  visual flag (same hollow-marker convention `plot_flux_profile` already
  uses for `unstable` fiducial points via `_safe_yerr`) so it's clear by eye
  that they were excluded from the fit, not just invisible.
- Scale length annotation: a text box (`ax.text` with a light background, or
  a legend entry) reporting `h_in`, `h_out` (or just `h` for
  `n_components=1`) with 1-sigma uncertainty, in the plot's native radius
  unit (kpc or R/Rvir depending on `bin_mode` — reuse
  `plotting._NATIVE_LABEL` for the unit string). e.g.
  `h_in = 0.18 ± 0.03 R_vir`, `h_out = 0.71 ± 0.15 R_vir`.
- If the fit fails (`success: False` from the helper — too few finite bins
  after exclusion, `curve_fit` non-convergence, etc.), plot proceeds exactly
  as it does today (no fit overlay) and prints a warning rather than
  raising, matching how the rest of `analysis.py` degrades (e.g. the
  `unstable`-band handling, the `unit_info` fallback).

## Success criteria

- `analysis.plot_flux_profile(boot, stacks, fit_exponential=True)` runs on
  real bootstrap output, draws a two-segment dashed exponential fit that
  visually tracks the break Austin already sees by eye, and prints/annotates
  `h_in`/`h_out` on the figure.
- `fit_exponential=False` (the default) reproduces today's figure
  byte-for-byte in behavior — no regression on existing notebook cells.
- `n_components=1` recovers a single global `h` for comparison, so "does
  two-component actually fit meaningfully better" is answerable (even if
  formal model comparison, e.g. an AIC/BIC readout, isn't built yet — flag
  as a nice-to-have, not a blocker for v1).
- `_fit_flux_profile_exponential` is callable standalone (not just through
  the plot function) with plain `r`, `y`, `yerr` arrays, so the follow-on
  fake-data/PSF-convolution test harness (separate spec, separate file, per
  Austin's ask) can feed it synthetic profiles with known injected `h_in`/
  `h_out` and check recovery — this spec doesn't build that harness, but the
  helper's interface is shaped so that harness doesn't need to reach into
  `plot_flux_profile` internals to get a testable fit result.

## Open questions before implementation (recap of **[DECIDE]** points above)

1. Option A (piecewise, break pinned to R_vir by default) vs. Option B (sum
   of two exponentials, no explicit break) — recommend A.
2. If A: force continuity at the break, or allow the two segments to not
   meet (a visible jump is itself informative)?
3. Any minimum bin count per segment below which the fit should refuse to
   run (e.g. outer segment needs >=3 finite bins to fit a 2-parameter
   model meaningfully)?
