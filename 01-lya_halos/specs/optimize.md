# optimize.py

Spec for `01-lya_halos/code/utils_lya_halo/optimize.py`, written against the module as it
exists today plus the reworks below. Written in reverse (code first, spec second) at Austin's
request, then forward from here — this file is the reference for what the module should be,
not just a record of what it was.

**Status: implemented (2026-07-10).** All three planned changes below are in the code:
`run_background_sweep` deleted, `continuum_noise_metric` has `verbose`/progress bar and the
`compute_line_snr` diagnostic (routed through a new `measure.integrated_line_flux_per_bin`),
and `plot_line_snr_summary` exists. Kept as the reference doc per the spec README convention —
update this file if the approach changes again, don't let it drift from the code.

## Goal

**`optimize.py` is a hub, not a single-purpose module.** It's the place pipeline variants get
scored/ranked under whatever objective is currently in question — background noise today,
possibly other sub-optimizations later (a different masking axis, a different combine choice,
whatever comes up). Everything below — the continuum-noise metric, its two dict keys, the
line-S/N diagnostic — is ONE sub-optimization living inside that hub: minimizing background/
continuum noise. Written so a second, unrelated sub-optimization could be added alongside it
later without restructuring this one; not designing that second one now, just not naming
functions or writing this doc as if `continuum_noise_metric` were the only thing this file will
ever hold.

Within that sub-optimization: score candidate pipeline variants (background/masking/extraction
choices, or combine-level choices like stack method) by how NOISY the continuum is in a
line-free sideband, as a proxy for how trustworthy the background subtraction feeding the real
Lya measurement is likely to be. Deliberately does not touch the Lya line itself when ranking —
see "Why sideband, not line" below; that non-negotiable is the reason this sub-optimization
exists in its current form and should survive the rework.

`run_combine_sweep` sits on top of the scoring machinery: reuses one already-extracted cube,
varies only combine-level knobs (`stack_method`, `sigma_clip_*`). Kept in for now, but Austin
hasn't actually run it either — comparisons today are done manually (build/load stacks by hand
in the notebook, call `noise_from_stacks` per label). Not confirmed load-bearing; don't assume
it's exercised by anything.

**`run_background_sweep` is being deleted.** It re-ran Stage 1 extraction per variant
(background/masking/smoothing axes); Austin configures and runs those manually and has no use
for the automated version — not "unused for now," not being kept around. Remove the function
and its exclusive helpers.

Scoring/ranking (`rank_scores`, `score_table`, `plot_score_summary`, `plot_bin_detail`) is
generic over however many labeled results you hand it — doesn't care whether they came from
`run_combine_sweep` or from manual notebook calls, as long as they're anchored to the same
galaxy set and radial binning.

### Why sideband, not line

Ranking pipeline variants by how strong the Lya line looks *in that variant's own stack* is
circular: numerator and denominator both come from the thing you're about to report, so a
variant can win by coincidentally biasing or smoothing the line, not by genuinely having a
quieter background. The sideband window (1150–1300 Å default, current real usage 1150–1350 Å)
sidesteps that — it's real data, reuses the same bootstrap/stack machinery the pipeline already
trusts (`measure.stack_galaxies`), and is a reasonable proxy for background quality without
requiring the exact published quantity to be the scoring input. **This constraint stays in
place for the rework below** — the new line-S/N number is a diagnostic sitting next to the
ranking, not a ranking input. See "Line/off-line diagnostic" for how that's enforced.

## Data

Consumes `stacks` dicts from `pipeline.run_stack(..., keep_cube=True)`:
`cube_flux`, `cube_err`, `cube_weights`, `rest_wave`, each `(ngal, nrad, nwave)` /
`(ngal, nrad)` / `(nwave,)`.

Real usage (from `5_Testing_Background.ipynb`, not just the docstring defaults):

```python
scores = {
    L: optimize.noise_from_stacks(
        stacks[L], cfg_for[L],
        window=(1150.0, 1350.0),
        lines=DEFAULT_UV_LINES, line_hw=5.0,
        stack_method=None, height_reduce="biweight",
        nboot=1000, reduce="rms", seed=random_seed,
    )
    for L in stacks
}
```

`nboot=1000` at this window width is the slow call with zero progress feedback today — the
motivating case for the progress-bar change below. `window=(1150, 1350)` matters for the
line-S/N rework: it already spans Lyα rest (1215.67 Å), so the sideband slice already contains
the line pixels before `_line_window_mask` discards them.

`lines=DEFAULT_UV_LINES` (from `validation.py`) is every cataloged UV line in range (Lyβ, Lyα,
N V, Si II, Si II*, O I+Si II, C II, Si IV, Si II, C IV, Fe II, He II), not just Lyα —
`line_hw` masks ±hw around each.

## Method

### Current: `continuum_noise_metric`

1. Select sideband pixels: `window` bounds intersected with `~_line_window_mask(rest_wave,
   lines, line_hw)` — every cataloged line masked out, not just Lya.
2. Fiducial (all-galaxy, no resampling) stack → `height_per_bin` via a LOCATION reducer
   (`biweight`/`median`/`mean` — signed, since it's a pedestal that can be negative from
   over-subtraction). Diagnostic only, never ranked.
3. Bootstrap the galaxy axis `nboot` times (`rng.integers(0, ngal, ngal)`, with replacement):
   each draw restacks via `stack_galaxies` (method from `config.measure_stack_method` unless
   overridden), producing `draws[b]` of shape `(nrad, n_window_px)`.
4. `err_spectrum = nanstd(draws, axis=0)` → per-pixel bootstrap scatter. Collapses to
   `noise_per_bin` via a SCALE reducer (`rms` default / `median` / `biweight`).
5. Returns noise, height, fiber count per bin, plus the raw error/continuum spectra for
   plotting.

`noise_from_stacks` is the everyday entry point — unpacks a `stacks` dict straight into
`continuum_noise_metric`.

### Current: scoring/plotting

- `rank_scores` — per-bin ordinal rank across methods + a log-space product of `noise_per_bin`
  across all bins (`noise_product`), plus `mean_rank`. Both lower-is-better.
- `score_table` — `rank_scores` as a sorted astropy Table.
- `plot_score_summary` — two bar charts (noise_product, mean_rank), green=good via
  `RdYlGn_r`.
- `plot_bin_detail` — same two-panel bar layout for ONE radial bin: error (ranked) + height
  (diagnostic, signed, diverging colormap).

### Removed: `run_background_sweep`

Delete the function, its docstring/example, and anything exclusively in service of it (its
`use_cache`/`table` forwarding into `pipeline.run_extract` was only ever exercised through this
path). `run_combine_sweep` and the scoring/plotting functions don't depend on it and are
untouched. If a future need for an automated extraction sweep shows up, it's a fresh
sub-optimization to design against this hub, not a resurrection of this function as-is.

### Planned change 1 — progress bar on the bootstrap loop

`continuum_noise_metric`'s `for b in range(nboot)` loop (the actual expensive part —
1000 draws, each a full `stack_galaxies` restack) currently has no `verbose` param and no
feedback at all. `measure.bootstrap_stack_error` and `measure.bootstrap_all` already solve
this exact problem with a house pattern:

```python
desc = run_header(label or "...", verbose=verbose, announce=announce, nboot=nboot, ...)
for b in tqdm(range(nboot), disable=not verbose, desc=desc):
    ...
```

Adopt the same pattern here rather than inventing a new one: add `verbose: bool = True` (and
optionally `announce: bool = True`, `label=None` to match) to `continuum_noise_metric` and
`noise_from_stacks`, import `run_header` from `.measure`, wrap the loop in `tqdm`.

`run_combine_sweep` already forwards arbitrary `**metric_kwargs` into `noise_from_stacks`, so
`verbose=False` is already passable per-call with no signature change there.

### Planned change 2 — line / off-line S/N diagnostic (optional, default off)

New optional parameter on `continuum_noise_metric`, e.g. `compute_line_snr: bool = False`.
Window/continuum-model parameters are **read from `config`**, not hand-rolled — same convention
`measure.measure_all_bins` already uses: `bounds = tuple(config.line_window)`,
`cont_bounds/cont_method/cont_order = getattr(config, "cont_bounds"/"cont_method"/"cont_order",
DEFAULT_*)`, `lya_center = float(config.LYA_REST)`. No new half-width parameter invented in
optimize.py — this was the point of routing through config: the line window this diagnostic
uses is the SAME line window the rest of Stage 3 uses, not a second independent definition of
"where the line is."

**Don't reinvent the integrator.** `measure.integrated_line_flux(wave, flux, err=None, bounds,
cont_bounds, lya_center, cont_method, cont_order)` already exists and is, per its own
docstring, "the ONE integrator the rest of the pipeline routes through (the z-scramble null,
the placebo null, the SN-spectrum line table... blue/red side flux)" — continuum-model
subtraction included, not a raw `nansum` over the line window like my first draft had it. Call
this per bin, per bootstrap draw, on the already-restacked `stack_bs[r]` — reusing it here
means the line-S/N number is built the same way as every other integrated-flux number in the
codebase, not a fourth convention.

**Efficiency, per Austin's ask to reuse "the efficient one" (`bootstrap_all`'s one-pass
pattern):** because `window` already spans the Lya line at typical usage (1150–1350 vs. Lyα
1215.67), no window-widening step is needed — the cube slice and the `idx` draws are already
identical to what the noise metric uses. The only change: the pixels inside `config.line_window`
are currently discarded before `fcube`/`wv` are built (removed by `~_line_window_mask`), so they
need to survive one step further so `integrated_line_flux` has something to integrate. Same
`for b in range(nboot)` loop, same `stack_bs` already computed for the noise draw — one extra
call per bin per draw, no second bootstrap loop, no second restack.

**New reusable piece in `measure.py`, not optimize.py**: `measure.integrated_line_flux_per_bin`
— a small per-bin loop helper (nrad calls to `integrated_line_flux` over one stack,
config-driven bounds). `bootstrap_all` already has this exact shape internally as the local
closure `_measure_stack` (loops over `nrad`, calls `measure_centroid` + `blue_red_side_ratio`
per bin), just not for `integrated_line_flux` and not exported. This is the exported equivalent,
so optimize.py calls into `measure` for "integrate the line, per bin, for one stack," the same
way it already calls into `measure.stack_galaxies` for the restack. Keeps the line-flux
integration logic in one place instead of duplicated between `bootstrap_all` and optimize.py.

Per-draw, per-bin: `flux_sum[b, r]` from `integrated_line_flux(...)["flux_sum"]`. **Errors come