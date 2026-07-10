"""
optimize.py -- a HUB for scoring pipeline variants under whatever objective is
currently in question, NOT a tuner that hands you a "best" config
automatically, and NOT a single-purpose module. What's below is ONE
sub-optimization living in this hub -- background/continuum noise -- written
so a future, unrelated sub-optimization can sit alongside it without
restructuring this one.

THE CONTINUUM-NOISE SUB-OPTIMIZATION, scope deliberately narrow:
  - What this measures: the NOISE (scatter) of the continuum in a line-free
    part of the rest-frame spectrum, as a proxy for how noisy the background
    estimate feeding the Lya measurement itself is likely to be. It does NOT
    measure or optimize against the continuum's residual DC LEVEL (bias) --
    that's tracked elsewhere and is a separate question from noise.
  - What the RANKING deliberately does NOT do: injection-recovery,
    random-position null stacks, or rank variants by the S/N of the Lya line
    itself. Numerator/denominator circularity -- a variant could win by
    coincidentally biasing/smoothing the line, not by a genuinely quieter
    background, if you ranked on the thing you're about to report.
  - Why the sideband instead: if background subtraction is noisy, the
    continuum UNDER the line is presumably noisy too -- related, not
    identical, to line-measurement noise, but doesn't require optimizing the
    exact quantity being reported, and reuses the same bootstrap machinery
    (measure.stack_galaxies) the pipeline already trusts, on the real data,
    not a synthetic injected replica.
  - The one place the line IS touched: continuum_noise_metric's optional
    compute_line_snr diagnostic (default off) bootstraps an integrated
    line-flux S/N alongside the noise metric, from the SAME bootstrap draws,
    purely so the two can be plotted/compared -- "does minimizing continuum
    noise actually raise line S/N, empirically, on real data." It is NEVER
    fed into rank_scores / score_table / plot_score_summary's ranked fields;
    those still read only noise_per_bin / height_per_bin. Circularity is
    avoided because the line-S/N error comes from the line's OWN bootstrap
    flux scatter, not from noise_per_bin.

run_combine_sweep reuses ONE already-extracted cube and only varies
combine-level knobs (stack_method, sigma_clip_*) -- no re-extraction. (There
used to be a second, expensive tier here that re-ran Stage 1 extraction per
variant; removed -- unused, extraction-level comparisons are done manually.)

For scoring galaxy FITS you ALREADY extracted (e.g. PRODUCT_PATHS_* dicts), do
it inline in the notebook: load + finite-cut + (optionally match_products to
anchor on a shared galaxy set) + stack.build_stacks + noise_from_stacks -- see
the notebook cells that accompany this module. That path is deliberately NOT
wrapped in a function here, so the anchoring/matching stays visible and under
your control.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import numpy as np
from astropy.table import Table
from astropy.stats import biweight_scale, biweight_location

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm optional
    def tqdm(x, **kwargs):
        return x

from . import pipeline
from .measure import stack_galaxies, run_header, integrated_line_flux_per_bin, LYA_REST
from .config import DEFAULT_CONT_BOUNDS, DEFAULT_CONT_METHOD, DEFAULT_CONT_ORDER
from .validation import DEFAULT_UV_LINES, _line_window_mask

if TYPE_CHECKING:
    from .config import PipelineConfig


# =====================================================================
# 1. THE METRIC
# =====================================================================

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(x ** 2)))


def _median_abs(x: np.ndarray) -> float:
    return float(np.nanmedian(x))


def _biweight(x: np.ndarray) -> float:
    fin = x[np.isfinite(x)]
    if fin.size < 2:
        return float("nan")
    return float(biweight_scale(fin, ignore_nan=True))


_REDUCERS = {"rms": _rms, "median": _median_abs, "biweight": _biweight}


# --- HEIGHT reducers: LOCATION estimators (central level), not scale/RMS.
#     The continuum height is a SIGNED DC level (the pedestal), so it needs a
#     location estimator; using an RMS-type reducer here would fold the sign
#     away and report |level|, which is wrong for a pedestal that can be
#     negative from over-subtraction.
def _loc_biweight(x: np.ndarray) -> float:
    fin = x[np.isfinite(x)]
    if fin.size < 2:
        return float("nan")
    return float(biweight_location(fin, ignore_nan=True))


_HEIGHT_REDUCERS = {
    "biweight": _loc_biweight,
    "median":   lambda x: float(np.nanmedian(x)),
    "mean":     lambda x: float(np.nanmean(x)),
}


def _resolve_reducer(reduce: Union[str, Callable]) -> Callable:
    """reduce may be 'rms' | 'median' | 'biweight', or your own callable
    (array -> float) if you want something else top-tier."""
    if callable(reduce):
        return reduce
    try:
        return _REDUCERS[reduce]
    except KeyError:
        raise ValueError(
            f"reduce must be one of {list(_REDUCERS)} or a callable, got {reduce!r}"
        ) from None


def _resolve_height_reducer(height_reduce: Union[str, Callable]) -> Callable:
    """height_reduce may be 'biweight' | 'median' | 'mean', or a callable.
    LOCATION estimators only -- height is a signed level, not a scatter."""
    if callable(height_reduce):
        return height_reduce
    try:
        return _HEIGHT_REDUCERS[height_reduce]
    except KeyError:
        raise ValueError(
            f"height_reduce must be one of {list(_HEIGHT_REDUCERS)} or a "
            f"callable, got {height_reduce!r}"
        ) from None


def continuum_noise_metric(
    cube_flux, cube_err, weights, rest_wave, config: "PipelineConfig",
    window: tuple = (1150.0, 1300.0),
    lines: Sequence = DEFAULT_UV_LINES, line_hw: float = 6.0,
    stack_method: Optional[str] = None,
    nboot: int = 200, reduce: Union[str, Callable] = "rms",
    height_reduce: Union[str, Callable] = "biweight", seed: int = 1,
    verbose: bool = True, announce: bool = True, label: Optional[str] = None,
    compute_line_snr: bool = False, line_snr_robust: bool = True,
) -> dict:
    """
    Bootstrap the galaxy axis, restack, and measure the per-pixel scatter in a
    LINE-MASKED sideband window -> one noise scalar per radial bin. ALSO reports
    the continuum HEIGHT (signed DC level / pedestal) in the same window -- for
    visibility only; height is deliberately NOT part of any ranking, since a
    residual level can come from many benign sources and the line measurement
    doesn't rest on it. Noise is the thing being minimized; height is a
    diagnostic shown alongside.

    cube_flux/cube_err : (ngal, nrad, nwave), e.g. stacks['cube_flux']/['cube_err']
        from run_stack(..., keep_cube=True). Use noise_from_stacks() instead of
        calling this directly if you already have a stacks dict.
    weights : (ngal, nrad) or None, e.g. stacks['cube_weights'] (defaults to
        per-bin fiber count). Only consumed if stack_method == 'weighted_median'
        (required in that case); otherwise ignored by the combine, but ALWAYS
        summed per bin and returned as n_fib_per_bin for visibility. NEVER
        divided into the noise number -- more fibers is not a confound here,
        it's the point of good masking.
    window : rest-frame span to draw noise pixels from. Default (1150, 1300)
        is wide for a stable estimate, but genuinely contains real cataloged
        features (NV 1238.82/1242.80, SiII 1260.42, SiII* 1264.74) -- these are
        masked out via `lines`/`line_hw` below, not just Lya. Do not shrink
        `line_hw` to zero; that puts NV/SiII pixels back into the "noise".
        Also, at typical usage (e.g. 1150-1350), this window already spans Lya
        rest (1215.67) -- see compute_line_snr below, which relies on that.
    lines : (name, rest_wavelength_A, kind) tuples to exclude around, default
        validation.DEFAULT_UV_LINES (every cataloged UV line, not just Lya).
    line_hw : half-width (A) masked around each entry in `lines`.
    stack_method : galaxy-combine method for each bootstrap draw AND the
        fiducial (all-galaxy) height stack. None -> config.measure_stack_method.
    reduce : how the per-pixel bootstrap-sigma spectrum collapses to one NOISE
        number per bin. 'rms' (default), 'median', 'biweight', or a callable.
        SCALE estimator (measures scatter).
    height_reduce : how the fiducial continuum collapses to one HEIGHT number
        per bin. 'biweight' (default), 'median', 'mean', or a callable.
        LOCATION estimator (measures signed level) -- NOT interchangeable with
        `reduce`; using an RMS reducer here would report |level| and lose the
        sign of an over-subtracted (negative) pedestal.
    verbose : print a run_header banner and show a tqdm progress bar over the
        nboot bootstrap draws -- this is the slow loop (a full stack_galaxies
        restack per draw), same house pattern as measure.bootstrap_stack_error
        / measure.bootstrap_all. announce/label forwarded to run_header.
    compute_line_snr : DIAGNOSTIC, default off. Bootstraps an integrated Lya
        line-flux S/N alongside the noise metric, from the SAME bootstrap
        draws (no extra restack, no extra rng draws) -- see module docstring
        "the one place the line IS touched." Window/continuum-model params
        come from `config` (line_window, cont_bounds, cont_method, cont_order,
        LYA_REST), matching measure.measure_all_bins' convention, not a second
        independently-invented line-window definition. Requires `window`
        (above) to already contain config.line_window and config.cont_bounds --
        true at typical usage, checked below. Adds ~one measure.
        integrated_line_flux_per_bin call per draw (cheap relative to the
        restack, but not free -- time it before assuming it's negligible on a
        large nboot).
    line_snr_robust : for compute_line_snr, how the line-flux bootstrap spread
        collapses to one error number per bin: True (default) -> 0.5*(p84-p16)
        (matches bootstrap_stack_error's robust=True), False -> plain std.
        This error comes from the line flux's OWN spread across draws, NOT
        from noise_per_bin -- see module docstring; that's what keeps the
        noise-vs-line-S/N comparison a real test instead of a tautology.

    Returns dict:
      noise_per_bin, height_per_bin, n_fib_per_bin  : (nrad,) each
      err_spectrum, continuum_spectrum              : (nrad, n_window_px) each
          (per-pixel bootstrap-sigma, and the fiducial stacked flux over the
           line-free window -- plot continuum_spectrum to SEE the pedestal /
           wiggle shape directly)
      window_wave (n_window_px,); n_window_px; stack_method; reduce (name);
      height_reduce (name); nboot.
      If compute_line_snr: ALSO line_snr_per_bin, line_flux_per_bin,
          line_flux_err_per_bin (each (nrad,)), line_bounds, line_snr_robust.
          NOT consumed by rank_scores/score_table/plot_score_summary -- those
          only ever read noise_per_bin/height_per_bin. Plot with
          plot_line_snr_summary instead.
    """
    stack_method = stack_method or config.measure_stack_method

    rest_wave = np.asarray(rest_wave, dtype=float)
    cube_flux = np.asarray(cube_flux, dtype=float)
    cube_err = np.asarray(cube_err, dtype=float) if cube_err is not None else None
    if cube_flux.ndim != 3:
        raise ValueError(f"cube_flux must be (ngal, nrad, nwave), got shape {cube_flux.shape}")
    ngal, nrad, _ = cube_flux.shape

    # --- select the FULL window first (line pixels included), THEN derive the
    #     line-masked sideband as a sub-selection of it. Old code selected the
    #     sideband directly and threw the line pixels away before they ever
    #     reached a numpy array; compute_line_snr needs them to survive one
    #     step further so integrated_line_flux_per_bin has something to
    #     integrate. When compute_line_snr=False this produces IDENTICAL
    #     noise_per_bin/height_per_bin/... to before -- noise_sel plays
    #     exactly the role the old `sel` did. ---
    sel_win = (rest_wave >= window[0]) & (rest_wave <= window[1])
    wv = rest_wave[sel_win]
    noise_sel = ~_line_window_mask(wv, lines, line_hw)
    if noise_sel.sum() < 5:
        raise ValueError(
            f"only {int(noise_sel.sum())} line-free pixels survive window={window}, "
            f"line_hw={line_hw} -- widen the window, shrink line_hw, or check "
            "rest_wave coverage before trusting this metric"
        )

    fcube = cube_flux[:, :, sel_win]
    ecube = cube_err[:, :, sel_win] if cube_err is not None else None
    w_all = None if weights is None else np.asarray(weights, dtype=float)

    reducer = _resolve_reducer(reduce)
    height_reducer = _resolve_height_reducer(height_reduce)

    if compute_line_snr:
        line_bounds = tuple(getattr(config, "line_window", (LYA_REST - 4.0, LYA_REST + 4.0)))
        cont_bounds = getattr(config, "cont_bounds", DEFAULT_CONT_BOUNDS)
        cont_method = getattr(config, "cont_method", DEFAULT_CONT_METHOD)
        cont_order = getattr(config, "cont_order", DEFAULT_CONT_ORDER)
        lya_center = float(getattr(config, "LYA_REST", LYA_REST))
        needed = list(line_bounds) + [b for side in cont_bounds for b in side]
        if min(needed) < wv.min() or max(needed) > wv.max():
            raise ValueError(
                f"compute_line_snr needs window={window} to contain "
                f"config.line_window={line_bounds} and config.cont_bounds="
                f"{cont_bounds}, but the sliced grid only spans "
                f"({wv.min():.1f}, {wv.max():.1f}) -- widen `window`."
            )

    # --- fiducial (all-galaxy) stack -> continuum height (level), no bootstrap ---
    fid_stack, _ = stack_galaxies(
        fcube, ecube, method=stack_method, weights=w_all,
        sigma=config.sigma_clip_sigma, maxiters=config.sigma_clip_maxiters,
    )                                                          # (nrad, n_window_px), FULL window
    height_per_bin = np.array([height_reducer(fid_stack[r, noise_sel]) for r in range(nrad)])

    # --- bootstrap the galaxy axis -> per-pixel scatter (noise), and
    #     optionally line-flux S/N, from the SAME draws ---
    rng = np.random.default_rng(seed)
    draws = np.empty((nboot, nrad, int(noise_sel.sum())), dtype=np.float64)
    line_flux_draws = np.full((nboot, nrad), np.nan) if compute_line_snr else None

    desc = run_header(
        label or "continuum noise bootstrap", verbose=verbose, announce=announce,
        nboot=nboot, window=window, stack=stack_method,
        compute_line_snr=compute_line_snr, seed=seed,
    )
    for b in tqdm(range(nboot), disable=not verbose, desc=desc):
        idx = rng.integers(0, ngal, ngal)
        flux_bs = fcube[idx]
        err_bs = ecube[idx] if ecube is not None else None
        w_bs = w_all[idx] if w_all is not None else None
        stack_bs, _ = stack_galaxies(
            flux_bs, err_bs, method=stack_method, weights=w_bs,
            sigma=config.sigma_clip_sigma, maxiters=config.sigma_clip_maxiters,
        )                                                       # (nrad, n_window_px), FULL window
        draws[b] = stack_bs[:, noise_sel]
        if compute_line_snr:
            flux_sum, _ = integrated_line_flux_per_bin(
                wv, stack_bs, bounds=line_bounds, cont_bounds=cont_bounds,
                lya_center=lya_center, cont_method=cont_method, cont_order=cont_order,
            )
            line_flux_draws[b] = flux_sum

    err_spectrum = np.nanstd(draws, axis=0)                     # (nrad, n_noise_px)
    noise_per_bin = np.array([reducer(err_spectrum[r]) for r in range(nrad)])
    n_fib_per_bin = None if w_all is None else np.nansum(w_all, axis=0)

    result = {
        "noise_per_bin": noise_per_bin,
        "height_per_bin": height_per_bin,
        "n_fib_per_bin": n_fib_per_bin,
        "err_spectrum": err_spectrum,
        "continuum_spectrum": fid_stack[:, noise_sel],
        "window_wave": wv[noise_sel],
        "n_window_px": int(noise_sel.sum()),
        "stack_method": stack_method,
        "reduce": reduce if isinstance(reduce, str) else getattr(reduce, "__name__", "custom"),
        "height_reduce": height_reduce if isinstance(height_reduce, str) else getattr(height_reduce, "__name__", "custom"),
        "nboot": nboot,
    }

    if compute_line_snr:
        line_flux_med = np.nanmedian(line_flux_draws, axis=0)
        if line_snr_robust:
            p16 = np.nanpercentile(line_flux_draws, 16, axis=0)
            p84 = np.nanpercentile(line_flux_draws, 84, axis=0)
            line_flux_err = 0.5 * (p84 - p16)
        else:
            line_flux_err = np.nanstd(line_flux_draws, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            line_snr_per_bin = line_flux_med / line_flux_err
        result.update({
            "line_snr_per_bin": line_snr_per_bin,
            "line_flux_per_bin": line_flux_med,
            "line_flux_err_per_bin": line_flux_err,
            "line_bounds": line_bounds,
            "line_snr_robust": line_snr_robust,
        })

    return result


def noise_from_stacks(stacks: dict, config: "PipelineConfig", **kwargs) -> dict:
    """
    Convenience wrapper: unpack a run_stack(..., keep_cube=True) output dict
    straight into continuum_noise_metric. `verbose`, `compute_line_snr`, and
    every other continuum_noise_metric keyword forward through **kwargs
    unchanged -- no separate signature to keep in sync.

    Example
    -------
        stacks = pipeline.run_stack(config, galaxy_fits_path, keep_cube=True)
        result = optimize.noise_from_stacks(stacks, config)
        print(result["noise_per_bin"], result["n_fib_per_bin"])

        # with the line-S/N diagnostic on:
        result = optimize.noise_from_stacks(stacks, config, compute_line_snr=True)
        print(result["line_snr_per_bin"])
    """
    missing = [k for k in ("cube_flux", "cube_weights", "rest_wave") if k not in stacks]
    if missing:
        raise ValueError(
            f"stacks is missing {missing} -- was it built with keep_cube=True?"
        )
    return continuum_noise_metric(
        stacks["cube_flux"], stacks.get("cube_err"), stacks["cube_weights"],
        stacks["rest_wave"], config, **kwargs,
    )


# =====================================================================
# 2. CHEAP TIER -- combine-level sweep on an ALREADY-EXTRACTED cube
#    (stack_method / sigma_clip_* only; no re-extraction)
# =====================================================================

def run_combine_sweep(
    config: "PipelineConfig", stacks: dict, variants: Sequence[dict], **metric_kwargs
) -> Table:
    """
    Score several combine-level variants against ONE already-built stacks
    cube (from run_stack(..., keep_cube=True)) -- no re-extraction, so this is
    cheap and safe to run broadly.

    variants : list of dicts of PipelineConfig field overrides that only
        affect Stage 2/3 combining, e.g.
            [{"measure_stack_method": "biweight"},
             {"measure_stack_method": "sigma_clip", "sigma_clip_sigma": 2.5},
             {"measure_stack_method": "median"}]
        Unknown field names raise immediately (dataclasses.replace), not a
        silent no-op.

    Returns an astropy Table, one row per variant, with the override columns
    plus noise_per_bin / n_fib_per_bin (each an (nrad,) array per row).

    Example
    -------
        stacks = pipeline.run_stack(config, galaxy_fits_path, keep_cube=True)
        variants = [{"measure_stack_method": m} for m in
                   ("biweight", "median", "sigma_clip")]
        table = optimize.run_combine_sweep(config, stacks, variants)
        table.pprint_all()
    """
    rows = []
    for variant in variants:
        cfg_v = replace(config, **variant)
        result = noise_from_stacks(stacks, cfg_v, **metric_kwargs)
        rows.append({
            **variant,
            "noise_per_bin": result["noise_per_bin"],
            "n_fib_per_bin": result["n_fib_per_bin"],
            "stack_method": result["stack_method"],
        })
    return Table(rows=rows)


# =====================================================================
# 3. SCORING + HISTOGRAM VIEWS
#    Turn a {label: metric-result} dict into (a) two summary bar charts --
#    noise-product and mean-rank across ALL bins -- and (b) a per-bin drill-
#    down showing the error and the continuum height for one chosen radius.
#    Pure display on top of noise_from_stacks output; nothing here re-extracts
#    or re-stacks. matplotlib is imported lazily inside the plotters so that
#    importing optimize (for the metric/sweep path on a headless node) never
#    drags in matplotlib -- same principle as plotting.py.
# =====================================================================

def _rank_column(col: np.ndarray) -> np.ndarray:
    """Ordinal rank of one bin's values across methods: 1 = lowest (best),
    NaN stays NaN (a method missing this bin is simply not ranked here, so it
    doesn't distort the others). Ties break by input order (stable)."""
    col = np.asarray(col, dtype=float)
    out = np.full(col.shape, np.nan)
    fin = np.isfinite(col)
    if fin.any():
        order = np.argsort(col[fin], kind="mergesort")     # ascending
        r = np.empty(order.size)
        r[order] = np.arange(1, order.size + 1)
        out[fin] = r
    return out


def _bar_colors(vals, cmap, lo=None, hi=None, gamma=1.0, center=None,
                bad_color=(0.6, 0.6, 0.6, 1.0)):
    """
    Map bar values -> RGBA. Two modes:
      sequential (center is None): normalize to [lo, hi] (data range if None),
        raise to `gamma` (>1 pushes mid values toward the low/'good' end of the
        map), then look up `cmap`. With cmap='RdYlGn_r' the LOW end is green, so
        low-is-good quantities (noise, rank) read green=good out of the box.
      diverging (center given): symmetric normalization about `center`, so a
        signed quantity (e.g. continuum height, +/- about 0) colors by sign and
        magnitude. Use a diverging cmap like 'coolwarm' / 'RdBu'.
    Non-finite values get `bad_color`.
    """
    import matplotlib.pyplot as plt
    vals = np.asarray(vals, dtype=float)
    cm = plt.get_cmap(cmap)
    if center is not None:
        m = np.nanmax(np.abs(vals - center))
        m = m if (np.isfinite(m) and m > 0) else 1.0
        t = (vals - center) / (2.0 * m) + 0.5
    else:
        lo = np.nanmin(vals) if lo is None else lo
        hi = np.nanmax(vals) if hi is None else hi
        span = (hi - lo) if (hi - lo) else 1.0
        t = np.clip((vals - lo) / span, 0.0, 1.0) ** gamma
    rgba = np.array([cm(x) if np.isfinite(x) else bad_color for x in t])
    return rgba


def _resolve_bin(bin_index: int, nrad: int) -> tuple:
    """(python index, human label) for a possibly-negative bin request."""
    idx = bin_index if bin_index >= 0 else nrad + bin_index
    if not (0 <= idx < nrad):
        raise IndexError(f"bin_index={bin_index} out of range for nrad={nrad}")
    tag = f"bin {bin_index}" + (f" (= index {idx})" if bin_index < 0 else "")
    return idx, f"{tag} of {nrad}"


def rank_scores(
    scores: dict,
    labels: Optional[Sequence[str]] = None,
    bin_slice: Optional[slice] = None,
    scale_decade: Union[str, int] = "auto",
) -> dict:
    """
    Collapse a {label: noise_from_stacks(...) result} dict into two per-METHOD
    numbers, the raw material for the summary histograms:

      noise_product : multiply each method's per-bin error together ACROSS ALL
          BINS (one number per row of err_spectrum -> one number per method).
          Done in log space (a product of ~1e-20 per-bin errors underflows
          float64 otherwise), then the whole set is shifted by a single common
          power of ten so the SMALLEST lands in [1, 10). This is a decimal-point
          move, NOT a divide-by-the-minimum: every method-to-method ratio is
          preserved exactly (score_i / score_j == prod_i / prod_j), only the
          overall magnitude is made readable. Lower = quieter continuum = better.
      mean_rank : within each bin, rank the methods by error (1 = lowest), then
          average a method's rank over the bins. Lower = better; with M methods
          the floor is 1 and the ceiling is M, so e.g. 3.4 out of 9 is good.

    Operates on `noise_per_bin` (the per-bin REDUCED scatter the metric already
    computed) -- NOT the raw per-pixel err_spectrum. Producting every pixel
    would amplify tiny per-method differences by the pixel count and blow the
    dynamic range past anything a bar chart can show; the per-bin reduction is
    the honest granularity for a cross-method score.

    FAIRNESS: this compares whatever you hand it bin-for-bin, so the methods
    must be anchored to the SAME galaxy set and the SAME radial binning (the
    match_products -> shared-galaxy notebook pattern). A bin that is NaN for one
    method is dropped from that method's product and skipped in that bin's
    ranking, so it neither helps nor penalizes -- but a method NaN in bins the
    others have will look artificially good on the product (fewer factors).
    Watch n_bins_used per method if you see NaNs.

    Parameters
    ----------
    scores : {label: dict}
        Each value is a noise_from_stacks / continuum_noise_metric result;
        needs "noise_per_bin" and "height_per_bin".
    labels : order/subset of methods to score. Default: all keys, dict order.
    bin_slice : restrict the score to a slice of bins, e.g. slice(1, None) to
        drop the innermost bin. Default None = all bins.
    scale_decade : "auto" -> shift by floor(min log10-product) so the smallest
        product is in [1, 10). Or pass an int k to force 10**(logprod - k)
        yourself (handy to hold the scale fixed across several score sets).

    Returns
    -------
    dict with:
        labels (list, len M); noise_per_bin, height_per_bin, rank_per_bin
        (each (M, nbin)); logprod, noise_product, mean_rank, n_bins_used
        (each (M,)); scale_decade (int).
    """
    labels = list(scores) if labels is None else list(labels)
    if not labels:
        raise ValueError("no labels to score")

    noise = np.array([np.asarray(scores[L]["noise_per_bin"], float) for L in labels])
    height = np.array([np.asarray(scores[L]["height_per_bin"], float) for L in labels])
    if bin_slice is not None:
        noise, height = noise[:, bin_slice], height[:, bin_slice]

    # per-bin ordinal ranks (rank each column across methods), then average
    rank_per_bin = np.column_stack([_rank_column(noise[:, j])
                                    for j in range(noise.shape[1])])
    mean_rank = np.nanmean(rank_per_bin, axis=1)

    # log-space product across bins; guard non-positive / non-finite errors
    safe = np.where(np.isfinite(noise) & (noise > 0), noise, np.nan)
    n_bins_used = np.sum(np.isfinite(safe), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logprod = np.nansum(np.log10(safe), axis=1)          # (M,)

    if scale_decade == "auto":
        finite_lp = logprod[np.isfinite(logprod)]
        k = int(np.floor(np.min(finite_lp))) if finite_lp.size else 0
    else:
        k = int(scale_decade)
    noise_product = 10.0 ** (logprod - k)

    return {
        "labels": labels,
        "noise_per_bin": noise,
        "height_per_bin": height,
        "rank_per_bin": rank_per_bin,
        "logprod": logprod,
        "noise_product": noise_product,
        "mean_rank": mean_rank,
        "n_bins_used": n_bins_used,
        "scale_decade": k,
    }


def score_table(scores: dict, sort_by: str = "noise_product", **rank_kwargs) -> Table:
    """
    rank_scores() as a printable astropy Table, one row per method, sorted best-
    first. Columns: label, noise_product, mean_rank, n_bins_used. Handy for a
    quick text ranking to sit beside the plots (or to hand to a larger report).

    Example
    -------
        optimize.score_table(scores).pprint_all()
    """
    R = rank_scores(scores, **rank_kwargs)
    tbl = Table({
        "label": R["labels"],
        "noise_product": np.round(R["noise_product"], 4),
        "mean_rank": np.round(R["mean_rank"], 3),
        "n_bins_used": R["n_bins_used"],
    })
    if sort_by in tbl.colnames:
        tbl.sort(sort_by)          # ascending: lower noise_product / rank first
    return tbl


def plot_score_summary(
    scores: dict,
    labels: Optional[Sequence[str]] = None,
    bin_slice: Optional[slice] = None,
    scale_decade: Union[str, int] = "auto",
    sort_by: str = "noise_product",
    figsize: tuple = (12.0, 4.8),
    noise_cmap: str = "RdYlGn_r",
    rank_cmap: str = "RdYlGn_r",
    rank_norm: tuple = (1, None),
    rank_gamma: float = 1.8,
    noise_gamma: float = 1.0,
    log_y_noise: bool = False,
    annotate: bool = True,
    axes=None,
):
    """
    THE TWO SUMMARY HISTOGRAMS (Function A): one bar per method, side by side.
      LEFT  -- noise_product across all bins (rescaled so the best is ~1-10).
      RIGHT -- mean_rank across all bins.
    Bars are colored by value so both panels read green=good -> red=bad.

    Parameters (all strung through so you can restyle or drop this into a larger
    figure via `axes`)
    ----------
    scores : {label: noise_from_stacks result}.
    labels, bin_slice, scale_decade : forwarded to rank_scores (see there).
    sort_by : "noise_product" | "mean_rank" | "none" -- bar order, best-first.
    figsize : used only when this creates its own figure (axes is None).
    noise_cmap / rank_cmap : bar colormaps. RdYlGn_r puts green at the low
        (good) end. The noise panel normalizes over the DATA range; the rank
        panel normalizes over an ABSOLUTE range (see rank_norm) so a good rank
        looks good on its own terms, not merely relative to the others.
    rank_norm : (lo, hi) for the mean_rank color scale. Default (1, None) ->
        lo=1 (best possible), hi=None means "number of methods" (worst
        possible). This is why 3.4/9 lands green rather than mid-scale: it's
        graded against 1..M, not against min..max of this particular set.
    rank_gamma : >1 widens the green band (pushes middling ranks greener). 1.8
        is tuned so mean_rank ~3-4 out of ~9 reads clearly green; raise it to
        green more aggressively, set 1.0 for a straight linear ramp.
    noise_gamma : same knob for the noise panel (default 1.0 = linear).
    log_y_noise : log-scale the noise_product axis if the spread is wide.
    annotate : print each bar's value on top.
    axes : optional (ax_left, ax_right) to draw into an existing figure. If
        None, a new (fig, (axL, axR)) is created.

    Returns (fig, table) -- table is score_table() so the numbers travel with
    the figure.

    Example
    -------
        # scores must carry height (metric change of 2026-07); rebuild if old:
        scores = {L: opt.noise_from_stacks(stacks[L], cfg_for[L],
                                            nboot=200, reduce="rms",
                                            height_reduce="biweight")
                  for L in stacks}
        fig, tbl = opt.plot_score_summary(scores)     # the two histograms
        tbl.pprint_all()                              # same ranking, as text
        # tweak: drop the innermost bin, green harder, log the noise axis
        opt.plot_score_summary(scores, bin_slice=slice(1, None),
                               rank_gamma=2.5, log_y_noise=True)
    """
    import matplotlib.pyplot as plt

    R = rank_scores(scores, labels=labels, bin_slice=bin_slice,
                    scale_decade=scale_decade)
    lab = np.array(R["labels"])
    npd, mrk = R["noise_product"], R["mean_rank"]

    key = {"noise_product": npd, "mean_rank": mrk}.get(sort_by)
    order = np.argsort(key) if key is not None else np.arange(len(lab))
    lab, npd, mrk = lab[order], npd[order], mrk[order]

    M = len(lab)
    rlo = rank_norm[0]
    rhi = M if rank_norm[1] is None else rank_norm[1]

    if axes is None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    else:
        axL, axR = axes
        fig = axL.figure
    x = np.arange(M)

    # LEFT: noise product (data-range color)
    cL = _bar_colors(npd, noise_cmap, gamma=noise_gamma)
    axL.bar(x, npd, color=cL, edgecolor="k", linewidth=0.4)
    axL.set_ylabel("noise product  (smallest rescaled to ~1-10)")
    kexp = R["scale_decade"]
    axL.set_title(f"NOISE PRODUCT over bins  (x10$^{{{kexp}}}$; lower = better)")
    if log_y_noise:
        axL.set_yscale("log")

    # RIGHT: mean rank (absolute 1..M color, gamma-widened green)
    cR = _bar_colors(mrk, rank_cmap, lo=rlo, hi=rhi, gamma=rank_gamma)
    axR.bar(x, mrk, color=cR, edgecolor="k", linewidth=0.4)
    axR.set_ylabel("mean rank across bins")
    axR.set_title(f"MEAN RANK  (1 = best of {M}; lower = better)")
    axR.set_ylim(0, M + 0.5)
    axR.axhline((1 + M) / 2.0, color="grey", lw=0.8, ls=":")  # 'middle of pack'

    for ax, vals in ((axL, npd), (axR, mrk)):
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
        ax.margins(x=0.01)
        if annotate:
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                                fontsize=7, xytext=(0, 1),
                                textcoords="offset points")

    fig.suptitle("Method scoring: noise product + mean rank (all bins)",
                 fontsize=12)
    fig.tight_layout()
    return fig, score_table(scores, sort_by=sort_by, labels=labels,
                            bin_slice=bin_slice, scale_decade=scale_decade)


def plot_bin_detail(
    scores: dict,
    bin_index: int,
    labels: Optional[Sequence[str]] = None,
    sort_by: str = "error",
    figsize: tuple = (12.0, 4.8),
    err_cmap: str = "RdYlGn_r",
    err_gamma: float = 1.0,
    height_cmap: str = "coolwarm",
    annotate: bool = True,
    axes=None,
):
    """
    THE PER-BIN DRILL-DOWN (Function B): pick ONE radial bin and show, one bar
    per method:
      LEFT  -- the error (noise_per_bin) at that bin. Green=good (low).
      RIGHT -- the continuum height (signed pedestal) at that bin. Diverging
               color about zero, with a zero line, so an over-subtracted
               (negative) pedestal is visually distinct from a positive one.
    Separate from plot_score_summary on purpose: that one ranks across all bins;
    this one opens up a single radius so you can see who is quiet/level THERE.

    Parameters
    ----------
    scores : {label: noise_from_stacks result}.
    bin_index : which radial bin. Negative indexes from the outside in
        (-1 = outermost), like normal python indexing.
    labels : order/subset of methods. Default: all, dict order.
    sort_by : "error" (default, quietest first) | "height" (most negative
        first) | "none" (dict order). Both panels share the one order.
    err_cmap / err_gamma : error-bar colormap + green-widen exponent.
    height_cmap : diverging colormap for the signed height panel.
    annotate : value labels on the bars.
    axes : optional (ax_left, ax_right) to draw into an existing figure.

    Returns the figure.

    Example
    -------
        opt.plot_bin_detail(scores, 0)     # innermost bin
        opt.plot_bin_detail(scores, -1)    # outermost bin
        opt.plot_bin_detail(scores, 4, sort_by="height")
    """
    import matplotlib.pyplot as plt

    labels = list(scores) if labels is None else list(labels)
    err = np.array([np.asarray(scores[L]["noise_per_bin"], float) for L in labels])
    hgt = np.array([np.asarray(scores[L]["height_per_bin"], float) for L in labels])
    nrad = err.shape[1]
    idx, bin_label = _resolve_bin(bin_index, nrad)

    lab = np.array(labels)
    e_b, h_b = err[:, idx], hgt[:, idx]

    if sort_by == "error":
        order = np.argsort(e_b)
    elif sort_by == "height":
        order = np.argsort(h_b)
    else:
        order = np.arange(len(lab))
    lab, e_b, h_b = lab[order], e_b[order], h_b[order]

    if axes is None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    else:
        axL, axR = axes
        fig = axL.figure
    x = np.arange(len(lab))

    # LEFT: error at this bin
    axL.bar(x, e_b, color=_bar_colors(e_b, err_cmap, gamma=err_gamma),
            edgecolor="k", linewidth=0.4)
    axL.set_ylabel("error (bootstrap-$\\sigma$)")
    axL.set_title(f"ERROR @ {bin_label}  (lower = better)")

    # RIGHT: continuum height at this bin (signed, diverging about 0)
    axR.bar(x, h_b, color=_bar_colors(h_b, height_cmap, center=0.0),
            edgecolor="k", linewidth=0.4)
    axR.axhline(0.0, color="k", lw=0.9, ls="--")
    axR.set_ylabel("continuum height (signed level)")
    axR.set_title(f"HEIGHT @ {bin_label}  (pedestal; diagnostic, unranked)")

    for ax, vals in ((axL, e_b), (axR, h_b)):
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
        ax.margins(x=0.01)
        if annotate:
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    va = "bottom" if v >= 0 else "top"
                    ax.annotate(f"{v:.2e}", (xi, v), ha="center", va=va,
                                fontsize=6.5, xytext=(0, 1 if v >= 0 else -1),
                                textcoords="offset points")

    fig.suptitle(f"Per-bin detail: {bin_label}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_line_snr_summary(
    scores: dict,
    labels: Optional[Sequence[str]] = None,
    sort_by: str = "mean_snr",
    figsize: tuple = (12.0, 4.8),
    snr_cmap: str = "RdYlGn",
    snr_gamma: float = 1.0,
    annotate: bool = True,
    axes=None,
):
    """
    Line-S/N companion to plot_score_summary -- DIAGNOSTIC, not ranked (see
    module docstring: continuum-noise ranking deliberately never touches the
    line; this plot is what "touching the line" looks like, kept separate on
    purpose, never fed back into rank_scores/score_table).

    LEFT  -- mean line S/N across bins, one bar per method. Higher = better
        here (note the flip from plot_score_summary's noise/rank panels,
        where lower is better) -- 'RdYlGn' (not '_r') puts green at the HIGH
        end accordingly.
    RIGHT -- line S/N vs radial bin, one curve per method, mirroring the
        noise-vs-radius plot elsewhere in the notebook, so you can see WHERE
        in radius a method's S/N (dis)advantage shows up, and eyeball it
        against noise_per_bin's own per-bin curve for the same methods.

    scores : {label: noise_from_stacks result}, each needing "line_snr_per_bin"
        (i.e. built with compute_line_snr=True).
    sort_by : "mean_snr" (default, best/highest first) | "none" (dict order).

    Returns (fig, table) -- table: label, mean_snr, sorted to match the bars.

    Example
    -------
        scores = {L: opt.noise_from_stacks(stacks[L], cfg_for[L],
                                            compute_line_snr=True, nboot=1000)
                  for L in stacks}
        fig, tbl = opt.plot_line_snr_summary(scores)
        tbl.pprint_all()
    """
    import matplotlib.pyplot as plt

    labels = list(scores) if labels is None else list(labels)
    missing = [L for L in labels if "line_snr_per_bin" not in scores[L]]
    if missing:
        raise ValueError(
            f"{missing} missing 'line_snr_per_bin' -- built with "
            "compute_line_snr=True?"
        )

    snr = np.array([np.asarray(scores[L]["line_snr_per_bin"], float) for L in labels])
    mean_snr = np.nanmean(snr, axis=1)

    order = np.argsort(-mean_snr) if sort_by == "mean_snr" else np.arange(len(labels))
    lab = np.array(labels)[order]
    m_snr = mean_snr[order]
    snr_ord = snr[order]

    if axes is None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    else:
        axL, axR = axes
        fig = axL.figure
    x = np.arange(len(lab))

    # LEFT: mean line S/N (higher = better -> RdYlGn, not the _r noise/rank cmaps)
    cL = _bar_colors(m_snr, snr_cmap, gamma=snr_gamma)
    axL.bar(x, m_snr, color=cL, edgecolor="k", linewidth=0.4)
    axL.set_ylabel("mean line S/N across bins")
    axL.set_title("LINE S/N  (higher = better; diagnostic, unranked)")
    axL.set_xticks(x)
    axL.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    axL.margins(x=0.01)
    if annotate:
        for xi, v in zip(x, m_snr):
            if np.isfinite(v):
                axL.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                            fontsize=7, xytext=(0, 1), textcoords="offset points")

    # RIGHT: line S/N vs radial bin, one curve per method
    nrad = snr.shape[1]
    for L, row in zip(lab, snr_ord):
        axR.plot(range(nrad), row, marker="o", label=L)
    axR.set_xlabel("radial bin index")
    axR.set_ylabel("line S/N")
    axR.set_title("LINE S/N vs radius, one curve per method")
    axR.legend(fontsize=7, ncol=2)
    axR.margins(x=0.02)

    fig.suptitle("Line S/N diagnostic -- NOT a ranking input (see module docstring)",
                fontsize=11)
    fig.tight_layout()

    tbl = Table({"label": lab, "mean_snr": np.round(m_snr, 4)})
    return fig, tbl
