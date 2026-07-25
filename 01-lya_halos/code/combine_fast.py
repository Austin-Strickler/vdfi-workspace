"""
combine_fast.py -- VALIDATION HARNESS for the accelerated biweight combine.

The implementation lives in `utils_lya_halo/extract.py` (`biweight_fused`,
`combine_fibers_fused`, `combine_fibers_threaded`), enabled by
`PipelineConfig.combine_threads` / `combine_fused`. This module imports it and
proves it against astropy. It holds no copy of its own -- a second copy would
be free to drift, and the two byte-level traps documented below are exactly
what drifts silently.

Run `validate()` and `validate_on_pool(prof)` after ANY change to the combine.

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
The 2026-07-25 profile (COSMOS, 278 galaxies, wave_block=16) measured

    6_bin_gather    78.8 s   (25%)   15.17 GB read
    7_bin_combine  216.4 s   (69%)
    TOTAL          311.8 s           -> 24.1 hr for COSMOS

so the combine is now the dominant term. It is dominated by two avoidable costs:

1. REDUNDANT MEDIANS. `combine_fibers(method="biweight")` calls
       out     = biweight_location(spec, ignore_nan=True)
       scatter = mad_std(spec, ignore_nan=True)
   `biweight_location` internally computes nanmedian(spec) and then
   nanmedian(|spec - median|). `mad_std` computes *the same two quantities
   again*. Four nan-median passes where two suffice.

   mad_std's MAD is the SAME ESTIMATOR biweight_location already computed --
   same data, same NaN policy -- so reusing it is not an approximation.

   One wrinkle, and it is the whole reason this file has a `strict` flag:
   astropy's `biweight_location` begins with `data.astype(np.float64)`, while
   `mad_std` does not. So the MAD inside the biweight is a float64 MAD and the
   MAD inside mad_std is a float32 MAD. They agree to ~1e-7 relative but are
   not the same bits. Consequences:

     default (strict=False) : FLUX is bit-identical to the current pipeline;
                              ERR differs by <~1e-7 relative (and is the more
                              accurate of the two, being the float64 MAD).
                              ~2.0x (axis=0) / ~2.8x (contiguous transposed).
     strict=True            : FLUX and ERR both bit-identical -- pays one extra
                              float32 MAD to reproduce mad_std exactly. ~1.5x.

   This matters because spec §4.1's proposal ("drop the unused Err, ~2x") is
   DEAD: `Err` IS consumed downstream. `stack.coadd_galaxies` feeds cube_err
   into `combine_fibers` for every method in `config.galaxy_combine_methods`,
   whose default includes 'inv_var' and 'mean' -- both of which REQUIRE err and
   raise/NaN without it. `stack_one_galaxy` also uses it for the per-bin line
   S/N. Fusing gets §4.1's ~2x while KEEPING Err.

2. STRIDED REDUCTIONS. With the full (n_fib, 1036) pool,
   `combine_fibers_chunked` hands `spec[:, j0:j1]` -- a strided view -- to
   np.nanmedian(axis=0). numpy's nanmedian for axis=0 moves the reduction axis
   to the end, so the scan walks memory with a 4144-byte stride: one cache line
   touched per element. Materialising each block as a contiguous TRANSPOSED
   copy (nwave_block, n_fib) and reducing along axis=1 is measurably faster
   even after paying for the copy.

Measured on the REAL gal1_bin11 pool (176,151 fibers x 1036 wave, 5.7% NaN).
Quote these, not the synthetic numbers -- synthetic Gaussians overstate the
fusion gain by ~50% because mad_std is a larger share of the cost on them:

    A  current: biweight_location + mad_std, axis=0   235.1 ms   1.00x
    C  fused, axis=0                                  185.0 ms   1.27x
    D  fused, axis=1 contiguous [+transpose]          136.0 ms   1.65x
    E  fused strict, axis=1     [+transpose]          181.5 ms   1.25x

Threading is a SEPARATE and larger effect, and is bit-identical for every
method: 4.63x on 8 threads (12 is slower than 8 -- memory bandwidth).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    import combine_fast as cf

    cf.validate()                 # bit-identity vs astropy, many edge cases
    cf.validate(strict=True)      # ... with Err bit-identical too
    cf.validate_on_pool(prof)     # the one that counts: a REAL fiber pool from
                                  # extract_profile.profile_galaxies
    cf.benchmark_threads(prof)    # thread-scaling sweep on this machine
    cf.benchmark(pool=...)        # fusion variants on this machine

Fusion is biweight-only (the configured `fiber_combine_method` and the only one
on the hot path); every other method falls through to the reference combine.
Threading applies to all of them.
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = [
    "biweight_fused", "combine_fibers_fast", "combine_fibers_chunked_fast",
    "combine_fibers_parallel",
    "validate", "validate_on_pool", "benchmark", "benchmark_threads",
]
# ---------------------------------------------------------------------
# The implementation now lives in utils_lya_halo/extract.py, next to the
# reference combine_fibers it accelerates. This module is the VALIDATION
# HARNESS for it, so it deliberately imports rather than reimplementing --
# a second copy would be free to drift, and the two byte-level traps below
# (astropy's MAD literal, and its asymmetric float64 upcast) are exactly the
# kind of thing that drifts silently.
# ---------------------------------------------------------------------
from utils_lya_halo.extract import (                       # noqa: E402
    biweight_fused,
    combine_fibers_fused as combine_fibers_fast,
    combine_fibers_threaded,
    shutdown_combine_threads,
)


def combine_fibers_chunked_fast(spec, wave_block=16, transpose=False, strict=False):
    """Serial fused combine. Thin wrapper on extract.combine_fibers_threaded."""
    return combine_fibers_threaded(
        spec, None, method="biweight", wave_block=wave_block, n_threads=1,
        fused=True, fused_strict=strict, fused_transpose=transpose, min_fibers=0)


def combine_fibers_parallel(spec, wave_block=16, n_threads=None, transpose=False,
                            strict=False, executor=None, fused=True):
    """Threaded combine. Thin wrapper on extract.combine_fibers_threaded."""
    if n_threads is None:
        n_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
            else (os.cpu_count() or 1)
    return combine_fibers_threaded(
        spec, None, method="biweight", wave_block=wave_block, n_threads=n_threads,
        fused=fused, fused_strict=strict, fused_transpose=transpose,
        min_fibers=0, executor=executor)


# =====================================================================
# validation -- run this BEFORE trusting anything above
# =====================================================================
def _reference(spec):
    """The current production path, verbatim."""
    from astropy.stats import biweight_location as biweight, mad_std
    n_eff = np.sum(np.isfinite(spec), axis=0).astype(float)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore",
                                                over="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        out = biweight(spec, axis=0, ignore_nan=True)
        scatter = mad_std(spec, axis=0, ignore_nan=True)
        out_err = scatter / np.sqrt(np.maximum(n_eff, 1))
    bad = n_eff == 0
    out = np.asarray(out, dtype=float).copy()
    out_err = np.asarray(out_err, dtype=float).copy()
    out[bad] = np.nan
    out_err[bad] = np.nan
    return out, out_err


def _compare(name, ref, got, flux_rtol=0.0, err_rtol=0.0, quiet=False):
    """
    Identity check with NaN==NaN, per the spec's correctness protocol §6.
    Tolerances are RELATIVE to the peak |value| of the reference; 0.0 means
    exact bit-for-bit equality.
    """
    (rf, re_), (gf, ge) = ref, got
    ok = True
    for label, a, b, tol in (("flux", rf, gf, flux_rtol),
                             ("err", re_, ge, err_rtol)):
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if not np.array_equal(nan_a, nan_b):
            print(f"  FAIL {name:30s} {label}: NaN pattern differs "
                  f"({nan_a.sum()} vs {nan_b.sum()})")
            ok = False
            continue
        av, bv = a[~nan_a], b[~nan_b]
        d = np.abs(av - bv)
        mx = float(d.max()) if d.size else 0.0
        scale = float(np.abs(av).max()) if d.size else 1.0
        rel = mx / max(scale, 1e-300)
        bad = (mx > 0.0) if tol == 0.0 else (rel > tol)
        if bad:
            ok = False
        if bad or not quiet:
            print(f"  {'FAIL' if bad else 'OK  '} {name:30s} {label}: "
                  f"max|diff| = {mx:.3e}  (rel {rel:.2e})")
    return ok


def validate(seed=0, strict=False, err_rtol=None, verbose=True):
    """
    Identity of the fused path against astropy, across the shapes and NaN
    patterns the pipeline actually produces: no NaNs, scattered NaNs, whole
    all-NaN columns, tiny pools, a degenerate zero-MAD column, a large-offset
    /small-scatter column, and big-endian input (the real file is '>f4').

    FLUX is required to be EXACTLY equal on the axis=0 paths, and within
    flux_rtol (1e-12) on the transposed paths, which reassociate the nansum.
    ERR is required to be exact when strict=True, and within err_rtol
    (default 1e-6) otherwise. See the module docstring for why.
    """
    if err_rtol is None:
        err_rtol = 0.0 if strict else 1e-6
    flux_rtol_T = 1e-12
    rng = np.random.default_rng(seed)
    cases = {}

    a = rng.normal(0, 1, (5000, 16)).astype(np.float32)
    cases["clean"] = a

    b = a.copy()
    b.ravel()[rng.choice(b.size, b.size // 20, replace=False)] = np.nan
    cases["3-5% scattered NaN"] = b

    c = b.copy()
    c[:, 3] = np.nan                      # a fully empty wavelength column
    cases["one all-NaN column"] = c

    d = rng.normal(0, 1, (7, 16)).astype(np.float32)
    cases["tiny pool (n=7)"] = d

    e = a.copy()
    e[:, 5] = 2.5                          # constant column -> MAD == 0
    cases["zero-MAD column"] = e

    f = (rng.normal(0, 1e-3, (2000, 16)) + 1e5).astype(np.float32)
    cases["large offset, small scatter"] = f

    g = rng.normal(0, 1, (3000, 16)).astype(np.float32).astype(">f4")
    cases["big-endian input (>f4)"] = g

    all_ok = True
    print("=" * 74)
    print(f"VALIDATE  fused vs astropy   strict={strict}")
    print(f"  flux: exact on axis=0, rtol {flux_rtol_T:g} on transposed")
    print(f"  err : {'exact' if err_rtol == 0 else f'rtol {err_rtol:g}'}")
    print("=" * 74)
    for name, arr in cases.items():
        ref = _reference(arr)
        all_ok &= _compare(name + " [axis0]", ref,
                           combine_fibers_fast(arr, axis=0, strict=strict),
                           0.0, err_rtol, quiet=not verbose)
        arrT = np.ascontiguousarray(arr.T)
        all_ok &= _compare(name + " [axis1/T]", ref,
                           combine_fibers_fast(arrT, axis=1, strict=strict),
                           flux_rtol_T, err_rtol, quiet=not verbose)

    # chunked path must equal the single-shot path for ANY block size
    print("-" * 74)
    big = rng.normal(0, 1, (4000, 1036)).astype(np.float32)
    big.ravel()[rng.choice(big.size, big.size // 30, replace=False)] = np.nan
    ref_big = _reference(big)
    for wb in (16, 64, 256, None):
        for tr in (False, True):
            got = combine_fibers_chunked_fast(big, wave_block=wb, transpose=tr,
                                              strict=strict)
            all_ok &= _compare(f"chunked wb={wb} transpose={tr}", ref_big, got,
                               flux_rtol_T if tr else 0.0, err_rtol,
                               quiet=not verbose)

    print("-" * 74)
    if all_ok:
        print("  PASS -- flux bit-identical on the default (transpose=False) path"
              + (", err bit-identical" if strict else f"; err within {err_rtol:g} rel"))
    else:
        print("  >> MISMATCHES ABOVE -- do not adopt")
    print()
    return all_ok


def validate_on_pool(profile_result=None, pool=None, wave_block=16,
                     strict=False, err_rtol=None):
    """
    The validation that actually counts: bit-identity on a REAL fiber pool.

    Pass the dict returned by `extract_profile.profile_galaxies(...)`; it keeps
    a fiber-subsampled copy of the largest bin in `biggest_pool`.
    """
    if err_rtol is None:
        err_rtol = 0.0 if strict else 1e-6
    # The transposed path reassociates the nansum, so it is held to the SAME
    # relative tolerance validate() uses (1e-12), not to exact equality. Holding
    # it to 0.0 here was a validator bug: it reported FAIL on a 1e-15 relative
    # difference that validate() correctly passes.
    flux_rtol_T = 1e-12
    if pool is None and profile_result is not None:
        pool = profile_result.get("biggest_pool", {}).get("pool")
    if pool is None:
        raise ValueError("no pool available -- run profile_galaxies with "
                         "keep_pool_for_bench=True, or pass pool= directly")
    label = (profile_result or {}).get("biggest_pool", {}).get("label", "pool")
    print("=" * 74)
    print(f"VALIDATE ON REAL POOL   {pool.shape[0]:,} fibers x {pool.shape[1]} wave"
          f"   ({label})")
    print("=" * 74)
    print(f"  nan fraction = {np.isnan(pool).mean():.3f}   dtype = {pool.dtype}")
    print(f"  flux: exact on contig (default), rtol {flux_rtol_T:g} on transposed")
    print(f"  err : {'exact' if err_rtol == 0 else f'rtol {err_rtol:g}'}")
    print("-" * 74)
    ref = _reference(pool)
    ok = _compare("chunked fast (contig, DEFAULT)", ref,
                  combine_fibers_chunked_fast(pool, wave_block, transpose=False,
                                              strict=strict), 0.0, err_rtol)
    ok &= _compare("chunked fast (transpose, opt-in)", ref,
                   combine_fibers_chunked_fast(pool, wave_block, transpose=True,
                                               strict=strict), flux_rtol_T, err_rtol)
    print("-" * 74)
    print("  PASS on real data" if ok else "  >> MISMATCH -- do not adopt")
    print()
    return ok


# =====================================================================
# benchmark
# =====================================================================
def benchmark(nfib=200_000, wave_block=16, nan_frac=0.03, reps=3, pool=None):
    """
    Time the four paths on one wavelength block, then extrapolate to a full
    1036-wave bin. Pass `pool` (e.g. from profile_galaxies) to use real data.
    """
    from astropy.stats import biweight_location as biweight, mad_std

    if pool is not None:
        a = np.ascontiguousarray(np.asarray(pool)[:, :wave_block], dtype=np.float32)
        nfib = a.shape[0]
        src = "real pool"
    else:
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (nfib, wave_block)).astype(np.float32)
        if nan_frac:
            a.ravel()[rng.choice(a.size, int(a.size * nan_frac), replace=False)] = np.nan
        src = f"synthetic, {100*nan_frac:.0f}% NaN"
    aT = np.ascontiguousarray(a.T)

    def _t(fn):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            fn(); gc.collect()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps

    def _cur():
        biweight(a, axis=0, ignore_nan=True)
        mad_std(a, axis=0, ignore_nan=True)

    t_cur = _t(_cur)
    t_bi = _t(lambda: biweight(a, axis=0, ignore_nan=True))
    t_f0 = _t(lambda: biweight_fused(a, axis=0))
    t_f1 = _t(lambda: biweight_fused(aT, axis=1))
    t_s1 = _t(lambda: biweight_fused(aT, axis=1, strict=True))
    t_tr = _t(lambda: np.ascontiguousarray(a.T))

    nblk = 1036 / wave_block
    print("=" * 74)
    print(f"COMBINE BENCHMARK  {nfib:,} fibers x {wave_block} wave   ({src})")
    print("=" * 74)
    rows = [
        ("A  current: biweight_location + mad_std, axis=0", t_cur, t_cur),
        ("   (biweight_location alone, for reference)", t_bi, t_bi),
        ("C  fused, axis=0 on (n,w)", t_f0, t_f0),
        ("D  fused, axis=1 on (w,n) contiguous  [+transpose]", t_f1, t_f1 + t_tr),
        ("E  fused strict, axis=1  [+transpose]", t_s1, t_s1 + t_tr),
    ]
    print(f"  {'path':52s} {'ms/blk':>8s} {'speedup':>8s} {'s/bin':>7s}")
    for name, t, t_all in rows:
        print(f"  {name:52s} {t*1e3:8.1f} {t_cur/max(t_all,1e-12):7.2f}x "
              f"{t_all*nblk:7.1f}")
    print(f"\n  transpose copy: {t_tr*1e3:.1f} ms/blk "
          f"({100*t_tr/max(t_f1,1e-12):.0f}% of D's reduction -- still a net win)")
    best_t = min(t_f0, t_f1 + t_tr)
    print(f"  best non-strict = {'D' if (t_f1+t_tr) < t_f0 else 'C'}: "
          f"{t_cur/best_t:.2f}x on the stage that is 69% of extraction runtime")
    print(f"  -> 216 s/galaxy combine becomes ~{216/(t_cur/best_t):.0f} s "
          f"(~{216 - 216/(t_cur/best_t):.0f} s/galaxy saved, "
          f"{278*(216 - 216/(t_cur/best_t))/3600:.1f} hr over COSMOS)")
    print()
    return dict(current=t_cur, biweight_only=t_bi, fused_axis0=t_f0,
                fused_axis1=t_f1, fused_strict=t_s1, transpose=t_tr,
                nfib=nfib, wave_block=wave_block)


def benchmark_threads(profile_result=None, pool=None, wave_block=16,
                      thread_counts=(1, 2, 4, 6, 8, 12), transpose=False,
                      strict=False, reps=1, verify=True):
    """
    Sweep thread counts on a REAL fiber pool and report scaling efficiency.

    This is the measurement that decides the whole optimisation plan: the
    combine is 69% of extraction runtime and threading is the only lever that
    scales past ~1.7x. Efficiency will fall off well before it hits the core
    count -- nan-median is memory-latency bound, not compute bound -- so the
    useful output is where the curve flattens, not the peak speedup.

    verify=True also asserts every thread count returns bit-identical results.
    """
    if pool is None and profile_result is not None:
        pool = profile_result.get("biggest_pool", {}).get("pool")
    if pool is None:
        raise ValueError("no pool available -- pass pool= or a profile_galaxies result")
    pool = np.asarray(pool)
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 1)

    def _t(fn):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            gc.collect()
            t0 = time.perf_counter()
            for _ in range(reps):
                r = fn()
            return (time.perf_counter() - t0) / reps, r

    print("=" * 74)
    print(f"THREAD SCALING   {pool.shape[0]:,} fibers x {pool.shape[1]} wave, "
          f"wave_block={wave_block}")
    print(f"  machine reports {ncpu} usable cores; transpose={transpose} strict={strict}")
    print("=" * 74)
    print(f"  {'threads':>8s} {'seconds':>9s} {'speedup':>8s} {'efficiency':>11s} "
          f"{'identical':>10s}")
    base = ref = None
    rows = {}
    for nt in thread_counts:
        if nt > 2 * ncpu:
            continue
        t, res = _t(lambda: combine_fibers_parallel(
            pool, wave_block=wave_block, n_threads=nt,
            transpose=transpose, strict=strict))
        if base is None:
            base, ref = t, res
        same = "-"
        if verify and ref is not None:
            f_ok = np.array_equal(np.isnan(ref[0]), np.isnan(res[0])) and \
                np.array_equal(ref[0][~np.isnan(ref[0])], res[0][~np.isnan(res[0])])
            e_ok = np.array_equal(np.isnan(ref[1]), np.isnan(res[1])) and \
                np.array_equal(ref[1][~np.isnan(ref[1])], res[1][~np.isnan(res[1])])
            same = "yes" if (f_ok and e_ok) else "NO <<<"
        rows[nt] = t
        print(f"  {nt:8d} {t:9.2f} {base/max(t,1e-9):7.2f}x "
              f"{100*base/max(t,1e-9)/nt:10.0f}% {same:>10s}")

    best = min(rows, key=rows.get)
    sp = base / rows[best]
    print(f"\n  best = {best} threads, {sp:.2f}x on the combine")
    print(f"  -> 216 s/galaxy combine becomes ~{216/sp:.0f} s")
    tot = 311.8 - 216.4 + 216.4 / sp
    print(f"  -> {tot:.0f} s/galaxy total, {278*tot/3600:.1f} hr for COSMOS's 278 "
          f"galaxies (gather unchanged at 78.8 s)")
    print("     NOTE: past this point GATHER is the bottleneck, not the combine.")
    print()
    return rows


if __name__ == "__main__":
    validate()
    benchmark(nfib=50_000)
