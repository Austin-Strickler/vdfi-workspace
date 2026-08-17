"""
extract_profile.py -- non-invasive profiler for Stage-1 `extract.extract_fibers`.

Nothing in utils_lya_halo is modified or imported-and-patched. This module
REPLICATES the inner loop of extract_fibers with instrumentation around each
stage, calls the SAME helper functions the real pipeline calls, and throws the
results away. It never writes to the per-galaxy cache and never reads from it
(reading the cache would measure nothing).

The question it exists to answer: is the 13-19 hr extraction I/O-bound
(page-cache thrashing on a 20-60 GB memmapped flux file, read ~once per galaxy
because the galaxy loop is outer and the memmap read is inner) or CPU-bound
(the robust combine over 500k-fiber outer bins)? Those need completely
different fixes, so measure before optimizing.

USAGE -- the whole thing in two lines:

    import extract_profile as ep
    res = ep.run_all(cfg_profile, n_gal=3)

Or step by step, reusing one setup across many calls:

    s    = ep.setup(cfg_profile)                          # build vdfi/run_table/mask once
    ep.environment_report(cfg_profile, s["vdfi_data"])    # file size vs RAM
    prof = ep.profile_galaxies(cfg_profile, s["vdfi_data"], s["run_table"],
                               s["bad_fiber_mask"], n_gal=3)
    warm = ep.warm_cache_test(cfg_profile, s["vdfi_data"], s["run_table"],
                              s["bad_fiber_mask"])        # the decisive I/O test
    bench = ep.benchmark_combine(prof)
    ep.print_report(prof, warm, bench)

Every function is read-only and safe to interrupt. Start with n_gal=2-3; a
single galaxy can take minutes if the file is cold.

Requires: psutil (pip install psutil). Degrades gracefully without it -- you
lose the I/O counters, which are the most useful numbers here, so install it.
"""

from __future__ import annotations

import os
import gc
import time
import resource
import warnings

import numpy as np
import astropy.units as u
from astropy.stats import biweight_location as biweight, mad_std

from utils_lya_halo import extract as _extract
from utils_lya_halo.extract import (
    radial_bin_edges, combine_fibers, combine_fibers_chunked,
)
from utils_lya_halo.background import (
    smooth_interp_then_gauss, smooth_spectrum_nan_safe_adaptive,
)
from utils_lya_halo.virial import cosmo

try:
    import psutil
    _PROC = psutil.Process()
except Exception:                                                # noqa: BLE001
    psutil = None
    _PROC = None

__all__ = [
    "setup", "run_all",
    "environment_report", "profile_galaxies", "warm_cache_test",
    "benchmark_combine", "print_profile", "print_warm", "print_benchmark",
    "print_report",
]

_GB = 1024.0 ** 3


# =====================================================================
# 0. setup -- build exactly what pipeline.run_extract builds
# =====================================================================
def setup(config, table=None, verbose=True):
    """
    Build (vdfi_data, run_table, bad_fiber_mask) the same way
    pipeline.run_extract does, with each step timed.

    This is the only expensive-ish setup step (mask building reads the
    segmentation image). Do it ONCE per notebook session and reuse the
    returned objects for every profiling call.

    Returns dict with keys 'vdfi_data', 'run_table', 'bad_fiber_mask', 'timing'.
    """
    if config.normalized_field() == "BOTH":
        raise ValueError("profile one field at a time ('AEGIS' or 'COSMOS'), not 'BOTH'.")

    timing = {}
    t0 = time.perf_counter()
    vdfi_data = _extract.load_field(config)
    timing["load_field"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_table = _extract.build_run_table(config, table=table)
    timing["build_run_table"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bad_fiber_mask = _extract.build_bad_fiber_mask(config, vdfi_data, run_table=run_table)
    timing["build_bad_fiber_mask"] = time.perf_counter() - t0

    if verbose:
        print("SETUP")
        for k, v in timing.items():
            print(f"  {k:24s} {v:8.2f} s")
        print(f"  run_table                {len(run_table)} galaxies")
        print(f"  bad_fiber_mask           {bad_fiber_mask.shape}, "
              f"{100*bad_fiber_mask.mean():.1f}% masked")
        print()
    return dict(vdfi_data=vdfi_data, run_table=run_table,
                bad_fiber_mask=bad_fiber_mask, timing=timing)


def run_all(config, table=None, *, n_gal=3, max_pool_gb=4.0, do_warm=True,
            do_bench=True, block_gather=False, s=None):
    """
    One call: setup -> environment -> profile -> warm-cache test -> combine
    benchmark -> verdict. Returns a dict of every intermediate result.

    s : an existing setup() result to reuse (skips rebuilding the mask).
    """
    if s is None:
        s = setup(config, table=table)
    vd, rt, bm = s["vdfi_data"], s["run_table"], s["bad_fiber_mask"]

    env = environment_report(config, vd)

    print("PROFILING GALAXIES (cache bypassed; this is a cold run)")
    prof = profile_galaxies(config, vd, rt, bm, n_gal=n_gal,
                            max_pool_gb=max_pool_gb, block_gather=block_gather,
                            verbose=True)
    print()

    warm = None
    if do_warm:
        print("WARM-CACHE TEST (same galaxy twice)")
        gc.collect()
        warm = warm_cache_test(config, vd, rt, bm, gal_index=0,
                               max_pool_gb=max_pool_gb, verbose=False)

    bench = None
    if do_bench:
        bench = benchmark_combine(prof, verbose=False)

    print_report(prof, warm, bench, setup_timing=s["timing"])
    return dict(setup=s, env=env, profile=prof, warm=warm, bench=bench)


# =====================================================================
# counters
# =====================================================================
def _io_read_bytes():
    """Bytes actually pulled from the block layer by this process (Linux)."""
    if _PROC is None:
        return np.nan
    try:
        return float(_PROC.io_counters().read_bytes)
    except Exception:                                            # noqa: BLE001
        return np.nan


def _majflt():
    """Major page faults -- the direct measure of memmap page-cache MISSES.
    Every one of these is a trip to disk. This is the cleanest thrashing signal."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_majflt)


def _rss_gb():
    if _PROC is None:
        return np.nan
    try:
        return _PROC.memory_info().rss / _GB
    except Exception:                                            # noqa: BLE001
        return np.nan


class _Stage:
    """Accumulates wall time, bytes read, and major faults per named stage."""

    def __init__(self):
        self.t = {}
        self.io = {}
        self.mf = {}

    def __call__(self, name):
        return _StageCtx(self, name)

    def add(self, name, dt, dio, dmf):
        self.t[name] = self.t.get(name, 0.0) + dt
        self.io[name] = self.io.get(name, 0.0) + dio
        self.mf[name] = self.mf.get(name, 0.0) + dmf


class _StageCtx:
    def __init__(self, parent, name):
        self.p, self.name = parent, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.io0 = _io_read_bytes()
        self.mf0 = _majflt()
        return self

    def __exit__(self, *exc):
        self.p.add(self.name,
                   time.perf_counter() - self.t0,
                   _io_read_bytes() - self.io0,
                   _majflt() - self.mf0)
        return False


# =====================================================================
# 1. environment
# =====================================================================
def environment_report(config, vdfi_data, verbose=True):
    """File sizes vs. available RAM, and the per-galaxy read estimate.

    If the flux file is larger than RAM, the page cache cannot hold it and the
    outer-galaxy / inner-read loop structure guarantees repeated re-reads. That
    single comparison usually settles the I/O-vs-CPU question before you run
    anything else."""
    base = config.resolve_vdfi_base_path()
    files = {}
    for fn in ("all_flux_final.fits", "all_error.fits", "all_info.fits",
               "all_initial_dar.fits"):
        p = os.path.join(base, fn)
        files[fn] = os.path.getsize(p) / _GB if os.path.exists(p) else np.nan

    RA = vdfi_data["RA"]
    nexp, nfib = RA.shape
    nwave = vdfi_data["spectra"].shape[-1]
    dtype = np.asarray(vdfi_data["spectra"]).dtype

    if psutil is not None:
        vm = psutil.virtual_memory()
        ram_total, ram_avail = vm.total / _GB, vm.available / _GB
    else:
        ram_total = ram_avail = np.nan

    # The node's RAM is NOT what this process gets. JupyterHub sessions are
    # cgroup-limited, and the cgroup limit -- not psutil's node-level
    # "available" -- is what decides whether the page cache can hold the file.
    cg_max = cg_cur = np.nan
    for p_max, p_cur in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(p_max) as f:
                v = f.read().strip()
            cg_max = np.inf if v == "max" else float(v) / _GB
            with open(p_cur) as f:
                cg_cur = float(f.read().strip()) / _GB
            break
        except Exception:                                        # noqa: BLE001
            continue
    # page cache counts against the cgroup, so the budget is limit - current
    cg_budget = (cg_max - cg_cur) if np.isfinite(cg_max) else np.nan
    effective = np.nanmin([x for x in (ram_avail, cg_budget) if np.isfinite(x)] or [np.nan])

    native = dtype.byteorder in ("=", "|") or (
        dtype.byteorder == "<" and np.little_endian) or (
        dtype.byteorder == ">" and not np.little_endian)

    out = dict(base_path=base, files=files, nexp=nexp, nfib=nfib, nwave=nwave,
               dtype=str(dtype), native_byteorder=bool(native),
               ram_total_gb=ram_total, ram_avail_gb=ram_avail,
               cgroup_limit_gb=cg_max, cgroup_current_gb=cg_cur,
               cgroup_budget_gb=cg_budget, effective_budget_gb=effective,
               flux_gb=files.get("all_flux_final.fits", np.nan),
               array_gb=nexp * nfib * nwave * dtype.itemsize / _GB,
               bytes_per_fiber_spectrum=nwave * dtype.itemsize,
               waveblock_gb_64=nexp * nfib * 64 * dtype.itemsize / _GB)

    if verbose:
        print("=" * 74)
        print("EXTRACTION ENVIRONMENT")
        print("=" * 74)
        print(f"  base path        : {base}")
        for k, v in files.items():
            print(f"  {k:24s} : {v:8.2f} GB" if np.isfinite(v)
                  else f"  {k:24s} :   (missing)")
        print(f"  array shape      : ({nexp} exp, {nfib} fib, {nwave} wave)  {dtype}"
              + ("" if native else "   <-- NON-NATIVE BYTE ORDER"))
        print(f"  array in RAM     : {out['array_gb']:.2f} GB as-is")
        print(f"  node RAM tot/avail : {ram_total:.1f} / {ram_avail:.1f} GB")
        if np.isfinite(cg_max):
            print(f"  cgroup limit/used  : {cg_max:.1f} / {cg_cur:.1f} GB "
                  f"-> budget {cg_budget:.1f} GB")
            print("     (page cache counts against the cgroup, so THIS is the number")
            print("      that decides whether the file stays cached, not node RAM)")
        elif cg_max == np.inf:
            print("  cgroup limit       : unlimited")
        print(f"  one fiber spectrum   = {out['bytes_per_fiber_spectrum']/1024:.1f} KB "
              f"(~{out['bytes_per_fiber_spectrum']/4096:.2f} pages)")
        print(f"  one 64-wave block    = {out['waveblock_gb_64']:.2f} GB "
              f"(all exposures, all fibers)")

        if not native:
            print()
            print("  >> The fiber array is NON-NATIVE byte order. Every read of")
            print("     spectra[...] must be byte-swapped to native float32 before")
            print("     arithmetic, allocating a temporary each time. Converting the")
            print("     array ONCE to native order removes that cost from every access.")

        f = out["flux_gb"]
        budget = effective
        if np.isfinite(f) and np.isfinite(budget):
            print()
            if f > budget:
                print(f"  >> flux file ({f:.1f} GB) EXCEEDS the effective budget ({budget:.1f} GB).")
                print("     The page cache cannot hold it. With the galaxy loop outer and")
                print("     the memmap read inner, expect repeated re-reads per galaxy.")
            else:
                print(f"  >> flux file ({f:.1f} GB) fits the effective budget ({budget:.1f} GB)")
                print("     ON PAPER. But the process's own pools compete with the page")
                print("     cache for that same budget, so this is necessary, not")
                print("     sufficient. The per-galaxy 'GB read' number is the real test:")
                print("     if it approaches the file size, the cache is being evicted.")
        print()
    return out


# =====================================================================
# 2. instrumented replica of the extract_fibers inner loop
# =====================================================================
def _locality(sel):
    """How contiguous is a boolean fiber selection? Scattered selections defeat
    readahead and force ~one page fault per fiber; contiguous IFU-ordered runs
    do not. Returns (n_selected, n_runs, index_span, fill_fraction_of_span)."""
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return 0, 0, 0, np.nan
    runs = 1 + int(np.count_nonzero(np.diff(idx) > 1))
    span = int(idx[-1] - idx[0] + 1)
    return int(idx.size), runs, span, idx.size / max(span, 1)


def profile_galaxies(config, vdfi_data, run_table, bad_fiber_mask, *,
                     n_gal=3, gal_indices=None, wave_block=None,
                     max_pool_gb=4.0, keep_pool_for_bench=True,
                     bench_pool_gb=0.8, block_gather=False, verbose=True):
    """
    Replicate extract_fibers' per-galaxy work with per-stage instrumentation.

    Cache is bypassed entirely (neither read nor written), so this measures the
    real cost of a cold galaxy. Results are discarded.

    n_gal / gal_indices : how many (or exactly which) run_table rows to profile.
    wave_block          : override config.combine_wave_block for the combine.
    max_pool_gb         : skip any radial bin whose fiber pool would exceed this
                          (protects the session from the largest outer bin).
                          Skipped bins are reported, not silently dropped.
    keep_pool_for_bench : retain a FIBER-SUBSAMPLED copy of the largest pool so
                          benchmark_combine can run on real data.
    bench_pool_gb       : hard cap on that retained copy (default 0.8 GB). It is
                          subsampled along the fiber axis to fit; the true fiber
                          count is recorded so timings can be scaled up.
    block_gather        : if True, gather and combine ONE WAVELENGTH BLOCK AT A
                          TIME instead of allocating the full (nfib, nwave) pool.
                          Peak pool memory drops by nwave/wave_block (~65x at
                          wave_block=16). This is the proposed restructuring --
                          set True to measure it directly against False.
    """
    wave = vdfi_data["wave"]
    RA, DEC = vdfi_data["RA"], vdfi_data["DEC"]
    spectra, error = vdfi_data["spectra"], vdfi_data["error"]
    nexp, nfib_total = RA.shape
    nwave = spectra.shape[-1]
    itemsize = np.asarray(spectra).dtype.itemsize

    fiber_method = config.fiber_combine_method.lower()
    bg_method = config.resolved_bg_combine_method().lower()
    bin_mode = config.bin_mode.lower()
    nrad = len(np.asarray(config.bins, dtype=float)) - 1
    needs_err = fiber_method in ("mean", "inv_var")
    bg_needs_err = bg_method in ("mean", "inv_var")
    sc_sigma = float(getattr(config, "sigma_clip_sigma", 3.0))
    sc_maxiters = getattr(config, "sigma_clip_maxiters", 5)
    wb = wave_block if wave_block is not None else getattr(config, "combine_wave_block", 64)

    if gal_indices is None:
        gal_indices = list(range(min(n_gal, len(run_table))))

    per_gal = []
    biggest = {"n": 0, "pool": None, "label": None}

    for cnt in gal_indices:
        S = _Stage()
        gc.collect()
        t_gal0 = time.perf_counter()
        io_gal0, mf_gal0 = _io_read_bytes(), _majflt()
        rss0 = _rss_gb()

        ra = run_table["RA"][cnt]
        dec = run_table["DEC"][cnt]
        z = float(run_table[config.z_col][cnt])
        m = float(run_table[config.mass_col][cnt])

        # ---- stage: geometry ----
        with S("1_geometry"):
            kpc_per_arcsec = cosmo.angular_diameter_distance(z).to(u.kpc).value / 206265
            radial_bins = radial_bin_edges(config, m, z)
            dist_all = np.full((nexp, nfib_total), np.nan)
            good_all = np.zeros((nexp, nfib_total), dtype=bool)
            for e in range(nexp):
                dra = (RA[e] - ra) * np.cos(np.deg2rad((DEC[e] + dec) / 2)) * 3600.
                ddec = (DEC[e] - dec) * 3600.
                dist_all[e] = np.sqrt(dra ** 2 + ddec ** 2)
                good_all[e] = (~bad_fiber_mask[e]) & np.isfinite(RA[e]) & np.isfinite(DEC[e])

        # ---- stage: background read (the memmap hit) ----
        bg_fibers = []
        n_bg_tot = 0
        with S("2_bg_read"):
            for e in range(nexp):
                bg_sel = ((dist_all[e] >= config.bg_inner_arcsec) &
                          (dist_all[e] < config.bg_outer_arcsec) & good_all[e])
                k = int(np.count_nonzero(bg_sel))
                if k == 0:
                    continue
                n_bg_tot += k
                bg_fibers.append(np.asarray(spectra[e, bg_sel]))

        with S("3_bg_combine"):
            if bg_fibers:
                sp_all = np.concatenate(bg_fibers, axis=0)
                bg_raw, _ = combine_fibers(sp_all, None if not bg_needs_err else None,
                                           method=bg_method, sigma=sc_sigma,
                                           maxiters=sc_maxiters)
                del sp_all
            else:
                bg_raw = np.full(nwave, np.nan)
        del bg_fibers

        with S("4_bg_smooth"):
            if getattr(config, "smooth_bg", True):
                meth = getattr(config, "bg_smooth_method", "interp_gauss").lower()
                if meth == "interp_gauss":
                    bg = smooth_interp_then_gauss(
                        bg_raw, wave, z, lya_rest=config.LYA_REST,
                        sigma_A=config.bg_smooth_sigma_A,
                        mask_halfwidth_A=config.line_mask_halfwidth_A)
                else:
                    sv = config.smoothing_values
                    bg = smooth_spectrum_nan_safe_adaptive(
                        bg_raw, wave, sigma_A=sv[0], edge_sigma_A=sv[1],
                        edge_width_A=sv[2])
            else:
                bg = bg_raw
        bg32 = np.asarray(bg, dtype=np.float32)

        # ---- stage: digitize ----
        with S("5_digitize"):
            bin_idx = np.full((nexp, nfib_total), -1, dtype=np.int16)
            for e in range(nexp):
                d = dist_all[e] if bin_mode == "arcsec" else dist_all[e] * kpc_per_arcsec
                idx = np.digitize(d, radial_bins) - 1
                idx[~good_all[e]] = -1
                idx[(idx < 0) | (idx >= nrad)] = -1
                bin_idx[e] = idx
            nfib_bin = np.zeros(nrad, dtype=np.int64)
            for e in range(nexp):
                v = bin_idx[e]
                v = v[v >= 0]
                if v.size:
                    nfib_bin += np.bincount(v, minlength=nrad)

        # ---- locality (free; explains the I/O pattern) ----
        touched = np.zeros(nfib_total, dtype=bool)
        for e in range(nexp):
            touched |= bin_idx[e] >= 0
        n_touch, n_runs, span, fill = _locality(touched)

        # ---- stage: per-bin gather + combine ----
        skipped, bin_rows = [], []
        for r in range(nrad):
            n_r = int(nfib_bin[r])
            if n_r == 0:
                continue
            blk = wb if (block_gather and wb) else nwave
            pool_gb = n_r * nwave * 4 / _GB
            peak_gb = n_r * blk * 4 / _GB          # what is actually allocated
            if peak_gb > max_pool_gb:
                skipped.append((r, n_r, peak_gb))
                continue

            t_gather = t_comb = 0.0
            io_gather = mf_gather = 0.0
            sel_cache = None
            for j0 in range(0, nwave, blk):
                j1 = min(j0 + blk, nwave)

                t0 = time.perf_counter(); io0, mf0 = _io_read_bytes(), _majflt()
                pool = np.empty((n_r, j1 - j0), dtype=np.float32)
                if sel_cache is None:
                    sel_cache = [(e, bin_idx[e] == r) for e in range(nexp)]
                    sel_cache = [(e, s) for e, s in sel_cache if s.any()]
                off = 0
                for e, sel in sel_cache:
                    k = int(np.count_nonzero(sel))
                    # NOTE: spectra is big-endian ('>f4'); this subtraction is
                    # where the byte-swap to native float32 happens.
                    pool[off:off + k] = spectra[e, sel, j0:j1] - bg32[np.newaxis, j0:j1]
                    off += k
                t_gather += time.perf_counter() - t0
                io_gather += _io_read_bytes() - io0
                mf_gather += _majflt() - mf0

                t0 = time.perf_counter()
                combine_fibers_chunked(pool, None, method=fiber_method,
                                       sigma=sc_sigma, maxiters=sc_maxiters,
                                       wave_block=(None if blk <= wb else wb))
                t_comb += time.perf_counter() - t0

                if (keep_pool_for_bench and n_r > biggest["n"]
                        and j0 == 0 and (j1 - j0) >= min(64, nwave)):
                    # retain a FIBER-SUBSAMPLED slice only -- retaining the full
                    # pool here is what OOM-killed an earlier version.
                    step = max(1, int(np.ceil(n_r * (j1 - j0) * 4 / (bench_pool_gb * _GB))))
                    biggest = {"n": n_r, "pool": np.ascontiguousarray(pool[::step]),
                               "label": f"gal{cnt}_bin{r} (subsampled 1/{step})",
                               "true_nfib": n_r, "subsample_step": step}
                del pool

            S.add("6_bin_gather", t_gather, io_gather, mf_gather)
            S.add("7_bin_combine", t_comb, 0.0, 0.0)
            del sel_cache

            bin_rows.append(dict(bin=r, nfib=n_r, pool_gb=pool_gb, peak_gb=peak_gb,
                                 t_gather=t_gather, t_combine=t_comb,
                                 io_gather_gb=io_gather / _GB,
                                 majflt_gather=mf_gather))
            gc.collect()

        del bin_idx, dist_all, good_all
        gc.collect()

        per_gal.append(dict(
            index=int(cnt), z=z, logM=m, nrad=nrad,
            nfib_bin=nfib_bin.copy(), n_bg_fibers=n_bg_tot,
            stages_t=dict(S.t), stages_io_gb={k: v / _GB for k, v in S.io.items()},
            stages_majflt=dict(S.mf),
            total_t=time.perf_counter() - t_gal0,
            total_io_gb=(_io_read_bytes() - io_gal0) / _GB,
            total_majflt=_majflt() - mf_gal0,
            rss_start_gb=rss0, rss_end_gb=_rss_gb(),
            locality=dict(n_touched=n_touch, n_runs=n_runs, span=span,
                          fill_fraction=fill, nfib_total=nfib_total),
            bins=bin_rows, skipped_bins=skipped,
        ))
        if verbose:
            g = per_gal[-1]
            print(f"  galaxy {cnt}: {g['total_t']:7.1f} s   "
                  f"read {g['total_io_gb']:6.2f} GB   "
                  f"majflt {g['total_majflt']:,.0f}"
                  + (f"   [{len(skipped)} bin(s) skipped, >{max_pool_gb} GB]"
                     if skipped else ""))

    return dict(per_gal=per_gal, wave_block=wb, fiber_method=fiber_method,
                nexp=nexp, nfib_total=nfib_total, nwave=nwave,
                itemsize=itemsize, biggest_pool=biggest,
                env=environment_report(config, vdfi_data, verbose=False))


# =====================================================================
# 3. the decisive I/O test
# =====================================================================
def warm_cache_test(config, vdfi_data, run_table, bad_fiber_mask, *,
                    gal_index=0, max_pool_gb=6.0, verbose=True):
    """
    Run the SAME galaxy twice back to back and compare.

    Pass 1 is cold (or as cold as the page cache currently is). Pass 2 re-reads
    exactly the same fibers, which are now in the page cache if it can hold
    them. Interpretation:

      pass2 ~ pass1              -> reads are NOT being cached (file >> RAM):
                                    I/O-bound, and the loop-inversion fix applies.
      pass2 << pass1             -> the cache holds this galaxy's working set.
                                    The real run is still slower than this test
                                    because consecutive galaxies evict each
                                    other, so check total_io_gb across several
                                    DIFFERENT galaxies in profile_galaxies too.
      both fast, combine dominant-> CPU-bound.
    """
    gc.collect()
    r1 = profile_galaxies(config, vdfi_data, run_table, bad_fiber_mask,
                          gal_indices=[gal_index], max_pool_gb=max_pool_gb,
                          keep_pool_for_bench=False, verbose=False)
    gc.collect()
    r2 = profile_galaxies(config, vdfi_data, run_table, bad_fiber_mask,
                          gal_indices=[gal_index], max_pool_gb=max_pool_gb,
                          keep_pool_for_bench=False, verbose=False)
    a, b = r1["per_gal"][0], r2["per_gal"][0]
    out = dict(cold=a, warm=b,
               speedup=a["total_t"] / max(b["total_t"], 1e-9),
               io_ratio=(b["total_io_gb"] / a["total_io_gb"]
                         if a["total_io_gb"] > 0 else np.nan))
    if verbose:
        print_warm(out)
    return out


# =====================================================================
# 4. combine micro-benchmark  (only matters if CPU-bound)
# =====================================================================
def benchmark_combine(profile_result=None, *, pool=None, nfib=200000, nwave=1036,
                      wave_blocks=(16, 32, 64, 128, 256, None), verbose=True):
    """
    Break the robust combine into its parts on a REAL fiber pool (from
    profile_galaxies' largest bin) or a synthetic one of the same shape.

    Specifically checks the suspected redundancy: combine_fibers calls
    biweight_location() -- which internally computes a nan-median and nan-MAD --
    and then calls mad_std(), which computes both again. If mad_std is ~half the
    total, and the Stage-1 `Err` it produces is unused downstream, dropping it is
    an immediate ~2x on the dominant operation.
    """
    if pool is None and profile_result is not None:
        pool = profile_result.get("biggest_pool", {}).get("pool")
    synthetic = pool is None
    if synthetic:
        rng = np.random.default_rng(0)
        pool = rng.normal(0, 1, (nfib, nwave)).astype(np.float32)
    n, nw = pool.shape

    def _t(fn, reps=1):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps

    sub = np.ascontiguousarray(pool[:, :min(64, nw)])
    sub_be = sub.astype(sub.dtype.newbyteorder(">"))
    parts = {
        "biweight_location (64 wave)": _t(lambda: biweight(sub, axis=0, ignore_nan=True)),
        "mad_std           (64 wave)": _t(lambda: mad_std(sub, axis=0, ignore_nan=True)),
        "nanmedian         (64 wave)": _t(lambda: np.nanmedian(sub, axis=0)),
        "combine_fibers    (64 wave)": _t(lambda: combine_fibers(sub, None, method="biweight")),
        "byteswap >f4 -> <f4        ": _t(lambda: sub_be.astype(np.float32)),
        "plain copy (native)        ": _t(lambda: sub.copy()),
    }
    del sub_be

    sweep = {}
    for wb in wave_blocks:
        gc.collect()
        rss0 = _rss_gb()
        t = _t(lambda: combine_fibers_chunked(pool, None, method="biweight", wave_block=wb))
        sweep[str(wb)] = dict(seconds=t, peak_rss_delta_gb=_rss_gb() - rss0)

    out = dict(nfib=n, nwave=nw, synthetic=synthetic, parts=parts, sweep=sweep,
               pool_gb=n * nw * 4 / _GB,
               label=(profile_result or {}).get("biggest_pool", {}).get("label"))
    if verbose:
        print_benchmark(out)
    return out


# =====================================================================
# 5. reporting
# =====================================================================
def _agg(per_gal, key):
    keys = sorted({k for g in per_gal for k in g[key]})
    return {k: float(np.mean([g[key].get(k, 0.0) for g in per_gal])) for k in keys}


def print_profile(prof):
    pg = prof["per_gal"]
    if not pg:
        print("no galaxies profiled")
        return
    t = _agg(pg, "stages_t")
    io = _agg(pg, "stages_io_gb")
    mf = _agg(pg, "stages_majflt")
    tot = float(np.mean([g["total_t"] for g in pg]))
    tot_io = float(np.mean([g["total_io_gb"] for g in pg]))
    tot_mf = float(np.mean([g["total_majflt"] for g in pg]))

    print("=" * 74)
    print(f"PER-GALAXY PROFILE   (mean of {len(pg)} galaxies, wave_block={prof['wave_block']})")
    print("=" * 74)
    print(f"{'stage':22s} {'seconds':>9s} {'% time':>7s} {'GB read':>9s} {'majflt':>12s}")
    print("-" * 74)
    for k in sorted(t):
        print(f"{k:22s} {t[k]:9.2f} {100*t[k]/max(tot,1e-9):6.1f}% "
              f"{io.get(k,0):9.2f} {mf.get(k,0):12,.0f}")
    print("-" * 74)
    print(f"{'TOTAL':22s} {tot:9.2f} {100.0:6.1f}% {tot_io:9.2f} {tot_mf:12,.0f}")

    ngal_full = 452
    print(f"\n  extrapolated to {ngal_full} galaxies: "
          f"{tot*ngal_full/3600:.1f} hr, {tot_io*ngal_full/1024:.1f} TB read")

    flux_gb = prof["env"].get("flux_gb", np.nan)
    if np.isfinite(flux_gb) and tot_io > 0:
        print(f"  bytes read per galaxy / flux file size = {tot_io/flux_gb:.2f}")
        if tot_io / flux_gb > 0.25:
            print("    >> a large fraction of the file is re-read EVERY galaxy.")
            print("       I/O-BOUND. Loop inversion (wavelength outer) is the fix.")
        elif tot_io / flux_gb < 0.02:
            print("    >> almost nothing is read from disk; the page cache is serving it.")
            print("       CPU-BOUND. Fuse biweight/mad_std, drop unused Err, parallelize.")

    loc = pg[0]["locality"]
    print(f"\n  fiber locality (galaxy {pg[0]['index']}): "
          f"{loc['n_touched']:,} / {loc['nfib_total']:,} fibers touched "
          f"({100*loc['n_touched']/max(loc['nfib_total'],1):.0f}%)")
    print(f"    contiguous runs = {loc['n_runs']:,}, index span = {loc['span']:,}, "
          f"fill within span = {loc['fill_fraction']:.2f}")
    if loc["n_runs"] > 0.1 * max(loc["n_touched"], 1):
        print("    >> highly fragmented selection: readahead is defeated and you pay")
        print("       roughly one page fault per fiber. Spatially sorting the galaxy")
        print("       loop (cheap) may recover real page-cache reuse.")
    else:
        print("    >> selections are fairly contiguous (IFU ordering tracks sky position),")
        print("       so sorting galaxies spatially should give good cache reuse.")

    print("\n  per-bin detail (first galaxy):")
    print(f"    {'bin':>4s} {'nfib':>10s} {'full GB':>8s} {'peak GB':>8s} "
          f"{'gather s':>9s} {'combine s':>10s}")
    for b in pg[0]["bins"]:
        print(f"    {b['bin']:4d} {b['nfib']:10,d} {b['pool_gb']:8.2f} "
              f"{b.get('peak_gb', b['pool_gb']):8.2f} "
              f"{b['t_gather']:9.2f} {b['t_combine']:10.2f}")
    for r, n, gb in pg[0]["skipped_bins"]:
        print(f"    {r:4d} {n:10,d} {'':>8s} {gb:8.2f}   SKIPPED (over max_pool_gb)")

    tg = t.get("6_bin_gather", 0.0)
    tc = t.get("7_bin_combine", 0.0)
    if tg + tc > 0:
        print(f"\n  gather vs combine: {tg:.1f} s vs {tc:.1f} s "
              f"({100*tg/(tg+tc):.0f}% / {100*tc/(tg+tc):.0f}%)")
        if tg > 2 * tc:
            print("    >> the READ dominates the arithmetic -> I/O-bound.")
        elif tc > 2 * tg:
            print("    >> the ARITHMETIC dominates the read -> CPU-bound.")
        else:
            print("    >> comparable; both paths are worth attacking.")
    print()


def print_warm(w):
    a, b = w["cold"], w["warm"]
    print("=" * 74)
    print("WARM-CACHE TEST  (same galaxy twice)")
    print("=" * 74)
    print(f"  pass 1 (cold): {a['total_t']:8.2f} s   {a['total_io_gb']:7.2f} GB read   "
          f"{a['total_majflt']:12,.0f} majflt")
    print(f"  pass 2 (warm): {b['total_t']:8.2f} s   {b['total_io_gb']:7.2f} GB read   "
          f"{b['total_majflt']:12,.0f} majflt")
    print(f"  speedup: {w['speedup']:.2f}x")
    if w["speedup"] < 1.3:
        print("  >> No meaningful speedup: reads are NOT cached (file >> RAM).")
        print("     I/O-BOUND. Loop inversion is the fix and the win is large.")
    else:
        print("  >> Second pass much faster: this galaxy's working set fits in cache.")
        print("     Check whether that survives CONSECUTIVE DIFFERENT galaxies --")
        print("     if total_io_gb stays high across distinct galaxies in")
        print("     profile_galaxies, the cache is being evicted between them and")
        print("     you are still effectively I/O-bound.")
    print()


def print_benchmark(bm):
    print("=" * 74)
    print(f"COMBINE BENCHMARK  ({'synthetic' if bm['synthetic'] else 'real'} pool: "
          f"{bm['nfib']:,} fibers x {bm['nwave']} wave, {bm['pool_gb']:.2f} GB"
          + (f", {bm['label']}" if bm.get('label') else "") + ")")
    print("=" * 74)
    p = bm["parts"]
    for k, v in p.items():
        print(f"  {k:30s} {v*1000:9.1f} ms")
    bw = p.get("biweight_location (64 wave)", np.nan)
    ms = p.get("mad_std           (64 wave)", np.nan)
    if np.isfinite(bw) and np.isfinite(ms) and (bw + ms) > 0:
        print(f"\n  mad_std is {100*ms/(bw+ms):.0f}% of the biweight+mad_std cost.")
        print("  If Stage-1 `Err` is unused downstream (bootstrap supplies real errors),")
        print(f"  dropping mad_std is an immediate ~{(bw+ms)/max(bw,1e-9):.1f}x on the combine.")
    print(f"\n  {'wave_block':>12s} {'seconds':>10s} {'RSS delta GB':>14s}")
    for k, v in bm["sweep"].items():
        print(f"  {k:>12s} {v['seconds']:10.2f} {v['peak_rss_delta_gb']:14.2f}")
    best = min(bm["sweep"], key=lambda k: bm["sweep"][k]["seconds"])
    print(f"\n  fastest wave_block = {best} "
          f"({bm['sweep'][best]['seconds']:.2f} s); current default is 64.")
    print()


def print_report(prof=None, warm=None, bench=None, setup_timing=None):
    if setup_timing:
        print("=" * 74)
        print("SETUP TIMING")
        print("=" * 74)
        for k, v in setup_timing.items():
            print(f"  {k:24s} {v:8.2f} s")
        print()
    if prof is not None:
        print_profile(prof)
    if warm is not None:
        print_warm(warm)
    if bench is not None:
        print_benchmark(bench)
    print("=" * 74)
    print("VERDICT CHECKLIST")
    print("=" * 74)
    print("  1. Is 'GB read per galaxy / flux file size' above ~0.25?      -> I/O-bound")
    print("  2. Is the warm-cache speedup below ~1.3x?                     -> I/O-bound")
    print("  3. Do majflt counts stay high on the 2nd and 3rd galaxy?      -> I/O-bound")
    print("  4. Is 6_bin_gather >> 7_bin_combine in the time column?       -> I/O-bound")
    print("  5. Otherwise, is 7_bin_combine dominant?                      -> CPU-bound")
    print()
    print("  I/O-bound  -> invert the loops (wavelength outer, galaxies inner),")
    print("                and/or transpose the flux file to wavelength-major once.")
    print("  CPU-bound  -> fuse biweight/mad_std, drop unused Err, parallelize blocks.")
    print()
