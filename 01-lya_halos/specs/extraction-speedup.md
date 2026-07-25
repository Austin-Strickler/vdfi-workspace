# Stage-1 extraction speedup

**Status: profiled and A/B'd 2026-07-25 (§1d). Verdict: the combine is the
binding constraint (69% of runtime), not the read. First fix written and
validated (`code/combine_fast.py`); not yet wired into `extract.py`.**

Companion tool: `code/extract_profile.py` — a standalone, read-only profiler that replicates the `extract_fibers` inner loop with instrumentation. It imports the real helpers, writes nothing, touches no cache, and modifies no existing file.

## Goal

Stage-1 extraction takes **13–19 hr** and new data will roughly double both fiber count and runtime. Almost every systematic test worth running (`specs/systematics-and-extraction.md` §2.1, §2.2, §2.3, §6.1, §7) is gated behind one or more re-extractions. **Making the extraction cheap is the precondition for the systematics program, not a side quest.**

Target: get a full extraction under ~1 hr without changing a single output value.

## Success criteria

1. A profile that says, with numbers, whether the bottleneck is I/O or CPU.
2. At least one change that produces **bit-identical** output — verified against a cached full run — at a measured speedup.
3. A documented `wave_block` / parallelism configuration that survives the data doubling.

---

## 1. What the code does now

`load_vdfi_field` opens `all_flux_final.fits` and `all_error.fits` **memmapped**, keeping the HDU handles alive deliberately, and reshapes each to `(nexp, nfib, nwave)`.

`extract_fibers` then loops **galaxies as the outer loop**:

```
for galaxy in ~452 galaxies:                     # OUTER
    compute distances for all (nexp x nfib) fibers
    for exposure in 130:
        read spectra[e, bg_sel]                  # 80-90" annulus -> memmap read
    biweight -> raw background -> mask/interp/smooth
    digitize every fiber into a radial bin
    for r in 8 radial bins:
        pool = empty((nfib_in_bin, 1036), float32)
        for exposure in 130:
            pool[...] = spectra[e, sel] - bg     # MEMMAP READ, scattered
        combine_fibers_chunked(pool, wave_block=64)
    cache to gal_<sig>_<i>.npz
```

Facts that matter:

| | |
|---|---|
| Exposures | ~130 |
| `all_flux_final.fits` | 20–60 GB |
| Available RAM (JupyterHub) | ~24 GB |
| `nwave` | 1036 |
| One fiber spectrum | 1036 × 4 B = **4,144 B ≈ 1.01 memory pages** |
| One 64-wave block, whole array | ~1–2 GB (fits easily) |
| Largest per-bin pool | ~500k × 1036 × 4 B = **2.07 GB** |
| Per-galaxy `.npz` | 10–100 MB |

**Already-good decisions worth not breaking:** the `error` array is never read for robust methods (`needs_err` is False for biweight), so the error file costs nothing today; pools are preallocated to exact size in float32 rather than list-and-concatenate; the per-bin pool is freed before the next bin, so peak RAM is set by the largest single bin; and the per-galaxy cache makes a re-run with an unchanged signature free.

---

## 1b. MEASURED — first profile run, COSMOS, 2026-07-25

```
all_flux_final.fits   17.80 GB      all_error.fits  17.80 GB
array shape           (132 exp, 34944 fib, 1036 wave)   dtype >f4   <-- BIG-ENDIAN
node RAM total/avail  94.3 / 35.6 GB
galaxy 0   479.1 s    read 21.23 GB    majflt 488,217
galaxy 1   544.6 s    read 26.37 GB    majflt 563,765
galaxy 2   338.3 s    read 16.36 GB    majflt 367,392
```
*(run with `max_pool_gb=20`; kernel was OOM-killed during the warm-cache test — see §1c)*

**Verdict: I/O-bound, decisively.** Per galaxy the process reads **16–26 GB against a 17.8 GB file** — i.e. roughly the *entire file, once per galaxy*, plus eviction-driven re-reads within a single galaxy. Extrapolated to COSMOS's 278 galaxies: **~35 hr and ~5.8 TB read.**

Three findings, in order of importance:

**F1 — the file nominally fits in RAM but is not being cached, because the process's own pools evict it.** 17.8 GB of data against 35.6 GB available should cache perfectly. It doesn't, because a single outer-bin pool is enormous: `pool = np.empty((n_r, 1036), float32)` for a 2M-fiber bin is **7.7 GB**, and with `max_pool_gb=20` the profiler allowed up to 20 GB. Page cache and process memory come from the same budget, so a multi-GB pool evicts the very file the next bin is about to re-read. **The pool size is not just a memory problem; it is the cause of the I/O problem.**

**F2 — the pool is allocated full-wavelength but consumed 16 wavelengths at a time.** `combine_fibers_chunked(..., wave_block=16)` processes 16 wavelengths per call, yet the pool holds all 1036. Only `n_r × 16 × 4` bytes are ever in use at once. Gathering per wavelength block instead drops peak pool memory by **65×**:

| bin fibers | full pool | block(16) pool |
|---|---|---|
| 500,000 | 1.93 GB | 0.030 GB |
| 2,000,000 | 7.72 GB | 0.119 GB |
| 4,000,000 | 15.44 GB | 0.238 GB |

**F3 — the fiber array is big-endian (`>f4`) on a little-endian machine.** Every `spectra[e, sel]` produces a `>f4` copy which must then be byte-swapped to native float32 for the subtraction — an extra full-size temporary and conversion pass on every access, on the hottest path in the pipeline. Converting the data to native order **once** removes that from every subsequent access forever.

### The fix these three imply — much simpler than file transposition

Because the file is only 17.8 GB and the node has 94 GB:

1. **Gather per wavelength block** (F2) → pools drop from ~8 GB to ~0.2 GB.
2. **That frees the page cache to hold the entire 17.8 GB file** (F1) → after the first galaxy, reads are RAM-speed and the 5.8 TB of disk traffic disappears.
3. **Store the data native-endian once** (F3) → either convert in memory at load, or write `all_flux_final_native.npy` (`<f4`, same 17.8 GB) once and memmap that. The `.npy` route is preferable: the page cache holds it, it's evictable under pressure rather than OOM-risky, and the byte-swap never happens again.

No loop inversion, no wavelength-major transposition, no HDF5. §5.2 is superseded for this dataset — keep it documented only in case the data grows past RAM.

Once I/O stops dominating, the residual cost is the biweight arithmetic, which §4 attacks (drop the unused `mad_std`, parallelize blocks).

### 1c. Profiler bug found and fixed (same day)

`profile_galaxies` retained `pool.copy()` of the largest bin so `benchmark_combine` could run on real data. With `max_pool_gb=20` that meant **holding up to 20 GB for the rest of the session** — which both starved the page cache (inflating the measured read volume) and OOM-killed the kernel during the warm-cache test.

Fixed: the retained pool is now a **fiber-subsampled slice capped at `bench_pool_gb=0.8 GB`**, `max_pool_gb` defaults to 4.0, and `gc.collect()` runs between passes. `profile_galaxies` also gained `block_gather=True` to measure the F2 restructuring directly, and `environment_report` now reads the **cgroup** limit (page cache counts against it; node-level "available" was the wrong number to compare against) and flags non-native byte order.

**The numbers above are directionally right but contaminated by that bug** — the retained pool starved the cache and inflated the read volume. Re-run with the fixed version before quoting them. The conclusion (reads ≈ file size per galaxy) is robust: galaxy 2 alone read 16.4 GB against a 17.8 GB file.

## 1d. MEASURED — fixed profiler, A/B of the F2 restructuring, COSMOS, 2026-07-25

278 galaxies, 17.80 GB flux file, `(132 exp, 34944 fib, 1036 wave)` `>f4`,
`wave_block=16`, `max_pool_gb=4.0` (no bins skipped — the largest pool is 1.85 GB).

| stage | `block_gather=False` (current) | `block_gather=True` (F2) |
|---|---|---|
| 2_bg_read | 10.46 s / 1.67 GB | 7.20 s / 0.16 GB |
| 3_bg_combine | 2.52 s | 2.71 s |
| **6_bin_gather** | **78.77 s** / 15.17 GB / 432,537 mf | **237.18 s** / 9.93 GB / 206,794 mf |
| **7_bin_combine** | **216.39 s** | **171.41 s** |
| TOTAL | **311.82 s** / 16.85 GB | **420.60 s** / 10.09 GB |

**Corrected extrapolation: COSMOS is 278 galaxies, not 452.** `print_profile`
hardcodes `ngal_full = 452`, so it over-reports by 1.63×. The real numbers are
**24.1 hr / 4.7 TB** (current) and **32.5 hr / 2.7 TB** (F2). Fix the constant.

### The verdict checklist contradicts itself — and both halves are right

Criterion 1 says I/O-bound (`read/file = 0.95`). Criterion 4 says CPU-bound
(gather 27% / combine 73%). They are measuring different things and both are true:

- **The read is genuinely enormous but genuinely fast.** 15.17 GB in 78.8 s is
  ~193 MB/s sustained, and 15.17 GB / 432,537 major faults = **36 KB per fault**,
  i.e. readahead is pulling ~9 pages per trip. The I/O is efficient; there is
  just a lot of it.
- **It is not thrashing *within* a galaxy.** Locality says 30,768 / 34,944 fibers
  (88%) are touched, and 0.88 × 17.8 GB = 15.7 GB ≈ the 15.17 GB measured. Each
  galaxy reads what it needs, **once**. There is no waste to recover inside a
  galaxy. §1b's "eviction-driven re-reads within a single galaxy" was the
  profiler bug (§1c) talking; withdraw it.
- **Therefore galaxy sorting (§5.3) cannot help much.** Every galaxy needs ~88%
  of the file, so consecutive galaxies already overlap almost completely. The
  problem is not *which* pages, it is that 17.8 GB will not stay resident.
- **And even at zero I/O cost the run is 216 s/galaxy = 16.7 hr.** The combine is
  now the binding constraint. Fixing I/O alone cannot reach the 1 hr target.

### What the A/B actually proved

**F2's restructuring is a net 35% regression (311.8 → 420.6 s), but for an
instructive reason, and half of it is worth keeping.**

- **Gather got 3× worse (78.8 → 237.2 s).** This is §5.1's page-size trap,
  which applies to *block gathering*, not only to loop inversion. One fiber
  spectrum is 4,144 B ≈ 1.01 pages, so reading 16 wavelengths of a fiber (64 B)
  still faults the whole page. Doing that 65 times per bin re-touches every page
  65 times. §5.1 warns about this for wavelength-outer loops; it was not carried
  over to F2, and F2 is the same mistake in a smaller box.
- **F1 was real, though.** Bytes read fell 16.85 → 10.09 GB and major faults
  halved (433,811 → 207,339); galaxy 1 read only 5.37 GB with 65,801 faults.
  Smaller pools *do* let the cache hold more. It just does not come close to
  paying for 65× the page touches.
- **Combine got 26% better (216.4 → 171.4 s), and this is the keeper.** The only
  thing that changed for the combine is that the pool became a contiguous
  `(n_fib, 16)` block instead of a strided `pool[:, j0:j1]` view of a
  `(n_fib, 1036)` array. numpy's `nanmedian(axis=0)` moves the reduction axis to
  the end and scans with a 4,144-byte stride — one cache line per element.
  **That 1.26× is available without touching the I/O at all**, via a contiguous
  copy of each block (~4 ms per block, against ~25 ms saved).

**Conclusion: keep `block_gather=False`. Take the contiguity win in the combine
instead, where it costs a memcpy rather than 65 passes over the file.**

### 1e. The revised fix stack

The outer two bins are 864,712 of ~880,000 fiber-exposures (98%) and ~93% of
combine time. Everything below is really about bins 10 and 11.

| # | change | bit-identical? | gather | combine | status |
|---|---|---|---|---|---|
| 1 | **Fused median/MAD** (`code/combine_fast.py`) | flux **yes**, err ~1e-7 rel (or exact with `strict=True`) | — | ÷1.27 | validated on the real pool |
| 1b | + contiguous **transposed** block | flux ~1e-15 rel | — | ÷1.65 | validated, opt-in |
| 2 | **Resident native-endian array** | yes | ÷~4 | — | needs the cgroup number (O7) |
| 3 | **Parallelise wavelength blocks** (§4.2) | yes | — | ÷n_cores | needs O6 |
| 4 | **Cap the outer radius** for systematics runs | n/a (different product) | ÷~4 | ÷~14 | config only |

**The ÷1.27 / ÷1.65 above are the REAL-POOL numbers and they are materially
worse than the synthetic ones** (÷1.9 / ÷2.6 on `rng.normal` data of the same
shape). Measured on `gal1_bin11`, 176,151 fibers × 1036 wave:

```
A  current: biweight_location + mad_std     235.1 ms/blk   1.00x
   (biweight_location alone)                168.6 ms/blk   1.39x   <-- mad_std is only 28% here
C  fused, axis=0                            185.0 ms/blk   1.27x
D  fused, axis=1 contiguous [+transpose]    136.0 ms/blk   1.65x
E  fused strict, axis=1                     181.5 ms/blk   1.25x
```

The ceiling is visible in row 2: on real fibers `mad_std` is only 28% of the
current cost, not the ~50% it is on Gaussian noise, so removing it entirely
could only ever give 1.39×. Two effects are eating the rest: the fused path
computes its surviving MAD in **float64** (astropy's `mad_std` does it in
float32, at half the memory traffic), and real pools have a NaN structure that
changes `nanmedian`'s cost. **Always quote the real-pool number.**

Projected per-galaxy, COSMOS (revised):

```
now                                    311.8 s   24.1 hr
+ fused combine (1)                    ~242   s  ~18.7 hr
+ fused + transposed (1b)              ~227   s  ~17.5 hr
+ resident array (2)                   ~168   s  ~13.0 hr
+ 8-way block parallelism (3)           ~53   s   ~4.1 hr
+ outer-bin cap (4, exploratory only)   ~12   s   ~0.9 hr
```

**The 1 hr target is not reachable on a JupyterHub session for the full radial
range**, and the combine work alone buys less than a factor of 2. §4.4 (a TACC
allocation) is on the critical path, not an optional extra — parallelism across
wavelength blocks is where the remaining order of magnitude lives. What *is*
reachable without one: ~13 hr production and ~1 hr for the capped exploratory
runs most of `systematics-and-extraction.md` needs.

### 1e-bis. The machine numbers, and what they kill (closes O6, O7)

```
cores: 12 | usable: 12
cgroup memory.max: 21474836480 = 20.0 GB
```

**The resident-array plan (item 2) is dead.** The flux file is 17.80 GB against a
**20.0 GB** cgroup budget, and page cache counts against that same budget. Add
the largest pool (1.85 GB) and the interpreter (~1.5 GB) and you need ~21 GB.
It misses by about a gigabyte.

That single number retro-explains everything §1d measured. The file *almost*
fits, which is why `block_gather=True`'s tiny pools cut bytes read by 40% and
let galaxy 1 come in at 5.37 GB — with a small enough pool the cache nearly
holds the file. It just can never do so stably, and buying that headroom via
65× page re-touching costs three times what it saves.

Implication: there is no configuration in which the read gets cheap on this
session. Gather is a ~79 s/galaxy floor until either the cgroup grows or the
data shrinks.

**12 cores is the lever, and it must be THREADS.** Twelve *processes* would each
carry their own 1.85 GB pool and their own cached pages — instantly over budget.
Twelve *threads* share one memmap, one page cache, and one pool.

The thing that makes this work, verified rather than assumed: **numpy releases
the GIL inside `partition`/`sort`/ufuncs, so `nanmedian` genuinely runs
concurrently.** Measured 1.74–1.79× on 2 cores (87–90% efficiency), with output
bit-identical across every thread count. `combine_fibers_parallel` in
`code/combine_fast.py` implements it; `benchmark_threads()` sweeps the count.

Expect efficiency to fall off well before 12 — nan-median is memory-latency
bound, not compute bound. At a plausible 6–8×:

```
combine   216.4 s  ->  ~30 s
gather     78.8 s  ->  unchanged
TOTAL     311.8 s  -> ~125 s   =  ~9.7 hr for COSMOS
```

**After which gather is 63% of runtime and the combine work is finished.** The
next target then becomes the gather, which is also thread-parallel (it is
`memcpy` plus page faults, both GIL-free), and after that the only remaining
lever is scope — capping the outer radius.

### 1e-ter. MEASURED thread scaling, real pool, 2026-07-25 (closes O8)

`gal1_bin11`, 176,151 fibers × 1036 wave (a 1/3 fiber subsample of the real bin;
5.7% NaN), `wave_block=16`, `transpose=False`:

| threads | seconds | speedup | efficiency | bit-identical |
|---|---|---|---|---|
| 1 | 15.24 | 1.00× | 100% | — |
| 2 | 9.30 | 1.64× | 82% | yes |
| 4 | 5.32 | 2.87× | 72% | yes |
| 6 | 4.08 | 3.74× | 62% | yes |
| **8** | **3.32** | **4.59×** | 57% | yes |
| 12 | 3.70 | 4.12× | 34% | yes |

**Use `n_threads=8`.** Twelve threads are *slower* than eight.

`lscpu`: `Thread(s) per core: 1`, `Core(s) per socket: 1`, 12 CPUs, model
"Intel Core Processor (Haswell, no TSX, IBRS)". So this is a **VM with 12
genuine vCPUs and no hyperthreading** — the 8→12 regression is not SMT
siblings. It is memory-bandwidth saturation and/or hypervisor vCPU contention.

Two consequences:

- **No placement fix is available.** Each vCPU is presented as its own socket,
  so the guest kernel has no true cache/NUMA topology to schedule against.
  There is nothing meaningful to pin threads to.
- **4.59× is a property of this VM, not of the algorithm.** It may drift with
  co-tenant load. Re-run `benchmark_threads` rather than hard-coding the value
  as a constant of nature.

Resulting budget: combine 216.4 → 47 s, gather unchanged at 78.8 s, **total
~143 s/galaxy = 11.0 hr for COSMOS.** Gather is now 55% of runtime and the
combine optimisation is essentially finished.

**This machine is close to a worst case for this workload**: a virtualised
older-generation CPU, memory bandwidth shared with co-tenants, and a 20 GB
cgroup against a 17.8 GB file. TACC (§4.4) fixes all three axes at once — real
physical cores, real memory bandwidth, and enough RAM that the flux file simply
stays in page cache and gather stops being I/O at all. It is now the highest-
leverage item in this document.

### 1e-quater. One further lever, NOT bit-identical: drop the float64 upcast

Since the ceiling is memory bandwidth, the way to raise it is to move fewer
bytes. astropy's `biweight_location` upcasts to float64, so every temporary is
8 bytes/element on data that is float32 on disk. Computing in float32 halves
that traffic: **1.34× faster serially, with unchanged thread scaling** (1.90×
vs 1.94× on 2 cores), i.e. it multiplies with the 4.59×.

Cost: the result deviates by **2.4e-7 in units where the fiber scatter is 1.0**
— roughly 10⁻⁴ of the statistical uncertainty on the biweight of a 120k-fiber
pool, and ~4 orders of magnitude below anything the bootstrap resolves. But it
is *not* bit-identical, so per §6.4 it needs an explicit decision and an
accuracy comparison on real pools, not a default flip.

### 1f. §4.1 is dead — `Err` IS consumed downstream (closes O2)

`stack.coadd_galaxies` passes `cube_err` straight into `combine_fibers` for
every method in `config.galaxy_combine_methods`, whose default is
`["biweight", "inv_var", "mean", "median", "sigma_clip", "weighted_median"]`.
Both `inv_var` and `mean` **require** `err` and raise without it — and with a
NaN-filled `Err` they would return all-NaN stacks. `stack.stack_one_galaxy` also
uses it for the per-bin line S/N via `integrated_line_flux(err=...)`.

So "drop the unused Err for a free 2×" is not available. **Fusing supersedes it:
it delivers the same ~2× and keeps `Err`.** The redundancy §4.1 identified was
real; the proposed remedy was aimed at the wrong end of it.

### 1g. Two byte-level traps found while building the fused path

Anyone reimplementing this needs both, or bit-identity fails for reasons that
look like bugs:

1. `astropy.stats.mad_std` returns `MAD * 1.482602218505602`. Dividing by
   `0.6744897501960817` instead — the same constant, algebraically — rounds
   differently and produced a **6e-8 float32 discrepancy**.
2. `biweight_location` opens with `data.astype(np.float64)`; `mad_std` does not.
   So the MAD inside the biweight is a float64 MAD and the MAD inside `mad_std`
   is a float32 MAD of the same data. They differ at ~1e-7 relative. `combine_fast`
   defaults to reproducing the **flux** exactly and computing `Err` from the
   (more accurate) float64 MAD; `strict=True` pays one extra pass to reproduce
   both bit-for-bit.

---

## 2. Two hypotheses — and both arithmetic paths land on 13–19 hr

This is why the profiler is necessary rather than optional: back-of-envelope numbers do not discriminate.

### Hypothesis A — I/O-bound (page-cache thrashing)

The galaxy loop is outer and the memmap read is inner, so the flux file is traversed **roughly once per galaxy**. With a 20–60 GB file against ~24 GB of RAM, the page cache cannot hold it: consecutive galaxies touch overlapping but non-identical fiber sets, so pages are evicted before reuse.

If each galaxy touches ~30–100% of the file (a 3000 kpc outer bin ≈ 357″ covers a large fraction of the field, and scattered fancy-indexing faults in whole pages regardless), then:

> 452 galaxies × 6–60 GB = **3–27 TB** read. At ~500 MB/s effective for scattered reads on shared storage → **1.5–15 hr**.

### Hypothesis B — CPU-bound on the robust combine

`combine_fibers` with `method="biweight"` does:

```python
out     = biweight(spec, axis=0, ignore_nan=True)   # internally: nanmedian + nanMAD
scatter = mad_std(spec, axis=0, ignore_nan=True)    # nanmedian + nanMAD AGAIN
```

That's **four nan-median passes per wavelength block**, two of which are recomputing what the first two already computed. `np.nanmedian` is not cheap: it copies the array to handle NaNs, then partitions.

For the largest bin: 1036/64 = 17 blocks × 4 medians × a (500k, 64) float32 nan-median at ~0.5–2 s each ≈ **35–140 s for one bin**. Across the bins that matter, ~100–200 s per galaxy:

> 452 × 150 s ≈ **19 hr**.

Both hypotheses reproduce the observation. Measure.

---

## 3. The diagnostic — `code/extract_profile.py`

Non-invasive by construction: it replicates the loop rather than patching it, calls the same `radial_bin_edges` / `combine_fibers` / `smooth_interp_then_gauss`, bypasses the cache in both directions, and discards results.

```python
import extract_profile as ep

ep.environment_report(cfg, vdfi_data)        # file size vs RAM — often settles it alone

prof = ep.profile_galaxies(cfg, vdfi_data, run_table, bad_fiber_mask, n_gal=3)
ep.print_profile(prof)

warm = ep.warm_cache_test(cfg, vdfi_data, run_table, bad_fiber_mask, gal_index=0)

bench = ep.benchmark_combine(prof)           # runs on the real largest pool
ep.print_all(prof, warm, bench)
```

What it measures, and why each one:

| Measurement | Why |
|---|---|
| Per-stage wall time (geometry / bg read / bg combine / bg smooth / digitize / bin gather / bin combine) | Separates the memmap read (`6_bin_gather`) from the arithmetic (`7_bin_combine`) — the single most informative split |
| `psutil` `io_counters().read_bytes` per stage | Bytes actually fetched from the block layer. Compare to file size: >0.25× per galaxy means the file is being re-read |
| `ru_majflt` (major page faults) per stage | The direct measure of memmap cache **misses**. Every one is a disk trip |
| **Warm-cache test** — same galaxy twice | If pass 2 isn't much faster, reads aren't being cached at all → I/O-bound, decisively |
| Fiber-index locality: touched count, contiguous runs, span, fill fraction | Whether IFU ordering makes a galaxy's fibers contiguous. Fragmented → readahead defeated, ~1 fault per fiber. Contiguous → spatially sorting the galaxy loop could recover real cache reuse for free |
| `biweight` vs `mad_std` vs `nanmedian` timed separately on the **real** largest pool | Directly quantifies the redundant-median waste in §4.1 |
| `wave_block` sweep 16/32/64/128/256/None with peak RSS | Finds the knee. The current 64 was chosen to survive OOM, not to be fast |

Safety: `max_pool_gb=6.0` skips (and reports) any bin whose pool would exceed that, so a 24 GB session survives. Start with `n_gal=2`.

**Reading the verdict** (`print_all` prints this checklist):

1. GB read per galaxy ÷ flux file size > ~0.25 → I/O-bound
2. Warm-cache speedup < ~1.3× → I/O-bound
3. `majflt` stays high on galaxies 2 and 3 → I/O-bound
4. `6_bin_gather` ≫ `7_bin_combine` → I/O-bound
5. Otherwise `7_bin_combine` dominant → CPU-bound

---

## 4. If CPU-bound — three changes, ~50× combined, none of them structural

This is the cheaper branch and, given the median arithmetic above, a live possibility. In rough order of reward per unit effort.

### 4.1 Stop computing an error nobody uses — expected ~2×

Confirmed with Austin: the Stage-1 `Err` is *"not really used, it is just great for validation checking,"* and **there is currently no config knob to turn it off.**

For robust methods it is only `mad_std(spec)/√N` — an analytic proxy that the Stage-3 galaxy bootstrap supersedes entirely. Yet it costs a full extra nan-median + nan-MAD pass over the largest arrays in the pipeline, i.e. roughly **half the combine time**.

**Change:** add `compute_fiber_err: bool = True` to `PipelineConfig`; when False, `combine_fibers` returns `out, None` for the robust methods and `Err` is filled with NaN. Keep the default True so nothing silently changes, and flip it for production runs.

**Before flipping it, confirm nothing downstream consumes it** — check `stack.build_stacks` (`inv_var`, `weighted_median` galaxy coadds), `measure.integrated_line_flux`'s `err=` path, and any plotting that reads the `ERROR` HDU. If something does, the flag becomes a deliberate trade rather than a free win. *(Open question O2.)*

### 4.2 Parallelize over wavelength blocks — expected near-linear in cores

`combine_fibers_chunked` already loops over independent wavelength blocks, and the result is per-wavelength independent, so this is embarrassingly parallel with **bit-identical output**. Peak memory does *not* grow proportionally if each worker holds only its own block.

Cleanest non-invasive form: a new `combine_fibers_parallel(...)` alongside the existing function, selected by a config flag, so the serial path stays as the reference implementation for regression testing.

Caveat: on a JupyterHub session with few cores this buys little. It is the main argument for §4.4.

### 4.3 Fuse the median computation — expected ~1.3–1.4× on top

Even with `Err` retained, `biweight()` and `mad_std()` each independently compute a nan-median and nan-MAD of the same array. A fused helper computing median once, MAD once, then deriving *both* the biweight location and the MAD-based scatter removes half of the four passes. More invasive than §4.1 (it reimplements a piece of astropy) so it should be validated against `astropy.stats` to floating-point tolerance before adoption.

**Speculative but worth benchmarking:** the median and MAD only set the biweight's *weights*. Estimating them from a random 50k-fiber subsample of a 500k-fiber pool changes them by ~0.4% of the distribution width while the weighted correction still uses all fibers — potentially another ~5–10× on the dominant operation for a quantifiable, testable approximation. Do not adopt without an explicit accuracy comparison on real pools; do measure it.

### 4.4 Ask TACC for a real allocation

A compute node has far more RAM and cores than a JupyterHub session. More RAM lets `wave_block` grow (less chunking overhead, fewer passes); more cores make §4.2 pay. This should be requested **before** the data doubles.

---

## 5. If I/O-bound — and the page-size trap that breaks the obvious fix

### 5.1 The trap

The intuitive fix is to invert the loops — wavelength outer, galaxies inner — so the file is read once instead of ~452 times:

```
for wave_block:
    read spectra[:, :, j0:j1] once
    for galaxy, for bin: gather -> biweight
```

**This does not work on the current file layout.** The array is `(nexp, nfib, nwave)` in C order, so `nwave` is the fastest-varying axis and one fiber's spectrum occupies 4,144 contiguous bytes ≈ 1.01 pages. Reading 64 wavelengths of a fiber is 256 bytes — but the OS must fault in the whole page. **A strided wave-block read therefore costs essentially the same I/O as reading the entire file.** Seventeen blocks would mean seventeen full reads, not one.

Anyone implementing loop inversion without noticing this will measure no improvement and conclude, wrongly, that the diagnosis was bad.

### 5.2 The fix that actually works: transpose once, wavelength-major

Rewrite the fiber data once into wavelength-major chunks, then a wave block is genuinely contiguous:

- **Simplest:** 17 memmapped `.npy` files, each `(nexp, nfib, 64)`, ~1–2 GB apiece. No new dependencies, trivially readable, trivially parallel.
- **Tidier:** one HDF5 file chunked `(nexp, nfib, 64)`. Requires `h5py`.

One-time cost is roughly one full read plus one full write (out-of-core, in fiber slabs so it never exceeds RAM) — a few hours **once**. After that every extraction reads the data exactly once, in sequential order, and §4.2's parallelism composes on top.

This is the change that turns 13–19 hr into minutes if Hypothesis A holds. It's also the biggest change in this document, which is precisely why the profile comes first.

### 5.3 Cheap partial mitigations to try before transposing

- **Sort the galaxy loop spatially** (e.g. by RA, or Hilbert/Morton order on RA/Dec). Consecutive galaxies then overlap heavily in their fiber sets, so the page cache is reused instead of thrashed. This is a **one-line reordering of `run_table`** and produces identical output — only the processing order changes. If the profiler reports high locality (few contiguous runs per galaxy, i.e. IFU index tracks sky position), this could be a large win for nearly zero effort. **Try this first.**
- **Cap the outer radius during exploratory runs.** The 2000–3000 kpc bin drives both the pool size and the fraction of the file touched. A run capped at 1000 kpc is much cheaper and adequate for most systematics sweeps.
- **Confirm the error file is never touched.** With `fiber_combine_method="biweight"`, `needs_err` is False and `error[...]` should never be indexed. The profiler's `read_bytes` will show if that's actually true.

---

## 6. Correctness protocol — non-negotiable

Every change here is a performance change that must not alter a single number. The failure mode to guard against is the "smoothly wrong" one: output that still looks like output.

1. **Freeze a regression fixture.** Keep the per-galaxy `.npz` cache from one complete current run, untouched, as ground truth.
2. **Assert bit-identity** (`np.testing.assert_array_equal`, or `assert_allclose(rtol=0, atol=0)` with NaN equality) for `spec`, `err`, `nfib`, `bg`, `vr` on at least 20 galaxies spanning the full range of fiber counts, before trusting any rewritten path.
3. **Changes that must be bit-identical:** wavelength-block parallelism, loop inversion, file transposition, spatial sorting of the galaxy loop, `wave_block` changes.
4. **Changes that are deliberately not bit-identical, and need their own justification:** disabling `Err` (§4.1 — output becomes NaN by design), fused median (§4.3 — validate against astropy to fp tolerance), subsampled median (§4.3 speculative — needs an explicit accuracy budget).
5. **Keep the serial path.** Whatever gets added should sit alongside the current implementation behind a config flag, not replace it, so the reference remains runnable.

---

## 7. Sequenced plan

**Step 1 — diagnose (one afternoon)**

Run `environment_report`, then `profile_galaxies(n_gal=3)`, then `warm_cache_test`, then `benchmark_combine`. Record the output in `docs/research-notes.md` under a new "Extraction performance" section. Do not change any code before this.

**Step 2 — free wins regardless of verdict**

- Resolve open question O2 (is `Err` consumed downstream?) and, if not, add the `compute_fiber_err` flag (§4.1).
- Sort the galaxy loop spatially (§5.3) — one line, identical output, potentially large if I/O-bound.
- Set `wave_block` to the measured knee rather than the survive-OOM value.
- Freeze the regression fixture (§6.1).

**Step 3 — take the branch the profile indicates**

- CPU-bound → §4.2 parallel blocks, then §4.3 fused median if still needed.
- I/O-bound → §5.2 transposition, then §4.2 on top.
- Request the TACC allocation either way (§4.4).

**Step 4 — only then, the science re-extraction**

Once the extraction is cheap, run the multi-variant extraction from `specs/systematics-and-extraction.md` §2.1 (`SPEC_NOBG`, raw unmasked `BG_VARIANTS`, `BG_NFIB`, `BG_HALVES`) plus §2.3 outer-bin subdivision, in a single pass. That one run then unblocks the entire background-systematics program permanently.

---

## 8. Open questions

- ~~**O1** — What does the profile actually say?~~ **ANSWERED (§1d).** Both, and
  the combine is now the binding constraint at 69% of runtime.
- ~~**O2** — Is Stage-1 `Err` consumed downstream?~~ **ANSWERED (§1f): yes.**
  `stack.coadd_galaxies` feeds it to `inv_var`/`mean`, and `stack_one_galaxy`
  uses it for line S/N. §4.1 is dead; fusing replaces it.
- ~~**O3** — Exact `nfib` and dtype.~~ **ANSWERED:** `(132, 34944, 1036)`, `>f4`,
  17.80 GB. Non-native byte order (§1g.2).
- ~~**O4** — Does fiber index order track sky position?~~ **ANSWERED: yes, and it
  does not help.** 86 contiguous runs over a 34,944 span, fill 0.88 — highly
  contiguous. But *every* galaxy touches 88% of the fibers, so consecutive
  galaxies already overlap almost totally and there is nothing for spatial
  sorting to recover (§1d). §5.3's first bullet is withdrawn.
- **O5** — Disk quota for a transposed copy. **Back on the table.** The resident
  array is ruled out by the 20 GB cgroup (§1e-bis), so if gather becomes the
  bottleneck after threading — it will — §5.2's wavelength-major transposition is
  the only remaining structural fix for the read.
- ~~**O6** — cores?~~ **ANSWERED: 12 usable** (§1e-bis). Threading is the lever;
  numpy releases the GIL, measured 87–90% efficiency on 2 cores.
- ~~**O7** — cgroup limit?~~ **ANSWERED: 20.0 GB** (§1e-bis), against a 17.80 GB
  file. The resident-array plan misses by ~1 GB and is dead.
- **O8 — NEW: how far does thread scaling actually go on 12 cores?**
  `combine_fast.benchmark_threads(a)` answers it in ~2 min. nan-median is
  memory-latency bound, so the curve will flatten below 12; where it flattens
  sets the real ceiling and decides whether §4.4 (TACC) is still needed.
- **O9 — NEW: is gather thread-parallel too?** It is `memcpy` plus page faults,
  both GIL-free, so in principle yes. It becomes 63% of runtime the moment the
  combine is threaded, so this is the next question after O8.

## 9. IMPLEMENTATION — threading + fusion of the per-bin combine

**BUILT AND TESTED 2026-07-25.** Both phases shipped. Defaults are unchanged, so
an untouched config runs the untouched serial reference path.

### 9.-1 What actually landed

| file | change |
|---|---|
| `utils_lya_halo/extract.py` | `biweight_fused`, `combine_fibers_fused`, `_combine_one_block`, `combine_fibers_threaded`, `get_combine_executor`, `shutdown_combine_threads`; call site at the per-bin combine dispatches to the fast path; `_extract_signature` hashes the fused knobs only |
| `utils_lya_halo/config.py` | `combine_threads=1`, `combine_min_fibers=50_000`, `combine_fused=False`, `combine_fused_strict=False`, `combine_fused_transpose=False` |
| `run_config.py` | same knobs, `combine_threads=8` |
| `utils_lya_halo/guide.py` | the six new names registered under `extract` |
| `code/combine_fast.py` | demoted to a pure validation harness; now IMPORTS the implementation instead of holding a second copy |

`combine_fibers` and `combine_fibers_chunked` are untouched and remain the
reference. Verified green:

```
A. threading bit-identical vs reference, threads 1/2/4/8
     biweight / median / sigma_clip     -> bit-identical
B. mean / inv_var still receive their err pool
                                        -> bit-identical
C. fused         flux max|d|=0.00e+00   err rel=1.27e-07
   fused+strict  flux max|d|=0.00e+00   err rel=0.00e+00
   fused+transp  flux max|d|=7.68e-17   err rel=1.27e-07
   fused+8 thr   flux max|d|=0.00e+00   err rel=1.27e-07
D. min_fibers guard sends small pools to the reference
                                        -> bit-identical
E. cache signature
     combine_threads / wave_block / min_fibers -> UNCHANGED hash
     combine_fused (+strict, +transpose)       -> distinct hashes
     a config predating these fields            -> UNCHANGED hash
```

That last line matters: existing caches stay valid.

### 9.-2 Three things worth knowing about the implementation

1. **The executor is a process-wide singleton** (`get_combine_executor`), not a
   per-call `with` block. 278 galaxies × 12 bins would be 3,336 pool creations,
   and in a notebook a per-call pool is also destroyed and rebuilt on every
   interrupt. Call `extract.shutdown_combine_threads()` to release the threads.
2. **Threading applies to every method; fusion is biweight-only.** A non-biweight
   `fiber_combine_method` with `combine_fused=True` emits a `RuntimeWarning` and
   falls back, rather than silently doing something else.
3. **Serial and threaded execute the same `_combine_one_block`.** There is one
   code path for the per-block arithmetic, so the two cannot drift apart.

### Original plan (kept for the reasoning)

Written 2026-07-25 after reading the current `extract_fibers`.

### 9.0 Why this is a small change

`combine_fibers_chunked` has exactly **one** call site in the whole package:

```python
# extract.py:707, inside the `for r in range(nrad)` loop
Spec[cnt, r], Err[cnt, r] = combine_fibers_chunked(
    pool_spec, pool_err, method=fiber_method,
    sigma=sc_sigma, maxiters=sc_maxiters, wave_block=wave_block,
)
```

Everything else about the galaxy loop — the pool preallocation, the
one-bin-at-a-time memory discipline, the cache, the background handling — stays
exactly as it is. We are changing how one function iterates, nothing else.

### 9.1 Split it into two phases, and ship phase 1 first

This is the important sequencing decision. **The threading and the fusing are
separate changes with completely different risk profiles, and lumping them
together would gate a zero-risk 4.6× behind a judgement call about float
precision.**

| | speedup | output |
|---|---|---|
| **Phase 1 — thread the EXISTING astropy path** | ~4.6× | **bit-identical, unconditionally** |
| Phase 2 — also fuse the median/MAD | ~1.27× more | flux exact, err ~1e-7 (or transpose: flux ~1e-15) |

Phase 1 changes *the order blocks are computed in*, and nothing else. Each
wavelength block reads a disjoint slice of the pool and writes a disjoint slice
of the output. There is no shared mutable state and no accumulation across
blocks, so the result is identical by construction — not "identical to
tolerance", identical because the same arithmetic happens on the same bytes.

Phase 2 changes the arithmetic. It is worth ~28% on top and it is validated
(`combine_fast.validate`), but it needs its own decision under §6.4. **Do not
bundle them.**

### 9.2 Phase 1 — the change

**New function in `extract.py`, directly beneath `combine_fibers_chunked`:**

```python
def combine_fibers_threaded(spec, err=None, method="biweight", sigma=3.0,
                            maxiters=5, wave_block=16, weights=None,
                            n_threads=8, min_fibers=50_000, executor=None):
    """
    Bit-identical to combine_fibers_chunked, with the wavelength blocks
    evaluated on a thread pool instead of sequentially.

    Safe because the blocks are independent: each reads spec[:, j0:j1] and
    writes out[j0:j1], and no block reads another's output. numpy releases the
    GIL inside partition/sort/ufuncs, so the nan-medians genuinely run
    concurrently -- measured 4.63x on 8 threads (see spec 1e-ter).

    Falls back to the serial path when the pool is small (min_fibers), when
    n_threads <= 1, or when wave_block is None -- threading a 12-fiber inner
    bin is pure dispatch overhead.
    """
```

Body: identical loop to `combine_fibers_chunked`, but the per-block work is
submitted to a `ThreadPoolExecutor` and each worker assigns into `out[j0:j1]` /
`out_err[j0:j1]`.

**Call site becomes:**

```python
n_threads = getattr(config, "combine_threads", 1)
...
combine = (combine_fibers_threaded if n_threads > 1 else combine_fibers_chunked)
Spec[cnt, r], Err[cnt, r] = combine(pool_spec, pool_err, method=fiber_method,
                                    sigma=sc_sigma, maxiters=sc_maxiters,
                                    wave_block=wave_block,
                                    **({"n_threads": n_threads,
                                        "executor": _pool} if n_threads > 1 else {}))
```

**Config, next to `combine_wave_block`:**

```python
combine_threads: int = 1   # PERFORMANCE KNOB ONLY (no effect on results):
                           #   threads for the per-bin fiber combine. 1 -> serial
                           #   (reference path). 8 measured best on the 12-vCPU
                           #   JupyterHub VM; 12 is SLOWER than 8 (memory
                           #   bandwidth, see spec 1e-ter). Re-measure with
                           #   combine_fast.benchmark_threads on a new machine.
```

### 9.3 Four things that will bite if missed

1. **`combine_threads` must NOT enter `_extract_signature`.** It is bit-identical,
   so a run at 8 threads must reuse a cache built at 1 thread. This is the same
   reasoning already applied to `combine_wave_block`, which is deliberately
   excluded — follow that precedent exactly. Getting this wrong silently
   invalidates every cached galaxy and costs a full re-extraction.

2. **Create the `ThreadPoolExecutor` once per `extract_fibers` call**, not per
   bin. 278 galaxies × 12 bins = 3,336 pool creations otherwise. Open it before
   the galaxy loop, close it in a `finally` so an interrupted run doesn't leak
   threads.

3. **Memory: threads multiply the temporaries.** Each holds its own float64
   working set for one block, ~`n_r × wave_block × 8 B × ~5 arrays`. For the real
   bin 11 (480k fibers, `wave_block=16`) that is ~307 MB per thread — **~2.5 GB
   across 8 threads, on top of the 1.85 GB pool.** Fine against 20 GB, but it
   squeezes the page cache, which is already the reason gather is slow (§1e-bis).
   Raising `combine_wave_block` would multiply this linearly; don't.

4. **`min_fibers` guard.** Bins 0–8 total under 5 s per galaxy; bins 9–11 are
   ~264 s. Dispatching 65 thread tasks for a 12-fiber bin costs more than it
   saves. Threshold at ~50k fibers so only bins 9/10/11 thread.

### 9.4 Verification, before it counts as done

1. `combine_fast.validate_on_pool(a)` — already passing.
2. **Freeze the fixture (§6.1)**: keep the current `.npz` cache from a completed
   serial run as ground truth.
3. Run 20 galaxies spanning the fiber-count range at `combine_threads=8` with
   the cache disabled, and `assert_array_equal` `spec`/`err`/`nfib`/`bg`/`vr`
   against the fixture. Bit-identical is the pass criterion, not "close".
4. Confirm a cache written at `combine_threads=1` is *reused*, not invalidated,
   by a run at 8 — i.e. that item 9.3.1 was done right.
5. `guide.check_guide()` after adding the function, per the module-registration
   convention.

### 9.5 What phase 1 buys, and what it does not

```
combine   216.4 s -> ~47 s
gather     78.8 s -> unchanged
TOTAL     311.8 s -> ~143 s   =  11.0 hr for COSMOS
```

**Gather then becomes 55% of runtime and this document's combine work is
finished.** The next question is O9 (does the gather thread too), and after that
the only remaining levers are scope (dropping the outermost bin, §1e) and the
machine itself (§4.4 — which on the evidence of §1e-bis is the real answer).

## See also

- `code/extract_profile.py` — the diagnostic
- `code/combine_fast.py` — the fused combine (§1e item 1) plus its astropy
  bit-identity validator and benchmark. Run `validate()` and
  `validate_on_pool(prof)` before wiring it into `extract.py`.
- `specs/systematics-and-extraction.md` — what this unblocks; §0.2 (the extraction boundary), §2.1 (multi-variant extraction), §2.4 (the summary version of this document)
- `utils_lya_halo/extract.py` — `extract_fibers`, `combine_fibers`, `combine_fibers_chunked`
- `utils_lya_halo/smoothing.py` — the background smoother called per galaxy
