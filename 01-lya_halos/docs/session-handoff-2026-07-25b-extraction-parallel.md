# Session handoff — 2026-07-25b (extraction parallelised, shipped)

Continues `session-handoff-2026-07-25.md`, which ended with the extraction
profiled but no code changed. **This session diagnosed the profile, built the
fix, and validated it end to end.** Stage-1 extraction is now 2.38× faster on
real data with bit-identical output.

Authoritative detail lives in `specs/extraction-speedup.md` §1d–§1g and §9.
This document is the orientation layer.

---

## Headline

```
COSMOS Stage-1 extraction, 278 galaxies
  before   262.7 s/galaxy    20.3 hr
  after    110.4 s/galaxy     8.5 hr        2.38x, bit-identical
```

Verified end to end on 3 real galaxies through `extract_fibers`:
`Spec`, `Err`, `NFIB`, `Background`, `VR` all bit-identical between the serial
and threaded runs, with `use_cache=False` on both so both actually computed.

---

## The diagnosis, in one paragraph

The profiler's verdict checklist contradicted itself — criterion 1 said
I/O-bound (reads ≈ 0.95 × file size per galaxy), criterion 4 said CPU-bound
(gather 27% / combine 73%). **Both were true and neither was actionable as
stated.** The read is large but efficient (~193 MB/s, 36 KB per major fault)
and *unavoidable*: locality shows each galaxy touches 88% of all fibers, and
0.88 × 17.8 GB ≈ the 15.17 GB measured — each galaxy reads what it needs, once.
Meanwhile the combine was 216 s/galaxy of single-threaded order statistics.
Per galaxy the bins hold 1,009,755 fiber-spectra × 1036 wavelengths = **1.05
billion values**, and 2.9 × 10¹¹ across COSMOS. At ~10⁷ elements/s for
nan-median that *is* 24 hours. It was never a bug; it was a compute budget, and
the only lever that scales against a compute budget is cores.

---

## What was built

| file | change |
|---|---|
| `utils_lya_halo/extract.py` | `biweight_fused`, `combine_fibers_fused`, `_combine_one_block`, `combine_fibers_threaded`, `get_combine_executor`, `shutdown_combine_threads`; fast-path dispatch at the per-bin combine; `_extract_signature` hashes the fused knobs only |
| `utils_lya_halo/config.py` | `combine_threads`, `combine_min_fibers`, `combine_fused`, `combine_fused_strict`, `combine_fused_transpose` |
| `run_config.py` | same knobs, `combine_threads = 8` |
| `utils_lya_halo/guide.py` | six new names registered under `extract` |
| `code/combine_fast.py` | validation harness — imports the implementation, holds no copy |
| `specs/extraction-speedup.md` | §1d–§1g measurements, §9 implementation record |

`combine_fibers` and `combine_fibers_chunked` are **untouched** and remain the
reference. All defaults are off, so an unchanged config runs unchanged code.

### To turn it on

```python
cfg.combine_threads = 8      # 4.63x, bit-identical, cache stays valid
cfg.combine_fused   = True   # +1.27x, flux exact, Err moves 1.5e-7, BUSTS cache
```

### Two facts about the knobs

- **`combine_threads` is not in the cache signature.** Wavelength blocks are
  independent, so threading is bit-identical by construction and a cache built
  at 1 thread must stay valid at 8. Verified: threads / wave_block / min_fibers
  all leave the hash unchanged, and a config predating these fields hashes the
  same as before — **existing caches remain valid.**
- **`combine_fused` IS in the signature**, because it shifts `Err` by ~1e-7.
  Turning it on deliberately invalidates cached galaxies.

---

## Numbers worth not re-deriving

**Thread scaling** (real pool, 176,151 fibers × 1036 wave, 12-vCPU VM), stable
across two runs:

| threads | 1 | 2 | 4 | 6 | **8** | 12 |
|---|---|---|---|---|---|---|
| speedup | 1.00× | 1.64× | 2.87× | 3.74× | **4.63×** | 4.09× |
| efficiency | 100% | 82% | 72% | 62% | 57% | 34% |

**12 threads is slower than 8.** `lscpu` shows `Thread(s) per core: 1` and
`Core(s) per socket: 1` — a VM with 12 genuine vCPUs, no hyperthreading. So the
regression is memory-bandwidth saturation / hypervisor contention, not SMT.
There is no placement fix: each vCPU is its own socket, so the guest kernel has
no real topology to schedule against.

**Fusion variants** (same real pool). Quote these, not synthetic — synthetic
Gaussians overstate the gain ~50% because `mad_std` is a bigger share of their cost:

```
A  current: biweight_location + mad_std   235.1 ms   1.00x
C  fused, axis=0                          185.0 ms   1.27x   flux exact, err 1.5e-7
D  fused, axis=1 contiguous [+transpose]  136.0 ms   1.65x   flux 1e-15, err 1.5e-7
E  fused strict, axis=1                   181.5 ms   1.25x   err exact, flux 1e-15
```

**Machine limits:** 12 usable cores; cgroup `memory.max` = **20.0 GB** against a
**17.80 GB** flux file.

---

## Things that were believed and are now false

- ~~"Drop the unused Stage-1 `Err` for a free 2×" (spec §4.1).~~ **`Err` IS
  consumed.** `stack.coadd_galaxies` feeds `cube_err` into `combine_fibers` for
  every method in `galaxy_combine_methods`, whose default includes `inv_var`
  and `mean` — both *require* it. `stack_one_galaxy` uses it for per-bin line
  S/N. Fusing supersedes the idea: same ~2× target, keeps `Err`.
- ~~"Sort the galaxy loop spatially" (spec §5.3).~~ Every galaxy already needs
  88% of the file; consecutive galaxies overlap almost totally. Nothing to recover.
- ~~"Hold the flux array resident in RAM."~~ 17.8 GB file + 1.85 GB pool +
  ~1.5 GB interpreter ≈ 21 GB against a 20 GB cgroup. **Misses by ~1 GB.**
- ~~"`block_gather` (gather one wavelength block at a time) will free the page
  cache."~~ It's a **35% regression** (311.8 → 420.6 s). A fiber spectrum is
  ~1.01 pages, so a 16-wavelength read still faults the whole page — 65× the
  page touches. This is spec §5.1's page-size trap applied to block gathering.
  It *did* cut bytes read 40% and cut combine 26%; the second effect is real and
  is what the contiguous-block copy in the new code captures for a memcpy
  instead of 65 passes over the file.
- ~~`print_profile`'s "extrapolated to 452 galaxies".~~ Hardcoded; COSMOS is 278.

---

## Two byte-level traps (if anyone reimplements the estimator)

Both cost real debugging time and both look like bugs when they bite:

1. `astropy.stats.mad_std` returns `MAD * 1.482602218505602`. Dividing by
   `0.6744897501960817` — algebraically identical — rounds differently and
   produced a **6e-8 float32 discrepancy**.
2. `biweight_location` opens with `data.astype(np.float64)`; `mad_std` does not.
   astropy therefore holds **two precisions of the same MAD**, differing ~3.4e-8
   relative. One shared value cannot equal both. That asymmetry — not the fusing
   — is the entire source of the `combine_fused` deviation. `strict=True`
   reproduces both exactly, which proves the fusing itself is lossless.

---

## Open, in priority order

1. ~~**Cache directory restructure.**~~ **DONE 2026-07-25c** — see
   "Cache + naming restructure" below.
2. ~~**Galaxy-slicing parameter.**~~ **DONE 2026-07-25c** — `gal_offset`; see
   "Galaxy slicing + Slurm array" below.
3. ~~**Script + SLURM submission.**~~ **DONE 2026-07-25c** — array job, same
   section. Job shape still matters: **processes on one node share the OS page
   cache**, so 4 processes × 8 threads on one large-memory node read the flux
   file once between them. Aim for that, not one process per core.
4. **Does gather thread?** It is `memcpy` plus page faults, both GIL-free. It is
   now ~62% of runtime (68 s of 110 s), so it is the only remaining lever here.
5. **Re-profile on TACC.** Every number in `extraction-speedup.md` is a property
   of this VM. None will survive the move.

---

## Cache + naming restructure (done 2026-07-25c)

Two changes, one idea: **derive every output name from the one signature that
already knows what changed.**

```
cache/gal_<sig>_<i>.npz                     ->  cache/<sig>/gal_<00001>.npz
galaxy_cat1a2b3c_b6_4f9e21_AEGIS_MOSDEF_    ->  galaxy_AEGIS_MOSDEF_b12_a1b2c3d4e5.fits
  kpc_biweight_segmap_bg80-90.fits
```

The FITS half fixes a real data-loss bug. `descriptive_filename` hand-listed 8
knobs while the cache hashed ~30, so changing any of the other ~22 recomputed
correctly *and wrote to a byte-identical filename* — silently overwriting the
previous extraction. Both names now come from `PipelineConfig.extract_signature`,
so a change that busts the cache necessarily changes the FITS name. Verified in
`test_framework` test 1: six knobs that used to collide now produce distinct
names, and `combine_threads` / `combine_min_fibers` / `combine_wave_block` still
change neither.

| file | change |
|---|---|
| `config.py` | `_SIGNATURE_KEYS`, `extract_signature`, `extract_signature_dict`, `resolved_run_cache_dir`; `descriptive_filename` gains `scheme=` (`'sig'` default, `'tagged'` = old spelling) |
| `extract.py` | `_extract_signature` now delegates to the config; `_write_cache_manifest`, `_save_cache_atomic`; loop writes `cache/<sig>/gal_<i:05d>.npz` |
| `io.py` | `_resolve_galaxy_fits_path` gains the signature name at the top of its fallback chain |
| `test_framework.py` | naming invariants above; also fixed a stale `MINBGFIB == 25` assert (default has been 999999) |

Notes:

- **No re-extraction.** Finished AEGIS/COSMOS FITS resolve through fallback rung
  2 with a warning. That warning is a prompt, not an all-clear — an old name
  cannot prove which knobs wrote it, which is the bug itself.
- **`_manifest.json`** in each `cache/<sig>/` decodes the hash (29 knobs + the
  FITS name it belongs to). Never read by the pipeline.
- **Atomic writes** (`.tmp.npz` → `os.replace`). Matters once Slurm array tasks
  share a `cache/<sig>/`: a task killed at walltime mid-`savez` would otherwise
  leave a truncated file that the next run accepts as a cache hit.
- Old cache is safe to move — nothing reads it: `mv cache cache_old_<date>`.
  Move the cache only, **not** all of `outputs/`, which holds the galaxy FITS.

## Galaxy slicing + Slurm array (done 2026-07-25c)

`config.gal_offset` (default 0) turns `num_gal` from a prefix into a **window**.
`config.galaxy_window(n_total)` is the single place offset+count become indices,
used by both `extract.py` and `core.py`. The cache key is `gal_start + cnt` --
the galaxy's GLOBAL run-table row -- so galaxy 70 is `gal_00070.npz` whoever
computed it.

**`gal_offset` is deliberately NOT in the signature.** It selects *which*
galaxies to compute, not *how*, so every array task hashes to the same
`cache/<sig>/`. If it entered the hash, each task would build its own directory
and the slices would never meet.

**There is no assembler.** After the array fills the cache, an ordinary
full-range run is 100% cache hits: it reads the `.npz` files and writes one
complete FITS in seconds without opening the flux file. That is
`--assemble`, and it is also just your normal notebook cell.

| file | change |
|---|---|
| `config.py` | `gal_offset` field, `galaxy_window()` |
| `extract.py` | window instead of prefix; cache key `gal_start + cnt` |
| `core.py` | `extract_core_spectra` uses the same window helper |
| `pipeline.py` | `run_extract(..., write=False)` -> cache-only, returns None; catalog aligned to the window |
| `run_extract_batch.py` | `--task-id/--ntasks` (or `$SLURM_ARRAY_TASK_ID/_COUNT`), `--assemble`; `RUN` dict replaces the inline `replace(...)` |
| `submit_extract_ls6.slurm` | `--array=0-3`, per-task walltime, resubmit-one-index notes |
| `test_framework.py` | `test_galaxy_slicing_split` -- serial vs 3-task split, bit-identical |

Verified: 4 tasks over 278 galaxies partition exactly (no overlap, full cover);
an over-provisioned array gives empty no-op tails; `gal_offset` changes neither
the cache dir nor the FITS name; and a 3-task split reassembles bit-identically
to a serial run across Spec/Err/NFIB/Background/VR.

Slices are **contiguous, not strided** -- consecutive galaxies overlap ~totally
in the fibers they touch, so a contiguous block reuses the page cache while
striding would make every task read the whole file.

Still open: (4) does gather thread, (5) re-profile on TACC.

## The data doubling, and why TACC stops being optional

If the flux file goes 17.8 → 35–40 GB with ~2× the fiber-exposures:

```
combine   194 s -> 388 s serial -> 84 s at 8 threads
gather     68 s -> 136 s (optimistic)
TOTAL     110 s -> ~220 s/gal  =  ~17 hr for COSMOS alone
```

The runtime is survivable. **The memory is the problem.** Bin 11's pool goes
1.85 → **3.98 GB**, and 8 threads of float64 temporaries go 2.5 → **4.9 GB** —
~10.5 GB of a 20 GB cgroup. It fits, but it leaves nothing for page cache, so
gather degrades worse than 2×, and one config change away is an OOM.

So the doubling doesn't make JupyterHub slow, it makes it **marginal**. That is
the argument for TACC — headroom, not speed. And TACC fixes both axes at once:
enough RAM that a 35 GB file simply stays page-cached (gather stops being I/O
at all, which is the one thing that cannot be fixed here), plus real cores.

---

## Method notes

- **Measure on real pools, not synthetic ones.** Synthetic Gaussians reported
  1.9× for fusion; the real pool gave 1.27×, because `mad_std` is ~28% of the
  cost on real fibers (5.7% NaN) versus ~50% on noise. The synthetic number was
  quoted in the spec for several hours before the real pool corrected it.
- **Separate changes by risk, not by convenience.** Threading (bit-identical,
  4.63×) and fusion (1e-7 deviation, 1.27×) were nearly shipped as one change,
  which would have gated a zero-risk 4.6× behind a judgement call about float
  precision. Splitting them meant the safe one could land unconditionally.
- **Watch for Amdahl before optimising further.** Fusion is only ~6% of *total*
  runtime once the combine is threaded down to 42 s of 110 s. Even an instant
  combine would only reach ~7.4 hr. The combine work is finished.
