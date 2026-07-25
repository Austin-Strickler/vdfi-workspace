"""
run_extract_batch.py -- non-interactive entry point for Stage-1 extraction,
meant to be submitted as a Slurm batch job (see submit_extract_ls6.slurm).

This is the exact `run_extract(replace(cfg, ...))` call from
0_Extractor_Galaxies2.ipynb, moved out of the notebook so it can run headless
on a compute node with no kernel attached. Edit the RUN block below the same way
you'd edit that notebook cell -- everything else here is just plumbing.

Must be run from this directory (01-lya_halos/code) since catalog_path and
run_config's paths are relative to it -- the sbatch script cd's here first.


TWO PHASES
----------
Extraction can be split so it runs as a Slurm ARRAY -- N tasks in parallel, each
computing its own slice of galaxies into the SHARED cache/<sig>/ directory:

  1. FILL  (parallel, the expensive part)
       --task-id I --ntasks N     one array task; computes its slice, writes no FITS
  2. ASSEMBLE  (serial, seconds)
       --assemble                 reads the finished cache, writes ONE galaxy FITS

Phase 2 is not a special mode -- it is an ordinary full-range run in which every
galaxy is already cached, so it never opens the flux file. Running your normal
notebook cell instead does exactly the same thing.

Usage:
    python run_extract_batch.py --smoke 3                 # test: 3 galaxies, writes FITS
    python run_extract_batch.py                           # serial: all galaxies, writes FITS
    python run_extract_batch.py --task-id 2 --ntasks 4    # array task 2 of 4 (cache only)
    python run_extract_batch.py --assemble                # gather cache -> one FITS
"""
import argparse
import math
import os
import sys
import time
from dataclasses import replace

from astropy.table import Table

from run_config import cfg
from utils_lya_halo import run_extract
from utils_lya_halo.extract import build_run_table

LYA_REST = 1215.67

CATALOG_PATH = "catalogs/lya_halo_catalog_cuts/lya_mosdef_catalog_KEEP.fits"

# ---------------------------------------------------------------------
# THE RUN -- edit this exactly as you would the notebook cell. Everything
# below it is plumbing; num_gal / gal_offset come from the CLI, not from here.
# ---------------------------------------------------------------------
RUN = dict(
    field           = "AEGIS",
    catalog         = "MOSDEF",
    catalog_path    = CATALOG_PATH,
    bins            = [0, 10, 16, 24, 36, 60, 100, 180, 350, 600, 1000, 2000, 3000],
    bin_mode        = "kpc",
    mask_method     = "segmap",
    id_col          = "ID",
    bg_inner_arcsec = 80.0,
    bg_outer_arcsec = 90.0,
    min_bg_fibers   = 9999999,
    fiber_combine_method   = "biweight",
    bg_smooth_method= "interp_gauss",
    bg_smooth_sigma_A      = 30.0,
    line_mask_halfwidth_A  = 10.0,
    mass_col        = "MASS_50",
    z_col           = "z",
    rest_delta      = 0.2,
    rest_wave_min   = 1100,
    rest_wave_max   = 1700,
    rest_density    = True,
    cut_radial_bin  = -1,
    min_good_wave   = 100,
    line_window     = [LYA_REST - 4, LYA_REST + 4],
    n_bootstrap     = 1000,
)


def _slice_for_task(base_cfg, catalog, task_id: int, ntasks: int):
    """
    Contiguous, even split of the run table across `ntasks` array tasks.
    Returns (start, count, n_total).

    Contiguous rather than strided (task i taking rows i, i+N, i+2N, ...)
    because consecutive galaxies overlap almost totally in the fibers they
    touch -- a contiguous block is the layout that reuses the OS page cache.
    Striding would make every task read essentially the whole file.

    The tail task takes a short slice, and a task whose start is past the end
    gets count=0 (harmless no-op) rather than an error, so an over-provisioned
    array fails nothing.
    """
    n_total = len(build_run_table(base_cfg, table=catalog))
    per = math.ceil(n_total / ntasks)
    start = task_id * per
    count = max(min(per, n_total - start), 0)
    return start, count, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", type=int, default=None,
                   help="only extract this many galaxies (test mode)")
    p.add_argument("--task-id", type=int, default=None,
                   help="Slurm array index (0-based). With --ntasks, selects this "
                        "task's slice and writes NO FITS -- cache only. "
                        "Defaults to $SLURM_ARRAY_TASK_ID when set.")
    p.add_argument("--ntasks", type=int, default=None,
                   help="total number of array tasks the run is split across. "
                        "Defaults to $SLURM_ARRAY_TASK_COUNT when set.")
    p.add_argument("--assemble", action="store_true",
                   help="gather a finished cache into one galaxy FITS. Full range, "
                        "cache hits only -- seconds, never reads the flux file.")
    args = p.parse_args()

    # Slurm exports these; the flags exist so the same script runs by hand.
    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    ntasks = args.ntasks
    if ntasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        ntasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    if (task_id is None) != (ntasks is None):
        p.error("--task-id and --ntasks must be given together "
                "(or both come from the Slurm array environment)")
    if args.assemble and task_id is not None:
        p.error("--assemble is the serial gather step; don't combine it with --task-id")

    catalog = Table.read(CATALOG_PATH, format="fits")
    print(f"Loaded catalog: {len(catalog)} rows", flush=True)

    base = replace(cfg, **RUN)
    write = True
    offset, count = 0, args.smoke          # smoke -> first N; None -> all

    if task_id is not None:
        offset, count, n_total = _slice_for_task(base, catalog, task_id, ntasks)
        write = False                      # a slice is not a product; see --assemble
        print(f"ARRAY TASK {task_id} of {ntasks}: galaxies "
              f"[{offset}, {offset + count}) of {n_total} -> cache only", flush=True)
        if count == 0:
            print("empty slice (array over-provisioned) -- nothing to do", flush=True)
            return 0
    elif args.assemble:
        print("ASSEMBLE: full range from cache -> galaxy FITS", flush=True)

    run_cfg = replace(base, num_gal=count, gal_offset=offset)
    print(f"cache dir : {run_cfg.resolved_run_cache_dir()}", flush=True)
    if write:
        print(f"will write: {run_cfg.descriptive_filename()}", flush=True)

    t0 = time.time()
    path = run_extract(run_cfg, write=write)
    dt = time.time() - t0

    if path:
        print(f"\nDone in {dt/3600:.2f} hr -> {path}", flush=True)
    else:
        print(f"\nCached {count} galaxies in {dt/3600:.2f} hr. Run with --assemble "
              f"(or your notebook cell) to write the FITS.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
