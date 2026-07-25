#!/bin/bash
# submit_extract_ls6.slurm -- Lonestar6 ARRAY submission for the Stage-1
# AEGIS/MOSDEF extraction run (run_extract_batch.py).
#
# Submit from THIS directory (01-lya_halos/code) so relative paths resolve:
#   cd .../01-lya_halos/code
#   sbatch submit_extract_ls6.slurm            # phase 1: fill the cache
#   python run_extract_batch.py --assemble     # phase 2: write the FITS
#
# Before submitting the real run, smoke-test on the development queue
# (see the notes at the bottom of this file).

#SBATCH -J extract_aegis
#SBATCH -o logs/extract_%A_%a.out    # %A = array job id, %a = task index
#SBATCH -e logs/extract_%A_%a.err
#SBATCH -p normal                    # 'development' (2 hr cap) for smoke tests only
#SBATCH -N 1                         # ONE node per task
#SBATCH -n 1                         # one process per task; parallelism is INSIDE it
                                     #   (cfg.combine_threads), plus across array tasks
#SBATCH --array=0-3                  # 4 tasks => ~70 galaxies each. Must match the
                                     #   split run_extract_batch.py computes; it reads
                                     #   $SLURM_ARRAY_TASK_COUNT, so changing this line
                                     #   alone is enough -- nothing else to keep in sync.
#SBATCH -t 06:00:00                  # PER TASK walltime. The job is KILLED at this
                                     #   limit -- set it above your worst-case estimate
                                     #   for ONE SLICE, not the whole run. 8.5 hr serial
                                     #   / 4 tasks ~ 2.2 hr, so 6 hr is a safe margin.
#SBATCH -A <YOUR_TACC_PROJECT>       # exact project/allocation name -- check `taccinfo`
                                     #   or portal.tacc.utexas.edu; submission fails
                                     #   without a valid one

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# --- match whatever environment your JupyterHub kernel actually uses ---
# Check with `which python` / `conda env list` from a JupyterHub terminal
# first, then mirror it here. Uncomment ONE of these:
#
# module load python3
# conda activate <your-env-name>
#
# source /path/to/your/venv/bin/activate

# No --task-id/--ntasks needed: the script reads SLURM_ARRAY_TASK_ID and
# SLURM_ARRAY_TASK_COUNT from the environment Slurm exports here.
python run_extract_batch.py

# ============================================================================
# AFTER THE ARRAY FINISHES -- the cache is full but no FITS exists yet. Run:
#
#   python run_extract_batch.py --assemble
#
# on a login node or in idev (it is seconds of cache reads, not a compute job),
# or just re-run your normal 0_Extractor_Galaxies2.ipynb cell, which does the
# same thing. Check completeness first if you like:
#
#   ls outputs/cache/<sig>/gal_*.npz | wc -l     # expect one per galaxy
#
# A task that hit its walltime leaves its slice short. Resubmit ONLY that index
# -- every other galaxy is already cached, so it picks up where it stopped:
#
#   sbatch --array=2 submit_extract_ls6.slurm
#
# ============================================================================
# TESTING FIRST (do this before ever submitting the block above):
#
#   idev -p development -N 1 -n 1 -m 30
#   cd .../01-lya_halos/code
#   python run_extract_batch.py --smoke 3
#
# Confirms imports, paths, and the environment all work end-to-end in a
# non-notebook process, on real hardware, before committing walltime.
#
# Then test the SPLIT itself without burning hours -- run two tasks of a
# deliberately huge array so each slice is tiny, and confirm the cache files
# they write do not collide:
#
#   python run_extract_batch.py --task-id 0 --ntasks 140
#   python run_extract_batch.py --task-id 1 --ntasks 140
#   ls outputs/cache/*/gal_*.npz          # expect gal_00000/1 AND gal_00002/3
#
# ONE NODE, FEWER TASKS is often better than many: processes on the same node
# share the OS page cache, so they read the 17.8 GB flux file once between them.
# If you can get a large-memory node, prefer 4 tasks there over 4 small nodes.
# ============================================================================
