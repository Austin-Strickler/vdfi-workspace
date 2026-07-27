"""
pipeline.py -- the orchestration layer.

These four functions are the public entry points. Each takes a PipelineConfig
and wires the helpers from extract.py / stack.py / measure.py for one stage, so
a full run is run_extract -> run_stack -> run_measure (or run_pipeline for all
three).

    Stage 1: EXTRACT          Stage 2: STACK            Stage 3: MEASURE
    (I/O-bound, run once)  -> (fast, re-run freely)  -> (fast)
           |                         |                         |
      galaxy FITS              the stacks                centroids + errors
"""

from __future__ import annotations

import os

import numpy as np

from . import extract, stack, measure
from .config import PipelineConfig
from .io import write_galaxy_fits, read_galaxy_fits, load_for_stack, apply_finite_cut


def run_extract(config: PipelineConfig, table=None, use_cache: bool = True,
                descriptive_name: bool = True, write: bool = True):
    """
    Stage 1 -- EXTRACT. Returns the path to the written galaxy FITS.

    `table` optionally passes an already-cut catalog (e.g. your z<2.75 cut)
    straight through to build_run_table for manual/stage-by-stage runs.

    `write=False` extracts and fills the per-galaxy cache but writes NO FITS,
    returning None. That is the mode a Slurm array task runs in: each task
    computes its `config.gal_offset` slice into the shared cache/<sig>/ and
    stops, because a quarter of the galaxies is not a product anyone wants on
    disk. Assembly is then just a normal `write=True` run over the full range --
    every galaxy is a cache hit, so it reads the .npz files and writes one
    complete FITS in seconds without touching the flux file.
    """
    if config.normalized_field() == "BOTH":
        raise ValueError(
            "run_extract cannot extract field='BOTH'. Extract 'AEGIS' and "
            "'COSMOS' separately; combine them at Stage 2 by setting field='BOTH' "
            "and calling run_stack."
        )

    vdfi_data = extract.load_field(config)
    run_table = extract.build_run_table(config, table=table)
    bad_fiber_mask = extract.build_bad_fiber_mask(config, vdfi_data, run_table=run_table)

    # return_extras=True always: the dict is empty unless this config asked for
    # background-variant products, so there is nothing to gate on and one code
    # path is better than two.
    spec, err, nfib, background, vr, extras = extract.extract_fibers(
        config, vdfi_data, run_table, bad_fiber_mask, use_cache=use_cache,
        return_extras=True,
    )

    if not write:
        return None

    # extract_fibers honors the gal_offset/num_gal window; align the catalog to
    # the galaxy axis using the SAME window so row i of CATALOG is galaxy i.
    gal_start, _ = config.galaxy_window(len(run_table))
    ngal = spec.shape[0]
    catalog = run_table[gal_start:gal_start + ngal]

    # VR_EDGES stores the bin DEFINITION in bin_mode units (uniform across
    # galaxies). For virial mode the per-galaxy kpc edges differ and are
    # recoverable from CATALOG mass+z; `vr` (ngal, nrad+1) holds them if you
    # later decide to add a per-galaxy HDU to the contract.
    r_edges = np.asarray(config.bins, dtype=float)

    os.makedirs(config.output_dir, exist_ok=True)
    if descriptive_name:
        fname = config.descriptive_filename()
    else:
        fname = f"galaxy_{config.normalized_field()}_{config.catalog}.fits"
    path = os.path.join(config.output_dir, fname)
    source_id = os.path.join(config.resolve_vdfi_base_path(), "all_flux_final.fits")

    return write_galaxy_fits(
        path, spec=spec, err=err, nfib=nfib, wave=vdfi_data["wave"],
        r_edges=r_edges, catalog=catalog, config=config,
        source_id=source_id, background=background, extras=extras, overwrite=True,
    )


def run_stack(config, galaxy_fits_path=None, keep_cube=True, verbose=True,
              weights=None):
    """
    Stage 2 driver. Loads the galaxy product (or reads galaxy_fits_path) and
    builds the stacks. `weights` (ngal, nrad) is forwarded to build_stacks for
    the 'weighted_median' combine; None -> product.nfib (the default).
    """
    if galaxy_fits_path is not None:
        product = read_galaxy_fits(galaxy_fits_path)
    else:
        product = load_for_stack(config, verbose=verbose)
    return stack.build_stacks(config, product, keep_cube=keep_cube, weights=weights)


def run_measure(config: PipelineConfig, stacks: dict) -> dict:
    """
    Stage 3 -- MEASURE.

    Bootstraps the Lya centroid (+ blue/red side ratio) and the per-pixel stack
    error from the Stage-2 per-galaxy cube, driven by config (line_window,
    n_bootstrap, LYA_REST). Requires stacks built with keep_cube=True.
    """
    return measure.measure_all_bins(config, stacks)


def run_pipeline(config: PipelineConfig) -> dict:
    """Convenience driver: EXTRACT -> STACK -> MEASURE."""
    galaxy_fits = run_extract(config)
    stacks = run_stack(config, galaxy_fits)
    return run_measure(config, stacks)
