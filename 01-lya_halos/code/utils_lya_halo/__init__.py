"""
utils_lya_halo -- Lyman-alpha halo stacking pipeline.

Three stages separated by on-disk data products:
    Stage 1 EXTRACT (run once) -> galaxy FITS
    Stage 2 STACK   (fast)     -> the stacks
    Stage 3 MEASURE (fast)     -> centroids + errors

Typical use:
    from utils_lya_halo import PipelineConfig, run_extract, run_stack, run_measure

    cfg = PipelineConfig(field="AEGIS", catalog="MOSDEF")
    path   = run_extract(cfg)       # slow, run once
    stacks = run_stack(cfg, path)   # fast, re-run freely
    result = run_measure(cfg, stacks)
"""

from .config import PipelineConfig
from .io import write_galaxy_fits, read_galaxy_fits, GalaxyProduct
from .pipeline import run_extract, run_stack, run_measure, run_pipeline
from .guide import guide, pipeline_map, check_guide

__all__ = [
    "PipelineConfig",
    "write_galaxy_fits",
    "read_galaxy_fits",
    "GalaxyProduct",
    "run_extract",
    "run_stack",
    "run_measure",
    "run_pipeline",
    "guide",
    "pipeline_map",
    "check_guide",
]
