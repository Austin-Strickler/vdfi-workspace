"""
masking.py -- MERGED INTO extract.py (2026-08-03). Compatibility shim only.

Every function that used to live here now lives in extract.py, under the
"FIBER MASKING" section. Nothing was renamed and nothing changed behaviourally;
the code was moved verbatim so that Stage 1.1 masking sits in the same module
as build_bad_fiber_mask, its only real caller.

This file re-exports those names so any lingering `from .masking import X`
keeps working. Write new code against extract.py directly -- this shim is a
safety net for the transition, not a supported import path.

    OLD                              NEW
    from .masking import X    ->     from .extract import X
"""
from __future__ import annotations

from .extract import (          # noqa: F401  (re-export shim)
    flag_fibers_on_mask,
    segmentation_mask,
    make_spectral_continuum_mask,
    apply_protect_radius,
    continuum_image_mask,
    make_circular_structure,
    get_pixscale_arcsec,
    nan_safe_gaussian_smooth,
)

__all__ = [
    "flag_fibers_on_mask",
    "segmentation_mask",
    "make_spectral_continuum_mask",
    "apply_protect_radius",
    "continuum_image_mask",
    "make_circular_structure",
    "get_pixscale_arcsec",
    "nan_safe_gaussian_smooth",
]
