"""
optics.py -- MERGED INTO extract.py (2026-08-03). Compatibility shim only.

Greg's PSF / DAR / continuum helpers now live in extract.py, under the
"OPTICS" section. Nothing was renamed and nothing changed behaviourally; the
code was moved verbatim.

SCOPE NOTE (unchanged): per pipeline_plan_v2.md the *binned* extraction does
NOT use Moffat point-source weighting or the DAR terms. These are the
PSF-weighted optimal-extraction path for the deferred core-extraction /
escape-fraction work. extract.load_vdfi_field touches only
moffat_psf_integration, and only when build_psf=True.

This file re-exports those names so any lingering `from .optics import X` or
`from . import optics` keeps working. Write new code against extract.py
directly -- this shim is a safety net for the transition, not a supported
import path.

    OLD                             NEW
    from .optics import X    ->     from .extract import X
"""
from __future__ import annotations

from .extract import (          # noqa: F401  (re-export shim)
    manual_convolution,
    pca_fit,
    get_continuum,
    wavelength_corrected_seeing,
    moffat_psf_integration,
    get_spectrum_exposure,
)

__all__ = [
    "manual_convolution",
    "pca_fit",
    "get_continuum",
    "wavelength_corrected_seeing",
    "moffat_psf_integration",
    "get_spectrum_exposure",
]
