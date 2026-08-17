"""
smoothing.py -- MERGED INTO background.py (2026-08-03). Compatibility shim only.

The background smoothers now live in background.py, under the "BACKGROUND
SMOOTHERS" section. Nothing was renamed and nothing changed behaviourally; the
code was moved verbatim. They sit in background.py rather than extract.py
because background.py already owns every other background choice.

IMPORT-DIRECTION WARNING. background.py imports extract.py at module scope, so
extract.py must NOT import the smoothers at ITS module scope -- that closes a
circular import. extract_fibers._maybe_smooth pulls them in lazily instead. If
you are tempted to hoist that import, don't; `import utils_lya_halo` will fail.

This file re-exports the names so any lingering `from .smoothing import X`
keeps working. Note that importing THIS shim drags in background.py (and hence
extract.py and measure.py), which the old smoothing.py did not -- another
reason to migrate call sites rather than lean on the shim.

    OLD                                NEW
    from .smoothing import X    ->     from .background import X
"""
from __future__ import annotations

from .background import (      # noqa: F401  (re-export shim)
    smooth_interp_then_gauss,
    smooth_spectrum_nan_safe_adaptive,
    smooth_fill_then_gauss,
    _patch_other_gaps,
)

__all__ = [
    "smooth_interp_then_gauss",
    "smooth_spectrum_nan_safe_adaptive",
    "smooth_fill_then_gauss",
]
