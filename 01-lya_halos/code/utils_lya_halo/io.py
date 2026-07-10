"""
io.py -- the Stage 1 -> Stage 2 data contract (galaxy FITS).

The galaxy FITS is the interface between extraction and stacking. Its galaxy
axis ordering is identical across all HDUs and matches the CATALOG table row
order; that alignment IS the interface.

    HDU         shape                 contents
    SPEC        (ngal, nrad, nwave)   background-subtracted spectra
    ERROR       (ngal, nrad, nwave)   propagated errors
    NFIB        (ngal, nrad)          fibers per bin
    BACKGROUND  (ngal, nwave)         galaxy-level background (diagnostic)
    WAVE        (nwave,)              observed wavelength grid
    R_EDGES    (nrad+1,)             radial bin edges actually used (bin_mode units)
    CATALOG     table                 full per-galaxy catalog (supersedes z/mass)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.table import Table

if TYPE_CHECKING:                      # avoid importing config at runtime
    from .config import PipelineConfig


# =====================================================================
# WRITE
# =====================================================================

def write_galaxy_fits(
    path: str,
    *,
    spec: np.ndarray,          # (ngal, nrad, nwave)
    err: np.ndarray,           # (ngal, nrad, nwave)
    nfib: np.ndarray,          # (ngal, nrad)
    wave: np.ndarray,          # (nwave,)
    r_edges: np.ndarray,      # (nrad+1,)
    catalog: Table,            # row order matches the galaxy axis above
    config: "PipelineConfig",
    source_id: str,
    background: np.ndarray | None = None,   # (ngal, nwave); optional diagnostic
    overwrite: bool = False,
) -> str:
    """
    Assemble and write the per-field galaxy FITS.

    NOTE: the galaxy axis of `spec`/`err`/`nfib` and the row order of `catalog`
    are assumed to be the same slice. We do not enforce len(catalog) == ngal
    here so that smoke tests (num_gal small) write cleanly. A real misalignment
    surfaces loudly in Stage 2 when redshifts no longer match the spectra.
    """
    ngal = spec.shape[0]

    primary = fits.PrimaryHDU()
    for k, v in config.to_header_dict(ngal, source_id).items():
        primary.header[k] = v

    cat_hdu = fits.table_to_hdu(catalog)
    cat_hdu.name = "CATALOG"

    hdus = [
        primary,
        fits.ImageHDU(spec.astype(np.float32), name="SPEC"),
        fits.ImageHDU(err.astype(np.float32), name="ERROR"),
        fits.ImageHDU(nfib.astype(np.int32), name="NFIB"),
    ]
    if background is not None:
        hdus.append(fits.ImageHDU(background.astype(np.float32), name="BACKGROUND"))
    hdus += [
        fits.ImageHDU(np.asarray(wave, dtype=np.float32), name="WAVE"),
        fits.ImageHDU(np.asarray(r_edges, dtype=np.float32), name="R_EDGES"),
        cat_hdu,
    ]
    hdul = fits.HDUList(hdus)

    abs_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(abs_dir, exist_ok=True)
    hdul.writeto(path, overwrite=overwrite)
    return path


# =====================================================================
# READ
# =====================================================================

@dataclass
class GalaxyProduct:
    """In-memory view of the galaxy FITS (Stage 2 reads this)."""
    spec: np.ndarray
    err: np.ndarray
    nfib: np.ndarray
    wave: np.ndarray
    r_edges: np.ndarray
    catalog: Table
    header: "fits.Header"
    background: "np.ndarray | None" = None   # present only if the file has it

    @property
    def ngal(self) -> int:
        return self.spec.shape[0]

    @property
    def nrad(self) -> int:
        return self.spec.shape[1]

    @property
    def nwave(self) -> int:
        return self.spec.shape[2]


def read_galaxy_fits(path: str) -> GalaxyProduct:
    """Load the galaxy FITS back into a GalaxyProduct."""
    with fits.open(path) as hdul:
        names = {hdu.name for hdu in hdul}
        background = hdul["BACKGROUND"].data if "BACKGROUND" in names else None
        return GalaxyProduct(
            spec=hdul["SPEC"].data,
            err=hdul["ERROR"].data,
            nfib=hdul["NFIB"].data,
            wave=hdul["WAVE"].data,
            r_edges=hdul["R_EDGES"].data,
            catalog=Table(hdul["CATALOG"].data),
            header=hdul[0].header,
            background=background,
        )


# =====================================================================
# STAGE 2 LOADING  --  field selection (incl. BOTH), concatenation, cut
# =====================================================================

def _galaxy_fits_path(config, field: str) -> str:
    """
    Canonical (current) path Stage 2 writes/expects for `field` -- the
    catalog- and bins-tagged descriptive name. Used for writing and for messages.
    """
    import os
    from dataclasses import replace
    cfg_field = replace(config, field=field)
    return os.path.join(config.output_dir, cfg_field.descriptive_filename())


def _header_bins_match(path: str, config, field: str) -> bool:
    """
    True if the galaxy FITS at `path` was written with the same radial-bin
    definition (BINMODE + BINS edges) as `config`. Used to validate older,
    pre-bins-tag filenames before reusing them, so a file whose name doesn't
    encode its bins can't be silently loaded for a different bin assortment.
    Reads only the primary header (cheap). On any parse failure, returns False
    (fail safe -> treat as a non-match and re-extract).
    """
    from dataclasses import replace
    try:
        hdr = fits.getheader(path, 0)
        if str(hdr.get("BINMODE", "")).lower() != replace(config, field=field).bin_mode.lower():
            return False
        file_edges = [float(x) for x in str(hdr.get("BINS", "")).split(",") if x != ""]
        cfg_edges = [float(b) for b in config.bins]
        return (len(file_edges) == len(cfg_edges)
                and np.allclose(file_edges, cfg_edges, rtol=0, atol=1e-9))
    except Exception:
        return False


def _resolve_galaxy_fits_path(config, field: str, verbose: bool = True):
    """
    Return an EXISTING galaxy-FITS path for `field`, or None.

    Prefers the current catalog- AND bins-tagged name. If absent, falls back to
    older names written by earlier pipeline versions, newest-scheme first:
      1. catalog+bins tagged   (current)            -- name encodes bins, used as-is
      2. catalog tagged only   (pre-bins-tag)       -- bins NOT in name; only used
                                                       if the header BINS matches config
      3. untagged              (legacy, pre-catalog-tag) -- same header-BINS guard
    The header guard on (2)/(3) is what makes adding the bins tag safe for files
    made before it existed: a stale file with different bins is skipped (so you
    re-extract cleanly) instead of being loaded as if it matched. A one-line
    warning is printed whenever a fallback name is used.
    """
    import os
    from dataclasses import replace
    cfg_field = replace(config, field=field)

    canonical = os.path.join(
        config.output_dir, cfg_field.descriptive_filename(catalog_tag=True, bins_tag=True))
    if os.path.exists(canonical):
        return canonical

    # Older naming schemes: bins are not encoded in the name, so verify the
    # file's header bins match this config before trusting it.
    fallbacks = [
        cfg_field.descriptive_filename(catalog_tag=True, bins_tag=False),   # pre-bins-tag
        cfg_field.descriptive_filename(catalog_tag=False, bins_tag=False),  # legacy
    ]
    for name in fallbacks:
        p = os.path.join(config.output_dir, name)
        if not os.path.exists(p):
            continue
        if not _header_bins_match(p, config, field):
            if verbose:
                print(f"NOTE: found older galaxy FITS '{name}' but its BINS do not "
                      f"match the current config bins -- skipping it. Re-run Stage 1 "
                      f"to write '{os.path.basename(canonical)}'.")
            continue
        if verbose:
            print(f"WARNING: using older galaxy FITS '{name}' (its bins match the "
                  f"current config). Re-run Stage 1 to regenerate it under the "
                  f"current name '{os.path.basename(canonical)}'.")
        return p
    return None


def _concat_products(a: GalaxyProduct, b: GalaxyProduct) -> GalaxyProduct:
    """Concatenate two products along the galaxy axis, checking compatibility."""
    if not np.allclose(a.wave, b.wave, equal_nan=True):
        raise ValueError("Cannot combine fields: WAVE grids differ.")
    if a.spec.shape[1] != b.spec.shape[1]:
        raise ValueError(
            f"Cannot combine fields: different nrad ({a.spec.shape[1]} vs {b.spec.shape[1]})."
        )
    if not np.allclose(a.r_edges, b.r_edges, equal_nan=True):
        raise ValueError("Cannot combine fields: R_EDGES differ.")

    # Warn (don't fail) if the two fields were extracted with different settings.
    # R_EDGES already guarantees bins match; this catches mask/combine/bg drift.
    if a.header is not None and b.header is not None:
        for key in ("BINMODE", "MASKMETH", "MASKPCT", "COMBINE", "BGCOMB",
                    "BGINNER", "BGOUTER", "MINBGFIB", "SMOOTHBG", "SMOOTHV", "CATALOG"):
            va, vb = a.header.get(key), b.header.get(key)
            if va is not None and vb is not None and va != vb:
                print(f"WARNING: combining fields with different {key}: "
                      f"{va!r} (first) vs {vb!r} (second).")

    from astropy.table import vstack
    bg = None
    if a.background is not None and b.background is not None:
        bg = np.concatenate([a.background, b.background], axis=0)
    return GalaxyProduct(
        spec=np.concatenate([a.spec, b.spec], axis=0),
        err=np.concatenate([a.err, b.err], axis=0),
        nfib=np.concatenate([a.nfib, b.nfib], axis=0),
        wave=a.wave,
        r_edges=a.r_edges,
        catalog=vstack([a.catalog, b.catalog]),
        header=a.header,
        background=bg,
    )


def load_for_stack(config, verbose: bool = True) -> GalaxyProduct:
    """
    Load the galaxy FITS for Stage 2, honoring config.field:

      'AEGIS' / 'COSMOS' -> that one file
      'BOTH'             -> AEGIS + COSMOS concatenated. If one is missing,
                            fall back to whichever exists (AEGIS preferred),
                            printing a warning.

    Then apply the finite-spectrum cut: keep galaxies with at least
    config.min_good_wave finite pixels in config.cut_radial_bin.
    """
    import os
    nf = config.normalized_field()

    def _read_if_exists(field_name):
        p = _resolve_galaxy_fits_path(config, field_name, verbose=verbose)
        return read_galaxy_fits(p) if p is not None else None

    if nf == "BOTH":
        a = _read_if_exists("AEGIS")
        c = _read_if_exists("COSMOS")
        if a is not None and c is not None:
            product = _concat_products(a, c)
            if verbose:
                print(f"Loaded BOTH: {a.ngal} AEGIS + {c.ngal} COSMOS = {product.ngal}")
        elif a is not None:
            product = a
            print("WARNING: COSMOS galaxy FITS not found; falling back to AEGIS only.")
        elif c is not None:
            product = c
            print("WARNING: AEGIS galaxy FITS not found; falling back to COSMOS only.")
        else:
            raise FileNotFoundError(
                f"Neither AEGIS nor COSMOS galaxy FITS found in {config.output_dir} "
                f"for catalog {config.catalog}."
            )
    else:
        p = _resolve_galaxy_fits_path(config, nf, verbose=verbose)
        if p is None:
            raise FileNotFoundError(
                f"Galaxy FITS not found in {config.output_dir} for field {nf}, "
                f"catalog {config.catalog}. Looked for "
                f"'{os.path.basename(_galaxy_fits_path(config, nf))}' "
                f"(and the legacy pre-tag name). Run Stage 1 (run_extract) first."
            )
        product = read_galaxy_fits(p)

    return apply_finite_cut(product, config.cut_radial_bin, config.min_good_wave, verbose)


def apply_finite_cut(product: GalaxyProduct, cut_radial_bin: int,
                     min_good_wave: int, verbose: bool = True) -> GalaxyProduct:
    """
    Keep galaxies with MORE THAN min_good_wave finite pixels in cut_radial_bin
    (strict >; a galaxy with exactly min_good_wave finite pixels is dropped).
    Slices every galaxy-axis array and the catalog consistently.
    """
    n_finite = np.sum(np.isfinite(product.spec), axis=2)   # (ngal, nrad)
    nrad = product.spec.shape[1]
    if not (-nrad <= cut_radial_bin < nrad):
        raise ValueError(f"cut_radial_bin={cut_radial_bin} invalid for nrad={nrad}")
    sel = n_finite[:, cut_radial_bin] > min_good_wave

    if verbose:
        b = cut_radial_bin if cut_radial_bin >= 0 else nrad + cut_radial_bin
        print(f"Finite-spectrum cut (bin {b}, > {min_good_wave} px): "
              f"kept {int(sel.sum())} / {len(sel)} galaxies")

    bg = product.background[sel] if product.background is not None else None
    return GalaxyProduct(
        spec=product.spec[sel],
        err=product.err[sel],
        nfib=product.nfib[sel],
        wave=product.wave,
        r_edges=product.r_edges,
        catalog=product.catalog[sel],
        header=product.header,
        background=bg,
    )
