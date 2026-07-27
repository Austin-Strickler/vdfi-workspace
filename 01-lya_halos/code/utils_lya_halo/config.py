"""
config.py -- the single source of truth for one pipeline run.

Every stage entry point takes a PipelineConfig. The provenance header
keywords in the galaxy FITS are populated directly from it (to_header_dict),
so the parameters you ran with and the provenance the file records cannot
drift apart -- they are the same object.

Pure standard library on purpose: no numpy / astropy import here, so the
config is cheap to construct and import anywhere.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field as _field, asdict
from typing import ClassVar, Optional, Sequence


# ---------------------------------------------------------------------
# Canonical Stage-3 measurement defaults (the single source of truth).
# Defined here and imported by measure.py, so its standalone per-spectrum
# helpers and the config-driven pipeline share ONE default -- change a default
# in exactly one place. These back the PipelineConfig fields of the same name,
# which are the per-run, provenance-recorded values the pipeline actually uses.
# ---------------------------------------------------------------------
# (blue, red) continuum sidebands in Angstrom, "line-safe": each band avoids the
# strong UV ISM features that would otherwise bias a stacked continuum -- the
# blue band (1195-1205) sits between Si II 1193.3 and Si III 1206.5, and the red
# band (1225-1236) stops short of N V 1238.8. Outside the inner ~2 bins the
# continuum is ~0 so this choice is a marginal (but strictly cleaner) change;
# it is a one-line edit here since the bounds live in the config.
DEFAULT_CONT_BOUNDS = ((1195, 1205), (1225, 1236))
DEFAULT_CONT_METHOD = "median"            # 'median' (flat robust level) | 'poly'
DEFAULT_CONT_ORDER = 1                     # polynomial order, used only for 'poly'
DEFAULT_CENTROID_METHOD = "flux_median"    # median50; see measure._CENTROID_DISPATCH for options


@dataclass
class PipelineConfig:
    # --- fixed physical constants (NOT run parameters) -------------------
    # ClassVar => shared by all configs, never in __init__ / as_dict, never edited.
    LYA_REST: ClassVar[float] = 1215.67   # Lyman-alpha rest wavelength, vacuum (Angstrom)

    # --- sample / inputs -------------------------------------------------
    field: str = "AEGIS"                 # 'AEGIS' | 'COSMOS' (aliases handled below)
    catalog: str = "MOSDEF"              # which CATALOG column value to select
    catalog_path: str = "../catalogs/lya_halo_catalog_cuts/lya_mosdef_catalog_KEEP.fits"
    catalog_format: str = "auto"         # 'auto' | 'fits' | 'ascii' -- 'auto' picks the
                                          #   reader from catalog_path's extension
                                          #   (.fits/.fit/.fz -> fits, else ascii); set
                                          #   explicitly only if the extension is misleading
    cont_image_path: Optional[str] = None  # required when mask_method == 'image' (legacy)
    seg_mask_path: Optional[str] = None    # required when mask_method == 'segmap':
                                           #   path to the BAKED binary mask FITS (nonzero =
                                           #   masked), pixel-identical to the field cutout and
                                           #   carrying its WCS. None / 'auto' handled like
                                           #   cont_image_path (per-field default via
                                           #   resolve_seg_mask_path).
    vdfi_base_path: Optional[str] = None   # fiber-data dir (all_info/all_flux_final/...);
                                           # None -> resolved from field below

    # --- radial binning --------------------------------------------------
    # `bins` are the edges in the unit named by `bin_mode`. For 'virial' these
    # are R/Rvir edges; per-galaxy kpc edges are derived inside Stage 1.
    bins: Sequence[float] = _field(
        default_factory=lambda: [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    )
    bin_mode: str = "virial"             # 'virial' | 'kpc' | 'arcsec'

    # --- fiber masking (Stage 1.1) --------------------------------------
    mask_method: str = "segmap"           # single method ('segmap') OR a '+'-joined
                                          #   composite ('segmap+spec_local') that OR-combines
                                          #   each component's boolean mask (bad in ANY
                                          #   component -> bad in the result). Components:
                                          #     'segmap'       baked binary mask (seg_mask_path);
                                          #                    ignores every masking knob below
                                          #                    except seg_mask_path.
                                          #     'image'        legacy percentile-image mask
                                          #                    (mask_percentile_image).
                                          #     'spec_global'  per-fiber continuum level, ONE
                                          #                    threshold pooled across all
                                          #                    exposures (mask_percentile_spec_*).
                                          #     'spec_local'   same as spec_global but threshold
                                          #                    recomputed per exposure.
                                          #   segmap catches faint real sources image-plane
                                          #   percentile masks miss; spec_* catches instrumental
                                          #   bad fibers no image-plane mask can see -- disjoint
                                          #   failure modes, hence OR-combining by default.
    mask_percentile_image: float = 90     # 'image' component ONLY; ignored otherwise
    mask_percentile_spec_high: Optional[float] = 99.5  # spec_* component ONLY: fibers ABOVE
                                          #   this percentile of per-fiber continuum are masked
                                          #   (bright/continuum leakage). None disables this tail.
    mask_percentile_spec_low: Optional[float] = 0.5    # spec_* component ONLY: fibers BELOW
                                          #   this percentile are masked (dim/noisy fibers --
                                          #   unexamined as of 2026-07). None disables this tail.
                                          #   Independent of mask_percentile_spec_high; both
                                          #   default on and OR together with it.
    mask_protect_radius_arcsec: Optional[float] = 10.0  # spec_* component ONLY: fibers within
                                          #   this radius of ANY galaxy in the run_table passed
                                          #   to build_bad_fiber_mask are force-UNMASKED in the
                                          #   spec_* component(s), regardless of percentile --
                                          #   protects real core science fibers from spectral
                                          #   percentile masking. None disables (no protection).
                                          #   Silently unused if mask_method has no spec_*
                                          #   component (segmap/image already avoid self-masking
                                          #   their own way).
    spec_mask_step: int = 5              # only used for spectral masking
    # --- catalog columns --------------------------------------------------
    id_col: str = "ID"                   # catalog column with the (usually int) galaxy ID;
                                          #   the default id_col for core.select_core_by_id and
                                          #   the plotting/diagnostic tools

    # --- background (Stage 1.3) -----------------------------------------
    bg_inner_arcsec: float = 80.0
    bg_outer_arcsec: float = 90.0
    min_bg_fibers: int = 999999
    smooth_bg: bool = True                # master on/off for background smoothing
    bg_smooth_method: str = "interp_gauss"  # 'interp_gauss' (default, fast) | 'adaptive' (legacy)
    # -- 'interp_gauss' params (mask Lya, interpolate, one uniform gaussian) --
    bg_smooth_sigma_A: float = 30.0       # uniform smoothing width (A)
    line_mask_halfwidth_A: float = 10.0   # half-width (observed A) of the masked Lya gap
    # -- 'adaptive' params (legacy ramped-sigma nan-safe smoother) --
    smoothing_values: Sequence[float] = _field(
        default_factory=lambda: [200, 20, 300]  # (sigma_A, edge_sigma_A, edge_width_A)
    )

    # --- background-variant products (specs/systematics-and-extraction.md 2.5) --
    # The extraction destroys the fiber dimension, so the pipeline can never
    # re-derive a background from a different annulus, re-mask, or re-smooth
    # after the fact. These four knobs are what let ONE run answer every
    # background question offline, forever:
    #
    #   store_raw_bg     -> the annulus coadd BEFORE the Lya mask + interpolate
    #                       + gaussian smooth. This is the single non-negotiable
    #                       product: 1.1's rest-frame stack test at 1215.67 A
    #                       cannot exist without it, because the stored
    #                       BACKGROUND HDU is post-mask/post-smooth (B9) and has
    #                       the feature interpolated away by construction.
    #   bg_variants      -> extra annuli, MEASURED AND STORED ONLY. Variant 0 is
    #                       always the fiducial (bg_inner_arcsec, bg_outer_arcsec)
    #                       and is the only one ever subtracted from science
    #                       fibers, so adding variants cannot move a science
    #                       number. Whether a Lya feature's amplitude falls with
    #                       annulus radius is what distinguishes a real two-halo
    #                       component from an instrumental artifact.
    #   store_spec_nobg  -> the science coadd with NO background subtracted, so
    #                       any variant can be re-subtracted in Stage 2 (see
    #                       resolved_bg_variants / background.reconstruct_spec
    #                       for why this is exact rather than approximate).
    #   store_bg_halves  -> each annulus split by azimuth; the two halves differ
    #                       by the background's SPATIAL variance (B10/B11).
    #
    # All default off/None, so an unchanged config hashes to an unchanged
    # extraction signature and existing caches stay valid (extract_signature_dict
    # omits them at their defaults).
    bg_variants: Optional[Sequence] = None   # None -> ((bg_inner_arcsec, bg_outer_arcsec),)
                                          #   else a sequence of (inner, outer) arcsec
                                          #   pairs whose FIRST entry must equal the
                                          #   fiducial annulus -- see
                                          #   resolved_bg_variants().
    store_raw_bg: bool = False           # write pre-mask, pre-smooth annulus coadds
    store_spec_nobg: bool = False        # write the un-background-subtracted coadd
    store_bg_halves: bool = False        # azimuthal split of each annulus

    # --- coaddition ------------------------------------------------------
    # TWO concepts (not three), because the background is SUBTRACTED from the
    # science fibers: the background estimator and the fiber estimator must be
    # the SAME statistic of the same fibers, or the subtraction leaves a
    # bin-dependent residual (improper background subtraction).
    #   fiber_combine_method   : fibers -> per-gal bin AND fibers -> background
    #                            (Stage 1.3 + 1.4; one fiber-level statistic)
    #   galaxy_combine_methods : per-gal spectra -> stacked bin
    #                            (Stage 2, galaxy-to-galaxy; independent -- runs
    #                             AFTER subtraction on clean spectra)
    # Stage 2 computes ALL listed galaxy methods in one pass (the expensive part
    # is the rest-frame resample; the combines themselves are ~free).
    fiber_combine_method: str = "biweight"   # 'biweight' | 'inv_var' | 'median' | 'mean'
    galaxy_combine_methods: Sequence[str] = _field(
        default_factory=lambda: ["biweight", "inv_var", "mean", "median",
                                 "sigma_clip", "weighted_median"]
    )
    # Background combine: None -> follow fiber_combine_method (recommended; keeps
    # background and science matched). Set explicitly only to deliberately
    # decouple them (you almost never want to).
    bg_combine_method: Optional[str] = None
    mass_col: str = "MASS_50"
    z_col: str = "z"

    # --- Stage 2 rest-frame resampling -----------------------------------
    rest_delta: float = 0.2              # rest-frame grid step (Angstrom); 0.2 is the
                                         #   point where the grid just begins to affect
                                         #   the centroid -- finer is only correlated
                                         #   oversampling of the ~0.6 A native sampling
    rest_wave_min: float = 1100          # rest-frame grid min (A); covers blue sideband
    rest_wave_max: float = 1700          # rest-frame grid max (A); covers red sideband
    flux_unit: str = "L_kpc2"            # 'raw' | 'flux_arcsec2' | 'L_fiber' | 'L_kpc2'
    flux_unit_scale: float = 1e-17       # multiplies stored flux (erg/s/cm^2/A per count)
    fiber_diam_arcsec: float = 1.5       # average-fiber aperture diameter (arcsec)
    rest_density: bool = True            # multiply flux density by (1+z) so rest-frame
                                         #   integral gives correct integrated line flux

    # --- Stage 2 sample cut ----------------------------------------------
    cut_radial_bin: int = -1             # which bin defines the finite-spectrum cut;
                                         #   -1 = outermost (~10% cut), 0 = inner (~70% cut)
    min_good_wave: int = 100             # drop galaxies with < this many finite pixels
                                         #   in cut_radial_bin

    # --- wavelength axis (defines the WAVE HDU) -------------------------
    wave_min: float = 3470
    wave_max: float = 5540
    nwave: int = 1036

    # --- run control -----------------------------------------------------
    num_gal: Optional[int] = None        # None -> all galaxies in the run table.
                                         #   With gal_offset, this is the COUNT of the
                                         #   window, not "the first N".
    gal_offset: int = 0                  # where this run's window STARTS in the run
                                         #   table. 0 (default) reproduces the old
                                         #   prefix behavior exactly. Set >0 only to
                                         #   split one extraction across Slurm array
                                         #   tasks; each task fills its own slice of
                                         #   the SHARED cache/<sig>/ directory.
                                         #   DELIBERATELY NOT in the extraction
                                         #   signature -- it selects WHICH galaxies to
                                         #   compute, not HOW, so every task must hash
                                         #   to the same cache directory or the slices
                                         #   would never meet.
    output_dir: str = "./outputs"
    cache_dir: Optional[str] = None      # per-galaxy Stage 1 cache (resumability).
                                         #   None -> <output_dir>/cache (kept inside
                                         #   outputs so a run is self-contained); set
                                         #   an explicit path only to override.
    combine_wave_block: Optional[int] = 16  # MEMORY KNOB ONLY (no effect on results):
                                         #   wavelength block size for the per-bin fiber
                                         #   combine. Smaller => lower peak RAM on huge
                                         #   outer bins, slightly more loop overhead.
                                         #   None -> single-shot (old behavior).

    # --- Stage 1 combine performance (see specs/extraction-speedup.md 9) ---
    # combine_threads is a PERFORMANCE KNOB ONLY and is deliberately EXCLUDED
    # from the cache signature, exactly like combine_wave_block: wavelength
    # blocks are independent, so threading them is bit-identical by
    # construction and a cache built at 1 thread MUST stay valid at 8.
    combine_threads: int = 1             # 1 -> serial reference path. 8 measured
                                         #   best on the 12-vCPU JupyterHub VM
                                         #   (4.63x); 12 is SLOWER than 8
                                         #   (memory bandwidth, not cores).
                                         #   Re-measure elsewhere with
                                         #   combine_fast.benchmark_threads().
    combine_min_fibers: int = 50_000     # pools smaller than this combine serially;
                                         #   thread dispatch costs more than it saves
                                         #   on the inner bins (<5 s/galaxy total).

    # The fused estimator DOES perturb Err (~1e-7 relative), so unlike the two
    # knobs above these three ARE part of the cache signature -- a cache must
    # record which estimator produced it. Biweight only; ignored otherwise.
    combine_fused: bool = False          # 1.27x. Flux stays EXACTLY bit-identical;
                                         #   Err moves ~1.5e-7 relative (astropy
                                         #   itself holds two precisions of the same
                                         #   MAD -- see extract.biweight_fused).
    combine_fused_strict: bool = False   # reproduce Err bit-for-bit too, at ~1.02x
                                         #   (the extra pass eats the gain -- only
                                         #   worthwhile combined with transpose).
    combine_fused_transpose: bool = False  # +1.3x more: reduce over contiguous
                                         #   (wave, fiber) blocks. Reassociates the
                                         #   nansum, so flux moves ~1e-15 relative.

    # --- Stage 3 line measurement ---------------------------------------
    # Continuum, centroid, and measure-stack choices live HERE (config is the
    # single source of truth) and are read by measure.measure_all_bins. The
    # module-level DEFAULT_* above back these fields and are what measure.py's
    # standalone notebook helpers use, so the default is defined in one place.
    line_window: Sequence[float] = _field(
        default_factory=lambda: [PipelineConfig.LYA_REST - 3.0,
                                 PipelineConfig.LYA_REST + 3.0]
    )                                              # +/-4 A about LYA_REST (1211.67-1219.67)
    n_bootstrap: int = 2000
    cont_bounds: Sequence = DEFAULT_CONT_BOUNDS    # (blue, red) sidebands, Angstrom
    cont_method: str = DEFAULT_CONT_METHOD         # 'median' | 'poly'
    cont_order: int = DEFAULT_CONT_ORDER           # used only when cont_method == 'poly'
    centroid_method: str = DEFAULT_CENTROID_METHOD  # 'flux_weighted'|'flux_median'|'gaussian'|'gauss_weighted'|'peak' (see measure.CENTROID_METHOD_SPECS)
    clip_negative: bool = True                     # consulted by 'flux_weighted' / 'moments'
    measure_stack_method: str = "biweight"         # galaxy combine used inside the measure bootstrap
    # sigma-clip combine controls (used when 'sigma_clip' is among the combine
    # methods). Declared as real fields so they are recorded in provenance /
    # the FITS header, not read via getattr with a silent default.
    sigma_clip_sigma: float = 3.0                  # clip threshold (sigma)
    sigma_clip_maxiters: int = 5                   # max clip iterations

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def normalized_field(self) -> str:
        """
        Collapse field aliasing into one place.
          'EGS'/'AEGIS' -> 'AEGIS', 'COS'/'COSMOS' -> 'COSMOS', 'BOTH' -> 'BOTH'.
        BOTH is a Stage 2 (stacking) concept only -- Stage 1 extraction must be
        a single field and rejects BOTH explicitly.
        """
        f = self.field.upper()
        if f in ("AEGIS", "EGS"):
            return "AEGIS"
        if f in ("COSMOS", "COS"):
            return "COSMOS"
        if f == "BOTH":
            return "BOTH"
        raise ValueError(f"Unknown field {self.field!r}")

    def resolve_vdfi_base_path(self) -> str:
        """
        Directory holding the VDFI fiber data. Explicit vdfi_base_path wins;
        otherwise fall back to the per-field default. Centralizing the map here
        keeps it out of the loader and safe when you switch fields.
        """
        if self.vdfi_base_path is not None and str(self.vdfi_base_path).lower() != "auto":
            return self.vdfi_base_path
        nf = self.normalized_field()
        if nf == "BOTH":
            raise ValueError(
                "field='BOTH' has no single VDFI path. Extract AEGIS and COSMOS "
                "separately in Stage 1; combine them at Stage 2 (run_stack)."
            )
        return {
            "AEGIS":  "../../../../HETDEX-Work/vdfi/egs",
            "COSMOS": "../../../../HETDEX-Work/vdfi/cosmos",
        }[nf]

    def resolve_cont_image_path(self) -> Optional[str]:
        """
        Continuum-image path for mask_method='image'. Mirrors
        resolve_vdfi_base_path so the per-field map lives in one place:
          None    -> None (only the image mask needs it; not an error here)
          'auto'  -> per-field default (groth for AEGIS, CFHT image for COSMOS)
          string  -> used as-is.
        build_bad_fiber_mask raises only if the RESOLVED path is still None while
        mask_method == 'image'.
        """
        if self.cont_image_path is None:
            return None
        if str(self.cont_image_path).lower() != "auto":
            return self.cont_image_path
        nf = self.normalized_field()
        if nf == "BOTH":
            raise ValueError(
                "field='BOTH' has no single continuum image; extract AEGIS and "
                "COSMOS separately in Stage 1."
            )
        return {
            "AEGIS":  "../../../../Hobby-Eberly-Telesco/imaging/candles_egs/groth/groth.fits",
            "COSMOS": "../../../../HETDEX-Work/vdfi/cosmos/CFHT_COSMOS_image.fits",
        }[nf]

    def resolve_seg_mask_path(self) -> Optional[str]:
        """
        Baked segmentation-mask path for mask_method='segmap'. Mirrors
        resolve_cont_image_path so the per-field map lives in one idiom:
          None    -> None (only the segmap mask needs it; not an error here)
          'auto'  -> per-field default (EGS mask for AEGIS, COSMOS mask for COSMOS)
          string  -> used as-is.
        build_bad_fiber_mask raises only if the RESOLVED path is still None while
        mask_method == 'segmap'.
        """
        if self.seg_mask_path is None:
            return None
        if str(self.seg_mask_path).lower() != "auto":
            return self.seg_mask_path
        nf = self.normalized_field()
        if nf == "BOTH":
            raise ValueError(
                "field='BOTH' has no single segmentation mask; extract AEGIS and "
                "COSMOS separately in Stage 1."
            )
        # EDIT THESE to the two baked-mask locations (read-only paths are fine;
        # avoid OneDrive-synced WRITE paths -- reading from OneDrive is OK, it was
        # streaming writes that crashed).
        return {
            "AEGIS":  "catalogs/images/fibermask_EGS.fits",
            "COSMOS": "catalogs/images/fibermask_COSMOS.fits",
        }[nf]

    def mask_components(self) -> list:
        """mask_method split on '+', lowercased, stripped. 'segmap' -> ['segmap'];
        'segmap+spec_local' -> ['segmap', 'spec_local']. Single source of truth
        for parsing the composite string -- extract.build_bad_fiber_mask and the
        provenance/header methods below both use this instead of re-splitting."""
        return [m.strip().lower() for m in self.mask_method.split("+") if m.strip()]

    def mask_has_spec(self) -> bool:
        """True if any component of mask_method is 'spec_global' or 'spec_local'."""
        return any(c in ("spec_global", "spec_local") for c in self.mask_components())

    def mask_has_segmap(self) -> bool:
        """True if 'segmap' is one of the (possibly composite) mask_method components."""
        return "segmap" in self.mask_components()

    def seg_mask_hash(self) -> str:
        """
        CONTENT hash (md5 of the file bytes) of the resolved segmentation mask,
        or 'noseg' when there is none. Hashing content, not the path, is
        deliberate and is the direct heir of catalog_tag()'s lesson: you WILL
        retweak the mask FITS in place, and a path hash would silently reuse
        stale cached extractions. This feeds SEGMHASH in the FITS header AND the
        per-galaxy cache signature (extract._extract_signature), so a retweaked
        mask invalidates the cache and forces a clean re-extraction.
        """
        try:
            path = self.resolve_seg_mask_path()
        except ValueError:
            path = None
        if not path or not os.path.exists(path):
            return "noseg"
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:10]

    def resolved_bg_combine_method(self) -> str:
        """Background combine method, falling back to the fiber method if None."""
        return self.bg_combine_method or self.fiber_combine_method

    def resolved_bg_variants(self) -> tuple:
        """
        The background annuli this run measures, as a tuple of (inner, outer)
        arcsec pairs. `bg_variants=None` (the default) gives exactly one entry,
        the fiducial `(bg_inner_arcsec, bg_outer_arcsec)` -- i.e. the old
        single-annulus behavior, expressed in the new vocabulary.

        VARIANT 0 IS THE FIDUCIAL, ALWAYS. It is the only annulus ever
        subtracted from science fibers; variants 1+ are measured and stored and
        never touch `Spec`. Validating that here rather than in extract.py is
        deliberate: it makes "adding a variant cannot change a science number"
        a property of the config object, so no caller can accidentally reorder
        the list and silently re-define what the fiducial background is.

        Raises ValueError on a malformed list or a first entry that disagrees
        with (bg_inner_arcsec, bg_outer_arcsec) -- fail at construction, not
        two hours into an extraction.
        """
        fiducial = (float(self.bg_inner_arcsec), float(self.bg_outer_arcsec))
        if self.bg_variants is None:
            return (fiducial,)

        out = []
        for i, pair in enumerate(self.bg_variants):
            try:
                lo, hi = (float(pair[0]), float(pair[1]))
            except (TypeError, IndexError, ValueError):
                raise ValueError(
                    f"bg_variants[{i}] = {pair!r} is not an (inner, outer) pair."
                ) from None
            if not (0.0 <= lo < hi):
                raise ValueError(
                    f"bg_variants[{i}] = ({lo}, {hi}) must satisfy 0 <= inner < outer."
                )
            out.append((lo, hi))

        if not out:
            raise ValueError("bg_variants is empty; use None for the fiducial annulus only.")
        if out[0] != fiducial:
            raise ValueError(
                f"bg_variants[0] = {out[0]} must equal the fiducial annulus "
                f"(bg_inner_arcsec, bg_outer_arcsec) = {fiducial}. Variant 0 is "
                "the annulus actually subtracted from the science fibers, so a "
                "mismatch would silently change what the extraction means."
            )
        return tuple(out)

    def stores_bg_products(self) -> bool:
        """
        True if this run writes any of the 2.5 background-variant products.
        The single predicate extract.py, io.py and extract_signature_dict all
        consult, so "is this a background-variant run?" has one definition.
        """
        return bool(self.store_raw_bg or self.store_spec_nobg
                    or self.store_bg_halves or self.bg_variants is not None)

    def galaxy_window(self, n_total: int) -> tuple:
        """
        (start, stop) row range of the run table this run should process --
        `run_table[start:stop]`. The single place `gal_offset` + `num_gal` are
        turned into indices, so extract.py and core.py cannot disagree.

        Defaults (`gal_offset=0`) give `[0, num_gal)`, i.e. exactly the old
        prefix behavior. `start` is what makes a Slurm array safe: cache entries
        are named by `start + loop_index`, the galaxy's GLOBAL row, so the task
        handling galaxies 70-139 writes gal_00070.npz instead of restarting its
        own count at gal_00000.npz and colliding with task 0.

        An offset past the end yields an empty window rather than an error, so
        an over-provisioned array (say 8 tasks for 278 galaxies at 40 each) has
        harmless no-op tasks at the tail instead of failures.
        """
        start = max(int(self.gal_offset), 0)
        if start >= n_total:
            return (n_total, n_total)
        stop = n_total if self.num_gal is None else min(start + int(self.num_gal), n_total)
        return (start, stop)

    def resolved_cache_dir(self) -> str:
        """
        Per-galaxy Stage 1 cache directory. Explicit cache_dir wins; otherwise
        it lives at <output_dir>/cache, so a run's intermediate cache and its
        final products stay together and the cache follows output_dir if that
        changes. Mirrors resolve_vdfi_base_path / resolved_bg_combine_method.
        """
        if self.cache_dir is not None:
            return self.cache_dir
        return os.path.join(self.output_dir, "cache")

    # -- extraction signature ------------------------------------------------
    # Lives here, not in extract.py, because it now names TWO things: the
    # per-galaxy cache directory AND the galaxy FITS (descriptive_filename).
    # Keeping one definition is the whole point -- the historical bug was that
    # the cache hashed ~30 knobs while the FITS name hand-listed 8, so ~22
    # settings recomputed correctly yet wrote to a byte-identical filename and
    # silently overwrote the previous extraction. Config is pure stdlib and
    # imported everywhere, so it can host this without an import cycle;
    # extract._extract_signature is now a thin delegating wrapper.
    _SIGNATURE_KEYS: ClassVar[tuple] = (
        "field", "catalog", "catalog_path", "bins", "bin_mode", "mask_method",
        "mask_percentile_image", "mask_percentile_spec_high",
        "mask_percentile_spec_low", "mask_protect_radius_arcsec",
        "spec_mask_step", "bg_inner_arcsec",
        "bg_outer_arcsec", "min_bg_fibers", "smooth_bg", "bg_smooth_method",
        "bg_smooth_sigma_A", "line_mask_halfwidth_A", "smoothing_values",
        "bg_combine_method", "fiber_combine_method", "mass_col", "z_col",
        "wave_min", "wave_max", "nwave",
    )

    def extract_signature_dict(self) -> dict:
        """
        The exact knob->value mapping that `extract_signature` hashes. Written
        verbatim to each cache directory as `_manifest.json` so a 10-hex folder
        name is decodable months later ('which run was 4f9e21a0bc?') without
        re-deriving it by hand.
        """
        d = {k: getattr(self, k) for k in self._SIGNATURE_KEYS}
        d["field"] = self.normalized_field()
        d["bg_combine_method"] = self.resolved_bg_combine_method()  # hash resolved value
        # sigma_clip knobs (optional on config): included so changing them busts
        # the cache, but read defensively so older configs without the fields
        # still hash.
        d["sigma_clip_sigma"] = getattr(self, "sigma_clip_sigma", 3.0)
        d["sigma_clip_maxiters"] = getattr(self, "sigma_clip_maxiters", 5)
        # Fused-combine knobs: these DO change Err (~1e-7) and, with transpose,
        # the flux (~1e-15), so a cache must record which estimator wrote it.
        # Read defensively and OMIT when off, so existing caches from before
        # this flag existed still hash to the same signature and stay valid.
        if getattr(self, "combine_fused", False):
            d["combine_fused"] = True
            d["combine_fused_strict"] = getattr(self, "combine_fused_strict", False)
            d["combine_fused_transpose"] = getattr(self, "combine_fused_transpose", False)
        # Background-variant products (2.5). OMITTED ENTIRELY at their defaults,
        # which is the whole point: adding these fields must leave every existing
        # signature bit-identical, or it orphans caches mid-flight (the fine-grid
        # extraction was running when this went in). test_framework asserts that.
        #
        # When they ARE set, they DO enter the hash -- not because they change
        # SPEC/ERROR/NFIB/BACKGROUND (they cannot; variant 0 is the fiducial and
        # variants 1+ are never subtracted) but because they change what a cache
        # ENTRY CONTAINS. A variant run therefore gets its own cache/<sig>/ and
        # its own FITS name, and can never collide with, or half-fill, a plain
        # run's cache. That isolation is worth recomputing SPEC for.
        if self.stores_bg_products():
            d["bg_variants"] = list(self.resolved_bg_variants())
            d["store_raw_bg"] = bool(self.store_raw_bg)
            d["store_spec_nobg"] = bool(self.store_spec_nobg)
            d["store_bg_halves"] = bool(self.store_bg_halves)
        # NOT hashed, deliberately: combine_wave_block, combine_threads,
        # combine_min_fibers -- all provably no-effect on the result, so a cache
        # built at 1 thread stays valid at 8.
        # Baked segmentation mask identity: CONTENT hash, so retweaking the mask
        # FITS in place busts the cache (path/name alone would not change).
        d["seg_mask_hash"] = self.seg_mask_hash()
        return d

    def extract_signature(self) -> str:
        """
        10-hex hash of every Stage-1 knob that can change the extracted numbers.
        Names the cache directory (`cache/<sig>/`) and closes the galaxy FITS
        (`descriptive_filename`), so any change that triggers a recompute also
        changes the output filename -- overwriting a previous extraction becomes
        impossible by construction rather than by remembering to update a list.
        """
        blob = json.dumps(self.extract_signature_dict(), sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:10]

    def resolved_run_cache_dir(self) -> str:
        """
        Per-RUN cache directory: `<resolved_cache_dir>/<extract_signature>`.

        The old layout wrote every galaxy of every config ever run flat into one
        folder as `gal_<sig>_<i>.npz`, which grew until listing it took minutes.
        One subdirectory per signature keeps a listing scoped to a single config
        and makes retiring a dead config `rm -rf cache/<sig>/` instead of a glob
        over tens of thousands of files.
        """
        return os.path.join(self.resolved_cache_dir(), self.extract_signature())

    def config_hash(self) -> str:
        """Short, stable hash of the full config (every field), so any result
        stamped with it can be matched back to the exact run that produced it."""
        payload = repr(sorted(asdict(self).items()))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def provenance(self) -> dict:
        """
        Compact, human-readable record of the knobs that shape a result, plus a
        config hash. Used by stamp_provenance to tag Stage-2/3 and validation
        outputs so figures and saved dicts are self-documenting (the galaxy FITS
        already carries to_header_dict; this is its in-memory analogue for the
        downstream, re-runnable products).
        """
        return {
            "config_hash": self.config_hash(),
            "field": self.normalized_field(),
            "catalog": self.catalog,
            "catalog_tag": self.catalog_tag(),
            "bins_tag": self.bins_tag(),
            "bins": list(self.bins),
            "bin_mode": self.bin_mode,
            "mask_method": self.mask_method,
            "mask_percentile_image": self.mask_percentile_image,
            "mask_percentile_spec_high": self.mask_percentile_spec_high,
            "mask_percentile_spec_low": self.mask_percentile_spec_low,
            "mask_protect_radius_arcsec": self.mask_protect_radius_arcsec,
            "id_col": self.id_col,
            "fiber_combine_method": self.fiber_combine_method,
            "galaxy_combine_methods": list(self.galaxy_combine_methods),
            "rest_delta": self.rest_delta,
            "rest_wave_min": self.rest_wave_min,
            "rest_wave_max": self.rest_wave_max,
            "flux_unit": self.flux_unit,
            "line_window": list(self.line_window),
            "n_bootstrap": self.n_bootstrap,
            "cont_bounds": [list(b) for b in self.cont_bounds],
            "cont_method": self.cont_method,
            "cont_order": self.cont_order,
            "centroid_method": self.centroid_method,
            "clip_negative": self.clip_negative,
            "measure_stack_method": self.measure_stack_method,
            "sigma_clip_sigma": self.sigma_clip_sigma,
            "sigma_clip_maxiters": self.sigma_clip_maxiters,
        }

    def to_header_dict(self, ngal: int, source_id: str) -> dict:
        """
        Provenance keywords for the galaxy FITS, per the plan's contract.

        Records everything that shaped Stage 1 output (so two files can be
        compared keyword-by-keyword). Does NOT record galaxy_combine_methods or
        the Stage 2 rest/unit params -- those are downstream, re-runnable choices
        recorded on the Stage 2 product instead.
        """
        bins_str = ",".join(str(b) for b in self.bins)
        variants = self.resolved_bg_variants()
        is_segmap = self.mask_has_segmap()      # True for 'segmap' AND 'segmap+spec_*'
        has_spec = self.mask_has_spec()
        has_image = "image" in self.mask_components()
        seg_path = self.resolve_seg_mask_path() if is_segmap else None
        return {
            "FIELD":    self.normalized_field(),
            "CATALOG":  self.catalog,
            "BINMODE":  self.bin_mode,
            "BINS":     bins_str,                     # exact edges (bin_mode units)
            "NRAD":     len(self.bins) - 1,
            "MASKMETH": self.mask_method,             # e.g. 'segmap+spec_global' (composite,
                                                        #   '+'-joined, OR-combined components)
            # Per-component percentiles; 0 when that component isn't active so
            # headers stay keyword-comparable across different mask_method runs.
            "MASKPCTI": self.mask_percentile_image if has_image else 0,
            "MASKPCTH": (self.mask_percentile_spec_high
                         if has_spec and self.mask_percentile_spec_high is not None else 0),
            "MASKPCTL": (self.mask_percentile_spec_low
                         if has_spec and self.mask_percentile_spec_low is not None else 0),
            "MASKPROT": (self.mask_protect_radius_arcsec
                         if has_spec and self.mask_protect_radius_arcsec is not None else 0),
            # Baked-mask provenance (segmap only; blank otherwise so headers stay
            # keyword-comparable across an image-vs-segmap A/B run).
            "SEGMFILE": os.path.basename(seg_path) if seg_path else "",
            "SEGMHASH": self.seg_mask_hash() if is_segmap else "",
            "IDCOL":    self.id_col,
            "COMBINE":  self.fiber_combine_method,    # fiber-to-fiber (Stage 1.3+1.4)
            "BGCOMB":   self.resolved_bg_combine_method(),  # = COMBINE unless decoupled
            "BGINNER":  self.bg_inner_arcsec,
            "BGOUTER":  self.bg_outer_arcsec,
            "MINBGFIB": self.min_bg_fibers,
            "SMOOTHBG": self.smooth_bg,
            "SMTHMETH": self.bg_smooth_method,
            "SMTHSIG":  self.bg_smooth_sigma_A,      # interp_gauss uniform sigma (A)
            "SMTHMHW":  self.line_mask_halfwidth_A,  # interp_gauss line-mask half-width (A)
            "SMOOTHV":  ",".join(str(v) for v in self.smoothing_values),  # adaptive params
            # --- background-variant products (2.5) ---------------------------
            # SPECBGSU is the keyword that stops anything downstream from having
            # to GUESS whether the SPEC HDU it is holding is background-
            # subtracted. It names the variant index that was subtracted (always
            # 0, the fiducial) so SPEC and SPEC_NOBG can never be confused.
            "SPECBGSU": 0,
            "BGVARN":   len(variants),
            "BGRAW":    bool(self.store_raw_bg),
            "NOBG":     bool(self.store_spec_nobg),
            "BGHALVES": bool(self.store_bg_halves),
            **{f"BGVAR{i}": f"{lo:g},{hi:g}" for i, (lo, hi) in enumerate(variants)},
            "MASSCOL":  self.mass_col,
            "ZCOL":     self.z_col,
            "WAVEMIN":  self.wave_min,
            "WAVEMAX":  self.wave_max,
            "NWAVE":    self.nwave,
            "NGAL":     ngal,
            "DATE":     _dt.datetime.now().isoformat(timespec="seconds"),
            "SRCFILE":  source_id,  # detects a stale cache vs reprocessed fiber data
        }

    def catalog_tag(self) -> str:
        """
        Short, stable hash of catalog_path, so a re-cut source catalog written
        to a NEW path produces a distinct galaxy FITS instead of silently
        reusing the old one. None/empty path -> 'nocat'.

        NB: this hashes the PATH STRING, not the file contents. If you overwrite
        the SAME path with a new cut, the tag (and the per-galaxy cache
        signature) will not change -- write re-cut catalogs to distinct
        filenames, or clear the cache, when you want a fresh extraction.
        """
        if not self.catalog_path:
            return "nocat"
        h = hashlib.md5(str(self.catalog_path).encode()).hexdigest()[:6]
        return f"cat{h}"

    def bins_tag(self) -> str:
        """
        Short, stable tag encoding the radial binning: 'b{nrad}_{hash6}', where
        nrad = number of radial bins (len(bins) - 1, the value printed as nrad in
        Stage 1/2) and hash6 is a 6-hex md5 of the normalized edge values. This
        lets different bin assortments -- e.g. jagged-edge robustness tests, or
        the 8-bin 20-Rvir run vs the default 6-bin run -- coexist in outputs/
        under distinct, self-describing names instead of overwriting each other.

        The edges are normalized to floats before hashing ('{:g}' formatting), so
        bins written as [0, 1, 2] and [0.0, 1.0, 2.0] hash identically; the same
        logical binning always yields the same tag regardless of int/float style.
        """
        nrad = max(len(self.bins) - 1, 0)
        norm = ",".join(f"{float(b):g}" for b in self.bins)
        h = hashlib.md5(norm.encode()).hexdigest()[:6]
        return f"b{nrad}_{h}"

    def descriptive_filename(self, catalog_tag: bool = True,
                             bins_tag: bool = True,
                             scheme: str = "sig") -> str:
        """
        Galaxy-FITS name. The CURRENT scheme ('sig', the default and the name
        Stage 2 writes and looks for first) is:

          galaxy_{FIELD}_{CATALOG}_b{nrad}_{extract_signature}.fits
          e.g.  galaxy_AEGIS_MOSDEF_b12_a1b2c3d4e5.fits

        Field, catalog and bin count stay readable because those are what you
        scan a directory for; EVERYTHING else -- bin mode, combine method, mask
        method, background radii, the catalog tag -- moves into the 10-hex
        signature. That is the point of the change, not a cosmetic trim.

        WHY: the old scheme hand-listed 8 knobs while the per-galaxy cache hashed
        ~30. Change one of the other ~22 and the cache correctly recomputed while
        the FITS name came out character-for-character identical -- so the new
        extraction silently overwrote the old one. Deriving the name from the
        same signature the cache uses makes that impossible by construction: if
        it changed the data, it changed the filename.

        `scheme` selects an OLDER spelling, which is how files written before
        this change keep loading (see io._resolve_galaxy_fits_path, which walks
        these newest-first under a header-BINS guard) -- no re-extraction:
          'sig'     galaxy_AEGIS_MOSDEF_b12_a1b2c3d4e5.fits                                            (current)
          'tagged'  galaxy_cat1a2b3c_b6_4f9e21_AEGIS_MOSDEF_virial_biweight_image_bg57-63.fits         (pre-signature)
          'tagged', bins_tag=False    galaxy_cat1a2b3c_AEGIS_MOSDEF_virial_biweight_image_bg57-63.fits (pre-bins-tag)
          'tagged', catalog_tag=False, bins_tag=False
                                      galaxy_AEGIS_MOSDEF_virial_biweight_image_bg57-63.fits           (legacy)
        `catalog_tag` / `bins_tag` apply only to 'tagged' and are ignored by
        'sig', where both are subsumed by the signature.
        """
        nf = self.normalized_field()
        if scheme == "sig":
            nrad = max(len(self.bins) - 1, 0)
            return (f"galaxy_{nf}_{self.catalog}_b{nrad}_"
                    f"{self.extract_signature()}.fits")
        if scheme != "tagged":
            raise ValueError(f"unknown naming scheme {scheme!r}; "
                             "expected 'sig' (current) or 'tagged' (pre-signature)")
        cat = f"{self.catalog_tag()}_" if catalog_tag else ""
        binp = f"{self.bins_tag()}_" if bins_tag else ""
        return (f"galaxy_{cat}{binp}{nf}_{self.catalog}_{self.bin_mode}_"
                f"{self.fiber_combine_method}_{self.mask_method}_"
                f"bg{int(self.bg_inner_arcsec)}-{int(self.bg_outer_arcsec)}.fits")

    def as_dict(self) -> dict:
        """Plain dict of every field (handy for logging / saving the run)."""
        return asdict(self)


def stamp_provenance(result: dict, config: "PipelineConfig", stage: str) -> dict:
    """
    Tag a result dict in place with provenance: the config's key knobs + hash,
    the producing `stage` (e.g. 'build_stacks', 'measure_all_bins',
    'redshift_scramble_null'), and a UTC timestamp. Returns the same dict.

    Centralized here (config is pure stdlib and imported everywhere) so Stage-2,
    Stage-3, and the validation tools all stamp identically, with no import
    cycles. A stamped result can always be traced back to the run that made it.
    """
    if result is None:
        return result
    prov = dict(config.provenance())
    prov["stage"] = stage
    prov["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    result["provenance"] = prov
    return result
