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
    mask_method: str = "segmap+spec_global"  # single method ('segmap') OR a '+'-joined
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
    bg_inner_arcsec: float = 55.0
    bg_outer_arcsec: float = 65.0
    min_bg_fibers: int = 25
    smooth_bg: bool = True                # master on/off for background smoothing
    bg_smooth_method: str = "interp_gauss"  # 'interp_gauss' (default, fast) | 'adaptive' (legacy)
    # -- 'interp_gauss' params (mask Lya, interpolate, one uniform gaussian) --
    bg_smooth_sigma_A: float = 25.0       # uniform smoothing width (A)
    line_mask_halfwidth_A: float = 10.0   # half-width (observed A) of the masked Lya gap
    # -- 'adaptive' params (legacy ramped-sigma nan-safe smoother) --
    smoothing_values: Sequence[float] = _field(
        default_factory=lambda: [200, 20, 300]  # (sigma_A, edge_sigma_A, edge_width_A)
    )

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
    rest_wave_max: float = 1400          # rest-frame grid max (A); covers red sideband
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
    num_gal: Optional[int] = None        # None -> all galaxies in the run table
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

    # --- Stage 3 line measurement ---------------------------------------
    # Continuum, centroid, and measure-stack choices live HERE (config is the
    # single source of truth) and are read by measure.measure_all_bins. The
    # module-level DEFAULT_* above back these fields and are what measure.py's
    # standalone notebook helpers use, so the default is defined in one place.
    line_window: Sequence[float] = _field(
        default_factory=lambda: [PipelineConfig.LYA_REST - 4.0,
                                 PipelineConfig.LYA_REST + 4.0]
    )                                              # +/-4 A about LYA_REST (1211.67-1219.67)
    n_bootstrap: int = 1000
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
                             bins_tag: bool = True) -> str:
        """
        Self-documenting galaxy-FITS name encoding the defining knobs, so
        different runs don't collide in outputs/. Layout:
          galaxy_{cattag}_{binstag}_{FIELD}_{CATALOG}_{binmode}_{fiber}_{mask}_bg{in}-{out}.fits

        Both tags default to True (the canonical WRITE name and the name Stage 2
        looks for first). Turning a tag off reproduces an older naming scheme that
        the Stage-2 reader falls back to (guarded by a header-BINS check) so files
        made before that tag existed still load without re-extraction:
          catalog_tag=True,  bins_tag=True   galaxy_cat1a2b3c_b6_4f9e21_AEGIS_MOSDEF_virial_biweight_image_bg57-63.fits  (current)
          catalog_tag=True,  bins_tag=False  galaxy_cat1a2b3c_AEGIS_MOSDEF_virial_biweight_image_bg57-63.fits            (pre-bins-tag)
          catalog_tag=False, bins_tag=False  galaxy_AEGIS_MOSDEF_virial_biweight_image_bg57-63.fits                      (legacy)
        """
        nf = self.normalized_field()
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
