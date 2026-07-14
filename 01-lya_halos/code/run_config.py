"""
run_config.py -- the editable per-run spec.

THIS is the file you open every run. Every parameter is written out explicitly
(even where it equals the default) so the full menu of options is in front of
you each time. Edit values here; tests and real runs both build from `cfg`.

    from run_config import cfg            # real run
    from run_config import smoke          # smoke test -> cfg with num_gal small
"""

from dataclasses import replace
from utils_lya_halo import PipelineConfig

LYA_REST = 1215.67

cfg = PipelineConfig(

    # ===== SAMPLE / INPUTS ===========================================
    field           = "AEGIS",           # Stage 1: 'AEGIS' | 'COSMOS'.  Stage 2 also allows 'BOTH'
                                          #   (aliases: 'EGS'->AEGIS, 'COS'->COSMOS)
    catalog         = "MOSDEF",           # which CATALOG column value to select: 'MOSDEF' | '3DHST'
    catalog_path    = "catalogs/lya_halo_catalog_cuts/lya_mosdef_catalog_KEEP.fits",
    catalog_format  = "auto",             # 'auto' | 'fits' | 'ascii' -- 'auto' picks the reader from
                                          #   catalog_path's extension (.fits/.fit/.fz -> fits, else
                                          #   ascii); set explicitly only if the extension is misleading
    cont_image_path = "auto",             # LEGACY (mask_method='image' only) -- kept for A/B
                                          # "../../../../Hobby-Eberly-Telesco/imaging/candles_egs/groth/groth.fits",
                                          #   COSMOS: "../../../../HETDEX-Work/vdfi/cosmos/CFHT_COSMOS_image.fits"
    seg_mask_path   = "auto",             # REQUIRED when mask_method='segmap'. "auto" -> the per-field
                                          #   baked mask resolved in config.resolve_seg_mask_path()
                                          #   (set the two paths there). Or give an explicit path:
                                          #     EGS:    "<baked EGS segmentation mask>.fits"
                                          #     COSMOS: "<baked COSMOS segmentation mask>.fits"
                                          #   Read-only paths are fine; avoid OneDrive-synced WRITE
                                          #   paths (reading from OneDrive is OK, writing crashed).
    vdfi_base_path  = None,               # None -> resolved from field; or set e.g. ".../HETDEX-Work/vdfi/egs"

    # ===== RADIAL BINNING ============================================
    bins      = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    #bins      = [0, 10, 20, 30, 50, 80, 140, 300, 600, 1000, 2000],

                                          # edges in the unit named by bin_mode
                                          #   virial -> R/Rvir edges (per-gal kpc derived in Stage 1)
    bin_mode  = "kpc",                    # 'virial' | 'kpc' | 'arcsec'

    # ===== FIBER MASKING (Stage 1.1) =================================
    mask_method     = "segmap",  # single method ('segmap') or '+'-joined composite
                                          #   ('segmap+spec_local') -- components OR-combine (bad
                                          #   in ANY component -> bad overall). Components:
                                          #   'segmap' (baked mask) | 'image' (legacy percentile-
                                          #   image, A/B only) | 'spec_global' | 'spec_local'.
    mask_percentile_image     = 90,       # 'image' component ONLY; ignored otherwise
    mask_percentile_spec_high = 99.5,     # spec_* component: mask fibers ABOVE this percentile
                                          #   of per-fiber continuum (bright/leakage tail).
                                          #   None disables this tail.
    mask_percentile_spec_low  = 0.5,      # spec_* component: mask fibers BELOW this percentile
                                          #   (dim/noisy tail -- unexamined as of 2026-07).
                                          #   Independent of the high cut; both default on.
                                          #   None disables this tail.
    mask_protect_radius_arcsec = 10.0,    # spec_* component ONLY: fibers within this radius
                                          #   (arcsec) of ANY galaxy in this run's table are
                                          #   force-unmasked, regardless of percentile -- keeps
                                          #   spec masking from ever eating real core science
                                          #   fibers. None disables protection.
    spec_mask_step  = 5,                  # int; ONLY used by the spectral mask methods
    id_col          = "ID",               # catalog column holding the galaxy ID (default id_col for
                                          #   plotting.plot_fiber_mask_appendix / core.select_core_by_id).

    # ===== BACKGROUND (Stage 1.3) ====================================
    bg_inner_arcsec = 80.0,               # float; annulus inner radius (arcsec)
    bg_outer_arcsec = 90.0,               # float; annulus outer radius (arcsec)
    min_bg_fibers   = 999999,             # int; below this, fall back to galaxy-level bg
    smooth_bg       = True,               # True | False  (False injects annulus shot-noise -- keep True)
    bg_smooth_method = "interp_gauss",    # 'interp_gauss' (DEFAULT, ~100x faster: mask Lya, interpolate,
                                          #   one uniform gaussian) | 'adaptive' (legacy ramped-sigma smoother)
    bg_smooth_sigma_A      = 30.0,        # interp_gauss: uniform smoothing width (A)
    line_mask_halfwidth_A  = 10.0,        # interp_gauss: half-width (observed A) of the masked Lya gap
    smoothing_values = [200, 20, 300],    # LEGACY ('adaptive' only): [sigma_A, edge_sigma_A, edge_width_A]

    # ===== COADDITION ================================================
    # TWO concepts: fiber-level (used for BOTH background and science -- they
    # must match or subtraction leaves residuals) and galaxy-level (independent).
    fiber_combine_method  = "biweight",    # fibers -> per-gal bin AND -> background
                                          #   'biweight' | 'inv_var' | 'median' | 'mean'
    galaxy_combine_methods = ["biweight", "inv_var", "mean", "median", "sigma_clip", "weighted_median"],
                                          # per-gal spectra -> stacked bin (ALL computed in one pass)
    bg_combine_method = None,             # None -> follow fiber_combine_method (recommended).
                                          #   Set explicitly ONLY to deliberately decouple bg from science.
    mass_col = "MASS_50",                 # catalog column for stellar mass
    z_col    = "z",                       # catalog column for redshift

    # ===== WAVELENGTH AXIS (defines the WAVE HDU) ====================
    wave_min = 3470,                      # float; observed-frame min (A)
    wave_max = 5540,                      # float; observed-frame max (A)
    nwave    = 1036,                      # int; number of wavelength pixels (matches fiber file)

    # ===== STAGE 2 REST-FRAME STACKING ===============================
    rest_delta      = 0.2,                # rest-frame grid step (A); e.g. 0.2
    rest_wave_min   = 1100,               # rest-frame grid min (A)
    rest_wave_max   = 1700,               # rest-frame grid max (A)
    flux_unit       = "L_kpc2",           # 'raw' | 'flux_arcsec2' | 'L_fiber' | 'L_kpc2'
    flux_unit_scale = 1e-17,              # multiplies stored flux (erg/s/cm^2/A per count)
    fiber_diam_arcsec = 1.5,              # average-fiber aperture diameter (arcsec)
    rest_density    = True,               # multiply flux density by (1+z) (rest-frame integral correct)

    # ===== STAGE 2 SAMPLE CUT ========================================
    cut_radial_bin  = -1,                 # bin for finite-spectrum cut: -1=outer (~10% cut), 0=inner (~70%)
    min_good_wave   = 100,                # drop galaxies with < this many finite pixels in cut_radial_bin

    # ===== RUN CONTROL ===============================================
    num_gal    = None,                    # None -> ALL galaxies | int -> first N (smoke test)
    output_dir = "./outputs",             # free path; where the galaxy FITS / results land
    cache_dir  = None,                    # None -> <output_dir>/cache (kept inside outputs).
                                          #   Set an explicit path only to override.
    combine_wave_block = 16,              #   MEMORY KNOB ONLY (no effect on results): wavelength
                                          #   block size for the per-bin fiber combine. Smaller =>
                                          #   lower peak RAM on huge outer bins, slightly more
                                          #   loop overhead. None -> single-shot (old behavior).

# ===== STAGE 3 LINE MEASUREMENT ==================================
    # NOTE: Lya rest wavelength is FIXED -> PipelineConfig.LYA_REST (1215.67 A, vacuum).
    line_window = [LYA_REST-3, LYA_REST+3],     # [lo, hi] rest-frame A bracketing LYA_REST
    n_bootstrap = 2000,                         # int; galaxy-resampling draws for the centroid error

    # --- continuum subtraction (sideband-anchored) ---
    cont_method  = "median",              # 'median' (flat robust level; NEW default) | 'poly'
    cont_bounds  = ((1195, 1205), (1225, 1236)),   # (blue, red) sidebands, REST A -- LINE-SAFE:
                                          #   blue band sits between Si II 1193.3 and Si III 1206.5;
                                          #   red band stops short of N V 1238.8. Marginal change
                                          #   (continuum ~0 beyond the inner ~2 bins) but strictly
                                          #   cleaner. Matches DEFAULT_CONT_BOUNDS in config.py.
    cont_order   = 1,                     # polynomial order; used ONLY when cont_method='poly'

    # --- centroid estimator ---
    centroid_method = "median50",   # 'flux_weighted' | 'flux_median' | 'gaussian' | 'gauss_weighted' | 'peak'
                                    #   (alias 'median50' == 'flux_median'; see measure.CENTROID_METHOD_SPECS)
    
    clip_negative   = True,         # zero negatives before weighting. Consulted ONLY by
                                    #   'flux_weighted' (and 'moments'); IGNORED by 'flux_median'.
    
    measure_stack_method = "biweight",    # galaxy combine used INSIDE the measure bootstrap
                                          #   (must be one of galaxy_combine_methods)
)

def smoke(n: int = 5) -> PipelineConfig:
    """Return a copy of cfg restricted to the first `n` galaxies (smoke test)."""
    return replace(cfg, num_gal=n)
