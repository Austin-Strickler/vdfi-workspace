"""
guide.py -- the LYAHALO field guide (front-end function reference).

A curated, user-friendly index of the functions you actually call in a
notebook. Back-end extraction helpers (the machinery run_extract/run_stack wire
up internally) are deliberately NOT listed here -- see the source for those.

USE IT (retro-terminal, argument-driven -- no blocking prompts):

    from utils_lya_halo import guide

    guide()                      # main menu: the six sections
    guide("measure")             # everything in a section
    guide("measure", "centroid") # one subsection
    guide("run_stack")           # full detail card for one function
    guide(search="centroid")     # grep names + purposes across all sections
    pipeline_map()               # a figure of the extraction->...->validation flow
    check_guide()                # audit: does this index still match the code?

EXAMPLE-CODE CONVENTION. Every entry's `example` assumes this standard notebook
preamble (so the snippets stay short and consistent):

    from run_config import cfg
    from utils_lya_halo import (run_extract, run_stack, run_measure,
                                read_galaxy_fits)
    from utils_lya_halo.analysis import run_measurement_suite
    path    = run_extract(cfg)                 # Stage 1 (slow, once)
    product = read_galaxy_fits(path)           # or: load_for_stack(cfg)
    stacks  = run_stack(cfg, path)             # Stage 2 (keep_cube=True)
    boot    = run_measure(cfg, stacks)         # Stage 3
    res     = run_measurement_suite(cfg, stacks)   # boot+stacks+errors bundle

MAINTENANCE. This index is a single registry (_ENTRIES below). check_guide()
diffs it against the live code and reports drift both ways -- entries whose
function no longer exists, and new public front-end functions not yet listed
(back-end names are ignored via _BACKEND). It runs inside test_framework.py, so
the smoke test fails if the guide falls out of sync. When you add/rename a
front-end function, add/edit its one entry here.
"""

from __future__ import annotations

import importlib
import textwrap

# ---------------------------------------------------------------------
# SECTION METADATA  (order + one-line blurb + subsection order)
# ---------------------------------------------------------------------
SECTIONS = {
    "extraction": ("Stage 1-2: turn the fiber data + catalog into the galaxy "
                   "FITS and the rest-frame stacks.",
                   ["Drivers", "Inputs & catalog", "Masking",
                    "Background smoothing", "Stacking", "I/O"]),
    "measure":    ("Stage 3: centroids, integrated flux, asymmetry, and the "
                   "galaxy-bootstrap errors on them.",
                   ["Drivers", "Centroid estimators", "Flux & asymmetry",
                    "Continuum", "Bootstrap", "Method comparison",
                    "Derived properties"]),
    "plotting":   ("Figures for stacks, radial profiles, and per-run "
                   "diagnostics.",
                   ["Stack spectra", "Radial profiles", "Two-sample overlays",
                    "Diagnostics", "One-call bundles", "Fiber & mask visuals"]),
    "validation": ("Prove the centroid is real: nulls, robustness sweeps, "
                   "error cross-checks, injection-recovery, and the dossier.",
                   ["Nulls", "Robustness sweeps", "Method scoring",
                    "Error checks", "Injection-recovery", "S/N spectrum",
                    "Synthesis", "Core characterization"]),
    "sample":     ("Slice, split, and match the sample -- pure Stage-2, no "
                   "re-extraction.",
                   ["Slice & select", "Split", "Multi-catalog", "Sky plots"]),
    "misc":       ("Science extras and separate tracks: virial conversions, "
                   "LSF/intrinsic profile, star-PSF & line profiles, core "
                   "(escape-fraction) extraction, single-galaxy halo profiles.",
                   ["Virial conversions", "LSF / intrinsic profile",
                    "Star-PSF & line profiles", "Core / escape-fraction",
                    "Single galaxy"]),
    "fitting":    ("Radial flux-profile fitting: PSF forward-modeling, the "
                   "two-exponential and expcore (core + two-halo) models, "
                   "and the UV-continuum single-component fit.",
                   ["PSF & convolution", "Two-exponential fit (Part 1)",
                    "Expcore fit (Part 2)", "Model comparison & boundary",
                    "UV-continuum fit (Part 3)"]),
    "uv_continuum": ("Part 3's standalone UV-continuum pipeline (CFHT-LS "
                     "r-band): per-galaxy cutout/photometry through coadd, "
                     "bootstrap, and stellar-PSF measurement. Not yet merged "
                     "into extract.py -- see uv_profile.py's own module note.",
                     ["Field & catalog I/O", "Per-galaxy extraction",
                      "Coadd & fit", "Stellar PSF"]),
}

# ---------------------------------------------------------------------
# THE REGISTRY  (single source of truth)
#   name, module, section, sub, desc (1-4 sentences), example (<=3 lines)
# ---------------------------------------------------------------------
def _e(name, module, section, sub, desc, example):
    return dict(name=name, module=module, section=section, sub=sub,
                desc=" ".join(desc.split()), example=example.strip("\n"))

_ENTRIES = [
    # ===================== EXTRACTION =====================
    _e("run_extract", "pipeline", "extraction", "Drivers",
       "Stage 1: extract every galaxy's binned, background-subtracted spectra "
       "and write the galaxy FITS. Slow and I/O-bound -- run it once per "
       "config; results are cached per galaxy so re-runs are cheap.",
       "path = run_extract(cfg)"),
    _e("run_stack", "pipeline", "extraction", "Drivers",
       "Stage 2: load the galaxy FITS and coadd into rest-frame stacks (one per "
       "combine method). Fast -- re-run freely. keep_cube=True is required if "
       "you will measure or bootstrap afterwards.",
       "stacks = run_stack(cfg, path, keep_cube=True)"),
    _e("run_pipeline", "pipeline", "extraction", "Drivers",
       "Convenience driver that chains EXTRACT -> STACK -> MEASURE in one call. "
       "Handy for a quick end-to-end run; use the individual stages when you "
       "want to iterate on Stage 2/3 without re-extracting.",
       "result = run_pipeline(cfg)"),
    _e("load_field", "extract", "extraction", "Inputs & catalog",
       "Load a field's raw fiber data (RA/DEC, spectra, errors, seeing) from "
       "the path implied by config. The in-memory input to extraction and to "
       "the fiber/mask diagnostics.",
       "vdfi = load_field(cfg)"),
    _e("build_run_table", "extract", "extraction", "Inputs & catalog",
       "Build the per-galaxy run table: read the catalog and select the rows "
       "matching this config's field and catalog tag. This row order defines "
       "the galaxy axis of every downstream product.",
       "run_table = build_run_table(cfg)"),
    _e("read_catalog_table", "extract", "extraction", "Inputs & catalog",
       "Read a catalog file into an astropy Table, auto-detecting FITS vs "
       "ASCII from the extension. Use when you want to inspect or pre-cut the "
       "catalog before handing it to extraction.",
       "tbl = read_catalog_table('catalogs/mycat.txt')"),
    _e("radial_bin_edges", "extract", "extraction", "Inputs & catalog",
       "The radial bin edges for ONE galaxy in the unit extraction bins in "
       "(virial -> kpc via that galaxy's mass & z; kpc/arcsec pass through). "
       "Mostly a diagnostic to see where a given galaxy's bins land.",
       "edges = radial_bin_edges(cfg, mass=10.5, z=2.3)"),
    _e("build_bad_fiber_mask", "extract", "extraction", "Masking",
       "The masking front door: dispatch on cfg.mask_method to produce the "
       "(nexp, nfib) bad-fiber mask. mask_method may be a single method or a "
       "'+'-joined COMPOSITE ('segmap+spec_global') whose component masks are "
       "OR-combined (bad in ANY component -> bad). Pass run_table to enable "
       "cfg.mask_protect_radius_arcsec (spec_* components only). Call it "
       "directly for a diagnostic or A/B comparison without running extraction.",
       "bad = build_bad_fiber_mask(cfg, vdfi, run_table=catalog)"),
    _e("segmentation_mask", "masking", "extraction", "Masking",
       "The default ('segmap') method: flag fibers landing on a baked binary "
       "mask FITS via a pure WCS lookup. All masking intelligence and dilation "
       "is frozen into the mask upstream, so this just reads the verdict.",
       "bad = segmentation_mask(cfg.resolve_seg_mask_path(), vdfi['RA'], vdfi['DEC'])"),
    _e("make_spectral_continuum_mask", "masking", "extraction", "Masking",
       "Alternative ('spec_global'/'spec_local') method: flag fibers by their "
       "own continuum level. Two INDEPENDENT tails OR'd together -- "
       "mask_percentile_high (bright/leakage) and mask_percentile_low "
       "(dim/noisy) -- either can be None to disable. No image needed; works "
       "straight off the spectra. mode 'global' (one threshold) | 'local' "
       "(per-exposure).",
       "bad = make_spectral_continuum_mask(vdfi['spectra'], mask_percentile_high=99.5, mask_percentile_low=0.5)"),
    _e("continuum_image_mask", "masking", "extraction", "Masking",
       "Legacy ('image') method, kept as the A/B baseline: smooth + threshold + "
       "dilate a continuum image, then flag fibers on masked pixels. Superseded "
       "by segmentation_mask, which bakes these decisions upstream.",
       "bad = continuum_image_mask(cfg.resolve_cont_image_path(), 'AEGIS', 90, vdfi['RA'], vdfi['DEC'])"),
    _e("apply_protect_radius", "masking", "extraction", "Masking",
       "Force-UNMASK fibers within radius_arcsec of ANY galaxy position "
       "(spec_* components only, applied inside build_bad_fiber_mask when "
       "cfg.mask_protect_radius_arcsec is set). Protects real core science "
       "fibers from spectral percentile masking. Returns a new mask; does not "
       "mutate the input.",
       "mask = apply_protect_radius(mask, vdfi['RA'], vdfi['DEC'], gal_RA, gal_DEC, 10.0)"),
    _e("smooth_interp_then_gauss", "smoothing", "extraction", "Background smoothing",
       "The default background smoother: mask the Lya line, linearly "
       "interpolate across the gap, then one uniform Gaussian. ~100x faster "
       "than the adaptive smoother and free of line/pulse artifacts; used "
       "internally by extraction but callable for A/B on a real background.",
       "bg = smooth_interp_then_gauss(prod.background[0], prod.wave, z=2.3)"),
    _e("smooth_spectrum_nan_safe_adaptive", "smoothing", "extraction", "Background smoothing",
       "The legacy adaptive smoother (ramped edge->interior sigma, NaN-safe). "
       "Kept for comparison; select it with cfg.bg_smooth_method='adaptive'.",
       "bg = smooth_spectrum_nan_safe_adaptive(prod.background[0], prod.wave, 200, 20, 300)"),
    _e("build_stacks", "stack", "extraction", "Stacking",
       "The Stage-2 engine (what run_stack calls): per radial bin, unit-convert, "
       "rest-frame resample, and coadd across galaxies for every combine method. "
       "Call directly if you already hold a product and want the stacks.",
       "stacks = build_stacks(cfg, product, keep_cube=True)"),
    _e("read_galaxy_fits", "io", "extraction", "I/O",
       "Load a galaxy FITS written by Stage 1 into an in-memory GalaxyProduct "
       "(spec/err/nfib/background/catalog). The starting point for any custom "
       "Stage-2 work on an already-extracted field.",
       "product = read_galaxy_fits(path)"),
    _e("load_for_stack", "io", "extraction", "I/O",
       "Resolve and load the galaxy FITS for this config (handling field='BOTH' "
       "concatenation) and apply the finite-spectrum cut. The convenience path "
       "run_stack uses when you don't hand it an explicit file.",
       "product = load_for_stack(cfg)"),
    _e("apply_finite_cut", "io", "extraction", "I/O",
       "Drop galaxies with too few finite pixels in a chosen radial bin, "
       "slicing every array and the catalog together. Use to re-cut a product "
       "by hand at a different bin/threshold than the config default.",
       "product = apply_finite_cut(product, cut_radial_bin=-1, min_good_wave=100)"),

    # ===================== MEASURE =====================
    _e("run_measure", "pipeline", "measure", "Drivers",
       "Stage 3: bootstrap the Lya centroid (+ blue/red side ratio) and the "
       "per-pixel stack error from the Stage-2 cube. Fast; requires stacks "
       "built with keep_cube=True.",
       "boot = run_measure(cfg, stacks)"),
    _e("measure_all_bins", "measure", "measure", "Drivers",
       "The config-driven measurement engine run_measure calls: centroid + "
       "side-ratio bootstrap and stack error for every radial bin. Use directly "
       "to override stack_method or skip the stack error.",
       "boot = measure_all_bins(cfg, stacks, stack_method='biweight')"),
    _e("run_measurement_suite", "analysis", "measure", "Drivers",
       "Run the full Stage-3 suite on a stacks dict and return ONE bundle "
       "(boot + stacks + stack error) ready for every plot in analysis.py. The "
       "recommended one-liner at the top of an analysis cell.",
       "res = run_measurement_suite(cfg, stacks)"),
    _e("measure_centroid", "measure", "measure", "Centroid estimators",
       "One entry point for every centroid estimator (dispatch by name), all "
       "sharing the same return contract. Use this rather than calling a "
       "specific estimator unless you need that estimator's extra kwargs.",
       "r = measure_centroid(rest_wave, flux, err, method='flux_median')"),
    _e("flux_weighted_centroid", "measure", "measure", "Centroid estimators",
       "Continuum-subtract then flux-weight to a centroid over the line window. "
       "The classic first-moment estimator; clip_negative controls whether "
       "negative pixels are zeroed before weighting.",
       "r = flux_weighted_centroid(rest_wave, flux, err)"),
    _e("gaussian_centroid", "measure", "measure", "Centroid estimators",
       "Fit a single Gaussian to the continuum-subtracted line; the centroid is "
       "the fitted mean. Best when the line is reasonably symmetric and "
       "well-detected.",
       "r = gaussian_centroid(rest_wave, flux, err)"),
    _e("flux_median_centroid", "measure", "measure", "Centroid estimators",
       "The wavelength where the cumulative positive line flux reaches 50% "
       "(alias 'median50'). Robust to wings and the pipeline default centroid.",
       "r = flux_median_centroid(rest_wave, flux, err)"),
    _e("peak_parabola_centroid", "measure", "measure", "Centroid estimators",
       "Sub-pixel peak (mode) from a 3-point parabola through the max pixel and "
       "its neighbours. A peak-chaser: fast but biases toward the largest noise "
       "excursion at low S/N.",
       "r = peak_parabola_centroid(rest_wave, flux, err)"),
    _e("gauss_weighted_centroid", "measure", "measure", "Centroid estimators",
       "Adaptive Gaussian-weighted first moment: iteratively re-weights around "
       "the current centroid, down-weighting far wings. A middle ground between "
       "flux-weighted and a full Gaussian fit.",
       "r = gauss_weighted_centroid(rest_wave, flux, err)"),
    _e("line_moments", "measure", "measure", "Centroid estimators",
       "Flux-weighted moments of the line in one pass: centroid, width, "
       "skewness. Use when you want the line SHAPE (asymmetry via skew), not "
       "just its center.",
       "m = line_moments(rest_wave, flux, err)"),
    _e("integrated_line_flux", "measure", "measure", "Flux & asymmetry",
       "The canonical continuum-subtracted integrated flux in a wavelength "
       "window. The single master flux routine every other flux measurement "
       "(and the core path) routes through.",
       "f = integrated_line_flux(rest_wave, flux, err)"),
    _e("integrated_line_flux_per_bin", "measure", "measure", "Flux & asymmetry",
       "integrated_line_flux looped over every radial bin of ONE restacked "
       "(nrad, nwave) spectrum -- the fiducial stack or a single bootstrap "
       "draw. bootstrap_all already has this per-bin loop shape internally "
       "(its _measure_stack closure) but isn't exported and doesn't call "
       "integrated_line_flux; this is the reusable version, e.g. for "
       "optimize.py's line-S/N diagnostic.",
       "flux_sum, err_sum = integrated_line_flux_per_bin(rest_wave, stack, bounds=cfg.line_window)"),
    _e("blue_red_side_ratio", "measure", "measure", "Flux & asymmetry",
       "Coarse blue/red asymmetry: continuum-subtracted flux summed on each "
       "side of line center, and their ratio. A quick asymmetry read; the "
       "bootstrap computes this per bin when compute_side_ratio=True.",
       "br = blue_red_side_ratio(rest_wave, flux)"),
    _e("flux_curve_of_growth", "measure", "measure", "Flux & asymmetry",
       "Cumulative Lya luminosity and flux fraction vs radius, built on a "
       "measure_all_bins/bootstrap_all summary: multiplies the per-bin "
       "total_flux_fid by a representative fiber footprint area "
       "(fiber_area_kpc2) and cumsums outward, dropping the noisy outermost "
       "bin by default. Bootstrap bands are computed per-draw, not as a ratio "
       "of percentiles. Bad (NaN/negative) bins warn and propagate rather "
       "than being hidden.",
       "cog = flux_curve_of_growth(boot, stacks)"),
    _e("flux_curve_of_growth_annulus", "measure", "measure", "Flux & asymmetry",
       "The area-correct sibling of flux_curve_of_growth: weights each bin's "
       "surface-brightness value (total_flux_fid) by its OWN geometric "
       "annulus area (pi*(r_out**2-r_in**2)) rather than one fixed "
       "fiber-footprint area applied at every radius, so outer bins aren't "
       "systematically under-weighted. Needs a stack built with "
       "flux_unit='L_kpc2' (the default).",
       "cog = flux_curve_of_growth_annulus(boot, stacks)"),
    _e("get_continuum_model", "measure", "measure", "Continuum",
       "The continuum model over the full grid from the sideband windows "
       "(median level or low-order poly). The one place the continuum is "
       "defined; every estimator subtracts this.",
       "cont = get_continuum_model(rest_wave, flux, cont_method='median')"),
    _e("fit_local_poly_continuum", "measure", "measure", "Continuum",
       "Robust local polynomial continuum fit from the sideband windows only "
       "(sigma-clipped). The 'poly' backend of get_continuum_model; call "
       "directly to inspect the fitted coefficients.",
       "c = fit_local_poly_continuum(rest_wave, flux, order=1)"),
    _e("bootstrap_all", "measure", "measure", "Bootstrap",
       "One-pass Stage-3 bootstrap: centroid (+ blue/red side ratio) AND the "
       "per-pixel stack error, from a single galaxy-resampling loop over the "
       "sliced measurement window. Supersedes calling bootstrap_measurements + "
       "bootstrap_stack_error separately -- same result, faster (one re-stack "
       "per draw instead of two, over a much smaller window).",
       "boot = bootstrap_all(flux_rf, err_rf, wave, nboot=1000)"),
    _e("bootstrap_stack_error", "measure", "measure", "Bootstrap",
       "Per-pixel 1-sigma flux error of the stack from galaxy resampling. Feeds "
       "the S/N spectrum and any plot that needs an empirical per-pixel error "
       "band.",
       "err = bootstrap_stack_error(cube_flux, cube_err, nboot=1000)"),
    _e("bootstrap_fit_profile", "analysis", "measure", "Bootstrap",
       "Refit-per-draw parameter uncertainty for the two-component flux-"
       "profile fit (h1/h2, or r_c/gamma for model='expcore'), as an "
       "alternative to reading errors off one fiducial fit's covariance "
       "matrix -- works for either model and generalizes the old "
       "fitting.bootstrap_fit_expcore. Expensive (nboot separate nonlinear "
       "fits); sanity-check with a small nboot first.",
       "bf = bootstrap_fit_profile(boot, r_edges, r_fine, model='expcore', R=R)"),
    _e("describe_subsample_properties", "analysis", "measure", "Derived properties",
       "Prints the full paper-ready summary for ONE subsample: the fit "
       "(delegating to describe_fit/describe_fit_expcore), core/two-halo "
       "boundary, core-vs-two-halo velocity, and the core/one-halo/two-halo "
       "luminosity split with ratios. Purely a reporting function -- every "
       "argument is an already-computed result dict, all optional. Added "
       "2026-07-18. subsample-derived-properties.md Part 5.",
       "describe_subsample_properties(label='low-z', fit_result=fr, vel=vel, "
       "core_lum=core, onehalo_lum=oh, twohalo_lum=twohalo, ratios=rat)"),
    _e("measure_core_twohalo_velocity", "measure", "measure", "Derived properties",
       "Core velocity (innermost bin's centroid, unmodified) vs. two-halo "
       "velocity (inv-var- or biweight-combined centroid beyond a "
       "core/two-halo boundary_radius), plus their difference -- all with "
       "bootstrap errors reusing boot's existing per-draw centroid arrays, "
       "no new resampling. Renamed from measure_core_halo_velocity "
       "(2026-07-21). subsample-derived-properties.md Part 2.",
       "vel = measure_core_twohalo_velocity(boot, boundary_radius)"),
    _e("measure_twohalo_luminosity", "measure", "measure", "Derived properties",
       "Integrated luminosity beyond a core/two-halo boundary_radius: each "
       "outer bin's surface brightness times its OWN annulus area, summed "
       "-- a genuine luminosity (erg/s), not a raw sum of surface-"
       "brightness values. This zone is two-halo/clustering-term "
       "dominated, not the galaxy's own CGM -- see "
       "measure_onehalo_luminosity for the true one-halo zone. Renamed "
       "from measure_halo_luminosity (2026-07-21). subsample-"
       "derived-properties.md Part 3.",
       "twohalo = measure_twohalo_luminosity(boot, boundary_radius)"),
    _e("measure_onehalo_luminosity", "measure", "measure", "Derived properties",
       "Integrated luminosity of the TRUE one-halo (CGM) zone: strictly "
       "between the core bin and the fitted core/two-halo boundary_radius. "
       "Same empirical zone-sum recipe as measure_twohalo_luminosity, just "
       "over the inner radius range -- draw-aligned with it and with "
       "measure_psf_corrected_core_luminosity's core_lum_all so ratios "
       "between all three are safe. Added 2026-07-18. subsample-derived-"
       "properties.md Part 3b.",
       "oh = measure_onehalo_luminosity(boot, boundary_radius)"),
    _e("measure_three_zone_ratios", "measure", "measure", "Derived properties",
       "Combines core_lum + onehalo_lum + twohalo_lum into per-draw-aligned "
       "ratios (onehalo/core, twohalo/onehalo, twohalo/total) and the "
       "reconciled total luminosity. Pure arithmetic on already-computed "
       "results, no new bootstrap. Added 2026-07-18. subsample-derived-"
       "properties.md Part 3b.",
       "rat = measure_three_zone_ratios(core_lum, onehalo_lum, twohalo_lum)"),
    _e("measure_outer_properties", "measure", "measure", "Derived properties",
       "One-call bundle of everything needing NO new stacking/bootstrap "
       "pass: measure_core_twohalo_velocity + measure_twohalo_luminosity, "
       "merged into one dict. Deliberately excludes core luminosity "
       "(measure_psf_corrected_core_luminosity), the one piece that DOES "
       "need a fresh restacking pass.",
       "props = measure_outer_properties(boot, boundary_radius)"),
    _e("measure_psf_corrected_core_luminosity", "measure", "measure", "Derived properties",
       "The core-side counterpart to measure_twohalo_luminosity: PSF-"
       "aperture-corrects the innermost bin PER GALAXY (a fixed literature "
       "Moffat converted to each galaxy's own kpc width via its z), "
       "re-stacks, and returns a true luminosity (erg/s) for that bin -- "
       "replaying boot's exact bootstrap seed/draw sequence so it's "
       "ratio-safe against the two-halo side. subsample-derived-properties.md "
       "Part 3, core side.",
       "core = measure_psf_corrected_core_luminosity(cfg, product, stacks, boot)"),
    _e("measure_psf_corrected_core_onehalo_luminosity", "measure", "measure", "Derived properties",
       "Core correction + one-halo PSF-leak subtraction together in ONE "
       "restack + bootstrap pass. Extends the core's per-galaxy Moffat EE "
       "model to a full profile evaluated at every one-halo bin edge, so "
       "the fraction of core flux the PSF model predicts leaked into each "
       "zone bin is subtracted from that galaxy's raw spectrum before "
       "stacking -- closes a double-count between measure_psf_corrected_"
       "core_luminosity (boosts the core) and measure_onehalo_luminosity "
       "(left the leaked light uncorrected in the zone). Returns "
       "(core_lum, onehalo_lum), key-compatible with both older functions' "
       "outputs (drop straight into measure_three_zone_ratios/"
       "describe_subsample_properties); onehalo_lum also carries "
       "onehalo_lum_raw_*/leaked_lum_* diagnostics. Bootstraps only the "
       "_measurement_window-sliced cube (biweight dominates runtime and "
       "scales worse than linearly in nwave -- same fix bootstrap_all "
       "already uses). assume_no_leak_beyond_boundary=True swaps in a much "
       "cheaper conservation-only mode (2 restacked columns per draw "
       "instead of 1+n_zone) if you're willing to assume no leaked flux "
       "crosses boundary_radius at all. Added 2026-07-22.",
       "core, oh = measure_psf_corrected_core_onehalo_luminosity(cfg, product, stacks, boot, boundary_radius)"),
    _e("compare_centroid_methods", "analysis", "measure", "Method comparison",
       "Point estimates (no bootstrap) from all standard estimators on the "
       "fiducial stack, as a printed table. The fastest way to see how much the "
       "centroid depends on estimator choice.",
       "compare_centroid_methods(stacks, cfg)"),
    _e("measure_centroid_methods", "analysis", "measure", "Method comparison",
       "The with-error-bars sibling: bootstrap the centroid with several "
       "estimators and return one boot dict each, ready to overlay with "
       "plot_centroid_methods.",
       "multi = measure_centroid_methods(stacks, cfg)"),
    _e("print_centroid_table", "analysis", "measure", "Method comparison",
       "Print a formatted per-radial-bin table of centroid + side-ratio results "
       "from a boot dict. The quick text readout of a measurement.",
       "print_centroid_table(boot, stacks['r_edges'])"),
    _e("measurement_confidence", "analysis", "measure", "Method comparison",
       "Per-bin 'how much do we trust this point' summary (flux S/N, success "
       "fraction, significance) from an existing boot dict -- no extra "
       "computation. Complements plot_success_frac.",
       "conf = measurement_confidence(boot, stacks, stacks['r_edges'])"),

    # ===================== PLOTTING =====================
    _e("plot_stack_panels", "plotting", "plotting", "Stack spectra",
       "One spectral panel per radial bin for a chosen combine method. The "
       "basic look at what the stacks contain across radius.",
       "plot_stack_panels(stacks, method='biweight')"),
    _e("plot_methods_overlay", "plotting", "plotting", "Stack spectra",
       "Overlay several galaxy-combine methods (biweight vs inv_var vs mean...) "
       "for ONE radial bin. Use to check the stack isn't driven by the combine "
       "choice.",
       "plot_methods_overlay(stacks, radial_index=0)"),
    _e("plot_radius_velocity_map", "plotting", "plotting", "Stack spectra",
       "2-D radius-velocity image of the stacked line: all bins at once, "
       "velocity about Lya on x, radial bin on y, flux as color. The compact "
       "'whole halo in one panel' view.",
       "plot_radius_velocity_map(stacks, method='biweight')"),
    _e("plot_centroid_profile", "analysis", "plotting", "Radial profiles",
       "The headline figure: centroid velocity +/- bootstrap 16/84 per radial "
       "bin, on the native + comparison radius axes. The main result plot.",
       "plot_centroid_profile(boot, stacks)"),
    _e("plot_flux_profile", "analysis", "plotting", "Radial profiles",
       "Integrated Lya flux +/- bootstrap 16/84 vs radius (log-y by default, to "
       "see the faint outer bins). The radial surface-brightness profile.",
       "plot_flux_profile(boot, stacks)"),
    _e("plot_flux_profile_fit", "analysis", "plotting", "Radial profiles",
       "The flux profile with the fitting.py two-component (A1*exp(-r/h1) + "
       "A2*exp(-r/h2)) model overlaid -- naive or PSF-aware (method='psf'), "
       "drawing the fitted intrinsic profile and per-component scale lengths on "
       "the measured bins. The figure behind the h1/h2 fit result.",
       "plot_flux_profile_fit(boot, stacks, method='psf')"),
    _e("plot_flux_curve_of_growth", "analysis", "plotting", "Radial profiles",
       "Two-panel curve of growth: cumulative Lya luminosity (top, log-y) and "
       "flux fraction L(<r)/L(<r_max) (bottom) vs radius, with bootstrap "
       "bands. Draws a measure.flux_curve_of_growth result -- the "
       "curve-of-growth companion to plot_flux_profile.",
       "plot_flux_curve_of_growth(cog, stacks)"),
    _e("plot_asymmetry_profile", "analysis", "plotting", "Radial profiles",
       "Two-panel asymmetry: blue fraction B/(B+R) vs radius (top) and absolute "
       "blue/red flux (bottom), with bootstrap bands. Where the line goes "
       "blue- or red-dominated with radius.",
       "plot_asymmetry_profile(boot, stacks)"),
    _e("plot_moments_profile", "analysis", "plotting", "Radial profiles",
       "Three-panel radial profile of the line moments (centroid, width, skew) "
       "on the fiducial stack. Use to read line-shape trends without the "
       "bootstrap.",
       "plot_moments_profile(stacks)"),
    _e("plot_centroid_vs_radius", "plotting", "plotting", "Radial profiles",
       "The low-level headline centroid figure (native-bottom / R-vir-top axes) "
       "that plot_centroid_profile wraps with defaults. Use directly for full "
       "control of ticks, limits, literature overlays.",
       "plot_centroid_vs_radius(boot, stacks['r_edges'])"),
    _e("plot_blue_red_vs_radius", "plotting", "plotting", "Radial profiles",
       "Overlay summed blue-side and red-side flux per radial bin (bootstrap "
       "bars) so you see where the two diverge. The flux-space companion to the "
       "asymmetry ratio.",
       "plot_blue_red_vs_radius(boot, stacks['r_edges'])"),
    _e("plot_centroid_profile_two", "analysis", "plotting", "Two-sample overlays",
       "Overlay the centroid profiles of TWO samples on one axis (e.g. low-z vs "
       "high-z). Replaces calling plot_centroid_profile twice.",
       "plot_centroid_profile_two(boot_a, boot_b, stacks_a, stacks_b, labels=('lo','hi'))"),
    _e("plot_flux_profile_two", "analysis", "plotting", "Two-sample overlays",
       "Overlay integrated flux +/- bootstrap for two samples on one axis "
       "(log-y default). The two-sample flux comparison.",
       "plot_flux_profile_two(boot_a, boot_b, stacks_a, stacks_b, labels=('lo','hi'))"),
    _e("plot_asymmetry_profile_two", "analysis", "plotting", "Two-sample overlays",
       "Two-sample asymmetry overlay in the same two-panel layout as "
       "plot_asymmetry_profile. Compares blue/red behaviour between subsamples.",
       "plot_asymmetry_profile_two(boot_a, boot_b, stacks_a, stacks_b, labels=('lo','hi'))"),
    _e("plot_centroid_comparison", "plotting", "plotting", "Two-sample overlays",
       "Overlay centroid-vs-radius for SEVERAL labelled samples on one axis "
       "(the N-sample generalization). The engine behind multicat.compare_centroids.",
       "plot_centroid_comparison({'lo': boot_a, 'hi': boot_b}, stacks['r_edges'])"),
    _e("plot_line_panels", "analysis", "plotting", "Diagnostics",
       "The most important per-run diagnostic: one Lya-zoom panel per radial "
       "bin showing the continuum fit, line window, sidebands, and measured "
       "centroid. Always eyeball this before trusting a profile.",
       "plot_line_panels(stacks, boot)"),
    _e("plot_success_frac", "analysis", "plotting", "Diagnostics",
       "Bootstrap centroid success fraction per bin; bars below ~0.8 mean the "
       "centroid is being computed from a truncated/failing line. A trust "
       "gate for the outer bins.",
       "plot_success_frac(boot, stacks)"),
    _e("plot_bootstrap_draws", "analysis", "plotting", "Diagnostics",
       "Histogram of the centroid bootstrap draws for one radial bin -- shows "
       "skew/bimodality the 16/84 summary hides. Use on a bin whose error bar "
       "looks suspicious.",
       "plot_bootstrap_draws(boot, radial_index=0, stacks=stacks)"),
    _e("plot_effective_n_per_bin", "plotting", "plotting", "Diagnostics",
       "Bar of the number of galaxies actually contributing (any finite flux) "
       "to each radial bin. Explains why an outer bin is noisy.",
       "plot_effective_n_per_bin(stacks)"),
    _e("plot_contributing_spectra", "plotting", "plotting", "Diagnostics",
       "The true effective N(lambda): how many galaxies have finite flux at "
       "each rest wavelength in a bin. Use when a bin's coverage varies across "
       "the line.",
       "plot_contributing_spectra(stacks, per_bin, radial_index=0)"),
    _e("plot_significance_comparison", "plotting", "plotting", "Diagnostics",
       "Two per-bin significances on one axis vs radius: flux detection S/N ('is "
       "there signal?') and centroid significance ('is the shift real?'). "
       "Separates detection from measurement confidence.",
       "plot_significance_comparison(conf, boot, boot['centroid_v_med'])"),
    _e("run_all_plots", "analysis", "plotting", "One-call bundles",
       "Run every core analysis plot in a sensible order with one call. The "
       "full-picture cell right after run_measurement_suite.",
       "run_all_plots(stacks, boot)"),
    _e("run_all_plots_two", "analysis", "plotting", "One-call bundles",
       "The two-sample companion: every half-sample comparison figure (panels + "
       "centroid + flux + asymmetry) in one call.",
       "run_all_plots_two(stacks_a, stacks_b, boot_a, boot_b, labels=('lo','hi'))"),
    _e("plot_core_stack", "plotting", "plotting", "Fiber & mask visuals",
       "Three-panel quicklook of a build_core_stacks result: full spectrum with "
       "rest-UV lines, clipped continuum view, and the Lya region. For the "
       "PSF-weighted core track.",
       "plot_core_stack(core_stack_result)"),
    _e("plot_top_sn_grid", "plotting", "plotting", "Fiber & mask visuals",
       "Grid of core spectra for a selection.top_sn_galaxies shortlist, one "
       "panel per galaxy (S/N-descending), Lya window shaded and annotated "
       "with S/N + ID + label. The eyeball step before picking IDs to stack.",
       "plot_top_sn_grid(sub)  # sub from top_sn_galaxies"),
    _e("plot_mask_spectra", "plotting", "plotting", "Fiber & mask visuals",
       "Plot the output of core.compare_mask_spectra: core spectra of a few "
       "galaxies overlaid per candidate mask, with a Lya zoom. The A/B mask "
       "validation figure.",
       "plot_mask_spectra(res_masks)  # res_masks from compare_mask_spectra(..., plot=False)"),
    _e("plot_fiber_overlay", "plotting", "plotting", "Fiber & mask visuals",
       "Overlay one galaxy's extraction fibers on the sky, colored by PSF "
       "weight, with the aperture and any masked fibers. Shows exactly what "
       "feeds a core extraction.",
       "plot_fiber_overlay(vdfi, run_table, gal_index=0, bad_fiber_mask=bad)"),
    _e("plot_fiber_mask_appendix", "plotting", "plotting", "Fiber & mask visuals",
       "Appendix grid: one cutout panel per galaxy that has a masked fiber near "
       "its own center. The before/after tool for confirming the mask stops "
       "clipping galaxy cores.",
       "fig, audit = plot_fiber_mask_appendix(vdfi, run_table, bad, cfg.resolve_cont_image_path())"),

    # ===================== VALIDATION =====================
    _e("redshift_scramble_null", "validation", "validation", "Nulls",
       "Permute z across galaxies, re-stack, and measure the integrated flux "
       "per bin many times: the scatter is the null with NO coherent line. Real "
       "flux above this band is the core detection test.",
       "null = redshift_scramble_null(cfg, product, n_realizations=200)"),
    _e("plot_scramble_null", "validation", "validation", "Nulls",
       "Real integrated flux vs radius over the z-scramble null band. Points "
       "above the grey envelope = real signal at that radius.",
       "plot_scramble_null(null)"),
    _e("placebo_wavelengths", "validation", "validation", "Nulls",
       "Run the estimator at many line-FREE rest wavelengths on the real "
       "stacks; the spread is a null built from the real continuum + noise. "
       "Catches estimator/continuum artifacts the z-scramble can't.",
       "plac = placebo_wavelengths(cfg, stacks)"),
    _e("plot_placebo", "validation", "validation", "Nulls",
       "Placebo null band with the real Lya measurement overlaid (flux by "
       "default). Real Lya should stand clear of the placebo cloud.",
       "plot_placebo(plac, real_boot=boot)"),
    _e("sweep_line_window", "validation", "validation", "Robustness sweeps",
       "Re-measure centroid AND flux for several line-window half-widths, each "
       "with its own bootstrap. Shows whether the result depends on where you "
       "draw the window.",
       "sw = sweep_line_window(cfg, stacks, half_windows=(3,4,5,6))"),
    _e("plot_window_sweep", "validation", "validation", "Robustness sweeps",
       "Overlay the chosen statistic vs radius across line windows. Tight "
       "clustering = window-insensitive result.",
       "plot_window_sweep(sw, statistic='centroid')"),
    _e("sweep_continuum_method", "validation", "validation", "Robustness sweeps",
       "Re-measure under several continuum settings (median vs poly, order, "
       "sidebands), each bootstrapped. Isolates how much the continuum choice "
       "moves the answer.",
       "cs = sweep_continuum_method(cfg, stacks)"),
    _e("plot_continuum_sweep", "validation", "validation", "Robustness sweeps",
       "Overlay the chosen statistic vs radius across continuum settings. "
       "Divergence flags a continuum-sensitive bin.",
       "plot_continuum_sweep(cs, statistic='centroid')"),
    _e("error_cross_check", "validation", "validation", "Error checks",
       "Cross-check the galaxy bootstrap error against a jackknife (optionally "
       "grouped/block) error per bin. A jackknife much larger than the "
       "bootstrap warns of a few dominant galaxies.",
       "ecc = error_cross_check(cfg, stacks, product)"),
    _e("plot_error_cross_check", "validation", "validation", "Error checks",
       "Same central values with bootstrap vs jackknife error bars overlaid, so "
       "you see directly whether the two error estimates agree.",
       "plot_error_cross_check(ecc)"),
    _e("collect_null_ensembles", "validation", "validation", "Error checks",
       "Gather every noise estimator (bootstrap, z-scramble, placebo) into one "
       "per-bin structure for the integrated flux. The input to the synthesis "
       "and calibration plots.",
       "nulls = collect_null_ensembles(cfg, stacks, product)"),
    _e("combined_bin_significance", "validation", "validation", "Combined significance",
       "Combine several radial bins into ONE 'is the mean centroid offset "
       "across this range really nonzero' test, using the covariance across "
       "bins induced by shared galaxies in the bootstrap draws already in "
       "real_boot -- so correlated neighboring bins aren't double-counted. The "
       "covariance-aware beyond-R_vir significance number (~1.7 sigma).",
       "combined_bin_significance(boot, r_min=1.0)"),
    _e("pool_bins_and_bootstrap", "validation", "validation", "Combined significance",
       "The flux-level counterpart: merge several radial bins into ONE wide bin "
       "BEFORE measuring a centroid, then bootstrap that single pooled bin "
       "fresh. Independent cross-check on combined_bin_significance (~1.5 "
       "sigma flux-pooled).",
       "pool_bins_and_bootstrap(stacks, r_min=1.0)"),
    _e("injection_recovery", "validation", "validation", "Injection-recovery",
       "Inject a synthetic Lya line of KNOWN velocity into the empty "
       "z-scrambled cube at fixed S/N, then recover it with each estimator. "
       "Quantifies centroid bias and bootstrap coverage -- the clip-bias test.",
       "inj = injection_recovery(cfg, product, snr=5.0)"),
    _e("injection_bias_fast", "validation", "validation", "Injection-recovery",
       "Fast sibling of injection_recovery with the per-draw bootstrap stripped "
       "out -- bias only, no coverage. Use for a quick estimator-bias scan.",
       "inj = injection_bias_fast(cfg, product)"),
    _e("plot_bias_summary", "validation", "validation", "Injection-recovery",
       "One figure that makes an injection result legible: recovered-vs-input, "
       "additive bias, and slope per radial bin. Works for either injection "
       "function.",
       "plot_bias_summary(inj)"),
    _e("plot_injection_recovery", "validation", "validation", "Injection-recovery",
       "Recovered vs input centroid for one radial bin with the 1:1 line. The "
       "close-up for a specific bin.",
       "plot_injection_recovery(inj, radial_index=0)"),
    _e("signal_to_noise_spectrum", "validation", "validation", "S/N spectrum",
       "Per-bin S/N(lambda) = (stack - continuum)/sigma using the empirical "
       "bootstrap per-pixel error. Shows where Lya -- and any other line -- is "
       "significant across the whole grid.",
       "sn = signal_to_noise_spectrum(cfg, stacks)"),
    _e("plot_sn_spectrum", "validation", "validation", "S/N spectrum",
       "One S/N(lambda) panel per radial bin with sigma guides and catalog-line "
       "markers. The 'is there signal, and only at Lya?' figure.",
       "plot_sn_spectrum(sn)"),
    _e("plot_sn_map", "validation", "validation", "S/N spectrum",
       "2-D radius-wavelength S/N map: all bins in one diverging-color image. "
       "The compact companion to plot_sn_spectrum.",
       "plot_sn_map(sn)"),
    _e("plot_continuum_check", "validation", "validation", "S/N spectrum",
       "Full rest-frame spectrum per bin with the sideband continuum overlaid. "
       "Confirms the continuum is anchored on clean regions and no line sits in "
       "a sideband.",
       "plot_continuum_check(stacks)"),
    _e("summarize_validation", "validation", "validation", "Synthesis",
       "Fuse the independent tests (S/N, placebo, error check, injection) into "
       "a per-bin PASS/WARN/FAIL verdict for the centroid. Consumes results you "
       "already computed.",
       "summ = summarize_validation(boot, sn, plac, ecc, inj)"),
    _e("print_validation_summary", "validation", "validation", "Synthesis",
       "Pretty per-bin table of a summarize_validation result. The text verdict.",
       "print_validation_summary(summ)"),
    _e("centroid_dossier", "validation", "validation", "Synthesis",
       "A single figure fusing all the centroid checks onto one shared radius "
       "axis, so 'is this bin real and trustworthy?' is one glance. The "
       "publication-ready validation panel.",
       "centroid_dossier(boot, sn, plac, ecc, inj, summ)"),
    _e("plot_null_synthesis", "validation", "validation", "Synthesis",
       "The verdict plot: integrated flux vs radius with EACH null drawn as its "
       "own +/-Nsigma band, so you see the spread the real flux must beat.",
       "plot_null_synthesis(nulls)"),
    _e("plot_null_envelope", "validation", "validation", "Synthesis",
       "The intuition plot: per bin, the real rest-frame stack spectrum on top "
       "of the z-scramble null envelope. Makes 'signal vs noise' visible line "
       "by line.",
       "zs = zswap_spectra(cfg, product); plot_null_envelope(zs)"),

    # ===================== SAMPLE =====================
    _e("slice_product", "selection", "sample", "Slice & select",
       "Slice every galaxy-axis array AND the catalog by a boolean mask, "
       "consistently. The primitive under every sample cut.",
       "sub = slice_product(product, product.catalog['z'] > 2.3)"),
    _e("cone_search", "selection", "sample", "Slice & select",
       "Select galaxies within a radius of a sky position. Handy for pulling a "
       "region or a single object's neighbourhood.",
       "sub = cone_search(product, ra=214.9, dec=52.9, radius_arcsec=60)"),
    _e("select_by", "selection", "sample", "Slice & select",
       "Select by an arbitrary criterion -- a column expression, a callable, or "
       "a mask. The flexible general-purpose selector.",
       "sub = select_by(product, lambda cat: cat['MASS_50'] > 10)"),
    _e("with_derived", "selection", "sample", "Slice & select",
       "Compute a per-galaxy quantity from catalog columns and attach it as a "
       "new catalog column (derive + with_column). Use before splitting on a "
       "derived property.",
       "product = with_derived(product, 'ssfr', lambda c: c['SFR']/10**c['MASS_50'])"),
    _e("top_sn_galaxies", "selection", "sample", "Slice & select",
       "Rank a CORE product (core.read_core_fits) by Lya S/N (sn_b, the fixed "
       "integration-window statistic, by default) and return the top N as a "
       "standalone core dict, printing a ranked table. Schema-aware (flat + "
       "hierarchical); the single-galaxy halo-triage front door.",
       "sub = top_sn_galaxies(core, n=20)"),
    _e("id_list", "selection", "sample", "Slice & select",
       "Plain Python list of catalog IDs from a (sliced) core or galaxy "
       "product, in its current row order. The hand-off from top_sn_galaxies "
       "to multicat.stack_id_list.",
       "ids = id_list(sub)[:5]"),
    _e("split_by_derived", "selection", "sample", "Split",
       "Split a product into sub-products by a DERIVED quantity (e.g. low/high "
       "sSFR). Returns a labelled dict ready for stack_each.",
       "parts = split_by_derived(product, lambda c: c['SFR'])"),
    _e("split_by_mask", "selection", "sample", "Split",
       "Split a product into sub-products by your own boolean mask (True/False "
       "groups). For an ad-hoc two-way cut you already computed.",
       "parts = split_by_mask(product, product.catalog['z'] > 2.3)"),
    _e("split_controlled", "selection", "sample", "Split",
       "Low/high split of one property while holding a control variable fixed "
       "(matched within control bins). Use to compare samples without a "
       "confounding mass/z difference.",
       "lo, hi = split_controlled(product.catalog, 'SFR', control_col='MASS_50')"),
    _e("split_by_redshift", "multicat", "sample", "Split",
       "Convenience split by the redshift column, by value edges or "
       "percentiles. The common low-z vs high-z cut in one call.",
       "parts = split_by_redshift(product, percentiles=(50,))"),
    _e("split_product_by", "multicat", "sample", "Split",
       "Split ONE product into sub-products by any catalog column (edges / "
       "percentiles / groups), optionally writing each to disk. The general "
       "splitter split_by_redshift wraps.",
       "parts = split_product_by(product, 'MASS_50', percentiles=(33,66))"),
    _e("load_products", "multicat", "sample", "Multi-catalog",
       "Load several galaxy FITS into one labelled dict of products. The entry "
       "point for any multi-field / multi-catalog workflow.",
       "prods = load_products({'AEGIS': cfg_a, 'COSMOS': cfg_c})"),
    _e("combine_products", "multicat", "sample", "Multi-catalog",
       "Concatenate a labelled dict of products along the galaxy axis, tagging "
       "each galaxy's origin in a catalog column. Pool fields into one sample.",
       "product = combine_products(prods)"),
    _e("match_catalog", "multicat", "sample", "Multi-catalog",
       "Positionally link a separate external catalog onto a product's galaxy "
       "axis. Returns the matched rows aligned to your galaxies.",
       "ext = match_catalog(product, external_table)"),
    _e("add_matched_column", "multicat", "sample", "Multi-catalog",
       "Return a copy of the product whose catalog carries a column from an "
       "external catalog, matched by sky position. For splitting on a property "
       "that lives in another file.",
       "product = add_matched_column(product, external_table, 'sSFR')"),
    _e("stack_each", "multicat", "sample", "Multi-catalog",
       "Run build_stacks on each product in a labelled dict -> {label: stacks}. "
       "The batch Stage-2 for a split sample.",
       "stacks_by = stack_each(cfg, parts)"),
    _e("stack_id_list", "multicat", "sample", "Multi-catalog",
       "Coadd a hand-picked list of galaxy IDs (e.g. from selection.id_list) "
       "into one mini halo profile. slice_product + build_stacks -- the ID-list "
       "front door for single-object candidates you want a stacked look at.",
       "mini = stack_id_list(cfg, prod, ids)"),
    _e("measure_each", "multicat", "sample", "Multi-catalog",
       "Run run_measure on each stacks dict -> {label: boot}. The batch Stage-3 "
       "for a split sample (needs keep_cube=True stacks).",
       "boot_by = measure_each(cfg, stacks_by)"),
    _e("compare_centroids", "multicat", "sample", "Multi-catalog",
       "Overlay centroid-vs-radius for several labelled samples with sensible "
       "multi-catalog defaults. The one-liner comparison figure for a split.",
       "compare_centroids(boot_by)"),
    _e("load_overdensity", "multicat", "sample", "Multi-catalog",
       "Load the Chartab et al. (2020) CANDELS environment-density catalog "
       "(VizieR J/ApJ/890/7) for one field. Folded in from the standalone "
       "environment.py (2026-07-21).",
       "od = load_overdensity('COSMOS')"),
    _e("attach_overdensity", "multicat", "sample", "Multi-catalog",
       "Positionally cross-match a product onto the Chartab overdensity catalog "
       "and attach the density column. add_matched_column, specialized for this "
       "one catalog. Folded in from environment.py (2026-07-21).",
       "product = attach_overdensity(product, field='COSMOS')"),
    _e("split_by_overdensity", "multicat", "sample", "Split",
       "Low/high environment-density split using the attached Chartab column, "
       "across one or both fields, with an optional mass-matching pass. "
       "split_product_by, specialized for this one catalog. Folded in from "
       "environment.py (2026-07-21).",
       "lo, hi = split_by_overdensity(product)"),
    _e("plot_sky", "selection", "sample", "Sky plots",
       "Scatter the sample on the sky, optionally highlighting a selection and "
       "drawing a cone. Sanity-check where a cut lands spatially.",
       "plot_sky(product)"),
    _e("plot_split_hist", "selection", "sample", "Sky plots",
       "Histogram of a (derived) quantity with the split boundaries drawn. "
       "Check that a split falls where you intended.",
       "plot_split_hist(product, lambda c: c['z'])"),

    # ===================== MISC =====================
    _e("virial_to_kpc_bins", "virial", "misc", "Virial conversions",
       "Convert R/Rvir bin edges to kpc for ONE galaxy's mass & z (Moster+2013 "
       "-> R200c). The per-galaxy mapping extraction uses; call it to see a "
       "galaxy's physical bin edges.",
       "kpc = virial_to_kpc_bins(cfg.bins, m=10.5, z=2.3)"),
    _e("virial_to_angular_bins", "virial", "misc", "Virial conversions",
       "Convert R/Rvir bin edges to arcsec for one galaxy's mass & z. Useful "
       "for overlaying bins on imaging.",
       "arcsec = virial_to_angular_bins(cfg.bins, m=10.5, z=2.3)"),
    _e("physical_kpc_to_arcsec", "virial", "misc", "Virial conversions",
       "Angular size (arcsec) of a physical size (kpc) at redshift z (Planck18). "
       "The basic cosmology conversion for annotating plots.",
       "theta = physical_kpc_to_arcsec(100.0, z=2.3)"),
    _e("estimate_M200c_R200c_from_Mstar", "virial", "misc", "Virial conversions",
       "Stellar mass -> (M200c, R200c) via Moster+2013 + R200c(z). The halo "
       "estimate behind the virial binning.",
       "M200c, R200c = estimate_M200c_R200c_from_Mstar(10**10.5, z=2.3)"),
    _e("median_virial_kpc_bins", "virial", "misc", "Virial conversions",
       "Sample-median mapping of R/Rvir edges -> kpc (median mass & z), for "
       "plots/tables only. NOT how the pipeline bins -- that is per-galaxy.",
       "kpc, m_med, z_med = median_virial_kpc_bins(cfg.bins, run_table)"),
    _e("stack_lsf", "speclsf", "misc", "LSF / intrinsic profile",
       "Build each galaxy's observed-frame line-spread kernel and route them "
       "through the SAME Stage-2 path as the science line to get the stacked "
       "rest-frame LSF. The instrumental resolution the stacked line sees.",
       "lsf = stack_lsf(cfg, product)"),
    _e("deconvolve_intrinsic_profile", "speclsf", "misc", "LSF / intrinsic profile",
       "Forward-model the intrinsic Lya profile: convolve a parametric model "
       "with the stacked LSF and fit the stacked line (single vs double peak "
       "via BIC/AIC). Recovers the true line width/peak separation.",
       "dec = deconvolve_intrinsic_profile(stacks['rest_wave'], stacks['stacks']['biweight']['flux'][0], err, lsf)"),
    _e("bootstrap_intrinsic_profile", "speclsf", "misc", "LSF / intrinsic profile",
       "Galaxy-resampling bootstrap of the intrinsic-profile fit for one radial "
       "bin -> confidence intervals on intrinsic sigma / peak separation.",
       "ib = bootstrap_intrinsic_profile(stacks, lsf, radial_bin=0, config=cfg)"),
    _e("plot_lsf", "plotting", "misc", "LSF / intrinsic profile",
       "Overview of a stack_lsf result: the stacked LSF with its FWHM, and "
       "(optionally) the science line on the same axis for a resolution "
       "comparison.",
       "plot_lsf(lsf, stacks)"),
    _e("plot_intrinsic_fit", "plotting", "misc", "LSF / intrinsic profile",
       "Diagnostic for deconvolve_intrinsic_profile: stacked line, the "
       "(intrinsic x LSF) model, and the deconvolved intrinsic profile "
       "overlaid. Confirms the deconvolution is sane.",
       "plot_intrinsic_fit(stacks['rest_wave'], stack_flux, stack_err, dec, lsf)"),
    _e("make_star_run_table", "starpsf", "misc", "Star-PSF & line profiles",
       "Build a run table at star positions so extraction produces a stellar "
       "PSF profile. The input to the star-PSF validation extraction.",
       "star_tbl = make_star_run_table(cfg, ra_stars, dec_stars)"),
    _e("make_random_sky_table", "starpsf", "misc", "Star-PSF & line profiles",
       "Random in-footprint blank-sky positions (rejecting real sources) for a "
       "source-free null. Feeds run_positions_extract for the blank-sky test.",
       "sky_tbl = make_random_sky_table(cfg, n=500)"),
    _e("run_positions_extract", "starpsf", "misc", "Star-PSF & line profiles",
       "Run Stage-1 extraction on a synthesized position table (stars or sky), "
       "returning an in-memory product. The extraction driver for the PSF/null "
       "tables.",
       "sky_product = run_positions_extract(cfg, sky_tbl)"),
    _e("radial_continuum_profile", "starpsf", "misc", "Star-PSF & line profiles",
       "Observed-frame continuum surface brightness per radial bin, coadded "
       "across objects. The stellar-PSF (or galaxy-continuum) radial profile.",
       "prof = radial_continuum_profile(star_product)"),
    _e("curve_of_growth", "starpsf", "misc", "Star-PSF & line profiles",
       "Cumulative enclosed flux vs radius from a radial_continuum_profile. "
       "Compare the Lya profile against the PSF curve of growth to test for "
       "extension.",
       "cog = curve_of_growth(prof)"),
    _e("blank_sky_centroid_null", "starpsf", "misc", "Star-PSF & line profiles",
       "Stack a blank-sky product and measure the 'Lya' centroid per bin: it "
       "should hug zero. A source-free null for the centroid itself.",
       "bs = blank_sky_centroid_null(cfg, sky_product)"),
    _e("measure_line_profiles", "starpsf", "misc", "Star-PSF & line profiles",
       "Radial profile of integrated flux (and centroid) for a list of lines, "
       "measured on the stacks EXACTLY as Lya. For interstellar lines / "
       "cross-checks on the same stack.",
       "lp = measure_line_profiles(cfg, stacks, lines)"),
    _e("bootstrap_line_profiles", "starpsf", "misc", "Star-PSF & line profiles",
       "Galaxy-resampling bootstrap of the line-profile measurement, with a "
       "per-line S/N ranking. The error-bar version of measure_line_profiles.",
       "lb = bootstrap_line_profiles(cfg, stacks, lines)"),
    _e("inspect_core_background", "core", "misc", "Core / escape-fraction",
       "Diagnostic: compute the per-exposure + pooled annulus background for a "
       "few galaxies, RAW and SMOOTHED, WITHOUT running the full PSF core "
       "extraction. Returns exactly what extract_core_spectra would subtract, "
       "so you can check a suspicious core spectrum (e.g. negative continuum "
       "or absorption lines dipping below zero) against its own background "
       "before deciding the subtraction scheme needs to change.",
       "bg = inspect_core_background(cfg, [17632, 30032], run_table=run_table)"),
    _e("extract_core_spectra", "core", "misc", "Core / escape-fraction",
       "C1: PSF-weighted optimal extraction of ONE aperture spectrum per galaxy "
       "(the core), a socket over optics.py. The start of the deferred "
       "escape-fraction track; also the engine behind the mask A/B tool.",
       "Spec, Err, wave, cat = extract_core_spectra(cfg)"),
    _e("run_core_hierarchical", "core", "misc", "Core / escape-fraction",
       "The two-tier core measurement pass: detect, then measure/classify and "
       "(optionally) escape fraction for every galaxy. The main driver of the "
       "core / escape-fraction analysis.",
       "results = run_core_hierarchical(core_spec, core_err, wave_obs, catalog)"),
    _e("lya_escape_fraction", "core", "misc", "Core / escape-fraction",
       "Lya escape fraction anchored on dust-corrected Hbeta (Case B). The "
       "physics endpoint of the core track.",
       "fesc = lya_escape_fraction(Flya, e_Flya, Hb, e_Hb, ebv, e_ebv)"),
    _e("compare_mask_spectra", "core", "misc", "Core / escape-fraction",
       "Extract a few galaxies' core spectra once per candidate bad-fiber mask "
       "and overlay them -- the A/B instrument for validating a new mask "
       "against the old. Warns when two masks flag identical core fibers.",
       "fig, res = compare_mask_spectra(cfg, [17632, 30032], masks=[('img', bad_img), ('seg', bad_seg)])"),
    _e("diagnose_core_fibers", "core", "misc", "Core / escape-fraction",
       "Explain WHY a core spectrum is NaN (or confirm it should extract) for a "
       "few galaxies, counting fibers at each stage. The debugger for empty "
       "cores.",
       "diagnose_core_fibers(cfg, [17632, 30032])"),
    _e("audit_galaxy_mask_overlap", "core", "misc", "Core / escape-fraction",
       "Per-galaxy count of masked fibers near each galaxy's own center -- 'does "
       "the bad-fiber mask overlap this galaxy?'. Drives the appendix figure "
       "and the self-masking audit.",
       "audit = audit_galaxy_mask_overlap(vdfi, run_table, bad)"),
    _e("read_core_fits", "core", "misc", "Core / escape-fraction",
       "Read a core product FITS back into plain arrays (flat or hierarchical "
       "schema). The loader for a saved core extraction.",
       "core = read_core_fits('outputs/core_AEGIS.fits')"),
    _e("select_core", "core", "misc", "Core / escape-fraction",
       "Pull galaxies from a core product by index, catalog id, or sky cone "
       "(pick one). The sample selector for the core track.",
       "sub = select_core(core, id_value=17632)"),
    _e("core_mask", "core", "misc", "Core / escape-fraction",
       "Boolean per-galaxy mask from a criterion over the core results+catalog "
       "table (e.g. emitters). Feeds select/slice for the core sample.",
       "emitters = core_mask(core, lambda t: t['sn'] > 5)"),

    # ---- Single galaxy (radius-resolved spectra for ONE bright object) ----
    _e("single_galaxy_spectra", "stack", "misc", "Single galaxy",
       "build_stacks minus the galaxy-axis coadd: radius-resolved rest-frame "
       "spectra for ONE galaxy (no re-extraction -- product.spec already holds "
       "it). For bright single objects (AGN, strong LAEs, double-peakers) found "
       "via selection.top_sn_galaxies. Optional merge_bins trades radial "
       "resolution for S/N.",
       "sg = single_galaxy_spectra(cfg, product, gid=12345)"),
    _e("plot_single_galaxy_panels", "plotting", "misc", "Single galaxy",
       "One panel per radial bin for a single_galaxy_spectra product, "
       "annotated with each bin's fiber count and crude line S/N -- how far "
       "out this object is trustworthy. plot_stack_panels(sg, method='galaxy') "
       "also works directly; this adds the nfib/S/N annotation.",
       "plot_single_galaxy_panels(sg)"),
    _e("measure_single_galaxy", "measure", "misc", "Single galaxy",
       "Per-bin centroid velocity + integrated flux for a single_galaxy_spectra "
       "product. No bootstrap (one object can't be resampled over galaxies); "
       "trust only bins where sg['sn'] is decent.",
       "m = measure_single_galaxy(sg, method='flux_weighted')"),

    # ---- Core characterization (moved from core.py) ----
    _e("run_core_characterization", "validation", "validation", "Core characterization",
       "One-call driver for the whole per-galaxy core-credibility suite: runs the "
       "synthetic + real-noise injections, the redshift null, and (optionally) the "
       "estimator comparison, extracting completeness curves along the way. Every "
       "sub-function's parameters are exposed here (syn_/samp_/null_/compare_) so "
       "nothing runs on a hidden default.",
       "cc = run_core_characterization(Spec, Err, wave_obs, catalog)"),
    _e("injection_recovery_synthetic", "validation", "validation", "Core characterization",
       "Method characterization on SYNTHETIC noise: inject a Lya line of known "
       "flux many times per level and report each estimator's bias, scatter, and "
       "detection rate. The input-flux=0 case exposes peak-chasing bias and sets "
       "the flux floor. (This is core's old injection_recovery, renamed to avoid "
       "clashing with validation's cube-based one.)",
       "inj = injection_recovery_synthetic(wave_obs, np.nanmedian(Err, axis=0))"),
    _e("injection_recovery_sample", "validation", "validation", "Core characterization",
       "Realistic completeness: inject at controlled input S/N into each galaxy's "
       "ACTUAL noise spectrum, so the recovery fraction reflects the real sample "
       "-- 'at input S/N=3, what fraction would I detect?'.",
       "inj = injection_recovery_sample(Err, wave_obs, catalog)"),
    _e("detection_completeness", "validation", "validation", "Core characterization",
       "Pull the completeness curve (detection rate vs input flux or input S/N) "
       "out of either injection result. The summarizing step you'd plot or quote.",
       "x, rate = detection_completeness(inj)"),
    _e("null_redshift_test", "validation", "validation", "Core characterization",
       "False-positive test: permute redshifts and measure 'Lya' at the WRONG z, "
       "so any line found is spurious. Calibrates the per-galaxy significance "
       "threshold (the per-galaxy analog of the z-scramble stack null).",
       "nz = null_redshift_test(Spec, Err, wave_obs, catalog['z'])"),
    _e("compare_estimators", "validation", "validation", "Core characterization",
       "Run all three core flux estimators on ONE rest-frame spectrum and "
       "tabulate their flux / S-N / classification side by side. A fast "
       "'do my estimators agree on this galaxy?' check.",
       "from utils_lya_halo.core import to_rest_frame\n"
       "wr, fr, er = to_rest_frame(wave_obs, Spec[0], Err[0], catalog['z'][0])\n"
       "tab = compare_estimators(wr, fr, er)"),

    # ===================== METHOD SCORING (optimize.py) =====================
    _e("load_products", "optimize", "validation", "Method scoring",
       "{label: fits_path} + ONE PipelineConfig -> {label: GalaxyProduct}, each "
       "already apply_finite_cut. Kills the products_testN boilerplate for "
       "comparing several already-extracted galaxy FITS side by side -- one "
       "shared config is enough since background/masking/smoothing are already "
       "baked into each FITS (see specs/optimize.md). NOTE: multicat also has a "
       "load_products (different signature -- a list of label/path/config "
       "specs); this one is optimize.load_products specifically.",
       "products = load_products({'100G': path1, 'im_200A': path2}, cfg_test)"),
    _e("build_stacks_many", "optimize", "validation", "Method scoring",
       "{label: GalaxyProduct} + ONE config -> {label: build_stacks(...) "
       "result}. The batch call that follows load_products (or "
       "match_products(...)['products']) and precedes noise_from_stacks. "
       "config_for is an escape hatch -- {label: override_config} -- for the "
       "rare label that needs different measure-phase settings.",
       "stacks = build_stacks_many(products, cfg_test)"),
    _e("noise_from_stacks", "optimize", "validation", "Method scoring",
       "Score ONE product's continuum noise: bootstrap the galaxy axis of a "
       "keep_cube=True stacks dict, measure per-bin scatter in a line-masked "
       "sideband, and report noise_per_bin + height_per_bin (+ the per-pixel "
       "err/continuum spectra). The building block every scoring view consumes; "
       "the RANKING never touches the line itself (no injection / no S-N-of-Lya "
       "tuning) -- pass compute_line_snr=True for an opt-in, diagnostic-only "
       "line-S/N bootstrap alongside it (see plot_line_snr_summary), which "
       "stays out of the ranking on purpose.",
       "res = noise_from_stacks(stacks, cfg, nboot=200, reduce='rms', height_reduce='biweight')"),
    _e("run_combine_sweep", "optimize", "validation", "Method scoring",
       "Score several combine-level variants (stack_method / sigma_clip_*) "
       "against ONE already-extracted cube -- no re-extraction. Returns a "
       "Table, one row per variant, carrying noise_per_bin.",
       "tbl = run_combine_sweep(cfg, stacks, [{'measure_stack_method': m} for m in ('biweight','median')])"),
    _e("rank_scores", "optimize", "validation", "Method scoring",
       "Collapse a {label: noise_from_stacks result} dict into two per-method "
       "numbers: noise_product (per-bin errors multiplied across ALL bins in "
       "log space, then shifted by one common power of ten so the best lands "
       "in ~1-10 -- ratios preserved, NOT divided by the minimum) and mean_rank "
       "(mean over bins of each method's within-bin error rank). Lower = better "
       "for both. Anchor the methods to the same galaxies/bins first.",
       "R = rank_scores(scores)              # then plot_score_summary(scores)"),
    _e("score_table", "optimize", "validation", "Method scoring",
       "rank_scores() as a printable, best-first astropy Table (label, "
       "noise_product, mean_rank, n_bins_used). The text ranking that sits "
       "beside the histograms.",
       "score_table(scores).pprint_all()"),
    _e("plot_score_summary", "optimize", "validation", "Method scoring",
       "The two summary histograms: one bar per method for noise_product and "
       "for mean_rank across all bins, colored green=good -> red=bad (the "
       "mean_rank scale is graded against 1..M, not min..max, so a genuinely "
       "good rank looks good). Returns (fig, score_table). All knobs "
       "strung through (bin_slice, sort_by, rank_gamma, log_y_noise, axes...).",
       "fig, tbl = plot_score_summary(scores)"),
    _e("plot_bin_detail", "optimize", "validation", "Method scoring",
       "Per-bin drill-down (separate from the all-bins summary): pick ONE "
       "radial bin (negative indexes from outside) and show, one bar per "
       "method, the error (green=good) and the signed continuum height "
       "(diverging about zero, with a zero line). Returns the figure.",
       "fig = plot_bin_detail(scores, -1)    # outermost bin; also 0, 4, ..."),
    _e("plot_line_snr_summary", "optimize", "validation", "Method scoring",
       "DIAGNOSTIC companion to plot_score_summary, for scores built with "
       "compute_line_snr=True: one bar per method for mean line S/N "
       "(higher=better, RdYlGn not _r), plus line S/N vs radial bin, one "
       "curve per method. Never consumed by rank_scores/score_table -- see "
       "noise_from_stacks; this is 'does minimizing continuum noise raise "
       "line S/N', plotted, not ranked.",
       "fig, tbl = plot_line_snr_summary(scores)"),

    # =====================================================================
    # fitting.py -- radial flux-profile fitting (registered 2026-07-21;
    # previously not a contributing module, so check_guide() never scanned
    # it. diagnose_crossover_failures/summarize_diagnosed_params were
    # deleted the same day (one-off investigative tooling, task done) so
    # they are NOT listed here.
    # =====================================================================
    _e("moffat_1d", "fitting", "fitting", "PSF & convolution",
       "Unnormalized Moffat PSF profile (1+(r/alpha)^2)^-beta, alpha derived "
       "from FWHM/beta. The literature-grounded PSF form used throughout "
       "this pipeline's fits.",
       "psf = moffat_1d(r_fine, fwhm=1.3, beta=3.0)"),
    _e("normalize_psf_flux", "fitting", "fitting", "PSF & convolution",
       "Rescale a 1D radial PSF array so integral(psf(r)*2*pi*r dr) == 1 -- "
       "flux-conserving normalization for any PSF curve, empirical or analytic.",
       "psf_norm = normalize_psf_flux(r_fine, psf)"),
    _e("moffat_encircled_energy_fraction", "fitting", "fitting", "PSF & convolution",
       "Fraction of a point source's total flux landing within a circular "
       "aperture of a given radius, for a Moffat PSF. Feeds the innermost-bin "
       "aperture correction.",
       "ee = moffat_encircled_energy_fraction(r_ap, fwhm=1.3, beta=3.0)"),
    _e("default_fine_grid", "fitting", "fitting", "PSF & convolution",
       "Build a sensible r_fine grid: n_per_bin points spread across EACH "
       "radial bin, for evaluating the intrinsic profile before PSF convolution.",
       "r_fine = default_fine_grid(r_edges, n_per_bin=20)"),
    _e("ring_convolution_matrix", "fitting", "fitting", "PSF & convolution",
       "R[i,j] = fraction of a unit-flux thin ring at r_fine[j] observed in "
       "radial bin i, after PSF blurring -- the closed-form 1D forward-model "
       "matrix every PSF-aware fit multiplies against.",
       "R = ring_convolution_matrix(r_edges, r_fine, fwhm=1.3, beta=3.0)"),
    _e("validate_ring_convolution", "fitting", "fitting", "PSF & convolution",
       "Regression check: compares the fast trapz-based ring convolution "
       "against a slow direct-integration reference, confirming flux "
       "conservation out to large radius. A test, not a measurement.",
       "validate_ring_convolution(r_edges, r_fine, fwhm=1.3, beta=3.0)"),
    _e("intrinsic_profile", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "I(r) = A1*exp(-r/h1) + A2*exp(-r/h2), h1<h2 by convention. The Part 1 "
       "two-exponential intrinsic (pre-PSF) flux-profile model.",
       "y = intrinsic_profile(r_fine, A1, h1, A2, h2)"),
    _e("estimate_truth_from_profile", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "Quick, no-optimizer, two-exponential ballpark estimate (A1,h1,A2,h2) "
       "from the binned data alone -- a seed generator, not a fit.",
       "p0 = estimate_truth_from_profile(r_mid, y_fid)"),
    _e("bin_average_no_psf", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "Mean intrinsic (no-PSF) flux in each bin: the fine-grid integral "
       "divided by bin width. What fit_naive predicts against.",
       "y_bin = bin_average_no_psf(r_edges, r_fine, A1, h1, A2, h2)"),
    _e("bin_average_psf", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "Mean PSF-smeared flux in each bin: (R @ fine_flux)/bin width. What "
       "fit_psf_aware predicts against.",
       "y_bin = bin_average_psf(R, r_fine, A1, h1, A2, h2)"),
    _e("fit_naive", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "Two-exponential fit with NO PSF correction: drops the inner bin(s) "
       "(fit_skip_inner) instead, on the assumption the PSF only meaningfully "
       "affects those. Multi-seed-then-lowest-chi2, amplitude-normalized.",
       "result = fit_naive(r_mid, r_edges, r_fine, y_fid, sigma)"),
    _e("fit_psf_aware", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "Two-exponential fit WITH the full PSF forward model (ring_convolution_"
       "matrix) -- every bin fit, including the innermost. The PSF-aware "
       "headline fit this pipeline defaults to.",
       "result = fit_psf_aware(r_mid, y_fid, sigma, R, r_fine, r_edges)"),
    _e("describe_fit", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "Pretty-print a fit_naive/fit_psf_aware result: recovered (A1,h1,A2,h2) "
       "with 1-sigma covariance errors, chi2/dof, success flag.",
       "describe_fit(result, label='full stack')"),
    _e("binned_model_from_result", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "The fit's prediction in the SAME per-bin average-flux units as the "
       "data, for overlaying a fitted curve's predicted bin means on a plot.",
       "y_pred = binned_model_from_result(result, r_fine, r_edges, R)"),
    _e("crossover_radius_twoexp", "fitting", "fitting", "Two-exponential fit (Part 1)",
       "Radius where the two-exponential model's two terms are equal -- "
       "closed-form/root-find on the fitted A1,h1,A2,h2.",
       "r_x = crossover_radius_twoexp(result)"),
    _e("intrinsic_profile_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "I(r) = A1*exp(-r/h1) + A2*(1+(r/r_c)^2)^(-gamma/2) -- Part 2's "
       "exponential-core + cored-power-law model, the two-halo term's "
       "physically-motivated stand-in for a clustering profile.",
       "y = intrinsic_profile_expcore(r_fine, A1, h1, A2, r_c, gamma)"),
    _e("bin_average_no_psf_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Mean intrinsic (no-PSF) expcore flux in each bin -- same role as "
       "bin_average_no_psf for the expcore model.",
       "y_bin = bin_average_no_psf_expcore(r_edges, r_fine, A1, h1, A2, r_c, gamma)"),
    _e("bin_average_psf_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Mean PSF-smeared expcore flux in each bin -- same role as "
       "bin_average_psf for the expcore model.",
       "y_bin = bin_average_psf_expcore(R, r_fine, A1, h1, A2, r_c, gamma)"),
    _e("fit_naive_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Expcore analogue of fit_naive: no PSF correction, drops fit_skip_inner "
       "bins. As of 2026-07-21: A2=f*A1 reparametrized (forces A1>A2) and "
       "r_c_fixed defaults to 400.0 (mirrors gamma_fixed's None-means-free "
       "convention) -- see [[lya-halos-expcore-ordering-rc-fixed-default]].",
       "result = fit_naive_expcore(r_mid, r_edges, r_fine, y_fid, sigma)"),
    _e("fit_psf_aware_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Expcore analogue of fit_psf_aware: full PSF forward model, all bins "
       "fit. The pipeline-standard expcore fit as of 2026-07-21 (A2=f*A1 "
       "ordering + r_c_fixed=400.0 default).",
       "result = fit_psf_aware_expcore(r_mid, y_fid, sigma, R, r_fine, r_edges)"),
    _e("describe_fit_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Pretty-print an expcore fit result -- same role as describe_fit, "
       "for the expcore model's parameters.",
       "describe_fit_expcore(result, label='low-density')"),
    _e("binned_model_from_result_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Expcore analogue of binned_model_from_result: the fit's prediction "
       "in per-bin average-flux units.",
       "y_pred = binned_model_from_result_expcore(result, r_fine, r_edges, R)"),
    _e("crossover_radius_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Radius where the expcore model's core and two-halo terms are equal -- "
       "root-find over a default (50*h1, 5*r_c)-scaled bracket. NOT r_c-"
       "independent -- report conditional on r_c_fixed.",
       "r_x = crossover_radius_expcore(result)"),
    _e("projected_slope_from_3d", "fitting", "fitting", "Expcore fit (Part 2)",
       "Limber approximation: converts a 3D correlation slope gamma_3d to "
       "the expected 2D-projected outer slope. Grounds expcore's gamma_fixed "
       "default (0.8) in the Limber-projected z~2-3 clustering slope.",
       "gamma_2d = projected_slope_from_3d(1.8)"),
    _e("deprojected_slope_bounds", "fitting", "fitting", "Expcore fit (Part 2)",
       "The two endpoints bracketing the expected 2D outer slope for a 3D "
       "clustering slope range -- used to sanity-check gamma_fixed choices.",
       "lo, hi = deprojected_slope_bounds()"),
    _e("effective_slope_expcore", "fitting", "fitting", "Expcore fit (Part 2)",
       "Local (point-by-point) log-log slope of the two-halo term ALONE, "
       "for overlaying against the total fitted profile's slope on a "
       "slope-vs-radius panel.",
       "slope = effective_slope_expcore(r_fine, r_c, gamma)"),
    _e("radius_of_slope_fraction", "fitting", "fitting", "Expcore fit (Part 2)",
       "Closed-form radius where the two-halo term's local log-log slope "
       "reaches a given fraction of its asymptotic (-gamma) value.",
       "r_frac = radius_of_slope_fraction(r_c, gamma, frac=0.9)"),
    _e("compare_models_aic_bic", "fitting", "fitting", "Model comparison & boundary",
       "Compare two fit results (e.g. twoexp vs expcore, or free-r_c vs "
       "r_c_fixed) via chi2/AIC/BIC. Used to justify r_c_fixed=400.0 as the "
       "2026-07-21 default -- see docs/research-notes.md.",
       "compare_models_aic_bic(result_a, result_b, label_a='free', label_b='fixed')"),
    _e("find_core_halo_boundary", "fitting", "fitting", "Model comparison & boundary",
       "THE core/two-halo boundary for one subsample's fit_result: the "
       "radius where the fitted core and two-halo terms cross, with a "
       "fallback-radius/fallback-fit chain if the fit itself has no crossing. "
       "Feeds measure.py's zone-based derived properties.",
       "boundary = find_core_halo_boundary(fit_result)"),
    _e("sphere_of_influence_kpc", "fitting", "fitting", "Model comparison & boundary",
       "Sorini+2018's ~7x-R_vir 'sphere of influence' scale -- the radius "
       "beyond which the two-halo/clustering term has fully taken over, for "
       "comparing against a fitted r_c.",
       "r_soi = sphere_of_influence_kpc(rvir_kpc=75.0)"),
    _e("local_loglog_slope", "fitting", "fitting", "Model comparison & boundary",
       "Generic numerical local log-log slope d(ln y)/d(ln r) via central "
       "differences, for any curve (not just the expcore two-halo term).",
       "slope = local_loglog_slope(r_fine, y_fine)"),
    _e("sersic_bn", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Ciotti & Bertin (1999) approximation for b_n, the constant in the "
       "Sersic profile's exponent -- feeds intrinsic_profile_uv_sersic.",
       "bn = sersic_bn(n=1.0)"),
    _e("intrinsic_profile_uv_exp", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "I(r) = A*exp(-r/h_uv) -- Part 3's default single-exponential "
       "UV-continuum model.",
       "y = intrinsic_profile_uv_exp(r_fine, A, h_uv)"),
    _e("intrinsic_profile_uv_sersic", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "I(r) = A*exp(-b_n*[(r/r_e)^(1/n) - 1]) -- Part 3's optional Sersic "
       "alternative to the single exponential.",
       "y = intrinsic_profile_uv_sersic(r_fine, A, r_e, n)"),
    _e("bin_average_no_psf_uv_exp", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Mean intrinsic single-exponential UV flux in each bin, no PSF.",
       "y_bin = bin_average_no_psf_uv_exp(r_edges, r_fine, A, h_uv)"),
    _e("bin_average_psf_uv_exp", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Mean PSF-smeared single-exponential UV surface brightness in each bin.",
       "y_bin = bin_average_psf_uv_exp(R, r_fine, A, h_uv)"),
    _e("bin_average_no_psf_uv_sersic", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Sersic analogue of bin_average_no_psf_uv_exp.",
       "y_bin = bin_average_no_psf_uv_sersic(r_edges, r_fine, A, r_e, n)"),
    _e("bin_average_psf_uv_sersic", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Sersic analogue of bin_average_psf_uv_exp.",
       "y_bin = bin_average_psf_uv_sersic(R, r_fine, A, r_e, n)"),
    _e("describe_fit_uv_exp", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Pretty-print a single-exponential UV fit result -- same role as "
       "describe_fit_expcore, for the UV-continuum model.",
       "describe_fit_uv_exp(result)"),
    _e("describe_fit_uv_sersic", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Pretty-print a Sersic UV fit result.",
       "describe_fit_uv_sersic(result)"),
    _e("fit_naive_uv_exp", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Single-exponential UV-continuum fit with NO PSF -- drop-inner-bin(s) "
       "convention, same as fit_naive.",
       "result = fit_naive_uv_exp(r_mid, r_edges, r_fine, y_fid, sigma)"),
    _e("fit_psf_aware_uv_exp", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Single-exponential UV-continuum fit WITH the PSF forward model -- "
       "the real UV/Lya-scale-ratio result (h1~16-17kpc Lya, UV~1.55kpc, "
       "ratio ~10:1) came from this fit.",
       "result = fit_psf_aware_uv_exp(r_mid, y_fid, sigma, R, r_fine, r_edges)"),
    _e("fit_naive_uv_sersic", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Sersic UV-continuum fit with NO PSF -- same naive/drop-inner-bin(s) "
       "convention as fit_naive_uv_exp.",
       "result = fit_naive_uv_sersic(r_mid, r_edges, r_fine, y_fid, sigma)"),
    _e("fit_psf_aware_uv_sersic", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Sersic UV-continuum fit WITH the PSF forward model -- same structure "
       "as fit_psf_aware_uv_exp.",
       "result = fit_psf_aware_uv_sersic(r_mid, y_fid, sigma, R, r_fine, r_edges)"),
    _e("binned_model_from_result_uv_exp", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "UV-exponential analogue of binned_model_from_result.",
       "y_pred = binned_model_from_result_uv_exp(result, r_fine, r_edges, R)"),
    _e("binned_model_from_result_uv_sersic", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "UV-Sersic analogue of binned_model_from_result_expcore.",
       "y_pred = binned_model_from_result_uv_sersic(result, r_fine, r_edges, R)"),
    _e("plot_uv_fit", "fitting", "fitting", "UV-continuum fit (Part 3)",
       "Notebook-testing convenience for halo-flux-fitting.md Part 3's UV "
       "fit -- quick-look plot of a UV coadd + its fitted curve.",
       "plot_uv_fit(r_mid, y_fid, result, r_fine)"),

    # =====================================================================
    # uv_profile.py -- Part 3's standalone UV-continuum pipeline (registered
    # 2026-07-21; previously not a contributing module). Per CHANGELOG
    # 2026-07-15, this file is a "standalone testbed, not yet merged into
    # extract.py" -- real results have come out of it (the UV/Lya scale
    # ratio, see [[uv-lya-scale-result]]), but it isn't wired into the main
    # Stage 1-3 driver path the way measure.py/analysis.py are.
    # =====================================================================
    _e("resolve_uv_image_path", "uv_profile", "uv_continuum", "Field & catalog I/O",
       "Per-field CFHT-LS r-band image path, from the normalized field name.",
       "path = resolve_uv_image_path('COSMOS')"),
    _e("resolve_uv_segmap_path", "uv_profile", "uv_continuum", "Field & catalog I/O",
       "Per-field background segmap path, or None if not configured for "
       "that field.",
       "path = resolve_uv_segmap_path('COSMOS')"),
    _e("read_seeing_fwhm_arcsec", "uv_profile", "uv_continuum", "Field & catalog I/O",
       "Seeing FWHM (arcsec) for a field, tried from each of the configured "
       "FITS header keys in turn.",
       "fwhm = read_seeing_fwhm_arcsec('COSMOS')"),
    _e("get_uv_sample_catalog", "uv_profile", "uv_continuum", "Field & catalog I/O",
       "Pull the RA/Dec/z/mass sample straight from the already-built Lya "
       "product -- the same galaxies the halo-profile pipeline uses.",
       "catalog = get_uv_sample_catalog(product)"),
    _e("compute_field_bbox", "uv_profile", "uv_continuum", "Field & catalog I/O",
       "RA/Dec bounding box for one field's subsample of the catalog, "
       "padded by a margin -- sizes build_field_cutout.",
       "bbox = compute_field_bbox(catalog, 'COSMOS')"),
    _e("validate_cutout_covers_sample", "uv_profile", "uv_continuum", "Field & catalog I/O",
       "Check whether an existing cutout FITS's WCS footprint covers every "
       "galaxy in the sample -- a sanity check before extraction, not a "
       "measurement.",
       "validate_cutout_covers_sample(cutout_path, catalog, 'COSMOS')"),
    _e("build_field_cutout", "uv_profile", "uv_continuum", "Field & catalog I/O",
       "Build a WCS-aware cutout FITS from the full ~1 deg^2 mosaic, sized "
       "to cover the field's sample with margin.",
       "cutout_path = build_field_cutout('COSMOS', catalog)"),
    _e("psf_fwhm_kpc_for_z", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "Convert an angular seeing FWHM (arcsec) to a physical FWHM (kpc) at "
       "one galaxy's redshift.",
       "fwhm_kpc = psf_fwhm_kpc_for_z(fwhm_arcsec, z)"),
    _e("make_galaxy_cutout", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "Small WCS-aware stamp around (ra, dec), sized by size_arcsec -- the "
       "per-galaxy starting point for centroiding and photometry.",
       "cutout = make_galaxy_cutout(field_cutout, ra, dec, size_arcsec=10)"),
    _e("centroid_galaxy", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "2D-Gaussian centroid (photutils.centroid_2dg) on a galaxy cutout, "
       "windowed to avoid neighbor contamination. Kept-but-flagged, never "
       "silently dropped, if it fails.",
       "xc, yc, ok = centroid_galaxy(cutout)"),
    _e("align_segmap_to_grid", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "Reproject a segmentation-mask FITS onto a target (wcs, shape) grid -- "
       "aligns the neighbor-masking segmap to a galaxy cutout's own pixels.",
       "seg_aligned = align_segmap_to_grid(segmap, target_wcs, target_shape)"),
    _e("compute_field_background", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "Global background level for one field's cutout: masked-median of "
       "unmasked sky pixels.",
       "bg = compute_field_background(field_cutout, segmap)"),
    _e("radial_bin_edges_uv", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "Per-galaxy kpc bin edges converted to arcsec, using that galaxy's "
       "own z -- reuses the halo-profile pipeline's bin_mode convention.",
       "edges_arcsec = radial_bin_edges_uv(r_edges_kpc, z)"),
    _e("measure_uv_annuli", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "Per-annulus flux, background-subtracted, reduced across pixels via "
       "masked-median with segmap-based neighbor masking.",
       "flux_per_bin = measure_uv_annuli(cutout, edges_arcsec, bg, segmap)"),
    _e("extract_uv_profiles_for_field", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "Single-field-at-a-time loop over every galaxy in the catalog: "
       "cutout -> centroid -> background -> annuli, one dict per galaxy.",
       "profiles = extract_uv_profiles_for_field('COSMOS', catalog, cfg)"),
    _e("summarize_uv_extraction", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "QC summary over a list of per-galaxy extraction dicts -- centroid "
       "success rate, flagged galaxies, basic sanity stats. Diagnostic, not "
       "part of the measurement itself.",
       "summarize_uv_extraction(profiles)"),
    _e("run_uv_extraction_testbed", "uv_profile", "uv_continuum", "Per-galaxy extraction",
       "One-call smoke test: get_uv_sample_catalog -> resolve image path -> "
       "extract_uv_profiles_for_field, for a quick end-to-end check. Testing "
       "convenience, not a production driver.",
       "profiles = run_uv_extraction_testbed('COSMOS', cfg)"),
    _e("coadd_uv_profiles", "uv_profile", "uv_continuum", "Coadd & fit",
       "Coadd per-galaxy UV profiles (extract_uv_profiles_for_field's list) "
       "into one stacked radial profile, via a chosen combine method.",
       "stack = coadd_uv_profiles(profiles, method='biweight')"),
    _e("bootstrap_uv_coadd", "uv_profile", "uv_continuum", "Coadd & fit",
       "Bootstrap error band for one coadd_uv_profiles() method, resampling "
       "the galaxy list.",
       "boot = bootstrap_uv_coadd(profiles, method='biweight', nboot=1000)"),
    _e("fit_and_plot_uv_coadd", "uv_profile", "uv_continuum", "Coadd & fit",
       "Fit + plot one coadd_uv_profiles() method's stacked profile, via "
       "fitting.fit_psf_aware_uv_exp/_uv_sersic.",
       "result = fit_and_plot_uv_coadd(boot, r_edges_kpc)"),
    _e("stack_and_bootstrap_uv", "uv_profile", "uv_continuum", "Coadd & fit",
       "Coadd + bootstrap ONCE, decoupled from fitting -- run this a single "
       "time, then fit_and_plot_uv_coadd/fitting.* as many times as needed "
       "without re-stacking.",
       "boot = stack_and_bootstrap_uv(profiles, r_edges_kpc)"),
    _e("bootstrap_and_fit_uv", "uv_profile", "uv_continuum", "Coadd & fit",
       "One-call coadd -> bootstrap -> fit -> plot for the UV-continuum "
       "sample -- the end-to-end convenience wrapping the steps above.",
       "result = bootstrap_and_fit_uv(profiles, r_edges_kpc)"),
    _e("measure_stellar_psf", "uv_profile", "uv_continuum", "Stellar PSF",
       "Measure the empirical stellar PSF for a field: each star's radial "
       "profile, coadded -- the psf_empirical alternative to a fitted Moffat.",
       "psf_r, psf_vals = measure_stellar_psf('COSMOS', star_catalog)"),
    _e("fit_moffat_psf", "uv_profile", "uv_continuum", "Stellar PSF",
       "Fit an analytic Moffat (both FWHM and beta free) to a measured "
       "stellar PSF curve -- the alternative to keeping the empirical curve "
       "as-is (see [[uv-psf-empirical-decision]]).",
       "fwhm, beta = fit_moffat_psf(psf_r, psf_vals)"),
    _e("psf_empirical_entry", "uv_profile", "uv_continuum", "Stellar PSF",
       "Turn a measured stellar-PSF curve into a paste-ready config entry "
       "for the psf_empirical slot.",
       "entry = psf_empirical_entry(psf_r, psf_vals)"),
    _e("build_effective_psf_uv", "uv_profile", "uv_continuum", "Stellar PSF",
       "Build the sample's effective PSF curve (psf_r, psf_vals) in kpc, "
       "per-galaxy kpc-converted and combined -- what the UV PSF-aware fit "
       "actually convolves with.",
       "psf_r, psf_vals = build_effective_psf_uv(profiles)"),
    _e("apply_psf_to_results", "uv_profile", "uv_continuum", "Stellar PSF",
       "(Re)assign each galaxy's PSF metadata -- psf_fwhm_arcsec, "
       "psf_fwhm_kpc, etc. -- onto an existing extraction-results list.",
       "profiles = apply_psf_to_results(profiles, psf_r, psf_vals)"),
]


# ---------------------------------------------------------------------
# BACK-END names to ignore in the drift check (known, intentionally
# undocumented). Everything else public in a contributing module that is
# NOT in _ENTRIES will be reported by check_guide() as possibly-new.
# ---------------------------------------------------------------------
_BACKEND = {
    "pipeline": set(),
    "extract": {"load_vdfi_field", "combine_fibers", "combine_fibers_chunked",
                "extract_fibers"},
    "masking": {"flag_fibers_on_mask", "make_circular_structure",
                "get_pixscale_arcsec", "nan_safe_gaussian_smooth"},
    "io": {"write_galaxy_fits"},
    "stack": {"convert_avg_fiber_bin", "rest_grid", "stack_spec_values",
              "coadd_galaxies", "sample_virial_radius_kpc", "build_core_stacks",
              "concat_core_prods", "resolve_galaxy_index"},
    "measure": {"stack_galaxies", "measure_centroid", "bootstrap_measurements",
                "bootstrap_stack_error", "run_header"},   # measure_centroid doc'd? yes -> keep out of backend
    "core": {"to_rest_frame", "lya_window", "measure_integration",
             "measure_gaussian", "measure_matched_filter", "measure_line",
             "classify_lya", "lya_ew_rest", "measure_core_hierarchical",
             "bootstrap_core_flux", "run_core", "coerce_catalog_dtypes",
             "read_keep_catalog", "write_core_fits", "write_core_hier_fits",
             "measure_params_from_file", "remeasure_core_fits",
             "load_field_with_psf", "fiber_extraction_geometry",
                                       "select_core_by_index", "select_core_by_id",
             "select_core_cone", "core_results_table"},
    "analysis": {"plot_centroid_methods", "compare_core_stacks"},
    "plotting": {"get_ylim_window", "plot_stack_panels_two",
                 "plot_radial_overlay", "plot_null_band"},
    "validation": {"zswap_spectra", "windowed_flux_scan", "plot_windowed_scan",
                   "plot_error_calibration", "plot_injection_bias"},
    "multicat": set(),
    "selection": {"derive", "with_column", "resolve_core_sn_fields"},
    "speclsf": {"instrumental_fwhm", "lsf_line_template", "intrinsic_sigma"},
    "starpsf": {"make_position_run_table", "psf_profiles_for_lines",
                "plot_radial_profile", "plot_curve_of_growth",
                "plot_blank_sky_null", "plot_line_sn", "plot_line_sn_radial",
                "plot_line_profiles"},
    "virial": {"moster13_mstar_from_mhalo", "mhalo_from_mstar_moster13",
               "R200c_from_M200c", "compute_annuli_from_virial",
               "virial_to_angular_bins"},   # angular is doc'd; leave the rest ignored
    "smoothing": set(),
    # continuum_noise_metric is the internal metric; noise_from_stacks is the
    # documented front door onto it. The _bar_colors/_rank_column/_resolve_bin
    # helpers start with '_' and are auto-ignored by check_guide.
    "optimize": {"continuum_noise_metric"},
}
# measure_centroid and virial_to_angular_bins ARE documented; drop them from ignore
_BACKEND["measure"].discard("measure_centroid")
_BACKEND["virial"].discard("virial_to_angular_bins")


# =====================================================================
# RETRO-TERMINAL DISPLAY
# =====================================================================
_W = 92                      # inner text width
_ACCENT = "\033[92m"         # green phosphor
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RST = "\033[0m"


def _c(s, code, color):
    return f"{code}{s}{_RST}" if color else s


def _wrap(text, width, indent=""):
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _rule(title="", ch="─", color=True):
    if title:
        left = f"── {title} "
        return _c(left + ch * (_W + 2 - len(left)), _DIM, color)
    return _c(ch * (_W + 2), _DIM, color)


def _box_top(title, color):
    t = f"╔══ {title} " + "═" * (_W - len(title) - 2) + "╗"
    return _c(t, _ACCENT, color)


def _box_line(s, color):
    s = s[:_W]
    return _c("║ ", _ACCENT, color) + f"{s:<{_W}}" + _c(" ║", _ACCENT, color)


def _box_bot(color):
    return _c("╚" + "═" * (_W + 2) + "╝", _ACCENT, color)


def _main_menu(color=True):
    out = [_box_top("LYAHALO · FIELD GUIDE", color)]
    out.append(_box_line("", color))
    out.append(_box_line("  pick a section, e.g.  guide(\"measure\")", color))
    out.append(_box_line("", color))
    for i, (key, (blurb, _subs)) in enumerate(SECTIONS.items(), 1):
        n = sum(e["section"] == key for e in _ENTRIES)
        # room computed from the ACTUAL plain-text prefix length (not a
        # hardcoded guess) so the blurb is clipped exactly where the line
        # itself gets clipped -- no extra mid-word chars lost to a mismatch.
        prefix = f"  ► {i}  {key.upper():<11}({n:>2})  "
        room = max(0, _W - len(prefix))
        raw = prefix + blurb[:room]
        out.append(_c("║ ", _ACCENT, color) + f"{raw:<{_W}}"[:_W] + _c(" ║", _ACCENT, color))
    out.append(_box_line("", color))
    out.append(_box_line("  detail: guide(\"run_stack\")   search: guide(search=\"flux\")", color))
    out.append(_box_line("  figure: pipeline_map()        audit:  check_guide()", color))
    out.append(_box_bot(color))
    print("\n".join(out))


def _fmt_entry_compact(e, color):
    name = _c(e["name"], _BOLD + _ACCENT, color)
    print(f"  {name}")
    print(_wrap(e["desc"], _W, indent="      "))


def _fmt_entry_full(e, color):
    tag = f"{e['section'].upper()} / {e['sub']}"
    top = f"┌─ {e['name']} " + "─" * max(1, _W - len(e['name']) - len(tag) - 5) + f" {tag} ┐"
    print(_c(top, _ACCENT, color))
    print(_wrap("purpose:  " + e["desc"], _W, indent="│ "))
    print(_c("│ ", _ACCENT, color) + f"module:   utils_lya_halo.{e['module']}")
    print(_c("│ ", _ACCENT, color) + "example:")
    for ln in e["example"].split("\n"):
        print(_c("│ ", _ACCENT, color) + "    " + _c(ln, _ACCENT, color))
    print(_c("└" + "─" * (_W + 2) + "┘", _ACCENT, color))


def _show_section(section, sub=None, color=True):
    blurb, subs = SECTIONS[section]
    print(_box_top(section.upper(), color))
    print(_box_line("  " + blurb[:_W - 4], color))
    if sub is None:
        subline = "  subsections: " + " · ".join(subs)
        print(_box_line(subline[:_W - 2], color))
    print(_box_bot(color))
    ents = [e for e in _ENTRIES if e["section"] == section]
    for s in subs:
        if sub is not None and s.lower() != sub.lower():
            continue
        group = [e for e in ents if e["sub"] == s]
        if not group:
            continue
        print(_rule(s, color=color))
        for e in group:
            _fmt_entry_compact(e, color)
    print()
    print(_c(f"  detail: guide(\"<name>\")   "
             f"e.g. guide(\"{ents[0]['name']}\")", _DIM, color))


def guide(query=None, sub=None, *, search=None, color=True):
    """
    The LYAHALO field guide -- retro, argument-driven.

      guide()                 main menu (the six sections)
      guide("measure")        list a whole section
      guide("measure","centroid")   one subsection
      guide("run_stack")      full detail card for one function
      guide(search="flux")    grep names + purposes across everything
      color=False             plain output (no ANSI) for logs/plain terminals

    Returns None (prints). The registry is guide._ENTRIES if you want the raw data.
    """
    if search is not None:
        q = search.lower()
        hits = [e for e in _ENTRIES
                if q in e["name"].lower() or q in e["desc"].lower()
                or q in e["sub"].lower()]
        print(_box_top(f"SEARCH · '{search}'  ({len(hits)} hits)", color))
        print(_box_bot(color))
        for e in hits:
            tag = _c(f"[{e['section']}/{e['sub']}]", _DIM, color)
            print(f"  {_c(e['name'], _BOLD + _ACCENT, color)}  {tag}")
            print(_wrap(e["desc"], _W, indent="      "))
        if not hits:
            print("  (no matches -- try a shorter term)")
        return

    if query is None:
        _main_menu(color)
        return

    key = str(query).lower()
    if key in SECTIONS:
        _show_section(key, sub, color)
        return

    # is it a function name?
    match = next((e for e in _ENTRIES if e["name"].lower() == key), None)
    if match is not None:
        _fmt_entry_full(match, color)
        return

    # fuzzy: section prefix or name substring
    sec = next((s for s in SECTIONS if s.startswith(key)), None)
    if sec:
        _show_section(sec, sub, color)
        return
    near = [e["name"] for e in _ENTRIES if key in e["name"].lower()]
    print(f"  '{query}' is not a section or a listed function.")
    if near:
        print("  did you mean:  " + ", ".join(near[:8]))
    else:
        print("  try guide() for the menu, or guide(search=\"...\").")


# =====================================================================
# DRIFT CHECK  (keeps the guide honest; runs in test_framework)
# =====================================================================
def check_guide(verbose=True):
    """
    Audit the guide against the live code. Reports two kinds of drift:
      * MISSING  -- an entry whose (module, function) no longer exists
                    (renamed / deleted): the guide is stale, fix the entry.
      * NEW?     -- a public function in a contributing module that is neither
                    documented here nor in _BACKEND: possibly a new front-end
                    function that should get an entry (or be added to _BACKEND).

    Returns {'missing': [...], 'new': [...]}. test_framework asserts 'missing'
    is empty; 'new' is advisory.
    """
    missing, new = [], []
    documented = {}
    for e in _ENTRIES:
        documented.setdefault(e["module"], set()).add(e["name"])

    # 1. every entry must resolve
    for e in _ENTRIES:
        try:
            mod = importlib.import_module(f"utils_lya_halo.{e['module']}")
        except Exception as exc:                       # noqa: BLE001
            missing.append(f"{e['module']}.{e['name']}  (module import failed: {exc})")
            continue
        if not hasattr(mod, e["name"]):
            missing.append(f"{e['module']}.{e['name']}  (not found)")

    # 2. undocumented public functions in contributing modules
    import inspect
    for module in sorted(set(e["module"] for e in _ENTRIES)):
        try:
            mod = importlib.import_module(f"utils_lya_halo.{module}")
        except Exception:                              # noqa: BLE001
            continue
        pub = {n for n, o in vars(mod).items()
               if inspect.isfunction(o) and not n.startswith("_")
               and getattr(o, "__module__", "").endswith(module)}
        known = documented.get(module, set()) | _BACKEND.get(module, set())
        for n in sorted(pub - known):
            new.append(f"{module}.{n}")

    if verbose:
        color = True
        print(_box_top("GUIDE AUDIT", color))
        print(_box_bot(color))
        if not missing and not new:
            print("  [OK] guide is in sync with the code.")
        if missing:
            print(f"  [X] {len(missing)} STALE ent(y/ies) -- function gone/renamed:")
            for m in missing:
                print("      - " + m)
        if new:
            print(f"  • {len(new)} public function(s) not in the guide "
                  f"(add an entry or list in _BACKEND):")
            for m in new:
                print("      - " + m)
    return {"missing": missing, "new": new}


# =====================================================================
# PIPELINE MAP  (figure: extraction -> measure -> plotting -> validation)
# =====================================================================
def pipeline_map(figsize=(12, 5.2), save_fig=False, savename="pipeline_map.png"):
    """
    Draw the pipeline as a flow: the four spine stages as boxes, the on-disk /
    in-memory data products on the arrows between them, and the headline
    function(s) inside each box. A visual companion to guide().

    Returns the matplotlib Figure.

        from utils_lya_halo.guide import pipeline_map
        pipeline_map()
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    stages = [
        ("EXTRACTION", "#1b4965", [
            "run_extract  (Stage 1)", "run_stack  (Stage 2)",
            "build_bad_fiber_mask", "segmentation_mask"]),
        ("MEASURE", "#5fa8d3", [
            "run_measure", "run_measurement_suite",
            "measure_centroid", "bootstrap_measurements"]),
        ("PLOTTING", "#62b6cb", [
            "plot_centroid_profile", "plot_flux_profile",
            "plot_line_panels", "run_all_plots"]),
        ("VALIDATION", "#cae9ff", [
            "redshift_scramble_null",             "signal_to_noise_spectrum", "centroid_dossier"]),
    ]
    products = ["galaxy FITS", "rest-frame stacks", "centroids + errors", "figures"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10 * len(stages))
    ax.set_ylim(0, 10)
    ax.axis("off")

    bw, bh, y0 = 18, 6.2, 2.0
    centers = []
    for i, (name, col, funcs) in enumerate(stages):
        x0 = i * 25 + 2
        cx = x0 + bw / 2
        centers.append((x0, cx))
        box = FancyBboxPatch((x0, y0), bw, bh,
                             boxstyle="round,pad=0.3,rounding_size=0.8",
                             linewidth=1.6, edgecolor="#0b2b3a",
                             facecolor=col, alpha=0.92)
        ax.add_patch(box)
        dark = i < 2
        tc = "white" if dark else "#0b2b3a"
        ax.text(cx, y0 + bh - 0.9, name, ha="center", va="center",
                fontsize=13, fontweight="bold", color=tc)
        for j, fn in enumerate(funcs):
            ax.text(cx, y0 + bh - 1.9 - j * 0.95, fn, ha="center", va="center",
                    fontsize=8.3, family="monospace", color=tc)

    # arrows + product labels between consecutive stages
    for i in range(len(stages)):
        x0, cx = centers[i]
        if i < len(stages) - 1:
            nx0, _ = centers[i + 1]
            a = FancyArrowPatch((x0 + bw, y0 + bh / 2), (nx0, y0 + bh / 2),
                                arrowstyle="-|>", mutation_scale=18,
                                linewidth=1.8, color="#0b2b3a")
            ax.add_patch(a)
            midx = (x0 + bw + nx0) / 2
            ax.text(midx, y0 + bh / 2 + 0.55, products[i], ha="center",
                    va="bottom", fontsize=8, style="italic", color="#0b2b3a")
        else:
            ax.text(x0 + bw / 2, y0 - 0.7, "→ " + products[i], ha="center",
                    va="top", fontsize=8, style="italic", color="#0b2b3a")

    ax.text(0.5 * 25 * len(stages) - 10, 9.3,
            "LYAHALO pipeline   ·   guide(\"<stage>\") for the full function list",
            ha="center", va="center", fontsize=10.5, color="#0b2b3a")

    # Sample + Misc note (off-spine sections)
    ax.text(2, 0.6, "SAMPLE  (slice/split/match, between EXTRACT & MEASURE)   "
                    "·   MISC  (virial · LSF · star-PSF · core)",
            ha="left", va="center", fontsize=8.2, color="#456",
            family="monospace")

    fig.tight_layout()
    if save_fig:
        fig.savefig(savename, dpi=150, bbox_inches="tight")
    return fig
