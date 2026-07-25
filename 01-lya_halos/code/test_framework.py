"""
test_framework.py -- smoke test for the pipeline FRAME (not the science).

Run from the directory that contains the `utils_lya_halo/` package:

    python test_framework.py

It exercises the FLOW end to end on tiny synthetic arrays -- it deliberately
does NOT run a real extraction or a heavy bootstrap. Everything is sized so the
whole thing finishes in well under a second:

  1. PipelineConfig builds, provenance keywords are sane, and the paths derive
     correctly (cache nested under outputs, catalog-tagged FITS name).
  2. write_galaxy_fits -> read_galaxy_fits round-trips on fake arrays, with the
     galaxy axis staying aligned to the CATALOG table.
  3. Stage 2 (run_stack) -> Stage 3 (run_measure) flow: a fake galaxy FITS is
     stacked and measured with n_bootstrap tiny, checking only that the stages
     chain together and return the expected keys/shapes.

Requires numpy + astropy in the active environment.
"""

import os
import tempfile

import numpy as np
from dataclasses import replace
from astropy.table import Table

from utils_lya_halo import (
    PipelineConfig, write_galaxy_fits, read_galaxy_fits,
    run_stack, run_measure,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _fake_catalog(ngal, rng):
    """A catalog whose row order defines the galaxy axis."""
    return Table({
        "RA":  rng.uniform(214, 215, ngal),
        "DEC": rng.uniform(52, 53, ngal),
        "z":   rng.uniform(2.0, 2.7, ngal),
        "MASS_50": rng.uniform(9.0, 11.0, ngal),
    })


def _fake_stage1_arrays(cfg, ngal, nwave, rng):
    """
    Fake a Stage-1 product. Each spectrum carries a weak Gaussian Lya line at
    that galaxy's OBSERVED Lya wavelength (1215.67*(1+z)), so after Stage 2's
    rest-frame resample the stack has a real line at 1215.67 and the centroid
    measurement has something to lock onto.
    """
    nrad = len(cfg.bins) - 1
    wave = np.linspace(cfg.wave_min, cfg.wave_max, nwave)
    cat = _fake_catalog(ngal, rng)
    z = np.asarray(cat["z"])

    spec = np.full((ngal, nrad, nwave), np.nan, dtype="f4")
    err = np.full((ngal, nrad, nwave), np.nan, dtype="f4")
    for g in range(ngal):
        lam0 = PipelineConfig.LYA_REST * (1 + z[g])      # observed Lya
        for r in range(nrad):
            amp = 50.0 * np.exp(-r)                        # fades with radius
            line = amp * np.exp(-0.5 * ((wave - lam0) / 4.0) ** 2)
            spec[g, r] = line + rng.normal(0, 5.0, nwave)
            err[g, r] = np.abs(rng.normal(5.0, 1.0, nwave))
    nfib = rng.integers(1, 5000, size=(ngal, nrad)).astype("i4")
    background = rng.normal(size=(ngal, nwave)).astype("f4")
    return wave, spec, err, nfib, background, cat


# ---------------------------------------------------------------------
# 1. config + provenance + derived paths
# ---------------------------------------------------------------------
def test_config_and_paths():
    cfg = PipelineConfig(field="EGS", catalog="MOSDEF", num_gal=5,
                         catalog_path="catalogs/example.txt")
    assert cfg.normalized_field() == "AEGIS"             # alias collapses
    hdr = cfg.to_header_dict(ngal=5, source_id="all_flux_final.fits")
    assert hdr["FIELD"] == "AEGIS" and hdr["BINMODE"] == "virial"
    # assert against the config, not a literal: this previously hardcoded 25 and
    # silently went stale when the default moved to 999999
    assert hdr["MINBGFIB"] == cfg.min_bg_fibers

    # cache nests under outputs by default and follows output_dir
    assert cfg.resolved_cache_dir() == os.path.join(cfg.output_dir, "cache")
    moved = replace(cfg, output_dir="/tmp/run_x")
    assert moved.resolved_cache_dir() == os.path.join("/tmp/run_x", "cache")
    assert replace(cfg, cache_dir="/scratch/c").resolved_cache_dir() == "/scratch/c"
    # per-run cache is one directory PER SIGNATURE beneath it
    assert cfg.resolved_run_cache_dir() == os.path.join(
        cfg.resolved_cache_dir(), cfg.extract_signature())

    # Current FITS name = field + catalog + bin count + extraction signature.
    # We check STRUCTURE rather than a hardcoded literal so the test does not
    # break every time the naming scheme grows a new token; it asserts the
    # invariants the name must satisfy, not its exact spelling.
    name = cfg.descriptive_filename()
    assert name.startswith("galaxy_") and name.endswith(".fits")
    for token in (cfg.normalized_field(), cfg.catalog,
                  f"b{len(cfg.bins) - 1}", cfg.extract_signature()):
        assert token in name, f"{token!r} missing from descriptive_filename {name!r}"

    # THE INVARIANT THIS SCHEME EXISTS FOR: any knob that busts the per-galaxy
    # cache must also change the FITS name, so a re-extraction can never
    # overwrite the previous one. The old hand-listed name encoded 8 knobs while
    # the cache hashed ~30; these are drawn from the ~22 that used to slip
    # through silently.
    for knob in ({"bg_smooth_sigma_A": 45.0}, {"min_bg_fibers": 12},
                 {"nwave": 512}, {"z_col": "z_spec"},
                 {"mask_percentile_image": 77.0}, {"line_mask_halfwidth_A": 7.5}):
        v = replace(cfg, **knob)
        assert v.extract_signature() != cfg.extract_signature(), knob
        assert v.descriptive_filename() != name, f"{knob} busts cache but NOT the FITS name"

    # ...while knobs that provably cannot change the numbers leave both alone,
    # so a cache built at 1 thread stays valid at 8 (see extraction-speedup 9.-1)
    for knob in ({"combine_threads": 8}, {"combine_min_fibers": 10_000}):
        try:
            v = replace(cfg, **knob)
        except TypeError:
            continue                                   # field not on this config
        assert v.extract_signature() == cfg.extract_signature(), knob
        assert v.descriptive_filename() == name, knob

    other = replace(cfg, catalog_path="catalogs/example_zcut.txt")
    assert other.descriptive_filename() != name          # different catalog -> different file

    # older spellings still constructible, which is what lets io fall back to
    # already-extracted files instead of forcing a re-run
    tagged = cfg.descriptive_filename(scheme="tagged", catalog_tag=True)
    legacy = cfg.descriptive_filename(scheme="tagged", catalog_tag=False, bins_tag=False)
    tag = cfg.catalog_tag()
    assert tag in tagged and tag not in legacy and tagged != legacy != name
    for token in (cfg.bin_mode, cfg.fiber_combine_method, cfg.mask_method):
        assert token in legacy, f"{token!r} missing from legacy name {legacy!r}"
    print("[1] config, provenance, derived paths OK:",
          {k: hdr[k] for k in ("FIELD", "BINMODE", "COMBINE")}, "|", name)


# ---------------------------------------------------------------------
# 2. galaxy FITS round-trip + axis alignment
# ---------------------------------------------------------------------
def test_fits_roundtrip(tmpdir):
    cfg = PipelineConfig(field="AEGIS", catalog="MOSDEF", num_gal=5,
                         output_dir=tmpdir, nwave=64)
    ngal, nrad, nwave = 5, len(cfg.bins) - 1, cfg.nwave
    rng = np.random.default_rng(0)
    wave, spec, err, nfib, background, cat = _fake_stage1_arrays(cfg, ngal, nwave, rng)
    r_edges = np.array(cfg.bins, dtype="f4")

    path = os.path.join(tmpdir, "_smoke_galaxy.fits")
    write_galaxy_fits(path, spec=spec, err=err, nfib=nfib, wave=wave,
                      r_edges=r_edges, catalog=cat, config=cfg,
                      source_id="all_flux_final.fits", background=background,
                      overwrite=True)

    prod = read_galaxy_fits(path)
    assert prod.ngal == ngal and prod.nrad == nrad and prod.nwave == nwave
    assert prod.background is not None and prod.background.shape == (ngal, nwave)
    assert prod.header["FIELD"] == "AEGIS"
    assert len(prod.catalog) == prod.ngal                # axis <-> catalog
    assert np.allclose(prod.catalog["z"], cat["z"])      # row order preserved
    print("[2] FITS round-trip + alignment OK:",
          f"shape={prod.spec.shape}, ngal==len(catalog)=={prod.ngal}")
    return path


# ---------------------------------------------------------------------
# 3. Stage 2 -> Stage 3 flow (tiny; checks wiring, not science)
# ---------------------------------------------------------------------
def test_stack_measure_flow(tmpdir):
    # tiny + fast: few galaxies, coarse rest grid, raw units (no cosmology),
    # tiny bootstrap. We only assert the stages chain and return shapes/keys.
    base = PipelineConfig(field="AEGIS", catalog="MOSDEF", output_dir=tmpdir,
                          nwave=64, rest_delta=1.0, flux_unit="raw",
                          min_good_wave=5, n_bootstrap=5,
                          catalog_path="catalogs/example.txt")
    ngal = 6
    rng = np.random.default_rng(1)
    wave, spec, err, nfib, background, cat = _fake_stage1_arrays(base, ngal, base.nwave, rng)
    path = os.path.join(tmpdir, base.descriptive_filename())
    write_galaxy_fits(path, spec=spec, err=err, nfib=nfib, wave=wave,
                      r_edges=np.array(base.bins, "f4"), catalog=cat, config=base,
                      source_id="fake", background=background, overwrite=True)

    nrad = len(base.bins) - 1

    # Stage 2
    stacks = run_stack(base, path, keep_cube=True, verbose=False)
    for k in ("rest_wave", "r_edges", "stacks", "cube_flux", "cube_err",
              "unit_info", "VR_biweight_v"):
        assert k in stacks, f"stacks missing {k}"
    nwave_rest = stacks["rest_wave"].size
    assert stacks["cube_flux"].shape[1:] == (nrad, nwave_rest)
    for method in base.galaxy_combine_methods:
        assert stacks["stacks"][method]["flux"].shape == (nrad, nwave_rest)
    print(f"[3a] Stage 2 OK: {len(base.galaxy_combine_methods)} methods, "
          f"cube {stacks['cube_flux'].shape}, VR_biweight_v={stacks['VR_biweight_v']}")

    # Stage 3 (the newly wired run_measure)
    results = run_measure(base, stacks)
    for k in ("centroid_v_med", "centroid_v_lo", "centroid_v_hi", "centroid_v_fid",
              "success_frac", "blue_over_red_med", "blue_flux_med", "red_flux_med",
              "stack_error", "rest_wave", "r_edges", "VR_biweight_v"):
        assert k in results, f"results missing {k}"
    assert np.shape(results["centroid_v_med"]) == (nrad,)
    assert results["stack_error"]["stack_err"].shape == (nrad, nwave_rest)
    print(f"[3b] Stage 3 (run_measure) OK: centroid_v_med shape "
          f"{np.shape(results['centroid_v_med'])}, "
          f"per-pixel err {results['stack_error']['stack_err'].shape}")


def test_galaxy_slicing_split(tmpdir):
    """
    THE Slurm-array invariant: splitting one extraction across array tasks must
    produce exactly the run a single process would have produced.

    Three things can break and all three are silent:
      1. cache keys named by loop index instead of global row -> tasks overwrite
         each other and you get N/ntasks galaxies with no error anywhere
      2. gal_offset leaking into the extraction signature -> each task builds
         its own cache/<sig>/ and the slices never meet
      3. the assemble pass reading the cache back in the wrong order

    So: run serially, wipe the cache, run the SAME work as 3 offset tasks, and
    assert the reassembled arrays are bit-identical to the serial ones.
    """
    import shutil
    from utils_lya_halo import extract

    rng = np.random.default_rng(7)
    ngal, nexp, nfib, nwave = 10, 2, 900, 24
    vdfi = dict(
        RA=150.0 + rng.normal(0, 0.02, (nexp, nfib)),
        DEC=2.0 + rng.normal(0, 0.02, (nexp, nfib)),
        spectra=rng.normal(0, 1, (nexp, nfib, nwave)).astype("f4"),
        error=np.abs(rng.normal(1, 0.1, (nexp, nfib, nwave))).astype("f4"),
        wave=np.linspace(3470, 5540, nwave),
    )
    tbl = Table({"RA": 150.0 + rng.normal(0, 0.01, ngal),
                 "DEC": 2.0 + rng.normal(0, 0.01, ngal),
                 "z": rng.uniform(2.0, 2.6, ngal),
                 "MASS_50": rng.uniform(9.5, 10.5, ngal),
                 "ID": np.arange(ngal)})
    bad = np.zeros((nexp, nfib), bool)
    cfg = PipelineConfig(field="AEGIS", catalog="MOSDEF", output_dir=tmpdir,
                         bins=[0, 50, 200, 1000], bin_mode="kpc",
                         mask_method="none", nwave=nwave, wave_min=3470,
                         wave_max=5540, smooth_bg=False, min_bg_fibers=9999999)
    go = lambda c: extract.extract_fibers(c, vdfi, tbl, bad, use_cache=True)

    serial = go(cfg)
    cdir = cfg.resolved_run_cache_dir()
    expected = sorted(f for f in os.listdir(cdir) if f.endswith(".npz"))
    assert len(expected) == ngal

    shutil.rmtree(cdir)                       # cold cache: the tasks must really compute
    for offset, count in [(0, 4), (4, 4), (8, 2)]:
        c = replace(cfg, gal_offset=offset, num_gal=count)
        # every task must resolve to the SAME cache dir, or the split is pointless
        assert c.resolved_run_cache_dir() == cdir
        assert c.descriptive_filename() == cfg.descriptive_filename()
        assert go(c)[0].shape[0] == count

    got = sorted(f for f in os.listdir(cdir) if f.endswith(".npz"))
    assert got == expected, f"cache filename collision: {len(got)} files for {ngal} galaxies"

    for name, a, b in zip(("spec", "err", "nfib", "bg", "vr"), serial, go(cfg)):
        assert np.array_equal(a, b, equal_nan=True), f"{name} differs after split+assemble"
    print(f"[4] galaxy slicing OK: 3-task split -> assemble == serial, "
          f"bit-identical, {len(got)} cache files")


def test_guide_in_sync():
    """The field guide must not drift: every entry resolves to a live function."""
    from utils_lya_halo import check_guide
    report = check_guide(verbose=False)
    assert not report["missing"], (
        "guide.py is stale -- these documented functions no longer exist:\n  "
        + "\n  ".join(report["missing"]))
    if report["new"]:
        print(f"[guide] NOTE: {len(report['new'])} public function(s) not yet "
              f"documented in guide.py: {', '.join(report['new'][:8])}"
              + (" ..." if len(report["new"]) > 8 else ""))
    print(f"[guide] in sync: {report and 'OK'} "
          f"(0 stale, {len(report['new'])} undocumented)")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_config_and_paths()
        test_fits_roundtrip(tmp)
        test_stack_measure_flow(tmp)
        test_galaxy_slicing_split(tmp)
        test_guide_in_sync()
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
