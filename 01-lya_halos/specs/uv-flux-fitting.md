# UV-continuum flux fitting — PSF creation + fit workflow

Companion to `halo-flux-fitting.md` Part 3. Documents how the UV-continuum
radial profile (CFHT-LS r-band, `uv_profile.py`) is measured, PSF-corrected,
and fit — the settled workflow to reuse from here on.

**Status: implemented in `utils_lya_halo/uv_profile.py` (+ the PSF forward
model in `fitting.py`).**

---

## PSF: measure once per field, then reuse everywhere

The survey seeing is a fixed *angular* FWHM per field, so the PSF is measured
from stars once and frozen into config. Recipe:

1. **Extract stars.** GAIA point sources in the field (`parallax_over_error > 5`,
   `phot_g_mean_mag` ~18–21 to avoid saturated cores).
2. **Measure the stellar profile** with the same machinery as galaxies but in
   **arcsec** (stars have no z) and with **source masking OFF**
   (`neighbor_mask=None`): per-star cutout → `centroid_galaxy` → `measure_uv_annuli`
   on fine arcsec bins → peak-normalize each → median-stack → empirical PSF.
   (A star is a point source, so its profile *is* the PSF.)
3. **Fit a Moffat, FWHM and β both free:** `fit_moffat_psf(r_mid, stack,
   sigma=star_scatter)`. It fits in **log space**, so the wings (not the near-unity
   core) pin β; bins with zero/non-finite sigma (e.g. the peak bin of a
   peak-normalized stack) are dropped automatically. Returns FWHM, β with 1σ errors
   — the **citable PSF numbers**.
4. **Save** into config and **run for BOTH fields**:
   ```python
   uvcfg.psf_moffat_params[field] = {"fwhm_arcsec": fitres["fwhm"], "beta": fitres["beta"]}
   ```

Header seeing (auto-read, `IQIMAGE`/`FINALIQ`) is the fallback only; the
star-fit overrides it. As measured: COSMOS/D2 ≈ 0.73″, β ≈ 2.7 (header
`IQIMAGE`=0.751″); AEGIS/D3 ≈ 0.785″, β ≈ 3 (header `FINALIQ`=0.83″) — β=3 is
too light in the wings, ~2.7 is better.

## Per-galaxy kpc PSF → effective PSF

The fixed angular FWHM is a *different* kpc FWHM per galaxy (via its z), and the
profile is stacked in fixed kpc bins. So each galaxy stores `psf_fwhm_kpc`
(= arcsec FWHM / arcsec-per-kpc at its z) and `psf_beta` from its field.
`build_effective_psf_uv(results, uvcfg)` averages the per-galaxy flux-normalized
Moffats into one effective PSF curve `(psf_r, psf_vals)` in kpc — correctly
blending the two fields' different seeing and wing shapes.

## The fit — run from here on

```python
# one-time after any new extraction (adds the per-galaxy field/psf tags):
res_cosmos = uv.run_uv_extraction_testbed(config, uvcfg, field="COSMOS", cutout_image_path=...)
res_aegis  = uv.run_uv_extraction_testbed(config, uvcfg, field="AEGIS",  cutout_image_path=...)
results_all = res_cosmos + res_aegis

# after both fields are in uvcfg.psf_moffat_params (star fits above):
uv.apply_psf_to_results(results_all, uvcfg)                       # push per-field (FWHM,β), no re-extract
psf_r, psf_vals = uv.build_effective_psf_uv(results_all, uvcfg)

coadd, boot = uv.stack_and_bootstrap_uv(results_all, uvcfg.bins_kpc,
                                         method="biweight", nboot=5000)   # EXPENSIVE — run ONCE
uv.fit_and_plot_uv_coadd(coadd, boot=boot, fit_method="naive")                          # fit freely...
uv.fit_and_plot_uv_coadd(coadd, boot=boot, fit_method="psf", psf_r=psf_r, psf_vals=psf_vals)
```

- **Stack/bootstrap is split from fitting** (`stack_and_bootstrap_uv`): the
  bootstrap is a property of the data, so run it once and fit as many models as
  you like off the same `(coadd, boot)`.
- **Tuning β later needs no re-extraction:** change `psf_moffat_params` →
  `apply_psf_to_results` → `build_effective_psf_uv` → refit.

## Correctness note (fixed bug)

The PSF forward model (`bin_average_psf_uv_exp`/`_sersic` in `fitting.py`) was
returning `2πr·SB` instead of surface brightness. Fixed: input ring flux now
carries the `2πr` circumference and the per-bin result divides by annulus area
`π(r_out²−r_in²)`, not bin width. Verified against a brute-force 2-D image
convolution to <0.1%. The naive (no-PSF) path was already correct and is
unchanged. `psf_fwhm` is in **kpc**, never arcsec.

## Interpretation caveat

The PSF (~6 kpc FWHM) is much larger than the intrinsic scale (`h_uv` ~1.5 kpc):
the galaxies are barely resolved, so `h_uv` is fragile and PSF-sensitive — check
its formal error and how much it moves with β. Any outer-bin excess above the
model (data flattening at large r) is either residual sky subtraction or PSF
wings; the heavier measured wings (β≈2.7 vs 3) are the clean test — if the
excess shrinks with the real PSF it was wing scatter, if it persists it's sky
(or a genuine extended component).
