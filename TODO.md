# VDFI To-Do

Working list across the workspace. Check items off as they land; log the actual change in `CHANGELOG.md` when they do. Newest items go at the bottom of their section unless priority says otherwise.

## Pipeline

- [ ] Revisit the virial-radius conversions — `utils_lya_halo/virial.py` (`estimate_M200c_R200c_from_Mstar`, `virial_to_kpc_bins`/`virial_to_angular_bins`). Confirm the Moster+2013 SMHM mapping is still the right call.
- [ ] Continue running parameter adjustments and optimize the background subtraction — `utils_lya_halo/smoothing.py`. Already in decent shape (default `smooth_interp_then_gauss`; legacy `smooth_spectrum_nan_safe_adaptive` kept for comparison); `run_background_sweep` (`optimize.py`) can score variants directly.

## Paper

- [ ] Begin writing Introduction, Data, Methods, and Results sections
- [ ] Improve and generate proper paper figures — some already exist, but this needs continual work: nail down what each figure should show and how, and bring them up to publication quality.

## Science

- [ ] Think through subsample splits and their physical predictions in the literature — candidates: `split_by_derived`/`split_controlled` (selection.py), `split_by_redshift`/`split_product_by` (multicat.py)
- [ ] Read more!
