# VDFI To-Do

Working list across the workspace. Check items off as they land; log the actual change in `CHANGELOG.md` when they do. Newest items go at the bottom of their section unless priority says otherwise.

## Pipeline

- [ ] Revisit the virial-radius conversions — `utils_lya_halo/virial.py` (`estimate_M200c_R200c_from_Mstar`, `virial_to_kpc_bins`/`virial_to_angular_bins`). Confirm the Moster+2013 SMHM mapping is still the right call.
- [ ] Ensure the PSF-aware model of 2-component flux curve fitting is working properly. Finalize this inner exponential scale number and add to pipeline.
- [ ] Finalize the UV-continuum exponential scale and derive halo scale / UV scale ratio.

## Paper

- [ ] Begin writing Introduction, Data, Methods, and Results sections
- [ ] Improve and generate proper paper figures — some already exist, but this needs continual work: nail down what each figure should show and how, and bring them up to publication quality.

## Science

- [ ] Think through subsample splits and their physical predictions in the literature — candidates: `split_by_derived`/`split_controlled` (selection.py), `split_by_redshift`/`split_product_by` (multicat.py)
- [ ] Read more!
- [ ] Check whether core measurement templates (single vs. double Gaussian) and their related separation/flux-ratio stacks (blue- vs. redshifted) are being fit to noise that resembles the template by construction — test whether fitting these templates to pure noise/null spectra converges toward a template-shaped stack at high N, and if so how much of any observed double-peaked structure is genuine vs. fitting-induced. See `01-lya_halos/docs/research-notes.md`.

## Add-ins (need review)
- [ ] Look into Lya EW and SFR and its predicted maximum to constrain possible emission sources out to certain radii (paper 2)
- [ ] Connect emission produced by SF to that which could be produced by gravitational cooling