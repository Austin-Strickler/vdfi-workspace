# VDFI To-Do

Working list across the workspace. Check items off as they land; log the actual change in `CHANGELOG.md` when they do. Newest items go at the bottom of their section unless priority says otherwise.

## Pipeline

- [ ] Revisit the virial-radius conversions — `utils_lya_halo/virial.py` (`estimate_M200c_R200c_from_Mstar`, `virial_to_kpc_bins`/`virial_to_angular_bins`). Confirm the Moster+2013 SMHM mapping is still the right call. *(Still open — being put off. The `mstar_to_rvir.py` Behroozi+2019 prototype at the workspace root is the candidate replacement; just needs a decision on whether it supersedes Moster and where it should live. Worth a focused hour when time frees up — it's the last unclosed pipeline choice.)*
- [x] **PSF-aware 2-component flux fitting — done.** `fit_psf_aware` (`fitting.py`) validated on real data (CFHT-LS): χ²/dof = 1.17, inner scale **h1 ≈ 16–17 kpc**, outer h2 ≈ 1552 kpc. Inner exponential scale is finalized and in the pipeline.
- [x] **UV-continuum exponential scale + halo/UV ratio — done.** UV-continuum fit runs well on CFHT-LS: **UV scale length ≈ 1.55 kpc** vs. Lyα ≈ 16–17 kpc → **halo/UV scale ratio ≈ 10:1** (in line with LaCOS II's ~10× and above Kikuta 2023's ~6–7×). Caveat: CFHT-LS PSF is the hard part; if the UV PSF correction ever proves untrustworthy, the fallback is to redo the UV profile on **3D-HST imaging** instead — not needed for now.

## Paper

- [ ] Begin writing Introduction, Data, Methods, and Results sections
- [ ] Improve and generate proper paper figures — some already exist, but this needs continual work: nail down what each figure should show and how, and bring them up to publication quality.

## Science

- [x] **Subsample splits — primary tests done.** The core splits are run and the paper-1 vs paper-2 assignment is essentially settled (`split_by_derived`/`split_controlled` in selection.py, `split_by_redshift`/`split_product_by` in multicat.py; see `docs/subsample_splits.md`). Final list stays lightly in flux by nature, but this is no longer blocking work.
- [ ] Read more!
- [ ] Check whether core measurement templates (single vs. double Gaussian) and their related separation/flux-ratio stacks (blue- vs. redshifted) are being fit to noise that resembles the template by construction — test whether fitting these templates to pure noise/null spectra converges toward a template-shaped stack at high N, and if so how much of any observed double-peaked structure is genuine vs. fitting-induced. See `01-lya_halos/docs/research-notes.md`.

## Add-ins (need review)
- [ ] Look into Lya EW and SFR and its predicted maximum to constrain possible emission sources out to certain radii (paper 2)
- [ ] Connect emission produced by SF to that which could be produced by gravitational cooling