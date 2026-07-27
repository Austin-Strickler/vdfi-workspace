# VDFI To-Do

Working list across the workspace. Check items off as they land; log the actual change in `CHANGELOG.md` when they do. Newest items go at the bottom of their section unless priority says otherwise.

## Pipeline

- [ ] Revisit the virial-radius conversions — `utils_lya_halo/virial.py` (`estimate_M200c_R200c_from_Mstar`, `virial_to_kpc_bins`/`virial_to_angular_bins`). Confirm the Moster+2013 SMHM mapping is still the right call. *(Still open — being put off. The `mstar_to_rvir.py` Behroozi+2019 prototype at the workspace root is the candidate replacement; just needs a decision on whether it supersedes Moster and where it should live. Worth a focused hour when time frees up — it's the last unclosed pipeline choice.)*
- [ ] Work on systematics and solidify/defend the pipeline choices and techniques
- [ ] **Codebase organization pass.** `utils_lya_halo/` has grown to ~34k lines across 28 modules and functions are no longer reliably where you'd look for them. Two specific known cases: (a) **background computations are split** — `background.py` holds the new Stage-2 offline testing, but the background *smoothers* live in `smoothing.py`, the annulus fiber selection and combine in `extract.py`, and `inspect_core_background` in `core.py`; consolidating the background logic into `background.py` is the obvious first target. (b) File/directory layout across `specs/`, `docs/`, and the notebook snapshots needs the same treatment. Do this as a deliberate pass, not incidentally — every move must keep `guide.py` registration and `check_guide()` clean, and `precedent`: the `environment.py` → `multicat.py` merge (2026-07-21) is the pattern that worked.
- [ ] **Decide the background architecture** once `specs/background-diagnostics.md` §2.4 lands. Single global annulus (today), a larger/further-out single annulus, or a radius-dependent background (two annuli with a switch, or a smooth function). Deferred deliberately until the 4-variant run is measured — see that spec's §0.3 for the two constraints any variable scheme has to satisfy (extraction cost, and self-subtraction against the science bins).

## Paper

- [ ] Edit the Introduction, Data, Methods, and Results sections
- [ ] Improve and generate proper paper figures — some already exist, but this needs continual work: nail down what each figure should show and how, and bring them up to publication quality.

## Science

- [ ] Read more!

## Revisit Later

- [ ] Run MCMC on flux fit to understand possible degeneracies between gamma and power-law scale radius
- [ ] Look into Lya EW and SFR and its predicted maximum to constrain possible emission sources out to certain radii (paper 3). Connect emission produced by SF to that which could be produced by gravitational cooling.
- [ ] **UV-continuum exponential scale + halo/UV ratio — potential improvement.** CFHT-LS PSF is the hard part; if the UV PSF correction ever proves untrustworthy, the fallback is to redo the UV profile on **3D-HST imaging**.
- [ ] Check whether core measurement templates (single vs. double Gaussian) and their related separation/flux-ratio stacks (blue- vs. redshifted) are being fit to noise that resembles the template by construction — test whether fitting these templates to pure noise/null spectra converges toward a template-shaped stack at high N, and if so how much of any observed double-peaked structure is genuine vs. fitting-induced. See `01-lya_halos/docs/research-notes.md`.
