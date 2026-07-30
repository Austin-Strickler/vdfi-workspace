# VDFI / HETDEX Background

Context for anyone (including future us) picking up this workspace cold. This is "why the data looks like this," not project-specific science — see each subproject's `docs/GOALS.md` for that.

## HETDEX

HETDEX (Hobby-Eberly Telescope Dark Energy Experiment) is a blind spectroscopic survey run on the Hobby-Eberly Telescope using VIRUS, a massively-replicated integral field spectrograph: 78 IFUs, 448 fibers each (~35,000 fibers total), covering 3500-5500 Å at R~800 (spectral resolution ~5.6 Å). The headline goal is mapping Lyα emitters across a wide area at cosmic noon (roughly z=1.9-3.5) to measure baryon acoustic oscillations, alongside a broad range of other galaxy-evolution science the same dataset enables.

## VDFI

VDFI is a sub-project built on the same VIRUS instrument, but pointed differently: instead of HETDEX's wide, shallow single-visit strategy, VDFI stacks many exposures over two specific fields — COSMOS (~0.25 deg²) and EGS (~0.1 deg²) — to reach a full-field depth of roughly 6 hours. The result is dense, contiguous IFU coverage: fiber spectra essentially every arcsecond across both fields, rather than sparse pointings.

Individually, each fiber spectrum is unremarkable (medium resolution, R~800), but the combination of near-complete spatial coverage, real depth, and careful systematics control makes the dataset unusually powerful for stacking-based science. Redshift coverage in both fields is the same cosmic-noon range HETDEX targets, roughly z=1.8/1.9-3.5.

This shared dataset feeds multiple VDFI subprojects:

- [`01-lya_halos`](../01-lya_halos/) — Lyα halo stacking (this doc's immediate context), and the
  umbrella for the whole science program: the virial-radius transition, the subsample splits, and
  the escape-fraction / geometry–throughput work. See
  [`01-lya_halos/docs/GOALS.md`](../01-lya_halos/docs/GOALS.md).
- [`02-lae_catalog`](../02-lae_catalog/) — LAE/AGN catalog construction, not yet started

> *(2026-07-30: the former `03-lya_escape` subproject was retired — escape work shares the `01`
> pipeline entirely, so it now lives there. Note also that the `02`/`03` numbers were previously
> swapped between this list and the actual directories; `02-lae_catalog` is correct.)*

## See also

- `docs/literature/` — reference literature
- `docs/VDFI Notes.txt` — running working notes (meeting notes, dated technical decisions)
