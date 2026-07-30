# vdfi-workspace
This is the hub of all VDFI projects lead by Austin Strickler

## Subprojects

- [`01-lya_halos`](01-lya_halos/) — **the main project.** Lyman-alpha halo stacking pipeline (Extract → Stack → Measure → Validate) *and* the umbrella for the whole exploratory science program: the virial-radius transition, the subsample splits, and the escape-fraction / geometry–throughput work. Scope and paper structure live in [`01-lya_halos/docs/GOALS.md`](01-lya_halos/docs/GOALS.md).
- [`02-lae_catalog`](02-lae_catalog/) — scaffolded, not yet started. Genuinely independent of `01`.

> **Note (2026-07-30):** the former `03-lya_escape` subproject has been retired. It never held code —
> only framework documents — and those now live in `01-lya_halos/docs/`
> (`Greg-Initial-Driving-Model.md`, `Greg-Origins-and-Structure-Outline.md`). Escape-fraction work
> shares the `01` pipeline entirely, so a separate folder bought nothing but stale
> cross-references. If it ever grows its own code tree it can be re-scaffolded then.

## Shared

- `docs/` — cross-project background, including reference literature in `docs/literature/`
- `templates/` — starter skeletons for notebooks, modules, and specs
- `briefings/` — dated weekly digests (changelog, specs, freshness, literature scan) — see `briefings/README.md`
- `CHANGELOG.md` — dated log of work across the workspace
- `TODO.md` — near-term working items
