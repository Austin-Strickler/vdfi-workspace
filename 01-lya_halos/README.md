# 01 - Lya Halos

Lyman-alpha halo stacking pipeline: measures the extended, diffuse Lya emission around galaxies by stacking spectra/imaging across a sample and characterizing the resulting halo signal (including comparisons from stellar mass to dark matter halo mass to virial radius).

## Scope — this folder is the whole program

As of 2026-07-30 this is not just the halo-stacking paper. **`01-lya_halos` is the umbrella folder
for the entire VDFI Lyα science program** while the paper structure is still exploratory:

- the **virial-radius transition** in an absolute velocity frame,
- the **subsample-split** program (one-halo tracks the galaxy, two-halo tracks the environment),
- the **escape-fraction / geometry–throughput** work (redistribution, not loss).

There is no folder-per-paper, deliberately. The measurement machinery (`measure_*`, `fitting.py`,
`background.py`, the three-zone decomposition) is shared across all of it, and **we do not yet know
whether this is one, two, or three papers** — splitting the tree before the papers are real just
creates stale cross-references. The retired `03-lya_escape` subproject is the cautionary example.

**Start at [`docs/GOALS.md`](docs/GOALS.md)** — that is the current scope document, including the
per-paper sections, the three-zone vocabulary glossary, and the open feasibility questions.

## Pipeline

Built around `utils_lya_halo`, a three-stage pipeline keyed on a single `PipelineConfig` (field, catalog, etc.) so every run and output is reproducible and traceable back to its config:

1. **Extract** (slow, run once) — pull galaxy data into FITS products
2. **Stack** (fast, re-run freely) — build stacks from extracted data
3. **Measure** (fast) — centroids, errors, and derived measurements

A **Validate** layer sits alongside the three stages (null tests, injection-recovery, method comparisons, literature checks) rather than being folded into Measure — "does this look right" is treated as a separate question from "how do we build it."

```python
from utils_lya_halo import PipelineConfig, run_extract, run_stack, run_measure

cfg = PipelineConfig(field="AEGIS", catalog="MOSDEF")
path   = run_extract(cfg)
stacks = run_stack(cfg, path)
result = run_measure(cfg, stacks)
```

## Layout

- `code/` — notebooks (numbered by stage: `0_Extractor_*`, `1_Measure_*`, `2_Validate_*`, `3_*_Viewer`, `4_Guide`, `5_Test*`) and the `utils_lya_halo` package
- `specs/` — design docs written before building a notebook or analysis (see `specs/README.md`)
- `docs/` — background, decisions, and literature notes that don't belong in code or a spec (see `docs/README.md`)

## Function reference

`utils_lya_halo.guide()` is the living function reference for this package — call it from a notebook for a browsable index. `check_guide()` diffs the guide against the real code and flags anything new, renamed, or missing, so the reference doesn't drift from what's actually implemented.
