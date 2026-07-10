# 01 - Lya Halos

Lyman-alpha halo stacking pipeline: measures the extended, diffuse Lya emission around galaxies by stacking spectra/imaging across a sample and characterizing the resulting halo signal (including comparisons from stellar mass to dark matter halo mass to virial radius).

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
