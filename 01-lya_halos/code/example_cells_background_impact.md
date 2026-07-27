# `compare_background_impact` — example cells

Section 2.6 of `background.py`. Needs a product written with `store_spec_nobg=True`
**and** `store_raw_bg=True`.

Run these **per field**, not on a merged product — production extracts COSMOS and
EGS separately and combines finalized galaxies afterward.

---

## Cell 1 — sanity first, before anything expensive

```python
import numpy as np
from utils_lya_halo import background as B
from utils_lya_halo.config import PipelineConfig

cfg = PipelineConfig()
p   = p_cosmos            # one field

# (a) does the offline reconstruction reproduce what the science was measured with?
ver = B.verify_reconstruction(p, cfg)

# (b) where does the annulus actually land in kpc? (a RANGE, not one radius)
span = B._annulus_kpc_span(p, cfg)
print(f"\n{span['annulus_arcsec'][0]:g}-{span['annulus_arcsec'][1]:g}\" -> "
      f"inner edge {span['inner_lo']:.0f}-{span['inner_hi']:.0f} kpc "
      f"(median {span['inner_med']:.0f}), outer edge "
      f"{span['outer_lo']:.0f}-{span['outer_hi']:.0f} kpc")

# (c) which science bins does each stored annulus overlap?
_ = B.bg_variant_radii(p, cfg, verbose=True)
```

If (a) fails, stop — nothing below means anything.

---

## Cell 2 — the main run

```python
imp = B.compare_background_impact(p, cfg, n_boot=200)
```

Prints three per-bin tables (flux, centroid, pedestal), the Q1-vs-Q2 split, and a
verdict. Default sweep is the 5 rows: fiducial, `fill=median`,
`fill=median_interp`, **no background at all**, outermost annulus.

Read the summary block at the bottom first:

- **Q1 fill-method spread** — how much the interpolation choice matters
- **Q2 no-background bracket** — how much subtracting *at all* matters
- **method spread as % of bracket** — if that's a few percent, Q1 is settled

---

## Cell 3 — the figures

```python
fig1, ax1 = B.plot_background_impact(imp)          # summary, 3 panels vs radius
fig2, ax2 = B.plot_background_impact_bins(imp)     # per-bin detail, outer 4 bins
```

In the per-bin lower panels: a **flat offset** means the choice only moved the zero
point (line shape and centroid untouched). A **Lyα-shaped** difference means the
choice is eating signal. Those are different problems.

To look at specific bins:

```python
fig, ax = B.plot_background_impact_bins(imp, bins=[3, 4], zoom_A=60)
```

---

## Cell 4 — custom sweep

`choices` takes any `reconstruct_spec` kwargs, plus `"no_background": True`:

```python
choices = [
    {"label": "fiducial",       "variant": 0},
    {"label": "mask +/-5 A",    "variant": 0, "mask_halfwidth_A": 5.0},
    {"label": "mask +/-20 A",   "variant": 0, "mask_halfwidth_A": 20.0},
    {"label": "sigma 8 A",      "variant": 0, "sigma_A": 8.0},
    {"label": "sigma 60 A",     "variant": 0, "sigma_A": 60.0},
    {"label": "no background",  "no_background": True},
]
imp2 = B.compare_background_impact(p, cfg, choices=choices, n_boot=200)
```

Only add these rows if Cell 2 showed the fill methods are *not* negligible —
one knob at a time, never a grid.

---

## Cell 5 — both fields, compared

```python
res = {}
for name, prod in (("COSMOS", p_cosmos), ("EGS", p_egs)):
    print(f"\n{'='*20} {name} {'='*20}")
    res[name] = B.compare_background_impact(prod, cfg, n_boot=200)

for name, r in res.items():
    print(f"{name:8s}  Q1 fill spread {r['method_spread_sigma']:5.2f} sigma   "
          f"Q2 bracket {r['bracket_sigma']:6.2f} sigma   "
          f"({100*r['method_spread_sigma']/r['bracket_sigma']:.0f}%)")
```

If the two fields disagree about the bracket, that's the ~0.003 pedestal offset
between them showing up in the science bins, and it needs its own sentence.

---

## Cell 6 — pull numbers out for the Methods text

```python
r_edges = imp["r_edges"]
clean   = imp["pedestal_is_clean"]
fid     = imp["rows"][imp["reference"]]

print("bins where the pedestal test is valid (no real UV continuum):")
for b in np.where(clean)[0]:
    print(f"  {r_edges[b]:g}-{r_edges[b+1]:g} kpc: "
          f"pedestal = {fid['pedestal'][b]:.3e} +/- {fid['pedestal_err'][b]:.3e} "
          f"({fid['pedestal_sigma'][b]:+.1f} sigma)")

print(f"\nlargest fill-method flux shift : {imp['method_spread_sigma']:.2f} sigma")
print(f"largest centroid shift, any    : {imp['max_dcent_sigma']:.2f} sigma")
print(f"no-background bracket          : {imp['bracket_sigma']:.2f} sigma")
```

---

## Cell 7 — test 1: read the paired columns

Nothing extra to run — `compare_background_impact` now prints two columns per choice:

- **`abs`** = Δ / reference's own 1σ → **does it matter?** Small means the shift is
  buried in your statistical error.
- **`pair`** = Δ / σ of the paired per-draw difference → **is it real?** The choices
  share galaxies and noise, so their difference is far better determined than either.

`pair = 20, abs = 0.1` means a reproducible systematic that is irrelevant next to your
error bars — that's the *good* outcome. `pair = 20, abs = 1.5` means it belongs in the
error budget and **more galaxies will not average it away**.

The pairing was free: `bootstrap_all` seeds a fresh RNG and draws
`rng.integers(0, ngal, ngal)` per iteration consuming nothing else, so with the same
seed/nboot/ngal every choice already saw identical galaxy sets. `paired_ok` on each
row records that those conditions actually held.

```python
r = imp["rows"][1]                       # any non-reference choice
for b in range(len(imp["r_mid"])):
    print(f"{imp['r_edges'][b]:g}-{imp['r_edges'][b+1]:g} kpc: "
          f"dflux = {r['dflux'][b]:+.3e}  "
          f"abs {r['dflux_sigma'][b]:+.2f}  pair {r['dflux_paired_sigma'][b]:+.1f}")
```

## Cell 8 — test 2: is the annulus dependence a continuum-window artifact?

```python
scan = B.continuum_window_scan(p, cfg, n_boot=200)
fig, ax = B.plot_continuum_window_scan(scan)
```

Walks `cont_bounds` from the production position out past the background smoothing
kernel and reports whether the annulus spread collapses. At z≈2.3 the kernel is
**9.1 Å rest** (σ=30 Å observed) and the production windows start 9.3–10.7 Å from Lyα
— about **one kernel σ**, where a Gaussian still carries ~60% of peak. So the residual
bump left by mask-then-smooth sits on the pixels that define your continuum.

Verdict is automatic: >50% drop = window artifact (one-line `cont_bounds` fix);
<20% = real annulus dependence for the error budget.

Custom window sets:

```python
scan = B.continuum_window_scan(p, cfg, n_boot=200, window_sets={
    "production": ((1195,1205), (1225,1236)),
    "2 sigma":    ((1180,1195), (1245,1260)),
    "3 sigma":    ((1165,1185), (1258,1280)),
    "red only":   ((1258,1285),),
})
```

Pick the **closest** windows that clear the kernel, not the furthest — further out
means lower S/N per continuum pixel and more room for real features.

## What's in the returned dict

| key | meaning |
|---|---|
| `rows` | one per choice: `flux`, `centroid`, `pedestal` (+ errors), `dflux_sigma`, `dcentroid_sigma`, `pedestal_sigma`, `stack_flux`, `rest_wave` |
| `rows[i]["dflux_paired_sigma"]` | Δ / paired-difference σ — **is it real** |
| `rows[i]["flux_gain"]` | how much tighter the paired error is than the absolute |
| `rows[i]["paired_ok"]` | whether seed/nboot/ngal actually matched the reference |
| `annulus_span` | the annulus in kpc, as a percentile range over the z distribution |
| `pedestal_is_clean` | per-bin bool — where the pedestal test is non-circular |
| `method_spread_sigma` / `bracket_sigma` | the Q1 and Q2 one-number summaries |
| `verify` | the `verify_reconstruction` result |

---

## Changes made to existing code

- `reconstruct_background` / `reconstruct_spec` gained `fill_method` and
  `fill_sideband_width_A`. Default `None` = the production path, byte-identical.
  `fill_method="linear"` is asserted equal to it on every run (verified: max
  difference exactly 0.0).
- **`_pedestal_per_bin` error is now a galaxy bootstrap.** It was
  `mad_std(pixels)/sqrt(npix)`, which assumes independent continuum pixels — but the
  stack is rest-frame resampled per galaxy (`np.interp` blends neighbours) on top of
  the LSF, so the effective N is several times smaller than `npix`. That estimator
  understated the error by roughly the square root of the correlation length and
  turned a ~2σ pedestal into a ~10σ one. **Every pedestal σ in the tables you ran
  before this change is inflated.**
- **`measure._measurement_window` now accepts any number of continuum sidebands.**
  It hard-unpacked exactly two, so a red-only continuum crashed the pipeline — even
  though `get_continuum_model` and `fit_local_poly_continuum` both loop over
  `cont_bounds` and always supported any number. A one-sided continuum is legitimate
  on Lyα (the blue side carries forest attenuation the red side doesn't), so the
  restriction was in the wrong place.
- `smooth_fill_then_gauss`, `compare_fill_methods` and `plot_fill_method_gallery`
  were never registered in `guide.py` — they are now, so `check_guide()` covers them.
