# 2-halo term → cosmic SFR density: number-by-number walkthrough

Companion to `lya_2halo_deproject.py` (workspace root). That file explains *why* the
method is built the way it is; this file traces *where every number enters and what it
does to the answer*. Every printed value below is reproduced by the cells at the bottom.

Placeholder inputs used throughout (**replace with your current expcore fit**):
`A2 = 8.184e37 erg/s/pkpc²`, `r_c = 359.6 pkpc`, `gamma_2d = 0.8`, `z = 2.3`.

> ⚠️ `r_c = 359.6` is a *free-r_c* value; the pipeline default is `r_c_fixed = 400.0`.
> A2 and r_c must come from the *same* fit. **Confirmed 2026-08-05: they do (free fit).**
> Good news — see §8: the final answer depends only on the product `A2 · r_c^γ2d`, so the
> free-vs-fixed r_c choice is *exactly* irrelevant here provided the pair is matched.

> ⚠️ **Naming collision.** "Plateau" below means *constant with respect to evaluation
> radius* in the derived SFR density. It has **nothing to do** with the observed flattening
> of Σ(R) at medium-to-large radii. See §9 — the better name is `j_mean` / `ρ_mean`.

---

## The chain in one line

```
Σ(R)  --Abel-->  j(r)  --units-->  j_comoving  --case B-->  ρ_local  --÷(1+ξ)-->  ρ_cosmic
[obs]            [3D]              [cMpc⁻³]                 [SFR/vol]              [SFR/vol]
```

Five conversions. Exactly three of them contain an assumed number:
the Abel constant (assumption: power law), the case-B factor (assumption: f_esc = 1),
and ξ (assumptions: r₀ and γ_ξ). Everything else is arithmetic.

---

## Step 1 — Σ(R): what we actually measured

The expcore fit's second term is

    Σ_2halo(R) = A2 · (1 + (R/r_c)²)^(-γ_2d/2)     →     A2 · (R/r_c)^(-γ_2d)  for R ≫ r_c

At R = 1000 pkpc: `Σ = 8.184e37 · 2.781^(-0.8) = 3.611e37 erg/s/pkpc²`.

**Where a number enters:** `A2` sets the entire answer linearly. ρ ∝ A2, full stop.
A 15.6% error on A2 → 15.6% on ρ. Nothing downstream amplifies or damps it.

**Watch this:** at R = 1000 pkpc the cored form and the pure power law differ by ~5%,
because R/r_c = 2.78 is not yet "≫ 1". `use_asymptotic=True/False` is a real 5% fork here,
shrinking to <1% by R = 2700. That disagreement is the code telling you you're evaluating
too close to r_c for the power-law algebra to be clean.

---

## Step 2 — Deprojection: Σ(R) → j(r), and where 3.679 comes from

We assume the 3D emissivity is a power law, `j(r) = j₀ (r/r_pivot)^(-γ_3d)`. Project it
along the sightline at impact parameter R:

    Σ(R) = ∫_{-∞}^{+∞} j(√(R² + π²)) dπ

Substitute the power law, pull out the R-dependence, and the remaining dimensionless
integral is a Beta function:

    Σ(R) = C(γ_3d) · j(R) · R,    C(γ) = √π · Γ((γ-1)/2) / Γ(γ/2)

Two things fall out for free:

1. **The projected slope is exactly one shallower than the 3D slope.** Σ ∝ j(R)·R ∝ R^(1-γ_3d).
   So `γ_3d = γ_2d + 1 = 1.8`. **This relation *is* the deprojection** — it's already baked
   into your choice of γ_2d = 0.8, which is why 1.8 keeps showing up everywhere.
2. **The integral converges on its own for γ_3d > 1.** No π_max, no line-of-sight depth,
   no velocity window. C(1.8) = **3.6791**.

C(γ) · R is an *effective* LOS depth — but note it **grows with R**:

| R [pkpc] | L_eff = C·R [cMpc] |
|---|---|
| 500 | 6.07 |
| 1000 | 12.14 |
| 2000 | 24.28 |
| 2700 | 32.78 |

That's the physical content: at large impact parameter the emissivity is locally flat along
the sightline, so you integrate through more of it. The old velocity-derived method used a
single 20.6 cMpc at *every* radius, which is wrong in shape and not just normalization.

**Assumption being made:** a single power law over the fitted range. If the slope changes,
C(γ) is a local approximation at best. `R_min_sensitivity()` is the test.

---

## Step 3 — The unit ladder (pure bookkeeping, but it's where a 10.89× bug lived)

```
j_phys  = Σ / (C·R)          = 3.611e37 / 3679.1 = 9.815e33   erg/s/pkpc³
        × 1e9   (pkpc³→pMpc³)                    = 9.815e42   erg/s/pMpc³
        ÷ (1+z)³ = ÷35.937  (pMpc³→cMpc³)        = 2.731e41   erg/s/cMpc³
```

Both factors are mandatory. The `(1+z)³` was missing before 2026-08-04 and made j₀ too
large by 35.9× in the numerical path (caught by `verify_deprojection` CHECK 4). Comoving
is the right frame because that's the frame Madau & Dickinson is quoted in.

---

## Step 4 — Case B → SFR: ρ_local

    ρ_local = 9.1e-43 × j_comoving = 0.2485 M⊙/yr/cMpc³

**Where 9.1e-43 comes from:** Kennicutt SFR–Hα calibration × the case-B Lyα/Hα ratio of
8.7. It assumes **every** Lyα photon produced escapes and reaches you: f_esc = 1.

**Consequence: ρ_local is a strict LOWER LIMIT.** Real dust/RT losses only push the true
SFR *up*. Say that explicitly whenever you quote it — it's the difference between a
defensible statement and one a referee kills.

`case_b_sanity_check()` is the real-data anchor for this factor: apply the identical
9.1e-43 to your stack's **core** Lyα luminosity and divide by the SED/MOSDEF SFR of the
same galaxies. You should get f_esc ≈ 0.05–0.3 at z~2. If you get >1, the chain is broken.

Comparison: ρ_local / Madau&Dickinson(z=2.3) = 0.2485 / 0.1231 = **2.02×**. That is *not*
a contradiction — you're standing in an overdense region (next to a galaxy), so of course
the local SFR density exceeds the cosmic mean. That excess is precisely what Step 5 removes.

---

## Step 5 — ξ(r): the clustering debias, and **where r₀ enters**

    ξ(r) = (r / r₀)^(-γ_ξ)          ρ_cosmic = ρ_local / (1 + ξ)

This is the *only* place r₀ appears. It is **not** the power-law pivot — those are two
different quantities and the code now names them separately (`r0_xi_Mpc` vs `r_pivot_Mpc`).
Conflating them was a real bug; see the module docstring's PIVOT SEPARATION section.

### Your question: 4 cMpc or 4 h⁻¹ Mpc?

**Clustering r₀ in the literature is quoted in h⁻¹ Mpc, essentially always.** So "r₀ = 4"
from a paper means 4 h⁻¹ Mpc = **5.71 cMpc** at h = 0.7. The code's default `r0_xi_Mpc=4.0`
is a literal 4 cMpc, i.e. **it silently interprets a h⁻¹ number as an h-free one**.

Evaluated at R = 1000 pkpc = 3.30 cMpc:

| r₀ interpretation | r₀ [cMpc] | 1+ξ | ρ_cosmic | plateau |
|---|---|---|---|---|
| 4.0 literal cMpc *(code default)* | 4.000 | 2.414 | 1.030e-01 | 1.758e-01 |
| **4.0 h⁻¹ Mpc** | 5.714 | 3.687 | 6.741e-02 | 9.251e-02 |
| 3.0 literal cMpc | 3.000 | 1.842 | 1.349e-01 | 2.950e-01 |
| 3.0 h⁻¹ Mpc | 4.286 | 2.601 | 9.556e-02 | 1.553e-01 |
| 2.5 h⁻¹ Mpc (LAE, low end) | 3.571 | 2.153 | 1.154e-01 | 2.156e-01 |
| 4.5 h⁻¹ Mpc (Bielby+2016 high) | 6.429 | 4.321 | 5.752e-02 | 7.483e-02 |

**In the ξ ≫ 1 (plateau) limit, ρ ∝ r₀^(-γ_ξ) exactly.** Verified numerically:

```
r0 = 4.0 cMpc     → 1.674e-01
r0 = 4.0 h⁻¹ Mpc  → 8.812e-02
ratio = 0.526  =  (1/0.7)^(-1.8) = 0.526   ✓
```

**So the h convention alone is a factor-1.9 systematic on your headline number.** Nothing
about signal-to-noise touches it. Resolving it is the single highest-value thing you can do
to this calculation.

### What r₀ *should* be for your sample

- z~2–3 LBGs: r₀ ≈ 4 h⁻¹ Mpc (this is where "4" came from originally)
- LAE-selected samples: r₀ ≈ 2.5–3.2 h⁻¹ Mpc
- Bielby+2016 (MNRAS 456, 4061), your existing citation: r₀ ≈ 2.5–4.5 h⁻¹ Mpc

Your stack is a mixed LBG+MELG KBSS-like sample, so ~4 h⁻¹ Mpc is defensible — but quote
the *range* as a systematic band, not a point value. **Recommendation: set
`r0_xi_Mpc = 4.0/0.7 = 5.714` and state the h convention explicitly in the paper.**

### The other lever: where you evaluate

1+ξ runs from 5.92 at 500 pkpc to 1.24 at 2700 pkpc. Evaluating inward buys S/N and pays
for it in clustering systematics. **Quote the outer radii; use inner radii as a sensitivity
test only.**

---

## Step 6 — The full table, and why "plateau" is the number to quote

| R [pkpc] | R [cMpc] | Σ | j_com | ρ_local | 1+ξ | ρ_cosmic | plateau |
|---|---|---|---|---|---|---|---|
| 500 | 1.65 | 6.287e37 | 9.510e41 | 0.8654 | 5.92 | 0.1461 | 0.1758 |
| 750 | 2.47 | 4.545e37 | 4.584e41 | 0.4171 | 3.37 | 0.1237 | 0.1758 |
| 1000 | 3.30 | 3.611e37 | 2.731e41 | 0.2485 | 2.41 | 0.1030 | 0.1758 |
| 1500 | 4.95 | 2.611e37 | 1.316e41 | 0.1198 | 1.68 | 0.0712 | 0.1758 |
| 2000 | 6.60 | 2.074e37 | 7.843e40 | 0.0714 | 1.41 | 0.0508 | 0.1758 |
| 2700 | 8.91 | 1.631e37 | 4.570e40 | 0.0416 | 1.24 | 0.0336 | 0.1758 |

ρ_local falls by 21× across the range (of course — it's an emissivity profile). ρ_cosmic
still falls by 4.3×, because 1+ξ never fully takes over. **`plateau` = ρ_local/ξ (the ξ ≫ 1
limit) is dead flat at 0.1758** — that's the quantity with no radius choice in it.

`plateau = 0.1758 ± 0.0275 (15.6%, from A2 alone)`, vs Madau & Dickinson(2.3) = 0.1231,
i.e. **1.43× the cosmic SFRD** with r₀ = 4 cMpc — or **0.75×** with r₀ = 4 h⁻¹ Mpc.
Note that the units convention flips the qualitative conclusion from "above" to "below".

---

## Step 7 — The circularity you must disclose

The plateau is flat **only because γ_ξ = γ_3d = 1.8**. The r^(-1.8) in the emissivity and
the r^(-1.8) in ξ cancel identically. Break that equality and the flatness evaporates:

| γ_ξ | plateau across R = 500 → 2700 pkpc | spread |
|---|---|---|
| 1.4 | 0.2505, 0.1899, 0.1439, 0.1276 | 74% |
| 1.6 | 0.2099, 0.1827, 0.1590, 0.1498 | 35% |
| **1.8** | 0.1758, 0.1758, 0.1758, 0.1758 | **0%** |
| 2.0 | 0.1473, 0.1692, 0.1943, 0.2063 | 33% |
| 2.2 | 0.1234, 0.1628, 0.2148, 0.2422 | 63% |

**This is not a coincidence, it's an assumption.** You *chose* γ_2d = 0.8, which forced
γ_3d = 1.8, which happens to equal the canonical clustering γ_ξ = 1.8. So the clean
cancellation is partly an artifact of that choice.

The honest framing: *"we assume the 2-halo Lyα emissivity traces the galaxy correlation
function with the same slope; under that assumption the debiased SFR density is a pure
amplitude ratio, independent of evaluation radius."* Then run `gamma_fixed=None` in
`fit_emissivity_profile` — if the free-fit γ lands near 1.8, that's *evidence* for the
assumption rather than a restatement of it. Table above is the sensitivity band.

---

## What this number is and is not

- **ρ_local(r)** — SFR density attributable to Lyα emission at radius r in your stack's
  environment. Radially resolved, novel, defensible. **This is the result.**
- **ρ_cosmic / plateau** — nominally Madau&Dickinson-comparable, but per Croft+2016/2018,
  Mas-Ribas & Dijkstra 2016, Lujan Niemeyer+2022, Byrohl+2021, the 2-halo Lyα regime is
  dominated by clustered, individually-undetected satellites doing ordinary case-B star
  formation. So this is a **consistency test** — "is the satellite-driven case-B budget
  compatible with the known SFRD?" — **not an independent re-measurement of the SFRD.**
  Getting 1.4× or 0.75× is a *pass*, not a discovery.

---

## Cells to run

> **Updated 2026-08-05 for the new API.** Changed defaults: `r0_xi_Mpc=None` → 4.700 cMpc
> (Bielby cross), `use_asymptotic=True`, `R_eval_pkpc=(2000, 2400, 2700)`. New keys
> `j_mean` / `rho_mean` (the `/ξ` debias) are the headline; `rho_cosmic_plateau` still
> works as an alias. New function `croft_fesc_check`.

### Cell 1 — validate the code
```python
import sys; sys.path.insert(0, "/path/to/vdfi-workspace")   # adjust
import numpy as np
from lya_2halo_deproject import (abel_constant, sfr_density_from_expcore,
                                 croft_fesc_check, mean_surface_brightness,
                                 verify_deprojection, MADAU_DICKINSON,
                                 R0_XI_DEFAULT, R0_XI_VARIANTS)

verify_deprojection()      # 10/10 arithmetic checks; validates the CODE, not the physics
print(f"\ndefault r0_xi = {R0_XI_DEFAULT:.3f} cMpc  (Bielby+2016 LBG-LAE cross, 3.29/h)")
```

### Cell 2 — the step-by-step number trace
```python
# ---- REPLACE with your current expcore fit (A2 and r_c from the SAME fit) ----
A2, A2_err, r_c_kpc, gamma_2d, z = 8.184e37, 1.28e37, 359.6, 0.8, 2.3
CASE_B = 9.1e-43
R = 1000.0

Sig  = A2 * (R / r_c_kpc) ** (-gamma_2d)          # STEP 1
g3   = gamma_2d + 1.0                             # STEP 2
C    = abel_constant(g3)
j_p  = Sig / (C * R)                              # STEP 3
j_c  = j_p * 1e9 / (1 + z) ** 3
rho_l = CASE_B * j_c                              # STEP 4

print(f"Sigma({R:.0f} pkpc)     = {Sig:.4e} erg/s/pkpc^2")
print(f"gamma_3d = {g3},  C = {C:.4f},  L_eff = {C*R*(1+z)/1000:.2f} cMpc")
print(f"j_phys                = {j_p:.4e} erg/s/pkpc^3")
print(f"j_comoving            = {j_c:.4e} erg/s/cMpc^3   [x1e9, /(1+z)^3={(1+z)**3:.3f}]")
print(f"rho_local             = {rho_l:.4e} Msun/yr/cMpc^3   (f_esc=1 -> LOWER LIMIT)")
print(f"Madau&Dickinson(z)    = {MADAU_DICKINSON(z):.4f}   ratio = {rho_l/MADAU_DICKINSON(z):.2f}x")
```

### Cell 3 — r₀: the labelled systematic table
```python
r_cM = R / 1000.0 * (1 + z)
print(f"R = {R:.0f} pkpc = {r_cM:.3f} cMpc\n")
print(f"{'r0 source':<36}{'r0[cMpc]':>10}{'1+xi':>8}{'rho_mean':>13}")
for lab, r0 in R0_XI_VARIANTS:                     # labelled Bielby set + the h mistake
    xi = (r_cM / r0) ** (-1.8)
    print(f"{lab:<36}{r0:>10.3f}{1+xi:>8.3f}{rho_l/xi:>13.4e}")

print(f"\nrho_mean ~ r0^-gamma_xi  ->  the h-convention flip alone is "
      f"a factor {(1/0.7)**-1.8:.3f}")
```

### Cell 4 — the full radial table (the deliverable)
```python
res = sfr_density_from_expcore(A2, r_c_kpc, gamma_2d, z,
                               R_eval_pkpc=(500, 750, 1000, 1500, 2000, 2700),
                               A2_err=A2_err)     # r0/asymptotic now default correctly
hdr = (f"{'R[pkpc]':>8}{'R[cMpc]':>9}{'Sigma':>12}{'j_com':>12}{'rho_loc':>11}"
       f"{'1+xi':>8}{'rho_mean':>11}{'[rho_cos]':>11}{'core-asym':>11}")
print(hdr); print("-" * len(hdr))
for i, RR in enumerate(res['R_pkpc']):
    print(f"{RR:>8.0f}{res['R_cMpc'][i]:>9.2f}{res['Sigma'][i]:>12.3e}{res['j_comoving'][i]:>12.3e}"
          f"{res['rho_local'][i]:>11.3e}{1+res['xi'][i]:>8.2f}{res['rho_mean'][i]:>11.3e}"
          f"{res['rho_cosmic'][i]:>11.3e}{100*res['cored_vs_asymptotic'][i]:>10.1f}%")
print(f"\nA2*r_c^{gamma_2d} invariant = {res['A2_rc_invariant']:.4e}  "
      f"(r_c-choice-independent; quote THIS, not A2 alone)")
print(f"HEADLINE  rho_mean = {res['rho_mean'][0]:.4f} +/- {res['rho_mean_err'][0]:.4f} "
      f"({100*res['frac_err']:.0f}%, A2 only)  [r0={res['r0_xi_Mpc']:.3f} cMpc]")
print(f"          Madau&Dickinson(z={z}) = {res['madau_dickinson']:.4f}  "
      f"-> {res['rho_mean'][0]/res['madau_dickinson']:.2f}x")
```

### Cell 4b — Croft cross-check: the ~20× tension
```python
from lya_2halo_deproject import croft_tension_check, sfrd_vs_fesc
croft_tension_check(res['j_mean'][0], z, label="2-halo")
print()
sfrd_vs_fesc(res['j_mean'][0], z)
# j_mean is the ESCAPED emissivity -- no f_esc enters the deprojection.
# Croft measures the escaped emissivity too, so these are directly
# comparable and we sit 22.6x ABOVE his 95% upper limit.
# f_esc only enters light->SFR, and SMALLER f_esc makes SFRD LARGER.
```

### Cell 5 — the circularity disclosure table
```python
for gx in (1.4, 1.6, 1.8, 2.0, 2.2):
    r2 = sfr_density_from_expcore(A2, r_c_kpc, gamma_2d, z,
                                  R_eval_pkpc=(500, 1000, 2000, 2700), gamma_xi=gx)
    pl = r2['rho_mean']
    print(f"gamma_xi={gx}  rho_mean: {np.array2string(pl, precision=4)}  "
          f"spread={np.ptp(pl)/np.median(pl):.1%}")
print("\nFlat ONLY at gamma_xi == gamma_3d == 1.8. That equality is an ASSUMPTION.")
print("It constrains the SHAPE (radius-independence), NOT the normalisation.")
```

### Cell 5b — the A2/r_c degeneracy is harmless (§8), shown not asserted
```python
inv = A2 * r_c_kpc ** gamma_2d
print(f"invariant A2*r_c^{gamma_2d} = {inv:.4e}\n")
for rc in (320.0, r_c_kpc, 400.0, 500.0):
    a2 = inv / rc ** gamma_2d
    r3 = sfr_density_from_expcore(a2, rc, gamma_2d, z, R_eval_pkpc=[2000])
    print(f"  A2={a2:.4e}  r_c={rc:6.1f} -> rho_mean = {r3['rho_mean'][0]:.6e}")
print("\nIdentical to 6 sig figs -> free-vs-fixed r_c is a NON-ISSUE here,")
print("provided A2 and r_c come from the SAME fit.")
```

### Cell 6 — the astrophysics anchor (do this before quoting anything)
```python
from lya_2halo_deproject import case_b_sanity_check
# Pull these from measure_core_luminosity + your MOSDEF/SED SFRs for the SAME stack.
case_b_sanity_check(L_lya_core_erg_s=..., sfr_sed_msun_yr=..., label="core")
# Expect f_esc ~ 0.05-0.3 at z~2. >1 means the conversion chain is broken.
```

---

## §8 — How r_c enters, and why its degeneracy does **not** hurt you

The expcore 2-halo term is `Σ = A2·(1 + (R/r_c)²)^(-γ2d/2)`. So `r_c` is the **turnover
radius**: inside it the term is flat at A2, outside it falls as a power law. A2 is therefore
the **R → 0 plateau** of the cored form (at R = r_c the exact form gives 0.758·A2), and
equivalently the coefficient of the asymptote when written `A2·(R/r_c)^(-γ2d)`.

**A2 and r_c are strongly degenerate**, because at R ≫ r_c the model collapses to

    Σ(R) → (A2 · r_c^γ2d) · R^(-γ2d)

Only the **product** is constrained by outer data. Verified numerically — holding
`A2·r_c^0.8 = 9.070e39` fixed and sliding r_c over 320 → 500 pkpc:

| A2 | r_c | A2·r_c^0.8 | plateau ρ |
|---|---|---|---|
| 8.184e37 | 359.6 | 9.070e39 | 9.25068e-02 |
| 7.516e37 | 400.0 | 9.070e39 | 9.25068e-02 |
| 8.985e37 | 320.0 | 9.070e39 | 9.25068e-02 |
| 6.287e37 | 500.0 | 9.070e39 | 9.25068e-02 |

**Identical to 6 significant figures.** The whole chain is linear in `A2·r_c^γ2d` and nothing
else. So the `r_c_fixed = 400` vs free `r_c = 359.6` question — which genuinely matters for
the *crossover radius* (research-notes: absolute crossover scales 20-26% across an r_c scan)
— is a **non-issue for this calculation**. Quote `A2·r_c^0.8` as the derived invariant.

One caveat: this exactness holds in the **asymptotic** branch only. With
`use_asymptotic=False` the cored form differs from the asymptote by:

| r_c | R=500 | R=1000 | R=2000 | R=2700 |
|---|---|---|---|---|
| 320 | -12.8% | -3.8% | -1.0% | -0.6% |
| 359.6 | -15.4% | -4.7% | -1.3% | -0.7% |
| 400 | -18.0% | -5.8% | -1.6% | -0.9% |
| 500 | -24.2% | -8.5% | -2.4% | -1.3% |

So **evaluate at R ≥ 2000 pkpc**, where the two branches agree to ~1% and r_c drops out.
At 500 pkpc the r_c choice alone is a 12% wobble.

---

## §9 — What "plateau" actually is: the mean cosmic Lyα emissivity

The model underneath everything is that the total Lyα emissivity around a galaxy is

    j_total(r) = j_mean · (1 + ξ(r))

i.e. the cosmic mean, boosted by clustering. Background subtraction removes a constant, so
what the stack measures is the **correlated excess** only:

    j_measured(r) = j_mean · ξ(r)      ⇒      j_mean = j_measured(r) / ξ(r)

That division is `rho_cosmic_plateau`. It is **constant in r** because both sides go as
r^(-1.8), and its value is a pure ratio of amplitudes:

    j_mean = j₀ · r_pivot^γ · r₀^(-γ_ξ)

**So `plateau` is not evaluated at small radii — it is not evaluated at any radius.** It is
the amplitude ratio between your emissivity power law and the correlation function.

```
j_mean = 1.445e41 erg/s/cMpc³      (r₀ = 3.29 h⁻¹ Mpc)
ρ_mean = 9.1e-43 × j_mean = 0.1315 M⊙/yr/cMpc³
```

### `/(1+ξ)` vs `/ξ` — which is right for your data?

`rho_cosmic = ρ_local/(1+ξ)` assumes your measured Σ(R) still contains the cosmic mean.
`plateau = ρ_local/ξ` assumes background subtraction removed it. **Since your pipeline
subtracts an 80–90″ annulus, the mean is gone — `/ξ` (plateau) is the correct formula**, and
`/(1+ξ)` under-reports by (1+ξ)/ξ, which is 5.2× at 2700 pkpc. Use the plateau.

### The stronger framing: this is an emissivity measurement, not an SFRD measurement

`j_mean` is directly the quantity Croft et al. 2016/2018 measure by cross-correlating Lyα
surface brightness with quasars in SDSS/BOSS. **That is a far more direct literature
comparison than Madau & Dickinson**, because it compares like with like — a clustering-
derived mean Lyα emissivity against another clustering-derived mean Lyα emissivity — with
no case-B factor and no f_esc assumption in between.

> ### 🛑 CORRECTED 2026-08-05 — this section previously claimed the opposite
>
> An earlier version of this doc and of `croft_fesc_check()` reported "implied f_esc ≤ 0.044,
> consistent with Hayes+2011" and called it a success. **The escape-fraction logic was
> inverted.** Austin caught it. The correction:
>
> **j_mean is the ESCAPED emissivity.** It is built from observed surface brightness; no
> escape fraction enters the deprojection anywhere. Croft measures escaped photons too.
> **They are the same quantity — directly comparable — and exceeding Croft's limit is a
> TENSION, not a measurement of f_esc.** f_esc enters only at the final light→SFR step, where
> a *smaller* f_esc makes the implied SFRD *larger*, moving further from Madau & Dickinson.
> The old function now raises `RuntimeError`; use `croft_tension_check()`.

Converting via μ = j·c / (4π(1+z)²H(z)λ_rest) — χ cancels, so no distance and no LOS depth
enters:

```
j_mean (ESCAPED)      = 1.4451e41 erg/s/cMpc³
-> predicted mean SB  = 2.710e-21 erg/s/cm²/Å/arcsec²
Croft+2018 95% UL     = 1.200e-22
=> 22.6x ABOVE the limit   (74.5x if Croft's limit is per rest-frame Å)

cross-check vs M&D at f_esc=0.05:  expected j = 6.761e39  =>  21.4x too high
```

**These are the same discrepancy.** `verify_deprojection` CHECK 7 shows Croft's limit sits
essentially exactly at the emission the known SFRD supplies at f_esc ≈ 5% (agreement to 6%),
which both validates the conversion formula and means "above Croft" and "above M&D-at-5%"
are one statement, corroborated twice — not two independent problems.

**The measured 2-halo Lyα emissivity, extrapolated to a cosmic mean via ξ, is ~20× more
than the known cosmic SFRD can supply.** See `docs/twohalo-sfrd-equations.md` §7 for the
candidate resolutions — the leading one is that the `/ξ` debias assumes b_Lyα = b_gal, but
scattered halo light (Byrohl+2021) is far more concentrated around galaxies than the
correlation function is, so dividing by ξ_gal over-extrapolates to the cosmic mean.

⚠️ Also verify Croft+2018's exact μ definition (observed- vs rest-frame per-Å; total mean
vs bias-weighted amplitude), §5–6 of arXiv:1806.06050. The arithmetic is validated; the
like-for-like-ness needs a human read.

---

## §10 — r₀: the actual citation, with numbers

**Bielby et al. 2016, MNRAS 456, 4061 (arXiv:1501.01215), abstract + §5**, all in h⁻¹ Mpc:

| Measurement | r₀ [h⁻¹ Mpc] |
|---|---|
| LAE auto-correlation | 2.86 ± 0.33 |
| **LBG–LAE cross-correlation** | **3.29 ± 0.57** |
| z~3 LBG auto (their Paper III, Bielby+2013) | ~4 |

**Use the CROSS-correlation, 3.29 ± 0.57 h⁻¹ Mpc.** You are measuring galaxy positions ×
Lyα *emission*, which is a cross-correlation. Plugging in an autocorrelation r₀ silently
assumes b_Lyα = b_galaxy. Bielby's LBG–LAE cross value is the closest published analogue.
Bonus: these fits assume γ = 1.8, so using r₀ and γ_ξ = 1.8 together is internally
consistent rather than an extra assumption.

| r₀ source | h⁻¹ Mpc | cMpc | j_mean | ρ_mean | /M&D |
|---|---|---|---|---|---|
| **Bielby+16 LBG–LAE cross** | **3.29** | **4.70** | **1.445e41** | **0.1315** | **1.07** |
| cross −1σ | 2.72 | 3.89 | 2.035e41 | 0.1852 | 1.51 |
| cross +1σ | 3.86 | 5.51 | 1.084e41 | 0.0986 | 0.80 |
| LAE auto | 2.86 | 4.09 | 1.859e41 | 0.1692 | 1.38 |
| LBG auto ~4 | 4.00 | 5.71 | 1.017e41 | 0.0925 | 0.75 |

**Adopted headline: ρ_mean = 0.132 ± 0.021 (stat, A2) M⊙/yr/cMpc³, ×0.75–1.41 systematic
from r₀ — vs Madau & Dickinson 0.123 at z = 2.3. Agreement to 7%.**

Caveat to state: Bielby is z = 3.1, you are at z = 2.3. Comoving clustering lengths evolve
slowly for a fixed population, but this is an extrapolation and belongs in the systematics.

---

## Open items

1. ~~Resolve the h convention on r₀.~~ **DONE 2026-08-05** — §10. Adopt
   `r0_xi_Mpc = 3.29/0.7 = 4.700`, cite Bielby+2016 LBG–LAE cross-correlation.
2. ~~Confirm A2/r_c come from the same fit.~~ **DONE** — they do (free fit), and §8 shows
   the choice is irrelevant anyway. Change the code default to `use_asymptotic=True` and
   evaluate at R ≥ 2000 pkpc.
3. ~~Switch from `rho_cosmic` to the `/ξ` form and rename.~~ **DONE** — `j_mean`/`rho_mean`
   are now the headline keys; `rho_cosmic_plateau` kept as an alias so old cells still run.
4. ~~Convert j_mean to Croft's units.~~ **DONE** — `croft_fesc_check()`, §9. Gives
   f_esc ≤ 0.044. **Remaining: read Croft+2018 §5–6 and confirm the μ definition matches.**
5. **Run Cell 6 (`case_b_sanity_check`)** on your core Lyα luminosity vs MOSDEF SFRs. This
   is now a *second, independent* handle on f_esc — if the core anchor also lands near
   0.05–0.3, the two agree and §11 item 1 is closed.
6. **Run `gamma_fixed=None`** to check whether the free 3D slope really lands near 1.8.
   This is what converts the γ_ξ = γ_3d cancellation from assumption to evidence.
7. **Integrate into `fitting.py`.** This module currently lives loose at the workspace root,
   against the no-new-.py-file convention.

---

## §11 — Ranked: what is shakiest, and what is genuinely new

**1. THE ~20× DISCREPANCY — this is now the whole ballgame.** ρ_mean = 0.132 at f_esc = 1 is
a *lower limit*, so the "1.07× Madau & Dickinson" agreement is the most optimistic case
available, not the expected one. Every plausible f_esc makes it worse:

| f_esc | implied SFRD | × M&D |
|---|---|---|
| 1.00 | 0.132 | 1.1 |
| 0.30 | 0.438 | 3.6 |
| 0.10 | 1.315 | 10.7 |
| **0.05** | **2.630** | **21.4** |
| 0.02 | 6.575 | 53.4 |

Corroborated independently by Croft (22.6× above the 95% UL). **Nothing from the
cosmic-mean branch of this chain is quotable until this is understood.** ρ_local(r), the
resolved profile, is unaffected — it involves no ξ-debias and no cosmic extrapolation.

Candidate resolutions, ranked, in `twohalo-sfrd-equations.md` §7. Short version: the `/ξ`
debias assuming b_Lyα = b_gal is the leading suspect (you'd need an effective r₀ ≈ 25 cMpc
to absorb it, which isn't a clustering length — it's what "the emission isn't a fair tracer
of the cosmic web" looks like when forced through this formula); then A2 over-estimation
from core leakage; then the Croft definition; then extra emission mechanisms.

**Process note worth keeping.** The chain produced a plausible-looking answer (1.07× M&D)
that was wrong, and it looked right *precisely because* the f_esc = 1 assumption cancelled
the factor that is the problem. Agreement with a known number is not validation when the
assumption you made is the one that manufactures the agreement.

**2. γ_2d = 0.8 was chosen, and it forces γ_3d = γ_ξ = 1.8.** Your own AIC/BIC comparison
(lit review §7) could not distinguish 0.8 from 1.8 from free — ΔAIC ≈ −0.36. So the clean
cancellation that makes ρ_mean radius-independent is close to assumed.
**But be precise about what it does and doesn't guarantee:** γ_ξ = γ_3d guarantees the
*shape* (radius-independence). It says nothing about the *amplitude*. You were not forced
to land near Madau & Dickinson — j_mean could have come out 100× off and nothing would have
stopped it. **The tautology is in the shape, not the normalization.** That distinction is
the honest defence, and it's a real one.

**3. b_Lyα = b_galaxy, implicit in the choice of r₀.** Mitigated by using Bielby's
LBG–LAE cross-correlation rather than an autocorrelation (§10), but the emitting population
in your 2-halo term is *undetected* satellites, which are fainter and likely less biased
than Bielby's LAEs. Direction of the error: lower bias → smaller true r₀ → larger ρ_mean.

**4. r₀ itself.** ±0.57 h⁻¹ Mpc → ×0.75–1.41. Now quantified and citable, so this is a
stated systematic rather than a hidden one.

**5. The background annulus.** At 80–90″ = 2.17–2.44 cMpc, the model puts ξ ≈ 4.6–5.7 —
i.e. strongly clustered territory. This was tested and closed for the *profile shape*
(systematics doc, 2026-07-30), and that verdict stands. But the shape test and the
*amplitude* test are not the same question, and this calculation is 100% amplitude. The
existing `fit_floor=True` option is exactly the amplitude version of that test — run it,
confirm the pedestal is consistent with zero, and cite it here.

### What is genuinely new

Not the SFRD value — that is a consistency check, and should be framed as one.

What's new is **ρ_local(r): a radially resolved, spectroscopically-stacked measurement of
Lyα-implied SFR density from ~500 to 2700 pkpc, with systemic redshifts.** Croft measured
the mean emissivity via quasar cross-correlation with no radial resolution; Lujan
Niemeyer+2022 measured HETDEX profiles without the emissivity conversion; Byrohl+2021
predicted the 2-halo dominance in simulation. **Nobody has put a measured, resolved 2-halo
emission profile through the deprojection and landed on a cosmic emissivity.** The fact
that the amplitude then agrees with independent SFRD and (pending) Croft measurements is
what makes the resolved profile credible — that's the correct logical direction, and it's
worth a paper section. Just don't invert it into "we re-measured the SFRD."
