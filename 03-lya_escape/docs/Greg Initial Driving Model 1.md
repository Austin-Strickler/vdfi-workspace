# Project: Compact vs. Total Lyα Escape — A Geometry–Time–Environment Framework

## Overview
We measure **compact** and **total** Lyα emission with a **PSF-aware forward model** designed for VDFI (≈2.0″ seeing) and anchored by **JWST** morphology:
- **Compact Lyα (ISM-scale):** flux within κ·`r_e,conv` where `r_e` is the JWST rest-frame optical size **convolved to the VDFI PSF** (κ ≈ 1–1.5).
- **Total Lyα (ISM+CGM):** curve-of-growth to ≳6″ capturing halo light.
Primary observables: **compact fraction** $R_{\rm comp} = F_{\rm comp}/F_{\rm tot}$, **halo fraction** HF = 1–$R_{\rm comp}$, and **halo scale** $r_{\rm halo}$ from a PSF-convolved exponential halo.

---

## The Effective Model (mental picture)

### Layer 1 — **Engine (production)**
Young stars set the **intrinsic** Lyα via ionizing photons → recombination. Scales with SFR, IMF, metallicity.

### Layer 2 — **Valve (escape through the ISM)**
Regulates **how** photons leave the stellar body:
- **Geometry / Opening:** feedback-cleared channels with opening angle $ \theta_{\rm open} $.
- **Column / Dust:** median $ \langle N_{\rm HI}\rangle $, dust optical depth $ \tau_{\rm dust} $.
- **Anisotropy & Shape:** **door vs. tunnel** channels (short “doors” vs. long “hallways”) alter both spectra and spatial compactness.
- **Turbulence (latent):** modifies effective covering fraction and channel stability (we treat it as part of the valve state, though rarely observable directly).

**Outcome:** Wider/cleaner channels ⇒ higher $R_{\rm comp}$. If the valve is restrictive (higher $N_{\rm HI}$, dust, narrow/asymmetric tunnels), photons **don’t vanish**—they leave as halo light.

### Layer 3 — **Distributor (transport in the CGM)**
Resonant scattering (and some fluorescence) redistributes Lyα into an extended halo with scale $r_{\rm halo}$ and fraction HF.

### Layer 4 — **Ecosystem / Environment (beyond the CGM)**
The **baryon cycle** couples galaxies to their surroundings:
- **Inflow along filaments** feeds gas to the CGM/ISM, replenishing columns.
- **Outflows** inject mass/metals into the CGM, widening channels over time.
- **Environment density (δ) & satellites** enhance outer-halo SB and can power in-situ emission at large radii.
- **Halo mass & time:** deeper halos (and later times) trend toward more dust/metallicity and steadier SFHs ⇒ lower $R_{\rm comp}$, higher HF at fixed production.

**Synthesis:** $R_{\rm comp}$ probes the **instantaneous valve state** (geometry + columns + dust), while HF and $r_{\rm halo}$ probe **redistribution** and the **ecosystem** that sustains it.

---

## What sets the trends (predictions you can visualize)

Let “↑/↓” refer to the **compact fraction** $R_{\rm comp}$.

| Observable | Physical link in the model | Expected trend in $R_{\rm comp}$ | Companion halo signal |
|---|---|---|---|
| **Δv = v(Lyα) − v(Hα)** | Kinematics × opacity (outflow & effective HI) | **Δv↓ ⇒ $R_{\rm comp}$↑** ; **Δv↑ (≳300–400 km/s) ⇒ $R_{\rm comp}$↓** | HF↑, $r_{\rm halo}$↑ with larger Δv |
| **E(B–V)** (nebular) | Dust along channels | **E(B–V)↑ ⇒ $R_{\rm comp}$↓** | HF↑; stronger in dusty bins |
| **Metallicity (gas or stellar)** | Dust/metals coupling | **Z↑ ⇒ $R_{\rm comp}$↓** (weak after controlling for dust) | HF↑ (mild) |
| **SFH phase** (burst → post-burst → steady) | Time evolution of $ \theta_{\rm open} $, covering | **Burst ⇒ $R_{\rm comp}$↑** ; **Post-burst ⇒ HF↑, $r_{\rm halo}$↑** ; **Steady ⇒ both low** | Sequence in stacks |
| **r_e (JWST, intrinsic)** | Structural compactness ↔ valve efficiency | **r_e↓ ⇒ $R_{\rm comp}$↑** | HF↓; $r_{\rm halo}$ unchanged or ↓ |
| **M\*** | Indirect via dust/geometry | Weak negative on its own; ~0 after controls | N/A |
| **Lyα EW (total)** | Production × transmission | At fixed dust, higher EW ⇒ **$R_{\rm comp}$↑** | HF↓ at fixed total |
| **Environment (δ, D_fil, satellites)** | Ecosystem feeding & illumination | **At fixed $R_{\rm comp}$**, **HF↑** and **outer SB↑** with δ | $r_{\rm halo}$↑ and outer slope flattens |

---

## Narratives We Will Test

### A) **Geometry rules: core vs. halo redistribution**
- **Claim.** Halos are near-universal; the **partition** (compact vs. halo) traces escape geometry. Core and halo anti-correlate at fixed total; halo scale weakly depends on central properties on average.
- **Key metric.** $R_{\rm comp}$, HF, $r_{\rm halo}$ from PSF-convolved core+halo modeling.

### B) **SFH sequencing: bursts build cores, post-bursts grow halos**
- **Claim.** Compact escape is maximized during bursts; within ~10–50 Myr, channels relax and scattering dominates ⇒ HF and $r_{\rm halo}$ rise.
- **Key metric.** Phase-binned stacks (burst/post-burst/steady) in $R_{\rm comp}$, HF, $r_{\rm halo}$.

*(Environment/baryon-cycle enters each narrative as a **second axis**: it modulates the halo side more than the compact core.)*

---

## Experiments (analysis-ready)

### E1 — **PSF-aware core+halo modeling (per object; stacks for depth)**
Fit **Sérsic(UV-tied core)** + **exponential halo**, both **convolved** with the VDFI PSF; compute $R_{\rm comp}$, HF, $r_{\rm halo}$.  
- Use κ·`r_e,conv` (κ≈1–1.5) for **compact**.  
- Treat $r_{\rm halo}<0.5″$ as **upper limits**.

### E2 — **Δv & Dust response (geometry vs. attenuation)**
Partial correlations and hierarchical regressions:
$$
\mathrm{logit}(R_{\rm comp}) \leftarrow \{\log(\Delta v), E(B\!-\!V), \log Z, \log r_e, {\rm SFH\ phase}, \log M_\*\}
$$
- Expect **β_{Δv}<0**, **β_{E}<0**, **β_{r}<0**, **β_{\rm SFH(burst)}>0**, **β_{M}\approx 0** after controls.

### E3 — **SFH phase stacks (burst → post-burst → steady)**
Stack VDFI Lyα by phase (from JWST SED + Hα EW).  
Measure $ \{ R_{\rm comp}, \mathrm{HF}, r_{\rm halo}\} $ and EW(a) vs aperture.  
- **Falsifiable ordering:** Burst: $R_{\rm comp}$↑, HF↓ → Post-burst: $R_{\rm comp}$↓, HF↑, $r_{\rm halo}$↑ → Steady: both low.

### E4 — **Environment modulation (ecosystem test)**
Cross-match to COSMOS/PRIMER/CEERS density and filament maps.  
At **fixed $R_{\rm comp}$**, test if **HF**, **outer SB**, and **$r_{\rm halo}$** increase with **δ** and **satellite density**.  
- Predict **outer-halo enhancement** with environment; core unchanged.

### E5 — **Lyα EW decomposition**
Split **EW_total** into **EW_compact** and **EW_halo** from the model image.  
- At fixed total Lyα flux, higher $R_{\rm comp}$ ⇒ **EW_compact↑**, **EW_halo↓**.

### E6 — **Robustness & calibration**
- Seeing homogenization; PSF uncertainty ±0.1–0.2″ propagated.  
- **Injection–recovery** of core+halo sources (HF 0.1–0.9, $r_{\rm halo}$ 0.3–1.2″) to set κ and aperture corrections.  
- **Method cross-check:** direct exponentials vs. forward models in stacks.

---

## What would *falsify* the model?
- **No Δv trend** after dust control (contradicts valve-geometry link).  
- **No phase ordering** in stacks (burst ≯ post-burst in $R_{\rm comp}$).  
- **No environment effect** on outer halos at fixed $R_{\rm comp}$ (contradicts ecosystem role).  
- **Strong Mass** dependence after controls (implies mass dominates geometry contrary to expectation).

---

## Outputs & Reporting
- Per-object: $R_{\rm comp}$, HF, $r_{\rm halo}$, Δv, E(B–V), Z, M\*, r_e, SFH phase, Lyα EW (total/compact/halo).  
- Stacks: same set + outer-halo slope and environment bins.  
- Tables with hierarchical-fit coefficients and posterior predictive checks.  
- A short “measurement-limits” box (PSF floors; upper limits on $r_{\rm halo}$).

---

## Data & Cross-survey linkage
- **VDFI:** Lyα maps/spectra (Δv, compact/total via forward modeling).  
- **JWST (COSMOS-Web, PRIMER, CEERS):** intrinsic `r_e` (morphology), SED-based SFH phase.  
- **3D-HST/MOSDEF:** Hα (systemic), E(B–V)_neb, gas-phase Z
- Optional:
  - **Ancillary LSS maps:** overdensity δ, filament distance, satellite counts.

## Methods

We model Lyα emission using a **PSF-convolved forward model** anchored to **JWST morphology**. For each galaxy, we fit a UV-tied Sérsic core and an exponential halo, both convolved with the empirical VDFI PSF (≈2.0″). Compact Lyα is defined within κ·rₑ,conv (κ≈1–1.5), where rₑ,conv is the JWST rest-frame optical size convolved to the VDFI PSF. Total Lyα is measured to ≳6″ using a curve of growth. This provides **compact fraction** $R_{\rm comp} = F_{\rm comp}/F_{\rm tot}$, **halo fraction** HF = 1 − $R_{\rm comp}$, and **halo scale** $r_{\rm halo}$.

We then compute **absolute Lyα escape fractions** by comparing each component’s observed luminosity to the intrinsic recombination luminosity inferred from dust-corrected Balmer lines (MOSDEF/3D-HST):
$$
L_{\rm Ly\alpha,int} = 8.7\,L_{\rm H\alpha,int} \quad \text{or} \quad 24.9\,L_{\rm H\beta,int}
$$
using Case B recombination. This yields
$$
f^{\rm comp}_{\rm esc} = \frac{L_{\rm Ly\alpha,comp}}{L_{\rm Ly\alpha,int}},\qquad
f^{\rm halo}_{\rm esc} = \frac{L_{\rm Ly\alpha,halo}}{L_{\rm Ly\alpha,int}},\qquad
f^{\rm tot}_{\rm esc} = f^{\rm comp}_{\rm esc} + f^{\rm halo}_{\rm esc}.
$$

The model simultaneously captures **geometry (compact vs. halo)** and **throughput (absolute escape)**.  Systematics are addressed via PSF homogenization, injection–recovery bias tests, and hierarchical regressions across Δv, E(B–V), Z, SFH phase, stellar mass, Lyα EW, and environment.  Uncertainties from PSF (±0.1–0.2″), dust correction, and core–halo covariance are propagated throughout.

---

## Checklist

**Inputs**
- VDFI Lyα cube or map  
- Empirical PSF and seeing (≈2.0″)  
- JWST rest-frame optical morphology (rₑ, rₑ,conv)  
- Balmer lines (Hα or Hβ), with $E(B\!-\!V)_{\rm neb}$  
- Derived quantities: Δv(Lyα–Hα), metallicity (gas/star), M\*, SFH phase, Lyα EW, environment metrics (δ, satellites, filaments)

**Modeling Steps**
1. Homogenize seeing; convolve JWST morphology to VDFI PSF.  
2. Fit PSF-convolved Sérsic(UV-tied core) + exponential halo → $F_{\rm comp}$, $F_{\rm halo}$, $r_{\rm halo}$.  
3. Validate total flux with curve-of-growth.  
4. Compute $L_{\rm Ly\alpha,int}$ from dust-corrected Hα/Hβ using Case B.  
5. Derive $R_{\rm comp}$, HF, $r_{\rm halo}$, $f^{\rm comp}_{\rm esc}$, $f^{\rm halo}_{\rm esc}$, $f^{\rm tot}_{\rm esc}$.  
6. Injection–recovery simulations to calibrate κ and flux biases.  
7. Stacking and regression vs. Δv, dust, metallicity, SFH, M\*, EW, and environment.

**Outputs**
- Per object and per stack: $R_{\rm comp}$, HF, $r_{\rm halo}$, $f^{\rm comp}_{\rm esc}$, $f^{\rm halo}_{\rm esc}$, $f^{\rm tot}_{\rm esc}$  
- Context: Δv, E(B–V), Z, M\*, rₑ, SFH phase, Lyα EW, environment metrics  
- Hierarchical-fit coefficients β (with uncertainties) for Δv, dust, Z, SFH, M\*, environment.

---

## Data Table Template

| ID | Field  | Δv (km s⁻¹) | E(B–V) | Z/Z☉ | SFH phase | rₑ (kpc) | M\* (log M☉) | Lyα EW (Å) | R_comp | HF | r_halo (″) | f_esc_comp | f_esc_halo | f_esc_tot | δ_env |
|----|--------|-------------|--------|-------|------------|-----------|---------------|-------------|---------|----|-------------|-------------|-------------|-------------|--------|
| G001 | EGS    | 180 | 0.12 | 0.4 | Burst | 1.0 | 9.3 | 45 | 0.68 | 0.32 | 0.6 | 0.11 | 0.05 | 0.16 | +0.3 |
| G002 | COSMOS | 320 | 0.25 | 0.9 | Post-burst | 2.2 | 10.0 | 35 | 0.34 | 0.66 | 1.0 | 0.04 | 0.09 | 0.13 | −0.2 |
| … | …      | … | … | … | … | … | … | … | … | … | … | … | … | … | … |

---

### Integration of Absolute Escape

Absolute Lyα escape transforms this project from a *relative geometry test* to a *radiative-efficiency experiment*.  
- $R_{\rm comp}$ and HF describe **where** the Lyα goes (geometry).  
- $f^{\rm tot}_{\rm esc}$ describes **how much** of it survives (throughput).  
- Together, $f^{\rm comp}_{\rm esc}$ and $f^{\rm halo}_{\rm esc}$ trace the **division of radiative energy between the ISM and CGM**, while environment and time govern **redistribution** beyond the galaxy.

---

**Deliverable:**  
A unified dataset and modeling framework enabling geometry–throughput coupling: direct correlations between Lyα escape topology ($R_{\rm comp}$, HF, $r_{\rm halo}$) and absolute throughput ($f_{\rm esc}$), across SFH, kinematic, and environmental axes.  This provides a falsifiable test of the “redistribution, not loss” model of Lyα transport within the full baryon cycle.


