# 2-halo → SFRD: the equations, derived

Markdown/LaTeX cells for the notebook. Every constant that appears as a bare number in
`lya_2halo_deproject.py` is derived here from the values it comes from.

---

## 1. The observable

The expcore fit's second term, in erg s⁻¹ (physical kpc)⁻²:

$$\Sigma_{2h}(R) = A_2\left[1 + \left(\frac{R}{r_c}\right)^{2}\right]^{-\gamma_{2d}/2}
\;\xrightarrow[R \gg r_c]{}\; A_2\left(\frac{R}{r_c}\right)^{-\gamma_{2d}}$$

$A_2$ is the $R\to0$ plateau of the cored form and the coefficient of the asymptote.
Only the product enters everything downstream:

$$\boxed{\;\Sigma_{2h}(R) \to \underbrace{\left(A_2\, r_c^{\gamma_{2d}}\right)}_{\text{the invariant}} R^{-\gamma_{2d}}\;}$$

so $A_2$ and $r_c$ are individually meaningless here — quote $A_2 r_c^{\gamma_{2d}}$.
Current value: $8.184\times10^{37} \times 359.6^{0.8} = 9.070\times10^{39}$.

---

## 2. The Abel deprojection — where 3.6791 comes from

Assume a power-law 3D emissivity $j(r) = j_0 (r/r_p)^{-\gamma_{3d}}$ and project it along
the line of sight at impact parameter $R$, with $r=\sqrt{R^2+\pi^2}$:

$$\Sigma(R) = \int_{-\infty}^{\infty} j\!\left(\sqrt{R^2+\pi^2}\right)\,d\pi
= j_0\,r_p^{\gamma_{3d}}\int_{-\infty}^{\infty}\left(R^2+\pi^2\right)^{-\gamma_{3d}/2} d\pi$$

Substitute $\pi = Ru$, so $d\pi = R\,du$ and $(R^2+\pi^2)^{-\gamma/2} = R^{-\gamma}(1+u^2)^{-\gamma/2}$:

$$\Sigma(R) = j_0\,r_p^{\gamma_{3d}}\,R^{1-\gamma_{3d}}
\underbrace{\int_{-\infty}^{\infty}\left(1+u^2\right)^{-\gamma_{3d}/2} du}_{\equiv\,C(\gamma_{3d})}$$

That integral is a Beta function. With $u = \tan\theta$ it becomes
$\int_{-\pi/2}^{\pi/2}\cos^{\gamma-2}\theta \,d\theta = B\!\left(\tfrac12,\tfrac{\gamma-1}{2}\right)$, hence

$$\boxed{\;C(\gamma) = \frac{\Gamma(\tfrac12)\,\Gamma\!\left(\tfrac{\gamma-1}{2}\right)}{\Gamma\!\left(\tfrac{\gamma}{2}\right)}
= \sqrt{\pi}\,\frac{\Gamma\!\left(\tfrac{\gamma-1}{2}\right)}{\Gamma\!\left(\tfrac{\gamma}{2}\right)}\;}$$

**Convergence:** finite iff $\gamma_{3d} > 1$. No $\pi_{\max}$, no depth, no velocity window.

**Two consequences.** First, since $j(R) = j_0 (R/r_p)^{-\gamma_{3d}}$,

$$\Sigma(R) = C(\gamma_{3d})\, j(R)\, R \qquad\Rightarrow\qquad
\boxed{\;j(R) = \frac{\Sigma(R)}{C(\gamma_{3d})\,R}\;}$$

so $L_{\rm eff} \equiv C(\gamma_{3d})R$ is an effective LOS depth **that grows with $R$**
(6.1 cMpc at 500 pkpc, 32.8 at 2700) — not a constant, which is where the old
velocity-derived 20.6 cMpc went wrong.

Second, $\Sigma \propto R^{1-\gamma_{3d}}$, i.e.

$$\boxed{\;\gamma_{3d} = \gamma_{2d} + 1\;}$$

**This relation *is* the deprojection.** Choosing $\gamma_{2d}=0.8$ already committed you to
$\gamma_{3d}=1.8$; that is why 1.8 appears everywhere downstream.

### Evaluating C(1.8)

$$C(1.8) = \sqrt{\pi}\,\frac{\Gamma(0.4)}{\Gamma(0.9)}
= \frac{1.772454 \times 2.218160}{1.068629} = 3.67910$$

```python
import numpy as np
from scipy.special import gamma as G
C = lambda g: np.sqrt(np.pi) * G((g-1)/2) / G(g/2)
print(f"Gamma(0.4) = {G(0.4):.6f},  Gamma(0.9) = {G(0.9):.6f},  sqrt(pi) = {np.sqrt(np.pi):.6f}")
for g in (1.4, 1.8, 2.2, 3.0):
    print(f"  C({g}) = {C(g):.6f}")
# C(3.0) = 0.5 exactly -- a useful closed-form anchor: sqrt(pi)*Gamma(1)/Gamma(1.5) = 2/sqrt(pi)... 
# no: sqrt(pi)*1/(sqrt(pi)/2) = 2. Check against the brute-force integral instead:
u = np.linspace(-4000, 4000, 4_000_001)
print(f"  numeric C(1.8) = {np.trapezoid((1+u**2)**-0.9, u):.6f}")
```

---

## 3. The unit ladder

$\Sigma$ is per **physical** kpc²; the projection integral ran over a **comoving** LOS
coordinate; Madau & Dickinson is quoted per **comoving** Mpc³. So:

$$j_{\rm com} = \underbrace{\frac{\Sigma(R)}{C(\gamma_{3d})\,R}}_{\rm erg/s/pkpc^3}
\times \underbrace{10^{9}}_{\rm pkpc^{3}\to pMpc^{3}}
\times \underbrace{(1+z)^{-3}}_{\rm pMpc^{3}\to cMpc^{3}}$$

At $z=2.3$: $(1+z)^3 = 35.937$. Both factors are mandatory; the second was missing until
2026-08-04 and inflated $j_0$ by exactly $35.9\times$.

---

## 4. Case B + Kennicutt — where 9.1e-43 comes from

Two ingredients, both citable:

**(a) Kennicutt 1998** (ARA&A 36, 189), Hα star-formation calibration, **Salpeter IMF**:

$$\mathrm{SFR}\;[{\rm M_\odot\,yr^{-1}}] = 7.9\times10^{-42}\; L(\mathrm{H}\alpha)\;[{\rm erg\,s^{-1}}]$$

**(b) Case B recombination** (Brocklehurst 1971; $T=10^4$ K, $n_e \sim 350\,$cm⁻³),
Lyα-to-Hα intrinsic ratio:

$$\frac{L(\mathrm{Ly}\alpha)}{L(\mathrm{H}\alpha)} = 8.7$$

Combine:

$$\mathrm{SFR} = \frac{7.9\times10^{-42}}{8.7}\,L(\mathrm{Ly}\alpha)
= \boxed{9.08\times10^{-43}\; L(\mathrm{Ly}\alpha)}$$

which is the `case_B_sfr = 9.1e-43` in the code.

```python
K98_HALPHA = 7.9e-42      # Kennicutt 1998, Salpeter IMF
CASE_B_RATIO = 8.7        # Brocklehurst 1971, T=1e4 K
print(f"case_B_sfr = {K98_HALPHA/CASE_B_RATIO:.4e}   (code uses 9.1e-43)")
print(f"Chabrier IMF (~/1.7) would give {K98_HALPHA/CASE_B_RATIO/1.7:.4e}")
```

⚠️ **IMF systematic.** This is Salpeter. If your MOSDEF/SED SFRs are Chabrier (they usually
are), they are ~1.7× lower for the same light, and mixing the two conventions is a 70%
error. Check which your SED fits use and convert one of them. **This is not currently
handled in the code.**

⚠️ **f_esc.** Applying this to *observed* Lyα gives the SFR only if every Lyα photon escaped.
It doesn't. See §6 — this is the crux.

---

## 5. The clustering debias

Model: total emissivity around a galaxy is the cosmic mean boosted by clustering,

$$j_{\rm tot}(r) = \bar{j}\left[1 + \xi(r)\right], \qquad
\xi(r) = \left(\frac{r}{r_0}\right)^{-\gamma_\xi}$$

Background subtraction removes the constant, so the stack measures the **correlated excess**:

$$j_{\rm meas}(r) = \bar{j}\,\xi(r)
\qquad\Rightarrow\qquad
\boxed{\;\bar{j} = \frac{j_{\rm meas}(r)}{\xi(r)}\;}$$

**Divide by $\xi$, not $(1+\xi)$** — the $(1+\xi)$ form is right only if the mean is still in
your profile, which background subtraction guarantees it isn't.

Because $j_{\rm meas}\propto r^{-\gamma_{3d}}$ and $\xi\propto r^{-\gamma_\xi}$, when
$\gamma_\xi=\gamma_{3d}$ the radial dependence cancels **identically** and

$$\bar{j} = j_0\,r_p^{\gamma_{3d}}\,r_0^{-\gamma_\xi} \quad\text{(a pure amplitude ratio, constant in }r)$$

**That flatness is an assumption, not a result.** It fixes the *shape*, never the
*normalisation* — $\bar{j}$ was free to come out anywhere.

### Units of $r_0$

Literature $r_0$ is in $h^{-1}$ Mpc. **Bielby et al. 2016**, MNRAS 456, 4061
([arXiv:1501.01215](https://arxiv.org/abs/1501.01215)), abstract:

| | $r_0$ [$h^{-1}$ Mpc] |
|---|---|
| LAE autocorrelation | $2.86 \pm 0.33$ |
| **LBG–LAE cross-correlation** | $\mathbf{3.29 \pm 0.57}$ ← use this |
| $z\sim3$ LBG auto (Bielby+2013) | $\sim 4$ |

Use the **cross**-correlation: you measure galaxies × Lyα *emission*. And since
$\bar{j}\propto r_0^{-\gamma_\xi}$, the h-convention alone is a factor
$(1/0.7)^{-1.8} = 0.526$.

$$r_0 = 3.29/h = 3.29/0.7 = 4.700\ \mathrm{cMpc}$$

---

## 6. Mean surface brightness — for the Croft comparison

A comoving shell of thickness $d\chi$ subtending $d\Omega$ has comoving volume
$d\Omega\,\chi^2 d\chi$ and luminosity $\bar{j}\,d\Omega\,\chi^2 d\chi$. The received flux is
$L/(4\pi D_L^2) = L/(4\pi\chi^2(1+z)^2)$, so $\chi^2$ cancels:

$$\frac{dF}{d\Omega} = \frac{\bar{j}\,d\chi}{4\pi (1+z)^2}$$

For a **line**, $\lambda_{\rm obs}=\lambda_0(1+z)$ so $d\lambda_{\rm obs}=\lambda_0\,dz$, and
$d\chi = c\,dz/H(z) = (c/H(z))\,d\lambda_{\rm obs}/\lambda_0$. Therefore

$$\boxed{\;\mu_\lambda = \frac{\bar{j}\;c}{4\pi\,(1+z)^2\,H(z)\,\lambda_0}\;}$$

per steradian, then $\mathrm{cMpc}^{-2}\to\mathrm{cm}^{-2}$ and $\mathrm{sr}\to\mathrm{arcsec}^{-2}$.
No distance and no LOS depth enters — $\chi$ cancelled.

$$H(2.3) = 70\sqrt{0.3(3.3)^3 + 0.7} = 237.2\ {\rm km/s/Mpc}, \qquad c/H = 1263.9\ {\rm cMpc}$$

**Validation.** Put the *known* SFRD through this at a literature escape fraction:

$$\bar{j}_{\rm expected} = \frac{\rho_{\rm MD}(z{=}2.3)}{9.1\times10^{-43}}\times f_{\rm esc}
= \frac{0.1231}{9.1\times10^{-43}}\times 0.05 = 6.76\times10^{39}\ {\rm erg/s/cMpc^3}$$

$$\Rightarrow\ \mu = 1.27\times10^{-22}\ \mathrm{erg\,s^{-1}cm^{-2}\AA^{-1}arcsec^{-2}}$$

versus Croft+2018's 95% upper limit of $1.2\times10^{-22}$ — **they agree to 6%**. That is a
strong independent check that this formula, the case-B factor and the unit ladder are all
right, and it pins down what the Croft limit *means*: it sits essentially exactly at the
Lyα emission expected from the known cosmic SFRD with $f_{\rm esc}\approx5\%$.

```python
import numpy as np
C_KMS, MPC_CM, LYA = 299792.458, 3.0856775814913673e24, 1215.67
SR_PER_ARCSEC2 = (np.pi/180/3600)**2

def mu_lambda(j_com, z, H0=70., Om=0.3, OL=0.7):
    Hz = H0*np.sqrt(Om*(1+z)**3 + OL)
    return j_com*C_KMS/(4*np.pi*(1+z)**2*Hz*LYA)/MPC_CM**2*SR_PER_ARCSEC2

z = 2.3
MD = 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)
j_expected = MD/9.1e-43*0.05
print(f"H(z)            = {70*np.sqrt(0.3*(1+z)**3+0.7):.1f} km/s/Mpc")
print(f"M&D SFRD        = {MD:.4f} Msun/yr/cMpc^3")
print(f"j at f_esc=0.05 = {j_expected:.3e} erg/s/cMpc^3")
print(f"-> mu           = {mu_lambda(j_expected, z):.3e}")
print(f"Croft+2018 UL   = 1.2e-22     ratio = {mu_lambda(j_expected,z)/1.2e-22:.2f}")
```

---

## 7. Putting it together, and the escape-fraction logic (corrected 2026-08-05)

$$\bar{j} = \frac{1}{\xi(r)}\cdot\frac{\Sigma(R)}{C(\gamma_{3d})R}\cdot\frac{10^9}{(1+z)^3}
\qquad
\rho_{\rm SFR}^{\rm min} = 9.1\times10^{-43}\,\bar{j}$$

**$\bar{j}$ is the ESCAPED Lyα emissivity.** No escape fraction enters the deprojection —
it is built entirely from photons that reached the telescope. $f_{\rm esc}$ enters only in
the *last* step, converting light to star formation:

$$\rho_{\rm SFR}^{\rm true} = \frac{9.1\times10^{-43}\,\bar{j}}{f_{\rm esc}}
\;\;\ge\;\; 9.1\times10^{-43}\,\bar{j}$$

so **$\rho_{\rm SFR}$ computed at $f_{\rm esc}=1$ is a LOWER LIMIT, and a smaller
$f_{\rm esc}$ makes the implied SFRD LARGER**, moving *away* from Madau & Dickinson:

| $f_{\rm esc}$ | implied SFRD | × M&D |
|---|---|---|
| 1.00 | 0.132 | 1.1 |
| 0.30 | 0.438 | 3.6 |
| 0.10 | 1.315 | 10.7 |
| **0.05** | **2.630** | **21.4** |
| 0.02 | 6.575 | 53.4 |

### Don't import an escape fraction — invert the question

> **Corrected 2026-08-05 (Austin).** An earlier version compared against "the SFRD at the
> Hayes+2011 global f_esc ≈ 5%." That was wrong three ways: Hayes+2011 (LARS) is **low
> redshift**; it is an **aperture, galaxy-scale** measurement (the fraction escaping an
> individual galaxy, averaged over their sample) despite often being called "global"; and
> z~2 galaxies in this mass range run nearer **20%**, so 5% isn't even the right
> galaxy-scale value. Most importantly, **the fraction of Lyα escaping a galaxy tells you
> nothing about the fraction of cosmic Lyα production observable as diffuse emission a Mpc
> away.** Those are different physical quantities, and **no published measurement of the
> latter exists** — it isn't directly observable, only model-dependent.

So don't import one. Use the ceiling that is exact and assumption-free, $f_{\rm esc}\le1$:

$$\bar{j}_{\rm SF}^{\rm max} = \frac{\rho_{\rm MD}(z)}{9.1\times10^{-43}}
= \frac{0.1231}{9.1\times10^{-43}} = 1.352\times10^{41}\ {\rm erg\,s^{-1}cMpc^{-3}}$$

$$\boxed{\;f_{\rm esc}^{\rm required} = \frac{\bar{j}_{\rm measured}}{\bar{j}_{\rm SF}^{\rm max}}
= \frac{1.445\times10^{41}}{1.352\times10^{41}} = \mathbf{1.069}\;}$$

**More Lyα escapes than the known cosmic SFRD can produce.** That falsifies an in-situ
case-B star-formation origin on a pure photon-budget argument, with no literature value
needed. Independently, μ sits **22.6× above Croft+2018's 95% UL** — and Croft's limit,
expressed self-consistently in these units, caps the escaped fraction of the SF budget at
**4.7%**. *That* is where the "≈5%" came from; it was Croft's own limit restated, never
Hayes.

The earlier "agreement to 7% with Madau & Dickinson" was an artifact of setting
$f_{\rm esc}=1$ — the most optimistic case, not the expected one. **Do not quote it.**

### Hypotheses — several may be true at once (`interpret_2halo()`)

| | Hypothesis | Status |
|---|---|---|
| **H1** | In-situ case-B star formation at that radius | **Falsified** — requires f_esc = 1.07 |
| **H2** | Scattered light from galaxies (Byrohl+2021) | Open, leading |
| **H3** | Collisional excitation / CGM cooling (Byrohl & Nelson 2023) | Open, leading |
| **H4** | UVB fluorescence | Contributor only, ≤10–34% (Bacon+2021) |
| **H5** | A2 / ξ-debias over-estimate | Open, testable now |

**H2 — Byrohl+2021** (MNRAS 506, 5129): in TNG50 "the majority of photons observed in Lyα
haloes are re-scattered photons from star-forming regions," and outer-profile flattening is
dominated by photons from *other halos*. Those photons are already in the cosmic SFRD —
merely moved in space — so converting $j(r)$ to an SFR density at that radius
**double-counts**. It also breaks the $/\xi$ debias: scattered halo light is far more
concentrated around galaxies than $\xi_{\rm gal}$ is, so $b_{\rm Ly\alpha}\gg b_{\rm gal}$
and dividing by $\xi_{\rm gal}$ over-extrapolates. Absorbing 21× needs an effective
$r_0\approx25$ cMpc — not a clustering length, which is exactly the tell.
*Test:* $L_{2h}/L_{\rm core}$ must be a plausible redistribution (order unity).

**H3 — Byrohl & Nelson 2023** (MNRAS 523, 5248; [arXiv:2212.08666](https://arxiv.org/abs/2212.08666)),
verbatim from the abstract: diffuse Lyα filaments "are, in fact, illuminated by photons
which originate, **not from the intergalactic medium itself**, but from within galaxies and
their gaseous halos… primarily sourced by **intermediate mass halos ($10^{10}-10^{11}$
M⊙**), principally due to **collisional excitations** in their circumgalactic media as well
as central, young stellar populations."

Two things follow, and this is the sharpest result of the search. First, **collisional
excitation is powered by gravitational/thermal energy, not recombination following star
formation** — so for that component case B + Kennicutt is not merely mis-calibrated, it is
**the wrong physics**, and any SFRD inferred from it is inflated by an unknown factor.
Second, the host masses ($10^{10}$–$10^{11}$ M⊙) are **below this sample's**, so the
emitters are not the stacked galaxies.

**Bottom line: the cosmic-mean SFRD extrapolation is not quotable.** $\rho_{\rm local}(r)$,
the resolved profile, is unaffected — it involves no $\xi$-debias and no cosmic
extrapolation. And H1's falsification is itself a real, citable result: *the 2-halo Lyα
cannot be in-situ star formation, on photon-budget grounds alone.*

### Also outstanding

- **IMF mismatch.** 7.9e-42 is Salpeter; MOSDEF/SED SFRs are usually Chabrier, ~1.7× lower
  for the same light. Not handled anywhere in the code, and it will bite the
  `case_b_sanity_check` core anchor too.
- **Croft's μ definition** — observed- vs rest-frame per-Å is a factor $(1+z)=3.3$, and in
  the unhelpful direction (rest-frame makes it 74×). Read Croft+2018 §5–6.
