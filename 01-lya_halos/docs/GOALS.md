# Goals — 01 Lya Halos

See `docs/vdfi-background.md` (workspace root) for the HETDEX/VDFI dataset context this project builds on.

## Overarching goal

Quantify the extent and kinematics of Lyα halos around a sample of cosmic-noon galaxies, using that to learn about galaxy formation and how galaxies interact with their environment — accretion, inflows/outflows, satellite contribution, and what actually powers the extended emission. Open questions this can speak to: how do galaxies accrete mass; what does a galaxy's Lyα halo tell you about the galaxy and its immediate environment; what can a *stack* of galaxies tell you that individual measurements can't (and what do we lose by stacking, given our medium-resolution spectrograph)?

## Dataset

- MOSDEF galaxies, magnitude-cut roughly H<24.5 (z~2 bin) and H<25 (z~3 bin) — precise cuts TBD/to confirm.
- ~1500 MOSDEF galaxies total across all MOSDEF fields; ~500 after cutting to the COSMOS/EGS footprint we have VDFI coverage for. Working number, not final.
- Precise rest-frame optical **systemic** redshifts from MOSDEF, reportedly ~60 km/s precision (unverified — came from one source, needs a real citation check before it goes in the paper).

**Why this dataset matters:** the systemic redshift is the project's central methodological advantage. Most comparable studies (e.g. Guo et al. 2024) measure Lyα centroid shifts *relative to the Lyα line itself* (self-referenced), which is a different and less direct quantity. Anchoring to an independent, systemic redshift means we can ask directly where the Lyα centroid sits relative to true zero velocity — blueshifted, redshifted, or at rest — rather than only tracking relative shifts within the line profile.

## Core measurements

1. **Centroid velocity vs radius** — the primary result this project is driving toward. Is the line at systemic velocity, blueshifted, redshifted, and how does that change with radius? Radial bins are already expressed in units of R/Rvir (stellar-mass → halo-mass → virial-radius mapping, currently Moster et al. 2013) — that scaling is implemented and in active use, including per-galaxy mini-extractions within a galaxy's own R/Rvir. `TODO.md` tracks revisiting whether Moster+2013 is still the right SMHM relation to use, not building the mapping itself.
2. **Light (flux) profile** vs radius.

Subsample splits (below) aren't a third measurement alongside these — they're an overarching axis that cuts across both: the real question is how the centroid and flux profiles *differ* across subsamples, not a separate quantity.

### Why centroid-vs-radius is the headline result

Guo et al. 2024 measured a similar red-to-blue centroid shift with radius, but relative to the line itself, not to a systemic redshift. Most other papers either lack a systemic anchor entirely or only measure individual (non-stacked) objects. Grounding the profile in MOSDEF's systemic redshift is what would make this measurement distinctive.

The physical read on *where* the centroid sits is still an open, tentative interpretation — worth investigating but not asserted as settled: a blueshifted peak is plausibly consistent with scattering off inflowing/accreting gas (not necessarily radiative cooling emission itself), a redshifted peak could point to outflows, and subsample splits (e.g. by environment) could speak to satellite clustering effects. Each of these needs a literature check before being stated confidently in the paper — this is one instance of a broader point: several observed features will have candidate physical explanations that need to be checked against the literature rather than assumed.

## Subsample splits (open, top-priority work — not yet started)

Splitting loses statistical power (roughly √N), but is necessary to connect the stack to physical mechanisms. Nothing below is a settled plan — these are candidate directions to explore and narrow down, not a committed list:

- Spectral classification (something like: star-forming, LAE, non-LAE emitter, absorber, null/no-detection, AGN — the categories themselves, especially "absorber," still need to be pinned down)
- Stellar mass
- Star formation rate
- Lyα velocity offset (once centroid-vs-radius is established, splitting the sample by its own offset)
- Other physical mechanisms worth splitting on as they come up — e.g. environment/satellite clustering, outflow vs inflow signatures

This directly feeds the `TODO.md` item on subsample splits and their physical predictions in the literature — that TODO item is the actual driver; treat the list above as a starting point for that thinking, not a plan to execute as-is.

## Validation / systematics (in progress)

These need to land in a final, publication-ready figure/summary before the centroid result can be trusted:

- **Bootstrap errors** — validated as consistent and reproducible; the standard error method for this pipeline.
- **Centroid method comparison** — consistent across estimators, and cross-checked against an independent paper's 3D-HST sample (agreement there is a good sign the result isn't an artifact of this particular dataset).
- **Line spread function (LSF) check** — confirming the ±4 Å measurement window isn't itself dominated by instrumental LSF width (currently looks fine — real signal, not just LSF smearing).
- **Star PSF comparison** — confirms the halo is genuinely more extended than the instrumental PSF.
- **Background noise structure** — outer radial bins are noisy, but the noise is *correlated*, not random — plausibly tied to how hundreds of thousands of fibers get coadded, the annulus definition, background smoothing choice, and masking method, likely compounded by the sample's redshift distribution. Actively being investigated as a way to drive down outer-bin noise; also the direct motivation for solidifying the pipeline choices in `TODO.md`.

## Related work (adjacent, not strictly required for this paper)

- **Core-extracted spectrum / classification** — the per-galaxy core spectrum is what the classification scheme (LAE / null / AGN / etc.) above is actually built from, so it's tied to this project, but some of the deeper escape-fraction work built on it more properly belongs to `02-lya_escape` ("paper 2").

## Definition of done (for a first paper)

- A validated centroid-vs-radius profile (with flux profile and at least one meaningful subsample split), backed by the full validation/systematics suite above, plus a literature comparison (Guo et al. 2024 and others).
- Pipeline choices solidified and defensible — able to justify *why* a given masking percentile or background method was chosen, not just that it was. Prerequisite for actually writing the Methods section; the current bottleneck.
- The paper itself written — Introduction, Data, Methods, Results (tracked in `TODO.md`).
- Final, publication-ready figures for each core measurement that we're actually happy with, not just working versions.

## See also

- `docs/vdfi-background.md` (workspace root) — HETDEX/VDFI dataset context
- `01-lya_halos/README.md` — pipeline stages and usage
- `01-lya_halos/specs/` — per-analysis design docs (write one before a new stacking/measurement approach)
- `TODO.md` (workspace root) — near-term working items
- `presentations/Lya_halos/6-16-26 - VIRAL-Halos Overview.pptx` — source for much of this doc; note per Austin (2026-07-10) that some of its content is outdated relative to what's written here
