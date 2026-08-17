"""
utils_lya_halo/splits_recipes.py -- THE recipe library. All of it, in one place.

This is the single source of truth for what splits exist. `splits.py` imports it
at module load, so `splits.LIBRARY`, `splits.show()`, `splits.preview()` and
`splits.run()` all see whatever is written here, with no extra step.

    from utils_lya_halo import splits
    splits.show()                      # already loaded -- nothing to call first

After editing this file:

    import importlib
    from utils_lya_halo import splits, splits_recipes
    importlib.reload(splits_recipes)
    splits.use_recipes(splits_recipes.RECIPES)     # merge the edits into LIBRARY

WHY THIS FILE EXISTS
--------------------
The recipes used to live inside splits.py, with a JSON override file layered on
top. Two problems, and they compounded: JSON cannot hold comments, so every
"why is this edge 9.5" and "why does this recipe need that flag cut" had nowhere
to live; and the override layer meant the actual recipe list was split across a
file that looked nearly empty and 250 lines buried in a 2800-line module.

The argument for the override layer was "don't edit the shipped library in place
or your copy diverges from future versions." That is the right argument for a
third-party package. There is no upstream here -- these are your own files -- so
the indirection bought nothing and cost discoverability.

WHAT A `predicted` STRING IS FOR
--------------------------------
It only counts as a pre-registration if it was written BEFORE you saw the number.
That is what committing this file to git buys: the timestamp is the proof. Every
prediction below still carries `predicted_by=None`, which means Claude drafted it
from docs/subsample_splits.md and nobody has signed off. `show()` flags those. A
prediction you did not write is not a pre-registration -- it is a guess with your
name on it. Rewrite the ones that matter and set predicted_by="austin".

Editing `doc`, `predicted` or `predicted_by` does NOT invalidate the results
cache -- they are excluded from recipe_hash deliberately, so fixing a typo or
signing a prediction cannot cost you a 40-minute batch. Editing anything else
does, correctly.

FAMILIES -- READ BEFORE ADDING THE SIXTEENTH SPLIT
--------------------------------------------------
    family 1  the mass sequence (mass, SFR, sSFR, SigmaSFR, size, dust)
              -- effective dimensionality ~2 NO MATTER HOW MANY YOU RUN
    family 2  externally set (inclination, environment, z, field)
              -- close to their full count in independent information
    family 3  Lya-internal (dv, FWHM, EW, L_Lya, f_esc)
              -- ~2, plus circularity with the thing you are measuring

~15 recipes buy ~7 effectively independent axes. Pick ACROSS families, not
within one. `splits.independence()` measures this on your actual catalog.
"""
from __future__ import annotations


# =====================================================================
# TEMPLATE -- every field a recipe can have
# =====================================================================
# Copy this to start a new one. Only `name`, `column` and `mode` are required;
# everything else has a default and can be deleted. normalize_recipe REJECTS any
# key not on this list, deliberately: a typo'd key ("controls", "n_bin") that
# silently did nothing would give you a perfectly plausible wrong answer.
#
# TEMPLATE = dict(
#     name           = "my_split",     # unique. ALSO THE CACHE FILENAME STEM --
#                                      #   change it and you orphan old results.
#     column         = "SOME_COLUMN",  # split-catalog column to cut on
#     mode           = "half",         # half | bounds | trend | groups
#                                      #   half   median split, 2 sides
#                                      #   bounds explicit value edges
#                                      #   trend  n_bins equal-N bins + Spearman
#                                      #   groups categorical, one side per group
#     n_bins         = 4,              # trend only -- SEE THE p_floor NOTE below
#     bounds         = [0, 9.5, 10.1, 99.0],   # bounds only, len = n_sides + 1
#     groups         = [["a"], ["b", "c"]],    # groups only; inner lists MERGE
#     labels         = ["low", "high"],# display labels, one per side
#     control        = "MASS_50",      # held fixed while splitting: the sample is
#                                      #   binned in `control` first and the split
#                                      #   is done WITHIN each bin, so both sides
#                                      #   end up with matched control
#                                      #   distributions. This is how you say
#                                      #   "at fixed mass".
#     n_control_bins = 4,              # more bins = tighter matching, fewer per
#                                      #   cell. 4 is right for ~450 galaxies.
#     base           = "CLEAN",        # CLEAN | KEEP | ALL, or any boolean column.
#                                      #   CLEAN = AGN-excluded. Only `agn` differs.
#     require        = ("FLAG == 0",), # extra cut expressions, evaluated on the
#                                      #   catalog. Tuple of strings.
#     needs_core     = None,           # None | "covered" | "detected"
#                                      #   covered  the core spectrum exists
#                                      #   detected the Lya line is detected
#     min_n          = 80,             # WARNS below this. NEVER refuses to run.
#     per_field      = False,          # cut inside each field, then pool. REQUIRED
#                                      #   when the column is normalized within
#                                      #   its own field (overdensity).
#     field_col      = "FIELD",
#     predicted      = "...",          # written BEFORE the run. Never enforced.
#     predicted_by   = "austin",       # anything but None/"claude-draft" clears
#                                      #   the UNREVIEWED flag.
#     confirmatory   = True,           # True pre-registered headline test
#                                      #   False exploratory
#     family         = 1,              # 1 mass-seq | 2 external | 3 Lya-internal
#     decomposition  = False,          # True = a decomposition of a headline
#                                      #   result, not an independent test of it
#     doc            = "...",          # pointer back to docs/subsample_splits.md
# )
#
# ---- p_floor: choosing n_bins for mode="trend" ----------------------
# The permutation null has only n_bins! distinct orderings, so the SMALLEST
# two-sided p the test can ever return is 2/n_bins!, attained when the bins come
# out perfectly monotonic:
#
#     n_bins = 3  ->  p_floor = 0.333    can never be significant at all
#     n_bins = 4  ->  p_floor = 0.083    cannot reach p < 0.05 EITHER
#     n_bins = 5  ->  p_floor = 0.017    the first n_bins that can clear 0.05
#     n_bins = 6  ->  p_floor = 0.0028   but per-bin S/N is collapsing by here
#
# With ~443 galaxies, 5 bins is ~89 per bin -- INNER/CORE QUANTITIES ONLY.


RECIPES = {

# =====================================================================
# FAMILY 1 -- THE MASS SEQUENCE
# =====================================================================
# All of these are correlated with each other by construction. Running six of
# them does not buy six results; it buys about two, plus a lot of table rows.

    # -----------------------------------------------------------------
    # mass_half -- the median mass split
    # -----------------------------------------------------------------
    # The S/N-optimal companion to mass_bounds below. Report the PAIR as one
    # result, not as two independent findings: same column, same galaxies, two
    # ways of drawing the line.
    "mass_half": dict(
        name="mass_half", column="MASS_50", mode="half",
        labels=["low M*", "high M*"],
        family=1, confirmatory=True,
        predicted="high M* -> larger h1 and larger crossover radius (bigger halo)",
        predicted_by=None,
        doc="Split #1. S/N-optimal companion to mass_bounds; ONE result together.",
    ),

    # -----------------------------------------------------------------
    # mass_bounds -- the value-cut mass split
    # -----------------------------------------------------------------
    # Edges CONFIRMED by Austin 2026-08-02: 9.5 and 10.1. (splits.py shipped
    # placeholders of 9.4/10.2; these are the real ones.)
    #
    # Why value cuts as well as percentiles: percentile edges move whenever the
    # eligible sample changes, so a percentile split is not comparable to a
    # theory curve from one paper to the next. Halo-mass physics -- shock
    # heating, the transition from cold-mode accretion -- happens at a SPECIFIC
    # mass. This is the version a theorist can overplot.
    "mass_bounds": dict(
        name="mass_bounds", column="MASS_50", mode="bounds",
        bounds=[0.0, 9.5, 10.1, 99.0],
        labels=["logM<9.5", "9.5-10.1", "logM>10.1"],
        family=1, confirmatory=True,
        predicted="a monotonic h1(M*) with the break near the shock-heating mass",
        predicted_by=None,
        doc="Split #1, value-cut version. Edges confirmed 2026-08-02.",
    ),

    # -----------------------------------------------------------------
    # sfr_massctrl -- SFR at fixed mass
    # -----------------------------------------------------------------
    # AT FIXED MASS THIS *IS* THE sSFR SPLIT. sSFR = SFR - M in log, so once mass
    # is controlled the two are the same variable (correlation exactly 1.0).
    # See `ssfr` below for why there is deliberately no ssfr_massctrl.
    "sfr_massctrl": dict(
        name="sfr_massctrl", column="SFR_50", mode="half", control="MASS_50",
        labels=["low SFR", "high SFR"],
        family=1, confirmatory=True,
        predicted="high SFR at fixed M* -> more redshifted core centroid (outflow)",
        predicted_by=None,
        doc="Split #1/#2. At fixed mass this IS the sSFR split -- see `ssfr`.",
    ),

    # -----------------------------------------------------------------
    # ssfr -- UNCONTROLLED, deliberately
    # -----------------------------------------------------------------
    # There is no `ssfr_massctrl` and there must not be: it would be a second
    # printed row of sfr_massctrl's measurement wearing a different name, and two
    # rows of the same measurement in one table is how a reader concludes you
    # found two things. Uncontrolled sSFR is a genuinely different cut because
    # mass floats.
    #
    # Note also: the catalog carries SSFR_50 directly. multicat deliberately does
    # NOT recompute it as SFR_50 - MASS_50, which would produce a second, subtly
    # different column with no way to tell which one a table was built on.
    "ssfr": dict(
        name="ssfr", column="SSFR_50", mode="half",
        labels=["low sSFR", "high sSFR"],
        family=1,
        predicted="high sSFR -> stronger outflow signature",
        predicted_by=None,
        doc="Split #1. UNCONTROLLED ONLY -- at fixed mass sSFR and SFR are the "
            "same variable (spec Sec 7.2a).",
    ),

    # -----------------------------------------------------------------
    # sigma_sfr_massctrl -- SFR surface density at fixed mass
    # -----------------------------------------------------------------
    # The feedback axis: high SFR per unit area drives outflows that open escape
    # channels for Lya.
    #
    # HONEST CAVEAT, and it belongs in the paper: at fixed mass AND fixed size
    # this collapses onto sfr_massctrl. It is a distinct axis here ONLY because
    # size floats -- which makes it partly a size split. Report it that way
    # rather than as a clean feedback measurement.
    #
    # SIGMA_SFR is derived (multicat.SPLIT_DERIVED) as
    # log SFR - log(2 pi Re_kpc^2), and ASSUMES SFR_50 is log10. A linear SFR_50
    # makes it silently wrong, not obviously wrong.
    "sigma_sfr_massctrl": dict(
        name="sigma_sfr_massctrl", column="SIGMA_SFR", mode="half",
        control="MASS_50", labels=["low SigSFR", "high SigSFR"],
        family=1, confirmatory=True,
        predicted="high SigmaSFR -> larger h1, more redshifted core centroid",
        predicted_by=None,
        doc="Split #2. At fixed mass AND size this collapses onto sfr_massctrl, "
            "so it is partly a size split -- say so.",
    ),

    # -----------------------------------------------------------------
    # size_massctrl -- half-light radius at fixed mass
    # -----------------------------------------------------------------
    # The `require` cuts are not boilerplate. RE_KPC tails out to 10-15 kpc for a
    # handful of galaxies, and at z~2 that is not a plausible half-light radius
    # -- it is a blended or failed GALFIT fit. Without the flag + error cuts
    # those objects land in the "extended" side and drag it.
    #
    # Cost: this recipe runs on ~150/443 galaxies (34%), so it is a materially
    # different sample from the catalog-only splits. Compare it to them with care.
    "size_massctrl": dict(
        name="size_massctrl", column="RE_KPC", mode="half", control="MASS_50",
        labels=["compact", "extended"],
        require=("GALFIT_FLAG_F160W == 0", "RE_ERR_F160W < 0.3 * RE_CIRC_BEST"),
        family=1,
        predicted="larger R_e at fixed M* -> larger h1 (baryonic, not purely "
                  "gravitational, halo scale)",
        predicted_by=None,
        doc="Split #11. Flag + error cuts remove failed GALFIT fits that would "
            "otherwise pile into the 'extended' side.",
    ),

    # -----------------------------------------------------------------
    # dust_massctrl -- attenuation at fixed mass
    # -----------------------------------------------------------------
    # FIXED 2026-08-02: was column="AV", which is not on the catalog, so this
    # reported BLOCKED. The catalog carries EBV.
    #
    # The switch is a LABELING fix, not a science change: A_V = R_V * E(B-V) with
    # R_V constant, so a median split on either puts exactly the same galaxies on
    # the same sides. Only the printed median value changes.
    #
    # ** OPEN QUESTION worth settling before this goes in the paper. ** Which
    # attenuation is this? If EBV is the stellar-continuum E(B-V) from SED
    # fitting -- which is what a MOSDEF-style catalog normally carries -- it is
    # continuum-weighted, whereas Lya escape is governed by attenuation along the
    # resonant-scattering path, closer to the nebular value from the Balmer
    # decrement. The two differ by roughly a factor of ~2 in E(B-V) at these
    # redshifts AND the ratio is mass-dependent, so it does not cancel in a
    # split. If a nebular E(B-V) exists in the catalog, prefer it and point
    # `column` at it.
    "dust_massctrl": dict(
        name="dust_massctrl", column="EBV", mode="half", control="MASS_50",
        labels=["low E(B-V)", "high E(B-V)"],
        family=1,
        predicted="higher attenuation -> lower Lya throughput, smaller core "
                  "luminosity; halo extent less affected if the halo is "
                  "scattered light",
        predicted_by=None,
        doc="Split #5. Lighter-weight proxy for the photon-budget question f_esc "
            "answers more fully. STELLAR vs NEBULAR E(B-V) is unresolved -- see "
            "the comment above this recipe.",
    ),

    # -----------------------------------------------------------------
    # delta_ms -- offset from the star-forming main sequence
    # -----------------------------------------------------------------
    # UNBLOCKED 2026-08-02: DELTA_MS is now a real column, added to
    # multicat.SPLIT_DERIVED using the Speagle+2014 main sequence (their eq. 28
    # "time" parameterisation, evaluated at each galaxy's own age of the universe
    # off the pipeline's Planck18 cosmology).
    #
    # THE CALIBRATION IS A CHOICE, not a detail. Speagle+2014 and Whitaker+2014
    # disagree by ~0.2 dex in normalization at z~2 and have different mass
    # slopes, so they move which galaxies land above vs below the line -- most at
    # the low-mass end, which is exactly where the logM<9.5 bin lives. Speagle is
    # used because it is a redshift-continuous fit to a 25-study compilation
    # rather than a binned single-survey fit. Override it via
    # build_split_catalog(derived_registry=...) if you want Whitaker.
    #
    # ** NOT AN INDEPENDENT AXIS. ** At fixed mass, Delta(MS) is sSFR up to the
    # relation's mass slope. So delta_ms, ssfr and sfr_massctrl are approximately
    # ONE family-1 axis with three names. Running all three is fine; reporting
    # them as three findings is not. The MS scatter is ~0.2 dex -- compare any
    # split against that, not against zero.
    "delta_ms": dict(
        name="delta_ms", column="DELTA_MS", mode="half",
        labels=["below MS", "above MS"],
        family=1,
        predicted="above the main sequence -> stronger outflow signature at "
                  "fixed M*",
        predicted_by=None,
        doc="Split #1/#2 variant. Speagle+2014 MS. Near-degenerate with `ssfr` "
            "and `sfr_massctrl` -- one axis, three names.",
    ),


# =====================================================================
# FAMILY 2 -- EXTERNALLY SET PROPERTIES
# =====================================================================
# These are the ones that actually buy independent information, because what
# sets them has nothing to do with the galaxy's internal Lya physics.

    # -----------------------------------------------------------------
    # inclination -- THE cleanest axis in the program
    # -----------------------------------------------------------------
    # Viewing angle is set by random orientation on the sky. It is independent of
    # every intrinsic property BY CONSTRUCTION, not by statistical adjustment.
    # Nothing else in this library can say that. It is also the cheapest real
    # addition -- the axis ratio is already in the GALFIT catalog.
    #
    # OPEN: the flag cut is inherited from the size split, and axis ratio is
    # better constrained than R_e for marginal fits -- so requiring
    # GALFIT_FLAG_F160W == 0 here may be needlessly strict and is costing sample
    # (this recipe runs on 161/443). Worth testing whether relaxing it changes
    # anything.
    "inclination": dict(
        name="inclination", column="INCL_DEG", mode="half",
        labels=["face-on", "edge-on"],
        require=("GALFIT_FLAG_F160W == 0",),
        family=2, confirmatory=True,
        predicted="edge-on -> narrower, more redshifted Lya (longer scattering "
                  "path through the disk); face-on -> more escape down the "
                  "outflow cone",
        predicted_by=None,
        doc="Split #8. Orientation is random, so this axis is independent of "
            "everything else by construction.",
    ),

    # -----------------------------------------------------------------
    # environment -- Chartab+2020 CANDELS overdensity
    # -----------------------------------------------------------------
    # per_field=True IS REQUIRED, NOT COSMETIC. The overdensity is normalized
    # within its own field, so pooling COSMOS and EGS before splitting mixes two
    # different normalizations and the resulting "overdense" side is partly just
    # "whichever field has the higher normalization." Consistent with the rest of
    # the pipeline, where the fields are extracted and diagnosed separately.
    "environment": dict(
        name="environment", column="DELTA", mode="half", per_field=True,
        labels=["underdense", "overdense"],
        family=2, confirmatory=True,
        predicted="overdense -> larger two-halo term, flatter outer profile "
                  "(satellite clustering)",
        predicted_by=None,
        doc="Split #6, Chartab+2020 CANDELS overdensity. per_field=True is "
            "required -- the density is normalized within each field.",
    ),

    # -----------------------------------------------------------------
    # redshift_massctrl -- z~2 vs z~3 at fixed mass
    # -----------------------------------------------------------------
    # Doubles as a systematics check, which is the more useful reading of it: the
    # two redshift bins carry DIFFERENT magnitude cuts (H<24.5 vs H<25). That is
    # a selection effect you can measure here rather than a confound you have to
    # accept elsewhere. ~1 Gyr is not long enough to expect much real evolution.
    "redshift_massctrl": dict(
        name="redshift_massctrl", column="z", mode="half", control="MASS_50",
        labels=["z~2", "z~3"],
        family=2,
        predicted="little change over ~1 Gyr; any change is as likely the "
                  "magnitude cut as evolution",
        predicted_by=None,
        doc="Split #4. Also a systematics check -- the two bins carry different "
            "magnitude cuts.",
    ),

    # -----------------------------------------------------------------
    # field -- COSMOS vs EGS
    # -----------------------------------------------------------------
    # A pure systematics split, not in the numbered list. If the two fields
    # disagree beyond their errors, EVERY other split's error bar in this program
    # is understated, because none of them include a field-to-field variance
    # term. Run it, and run it early.
    #
    # 'AEGIS' and 'EGS' are the same field under two naming conventions, hence
    # both strings in the second group.
    "field": dict(
        name="field", column="FIELD", mode="groups",
        groups=[["COSMOS"], ["AEGIS", "EGS"]],
        labels=["COSMOS", "EGS"],
        family=2,
        predicted="no difference; any difference is instrumental or a "
                  "field-to-field variance floor",
        predicted_by=None,
        doc="Systematics split. Disagreement beyond the errors means every other "
            "error bar here is understated.",
    ),


# =====================================================================
# FAMILY 3 -- Lya-INTERNAL
# =====================================================================
# Splitting the halo measurement by a property of the same line you are
# measuring. Powerful, and the easiest place in the program to fool yourself.
#
# THE SHARED CAVEAT (spec Sec 7.5): core and outer-bin photon noise are
# independent, and large-scale kinematics are assumed disjoint from internal
# radiative transfer -- so these splits are NOT automatically regression to the
# mean. The ONE channel that couples them is the systemic-redshift error, which
# enters the core Delta_v and the halo centroid WITH THE SAME SIGN. Run the
# redshift-quality stratification before believing any detection here.

    # -----------------------------------------------------------------
    # delta_v_lya -- blueshifted vs redshifted core
    # -----------------------------------------------------------------
    # A DECOMPOSITION of the centroid-vs-radius headline result, not an
    # independent test of it. Hence confirmatory=False and decomposition=True:
    # `table()` and `describe()` label it as such so it cannot be read as a
    # second, corroborating detection.
    #
    # The prediction is a NULL at large radius, and that is the informative
    # outcome. A large-radius response would be one of: a shared systemic-z
    # error, real physics, or a bug -- and only the first is directly testable.
    "delta_v_lya": dict(
        name="delta_v_lya", column="CORE_V_OFFSET", mode="half",
        labels=["blueshifted core", "redshifted core"],
        needs_core="detected", family=3, decomposition=True,
        predicted="null at large radius. Core Delta_v and the halo centroid "
                  "share no photon noise; a large-radius response would be a "
                  "shared systemic-z error, real physics, or a bug -- and the "
                  "first is testable",
        predicted_by=None,
        doc="Split #3. A decomposition of the centroid-vs-radius result, not an "
            "independent test of it (spec Sec 7.5a).",
    ),

    # -----------------------------------------------------------------
    # fwhm_lya -- narrow vs broad core
    # -----------------------------------------------------------------
    # Same Sec 7.5 caveats as delta_v_lya. Note the resolution context: the
    # effective resolution is 1.9 A rest-frame, so a double peak IS resolvable at
    # R~750 in wide-peak stacks -- CORE_SIGMA is a real measurement here, not a
    # blend, though that is sample-dependent.
    "fwhm_lya": dict(
        name="fwhm_lya", column="CORE_SIGMA", mode="half",
        labels=["narrow Lya", "broad Lya"],
        needs_core="detected", family=3, decomposition=True,
        predicted="broader core -> larger scattering optical depth -> more "
                  "extended halo",
        predicted_by=None,
        doc="Split #9 variant. Same Sec 7.5 caveats as delta_v_lya.",
    ),

    # -----------------------------------------------------------------
    # ew_lya -- rest-frame equivalent width
    # -----------------------------------------------------------------
    # The continuous-variable complement to lya_class, and the split with the
    # most direct literature comparison (Steidel+2011, Momose+2014 both find
    # higher-EW objects have more extended halos).
    "ew_lya": dict(
        name="ew_lya", column="CORE_EW_REST", mode="half",
        labels=["low EW", "high EW"],
        needs_core="detected", family=3,
        predicted="high EW -> more extended halo (Steidel+2011, Momose+2014)",
        predicted_by=None,
        doc="Split #7. Continuous-variable complement to lya_class; the best "
            "direct literature comparison in the program.",
    ),

    # -----------------------------------------------------------------
    # lya_lum -- faint vs bright core
    # -----------------------------------------------------------------
    # PARTLY CIRCULAR WITH fesc_hactrl by construction, since f_esc ~
    # L_Lya/L_Ha,corr. Do not report the two as independent results.
    #
    # CORE_LOGL_LYA is derived from the matched-filter core flux, so it is
    # defined for non-detections -- but it is NaN for absorbers (no log of a
    # non-positive flux), which is why this recipe drops to 256 galaxies.
    "lya_lum": dict(
        name="lya_lum", column="CORE_LOGL_LYA", mode="half",
        labels=["faint Lya", "bright Lya"],
        needs_core="detected", family=3,
        predicted="brighter core -> brighter halo, roughly self-similar profile",
        predicted_by=None,
        doc="Split #12-adjacent. Partly circular with fesc_hactrl -- not "
            "independent of it.",
    ),

    # -----------------------------------------------------------------
    # fesc_hbctrl -- Lya escape fraction at fixed Hbeta
    # -----------------------------------------------------------------
    # f_esc ~ L_Lya / L_Hb,corr, so it is partly circular with lya_lum BY
    # CONSTRUCTION. Controlling for HB_FLUX is the mitigation, and it is not a
    # complete one -- SAY SO IN THE PAPER rather than letting a referee find it.
    "fesc_hbctrl": dict(
        name="fesc_hbctrl", column="CORE_FESC", mode="half", control="HB_FLUX",
        labels=["low f_esc", "high f_esc"],
        needs_core="detected", family=3,
        predicted="high f_esc -> more core-dominated, LESS extended halo if the "
                  "halo is scattered light that failed to escape directly",
        predicted_by='austin',
        doc="Split #12. Circular with lya_lum by construction; the HB_FLUX "
            "control is a partial mitigation, not a fix. Disclose it.",
    ),

    # -----------------------------------------------------------------
    # fesc -- Lya escape fraction only
    # -----------------------------------------------------------------
    # f_esc ~ L_Lya / L_Hb,corr, so it is partly circular with lya_lum BY
    # CONSTRUCTION.
    "fesc": dict(
        name="fesc", column="CORE_FESC", mode="half", control=None,
        labels=["low f_esc", "high f_esc"],
        needs_core="detected", family=3,
        predicted="high f_esc -> more core-dominated, LESS extended halo if the "
                  "halo is scattered light that failed to escape directly",
        predicted_by='austin',
        doc="Split #12. Circular with lya_lum by construction; the HA_FLUX "
            "control is a partial mitigation, not a fix. Disclose it.",
    ),

    # -----------------------------------------------------------------
    # lya_class -- absorber / null / emitter
    # -----------------------------------------------------------------
    # needs_core="covered", NOT "detected". Requiring a detection would delete
    # the null and absorber classes, which ARE the point of this split.
    #
    # ** THE ABSORBER SIDE HAS NO MEASURABLE PROFILE FIT. ** It is N=42 with no
    # positive core, so the exp-core term's amplitude collapses, the power law
    # dominates at every radius, and the fitted "crossover" comes back at ~0.09
    # kpc -- a numerical artifact, not a radius. This crashed the 2026-08-01
    # batch; splits.py now catches it and reports NO BOUNDARY instead. Expect
    # that, and treat absorber h1 / r_x as NOT MEASURED. The measurable claim for
    # that side is the stacked halo FLUX at fixed radius, which is what the
    # Trainor+2025 comparison actually needs.
    #
    # Group strings must match core.classify_lya's labels EXACTLY.
    "lya_class": dict(
        name="lya_class", column="CORE_LABEL", mode="groups",
        groups=[["absorber"], ["null"], ["emitter"]],
        labels=["absorber", "null", "emitter"],
        needs_core="covered", family=3,
        predicted="even absorbers are net halo emitters (Trainor+2025)",
        predicted_by=None,
        doc="Split #13b. Absorber side: expect a degenerate fit -- quote stacked "
            "halo flux, never h1.",
    ),

    # -----------------------------------------------------------------
    # agn -- the only base='KEEP' recipe
    # -----------------------------------------------------------------
    # Every other recipe defaults to base='CLEAN' (AGN-excluded). This one needs
    # the AGN back, so it is the single entry that overrides `base`.
    #
    # It has a downstream consequence worth remembering: if the two populations
    # come out indistinguishable in the halo, promoting the GLOBAL default to
    # KEEP buys ~15% more galaxies in every other split. That promotion waits on
    # this result. It is not assumed.
    "agn": dict(
        name="agn", column="AGN_FLAG", mode="groups",
        groups=[[False], [True]],
        labels=["non-AGN", "AGN"],
        base="KEEP", family=3,
        predicted="AGN -> brighter, more extended halo (fluorescence)",
        predicted_by=None,
        doc="Split #13a. The only base='KEEP' entry. A null here would justify "
            "promoting the global default to KEEP (+15% sample everywhere).",
    ),


# =====================================================================
# TREND RECIPES
# =====================================================================

    # -----------------------------------------------------------------
    # mass_trend4 -- monotonic h1(M*) across 4 bins
    # -----------------------------------------------------------------
    # This is split #14 done SAFELY. The original #14 (sort by per-galaxy Lya
    # scale length, then measure scale length) is structurally circular and is
    # excluded from this library on purpose. The non-circular version is to sort
    # by a CLEANLY MEASURED EXTERNAL property, stack in 4-5 groups, and fit h1
    # per group.
    #
    # A monotonic trend across 4 bins is 1-in-12 under the null, versus 1-in-2
    # for a binary split -- that ordering constraint is the entire reason trend
    # mode exists. BUT: see the p_floor table at the top. 4 bins CANNOT reach
    # p<0.05 no matter how clean the trend is. If you want a p-value, use 5 bins
    # (mass_trend5, below). Keep this one for the 4-bin visual.
    #
    # Per-bin S/N drops to ~110 galaxies, so this is INNER/CORE QUANTITIES ONLY.
    "mass_trend4": dict(
        name="mass_trend4", column="MASS_50", mode="trend", n_bins=4,
        family=1,
        predicted="monotonic h1(M*)",
        predicted_by=None,
        doc="Split #14, non-circular version. 4 bins cannot reach p<0.05 -- see "
            "mass_trend5. Inner/core quantities only.",
    ),

    # -----------------------------------------------------------------
    # mass_trend5 -- the version that can actually return a p-value
    # -----------------------------------------------------------------
    # Same idea as mass_trend4 with the one change that matters: 5 bins puts
    # p_floor at 0.017, so a perfectly monotonic result is reportable. ~89
    # galaxies per bin.
    "mass_trend5": dict(
        name="mass_trend5", column="MASS_50", mode="trend", n_bins=5,
        family=1,
        predicted="monotonic h1(M*)",
        predicted_by=None,
        doc="Split #14 at 5 bins -- the smallest n_bins whose permutation test "
            "can clear p<0.05 (p_floor=0.017). ~89 galaxies/bin.",
    ),

}


# =====================================================================
# BATCHES -- named lists to run together
# =====================================================================
# Here rather than in the notebook so a batch is a committed, reproducible object
# instead of a cell you edited three times today.

BATCH_SMOKE = ["mass_half", "inclination"]      # fast: one per family, both cheap

BATCH_HEADLINE = [
    # The confirmatory set, spread ACROSS families (spec Sec 7.2).
    "mass_half",            # family 1
    "sfr_massctrl",         # family 1
    "sigma_sfr_massctrl",   # family 1
    "inclination",          # family 2 -- the cleanest axis in the program
    "environment",          # family 2
]

BATCH_FULL = BATCH_HEADLINE + [
    "mass_bounds",
    "redshift_massctrl",
    "field",                # systematics: run this one early
    "delta_v_lya",          # family 3 -- decompositions, not independent tests
    "fwhm_lya",
    "ew_lya",
    "lya_lum",
    "lya_class",            # expect the absorber side to report NO BOUNDARY
    "agn",                  # the only base='KEEP' recipe
]

# Everything, including the recipes that run on materially reduced samples.
BATCH_EVERYTHING = list(RECIPES)
