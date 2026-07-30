"""
wave_systematics.py -- LOOK at the rest-UV stack before measuring anything.

Display + inventory layer for the wavelength / velocity zero-point work
(`specs/wavelength-zero-point.md`). It answers one question:

    "Do we have the signal to do this?"

Two layers:

    LOOK       line_table / plot_overview / plot_line_zooms / quicklook
    MEASURE    fit_line / fit_lines_joint / plot_fit / scan_velocity

    import wave_systematics as ws

    ws.print_line_catalog()              # the reference table (no data needed)
    ws.add_line("FeII2600","Fe II",2600.17,"resonant")   # extend it yourself

    ws.line_table(stack)                 # what's in range, and is it usable?
    ws.plot_overview(stack)              # whole rest-frame stack, lines marked
    ws.plot_line_zooms(stack)            # zoom panel per line, velocity axis

    f = ws.fit_line(stack, "OIII]")      # one line
    ws.plot_fit(stack, "OIII]", f)       # SEE the fit + residuals
    ws.plot_fit(stack, "OIII]", v_try=[-100, 0, 125])   # nudge it by hand
    ws.scan_velocity(stack, ["OIII]","SiIII]","HeII"])  # chi2 vs v
    j = ws.fit_lines_joint(stack, ["HeII","OIII]","SiIII]","CIII]"])

HOW THE FIT WORKS, AND WHY -- READ THIS, THE SIGNAL IS LOW.

Over a window around each line the model is

    flux(lam) = polynomial_continuum(lam) + SUM_i A_i * Gauss(lam; lam_i(1+v/c), sigma)

fitted by weighted least squares. Four choices are made specifically because
the S/N is low, and each is defensible for a reason worth knowing:

1. SIGMA IS FIXED TO THE INSTRUMENTAL LSF BY DEFAULT (sigma_free=False).
   This is the single biggest stabiliser. With a free width at low S/N the
   Gaussian broadens to swallow a neighbouring noise bump and drags the centre
   with it -- the width and the centroid are strongly covariant, so a runaway
   width IS a corrupted velocity.

   *** THIS JUSTIFICATION IS KIND-DEPENDENT. *** It holds for the NEBULAR
   calibrators: recombination and semi-forbidden lines form in H II gas with
   intrinsic widths of tens of km/s, well under the ~150 km/s LSF sigma, so
   they really are unresolved and a free width buys variance, not physics.
   It does NOT hold for the RESONANT lines. Those are saturated outflow
   troughs spanning several hundred km/s -- genuinely resolved, and forcing
   them to the LSF width makes the fit choose between matching the depth and
   matching the wings, which it resolves by inflating the amplitude. If you
   are fitting resonant lines, set sigma_free=True. Note also that stacking
   broadens EVERYTHING by the galaxy-to-galaxy redshift error, so even the
   nebular lines in a composite are wider than the single-object LSF.

   Whether fixed or free, the width can never go BELOW the LSF -- see
   sigma_bounds in fit_lines_joint.

   *** AND DO NOT SHARE ONE FREE SIGMA ACROSS KINDS. *** fit_lines_joint ties
   a single sigma across all groups. Mixing a nebular calibrator with a
   resonant trough in one sigma_free=True call lets the broad trough drag the
   nebular width up, which drags the SHARED VELOCITY with it -- the exact
   covariance this section warns about, imported through the back door. Fit
   the calibrators jointly; fit the resonant lines separately.

2. THE CONTINUUM IS FIT SIMULTANEOUSLY, NOT SUBTRACTED FIRST. Fitting a
   continuum, subtracting it, then fitting a line treats the continuum as
   known and underestimates the velocity error -- the continuum tilt and the
   line centroid are covariant, and only a joint fit propagates that.

3. ALL COMPONENTS SHARE ONE VELOCITY. A doublet is one kinematic object.
   Fitting components independently spends degrees of freedom on a quantity
   atomic physics already fixed.

4. THE JOINT FIT IS WHERE THE PRECISION ACTUALLY COMES FROM. fit_lines_joint
   ties ONE velocity across several lines, each keeping its own amplitudes and
   its own local continuum. Four anchors at S/N~3 each constrain v roughly
   twice as well as the best single one. If any single line looks marginal --
   and it will -- this is the answer, not a longer exposure.

AND THE HEALTH WARNINGS:

*  scan_velocity IS MORE HONEST THAN AN ERROR BAR. At low S/N the chi2-vs-v
   curve can be flat-bottomed or double-minimum, which means the centroid is
   undetermined no matter how tidy the covariance error looks. Always look at
   the scan before quoting a velocity.
*  THE REPORTED v_err IS A LOWER BOUND. It comes from the fit covariance,
   which assumes the model is right, the errors are Gaussian and uncorrelated,
   and the continuum order is correct. Your stack is oversampled, so adjacent
   pixels ARE correlated and this error is optimistic. The number that goes in
   a paper comes from a galaxy bootstrap, not from here.
*  CHECK split_kms BEFORE BELIEVING ANY VELOCITY. See Line.split_kms.

INPUT. Everything takes the same first argument and figures out what it is:

    ws.plot_overview(wave, flux, err)              # three arrays
    ws.plot_overview({"wave": w, "flux": f, ...})  # a dict
    ws.plot_overview(stacks_result)                # build_stacks output
    ws.plot_overview(stack, cube=rest_cube)        # + (ngal, nwave) cube, so
                                                   #   N-per-pixel is real

WHY N-PER-PIXEL MATTERS HERE. Rest-frame coverage varies across the stack: a
galaxy at z contributes only over [3470/(1+z), 5540/(1+z)]. Over z = 1.17-1.80
every galaxy covers 1600-1979 A, but outside that window the effective N drops
and eventually hits zero. A line sitting where N has fallen to a third of the
sample is not the same measurement as one in the fully-covered region, and the
difference is invisible in the stacked flux alone. Pass `cube=` (or `nspec=`)
and every plot and table reports it.

SEE ALSO
    ../../specs/wavelength-zero-point.md   -- why any of this exists
    plotting.py                            -- the Lya-side display layer
    measure.py::get_continuum_model        -- the continuum model used here
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

C_KMS = 299792.458


# =====================================================================
# 0.  THE LINE LIST
# =====================================================================
# All wavelengths are VACUUM rest-frame Angstrom -- the same convention as
# config.LYA_REST = 1215.67. Do not mix an air list in here (that is the very
# systematic this module exists to help chase; see spec section 1.1).
#
# `kind` is the important field. It says whether a line can anchor a velocity:
#
#   "nebular"        Recombination or semi-forbidden. Formed in HII gas, NOT
#                    resonantly scattered, so it sits at SYSTEMIC velocity.
#                    THESE ARE THE CALIBRATORS.
#   "fine_structure" Si II* fluorescent emission. Escapes after one scattering,
#                    so it is emission rather than absorption, and is roughly
#                    symmetric about systemic -- but occultation and dust bias
#                    it blueward by a model-dependent tens of km/s. Useful as a
#                    cross-check, NOT as an anchor. (spec section 4.6)
#   "resonant"       Resonantly scattered / outflow-driven. Systematically
#                    blueshifted by ~150-200 km/s with comparable scatter.
#                    NOT calibrators. Worth looking at as physics. (spec s.6)
#   "stellar"        Photospheric or stellar-wind (P-Cygni) features. Traces
#                    the stars, not the gas; broad and asymmetric.

# WHICH WAY DOES IT POINT? *** NOT EVERYTHING HERE IS AN EMISSION LINE. ***
# The resonant lines are interstellar ABSORPTION against the stellar continuum
# and will point DOWN. Expecting emission everywhere is an easy way to misread
# a panel. Default expectation per kind:
_KIND_EXPECT = {
    "nebular":        "emission",
    "fine_structure": "emission",
    "resonant":       "ABSORPTION",   # interstellar, against the continuum
    "stellar":        "P-Cygni",      # blue absorption + red emission
}


@dataclass(frozen=True)
class Line:
    name: str
    wave: float                      # zoom centre (A, vacuum rest)
    kind: str                        # see above
    components: tuple = ()           # all wavelengths in the feature
    note: str = ""
    expect: str = ""                 # override; else derived from kind

    @property
    def label(self) -> str:
        return f"{self.name} {self.wave:.2f}"

    @property
    def sense(self) -> str:
        """'emission' / 'ABSORPTION' / 'P-Cygni' / 'mixed' -- which way to look."""
        return self.expect or _KIND_EXPECT.get(self.kind, "?")

    @property
    def split_kms(self):
        """
        Velocity separation of the outermost components, or None if single.

        *** THIS IS THE REDSHIFT-INDEPENDENT CHECK. *** A doublet separation is
        fixed by atomic physics, so measuring it tests the DISPERSION SCALE with
        no reference to any galaxy redshift and no reference to the wavelength
        zero point. If a doublet comes out at its known separation, the feature
        is real and the scale is right, and any remaining offset is a pure zero
        point -- a different and far more tractable problem. If the separation
        is wrong, you are looking at noise or at a misidentification, and the
        offset means nothing. Check this BEFORE interpreting any velocity.
        """
        if len(self.components) < 2:
            return None
        lo, hi = min(self.components), max(self.components)
        return (hi - lo) / self.wave * C_KMS


def _L(name, wave, kind, components=(), note="", expect=""):
    return Line(name, wave, kind, tuple(components) or (wave,), note, expect)


LINES = {
    # ---- nebular: SYSTEMIC. these are the calibrators --------------------
    "NIV]":    _L("N IV]",   1486.50, "nebular", (1483.32, 1486.50),
                  "Semi-forbidden doublet, 641 km/s apart -> resolved. Weak; "
                  "strong only in hard-ionizing / low-metallicity systems."),
    "HeII":    _L("He II",   1640.42, "nebular", (1640.42,),
                  "Recombination, single line, systemic. WATCH: can carry a "
                  "BROAD stellar (Wolf-Rayet) component on top of the narrow "
                  "nebular one -- if the profile looks too wide, that is why, "
                  "and the broad part is not a velocity anchor."),
    "OIII]":   _L("O III]",  1666.15, "nebular", (1660.81, 1666.15),
                  "Semi-forbidden pair, 961 km/s apart -> RESOLVED at VIRUS "
                  "resolution. 1666 is typically ~2.5x 1661. Good anchor. "
                  "*** WARNING: Al II 1670.79 ABSORPTION sits only 4.6 A "
                  "redward = +835 km/s from 1666.15, inside a +/-2000 km/s "
                  "panel. It eats the red continuum sideband and will drag a "
                  "naive centroid BLUE. Use a red sideband that stops short of "
                  "+700 km/s, or fit Al II simultaneously."),
    "NIII]":   _L("N III]",  1749.67, "nebular",
                  (1746.82, 1748.65, 1749.67, 1752.16),
                  "Semi-forbidden multiplet spanning ~900 km/s. Blended and "
                  "usually weak -- inventory only, low expectations. "
                  "*** CONTAMINATED: Ni II 1741.55 and 1751.92 ABSORPTION sit "
                  "inside this multiplet. Any apparent asymmetry here is more "
                  "likely Ni II than N III] kinematics."),
    "NII]":    _L("N II]",   2139.68, "nebular", (2139.68,),
                  "Semi-forbidden, single, systemic. Weak but CLEAN -- no "
                  "close blend, and the region is otherwise quiet. Worth a "
                  "look now that the stack reaches ~2000+ A."),
    "OIII2321":_L("[O III]", 2321.66, "nebular", (2321.66,),
                  "Semi-forbidden, systemic. Only reachable by the LOWEST-z "
                  "galaxies (needs z <= 1.386 to stay under 5540 A), so N will "
                  "be a fraction of the sample -- check Ngal before believing "
                  "it."),
    "CII]2326":_L("C II]",   2326.11, "nebular",
                  (2324.21, 2325.40, 2326.11, 2327.64, 2328.83),
                  "Semi-forbidden multiplet, systemic, spanning ~595 km/s. "
                  "Same low-z-only caveat as [O III] 2321."),
    "SiIII]":  _L("Si III]", 1892.03, "nebular", (1882.71, 1892.03),
                  "*** THE CLEAN ANCHOR. *** Components are 1430 km/s apart = "
                  "~3.6x the LSF FWHM, so 1892 is a RESOLVED SINGLE LINE with "
                  "no blend systematic. Weaker than C III] (~0.2-0.5x) but "
                  "that is the whole point -- see spec section 4.2/4.5."),
    "CIII]":   _L("C III]",  1907.71, "nebular", (1906.68, 1908.73),
                  "*** BRIGHTEST, BUT BLENDED. *** Components only 322 km/s "
                  "apart vs ~400 km/s LSF FWHM. The 1907/1909 ratio is "
                  "DENSITY-DEPENDENT, so a single-Gaussian centroid is biased "
                  "by up to 62 km/s. Never centroid this alone -- fit it "
                  "jointly with Si III]. Zoom centre here is the midpoint, "
                  "NOT a physical wavelength."),

    # ---- fine structure: emission, but model-dependent offset ------------
    "SiII*1194": _L("Si II*", 1194.50, "fine_structure", (1194.50, 1197.39),
                    "Fluoresces after Si II 1190/1193 absorption."),
    "SiII*1265": _L("Si II*", 1264.74, "fine_structure", (1264.74, 1265.02),
                    "Follows Si II 1260. Redward of Lya, so unlike the 1260 "
                    "resonance line it is clear of the forest edge."),
    "SiII*1309": _L("Si II*", 1309.28, "fine_structure", (1309.28,),
                    "Follows the O I 1302 + Si II 1304 blend."),
    "SiII*1533": _L("Si II*", 1533.43, "fine_structure", (1533.43,),
                    "Follows Si II 1526.71. Pairing THIS with that absorption "
                    "line gives an outflow velocity with the wavelength "
                    "calibration cancelling exactly (spec section 4.6)."),

    # ---- resonant: NOT calibrators ---------------------------------------
    "Lya":     _L("Lya",     1215.67, "resonant", (1215.67,),
                  "Reference marker. Not in band below z=1.854."),
    "SiII1190":_L("Si II",   1193.28, "resonant", (1190.42, 1193.28), ""),
    "SiII1260":_L("Si II",   1260.42, "resonant", (1260.42,),
                  "Sits at the Lya forest edge -- treat with suspicion."),
    "OI1302":  _L("O I+Si II", 1303.27, "resonant", (1302.17, 1304.37),
                  "Permanent blend at this resolution."),
    "CII1334": _L("C II",    1334.53, "resonant", (1334.53,), ""),
    "SiIV":    _L("Si IV",   1393.76, "resonant", (1393.76, 1402.77),
                  "Also a stellar P-Cygni wind feature -- mixed origin."),
    "SiII1526":_L("Si II",   1526.71, "resonant", (1526.71,),
                  "Pairs with Si II* 1533.43 for the outflow measurement."),
    "CIV":     _L("C IV",    1549.48, "resonant", (1548.19, 1550.77),
                  "Strong, but interstellar absorption + stellar P-Cygni + "
                  "sometimes nebular emission all superposed. Unusable as an "
                  "anchor; informative about the wind.", expect="mixed"),
    "FeII1608":_L("Fe II",   1608.45, "resonant", (1608.45,), ""),
    "AlII1670":_L("Al II",   1670.79, "resonant", (1670.79,),
                  "Clean, strong LIS line. Good OUTFLOW tracer -- and it sits "
                  "right next to O III] 1666, so check they are separated "
                  "before trusting either."),
    "AlIII":   _L("Al III",  1854.72, "resonant", (1854.72, 1862.79),
                  "Just blueward of Si III] -- worth watching so it does not "
                  "eat your Si III] blue continuum sideband."),
    "NiII1709":_L("Ni II",   1709.60, "resonant", (1709.60,),
                  "Weak ISM absorption."),
    "NiII1741":_L("Ni II",   1741.55, "resonant", (1741.55, 1751.92),
                  "*** Falls INSIDE the N III] 1747-1752 multiplet. If you "
                  "measure N III], you are measuring N III] minus Ni II."),
    "SiII1808":_L("Si II",   1808.01, "resonant", (1808.01,),
                  "Weak, but a clean isolated ISM line in an otherwise empty "
                  "stretch -- a decent outflow-velocity check."),
    "FeIII":   _L("Fe III",  1914.06, "resonant",
                  (1895.46, 1914.06, 1926.30),
                  "*** UV34 multiplet, and it BRACKETS BOTH ANCHORS. 1895.46 "
                  "is 540 km/s from Si III] 1892.03; 1914.06 is 835 km/s from "
                  "C III] 1908.73. Usually weak in star-forming galaxies, but "
                  "if it is present it lands directly in the continuum "
                  "sidebands of the joint fit. Check before trusting the "
                  "1880-1915 A continuum."),
    "ZnII2026":_L("Zn II+Mg I", 2026.14, "resonant", (2026.14, 2026.48),
                  "Permanent blend (Zn II 2026.14 + Mg I 2026.48, 50 km/s "
                  "apart). Classic ISM/DLA absorption; weak but real."),
    "CrII2062":_L("Cr II+Zn II", 2062.24, "resonant",
                  (2056.26, 2062.24, 2062.66, 2066.16),
                  "Cr II triplet with Zn II 2062.66 blended into the middle "
                  "component. Weak absorption complex."),
    "FeII2250":_L("Fe II",   2260.78, "resonant", (2249.88, 2260.78),
                  "Weak ISM absorption; low-z galaxies only."),

    # ---- stellar ----------------------------------------------------------
    "NV":      _L("N V",     1240.81, "stellar", (1238.82, 1242.80),
                  "Stellar wind P-Cygni."),
}

#: kinds that can legitimately anchor a systemic velocity
CALIBRATOR_KINDS = ("nebular",)

_KIND_STYLE = {
    "nebular":        dict(color="tab:green",  ls="-",  label="nebular (systemic)"),
    "fine_structure": dict(color="tab:purple", ls="-.", label="fine structure"),
    "resonant":       dict(color="tab:red",    ls="--", label="resonant (NOT systemic)"),
    "stellar":        dict(color="tab:orange", ls=":",  label="stellar"),
}


# =====================================================================
# 1.  INPUT UNPACKING
# =====================================================================
_WAVE_KEYS = ("rest_wave", "wave_rest", "wave", "wavelength", "lam")
_FLUX_KEYS = ("flux", "stack", "spec", "y")
_ERR_KEYS = ("err", "error", "flux_err", "stack_err", "sigma")


@dataclass
class Spectrum:
    """One 1-D rest-frame spectrum plus optional per-pixel galaxy counts."""
    wave: np.ndarray
    flux: np.ndarray
    err: Optional[np.ndarray] = None
    nspec: Optional[np.ndarray] = None
    info: str = ""

    @property
    def sn(self) -> np.ndarray:
        """Per-pixel S/N. NaN where err is missing or non-positive."""
        if self.err is None:
            return np.full_like(self.flux, np.nan, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.asarray(self.flux, float) / np.asarray(self.err, float)
        out[~np.isfinite(out)] = np.nan
        return out


def _first_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def as_spectrum(obj, flux=None, err=None, *, method=None, radial_bin=0,
                cube=None, nspec=None, info="") -> Spectrum:
    """
    Coerce whatever you have into a Spectrum. Accepts, in order:

    1. ``as_spectrum(wave, flux, err)``            -- three arrays
    2. ``as_spectrum({"wave":..., "flux":...})``   -- a dict, any of the usual
       key spellings (rest_wave / wave_rest / wave / wavelength; flux / stack /
       spec; err / error / flux_err / stack_err)
    3. ``as_spectrum(stacks_result)``              -- build_stacks output, i.e.
       a dict carrying ``stacks[method]["flux"]``. `method` defaults to the
       first available; `radial_bin` picks the row if flux is 2-D.
    4. Already a Spectrum -- returned unchanged (with cube/nspec merged in).

    cube : optional (ngal, nwave) rest-frame per-galaxy array. Used ONLY to
           count how many galaxies are finite at each pixel -> Spectrum.nspec.
           This is the honest N-per-line number; pass it if you have it.
    nspec: pass the counts directly instead of the cube.
    """
    if isinstance(obj, Spectrum):
        sp = obj
    elif flux is None and isinstance(obj, (tuple, list)) and 2 <= len(obj) <= 3 \
            and all(np.ndim(a) >= 1 for a in obj):           # form 1b: packed
        obj, flux, *rest = obj
        err = rest[0] if rest else None
        sp = Spectrum(np.asarray(obj, float), np.asarray(flux, float),
                      None if err is None else np.asarray(err, float), info=info)
    elif flux is not None:                                   # form 1
        sp = Spectrum(np.asarray(obj, float), np.asarray(flux, float),
                      None if err is None else np.asarray(err, float),
                      info=info)
    elif isinstance(obj, dict):
        w = _first_key(obj, _WAVE_KEYS)
        if w is None:
            raise KeyError(
                f"no wavelength key found; looked for {_WAVE_KEYS}. "
                f"Got keys: {sorted(obj)}"
            )
        f = _first_key(obj, _FLUX_KEYS)
        e = _first_key(obj, _ERR_KEYS)
        used = info
        if f is None and "stacks" in obj:                    # form 3
            methods = list(obj["stacks"])
            if not methods:
                raise ValueError("stacks_result has an empty 'stacks' dict.")
            m = method or methods[0]
            if m not in obj["stacks"]:
                raise KeyError(f"method {m!r} not in stacks: {methods}")
            f = obj["stacks"][m].get("flux")
            e = obj["stacks"][m].get("err")
            used = info or f"method={m}"
        if f is None:
            raise KeyError(
                f"no flux key found; looked for {_FLUX_KEYS} and 'stacks'. "
                f"Got keys: {sorted(obj)}"
            )
        f = np.asarray(f, float)
        e = None if e is None else np.asarray(e, float)
        if f.ndim == 2:                                      # (nrad, nwave)
            used = f"{used}, radial_bin={radial_bin}".strip(", ")
            f = f[radial_bin]
            if e is not None and e.ndim == 2:
                e = e[radial_bin]
        sp = Spectrum(np.asarray(w, float), f, e, info=used)
        if nspec is None:
            nspec = _first_key(obj, ("nspec", "ngal_pix", "n_pix", "counts"))
        if cube is None:
            cube = _first_key(obj, ("cube_flux", "cube"))
    else:
        raise TypeError(
            "pass (wave, flux[, err]), a dict, or a stacks_result -- got "
            f"{type(obj).__name__}."
        )

    # fail loudly on a length mismatch rather than letting numpy broadcast it
    # into something plausible-looking three functions later
    if sp.flux.shape != sp.wave.shape:
        raise ValueError(
            f"flux {sp.flux.shape} does not match wave {sp.wave.shape}. "
            "Most likely the stack and the rest grid came from different "
            "configs (check rest_wave_min/max/delta)."
        )
    if sp.err is not None and sp.err.shape != sp.wave.shape:
        raise ValueError(
            f"err {sp.err.shape} does not match wave {sp.wave.shape}."
        )

    if nspec is not None:
        sp.nspec = np.asarray(nspec, float)
    elif cube is not None:
        c = np.asarray(cube, float)
        if c.ndim == 3:                                      # (ngal,nrad,nwave)
            c = c[:, radial_bin, :]
        sp.nspec = np.isfinite(c).sum(axis=0).astype(float)
    if sp.nspec is not None and sp.nspec.shape != sp.wave.shape:
        raise ValueError(
            f"nspec shape {sp.nspec.shape} does not match wave "
            f"{sp.wave.shape} -- is the cube on the same rest grid?"
        )
    return sp


# =====================================================================
# 2.  INVENTORY -- what is actually in range, and is it usable?
# =====================================================================
def add_line(key, name, wave, kind="nebular", components=(), note="",
             expect="", catalog=None, overwrite=False) -> Line:
    """
    Register a line at runtime -- the intended way to extend the list without
    editing this file.

        import wave_systematics as ws
        ws.add_line("FeII2600", "Fe II", 2600.17, "resonant")
        ws.add_line("MyBlend", "X", 2000.0, "nebular",
                    components=(1998.5, 2001.5), note="testing")

    Wavelengths must be VACUUM rest-frame Angstrom. `kind` must be one of
    _KIND_EXPECT ('nebular', 'fine_structure', 'resonant', 'stellar') unless
    you also pass an explicit `expect` string.

    catalog : dict to write into. None -> the module-level LINES.
    """
    cat = LINES if catalog is None else catalog
    if key in cat and not overwrite:
        raise KeyError(f"{key!r} already exists; pass overwrite=True to replace.")
    if kind not in _KIND_EXPECT and not expect:
        raise ValueError(
            f"unknown kind {kind!r}; use one of {sorted(_KIND_EXPECT)} or pass "
            "an explicit expect= string."
        )
    cat[key] = _L(name, float(wave), kind, components, note, expect)
    return cat[key]


def print_line_catalog(catalog=None, *, kinds=None, wave_range=None):
    """
    Dump the full line list -- no spectrum required. This is the reference
    table: what is defined, where, which way it points, and what it is blended
    with. `wave_range=(lo, hi)` filters by rest wavelength.
    """
    cat = LINES if catalog is None else catalog
    rows = sorted(cat.items(), key=lambda kv: kv[1].wave)
    hdr = (f"  {'key':<12}{'line':<13}{'lam_rest':>10}{'kind':>15}"
           f"{'expect':>12}{'split':>8}  components")
    print(f"\n  LINE CATALOG -- {len(cat)} entries "
          "(vacuum rest-frame Angstrom)\n")
    print(hdr)
    print("  " + "-" * (len(hdr) + 22))
    for key, ln in rows:
        if kinds is not None and ln.kind not in kinds:
            continue
        if wave_range is not None and not (wave_range[0] <= ln.wave <= wave_range[1]):
            continue
        sk = ln.split_kms
        sp = "-" if sk is None else f"{sk:.0f}"
        comp = ", ".join(f"{c:.2f}" for c in ln.components)
        star = "*" if ln.kind in CALIBRATOR_KINDS else " "
        print(f" {star}{key:<12}{ln.name:<13}{ln.wave:>10.2f}{ln.kind:>15}"
              f"{ln.sense:>12}{sp:>8}  {comp}")
    print("\n  * = systemic, can anchor a velocity.  split = component "
          "separation [km/s].")
    print("  Add your own with ws.add_line(key, name, wave, kind, "
          "components=..., note=...)\n")
    return rows


def available_lines(spec_or_wave, *, kinds=None, pad=0.0, require_all=False,
                    names=None, catalog=None) -> list:
    """
    Which `LINES` entries fall inside the spectrum's wavelength range.

    kinds       : restrict to these `kind` values (e.g. ("nebular",)). None=all.
    names       : restrict to these LINES keys. None = all.
    pad         : require this much margin (A) inside each edge -- use it to
                  demand room for continuum sidebands, not just the line.
    require_all : if True a line only counts when EVERY component is in range
                  (default: the zoom centre being in range is enough).

    Returns Line objects, sorted by wavelength.
    """
    wave = (spec_or_wave.wave if isinstance(spec_or_wave, Spectrum)
            else np.asarray(as_spectrum(spec_or_wave).wave
                            if isinstance(spec_or_wave, dict)
                            else spec_or_wave, float))
    finite = wave[np.isfinite(wave)]
    if finite.size == 0:
        return []
    lo, hi = finite.min() + pad, finite.max() - pad

    out = []
    for key, ln in (LINES if catalog is None else catalog).items():
        if names is not None and key not in names:
            continue
        if kinds is not None and ln.kind not in kinds:
            continue
        probe = ln.components if require_all else (ln.wave,)
        if all(lo <= w <= hi for w in probe):
            out.append(ln)
    return sorted(out, key=lambda l: l.wave)


def line_table(obj, flux=None, err=None, *, kinds=None, names=None,
               halfwidth_kms=1500.0, pad=0.0, verbose=True, catalog=None,
               **kw) -> list:
    """
    Print (and return) the inventory: every line in range, with a first look at
    whether there is anything there.

    For each line, inside +/- `halfwidth_kms` of its centre:
        Npix   pixels in the window
        Ngal   median galaxies contributing (needs cube=/nspec=; else '-')
        pkSN   peak |per-pixel S/N| -- a CRUDE look, not a detection
        medSN  median per-pixel S/N

    pkSN IS NOT A DETECTION SIGNIFICANCE. It is the single most extreme pixel
    in the window, so it is biased high and rises with window width. Use it to
    triage which lines are worth a real measurement, nothing more. This module
    deliberately does not fit anything.
    """
    sp = as_spectrum(obj, flux, err, **kw)
    lines = available_lines(sp, kinds=kinds, names=names, pad=pad,
                            catalog=catalog)

    rows = []
    for ln in lines:
        m = _window_mask(sp.wave, ln.wave, halfwidth_kms)
        sn = sp.sn[m]
        sn = sn[np.isfinite(sn)]
        ng = (np.nanmedian(sp.nspec[m]) if sp.nspec is not None
              and np.any(m) else np.nan)
        rows.append(dict(
            line=ln,
            npix=int(np.sum(m)),
            ngal=ng,
            peak_sn=float(np.nanmax(np.abs(sn))) if sn.size else np.nan,
            med_sn=float(np.nanmedian(sn)) if sn.size else np.nan,
        ))

    if verbose:
        w = sp.wave[np.isfinite(sp.wave)]
        print(f"\nrest coverage : {w.min():.1f} - {w.max():.1f} A"
              + (f"   [{sp.info}]" if sp.info else ""))
        if sp.nspec is not None:
            print(f"galaxies      : {np.nanmax(sp.nspec):.0f} peak, "
                  f"{np.nanmedian(sp.nspec):.0f} median across the grid")
        print(f"window        : +/-{halfwidth_kms:.0f} km/s about each centre")
        print("\n  peak/med S/N are per-PIXEL and NOT detection significances.")
        print("  Triage only. No fitting happens in this module.\n")
        hdr = (f"  {'line':<12}{'lam_rest':>10}{'kind':>15}{'expect':>12}"
               f"{'split':>9}{'Npix':>6}{'Ngal':>6}{'pkSN':>7}{'medSN':>7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            ln = r["line"]
            star = "*" if ln.kind in CALIBRATOR_KINDS else " "
            ng = "-" if not np.isfinite(r["ngal"]) else f"{r['ngal']:.0f}"
            sk = ln.split_kms
            sp = "-" if sk is None else f"{sk:.0f}"
            print(f" {star}{ln.name:<12}{ln.wave:>10.2f}{ln.kind:>15}"
                  f"{ln.sense:>12}{sp:>9}{r['npix']:>6}{ng:>6}"
                  f"{r['peak_sn']:>7.1f}{r['med_sn']:>7.1f}")
        print("\n  * = can anchor a systemic velocity. Everything else is "
              "context, not calibration.")
        print("  expect = which way the feature should point. ABSORPTION lines "
              "point DOWN;\n           do not read them as failed emission.")
        print("  split  = component separation in km/s. REDSHIFT-INDEPENDENT: "
              "recovering it\n           confirms the feature is real and the "
              "dispersion scale is right, before\n           any velocity "
              "offset means anything.\n")
    return rows


# =====================================================================
# 3.  PLOTS
# =====================================================================
def _window_mask(wave, centre, halfwidth_kms):
    hw_A = centre * halfwidth_kms / C_KMS
    return np.isfinite(wave) & (np.abs(wave - centre) <= hw_A)


def _to_velocity(wave, centre):
    return (np.asarray(wave, float) - centre) / centre * C_KMS


def _local_continuum(sp, centre, halfwidth_kms, gap_kms, order=1):
    """
    Straight line fit to two sidebands flanking the line, evaluated across the
    window. Sidebands run from `gap_kms` to `halfwidth_kms` on each side, so
    the line itself is excluded by construction.

    Returns (model_on_window_mask, window_mask) or (None, mask) if there are
    too few sideband pixels to fit.
    """
    m = _window_mask(sp.wave, centre, halfwidth_kms)
    v = _to_velocity(sp.wave, centre)
    side = m & (np.abs(v) >= gap_kms) & np.isfinite(sp.flux)
    if side.sum() < order + 2:
        return None, m
    coef = np.polyfit(sp.wave[side] - centre, sp.flux[side], order)
    return np.polyval(coef, sp.wave[m] - centre), m


def plot_overview(obj, flux=None, err=None, *, kinds=None, names=None,
                  smooth=0, show_err=True, show_nspec=True, ax=None,
                  figsize=(15, 5), label_lines=True, ylim_percentiles=(1, 99),
                  catalog=None, **kw):
    """
    The whole rest-frame stack, with every in-range line marked.

    smooth       : boxcar width in pixels for a guide-to-the-eye overlay. The
                   UNSMOOTHED spectrum is always drawn underneath -- smoothing
                   here is cosmetic and never what you measure.
    show_nspec   : draw the per-pixel galaxy count on a twin axis (needs cube=
                   or nspec=). This is where you see coverage falling off the
                   ends of the grid.
    kinds/names  : restrict which lines get marked (e.g. kinds=("nebular",)).

    Returns (fig, ax).
    """
    sp = as_spectrum(obj, flux, err, **kw)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    good = np.isfinite(sp.flux)
    if show_err and sp.err is not None:
        ax.fill_between(sp.wave, sp.flux - sp.err, sp.flux + sp.err,
                        color="0.75", lw=0, zorder=1, label="+/-1 sigma")
    ax.plot(sp.wave, sp.flux, color="0.25", lw=0.7, zorder=2, label="stack")
    if smooth and smooth > 1:
        k = np.ones(int(smooth)) / int(smooth)
        f = np.where(good, sp.flux, 0.0)
        n = np.convolve(good.astype(float), k, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.convolve(f, k, mode="same") / n
        ax.plot(sp.wave, sm, color="tab:blue", lw=1.4, zorder=3,
                label=f"boxcar {int(smooth)} px (cosmetic)")
    ax.axhline(0, color="k", lw=0.6, ls=":", zorder=0)

    lines = available_lines(sp, kinds=kinds, names=names, catalog=catalog)
    seen = set()
    for ln in lines:
        st = _KIND_STYLE[ln.kind]
        lab = st["label"] if ln.kind not in seen else None
        seen.add(ln.kind)
        for w in ln.components:
            ax.axvline(w, color=st["color"], ls=st["ls"], lw=1.0, alpha=0.75,
                       zorder=4, label=lab)
            lab = None
        if label_lines:
            ax.annotate(ln.name, xy=(ln.wave, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7,
                        rotation=90, color=st["color"])

    if np.any(good):
        y = sp.flux[good]
        lo, hi = np.nanpercentile(y, ylim_percentiles)
        pad = 0.35 * (hi - lo if hi > lo else abs(hi) or 1.0)
        ax.set_ylim(min(lo - pad, 0), hi + pad)
    ax.set_xlim(np.nanmin(sp.wave), np.nanmax(sp.wave))
    ax.set_xlabel("rest wavelength [A]")
    ax.set_ylabel("stacked flux")
    ax.set_title("rest-UV stack" + (f"   [{sp.info}]" if sp.info else ""))
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    if show_nspec and sp.nspec is not None:
        ax2 = ax.twinx()
        ax2.plot(sp.wave, sp.nspec, color="tab:cyan", lw=1.0, alpha=0.7)
        ax2.set_ylabel("N galaxies", color="tab:cyan")
        ax2.tick_params(axis="y", colors="tab:cyan")
        ax2.set_ylim(0, np.nanmax(sp.nspec) * 1.6)
    fig.tight_layout()
    return fig, ax


def plot_line_zooms(obj, flux=None, err=None, *, lines=None, kinds=None,
                    names=None, halfwidth_kms=2000.0, velocity=True,
                    continuum=True, cont_gap_kms=800.0, subtract_continuum=False,
                    ncols=3, panel_size=(4.2, 3.0), show_nspec=True,
                    sharey=False, catalog=None, **kw):
    """
    One zoom panel per line. This is the "do we have signal?" figure.

    lines        : explicit list of Line objects or LINES keys. None -> every
                   in-range line passing `kinds`/`names`.
    halfwidth_kms: panel half-width. 2000 km/s is wide enough to see the local
                   continuum and any neighbouring feature; drop to ~600 once
                   you know a line is there.
    velocity     : x-axis in km/s relative to the line's rest wavelength
                   (default). Zero is systemic, so an offset is readable by
                   eye -- which is the entire point of this exercise. Set
                   False for a wavelength axis.
    continuum    : overlay a straight-line fit to the two sidebands running
                   from cont_gap_kms out to halfwidth_kms.
    subtract_continuum : plot flux minus that model instead. Easier to judge a
                   faint line; harder to spot a continuum problem. Look at both.

    NOTE ON DOUBLETS. Every component is marked. For C III] the panel centre is
    the 1906.68/1908.73 midpoint, which is NOT a physical wavelength and NOT
    where the line should peak -- the blend centroid depends on the density-
    sensitive ratio (spec section 4.5). Do not read a velocity off that panel.

    Returns (fig, axes).
    """
    sp = as_spectrum(obj, flux, err, **kw)

    if lines is None:
        sel = available_lines(sp, kinds=kinds, names=names, catalog=catalog)
    else:
        _cat = LINES if catalog is None else catalog
        sel = [_cat[l] if isinstance(l, str) else l for l in lines]
        sel = [l for l in sel if _window_mask(sp.wave, l.wave, halfwidth_kms).any()]
    if not sel:
        raise ValueError(
            "no lines in range. Check the stack's rest coverage against "
            "LINES -- if rest_wave_max is 1700 you have cut off Si III] and "
            "C III] (see spec section 4 and the config review)."
        )

    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharey=sharey,
                             figsize=(panel_size[0] * ncols,
                                      panel_size[1] * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, ln in zip(axes, sel):
        cont, m = (_local_continuum(sp, ln.wave, halfwidth_kms, cont_gap_kms)
                   if continuum or subtract_continuum else (None, None))
        if m is None:
            m = _window_mask(sp.wave, ln.wave, halfwidth_kms)
        x = _to_velocity(sp.wave[m], ln.wave) if velocity else sp.wave[m]
        y = sp.flux[m]
        e = sp.err[m] if sp.err is not None else None
        if subtract_continuum and cont is not None:
            y = y - cont

        if e is not None:
            ax.fill_between(x, y - e, y + e, color="0.8", lw=0, zorder=1)
        ax.step(x, y, where="mid", color="0.15", lw=0.9, zorder=2)
        if cont is not None and not subtract_continuum:
            ax.plot(x, cont, color="tab:blue", lw=1.0, ls="--", zorder=3,
                    label="sideband continuum")
        ax.axhline(0, color="k", lw=0.6, ls=":", zorder=0)

        st = _KIND_STYLE[ln.kind]
        for w in ln.components:
            xv = _to_velocity(w, ln.wave) if velocity else w
            ax.axvline(xv, color=st["color"], ls=st["ls"], lw=1.1, zorder=4)
        if velocity:
            for g in (-cont_gap_kms, cont_gap_kms):
                ax.axvline(g, color="tab:blue", lw=0.6, ls=":", alpha=0.6)

        ttl = f"{ln.name} {ln.wave:.2f}"
        if len(ln.components) > 1:
            ttl += f"  ({len(ln.components)} comp)"
        ng = (np.nanmedian(sp.nspec[m]) if show_nspec and sp.nspec is not None
              else np.nan)
        sub = ln.kind + (f"   N={ng:.0f}" if np.isfinite(ng) else "")
        ax.set_title(ttl, fontsize=10, color=st["color"])
        ax.text(0.02, 0.95, sub, transform=ax.transAxes, fontsize=7,
                va="top", ha="left", color="0.4")
        ax.set_xlabel("v [km/s] (0 = systemic)" if velocity else "rest lambda [A]")
        ax.set_ylabel("flux" + (" - cont" if subtract_continuum else ""))

    for ax in axes[len(sel):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axes


def quicklook(obj, flux=None, err=None, *, kinds=CALIBRATOR_KINDS,
              halfwidth_kms=2000.0, smooth=5, **kw):
    """
    Inventory + overview + zooms in one call. Defaults to the CALIBRATOR lines
    only, since that is the triage question. Pass kinds=None to see everything.

    Returns (rows, fig_overview, fig_zooms).
    """
    sp = as_spectrum(obj, flux, err, **kw)
    rows = line_table(sp, kinds=kinds, halfwidth_kms=halfwidth_kms)
    f1, _ = plot_overview(sp, kinds=None, smooth=smooth)
    f2, _ = plot_line_zooms(sp, kinds=kinds, halfwidth_kms=halfwidth_kms)
    return rows, f1, f2


# =====================================================================
# 4.  MEASURE -- see the module docstring for why the fit is built this way
# =====================================================================
#: default Gaussian sigma, km/s. VIRUS LSF ~5.6 A FWHM; at ~4800 A that is
#: sigma ~149 km/s, and the LSF dominates over the intrinsic width of an
#: unresolved nebular line. Override per call, or recompute with lsf_sigma_kms.
DEFAULT_SIGMA_KMS = 149.0


def lsf_sigma_kms(fwhm_obs_A=5.6, lam_obs_A=4800.0):
    """
    Instrumental sigma in km/s from the observed-frame LSF FWHM.

    Velocity width is frame-independent, so this is also the rest-frame sigma
    -- no (1+z) anywhere. Substitute your measured LSF (speclsf.py) rather than
    trusting the 5.6 A default.
    """
    return (fwhm_obs_A / 2.3548) / lam_obs_A * C_KMS


def _resolve(line, catalog=None):
    """Line object from a Line, a catalog key, or a bare wavelength."""
    if isinstance(line, Line):
        return line
    if isinstance(line, str):
        cat = LINES if catalog is None else catalog
        if line not in cat:
            raise KeyError(f"{line!r} not in the catalog. Known: {sorted(cat)}")
        return cat[line]
    return _L(f"lam{float(line):.1f}", float(line), "nebular")


def _design(wave, comps, v_kms, sigma_kms):
    """(npix, ncomp) matrix of unit-amplitude Gaussians at velocity v."""
    out = np.empty((wave.size, len(comps)))
    for i, lam0 in enumerate(comps):
        mu = lam0 * (1.0 + v_kms / C_KMS)
        sig = sigma_kms / C_KMS * lam0
        out[:, i] = np.exp(-0.5 * ((wave - mu) / sig) ** 2)
    return out


def _group_model(wave, ref, comps, v, sigma, amps, cont):
    m = _design(wave, comps, v, sigma) @ np.asarray(amps)
    return m + np.polyval(np.asarray(cont)[::-1], wave - ref)


def fit_lines_joint(obj, lines=None, flux=None, err=None, *, catalog=None,
                    v_guess=0.0, v_fixed=None, v_bound_kms=1200.0,
                    sigma_kms=DEFAULT_SIGMA_KMS, sigma_free=False,
                    sigma_bounds=None,
                    halfwidth_kms=2000.0, cont_order=1, kinds=CALIBRATOR_KINDS,
                    amp_sign=0, verbose=True, **kw) -> dict:
    """
    Fit several line groups with ONE SHARED VELOCITY. This is the measurement.

    Each group keeps its own free amplitudes (one per component) and its own
    local polynomial continuum; only `v` (and optionally `sigma`) are shared.
    See the module docstring for why.

    lines        : list of catalog keys / Line objects / bare wavelengths.
                   None -> every in-range line of `kinds`.
    v_guess      : starting velocity, km/s.
    v_fixed      : freeze v here instead of fitting it. Use this to SEE the
                   model at a chosen velocity (plot_fit's v_try uses it).
    v_bound_kms  : hard bound on |v| so the fit cannot wander out of the window.
    sigma_kms    : Gaussian sigma, km/s. Fixed unless sigma_free=True. This
                   doubles as the LSF value, so pass the real one from
                   lsf_sigma_kms() rather than trusting DEFAULT_SIGMA_KMS.
    sigma_bounds : (lo, hi) km/s when sigma_free=True. None (default) ->
                   (sigma_kms, 800), i.e. THE LOWER BOUND IS THE LSF. A line
                   cannot be narrower than the instrument that measured it;
                   allowing it to be is how a free width collapses onto one
                   pixel and blows up that group's amplitude. Only override
                   the floor if you have a measured LSF you trust less than
                   the data.
    cont_order   : per-group continuum polynomial order. 1 (a line) is the
                   default on purpose -- a higher order at low S/N will start
                   absorbing the line itself.
    amp_sign     : 0 = amplitudes free (default; correct for a mixed set).
                   +1 = force emission, -1 = force absorption. Only use these
                   if you already know what the line does -- forcing the sign
                   guarantees a "detection" of the right sign out of pure noise.

    Returns a dict: v_kms, v_err_kms, sigma_kms, chi2, ndof, chi2_red, groups
    (per-line amplitudes / continuum / model / mask), plus `success` and any
    `warnings` raised along the way.
    """
    from scipy.optimize import least_squares

    sp = as_spectrum(obj, flux, err, **kw)
    if lines is None:
        sel = available_lines(sp, kinds=kinds, catalog=catalog)
    else:
        sel = [_resolve(l, catalog) for l in np.atleast_1d(lines)]
    if not sel:
        raise ValueError("no lines to fit (none in range, or none matched).")

    # ---- per-group window + weights -----------------------------------
    groups, warn = [], []
    for ln in sel:
        m = _window_mask(sp.wave, ln.wave, halfwidth_kms) & np.isfinite(sp.flux)
        if sp.err is not None:
            m &= np.isfinite(sp.err) & (sp.err > 0)
        npar = len(ln.components) + cont_order + 1
        if m.sum() <= npar:
            warn.append(f"{ln.name} {ln.wave:.1f}: only {int(m.sum())} usable "
                        f"pixels for {npar} parameters -- dropped.")
            continue
        w = 1.0 / sp.err[m] if sp.err is not None else np.ones(int(m.sum()))
        groups.append(dict(line=ln, mask=m, w=w, ref=ln.wave,
                           ncomp=len(ln.components)))
    if not groups:
        raise ValueError("every group was dropped:\n  " + "\n  ".join(warn))

    # ---- WIDTH BOUNDS: THE FLOOR IS THE LSF, NOT AN ARBITRARY SMALL NUMBER
    # Convolution adds in quadrature, sigma_obs = sqrt(sig_int^2 + sig_LSF^2),
    # so sigma_obs >= sigma_LSF exactly. The old default floor of 40 km/s is
    # well below the ~140-170 km/s LSF sigma and let a free width collapse onto
    # a single deep/bright pixel -- which then inflates that group's amplitude
    # to preserve the equivalent width, while other groups stay near zero.
    # That is the "amplitudes huge here, tiny there" failure mode, and it is a
    # BOUNDS problem, not an amplitude problem.
    if sigma_bounds is None:
        sigma_bounds = (float(sigma_kms), 800.0)
    else:
        sigma_bounds = (float(sigma_bounds[0]), float(sigma_bounds[1]))
        if sigma_free and sigma_bounds[0] < 0.999 * float(sigma_kms):
            warn.append(
                f"sigma lower bound {sigma_bounds[0]:.0f} km/s is BELOW the LSF "
                f"({sigma_kms:.0f} km/s). A line cannot be narrower than the "
                f"instrument; a fit that lands there is unphysical.")
    if sigma_free and not (sigma_bounds[0] <= sigma_kms <= sigma_bounds[1]):
        sigma_kms = float(np.clip(sigma_kms, *sigma_bounds))

    fit_v = v_fixed is None
    n_shared = int(fit_v) + int(sigma_free)

    def _unpack(p):
        i = 0
        v = p[i] if fit_v else float(v_fixed);  i += int(fit_v)
        s = p[i] if sigma_free else float(sigma_kms);  i += int(sigma_free)
        out = []
        for g in groups:
            a = p[i:i + g["ncomp"]];        i += g["ncomp"]
            c = p[i:i + cont_order + 1];    i += cont_order + 1
            out.append((a, c))
        return v, s, out

    def _resid(p):
        v, s, per = _unpack(p)
        chunks = []
        for g, (a, c) in zip(groups, per):
            lam = sp.wave[g["mask"]]
            mod = _group_model(lam, g["ref"], g["line"].components, v, s, a, c)
            chunks.append((sp.flux[g["mask"]] - mod) * g["w"])
        return np.concatenate(chunks)

    # ---- initial guess: continuum from a median, amplitude from the peak
    p0, lo, hi = [], [], []
    if fit_v:
        p0.append(float(v_guess)); lo.append(-v_bound_kms); hi.append(v_bound_kms)
    if sigma_free:
        p0.append(float(sigma_kms)); lo.append(sigma_bounds[0]); hi.append(sigma_bounds[1])
    for g in groups:
        ln = g["line"]
        f = sp.flux[g["mask"]]
        c0 = float(np.nanmedian(f))
        abound = {0: (-np.inf, np.inf), 1: (0.0, np.inf), -1: (-np.inf, 0.0)}[amp_sign]

        # ---- STARTING AMPLITUDE: SIGN MATTERS, AND SO DOES A=0 -------------
        # The old guess was nanmax(flux) - median, i.e. ALWAYS POSITIVE. For a
        # resonant LIS line -- which is interstellar ABSORPTION against the
        # continuum and points DOWN -- that starts the solver on the wrong side
        # of zero. That is not merely a slow start: the model is
        #     m = A * exp(-((lam - mu(v))/sigma)^2 / 2)
        # so d m / d v is PROPORTIONAL TO A. At A = 0 the velocity column of the
        # Jacobian vanishes identically and v is completely unconstrained. An
        # absorption line started at A > 0 has to travel through that degenerate
        # point, and least_squares frequently stalls near it -- which is exactly
        # the "amplitude came out weirdly small and the velocity is arbitrary"
        # failure. Start on the correct side instead.
        #
        # It was also taken over the FULL +/-halfwidth window, so for a marginal
        # line it measured the largest noise excursion in +/-2000 km/s rather
        # than the line. Measure in a narrow core around the centre instead.
        v_win = _to_velocity(sp.wave[g["mask"]], ln.wave)
        core = np.abs(v_win) <= max(2.0 * sigma_kms, 150.0)
        fc = f[core] if core.sum() >= 3 else f
        if amp_sign > 0:
            sense = +1
        elif amp_sign < 0:
            sense = -1
        else:                                   # infer from the line's kind
            sense = -1 if ln.sense == "ABSORPTION" else +1
        amp0 = float((np.nanmax(fc) if sense > 0 else np.nanmin(fc)) - c0)
        scale = float(np.nanstd(f)) or abs(c0) or 1.0
        if not np.isfinite(amp0) or abs(amp0) < 1e-3 * scale:
            amp0 = sense * scale             # never hand the solver A ~ 0
        amp0 = float(np.clip(amp0, abound[0], abound[1]))

        p0 += [amp0] * g["ncomp"] + [c0] + [0.0] * cont_order
        lo += [abound[0]] * g["ncomp"] + [-np.inf] * (cont_order + 1)
        hi += [abound[1]] * g["ncomp"] + [np.inf] * (cont_order + 1)

    res = least_squares(_resid, np.asarray(p0, float),
                        bounds=(np.asarray(lo), np.asarray(hi)))
    v, s, per = _unpack(res.x)

    # ---- did anything rail? (relative tolerance -- a bounded trust-region
    # solver parks ~1e-8 from a bound, so an absolute test silently misses it)
    def _at(val, bound, scale):
        return np.isfinite(bound) and abs(val - bound) <= 1e-4 * max(abs(scale), 1e-12)

    if fit_v and (_at(v, -v_bound_kms, v_bound_kms) or _at(v, v_bound_kms, v_bound_kms)):
        warn.append(f"v railed at |v| = {v_bound_kms:.0f} km/s -- the fit wants "
                    f"to leave the window. The velocity is meaningless.")
    if sigma_free and _at(s, sigma_bounds[0], sigma_bounds[0]):
        warn.append(f"sigma railed at the lower bound ({sigma_bounds[0]:.0f} "
                    f"km/s). The fit wants a line NARROWER than allowed; if "
                    f"that bound is the LSF the model is wrong somewhere else "
                    f"(usually the continuum eating the line).")
    if sigma_free and _at(s, sigma_bounds[1], sigma_bounds[1]):
        warn.append(f"sigma railed at the upper bound ({sigma_bounds[1]:.0f} "
                    f"km/s) -- the Gaussian is broadening to swallow the "
                    f"continuum, and v is dragged with it.")

    # ---- covariance -> v error (a LOWER BOUND; see module docstring) ----
    npts = sum(int(g["mask"].sum()) for g in groups)
    ndof = max(npts - res.x.size, 1)
    chi2 = float(2 * res.cost)
    v_err = np.nan
    if fit_v:
        try:
            _, sv, VT = np.linalg.svd(res.jac, full_matrices=False)
            sv = sv[sv > 1e-12 * sv[0]]
            cov = (VT[:sv.size].T / sv ** 2) @ VT[:sv.size]
            v_err = float(np.sqrt(abs(cov[0, 0])))
            if sp.err is None:                      # no errors -> rescale by rms
                v_err *= np.sqrt(chi2 / ndof)
        except Exception as e:                       # pragma: no cover
            warn.append(f"covariance failed ({e}); v_err is NaN.")

    out = dict(v_kms=v, v_err_kms=v_err, sigma_kms=s, sigma_free=sigma_free,
               chi2=chi2, ndof=ndof, chi2_red=chi2 / ndof, npts=npts,
               success=bool(res.success), warnings=warn, spec=sp,
               v_fixed=v_fixed, halfwidth_kms=halfwidth_kms,
               cont_order=cont_order, groups=[])
    for g, (a, c) in zip(groups, per):
        lam = sp.wave[g["mask"]]
        out["groups"].append(dict(
            line=g["line"], mask=g["mask"], amps=np.asarray(a),
            cont=np.asarray(c), ref=g["ref"],
            model=_group_model(lam, g["ref"], g["line"].components, v, s, a, c),
            cont_only=np.polyval(np.asarray(c)[::-1], lam - g["ref"]),
        ))

    if verbose:
        names = ", ".join(f"{g['line'].name} {g['line'].wave:.1f}"
                          for g in out["groups"])
        print(f"\n  joint fit: {names}")
        print(f"  v      = {v:+.1f} +/- {v_err:.1f} km/s"
              + ("   [FIXED]" if not fit_v else "")
              + "        <- covariance error, a LOWER BOUND")
        print(f"  sigma  = {s:.1f} km/s" + ("  (free)" if sigma_free else "  (fixed)"))
        print(f"  chi2/n = {chi2/ndof:.2f}   ({npts} px, {res.x.size} par)")
        for g in out["groups"]:
            amp = g["amps"]
            print(f"    {g['line'].name:<11}{g['line'].wave:>9.2f}   "
                  + "  ".join(f"A{i+1}={a:+.3g}" for i, a in enumerate(amp)))
        for w_ in warn:
            print(f"  ! {w_}")
        print("  Look at scan_velocity() before quoting this.\n")
    return out


def fit_line(obj, line, flux=None, err=None, **kw) -> dict:
    """One line group. Thin wrapper on fit_lines_joint -- same arguments."""
    return fit_lines_joint(obj, [line], flux, err, **kw)


def scan_velocity(obj, lines=None, flux=None, err=None, *,
                  v_range=(-900.0, 900.0), n=121, plot=True, ax=None,
                  **kw) -> dict:
    """
    Profile chi2 as a function of the shared velocity: refit everything else at
    each fixed v. THE HONEST LOW-S/N DIAGNOSTIC.

    What to look for:
      clean single parabola  -> the velocity is measured. Delta-chi2 = 1 gives
                                the 1-sigma interval directly, and it is more
                                trustworthy than the covariance error.
      flat bottom            -> undetermined. Any quoted v is arbitrary.
      two minima             -> you are fitting a noise bump, OR a blend is
                                pulling the solution (check the neighbours
                                flagged in the line notes).

    Returns dict(v_grid, chi2, v_best, dchi2, sigma_interval).

    GRID RESOLUTION MATTERS. sigma_interval is read off the grid, so it cannot
    be narrower than one step. Default (-900, 900, n=121) is 15 km/s per step
    -- fine for "is there a minimum at all", too coarse to quote an interval
    from. Once you have located the minimum, re-scan narrow, e.g.
    v_range=(v_best-150, v_best+150), n=151 -> 2 km/s steps.
    """
    kw.pop("verbose", None)
    vg = np.linspace(v_range[0], v_range[1], int(n))
    chi2 = np.array([fit_lines_joint(obj, lines, flux, err, v_fixed=float(v),
                                     verbose=False, **kw)["chi2"] for v in vg])
    i = int(np.nanargmin(chi2))
    d = chi2 - chi2[i]
    below = vg[d <= 1.0]
    interval = (float(below.min()), float(below.max())) if below.size else (np.nan, np.nan)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 4))
        else:
            fig = ax.figure
        ax.plot(vg, d, color="0.2", lw=1.4)
        ax.axhline(1.0, color="tab:red", ls="--", lw=1,
                   label=r"$\Delta\chi^2=1$ (1$\sigma$)")
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.axvline(0, color="tab:green", lw=1, ls="-", alpha=.7, label="systemic")
        ax.axvline(vg[i], color="tab:blue", lw=1.2,
                   label=f"best {vg[i]:+.0f} km/s")
        if np.isfinite(interval[0]):
            ax.axvspan(*interval, color="tab:blue", alpha=.12)
        ax.set_xlabel("shared velocity v [km/s]")
        ax.set_ylabel(r"$\Delta\chi^2$")
        ax.set_ylim(-0.3, max(6.0, float(np.nanpercentile(d, 90))))
        ax.legend(fontsize=8)
        ax.set_title("velocity scan -- is the minimum actually a minimum?")
        fig.tight_layout()
    return dict(v_grid=vg, chi2=chi2, dchi2=d, v_best=float(vg[i]),
                sigma_interval=interval)


def plot_fit(obj, lines=None, fit=None, flux=None, err=None, *, v_try=None,
             ncols=None, panel_size=(4.6, 4.2), velocity=True, **kw):
    """
    SEE the fit. Per line: data + model + continuum on top, weighted residuals
    below. This is the visual confirmation step -- do not trust a number from
    fit_lines_joint you have not looked at here.

    fit   : an existing result from fit_lines_joint. None -> fit now with **kw.
    v_try : list of velocities (km/s) to overlay as FIXED-v models, e.g.
            v_try=[-100, 0, 125]. Everything else is refit at each one, so you
            are seeing the best the model can do at that velocity -- the way to
            judge by eye whether a shift is really preferred. Cheap and more
            informative than staring at one curve.

    Returns (fig, axes).
    """
    if fit is None:
        fit = fit_lines_joint(obj, lines, flux, err, verbose=False, **kw)
    sp = fit["spec"]
    gs = fit["groups"]
    ncols = ncols or min(3, len(gs))
    nrows = int(np.ceil(len(gs) / ncols))

    tries = []
    for vt in (v_try or []):
        kw2 = {k: v for k, v in kw.items() if k not in ("v_guess", "v_fixed")}
        try:
            tries.append((vt, fit_lines_joint(sp, lines, v_fixed=float(vt),
                                              verbose=False, **kw2)))
        except Exception:
            tries.append((vt, None))

    fig = plt.figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    gsp = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3, 1] * nrows,
                           hspace=0.06, wspace=0.28)
    axes = []
    for k, g in enumerate(gs):
        r, c = divmod(k, ncols)
        ax = fig.add_subplot(gsp[2 * r, c])
        axr = fig.add_subplot(gsp[2 * r + 1, c], sharex=ax)
        ln, m = g["line"], g["mask"]
        lam = sp.wave[m]
        x = _to_velocity(lam, ln.wave) if velocity else lam

        if sp.err is not None:
            ax.fill_between(x, sp.flux[m] - sp.err[m], sp.flux[m] + sp.err[m],
                            color="0.85", lw=0, zorder=1)
        ax.step(x, sp.flux[m], where="mid", color="0.15", lw=0.9, zorder=2,
                label="stack")
        ax.plot(x, g["cont_only"], color="tab:blue", ls="--", lw=1.0, zorder=3,
                label="continuum")
        ax.plot(x, g["model"], color="tab:red", lw=1.6, zorder=5,
                label=f"fit  v={fit['v_kms']:+.0f}")
        for j, (vt, ft) in enumerate(tries):
            if ft is None:
                continue
            gg = next((q for q in ft["groups"] if q["line"].wave == ln.wave), None)
            if gg is not None:
                ax.plot(x, gg["model"], lw=1.0, ls=":", alpha=.85, zorder=4,
                        color=plt.cm.viridis(j / max(len(tries) - 1, 1)),
                        label=f"v={vt:+.0f} fixed")
        st = _KIND_STYLE[ln.kind]
        for wc in ln.components:
            ax.axvline(_to_velocity(wc, ln.wave) if velocity else wc,
                       color=st["color"], ls=st["ls"], lw=1.0, alpha=.7, zorder=0)

        res = (sp.flux[m] - g["model"]) / (sp.err[m] if sp.err is not None else 1.0)
        axr.step(x, res, where="mid", color="0.3", lw=0.8)
        axr.axhline(0, color="k", lw=0.6)
        for lvl in (-1, 1):
            axr.axhline(lvl, color="tab:red", lw=0.6, ls=":")
        axr.set_ylabel("resid/σ" if sp.err is not None else "resid")
        axr.set_xlabel("v [km/s] (0 = systemic)" if velocity else "rest λ [A]")

        ax.set_title(f"{ln.name} {ln.wave:.2f}   [{ln.sense}]",
                     fontsize=10, color=st["color"])
        ax.tick_params(labelbottom=False)
        ax.set_ylabel("flux")
        if k == 0:
            ax.legend(fontsize=7, loc="best")
        axes += [ax, axr]

    ttl = (f"v = {fit['v_kms']:+.1f} +/- {fit['v_err_kms']:.1f} km/s   "
           f"(sigma {fit['sigma_kms']:.0f} "
           f"{'free' if fit['sigma_free'] else 'fixed'}, "
           f"chi2/n = {fit['chi2_red']:.2f})   "
           "-- v_err is a LOWER BOUND, see scan_velocity")
    fig.suptitle(ttl, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes
