"""
star_lines.py -- stellar line velocities per star -> the VIRUS wavelength offset.

TEST BENCH. Lives at code/ level next to wave_systematics.py, outside
utils_lya_halo/, so guide.check_guide() never sees it.

----------------------------------------------------------------------
THE MEASUREMENT
----------------------------------------------------------------------
Gaia gives each star a radial velocity to 1-5 km/s, so the reference velocity is
KNOWN. For each line:

    lam_expect = lam_rest * (1 + z_gaia)        where it SHOULD land
    eps        = c * (lam_meas - lam_expect) / lam_expect

eps is the instrument's error at that wavelength, in km/s. N stars give N
INDEPENDENT estimates, so nothing here ever stacks.

----------------------------------------------------------------------
READ THIS BEFORE TRUSTING ANY NUMBER
----------------------------------------------------------------------
ONE PIXEL IS 110-140 km/s. VIRUS dispersion is ~1.7-2 A; at 4340 A that is
138 km/s per pixel, at 5170 A it is 116. You are chasing an 85 km/s effect, so
EVERY estimate here -- fitted or eyeballed -- is sub-pixel. Two consequences:

  * A visual estimate is not obviously worse than a fit. Reading a trough centre
    to half a pixel is ~60 km/s, and a bad continuum moves an automated centroid
    by more than that. Hence the visual path below is a first-class citizen, not
    a fallback.
  * Nothing is measurable from one line of one star. The number comes from
    averaging lines within a star and stars within a field, and the SCATTER is
    the error bar. Do not quote a covariance error from a single fit.

AIR vs VACUUM is the systematic under test, not a detail. Ca II K is 3933.66 air
/ 3934.78 vac = 85.1 km/s, the exact size and sign of the anomaly. The catalog
stores AIR (universal for optical stellar lists) and converts via air_to_vac(),
so there is one formula and no hand-typed vacuum values. Run everything both
ways; the two answers MUST differ by ~85 km/s or the bookkeeping is broken.

THE SPECTRUM IS NEVER SHIFTED. Resampling to apply the RV correlates pixels and
smears the feature being centroided. The RV moves the LINE LIST instead.

COOL STARS HAVE NO CONTINUUM in the blue. Below ~4500 A a K star is a forest of
blended Fe lines; what the windows measure is a pseudo-continuum. This is the
dominant systematic on eps, which is why plot_line() SHADES the continuum
windows -- look at them before believing anything.

----------------------------------------------------------------------
WORKFLOW
----------------------------------------------------------------------
    import star_lines as sl

    prod = sl.load("outputs_starpsf/star_core_cosmos.fits")
    s = sl.get_star(prod, 0)          # everything about star i, one object
    print(s)                          # id, G, Teff, RV, S/N, coverage
    sl.plot_spectrum(s)               # full band, lines marked, grades coloured
    sl.suggest(s)                     # which lines are worth trying for THIS Teff

    sl.plot_line(s, "Mgb")            # 50 A window, Gaia at 0, windows shaded
    sl.plot_line(s, "Mgb", v=[-60, -85, -110])    # overlay trial velocities

    v  = {"CaK": -70, "Mgb": -95, "CaI4226": -80} # what you read off
    v2 = sl.snap_all(s, v)                        # refine to the trough minima
    sl.plot_lines(s, v=v2)                        # confirm all at once
    off = sl.visual_offset(s, v2)                 # -> eps for this star

    sl.fit_lines(s, ["CaK", "Mgb"])   # cross-check only, see below

PRIMARY ESTIMATOR: THE TROUGH MINIMUM (Austin, 2026-07-30). snap()/visual_offset
is the measurement path; fit_line() is a cross-check on the same lines. The two
are different statistics -- on an asymmetric trough the minimum sits redward of
the Gaussian centroid -- so never average them together. A large gap between
them on one line is a continuum-sensitivity warning worth quoting as a
systematic, which is the main thing the fit is still good for.

PUBLIC API
    air_to_vac / vac_to_air / rest_wave / z_from_rv
    LINES / line_spec / suggest
    GOOD_LINES / ALL_LINES / COOL / WARM / DRIFT / ANCHORS    <- named line sets
    load / load_stars / Star / get_star
    plot_spectrum / plot_line / plot_lines
    snap / snap_all / visual_offset / measure / measure_all   <- measurement path
    fit_line / fit_lines                                      <- cross-check
    combine / combine_stars / describe
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493

try:
    from utils_lya_halo.speclsf import instrumental_fwhm, VIRUS_FWHM_A
except Exception:                       # pragma: no cover
    VIRUS_FWHM_A = 5.6

    def instrumental_fwhm(wave_obs, mode="fwhm", fwhm_A=VIRUS_FWHM_A, **kw):
        return np.full_like(np.asarray(wave_obs, float), float(fwhm_A))

try:
    from scipy.optimize import curve_fit
except Exception:                       # pragma: no cover
    curve_fit = None


# =====================================================================
# 1.  AIR <-> VACUUM
# =====================================================================

def air_to_vac(lam_air):
    """Air -> vacuum, Morton (2000) ApJS 130, 403. Check: 3933.66 -> 3934.777."""
    lam_air = np.asarray(lam_air, float)
    s2 = (1e4 / lam_air) ** 2
    n = (1.0 + 8.336624212083e-5
         + 2.408926869968e-2 / (130.1065924522 - s2)
         + 1.599740894897e-4 / (38.92568793293 - s2))
    return lam_air * n


def vac_to_air(lam_vac):
    """Vacuum -> air, Ciddor (1996)."""
    lam_vac = np.asarray(lam_vac, float)
    s2 = (1e4 / lam_vac) ** 2
    n = (1.0 + 0.0000834254 + 0.02406147 / (130.0 - s2)
         + 0.00015998 / (38.9 - s2))
    return lam_vac / n


def rest_wave(lam_air, frame="air"):
    f = str(frame).lower()
    if f in ("air", "a"):
        return np.asarray(lam_air, float)
    if f in ("vac", "vacuum", "v"):
        return air_to_vac(lam_air)
    raise ValueError("frame must be 'air' or 'vac' -- state it explicitly")


def z_from_rv(rv_kms):
    b = np.asarray(rv_kms, float) / C_KMS
    return np.sqrt((1.0 + b) / (1.0 - b)) - 1.0


# =====================================================================
# 2.  LINE CATALOG
# =====================================================================

@dataclass(frozen=True)
class Component:
    name: str
    lam_air: float
    sign: int = -1


@dataclass(frozen=True)
class Line:
    key: str
    name: str
    lam_air: float
    sign: int = -1
    blue: tuple = ()                    # continuum window, rest A
    red: tuple = ()
    partners: tuple = ()
    ncomp: int = 1                      # components for an unbiased velocity
    teff: tuple = (0.0, 1e5)            # where this feature is actually strong
    grade: str = "B"                    # A trust / B usable / C diagnostic only
    note: str = ""
    visual_ok: bool = True              # False = fit-only. snap() has no ncomp
                                         # handling -- it's a bare argmin -- so a
                                         # blend anchor (ncomp>1) must not be
                                         # reachable via the trough-minimum path.

    @property
    def components(self):
        return (Component(self.name, self.lam_air, self.sign),) + tuple(self.partners)

    @property
    def span(self):
        return (float(self.blue[0]), float(self.red[1]))


#: Windows are WIDENED and pushed further from the line than v1, after the first
#: real fits showed the Mg b red window sitting on the Fe I blend and pinning the
#: continuum low. They are still chosen by eye, not from a published table --
#: plot_line() shades them so you can check each one against the data.
LINES: dict = {
    "CaK": Line(
        "CaK", "Ca II K", 3933.66, -1, (3895, 3912), (3947, 3958),
        grade="A", teff=(3000, 7000),
        note="Strongest feature in band for FGK, and the sharpest air/vac test "
             "(vac 3934.777, +85.1 km/s). Damping wings are wide in dwarfs, so "
             "sigma must stay free and the windows sit well clear of the core. "
             "Red window stops before Ca II H."),
    "CaH": Line(
        "CaH", "Ca II H + Heps", 3968.47, -1, (3947, 3958), (3990, 4005),
        partners=(Component("Heps", 3970.07, -1),), ncomp=2,
        grade="C", teff=(3000, 7000),
        note="Blended with H-epsilon 121 km/s away -- far inside the LSF, so a "
             "single Gaussian returns a blend centroid. Corroboration for CaK "
             "only, never an independent anchor."),
    "CaI4226": Line(
        "CaI4226", "Ca I 4226", 4226.73, -1, (4195, 4213), (4241, 4256),
        grade="A", teff=(3000, 5800),
        note="THE line for your coolest stars. Strong neutral-Ca resonance "
             "line, clean singlet, deep below ~5000 K, fading above ~5800 K. "
             "Promoted to grade A for cool targets -- it does the job the "
             "Balmer lines cannot."),
    "Gband": Line(
        "Gband", "G band (CH)", 4300.0, -1, (4255, 4272), (4335, 4350),
        grade="C", teff=(3500, 6000),
        note="A MOLECULAR BAND, not a line. No single rest wavelength; its "
             "apparent centroid moves with Teff and gravity. Confirms the star "
             "is cool and the extraction is sane. NEVER put it in a combine."),
    "Hd": Line(
        "Hd", "H-delta", 4101.74, -1, (4062, 4084), (4120, 4138),
        grade="B", teff=(5200, 12000),
        note="Teff floor raised from 4500 to 5200: at 4805 K the line is too "
             "shallow to centroid at VIRUS S/N, which is what the first run "
             "found. Only the 5859 K star qualifies."),
    "Hg": Line(
        "Hg", "H-gamma", 4340.47, -1, (4300, 4322), (4358, 4378),
        grade="B", teff=(5200, 12000),
        note="As Hd, and worse: the blue window brushes the G band in cool "
             "stars. First run returned depth 0.008 -- i.e. absent."),
    "FeI4384": Line(
        "FeI4384", "Fe I 4383 blend", 4383.55, -1, (4352, 4370), (4396, 4412),
        grade="B", teff=(3000, 6500),
        note="Strongest single Fe I feature in band, but a crowded blend of "
             "weak Fe lines at R~750. Fitted sigma will exceed the LSF "
             "substantially -- that is real, not a failure."),
    "Hb": Line(
        "Hb", "H-beta", 4861.33, -1, (4805, 4835), (4890, 4915),
        grade="B", teff=(5200, 12000),
        note="Cleanest Balmer line in band and the best red-end anchor IF the "
             "star is hot enough. At 3631-4805 K it is not there, and the first "
             "run duly railed on a 2-pixel noise dip."),
    "Mgb": Line(
        "Mgb", "Mg b (5183.60 component)", 5183.60, -1, (5100, 5135), (5220, 5245),
        grade="A", teff=(3000, 6500),
        note="THE VISUAL/SNAP ANCHOR for this feature (Austin, 2026-07-30, "
             "reanchored). The old single-key 'Mgb' sat on the 5172.68 "
             "component and was found via snap()'s bare argmin -- no ncomp "
             "handling -- which is exactly the blend-centroid trap the "
             "'ncomp=1 gives a Teff-dependent centroid' warning below is about. "
             "5172.68 sits only 0.76 LSF-widths from 5167.32 (fully blended) "
             "and 1.5 LSF-widths from 5183.60. This line, 5183.60, is the "
             "strongest of the three and the most isolated of the three "
             "pairings -- still not perfectly clean, so check the window "
             "shading in plot_line before trusting a given star. For the full "
             "3-component joint fit (the statistically correct treatment of "
             "the whole complex) use MgbBlend instead; snap() cannot do that "
             "fit, only fit_line() can."),
    "MgbBlend": Line(
        "MgbBlend", "Mg b triplet (joint fit only)", 5172.68, -1,
        (5100, 5135), (5220, 5245),
        partners=(Component("Mgb5167", 5167.32, -1),
                  Component("Mgb5184", 5183.60, -1)), ncomp=3,
        grade="A", teff=(3000, 6500), visual_ok=False,
        note="Deepest metal feature in band for cool stars (depth ~0.3 "
             "measured). Triplet spans 945 km/s ~ 2.3 LSF FWHM: use ncomp=3, "
             "ncomp=1 gives a Teff-dependent blend centroid. FIT-ONLY "
             "(visual_ok=False): fit_line() shares one velocity across all "
             "3 Gaussians correctly; snap()/visual_offset() have no ncomp "
             "concept and will silently reduce this to an ncomp=1 read on "
             "whichever component happens to sit deepest for this star's "
             "line-strength ratios. Use 'Mgb' (5183.60 alone) for the primary "
             "trough-minimum path; use this only via fit_lines()/fit_line() "
             "as the cross-check on the whole complex. WINDOWS MOVED OUTWARD "
             "from (5130,5150)/(5200,5215): the old red window sat on the "
             "Fe I blend and pinned the continuum ~10%% low, which is why the "
             "v1 model could not reach the blue wing."),
    "FeI5270": Line(
        "FeI5270", "Fe I 5270 blend", 5269.54, -1, (5235, 5252), (5290, 5310),
        grade="C", teff=(3000, 6500),
        note="Blended with Fe I 5266/5273 and Ca I 5265. Included only for the "
             "drift lever arm -- it is the reddest feature available."),
}

#: Default set. Metal lines, because the sample is cool. Balmer lines are NOT
#: here -- suggest() will add them if a star is hot enough.
GOOD_LINES = ("CaK", "CaI4226", "FeI4384", "Mgb")

#: Ready-made line sets. Pass any of these as lines=... to snap_all / measure /
#: plot_lines / fit_lines, or build your own list of keys.
ALL_LINES = tuple(LINES)                                   # everything, incl. grade C
COOL = ("CaK", "CaI4226", "FeI4384", "Mgb")                # Teff < 5200: metals only
WARM = ("CaK", "Hd", "Hg", "Hb", "Mgb")                    # Teff > 5200: Balmer usable
DRIFT = ("CaK", "CaI4226", "Mgb", "FeI5270")               # max lever arm 3934->5270 A
ANCHORS = ("CaK", "CaI4226", "Mgb")                        # grade A only


def _as_guesses(star, lines):
    """
    Normalize the many ways of naming lines into {key: starting_velocity}.

    Accepts:
        None                      -> suggest(star), all starting at 0
        "CaK" or a Line           -> that one line
        ["CaK", "Mgb"] / COOL     -> those lines, all starting at 0
        {"CaK": -70, "Mgb": -95}  -> explicit starting guesses (what you read off)
    """
    if lines is None:
        return {k: 0.0 for k in suggest(star, verbose=False)}
    if isinstance(lines, dict):
        return {(k.key if isinstance(k, Line) else k): float(v)
                for k, v in lines.items()}
    if isinstance(lines, (str, Line)):
        lines = [lines]
    return {(k.key if isinstance(k, Line) else k): 0.0 for k in lines}


def line_spec(line) -> Line:
    if isinstance(line, Line):
        return line
    try:
        return LINES[line]
    except KeyError:
        raise KeyError(f"unknown line {line!r}; known: {sorted(LINES)}") from None


# =====================================================================
# 3.  THE STAR OBJECT  --  one call, everything downstream takes it
# =====================================================================

@dataclass
class Star:
    """One star's spectrum plus its Gaia parameters. Everything else takes this."""
    wave: np.ndarray
    flux: np.ndarray
    err: Optional[np.ndarray]
    sid: str = ""
    rv: float = 0.0
    rv_err: float = 0.0
    teff: float = np.nan
    gmag: float = np.nan
    row: object = None
    frame: str = "air"                  # rest-wavelength convention in use

    @property
    def opz(self):
        return 1.0 + z_from_rv(self.rv)

    @property
    def snr(self):
        if self.err is None:
            return np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(np.nanmedian(self.flux / self.err))

    def lam_obs(self, line):
        """Expected observed wavelength of a line for this star."""
        s = line_spec(line)
        return float(rest_wave(s.lam_air, self.frame)) * self.opz

    def pix_kms(self, lam=None):
        """Pixel scale in km/s at lam (default: band centre). The floor on any
        centroid, visual or fitted."""
        lam = float(np.nanmedian(self.wave)) if lam is None else float(lam)
        dl = float(np.nanmedian(np.diff(self.wave)))
        return C_KMS * dl / lam

    def __repr__(self):
        return (f"Star({self.sid}  G={self.gmag:.2f}  Teff={self.teff:.0f}K  "
                f"RV={self.rv:+.1f}+/-{self.rv_err:.1f} km/s  "
                f"S/N={self.snr:.1f}  npix={np.isfinite(self.flux).sum()}  "
                f"pix={self.pix_kms():.0f} km/s  frame={self.frame})")


def load(path):
    """
    Read a star-core FITS (the scrappy SPEC/ERROR/WAVE/CATALOG product) into the
    dict that get_star() consumes.

        prod = sl.load("outputs_starpsf/star_core_cosmos.fits")
        s    = sl.get_star(prod, 0)

    Returns a plain dict, not a Table -- so it is prod['cat'][:5] to peek at the
    catalog, not prod[:5].
    """
    from astropy.io import fits
    from astropy.table import Table
    out = {}
    with fits.open(path) as h:
        out["spec"] = h["SPEC"].data.astype(float)
        out["err"] = h["ERROR"].data.astype(float)
        out["wave"] = h["WAVE"].data.astype(float)
        out["cat"] = Table(h["CATALOG"].data)
        out["header"] = h[0].header
        for name in ("BKG", "BGPOOL"):
            if name in h:
                out[name.lower()] = h[name].data.astype(float)
    out["path"] = str(path)
    return out


def load_stars(path, frame="air"):
    """Every star in a file, as a list of Star objects. For the outer loop."""
    prod = load(path)
    return [get_star(prod, i, frame=frame) for i in range(len(prod["cat"]))]


_COLS = dict(sid=("ID", "SOURCE_ID", "source_id"),
             rv=("radial_velocity", "RV"),
             rv_err=("radial_velocity_error", "RV_ERR"),
             teff=("teff_gspphot", "TEFF"),
             gmag=("phot_g_mean_mag", "GMAG"))


def get_star(prod, i, frame="air") -> Star:
    """
    Star i from a read_star_core() product -- spectrum trimmed to finite pixels,
    Gaia parameters pulled from the CATALOG table. One call, no unpacking.

    prod : dict with 'wave', 'spec', 'err', 'cat' (the star-core FITS product).
    """
    w = np.asarray(prod["wave"], float)
    f = np.asarray(prod["spec"][i], float)
    e = np.asarray(prod["err"][i], float) if prod.get("err") is not None else None
    g = np.isfinite(f)
    if e is not None:
        g &= np.isfinite(e) & (e > 0)

    row = prod["cat"][i]
    cols = row.colnames if hasattr(row, "colnames") else list(getattr(row, "keys", lambda: [])())

    def _get(names, default=np.nan):
        for n in names:
            if n in cols:
                return row[n]
        return default

    return Star(wave=w[g], flux=f[g], err=(None if e is None else e[g]),
                sid=str(_get(_COLS["sid"], f"star{i}")),
                rv=float(_get(_COLS["rv"], 0.0) or 0.0),
                rv_err=float(_get(_COLS["rv_err"], 0.0) or 0.0),
                teff=float(_get(_COLS["teff"])),
                gmag=float(_get(_COLS["gmag"])),
                row=row, frame=frame)


def suggest(star: Star, verbose=True, mode="visual"):
    """
    Which lines are worth trying for THIS star -- in band, and expected to be
    strong at its Teff. Returns the list of keys.

    mode : "visual" (default) additionally requires visual_ok. snap() is a bare
        argmin with no ncomp handling, so a blend anchor like MgbBlend
        (ncomp=3) must not come back from the default suggest() that feeds
        snap_all()/measure()/visual_offset() -- that was silently doing an
        ncomp=1 read on a genuine triplet. Pass mode="fit" (what fit_lines()
        uses) to get every grade A/B line including blend anchors -- fit_line()
        DOES know how to share one velocity across ncomp components correctly.

    The first run fit Hd/Hg/Hb on a 3631 K star and got depth 0.008 and a railed
    fit. That is not a fitting failure, it is fitting a line that is not there.
    Use this instead of a hard-coded LINES list.
    """
    wmin, wmax = float(np.nanmin(star.wave)), float(np.nanmax(star.wave))
    keep, rows = [], []
    for k, s in LINES.items():
        lam = star.lam_obs(k)
        lo, hi = s.span
        in_band = (lo * star.opz >= wmin - 2) and (hi * star.opz <= wmax + 2)
        hot_ok = (not np.isfinite(star.teff)) or (s.teff[0] <= star.teff <= s.teff[1])
        vis_ok = (mode != "visual") or s.visual_ok
        use = in_band and hot_ok and vis_ok and s.grade in ("A", "B")
        if use:
            keep.append(k)
        rows.append((k, s.grade, lam, in_band, hot_ok, use))
    if verbose:
        print(f"{star.sid}  Teff={star.teff:.0f} K  mode={mode}")
        print(f"  {'line':10s} {'gr':3s} {'lam_obs':>9s} {'band':>5s} {'Teff':>5s}  use")
        for k, gr, lam, ib, ht, us in rows:
            print(f"  {k:10s} {gr:3s} {lam:9.2f} {str(ib):>5s} {str(ht):>5s}  "
                  f"{'YES' if us else '-'}")
        print(f"  -> {keep}")
    return keep


# =====================================================================
# 4.  LOOK  --  the plots that do the actual work
# =====================================================================

def _band(ax, x, y, e, color="0.75"):
    if e is not None:
        ax.fill_between(x, y - e, y + e, color=color, lw=0, zorder=1)


def plot_spectrum(star: Star, lines=None, ax=None, smooth=0, figsize=(13, 4.0)):
    """Full band, every catalog line marked at its RV-shifted position.
    Green = grade A, blue = B, red = C (never combine those)."""
    import matplotlib.pyplot as plt
    y = star.flux
    if smooth and smooth > 1:
        y = np.convolve(y, np.ones(int(smooth)) / int(smooth), mode="same")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _band(ax, star.wave, y, star.err)
    ax.plot(star.wave, y, lw=0.8, color="k", zorder=3)

    keys = list(LINES) if lines is None else list(lines)
    colors = {"A": "tab:green", "B": "tab:blue", "C": "tab:red"}
    for k in keys:
        s = line_spec(k)
        lam = star.lam_obs(k)
        if not (star.wave.min() <= lam <= star.wave.max()):
            continue
        ax.axvline(lam, color=colors[s.grade], ls=":", lw=1.0, zorder=2)
        ax.annotate(k, (lam, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5,
                    color=colors[s.grade], rotation=90)
    ax.set_xlim(star.wave.min(), star.wave.max())
    ax.set_xlabel(f"observed wavelength [A]   ({star.frame} rest, "
                  f"RV={star.rv:+.1f} km/s applied)")
    ax.set_ylabel("flux")
    ax.set_title(repr(star), fontsize=8.5)
    ax.margins(y=0.10)
    return ax


def plot_line(star: Star, line, v=None, halfwidth_A=25.0, ax=None,
              figsize=(6.6, 4.0), velocity_axis=False, show_windows=True,
              show_partners=True, smooth=0, normalize=False):
    """
    THE VISUAL-FIT PANEL. One line, a ~50 A window, drawn as a LINE not a scatter.

    v : a velocity in km/s, or a list of them, drawn as vertical markers so you
        can slide a trial offset until it sits on the trough. This is the
        manual fit -- read the number off, then pass it to visual_offset().

    show_windows shades the continuum windows. LOOK AT THEM. The v1 Mg b red
    window sat on the Fe I blend and pinned the continuum ~10% low; that is
    invisible in a fit result and obvious here.

    show_partners marks the blended companions (Mg b's other two lines, H-eps
    under Ca II H) at the same trial velocity, so you can check that the whole
    complex lines up rather than just the one you are looking at.
    """
    import matplotlib.pyplot as plt
    s = line_spec(line)
    lam_e = star.lam_obs(line)
    m = np.abs(star.wave - lam_e) <= halfwidth_A
    if m.sum() < 3:
        print(f"{s.key}: only {int(m.sum())} pixels in window")
        return None

    x, y = star.wave[m], star.flux[m]
    e = None if star.err is None else star.err[m]
    if smooth and smooth > 1:
        y = np.convolve(y, np.ones(int(smooth)) / int(smooth), mode="same")
    if normalize:
        n = _pseudo_cont(star, s, halfwidth_A)
        if n is not None:
            cy = np.interp(x, n[0], n[1])
            y = y / cy
            e = None if e is None else e / np.abs(cy)

    xx = (x - lam_e) / lam_e * C_KMS if velocity_axis else x
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    _band(ax, xx, y, e)
    ax.plot(xx, y, "-", lw=1.3, color="k", drawstyle="steps-mid", zorder=3)

    if show_windows:
        for w0, w1 in (s.blue, s.red):
            a, b = w0 * star.opz, w1 * star.opz
            if velocity_axis:
                a = (a - lam_e) / lam_e * C_KMS
                b = (b - lam_e) / lam_e * C_KMS
            ax.axvspan(a, b, color="tab:orange", alpha=0.13, lw=0, zorder=0)

    zero = 0.0 if velocity_axis else lam_e
    ax.axvline(zero, color="tab:green", ls="--", lw=1.4, zorder=4,
               label=f"Gaia ({star.rv:+.1f} km/s)")

    vs = [] if v is None else (list(v) if np.ndim(v) else [float(v)])
    cyc = ["tab:blue", "tab:purple", "tab:brown", "tab:cyan", "magenta"]
    for j, vv in enumerate(vs):
        c = cyc[j % len(cyc)]
        pos = vv if velocity_axis else lam_e * (1.0 + vv / C_KMS)
        ax.axvline(pos, color=c, lw=1.5, zorder=5, label=f"{vv:+.0f} km/s")
        if show_partners:
            for comp in s.components[1:]:
                lc = float(rest_wave(comp.lam_air, star.frame)) * star.opz
                p = lc * (1.0 + vv / C_KMS)
                p = (p - lam_e) / lam_e * C_KMS if velocity_axis else p
                ax.axvline(p, color=c, lw=1.0, ls=":", alpha=0.8, zorder=5)

    pk = star.pix_kms(lam_e)
    ax.set_xlabel("velocity vs Gaia [km/s]" if velocity_axis
                  else "observed wavelength [A]")
    ax.set_ylabel("normalized flux" if normalize else "flux")
    ax.set_title(f"{star.sid}   {s.key} ({s.name}, grade {s.grade})   "
                 f"lam_exp={lam_e:.2f} A   1 pix = {pk:.0f} km/s",
                 fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right")
    return ax


def plot_lines(star: Star, lines=None, v=None, ncols=2, halfwidth_A=25.0,
               figsize_per=(6.2, 3.4), **kw):
    """
    Grid of line panels for one star -- the visual-inspection sweep.

    v : dict {line_key: velocity} to mark a different trial on each panel, or a
        scalar/list applied to all. Pass the dict you are building up and re-run
        until every panel looks right.

    If `lines` is omitted and `v` is a dict, the panels are TAKEN FROM v -- so
    plot_lines(s, v=v2) shows exactly the lines you measured, not a default set.
    """
    import matplotlib.pyplot as plt
    if lines is None:
        keys = list(v) if isinstance(v, dict) else list(GOOD_LINES)
    else:
        keys = [k.key if isinstance(k, Line) else k for k in
                ([lines] if isinstance(lines, (str, Line)) else lines)]
    nrow = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrow, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        vv = v.get(k) if isinstance(v, dict) else v
        plot_line(star, k, v=vv, ax=ax, halfwidth_A=halfwidth_A, **kw)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# =====================================================================
# 5.  VISUAL FIT  --  eyeball -> number
# =====================================================================

def _pseudo_cont(star: Star, s: Line, halfwidth_A):
    """Straight line through the MEDIANS of the two windows. Median, not mean:
    in a cool star the windows are peppered with weak lines and the mean is
    dragged down by them."""
    lo, hi = s.blue[0] * star.opz, s.red[1] * star.opz
    m = (star.wave >= lo) & (star.wave <= hi)
    if m.sum() < 6:
        return None
    w, f = star.wave[m], star.flux[m]
    mb = (w >= s.blue[0] * star.opz) & (w <= s.blue[1] * star.opz)
    mr = (w >= s.red[0] * star.opz) & (w <= s.red[1] * star.opz)
    if mb.sum() < 2 or mr.sum() < 2:
        return None
    xb, yb = np.nanmedian(w[mb]), np.nanmedian(f[mb])
    xr, yr = np.nanmedian(w[mr]), np.nanmedian(f[mr])
    slope = (yr - yb) / (xr - xb) if xr != xb else 0.0
    return w, yb + slope * (w - xb)


def snap(star: Star, line, v0=0.0, search_kms=250.0, refine=True, verbose=True):
    """
    Refine an eyeballed velocity to the local extremum -- the trough minimum
    within +/- search_kms of your guess, with a 3-point parabolic sub-pixel
    refinement (one pixel is >100 km/s, so the refinement is not cosmetic).

    This is Weldon's composite estimator (sec 3.5, "the wavelength at the
    absolute minimum of the trough") applied per star. It is a DIFFERENT
    statistic from a Gaussian centroid: on an asymmetric trough the minimum sits
    redward of the centroid. For a stellar photospheric line the asymmetry is
    small, but do not mix the two estimators when averaging.
    """
    s = line_spec(line)
    lam_e = star.lam_obs(line)
    lo = lam_e * (1.0 + (v0 - search_kms) / C_KMS)
    hi = lam_e * (1.0 + (v0 + search_kms) / C_KMS)
    m = (star.wave >= lo) & (star.wave <= hi) & np.isfinite(star.flux)
    if m.sum() < 3:
        if verbose:
            print(f"{s.key}: only {int(m.sum())} pixels in search range")
        return np.nan

    w, f = star.wave[m], star.flux[m]
    pc = _pseudo_cont(star, s, 0)
    nf = f / np.interp(w, pc[0], pc[1]) if pc is not None else f
    j = int(np.argmin(nf) if s.sign < 0 else np.argmax(nf))
    lam = w[j]
    if refine and 0 < j < len(w) - 1:
        d = np.median(np.diff(w))
        den = nf[j - 1] - 2 * nf[j] + nf[j + 1]
        if np.isfinite(den) and den != 0:
            sh = 0.5 * (nf[j - 1] - nf[j + 1]) / den
            if abs(sh) <= 1.0:
                lam = lam + sh * d
    v = C_KMS * (lam - lam_e) / lam_e
    depth = float(abs(1.0 - nf[j])) if pc is not None else np.nan
    if verbose:
        print(f"{s.key}: guess {v0:+.0f} -> snapped {v:+.1f} km/s "
              f"(depth {depth:.3f}, 1 pix = {star.pix_kms(lam_e):.0f} km/s)")
    return float(v)


def snap_all(star: Star, lines=None, search_kms=250.0, verbose=True):
    """
    snap() a whole set of lines at once. Returns {line: velocity} of refined
    values -- feed it straight back into plot_lines(star, v=...) to confirm the
    refinement landed where you meant it to.

    `lines` takes any of these (see _as_guesses):
        sl.snap_all(s)                          # whatever suggest() picks
        sl.snap_all(s, sl.WARM)                 # a named set, all starting at 0
        sl.snap_all(s, ["CaK", "Mgb"])          # your own list
        sl.snap_all(s, {"CaK": -70, "Mgb": -95})  # explicit eyeballed guesses

    Starting at 0 lets snap find the nearest minimum within +/- search_kms. That
    is fine when the line is unambiguous, but a 0-start next to a strong blend
    will happily lock onto the WRONG feature and will not tell you. Always look
    at plot_lines afterwards.
    """
    out = {}
    for k, v0 in _as_guesses(star, lines).items():
        v = snap(star, k, v0, search_kms=search_kms, verbose=verbose)
        if np.isfinite(v):
            out[k] = float(v)
    return out


def visual_offset(star: Star, v_by_line, v_err_kms=None, snap_first=True,
                  search_kms=250.0):
    """
    Turn eyeballed per-line velocities into one eps for this star.

    THE TROUGH MINIMUM IS THE PRIMARY ESTIMATOR for this project (Austin,
    2026-07-30), so snap_first defaults to True: the eyeball picks the right
    feature, snap() sets the value from the data. Pass snap_first=False only if
    you deliberately want the raw click positions.

    Because the minimum is the primary statistic, do NOT average these with
    fit_line()'s Gaussian centroids -- they are different estimators and on an
    asymmetric trough they differ systematically. Use the fit as a cross-check
    on the same line, and quote the difference as a systematic if it is large.

    v_by_line : anything _as_guesses accepts -- a dict of eyeballed velocities,
        a list of line keys (each starting at 0), a named set like sl.WARM, or
        None for suggest(star).
    v_err_kms : per-line uncertainty. None (default) assigns HALF A PIXEL at
        each line's wavelength, which is an honest floor on a visual read and is
        typically 55-70 km/s. Pass a dict or a scalar to override.
    snap_first : run snap() on each guess before combining.

    Grade-C lines are refused. Gband especially: a molecular band has no rest
    wavelength and cannot carry a velocity.

    Returns the same shape as combine(), so visual and fitted results pool
    together in combine_stars().
    """
    vals, errs, names = [], [], []
    for k, v0 in _as_guesses(star, v_by_line).items():
        s = line_spec(k)
        if s.grade == "C":
            print(f"  [skipped {k}: grade C -- {s.note.split('.')[0]}]")
            continue
        if not s.visual_ok:
            print(f"  [skipped {k}: fit-only (visual_ok=False), the trough "
                  f"minimum has no ncomp handling for this blend -- "
                  f"{s.note.split('.')[0]}]")
            continue
        v = snap(star, k, v0, search_kms=search_kms, verbose=False) if snap_first else float(v0)
        if not np.isfinite(v):
            continue
        if v_err_kms is None:
            ev = 0.5 * star.pix_kms(star.lam_obs(k))
        elif isinstance(v_err_kms, dict):
            ev = float(v_err_kms.get(k, 0.5 * star.pix_kms(star.lam_obs(k))))
        else:
            ev = float(v_err_kms)
        vals.append(v); errs.append(ev); names.append(k)

    return _combine_values(vals, errs, names, star.rv_err, method="visual")


def measure(star: Star, lines=None, search_kms=250.0, v_err_kms=None,
            verbose=False):
    """
    snap_all + visual_offset in one call. The whole per-star measurement.

        v, off = sl.measure(s)                # suggest() picks the lines
        v, off = sl.measure(s, sl.WARM)       # or name a set
        v, off = sl.measure(s, {"CaK": -70})  # or hand it your eyeballed reads

    Returns (v_snapped, offset) so you can still plot the snapped velocities:
        sl.plot_lines(s, v=v)
    """
    v = snap_all(star, lines, search_kms=search_kms, verbose=verbose)
    return v, visual_offset(star, v, v_err_kms=v_err_kms, snap_first=False)


def measure_all(stars, lines=None, min_snr=None, min_pixels=100,
                teff_range=None, verbose=True, **kw):
    """
    Run measure() over a list of Stars and pool them. Returns
    (per_star, composite, table_rows).

    min_snr / min_pixels / teff_range are SAMPLE CUTS, applied before measuring.
    Use them deliberately and report them: cutting to the high-S/N (hotter)
    stars is defensible, but it correlates the sample with Teff, so if eps also
    correlates with Teff you have just selected on the systematic. Run the full
    sample and the cut sample and compare before choosing one.
    """
    per_star, rows = {}, []
    for st in stars:
        if min_pixels and np.isfinite(st.flux).sum() < min_pixels:
            continue
        if min_snr is not None and not (st.snr >= min_snr):
            continue
        if teff_range is not None and not (
                teff_range[0] <= st.teff <= teff_range[1]):
            continue
        v, off = measure(st, lines, **kw)
        if not np.isfinite(off["eps_kms"]):
            continue
        per_star[st.sid] = off
        rows.append(dict(sid=st.sid, G=st.gmag, Teff=st.teff, snr=st.snr,
                         n_lines=off["n_lines"], eps=off["eps_kms"],
                         err=off["eps_err_total_kms"],
                         scatter=off["scatter_kms"], chi2=off["chi2_dof"],
                         lines=",".join(off["lines"])))
        if verbose:
            print(f"{st.sid:22s} G={st.gmag:5.2f} Teff={st.teff:5.0f} "
                  f"S/N={st.snr:5.1f} n={off['n_lines']}  "
                  f"eps={off['eps_kms']:+7.1f} +/- {off['eps_err_kms']:5.1f}")
    return per_star, combine_stars(per_star), rows


# =====================================================================
# 6.  AUTOMATED FIT  (secondary -- the visual path is not inferior)
# =====================================================================

def fit_line(star: Star, line, ncomp=None, cont_order=1,
             eps_bounds_kms=(-350.0, 350.0), sigma_max_factor=8.0,
             min_depth_sigma=3.0, min_pixels=12, n_mc=60, seed=0,
             lsf_fwhm_A=None):
    """
    Poly continuum + N Gaussians sharing one velocity and one width, amplitudes
    sign-constrained, fitted in the OBSERVED frame.

    Changes from v1, all driven by the first real run:
      * eps_bounds tightened to +/-350 (was +/-600). H-beta railed at +600 on a
        2-pixel noise dip; a real wavelength-solution error of >350 km/s would
        be visible by eye in the overview plot.
      * DETECTION GATE BEFORE the velocity is reported: the line must be deeper
        than min_depth_sigma times the local noise. H-gamma came back at depth
        0.008 with a reported eps of -38.5 +/- 4.3 km/s -- a precise measurement
        of nothing. A shallow line now fails outright.
      * ncomp defaults to the line's own recommendation (3 for Mg b), not 1.
      * sigma floor stays AT the LSF, not 0.5x.
    """
    s = line_spec(line)
    nc = s.ncomp if ncomp is None else int(ncomp)
    comps = s.components[:max(1, nc)]
    n = len(comps)

    lam_e = star.lam_obs(line)
    lo, hi = s.span
    m = (star.wave >= lo * star.opz) & (star.wave <= hi * star.opz)
    if star.err is not None:
        m &= np.isfinite(star.err) & (star.err > 0)
    m &= np.isfinite(star.flux)

    fw = float(np.atleast_1d(instrumental_fwhm(
        np.array([lam_e]), mode="fwhm",
        **({} if lsf_fwhm_A is None else {"fwhm_A": float(lsf_fwhm_A)})))[0])
    sig_lsf = fw * FWHM_TO_SIGMA

    out = dict(line=s.key, name=s.name, grade=s.grade, frame=star.frame,
               ncomp=n, method="fit", success=False, reason=None,
               eps_kms=np.nan, eps_err_kms=np.nan, eps_point_kms=np.nan,
               sigma_A=np.nan, sigma_lsf_A=sig_lsf, sigma_over_lsf=np.nan,
               depth=np.nan, depth_sigma=np.nan, snr=np.nan,
               lam_obs_expect=lam_e, lam_obs_meas=np.nan,
               rv_err_kms=star.rv_err, _x=None, _y=None, _e=None,
               _model=None, _cont=None)

    if curve_fit is None:
        return {**out, "reason": "scipy unavailable"}
    if m.sum() < min_pixels:
        return {**out, "reason": f"{int(m.sum())} pixels (< {min_pixels})"}

    x, y = star.wave[m], star.flux[m]
    e = None if star.err is None else star.err[m]
    lam_c = np.array([float(rest_wave(c.lam_air, star.frame)) * star.opz
                      for c in comps])
    signs = np.array([c.sign for c in comps], float)
    span_A = (hi - lo) * star.opz
    ncoef = int(cont_order) + 1

    def _mod(xx, *p):
        eps, sg = p[0], p[1]
        a = np.asarray(p[2:2 + n], float)
        cc = np.asarray(p[2 + n:], float)
        r = np.polyval(cc, (xx - lam_e) / span_A)
        sh = 1.0 + eps / C_KMS
        for ai, lc in zip(a, lam_c):
            r = r + ai * np.exp(-0.5 * ((xx - lc * sh) / sg) ** 2)
        return r

    c0 = np.nanmedian(y)
    core = np.abs(x - lam_e) <= max(2 * sig_lsf, 3.0)
    d0 = (c0 - np.nanmin(y[core])) if core.sum() > 2 else 0.1 * abs(c0)
    p0 = [0.0, 1.5 * sig_lsf] + [float(sg * abs(d0)) for sg in signs] + \
         [0.0] * (ncoef - 1) + [float(c0)]
    lb = [eps_bounds_kms[0], sig_lsf] + \
         [(-np.inf if sg < 0 else 0.0) for sg in signs] + [-np.inf] * ncoef
    ub = [eps_bounds_kms[1], sigma_max_factor * sig_lsf] + \
         [(0.0 if sg < 0 else np.inf) for sg in signs] + [np.inf] * ncoef
    p0 = [float(np.clip(v, a, b)) for v, a, b in zip(p0, lb, ub)]

    def _run(yy):
        try:
            return curve_fit(_mod, x, yy, p0=p0, bounds=(lb, ub), sigma=e,
                             absolute_sigma=(e is not None), maxfev=20000)[0]
        except Exception:
            return None

    p = _run(y)
    if p is None:
        return {**out, "reason": "did not converge"}

    eps_p, sig_p = float(p[0]), float(p[1])
    amps = np.asarray(p[2:2 + n], float)
    cont = np.polyval(np.asarray(p[2 + n:], float), (x - lam_e) / span_A)
    model = _mod(x, *p)
    cmed = float(np.nanmedian(np.abs(cont)))
    depth = float(abs(amps[0]) / cmed) if cmed > 0 else np.nan
    noise = float(np.nanmedian(e)) if e is not None else np.nan
    dsig = float(abs(amps[0]) / noise) if np.isfinite(noise) and noise > 0 else np.nan

    out.update(depth=depth, depth_sigma=dsig, snr=dsig,
               sigma_A=sig_p, sigma_over_lsf=sig_p / sig_lsf,
               eps_point_kms=eps_p, _x=x, _y=y, _e=e, _model=model, _cont=cont)

    # ---- DETECTION GATE, before any velocity is believed ----
    if np.isfinite(dsig) and dsig < min_depth_sigma:
        return {**out, "reason": f"NOT DETECTED (depth {dsig:.1f} sigma "
                                 f"< {min_depth_sigma}); line absent at this Teff?"}
    if abs(eps_p - eps_bounds_kms[0]) < 1 or abs(eps_p - eps_bounds_kms[1]) < 1:
        return {**out, "reason": f"eps railed on {eps_bounds_kms}"}
    if sig_p <= sig_lsf * 1.001:
        return {**out, "reason": "sigma railed AT the LSF floor -- continuum "
                                 "ate the wings, centroid untrustworthy"}
    if sig_p >= sigma_max_factor * sig_lsf * 0.999:
        return {**out, "reason": f"sigma railed at {sigma_max_factor}x LSF"}

    eps_mc = np.full(int(n_mc), np.nan)
    if e is not None and n_mc:
        rng = np.random.default_rng(seed)
        for i in range(int(n_mc)):
            pi = _run(y + rng.standard_normal(y.shape) * e)
            if pi is not None:
                eps_mc[i] = float(pi[0])
    ok = np.isfinite(eps_mc)
    eps = float(np.mean(eps_mc[ok])) if ok.sum() > 1 else eps_p
    err = float(np.std(eps_mc[ok], ddof=1)) if ok.sum() > 1 else np.nan

    out.update(success=True, reason=None, eps_kms=eps, eps_err_kms=err,
               lam_obs_meas=lam_e * (1 + eps / C_KMS))
    return out


def fit_lines(star: Star, lines=None, **kw):
    """Loop fit_line over a star's lines. lines=None -> suggest(star, mode='fit')
    -- the fit path, unlike snap(), knows how to share one velocity across a
    blend's ncomp components, so it gets MgbBlend as well as everything the
    visual path gets."""
    keys = suggest(star, verbose=False, mode="fit") if lines is None else list(lines)
    return {k: fit_line(star, k, **kw) for k in keys}


# =====================================================================
# 7.  COMBINE
# =====================================================================

def _combine_values(vals, errs, names, rv_err, method="fit"):
    n = len(vals)
    if n == 0:
        return dict(eps_kms=np.nan, eps_err_kms=np.nan, eps_err_total_kms=np.nan,
                    scatter_kms=np.nan, chi2_dof=np.nan, n_lines=0,
                    lines=[], rv_err_kms=rv_err, method=method,
                    per_line={}, drift_suspect=False)
    v = np.asarray(vals, float); s = np.asarray(errs, float)
    w = 1.0 / s ** 2
    mu = float(np.sum(w * v) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    sc = float(np.std(v, ddof=1)) if n > 1 else np.nan
    chi2 = float(np.sum(w * (v - mu) ** 2) / (n - 1)) if n > 1 else np.nan
    return dict(eps_kms=mu, eps_err_kms=err,
                eps_err_total_kms=float(np.hypot(err, rv_err)),
                eps_err_inflated_kms=(err * np.sqrt(max(chi2, 1.0))
                                      if np.isfinite(chi2) else err),
                scatter_kms=sc, chi2_dof=chi2, n_lines=n, lines=names,
                rv_err_kms=rv_err, method=method,
                per_line=dict(zip(names, v)),
                drift_suspect=bool(np.isfinite(chi2) and chi2 > 3.0))


def combine(results, min_grade="B"):
    """
    One eps per star from fit_lines output. Same return shape as visual_offset.

    Every line in one star sees the SAME wavelength solution, so if the solution
    is merely offset they must agree within errors. chi2_dof >> 1 means they do
    not: check blends first, then suspect a real wavelength-dependent drift,
    which is not something to average away.
    """
    ok = {"A": {"A"}, "B": {"A", "B"}, "C": {"A", "B", "C"}}[min_grade]
    v, s, nm = [], [], []
    rv_err = 0.0
    for k, r in results.items():
        if not r["success"] or r["grade"] not in ok:
            continue
        if not (np.isfinite(r["eps_kms"]) and np.isfinite(r["eps_err_kms"])
                and r["eps_err_kms"] > 0):
            continue
        v.append(r["eps_kms"]); s.append(r["eps_err_kms"]); nm.append(k)
        rv_err = max(rv_err, float(r.get("rv_err_kms", 0.0) or 0.0))
    return _combine_values(v, s, nm, rv_err, method="fit")


def combine_stars(offsets):
    """
    Pool per-star eps. offsets : {star_id: combine()/visual_offset() result}.

    THE SCATTER IS THE NUMBER THAT MATTERS. Each star independently realises the
    same instrumental offset, so scatter much larger than the individual errors
    means something star-dependent is in play -- saturation in the bright ones,
    Teff-dependent blending, footprint position. Check eps against G and Teff
    before quoting the mean.
    """
    ids = list(offsets)
    v, s, keep = [], [], []
    for i in ids:
        o = offsets[i]
        ev, ee = o.get("eps_kms"), o.get("eps_err_total_kms", o.get("eps_err_kms"))
        if np.isfinite(ev) and np.isfinite(ee) and ee > 0:
            v.append(ev); s.append(ee); keep.append(i)
    n = len(v)
    if n == 0:
        return dict(eps_kms=np.nan, n_stars=0)
    v = np.asarray(v); s = np.asarray(s)
    w = 1.0 / s ** 2
    mu = float(np.sum(w * v) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    sc = float(np.std(v, ddof=1)) if n > 1 else np.nan
    chi2 = float(np.sum(w * (v - mu) ** 2) / (n - 1)) if n > 1 else np.nan
    return dict(eps_kms=mu, eps_err_kms=err, scatter_kms=sc,
                sem_kms=float(sc / np.sqrt(n)) if np.isfinite(sc) else np.nan,
                eps_err_inflated_kms=(err * np.sqrt(max(chi2, 1.0))
                                      if np.isfinite(chi2) else err),
                chi2_dof=chi2, n_stars=n, stars=keep,
                eps_per_star=dict(zip(keep, v)), err_per_star=dict(zip(keep, s)))


def describe(results, offset=None, title=None):
    """Per-star summary table. Works on fit_lines output."""
    if title:
        print(title)
    print(f"{'line':10s} {'gr':3s} {'lam_obs':>9s} {'eps':>8s} {'+/-':>6s} "
          f"{'depth_s':>7s} {'sig/LSF':>7s}  note")
    print("-" * 84)
    for k, r in results.items():
        flag = "" if r["success"] else f"  <-- {r['reason']}"
        print(f"{k:10s} {r['grade']:3s} {r['lam_obs_expect']:9.2f} "
              f"{r['eps_kms']:+8.1f} {r['eps_err_kms']:6.1f} "
              f"{r['depth_sigma']:7.1f} {r['sigma_over_lsf']:7.2f}{flag}")
    if offset is not None:
        print("-" * 84)
        print(f"  eps = {offset['eps_kms']:+.1f} +/- {offset['eps_err_kms']:.1f} "
              f"(meas) +/- {offset['rv_err_kms']:.1f} (Gaia)  "
              f"[{offset['n_lines']} lines: {', '.join(offset['lines'])}]  "
              f"method={offset['method']}")
        print(f"  line scatter = {offset['scatter_kms']:.1f} km/s, "
              f"chi2/dof = {offset['chi2_dof']:.2f}")
        if offset["drift_suspect"]:
            print("  ** lines DISAGREE. Check blends, then suspect drift. "
                  "Do not quote the mean.")
    return results
