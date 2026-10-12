"""
plotting.py -- display layer (Stage 3+ outputs).

Purely for visualizing results AFTER computation. Consumes the Stage 2 stacks
dict from build_stacks / run_stack:

    stacks_result = {
        "rest_wave": (nwave,),
        "r_edges":   (nrad+1,),
        "stacks":    {method: {"flux": (nrad,nwave), "err": (nrad,nwave)}},
        ...
    }

Nothing here computes science; it only draws. Keeping it isolated means import
of the pipeline never drags in matplotlib, and plots can evolve freely.
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from .config import PipelineConfig, DEFAULT_CONT_BOUNDS
from .formatting import apply_tick_style

C_KMS = 299792.458   # speed of light (km/s); wavelength <-> velocity about Lya


# sentinel distinguishing "use the auto-generated title" (default) from
# "no title at all" (title=None, an explicit user choice)
_AUTO = object()


def _resolve_savename(save_name: str | None, default: str) -> str:
    """
    Turn a user-supplied `save_name` (with or without an extension, e.g.
    'Flux_Profile' or 'Flux_Profile.png') into a concrete filename for
    plt.savefig, falling back to `default` when save_name is None.
    """
    if not save_name:
        return default
    known_ext = (".png", ".pdf", ".jpg", ".jpeg", ".svg", ".eps", ".tif", ".tiff")
    return save_name if save_name.lower().endswith(known_ext) else save_name + ".png"


def get_ylim_window(wave, flux, err=None, xlim=(1150, 1300), pad=0.20,
                    percentiles=(1, 99), include_zero=True, min_span=None):
    """Robust y-limits from the spectrum within xlim (optionally incl. +/- err)."""
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    m = (wave >= xlim[0]) & (wave <= xlim[1]) & np.isfinite(flux)
    if err is not None:
        err = np.asarray(err)
        m &= np.isfinite(err)
        y = np.concatenate([flux[m] - err[m], flux[m] + err[m]])
    else:
        y = flux[m]
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return -1, 1
    ymin, ymax = np.nanpercentile(y, percentiles)
    if include_zero:
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)
    span = ymax - ymin
    if min_span is not None:
        span = max(span, min_span)
    if span == 0 or not np.isfinite(span):
        span = abs(ymax) if ymax != 0 else 1
    pad_amt = pad * span
    return ymin - pad_amt, ymax + (2 * pad_amt)


def plot_stack_panels(
    stacks_result,
    method="biweight",
    distance_type="virial radii",
    title=None,
    xlim=(1150, 1300),
    lya_center=PipelineConfig.LYA_REST,
    lya_window=4,
    figsize_width=10,
    panel_height=2.5,
    show_err=True,
    smooth=None,
    savepath=None,
    dpi=300,
):
    """
    One panel per radial bin for a chosen galaxy-combine method.

    stacks_result : the dict returned by run_stack / build_stacks.
    method        : which key in stacks_result['stacks'] to plot.
    distance_type : radial-unit label for the per-panel annotation.

    Returns (fig, axs).
    """
    rest_wave = np.asarray(stacks_result["rest_wave"])
    r_edges = np.round(np.asarray(stacks_result["r_edges"]), 3)
    block = stacks_result["stacks"][method]
    flux_all = np.asarray(block["flux"])     # (nrad, nwave)
    err_all = np.asarray(block["err"])
    nrad = flux_all.shape[0]

    y_unit = (stacks_result.get("unit_info") or {}).get("y_unit", "")

    fig, axs = plt.subplots(
        nrad, 1, figsize=(figsize_width, panel_height * nrad),
        sharex=True, constrained_layout=True,
    )
    if nrad == 1:
        axs = [axs]

    win_l, win_r = lya_center - lya_window, lya_center + lya_window

    for i, ax in enumerate(axs):
        flux = flux_all[i]
        err = err_all[i] if show_err else None

        y = gaussian_filter1d(flux, smooth) if smooth is not None else flux
        ax.step(rest_wave, y, where="mid", lw=1, label=f"{method} stack")

        if show_err and err is not None:
            ax.fill_between(rest_wave, flux - err, flux + err,
                            step="mid", color="0.75", alpha=0.35, lw=0)

        ax.axvspan(win_l, win_r, color="magenta", alpha=0.03)
        ax.axvline(lya_center, color="magenta", lw=0.8, alpha=0.6)
        ax.axvline(win_l, color="magenta", lw=0.8, alpha=0.3)
        ax.axvline(win_r, color="magenta", lw=0.8, alpha=0.3)
        ax.axhline(0, color="tomato", lw=1, alpha=0.8)

        ymin, ymax = get_ylim_window(rest_wave, flux, err=err, xlim=xlim,
                                     pad=0.25, percentiles=(1, 99),
                                     include_zero=False)
        ax.set_ylim(ymin, ymax)

        label = f"{np.round(r_edges[i], 3)}–{np.round(r_edges[i+1], 3)} {distance_type}"
        ax.text(0.02, 0.82, label, transform=ax.transAxes, fontsize=11,
                color="dodgerblue",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.85",
                          boxstyle="round,pad=0.25"))
        ax.tick_params(axis="both", which="major", labelsize=10, direction="in")
        ax.grid(alpha=0.18, lw=0.6)
        if i == 0:
            ax.legend(fontsize=12, frameon=False, loc="upper right")

    axs[-1].set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=15)
    axs[-1].set_xlim(xlim)
    fig.supylabel(f"Flux density [{y_unit}]" if y_unit else "Flux density", fontsize=14)

    if title is None:
        title = f"Halo Spectrum using {distance_type} bins ({method})"
    fig.suptitle(title, fontsize=17, y=1.01)

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axs


def plot_single_galaxy_panels(sg, *, xlim=(1150, 1300),
                              lya_center=PipelineConfig.LYA_REST, lya_window=4.0,
                              figsize_width=10, panel_height=2.3,
                              smooth=None, show_err=True,
                              title=None, savepath=None, dpi=300):
    """
    One rest-frame spectral panel per radial bin for a SINGLE galaxy, each panel
    annotated with the bin's fiber count and crude line S/N -- so you can see
    how far out this object is trustworthy. The single-galaxy analog of
    plot_stack_panels (same panel style; sg = stack.single_galaxy_spectra
    output, method fixed to its one 'galaxy' key).

    Because sg is stack-shaped, plot_stack_panels(sg, method='galaxy') and
    plot_radius_velocity_map(sg, method='galaxy') also work directly -- this
    wrapper just adds the nfib/S/N annotation those don't carry.

    sg : the dict from stack.single_galaxy_spectra.
    Returns (fig, axs).
    """
    rest_wave = np.asarray(sg["rest_wave"])
    r_edges   = np.round(np.asarray(sg["r_edges"]), 3)
    flux_all  = np.asarray(sg["stacks"]["galaxy"]["flux"])   # (nrad, nwave)
    err_all   = np.asarray(sg["stacks"]["galaxy"]["err"])
    nfib      = np.asarray(sg["nfib"])
    sn        = np.asarray(sg["sn"])
    nrad      = flux_all.shape[0]
    unit      = (sg.get("unit_info") or {}).get("y_unit", "")
    dist      = {"virial": r"$R/R_{\rm vir}$", "vr": r"$R/R_{\rm vir}$",
                 "kpc": "kpc", "arcsec": '"'}.get(str(sg.get("bin_mode", "")).lower(), "")

    fig, axs = plt.subplots(nrad, 1, figsize=(figsize_width, panel_height * nrad),
                            sharex=True, constrained_layout=True)
    if nrad == 1:
        axs = [axs]
    win_l, win_r = lya_center - lya_window, lya_center + lya_window

    for r, ax in enumerate(axs):
        flux = flux_all[r]
        err  = err_all[r] if show_err else None
        y = gaussian_filter1d(flux, smooth) if smooth is not None else flux
        ax.step(rest_wave, y, where="mid", lw=1, color="k")
        if show_err and err is not None:
            ax.fill_between(rest_wave, flux - err, flux + err, step="mid",
                            color="0.75", alpha=0.35, lw=0)
        ax.axvspan(win_l, win_r, color="magenta", alpha=0.05)
        ax.axvline(lya_center, color="magenta", lw=0.8, alpha=0.6)
        ax.axhline(0, color="tomato", lw=1, alpha=0.8)

        win = (rest_wave >= xlim[0]) & (rest_wave <= xlim[1]) & np.isfinite(flux)
        if np.any(win):
            fmin, fmax = np.nanpercentile(flux[win], [1, 99])
            pad = 0.25 * (fmax - fmin if fmax > fmin else abs(fmax) + 1)
            ax.set_ylim(fmin - pad, fmax + pad)

        sn_txt = f"   S/N={sn[r]:.1f}" if np.isfinite(sn[r]) else ""
        label = (f"{np.round(r_edges[r], 3)}\u2013{np.round(r_edges[r+1], 3)} {dist}"
                f"   N$_{{\\rm fib}}$={int(nfib[r])}{sn_txt}")
        ax.text(0.02, 0.80, label, transform=ax.transAxes, fontsize=10,
                color="dodgerblue",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.85",
                          boxstyle="round,pad=0.25"))
        ax.tick_params(axis="both", which="major", labelsize=10, direction="in")
        ax.grid(alpha=0.18, lw=0.6)

    axs[-1].set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=14)
    axs[-1].set_xlim(xlim)
    fig.supylabel(f"Flux density [{unit}]" if unit else "Flux density", fontsize=13)
    if title is None:
        gid = sg.get("gid"); idx = sg.get("index")
        who = f"ID {gid}" if gid not in (None, "None") else f"row {idx}"
        title = (rf"Single galaxy {who}   ($z={sg.get('z', float('nan')):.3f}$, "
                rf"$\log M={sg.get('mass', float('nan')):.2f}$)")
    fig.suptitle(title, fontsize=15, y=1.01)

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axs


def plot_stack_panels_two(
    stacks_a,
    stacks_b,
    labels=("A", "B"),
    method="biweight",
    layout="overlay",
    distance_type="virial radii",
    title=None,
    xlim=(1150, 1300),
    lya_center=PipelineConfig.LYA_REST,
    lya_window=4,
    figsize_width=10,
    panel_height=2.5,
    show_err=True,
    smooth=None,
    colors=("tab:blue", "tab:red"),
    savepath=None,
    dpi=300,
):
    """
    Two-stack version of plot_stack_panels: one row per radial bin, with BOTH
    stacks drawn so the half-sample comparison is read on a single figure.

    layout : {"overlay", "side"}
        "overlay" -> both stacks on the SAME axes per bin (one column). The y-limit
        for each bin is computed from get_ylim_window on EACH stack separately and
        then unioned (lowest low, highest high), so neither curve is clipped and
        both bins share an honest common scale.
        "side"    -> two columns (stack A | stack B) sharing the per-bin y-axis, so
        the same bin in each column lines up for direct side-by-side viewing.

    Both stacks must share the same r_edges and rest_wave grid (true for two cuts
    of the same run). Returns (fig, axes); axes is shape (nrad,) for overlay or
    (nrad, 2) for side.

    Example
    -------
        from utils_lya_halo import plotting
        plotting.plot_stack_panels_two(stacks_low_m, stacks_high_m,
                                       labels=("low-M", "high-M"))            # overlay
        plotting.plot_stack_panels_two(stacks_low_m, stacks_high_m,
                                       labels=("low-M", "high-M"), layout="side")
    """
    if layout not in ("overlay", "side"):
        raise ValueError("layout must be 'overlay' or 'side'")

    rest_wave = np.asarray(stacks_a["rest_wave"])
    r_edges = np.round(np.asarray(stacks_a["r_edges"]), 3)
    blocks = [np.asarray(stacks_a["stacks"][method]["flux"]),
              np.asarray(stacks_b["stacks"][method]["flux"])]
    errs = [np.asarray(stacks_a["stacks"][method]["err"]),
            np.asarray(stacks_b["stacks"][method]["err"])]
    nrad = blocks[0].shape[0]
    if blocks[1].shape[0] != nrad:
        raise ValueError(f"stacks have different nrad ({nrad} vs "
                         f"{blocks[1].shape[0]}); they must share r_edges.")
    y_unit = (stacks_a.get("unit_info") or {}).get("y_unit", "")
    win_l, win_r = lya_center - lya_window, lya_center + lya_window

    def _bin_ylim(i):
        """Union of the robust y-window over BOTH stacks for bin i."""
        los, his = [], []
        for flux_all, err_all in zip(blocks, errs):
            e = err_all[i] if show_err else None
            lo, hi = get_ylim_window(rest_wave, flux_all[i], err=e, xlim=xlim,
                                     pad=0.25, percentiles=(1, 99),
                                     include_zero=False)
            los.append(lo); his.append(hi)
        return min(los), max(his)

    ncol = 1 if layout == "overlay" else 2
    fig, axs = plt.subplots(
        nrad, ncol,
        figsize=(figsize_width * (1 if layout == "overlay" else 1.7),
                 panel_height * nrad),
        sharex=True, squeeze=False, constrained_layout=True,
    )

    for i in range(nrad):
        ymin, ymax = _bin_ylim(i)           # shared across both stacks (and columns)
        label_bin = f"{r_edges[i]}\u2013{r_edges[i+1]} {distance_type}"

        if layout == "overlay":
            ax = axs[i, 0]
            cells = [(ax, 0), (ax, 1)]
        else:
            cells = [(axs[i, 0], 0), (axs[i, 1], 1)]

        for ax, k in cells:
            flux = blocks[k][i]
            y = gaussian_filter1d(flux, smooth) if smooth is not None else flux
            ax.step(rest_wave, y, where="mid", lw=1.1, color=colors[k],
                    label=labels[k])
            if show_err:
                err = errs[k][i]
                ax.fill_between(rest_wave, flux - err, flux + err, step="mid",
                                color=colors[k], alpha=0.18, lw=0)
            ax.axvspan(win_l, win_r, color="magenta", alpha=0.03)
            ax.axvline(lya_center, color="magenta", lw=0.8, alpha=0.6)
            ax.axhline(0, color="tomato", lw=1, alpha=0.8)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(xlim)
            ax.grid(alpha=0.18, lw=0.6)
            ax.tick_params(axis="both", which="major", labelsize=10, direction="in")

        # bin label + legend
        if layout == "overlay":
            axs[i, 0].text(0.02, 0.82, label_bin, transform=axs[i, 0].transAxes,
                           fontsize=11, color="0.25",
                           bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.85",
                                     boxstyle="round,pad=0.25"))
            if i == 0:
                axs[i, 0].legend(fontsize=11, frameon=False, loc="upper right")
        else:
            for c in (0, 1):
                axs[i, c].text(0.02, 0.82, label_bin, transform=axs[i, c].transAxes,
                               fontsize=10, color="0.25",
                               bbox=dict(facecolor="white", alpha=0.85,
                                         edgecolor="0.85", boxstyle="round,pad=0.25"))
            if i == 0:
                for c in (0, 1):
                    axs[i, c].set_title(labels[c], fontsize=13)

    for c in range(ncol):
        axs[-1, c].set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=14)
    fig.supylabel(f"Flux density [{y_unit}]" if y_unit else "Flux density",
                  fontsize=14)
    if title is None:
        title = (f"{labels[0]} vs {labels[1]} \u2014 {distance_type} bins ({method})")
    fig.suptitle(title, fontsize=16, y=1.01)

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, (axs[:, 0] if layout == "overlay" else axs)


def plot_methods_overlay(
    stacks_result,
    methods=None,
    radial_index=0,
    xlim=(1150, 1300),
    lya_center=PipelineConfig.LYA_REST,
    smooth=None,
    title=None,
    set_ylim=True,
    ylim_pad=0.25,
    ylim_percentiles=(1, 99),
    include_zero=False,
):
    """
    Overlay several galaxy-combine methods for ONE radial bin -- handy for
    comparing biweight vs inv_var vs mean vs median on the same axes.

    set_ylim : bool
        If True, set robust y-limits from the line region (via get_ylim_window)
        instead of letting matplotlib autoscale to the noisy sideband ends. The
        window is computed for every overlaid method and unioned, so all curves
        stay in frame while the line region drives the scale. ylim_pad,
        ylim_percentiles, and include_zero are passed straight through.
    """
    rest_wave = np.asarray(stacks_result["rest_wave"])
    available = list(stacks_result["stacks"].keys())
    methods = methods or available

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ymins, ymaxs = [], []
    for m in methods:
        flux = np.asarray(stacks_result["stacks"][m]["flux"])[radial_index]
        y = gaussian_filter1d(flux, smooth) if smooth is not None else flux
        ax.step(rest_wave, y, where="mid", lw=1, label=m)

        if set_ylim:
            # scale off the displayed curve (smoothed if smooth is set)
            lo, hi = get_ylim_window(rest_wave, y, err=None, xlim=xlim,
                                     pad=ylim_pad, percentiles=ylim_percentiles,
                                     include_zero=include_zero)
            ymins.append(lo)
            ymaxs.append(hi)

    ax.axvline(lya_center, color="magenta", lw=0.8, alpha=0.6)
    ax.axhline(0, color="tomato", lw=1, alpha=0.8)
    ax.set_xlim(xlim)
    if set_ylim and ymins:
        ax.set_ylim(min(ymins), max(ymaxs))
    ax.set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=13)
    r_edges = np.round(np.asarray(stacks_result["r_edges"]), 3)
    ax.set_title(title or f"Bin {radial_index}: "
                 f"{np.round(r_edges[radial_index],3)}–"
                 f"{np.round(r_edges[radial_index+1],3)}", fontsize=13)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(alpha=0.18, lw=0.6)
    plt.show()
    return fig, ax


def _centers_to_edges(centers):
    """Midpoint edges for pcolormesh from monotonic bin centers (len n -> n+1)."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mid = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def plot_radius_velocity_map(
    stacks_result,
    method="biweight",
    distance_type="virial radii",
    vlim=(-2000, 2000),
    lya_center=PipelineConfig.LYA_REST,
    normalize="peak",
    cmap="RdBu_r",
    smooth=None,
    show_window=None,
    title=None,
    figsize=(7.5, 5.0),
    savepath=None,
    dpi=300,
):
    """
    2-D radius-velocity map of the stacked line: ALL radial bins in one image,
    velocity about Lya on x, radial bin on y, flux as color.

    Where plot_stack_panels draws one spectrum per panel, this compresses the
    whole run into a single figure so the *radial evolution* of the line is read
    at a glance -- how it fades outward and where its peak sits relative to
    systemic (v = 0, dashed). It is the natural visual companion to the Stage-3
    centroid-vs-radius result, but (like everything here) it computes no science:
    it only reshapes and displays stacks_result["stacks"][method]["flux"].

    Rows are drawn at EQUAL height (indexed by bin, labelled with their radial
    range) rather than on a physical-radius axis, so the thin inner bins are not
    visually crushed by the wide outer ones.

    Parameters
    ----------
    stacks_result : dict
        As returned by run_stack / build_stacks (same keys plot_stack_panels uses).
    method : str
        Which key in stacks_result["stacks"] to display.
    vlim : (float, float)
        Velocity window in km/s about Lya (x-limits and the slice shown).
    normalize : {"peak", "none"}
        "peak" divides each row by its own max(|flux|) inside the window, so the
        profile SHAPE / SHIFT is comparable across bins of very different
        brightness (inner bins are ~100x the outer ones); the colour scale is the
        symmetric [-1, 1]. Note this keys off |max|, so on spectra whose continuum
        has NOT been subtracted an outer row can be continuum-dominated. "none"
        keeps raw flux on a single shared scale, showing the amplitude DECLINE.
    show_window : float or None
        If given, a half-width in Angstrom drawn as dotted velocity guides
        (e.g. pass lya_window or half the config.line_window span).
    smooth : float or None
        Optional Gaussian sigma (pixels) applied along velocity for display only.

    Returns
    -------
    (fig, ax)
    """
    rest_wave = np.asarray(stacks_result["rest_wave"])
    r_edges = np.round(np.asarray(stacks_result["r_edges"]), 3)
    block = stacks_result["stacks"][method]
    flux_all = np.asarray(block["flux"])             # (nrad, nwave)
    nrad = flux_all.shape[0]
    y_unit = (stacks_result.get("unit_info") or {}).get("y_unit", "")

    # wavelength -> velocity about Lya, then restrict to the velocity window
    vel = C_KMS * (rest_wave - lya_center) / lya_center
    in_win = (vel >= vlim[0]) & (vel <= vlim[1])
    if not np.any(in_win):
        raise ValueError(f"No rest_wave samples fall inside vlim={vlim} km/s.")
    vel_w = vel[in_win]
    order = np.argsort(vel_w)                         # monotonic x for pcolormesh
    vel_w = vel_w[order]

    img = flux_all[:, in_win][:, order].astype(float)   # (nrad, nvel)
    if smooth is not None:
        img = gaussian_filter1d(img, smooth, axis=1)

    # display-only normalization (no science computed here)
    if normalize == "peak":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
            peak = np.nanmax(np.abs(img), axis=1, keepdims=True)
        peak = np.where(~np.isfinite(peak) | (peak == 0), np.nan, peak)
        img = img / peak
        vmax = 1.0
        cbar_label = "flux / per-bin peak"
    elif normalize == "none":
        vmax = np.nanpercentile(np.abs(img), 99)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        cbar_label = f"flux density [{y_unit}]" if y_unit else "flux density"
    else:
        raise ValueError("normalize must be 'peak' or 'none'")
    vmin = -vmax

    vel_edges = _centers_to_edges(vel_w)
    row_edges = np.arange(nrad + 1)                  # equal-height rows

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    mesh = ax.pcolormesh(vel_edges, row_edges, img,
                         cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")

    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.7)          # systemic Lya
    if show_window is not None:
        dv = C_KMS * show_window / lya_center
        for s_edge in (-dv, dv):
            ax.axvline(s_edge, color="0.25", lw=0.8, ls=":", alpha=0.6)

    # label each row with its radial range; innermost bin on top
    ax.set_yticks(np.arange(nrad) + 0.5)
    ax.set_yticklabels(
        [f"{np.round(r_edges[i], 3)}–{np.round(r_edges[i + 1], 3)}"
         for i in range(nrad)], fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_ylabel(f"Radial bin [{distance_type}]", fontsize=12)
    ax.set_xlabel(r"Velocity about Ly$\alpha$ [km s$^{-1}$]", fontsize=12)
    ax.set_xlim(vlim)

    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(cbar_label, fontsize=11)

    if title is None:
        title = rf"Ly$\alpha$ radius–velocity map ({method})"
    ax.set_title(title, fontsize=14)

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, ax


def _get_vr_biweight_v(VR_biweight_v, stacks_result):
    """kpc-per-(R/Rvir) factor: explicit arg wins, else stacks_result, else None
    (only the comparison axis needs it; the native axis never does)."""
    if VR_biweight_v is not None:
        return float(VR_biweight_v)
    if stacks_result is not None and stacks_result.get("VR_biweight_v") is not None:
        return float(stacks_result["VR_biweight_v"])
    return None


def _as_vr_err(e):
    """float -> symmetric (e, e); (lo, hi) -> (lo, hi) floats; None -> None."""
    if e is None:
        return None
    if np.ndim(e) == 0:
        return (float(e), float(e))
    lo, hi = e
    return (float(lo), float(hi))


def _get_vr_biweight_e(VR_biweight_error, stacks_result):
    """Uncertainty (kpc) on the sample's virial radius as (minus, plus):
    explicit arg wins (a float = symmetric, or a (minus, plus) pair), else
    stacks_result['VR_biweight_e'] -- the bootstrapped 16/84 band written by
    analysis.run_measurement_suite (or build_stacks' quick estimate before
    that) -- else None (no band drawn)."""
    if VR_biweight_error is not None:
        return _as_vr_err(VR_biweight_error)
    if stacks_result is not None and stacks_result.get("VR_biweight_e") is not None:
        return _as_vr_err(stacks_result["VR_biweight_e"])
    return None


def _resolve_bin_mode(bin_mode, stacks_result):
    """Native radial unit: explicit arg wins, else stacks_result['bin_mode'],
    else 'virial'."""
    bm = bin_mode or (stacks_result or {}).get("bin_mode") or "virial"
    bm = bm.lower()
    if bm == "vr":
        bm = "virial"
    if bm not in ("virial", "kpc", "arcsec"):
        raise ValueError(f"bin_mode must be 'virial', 'kpc', or 'arcsec' (got {bm!r})")
    return bm


def _vr_to_native(r_vr, bin_mode, VR_biweight_v):
    """Convert an R/Rvir value to the NATIVE axis unit (for overlays). Returns
    None when the conversion is unavailable (arcsec, or kpc without VR)."""
    r_vr = np.asarray(r_vr, dtype=float)
    if bin_mode == "virial":
        return r_vr
    if bin_mode == "kpc":
        return r_vr * VR_biweight_v if VR_biweight_v is not None else None
    return None  # arcsec: no clean conversion


_NATIVE_LABEL = {"virial": r"$R/R_{\rm vir}$", "kpc": "Radius [kpc]",
                 "arcsec": "Radius [arcsec]"}


def _radius_points(radial_bins):
    """Geometric bin midpoints + x error bars, in the bins' own (native) unit.
    Rounded to 3 dp to kill float-display artifacts (0.1 -> 0.1000000147)."""
    bins = np.round(np.asarray(radial_bins), 3)
    r_mid = np.sqrt(bins[:-1] * bins[1:])
    if bins[0] == 0:
        r_mid[0] = 0.5 * (bins[0] + bins[1])
    xerr = np.vstack([r_mid - bins[:-1], bins[1:] - r_mid])
    return bins, r_mid, xerr


def _setup_radius_axis(ax, radial_bins, bin_mode, VR_biweight_v, VR_biweight,
                       vr_ticks, xlims, show_vr=True, VR_biweight_error=None,
                       tick_direction=None):
    """
    Put the NATIVE bin unit on the bottom axis and the comparison unit on top:
      virial -> bottom R/Rvir, top kpc (if VR_biweight_v known)
      kpc    -> bottom kpc,    top R/Rvir (if VR_biweight_v known)
      arcsec -> bottom arcsec, NO top axis (nothing to convert to)
    Draws the Rvir reference line (and, if VR_biweight_error is given, a
    shaded +/- error band around it) where it is meaningful. Returns
    (r_mid, xerr) in native units.

    show_vr           : master toggle for the Rvir line + error band (both
                         off together when False). Does not affect the
                         secondary (top) comparison axis.
    VR_biweight_error  : uncertainty on the sample's virial radius, a float
                         (symmetric) or (minus, plus) pair; band spans
                         [R_vir - minus, R_vir + plus]. ALWAYS in kpc (same
                         unit as VR_biweight_v); normally the bootstrapped
                         16/84 from stack.bootstrap_virial_radius.
                         None -> no band drawn. Converted internally to native
                         units: divided by VR_biweight_v for bin_mode='virial'
                         (R/Rvir-space width), used as-is for bin_mode='kpc'.
                         No effect for bin_mode='arcsec' (no clean conversion,
                         same limitation as the Rvir line/top axis there).
    tick_direction     : None (matplotlib default) or 'in'/'out'/'inout' for
                         the secondary (top) axis ticks, so it matches the
                         main axes styled by formatting.apply_tick_style.
    """
    bins, r_mid, xerr = _radius_points(radial_bins)

    ax.set_xscale("log")
    if xlims is None:
        pos = bins[bins > 0]
        if pos.size:
            xlims = (float(pos.min()) * 0.6, float(bins.max()) * 1.6)
    if xlims is not None:
        ax.set_xlim(xlims)
    ax.set_xlabel(_NATIVE_LABEL[bin_mode])
    ax.minorticks_on()
    ax.grid(True, which="major", lw=0.8, alpha=0.9)
    ax.grid(True, which="minor", lw=0.5, alpha=0.4)

    def _rvir_line(xpos):
        ax.axvline(xpos, ls="--", color="0.35", lw=1.5, alpha=0.8)
        ax.text(xpos, 0.95, r"$R_{\rm vir}$", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=15, color="0.25")

    def _rvir_band(xpos, err):
        err = _as_vr_err(err)
        if err is None or not np.all(np.isfinite(err)):
            return
        lo, hi = xpos - err[0], xpos + err[1]     # = the 16th..84th band
        if hi <= lo:
            return
        if lo <= 0:
            lo = xpos * 1e-3   # keep positive for the log x-axis
        ax.axvspan(lo, hi, color="0.35", alpha=0.12, lw=0, zorder=0)

    if bin_mode == "virial":
        ax.set_xticks(vr_ticks)
        ax.set_xticklabels([str(t) for t in vr_ticks])
        if show_vr:
            _rvir_line(1.0)
            if VR_biweight_error is not None and VR_biweight_v:
                _e = _as_vr_err(VR_biweight_error)
                _rvir_band(1.0, (_e[0] / VR_biweight_v, _e[1] / VR_biweight_v))
        if VR_biweight_v is not None:
            secax = ax.secondary_xaxis(
                "top", functions=(lambda r: r * VR_biweight_v,
                                  lambda k: k / VR_biweight_v))
            secax.set_xlabel("Radius [kpc]")
            secax.set_xscale("log")
            if tick_direction:
                secax.tick_params(which="both", direction=tick_direction)
    elif bin_mode == "kpc":
        rvir_kpc = VR_biweight if VR_biweight is not None else VR_biweight_v
        if show_vr and rvir_kpc is not None:
            _rvir_line(rvir_kpc)
            if VR_biweight_error is not None:
                _rvir_band(rvir_kpc, VR_biweight_error)
        if VR_biweight_v is not None:
            secax = ax.secondary_xaxis(
                "top", functions=(lambda k: k / VR_biweight_v,
                                  lambda r: r * VR_biweight_v))
            secax.set_xlabel(r"$R/R_{\rm vir}$")
            secax.set_xscale("log")
            if tick_direction:
                secax.tick_params(which="both", direction=tick_direction)
            secax.set_xticks(vr_ticks)
            secax.set_xticklabels([str(t) for t in vr_ticks])
    # arcsec: native bottom only -- no Rvir line/band (no clean conversion)

    return r_mid, xerr


def _safe_yerr(center, lo, hi):
    """
    Asymmetric errorbar lengths from a bootstrap 16/84 band, guarded against
    the plotted center falling outside its own band.

    matplotlib's errorbar raises ValueError on a negative yerr length. That
    happens whenever `center` is NOT itself a statistic of the [lo, hi]
    percentile distribution -- e.g. plotting the FIDUCIAL (full-sample) point
    estimate against BOOTSTRAP percentiles. The two are only guaranteed
    consistent when center is the bootstrap median (p16 <= p50 <= p84 always
    holds for percentiles of the same array); a fiducial point estimate has no
    such guarantee and can land outside its own band, especially for a
    ratio-type statistic (e.g. blue/(blue+red)) in a faint bin where the
    denominator is near zero. Same guard convention as validation.py's
    injection-recovery plot (`np.clip(rec - lo, 0, None)`), centralized here
    and paired with a mask so the offending points can be flagged rather than
    silently clipped.

    Returns
    -------
    yerr     : (2, N) array [center-lo, hi-center], each floored at 0 -- safe
               to pass straight to ax.errorbar(..., yerr=yerr).
    unstable : (N,) bool -- True where center was outside [lo, hi] (before
               flooring) at a finite point, i.e. where flooring actually did
               something. Caller should mark these (e.g. hollow marker) so the
               instability stays visible instead of vanishing into a clipped
               bar. NaN entries (missing/failed bins) are never flagged.
    """
    center = np.asarray(center, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    lo_err = center - lo
    hi_err = hi - center
    finite = np.isfinite(lo_err) & np.isfinite(hi_err)
    unstable = finite & ((lo_err < 0) | (hi_err < 0))
    yerr = np.vstack([np.clip(lo_err, 0, None), np.clip(hi_err, 0, None)])
    return yerr, unstable


def plot_centroid_vs_radius(
    boot, radial_bins, VR_biweight_v=None, VR_biweight=None,
    stacks_result=None, bin_mode=None,
    z_err_kms=0, vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    figsize=(7.2, 4.7), title=r"Bootstrapped Ly$\alpha$ centroid vs. radius",
    label="Bootstrapped centroid",
    ylims=(-200, 200),
    xlims=None,
    show_vr=True,
    VR_biweight_error=None,
    add_point=None,
    add_axhline=None,
    plot_literature=False,
    tick_direction="in",       # house tick style (formatting.apply_tick_style)
    tick_right=True,
    save_fig=False,
    savename="Figure_centroid_profile.png",
):
    """
    Headline Stage-3 figure: centroid velocity per radial bin with its bootstrap
    16/84 interval. The bottom x-axis is the NATIVE bin unit (config.bin_mode);
    the top axis is the comparison unit (kpc<->R/Rvir), omitted for arcsec.

    boot          : dict with 'centroid_v_fid', 'centroid_v_lo', 'centroid_v_hi'
                    (e.g. run_measure / bootstrap_measurements output). The
                    plotted point is the FIDUCIAL (full, unresampled sample)
                    centroid, matching print_centroid_table / plot_centroid_methods;
                    only the 16/84 error bars come from the bootstrap.
    radial_bins   : bin edges in the NATIVE unit (stacks['r_edges']), length nrad+1.
    bin_mode      : 'virial' | 'kpc' | 'arcsec'. If None, read from stacks_result.
    VR_biweight_v : kpc per (R/Rvir). Needed ONLY for the comparison top axis
                    (and the Guo overlay); if None it is read from stacks_result,
                    and if still unknown the top axis is simply omitted.
    xlims         : in NATIVE units now (None -> auto from the bins).
    z_err_kms     : systemic-z band (km/s); off by default for a stack.
    show_vr       : draw the dashed Rvir reference line (and its error band,
                    if any). False hides both.
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                    radius; draws a shaded band around the Rvir line. None
                    (default) -> read stacks_result['VR_biweight_e'] (set by
                    stack.sample_virial_radius_kpc) if available, else no band.

    Returns (fig, ax).
    """
    bin_mode = _resolve_bin_mode(bin_mode, stacks_result)
    VR_biweight_v = _get_vr_biweight_v(VR_biweight_v, stacks_result)
    VR_biweight_error = _get_vr_biweight_e(VR_biweight_error, stacks_result)

    v_fid = np.asarray(boot["centroid_v_fid"])
    v_lo = np.asarray(boot["centroid_v_lo"])
    v_hi = np.asarray(boot["centroid_v_hi"])
    yerr, unstable = _safe_yerr(v_fid, v_lo, v_hi)

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bin_mode, VR_biweight_v,
                                     VR_biweight, vr_ticks, xlims,
                                     show_vr=show_vr, VR_biweight_error=VR_biweight_error,
                                     tick_direction=tick_direction)
    eb = ax.errorbar(r_mid, v_fid, xerr=xerr, yerr=yerr, fmt="o",
                     capsize=3.5, ms=6, lw=1.5, label=label)
    if np.any(unstable):
        ax.scatter(r_mid[unstable], v_fid[unstable], s=70, facecolors="none",
                   edgecolors=eb[0].get_color(), linewidths=1.3, zorder=5,
                   label="fiducial outside 16–84 band")

    if add_point is not None:
        for i in range(add_point.shape[0]):
            ax.scatter(add_point[i][0], add_point[i][1], c="tomato", s=30,
                       label="External Data")
    if add_axhline is not None:
        for i in range(len(add_axhline[0])):
            ax.axhline(add_axhline[0][i], c=add_axhline[2][i], label=add_axhline[1][i])

    if plot_literature:
        # Guo+2024's own numbers, in Guo's own units -- deliberately NEVER
        # run through THIS sample's VR_biweight_v. Guo's virial radius (20
        # kpc, guo_VR below) is a property of Guo's own galaxy sample, not
        # ours; using our Rvir to place Guo's points would silently rescale
        # them into our sample's frame instead of showing what Guo actually
        # measured. So: bin_mode='kpc' -> Guo's raw kpc values, unconverted.
        # bin_mode='virial' -> Guo's own R/Rvir (kpc / Guo's 20 kpc Rvir).
        # Two independent unit choices, each self-contained to Guo's data.
        guo_VR = 20  # Guo+2024's own assumed/measured virial radius (kpc)
        guo_r_edges_kpc = np.array([0, 7, 14, 29, 59])
        guo_r_mid_kpc = np.array([3.5, 10.5, 21.5, 44])
        guo_v = np.array([174, 173, 81, -72])
        guo_ev = np.array([11, 15, 51, 57])

        if bin_mode == "kpc":
            gx, g_edges = guo_r_mid_kpc, guo_r_edges_kpc
        elif bin_mode == "virial":
            gx, g_edges = guo_r_mid_kpc / guo_VR, guo_r_edges_kpc / guo_VR
        else:
            gx = g_edges = None

        if gx is not None:
            lo = gx - g_edges[:-1]
            hi = g_edges[1:] - gx
            ax.errorbar(gx, guo_v, xerr=np.vstack([lo, hi]), yerr=guo_ev,
                        fmt="s--", capsize=3, ms=5, lw=1.2,
                        label="Guo+2024 LAEs, median stack")
        else:
            print("plot_literature skipped: no clean Guo overlay for "
                  f"bin_mode={bin_mode!r} -- 'kpc' uses Guo's own base kpc "
                  "values, 'virial' uses Guo's own Rvir=20 kpc; arcsec has "
                  "neither.")

    ax.axhline(0, color="tomato", alpha=0.4, lw=1)
    if z_err_kms and z_err_kms > 0:
        ax.axhspan(-z_err_kms, z_err_kms, color="tomato", alpha=0.1, lw=0.5,
                   label=r"$z_{\rm sys}$ uncertainty")

    ax.set_ylim(ylims)
    ax.set_ylabel(r"Ly$\alpha$ centroid velocity [km s$^{-1}$]")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=10)
    apply_tick_style(ax, direction=tick_direction, right=tick_right)
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_blue_red_vs_radius(
    boot, radial_bins, VR_biweight_v=None, VR_biweight=None,
    stacks_result=None, bin_mode=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    figsize=(7.6, 5.0),
    title=r"Blue vs. red side flux vs. radius",
    ylims=None, xlims=None,
    show_ratio=True, use_fiducial=False,
    logy=False, floor=None,
    save_fig=False,
):
    """
    Overlay summed continuum-subtracted BLUE-side and RED-side flux per radial
    bin (bootstrap 16/84 bars) so you can see where the two diverge with radius;
    optionally overlay the blue/red ratio on a right-hand axis. Same native-unit
    bottom axis / comparison top axis as plot_centroid_vs_radius.

    Consumes bootstrap_measurements side-flux keys: blue_flux_med/lo/hi (+ _fid),
    red_flux_med/lo/hi (+ _fid), blue_over_red_med/lo/hi (+ _fid).

    logy : if True, set the MAIN (flux) axis to log so you can see where a side
           dips low. Summed continuum-subtracted flux can go non-positive (noise
           or genuine absorption), which log cannot show, so non-positive central
           values and lower error bounds are FLOORED to a small positive value
           (`floor`, default = 1e-3 x the largest plotted flux). The secondary
           blue/red RATIO axis is left linear regardless. Off by default.

    Returns (fig, ax) if show_ratio is False, else (fig, ax, ax_ratio).
    """
    bin_mode = _resolve_bin_mode(bin_mode, stacks_result)
    VR_biweight_v = _get_vr_biweight_v(VR_biweight_v, stacks_result)

    suff = "_fid" if use_fiducial else "_med"
    blue = np.asarray(boot["blue_flux" + suff]); red = np.asarray(boot["red_flux" + suff])
    blue_lo = np.asarray(boot["blue_flux_lo"]); blue_hi = np.asarray(boot["blue_flux_hi"])
    red_lo = np.asarray(boot["red_flux_lo"]); red_hi = np.asarray(boot["red_flux_hi"])

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bin_mode, VR_biweight_v,
                                     VR_biweight, vr_ticks, xlims)

    if logy:
        # floor non-positive central values + lower bounds so log can render them;
        # error bars whose lower bound is floored show as reaching the floor.
        all_pos = np.concatenate([blue, red, blue_hi, red_hi])
        all_pos = all_pos[np.isfinite(all_pos) & (all_pos > 0)]
        fl = floor if floor is not None else (
            (np.nanmax(all_pos) * 1e-3) if all_pos.size else 1e-30)
        blue = np.where(np.isfinite(blue) & (blue > fl), blue, fl)
        red = np.where(np.isfinite(red) & (red > fl), red, fl)
        blue_lo = np.where(np.isfinite(blue_lo) & (blue_lo > fl), blue_lo, fl)
        red_lo = np.where(np.isfinite(red_lo) & (red_lo > fl), red_lo, fl)
        blue_hi = np.where(np.isfinite(blue_hi) & (blue_hi > fl), blue_hi, fl)
        red_hi = np.where(np.isfinite(red_hi) & (red_hi > fl), red_hi, fl)
        ax.set_yscale("log")

    blue_yerr, blue_unstable = _safe_yerr(blue, blue_lo, blue_hi)
    red_yerr, red_unstable = _safe_yerr(red, red_lo, red_hi)
    ax.errorbar(r_mid, blue, xerr=xerr, yerr=blue_yerr,
                fmt="o-", color="tab:blue", capsize=3.5, ms=6, lw=1.5, label="Blue-side flux")
    ax.errorbar(r_mid, red, xerr=xerr, yerr=red_yerr,
                fmt="s-", color="tab:red", capsize=3.5, ms=6, lw=1.5, label="Red-side flux")
    if np.any(blue_unstable):
        ax.scatter(r_mid[blue_unstable], blue[blue_unstable], s=70, facecolors="none",
                   edgecolors="tab:blue", linewidths=1.3, zorder=5)
    if np.any(red_unstable):
        ax.scatter(r_mid[red_unstable], red[red_unstable], s=70, facecolors="none",
                   edgecolors="tab:red", linewidths=1.3, zorder=5)
    if not logy:
        ax.axhline(0, color="0.5", lw=1, alpha=0.7)

    y_unit = (stacks_result.get("unit_info") or {}).get("y_unit", "") if stacks_result else ""
    ax.set_ylabel(f"Summed (flux - continuum) [{y_unit}]" if y_unit
                  else "Summed (flux - continuum)")
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_title(title)

    ax_ratio = None
    if show_ratio:
        ratio = np.asarray(boot["blue_over_red" + suff])
        r_lo = np.asarray(boot["blue_over_red_lo"]); r_hi = np.asarray(boot["blue_over_red_hi"])
        ax_ratio = ax.twinx()
        ratio_yerr, ratio_unstable = _safe_yerr(ratio, r_lo, r_hi)
        ax_ratio.errorbar(r_mid, ratio, yerr=ratio_yerr,
                          fmt="D:", color="0.35", ms=4, lw=1, capsize=2.5, alpha=0.8,
                          label="blue / red")
        if np.any(ratio_unstable):
            ax_ratio.scatter(r_mid[ratio_unstable], ratio[ratio_unstable], s=50,
                             facecolors="none", edgecolors="0.35", linewidths=1.2, zorder=5)
        ax_ratio.axhline(1.0, color="0.6", ls=":", lw=1, alpha=0.7)
        ax_ratio.set_ylabel("blue / red", color="0.35")
        ax_ratio.tick_params(axis="y", labelcolor="0.35")

    handles, labels = ax.get_legend_handles_labels()
    if ax_ratio is not None:
        h2, l2 = ax_ratio.get_legend_handles_labels()
        handles += h2; labels += l2
    ax.legend(handles, labels, frameon=False, fontsize=10, loc="best")

    if save_fig:
        plt.savefig("Figure_blue_red.png", dpi=300)
    plt.show()
    return (fig, ax, ax_ratio) if show_ratio else (fig, ax)


def plot_centroid_comparison(
    boots, radial_bins=None, VR_biweight_v=None, VR_biweight=None,
    stacks_result=None, bin_mode=None, vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    figsize=(7.6, 4.9), ylims=(-200, 200), xlims=None, jitter=0.04,
    title="Centroid vs. radius by sample", save_fig=False,
):
    """
    Overlay centroid-vs-radius for SEVERAL labelled samples on one axis -- the
    multi-sample companion to plot_centroid_vs_radius (e.g. low-z vs high-z, or
    different catalogs). Same native-unit bottom / comparison top axis as every
    other radial figure here, for visual consistency.

    boots        : dict {label: boot}, each boot a bootstrap_measurements /
                   run_measure output (needs centroid_v_fid/lo/hi). Plots the
                   FIDUCIAL centroid per sample, bootstrap 16/84 error bars.
    radial_bins  : bin edges (native unit). If None, taken from the first boot's
                   'r_edges' (run_measure carries it).
    bin_mode / VR_biweight_v : default from the first boot if it carries them.
    jitter       : multiplicative x-offset per curve on the log axis so the
                   error bars stay legible (0 disables).

    Returns (fig, ax).
    """
    labels = list(boots.keys())
    first = boots[labels[0]]
    bm = _resolve_bin_mode(bin_mode, first if stacks_result is None else stacks_result)
    vr = _get_vr_biweight_v(VR_biweight_v,
                            first if stacks_result is None else stacks_result)
    if radial_bins is None:
        radial_bins = np.asarray(first["r_edges"])

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, _ = _setup_radius_axis(ax, radial_bins, bm, vr, VR_biweight, vr_ticks, xlims)
    K = len(labels)
    for k, lab in enumerate(labels):
        b = boots[lab]
        v = np.asarray(b["centroid_v_fid"])
        lo = np.asarray(b["centroid_v_lo"]); hi = np.asarray(b["centroid_v_hi"])
        jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
        yerr, unstable = _safe_yerr(v, lo, hi)
        eb = ax.errorbar(jit, v, yerr=yerr, fmt="o-",
                         capsize=3, ms=5, lw=1.3, label=lab)
        if np.any(unstable):
            ax.scatter(jit[unstable], v[unstable], s=70, facecolors="none",
                       edgecolors=eb[0].get_color(), linewidths=1.3, zorder=5)
    ax.axhline(0, color="tomato", alpha=0.4, lw=1)
    ax.set_ylim(ylims)
    ax.set_ylabel(r"Ly$\alpha$ centroid velocity [km s$^{-1}$]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, title="sample")
    if save_fig:
        plt.savefig("Figure_centroid_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# REUSABLE RADIAL ENGINES  (one home for the shared radius axis + style)
# =====================================================================
def plot_radial_overlay(
    series, radial_bins, bin_mode=None, VR_biweight_v=None, VR_biweight=None,
    stacks_result=None, vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.6, 4.9),
    ylims=None, xlims=None, ylabel="", title="", jitter=0.04, zero_line=0.0,
    hline=None, legend_title=None, logy=False, save_fig=False, savename=None,
    show_vr=True, VR_biweight_error=None, tick_direction="in", tick_right=True,
):
    """
    Overlay several labelled curves on the standard native-bottom / comparison-top
    radius axis. The shared engine behind the line-window sweep, the continuum
    sweep, and the sample comparison, so they all look identical.

    series : list of dicts {label, y, [lo], [hi], [fmt]}; lo/hi optional (drawn
             as 16/84 bars when present). Curves are multiplicatively jittered on
             the log axis so error bars stay legible (jitter=0 disables).
    show_vr, VR_biweight_error : Rvir line / error band, same as the single-sample
             profile plots (VR_biweight_error None -> auto-read from stacks_result).
    tick_direction, tick_right : house tick style (formatting.apply_tick_style) --
             ticks "in", y ticks mirrored on the right; top axis matches.
    """
    bm = _resolve_bin_mode(bin_mode, stacks_result)
    vr = _get_vr_biweight_v(VR_biweight_v, stacks_result)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks_result)
    fig, ax = plt.subplots(figsize=figsize)
    r_mid, _ = _setup_radius_axis(ax, radial_bins, bm, vr, VR_biweight, vr_ticks, xlims,
                                  show_vr=show_vr, VR_biweight_error=vr_e,
                                  tick_direction=tick_direction)
    K = len(series)
    for k, s in enumerate(series):
        y = np.asarray(s["y"], dtype=float)
        jit = r_mid * (1 + jitter * (k - (K - 1) / 2.0)) if jitter else r_mid
        lo, hi = s.get("lo"), s.get("hi")
        if lo is not None and hi is not None:
            lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
            yerr, unstable = _safe_yerr(y, lo, hi)
            eb = ax.errorbar(jit, y, yerr=yerr, fmt=s.get("fmt", "o-"),
                             capsize=3, ms=5, lw=1.3, label=s["label"])
            if np.any(unstable):
                ax.scatter(jit[unstable], y[unstable], s=70, facecolors="none",
                          edgecolors=eb[0].get_color(), linewidths=1.3, zorder=5)
        else:
            ax.errorbar(jit, y, fmt=s.get("fmt", "o-"), ms=5, lw=1.3, label=s["label"])
    if zero_line is not None and not logy:
        ax.axhline(zero_line, color="tomato", alpha=0.4, lw=1)
    if hline is not None:
        ax.axhline(hline, color="0.6", ls=":", lw=1, alpha=0.7)
    if logy:
        ax.set_yscale("log")
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, title=legend_title)
    apply_tick_style(ax, direction=tick_direction, right=tick_right)
    if save_fig:
        plt.savefig(savename or "Figure_radial_overlay.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_null_band(
    radial_bins, null_lo, null_hi, null_med=None, real_y=None, real_lo=None,
    real_hi=None, bin_mode=None, VR_biweight_v=None, stacks_result=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.4, 4.8), ylims=None, xlims=None,
    ylabel="", title="", null_label="null (16–84%)", real_label="real",
    snr=None, logy=False, save_fig=False, savename=None,
):
    """
    Grey null envelope (+ optional null median) with the real points overlaid --
    the shared engine behind the z-scramble null, the placebo null, and the
    blank-sky null. If `snr` (per bin) is given, each real point is annotated
    with its sigma.
    """
    bm = _resolve_bin_mode(bin_mode, stacks_result)
    vr = _get_vr_biweight_v(VR_biweight_v, stacks_result)
    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims)
    ax.fill_between(r_mid, np.asarray(null_lo, float), np.asarray(null_hi, float),
                    color="0.6", alpha=0.30, lw=0, label=null_label)
    if null_med is not None:
        ax.plot(r_mid, np.asarray(null_med, float), color="0.45", lw=1, ls="--", alpha=0.8)
    if not logy:
        ax.axhline(0, color="tomato", alpha=0.4, lw=1)
    if real_y is not None:
        real_y = np.asarray(real_y, dtype=float)
        if real_lo is not None and real_hi is not None:
            real_lo = np.asarray(real_lo, float)
            real_hi = np.asarray(real_hi, float)
            # Clip to >= 0: matplotlib requires non-negative yerr values.
            # The fiducial (real_y) is the FULL-SAMPLE measurement, not the
            # bootstrap median, so it can land outside [real_lo, real_hi] in
            # noisy outer bins. That is informative (not a bug), but a zero bar
            # on that side is the right display choice rather than a crash.
            yerr_lo = np.maximum(real_y - real_lo, 0.0)
            yerr_hi = np.maximum(real_hi - real_y, 0.0)
            # warn if any bin actually had the fiducial outside the band
            bad = np.where(
                (real_y < real_lo) | (real_y > real_hi)
            )[0]
            if len(bad):
                import warnings as _w
                _w.warn(
                    f"plot_null_band: fiducial real_y falls outside bootstrap "
                    f"[lo, hi] in {len(bad)} bin(s) {bad.tolist()} -- "
                    f"zeroing that error bar side. "
                    f"This usually means the outer-bin flux is near zero and "
                    f"the bootstrap distribution is asymmetric.",
                    stacklevel=3,
                )
            ax.errorbar(r_mid, real_y, xerr=xerr,
                        yerr=np.vstack([yerr_lo, yerr_hi]),
                        fmt="o", capsize=3.5, ms=6, lw=1.5,
                        color="tab:blue", label=real_label)
        else:
            ax.errorbar(r_mid, real_y, xerr=xerr, fmt="o", capsize=3.5, ms=6, lw=1.5,
                        color="tab:blue", label=real_label)
        if snr is not None:
            for x, y, s in zip(r_mid, real_y, np.asarray(snr, float)):
                if np.isfinite(y) and np.isfinite(s):
                    ax.annotate(f"{s:.1f}σ", (x, y), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=8, color="0.3")
    if logy:
        ax.set_yscale("log")
        if ylims is None:
            # auto-range using only positive values from both the null band and
            # the real points -- matplotlib's log-scale autoscale fails silently
            # when any value is <= 0
            pos_vals = []
            for arr in (null_lo, null_hi, real_y):
                if arr is not None:
                    a = np.asarray(arr, float).ravel()
                    pos_vals.extend(a[np.isfinite(a) & (a > 0)].tolist())
            if pos_vals:
                vmin = min(pos_vals)
                vmax = max(pos_vals)
                ax.set_ylim(vmin * 0.3, vmax * 4.0)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="best")
    if save_fig:
        plt.savefig(savename or "Figure_null_band.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# NEW DIAGNOSTIC PLOTS
# =====================================================================
def _finite_galaxies_per_bin(stacks_or_product):
    """N galaxies with any finite flux per radial bin, from a stacks dict
    (cube_flux) or a GalaxyProduct (spec)."""
    if isinstance(stacks_or_product, dict):
        cube = stacks_or_product.get("cube_flux")
        if cube is None:
            raise KeyError("stacks has no cube_flux; build with keep_cube=True "
                           "or pass the GalaxyProduct.")
        r_edges = np.asarray(stacks_or_product["r_edges"])
    else:
        cube = stacks_or_product.spec
        r_edges = np.asarray(stacks_or_product.r_edges)
    cube = np.asarray(cube, dtype=float)
    n = np.array([int(np.sum(np.any(np.isfinite(cube[:, r, :]), axis=1)))
                  for r in range(cube.shape[1])])
    return n, r_edges


def plot_effective_n_per_bin(
    stacks_or_product, bin_mode=None, VR_biweight_v=None, stacks_result=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5), figsize=(7.4, 3.8), title="Effective N per radial bin",
    save_fig=False, savename=None,
):
    """
    Bar of the number of galaxies actually contributing (any finite flux) to each
    radial bin, on the standard radius axis. With cut_radial_bin=-1 the inner
    bins are not finiteness-filtered, so this shows how many objects each inner
    point really rests on -- worth showing under the centroid figure.
    """
    n, r_edges = _finite_galaxies_per_bin(stacks_or_product)
    sr = stacks_result if stacks_result is not None else (
        stacks_or_product if isinstance(stacks_or_product, dict) else None)
    bm = _resolve_bin_mode(bin_mode, sr)
    vr = _get_vr_biweight_v(VR_biweight_v, sr)
    fig, ax = plt.subplots(figsize=figsize)
    r_mid, _ = _setup_radius_axis(ax, r_edges, bm, vr, None, vr_ticks, None)
    lo = np.asarray(r_edges[:-1], float); hi = np.asarray(r_edges[1:], float)
    widths = (hi - lo) * 0.8
    lo_plot = np.where(lo > 0, lo, hi / 2.0)
    ax.bar(r_mid, n, width=np.maximum(widths, r_mid * 0.3), align="center",
           color="0.6", edgecolor="0.3", alpha=0.85)
    for x, c in zip(r_mid, n):
        ax.annotate(str(c), (x, c), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color="0.2")
    ax.set_ylabel("N galaxies (finite)")
    ax.set_ylim(0, max(n) * 1.18 if len(n) and max(n) > 0 else 1)
    ax.set_title(title)
    if save_fig:
        plt.savefig(savename or "Figure_effective_N.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax, n


def plot_contributing_spectra(
    stacks, per_bin=True, radial_index=None, lya_center=1215.67, mark_lya=True,
    figsize=(8.6, 4.6), title="Contributing galaxies vs. rest wavelength",
    save_fig=False, savename=None,
):
    """
    Number of galaxies with FINITE (real, post-mask, post-resample) flux at each
    REST wavelength -- the true effective N(lambda) that actually went into the
    stack. Computed from the per-galaxy cube (needs keep_cube=True). The dropoff
    at the rest-frame edges is the redshift spread; dips inside it are masking /
    chip gaps. per_bin=True draws one curve per radial bin; otherwise a single
    bin (radial_index) or the all-bin union.
    """
    cube = stacks.get("cube_flux")
    if cube is None:
        raise KeyError("plot_contributing_spectra needs cube_flux (keep_cube=True).")
    cube = np.asarray(cube, dtype=float)
    wave = np.asarray(stacks["rest_wave"], dtype=float)
    r_edges = np.round(np.asarray(stacks["r_edges"]), 3)
    nrad = cube.shape[1]
    finite = np.isfinite(cube)                      # (ngal, nrad, nwave)

    fig, ax = plt.subplots(figsize=figsize)
    if per_bin:
        for r in range(nrad):
            n_lambda = finite[:, r, :].sum(axis=0)
            ax.step(wave, n_lambda, where="mid", lw=1.1,
                    label=f"{r_edges[r]:g}–{r_edges[r+1]:g}")
        ax.legend(frameon=False, fontsize=8, ncol=2, title="radial bin")
    else:
        if radial_index is None:
            n_lambda = (finite.any(axis=1)).sum(axis=0)        # union over bins
            lab = "any bin"
        else:
            n_lambda = finite[:, radial_index, :].sum(axis=0)
            lab = f"{r_edges[radial_index]:g}–{r_edges[radial_index+1]:g}"
        ax.step(wave, n_lambda, where="mid", lw=1.3, color="black", label=lab)
        ax.legend(frameon=False, fontsize=9)
    if mark_lya:
        ax.axvline(lya_center, color="magenta", lw=0.9, ls=":", alpha=0.7)
        ax.text(lya_center, 0.98, r"Ly$\alpha$", transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=8, color="magenta")
    ax.set_xlabel(r"Rest-frame wavelength [$\AA$]")
    ax.set_ylabel("N contributing galaxies")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    if save_fig:
        plt.savefig(savename or "Figure_contributing_spectra.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_significance_comparison(conf, boot=None, centroid_v=None, bin_unit="virial",
                                 figsize=(7.6, 4.8), sigma_levels=(1, 2, 3),
                                 reliable_sn=2.0, signed_centroid=True,
                                 title=None, save_fig=False, savename=None):
    """
    Compare the two per-bin significances on one sigma axis vs radius:
      flux_sn       -- detection S/N of the integrated Lya flux ("is there signal?")
      nsigma_offset -- significance of the centroid velocity offset from 0
                       ("is the line kinematically shifted?")
    Both are in sigma, so they share a y-axis. A centroid offset is only
    trustworthy where the flux is detected, so bins with flux_sn < reliable_sn
    get hollow centroid markers.

    Primary call is plot_significance_comparison(conf, boot): conf supplies both
    significances (flux_sn, nsigma_offset) and boot supplies the centroid SIGN.
    conf["nsigma_offset"] is a MAGNITUDE (|v_fid| / err_to_zero) and cannot tell a
    blueshift from a redshift on its own, so with signed_centroid=True the centroid
    significance is drawn with the sign of the centroid velocity: blueshifted bins
    (v_fid < 0) fall BELOW zero and you can watch the line flip sign with radius.
    The sign is taken from (in order) centroid_v=, then boot["centroid_v_fid"],
    then a signed key already in conf. Detection flux_sn is a magnitude by nature
    and is always drawn positive. Set signed_centroid=False for the old
    magnitude-only view.

    Parameters
    ----------
    conf : dict
        Output of analysis.measurement_confidence -- needs "r_edges", "flux_sn",
        "nsigma_offset".
    boot : dict, optional
        The bootstrap dict the conf was built from; read for "centroid_v_fid" to
        sign the centroid curve. The intended primary argument.
    centroid_v : array, optional
        Explicit per-bin centroid velocities (km/s); overrides boot for the sign.
    signed_centroid : bool
        If True (default) draw the centroid significance signed; if False draw the
        |nsigma| magnitude as before.

    Returns
    -------
    (fig, ax)
    """
    edges = np.asarray(conf["r_edges"], float)
    fsn   = np.asarray(conf["flux_sn"], float)
    noff  = np.asarray(conf["nsigma_offset"], float)      # magnitude (>= 0)

    # --- recover the centroid sign so the curve can dip negative ------------
    v_src = None
    if centroid_v is not None:
        v_src = np.asarray(centroid_v, float)
    elif boot is not None and "centroid_v_fid" in boot:
        v_src = np.asarray(boot["centroid_v_fid"], float)
    else:
        for k in ("centroid_v_fid", "v_fid", "centroid_v"):
            if k in conf:
                v_src = np.asarray(conf[k], float)
                break

    if signed_centroid and v_src is not None:
        sgn = np.where(v_src < 0, -1.0, 1.0)              # 0 and + map to +1
        noff_plot = sgn * noff
        signed = True
    else:
        if signed_centroid and v_src is None:
            warnings.warn("plot_significance_comparison: no centroid sign source "
                          "(pass boot=... or centroid_v=...); showing |nsigma|.",
                          stacklevel=2)
        noff_plot = noff
        signed = False

    left, right = edges[:-1].copy(), edges[1:]
    left[left <= 0] = right[left <= 0] / 3.0              # floor 0 edge for log axis
    ctr  = np.sqrt(left * right)
    xerr = np.vstack([ctr - left, right - ctr])

    fig, ax = plt.subplots(figsize=figsize)
    band_lo = -1 if signed else 0
    ax.axhspan(band_lo, 1, color="0.85", alpha=0.5, zorder=0)   # "not significant" band
    ax.axhline(0, color="k", lw=1, zorder=1)

    # sigma guide lines, mirrored below zero when signed
    for s in sigma_levels:
        c  = "tomato" if s == 1 else "0.6"
        ls = "--" if s == 1 else ":"
        for lev in ((s, -s) if signed else (s,)):
            ax.axhline(lev, color=c, ls=ls, lw=1, alpha=0.8, zorder=1)
            ax.text(right[-1], lev, f"{'+' if lev > 0 else ''}{lev:g}\u03c3",
                    va="bottom", ha="right", fontsize=8, color=c)

    ax.errorbar(ctr, fsn, xerr=xerr, fmt="o-", ms=6, lw=1.5, capsize=3,
                color="tab:blue", label="flux S/N  (detection)", zorder=3)
    lbl_c = ("centroid offset  (signed n\u03c3)" if signed
             else "centroid offset  (n\u03c3 from 0)")
    ax.errorbar(ctr, noff_plot, xerr=xerr, fmt="s--", ms=6, lw=1.5, capsize=3,
                color="crimson", label=lbl_c, zorder=3)

    weak = fsn < reliable_sn
    if weak.any():
        ax.plot(ctr[weak], noff_plot[weak], "s", ms=9, mfc="white", mec="crimson",
                mew=1.4, zorder=4, label=f"centroid where flux S/N < {reliable_sn:g}")

    ax.set_xscale("log")
    ax.set_xlabel(f"projected radius [{bin_unit}]")
    ax.set_ylabel(r"significance [$\sigma$]")
    ax.set_title(title or "Detection vs. centroid-offset significance per bin")

    # y-limits that include the negative excursions and the lowest sigma guide
    yvals = np.concatenate([fsn, noff_plot, np.asarray(sigma_levels, float),
                            -np.asarray(sigma_levels, float) if signed
                            else np.array([])])
    yvals = yvals[np.isfinite(yvals)]
    ylo = min(band_lo - 0.5, yvals.min() - 0.5) if yvals.size else band_lo - 0.5
    yhi = (yvals.max() + 0.7) if yvals.size else 1.0
    ax.set_ylim(ylo, yhi)

    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    if save_fig:
        fig.savefig(savename or "Figure_significance_comparison.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# =====================================================================
# INTRINSIC-PROFILE / LSF DECONVOLUTION DIAGNOSTIC
# =====================================================================

def plot_intrinsic_fit(rest_wave, stack_flux, stack_err, decon_result,
                       lsf_result=None, lya_center=1215.67, xlim=None,
                       show_both=True, show_intrinsic=True, show_lsf=True,
                       boot=None, title=None, figsize=(7.6, 5.8),
                       save_fig=False, savename="Figure_intrinsic_fit.png"):
    """
    Diagnostic for speclsf.deconvolve_intrinsic_profile: overlay the stacked line,
    the selected (intrinsic (x) LSF) model, the un-convolved INTRINSIC profile, and
    the LSF for scale, with a residual panel below.

    Accepts either form of decon_result:
      * model='both'   -> {selected, single, double, comparison}
      * model='single'/'double' -> a single fit dict (has model_name, model, ...)

    Parameters
    ----------
    rest_wave, stack_flux, stack_err : the bin's stacked line and per-pixel error
    decon_result : output of speclsf.deconvolve_intrinsic_profile
    lsf_result   : output of speclsf.stack_lsf (only used to draw the LSF for scale)
    show_both    : if a non-selected fit exists, draw it faint for comparison
    show_intrinsic : draw the deconvolved (un-convolved) intrinsic profile
    boot         : optional speclsf.bootstrap_intrinsic_profile result, to annotate
                   sigma_int / peak_sep CIs and the double_fraction

        from utils_lya_halo import speclsf, plotting
        lsf  = speclsf.stack_lsf(cfg, product, z_err_kms=60)
        flux = stacks['stacks']['biweight']['flux'][b]
        ferr = boot['stack_error']['stack_err'][b]
        res  = speclsf.deconvolve_intrinsic_profile(stacks['rest_wave'], flux, ferr, lsf)
        plotting.plot_intrinsic_fit(stacks['rest_wave'], flux, ferr, res, lsf_result=lsf)

    Returns (fig, (ax, ax_resid)).
    """
    rest_wave = np.asarray(rest_wave, float)
    data = np.asarray(stack_flux, float)
    err = np.asarray(stack_err, float)

    # normalize result into a primary fit (+ the alternative, if any)
    if isinstance(decon_result, dict) and "selected" in decon_result:
        sel = decon_result["selected"]
        primary = decon_result.get(sel)
        single = decon_result.get("single")
        double = decon_result.get("double")
        comp = decon_result.get("comparison", {})
        other = double if sel == "single" else single
    else:
        primary = decon_result
        sel = decon_result.get("model_name")
        comp = {}
        other = None
    if primary is None or not primary.get("success"):
        raise ValueError("deconvolution result has no successful fit to plot")

    if xlim is None:
        xlim = (lya_center - 7, lya_center + 7)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

    ax.fill_between(rest_wave, data - err, data + err, color="0.85", lw=0,
                    label=r"$\pm1\sigma$ (stack err)")
    ax.step(rest_wave, data, where="mid", color="k", lw=1.1, label="stacked line")
    ax.plot(rest_wave, primary["model"], color="crimson", lw=1.7,
            label=f"{sel} $\\otimes$ LSF (selected)")
    if show_both and other is not None and other.get("success"):
        ax.plot(rest_wave, other["model"], color="crimson", lw=1.0, ls=":",
                alpha=0.55, label=f"{other['model_name']} $\\otimes$ LSF")
    if show_intrinsic and "intrinsic" in primary:
        ax.plot(rest_wave, primary["intrinsic"], color="tab:blue", lw=1.5, ls="--",
                label=f"intrinsic ({sel}, deconvolved)")
    if show_lsf and lsf_result is not None:
        k = np.asarray(lsf_result["lsf"], float)
        kmax = np.nanmax(k)
        if kmax > 0:
            kk = k / kmax * np.nanmax(data) * 0.45
            ax.plot(np.asarray(lsf_result["rest_wave"], float), kk, color="0.55",
                    lw=1.0, alpha=0.8, label="LSF (scaled)")

    ax.axvline(lya_center, color="magenta", lw=0.7, ls=":")
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlim(xlim)
    ax.set_ylabel("flux")

    # annotate selection stats + bootstrap CIs
    leg_title = None
    if comp:
        leg_title = (f"F p={comp.get('f_pvalue', float('nan')):.2g}, "
                     f"$\\Delta$BIC={comp.get('delta_bic', float('nan')):.0f}")
    txt = []
    if sel == "single" and "sigma_int_kms" in primary:
        txt.append(rf"$\sigma_{{int}}={primary['sigma_int_kms']:.0f}$ km/s")
    if sel == "double" and "peak_sep_kms" in primary:
        txt.append(rf"$\Delta v={primary['peak_sep_kms']:.0f}$ km/s")
    if boot is not None:
        if np.isfinite(boot.get("sigma_int_A", {}).get("med", np.nan)):
            s = boot["sigma_int_A"]
            txt.append(rf"$\sigma_{{int}}$ CI [{s['lo']:.2f},{s['hi']:.2f}] Å")
        if np.isfinite(boot.get("double_fraction", np.nan)):
            txt.append(f"double in {boot['double_fraction']*100:.0f}% of draws")
    if txt:
        ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, va="top",
                fontsize=8.5, color="0.25",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.legend(frameon=False, fontsize=8, loc="upper right", title=leg_title)
    if title:
        ax.set_title(title)

    # residuals
    with np.errstate(invalid="ignore", divide="ignore"):
        rres = (data - primary["model"]) / err
    axr.axhspan(-1, 1, color="0.9", lw=0)
    axr.axhline(0, color="0.6", lw=0.7)
    axr.step(rest_wave, rres, where="mid", color="k", lw=0.9)
    axr.set_ylim(-4, 4)
    axr.set_xlim(xlim)
    axr.set_xlabel(r"rest wavelength [$\AA$]")
    axr.set_ylabel(r"resid/$\sigma$")

    plt.setp(ax.get_xticklabels(), visible=False)
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, (ax, axr)


def plot_lsf(lsf_result, stacks=None, radial_bin=0, stack_method="biweight",
             lya_center=1215.67, xlim=None, figsize=(7.2, 5.4),
             save_fig=False, savename="Figure_lsf.png"):
    """
    LSF overview from speclsf.stack_lsf. Top panel: the stacked rest-frame LSF
    with its FWHM marked. Bottom panel (when `stacks` is given): a stacked
    spectrum (default the first radial bin) with the LSF overlaid (scaled to the
    line peak) so you can SEE the instrumental width against the line.

        from utils_lya_halo import speclsf, plotting
        lsf = speclsf.stack_lsf(cfg, product, z_err_kms=60)
        plotting.plot_lsf(lsf, stacks, radial_bin=0)

    Returns (fig, axes).
    """
    rw = np.asarray(lsf_result["rest_wave"], float)
    lp = np.asarray(lsf_result["lsf"], float)
    pk = float(lsf_result.get("peak_A", lya_center))
    fwhm = lsf_result.get("fwhm_A", np.nan)
    fwhm_kms = lsf_result.get("fwhm_kms", np.nan)
    sig = lsf_result.get("sigma_A", np.nan)
    if xlim is None:
        xlim = (lya_center - 10, lya_center + 10)

    if stacks is not None:
        fig, (axL, axS) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                       gridspec_kw={"hspace": 0.08})
    else:
        fig, axL = plt.subplots(figsize=(figsize[0], figsize[1] * 0.55))
        axS = None

    # --- top: the LSF itself ---
    axL.fill_between(rw, lp, color="0.85", lw=0)
    axL.plot(rw, lp, color="0.35", lw=1.4)
    if np.isfinite(fwhm):
        half = 0.5 * np.nanmax(lp)
        axL.hlines(half, pk - fwhm / 2, pk + fwhm / 2, color="crimson", lw=1.6)
        axL.text(pk, half * 1.06, f"FWHM = {fwhm:.2f} Å ({fwhm_kms:.0f} km/s)",
                 ha="center", va="bottom", fontsize=9, color="crimson")
    axL.axvline(lya_center, color="magenta", lw=0.7, ls=":")
    axL.set_ylabel("LSF (norm.)")
    axL.set_xlim(xlim)
    axL.set_title(f"Stacked rest-frame LSF  (σ = {sig:.2f} Å)")
    axL.grid(alpha=0.12)

    # --- bottom: a spectrum with the LSF for scale ---
    if axS is not None:
        wave = np.asarray(stacks["rest_wave"], float)
        flux = np.asarray(stacks["stacks"][stack_method]["flux"][radial_bin], float)
        edges = np.round(np.asarray(stacks["r_edges"]), 3)
        lbl = (f"{edges[radial_bin]:g}–{edges[radial_bin+1]:g}"
               if len(edges) > radial_bin + 1 else f"bin {radial_bin}")
        axS.step(wave, flux, where="mid", color="k", lw=1.1,
                 label=f"stacked line ({lbl})")
        inwin = (wave >= xlim[0]) & (wave <= xlim[1])
        fmax = np.nanmax(flux[inwin]) if np.any(inwin) else np.nanmax(flux)
        lk = lp / np.nanmax(lp) * fmax
        axS.plot(rw, lk, color="0.55", lw=1.2, alpha=0.85, label="LSF (scaled)")
        axS.axvline(lya_center, color="magenta", lw=0.7, ls=":")
        axS.axhline(0, color="0.8", lw=0.6)
        axS.set_xlim(xlim)
        axS.set_xlabel(r"rest wavelength [$\AA$]")
        axS.set_ylabel("flux")
        axS.legend(frameon=False, fontsize=9)
        axS.grid(alpha=0.12)
    else:
        axL.set_xlabel(r"rest wavelength [$\AA$]")

    plt.tight_layout()
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ((axL, axS) if axS is not None else axL)


# ---------------------------------------------------------------------
# CORE-STACK QUICKLOOK  (consumes a stack.build_core_stacks result)
# ---------------------------------------------------------------------
# (name, rest_vac_A, type)  type: 'lya' | 'em' (emission) | 'abs' (ISM absorption)
_REST_UV = [
    ("Lya",      1215.67, "lya"),
    ("N V",      1240.14, "em"),    # AGN / P-Cygni
    ("Si II",    1260.42, "abs"),
    ("O I+Si II",1303.0,  "abs"),   # 1302.17 + 1304.37 blend  <- ~1300 feature
    ("C II",     1334.53, "abs"),
    ("Si IV",    1393.76, "abs"),   # +1402.77 doublet
    ("Si II",    1526.71, "abs"),   # <- ~1530 feature
    ("C IV",     1549.48, "em"),    # AGN emission + ISM absorption (P-Cygni)
    ("Fe II",    1608.45, "abs"),
    ("He II",    1640.42, "em"),
    ("O III]",   1663.5,  "em"),
    ("Al II",    1670.79, "abs"),
]
_LINE_C = {"lya": "r", "em": "#1f77b4", "abs": "0.5"}


def _mark_lines(ax):
    """Mark the rest-UV lines that fall inside ax's current xlim. Emission lines
    are labelled at the top, ISM absorption at the bottom, to avoid overlap."""
    lo, hi = ax.get_xlim()
    for nm, l0, kind in _REST_UV:
        if lo <= l0 <= hi:
            c = _LINE_C[kind]
            ax.axvline(l0, color=c, ls="--", lw=0.8, alpha=0.7)
            y, va = (0.99, "top") if kind != "abs" else (0.02, "bottom")
            ax.text(l0, y, nm, rotation=90, va=va, ha="right", fontsize=7,
                    color=c, transform=ax.get_xaxis_transform())


def _robust_ylim(w, f, win=None, exclude_center=1215.67, exclude_hw=12.0,
                 pct=(2, 98), pad=0.5, include_zero=True):
    """Semi-automatic y-limits for a zoomed stack panel: the percentile range of
    the flux inside `win`, EXCLUDING the Lya core (so the strong peak doesn't set
    the scale), padded so weak features (LIS absorption) aren't clipped. Returns
    (ylo, yhi) or None if too few points."""
    w = np.asarray(w, float); f = np.asarray(f, float)
    m = np.isfinite(f)
    if win is not None:
        m &= (w >= win[0]) & (w <= win[1])
    if exclude_hw and exclude_hw > 0:
        m &= np.abs(w - exclude_center) > exclude_hw
    fz = f[m]
    if fz.size < 3:
        return None
    lo, hi = np.nanpercentile(fz, pct[0]), np.nanpercentile(fz, pct[1])
    span = (hi - lo) if hi > lo else (abs(hi) or 1.0)
    ylo, yhi = lo - pad * span, hi + pad * span
    if include_zero:
        ylo, yhi = min(ylo, 0.0), max(yhi, 0.0)
    return float(ylo), float(yhi)

def plot_core_stack(res, method="biweight", lya_hw=40.0,
                    mid_lo=1150.0, mid_exclude_hw=12.0, mid_pct=(2, 98), mid_pad=0.5,
                    smooth_sigma=None):
    """Three-panel quicklook of a stack.build_core_stacks result: full spectrum
    with rest-UV lines, a clipped continuum view, and the Lya region.

    smooth_sigma : float or None
        If not None, the displayed flux (and error band) are smoothed with a
        Gaussian kernel of this sigma (in pixels). None = no smoothing.
    """
    w = res["rest_wave"]; S = res["stacks"][method]
    f, e = S["flux"], S["err"]

    # --- optional Gaussian smoothing for display ---
    if smooth_sigma is not None:
        f = gaussian_filter1d(f, smooth_sigma)
        e = gaussian_filter1d(e, smooth_sigma)

    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    wlo, whi = float(w[0]), float(w[-1])    # the rest-grid bounds (e.g. 1100..1400/1700)
    # --- panel 0: full spectrum, clamped to the data bounds (no auto-cushion) ---
    ax[0].plot(w, f, "k", lw=0.8)
    ax[0].fill_between(w, f-e, f+e, alpha=0.2, lw=0)
    ax[0].axhline(0, color="0.6", lw=0.7)
    ax[0].set_xlim(wlo, whi)
    _mark_lines(ax[0])
    ax[0].set_xlabel("rest wavelength [A]"); ax[0].set_ylabel(res["unit_info"]["y_unit"])
    ax[0].set_title(f"core stack - {method},  N={res['ngal']},  z_med={res['z_median']:.3f}")
    # --- panel 1: LIS-absorption zoom, 1150 -> END, semi-automatic y-range ---
    ax[1].plot(w, f, "k", lw=0.8)
    ax[1].fill_between(w, f-e, f+e, alpha=0.2, lw=0)
    ax[1].axhline(0, color="0.6", lw=0.7)
    ax[1].set_xlim(mid_lo, whi)                 # set BEFORE _mark_lines (it reads xlim)
    _mark_lines(ax[1])
    yl = _robust_ylim(w, f, win=(mid_lo, whi), exclude_hw=mid_exclude_hw,
                      pct=mid_pct, pad=mid_pad)
    if yl:
        ax[1].set_ylim(*yl)
    ax[1].set_xlabel("rest wavelength [A]"); ax[1].set_ylabel(res["unit_info"]["y_unit"])
    ax[1].set_title("LIS absorption region (Lya core excluded from y-scale)")
    # --- panel 2: Lya region ---
    sel = (w > 1215.67-lya_hw) & (w < 1215.67+lya_hw)
    ax[2].plot(w[sel], f[sel], "k", lw=1.0)
    ax[2].fill_between(w[sel], (f-e)[sel], (f+e)[sel], alpha=0.2, lw=0)
    ax[2].axvline(1216.67, color="darkred", ls="--", lw=0.5)
    ax[2].axvline(1215.67, color="tomato", ls="--", lw=2)
    ax[2].axvline(1214.67, color="cornflowerblue", ls="--", lw=0.5)
    ax[2].axvline(1213.67, color="darkblue", ls="--", lw=0.5)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("rest wavelength [A]"); ax[2].set_title("near Lya")
    plt.tight_layout(); plt.show()


def plot_top_sn_grid(sub, *, ncols=4, rest_frame=True, xlim=(1180, 1255),
                     lya_center=None, hw=6.0, red_offset=0.5, id_col="ID",
                     panel_size=(3.4, 2.4), smooth=None, savepath=None, dpi=200):
    """
    Grid of core spectra for a shortlist from selection.top_sn_galaxies -- one
    panel per galaxy, already S/N-descending -- so you can eyeball which are
    worth a halo profile before committing to IDs. The Lya integration window
    is shaded; each panel is annotated with the same S/N column
    top_sn_galaxies ranked on, plus ID and label.

        from .selection import top_sn_galaxies, id_list
        sub = top_sn_galaxies(core, n=20)
        plot_top_sn_grid(sub)
        ids = id_list(sub)[:5]          # -> multicat.stack_id_list

    Reads sn/flux/label straight from sub['results'] via
    selection.resolve_core_sn_fields -- NOT via core.CoreSpectrum.meas, which
    is empty for a hierarchical-schema product (see the note in selection.py)
    and is why an earlier version of this plot always showed S/N=nan.

    Returns (fig, axs).
    """
    from .core import to_rest_frame, lya_window
    from .selection import resolve_core_sn_fields

    lya_center = PipelineConfig.LYA_REST if lya_center is None else lya_center
    spec = np.asarray(sub["core_spec"])
    err_all = sub.get("core_err")
    wave = np.asarray(sub["wave_obs"])
    cat = sub["catalog"]
    results = sub.get("results") or {}
    f = resolve_core_sn_fields(results) if results else None

    n = spec.shape[0]
    if n == 0:
        raise ValueError("nothing to plot -- sub['core_spec'] is empty.")
    ncols = int(min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                            squeeze=False)
    win_l, win_r = lya_window(lya_center, hw, red_offset)

    sn_arr = (np.asarray(sub["sn_ranked"]) if "sn_ranked" in sub
             else (np.asarray(results[f["sn"]]) if f else np.full(n, np.nan)))
    label_arr = (np.asarray(results[f["label"]]) if f and f["label"] in results
                else np.full(n, "", dtype="<U1"))

    for k in range(n):
        ax = axs[k // ncols][k % ncols]
        w, fl = wave, spec[k]
        e = err_all[k] if err_all is not None else None
        if rest_frame and "z" in cat.colnames:
            zz = float(cat["z"][k])
            if zz:
                w, fl, e = to_rest_frame(w, fl, e, z=zz)
        y = gaussian_filter1d(fl, smooth) if smooth is not None else fl
        ax.step(w, y, where="mid", lw=0.9, color="k")
        if e is not None:
            ax.fill_between(w, fl - e, fl + e, step="mid", color="0.75", alpha=0.35, lw=0)
        ax.axvspan(win_l, win_r, color="magenta", alpha=0.10)
        ax.axvline(lya_center, color="magenta", lw=0.7, alpha=0.6)
        ax.axhline(0, color="tomato", lw=0.8, alpha=0.7)
        ax.set_xlim(xlim)

        win = (w >= xlim[0]) & (w <= xlim[1]) & np.isfinite(fl)
        if np.any(win):
            fmin, fmax = np.nanpercentile(fl[win], [2, 99])
            pad = 0.25 * (fmax - fmin if fmax > fmin else abs(fmax) + 1)
            ax.set_ylim(fmin - pad, fmax + pad)

        gid = cat[id_col][k] if id_col in cat.colnames else k
        ax.text(0.03, 0.86, f"{id_col}={gid}\nS/N={sn_arr[k]:.1f} {label_arr[k]}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.85",
                          boxstyle="round,pad=0.2"))
        ax.tick_params(labelsize=8, direction="in")

    for k in range(n, nrows * ncols):
        axs[k // ncols][k % ncols].axis("off")

    rank_key = sub.get("rank_key", "S/N")
    fig.supxlabel("rest-frame wavelength [\u00c5]" if rest_frame else
                  "observed wavelength [\u00c5]", fontsize=11)
    fig.supylabel("flux", fontsize=11)
    fig.suptitle(f"Top {n} by {rank_key}", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axs

# =====================================================================
# FIBER-MASK VISUALS  (moved from core.py -- plotting owns display)
# =====================================================================

# UV features commonly used to interpret core spectra, REST-frame Angstrom.
# (name, rest_wavelength, kind) -- kind drives the line style in plot_mask_spectra.
DEFAULT_UV_LINES = (
    ("Ly\u03b1",   1215.67, "lya"),
    ("N V",       1238.82, "emission"),
    ("Si II",     1260.42, "absorption"),
    ("Si III",    1206.50, "absorption"),
    ("O I/Si II", 1303.00, "absorption"),
    ("C II",      1334.53, "absorption"),
)


def plot_mask_spectra(
    results, labels=None, ylim=None, lya_zoom=True, zoom_hw_A=25.0,
    lines=DEFAULT_UV_LINES, show_lines=True, figsize_per_row=(10, 3.0),
    zoom_width_frac=0.32,
):
    """
    Plot the output of core.compare_mask_spectra(..., plot=False). Separated
    from the extraction so you can re-style freely (ylims, lines, zoom) without
    re-running the (uncached, slow) core extraction.

    One row per galaxy: the full observed spectrum extracted under every
    candidate mask overlaid, plus -- when lya_zoom -- a narrow inset panel on
    the right zoomed to +/- zoom_hw_A around that galaxy's OBSERVED Lya, so the
    line profile (e.g. a double peak) is legible even when the full panel is
    dominated by continuum/noise.

    results   : the dict returned by compare_mask_spectra. Must carry
                per-galaxy 'z' (from config.z_col) to place rest-frame lines in
                the observed frame; galaxies with NaN z get no line overlays.
    labels    : subset of results['labels'] to draw (default: all).
    ylim      : (lo, hi) applied to every full panel, or None to autoscale.
                The Lya-zoom insets always autoscale to their own window so the
                line is visible regardless.
    lines     : iterable of (name, rest_wavelength_A, kind). Default
                DEFAULT_UV_LINES (Lya, N V, Si III, Si II, O I/Si II, C II).
                kind 'lya' -> magenta solid; 'emission' -> red dashed;
                'absorption' -> steel-blue dotted.
    show_lines: master toggle for the line overlays.
    zoom_hw_A : half-width (observed A) of the Lya inset window.

    Returns the Figure.
    """
    LYA = PipelineConfig.LYA_REST

    wave = results["wave"]
    spectra = results["spectra"]
    all_labels = results["labels"]
    labels = list(all_labels if labels is None else labels)
    ids = results["ids"]
    idc = results.get("id_col", "ID")
    zvals = results.get("z", np.full(len(ids), np.nan))
    n = len(ids)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(labels)))

    def _draw_lines(ax, z, obs_lo, obs_hi):
        if not show_lines or not np.isfinite(z):
            return
        style = {"lya": dict(color="magenta", ls="-", lw=1.2, alpha=0.9),
                 "emission": dict(color="red", ls="--", lw=0.9, alpha=0.7),
                 "absorption": dict(color="steelblue", ls=":", lw=0.9, alpha=0.7)}
        for name, rest, kind in lines:
            obs = rest * (1.0 + z)
            if not (obs_lo <= obs <= obs_hi):
                continue
            ax.axvline(obs, **style.get(kind, style["absorption"]), zorder=1)
            ax.annotate(name, xy=(obs, 1.0), xycoords=("data", "axes fraction"),
                       xytext=(1, -2), textcoords="offset points", rotation=90,
                       va="top", ha="left", fontsize=6.5,
                       color=style.get(kind, style["absorption"])["color"])

    ncol = 2 if lya_zoom else 1
    width_ratios = [1.0 - zoom_width_frac, zoom_width_frac] if lya_zoom else [1.0]
    fig, axes = plt.subplots(
        n, ncol, squeeze=False,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n),
        gridspec_kw={"width_ratios": width_ratios},
    )

    for i in range(n):
        axm = axes[i, 0]
        z = float(zvals[i])
        for lab, c in zip(labels, cmap):
            axm.plot(wave, spectra[lab][i], color=c, lw=1, alpha=0.85, label=lab)
        axm.axhline(0, color="0.8", lw=0.6, zorder=0)
        if wave is not None and np.any(np.isfinite(wave)):
            axm.set_xlim(np.nanmin(wave), np.nanmax(wave))
        if ylim is not None:
            axm.set_ylim(*ylim)
        _draw_lines(axm, z, *axm.get_xlim())
        axm.set_title(f"{idc} = {ids[i]}" + (f"   z={z:.3f}" if np.isfinite(z) else "   z=?"),
                     fontsize=10)
        axm.set_ylabel("flux")
        if i == n - 1:
            axm.set_xlabel("observed wavelength [\u00c5]")
        if i == 0:
            axm.legend(fontsize=8, loc="upper right", ncol=len(labels))

        if lya_zoom:
            axz = axes[i, 1]
            if np.isfinite(z):
                lya_obs = LYA * (1.0 + z)
                z0, z1 = lya_obs - zoom_hw_A, lya_obs + zoom_hw_A
                for lab, c in zip(labels, cmap):
                    axz.plot(wave, spectra[lab][i], color=c, lw=1.1, alpha=0.9)
                axz.axhline(0, color="0.8", lw=0.6, zorder=0)
                axz.set_xlim(z0, z1)
                # autoscale y to just the zoom window so the line is visible
                win = (wave >= z0) & (wave <= z1)
                yy = np.concatenate([spectra[lab][i][win] for lab in labels])
                if np.any(np.isfinite(yy)):
                    lo, hi = np.nanmin(yy), np.nanmax(yy)
                    pad = 0.1 * (hi - lo if hi > lo else abs(hi) + 1)
                    axz.set_ylim(lo - pad, hi + pad)
                _draw_lines(axz, z, z0, z1)
                axz.set_title("Ly\u03b1 zoom", fontsize=8)
            else:
                axz.text(0.5, 0.5, "no z\n(no Ly\u03b1 zoom)", ha="center",
                        va="center", transform=axz.transAxes, fontsize=8, color="0.5")
                axz.set_xticks([]); axz.set_yticks([])
            if i == n - 1 and np.isfinite(z):
                axz.set_xlabel("observed \u00c5")

    fig.tight_layout()
    return fig


def plot_fiber_overlay(
    vdfi, catalog, gal_index=0, bad_fiber_mask=None,
    core_radius=3.5, ra_col="RA", dec_col="DEC",
    image=None, image_wcs=None, image_extent=None,
    pad_arcsec=2.0, ax=None, cmap="viridis", show_outside=True,
    title=None,
):
    """
    Overlay the extraction fibers for ONE galaxy on the sky.

    Draws, in galaxy-centered arcsec coordinates (East-left, North-up):
      * the galaxy position (a marker at the origin),
      * the extraction aperture as a dashed circle of radius `core_radius`,
      * every fiber as a dot; fibers INSIDE the aperture are colored/sized by
        their PSF weight with alpha proportional to that weight (so the optimal
        weighting is literally visible -- bright opaque dots near center, faint
        dots at the edge), and fibers OUTSIDE are drawn faintly in grey for
        context (toggle with show_outside),
      * optionally an underlay image (continuum cutout). Two ways to supply it:
          - image=2D array + image_extent=(xlo,xhi,ylo,yhi) in arcsec offsets, OR
          - image=2D array + image_wcs=astropy WCS (the cutout's WCS); the galaxy
            RA/DEC is used to convert the WCS grid into arcsec offsets.

    Parameters
    ----------
    vdfi          : the loaded field dict (from load_field_with_psf), giving
                    RA/DEC/PSF/seeing. Pass bad_fiber_mask to grey out masked fibers.
    catalog       : the run-table / catalog; row `gal_index` supplies RA/DEC.
    gal_index     : which galaxy (row) to draw.
    image         : optional 2-D continuum cutout to show underneath.
    image_extent  : (xlo, xhi, ylo, yhi) arcsec, if you already have the cutout
                    registered to arcsec offsets from the galaxy.
    image_wcs     : astropy WCS for the cutout; used (with the galaxy RA/DEC) to
                    build the arcsec extent if image_extent is not given.

    Returns (ax, geom) where geom is the fiber_extraction_geometry dict.
    """
    from matplotlib.patches import Circle
    from .core import fiber_extraction_geometry   # lazy: avoids core<->plotting cycle

    ra  = float(catalog[ra_col][gal_index])
    dec = float(catalog[dec_col][gal_index])
    geom = fiber_extraction_geometry(vdfi, ra, dec, bad_fiber_mask=bad_fiber_mask,
                                     core_radius=core_radius)
    dra, ddec = geom["dra"], geom["ddec"]
    w, in_ap, good = geom["weight"], geom["in_aperture"], geom["good"]

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.6))

    # --- optional image underlay ---
    if image is not None:
        if image_extent is None and image_wcs is not None:
            image_extent = _wcs_to_arcsec_extent(image, image_wcs, ra, dec)
        if image_extent is not None:
            ax.imshow(image, origin="lower", extent=image_extent,
                      cmap="Greys", zorder=0,
                      vmin=np.nanpercentile(image, 5),
                      vmax=np.nanpercentile(image, 99))

    # --- fibers outside the aperture: faint grey context ---
    if show_outside:
        out = good & ~in_ap & np.isfinite(dra) & np.isfinite(ddec)
        ax.scatter(dra[out], ddec[out], s=14, c="0.78", lw=0, zorder=1)
        bad = (~good) & np.isfinite(dra) & np.isfinite(ddec)
        ax.scatter(dra[bad], ddec[bad], s=10, marker="x", c="0.6",
                   lw=0.6, zorder=1, label="masked")

    # --- fibers inside the aperture: color + alpha by PSF weight ---
    wi = w[in_ap]
    wmax = np.nanmax(wi) if np.any(np.isfinite(wi)) else 1.0
    wmax = wmax if (np.isfinite(wmax) and wmax > 0) else 1.0
    wnorm = np.clip(np.nan_to_num(wi / wmax, nan=0.0), 0.0, 1.0)
    sc = ax.scatter(dra[in_ap], ddec[in_ap], c=wi, s=60 + 140 * wnorm,
                    cmap=cmap, alpha=None, edgecolors="k", linewidths=0.4,
                    zorder=3, vmin=0)
    # per-point alpha (scatter takes a scalar alpha, so set it on the collection)
    rgba = sc.get_facecolor()
    if rgba.shape[0] == wnorm.size:
        rgba[:, 3] = 0.25 + 0.75 * wnorm
        sc.set_facecolor(rgba)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"PSF weight @ {geom['wave_ref']:.0f} Å")

    # --- extraction aperture + galaxy center ---
    ax.add_patch(Circle((0, 0), core_radius, fill=False, ls="--",
                        ec="crimson", lw=1.6, zorder=4,
                        label=f"aperture {core_radius:g}\""))
    ax.plot(0, 0, marker="*", ms=16, mfc="gold", mec="k", mew=0.8,
            zorder=5, label="galaxy")

    lim = core_radius + pad_arcsec
    ax.set_xlim(lim, -lim)          # East left (RA increases left)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta$RA (arcsec)")
    ax.set_ylabel(r"$\Delta$Dec (arcsec)")
    ax.set_title(title or
                 f"galaxy {gal_index}: {geom['n_in_aperture']} fibers "
                 f"in {core_radius:g}\" aperture")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    return ax, geom


def _wcs_to_arcsec_extent(image, wcs, ra, dec):
    """
    Build an imshow `extent` (arcsec offsets from the galaxy) for a cutout image
    given its WCS and the galaxy RA/DEC. Uses the cutout corners; assumes the
    cutout is small enough that the local tangent-plane approximation holds (true
    for a few-arcsec core cutout).
    """
    ny, nx = image.shape
    # pixel corners (0-based) -> world
    corners_pix = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]],
                           dtype=float)
    try:
        world = wcs.wcs_pix2world(corners_pix, 0)
    except Exception:
        return None
    cra, cdec = world[:, 0], world[:, 1]
    dra  = (cra - ra) * np.cos(np.deg2rad(dec)) * 3600.
    ddec = (cdec - dec) * 3600.
    return (float(dra.max()), float(dra.min()),     # East-left
            float(ddec.min()), float(ddec.max()))




def plot_fiber_mask_appendix(
    vdfi, catalog, bad_fiber_mask, cont_image_path,
    radius_arcsec=5.0, cutout_arcsec=15.0, ra_col="RA", dec_col="DEC",
    ncols=4, max_panels=None, panel_size=2.4, id_col=None,
    shared_scale=False, galaxy_indices=None, fiber_alpha=0.5,
    mark_dead_data=True,
):
    """
    Appendix-style grid figure: one cutout_arcsec x cutout_arcsec (default
    15"x15", giving some field context around the 5" audit circle) panel per
    galaxy that has a masked fiber within radius_arcsec of its own center --
    the self-masking cases that should be FIXED AT BAKE TIME by unmasking
    that galaxy's segment in the baked segmap FITS (star-contaminated cores
    stay masked). Each panel shows a continuum-image cutout (cont_image_path;
    the CFHTLS cutout works for any mask method) with every fiber in the box
    overlaid: masked fibers in RED, extracting (good) fibers in BLUE, galaxy
    center as a gold star, and a dashed circle at radius_arcsec. fiber_alpha
    (default 0.5) sets the transparency of the fiber dots only -- lets
    overlapping fibers and the underlying image both stay visible.

    mark_dead_data (default True): a fiber is drawn as a hollow grey ring if its
        spectrum is entirely NaN in vdfi['spectra'] -- i.e. the fiber points at
        this patch of sky but recorded NO usable data. This is what the plain
        red/blue coloring can't show: a bright AGN/star core often has fibers
        that are positionally present (so they'd plot as normal dots) but whose
        flux is flagged/saturated to NaN in the reduction, so the core spectrum
        is NaN even though the panel looks full of fibers. Hollow rings flag
        exactly those, matching diagnose_core_fibers' 'DATA-STARVED' verdict.
        Set False to restore the pure position+mask view. Requires
        vdfi['spectra']; silently skipped if absent.

    Selection is done by audit_galaxy_mask_overlap(..., radius_arcsec), so the
    same radius drives both which galaxies get a panel and the circle drawn on
    each one -- independent of cutout_arcsec, which is purely how much extra
    field context each panel shows. Pass the legacy image mask to document the
    self-masking problem, or the baked segmap to confirm the panels vanish.

    Contrast: each panel's display range comes from astropy's ZScaleInterval
    (the same algorithm DS9/IRAF use), which is far better at pulling faint
    features out of a CFHT-style image than a flat 5/99th-percentile stretch.
    shared_scale=False (default) computes it per panel, so each galaxy's own
    field is individually well-contrasted; shared_scale=True computes ONE
    vmin/vmax from the whole continuum image and reuses it everywhere, which
    sacrifices some per-galaxy contrast but makes brightness directly
    comparable panel-to-panel.

    max_panels caps the grid to the worst offenders (most nearby masked
    fibers) first; leave None to plot every flagged galaxy.

    galaxy_indices : optional explicit row indices into `catalog`, bypassing
        the audit-based selection -- exactly these galaxies are plotted, in
        this order (audit_galaxy_mask_overlap still runs and is returned, but
        doesn't drive which panels are drawn). This is the tool for a
        before/after comparison: audit the OLD (image) mask to get the flagged
        indices, then call this twice -- once with the old mask, once with the
        baked segmap -- both with the SAME galaxy_indices, so you see the
        identical galaxies with red cores (before) next to blue cores (after):

            audit0 = audit_galaxy_mask_overlap(vdfi, catalog, bad_image, radius_arcsec=5.0)
            idx = np.flatnonzero(audit0["flagged"])
            fig_before, _ = plot_fiber_mask_appendix(vdfi, catalog, bad_image,  cont_path,
                                                     galaxy_indices=idx)
            fig_after,  _ = plot_fiber_mask_appendix(vdfi, catalog, bad_segmap, cont_path,
                                                     galaxy_indices=idx)

    Returns (fig, audit_table). audit_table is the FULL per-galaxy output of
    audit_galaxy_mask_overlap with a 'plotted' column added (True for the
    rows actually drawn, so you can see what max_panels/galaxy_indices cut).
    Returns (None, audit_table) if there's nothing to plot (no galaxy flagged
    and galaxy_indices is None, or galaxy_indices is an empty list).
    """
    from matplotlib.patches import Circle
    from astropy.io import fits as _fits
    from astropy.wcs import WCS as _WCS
    from astropy.nddata import Cutout2D
    from astropy.coordinates import SkyCoord
    from astropy.visualization import ZScaleInterval
    import astropy.units as u

    from .core import fiber_extraction_geometry, audit_galaxy_mask_overlap  # lazy

    zscale = ZScaleInterval()

    def _panel_limits(data):
        finite = data[np.isfinite(data)]
        if finite.size < 5:
            return 0.0, 1.0
        try:
            return zscale.get_limits(finite)
        except Exception:
            return (float(np.nanpercentile(finite, 5)),
                   float(np.nanpercentile(finite, 99)))

    audit = audit_galaxy_mask_overlap(vdfi, catalog, bad_fiber_mask,
                                      radius_arcsec=radius_arcsec, verbose=True)
    if galaxy_indices is not None:
        sel_idx = np.asarray(galaxy_indices, dtype=int)
    else:
        sel_idx = np.flatnonzero(np.asarray(audit["flagged"]))
        # worst offenders first, so max_panels (if set) keeps the most relevant cases
        sel_idx = sel_idx[np.argsort(-np.asarray(audit["n_bad_near"])[sel_idx])]
    plotted = np.zeros(len(audit), dtype=bool)
    if max_panels is not None:
        sel_idx = sel_idx[:max_panels]
    plotted[sel_idx] = True
    audit["plotted"] = plotted

    if sel_idx.size == 0:
        print(f"plot_fiber_mask_appendix: nothing to plot (no galaxy flagged within "
              f"{radius_arcsec:g}\" and no galaxy_indices given).")
        return None, audit

    with _fits.open(cont_image_path) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32)
        wcs_image = _WCS(hdul[0].header)

    ra = np.asarray(catalog[ra_col], dtype=float)
    dec = np.asarray(catalog[dec_col], dtype=float)
    half = cutout_arcsec / 2.0

    shared_vmin = shared_vmax = None
    if shared_scale:
        shared_vmin, shared_vmax = _panel_limits(image)

    n = sel_idx.size
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size * ncols, panel_size * nrows),
                             squeeze=False)

    for k, i in enumerate(sel_idx):
        ax = axes[k // ncols, k % ncols]
        gal_ra, gal_dec = float(ra[i]), float(dec[i])

        # --- continuum-image cutout underlay ---
        try:
            center = SkyCoord(gal_ra * u.deg, gal_dec * u.deg)
            cut = Cutout2D(image, center, size=cutout_arcsec * u.arcsec,
                           wcs=wcs_image, mode="partial", fill_value=np.nan)
            extent = _wcs_to_arcsec_extent(cut.data, cut.wcs, gal_ra, gal_dec)
            if extent is not None and np.any(np.isfinite(cut.data)):
                vmin, vmax = (shared_vmin, shared_vmax) if shared_scale else _panel_limits(cut.data)
                ax.imshow(cut.data, origin="lower", extent=extent, cmap="Greys",
                         vmin=vmin, vmax=vmax, zorder=0)
        except Exception as e:
            ax.text(0.5, 0.5, f"cutout failed:\n{e}", ha="center", va="center",
                   transform=ax.transAxes, fontsize=6)

        # --- fibers in the box ---
        geom = fiber_extraction_geometry(vdfi, gal_ra, gal_dec,
                                         bad_fiber_mask=bad_fiber_mask,
                                         core_radius=half)
        dra, ddec, good = geom["dra"], geom["ddec"], geom["good"]
        inbox = (np.isfinite(dra) & np.isfinite(ddec) &
                (np.abs(dra) <= half) & (np.abs(ddec) <= half))
        bad = inbox & ~good
        ok = inbox & good

        # data-dead fibers: positionally present but spectrum entirely NaN.
        # fiber_extraction_geometry pools every exposure's full fiber list in
        # order, so we build the same-shaped 'has data' mask the same way.
        has_data = None
        if mark_dead_data and vdfi.get("spectra") is not None:
            spectra = vdfi["spectra"]
            has_data = np.concatenate([
                np.any(np.isfinite(spectra[e]), axis=1) for e in range(spectra.shape[0])
            ])
            if has_data.shape != dra.shape:      # shape mismatch -> skip rather than misalign
                has_data = None

        if has_data is not None:
            ok_live,  ok_dead  = ok & has_data,  ok & ~has_data
            bad_live, bad_dead = bad & has_data, bad & ~has_data
            ax.scatter(dra[ok_live], ddec[ok_live], s=22, c="dodgerblue",
                      edgecolors="k", linewidths=0.3, alpha=fiber_alpha, zorder=2)
            ax.scatter(dra[bad_live], ddec[bad_live], s=26, c="red",
                      edgecolors="k", linewidths=0.4, alpha=fiber_alpha, zorder=3)
            # dead-data fibers: hollow grey rings (facecolor 'none'), regardless of mask
            dead = ok_dead | bad_dead
            ax.scatter(dra[dead], ddec[dead], s=24, facecolors="none",
                      edgecolors="0.5", linewidths=0.8, zorder=2)
        else:
            ax.scatter(dra[ok], ddec[ok], s=22, c="dodgerblue", edgecolors="k",
                      linewidths=0.3, alpha=fiber_alpha, zorder=2)
            ax.scatter(dra[bad], ddec[bad], s=26, c="red", edgecolors="k",
                      linewidths=0.4, alpha=fiber_alpha, zorder=3)
        ax.add_patch(Circle((0, 0), radius_arcsec, fill=False, ls="--",
                            ec="0.4", lw=1.0, zorder=4))
        ax.plot(0, 0, marker="*", ms=10, mfc="gold", mec="k", mew=0.6, zorder=5)

        ax.set_xlim(half, -half)            # East left
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        gid = catalog[id_col][i] if id_col else i
        ax.set_title(f"{gid}  ({int(bad.sum())} near)", fontsize=8)

    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue",
              markeredgecolor="k", markersize=7, label="extracting (good) fiber"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
              markeredgecolor="k", markersize=7, label="masked fiber"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
              markeredgecolor="k", markersize=10, label="galaxy center"),
        Line2D([0], [0], color="0.4", ls="--", lw=1.2,
              label=f"{radius_arcsec:g}\" audit radius"),
    ]
    if mark_dead_data and vdfi.get("spectra") is not None:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                  markeredgecolor="0.5", markersize=7, label="no data (all-NaN spectrum)"))
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(
        f"Galaxies with a masked fiber within {radius_arcsec:g}\" of center "
        f"({n}/{len(catalog)} shown)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig, audit


def list_exposures(vdfi, exposures=None, catalog=None, ra_col="RA", dec_col="DEC",
                   match_radius_arcsec=None, verbose=True):
    """
    Summarize each exposure ("slice") in a loaded VDFI field: fiber count and
    sky footprint. The "what have I got" table to look at BEFORE picking which
    exposures to hand to plot_field_coverage -- for AEGIS a full-field view
    typically needs both IFU dither pointings (left/right of the field), for
    COSMOS the same idea repeated ~4x over the larger footprint. Rather than
    guessing exposure IDs and hoping they line up, use this to see where each
    one actually sits, or just pass exposures=None to plot_field_coverage and
    use everything.

    Parameters
    ----------
    vdfi        : loaded field dict (extract.load_field / load_vdfi_field).
    exposures   : iterable of exposure indices to summarize; None -> all.
    catalog     : optional astropy Table (e.g. run_table); if given together
                  with match_radius_arcsec, adds a column counting how many
                  catalog rows fall within that radius of each exposure's
                  mean center (a rough guide to which slices matter for your
                  sample).
    match_radius_arcsec : radius (arcsec) for the optional catalog column
                  above. Ignored if catalog is None.

    Returns an astropy Table, one row per exposure (index, n_fibers, mean RA/
    DEC, and the RA/DEC extent), printed if verbose.
    """
    from astropy.table import Table

    RA, DEC = vdfi["RA"], vdfi["DEC"]
    nexp = vdfi.get("nexp", RA.shape[0])
    idx = np.arange(nexp) if exposures is None else np.atleast_1d(exposures).astype(int)

    rows = []
    for i in idx:
        ra_i, dec_i = RA[i], DEC[i]
        good = np.isfinite(ra_i) & np.isfinite(dec_i)
        if not good.any():
            rows.append((int(i), 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        ra_g, dec_g = ra_i[good], dec_i[good]
        rows.append((int(i), int(good.sum()),
                     float(np.mean(ra_g)), float(np.mean(dec_g)),
                     float(ra_g.min()), float(ra_g.max()),
                     float(dec_g.min()), float(dec_g.max())))

    t = Table(rows=rows, names=["exposure", "n_fibers", "ra_mean", "dec_mean",
                                "ra_min", "ra_max", "dec_min", "dec_max"])

    if catalog is not None and match_radius_arcsec is not None:
        cra = np.asarray(catalog[ra_col], dtype=float)
        cdec = np.asarray(catalog[dec_col], dtype=float)
        n_near = np.zeros(len(t), dtype=int)
        for k in range(len(t)):
            if not np.isfinite(t["ra_mean"][k]):
                continue
            dra = (cra - t["ra_mean"][k]) * np.cos(np.deg2rad(t["dec_mean"][k])) * 3600.
            ddec = (cdec - t["dec_mean"][k]) * 3600.
            n_near[k] = int((np.hypot(dra, ddec) <= match_radius_arcsec).sum())
        t[f"n_catalog_within_{match_radius_arcsec:g}asec"] = n_near

    if verbose:
        print(f"{vdfi.get('field', '?')}: {len(t)} exposure(s) of {nexp} total")
        t.pprint(max_lines=-1, max_width=-1)
    return t


def plot_field_coverage(vdfi, catalog=None, *, exposures=None, xc=None, yc=None,
                        radii_arcsec=(100.0, 200.0, 300.0),
                        radii_kpc=None, z_ref=None, z_col="z",
                        radius_colors=("limegreen", "gold", "crimson"),
                        band_alpha=0.25, ring_lw=1.4,
                        ra_col="RA", dec_col="DEC",
                        fiber_size=3.0, fiber_alpha=0.15, fiber_color="0.5",
                        background_alpha=None,
                        catalog_kwargs=None, catalog_label="MOSDEF",
                        extra_catalogs=None, extra_catalog_kwargs=None,
                        tick_direction="in", tick_right=True, tick_top=False,
                        tick_minor=False,
                        ax=None, title=None, verbose=True):
    """
    Plot the sky footprint of one or more VDFI exposures ("slices") in grey,
    with concentric radius bands drawn as TRANSPARENT FILLED ANNULI on top of
    the fibers, plus one or more galaxy catalogs (e.g. MOSDEF run_table, and
    3D-HST in a second color) overplotted on top of everything.

    Layers, bottom -> top:
        1. every fiber in the selected exposures, one grey color
        2. filled annuli [0, r1], [r1, r2], ... (band_alpha), full circles --
           they are drawn even where they run off the fiber footprint or
           into a corner, and the axis limits grow to show them whole
        3. ring outlines at each radius
        4. extra catalogs (e.g. 3D-HST), then the primary catalog on top

    Radii in kpc
    ------------
    Our radial bins are defined in physical kpc, but the sky plot is in
    angles, so a kpc radius has to be turned into arcsec at SOME redshift.
    With radii_kpc given, this function does:

        z_ref  = mean(catalog[z_col])            # unless z_ref is passed
        scale  = arcsec per physical kpc at z_ref (Planck18, via
                 virial.physical_kpc_to_arcsec -- same as the pipeline)
        radii_arcsec = radii_kpc * scale

    and prints the conversion. It's a representative ring, not any one
    galaxy's aperture: each galaxy's real bin edge is at its own z. Around
    z~2.3 the angular scale is nearly flat in z, so the spread is small.

    Parameters
    ----------
    vdfi          : loaded field dict (extract.load_field / load_vdfi_field).
    catalog       : primary astropy Table (e.g. MOSDEF run_table). Also the
                    source of z_ref when radii_kpc is used.
    exposures     : iterable of exposure indices to include; None -> all.
    xc, yc        : center of the radius bands, in degrees. None -> mean fiber
                    position of the selected exposures (the mosaic center).
    radii_arcsec  : outer radius of each band, in arcsec. Ignored if
                    radii_kpc is given.
    radii_kpc     : outer radius of each band, in physical kpc (e.g. your bin
                    edges). Converted to arcsec at z_ref (see above).
    z_ref         : redshift for the kpc -> arcsec conversion. None -> mean of
                    catalog[z_col] (finite values only).
    z_col         : redshift column in `catalog` (cfg.z_col, default "z").
    radius_colors : band colors, inside-out. If there are more radii than
                    colors, colors are drawn from the viridis colormap instead.
    band_alpha    : transparency of the filled annuli.
    ring_lw       : line width of the ring outlines (0 -> no outlines).
    fiber_size / fiber_alpha / fiber_color : styling of the grey fibers.
    background_alpha : DEPRECATED, ignored (all fibers are now one grey
                    layer styled by fiber_alpha).
    catalog_kwargs : scatter() kwargs overriding the primary catalog style.
    catalog_label : legend label for the primary catalog.
    extra_catalogs : dict {label: Table} of additional catalogs to overplot
                    underneath the primary one, e.g. {"3D-HST": t3dhst}.
                    Each gets its own default color (darkorange, then
                    deepskyblue, magenta, ...).
    extra_catalog_kwargs : dict {label: scatter kwargs} to restyle any of the
                    extra catalogs.
    tick_direction / tick_right / tick_top / tick_minor : house tick style
                    (formatting.apply_tick_style).
    verbose       : print the kpc -> arcsec conversion when used.

    Returns the Axes. (RA increases leftward, matching selection.plot_sky.)
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    if background_alpha is not None:
        warnings.warn("plot_field_coverage: background_alpha is deprecated and "
                      "ignored -- all fibers are one grey layer now; use "
                      "fiber_alpha.", DeprecationWarning, stacklevel=2)

    RA, DEC = vdfi["RA"], vdfi["DEC"]
    nexp = vdfi.get("nexp", RA.shape[0])
    idx = np.arange(nexp) if exposures is None else np.atleast_1d(exposures).astype(int)

    ra_all = np.concatenate([np.asarray(RA[i], dtype=float).ravel() for i in idx])
    dec_all = np.concatenate([np.asarray(DEC[i], dtype=float).ravel() for i in idx])
    good = np.isfinite(ra_all) & np.isfinite(dec_all)
    ra_all, dec_all = ra_all[good], dec_all[good]

    if xc is None:
        xc = float(np.mean(ra_all))
    if yc is None:
        yc = float(np.mean(dec_all))
    cosd = np.cos(np.deg2rad(yc))

    # ---- radii: kpc -> arcsec at a single representative redshift --------
    if radii_kpc is not None:
        from .virial import physical_kpc_to_arcsec
        if z_ref is None:
            if catalog is None or z_col not in catalog.colnames:
                raise ValueError("radii_kpc needs z_ref, or a catalog with a "
                                 f"'{z_col}' column to average.")
            zz = np.asarray(catalog[z_col], dtype=float)
            zz = zz[np.isfinite(zz) & (zz > 0)]
            z_ref = float(np.mean(zz))
            z_src = f"mean {z_col} of {zz.size} catalog galaxies"
        else:
            z_ref = float(z_ref)
            z_src = "given"
        radii_kpc = np.sort(np.atleast_1d(radii_kpc).astype(float))
        scale = float(physical_kpc_to_arcsec(1.0, z_ref))      # arcsec / kpc
        radii = radii_kpc * scale
        labels = [f"{rk:g} kpc ({ra:.0f}\")" for rk, ra in zip(radii_kpc, radii)]
        if verbose:
            print(f"kpc -> arcsec at z_ref = {z_ref:.3f} ({z_src}): "
                  f"{scale:.4f} \"/kpc  ({1/scale:.2f} kpc/\")")
            for lab in labels:
                print("   ", lab)
    else:
        radii = np.sort(np.atleast_1d(radii_arcsec).astype(float))
        labels = [f"{r:g}\"" for r in radii]

    nr = radii.size
    if len(radius_colors) >= nr:
        colors = list(radius_colors)[:nr]
    else:
        colors = [plt.cm.viridis(v) for v in np.linspace(0.15, 0.9, nr)]

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.0))

    # 1. all fibers, one grey layer
    ax.scatter(ra_all, dec_all, s=fiber_size, c=fiber_color, alpha=fiber_alpha,
               lw=0, zorder=1, rasterized=True,
               label=f"fibers ({idx.size} exposure(s))")

    # 2+3. filled annuli on top, plus ring outlines. Each annulus is a closed
    # outer ring + reversed inner ring (a hole), widened in RA by 1/cos(dec)
    # so it renders as a true circle with aspect = 1/cos(dec).
    th = np.linspace(0, 2 * np.pi, 361)

    def _ring(r_as):
        rdeg = r_as / 3600.
        return np.column_stack([xc + rdeg * np.cos(th) / cosd,
                                yc + rdeg * np.sin(th)])

    r_in = 0.0
    for r_out, c, lab in zip(radii, colors, labels):
        outer = _ring(r_out)
        if r_in > 0:
            inner = _ring(r_in)[::-1]
            verts = np.vstack([outer, inner])
            codes = np.concatenate([
                [Path.MOVETO], np.full(len(outer) - 1, Path.LINETO),
                [Path.MOVETO], np.full(len(inner) - 1, Path.LINETO)])
        else:
            verts, codes = outer, None
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=c, edgecolor="none",
                               alpha=band_alpha, zorder=2, label=f"<= {lab}"))
        if ring_lw:
            ax.plot(outer[:, 0], outer[:, 1], color=c, lw=ring_lw, zorder=3)
        r_in = r_out
    ax.plot([xc], [yc], "+", color="k", ms=10, mew=1.6, zorder=4)

    # 4. catalogs: extras underneath, primary on top
    default_extra = ["darkorange", "deepskyblue", "magenta", "saddlebrown"]
    for k, (lab, tab) in enumerate((extra_catalogs or {}).items()):
        ekw = dict(s=12, marker="o", c=default_extra[k % len(default_extra)],
                   edgecolors="none", alpha=0.6, zorder=5 + 0.01 * k,
                   label=f"{lab} ({len(tab)})")
        ekw.update((extra_catalog_kwargs or {}).get(lab, {}))
        ax.scatter(np.asarray(tab[ra_col], dtype=float),
                   np.asarray(tab[dec_col], dtype=float), **ekw)

    if catalog is not None:
        ckw = dict(s=26, marker="*", c="k", edgecolors="white", linewidths=0.5,
                   zorder=7, label=f"{catalog_label} ({len(catalog)})")
        if catalog_kwargs:
            ckw.update(catalog_kwargs)
        ax.scatter(np.asarray(catalog[ra_col], dtype=float),
                   np.asarray(catalog[dec_col], dtype=float), **ckw)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.autoscale_view()
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_aspect(1.0 / cosd)
    apply_tick_style(ax, direction=tick_direction, right=tick_right,
                     top=tick_top, minor=tick_minor)
    ax.legend(loc="best", fontsize=8, frameon=False, markerscale=2)
    ax.set_title(title or
                 f"{vdfi.get('field', '?')}: {idx.size} exposure(s), "
                 f"center=({xc:.4f}, {yc:.4f})")
    return ax


# =====================================================================
# SPECTRAL LINE PANELS (moved from analysis.py, 2026-09-16)
# =====================================================================

def plot_line_panels(
    stacks: dict,
    boot: dict | None = None,
    stack_method: str = "biweight",
    bounds=None,
    config=None,
    cont_bounds=DEFAULT_CONT_BOUNDS,
    cont_method: str = "poly",
    cont_order: int = 1,
    lya_center: float = PipelineConfig.LYA_REST,
    xlim: tuple = (1208.0, 1226.0),
    xunit: str = "wavelength",
    panel_height: float = 2.6,
    figwidth: float = 8.0,
    ncols: int = 1,
    fill_order: str = "row",
    show_cont: bool = True,
    show_window: bool = True,
    show_sidebands: bool = True,
    show_centroid: bool = True,
    stack_color: str = "k",
    band_color: str = "0.82",
    band_alpha: float = 1.0,
    cont_color: str = "tomato",
    window_color: str = "magenta",
    window_alpha: float = 0.15,
    sideband_color: str = "0.55",
    sideband_alpha: float = 0.25,
    centroid_color: str = "magenta",
    lya_line_color: str = "0.7",
    zero_color: str = "0.7",
    label_fontsize: float = 9.5,
    legend_fontsize: float = 9,
    tick_fontsize: float = 10,
    tick_direction: str = "in",
    tick_right: bool = True,
    tick_top: bool = False,
    tick_minor: bool = False,
    xlabel_fontsize: float = 12,
    ylabel_fontsize: float = 11,
    title_fontsize: float = 13,
    xlabel_text: str | None = None,
    ylabel_text: str = "Flux",
    title_text: str | None = None,
    xlabel_y: float = 0.01,
    ylabel_x: float = 0.01,
    title_y: float | None = None,
    legend_y: float = 0.95,
    top_margin: float = 0.87,
    band_label: str = "bootstrap 16–84%",
    stack_label: str | None = None,
    cont_label: str = "continuum",
    save_fig: bool = False,
    savename: str = "Figure_line_panels.png",
):
    """
    The most important per-run diagnostic: one spectral panel per radial bin
    zoomed to the Lyα region, showing:

      • Bootstrap shaded band (grey; needs boot["stack_error"]): the RAW
        16th-84th percentiles of the resampled stacks, per pixel -- the same
        percentile-interval convention as every other error bar in the
        figure set (flux profile, centroid, R_vir band). The fiducial curve
        is not forced to be the band's midline (2026-09-16; was re-anchored
        onto the fiducial before).
      • Biweight (or chosen method) fiducial stack (black)
      • Polynomial continuum fit (red dashed; optional)
      • Line window shaded (magenta by default; optional)
      • Sideband regions shaded (grey by default; optional)
      • Centroid marker: vertical dashed line at v_fid, with a short horizontal
        bar at the top spanning the 16/84 interval (optional; needs boot)

    Every color and font size above is a keyword argument (see below), so the
    whole figure is restylable without touching the function body.

    This is the visual check that the centroid is landing on the line peak, that
    the continuum is flat in the sidebands, and that the window captures the
    emission without clipping the wings.

    Parameters
    ----------
    stacks       : Stage-2 dict (keep_cube=True for the boot error band)
    boot         : Stage-3 bootstrap result; None -> no error band / centroid marker
    stack_method : which stacks[method] to display
    bounds       : line window (lo, hi) Å. None -> resolved in priority order:
                   (1) config.line_window if `config` is given, (2) the
                   stacks["provenance"]["line_window"] this stacks dict was
                   actually built with (stamped by build_stacks), (3) only if
                   neither is available, the hardcoded (lya_center-4,
                   lya_center+4) fallback. This means the panel's shaded
                   window always matches the bounds actually used to make the
                   stack, instead of silently showing a fixed +/-4 Angstrom
                   box regardless of what config.line_window really was.
    config       : optional PipelineConfig; only consulted to resolve `bounds`
                   when bounds=None (see above). Not required -- if omitted,
                   the stacks dict's own provenance is used instead.
    xlim         : x-axis range shown in each panel. ALWAYS given in
                   Angstroms regardless of xunit (same for bounds/cont_bounds/
                   lya_center) -- these stay the physical, science-facing
                   inputs; xunit only changes what's DISPLAYED. e.g. the
                   default xlim=(1208, 1226) Å auto-converts to roughly
                   (-1890, +2550) km/s when xunit="velocity".
    xunit        : "wavelength" (default, Å, rest-frame) or "velocity" (km/s
                   relative to lya_center, via v = c*(wave-lya_center)/lya_center
                   -- the same convention already used for the centroid). The
                   vertical dotted rest-frame marker line sits at v=0 in
                   velocity mode. The x-axis label switches automatically too
                   ("Velocity [km s$^{-1}$]") unless you set xlabel_text.
    ncols        : number of grid columns (default 1 = the original single tall
                   column). For a paper figure with many radial bins, ncols=2
                   (or more) arranges the panels in a grid instead of one very
                   tall column. x-axis tick labels are only drawn on the
                   bottom-most used panel of each column. Try e.g.
                   ncols=2, figwidth=11, panel_height=2.6, show_centroid=False
                   for a compact 2x5 print figure with the centroid marker
                   left out (since that belongs on the separate centroid-vs-
                   radius plot).
    fill_order   : "row" (default) -- panels fill left-to-right, then wrap to
                   the next row (bin 0 top-left, bin 1 top-right, ...).
                   "col" -- panels fill top-to-bottom within a column, then
                   continue into the next column (the original grid ordering).
                   No effect when ncols=1.
    stack_color, band_color, cont_color, window_color, sideband_color,
    centroid_color, lya_line_color, zero_color : per-element colors, all
                   independently overridable. window_color/sideband_color
                   default to magenta/grey; band_alpha, window_alpha,
                   sideband_alpha control their fill opacity.
    label_fontsize, legend_fontsize, tick_fontsize, xlabel_fontsize,
    ylabel_fontsize, title_fontsize : text sizes for the per-panel bin label,
                   the legend, the tick labels, the shared x/y axis labels,
                   and the figure suptitle, respectively.
    tick_direction, tick_right, tick_top, tick_minor : tick styling, passed
                   to formatting.apply_tick_style (the house tick style).
                   Defaults: ticks point INTO the panel ("in"; use "out"
                   or "inout" for the old look), y ticks mirrored on the
                   right spine (ticks only, no labels), no top ticks, no
                   added minor ticks.
    xlabel_text, ylabel_text, title_text : override the shared x-axis label,
                   shared y-axis label, and suptitle strings. xlabel_text=None
                   picks the wavelength/velocity default based on xunit;
                   title_text=None keeps the auto-generated "Lyα line panels
                   -- {method} stack". The x-axis label is drawn ONCE for the
                   whole figure (like the y-axis "Flux" label) via
                   fig.supxlabel, not repeated under every column.
    band_label, stack_label, cont_label : legend text for the bootstrap 1σ
                   band, the fiducial stack line, and the continuum line.
                   stack_label=None defaults to f"{stack_method} stack" (so
                   it tracks whichever method you're plotting); the other two
                   default to "bootstrap 16–84%" and
                   "continuum".
    xlabel_y, ylabel_x : figure-fraction position of the shared x-/y-axis
                   label (passed straight to fig.supxlabel(y=...) /
                   fig.supylabel(x=...)); nudge these if the label sits too
                   close to or far from the tick numbers.
    title_y      : figure-fraction y position of the suptitle. None -> auto
                   (near the very top for ncols=1, or above the reserved
                   legend strip for a grid).
    legend_y     : figure-fraction y position (bbox_to_anchor) of the shared
                   legend drawn above the grid when ncols>1. No effect when
                   ncols=1 (legend sits inside the first panel instead).
    top_margin   : fraction of the figure height given to the panel grid
                   when ncols>1 (the rest, above it, is reserved for the
                   legend + title). Lower this (e.g. 0.80) if the legend is
                   crowding the top row -- this is the fix for a squeezed
                   legend with many columns or with axes that show a
                   scientific-notation exponent (e.g. "1e38") above them.
    """
    from .measure import get_continuum_model   # lazy: plotting stays import-light

    if xunit not in ("wavelength", "velocity"):
        raise ValueError("xunit must be 'wavelength' or 'velocity'")

    wave   = np.asarray(stacks["rest_wave"])
    fid    = np.asarray(stacks["stacks"][stack_method]["flux"])   # (nrad, nwave)
    edges  = np.round(np.asarray(stacks["r_edges"]), 3)
    nrad   = fid.shape[0]
    if bounds is not None:
        bounds = tuple(bounds)
    elif config is not None:
        bounds = tuple(config.line_window)
    else:
        prov_window = stacks.get("provenance", {}).get("line_window")
        bounds = tuple(prov_window) if prov_window is not None else (lya_center - 4, lya_center + 4)
    bm     = stacks.get("bin_mode", "")
    stack_label_final = stack_label if stack_label is not None else f"{stack_method} stack"

    # xunit only changes what's DISPLAYED -- bounds/cont_bounds/xlim/lya_center
    # stay in Angstroms as the science-facing inputs (and the continuum fit
    # below is always computed in wavelength space); here we just derive the
    # x-COORDINATES actually handed to the plotting calls, converting to km/s
    # (same convention as the centroid: v = c*(wave-lya_center)/lya_center)
    # when xunit="velocity".
    def _to_velocity(wl):
        return C_KMS * (np.asarray(wl, dtype=float) - lya_center) / lya_center

    if xunit == "velocity":
        x_plot      = _to_velocity(wave)
        bounds_x    = tuple(_to_velocity(bounds))
        cont_bounds_x = [tuple(_to_velocity(cb)) for cb in cont_bounds]
        xlim_x      = tuple(_to_velocity(xlim))
        lya_line_x  = 0.0
    else:
        x_plot      = wave
        bounds_x    = bounds
        cont_bounds_x = cont_bounds
        xlim_x      = xlim
        lya_line_x  = lya_center

    if xlabel_text is None:
        xlabel_text_final = (r"Velocity [km s$^{-1}$]" if xunit == "velocity"
                             else r"Rest-frame wavelength [$\AA$]")
    else:
        xlabel_text_final = xlabel_text

    # bootstrap error band -- RAW 16th/84th percentiles of the resampled
    # stacks (percentile interval), the same convention as the flux-profile,
    # centroid and R_vir error bars. Decided 2026-09-16; the previous version
    # re-anchored the (median-p16, p84-median) spread onto `fid`. With raw
    # percentiles the fiducial curve can sit off the band's midline in faint,
    # small-N bins -- that offset is real information, not a plotting bug.
    band_lo = band_hi = None
    if boot is not None and boot.get("stack_error") is not None:
        band_lo = np.asarray(boot["stack_error"]["p16"])         # (nrad, nwave)
        band_hi = np.asarray(boot["stack_error"]["p84"])

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(nrad / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols,
                                  figsize=(figwidth, panel_height * nrows),
                                  sharex=True, squeeze=False)

    # bin index r -> (row, col). "row" fill_order goes left-to-right then
    # wraps to the next row (bin 0 top-left, bin 1 top-right, ...); "col"
    # fill_order goes top-to-bottom within a column, then continues into the
    # next column. With ncols=1 both reduce to the original single-column order.
    if fill_order == "row":
        grid_pos = [(r // ncols, r % ncols) for r in range(nrad)]
    else:
        grid_pos = [(r % nrows, r // nrows) for r in range(nrad)]
    axes = [axes_grid[row, col] for row, col in grid_pos]

    # hide any grid cells left empty because nrad doesn't evenly fill nrows*ncols
    used = set(grid_pos)
    for row in range(nrows):
        for col in range(ncols):
            if (row, col) not in used:
                axes_grid[row, col].axis("off")

    for r, ax in enumerate(axes):
        # --- bootstrap band (raw 16th-84th percentiles) ---
        if band_lo is not None:
            ax.fill_between(x_plot, band_lo[r], band_hi[r], color=band_color, lw=0,
                            alpha=band_alpha, zorder=0,
                            label=band_label if r == 0 else None)

        # --- sideband regions ---
        if show_sidebands:
            for (lo, hi) in cont_bounds_x:
                ax.axvspan(lo, hi, color=sideband_color, alpha=sideband_alpha, lw=0)

        # --- line window ---
        if show_window:
            ax.axvspan(bounds_x[0], bounds_x[1], color=window_color, alpha=window_alpha, lw=0)

        # --- fiducial stack ---
        ax.plot(x_plot, fid[r], color=stack_color, lw=1.1, zorder=3,
                label=stack_label_final if r == 0 else None)

        # --- continuum (always fit in wavelength space; only the DRAWN curve
        # is shown on the x_plot axis, whichever unit that is) ---
        if show_cont:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cont = get_continuum_model(wave, fid[r], cont_bounds=cont_bounds,
                                           cont_method=cont_method, cont_order=cont_order)
            ax.plot(x_plot, cont, color=cont_color, lw=1.0, ls="--", zorder=2,
                    label=cont_label if r == 0 else None)

        # --- centroid marker ---
        if show_centroid and boot is not None:
            v_f  = boot["centroid_v_fid"][r]
            v_lo = boot["centroid_v_lo"][r]
            v_hi = boot["centroid_v_hi"][r]
            if xunit == "velocity":
                # already in km/s -- no need to round-trip through wavelength
                x_c, x_lo, x_hi = v_f, v_lo, v_hi
            else:
                x_c  = lya_center * (1 + v_f  / C_KMS)
                x_lo = lya_center * (1 + v_lo / C_KMS)
                x_hi = lya_center * (1 + v_hi / C_KMS)
            # vertical dashed line
            ax.axvline(x_c, color=centroid_color, lw=1.1, ls="--", zorder=4,
                       label=f"centroid {v_f:+.0f} km/s" if r == 0 else None)
            # 16/84 bar at top of panel (0.92 in axes coords)
            y_bar = ax.get_ylim()[1] * 0.0   # will be set after ylim below; use transform
            ax.annotate("", xy=(x_hi, 1.0), xytext=(x_lo, 1.0),
                        xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="|-|", color=centroid_color,
                                        lw=1.1, mutation_scale=4))

        # axes / labels
        lbl = f"{edges[r]:g}–{edges[r+1]:g} {bm}"
        ax.text(0.02, 0.88, lbl, transform=ax.transAxes, fontsize=label_fontsize,
                color="0.3",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                          boxstyle="round,pad=0.15"))
        ax.axhline(0, color=zero_color, lw=0.6)
        ax.axvline(lya_line_x, color=lya_line_color, lw=0.6, ls=":")
        ax.set_xlim(xlim_x)
        ax.grid(alpha=0.12)
        apply_tick_style(ax, direction=tick_direction, right=tick_right,
                         top=tick_top, minor=tick_minor, labelsize=tick_fontsize)

        in_xlim = (x_plot >= xlim_x[0]) & (x_plot <= xlim_x[1])
        yvals = fid[r][in_xlim]
        if band_lo is not None:
            # both edges -- the old version only looked at the lower edge
            # (p16), so the top of the band could silently clip out of view.
            yvals = np.concatenate([yvals, band_lo[r][in_xlim], band_hi[r][in_xlim]])
        fin = yvals[np.isfinite(yvals)]
        if len(fin):
            lo_, hi_ = np.nanpercentile(fin, [1, 99])
            pad = 0.25 * (hi_ - lo_)
            ax.set_ylim(min(lo_ - pad, 0), hi_ + pad)

    if ncols == 1:
        # single narrow column -- a full-width in-panel legend fits fine,
        # matches the original look exactly.
        axes[0].legend(frameon=False, fontsize=legend_fontsize, ncol=3, loc="upper right")
        title_y_final = title_y if title_y is not None else 1.005
        reserve_top = None
    else:
        # grid mode -- panels are narrower, so an in-panel legend collides
        # with the bin-range label in the corner. Pull the handles off the
        # first panel and draw ONE shared legend in a reserved strip above
        # the panel grid instead. Reserving the strip via subplots_adjust
        # AFTER tight_layout guarantees the axes grid is pushed down out of
        # the legend/title's way (tight_layout's own rect= doesn't reliably
        # leave room for manually-placed fig.legend/suptitle artists).
        # top_margin/legend_y control how much room that strip gets -- widen
        # the gap (lower top_margin and/or raise legend_y) if the legend
        # crowds the top row, which happens easily with many columns or with
        # axes that draw a scientific-notation exponent (e.g. "1e38") just
        # above their top spine.
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=legend_fontsize,
                      ncol=len(labels), loc="upper center",
                      bbox_to_anchor=(0.5, legend_y))
        title_y_final = title_y if title_y is not None else min(legend_y + 0.04, 0.995)
        reserve_top = top_margin

    # x-axis tick labels only on the bottom-most USED panel of each column;
    # hide tick labels on the panels above it so a multi-column grid doesn't
    # repeat the wavelength axis tick numbers down every row. The x-axis
    # TITLE itself is drawn once for the whole figure below (fig.supxlabel),
    # not repeated per column.
    last_row_in_col = {}
    for row, col in grid_pos:
        last_row_in_col[col] = max(last_row_in_col.get(col, -1), row)
    for row, col in grid_pos:
        if row != last_row_in_col[col]:
            axes_grid[row, col].tick_params(labelbottom=False)

    fig.supxlabel(xlabel_text_final, fontsize=xlabel_fontsize, y=xlabel_y)
    fig.supylabel(ylabel_text, fontsize=ylabel_fontsize, x=ylabel_x)
    fig.suptitle(title_text if title_text is not None
                else f"Lyα line panels — {stack_method} stack",
                fontsize=title_fontsize, y=title_y_final)
    plt.tight_layout()
    if reserve_top is not None:
        fig.subplots_adjust(top=reserve_top)
    if save_fig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes


# =====================================================================
# RADIAL PROFILE PLOTS (moved from analysis.py, 2026-09-16)
# =====================================================================

def plot_centroid_profile(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    ylims=(-250, 250),
    xlims=None,
    z_err_kms: float = 0,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    plot_literature: bool = False,
    figsize=(7.2, 4.7),
    title: str | None = r"Bootstrapped Ly$\alpha$ centroid vs. radius",
    tick_direction: str = "in",
    tick_right: bool = True,
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    The headline figure: centroid velocity ± bootstrap 16/84 per radial bin.
    Thin wrapper around plotting.plot_centroid_vs_radius with sane defaults.

    Accepts the full `boot` dict from run_measurement_suite (which also carries
    r_edges, VR_biweight_v, etc.), so you just call:

        analysis.plot_centroid_profile(boot, stacks)

    title     : axis title; pass None to omit the title entirely.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_centroid_profile.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    tick_direction, tick_right : house tick style (formatting.apply_tick_style):
                ticks point "in" by default ("out" for the old look);
                y ticks mirrored on the right spine. The top R/Rvir or
                kpc axis uses the same direction.
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present.
    """
    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    savename = _resolve_savename(save_name, "Figure_centroid_profile.png")
    return plot_centroid_vs_radius(
        boot, radial_bins, VR_biweight_v=VR_biweight_v,
        stacks_result=stacks or boot,
        bin_mode=bin_mode, z_err_kms=z_err_kms,
        vr_ticks=vr_ticks, figsize=figsize, title=title,
        ylims=ylims, xlims=xlims,
        show_vr=show_vr, VR_biweight_error=VR_biweight_error,
        plot_literature=plot_literature,
        tick_direction=tick_direction, tick_right=tick_right,
        save_fig=save_fig,
        savename=savename,
    )


def plot_flux_profile(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    logy: bool = True,
    ylims=None,
    xlims=None,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    figsize=(7.2, 4.7),
    title: str | None = "Integrated Lyα flux vs. radius",
    tick_direction: str = "in",
    tick_right: bool = True,
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Integrated Lyα flux (total_flux_fid) ± bootstrap 16/84 vs radius.
    Log-y (the default) is strongly recommended for seeing the faint outer bins;
    pass logy=False for a linear axis.

    Requires compute_side_ratio=True in the bootstrap (the default), which
    provides total_flux_fid = blue_flux_fid + red_flux_fid.

    title     : axis title; pass None to omit the title entirely.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_flux_profile.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    tick_direction, tick_right : house tick style (formatting.apply_tick_style):
                ticks point "in" by default ("out" for the old look);
                y ticks mirrored on the right spine. The top R/Rvir or
                kpc axis uses the same direction.
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present.
    """
    if "total_flux_fid" not in boot:
        raise KeyError("boot does not contain total_flux_fid; re-run with "
                       "compute_side_ratio=True (the default).")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks or boot)

    y    = np.asarray(boot["total_flux_fid"])
    y_lo = np.asarray(boot["total_flux_lo"])
    y_hi = np.asarray(boot["total_flux_hi"])

    fig, ax = plt.subplots(figsize=figsize)
    r_mid, xerr = _setup_radius_axis(ax, radial_bins, bm, vr, None, vr_ticks, xlims,
                                     show_vr=show_vr, VR_biweight_error=vr_e,
                                         tick_direction=tick_direction)
    yerr, unstable = _safe_yerr(y, y_lo, y_hi)
    ax.errorbar(r_mid, y, xerr=xerr, yerr=yerr,
                fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:blue",
                label="integrated flux (bootstrap 16–84)")
    if np.any(unstable):
        ax.scatter(r_mid[unstable], y[unstable], s=70, facecolors="none",
                   edgecolors="tab:blue", linewidths=1.3, zorder=5,
                   label="fiducial outside 16–84 band")
    ax.axhline(0, color="0.7", lw=0.7)
    if logy:
        pos = y[y > 0]
        if len(pos):
            ax.set_yscale("log")
            ax.set_ylim(pos.min() * 0.3, y.max() * 3)
    elif ylims is not None:
        ax.set_ylim(ylims)
    _ui = boot.get("unit_info") or (stacks or {}).get("unit_info") or {}
    unit = _ui.get("line_flux_unit") or "/".join(
        p for p in _ui.get("y_unit", "").split("/") if p not in ("A", "A_rest"))
    ax.set_ylabel(f"Integrated Lyα flux [{unit}]" if unit else "Integrated Lyα flux")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    apply_tick_style(ax, direction=tick_direction, right=tick_right)
    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_flux_profile.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_flux_curve_of_growth(
    cog: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    logy: bool = True,
    xlims=None,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    figsize=None,
    panels: str = "both",
    title1: str | None = _AUTO,
    title2: str | None = "Lyα flux fraction vs. radius",
    tick_direction: str = "in",
    tick_right: bool = True,
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Lyα curve of growth: cumulative luminosity vs radius (center outward), and
    the same normalized to a flux FRACTION of the total summed out to r_max.
    Both panels show the bootstrap 16-84 band.

    Takes `cog`, the dict returned by measure.flux_curve_of_growth -- this
    function only DRAWS it (measure.py computes, analysis.py draws, same split
    as every other profile here). `stacks` is optional and only used as a
    fallback for bin_mode / VR_biweight_v (cog itself does not carry them):

        cog = measure.flux_curve_of_growth(boot, stacks)   # r_max_kpc=None ->
                                                             # drops the last bin
        analysis.plot_flux_curve_of_growth(cog, stacks)

    The top panel's y-unit is derived from cog['unit_info']['y_unit'] with the
    '/kpc^2' stripped (multiplying by the fiber area cancels it) -- same
    fiber-footprint convention as plot_flux_profile's total_flux_fid, not a
    true annulus-integrated luminosity (see measure.flux_curve_of_growth).

    panels    : "both" (default, two stacked panels) | "cumulative" (top panel
                only) | "fraction" (bottom panel only). Use the single-panel
                modes to grab just one of the two images, e.g. to save it on
                its own with a distinct save_name.
    title1    : cumulative-luminosity panel title; omitted (_AUTO, the
                default) auto-generates "Lyα curve of growth (r_max = ... kpc)";
                pass a string to override, or None for no title.
    title2    : flux-fraction panel title; pass None for no title.
    figsize   : None -> (7.2, 8.0) for panels="both", (7.2, 4.5) for a single
                panel.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_flux_curve_of_growth.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    tick_direction, tick_right : house tick style (formatting.apply_tick_style):
                ticks point "in" by default ("out" for the old look);
                y ticks mirrored on the right spine. The top R/Rvir or
                kpc axis uses the same direction.
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present. (cog itself does not carry this -- it comes from
                `stacks`, same fallback as VR_biweight_v.)

    Returns (fig, (ax1, ax2)) for panels="both", else (fig, ax).
    """
    for key in ("flux_cumulative_fid", "flux_fraction_fid"):
        if key not in cog:
            raise KeyError(f"cog missing {key!r}; pass the dict returned by "
                           "measure.flux_curve_of_growth.")
    if panels not in ("both", "cumulative", "fraction"):
        raise ValueError("panels must be 'both', 'cumulative', or 'fraction'")

    radial_bins = np.asarray(r_edges if r_edges is not None else cog["r_edges_used"])
    bm  = _resolve_bin_mode(bin_mode, stacks)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks)

    if figsize is None:
        figsize = (7.2, 8.0) if panels == "both" else (7.2, 4.5)

    if panels == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    elif panels == "cumulative":
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=figsize)
        ax1 = None

    # panel 1: cumulative luminosity
    if ax1 is not None:
        r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims,
                                         show_vr=show_vr, VR_biweight_error=vr_e,
                                         tick_direction=tick_direction)
        y    = np.asarray(cog["flux_cumulative_fid"])
        y_lo = np.asarray(cog["flux_cumulative_lo"])
        y_hi = np.asarray(cog["flux_cumulative_hi"])
        yerr, unstable = _safe_yerr(y, y_lo, y_hi)
        ax1.errorbar(r_mid, y, xerr=xerr, yerr=yerr,
                    fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:green",
                    label="cumulative flux (bootstrap 16–84)")
        if np.any(unstable):
            ax1.scatter(r_mid[unstable], y[unstable], s=70, facecolors="none",
                       edgecolors="tab:green", linewidths=1.3, zorder=5,
                       label="fiducial outside 16–84 band")
        ax1.axhline(0, color="0.7", lw=0.7)
        if logy:
            pos = y[y > 0]
            if len(pos):
                ax1.set_yscale("log")
                ax1.set_ylim(pos.min() * 0.3, y.max() * 3)
        _ui = cog.get("unit_info") or {}
        base_unit = _ui.get("line_flux_unit") or "/".join(
            p for p in _ui.get("y_unit", "").split("/") if p not in ("A", "A_rest"))
        lum_unit = base_unit.replace("/kpc^2", "") if "kpc^2" in base_unit else base_unit
        ax1.set_ylabel(f"Cumulative Lyα luminosity [{lum_unit}]" if lum_unit
                       else "Cumulative Lyα luminosity")
        if title1 is _AUTO:
            ax1.set_title(f"Lyα curve of growth (r_max = {cog['meta']['r_max_kpc']:g} kpc)")
        elif title1:
            ax1.set_title(title1)
        ax1.legend(frameon=False, fontsize=9)
        ax1.grid(alpha=0.15)

    # panel 2: flux fraction
    if ax2 is not None:
        r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims,
                                           show_vr=show_vr, VR_biweight_error=vr_e,
                                         tick_direction=tick_direction)
        fy    = np.asarray(cog["flux_fraction_fid"])
        fy_lo = np.asarray(cog["flux_fraction_lo"])
        fy_hi = np.asarray(cog["flux_fraction_hi"])
        fyerr, funstable = _safe_yerr(fy, fy_lo, fy_hi)
        ax2.errorbar(r_mid2, fy, xerr=xerr2, yerr=fyerr,
                    fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:orange",
                    label="flux fraction (bootstrap 16–84)")
        if np.any(funstable):
            ax2.scatter(r_mid2[funstable], fy[funstable], s=70, facecolors="none",
                       edgecolors="tab:orange", linewidths=1.3, zorder=5)
        ax2.axhline(0.5, color="0.5", lw=0.8, ls="--", label="half-light radius")
        ax2.axhline(1.0, color="0.7", lw=0.6)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel(r"Flux fraction  $L(<r) / L(<r_{\rm max})$")
        if title2:
            ax2.set_title(title2)
        ax2.legend(frameon=False, fontsize=9)
        ax2.grid(alpha=0.15)

    apply_tick_style([ax1, ax2], direction=tick_direction, right=tick_right)

    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_flux_curve_of_growth.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    if panels == "both":
        return fig, (ax1, ax2)
    return fig, (ax1 if ax1 is not None else ax2)


def plot_asymmetry_profile(
    boot: dict,
    stacks: dict | None = None,
    r_edges=None,
    bin_mode=None,
    VR_biweight_v=None,
    vr_ticks=(0.1, 0.2, 0.5, 1, 2, 5),
    xlims=None,
    show_vr: bool = True,
    VR_biweight_error: float | None = None,
    figsize=None,
    panels: str = "both",
    title1: str | None = "Lyα asymmetry: blue flux fraction",
    title2: str | None = "Blue and red side flux vs. radius",
    tick_direction: str = "in",
    tick_right: bool = True,
    save_fig: bool = False,
    save_name: str | None = None,
):
    """
    Asymmetry diagnostic: blue fraction B/(B+R) vs radius with the 16/84
    bootstrap band, and absolute blue and red flux vs radius on the same axis
    so you can see which side is driving the asymmetry.

    fraction = blue_flux / (blue_flux + red_flux), bounded to [0, 1]:
    = 0.5 → symmetric line.
    > 0.5 → more blue flux (blueshifted / absorption-driven?).
    < 0.5 → more red flux (the expected resonant scattering signature).

    Replaces the old blue/red ratio (which ran 0→∞ and put the symmetric point
    at 1). The fraction is computed PER bootstrap draw from blue_flux_all /
    red_flux_all so the 16/84 band is correct (you cannot divide percentiles).

    panels    : "both" (default, two stacked panels) | "fraction" (blue-fraction
                panel only) | "flux" (blue/red flux panel only). Use the
                single-panel modes to grab just one of the two images.
    title1    : blue-fraction panel title; None for no title.
    title2    : blue/red-flux panel title; None for no title.
    figsize   : None -> (7.2, 8.0) for panels="both", (7.2, 4.5) for a single
                panel.
    save_name : filename (with or without extension) used when save_fig=True.
                None -> "Figure_asymmetry_profile.png".
    show_vr   : draw the dashed Rvir reference line (+ error band, if any).
    tick_direction, tick_right : house tick style (formatting.apply_tick_style):
                ticks point "in" by default ("out" for the old look);
                y ticks mirrored on the right spine. The top R/Rvir or
                kpc axis uses the same direction.
    VR_biweight_error : 1-sigma scatter (kpc) on the sample's biweight virial
                radius, shaded around the Rvir line. None -> auto-read
                stacks['VR_biweight_e'] (set by stack.sample_virial_radius_kpc)
                if present.

    Returns (fig, (ax1, ax2)) for panels="both", else (fig, ax).
    """
    if "blue_over_red_fid" not in boot:
        raise KeyError("boot missing blue/red keys; re-run with compute_side_ratio=True.")
    if panels not in ("both", "fraction", "flux"):
        raise ValueError("panels must be 'both', 'fraction', or 'flux'")

    radial_bins = np.asarray(r_edges if r_edges is not None
                             else boot.get("r_edges",
                             stacks["r_edges"] if stacks else []))
    bm  = _resolve_bin_mode(bin_mode, stacks or boot)
    vr  = _get_vr_biweight_v(VR_biweight_v, stacks or boot)
    vr_e = _get_vr_biweight_e(VR_biweight_error, stacks or boot)

    # --- blue fraction per draw -> correct 16/84 (fall back to fid-only) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        bf_fid = np.asarray(boot["blue_flux_fid"], float)
        rf_fid = np.asarray(boot["red_flux_fid"], float)
        bfrac_fid = bf_fid / (bf_fid + rf_fid)

        if "blue_flux_all" in boot and "red_flux_all" in boot:
            ball = np.asarray(boot["blue_flux_all"], float)   # (nboot, nrad)
            rall = np.asarray(boot["red_flux_all"], float)
            bfrac_all = ball / (ball + rall)
            bfrac_lo = np.nanpercentile(bfrac_all, 16, axis=0)
            bfrac_hi = np.nanpercentile(bfrac_all, 84, axis=0)
        else:
            # no per-draw arrays: approximate band from blue/red lo/hi
            warnings.warn("blue_flux_all/red_flux_all not in boot; approximating "
                          "B/total error bars from blue_flux_lo/hi.")
            bl = np.asarray(boot["blue_flux_lo"], float); bh = np.asarray(boot["blue_flux_hi"], float)
            rl = np.asarray(boot["red_flux_lo"],  float); rh = np.asarray(boot["red_flux_hi"],  float)
            bfrac_lo = bl / (bl + rh)
            bfrac_hi = bh / (bh + rl)

    if figsize is None:
        figsize = (7.2, 8.0) if panels == "both" else (7.2, 4.5)

    if panels == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    elif panels == "fraction":
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=figsize)
        ax1 = None

    # panel 1: blue fraction
    if ax1 is not None:
        r_mid, xerr = _setup_radius_axis(ax1, radial_bins, bm, vr, None, vr_ticks, xlims,
                                         show_vr=show_vr, VR_biweight_error=vr_e,
                                         tick_direction=tick_direction)
        yerr1, unstable1 = _safe_yerr(bfrac_fid, bfrac_lo, bfrac_hi)
        ax1.errorbar(r_mid, bfrac_fid, xerr=xerr, yerr=yerr1,
                     fmt="o", capsize=3.5, ms=6, lw=1.5, color="tab:purple")
        if np.any(unstable1):
            ax1.scatter(r_mid[unstable1], bfrac_fid[unstable1], s=70, facecolors="none",
                        edgecolors="tab:purple", linewidths=1.3, zorder=5,
                        label="fiducial outside 16–84 band")
        ax1.axhline(0.5, color="0.5", lw=0.8, ls="--", label="symmetric (B/total = 0.5)")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Blue fraction  B / (B+R)")
        if title1:
            ax1.set_title(title1)
        ax1.legend(frameon=False, fontsize=9)
        ax1.grid(alpha=0.15)

    # panel 2: absolute fluxes
    if ax2 is not None:
        r_mid2, xerr2 = _setup_radius_axis(ax2, radial_bins, bm, vr, None, vr_ticks, xlims,
                                           show_vr=show_vr, VR_biweight_error=vr_e,
                                         tick_direction=tick_direction)
        bf    = np.asarray(boot["blue_flux_fid"])
        rf    = np.asarray(boot["red_flux_fid"])
        bf_lo = np.asarray(boot["blue_flux_lo"]);  bf_hi = np.asarray(boot["blue_flux_hi"])
        rf_lo = np.asarray(boot["red_flux_lo"]);   rf_hi = np.asarray(boot["red_flux_hi"])
        dx = (r_mid2[1] - r_mid2[0]) * 0.05 if len(r_mid2) > 1 else 0
        bf_yerr, bf_unstable = _safe_yerr(bf, bf_lo, bf_hi)
        rf_yerr, rf_unstable = _safe_yerr(rf, rf_lo, rf_hi)
        ax2.errorbar(r_mid2 - dx, bf, xerr=xerr2, yerr=bf_yerr,
                     fmt="o", capsize=3, ms=5, lw=1.3, color="royalblue", label="blue flux")
        ax2.errorbar(r_mid2 + dx, rf, xerr=xerr2, yerr=rf_yerr,
                     fmt="s", capsize=3, ms=5, lw=1.3, color="tomato", label="red flux")
        if np.any(bf_unstable):
            ax2.scatter((r_mid2 - dx)[bf_unstable], bf[bf_unstable], s=60, facecolors="none",
                        edgecolors="royalblue", linewidths=1.3, zorder=5)
        if np.any(rf_unstable):
            ax2.scatter((r_mid2 + dx)[rf_unstable], rf[rf_unstable], s=60, facecolors="none",
                        edgecolors="tomato", linewidths=1.3, zorder=5)
        ax2.axhline(0, color="0.7", lw=0.6)
        ax2.set_ylabel("Flux")
        if title2:
            ax2.set_title(title2)
        ax2.legend(frameon=False, fontsize=9)
        ax2.grid(alpha=0.15)

    apply_tick_style([ax1, ax2], direction=tick_direction, right=tick_right)

    plt.tight_layout()
    if save_fig:
        plt.savefig(_resolve_savename(save_name, "Figure_asymmetry_profile.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    if panels == "both":
        return fig, (ax1, ax2)
    return fig, (ax1 if ax1 is not None else ax2)
