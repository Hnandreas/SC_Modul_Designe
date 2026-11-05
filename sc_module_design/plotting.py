"""Figures in the style of the dissertation (matplotlib). Every function returns a Figure.

Text in the figures is English (thesis language). Save with save_fig() as PDF/SVG/PNG.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

from . import equations as eq
from . import style
from .profile import decimate_for_plot, spectrum
from .style import (BAND, CM, COLORS, DARK_GREY, DARKGREEN, GREY, LW, MAGENTA, PINK, UIBK_BLUE,
                    UIBK_ORANGE, W1, W2, legend_below, panel_label, panel_legend)

style.apply()


def save_fig(fig: Figure, path: str | Path, formats=("pdf", "svg", "png"), exact_width: bool = True) -> list[Path]:
    """Save as PDF/SVG/PNG. exact_width: the file is exactly as wide as the figure (8 / 16 cm), only the
    height is cropped (the thesis includes the PDFs unscaled)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches=style.exact_width_bbox(fig)) if exact_width else {}
    out = []
    for f in formats:
        p = path.with_suffix("." + f)
        fig.savefig(p, **kw)
        out.append(p)
    return out


# ---------------------------------------------------------------------- free label positions
def _text_box(art: Text, renderer):
    """Display box of the text only (an annotation's arrow is not part of it)."""
    if hasattr(art, "update_positions"):
        art.update_positions(renderer)
    return Text.get_window_extent(art, renderer)


def _hits(ax, bbox, renderer, skip=()) -> int:
    """Number of artists in ax that the display box bbox covers: lines, bars, fills, markers, texts, legend."""
    n = 0
    for ln in ax.lines:
        if ln in skip or not ln.get_visible():
            continue
        path = ln.get_transform().transform_path(ln.get_path())
        v = path.vertices
        if v.size and np.isfinite(v).all() and path.intersects_bbox(bbox, filled=False):
            n += 1
        elif v.size and not np.isfinite(v).all():
            v = v[np.isfinite(v).all(axis=1)]
            n += int(np.any((v[:, 0] >= bbox.x0) & (v[:, 0] <= bbox.x1) & (v[:, 1] >= bbox.y0) & (v[:, 1] <= bbox.y1)))
        if ln.get_marker() not in (None, "None", "", " ") and v.size:
            grow = bbox.padded(ln.get_markersize() * renderer.dpi / 72 / 2)
            n += int(np.any((v[:, 0] >= grow.x0) & (v[:, 0] <= grow.x1) & (v[:, 1] >= grow.y0) & (v[:, 1] <= grow.y1)))
    for p in ax.patches:
        if p not in skip and p.get_visible() and p.get_window_extent(renderer).overlaps(bbox):
            n += 1
    for c in ax.collections:
        if c in skip or not c.get_visible():
            continue
        if isinstance(c, PolyCollection):
            tr = c.get_transform()
            n += int(any(tr.transform_path(pp).intersects_bbox(bbox, filled=True) for pp in c.get_paths()))
        elif isinstance(c, PathCollection) and len(c.get_offsets()):
            pts = c.get_offset_transform().transform(c.get_offsets())
            grow = bbox.padded(5)
            n += int(np.any((pts[:, 0] >= grow.x0) & (pts[:, 0] <= grow.x1)
                            & (pts[:, 1] >= grow.y0) & (pts[:, 1] <= grow.y1)))
    for t in ax.texts:
        if t not in skip and t.get_visible() and t.get_text() and _text_box(t, renderer).overlaps(bbox):
            n += 1
    leg = ax.get_legend()
    if leg is not None and leg.get_window_extent(renderer).overlaps(bbox):
        n += 1
    return n


def _arrow_hits(ax, art, bb, renderer) -> int:
    """Other labels crossed by the arrow of an annotation (text centre to target point)."""
    if not getattr(art, "arrowprops", None):
        return 0
    try:
        tx, ty = art._get_xy_transform(renderer, art.xycoords).transform(art.xy)   # arrow target (display)
    except Exception:
        return 0
    seg = MplPath([[(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2], [tx, ty]])
    return sum(1 for t in ax.texts if t is not art and t.get_visible() and t.get_text()
               and seg.intersects_bbox(_text_box(t, renderer).padded(2), filled=True))


def _place_free(ax, makers: list):
    """Keep the first label candidate that covers no data and stays inside the axes.

    makers: callables that each add one candidate (Text or Annotation) to ax. They are tried in the given
    order; if every candidate covers something, the one with the fewest overlaps is kept.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inner = ax.bbox.padded(1)
    best = None
    for i, mk in enumerate(makers):
        art = mk()
        bb = _text_box(art, r).padded(2)
        inside = inner.x0 <= bb.x0 and bb.x1 <= inner.x1 and inner.y0 <= bb.y0 and bb.y1 <= inner.y1
        cost = _hits(ax, bb, r, skip=(art,)) + (0 if inside else 100) + _arrow_hits(ax, art, bb, r)
        art.remove()
        if best is None or cost < best[0]:
            best = (cost, i)
        if cost == 0:
            break
    return makers[best[1]]()


def _annotate_free(ax, text: str, xy, xs=(0.25, 0.4, 0.5, 0.6, 0.75), ys=(0.94, 0.82, 0.70)):
    """Annotation with arrow to xy (data); text in the first free spot near the top (axes fraction)."""
    px = ax.transAxes.inverted().transform(ax.transData.transform(xy))[0]
    cand = [(x, y) for y in ys for x in sorted(xs, key=lambda x: abs(x - px))]
    kw = dict(xycoords="data", textcoords="axes fraction", ha="center", va="top",
              arrowprops=dict(arrowstyle="->", lw=0.7), bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    return _place_free(ax, [lambda c=c: ax.annotate(text, xy, xytext=c, **kw) for c in cand])


def _stack(n: int, width: float = W1, h_each: float = 3.6, sharex: bool = True, ratios=None):
    fig, axes = plt.subplots(n, 1, figsize=(width, (h_each + 0.45) * n * CM + 0.6 * CM), sharex=sharex,
                             gridspec_kw=dict(height_ratios=ratios) if ratios else None)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        if n > 1:
            panel_label(ax, f"({chr(97 + i)})")
        if i < n - 1:
            ax.tick_params(labelbottom=False)
    return fig, axes


# ---------------------------------------------------------------------- load profile
def fig_profile(res, t_window: Optional[tuple] = None) -> Figure:
    prof = res.profile
    fig, ax = plt.subplots(figsize=(W2, 6.0 * CM))
    t, p = prof.t, prof.p
    m = np.ones_like(t, bool) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    td, pd_ = decimate_for_plot(t[m], p[m])
    ax.plot(td, pd_, color=UIBK_BLUE, lw=1.0, label="Load")
    if res.p_bat is not None:
        tb, pb = decimate_for_plot(t[m], res.p_bat[m])
        ax.plot(tb, pb, color=UIBK_ORANGE, lw=LW, label="Battery (split)")
    ax.axhline(np.mean(p), color="k", ls="--", lw=1.0, label=f"Mean ({np.mean(p):.1f} W)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [W]")
    ax.set_xlim(td[0], td[-1])
    legend_below(fig, ncol=3)
    return fig


def fig_peaks(res) -> Figure:
    pk = res.peaks
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 6.8 * CM), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 3]))
    a1.boxplot(pk["peaks"], vert=False, widths=0.55, whis=1.5,
               flierprops=dict(marker="+", markeredgecolor=MAGENTA, markersize=5),
               medianprops=dict(color=MAGENTA, lw=LW), boxprops=dict(color=UIBK_BLUE, lw=1.0),
               whiskerprops=dict(color=UIBK_BLUE), capprops=dict(color=UIBK_BLUE))
    a1.set_yticks([])
    a1.tick_params(labelbottom=False)
    lines = (pk["upper_fence"], pk["max_peak"])      # Q3 + 1.5 IQR and largest peak
    for v in lines:
        a2.axvline(v, color="k", ls="--", lw=0.9)
    a2.hist(pk["peaks"], bins=15, color=BAND, edgecolor=UIBK_BLUE, lw=0.6)
    a2.set_xlabel("Power [W]")
    a2.set_ylabel("Number of peaks [-]")
    # values horizontally in the histogram next to their line (left first), nothing above the box plot;
    # if the lines are close, the text moves further out with a short arrow to its line
    heights = (0.97, 0.87, 0.77, 0.67, 0.57)
    for v in lines:
        plain = [lambda v=v, y=y, s=s: a2.annotate(
            f"{v:.0f} W", (v, y), xycoords=("data", "axes fraction"), xytext=(-3 * s, 0),
            textcoords="offset points", ha="right" if s > 0 else "left", va="center")
            for y in heights for s in (1, -1)]
        arrow = [lambda v=v, y=y, s=s, dx=dx: a2.annotate(
            f"{v:.0f} W", (v, y), xycoords=("data", "axes fraction"), xytext=(-dx * s, 0),
            textcoords="offset points", ha="right" if s > 0 else "left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.7, shrinkA=1, shrinkB=0))
            for y in heights for dx in (24, 40, 56) for s in (1, -1)]
        _place_free(a2, plain + arrow)
    return fig


def fig_fft(res, f_max: float = 100.0) -> Figure:
    f, a = spectrum(res.profile)
    m = (f > 0) & (f <= f_max)
    fig, ax = plt.subplots(figsize=(W1, 5.0 * CM))
    ax.semilogx(f[m], a[m], color=UIBK_BLUE, lw=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [W]")
    return fig


def fig_split_compare(t, p, splits: dict, t_window: Optional[tuple] = None) -> Figure:
    fig, ax = plt.subplots(figsize=(W2, 6.0 * CM))
    m = np.ones_like(t, bool) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    td, pd_ = decimate_for_plot(t[m], p[m])
    ax.plot(td, pd_, color=GREY, lw=0.6, label="Load")
    for i, (name, pb) in enumerate(splits.items()):
        tb, yb = decimate_for_plot(t[m], pb[m])
        ax.plot(tb, yb, lw=LW, color=COLORS[(i + 1) % len(COLORS)] if i else UIBK_BLUE, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Battery power [W]")
    ax.set_xlim(td[0], td[-1])
    legend_below(fig, ncol=3)
    return fig


# ---------------------------------------------------------------------- voltages
def fig_discharge_ratio(res=None) -> Figure:
    x = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(W1, 5.2 * CM))
    ax.plot(x, eq.energy_ratio(x), color=UIBK_BLUE)
    if res is not None and res.voltages:
        r = res.voltages["r_Q"]
        ax.plot(r, eq.energy_ratio(r), "o", color=MAGENTA, mec="white", mew=0.6, ms=5)
        ax.annotate(f"$r_Q$ = {r:.2f}: {100 * eq.energy_ratio(r):.0f} %", (r, eq.energy_ratio(r)),
                    xytext=(-60, -28), textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Discharge ratio $r_Q$ [-]")
    ax.set_ylabel("Energy ratio $W_\\mathrm{used}/W_\\mathrm{total}$ [-]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    return fig


def fig_voltage_levels(res, scenario: str = "profile_braking", series_id: Optional[str] = None) -> Figure:
    sc = res.scenarios.get(scenario) or next(iter(res.scenarios.values()))
    sid = series_id or next(iter(res.series_info))
    U_N = res.series_info[sid]["U_C0N"]
    zones = [(0, sc.U_min, "Current/\npower limit", "#e9b3e9"), (sc.U_min, sc.U_inM, "Discharging", "#a6dba0"),
             (sc.U_inM, sc.U_max, "Charging", "#a6dba0"), (sc.U_max, U_N, "Lifetime\nlimited", "#fff3a0"),
             (U_N, U_N * 1.12, "Forbidden", "#f4a6a6")]
    fig, ax = plt.subplots(figsize=(W2, 2.6 * CM))
    for a, b, lab, col in zones:
        ax.add_patch(Rectangle((a, 0), b - a, 1, facecolor=col, edgecolor="k", lw=0.6))
        ax.text((a + b) / 2, 0.5, lab, ha="center", va="center", fontsize=style.FONT_PT)
    ticks = [0, sc.U_min, sc.U_inM, sc.U_max, U_N]
    labels = ["0", f"$U_\\mathrm{{C0min}}$\n{sc.U_min:.1f} V", f"$U_\\mathrm{{C0inM}}$\n{sc.U_inM:.1f} V",
              f"$U_\\mathrm{{C0max}}$\n{sc.U_max:.1f} V", f"$U_\\mathrm{{C0N}}$\n{U_N:.1f} V"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.set_xlim(0, U_N * 1.12)
    ax.set_ylim(0, 1)
    ax.grid(False)
    return fig


def fig_lifetime(res, db, t_range=(1e3, 5e4)) -> Figure:
    req = res.inputs.req
    t = np.logspace(math.log10(t_range[0]), math.log10(t_range[1]), 400)
    fig, ax = plt.subplots(figsize=(W1, 7.0 * CM))     # taller: the legend below has four lines
    life = db.lifetime
    rows = life[life.series_id.isin(set(res.series_info)) | (life["default"].astype(str).str.lower() == "true")]
    for i, (_, s) in enumerate(rows.iterrows()):
        u = s.u0_V - (np.log2(t / s.t0_h) - (s.T0_C - req.T_cell_C) / s.dT_K) * s.dU_V
        # legend as in the thesis figure (series, T0, rated voltage, voltage step per lifetime doubling)
        lab = (f"Series {s.series}\n$T_0$ = {s.T0_C:g} °C, $U_\\mathrm{{C0N1}}$ = {s.u0_V:.1f} V, "
               f"$\\Delta U_\\mathrm{{tC01}}$ = {s.dU_V:.1f} V")
        ax.semilogx(t, np.minimum(u, s.u0_V), color=COLORS[i % len(COLORS)], label=lab)
    pts = {(float(req.t_life_h), round(float(si["U_C01max"]), 6)) for si in res.series_info.values()}
    for x, y in pts:
        ax.plot(x, y, "o", color=MAGENTA, mec="white", mew=0.6, ms=5)
    ax.set_xlabel("Estimated lifetime [h]")
    ax.set_ylabel("Max. cell voltage [V]")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    legend_below(fig, ncol=1)
    # assumed cell temperature T of Eq. (voltagelife); T_0 in the legend is the reference of the series
    t_txt = f"Assumed cell temperature $T$ = {req.T_cell_C:g} °C"
    _place_free(ax, [lambda c=c: ax.text(c[0], c[1], t_txt, transform=ax.transAxes, ha=c[2], va="top")
                     for c in ((0.98, 0.96, "right"), (0.98, 0.86, "right"), (0.5, 0.96, "center"),
                               (0.02, 0.12, "left"))])
    for x, y in sorted(pts):
        # label above the point (free area), else left of it or below; never across a curve
        offs = [(0, 26, "center", "bottom"), (0, 36, "center", "bottom"), (-22, 26, "right", "bottom"),
                (0, -24, "center", "top"), (-22, -22, "right", "top"), (-30, 0, "right", "center")]
        _place_free(ax, [lambda o=o: ax.annotate(f"{y:.2f} V", (x, y), xytext=o[:2], textcoords="offset points",
                                                 ha=o[2], va=o[3], arrowprops=dict(arrowstyle="->", lw=0.7,
                                                                                   shrinkB=4))
                         for o in offs])
    return fig


def fig_energy_excursions(res, windows: str | tuple = "auto", half_width: float = 10.0) -> Figure:
    t, p = res.profile.t, res.p_sc
    ex = res.excursions
    if windows == "iecon":
        wins = [(0, 20), (110, 130)]
    elif isinstance(windows, tuple):
        wins = list(windows)
    else:
        wins = []
        for seg in (ex.dch_segment, ex.ch_segment):
            c = t[min((seg[0] + seg[1]) // 2, len(t) - 1)]
            a = max(t[0], c - half_width)
            wins.append((a, a + 2 * half_width))
    fig, axes = _stack(2, W1, 3.4, sharex=False)
    notes = []
    for ax, (w0, w1), seg, val, col, lab in zip(
            axes, wins, (ex.dch_segment, ex.ch_segment), (ex.E_dch_J, -ex.E_ch_J), (UIBK_ORANGE, PINK),
            ("$E_\\mathrm{DCH,max}$", "$E_\\mathrm{CH,max}$")):
        m = (t >= w0) & (t <= w1)
        td, pd_ = decimate_for_plot(t[m], p[m], 8000)
        ax.plot(td, pd_, color=UIBK_BLUE, lw=1.0, label="SC power")
        a, b = seg
        ax.fill_between(t[a:b], p[a:b], color=col, alpha=0.6, lw=0,
                        label="Max. discharge excursion" if val > 0 else "Max. charge excursion")
        notes.append((ax, f"{lab} = {val:.1f} J", (t[(a + b) // 2], np.mean(p[a:b]))))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(w0, w1)
        ax.set_ylabel("SC power [W]")
        ax.tick_params(labelbottom=True)
    axes[-1].set_xlabel("Time [s]")
    legend_below(fig, ncol=2)                        # one legend for both panels, SC power only once
    for ax, txt, xy in notes:
        _annotate_free(ax, txt, xy)
    return fig


# ---------------------------------------------------------------------- trade-off (Fig. modules Cells)
GREY_OUT = "#d4d4d4"


TRADEOFF_QUANTITIES = {
    "cost_EUR": ("Module cost [Euro]", "[Euro]"),
    "volume_dm3": ("Module volume [dm$^3$]", "[dm$^3$]"),
    "pcb_area_cm2": ("PCB area [cm$^2$]", "[cm$^2$]"),
    "density_kg_dm3": ("Module density [kg/dm$^3$]", "[kg/dm$^3$]"),
    "mass_kg": ("Module mass [kg]", "[kg]"),
    "rel_volume_dm3": ("Volume vs. reference [%]", "[%]"),
    "rel_pcb_area_cm2": ("PCB area vs. reference [%]", "[%]"),
    "rel_density_kg_dm3": ("Density vs. reference [%]", "[%]"),
    "rel_mass_kg": ("Mass vs. reference [%]", "[%]"),
    "rel_cost_EUR": ("Cost vs. reference [%]", "[%]"),
    "n_cells": ("Number of cells [-]", "[-]"),
    "module_height_mm": ("Module height [mm]", "[mm]"),
}


def fig_tradeoff(res, scenarios=("profile", "profile_braking"), cmap: str = "jet", y_top: str = "cost_EUR",
                 y_bottom: str = "density_kg_dm3", size_top: str = "density_kg_dm3",
                 size_bottom: str = "cost_EUR", titles: Optional[dict] = None,
                 highlight: Optional[dict] = None, show_infeasible: bool = True,
                 cands: Optional[dict] = None) -> Figure:
    """IECON-style overview (Modules_Cells). Ball size in the foreground, colour = module ESR.

    Configurations that violate the specification (or the thermal / P limit) are drawn greyed out.
    With geometry 'pcb' the density is the module density (cells + PCB in the PCB box), else the cell density.
    """
    cands = cands or res.candidates
    scen = [s for s in scenarios if s in cands]
    titles = titles or {"profile": "Module 1", "profile_braking": "Module 2"}
    alls = [cands[s] for s in scen]
    dfs = [d[d.feasible] for d in alls]
    bad = [d[~d.feasible] for d in alls]
    ok_esr = np.concatenate([d.ESR_module_mOhm.to_numpy() for d in dfs]) if any(len(d) for d in dfs) else \
        np.concatenate([d.ESR_module_mOhm.to_numpy() for d in alls])
    norm = Normalize(np.nanmin(ok_esr), np.nanmax(ok_esr))
    module_geo = getattr(res.inputs.opt, "geometry", "pcb") == "pcb"
    dens_lab = "Module density [kg/dm$^3$]" if module_geo else "Cell density [kg/dm$^3$]"
    units = {k: v[1] for k, v in TRADEOFF_QUANTITIES.items()}
    units["E_density_Wh_dm3"] = "[Wh/dm$^3$]"
    ylab = {k: v[0] for k, v in TRADEOFF_QUANTITIES.items()}
    ylab["density_kg_dm3"] = dens_lab
    ylab["E_density_Wh_dm3"] = "Module energy density [Wh/dm$^3$]"

    def rng(col):
        src = alls if show_infeasible else dfs
        v = np.concatenate([d[col].to_numpy(float) for d in src])
        return np.nanmin(v), np.nanmax(v)

    fig = plt.figure(figsize=(W1, 9.6 * CM))
    gs = fig.add_gridspec(3, len(scen) + 1, height_ratios=[1, 1, 0.07], width_ratios=[1] * len(scen) + [0.32])
    handle = None
    any_bad = False
    for i, (ycol, scol) in enumerate(((y_top, size_top), (y_bottom, size_bottom))):
        smin, smax = rng(scol)

        def sizes(v):
            return 18 + (v - smin) / (smax - smin) * 70 if smax > smin else np.full_like(v, 45)
        for j, (s, d, db_) in enumerate(zip(scen, dfs, bad)):
            ax = fig.add_subplot(gs[i, j])
            if show_infeasible and len(db_):
                any_bad = True
                vb = sizes(db_[scol].to_numpy(float))
                ax.scatter(db_.c_nom_F, db_[ycol], s=vb, facecolors=GREY_OUT, edgecolors="#9a9a9a", linewidths=0.35,
                           zorder=3)
            v = d[scol].to_numpy(float)
            size = sizes(v)
            order = np.argsort(-size)                      # large balls first, small ones on top
            if len(d):
                handle = ax.scatter(d.c_nom_F.to_numpy()[order], d[ycol].to_numpy()[order], s=size[order],
                                    c=d.ESR_module_mOhm.to_numpy()[order], cmap=cmap, norm=norm, edgecolors="k",
                                    linewidths=0.35, zorder=5)
            marks = dict(highlight or {})
            sel = (res.inputs.selected or {}).get(s)
            if sel:
                marks.setdefault(s, sel.get("cell_id"))
            cid = marks.get(s)
            if cid:
                r = d[d.cell_id == cid]
                if not r.empty:
                    ax.scatter(r.c_nom_F, r[ycol], s=150, facecolors="none", edgecolors="k", lw=1.0, zorder=6)
            if ycol.startswith("rel_"):
                ax.axhline(0, color="k", lw=0.8, ls="--", zorder=1)
            if i == 0:
                ax.text(0.5, 1.03, titles.get(s, s), transform=ax.transAxes, ha="center", va="bottom")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Cell capacitance [F]")
            if j == 0:
                ax.set_ylabel(ylab.get(ycol, ycol))
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlim(0, 108)
        lax = fig.add_subplot(gs[i, -1])
        lax.axis("off")
        lax.scatter([0.15, 0.15, 0.15], [0.1, 0.5, 0.9], s=[18, 53, 88], c=["k", "none", "k"], edgecolors="k",
                    lw=0.5, zorder=5, clip_on=False)
        fmt = ("{:.0f}" if max(abs(smax), abs(smin)) >= 100 or scol.startswith("rel_")
               or scol in ("n_cells", "module_height_mm") else "{:.2f}")
        lax.text(0.55, 0.9, fmt.format(smax), va="center")
        lax.text(0.55, 0.1, fmt.format(smin), va="center")
        lax.text(1.4, 0.5, units.get(scol, scol), rotation=90, va="center", ha="center")
        lax.set_xlim(0, 1.7)
        lax.set_ylim(-0.1, 1.1)
    # share the y range per row so that both modules are comparable
    axs = [a for a in fig.axes if a.get_xlim() == (0.0, 108.0)]
    for r in range(2):
        row = axs[r * len(scen):(r + 1) * len(scen)]
        lo = min(a.get_ylim()[0] for a in row)
        hi = max(a.get_ylim()[1] for a in row)
        for a in row:
            a.set_ylim(lo, hi)
    cax = fig.add_subplot(gs[2, : len(scen)])
    if handle is not None:
        cb = fig.colorbar(handle, cax=cax, orientation="horizontal")
        cb.set_label("Module ESR [m$\\Omega$]")
        cb.outline.set_edgecolor(GREY)
    if any_bad:
        fig.legend([plt.Line2D([], [], ls="", marker="o", ms=5, mfc=GREY_OUT, mec="#9a9a9a")],
                   ["Outside specification"], loc="upper center", bbox_to_anchor=(0.45, -0.005), frameon=False)
    return fig


CRITERIA_EN = {"cost": "Cost", "mass": "Mass", "volume": "Volume", "losses": "Losses", "thermal": "Temperature rise",
               "cells": "Number of cells", "reserve": "Capacitance reserve", "data": "Measured data"}


def fig_scores(ranked, contributions: dict, labels: Optional[dict] = None, top: int = 8) -> Figure:
    """Horizontal stacked bars: weighted score contributions per criterion (higher = better)."""
    d = ranked.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(W2, (1.2 + 0.55 * len(d)) * CM + 1.2 * CM))
    left = np.zeros(len(d))
    for i, (k, vals) in enumerate(contributions.items()):
        v = vals.loc[d.index].to_numpy()
        ax.barh(np.arange(len(d)), v, left=left, color=COLORS[i % len(COLORS)], edgecolor="white", lw=0.5,
                label=(labels or CRITERIA_EN).get(k, k))
        left += v
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([f"{c.replace('VISHAY_HVR235_', '')}  {s}" for c, s in zip(d.cell_id, d.setup)])
    ax.set_xlabel("Weighted score [%]")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    legend_below(fig, ncol=4)
    return fig


# ---------------------------------------------------------------------- C(U)
def fig_cu(fit: dict, cell: dict, u_lo: float, u_hi: float, c_eq: float) -> Figure:
    u = np.linspace(0, cell["u_rated_V"], 200)
    fig, ax = plt.subplots(figsize=(W1, 5.2 * CM))
    ax.axvspan(u_lo, u_hi, color=BAND, alpha=0.45, lw=0, label="Cell voltage range")
    ax.plot(u, fit["C0_F"] + fit["k_FperV"] * u, color=UIBK_BLUE, label="$C(U) = C_0 + kU$")
    ax.axhline(cell["c_nom_F"], color="k", ls="--", lw=1.0, label="Rated $C$")
    ax.hlines(c_eq, u_lo, u_hi, color=UIBK_ORANGE, lw=2.0, label="$C_\\mathrm{eq,E}$")
    ax.set_xlabel("Cell voltage [V]")
    ax.set_ylabel("Capacitance [F]")
    legend_below(fig, ncol=2)
    return fig


# ---------------------------------------------------------------------- thermal
def fig_thermal(det, sc, dT_max: float, title: str = "") -> Figure:
    tr = det.current
    t = tr.t
    fig, axes = _stack(3, W1, 3.0)
    axes[0].plot(t, np.sqrt(tr.I_sq), color=UIBK_BLUE, lw=0.7, label="Module current (RMS per step)")
    axes[0].set_ylabel("Module\ncurrent [A]")
    axes[1].plot(t, tr.U, color=UIBK_ORANGE, label="Module voltage")
    axes[1].axhline(sc.U_min, color="k", ls="--", lw=0.8, label="$U_\\mathrm{C0min}$, $U_\\mathrm{C0max}$")
    axes[1].axhline(sc.U_max, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Module\nvoltage [V]")
    axes[2].plot(t, det.dT, color=MAGENTA, label="Cell temperature rise")
    axes[2].axhline(dT_max, color="k", ls="--", lw=1.0, label="Limit")
    axes[2].set_ylabel("$\\Delta T$ [K]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylim(0, max(dT_max, float(np.max(det.dT))) * 1.12)
    axes[0].set_xlim(t[0], t[-1])
    for ax in axes:
        panel_legend(ax, ncol=2)
    return fig


def fig_thermal_breakdown(steps: list, dT_max: float) -> Figure:
    """Cell temperature rise step by step (model assumptions)."""
    fig, ax = plt.subplots(figsize=(W2, (1.0 + 0.6 * len(steps)) * CM + 1.0 * CM))
    y = np.arange(len(steps))[::-1]
    vals = [s[1] for s in steps]
    prev = 0.0
    for yi, (lab, v, *_), c in zip(y, steps, [UIBK_BLUE, LIGHT(), UIBK_ORANGE, MAGENTA, DARKGREEN]):
        ax.barh(yi, v, color=c, edgecolor="white", height=0.62)
        ax.text(v + 0.4, yi, f"{v:.1f} K", va="center")
        prev = v
    ax.axvline(dT_max, color="k", ls="--", lw=1.0)
    ax.text(dT_max, y[0] + 0.55, " Limit", va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in steps])
    ax.set_xlabel("Max. cell temperature rise [K]")
    ax.set_xlim(0, max(dT_max, max(vals)) * 1.18)
    ax.grid(axis="y", visible=False)
    return fig


def LIGHT():
    return "#5b8fbf"


def fig_dT_candidates(df, dT_max: float) -> Figure:
    """Max. cell temperature rise per candidate; candidates outside the specification are greyed out."""
    d = df[df.dT_max_K.notna()]
    fig, ax = plt.subplots(figsize=(W2, 5.5 * CM))
    x = np.arange(len(d))
    feas = d.feasible.to_numpy() if "feasible" in d else np.ones(len(d), bool)
    cols = [GREY_OUT if not f else (MAGENTA if v > dT_max else UIBK_BLUE) for v, f in zip(d.dT_max_K, feas)]
    ax.bar(x, d.dT_max_K, color=cols, edgecolor="white", width=0.7)
    ax.axhline(dT_max, color="k", ls="--", lw=1.0, label=f"Limit {dT_max:g} K")
    if not feas.all():
        ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=GREY_OUT, label="Outside specification"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:g} F\n{s}" for c, s in zip(d.c_nom_F, d.setup)])
    ax.set_ylabel("Max. cell temperature rise [K]")
    ax.grid(axis="x", visible=False)
    panel_legend(ax)
    return fig


# ---------------------------------------------------------------------- emulation
def fig_emulated_profile(trace, module, sc=None, t_window: Optional[tuple] = None, i_limit: Optional[float] = None,
                         u_conv_min: Optional[float] = None, u_bus: Optional[float] = None,
                         u_bus_used: Optional[float] = None, thermal: bool = True, net: Optional[dict] = None,
                         converter_max_V: Optional[float] = None, battery_reference_W=None
                         ) -> Figure:
    """Emulated HESS over one periodic cycle.

    (a) voltages: bus/battery voltage, SC internal voltage U_C and terminal voltage U_C + R I
    (b) bus currents: load, battery and SC on the bus side (P / U_bus)
    (c) SC currents: module current I_SC and bus-side current I_SC,bus
    (d) optional: cell temperature rise (thermal network if available, else block model)
    """
    t = trace.t - trace.t[0]
    m = np.ones_like(t, bool) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    ub = u_bus_used or u_bus or 48.0
    has_dT = thermal and ((net is not None) or (trace.dT is not None and np.any(trace.dT[m] != 0)))
    n = 4 if has_dT else 3
    fig, axes = _stack(n, W2, 3.3)
    ax = axes[0]
    if u_bus:
        ax.axhline(u_bus, color=UIBK_ORANGE, lw=LW, label="Bus / battery $U_\\mathrm{Bus}$")
    if u_bus_used and u_bus and abs(u_bus_used - u_bus) > 1e-9:
        ax.axhline(u_bus_used, color=UIBK_ORANGE, ls=":", lw=1.0, label="$U_\\mathrm{Bus,min}$ (conservative)")
    if converter_max_V is not None:
        ax.axhline(converter_max_V, color=GREY, ls=":", lw=0.8, label="Converter ceiling $D_\\max U_\\mathrm{Bus}$")
    ax.plot(t[m], trace.U_sc[m], color=UIBK_BLUE, label="SC internal $U_C$")
    ax.plot(t[m], trace.U_term[m], color=PINK, lw=0.6, label="SC terminal $U_C + R I$")
    ax.axhline(module.U_min, color="k", ls="--", lw=0.8, label="$U_\\mathrm{C0min}$, $U_\\mathrm{C0max}$")
    ax.axhline(module.U_max, color="k", ls="--", lw=0.8)
    if u_conv_min:
        ax.axhline(u_conv_min, color=MAGENTA, ls="-.", lw=0.9, label="Converter min. input")
    ax.set_ylabel("Voltage [V]")
    ax = axes[1]
    ax.plot(t[m], trace.P_load[m] / ub, color=GREY, lw=0.6, label="Load $I_\\mathrm{Last}$")
    ax.plot(t[m], trace.P_bat[m] / ub, color=UIBK_ORANGE, lw=LW, label="Battery $I_\\mathrm{Bat}$")
    if battery_reference_W is not None:
        ax.plot(t[m], np.asarray(battery_reference_W)[m] / ub, color=GREY, ls="--", lw=0.8,
                label="Battery filter reference (without PI)")
    ax.plot(t[m], trace.P_sc[m] / ub, color=UIBK_BLUE, lw=0.8, label="SC bus side $I_\\mathrm{SC,Bus}$")
    ax.set_ylabel("Bus current, conservative [A]" if u_bus and abs(ub-u_bus) > 1e-9 else "Bus current [A]")
    ax = axes[2]
    ax.plot(t[m], trace.I_sc[m], color=UIBK_BLUE, lw=0.8, label="SC module $I_\\mathrm{SC}$ (+ charge / − discharge)")
    ax.plot(t[m], trace.I_bus[m], color=UIBK_ORANGE, lw=0.7, label="SC bus side $I_\\mathrm{SC,Bus}$")
    if i_limit:
        ax.axhline(i_limit, color="k", ls="--", lw=0.8, label="Current limit $I_\\mathrm{C0max}$")
        ax.axhline(-i_limit, color="k", ls="--", lw=0.8)
    limited = getattr(trace, "current_limited", None)
    requested = getattr(trace, "requested_current_extreme_A", None)
    if limited is not None and requested is not None and i_limit:
        hit = m & limited
        if np.any(hit):
            ax.scatter(t[hit], np.sign(requested[hit]) * i_limit, color=MAGENTA, s=15,
                       zorder=6, label="SC current limiter active")
            ax.scatter(t[hit], requested[hit], color=MAGENTA, marker="x", s=12,
                       zorder=6, label="Requested current before limit")
    ax.set_ylabel("SC current [A]")
    if has_dT:
        ax = axes[3]
        if net is not None:
            T = net["T"]
            T_inf = float(net["steady"].params.T_inf_C)
            tn = net["t"] - net["t"][0]
            tn = tn * (t[-1] / tn[-1]) if tn[-1] > 0 else tn
            ax.plot(tn, T[:, net["hot"]] - T_inf, color=MAGENTA, label="Hottest cell (network)")
            ax.plot(tn, T[:, net["outer"]].mean(1) - T_inf, color=UIBK_BLUE, label="Outer cells (network)")
            top = float(T.max() - T_inf)
        else:
            ax.plot(t[m], trace.dT[m], color=MAGENTA, label="Cell temperature rise (block)")
            top = float(np.max(trace.dT[m]))
        ax.set_ylabel("$\\Delta T$ [K]")
        ax.set_ylim(0, top * 1.15 if top > 0 else 1)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_xlim(t[m][0], t[m][-1])
    for ax in axes:
        panel_legend(ax, ncol=3)
    return fig


def fig_emulation(trace, history, module=None) -> Figure:
    """Emulated cycle plus convergence of the endless loop."""
    fig = plt.figure(figsize=(W2, 15 * CM))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.9])
    t = trace.t
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for i, ax in enumerate(axes):
        panel_label(ax, f"({chr(97 + i)})")
    axes[0].plot(t, trace.P_load, color=GREY, lw=0.6, label="Load")
    axes[0].plot(t, trace.P_bat, color=UIBK_ORANGE, lw=LW, label="Battery")
    axes[0].plot(t, trace.P_sc, color=UIBK_BLUE, lw=0.8, label="Supercapacitor")
    axes[0].set_ylabel("Power [W]")
    axes[1].plot(t, trace.U_sc, color=UIBK_BLUE, label="SC module voltage")
    if module is not None:
        axes[1].axhline(module.U_min, color="k", ls="--", lw=0.8, label="Voltage limits")
        axes[1].axhline(module.U_max, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Voltage [V]")
    axes[2].plot(t, trace.dT, color=MAGENTA, label="Cell temperature rise")
    axes[2].set_ylabel("$\\Delta T$ [K]")
    axes[2].set_xlabel("Time [s]")
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:3]:
        ax.set_xlim(t[0], t[-1])
    cyc = [h.cycle + 1 for h in history]
    axes[3].plot(cyc, [h.U_end for h in history], "o-", color=UIBK_BLUE, ms=3.5, label="Voltage at cycle end")
    a2 = axes[3].twinx()
    a2.plot(cyc, [h.dT_max for h in history], "s-", color=MAGENTA, ms=3.5, label="Max. temperature rise")
    a2.grid(False)
    a2.set_ylabel("$\\Delta T_\\mathrm{max}$ [K]")
    axes[3].set_xlabel("Cycle [-]")
    axes[3].set_ylabel("Voltage [V]")
    axes[3].ticklabel_format(axis="y", useOffset=False)
    axes[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes[:3]:
        panel_legend(ax, ncol=3)
    h1, l1 = axes[3].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    axes[3].legend(h1 + h2, l1 + l2, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                   borderaxespad=0.15)
    return fig


def fig_emudesign(iterations: list, dT_max: float, title_cells: str = "", P_used: Optional[int] = None) -> Figure:
    """Energy swing / C_req and temperature over P (P_used: the checked module, marked)."""
    P = [i.P for i in iterations]
    fig, axes = _stack(2, W1, 3.4)
    axes[0].plot(P, [i.C_req_emu_F for i in iterations], "o-", color=UIBK_BLUE, label="$C_\\mathrm{req,emu}$")
    axes[0].plot(P, [i.C_module_F for i in iterations], "s--", color=UIBK_ORANGE, label="$C_\\mathrm{module}$")
    axes[0].set_ylabel("Capacitance [F]")
    axes[1].plot(P, [i.dT_max_K for i in iterations], "o-", color=MAGENTA, label="$\\Delta T_\\mathrm{max}$")
    axes[1].axhline(dT_max, color="k", ls="--", lw=0.9, label="Limit")
    axes[1].set_ylabel("$\\Delta T$ [K]")
    axes[1].set_xlabel("Parallel strings P [-]")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    if P_used is not None:
        for ax in axes:
            ax.axvline(P_used, color="0.5", ls=":", lw=1.0, label=f"Module P = {P_used}" if ax is axes[1] else None)
    for ax in axes:
        panel_legend(ax, ncol=2)
    return fig


def fig_cell_thermal_chain(chain: dict, dT_max: Optional[float] = None, mean_window_s: float = 1.0) -> Figure:
    """Cell current -> Joule loss in the ESR -> thermal low-pass (R_th, C_th) -> temperature rise."""
    t = chain["t"]
    dt = t[1] - t[0]
    fig, axes = _stack(3, W2, 3.4)
    td, yd = decimate_for_plot(t, chain["i_cell"], 30000)
    axes[0].plot(td, yd, color=UIBK_BLUE, lw=0.6, label="Cell current")
    axes[0].axhline(chain["I_rms"], color=UIBK_ORANGE, ls="--", lw=1.0, label=f"RMS {chain['I_rms']:.2f} A")
    axes[0].set_ylabel("Cell current [A]")
    td, yd = decimate_for_plot(t, chain["p_v"], 30000)
    axes[1].plot(td, yd, color=PINK, lw=0.5, label="Loss per cell $i^2 R_\\mathrm{ESR}$")
    n = max(1, int(round(mean_window_s / dt)))
    c = np.concatenate([[0.0], np.cumsum(chain["p_v"])])
    mov = (c[n:] - c[:-n]) / n
    axes[1].plot(t[n - 1:][:: max(1, len(mov) // 20000)], mov[:: max(1, len(mov) // 20000)], color=MAGENTA, lw=1.2,
                 label=f"Loss, {mean_window_s:g} s mean")
    axes[1].axhline(chain["P_v_mean"], color="k", ls="--", lw=0.9, label=f"Mean {chain['P_v_mean'] * 1000:.0f} mW")
    axes[1].set_ylabel("Loss per cell [W]")
    axes[1].set_ylim(0, max(float(np.percentile(chain["p_v"], 99.9)) * 1.1, chain["P_v_mean"] * 3))
    td, yd = decimate_for_plot(t, chain["dT"], 20000)
    axes[2].plot(td, yd, color=MAGENTA, label=f"Temperature rise after thermal low-pass (max {np.max(chain['dT']):.1f} K)")
    if dT_max:
        axes[2].axhline(dT_max, color="k", ls="--", lw=1.0, label=f"Limit {dT_max:g} K")
    axes[2].set_ylim(0, max(float(np.max(chain["dT"])) * 1.2, (dT_max or 0) * 1.05))
    axes[2].set_ylabel("$\\Delta T$ [K]")
    axes[2].set_xlabel("Time [s]")
    axes[0].set_xlim(t[0], t[-1])
    for ax in axes:
        panel_legend(ax, ncol=3)
    return fig


def _draw_layout(ax, lay, show_path: bool = True) -> None:
    from matplotlib.patches import Circle
    ax.add_patch(Rectangle((0, 0), lay.width_mm, lay.depth_mm, facecolor="#eef3ea", edgecolor=DARKGREEN, lw=0.9,
                           zorder=0))
    for x, y, g, k in lay.cells:
        col = UIBK_BLUE if g % 2 == 0 else "#5b8fbf"
        ax.add_patch(Circle((x, y), lay.d_mm / 2, facecolor=col, edgecolor="#1a1a1a", lw=0.4, zorder=2))
    for c in lay.connectors:
        ax.add_patch(Rectangle((c["x"], c["y"]), c["w"], c["h"], facecolor="#3a3a3a", edgecolor="k", lw=0.6,
                               zorder=4))
    if show_path and len(lay.path) > 1:
        px, py = zip(*lay.path)
        ax.plot(px, py, color=UIBK_ORANGE, lw=1.3, zorder=3)
    for (x, y), lab, c in ((lay.terminal_plus, "+", MAGENTA), (lay.terminal_minus, "\u2212", "#6e6e6e")):
        ax.plot(x, y, "s", color=c, ms=6.0, zorder=5)
        ax.text(x, y, lab, color="white", ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=6)
    ax.set_xlim(-2, lay.width_mm + 2)
    ax.set_ylim(-2, lay.depth_mm + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Width [mm]")
    ax.grid(False)


def _layout_handles(ax, lay) -> None:
    ax.plot([], [], "o", color=UIBK_BLUE, ms=6, label=f"Cell, d = {lay.d_mm:g} mm, gap {lay.gap_mm:g} mm "
                                                     f"(groups of {lay.P} in parallel)")
    ax.plot([], [], color=UIBK_ORANGE, lw=1.3, label="Series connection (U, back to the front)")
    if lay.connectors:
        ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="#3a3a3a", edgecolor="k", lw=0.6,
                               label="Connector in free space" if lay.connectors_in_free_space
                               else "Connector (front strip)"))
    ax.plot([], [], "s", color=MAGENTA, ms=6, label="Terminals +/\u2212")


def _module_text(lay, mass_cells_kg: Optional[float], pcb_g_cm2: float) -> str:
    txt = f"{lay.width_mm:.0f} x {lay.depth_mm:.0f} mm, {lay.area_mm2 / 100:.1f} cm$^2$, {lay.volume_dm3:.2f} dm$^3$"
    if mass_cells_kg is not None and mass_cells_kg == mass_cells_kg:
        m = mass_cells_kg + lay.pcb_mass_kg(pcb_g_cm2)
        txt += f", {m / lay.volume_dm3:.2f} kg/dm$^3$"
    return txt


def fig_pcb_layout(lay, cell_label: str = "", mass_cells_kg: Optional[float] = None,
                   pcb_g_cm2: float = 0.35, extent: Optional[tuple] = None) -> Figure:
    """Top view of the module PCB: cells, parallel groups, U-shaped series path, connectors at the front.
    extent = (width, depth) in mm: common scale for all modules (largest module fills the frame)."""
    W, D = (max(extent[0], lay.width_mm), max(extent[1], lay.depth_mm)) if extent else (lay.width_mm, lay.depth_mm)
    w_cm = 8.0 if W <= 1.3 * D else 12.0
    fig, ax = plt.subplots(figsize=(w_cm * CM, w_cm * CM * D / W + 2.6 * CM))
    _draw_layout(ax, lay)
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(-2, D + 2)
    ax.set_ylabel("Depth [mm]")
    ax.text(0.5, 1.02, (cell_label + ": " if cell_label else "") + _module_text(lay, mass_cells_kg, pcb_g_cm2),
            transform=ax.transAxes, ha="center", va="bottom")
    _layout_handles(ax, lay)
    legend_below(fig, ncol=2)
    return fig


def fig_layout_compare(sq, hx, mass_cells_kg: Optional[float] = None, pcb_g_cm2: float = 0.35,
                       extent: Optional[tuple] = None) -> Figure:
    """Square versus hexagonal (offset) packing of the same module: module (PCB) area, volume and density.
    extent = (width, depth) in mm: common scale for all modules (largest module fills the frame)."""
    W = max(sq.width_mm, hx.width_mm, extent[0] if extent else 0)
    D = max(sq.depth_mm, hx.depth_mm, extent[1] if extent else 0)
    wide = W > 1.3 * D
    if wide:
        aw = W2 - 1.6 * CM
        fig, axes = plt.subplots(2, 1, figsize=(W2, 2 * aw * D / W + 4.2 * CM), sharex=True)
    else:
        aw = (W2 - 2.2 * CM) / 2
        fig, axes = plt.subplots(1, 2, figsize=(W2, aw * D / W + 3.6 * CM), sharey=True)
    a_sq, a_hx = sq.area_mm2, hx.area_mm2
    for ax, lay, lab, name in ((axes[0], sq, "(a)", "Square"), (axes[1], hx, "(b)", "Hexagonal (offset)")):
        _draw_layout(ax, lay)
        panel_label(ax, lab)
        if lay is hx:
            name += f" ({100 * (a_hx / a_sq - 1):+.1f} % area)".replace("-", "\u2212")
        sep = ": " if wide else "\n"
        ax.text(0.5, 1.02, name + sep + _module_text(lay, mass_cells_kg, pcb_g_cm2), transform=ax.transAxes,
                ha="center", va="bottom", linespacing=1.3)
        ax.set_xlim(-2, W + 2)
        ax.set_ylim(-2, D + 2)
        if wide and lay is sq:
            ax.set_xlabel("")
    axes[0].set_ylabel("Depth [mm]")
    if wide:
        axes[1].set_ylabel("Depth [mm]")
    _layout_handles(axes[0], sq)
    legend_below(fig, ncol=2)
    return fig


# ---------------------------------------------------------------------- thermal network (precheck)
def fig_network_heatmap(net, extent: Optional[tuple] = None, title: str = "") -> Figure:
    """Steady temperature rise of every cell (thermal network, Andy's specification), top view."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Circle
    g, p = net.grid, net.params
    dT = net.T - p.T_inf_C
    xy = g.xy
    r = p.d_mm / 2
    x0, y0 = xy[:, 0].min() - r - 3, xy[:, 1].min() - r - 3
    W = xy[:, 0].max() + r + 3 - x0
    D = xy[:, 1].max() + r + 3 - y0
    if extent:
        W, D = max(W, extent[0]), max(D, extent[1])
    w_cm = 8.0 if W <= 1.3 * D else 12.0
    fig, ax = plt.subplots(figsize=(w_cm * CM + 1.6 * CM, w_cm * CM * D / W + 1.8 * CM))
    norm = Normalize(0, max(float(dT.max()), 1e-3))
    cmap = plt.get_cmap("plasma")
    for (x, y), v in zip(xy, dT):
        ax.add_patch(Circle((x, y), r, facecolor=cmap(norm(v)), edgecolor="#1a1a1a", lw=0.4))
    ax.plot(*xy[g.centre], "x", color="white", ms=5, mew=1.2)
    ax.set_xlim(x0, x0 + W)
    ax.set_ylim(y0, y0 + D)
    ax.set_aspect("equal")
    ax.set_xlabel("Width [mm]")
    ax.set_ylabel("Depth [mm]")
    ax.grid(False)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Temperature rise $\\Delta T$ [K]")
    cb.outline.set_edgecolor(GREY)
    s = net.summary()
    ax.text(0.5, 1.02, title or (f"Centre {s['dT_centre_K']:.1f} K, outer mean {s['dT_outer_mean_K']:.1f} K, "
                                 f"max {s['dT_max_K']:.1f} K"), transform=ax.transAxes, ha="center", va="bottom")
    return fig


def fig_network_transient(t, T, grid, T_inf: float, dT_max: Optional[float] = None) -> Figure:
    """Heating of the module: centre cell, hottest cell and mean of the outer cells."""
    fig, ax = plt.subplots(figsize=(W2, 5.0 * CM))
    tm = t / 60
    dT = T - T_inf
    hot = int(np.argmax(dT[-1]))
    ax.plot(tm, dT[:, grid.centre], color=MAGENTA, label="Centre cell")
    if hot != grid.centre:
        ax.plot(tm, dT[:, hot], color=UIBK_ORANGE, ls="--", label="Hottest cell")
    ax.plot(tm, dT[:, grid.outer].mean(1), color=UIBK_BLUE, label="Outer cells (mean)")
    if dT_max:
        ax.axhline(dT_max, color="k", ls="--", lw=0.9, label=f"Limit {dT_max:g} K")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("$\\Delta T$ [K]")
    ax.set_xlim(0, tm[-1])
    ax.set_ylim(0, None)
    panel_legend(ax, ncol=4)
    return fig


def fig_network_layers(rows: list, dT_max: Optional[float] = None) -> Figure:
    """Steady temperatures over the number of neighbour layers (layer convergence)."""
    fig, ax = plt.subplots(figsize=(W1, 5.2 * CM))
    n = [r["layers"] for r in rows]
    ax.plot(n, [r["dT_centre_K"] for r in rows], "o-", color=MAGENTA, ms=3, label="Centre cell")
    ax.plot(n, [r["dT_outer_mean_K"] for r in rows], "s-", color=UIBK_BLUE, ms=3, label="Outer layer (mean)")
    if dT_max:
        ax.axhline(dT_max, color="k", ls="--", lw=0.9, label=f"Limit {dT_max:g} K")
    ax.set_xlabel("Neighbour layers [-]")
    ax.set_ylabel("$\\Delta T$ [K]")
    panel_legend(ax, ncol=2)
    return fig
