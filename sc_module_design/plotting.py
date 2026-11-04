"""Figures in the style of the dissertation (matplotlib). Every function returns a Figure.

Text in the figures is English (thesis language). Save with save_fig() as PDF/SVG/PNG.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from . import equations as eq
from . import style
from .profile import decimate_for_plot, spectrum
from .style import (BAND, CM, COLORS, DARK_GREY, DARKGREEN, GREY, LW, MAGENTA, PINK, UIBK_BLUE,
                    UIBK_ORANGE, W1, W2, legend_below, panel_label, panel_legend)

style.apply()


def save_fig(fig: Figure, path: str | Path, formats=("pdf", "svg", "png")) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = path.with_suffix("." + f)
        fig.savefig(p)
        out.append(p)
    return out


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
    for v in (pk["upper_fence"], pk["max_peak"]):
        a1.annotate(f"{v:.0f} W", (v, 1.33), ha="center", va="bottom", fontsize=style.FONT_PT)
        a2.axvline(v, color="k", ls="--", lw=0.9)
    a2.hist(pk["peaks"], bins=15, color=BAND, edgecolor=UIBK_BLUE, lw=0.6)
    a2.set_xlabel("Power [W]")
    a2.set_ylabel("Number of peaks [-]")
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
    fig, ax = plt.subplots(figsize=(W1, 6.2 * CM))
    life = db.lifetime
    rows = life[life.series_id.isin(set(res.series_info)) | (life["default"].astype(str).str.lower() == "true")]
    for i, (_, s) in enumerate(rows.iterrows()):
        u = s.u0_V - (np.log2(t / s.t0_h) - (s.T0_C - req.T_cell_C) / s.dT_K) * s.dU_V
        ax.semilogx(t, np.minimum(u, s.u0_V), color=COLORS[i % len(COLORS)],
                    label=f"Series {s.series.split()[0]}, $U_\\mathrm{{C0N1}}$ = {s.u0_V:g} V")
    for si in res.series_info.values():
        ax.plot(req.t_life_h, si["U_C01max"], "o", color=MAGENTA, mec="white", mew=0.6, ms=5)
        ax.annotate(f"{si['U_C01max']:.2f} V", (req.t_life_h, si["U_C01max"]), xytext=(-42, 22),
                    textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Estimated lifetime [h]")
    ax.set_ylabel("Max. cell voltage [V]")
    ax.text(0.98, 0.95, f"T = {req.T_cell_C:g} °C", transform=ax.transAxes, ha="right", va="top")
    legend_below(fig, ncol=1)
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
    for ax, (w0, w1), seg, val, col, lab in zip(
            axes, wins, (ex.dch_segment, ex.ch_segment), (ex.E_dch_J, -ex.E_ch_J), (UIBK_ORANGE, PINK),
            ("$E_\\mathrm{DCH,max}$", "$E_\\mathrm{CH,max}$")):
        m = (t >= w0) & (t <= w1)
        td, pd_ = decimate_for_plot(t[m], p[m], 8000)
        ax.plot(td, pd_, color=UIBK_BLUE, lw=1.0, label="SC power")
        a, b = seg
        ax.fill_between(t[a:b], p[a:b], color=col, alpha=0.6, lw=0,
                        label="Max. discharge excursion" if val > 0 else "Max. charge excursion")
        ax.annotate(f"{lab} = {val:.1f} J", (t[(a + b) // 2], np.mean(p[a:b])), xytext=(0.62, 0.86),
                    textcoords="axes fraction", ha="center", va="top", arrowprops=dict(arrowstyle="->", lw=0.7),
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(w0, w1)
        ax.set_ylabel("SC power [W]")
        ax.tick_params(labelbottom=True)
        panel_legend(ax, ncol=2)
    axes[-1].set_xlabel("Time [s]")
    return fig


# ---------------------------------------------------------------------- trade-off (Fig. modules Cells)
GREY_OUT = "#d4d4d4"


def fig_tradeoff(res, scenarios=("profile", "profile_braking"), cmap: str = "jet", y_top: str = "cost_EUR",
                 y_bottom: str = "density_kg_dm3", size_top: str = "density_kg_dm3",
                 size_bottom: str = "cost_EUR", titles: Optional[dict] = None,
                 highlight: Optional[dict] = None, show_infeasible: bool = True) -> Figure:
    """IECON-style overview (Modules_Cells). Ball size in the foreground, colour = module ESR.

    Configurations that violate the specification (or the thermal / P limit) are drawn greyed out.
    With geometry 'pcb' the density is the module density (cells + PCB in the PCB box), else the cell density.
    """
    scen = [s for s in scenarios if s in res.candidates]
    titles = titles or {"profile": "Module 1", "profile_braking": "Module 2"}
    alls = [res.candidates[s] for s in scen]
    dfs = [d[d.feasible] for d in alls]
    bad = [d[~d.feasible] for d in alls]
    ok_esr = np.concatenate([d.ESR_module_mOhm.to_numpy() for d in dfs]) if any(len(d) for d in dfs) else \
        np.concatenate([d.ESR_module_mOhm.to_numpy() for d in alls])
    norm = Normalize(np.nanmin(ok_esr), np.nanmax(ok_esr))
    module_geo = getattr(res.inputs.opt, "geometry", "pcb") == "pcb"
    dens_lab = "Module density [kg/dm$^3$]" if module_geo else "Cell density [kg/dm$^3$]"
    units = {"cost_EUR": "[Euro]", "density_kg_dm3": "[kg/dm$^3$]", "mass_kg": "[kg]", "volume_dm3": "[dm$^3$]",
             "E_density_Wh_dm3": "[Wh/dm$^3$]"}
    ylab = {"cost_EUR": "Module cost [Euro]", "density_kg_dm3": dens_lab, "mass_kg": "Module mass [kg]",
            "volume_dm3": "Module volume [dm$^3$]", "E_density_Wh_dm3": "Module energy density [Wh/dm$^3$]"}

    def rng(col):
        src = alls if show_infeasible else dfs
        v = np.concatenate([d[col].to_numpy(float) for d in src])
        return np.nanmin(v), np.nanmax(v)

    fig = plt.figure(figsize=(8.5 * CM, 9.6 * CM))
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
        fmt = "{:.0f}" if smax >= 100 else "{:.2f}"
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
                         u_conv_min: Optional[float] = None) -> Figure:
    """Emulated supercapacitor profile over one periodic cycle: power, current, voltage, temperature."""
    t = trace.t - trace.t[0]
    m = np.ones_like(t, bool) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    has_dT = trace.dT is not None and np.any(trace.dT[m] != 0)
    n = 4 if has_dT else 3
    fig, axes = _stack(n, W2, 3.3)
    ax = axes[0]
    ax.plot(t[m], trace.P_load[m], color=GREY, lw=0.6, label="Load")
    ax.plot(t[m], trace.P_bat[m], color=UIBK_ORANGE, lw=LW, label="Battery")
    ax.plot(t[m], trace.P_sc[m], color=UIBK_BLUE, lw=0.8, label="Supercapacitor (bus side)")
    ax.set_ylabel("Power [W]")
    ax = axes[1]
    ax.plot(t[m], trace.I_sc[m], color=UIBK_BLUE, lw=0.8, label="SC current $I_\\mathrm{SC}$")
    ax.plot(t[m], trace.I_bus[m], color=UIBK_ORANGE, lw=0.7, label="Bus current $I_\\mathrm{bus}$")
    if i_limit:
        ax.axhline(i_limit, color="k", ls="--", lw=0.8, label="Current limit")
        ax.axhline(-i_limit, color="k", ls="--", lw=0.8)
    ax.set_ylabel("SC current [A]")
    ax = axes[2]
    ax.plot(t[m], trace.U_sc[m], color=UIBK_BLUE, label="Internal voltage $U_C$")
    ax.plot(t[m], trace.U_term[m], color=PINK, lw=0.6, label="Terminal voltage $U_C - R I$")
    ax.axhline(module.U_min, color="k", ls="--", lw=0.8, label="$U_\\mathrm{C0min}$, $U_\\mathrm{C0max}$")
    ax.axhline(module.U_max, color="k", ls="--", lw=0.8)
    if u_conv_min:
        ax.axhline(u_conv_min, color=MAGENTA, ls="-.", lw=0.9, label="Converter min. input voltage")
    if sc is not None:
        ax.axhline(sc.U_inM, color=DARK_GREY, ls=":", lw=0.9, label="$U_\\mathrm{C0inM}$")
    ax.set_ylabel("Voltage [V]")
    if has_dT:
        ax = axes[3]
        ax.plot(t[m], trace.dT[m], color=MAGENTA, label="Cell temperature rise")
        ax.set_ylabel("$\\Delta T$ [K]")
        ax.set_ylim(0, float(np.max(trace.dT[m])) * 1.15)
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


def fig_emudesign(iterations: list, dT_max: float, title_cells: str = "") -> Figure:
    """Closed-loop design iterations: energy swing / C_req and temperature over P."""
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
