"""Thesis-style figures (matplotlib). Every function returns a Figure; save with save_fig()."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import equations as eq
from . import style
from .profile import decimate_for_plot, spectrum
from .style import COLORS, DARKGREEN, GREY, MAGENTA, UIBK_BLUE, UIBK_ORANGE, legend_below, sublabel

style.apply()


def save_fig(fig: Figure, path: str | Path, formats=("pdf", "svg", "png")) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = path.with_suffix("." + f)
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    return out


# ---------------------------------------------------------------------- profile
def fig_profile(res, t_window: Optional[tuple] = None) -> Figure:
    prof = res.profile
    fig, ax = plt.subplots(figsize=style.DOUBLE)
    t, p = prof.t, prof.p
    m = np.ones_like(t, bool) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    td, pd_ = decimate_for_plot(t[m], p[m])
    ax.plot(td, pd_, color=UIBK_BLUE, lw=0.6, label=r"$P_\mathrm{load}$")
    if res.p_bat is not None:
        tb, pb = decimate_for_plot(t[m], res.p_bat[m])
        ax.plot(tb, pb, color=UIBK_ORANGE, lw=1.2, label=r"$P_\mathrm{bat}$")
    ax.axhline(np.mean(p), color=DARKGREEN, ls="--", lw=1.0, label=r"$P_\mathrm{avg}$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Power $P$ [W]")
    legend_below(ax, ncol=3)
    return fig


def fig_peaks(res) -> Figure:
    pk = res.peaks
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8 * style.CM, 7 * style.CM), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 3]))
    a1.boxplot(pk["peaks"], vert=False, widths=0.6, flierprops=dict(marker="+", markeredgecolor=MAGENTA),
               medianprops=dict(color=MAGENTA), boxprops=dict(color=UIBK_BLUE), whis=1.5)
    a1.set_yticks([])
    for v in (pk["upper_fence"], pk["max_peak"]):
        a1.annotate(f"{v:.0f}", (v, 1.35), ha="center", fontsize=style.FONT_PT)
        a2.axvline(v, color="k", ls="--", lw=0.8)
    a2.hist(pk["peaks"], bins=15, color=LIGHT(), edgecolor="k", lw=0.4)
    a2.set_xlabel(r"Power $P$ [W]")
    a2.set_ylabel(r"Number of power peaks")
    return fig


def LIGHT():
    return "#5aa7d6"


def fig_fft(res, f_max: float = 100.0) -> Figure:
    f, a = spectrum(res.profile)
    m = (f > 0) & (f <= f_max)
    fig, ax = plt.subplots(figsize=style.SINGLE)
    ax.semilogx(f[m], a[m], color=UIBK_BLUE, lw=0.6)
    ax.set_xlabel(r"Frequency $f$ [Hz]")
    ax.set_ylabel(r"Amplitude $|P(f)|$ [W]")
    return fig


def fig_split_compare(t, p, splits: dict, t_window: Optional[tuple] = None) -> Figure:
    fig, ax = plt.subplots(figsize=style.DOUBLE)
    m = np.ones_like(t, bool) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    td, pd_ = decimate_for_plot(t[m], p[m])
    ax.plot(td, pd_, color=GREY, lw=0.5, label=r"$P_\mathrm{load}$")
    for i, (name, pb) in enumerate(splits.items()):
        tb, yb = decimate_for_plot(t[m], pb[m])
        ax.plot(tb, yb, lw=1.1, color=COLORS[i % len(COLORS)], label=name)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Battery power $P_\mathrm{bat}$ [W]")
    legend_below(ax, ncol=3)
    return fig


# ---------------------------------------------------------------------- voltages
def fig_discharge_ratio(res=None) -> Figure:
    """Energy ratio versus discharge ratio (Fig. energy_ratio)."""
    x = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=style.SINGLE)
    ax.plot(x, eq.energy_ratio(x), color=UIBK_BLUE)
    if res is not None and res.voltages:
        r = res.voltages["r_Q"]
        ax.plot(r, eq.energy_ratio(r), "o", color=MAGENTA, ms=4)
        ax.annotate(fr"$r_\mathrm{{Q}}$ = {r:.2f}: {100 * eq.energy_ratio(r):.0f} %", (r, eq.energy_ratio(r)),
                    xytext=(-50, -25), textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel(r"Discharge ratio $r_\mathrm{Q}$")
    ax.set_ylabel(r"Energy ratio $W_\mathrm{SC,used}/W_\mathrm{SC,total}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return fig


def fig_voltage_levels(res, scenario: str = "profile_braking", series_id: Optional[str] = None) -> Figure:
    """Definition of the module operating voltages with the computed values (Fig. SC_Voltages)."""
    sc = res.scenarios.get(scenario) or next(iter(res.scenarios.values()))
    sid = series_id or next(iter(res.series_info))
    U_N = res.series_info[sid]["U_C0N"]
    zones = [(0, sc.U_min, "Current/\nPower limit", "#e7a6e7"), (sc.U_min, sc.U_inM, "Discharging", "#8fd68f"),
             (sc.U_inM, sc.U_max, "Charging", "#8fd68f"), (sc.U_max, U_N, "Lifetime\nlimited", "#fff07a"),
             (U_N, U_N * 1.12, "Forbidden", "#f08080")]
    fig, ax = plt.subplots(figsize=(16 * style.CM, 2.6 * style.CM))
    for a, b, lab, col in zones:
        ax.add_patch(Rectangle((a, 0), b - a, 1, facecolor=col, edgecolor="k", lw=0.6))
        ax.text((a + b) / 2, 0.5, lab, ha="center", va="center", fontsize=style.FONT_PT)
    ticks = [0, sc.U_min, sc.U_inM, sc.U_max, U_N]
    labels = ["0", fr"$U_\mathrm{{C0min}}$" + f"\n{sc.U_min:.1f} V", fr"$U_\mathrm{{C0inM}}$" + f"\n{sc.U_inM:.1f} V",
              fr"$U_\mathrm{{C0max}}$" + f"\n{sc.U_max:.1f} V", fr"$U_\mathrm{{C0N}}$" + f"\n{U_N:.1f} V"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.set_xlim(0, U_N * 1.12)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.set_xlabel(r"Module voltage $u_\mathrm{C0}$ [V]")
    return fig


def fig_lifetime(res, db, t_range=(1e3, 5e4)) -> Figure:
    """Permissible cell voltage versus required lifetime (Fig. voltagelife)."""
    req = res.inputs.req
    t = np.logspace(math.log10(t_range[0]), math.log10(t_range[1]), 400)
    fig, ax = plt.subplots(figsize=style.SINGLE)
    shown = set(res.series_info)
    life = db.lifetime
    rows = life[life.series_id.isin(shown) | (life["default"].astype(str).str.lower() == "true")]
    for i, (_, s) in enumerate(rows.iterrows()):
        u = s.u0_V - (np.log2(t / s.t0_h) - (s.T0_C - req.T_cell_C) / s.dT_K) * s.dU_V
        lab = fr"{s.series}, $T_0$ = {s.T0_C:g} °C, $U_\mathrm{{C0N1}}$ = {s.u0_V:g} V"
        ax.semilogx(t, np.minimum(u, s.u0_V), color=COLORS[i % len(COLORS)], label=lab)
    for sid, si in res.series_info.items():
        ax.plot(req.t_life_h, si["U_C01max"], "o", color=MAGENTA, ms=3.5)
        ax.annotate(f"{si['U_C01max']:.3f} V", (req.t_life_h, si["U_C01max"]), xytext=(-40, 25),
                    textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel(r"Estimated lifetime $t_\mathrm{est}$ [h]")
    ax.set_ylabel(r"Cell voltage $U_\mathrm{C01max}$ [V]")
    ax.set_title(fr"$T$ = {req.T_cell_C:g} °C", fontsize=style.FONT_PT)
    legend_below(ax, ncol=1, bbox_y=-0.28)
    return fig


def fig_energy_excursions(res, windows: str | tuple = "auto", half_width: float = 10.0) -> Figure:
    """Max. discharge and charge excursions of P_SC (Fig. energydata). windows='iecon' -> [0,20], [110,130]."""
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
    fig, axes = plt.subplots(2, 1, figsize=(8 * style.CM, 7 * style.CM))
    for ax, (w0, w1), seg, val, col, lab in zip(
            axes, wins, (ex.dch_segment, ex.ch_segment), (ex.E_dch_J, -ex.E_ch_J), (UIBK_ORANGE, MAGENTA),
            (r"$E_\mathrm{DCH,max}$", r"$E_\mathrm{CH,max}$")):
        m = (t >= w0) & (t <= w1)
        ax.plot(t[m], p[m], color=UIBK_BLUE, lw=0.8)
        a, b = seg
        ax.fill_between(t[a:b], p[a:b], color=col, alpha=0.55, lw=0)
        xm = t[(a + b) // 2]
        ax.annotate(f"{lab} = {val:.1f} J", (xm, np.mean(p[a:b])), xytext=(0.5, 0.88),
                    textcoords="axes fraction", ha="center", va="top",
                    arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=style.FONT_PT,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(w0, w1)
        ax.set_ylabel(r"Power $P_\mathrm{SC}$ [W]")
    axes[-1].set_xlabel(r"Time $t$ [s]")
    sublabel(axes[0], "(a)", -0.18)
    sublabel(axes[1], "(b)", -0.38)
    return fig


# ---------------------------------------------------------------------- trade-off (Fig. modules Cells)
def fig_tradeoff(res, scenarios=("profile", "profile_braking"), cmap: str = "jet",
                 y_top: str = "cost_EUR", y_bottom: str = "density_kg_dm3",
                 size_top: str = "density_kg_dm3", size_bottom: str = "cost_EUR",
                 titles: Optional[dict] = None) -> Figure:
    """Fig. 'modules Cells': cost and density versus cell capacitance, colour = module ESR.

    Marker size of the top row scales with density, of the bottom row with cost (legends on the right).
    """
    scen = [s for s in scenarios if s in res.candidates]
    titles = titles or {"profile": "Module 1", "profile_braking": "Module 2"}
    dfs = [res.candidates[s][res.candidates[s].feasible] for s in scen]
    allv = np.concatenate([d.ESR_module_mOhm.to_numpy() for d in dfs])
    norm = Normalize(np.nanmin(allv), np.nanmax(allv))
    units = {"cost_EUR": "[Euro]", "density_kg_dm3": "[kg/dm$^3$]", "mass_kg": "[kg]", "volume_dm3": "[dm$^3$]",
             "ESR_module_mOhm": r"[m$\Omega$]"}
    ylab = {"cost_EUR": "Module\nCost [Euro]", "density_kg_dm3": "Cell\nDensity [kg/dm$^3$]",
            "mass_kg": "Module\nMass [kg]", "volume_dm3": "Module\nVolume [dm$^3$]"}

    def rng(col):
        v = np.concatenate([d[col].to_numpy(float) for d in dfs])
        return np.nanmin(v), np.nanmax(v)

    fig = plt.figure(figsize=(8.5 * style.CM, 9.0 * style.CM))
    gs = fig.add_gridspec(3, len(scen) + 1, height_ratios=[1, 1, 0.07], width_ratios=[1] * len(scen) + [0.3])
    handle = None
    for i, (ycol, scol) in enumerate(((y_top, size_top), (y_bottom, size_bottom))):
        smin, smax = rng(scol)
        for j, (s, d) in enumerate(zip(scen, dfs)):
            ax = fig.add_subplot(gs[i, j])
            v = d[scol].to_numpy(float)
            size = 25 + (v - smin) / (smax - smin) * 75 if smax > smin else np.full_like(v, 50)
            handle = ax.scatter(d.c_nom_F, d[ycol], s=size, c=d.ESR_module_mOhm, cmap=cmap, norm=norm,
                                edgecolors="none")
            sel = (res.inputs.selected or {}).get(s)
            if sel:
                r = d[d.cell_id == sel.get("cell_id")]
                if not r.empty:
                    ax.scatter(r.c_nom_F, r[ycol], s=130, facecolors="none", edgecolors="k", lw=0.8)
            if i == 0:
                ax.set_title(titles.get(s, s), fontsize=style.FONT_PT)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Cell capacitance [F]")
            if j == 0:
                ax.set_ylabel(ylab.get(ycol, ycol))
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlim(0, 105)
        lax = fig.add_subplot(gs[i, -1])
        lax.axis("off")
        lax.scatter([0.15, 0.15, 0.15], [0.1, 0.5, 0.9], s=[25, 62, 100], c=["k", "none", "k"],
                    edgecolors="k", lw=0.5)
        fmt = "{:.0f}" if smax >= 100 else "{:.2f}"
        lax.text(0.62, 0.9, fmt.format(smax), va="center", fontsize=style.FONT_PT)
        lax.text(0.62, 0.1, fmt.format(smin), va="center", fontsize=style.FONT_PT)
        lax.text(1.35, 0.5, units.get(scol, scol), rotation=90, va="center", ha="center", fontsize=style.FONT_PT)
        lax.set_xlim(0, 1.6)
        lax.set_ylim(-0.1, 1.1)
    cax = fig.add_subplot(gs[2, : len(scen)])
    cb = fig.colorbar(handle, cax=cax, orientation="horizontal")
    cb.set_label(r"Module ESR [m$\Omega$]")
    return fig


# ---------------------------------------------------------------------- C(U)
def fig_cu(fit: dict, cell: dict, u_lo: float, u_hi: float, c_eq: float) -> Figure:
    u = np.linspace(0, cell["u_rated_V"], 200)
    fig, ax = plt.subplots(figsize=style.SINGLE)
    ax.plot(u, fit["C0_F"] + fit["k_FperV"] * u, color=UIBK_BLUE, label=r"$C(U)=C_0+kU$")
    ax.axhline(cell["c_nom_F"], color=GREY, ls="--", label=r"$C_\mathrm{cell}$")
    ax.hlines(c_eq, u_lo, u_hi, color=MAGENTA, lw=1.6, label=r"$C_\mathrm{eq,E}$")
    ax.axvspan(u_lo, u_hi, color=UIBK_ORANGE, alpha=0.15, lw=0)
    ax.set_xlabel(r"Cell voltage $U$ [V]")
    ax.set_ylabel(r"Capacitance $C$ [F]")
    legend_below(ax, ncol=3)
    return fig


# ---------------------------------------------------------------------- thermal
def fig_thermal(det, sc, dT_max: float, title: str = "") -> Figure:
    tr = det.current
    t = tr.t
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8 * style.CM, 7 * style.CM), sharex=True)
    a1.plot(t, np.sqrt(tr.I_sq), color=UIBK_BLUE, lw=0.6, label=r"$I_\mathrm{module,RMS,bin}$")
    a1b = a1.twinx()
    a1b.plot(t, tr.U, color=UIBK_ORANGE, lw=1.0, label=r"$U_\mathrm{C0}$")
    a1b.set_ylabel(r"Voltage $U_\mathrm{C0}$ [V]")
    a1b.grid(False)
    a1.set_ylabel(r"Current $I$ [A]")
    a2.plot(t, det.dT, color=MAGENTA, label=r"$\Delta T$")
    a2.axhline(dT_max, color="k", ls="--", lw=0.8, label=r"$\Delta T_\mathrm{max}$")
    a2.set_ylabel(r"Temperature rise $\Delta T$ [K]")
    a2.set_xlabel(r"Time $t$ [s]")
    if title:
        a1.set_title(title, fontsize=style.FONT_PT)
    legend_below(a2, ncol=2, bbox_y=-0.35)
    return fig


# ---------------------------------------------------------------------- emulation
def fig_emulation(trace, history, module=None) -> Figure:
    fig, axes = plt.subplots(4, 1, figsize=(16 * style.CM, 14 * style.CM))
    t = trace.t
    a = axes[0]
    a.plot(t, trace.P_load, color=GREY, lw=0.5, label=r"$P_\mathrm{load}$")
    a.plot(t, trace.P_bat, color=UIBK_ORANGE, lw=0.8, label=r"$P_\mathrm{bat}$")
    a.plot(t, trace.P_sc, color=UIBK_BLUE, lw=0.6, label=r"$P_\mathrm{SC}$")
    a.set_ylabel(r"Power $P$ [W]")
    legend_below(a, ncol=3, bbox_y=1.25)
    a = axes[1]
    a.plot(t, trace.U_sc, color=UIBK_BLUE, label=r"$U_\mathrm{C0}$")
    if module is not None:
        a.axhline(module.U_min, color=MAGENTA, ls="--", lw=0.8)
        a.axhline(module.U_max, color=MAGENTA, ls="--", lw=0.8)
    a.set_ylabel(r"Voltage $U_\mathrm{C0}$ [V]")
    a = axes[2]
    a.plot(t, trace.dT, color=MAGENTA)
    a.set_ylabel(r"$\Delta T$ [K]")
    a.set_xlabel(r"Time $t$ [s]")
    a = axes[3]
    cyc = [h.cycle for h in history]
    a.plot(cyc, [h.U_end for h in history], "o-", ms=2.5, color=UIBK_BLUE, label=r"$U_\mathrm{C0,end}$")
    a2 = a.twinx()
    a2.plot(cyc, [h.dT_max for h in history], "s-", ms=2.5, color=MAGENTA, label=r"$\Delta T_\mathrm{max}$")
    a2.grid(False)
    a.set_xlabel("Cycle $n$")
    a.set_ylabel(r"$U_\mathrm{C0,end}$ [V]")
    a.ticklabel_format(axis="y", useOffset=False)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a2.set_ylabel(r"$\Delta T_\mathrm{max}$ [K]")
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(1.0, 1.02, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=style.FONT_PT)
    return fig
