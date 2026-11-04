"""Presentation helpers for the GUI: theme, figures in thesis quality, formatted tables."""
from __future__ import annotations

import io
import math
from typing import Optional

import pandas as pd
import streamlit as st

PX_PER_CM = 62          # display scale for thesis figures (8 cm -> about 500 px)

CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px;}
h1, h2, h3, h4 {color: #003361; font-weight: 650;}
[data-testid="stSidebar"] {background: #f3f5f9;}
[data-testid="stSidebar"] h1 {font-size: 1.25rem;}
.step-head {border-left: 6px solid #f39200; padding: 2px 0 2px 14px; margin: 0 0 0.6rem 0;}
.step-head .kicker {color: #f39200; font-weight: 700; font-size: 0.85rem; letter-spacing: .04em;}
.step-head .title {color: #003361; font-size: 1.65rem; font-weight: 700; line-height: 1.25;}
.step-head .sub {color: #4d5866; font-size: 0.95rem; margin-top: 2px;}
div[data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e2e7ef; border-radius: 10px;
  padding: 10px 14px;}
div[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #4d5866;}
div[data-testid="stMetricValue"] {font-size: 1.35rem; color: #003361;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 12px; font-size: .8rem; font-weight: 600;
  margin-right: 6px;}
.pill.ok {background:#dcefdc; color:#1b5e20;} .pill.warn {background:#fff3c4; color:#7a5b00;}
.pill.skip {background:#eceff3; color:#5f6b7a;} .pill.err {background:#f8d0d0; color:#8a1c1c;}
.pill.open {background:#e6eef8; color:#003361;}
.card-title {font-weight: 650; color:#003361; margin-bottom: .2rem;}
.small {font-size: .85rem; color: #4d5866;}
</style>
"""

STATUS_PILL = {"done": ("ok", "gerechnet"), "warning": ("warn", "Warnung"), "skipped": ("skip", "übersprungen"),
               "error": ("err", "Fehler"), "open": ("open", "offen")}


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def step_header(kicker: str, title: str, sub: str = "") -> None:
    st.markdown(f'<div class="step-head"><div class="kicker">{kicker}</div><div class="title">{title}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def pill(status: str) -> str:
    cls, txt = STATUS_PILL.get(status, ("open", status))
    return f'<span class="pill {cls}">{txt}</span>'


def show_fig(fig, key: str, width_cm: Optional[float] = None, container=None, plt_module=None) -> None:
    """Render a matplotlib figure at print resolution and offer PDF/SVG/PNG downloads."""
    c = container or st
    w_in = fig.get_size_inches()[0]
    width_cm = width_cm or w_in * 2.54
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=250)
    pdf = io.BytesIO()
    fig.savefig(pdf, format="pdf")
    svg = io.BytesIO()
    fig.savefig(svg, format="svg")
    c.image(png.getvalue(), width=int(width_cm * PX_PER_CM))
    with c.popover("Abbildung herunterladen"):
        st.download_button("PDF (Diss)", pdf.getvalue(), file_name=f"{key}.pdf", mime="application/pdf",
                           key=f"dl_pdf_{key}")
        st.download_button("SVG", svg.getvalue(), file_name=f"{key}.svg", mime="image/svg+xml", key=f"dl_svg_{key}")
        st.download_button("PNG (300 dpi)", png.getvalue(), file_name=f"{key}.png", mime="image/png",
                           key=f"dl_png_{key}")
    if plt_module is not None:
        plt_module.close(fig)


# column name, format, help
COLUMNS = {
    "cell_id": ("Zelle", None, None),
    "setup": ("P/S", None, "parallele Stränge / Zellen in Serie"),
    "P_C": ("P_C", "%d", "aus Kapazität"),
    "P_I": ("P_I", "%d", "aus Maximalstrom"),
    "P_T": ("P_T", "%d", "aus thermischem RMS-Strom"),
    "n_cells": ("Zellen", "%d", None),
    "C_cell_used_F": ("C Zelle [F]", "%.2f", "verwendete Zellkapazität (Datenblatt oder C_eq,E)"),
    "ESR_cell_mOhm": ("ESR Zelle [mΩ]", "%.1f", None),
    "C_module_F": ("C Modul [F]", "%.2f", None),
    "C_reserve": ("Reserve [%]", "percent", "C_Modul / C_req - 1"),
    "ESR_module_mOhm": ("ESR Modul [mΩ]", "%.0f", None),
    "I_module_max_A": ("I_max Modul [A]", "%.0f", None),
    "cost_EUR": ("Kosten [€]", "%.0f", None),
    "mass_kg": ("Masse [kg]", "%.2f", None),
    "volume_dm3": ("Volumen [dm³]", "%.2f", None),
    "density_kg_dm3": ("Moduldichte [kg/dm³]", "%.2f", "Modulmasse (Zellen + Leiterplatte) / Modulvolumen"),
    "E_density_Wh_dm3": ("Energie [Wh/dm³]", "%.2f", "Energiedichte des Moduls"),
    "pcb_mm": ("Platine [mm]", None, "Breite x Tiefe"),
    "layout": ("Layout", None, None),
    "E_density_Wh_kg": ("Energie [Wh/kg]", "%.2f", None),
    "P_loss_avg_W": ("Verluste [W]", "%.1f", "mittlere ESR-Verluste über das Profil"),
    "I_cell_rms_A": ("I Zelle RMS [A]", "%.2f", None),
    "R_th_KperW": ("R_th Zelle [K/W]", "%.1f", None),
    "k_coupling": ("k Packung", "%.2f", "Kopplungsfaktor Modulpackung"),
    "R_th_eff_KperW": ("R_th eff [K/W]", "%.1f", None),
    "dT_max_K": ("ΔT max [K]", "%.1f", None),
    "T_cell_max_C": ("T Zelle max [°C]", "%.1f", "T_amb + ΔT_max"),
    "thermal_status": ("Thermik", None, None),
    "score": ("Bewertung", "progress", "gewichtete Gesamtbewertung, 100 % = in allen Kriterien am besten"),
    "rank": ("Rang", "%d", None),
    "C_source": ("Kapazität aus", None, None),
    "ESR_source": ("ESR aus", None, None),
    "status": ("Status", None, None),
    "warnings": ("Hinweise", None, None),
}


def table(df: pd.DataFrame, cols: list, key: Optional[str] = None, selectable: bool = False, height=None):
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    if "C_reserve" in d:
        d["C_reserve"] = 100 * d["C_reserve"]
    if "cell_id" in d:
        d["cell_id"] = d["cell_id"].str.replace("VISHAY_HVR235_", "HVR235 ", regex=False)
    cfg = {}
    for c in cols:
        name, fmt, hlp = COLUMNS.get(c, (c, None, None))
        if fmt == "progress":
            cfg[c] = st.column_config.ProgressColumn(name, min_value=0, max_value=100, format="%.0f %%", help=hlp)
        elif fmt == "percent":
            cfg[c] = st.column_config.NumberColumn(name, format="%.0f %%", help=hlp)
        elif fmt:
            cfg[c] = st.column_config.NumberColumn(name, format=fmt, help=hlp)
        else:
            cfg[c] = st.column_config.TextColumn(name, help=hlp)
    kw = dict(hide_index=True, column_config=cfg, width="stretch", key=key)
    if height:
        kw["height"] = height
    if selectable:
        kw.update(on_select="rerun", selection_mode="single-row")
    return st.dataframe(d, **kw)


def short(cell_id: str) -> str:
    return cell_id.replace("VISHAY_HVR235_", "HVR235 ")


def fmt(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}{(' ' + unit) if unit else ''}"
