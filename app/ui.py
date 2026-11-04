"""Presentation helpers for the GUI: theme, figures in thesis quality, formatted tables."""
from __future__ import annotations

import io
import math
from typing import Optional

import pandas as pd
import streamlit as st

from sc_module_design import formulas as FX

PX_PER_CM = 62          # display scale for thesis figures (8 cm -> about 500 px)

CSS = """
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1450px;}
[data-testid="stSidebar"] {min-width: 360px;}
div[data-testid="stButtonGroup"] button p {font-size: .82rem;}
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
div[data-testid="stAlertContentWarning"], div[data-testid="stAlertContentError"] {font-weight: 600;}
.src {font-size: .8rem; color: #5f6b7a; font-style: italic;}
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


def fx_text(key: str, calc: str = "") -> str:
    return FX.text(key, calc)


def fx(keys, calcs: Optional[dict] = None, container=None, label: str = "ƒ Formeln und Herkunft") -> None:
    """Pop-up with the formulas used (LaTeX), the inserted values and the origin of each relation."""
    c = container or st
    calcs = calcs or {}
    with c.popover(label):
        for k in keys:
            f = FX.F.get(k)
            if f is None:
                continue
            st.markdown(f"**{f.title}**")
            st.latex(f.latex)
            if calcs.get(k):
                st.markdown(f"Eingesetzt: {calcs[k]}")
            st.markdown(f"<div class='src'>Herkunft: {f.origin}{(', ' + f.source) if f.source else ''}</div>",
                        unsafe_allow_html=True)
            st.divider()


def metric(container, label: str, value, delta=None, key: Optional[str] = None, calc: str = "",
           delta_color: str = "off") -> None:
    """st.metric with the formula, inserted values and origin as tooltip."""
    container.metric(label, value, delta, delta_color=delta_color, help=FX.text(key, calc) if key else None,
                     delta_arrow="off" if delta_color == "off" else "auto")


def warn(msg: str, container=None) -> None:
    """Warnings are red and hard to overlook."""
    (container or st).error(msg, icon=":material/warning:")


def alerts(messages: list, container=None, title: str = "Hinweise") -> None:
    msgs = list(dict.fromkeys(m for m in messages if m))
    if not msgs:
        return
    c = container or st
    if len(msgs) <= 3:
        c.error("\n".join(f"- {m}" for m in msgs), icon=":material/warning:")
    else:
        c.error("\n".join(f"- {m}" for m in msgs[:3]) + f"\n- ... und {len(msgs) - 3} weitere (aufklappen)",
                icon=":material/warning:")
        with c.expander(f"Alle {len(msgs)} {title}"):
            for m in msgs:
                st.markdown(f"- {m}")


def show_fig(fig, key: str, width_cm: Optional[float] = None, container=None, plt_module=None,
             fx_keys: Optional[list] = None, calcs: Optional[dict] = None) -> None:
    """Render a matplotlib figure at print resolution and offer PDF/SVG/PNG downloads (+ formulas)."""
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
    cc = c.columns([1, 1, 2]) if fx_keys else [c]
    if fx_keys:
        fx(fx_keys, calcs, container=cc[1], label="ƒ Formeln")
    with cc[0].popover("Download"):
        st.download_button("PDF (Diss)", pdf.getvalue(), file_name=f"{key}.pdf", mime="application/pdf",
                           key=f"dl_pdf_{key}")
        st.download_button("SVG", svg.getvalue(), file_name=f"{key}.svg", mime="image/svg+xml", key=f"dl_svg_{key}")
        st.download_button("PNG (300 dpi)", png.getvalue(), file_name=f"{key}.png", mime="image/png",
                           key=f"dl_png_{key}")
    if plt_module is not None:
        plt_module.close(fig)


# column name, format, help (formula and origin)
COLUMNS = {
    "cell_id": ("Zelle", None, None),
    "setup": ("P/S", None, "P parallele Stränge / S Zellen in Serie"),
    "P_C": ("P_C", "%d", "P_C = ceil(S C_req / C_Zelle) (Diss), C_req inklusive ESR-Verluste beim Entladen"),
    "P_I": ("P_I", "%d", "P_I = ceil(I_C0max / I_Zelle,max) (Diss)"),
    "P_T": ("P_T", "%d", "P_T = ceil(I_Modul,RMS / I_Zelle,RMS,max), I_Zelle,RMS,max = sqrt(ΔT_max/(ESR R_th,eff)) "
                         "(Diss A01)"),
    "P_driver": ("P bestimmt durch", None, "Welcher Term P = max(P_C, P_I, P_T, ...) festlegt"),
    "P_C_no_loss": ("P_C ohne Verluste", "%d", "P_C ohne ESR-Verluste beim Entladen"),
    "E_loss_dch_J": ("Verluste Entladen [J]", "%.0f", "E_V,DCH = (Σ I² R / Σ P) über Entladeabschnitte x E_DCH,max, "
                                                      "Beharrungszustand bei U_C0inM"),
    "C_req_loss_F": ("C_req inkl. Verluste [F]", "%.3f", "2 (W_used + E_V,DCH) / (L_EOL (U_C0max² - U_C0min²))"),
    "I_peak_Umin_A": ("I Spitze bei U_C0min [A]", "%.1f", "inkl. ESR-Abfall: (U - sqrt(U² - 4RP))/(2R); leer = "
                                                         "Leistungsgrenze U²/(4R) überschritten"),
    "pcb_area_cm2": ("Platinenfläche [cm²]", "%.0f", "Breite x Tiefe der Leiterplatte"),
    "n_cells": ("Zellen", "%d", "n = S P"),
    "C_cell_used_F": ("C Zelle [F]", "%.2f", "Datenblatt oder C_eq,E = C0 + (2k/3)(u2³-u1³)/(u2²-u1²) (Diss A02)"),
    "ESR_cell_mOhm": ("ESR Zelle [mΩ]", "%.1f", "Datenblatt oder eigene Messung (M4, M3, M2, M1)"),
    "C_module_F": ("C Modul [F]", "%.2f", "C_Modul = P/S C_Zelle"),
    "C_reserve": ("Reserve EOL [%]", "percent",
                  "Reserve am Lebensende = C_Modul / C_req - 1 (C_req inkl. Verluste). Sehr große Werte, "
                  "wenn P durch Strom oder Thermik bestimmt ist (Spalte P bestimmt durch)."),
    "C_reserve_BOL": ("Reserve neu [%]", "percent", "Notiz: Neuzustand C_Modul / C_benötigt - 1"),
    "ESR_module_mOhm": ("ESR Modul [mΩ]", "%.0f", "R_Modul = S/P ESR_Zelle"),
    "I_module_max_A": ("I_max Modul [A]", "%.0f", "P I_Zelle,max"),
    "cost_EUR": ("Kosten [€]", "%.0f", "S P Zellpreis"),
    "mass_kg": ("Masse [kg]", "%.2f", "S P m_Zelle + Leiterplatte (0,35 g/cm², Annahme)"),
    "volume_dm3": ("Modulvolumen [dm³]", "%.2f", "Platine B x T x (L_Zelle + 1,6 mm + Abstand), Packung hexagonal "
                                                 "(Standard)"),
    "density_kg_dm3": ("Moduldichte [kg/dm³]", "%.2f", "Modulmasse (Zellen + Leiterplatte) / Modulvolumen"),
    "E_density_Wh_dm3": ("Energie [Wh/dm³]", "%.2f", "Energiedichte des Moduls"),
    "pcb_mm": ("Platine [mm]", None, "Breite x Tiefe"),
    "layout": ("Layout", None, None),
    "E_density_Wh_kg": ("Energie [Wh/kg]", "%.2f", None),
    "P_loss_avg_W": ("Verluste [W]", "%.1f", "P_v = I_Modul,RMS² R_Modul über das Profil"),
    "I_cell_rms_A": ("I Zelle RMS [A]", "%.2f", "I_Modul,RMS / P"),
    "R_th_KperW": ("R_th Zelle [K/W]", "%.1f", "gemessen (50 F, Andy) oder skaliert (R_th ~ 1/Oberfläche, Annahme)"),
    "k_coupling": ("k Packung", "%.2f", "k = n A_Zelle / A_Box (Lumped, Annahme Claude; Incropera Kap. 5)"),
    "R_th_eff_KperW": ("R_th eff [K/W]", "%.1f", "R_th,eff = k R_th"),
    "dT_max_K": ("ΔT max Lumped [K]", "%.1f", "C_th dΔT/dt = P_v - ΔT/R_th,eff, Maximum im periodischen Zustand"),
    "T_cell_max_C": ("T Zelle max [°C]", "%.1f", "T_amb + ΔT_max"),
    "dT_net_max_K": ("ΔT Netzwerk heißeste [K]", "%.1f", "Netzwerkmodell (Spezifikation Andy), stationär, Precheck"),
    "dT_net_centre_K": ("ΔT Netzwerk Mitte [K]", "%.1f", "Netzwerkmodell, Zelle in der Mitte"),
    "dT_net_outer_K": ("ΔT Netzwerk außen [K]", "%.1f", "Netzwerkmodell, Mittel der Randzellen"),
    "thermal_status": ("Thermik", None, None),
    "score": ("Bewertung", "progress", "gewichtete Gesamtbewertung, 100 % = in allen Kriterien am besten"),
    "rank": ("Rang", "%d", None),
    "C_source": ("Kapazität aus", None, None),
    "ESR_source": ("ESR aus", None, None),
    "status": ("Status", None, None),
    "warnings": ("Hinweise", None, None),
}


def table(df: pd.DataFrame, cols: list, key: Optional[str] = None, selectable: bool = False, height=None,
          fx_keys: Optional[list] = None):
    ev = _table(df, cols, key, selectable, height)
    if "warnings" in df.columns:
        w = df[df.warnings.astype(str).str.len() > 0]
        if len(w):
            warn(f"{len(w)} von {len(df)} Zeilen mit Hinweisen (Spalte Hinweise). Details aufklappen.")
            with st.expander("Hinweise je Zelle"):
                for _, r in w.iterrows():
                    st.markdown(f"**{short(str(r.cell_id))} {r.get('setup', '')}**: {r.warnings}")
    if fx_keys:
        fx(fx_keys, label="ƒ Formeln der Spalten")
    return ev


def _table(df: pd.DataFrame, cols: list, key: Optional[str] = None, selectable: bool = False, height=None):
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    for c in ("C_reserve", "C_reserve_BOL"):
        if c in d:
            d[c] = 100 * d[c]
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
