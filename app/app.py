"""SC Module Designer, Streamlit GUI (guided, one page per step of the flowchart).

Start:  ./run_gui.sh   (or: streamlit run app/app.py)
"""
from __future__ import annotations

import copy
import io
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import flownav  # noqa: E402
import ui  # noqa: E402
import widgets as W  # noqa: E402
from sc_module_design import equations as eq  # noqa: E402
from sc_module_design import plotting as pl  # noqa: E402
from sc_module_design.config import DesignInputs, list_presets, load_preset  # noqa: E402
from sc_module_design.database import TABLES, get_database  # noqa: E402
from sc_module_design.emudesign import emulation_design  # noqa: E402
from sc_module_design.emulation import EmulationConfig, HESSEmulator, ModuleModel  # noqa: E402
from sc_module_design.energy import METHODS as ENERGY_METHODS, energy_excursions  # noqa: E402
from sc_module_design.paths import DB_DIR, PRESET_DIR, PRIVATE_PROFILE_DIR, PROJECT_ROOT, resolve  # noqa: E402
from sc_module_design.pipeline import SCENARIOS, prepare_profile, run_design  # noqa: E402
from sc_module_design.profile import decimate_for_plot, list_profiles  # noqa: E402
from sc_module_design.report import compare_with_reference, export_run, latex_results_table, summary_dict  # noqa: E402
from sc_module_design.layout import GROUP_SHAPES, ORIENTATIONS, PACKINGS  # noqa: E402
from sc_module_design.selection import (C_SOURCES, ESR_SOURCES, GEOMETRY_MODES, THERMAL_MODES,  # noqa: E402
                                        coupling_factor, evaluate_candidate, layout_for, layout_pair,
                                        max_P_in_space, network_params)
from sc_module_design import thermal_network as TN  # noqa: E402
from sc_module_design.splits import REGISTRY, apply_split, get_split, load_plugins  # noqa: E402
from sc_module_design.thermal import COUPLING_MODES, CURRENT_MODES, PROFILE_MODES, block_layout  # noqa: E402
from sc_module_design.tradeoff import CRITERIA, WEIGHT_PRESETS, preset_weights, score  # noqa: E402

st.set_page_config(page_title="SC Module Designer", layout="wide", initial_sidebar_state="expanded")
ui.apply_theme()
load_plugins()
db = get_database()
COLORS = ["#003361", "#f39200", "#c2003e", "#00665f", "#420066", "#18B6EE"]
PLOTLY_FONT = dict(family="Times New Roman, Times, serif", size=13)

# ============================================================================ state
if "inp" not in st.session_state:
    names = [p.stem for p in list_presets()]
    st.session_state.inp = load_preset("thesis_characterized" if "thesis_characterized" in names else names[0])
    st.session_state.ver = 0
inp: DesignInputs = st.session_state.inp

PAGES = [
    ("start", "Start und Überblick", []),
    ("profil", "1  Lastprofil und Bauraum", ["0"]),
    ("split", "2  Leistungsaufteilung", ["0"]),
    ("spannung", "3  Spannungen und Wandlerstrom", ["1", "2", "3"]),
    ("kapazitaet", "4  Kapazität", ["4"]),
    ("zellen", "5  Zellauswahl und Layout", ["5-11"]),
    ("thermik", "6  Thermik", ["th"]),
    ("tradeoff", "7  Bewertung und Auswahl", ["12"]),
    ("balancing", "Balancing", ["13"]),
    ("emulation", "8  Emulation (Prüfung)", ["14"]),
    ("detail", "Zelldetail", []),
    ("db", "Datenbank", ["db"]),
    ("export", "Export", []),
]
SHORT = {"start": "Start", "profil": "1 Profil", "split": "2 Aufteilung", "spannung": "3 Spannung",
         "kapazitaet": "4 Kapazität", "zellen": "5 Zellen", "thermik": "6 Thermik", "tradeoff": "7 Bewertung",
         "balancing": "Balancing", "emulation": "8 Emulation", "detail": "Detail", "db": "Datenbank",
         "export": "Export"}
PAGE_KEYS = [p[0] for p in PAGES]


def db_stamp() -> float:
    return max((DB_DIR / f).stat().st_mtime for f in TABLES.values() if (DB_DIR / f).exists())


@st.cache_resource(max_entries=4, show_spinner="Lade Lastprofil ...")
def cached_profile(spec_json: str, mtime: float):
    d = DesignInputs()
    d.profile.__dict__.update(json.loads(spec_json))
    return prepare_profile(d)


def profile_key() -> tuple:
    if not inp.profile.path:
        return ("", 0.0)
    p = resolve(inp.profile.path)
    return (json.dumps(asdict(inp.profile), sort_keys=True), p.stat().st_mtime if p.exists() else 0.0)


@st.cache_resource(max_entries=12, show_spinner="Teile Leistung auf ...")
def cached_split(pkey: tuple, name: str, params_json: str):
    prof = cached_profile(*pkey)
    return apply_split(name, prof.t, prof.p, json.loads(params_json))


@st.cache_resource(max_entries=8, show_spinner="Rechne Auslegung ...")
def cached_design(inp_yaml: str, stamp: float, pkey: tuple):
    d = DesignInputs.from_dict(yaml.safe_load(inp_yaml))
    prof = cached_profile(*pkey) if pkey[0] and pkey[1] else None
    split = cached_split(pkey, d.split.name, json.dumps(d.split.params, sort_keys=True)) if prof is not None else None
    return run_design(d, db=get_database(), profile=prof, split=split, load_profile_file=False)


def compute():
    pkey = profile_key()
    if pkey[0] and not pkey[1]:
        pkey = ("", 0.0)
    design_inp = inp.copy()
    design_inp.selected = {}          # selection does not change the computation
    r = cached_design(yaml.safe_dump(design_inp.to_dict(), sort_keys=True), db_stamp(), pkey)
    r.inputs = inp                    # current selection for figures and tables
    st.session_state["last_status"] = flow_status(r)
    return r


def flow_status(r) -> dict:
    """Status per flowchart box (see flownav.BOXES)."""
    out = {k: s.status for k, s in r.steps.items()}
    if not inp.req.I_C0max_confirmed and "2" in out:
        out["2"] = "error"                     # converter current must be confirmed
    th = [df for df in r.candidates.values()]
    if th:
        done = any((~df.thermal_status.str.startswith("übersprungen")).any() for df in th)
        net_hot = any((df.get("dT_net_max_K", pd.Series(dtype=float)) > inp.req.dT_max_K).any() for df in th)
        out["th"] = "warning" if (done and net_hot) else ("done" if done else "skipped")
    out["db"] = "warning" if r.skipped_cells else "done"
    out["fin"] = "done" if inp.selected else "open"
    out.setdefault("13", "skipped")
    return out


def page_status(steps: list) -> str:
    stt = st.session_state.get("last_status", {})
    if not steps:
        return ""
    vals = [stt.get(s) for s in steps if stt.get(s)]
    if not vals:
        return ""
    if "error" in vals:
        return "✕ "
    if "warning" in vals:
        return "! "
    if all(v == "skipped" for v in vals):
        return "· "
    return "✓ "


def go_to(key: str):
    st.session_state["nav"] = key


def nav_buttons(cur: str):
    i = PAGE_KEYS.index(cur)
    st.divider()
    c1, _, c3 = st.columns([1, 3, 1])
    if i > 0:
        c1.button(f"← {PAGES[i - 1][1].strip()}", on_click=go_to, args=(PAGE_KEYS[i - 1],), width="stretch")
    if i < len(PAGES) - 1:
        c3.button(f"{PAGES[i + 1][1].strip()} →", on_click=go_to, args=(PAGE_KEYS[i + 1],), type="primary",
                  width="stretch")


def selected_cell(scenario: str) -> str | None:
    return (inp.selected or {}).get(scenario, {}).get("cell_id")


def chosen_cell(res, scenario: str, ids: list) -> tuple:
    """Cell to show by default: selection of this module, else of the other module, else the recommendation."""
    cid = selected_cell(scenario)
    if cid in ids:
        return cid, "gewählt in Schritt 7"
    for sk_ in (inp.selected or {}):
        other = selected_cell(sk_)
        if other in ids:
            return other, f"gewählt für {SCENARIOS.get(sk_, sk_)}"
    df_ = res.candidates.get(scenario)
    if df_ is not None:
        f_ = df_[df_.feasible & df_.cell_id.isin(ids)]
        if len(f_) and "score" in f_:
            return f_.sort_values("score", ascending=False).iloc[0].cell_id, "Empfehlung (keine Auswahl)"
    return (ids[0] if ids else None), "erste Zelle (keine Auswahl)"


def _select_cell(sk: str, key: str, all_modules_key: str):
    cid = st.session_state.get(key)
    if not cid:
        return
    targets = list(inp.opt.scenarios) if st.session_state.get(all_modules_key, True) else [sk]
    for t in targets:
        prev = inp.selected.get(t) or {}
        inp.selected[t] = {"cell_id": cid} if prev.get("cell_id") != cid else prev


def plotly_layout(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), font=PLOTLY_FONT,
                      legend=dict(orientation="h", y=-0.18), plot_bgcolor="white")
    fig.update_xaxes(showline=True, linecolor="#999", gridcolor="#e3e3e3", ticks="inside")
    fig.update_yaxes(showline=True, linecolor="#999", gridcolor="#e3e3e3", ticks="inside")
    return fig


# ============================================================================ sidebar
with st.sidebar:
    st.title("SC Module Designer")
    st.caption("Supercap-Modulauslegung nach Flussdiagramm der Dissertation")
    presets = [p.stem for p in list_presets()]
    psel = st.selectbox("Preset", presets, index=presets.index(inp.name) if inp.name in presets else 0,
                        help="iecon2023_legacy: exakt wie IECON 2023. thesis_datasheet: Datenblatt, korrigiert. "
                             "thesis_characterized: gemessene C0/k, ESR, Thermik.")
    if st.button("Preset laden", width="stretch", type="primary"):
        st.session_state.inp = load_preset(psel)
        st.session_state.ver = st.session_state.get("ver", 0) + 1
        for k in list(st.session_state.keys()):
            if k.startswith(("emu_", "live_", "detail_")):
                st.session_state.pop(k, None)
        st.rerun()
    if "nav" not in st.session_state:
        st.session_state["nav"] = "start"
    st.markdown("**Flussdiagramm** (Kasten anklicken)")
    flow_slot = st.empty()
    missing_slot = st.empty()
    with st.expander("Design speichern"):
        inp.name = st.text_input("Name", inp.name, key=W.key("name"))
        if st.button("Als Preset speichern", width="stretch"):
            p = inp.to_yaml(PRESET_DIR / f"{inp.name}.yaml")
            st.success(f"Gespeichert: {p.relative_to(PROJECT_ROOT)}")
        st.download_button("YAML herunterladen", yaml.safe_dump(inp.to_dict(), sort_keys=False, allow_unicode=True),
                           file_name=f"{inp.name}.yaml", width="stretch")
    if st.button("Cache leeren und neu rechnen", width="stretch"):
        cached_design.clear()
        st.rerun()


# ============================================================================ pages
def page_start():
    ui.step_header("ÜBERBLICK", "Supercap-Modul für das hybride Speichersystem",
                   "Kasten im Flussdiagramm anklicken (hier oder links) oder die Schritte oben verwenden. Jede "
                   "Eingabe rechnet sofort neu, fehlende Daten werden übersprungen und rot markiert.")
    with st.container(border=True):
        c = st.columns([1.2, 2])
        W.check("Thermisches Netzwerkmodell (Precheck) aktiv", inp.opt, "thermal_network", "net_on", container=c[0],
                help="Eigener Ansatz (nicht validiert): jede Zelle eigene Wärmequelle, Nachbarkopplung, Gradient innen "
                     "nach außen. Aus = nur Einzelzell- und Blockmodell (Lumped).")
        c[1].caption("C Netzwerkmodell: eigener Ansatz, nicht validiert, FEM empfohlen (Precheck). A Einzelzelle: "
                     "Diss Anhang A01, gemessen. B Blockmodell: Lehrbuch (Incropera et al., Kap. 5), für SC-Module "
                     "nicht validiert, FEM empfohlen.")
    res = compute()
    c1, c2 = st.columns([1.05, 1])
    with c1:
        clicked = flownav.flowchart(st.session_state.get("last_status", {}), "start", key="flownav_start", width=520)
        st.markdown(flownav.legend_html(), unsafe_allow_html=True)
        if clicked and clicked != "start":
            st.session_state["nav"] = clicked
            st.rerun()
    with c2:
        st.markdown(f"**Preset:** `{inp.name}`")
        if inp.description:
            st.caption(inp.description)
        for k, s in res.steps.items():
            st.markdown(f"{ui.pill(s.status)} **{k} {s.title}**<br><span class='small'>{s.summary}</span>",
                        unsafe_allow_html=True)
    if res.scenarios:
        st.markdown("#### Gewähltes Modul")
        cols = st.columns(len(res.scenarios))
        for col, (sk, sc) in zip(cols, res.scenarios.items()):
            with col.container(border=True):
                st.markdown(f"<div class='card-title'>{sc.label}</div>", unsafe_allow_html=True)
                cid = selected_cell(sk)
                df = res.candidates.get(sk)
                if cid and df is not None and (df.cell_id == cid).any():
                    r = df[df.cell_id == cid].iloc[0]
                    pf = None                       # P always from the flowchart (the emulation only checks)
                    m = st.columns(4)
                    m[0].metric("Zelle", ui.short(cid))
                    m[1].metric("P/S", f"{pf}/{r.S}" if pf else r.setup,
                                "aus Emulation" if pf else "analytisch", delta_color="off", delta_arrow="off")
                    m[2].metric("C_req", f"{sc.C_req:.2f} F")
                    dts = (inp.selected[sk] or {}).get("dT_max_K", r.dT_max_K) if pf else r.dT_max_K
                    m[3].metric("ΔT max", ui.fmt(dts, 1, "K"))
                else:
                    st.info("Noch keine Zelle gewählt (Schritt 7).")
    st.button("Los geht's: 1 Lastprofil →", on_click=go_to, args=("profil",), type="primary")


def page_balancing():
    ui.step_header("SCHRITT 13", "Balancing-Konzept",
                   "Im Flussdiagramm vorgesehen, im Tool bisher nur als Schnittstelle umgesetzt.")
    st.info("Eigene Balancing-Modelle lassen sich mit @register_balancing in sc_module_design/balancing.py "
            "registrieren. Ein Modell ist noch nicht hinterlegt, deshalb ist der Schritt grau.")
    nav_buttons("balancing")


def page_profil():
    ui.step_header("SCHRITT 1", "Lastprofil und Anforderungen",
                   "Gemessenes Leistungsprofil der Anwendung (positiv = Last, negativ = Rekuperation) und der "
                   "verfügbare Bauraum. Ist nur der Bauraum bekannt, zeigt Seite 5, was hineinpasst.")
    spec_options(inp.req, expanded=True)
    with st.container(border=True):
        c = st.columns([2.2, 1, 1])
        profs = [str(p.relative_to(PROJECT_ROOT)) for p in list_profiles()]
        none_lbl = "(kein Profil: manuelle Werte)"
        cur = inp.profile.path if inp.profile.path in profs else (none_lbl if not inp.profile.path else inp.profile.path)
        options = [none_lbl] + profs + ([cur] if cur not in profs and cur != none_lbl else [])
        choice = c[0].selectbox("Profil", options, index=options.index(cur), key=W.key("profile"),
                                help="Eigene Messungen liegen in data/profiles/private (nicht in git).")
        inp.profile.path = "" if choice == none_lbl else choice
        W.num("Skalierung", inp.profile, "scale", "pscale", fmt="%.6g", container=c[1])
        W.num("P = U I x Faktor", inp.profile, "phase_factor", "pphase", container=c[2],
              help="Nur wenn die Datei Spannung und Strom statt Leistung enthält (3 für 3-phasig).")
        if inp.profile.path and not resolve(inp.profile.path).exists():
            ui.warn("Datei nicht gefunden. Es werden die manuellen Werte verwendet.")
        with st.expander("Profil hochladen, Zeitfenster, manuelle Werte"):
            up = st.file_uploader("CSV mit Zeit [s] und Leistung [W] (oder Spannung und Strom)", type=["csv", "txt"])
            if up is not None:
                dest = PRIVATE_PROFILE_DIR / "uploads" / up.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(up.getvalue())
                st.success(f"Gespeichert in {dest.relative_to(PROJECT_ROOT)} (privat). Oben auswählen.")
            c1, c2 = st.columns(2)
            W.opt_num("t_start [s]", inp.profile, "t_start", "tstart", 0.0, container=c1)
            W.opt_num("t_end [s]", inp.profile, "t_end", "tend", 10.0, container=c2)
            W.select("Manuelle Werte verwenden", inp.manual_load, "use", "ml_use",
                     {"never": "nie", "if_no_profile": "nur ohne Profil", "always": "immer (überschreibt)"})
            c = st.columns(4)
            W.opt_num("P_peak [W]", inp.manual_load, "P_peak_W", "ml_pp", 2643.0, container=c[0])
            W.opt_num("P_avg [W]", inp.manual_load, "P_avg_W", "ml_pa", 392.3, container=c[1])
            W.opt_num("E_DCH,max [J]", inp.manual_load, "E_dch_J", "ml_ed", 1214.1, container=c[2])
            W.opt_num("E_CH,max [J]", inp.manual_load, "E_ch_J", "ml_ec", 557.0, container=c[3])
    res = compute()
    prof = res.profile
    if prof is None:
        st.info("Kein Profil geladen. Gerechnet wird mit den manuellen Werten, Thermik und Emulation brauchen ein Profil.")
        return
    s = res.stats
    c = st.columns(6)
    ui.metric(c[0], "P_avg", f"{s['P_avg_W']:.1f} W", key="P_avg")
    ui.metric(c[1], "P_peak", f"{s['P_peak_W']:.0f} W", key="P_sc_max")
    ui.metric(c[2], "P_RMS", f"{s['P_rms_W']:.0f} W", key="P_rms")
    ui.metric(c[3], "Peak / Mittelwert", f"{s['peak_to_avg']:.1f}", key="P_avg",
              calc=f"{s['P_peak_W']:.0f} W / {s['P_avg_W']:.1f} W")
    c[4].metric("Dauer", f"{s['duration_s']:.1f} s", f"Abtastung {s['dt_s'] * 1e3:.3g} ms", delta_color="off", delta_arrow="off")
    ui.metric(c[5], "Grenze Leistungsspitzen", f"{res.peaks['upper_fence']:.0f} W", "Q3 + 1.5 IQR", key="fence")
    td, pd_ = decimate_for_plot(prof.t, prof.p, 30000)
    fig = go.Figure(go.Scattergl(x=td, y=pd_, name="Last", line=dict(color=COLORS[0], width=1)))
    fig.add_hline(y=s["P_avg_W"], line_dash="dash", line_color="black")
    fig.update_xaxes(title_text="Time [s]", rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(title_text="Power [W]")
    st.plotly_chart(plotly_layout(fig, 380), width="stretch")
    st.markdown("##### Abbildungen (Diss-Stil)")
    c1, c2 = st.columns(2)
    ui.show_fig(pl.fig_peaks(res), "power_peaks", container=c1, plt_module=pl.plt, fx_keys=["fence"])
    ui.show_fig(pl.fig_fft(res), "power_fft", container=c2, plt_module=pl.plt)
    nav_buttons("profil")


def page_split():
    ui.step_header("SCHRITT 2", "Leistungsaufteilung Batterie / Supercap",
                   "Die Batterie liefert den langsamen Anteil, der Supercap den Rest. Der Wandler-Wirkungsgrad wird "
                   "in jedem Zeitschritt angewendet. Daraus folgen E_DCH,max und E_CH,max.")
    with st.container(border=True):
        names = list(REGISTRY)
        c1, c2 = st.columns([1, 2])
        sname = c1.selectbox("Strategie", names, index=names.index(inp.split.name) if inp.split.name in names else 0,
                             format_func=lambda n: REGISTRY[n].label, key=W.key("split"))
        strat = get_split(sname)
        c2.caption(strat.description)
        if sname != inp.split.name:
            inp.split.params = strat.defaults()
        inp.split.name = sname
        if strat.params:
            pc = st.columns(len(strat.params))
            newp = {}
            for i, p in enumerate(strat.params):
                newp.update(W.param_widgets(type(strat)(strat.name, strat.label, strat.func, [p]), inp.split.params,
                                            f"sp_{sname}", container=pc[i]))
            inp.split.params = newp
        c = st.columns(3)
        r = inp.req
        eta_d = c[0].number_input("Wirkungsgrad Entladen η_dch [%]", 50.0, 100.0, float(r.eta_dch * 100), 0.5,
                                  key=W.key("eta_d"), help="SC -> Bus: P_SC,Zelle = P_SC,Bus / η")
        eta_c = c[1].number_input("Wirkungsgrad Laden η_ch [%]", 50.0, 100.0, float(r.eta_ch * 100), 0.5,
                                  key=W.key("eta_c"), help="Bus -> SC: P_SC,Zelle = P_SC,Bus η")
        r.eta_dch, r.eta_ch = eta_d / 100, eta_c / 100
        W.select("Energieauswertung", inp.opt, "energy_method", "emeth", ENERGY_METHODS, container=c[2])
        st.markdown("**Bremsenergie der mobilen Plattform (Modul 2)**")
        c = st.columns(5)
        W.check("berücksichtigen", inp.braking, "enabled", "br_on", container=c[0])
        W.num("Masse [kg]", inp.braking, "mass_kg", "br_m", container=c[1])
        W.num("Geschwindigkeit [m/s]", inp.braking, "v_mps", "br_v", container=c[2])
        W.num("Bremszeit [s]", inp.braking, "t_brake_s", "br_t", container=c[3],
              help="P = 2E/t. 1.0 s ergibt 1980 W (Diss), 1.2 s ergibt 1650 W (IECON-Folie).")
    res = compute()
    prof = res.profile
    if prof is None:
        st.info("Ohne Profil gibt es keine Aufteilung, es gelten die manuellen Energien.")
        nav_buttons("split")
        return
    L = res.load
    c = st.columns(5)
    ui.metric(c[0], "E_DCH,max", f"{L['E_dch_J']:.1f} J", key="E_dch")
    ui.metric(c[1], "E_CH,max", f"{L['E_ch_J']:.1f} J", key="E_ch")
    ui.metric(c[2], "P_SC,max (SC-Seite)", f"{L['P_SC_max_W']:.0f} W", key="P_sc_max",
              calc=f"{L['P_peak_W']:.0f} W / {inp.req.eta_dch:.3f}")
    if inp.braking.enabled:
        ui.metric(c[3], "E_Bremsung (SC-Seite)", f"{L['E_brake_J']:.0f} J", f"P = {L['P_brake_W']:.0f} W",
                  key="E_brake", calc=f"½ {inp.braking.mass_kg:g} kg ({inp.braking.v_mps:g} m/s)² x {inp.req.eta_ch:g}")
    E_pre = -float(np.sum(res.p_sc)) * prof.dt
    nw = int(round(float(inp.split.params.get("window_s", 0.1 * prof.t[-1])) / prof.dt))
    nw = max(1, min(nw, len(res.p_sc) // 2))
    E_in = -float(np.sum(res.p_sc[:nw])) * prof.dt
    E_out = -float(np.sum(res.p_sc[-nw:])) * prof.dt
    ui.metric(c[4], "Precharge-Energie (erste Ladung)", f"{E_pre:.0f} J",
              f"Anlauf {E_in:.0f} J, Auslauf {E_out:.0f} J", key="E_pre",
              calc=f"-∫P_SC dt = {E_pre:.0f} J; Anlauf/Auslauf = Randfenster {nw * prof.dt:.1f} s")
    td, pd_ = decimate_for_plot(prof.t, prof.p, 25000)
    tb, pb = decimate_for_plot(prof.t, res.p_bat, 25000)
    ts, ps = decimate_for_plot(prof.t, res.p_sc, 25000)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scattergl(x=td, y=pd_, name="Last", line=dict(color="#999", width=1)), 1, 1)
    fig.add_trace(go.Scattergl(x=tb, y=pb, name="Batterie", line=dict(color=COLORS[1], width=2)), 1, 1)
    fig.add_trace(go.Scattergl(x=ts, y=ps, name="Supercap (SC-Seite)", line=dict(color=COLORS[0], width=1)), 2, 1)
    fig.update_yaxes(title_text="Power [W]", row=1, col=1)
    fig.update_yaxes(title_text="SC power [W]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    st.plotly_chart(plotly_layout(fig, 460), width="stretch")
    legacy = inp.opt.energy_method == "zero_crossing_legacy" and inp.split.params.get("alignment") == "legacy_matlab"
    c1, c2 = st.columns(2)
    ui.show_fig(pl.fig_energy_excursions(res, "iecon" if legacy else "auto"), "energy_excursions", container=c1,
                plt_module=pl.plt, fx_keys=["split", "eta", "E_dch", "E_ch"])
    with c2:
        st.markdown("**Filtervergleich**")
        cmp_names = st.multiselect("Strategien", list(REGISTRY), default=[inp.split.name],
                                   format_func=lambda n: REGISTRY[n].label, key=W.key("cmp"))
        if cmp_names and res.voltages:
            rows = []
            U_max, U_min = res.voltages["U_C0max"], res.voltages["U_C0min"]
            pkey = profile_key()
            for n in cmp_names:
                prm = inp.split.params if n == inp.split.name else REGISTRY[n].defaults()
                try:
                    pbn, psn = cached_split(pkey, n, json.dumps(prm, sort_keys=True))
                except Exception as exc:
                    st.error(f"{n}: {exc}")
                    continue
                ex = energy_excursions(psn, prof.dt, inp.opt.energy_method)
                rows.append({"Strategie": REGISTRY[n].label, "E_DCH,max [J]": ex.E_dch_J, "E_CH,max [J]": ex.E_ch_J,
                             "U_C0inM [V]": eq.u_c0inm(ex.E_dch_J, ex.E_ch_J, U_max, U_min),
                             "C_req Modul 1 [F]": eq.c_required(ex.E_dch_J + ex.E_ch_J, U_max, U_min, inp.req.L_EOL),
                             "P_Batt max [W]": float(pbn.max())})
            st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    nav_buttons("split")


def _take_I_rec(val: float):
    inp.req.I_C0max_A = float(val)
    st.session_state[W.key("ic0max")] = float(val)


U_MODES = {"r_Q": "r_Q vorgeben (U_C0min = r_Q U_C0max)", "current": "aus Wandlerstrom (P_SC,max / I_C0max, Diss)",
           "fixed": "U_C0min fest"}


def page_spannung():
    ui.step_header("SCHRITTE 1 BIS 3", "Modulspannungen, U_C0min und Wandlerstrom",
                   "U_C0max aus dem DC-Bus, Spannungsderating nach Arrhenius für die Lebensdauer, U_C0min über das "
                   "Entladeverhältnis r_Q (Start 0,5), daraus der nötige Strom des Interface-Wandlers.")
    r = inp.req
    with st.container(border=True):
        c = st.columns(4)
        W.num("U_Bus [V]", r, "U_bus_V", "ubus", container=c[0])
        W.opt_num("U_C0max [V]", r, "U_C0max_V", "uc0max", r.U_bus_V, container=c[1])
        W.num("Zelltemperatur [°C]", r, "T_cell_C", "tcell", container=c[2])
        W.num("Lebensdauer [h]", r, "t_life_h", "tlife", step=1000.0, container=c[3])
        c = st.columns(4)
        W.select("P_SC,max", r, "P_max_mode", "pmaxmode",
                 {"peak": "P_peak (Diss)", "peak_minus_avg": "P_peak - P_avg", "manual": "manuell"}, container=c[0])
        if r.P_max_mode == "manual":
            W.opt_num("P_SC,max [W]", r, "P_max_W", "pmaxw", 2643.0, container=c[1])
        series = ["(je Zelle aus DB)"] + list(db.lifetime.series_id)
        cur = inp.opt.series_override or series[0]
        sser = c[2].selectbox("Lebensdauer-Serie", series, index=series.index(cur) if cur in series else 0,
                              key=W.key("series"))
        inp.opt.series_override = None if sser == series[0] else sser
    with st.container(border=True):
        st.markdown("**U_C0min**")
        c = st.columns([1.5, 1, 0.2, 1.6])
        W.select("Festlegung", r, "U_C0min_mode", "uminmode", U_MODES, container=c[0])
        if r.U_C0min_mode == "r_Q":
            W.num("r_Q = U_C0min / U_C0max", r, "r_Q", "rq", step=0.05, min_value=0.05, max_value=0.95, container=c[1],
                  help="Start 0,5: 75 % der Energie nutzbar (1 - r_Q²).")
        elif r.U_C0min_mode == "fixed":
            if r.U_C0min_override_V is None:
                r.U_C0min_override_V = 0.5 * (r.U_C0max_V or r.U_bus_V)
            W.num("U_C0min [V]", r, "U_C0min_override_V", "uminfix", step=0.5, container=c[1])
        W.opt_num("Wandler U_ein,min [V]", r, "U_conv_min_V", "uconv", 0.4 * (r.U_C0max_V or r.U_bus_V), container=c[3],
                  help="Optional: minimale Eingangsspannung des Wandlers (Klemmenspannung). Prüfung in der Emulation.")
    res = compute()
    if not res.voltages:
        st.error(res.steps.get("2").summary if res.steps.get("2") else "Keine Spannungen berechnet.")
        return
    v = res.voltages
    si = next(iter(res.series_info.values()))
    U_max = v["U_C0max"]
    c = st.columns(5)
    ui.metric(c[0], "U_C01max je Zelle", f"{si['U_C01max']:.3f} V", key="U_C01max",
              calc=f"T = {r.T_cell_C:g} °C, t_life = {r.t_life_h:g} h")
    ui.metric(c[1], "Zellen in Serie S", si["S"], f"U_C0N = {si['U_C0N']:.1f} V", key="S",
              calc=f"ceil({U_max:.1f} V / {si['U_C01max']:.3f} V) = {si['S']}")
    calc_u = {"r_Q": f"{r.r_Q:.2f} x {U_max:.1f} V = {v['U_C0min']:.2f} V",
              "current": f"{res.load['P_SC_max_W']:.0f} W / {r.I_C0max_A:g} A = {v['U_C0min']:.2f} V",
              "fixed": f"fest {v['U_C0min']:.2f} V"}[v["mode"]]
    ui.metric(c[2], "U_C0min", f"{v['U_C0min']:.2f} V", U_MODES[v["mode"]].split(" (")[0], key="U_C0min",
              calc=calc_u)
    ui.metric(c[3], "Nutzbare Energie", f"{100 * v['energy_ratio']:.0f} %", f"r_Q = {v['r_Q']:.2f}", key="r_Q",
              calc=f"1 - {v['r_Q']:.2f}² = {v['energy_ratio']:.2f}")
    ui.metric(c[4], "U_C0inM", " / ".join(f"{sc.U_inM:.1f}" for sc in res.scenarios.values()) + " V",
              "Modul 1 / Modul 2", key="U_C0inM")
    if res.steps["2"].status == "warning":
        ui.warn(res.steps["2"].summary)
    with st.container(border=True):
        st.markdown("**Interface-Wandler: Strom aus dem Profil**")
        c = st.columns(4)
        I_rec = v["I_C0max_rec"]
        ui.metric(c[0], "Spitzenstrom bei U_C0min", f"{I_rec:.1f} A", key="I_rec",
                  calc=f"{res.load['P_SC_max_W']:.0f} W / {v['U_C0min']:.2f} V = {I_rec:.1f} A")
        if v.get("I_rms_at_Umin") == v.get("I_rms_at_Umin"):
            ui.metric(c[1], "Dauerstrom (RMS)", f"{v['I_rms_at_Umin']:.1f} A",
                      f"typisch {v.get('I_rms_at_UinM', float('nan')):.1f} A bei U_C0inM", key="I_rec",
                      calc=f"P_SC,RMS = {v['P_SC_rms']:.0f} W / {v['U_C0min']:.2f} V")
        W.num("I_C0max (Wandler) [A]", r, "I_C0max_A", "ic0max", step=0.5, container=c[2],
              help="Strombegrenzung des Interface-Wandlers. Geht in P_I und die Emulation ein.")
        rec_val = math.ceil(I_rec * 10) / 10
        c[3].button(f"Empfehlung {rec_val:g} A übernehmen", on_click=_take_I_rec, args=(rec_val,),
                    key=W.key("take_irec"), width="stretch")
        W.check(f"Ich bestätige I_C0max = {r.I_C0max_A:g} A für den Wandler", r, "I_C0max_confirmed", "ic0conf",
                container=c[3])
        if not r.I_C0max_confirmed:
            ui.warn(f"I_C0max = {r.I_C0max_A:g} A ist nicht bestätigt. Empfehlung aus dem Profil: {I_rec:.1f} A "
                    f"Spitze bei U_C0min, Dauerstrom {v.get('I_rms_at_Umin', float('nan')):.1f} A (RMS). Bitte prüfen "
                    f"und bestätigen.")
        ui.fx(["I_rec", "U_C0min", "r_Q", "P_I"])
    c1, c2 = st.columns(2)
    ui.show_fig(pl.fig_lifetime(res, db), "lifetime_SC", container=c1, plt_module=pl.plt, fx_keys=["U_C01max", "S"])
    ui.show_fig(pl.fig_discharge_ratio(res), "dischargeratio", container=c2, plt_module=pl.plt, fx_keys=["r_Q"])
    sk = st.selectbox("Spannungsdiagramm für", list(res.scenarios), format_func=lambda s: SCENARIOS[s],
                      index=len(res.scenarios) - 1, key=W.key("vlev"))
    ui.show_fig(pl.fig_voltage_levels(res, sk), f"voltage_levels_{sk}", plt_module=pl.plt,
                fx_keys=["U_C0min", "U_C0inM", "U_C01max"])
    nav_buttons("spannung")


def page_kapazitaet():
    ui.step_header("SCHRITT 4", "Erforderliche Modulkapazität",
                   "Die nutzbare Energie zwischen U_C0max und U_C0min muss am Lebensende (L_EOL) reichen.")
    with st.container(border=True):
        c = st.columns(2)
        W.num("L_EOL (Restkapazität am Lebensende)", inp.req, "L_EOL", "leol", min_value=0.05, max_value=1.0,
              container=c[0])
        scen = c[1].multiselect("Szenarien", list(SCENARIOS), default=[s for s in inp.opt.scenarios if s in SCENARIOS],
                                format_func=lambda s: SCENARIOS[s], key=W.key("scen"))
        inp.opt.scenarios = scen or ["profile"]
        W.check("ESR-Verluste beim Entladen einrechnen (je Zelle, Beharrungszustand)", inp.opt, "include_esr_losses",
                "esrloss", help="Entladen braucht E_DCH + I² R-Verluste. Weil R vom Modul abhängt, wird das je "
                                "Kandidat in Schritt 5 gerechnet (Spalten Verluste Entladen, C_req inkl. Verluste). "
                                "Aus = wie IECON/Diss.")
    res = compute()
    st.latex(r"C_\mathrm{req}=\frac{2W_\mathrm{SC,used}}{L_\mathrm{EOL}\left(U_\mathrm{C0max}^2-U_\mathrm{C0min}^2"
             r"\right)},\qquad U_\mathrm{C0inM}=\sqrt{\frac{E_\mathrm{DCH,max}U_\mathrm{C0max}^2+E_\mathrm{CH,max}"
             r"U_\mathrm{C0min}^2}{E_\mathrm{DCH,max}+E_\mathrm{CH,max}}}")
    cols = st.columns(max(1, len(res.scenarios)))
    for col, sc in zip(cols, res.scenarios.values()):
        with col.container(border=True):
            st.markdown(f"<div class='card-title'>{sc.label}</div>", unsafe_allow_html=True)
            m = st.columns(3)
            ui.metric(m[0], "W_used", f"{sc.W_used_J:.0f} J", f"{sc.E_dch_J:.0f} + {sc.E_ch_J:.0f} J", key="C_req",
                      calc=f"W_used = {sc.E_dch_J:.1f} J + {sc.E_ch_J:.1f} J")
            ui.metric(m[1], "U_C0inM", f"{sc.U_inM:.2f} V", key="U_C0inM",
                      calc=f"sqrt(({sc.E_dch_J:.0f} x {sc.U_max:.1f}² + {sc.E_ch_J:.0f} x {sc.U_min:.1f}²) / "
                           f"{sc.W_used_J:.0f}) = {sc.U_inM:.2f} V")
            ui.metric(m[2], "C_req", f"{sc.C_req:.3f} F", key="C_req",
                      calc=f"2 x {sc.W_used_J:.0f} J / ({inp.req.L_EOL:g} ({sc.U_max:.1f}² - {sc.U_min:.1f}²) V²) = "
                           f"{sc.C_req:.3f} F")
    ui.fx(["C_req", "U_C0inM", "E_loss"])
    nav_buttons("kapazitaet")


SEL_COLS = ["cell_id", "setup", "P_driver", "P_C_no_loss", "P_C", "P_I", "P_T", "E_loss_dch_J", "C_req_loss_F",
            "C_cell_used_F", "ESR_cell_mOhm", "C_module_F", "C_reserve", "C_reserve_BOL", "I_peak_Umin_A",
            "ESR_module_mOhm", "I_module_max_A", "cost_EUR", "mass_kg", "volume_dm3", "P_loss_avg_W", "dT_max_K",
            "dT_net_max_K", "pcb_mm", "density_kg_dm3", "status", "warnings"]


def page_zellen():
    ui.step_header("SCHRITTE 5 BIS 6, 11", "Zellauswahl und Parallelschaltung",
                   "Für jede Kandidatenzelle: P_C aus der Kapazität, P_I aus dem Maximalstrom, p = max(P_C, P_I). "
                   "Kapazität und ESR wahlweise aus dem Datenblatt oder aus der eigenen Charakterisierung.")
    o = inp.opt
    with st.container(border=True):
        c = st.columns(3)
        W.select("Kapazität", o, "c_source", "csrc", C_SOURCES, container=c[0])
        W.select("ESR", o, "esr_source", "esrsrc", ESR_SOURCES, container=c[1])
        W.select("Volumen", o, "geometry", "geom", GEOMETRY_MODES, container=c[2])
        if o.c_source != "datasheet":
            c = st.columns(3)
            W.select("Spannungsfenster C0/k", o, "cfit_range", "crng",
                     {"90-20": "90-20 % (geringster Fehler)", "80-40": "80-40 % (IEC 62391-1)"}, container=c[0],
                     help="90-70 % ist nur für ESR geeignet und wird hier nicht angeboten.")
            cur_opts = ["highest", "lowest", "nearest_rms"]
            cv = o.cfit_current if o.cfit_current in cur_opts else "highest"
            o.cfit_current = c[1].selectbox("Messstrom", cur_opts, index=cur_opts.index(cv), key=W.key("ccur"),
                                            format_func={"highest": "höchster", "lowest": "niedrigster",
                                                         "nearest_rms": "nächst am RMS-Zellstrom"}.get)
            dsets = ["auto", "a", "b", "c", "d"]
            o.cfit_dataset = c[2].selectbox("Datensatz", dsets, index=dsets.index(o.cfit_dataset), key=W.key("cds"))
        if o.geometry == "pcb":
            pcb_options(o)
        spec_summary(inp.req)
        with st.expander("Kandidatenzellen"):
            allc = list(db.cells.cell_id)
            dflt = o.candidate_ids or list(db.cells[db.cells.candidate].cell_id)
            sel = st.multiselect("Zellen", allc, default=[x for x in dflt if x in allc], key=W.key("cands"),
                                 format_func=ui.short)
            o.candidate_ids = sel if set(sel) != set(db.cells[db.cells.candidate].cell_id) else None
    res = compute()
    for sk, df in res.candidates.items():
        sc = res.scenarios[sk]
        st.markdown(f"#### {sc.label}")
        st.caption(f"C_req = {sc.C_req:.3f} F ohne Verluste (je Zelle inkl. ESR-Verluste beim Entladen: Spalte C_req "
                   f"inkl. Verluste), U_C0min = {sc.U_min:.1f} V, I_C0max = {sc.I_C0max:g} A. Zeile anklicken, um die "
                   f"Zelle im Zelldetail zu öffnen.")
        dfo = pd.concat([df[df.feasible], df[~df.feasible]])
        if (~df.feasible).any():
            st.caption(f"{int((~df.feasible).sum())} Konfiguration(en) außerhalb der Vorgabe stehen unten "
                       f"(Spalte Status) und sind in den Abbildungen ausgegraut.")
        ev = ui.table(dfo, SEL_COLS, key=W.key(f"tbl_{sk}"), selectable=True,
                      fx_keys=["E_loss", "P_C", "P_I", "P_T", "P_driver", "C_eq", "C_mod", "reserve", "I_peak", "mass",
                               "cost", "P_loss", "R_th_eff"])
        rows = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
        if rows:
            cid = dfo.iloc[rows[0]].cell_id
            st.session_state["detail_cell"], st.session_state["detail_scen"] = cid, sk
            st.button(f"{ui.short(cid)} im Zelldetail öffnen", on_click=go_to, args=("detail",), key=W.key(f"od_{sk}"))
    if res.candidates:
        space_section(res)
    if res.candidates and o.geometry == "pcb":
        layout_section(res)
    if res.skipped_cells:
        with st.expander(f"{len(res.skipped_cells)} Zellen übersprungen (fehlende Daten)"):
            st.write("\n".join(f"- {m}" for m in res.skipped_cells))
    nav_buttons("zellen")


def pcb_options(o):
    with st.expander("Leiterplatte: Packung, Ausrichtung, Stecker", expanded=False):
        st.caption("Die Gruppen (P Zellen parallel) laufen als U: vorne in der Mitte los, links nach hinten, "
                   "rechts zurück nach vorne. Plus und Minus liegen vorne nebeneinander. Freie Plätze bleiben "
                   "vorne in der Mitte, dort sitzen die Stecker, damit die Platine nicht größer wird.")
        c = st.columns(4)
        W.select("Packung", o, "packing", "pk", PACKINGS, container=c[0])
        W.select("Ausrichtung", o, "pcb_orientation", "pcbor", ORIENTATIONS, container=c[1],
                 help="lang = Tiefe (von vorne nach hinten) größer als die Breite der Front")
        W.select("Gruppenform", o, "pcb_group_shape", "pcbgs", GROUP_SHAPES, container=c[2])
        W.integer("Spalten (0 = auto)", o, "pcb_bands", "pcbb", min_value=0, max_value=60, container=c[3],
                  help="Anzahl Gruppenspalten über die Breite (gerade, je Hälfte des U gleich viele)")
        c = st.columns(4)
        W.num("Zellabstand [mm]", o, "cell_spacing_mm", "gap", 0.1, min_value=0.0, container=c[0])
        W.num("Rand [mm]", o, "pcb_margin_mm", "marg", 0.5, min_value=0.0, container=c[1])
        W.num("Ziel-Seitenverhältnis lang/breit", o, "pcb_target_aspect", "tasp", 0.1, min_value=1.0, container=c[2])
        W.num("Leiterplatte [g/cm²]", o, "pcb_areal_mass_g_cm2", "pcbm", 0.05, min_value=0.0, container=c[3],
              help="FR4 1,6 mm mit Kupfer etwa 0,35 g/cm². Geht in Modulmasse und Moduldichte ein.")
        c = st.columns(4)
        W.num("Leistungsstecker B [mm]", o, "conn_power_w_mm", "cpw", 1.0, min_value=0.0, container=c[0])
        W.num("Leistungsstecker T [mm]", o, "conn_power_d_mm", "cpd", 1.0, min_value=0.0, container=c[1])
        W.num("Signalstecker B [mm]", o, "conn_signal_w_mm", "csw", 1.0, min_value=0.0, container=c[2],
              help="z. B. Balancing/Sense, 0 = keiner")
        W.num("Signalstecker T [mm]", o, "conn_signal_d_mm", "csd", 1.0, min_value=0.0, container=c[3])


SPEC_KEYS = ("max_width_mm", "max_depth_mm", "max_height_mm", "max_volume_dm3", "max_mass_kg", "max_cost_EUR")


def spec_summary(r):
    act = [(k, getattr(r, k)) for k in SPEC_KEYS if getattr(r, k) is not None]
    names = dict(max_width_mm="Breite", max_depth_mm="Tiefe", max_height_mm="Höhe", max_volume_dm3="Volumen",
                 max_mass_kg="Masse", max_cost_EUR="Kosten")
    units = dict(max_width_mm="mm", max_depth_mm="mm", max_height_mm="mm", max_volume_dm3="dm³", max_mass_kg="kg",
                 max_cost_EUR="€")
    c = st.columns([4, 1])
    if act:
        c[0].markdown("**Bauraum und Vorgaben:** " + ", ".join(f"{names[k]} ≤ {v:g} {units[k]}" for k, v in act))
    else:
        c[0].caption("Keine Bauraum-Vorgabe gesetzt.")
    c[1].button("Bauraum ändern", on_click=go_to, args=("profil",), key=W.key("go_spec"), width="stretch")


def spec_options(r, expanded: bool = False):
    with st.expander("Bauraum und Vorgaben (Modulgröße, Masse, Kosten)", expanded=expanded or any(
            getattr(r, k) is not None for k in SPEC_KEYS)):
        st.caption("Bauraum für das Modul (Leiterplatte von oben: Breite = Front mit Steckern, Tiefe = nach hinten, "
                   "Höhe = Zelllänge + Leiterplatte + Abstand). Konfigurationen, die eine Vorgabe verletzen, sind "
                   "unzulässig und in allen Abbildungen grau. Breite und Tiefe steuern auch die Wahl des Layouts.")
        c = st.columns(3)
        W.opt_num("Breite max [mm]", r, "max_width_mm", "mxw", 200.0, container=c[0])
        W.opt_num("Tiefe max [mm]", r, "max_depth_mm", "mxd", 200.0, container=c[1])
        W.opt_num("Höhe max [mm]", r, "max_height_mm", "mxh", 50.0, container=c[2])
        c = st.columns(3)
        W.opt_num("Volumen max [dm³]", r, "max_volume_dm3", "mxv", 1.5, container=c[0])
        W.opt_num("Masse max [kg]", r, "max_mass_kg", "mxm", 1.5, container=c[1])
        W.opt_num("Kosten max [€]", r, "max_cost_EUR", "mxc", 300.0, container=c[2])


def layout_extent(res) -> tuple:
    """Largest PCB (width, depth) of all candidates and both packings: common scale for every layout view."""
    W_, D_ = 0.0, 0.0
    for df in res.candidates.values():
        for _, r in df.iterrows():
            cell = db.cell(r.cell_id)
            for lay in layout_pair(cell, int(r.S), int(r.P), inp.opt, inp.req):
                if lay is not None:
                    W_, D_ = max(W_, lay.width_mm), max(D_, lay.depth_mm)
    return (W_, D_) if W_ else None


def space_section(res):
    """'Only the space is given': largest P that fits per cell and which requirements it meets."""
    r = inp.req
    if all(getattr(r, k) is None for k in SPEC_KEYS):
        return
    st.markdown("#### Was passt in den Bauraum?")
    sk = list(res.candidates)[-1]
    sc = res.scenarios[sk]
    rows = []
    for _, row in res.candidates[sk].iterrows():
        cell = db.cell(row.cell_id)
        S = int(row.S)
        Pm = max_P_in_space(cell, S, inp.opt, r)
        C_cell = float(row.C_cell_used_F)
        C_m = Pm * C_cell / S if Pm else 0.0
        R_m = S * float(row.ESR_cell_mOhm) / Pm if Pm else float("nan")
        P_need = int(row.P)
        rows.append({"Zelle": ui.short(row.cell_id), "S": S, "P benötigt": P_need, "P max im Bauraum": Pm,
                     "passt": "ja" if Pm >= P_need else "nein",
                     "C Modul bei P max [F]": C_m,
                     "E nutzbar EOL [J]": 0.5 * C_m * r.L_EOL * (sc.U_max ** 2 - sc.U_min ** 2),
                     "I max Modul [A]": Pm * float(cell["i_max_A"]), "ESR Modul [mΩ]": R_m,
                     "C_req erfüllt": "ja" if Pm >= int(row.P_C) else "nein",
                     "Strom erfüllt": "ja" if Pm >= int(row.P_I) else "nein",
                     "Thermik erfüllt": ("ja" if Pm >= P_need else "nein") if row.P_T == row.P_T else "-"})
    d = pd.DataFrame(rows)
    st.caption(f"{sc.label}: je Zelle das größte P, dessen Modul (Layout {PACKINGS.get(inp.opt.packing)}) in den "
               f"Bauraum passt, und welche Anforderungen damit erfüllt sind. Thermik erfüllt = P max >= P "
               f"(inklusive P_T und thermischer Erhöhung).")
    st.dataframe(d.round(2), hide_index=True, width="stretch")
    n_ok = int((d["passt"] == "ja").sum())
    if n_ok == 0:
        ui.warn("Keine Zelle erfüllt alle Anforderungen im vorgegebenen Bauraum.")


def layout_section(res):
    st.markdown("#### Leiterplatten-Layout (Modulvolumen, Standard hexagonal)")
    st.caption("Alle Module im selben Maßstab: das größte Modul aller Kandidaten füllt den Rahmen.")
    keys = list(res.candidates)
    c = st.columns([1, 1.4, 2])
    sk = c[0].selectbox("Szenario", keys, index=len(keys) - 1, format_func=lambda s: SCENARIOS[s],
                        key=W.key("lay_scen"))
    df = res.candidates[sk]
    ids = list(df.cell_id)
    cur = selected_cell(sk) if selected_cell(sk) in ids else ids[0]
    cid = c[1].selectbox("Zelle", ids, index=ids.index(cur), format_func=ui.short, key=W.key(f"lay_cell_{sk}_{cur}"))
    row = df[df.cell_id == cid].iloc[0]
    cell = db.cell(cid)
    sq, hx = layout_pair(cell, int(row.S), int(row.P), inp.opt, inp.req)
    if sq is None:
        st.info("Für diese Zelle fehlen Durchmesser oder Länge.")
        return
    main = hx if inp.opt.packing == "hex" else sq
    m_cells = int(row.S) * int(row.P) * float(cell["mass_g"]) / 1000
    m_mod = m_cells + main.pcb_mass_kg(inp.opt.pcb_areal_mass_g_cm2)
    with c[2]:
        cc = st.columns(3)
        ui.metric(cc[0], "Platine", f"{main.width_mm:.0f} x {main.depth_mm:.0f} mm", key="layout",
                  calc=f"p = {main.d_mm:g} + {main.gap_mm:g} mm, {main.label()}")
        ui.metric(cc[1], "Modulvolumen", f"{main.volume_dm3:.2f} dm³", key="mass",
                  calc=f"{main.width_mm:.0f} x {main.depth_mm:.0f} x {main.height_mm:.1f} mm")
        ui.metric(cc[2], "Moduldichte", f"{m_mod / main.volume_dm3:.2f} kg/dm³", key="mass",
                  calc=f"({m_cells:.3f} kg Zellen + {main.pcb_mass_kg(inp.opt.pcb_areal_mass_g_cm2):.3f} kg "
                       f"Leiterplatte) / {main.volume_dm3:.3f} dm³")
    for n in main.notes:
        ui.warn(n)
    if not row.feasible:
        ui.warn(f"{ui.short(cid)}: {row.status}")
    ui.show_fig(pl.fig_layout_compare(sq, hx, m_cells, inp.opt.pcb_areal_mass_g_cm2, extent=layout_extent(res)),
                f"layout_compare_{ui.short(cid).replace(' ', '_')}_{row.setup.replace('/', 'x')}", plt_module=pl.plt,
                fx_keys=["layout", "mass"])


TH_COLS = ["cell_id", "setup", "P_driver", "P_T", "I_cell_rms_A", "ESR_cell_mOhm", "R_th_KperW", "k_coupling",
           "R_th_eff_KperW", "dT_max_K", "dT_net_max_K", "dT_net_centre_K", "dT_net_outer_K", "T_cell_max_C",
           "thermal_status", "warnings"]

TH_INFO = {
    "A": ("A  Einzelzelle (Diss Anhang A01)",
          "C_th dΔT/dt = P_v - ΔT/R_th mit R_th = 22,1 K/W und C_th = 19,7 J/K, an der Einzelzelle gemessen "
          "(Diss Kap. 5). Gilt für eine freie Zelle in ruhender Luft."),
    "B": ("B  Blockmodell / Lumped (Lehrbuch)",
          "R_th,eff = k R_th mit k = n A_Zelle / A_Box: Newton-Kühlung und Lumped Capacitance nach Incropera et al., "
          "Fundamentals of Heat and Mass Transfer, Kap. 5 (gültig für Bi < 0,1). Für SC-Module nicht validiert, "
          "FEM empfohlen. Eine Temperatur für alle Zellen, kein Gradient."),
    "C": ("C  Netzwerkmodell (eigener Ansatz, Precheck)",
          "Eigener Ansatz von Andy, nicht validiert, FEM empfohlen. Jede Zelle Wärmequelle, Nachbarkopplung über "
          "R_gap (Näherung MathWorks), lokale Luft als gewählte Mittelungsregel. Zeigt den Gradienten innen nach "
          "außen, die Höhe hängt stark von der nicht gemessenen Mittelungsregel ab. Netzwerkansätze für SC-Stacks: "
          "Hijazi et al., IEEE TIE 59(2), 2012."),
}


def th_head(key: str):
    t, txt = TH_INFO[key]
    st.markdown(f"#### {t}")
    st.caption(txt)


def page_thermik():
    ui.step_header("SCHRITTE 7 BIS 10", "Thermische Prüfung",
                   "Drei Modelle, getrennt nach Herkunft: A Einzelzelle (Diss, gemessen), B Blockmodell (Lehrbuch, "
                   "für SC-Module nicht validiert), C Netzwerk (eigener Ansatz, nicht validiert). Für B und C ist "
                   "eine FEM-Rechnung oder Messung am Modul empfohlen.")
    o, r = inp.opt, inp.req
    with st.container(border=True):
        st.markdown("**Gemeinsame Annahmen**")
        c = st.columns(4)
        W.select("R_th / C_th", o, "thermal_mode", "thm", THERMAL_MODES, container=c[0])
        esr_opts = {1.0: "neue Zelle (x1)", 2.0: "Lebensende (x2)"}
        cur = o.thermal_esr_factor if o.thermal_esr_factor in esr_opts else 1.0
        o.thermal_esr_factor = c[1].selectbox("ESR für Verluste", list(esr_opts), index=list(esr_opts).index(cur),
                                              format_func=esr_opts.get, key=W.key("esrf"))
        W.num("ΔT_max [K]", r, "dT_max_K", "dtmax", container=c[2])
        W.num("T_amb [°C]", r, "T_amb_C", "tamb", container=c[3])
        c = st.columns(4)
        W.select("Modulstrom", o, "thermal_current", "thc", CURRENT_MODES, container=c[0])
        W.select("Profil", o, "thermal_profile", "thp", PROFILE_MODES, container=c[1])
        W.select("Wärmeabfuhr A/B", o, "thermal_coupling", "thc_coup", COUPLING_MODES, container=c[2],
                 help="single_cell = Modell A, packed_block = Modell B, factor = eigener Faktor")
        if o.thermal_coupling == "factor":
            W.num("Faktor auf R_th", o, "thermal_coupling_factor", "kfac", container=c[3])
        else:
            W.num("Zellabstand [mm]", o, "cell_spacing_mm", "gap", container=c[3],
                  help="Gilt für Layout, Blockmodell und Luftspalt im Netzwerk.")
    res = compute()
    if not res.candidates:
        st.info("Keine Kandidaten.")
        return
    c = st.columns([1, 1.4])
    sk = c[0].selectbox("Szenario", list(res.candidates), index=len(res.candidates) - 1,
                        format_func=lambda s: SCENARIOS[s], key=W.key("th_scen"))
    df = res.candidates[sk]
    cands = list(df[df.dT_max_K.notna()].cell_id)
    tabs = st.tabs(["A/B  Schritte 7 bis 10 (Einzelzelle, Block)", "C  Netzwerk (Precheck)", "Vergleich und Messung",
                    "C  Netzwerk: ideale Anordnung"])
    with tabs[0]:
        th_head("A")
        th_head("B")
        cols = [c_ for c_ in TH_COLS if not c_.startswith("dT_net")]
        ui.table(df, cols, key=W.key(f"th_tbl_{sk}"), fx_keys=["P_T", "th_first", "R_th_eff", "th_scale", "esr_eol"])
        if not cands:
            st.info("Keine Zelle mit R_th/C_th (Schritte 8 bis 10 übersprungen).")
        else:
            ui.show_fig(pl.fig_dT_candidates(df, r.dT_max_K), f"dT_candidates_{sk}", plt_module=pl.plt,
                        fx_keys=["th_first", "R_th_eff"])
    if not cands:
        nav_buttons("thermik")
        return
    dflt = selected_cell(sk) if selected_cell(sk) in cands else cands[0]
    cid = c[1].selectbox("Zelle für die Detailansicht", cands, index=cands.index(dflt), format_func=ui.short,
                         key=W.key(f"th_cell_{sk}_{dflt}"))
    det = res.details[sk][cid]
    row = df[df.cell_id == cid].iloc[0]
    with tabs[0]:
        st.markdown(f"**{ui.short(cid)} {row.setup}**: Platine {row.pcb_mm} mm, I_Zelle,RMS = {row.I_cell_rms_A:.2f} A")
        c1, c2 = st.columns(2)
        if det.breakdown:
            ui.show_fig(pl.fig_thermal_breakdown(det.breakdown, r.dT_max_K), f"thermal_breakdown_{cid}",
                        container=c1, plt_module=pl.plt, fx_keys=["th_first", "th_scale", "R_th_eff", "esr_eol"])
        if det.dT is not None:
            ui.show_fig(pl.fig_thermal(det, res.scenarios[sk], r.dT_max_K), f"thermal_{cid}", container=c2,
                        plt_module=pl.plt, fx_keys=["th_first"])
        with st.expander("Rechenweg"):
            st.code("\n".join(det.log))
            for w in det.warnings:
                ui.warn(w)
    with tabs[1]:
        th_head("C")
        if not o.thermal_network:
            st.info("Netzwerkmodell ist auf der Startseite ausgeschaltet.")
        else:
            c = st.columns(3)
            W.check("Strahlung oben", o, "net_radiation", "netrad", container=c[0])
            W.num("Emissionsgrad ε", o, "net_eps", "neteps", step=0.01, container=c[1])
            W.check("Netzwerk für Auslegung (P erhöhen)", o, "net_use_for_design", "netdes", container=c[2],
                    help="P erhöhen, bis die heißeste Zelle im Netzwerk unter ΔT_max liegt. Aus = nur Prüfung.")
            ui.table(df, ["cell_id", "setup", "I_cell_rms_A", "dT_net_max_K", "dT_net_centre_K", "dT_net_outer_K",
                          "dT_max_K"], key=W.key(f"th_net_tbl_{sk}"), fx_keys=["net", "R_gap", "Q_rad"])
            network_section(res, sk, cid, row, det)
    with tabs[2]:
        model_compare_section(res, sk, cid, row, det)
    with tabs[3]:
        th_head("C")
        if o.thermal_network:
            network_ideal_section()
        else:
            st.info("Netzwerkmodell ist auf der Startseite ausgeschaltet.")
    nav_buttons("thermik")


def network_section(res, sk, cid, row, det):
    """Andy's thermal network on the real PCB layout of the chosen configuration (precheck)."""
    o, r = inp.opt, inp.req
    st.markdown(f"**Auf dem Leiterplatten-Layout: {ui.short(cid)} {row.setup}** (nicht validiert, FEM empfohlen)")
    net = det.network
    if net is None:
        st.info("Kein Netzwerkergebnis (braucht R_th/C_th, ein Profil und das Leiterplatten-Layout).")
        return
    s_ = net.summary()
    c = st.columns(6)
    ui.metric(c[0], "Mitte ΔT", f"{s_['dT_centre_K']:.1f} K", key="net")
    ui.metric(c[1], "Heißeste ΔT", f"{s_['dT_max_K']:.1f} K", key="net")
    ui.metric(c[2], "Außen ΔT", f"{s_['dT_outer_mean_K']:.1f} K", key="net")
    ui.metric(c[3], "Mitte - außen", f"{s_['dT_centre_outer_K']:.1f} K", key="net")
    ui.metric(c[4], "R_gap", f"{s_['R_gap_KperW']:.1f} K/W", f"s = {net.params.s_mm:g} mm", key="R_gap",
              calc=f"r = {net.params.d_mm / 2:g} mm, L = {net.params.L_mm:g} mm, λ = {net.params.lam:g} W/(m K)")
    ui.metric(c[5], "P Zelle", f"{s_['P_cell_W'] * 1000:.0f} mW", f"{s_['P_total_W']:.1f} W, "
              f"{s_['N']} Zellen", key="th_first",
              calc=f"{row.I_cell_rms_A:.2f}² A² x {row.ESR_cell_mOhm * o.thermal_esr_factor:.1f} mΩ")
    c = st.columns(4)
    c[0].caption(f"Lokale Luft Mitte {s_['T_air_centre_C']:.1f} °C, außen {s_['T_air_outer_mean_C']:.1f} °C")
    if o.net_radiation:
        c[1].caption(f"Strahlung gesamt {s_['Q_rad_total_W'] * 1000:.0f} mW (ε = {o.net_eps:g})")
    if s_["dT_max_K"] > r.dT_max_K:
        ui.warn(f"Heißeste Zelle {s_['dT_max_K']:.1f} K über ΔT_max = {r.dT_max_K:g} K (Netzwerk, Precheck).")
    ext = layout_extent(res)
    c1, c2 = st.columns(2)
    ui.show_fig(pl.fig_network_heatmap(net, extent=ext), f"network_{ui.short(cid).replace(' ', '_')}_{row.setup.replace('/', 'x')}",
                container=c1, plt_module=pl.plt, fx_keys=["net", "R_gap", "Q_rad"])
    with c2:
        t_end = st.select_slider("Aufheizen bis [min]", [30, 60, 120, 240, 480], value=120, key=W.key("net_tend"))
        t, T = _net_transient(net, (cid, row.setup, round(net.P_cell_W, 9), net.params.s_mm, net.params.radiation,
                                    net.params.eps, net.params.R_air, net.grid.n), 60.0 * t_end)
        ui.show_fig(pl.fig_network_transient(t, T, net.grid, net.params.T_inf_C, r.dT_max_K),
                    f"network_heating_{ui.short(cid).replace(' ', '_')}", plt_module=pl.plt, fx_keys=["net"])


def model_compare_section(res, sk, cid, row, det):
    """Single cell, block (lumped) and network side by side, plus own measurement."""
    o, r = inp.opt, inp.req
    net = det.network
    if net is None:
        st.info("Vergleich braucht das Netzwerkergebnis (Netzwerk auf der Startseite aktiv, R_th/C_th, Profil, Layout).")
        return
    s_ = net.summary()
    th = det.thermal or {}
    P_v = net.P_cell_W
    rth = th.get("rth", float("nan"))
    k = row.k_coupling if row.k_coupling == row.k_coupling else 1.0
    cmp_rows = [
        {"Modell": "A Einzelzelle (k = 1), Mittelwert", "ΔT [K]": P_v * rth,
         "Herkunft / Stand": "Diss A01, R_th an der Einzelzelle gemessen"},
        {"Modell": f"B Block/Lumped (k = {k:.2f}), Mittelwert", "ΔT [K]": P_v * rth * k,
         "Herkunft / Stand": "Lehrbuchansatz (Incropera Kap. 5), für SC-Module nicht validiert"},
        {"Modell": "B Block/Lumped transient (Maximum)", "ΔT [K]": row.dT_max_K, "Herkunft / Stand": "wie B"},
        {"Modell": "C Netzwerk: Mitte", "ΔT [K]": s_["dT_centre_K"],
         "Herkunft / Stand": "eigener Ansatz Andy, nicht validiert, FEM empfohlen"},
        {"Modell": "C Netzwerk: heißeste Zelle", "ΔT [K]": s_["dT_max_K"], "Herkunft / Stand": "wie C"},
        {"Modell": "C Netzwerk: Randzellen (Mittel)", "ΔT [K]": s_["dT_outer_mean_K"], "Herkunft / Stand": "wie C"},
    ]
    if True:
        st.dataframe(pd.DataFrame(cmp_rows).round(2), hide_index=True, width="stretch")
        st.caption("Einzelzelle und Block geben eine Temperatur für alle Zellen (kein Gradient). Das Netzwerk "
                   "verteilt die Wärme über Nachbarn und lokale Luft. Der Gradient im Netzwerk kommt vor allem aus "
                   "der Mittelungsregel der lokalen Luft (Gleichgewichtung, nicht gemessen), weil R_gap etwa zehnmal "
                   "größer ist als R_air.")
    st.markdown("##### Vergleich mit einer eigenen Messung")
    if True:
        st.caption("Messwerte eintragen. Für einen belastbaren Vergleich müssen Messorte, Zeitpunkt, Zellanzahl, "
                   "Verlustleistung je Zelle und Umgebung zum Modell passen.")
        c = st.columns(4)
        m_in = c[0].number_input("ΔT innen gemessen [K]", value=0.0, key=W.key("meas_in"))
        m_out = c[1].number_input("ΔT außen gemessen [K]", value=0.0, key=W.key("meas_out"))
        if m_in or m_out:
            c[2].metric("Abweichung innen", f"{s_['dT_centre_K'] - m_in:+.1f} K")
            c[3].metric("Abweichung außen", f"{s_['dT_outer_mean_K'] - m_out:+.1f} K",
                        f"Gradient Modell {s_['dT_centre_outer_K']:.1f} K / Messung {m_in - m_out:.1f} K",
                        delta_color="off", delta_arrow="off")


@st.cache_data(show_spinner=False, max_entries=32)
def _ideal(arr: str, nl: int, Pc: float, sg: float, rad: bool, eps: float, lam: float, T_inf: float):
    p = TN.NetworkParams(s_mm=sg, radiation=rad, eps=eps, lam=lam, T_inf_C=T_inf)
    net = TN.steady_state(TN.ring_grid(arr, nl, p.d_mm + sg), p, Pc)
    rows, conv, nconv = TN.layer_study(arr, p, Pc)
    t, T = TN.transient(net.grid, p, Pc, t_end_s=7200, n_points=150)
    return net, rows, conv, nconv, t, T


@st.cache_data(show_spinner=False, max_entries=32)
def _net_transient(_net, key: tuple, t_end_s: float):
    return TN.transient(_net.grid, _net.params, _net.P_cell_W, t_end_s=t_end_s, n_points=200)


def network_ideal_section():
    """Andy's demo: ideal square or hexagonal arrangement grown in layers around a centre cell."""
    o, r = inp.opt, inp.req
    if True:
        c = st.columns(5)
        arr = c[0].selectbox("Anordnung", ["hex", "square"], format_func={"hex": "hexagonal (z = 6)",
                                                                           "square": "quadratisch (z = 4)"}.get,
                             key=W.key("ni_arr"))
        nl = int(c[1].number_input("Nachbarschichten", 0, 12, 3, key=W.key("ni_n")))
        Pc = c[2].number_input("Verlust je Zelle [W]", 0.0, 10.0, 0.7, 0.05, key=W.key("ni_p"))
        sg = c[3].number_input("kleinster Abstand s [mm]", 0.0, 20.0, 0.0, 0.1, key=W.key("ni_s"))
        rad = c[4].checkbox("Strahlung", value=True, key=W.key("ni_rad"))
        net, rows, conv, nconv, t, T = _ideal(arr, nl, float(Pc), float(sg), bool(rad), float(o.net_eps),
                                              float(o.net_lambda), float(r.T_amb_C))
        p = net.params
        s_ = net.summary()
        c = st.columns(5)
        c[0].metric("Zellen / P gesamt", f"{s_['N']}", f"{s_['P_total_W']:.1f} W", delta_color="off", delta_arrow="off")
        c[1].metric("T Mitte", f"{s_['T_centre_C']:.1f} °C", f"Luft {s_['T_air_centre_C']:.1f} °C", delta_color="off", delta_arrow="off")
        c[2].metric("T außen (Mittel)", f"{s_['T_outer_mean_C']:.1f} °C", f"Luft {s_['T_air_outer_mean_C']:.1f} °C",
                    delta_color="off", delta_arrow="off")
        c[3].metric("Mitte - außen", f"{s_['dT_centre_outer_K']:.1f} K")
        c[4].metric("R_gap / Strahlung", f"{s_['R_gap_KperW']:.1f} K/W", f"{s_['Q_rad_total_W'] * 1000:.0f} mW",
                    delta_color="off", delta_arrow="off")
        if conv:
            st.success(f"Schichtkonvergenz: |ΔT_Mitte| < 0,05 K bei zwei Erweiterungen nach {nconv} Schichten.")
        else:
            ui.warn(f"Schichtkonvergenz bis {nconv} Schichten nicht erreicht (keine Aussage über die stationäre "
                    f"Rechnung selbst).")
        c1, c2, c3 = st.columns(3)
        ui.show_fig(pl.fig_network_heatmap(net), f"network_ideal_{arr}_{nl}", container=c1, plt_module=pl.plt,
                    fx_keys=["net", "R_gap", "Q_rad"])
        ui.show_fig(pl.fig_network_layers(rows, r.dT_max_K), f"network_layers_{arr}", container=c2, plt_module=pl.plt)
        ui.show_fig(pl.fig_network_transient(t, T, net.grid, p.T_inf_C, r.dT_max_K), f"network_ideal_heating_{arr}",
                    container=c3, plt_module=pl.plt)
        st.dataframe(pd.DataFrame([{"Schichten": x["layers"], "Zellen": x["N"], "T Mitte [°C]": x["T_centre_C"],
                                    "T außen [°C]": x["T_outer_mean_C"], "Änderung Mitte [K]": x["delta_centre_K"]}
                                   for x in rows]).round(3), hide_index=True, width="stretch")


RANK_COLS = ["rank", "cell_id", "setup", "score", "cost_EUR", "mass_kg", "volume_dm3", "density_kg_dm3",
             "P_loss_avg_W", "dT_max_K", "dT_net_max_K", "n_cells", "P_driver", "C_reserve", "C_reserve_BOL",
             "C_source"]


def page_tradeoff():
    ui.step_header("SCHRITT 12", "Bewertung und Auswahl",
                   "Das günstigste Modul ist nicht automatisch das beste. Alle zulässigen Konfigurationen werden je "
                   "Kriterium von 0 bis 100 % bewertet (100 % = bester Kandidat) und gewichtet gemittelt.")
    o = inp.opt
    with st.container(border=True):
        keys = list(WEIGHT_PRESETS)
        cur = o.weight_preset if o.weight_preset in keys else "balanced"
        wp = st.segmented_control("Gewichtung", keys, default=cur, format_func=lambda k: WEIGHT_PRESETS[k][0],
                                  key=W.key("wpreset"))
        o.weight_preset = wp or cur
        base = preset_weights(o.weight_preset) or o.weights
        if o.weight_preset == "custom":
            c = st.columns(4)
            for i, k in enumerate(CRITERIA):
                o.weights[k] = c[i % 4].slider(CRITERIA[k][2], 0.0, 3.0, float(o.weights.get(k, 1.0)), 0.25,
                                               key=W.key(f"w_{k}"))
            base = o.weights
        else:
            st.caption("Gewichte: " + ", ".join(f"{CRITERIA[k][2]} {v:g}" for k, v in base.items() if v))
    res = compute()
    for sk, df in res.candidates.items():
        sc = res.scenarios[sk]
        feas = df[df.feasible].sort_values("score", ascending=False)
        st.markdown(f"#### {sc.label}")
        best = feas.iloc[0].cell_id if not feas.empty else None
        cur_sel = selected_cell(sk)
        c = st.columns([1, 1, 1.4])
        c[0].metric("Empfehlung (beste Bewertung)", ui.short(best) if best else "-",
                    f"{feas.iloc[0].score:.0f} %" if best else None, delta_color="off", delta_arrow="off")
        pk = res.picks.get(sk, {}).get("min_cost")
        c[1].metric("Günstigste", ui.short(pk["cell_id"]) if pk else "-")
        with c[2]:
            opts = list(feas.cell_id)
            if cur_sel and cur_sel not in opts and cur_sel in set(df.cell_id):
                opts = [cur_sel] + opts
            idx = opts.index(cur_sel) if cur_sel in opts else None
            kk = W.key(f"choice_{sk}")
            ka = W.key(f"choice_all_{sk}")
            st.selectbox("Gewählte Zelle (gilt sofort)", opts, index=idx, format_func=ui.short, key=kk,
                         placeholder="Zelle wählen", on_change=_select_cell, args=(sk, kk, ka))
            b1, b2 = st.columns(2)
            b1.checkbox("für alle Module", value=True, key=ka,
                        help="Die Auswahl gilt für Modul 1 und Modul 2 (wie in der Diss eine Zelle für beide).")
            if cur_sel and b2.button("Auswahl entfernen", key=W.key(f"drop_{sk}"), width="stretch"):
                inp.selected.pop(sk, None)
                st.session_state.pop(kk, None)
                st.rerun()
            if cur_sel:
                st.caption(f"Gewählt: **{ui.short(cur_sel)}**")
            else:
                ui.warn("Noch keine Zelle gewählt.")
        ui.table(feas, RANK_COLS, key=W.key(f"rank_{sk}"), fx_keys=["score", "cost", "mass", "P_loss", "reserve"])
        s, contrib = score(feas, base, o.reserve_cap)
        ranked = feas.assign(score=s).sort_values("score", ascending=False)
        ui.show_fig(pl.fig_scores(ranked, contrib), f"scores_{sk}", plt_module=pl.plt, fx_keys=["score"])
    if res.candidates:
        st.markdown("#### Übersicht wie IECON 2023 (Kosten und Moduldichte über Zellkapazität, Farbe = Modul-ESR, "
                    "grau = außerhalb der Vorgabe)")
        QN = {"cost_EUR": "Kosten", "volume_dm3": "Modulvolumen", "pcb_area_cm2": "Platinenfläche",
              "density_kg_dm3": "Moduldichte", "mass_kg": "Modulmasse", "rel_volume_dm3": "Volumen zur Referenz [%]",
              "rel_pcb_area_cm2": "Platinenfläche zur Referenz [%]", "rel_density_kg_dm3": "Dichte zur Referenz [%]",
              "rel_mass_kg": "Masse zur Referenz [%]", "rel_cost_EUR": "Kosten zur Referenz [%]"}
        with st.container(border=True):
            q = list(QN)
            c = st.columns(5)
            yt = c[0].selectbox("Oben: y-Achse", q, index=0, format_func=QN.get, key=W.key("mc_yt"))
            st_ = c[1].selectbox("Oben: Kugelgröße", q, index=q.index("volume_dm3"), format_func=QN.get, key=W.key("mc_st"))
            yb = c[2].selectbox("Unten: y-Achse", q, index=q.index("volume_dm3"), format_func=QN.get, key=W.key("mc_yb"))
            sb = c[3].selectbox("Unten: Kugelgröße", q, index=0, format_func=QN.get, key=W.key("mc_sb"))
            refs = {"selected": "gewähltes Modul (Schritt 7)", "space": "Bauraumvorgabe", "custom": "eigene Werte"}
            ref = c[4].selectbox("Referenz für %", list(refs), format_func=refs.get, key=W.key("mc_ref"))
            ref_vals = {}
            if ref == "custom":
                cc = st.columns(5)
                ref_vals = {"volume_dm3": cc[0].number_input("Volumen [dm³]", value=1.0, key=W.key("mcr_v")),
                            "pcb_area_cm2": cc[1].number_input("Fläche [cm²]", value=300.0, key=W.key("mcr_a")),
                            "density_kg_dm3": cc[2].number_input("Dichte [kg/dm³]", value=1.0, key=W.key("mcr_d")),
                            "mass_kg": cc[3].number_input("Masse [kg]", value=1.0, key=W.key("mcr_m")),
                            "cost_EUR": cc[4].number_input("Kosten [€]", value=250.0, key=W.key("mcr_c"))}
        cands = {}
        missing_ref = []
        for sk_, df_ in res.candidates.items():
            d_ = df_.copy()
            if ref == "selected":
                cid_ = selected_cell(sk_)
                rr = d_[d_.cell_id == cid_]
                if rr.empty and len(d_[d_.feasible]):
                    rr = d_[d_.feasible].sort_values("score", ascending=False).head(1)
                rv = {k: float(rr.iloc[0][k]) for k in ("volume_dm3", "pcb_area_cm2", "density_kg_dm3", "mass_kg",
                                                         "cost_EUR")} if len(rr) else {}
            elif ref == "space":
                r_ = inp.req
                rv = {}
                if r_.max_width_mm and r_.max_depth_mm:
                    rv["pcb_area_cm2"] = r_.max_width_mm * r_.max_depth_mm / 100
                    if r_.max_height_mm:
                        rv["volume_dm3"] = r_.max_width_mm * r_.max_depth_mm * r_.max_height_mm / 1e6
                if r_.max_volume_dm3:
                    rv["volume_dm3"] = r_.max_volume_dm3
                if r_.max_mass_kg:
                    rv["mass_kg"] = r_.max_mass_kg
                if r_.max_cost_EUR:
                    rv["cost_EUR"] = r_.max_cost_EUR
                if "mass_kg" in rv and "volume_dm3" in rv:
                    rv["density_kg_dm3"] = rv["mass_kg"] / rv["volume_dm3"]
            else:
                rv = ref_vals
            for k in ("volume_dm3", "pcb_area_cm2", "density_kg_dm3", "mass_kg", "cost_EUR"):
                if rv.get(k):
                    d_[f"rel_{k}"] = 100 * (d_[k] / rv[k] - 1)
                else:
                    d_[f"rel_{k}"] = np.nan
                    if any(x == f"rel_{k}" for x in (yt, st_, yb, sb)):
                        missing_ref.append(QN[f"rel_{k}"])
            cands[sk_] = d_
        if missing_ref:
            ui.warn("Keine Referenz für: " + ", ".join(sorted(set(missing_ref))) + ". Referenz setzen (Bauraum auf "
                    "Seite 1, Auswahl in Schritt 7 oder eigene Werte).")
        c1, c2 = st.columns([1, 1])
        ui.show_fig(pl.fig_tradeoff(res, scenarios=tuple(res.candidates), y_top=yt, y_bottom=yb, size_top=st_,
                                    size_bottom=sb, cands=cands),
                    f"Modules_Cells_{yt}_{yb}", container=c1, plt_module=pl.plt, fx_keys=["cost", "mass", "layout"])
        with c2:
            axes_opts = {"cost_EUR": "Kosten [€]", "mass_kg": "Masse [kg]", "volume_dm3": "Volumen [dm³]",
                         "ESR_module_mOhm": "ESR [mΩ]", "P_loss_avg_W": "Verluste [W]", "dT_max_K": "ΔT [K]",
                         "C_module_F": "C Modul [F]", "c_nom_F": "C Zelle [F]", "score": "Bewertung [%]"}
            xa = st.selectbox("x", list(axes_opts), index=0, format_func=axes_opts.get, key=W.key("px"))
            ya = st.selectbox("y", list(axes_opts), index=4, format_func=axes_opts.get, key=W.key("py"))
            fig = go.Figure()
            for i, (sk, df) in enumerate(res.candidates.items()):
                nb = df[~df.feasible]
                if len(nb):
                    fig.add_trace(go.Scatter(x=nb[xa], y=nb[ya], mode="markers", name=f"Modul {i + 1} außerhalb Vorgabe",
                                             marker=dict(size=9, color="#d4d4d4", line=dict(width=1, color="#9a9a9a")),
                                             customdata=nb[["cell_id", "setup", "status"]],
                                             hovertemplate="%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}"))
                d = df[df.feasible]
                fig.add_trace(go.Scatter(x=d[xa], y=d[ya], mode="markers+text", name=f"Modul {i + 1}",
                                         text=d.c_nom_F.astype(int).astype(str) + " F", textposition="top center",
                                         marker=dict(size=np.where(d.cell_id == selected_cell(sk), 16, 10),
                                                     color=COLORS[i], line=dict(width=np.where(d.pareto, 2, 0),
                                                                                color="black")),
                                         customdata=d[["cell_id", "setup"]],
                                         hovertemplate="%{customdata[0]} %{customdata[1]}<br>%{x:.3g} / %{y:.3g}"))
            fig.update_xaxes(title_text=axes_opts[xa])
            fig.update_yaxes(title_text=axes_opts[ya])
            st.plotly_chart(plotly_layout(fig, 430), width="stretch")
            st.caption("Groß = gewählt, schwarzer Rand = Pareto-optimal (Kosten, Masse, Volumen, ESR), "
                       "grau = außerhalb der Vorgabe.")
        if inp.opt.geometry == "legacy_iecon":
            cmp_ = compare_with_reference(res, db)
            if not cmp_.empty:
                with st.expander(f"Vergleich mit der IECON-2023-Tabelle: {int(cmp_.match.sum())}/{len(cmp_)} Werte"):
                    st.dataframe(cmp_, hide_index=True, width="stretch")
        with st.expander("LaTeX-Tabelle (Layout wie Tab. results)"):
            st.code(latex_results_table(res, tuple(res.candidates)), language="latex")
    nav_buttons("tradeoff")


def page_emulation():
    ui.step_header("THERMISCHE SIMULATION", "Emulation: Prüfung des Moduls aus dem Flussdiagramm",
                   "P und S bleiben wie in Schritt 5 bis 11, die Emulation ändert nichts. Das Profil läuft einmal ab "
                   "U_C0inM (vorgeladen), Aufteilung wie in Schritt 2 (der Split lädt den SC nach, kein eigener "
                   "Nachladeregler). Je Zeitschritt: Wandler-Wirkungsgrad, I aus P und Klemmenspannung U_C - R I, "
                   "ESR-Verluste, Stromgrenze I_C0max und Spannungsgrenzen (dort liefert die Batterie den Rest), "
                   "Zelltemperatur mit demselben Modell wie Seite 6.")
    res = compute()
    if res.profile is None or not res.candidates:
        st.info("Für die Emulation werden ein Lastprofil und Kandidaten benötigt.")
        return
    keys_ = list(res.candidates)
    with_sel = [k for k in keys_ if selected_cell(k)]
    sk = st.selectbox("Szenario", keys_, index=keys_.index(with_sel[-1]) if with_sel else len(keys_) - 1,
                      format_func=lambda s: SCENARIOS[s], key=W.key("emu_scen"))
    dfa = res.candidates[sk]
    opts = list(dfa[dfa.feasible].cell_id)
    sel = selected_cell(sk)
    if sel and sel not in opts and sel in set(dfa.cell_id):
        opts = [sel] + opts                      # keep the selection even if it is outside the specification
        ui.warn(f"Die gewählte Zelle {ui.short(sel)} liegt außerhalb der Vorgabe ({dfa.set_index('cell_id').loc[sel].status}).")
    main, src = chosen_cell(res, sk, opts)
    if src != "gewählt in Schritt 7":
        ui.warn(f"Für {SCENARIOS[sk]} ist in Schritt 7 keine Zelle gewählt. Verwendet: {ui.short(main)} ({src}).")
    with st.container(border=True):
        c = st.columns([1.3, 1.7])
        cells = c[0].multiselect("Zellen (die gewählte zuerst, weitere zum Vergleich)", opts, default=[main],
                                 format_func=ui.short, key=W.key(f"emu_cells_{sk}_{main}"))
        c[1].caption(f"Verwendet: **{ui.short(main)}** ({src}). Weitere Zellen nur zum Vergleich.")
        c = st.columns(5)
        eta_d = c[0].number_input("η Entladen [%]", 50.0, 100.0, float(inp.req.eta_dch * 100), 0.5,
                                  key=W.key("emu_etad"))
        eta_c = c[1].number_input("η Laden [%]", 50.0, 100.0, float(inp.req.eta_ch * 100), 0.5, key=W.key("emu_etac"))
        eol = c[2].toggle("Prüfung am Lebensende", value=True, key=W.key("emu_eol"),
                          help="C x L_EOL und ESR x 2. Das ist der Zustand, für den C_req ausgelegt ist.")
        brk = c[3].toggle("Bremsereignis einfügen", value=sk == "profile_braking" and inp.braking.enabled,
                          key=W.key("emu_brk"), help="Zum ungünstigsten Zeitpunkt (maximale SC-Energie).")
        dtb = c[4].number_input("Zeitschritt [s]", 0.001, 0.1, 0.01, 0.001, format="%.3f", key=W.key("emu_dt"))
        c = st.columns([1, 1.2, 1, 1.4])
        th_on = c[0].toggle("Thermik anzeigen", value=True, key=W.key("emu_th"),
                            help="Thermik-Abbildungen ein/aus. Gerechnet wird mit demselben Modell wie auf Seite 6 "
                                 "(Netzwerk, wenn auf der Startseite aktiv, sonst Blockmodell).")
        cons = c[1].toggle("Busspannung konservativ (U_Bus,min)", value=inp.req.U_bus_min_V is not None,
                           key=W.key("emu_cons"), help="Busströme mit der niedrigsten Bus-/Batteriespannung statt "
                                                       "der nominalen U_Bus.")
        if cons:
            if inp.req.U_bus_min_V is None:
                inp.req.U_bus_min_V = round(0.85 * inp.req.U_bus_V, 1)
            W.num("U_Bus,min [V]", inp.req, "U_bus_min_V", "ubusmin", step=0.5, container=c[2])
        else:
            inp.req.U_bus_min_V = None
        u_bus_used = inp.req.U_bus_min_V if cons and inp.req.U_bus_min_V else inp.req.U_bus_V
        c[3].caption(f"Busströme mit U = {u_bus_used:g} V. Thermik: "
                     f"{'Netzwerk (heißeste Zelle)' if inp.opt.thermal_network else 'Blockmodell'}, ESR x "
                     f"{inp.opt.thermal_esr_factor:g} wie auf Seite 6.")
    cfg_key = json.dumps(dict(sk=sk, cells=cells, eta=(eta_d, eta_c), eol=eol, brk=brk, dt=dtb, ub=u_bus_used,
                              inp=yaml.safe_dump({**inp.to_dict(), "selected": {}}, sort_keys=True)), sort_keys=True)
    if st.button("Emulation rechnen (Prüfung bei P aus Schritt 5)", type="primary"):
        out = {}
        with st.spinner("Emuliere ..."):
            for cid in cells:
                cfg = EmulationConfig(dt_bin_s=dtb, soc_control=False, eta_dch=eta_d / 100, eta_ch=eta_c / 100,
                                      U_conv_min_V=inp.req.U_conv_min_V, U_bus_V=u_bus_used)
                out[cid] = emulation_design(res, sk, cid, cfg, check_eol=eol, include_braking=brk)
        st.session_state["emu_result"] = (cfg_key, out)
    stored = st.session_state.get("emu_result")
    if not stored:
        st.info("Einstellungen prüfen und auf 'Emulation rechnen' klicken.")
        return
    if stored[0] != cfg_key:
        ui.warn("Einstellungen geändert: Ergebnisse unten gehören zu den vorherigen Einstellungen. Neu rechnen.")
    out = stored[1]
    sc = res.scenarios[sk]
    rows = []
    for cid, d in out.items():
        last = d.iterations[-1]
        crow = res.candidates[sk].set_index("cell_id").loc[cid]
        cell = db.cell(cid)
        n = d.S * d.P_final
        lay = layout_for(cell, d.S, d.P_final, inp.opt, inp.req)
        m_mod = n * cell["mass_g"] / 1000 + (lay.pcb_mass_kg(inp.opt.pcb_areal_mass_g_cm2) if lay else 0.0)
        rows.append({"Zelle": ui.short(cid), "P/S (Schritt 5)": f"{d.P_final}/{d.S}",
                     "Zellen": n, "Kosten [€]": n * cell["price_EUR"], "Masse [kg]": m_mod,
                     "Modulvolumen [dm³]": lay.volume_dm3 if lay else float("nan"),
                     "Platine [mm]": f"{lay.width_mm:.0f} x {lay.depth_mm:.0f}" if lay else "",
                     "C Modul [F]": last.C_module_F, "U min / max [V]": f"{last.U_op_min_V:.1f} / {last.U_op_max_V:.1f}",
                     "t an U_C0min [s]": last.t_at_Umin_s, "Batterie übernimmt [J]": last.E_overflow_J,
                     "ESR-Verlust [J]": last.E_loss_esr_J, "I_peak [A]": last.I_peak_A,
                     "I_Zelle RMS [A]": last.I_rms_cell_A, "ΔT Block [K]": last.dT_block_max_K,
                     "ΔT Netzwerk heißeste [K]": last.dT_net_max_K,
                     "Ergebnis": "erfüllt" if d.feasible else last.reason})
    st.markdown("#### Ergebnis (Prüfung, P unverändert)")
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    for cid, d in out.items():
        if not d.feasible:
            ui.warn(f"{ui.short(cid)} {d.P_final}/{d.S}: {d.iterations[-1].reason}. Hinweis, P wird nicht geändert.")
    st.latex(r"P_\mathrm{SC,Zelle}=\begin{cases}P_\mathrm{SC,Bus}/\eta_\mathrm{dch}\\P_\mathrm{SC,Bus}\,"
             r"\eta_\mathrm{ch}\end{cases},\quad I_\mathrm{SC}=\frac{U_\mathrm{C}-\sqrt{U_\mathrm{C}^2-4RP_"
             r"\mathrm{SC,Zelle}}}{2R},\quad \frac{\mathrm{d}E_\mathrm{SC}}{\mathrm{d}t}=-\left(P_\mathrm{SC,Zelle}"
             r"+RI_\mathrm{SC}^2\right),\quad E_\mathrm{SC}(0)=\tfrac{1}{2}C\,U_\mathrm{C0inM}^2")
    st.caption("An U_C0min bzw. U_C0max fließt kein weiterer SC-Strom, die Batterie liefert bzw. nimmt den Rest "
               "(wie bei der Stromgrenze I_C0max). Der Split sieht nur die Last, die ESR-Verluste des SC werden "
               "dadurch nicht nachgeladen.")
    for cid, d in out.items():
        with st.expander(f"{ui.short(cid)}: emuliertes Profil und Rechnung", expanded=cid == cells[0]):
            for n in d.notes:
                st.caption(n)
            it = pd.DataFrame([asdict(i) for i in d.iterations]).drop(
                columns=["C_req_emu_F", "P_C_emu", "P_I_emu", "cycles", "ok"], errors="ignore")
            it = it.rename(columns={"P": "P", "C_module_F": "C Modul [F]", "C_module_eol_F": "C Modul EOL [F]",
                                    "U_op_min_V": "U min [V]", "U_op_max_V": "U max [V]",
                                    "E_swing_J": "Energiehub [J]", "C_req_emu_F": "C_req,emu [F]",
                                    "P_C_emu": "P_C,emu", "I_peak_A": "I_peak [A]", "P_I_emu": "P_I,emu",
                                    "I_rms_module_A": "I_Modul RMS [A]", "I_rms_cell_A": "I_Zelle RMS [A]",
                                    "dT_max_K": "ΔT Kriterium [K]", "dT_block_max_K": "ΔT Block [K]",
                                    "dT_net_max_K": "ΔT Netzwerk heißeste [K]", "t_at_Umin_s": "t an U_min [s]",
                                    "t_at_Umax_s": "t an U_max [s]", "t_current_limit_s": "t an I_C0max (Batterie) [s]",
                                    "E_overflow_J": "Batterie übernimmt [J]", "E_loss_esr_J": "ESR-Verlust [J]",
                                    "E_loss_conv_J": "Wandlerverlust [J]", "cycles": "Zyklen", "ok": "ok",
                                    "reason": "Begründung"})
            st.dataframe(it.round(2), hide_index=True, width="stretch")
            if d.iterations:
                it0 = d.iterations[0]
                trow = res.candidates[sk].set_index("cell_id").loc[cid]
                ref_net = trow.get("dT_net_max_K", float("nan"))
                st.caption(f"Abgleich bei P = {it0.P} (analytisch): Seite 6 Netzwerk {ui.fmt(ref_net, 1, 'K')}, Block "
                           f"{ui.fmt(trow.dT_max_K, 1, 'K')} | Emulation Netzwerk {ui.fmt(it0.dT_net_max_K, 1, 'K')}, "
                           f"Block {ui.fmt(it0.dT_block_max_K, 1, 'K')}. Unterschied nur durch den emulierten "
                           f"Strom (Spannungsverlauf, kein SC-Strom an U_C0min).")
            ui.show_fig(pl.fig_emulated_profile(d.trace, d.module, sc, i_limit=d.module.I_max_A,
                                                u_conv_min=inp.req.U_conv_min_V, u_bus=inp.req.U_bus_V,
                                                u_bus_used=u_bus_used, thermal=th_on, net=d.net),
                        f"emulated_profile_{cid}_{d.P_final}", plt_module=pl.plt,
                        fx_keys=["eta", "I_sc", "P_limit", "U_lim", "U_term", "th_first"])
            if d.chain is not None and th_on:
                ch = d.chain
                st.markdown("**Zelltemperatur aus dem Zellstrom** (volle Auflösung, je Zeitschritt)")
                st.latex(r"P_\mathrm{Bus}=U_\mathrm{Bus}I_\mathrm{Bus}=U_\mathrm{SC}I_\mathrm{SC},\quad "
                         r"U_\mathrm{SC}=U_\mathrm{C}-R_\mathrm{ESR}I_\mathrm{SC}\;\Rightarrow\;"
                         r"I_\mathrm{SC}=\frac{U_\mathrm{C}-\sqrt{U_\mathrm{C}^2-4R_\mathrm{ESR}P}}{2R_\mathrm{ESR}},"
                         r"\quad P\le\frac{U_\mathrm{C}^2}{4R_\mathrm{ESR}}")
                st.latex(r"i_\mathrm{Zelle}=\frac{I_\mathrm{SC}}{P},\quad p_\mathrm{v}(t)=i_\mathrm{Zelle}^2(t)\,"
                         r"R_\mathrm{ESR,Zelle},\quad C_\mathrm{th}\frac{\mathrm{d}\Delta T}{\mathrm{d}t}="
                         r"p_\mathrm{v}-\frac{\Delta T}{R_\mathrm{th,eff}}")
                m = st.columns(5)
                m[0].metric("I Zelle RMS", f"{ch['I_rms']:.2f} A")
                m[1].metric("ESR Zelle (Verluste)", f"{d.R_esr_cell_used * 1000:.1f} mΩ")
                m[2].metric("p_v Mittel", f"{ch['P_v_mean']:.2f} W")
                m[3].metric("R_th,eff / τ", f"{d.module.R_th:.1f} K/W / {ch['tau_s']:.0f} s")
                m[4].metric("ΔT Block max", f"{ch['dT_max']:.1f} K", f"stationär {ch['dT_ss']:.1f} K", delta_color="off",
                            delta_arrow="off")
                st.caption(f"Kontrolle: I_RMS² x ESR x R_th,eff = {ch['I_rms']:.2f}² x {d.R_esr_cell_used:.4f} Ω x "
                           f"{d.module.R_th:.1f} K/W = {ch['I_rms'] ** 2 * d.R_esr_cell_used * d.module.R_th:.1f} K "
                           f"(Mittelwert), Maximum durch die Welligkeit {ch['dT_max']:.1f} K.")
                ui.show_fig(pl.fig_cell_thermal_chain(ch, inp.req.dT_max_K), f"cell_thermal_chain_{cid}_{d.P_final}",
                            plt_module=pl.plt, fx_keys=["I_sc", "th_first", "R_th_eff", "esr_eol"])
                if d.net is not None:
                    nt = d.net
                    st.markdown(f"**Netzwerk (wie Seite 6) mit dem emulierten Verlustprofil**, P = {d.P_final}: "
                                f"p_v(t) = i(t)² ESR je Zelle, periodischer Zustand")
                    m = st.columns(4)
                    ui.metric(m[0], "Heißeste Zelle max", f"{nt['dT_max']:.1f} K", key="net")
                    ui.metric(m[1], "Mitte max", f"{nt['dT_centre_max']:.1f} K", key="net")
                    ui.metric(m[2], "Randzellen (Mittel)", f"{nt['dT_outer_mean']:.1f} K", key="net")
                    ui.metric(m[3], "p_v Mittel je Zelle", f"{nt['P_mean'] * 1000:.0f} mW", key="th_first")
                    ui.show_fig(pl.fig_network_heatmap(nt["steady"], extent=layout_extent(res)),
                                f"emu_network_{ui.short(cid).replace(' ', '_')}_{d.P_final}", plt_module=pl.plt,
                                fx_keys=["net", "R_gap", "Q_rad"])
            csv = io.StringIO()
            pd.DataFrame({"t_s": d.trace.t - d.trace.t[0], "P_load_W": d.trace.P_load, "P_bat_W": d.trace.P_bat,
                          "P_sc_bus_W": d.trace.P_sc, "P_sc_cell_W": d.trace.P_sc_cell, "I_sc_A": d.trace.I_sc,
                          "I_sc_rms_A": d.trace.I_sc_rms, "U_sc_V": d.trace.U_sc, "E_sc_J": d.trace.E_sc,
                          "dT_K": d.trace.dT}).to_csv(csv, index=False)
            st.download_button("Emuliertes Profil als CSV", csv.getvalue(), file_name=f"emulated_{cid}.csv",
                               key=W.key(f"emu_csv_{cid}"))
    live_section(out[cells[0]], sc)
    nav_buttons("emulation")


def live_section(d, sc):
    st.markdown("#### Live-Ansicht (Endlosschleife)")
    st.caption("Profil wiederholt, Aufteilung wie Schritt 2, kein Nachladeregler.")
    prof = compute().profile
    res = compute()
    c = st.columns(4)
    live = c[0].toggle("Live starten", key="live_on")
    c[1].select_slider("Profilzeit je Aktualisierung [s]", [1, 2, 5, 10, 20, 50], value=5, key="live_speed")
    c[2].select_slider("Fenster [s]", [20, 60, 150, 300, 600], value=60, key="live_win")
    if c[3].button("Zurücksetzen"):
        st.session_state.pop("live_emu", None)
        st.session_state.pop("live_buf", None)
    if live and "live_emu" not in st.session_state:
        em = HESSEmulator(prof.t, prof.p, res.p_bat, d.module,
                          EmulationConfig(eta_dch=inp.req.eta_dch, eta_ch=inp.req.eta_ch, soc_control=False))
        if d.chain:
            em.dT = float(d.chain["dT0"])          # start in the thermal periodic steady state
        st.session_state["live_emu"] = em
        st.session_state["live_buf"] = None

    def live_view():
        em = st.session_state.get("live_emu")
        if not st.session_state.get("live_on") or em is None:
            st.caption("Gestoppt.")
            return
        tr = em._advance(max(1, int(round(st.session_state.get("live_speed", 5) / em.dt))))
        new = np.vstack([tr.t, tr.P_load, tr.P_bat, tr.P_sc, tr.U_sc, tr.dT, tr.I_sc])
        buf = st.session_state.get("live_buf")
        buf = new if buf is None else np.hstack([buf, new])
        buf = buf[:, buf[0] >= buf[0, -1] - st.session_state.get("live_win", 60)]
        st.session_state["live_buf"] = buf
        f = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04)
        f.add_trace(go.Scattergl(x=buf[0], y=buf[1], name="Last", line=dict(color="#999", width=1)), 1, 1)
        f.add_trace(go.Scattergl(x=buf[0], y=buf[2], name="Batterie", line=dict(color=COLORS[1], width=2)), 1, 1)
        f.add_trace(go.Scattergl(x=buf[0], y=buf[3], name="Supercap", line=dict(color=COLORS[0], width=1)), 1, 1)
        f.add_trace(go.Scattergl(x=buf[0], y=buf[6], name="I_SC", line=dict(color=COLORS[0], width=1)), 2, 1)
        f.add_trace(go.Scattergl(x=buf[0], y=buf[4], name="U_SC", line=dict(color=COLORS[0])), 3, 1)
        f.add_hline(y=em.m.U_min, line_dash="dash", row=3, col=1)
        f.add_hline(y=em.m.U_max, line_dash="dash", row=3, col=1)
        f.add_trace(go.Scattergl(x=buf[0], y=buf[5], name="ΔT", line=dict(color=COLORS[2])), 4, 1)
        for i, t in enumerate(["Power [W]", "Current [A]", "Voltage [V]", "ΔT [K]"]):
            f.update_yaxes(title_text=t, row=i + 1, col=1)
        f.update_yaxes(range=[em.m.U_min - 2, em.m.U_max + 2], row=3, col=1)
        f.update_xaxes(title_text="Time [s]", row=4, col=1)
        plotly_layout(f, 640).update_layout(uirevision="live")
        st.caption(f"Zyklus {em.cycle + 1}, Emulationszeit {buf[0, -1]:.1f} s, U = {buf[4, -1]:.2f} V, "
                   f"ΔT = {buf[5, -1]:.2f} K")
        st.plotly_chart(f, width="stretch", key="live_chart")

    st.fragment(run_every=1.0 if live else None)(live_view)()


def page_detail():
    ui.step_header("ZELLDETAIL", "Eine Zelle mit eigenen Optionen durchrechnen",
                   "Alle Optionen gelten nur für diese Ansicht und ändern das Gesamtdesign nicht.")
    res = compute()
    if not res.candidates:
        st.info("Keine Kandidaten.")
        return
    scen_keys = list(res.candidates)
    dsc = st.session_state.get("detail_scen", scen_keys[-1])
    sk = st.selectbox("Szenario", scen_keys, index=scen_keys.index(dsc) if dsc in scen_keys else 0,
                      format_func=lambda s: SCENARIOS[s], key=W.key(f"det_scen_{dsc}"))
    cells_all = list(db.cells.cell_id)
    dcell = st.session_state.get("detail_cell") or selected_cell(sk) or res.candidates[sk].cell_id.iloc[0]
    cid = st.selectbox("Zelle", cells_all, index=cells_all.index(dcell) if dcell in cells_all else 0,
                       format_func=ui.short, key=W.key(f"det_cell_{dcell}"))
    st.session_state["detail_cell"], st.session_state["detail_scen"] = cid, sk
    cell = db.cell(cid)
    miss = db.missing_fields(cell)
    if miss:
        st.error(f"Für {cid} fehlen Daten: {', '.join(miss)}. Bitte in der Datenbank ergänzen.")
        return
    o = copy.deepcopy(inp.opt)
    with st.container(border=True):
        c = st.columns(4)
        o.c_source = c[0].selectbox("Kapazität", list(C_SOURCES), index=list(C_SOURCES).index(o.c_source),
                                    format_func=C_SOURCES.get, key=f"det_csrc_{cid}")
        o.cfit_range = c[1].selectbox("Fenster C0/k", ["90-20", "80-40"], index=["90-20", "80-40"].index(o.cfit_range),
                                      key=f"det_crng_{cid}")
        fits = db.cfits[(db.cfits.cell_id == cid) & (db.cfits.voltage_range == o.cfit_range)]
        cur_choices = ["highest", "lowest"] + [f"{a:g}" for a in sorted(fits.current_A.unique())]
        cc = c[2].selectbox("Messstrom [A]", cur_choices, key=f"det_ccur_{cid}")
        o.cfit_current = cc if cc in ("highest", "lowest") else float(cc)
        o.cfit_dataset = c[3].selectbox("Datensatz", ["auto"] + sorted(fits.dataset.unique()), key=f"det_cds_{cid}")
        c = st.columns(4)
        o.esr_source = c[0].selectbox("ESR", list(ESR_SOURCES), index=list(ESR_SOURCES).index(o.esr_source),
                                      format_func=ESR_SOURCES.get, key=f"det_esr_{cid}")
        tmodes = list(THERMAL_MODES) + ["manual"]
        tm = c[1].selectbox("R_th / C_th", tmodes, index=tmodes.index(o.thermal_mode), key=f"det_tm_{cid}",
                            format_func=lambda m: {"manual": "eigene Werte"}.get(m, THERMAL_MODES.get(m, m)))
        override = None
        if tm == "manual":
            rth = c[2].number_input("R_th [K/W]", value=22.1, key=f"det_rth_{cid}")
            cth = c[3].number_input("C_th [J/K]", value=19.7, key=f"det_cth_{cid}")
            override = dict(rth=rth, cth=cth, note="manuell", warning="R_th/C_th manuell eingegeben.")
        else:
            o.thermal_mode = tm
        c = st.columns(4)
        o.thermal_coupling = c[0].selectbox("Wärmeabfuhr", list(COUPLING_MODES),
                                            index=list(COUPLING_MODES).index(o.thermal_coupling),
                                            key=f"det_coup_{cid}")
        o.thermal_esr_factor = c[1].selectbox("ESR für Verluste", [1.0, 2.0], index=[1.0, 2.0].index(
            o.thermal_esr_factor) if o.thermal_esr_factor in (1.0, 2.0) else 0,
            format_func={1.0: "neu (x1)", 2.0: "Lebensende (x2)"}.get, key=f"det_esrf_{cid}")
        pmin = c[2].number_input("P mindestens", 1, 500, 1, key=f"det_pmin_{cid}")
    sc = res.scenarios[sk]
    sid = o.series_override or cell["lifetime_series_id"]
    si = res.series_info.get(sid)
    row, det = evaluate_candidate(cell, sc, si["S"], si["U_C01max"], inp.req, o, db, thermal_override=override,
                                  P_min=int(pmin))
    c = st.columns(6)
    ui.metric(c[0], "P/S", row["setup"], row["P_driver"], key="P_driver",
              calc=f"P_C = {row['P_C']}, P_I = {row['P_I']}, P_T = {row['P_T']}")
    ui.metric(c[1], "C Modul", f"{row['C_module_F']:.2f} F", f"Reserve EOL {100 * row['C_reserve']:.0f} %", key="C_mod",
              calc=f"{row['P']}/{row['S']} x {row['C_cell_used_F']:.2f} F")
    ui.metric(c[2], "ESR Modul", f"{row['ESR_module_mOhm']:.0f} mΩ", key="C_mod",
              calc=f"{row['S']}/{row['P']} x {row['ESR_cell_mOhm']:.1f} mΩ")
    ui.metric(c[3], "Kosten", ui.fmt(row["cost_EUR"], 0, "€"), key="cost")
    ui.metric(c[4], "Masse / Volumen", ui.fmt(row["mass_kg"], 2, "kg"), ui.fmt(row["volume_dm3"], 2, "dm³"), key="mass")
    ui.metric(c[5], "ΔT max (Lumped / Netz)", ui.fmt(row["dT_max_K"], 1, "K"),
              ui.fmt(row.get("dT_net_max_K", float("nan")), 1, "K Netzwerk"), key="th_first")
    st.code("\n".join(det.log))
    for w in det.warnings:
        ui.warn(w)
    st.caption(f"Kapazität: {row['C_source']}  |  ESR: {row['ESR_source']}")
    c1, c2 = st.columns(2)
    if det.fit is not None:
        S = row["S"]
        ceq = eq.c_eq_energy(det.fit["C0_F"], det.fit["k_FperV"], sc.U_min / S, sc.U_max / S)
        ui.show_fig(pl.fig_cu(det.fit, cell, sc.U_min / S, sc.U_max / S, ceq), f"cu_{cid}", container=c1,
                    plt_module=pl.plt)
    if det.breakdown:
        ui.show_fig(pl.fig_thermal_breakdown(det.breakdown, inp.req.dT_max_K), f"breakdown_{cid}", container=c2,
                    plt_module=pl.plt)
    if o.geometry == "pcb":
        lay = layout_for(cell, int(row["S"]), int(row["P"]), o, inp.req)
        if lay is not None:
            m_cells = row["n_cells"] * float(cell["mass_g"]) / 1000
            ui.show_fig(pl.fig_pcb_layout(lay, ui.short(cid) + " " + row["setup"], m_cells, o.pcb_areal_mass_g_cm2,
                                          extent=layout_extent(res)),
                        f"pcb_{ui.short(cid).replace(' ', '_')}_{row['setup'].replace('/', 'x')}", plt_module=pl.plt)
    if not row["feasible"]:
        ui.warn(row["status"])
    if st.button("Diese Zelle als Auswahl übernehmen", type="primary"):
        inp.selected[sk] = {"cell_id": cid}
        st.success("Übernommen.")
    nav_buttons("detail")


def page_db():
    ui.step_header("DATENBANK", "Zellen und Messdaten",
                   "CSV-Dateien in data/database sind die Quelle, die SQLite-Datei wird automatisch gebaut.")
    tname = st.selectbox("Tabelle", list(TABLES), key="db_table")
    df = db.tables[tname]
    ed = st.data_editor(df, num_rows="dynamic", width="stretch", key=f"db_edit_{tname}")
    c1, c2 = st.columns([1, 3])
    if c1.button("Änderungen speichern", type="primary"):
        db.save_table(tname, ed)
        get_database(reload=True)
        cached_design.clear()
        st.success(f"{TABLES[tname]} gespeichert.")
        st.rerun()
    c2.caption("Neue Zelle: Zeile in 'cells', candidate = True. Pflicht: c_nom_F, u_rated_V, esr_dc_mOhm, i_max_A, "
               "lifetime_series_id. Für den Trade-off zusätzlich Masse, Maße, Preis.")
    with st.expander("SQL-Abfrage"):
        q = st.text_area("SQL", "SELECT cell_id, dataset, voltage_range, current_A, C0_F, k_FperV FROM "
                         "capacitance_fits WHERE voltage_range = '90-20' ORDER BY cell_id", key="sql")
        if st.button("Ausführen"):
            try:
                st.dataframe(db.query(q), width="stretch")
            except Exception as exc:
                st.error(str(exc))
    src = DB_DIR / "SOURCES.md"
    if src.exists():
        with st.expander("Quellen"):
            st.markdown(src.read_text(encoding="utf8"))
    nav_buttons("db")


def page_export():
    ui.step_header("EXPORT", "Ergebnisse und Abbildungen",
                   "Schreibt alles nach outputs/: Eingaben, Zusammenfassung, Kandidaten, LaTeX-Tabelle, Abbildungen.")
    res = compute()
    out_name = st.text_input("Ordner unter outputs/", f"{inp.name}", key="out_name")
    figs = st.checkbox("Abbildungen (PDF/SVG/PNG, Diss-Stil)", value=True, key="out_figs")
    if st.button("Exportieren", type="primary"):
        with st.spinner("Exportiere ..."):
            out = export_run(res, PROJECT_ROOT / "outputs" / out_name, figures=figs, db=db)
        st.success(f"Geschrieben nach {out.relative_to(PROJECT_ROOT)}")
        st.write(sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))
    c = st.columns(3)
    c[0].download_button("summary.json", json.dumps(summary_dict(res), indent=2, ensure_ascii=False),
                         file_name=f"{inp.name}_summary.json")
    if res.candidates:
        c[1].download_button("tab_results.tex", latex_results_table(res, tuple(res.candidates)), file_name="tab_results.tex")
        buf = io.StringIO()
        pd.concat([d.assign(scenario=k) for k, d in res.candidates.items()]).to_csv(buf, index=False)
        c[2].download_button("candidates.csv", buf.getvalue(), file_name=f"{inp.name}_candidates.csv")


PAGE_FUNCS = dict(start=page_start, profil=page_profil, split=page_split, spannung=page_spannung,
                  kapazitaet=page_kapazitaet, zellen=page_zellen, thermik=page_thermik, tradeoff=page_tradeoff,
                  balancing=page_balancing, emulation=page_emulation, detail=page_detail, db=page_db,
                  export=page_export)


def _set_nav_from_top():
    v = st.session_state.get("nav_top")
    if v:
        st.session_state["nav"] = v


def missing_items(r) -> list:
    """What is missing or needs a decision (shown red)."""
    out = []
    if not inp.req.I_C0max_confirmed:
        out.append("Wandlerstrom I_C0max bestätigen (Seite 3)")
    if r.profile is None:
        out.append("Kein Lastprofil: Thermik und Emulation brauchen ein Profil (Seite 1)")
    if r.skipped_cells:
        out.append(f"{len(r.skipped_cells)} Zellen mit fehlenden Daten übersprungen (Datenbank)")
    for sk, df in r.candidates.items():
        n = int((df.thermal_status.str.startswith("übersprungen")).sum())
        if n:
            out.append(f"{SCENARIOS.get(sk, sk)}: Thermik für {n} Zellen übersprungen (R_th/C_th fehlt)")
    if r.candidates and not inp.selected:
        out.append("Noch keine Zelle gewählt (Seite 7)")
    return out


page = st.session_state.get("nav", "start")
if page not in PAGE_FUNCS:
    page = "start"
_res0 = compute()
_stt = st.session_state.get("last_status", {})
st.session_state["nav_top"] = page
st.segmented_control("Schritte", PAGE_KEYS, key="nav_top", on_change=_set_nav_from_top, label_visibility="collapsed",
                     format_func=lambda k: f"{page_status(dict((p[0], p[2]) for p in PAGES)[k])}{SHORT[k]}")
ui.alerts(_res0.warnings)
with flow_slot.container():
    clicked = flownav.flowchart(_stt, page, key="flownav_side", width="stretch")
    st.markdown(flownav.legend_html(), unsafe_allow_html=True)
if clicked and clicked != page:
    st.session_state["nav"] = clicked
    st.rerun()
with missing_slot.container():
    miss = missing_items(_res0)
    if miss:
        st.markdown("**Was fehlt**")
        for m in miss:
            st.markdown(f"<div style='color:#b71c1c;font-size:.85rem;margin:2px 0'>■ {m}</div>",
                        unsafe_allow_html=True)
PAGE_FUNCS[page]()
