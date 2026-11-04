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
                                        coupling_factor, evaluate_candidate, layout_for, layout_pair)
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
    ("profil", "1  Lastprofil", ["0"]),
    ("split", "2  Leistungsaufteilung", ["0"]),
    ("spannung", "3  Spannungen und Lebensdauer", ["1", "2", "3"]),
    ("kapazitaet", "4  Kapazität", ["4"]),
    ("zellen", "5  Zellauswahl", ["5-11"]),
    ("thermik", "6  Thermik", ["5-11"]),
    ("tradeoff", "7  Bewertung und Auswahl", ["12"]),
    ("emulation", "8  Emulation und Auslegung", ["14"]),
    ("detail", "Zelldetail", []),
    ("db", "Datenbank", []),
    ("export", "Export", []),
]
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
    st.session_state["last_status"] = {k: s.status for k, s in r.steps.items()}
    return r


def page_status(steps: list) -> str:
    stt = st.session_state.get("last_status", {})
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
    st.markdown("**Ablauf**")
    if "nav" not in st.session_state:
        st.session_state["nav"] = "start"
    page = st.radio("Ablauf", PAGE_KEYS, key="nav", label_visibility="collapsed",
                    format_func=lambda k: dict((p[0], p[1]) for p in PAGES)[k])
    status_slot = st.empty()
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
                   "Die Seiten links führen Schritt für Schritt durch das Flussdiagramm. Jede Eingabe rechnet "
                   "sofort neu, fehlende Daten werden übersprungen und markiert.")
    res = compute()
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.graphviz_chart(flowchart_dot(res))
    with c2:
        st.markdown(f"**Preset:** `{inp.name}`")
        if inp.description:
            st.caption(inp.description)
        for k, s in res.steps.items():
            st.markdown(f"{ui.pill(s.status)} **{k} {s.title}**<br><span class='small'>{s.summary}</span>",
                        unsafe_allow_html=True)
        if res.warnings:
            st.warning("\n".join(f"- {w}" for w in res.warnings))
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
                    pf = (inp.selected[sk] or {}).get("P")
                    m = st.columns(4)
                    m[0].metric("Zelle", ui.short(cid))
                    m[1].metric("P/S", f"{pf}/{r.S}" if pf else r.setup,
                                "aus Emulation" if pf else "analytisch", delta_color="off")
                    m[2].metric("C_req", f"{sc.C_req:.2f} F")
                    dts = (inp.selected[sk] or {}).get("dT_max_K", r.dT_max_K) if pf else r.dT_max_K
                    m[3].metric("ΔT max", ui.fmt(dts, 1, "K"))
                else:
                    st.info("Noch keine Zelle gewählt (Schritt 7).")
    st.button("Los geht's: 1 Lastprofil →", on_click=go_to, args=("profil",), type="primary")


def flowchart_dot(res) -> str:
    st_ = {k: v.status for k, v in res.steps.items()}
    th_any = any((~df.thermal_status.str.startswith("übersprungen")).any() for df in res.candidates.values())
    th = "done" if th_any else "skipped"
    s5 = st_.get("5-11", "open")
    col = {"done": "#c8e6c9", "warning": "#fff59d", "skipped": "#eeeeee", "error": "#ef9a9a", "open": "#ffffff"}

    def node(nid, label, status, shape="box", dashed=False):
        style = "rounded,filled" + (",dashed" if dashed or status == "skipped" else "")
        if shape == "diamond":
            style = "filled" + (",dashed" if status == "skipped" else "")
        return f'{nid} [label="{label}", shape={shape}, style="{style}", fillcolor="{col.get(status, "#fff")}", fontsize=10];'

    return "\n".join([
        "digraph G {", "rankdir=TB; node [fontname=Helvetica]; edge [fontsize=9];",
        node("req", "Application Requirements", st_.get("0", "open")),
        node("dbn", "Cell Database", "done", dashed=True),
        'subgraph cluster_a {label="Module Specification"; color="#003361";',
        node("s1", "1 U_C0max, U_rated\\nlifetime derating, S", st_.get("1", "open")),
        node("s2", "2 U_C0min", st_.get("2", "open")), node("s3", "3 U_C0inM", st_.get("3", "open")),
        node("s4", "4 C_req", st_.get("4", "open")), "}",
        'subgraph cluster_b {label="Candidate-Cell Selection"; color="#f39200";',
        node("s5", "5 Select candidate cell", s5), node("s6", "6 p = max(P_C, P_I)", s5),
        node("s7", "7 R_th, C_th\\navailable?", th, "diamond"), node("s8", "8 P_T", th),
        node("s9", "9 Transient thermal\\nverification", th), node("s10", "10 Feasible?", th, "diamond"),
        node("s11", "11 More candidates?", s5, "diamond"), "}",
        node("s12", "12 Trade-off analysis", st_.get("12", "open")),
        node("s13", "13 Balancing concept", st_.get("13", "open"), dashed=True),
        node("s14", "Thermal simulation\\n(emulation)", st_.get("14", "open"), dashed=True),
        node("fin", "Final module design", "done" if inp.selected else "open"),
        "req -> s1; s1 -> s2; s2 -> s3; s3 -> s4; s4 -> s5; dbn -> s1 [style=dashed]; dbn -> s5 [style=dashed];",
        "s5 -> s6; s6 -> s7; s7 -> s8 [label=yes]; s7 -> s11 [label=no]; s8 -> s9; s9 -> s10;",
        's10 -> s11 [label=yes]; s10 -> s8 [label="no (p+1)"]; s11 -> s5 [label=yes]; s11 -> s12 [label=no];',
        "s12 -> s13; s13 -> s14; s14 -> fin;", "}"])


def page_profil():
    ui.step_header("SCHRITT 1", "Lastprofil und Anforderungen",
                   "Gemessenes Leistungsprofil der Anwendung. Positiv = Last, negativ = Rekuperation.")
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
            st.warning("Datei nicht gefunden. Es werden die manuellen Werte verwendet.")
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
    c[0].metric("P_avg", f"{s['P_avg_W']:.1f} W")
    c[1].metric("P_peak", f"{s['P_peak_W']:.0f} W")
    c[2].metric("P_RMS", f"{s['P_rms_W']:.0f} W")
    c[3].metric("Peak / Mittelwert", f"{s['peak_to_avg']:.1f}")
    c[4].metric("Dauer", f"{s['duration_s']:.1f} s", f"Abtastung {s['dt_s'] * 1e3:.3g} ms", delta_color="off")
    c[5].metric("Grenze Leistungsspitzen", f"{res.peaks['upper_fence']:.0f} W", "Q3 + 1.5 IQR", delta_color="off")
    td, pd_ = decimate_for_plot(prof.t, prof.p, 30000)
    fig = go.Figure(go.Scattergl(x=td, y=pd_, name="Last", line=dict(color=COLORS[0], width=1)))
    fig.add_hline(y=s["P_avg_W"], line_dash="dash", line_color="black")
    fig.update_xaxes(title_text="Time [s]", rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(title_text="Power [W]")
    st.plotly_chart(plotly_layout(fig, 380), width="stretch")
    st.markdown("##### Abbildungen (Diss-Stil)")
    c1, c2 = st.columns(2)
    ui.show_fig(pl.fig_peaks(res), "power_peaks", container=c1, plt_module=pl.plt)
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
    c[0].metric("E_DCH,max", f"{L['E_dch_J']:.1f} J")
    c[1].metric("E_CH,max", f"{L['E_ch_J']:.1f} J")
    c[2].metric("P_SC,max (SC-Seite)", f"{L['P_SC_max_W']:.0f} W")
    if inp.braking.enabled:
        c[3].metric("E_Bremsung (SC-Seite)", f"{L['E_brake_J']:.0f} J", f"P = {L['P_brake_W']:.0f} W", delta_color="off")
    c[4].metric("Energiedrift SC / Profil", f"{-np.sum(res.p_sc) * prof.dt:.0f} J",
                help="Bilanz der SC-Energie über ein Profil. Verluste muss die Nachladung decken.")
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
                plt_module=pl.plt)
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


def page_spannung():
    ui.step_header("SCHRITTE 1 BIS 3", "Modulspannungen und Lebensdauer",
                   "U_C0max aus dem DC-Bus, Spannungsderating nach Arrhenius für die geforderte Lebensdauer, "
                   "U_C0min aus Leistung und Stromgrenze, U_C0inM energiegewichtet.")
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
        W.num("I_C0max [A]", r, "I_C0max_A", "ic0max", container=c[2], help="Modul- bzw. Wandler-Stromgrenze")
        W.opt_num("U_C0min fest [V]", r, "U_C0min_override_V", "uminov", 33.0, container=c[3],
                  help="Leer = P_SC,max / I_C0max. IECON 2023 nutzt 33 V.")
        series = ["(je Zelle aus DB)"] + list(db.lifetime.series_id)
        cur = inp.opt.series_override or series[0]
        s = st.selectbox("Lebensdauer-Serie", series, index=series.index(cur) if cur in series else 0,
                         key=W.key("series"))
        inp.opt.series_override = None if s == series[0] else s
    res = compute()
    if not res.voltages:
        st.error(res.steps.get("2").summary if res.steps.get("2") else "Keine Spannungen berechnet.")
        return
    v = res.voltages
    c = st.columns(5)
    si = next(iter(res.series_info.values()))
    c[0].metric("U_C01max je Zelle", f"{si['U_C01max']:.3f} V")
    c[1].metric("Zellen in Serie S", si["S"], f"U_C0N = {si['U_C0N']:.1f} V", delta_color="off")
    c[2].metric("U_C0min", f"{v['U_C0min']:.2f} V", f"berechnet {v['U_C0min_calc']:.2f} V", delta_color="off")
    c[3].metric("Nutzbare Energie", f"{100 * v['energy_ratio']:.0f} %", f"r_Q = {v['r_Q']:.2f}", delta_color="off")
    c[4].metric("U_C0inM", " / ".join(f"{sc.U_inM:.1f}" for sc in res.scenarios.values()) + " V",
                "Modul 1 / Modul 2", delta_color="off")
    if res.steps["2"].status == "warning":
        st.warning(res.steps["2"].summary)
    st.latex(r"U_\mathrm{C01max}=U_\mathrm{C0N1}-\left(\log_2\frac{t_\mathrm{life}}{t_0}-\frac{T_0-T}{\Delta T}\right)"
             r"\Delta U,\quad S=\left\lceil\frac{U_\mathrm{C0max}}{U_\mathrm{C01max}}\right\rceil,\quad "
             r"U_\mathrm{C0min}=\frac{P_\mathrm{SC,max}}{I_\mathrm{C0max}}")
    c1, c2 = st.columns(2)
    ui.show_fig(pl.fig_lifetime(res, db), "lifetime_SC", container=c1, plt_module=pl.plt)
    ui.show_fig(pl.fig_discharge_ratio(res), "dischargeratio", container=c2, plt_module=pl.plt)
    sk = st.selectbox("Spannungsdiagramm für", list(res.scenarios), format_func=lambda s: SCENARIOS[s],
                      index=len(res.scenarios) - 1, key=W.key("vlev"))
    ui.show_fig(pl.fig_voltage_levels(res, sk), f"voltage_levels_{sk}", plt_module=pl.plt)
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
    res = compute()
    st.latex(r"C_\mathrm{req}=\frac{2W_\mathrm{SC,used}}{L_\mathrm{EOL}\left(U_\mathrm{C0max}^2-U_\mathrm{C0min}^2"
             r"\right)},\qquad U_\mathrm{C0inM}=\sqrt{\frac{E_\mathrm{DCH,max}U_\mathrm{C0max}^2+E_\mathrm{CH,max}"
             r"U_\mathrm{C0min}^2}{E_\mathrm{DCH,max}+E_\mathrm{CH,max}}}")
    cols = st.columns(max(1, len(res.scenarios)))
    for col, sc in zip(cols, res.scenarios.values()):
        with col.container(border=True):
            st.markdown(f"<div class='card-title'>{sc.label}</div>", unsafe_allow_html=True)
            m = st.columns(3)
            m[0].metric("W_used", f"{sc.W_used_J:.0f} J", f"{sc.E_dch_J:.0f} + {sc.E_ch_J:.0f} J", delta_color="off")
            m[1].metric("U_C0inM", f"{sc.U_inM:.2f} V")
            m[2].metric("C_req", f"{sc.C_req:.3f} F")
    nav_buttons("kapazitaet")


SEL_COLS = ["cell_id", "setup", "P_C", "P_I", "P_T", "C_cell_used_F", "ESR_cell_mOhm", "C_module_F", "C_reserve",
            "ESR_module_mOhm", "I_module_max_A", "cost_EUR", "mass_kg", "volume_dm3", "P_loss_avg_W", "dT_max_K",
            "pcb_mm", "density_kg_dm3", "status", "warnings"]


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
        spec_options(inp.req)
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
        st.caption(f"C_req = {sc.C_req:.3f} F, U_C0min = {sc.U_min:.1f} V, I_C0max = {sc.I_C0max:g} A. "
                   f"Zeile anklicken, um die Zelle im Zelldetail zu öffnen.")
        dfo = pd.concat([df[df.feasible], df[~df.feasible]])
        if (~df.feasible).any():
            st.caption(f"{int((~df.feasible).sum())} Konfiguration(en) außerhalb der Vorgabe stehen unten "
                       f"(Spalte Status) und sind in den Abbildungen ausgegraut.")
        ev = ui.table(dfo, SEL_COLS, key=W.key(f"tbl_{sk}"), selectable=True)
        rows = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
        if rows:
            cid = dfo.iloc[rows[0]].cell_id
            st.session_state["detail_cell"], st.session_state["detail_scen"] = cid, sk
            st.button(f"{ui.short(cid)} im Zelldetail öffnen", on_click=go_to, args=("detail",), key=W.key(f"od_{sk}"))
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


def spec_options(r):
    with st.expander("Vorgaben (Bauraum, Masse, Kosten)", expanded=any(
            getattr(r, k) is not None for k in ("max_width_mm", "max_depth_mm", "max_height_mm", "max_volume_dm3",
                                                "max_mass_kg", "max_cost_EUR"))):
        st.caption("Konfigurationen, die eine Vorgabe verletzen, werden als unzulässig markiert und in allen "
                   "Abbildungen ausgegraut. Breite und Tiefe steuern zusätzlich die Wahl des Layouts.")
        c = st.columns(3)
        W.opt_num("Breite max [mm]", r, "max_width_mm", "mxw", 200.0, container=c[0])
        W.opt_num("Tiefe max [mm]", r, "max_depth_mm", "mxd", 200.0, container=c[1])
        W.opt_num("Höhe max [mm]", r, "max_height_mm", "mxh", 50.0, container=c[2])
        c = st.columns(3)
        W.opt_num("Volumen max [dm³]", r, "max_volume_dm3", "mxv", 1.5, container=c[0])
        W.opt_num("Masse max [kg]", r, "max_mass_kg", "mxm", 1.5, container=c[1])
        W.opt_num("Kosten max [€]", r, "max_cost_EUR", "mxc", 300.0, container=c[2])


def layout_section(res):
    st.markdown("#### Leiterplatten-Layout")
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
    with c[2]:
        cc = st.columns(3)
        cc[0].metric("Platine", f"{main.width_mm:.0f} x {main.depth_mm:.0f} mm")
        cc[1].metric("Volumen", f"{main.volume_dm3:.2f} dm³")
        cc[2].metric("Moduldichte", f"{(m_cells + main.pcb_mass_kg(inp.opt.pcb_areal_mass_g_cm2)) / main.volume_dm3:.2f} kg/dm³")
    for n in main.notes:
        st.warning(n)
    if not row.feasible:
        st.warning(f"{ui.short(cid)}: {row.status}")
    ui.show_fig(pl.fig_layout_compare(sq, hx, m_cells, inp.opt.pcb_areal_mass_g_cm2),
                f"layout_compare_{ui.short(cid).replace(' ', '_')}_{row.setup.replace('/', 'x')}", plt_module=pl.plt)


TH_COLS = ["cell_id", "setup", "P_T", "I_cell_rms_A", "ESR_cell_mOhm", "R_th_KperW", "k_coupling", "R_th_eff_KperW",
           "dT_max_K", "T_cell_max_C", "thermal_status", "warnings"]


def page_thermik():
    ui.step_header("SCHRITTE 7 BIS 10", "Thermische Prüfung",
                   "Verluste I²ESR je Zelle, Temperatur mit R_th und C_th (1. Ordnung) über das Profil. "
                   "Die Modellstufen zeigen, wie stark Packung, Zellgröße und Alterung den Anstieg erhöhen.")
    o, r = inp.opt, inp.req
    with st.container(border=True):
        c = st.columns(3)
        W.select("R_th / C_th", o, "thermal_mode", "thm", THERMAL_MODES, container=c[0])
        W.select("Wärmeabfuhr", o, "thermal_coupling", "thc_coup", COUPLING_MODES, container=c[1])
        if o.thermal_coupling == "packed_block":
            W.num("Abstand zwischen Zellen [mm]", o, "cell_spacing_mm", "gap", container=c[2])
        elif o.thermal_coupling == "factor":
            W.num("Faktor auf R_th", o, "thermal_coupling_factor", "kfac", container=c[2])
        c = st.columns(4)
        esr_opts = {1.0: "neue Zelle (x1)", 2.0: "Lebensende (x2)"}
        cur = o.thermal_esr_factor if o.thermal_esr_factor in esr_opts else 1.0
        o.thermal_esr_factor = c[0].selectbox("ESR für Verluste", list(esr_opts), index=list(esr_opts).index(cur),
                                              format_func=esr_opts.get, key=W.key("esrf"))
        W.select("Modulstrom", o, "thermal_current", "thc", CURRENT_MODES, container=c[1])
        W.select("Profil", o, "thermal_profile", "thp", PROFILE_MODES, container=c[2])
        W.num("ΔT_max [K]", r, "dT_max_K", "dtmax", container=c[3])
        c = st.columns(4)
        W.num("T_amb [°C]", r, "T_amb_C", "tamb", container=c[0])
    res = compute()
    if not res.candidates:
        st.info("Keine Kandidaten.")
        return
    sk = st.selectbox("Szenario", list(res.candidates), index=len(res.candidates) - 1,
                      format_func=lambda s: SCENARIOS[s], key=W.key("th_scen"))
    df = res.candidates[sk]
    ui.table(df, TH_COLS, key=W.key(f"th_tbl_{sk}"))
    if df.dT_max_K.notna().any():
        ui.show_fig(pl.fig_dT_candidates(df, r.dT_max_K), f"dT_candidates_{sk}", plt_module=pl.plt)
        cands = list(df[df.dT_max_K.notna()].cell_id)
        dflt = selected_cell(sk) if selected_cell(sk) in cands else cands[0]
        cid = st.selectbox("Aufschlüsselung für Zelle", cands, index=cands.index(dflt), format_func=ui.short,
                           key=W.key(f"th_cell_{sk}_{dflt}"))
        det = res.details[sk][cid]
        row = df[df.cell_id == cid].iloc[0]
        st.markdown(f"**{ui.short(cid)} {row.setup}**: Block {row.block_mm} mm, I_Zelle,RMS = {row.I_cell_rms_A:.2f} A")
        c1, c2 = st.columns(2)
        if det.breakdown:
            ui.show_fig(pl.fig_thermal_breakdown(det.breakdown, r.dT_max_K), f"thermal_breakdown_{cid}",
                        container=c1, plt_module=pl.plt)
        if det.dT is not None:
            ui.show_fig(pl.fig_thermal(det, res.scenarios[sk], r.dT_max_K), f"thermal_{cid}", container=c2,
                        plt_module=pl.plt)
        with st.expander("Rechenweg"):
            st.code("\n".join(det.log))
            for w in det.warnings:
                st.warning(w)
    nav_buttons("thermik")


RANK_COLS = ["rank", "cell_id", "setup", "score", "cost_EUR", "mass_kg", "volume_dm3", "P_loss_avg_W", "dT_max_K",
             "n_cells", "C_reserve", "C_source"]


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
                    f"{feas.iloc[0].score:.0f} %" if best else None, delta_color="off")
        pk = res.picks.get(sk, {}).get("min_cost")
        c[1].metric("Günstigste", ui.short(pk["cell_id"]) if pk else "-")
        with c[2]:
            opts = list(feas.cell_id)
            idx = opts.index(cur_sel) if cur_sel in opts else (opts.index(best) if best in opts else 0)
            choice = st.selectbox("Gewählte Zelle", opts, index=idx, format_func=ui.short,
                                  key=W.key(f"choice_{sk}_{cur_sel}"))
            b1, b2 = st.columns(2)
            if b1.button("Auswahl übernehmen", type="primary", key=W.key(f"take_{sk}"), width="stretch"):
                inp.selected[sk] = {"cell_id": choice}
                st.rerun()
            if cur_sel and b2.button("Auswahl entfernen", key=W.key(f"drop_{sk}"), width="stretch"):
                inp.selected.pop(sk, None)
                st.rerun()
            st.caption(f"Aktuell gewählt: **{ui.short(cur_sel) if cur_sel else 'keine'}**")
        ui.table(feas, RANK_COLS, key=W.key(f"rank_{sk}"))
        s, contrib = score(feas, base, o.reserve_cap)
        ranked = feas.assign(score=s).sort_values("score", ascending=False)
        ui.show_fig(pl.fig_scores(ranked, contrib), f"scores_{sk}", plt_module=pl.plt)
    if res.candidates:
        st.markdown("#### Übersicht wie IECON 2023 (Kosten und Moduldichte über Zellkapazität, Farbe = Modul-ESR, "
                    "grau = außerhalb der Vorgabe)")
        c1, c2 = st.columns([1, 1])
        ui.show_fig(pl.fig_tradeoff(res, scenarios=tuple(res.candidates)), "Modules_Cells", container=c1,
                    plt_module=pl.plt)
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
    ui.step_header("THERMISCHE SIMULATION", "Emulation und emulationsbasierte Auslegung",
                   "Das Profil läuft endlos. Je Zeitschritt: Wandler-Wirkungsgrad, I = P/U, ESR-Verluste, Nachladung, "
                   "Spannungs- und Stromgrenzen, Zelltemperatur. Aus dem emulierten SC-Profil werden Energiehub, "
                   "C_req, Spitzen- und RMS-Strom neu bestimmt und P erhöht, bis alles eingehalten ist.")
    res = compute()
    if res.profile is None or not res.candidates:
        st.info("Für die Emulation werden ein Lastprofil und Kandidaten benötigt.")
        return
    sk = st.selectbox("Szenario", list(res.candidates), index=len(res.candidates) - 1,
                      format_func=lambda s: SCENARIOS[s], key=W.key("emu_scen"))
    df = res.candidates[sk][res.candidates[sk].feasible]
    opts = list(df.cell_id)
    main = selected_cell(sk) if selected_cell(sk) in opts else opts[0]
    with st.container(border=True):
        c = st.columns([1.3, 1.7])
        cells = c[0].multiselect("Zellen (die gewählte zuerst, weitere zum Vergleich)", opts, default=[main],
                                 format_func=ui.short, key=W.key(f"emu_cells_{sk}_{main}"))
        c[1].caption(f"Gewählt in Schritt 7: **{ui.short(main)}**. Weitere Zellen nur zum Vergleich.")
        c = st.columns(5)
        eta_d = c[0].number_input("η Entladen [%]", 50.0, 100.0, float(inp.req.eta_dch * 100), 0.5,
                                  key=W.key("emu_etad"))
        eta_c = c[1].number_input("η Laden [%]", 50.0, 100.0, float(inp.req.eta_ch * 100), 0.5, key=W.key("emu_etac"))
        eol = c[2].toggle("Prüfung am Lebensende", value=True, key=W.key("emu_eol"),
                          help="C x L_EOL und ESR x 2. Das ist der Zustand, für den C_req ausgelegt ist.")
        brk = c[3].toggle("Bremsereignis einfügen", value=sk == "profile_braking" and inp.braking.enabled,
                          key=W.key("emu_brk"), help="Zum ungünstigsten Zeitpunkt (maximale SC-Energie).")
        dtb = c[4].number_input("Zeitschritt [s]", 0.001, 0.1, 0.01, 0.001, format="%.3f", key=W.key("emu_dt"))
        with st.expander("Nachladeregler (Energiemanagement)"):
            c = st.columns(3)
            K = c[0].number_input("K [1/s]", value=0.05, format="%.3f", key=W.key("emu_K"))
            Prm = c[1].number_input("max. Nachladeleistung [W]", value=300.0, key=W.key("emu_prm"))
            tgt = c[2].selectbox("Ziel", ["U_C0inM", "U_C0max"], key=W.key("emu_tgt"))
    cfg_key = json.dumps(dict(sk=sk, cells=cells, eta=(eta_d, eta_c), eol=eol, brk=brk, dt=dtb, K=K, P=Prm, t=tgt,
                              inp=yaml.safe_dump({**inp.to_dict(), "selected": {}}, sort_keys=True)), sort_keys=True)
    if st.button("Emulationsbasierte Auslegung rechnen", type="primary"):
        out = {}
        with st.spinner("Emuliere ..."):
            for cid in cells:
                cfg = EmulationConfig(dt_bin_s=dtb, K_soc_per_s=K, P_rech_max_W=Prm, soc_target=tgt,
                                      eta_dch=eta_d / 100, eta_ch=eta_c / 100)
                out[cid] = emulation_design(res, sk, cid, cfg, check_eol=eol, include_braking=brk)
        st.session_state["emu_result"] = (cfg_key, out)
    stored = st.session_state.get("emu_result")
    if not stored:
        st.info("Einstellungen prüfen und auf 'Emulationsbasierte Auslegung rechnen' klicken.")
        return
    if stored[0] != cfg_key:
        st.warning("Einstellungen geändert: Ergebnisse unten gehören zu den vorherigen Einstellungen. Neu rechnen.")
    out = stored[1]
    sc = res.scenarios[sk]
    rows = []
    for cid, d in out.items():
        last = d.iterations[-1]
        crow = res.candidates[sk].set_index("cell_id").loc[cid]
        cell = db.cell(cid)
        n = d.S * d.P_final
        rows.append({"Zelle": ui.short(cid), "P/S analytisch": crow.setup, "P/S Emulation": f"{d.P_final}/{d.S}",
                     "Zellen": n, "Kosten [€]": n * cell["price_EUR"], "Masse [kg]": n * cell["mass_g"] / 1000,
                     "Energiehub [J]": last.E_swing_J, "C_req,emu [F]": last.C_req_emu_F,
                     "C Modul [F]": last.C_module_F, "I_peak [A]": last.I_peak_A,
                     "I_Zelle RMS [A]": last.I_rms_cell_A, "ΔT max [K]": last.dT_max_K,
                     "Ergebnis": "erfüllt" if d.feasible else last.reason})
    st.markdown("#### Ergebnis")
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    st.latex(r"W_\mathrm{swing}=\max E_\mathrm{SC}(t)-\min E_\mathrm{SC}(t),\quad C_\mathrm{req,emu}="
             r"\frac{2W_\mathrm{swing}}{L_\mathrm{EOL}\left(U_\mathrm{C0max}^2-U_\mathrm{C0min}^2\right)},\quad "
             r"I_\mathrm{SC}(t)=\frac{P_\mathrm{SC,Zelle}(t)}{U_\mathrm{SC}(t)},\quad P_\mathrm{SC,Zelle}="
             r"\begin{cases}P_\mathrm{SC,Bus}/\eta_\mathrm{dch}\\P_\mathrm{SC,Bus}\,\eta_\mathrm{ch}\end{cases}")
    for cid, d in out.items():
        with st.expander(f"{ui.short(cid)}: emuliertes Profil und Rechnung", expanded=cid == cells[0]):
            for n in d.notes:
                st.caption(n)
            it = pd.DataFrame([asdict(i) for i in d.iterations])
            it = it.rename(columns={"P": "P", "C_module_F": "C Modul [F]", "C_module_eol_F": "C Modul EOL [F]",
                                    "U_op_min_V": "U min [V]", "U_op_max_V": "U max [V]",
                                    "E_swing_J": "Energiehub [J]", "C_req_emu_F": "C_req,emu [F]",
                                    "P_C_emu": "P_C,emu", "I_peak_A": "I_peak [A]", "P_I_emu": "P_I,emu",
                                    "I_rms_module_A": "I_Modul RMS [A]", "I_rms_cell_A": "I_Zelle RMS [A]",
                                    "dT_max_K": "ΔT max [K]", "t_at_Umin_s": "t an U_min [s]",
                                    "t_at_Umax_s": "t an U_max [s]", "t_current_limit_s": "t an I-Grenze [s]",
                                    "E_overflow_J": "Überlauf [J]", "E_loss_esr_J": "ESR-Verlust [J]",
                                    "E_loss_conv_J": "Wandlerverlust [J]", "cycles": "Zyklen", "ok": "ok",
                                    "reason": "Begründung"})
            st.dataframe(it.round(2), hide_index=True, width="stretch")
            ui.show_fig(pl.fig_emulated_profile(d.trace, d.module, sc, i_limit=d.module.I_max_A),
                        f"emulated_profile_{cid}_{d.P_final}", plt_module=pl.plt)
            if d.chain is not None:
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
                m[4].metric("ΔT max", f"{ch['dT_max']:.1f} K", f"stationär {ch['dT_ss']:.1f} K", delta_color="off")
                st.caption(f"Kontrolle: I_RMS² x ESR x R_th,eff = {ch['I_rms']:.2f}² x {d.R_esr_cell_used:.4f} Ω x "
                           f"{d.module.R_th:.1f} K/W = {ch['I_rms'] ** 2 * d.R_esr_cell_used * d.module.R_th:.1f} K "
                           f"(Mittelwert), Maximum durch die Welligkeit {ch['dT_max']:.1f} K.")
                ui.show_fig(pl.fig_cell_thermal_chain(ch, inp.req.dT_max_K), f"cell_thermal_chain_{cid}_{d.P_final}",
                            plt_module=pl.plt)
            c1, c2 = st.columns(2)
            ui.show_fig(pl.fig_emudesign(d.iterations, inp.req.dT_max_K), f"emudesign_{cid}", container=c1,
                        plt_module=pl.plt)
            with c2:
                st.markdown("**Als Endauslegung übernehmen**")
                st.caption(f"{ui.short(cid)} mit P = {d.P_final}, S = {d.S} ({d.S * d.P_final} Zellen)")
                if st.button("Übernehmen", key=W.key(f"emu_take_{cid}"), type="primary"):
                    last = d.iterations[-1]
                    inp.selected[sk] = {"cell_id": cid, "P": int(d.P_final), "dT_max_K": float(last.dT_max_K),
                                        "C_module_F": float(last.C_module_F), "source": "emulation"}
                    st.success("Übernommen.")
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
                          EmulationConfig(eta_dch=inp.req.eta_dch, eta_ch=inp.req.eta_ch))
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
    c[0].metric("P/S", row["setup"])
    c[1].metric("C Modul", f"{row['C_module_F']:.2f} F")
    c[2].metric("ESR Modul", f"{row['ESR_module_mOhm']:.0f} mΩ")
    c[3].metric("Kosten", ui.fmt(row["cost_EUR"], 0, "€"))
    c[4].metric("Masse", ui.fmt(row["mass_kg"], 2, "kg"))
    c[5].metric("ΔT max", ui.fmt(row["dT_max_K"], 1, "K"))
    st.code("\n".join(det.log))
    for w in det.warnings:
        st.warning(w)
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
            ui.show_fig(pl.fig_pcb_layout(lay, ui.short(cid) + " " + row["setup"], m_cells, o.pcb_areal_mass_g_cm2),
                        f"pcb_{ui.short(cid).replace(' ', '_')}_{row['setup'].replace('/', 'x')}", plt_module=pl.plt)
    if not row["feasible"]:
        st.warning(row["status"])
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
                  emulation=page_emulation, detail=page_detail, db=page_db, export=page_export)
PAGE_FUNCS[page]()

with status_slot.container():
    stt = st.session_state.get("last_status", {})
    if stt:
        html = "".join(f"<div style='margin:2px 0'>{ui.pill(v)}<span class='small'>Schritt {k}</span></div>"
                       for k, v in stt.items())
        st.markdown(f"<div style='margin-top:.3rem'>{html}</div>", unsafe_allow_html=True)
