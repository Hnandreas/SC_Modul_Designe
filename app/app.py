"""SC Module Designer, Streamlit GUI.

Start:  streamlit run app/app.py   (or ./run_gui.sh)
The tabs follow the flowchart of the thesis. Inputs are collected first, then the design is computed
(cached), then the results are written into the same tabs.
"""
from __future__ import annotations

import copy
import io
import json
import math
import sys
import time
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

import widgets as W  # noqa: E402
from sc_module_design import equations as eq  # noqa: E402
from sc_module_design import plotting as pl  # noqa: E402
from sc_module_design.config import DesignInputs, list_presets, load_preset  # noqa: E402
from sc_module_design.database import TABLES, get_database  # noqa: E402
from sc_module_design.emulation import EmulationConfig, HESSEmulator, ModuleModel  # noqa: E402
from sc_module_design.energy import METHODS as ENERGY_METHODS, energy_excursions  # noqa: E402
from sc_module_design.paths import DB_DIR, PRESET_DIR, PRIVATE_PROFILE_DIR, PROJECT_ROOT, resolve  # noqa: E402
from sc_module_design.pipeline import SCENARIOS, prepare_profile, run_design  # noqa: E402
from sc_module_design.profile import decimate_for_plot, list_profiles  # noqa: E402
from sc_module_design.report import compare_with_reference, export_run, latex_results_table, summary_dict  # noqa: E402
from sc_module_design.selection import (C_SOURCES, ESR_SOURCES, GEOMETRY_MODES, THERMAL_MODES,  # noqa: E402
                                        evaluate_candidate, resolve_capacitance)
from sc_module_design.splits import REGISTRY, apply_split, get_split, load_plugins  # noqa: E402
from sc_module_design.thermal import CURRENT_MODES, PROFILE_MODES  # noqa: E402

st.set_page_config(page_title="SC Module Designer", layout="wide")
load_plugins()
db = get_database()

COLORS = ["#003361", "#f39200", "#c2003e", "#00665f", "#420066", "#18B6EE"]
STATUS_COLOR = {"done": "#c8e6c9", "warning": "#fff59d", "skipped": "#eeeeee", "error": "#ef9a9a",
                "open": "#ffffff"}

# ============================================================================ state
if "inp" not in st.session_state:
    names = [p.stem for p in list_presets()]
    st.session_state.inp = load_preset("thesis_datasheet" if "thesis_datasheet" in names else names[0])
    st.session_state.ver = 0
inp: DesignInputs = st.session_state.inp


def db_stamp() -> float:
    return max((DB_DIR / f).stat().st_mtime for f in TABLES.values() if (DB_DIR / f).exists())


@st.cache_resource(max_entries=4, show_spinner="Loading profile ...")
def cached_profile(spec_json: str, mtime: float):
    d = DesignInputs()
    d.profile.__dict__.update(json.loads(spec_json))
    return prepare_profile(d)


def profile_key() -> tuple:
    if not inp.profile.path:
        return ("", 0.0)
    p = resolve(inp.profile.path)
    return (json.dumps(asdict(inp.profile), sort_keys=True), p.stat().st_mtime if p.exists() else 0.0)


def get_profile():
    k = profile_key()
    if not k[0] or not k[1]:
        return None
    try:
        return cached_profile(*k)
    except Exception as exc:
        st.error(f"Profile could not be loaded: {exc}")
        return None


@st.cache_resource(max_entries=12, show_spinner="Splitting power ...")
def cached_split(pkey: tuple, name: str, params_json: str):
    prof = cached_profile(*pkey)
    return apply_split(name, prof.t, prof.p, json.loads(params_json))


@st.cache_resource(max_entries=6, show_spinner="Running design procedure ...")
def cached_design(inp_yaml: str, stamp: float, pkey: tuple, split_json: str):
    d = DesignInputs.from_dict(yaml.safe_load(inp_yaml))
    prof = cached_profile(*pkey) if pkey[0] and pkey[1] else None
    split = None
    if prof is not None:
        split = cached_split(pkey, d.split.name, json.dumps(d.split.params, sort_keys=True))
    return run_design(d, db=get_database(), profile=prof, split=split, load_profile_file=False)


def fig_show(fig, container=st):
    container.pyplot(fig, clear_figure=True)
    pl.plt.close(fig)


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


# ============================================================================ sidebar
with st.sidebar:
    st.header("SC Module Designer")
    presets = [p.stem for p in list_presets()]
    psel = st.selectbox("Preset", presets, index=presets.index(inp.name) if inp.name in presets else 0)
    c1, c2 = st.columns(2)
    if c1.button("Preset laden", width="stretch"):
        st.session_state.inp = load_preset(psel)
        st.session_state.ver = st.session_state.get("ver", 0) + 1
        for k in ("emu", "emu_hist", "emu_trace", "live_emu", "live_buf"):
            st.session_state.pop(k, None)
        st.rerun()
    if c2.button("Neu rechnen", width="stretch"):
        cached_design.clear()
    inp.name = st.text_input("Name des Designs", inp.name, key=W.key("name"))
    with st.expander("Beschreibung"):
        inp.description = st.text_area("Beschreibung", inp.description, key=W.key("desc"),
                                       label_visibility="collapsed")
    with st.expander("Preset speichern"):
        fname = st.text_input("Dateiname", inp.name, key=W.key("savename"))
        if st.button("In data/presets speichern"):
            p = inp.to_yaml(PRESET_DIR / f"{fname}.yaml")
            st.success(f"Gespeichert: {p.relative_to(PROJECT_ROOT)}")
        st.download_button("YAML herunterladen", yaml.safe_dump(inp.to_dict(), sort_keys=False, allow_unicode=True),
                           file_name=f"{inp.name}.yaml")
    status_box = st.container()

tabs = st.tabs(["Übersicht", "1 Anforderungen & Profil", "2 Spannungen & Lebensdauer", "3 Kapazität",
                "4 Zellauswahl & Trade-off", "5 Zelldetail", "6 Emulation (Endlosschleife)", "7 Datenbank",
                "8 Export"])
T0, T1, T2, T3, T4, T5, T6, T7, T8 = tabs

# ============================================================================ phase A: inputs
with T1:
    st.subheader("Anwendungsanforderungen und Lastprofil")
    ca, cb, cc = st.columns([2, 1, 1])
    profs = [str(p.relative_to(PROJECT_ROOT)) for p in list_profiles()]
    none_lbl = "(kein Profil: manuelle Werte)"
    cur = inp.profile.path if inp.profile.path in profs else (none_lbl if not inp.profile.path else inp.profile.path)
    options = [none_lbl] + profs + ([cur] if cur not in profs and cur != none_lbl else [])
    choice = ca.selectbox("Profil (data/profiles/...)", options, index=options.index(cur), key=W.key("profile"),
                          help="Private Profile liegen in data/profiles/private (nicht in git).")
    inp.profile.path = "" if choice == none_lbl else choice
    if inp.profile.path and not resolve(inp.profile.path).exists():
        ca.warning("Datei nicht gefunden (privates Profil fehlt auf diesem Rechner?). Es werden die manuellen "
                   "Werte verwendet, sofern gesetzt.")
    W.num("Skalierung", inp.profile, "scale", "pscale", fmt="%.6g", container=cb,
          help="P = Skalierung x Spalte (z. B. 400 für das normierte CSV)")
    W.num("Faktor P = U I x", inp.profile, "phase_factor", "pphase", container=cc,
          help="Nur wenn die Datei Spannung und Strom statt Leistung enthält (3 für 3-phasig).")
    with st.expander("Profil hochladen / Zeitfenster"):
        up = st.file_uploader("CSV mit Zeit [s] und Leistung [W] (oder Spannung und Strom)", type=["csv", "txt"])
        if up is not None:
            dest = PRIVATE_PROFILE_DIR / "uploads" / up.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(up.getvalue())
            st.success(f"Gespeichert in {dest.relative_to(PROJECT_ROOT)} (privat, nicht in git). Oben auswählen.")
        c1, c2 = st.columns(2)
        W.opt_num("t_start [s]", inp.profile, "t_start", "tstart", 0.0, container=c1)
        W.opt_num("t_end [s]", inp.profile, "t_end", "tend", 10.0, container=c2)
    with st.expander("Manuelle Lastwerte (ersetzen oder ergänzen das Profil)",
                     expanded=not inp.profile.path):
        W.select("Verwendung", inp.manual_load, "use", "ml_use",
                 {"never": "nie", "if_no_profile": "nur wenn kein Profil vorhanden",
                  "always": "immer (überschreibt Profilwerte)"})
        c = st.columns(4)
        W.opt_num("P_peak [W]", inp.manual_load, "P_peak_W", "ml_pp", 2643.0, container=c[0])
        W.opt_num("P_avg [W]", inp.manual_load, "P_avg_W", "ml_pa", 392.3, container=c[1])
        W.opt_num("E_DCH,max [J]", inp.manual_load, "E_dch_J", "ml_ed", 1214.1, container=c[2])
        W.opt_num("E_CH,max [J]", inp.manual_load, "E_ch_J", "ml_ec", 557.0, container=c[3])

    st.markdown("**Leistungsaufteilung Batterie / Supercap (Filter)**")
    names = list(REGISTRY)
    c1, c2 = st.columns([1, 2])
    sname = c1.selectbox("Strategie", names, index=names.index(inp.split.name) if inp.split.name in names else 0,
                         format_func=lambda n: REGISTRY[n].label, key=W.key("split"))
    strat = get_split(sname)
    c2.caption(strat.description)
    if sname != inp.split.name:
        inp.split.params = strat.defaults()
    inp.split.name = sname
    pc = st.columns(max(1, len(strat.params)))
    newp = {}
    for i, p in enumerate(strat.params):
        newp.update(W.param_widgets(type(strat)(strat.name, strat.label, strat.func, [p]), inp.split.params,
                                    f"sp_{sname}", container=pc[i]))
    inp.split.params = newp
    W.select("Energie-Auswertung (E_DCH,max / E_CH,max)", inp.opt, "energy_method", "emeth", ENERGY_METHODS)

    st.markdown("**Bremsenergie der mobilen Plattform (Module 2)**")
    c = st.columns(5)
    W.check("berücksichtigen", inp.braking, "enabled", "br_on", container=c[0])
    W.select("Modus", inp.braking, "mode", "br_mode", {"kinematic": "aus m, v", "direct": "E, P direkt"},
             container=c[1])
    if inp.braking.mode == "kinematic":
        W.num("Masse m [kg]", inp.braking, "mass_kg", "br_m", container=c[2])
        W.num("Geschwindigkeit v [m/s]", inp.braking, "v_mps", "br_v", container=c[3])
        W.num("Bremszeit t [s]", inp.braking, "t_brake_s", "br_t", container=c[4],
              help="P = 2E/t (lineare Rampe). 1.0 s ergibt 1980 W (Diss), 1.2 s ergibt 1650 W (IECON-Folie).")
    else:
        W.opt_num("E [J]", inp.braking, "E_J", "br_E", 990.0, container=c[2])
        W.opt_num("P [W]", inp.braking, "P_W", "br_P", 1980.0, container=c[3])

with T2:
    st.subheader("Modulspannungen und Lebensdauer (Schritte 1 bis 3)")
    r = inp.req
    c = st.columns(4)
    W.num("U_Bus [V]", r, "U_bus_V", "ubus", container=c[0])
    W.opt_num("U_C0max [V] (sonst = U_Bus)", r, "U_C0max_V", "uc0max", r.U_bus_V, container=c[1])
    W.num("Zelltemperatur T [°C] (Lebensdauer)", r, "T_cell_C", "tcell", container=c[2])
    W.num("Lebensdauer t_life [h]", r, "t_life_h", "tlife", step=1000.0, container=c[3])
    c = st.columns(4)
    W.select("P_SC,max für U_C0min", r, "P_max_mode", "pmaxmode",
             {"peak": "P_peak (konservativ, Diss)", "peak_minus_avg": "P_peak - P_avg", "manual": "manuell"},
             container=c[0])
    if r.P_max_mode == "manual":
        W.opt_num("P_SC,max [W]", r, "P_max_W", "pmaxw", 2643.0, container=c[1])
    W.num("I_C0max [A] (Modul/Wandler)", r, "I_C0max_A", "ic0max", container=c[2])
    W.opt_num("U_C0min fest [V]", r, "U_C0min_override_V", "uminov", 33.0, container=c[3],
              help="Leer = berechnet aus P_SC,max / I_C0max. IECON 2023 nutzt 33 V.")
    series = ["(je Zelle aus DB)"] + list(db.lifetime.series_id)
    cur = inp.opt.series_override or series[0]
    s = st.selectbox("Lebensdauer-Serie (Arrhenius-Parameter)", series, index=series.index(cur) if cur in series else 0,
                     key=W.key("series"))
    inp.opt.series_override = None if s == series[0] else s

with T3:
    st.subheader("Kapazitätsbemessung (Schritt 4)")
    c = st.columns(3)
    W.num("L_EOL (Restkapazität am Lebensende)", inp.req, "L_EOL", "leol", min_value=0.05, max_value=1.0,
          container=c[0])
    scen = c[1].multiselect("Szenarien", list(SCENARIOS), default=[s for s in inp.opt.scenarios if s in SCENARIOS],
                            format_func=lambda s: SCENARIOS[s], key=W.key("scen"))
    inp.opt.scenarios = scen or ["profile"]

with T4:
    st.subheader("Zellauswahl, Parallelschaltung und Trade-off (Schritte 5 bis 12)")
    o = inp.opt
    with st.expander("Optionen der Zellauswahl", expanded=True):
        c = st.columns(3)
        W.select("Kapazitätsquelle", o, "c_source", "csrc", C_SOURCES, container=c[0])
        if o.c_source != "datasheet":
            W.select("Spannungsfenster C0/k", o, "cfit_range", "crng",
                     {"90-20": "90-20 % (geringster Rekonstruktionsfehler)", "80-40": "80-40 % (IEC 62391-1)"},
                     container=c[1],
                     help="90-70 % wird für C0/k nicht angeboten (nur für ESR geeignet).")
            cur_opts = ["highest", "lowest", "nearest_rms"]
            cv = o.cfit_current if o.cfit_current in cur_opts else "highest"
            o.cfit_current = c[2].selectbox("Messstrom", cur_opts, index=cur_opts.index(cv), key=W.key("ccur"),
                                            format_func={"highest": "höchster Messstrom", "lowest": "niedrigster",
                                                         "nearest_rms": "nächst am RMS-Zellstrom"}.get)
            dsets = ["auto", "a", "b", "c", "d"]
            o.cfit_dataset = c[0].selectbox("Datensatz", dsets, index=dsets.index(o.cfit_dataset),
                                            key=W.key("cds"),
                                            help="auto: c (10 DUTs) vor a (8) vor d (3) vor b (1 DUT)")
        c = st.columns(3)
        W.select("ESR-Quelle", o, "esr_source", "esrsrc", ESR_SOURCES, container=c[0])
        if o.esr_source == "measured":
            o.esr_methods = c[1].multiselect("Methoden-Reihenfolge", ["M4", "M3", "M2", "M1"], default=o.esr_methods,
                                             key=W.key("esrm")) or ["M4"]
            ev = o.esr_current if o.esr_current in ("highest", "lowest") else "highest"
            o.esr_current = c[2].selectbox("ESR Messstrom", ["highest", "lowest"],
                                           index=["highest", "lowest"].index(ev), key=W.key("esrc"))
        c = st.columns(3)
        W.select("Volumen/Dichte", o, "geometry", "geom", GEOMETRY_MODES, container=c[0])
        if o.geometry == "corrected":
            W.num("Packungsfaktor", o, "packing_factor", "pack", min_value=0.1, max_value=1.0, container=c[1],
                  help="1.0 = reines Zellvolumen, 0.907 = hexagonale Packung")
        st.markdown("**Thermik (Schritte 7 bis 10)**")
        c = st.columns(4)
        W.select("Thermik-Modus", o, "thermal_mode", "thm", THERMAL_MODES, container=c[0])
        W.select("Modulstrom I(t)", o, "thermal_current", "thc", CURRENT_MODES, container=c[1])
        W.select("Profil", o, "thermal_profile", "thp", PROFILE_MODES, container=c[2])
        W.num("ΔT_max [K]", inp.req, "dT_max_K", "dtmax", container=c[3])
        c = st.columns(4)
        W.num("Zeitauflösung Thermik [s]", o, "thermal_dt_bin_s", "thdt", fmt="%.4f", container=c[0])
        W.integer("P-Grenze", o, "P_limit", "plim", container=c[1])
        allc = list(db.cells.cell_id)
        dflt = o.candidate_ids or list(db.cells[db.cells.candidate].cell_id)
        sel = c[2].multiselect("Kandidatenzellen", allc, default=[x for x in dflt if x in allc], key=W.key("cands"))
        o.candidate_ids = sel if set(sel) != set(db.cells[db.cells.candidate].cell_id) else None

# ============================================================================ phase B: compute
pkey = profile_key()
if pkey[0] and not pkey[1]:
    pkey = ("", 0.0)
inp_yaml = yaml.safe_dump(inp.to_dict(), sort_keys=True)
res = cached_design(inp_yaml, db_stamp(), pkey, json.dumps(inp.split.params, sort_keys=True))
prof = res.profile

with status_box:
    st.markdown("**Status der Schritte**")
    for k, s in res.steps.items():
        icon = {"done": "ok", "warning": "Warnung", "skipped": "übersprungen", "error": "FEHLER",
                "open": "offen"}.get(s.status, s.status)
        st.caption(f"{k}: {s.title} ({icon})")

# ============================================================================ phase C: results
def flowchart_dot(res) -> str:
    st_ = {k: v.status for k, v in res.steps.items()}
    th_any = any((~df.thermal_status.str.startswith("skipped")).any() for df in res.candidates.values())
    th = "done" if th_any else "skipped"
    s5 = st_.get("5-11", "open")

    def node(nid, label, status, shape="box", dashed=False):
        col = STATUS_COLOR.get(status, "#ffffff")
        style = "rounded,filled" + (",dashed" if dashed or status == "skipped" else "")
        if shape == "diamond":
            style = "filled" + (",dashed" if status == "skipped" else "")
        return f'{nid} [label="{label}", shape={shape}, style="{style}", fillcolor="{col}", fontsize=10];'

    lines = ["digraph G {", "rankdir=TB; node [fontname=Helvetica]; edge [fontsize=9];",
             node("req", "Application Requirements", st_.get("0", "open")),
             node("dbn", "Cell Database", "done", dashed=True),
             'subgraph cluster_a {label="Module Specification"; color="#003361";',
             node("s1", "1 Select U_C0max, U_rated\\nlifetime derating, S", st_.get("1", "open")),
             node("s2", "2 Select U_C0min", st_.get("2", "open")),
             node("s3", "3 Select U_C0inM", st_.get("3", "open")),
             node("s4", "4 Determine C_req", st_.get("4", "open")), "}",
             'subgraph cluster_b {label="Candidate-Cell Selection"; color="#f39200";',
             node("s5", "5 Select candidate cell", s5),
             node("s6", "6 p = max(P_C, P_I)", s5),
             node("s7", "7 R_th, C_th\\navailable?", th, "diamond"),
             node("s8", "8 P_T, p = max(p, P_T)", th),
             node("s9", "9 Transient thermal\\nverification", th),
             node("s10", "10 Feasible?", th, "diamond"),
             node("s11", "11 More candidates?", s5, "diamond"), "}",
             node("s12", "12 Trade-off analysis", st_.get("12", "open")),
             node("s13", "13 Balancing concept", st_.get("13", "open"), dashed=True),
             node("s14", "Thermal simulation\\n(cooling concept, emulation)", st_.get("14", "open"), dashed=True),
             node("fin", "Final module design", "done" if inp.selected else "open"),
             "req -> s1; s1 -> s2; s2 -> s3; s3 -> s4; s4 -> s5; dbn -> s1 [style=dashed]; dbn -> s5 [style=dashed];",
             "s5 -> s6; s6 -> s7; s7 -> s8 [label=yes]; s7 -> s11 [label=no]; s8 -> s9; s9 -> s10;",
             's10 -> s11 [label=yes]; s10 -> s8 [label="no (p+1)"]; s11 -> s5 [label=yes]; s11 -> s12 [label=no];',
             "s12 -> s13; s13 -> s14; s14 -> fin;", "}"]
    return "\n".join(lines)


with T0:
    st.subheader("Ablauf nach Flussdiagramm")
    c1, c2 = st.columns([1, 1])
    c1.graphviz_chart(flowchart_dot(res))
    with c2:
        st.caption("Farben: grün = gerechnet, gelb = Warnung, grau gestrichelt = übersprungen, rot = Fehler")
        for k, s in res.steps.items():
            st.markdown(f"**{k} {s.title}** `{s.status}`  \n{s.summary}")
        if res.warnings:
            st.warning("\n".join(f"- {w}" for w in res.warnings))
        if res.skipped_cells:
            with st.expander(f"{len(res.skipped_cells)} Zellen übersprungen (fehlende Daten)"):
                st.write("\n".join(f"- {m}" for m in res.skipped_cells))
    if res.scenarios:
        st.markdown("**Kennwerte**")
        cols = st.columns(len(res.scenarios) + 1)
        v = res.voltages
        cols[0].metric("U_C0min / U_C0max", f"{v['U_C0min']:.1f} / {v['U_C0max']:.1f} V")
        for i, (k, sc) in enumerate(res.scenarios.items()):
            cols[i + 1].metric(sc.label.split(":")[0], f"C_req = {sc.C_req:.3f} F", f"U_C0inM = {sc.U_inM:.2f} V",
                               delta_color="off")

with T1:
    st.divider()
    if prof is not None:
        s = res.stats
        c = st.columns(7)
        c[0].metric("P_avg", f"{s['P_avg_W']:.1f} W")
        c[1].metric("P_peak", f"{s['P_peak_W']:.0f} W")
        c[2].metric("P_RMS", f"{s['P_rms_W']:.0f} W")
        c[3].metric("Dauer", f"{s['duration_s']:.1f} s")
        c[4].metric("Abtastung", f"{s['dt_s'] * 1e3:.3g} ms")
        c[5].metric("Peak/Avg", f"{s['peak_to_avg']:.1f}")
        c[6].metric("Grenze Peaks (Q3+1.5 IQR)", f"{res.peaks['upper_fence']:.0f} W")
        td, pd_ = decimate_for_plot(prof.t, prof.p, 30000)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=("Last und Batterie", "Supercap"))
        fig.add_trace(go.Scattergl(x=td, y=pd_, name="P_load", line=dict(color=COLORS[0], width=1)), 1, 1)
        if res.p_bat is not None:
            tb, pb = decimate_for_plot(prof.t, res.p_bat, 30000)
            fig.add_trace(go.Scattergl(x=tb, y=pb, name="P_bat", line=dict(color=COLORS[1], width=2)), 1, 1)
            ts, ps = decimate_for_plot(prof.t, res.p_sc, 30000)
            fig.add_trace(go.Scattergl(x=ts, y=ps, name="P_SC", line=dict(color=COLORS[2], width=1)), 2, 1)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        fig.update_yaxes(title_text="P [W]")
        fig.update_xaxes(title_text="t [s]", row=2, col=1)
        st.plotly_chart(fig, width="stretch")
        c1, c2, c3 = st.columns(3)
        fig_show(pl.fig_peaks(res), c1)
        fig_show(pl.fig_fft(res), c2)
        legacy = inp.opt.energy_method == "zero_crossing_legacy" and inp.split.params.get("alignment") == "legacy_matlab"
        fig_show(pl.fig_energy_excursions(res, "iecon" if legacy else "auto"), c3)

        st.markdown("**Filtervergleich** (rechnet E_DCH,max, E_CH,max, U_C0inM und C_req für mehrere Strategien)")
        cmp_names = st.multiselect("Strategien (Standardparameter, aktuelle Strategie mit deinen Parametern)",
                                   list(REGISTRY), default=[inp.split.name], format_func=lambda n: REGISTRY[n].label,
                                   key=W.key("cmp"))
        if cmp_names and res.voltages:
            rows, curves = [], {}
            U_max, U_min = res.voltages["U_C0max"], res.voltages["U_C0min"]
            for n in cmp_names:
                prm = inp.split.params if n == inp.split.name else REGISTRY[n].defaults()
                try:
                    pb, psc = cached_split(pkey, n, json.dumps(prm, sort_keys=True))
                except Exception as exc:
                    st.error(f"{n}: {exc}")
                    continue
                ex = energy_excursions(psc, prof.dt, inp.opt.energy_method)
                W1 = ex.E_dch_J + ex.E_ch_J
                rows.append(dict(strategy=REGISTRY[n].label, params=json.dumps(prm), E_DCH_max_J=ex.E_dch_J,
                                 E_CH_max_J=ex.E_ch_J, U_C0inM_V=eq.u_c0inm(ex.E_dch_J, ex.E_ch_J, U_max, U_min),
                                 C_req_module1_F=eq.c_required(W1, U_max, U_min, inp.req.L_EOL),
                                 C_req_module2_F=eq.c_required(ex.E_dch_J + res.load.get("E_brake_J", 0.0), U_max,
                                                               U_min, inp.req.L_EOL),
                                 P_SC_rms_W=float(np.sqrt(np.mean(psc ** 2))), P_bat_max_W=float(pb.max()),
                                 P_bat_min_W=float(pb.min()), SC_energy_drift_J=float(-np.sum(psc) * prof.dt)))
                curves[REGISTRY[n].label] = pb
            st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)
            if len(curves) > 1:
                fig_show(pl.fig_split_compare(prof.t, prof.p, curves))
    else:
        st.info("Kein Profil geladen: Es wird mit den manuellen Werten gerechnet. Thermik und Emulation "
                "benötigen ein Profil.")
    if res.load:
        st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.load.items()}, expanded=False)

with T2:
    st.divider()
    if res.series_info:
        st.dataframe(pd.DataFrame(res.series_info.values()).round(4), width="stretch", hide_index=True)
    if res.voltages:
        v = res.voltages
        c = st.columns(4)
        c[0].metric("U_C0max", f"{v['U_C0max']:.2f} V")
        c[1].metric("U_C0min (verwendet)", f"{v['U_C0min']:.2f} V", f"berechnet {v['U_C0min_calc']:.2f} V",
                    delta_color="off")
        c[2].metric("r_Q", f"{v['r_Q']:.3f}", f"Energieanteil {100 * v['energy_ratio']:.1f} %", delta_color="off")
        for k, sc in res.scenarios.items():
            c[3].metric(f"U_C0inM ({k})", f"{sc.U_inM:.2f} V")
        st.latex(r"U_\mathrm{C01max}=U_\mathrm{C0N1}-\left(\log_2\frac{t_\mathrm{cell}}{t_0}-\frac{T_0-T}{\Delta T}"
                 r"\right)\Delta U_\mathrm{tC01},\qquad S=\left\lceil\frac{U_\mathrm{C0max}}{U_\mathrm{C01max}}"
                 r"\right\rceil")
        c1, c2 = st.columns(2)
        fig_show(pl.fig_lifetime(res, db), c1)
        fig_show(pl.fig_discharge_ratio(res), c2)
        sk = st.selectbox("Spannungsdiagramm für", list(res.scenarios), format_func=lambda s: SCENARIOS[s],
                          key=W.key("vlev"))
        fig_show(pl.fig_voltage_levels(res, sk))

with T3:
    st.divider()
    st.latex(r"C_\mathrm{req}=\frac{2W_\mathrm{SC,used}}{L_\mathrm{EOL}\left(U_\mathrm{C0max}^2-"
             r"U_\mathrm{C0min}^2\right)},\qquad U_\mathrm{C0inM}=\sqrt{\frac{E_\mathrm{DCH,max}U_\mathrm{C0max}^2+"
             r"E_\mathrm{CH,max}U_\mathrm{C0min}^2}{E_\mathrm{DCH,max}+E_\mathrm{CH,max}}}")
    if res.scenarios:
        st.dataframe(pd.DataFrame([dict(scenario=s.label, E_DCH_max_J=s.E_dch_J, E_2=s.E_ch_label,
                                        E_2_J=s.E_ch_J, W_used_J=s.W_used_J, U_C0min_V=s.U_min,
                                        U_C0inM_V=s.U_inM, U_C0max_V=s.U_max, L_EOL=inp.req.L_EOL,
                                        C_req_F=s.C_req) for s in res.scenarios.values()]).round(3),
                     width="stretch", hide_index=True)

SHOW_COLS = ["cell_id", "setup", "P_C", "P_I", "P_T", "C_cell_used_F", "ESR_cell_mOhm", "C_module_F",
             "ESR_module_mOhm", "I_module_max_A", "cost_EUR", "mass_kg", "volume_dm3", "density_kg_dm3",
             "E_density_Wh_kg", "P_loss_avg_W", "thermal_status", "dT_max_K", "pareto", "score", "feasible",
             "warnings"]

with T4:
    if not res.candidates:
        st.error("Keine Kandidaten berechnet (siehe Übersicht).")
    for skey, df in res.candidates.items():
        sc = res.scenarios[skey]
        st.markdown(f"### {sc.label}")
        st.caption(f"C_req = {sc.C_req:.3f} F, U_C0min = {sc.U_min:.2f} V, U_C0max = {sc.U_max:.2f} V, "
                   f"I_C0max = {sc.I_C0max:g} A. Zeile anklicken für Details / Auswahl.")
        view = df[[c for c in SHOW_COLS if c in df]].copy()
        ev = st.dataframe(view.round(3), width="stretch", hide_index=True, on_select="rerun",
                          selection_mode="single-row", key=W.key(f"tbl_{skey}"))
        rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
        c1, c2, c3 = st.columns([2, 1, 1])
        pk = res.picks.get(skey, {})
        c1.write(" | ".join(f"**{v['label']}**: {v['cell_id']}" + (f" (gleichauf: {', '.join(v['ties'])})"
                                                                    if v["ties"] else "") for v in pk.values()))
        if rows:
            cid = view.iloc[rows[0]].cell_id
            st.session_state["detail_cell"] = cid
            st.session_state["detail_scen"] = skey
            if c2.button(f"{cid} als Auswahl übernehmen", key=W.key(f"selbtn_{skey}")):
                inp.selected[skey] = {"cell_id": cid}
                st.rerun()
        cur_sel = inp.selected.get(skey, {}).get("cell_id")
        if cur_sel:
            c3.success(f"Ausgewählt: {cur_sel}")
            if c3.button("Auswahl entfernen", key=W.key(f"unsel_{skey}")):
                inp.selected.pop(skey, None)
                st.rerun()
    if res.candidates:
        with st.expander("Gewichtung für den Score (normiert, kleiner ist besser)"):
            c = st.columns(4)
            for i, k in enumerate(["cost", "mass", "volume", "esr"]):
                inp.opt.weights[k] = c[i].slider(k, 0.0, 3.0, float(inp.opt.weights.get(k, 1.0)), 0.1,
                                                 key=W.key(f"w_{k}"))
        c1, c2 = st.columns([1, 1])
        cmap = c2.selectbox("Farbskala", ["jet", "viridis", "plasma", "cividis"], key=W.key("cmap"))
        fig_show(pl.fig_tradeoff(res, scenarios=tuple(res.candidates), cmap=cmap), c1)
        with c2:
            axes_opts = ["c_nom_F", "C_module_F", "cost_EUR", "mass_kg", "volume_dm3", "density_kg_dm3",
                         "ESR_module_mOhm", "E_density_Wh_kg", "dT_max_K", "P_loss_avg_W", "n_cells"]
            xa = st.selectbox("x", axes_opts, index=2, key=W.key("px"))
            ya = st.selectbox("y", axes_opts, index=3, key=W.key("py"))
            fig = go.Figure()
            for i, (skey, df) in enumerate(res.candidates.items()):
                d = df[df.feasible]
                fig.add_trace(go.Scatter(x=d[xa], y=d[ya], mode="markers+text", text=d.c_nom_F.astype(int).astype(str) + " F",
                                         textposition="top center", name=skey,
                                         marker=dict(size=np.where(d.pareto, 14, 9), color=COLORS[i],
                                                     line=dict(width=np.where(d.pareto, 2, 0), color="black")),
                                         customdata=d[["cell_id", "setup"]],
                                         hovertemplate="%{customdata[0]} %{customdata[1]}<br>x=%{x:.3g}<br>y=%{y:.3g}"))
            fig.update_layout(height=420, xaxis_title=xa, yaxis_title=ya, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
            st.caption("Umrandet = Pareto-optimal (Kosten, Masse, Volumen, ESR).")
        if inp.opt.geometry == "legacy_iecon":
            cmp_ = compare_with_reference(res, db)
            if not cmp_.empty:
                st.markdown(f"**Vergleich mit der IECON-2023-Tabelle**: {int(cmp_.match.sum())}/{len(cmp_)} Werte "
                            f"reproduziert")
                st.dataframe(cmp_, width="stretch", hide_index=True)
        with st.expander("LaTeX-Tabelle (Layout wie Tab. results)"):
            st.code(latex_results_table(res, tuple(res.candidates)), language="latex")

# ---------------------------------------------------------------------------- cell detail
with T5:
    st.subheader("Zelldetail: eine Zelle mit eigenen Optionen durchrechnen")
    if not res.candidates:
        st.info("Keine Kandidaten.")
    else:
        scen_keys = list(res.candidates)
        dsc = st.session_state.get("detail_scen", scen_keys[-1])
        skey = st.selectbox("Szenario", scen_keys, index=scen_keys.index(dsc) if dsc in scen_keys else 0,
                            format_func=lambda s: SCENARIOS[s], key="det_scen")
        cells_all = list(db.cells.cell_id)
        dcell = st.session_state.get("detail_cell") or (inp.selected.get(skey, {}).get("cell_id")) or \
            res.candidates[skey].cell_id.iloc[0]
        cid = st.selectbox("Zelle", cells_all, index=cells_all.index(dcell) if dcell in cells_all else 0,
                           key="det_cell")
        cell = db.cell(cid)
        miss = db.missing_fields(cell)
        if miss:
            st.error(f"Für {cid} fehlen Daten: {', '.join(miss)}. Bitte in der Datenbank ergänzen.")
        else:
            o = copy.deepcopy(inp.opt)
            c = st.columns(4)
            o.c_source = c[0].selectbox("Kapazitätsquelle", list(C_SOURCES), index=list(C_SOURCES).index(o.c_source),
                                        key="det_csrc")
            o.cfit_range = c[1].selectbox("Fenster C0/k", ["90-20", "80-40"],
                                          index=["90-20", "80-40"].index(o.cfit_range), key="det_crng")
            fits = db.cfits[(db.cfits.cell_id == cid) & (db.cfits.voltage_range == o.cfit_range)]
            cur_choices = ["highest", "lowest"] + [f"{a:g}" for a in sorted(fits.current_A.unique())]
            cc = c[2].selectbox("Messstrom [A]", cur_choices, key="det_ccur")
            o.cfit_current = cc if cc in ("highest", "lowest") else float(cc)
            dsets = ["auto"] + sorted(fits.dataset.unique())
            o.cfit_dataset = c[3].selectbox("Datensatz", dsets, key="det_cds")
            c = st.columns(4)
            o.esr_source = c[0].selectbox("ESR-Quelle", list(ESR_SOURCES), index=list(ESR_SOURCES).index(o.esr_source),
                                          key="det_esr")
            tmodes = list(THERMAL_MODES) + ["manual"]
            tm = c[1].selectbox("Thermik", tmodes, index=tmodes.index(o.thermal_mode), key="det_tm",
                                format_func=lambda m: {"manual": "manual: eigene R_th/C_th"}.get(m, f"{m}: {THERMAL_MODES.get(m, '')}"))
            override = None
            if tm == "manual":
                rth = c[2].number_input("R_th [K/W]", value=22.1, key="det_rth")
                cth = c[3].number_input("C_th [J/K]", value=19.7, key="det_cth")
                override = dict(rth=rth, cth=cth, note="manual input", warning="Manual R_th/C_th (not from DB).")
            else:
                o.thermal_mode = tm
            c = st.columns(4)
            o.thermal_current = c[0].selectbox("Strom I(t)", list(CURRENT_MODES),
                                               index=list(CURRENT_MODES).index(o.thermal_current), key="det_tc")
            o.thermal_profile = c[1].selectbox("Profil", list(PROFILE_MODES),
                                               index=list(PROFILE_MODES).index(o.thermal_profile), key="det_tp")
            pmin = c[2].number_input("P mindestens", 1, 500, 1, key="det_pmin")
            sc = res.scenarios[skey]
            sid = o.series_override or cell["lifetime_series_id"]
            si = res.series_info.get(sid)
            if si is None:
                st.error(f"Lebensdauer-Serie {sid} nicht berechnet.")
            else:
                row, det = evaluate_candidate(cell, sc, si["S"], si["U_C01max"], inp.req, o, db,
                                              thermal_override=override, P_min=int(pmin))
                st.session_state["detail_row"] = (skey, row, det.fit)
                c = st.columns(6)
                c[0].metric("Setup P/S", row["setup"])
                c[1].metric("C_module", f"{row['C_module_F']:.3f} F")
                c[2].metric("ESR_module", f"{row['ESR_module_mOhm']:.1f} mΩ")
                c[3].metric("Kosten", f"{fmt(row['cost_EUR'], 0)} €")
                c[4].metric("Masse / Volumen", f"{fmt(row['mass_kg'])} kg / {fmt(row['volume_dm3'])} dm³")
                c[5].metric("ΔT_max", f"{fmt(row['dT_max_K'])} K")
                st.markdown("**Rechenweg**")
                st.code("\n".join(det.log))
                for w in det.warnings:
                    st.warning(w)
                st.caption(f"Kapazität: {row['C_source']} | ESR: {row['ESR_source']}")
                c1, c2 = st.columns(2)
                if det.fit is not None:
                    S = row["S"]
                    fig_show(pl.fig_cu(det.fit, cell, sc.U_min / S, sc.U_max / S, row["C_cell_used_F"]
                                       if o.c_source == "ceq_M1" else eq.c_eq_energy(det.fit["C0_F"], det.fit["k_FperV"],
                                                                                    sc.U_min / S, sc.U_max / S)), c1)
                if det.dT is not None:
                    fig_show(pl.fig_thermal(det, sc, inp.req.dT_max_K, f"{cid} {row['setup']}"), c2)
                if st.button("Diese Konfiguration für die Emulation verwenden"):
                    st.session_state["emu_source"] = "detail"
                    st.success("Übernommen. Weiter im Tab Emulation.")
                with st.expander("Alle Werte"):
                    st.json({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()})

# ---------------------------------------------------------------------------- emulation
with T6:
    st.subheader("HESS-Emulation: Profil in Endlosschleife")
    if prof is None or res.p_bat is None:
        st.info("Für die Emulation wird ein Lastprofil benötigt.")
    elif not res.candidates:
        st.info("Keine Kandidaten.")
    else:
        src_opts = []
        for skey in res.candidates:
            if inp.selected.get(skey):
                src_opts.append(f"Auswahl {skey}: {inp.selected[skey]['cell_id']}")
        if "detail_row" in st.session_state:
            src_opts.append("Zelldetail-Konfiguration")
        if not src_opts:
            st.info("Bitte zuerst eine Zelle auswählen (Tab 4) oder im Zelldetail übernehmen.")
        else:
            src = st.selectbox("Modul", src_opts, index=len(src_opts) - 1
                               if st.session_state.get("emu_source") == "detail" else 0)
            if src.startswith("Auswahl"):
                skey = src.split()[1].rstrip(":")
                df = res.candidates[skey]
                row = df[df.cell_id == inp.selected[skey]["cell_id"]].iloc[0].to_dict()
                fit = res.details[skey][row["cell_id"]].fit
            else:
                skey, row, fit = st.session_state["detail_row"]
            sc = res.scenarios[skey]
            with st.expander("Modulparameter (editierbar)", expanded=False):
                c = st.columns(5)
                S = c[0].number_input("S", 1, 500, int(row["S"]), key="em_S")
                P = c[1].number_input("P", 1, 500, int(row["P"]), key="em_P")
                Cc = c[2].number_input("C_cell [F]", value=float(row["C_cell_used_F"]), key="em_C")
                Rc = c[3].number_input("ESR_cell [mΩ]", value=float(row["ESR_cell_mOhm"]), key="em_R")
                Ilim = c[4].number_input("I-Grenze [A]", value=float(min(inp.req.I_C0max_A, row["I_module_max_A"])),
                                         key="em_I")
                c = st.columns(5)
                Umin = c[0].number_input("U_min [V]", value=float(sc.U_min), key="em_umin")
                Umax = c[1].number_input("U_max [V]", value=float(sc.U_max), key="em_umax")
                U0 = c[2].number_input("U_start [V]", value=float(sc.U_inM), key="em_u0")
                rth0 = row.get("R_th_KperW")
                has_th = rth0 is not None and not (isinstance(rth0, float) and math.isnan(rth0))
                th_on = c[3].checkbox("Thermik", value=has_th, key="em_th")
                use_cu = c[4].checkbox("C(U) = C0 + kU", value=False, disabled=fit is None, key="em_cu",
                                       help="Nur wenn für die Zelle C0/k in der DB vorhanden sind.")
                c = st.columns(5)
                Rth = c[0].number_input("R_th [K/W]", value=float(rth0) if has_th else 22.1, key="em_rth")
                Cth = c[1].number_input("C_th [J/K]", value=float(row.get("C_th_JperK")) if has_th else 19.7,
                                        key="em_cth")
            with st.expander("Energiemanagement und Simulation", expanded=True):
                c = st.columns(5)
                soc = c[0].checkbox("SOC-Regler (Nachladen)", value=True, key="em_soc",
                                    help="Lädt den SC über die Batterie auf das Ziel nach. Beim Peak-Shaver nötig.")
                tgt = c[1].selectbox("Ziel", ["U_C0inM", "U_C0max"], key="em_tgt")
                K = c[2].number_input("K [1/s]", value=0.05, format="%.4f", key="em_K")
                Prm = c[3].number_input("P_nachladen,max [W]", value=300.0, key="em_prm")
                dtb = c[4].number_input("Zeitschritt [s]", value=0.01, format="%.4f", key="em_dt")
                c = st.columns(5)
                eol = c[0].checkbox("End of Life", value=False, key="em_eol")
                esr_eol = c[1].number_input("ESR-Faktor EOL", value=2.0, key="em_esreol")
                maxc = c[2].number_input("max. Zyklen", 1, 2000, 200, key="em_maxc")
            module = ModuleModel(S=int(S), P=int(P), C_cell_F=float(Cc), R_cell_Ohm=float(Rc) / 1000, U_min=Umin,
                                 U_max=Umax, U_init=U0, I_max_A=Ilim,
                                 C0_cell_F=float(fit["C0_F"]) if (use_cu and fit) else None,
                                 k_cell_FperV=float(fit["k_FperV"]) if (use_cu and fit) else None,
                                 R_th=Rth if th_on else None, C_th=Cth if th_on else None,
                                 label=f"{row['cell_id']} {int(P)}/{int(S)}")
            cfg = EmulationConfig(dt_bin_s=dtb, soc_control=soc, soc_target=tgt, K_soc_per_s=K, P_rech_max_W=Prm,
                                  eol=eol, L_EOL=inp.req.L_EOL, esr_eol_factor=esr_eol, max_cycles=int(maxc))
            c1, c2 = st.columns(2)
            if c1.button("Bis zum periodischen Beharrungszustand rechnen", type="primary"):
                with st.spinner("Emulation läuft ..."):
                    em = HESSEmulator(prof.t, prof.p, res.p_bat, module, cfg)
                    hist, tr = em.run_until_steady()
                st.session_state["emu_hist"], st.session_state["emu_trace"] = hist, tr
                st.session_state["emu_module"] = module
            if "emu_hist" in st.session_state:
                hist, tr = st.session_state["emu_hist"], st.session_state["emu_trace"]
                last = hist[-1]
                c = st.columns(6)
                c[0].metric("Zyklen bis periodisch", len(hist))
                c[1].metric("U_C0 min / max", f"{last.U_min:.1f} / {last.U_max:.1f} V")
                c[2].metric("ΔT_max Zelle", f"{last.dT_max:.2f} K")
                c[3].metric("P_bat max / RMS", f"{last.P_bat_max_W:.0f} / {last.P_bat_rms_W:.0f} W")
                c[4].metric("ESR-Verluste / Zyklus", f"{last.E_loss_J:.0f} J")
                c[5].metric("Überlauf an Batterie", f"{last.E_overflow_J:.0f} J",
                            help="Energie, die der SC wegen Spannungs- oder Stromgrenze nicht liefern/aufnehmen konnte")
                if last.t_at_Umin_s > 0 or last.t_at_Umax_s > 0 or last.t_current_limit_s > 0:
                    st.warning(f"Grenzen erreicht pro Zyklus: U_min {last.t_at_Umin_s:.2f} s, U_max "
                               f"{last.t_at_Umax_s:.2f} s, Stromgrenze {last.t_current_limit_s:.2f} s")
                fig_show(pl.fig_emulation(tr, hist, st.session_state.get("emu_module")))
                st.dataframe(pd.DataFrame([asdict(h) for h in hist]).round(3), width="stretch", hide_index=True)

            st.markdown("**Live-Emulation (läuft endlos, bis du sie stoppst)**")
            c = st.columns(4)
            live = c[0].toggle("Live starten", key="live_on")
            speed = c[1].select_slider("Profilzeit pro Aktualisierung [s]", [1, 2, 5, 10, 20, 50], value=5,
                                       key="live_speed")
            win = c[2].select_slider("Fenster [s]", [20, 60, 150, 300, 600], value=60, key="live_win")
            if c[3].button("Zurücksetzen"):
                st.session_state.pop("live_emu", None)
                st.session_state.pop("live_buf", None)
            if live and "live_emu" not in st.session_state:
                st.session_state["live_emu"] = HESSEmulator(prof.t, prof.p, res.p_bat, module, cfg)
                st.session_state["live_buf"] = None

            def live_view():
                em = st.session_state.get("live_emu")
                if not st.session_state.get("live_on") or em is None:
                    st.caption("Gestoppt.")
                    return
                tr = em._advance(max(1, int(round(st.session_state.get("live_speed", 5) / em.dt))))
                buf = st.session_state.get("live_buf")
                cols = ["t", "P_load", "P_bat", "P_sc", "U_sc", "dT"]
                new = np.vstack([tr.t, tr.P_load, tr.P_bat, tr.P_sc, tr.U_sc, tr.dT])
                buf = new if buf is None else np.hstack([buf, new])
                wn = st.session_state.get("live_win", 60)
                buf = buf[:, buf[0] >= buf[0, -1] - wn]
                st.session_state["live_buf"] = buf
                f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                f.add_trace(go.Scattergl(x=buf[0], y=buf[1], name="P_load", line=dict(color="#999999", width=1)), 1, 1)
                f.add_trace(go.Scattergl(x=buf[0], y=buf[2], name="P_bat", line=dict(color=COLORS[1], width=2)), 1, 1)
                f.add_trace(go.Scattergl(x=buf[0], y=buf[3], name="P_SC", line=dict(color=COLORS[0], width=1)), 1, 1)
                f.add_trace(go.Scattergl(x=buf[0], y=buf[4], name="U_C0", line=dict(color=COLORS[0])), 2, 1)
                f.add_hline(y=em.m.U_min, line_dash="dash", line_color=COLORS[2], row=2, col=1)
                f.add_hline(y=em.m.U_max, line_dash="dash", line_color=COLORS[2], row=2, col=1)
                f.add_trace(go.Scattergl(x=buf[0], y=buf[5], name="ΔT", line=dict(color=COLORS[2])), 3, 1)
                f.update_yaxes(title_text="P [W]", row=1, col=1)
                f.update_yaxes(title_text="U [V]", row=2, col=1,
                               range=[em.m.U_min - 0.05 * (em.m.U_max - em.m.U_min),
                                      em.m.U_max + 0.05 * (em.m.U_max - em.m.U_min)])
                f.update_yaxes(title_text="ΔT [K]", row=3, col=1)
                f.update_xaxes(title_text="t [s]", row=3, col=1)
                f.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"),
                                uirevision="live")
                st.caption(f"Zyklus {em.cycle + 1}, Emulationszeit {buf[0, -1]:.1f} s, U_C0 = {buf[4, -1]:.2f} V, "
                           f"ΔT = {buf[5, -1]:.2f} K")
                st.plotly_chart(f, width="stretch", key="live_chart")

            st.fragment(run_every=1.0 if live else None)(live_view)()

# ---------------------------------------------------------------------------- database
with T7:
    st.subheader("Datenbank (CSV-Dateien in data/database, SQLite wird automatisch gebaut)")
    tname = st.selectbox("Tabelle", list(TABLES), key="db_table")
    df = db.tables[tname]
    ed = st.data_editor(df, num_rows="dynamic", width="stretch", key=f"db_edit_{tname}")
    c1, c2 = st.columns([1, 3])
    if c1.button("Änderungen speichern", type="primary"):
        db.save_table(tname, ed)
        get_database(reload=True)
        cached_design.clear()
        st.success(f"{TABLES[tname]} gespeichert, SQLite neu gebaut.")
        st.rerun()
    c2.caption("Neue Zelle: Zeile in 'cells' ergänzen, candidate = True setzen. Pflichtfelder: c_nom_F, u_rated_V, "
               "esr_dc_mOhm, i_max_A, lifetime_series_id. Masse, Maße und Preis sind für den Trade-off nötig. "
               "Thermikdaten: thermal_params (gemessen) oder rth_KperW / cth_JperK in cells (Datenblatt).")
    with st.expander("SQL-Abfrage"):
        q = st.text_area("SQL", "SELECT cell_id, dataset, voltage_range, current_A, C0_F, k_FperV FROM capacitance_fits "
                         "WHERE voltage_range = '90-20' ORDER BY cell_id", key="sql")
        if st.button("Ausführen"):
            try:
                st.dataframe(db.query(q), width="stretch")
            except Exception as exc:
                st.error(str(exc))
    src = DB_DIR / "SOURCES.md"
    if src.exists():
        with st.expander("Quellen der Daten"):
            st.markdown(src.read_text(encoding="utf8"))

# ---------------------------------------------------------------------------- export
with T8:
    st.subheader("Export")
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
        c[1].download_button("tab_results.tex", latex_results_table(res, tuple(res.candidates)),
                             file_name="tab_results.tex")
        buf = io.StringIO()
        pd.concat([df.assign(scenario=k) for k, df in res.candidates.items()]).to_csv(buf, index=False)
        c[2].download_button("candidates.csv", buf.getvalue(), file_name=f"{inp.name}_candidates.csv")
