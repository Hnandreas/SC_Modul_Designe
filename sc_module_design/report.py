"""Export of a design run: inputs (YAML), step summary, candidate tables, LaTeX table, figures."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import plotting as pl
from .database import Database, get_database
from .paths import OUTPUT_DIR


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def summary_dict(res) -> dict:
    return _clean(dict(
        name=res.inputs.name,
        created=datetime.now().isoformat(timespec="seconds"),
        steps={k: dict(title=s.title, status=s.status, summary=s.summary) for k, s in res.steps.items()},
        load=res.load, voltages=res.voltages, series=res.series_info,
        scenarios={k: dict(label=s.label, E_dch_J=s.E_dch_J, E_ch_J=s.E_ch_J, W_used_J=s.W_used_J,
                           U_C0min=s.U_min, U_C0inM=s.U_inM, U_C0max=s.U_max, C_req_F=s.C_req)
                   for k, s in res.scenarios.items()},
        picks=res.picks, selected=res.inputs.selected, warnings=res.warnings, skipped_cells=res.skipped_cells,
    ))


def latex_results_table(res, scenarios=("profile", "profile_braking"),
                        picks_order=("max_density", "min_cost")) -> str:
    """Table in the layout of Tab. results (thesis) / IECON 2023 summary."""
    cols = []
    names = {"max_density": "Density", "min_cost": "Cost", "min_mass": "Mass", "min_volume": "Volume",
             "min_esr": "ESR"}
    for s in scenarios:
        if s not in res.candidates:
            continue
        df = res.candidates[s].set_index("cell_id")
        for p in picks_order:
            pk = res.picks[s].get(p)
            if pk:
                cols.append((s, names.get(p, p), df.loc[pk["cell_id"]]))
        sel = (res.inputs.selected or {}).get(s)
        if sel and sel.get("cell_id") in df.index:
            cols.append((s, "Selected", df.loc[sel["cell_id"]]))
    mods = {"profile": "Module 1", "profile_braking": "Module 2"}
    groups = []
    for s, _, _ in cols:
        if not groups or groups[-1][0] != s:
            groups.append([s, 0])
        groups[-1][1] += 1
    head1 = " & ".join(fr"\multicolumn{{{n}}}{{c}}{{\footnotesize {mods.get(s, s)}}}" for s, n in groups)
    head2 = " & ".join(fr"\footnotesize {lab}" for _, lab, _ in cols)

    def row(label, key, fmt, unit):
        vals = " & ".join(fr"\footnotesize {fmt(r[key])}" for _, _, r in cols)
        return fr"& \footnotesize {label} & {vals} & \footnotesize {unit} \\"

    f0 = lambda v: f"{v:.0f}"  # noqa: E731
    f1 = lambda v: f"{v:.1f}"  # noqa: E731
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    lines = [
        r"\begin{tabular}{c" + "c" * (len(cols) + 2) + "}", r"\toprule",
        "& & " + head1 + r" & \\", r"\midrule",
        "& & " + head2 + r" & \footnotesize Units \\", r"\midrule",
        r"\multirow{3}{*}{C}",
        row("Cell capacitance", "c_nom_F", f0, "[F]"),
        row("Density", "density_kg_dm3", f1, r"[kg/dm$^3$]"),
        r"& \footnotesize Setup $P/S$ & " + " & ".join(fr"\footnotesize {r['setup']}" for _, _, r in cols)
        + r" & \footnotesize -- \\",
        r"\midrule",
        r"\multirow{5}{*}{\rotatebox[origin=c]{90}{Module}}",
        row("ESR", "ESR_module_mOhm", f0, r"[m$\Omega$]"),
        row("Capacitance", "C_module_F", f1, "[F]"),
        row("Cost", "cost_EUR", f0, r"[\euro]"),
        row("Weight", "mass_kg", f2, "[kg]"),
        row("Volume", "volume_dm3", f2, r"[dm$^3$]"),
        r"\bottomrule", r"\end{tabular}",
    ]
    return "\n".join(lines)


def compare_with_reference(res, db: Optional[Database] = None) -> pd.DataFrame:
    """Compare the picks with the values printed in the IECON 2023 table."""
    db = db or get_database()
    ref = db.tables.get("reference_iecon2023_table")
    if ref is None or ref.empty:
        return pd.DataFrame()
    out = []
    pmap = {"density": "max_density", "cost": "min_cost"}
    for _, r in ref.iterrows():
        s = {"module1": "profile", "module2": "profile_braking"}[r.scenario]
        if s not in res.candidates:
            continue
        df = res.candidates[s]
        if r.pick == "selected":
            cid = (res.inputs.selected or {}).get(s, {}).get("cell_id")
        else:
            cid = res.picks[s].get(pmap[r.pick], {}).get("cell_id")
        if not cid:
            continue
        c = df[df.cell_id == cid].iloc[0]
        for key, col in [("cell_c_F", "c_nom_F"), ("setup", "setup"), ("esr_mOhm", "ESR_module_mOhm"),
                         ("c_module_F", "C_module_F"), ("cost_EUR", "cost_EUR"), ("mass_kg", "mass_kg"),
                         ("volume_dm3", "volume_dm3"), ("density_kg_dm3", "density_kg_dm3")]:
            calc = c[col]
            refv = r[key]
            if isinstance(calc, str) or isinstance(refv, str):
                match = str(calc) == str(refv)
            else:
                tol = 0.06 * abs(refv) + (0.051 if key in ("c_module_F", "mass_kg", "volume_dm3", "density_kg_dm3")
                                          else 0.51)
                match = abs(float(calc) - float(refv)) <= tol
            out.append(dict(scenario=r.scenario, pick=r.pick, cell=cid, quantity=key, iecon=str(refv),
                            calculated=str(calc) if isinstance(calc, str) else f"{float(calc):.3f}", match=match))
    return pd.DataFrame(out)


def export_run(res, out_dir: Optional[str | Path] = None, figures: bool = True, db: Optional[Database] = None
               ) -> Path:
    db = db or get_database()
    out = Path(out_dir) if out_dir else OUTPUT_DIR / f"{res.inputs.name}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    res.inputs.to_yaml(out / "inputs.yaml")
    (out / "summary.json").write_text(json.dumps(summary_dict(res), indent=2, ensure_ascii=False), encoding="utf8")
    for k, df in res.candidates.items():
        df.to_csv(out / f"candidates_{k}.csv", index=False)
    if res.candidates:
        (out / "tab_results.tex").write_text(latex_results_table(res), encoding="utf8")
    cmp_ = compare_with_reference(res, db)
    if not cmp_.empty and res.inputs.opt.geometry == "legacy_iecon":
        cmp_.to_csv(out / "comparison_iecon2023.csv", index=False)
    lines = [f"# {res.inputs.name}", "", res.inputs.description or "", "", "## Flowchart steps", ""]
    for k, s in res.steps.items():
        lines.append(f"- **{k} {s.title}** [{s.status}]: {s.summary}")
    if res.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in res.warnings]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf8")
    if figures:
        fdir = out / "figures"
        figs = {}
        figs["dischargeratio"] = pl.fig_discharge_ratio(res) if res.voltages else None
        if res.scenarios:
            figs["voltage_levels"] = pl.fig_voltage_levels(res)
        if res.series_info:
            figs["lifetime_SC"] = pl.fig_lifetime(res, db)
        if res.profile is not None:
            figs["power_profile"] = pl.fig_profile(res)
            figs["power_peaks"] = pl.fig_peaks(res)
            figs["fft"] = pl.fig_fft(res)
            legacy = res.inputs.opt.energy_method == "zero_crossing_legacy" and res.inputs.split.params.get(
                "alignment") == "legacy_matlab"
            figs["energy_excursions"] = pl.fig_energy_excursions(res, "iecon" if legacy else "auto")
        if res.candidates:
            figs["Modules_Cells"] = pl.fig_tradeoff(res, scenarios=tuple(res.candidates))
            if res.inputs.opt.geometry == "pcb":
                from .selection import layout_pair
                for sk, sel in (res.inputs.selected or {}).items():
                    df = res.candidates.get(sk)
                    if df is None or sel.get("cell_id") not in set(df.cell_id):
                        continue
                    row = df[df.cell_id == sel["cell_id"]].iloc[0]
                    P = int(row.P)                  # P from the flowchart (the emulation only checks)
                    cell = db.cell(row.cell_id)
                    sq, hx = layout_pair(cell, int(row.S), P, res.inputs.opt, res.inputs.req)
                    if sq is not None:
                        m_cells = int(row.S) * P * float(cell["mass_g"]) / 1000
                        figs[f"layout_{sk}"] = pl.fig_layout_compare(sq, hx, m_cells,
                                                                      res.inputs.opt.pcb_areal_mass_g_cm2)
        for name, fig in figs.items():
            if fig is not None:
                pl.save_fig(fig, fdir / name)
                pl.plt.close(fig)
    return out
