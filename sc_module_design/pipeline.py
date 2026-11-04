"""Flowchart orchestration: runs the design procedure step by step.

Every step writes a :class:`StepRecord` with a status (done / skipped / warning / error / open),
so missing data never stops the run: the corresponding step is skipped and reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import equations as eq
from .config import DesignInputs
from .database import Database, get_database
from .converter import bus_to_cell
from .energy import Excursions, braking_energy, braking_power, energy_excursions
from .paths import resolve
from .profile import PowerProfile, bin_signal, load_profile, peak_analysis
from .selection import ScenarioContext, evaluate_all, series_voltage
from .splits import apply_split, get_split
from .tradeoff import annotate, picks, preset_weights

SCENARIOS = {
    "profile": "Module 1: load profile (W = E_DCH,max + E_CH,max)",
    "profile_braking": "Module 2: with braking energy (W = E_DCH,max + E_braking)",
}


@dataclass
class StepRecord:
    id: str
    title: str
    status: str = "open"        # done | skipped | warning | error | open
    summary: str = ""
    values: dict = field(default_factory=dict)


@dataclass
class DesignResult:
    inputs: DesignInputs
    steps: dict = field(default_factory=dict)
    profile: Optional[PowerProfile] = None
    stats: dict = field(default_factory=dict)
    peaks: Optional[dict] = None
    p_bat: Optional[np.ndarray] = None
    p_sc: Optional[np.ndarray] = None          # SC-side power (after converter efficiency), used for the design
    p_sc_bus: Optional[np.ndarray] = None      # SC share of the load on the DC bus (output of the split)
    excursions: Optional[Excursions] = None
    load: dict = field(default_factory=dict)
    voltages: dict = field(default_factory=dict)
    series_info: dict = field(default_factory=dict)
    scenarios: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    picks: dict = field(default_factory=dict)
    skipped_cells: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def step(self, sid: str, title: str) -> StepRecord:
        rec = StepRecord(sid, title)
        self.steps[sid] = rec
        return rec

    def selected_row(self, scenario: str) -> Optional[pd.Series]:
        sel = self.inputs.selected.get(scenario) if self.inputs.selected else None
        df = self.candidates.get(scenario)
        if not sel or df is None:
            return None
        d = df[df.cell_id == sel.get("cell_id")]
        return None if d.empty else d.iloc[0]


def prepare_profile(inp: DesignInputs) -> Optional[PowerProfile]:
    ps = inp.profile
    if not ps.path:
        return None
    path = resolve(ps.path)
    if not path.exists():
        return None
    prof = load_profile(path, scale=ps.scale, time_col=ps.time_col, power_col=ps.power_col,
                        phase_factor=ps.phase_factor, resample_dt=ps.resample_dt)
    if ps.t_start is not None or ps.t_end is not None:
        prof = prof.window(ps.t_start, ps.t_end)
    return prof


def run_design(inp: DesignInputs, db: Optional[Database] = None, profile: Optional[PowerProfile] = None,
               split: Optional[tuple] = None, load_profile_file: bool = True) -> DesignResult:
    db = db or get_database()
    res = DesignResult(inputs=inp)
    req, opt = inp.req, inp.opt

    # ------------------------------------------------------------ step 0: application requirements
    s0 = res.step("0", "Application requirements")
    if profile is None and load_profile_file:
        try:
            profile = prepare_profile(inp)
        except Exception as exc:  # broken file -> continue with manual values
            res.warnings.append(f"Profile could not be loaded: {exc}")
            profile = None
    ml = inp.manual_load
    use_manual = ml.use == "always" or (ml.use == "if_no_profile" and profile is None)
    load = {}
    if profile is not None:
        res.profile = profile
        res.stats = profile.stats()
        res.peaks = peak_analysis(profile)
        if split is None:
            p_bat, p_sc = apply_split(inp.split.name, profile.t, profile.p, inp.split.params)
        else:
            p_bat, p_sc = split
        res.p_bat, res.p_sc_bus = p_bat, p_sc
        res.p_sc = bus_to_cell(p_sc, req.eta_dch, req.eta_ch) if (req.eta_dch != 1 or req.eta_ch != 1) else p_sc
        ex = energy_excursions(res.p_sc, profile.dt, opt.energy_method)
        res.excursions = ex
        load.update(P_peak_W=res.stats["P_peak_W"], P_avg_W=res.stats["P_avg_W"], E_dch_J=ex.E_dch_J,
                    E_ch_J=ex.E_ch_J, source=f"profile {profile.name} ({inp.split.name}, {opt.energy_method}, "
                                             f"eta_dch = {req.eta_dch:g}, eta_ch = {req.eta_ch:g})")
        for n in ex.notes:
            res.warnings.append(n)
    if use_manual:
        missing = [k for k in ("P_peak_W", "E_dch_J", "E_ch_J") if getattr(ml, k) is None and k not in load]
        if missing and profile is None:
            s0.status, s0.summary = "error", f"No profile and no manual values for {', '.join(missing)}."
            return res
        for k in ("P_peak_W", "P_avg_W", "E_dch_J", "E_ch_J"):
            if getattr(ml, k) is not None:
                load[k] = float(getattr(ml, k))
        load["source"] = "manual values" if profile is None else load.get("source", "") + " + manual override"
    load.setdefault("P_avg_W", float("nan"))
    if req.P_max_mode == "peak":
        load["P_SC_max_W"] = load["P_peak_W"] / req.eta_dch
    elif req.P_max_mode == "peak_minus_avg":
        load["P_SC_max_W"] = (load["P_peak_W"] - load["P_avg_W"]) / req.eta_dch
    else:
        load["P_SC_max_W"] = float(req.P_max_W) / req.eta_dch
    load["P_SC_tr_W"] = load["P_peak_W"] - load["P_avg_W"]
    br = inp.braking
    if br.enabled:
        E_b = br.E_J if (br.mode == "direct" and br.E_J) else braking_energy(br.mass_kg, br.v_mps)
        P_b = br.P_W if (br.mode == "direct" and br.P_W) else braking_power(E_b, br.t_brake_s)
        load.update(E_brake_J=E_b * req.eta_ch, P_brake_W=P_b * req.eta_ch, E_brake_bus_J=E_b)
    res.load = load
    s0.status = "done"
    s0.summary = (f"P_peak = {load['P_peak_W']:.1f} W, P_avg = {load['P_avg_W']:.1f} W, "
                  f"E_DCH,max = {load['E_dch_J']:.1f} J, E_CH,max = {load['E_ch_J']:.1f} J"
                  + (f", E_braking = {load['E_brake_J']:.0f} J" if br.enabled else ""))
    s0.values = dict(load)

    # ------------------------------------------------------------ step 1: U_C0max, lifetime, S
    s1 = res.step("1", "Select U_C0max, U_rated (lifetime derating, S)")
    U_max = req.U_C0max_V or req.U_bus_V
    if U_max > req.U_bus_V + 1e-9:
        res.warnings.append("U_C0max is above U_bus (Eq. C0maxModule).")
    cells, skipped = db.candidate_cells(opt.candidate_ids)
    res.skipped_cells = skipped
    series_ids = sorted({opt.series_override} if opt.series_override else set(cells.lifetime_series_id))
    msgs = []
    for sid in series_ids:
        try:
            ser = db.series(sid)
        except KeyError as exc:
            msgs.append(str(exc))
            continue
        u_raw = eq.u_cell_max_lifetime(req.t_life_h, req.T_cell_C, ser["u0_V"], ser["T0_C"], ser["t0_h"],
                                       ser["dU_V"], ser["dT_K"])
        u = series_voltage(ser, req)
        if u <= 0:
            msgs.append(f"{sid}: lifetime requirement not reachable (U_C01max = {u_raw:.2f} V)")
            continue
        S = eq.n_series(U_max, u)
        res.series_info[sid] = dict(series_id=sid, series=ser["series"], U_C0N1=ser["u0_V"], U_C01max=u,
                                    U_C01max_raw=u_raw, S=S, U_C0N=S * ser["u0_V"],
                                    U_cell_at_Umax=U_max / S, clipped=u_raw > ser["u0_V"])
        if u_raw > ser["u0_V"]:
            msgs.append(f"{sid}: requirement milder than rating, U_C01max limited to U_rated = {ser['u0_V']} V")
    cells = cells[cells.lifetime_series_id.isin(res.series_info) | bool(opt.series_override)]
    s1.values = dict(U_C0max=U_max, series=res.series_info)
    s1.status = "done" if res.series_info else "error"
    s1.summary = "; ".join(f"{v['series']} ({k}): U_C01max = {v['U_C01max']:.3f} V, S = {v['S']}, "
                           f"U_C0N = {v['U_C0N']:.1f} V" for k, v in res.series_info.items())
    if msgs:
        s1.status = "warning" if res.series_info else "error"
        s1.summary += (" | " if s1.summary else "") + " | ".join(msgs)
    if not res.series_info:
        return res

    # ------------------------------------------------------------ step 2: U_C0min
    s2 = res.step("2", "Select U_C0min")
    P_scmax = load["P_SC_max_W"]
    U_min_calc = eq.u_c0min_from_current(P_scmax, req.I_C0max_A)
    mode = req.U_C0min_mode
    if mode == "fixed" and req.U_C0min_override_V:
        U_min = float(req.U_C0min_override_V)
    elif mode == "current":
        U_min = U_min_calc
    else:
        mode = "r_Q"
        U_min = req.r_Q * U_max
    I_at_min = P_scmax / U_min
    # converter current recommendation from the profile
    p_rms = float(np.sqrt(np.mean(res.p_sc ** 2))) if res.p_sc is not None else float("nan")
    I_rec = I_at_min
    s2.values = dict(U_C0min=U_min, U_C0min_calc=U_min_calc, I_at_U_C0min=I_at_min, P_SC_max=P_scmax, mode=mode,
                     I_C0max_rec=I_rec, P_SC_rms=p_rms)
    s2.status = "done"
    s2.summary = {"r_Q": f"U_C0min = r_Q U_C0max = {req.r_Q:.2f} x {U_max:.1f} V = {U_min:.2f} V",
                  "current": f"U_C0min = P_SC,max / I_C0max = {P_scmax:.0f} W / {req.I_C0max_A:g} A = {U_min:.2f} V",
                  "fixed": f"U_C0min = {U_min:.2f} V (fest, berechnet {U_min_calc:.2f} V)"}[mode]
    if req.I_C0max_A < I_rec - 1e-6:
        s2.status = "warning"
        msg = (f"Wandler-Stromgrenze I_C0max = {req.I_C0max_A:g} A ist kleiner als P_SC,max / U_C0min = "
               f"{P_scmax:.0f} W / {U_min:.1f} V = {I_rec:.1f} A: die Spitzenleistung ist bei U_C0min nicht lieferbar.")
        s2.summary += " | " + msg
        res.warnings.append(msg)
    if not req.I_C0max_confirmed:
        res.warnings.append(f"Wandler-Stromgrenze I_C0max = {req.I_C0max_A:g} A ist noch nicht bestätigt "
                            f"(Empfehlung aus dem Profil: {I_rec:.1f} A Spitze bei U_C0min).")
    if br.enabled and load.get("P_brake_W", 0) > P_scmax:
        res.warnings.append("Braking power exceeds P_SC,max used for U_C0min.")
    if U_min >= U_max:
        s2.status, s2.summary = "error", s2.summary + " (U_C0min >= U_C0max)"
        return res
    res.voltages = dict(U_C0max=U_max, U_C0min=U_min, U_C0min_calc=U_min_calc, r_Q=U_min / U_max,
                        energy_ratio=float(eq.energy_ratio(U_min / U_max)), mode=mode, I_C0max_rec=I_rec,
                        P_SC_rms=p_rms, I_rms_at_Umin=p_rms / U_min if p_rms == p_rms else float("nan"))

    # ------------------------------------------------------------ steps 3/4 per scenario
    s3 = res.step("3", "Select U_C0inM (energy-weighted midpoint)")
    s4 = res.step("4", "Determine C_req from W_used, L_EOL")
    psc_b = bin_signal(res.p_sc, res.profile.dt, opt.thermal_dt_bin_s) if res.p_sc is not None else None
    for key in opt.scenarios:
        if key == "profile":
            E_ch, lab = load["E_ch_J"], "E_CH,max"
        elif key == "profile_braking":
            if not br.enabled:
                res.warnings.append("Scenario 'profile_braking' skipped: braking disabled.")
                continue
            E_ch, lab = load["E_brake_J"], "E_braking"
        else:
            res.warnings.append(f"Unknown scenario {key}")
            continue
        W = load["E_dch_J"] + E_ch
        U_inM = eq.u_c0inm(load["E_dch_J"], E_ch, U_max, U_min)
        C_req = eq.c_required(W, U_max, U_min, req.L_EOL)
        res.scenarios[key] = ScenarioContext(key, SCENARIOS[key], load["E_dch_J"], E_ch, W, U_min, U_inM, U_max,
                                             C_req, req.I_C0max_A, psc_b, lab)
    if res.scenarios and p_rms == p_rms:
        U_inM0 = next(iter(res.scenarios.values())).U_inM
        res.voltages["I_rms_at_UinM"] = p_rms / U_inM0
    s3.status = s4.status = "done" if res.scenarios else "error"
    s3.summary = "; ".join(f"{k}: U_C0inM = {s.U_inM:.2f} V" for k, s in res.scenarios.items())
    s4.summary = "; ".join(f"{k}: W_used = {s.W_used_J:.1f} J -> C_req = {s.C_req:.3f} F"
                           for k, s in res.scenarios.items())
    s3.values = {k: s.U_inM for k, s in res.scenarios.items()}
    s4.values = {k: dict(W_used=s.W_used_J, C_req=s.C_req) for k, s in res.scenarios.items()}

    # ------------------------------------------------------------ steps 5-11
    s5 = res.step("5-11", "Candidate cells, P = max(P_C, P_I, P_T), thermal verification")
    if cells.empty:
        s5.status, s5.summary = "error", "No candidate cells with complete data."
        return res
    for key, sc in res.scenarios.items():
        df, det = evaluate_all(cells, sc, res.series_info, req, opt, db)
        weights = preset_weights(opt.weight_preset) or opt.weights
        res.candidates[key] = annotate(df, weights, opt.reserve_cap)
        res.details[key] = det
    n_th = {k: int((~df.thermal_status.str.startswith("übersprungen")).sum()) for k, df in res.candidates.items()}
    n_feas = {k: int(df.feasible.sum()) for k, df in res.candidates.items()}
    s5.status = "done" if not skipped else "warning"
    s5.summary = "; ".join(f"{k}: {n_feas[k]}/{len(res.candidates[k])} feasible, thermal check for {n_th[k]}"
                           for k in res.candidates)
    if skipped:
        s5.summary += f" | {len(skipped)} cells skipped (missing data)"
    s5.values = dict(skipped=skipped, thermal_mode=opt.thermal_mode)

    # ------------------------------------------------------------ step 12
    s12 = res.step("12", "Trade-off analysis")
    for key, df in res.candidates.items():
        res.picks[key] = picks(df)
    s12.status = "done"
    s12.summary = "; ".join(
        f"{k}: score -> {p.get('best_score', {}).get('cell_id', '-')}, cost -> {p.get('min_cost', {}).get('cell_id', '-')}"
        for k, p in res.picks.items())
    if inp.selected:
        sel = [f"{k}: {v.get('cell_id')}" for k, v in inp.selected.items()]
        s12.summary += " | selected " + ", ".join(sel)

    # ------------------------------------------------------------ steps 13 / 14 (interfaces)
    s13 = res.step("13", "Balancing concept selection")
    s13.status, s13.summary = "skipped", "Interface only (no balancing model registered, see balancing.py)."
    s14 = res.step("14", "Thermal simulation and cooling concept")
    s14.status, s14.summary = "open", "Run the HESS emulation (endless loop) for the selected module."
    return res
