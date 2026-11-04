"""Emulation-based module design (closed loop).

The analytic procedure (steps 1 to 11) gives a first P. Then the HESS is emulated with the load profile
repeated endlessly (converter efficiency per time step, ESR losses, recharge controller, current and
voltage limits, cell temperature) and the requirements are re-derived from the emulated SC profile:

    W_swing   = max(E_SC) - min(E_SC) in the periodic cycle          (stored-energy swing)
    C_req,emu = 2 W_swing / (L_EOL (U_C0max^2 - U_C0min^2))
    P_C,emu   = ceil(S C_req,emu / C_cell)
    P_I,emu   = ceil(min(I_peak, I_C0max) / I_cell,max)
    dT_max    from the emulated cell losses (R_th,eff, C_th)

P is increased until no voltage limit is reached, P >= P_C,emu, P_I,emu and dT_max <= dT_max,limit.
The check runs at end of life by default (C x L_EOL, ESR x factor), the condition C_req is sized for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import equations as eq
from .database import Database, _isnan, get_database
from .emulation import CycleTrace, EmulationConfig, HESSEmulator, ModuleModel
from .selection import coupling_factor
from .thermal import cell_loss_lowpass


@dataclass
class EmuIteration:
    P: int
    C_module_F: float
    C_module_eol_F: float
    U_op_min_V: float
    U_op_max_V: float
    E_swing_J: float
    C_req_emu_F: float
    P_C_emu: int
    I_peak_A: float
    P_I_emu: int
    I_rms_module_A: float
    I_rms_cell_A: float
    P_v_cell_mean_W: float
    dT_max_K: float
    t_at_Umin_s: float
    t_at_Umax_s: float
    t_current_limit_s: float
    E_overflow_J: float
    E_loss_esr_J: float
    E_loss_conv_J: float
    cycles: int
    ok: bool
    reason: str


@dataclass
class EmuDesign:
    cell_id: str
    scenario: str
    S: int
    P_analytic: int
    P_final: int
    feasible: bool
    iterations: list = field(default_factory=list)
    trace: Optional[CycleTrace] = None
    history: list = field(default_factory=list)
    module: Optional[ModuleModel] = None
    chain: Optional[dict] = None              # i_cell -> p_v = i^2 ESR -> thermal low-pass (full resolution)
    R_esr_cell_used: float = float("nan")
    braking_at_s: Optional[float] = None
    notes: list = field(default_factory=list)


def add_braking_event(t: np.ndarray, p: np.ndarray, t0: float, E_J: float, t_brake_s: float) -> np.ndarray:
    """Superimpose a triangular braking ramp (regeneration, negative power) starting at t0."""
    out = p.copy()
    P0 = 2.0 * E_J / t_brake_s
    m = (t >= t0) & (t < t0 + t_brake_s)
    out[m] -= P0 * (1.0 - (t[m] - t0) / t_brake_s)
    return out


def _module(row: dict, cell: dict, sc, P: int, req, opt, i_c0max: float) -> ModuleModel:
    S = int(row["S"])
    rth, cth = row.get("R_th_KperW"), row.get("C_th_JperK")
    has_th = rth is not None and not _isnan(rth)
    k = coupling_factor(cell, S * P, opt, S, req) if has_th else 1.0
    return ModuleModel(S=S, P=P, C_cell_F=float(row["C_cell_used_F"]), R_cell_Ohm=float(row["ESR_cell_mOhm"]) / 1000,
                       U_min=sc.U_min, U_max=sc.U_max, U_init=sc.U_inM,
                       I_max_A=min(i_c0max, P * float(cell["i_max_A"])),
                       R_th=float(rth) * k if has_th else None, C_th=float(cth) if has_th else None,
                       label=f"{row['cell_id']} {P}/{S}")


def emulation_design(res, scenario: str, cell_id: str, cfg: Optional[EmulationConfig] = None,
                     check_eol: bool = True, include_braking: Optional[bool] = None,
                     braking_at_s: Optional[float] = None, max_iter: int = 40,
                     db: Optional[Database] = None) -> EmuDesign:
    db = db or get_database()
    inp, req, opt = res.inputs, res.inputs.req, res.inputs.opt
    sc = res.scenarios[scenario]
    df = res.candidates[scenario]
    row = df[df.cell_id == cell_id].iloc[0].to_dict()
    cell = db.cell(cell_id)
    S = int(row["S"])
    if cfg is None:
        cfg = EmulationConfig(eta_dch=req.eta_dch, eta_ch=req.eta_ch)
    cfg.eol = check_eol
    cfg.L_EOL = req.L_EOL
    prof = res.profile
    t, p_load, p_bat = prof.t, prof.p, res.p_bat
    out = EmuDesign(cell_id, scenario, S, int(row["P"]), int(row["P"]), False)
    if include_braking is None:
        include_braking = scenario == "profile_braking" and inp.braking.enabled
    if include_braking:
        if braking_at_s is None:
            em0 = HESSEmulator(t, p_load, p_bat, _module(row, cell, sc, int(row["P"]), req, opt, req.I_C0max_A), cfg)
            _, tr0 = em0.run_until_steady(max_cycles=60)
            braking_at_s = float(tr0.t[int(np.argmax(tr0.E_sc))] - tr0.t[0])
            out.notes.append(f"Bremsereignis bei t = {braking_at_s:.1f} s (Zeitpunkt maximaler SC-Energie im "
                             f"periodischen Zyklus, ungünstigster Fall).")
        E_b = res.load.get("E_brake_bus_J", res.load.get("E_brake_J", 0.0))
        p_load = add_braking_event(t, p_load, braking_at_s, E_b, inp.braking.t_brake_s)
        out.braking_at_s = braking_at_s
    P = int(row["P"])
    seen = set()
    L = req.L_EOL if check_eol else 1.0
    for _ in range(max_iter):
        mod = _module(row, cell, sc, P, req, opt, req.I_C0max_A)
        em = HESSEmulator(t, p_load, p_bat, mod, cfg)
        hist, tr = em.run_until_steady()
        last = hist[-1]
        W = last.E_swing_J
        C_req_emu = 2.0 * W / (L * (sc.U_max ** 2 - sc.U_min ** 2))
        C_cell = float(row["C_cell_used_F"])
        P_C_emu = eq.p_capacitance(S, C_req_emu, C_cell) if W > 0 else 1
        I_pk = float(np.max(tr.I_sc_peak))
        P_I_emu = eq.p_current(min(I_pk, req.I_C0max_A), float(cell["i_max_A"]))
        chain = None
        if mod.R_th is not None:
            r_esr = mod.R_cell_Ohm * (cfg.esr_eol_factor if check_eol else 1.0)
            i_cell = em.cell_current_full(tr, p_load, p_bat, prof.dt)
            chain = cell_loss_lowpass(i_cell, prof.dt, r_esr, mod.R_th, mod.C_th, periodic=True)
            chain["t"] = np.arange(len(i_cell)) * prof.dt
            chain["i_cell"] = i_cell
            out.R_esr_cell_used = r_esr
        dTm = chain["dT_max"] if chain else float("nan")
        reasons = []
        if last.t_at_Umin_s > 0:
            reasons.append(f"U_C0min {last.t_at_Umin_s:.2f} s erreicht")
        if last.t_at_Umax_s > 0:
            reasons.append(f"U_C0max {last.t_at_Umax_s:.2f} s erreicht")
        if P < P_C_emu:
            reasons.append(f"P < P_C,emu = {P_C_emu}")
        if P < P_I_emu:
            reasons.append(f"P < P_I,emu = {P_I_emu}")
        if last.t_power_limit_s > 0:
            reasons.append(f"Leistungsgrenze U_C^2/(4R) {last.t_power_limit_s:.2f} s erreicht")
        if dTm == dTm and dTm > req.dT_max_K:
            reasons.append(f"dT = {dTm:.1f} K > {req.dT_max_K:g} K")
        ok = not reasons
        out.iterations.append(EmuIteration(
            P=P, C_module_F=mod.C_module, C_module_eol_F=mod.C_module * L, U_op_min_V=last.U_min,
            U_op_max_V=last.U_max, E_swing_J=W, C_req_emu_F=C_req_emu, P_C_emu=P_C_emu, I_peak_A=I_pk,
            P_I_emu=P_I_emu, I_rms_module_A=last.I_sc_rms_A,
            I_rms_cell_A=chain["I_rms"] if chain else last.I_sc_rms_A / P,
            P_v_cell_mean_W=chain["P_v_mean"] if chain else float("nan"), dT_max_K=dTm,
            t_at_Umin_s=last.t_at_Umin_s, t_at_Umax_s=last.t_at_Umax_s, t_current_limit_s=last.t_current_limit_s,
            E_overflow_J=last.E_overflow_J, E_loss_esr_J=last.E_loss_J, E_loss_conv_J=last.E_conv_loss_J,
            cycles=len(hist), ok=ok, reason="erfüllt" if ok else "; ".join(reasons)))
        out.trace, out.history, out.module, out.chain = tr, hist, mod, chain
        if ok:
            out.P_final, out.feasible = P, True
            break
        seen.add(P)
        P_next = max(P + 1, P_C_emu, P_I_emu)
        if P_next > opt.P_limit:
            out.notes.append("P-Grenze erreicht, keine zulässige Konfiguration gefunden.")
            break
        P = P_next
    out.P_final = P
    if out.iterations and not out.iterations[-1].ok and not out.notes:
        out.notes.append("Maximale Iterationszahl erreicht.")
    return out
