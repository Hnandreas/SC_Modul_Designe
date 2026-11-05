"""Continuous, PI-controlled emulation check of the existing module (step 14).

S/P and all analytic sizing inputs are retained. Profiles repeat with periodic
filter continuation; energy, PI and thermal states continue across periods.
Voltage limits curtail actual SC current; ESR losses always reduce stored energy.
The explicit increase_P study remains opt-in and is not used by the GUI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import equations as eq
from .database import Database, _isnan, get_database
from .emulation import CycleTrace, EmulationConfig, HESSEmulator, ModuleModel
from .selection import coupling_factor, layout_for, network_params
from .splits import periodic_emulation_split
from .thermal import cell_loss_lowpass
from . import thermal_network as TN


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
    dT_max_K: float                 # criterion: hottest cell of the network (if active) else block model
    dT_block_max_K: float
    dT_net_max_K: float
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
    net: Optional[dict] = None                # periodic network response (thermal network active)
    braking_at_s: Optional[float] = None
    notes: list = field(default_factory=list)
    info: list = field(default_factory=list)  # hints that are no failure (U at U_C0min, current limit)
    emulator: Optional[HESSEmulator] = None
    converged: bool = False
    study: list = field(default_factory=list)  # dT over P for the display (P_final unchanged)


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
                     db: Optional[Database] = None, P: Optional[int] = None, increase_P: bool = False,
                     single_pass: bool = False, I_C0max: Optional[float] = None,
                     thermal_study: int = 0) -> EmuDesign:
    """Emulation check of a configuration.

    Default: the module of step 5 (P from the flowchart, or P given) is only checked, P is NOT changed.
    Default: carry energy and PI state across repeated periods until periodic.
    single_pass=True is an explicitly requested startup study, not the GUI default.
    thermal_study is opt-in and never changes the selected module.
    """
    db = db or get_database()
    inp, req, opt = res.inputs, res.inputs.req, res.inputs.opt
    i_lim = float(I_C0max) if I_C0max else float(req.I_C0max_A)
    sc = res.scenarios[scenario]
    df = res.candidates[scenario]
    row = df[df.cell_id == cell_id].iloc[0].to_dict()
    cell = db.cell(cell_id)
    S = int(row["S"])
    if cfg is None:
        cfg = EmulationConfig(eta_dch=req.eta_dch, eta_ch=req.eta_ch, U_bus_V=req.U_bus_V)
    if cfg.U_conv_min_V is None:
        cfg.U_conv_min_V = req.U_conv_min_V
    cfg.eol = check_eol
    cfg.L_EOL = req.L_EOL
    prof = res.profile
    t, p_load = prof.t, prof.p
    p_bat, _ = periodic_emulation_split(inp.split.name, t, p_load, inp.split.params)
    P0 = int(P) if P else int(row["P"])
    out = EmuDesign(cell_id, scenario, S, P0, P0, False)
    if inp.split.name in ("moving_average", "lowpass_1st", "butterworth", "rate_limiter"):
        out.notes.append("Emulation: kausale periodische Filterfortsetzung ohne Rand-Resets; "
                         "die analytische Auslegung und ihre Referenzaufteilung bleiben unverändert.")
    if include_braking is None:
        include_braking = False  # Synthetic events require an explicit API request.
    if include_braking:
        if braking_at_s is None:
            em0 = HESSEmulator(t, p_load, p_bat, _module(row, cell, sc, P0, req, opt, i_lim), cfg)
            _, tr0 = em0.run_cycle() if single_pass else em0.run_until_steady(max_cycles=60)
            braking_at_s = float(tr0.t[int(np.argmax(tr0.E_sc))] - tr0.t[0])
            out.notes.append(f"Bremsereignis bei t = {braking_at_s:.1f} s (Zeitpunkt maximaler SC-Energie, "
                             f"ungünstigster Fall).")
        E_b = res.load.get("E_brake_bus_J", res.load.get("E_brake_J", 0.0))
        p_load = add_braking_event(t, p_load, braking_at_s, E_b, inp.braking.t_brake_s)
        out.braking_at_s = braking_at_s
    P = P0
    seen = set()
    study_mode = False
    L = cfg.aging_capacity_factor if cfg.aging_capacity_factor is not None else (req.L_EOL if check_eol else 1.0)
    for _ in range(max_iter):
        mod = _module(row, cell, sc, P, req, opt, i_lim)
        em = HESSEmulator(t, p_load, p_bat, mod, cfg)
        if single_pass:
            s1, tr = em.run_cycle()
            hist = [s1]
        else:
            hist, tr = em.run_until_steady()
        last = hist[-1]
        W = last.E_swing_J
        C_req_emu = 2.0 * W / (L * (sc.U_max ** 2 - sc.U_min ** 2))
        C_cell = float(row["C_cell_used_F"])
        P_C_emu = eq.p_capacitance(S, C_req_emu, C_cell) if W > 0 else 1
        I_pk = float(np.max(tr.I_sc_peak))
        P_I_emu = eq.p_current(min(I_pk, i_lim), float(cell["i_max_A"]))
        chain = None
        netr = None
        if mod.R_th is not None:
            # same loss assumption as the thermal page: ESR x thermal_esr_factor
            r_esr = em.R_cell_eff
            i_cell = em.cell_current_full(tr, p_load, p_bat, prof.dt)
            chain = cell_loss_lowpass(i_cell, prof.dt, r_esr, mod.R_th, mod.C_th, periodic=True)
            chain["t"] = np.arange(len(i_cell)) * prof.dt
            chain["i_cell"] = i_cell
            out.R_esr_cell_used = r_esr
            if opt.thermal_network and opt.geometry == "pcb":
                lay = layout_for(cell, S, P, opt, req)
                if lay is not None:
                    th = dict(rth=float(row["R_th_KperW"]), cth=float(row["C_th_JperK"]))
                    netr = TN.periodic_response(TN.layout_grid(lay), network_params(cell, th, opt, req),
                                                chain["t"], chain["p_v"])
        dT_block = chain["dT_max"] if chain else float("nan")
        dT_net = netr["dT_max"] if netr else float("nan")
        dTm = dT_net if netr else dT_block
        reasons = []
        if not single_pass and not em.converged:
            reasons.append("Periodischer Zustand bei maximaler Zykluszahl nicht erreicht")
        if increase_P and last.t_at_Umin_s > 0:
            reasons.append(f"U_C0min {last.t_at_Umin_s:.2f} s erreicht")
        if increase_P and last.t_at_Umax_s > 0:
            reasons.append(f"U_C0max {last.t_at_Umax_s:.2f} s erreicht")
        if increase_P and P < P_C_emu:
            reasons.append(f"P < P_C,emu = {P_C_emu}")
        if increase_P and P < P_I_emu:
            reasons.append(f"P < P_I,emu = {P_I_emu}")
        # current limit I_C0max (hard) and power limit U_C^2/(4R): the SC delivers what it can, the battery the
        # rest (overflow). Reported, but not a reason to increase P.
        if em.U_reference > cfg.converter_max_duty * cfg.U_bus_V + 1e-9:
            reasons.append(f"Spannungsreferenz {em.U_reference:.2f} V bei U_Bus={cfg.U_bus_V:g} V "
                           f"nicht erreichbar (Wandlergrenze {cfg.converter_max_duty * cfg.U_bus_V:.2f} V)")
        if last.t_converter_range_s > 0:
            reasons.append(f"SC-Klemmenspannung oberhalb des regelbaren Wandlerbereichs "
                           f"({cfg.converter_max_duty * cfg.U_bus_V:g} V) für {last.t_converter_range_s:.2f} s")
        if last.t_Uconv_min_s > 0:
            reasons.append(f"Klemmenspannung < U_Wandler,min = {cfg.U_conv_min_V:g} V für {last.t_Uconv_min_s:.2f} s")
        if dTm == dTm and dTm > req.dT_max_K:
            reasons.append(f"dT = {dTm:.1f} K > {req.dT_max_K:g} K" + (" (Netzwerk, heißeste Zelle)" if netr else ""))
        info = []
        if last.t_bus_voltage_limit_s > 0:
            info.append(f"Ladestrom für {last.t_bus_voltage_limit_s:.2f} s durch "
                        f"Wandler-Spannungsreserve begrenzt (D_max={cfg.converter_max_duty:g}, "
                        f"U_Bus={cfg.U_bus_V:g} V); Batterie übernimmt den Rest")
        if last.t_at_Umin_s > 0:
            info.append(f"an U_C0min {last.t_at_Umin_s:.2f} s, SC-Entladung wird begrenzt, Batterie übernimmt den fehlenden Anteil")
        if last.t_current_limit_s > 0:
            info.append(f"an I_C0max = {i_lim:g} A {last.t_current_limit_s:.2f} s, Batterie übernimmt anteilig")
        if last.t_at_Umax_s > 0:
            info.append(f"an U_C0max {last.t_at_Umax_s:.2f} s, Batterie nimmt den Rest")
        ok = not reasons
        it = (EmuIteration(
            P=P, C_module_F=mod.C_module, C_module_eol_F=mod.C_module * L, U_op_min_V=last.U_min,
            U_op_max_V=last.U_max, E_swing_J=W, C_req_emu_F=C_req_emu, P_C_emu=P_C_emu, I_peak_A=I_pk,
            P_I_emu=P_I_emu, I_rms_module_A=last.I_sc_rms_A,
            I_rms_cell_A=chain["I_rms"] if chain else last.I_sc_rms_A / P,
            P_v_cell_mean_W=chain["P_v_mean"] if chain else float("nan"), dT_max_K=dTm,
            dT_block_max_K=dT_block, dT_net_max_K=dT_net,
            t_at_Umin_s=last.t_at_Umin_s, t_at_Umax_s=last.t_at_Umax_s, t_current_limit_s=last.t_current_limit_s,
            E_overflow_J=last.E_overflow_J, E_loss_esr_J=last.E_loss_J, E_loss_conv_J=last.E_conv_loss_J,
            cycles=len(hist), ok=ok, reason="erfüllt" if ok else "; ".join(reasons)))
        if study_mode:                               # display only: dT over P, the checked module stays P0
            out.study.append(it)
            if not (dTm == dTm) or dTm <= req.dT_max_K or P >= min(P0 + thermal_study, opt.P_limit):
                break
            P += 1
            continue
        out.info = info
        out.iterations.append(it)
        out.trace, out.history, out.module, out.chain, out.net = tr, hist, mod, chain, netr
        out.emulator, out.converged = em, em.converged
        if ok or not increase_P:
            out.P_final, out.feasible = P, ok
            if not increase_P and thermal_study > 0 and dTm == dTm and dTm > req.dT_max_K and P < opt.P_limit:
                out.study.append(it)
                study_mode = True
                P += 1
                continue
            break
        seen.add(P)
        P_next = max(P + 1, P_C_emu, P_I_emu)
        if P_next > opt.P_limit:
            out.notes.append("P-Grenze erreicht, keine zulässige Konfiguration gefunden.")
            break
        P = P_next
    if increase_P:
        out.P_final = P
    if increase_P and out.iterations and not out.iterations[-1].ok and not out.notes:
        out.notes.append("Maximale Iterationszahl erreicht.")
    return out
