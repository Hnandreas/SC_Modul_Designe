"""Candidate-cell selection and parallel-cell determination (flowchart steps 5 to 11)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import equations as eq
from .config import Options, Requirements
from .database import Database, _isnan
from .profile import Binned
from .thermal import module_current, temperature_rise

C_SOURCES = {
    "datasheet": "Rated datasheet capacitance C_cell",
    "ceq_M1": "Energy-equivalent C_eq,E from characterized C0, k (model M1)",
    "cn_M2": "Measured constant capacitance C_N (model M2)",
}
ESR_SOURCES = {
    "datasheet_dc": "Datasheet ESR (DC), as in IECON 2023",
    "datasheet_ac": "Datasheet ESR (AC, 1 kHz)",
    "measured": "Characterized ESR (method order as configured, plausible values only)",
}
THERMAL_MODES = {
    "skip": "Always skip the thermal steps 7 to 10",
    "available_only": "Only cells with own R_th/C_th (flowchart default)",
    "proxy": "Own R_th/C_th, else same-geometry or next larger characterized cell (with warning)",
}
GEOMETRY_MODES = {
    "corrected": "Module volume = S P pi (D/2)^2 L / packing factor",
    "legacy_iecon": "IECON 2023 script: V = P pi D^2 L (diameter as radius, factor S missing)",
}


@dataclass
class ScenarioContext:
    key: str
    label: str
    E_dch_J: float
    E_ch_J: float
    W_used_J: float
    U_min: float
    U_inM: float
    U_max: float
    C_req: float
    I_C0max: float
    psc: Optional[Binned] = None
    E_ch_label: str = "E_CH,max"


@dataclass
class CandidateDetail:
    cell_id: str
    log: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    current: object = None          # CurrentTrajectory
    dT: Optional[np.ndarray] = None
    dT0: float = 0.0
    fit: Optional[dict] = None
    esr_row: Optional[dict] = None
    thermal: Optional[dict] = None


# ---------------------------------------------------------------------- resolvers
def series_voltage(series: dict, req: Requirements) -> float:
    u = eq.u_cell_max_lifetime(req.t_life_h, req.T_cell_C, series["u0_V"], series["T0_C"], series["t0_h"],
                               series["dU_V"], series["dT_K"])
    return min(u, float(series["u0_V"]))


def resolve_capacitance(cell: dict, opt: Options, db: Database, u_lo: float, u_hi: float,
                        i_cell_rms: Optional[float] = None) -> tuple[float, str, str, Optional[dict]]:
    c_nom = float(cell["c_nom_F"])
    if opt.c_source == "datasheet":
        return c_nom, "datasheet C_cell", "", None
    cur = opt.cfit_current
    if cur == "nearest_rms":
        cur = i_cell_rms if i_cell_rms else "highest"
    fit = db.capacitance_fit(cell["cell_id"], opt.cfit_range, cur, opt.cfit_dataset)
    if fit is None:
        return c_nom, "datasheet C_cell (no characterization in DB)", "", None
    tag = f"Dataset {fit['dataset']}, {fit['voltage_range']} %, {fit['current_A']:g} A"
    if opt.c_source == "cn_M2":
        return float(fit["CN_F"]), f"C_N (M2), {tag}", "", fit
    ceq = eq.c_eq_energy(float(fit["C0_F"]), float(fit["k_FperV"]), u_lo, u_hi)
    warn = ""
    lo_fit, hi_fit = fit["u_lo_frac"] * cell["u_rated_V"], fit["u_hi_frac"] * cell["u_rated_V"]
    if u_lo < lo_fit - 1e-6 or u_hi > hi_fit + 1e-6:
        warn = (f"Cell voltage range {u_lo:.2f} to {u_hi:.2f} V exceeds the fitted range "
                f"{lo_fit:.2f} to {hi_fit:.2f} V (extrapolated C(U)).")
    return ceq, f"C_eq,E = {ceq:.2f} F from C0 = {fit['C0_F']:.2f} F, k = {fit['k_FperV']:.2f} F/V ({tag})", warn, fit


def resolve_esr(cell: dict, opt: Options, db: Database) -> tuple[float, str, Optional[dict]]:
    """Returns ESR in Ohm."""
    if opt.esr_source == "datasheet_ac" and not _isnan(cell.get("esr_ac_mOhm")):
        return float(cell["esr_ac_mOhm"]) / 1000, "datasheet ESR (AC)", None
    if opt.esr_source == "measured":
        row = db.esr_measured(cell["cell_id"], tuple(opt.esr_methods), opt.esr_current)
        if row:
            return (float(row["esr_mOhm"]) / 1000,
                    f"measured {row['method']}, Dataset {row['dataset']}, {row['current_A']:g} A", row)
        return float(cell["esr_dc_mOhm"]) / 1000, "datasheet ESR (DC), no measurement in DB", None
    return float(cell["esr_dc_mOhm"]) / 1000, "datasheet ESR (DC)", None


def resolve_thermal(cell: dict, mode: str, db: Database) -> Optional[dict]:
    if mode == "skip":
        return None
    if mode == "available_only":
        return db.thermal_own(cell["cell_id"])
    if mode == "proxy":
        return db.thermal_proxy(cell["cell_id"])
    raise ValueError(f"unknown thermal mode {mode}")


def module_geometry(cell: dict, S: int, P: int, opt: Options) -> tuple[float, float, float]:
    """mass [kg], volume [dm^3], density [kg/dm^3]."""
    m, d, l = cell.get("mass_g"), cell.get("diameter_mm"), cell.get("length_mm")
    mass = S * P * m / 1000 if not _isnan(m) else np.nan
    if _isnan(d) or _isnan(l):
        return mass, np.nan, np.nan
    if opt.geometry == "legacy_iecon":
        vol = P * math.pi * d ** 2 * l / 1e6
    else:
        vol = S * P * math.pi * (d / 2) ** 2 * l / 1e6 / opt.packing_factor
    return mass, vol, mass / vol if vol else np.nan


# ---------------------------------------------------------------------- candidate
def evaluate_candidate(cell: dict, sc: ScenarioContext, S: int, u_cell_max_life: float, req: Requirements,
                       opt: Options, db: Database, thermal_mode: Optional[str] = None,
                       thermal_override: Optional[dict] = None, P_min: int = 1) -> tuple[dict, CandidateDetail]:
    """Steps 5 to 11 for one cell.

    thermal_override: {'rth': K/W, 'cth': J/K, 'note': ...} to use own values (cell detail view).
    P_min: force at least this number of parallel strings.
    """
    det = CandidateDetail(cell["cell_id"])
    thermal_mode = thermal_mode or opt.thermal_mode
    u_lo, u_hi = sc.U_min / S, sc.U_max / S

    # conservative RMS estimate for 'nearest_rms' current selection
    i_rms_est = None
    if sc.psc is not None:
        i_mod_rms_cons = math.sqrt(float(np.mean(sc.psc.mean_sq))) / sc.U_min
        p0 = eq.p_capacitance(S, sc.C_req, float(cell["c_nom_F"]))
        i_rms_est = i_mod_rms_cons / max(p0, eq.p_current(sc.I_C0max, float(cell["i_max_A"])))

    C_cell, c_note, c_warn, fit = resolve_capacitance(cell, opt, db, u_lo, u_hi, i_rms_est)
    det.fit = fit
    if c_warn:
        det.warnings.append(c_warn)
    R_cell, r_note, r_row = resolve_esr(cell, opt, db)
    det.esr_row = r_row

    P_C = eq.p_capacitance(S, sc.C_req, C_cell)
    P_I = eq.p_current(sc.I_C0max, float(cell["i_max_A"]))
    P = max(P_C, P_I, int(P_min))
    det.log.append(f"Step 6: P_C = ceil({S} x {sc.C_req:.3f} F / {C_cell:.2f} F) = {P_C}, "
                   f"P_I = ceil({sc.I_C0max:g} A / {cell['i_max_A']:g} A) = {P_I} -> p = {P}")

    th = thermal_override if thermal_override else resolve_thermal(cell, thermal_mode, db)
    det.thermal = th
    P_T = np.nan
    dTmax = np.nan
    i_mod_rms = np.nan
    i_cell_rms_max = np.nan
    thermal_status = "skipped"
    feasible = True
    status = "feasible"
    traj = None
    if th is None:
        det.log.append("Step 7: R_th, C_th not available -> thermal steps 8 to 10 skipped")
    elif sc.psc is None:
        det.log.append("Step 7: R_th, C_th available, but no load profile -> thermal steps skipped")
        thermal_status = "skipped (no profile)"
    else:
        if th.get("warning"):
            det.warnings.append(th["warning"])
        rth, cth = th["rth"], th["cth"]
        det.log.append(f"Step 7: R_th = {rth:g} K/W, C_th = {cth:g} J/K ({th.get('note', '')})")

        def trajectory(p_par: int):
            C_mod = eq.module_capacitance(p_par, S, C_cell)
            R_mod = eq.module_esr(p_par, S, R_cell)
            return module_current(sc.psc, opt.thermal_current, sc.U_min, sc.U_inM, sc.U_max, C_mod, R_mod)

        traj = trajectory(P)
        i_mod_rms = traj.I_rms
        i_cell_rms_max = eq.i_cell_rms_max(req.dT_max_K, R_cell, rth)
        P_T = eq.p_thermal(i_mod_rms, i_cell_rms_max)
        det.log.append(f"Step 8: I_module,RMS = {i_mod_rms:.2f} A, I_cell,RMS,max = sqrt({req.dT_max_K:g} K / "
                       f"({R_cell * 1000:.1f} mOhm x {rth:g} K/W)) = {i_cell_rms_max:.2f} A -> P_T = {P_T}")
        if P_T > P:
            P = P_T
            traj = trajectory(P)
        thermal_status = "ok"
        while True:
            ploss_cell = R_cell * traj.I_sq / P ** 2
            dT, dT0 = temperature_rise(ploss_cell, sc.psc.dt, rth, cth, opt.thermal_profile)
            dTmax = float(np.max(dT))
            det.log.append(f"Step 9: p = {P}: max dT = {dTmax:.2f} K (limit {req.dT_max_K:g} K, "
                           f"{opt.thermal_profile})")
            if dTmax <= req.dT_max_K:
                break
            if P >= opt.P_limit:
                feasible, status, thermal_status = False, "infeasible (thermal, P limit)", "failed"
                break
            P += 1
            thermal_status = "P increased"
            traj = trajectory(P)
        det.current, det.dT, det.dT0 = traj, dT, dT0
        if th.get("proxy_of"):
            thermal_status += " (proxy)"
        for n in traj.notes:
            det.warnings.append(n)

    C_mod = eq.module_capacitance(P, S, C_cell)
    R_mod = eq.module_esr(P, S, R_cell)
    mass, vol, dens = module_geometry(cell, S, P, opt)
    price = cell.get("price_EUR")
    cost = S * P * price if not _isnan(price) else np.nan
    E_use_eol = eq.usable_energy(C_mod, sc.U_max, sc.U_min, req.L_EOL)
    if traj is not None:
        p_loss = float(np.mean(traj.I_sq)) * R_mod
        i_rms_out = traj.I_rms
    elif sc.psc is not None:
        i_rms_out = math.sqrt(float(np.mean(sc.psc.mean_sq))) / sc.U_min
        p_loss = i_rms_out ** 2 * R_mod
    else:
        i_rms_out, p_loss = np.nan, np.nan
    if E_use_eol + 1e-6 < sc.W_used_J:
        det.warnings.append("Usable EOL energy below requirement.")
    row = dict(
        cell_id=cell["cell_id"], manufacturer=cell.get("manufacturer"), c_nom_F=float(cell["c_nom_F"]),
        S=S, P=P, P_C=P_C, P_I=P_I, P_T=P_T, setup=f"{P}/{S}", n_cells=S * P,
        u_cell_max_life_V=u_cell_max_life, C_cell_used_F=C_cell, C_source=c_note,
        ESR_cell_mOhm=R_cell * 1000, ESR_source=r_note,
        C_module_F=C_mod, ESR_module_mOhm=R_mod * 1000, I_module_max_A=P * float(cell["i_max_A"]),
        mass_kg=mass, volume_dm3=vol, density_kg_dm3=dens, cost_EUR=cost,
        E_usable_EOL_J=E_use_eol, energy_margin=E_use_eol / sc.W_used_J - 1 if sc.W_used_J else np.nan,
        E_density_Wh_kg=(E_use_eol / req.L_EOL / 3600 / mass) if mass and not _isnan(mass) else np.nan,
        I_module_rms_A=i_rms_out, P_loss_avg_W=p_loss,
        thermal_status=thermal_status, R_th_KperW=th["rth"] if th else np.nan,
        C_th_JperK=th["cth"] if th else np.nan, dT_max_K=dTmax, I_cell_rms_max_A=i_cell_rms_max,
        feasible=feasible, status=status, warnings=" | ".join(det.warnings),
    )
    return row, det


def evaluate_all(cells: pd.DataFrame, sc: ScenarioContext, series_info: dict, req: Requirements, opt: Options,
                 db: Database) -> tuple[pd.DataFrame, dict]:
    rows, details = [], {}
    for _, c in cells.iterrows():
        cell = c.to_dict()
        sid = opt.series_override or cell["lifetime_series_id"]
        si = series_info[sid]
        row, det = evaluate_candidate(cell, sc, si["S"], si["U_C01max"], req, opt, db)
        row["lifetime_series_id"] = sid
        rows.append(row)
        details[cell["cell_id"]] = det
    return pd.DataFrame(rows), details
