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
from .layout import pcb_layout
from .thermal import (block_coupling_factor, block_layout, cell_surface_mm2, module_current, scale_thermal,
                      temperature_rise)

C_SOURCES = {
    "datasheet": "Datenblatt-Kapazität C_cell (IECON 2023)",
    "ceq_M1": "Energieäquivalente C_eq,E aus gemessenem C0, k (Modell M1)",
    "cn_M2": "Gemessene konstante Kapazität C_N (Modell M2)",
}
ESR_SOURCES = {
    "datasheet_dc": "Datenblatt-ESR (DC), wie IECON 2023",
    "datasheet_ac": "Datenblatt-ESR (AC, 1 kHz)",
    "measured": "Gemessener ESR (Methodenreihenfolge, nur plausible Werte)",
}
THERMAL_MODES = {
    "skip": "Thermik immer überspringen",
    "available_only": "Nur Zellen mit eigenen R_th/C_th (Flussdiagramm)",
    "proxy_scaled": "Eigene Werte, sonst von charakterisierter Zelle auf die Zellgröße skaliert (Warnung)",
    "proxy": "Eigene Werte, sonst unverändert von gleicher/nächst größerer Zelle übernommen (Warnung)",
}
GEOMETRY_MODES = {
    "pcb": "Leiterplatten-Layout (hexagonal oder quadratisch), Volumen = Platine x Bauhöhe",
    "corrected": "Reines Zellvolumen S P pi (D/2)^2 L / Packungsfaktor",
    "legacy_iecon": "IECON-2023-Skript: V = P pi D^2 L (Durchmesser als Radius, Faktor S fehlt)",
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
    breakdown: list = field(default_factory=list)


# ---------------------------------------------------------------------- resolvers
def series_voltage(series: dict, req: Requirements) -> float:
    u = eq.u_cell_max_lifetime(req.t_life_h, req.T_cell_C, series["u0_V"], series["T0_C"], series["t0_h"],
                               series["dU_V"], series["dT_K"])
    return min(u, float(series["u0_V"]))


def resolve_capacitance(cell: dict, opt: Options, db: Database, u_lo: float, u_hi: float,
                        i_cell_rms: Optional[float] = None) -> tuple[float, str, str, Optional[dict]]:
    c_nom = float(cell["c_nom_F"])
    if opt.c_source == "datasheet":
        return c_nom, "Datenblatt", "", None
    cur = opt.cfit_current
    if cur == "nearest_rms":
        cur = i_cell_rms if i_cell_rms else "highest"
    fit = db.capacitance_fit(cell["cell_id"], opt.cfit_range, cur, opt.cfit_dataset)
    if fit is None:
        return c_nom, "Datenblatt (keine Charakterisierung)", "", None
    tag = f"Datensatz {fit['dataset']}, {fit['voltage_range']} %, {fit['current_A']:g} A"
    if opt.c_source == "cn_M2":
        return float(fit["CN_F"]), f"C_N (M2), {tag}", "", fit
    ceq = eq.c_eq_energy(float(fit["C0_F"]), float(fit["k_FperV"]), u_lo, u_hi)
    warn = ""
    lo_fit, hi_fit = fit["u_lo_frac"] * cell["u_rated_V"], fit["u_hi_frac"] * cell["u_rated_V"]
    if u_lo < lo_fit - 1e-6 or u_hi > hi_fit + 1e-6:
        warn = (f"Zellspannung {u_lo:.2f} bis {u_hi:.2f} V liegt außerhalb des Fit-Bereichs "
                f"{lo_fit:.2f} bis {hi_fit:.2f} V (C(U) extrapoliert).")
    return ceq, f"C_eq,E = {ceq:.2f} F aus C0 = {fit['C0_F']:.2f} F, k = {fit['k_FperV']:.2f} F/V ({tag})", warn, fit


def resolve_esr(cell: dict, opt: Options, db: Database) -> tuple[float, str, Optional[dict]]:
    """Returns ESR in Ohm."""
    if opt.esr_source == "datasheet_ac" and not _isnan(cell.get("esr_ac_mOhm")):
        return float(cell["esr_ac_mOhm"]) / 1000, "Datenblatt (AC)", None
    if opt.esr_source == "measured":
        row = db.esr_measured(cell["cell_id"], tuple(opt.esr_methods), opt.esr_current)
        if row:
            return (float(row["esr_mOhm"]) / 1000,
                    f"gemessen {row['method']}, Datensatz {row['dataset']}, {row['current_A']:g} A", row)
        return float(cell["esr_dc_mOhm"]) / 1000, "Datenblatt (DC), keine Messung", None
    return float(cell["esr_dc_mOhm"]) / 1000, "Datenblatt (DC)", None


def resolve_thermal(cell: dict, mode: str, db: Database) -> Optional[dict]:
    if mode == "skip":
        return None
    if mode == "available_only":
        return db.thermal_own(cell["cell_id"])
    if mode in ("proxy", "proxy_scaled"):
        th = db.thermal_proxy(cell["cell_id"])
        if th and th.get("proxy_of") and mode == "proxy_scaled":
            ref = db.cell(th["proxy_of"])
            if not any(_isnan(x.get(k)) for x in (ref, cell) for k in ("diameter_mm", "length_mm", "mass_g")):
                th = dict(th)
                th["rth_ref"], th["cth_ref"] = th["rth"], th["cth"]
                th["rth"], th["cth"] = scale_thermal(th["rth"], th["cth"], ref, cell)
                th["warning"] = (f"R_th/C_th von {th['proxy_of']} auf die Zellgröße skaliert "
                                 f"(R_th ~ 1/Oberfläche, C_th ~ Masse): {th['rth_ref']:.1f} -> {th['rth']:.1f} K/W, "
                                 f"{th['cth_ref']:.1f} -> {th['cth']:.1f} J/K. Nicht gemessen.")
                th["note"] = f"skaliert von {th['proxy_of']}"
        return th
    raise ValueError(f"unknown thermal mode {mode}")


def _connectors(opt: Options) -> tuple:
    out = []
    if opt.conn_power_w_mm and opt.conn_power_d_mm:
        out.append(("Leistung", opt.conn_power_w_mm, opt.conn_power_d_mm))
    if opt.conn_signal_w_mm and opt.conn_signal_d_mm:
        out.append(("Signal", opt.conn_signal_w_mm, opt.conn_signal_d_mm))
    return tuple(out)


def layout_for(cell: dict, S: int, P: int, opt: Options, req: Optional[Requirements] = None,
               packing: Optional[str] = None, bands: Optional[int] = None, group_shape: Optional[str] = None):
    """PCB layout of the module (U path, connectors in the free space, orientation and size limits)."""
    d, l = cell.get("diameter_mm"), cell.get("length_mm")
    if _isnan(d) or _isnan(l):
        return None
    return pcb_layout(S, P, float(d), float(l), packing or opt.packing, opt.cell_spacing_mm, opt.pcb_margin_mm,
                      opt.pcb_bands if bands is None else bands, 1.6, opt.pcb_clearance_mm, opt.pcb_orientation,
                      group_shape or opt.pcb_group_shape, _connectors(opt),
                      getattr(req, "max_width_mm", None), getattr(req, "max_depth_mm", None),
                      opt.pcb_target_aspect)


def layout_pair(cell: dict, S: int, P: int, opt: Options, req: Optional[Requirements] = None):
    """Square and hexagonal packing with the same arrangement (for the comparison figure)."""
    main = layout_for(cell, S, P, opt, req)
    if main is None:
        return None, None
    other = "square" if main.packing == "hex" else "hex"
    sec = layout_for(cell, S, P, opt, req, packing=other, bands=main.bands, group_shape=f"{main.gx}x{main.gy}")
    return (main, sec) if main.packing == "square" else (sec, main)


def coupling_factor(cell: dict, n_cells: int, opt: Options, S: Optional[int] = None,
                    req: Optional[Requirements] = None) -> float:
    """k = n A_cell / A_block. With S given, the block is the PCB layout box, else a square grid."""
    if opt.thermal_coupling == "single_cell":
        return 1.0
    if opt.thermal_coupling == "factor":
        return float(opt.thermal_coupling_factor)
    d, l = cell.get("diameter_mm"), cell.get("length_mm")
    if _isnan(d) or _isnan(l):
        return 1.0
    if S:
        lay = layout_for(cell, S, max(1, n_cells // S), opt, req)
        return max(1.0, n_cells * cell_surface_mm2(float(d), float(l)) / lay.box_area_mm2)
    return block_coupling_factor(n_cells, float(d), float(l), opt.cell_spacing_mm)


def module_geometry(cell: dict, S: int, P: int, opt: Options, req: Optional[Requirements] = None
                    ) -> tuple[float, float, float]:
    """mass [kg], volume [dm^3], density [kg/dm^3]. Geometry 'pcb': module = cells + PCB, volume = PCB box."""
    m, d, l = cell.get("mass_g"), cell.get("diameter_mm"), cell.get("length_mm")
    mass = S * P * m / 1000 if not _isnan(m) else np.nan
    if _isnan(d) or _isnan(l):
        return mass, np.nan, np.nan
    if opt.geometry == "legacy_iecon":
        vol = P * math.pi * d ** 2 * l / 1e6
    elif opt.geometry == "pcb":
        lay = layout_for(cell, S, P, opt, req)
        vol = lay.volume_dm3
        if mass == mass:
            mass += lay.pcb_mass_kg(opt.pcb_areal_mass_g_cm2)
    else:
        vol = S * P * math.pi * (d / 2) ** 2 * l / 1e6 / opt.packing_factor
    return mass, vol, mass / vol if vol else np.nan


def spec_violations(row: dict, lay, req: Requirements) -> list[str]:
    """Specification limits (Vorgaben) that a configuration violates."""
    out = []
    checks = [("max_width_mm", lay.width_mm if lay else np.nan, "Breite", "mm"),
              ("max_depth_mm", lay.depth_mm if lay else np.nan, "Tiefe", "mm"),
              ("max_height_mm", lay.height_mm if lay else np.nan, "Höhe", "mm"),
              ("max_volume_dm3", row.get("volume_dm3"), "Volumen", "dm³"),
              ("max_mass_kg", row.get("mass_kg"), "Masse", "kg"),
              ("max_cost_EUR", row.get("cost_EUR"), "Kosten", "€")]
    for key, val, name, unit in checks:
        lim = getattr(req, key, None)
        if lim is not None and val is not None and val == val and val > lim + 1e-9:
            out.append(f"{name} {val:.1f} > {lim:g} {unit}")
    return out


def cell_temperature(traj, R_cell: float, P: int, rth: float, cth: float, dt: float, mode: str):
    """dT(t) of one cell for P parallel strings (equal current sharing)."""
    ploss = R_cell * traj.I_sq / P ** 2
    return temperature_rise(ploss, dt, rth, cth, mode)


def thermal_breakdown(cell: dict, traj, S: int, P: int, R_cell_bol: float, th: dict, opt: Options,
                      dt: float, req: Optional[Requirements] = None) -> list:
    """dT_max step by step: measurement condition -> size scaling -> module packing -> end of life."""
    steps = []
    rth0 = th.get("rth_ref", th["rth"])
    cth0 = th.get("cth_ref", th["cth"])
    dT, _ = cell_temperature(traj, R_cell_bol, P, rth0, cth0, dt, opt.thermal_profile)
    src = th.get("proxy_of") or cell["cell_id"]
    steps.append(("Single cell, still air (measured $R_{th}$)" + (f", from {src.replace('VISHAY_HVR235_', '')}" if th.get("proxy_of") else ""),
                  float(dT.max()), rth0, R_cell_bol))
    rth, cth = th["rth"], th["cth"]
    if "rth_ref" in th:
        dT, _ = cell_temperature(traj, R_cell_bol, P, rth, cth, dt, opt.thermal_profile)
        steps.append(("+ $R_{th}$, $C_{th}$ scaled to cell size", float(dT.max()), rth, R_cell_bol))
    k = coupling_factor(cell, S * P, opt, S, req)
    if k > 1.0 + 1e-9:
        dT, _ = cell_temperature(traj, R_cell_bol, P, rth * k, cth, dt, opt.thermal_profile)
        lab = "+ packed module block" if opt.thermal_coupling == "packed_block" else "+ coupling factor"
        steps.append((f"{lab} (k = {k:.2f})", float(dT.max()), rth * k, R_cell_bol))
    if abs(opt.thermal_esr_factor - 1.0) > 1e-9:
        dT, _ = cell_temperature(traj, R_cell_bol * opt.thermal_esr_factor, P, rth * k, cth, dt,
                                 opt.thermal_profile)
        steps.append((f"+ end-of-life ESR (x {opt.thermal_esr_factor:g})", float(dT.max()), rth * k,
                      R_cell_bol * opt.thermal_esr_factor))
    return steps


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
    det.log.append(f"Schritt 6: P_C = ceil({S} x {sc.C_req:.3f} F / {C_cell:.2f} F) = {P_C}, "
                   f"P_I = ceil({sc.I_C0max:g} A / {cell['i_max_A']:g} A) = {P_I}  ->  p = {P}")

    th = thermal_override if thermal_override else resolve_thermal(cell, thermal_mode, db)
    det.thermal = th
    P_T = np.nan
    dTmax = np.nan
    i_cell_rms_max = np.nan
    k_th = np.nan
    rth_eff = np.nan
    thermal_status = "übersprungen"
    feasible = True
    status = "zulässig"
    traj = None
    R_th_loss = R_cell * opt.thermal_esr_factor
    if th is None:
        det.log.append("Schritt 7: R_th, C_th nicht vorhanden -> Schritte 8 bis 10 übersprungen")
    elif sc.psc is None:
        det.log.append("Schritt 7: R_th, C_th vorhanden, aber kein Lastprofil -> Schritte 8 bis 10 übersprungen")
        thermal_status = "übersprungen (kein Profil)"
    else:
        if th.get("warning"):
            det.warnings.append(th["warning"])
        rth, cth = th["rth"], th["cth"]

        def trajectory(p_par: int):
            C_mod = eq.module_capacitance(p_par, S, C_cell)
            R_mod = eq.module_esr(p_par, S, R_cell)
            return module_current(sc.psc, opt.thermal_current, sc.U_min, sc.U_inM, sc.U_max, C_mod, R_mod)

        k_th = coupling_factor(cell, S * P, opt, S, req)
        rth_eff = rth * k_th
        det.log.append(f"Schritt 7: R_th = {rth:.1f} K/W, C_th = {cth:.1f} J/K ({th.get('note', '')}); "
                       f"Kopplung k = {k_th:.2f} -> R_th,eff = {rth_eff:.1f} K/W; "
                       f"ESR für Verluste = {R_th_loss * 1000:.1f} mOhm (Faktor {opt.thermal_esr_factor:g})")
        traj = trajectory(P)
        i_cell_rms_max = eq.i_cell_rms_max(req.dT_max_K, R_th_loss, rth_eff)
        P_T = eq.p_thermal(traj.I_rms, i_cell_rms_max)
        det.log.append(f"Schritt 8: I_Modul,RMS = {traj.I_rms:.2f} A, I_Zelle,RMS,max = sqrt({req.dT_max_K:g} K / "
                       f"({R_th_loss * 1000:.1f} mOhm x {rth_eff:.1f} K/W)) = {i_cell_rms_max:.2f} A -> P_T = {P_T}")
        if P_T > P:
            P = P_T
            traj = trajectory(P)
        thermal_status = "ok"
        while True:
            k_th = coupling_factor(cell, S * P, opt, S, req)
            rth_eff = rth * k_th
            dT, dT0 = cell_temperature(traj, R_th_loss, P, rth_eff, cth, sc.psc.dt, opt.thermal_profile)
            dTmax = float(np.max(dT))
            det.log.append(f"Schritt 9: p = {P}: I_Zelle,RMS = {traj.I_rms / P:.2f} A, max dT = {dTmax:.1f} K "
                           f"(Grenze {req.dT_max_K:g} K, {opt.thermal_profile})")
            if dTmax <= req.dT_max_K:
                det.log.append("Schritt 10: zulässig")
                break
            if P >= opt.P_limit:
                feasible, status, thermal_status = False, "unzulässig (Thermik, P-Grenze)", "nicht erfüllt"
                break
            P += 1
            thermal_status = "P erhöht"
            traj = trajectory(P)
        det.current, det.dT, det.dT0 = traj, dT, dT0
        det.breakdown = thermal_breakdown(cell, traj, S, P, R_cell, th, opt, sc.psc.dt, req)
        if th.get("proxy_of"):
            thermal_status += " (Proxy)"
        for n in traj.notes:
            det.warnings.append(n)
        if req.T_amb_C + dTmax > req.T_cell_C + 1e-9:
            det.warnings.append(f"T_amb + dT_max = {req.T_amb_C + dTmax:.1f} °C liegt über der Zelltemperatur "
                                f"{req.T_cell_C:g} °C der Lebensdauerauslegung.")

    C_mod = eq.module_capacitance(P, S, C_cell)
    R_mod = eq.module_esr(P, S, R_cell)
    mass, vol, dens = module_geometry(cell, S, P, opt, req)
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
        det.warnings.append("Nutzbare Energie am Lebensende kleiner als gefordert.")
    lay = layout_for(cell, S, P, opt, req) if opt.geometry == "pcb" else None
    row = dict(
        cell_id=cell["cell_id"], manufacturer=cell.get("manufacturer"), c_nom_F=float(cell["c_nom_F"]),
        S=S, P=P, P_C=P_C, P_I=P_I, P_T=P_T, setup=f"{P}/{S}", n_cells=S * P,
        u_cell_max_life_V=u_cell_max_life, C_cell_used_F=C_cell, C_source=c_note,
        ESR_cell_mOhm=R_cell * 1000, ESR_source=r_note,
        C_module_F=C_mod, C_reserve=C_mod / sc.C_req - 1 if sc.C_req else np.nan,
        ESR_module_mOhm=R_mod * 1000, I_module_max_A=P * float(cell["i_max_A"]),
        mass_kg=mass, volume_dm3=vol, density_kg_dm3=dens, cost_EUR=cost,
        E_usable_EOL_J=E_use_eol, energy_margin=E_use_eol / sc.W_used_J - 1 if sc.W_used_J else np.nan,
        E_density_Wh_kg=(E_use_eol / req.L_EOL / 3600 / mass) if mass and not _isnan(mass) else np.nan,
        I_module_rms_A=i_rms_out, I_cell_rms_A=i_rms_out / P if i_rms_out == i_rms_out else np.nan,
        P_loss_avg_W=p_loss,
        thermal_status=thermal_status, R_th_KperW=th["rth"] if th else np.nan,
        C_th_JperK=th["cth"] if th else np.nan, k_coupling=k_th, R_th_eff_KperW=rth_eff,
        dT_max_K=dTmax, T_cell_max_C=req.T_amb_C + dTmax if dTmax == dTmax else np.nan,
        I_cell_rms_max_A=i_cell_rms_max,
        block_mm=(f"{lay.width_mm:.0f} x {lay.depth_mm:.0f} x {lay.height_mm:.0f}" if lay else ""),
        pcb_width_mm=lay.width_mm if lay else np.nan, pcb_depth_mm=lay.depth_mm if lay else np.nan,
        c_measured=fit is not None and opt.c_source != "datasheet",
        esr_measured=r_row is not None, thermal_measured=bool(th) and not th.get("proxy_of"),
        pcb_mm=(f"{lay.width_mm:.0f} x {lay.depth_mm:.0f}" if lay else ""),
        layout=(f"{lay.packing}, Gruppe {lay.gx}x{lay.gy}, {lay.bands} Spalten" if lay else ""),
        E_density_Wh_dm3=(E_use_eol / req.L_EOL / 3600 / vol) if vol and not _isnan(vol) else np.nan,
        feasible=feasible, status=status, warnings=" | ".join(det.warnings),
    )
    if lay is not None:
        for n in lay.notes:
            det.warnings.append("Layout: " + n)
        row["warnings"] = " | ".join(det.warnings)
    viol = spec_violations(row, lay, req)
    row["spec_ok"] = not viol
    row["spec_violations"] = "; ".join(viol)
    if viol:
        row["feasible"] = False
        row["status"] = "Vorgabe verletzt: " + "; ".join(viol)
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
