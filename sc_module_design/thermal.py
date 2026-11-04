"""Thermal verification (Appendix A01): module current trajectory and first-order cell temperature."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from .current import current_array, current_scalar
from .profile import Binned

CURRENT_MODES = {
    "energy_balance": "P_SC = (U_C - R I) I mit U_C(t) aus Energiebilanz ab U_C0inM (realistisch)",
    "conservative_umin": "P_SC = (U_C0min - R I) I (Worst Case, U_C = U_C0min)",
}
PROFILE_MODES = {
    "periodic": "Profil endlos, periodischer Beharrungszustand (Dauerbetrieb)",
    "single_pass": "Ein Durchlauf ab thermischem Gleichgewicht (Anhang A01)",
}


@dataclass
class CurrentTrajectory:
    t: np.ndarray
    U: np.ndarray            # module voltage per bin [V]
    I_mean: np.ndarray       # signed mean module current per bin [A]
    I_sq: np.ndarray         # mean square module current per bin [A^2]
    mode: str
    notes: list = field(default_factory=list)

    @property
    def I_rms(self) -> float:
        return float(math.sqrt(np.mean(self.I_sq)))


def module_current(psc: Binned, mode: str, U_min: float, U_inM: float, U_max: float,
                   C_module: float = float("nan"), R_module: float = 0.0) -> CurrentTrajectory:
    """Module current from the SC terminal power: P = (U_C - R I) I (ESR drop, power balance at the converter)."""
    if mode == "conservative_umin":
        U = np.full(psc.n, U_min)
        i_m, _ = current_array(psc.mean, U, R_module)
        prms = np.sqrt(psc.mean_sq) * np.where(psc.mean >= 0, 1.0, -1.0)
        i_r, _ = current_array(prms, U, R_module)
        return CurrentTrajectory(psc.t, U, i_m, np.maximum(i_r ** 2, i_m ** 2), mode)
    if mode != "energy_balance":
        raise ValueError(f"unknown current mode {mode}")
    dt = psc.dt
    C = C_module
    R = R_module
    E = 0.5 * C * U_inM ** 2
    E_lo, E_hi = 0.5 * C * U_min ** 2, 0.5 * C * U_max ** 2
    n = psc.n
    U = np.empty(n)
    I_m = np.empty(n)
    I_sq = np.empty(n)
    pm, pq = psc.mean.tolist(), psc.mean_sq.tolist()
    n_lo = n_hi = n_pl = 0
    sqrt = math.sqrt
    for k in range(n):
        u = sqrt(2.0 * E / C)
        U[k] = u
        p = pm[k]
        i, lim = current_scalar(p, u, R)
        prms = sqrt(pq[k]) if pq[k] > 0 else 0.0
        ir, lim2 = current_scalar(prms if p >= 0 else -prms, u, R)
        if lim or lim2:
            n_pl += 1
        i2 = max(ir * ir, i * i)
        I_m[k], I_sq[k] = i, i2
        E -= (p + R * i2) * dt
        if E < E_lo:
            E, n_lo = E_lo, n_lo + 1
        elif E > E_hi:
            E, n_hi = E_hi, n_hi + 1
    notes = []
    if n_lo:
        notes.append(f"U_C0min erreicht für {n_lo * dt:.2f} s im Profil (ohne Nachladung).")
    if n_hi:
        notes.append(f"U_C0max erreicht für {n_hi * dt:.2f} s im Profil.")
    if n_pl:
        notes.append(f"Leistungsgrenze U_C^2/(4R) erreicht für {n_pl * dt:.2f} s.")
    return CurrentTrajectory(psc.t, U, I_m, I_sq, mode, notes)


def first_order_response(P: np.ndarray, dt: float, R_th: float, C_th: float, dT0: float = 0.0) -> np.ndarray:
    """Exact ZOH discretisation of C_th dT/dt = P - dT/R_th  [app_transient_thermal_model]."""
    a = math.exp(-dt / (R_th * C_th))
    return lfilter([(1 - a) * R_th], [1.0, -a], P, zi=[a * dT0])[0]


def temperature_rise(P_loss_cell: np.ndarray, dt: float, R_th: float, C_th: float,
                     mode: str = "periodic") -> tuple[np.ndarray, float]:
    """Temperature rise of one cell over the profile. Returns (dT(t), dT at profile start)."""
    if mode == "single_pass":
        return first_order_response(P_loss_cell, dt, R_th, C_th, 0.0), 0.0
    F = first_order_response(P_loss_cell, dt, R_th, C_th, 0.0)[-1]
    aN = math.exp(-len(P_loss_cell) * dt / (R_th * C_th))
    dT0 = F / (1 - aN)
    return first_order_response(P_loss_cell, dt, R_th, C_th, dT0), dT0


# ---------------------------------------------------------------------- thermal model extensions
COUPLING_MODES = {
    "single_cell": "Einzelzelle in ruhender Luft (Messbedingung von R_th, günstigster Fall)",
    "packed_block": "Zellen als Block auf der Leiterplatte, Wärme nur über die Blockoberfläche (ohne Lüfter)",
    "factor": "Eigener Faktor auf R_th (z. B. aus Modulmessung oder FEM)",
}


def cell_surface_mm2(d_mm: float, l_mm: float) -> float:
    """Full surface of a cylindrical cell (mantle + both ends)."""
    return math.pi * d_mm * l_mm + 2 * math.pi * (d_mm / 2) ** 2


def block_layout(n: int, d_mm: float, l_mm: float, gap_mm: float = 2.0) -> dict:
    """Near-square grid of n standing cells and its outer box."""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pitch = d_mm + gap_mm
    w, dp, h = cols * pitch, rows * pitch, l_mm
    a_box = 2 * (w * dp + w * h + dp * h)
    return dict(cols=cols, rows=rows, width_mm=w, depth_mm=dp, height_mm=h, area_mm2=a_box)


def block_coupling_factor(n: int, d_mm: float, l_mm: float, gap_mm: float = 2.0) -> float:
    """k = n A_cell / A_block (>= 1).

    Same heat transfer coefficient as the single cell (from its measured R_th), but the heat of all
    n cells has to leave through the outer surface of the block. Per cell: R_th,eff = k R_th.
    """
    lay = block_layout(n, d_mm, l_mm, gap_mm)
    return max(1.0, n * cell_surface_mm2(d_mm, l_mm) / lay["area_mm2"])


def scale_thermal(rth_ref: float, cth_ref: float, ref_cell: dict, cell: dict) -> tuple[float, float]:
    """Transfer R_th/C_th of a characterized cell to another size: R_th ~ 1/surface, C_th ~ mass."""
    a_ref = cell_surface_mm2(ref_cell["diameter_mm"], ref_cell["length_mm"])
    a = cell_surface_mm2(cell["diameter_mm"], cell["length_mm"])
    rth = rth_ref * a_ref / a
    cth = cth_ref * cell["mass_g"] / ref_cell["mass_g"]
    return rth, cth


# ---------------------------------------------------------------------- current -> I^2 R -> low-pass
def cell_loss_lowpass(i_cell: np.ndarray, dt: float, R_esr: float, R_th: float, C_th: float,
                      periodic: bool = True) -> dict:
    """Cell temperature from the cell current, sample by sample.

    p_v(t) = i_cell(t)^2 ESR      (Joule loss in the ESR of one cell)
    C_th dT/dt = p_v - dT / R_th  (first-order thermal low-pass, exact ZOH discretisation)
    periodic: initial value of the periodic steady state (profile repeated endlessly).
    """
    p_v = i_cell * i_cell * R_esr
    dT, dT0 = temperature_rise(p_v, dt, R_th, C_th, "periodic" if periodic else "single_pass")
    return dict(p_v=p_v, dT=dT, dT0=dT0, P_v_mean=float(np.mean(p_v)), P_v_max=float(np.max(p_v)),
                I_rms=float(np.sqrt(np.mean(i_cell ** 2))), dT_max=float(np.max(dT)),
                dT_ss=float(np.mean(p_v)) * R_th, tau_s=R_th * C_th)
