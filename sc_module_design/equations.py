"""Closed-form design equations of Sec. 'Supercapacitor Module Design' and Appendix A01/A02.

The equation labels of the thesis are given in brackets.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


def ceil_safe(x: float) -> int:
    """ceil() that ignores floating point noise (7.0000000001 -> 7)."""
    return int(math.ceil(x - EPS))


# ---------------------------------------------------------------------- energy / voltages
def stored_energy(C: float, U: float) -> float:
    """W = 1/2 C U^2  [cap22]."""
    return 0.5 * C * U ** 2


def usable_energy(C: float, U_max: float, U_min: float, L_EOL: float = 1.0) -> float:
    """W_used = L_EOL/2 C (Umax^2 - Umin^2)  [cap2, cap5]."""
    return 0.5 * L_EOL * C * (U_max ** 2 - U_min ** 2)


def discharge_ratio(U_min: float, U_max: float) -> float:
    """r_Q = U_C0min / U_C0max  [cap3]."""
    return U_min / U_max


def energy_ratio(r_q):
    """W_used / W_total = 1 - r_Q^2  [cap4]."""
    return 1.0 - np.asarray(r_q) ** 2


def u_c0min_from_current(P_max: float, I_C0max: float) -> float:
    """U_C0min = P_max / I_C0max  [currentlimit]."""
    return P_max / I_C0max


def u_c0inm(E_dch: float, E_ch: float, U_max: float, U_min: float) -> float:
    """Energy-weighted intermediate voltage  [Intermediate]."""
    E_dch, E_ch = abs(E_dch), abs(E_ch)
    if E_dch + E_ch <= 0:
        return U_max
    return math.sqrt((E_dch * U_max ** 2 + E_ch * U_min ** 2) / (E_dch + E_ch))


# ---------------------------------------------------------------------- lifetime
def u_cell_max_lifetime(t_life_h: float, T_C: float, u0_V: float, T0_C: float, t0_h: float,
                        dU_V: float = 0.2, dT_K: float = 10.0) -> float:
    """Permissible cell voltage for a required lifetime  [voltagelife].

    U_C01max = U_C0N1 - (log2(t/t0) - (T0 - T)/dT) dU
    """
    return u0_V - (math.log2(t_life_h / t0_h) - (T0_C - T_C) / dT_K) * dU_V


def lifetime_h(U_V: float, T_C: float, u0_V: float, T0_C: float, t0_h: float, dU_V: float = 0.2,
               dT_K: float = 10.0) -> float:
    """Arrhenius lifetime estimate  [arrhenius]: t = t0 2^((T0-T)/dT) 2^((U0-U)/dU)."""
    return t0_h * 2 ** ((T0_C - T_C) / dT_K) * 2 ** ((u0_V - U_V) / dU_V)


def n_series(U_C0max: float, U_C01max: float) -> int:
    """S = ceil(U_C0max / U_C01max)  [seriescells]."""
    return ceil_safe(U_C0max / U_C01max)


# ---------------------------------------------------------------------- capacitance
def c_required(W_used: float, U_max: float, U_min: float, L_EOL: float) -> float:
    """C_req = 2 W_used / (L_EOL (Umax^2 - Umin^2))  [cap6]."""
    return 2.0 * W_used / (L_EOL * (U_max ** 2 - U_min ** 2))


def c_eq_energy(C0: float, k: float, u_min: float, u_max: float) -> float:
    """Energy-equivalent capacitance of C(U) = C0 + kU on [u_min, u_max]  [energy_equivalent_capacitance2].

    Derivation (Appendix A02): W = int C(u) u du = C0 (u2^2-u1^2)/2 + k (u2^3-u1^3)/3 = C_eq (u2^2-u1^2)/2.
    """
    if abs(u_max - u_min) < 1e-12:
        return C0 + k * u_max
    return C0 + (2.0 * k / 3.0) * (u_max ** 3 - u_min ** 3) / (u_max ** 2 - u_min ** 2)


def cell_energy_cu(C0: float, k: float, u):
    """Energy of a cell with C(U) = C0 + kU charged from 0 to u: C0 u^2/2 + k u^3/3."""
    u = np.asarray(u, dtype=float)
    return C0 * u ** 2 / 2 + k * u ** 3 / 3


# ---------------------------------------------------------------------- module
def p_capacitance(S: int, C_req: float, C_cell: float) -> int:
    """P_C = ceil(S C_req / C_cell)  [parallel_capacitance]."""
    return max(1, ceil_safe(S * C_req / C_cell))


def p_current(I_C0max: float, I_cell_max: float) -> int:
    """P_I = ceil(I_C0max / I_cell,max)  [parallel_current]."""
    return max(1, ceil_safe(I_C0max / I_cell_max))


def module_capacitance(P: int, S: int, C_cell: float) -> float:
    return P / S * C_cell


def module_esr(P: int, S: int, R_cell: float) -> float:
    return S / P * R_cell


# ---------------------------------------------------------------------- thermal (Appendix A01)
def i_cell_rms_max(dT_max: float, R_esr_cell: float, R_th: float) -> float:
    """Thermally permissible continuous RMS cell current  [app_thermal_rms_limit]."""
    return math.sqrt(dT_max / (R_esr_cell * R_th))


def p_thermal(I_module_rms: float, I_cell_rms_max: float) -> int:
    """P_T = ceil(I_module,RMS / I_cell,RMS,max)  [app_parallel_thermal]."""
    return max(1, ceil_safe(I_module_rms / I_cell_rms_max))


def t_overload_max(I_cell_rms: float, R_esr_cell: float, R_th: float, C_th: float, dT_max: float) -> float:
    """Max. duration of an overload before dT_max is reached  [app_maximum_overload_duration]."""
    dT_ss = I_cell_rms ** 2 * R_esr_cell * R_th
    if dT_ss <= dT_max:
        return math.inf
    return -R_th * C_th * math.log(1 - dT_max / dT_ss)
