"""Thermal verification (Appendix A01): module current trajectory and first-order cell temperature."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from .profile import Binned

CURRENT_MODES = {
    "energy_balance": "P_SC(t)/U(t), U(t) from the energy balance starting at U_C0inM (realistic)",
    "conservative_umin": "P_SC(t)/U_C0min (worst case, independent of P)",
}
PROFILE_MODES = {
    "periodic": "Profile repeated endlessly, periodic thermal steady state (continuous operation)",
    "single_pass": "One pass of the profile starting from thermal equilibrium (Appendix A01)",
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
    if mode == "conservative_umin":
        U = np.full(psc.n, U_min)
        return CurrentTrajectory(psc.t, U, psc.mean / U_min, psc.mean_sq / U_min ** 2, mode)
    if mode != "energy_balance":
        raise ValueError(f"unknown current mode {mode}")
    dt = psc.dt
    C = C_module
    E = 0.5 * C * U_inM ** 2
    E_lo, E_hi = 0.5 * C * U_min ** 2, 0.5 * C * U_max ** 2
    n = psc.n
    U = np.empty(n)
    pm, pq = psc.mean.tolist(), psc.mean_sq.tolist()
    n_lo = n_hi = 0
    for k in range(n):
        u = math.sqrt(2.0 * E / C)
        U[k] = u
        loss = R_module * pq[k] / (u * u)
        E -= (pm[k] + loss) * dt
        if E < E_lo:
            E, n_lo = E_lo, n_lo + 1
        elif E > E_hi:
            E, n_hi = E_hi, n_hi + 1
    notes = []
    if n_lo:
        notes.append(f"U_C0min reached in {n_lo * dt:.2f} s of the profile (current evaluated at U_C0min).")
    if n_hi:
        notes.append(f"U_C0max reached in {n_hi * dt:.2f} s of the profile.")
    return CurrentTrajectory(psc.t, U, psc.mean / U, psc.mean_sq / U ** 2, mode, notes)


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
