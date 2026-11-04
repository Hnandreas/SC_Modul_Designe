"""Interface converter between DC bus and supercapacitor (efficiency per time step).

Sign convention: positive power = SC discharges (delivers power to the bus).
Discharging: the SC has to deliver P_bus / eta_dch. Charging: the SC receives P_bus eta_ch.
A power-dependent efficiency can be added by replacing these functions (same signature).
"""
from __future__ import annotations

import numpy as np


def bus_to_cell(p_bus, eta_dch: float = 1.0, eta_ch: float = 1.0):
    p_bus = np.asarray(p_bus, dtype=float)
    return np.where(p_bus > 0, p_bus / eta_dch, p_bus * eta_ch)


def bus_to_cell_scalar(p_bus: float, eta_dch: float = 1.0, eta_ch: float = 1.0) -> float:
    return p_bus / eta_dch if p_bus > 0 else p_bus * eta_ch
