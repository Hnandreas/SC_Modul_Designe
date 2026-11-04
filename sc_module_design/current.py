"""Supercapacitor current from the required terminal power (ESR voltage drop included).

The interface converter transfers the power between the DC bus (e.g. 48 V) and the SC. At the SC terminals
the power P must be delivered at the momentary SC voltage. With the internal (open-circuit) voltage U_C and
the ESR R:

    discharge (P > 0):  P = (U_C - R I) I   ->  I = (U_C - sqrt(U_C^2 - 4 R P)) / (2 R)
    charge    (P < 0):  |P| = (U_C + R |I|) |I|  ->  |I| = (sqrt(U_C^2 + 4 R |P|) - U_C) / (2 R)

The SC can deliver at most P_max = U_C^2 / (4 R) (at I = U_C / (2 R)). Sign: + = discharge.
"""
from __future__ import annotations

import math

import numpy as np


def current_scalar(p: float, U: float, R: float) -> tuple[float, bool]:
    """Returns (current, power_limited)."""
    if R <= 1e-12:
        return p / U, False
    if p >= 0:
        disc = U * U - 4.0 * R * p
        if disc < 0:
            return U / (2.0 * R), True
        return (U - math.sqrt(disc)) / (2.0 * R), False
    return -(math.sqrt(U * U - 4.0 * R * p) - U) / (2.0 * R), False


def current_array(p, U, R: float):
    """Vectorised version. Returns (current, power_limited_mask)."""
    p = np.asarray(p, dtype=float)
    U = np.broadcast_to(np.asarray(U, dtype=float), p.shape)
    if R <= 1e-12:
        return p / U, np.zeros(p.shape, bool)
    disc = U * U - 4.0 * R * p
    lim = (p > 0) & (disc < 0)
    i = np.where(p >= 0, (U - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * R),
                 -(np.sqrt(U * U - 4.0 * R * np.minimum(p, 0.0)) - U) / (2.0 * R))
    i = np.where(lim, U / (2.0 * R), i)
    return i, lim


def terminal_power(i: float, U: float, R: float) -> float:
    """Power at the SC terminals for a given current (inverse of current_scalar)."""
    return (U - R * i) * i
