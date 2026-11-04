"""Charge/discharge energy excursions of the SC power (Eqs. DChmax / Chmax)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHODS = {
    "zero_crossing_legacy": "Zero crossings, IECON 2023 MATLAB algorithm (reproduces published values)",
    "zero_crossing_exact": "Zero crossings, exact sign segments (integration between interpolated crossings)",
    "cumulative_swing": "Range of the cumulative SC energy max(E)-min(E) (multi-peak sequences, extension)",
}


@dataclass
class Excursions:
    E_dch_J: float                 # largest discharge excursion (positive)
    E_ch_J: float                  # largest charge excursion (as positive number)
    method: str
    dch_segment: tuple = (0, 0)    # sample indices [a, b) of the max discharge excursion
    ch_segment: tuple = (0, 0)
    segment_energies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    notes: list = field(default_factory=list)


def _legacy(p_sc: np.ndarray, dt: float) -> Excursions:
    """1:1 port of the integration block in the IECON 2023 live script.

    fAbove = f.*(f>=0); positions = find(diff(fAbove>0));
    a = 1 (first) or positions(i)+1; b = positions(i+1); integrated(i) = trapz(f(a:b))*T
    """
    f_above = p_sc * (p_sc >= 0)
    pos = np.flatnonzero(np.diff((f_above > 0).astype(np.int8))) + 1   # MATLAB 1-based index of crossing
    if len(pos) < 2:
        return Excursions(0.0, 0.0, "zero_crossing_legacy", notes=["fewer than two zero crossings"])
    starts = np.concatenate([[0], pos[1:-1]])          # 0-based a
    ends = pos[1:]                                     # exclusive end (= MATLAB b inclusive)
    c = np.concatenate([[0.0], np.cumsum(p_sc)])
    sums = c[ends] - c[starts]
    # trapz over samples a..b = sum - (first+last)/2
    integ = (sums - 0.5 * (p_sc[starts] + p_sc[ends - 1])) * dt
    i_max, i_min = int(np.argmax(integ)), int(np.argmin(integ))
    return Excursions(float(max(integ[i_max], 0.0)), float(max(-integ[i_min], 0.0)), "zero_crossing_legacy",
                      (int(starts[i_max]), int(ends[i_max])), (int(starts[i_min]), int(ends[i_min])), integ)


def _exact(p_sc: np.ndarray, dt: float) -> Excursions:
    s = np.sign(p_sc)
    idx = np.flatnonzero(s[1:] * s[:-1] < 0) + 1        # first sample after a sign change
    bounds = np.concatenate([[0], idx, [len(p_sc)]])
    c = np.concatenate([[0.0], np.cumsum(p_sc)])
    integ = (c[bounds[1:]] - c[bounds[:-1]]) * dt
    i_max, i_min = int(np.argmax(integ)), int(np.argmin(integ))
    return Excursions(float(max(integ[i_max], 0.0)), float(max(-integ[i_min], 0.0)), "zero_crossing_exact",
                      (int(bounds[i_max]), int(bounds[i_max + 1])), (int(bounds[i_min]), int(bounds[i_min + 1])),
                      integ)


def _cumulative(p_sc: np.ndarray, dt: float) -> Excursions:
    e = -np.cumsum(p_sc) * dt            # stored energy change (discharge lowers it)
    i_hi, i_lo = int(np.argmax(e)), int(np.argmin(e))
    swing = float(e[i_hi] - e[i_lo])
    notes = []
    drift = float(-np.sum(p_sc) * dt)
    if abs(drift) > 0.05 * max(swing, 1e-9):
        notes.append(f"SC energy is not balanced over the profile (drift {drift:.1f} J). The swing then "
                     f"depends on the recharge strategy, check with the emulation.")
    # split swing into a discharge part (from start level down) and a charge part
    e0 = 0.0
    E_dch = float(max(e0 - e[i_lo], 0.0))
    E_ch = float(max(e[i_hi] - e0, 0.0))
    if E_dch + E_ch < swing:          # extremes ordered such that the start is not between them
        E_dch = swing - E_ch
    a, b = sorted((i_hi, i_lo))
    return Excursions(E_dch, E_ch, "cumulative_swing", (a, b), (a, b), np.array([swing]), notes)


def energy_excursions(p_sc: np.ndarray, dt: float, method: str = "zero_crossing_legacy") -> Excursions:
    if method == "zero_crossing_legacy":
        return _legacy(p_sc, dt)
    if method == "zero_crossing_exact":
        return _exact(p_sc, dt)
    if method == "cumulative_swing":
        return _cumulative(p_sc, dt)
    raise ValueError(f"unknown method {method}, use one of {list(METHODS)}")


def braking_energy(mass_kg: float, v_mps: float) -> float:
    """E_braking = 1/2 m v^2."""
    return 0.5 * mass_kg * v_mps ** 2


def braking_power(E_J: float, t_brake_s: float) -> float:
    """Peak power of a linear (triangular) braking ramp: P = 2 E / t."""
    return 2.0 * E_J / t_brake_s
