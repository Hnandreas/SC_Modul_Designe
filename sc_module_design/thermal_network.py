"""Thermal network model of a supercapacitor module (precheck).

Implements Andy's specification (project doc `thermal_network_model_spec.md`) without changing its
assumptions:

* every cell is a heat source with the same loss P_cell = I_rms^2 ESR,
* every cell (also inside) releases heat to its local air through its own R_air (measured single-cell R_th),
* local air temperature by the algebraic averaging rule
      T_air,i = (sum_{j in N_i} T_j + (z - m_i + 1) T_inf) / (z + 1),
  i.e. present neighbours contribute their temperature, missing neighbours and the direction upwards the
  outer ambient (equal weighting, a chosen simplification of the demo, not measured),
* conduction through the air gap between direct neighbours
      R_gap = (s + r (2 - pi/2)) / (2 r L lambda)     (MathWorks, inter-cell thermal path),
* optional radiation of the top circular face only: eps sigma A_top (T^4 - T_rad^4),
* cell balance  C_th dT_i/dt = P - (T_i - T_air,i)/R_air - sum_j (T_i - T_j)/R_gap - Q_rad,i.

Arrangements: ideal square (z = 4) or hexagonal (z = 6) lattice grown in layers around a centre cell, or the
real neighbourhood of a PCB layout (cells closer than 1.02 pitch are direct neighbours).

Results are a precheck, neither a validated prediction nor a proven conservative limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

SIGMA = 5.670374419e-8
K0 = 273.15
SOURCES = {
    "spec": "Spezifikation Andy (claude/thermal_network_model_spec.md), Precheck, nicht validiert",
    "gap": "MathWorks, Simscape Battery: Model Heat Exchange Between Cells (inter-cell thermal path)",
    "rad": "COMSOL Heat Transfer: Surface-to-Ambient Radiation; IFRF: emissivity of metals (eps = 0.05)",
    "network_lit": "A. Hijazi et al., Thermal Network Model of Supercapacitors Stack, IEEE TIE 59(2), 2012",
}


@dataclass
class NetworkParams:
    d_mm: float = 18.0
    L_mm: float = 35.0
    R_air: float = 22.1            # K/W, cell to local air (measured single cell)
    C_th: float = 19.7             # J/K
    T_inf_C: float = 25.0          # outer ambient (also radiation ambient)
    lam: float = 0.026             # W/(m K), air
    s_mm: float = 0.0              # smallest surface-to-surface distance
    radiation: bool = True
    eps: float = 0.05              # bare aluminium (working assumption)

    @property
    def r_m(self) -> float:
        return self.d_mm / 2000

    @property
    def A_top(self) -> float:
        return math.pi * self.r_m ** 2

    @property
    def R_gap(self) -> float:
        return r_gap(self.d_mm, self.L_mm, self.s_mm, self.lam)


def r_gap(d_mm: float, L_mm: float, s_mm: float, lam: float = 0.026) -> float:
    """Air-gap resistance between two equal parallel cylinders [K/W] (lengths in m)."""
    r, L, s = d_mm / 2000, L_mm / 1000, s_mm / 1000
    return (s + r * (2 - math.pi / 2)) / (2 * r * L * lam)


@dataclass
class Grid:
    xy: np.ndarray                 # (n, 2) cell centres [mm]
    nbr: list                      # list of neighbour index arrays
    z: int                         # neighbour directions (4 square, 6 hex)
    layer: np.ndarray              # layer index (ring grids) or distance to the edge (layouts)
    centre: int                    # index of the centre cell
    outer: np.ndarray              # indices of the outermost layer / edge cells
    kind: str = ""

    @property
    def n(self) -> int:
        return len(self.xy)

    @property
    def m(self) -> np.ndarray:
        return np.array([len(a) for a in self.nbr])


def ring_grid(arrangement: str, n_layers: int, pitch_mm: float = 18.0) -> Grid:
    """Centre cell plus n complete neighbour layers. square: N = 1 + 2n(n+1), hex: N = 1 + 3n(n+1)."""
    pts, lay = [], []
    if arrangement == "square":
        z = 4
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i in range(-n_layers, n_layers + 1):
            for j in range(-n_layers, n_layers + 1):
                if abs(i) + abs(j) <= n_layers:
                    pts.append((i, j))
                    lay.append(abs(i) + abs(j))
        xy = np.array([(i * pitch_mm, j * pitch_mm) for i, j in pts], float)
    else:
        z = 6
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        for q in range(-n_layers, n_layers + 1):
            for r in range(-n_layers, n_layers + 1):
                dist = max(abs(q), abs(r), abs(q + r))
                if dist <= n_layers:
                    pts.append((q, r))
                    lay.append(dist)
        xy = np.array([(pitch_mm * (q + r / 2), pitch_mm * r * math.sqrt(3) / 2) for q, r in pts], float)
    idx = {p: k for k, p in enumerate(pts)}
    nbr = [np.array([idx[(p[0] + a, p[1] + b)] for a, b in dirs if (p[0] + a, p[1] + b) in idx], int)
           for p in pts]
    lay = np.array(lay)
    return Grid(xy, nbr, z, lay, int(idx[(0, 0)]), np.where(lay == n_layers)[0], arrangement)


def layout_grid(lay) -> Grid:
    """Real neighbourhood of a PCB layout (layout.PCBLayout): neighbours closer than 1.02 pitch."""
    xy = np.array([(c[0], c[1]) for c in lay.cells], float)
    pitch = lay.d_mm + lay.gap_mm
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nbr = [np.where(row <= 1.02 * pitch)[0] for row in d]
    z = 6 if lay.packing == "hex" else 4
    m = np.array([len(a) for a in nbr])
    edge = np.where(m < z)[0]
    # graph distance to the edge (layer counted from outside)
    dist = np.full(len(xy), -1)
    dist[edge] = 0
    front = list(edge)
    while front:
        nxt = []
        for i in front:
            for j in nbr[i]:
                if dist[j] < 0:
                    dist[j] = dist[i] + 1
                    nxt.append(j)
        front = nxt
    dist[dist < 0] = 0
    ctr = xy.mean(0)
    deepest = np.where(dist == dist.max())[0]
    centre = int(deepest[np.argmin(((xy[deepest] - ctr) ** 2).sum(1))])
    return Grid(xy, nbr, z, dist, centre, edge, "layout")


def _matrices(g: Grid, p: NetworkParams):
    Ga, Gg, w = 1.0 / p.R_air, 1.0 / p.R_gap, 1.0 / (g.z + 1)
    m = g.m
    rows, cols, vals = [], [], []
    for i, nb in enumerate(g.nbr):
        rows += [i] * len(nb)
        cols += list(nb)
        vals += [-(Ga * w + Gg)] * len(nb)
    off = csr_matrix((vals, (rows, cols)), shape=(g.n, g.n))
    M = (off + diags(Ga + Gg * m)).tocsr()
    b_amb = Ga * w * (g.z - m + 1) * p.T_inf_C
    return M, b_amb


def _rad(T: np.ndarray, p: NetworkParams):
    if not p.radiation:
        return np.zeros_like(T), np.zeros_like(T)
    TK = T + K0
    k = p.eps * SIGMA * p.A_top
    return k * (TK ** 4 - (p.T_inf_C + K0) ** 4), 4 * k * TK ** 3


def local_air(g: Grid, T: np.ndarray, p: NetworkParams) -> np.ndarray:
    s = np.array([T[nb].sum() for nb in g.nbr])
    return (s + (g.z - g.m + 1) * p.T_inf_C) / (g.z + 1)


@dataclass
class NetworkResult:
    grid: Grid
    params: NetworkParams
    P_cell_W: float
    T: np.ndarray                  # steady cell temperatures [degC]
    T_air: np.ndarray
    Q_rad: np.ndarray
    notes: list = field(default_factory=list)

    @property
    def dT(self) -> np.ndarray:
        return self.T - self.params.T_inf_C

    def summary(self) -> dict:
        g, T, Ta = self.grid, self.T, self.T_air
        o = g.outer
        return dict(N=g.n, P_cell_W=self.P_cell_W, P_total_W=self.P_cell_W * g.n, R_gap_KperW=self.params.R_gap,
                    T_centre_C=float(T[g.centre]), T_outer_mean_C=float(T[o].mean()),
                    dT_centre_outer_K=float(T[g.centre] - T[o].mean()), T_max_C=float(T.max()),
                    dT_max_K=float(T.max() - self.params.T_inf_C), dT_centre_K=float(T[g.centre] - self.params.T_inf_C),
                    dT_outer_mean_K=float(T[o].mean() - self.params.T_inf_C),
                    T_air_centre_C=float(Ta[g.centre]), T_air_outer_mean_C=float(Ta[o].mean()),
                    Q_rad_total_W=float(self.Q_rad.sum()), Q_rad_cell_max_W=float(self.Q_rad.max()))


def steady_state(g: Grid, p: NetworkParams, P_cell: float, tol: float = 1e-9, it_max: int = 50) -> NetworkResult:
    """Stationary solution (dT/dt = 0) with Newton iteration for the radiation term."""
    M, b_amb = _matrices(g, p)
    b = P_cell + b_amb
    T = spsolve(M.tocsc(), b)
    for _ in range(it_max):
        q, dq = _rad(T, p)
        F = M @ T + q - b
        if np.max(np.abs(F)) < tol:
            break
        T = T - spsolve((M + diags(dq)).tocsc(), F)
    q, _ = _rad(T, p)
    return NetworkResult(g, p, P_cell, T, local_air(g, T, p), q)


def transient(g: Grid, p: NetworkParams, P_cell, t_end_s: float = 7200.0, n_points: int = 300,
              T0_C: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Heating from T0 (default ambient). P_cell: constant [W] or callable P(t) [W]. Returns (t, T[n_t, n])."""
    M, b_amb = _matrices(g, p)
    Pf: Callable = P_cell if callable(P_cell) else (lambda t: P_cell)
    C = p.C_th

    def f(t, T):
        q, _ = _rad(T, p)
        return (Pf(t) + b_amb - M @ T - q) / C

    def jac(t, T):
        _, dq = _rad(T, p)
        return (-(M + diags(dq)) / C).toarray() if g.n <= 400 else -(M + diags(dq)) / C

    T0 = np.full(g.n, p.T_inf_C if T0_C is None else T0_C, float)
    te = np.linspace(0, t_end_s, n_points)
    sol = solve_ivp(f, (0, t_end_s), T0, method="BDF", t_eval=te, jac=jac, rtol=1e-6, atol=1e-6)
    return sol.t, sol.y.T


def layer_study(arrangement: str, p: NetworkParams, P_cell: float, n_max: int = 12, tol_K: float = 0.05
                ) -> tuple[list, bool, int]:
    """Steady results over the number of layers. Converged when |T_c(n) - T_c(n-1)| < tol_K for two
    consecutive expansions. Returns (rows, converged, n_converged)."""
    rows, prev, hits, conv_n = [], None, 0, None
    pitch = p.d_mm + p.s_mm
    for n in range(0, n_max + 1):
        res = steady_state(ring_grid(arrangement, n, pitch), p, P_cell)
        s = res.summary()
        s["layers"] = n
        d = None if prev is None else abs(s["T_centre_C"] - prev)
        s["delta_centre_K"] = d
        rows.append(s)
        if d is not None and d < tol_K:
            hits += 1
            if hits >= 2 and conv_n is None:
                conv_n = n
                break
        else:
            hits = 0
        prev = s["T_centre_C"]
    return rows, conv_n is not None, conv_n if conv_n is not None else n_max


def n_cells(arrangement: str, n: int) -> int:
    return 1 + (2 if arrangement == "square" else 3) * n * (n + 1)
