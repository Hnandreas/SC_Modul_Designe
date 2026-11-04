"""HESS emulation with the load profile repeated endlessly (thermal simulation / cooling concept, step 14).

Power balance at the interface converter in every time step:

    P_bus = U_bus I_bus                         (SC share on the DC bus, e.g. 48 V)
    P_SC  = P_bus / eta_dch  (discharge)        P_SC = P_bus eta_ch (charge)
    P_SC  = U_SC I_SC  with  U_SC = U_C - R I_SC (terminal voltage = internal voltage minus ESR drop)
    ->  I_SC = (U_C - sqrt(U_C^2 - 4 R P_SC)) / (2 R)        (see current.py)

Per bin (default 10 ms, mean and mean square of the full-resolution power are kept):
  battery    P_bat = p_bat_split + P_rech + P_overflow
  SC         P_bus,SC = p_sc_split - P_rech, limited by the converter current limit I_C0max (hard) and the power
             limit U_C^2/(4R), the rest is delivered by the battery (overflow). At U_C0min / U_C0max the voltage is
             held and the battery delivers / takes the energy, the SC current keeps flowing (the limit is the
             current of the converter, not the voltage).
  recharge   P_rech = K (E_target - E), limited to +-P_rech_max (optional, off by default: the split, e.g. the
             moving average, recharges the SC)
  SC energy  dE/dt = -P_SC                     (simplified: the ESR loss R i^2 heats the cells but is not taken
                                               from the stored energy; optional esr_loss_in_energy)
  cell temperature  C_th dT/dt = R_cell i_cell^2 - dT / R_th

``run_cycle()`` advances one profile period, ``run_until_steady()`` repeats until U and dT are
periodic, ``stream()`` / ``_advance()`` continue forever (live view in the GUI).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from . import equations as eq
from .current import current_array
from .profile import bin_signal


@dataclass
class ModuleModel:
    S: int
    P: int
    C_cell_F: float                 # constant capacitance per cell (datasheet or C_eq,E)
    R_cell_Ohm: float
    U_min: float
    U_max: float
    U_init: float
    I_max_A: float
    C0_cell_F: Optional[float] = None   # if given (with k), C(U) = C0 + kU is used
    k_cell_FperV: Optional[float] = None
    R_th: Optional[float] = None
    C_th: Optional[float] = None
    label: str = ""

    @property
    def C_module(self) -> float:
        return self.P / self.S * self.C_cell_F

    @property
    def R_module(self) -> float:
        return self.S / self.P * self.R_cell_Ohm

    @property
    def n_cells(self) -> int:
        return self.S * self.P


@dataclass
class EmulationConfig:
    dt_bin_s: float = 0.01
    soc_control: bool = False           # extra recharge controller (off: the split, e.g. moving average, recharges)
    soc_target: str = "U_C0inM"         # U_C0inM | U_C0max
    K_soc_per_s: float = 0.05           # P_rech = K (E_target - E)
    P_rech_max_W: float = 300.0
    eol: bool = False                   # end of life: C x L_EOL, R x esr_eol_factor
    L_EOL: float = 0.7
    esr_eol_factor: float = 2.0
    T_amb_C: float = 25.0
    eta_dch: float = 1.0                # converter efficiency SC -> bus (applied in every time step)
    eta_ch: float = 1.0                 # converter efficiency bus -> SC
    esr_loss_in_energy: bool = False    # simplified (default): ESR losses heat the cells but are not taken from E_SC
    U_bus_V: float = 48.0               # DC bus voltage (for the bus current I_bus = P_bus / U_bus)
    U_conv_min_V: Optional[float] = None  # minimum input voltage of the converter (terminal voltage U_C - R I)
    max_cycles: int = 200
    tol_U_V: float = 1e-3
    tol_T_K: float = 0.02


@dataclass
class CycleTrace:
    t: np.ndarray
    P_load: np.ndarray
    P_bat: np.ndarray
    P_sc: np.ndarray                # SC share on the bus side
    P_rech: np.ndarray
    U_sc: np.ndarray                # internal SC voltage U_C (open circuit)
    I_sc_rms: np.ndarray            # RMS SC current per step
    dT: np.ndarray
    overflow: np.ndarray
    P_sc_cell: np.ndarray           # power at the SC terminals
    I_sc: np.ndarray                # SC current (mean per step), + = discharge
    E_sc: np.ndarray                # stored energy at the end of each step
    I_sc_peak: np.ndarray           # largest |I_SC| inside each step (full resolution)
    U_term: np.ndarray              # terminal voltage U_C - R I
    I_bus: np.ndarray               # bus-side current of the SC path, P_bus / U_bus


@dataclass
class CycleSummary:
    cycle: int
    U_start: float
    U_end: float
    U_min: float
    U_max: float
    dT_start: float
    dT_max: float
    P_bat_max_W: float
    P_bat_min_W: float
    P_bat_rms_W: float
    I_sc_rms_A: float
    I_sc_peak_A: float
    E_loss_J: float
    E_overflow_J: float
    t_at_Umin_s: float
    t_at_Umax_s: float
    t_current_limit_s: float
    E_conv_loss_J: float = 0.0
    E_swing_J: float = 0.0
    I_sc_rms_cell_A: float = 0.0
    U_term_min_V: float = 0.0
    t_power_limit_s: float = 0.0
    t_Uconv_min_s: float = 0.0


class HESSEmulator:
    def __init__(self, t: np.ndarray, p_load: np.ndarray, p_bat: np.ndarray, module: ModuleModel,
                 cfg: Optional[EmulationConfig] = None):
        self.cfg = cfg or EmulationConfig()
        self.m = module
        dt = float(t[1] - t[0])
        self.load = bin_signal(p_load, dt, self.cfg.dt_bin_s)
        self.bat = bin_signal(p_bat, dt, self.cfg.dt_bin_s)
        self.sc = bin_signal(p_load - p_bat, dt, self.cfg.dt_bin_s)
        self.dt = self.load.dt
        self.n = self.load.n
        self.period = self.n * self.dt
        c_fac = self.cfg.L_EOL if self.cfg.eol else 1.0
        r_fac = self.cfg.esr_eol_factor if self.cfg.eol else 1.0
        self.R_mod = module.R_module * r_fac
        self.R_cell_eff = module.R_cell_Ohm * r_fac
        self._setup_energy(c_fac)
        self.E = self.energy(module.U_init)
        target = module.U_init if self.cfg.soc_target == "U_C0inM" else module.U_max
        self.E_target = self.energy(target)
        self.E_lo, self.E_hi = self.energy(module.U_min), self.energy(module.U_max)
        self.dT = 0.0
        self.cycle = 0
        self.pos = 0
        self.history: list[CycleSummary] = []
        self._reset_stats()
        self.thermal = module.R_th is not None and module.C_th is not None
        if self.thermal:
            self.a_th = math.exp(-self.dt / (module.R_th * module.C_th))

    # ------------------------------------------------------------------ energy <-> voltage
    def _setup_energy(self, c_fac: float) -> None:
        m = self.m
        self.cu = m.C0_cell_F is not None and m.k_cell_FperV is not None
        if self.cu:
            u = np.linspace(0.0, m.U_max / m.S * 1.05, 2001)
            e_cell = eq.cell_energy_cu(m.C0_cell_F * c_fac, m.k_cell_FperV * c_fac, u)
            self._U_tab = u * m.S
            self._E_tab = e_cell * m.n_cells
        else:
            self.C = m.C_module * c_fac

    def energy(self, U: float) -> float:
        if self.cu:
            return float(np.interp(U, self._U_tab, self._E_tab))
        return 0.5 * self.C * U * U

    def voltage(self, E: float) -> float:
        if self.cu:
            return float(np.interp(E, self._E_tab, self._U_tab))
        return math.sqrt(max(2.0 * E / self.C, 0.0))

    # ------------------------------------------------------------------ core loop
    def _advance(self, n_bins: int) -> CycleTrace:
        cfg, m = self.cfg, self.m
        dt, R = self.dt, self.R_mod
        pl, pb = self.load.mean.tolist(), self.bat.mean.tolist()
        sm, sq = self.sc.mean.tolist(), self.sc.mean_sq.tolist()
        smax, smin = self.sc.vmax.tolist(), self.sc.vmin.tolist()
        (o_t, o_pl, o_pb, o_ps, o_pr, o_u, o_i, o_dT, o_of, o_pc, o_is, o_E, o_ip, o_ut,
         o_ib) = (np.empty(n_bins) for _ in range(15))
        E, dT = self.E, self.dT
        Ksoc, Pr_max, Et = cfg.K_soc_per_s, cfg.P_rech_max_W, self.E_target
        Elo, Ehi = self.E_lo, self.E_hi
        Imax = m.I_max_A
        npc = m.n_cells
        th = self.thermal
        a_th = self.a_th if th else 0.0
        rth = m.R_th if th else 0.0
        cu = self.cu
        C = None if cu else self.C
        voltage = self.voltage
        stats = self._stats
        soc = cfg.soc_control
        eloss = cfg.esr_loss_in_energy
        ed, ec = cfg.eta_dch, cfg.eta_ch
        Ubus = cfg.U_bus_V
        t_lo = t_hi = t_ilim = t_plim = E_loss = E_of = E_conv = 0.0
        sqrt = math.sqrt
        k = self.pos
        n = self.n
        t_base = self.cycle * self.period
        R2 = 2.0 * R
        noR = R <= 1e-12

        def cur(p, U):
            """SC current for terminal power p at internal voltage U (ESR drop). Returns (i, limited)."""
            if noR:
                return p / U, False
            if p >= 0:
                d = U * U - 4.0 * R * p
                if d < 0:
                    return U / R2, True
                return (U - sqrt(d)) / R2, False
            return -(sqrt(U * U - 4.0 * R * p) - U) / R2, False

        for j in range(n_bins):
            U = voltage(E) if cu else sqrt(2.0 * E / C)
            p_rech = 0.0
            if soc:
                p_rech = Ksoc * (Et - E)
                if p_rech > Pr_max:
                    p_rech = Pr_max
                elif p_rech < -Pr_max:
                    p_rech = -Pr_max
            smk = sm[k]
            ps = smk - p_rech                       # SC share on the bus side
            dis = ps > 0
            pc = ps / ed if dis else ps * ec        # power at the SC terminals
            i, plim = cur(pc, U)
            limited = plim
            if i > Imax:
                i, limited = Imax, True
            elif i < -Imax:
                i, limited = -Imax, True
            if plim:
                t_plim += dt
            overflow = 0.0
            if limited:
                if i == Imax or i == -Imax:
                    t_ilim += dt
                pc_new = (U - R * i) * i
                bus = pc_new * ed if pc_new > 0 else pc_new / ec
                overflow = ps - bus
                ps, pc = bus, pc_new
            # peak current inside the bin
            hi = smax[k] - p_rech
            lo = smin[k] - p_rech
            ph = hi / ed if hi > 0 else hi * ec
            plo = lo / ed if lo > 0 else lo * ec
            ipk = max(abs(cur(ph, U)[0]), abs(cur(plo, U)[0]))
            if ipk > Imax:
                ipk = Imax
                if not limited:
                    t_ilim += dt
            # mean square current inside the bin (RMS of the terminal power converted to current)
            msq = sq[k] - 2.0 * p_rech * smk + p_rech * p_rech
            msq = msq / (ed * ed) if dis else msq * ec * ec
            prms = sqrt(msq) if msq > 0 else 0.0
            irms = abs(cur(prms if pc >= 0 else -prms, U)[0])
            if irms > Imax:
                irms = Imax
            i2 = irms * irms
            if i2 < i * i:
                i2 = i * i
            loss = R * i2
            E_new = E - (pc + loss) * dt if eloss else E - pc * dt
            if E_new < Elo:
                # at U_C0min the voltage is held, the SC current keeps flowing (limited only by I_C0max), the
                # battery delivers the missing energy
                overflow += (Elo - E_new) / dt * (ed if dis else 1.0)
                E_new = Elo
                t_lo += dt
            elif E_new > Ehi:
                overflow -= (E_new - Ehi) / dt / (ec if not dis else 1.0)
                E_new = Ehi
                t_hi += dt
            E = E_new
            if th:
                dT = a_th * dT + (1.0 - a_th) * rth * (loss / npc)
            E_loss += loss * dt
            E_conv += abs(pc - ps) * dt
            E_of += (overflow if overflow > 0 else -overflow) * dt
            o_t[j] = t_base + k * dt
            o_pl[j] = pl[k]
            o_pb[j] = pb[k] + p_rech + overflow
            o_ps[j] = ps
            o_pr[j] = p_rech
            o_u[j] = U
            o_i[j] = sqrt(i2)
            o_dT[j] = dT
            o_of[j] = overflow
            o_pc[j] = pc
            o_is[j] = i
            o_E[j] = E
            o_ip[j] = ipk
            o_ut[j] = U - R * i
            o_ib[j] = ps / Ubus
            k += 1
            if k >= n:
                k = 0
                self.cycle += 1
                t_base = self.cycle * self.period
        self.pos = k
        for key, v in (("t_lo", t_lo), ("t_hi", t_hi), ("t_ilim", t_ilim), ("t_plim", t_plim), ("E_loss", E_loss),
                       ("E_of", E_of), ("E_conv", E_conv)):
            stats[key] += v
        self.E, self.dT = E, dT
        return CycleTrace(o_t, o_pl, o_pb, o_ps, o_pr, o_u, o_i, o_dT, o_of, o_pc, o_is, o_E, o_ip, o_ut, o_ib)

    def _reset_stats(self) -> None:
        self._stats = dict(t_lo=0.0, t_hi=0.0, t_ilim=0.0, t_plim=0.0, E_loss=0.0, E_of=0.0, E_conv=0.0)

    def run_cycle(self) -> tuple[CycleSummary, CycleTrace]:
        """Advance exactly one profile period (from the current position)."""
        self._reset_stats()
        U0, dT0, cyc = self.voltage(self.E), self.dT, self.cycle
        tr = self._advance(self.n)
        st = self._stats
        E0 = self.energy(U0)
        s = CycleSummary(
            cycle=cyc, U_start=U0, U_end=self.voltage(self.E), U_min=float(tr.U_sc.min()),
            U_max=float(tr.U_sc.max()), dT_start=dT0, dT_max=float(tr.dT.max()),
            P_bat_max_W=float(tr.P_bat.max()), P_bat_min_W=float(tr.P_bat.min()),
            P_bat_rms_W=float(np.sqrt(np.mean(tr.P_bat ** 2))), I_sc_rms_A=float(np.sqrt(np.mean(tr.I_sc_rms ** 2))),
            I_sc_peak_A=float(np.max(tr.I_sc_peak)), E_loss_J=st["E_loss"], E_overflow_J=st["E_of"],
            t_at_Umin_s=st["t_lo"], t_at_Umax_s=st["t_hi"], t_current_limit_s=st["t_ilim"],
            E_conv_loss_J=st["E_conv"],
            E_swing_J=float(max(tr.E_sc.max(), E0) - min(tr.E_sc.min(), E0)),
            I_sc_rms_cell_A=float(np.sqrt(np.mean(tr.I_sc_rms ** 2))) / self.m.P,
            U_term_min_V=float(tr.U_term.min()), t_power_limit_s=st["t_plim"],
            t_Uconv_min_s=(float(np.sum(tr.U_term < self.cfg.U_conv_min_V)) * self.dt
                           if self.cfg.U_conv_min_V else 0.0))
        self.history.append(s)
        return s, tr

    def cell_current_full(self, trace: CycleTrace, p_load: np.ndarray, p_bat: np.ndarray,
                          dt_full: float) -> np.ndarray:
        """Cell current at full time resolution for one periodic cycle.

        P_bus,SC(t) - P_rech -> P_SC(t) with the converter efficiency -> I_SC(t) from P_SC = (U_C - R I) I with
        U_C of the step (power balance U_bus I_bus = U_SC I_SC), limited to I_max, divided by P.
        """
        k = max(1, int(round(self.dt / dt_full)))
        n = self.n * k
        p_sc_bus = (p_load - p_bat)[:n]
        rech = np.repeat(trace.P_rech, k)[: len(p_sc_bus)]
        U = np.repeat(trace.U_sc, k)[: len(p_sc_bus)]
        ps = p_sc_bus - rech
        pc = np.where(ps > 0, ps / self.cfg.eta_dch, ps * self.cfg.eta_ch)
        i_mod, _ = current_array(pc, U, self.R_mod)
        i_mod = np.clip(i_mod, -self.m.I_max_A, self.m.I_max_A)
        return i_mod / self.m.P

    def run_until_steady(self, max_cycles: Optional[int] = None, accelerate_thermal: bool = True
                         ) -> tuple[list[CycleSummary], CycleTrace]:
        """Repeat the profile until U and dT are periodic. Returns all summaries and the last trace."""
        max_cycles = max_cycles or self.cfg.max_cycles
        tr = None
        for _ in range(max_cycles):
            s, tr = self.run_cycle()
            u_ok = abs(s.U_end - s.U_start) < self.cfg.tol_U_V
            t_ok = (not self.thermal) or abs(self.dT - s.dT_start) < self.cfg.tol_T_K
            if u_ok and t_ok:
                break
            if u_ok and self.thermal and accelerate_thermal and len(self.history) >= 2:
                aN = math.exp(-self.period / (self.m.R_th * self.m.C_th))
                F = self.dT - aN * s.dT_start
                self.dT = F / (1.0 - aN)
        return self.history, tr

    def stream(self, chunk_s: float = 5.0) -> Iterator[CycleTrace]:
        """Endless generator: yields the next chunk of the emulation, forever."""
        n = max(1, int(round(chunk_s / self.dt)))
        while True:
            self._reset_stats()
            yield self._advance(n)


def module_from_candidate(row: dict, U_min: float, U_max: float, U_init: float, I_C0max: float,
                          fit: Optional[dict] = None, use_cu: bool = False) -> ModuleModel:
    """Module model for the emulation. SC current limit = min(I_C0max, P I_cell,max)."""
    kw = {}
    if use_cu and fit is not None:
        kw = dict(C0_cell_F=float(fit["C0_F"]), k_cell_FperV=float(fit["k_FperV"]))
    rth, cth = row.get("R_th_KperW"), row.get("C_th_JperK")
    ok = rth is not None and not (isinstance(rth, float) and math.isnan(rth))
    return ModuleModel(S=int(row["S"]), P=int(row["P"]), C_cell_F=float(row["C_cell_used_F"]),
                       R_cell_Ohm=float(row["ESR_cell_mOhm"]) / 1000, U_min=U_min, U_max=U_max, U_init=U_init,
                       I_max_A=min(float(I_C0max), float(row["I_module_max_A"])),
                       R_th=float(rth) if ok else None, C_th=float(cth) if ok else None,
                       label=f"{row['cell_id']} {row['P']}/{row['S']}", **kw)
