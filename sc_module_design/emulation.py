"""Continuous HESS precheck, with a voltage PI and a charge-positive SC current.

I_SC > 0 charges the capacitor; U_terminal = U_C + R I_SC.
P_SC > 0 supplies the DC bus (the existing load-power convention is retained).
Every actual terminal joule and ESR-loss joule is included in stored energy.
Neither voltage limits nor the PI reference are ideal energy sources/sinks.

The user-specified voltage-loop clock is 100 kHz / 5 = 20 kHz. Within
one numerical bin the measured voltage is held; the discrete firmware PI calls
are aggregated analytically, including conditional integration and saturation.
The clock remainder, integrator, energy and temperature survive all cycle/chunk
boundaries. This is an averaged plant, not a switching simulation.
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
    C_cell_F: float
    R_cell_Ohm: float
    U_min: float
    U_max: float
    U_init: float
    I_max_A: float
    C0_cell_F: Optional[float] = None
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
    soc_control: bool = True  # voltage PI; legacy argument retained, no energy-P controller
    soc_target: str = "U_C0inM"
    voltage_reference_V: Optional[float] = None
    initial_voltage_V: Optional[float] = None
    pi_kp: float = 0.2
    pi_ki: float = 0.5
    pi_min_A: float = -5.0
    pi_max_A: float = 5.0
    switching_frequency_Hz: float = 100_000.0
    pi_every_pwm_cycles: int = 5
    eol: bool = False
    L_EOL: float = 0.7  # existing design assumption; not changed by emulation
    esr_eol_factor: float = 2.0
    aging_capacity_factor: Optional[float] = None
    aging_esr_factor: Optional[float] = None
    T_amb_C: float = 25.0
    eta_dch: float = 1.0
    eta_ch: float = 1.0
    esr_loss_in_energy: bool = True
    U_bus_V: float = 48.0
    converter_max_duty: float = 0.98  # Interleaved-6 inner PI upper duty clamp
    U_conv_min_V: Optional[float] = None
    max_cycles: int = 200
    tol_U_V: float = 1e-3
    tol_T_K: float = 0.02
    tol_pi_A: float = 1e-3

    @property
    def pi_period_s(self) -> float:
        return self.pi_every_pwm_cycles / self.switching_frequency_Hz

    def __post_init__(self):
        if self.switching_frequency_Hz <= 0 or self.pi_every_pwm_cycles < 1:
            raise ValueError("PI clock requires positive switching frequency and divider")
        if self.pi_min_A >= self.pi_max_A or self.pi_kp < 0 or self.pi_ki < 0:
            raise ValueError("Invalid PI gains or output limits")
        if self.U_bus_V <= 0 or not 0 < self.converter_max_duty <= 1:
            raise ValueError("Invalid bus voltage or converter duty limit")
        if not self.esr_loss_in_energy:
            raise ValueError("The emulation must include ESR losses in stored energy")
        if not (0 < self.eta_dch <= 1 and 0 < self.eta_ch <= 1) or self.dt_bin_s <= 0:
            raise ValueError("Invalid efficiency or numerical timestep")


class VoltagePI:
    """Interleaved-6 conditional-integration PI, retimed by the user to 20 kHz.

    Ki is A/(V*s): each real PI call adds Ki*error*Ts, exactly once.
    Kp is A/V and is NOT multiplied by frequency or by the number of calls.
    advance() holds feedback across a plant bin and returns the mean command
    of its real PI calls. It is algebraically equivalent to calling the firmware
    PI n times at this held error, without a Python loop at 20 kHz.
    """
    def __init__(self, cfg: EmulationConfig):
        self.cfg = cfg
        self.integral = 0.0
        self.output = 0.0
        self.remainder_s = 0.0
        self.calls = 0

    @staticmethod
    def _sum_clipped(start, step, n, lo, hi):
        if n <= 0:
            return 0.0
        if step == 0:
            return n * min(hi, max(lo, start))
        if step < 0:
            start += (n - 1) * step
            step = -step
        left = min(n, max(0, int(math.ceil((lo - start) / step))))
        right = min(n, max(left, int(math.ceil((hi - start) / step))))
        count = right - left
        middle = count * (start + step * (left + right - 1) / 2)
        return left * lo + middle + (n - right) * hi

    def advance(self, error: float, elapsed_s: float) -> float:
        ts = self.cfg.pi_period_s
        total = self.remainder_s + elapsed_s
        n = int(math.floor(total / ts + 1e-9))
        self.remainder_s = max(0.0, total - n * ts)
        if n == 0:
            return self.output
        lo, hi = self.cfg.pi_min_A, self.cfg.pi_max_A
        y = self.cfg.pi_kp * error + self.integral
        delta = self.cfg.pi_ki * error * ts
        updates = 0
        if delta > 0 and y < hi:
            updates = min(n, max(0, int(math.ceil((hi - y) / delta))))
        elif delta < 0 and y > lo:
            updates = min(n, max(0, int(math.ceil((lo - y) / delta))))
        total_output = self._sum_clipped(y, delta, updates, lo, hi)
        frozen = min(hi, max(lo, y + updates * delta))
        total_output += (n - updates) * frozen
        last_y = y + min(updates, n - 1) * delta
        self.output = min(hi, max(lo, last_y))
        self.integral += updates * delta
        self.calls += n
        return total_output / n


@dataclass
class CycleTrace:
    t: np.ndarray
    P_load: np.ndarray
    P_bat: np.ndarray
    P_sc: np.ndarray                 # positive = supplies the bus
    P_rech: np.ndarray               # requested bus recharge contribution
    U_sc: np.ndarray                 # internal voltage at START of bin
    I_sc_rms: np.ndarray             # always nonnegative
    dT: np.ndarray
    overflow: np.ndarray
    P_sc_cell: np.ndarray            # module terminal output power, NOT per-cell power
    I_sc: np.ndarray                 # positive = charge, negative = discharge
    E_sc: np.ndarray                 # stored energy at END of bin
    I_sc_peak: np.ndarray
    U_term: np.ndarray               # U_C + R I_SC
    I_bus: np.ndarray                # charge-positive bus-side SC current
    I_pi: Optional[np.ndarray] = None
    pi_integral: Optional[np.ndarray] = None
    P_required: Optional[np.ndarray] = None
    P_available: Optional[np.ndarray] = None
    power_limited: Optional[np.ndarray] = None
    dt: Optional[np.ndarray] = None
    I_sc_full: Optional[np.ndarray] = None
    voltage_limited: Optional[np.ndarray] = None
    U_term_peak: Optional[np.ndarray] = None
    converter_range_violation: Optional[np.ndarray] = None
    unmet_discharge_W: Optional[np.ndarray] = None
    current_limited: Optional[np.ndarray] = None
    requested_current_extreme_A: Optional[np.ndarray] = None


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
    pi_integral_start_A: float = 0.0
    pi_integral_end_A: float = 0.0
    t_bus_voltage_limit_s: float = 0.0
    t_converter_range_s: float = 0.0


class HESSEmulator:
    def __init__(self, t, p_load, p_bat, module: ModuleModel, cfg=None):
        self.cfg = cfg or EmulationConfig()
        self.m = module
        if not (0 < module.U_min < module.U_max and module.I_max_A > 0):
            raise ValueError("Invalid existing module voltage/current limits")
        self.raw_dt = float(t[1] - t[0])
        self.load = bin_signal(p_load, self.raw_dt, self.cfg.dt_bin_s)
        self.bat = bin_signal(p_bat, self.raw_dt, self.cfg.dt_bin_s)
        self.sc = bin_signal(p_load - p_bat, self.raw_dt, self.cfg.dt_bin_s)
        self.dt, self.n = self.load.dt, self.load.n
        group = max(1, int(round(self.dt / self.raw_dt)))
        self.raw_sc = [np.asarray(p_load[a:a + group] - p_bat[a:a + group], float)
                       for a in range(0, len(p_load), group)]
        self.bin_dt = np.array([len(a) * self.raw_dt for a in self.raw_sc])
        self.period = float(self.bin_dt.sum())
        c_fac = self.cfg.aging_capacity_factor if self.cfg.aging_capacity_factor is not None else (self.cfg.L_EOL if self.cfg.eol else 1.0)
        r_fac = self.cfg.aging_esr_factor if self.cfg.aging_esr_factor is not None else (self.cfg.esr_eol_factor if self.cfg.eol else 1.0)
        self.R_mod = module.R_module * r_fac
        self.R_cell_eff = module.R_cell_Ohm * r_fac
        self._setup_energy(c_fac)
        initial = self.cfg.initial_voltage_V if self.cfg.initial_voltage_V is not None else module.U_init
        if not module.U_min <= initial <= module.U_max:
            raise ValueError("Initial voltage is outside the existing module limits")
        self.E = self.energy(initial)
        self.U_reference = self.cfg.voltage_reference_V
        if self.U_reference is None:
            self.U_reference = module.U_max if self.cfg.soc_target == "U_C0max" else module.U_init
        if not module.U_min <= self.U_reference <= module.U_max:
            raise ValueError("Voltage reference is outside the existing module limits")
        self.E_lo, self.E_hi = self.energy(module.U_min), self.energy(module.U_max)
        self.pi = VoltagePI(self.cfg)
        self.last_current = 0.0
        self.dT = 0.0
        self.cycle = self.pos = 0
        self.elapsed = 0.0
        self.history = []
        self.converged = False
        self._reset_stats()
        self.thermal = module.R_th is not None and module.C_th is not None

    def _setup_energy(self, c_fac):
        m = self.m
        self.cu = m.C0_cell_F is not None and m.k_cell_FperV is not None
        if self.cu:
            u = np.linspace(0.0, m.U_max / m.S * 1.05, 2001)
            self._U_tab = u * m.S
            self._E_tab = eq.cell_energy_cu(m.C0_cell_F * c_fac, m.k_cell_FperV * c_fac, u) * m.n_cells
        else:
            self.C = m.C_module * c_fac

    def energy(self, U):
        return float(np.interp(U, self._U_tab, self._E_tab)) if self.cu else 0.5 * self.C * U * U

    def voltage(self, E):
        return float(np.interp(E, self._E_tab, self._U_tab)) if self.cu else math.sqrt(max(2 * E / self.C, 0))

    def _to_terminal(self, p_bus):
        return np.where(p_bus > 0, p_bus / self.cfg.eta_dch, p_bus * self.cfg.eta_ch)

    def _to_bus(self, p_terminal):
        return np.where(p_terminal > 0, p_terminal * self.cfg.eta_dch, p_terminal / self.cfg.eta_ch)

    def _advance(self, n_bins):
        fields = [np.empty(n_bins) for _ in range(21)]
        (ot, opl, opb, ops, opr, ou, oi, odt, oof, opc, ois, oe, oip, out, oib,
         opi, oint, oreq, oavail, olim, oduration) = fields
        raw_currents = []
        voltage_limited = np.empty(n_bins, dtype=bool)
        terminal_peak = np.empty(n_bins)
        unmet_discharge = np.empty(n_bins)
        current_limited = np.empty(n_bins, dtype=bool)
        requested_extreme = np.empty(n_bins)
        range_violation = np.empty(n_bins, dtype=bool)
        R, cfg, m = self.R_mod, self.cfg, self.m
        converter_ceiling = cfg.converter_max_duty * cfg.U_bus_V
        converter_energy_ceiling = self.energy(min(m.U_max, converter_ceiling))
        for j in range(n_bins):
            k, E = self.pos, self.E
            dt, U = self.bin_dt[k], self.voltage(E)
            feedback = U + R * self.last_current
            ipi = self.pi.advance(self.U_reference - feedback, dt) if cfg.soc_control else 0.0
            base_pc = self._to_terminal(self.raw_sc[k])
            # current.py retains its discharge-positive convention for analytic sizing.
            base_i, _ = current_array(base_pc, U, R)
            base_i = -base_i
            requested = base_pc - (U + R * (2 * base_i + ipi)) * ipi
            wanted_i = base_i + ipi
            available = U * U / (4 * R) if R > 1e-12 else float("inf")
            plim = bool(np.any(requested > available + 1e-9))
            # Remain on the stable constant-power branch; keep existing converter limits.
            lo = -min(m.I_max_A, U / (2 * R)) if R > 1e-12 else -m.I_max_A
            hi = m.I_max_A
            # Averaged buck charge port: U_C + R*I <= D_max*U_bus.
            # A pre-existing U_C above that ceiling must NOT force a discharge.
            if R > 1e-12:
                charge_headroom_A = max(0.0, (converter_ceiling - U) / R)
            else:
                charge_headroom_A = m.I_max_A if U <= converter_ceiling else 0.0
            voltage_limited[j] = bool(np.any(wanted_i > charge_headroom_A + 1e-9))
            hi = min(hi, charge_headroom_A)
            # Also avoid stepping the internal voltage past the converter's
            # reachable charging voltage within a plant bin (including R = 0).
            hi = min(hi, max(0.0, (converter_energy_ceiling - E) / (U * dt)))
            ilim = bool(np.any(np.abs(wanted_i) > m.I_max_A))
            current_limited[j] = ilim
            requested_extreme[j] = wanted_i[int(np.argmax(np.abs(wanted_i)))]
            # Bound actual current/energy, not just the plotted voltage.
            lo = max(lo, (self.E_lo - E) / (U * dt))
            hi = min(hi, (self.E_hi - E) / (U * dt))
            actual_i = np.clip(wanted_i, lo, hi)
            imean, i2 = float(actual_i.mean()), float(np.mean(actual_i ** 2))
            pc = -(U + R * actual_i) * actual_i
            bus = self._to_bus(pc)
            req_bus = self._to_bus(requested)
            unmet_discharge[j] = float(np.max(np.where(req_bus > 0, np.maximum(req_bus - bus, 0), 0)))
            loss = R * i2
            pcmean, ps = float(pc.mean()), float(bus.mean())
            overflow = float((req_bus - bus).mean())
            Enew = E - (pcmean + loss) * dt
            # Roundoff only: all meaningful limiting was applied to current above.
            self.E = min(self.E_hi, max(self.E_lo, Enew))
            if self.thermal:
                decay = math.exp(-dt / (m.R_th * m.C_th))
                self.dT = decay * self.dT + (1 - decay) * m.R_th * loss / m.n_cells
            st = self._stats
            st['t_lo'] += dt if self.E <= self.E_lo + 1e-8 else 0
            st['t_hi'] += dt if self.E >= self.E_hi - 1e-8 else 0
            st['t_ilim'] += dt if ilim else 0
            st['t_plim'] += dt if plim else 0
            st['E_loss'] += loss * dt
            st['E_of'] += float(np.mean(np.abs(req_bus - bus))) * dt
            st['E_conv'] += float(np.mean(np.abs(pc - bus))) * dt
            ot[j], opl[j], opb[j], ops[j] = self.elapsed, self.load.mean[k], self.load.mean[k] - ps, ps
            opr[j] = float((self.raw_sc[k] - req_bus).mean())
            ou[j], oi[j], odt[j], oof[j], opc[j] = U, math.sqrt(i2), self.dT, overflow, pcmean
            ois[j], oe[j], oip[j] = imean, self.E, float(np.max(np.abs(actual_i)))
            out[j], oib[j] = U + R * imean, -ps / cfg.U_bus_V
            terminal_peak[j] = float(np.max(U + R * actual_i))
            range_violation[j] = terminal_peak[j] > converter_ceiling + 1e-8
            opi[j], oint[j] = ipi, self.pi.integral
            oreq[j], oavail[j], olim[j], oduration[j] = float(requested.max()), available, plim, dt
            raw_currents.append(actual_i)
            self.last_current = float(actual_i[-1])
            self.elapsed += dt
            self.pos += 1
            if self.pos == self.n:
                self.pos = 0
                self.cycle += 1
        return CycleTrace(*fields, I_sc_full=np.concatenate(raw_currents) if raw_currents else np.array([]),
                          voltage_limited=voltage_limited, U_term_peak=terminal_peak,
                          converter_range_violation=range_violation, unmet_discharge_W=unmet_discharge,
                          current_limited=current_limited, requested_current_extreme_A=requested_extreme)

    def _reset_stats(self):
        self._stats = dict(t_lo=0.0, t_hi=0.0, t_ilim=0.0, t_plim=0.0, E_loss=0.0, E_of=0.0, E_conv=0.0)

    def run_cycle(self):
        self._reset_stats()
        U0, dT0, cyc, pi0 = self.voltage(self.E), self.dT, self.cycle, self.pi.integral
        E0 = self.E
        tr = self._advance(self.n)
        st = self._stats
        rms = math.sqrt(float(np.average(tr.I_sc_rms ** 2, weights=tr.dt)))
        s = CycleSummary(
            cycle=cyc, U_start=U0, U_end=self.voltage(self.E),
            U_min=min(U0, self.voltage(float(tr.E_sc.min()))), U_max=max(U0, self.voltage(float(tr.E_sc.max()))),
            dT_start=dT0, dT_max=float(tr.dT.max()), P_bat_max_W=float(tr.P_bat.max()),
            P_bat_min_W=float(tr.P_bat.min()), P_bat_rms_W=float(np.sqrt(np.average(tr.P_bat ** 2, weights=tr.dt))),
            I_sc_rms_A=rms, I_sc_peak_A=float(tr.I_sc_peak.max()), E_loss_J=st['E_loss'],
            E_overflow_J=st['E_of'], t_at_Umin_s=st['t_lo'], t_at_Umax_s=st['t_hi'],
            t_current_limit_s=st['t_ilim'], E_conv_loss_J=st['E_conv'],
            E_swing_J=float(max(tr.E_sc.max(), E0) - min(tr.E_sc.min(), E0)), I_sc_rms_cell_A=rms / self.m.P,
            U_term_min_V=float(tr.U_term.min()), t_power_limit_s=st['t_plim'],
            t_Uconv_min_s=float(tr.dt[tr.U_term < self.cfg.U_conv_min_V].sum()) if self.cfg.U_conv_min_V else 0,
            pi_integral_start_A=pi0, pi_integral_end_A=self.pi.integral,
            t_bus_voltage_limit_s=float(tr.dt[tr.voltage_limited].sum()),
            t_converter_range_s=float(tr.dt[tr.converter_range_violation].sum()))
        self.history.append(s)
        return s, tr

    def cell_current_full(self, trace, p_load, p_bat, dt_full):
        """Actual, limited charge-positive current, including recharge and voltage boundaries."""
        if trace.I_sc_full is None:
            raise ValueError("Trace does not contain the actual full-resolution SC current")
        return trace.I_sc_full / self.m.P

    def run_until_steady(self, max_cycles=None, accelerate_thermal=True):
        stable = 0
        self.converged = False
        for _ in range(max_cycles or self.cfg.max_cycles):
            s, tr = self.run_cycle()
            u_ok = abs(s.U_end - s.U_start) < self.cfg.tol_U_V
            pi_ok = abs(s.pi_integral_end_A - s.pi_integral_start_A) < self.cfg.tol_pi_A
            t_ok = not self.thermal or abs(self.dT - s.dT_start) < self.cfg.tol_T_K
            stable = stable + 1 if u_ok and pi_ok and t_ok else 0
            if stable >= 2:
                self.converged = True
                break
            if u_ok and pi_ok and self.thermal and accelerate_thermal and len(self.history) >= 2:
                aN = math.exp(-self.period / (self.m.R_th * self.m.C_th))
                self.dT = (self.dT - aN * s.dT_start) / (1 - aN)
        return self.history, tr

    def stream(self, chunk_s=5.0) -> Iterator[CycleTrace]:
        while True:
            self._reset_stats()
            yield self._advance(max(1, int(round(chunk_s / self.dt))))


def module_from_candidate(row, U_min, U_max, U_init, I_C0max, fit=None, use_cu=False):
    kw = {}
    if use_cu and fit is not None:
        kw = dict(C0_cell_F=float(fit['C0_F']), k_cell_FperV=float(fit['k_FperV']))
    rth, cth = row.get('R_th_KperW'), row.get('C_th_JperK')
    ok = rth is not None and not (isinstance(rth, float) and math.isnan(rth))
    return ModuleModel(S=int(row['S']), P=int(row['P']), C_cell_F=float(row['C_cell_used_F']),
                       R_cell_Ohm=float(row['ESR_cell_mOhm']) / 1000, U_min=U_min, U_max=U_max, U_init=U_init,
                       I_max_A=min(float(I_C0max), float(row['I_module_max_A'])),
                       R_th=float(rth) if ok else None, C_th=float(cth) if ok else None,
                       label=f"{row['cell_id']} {row['P']}/{row['S']}", **kw)
