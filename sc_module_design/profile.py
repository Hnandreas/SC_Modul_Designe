"""Load-profile handling: import, resampling, statistics, peaks, spectrum, binning."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .paths import PROFILE_DIR, resolve

TIME_NAMES = ["time_s", "time", "t", "zeit", "time [s]", "t_s", "timems", "time_ms"]
POWER_NAMES = ["power_w", "power", "p", "leistung", "activepower", "p_w", "power [w]"]
VOLT_NAMES = ["voltage", "voltage_v", "u", "spannung", "outputvoltagev", "v"]
CURR_NAMES = ["current", "current_a", "i", "strom", "outputcurrenta"]


@dataclass
class PowerProfile:
    """Uniformly sampled load power. Positive = load demand, negative = regeneration."""
    t: np.ndarray
    p: np.ndarray
    name: str = "profile"
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0] + self.dt)

    def window(self, t_start: Optional[float] = None, t_end: Optional[float] = None) -> "PowerProfile":
        m = np.ones_like(self.t, dtype=bool)
        if t_start is not None:
            m &= self.t >= t_start
        if t_end is not None:
            m &= self.t <= t_end
        return PowerProfile(self.t[m] - self.t[m][0], self.p[m], self.name, self.source, dict(self.meta))

    def stats(self) -> dict:
        p = self.p
        dt = self.dt
        return dict(
            duration_s=self.duration, dt_s=dt, n=len(p),
            P_avg_W=float(np.mean(p)), P_peak_W=float(np.max(p)), P_min_W=float(np.min(p)),
            P_rms_W=float(np.sqrt(np.mean(p ** 2))),
            E_pos_J=float(np.sum(p[p > 0]) * dt), E_neg_J=float(np.sum(p[p < 0]) * dt),
            peak_to_avg=float(np.max(p) / np.mean(p)) if np.mean(p) != 0 else np.nan,
        )


def _pick(cols: list[str], names: list[str]) -> Optional[str]:
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def load_profile(path: str | Path, scale: float = 1.0, time_col: Optional[str] = None,
                 power_col: Optional[str] = None, time_unit: str = "auto", phase_factor: float = 1.0,
                 resample_dt: Optional[float] = None, name: Optional[str] = None) -> PowerProfile:
    """Read a profile from .csv/.csv.gz/.txt/.npz.

    CSV: time + power columns (auto-detected: zeit/leistung, Time_s/Power_W, ...). If no power
    column exists but voltage and current columns do, P = U I phase_factor.
    NPZ: arrays ``p`` and either ``t`` or ``dt``.
    """
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    name = name or path.name.split(".")[0]
    if path.suffix == ".npz":
        z = np.load(path)
        p = np.asarray(z["p"], dtype=float)
        t = np.asarray(z["t"], dtype=float) if "t" in z else np.arange(len(p)) * float(z["dt"])
    else:
        df = pd.read_csv(path, sep=None, engine="python") if path.suffix == ".txt" else pd.read_csv(path)
        tc = time_col or _pick(list(df.columns), TIME_NAMES) or df.columns[0]
        pc = power_col or _pick(list(df.columns), POWER_NAMES)
        t = df[tc].to_numpy(float)
        if pc is None:
            vc, ic = _pick(list(df.columns), VOLT_NAMES), _pick(list(df.columns), CURR_NAMES)
            if vc and ic:
                p = df[vc].to_numpy(float) * df[ic].to_numpy(float) * phase_factor
            else:
                p = df[df.columns[1]].to_numpy(float)
        else:
            p = df[pc].to_numpy(float)
        if time_unit == "ms" or (time_unit == "auto" and "ms" in tc.lower()):
            t = t / 1000.0
    p = p * scale
    ok = ~(np.isnan(t) | np.isnan(p))
    t, p = t[ok], p[ok]
    t, idx = np.unique(t, return_index=True)
    p = p[idx]
    dts = np.diff(t)
    uniform = np.allclose(dts, dts[0], rtol=1e-3, atol=1e-9)
    if resample_dt or not uniform:
        dt = resample_dt or float(np.median(dts))
        tu = np.arange(t[0], t[-1] + dt / 2, dt)
        p = np.interp(tu, t, p)
        t = tu
    t = t - t[0]
    return PowerProfile(t, p, name=name, source=str(path))


def list_profiles() -> list[Path]:
    exts = (".csv", ".gz", ".npz", ".txt")
    return sorted(p for p in PROFILE_DIR.rglob("*") if p.is_file() and p.name.endswith(exts)
                  and not p.name.startswith("."))


# ---------------------------------------------------------------------- analysis
def peak_analysis(prof: PowerProfile) -> dict:
    """Local maxima of the load power and quartile-based outliers (IECON slide 13, Eqs. stat1/stat2).

    Quartiles as in MATLAB (prctile, 'hazen' definition). The upper fence Q3 + 1.5 IQR is the
    1486 W limit between power peaks and high-power peaks of the IECON 2023 boxplot.
    """
    idx, _ = find_peaks(prof.p)
    pks = prof.p[idx]
    q1, q3 = np.percentile(pks, [25, 75], method="hazen")
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    whisker = float(pks[pks <= upper].max()) if np.any(pks <= upper) else float(q3)
    return dict(peaks=pks, t_peaks=prof.t[idx], q1=float(q1), median=float(np.median(pks)), n_peaks=int(len(pks)), q3=float(q3),
                upper_fence=float(upper), whisker_high=whisker, n_outliers=int(np.sum(pks > upper)),
                outliers=pks[pks > upper], max_peak=float(pks.max()))


def spectrum(prof: PowerProfile, remove_mean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum."""
    x = prof.p - np.mean(prof.p) if remove_mean else prof.p
    n = len(x)
    y = np.fft.rfft(x) / n
    a = np.abs(y)
    a[1:] *= 2
    f = np.fft.rfftfreq(n, prof.dt)
    return f, a


@dataclass
class Binned:
    """Bin-averaged signal that keeps mean and mean square (for exact RMS / Joule losses)."""
    dt: float
    mean: np.ndarray
    mean_sq: np.ndarray
    vmax: np.ndarray
    vmin: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.mean)) * self.dt

    @property
    def n(self) -> int:
        return len(self.mean)


def bin_signal(x: np.ndarray, dt: float, dt_bin: float) -> Binned:
    k = max(1, int(round(dt_bin / dt)))
    n = len(x) // k
    rest = len(x) - n * k
    xs = x[: n * k].reshape(n, k)
    mean, msq, vmax, vmin = xs.mean(1), (xs ** 2).mean(1), xs.max(1), xs.min(1)
    if rest:
        tail = x[n * k:]
        mean = np.append(mean, tail.mean())
        msq = np.append(msq, (tail ** 2).mean())
        vmax = np.append(vmax, tail.max())
        vmin = np.append(vmin, tail.min())
    return Binned(dt=k * dt, mean=mean, mean_sq=msq, vmax=vmax, vmin=vmin)


def decimate_for_plot(t: np.ndarray, y: np.ndarray, n_max: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation that keeps peaks visible in plots."""
    if len(y) <= n_max:
        return t, y
    k = int(np.ceil(len(y) / (n_max / 2)))
    n = len(y) // k
    ys = y[: n * k].reshape(n, k)
    ts = t[: n * k].reshape(n, k)
    imin, imax = ys.argmin(1), ys.argmax(1)
    rows = np.arange(n)
    first = np.minimum(imin, imax)
    second = np.maximum(imin, imax)
    tt = np.column_stack([ts[rows, first], ts[rows, second]]).ravel()
    yy = np.column_stack([ys[rows, first], ys[rows, second]]).ravel()
    return tt, yy
