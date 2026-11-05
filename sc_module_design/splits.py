"""Power-split strategies (battery = slow/base part, supercapacitor = rest).

Every strategy is registered with :func:`register_split` and declares its parameters,
so the GUI and the YAML presets pick it up automatically. Own strategies can be
added without touching this file: put a ``*.py`` file into ``plugins/`` at the
project root (see ``plugins/example_split_plugin.py``).

All strategies return ``p_bat`` and ``p_sc = p - p_bat``.
Sign convention: positive power = delivered to the load (discharge), negative = regeneration.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import signal

from .paths import PLUGIN_DIR


@dataclass
class Param:
    name: str
    default: object
    label: str
    kind: str = "float"            # float | int | bool | choice | text
    choices: tuple = ()
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class SplitStrategy:
    name: str
    label: str
    func: Callable
    params: list = field(default_factory=list)
    description: str = ""
    kind: str = "filter"           # filter (zero mean SC power) | peak_shaver (one-sided)

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def __call__(self, t: np.ndarray, p: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray]:
        args = self.defaults()
        args.update({k: v for k, v in kw.items() if k in args})
        p_bat = np.asarray(self.func(t, p, **args), dtype=float)
        return p_bat, p - p_bat


REGISTRY: dict[str, SplitStrategy] = {}


def register_split(name: str, label: str, params: Optional[list] = None, description: str = "",
                   kind: str = "filter"):
    def deco(func):
        REGISTRY[name] = SplitStrategy(name, label, func, params or [], description, kind)
        return func
    return deco


def get_split(name: str) -> SplitStrategy:
    load_plugins()
    if name not in REGISTRY:
        raise KeyError(f"Unknown split strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]


def apply_split(name: str, t: np.ndarray, p: np.ndarray, params: Optional[dict] = None):
    return get_split(name)(t, p, **(params or {}))


# ---------------------------------------------------------------------- helpers
def _moving_sum_window(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(p)])
    lo = np.clip(lo, 0, len(p))
    hi = np.clip(hi, 0, len(p))
    return c[hi] - c[lo]


# ---------------------------------------------------------------------- strategies
@register_split(
    "moving_average", "Moving average (battery = moving average)",
    [Param("window_s", 20.0, "Window length [s]", min=0.001),
     Param("alignment", "legacy_matlab", "Alignment", "choice",
           ("legacy_matlab", "causal", "centered", "periodic_causal"),
           help="legacy_matlab: conv(...,'same') + circshift(-N/2) as in the IECON 2023 script "
                "(forward-looking window, reproduces the published numbers). causal: Eq. (movave) of the "
                "thesis, mean of the last N samples (real-time capable). centered: symmetric window. "
                "periodic_causal: causal with periodic continuation (endless profile).")],
    "Battery supplies the moving average, the SC the deviation from it (charge and discharge).")
def _moving_average(t, p, window_s=20.0, alignment="legacy_matlab"):
    dt = t[1] - t[0]
    n = max(1, int(round(window_s / dt)))
    L = len(p)
    k = np.arange(L)
    if alignment == "legacy_matlab":
        # MATLAB: y = conv(p, ones(N,1)/N, 'same'); y = circshift(y, -floor(N/2))
        half = n // 2
        y_same = _moving_sum_window(p, k + half + 1 - n, k + half + 1) / n
        return np.roll(y_same, -half)
    if alignment == "causal":
        cnt = np.minimum(k + 1, n)
        return _moving_sum_window(p, k + 1 - n, k + 1) / cnt
    if alignment == "centered":
        lo, hi = k - n // 2, k - n // 2 + n
        cnt = np.clip(hi, 0, L) - np.clip(lo, 0, L)
        return _moving_sum_window(p, lo, hi) / cnt
    if alignment == "periodic_causal":
        pp = np.concatenate([p[-n:], p]) if n <= L else np.tile(p, int(np.ceil(n / L)) + 1)[-(L + n):]
        c = np.concatenate([[0.0], np.cumsum(pp)])
        off = len(pp) - L
        return (c[k + off + 1] - c[k + off + 1 - n]) / n
    raise ValueError(f"unknown alignment {alignment}")


@register_split(
    "global_mean", "Ideal average split (battery = P_avg)", [],
    "Battery supplies exactly the profile average, the SC everything else (Eq. sc_transient_power).")
def _global_mean(t, p):
    return np.full_like(p, np.mean(p))


@register_split(
    "peak_shaver", "Peak shaver (battery limited to a power band)",
    [Param("p_upper_W", 1500.0, "Upper battery power limit [W]"),
     Param("p_lower_W", -1e9, "Lower battery power limit [W] (regeneration)",
           help="-1e9 = battery takes all regeneration. 0 = SC absorbs all regeneration.")],
    "Battery follows the load inside [p_lower, p_upper]; everything outside goes to the SC. "
    "One-sided SC power: needs a recharge strategy in continuous operation (see emulation).",
    kind="peak_shaver")
def _peak_shaver(t, p, p_upper_W=1500.0, p_lower_W=-1e9):
    return np.clip(p, p_lower_W, p_upper_W)


@register_split(
    "lowpass_1st", "First-order low-pass",
    [Param("fc_Hz", 0.05, "Cut-off frequency [Hz]", min=1e-6),
     Param("zero_phase", False, "Zero phase (filtfilt, not causal)", "bool")],
    "Battery = first-order low-pass of the load.")
def _lowpass_1st(t, p, fc_Hz=0.05, zero_phase=False):
    dt = t[1] - t[0]
    a = np.exp(-2 * np.pi * fc_Hz * dt)
    b, aa = [1 - a], [1, -a]
    if zero_phase:
        return signal.filtfilt(b, aa, p)
    zi = signal.lfilter_zi(b, aa) * p[0]
    return signal.lfilter(b, aa, p, zi=zi)[0]


@register_split(
    "butterworth", "Butterworth low-pass",
    [Param("fc_Hz", 0.05, "Cut-off frequency [Hz]", min=1e-6),
     Param("order", 2, "Order", "int", min=1, max=8),
     Param("zero_phase", False, "Zero phase (filtfilt, not causal)", "bool")],
    "Battery = Butterworth low-pass of the load.")
def _butterworth(t, p, fc_Hz=0.05, order=2, zero_phase=False):
    fs = 1.0 / (t[1] - t[0])
    sos = signal.butter(int(order), fc_Hz, fs=fs, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, p)
    zi = signal.sosfilt_zi(sos) * p[0]
    return signal.sosfilt(sos, p, zi=zi)[0]


@register_split(
    "rate_limiter", "Battery slew-rate limiter",
    [Param("max_slope_W_per_s", 200.0, "Max. battery power slope [W/s]", min=1e-6)],
    "Battery follows the load with a limited rate of change, the SC covers the difference.")
def _rate_limiter(t, p, max_slope_W_per_s=200.0):
    dt = t[1] - t[0]
    step = max_slope_W_per_s * dt
    out = np.empty_like(p)
    y = p[0]
    for i, x in enumerate(p.tolist()):
        d = x - y
        y += step if d > step else (-step if d < -step else d)
        out[i] = y
    return out


@register_split(
    "custom_expression", "Custom (Python expression)",
    [Param("expression", "np.clip(p, -500, 1500)", "Expression for p_bat", "text",
           help="Variables: t, p, dt (arrays/float), np, signal (scipy.signal). The result is the battery "
                "power p_bat, the SC gets p - p_bat. Example: signal.savgol_filter(p, 200001, 1)")],
    "Own filter as a Python expression (local use only).")
def _custom_expression(t, p, expression="p"):
    env = {"np": np, "signal": signal, "t": t, "p": p, "dt": float(t[1] - t[0])}
    return np.broadcast_to(np.asarray(eval(expression, {"__builtins__": {}}, env), dtype=float), p.shape).copy()


# ---------------------------------------------------------------------- plugins
_PLUGINS_LOADED = False


def load_plugins(force: bool = False) -> list[str]:
    """Import every ``plugins/*.py`` file once (they call register_split themselves)."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED and not force:
        return []
    loaded = []
    if PLUGIN_DIR.exists():
        for f in sorted(PLUGIN_DIR.glob("*.py")):
            if f.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(f"scmd_plugin_{f.stem}", f)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                loaded.append(f.name)
            except Exception as exc:  # a broken plugin must not break the tool
                print(f"[plugins] {f.name} failed: {exc}")
    _PLUGINS_LOADED = True
    return loaded


def periodic_emulation_split(name, t, p, params=None):
    """Periodic continuation for emulation only; analytic/reference splits stay unchanged.

    Causal filter states are solved at the periodic boundary. No zero padding or
    fresh filter startup is inserted every time the recorded profile wraps.
    """
    args = get_split(name).defaults()
    args.update(params or {})
    if name == 'moving_average':
        args['alignment'] = 'periodic_causal'
        return apply_split(name, t, p, args)
    if name in ('lowpass_1st', 'butterworth'):
        fs = 1 / float(t[1] - t[0])
        order = int(args.get('order', 1))
        if name == 'lowpass_1st':
            a = np.exp(-2 * np.pi * args['fc_Hz'] / fs)
            out, z = signal.lfilter([1 - a], [1, -a], p, zi=[0.0])
            den = -np.expm1(len(p) * np.log(a)) if a > 0 else 1.0
            out, _ = signal.lfilter([1 - a], [1, -a], p, zi=z / den)
        else:
            sos = signal.butter(order, args['fc_Hz'], fs=fs, output='sos')
            shape = (len(sos), 2)
            _, z0 = signal.sosfilt(sos, p, zi=np.zeros(shape))
            size = z0.size
            matrix = np.empty((size, size))
            for j in range(size):
                basis = np.zeros(size)
                basis[j] = 1
                _, zj = signal.sosfilt(sos, p, zi=basis.reshape(shape))
                matrix[:, j] = (zj - z0).ravel()
            initial = np.linalg.solve(np.eye(size) - matrix, z0.ravel()).reshape(shape)
            out, _ = signal.sosfilt(sos, p, zi=initial)
        return out, p - out
    if name == 'rate_limiter':
        from scipy.optimize import brentq
        step = args['max_slope_W_per_s'] * (t[1] - t[0])
        def run(initial):
            out = np.empty_like(p)
            y = initial
            for j, value in enumerate(p):
                y += np.clip(value - y, -step, step)
                out[j] = y
            return out
        lo, hi = float(np.min(p)), float(np.max(p))
        initial = lo if lo == hi else brentq(lambda v: run(v)[-1] - v, lo, hi)
        out = run(initial)
        return out, p - out
    return apply_split(name, t, p, args)
