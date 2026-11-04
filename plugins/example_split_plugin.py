"""Example plugin: own power-split strategy.

Every *.py file in this folder is loaded automatically by the tool (GUI and CLI). Register a function
that returns the battery power p_bat; the supercapacitor gets p - p_bat.
"""
import numpy as np

from sc_module_design.splits import Param, register_split


@register_split(
    "dead_band_mean", "Example: moving average with dead band",
    [Param("window_s", 10.0, "Window length [s]", min=0.01),
     Param("band_W", 100.0, "Dead band around the average [W]", min=0.0)],
    "Battery follows the load inside +-band around the moving average, the SC covers the rest.")
def dead_band_mean(t, p, window_s=10.0, band_W=100.0):
    dt = t[1] - t[0]
    n = max(1, int(round(window_s / dt)))
    c = np.concatenate([[0.0], np.cumsum(p)])
    k = np.arange(len(p))
    lo = np.clip(k + 1 - n, 0, None)
    avg = (c[k + 1] - c[lo]) / (k + 1 - lo)
    return np.clip(p, avg - band_W, avg + band_W)
