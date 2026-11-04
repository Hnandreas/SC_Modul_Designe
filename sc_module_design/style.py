"""Figure style of the thesis (port of PlotWorkspace/styles/cosmetics.py).

8 pt serif font, square-bracket units, legend below the axes, UIBK colours,
IEEE widths 8 cm (single column) and 16 cm (double column).
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UIBK_BLUE = "#003361"
UIBK_ORANGE = "#f39200"
MAGENTA = "#c2003e"
DARKGREEN = "#00665f"
PINK = "#420066"
LIGHT_BLUE = "#18B6EE"
GREY = "#999999"
COLORS = [UIBK_BLUE, UIBK_ORANGE, MAGENTA, DARKGREEN, PINK, LIGHT_BLUE]

CM = 1 / 2.54
SINGLE = (8.0 * CM, 5.5 * CM)
SINGLE_SQUARE = (8.0 * CM, 8.0 * CM)
DOUBLE = (16.0 * CM, 8.25 * CM)
FONT_PT = 8

RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "stix",
    "figure.figsize": SINGLE,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.8,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.45,
    "grid.color": "#dbdbdb",
    "figure.constrained_layout.use": True,
    "lines.linewidth": 1.0,
    "axes.prop_cycle": mpl.cycler(color=COLORS),
    "savefig.dpi": 300,
}


def apply() -> None:
    plt.rcParams.update(RC)


def legend_below(ax, ncol: int = 4, bbox_y: float = -0.18, **kw):
    args = dict(loc="upper center", bbox_to_anchor=(0.5, bbox_y), ncol=ncol, frameon=False, fancybox=False,
                shadow=False, fontsize=FONT_PT)
    args.update(kw)
    return ax.legend(**args)


def sublabel(ax, text: str, y: float = -0.32) -> None:
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=FONT_PT)


apply()
