"""Figure style of the dissertation.

Matches the Python figures of the thesis (e.g. legrobot_power_profile, esr_error_panel):
Times New Roman 8 pt, grey frame, light grid, ticks inside, UIBK colours, plain axis labels with
units in square brackets, bold panel labels (a), (b) at the top left, legend below without frame,
no titles, widths 8 cm (single column) and 16 cm (double column).
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UIBK_BLUE = "#003361"
UIBK_ORANGE = "#f39200"
MAGENTA = "#c2003e"
PINK = "#e07a9f"
DARKGREEN = "#00665f"
PURPLE = "#420066"
LIGHT_BLUE = "#18B6EE"
DARK_GREY = "#4d4d4d"
GREY = "#999999"
BAND = "#a9bdd3"
COLORS = [UIBK_BLUE, UIBK_ORANGE, MAGENTA, DARKGREEN, PURPLE, LIGHT_BLUE, DARK_GREY]

CM = 1 / 2.54
W1 = 8.0 * CM                 # single column
W2 = 16.0 * CM                # double column
SINGLE = (W1, 5.5 * CM)
DOUBLE = (W2, 6.8 * CM)
FONT_PT = 8
LW = 1.3

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
    "figure.dpi": 150,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.8,
    "axes.labelcolor": "black",
    "axes.titlesize": FONT_PT,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.color": "#dcdcdc",
    "axes.axisbelow": True,
    "figure.constrained_layout.use": True,
    "lines.linewidth": LW,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.columnspacing": 2.0,
    "axes.prop_cycle": mpl.cycler(color=COLORS),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply() -> None:
    plt.rcParams.update(RC)


def legend_below(ax_or_fig, handles=None, labels=None, ncol: int = 3, y: float = -0.02):
    """Legend centred below the figure (as in the thesis)."""
    fig = ax_or_fig if hasattr(ax_or_fig, "legend") and not hasattr(ax_or_fig, "plot") else ax_or_fig.figure
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            h, lab = ax.get_legend_handles_labels()
            for hh, ll in zip(h, lab):
                if ll not in labels and not ll.startswith("_"):
                    handles.append(hh)
                    labels.append(ll)
    return fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)


def panel_label(ax, text: str) -> None:
    """Bold panel label (a), (b), ... above the top left corner of the axes."""
    ax.text(-0.01, 1.03, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_PT,
            fontweight="bold")


apply()


def panel_legend(ax, ncol: int = 3, **kw):
    """Legend of one subplot, placed above its top right corner (does not cover data)."""
    h, lab = ax.get_legend_handles_labels()
    if not h:
        return None
    args = dict(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, frameon=False, borderaxespad=0.15,
                handlelength=1.8, columnspacing=1.4, fontsize=FONT_PT)
    args.update(kw)
    return ax.legend(h, lab, **args)
