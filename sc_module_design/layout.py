"""Mechanical layout of the module on a PCB.

Cells stand on the PCB (radial cells). P cells in parallel form one group, S groups are connected in series.

Series path (U routing): the board is split into a left and a right leg. The path starts at the front in
the middle of the board, runs through the left leg to the back (row by row, meandering), crosses to the right
leg at the back and returns to the front. Both module terminals (+ and -) therefore end at the front edge,
directly next to each other. Unused group slots are left free at the front centre, and the connectors are
placed in this free space so that the board does not grow (a front strip is only added if they do not fit).

Packing: 'square' (rows aligned) or 'hex' (every second cell row shifted by half a pitch, row distance
pitch sqrt(3)/2, denser for large arrays).

Orientation: 'auto' (smallest board, as square as possible), 'long' (depth from the front edge >= width,
narrow front) or 'wide' (wide front edge). Optional limits on width and depth select a variant that fits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

PACKINGS = {"hex": "Hexagonal (versetzte Reihen)", "square": "Quadratisch (Raster)"}
ORIENTATIONS = {"auto": "Auto (kleinste Fläche, möglichst quadratisch)",
                "long": "Eher lang (Tiefe >= Breite, schmale Front)",
                "wide": "Eher breit (breite Front, geringe Tiefe)"}
GROUP_SHAPES = {"auto": "Auto", "row": "Reihe (P Zellen nebeneinander)", "column": "Spalte (P Zellen hintereinander)",
                "block": "Block (möglichst quadratisch)"}


@dataclass
class PCBLayout:
    S: int
    P: int
    d_mm: float
    l_mm: float
    packing: str
    gap_mm: float
    margin_mm: float
    bands: int                    # number of group columns (2 legs x c)
    gx: int                       # cells of one group in x
    gy: int                       # cells of one group in y
    cols: int                     # cell columns
    rows: int                     # cell rows
    width_mm: float               # PCB width (x, along the front edge)
    depth_mm: float               # PCB depth (y, from the front edge to the back)
    height_mm: float              # cell length + PCB + clearance
    cells: list = field(default_factory=list)       # (x, y, group, index_in_group)
    path: list = field(default_factory=list)        # series path: + pin, group centres, - pin
    terminal_plus: tuple = (0.0, 0.0)
    terminal_minus: tuple = (0.0, 0.0)
    connectors: list = field(default_factory=list)  # dict(name, x, y, w, h)
    connectors_in_free_space: bool = True
    extra_depth_mm: float = 0.0
    free_slots: int = 0
    orientation: str = "auto"
    legs: tuple = (0, 0)          # groups in the left leg (front -> back) and in the right leg (back -> front)
    fits_limits: bool = True
    notes: list = field(default_factory=list)

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.depth_mm

    @property
    def volume_dm3(self) -> float:
        return self.width_mm * self.depth_mm * self.height_mm / 1e6

    @property
    def box_area_mm2(self) -> float:
        w, d, h = self.width_mm, self.depth_mm, self.height_mm
        return 2 * (w * d + w * h + d * h)

    @property
    def cell_area_mm2(self) -> float:
        """Envelope of the cell array without PCB margin and connector strip."""
        return (self.width_mm - 2 * self.margin_mm - self.gap_mm) * (
            self.depth_mm - 2 * self.margin_mm - self.extra_depth_mm)

    @property
    def aspect(self) -> float:
        """depth / width (> 1: long, < 1: wide)."""
        return self.depth_mm / self.width_mm

    @property
    def terminals_adjacent(self) -> bool:
        return True

    @property
    def fill_factor(self) -> float:
        return self.S * self.P * math.pi * (self.d_mm / 2) ** 2 / self.area_mm2

    def pcb_mass_kg(self, areal_mass_g_cm2: float) -> float:
        return self.area_mm2 / 100 * areal_mass_g_cm2 / 1000

    def label(self) -> str:
        return (f"{self.width_mm:.0f} x {self.depth_mm:.0f} x {self.height_mm:.0f} mm, "
                f"{PACKINGS.get(self.packing, self.packing)}, Gruppe {self.gx}x{self.gy}, {self.bands // 2} Spalte(n) "
                f"je Schenkel, {self.legs_text()}")

    def legs_text(self) -> str:
        return f"U: links {self.legs[0]} Gruppen nach hinten, rechts {self.legs[1]} nach vorne"


def _shapes(P: int, mode: str) -> list[tuple[int, int]]:
    divs = [(a, P // a) for a in range(1, P + 1) if P % a == 0]
    if mode == "row":
        return [(P, 1)]
    if mode == "column":
        return [(1, P)]
    if mode == "block":
        return [min(divs, key=lambda s: (abs(math.log(s[0] / s[1])), -s[0]))]
    if "x" in str(mode):
        a, b = (int(v) for v in str(mode).split("x"))
        return [(a, b)]
    return divs


def _dims(S, P, d, gap, m, hexp, c, gx, gy, conns):
    pitch = d + gap
    rp = pitch * math.sqrt(3) / 2 if hexp else pitch
    R = int(math.ceil(S / (2 * c)))
    if c > 1 and R % 2:
        R += 1                                  # meander must end at the inner side at the back
    F = 2 * c * R - S
    kL = F // 2
    kR = F - kL
    cols, rows = 2 * c * gx, R * gy
    width = 2 * m + cols * pitch + (pitch / 2 if hexp and rows > 1 else 0.0)
    depth_cells = (rows - 1) * rp + d
    wl, wr = min(kL, c) * gx * pitch, min(kR, c) * gx * pitch
    x_mid = m + c * gx * pitch
    # hex with several cell rows per group: the odd rows of the left neighbour are shifted into the free slot
    shift = pitch / 2 if (hexp and gy >= 2) else 0.0
    fx0, fx1 = x_mid - wl + shift + gap / 2, x_mid + wr - gap / 2
    free_w = max(0.0, fx1 - fx0) if wl + wr > 0 else 0.0
    free_d = m + gy * rp - gap
    rot = False
    if conns:
        variants = []
        for r in (False, True):                 # connectors as given or rotated by 90 degrees
            cw = sum((h if r else w) for _, w, h in conns) + gap * (len(conns) - 1)
            cd = max((w if r else h) for _, w, h in conns)
            if cw <= free_w:
                variants.append((max(0.0, cd - free_d), r, cw, cd))
        if variants:
            extra, rot, cw, cd = min(variants, key=lambda v: (v[0], v[1]))
            in_free = extra == 0.0
        else:
            cw = sum(w for _, w, _ in conns) + gap * (len(conns) - 1)
            cd = max(h for _, _, h in conns)
            extra, in_free = max(0.0, cd + gap - m), False
    else:
        cw = cd = 0.0
        extra, in_free = 0.0, True
    depth = 2 * m + depth_cells + extra
    return dict(c=c, gx=gx, gy=gy, R=R, F=F, kL=kL, kR=kR, cols=cols, rows=rows, width=width, depth=depth,
                extra=extra, in_free=in_free, wl=wl, wr=wr, cw=cw, cd=cd, rot=rot, free_w=free_w, fx0=fx0, fx1=fx1,
                area=width * depth)


def pcb_layout(S: int, P: int, d_mm: float, l_mm: float, packing: str = "hex", gap_mm: float = 0.5,
               margin_mm: float = 5.0, bands: int = 0, pcb_mm: float = 1.6, clearance_mm: float = 3.0,
               orientation: str = "auto", group_shape: str = "auto", connectors=(("Leistung", 25.0, 15.0),),
               max_width_mm: Optional[float] = None, max_depth_mm: Optional[float] = None,
               target_aspect: float = 2.0, max_aspect: float = 2.5, area_tol: float = 0.08) -> PCBLayout:
    """bands = 0: automatic choice of the number of group columns, else 2c columns (c per leg).

    connectors: sequence of (name, width_mm, depth_mm); width 0 entries are ignored.
    """
    conns = tuple((str(n), float(w), float(h)) for n, w, h in (connectors or ()) if w and h and w > 0 and h > 0)
    return _cached(int(S), int(P), float(d_mm), float(l_mm), packing, float(gap_mm), float(margin_mm), int(bands or 0),
                   float(pcb_mm), float(clearance_mm), orientation, str(group_shape), conns,
                   None if max_width_mm is None else float(max_width_mm),
                   None if max_depth_mm is None else float(max_depth_mm),
                   float(target_aspect), float(max_aspect), float(area_tol))


@lru_cache(maxsize=4096)
def _cached(S, P, d, l, packing, gap, m, bands, pcb_mm, clearance, orientation, group_shape, conns, max_w, max_d,
            target, max_aspect, area_tol) -> PCBLayout:
    hexp = packing == "hex"
    cs = [max(1, (int(bands) + 1) // 2)] if bands else list(range(1, max(1, int(math.ceil(S / 2))) + 1))
    opts = [_dims(S, P, d, gap, m, hexp, c, gx, gy, conns) for gx, gy in _shapes(P, group_shape) for c in cs]
    notes = []
    fits = [o for o in opts if (max_w is None or o["width"] <= max_w + 1e-9) and
            (max_d is None or o["depth"] <= max_d + 1e-9)]
    ok = bool(fits)
    pool = fits or opts
    if not ok:
        notes.append("Keine Variante passt in die Vorgabe für Breite/Tiefe.")
    if orientation == "long":
        sel = [o for o in pool if o["depth"] >= o["width"] * 0.999]
        key = lambda o: abs(math.log(o["depth"] / o["width"] / target))
    elif orientation == "wide":
        sel = [o for o in pool if o["width"] >= o["depth"] * 0.999]
        key = lambda o: abs(math.log(o["width"] / o["depth"] / target))
    else:
        sel = [o for o in pool if max(o["width"] / o["depth"], o["depth"] / o["width"]) <= max_aspect]
        key = lambda o: abs(math.log(o["depth"] / o["width"]))
    if not sel:
        if orientation in ("long", "wide"):
            notes.append(f"Ausrichtung '{orientation}' nicht möglich, nächstbeste Variante gewählt.")
        sel = pool
    amin = min(o["area"] for o in sel)
    tol = area_tol if orientation in ("long", "wide") else area_tol / 4
    near = [o for o in sel if o["area"] <= amin * (1 + tol)]
    best = min(near, key=lambda o: (key(o), o["area"], o["c"], o["gy"]))
    lay = _build(S, P, d, l, packing, gap, m, pcb_mm, clearance, best, conns)
    lay.orientation, lay.fits_limits = orientation, ok
    if not lay.connectors_in_free_space and conns:
        notes.append(f"Stecker passen nicht in den freien Platz, Platine {lay.extra_depth_mm:.1f} mm tiefer.")
    lay.notes = notes
    return lay


def _build(S, P, d, l, packing, gap, m, pcb_mm, clearance, o, conns) -> PCBLayout:
    hexp = packing == "hex"
    pitch = d + gap
    rp = pitch * math.sqrt(3) / 2 if hexp else pitch
    c, gx, gy, R = o["c"], o["gx"], o["gy"], o["R"]
    y0 = m + o["extra"]
    left, right = [], []
    for r in range(R):                                   # left leg: front -> back, starts at the inner side
        cols = range(c - 1, -1, -1) if r % 2 == 0 else range(c)
        left += [(col, r) for col in cols]
    for i, r in enumerate(range(R - 1, -1, -1)):         # right leg: back -> front, starts at the inner side
        cols = range(c, 2 * c) if i % 2 == 0 else range(2 * c - 1, c - 1, -1)
        right += [(col, r) for col in cols]
    slots = left[o["kL"]:] + (right[: len(right) - o["kR"]] if o["kR"] else right)
    cells, centres = [], []
    for g, (col, r) in enumerate(slots[:S]):
        pts = []
        for ky in range(gy):
            for kx in range(gx):
                cx, ry = col * gx + kx, r * gy + ky
                x = m + pitch / 2 + cx * pitch + (pitch / 2 if (hexp and ry % 2 == 1) else 0.0)
                y = y0 + d / 2 + ry * rp
                cells.append((x, y, g, ky * gx + kx))
                pts.append((x, y))
        centres.append((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    width, depth = o["width"], o["depth"]
    x_mid = m + c * gx * pitch
    con_list = []
    if conns:
        xc = (o["fx0"] + o["fx1"]) / 2 if o["cw"] <= o["free_w"] else x_mid
        x = xc - o["cw"] / 2
        for name, w, h in conns:
            if o["rot"]:
                w, h = h, w
            con_list.append(dict(name=name, x=x, y=0.0, w=w, h=h, rotated=bool(o["rot"])))
            x += w + gap
        main = con_list[0]
        if main["rotated"]:
            t_plus = (main["x"] + main["w"] * 0.5, main["h"] * 0.75)
            t_minus = (main["x"] + main["w"] * 0.5, main["h"] * 0.25)
        else:
            t_plus = (main["x"] + main["w"] * 0.25, main["h"] * 0.5)
            t_minus = (main["x"] + main["w"] * 0.75, main["h"] * 0.5)
    else:
        t_plus = (centres[0][0], m / 2)
        t_minus = (centres[-1][0], m / 2)
    path = [t_plus] + centres + [t_minus]
    n_left = min(S, len(left) - o["kL"])
    lay = PCBLayout(S, P, d, l, packing, gap, m, 2 * c, gx, gy, o["cols"], o["rows"], width, depth,
                    l + pcb_mm + clearance, cells, path, t_plus, t_minus, con_list, bool(o["in_free"]),
                    float(o["extra"]), int(o["F"]))
    lay.legs = (n_left, S - n_left)
    return lay
