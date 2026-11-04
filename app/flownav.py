"""Clickable flowchart navigation: the original flowchart of the dissertation (Skript.pdf) with status markers.

Each box links to the page of the GUI that handles this step. Markers: green = calculated, amber = warning,
red = action needed (error, missing confirmation), grey = skipped (missing data), white = open.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:                                    # optional component (requirements.txt), fallback without clicks
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:                       # pragma: no cover
    streamlit_image_coordinates = None

IMG = Path(__file__).resolve().parent / "assets" / "flowchart.png"
SCALE = 200 / 72                        # PNG rendered at 200 dpi from the PDF (points)

# name, box in PDF points (x0, y0, x1, y1), page, status key
BOXES = [
    ("req", (93.4, 9.4, 165.6, 33.4), "profil", "0"),
    ("db", (185.2, 1.9, 258.1, 41.0), "db", "db"),
    ("s1", (93.4, 44.5, 165.6, 71.1), "spannung", "1"),
    ("s2", (93.4, 78.8, 165.6, 98.8), "spannung", "2"),
    ("s3", (93.4, 104.4, 165.6, 123.6), "spannung", "3"),
    ("s4", (93.4, 129.2, 165.6, 148.2), "kapazitaet", "4"),
    ("s5", (93.4, 161.3, 165.6, 175.1), "zellen", "5-11"),
    ("s6", (87.0, 180.7, 172.0, 200.4), "zellen", "5-11"),
    ("s7", (76.0, 208.2, 183.0, 237.8), "thermik", "th"),
    ("s8", (87.0, 246.6, 172.0, 273.7), "thermik", "th"),
    ("s9", (86.4, 281.4, 172.7, 308.1), "thermik", "th"),
    ("s10", (84.7, 315.9, 174.3, 342.1), "thermik", "th"),
    ("s11", (76.5, 353.2, 182.5, 384.3), "zellen", "5-11"),
    ("s12", (93.4, 393.1, 165.6, 408.0), "tradeoff", "12"),
    ("s13", (93.4, 413.6, 165.6, 432.3), "balancing", "13"),
    ("sim", (91.3, 445.4, 167.7, 462.6), "emulation", "14"),
    ("fin", (93.4, 475.8, 165.6, 490.7), "start", "fin"),
]
DIAMONDS = ("s7", "s10", "s11")
COLORS = {"done": (46, 125, 50), "warning": (245, 166, 35), "error": (198, 40, 40), "skipped": (150, 150, 150),
          "open": (255, 255, 255)}
SYMBOL = {"done": "✓", "warning": "!", "error": "✕", "skipped": "-", "open": ""}
LEGEND = {"done": "gerechnet", "warning": "Warnung", "error": "fehlt / bestätigen", "skipped": "übersprungen",
          "open": "offen"}
ACTIVE = (243, 146, 0)


def _font(size: int):
    for n in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


@st.cache_data(show_spinner=False, max_entries=64)
def render(status_items: tuple, page: str) -> Image.Image:
    status = dict(status_items)
    im = Image.open(IMG).convert("RGB")
    dr = ImageDraw.Draw(im)
    f = _font(17)
    for name, (x0, y0, x1, y1), pg, sk in BOXES:
        X0, Y0, X1, Y1 = x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE
        if pg == page:
            if name in DIAMONDS:
                cxm, cym = (X0 + X1) / 2, (Y0 + Y1) / 2
                pts = [(cxm, Y0 - 7), (X1 + 9, cym), (cxm, Y1 + 7), (X0 - 9, cym)]
                dr.line(pts + [pts[0]], fill=ACTIVE, width=5, joint="curve")
            else:
                dr.rounded_rectangle((X0 - 5, Y0 - 5, X1 + 5, Y1 + 5), radius=12, outline=ACTIVE, width=5)
        s = status.get(sk, "open")
        col = COLORS.get(s, COLORS["open"])
        r = 13
        if name in DIAMONDS:                  # upper right edge of the diamond
            cx, cy = (X0 + X1) / 2 + 0.42 * (X1 - X0) / 2 + 12, Y0 + 0.29 * (Y1 - Y0) - 6
        else:
            cx, cy = min(X1 - 2, im.size[0] - r - 3), max(Y0 + 2, r + 3)
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col, outline=(40, 40, 40), width=2)
        sym = SYMBOL.get(s, "")
        if sym:
            tc = (255, 255, 255) if s != "warning" else (60, 40, 0)
            dr.text((cx, cy), sym, fill=tc, font=f, anchor="mm")
    return im


def legend_html() -> str:
    items = "".join(
        f"<span style='display:inline-block;margin-right:8px'><span style='display:inline-block;width:10px;"
        f"height:10px;border-radius:5px;border:1px solid #333;background:rgb{COLORS[k]};margin-right:3px'></span>"
        f"{v}</span>" for k, v in LEGEND.items())
    return f"<div class='small'>{items}<br><span style='color:rgb{ACTIVE}'>■</span> aktuelle Seite</div>"


def page_at(x: float, y: float, disp_w: float) -> str | None:
    im_w = Image.open(IMG).size[0]
    k = im_w / disp_w / SCALE            # display px -> PDF points
    px, py = x * k, y * k
    for name, (x0, y0, x1, y1), pg, sk in BOXES:
        if x0 - 3 <= px <= x1 + 3 and y0 - 3 <= py <= y1 + 3:
            return pg
    return None


def flowchart(status: dict, page: str, key: str, width: int | str = "stretch") -> str | None:
    """Show the flowchart; returns the page of a newly clicked box (or None)."""
    im = render(tuple(sorted(status.items())), page)
    if streamlit_image_coordinates is None:
        st.image(im, width=width if isinstance(width, int) else "stretch")
        st.error("Flussdiagramm nicht anklickbar: Paket streamlit-image-coordinates fehlt. GUI beenden und "
                 "./run_gui.sh neu starten (installiert fehlende Pakete) oder ./setup_env.sh ausführen.",
                 icon=":material/warning:")
        return None
    val = streamlit_image_coordinates(im, key=key, width=width, cursor="pointer")
    if not val:
        return None
    stamp = (val.get("x"), val.get("y"), val.get("unix_time"))
    if st.session_state.get(f"{key}_last") == stamp:
        return None
    st.session_state[f"{key}_last"] = stamp
    disp_w = val.get("width") or im.size[0]
    return page_at(val["x"], val["y"], disp_w)
