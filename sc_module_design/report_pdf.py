"""Design report as one PDF (English).

Contents: inputs, flowchart steps, at most three candidates per module (the selected cell first, then the best
rated ones), the two modules side by side, the module layout (PCB), all figures with captions and tool messages.

LaTeX (latexmk or pdflatex) gives the thesis look: Times, vector figures exactly 8 / 16 cm wide. The .tex source
is kept in report_src/. Without a LaTeX installation a simpler PDF is written with matplotlib.
"""
from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import plotting as pl
from .database import Database, get_database
from .report import save_standard_figures

MODULES = {"profile": "Module 1", "profile_braking": "Module 2"}
SCENARIO_TEXT = {"profile": "load profile (W = E_DCH,max + E_CH,max)",
                 "profile_braking": "load profile with braking energy (W = E_DCH,max + E_braking)"}
SPLITS = {"moving_average": "moving average"}
TEX_DIRS = ["/Library/TeX/texbin", "/usr/texbin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]


@dataclass
class ExtraFigure:
    """Figure from the GUI (e.g. emulation) with caption and optional key results."""
    name: str
    caption: str
    fig: object
    rows: list = field(default_factory=list)          # (label, value) pairs shown as a table


# ---------------------------------------------------------------------- helpers
def _tex(s) -> str:
    """Escape plain text for LaTeX."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "²": r"\textsuperscript{2}",
           "³": r"\textsuperscript{3}", "Ω": r"$\Omega$", "Δ": r"$\Delta$", "η": r"$\eta$", "°": r"\textdegree{}",
           "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$", "–": "-", "—": "-", "µ": r"$\mu$", "×": "x",
           "·": " ", "±": r"$\pm$", "√": r"$\surd$"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def _f(v, d: int = 1, unit: str = "") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v) if v not in (None, "") else "-"
    if not math.isfinite(x):
        return "-"
    s = f"{x:,.{d}f}".replace(",", " ")
    return f"{s} {unit}".strip()


def cell_label(row, db: Database) -> str:
    try:
        c = db.cell(row["cell_id"])
        return f"{c['manufacturer']} {c['series']} {float(c['c_nom_F']):g} F ({c['diameter_mm']:g} x {c['length_mm']:g} mm)"
    except Exception:
        return str(row["cell_id"])


def report_candidates(res, scenario: str, n: int = 3) -> list:
    """At most n candidates: the selected cell first, then the best rated feasible ones. [(role, row)]"""
    df = res.candidates[scenario]
    sel = (res.inputs.selected or {}).get(scenario, {}).get("cell_id")
    out = []
    if sel and sel in set(df.cell_id):
        out.append(("Selected", df[df.cell_id == sel].iloc[0]))
    feas = df[df.feasible].sort_values("score", ascending=False)
    for _, r in feas.iterrows():
        if len(out) >= n:
            break
        if r.cell_id != sel:
            rk = r.get("rank")
            out.append((f"Rank {int(rk)}" if rk == rk and rk is not None else "Candidate", r))
    return out


def module_choice(res, scenario: str):
    """Module shown in the comparison: the selected cell, else the best rated one. (role, row) or None."""
    c = report_candidates(res, scenario, 1)
    return c[0] if c else None


CAND_ROWS = [  # (label, column, digits, unit)
    ("Setup P/S", "setup", None, ""), ("Number of cells", "n_cells", 0, ""),
    ("Module capacitance", "C_module_F", 2, "F"), ("Capacitance reserve (EOL)", "C_reserve", None, "%"),
    ("Module ESR", "ESR_module_mOhm", 0, "mΩ"), ("Module cost", "cost_EUR", 0, "Euro"),
    ("Module mass", "mass_kg", 2, "kg"), ("Module volume", "volume_dm3", 2, "dm³"),
    ("Module density", "density_kg_dm3", 2, "kg/dm³"), ("PCB box W x D x H", "block_mm", None, "mm"),
    ("Mean ESR losses", "P_loss_avg_W", 1, "W"), ("ΔT block model", "dT_max_K", 1, "K"),
    ("ΔT network, hottest cell", "dT_net_max_K", 1, "K"), ("Peak current at U_C0min", "I_peak_Umin_A", 1, "A"),
    ("Power demand / limit (EOL)", "P_ratio", 2, ""), ("P from power criterion", "P_P", 0, ""),
    ("P determined by", "P_driver", None, ""), ("Score (weighted)", "score", 0, "%"), ("Status", "status", None, ""),
]


# German tool terms in the status / P driver columns, translated for the English report
EN_TERMS = [("unzulässig", "not feasible"), ("zulässig", "feasible"), ("Vorgabe verletzt", "specification violated"),
            ("Thermik, P-Grenze", "thermal, P limit"), ("Leistung U²/(4R)", "power U²/(4R)"),
            ("Thermik Netzwerk", "thermal network"), ("Thermik transient", "thermal transient"),
            ("Thermik RMS", "thermal RMS"), ("Strom", "current"), ("Kapazität", "capacitance"),
            ("P mindestens", "P minimum")]


def _en(s: str) -> str:
    for de, en in EN_TERMS:
        s = s.replace(de, en)
    return s


def _cell_value(row, col, d, unit):
    v = row.get(col, None)
    if col in ("status", "P_driver") and isinstance(v, str):
        return _en(v)
    if col == "I_peak_Umin_A" and (v is None or v != v):
        return "power limit exceeded"
    if col == "C_reserve":
        return _f(100 * float(v), 0, "%") if v == v and v is not None else "-"
    if d is None:
        s = "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) or v == "" else str(v)
        return f"{s} {unit}".strip() if s != "-" else s
    return _f(v, d, unit)


def _inputs(res, db) -> list:
    inp, req, o = res.inputs, res.inputs.req, res.inputs.opt
    prof = Path(inp.profile.path).stem if inp.profile.path else "manual load values"
    if "private" in str(inp.profile.path):
        prof = f"private measurement ({prof})"
    v = res.voltages or {}
    rows = [
        ("Load profile", prof + (f", scaled x {inp.profile.scale:g}" if inp.profile.scale != 1 else "")),
        ("Power split", f"{SPLITS.get(inp.split.name, inp.split.name)}"
         + (f", window {inp.split.params.get('window_s'):g} s" if inp.split.params.get("window_s") else "")),
        ("Energy method", o.energy_method),
        ("Braking energy", f"m = {inp.braking.mass_kg:g} kg, v = {inp.braking.v_mps:g} m/s" if inp.braking.enabled
         else "not considered"),
        ("Bus voltage U_Bus", _f(req.U_bus_V, 1, "V")),
        ("Max. module voltage U_C0max", _f(v.get("U_C0max"), 1, "V")),
        ("Min. module voltage U_C0min", f"{_f(v.get('U_C0min'), 2, 'V')} (mode {req.U_C0min_mode}, r_Q = {req.r_Q:g})"),
        ("Converter current limit I_C0max", _f(req.I_C0max_A, 1, "A")),
        ("Converter efficiency η (discharge / charge)", f"{req.eta_dch:g} / {req.eta_ch:g}"),
        ("Required lifetime", _f(req.t_life_h, 0, "h") + f" at T = {req.T_cell_C:g} °C"),
        ("End of life", f"C x {req.L_EOL:g}, ESR x 2"),
        ("Thermal limit", f"ΔT_max = {req.dT_max_K:g} K at T_amb = {req.T_amb_C:g} °C"),
        ("Capacitance / ESR source", f"{o.c_source} / {o.esr_source}"),
        ("Geometry", f"{o.geometry}, {o.packing} packing, cell gap {o.cell_spacing_mm:g} mm"),
        ("Thermal model", f"{o.thermal_mode}" + (", network model active" if o.thermal_network else "")),
        ("Trade-off weighting", f"{o.weight_preset} (measured data gives no advantage)"),
    ]
    load = res.load or {}
    if load:
        rows += [("Peak / mean load power", f"{_f(load.get('P_peak_W'), 0, 'W')} / {_f(load.get('P_avg_W'), 1, 'W')}"),
                 ("Max. energy excursions E_DCH,max / E_CH,max",
                  f"{_f(load.get('E_dch_J'), 1, 'J')} / {_f(load.get('E_ch_J'), 1, 'J')}")]
        if load.get("E_brake_J"):
            rows.append(("Braking energy", _f(load.get("E_brake_J"), 0, "J")))
    for si in (res.series_info or {}).values():
        rows.append((f"Series {si['series']}", f"U_C01max = {si['U_C01max']:.3f} V, S = {si['S']}, "
                                               f"U_C0N = {si['U_C0N']:.1f} V"))
    return rows


# ---------------------------------------------------------------------- figures
def _layout_figure(res, scenario: str, row, db: Database):
    if res.inputs.opt.geometry != "pcb":
        return None
    from .selection import layout_pair
    cell = db.cell(row.cell_id)
    S, P = int(row.S), int(row.P)
    sq, hx = layout_pair(cell, S, P, res.inputs.opt, res.inputs.req)
    if sq is None:
        return None
    return pl.fig_layout_compare(sq, hx, S * P * float(cell["mass_g"]) / 1000, res.inputs.opt.pcb_areal_mass_g_cm2)


def prepare_figures(res, out: Path, db: Database, extra: Optional[list] = None) -> tuple:
    """Save every figure the report needs into out/figures.

    Returns (module figures: layout, thermal check and emulation per module; design figures), each a list of
    (stem, caption, None, rows).
    """
    fdir = out / "figures"
    names = set(save_standard_figures(res, fdir, db))
    module_items, items = [], []

    def add(stem, cap, rows=None, module=False):
        if (fdir / f"{stem}.pdf").exists():
            (module_items if module else items).append((stem, cap, None, rows or []))

    add("power_profile", "Load profile and battery power after the power split.")
    add("power_peaks", "Local power peaks of the load profile. Box plot and histogram; dashed lines: upper fence "
                       "Q3 + 1.5 IQR and largest peak.")
    add("fft", "Amplitude spectrum of the load profile.")
    add("energy_excursions", "Maximum discharge (a) and charge (b) energy excursion of the supercapacitor power.")
    add("voltage_levels", "Operating voltage levels of the supercapacitor module.")
    add("dischargeratio", "Usable energy ratio over the discharge ratio r_Q.")
    add("lifetime_SC", "Maximum permissible cell voltage as a function of the required lifetime "
                       "(Arrhenius approximation).")
    add("Modules_Cells", "Module cost and module density over the cell capacitance for Module 1 and Module 2. "
                         "Colour: module ESR; ball size: density (top) and cost (bottom); circle: selected cell.")
    req = res.inputs.req
    for sk in res.candidates:                     # the two module layouts one after the other
        ch = module_choice(res, sk)
        if ch is None:
            continue
        role, row = ch
        stem = f"layout_{sk}"
        if stem not in names:
            fig = _layout_figure(res, sk, row, db)
            if fig is not None:
                pl.save_fig(fig, fdir / stem)
                pl.plt.close(fig)
        add(stem, f"{MODULES.get(sk, sk)}: PCB layout of the {role.lower()} cell {cell_label(row, db)}, setup "
                  f"P/S = {row.setup}; square and hexagonal packing (hexagonal is the reference).", module=True)
    for sk, df in res.candidates.items():         # then the thermal check of both modules
        if df.dT_max_K.notna().any():
            fig = pl.fig_dT_candidates(df, req.dT_max_K)
            pl.save_fig(fig, fdir / f"dT_candidates_{sk}")
            pl.plt.close(fig)
            add(f"dT_candidates_{sk}", f"{MODULES.get(sk, sk)}: temperature rise of the candidates (block model, "
                                       "precheck; not validated, FEM recommended).", module=True)
    for e in extra or []:
        try:
            pl.save_fig(e.fig, fdir / e.name)
            pl.plt.close(e.fig)
            add(e.name, e.caption, e.rows, module=True)
        except Exception:  # an extra GUI figure must never break the report
            continue
    return module_items, items


# ---------------------------------------------------------------------- LaTeX
def _table(header: list, rows: list, colspec: str) -> list:
    out = [r"\begin{center}\small", r"\begin{tabular}{" + colspec + "}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return out


def _figs_tex(items: list) -> list:
    L = []
    for stem, cap, _, rows in items:
        L += [r"\begin{figure}[H]", r"\centering", rf"\includegraphics{{../figures/{stem}.pdf}}",
              rf"\caption{{{_tex(cap)}}}", r"\end{figure}"]
        if rows:
            L += _table(["Quantity", "Value"], [[_tex(a), _tex(b)] for a, b in rows], "p{6.2cm}p{7cm}")
    return L


def build_tex(res, out: Path, figures: tuple, db: Database) -> str:
    module_items, items = figures
    inp = res.inputs
    L = [r"\documentclass[10pt,a4paper]{article}",
         r"\usepackage[margin=2.5cm]{geometry}",
         r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
         r"\IfFileExists{newtxtext.sty}{\usepackage{newtxtext,newtxmath}}{\usepackage{mathptmx}}",
         r"\usepackage{graphicx,booktabs,float,array,textcomp}",
         r"\usepackage[font=small,labelfont=bf]{caption}",
         r"\usepackage[hidelinks]{hyperref}",
         r"\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}",
         r"\begin{document}",
         r"{\Large\bfseries Supercapacitor Module Design Report}\par\medskip",
         rf"Preset: \texttt{{{_tex(inp.name)}}}\quad Date: {datetime.now():%Y-%m-%d %H:%M}\par",
         r"Generated with the SC Module Designer (sizing flowchart of the dissertation).\par"]
    if inp.description:
        L.append(r"\smallskip{\small " + _tex(" ".join(str(inp.description).split())) + r"}\par")

    L += [r"\section{Requirements and inputs}"]
    L += _table(["Quantity", "Value"], [[_tex(a), _tex(b)] for a, b in _inputs(res, db)], "p{6.2cm}p{9cm}")

    L += [r"\section{Flowchart steps}"]
    L += _table(["Step", "Title", "Status", "Result"],
                [[_tex(k), _tex(s.title), _tex(s.status), _tex(s.summary)] for k, s in res.steps.items()],
                "p{0.8cm}p{3.6cm}p{1.2cm}p{9.6cm}")

    if res.candidates:
        L += [r"\section{Candidates (at most three per module)}",
              "Selected cell first, then the best rated feasible candidates (weighting: "
              + _tex(inp.opt.weight_preset) + ")."]
        for sk in res.candidates:
            cands = report_candidates(res, sk, 3)
            if not cands:
                continue
            L.append(r"\subsection*{" + _tex(f"{MODULES.get(sk, sk)}: {SCENARIO_TEXT.get(sk, sk)}") + "}")
            head = ["Quantity"] + [_tex(f"{role}") for role, _ in cands]
            rows = [["Cell"] + [_tex(cell_label(r, db)) for _, r in cands]]
            rows += [[_tex(lab)] + [_tex(_cell_value(r, col, d, u)) for _, r in cands] for lab, col, d, u in CAND_ROWS]
            L += _table(head, rows, "p{4.4cm}" + "p{3.5cm}" * len(cands))

        L += [r"\section{Module comparison}"]
        chosen = {sk: module_choice(res, sk) for sk in res.candidates}
        chosen = {k: v for k, v in chosen.items() if v is not None}
        if chosen:
            head = ["Quantity"] + [_tex(f"{MODULES.get(k, k)} ({v[0].lower()})") for k, v in chosen.items()]
            rows = [["Design case"] + [_tex(SCENARIO_TEXT.get(k, k)) for k in chosen]]
            scen_rows = [("Used energy W", "W_used_J", 1, "J"), ("U_C0min", "U_min", 2, "V"),
                         ("U_C0inM", "U_inM", 2, "V"), ("U_C0max", "U_max", 2, "V"),
                         ("Required capacitance C_req", "C_req", 3, "F")]
            for lab, att, d, u in scen_rows:
                rows.append([_tex(lab)] + [_tex(_f(getattr(res.scenarios[k], att, None), d, u)) for k in chosen])
            rows.append(["Cell"] + [_tex(cell_label(v[1], db)) for v in chosen.values()])
            rows += [[_tex(lab)] + [_tex(_cell_value(v[1], col, d, u)) for v in chosen.values()]
                     for lab, col, d, u in CAND_ROWS]
            L += _table(head, rows, "p{4.4cm}" + "p{5.2cm}" * len(chosen))

    if module_items:
        L += [r"\section{Module layout, thermal check and emulation}"] + _figs_tex(module_items)
    L += [r"\section{Design figures}"] + _figs_tex(items)

    if res.warnings:
        L += [r"\section{Tool messages}", "As shown in the GUI (German).", r"\begin{itemize}"]
        L += [r"\item " + _tex(w) for w in res.warnings] + [r"\end{itemize}"]
    L.append(r"\end{document}")
    return "\n".join(L)


def find_latex() -> tuple[Optional[str], dict]:
    """latexmk or pdflatex, also in the usual TeX folders that a GUI process may not have on its PATH."""
    env = dict(os.environ)
    extra = [d for d in TEX_DIRS + sorted(glob.glob("/usr/local/texlive/*/bin/*")) if os.path.isdir(d)]
    env["PATH"] = os.pathsep.join([env.get("PATH", "")] + extra)
    for exe in ("latexmk", "pdflatex"):
        p = shutil.which(exe, path=env["PATH"])
        if p:
            return p, env
    return None, env


def _compile(tex_file: Path, exe: str, env: dict) -> tuple[bool, str]:
    cwd = tex_file.parent
    if Path(exe).name == "latexmk":
        cmds = [[exe, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_file.name]]
    else:
        cmds = [[exe, "-interaction=nonstopmode", "-halt-on-error", tex_file.name]] * 2
    log = ""
    for c in cmds:
        try:
            r = subprocess.run(c, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        log = (r.stdout or "")[-3000:]
        if r.returncode != 0:
            return False, log
    return tex_file.with_suffix(".pdf").exists(), log


# ---------------------------------------------------------------------- matplotlib fallback
def _mpl_report(res, out: Path, figures: tuple, db: Database, pdf_path: Path) -> None:
    items = list(figures[0]) + list(figures[1])
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    plt = pl.plt
    A4 = (8.27, 11.69)

    def text_page(pp, title, lines):
        for i in range(0, max(len(lines), 1), 60):
            fig = plt.figure(figsize=A4)
            fig.text(0.08, 0.95, title, fontsize=13, weight="bold", va="top")
            fig.text(0.08, 0.92, "\n".join(lines[i:i + 60]), fontsize=8, va="top", family="monospace")
            pp.savefig(fig)
            plt.close(fig)

    with PdfPages(pdf_path) as pp:
        inp = res.inputs
        lines = [f"Preset: {inp.name}", f"Date: {datetime.now():%Y-%m-%d %H:%M}", ""]
        lines += [f"{a:<44s} {b}" for a, b in _inputs(res, db)]
        lines += ["", "Flowchart steps:"] + [f"{k:>5s} [{s.status}] {s.title}: {s.summary}"[:120]
                                             for k, s in res.steps.items()]
        text_page(pp, "Supercapacitor Module Design Report", lines)
        for sk in res.candidates:
            c = report_candidates(res, sk, 3)
            if not c:
                continue
            ln = [f"{'Quantity':<30s}" + "".join(f"{role:<24s}" for role, _ in c),
                  f"{'Cell':<30s}" + "".join(f"{cell_label(r, db)[:23]:<24s}" for _, r in c)]
            ln += [f"{lab:<30s}" + "".join(f"{_cell_value(r, col, d, u)[:23]:<24s}" for _, r in c)
                   for lab, col, d, u in CAND_ROWS]
            text_page(pp, f"{MODULES.get(sk, sk)}: candidates", ln)
        for stem, cap, _, rows in items:
            png = out / "figures" / f"{stem}.png"
            if not png.exists():
                continue
            img = mpimg.imread(png)
            fig = plt.figure(figsize=A4)
            h, w = img.shape[:2]
            fw = 0.84 if w / h > 1.4 else 0.6
            fh = fw * A4[0] / A4[1] * h / w
            ax = fig.add_axes([(1 - fw) / 2, 0.9 - fh, fw, fh])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.08, 0.86 - fh, cap + ("\n" + "\n".join(f"{a}: {b}" for a, b in rows) if rows else ""),
                     fontsize=9, va="top", wrap=True)
            pp.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------- entry point
def build_report(res, out_dir: str | Path, db: Optional[Database] = None, extra: Optional[list] = None,
                 engine: str = "auto") -> tuple[Path, str]:
    """Write <name>_report.pdf into out_dir. engine: auto | latex | matplotlib. Returns (path, engine used)."""
    db = db or get_database()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = prepare_figures(res, out, db, extra)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in res.inputs.name) or "design"
    pdf_path = out / f"{safe}_report.pdf"
    src = out / "report_src"
    src.mkdir(exist_ok=True)
    tex_file = src / f"{safe}_report.tex"
    tex_file.write_text(build_tex(res, out, figures, db), encoding="utf8")
    if engine in ("auto", "latex"):
        exe, env = find_latex()
        if exe:
            ok, log = _compile(tex_file, exe, env)
            if ok:
                shutil.copyfile(tex_file.with_suffix(".pdf"), pdf_path)
                return pdf_path, "latex"
            (src / "latex_error.log").write_text(log, encoding="utf8")
        if engine == "latex":
            raise RuntimeError("LaTeX not available or failed (see report_src/latex_error.log)")
    _mpl_report(res, out, figures, db, pdf_path)
    return pdf_path, "matplotlib"
