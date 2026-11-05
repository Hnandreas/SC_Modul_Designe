"""v6: exact figure width, legends below, free label positions, data criterion off, report, preset GUI state."""
import math

import pytest
from pypdf import PdfReader

from sc_module_design import plotting as pl
from sc_module_design import tradeoff as T
from sc_module_design.config import DesignInputs, load_preset
from sc_module_design.database import get_database
from sc_module_design.pipeline import run_design
from sc_module_design.report_pdf import build_report, report_candidates


@pytest.fixture(scope="module")
def res():
    return run_design(load_preset("legged_robot_48V"))


def _width_cm(pdf):
    return float(PdfReader(str(pdf)).pages[0].mediabox.width) / 72 * 2.54


@pytest.mark.parametrize("name", ["peaks", "lifetime", "excursions", "tradeoff"])
def test_thesis_figures_exactly_8cm(res, tmp_path, name):
    db = get_database()
    fig = {"peaks": lambda: pl.fig_peaks(res), "lifetime": lambda: pl.fig_lifetime(res, db),
           "excursions": lambda: pl.fig_energy_excursions(res),
           "tradeoff": lambda: pl.fig_tradeoff(res, scenarios=tuple(res.candidates))}[name]()
    out = pl.save_fig(fig, tmp_path / name, formats=("pdf",))[0]
    assert math.isclose(_width_cm(out), 8.0, abs_tol=0.01)


def test_excursions_one_legend_below_sc_power_once(res):
    fig = pl.fig_energy_excursions(res)
    assert all(ax.get_legend() is None for ax in fig.axes)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels.count("SC power") == 1 and len(labels) == 3


def test_value_labels_do_not_cover_curves(res):
    db = get_database()
    for fig, ax in ((pl.fig_lifetime(res, db), 0), (pl.fig_peaks(res), 1)):
        a = fig.axes[ax]
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for t in a.texts:
            if t.get_text().endswith((" V", " W")):
                assert pl._hits(a, pl._text_box(t, r), r, skip=(t,)) == 0, t.get_text()


def test_peak_values_in_histogram_not_above_box(res):
    fig = pl.fig_peaks(res)
    assert not fig.axes[0].texts
    assert len([t for t in fig.axes[1].texts if t.get_text().endswith(" W")]) == 2


def test_lifetime_legend_as_in_thesis(res):
    fig = pl.fig_lifetime(res, get_database())
    txt = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "EDLC-HVR" in txt and "T_0" in txt and "U_\\mathrm{C0N1}" in txt and "\\Delta U_\\mathrm{tC01}" in txt


def test_new_tradeoff_quantities(res):
    df = next(iter(res.candidates.values()))
    assert df.module_height_mm.notna().all()
    pl.fig_tradeoff(res, scenarios=tuple(res.candidates), y_bottom="module_height_mm", size_bottom="n_cells")


def test_measured_data_gives_no_advantage(res):
    assert "data" in T.CRITERIA and "data" not in T.ACTIVE_CRITERIA
    for _, (_, w) in T.WEIGHT_PRESETS.items():
        assert not w or "data" not in w
    df = next(iter(res.candidates.values()))
    df = df[df.feasible].copy()
    df["data_quality"] = 0.0
    s0, _ = T.score(df, {k: 1.0 for k in T.CRITERIA})
    df.iloc[0, df.columns.get_loc("data_quality")] = 1.0
    s1, _ = T.score(df, {k: 1.0 for k in T.CRITERIA})          # even a custom weight on "data" is ignored
    assert (s0 - s1).abs().max() < 1e-12


def test_report_candidates_at_most_three_selected_first(res):
    for sk, df in res.candidates.items():
        c = report_candidates(res, sk, 3)
        assert len(c) <= 3
        sel = (res.inputs.selected or {}).get(sk, {}).get("cell_id")
        if sel:
            assert c[0][0] == "Selected" and c[0][1].cell_id == sel


def test_report_pdf_without_latex(res, tmp_path):
    pdf, engine = build_report(res, tmp_path, engine="matplotlib")
    assert engine == "matplotlib" and pdf.exists() and len(PdfReader(str(pdf)).pages) > 3
    tex = (tmp_path / "report_src" / pdf.with_suffix(".tex").name).read_text(encoding="utf8")
    assert "Module comparison" in tex and "Origin" not in tex


def test_preset_keeps_gui_state_out_of_design_key():
    d = load_preset("legged_robot_48V")
    d.ui_state = {"emu_pi_kp": 0.3}
    d2 = DesignInputs.from_dict(d.to_dict())
    assert d2.ui_state == {"emu_pi_kp": 0.3}
    assert "ui_state" not in d.design_dict() and "name" not in d.design_dict()
