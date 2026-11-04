"""Regression tests: thesis / IECON 2023 numbers and core formulas.

Tests that need the private robot-arm profile are skipped when it is not on this machine.
"""
import math

import numpy as np
import pytest

from sc_module_design import equations as eq
from sc_module_design.config import DesignInputs, load_preset
from sc_module_design.database import get_database
from sc_module_design.emulation import EmulationConfig, HESSEmulator, ModuleModel
from sc_module_design.energy import energy_excursions
from sc_module_design.paths import PRIVATE_PROFILE_DIR
from sc_module_design.pipeline import run_design
from sc_module_design.splits import apply_split
from sc_module_design.thermal import first_order_response, temperature_rise

PRIVATE = (PRIVATE_PROFILE_DIR / "robot_arm_profile.npz").exists()


# ---------------------------------------------------------------------- formulas
def test_lifetime_voltage_and_series():
    u = eq.u_cell_max_lifetime(40000, 65, 3.0, 65, 2000, 0.2, 10)
    assert u == pytest.approx(2.1356, abs=1e-4)          # thesis: approx. 2.13 V
    assert eq.n_series(48, u) == 23
    assert 23 * 3.0 == 69.0


def test_intermediate_voltage_and_capacitance():
    assert eq.u_c0inm(1214.1, 557.0, 48, 33) == pytest.approx(43.84, abs=0.01)
    assert eq.u_c0inm(1214.1, 990.0, 48, 33) == pytest.approx(41.93, abs=0.01)
    assert eq.c_required(1214.1 + 557.0, 48, 33, 0.7) == pytest.approx(4.165, abs=1e-3)
    assert eq.c_required(1214.1 + 990.0, 48, 33, 0.7) == pytest.approx(5.183, abs=1e-3)
    assert eq.energy_ratio(0.7) == pytest.approx(0.51)


def test_ceq_energy():
    assert eq.c_eq_energy(40.0, 0.0, 1.0, 2.0) == 40.0
    # energy identity: C_eq (u2^2-u1^2)/2 == int (C0 + k u) u du
    C0, k, u1, u2 = 40.38, 5.28, 33 / 23, 48 / 23
    ceq = eq.c_eq_energy(C0, k, u1, u2)
    W = eq.cell_energy_cu(C0, k, u2) - eq.cell_energy_cu(C0, k, u1)
    assert ceq * (u2 ** 2 - u1 ** 2) / 2 == pytest.approx(W)


def test_parallel_counts():
    assert eq.p_capacitance(23, 4.16, 12) == 8
    assert eq.p_current(77.7, 17) == 5
    assert eq.p_capacitance(23, 5.18, 30) == 4
    assert eq.p_thermal(9.99, 8.68) == 2


def test_thermal_periodic_steady_state():
    rng = np.random.default_rng(0)
    P = rng.random(5000)
    dt, R, C = 0.05, 22.1, 19.7
    dT, dT0 = temperature_rise(P, dt, R, C, "periodic")
    # brute force: repeat the profile many times
    x = 0.0
    for _ in range(60):
        x = first_order_response(P, dt, R, C, x)[-1]
    assert dT[-1] == pytest.approx(x, rel=1e-6)
    assert dT0 == pytest.approx(x, rel=1e-6)


# ---------------------------------------------------------------------- energy algorithm
def test_legacy_zero_crossing_matches_reference_loop():
    rng = np.random.default_rng(1)
    p = np.repeat(rng.normal(0, 100, 400), 25)
    dt = 1e-3
    ex = energy_excursions(p, dt, "zero_crossing_legacy")
    f_above = p * (p >= 0)
    pos = np.flatnonzero(np.diff((f_above > 0).astype(int))) + 1
    ints = []
    for i in range(len(pos) - 1):
        a = 0 if i == 0 else pos[i]
        seg = p[a:pos[i + 1]]
        ints.append((seg.sum() - 0.5 * (seg[0] + seg[-1])) * dt)
    assert ex.E_dch_J == pytest.approx(max(ints))
    assert ex.E_ch_J == pytest.approx(-min(ints))


def test_splits_sum_to_load():
    t = np.arange(0, 60, 0.01)
    p = 300 + 200 * np.sin(2 * np.pi * 0.3 * t) + 50 * np.sign(np.sin(2 * np.pi * 2 * t))
    for name, prm in [("moving_average", {"window_s": 5, "alignment": a}) for a in
                      ("legacy_matlab", "causal", "centered", "periodic_causal")] + [
            ("peak_shaver", {"p_upper_W": 400}), ("lowpass_1st", {}), ("butterworth", {}),
            ("global_mean", {}), ("rate_limiter", {"max_slope_W_per_s": 1000})]:
        pb, ps = apply_split(name, t, p, prm)
        assert np.allclose(pb + ps, p), name


# ---------------------------------------------------------------------- full procedure
def _legacy(no_profile: bool):
    inp = load_preset("iecon2023_legacy")
    if no_profile:
        inp.profile.path = ""
    return run_design(inp, db=get_database())


@pytest.mark.parametrize("no_profile", [True] + ([False] if PRIVATE else []))
def test_iecon2023_reproduction(no_profile):
    r = _legacy(no_profile)
    assert r.load["E_dch_J"] == pytest.approx(1214.1, abs=0.1)
    assert r.load["E_ch_J"] == pytest.approx(557.0, abs=0.1)
    assert r.series_info["VISHAY_235HVR_3V0_65C"]["S"] == 23
    assert r.scenarios["profile"].C_req == pytest.approx(4.165, abs=2e-3)
    assert r.scenarios["profile_braking"].C_req == pytest.approx(5.183, abs=2e-3)
    m2 = r.candidates["profile_braking"].set_index("cell_id")
    sel = m2.loc["VISHAY_HVR235_30F"]
    assert sel.setup == "4/23"
    assert sel.ESR_module_mOhm == pytest.approx(149.5)
    assert sel.mass_kg == pytest.approx(0.92)
    assert sel.volume_dm3 == pytest.approx(0.1018, abs=1e-3)       # legacy formula
    assert sel.density_kg_dm3 == pytest.approx(9.04, abs=0.01)
    m1 = r.candidates["profile"].set_index("cell_id")
    assert m1.loc["VISHAY_HVR235_12F"].setup == "8/23"
    assert r.picks["profile"]["max_density"]["cell_id"] == "VISHAY_HVR235_12F"
    assert r.picks["profile_braking"]["min_cost"]["cell_id"] == "VISHAY_HVR235_40F"


def test_corrected_density_is_physical():
    inp = load_preset("iecon2023_legacy")
    inp.profile.path = ""
    inp.opt.geometry = "corrected"
    r = run_design(inp)
    d = r.candidates["profile"].density_kg_dm3
    assert d.between(1.0, 2.5).all()


@pytest.mark.skipif(not PRIVATE, reason="private robot-arm profile not available")
def test_characterized_with_thermal():
    r = run_design(load_preset("thesis_characterized"))
    df = r.candidates["profile_braking"].set_index("cell_id")
    row = df.loc["VISHAY_HVR235_50F"]
    assert row.thermal_status == "ok"
    assert 0 < row.dT_max_K < 40
    assert "C_eq,E" in row.C_source


def test_emulation_energy_balance():
    t = np.arange(0, 100, 0.001)
    p = 400 + 800 * (np.sin(2 * np.pi * 0.1 * t) > 0.8)
    pb = np.full_like(p, p.mean())
    m = ModuleModel(S=23, P=4, C_cell_F=30, R_cell_Ohm=0.026, U_min=33, U_max=48, U_init=42, I_max_A=77.7,
                    R_th=22.1, C_th=19.7)
    em = HESSEmulator(t, p, pb, m, EmulationConfig(soc_control=True))
    hist, tr = em.run_until_steady(max_cycles=100)
    last = hist[-1]
    assert abs(last.U_end - last.U_start) < 5e-3
    assert last.dT_max > 0
    # battery + SC must cover the load in every bin
    assert np.allclose(tr.P_bat + tr.P_sc - tr.P_load, 0, atol=1e-6)


# ---------------------------------------------------------------------- converter, current, thermal chain
def test_converter_efficiency():
    from sc_module_design.converter import bus_to_cell
    p = np.array([100.0, -100.0, 0.0])
    out = bus_to_cell(p, 0.9, 0.8)
    assert out == pytest.approx([100 / 0.9, -80.0, 0.0])
    assert bus_to_cell(p, 1.0, 1.0) == pytest.approx(p)


def test_current_power_balance_with_esr_drop():
    from sc_module_design.current import current_scalar
    U, R, P = 40.0, 0.2, 1000.0
    i, lim = current_scalar(P, U, R)
    assert not lim
    assert (U - R * i) * i == pytest.approx(P)               # P = U_term I_SC
    assert i > P / U                                          # ESR drop increases the current
    i_c, _ = current_scalar(-P, U, R)
    assert (U - R * i_c) * i_c == pytest.approx(-P)
    _, lim = current_scalar(U * U / (4 * R) * 1.01, U, R)    # beyond the power limit U^2/(4R)
    assert lim


def test_cell_loss_lowpass_squares_the_current():
    from sc_module_design.thermal import cell_loss_lowpass
    dt = 0.01
    i = np.where(np.arange(100000) % 200 < 100, 3.0, -3.0)   # +-3 A square wave, RMS 3 A
    ch = cell_loss_lowpass(i, dt, 0.024, 22.1, 19.7)
    assert ch["I_rms"] == pytest.approx(3.0)
    assert ch["P_v_mean"] == pytest.approx(9 * 0.024)
    assert ch["dT_ss"] == pytest.approx(9 * 0.024 * 22.1)    # 4.77 K
    assert ch["dT_max"] == pytest.approx(ch["dT_ss"], rel=0.02)


@pytest.mark.skipif(not PRIVATE, reason="private robot-arm profile not available")
def test_emulation_design_and_thermal_chain():
    from sc_module_design.emudesign import emulation_design
    r = run_design(load_preset("thesis_characterized"))
    d = emulation_design(r, "profile", "VISHAY_HVR235_50F", check_eol=False)
    assert d.feasible and d.P_final >= d.P_analytic
    assert d.chain is not None and d.chain["dT_max"] <= r.inputs.req.dT_max_K
    assert np.all(d.chain["p_v"] >= 0)


# ---------------------------------------------------------------------- PCB layout
def test_layout_reference_comparison():
    """EDLC layout comparison N = 60 x 1, d = 24 mm, gap 0.5 mm: 358.2 cm^2 (square) vs 334.0 cm^2 (hex)."""
    from sc_module_design.layout import pcb_layout
    sq = pcb_layout(60, 1, 24, 40, "square", 0.5, 5, bands=10, connectors=())
    hx = pcb_layout(60, 1, 24, 40, "hex", 0.5, 5, bands=10, connectors=())
    assert sq.cell_area_mm2 / 100 == pytest.approx(358.2, abs=0.1)
    assert hx.cell_area_mm2 / 100 == pytest.approx(334.0, abs=0.1)
    assert 100 * (hx.cell_area_mm2 / sq.cell_area_mm2 - 1) == pytest.approx(-6.8, abs=0.05)


@pytest.mark.parametrize("orientation", ["auto", "long", "wide"])
@pytest.mark.parametrize("packing", ["hex", "square"])
def test_layout_u_path_and_connector(orientation, packing):
    from sc_module_design.layout import pcb_layout
    lay = pcb_layout(23, 5, 18, 35, packing, 0.5, 5, orientation=orientation)
    groups = {c[2] for c in lay.cells}
    assert groups == set(range(23)) and len(lay.cells) == 115
    # + and - next to each other at the front edge, inside the connector
    con = lay.connectors[0]
    for x, y in (lay.terminal_plus, lay.terminal_minus):
        assert con["x"] <= x <= con["x"] + con["w"] and y <= con["h"]
    # cells do not overlap each other or the connectors
    pts = np.array([(c[0], c[1]) for c in lay.cells])
    dist = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)) + np.eye(len(pts)) * 1e9
    assert dist.min() >= 18 - 1e-6
    for x, y in pts:
        cx = min(max(x, con["x"]), con["x"] + con["w"])
        cy = min(max(y, con["y"]), con["y"] + con["h"])
        assert math.hypot(x - cx, y - cy) >= 9 - 1e-6
    assert 0 < pts[:, 0].min() - 9 and pts[:, 0].max() + 9 <= lay.width_mm
    assert pts[:, 1].max() + 9 <= lay.depth_mm
    if orientation == "long":
        assert lay.depth_mm >= lay.width_mm
    if orientation == "wide":
        assert lay.width_mm >= lay.depth_mm


def test_specification_limits_grey_out():
    inp = load_preset("legged_robot_48V")
    inp.req.max_volume_dm3 = 0.5
    r = run_design(inp)
    df = r.candidates["profile"]
    bad = df[df.volume_dm3 > 0.5]
    assert len(bad) and not bad.feasible.any()
    assert bad.status.str.startswith("Vorgabe verletzt").all()
    assert df[df.volume_dm3 <= 0.5].feasible.all()
