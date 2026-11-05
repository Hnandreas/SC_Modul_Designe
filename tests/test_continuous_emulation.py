"""Regression checks for the user-specified SC current, PI clock and energy balance."""
import copy
import numpy as np
import pytest

from sc_module_design.emulation import EmulationConfig, HESSEmulator, ModuleModel, VoltagePI
from sc_module_design.splits import apply_split, periodic_emulation_split


def module():
    return ModuleModel(23, 3, 50, .022, 33, 48, 42, 77.7)


def emulator(power, battery=0, seconds=2, dt=.001, **kw):
    t = np.arange(0, seconds, dt)
    return HESSEmulator(t, np.full_like(t, power), np.full_like(t, battery), module(), EmulationConfig(**kw))


def firmware_pi(errors, cfg, integral=0):
    """Literal recurrence in Interleaved 6, at the explicitly requested new Ts."""
    outputs = []
    for error in errors:
        y = cfg.pi_kp * error + integral
        if (cfg.pi_min_A < y < cfg.pi_max_A or
            y >= cfg.pi_max_A and error < 0 or y <= cfg.pi_min_A and error > 0):
            integral += cfg.pi_ki * error * cfg.pi_period_s
        outputs.append(np.clip(y, cfg.pi_min_A, cfg.pi_max_A))
    return integral, outputs


@pytest.mark.parametrize('initial,error,n', [(0, 1, 2000), (4.9, 10, 2000), (-4.9, -10, 2000),
                                           (-8, 2, 2000), (8, -2, 2000), (8, 1, 2000), (0, 0, 20)])
def test_aggregated_pi_matches_firmware(initial, error, n):
    cfg = EmulationConfig()
    pi = VoltagePI(cfg)
    pi.integral = initial
    mean = pi.advance(error, n * cfg.pi_period_s)
    expected_integral, outputs = firmware_pi([error] * n, cfg, initial)
    assert pi.integral == pytest.approx(expected_integral, abs=1e-10)
    assert mean == pytest.approx(np.mean(outputs), abs=1e-10)
    assert pi.output == pytest.approx(outputs[-1], abs=1e-10)
    assert pi.calls == n


def test_requested_20khz_clock_and_gain_units():
    cfg = EmulationConfig()
    assert cfg.switching_frequency_Hz == 100000
    assert cfg.pi_every_pwm_cycles == 5
    assert cfg.pi_period_s == pytest.approx(50e-6)
    pi = VoltagePI(cfg)
    assert pi.advance(1, cfg.pi_period_s) == pytest.approx(.2)
    assert pi.integral == pytest.approx(.5 * 50e-6)
    for divider in (5, 10):
        pi = VoltagePI(EmulationConfig(pi_every_pwm_cycles=divider))
        pi.advance(1, 1)
        assert pi.integral == pytest.approx(.5)
        assert pi.calls == 100000 // divider


def test_clock_fractional_remainder_survives_chunks():
    pi = VoltagePI(EmulationConfig())
    pi.advance(1, 20e-6)
    assert pi.calls == 0
    pi.advance(1, 30e-6)
    assert pi.calls == 1
    assert pi.integral == pytest.approx(25e-6)


@pytest.mark.parametrize('power,battery,sign', [(0, 100, 1), (200, 0, -1)])
def test_charge_positive_esr_voltage_and_storage_balance(power, battery, sign):
    em = emulator(power, battery, soc_control=False)
    before = em.E
    summary, tr = em.run_cycle()
    assert np.all(sign * tr.I_sc > 0)
    assert np.all(tr.I_sc_rms >= 0)
    assert np.allclose(tr.U_term, tr.U_sc + em.R_mod * tr.I_sc)
    assert sign * (em.E - before) > 0
    assert before - em.E == pytest.approx(np.sum(tr.P_sc_cell * tr.dt) + summary.E_loss_J, abs=1e-8)
    assert np.allclose(tr.P_bat + tr.P_sc, tr.P_load)
    assert np.allclose(tr.P_sc_cell, -tr.U_term * tr.I_sc)


def test_no_energy_jump_when_starting_above_reference():
    em = emulator(0, initial_voltage_V=47, voltage_reference_V=42)
    before = em.E
    tr = em._advance(1)
    assert em.voltage(em.E) > 46.99
    assert em.E < before and em.E > em.energy(42)
    assert -5 <= tr.I_pi[0] <= 5


def test_at_upper_boundary_no_phantom_charging():
    em = emulator(0, 100, initial_voltage_V=48, soc_control=False)
    _, tr = em.run_cycle()
    assert np.allclose(tr.I_sc, 0)
    assert np.allclose(tr.P_sc, 0)
    assert np.allclose(tr.P_bat, 0)
    assert np.allclose(tr.E_sc, em.energy(48))


def test_continuation_matches_one_long_run():
    a = emulator(0, 0, seconds=.123, voltage_reference_V=44)
    b = copy.deepcopy(a)
    whole = a._advance(3 * a.n)
    pieces = [b._advance(a.n), b._advance(3), b._advance(2 * a.n - 3)]
    assert np.allclose(whole.I_sc, np.concatenate([p.I_sc for p in pieces]))
    assert np.allclose(whole.E_sc, np.concatenate([p.E_sc for p in pieces]))
    assert a.pi.integral == pytest.approx(b.pi.integral)
    assert a.pi.calls == b.pi.calls == round(3 * .123 * 20000)
    assert a.elapsed == pytest.approx(3 * .123)
    assert a.cycle == b.cycle == 3


def test_pi_response_independent_of_display_bin():
    a = emulator(0, seconds=2, voltage_reference_V=44, dt_bin_s=.01)
    b = emulator(0, seconds=2, voltage_reference_V=44, dt_bin_s=.005)
    a.run_cycle()
    b.run_cycle()
    assert a.pi.calls == b.pi.calls == 40000
    assert a.voltage(a.E) == pytest.approx(b.voltage(b.E), abs=.005)
    assert a.pi.integral == pytest.approx(b.pi.integral, abs=.005)


@pytest.mark.parametrize('name,params', [('moving_average', {'window_s':2, 'alignment':'legacy_matlab'}),
                                      ('lowpass_1st', {'fc_Hz':.5}),
                                      ('butterworth', {'fc_Hz':.5, 'order':2}),
                                      ('rate_limiter', {'max_slope_W_per_s':10})])
def test_constant_profile_has_no_periodic_filter_energy_drain(name, params):
    t = np.arange(0, 10, .01)
    p = np.full_like(t, 100.)
    before = dict(params)
    pb, ps = periodic_emulation_split(name, t, p, params)
    assert np.allclose(pb, p, atol=1e-8)
    assert abs(ps.sum() * .01) < 1e-6
    assert params == before
    if name == 'moving_average':
        _, legacy = apply_split(name, t, p, params)
        assert legacy.sum() * .01 == pytest.approx(50)  # published reference path is retained


def test_eol_esr_and_module_design_are_not_changed():
    m = module()
    original = copy.deepcopy(m)
    t = np.arange(0, 1, .01)
    em = HESSEmulator(t, t*0, t*0, m, EmulationConfig(eol=True))
    assert em.R_mod == pytest.approx(23 * .044 / 3)
    em.run_cycle()
    assert m == original


def test_impossible_profile_power_is_reported_not_resized():
    em = emulator(10000, seconds=.1, soc_control=False, eol=True)
    m = copy.deepcopy(em.m)
    summary, tr = em.run_cycle()
    assert np.any(tr.power_limited)
    assert summary.t_power_limit_s > 0
    assert np.all(tr.P_sc_cell <= tr.P_available + 1e-7)
    assert np.allclose(tr.P_bat + tr.P_sc, tr.P_load)
    assert em.m == m


def test_mixed_power_bin_rms_uses_actual_currents():
    t = np.arange(0, .1, .001)
    power = np.tile([-200., 300.], 50)
    em = HESSEmulator(t, power, np.zeros_like(t), module(), EmulationConfig(soc_control=False, dt_bin_s=.01))
    _, tr = em.run_cycle()
    full = tr.I_sc_full.reshape(-1, 10)
    assert np.allclose(tr.I_sc_rms**2, np.mean(full**2, axis=1))
    assert np.allclose(tr.I_sc, np.mean(full, axis=1))


def test_buck_regeneration_cannot_raise_terminal_above_bus():
    em = emulator(-2000, seconds=2, initial_voltage_V=46, soc_control=False)
    before = em.E
    summary, tr = em.run_cycle()
    ceiling = .98 * 48
    assert np.any(tr.voltage_limited)
    assert np.all(tr.U_term_peak <= ceiling + 1e-9)
    assert np.all(tr.U_term <= ceiling + 1e-9)
    assert em.voltage(em.E) <= ceiling + 1e-9
    assert summary.t_bus_voltage_limit_s > 0
    assert summary.t_converter_range_s == 0
    assert np.allclose(tr.P_bat + tr.P_sc, tr.P_load)
    assert before - em.E == pytest.approx(np.sum(tr.P_sc_cell * tr.dt) + summary.E_loss_J, abs=1e-8)
    assert em.m.U_max == 48  # design unchanged; operating limit belongs to converter


def test_buck_voltage_limit_applies_to_raw_peaks_not_only_mean():
    t = np.arange(0, .1, .001)
    p = np.tile([-10000., 100.], 50)
    em = HESSEmulator(t, p, p*0, module(),
                      EmulationConfig(soc_control=False, initial_voltage_V=46))
    _, tr = em.run_cycle()
    currents = tr.I_sc_full.reshape(-1, 10)
    terminals = tr.U_sc[:, None] + em.R_mod * currents
    assert np.all(terminals <= .98 * 48 + 1e-9)
    assert np.allclose(terminals.max(axis=1), tr.U_term_peak)


def test_unreachable_charge_reference_does_not_change_module_or_create_voltage():
    em = emulator(0, seconds=20, initial_voltage_V=46.9, voltage_reference_V=47.5)
    _, tr = em.run_cycle()
    assert np.all(tr.U_term_peak <= .98 * 48 + 1e-9)
    assert em.voltage(em.E) <= .98 * 48 + 1e-9
    assert em.U_reference == 47.5 and em.m.U_max == 48
    assert np.any(tr.voltage_limited)


def test_already_charged_above_bus_is_flagged_not_artificially_dumped():
    em = emulator(0, seconds=1, initial_voltage_V=47, voltage_reference_V=47, U_bus_V=40)
    before = em.E
    summary, tr = em.run_cycle()
    assert np.allclose(tr.I_sc, 0)
    assert em.E == pytest.approx(before)
    assert np.all(tr.converter_range_violation)
    assert summary.t_converter_range_s == pytest.approx(1)


def test_ideal_zero_esr_cap_also_cannot_charge_past_converter_ceiling():
    m = module()
    m.R_cell_Ohm = 0
    t = np.arange(0, 1, .001)
    em = HESSEmulator(t, t*0-2000, t*0, m, EmulationConfig(soc_control=False, U_bus_V=44))
    _, tr = em.run_cycle()
    assert em.voltage(em.E) <= .98 * 44 + 1e-9
    assert np.all(tr.U_term_peak <= .98 * 44 + 1e-9)
