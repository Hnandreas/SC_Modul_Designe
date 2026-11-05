from sc_module_design.tuning import tuning_candidates
from sc_module_design.emulation import EmulationConfig, VoltagePI
import pytest


def test_search_bounded_and_filter_independent():
    variants=list(tuning_candidates(5,20,'both'))
    assert variants[0] == (20,5)
    assert all(5<=limit<=10 for _,limit in variants)
    assert set(w for w,_ in variants)=={20,15,10,5}
    assert list(tuning_candidates(5,20,'off'))==[(20,5)]
    assert list(tuning_candidates(9.5,None,'charge'))==[(None,9.5),(None,10)]


def test_custom_clock_and_bidirectional_pi_do_not_change_gain_units():
    for fsw, divider in [(100000,5),(80000,8)]:
        cfg=EmulationConfig(switching_frequency_Hz=fsw,pi_every_pwm_cycles=divider,
                            pi_kp=.3, pi_ki=.4,pi_min_A=-5,pi_max_A=10)
        for error in [-1,1]:
            pi=VoltagePI(cfg)
            pi.advance(error,.1)
            assert pi.integral==pytest.approx(error*.04)
            assert pi.calls==fsw/divider*.1
            assert error*pi.output>0
