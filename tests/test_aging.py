import pytest
from sc_module_design.aging import linear_factors, find_functional_lifetime
from sc_module_design.equations import lifetime_h

def test_linear_degradation_and_arrhenius_timebase():
    reference = lifetime_h(2.8, 55, 3, 65, 2000)
    assert reference == pytest.approx(8000)
    assert linear_factors(0, reference) == (1,1)
    assert linear_factors(.8*reference, reference) == pytest.approx((.76,1.8))
    assert linear_factors(reference, reference) == pytest.approx((.7,2))
    with pytest.raises(ValueError): linear_factors(reference+1,reference)

def test_search_shortens_lifetime_on_battery_takeover():
    lo, hi, result, bol = find_functional_lifetime(lambda h:(h<=800, h),1000)
    assert bol and lo <= 800 < hi
    assert hi-lo <= 1000/128
    assert result == lo

def test_bol_failure_is_no_verified_life():
    lo,hi,_,bol = find_functional_lifetime(lambda h:(False,h),1000)
    assert (lo,hi,bol)==(0,0,False)

def test_full_reference_life_if_all_pass():
    lo,hi,_,bol = find_functional_lifetime(lambda h:(True,h),1000)
    assert (lo,hi,bol)==(1000,1000,True)


def test_only_additional_discharge_is_battery_takeover():
    import numpy as np
    from sc_module_design.emulation import ModuleModel, EmulationConfig, HESSEmulator
    m = ModuleModel(23,3,50,.022,33,48,42,1)
    t = np.arange(0,1,.001)
    cfg = EmulationConfig(soc_control=False)
    normal = HESSEmulator(t,t*0+20,t*0+10,m,cfg)._advance(1)
    assert normal.unmet_discharge_W[0] == pytest.approx(0)
    overloaded = HESSEmulator(t,t*0+1000,t*0+10,m,cfg)._advance(1)
    assert overloaded.unmet_discharge_W[0] > 900
    charging = HESSEmulator(t,t*0-1000,t*0,m,cfg)._advance(1)
    assert charging.unmet_discharge_W[0] == 0


def test_lifetime_uses_actual_current_limiter_not_battery_sharing():
    from types import SimpleNamespace as NS
    from sc_module_design.aging import lifetime_state_passes
    def state(limited, feasible=True):
        return NS(feasible=feasible, history=[NS(t_current_limit_s=limited)],
                  trace=NS(unmet_discharge_W=[329.34], P_bat=[1000]))
    assert lifetime_state_passes(state(0))
    assert not lifetime_state_passes(state(.001))
    assert not lifetime_state_passes(state(0, False))


def test_current_limiter_diagnostic_preserves_requested_peaks():
    import numpy as np
    from sc_module_design.emulation import ModuleModel, EmulationConfig, HESSEmulator
    m = ModuleModel(23,3,50,.022,33,48,42,1)
    t = np.arange(0,1,.001)
    em = HESSEmulator(t,t*0+1000,t*0,m,EmulationConfig(soc_control=False))
    trace = em._advance(1)
    assert trace.current_limited[0]
    assert trace.requested_current_extreme_A[0] < -1
    assert trace.I_sc_peak[0] == pytest.approx(1)
