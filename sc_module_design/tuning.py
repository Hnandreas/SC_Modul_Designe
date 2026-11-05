"""Bounded emulation-only tuning. Analytic sizing and module design stay fixed."""
from copy import copy, deepcopy
import math
import numpy as np
from .aging import reference_lifetime, linear_factors, lifetime_check, lifetime_state_passes
from .emudesign import emulation_design


def tuning_candidates(start_A, window_s, mode):
    if not 0 < start_A <= 10:
        raise ValueError("Recharge limit must be positive and at most 10 A")
    limits = [float(start_A)]
    if mode != 'off':
        limits += [float(x) for x in range(math.floor(start_A)+1,11)]
    windows = [window_s]
    if mode == 'both' and window_s is not None:
        windows += [max(.1, window_s*f) for f in (.75,.5,.25)]
    seen = set()
    for window in windows:
        for limit in limits:
            if (window,limit) not in seen:
                seen.add((window,limit))
                yield window,limit


def run_tuned_emulation(res, scenario, cell_id, cfg, check_eol, check_life,
                        window_s, mode, db):
    attempts, best = [], None
    for window, limit in tuning_candidates(cfg.pi_max_A, window_s, mode):
        local_res = copy(res)
        local_res.inputs = deepcopy(res.inputs)
        if window is not None:
            local_res.inputs.split.params['window_s'] = window
        local_cfg = deepcopy(cfg)
        local_cfg.pi_max_A = limit
        if check_life:
            ref, _, _ = reference_lifetime(local_res, scenario, cell_id, db)
            local_cfg.aging_capacity_factor, local_cfg.aging_esr_factor = linear_factors(
                min(res.inputs.req.t_life_h, ref), ref, res.inputs.req.L_EOL)
        result = emulation_design(local_res, scenario, cell_id, local_cfg,
                                  check_eol=check_eol and not check_life,
                                  include_braking=False, db=db, increase_P=False)
        passed = lifetime_state_passes(result)
        baseline = result.emulator.bat.mean
        rms_error = float(np.sqrt(np.mean((result.trace.P_bat-baseline)**2))/cfg.U_bus_V)
        duration = float(result.history[-1].t_current_limit_s)
        attempts.append(dict(window_s=window, recharge_A=limit, passed=passed,
                             current_limit_s=duration, battery_deviation_rms_A=rms_error))
        rank = (bool(result.feasible), -duration, -rms_error)
        if best is None or rank > best[0]:
            best = (rank,result,local_res,deepcopy(local_cfg),window,limit)
        if passed:
            best = (rank,result,local_res,deepcopy(local_cfg),window,limit)
            break
    _,result,local_res,local_cfg,window,limit = best
    if check_life:
        result = lifetime_check(local_res, scenario, cell_id, local_cfg,
                                res.inputs.req.t_life_h, db)
    result.tuning_report = dict(mode=mode, window_s=window, recharge_A=limit,
                               discharge_A=-cfg.pi_min_A, attempts=attempts,
                               matched=any(a['passed'] for a in attempts))
    return result
