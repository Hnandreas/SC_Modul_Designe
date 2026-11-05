"""Simple linear degradation, normalized by the established Arrhenius lifetime.

R/R0=1+t/Lref; C/C0=1-(1-L_EOL)*t/Lref. The linear trajectory is
an explicit engineering assumption; no humidity-test curve is used.
"""
from copy import deepcopy
import numpy as np
from .equations import lifetime_h


def linear_factors(hours, reference_h, capacity_eol=.7):
    if reference_h <= 0 or not np.isfinite(reference_h) or hours < 0 or not 0 < capacity_eol <= 1:
        raise ValueError("Invalid lifetime inputs")
    fraction = float(hours) / reference_h
    if not np.isfinite(fraction) or fraction > 1 + 1e-12:
        raise ValueError("Linear precheck is limited to the assumed EOL endpoint")
    return 1 - (1-capacity_eol)*fraction, 1+fraction


def find_functional_lifetime(evaluate, reference_h, steps=7):
    """Largest passing age bracket, assuming functional margin decreases with age.

    Returns a tested passing lower bound and failing upper bound. If BOL
    already fails, there is no positive verified service life.
    """
    good, initial = evaluate(0.)
    if not good:
        return 0., 0., initial, False
    good, final = evaluate(reference_h)
    if good:
        return reference_h, reference_h, final, True
    lo, hi, best = 0., reference_h, initial
    for _ in range(steps):
        mid = (lo+hi)/2
        passed, result = evaluate(mid)
        if passed: lo, best = mid, result
        else: hi = mid
    return lo, hi, best, True


def lifetime_state_passes(design):
    """Normal battery sharing is allowed; actual SC current-limit activation fails."""
    return bool(design.feasible and design.history[-1].t_current_limit_s <= 1e-12)


def reference_lifetime(res, scenario, cell_id, db):
    cell = db.cell(cell_id)
    series = db.series(res.inputs.opt.series_override or cell['lifetime_series_id'])
    row = res.candidates[scenario].set_index('cell_id').loc[cell_id]
    voltage = res.scenarios[scenario].U_max / int(row['S'])
    temperature = res.inputs.req.T_cell_C
    reference = lifetime_h(voltage, temperature, **{k:float(series[k]) for k in ('u0_V','T0_C','t0_h','dU_V','dT_K')})
    return reference, temperature, voltage


def lifetime_check(res, scenario, cell_id, cfg, desired_h, db):
    from .emudesign import emulation_design
    req = res.inputs.req
    reference, temperature, voltage = reference_lifetime(res, scenario, cell_id, db)
    evaluated = {}
    def evaluate(hours):
        if hours not in evaluated:
            local = deepcopy(cfg)
            local.aging_capacity_factor, local.aging_esr_factor = linear_factors(hours, reference, req.L_EOL)
            d = emulation_design(res, scenario, cell_id, local, check_eol=False,
                                 include_braking=False, db=db, increase_P=False)
            passed = lifetime_state_passes(d)
            evaluated[hours] = (passed, d)
        return evaluated[hours]
    lower, upper, best, bol_passed = find_functional_lifetime(evaluate, reference)
    if desired_h <= reference:
        target_passed, target = evaluate(desired_h)
        if target_passed and desired_h > lower:
            lower, best = desired_h, target
    else:
        target_passed, target = False, best
    shown = target if target_passed else best
    shown_hours = desired_h if target_passed else lower
    shown.lifetime_report = dict(desired_h=desired_h, reference_h=reference,
        lower_h=lower, upper_h=upper, target_passed=target_passed,
        bol_passed=bol_passed, shown_h=shown_hours, temperature_C=temperature,
        voltage_V=voltage, esr_factor=shown.emulator.cfg.aging_esr_factor,
        capacity_factor=shown.emulator.cfg.aging_capacity_factor,
        current_limit_s=shown.history[-1].t_current_limit_s)
    if not bol_passed:
        shown.feasible = False
        shown.iterations[-1].ok = False
        shown.iterations[-1].reason = 'Verworfen: schon ungealtert nicht erfüllt; ggf. P erhöhen und neu prüfen'
        shown.notes.append('Keine Lebensdauersuche: Kandidat besteht die Prüfung im Neuzustand nicht.')
    elif not target_passed:
        shown.feasible = False
        shown.iterations[-1].ok = False
        shown.iterations[-1].reason = f'Wunschlebensdauer nicht erfüllt; geprüft erreichbar: {lower:.0f} h'
        shown.notes.append('Wunschlebensdauer nicht erfüllt; dargestellt ist der geprüfte frühere Zustand.')
    return shown
