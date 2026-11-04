"""Step 13: balancing concept selection (interface).

Not part of the quantitative procedure yet. A balancing model can be registered here later, e.g.
passive resistor balancing from the cell leakage current spread or active balancing from the
charge mismatch per cycle. The pipeline reports the step as 'skipped' until a model is registered.

    from sc_module_design.balancing import register_balancing

    @register_balancing("passive_resistor")
    def passive(row: dict, inputs) -> dict:
        ...
        return {"status": "done", "summary": "...", "values": {...}}
"""
from __future__ import annotations

from typing import Callable

BALANCING_MODELS: dict[str, Callable] = {}


def register_balancing(name: str):
    def deco(func):
        BALANCING_MODELS[name] = func
        return func
    return deco


def run_balancing(name: str, row: dict, inputs) -> dict:
    if name not in BALANCING_MODELS:
        return {"status": "skipped", "summary": "No balancing model registered.", "values": {}}
    return BALANCING_MODELS[name](row, inputs)
