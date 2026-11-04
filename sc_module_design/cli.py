"""Command line: run a preset and export everything.

    python -m sc_module_design.cli --preset iecon2023_legacy
    python -m sc_module_design.cli --preset thesis_characterized --out outputs/thesis --no-figures
    python -m sc_module_design.cli --list
"""
from __future__ import annotations

import argparse

from .config import list_presets, load_preset
from .pipeline import run_design
from .report import compare_with_reference, export_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Supercapacitor module design (flowchart of the thesis)")
    ap.add_argument("--preset", default="iecon2023_legacy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--list", action="store_true", help="list presets")
    a = ap.parse_args(argv)
    if a.list:
        for p in list_presets():
            print(p.stem)
        return 0
    inp = load_preset(a.preset)
    res = run_design(inp)
    for k, s in res.steps.items():
        print(f"[{s.status:8s}] {k:5s} {s.title}: {s.summary}")
    for w in res.warnings:
        print("WARNING:", w)
    if inp.opt.geometry == "legacy_iecon":
        cmp_ = compare_with_reference(res)
        if not cmp_.empty:
            bad = cmp_[~cmp_.match]
            print(f"\nIECON 2023 table: {cmp_.match.sum()}/{len(cmp_)} values reproduced. Deviations:")
            print(bad.to_string(index=False) if not bad.empty else "none")
    out = export_run(res, a.out, figures=not a.no_figures)
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
