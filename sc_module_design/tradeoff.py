"""Trade-off analysis (flowchart step 12): multi-criteria score, picks, Pareto front.

The cheapest module is not automatically the best one. Every feasible configuration is rated per
criterion on a 0 to 100 % scale (100 % = best candidate of this run), and the weighted mean is the
score. Weight presets cover typical design goals; the weights can be set freely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# key: (column, direction, label)
CRITERIA = {
    "cost": ("cost_EUR", "min", "Kosten"),
    "mass": ("mass_kg", "min", "Masse"),
    "volume": ("volume_dm3", "min", "Volumen"),
    "losses": ("P_loss_avg_W", "min", "Verluste (ESR)"),
    "thermal": ("dT_max_K", "min", "Temperaturanstieg"),
    "cells": ("n_cells", "min", "Zellanzahl (Montage, Balancing)"),
    "reserve": ("C_reserve", "max_capped", "Kapazitätsreserve"),
    "data": ("data_quality", "max", "Datenbasis (gemessen statt Datenblatt)"),
}
WEIGHT_PRESETS = {
    "balanced": ("Ausgewogen", {k: 1.0 for k in CRITERIA}),
    "cost": ("Kosten (wie IECON)", {"cost": 1.0}),
    "compact": ("Kompakt", {"mass": 1.0, "volume": 1.0, "cells": 0.5}),
    "robust": ("Robust / Lebensdauer", {"losses": 1.0, "thermal": 1.0, "reserve": 1.0, "data": 1.0, "cells": 0.5,
                                        "cost": 0.5}),
    "custom": ("Eigene Gewichtung", None),
}
PICKS = {
    "min_cost": ("cost_EUR", "min", "Niedrigste Kosten"),
    "min_mass": ("mass_kg", "min", "Niedrigste Masse"),
    "min_volume": ("volume_dm3", "min", "Kleinstes Volumen"),
    "max_density": ("density_kg_dm3", "max", "Höchste Dichte (IECON-Spalte 'Density')"),
    "min_esr": ("ESR_module_mOhm", "min", "Niedrigster Modul-ESR"),
    "max_energy_density": ("E_density_Wh_kg", "max", "Höchste nutzbare Energie je Masse"),
}
TIE_BREAK = ["cost_EUR", "ESR_module_mOhm", "n_cells"]


def preset_weights(name: str) -> dict:
    w = WEIGHT_PRESETS.get(name, WEIGHT_PRESETS["balanced"])[1]
    return {k: float(w.get(k, 0.0)) for k in CRITERIA} if w else {}


def picks(df: pd.DataFrame) -> dict:
    out = {}
    feas = df[df.feasible] if "feasible" in df else df
    for key, (col, how, label) in PICKS.items():
        if col not in feas:
            continue
        d = feas.dropna(subset=[col])
        if d.empty:
            continue
        best = d[col].max() if how == "max" else d[col].min()
        tied = d[np.isclose(d[col], best, rtol=1e-6, atol=1e-9)]
        tied = tied.sort_values([c for c in TIE_BREAK if c in tied])
        out[key] = dict(label=label, metric=col, cell_id=tied.iloc[0].cell_id,
                        ties=list(tied.cell_id) if len(tied) > 1 else [])
    if "score" in feas and feas.score.notna().any():
        best = feas.loc[feas.score.idxmax()]
        out["best_score"] = dict(label="Beste Gesamtbewertung", metric="score", cell_id=best.cell_id, ties=[])
    return out


def pareto_mask(df: pd.DataFrame, cols=("cost_EUR", "mass_kg", "volume_dm3", "ESR_module_mOhm")) -> np.ndarray:
    cols = [c for c in cols if c in df and df[c].notna().all()]
    x = df[cols].to_numpy(float)
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(x <= x[i], axis=1) & np.any(x < x[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def criterion_scores(df: pd.DataFrame, reserve_cap: float = 0.3) -> pd.DataFrame:
    """Per criterion 0..1 (1 = best in this run). NaN if the criterion is not available."""
    out = pd.DataFrame(index=df.index)
    for k, (col, how, _) in CRITERIA.items():
        if col not in df or df[col].isna().all():
            out[k] = np.nan
            continue
        v = df[col].astype(float)
        if how == "max_capped":
            v = v.clip(lower=0, upper=reserve_cap)
            how = "max"
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-12:
            out[k] = 1.0
        elif how == "min":
            out[k] = (hi - v) / (hi - lo)
        else:
            out[k] = (v - lo) / (hi - lo)
    return out


def score(df: pd.DataFrame, weights: dict, reserve_cap: float = 0.3) -> tuple[pd.Series, pd.DataFrame]:
    """Weighted score in % (100 = best in every weighted criterion) and the contribution per criterion."""
    cs = criterion_scores(df, reserve_cap)
    w = {k: float(weights.get(k, 0.0)) for k in CRITERIA}
    w = {k: v for k, v in w.items() if v > 0 and cs[k].notna().any()}
    tot = sum(w.values())
    contrib = pd.DataFrame(index=df.index)
    if not tot:
        return pd.Series(np.nan, index=df.index), contrib
    for k, v in w.items():
        contrib[k] = 100.0 * v / tot * cs[k].fillna(0.0)
    return contrib.sum(axis=1), contrib


def annotate(df: pd.DataFrame, weights: dict, reserve_cap: float = 0.3) -> pd.DataFrame:
    df = df.copy()
    if "c_measured" in df:
        df["data_quality"] = (df.c_measured.astype(float) + df.esr_measured.astype(float)
                              + df.thermal_measured.astype(float)) / 3.0
    feas = df.feasible if "feasible" in df else pd.Series(True, index=df.index)
    df["pareto"] = False
    df["score"] = np.nan
    if feas.any():
        df.loc[feas, "pareto"] = pareto_mask(df[feas])
        s, _ = score(df[feas], weights, reserve_cap)
        df.loc[feas, "score"] = s
        df["rank"] = df["score"].rank(ascending=False, method="min")
    return df
