"""Trade-off analysis (flowchart step 12): picks, Pareto front, weighted score."""
from __future__ import annotations

import numpy as np
import pandas as pd

PICKS = {
    "min_cost": ("cost_EUR", "min", "Lowest cost"),
    "min_mass": ("mass_kg", "min", "Lowest mass"),
    "min_volume": ("volume_dm3", "min", "Lowest volume"),
    "max_density": ("density_kg_dm3", "max", "Highest density (IECON 'Density' column)"),
    "min_esr": ("ESR_module_mOhm", "min", "Lowest module ESR"),
    "max_energy_density": ("E_density_Wh_kg", "max", "Highest usable energy per mass"),
}
TIE_BREAK = ["cost_EUR", "ESR_module_mOhm", "n_cells"]


def picks(df: pd.DataFrame) -> dict:
    out = {}
    feas = df[df.feasible] if "feasible" in df else df
    for key, (col, how, label) in PICKS.items():
        d = feas.dropna(subset=[col])
        if d.empty:
            continue
        best = d[col].max() if how == "max" else d[col].min()
        tied = d[np.isclose(d[col], best, rtol=1e-6, atol=1e-9)]
        tied = tied.sort_values([c for c in TIE_BREAK if c in tied])
        out[key] = dict(label=label, metric=col, cell_id=tied.iloc[0].cell_id,
                        ties=list(tied.cell_id) if len(tied) > 1 else [])
    return out


def pareto_mask(df: pd.DataFrame, cols=("cost_EUR", "mass_kg", "volume_dm3", "ESR_module_mOhm")) -> np.ndarray:
    cols = [c for c in cols if c in df and df[c].notna().all()]
    x = df[cols].to_numpy(float)
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(x <= x[i], axis=1) & np.any(x < x[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def weighted_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    cmap = {"cost": "cost_EUR", "mass": "mass_kg", "volume": "volume_dm3", "esr": "ESR_module_mOhm"}
    score = pd.Series(0.0, index=df.index)
    wsum = 0.0
    for k, w in weights.items():
        col = cmap.get(k, k)
        if col not in df or not w or df[col].isna().all():
            continue
        v = df[col].astype(float)
        rng = v.max() - v.min()
        score += w * ((v - v.min()) / rng if rng > 0 else 0.0)
        wsum += w
    return score / wsum if wsum else score


def annotate(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()
    feas = df.feasible if "feasible" in df else pd.Series(True, index=df.index)
    df["pareto"] = False
    if feas.any():
        df.loc[feas, "pareto"] = pareto_mask(df[feas])
    df["score"] = np.nan
    if feas.any():
        df.loc[feas, "score"] = weighted_score(df[feas], weights)
    return df
