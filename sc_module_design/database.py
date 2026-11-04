"""Cell and measurement database.

The CSV files in ``data/database`` are the editable source of truth (one file per
table, every row carries a ``source`` column). A SQLite file is built from them
automatically (``sc_database.sqlite``) so the data can also be queried with SQL.

Tables
------
cells                   datasheet data per cell (candidate flag, lifetime series)
lifetime_series         Arrhenius parameters per series (U0, T0, t0, dU, dT)
capacitance_fits        C0/k (model M1) and C_N (model M2) per cell/dataset/range/current
esr_measurements        ESR per cell/dataset/current/method
thermal_params          R_th / C_th per cell (position, test condition)
thermal_deltaT          measured temperature rise per sensor position (reference)
reference_iecon2023_table  values printed in the IECON 2023 table (for comparison)
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .paths import DB_DIR

TABLES = {
    "cells": "cells.csv",
    "lifetime_series": "lifetime_series.csv",
    "capacitance_fits": "capacitance_fits.csv",
    "esr_measurements": "esr_measurements.csv",
    "thermal_params": "thermal_params.csv",
    "thermal_deltaT": "thermal_deltaT.csv",
    "reference_iecon2023_table": "reference_iecon2023_table.csv",
}

# Fields a cell needs to take part in the candidate loop.
REQUIRED_CANDIDATE_FIELDS = ["c_nom_F", "u_rated_V", "esr_dc_mOhm", "i_max_A", "lifetime_series_id"]
# Fields that are only needed for the trade-off (missing -> NaN in the result).
OPTIONAL_TRADEOFF_FIELDS = ["mass_g", "diameter_mm", "length_mm", "price_EUR"]

# Dataset preference when several characterizations exist ("auto"): many DUTs first.
DATASET_PRIORITY = ["c", "a", "d", "b"]


@dataclass
class Resolved:
    """A value taken from the database together with where it came from."""
    value: float
    note: str
    warning: str = ""
    row: Optional[dict] = None


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)
    except (TypeError, ValueError):
        return False


class Database:
    def __init__(self, db_dir: Path | str = DB_DIR, build_sqlite: bool = True):
        self.db_dir = Path(db_dir)
        self.tables: dict[str, pd.DataFrame] = {}
        self.reload()
        if build_sqlite:
            try:
                self.build_sqlite()
            except Exception:  # read-only folder etc. -> CSV still works
                pass

    # ------------------------------------------------------------------ io
    def reload(self) -> None:
        for name, fname in TABLES.items():
            path = self.db_dir / fname
            self.tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        cells = self.tables["cells"]
        if not cells.empty and "candidate" in cells:
            cells["candidate"] = cells["candidate"].astype(str).str.lower().isin(["true", "1", "yes"])

    @property
    def sqlite_path(self) -> Path:
        return self.db_dir / "sc_database.sqlite"

    def build_sqlite(self) -> Path:
        """(Re)build the SQLite file if any CSV is newer than it."""
        path = self.sqlite_path
        csv_mtime = max((self.db_dir / f).stat().st_mtime for f in TABLES.values() if (self.db_dir / f).exists())
        if path.exists() and path.stat().st_mtime >= csv_mtime:
            return path
        with sqlite3.connect(path) as con:
            for name, df in self.tables.items():
                if not df.empty:
                    df.to_sql(name, con, if_exists="replace", index=False)
        return path

    def save_table(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.db_dir / TABLES[name], index=False)
        self.reload()
        if self.sqlite_path.exists():
            self.sqlite_path.unlink()
        self.build_sqlite()

    def query(self, sql: str) -> pd.DataFrame:
        self.build_sqlite()
        with sqlite3.connect(self.sqlite_path) as con:
            return pd.read_sql_query(sql, con)

    # ------------------------------------------------------------------ shortcuts
    @property
    def cells(self) -> pd.DataFrame:
        return self.tables["cells"]

    @property
    def lifetime(self) -> pd.DataFrame:
        return self.tables["lifetime_series"]

    @property
    def cfits(self) -> pd.DataFrame:
        return self.tables["capacitance_fits"]

    @property
    def esr(self) -> pd.DataFrame:
        return self.tables["esr_measurements"]

    @property
    def thermal(self) -> pd.DataFrame:
        return self.tables["thermal_params"]

    def cell(self, cell_id: str) -> dict:
        rows = self.cells[self.cells.cell_id == cell_id]
        if rows.empty:
            raise KeyError(f"Cell '{cell_id}' not in database")
        return rows.iloc[0].to_dict()

    def series(self, series_id: str) -> dict:
        rows = self.lifetime[self.lifetime.series_id == series_id]
        if rows.empty:
            raise KeyError(f"Lifetime series '{series_id}' not in database")
        return rows.iloc[0].to_dict()

    # ------------------------------------------------------------------ candidates
    def missing_fields(self, cell: dict) -> list[str]:
        return [f for f in REQUIRED_CANDIDATE_FIELDS if _isnan(cell.get(f)) or cell.get(f) == ""]

    def candidate_cells(self, include_ids: Optional[list[str]] = None) -> tuple[pd.DataFrame, list[str]]:
        """Cells usable in the candidate loop plus a list of skip messages."""
        df = self.cells.copy()
        msgs = []
        if include_ids:
            df = df[df.cell_id.isin(include_ids)]
        else:
            df = df[df.candidate]
        keep = []
        for _, row in df.iterrows():
            miss = self.missing_fields(row.to_dict())
            if miss:
                msgs.append(f"{row.cell_id}: skipped, missing {', '.join(miss)}")
            else:
                keep.append(row.cell_id)
        return df[df.cell_id.isin(keep)].reset_index(drop=True), msgs

    # ------------------------------------------------------------------ capacitance
    def capacitance_fit(self, cell_id: str, voltage_range: str = "90-20", current: str | float = "highest",
                        dataset: str = "auto") -> Optional[dict]:
        df = self.cfits
        if df.empty:
            return None
        df = df[(df.cell_id == cell_id) & (df.voltage_range == voltage_range)]
        if df.empty:
            return None
        if dataset != "auto":
            df = df[df.dataset == dataset]
        else:
            for ds in DATASET_PRIORITY:
                if (df.dataset == ds).any():
                    df = df[df.dataset == ds]
                    break
        if df.empty:
            return None
        if current == "highest":
            row = df.loc[df.current_A.idxmax()]
        elif current == "lowest":
            row = df.loc[df.current_A.idxmin()]
        else:
            row = df.loc[(df.current_A - float(current)).abs().idxmin()]
        return row.to_dict()

    # ------------------------------------------------------------------ esr
    def esr_measured(self, cell_id: str, methods=("M4", "M3", "M2", "M1"), current: str | float = "highest",
                     dataset: str = "auto") -> Optional[dict]:
        df = self.esr
        if df.empty:
            return None
        df = df[(df.cell_id == cell_id) & (df.plausible.astype(str).str.lower() == "true")]
        if dataset != "auto":
            df = df[df.dataset == dataset]
        for m in methods:
            d = df[df.method == m]
            if d.empty:
                continue
            if dataset == "auto":
                for ds in DATASET_PRIORITY:
                    if (d.dataset == ds).any():
                        d = d[d.dataset == ds]
                        break
            if current == "highest":
                return d.loc[d.current_A.idxmax()].to_dict()
            if current == "lowest":
                return d.loc[d.current_A.idxmin()].to_dict()
            return d.loc[(d.current_A - float(current)).abs().idxmin()].to_dict()
        return None

    # ------------------------------------------------------------------ thermal
    def thermal_own(self, cell_id: str) -> Optional[dict]:
        """R_th/C_th of the cell itself: measured (worst-case position) or datasheet columns."""
        df = self.thermal
        if not df.empty:
            d = df[df.cell_id == cell_id]
            if not d.empty:
                if "worst_case" in d and d.worst_case.astype(str).str.lower().eq("true").any():
                    d = d[d.worst_case.astype(str).str.lower() == "true"]
                row = d.loc[d.rth_KperW.idxmax()].to_dict()
                return dict(rth=float(row["rth_KperW"]), cth=float(row["cth_JperK"]), cell_id=cell_id,
                            note=f"measured, {row.get('position', '')}, cond. {row.get('condition', '')}",
                            source=row.get("source", ""))
        c = self.cell(cell_id)
        if not _isnan(c.get("rth_KperW")) and not _isnan(c.get("cth_JperK")):
            return dict(rth=float(c["rth_KperW"]), cth=float(c["cth_JperK"]), cell_id=cell_id,
                        note="datasheet", source=c.get("source", ""))
        return None

    def thermal_proxy(self, cell_id: str) -> Optional[dict]:
        """Equivalent from the database: same geometry first, else the next larger characterized cell."""
        own = self.thermal_own(cell_id)
        if own:
            return own
        cell = self.cell(cell_id)
        d, l = cell.get("diameter_mm"), cell.get("length_mm")
        vol = (math.pi * (d / 2) ** 2 * l) if not (_isnan(d) or _isnan(l)) else np.nan
        options = []
        for cid in self.cells.cell_id:
            if cid == cell_id:
                continue
            th = self.thermal_own(cid)
            if not th:
                continue
            o = self.cell(cid)
            od, ol = o.get("diameter_mm"), o.get("length_mm")
            ovol = math.pi * (od / 2) ** 2 * ol if not (_isnan(od) or _isnan(ol)) else np.nan
            same = (not _isnan(d)) and od == d and ol == l
            options.append((cid, th, same, ovol, o.get("c_nom_F")))
        if not options:
            return None
        same = [o for o in options if o[2]]
        if same:
            cid, th, *_ = same[0]
            th = dict(th)
            th["warning"] = (f"R_th/C_th taken from {cid} (same geometry). Proxy value, not measured for "
                             f"{cell_id}.")
            th["proxy_of"] = cid
            return th
        ref = vol if not _isnan(vol) else cell.get("c_nom_F")
        key = 3 if not _isnan(vol) else 4
        larger = [o for o in options if not _isnan(o[key]) and o[key] >= ref]
        pool = sorted(larger, key=lambda o: o[key]) if larger else sorted(options, key=lambda o: -o[key])
        cid, th, *_ = pool[0]
        th = dict(th)
        rel = "next larger" if larger else "largest available (smaller than this cell)"
        th["warning"] = (f"R_th/C_th taken from {cid} ({rel} characterized cell). Proxy value, not measured "
                         f"for {cell_id}. A larger can usually has a lower R_th, so the temperature can be "
                         f"underestimated.")
        th["proxy_of"] = cid
        return th


_DB_SINGLETON: Optional[Database] = None


def get_database(reload: bool = False) -> Database:
    global _DB_SINGLETON
    if _DB_SINGLETON is None or reload:
        _DB_SINGLETON = Database()
    return _DB_SINGLETON
