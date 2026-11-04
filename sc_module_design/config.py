"""Design inputs (application requirements + method options). Load/save as YAML presets."""
from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional

import yaml

from .paths import PRESET_DIR, resolve


@dataclass
class ProfileSpec:
    path: str = ""                       # relative to project root; empty -> manual load values
    scale: float = 1.0
    time_col: Optional[str] = None
    power_col: Optional[str] = None
    phase_factor: float = 1.0            # P = U I phase_factor if only U and I are given
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    resample_dt: Optional[float] = None


@dataclass
class ManualLoad:
    """Fallback / override when no profile is available (interface can be skipped or filled by hand)."""
    use: str = "if_no_profile"           # never | if_no_profile | always
    P_peak_W: Optional[float] = None
    P_avg_W: Optional[float] = None
    E_dch_J: Optional[float] = None
    E_ch_J: Optional[float] = None


@dataclass
class SplitSpec:
    name: str = "moving_average"
    params: dict = field(default_factory=lambda: {"window_s": 20.0, "alignment": "legacy_matlab"})


@dataclass
class BrakingSpec:
    enabled: bool = True
    mode: str = "kinematic"              # kinematic (m, v) | direct (E, P)
    mass_kg: float = 880.0
    v_mps: float = 1.5
    t_brake_s: float = 1.0               # P = 2E/t (1.0 s -> 1980 W as in the thesis, 1.2 s -> 1650 W slide)
    E_J: Optional[float] = None
    P_W: Optional[float] = None


@dataclass
class Requirements:
    U_bus_V: float = 48.0
    U_C0max_V: Optional[float] = None    # None -> U_bus
    P_max_mode: str = "peak"             # peak | peak_minus_avg | manual
    P_max_W: Optional[float] = None      # used when P_max_mode == manual
    I_C0max_A: float = 77.7              # module / interface converter current limit
    U_C0min_override_V: Optional[float] = None
    T_cell_C: float = 65.0               # temperature for the lifetime derating
    t_life_h: float = 40000.0
    L_EOL: float = 0.7
    T_amb_C: float = 25.0                # ambient for the thermal check
    dT_max_K: float = 40.0


@dataclass
class Options:
    energy_method: str = "zero_crossing_legacy"
    scenarios: list = field(default_factory=lambda: ["profile", "profile_braking"])
    series_override: Optional[str] = None          # lifetime series for all cells (else per cell)
    candidate_ids: Optional[list] = None           # None -> all cells with candidate == True
    # capacitance source
    c_source: str = "datasheet"                    # datasheet | ceq_M1 | cn_M2
    cfit_range: str = "90-20"                      # 90-20 | 80-40
    cfit_current: object = "highest"               # highest | lowest | nearest_rms | <float A>
    cfit_dataset: str = "auto"
    # ESR source
    esr_source: str = "datasheet_dc"               # datasheet_dc | datasheet_ac | measured
    esr_methods: list = field(default_factory=lambda: ["M4", "M3", "M2", "M1"])
    esr_current: object = "highest"
    # geometry / trade-off
    geometry: str = "corrected"                    # corrected | legacy_iecon
    packing_factor: float = 1.0                    # module volume = cell volume / packing_factor
    # thermal
    thermal_mode: str = "available_only"           # skip | available_only | proxy
    thermal_current: str = "energy_balance"        # energy_balance | conservative_umin
    thermal_profile: str = "periodic"              # periodic | single_pass
    thermal_dt_bin_s: float = 0.01
    P_limit: int = 200
    # trade-off weights (normalised metrics, lower is better)
    weights: dict = field(default_factory=lambda: {"cost": 1.0, "mass": 1.0, "volume": 1.0, "esr": 1.0})


@dataclass
class DesignInputs:
    name: str = "design"
    description: str = ""
    profile: ProfileSpec = field(default_factory=ProfileSpec)
    manual_load: ManualLoad = field(default_factory=ManualLoad)
    split: SplitSpec = field(default_factory=SplitSpec)
    braking: BrakingSpec = field(default_factory=BrakingSpec)
    req: Requirements = field(default_factory=Requirements)
    opt: Options = field(default_factory=Options)
    selected: dict = field(default_factory=dict)   # scenario -> {cell_id, P} final choice
    reference: dict = field(default_factory=dict)  # expected values for comparison (optional)

    # ------------------------------------------------------------------ yaml
    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "DesignInputs":
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DesignInputs":
        return _from_dict(cls, d or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DesignInputs":
        with open(resolve(path), encoding="utf8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> Path:
        path = resolve(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
        return path


def _from_dict(cls, d: dict):
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        default = f.default_factory() if f.default_factory is not MISSING else None
        if is_dataclass(default) and isinstance(v, dict):
            kw[f.name] = _from_dict(type(default), v)
        else:
            kw[f.name] = v
    return cls(**kw)


def list_presets() -> list[Path]:
    return sorted(PRESET_DIR.glob("*.yaml"))


def load_preset(name: str) -> DesignInputs:
    p = PRESET_DIR / (name if name.endswith(".yaml") else name + ".yaml")
    return DesignInputs.from_yaml(p)
