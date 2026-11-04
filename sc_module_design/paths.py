"""Central project paths. Everything is resolved relative to the project root."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "database"
PRESET_DIR = DATA_DIR / "presets"
PROFILE_DIR = DATA_DIR / "profiles"
PRIVATE_PROFILE_DIR = PROFILE_DIR / "private"
PLUGIN_DIR = PROJECT_ROOT / "plugins"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def resolve(path: str | Path) -> Path:
    """Resolve a path given relative to the project root (or absolute)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)
