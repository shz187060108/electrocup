from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path = 'configs/default.yaml') -> Dict[str, Any]:
    """Load YAML config and return a mutable dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def deep_update(base: Dict[str, Any], patch: Dict[str, Any] | None) -> Dict[str, Any]:
    """Recursively update a dictionary. Used for scenario overrides."""
    if not patch:
        return base
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ['output_dir', 'figure_dir', 'log_dir']:
        Path(cfg['project'][key]).mkdir(parents=True, exist_ok=True)
