from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AnalysisConfig:
    """Lightweight container for runtime analysis settings."""

    raw: Dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "AnalysisConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(raw=data)


class ConfigManager:
    """Loads and stores the active analysis configuration."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._config = AnalysisConfig.from_path(config_path)

    @property
    def config(self) -> AnalysisConfig:
        return self._config

    def reload(self) -> AnalysisConfig:
        self._config = AnalysisConfig.from_path(self._config_path)
        return self._config
