from __future__ import annotations

from typing import Any, Dict


class Aggregator:
    """Combines module outputs into a single report."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def add(self, module_name: str, metrics: Dict[str, Any]) -> None:
        self._data[module_name] = metrics

    def finalize(self) -> Dict[str, Any]:
        return self._data
