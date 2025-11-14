from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from ..config_manager import AnalysisConfig

if TYPE_CHECKING:
    from ..text_analyzer import TextPayload


class AnalysisModule(ABC):
    """Interface for metric modules attached to the analyzer."""

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def enabled(self) -> bool:
        flags = self._config.raw.get("analysis_flags", {})
        return flags.get(self.name, True)

    @abstractmethod
    def compute(self, payload: "TextPayload", context: Dict[str, Any]) -> Dict[str, Any]:
        ...
