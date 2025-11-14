from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .config_manager import AnalysisConfig
from .aggregator import Aggregator
from .modules import (
    AIDetectionMetric,
    DiscourseFlowMetric,
    LexicalMetrics,
    PatternMatchingModule,
    PunctuationMetrics,
    RhythmMetrics,
    SentenceMetrics,
    SentimentMetrics,
    SemanticMetrics,
    StylisticDeviceMetric,
    SyntaxMetrics,
    VocabularySophisticationMetric,
    WordListMetrics,
)
from .preprocessing import Preprocessor


@dataclass
class TextPayload:
    path: Path
    content: str


class TextAnalyzer:
    """Coordinates preprocessing and modular metric extraction."""

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._preprocessor = Preprocessor(config)
        self._modules = self._build_modules()

    def _build_modules(self) -> List[Any]:
        modules = [
            SentenceMetrics(self._config),
            PatternMatchingModule(self._config),
            WordListMetrics(self._config),
            LexicalMetrics(self._config),
            SyntaxMetrics(self._config),
            PunctuationMetrics(self._config),
            SentimentMetrics(self._config),
            SemanticMetrics(self._config),
            RhythmMetrics(self._config),
            VocabularySophisticationMetric(self._config),
            StylisticDeviceMetric(self._config),
            DiscourseFlowMetric(self._config),
            AIDetectionMetric(self._config),
        ]
        return [module for module in modules if module.enabled()]

    def load_text(self, path: Path) -> TextPayload:
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        content = path.read_text(encoding="utf-8")
        return TextPayload(path=path, content=content)

    def analyze(self, payload: TextPayload) -> Dict[str, Any]:
        context = self._preprocessor.process(payload)
        aggregator = Aggregator()

        base_metadata = {
            "text_length": len(payload.content),
            "config_sections": list(self._config.raw.keys()),
        }
        aggregator.add("metadata", base_metadata)

        for module in self._modules:
            metrics = module.compute(payload, context)
            aggregator.add(module.name, metrics)

        return aggregator.finalize()
