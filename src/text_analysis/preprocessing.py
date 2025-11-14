from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List

from .config_manager import AnalysisConfig

if TYPE_CHECKING:
	from .text_analyzer import TextPayload


class Preprocessor:
	"""Lightweight text preprocessor feeding downstream modules."""

	SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

	def __init__(self, config: AnalysisConfig) -> None:
		self._config = config

	def process(self, payload: "TextPayload") -> Dict[str, List[str]]:
		sentences = [s.strip() for s in self.SENTENCE_SPLIT.split(payload.content) if s.strip()]
		paragraphs = [p.strip() for p in re.split(r"\n\s*\n", payload.content) if p.strip()]
		if not paragraphs:
			paragraphs = [payload.content.strip()] if payload.content.strip() else []
		tokens = payload.content.split()
		return {
			"sentences": sentences,
			"paragraphs": paragraphs,
			"tokens": tokens,
		}
