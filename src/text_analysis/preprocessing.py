from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional

import spacy
from spacy.lang.en import English

from .config_manager import AnalysisConfig

if TYPE_CHECKING:
    from .text_analyzer import TextPayload


class Preprocessor:
    """Enhanced text preprocessor with spaCy NLP pipeline."""

    SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._nlp: Optional[spacy.Language] = None
        self._initialize_nlp()

    def _initialize_nlp(self) -> None:
        """Initialize spaCy pipeline with error handling for missing models."""
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to basic English pipeline if model not available
            print("Warning: en_core_web_sm model not found. Using basic tokenizer.")
            self._nlp = English()
            self._nlp.add_pipe("sentencizer")

    def process(self, payload: "TextPayload") -> Dict[str, List[str]]:
        # Basic preprocessing (fallback)
        sentences = [s.strip() for s in self.SENTENCE_SPLIT.split(payload.content) if s.strip()]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", payload.content) if p.strip()]
        if not paragraphs:
            paragraphs = [payload.content.strip()] if payload.content.strip() else []
        tokens = payload.content.split()

        # Enhanced preprocessing with spaCy
        enhanced_context = {
            "sentences": sentences,
            "paragraphs": paragraphs,
            "tokens": tokens,
        }

        if self._nlp is not None:
            doc = self._nlp(payload.content)
            enhanced_context.update({
                "spacy_doc": doc,
                "spacy_sentences": [sent.text.strip() for sent in doc.sents if sent.text.strip()],
                "spacy_tokens": [token.text for token in doc if not token.is_space],
                "pos_tags": [token.pos_ for token in doc if not token.is_space],
                "lemmas": [token.lemma_ for token in doc if not token.is_space],
                "dependencies": [(token.text, token.dep_, token.head.text) for token in doc if not token.is_space],
            })

        return enhanced_context
