import unittest
from pathlib import Path

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.modules.pattern_matching import PatternMatchingModule
from src.text_analysis.text_analyzer import TextPayload


class PatternModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = TextPayload(
            path=Path("/tmp/text.txt"),
            content=(
                "First, do this. Second, do that. Third, wrap up."
            ),
        )
        self.context = {
            "sentences": [
                "First, do this.",
                "Second, do that.",
                "Third, wrap up.",
            ],
            "paragraphs": [
                "First, do this. Second, do that. Third, wrap up."
            ],
            "tokens": self.payload.content.split(),
        }

    def test_sentence_scope_counts_per_sentence(self) -> None:
        config = AnalysisConfig(
            raw={
                "metrics": {"normalization_basis": 1000},
                "regex_patterns": [
                    {
                        "name": "enumerations",
                        "scope": "sentence",
                        "variants": [
                            {"pattern": r"\bfirst\b"},
                            {"pattern": r"\bsecond\b"},
                        ],
                    }
                ],
            }
        )
        module = PatternMatchingModule(config)
        result = module.compute(self.payload, self.context)
        enumerations = result["enumerations"]

        self.assertEqual(enumerations["variants"], {"variant_1": 1, "variant_2": 1})
        self.assertEqual(enumerations["total_occurrences"], 2)
        # mean should be total / sentence_count = 2/3
        self.assertAlmostEqual(
            enumerations["mean_occurrences_per_sentence"],
            2 / 3,
            places=6,
        )

    def test_paragraph_scope_counts(self) -> None:
        payload = TextPayload(
            path=Path("/tmp/text.txt"),
            content="Paragraph one mentions however.\n\nParagraph two also however.",
        )
        context = {
            "sentences": payload.content.split("."),
            "paragraphs": [p.strip() for p in payload.content.split("\n\n") if p.strip()],
            "tokens": payload.content.split(),
        }
        config = AnalysisConfig(
            raw={
                "regex_patterns": [
                    {
                        "name": "contrast",
                        "scope": "paragraph",
                        "variants": [
                            {"pattern": r"however"},
                        ],
                    }
                ]
            }
        )
        module = PatternMatchingModule(config)
        result = module.compute(payload, context)
        contrast = result["contrast"]
        self.assertEqual(contrast["total_occurrences"], 2)
        self.assertEqual(contrast["mean_occurrences_per_paragraph"], 1.0)

    def test_multiple_scopes(self) -> None:
        config = AnalysisConfig(
            raw={
                "regex_patterns": [
                    {
                        "name": "enumerations",
                        "scopes": ["sentence", "paragraph"],
                        "variants": [
                            {"pattern": r"\bfirst\b"},
                        ],
                    }
                ]
            }
        )
        module = PatternMatchingModule(config)
        result = module.compute(self.payload, self.context)
        enumerations = result["enumerations"]
        self.assertIn("mean_occurrences_per_sentence", enumerations)
        self.assertIn("mean_occurrences_per_paragraph", enumerations)
        self.assertEqual(enumerations["total_occurrences"], 1)


if __name__ == "__main__":
    unittest.main()
