from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .base import AnalysisModule
from ..utils.stats import summarize_counts


class PatternMatchingModule(AnalysisModule):
    """Counts regex variants per logical pattern and reports dispersion stats."""

    @property
    def name(self) -> str:
        return "pattern_metrics"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        settings = self._config.raw
        normalization_basis = settings.get("metrics", {}).get("normalization_basis")
        total_tokens = len(context.get("tokens", [])) or None

        patterns: List[Dict[str, Any]] = settings.get("regex_patterns", [])
        report: Dict[str, Any] = {}

        sentences = context.get("sentences", [])
        paragraphs = context.get("paragraphs", [])

        for entry in patterns:
            pattern_name = entry.get("name")
            raw_variants = entry.get("variants", [])
            fallback_pattern = entry.get("pattern")
            if not pattern_name or (not raw_variants and not fallback_pattern):
                continue

            if not raw_variants and fallback_pattern:
                raw_variants = [{"pattern": fallback_pattern}]

            scopes = entry.get("scopes")
            if isinstance(scopes, list) and scopes:
                scope_list = [str(scope).lower() for scope in scopes if scope]
            else:
                scope_list = [(entry.get("scope") or "document").lower()]

            variant_details: Dict[str, int] = {}
            # Precompile variant expressions for reuse within unit counting
            compiled_variants = []
            for idx, variant in enumerate(raw_variants):
                expr = variant.get("pattern")
                label = variant.get("label") or f"variant_{idx+1}"
                if not expr:
                    continue
                compiled_variants.append((label, re.compile(expr, flags=re.IGNORECASE)))

            for label, pattern in compiled_variants:
                variant_details[label] = len(pattern.findall(payload.content))

            entry_report: Dict[str, Any] = {
                "variant_counts": list(variant_details.values()),
                "variant_labels": list(variant_details.keys()),
                "total_matches": sum(variant_details.values()),
            }
            frequency_value = None
            frequency_key_name = None

            for idx_scope, scope in enumerate(scope_list):
                text_units = self._get_units(payload.content, sentences, paragraphs, scope)
                if not text_units:
                    text_units = [payload.content]

                unit_counts: List[int] = []
                for unit in text_units:
                    count = sum(len(pattern.findall(unit)) for _, pattern in compiled_variants)
                    unit_counts.append(count)

                stats = summarize_counts(
                    unit_counts,
                    total_reference=total_tokens,
                    normalization_basis=normalization_basis,
                )

                unit_label = self._unit_label(scope)
                entry_report[f"mean_occurrences_per_{unit_label}"] = stats["mean_per_unit"]
                entry_report[f"std_dev_occurrences_per_{unit_label}"] = stats["std_dev_per_unit"]
                entry_report[f"variance_occurrences_per_{unit_label}"] = stats["variance"]
                cv_value = stats["coefficient_of_variation"]
                if idx_scope == 0:
                    entry_report["occurrence_coefficient_of_variation"] = cv_value
                entry_report[f"occurrence_coefficient_of_variation_{unit_label}"] = cv_value
                entry_report.setdefault("total_occurrences", stats["total"])

                frequency_key = next(
                    (key for key in stats.keys() if key.startswith("frequency_per_")),
                    None,
                )
                if frequency_key:
                    frequency_value = stats[frequency_key]
                    frequency_key_name = frequency_key

            if frequency_value is not None and frequency_key_name:
                entry_report[frequency_key_name] = frequency_value

            report[pattern_name] = entry_report

        return report

    @staticmethod
    def _get_units(
        content: str,
        sentences: Iterable[str],
        paragraphs: Iterable[str],
        scope: str,
    ) -> List[str]:
        if scope == "sentence":
            return list(sentences)
        if scope == "paragraph":
            para_list = [p.strip() for p in paragraphs if p.strip()]
            if para_list:
                return para_list
            paragraphs_fallback = [p.strip() for p in content.split("\n\n") if p.strip()]
            return paragraphs_fallback
        # document/default scope -> treat full content as single unit
        return [content]

    @staticmethod
    def _unit_label(scope: str) -> str:
        if scope == "sentence":
            return "sentence"
        if scope == "paragraph":
            return "paragraph"
        return "document"
