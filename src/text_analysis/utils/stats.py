from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable, Optional


def _safe_mean(values: Iterable[float]) -> float:
    data = list(values)
    return mean(data) if data else 0.0


def _safe_std(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    return pstdev(data)


def summarize_counts(
    counts: Iterable[int],
    total_reference: Optional[int] = None,
    normalization_basis: Optional[int] = None,
) -> dict:
    data = list(counts)
    total = sum(data)
    mean_value = _safe_mean(data)
    std_dev = _safe_std(data)
    variance = std_dev**2
    coefficient_of_variation = std_dev / mean_value if mean_value else 0.0

    normalized = None
    normalized_key = None
    if total_reference and normalization_basis:
        normalized = (total / total_reference) * normalization_basis
        normalized_key = f"frequency_per_{normalization_basis}_tokens"

    return {
        "total": total,
        "mean_per_unit": mean_value,
        "std_dev_per_unit": std_dev,
        "variance": variance,
        "coefficient_of_variation": coefficient_of_variation,
        **({normalized_key: normalized} if normalized_key else {}),
    }


def summarize_numeric(values: Iterable[float]) -> dict:
    data = list(values)
    if not data:
        return {
            "sample_size": 0,
            "mean_value": 0.0,
            "std_dev": 0.0,
            "variance": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
            "coefficient_of_variation": 0.0,
        }

    mean_value = _safe_mean(data)
    std_dev = _safe_std(data)
    variance = std_dev**2
    coefficient_of_variation = std_dev / mean_value if mean_value else 0.0

    return {
        "sample_size": len(data),
        "mean_value": mean_value,
        "std_dev": std_dev,
        "variance": variance,
        "min_value": min(data),
        "max_value": max(data),
        "coefficient_of_variation": coefficient_of_variation,
    }
