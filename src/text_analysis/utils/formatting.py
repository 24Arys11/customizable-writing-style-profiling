from __future__ import annotations

from typing import Any, Dict, Iterable


def format_summary(report: Dict[str, Any]) -> str:
    lines: list[str] = []
    for section, data in report.items():
        lines.extend(_format_section(section, data))
        lines.append("")
    return "\n".join(line for line in lines if line.strip() or line == "")


def _format_section(name: str, data: Any, indent: int = 0) -> list[str]:
    indent_str = "  " * indent
    formatted: list[str] = []

    if isinstance(data, dict):
        formatted.append(f"{indent_str}{name}:")
        scalar_items = {
            key: value
            for key, value in data.items()
            if not isinstance(value, (dict, list, tuple))
        }
        if scalar_items:
            scalar_line = ", ".join(
                f"{key}={_format_value(value)}" for key, value in scalar_items.items()
            )
            formatted.append(f"{indent_str}  {scalar_line}")
        for key, value in data.items():
            if isinstance(value, dict):
                formatted.extend(_format_section(key, value, indent + 1))
            elif isinstance(value, (list, tuple)):
                list_values = ", ".join(_format_value(item) for item in value)
                formatted.append(f"{indent_str}  {key}: {list_values}")
    elif isinstance(data, (list, tuple)):
        formatted.append(f"{indent_str}{name}: {', '.join(_format_value(item) for item in data)}")
    else:
        formatted.append(f"{indent_str}{name}: {_format_value(data)}")

    return formatted


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
