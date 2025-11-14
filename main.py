from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.text_analysis.config_manager import ConfigManager
from src.text_analysis.text_analyzer import TextAnalyzer
from src.text_analysis.utils.formatting import format_summary, format_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular text analysis.")
    parser.add_argument("text_path", type=Path, help="Path to .txt file to analyze")
    parser.add_argument("config_path", type=Path, help="Path to config YAML file")
    parser.add_argument(
        "--output-type",
        choices=["summary", "json", "table"],
        default="summary",
        help="Choose output format: summary (insights), json (raw data), or table (metrics)",
    )
    parser.add_argument(
        "--verbosity",
        choices=["compact", "standard", "detailed"],
        default="standard",
        help="Output verbosity: compact (core metrics), standard (main insights), detailed (full analysis)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_manager = ConfigManager(args.config_path)
    analyzer = TextAnalyzer(config_manager.config)

    payload = analyzer.load_text(args.text_path)
    result = analyzer.analyze(payload)

    if args.output_type == "json":
        print(json.dumps(result, indent=2))
    elif args.output_type == "table":
        print(format_table(result, verbosity=args.verbosity))
    else:
        print(format_summary(result, verbosity=args.verbosity))


if __name__ == "__main__":
    main()
