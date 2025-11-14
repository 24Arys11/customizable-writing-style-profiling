# Text Analysis Engine

A modular text-profiling toolkit that extracts stylistic and pattern-based signals from plain-text documents. It consumes a YAML config preset, runs a pipeline of metric modules, and outputs structured JSON that you can compare across authors or against LLM outputs.

## Project layout

```
config/                     # Multiple presets (select via CLI argument)
main.py                     # CLI entrypoint
src/text_analysis/
	├── config_manager.py     # YAML loader + hot reload
	├── preprocessing.py      # Lightweight tokenizer/sentence splitter
	├── modules/              # Analysis modules (sentence, pattern, lexical, word lists)
	├── utils/stats.py        # Shared statistics helpers
	└── text_analyzer.py      # Orchestrates preprocessing/modules/aggregation
```

## Features (current snapshot)

- Configurable regex patterns with multiple variants under one logical label plus dispersion stats.
- Word-list metrics that report totals, normalized frequency, standard deviation, coefficient of variation, and per-term counts.
- Sentence + paragraph metrics (counts, sentence-length dispersion) derived from the preprocessor output.
- Lexical richness metrics (token/type counts, TTR, hapax ratio, word-length distribution).
- Modular architecture: enable/disable metric families via `analysis_flags` in the config file.
- Flexible pattern scopes (sentence/paragraph/document) with optional multi-scope reports per pattern.

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py "data\me\GPT Conversations.txt" config\sample_config.yaml
```

Swap `config\sample_config.yaml` for any preset in `config/` (e.g., `authorship_focus.yaml`). By default the CLI prints a human-readable summary (metrics rounded to 3 decimals). Pass `--output-type json` if you prefer the raw structured payload.

## Config presets

- `sample_config.yaml`: showcases contrastive-definition patterns, “feels vs is” variants, hedges, and intensifiers.
- `authorship_focus.yaml`: geared toward attribution experiments (rhetorical contrasts, enumerations, function words).

Each pattern can declare several `variants` and one or more `scopes` (sentence/paragraph/document), while word lists include descriptions and term inventories. A global `metrics.normalization_basis` (default 1000) controls frequency calculations (per 1k words).

## Extending metrics

Add a new module under `src/text_analysis/modules/`, inherit from `AnalysisModule`, and register it inside `TextAnalyzer._build_modules()`. Reuse `utils.stats` helpers to compute totals/frequency/std/CV so outputs stay consistent. Additional presets can be dropped into `config/` without touching code; just pass the desired file to `main.py`.

## Testing philosophy

This project is for personal experimentation; only lightweight smoke tests (e.g., running the CLI on sample data) are expected. Add focused unit tests if you create intricate math helpers or parsers, but there is no heavy CI requirement. A small regression test exists for the pattern module to guard basic counting logic.
