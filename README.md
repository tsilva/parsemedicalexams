<div align="center">
  <img src="logo.png" alt="medical-exams-parser" width="280"/>

  # medical-exams-parser

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
  [![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-purple.svg)](https://openrouter.ai)

  **Extract and summarize medical exam reports from PDFs using Vision AI**

  [Features](#features) · [Quick Start](#quick-start) · [Configuration](#configuration) · [Output Format](#output-format)
</div>

---

## Features

- **Vision-powered extraction** — Uses Vision LLMs to read X-rays, MRIs, ultrasounds, endoscopies, and more directly from PDF scans
- **Self-consistency voting** — Runs multiple extractions and votes on the best result for maximum reliability
- **Intelligent classification** — Automatically categorizes exams (imaging, ultrasound, endoscopy, other) and standardizes naming
- **Clinical summarization** — Preserves all findings, impressions, and recommendations while filtering noise
- **Markdown output with YAML frontmatter** — Clean, structured files ready for Obsidian, static sites, or further processing
- **Smart caching** — Persistent JSON caches avoid redundant API calls and allow manual overrides

## Quick Start

### 1. Install

```bash
pip install -e .
```

> **Requires [Poppler](https://poppler.freedesktop.org/)** for PDF processing:
> - macOS: `brew install poppler`
> - Ubuntu: `apt-get install poppler-utils`

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
OPENROUTER_API_KEY=your_api_key_here
INPUT_PATH=/path/to/your/exam/pdfs
OUTPUT_PATH=/path/to/output
```

### 3. Run

```bash
python main.py
```

## How It Works

```
┌─────────────┐    ┌─────────────────┐    ┌────────────────┐    ┌──────────────┐    ┌────────────┐
│  PDF Input  │───▶│  Preprocessing  │───▶│ Vision LLM ×N  │───▶│ Standardize  │───▶│  Markdown  │
│             │    │  (grayscale,    │    │  + voting      │    │  + classify  │    │   Output   │
│             │    │   resize)       │    │                │    │              │    │            │
└─────────────┘    └─────────────────┘    └────────────────┘    └──────────────┘    └────────────┘
```

1. **PDF → Images** — Converts each page to grayscale, resizes, and enhances contrast
2. **Vision LLM extraction** — Extracts exam data using function calling (runs N times for reliability)
3. **Self-consistency voting** — If extractions differ, LLM votes on the best result
4. **Standardization** — Classifies exam type and standardizes the name via LLM with caching
5. **Summarization** — Generates document-level clinical summaries preserving all findings

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key ([get one here](https://openrouter.ai)) |
| `INPUT_PATH` | Directory containing exam PDFs |
| `OUTPUT_PATH` | Where to write output files |
| `EXTRACT_MODEL_ID` | Vision model for extraction (default: `google/gemini-2.5-flash`) |
| `SUMMARIZE_MODEL_ID` | Model for summarization |
| `N_EXTRACTIONS` | Number of extraction runs for voting (default: 3) |

### Using Profiles

Profiles let you save different input/output configurations for different use cases:

```bash
# Create a profile from template
cp profiles/_template.json profiles/myprofile.json

# Run with profile
python main.py --profile myprofile

# List available profiles
python main.py --list-profiles
```

Profile files support `"inherit_from_env": true` to use `.env` values as defaults.

### Advanced CLI Options

```bash
# Regenerate markdown files from existing JSON extraction data
python main.py --profile myprofile --regenerate

# Reprocess a specific document
python main.py -p myprofile -d exam_2024.pdf

# Reprocess a specific page within a document
python main.py -p myprofile -d exam_2024.pdf --page 2
```

## Output Format

The parser generates structured markdown files with YAML frontmatter:

```
output/
├── {document}/
│   ├── {document}.1.md      # Page 1 transcription + metadata (YAML frontmatter)
│   ├── {document}.2.md      # Page 2 transcription + metadata
│   └── {document}.summary.md # Document-level summary
```

### Transcription File Structure

Each `.md` file contains:

```yaml
---
date: 2024-01-15
exam_type: imaging
exam_name_raw: "RX TORAX PA Y LAT"
exam_name_standardized: "Chest X-Ray PA and Lateral"
source_file: exam_2024.pdf
page_number: 1
---

[Full transcription text here...]
```

### Metadata Fields

| Field | Description |
|-------|-------------|
| `date` | Exam date (YYYY-MM-DD) |
| `exam_type` | Category: `imaging`, `ultrasound`, `endoscopy`, `other` |
| `exam_name_raw` | Exam name exactly as written in document |
| `exam_name_standardized` | Cleaned, English-standardized name |
| `source_file` | Source PDF filename |
| `page_number` | Page number in PDF |

## Architecture

```
medical-exams-parser/
├── main.py              # Pipeline orchestration, CLI handling
├── extraction.py        # Pydantic models, Vision LLM extraction, voting
├── standardization.py   # Exam type classification with JSON cache
├── summarization.py     # Document-level clinical summarization
├── config.py            # ExtractionConfig (.env) + ProfileConfig (profiles/)
├── utils.py             # Image preprocessing, logging, JSON utilities
├── prompts/             # Externalized LLM prompts as markdown
├── profiles/            # User-specific path configurations
└── config/cache/        # Persistent LLM response caches (user-editable)
```

### Key Design Patterns

- **Two-column naming**: `*_raw` (exact from document) + `*_standardized` (LLM-mapped)
- **Persistent caching**: LLM standardization results cached in `config/cache/*.json`
- **Editable caches**: Manually override cached values to fix misclassifications
- **Profile inheritance**: Profiles can inherit from `.env` with overrides

## Requirements

- Python 3.8+
- [Poppler](https://poppler.freedesktop.org/) for PDF processing
- [OpenRouter API key](https://openrouter.ai) for Vision LLM access

## License

MIT
