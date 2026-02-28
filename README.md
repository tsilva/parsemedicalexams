<div align="center">
  <img src="logo.png" alt="parsemedicalexams" width="512"/>

  # parsemedicalexams

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
  [![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-purple.svg)](https://openrouter.ai)

  **🏥 Extract and summarize medical exam reports from PDFs using Vision AI 📄**

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
- **Multi-era document handling** — Frequency-based date voting correctly handles documents spanning multiple time periods

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
2. **Document classification** — Determines if the document is a medical exam before processing
3. **Vision LLM transcription** — Transcribes each page verbatim using function calling (runs N times for reliability)
4. **Self-consistency voting** — If transcriptions differ, LLM votes on the best result
5. **Standardization** — Classifies exam type and standardizes the name via LLM with caching
6. **Summarization** — Generates document-level clinical summaries preserving all findings

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key ([get one here](https://openrouter.ai)) | *required* |
| `INPUT_PATH` | Directory containing exam PDFs | *required* |
| `OUTPUT_PATH` | Where to write output files | *required* |
| `EXTRACT_MODEL_ID` | Vision model for extraction | `google/gemini-2.5-flash` |
| `SUMMARIZE_MODEL_ID` | Model for summarization | `google/gemini-2.5-flash` |
| `SELF_CONSISTENCY_MODEL_ID` | Model for voting | `google/gemini-2.5-flash` |
| `N_EXTRACTIONS` | Number of extraction runs for voting | `3` |
| `MAX_WORKERS` | Parallel workers for PDF processing | `1` |
| `INPUT_FILE_REGEX` | Regex pattern for input files | `.*\.pdf` |

### Using Profiles

Profiles let you save different input/output configurations for different use cases:

```bash
# Create a profile from template
cp profiles/_template.yaml profiles/myprofile.yaml

# Run with profile
python main.py --profile myprofile

# List available profiles
python main.py --list-profiles
```

Profile files (YAML or JSON) support path overrides and model configuration:

```yaml
name: myprofile
input_path: /path/to/input
output_path: /path/to/output
input_file_regex: ".*\\.pdf"
model: google/gemini-2.5-flash  # Optional override
workers: 1                       # Optional override
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--profile`, `-p` | Profile name to use |
| `--list-profiles` | List available profiles |
| `--regenerate` | Regenerate markdown files from existing JSON data |
| `--reprocess-all` | Force reprocess all documents |
| `--document`, `-d` | Process only this document (filename or stem) |
| `--page` | Process only this page number (requires `--document`) |
| `--model`, `-m` | Override model ID |
| `--workers`, `-w` | Override worker count |
| `--pattern` | Override input file regex |

**Examples:**

```bash
# Process all new PDFs
python main.py --profile tsilva

# Regenerate summaries from existing transcription files
python main.py --profile tsilva --regenerate

# Force reprocess all documents
python main.py --profile tsilva --reprocess-all

# Reprocess a specific document
python main.py -p tsilva -d exam_2024.pdf

# Reprocess a specific page within a document
python main.py -p tsilva -d exam_2024.pdf --page 2
```

## Output Format

The parser generates structured markdown files with YAML frontmatter:

```
output/
├── {document}/
│   ├── {document}.pdf            # Source PDF copy
│   ├── {document}.001.jpg        # Page 1 image
│   ├── {document}.001.md         # Page 1 transcription + metadata
│   ├── {document}.002.jpg        # Page 2 image
│   ├── {document}.002.md         # Page 2 transcription + metadata
│   └── {document}.summary.md     # Document-level summary
```

### Transcription File Structure

Each `.md` file contains YAML frontmatter with metadata followed by the verbatim transcription:

```yaml
---
date: 2024-01-15
title: "Chest X-Ray PA and Lateral"
category: imaging
exam_name_raw: "RX TORAX PA Y LAT"
doctor: "Dr. Smith"
facility: "Hospital Central"
confidence: 0.95
page: 1
source: exam_2024.pdf
---

[Full verbatim transcription text here...]
```

### Metadata Fields

| Field | Description |
|-------|-------------|
| `date` | Exam date (YYYY-MM-DD) |
| `title` | Standardized exam name (English) |
| `category` | Exam type: `imaging`, `ultrasound`, `endoscopy`, `other` |
| `exam_name_raw` | Exam name exactly as written in document |
| `doctor` | Physician name (if found) |
| `facility` | Healthcare facility name |
| `department` | Department within facility |
| `confidence` | Self-consistency confidence score (0.0-1.0) |
| `page` | Page number in source PDF |
| `source` | Source PDF filename |

## Architecture

```
parsemedicalexams/
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

- **Two-phase processing**: Classify document first, then transcribe all pages
- **Two-column naming**: `*_raw` (exact from document) + `*_standardized` (LLM-mapped)
- **Persistent caching**: LLM standardization results cached in `config/cache/*.json`
- **Editable caches**: Manually override cached values to fix misclassifications
- **Profile inheritance**: Profiles can inherit from `.env` with overrides
- **Frequency-based date voting**: Handles multi-era documents (e.g., 2024 cover letter + 1997 records)

## Requirements

- Python 3.8+
- [Poppler](https://poppler.freedesktop.org/) for PDF processing
- [OpenRouter API key](https://openrouter.ai) for Vision LLM access

## License

MIT
