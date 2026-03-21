<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/parsemedicalexams/main/logo.png" alt="parsemedicalexams" width="512"/>

  # parsemedicalexams

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
  [![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-purple.svg)](https://openrouter.ai)

  **🏥 Extract and summarize medical exam reports from PDFs using Vision AI 📄**

  [Features](#features) · [Quick Start](#quick-start) · [Configuration](#configuration) · [Output Format](#output-format)
</div>

---

## Features

[![CI](https://github.com/tsilva/parsemedicalexams/actions/workflows/release.yml/badge.svg)](https://github.com/tsilva/parsemedicalexams/actions/workflows/release.yml)

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
uv tool install . --editable
```

> **Requires [Poppler](https://poppler.freedesktop.org/)** for PDF processing:
> - macOS: `brew install poppler`
> - Ubuntu: `apt-get install poppler-utils`

### 2. Configure

```bash
mkdir -p ~/.config/parsemedicalexams
cp profiles/template.yaml.example ~/.config/parsemedicalexams/tsilva.yaml
```

Edit `~/.config/parsemedicalexams/tsilva.yaml` with your settings:

```yaml
name: tsilva
openrouter_api_key: your_api_key_here
input_path: /path/to/your/exam/pdfs
output_path: /path/to/output
model: google/gemini-2.5-flash
workers: 4
```

### 3. Run

```bash
medicalexamsparser --profile tsilva
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

### Profiles

All runtime configuration now lives in profile files stored in `~/.config/parsemedicalexams/`.

```bash
# Create a profile from template
mkdir -p ~/.config/parsemedicalexams
cp profiles/template.yaml.example ~/.config/parsemedicalexams/myprofile.yaml

# Run with profile
medicalexamsparser --profile myprofile

# List available profiles
medicalexamsparser --list-profiles
```

Profiles can be YAML or JSON. A flat YAML profile looks like this:

```yaml
name: myprofile
openrouter_api_key: your_api_key_here
input_path: /path/to/input
output_path: /path/to/output
input_file_regex: ".*\\.pdf"
model: google/gemini-2.5-flash
workers: 4
n_extractions: 3
validation_model_id: anthropic/claude-haiku-4.5
summarize_max_input_tokens: 100000
full_name: "Patient Name"
birth_date: "1980-01-31"
locale: "pt-PT"
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
medicalexamsparser --profile tsilva

# Regenerate summaries from existing transcription files
medicalexamsparser --profile tsilva --regenerate

# Force reprocess all documents
medicalexamsparser --profile tsilva --reprocess-all

# Reprocess a specific document
medicalexamsparser -p tsilva -d exam_2024.pdf

# Reprocess a specific page within a document
medicalexamsparser -p tsilva -d exam_2024.pdf --page 2
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
├── config.py            # ExtractionConfig/ProfileConfig (global profile config)
├── utils.py             # Image preprocessing, logging, JSON utilities
├── prompts/             # Externalized LLM prompts as markdown
└── profiles/            # Example profile template
```

### Key Design Patterns

- **Two-phase processing**: Classify document first, then transcribe all pages
- **Two-column naming**: `*_raw` (exact from document) + `*_standardized` (LLM-mapped)
- **Persistent caching**: LLM standardization results cached in `~/.config/parsemedicalexams/cache/*.json`
- **Editable caches**: Manually override cached values to fix misclassifications
- **Self-contained profiles**: Each profile carries API, model, path, and patient context configuration
- **Frequency-based date voting**: Handles multi-era documents (e.g., 2024 cover letter + 1997 records)

## Requirements

- Python 3.8+
- [Poppler](https://poppler.freedesktop.org/) for PDF processing
- [OpenRouter API key](https://openrouter.ai) for Vision LLM access

## License

MIT
