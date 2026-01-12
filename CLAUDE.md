# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical Exams Parser extracts and summarizes medical exam reports (X-rays, MRIs, ultrasounds, endoscopies, etc.) from PDF documents using Vision LLMs via OpenRouter. It outputs individual markdown files per exam with YAML frontmatter for metadata.

## Commands

```bash
# Install dependencies
pip install -e .

# Run the pipeline (uses .env configuration)
python main.py

# Run with a specific profile
python main.py --profile tsilva

# List available profiles
python main.py --list-profiles

# Regenerate .md files from existing JSON extraction data
python main.py --profile tsilva --regenerate
```

## Architecture

### Pipeline Flow
1. **PDF → Images**: Convert PDF pages to preprocessed JPG images (grayscale, resize, contrast enhancement)
2. **Vision LLM Extraction**: Extract exam data using function calling with self-consistency voting (N extractions, LLM votes on best)
3. **Standardization**: Classify exam types (imaging/ultrasound/endoscopy/other) and standardize names via LLM with JSON cache
4. **Summarization**: Aggressive filtering to keep only findings, impressions, and recommendations
5. **Output**: Individual markdown files per exam with YAML frontmatter

### Key Modules
- **main.py**: Pipeline orchestration, PDF processing loop, CLI argument handling
- **extraction.py**: Pydantic models (`MedicalExam`, `MedicalExamReport`), Vision LLM extraction with function calling, self-consistency voting
- **standardization.py**: Exam type classification using LLM with persistent JSON cache in `config/cache/`
- **summarization.py**: Aggressive text summarization using LLM with hash-based caching
- **config.py**: `ExtractionConfig` (from .env) and `ProfileConfig` (from profiles/*.json)
- **utils.py**: Image preprocessing, logging setup, JSON parsing utilities

### Configuration
- `.env`: API keys, model IDs, default input/output paths
- `profiles/*.json`: User-specific path overrides (inherit from .env with `inherit_from_env: true`)
- `config/cache/*.json`: LLM response caches (user-editable for overrides)
- `prompts/*.md`: All LLM prompts externalized as markdown files

### Output Format
Each page produces three files:
- **`{doc_stem}.{page}.md`**: Raw transcription verbatim
- **`{doc_stem}.{page}.summary.md`**: Summary only
- **`{doc_stem}.{page}.json`**: Metadata only (no transcription)

## Patterns from labs-parser

This project follows labs-parser conventions:
- OpenRouter API for multi-model LLM access
- Self-consistency voting for extraction reliability
- Two-column naming pattern: `*_raw` (exact from document), `*_standardized` (LLM-mapped)
- Persistent JSON caches for LLM standardization (avoids repeated API calls, user-editable)
- Profile system for user-specific input/output paths
