# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical Exams Parser extracts and summarizes medical exam reports (X-rays, MRIs, ultrasounds, endoscopies, etc.) from PDF documents using Vision LLMs via OpenRouter. It outputs individual markdown files per exam with YAML frontmatter for metadata.

## Commands

```bash
# Verify all modules import cleanly (no test suite exists — this is the baseline check)
python3 -c "from parsemedicalexams import *; print('All imports OK')"

# Install as an editable CLI tool
uv tool install . --editable

# Run the pipeline
medicalexamsparser --profile tiago

# Run with a specific profile
medicalexamsparser --profile tiago

# List available profiles
medicalexamsparser --list-profiles

# Regenerate .md files from existing JSON extraction data
medicalexamsparser --profile tiago --regenerate

# Reprocess a specific document (by filename or stem)
medicalexamsparser -p tiago -d exam_2024.pdf

# Reprocess a specific page within a document
medicalexamsparser -p tiago -d exam_2024.pdf --page 2
```

## Architecture

### Pipeline Flow
1. **PDF → Images**: Convert PDF pages to preprocessed JPG images (grayscale, resize, contrast enhancement)
2. **Vision LLM Extraction**: Extract exam data using function calling with self-consistency voting (N extractions, LLM votes on best)
3. **Standardization**: Classify exam types (imaging/ultrasound/endoscopy/other) and standardize names via LLM with JSON cache
4. **Summarization**: Document-level clinical summary preserving all findings, impressions, and recommendations
5. **Output**: Per-page transcription files + one comprehensive summary per document

### Key Modules
- **main.py**: Pipeline orchestration, PDF processing loop, CLI argument handling
- **extraction.py**: Pydantic models (`MedicalExam`, `MedicalExamReport`), Vision LLM extraction with function calling, self-consistency voting
- **standardization.py**: Exam type classification using LLM with persistent JSON cache in `~/.config/parsemedicalexams/cache/`
- **summarization.py**: Document-level clinical summarization using LLM (preserves all clinical details for medical records)
- **config.py**: `ExtractionConfig` + `ProfileConfig` loaded from `~/.config/parsemedicalexams/`
- **utils.py**: Image preprocessing, logging setup, JSON parsing utilities

### Configuration
- `~/.config/parsemedicalexams/*.yaml|json`: User-specific self-contained profiles
- `~/.config/parsemedicalexams/cache/*.json`: LLM response caches (user-editable for overrides)
- `prompts/*.md`: All LLM prompts externalized as markdown files

### Output Format
Each page produces one file:
- **`{doc_stem}.{page}.md`**: Raw transcription with YAML frontmatter (metadata embedded)

Each document produces one summary file:
- **`{doc_stem}.summary.md`**: Comprehensive clinical summary of all exams in the document (preserves all findings for medical records)

## Development Notes

- **No test suite** — import verification above is the only automated baseline check
- **Worktrees**: `.worktrees/` is gitignored and ready to use; no setup needed
- **Sandbox mode**: `uv tool install . --editable` and `medicalexamsparser --list-profiles` may fail in sandboxed environments; use the import check instead
- **Dependencies already installed globally** — no reinstall needed when creating new worktrees
- **Worktree cleanup order**: `git worktree remove` must run *before* `git branch -d` (git blocks branch deletion while a worktree has it checked out)

## Key Helpers (main.py)

- `_FRONTMATTER_MAP` — canonical dict mapping exam dict keys → YAML frontmatter keys; used by `build_exam_frontmatter()` and `frontmatter_to_exam()`
- `_transcription_files(doc_dir, doc_stem)` — returns all `.md` files in a doc dir excluding `.summary.md`
- `extract_dates_from_text(text)` — in `utils.py`; extracts YYYY-MM-DD dates from DD/MM/YYYY, DD-MM-YYYY, and ISO formats
- `self_consistency(fn, model_id, n, *args, client=None, **kwargs)` — pass the existing `OpenAI` client as `client=`; do not pass `base_url`/`api_key`

## Patterns from labs-parser

This project follows labs-parser conventions:
- OpenRouter API for multi-model LLM access
- Self-consistency voting for extraction reliability
- Two-column naming pattern: `*_raw` (exact from document), `*_standardized` (LLM-mapped)
- Persistent JSON caches for LLM standardization (avoids repeated API calls, user-editable)
- Profile system for user-specific input/output paths
