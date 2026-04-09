# Pipeline Technical Reference

## Overview

The pipeline converts PDF medical exam documents into structured markdown files. For each PDF it:
1. converts pages to preprocessed images
2. classifies the document as a medical exam or not
3. transcribes each page verbatim via vision LLM
4. corrects the exam date by frequency voting across pages
5. standardizes exam names to a canonical type and title
6. saves per-page `.md` files with YAML frontmatter
7. generates a comprehensive `.summary.md` via incremental chunked summarization

Entry point: `parsemedicalexams.cli:main() → parsemedicalexams.pipeline.run_profile() → process_single_pdf()`.

---

## Configuration

### Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | Base URL (default: `https://openrouter.ai/api/v1`) |
| `EXTRACT_MODEL_ID` | Vision model for classification and transcription |
| `SELF_CONSISTENCY_MODEL_ID` | Model used when `N_EXTRACTIONS > 1` |
| `SUMMARIZE_MODEL_ID` | Model for summarization and standardization |
| `VALIDATION_MODEL_ID` | Model for refusal detection (default: `anthropic/claude-haiku-4.5`) |
| `N_EXTRACTIONS` | Extractions per page (default: `1`; `> 1` enables self-consistency) |
| `MAX_WORKERS` | Parallel page workers (default: `1`) |
| `SUMMARIZE_MAX_INPUT_TOKENS` | Token budget for summarization input (default: `100000`) |
| `INPUT_PATH` | Source PDF directory |
| `OUTPUT_PATH` | Output directory |
| `INPUT_FILE_REGEX` | Filename filter regex (default: `.*\.pdf`) |

### Profile Files (`profiles/*.yaml` or `profiles/*.json`)

Profiles override `.env` paths and optionally model/workers. Fields: `name`, `input_path`, `output_path`, `input_file_regex`, `model`, `workers`, `full_name`, `birth_date`, `locale`.

**Override priority:** CLI args > profile > `.env`

---

## Pipeline Steps

### Step 1: PDF to Images

**Location:** `pipeline.py → process_single_pdf()`

```
pdf2image.convert_from_path(str(pdf_path))       # PIL Image per page
  → preprocess_page_image(image)                  # utils.py
      .convert("L")                               # grayscale
      .resize(...)                                # max 1000px long side, LANCZOS
      ImageEnhance.Contrast(...).enhance(2.0)     # 2× contrast
  → saved as {doc_stem}.{page_num:03d}.jpg (quality=80)
```

Images are stored in `{output_path}/{doc_stem}/` and reused on subsequent runs. Use `--document` to force regeneration.

---

### Step 2: Document Classification

**Location:** `extraction.py → classify_document()`

All page images are sent together in a single vision LLM call with forced function calling:

```python
client.chat.completions.create(
    model=extract_model_id,
    messages=[system (classification_system.md), user (classification_user.md + all images)],
    temperature=0.1,
    max_tokens=1024,
    tools=CLASSIFICATION_TOOLS,
    tool_choice={"type": "function", "function": {"name": "classify_document"}},
)
```

The response is parsed into `DocumentClassification`:

| Field | Description |
|---|---|
| `is_exam` | Whether the document is a medical exam |
| `exam_name_raw` | Document title as written |
| `exam_date` | Date in YYYY-MM-DD |
| `facility_name` | Healthcare facility |
| `physician_name` | Signing physician |
| `department` | Department/service |

**Fail-open:** errors and missing tool calls default to `is_exam=True`. Non-exams return `"skipped"` and are not processed further. The CLI currently rejects `--page` before orchestration begins.

**Prompts:** `prompts/classification_system.md` (formatted with `{patient_context}`), `prompts/classification_user.md`

---

### Step 3: Per-Page Transcription

**Location:** `pipeline.py → process_page()` (inner function), called in parallel via `ThreadPoolExecutor(max_workers)`

Two modes depending on `N_EXTRACTIONS`:

#### Mode A: Single extraction with retry (`N_EXTRACTIONS = 1`)

`extraction.py → transcribe_with_retry()`

Tries up to 4 prompt variants in order, stopping at the first valid (non-refusal) result:

1. `transcription_system` (primary)
2. `transcription_system_alt1`
3. `transcription_system_alt2`
4. `transcription_system_alt3`

Each attempt calls `transcribe_page()` then `validate_transcription()`.

**`transcribe_page()`** LLM call:
```python
client.chat.completions.create(
    model=extract_model_id,
    messages=[system (transcription_system*.md), user (transcription_user.md + image)],
    temperature=0.1,
    max_tokens=16384,
)
```

**`validate_transcription()`** refusal check:
```python
client.chat.completions.create(
    model=validation_model_id,
    messages=[user (inline refusal check prompt)],
    temperature=0,
    max_tokens=10,
)
# Returns "yes"/"no" — retries on "yes"
```

#### Mode B: Self-consistency (`N_EXTRACTIONS > 1`)

`extraction.py → self_consistency(transcribe_page, self_consistency_model_id, n, ...)`

1. Runs `transcribe_page()` `n` times in parallel with `temperature=0.5`
2. If all results identical → returns first result
3. Otherwise → `vote_on_best_result()` calls LLM with `voting_system.md` at `temperature=0.1`
4. `score_transcription_confidence()` compares merged result to originals via `confidence_scoring_system.md` at `temperature=0.1`; returns float 0.0–1.0

---

### Step 4: Date Correction

**Location:** `pipeline.py → select_most_frequent_date()`

After all pages are transcribed, a frequency vote selects the document date:

```python
for exam in exams:
    page_dates = extract_dates_from_text(exam["transcription"])  # utils.py
    # Handles: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
    # Excludes birth_date from profile if set
    page_date = min(page_dates)   # earliest date per page → likely exam date
    all_dates.append(page_date)

most_common_date = Counter(all_dates).most_common(1)[0][0]
```

**Filename tiebreaker:** if `most_common_date` conflicts with a date embedded in the filename and the filename date appears on at least one page, the filename date wins (handles DD/MM vs MM/DD software timestamp ambiguity).

`extract_date_from_filename()` recognises patterns: `YYYY-MM-DD`, `YYYY_MM_DD`, `YYYYMMDD`.

The corrected date is written back to all exam dicts.

---

### Step 5: Exam Type Standardization

**Location:** `standardization.py → standardize_exam_types()`

Maps raw exam names to `(exam_type, standardized_name)` with a persistent cache:

```
Cache file: config/cache/exam_type_standardization.json
Cache key:  name.lower().strip()
```

LLM is called only for uncached names:
```python
client.chat.completions.create(
    model=extract_model_id,
    messages=[system (standardization_system.md), user (standardization_user.md + names JSON)],
    temperature=0.1,
    max_tokens=4000,
)
# Returns JSON: { raw_name: { exam_type, standardized_name } }
```

Cache is saved after each batch. Failures fall back to `exam_type="other"` with the raw name as-is. The cache is user-editable for manual overrides.

**Exam types:** `imaging`, `ultrasound`, `endoscopy`, `other`

---

### Step 6: Save Transcriptions

**Location:** `document_io.py → save_transcription_file()`

Output file: `{output_path}/{doc_stem}/{doc_stem}.{page_num:03d}.md`

YAML frontmatter is built from `_FRONTMATTER_MAP`:

| Exam dict key | Frontmatter key |
|---|---|
| `exam_date` | `exam_date` |
| `exam_name_raw` | `exam_name_raw` |
| `exam_name_standardized` | `title` |
| `exam_type` | `category` |
| `physician_name` | `doctor` |
| `facility_name` | `facility` |
| `department` | `department` |

Additional frontmatter fields: `page`, `source`, `prompt_variant`, `confidence` (self-consistency only), `retry_attempts` (if > 1).

File body: verbatim transcription text.

---

### Step 7: Document Summarization

**Location:** `summarization.py → summarize_document()`

Generates a single clinical summary across all pages using incremental chunked summarization:

```
token_budget = max_input_tokens - (len(system_prompt) // 4 + 200)
chunk_budget = token_budget - 2000   # reserve for running summary overhead
```

Chunks are built greedily from `_build_exam_list()` + `_build_transcriptions()` text, estimated at 4 chars/token.

For each chunk:
- **Chunk 1:** `summarization_user.md` template (exam_count, exam_list, transcriptions)
- **Chunk 2+:** `summarization_incremental_user.md` template (existing_summary + new chunk)

Each chunk LLM call:
```python
client.chat.completions.create(
    model=summarize_model_id,
    messages=[system (summarization_system.md), user (chunk prompt)],
    temperature=0.1,
    max_tokens=4000,
)
```

Output file: `{output_path}/{doc_stem}/{doc_stem}.summary.md` with YAML frontmatter from the first exam.

---

## Output Structure

```
{output_path}/
  {doc_stem}/
    {doc_stem}.pdf              # source PDF copy
    {doc_stem}.001.jpg          # preprocessed page images
    {doc_stem}.002.jpg
    ...
    {doc_stem}.001.md           # per-page transcription with YAML frontmatter
    {doc_stem}.002.md
    ...
    {doc_stem}.summary.md       # document-level clinical summary
  logs/
    info.log
    error.log
```

**Frontmatter schema** (per-page `.md`):
```yaml
---
exam_date: "2024-03-15"
exam_name_raw: "RADIOGRAFIA TÓRAX PA"
title: "Chest X-Ray"
category: imaging
doctor: "Dr. Ana Silva"
facility: "Hospital São João"
department: "Radiologia"
page: 1
source: "exam_2024.pdf"
prompt_variant: transcription_system
confidence: 0.95        # self-consistency only
retry_attempts: 2       # if > 1
---
```

---

## LLM Calls Reference

| Step | Function | Model config key | Temp | Max tokens | Purpose |
|---|---|---|---|---|---|
| Classification | `classify_document()` | `EXTRACT_MODEL_ID` | 0.1 | 1024 | Is this a medical exam? Extract metadata. |
| Transcription | `transcribe_page()` | `EXTRACT_MODEL_ID` | 0.1 | 16384 | Verbatim OCR of page image |
| Refusal check | `validate_transcription()` | `VALIDATION_MODEL_ID` | 0 | 10 | Detect if transcription is a refusal |
| SC voting | `vote_on_best_result()` | `SELF_CONSISTENCY_MODEL_ID` | 0.1 | — | Pick best from N transcriptions |
| SC confidence | `score_transcription_confidence()` | `SELF_CONSISTENCY_MODEL_ID` | 0.1 | — | Score 0–1 semantic agreement |
| Standardization | `standardize_exam_types()` | `EXTRACT_MODEL_ID` | 0.1 | 4000 | Map raw name → (type, standard name) |
| Summarization | `_llm_summarize()` | `SUMMARIZE_MODEL_ID` | 0.1 | 4000 | Clinical summary per chunk |

---

## Prompts Reference

All 14 prompt files live in `prompts/`:

| File | Used by |
|---|---|
| `classification_system.md` | `classify_document()` — system prompt (formatted with `{patient_context}`) |
| `classification_user.md` | `classify_document()` — user prompt |
| `transcription_system.md` | `transcribe_page()` — primary system prompt |
| `transcription_system_alt1.md` | `transcribe_with_retry()` — retry variant 1 |
| `transcription_system_alt2.md` | `transcribe_with_retry()` — retry variant 2 |
| `transcription_system_alt3.md` | `transcribe_with_retry()` — retry variant 3 |
| `transcription_user.md` | `transcribe_page()` — user prompt |
| `voting_system.md` | `vote_on_best_result()` — self-consistency voting |
| `confidence_scoring_system.md` | `score_transcription_confidence()` |
| `standardization_system.md` | `standardize_exam_types()` — system prompt |
| `standardization_user.md` | `standardize_exam_types()` — user prompt (formatted with `{exam_names}`) |
| `summarization_system.md` | `_llm_summarize()` — system prompt |
| `summarization_user.md` | `_incremental_summarize()` — first chunk user prompt |
| `summarization_incremental_user.md` | `_incremental_summarize()` — subsequent chunk user prompt |
