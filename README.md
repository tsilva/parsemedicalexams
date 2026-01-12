# Medical Exams Parser

Extract and summarize medical exam reports (X-rays, MRIs, ultrasounds, endoscopies, etc.) from PDFs using Vision AI.

## What it does

1. **Extracts** complete text from medical exam PDFs using Vision LLMs
2. **Classifies** exams by type (imaging, ultrasound, endoscopy, other)
3. **Summarizes** reports aggressively — keeping only findings, impressions, and recommendations
4. **Outputs** structured CSV/Excel files ready for analysis or integration

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

Copy the example environment file and add your settings:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENROUTER_API_KEY=your_api_key_here
INPUT_PATH=/path/to/your/exam/pdfs
OUTPUT_PATH=/path/to/output
EXTRACT_MODEL_ID=google/gemini-2.5-flash
SUMMARIZE_MODEL_ID=google/gemini-2.5-flash
SELF_CONSISTENCY_MODEL_ID=google/gemini-2.5-flash
N_EXTRACTIONS=3
```

### 3. Run

```bash
python main.py
```

## Using Profiles

Profiles let you save different input/output configurations:

```bash
# Create a profile
cp profiles/_template.json profiles/myprofile.json
# Edit with your paths

# Run with profile
python main.py --profile myprofile

# List available profiles
python main.py --list-profiles
```

## Output

The parser generates:

```
output/
├── all.csv          # All exams merged (main output)
├── all.xlsx         # Excel version
└── {document}/      # Per-document folder
    ├── {document}.csv
    ├── {document}.001.jpg   # Preprocessed page images
    └── {document}.001.json  # Raw extraction data
```

### CSV Columns

| Column | Description |
|--------|-------------|
| `date` | Exam date (YYYY-MM-DD) |
| `exam_type` | Category: imaging, ultrasound, endoscopy, other |
| `exam_name_raw` | Exam name exactly as in document |
| `exam_name_standardized` | Cleaned English name |
| `transcription` | Full text from the report |
| `summary` | Only findings, impressions, recommendations |
| `source_file` | Source PDF filename |
| `page_number` | Page number in PDF |

## Integration with health-logs-parser

Point health-logs-parser to use this output:

```bash
LABS_PARSER_OUTPUT_PATH=/path/to/medical-exams-parser/output
```

## Requirements

- Python 3.8+
- [Poppler](https://poppler.freedesktop.org/) (for PDF processing)
  - macOS: `brew install poppler`
  - Ubuntu: `apt-get install poppler-utils`
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))

## How it works

1. **PDF → Images**: Converts each page to grayscale, resizes, enhances contrast
2. **Vision LLM**: Extracts exam data using function calling (runs N times for reliability)
3. **Self-consistency**: If extractions differ, LLM votes on the best result
4. **Standardization**: LLM classifies exam type and standardizes the name
5. **Summarization**: LLM aggressively filters to clinical findings only
6. **Caching**: Results cached in `config/cache/` to avoid redundant API calls

## License

MIT
