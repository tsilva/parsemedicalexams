---
name: parsemedicalexams_extraction-quality-audit
description: Comprehensive quality assessment for document extraction pipelines. Use when (1) After implementing extraction pipeline, (2) Before production deployment, (3) Periodic quality audits, (4) Investigating suspected quality issues, (5) Validating pipeline changes, (6) Benchmarking extraction accuracy across document types/eras. Performs stratified sampling, systematic verification, pattern analysis, and generates detailed quality reports.
---

# Extraction Quality Audit

## Overview

This skill performs comprehensive quality assessment of document extraction pipelines through stratified sampling, systematic verification, pattern analysis, and detailed quality reporting.

**Use this skill when:**
- After implementing or modifying an extraction pipeline
- Before production deployment of document processing systems
- Conducting periodic quality audits (quarterly/annual)
- Investigating suspected quality issues or accuracy concerns
- Validating prompt modifications or model changes
- Benchmarking extraction accuracy across document types or time periods
- Evaluating LLM-based document processing systems

**Prerequisites:**
- Extraction pipeline outputs exist (structured files with metadata)
- Source documents available for comparison
- Codex has vision access for document comparison
- Output files contain metadata (YAML frontmatter, JSON, or similar)

**Expected Time:**
- Phase 0 (Sentinel scan): 1-3 minutes
- Phase 1 (Inventory): 5-10 minutes
- Phase 2 (Sampling): 2-5 minutes
- Phase 3 (Verification): 2-4 hours for 25-30 documents
- Phase 4 (Analysis): 15-30 minutes
- Phase 5 (Report): 5-10 minutes

## Quick Start

Run the commands below from the repository root. Helper paths are relative to
that root; the helpers remain beside this skill.

Use a private task-artifact directory outside this source checkout and original
document directories by default. The old root-level JSON/Markdown artifact names
are not Git-ignored here. If the user selects an ignored work folder instead,
verify its effective Git exclusion before saving private metadata. Replace
`/absolute/private-audit` below with the chosen existing artifact directory;
keep inventories, results, reports and per-run configuration there.

```bash
# 0. Run sentinel scan for hard failures and non-transcription output
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/sentinel_scan.py \
  --output-dir /path/to/extraction/outputs \
  --json-output /absolute/private-audit/sentinel_failures.json

# 1. Build the inventory using the selected paths
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/generate_inventory.py \
  --input-dir /path/to/source/documents \
  --output-dir /path/to/extraction/outputs \
  --profile-name your_profile \
  --output /absolute/private-audit/inventory.json

# 2. Run stratified sampling
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/stratified_sampling.py \
  --inventory /absolute/private-audit/inventory.json \
  --sentinel-report /absolute/private-audit/sentinel_failures.json \
  --target-size 30 \
  --output /absolute/private-audit/sample.json

# 3. Perform systematic verification (interactive with Codex)
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/verification_framework.py \
  --sample /absolute/private-audit/sample.json \
  --results /absolute/private-audit/verification_results.json

# 4. Generate comprehensive report
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/report_generator.py \
  --inventory /absolute/private-audit/inventory.json \
  --sample /absolute/private-audit/sample.json \
  --results /absolute/private-audit/verification_results.json \
  --sentinel-report /absolute/private-audit/sentinel_failures.json \
  --output /absolute/private-audit/QUALITY_REPORT.md
```

## Phase 0: Sentinel Failure Scan

**Purpose:** Catch systemic extraction failures before spending tokens on sampling and page-by-page review.

**What it does:**
- Scans extraction outputs for known hard-failure strings and non-document narration
- Counts affected files and documents
- Surfaces examples immediately so they can be promoted into the audit sample
- Reduces wasted effort on documents already known to be broken

**Script:** `scripts/sentinel_scan.py`

**Usage:**
```bash
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/sentinel_scan.py \
  --output-dir /path/to/extraction/outputs \
  --json-output /absolute/private-audit/sentinel_failures.json
```

**Default sentinel patterns:**
- API / pipeline failure strings such as `Request too large. Try with a smaller file.`
- Model narration such as `This image shows`, `No readable text is visible on this page`, `The following text elements are partially visible`, or `The remaining technical parameters`
- Placeholder institution text such as `Unknown Institution`

**Triage rule:**
- If sentinel failures exist, report their counts first.
- Promote every affected document into the working sample before random fill.
- Treat literal failure strings in page outputs or summaries as confirmed extraction failures, not low-confidence warnings.

Pass the saved sentinel JSON to the sampler and report generator. The sampler
matches full document directories against the explicit inventory and includes
every matched failure before priority sampling and random fill. Repeated hits
are deduplicated; missing or ambiguous matches stop rather than silently losing
known failures. Required cases can exceed the nominal target. Document identity
is `(profile, doc_stem)` throughout selection and verification; never treat one
profile's same-named file as verified for another. Unbound historical results
stay stored but do not complete a profile's review.

## Phase 1: Document Inventory

**Purpose:** Create a comprehensive catalog of all processed documents with metadata for sampling.

**What it does:**
- Scans output directory for all extraction results
- Extracts metadata from each document (dates, categories, confidence scores, page counts)
- Classifies documents by complexity, era, and characteristics
- Tags documents with priority markers for sampling
- Outputs structured JSON inventory

**Script:** `scripts/generate_inventory.py`

**Usage:**
```bash
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/generate_inventory.py \
  --input-dir /path/to/source/docs \
  --output-dir /path/to/extraction/outputs \
  --profile-name profile_name \
  --output /absolute/private-audit/inventory.json
```

**Configuration Options:**
- `--input-dir`: Directory containing source documents (PDFs, images, etc.)
- `--output-dir`: Directory containing extraction outputs (markdown, JSON, etc.)
- `--profile-name`: Profile or category name for this document set
- `--output`: Output filename (default: inventory.json)
- `--metadata-format`: Format of metadata (yaml, json, frontmatter)

**Output Structure:**
```json
[
  {
    "profile": "profile_name",
    "doc_stem": "document_name",
    "doc_dir": "/path/to/output/dir",
    "page_count": 3,
    "has_summary": true,
    "confidence_scores": [0.95, 0.89, 0.92],
    "min_confidence": 0.89,
    "max_confidence": 0.95,
    "avg_confidence": 0.92,
    "date": "2024-01-15",
    "category": "imaging",
    "era": "2020s",
    "tags": ["CAT_IMAGING", "MULTI_PAGE"]
  }
]
```

**Default Priority Tags:**
- `LOW_CONF`: Minimum confidence < 0.7
- `OLD_DOC`: Documents from 1990s-2000s (configurable)
- `COMPLEX`: 10+ pages
- `MULTI_PAGE`: 3-9 pages
- `CAT_[CATEGORY]`: Document category tags

**Customization:** Tag definitions are maintained implementation code. Reuse them for an audit; change them only as scoped implementation work with tests. Keep per-audit coverage and priority settings in an external sampling config.

**Reference:** See `references/sampling_methodology.md` for detailed tag design rationale.

## Phase 2: Stratified Sampling

**Purpose:** Select a priority-based review sample with mandatory known failures, high-risk coverage, diversity checks, and random fill.

**What it does:**
- Applies priority-based stratified sampling across multiple dimensions
- Ensures coverage of high-risk categories (low confidence, complex, old)
- Balances representation across document types, eras, and categories
- Provides sampling justification for each selected document
- Outputs sample with selection reasons

**Script:** `scripts/stratified_sampling.py`

**Usage:**
```bash
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/stratified_sampling.py \
  --inventory /absolute/private-audit/inventory.json \
  --sentinel-report /absolute/private-audit/sentinel_failures.json \
  --target-size 30 \
  --output /absolute/private-audit/sample.json
```

**Sampling Strategy:**

1. **100% coverage of critical cases:**
   - All LOW_CONF documents (highest priority)
   - All COMPLEX documents (10+ pages)

2. **High coverage of challenging cases:**
   - 60-80% of OLD_DOC (older documents)
   - 40-60% of MULTI_PAGE (3-9 pages)

3. **Balanced representation:**
   - Existing category quotas, recent-era selection and high-confidence baseline where slots remain
   - Inspect actual profile, category and era coverage; the helper does not guarantee proportional representation

4. **Random fill:**
   - Remaining slots filled randomly to reach target size

**Target Sample Size:**
- **Minimum review target:** 25-30 documents; this is a work-budget convention, not a statistical-significance guarantee
- **Large corpora (>500):** 50-75 documents
- **Very large (>1000):** 100+ documents

**Customizing Sampling:**

Create a per-audit `sampling_config.json` in the private artifact directory and pass `--config /absolute/private-audit/sampling_config.json` only when that existing file is selected:
```json
{
  "priority_tags": {
    "LOW_CONF": {"coverage": 1.0, "priority": 1000},
    "COMPLEX": {"coverage": 1.0, "priority": 100},
    "OLD_DOC": {"coverage": 0.7, "priority": 10},
    "MULTI_PAGE": {"coverage": 0.4, "priority": 1}
  },
  "random_seed": 42
}
```

**Reference:** See `references/sampling_methodology.md` for the implemented sampling policy and its evidence limits.

The sampler honors `--seed 0`. An explicitly supplied missing configuration is
an error. Unknown or invalid confidence stays unknown in lists and summaries;
it is never a high-confidence baseline. Input metadata is not mutated in memory.
Sample/report destinations cannot overwrite their explicit input metadata
through direct paths, symlinks or hardlinks. This output check is not a lock;
keep artifacts separate from original documents and preserve concurrency checks.
These helpers perform no extraction, provider request or original-document rewrite.

Reports calculate selected and recorded-review coverage from explicit
profile/document IDs and critical tags plus the supplied sentinel report.
Duplicate verification records require reconciliation; unbound or out-of-sample
records are excluded from aggregates and counted visibly without changing their
source files. Missing scores remain N/A, while a real zero remains a score.
The report describes the reviewed subset. It does not infer production readiness,
population confidence or a margin of error from sample size or average ratings.

## Phase 3: Systematic Verification

**Purpose:** Deep quality assessment of sampled documents through side-by-side comparison of source and extraction.

**What it does:**
- Presents verification checklist for each sampled document
- Loads source documents and extraction outputs
- Guides systematic comparison across quality dimensions
- Tracks verification progress
- Saves detailed scoring and findings incrementally

**Script:** `scripts/verification_framework.py`

**Usage:**
```bash
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/verification_framework.py \
  --sample /absolute/private-audit/sample.json \
  --results /absolute/private-audit/verification_results.json
```

**Verification Workflow:**

For each document in sample:

1. **Run sentinel check first:**
   - Search the document's outputs for known failure strings or narration
   - If found, record the failure type before deeper review
   - Do not spend time scoring obvious hard failures as if they were normal OCR noise

2. **Load Files:**
   - Read source document (PDF, image, etc.) using Codex's vision
   - Read all extraction output files (pages, summaries)

3. **Verify Metadata Accuracy (Score 0-10):**
   - Date correctness
   - Document title/name accuracy
   - Category classification
   - Facility, doctor, department names
   - Other structured fields

4. **Verify Transcription Quality (Score 0-10):**
   - **Completeness:** All text captured? (Score 0-10)
   - **Accuracy:** Text matches source? (Score 0-10)
   - **Layout:** Structure preserved? (Score 0-10)
   - Document specific issues

5. **Verify Domain-Specific Terminology:**
   - Medical: Anatomical terms, diagnostic phrases, measurements
   - Legal: Contract clauses, case citations, legal entities
   - Financial: Accounting terms, currency handling
   - Technical: Code snippets, API references, formulas
   - **Rating:** Excellent/Good/Fair/Poor

6. **Verify Summary Quality (Score 0-10):**
   - Completeness of key information
   - Accuracy of summary content
   - Appropriate level of detail
   - No hallucinations

7. **Multi-Page Coherence (for multi-page docs):**
   - Metadata consistency across pages
   - Page numbering accuracy
   - No duplicate content
   - Cross-page references preserved

8. **Low Confidence Investigation (for LOW_CONF docs):**
   - Identify root cause of low confidence
   - Assess if confidence is justified
   - Determine if extraction quality is still acceptable

**Verification Template:**

The JSON below is the inner `verification` object. Store it inside a result
record retaining the sampled document's explicit `profile` and `doc_stem`. Use
`null` for unrated dimensions, as the helper's new template does; reserve zero
for a deliberately observed score. Existing saved results are not rewritten.
```json
{
  "metadata_accuracy": {
    "score": 10,
    "issues": []
  },
  "transcription_quality": {
    "completeness_score": 10,
    "accuracy_score": 10,
    "layout_preservation_score": 9,
    "overall_score": 9.7,
    "issues": ["Minor spacing variation"]
  },
  "domain_terminology": {
    "rating": "Excellent",
    "issues": []
  },
  "summary_quality": {
    "score": 10,
    "issues": []
  },
  "overall_assessment": "Excellent",
  "notable_issues": [],
  "recommendations": [],
  "verified_date": "2026-01-21"
}
```

**Progress Tracking:**

The framework saves results incrementally, allowing you to:
- Pause and resume verification
- Track how many documents remain
- Review completed verifications

**Reference:** See `references/verification_checklist.md` for detailed verification steps and scoring rubrics.

## Phase 4: Pattern Analysis

**Purpose:** Aggregate findings across all verified documents to identify systematic issues and quality patterns.

**What it does:**
- Analyzes verification results to find patterns
- Identifies issues by document type, era, complexity, confidence level
- Calculates aggregate quality metrics
- Correlates confidence scores with actual quality
- Detects systematic vs. random errors

**Analysis Dimensions:**

1. **By Document Type/Category:**
   - Which categories have lower quality?
   - Are specific document types problematic?

2. **By Time Period/Era:**
   - Do older documents have quality issues?
   - Are recent documents better?

3. **By Complexity:**
   - Do multi-page documents have coherence issues?
   - Are simple documents more accurate?

4. **By Confidence Score:**
   - Do low confidence docs have poor quality?
   - Are there false positives/negatives in confidence?

5. **Error Type Analysis:**
   - What types of errors occur most frequently?
   - Are errors systematic (same issue across docs) or random?

**Interactive Analysis with Codex:**

This phase is typically performed interactively with Codex:

```
User: "Analyze the verification results and identify any patterns"

Codex will:
- Load inventory.json, sample.json, verification_results.json
- Compute aggregate statistics
- Identify correlations
- Report systematic issues
- Provide visualizations (ASCII tables, markdown)
```

**Key Metrics to Calculate:**

- Overall average quality scores by dimension
- Error rates (critical, major, minor)
- Confidence score reliability (true positive rate)
- Quality distribution by category/era/complexity
- Metadata extraction accuracy rate
- Summary quality consistency

## Phase 5: Report Generation

**Purpose:** Generate comprehensive markdown report with findings, recommendations, and supporting data.

**Script:** `scripts/report_generator.py`

**Usage:**
```bash
python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/report_generator.py \
  --inventory /absolute/private-audit/inventory.json \
  --sample /absolute/private-audit/sample.json \
  --results /absolute/private-audit/verification_results.json \
  --sentinel-report /absolute/private-audit/sentinel_failures.json \
  --output /absolute/private-audit/QUALITY_REPORT.md
```

**Generated report:** Corpus composition, actual sample/tag coverage, example
verification records, observed score averages, recorded recommendations, and
critical-case selection/recorded-review counts with evidence limits. Domain
pattern analysis remains an evidence-based review task; the generator does not
calculate error rates, correlations or population confidence automatically.

**Template:** See `references/report_template.md` for a concise review guide. The script generates the maintained baseline report; add narrative analysis only when supported by the records.

## Customization

### Domain-Specific Verification

The skill is designed to work with ANY document extraction pipeline. For domain-specific verification:

**Medical Documents:**
- Use `references/medical_terminology_guide.md`
- Verify accent preservation (Portuguese, Spanish, etc.)
- Check anatomical terms, measurements, diagnostic phrases

**Legal Documents:**
Create `references/legal_terminology_guide.md`:
- Contract clauses and provisions
- Case law citations (e.g., "Brown v. Board of Education, 347 U.S. 483")
- Legal entities and jurisdictions
- Dates and monetary amounts

**Financial Documents:**
Create `references/financial_terminology_guide.md`:
- Accounting terms (GAAP, EBITDA, etc.)
- Currency handling and precision
- Numerical accuracy (percentages, decimals)
- Financial formulas and calculations

**Scientific Documents:**
Create `references/scientific_terminology_guide.md`:
- Chemical formulas (H₂O, CO₂, etc.)
- Mathematical equations
- Scientific notation
- Citations and references

**Technical Documents:**
Create `references/technical_terminology_guide.md`:
- Code snippets and syntax
- API references and URLs
- Technical specifications
- Version numbers

### Custom Priority Tags

Adding tags changes maintained implementation code and needs a separately scoped change with tests. The example below explains the extension point; do not edit helper sources as a per-audit configuration step:

```python
# Define custom priority tags
PRIORITY_TAGS = {
    'LOW_CONF': lambda doc: isinstance(doc.get('min_confidence'), (int, float)) and doc['min_confidence'] < 0.7,
    'OLD_DOC': lambda doc: doc.get('era') in ['1990s', '2000s'],
    'COMPLEX': lambda doc: doc['page_count'] >= 10,
    'MULTI_PAGE': lambda doc: 3 <= doc['page_count'] < 10,

    # Add custom tags
    'HAS_IMAGES': lambda doc: doc.get('has_images', False),
    'HANDWRITTEN': lambda doc: 'handwritten' in doc.get('tags', []),
    'LONG_TEXT': lambda doc: doc.get('word_count', 0) > 5000,
}
```

### Adjusting Sample Size

For different corpus sizes:

- **Small (<100 docs):** 25-30 documents (25-30% coverage)
- **Medium (100-500):** 30-50 documents (10-20% coverage)
- **Large (500-1000):** 50-75 documents (5-10% coverage)
- **Very large (>1000):** 100+ documents (5% coverage)

Always ensure 100% coverage of critical cases (LOW_CONF, COMPLEX).

## Expected Outputs

**Files Generated:**

1. **inventory.json:** Complete document catalog with metadata
2. **sample.json:** Selected documents with sampling justification
3. **verification_results.json:** Detailed verification scores and findings
4. **QUALITY_REPORT.md:** Comprehensive quality assessment report

**Report Deliverables:**

- Executive summary with actionable recommendations
- Quality grade (A through F)
- Quantitative metrics and aggregate scores
- Systematic issue identification
- Documents requiring reprocessing (if any)
- Configuration tuning recommendations

## Interpretation Guide

### Reviewed-sample ratings

The existing A-F cutoffs describe observed review scores: A ≥9, B ≥7, C ≥5,
D ≥3, and F below 3. Unrated dimensions remain N/A. These labels are not
production-readiness decisions and cannot clear recorded critical issues or
characterize unreviewed documents.

### Confidence Score Interpretation

**Important:** Low confidence does NOT necessarily mean poor quality.

- **HIGH_CONF (≥0.9):** Standard content, consistent extraction
- **MED_CONF (0.7-0.9):** Some variation in extraction attempts
- **LOW_CONF (<0.7):** Challenging content (images, handwriting, complex layouts)

**Low confidence may indicate:**
- Image-heavy pages (legitimate uncertainty about visual content)
- Handwritten text (inherent ambiguity in interpretation)
- Complex layouts (inconsistent extraction across attempts)
- Very long documents (more opportunities for variation)

**Verify that:**
- LOW_CONF documents are appropriately flagged
- Extraction quality is acceptable despite low confidence
- No HIGH_CONF documents with poor quality (false negatives)

### Error Severity Classification

- **Critical:** Missing clinical/legal findings, hallucinations, wrong patient/party names
- **Major:** Incorrect technical terms, wrong measurements/amounts, incorrect dates
- **Minor:** Spacing variations, case inconsistencies, handwriting uncertainty

**Acceptance policy:** Do not invent production pass/fail thresholds from this
review sample. Report observed severity with an explicit denominator and any
applicable caller-approved acceptance policy. Missing policy or evidence remains
unresolved; an average score does not authorize deployment or reprocessing.

## Troubleshooting

**Issue:** Inventory generation fails with encoding errors

**Solution:** Specify encoding in script configuration:
```python
with open(file, 'r', encoding='utf-8') as f:
```

**Issue:** Can't find source documents for verification

**Solution:** Supply an existing per-audit path config using
`--path-config /absolute/private-audit/path_config.json`, for example:
```json
{"input_dir": "/path/to/selected/source/documents"}
```
The current mapping finds a flat `<doc_stem>.pdf` under that directory. A
different layout requires a scoped implementation change; do not edit maintained
helper code as a routine audit step.

**Issue:** Sample is too small or unrepresentative

**Solution:**
- Increase `--target-size` parameter
- Adjust priority tag coverage in sampling config
- Add custom tags for under-represented document types

**Issue:** Verification taking too long

**Solution:**
- Reduce sample size for initial audit
- Focus verification on highest priority documents only
- Use parallel verification (multiple Codex sessions)

**Issue:** Report generation fails

**Solution:**
- Ensure all input files exist (inventory, sample, results)
- Check JSON format validity
- Reconcile duplicate or unbound profile/document identities
- Check that the same sentinel report matches the selected inventory

## Best Practices

1. **Run inventory immediately after extraction pipeline completes** to capture fresh metadata

2. **Always include 100% of LOW_CONF and COMPLEX documents** in sample for comprehensive risk assessment

3. **Verify documents interactively with Codex vision** for highest accuracy in quality assessment

4. **Save verification results incrementally** to avoid losing progress

5. **Focus on systematic patterns** rather than individual document issues

6. **Re-run audit after significant pipeline changes** (prompt modifications, model changes, configuration updates)

7. **Document domain-specific terminology guides** for consistent verification across audits

8. **Track quality trends over time** by comparing reports from multiple audit runs

## Limitations

- Requires Codex vision access for source document comparison
- Verification phase is time-intensive (manual review required)
- Statistical confidence depends on sample size
- Domain-specific terminology verification requires expertise
- Not suitable for real-time quality monitoring (batch assessment only)

## Integration with CI/CD

For automated quality audits in CI/CD pipelines:

```yaml
# .github/workflows/quality-audit.yml
name: Quality Audit
on:
  schedule:
    - cron: '0 0 1 * *'  # Monthly

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - name: Generate Inventory
        run: python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/generate_inventory.py ...

      - name: Run Sampling
        run: python .agents/skills/parsemedicalexams_extraction-quality-audit/scripts/stratified_sampling.py ...

      - name: Trigger Manual Verification
        uses: peter-evans/create-issue-from-file@v4
        with:
          title: Monthly Quality Audit Required
          content-filepath: verification_checklist.md
```

Note: Verification phase requires human review and cannot be fully automated.

## References

- **Sampling Methodology:** `references/sampling_methodology.md`
- **Verification Checklist:** `references/verification_checklist.md`
- **Medical Terminology Guide:** `references/medical_terminology_guide.md`
- **Report Template:** `references/report_template.md`

## Version History

- **v1.0:** Initial release (2026-01-21)
  - Full 5-phase quality audit workflow
  - Stratified sampling implementation
  - Domain-agnostic design with medical example
  - Comprehensive report generation

## Support

For issues or questions:
- Check `references/` directory for detailed guidance
- Review example outputs in skill directory
- Consult QUALITY_INVESTIGATION_REPORT.md for reference implementation

## License

This skill is part of the Codex skills ecosystem and is provided as-is for quality assessment of document extraction pipelines.

## Helper fixture checks

Run from the repository root:

```sh
python3 -m unittest discover -s .agents/skills/parsemedicalexams_extraction-quality-audit/tests -v
```

Tests use synthetic metadata and text, not real profiles, source PDFs, extraction
outputs or provider credentials. Human verification of actual documents remains
required for a requested audit.
