# Stratified Sampling Methodology

This document explains the sampling methodology used in extraction quality audits.

## Review objective and limits

This helper builds a priority-based review sample: mandatory known failures,
overlapping risk tags, category quotas, and random fill. It aims to find and
characterize defects in the selected scope. It does not establish a probability
sampling design or guarantee a representative population estimate.

Scores and issue counts describe reviewed documents. A confidence level or
margin of error needs a justified estimator and sampling model; it does not
follow from reaching 30 documents. See the
[NIST confidence-interval explanation](https://itl.nist.gov/div898/handbook/prc/section1/prc14.htm).
The report calculates explicit coverage and does not manufacture statistical
confidence or production-readiness conclusions.

## Priority Tag System

Priority tags identify documents with higher risk or importance for quality assessment.

### Default Priority Tags

| Tag | Definition | Priority Weight | Default Coverage |
|-----|------------|-----------------|------------------|
| `LOW_CONF` | min_confidence < 0.7 | 1000 | 100% |
| `COMPLEX` | page_count >= 10 | 100 | 100% |
| `OLD_DOC` | era in [1990s, 2000s] | 10 | 60-80% |
| `MULTI_PAGE` | 3 <= page_count < 10 | 1 | 40-60% |

**Priority weights** determine sampling order (higher = sampled first).

**Coverage** specifies what percentage of documents with this tag should be included in the sample.

### Custom Priority Tags

Create custom tags for your domain:

**Medical Documents:**
```python
'HANDWRITTEN': lambda doc: 'prescription' in doc.get('category', '').lower(),
'IMAGE_HEAVY': lambda doc: 'endoscopy' in doc.get('category', '').lower(),
'COMPLEX_ANATOMY': lambda doc: doc.get('word_count', 0) > 5000,
```

**Legal Documents:**
```python
'MULTI_PARTY': lambda doc: doc.get('party_count', 0) > 2,
'LONG_FORM': lambda doc: doc.get('page_count', 0) > 50,
'HISTORICAL': lambda doc: doc.get('year', 9999) < 1950,
```

**Financial Documents:**
```python
'HIGH_VALUE': lambda doc: doc.get('transaction_amount', 0) > 1000000,
'MULTI_CURRENCY': lambda doc: doc.get('currency_count', 0) > 1,
'COMPLEX_TABLES': lambda doc: doc.get('table_count', 0) > 10,
```

### Tag Design Guidelines

**Good tags:**
- ✓ Objective criteria (confidence score, page count, date)
- ✓ Actionable (identifies specific quality concerns)
- ✓ Mutually inclusive (documents can have multiple tags)
- ✓ Stable across audit runs

**Avoid:**
- ✗ Subjective criteria requiring manual judgment
- ✗ Redundant with existing tags
- ✗ Too granular (creates tiny strata)

## Sample Size Calculation

### Recommended Sizes by Corpus Size

| Corpus Size | Minimum Sample | Recommended Sample | Coverage |
|-------------|----------------|-------------------|----------|
| < 100 docs | 25-30 | 30-40 | 25-40% |
| 100-500 docs | 30-40 | 40-60 | 10-20% |
| 500-1000 docs | 40-60 | 60-100 | 6-12% |
| 1000-5000 docs | 60-100 | 100-200 | 2-5% |
| > 5000 docs | 100-200 | 200-500 | 1-4% |

### Review budget and coverage

The size ranges above are review-budget examples, not inferential guarantees.
Report the known critical-case denominator, how many were selected, and how
many have a matching verification record. Zero known critical cases means N/A,
not an assumed 100%. State when the sentinel report was not supplied.

### Critical Cases Always Included

**100% coverage** of critical cases is non-negotiable:
- **All LOW_CONF documents** (often <5% of corpus)
- **All COMPLEX documents** (often <5% of corpus)
- **All ERROR_FLAGGED documents** (if available)

Even if this exceeds target sample size, include them all.

## Sampling Phases

### Phase 0: Mandatory sentinel failures

Use `sentinel_scan.py --json-output` and pass that same report through
`stratified_sampling.py --sentinel-report` and `report_generator.py --sentinel-report`.
Match full document directories to the explicit inventory; unmatched or ambiguous
failures block sample generation. Include every matched failure even when the
nominal target is exceeded. Existing priority coverage then applies to remaining
documents. Selection does not establish successful verification.

### Phase 1: Priority Tags (High Coverage)

Sample documents based on priority tags, starting with highest priority.

**Example (target n=30):**
```
1. LOW_CONF (5 docs) → 100% coverage = 5 sampled
2. COMPLEX (4 docs) → 100% coverage = 4 sampled
3. OLD_DOC (18 docs) → floor(18 × 0.7) = 12 sampled
4. MULTI_PAGE (44 docs) → floor(44 × 0.4) = 17 sampled
   With disjoint groups: 38 documents (8 over target, but acceptable)
```

**Overlap handling:** Documents can satisfy multiple tags. Track `(profile, doc_stem)` identities to avoid duplicates without dropping same-named documents from different profiles.

### Phase 2: Category Balancing

Ensure representation across document categories.

**Target distribution:** Proportional to corpus or minimum per category.

**Example:**
```python
category_balance = {
    'imaging': 3,      # Up to 3 additional unselected imaging documents
    'ultrasound': 3,
    'endoscopy': 3,
    'other': 2,
}
```

The current helper adds up to each configured category count from documents
not already selected, when this phase starts below target. These are additional
quotas, not total-category minima or proportional allocation; multiple categories
can take the sample over target. Inspect the resulting distribution.

### Phase 3: Era Diversity

Ensure temporal diversity across document eras.

The current helper adds up to three unselected `2020s` documents when slots
remain. It does not implement per-era minima. Inspect the remaining eras
against the requested review goals rather than claiming guaranteed coverage.

### Phase 4: Confidence Baseline

Include high-confidence documents as **baseline/control group**.

**Purpose:**
- Verify that high confidence correlates with high quality
- Establish quality ceiling
- Detect false negatives (high confidence but poor quality)

**Implemented selection:** Up to three unselected high-confidence documents
(confidence ≥ 0.9) when slots remain. Unknown confidence is never a baseline.

### Phase 5: Random Fill

Fill remaining slots with simple random sampling.

**Purpose:**
- Reach target sample size
- Add serendipity (discover unexpected patterns)
- Check coverage against the requested review goals

## Sampling Configuration

### Configuration File Format

```json
{
  "priority_tags": {
    "LOW_CONF": {
      "coverage": 1.0,
      "priority": 1000
    },
    "COMPLEX": {
      "coverage": 1.0,
      "priority": 100
    },
    "OLD_DOC": {
      "coverage": 0.7,
      "priority": 10
    },
    "MULTI_PAGE": {
      "coverage": 0.4,
      "priority": 1
    }
  },
  "category_balance": {
    "imaging": 3,
    "ultrasound": 3,
    "endoscopy": 3,
    "other": 2
  },
  "random_seed": 42
}
```

### Adjusting Coverage Rates

**When to increase coverage:**
- High error rates discovered in initial verification
- Critical document type requires thorough assessment
- Compliance/audit requirements
- Small absolute numbers (e.g., only 3 OLD_DOC total)

**When to decrease coverage:**
- Very large corpus (>5000 documents)
- Time/budget constraints
- Preliminary/exploratory audit
- Consistent high quality observed

## Common Sampling Scenarios

### Scenario 1: Small Corpus (<100 docs)

**Approach:** Near-exhaustive sampling
- Target: 30-50% of corpus
- Include ALL critical cases
- High coverage of all tags (80-100%)
- Minimal random fill needed

### Scenario 2: Large Corpus (>1000 docs)

**Approach:** Targeted sampling
- Target: 1-5% of corpus (100-200 docs)
- 100% of critical cases (may be substantial)
- Medium coverage of high-priority tags (50-70%)
- Lower coverage of low-priority tags (20-30%)
- Significant random fill

### Scenario 3: Suspected Quality Issues

**Approach:** Problem-focused sampling
- Over-sample problem areas (e.g., 100% OLD_DOC if suspected issue)
- Include all LOW_CONF documents
- Add custom tags for suspected issues
- Reduce random fill, increase targeted sampling

### Scenario 4: Periodic Monitoring

**Approach:** Trend tracking
- Consistent sample size across audit runs
- Same random seed for reproducibility
- Track quality changes over time
- Include new document types as they appear

## Sample Validation

### Checking Sample Quality

Before verification, validate that the sample covers the requested review goals:

**1. Profile/User Distribution:**
```
Is profile distribution in sample similar to corpus?
Example: 32% Tiago, 68% Cristina in both
```

**2. Category Coverage:**
```
Are all major categories represented?
Any category < 5% of corpus but 0% of sample?
```

**3. Era Distribution:**
```
Does sample span full time range?
Any era completely missing?
```

**4. Complexity Distribution:**
```
Mix of simple, multi-page, and complex documents?
```

**5. Confidence Distribution:**
```
Sample includes low, medium, and high confidence?
```

### Red Flags

**Warning signs of poor sampling:**
- ⚠️ All sampled docs from single profile (if multi-profile corpus)
- ⚠️ Zero representation of major category (>10% of corpus)
- ⚠️ No high-confidence documents (can't establish baseline)
- ⚠️ All documents from single era (temporal bias)
- ⚠️ Sample entirely from problem cases (can't assess typical quality)

**Action:** Regenerate sample with adjusted parameters.

## Reporting Sample Composition

Always report sampling methodology in final report:

**Include:**
1. Target sample size and actual size
2. Priority tags used and coverage rates
3. Sampling phases applied
4. Profile/category/era distribution comparison (corpus vs. sample)
5. Measured critical-case selection and recorded-review coverage, plus evidence limits
6. Random seed (for reproducibility)

**Example:**
```
Sample Size: 44 documents (22% of 199 corpus)

Priority Coverage:
- LOW_CONF: 5/5 (100%)
- COMPLEX: 4/4 (100%)
- OLD_DOC: 15/18 (83%)
- MULTI_PAGE: 17/44 (39%)

Known critical cases selected: 9/9 (100%)
Known critical cases with matching verification records: 6/9 (66.7%)
Population confidence level and margin of error: not computed
Random Seed: 42
```

## Advanced Topics

### Adaptive Sampling

Adjust sampling during verification if unexpected patterns emerge:

**Example:**
- Initial sample reveals 50% error rate in category X
- Add more category X documents mid-audit
- Report: "Adaptive sampling applied: added 10 category X documents after initial findings"

### Multi-Stage Sampling

For very large corpora (>10,000 docs):

1. **Stage 1:** Sample documents (e.g., 200 docs)
2. **Stage 2:** For multi-page documents, sample pages (e.g., 2-3 pages per doc)

### Cluster Sampling

If documents organized in natural clusters (e.g., by patient, case, project):

1. Sample clusters (e.g., 20 patients)
2. Verify all documents within selected clusters

**Trade-off:** Easier to verify (fewer context switches) but potentially less representative.

## References

- Cochran, W. G. (1977). *Sampling Techniques*. Wiley.
- Lohr, S. L. (2019). *Sampling: Design and Analysis*. Chapman and Hall/CRC.
- Thompson, S. K. (2012). *Sampling*. Wiley.

## Version History

- **v1.0** (2026-01-21): Initial methodology based on parsemedicalexams quality investigation
