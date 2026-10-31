#!/usr/bin/env python3
"""
Generate comprehensive quality assessment report from verification results.
Produces markdown report with statistics, findings, and recommendations.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from stratified_sampling import document_key, known_confidence, known_number, mandatory_documents, protect_inputs


def load_json(file_path: Path) -> any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def known_score(value):
    return known_number(value, 10)


def review_coverage(inventory, sample, results, sentinel_report=None):
    """Count explicit identities; matching records do not certify verification quality."""
    if not isinstance(inventory, list) or not inventory or not isinstance(sample, list) or not isinstance(results, list):
        raise ValueError('Report requires a nonempty inventory, sample list and result list')
    inventory_keys = [document_key(doc) for doc in inventory]
    sample_keys = [document_key(doc) for doc in sample]
    if len(set(inventory_keys)) != len(inventory_keys) or len(set(sample_keys)) != len(sample_keys):
        raise ValueError('Duplicate document identities in inventory or sample')
    if not set(sample_keys).issubset(inventory_keys):
        raise ValueError('Sample contains documents outside the selected inventory')
    selected = set(sample_keys)
    matched, seen, excluded = [], set(), 0
    for result in results:
        try:
            key = document_key(result)
        except ValueError:
            excluded += 1
            continue
        if key not in selected:
            excluded += 1
            continue
        if not isinstance(result.get('verification'), dict):
            raise ValueError('A matched result needs a verification object')
        if key in seen:
            raise ValueError('Duplicate verification records need reconciliation before reporting')
        seen.add(key)
        matched.append(result)
    critical = {document_key(doc) for doc in inventory
                if {'LOW_CONF', 'COMPLEX', 'ERROR_FLAGGED'} & set(doc.get('tags', []))}
    if sentinel_report is not None:
        critical.update(mandatory_documents(inventory, sentinel_report))
    return {'critical_total': len(critical), 'critical_selected': len(critical & selected),
            'critical_recorded': len(critical & seen), 'excluded_records': excluded,
            'matched_records': len(matched), 'sentinel_supplied': sentinel_report is not None}, matched


def calculate_quality_grade(avg_score: float) -> str:
    """Calculate letter grade from average score."""
    if avg_score is None:
        return "N/A"
    if avg_score >= 9:
        return "A (Excellent)"
    elif avg_score >= 7:
        return "B (Good)"
    elif avg_score >= 5:
        return "C (Fair)"
    elif avg_score >= 3:
        return "D (Poor)"
    else:
        return "F (Failing)"


def generate_executive_summary(
    inventory: List[Dict],
    sample: List[Dict],
    results: List[Dict]
) -> str:
    """Generate executive summary section."""
    verified_count = len(results)
    total_docs = len(inventory)

    # Calculate aggregate scores
    scores = []
    for result in results:
        ver = result.get('verification', {})
        trans = ver.get('transcription_quality', {})
        overall = known_score(trans.get('overall_score'))
        if overall is not None:
            scores.append(overall)

    avg_score = sum(scores) / len(scores) if scores else None
    grade = calculate_quality_grade(avg_score) if avg_score is not None else "N/A (no observed overall score)"
    average_text = f"{avg_score:.2f}/10" if avg_score is not None else "N/A"

    # Count critical issues
    critical_issues = []
    for result in results:
        ver = result.get('verification', {})
        for issue in ver.get('notable_issues', []):
            if 'critical' in issue.lower() or 'hallucination' in issue.lower():
                critical_issues.append(issue)

    # Confidence scoring assessment
    low_conf_docs = [d for d in inventory if known_confidence(d) is not None and known_confidence(d) < 0.7]
    low_conf_verified = [r for r in results if 'LOW_CONF' in r.get('tags', [])]

    summary = f"""## Executive Summary

This review inventoried **{total_docs} documents** and matched verification records for **{verified_count} selected documents**. The ratings below describe those records, not population-wide extraction accuracy.

### Reviewed-Sample Rating: **{grade}**

**Average Observed Quality Score:** {average_text}

### Key Findings

"""

    # Add key findings based on data
    if avg_score is None:
        summary += "No overall quality rating is available from the matched records.\n"
    elif avg_score >= 9:
        summary += "✅ **Reviewed Sample: HIGH RATINGS**\n"
        summary += "- High observed scores in the reviewed records\n"
    elif avg_score >= 7:
        summary += "✓ **Reviewed Sample: GOOD RATINGS**\n"
        summary += "- Reviewed records have generally high scores; retain the listed limitations\n"
    else:
        summary += "⚠️ **Reviewed Sample: ISSUES RECORDED**\n"
        summary += "- Review the recorded issues; these scores do not characterize unreviewed documents\n"

    summary += f"\n**Confidence Scoring:**\n"
    summary += f"- Low confidence documents: {len(low_conf_docs)}/{total_docs} ({len(low_conf_docs)/total_docs*100:.1f}%)\n"

    if len(low_conf_verified) > 0:
        summary += f"- {len(low_conf_verified)} low confidence documents verified\n"

    if len(critical_issues) > 0:
        summary += f"\n**Issue mentions containing critical/hallucination terms (review required):** {len(critical_issues)}\n"
        for issue in critical_issues[:3]:
            summary += f"- {issue}\n"
    else:
        summary += f"\nNo critical/hallucination issue mentions are recorded in the matched reviews; unreviewed documents remain unassessed.\n"

    return summary


def generate_data_landscape(inventory: List[Dict]) -> str:
    """Generate data landscape section."""
    total = len(inventory)

    # Profile distribution
    profiles = {}
    for doc in inventory:
        p = doc.get('profile', 'unknown')
        profiles[p] = profiles.get(p, 0) + 1

    # Era distribution
    eras = {}
    for doc in inventory:
        era = doc.get('era', 'unknown')
        eras[era] = eras.get(era, 0) + 1

    # Category distribution
    categories = {}
    for doc in inventory:
        cat = doc.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    # Confidence distribution
    high_conf = len([d for d in inventory if known_confidence(d) is not None and known_confidence(d) >= 0.9])
    med_conf = len([d for d in inventory if known_confidence(d) is not None and 0.7 <= known_confidence(d) < 0.9])
    low_conf = len([d for d in inventory if known_confidence(d) is not None and known_confidence(d) < 0.7])
    unknown_conf = total - high_conf - med_conf - low_conf

    # Complexity distribution
    simple = len([d for d in inventory if d.get('page_count', 0) <= 2])
    multi = len([d for d in inventory if 3 <= d.get('page_count', 0) < 10])
    complex_docs = len([d for d in inventory if d.get('page_count', 0) >= 10])

    section = f"""## Data Landscape

### Corpus Statistics

**Total Documents:** {total}

"""

    if len(profiles) > 1:
        section += "| Profile | Documents | Percentage |\n"
        section += "|---------|-----------|------------|\n"
        for profile, count in sorted(profiles.items()):
            pct = count / total * 100
            section += f"| {profile} | {count} | {pct:.0f}% |\n"
        section += "\n"

    section += """### Temporal Distribution

| Era | Count | Percentage |
|-----|-------|------------|
"""
    for era, count in sorted(eras.items()):
        pct = count / total * 100
        section += f"| {era} | {count} | {pct:.0f}% |\n"

    section += """
### Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
"""
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        section += f"| {cat} | {count} | {pct:.0f}% |\n"

    section += f"""
### Confidence Score Distribution

| Level | Count | Percentage |
|-------|-------|------------|
| High (≥0.9) | {high_conf} | {high_conf/total*100:.0f}% |
| Medium (0.7-0.9) | {med_conf} | {med_conf/total*100:.0f}% |
| Low (<0.7) | {low_conf} | {low_conf/total*100:.1f}% |
| Unknown/invalid | {unknown_conf} | {unknown_conf/total*100:.1f}% |

### Complexity Distribution

| Complexity | Count | Percentage |
|------------|-------|------------|
| Simple (1-2 pages) | {simple} | {simple/total*100:.0f}% |
| Multi-page (3-9) | {multi} | {multi/total*100:.0f}% |
| Complex (10+) | {complex_docs} | {complex_docs/total*100:.1f}% |
"""

    return section


def generate_sampling_summary(sample: List[Dict], inventory: List[Dict]) -> str:
    """Generate sampling methodology section."""
    sample_size = len(sample)
    total = len(inventory)

    # Count priority tag coverage
    tag_coverage = {}
    priority_tags = ['LOW_CONF', 'COMPLEX', 'OLD_DOC', 'MULTI_PAGE']

    for tag in priority_tags:
        in_inventory = len([d for d in inventory if tag in d.get('tags', [])])
        in_sample = len([d for d in sample if tag in d.get('tags', [])])
        if in_inventory > 0:
            coverage = in_sample / in_inventory * 100
            tag_coverage[tag] = (in_inventory, in_sample, coverage)

    section = f"""## Sampling Methodology

**Sample Size:** {sample_size} documents ({sample_size/total*100:.1f}% of corpus)

**Priority Coverage:**

| Priority Tag | Population | Sampled | Coverage |
|--------------|------------|---------|----------|
"""

    for tag in priority_tags:
        if tag in tag_coverage:
            pop, samp, cov = tag_coverage[tag]
            section += f"| {tag} | {pop} | {samp} | {cov:.0f}% |\n"

    return section


def score_cell(value):
    observed = known_score(value)
    return f'{observed:.1f}/10' if observed is not None else 'N/A'


def generate_verification_results(results: List[Dict]) -> str:
    """Generate detailed verification results section."""
    if not results:
        return "## Verification Results\n\nNo documents verified yet.\n"

    section = "## Detailed Verification Results\n\n"

    # Show top 3 documents as examples
    section += "### Example Verified Documents\n\n"

    for i, result in enumerate(results[:3], 1):
        doc_stem = result.get('doc_stem', 'Unknown')
        ver = result.get('verification', {})

        section += f"#### Document {i}: {doc_stem}\n\n"
        section += f"**Profile:** {result.get('profile', 'N/A')}\n"
        section += f"**Pages:** {result.get('page_count', 'N/A')}\n"
        section += f"**Confidence:** {known_confidence(result) if known_confidence(result) is not None else 'unknown'}\n"
        section += f"**Category:** {result.get('category', 'N/A')}\n"
        section += f"**Tags:** {', '.join(result.get('tags', []))}\n\n"

        section += "**Verification Scores:**\n\n"
        section += "| Quality Dimension | Score |\n"
        section += "|-------------------|-------|\n"

        meta = ver.get('metadata_accuracy', {})
        section += f"| Metadata Accuracy | {score_cell(meta.get('score'))} |\n"

        trans = ver.get('transcription_quality', {})
        section += f"| Transcription Completeness | {score_cell(trans.get('completeness_score'))} |\n"
        section += f"| Transcription Accuracy | {score_cell(trans.get('accuracy_score'))} |\n"
        section += f"| Layout Preservation | {score_cell(trans.get('layout_preservation_score'))} |\n"

        term = ver.get('domain_terminology', {})
        section += f"| Domain Terminology | {term.get('rating', 'N/A')} |\n"

        summ = ver.get('summary_quality', {})
        section += f"| Summary Quality | {score_cell(summ.get('score'))} |\n"

        section += f"\n**Overall Assessment:** {ver.get('overall_assessment', 'N/A')}\n\n"

        if ver.get('notable_issues'):
            section += "**Notable Issues:**\n"
            for issue in ver.get('notable_issues', [])[:3]:
                section += f"- {issue}\n"
            section += "\n"

    return section


def generate_quantitative_metrics(results: List[Dict]) -> str:
    """Describe only observed review scores; never impute missing ratings as zero."""
    if not results:
        return ""
    section = "## Quantitative Metrics\n\n### Observed Review Scores\n\n"
    section += "| Quality Dimension | Average Score | Grade |\n| --- | --- | --- |\n"
    dimensions = [('Metadata Accuracy', 'metadata_accuracy', 'score'),
                  ('Transcription Completeness', 'transcription_quality', 'completeness_score'),
                  ('Transcription Accuracy', 'transcription_quality', 'accuracy_score'),
                  ('Layout Preservation', 'transcription_quality', 'layout_preservation_score'),
                  ('Summary Quality', 'summary_quality', 'score'),
                  ('Overall Quality', 'transcription_quality', 'overall_score')]
    for label, dimension, field in dimensions:
        values = [known_score(row.get('verification', {}).get(dimension, {}).get(field)) for row in results]
        observed = [value for value in values if value is not None]
        mean = sum(observed) / len(observed) if observed else None
        cells = f'{score_cell(mean)} | {calculate_quality_grade(mean)}' if mean is not None else 'N/A | N/A'
        section += f'| {label} | {cells} |\n'
    return section


def generate_recommendations(results: List[Dict], avg_score: float) -> str:
    """Preserve recorded recommendations without inventing pipeline disposition."""
    section = "## Recommendations\n\n"
    if avg_score is None:
        section += "No overall score is available. Review the recorded issues and evidence gaps.\n\n"
    else:
        section += "The average is descriptive of the reviewed records. It does not establish production readiness, rule out critical issues, or authorize reprocessing.\n\n"
    recommendations = []
    for result in results:
        recommendations.extend(result.get('verification', {}).get('recommendations', []))
    if recommendations:
        section += "### Recommendations Recorded During Verification\n\n"
        for recommendation in dict.fromkeys(recommendations):
            section += f"- {recommendation}\n"
    return section


def generate_report(
    inventory: List[Dict],
    sample: List[Dict],
    results: List[Dict],
    output_file: Path,
    sentinel_report=None
):
    """Generate complete markdown report."""

    coverage, results = review_coverage(inventory, sample, results, sentinel_report)

    # Calculate average score for use in multiple sections
    scores = []
    for result in results:
        ver = result.get('verification', {})
        trans = ver.get('transcription_quality', {})
        overall = known_score(trans.get('overall_score'))
        if overall is not None:
            scores.append(overall)

    avg_score = sum(scores) / len(scores) if scores else None

    # Generate report sections
    report = f"""# Extraction Quality Assessment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total Documents:** {len(inventory)}
**Sample Size:** {len(sample)}
**Documents Verified:** {len(results)}

---

"""

    report += generate_executive_summary(inventory, sample, results)
    report += "\n---\n\n"
    report += generate_data_landscape(inventory)
    report += "\n---\n\n"
    report += generate_sampling_summary(sample, inventory)
    report += "\n---\n\n"
    report += generate_verification_results(results)
    report += "\n---\n\n"
    report += generate_quantitative_metrics(results)
    report += "\n---\n\n"
    report += generate_recommendations(results, avg_score)
    report += "\n---\n\n"

    total = coverage['critical_total']
    def covered(count):
        return f'{count}/{total} ({count / total * 100:.1f}%)' if total else 'N/A (no known critical cases)'
    sentinel_scope = 'included in the known-critical denominator' if coverage['sentinel_supplied'] else 'not supplied; denominator uses inventory critical tags only'
    report += f"""## Appendices

### A. Coverage and Evidence Limits

- Critical tags counted: LOW_CONF, COMPLEX, ERROR_FLAGGED
- Known critical cases selected: {covered(coverage['critical_selected'])}
- Known critical cases with a matching verification record: {covered(coverage['critical_recorded'])}
- Unbound or out-of-sample result records excluded from aggregates: {coverage['excluded_records']}
- Sentinel report: {sentinel_scope}

A matching record is evidence of recorded review, not independent proof that its
contents are correct. This priority-based sample oversamples known risks and
uses overlapping tags and category quotas. No population confidence level or
margin of error is computed; sample size alone does not establish one. Reported
scores describe the reviewed subset and are not population accuracy estimates.

### B. Methodology

1. Priority-based sampling, including supplied known failures before random fill
2. Systematic verification with the existing checklist
3. Descriptive review ratings for observed dimensions; missing scores remain N/A
4. Pattern analysis limited to the available evidence

### C. References

- Verification checklist: `references/verification_checklist.md`
- Sampling policy: `references/sampling_methodology.md`

---

**Report End**
"""

    # Write report to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✓ Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate quality assessment report'
    )
    parser.add_argument(
        '--inventory',
        type=Path,
        default=Path('inventory.json'),
        help='Inventory file'
    )
    parser.add_argument(
        '--sample',
        type=Path,
        default=Path('sample.json'),
        help='Sample file'
    )
    parser.add_argument(
        '--results',
        type=Path,
        default=Path('verification_results.json'),
        help='Verification results file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('QUALITY_REPORT.md'),
        help='Output report filename'
    )

    parser.add_argument('--sentinel-report', type=Path, help='Same sentinel JSON used for sampling; optional explicit evidence')
    args = parser.parse_args()
    try:
        protect_inputs(args.output, [args.inventory, args.sample, args.results, args.sentinel_report])
    except (OSError, ValueError):
        parser.error('Report output must not overwrite audit evidence')

    # Load data
    print("Loading data...")
    inventory = load_json(args.inventory)
    sample = load_json(args.sample)
    results = load_json(args.results)

    print(f"  Inventory: {len(inventory)} documents")
    print(f"  Sample: {len(sample)} documents")
    print(f"  Result records supplied: {len(results)} (binding checked before reporting)")

    # Generate report
    print("\nGenerating report...")
    try:
        sentinel = load_json(args.sentinel_report) if args.sentinel_report else None
        protect_inputs(args.output, [args.inventory, args.sample, args.results, args.sentinel_report])
        generate_report(inventory, sample, results, args.output, sentinel)
    except (OSError, ValueError, TypeError, KeyError):
        parser.error('Invalid or ambiguous audit evidence; report was not completed')


if __name__ == '__main__':
    main()
