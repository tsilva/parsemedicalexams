# Quality review report guide

Use `scripts/report_generator.py` for the maintained baseline report. Its inputs
are the selected inventory, sample, verification records and, when used during
sampling, the same sentinel JSON. Keep all per-run artifacts in the private
artifact directory described in the skill. This guide supplements that report;
it is not a second calculator or a requirement to repeat every section manually.

## Required evidence

- State the scope, artifact locations, review date, corpus size, selected count
  and number of matching verification records.
- Bind every document and result to `(profile, doc_stem)`. Reconcile duplicates
  before reporting. Count excluded unbound or out-of-sample records visibly.
- Report known critical cases selected and those with a matching review record
  as counts over the actual denominator. Include LOW_CONF, COMPLEX,
  ERROR_FLAGGED and supplied sentinel failures. No known cases means N/A.
  State explicitly when the sentinel report was not supplied.
- Keep valid zero scores. Use N/A for missing or invalid scores; do not infer a
  grade from absent ratings. The existing A-F grade is a descriptive rating of
  observed review scores, not a population accuracy estimate.

A result record documents review; it does not independently prove its contents
correct. Distinguish selection, recorded review and substantive verification.

## Concise narrative structure

The generator covers corpus composition, sampling coverage, example records,
observed scores, recorded recommendations and evidence limits. Add only useful
analysis supported by the records:

1. **Findings:** Explain the most consequential observed defects, with the
   profile/document identity and a source-versus-output example. Keep clinical,
   legal or other domain judgments with a qualified reviewer when needed.
2. **Patterns:** Compare reviewed categories, eras or complexity groups only
   when enough comparable records exist. Give each group's count and observed
   score or issue denominator. Describe sparse or conflicting evidence plainly.
   Review keyword matches before assigning issue severity.
3. **Recommendations:** Link each proposed action to its recorded issue and
   affected documents. Separate observations, hypotheses and requested work.
   Reprocessing, deployment, messaging and monitoring require their applicable
   authorization; generating this report does not perform those actions.
4. **Limits and next review:** Identify unreviewed documents, missing source
   comparisons, unavailable ratings, excluded records and unresolved questions.
   Recommend further work only when the findings justify it.

Example coverage excerpt (synthetic):

```text
Corpus: 100 documents
Selected: 30
Matching verification records: 20
Known critical cases selected: 9/9 (100%)
Known critical cases with a matching verification record: 6/9 (66.7%)
Unbound or out-of-sample records excluded: 2
Sentinel report: supplied and included
Population confidence level and margin of error: not computed
```

## Statistical and acceptance limits

This priority sample includes mandatory failures, overlapping tags, category
quotas and random fill. Reaching a particular sample size does not produce a
confidence level or margin of error. Do not copy old 90%/95% claims or extrapolate
an A-F rating to the entire corpus. See the sampling methodology for context.

An average score does not rule out a critical defect or establish production
readiness. Apply a caller-approved acceptance policy only when both its
requirements and supporting evidence are available. Otherwise leave acceptance
unresolved. Do not invent numeric safety thresholds, clear all actions because
scores are high, or prescribe a recurring schedule without a supported need.

## Final review

Check counts and denominators against the supplied metadata, preserve unknowns,
confirm every example's profile, and verify that conclusions stay within the
reviewed subset. Do not include unnecessary personal document text or identify
unreviewed documents as successfully verified. The verification checklist owns
scoring guidance; the sampling methodology owns selection policy.
