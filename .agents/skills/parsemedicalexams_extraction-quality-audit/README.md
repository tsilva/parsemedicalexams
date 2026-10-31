# Extraction Quality Audit

This local parsemedicalexams skill reviews document extraction quality with
Codex. Use [SKILL.md](SKILL.md) for the maintained workflow and commands from the
repository root; this README does not repeat them.

The existing helpers inventory selected outputs, scan for known failure strings,
select a priority-based sample, guide source-versus-output review and generate a
descriptive report. Pass the scanner's JSON into sampling and reporting so known
failures are included without manually editing a sample list. Review and scoring
still require evidence-based judgment.

Use the project's existing Python environment and PyYAML dependency. Keep private
inventories, results and reports in the selected private task-artifact directory,
not the source checkout or original-document folders. No provider call,
extraction run, deployment or reprocessing is performed by the sampler or report
generator.

Document identity includes both profile and document stem. Unknown scores remain
unavailable. Selection counts, review-record counts and substantive correctness
are different claims. The report does not infer population confidence or
production readiness from sample size or average grades. Earlier example
reports are historical evidence and do not establish current quality.

- [Sampling policy and limits](references/sampling_methodology.md)
- [Verification checklist](references/verification_checklist.md)
- [Medical terminology guide](references/medical_terminology_guide.md)
- [Report review guide](references/report_template.md)

Run isolated synthetic-fixture checks from the repository root:

```bash
python3 -m unittest discover -s .agents/skills/parsemedicalexams_extraction-quality-audit/tests -v
```

[MIT License](../../../LICENSE).
