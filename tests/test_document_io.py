import os
import shutil
from types import SimpleNamespace

from parsemedicalexams.document_io import (
    collect_output_assertions,
    extract_doc_date_prefix,
    frontmatter_to_exam,
    get_document_output_issue,
    parse_frontmatter,
    pdf_copy_is_current,
    regenerate_summaries,
    save_transcription_file,
    validate_frontmatter,
    validate_orphan_output_dirs,
    write_markdown_with_frontmatter,
)
from parsemedicalexams.models import ExamRecord


def make_exam(**overrides):
    exam = ExamRecord(
        exam_date="2024-01-15",
        exam_name_raw="RX TORAX PA Y LAT",
        exam_name_standardized="Chest X-Ray PA and Lateral",
        exam_type="imaging",
        physician_name="Dr. Smith",
        facility_name="Hospital Central",
        department="Radiology",
        page_number=1,
        source_file="exam.pdf",
        prompt_variant="transcription_system",
        page_kind="chart",
        validation_status="ok",
        failure_type=None,
        source_mode="hybrid",
        chart_type="audiogram",
        chart_data_status="ok",
        transcription_confidence=0.95,
        retry_attempts=2,
        transcription="Structured exam transcription body.",
    )
    for key, value in overrides.items():
        setattr(exam, key, value)
    return exam


def test_frontmatter_round_trip_preserves_extended_fields(tmp_path):
    doc_dir = tmp_path / "exam"
    doc_dir.mkdir()
    exam = make_exam()

    save_transcription_file([exam], doc_dir, "exam", 1)

    content = (doc_dir / "exam.001.md").read_text(encoding="utf-8")
    frontmatter, transcription = parse_frontmatter(content)
    rebuilt = frontmatter_to_exam(frontmatter, transcription, 1, "exam.pdf")

    assert rebuilt.transcription == exam.transcription
    assert rebuilt.page_kind == "chart"
    assert rebuilt.validation_status == "ok"
    assert rebuilt.source_mode == "hybrid"
    assert rebuilt.chart_type == "audiogram"
    assert rebuilt.chart_data_status == "ok"
    assert rebuilt.transcription_confidence == 0.95


def test_get_document_output_issue_detects_changed_source_pdf(tmp_path):
    source_pdf = tmp_path / "exam.pdf"
    source_pdf.write_bytes(b"new-source")

    output_path = tmp_path / "out"
    doc_dir = output_path / "exam"
    doc_dir.mkdir(parents=True)
    (doc_dir / "exam.pdf").write_bytes(b"old-source")

    assert (
        get_document_output_issue(source_pdf, output_path)
        == "source PDF changed since last processing"
    )


def test_get_document_output_issue_ignores_metadata_only_pdf_changes(tmp_path):
    source_pdf = tmp_path / "exam.pdf"
    source_pdf.write_bytes(b"same-source")

    output_path = tmp_path / "out"
    doc_dir = output_path / "exam"
    doc_dir.mkdir(parents=True)
    copied_pdf = doc_dir / "exam.pdf"
    shutil.copy2(source_pdf, copied_pdf)

    new_mtime = source_pdf.stat().st_mtime + 5
    os.utime(source_pdf, (new_mtime, new_mtime))

    (doc_dir / ".skip").write_text(
        "status: skipped\nsource: exam.pdf\nreason: not a medical exam\n",
        encoding="utf-8",
    )

    assert get_document_output_issue(source_pdf, output_path) is None


def test_pdf_copy_is_current_accepts_sub_microsecond_mtime_drift(tmp_path, monkeypatch):
    source_pdf = tmp_path / "exam.pdf"
    copied_pdf = tmp_path / "copied.pdf"
    source_pdf.write_bytes(b"same-source")
    copied_pdf.write_bytes(b"same-source")

    source_stat = source_pdf.stat()
    os.utime(copied_pdf, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 100))

    monkeypatch.setattr(
        "parsemedicalexams.document_io._files_have_same_content",
        lambda *_: (_ for _ in ()).throw(AssertionError("hash fallback should not run")),
    )

    assert pdf_copy_is_current(source_pdf, copied_pdf) is True


def test_get_document_output_issue_rejects_transcription_without_prompt_variant(
    tmp_path, monkeypatch
):
    source_pdf = tmp_path / "exam.pdf"
    source_pdf.write_bytes(b"source")
    monkeypatch.setattr("parsemedicalexams.document_io.count_pdf_pages", lambda _: 1)

    output_path = tmp_path / "out"
    doc_dir = output_path / source_pdf.stem
    doc_dir.mkdir(parents=True)
    shutil.copy2(source_pdf, doc_dir / source_pdf.name)
    (doc_dir / "exam.001.jpg").write_bytes(b"jpg")
    write_markdown_with_frontmatter(
        doc_dir / "exam.001.md",
        {
            "exam_date": "2024-01-15",
            "exam_name_raw": "Legacy exam",
            "title": "Legacy exam",
            "category": "other",
            "page": 1,
            "source": "exam.pdf",
        },
        "Legacy transcription body long enough to pass content checks.\n",
    )
    (doc_dir / "exam.summary.md").write_text(
        (
            "---\n"
            "exam_date: '2024-01-15'\n"
            "exam_name_raw: Summary\n"
            "title: Summary\n"
            "category: other\n"
            "---\n\n"
            "Legacy summary body long enough to satisfy summary validation.\n"
        ),
        encoding="utf-8",
    )

    assert (
        get_document_output_issue(source_pdf, output_path)
        == "invalid transcription output in exam.001.md: missing_prompt_variant"
    )


def test_extract_doc_date_prefix_reads_prefix():
    assert extract_doc_date_prefix("2025-08-22 - exame - questionario noite") == "2025-08-22"
    assert extract_doc_date_prefix("questionario noite") is None


def test_get_document_output_issue_detects_exam_date_mismatch_doc_prefix(tmp_path, monkeypatch):
    source_pdf = tmp_path / "2025-08-22 - exam.pdf"
    source_pdf.write_bytes(b"source")
    monkeypatch.setattr("parsemedicalexams.document_io.count_pdf_pages", lambda _: 1)

    output_path = tmp_path / "out"
    doc_dir = output_path / source_pdf.stem
    doc_dir.mkdir(parents=True)
    shutil.copy2(source_pdf, doc_dir / source_pdf.name)
    (doc_dir / f"{source_pdf.stem}.001.jpg").write_bytes(b"jpg")

    exam = make_exam(
        exam_date="1984-11-16",
        page_kind="text",
        chart_type=None,
        chart_data_status=None,
        source_file=source_pdf.name,
    )
    save_transcription_file([exam], doc_dir, source_pdf.stem, 1)
    (doc_dir / f"{source_pdf.stem}.summary.md").write_text(
        (
            "---\n"
            "exam_date: '2025-08-22'\n"
            "category: other\n"
            "title: Summary\n"
            "---\n\n"
            "Long enough summary body.\n"
        ),
        encoding="utf-8",
    )

    assert (
        get_document_output_issue(source_pdf, output_path)
        == "invalid transcription output in "
        f"{source_pdf.stem}.001.md: exam_date_mismatch_doc_prefix"
    )


def test_get_document_output_issue_accepts_skip_marker(tmp_path):
    source_pdf = tmp_path / "exam.pdf"
    source_pdf.write_bytes(b"source")

    output_path = tmp_path / "out"
    doc_dir = output_path / source_pdf.stem
    doc_dir.mkdir(parents=True)
    shutil.copy2(source_pdf, doc_dir / source_pdf.name)
    (doc_dir / ".skip").write_text(
        "status: skipped\nsource: exam.pdf\nreason: not a medical exam\n",
        encoding="utf-8",
    )

    assert get_document_output_issue(source_pdf, output_path) is None


def test_get_document_output_issue_detects_page_image_count_mismatch(tmp_path, monkeypatch):
    source_pdf = tmp_path / "exam.pdf"
    source_pdf.write_bytes(b"source")
    monkeypatch.setattr("parsemedicalexams.document_io.count_pdf_pages", lambda _: 2)

    output_path = tmp_path / "out"
    doc_dir = output_path / source_pdf.stem
    doc_dir.mkdir(parents=True)
    shutil.copy2(source_pdf, doc_dir / source_pdf.name)
    (doc_dir / "exam.001.jpg").write_bytes(b"jpg")

    exam = make_exam(
        page_kind="text",
        chart_type=None,
        chart_data_status=None,
        source_file=source_pdf.name,
    )
    save_transcription_file([exam], doc_dir, source_pdf.stem, 1)
    (doc_dir / "exam.summary.md").write_text(
        (
            "---\n"
            "exam_date: '2024-01-15'\n"
            "exam_name_raw: Summary\n"
            "title: Summary\n"
            "category: other\n"
            "---\n\n"
            "Long enough summary body.\n"
        ),
        encoding="utf-8",
    )

    assert (
        get_document_output_issue(source_pdf, output_path)
        == "page image count mismatch (1 images, 2 PDF pages)"
    )


def test_validate_orphan_output_dirs_detects_output_without_source_pdf(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "out"
    input_path.mkdir()
    output_path.mkdir()

    (input_path / "exam.pdf").write_bytes(b"pdf")
    (output_path / "exam").mkdir()
    (output_path / "orphan").mkdir()
    (output_path / "logs").mkdir()

    issues = validate_orphan_output_dirs(output_path, input_path)

    assert issues == ["orphan: output directory has no matching source PDF"]


def test_collect_output_assertions_groups_all_post_run_issues(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "out"
    input_path.mkdir()
    output_path.mkdir()

    source_pdf = input_path / "exam.pdf"
    source_pdf.write_bytes(b"pdf")
    (output_path / "orphan").mkdir()

    grouped = collect_output_assertions([source_pdf], output_path, input_path)

    assert grouped["output bundle issues"] == ["exam: missing output folder"]
    assert grouped["orphaned output directories"] == [
        "orphan: output directory has no matching source PDF"
    ]


def test_validate_frontmatter_detects_exam_date_mismatch_doc_prefix(tmp_path):
    output_path = tmp_path / "out"
    doc_stem = "2025-08-22 - exam"
    doc_dir = output_path / doc_stem
    doc_dir.mkdir(parents=True)

    exam = make_exam(
        exam_date="1984-11-16",
        page_kind="text",
        chart_type=None,
        chart_data_status=None,
        source_file=f"{doc_stem}.pdf",
    )
    save_transcription_file([exam], doc_dir, doc_stem, 1)

    issues = validate_frontmatter(output_path)

    assert (
        f"Exam date 1984-11-16 does not match document date prefix 2025-08-22: {doc_stem}.001.md"
        in issues
    )


def test_validate_frontmatter_detects_page_metadata_invariants(tmp_path):
    output_path = tmp_path / "out"
    doc_stem = "2025-08-22 - exam"
    doc_dir = output_path / doc_stem
    doc_dir.mkdir(parents=True)

    write_markdown_with_frontmatter(
        doc_dir / f"{doc_stem}.001.md",
        {
            "exam_date": "2025-08-22",
            "exam_name_raw": "Raw name",
            "title": "Title",
            "category": "not-a-category",
            "page": 2,
            "source": "wrong.pdf",
        },
        "Body text long enough to avoid output-level empty checks.\n",
    )

    issues = validate_frontmatter(output_path)

    assert f"invalid_category: {doc_stem}.001.md" in issues
    assert f"page_number_mismatch_filename: {doc_stem}.001.md" in issues
    assert f"source_mismatch_doc_pdf: {doc_stem}.001.md" in issues
    assert f"missing_prompt_variant: {doc_stem}.001.md" in issues


def test_validate_frontmatter_detects_invalid_enhanced_page_metadata(tmp_path):
    output_path = tmp_path / "out"
    doc_stem = "2025-08-22 - exam"
    doc_dir = output_path / doc_stem
    doc_dir.mkdir(parents=True)

    write_markdown_with_frontmatter(
        doc_dir / f"{doc_stem}.001.md",
        {
            "exam_date": "2025-08-22",
            "exam_name_raw": "Raw name",
            "title": "Title",
            "category": "other",
            "page": 1,
            "source": f"{doc_stem}.pdf",
            "prompt_variant": "chart_transcription_system",
            "page_kind": "text",
            "validation_status": "ok",
            "source_mode": "hybrid",
            "chart_type": "audiogram",
            "chart_data_status": "non_discrete_visual",
            "failure_type": "hard_failure",
        },
        "Body text long enough to avoid output-level empty checks.\n",
    )

    issues = validate_frontmatter(output_path)

    assert f"chart_type_without_chart_page: {doc_stem}.001.md" in issues
    assert f"discrete_chart_missing_ok_status: {doc_stem}.001.md" in issues
    assert f"ok_page_has_failure_type: {doc_stem}.001.md" in issues
    assert f"failure_type_without_failed_status: {doc_stem}.001.md" in issues


def test_validate_frontmatter_rejects_page_only_fields_on_summary(tmp_path):
    output_path = tmp_path / "out"
    doc_stem = "2025-08-22 - exam"
    doc_dir = output_path / doc_stem
    doc_dir.mkdir(parents=True)

    write_markdown_with_frontmatter(
        doc_dir / f"{doc_stem}.summary.md",
        {
            "exam_date": "2025-08-22",
            "exam_name_raw": "Raw name",
            "title": "Title",
            "category": "other",
            "page": 1,
            "source": f"{doc_stem}.pdf",
        },
        "Long enough summary body content for the metadata validator.\n",
    )

    issues = validate_frontmatter(output_path)

    assert f"summary_has_page_fields=['page', 'source']: {doc_stem}.summary.md" in issues


def test_regenerate_summaries_reads_markdown_and_writes_summary(tmp_path, monkeypatch):
    output_path = tmp_path / "out"
    doc_dir = output_path / "exam"
    doc_dir.mkdir(parents=True)
    save_transcription_file([make_exam(page_kind="text")], doc_dir, "exam", 1)

    monkeypatch.setattr(
        "parsemedicalexams.document_io.summarize_document",
        lambda exams, model_id, client, max_input_tokens: (
            "Comprehensive summary with enough length to pass validation."
        ),
    )

    config = SimpleNamespace(
        summarize_model_id="fake-model",
        summarize_max_input_tokens=1000,
    )

    total = regenerate_summaries(output_path, config, client=object())

    assert total == 1
    summary_path = doc_dir / "exam.summary.md"
    assert summary_path.exists()
    assert "Comprehensive summary" in summary_path.read_text(encoding="utf-8")
