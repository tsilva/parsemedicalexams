"""Reject model and transport failures before creating clinical summaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from parsemedicalexams.extraction import validate_transcription
from parsemedicalexams.pipeline import discover_pdf_files
from parsemedicalexams.validation import (
    first_blocking_issue,
    validate_page_output,
    validate_source_units,
    validate_summary_output,
)


@pytest.mark.parametrize(
    "text",
    [
        'API Error: 400 {"error":{"message":"Image dimensions exceed max allowed size"}}',
        "I'm not going to transcribe this document. It contains private medical records.",
        "Not processing private medical documents. Ready to help with software engineering tasks.",
        "The conversation summary indicates I was processing medical documents "
        "through an automated pipeline.",
    ],
)
def test_failures_are_blocked_in_pages_and_summaries(text):
    assert first_blocking_issue(validate_page_output(text))
    assert first_blocking_issue(validate_summary_output(text))


@pytest.mark.parametrize("answer", ["", "uncertain", "no, but perhaps yes"])
def test_refusal_check_requires_unambiguous_no(answer):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    assert not validate_transcription(
        "Clinical result: normal examination, without significant findings.", "synthetic", client
    )[0]


def test_explicit_nested_document_bypasses_filename_filter(tmp_path):
    nested = tmp_path / "misc"
    nested.mkdir()
    source = nested / "undated prescription.pdf"
    source.touch()
    assert discover_pdf_files(tmp_path, r"^\d{4}-.*\.pdf$", source.name) == [source]


def test_discovery_filter_is_case_insensitive(tmp_path):
    source = tmp_path / "2024-01-01 - Analises.pdf"
    source.touch()
    assert discover_pdf_files(tmp_path, r"^\d{4}-(?!.*analises).*\.pdf$") == []


def test_transport_failure_does_not_retry_a_different_transcription_prompt(tmp_path, monkeypatch):
    import httpx
    from openai import APIError

    from parsemedicalexams.extraction import transcribe_with_retry

    transcribe = Mock(
        side_effect=APIError(
            "synthetic transport error",
            request=httpx.Request("POST", "https://example.test"),
            body=None,
        )
    )
    monkeypatch.setattr("parsemedicalexams.extraction.transcribe_page", transcribe)
    with pytest.raises(APIError):
        transcribe_with_retry(tmp_path / "synthetic.jpg", "synthetic", Mock(), "synthetic")
    assert transcribe.call_count == 1


def test_discovery_includes_uppercase_pdf_extension_in_nested_sources(tmp_path):
    nested = tmp_path / "misc"
    nested.mkdir()
    source = nested / "Undated prescription.PDF"
    source.touch()
    assert discover_pdf_files(tmp_path, r".*\.pdf$") == [source]


def test_source_units_block_thousandfold_mass_scale_change():
    source = "Mineral A 120 mcg/g(ppm)\nMineral B 15 mcg/g(ppm)"
    issue = first_blocking_issue(validate_source_units(source, "Mineral A 120 mg/g(ppm)"))
    assert issue and issue.kind == "source_unit_mismatch"
    assert not validate_source_units(source, "Mineral A 120 µg/g (ppm)")


def test_source_units_preserve_mixed_printed_scales_and_unknown_source():
    assert not validate_source_units("A 1 mg/mL; B 2 mcg/mL", "A 1 mg/mL; B 2 µg/mL")
    assert not validate_source_units("", "A 1 mg/g")
    assert not validate_source_units("Weight 1 g/kg", "A 1 mg/mL")


@pytest.mark.parametrize("bad_stage", ["page", "summary"])
def test_cached_outputs_reject_source_mass_scale_change(tmp_path, monkeypatch, bad_stage):
    from PIL import Image

    import parsemedicalexams.document_io as document_io
    from parsemedicalexams.models import ExamRecord

    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"synthetic source")
    output = tmp_path / "out"
    folder = output / source.stem
    folder.mkdir(parents=True)
    document_io.copy_source_pdf(source, folder)
    Image.new("RGB", (10, 10)).save(folder / "synthetic.001.jpg")
    text = (
        "Mineral A: 120 mcg/g (ppm). The source reports the concentration in micrograms per gram."
    )
    exam = ExamRecord(
        exam_name_raw="Synthetic mineral panel",
        exam_name_standardized="Synthetic mineral panel",
        exam_date=None,
        exam_type="other",
        transcription=text,
        page_number=1,
        source_file=source.name,
        prompt_variant="transcription_system",
    )
    if bad_stage == "page":
        exam.transcription = text.replace("mcg/g", "mg/g")
    document_io.save_transcription_file([exam], folder, source.stem, 1)
    document_io.save_document_summary(text.replace("mcg/g", "mg/g"), folder, source.stem, [exam])
    monkeypatch.setattr(document_io, "count_pdf_pages", lambda path: 1)
    monkeypatch.setattr(document_io, "extract_pdf_page_text", lambda path, page: text)
    issue = document_io.get_document_output_issue(source, output)
    assert issue and "source unit mismatch" in issue
    assert ("summary" in issue) == (bad_stage == "summary")


def test_unknown_clinical_date_survives_scan_filename_birth_and_validity_dates(
    tmp_path, monkeypatch
):
    import parsemedicalexams.pipeline as pipeline

    pdf = tmp_path / "Scanned_20260102.pdf"
    pdf.write_bytes(b"synthetic PDF")
    image = tmp_path / "synthetic.jpg"
    captured = []
    monkeypatch.setattr(
        pipeline,
        "preprocess_pdf_images_to_temp",
        lambda *args: (
            SimpleNamespace(cleanup=lambda: None),
            [image],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "classify_document",
        lambda *args, **kwargs: SimpleNamespace(
            is_exam=True,
            exam_name_raw="Undated referral",
            exam_date=None,
            facility_name=None,
            physician_name=None,
            department=None,
        ),
    )
    monkeypatch.setattr(pipeline, "copy_source_pdf", lambda *args: None)
    monkeypatch.setattr(pipeline, "persist_temp_images", lambda *args: [image])
    monkeypatch.setattr(pipeline, "extract_pdf_page_text", lambda *args: "")
    monkeypatch.setattr(
        pipeline,
        "transcribe_with_retry",
        lambda **kwargs: (
            "Undated referral. Birth date: 1960-01-01. Valid until: 2027-02-02. "
            "A consultation was requested; completion is not recorded.",
            "synthetic",
            1,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "standardize_exam_types",
        lambda *args: {
            "Undated referral": ("appointment", "Undated referral"),
        },
    )
    monkeypatch.setattr(
        pipeline, "save_transcription_file", lambda exams, *args: captured.extend(exams)
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_document",
        lambda *args, **kwargs: (
            "An undated referral requests a consultation. Completion is not recorded."
        ),
    )
    monkeypatch.setattr(pipeline, "save_document_summary", lambda *args: None)
    config = SimpleNamespace(
        dry_run=False,
        extract_model_id="synthetic",
        n_extractions=1,
        max_workers=1,
        validation_model_id="synthetic",
        summarize_model_id="synthetic",
        summarize_max_input_tokens=1000,
    )
    assert pipeline.process_single_pdf(pdf, tmp_path / "out", config, object()) == 1
    assert len(captured) == 1 and captured[0].exam_date is None
