import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import parsemedicalexams.pipeline as pipeline
from parsemedicalexams.config import ExtractionConfig


def make_args(**overrides):
    args = SimpleNamespace(
        audit_outputs=False,
        resummarize=False,
        regenerate=False,
        dry_run=False,
        document=None,
        reprocess_all=False,
        model=None,
        workers=None,
        pattern=None,
        page=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_runtime_config(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    return ExtractionConfig(
        input_path=input_path,
        input_file_regex=r".*\.pdf",
        output_path=output_path,
        self_consistency_model_id="extract-model",
        extract_model_id="extract-model",
        summarize_model_id="summary-model",
        n_extractions=1,
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.com",
        validation_model_id="validation-model",
        max_workers=1,
        summarize_max_input_tokens=1000,
    )


def patch_profile_loading(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: test\n", encoding="utf-8")
    profile = SimpleNamespace(
        name="test",
        source_path=profile_path,
        full_name=None,
        birth_date=None,
        locale=None,
    )
    config = make_runtime_config(tmp_path)

    monkeypatch.setattr(
        pipeline.ProfileConfig,
        "find_profile",
        classmethod(lambda cls, profile_name: profile_path),
    )
    monkeypatch.setattr(
        pipeline.ProfileConfig,
        "from_file",
        classmethod(lambda cls, path: profile),
    )
    monkeypatch.setattr(
        pipeline.ExtractionConfig,
        "from_profile",
        classmethod(lambda cls, loaded_profile: config),
    )
    monkeypatch.setattr(pipeline, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "tqdm", lambda items, **kwargs: items)
    return config


def test_resolve_run_mode_prefers_explicit_modes():
    assert pipeline.resolve_run_mode(make_args(audit_outputs=True)) == pipeline.RunMode.AUDIT
    assert pipeline.resolve_run_mode(make_args(resummarize=True)) == pipeline.RunMode.RESUMMARIZE
    assert pipeline.resolve_run_mode(make_args(regenerate=True)) == pipeline.RunMode.REGENERATE
    assert pipeline.resolve_run_mode(make_args(dry_run=True)) == pipeline.RunMode.DRY_RUN
    assert pipeline.resolve_run_mode(make_args()) == pipeline.RunMode.PROCESS


def test_match_requested_document_matches_case_insensitive_stem(tmp_path):
    pdf_path = tmp_path / "Exam_2024.pdf"
    pdf_path.write_bytes(b"pdf")

    matches = pipeline.match_requested_document([pdf_path], "exam_2024")

    assert matches == [pdf_path]


def test_discover_pdf_files_raises_for_unreadable_input_dir(tmp_path, monkeypatch):
    original_iterdir = Path.iterdir

    def guarded_iterdir(self):
        if self == tmp_path:
            raise PermissionError("permission denied")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    with pytest.raises(PermissionError, match="Input path is not readable"):
        pipeline.discover_pdf_files(tmp_path, r".*\.pdf")


def test_select_documents_to_process_skips_complete_outputs(tmp_path, monkeypatch):
    processed = tmp_path / "done.pdf"
    pending = tmp_path / "todo.pdf"
    processed.write_bytes(b"done")
    pending.write_bytes(b"todo")

    monkeypatch.setattr(
        pipeline,
        "get_document_output_issue",
        lambda pdf_path, output_path: None if pdf_path == processed else "missing document summary",
    )

    to_process, already_processed = pipeline.select_documents_to_process(
        [processed, pending],
        tmp_path / "out",
        document=None,
        reprocess_all=False,
    )

    assert to_process == [pending]
    assert already_processed == 1


def test_run_profile_dry_run_uses_process_loop(tmp_path, monkeypatch):
    config = patch_profile_loading(monkeypatch, tmp_path)
    calls = []
    pdf_path = config.input_path / "exam.pdf"
    pdf_path.write_bytes(b"pdf")

    monkeypatch.setattr(pipeline, "discover_pdf_files", lambda *args, **kwargs: [pdf_path])
    monkeypatch.setattr(
        pipeline,
        "select_documents_to_process",
        lambda *args, **kwargs: ([pdf_path], 0),
    )
    monkeypatch.setattr(
        pipeline,
        "process_single_pdf",
        lambda *args, **kwargs: calls.append(kwargs["profile_context"]) or 2,
    )

    assert pipeline.run_profile("test", make_args(dry_run=True)) is True
    assert config.dry_run is True
    assert len(calls) == 1


def test_run_profile_returns_false_for_unreadable_input_dir(tmp_path, monkeypatch):
    patch_profile_loading(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline,
        "discover_pdf_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("Input path is not readable")),
    )

    assert pipeline.run_profile("test", make_args()) is False


def test_run_profile_audit_uses_output_validation(tmp_path, monkeypatch):
    config = patch_profile_loading(monkeypatch, tmp_path)
    pdf_path = config.input_path / "exam.pdf"
    pdf_path.write_bytes(b"pdf")
    audit_calls = []

    monkeypatch.setattr(pipeline, "discover_pdf_files", lambda *args, **kwargs: [pdf_path])
    monkeypatch.setattr(
        pipeline,
        "log_output_assertions_report",
        lambda pdf_files, output_path, input_path, label="": audit_calls.append(
            (pdf_files, output_path, input_path, label)
        )
        or True,
    )
    monkeypatch.setattr(
        pipeline,
        "process_single_pdf",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not process")),
    )

    assert pipeline.run_profile("test", make_args(audit_outputs=True)) is True
    assert audit_calls == [([pdf_path], config.output_path, config.input_path, "Output audit")]


def test_run_profile_regenerate_uses_summary_regeneration(tmp_path, monkeypatch):
    config = patch_profile_loading(monkeypatch, tmp_path)
    calls = []

    monkeypatch.setattr(
        pipeline,
        "regenerate_summaries",
        lambda output_path, config, client, input_path, doc_filter=None: calls.append(
            (output_path, doc_filter)
        )
        or 3,
    )
    pdf_path = config.input_path / "exam.pdf"
    pdf_path.write_bytes(b"pdf")
    monkeypatch.setattr(
        pipeline,
        "discover_pdf_files",
        lambda *args, **kwargs: [pdf_path],
    )
    monkeypatch.setattr(
        pipeline,
        "log_output_assertions_report",
        lambda *args, **kwargs: True,
    )

    assert pipeline.run_profile("test", make_args(regenerate=True)) is True
    assert len(calls) == 1
    assert calls[0][1] is None


def test_run_profile_normal_processes_and_validates_outputs(tmp_path, monkeypatch):
    config = patch_profile_loading(monkeypatch, tmp_path)
    pdf_path = config.input_path / "exam.pdf"
    pdf_path.write_bytes(b"pdf")
    processed = []

    monkeypatch.setattr(pipeline, "discover_pdf_files", lambda *args, **kwargs: [pdf_path])
    monkeypatch.setattr(
        pipeline,
        "select_documents_to_process",
        lambda *args, **kwargs: ([pdf_path], 0),
    )
    monkeypatch.setattr(
        pipeline,
        "process_single_pdf",
        lambda pdf_path, *args, **kwargs: processed.append(pdf_path) or 1,
    )
    monkeypatch.setattr(
        pipeline,
        "log_output_assertions_report",
        lambda *args, **kwargs: True,
    )

    assert pipeline.run_profile("test", make_args()) is True
    assert processed == [pdf_path]


def test_process_single_pdf_skips_non_exam_without_reason_attribute(tmp_path, monkeypatch):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(b"pdf")
    output_path = tmp_path / "out"
    output_path.mkdir()
    config = make_runtime_config(tmp_path)
    config.output_path = output_path

    image_path = tmp_path / "invoice.001.jpg"
    image_path.write_bytes(b"jpg")
    skip_calls = []

    class ClassificationWithoutReason:
        is_exam = False

    monkeypatch.setattr(
        pipeline,
        "preprocess_pdf_images_to_temp",
        lambda *args, **kwargs: (SimpleNamespace(cleanup=lambda: None), [image_path]),
    )
    monkeypatch.setattr(
        pipeline,
        "classify_document",
        lambda *args, **kwargs: ClassificationWithoutReason(),
    )
    monkeypatch.setattr(pipeline, "purge_derived_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "copy_source_pdf", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "write_skip_marker",
        lambda doc_output_dir, pdf_name, reason: skip_calls.append((doc_output_dir, pdf_name, reason)),
    )

    result = pipeline.process_single_pdf(
        pdf_path,
        output_path,
        config,
        client=object(),
    )

    assert result == "skipped"
    assert skip_calls == [
        (
            output_path / "invoice",
            "invoice.pdf",
            "classified as non-medical document",
        )
    ]


def test_run_profile_validates_all_discovered_documents_including_skipped(
    tmp_path, monkeypatch
):
    config = patch_profile_loading(monkeypatch, tmp_path)
    processed_pdf = config.input_path / "exam.pdf"
    skipped_pdf = config.input_path / "skip.pdf"
    processed_pdf.write_bytes(b"pdf")
    skipped_pdf.write_bytes(b"pdf")
    audit_calls = []

    monkeypatch.setattr(
        pipeline,
        "discover_pdf_files",
        lambda *args, **kwargs: [processed_pdf, skipped_pdf],
    )
    monkeypatch.setattr(
        pipeline,
        "select_documents_to_process",
        lambda *args, **kwargs: ([processed_pdf, skipped_pdf], 0),
    )
    monkeypatch.setattr(
        pipeline,
        "process_single_pdf",
        lambda pdf_path, *args, **kwargs: 1 if pdf_path == processed_pdf else "skipped",
    )
    monkeypatch.setattr(
        pipeline,
        "log_output_assertions_report",
        lambda pdf_files, output_path, input_path, label="": audit_calls.append(
            (pdf_files, output_path, input_path, label)
        )
        or True,
    )

    assert pipeline.run_profile("test", make_args()) is True
    assert audit_calls == [
        ([processed_pdf, skipped_pdf], config.output_path, config.input_path, "Post-extraction validation")
    ]


def test_run_profile_returns_false_when_post_extraction_validation_fails(
    tmp_path, monkeypatch
):
    config = patch_profile_loading(monkeypatch, tmp_path)
    pdf_path = config.input_path / "exam.pdf"
    pdf_path.write_bytes(b"pdf")

    monkeypatch.setattr(pipeline, "discover_pdf_files", lambda *args, **kwargs: [pdf_path])
    monkeypatch.setattr(
        pipeline,
        "select_documents_to_process",
        lambda *args, **kwargs: ([pdf_path], 0),
    )
    monkeypatch.setattr(pipeline, "process_single_pdf", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        pipeline,
        "log_output_assertions_report",
        lambda *args, **kwargs: False,
    )

    assert pipeline.run_profile("test", make_args()) is False
