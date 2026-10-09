"""Reject model and transport failures before creating clinical summaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from parsemedicalexams.extraction import validate_transcription
from parsemedicalexams.pipeline import discover_pdf_files
from parsemedicalexams.validation import (
    first_blocking_issue,
    validate_page_output,
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
