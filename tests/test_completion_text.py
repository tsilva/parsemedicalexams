import json
from types import SimpleNamespace

import pytest

from parsemedicalexams.extraction import (
    classify_document,
    score_transcription_confidence,
    validate_transcription,
    vote_on_best_result,
)
from parsemedicalexams.standardization import standardize_exam_types
from parsemedicalexams.summarization import _llm_summarize
from parsemedicalexams.utils import extract_completion_text


def make_completion(
    content=None,
    include_choices=True,
    include_message=True,
    tool_calls=None,
    finish_reason="stop",
):
    if not include_choices:
        return SimpleNamespace(choices=[])

    if not include_message:
        choice = SimpleNamespace()
    else:
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        )

    return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, completion):
        self._completions = completion if isinstance(completion, list) else [completion]
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self._completions) == 1:
            return self._completions[0]
        return self._completions.pop(0)


def make_tool_call(arguments):
    return [
        SimpleNamespace(
            function=SimpleNamespace(
                name="classify_document",
                arguments=json.dumps(arguments),
            )
        )
    ]


def test_extract_completion_text_returns_stripped_text():
    completion = make_completion("  hello world  ")

    assert extract_completion_text(completion, "test") == "hello world"


def test_extract_completion_text_handles_missing_content():
    completion = make_completion(None)

    assert extract_completion_text(completion, "test") == ""


def test_extract_completion_text_handles_empty_choices():
    completion = make_completion(include_choices=False)

    assert extract_completion_text(completion, "test") == ""


def test_extract_completion_text_handles_missing_message():
    completion = make_completion(include_message=False)

    assert extract_completion_text(completion, "test") == ""


def test_extract_completion_text_handles_non_string_content():
    completion = make_completion(["not", "a", "string"])

    assert extract_completion_text(completion, "test") == ""


def test_validate_transcription_allows_empty_refusal_response(caplog):
    client = FakeClient(make_completion(None))

    with caplog.at_level("WARNING"):
        is_valid, reason = validate_transcription(
            "This is a sufficiently long transcription payload.",
            "fake-model",
            client,
        )

    assert (is_valid, reason) == (True, "ok")
    assert "Empty refusal check response" in caplog.text


def test_classify_document_retries_missing_tool_call_for_gemini(tmp_path, caplog):
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake image")
    client = FakeClient(
        [
            make_completion(tool_calls=None, finish_reason="length"),
            make_completion(
                tool_calls=make_tool_call(
                    {
                        "is_exam": True,
                        "exam_name_raw": "Receita EZICLEN",
                        "reason": "prescription",
                    }
                ),
                finish_reason="tool_calls",
            ),
        ]
    )

    with caplog.at_level("WARNING"):
        classification = classify_document(
            [image_path],
            "google/gemini-3.7-flash",
            client,
        )

    assert classification.is_exam is True
    assert classification.exam_name_raw == "Receita EZICLEN"
    assert len(client.calls) == 2
    assert client.calls[0]["max_tokens"] == 1024
    assert "extra_body" not in client.calls[0]
    assert client.calls[1]["max_tokens"] == 4096
    assert client.calls[1]["extra_body"] == {"reasoning": {"effort": "minimal"}}
    assert "finish_reason=length" in caplog.text


def test_classify_document_reports_finish_reason_after_retry_failure(tmp_path):
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake image")
    client = FakeClient(
        [
            make_completion(tool_calls=None, finish_reason="length"),
            make_completion(tool_calls=None, finish_reason="length"),
        ]
    )

    with pytest.raises(RuntimeError, match="finish_reason=length"):
        classify_document(
            [image_path],
            "google/gemini-3.7-flash",
            client,
        )


def test_vote_on_best_result_raises_on_empty_response():
    client = FakeClient(make_completion(None))
    results = ["first", "second"]

    with pytest.raises(RuntimeError, match="Missing completion text"):
        vote_on_best_result(results, "fake-model", "transcribe_page", client)


def test_score_transcription_confidence_raises_on_empty_response():
    client = FakeClient(make_completion(None))

    with pytest.raises(RuntimeError, match="Missing completion text"):
        score_transcription_confidence(
            "merged",
            ["one", "two"],
            "fake-model",
            client,
        )


def test_standardize_exam_types_raises_on_empty_response(monkeypatch):
    client = FakeClient(make_completion(None))
    monkeypatch.setattr("parsemedicalexams.standardization.load_cache", lambda name: {})
    monkeypatch.setattr("parsemedicalexams.standardization.save_cache", lambda name, cache: None)

    with pytest.raises(RuntimeError, match="Missing completion text"):
        standardize_exam_types(["Chest X-Ray"], "fake-model", client)


def test_standardize_exam_types_rejects_invalid_exam_type_before_cache_write(monkeypatch):
    client = FakeClient(
        make_completion(
            '{"Chest X-Ray": {"exam_type": "invalid", "standardized_name": "Chest X-Ray"}}'
        )
    )
    saved = []
    monkeypatch.setattr("parsemedicalexams.standardization.load_cache", lambda name: {})
    monkeypatch.setattr(
        "parsemedicalexams.standardization.save_cache",
        lambda name, cache: saved.append((name, cache)),
    )

    with pytest.raises(ValueError, match="Invalid exam_type"):
        standardize_exam_types(["Chest X-Ray"], "fake-model", client)

    assert saved == []


def test_llm_summarize_raises_on_empty_response():
    client = FakeClient(make_completion(None))

    with pytest.raises(RuntimeError, match="Missing completion text"):
        _llm_summarize([{"role": "user", "content": "hello"}], "fake-model", client)
