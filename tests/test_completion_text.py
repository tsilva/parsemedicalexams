from types import SimpleNamespace

import pytest

from parsemedicalexams.extraction import (
    score_transcription_confidence,
    validate_transcription,
    vote_on_best_result,
)
from parsemedicalexams.standardization import standardize_exam_types
from parsemedicalexams.summarization import _llm_summarize
from parsemedicalexams.utils import extract_completion_text


def make_completion(content=None, include_choices=True, include_message=True):
    if not include_choices:
        return SimpleNamespace(choices=[])

    if not include_message:
        choice = SimpleNamespace()
    else:
        choice = SimpleNamespace(message=SimpleNamespace(content=content))

    return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, completion):
        self._completion = completion
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        return self._completion


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
