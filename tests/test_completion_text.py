import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_vote_on_best_result_falls_back_on_empty_response():
    client = FakeClient(make_completion(None))
    results = ["first", "second"]

    voted, all_results = vote_on_best_result(results, "fake-model", "transcribe_page", client)

    assert voted == "first"
    assert all_results == results


def test_score_transcription_confidence_returns_neutral_on_empty_response():
    client = FakeClient(make_completion(None))

    confidence = score_transcription_confidence(
        "merged",
        ["one", "two"],
        "fake-model",
        client,
    )

    assert confidence == 0.5


def test_standardize_exam_types_uses_defaults_on_empty_response(monkeypatch):
    client = FakeClient(make_completion(None))
    monkeypatch.setattr("parsemedicalexams.standardization.load_cache", lambda name: {})
    monkeypatch.setattr("parsemedicalexams.standardization.save_cache", lambda name, cache: None)

    result = standardize_exam_types(["Chest X-Ray"], "fake-model", client)

    assert result == {"Chest X-Ray": ("other", "Chest X-Ray")}


def test_llm_summarize_returns_empty_string_on_empty_response():
    client = FakeClient(make_completion(None))

    summary = _llm_summarize([{"role": "user", "content": "hello"}], "fake-model", client)

    assert summary == ""
