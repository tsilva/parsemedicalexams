import pytest

from parsemedicalexams.cli import parse_args
from parsemedicalexams.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_N_EXTRACTIONS,
    DEFAULT_SUMMARIZE_MAX_INPUT_TOKENS,
    ExtractionConfig,
    ProfileConfig,
)


def test_parse_args_rejects_removed_page_option(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["medicalexamsparser", "--profile", "test", "--page", "2"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_profile_config_ignores_legacy_model_and_workers_keys(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: test",
                f"input_path: {input_path}",
                f"output_path: {output_path}",
                "model: legacy-model",
                "workers: 9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = ProfileConfig.from_file(profile_path)

    assert profile.extract_model_id is None
    assert profile.max_workers == 1
    assert profile.n_extractions == DEFAULT_N_EXTRACTIONS
    assert profile.summarize_max_input_tokens == DEFAULT_SUMMARIZE_MAX_INPUT_TOKENS


def test_extraction_config_uses_default_model_when_legacy_profile_aliases_removed(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: test",
                f"input_path: {input_path}",
                f"output_path: {output_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = ProfileConfig.from_file(profile_path)
    profile.openrouter_api_key = "test-key"

    extraction_config = ExtractionConfig.from_profile(profile)

    assert extraction_config.extract_model_id == DEFAULT_MODEL_ID
