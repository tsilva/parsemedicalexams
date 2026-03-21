"""Configuration management for medical exams parser."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

APP_NAME = "medicalexamsparser"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / APP_NAME
PROFILE_EXTENSIONS = (".yaml", ".yml", ".json")
DEFAULT_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_VALIDATION_MODEL_ID = "anthropic/claude-haiku-4.5"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_N_EXTRACTIONS = 1
DEFAULT_MAX_WORKERS = 1
DEFAULT_SUMMARIZE_MAX_INPUT_TOKENS = 100_000
DEFAULT_INPUT_FILE_REGEX = r".*\.pdf"


def _is_running_in_docker() -> bool:
    """Detect if running inside a Docker container."""
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def resolve_base_url(url: str) -> str:
    """Swap 127.0.0.1/localhost with host.docker.internal when running in Docker."""
    if _is_running_in_docker():
        url = url.replace("://127.0.0.1", "://host.docker.internal")
        url = url.replace("://localhost", "://host.docker.internal")
    return url


def get_config_dir() -> Path:
    """Return the global config directory used by the CLI."""
    return DEFAULT_CONFIG_DIR


def get_cache_dir() -> Path:
    """Return the global cache directory used by the CLI."""
    return get_config_dir() / "cache"


def ensure_config_dir() -> Path:
    """Create the global config directory if needed."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _load_profile_data(profile_path: Path) -> dict[str, Any]:
    """Load YAML or JSON profile content from disk."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    content = profile_path.read_text(encoding="utf-8")
    if profile_path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError(f"Profile must contain a mapping: {profile_path}")
    return data


def _first_value(*values: Any) -> Any:
    """Return the first non-empty value."""
    for value in values:
        if value is not None:
            return value
    return None


def _parse_positive_int(value: Any, default: int, field_name: str) -> int:
    """Parse a positive integer from profile data."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not valid. Using %s.", field_name, value, default)
        return default
    if parsed < 1:
        logger.warning("%s=%r is not valid. Using %s.", field_name, value, default)
        return default
    return parsed


def _resolve_profile_path(profile_path: Path, raw_path: Optional[str]) -> Optional[Path]:
    """Resolve a path declared inside a profile."""
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return profile_path.parent / path


@dataclass
class ProfileConfig:
    """Configuration for a self-contained user profile."""

    name: str
    source_path: Path
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    input_file_regex: Optional[str] = None

    openrouter_api_key: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    extract_model_id: Optional[str] = None
    summarize_model_id: Optional[str] = None
    self_consistency_model_id: Optional[str] = None
    validation_model_id: Optional[str] = None
    n_extractions: int = DEFAULT_N_EXTRACTIONS
    max_workers: int = DEFAULT_MAX_WORKERS
    summarize_max_input_tokens: int = DEFAULT_SUMMARIZE_MAX_INPUT_TOKENS

    # Convenience aliases
    model: Optional[str] = None
    workers: Optional[int] = None

    # Patient context
    full_name: Optional[str] = None
    birth_date: Optional[str] = None
    locale: Optional[str] = None

    @classmethod
    def config_dir(cls) -> Path:
        """Return the global profile/config directory."""
        return get_config_dir()

    @classmethod
    def find_profile(
        cls, profile_name: str, profiles_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Return the profile path for the given profile name."""
        search_dir = profiles_dir or cls.config_dir()
        for ext in PROFILE_EXTENSIONS:
            candidate = search_dir / f"{profile_name}{ext}"
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def from_file(cls, profile_path: Path) -> "ProfileConfig":
        """Load profile from YAML or JSON file."""
        data = _load_profile_data(profile_path)

        paths = data.get("paths", {}) if isinstance(data.get("paths"), dict) else {}
        models = data.get("models", {}) if isinstance(data.get("models"), dict) else {}
        processing = (
            data.get("processing", {})
            if isinstance(data.get("processing"), dict)
            else {}
        )
        patient = data.get("patient", {}) if isinstance(data.get("patient"), dict) else {}
        openrouter = (
            data.get("openrouter", {})
            if isinstance(data.get("openrouter"), dict)
            else {}
        )

        input_path_str = _first_value(paths.get("input_path"), data.get("input_path"))
        output_path_str = _first_value(paths.get("output_path"), data.get("output_path"))
        input_file_regex = _first_value(
            paths.get("input_file_regex"),
            data.get("input_file_regex"),
        )

        model = _first_value(models.get("model"), data.get("model"))
        workers = _first_value(
            processing.get("workers"),
            processing.get("max_workers"),
            data.get("workers"),
            data.get("max_workers"),
        )

        return cls(
            name=data.get("name", profile_path.stem),
            source_path=profile_path,
            input_path=_resolve_profile_path(profile_path, input_path_str),
            output_path=_resolve_profile_path(profile_path, output_path_str),
            input_file_regex=input_file_regex,
            openrouter_api_key=_first_value(
                openrouter.get("api_key"),
                openrouter.get("openrouter_api_key"),
                data.get("openrouter_api_key"),
            ),
            openrouter_base_url=resolve_base_url(
                _first_value(
                    openrouter.get("base_url"),
                    openrouter.get("openrouter_base_url"),
                    data.get("openrouter_base_url"),
                    DEFAULT_OPENROUTER_BASE_URL,
                )
            ),
            extract_model_id=_first_value(
                models.get("extract_model_id"),
                data.get("extract_model_id"),
            ),
            summarize_model_id=_first_value(
                models.get("summarize_model_id"),
                data.get("summarize_model_id"),
            ),
            self_consistency_model_id=_first_value(
                models.get("self_consistency_model_id"),
                data.get("self_consistency_model_id"),
            ),
            validation_model_id=_first_value(
                models.get("validation_model_id"),
                data.get("validation_model_id"),
                DEFAULT_VALIDATION_MODEL_ID,
            ),
            n_extractions=_parse_positive_int(
                _first_value(
                    processing.get("n_extractions"),
                    data.get("n_extractions"),
                ),
                DEFAULT_N_EXTRACTIONS,
                "n_extractions",
            ),
            max_workers=_parse_positive_int(
                workers,
                DEFAULT_MAX_WORKERS,
                "max_workers",
            ),
            summarize_max_input_tokens=_parse_positive_int(
                _first_value(
                    processing.get("summarize_max_input_tokens"),
                    data.get("summarize_max_input_tokens"),
                ),
                DEFAULT_SUMMARIZE_MAX_INPUT_TOKENS,
                "summarize_max_input_tokens",
            ),
            model=model,
            workers=_parse_positive_int(workers, DEFAULT_MAX_WORKERS, "workers")
            if workers is not None
            else None,
            full_name=_first_value(patient.get("full_name"), data.get("full_name")),
            birth_date=_first_value(patient.get("birth_date"), data.get("birth_date")),
            locale=_first_value(patient.get("locale"), data.get("locale")),
        )

    @classmethod
    def list_profiles(cls, profiles_dir: Optional[Path] = None) -> list[str]:
        """List available profile names from the global config directory."""
        search_dir = profiles_dir or cls.config_dir()
        if not search_dir.exists():
            return []

        profiles = []
        for ext in PROFILE_EXTENSIONS:
            for profile_path in search_dir.glob(f"*{ext}"):
                if not profile_path.name.startswith(("_", ".")):
                    profiles.append(profile_path.stem)
        return sorted(set(profiles))


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline."""

    input_path: Path
    input_file_regex: str
    output_path: Path
    self_consistency_model_id: str
    extract_model_id: str
    summarize_model_id: str
    n_extractions: int
    openrouter_api_key: str
    openrouter_base_url: str
    validation_model_id: str
    max_workers: int
    summarize_max_input_tokens: int
    dry_run: bool = False

    @classmethod
    def from_profile(cls, profile: ProfileConfig) -> "ExtractionConfig":
        """Build runtime configuration from a single self-contained profile."""
        if not profile.input_path:
            raise ValueError(f"Profile '{profile.name}' has no input_path defined.")
        if not profile.output_path:
            raise ValueError(f"Profile '{profile.name}' has no output_path defined.")
        if not profile.openrouter_api_key:
            raise ValueError(
                f"Profile '{profile.name}' has no openrouter_api_key defined."
            )

        default_model = profile.model or DEFAULT_MODEL_ID

        return cls(
            input_path=profile.input_path,
            input_file_regex=profile.input_file_regex or DEFAULT_INPUT_FILE_REGEX,
            output_path=profile.output_path,
            self_consistency_model_id=(
                profile.self_consistency_model_id or default_model
            ),
            extract_model_id=profile.extract_model_id or default_model,
            summarize_model_id=profile.summarize_model_id or default_model,
            n_extractions=profile.n_extractions,
            openrouter_api_key=profile.openrouter_api_key,
            openrouter_base_url=resolve_base_url(
                profile.openrouter_base_url or DEFAULT_OPENROUTER_BASE_URL
            ),
            validation_model_id=(
                profile.validation_model_id or DEFAULT_VALIDATION_MODEL_ID
            ),
            max_workers=profile.max_workers,
            summarize_max_input_tokens=profile.summarize_max_input_tokens,
        )
