"""Configuration management for medical exams parser."""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import yaml  # type: ignore[import-untyped]
from dotenv import dotenv_values

logger = logging.getLogger(__name__)
T = TypeVar("T")

APP_NAME = "parsemedicalexams"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / APP_NAME
LEGACY_CONFIG_DIR = Path.home() / ".config" / "medicalexamsparser"
PROFILE_EXTENSIONS = (".yaml", ".yml", ".json")
DEFAULT_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_VALIDATION_MODEL_ID = "anthropic/claude-haiku-4.5"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_N_EXTRACTIONS = 1
DEFAULT_MAX_WORKERS = 1
DEFAULT_SUMMARIZE_MAX_INPUT_TOKENS = 100_000
DEFAULT_INPUT_FILE_REGEX = r".*\.pdf"
ENV_FILENAME = ".env"
ENV_EXAMPLE_FILENAME = ".env.example"


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


def get_env_path() -> Path:
    """Return the shared .env path used by the CLI."""
    return get_config_dir() / ENV_FILENAME


def get_env_example_path() -> Path:
    """Return the shared .env example path used by the CLI."""
    return get_config_dir() / ENV_EXAMPLE_FILENAME


def ensure_config_dir() -> Path:
    """Create the global config directory if needed."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def migrate_profiles(*source_dirs: Path) -> list[Path]:
    """Move profile and example files into the global config directory."""
    config_dir = ensure_config_dir()
    moved_files: list[Path] = []
    candidate_extensions = PROFILE_EXTENSIONS + (".example",)

    for source_dir in source_dirs:
        if not source_dir.exists() or source_dir.resolve() == config_dir.resolve():
            continue

        for source_path in sorted(source_dir.iterdir()):
            if not source_path.is_file():
                continue

            if not any(source_path.name.endswith(ext) for ext in candidate_extensions):
                continue

            target_path = config_dir / source_path.name
            if target_path.exists():
                continue

            shutil.move(str(source_path), str(target_path))
            moved_files.append(target_path)

    return moved_files


def migrate_env_file(*source_dirs: Path) -> Path | None:
    """Move a real .env file into the global config directory if needed."""
    env_path = get_env_path()
    if env_path.exists():
        return env_path

    for source_dir in source_dirs:
        source_path = source_dir / ENV_FILENAME
        if not source_path.exists():
            continue
        ensure_config_dir()
        shutil.move(str(source_path), str(env_path))
        return env_path

    return None


def sync_example_file(source_path: Path, target_path: Path) -> Path:
    """Copy an example file into the config directory."""
    ensure_config_dir()
    shutil.copy2(source_path, target_path)
    return target_path


def load_shared_env(config_dir: Path | None = None) -> dict[str, str]:
    """Load shared model and credential settings from the config .env file."""
    env_path = (config_dir or get_config_dir()) / ENV_FILENAME
    if not env_path.exists():
        return {}
    return {key: value for key, value in dotenv_values(env_path).items() if value is not None}


def _load_profile_data(profile_path: Path) -> dict[str, object]:
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
    return dict(data)


def _first_value(*values: T | None) -> T | None:
    """Return the first non-empty value."""
    for value in values:
        if value is not None:
            return value
    return None


def _mapping_section(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _parse_positive_int(value: object, default: int, field_name: str) -> int:
    """Parse a positive integer from profile data."""
    if value is None:
        return default
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        logger.warning("%s=%r is not valid. Using %s.", field_name, value, default)
        return default
    if parsed < 1:
        logger.warning("%s=%r is not valid. Using %s.", field_name, value, default)
        return default
    return parsed


def _resolve_profile_path(profile_path: Path, raw_path: str | None) -> Path | None:
    """Resolve a path declared inside a profile."""
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return profile_path.parent / path


@dataclass
class ProfileConfig:
    """Configuration for a user profile plus shared .env defaults."""

    name: str
    source_path: Path
    input_path: Path | None = None
    output_path: Path | None = None
    input_file_regex: str | None = None

    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None
    extract_model_id: str | None = None
    summarize_model_id: str | None = None
    self_consistency_model_id: str | None = None
    validation_model_id: str | None = None
    n_extractions: int = DEFAULT_N_EXTRACTIONS
    max_workers: int = DEFAULT_MAX_WORKERS
    summarize_max_input_tokens: int = DEFAULT_SUMMARIZE_MAX_INPUT_TOKENS

    full_name: str | None = None
    birth_date: str | None = None
    locale: str | None = None

    @classmethod
    def config_dir(cls) -> Path:
        """Return the global profile/config directory."""
        return get_config_dir()

    @classmethod
    def find_profile(cls, profile_name: str, profiles_dir: Path | None = None) -> Path | None:
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

        paths = _mapping_section(data, "paths")
        models = _mapping_section(data, "models")
        processing = _mapping_section(data, "processing")
        patient = _mapping_section(data, "patient")
        openrouter = _mapping_section(data, "openrouter")

        input_path_str = _optional_str(
            _first_value(
                _optional_str(paths.get("input_path")),
                _optional_str(data.get("input_path")),
            )
        )
        output_path_str = _optional_str(
            _first_value(
                _optional_str(paths.get("output_path")),
                _optional_str(data.get("output_path")),
            )
        )
        input_file_regex = _optional_str(
            _first_value(
                _optional_str(paths.get("input_file_regex")),
                _optional_str(data.get("input_file_regex")),
            )
        )
        max_workers = _first_value(
            processing.get("max_workers"),
            data.get("max_workers"),
        )
        base_url = _optional_str(
            _first_value(
                _optional_str(openrouter.get("base_url")),
                _optional_str(openrouter.get("openrouter_base_url")),
                _optional_str(data.get("openrouter_base_url")),
            )
        )
        name = _optional_str(data.get("name")) or profile_path.stem

        return cls(
            name=name,
            source_path=profile_path,
            input_path=_resolve_profile_path(profile_path, input_path_str),
            output_path=_resolve_profile_path(profile_path, output_path_str),
            input_file_regex=input_file_regex,
            openrouter_api_key=_optional_str(
                _first_value(
                    _optional_str(openrouter.get("api_key")),
                    _optional_str(openrouter.get("openrouter_api_key")),
                    _optional_str(data.get("openrouter_api_key")),
                )
            ),
            openrouter_base_url=resolve_base_url(base_url) if base_url else None,
            extract_model_id=_optional_str(
                _first_value(
                    _optional_str(models.get("extract_model_id")),
                    _optional_str(data.get("extract_model_id")),
                )
            ),
            summarize_model_id=_optional_str(
                _first_value(
                    _optional_str(models.get("summarize_model_id")),
                    _optional_str(data.get("summarize_model_id")),
                )
            ),
            self_consistency_model_id=_optional_str(
                _first_value(
                    _optional_str(models.get("self_consistency_model_id")),
                    _optional_str(data.get("self_consistency_model_id")),
                )
            ),
            validation_model_id=_optional_str(
                _first_value(
                    _optional_str(models.get("validation_model_id")),
                    _optional_str(data.get("validation_model_id")),
                )
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
                max_workers,
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
            full_name=_optional_str(
                _first_value(
                    _optional_str(patient.get("full_name")),
                    _optional_str(data.get("full_name")),
                )
            ),
            birth_date=_optional_str(
                _first_value(
                    _optional_str(patient.get("birth_date")),
                    _optional_str(data.get("birth_date")),
                )
            ),
            locale=_optional_str(
                _first_value(
                    _optional_str(patient.get("locale")),
                    _optional_str(data.get("locale")),
                )
            ),
        )

    @classmethod
    def list_profiles(cls, profiles_dir: Path | None = None) -> list[str]:
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
        """Build runtime configuration from a profile plus shared .env settings."""
        if not profile.input_path:
            raise ValueError(f"Profile '{profile.name}' has no input_path defined.")
        if not profile.output_path:
            raise ValueError(f"Profile '{profile.name}' has no output_path defined.")
        shared_env = load_shared_env(profile.source_path.parent)

        openrouter_api_key = profile.openrouter_api_key or shared_env.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                f"Missing OPENROUTER_API_KEY in {get_env_path()}."
                f" Copy {get_env_example_path()} to {get_env_path()} and fill it in."
            )

        default_model = shared_env.get("EXTRACT_MODEL_ID") or DEFAULT_MODEL_ID

        return cls(
            input_path=profile.input_path,
            input_file_regex=profile.input_file_regex or DEFAULT_INPUT_FILE_REGEX,
            output_path=profile.output_path,
            self_consistency_model_id=(
                profile.self_consistency_model_id
                or shared_env.get("SELF_CONSISTENCY_MODEL_ID")
                or default_model
            ),
            extract_model_id=(
                profile.extract_model_id or shared_env.get("EXTRACT_MODEL_ID") or default_model
            ),
            summarize_model_id=(
                profile.summarize_model_id or shared_env.get("SUMMARIZE_MODEL_ID") or default_model
            ),
            n_extractions=profile.n_extractions,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=resolve_base_url(
                profile.openrouter_base_url
                or shared_env.get("OPENROUTER_BASE_URL")
                or DEFAULT_OPENROUTER_BASE_URL
            ),
            validation_model_id=(
                profile.validation_model_id
                or shared_env.get("VALIDATION_MODEL_ID")
                or DEFAULT_VALIDATION_MODEL_ID
            ),
            max_workers=profile.max_workers,
            summarize_max_input_tokens=profile.summarize_max_input_tokens,
        )
