"""Configuration management for medical exams parser."""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

UNKNOWN_VALUE = "$UNKNOWN$"


def _is_running_in_docker() -> bool:
    """Detect if running inside a Docker container."""
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def resolve_base_url(url: str) -> str:
    """Swap 127.0.0.1/localhost with host.docker.internal when running in Docker."""
    if _is_running_in_docker():
        url = url.replace("://127.0.0.1", "://host.docker.internal")
        url = url.replace("://localhost", "://host.docker.internal")
    return url


@dataclass
class ProfileConfig:
    """Configuration for a user profile.

    Supports both YAML and JSON formats. YAML is preferred.
    """
    name: str
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    input_file_regex: Optional[str] = None

    # Optional overrides
    model: Optional[str] = None
    workers: Optional[int] = None

    # Demographics (for extraction context)
    full_name: Optional[str] = None
    birth_date: Optional[str] = None  # YYYY-MM-DD
    gender: Optional[str] = None
    nationality: Optional[str] = None
    locale: Optional[str] = None  # e.g. "pt-PT"

    @classmethod
    def from_file(cls, profile_path: Path) -> 'ProfileConfig':
        """Load profile from YAML or JSON file."""
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        content = profile_path.read_text(encoding='utf-8')

        # Parse based on extension
        if profile_path.suffix in ('.yaml', '.yml'):
            data = yaml.safe_load(content)
        else:
            # Default to JSON for backwards compatibility
            data = json.load(open(profile_path, 'r', encoding='utf-8'))

        # Extract paths (support both flat and nested structures for backwards compatibility)
        paths = data.get('paths', {})
        input_path_str = paths.get('input_path') or data.get('input_path')
        output_path_str = paths.get('output_path') or data.get('output_path')
        input_file_regex = paths.get('input_file_regex') or data.get('input_file_regex')

        # Extract optional overrides
        model = data.get('model')
        workers = data.get('workers')

        return cls(
            name=data.get('name', profile_path.stem),
            input_path=Path(input_path_str) if input_path_str else None,
            output_path=Path(output_path_str) if output_path_str else None,
            input_file_regex=input_file_regex,
            model=model,
            workers=workers,
            full_name=data.get('full_name'),
            birth_date=data.get('birth_date'),
            gender=data.get('gender'),
            nationality=data.get('nationality'),
            locale=data.get('locale'),
        )

    @classmethod
    def list_profiles(cls, profiles_dir: Path = Path("profiles")) -> list[str]:
        """List available profile names."""
        if not profiles_dir.exists():
            return []
        profiles = []
        for ext in ('*.json', '*.yaml', '*.yml'):
            for f in profiles_dir.glob(ext):
                if not f.name.startswith('_'):  # Skip templates
                    profiles.append(f.stem)
        return sorted(set(profiles))


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline."""
    input_path: Optional[Path]
    input_file_regex: Optional[str]
    output_path: Optional[Path]
    self_consistency_model_id: str
    extract_model_id: str
    summarize_model_id: str
    n_extractions: int
    openrouter_api_key: str
    openrouter_base_url: str
    validation_model_id: str
    max_workers: int
    summarize_max_input_tokens: int

    @classmethod
    def from_env(cls) -> 'ExtractionConfig':
        """Load configuration from environment variables.

        Note: input_path, output_path, and input_file_regex can be None here
        if they will be provided by a profile.
        """
        input_path_str = os.getenv("INPUT_PATH")
        input_file_regex = os.getenv("INPUT_FILE_REGEX")
        output_path_str = os.getenv("OUTPUT_PATH")
        self_consistency_model_id = os.getenv("SELF_CONSISTENCY_MODEL_ID")
        extract_model_id = os.getenv("EXTRACT_MODEL_ID")
        summarize_model_id = os.getenv("SUMMARIZE_MODEL_ID")
        n_extractions = int(os.getenv("N_EXTRACTIONS", 1))
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        validation_model_id = os.getenv("VALIDATION_MODEL_ID", "anthropic/claude-haiku-4.5")
        max_workers_str = os.getenv("MAX_WORKERS", "1")
        summarize_max_input_tokens = int(os.getenv("SUMMARIZE_MAX_INPUT_TOKENS", "100000"))

        # Load base URL with fallback to OpenRouter
        openrouter_base_url = resolve_base_url(
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )

        # Validate required fields (paths can be provided by profile)
        if not self_consistency_model_id:
            raise ValueError("SELF_CONSISTENCY_MODEL_ID not set")
        if not extract_model_id:
            raise ValueError("EXTRACT_MODEL_ID not set")
        if not summarize_model_id:
            raise ValueError("SUMMARIZE_MODEL_ID not set")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # Parse max_workers
        try:
            max_workers = max(1, int(max_workers_str))
        except ValueError:
            logger.warning(f"MAX_WORKERS ('{max_workers_str}') is not valid. Defaulting to 1.")
            max_workers = 1

        # Parse paths (can be None if profile provides them)
        input_path = Path(input_path_str) if input_path_str else None
        output_path = Path(output_path_str) if output_path_str else None

        return cls(
            input_path=input_path,
            input_file_regex=input_file_regex,
            output_path=output_path,
            self_consistency_model_id=self_consistency_model_id,
            extract_model_id=extract_model_id,
            summarize_model_id=summarize_model_id,
            n_extractions=n_extractions,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url,
            validation_model_id=validation_model_id,
            max_workers=max_workers,
            summarize_max_input_tokens=summarize_max_input_tokens,
        )
