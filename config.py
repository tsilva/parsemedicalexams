"""Configuration management for medical exams parser."""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

UNKNOWN_VALUE = "$UNKNOWN$"


@dataclass
class ProfileConfig:
    """Configuration for a user profile."""
    name: str
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    input_file_regex: Optional[str] = None

    @classmethod
    def from_file(cls, profile_path: Path, env_config: Optional['ExtractionConfig'] = None) -> 'ProfileConfig':
        """Load profile from JSON file, inheriting from env_config where needed."""
        with open(profile_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse paths, with inheritance from env_config
        paths = data.get('paths', {})
        inherit = data.get('settings', {}).get('inherit_from_env', True)

        input_path = None
        if paths.get('input_path'):
            input_path = Path(paths['input_path'])
        elif inherit and env_config:
            input_path = env_config.input_path

        output_path = None
        if paths.get('output_path'):
            output_path = Path(paths['output_path'])
        elif inherit and env_config:
            output_path = env_config.output_path

        input_file_regex = paths.get('input_file_regex')
        if not input_file_regex and inherit and env_config:
            input_file_regex = env_config.input_file_regex

        return cls(
            name=data.get('name', profile_path.stem),
            input_path=input_path,
            output_path=output_path,
            input_file_regex=input_file_regex,
        )

    @classmethod
    def list_profiles(cls, profiles_dir: Path = Path("profiles")) -> list[str]:
        """List available profile names."""
        if not profiles_dir.exists():
            return []
        profiles = []
        for f in profiles_dir.glob("*.json"):
            if not f.name.startswith("_"):  # Skip templates like _template.json
                profiles.append(f.stem)
        return sorted(profiles)


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
    max_workers: int

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
        max_workers_str = os.getenv("MAX_WORKERS", "1")

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
            max_workers=max_workers,
        )
