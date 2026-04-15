"""Exam type standardization using LLM with persistent cache."""

import json
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import get_cache_dir
from .models import StandardizedExamEntry
from .utils import load_prompt, parse_json_mapping, require_completion_text

logger = logging.getLogger(__name__)

CACHE_DIR = get_cache_dir()


def load_cache(name: str) -> dict[str, StandardizedExamEntry]:
    """Load JSON cache file. User-editable for overriding LLM decisions."""
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                raw_cache = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache %s: %s", name, exc)
            return {}

        if isinstance(raw_cache, dict):
            cache: dict[str, StandardizedExamEntry] = {}
            for key, value in raw_cache.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                exam_type = value.get("exam_type")
                standardized_name = value.get("standardized_name")
                if isinstance(exam_type, str) and isinstance(standardized_name, str):
                    cache[key] = {
                        "exam_type": exam_type,
                        "standardized_name": standardized_name,
                    }
            return cache
    return {}


def save_cache(name: str, cache: dict[str, StandardizedExamEntry]) -> None:
    """Save cache to JSON, sorted alphabetically for easy editing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _default_entry(raw_name: str) -> StandardizedExamEntry:
    return {"exam_type": "other", "standardized_name": raw_name}


def _cache_key(name: str) -> str:
    return name.lower().strip()


def standardize_exam_types(
    raw_exam_names: list[str], model_id: str, client: OpenAI
) -> dict[str, tuple[str, str]]:
    """
    Map raw exam names to (exam_type, standardized_name) using LLM with cache.

    Args:
        raw_exam_names: List of raw exam names from extraction
        model_id: Model to use for standardization
        client: OpenAI client instance

    Returns:
        Dict mapping raw_name -> (exam_type, standardized_name)
    """
    if not raw_exam_names:
        return {}

    cache = load_cache("exam_type_standardization")
    unique_raw_names = list(set(raw_exam_names))
    uncached_names = [name for name in unique_raw_names if _cache_key(name) not in cache]
    if uncached_names:
        logger.info(
            "[exam_type_standardization] %s uncached names, calling LLM...",
            len(uncached_names),
        )

        system_prompt = load_prompt("standardization_system")
        user_prompt_template = load_prompt("standardization_user")
        user_prompt = user_prompt_template.format(
            exam_names=json.dumps(uncached_names, ensure_ascii=False, indent=2)
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=4000,
        )
        response_text = require_completion_text(completion, "exam standardization")
        llm_result = parse_json_mapping(response_text, "exam standardization")

        for raw_name in uncached_names:
            raw_entry = llm_result.get(raw_name)
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Missing standardization mapping for '{raw_name}'")
            exam_type = raw_entry.get("exam_type")
            standardized_name = raw_entry.get("standardized_name")
            if not isinstance(exam_type, str) or not isinstance(standardized_name, str):
                raise ValueError(f"Invalid standardization mapping for '{raw_name}'")
            cache[_cache_key(raw_name)] = {
                "exam_type": exam_type,
                "standardized_name": standardized_name,
            }

        save_cache("exam_type_standardization", cache)
        logger.info(
            "[exam_type_standardization] Cache updated with %s entries",
            len(uncached_names),
        )

    result: dict[str, tuple[str, str]] = {}
    for name in raw_exam_names:
        cached = cache.get(_cache_key(name), _default_entry(name))
        result[name] = (cached["exam_type"], cached["standardized_name"])

    return result
