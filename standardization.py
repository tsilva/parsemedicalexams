"""Exam type standardization using LLM with persistent cache."""

import json
import logging
from pathlib import Path
from openai import OpenAI

from utils import parse_llm_json_response, load_prompt

logger = logging.getLogger(__name__)

# Cache directory for LLM standardization results (user-editable JSON files)
CACHE_DIR = Path("config/cache")


def load_cache(name: str) -> dict:
    """Load JSON cache file. User-editable for overriding LLM decisions."""
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        try:
            return json.load(open(path, encoding='utf-8'))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache {name}: {e}")
    return {}


def save_cache(name: str, cache: dict):
    """Save cache to JSON, sorted alphabetically for easy editing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, sort_keys=True)


def standardize_exam_types(
    raw_exam_names: list[str],
    model_id: str,
    client: OpenAI
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

    # Load cache
    cache = load_cache("exam_type_standardization")

    # Get unique raw names
    unique_raw_names = list(set(raw_exam_names))

    # Split into cached and uncached
    def cache_key(name):
        return name.lower().strip()

    uncached_names = [n for n in unique_raw_names if cache_key(n) not in cache]

    # Call LLM only for uncached names
    if uncached_names:
        logger.info(f"[exam_type_standardization] {len(uncached_names)} uncached names, calling LLM...")

        system_prompt = load_prompt("standardization_system")
        user_prompt_template = load_prompt("standardization_user")
        user_prompt = user_prompt_template.format(
            exam_names=json.dumps(uncached_names, ensure_ascii=False, indent=2)
        )

        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            if not completion or not completion.choices:
                logger.error("Invalid completion response for exam standardization")
                llm_result = {}
            else:
                response_text = completion.choices[0].message.content.strip()
                llm_result = parse_llm_json_response(response_text, fallback={})

            # Update cache with LLM results
            for raw_name in uncached_names:
                if raw_name in llm_result:
                    result = llm_result[raw_name]
                    exam_type = result.get("exam_type", "other")
                    std_name = result.get("standardized_name", raw_name)
                    cache[cache_key(raw_name)] = {"exam_type": exam_type, "standardized_name": std_name}
                else:
                    logger.warning(f"LLM didn't return mapping for '{raw_name}', using raw name")
                    cache[cache_key(raw_name)] = {"exam_type": "other", "standardized_name": raw_name}

            save_cache("exam_type_standardization", cache)
            logger.info(f"[exam_type_standardization] Cache updated with {len(uncached_names)} entries")

        except Exception as e:
            logger.error(f"Error during exam standardization: {e}")
            # Fill in defaults for uncached names
            for raw_name in uncached_names:
                cache[cache_key(raw_name)] = {"exam_type": "other", "standardized_name": raw_name}

    # Return results for all names from cache
    result = {}
    for name in raw_exam_names:
        cached = cache.get(cache_key(name), {"exam_type": "other", "standardized_name": name})
        result[name] = (cached["exam_type"], cached["standardized_name"])

    return result
