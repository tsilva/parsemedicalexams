"""Aggressive summarization of medical exam transcriptions."""

import hashlib
import json
import logging
from pathlib import Path
from openai import OpenAI

from utils import load_prompt

logger = logging.getLogger(__name__)

# Cache directory for summarization results
CACHE_DIR = Path("config/cache")


def load_cache(name: str) -> dict:
    """Load JSON cache file."""
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        try:
            return json.load(open(path, encoding='utf-8'))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache {name}: {e}")
    return {}


def save_cache(name: str, cache: dict):
    """Save cache to JSON."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, sort_keys=True)


def summarize_exam(
    transcription: str,
    exam_type: str,
    exam_name: str,
    model_id: str,
    client: OpenAI
) -> str:
    """
    Generate aggressive summary keeping only findings, impressions, recommendations.

    Uses caching based on hash of transcription to avoid re-summarizing identical text.

    Args:
        transcription: Full text transcription of the exam
        exam_type: Type of exam (imaging, ultrasound, endoscopy, other)
        exam_name: Name of the specific exam
        model_id: Model to use for summarization
        client: OpenAI client instance

    Returns:
        Aggressive summary containing only clinical findings, impressions, and recommendations
    """
    if not transcription or not transcription.strip():
        return ""

    # Load cache (keyed by hash of transcription)
    cache = load_cache("summarization")
    cache_key = hashlib.md5(transcription.encode()).hexdigest()

    if cache_key in cache:
        logger.debug(f"Using cached summary for hash {cache_key[:8]}...")
        return cache[cache_key]

    # Load prompts
    system_prompt = load_prompt("summarization_system")
    user_prompt_template = load_prompt("summarization_user")
    user_prompt = user_prompt_template.format(
        exam_type=exam_type,
        exam_name=exam_name,
        transcription=transcription
    )

    # LLM call
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        if not completion or not completion.choices:
            logger.error("Invalid completion response for summarization")
            return ""

        summary = completion.choices[0].message.content.strip()

        # Cache result
        cache[cache_key] = summary
        save_cache("summarization", cache)

        return summary

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return ""


def batch_summarize_exams(
    exams: list[dict],
    model_id: str,
    client: OpenAI
) -> list[dict]:
    """
    Add summary field to all exams.

    Args:
        exams: List of exam dictionaries with transcription, exam_type, and exam_name_standardized
        model_id: Model to use for summarization
        client: OpenAI client instance

    Returns:
        List of exam dictionaries with summary field added
    """
    for i, exam in enumerate(exams):
        transcription = exam.get("transcription", "")
        exam_type = exam.get("exam_type", "other")
        exam_name = exam.get("exam_name_standardized") or exam.get("exam_name_raw", "Unknown")

        if transcription:
            logger.info(f"Summarizing exam {i+1}/{len(exams)}: {exam_name}")
            exam["summary"] = summarize_exam(
                transcription=transcription,
                exam_type=exam_type,
                exam_name=exam_name,
                model_id=model_id,
                client=client
            )
        else:
            exam["summary"] = ""

    return exams
