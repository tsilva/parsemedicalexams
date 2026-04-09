"""Medical exam extraction from images using vision models."""

import json
import base64
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from openai import OpenAI, APIError

from .utils import (
    parse_llm_json_response,
    load_prompt,
    strip_markdown_fences,
    extract_dates_from_text,
    extract_completion_text,
)
from .validation import (
    first_blocking_issue,
    validate_page_output,
)

logger = logging.getLogger(__name__)


# ========================================
# Pydantic Models
# ========================================


class DocumentClassification(BaseModel):
    """Document classification result."""

    is_exam: bool = Field(
        description="True if the document contains medical exam results, clinical reports, or medical content that should be transcribed"
    )
    exam_name_raw: Optional[str] = Field(
        default=None,
        description="Document title or exam name exactly as written (e.g., 'CABELO: NUTRIENTES E METAIS TÓXICOS')",
    )
    exam_date: Optional[str] = Field(default=None, description="Exam date in YYYY-MM-DD format")
    facility_name: Optional[str] = Field(
        default=None,
        description="Healthcare facility name (e.g., 'SYNLAB', 'Hospital Santo António')",
    )
    physician_name: Optional[str] = Field(
        default=None,
        description="Name of the physician/doctor who performed or signed the exam",
    )
    department: Optional[str] = Field(
        default=None,
        description="Department or service within the facility (e.g., 'Radiologia', 'Cardiologia')",
    )


CLASSIFICATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_document",
            "description": "Classifies whether a document contains medical exam results, clinical reports, or other medical content that should be transcribed.",
            "parameters": DocumentClassification.model_json_schema(),
        },
    }
]


# ========================================
# Self-Consistency
# ========================================


def self_consistency(fn, model_id, n, *args, client=None, **kwargs):
    """Run fn n times and vote on the best result. Returns (best_result, all_results)."""
    if n == 1:
        result = fn(*args, **kwargs)
        return result, [result]

    results = []

    # Fixed temperature for i.i.d. sampling (aligned with self-consistency research)
    SELF_CONSISTENCY_TEMPERATURE = 0.5

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = []
        for i in range(n):
            effective_kwargs = kwargs.copy()
            # Use fixed temperature if function accepts it and not already set
            if "temperature" in fn.__code__.co_varnames and "temperature" not in kwargs:
                effective_kwargs["temperature"] = SELF_CONSISTENCY_TEMPERATURE
            futures.append(executor.submit(fn, *args, **effective_kwargs))

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error during self-consistency task execution: {e}")
                for f_cancel in futures:
                    if not f_cancel.done():
                        f_cancel.cancel()
                raise

    if not results:
        raise RuntimeError("All self-consistency calls failed.")

    # If all results are identical, return the first
    if all(r == results[0] for r in results):
        return results[0], results

    # Vote on best result using LLM
    return vote_on_best_result(results, model_id, fn.__name__, client=client)


def vote_on_best_result(results: list, model_id: str, fn_name: str, client: OpenAI):
    """Use LLM to vote on the most consistent result."""
    system_prompt = load_prompt("voting_system")

    prompt = "".join(
        f"--- Output {i + 1} ---\n{json.dumps(v, ensure_ascii=False) if type(v) in [list, dict] else v}\n\n"
        for i, v in enumerate(results)
    )

    try:
        completion = client.chat.completions.create(
            model=model_id,
            temperature=0.1,  # Low temperature for deterministic voting
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        voted_raw = extract_completion_text(completion, f"self-consistency voting for {fn_name}")
        if not voted_raw:
            logger.warning(f"Empty voting response for {fn_name}, falling back to first result")
            return results[0], results
        return voted_raw, results

    except Exception as e:
        logger.error(f"Error during self-consistency voting: {e}")
        return results[0], results


def classify_document(
    image_paths: List[Path],
    model_id: str,
    client: OpenAI,
    temperature: float = 0.1,
    profile_context: str = "",
) -> DocumentClassification:
    """
    Classify whether a document is a medical exam by analyzing all pages.

    Args:
        image_paths: List of paths to preprocessed page images
        model_id: Vision model to use for classification
        client: OpenAI client instance
        temperature: Temperature for sampling (low for classification)

    Returns:
        DocumentClassification with is_exam, exam_name_raw, exam_date, facility_name
    """
    # Build image content for all pages
    image_content = []
    for image_path in image_paths:
        with open(image_path, "rb") as img_file:
            img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
        image_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
            }
        )

    system_prompt = load_prompt("classification_system")
    system_prompt = system_prompt.format(patient_context=profile_context)
    user_prompt = load_prompt("classification_user")

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}, *image_content],
                },
            ],
            temperature=temperature,
            max_tokens=1024,
            tools=CLASSIFICATION_TOOLS,
            tool_choice={"type": "function", "function": {"name": "classify_document"}},
        )
        if not completion or not completion.choices or len(completion.choices) == 0:
            logger.error("Invalid completion response for classification")
            return DocumentClassification(is_exam=True)
        if not completion.choices[0].message.tool_calls:
            logger.warning("No tool call by model for document classification")
            return DocumentClassification(is_exam=True)
        tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
        tool_result_dict = json.loads(tool_args_raw)
        if tool_result_dict.get("exam_date"):
            tool_result_dict["exam_date"] = _normalize_date_format(tool_result_dict["exam_date"])
        return DocumentClassification(**tool_result_dict)
    except Exception as e:
        logger.error(f"Error during document classification: {e}")
        return DocumentClassification(is_exam=True)


def transcribe_page(
    image_path: Path,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.1,
    prompt_variant: str = "transcription_system",
    user_prompt_name: str = "transcription_user",
    user_prompt_text: str = "",
    profile_context: str = "",
) -> str:
    """
    Transcribe all visible text from a page verbatim.

    Args:
        image_path: Path to the preprocessed page image
        model_id: Vision model to use for transcription
        client: OpenAI client instance
        temperature: Temperature for sampling (low for OCR)
        prompt_variant: Which system prompt variant to use
        profile_context: Patient context string for prompt formatting

    Returns:
        String with complete verbatim transcription
    """
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    system_prompt = load_prompt(prompt_variant)
    system_prompt = system_prompt.format(patient_context=profile_context)
    user_prompt = user_prompt_text or load_prompt(user_prompt_name)

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=16384,
        )
    except APIError as e:
        logger.error(f"API Error during page transcription from {image_path.name}: {e}")
        return ""

    if not completion or not completion.choices or len(completion.choices) == 0:
        logger.error(f"Invalid completion response for transcription of {image_path.name}")
        return ""

    content = completion.choices[0].message.content
    if content is None:
        logger.warning(f"No content in response for transcription of {image_path.name}")
        return ""

    content = strip_markdown_fences(content.strip())

    # Try to parse as JSON and extract transcription field
    if content.startswith("{"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "transcription" in parsed:
                return parsed["transcription"]
        except json.JSONDecodeError:
            pass  # Not valid JSON, return as-is

    return content

def build_chart_user_prompt(chart_type: str, embedded_text: str) -> str:
    """Build chart extraction prompt with embedded-text context."""
    prompt = load_prompt("chart_transcription_user")
    embedded_excerpt = embedded_text.strip()
    if len(embedded_excerpt) > 3000:
        embedded_excerpt = embedded_excerpt[:3000]
    return prompt.format(chart_type=chart_type, embedded_text=embedded_excerpt or "[none]")


# Prompt variants for retry on transcription refusal
TRANSCRIPTION_PROMPT_VARIANTS = [
    "transcription_system",
    "transcription_system_alt1",
    "transcription_system_alt2",
    "transcription_system_alt3",
]


def transcribe_with_retry(
    image_path: Path,
    model_id: str,
    client: OpenAI,
    validation_model_id: str,
    temperature: float = 0.1,
    profile_context: str = "",
    max_retries: int = 3,
    page_kind: str = "text",
    chart_type: str | None = None,
    prompt_variants: list[str] | None = None,
    user_prompt_name: str = "transcription_user",
    user_prompt_text: str = "",
) -> tuple[str, str, int]:
    """
    Transcribe page with automatic retry on refusal using different prompt variants.

    Args:
        image_path: Path to the preprocessed page image
        model_id: Vision model to use for transcription
        client: OpenAI client instance
        validation_model_id: Model to use for refusal detection
        temperature: Temperature for sampling (low for OCR)
        profile_context: Patient context string for prompt formatting
        max_retries: Maximum number of prompt variants to try (default 3 = original + 2 alts)

    Returns:
        Tuple of (transcription_text, prompt_variant_used, attempts_made)
    """
    variants = prompt_variants or TRANSCRIPTION_PROMPT_VARIANTS
    transcription = ""
    last_prompt_variant = variants[0]
    for attempt, prompt_variant in enumerate(variants[: max_retries + 1]):
        try:
            logger.debug(
                f"Transcription attempt {attempt + 1} using {prompt_variant} for {image_path.name}"
            )
            last_prompt_variant = prompt_variant

            transcription = transcribe_page(
                image_path=image_path,
                model_id=model_id,
                client=client,
                temperature=temperature,
                prompt_variant=prompt_variant,
                user_prompt_name=user_prompt_name,
                user_prompt_text=user_prompt_text,
                profile_context=profile_context,
            )

            # Check if transcription is valid (not a refusal or other invalid output)
            is_valid, reason = validate_transcription(
                transcription,
                validation_model_id,
                client,
                page_kind=page_kind,
                chart_type=chart_type,
            )

            if is_valid:
                if attempt > 0:
                    logger.info(
                        f"Transcription succeeded with alternative prompt "
                        f"({prompt_variant}) on attempt {attempt + 1} for {image_path.name}"
                    )
                return transcription, prompt_variant, attempt + 1

            # Retry on any retryable invalid output
            logger.warning(
                f"Transcription validation failure ({reason}) with {prompt_variant} "
                f"for {image_path.name}, trying alternative prompt..."
            )

        except Exception as e:
            logger.error(f"Transcription failed with {prompt_variant} for {image_path.name}: {e}")
            continue

    # All variants exhausted - return the last transcription even if invalid
    logger.error(
        f"All {max_retries + 1} prompt variants failed for {image_path.name}. "
        f"Returning last transcription."
    )
    return transcription, last_prompt_variant, max_retries + 1


def validate_transcription(
    transcription: str,
    model_id: str,
    client: OpenAI,
    page_kind: str = "text",
    chart_type: str | None = None,
) -> tuple[bool, str]:
    """Returns (is_valid, reason). Uses local validation plus LLM refusal detection."""
    # Empty or too short
    if not transcription or len(transcription.strip()) < 20:
        return (False, "empty")

    issues = validate_page_output(
        transcription,
        page_kind=page_kind,
        chart_type=chart_type,
    )
    blocking_issue = first_blocking_issue(issues)
    if blocking_issue:
        return (False, blocking_issue.kind)

    # Use LLM to detect refusal
    prompt = (
        """You are checking if the following text is a refusal to transcribe medical content.

A refusal would be text where the model says it cannot or will not transcribe the medical document, mentions privacy concerns, or declines to process the request.

Text to check:
"""
        + transcription
        + """

Is this a refusal to transcribe medical content? Reply with only "yes" or "no"."""
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        result = extract_completion_text(response, "refusal check").lower()
        if not result:
            logger.warning("Empty refusal check response, assuming transcription is valid")
            return (True, "ok")
        if "yes" in result:
            return (False, "refusal")
    except Exception as e:
        logger.warning(f"Failed to check for refusal with LLM: {e}")
        # If LLM check fails, assume valid to avoid blocking
        pass

    return (True, "ok")


def _normalize_date_format(date_str: Optional[str]) -> Optional[str]:
    """Normalize date string to YYYY-MM-DD format."""
    if not date_str or date_str == "0000-00-00":
        return None
    dates = extract_dates_from_text(date_str)
    return dates[0] if dates else None


def score_transcription_confidence(
    merged_transcription: str,
    original_transcriptions: list[str],
    model_id: str,
    client: OpenAI,
) -> float:
    """
    Use LLM to assess confidence by comparing merged transcription against originals.

    Args:
        merged_transcription: The final voted/merged transcription
        original_transcriptions: List of original transcription attempts
        model_id: Model to use for confidence scoring
        client: OpenAI client instance

    Returns:
        Confidence score from 0.0 to 1.0
    """
    # If all originals are identical, confidence is 1.0
    if all(t == original_transcriptions[0] for t in original_transcriptions):
        return 1.0

    system_prompt = load_prompt("confidence_scoring_system")

    # Build comparison prompt
    prompt_parts = [f"## Final Merged Transcription:\n{merged_transcription}\n"]
    for i, orig in enumerate(original_transcriptions, 1):
        prompt_parts.append(f"## Original Transcription {i}:\n{orig}\n")

    user_prompt = "\n".join(prompt_parts)

    try:
        completion = client.chat.completions.create(
            model=model_id,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response = extract_completion_text(completion, "confidence scoring")
        if not response:
            logger.warning("Empty confidence scoring response, using neutral fallback")
            return 0.5

        # Parse JSON response
        result = parse_llm_json_response(response, fallback=None)
        if result and "confidence" in result:
            confidence = float(result["confidence"])
            # Clamp to valid range
            return max(0.0, min(1.0, confidence))
        else:
            logger.warning(f"Could not parse confidence response: {response[:100]}")
            return 0.5  # Default to neutral confidence

    except Exception as e:
        logger.error(f"Error during confidence scoring: {e}")
        return 0.5  # Default to neutral confidence
