"""Medical exam extraction from images using vision models."""

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, cast

from openai import APIError, OpenAI
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field

from .utils import (
    extract_completion_text,
    extract_dates_from_text,
    load_prompt,
    parse_json_mapping,
    require_completion_text,
    strip_markdown_fences,
)
from .validation import first_blocking_issue, validate_page_output

logger = logging.getLogger(__name__)


class DocumentClassification(BaseModel):
    """Document classification result."""

    is_exam: bool = Field(
        description=(
            "True if the document contains medical exam results, clinical reports, "
            "or medical content that should be transcribed"
        )
    )
    exam_name_raw: str | None = Field(
        default=None,
        description=(
            "Document title or exam name exactly as written "
            "(e.g., 'CABELO: NUTRIENTES E METAIS TÓXICOS')"
        ),
    )
    exam_date: str | None = Field(
        default=None,
        description="Exam date in YYYY-MM-DD format",
    )
    facility_name: str | None = Field(
        default=None,
        description="Healthcare facility name (e.g., 'SYNLAB', 'Hospital Santo António')",
    )
    physician_name: str | None = Field(
        default=None,
        description="Name of the physician/doctor who performed or signed the exam",
    )
    department: str | None = Field(
        default=None,
        description="Department or service within the facility (e.g., 'Radiologia', 'Cardiologia')",
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Short explanation for the classification decision, "
            "especially when is_exam is false"
        ),
    )


CLASSIFICATION_TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "classify_document",
            "description": (
                "Classifies whether a document contains medical exam results, "
                "clinical reports, or other medical content that should be transcribed."
            ),
            "parameters": cast(dict[str, object], DocumentClassification.model_json_schema()),
        },
    }
]
CLASSIFICATION_TOOL_CHOICE: ChatCompletionNamedToolChoiceParam = {
    "type": "function",
    "function": {"name": "classify_document"},
}

def _encode_image(image_path: Path) -> ChatCompletionContentPartImageParam:
    with image_path.open("rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
    }


def _parse_classification_tool_args(raw_args: str) -> dict[str, object]:
    parsed = json.loads(raw_args)
    if not isinstance(parsed, dict):
        raise ValueError("Classification tool call arguments must be a JSON object")
    return cast(dict[str, object], parsed)


def self_consistency(
    fn: Callable[..., str],
    model_id: str,
    n: int,
    *args: object,
    client: OpenAI,
    **kwargs: object,
) -> tuple[str, list[str]]:
    """Run fn n times and vote on the best result. Returns (best_result, all_results)."""
    if n == 1:
        result = fn(*args, **kwargs)
        return result, [result]

    results: list[str] = []
    SELF_CONSISTENCY_TEMPERATURE = 0.5

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = []
        for i in range(n):
            effective_kwargs = kwargs.copy()
            if "temperature" in fn.__code__.co_varnames and "temperature" not in kwargs:
                effective_kwargs["temperature"] = SELF_CONSISTENCY_TEMPERATURE
            futures.append(executor.submit(fn, *args, **effective_kwargs))

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error("Error during self-consistency task execution: %s", exc)
                for f_cancel in futures:
                    if not f_cancel.done():
                        f_cancel.cancel()
                raise

    if not results:
        raise RuntimeError("All self-consistency calls failed.")

    if all(r == results[0] for r in results):
        return results[0], results

    return vote_on_best_result(results, model_id, fn.__name__, client=client)


def vote_on_best_result(
    results: list[str], model_id: str, fn_name: str, client: OpenAI
) -> tuple[str, list[str]]:
    """Use LLM to vote on the most consistent result."""
    system_prompt = load_prompt("voting_system")

    prompt = "".join(
        f"--- Output {index + 1} ---\n{value}\n\n"
        for index, value in enumerate(results)
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    completion = client.chat.completions.create(
        model=model_id,
        temperature=0.1,
        messages=messages,
    )
    return require_completion_text(completion, f"self-consistency voting for {fn_name}"), results


def classify_document(
    image_paths: list[Path],
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
    image_content = [_encode_image(image_path) for image_path in image_paths]

    system_prompt = load_prompt("classification_system")
    system_prompt = system_prompt.format(patient_context=profile_context)
    user_prompt = load_prompt("classification_user")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": cast(
                list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam],
                [{"type": "text", "text": user_prompt}, *image_content],
            ),
        },
    ]
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        tools=CLASSIFICATION_TOOLS,
        tool_choice=CLASSIFICATION_TOOL_CHOICE,
    )
    if not completion.choices:
        raise RuntimeError("Missing classification choices")

    tool_calls = completion.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError("Missing classification tool call")

    tool_result_dict = _parse_classification_tool_args(tool_calls[0].function.arguments)
    exam_date = tool_result_dict.get("exam_date")
    if isinstance(exam_date, str):
        tool_result_dict["exam_date"] = _normalize_date_format(exam_date)
    return DocumentClassification.model_validate(tool_result_dict)


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
    system_prompt = load_prompt(prompt_variant)
    system_prompt = system_prompt.format(patient_context=profile_context)
    user_prompt = user_prompt_text or load_prompt(user_prompt_name)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": cast(
                list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam],
                [{"type": "text", "text": user_prompt}, _encode_image(image_path)],
            ),
        },
    ]
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=16384,
    )
    content = strip_markdown_fences(
        require_completion_text(completion, f"transcription of {image_path.name}")
    )

    if content.startswith("{"):
        try:
            parsed = parse_json_mapping(content, f"transcription of {image_path.name}")
            if isinstance(parsed, dict) and "transcription" in parsed:
                transcription = parsed["transcription"]
                if isinstance(transcription, str):
                    return transcription
        except json.JSONDecodeError:
            logger.debug("Returning raw fenced transcription for %s", image_path.name)

    return content

def build_chart_user_prompt(chart_type: str, embedded_text: str) -> str:
    """Build chart extraction prompt with embedded-text context."""
    prompt = load_prompt("chart_transcription_user")
    embedded_excerpt = embedded_text.strip()
    if len(embedded_excerpt) > 3000:
        embedded_excerpt = embedded_excerpt[:3000]
    return prompt.format(chart_type=chart_type, embedded_text=embedded_excerpt or "[none]")


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
    for attempt, prompt_variant in enumerate(variants[: max_retries + 1]):
        try:
            logger.debug(
                f"Transcription attempt {attempt + 1} using {prompt_variant} for {image_path.name}"
            )

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

            logger.warning(
                f"Transcription validation failure ({reason}) with {prompt_variant} "
                f"for {image_path.name}, trying alternative prompt..."
            )

        except APIError as exc:
            logger.error(
                "Transcription failed with %s for %s: %s",
                prompt_variant,
                image_path.name,
                exc,
            )
            continue

    if transcription:
        raise RuntimeError(
            f"All prompt variants produced invalid transcription for {image_path.name}"
        )
    raise RuntimeError(f"All prompt variants failed for {image_path.name}")


def validate_transcription(
    transcription: str,
    model_id: str,
    client: OpenAI,
    page_kind: str = "text",
    chart_type: str | None = None,
) -> tuple[bool, str]:
    """Returns (is_valid, reason). Uses local validation plus LLM refusal detection."""
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

    prompt = (
        """You are checking if the following text is a refusal to transcribe medical content.

    A refusal would be text where the model says it cannot or will not transcribe
    the medical document, mentions privacy concerns, or declines to process the
    request.

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
    except APIError as exc:
        logger.warning("Failed to check for refusal with LLM: %s", exc)

    return (True, "ok")


def _normalize_date_format(date_str: str | None) -> str | None:
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
    if all(t == original_transcriptions[0] for t in original_transcriptions):
        return 1.0

    system_prompt = load_prompt("confidence_scoring_system")
    prompt_parts = [f"## Final Merged Transcription:\n{merged_transcription}\n"]
    for i, orig in enumerate(original_transcriptions, 1):
        prompt_parts.append(f"## Original Transcription {i}:\n{orig}\n")

    user_prompt = "\n".join(prompt_parts)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = client.chat.completions.create(
        model=model_id,
        temperature=0.1,
        messages=messages,
    )
    result = parse_json_mapping(
        require_completion_text(completion, "confidence scoring"),
        "confidence scoring",
    )
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise ValueError("Confidence scoring response is missing a numeric confidence")
    return max(0.0, min(1.0, float(confidence)))
