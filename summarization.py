"""Document-level summarization of medical exam transcriptions."""

import logging
from openai import OpenAI

from utils import load_prompt

logger = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_TOKENS = 100_000


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length (rough 4 chars per token)."""
    return len(text) // 4


def _build_exam_list(exams: list[dict]) -> str:
    """Build a numbered exam list string for prompt context."""
    items = []
    for i, exam in enumerate(exams, 1):
        exam_name = exam.get("exam_name_standardized") or exam.get("exam_name_raw", "Unknown")
        exam_type = exam.get("exam_type", "other")
        exam_date = exam.get("exam_date", "")
        date_str = f" ({exam_date})" if exam_date else ""
        items.append(f"{i}. {exam_name} [{exam_type}]{date_str}")
    return "\n".join(items)


def _build_transcriptions(exams: list[dict]) -> str:
    """Concatenate exam transcriptions with separators."""
    parts = []
    for i, exam in enumerate(exams, 1):
        exam_name = exam.get("exam_name_standardized") or exam.get("exam_name_raw", "Unknown")
        page_num = exam.get("page_number", "?")
        transcription = exam.get("transcription", "").strip()
        parts.append(f"--- EXAM {i}: {exam_name} (Page {page_num}) ---\n{transcription}")
    return "\n\n".join(parts)


def _llm_summarize(messages: list[dict], model_id: str, client: OpenAI) -> str:
    """Make a single LLM summarization call."""
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.1,
        max_tokens=4000,
    )
    if not completion or not completion.choices:
        logger.error("Invalid completion response for summarization")
        return ""
    return completion.choices[0].message.content.strip()


def summarize_document(
    exams: list[dict],
    model_id: str,
    client: OpenAI,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> str:
    """
    Generate a comprehensive clinical summary for an entire document.

    Always uses incremental summarization: exams are split into chunks that
    fit within the token budget and processed sequentially, accumulating a
    running summary.

    Args:
        exams: List of exam dictionaries with transcription, exam_type, and exam_name_standardized
        model_id: Model to use for summarization
        client: OpenAI client instance
        max_input_tokens: Maximum estimated tokens for LLM input

    Returns:
        Comprehensive clinical summary of all exams in the document
    """
    if not exams:
        return ""

    exams_with_content = [e for e in exams if e.get("transcription", "").strip()]
    if not exams_with_content:
        return ""

    system_prompt = load_prompt("summarization_system")
    user_prompt_template = load_prompt("summarization_user")

    # Calculate fixed overhead (system prompt + template chrome)
    fixed_overhead_tokens = _estimate_tokens(system_prompt) + 200
    content_budget = max_input_tokens - fixed_overhead_tokens

    return _incremental_summarize(
        exams_with_content, system_prompt, user_prompt_template,
        content_budget, model_id, client
    )


def _incremental_summarize(
    exams: list[dict],
    system_prompt: str,
    user_prompt_template: str,
    content_budget: int,
    model_id: str,
    client: OpenAI,
) -> str:
    """Summarize exams in chunks, building a running summary incrementally."""
    incremental_template = load_prompt("summarization_incremental_user")

    chunks = _split_into_chunks(exams, content_budget)
    logger.info(f"Summarizing {len(exams)} exam(s) in {len(chunks)} chunk(s)")

    running_summary = ""

    for chunk_idx, chunk in enumerate(chunks):
        exam_list = _build_exam_list(chunk)
        transcriptions = _build_transcriptions(chunk)

        try:
            if chunk_idx == 0:
                user_prompt = user_prompt_template.format(
                    exam_count=len(chunk),
                    exam_list=exam_list,
                    transcriptions=transcriptions,
                )
            else:
                user_prompt = incremental_template.format(
                    existing_summary=running_summary,
                    new_exam_count=len(chunk),
                    new_exam_list=exam_list,
                    new_transcriptions=transcriptions,
                )

            logger.info(f"Summarizing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} exams)")
            running_summary = _llm_summarize(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model_id,
                client,
            )

            if not running_summary:
                logger.error(f"Empty summary from chunk {chunk_idx + 1}, aborting")
                return ""

        except Exception as e:
            logger.error(f"Error during summarization (chunk {chunk_idx + 1}): {e}")
            return running_summary

    return running_summary


def _split_into_chunks(exams: list[dict], content_budget: int) -> list[list[dict]]:
    """Split exams into chunks that each fit within the token budget."""
    # Reserve space for the running summary in incremental passes
    incremental_overhead = 2000
    chunk_budget = content_budget - incremental_overhead

    chunks = []
    current_chunk = []
    current_tokens = 0

    for exam in exams:
        exam_list_text = _build_exam_list([exam])
        exam_transcription = _build_transcriptions([exam])
        exam_tokens = _estimate_tokens(exam_list_text + exam_transcription)

        if exam_tokens > chunk_budget and not current_chunk:
            chunks.append([exam])
            continue

        if current_tokens + exam_tokens > chunk_budget and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(exam)
        current_tokens += exam_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
