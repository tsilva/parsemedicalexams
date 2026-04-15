"""Document-level summarization of medical exam transcriptions."""

from __future__ import annotations

import logging

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .models import ExamRecord
from .utils import load_prompt, require_completion_text
from .validation import first_blocking_issue, validate_summary_output

logger = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_TOKENS = 100_000


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length (rough 4 chars per token)."""
    return len(text) // 4


def _build_exam_list(exams: list[ExamRecord]) -> str:
    """Build a numbered exam list string for prompt context."""

    def row(index: int, exam: ExamRecord) -> str:
        name = exam.exam_name_standardized or exam.exam_name_raw
        exam_date = exam.exam_date or ""
        exam_type = exam.exam_type or "other"
        suffix = f" ({exam_date})" if exam_date else ""
        return f"{index}. {name} [{exam_type}]{suffix}"

    return "\n".join(row(index, exam) for index, exam in enumerate(exams, 1))


def _build_transcriptions(exams: list[ExamRecord]) -> str:
    """Concatenate exam transcriptions with separators."""

    def row(index: int, exam: ExamRecord) -> str:
        name = exam.exam_name_standardized or exam.exam_name_raw
        return (
            f"--- EXAM {index}: {name} (Page {exam.page_number}) ---\n{exam.transcription.strip()}"
        )

    return "\n\n".join(row(index, exam) for index, exam in enumerate(exams, 1))


def _llm_summarize(
    messages: list[ChatCompletionMessageParam],
    model_id: str,
    client: OpenAI,
) -> str:
    """Make a single LLM summarization call."""
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.1,
        max_tokens=4000,
    )
    return require_completion_text(completion, "summarization")


def summarize_document(
    exams: list[ExamRecord],
    model_id: str,
    client: OpenAI,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> str:
    """Generate a comprehensive clinical summary for all exams in a document.
    Uses incremental chunked summarization to fit within token budget."""
    if not exams:
        return ""

    exams_with_content = [
        exam
        for exam in exams
        if exam.transcription.strip()
        and (exam.validation_status == "ok" or exam.chart_data_status == "non_discrete_visual")
    ]
    if not exams_with_content:
        return ""

    system_prompt = load_prompt("summarization_system")
    user_prompt_template = load_prompt("summarization_user")

    fixed_overhead_tokens = _estimate_tokens(system_prompt) + 200
    content_budget = max_input_tokens - fixed_overhead_tokens

    attempts = 2
    running_summary = ""
    for attempt in range(1, attempts + 1):
        running_summary = _incremental_summarize(
            exams_with_content,
            system_prompt,
            user_prompt_template,
            content_budget,
            model_id,
            client,
        )
        issues = validate_summary_output(running_summary)
        blocking_issue = first_blocking_issue(issues)
        if not blocking_issue:
            return running_summary
        logger.warning(
            "Summary validation failure on attempt %s/%s: %s",
            attempt,
            attempts,
            blocking_issue.kind,
        )

    raise RuntimeError("Summary validation failed after all summarization attempts")


def _incremental_summarize(
    exams: list[ExamRecord],
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

        logger.info(
            "Summarizing chunk %s/%s (%s exams)",
            chunk_idx + 1,
            len(chunks),
            len(chunk),
        )
        running_summary = _llm_summarize(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model_id,
            client,
        )

    return running_summary


def _split_into_chunks(exams: list[ExamRecord], content_budget: int) -> list[list[ExamRecord]]:
    """Split exams into chunks that each fit within the token budget."""
    incremental_overhead = 2000
    chunk_budget = content_budget - incremental_overhead

    chunks: list[list[ExamRecord]] = []
    current_chunk: list[ExamRecord] = []
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
