"""Document-level summarization of medical exam transcriptions."""

import logging
from openai import OpenAI

from utils import load_prompt

logger = logging.getLogger(__name__)


def summarize_document(
    exams: list[dict],
    model_id: str,
    client: OpenAI
) -> str:
    """
    Generate a comprehensive clinical summary for an entire document.

    Concatenates all exam transcriptions and creates one unified summary
    that preserves all clinically relevant details for medical records.

    Args:
        exams: List of exam dictionaries with transcription, exam_type, and exam_name_standardized
        model_id: Model to use for summarization
        client: OpenAI client instance

    Returns:
        Comprehensive clinical summary of all exams in the document
    """
    if not exams:
        return ""

    # Filter exams with transcriptions
    exams_with_content = [e for e in exams if e.get("transcription", "").strip()]
    if not exams_with_content:
        return ""

    # Build exam list for context
    exam_list_items = []
    for i, exam in enumerate(exams_with_content, 1):
        exam_name = exam.get("exam_name_standardized") or exam.get("exam_name_raw", "Unknown")
        exam_type = exam.get("exam_type", "other")
        exam_date = exam.get("exam_date", "")
        date_str = f" ({exam_date})" if exam_date else ""
        exam_list_items.append(f"{i}. {exam_name} [{exam_type}]{date_str}")

    exam_list = "\n".join(exam_list_items)

    # Concatenate all transcriptions with clear separators
    transcription_parts = []
    for i, exam in enumerate(exams_with_content, 1):
        exam_name = exam.get("exam_name_standardized") or exam.get("exam_name_raw", "Unknown")
        page_num = exam.get("page_number", "?")
        transcription = exam.get("transcription", "").strip()
        transcription_parts.append(f"--- EXAM {i}: {exam_name} (Page {page_num}) ---\n{transcription}")

    transcriptions = "\n\n".join(transcription_parts)

    # Load prompts
    system_prompt = load_prompt("summarization_system")
    user_prompt_template = load_prompt("summarization_user")
    user_prompt = user_prompt_template.format(
        exam_count=len(exams_with_content),
        exam_list=exam_list,
        transcriptions=transcriptions
    )

    # LLM call
    try:
        logger.info(f"Summarizing document with {len(exams_with_content)} exam(s)")
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000  # Increased for comprehensive summaries
        )

        if not completion or not completion.choices:
            logger.error("Invalid completion response for summarization")
            return ""

        return completion.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error during document summarization: {e}")
        return ""
