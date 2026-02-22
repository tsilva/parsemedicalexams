"""Medical exam extraction from images using vision models."""

import json
import re
import base64
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from openai import OpenAI, APIError

from config import resolve_base_url
from utils import parse_llm_json_response, load_prompt

logger = logging.getLogger(__name__)


# ========================================
# Pydantic Models
# ========================================


class MedicalExam(BaseModel):
    """Single medical exam extraction result."""

    # Raw extraction fields
    exam_date: Optional[str] = Field(
        default=None, description="Exam date in YYYY-MM-DD format"
    )
    exam_name_raw: str = Field(
        description="Document title EXACTLY as shown (e.g., 'Radiografia do Tórax', 'Ecografia Abdominal', 'Estudo do Sono (Questionário de Hábitos)')"
    )
    transcription: str = Field(
        description="Full text of the document EXACTLY as written. Include ALL visible text: questions, answers, checkboxes, values, findings, conclusions."
    )

    # Internal fields (added by pipeline, not by LLM)
    exam_type: Optional[str] = Field(
        default=None,
        description="Standardized category: imaging, ultrasound, endoscopy, other",
    )
    exam_name_standardized: Optional[str] = Field(
        default=None,
        description="Standardized exam name (e.g., 'Chest X-ray', 'Abdominal Ultrasound')",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Aggressive summary: ONLY findings, impressions, recommendations",
    )
    page_number: Optional[int] = Field(
        default=None, ge=1, description="Page number in PDF"
    )
    source_file: Optional[str] = Field(
        default=None, description="Source file identifier"
    )


class MedicalExamReport(BaseModel):
    """Document-level medical exam report."""

    report_date: Optional[str] = Field(
        default=None, description="Report issue date in YYYY-MM-DD format"
    )
    facility_name: Optional[str] = Field(
        default=None, description="Healthcare facility name"
    )
    page_has_exam_data: Optional[bool] = Field(
        default=None,
        description="True if page contains medical content (exam results, questionnaire responses, test data). False only for blank pages or administrative headers.",
    )
    exams: List[MedicalExam] = Field(
        default_factory=list,
        description="List of medical documents extracted from this page. Include exam reports, questionnaires, and any medical forms with content.",
    )
    source_file: Optional[str] = Field(default=None, description="Source PDF filename")

    def normalize_empty_optionals(self):
        """Convert empty strings to None for optional fields."""
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            field_info = self.model_fields[field_name]
            is_optional_type = field_info.is_required() is False
            if value == "" and is_optional_type:
                setattr(self, field_name, None)

        for exam in self.exams:
            for field_name in exam.model_fields:
                value = getattr(exam, field_name)
                field_info = exam.model_fields[field_name]
                is_optional_type = field_info.is_required() is False
                if value == "" and is_optional_type:
                    setattr(exam, field_name, None)


class DocumentClassification(BaseModel):
    """Document classification result."""

    is_exam: bool = Field(
        description="True if the document contains medical exam results, clinical reports, or medical content that should be transcribed"
    )
    exam_name_raw: Optional[str] = Field(
        default=None,
        description="Document title or exam name exactly as written (e.g., 'CABELO: NUTRIENTES E METAIS TÓXICOS')",
    )
    exam_date: Optional[str] = Field(
        default=None, description="Exam date in YYYY-MM-DD format"
    )
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


# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_medical_documents",
            "description": "Extracts and transcribes ALL content from medical document images including: exam reports, clinical notes, discharge summaries, administrative letters, cover pages, correspondence, and any other medical documentation. Must be called for ANY page with readable text.",
            "parameters": MedicalExamReport.model_json_schema(),
        },
    }
]

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


def self_consistency(fn, model_id, n, *args, base_url=None, api_key=None, **kwargs):
    """
    Run a function multiple times and vote on the best result.

    Args:
        fn: Function to run
        model_id: Model to use for voting
        n: Number of times to run the function
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (best_result, all_results)
    """
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
    return vote_on_best_result(
        results, model_id, fn.__name__, base_url=base_url, api_key=api_key
    )


def vote_on_best_result(
    results: list,
    model_id: str,
    fn_name: str,
    base_url: str = None,
    api_key: str = None,
):
    """Use LLM to vote on the most consistent result."""
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url=base_url
        or resolve_base_url(
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        ),
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
    )

    system_prompt = load_prompt("voting_system")

    prompt = "".join(
        f"--- Output {i + 1} ---\n{json.dumps(v, ensure_ascii=False) if type(v) in [list, dict] else v}\n\n"
        for i, v in enumerate(results)
    )

    voted_raw = None
    try:
        completion = client.chat.completions.create(
            model=model_id,
            temperature=0.1,  # Low temperature for deterministic voting
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        voted_raw = completion.choices[0].message.content.strip()

        if fn_name == "extract_exams_from_page_image":
            voted_result = parse_llm_json_response(voted_raw, fallback=None)
            if voted_result:
                return voted_result, results
            else:
                logger.error("Failed to parse voted result as JSON")
                return results[0], results
        else:
            return voted_raw, results

    except Exception as e:
        logger.error(f"Error during self-consistency voting: {e}")
        return results[0], results


# ========================================
# Extraction Function
# ========================================


def extract_exams_from_page_image(
    image_path: Path,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.3,
    profile_context: str = "",
) -> dict:
    """
    Extract medical exams from a page image using vision model.

    Args:
        image_path: Path to the preprocessed page image
        model_id: Vision model to use for extraction
        client: OpenAI client instance
        temperature: Temperature for sampling

    Returns:
        Dictionary with extracted report data (validated by Pydantic)
    """
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    system_prompt = load_prompt("extraction_system")
    system_prompt = system_prompt.format(patient_context=profile_context)
    user_prompt = load_prompt("extraction_user")

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
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "extract_medical_documents"},
            },
        )
    except APIError as e:
        logger.error(f"API Error during exam extraction from {image_path.name}: {e}")
        raise RuntimeError(f"Exam extraction failed for {image_path.name}: {e}")

    # Check for valid response structure
    if not completion or not completion.choices or len(completion.choices) == 0:
        logger.error(f"Invalid completion response structure")
        return MedicalExamReport(exams=[]).model_dump(mode="json")

    if not completion.choices[0].message.tool_calls:
        logger.warning(
            f"No tool call by model for exam extraction from {image_path.name}"
        )
        return MedicalExamReport(exams=[]).model_dump(mode="json")

    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try:
        tool_result_dict = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for tool args: {e}")
        return MedicalExamReport(exams=[]).model_dump(mode="json")

    # Fix date formats
    tool_result_dict = _fix_date_formats(tool_result_dict)

    # Validate with Pydantic
    try:
        report_model = MedicalExamReport(**tool_result_dict)
        report_model.normalize_empty_optionals()

        # Check for extraction quality
        if report_model.exams:
            empty_count = sum(
                1
                for e in report_model.exams
                if not e.transcription or len(e.transcription.strip()) < 50
            )
            total_count = len(report_model.exams)

            if empty_count > 0:
                logger.warning(
                    f"Extraction quality issue: {empty_count}/{total_count} exams have very short transcriptions. "
                    f"This suggests incomplete extraction.\n"
                    f"\t- {image_path}"
                )
        else:
            if report_model.page_has_exam_data is False:
                logger.debug(f"Page confirmed to have no exam data:\n\t- {image_path}")
            else:
                logger.warning(
                    f"Extraction returned 0 exams. "
                    f"This may indicate a model extraction failure - image should be manually reviewed.\n"
                    f"\t- {image_path}"
                )

        return report_model.model_dump(mode="json")
    except Exception as e:
        num_exams = len(tool_result_dict.get("exams", []))
        logger.error(f"Model validation error for report with {num_exams} exams: {e}")
        return MedicalExamReport(exams=[]).model_dump(mode="json")


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
    except APIError as e:
        logger.error(f"API Error during document classification: {e}")
        # Default to is_exam=True to avoid missing medical content
        return DocumentClassification(is_exam=True)

    if not completion or not completion.choices or len(completion.choices) == 0:
        logger.error("Invalid completion response for classification")
        return DocumentClassification(is_exam=True)

    if not completion.choices[0].message.tool_calls:
        logger.warning("No tool call by model for document classification")
        return DocumentClassification(is_exam=True)

    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try:
        tool_result_dict = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for classification: {e}")
        return DocumentClassification(is_exam=True)

    # Normalize date format
    if tool_result_dict.get("exam_date"):
        tool_result_dict["exam_date"] = _normalize_date_format(
            tool_result_dict["exam_date"]
        )

    try:
        return DocumentClassification(**tool_result_dict)
    except Exception as e:
        logger.error(f"Validation error for classification: {e}")
        return DocumentClassification(is_exam=True)


def transcribe_page(
    image_path: Path, model_id: str, client: OpenAI, temperature: float = 0.1
) -> str:
    """
    Transcribe all visible text from a page verbatim.

    Args:
        image_path: Path to the preprocessed page image
        model_id: Vision model to use for transcription
        client: OpenAI client instance
        temperature: Temperature for sampling (low for OCR)

    Returns:
        String with complete verbatim transcription
    """
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    system_prompt = load_prompt("transcription_system")
    user_prompt = load_prompt("transcription_user")

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
        logger.error(
            f"Invalid completion response for transcription of {image_path.name}"
        )
        return ""

    content = completion.choices[0].message.content
    if content is None:
        logger.warning(f"No content in response for transcription of {image_path.name}")
        return ""

    content = content.strip()

    # Handle case where model returns JSON (legacy behavior from function calling)
    # Strip markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        content = "\n".join(lines).strip()

    # Try to parse as JSON and extract transcription field
    if content.startswith("{"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "transcription" in parsed:
                return parsed["transcription"]
        except json.JSONDecodeError:
            pass  # Not valid JSON, return as-is

    return content


def validate_transcription(
    transcription: str, model_id: str, client: OpenAI
) -> tuple[bool, str]:
    """Returns (is_valid, reason). Uses LLM to check if transcription is a refusal."""
    # Empty or too short
    if not transcription or len(transcription.strip()) < 20:
        return (False, "empty")

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
        result = response.choices[0].message.content.strip().lower()
        if "yes" in result:
            return (False, "refusal")
    except Exception as e:
        logger.warning(f"Failed to check for refusal with LLM: {e}")
        # If LLM check fails, assume valid to avoid blocking
        pass

    return (True, "ok")


def _normalize_date_format(date_str: Optional[str]) -> Optional[str]:
    """
    Normalize date strings to YYYY-MM-DD format.

    Handles common formats:
    - DD/MM/YYYY (e.g., 20/11/2024 -> 2024-11-20)
    - DD-MM-YYYY (e.g., 20-11-2024 -> 2024-11-20)
    - YYYY-MM-DD (already correct)
    """
    if not date_str or date_str == "0000-00-00":
        return None

    # Already in correct format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # DD/MM/YYYY or DD-MM-YYYY format
    match = re.match(r"^(\d{2})[/-](\d{2})[/-](\d{4})$", date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"

    logger.warning(f"Unable to normalize date format: {date_str}")
    return None


def _fix_malformed_json_string(text: str) -> str:
    """
    Fix malformed JSON strings returned by Gemini.
    Issues handled:
    - Unescaped newlines inside string values
    - Unescaped quotes inside string values (from OCR errors like * -> ")
    """
    # First, try a regex-based approach to extract and fix the transcription field
    # which is where most issues occur
    import re

    def fix_transcription_value(match):
        """Escape problematic characters in transcription value."""
        content = match.group(1)
        # Escape any unescaped quotes (but not the ones that are already escaped)
        # Replace " with \" unless preceded by \
        fixed = re.sub(r'(?<!\\)"', r"\"", content)
        # Escape literal newlines
        fixed = fixed.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"transcription": "{fixed}"'

    # Try to fix the transcription field specifically
    fixed = re.sub(
        r'"transcription":\s*"((?:[^"\\]|\\.)*)(?="[,}]|\Z)',
        fix_transcription_value,
        text,
        flags=re.DOTALL,
    )

    # If regex didn't help, fall back to character-by-character approach
    if fixed == text:
        result = []
        in_string = False
        i = 0
        while i < len(text):
            char = text[i]
            if char == '"':
                # Check if this quote is a string delimiter or content
                if not in_string:
                    in_string = True
                    result.append(char)
                elif i + 1 < len(text) and text[i + 1] in ",}]:":
                    # This quote ends a string (followed by JSON structure)
                    in_string = False
                    result.append(char)
                elif i > 0 and text[i - 1] == "\\":
                    # Already escaped
                    result.append(char)
                else:
                    # Unescaped quote inside string - escape it
                    result.append('\\"')
            elif char == "\n" and in_string:
                result.append("\\n")
            elif char == "\r" and in_string:
                result.append("\\r")
            elif char == "\t" and in_string:
                result.append("\\t")
            else:
                result.append(char)
            i += 1
        fixed = "".join(result)

    return fixed


def _parse_yaml_like_exam(text: str) -> Optional[dict]:
    """
    Parse YAML-like exam string that Gemini sometimes returns.
    Format: "key: value\nkey: value\n..."
    """
    if not text or ":" not in text:
        return None

    result = {}
    lines = text.split("\n")
    current_key = None
    current_value_lines = []

    for line in lines:
        # Check if this line starts a new key
        if ": " in line and not line.startswith(" "):
            # Save previous key-value pair
            if current_key:
                result[current_key] = "\n".join(current_value_lines).strip()

            # Parse new key-value
            colon_idx = line.index(": ")
            current_key = line[:colon_idx].strip()
            current_value_lines = [line[colon_idx + 2 :]]
        elif current_key:
            # Continuation of multi-line value
            current_value_lines.append(line)

    # Save last key-value pair
    if current_key:
        result[current_key] = "\n".join(current_value_lines).strip()

    # Validate required fields
    if "exam_name_raw" in result and "transcription" in result:
        return result

    return None


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
        response = completion.choices[0].message.content.strip()

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


def _fix_date_formats(tool_result_dict: dict) -> dict:
    """Fix common date formatting issues and handle malformed exam entries."""
    # Fix date at report level
    if "report_date" in tool_result_dict:
        tool_result_dict["report_date"] = _normalize_date_format(
            tool_result_dict["report_date"]
        )

    # Fix dates in exams, also handle string exams (Gemini sometimes returns JSON strings instead of objects)
    if "exams" in tool_result_dict and isinstance(tool_result_dict["exams"], list):
        fixed_exams = []
        for exam in tool_result_dict["exams"]:
            # Skip None values
            if exam is None:
                continue

            # Parse string exams (Gemini bug: sometimes returns strings instead of objects)
            if isinstance(exam, str):
                original_str = exam
                # Fix invalid JSON escapes that Gemini sometimes produces
                # \' is valid in JS/Python but NOT in JSON - replace with unescaped '
                exam = exam.replace("\\'", "'")
                # Try JSON first (Gemini sometimes returns unescaped newlines in strings)
                try:
                    exam = json.loads(exam)
                except json.JSONDecodeError:
                    # Try fixing malformed JSON (unescaped newlines, quotes)
                    try:
                        fixed_str = _fix_malformed_json_string(
                            exam
                        )  # Use already-fixed string
                        exam = json.loads(fixed_str)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error after fix: {e}")
                        # Try YAML-like format: "key: value\nkey: value"
                        exam = _parse_yaml_like_exam(original_str)
                        if exam is None:
                            logger.warning(
                                f"Failed to parse exam string: {original_str[:100]}..."
                            )
                            logger.debug(
                                f"Full exam string ({len(original_str)} chars): {original_str}"
                            )
                            continue

            # Fix date format
            if isinstance(exam, dict) and "exam_date" in exam:
                exam["exam_date"] = _normalize_date_format(exam["exam_date"])

            if isinstance(exam, dict):
                fixed_exams.append(exam)

        tool_result_dict["exams"] = fixed_exams

    return tool_result_dict
