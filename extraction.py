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

from utils import parse_llm_json_response

logger = logging.getLogger(__name__)


# ========================================
# Pydantic Models
# ========================================

class MedicalExam(BaseModel):
    """Single medical exam extraction result."""

    # Raw extraction fields
    exam_date: Optional[str] = Field(
        default=None,
        description="Exam date in YYYY-MM-DD format"
    )
    exam_name_raw: str = Field(
        description="Exam name EXACTLY as shown in document (e.g., 'Radiografia do Tórax', 'Ecografia Abdominal')"
    )
    transcription: str = Field(
        description="Full text of the exam report EXACTLY as written. Include ALL text visible in the report."
    )

    # Internal fields (added by pipeline, not by LLM)
    exam_type: Optional[str] = Field(
        default=None,
        description="Standardized category: imaging, ultrasound, endoscopy, other"
    )
    exam_name_standardized: Optional[str] = Field(
        default=None,
        description="Standardized exam name (e.g., 'Chest X-ray', 'Abdominal Ultrasound')"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Aggressive summary: ONLY findings, impressions, recommendations"
    )
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number in PDF")
    source_file: Optional[str] = Field(default=None, description="Source file identifier")


class MedicalExamReport(BaseModel):
    """Document-level medical exam report."""

    report_date: Optional[str] = Field(
        default=None,
        description="Report issue date in YYYY-MM-DD format"
    )
    facility_name: Optional[str] = Field(
        default=None,
        description="Healthcare facility name"
    )
    page_has_exam_data: Optional[bool] = Field(
        default=None,
        description="True if page contains exam results, False if administrative content"
    )
    exams: List[MedicalExam] = Field(
        default_factory=list,
        description="List of medical exams extracted from this page"
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


# Tool definition for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_medical_exams",
            "description": "Extracts medical exam reports from document image",
            "parameters": MedicalExamReport.model_json_schema()
        }
    }
]


# ========================================
# Extraction Prompts
# ========================================

EXTRACTION_SYSTEM_PROMPT = """
You are a medical exam report transcription specialist. Your PRIMARY goal is to extract the COMPLETE text content from medical imaging, ultrasound, endoscopy, and other diagnostic exam reports.

CRITICAL RULES:

1. TRANSCRIBE COMPLETELY: Extract the FULL text of the exam report exactly as written
   - Copy ALL visible text including headers, findings, impressions, and conclusions
   - Preserve paragraph structure and formatting where possible
   - Do NOT summarize or condense - we need the complete transcription

2. EXAM IDENTIFICATION:
   - Identify the exam type from the document (X-ray, MRI, CT, Ultrasound, Echo, Endoscopy, etc.)
   - Extract the exam name exactly as written in the document

3. DATE EXTRACTION:
   - Look for exam date, report date, or study date
   - Convert to YYYY-MM-DD format
   - If only one date visible, use it as exam_date

4. MULTIPLE EXAMS:
   - If a document contains MULTIPLE different exams, extract each as a separate entry
   - Each exam should have its own complete transcription

5. LANGUAGE PRESERVATION:
   - Keep text in the original language (Portuguese, English, etc.)
   - Do NOT translate medical terminology

EXAM TYPES TO RECOGNIZE:
- Imaging: X-ray (Radiografia), MRI (Ressonância Magnética), CT (Tomografia), Mammography (Mamografia)
- Ultrasound: Abdominal, Pelvic, Thyroid, Echocardiogram (Ecocardiograma)
- Endoscopy: Gastroscopy (EDA), Colonoscopy (Colonoscopia), Bronchoscopy
- Other: ECG, EEG, Spirometry, Stress Test, Sleep Study, Pathology

PAGE CLASSIFICATION:
- `page_has_exam_data`: Set to true if this page contains ANY exam report content
- Set to false if this is a cover page, instructions, administrative content, or has no exam data
- This helps distinguish empty pages from extraction failures

Remember: Your job is to transcribe COMPLETELY. Extract EVERYTHING visible in the report.
""".strip()

EXTRACTION_USER_PROMPT = """
Please extract ALL medical exam reports from this document image.

For EACH exam found, provide:
1. exam_date - The date of the exam/study (YYYY-MM-DD format)
2. exam_name_raw - The exam name exactly as written
3. transcription - The COMPLETE text of the exam report (include everything visible)

IMPORTANT:
- Transcribe the FULL content, do not summarize
- If multiple exams are in the document, extract each separately
- Set page_has_exam_data to false if this is just a cover page or administrative content
""".strip()


# ========================================
# Self-Consistency
# ========================================

def self_consistency(fn, model_id, n, *args, **kwargs):
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
            if 'temperature' in fn.__code__.co_varnames and 'temperature' not in kwargs:
                effective_kwargs['temperature'] = SELF_CONSISTENCY_TEMPERATURE
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
    return vote_on_best_result(results, model_id, fn.__name__)


def vote_on_best_result(results: list, model_id: str, fn_name: str):
    """Use LLM to vote on the most consistent result."""
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    system_prompt = (
        "You are an expert at comparing multiple outputs of the same extraction task. "
        "We have extracted several samples from the same prompt in order to average out any errors or inconsistencies. "
        "Your job is to select the output that is most consistent with the majority of the provided samples. "
        "Prioritize agreement on extracted content (exam names, dates, transcription text, etc.). "
        "Ignore formatting, whitespace, and layout differences. "
        "Return ONLY the best output, verbatim, with no extra commentary. "
        "Do NOT include any delimiters, output numbers, or extra labels in your response."
    )

    prompt = "".join(
        f"--- Output {i+1} ---\n{json.dumps(v, ensure_ascii=False) if type(v) in [list, dict] else v}\n\n"
        for i, v in enumerate(results)
    )

    voted_raw = None
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        voted_raw = completion.choices[0].message.content.strip()

        if fn_name == 'extract_exams_from_page_image':
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
    temperature: float = 0.3
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

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": EXTRACTION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]}
            ],
            temperature=temperature,
            max_tokens=16384,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "extract_medical_exams"}}
        )
    except APIError as e:
        logger.error(f"API Error during exam extraction from {image_path.name}: {e}")
        raise RuntimeError(f"Exam extraction failed for {image_path.name}: {e}")

    # Check for valid response structure
    if not completion or not completion.choices or len(completion.choices) == 0:
        logger.error(f"Invalid completion response structure")
        return MedicalExamReport(exams=[]).model_dump(mode='json')

    if not completion.choices[0].message.tool_calls:
        logger.warning(f"No tool call by model for exam extraction from {image_path.name}")
        return MedicalExamReport(exams=[]).model_dump(mode='json')

    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try:
        tool_result_dict = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for tool args: {e}")
        return MedicalExamReport(exams=[]).model_dump(mode='json')

    # Fix date formats
    tool_result_dict = _fix_date_formats(tool_result_dict)

    # Validate with Pydantic
    try:
        report_model = MedicalExamReport(**tool_result_dict)
        report_model.normalize_empty_optionals()

        # Check for extraction quality
        if report_model.exams:
            empty_count = sum(1 for e in report_model.exams if not e.transcription or len(e.transcription.strip()) < 50)
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

        return report_model.model_dump(mode='json')
    except Exception as e:
        num_exams = len(tool_result_dict.get("exams", []))
        logger.error(f"Model validation error for report with {num_exams} exams: {e}")
        return MedicalExamReport(exams=[]).model_dump(mode='json')


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


def _fix_date_formats(tool_result_dict: dict) -> dict:
    """Fix common date formatting issues."""
    # Fix date at report level
    if "report_date" in tool_result_dict:
        tool_result_dict["report_date"] = _normalize_date_format(tool_result_dict["report_date"])

    # Fix dates in exams
    if "exams" in tool_result_dict and isinstance(tool_result_dict["exams"], list):
        for exam in tool_result_dict["exams"]:
            if isinstance(exam, dict) and "exam_date" in exam:
                exam["exam_date"] = _normalize_date_format(exam["exam_date"])

    return tool_result_dict
