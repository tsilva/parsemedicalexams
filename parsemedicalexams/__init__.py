"""Medical exams parser package."""

__version__ = "0.2.0"

from .config import ExtractionConfig, ProfileConfig
from .document_io import extract_pdf_page_text
from .extraction import (
    build_chart_user_prompt,
    transcribe_with_retry,
    self_consistency,
    classify_document,
    transcribe_page,
    score_transcription_confidence,
    DocumentClassification,
)
from .standardization import standardize_exam_types
from .summarization import summarize_document
from .utils import (
    preprocess_page_image,
    setup_logging,
    extract_dates_from_text,
)

__all__ = [
    "__version__",
    "ExtractionConfig",
    "ProfileConfig",
    "build_chart_user_prompt",
    "extract_pdf_page_text",
    "transcribe_with_retry",
    "self_consistency",
    "classify_document",
    "transcribe_page",
    "score_transcription_confidence",
    "DocumentClassification",
    "standardize_exam_types",
    "summarize_document",
    "preprocess_page_image",
    "setup_logging",
    "extract_dates_from_text",
]
