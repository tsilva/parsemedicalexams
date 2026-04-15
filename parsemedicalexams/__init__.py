"""Medical exams parser package."""

__version__ = "0.2.0"

from .config import ExtractionConfig, ProfileConfig
from .document_io import extract_pdf_page_text
from .extraction import (
    DocumentClassification,
    build_chart_user_prompt,
    classify_document,
    score_transcription_confidence,
    self_consistency,
    transcribe_page,
    transcribe_with_retry,
)
from .standardization import standardize_exam_types
from .summarization import summarize_document
from .utils import (
    extract_dates_from_text,
    preprocess_page_image,
    setup_logging,
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
