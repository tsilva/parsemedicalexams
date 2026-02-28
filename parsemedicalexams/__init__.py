"""Medical exams parser package."""

__version__ = "0.1.1"

from .config import ExtractionConfig, ProfileConfig
from .extraction import (
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
    load_dotenv_with_env,
    extract_dates_from_text,
)

__all__ = [
    "__version__",
    "ExtractionConfig",
    "ProfileConfig",
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
    "load_dotenv_with_env",
    "extract_dates_from_text",
]
