"""Shared runtime models and serialized document shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from typing_extensions import NotRequired, TypedDict

ExamCategory = str
PageKind = str
ValidationStatus = str
SourceMode = str
ChartType = str
ChartDataStatus = str

FRONTMATTER_FIELD_MAP: Final[dict[str, str]] = {
    "exam_date": "exam_date",
    "exam_name_raw": "exam_name_raw",
    "exam_name_standardized": "title",
    "exam_type": "category",
    "physician_name": "doctor",
    "facility_name": "facility",
    "department": "department",
}
INVERSE_FRONTMATTER_FIELD_MAP: Final[dict[str, str]] = {
    frontmatter_key: exam_key for exam_key, frontmatter_key in FRONTMATTER_FIELD_MAP.items()
}

REQUIRED_METADATA_FIELDS: Final[set[str]] = {
    "exam_date",
    "exam_name_raw",
    "category",
    "title",
}
ALLOWED_CATEGORIES: Final[set[ExamCategory]] = {
    "appointment",
    "endoscopy",
    "imaging",
    "other",
    "prescription",
    "ultrasound",
}
ALLOWED_PAGE_KINDS: Final[set[PageKind]] = {"text", "chart", "image_only"}
ALLOWED_VALIDATION_STATUSES: Final[set[ValidationStatus]] = {
    "ok",
    "retryable_failure",
    "unsupported_visual",
    "failed",
}
ALLOWED_SOURCE_MODES: Final[set[SourceMode]] = {"embedded_text", "vision", "hybrid"}
ALLOWED_CHART_TYPES: Final[set[ChartType]] = {
    "audiogram",
    "eeg_signal_trace",
    "sleep_summary_graph",
    "tympanometry",
}
ALLOWED_CHART_DATA_STATUSES: Final[set[ChartDataStatus]] = {"ok", "non_discrete_visual"}
PAGE_ONLY_METADATA_FIELDS: Final[set[str]] = {
    "page",
    "source",
    "prompt_variant",
    "page_kind",
    "validation_status",
    "failure_type",
    "source_mode",
    "chart_type",
    "chart_data_status",
    "retry_attempts",
}
FRONTMATTER_FIELDS: Final[set[str]] = (
    set(FRONTMATTER_FIELD_MAP.values())
    | PAGE_ONLY_METADATA_FIELDS
    | {"confidence"}
)
ENHANCED_PAGE_FIELDS: Final[set[str]] = {
    "page_kind",
    "validation_status",
    "failure_type",
    "source_mode",
    "chart_type",
    "chart_data_status",
}


class ValidationMetadata(TypedDict):
    validation_status: ValidationStatus
    failure_type: str | None
    chart_data_status: ChartDataStatus | None


class StandardizedExamEntry(TypedDict):
    exam_type: ExamCategory
    standardized_name: str


class ExamFrontmatter(TypedDict, total=False):
    exam_date: str
    exam_name_raw: str
    title: str
    category: ExamCategory
    doctor: str
    facility: str
    department: str
    page: int
    source: str
    prompt_variant: str
    page_kind: PageKind
    validation_status: ValidationStatus
    failure_type: str | None
    source_mode: SourceMode
    chart_type: ChartType
    chart_data_status: ChartDataStatus
    retry_attempts: int
    confidence: float


class SerializedExamRecord(TypedDict):
    exam_name_raw: str
    exam_date: str | None
    transcription: str
    page_number: int
    source_file: str
    transcription_confidence: NotRequired[float | None]
    prompt_variant: NotRequired[str | None]
    retry_attempts: NotRequired[int]
    physician_name: NotRequired[str | None]
    department: NotRequired[str | None]
    facility_name: NotRequired[str | None]
    page_kind: NotRequired[PageKind]
    validation_status: NotRequired[ValidationStatus]
    failure_type: NotRequired[str | None]
    source_mode: NotRequired[SourceMode]
    chart_type: NotRequired[ChartType | None]
    chart_data_status: NotRequired[ChartDataStatus | None]
    exam_type: NotRequired[ExamCategory | None]
    exam_name_standardized: NotRequired[str | None]


@dataclass(slots=True)
class ExamRecord:
    exam_name_raw: str
    exam_date: str | None
    transcription: str
    page_number: int
    source_file: str
    transcription_confidence: float | None = None
    prompt_variant: str | None = None
    retry_attempts: int = 1
    physician_name: str | None = None
    department: str | None = None
    facility_name: str | None = None
    page_kind: PageKind = "text"
    validation_status: ValidationStatus = "ok"
    failure_type: str | None = None
    source_mode: SourceMode = "vision"
    chart_type: ChartType | None = None
    chart_data_status: ChartDataStatus | None = None
    exam_type: ExamCategory | None = None
    exam_name_standardized: str | None = None

    def to_serialized(self) -> SerializedExamRecord:
        return {
            "exam_name_raw": self.exam_name_raw,
            "exam_date": self.exam_date,
            "transcription": self.transcription,
            "page_number": self.page_number,
            "source_file": self.source_file,
            "transcription_confidence": self.transcription_confidence,
            "prompt_variant": self.prompt_variant,
            "retry_attempts": self.retry_attempts,
            "physician_name": self.physician_name,
            "department": self.department,
            "facility_name": self.facility_name,
            "page_kind": self.page_kind,
            "validation_status": self.validation_status,
            "failure_type": self.failure_type,
            "source_mode": self.source_mode,
            "chart_type": self.chart_type,
            "chart_data_status": self.chart_data_status,
            "exam_type": self.exam_type,
            "exam_name_standardized": self.exam_name_standardized,
        }

    @classmethod
    def from_serialized(cls, exam: SerializedExamRecord) -> "ExamRecord":
        return cls(
            exam_name_raw=exam["exam_name_raw"],
            exam_date=exam["exam_date"],
            transcription=exam["transcription"],
            page_number=exam["page_number"],
            source_file=exam["source_file"],
            transcription_confidence=exam.get("transcription_confidence"),
            prompt_variant=exam.get("prompt_variant"),
            retry_attempts=exam.get("retry_attempts", 1),
            physician_name=exam.get("physician_name"),
            department=exam.get("department"),
            facility_name=exam.get("facility_name"),
            page_kind=exam.get("page_kind", "text"),
            validation_status=exam.get("validation_status", "ok"),
            failure_type=exam.get("failure_type"),
            source_mode=exam.get("source_mode", "vision"),
            chart_type=exam.get("chart_type"),
            chart_data_status=exam.get("chart_data_status"),
            exam_type=exam.get("exam_type"),
            exam_name_standardized=exam.get("exam_name_standardized"),
        )
