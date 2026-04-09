"""Validation helpers for page and summary outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

HARD_FAILURE_PATTERNS = [
    re.compile(r"Request too large\. Try with a smaller file\.", re.IGNORECASE),
]

MODEL_NARRATION_PATTERNS = [
    re.compile(r"\bThis image shows\b", re.IGNORECASE),
    re.compile(r"\bNo readable text is visible on this page\b", re.IGNORECASE),
    re.compile(r"\bNo readable text is clearly visible\b", re.IGNORECASE),
    re.compile(r"\bThe following text elements are partially visible\b", re.IGNORECASE),
    re.compile(r"\bThe remaining technical parameters\b", re.IGNORECASE),
    re.compile(r"\bImage contains ultrasound scans only\b", re.IGNORECASE),
    re.compile(
        r"\bThe page consists of multiple medical ultrasound image frames\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bThe page consists of ultrasound/scan images\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bThere is no clearly readable text on this page image\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btoo small/blurred to transcribe exactly\b",
        re.IGNORECASE,
    ),
]

SOFT_WARNING_PATTERNS = [
    re.compile(r"\bUnknown Institution\b", re.IGNORECASE),
]

AUDIOGRAM_FREQUENCY_TOKENS = [
    "125 Hz",
    "250 Hz",
    "500 Hz",
    "750 Hz",
    "1K Hz",
    "1.5K Hz",
    "2K Hz",
    "3K Hz",
    "4K Hz",
    "6K Hz",
    "8K Hz",
    "12K Hz",
]


@dataclass(frozen=True)
class OutputIssue:
    kind: str
    severity: str
    scope: str
    page: Optional[int]
    reason: str
    snippet: str = ""


def normalize_body(text: str) -> str:
    return text.strip()


def has_meaningful_embedded_text(text: str) -> bool:
    body = normalize_body(text)
    words = re.findall(r"\S+", body)
    return len(body) >= 200 and len(words) >= 30


def detect_chart_type(text: str) -> Optional[str]:
    lower = normalize_body(text).lower()
    if not lower:
        return None
    if "[tympanometry]" in lower:
        return "tympanometry"
    if "[audiogram]" in lower:
        return "audiogram"
    if "[non_discrete_visual_chart]" in lower:
        marker_match = re.search(r"chart_type:\s*([a-z_]+)", lower)
        if marker_match:
            return marker_match.group(1)
    if "summary graph" in lower and "spo2" in lower and "arousal" in lower:
        return "sleep_summary_graph"
    if (
        "tímpano 226 hz direito" in lower
        and "tímpano 226 hz esquerdo" in lower
        and "pressão" in lower
        and "complacência" in lower
        and "dapa" in lower
    ):
        return "tympanometry"
    if "audiograma vocal" in lower or (
        "ouvido direito" in lower and "ouvido esquerdo" in lower
    ):
        return "audiogram"
    return None


def determine_page_strategy(
    embedded_text: str,
    document_exam_name: str = "",
) -> tuple[str, Optional[str], str]:
    chart_type = detect_chart_type(embedded_text)
    if chart_type:
        return ("chart", chart_type, "hybrid")
    exam_name = normalize_body(document_exam_name).lower()
    if not normalize_body(embedded_text) and (
        "electroencefalografia" in exam_name
        or re.search(r"\be\.?e\.?g\b", exam_name)
    ):
        return ("chart", "eeg_signal_trace", "hybrid")
    if has_meaningful_embedded_text(embedded_text):
        return ("text", None, "embedded_text")
    return ("text", None, "vision")


def build_no_readable_text_marker() -> str:
    return "[NO_READABLE_TEXT]"


def build_non_discrete_chart_marker(chart_type: str, visible_labels: Iterable[str]) -> str:
    labels = ", ".join(sorted({label for label in visible_labels if label})) or "none"
    return (
        "[NON_DISCRETE_VISUAL_CHART]\n"
        f"chart_type: {chart_type}\n"
        "chart_data_status: non_discrete_visual\n"
        f"visible_labels: {labels}"
    )


def derive_validation_metadata(
    issues: list[OutputIssue], page_kind: str, chart_type: Optional[str]
) -> dict[str, Optional[str]]:
    blocking = first_blocking_issue(issues)
    if blocking:
        status = "retryable_failure"
        failure_type = blocking.kind
    elif page_kind == "image_only":
        status = "unsupported_visual"
        failure_type = None
    elif chart_type in {"sleep_summary_graph", "eeg_signal_trace"}:
        status = "unsupported_visual"
        failure_type = None
    else:
        status = "ok"
        failure_type = None

    if chart_type in {"sleep_summary_graph", "eeg_signal_trace"}:
        chart_data_status = "non_discrete_visual"
    elif chart_type in {"audiogram", "tympanometry"} and not blocking:
        chart_data_status = "ok"
    else:
        chart_data_status = None

    return {
        "validation_status": status,
        "failure_type": failure_type,
        "chart_data_status": chart_data_status,
    }


def first_blocking_issue(issues: list[OutputIssue]) -> Optional[OutputIssue]:
    return next((issue for issue in issues if issue.severity == "blocking"), None)


def _match_issues(
    text: str,
    patterns: list[re.Pattern[str]],
    kind: str,
    severity: str,
    scope: str,
    page: Optional[int],
    reason: str,
) -> list[OutputIssue]:
    issues: list[OutputIssue] = []
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            issues.append(
                OutputIssue(
                    kind=kind,
                    severity=severity,
                    scope=scope,
                    page=page,
                    reason=reason,
                    snippet=match.group(0),
                )
            )
    return issues


def validate_page_output(
    text: str,
    page_kind: str = "text",
    chart_type: Optional[str] = None,
    page: Optional[int] = None,
) -> list[OutputIssue]:
    body = normalize_body(text)
    issues: list[OutputIssue] = []
    issues.extend(
        _match_issues(
            body,
            HARD_FAILURE_PATTERNS,
            "hard_failure",
            "blocking",
            "page",
            page,
            "Sentinel hard failure string present in page output",
        )
    )
    issues.extend(
        _match_issues(
            body,
            MODEL_NARRATION_PATTERNS,
            "model_narration",
            "blocking",
            "page",
            page,
            "Model narration present instead of transcription",
        )
    )
    issues.extend(
        _match_issues(
            body,
            SOFT_WARNING_PATTERNS,
            "placeholder_metadata",
            "warning",
            "page",
            page,
            "Placeholder metadata phrase detected",
        )
    )

    if page_kind == "image_only":
        if body != build_no_readable_text_marker():
            issues.append(
                OutputIssue(
                    kind="invalid_image_only_marker",
                    severity="blocking",
                    scope="page",
                    page=page,
                    reason="Image-only pages must use the structured no-text marker",
                    snippet=body[:120],
                )
            )
        return issues

    if chart_type in {"sleep_summary_graph", "eeg_signal_trace"}:
        if not body.startswith("[NON_DISCRETE_VISUAL_CHART]"):
            issues.append(
                OutputIssue(
                    kind="chart_scaffold_only"
                    if chart_type == "sleep_summary_graph"
                    else "signal_trace_requires_marker",
                    severity="blocking",
                    scope="page",
                    page=page,
                    reason=(
                        "Sleep summary graph must use the non-discrete visual marker"
                        if chart_type == "sleep_summary_graph"
                        else "Signal-trace pages must use the non-discrete visual marker"
                    ),
                    snippet=body[:160],
                )
            )
        return issues

    if len(body) < 20:
        issues.append(
            OutputIssue(
                kind="empty_output",
                severity="blocking",
                scope="page",
                page=page,
                reason="Page output is empty or too short",
                snippet=body,
            )
        )

    illegible_tokens = body.lower().count("[illegible]")
    alpha_words = re.findall(r"[a-zà-ÿ]{4,}", body.lower())
    alpha_non_illegible = [word for word in alpha_words if word != "illegible"]
    residual_after_illegible = re.sub(r"\[illegible\]", "", body, flags=re.IGNORECASE)
    if (
        illegible_tokens >= 1
        and not re.search(r"[a-zà-ÿ]{4,}", residual_after_illegible.lower())
    ) or (illegible_tokens >= 4 and len(alpha_non_illegible) <= 3):
        issues.append(
            OutputIssue(
                kind="illegible_only",
                severity="blocking",
                scope="page",
                page=page,
                reason="Page output is only illegible placeholders, not a usable transcription",
                snippet=body[:160],
            )
        )

    fragment_lines = [line.strip() for line in body.splitlines() if line.strip()]
    short_fragment_lines = sum(
        1
        for line in fragment_lines
        if len(line) <= 80 and not re.search(r"[.!?]", line)
    )
    if illegible_tokens >= 1 and len(alpha_non_illegible) <= 8 and short_fragment_lines >= 3:
        issues.append(
            OutputIssue(
                kind="low_signal_ocr",
                severity="blocking",
                scope="page",
                page=page,
                reason="Page output only contains low-signal OCR fragments and illegible placeholders",
                snippet=body[:160],
            )
        )

    if chart_type == "audiogram":
        if "[AUDIOGRAM]" not in body:
            issues.append(
                OutputIssue(
                    kind="chart_scaffold_only",
                    severity="blocking",
                    scope="page",
                    page=page,
                    reason="Audiogram pages must use the structured audiogram format",
                    snippet=body[:160],
                )
            )
            return issues

        frequency_hits = sum(1 for token in AUDIOGRAM_FREQUENCY_TOKENS if token in body)
        if frequency_hits < 6 or "Right ear thresholds" not in body or "Left ear thresholds" not in body:
            issues.append(
                OutputIssue(
                    kind="chart_scaffold_only",
                    severity="blocking",
                    scope="page",
                    page=page,
                    reason="Audiogram output is missing extracted threshold data",
                    snippet=body[:200],
                )
            )

    if chart_type == "tympanometry":
        if "[TYMPANOMETRY]" not in body:
            issues.append(
                OutputIssue(
                    kind="chart_scaffold_only",
                    severity="blocking",
                    scope="page",
                    page=page,
                    reason="Tympanometry pages must use the structured tympanometry format",
                    snippet=body[:160],
                )
            )
            return issues

        required_snippets = [
            "Right ear:",
            "Left ear:",
            "- Volume:",
            "- Pressure:",
            "- Compliance:",
            "- Gradient:",
        ]
        missing = [snippet for snippet in required_snippets if snippet not in body]
        if missing:
            issues.append(
                OutputIssue(
                    kind="chart_scaffold_only",
                    severity="blocking",
                    scope="page",
                    page=page,
                    reason="Tympanometry output is missing required per-ear fields",
                    snippet=", ".join(missing[:4]),
                )
            )

    return issues


def validate_summary_output(text: str) -> list[OutputIssue]:
    body = normalize_body(text)
    issues: list[OutputIssue] = []
    issues.extend(
        _match_issues(
            body,
            HARD_FAILURE_PATTERNS,
            "hard_failure",
            "blocking",
            "summary",
            None,
            "Sentinel hard failure string present in summary output",
        )
    )
    issues.extend(
        _match_issues(
            body,
            MODEL_NARRATION_PATTERNS,
            "model_narration",
            "blocking",
            "summary",
            None,
            "Model narration present instead of summary content",
        )
    )
    if len(body) < 40:
        issues.append(
            OutputIssue(
                kind="empty_output",
                severity="blocking",
                scope="summary",
                page=None,
                reason="Summary output is empty or too short",
                snippet=body,
            )
        )
    return issues
