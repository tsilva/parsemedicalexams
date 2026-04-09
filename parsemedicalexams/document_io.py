"""Document and output I/O helpers."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from PIL import Image

from .summarization import summarize_document
from .utils import preprocess_page_image
from .validation import (
    determine_page_strategy,
    first_blocking_issue,
    validate_page_output,
    validate_summary_output,
)

if TYPE_CHECKING:
    from openai import OpenAI

    from .config import ExtractionConfig

logger = logging.getLogger(__name__)


_FRONTMATTER_MAP = {
    "exam_date": "exam_date",
    "exam_name_raw": "exam_name_raw",
    "exam_name_standardized": "title",
    "exam_type": "category",
    "physician_name": "doctor",
    "facility_name": "facility",
    "department": "department",
}

REQUIRED_METADATA_FIELDS = {"exam_date", "exam_name_raw", "category", "title"}
ALLOWED_CATEGORIES = {
    "appointment",
    "endoscopy",
    "imaging",
    "other",
    "prescription",
    "ultrasound",
}
ALLOWED_PAGE_KINDS = {"text", "chart", "image_only"}
ALLOWED_VALIDATION_STATUSES = {"ok", "retryable_failure", "unsupported_visual", "failed"}
ALLOWED_SOURCE_MODES = {"embedded_text", "vision", "hybrid"}
ALLOWED_CHART_TYPES = {
    "audiogram",
    "eeg_signal_trace",
    "sleep_summary_graph",
    "tympanometry",
}
ALLOWED_CHART_DATA_STATUSES = {"ok", "non_discrete_visual"}
PAGE_ONLY_METADATA_FIELDS = {
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
ENHANCED_PAGE_FIELDS = {
    "page_kind",
    "validation_status",
    "failure_type",
    "source_mode",
    "chart_type",
    "chart_data_status",
}
SKIP_MARKER_FILENAME = ".skip"


def extract_doc_date_prefix(name: str) -> str | None:
    """Extract a YYYY-MM-DD prefix from a document stem or filename."""
    match = re.match(r"^(\d{4}-\d{2}-\d{2})\b", name)
    return match.group(1) if match else None


def _is_iso_date_string(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _expected_page_number(md_path: Path, doc_stem: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(doc_stem)}\.(\d+)\.md", md_path.name)
    if not match:
        return None
    return int(match.group(1))


def validate_metadata_frontmatter(
    md_path: Path,
    doc_stem: str,
    frontmatter: dict,
) -> list[str]:
    """Validate deterministic metadata invariants for a markdown output file."""
    issues: list[str] = []
    missing_fields = REQUIRED_METADATA_FIELDS - set(frontmatter.keys())
    if missing_fields:
        issues.append(f"missing_fields={sorted(missing_fields)}")

    exam_date = frontmatter.get("exam_date")
    if exam_date is not None and not _is_iso_date_string(exam_date):
        issues.append("invalid_exam_date_format")

    expected_doc_date = extract_doc_date_prefix(doc_stem)
    if (
        expected_doc_date
        and isinstance(exam_date, str)
        and exam_date
        and exam_date != expected_doc_date
    ):
        issues.append("exam_date_mismatch_doc_prefix")

    category = frontmatter.get("category")
    if category is not None and category not in ALLOWED_CATEGORIES:
        issues.append("invalid_category")

    for field in ("exam_name_raw", "title"):
        value = frontmatter.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            issues.append(f"invalid_{field}")

    is_summary = md_path.name.endswith(".summary.md")
    if is_summary:
        unexpected = sorted(PAGE_ONLY_METADATA_FIELDS & set(frontmatter.keys()))
        if unexpected:
            issues.append(f"summary_has_page_fields={unexpected}")
        return issues

    page = frontmatter.get("page")
    if not isinstance(page, int) or page < 1:
        issues.append("invalid_page_number")
    expected_page = _expected_page_number(md_path, doc_stem)
    if expected_page is not None and page != expected_page:
        issues.append("page_number_mismatch_filename")

    source = frontmatter.get("source")
    if source != f"{doc_stem}.pdf":
        issues.append("source_mismatch_doc_pdf")

    prompt_variant = frontmatter.get("prompt_variant")
    if not isinstance(prompt_variant, str) or not prompt_variant.strip():
        issues.append("missing_prompt_variant")

    if not any(field in frontmatter for field in ENHANCED_PAGE_FIELDS):
        return issues

    page_kind = frontmatter.get("page_kind")
    if page_kind not in ALLOWED_PAGE_KINDS:
        issues.append("invalid_page_kind")

    validation_status = frontmatter.get("validation_status")
    if validation_status not in ALLOWED_VALIDATION_STATUSES:
        issues.append("invalid_validation_status")

    source_mode = frontmatter.get("source_mode")
    if source_mode not in ALLOWED_SOURCE_MODES:
        issues.append("invalid_source_mode")

    chart_type = frontmatter.get("chart_type")
    if chart_type is not None and chart_type not in ALLOWED_CHART_TYPES:
        issues.append("invalid_chart_type")

    chart_data_status = frontmatter.get("chart_data_status")
    if (
        chart_data_status is not None
        and chart_data_status not in ALLOWED_CHART_DATA_STATUSES
    ):
        issues.append("invalid_chart_data_status")

    failure_type = frontmatter.get("failure_type")

    if page_kind == "chart" and not chart_type:
        issues.append("chart_page_missing_chart_type")
    if chart_type and page_kind != "chart":
        issues.append("chart_type_without_chart_page")
    if chart_data_status and not chart_type:
        issues.append("chart_data_status_without_chart_type")
    if page_kind == "image_only" and chart_type:
        issues.append("image_only_page_has_chart_type")

    if chart_type in {"audiogram", "tympanometry"} and chart_data_status != "ok":
        issues.append("discrete_chart_missing_ok_status")
    if (
        chart_type in {"sleep_summary_graph", "eeg_signal_trace"}
        and chart_data_status != "non_discrete_visual"
    ):
        issues.append("non_discrete_chart_missing_marker_status")

    if validation_status == "unsupported_visual" and page_kind == "text":
        issues.append("text_page_marked_unsupported_visual")
    if validation_status == "ok" and failure_type:
        issues.append("ok_page_has_failure_type")
    if validation_status in {"retryable_failure", "failed"} and not failure_type:
        issues.append("failed_page_missing_failure_type")
    if failure_type and validation_status not in {"retryable_failure", "failed"}:
        issues.append("failure_type_without_failed_status")

    return issues


def transcription_files(doc_dir: Path, doc_stem: str) -> list[Path]:
    """Return all transcription .md files (excluding .summary.md) in doc_dir."""
    return [
        path
        for path in doc_dir.glob(f"{doc_stem}.*.md")
        if not path.name.endswith(".summary.md")
    ]


def skip_marker_path(doc_output_dir: Path) -> Path:
    """Return the skip marker path for a document output directory."""
    return doc_output_dir / SKIP_MARKER_FILENAME


def write_skip_marker(doc_output_dir: Path, pdf_name: str, reason: str) -> None:
    """Persist a skip marker so non-exam documents are not retried automatically."""
    skip_marker_path(doc_output_dir).write_text(
        f"status: skipped\nsource: {pdf_name}\nreason: {reason}\n",
        encoding="utf-8",
    )


def remove_skip_marker(doc_output_dir: Path) -> None:
    """Remove a stale skip marker if present."""
    marker = skip_marker_path(doc_output_dir)
    if marker.exists():
        marker.unlink()


def pdf_copy_is_current(source_pdf: Path, copied_pdf: Path) -> bool:
    """Return True when the copied PDF still matches the source PDF metadata."""
    if not copied_pdf.exists():
        return False

    source_stat = source_pdf.stat()
    copied_stat = copied_pdf.stat()
    return (
        source_stat.st_size == copied_stat.st_size
        and source_stat.st_mtime_ns == copied_stat.st_mtime_ns
    )


def count_pdf_pages(pdf_path: Path) -> int:
    """Count pages in a PDF, using metadata tools before raster fallback."""
    try:
        from pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(str(pdf_path))
        pages = info.get("Pages")
        if isinstance(pages, int) and pages > 0:
            return pages
    except Exception:
        pass

    return len(convert_pdf_to_images(pdf_path))


def build_exam_frontmatter(exam: dict, extra_fields: dict | None = None) -> dict:
    """Build YAML frontmatter dict from an exam dict."""
    frontmatter = {
        frontmatter_key: exam[exam_key]
        for exam_key, frontmatter_key in _FRONTMATTER_MAP.items()
        if exam.get(exam_key)
    }
    if extra_fields:
        frontmatter.update(extra_fields)
    return frontmatter


def write_markdown_with_frontmatter(path: Path, frontmatter: dict, body: str) -> None:
    """Write a markdown file with YAML frontmatter."""
    with path.open("w", encoding="utf-8") as handle:
        if frontmatter:
            handle.write("---\n")
            handle.write(
                yaml.dump(
                    frontmatter,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            )
            handle.write("---\n\n")
        handle.write(body)


def purge_derived_outputs(
    doc_output_dir: Path, doc_stem: str, remove_images: bool
) -> None:
    """Delete generated page markdown and summary outputs for a document."""
    if not doc_output_dir.exists():
        return

    for md_path in doc_output_dir.glob(f"{doc_stem}.*.md"):
        md_path.unlink()
    for summary_path in doc_output_dir.glob("*.summary.md"):
        summary_path.unlink()
    if remove_images:
        for image_path in doc_output_dir.glob(f"{doc_stem}.*.jpg"):
            image_path.unlink()
    remove_skip_marker(doc_output_dir)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content."""
    frontmatter = {}
    transcription = content.strip()

    if transcription.startswith("---"):
        end_marker = transcription.find("---", 3)
        if end_marker != -1:
            frontmatter_str = transcription[3:end_marker].strip()
            try:
                frontmatter = yaml.safe_load(frontmatter_str) or {}
            except yaml.YAMLError:
                pass
            transcription = transcription[end_marker + 3 :].strip()

    return frontmatter, transcription


def frontmatter_to_exam(
    frontmatter: dict,
    transcription: str,
    page_num: int,
    source_file: str | None = None,
) -> dict:
    """Convert frontmatter fields to internal exam dict format."""
    inverse_map = {frontmatter_key: exam_key for exam_key, frontmatter_key in _FRONTMATTER_MAP.items()}
    result = {
        exam_key: frontmatter.get(frontmatter_key)
        for frontmatter_key, exam_key in inverse_map.items()
    }
    result["transcription_confidence"] = frontmatter.get("confidence")
    result["transcription"] = transcription
    result["page_number"] = frontmatter.get("page") or page_num
    result["source_file"] = frontmatter.get("source") or source_file
    result["page_kind"] = frontmatter.get("page_kind") or "text"
    result["validation_status"] = frontmatter.get("validation_status") or "ok"
    result["failure_type"] = frontmatter.get("failure_type")
    result["source_mode"] = frontmatter.get("source_mode") or "vision"
    result["chart_type"] = frontmatter.get("chart_type")
    result["chart_data_status"] = frontmatter.get("chart_data_status")
    return result


def _validate_existing_transcription_file(md_path: Path) -> list[str]:
    frontmatter, transcription = parse_frontmatter(md_path.read_text(encoding="utf-8"))
    doc_stem = md_path.name.rsplit(".", 2)[0]
    problems = validate_metadata_frontmatter(md_path, doc_stem, frontmatter)
    inferred_page_kind, inferred_chart_type, _ = determine_page_strategy(
        transcription,
        document_exam_name=frontmatter.get("exam_name_raw", ""),
    )
    page_kind = frontmatter.get("page_kind") or inferred_page_kind
    chart_type = frontmatter.get("chart_type") or inferred_chart_type
    page = frontmatter.get("page")
    status = frontmatter.get("validation_status")
    chart_data_status = frontmatter.get("chart_data_status")
    issues = validate_page_output(
        transcription,
        page_kind=page_kind,
        chart_type=chart_type,
        page=page,
    )
    blocking_issue = first_blocking_issue(issues)
    if blocking_issue:
        problems.append(blocking_issue.kind)
    if status in {"retryable_failure", "failed"}:
        problems.append(f"stored_{status}")
    if page_kind == "text" and status == "unsupported_visual":
        problems.append("text_page_marked_unsupported_visual")
    if chart_type == "audiogram" and chart_data_status != "ok":
        problems.append("audiogram_missing_chart_data")
    return problems


def _validate_existing_summary_file(summary_path: Path) -> list[str]:
    frontmatter, summary = parse_frontmatter(summary_path.read_text(encoding="utf-8"))
    doc_stem = summary_path.name[: -len(".summary.md")]
    problems = validate_metadata_frontmatter(summary_path, doc_stem, frontmatter)
    blocking_issue = first_blocking_issue(validate_summary_output(summary))
    if blocking_issue:
        problems.append(blocking_issue.kind)
    return problems


def get_document_output_issue(pdf_path: Path, output_path: Path) -> str | None:
    """Return why a document output is incomplete, or None when it is complete."""
    doc_stem = pdf_path.stem
    doc_output_dir = output_path / doc_stem
    if not doc_output_dir.exists():
        return "missing output folder"

    copied_pdf = doc_output_dir / pdf_path.name
    if not copied_pdf.exists():
        return "missing copied source PDF"
    if not pdf_copy_is_current(pdf_path, copied_pdf):
        return "source PDF changed since last processing"

    if skip_marker_path(doc_output_dir).exists():
        return None

    markdown_files = transcription_files(doc_output_dir, doc_stem)
    if not markdown_files:
        return "missing transcription markdown files"

    jpg_files = list(doc_output_dir.glob(f"{doc_stem}.*.jpg"))
    if not jpg_files:
        return "missing page images"

    expected_page_count = count_pdf_pages(copied_pdf)
    if len(jpg_files) != expected_page_count:
        return (
            f"page image count mismatch ({len(jpg_files)} images, "
            f"{expected_page_count} PDF pages)"
        )

    if len(jpg_files) != len(markdown_files):
        return (
            f"page count mismatch ({len(jpg_files)} images, "
            f"{len(markdown_files)} transcriptions)"
        )

    summary_path = doc_output_dir / f"{doc_stem}.summary.md"
    if not summary_path.exists():
        return "missing document summary"

    for md_path in markdown_files:
        problems = _validate_existing_transcription_file(md_path)
        if problems:
            return f"invalid transcription output in {md_path.name}: {', '.join(problems)}"

    summary_problems = _validate_existing_summary_file(summary_path)
    if summary_problems:
        return f"invalid summary output: {', '.join(summary_problems)}"

    return None


def is_document_processed(pdf_path: Path, output_path: Path) -> bool:
    """Check if a PDF already has a complete output bundle."""
    return get_document_output_issue(pdf_path, output_path) is None


def convert_pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    """Convert a PDF into PIL images, using pdftoppm as a fallback."""
    try:
        from pdf2image import convert_from_path

        return convert_from_path(str(pdf_path))
    except ModuleNotFoundError:
        if not shutil.which("pdftoppm"):
            raise

        with tempfile.TemporaryDirectory() as render_dir:
            prefix = Path(render_dir) / "page"
            result = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "150", str(pdf_path), str(prefix)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "pdftoppm failed")

            image_paths = sorted(Path(render_dir).glob("page-*.jpg"))
            images = []
            for image_path in image_paths:
                with Image.open(image_path) as img:
                    images.append(img.copy())
            return images


def extract_pdf_page_text(pdf_path: Path, page_num: int) -> str:
    """Extract embedded PDF text for a specific page using pdftotext."""
    try:
        result = subprocess.run(
            [
                "pdftotext",
                "-layout",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                str(pdf_path),
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("pdftotext not available, falling back to vision OCR")
        return ""

    if result.returncode != 0:
        logger.debug(
            "pdftotext failed for %s page %s: %s",
            pdf_path.name,
            page_num,
            result.stderr.strip(),
        )
        return ""

    return result.stdout.replace("\f", "").strip()


def save_transcription_file(
    exams: list[dict], doc_output_dir: Path, doc_stem: str, page_num: int
) -> None:
    """Save a page transcription as markdown file with YAML frontmatter."""
    md_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.md"

    frontmatter: dict = {}
    if exams:
        exam = exams[0]
        extra = {
            key: exam[exam_key]
            for key, exam_key in [
                ("page", "page_number"),
                ("source", "source_file"),
                ("prompt_variant", "prompt_variant"),
                ("page_kind", "page_kind"),
                ("validation_status", "validation_status"),
                ("failure_type", "failure_type"),
                ("source_mode", "source_mode"),
                ("chart_type", "chart_type"),
                ("chart_data_status", "chart_data_status"),
            ]
            if exam.get(exam_key)
        }
        if exam.get("transcription_confidence") is not None:
            extra["confidence"] = exam["transcription_confidence"]
        if (exam.get("retry_attempts") or 0) > 1:
            extra["retry_attempts"] = exam["retry_attempts"]
        frontmatter = build_exam_frontmatter(exam, extra)

    transcriptions = [exam.get("transcription", "") for exam in exams]
    write_markdown_with_frontmatter(
        md_path, frontmatter, "\n\n".join(transcriptions).strip() + "\n"
    )


def save_document_summary(
    summary: str, doc_output_dir: Path, doc_stem: str, exams: list[dict] | None = None
) -> None:
    """Save a document-level summary as markdown file with YAML frontmatter."""
    if not summary:
        return

    summary_path = doc_output_dir / f"{doc_stem}.summary.md"
    frontmatter = build_exam_frontmatter(exams[0]) if exams else {}
    write_markdown_with_frontmatter(summary_path, frontmatter, summary.strip() + "\n")


def copy_source_pdf(source_pdf: Path, doc_output_dir: Path) -> None:
    """Copy or refresh the source PDF inside a document output directory."""
    copied_pdf = doc_output_dir / source_pdf.name
    if pdf_copy_is_current(source_pdf, copied_pdf):
        return
    try:
        shutil.copy2(source_pdf, copied_pdf)
    except PermissionError:
        logger.warning(
            "Could not copy PDF to output (permission denied): %s",
            source_pdf.name,
        )


def preprocess_pdf_images_to_temp(pdf_path: Path, doc_stem: str):
    """Convert and preprocess PDF pages into a temporary directory."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    temp_image_paths: list[Path] = []

    for page_num, page_image in enumerate(convert_pdf_to_images(pdf_path), start=1):
        processed_image = preprocess_page_image(page_image)
        temp_image_path = temp_path / f"{doc_stem}.{page_num:03d}.jpg"
        processed_image.save(str(temp_image_path), "JPEG", quality=80)
        temp_image_paths.append(temp_image_path)

    return temp_dir, temp_image_paths


def persist_temp_images(temp_image_paths: list[Path], doc_output_dir: Path) -> list[Path]:
    """Copy preprocessed temp images into the document output directory."""
    image_paths: list[Path] = []
    for temp_image_path in temp_image_paths:
        final_image_path = doc_output_dir / temp_image_path.name
        shutil.copy2(temp_image_path, final_image_path)
        image_paths.append(final_image_path)
    return image_paths


def regenerate_summaries(
    output_path: Path,
    config: "ExtractionConfig",
    client: "OpenAI",
    input_path: Path | None = None,
    doc_filter: str | None = None,
) -> int:
    """Regenerate document-level summary files from existing transcription markdown files."""
    doc_dirs = [doc_dir for doc_dir in output_path.iterdir() if doc_dir.is_dir() and doc_dir.name != "logs"]

    if doc_filter:
        query = doc_filter.lower()
        query_stem = query[:-4] if query.endswith(".pdf") else query
        doc_dirs = [
            doc_dir
            for doc_dir in doc_dirs
            if doc_dir.name.lower() == query_stem or doc_dir.name.lower() == query
        ]
        if not doc_dirs:
            logger.error("No matching document directory found for: %s", doc_filter)
            return 0

    logger.info("Found %s document directories to regenerate", len(doc_dirs))

    total_exams = 0
    for doc_dir in doc_dirs:
        doc_stem = doc_dir.name

        jpg_files = list(doc_dir.glob(f"{doc_stem}.*.jpg"))
        md_transcription_files = transcription_files(doc_dir, doc_stem)
        if jpg_files and len(jpg_files) != len(md_transcription_files):
            logger.error(
                "Skipping %s: page count mismatch — %s images but %s transcriptions",
                doc_stem,
                len(jpg_files),
                len(md_transcription_files),
            )
            continue

        if input_path:
            source_pdfs = list(input_path.glob(f"**/{doc_stem}.pdf"))
            if source_pdfs:
                copy_source_pdf(source_pdfs[0], doc_dir)

        md_files = sorted(transcription_files(doc_dir, doc_stem))
        if not md_files:
            logger.warning("No transcription files found in %s", doc_dir)
            continue

        all_exams = []
        for md_path in md_files:
            parts = md_path.stem.split(".")
            if len(parts) < 2:
                continue
            try:
                page_num = int(parts[-1])
            except ValueError:
                continue

            frontmatter, transcription = parse_frontmatter(
                md_path.read_text(encoding="utf-8")
            )
            all_exams.append(
                frontmatter_to_exam(frontmatter, transcription, page_num, f"{doc_stem}.pdf")
            )

        if not all_exams:
            logger.warning("No exams found in %s", doc_dir)
            continue

        for old_summary in doc_dir.glob("*.summary.md"):
            old_summary.unlink()

        document_summary = summarize_document(
            all_exams,
            config.summarize_model_id,
            client,
            max_input_tokens=config.summarize_max_input_tokens,
        )
        if first_blocking_issue(validate_summary_output(document_summary)):
            logger.error("Summary validation failed for %s", doc_stem)
            continue

        save_document_summary(document_summary, doc_dir, doc_stem, all_exams)
        logger.info("Regenerated summary for %s exams: %s", len(all_exams), doc_stem)
        total_exams += len(all_exams)

    return total_exams


def validate_pipeline_outputs(pdf_files: list[Path], output_path: Path) -> list[str]:
    """Validate expected output files exist and pass content validation."""
    issues = []
    for pdf_path in pdf_files:
        issue = get_document_output_issue(pdf_path, output_path)
        if issue:
            issues.append(f"{pdf_path.stem}: {issue}")
    return issues


def validate_frontmatter(output_path: Path) -> list[str]:
    """Validate that all .md files have YAML frontmatter with required fields."""
    issues = []
    doc_dirs = [doc_dir for doc_dir in output_path.iterdir() if doc_dir.is_dir() and doc_dir.name != "logs"]
    for doc_dir in doc_dirs:
        doc_stem = doc_dir.name
        for md_path in doc_dir.glob(f"{doc_stem}.*.md"):
            try:
                frontmatter, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
            except Exception as exc:
                issues.append(f"Error reading {md_path.name}: {exc}")
                continue

            if not frontmatter:
                issues.append(f"Missing frontmatter: {md_path.name}")
                continue

            for problem in validate_metadata_frontmatter(md_path, doc_stem, frontmatter):
                if problem == "exam_date_mismatch_doc_prefix":
                    exam_date = frontmatter.get("exam_date")
                    expected_doc_date = extract_doc_date_prefix(doc_stem)
                    issues.append(
                        f"Exam date {exam_date} does not match document date prefix {expected_doc_date}: {md_path.name}"
                    )
                else:
                    issues.append(f"{problem}: {md_path.name}")
    return issues
