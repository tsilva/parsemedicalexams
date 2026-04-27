"""Pipeline orchestration for medical exam processing."""

from __future__ import annotations

import logging
import re
import tempfile
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import httpx
from openai import APIError, OpenAI
from PIL import Image  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from .config import ExtractionConfig, ProfileConfig
from .document_io import (
    collect_output_assertions,
    convert_pdf_to_images,
    copy_source_pdf,
    count_pdf_pages,
    extract_pdf_page_text,
    get_document_output_issue,
    persist_temp_images,
    preprocess_pdf_images_to_temp,
    purge_derived_outputs,
    regenerate_summaries,
    remove_skip_marker,
    save_document_summary,
    save_transcription_file,
    write_skip_marker,
)
from .extraction import (
    build_chart_user_prompt,
    classify_document,
    score_transcription_confidence,
    self_consistency,
    transcribe_page,
    transcribe_with_retry,
)
from .models import ExamRecord
from .standardization import standardize_exam_types
from .summarization import summarize_document
from .utils import extract_dates_from_text, setup_logging
from .validation import (
    build_no_readable_text_marker,
    build_non_discrete_chart_marker,
    derive_validation_metadata,
    determine_page_strategy,
    first_blocking_issue,
    validate_page_output,
    validate_summary_output,
)

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    PROCESS = "process"
    DRY_RUN = "dry_run"
    AUDIT = "audit"
    REGENERATE = "regenerate"
    RESUMMARIZE = "resummarize"


def resolve_run_mode(args: Namespace) -> RunMode:
    """Resolve the primary run mode from CLI arguments."""
    if args.audit_outputs:
        return RunMode.AUDIT
    if args.resummarize:
        return RunMode.RESUMMARIZE
    if args.regenerate:
        return RunMode.REGENERATE
    if args.dry_run:
        return RunMode.DRY_RUN
    return RunMode.PROCESS


def discover_pdf_files(
    input_path: Path, input_file_regex: str, document: str | None = None
) -> list[Path]:
    """Discover candidate PDFs, preferring a direct requested path when present."""
    pdf_pattern = re.compile(input_file_regex)
    if document:
        direct_candidates = [input_path / document]
        if not document.lower().endswith(".pdf"):
            direct_candidates.append(input_path / f"{document}.pdf")
        direct_matches = [candidate for candidate in direct_candidates if candidate.exists()]
        if direct_matches:
            return direct_matches

    try:
        next(input_path.iterdir(), None)
    except PermissionError as exc:
        raise PermissionError(
            f"Input path is not readable: {input_path}. "
            "Grant the running app read access to that Google Drive folder and try again."
        ) from exc

    return sorted(
        [
            pdf_path
            for pdf_path in input_path.glob("**/*.pdf")
            if pdf_pattern.match(pdf_path.name)
        ]
    )


def match_requested_document(pdf_files: list[Path], document: str) -> list[Path]:
    """Match a requested document by filename or stem, case-insensitively."""
    doc_query = document.lower()
    doc_query_stem = doc_query[:-4] if doc_query.endswith(".pdf") else doc_query
    matches = [
        pdf_path
        for pdf_path in pdf_files
        if pdf_path.name.lower() == doc_query or pdf_path.stem.lower() == doc_query_stem
    ]
    if not matches:
        raise ValueError(f"Document not found: {document}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for '{document}': {[match.name for match in matches]}")
    return matches


def _existing_image_page_numbers(image_paths: list[Path], doc_stem: str) -> list[int | None]:
    """Return page numbers encoded in existing image filenames."""
    page_numbers: list[int | None] = []
    pattern = re.compile(rf"^{re.escape(doc_stem)}\.(\d+)\.jpg$")
    for image_path in image_paths:
        match = pattern.fullmatch(image_path.name)
        page_numbers.append(int(match.group(1)) if match else None)
    return page_numbers


def reusable_existing_images(
    image_paths: list[Path], doc_stem: str, expected_page_count: int
) -> bool:
    """Return True when existing images are a complete ordered page set."""
    return _existing_image_page_numbers(image_paths, doc_stem) == list(
        range(1, expected_page_count + 1)
    )


def select_documents_to_process(
    pdf_files: list[Path],
    output_path: Path,
    document: str | None,
    reprocess_all: bool,
) -> tuple[list[Path], int]:
    """Select which documents should be processed for a run."""
    already_processed = 0

    if document:
        return match_requested_document(pdf_files, document), already_processed

    if reprocess_all:
        return pdf_files, already_processed

    to_process: list[Path] = []
    for pdf_path in pdf_files:
        output_issue = get_document_output_issue(pdf_path, output_path)
        if output_issue is None:
            logger.info(
                "Skipping (already processed): %s -> %s",
                pdf_path.name,
                output_path / pdf_path.stem,
            )
            already_processed += 1
        else:
            logger.info(
                "Reprocessing incomplete output: %s (%s)",
                pdf_path.name,
                output_issue,
            )
            to_process.append(pdf_path)

    return to_process, already_processed


def build_profile_context(profile: ProfileConfig) -> str:
    """Build patient context injected into extraction prompts."""
    if not profile.birth_date and not profile.full_name:
        return ""

    parts = []
    if profile.full_name:
        parts.append(f"Patient name: {profile.full_name}")
    if profile.birth_date:
        parts.append(f"Patient date of birth: {profile.birth_date}")
        parts.append(
            "IMPORTANT: "
            f"{profile.birth_date} is the patient's birth date — NEVER use it as exam_date"
        )
    if profile.locale:
        parts.append(
            f"Locale: {profile.locale} (dates in documents typically use DD/MM/YYYY format)"
        )
    return "PATIENT CONTEXT:\n" + "\n".join(parts)


def _extract_visible_chart_labels(text: str) -> list[str]:
    labels = []
    for label in [
        "Arousal",
        "U",
        "W",
        "R",
        "N1",
        "N2",
        "N3",
        "Supine",
        "Left",
        "Prone",
        "Right",
        "Upright",
        "PLM",
        "Desat",
        "SpO2",
        "Heart Rate",
        "Snore",
        "RMI",
        "Central",
        "Apnea",
        "Obstructive",
        "Hypopnea",
        "Autonomic",
    ]:
        if label.lower() in text.lower():
            labels.append(label)
    return labels


def _extract_visible_signal_labels(
    text: str, document_date: str | None = None
) -> list[str]:
    """Extract a conservative set of visible labels from EEG trace OCR."""
    body = text.strip()
    if not body:
        return []

    suspicious_patterns = [
        r"\bPatient\s*:",
        r"\bComputer analysis\b",
        r"\bECG:\s*12\s*Lead\b",
        r"\bRhythm\s*:",
        r"\bInterpretation\s*:",
        r"\bEcho\s*:",
        r"\bCat\s*:",
    ]
    if any(re.search(pattern, body, re.IGNORECASE) for pattern in suspicious_patterns):
        return []

    if document_date:
        doc_year = document_date[:4]
        years = re.findall(r"\b(?:19|20)\d{2}\b", body)
        if any(year != doc_year for year in years):
            return []

    labels: set[str] = set()
    channel_pattern = re.compile(
        r"\b(?:FP1|FP2|F7|F8|F3|F4|FZ|CZ|PZ|T3|T4|T5|T6|C3|C4|P3|P4|O1|O2|A1|A2)\s*-\s*"
        r"(?:FP1|FP2|F7|F8|F3|F4|FZ|CZ|PZ|T3|T4|T5|T6|C3|C4|P3|P4|O1|O2|A1|A2)\b",
        re.IGNORECASE,
    )
    labels.update(match.group(0).upper() for match in channel_pattern.finditer(body))

    keyword_patterns = [
        r"Post\s+HV\s+\d+\s+Sec",
        r"Gain:\s*[\d.]+\s*uV/mm",
        r"LFF:\s*[\d.]+\s*Hz",
        r"HFF:\s*[\d.]+\s*Hz",
        r"Notch:\s*[\d.]+\s*Hz",
        r"Page:\s*\d+",
    ]
    for pattern in keyword_patterns:
        for match in re.finditer(pattern, body, re.IGNORECASE):
            labels.add(match.group(0))

    return sorted(labels)


def _transcribe_rotated_image(
    image_path: Path,
    model_id: str,
    client: OpenAI,
    validation_model_id: str,
    profile_context: str,
) -> tuple[str, str, int]:
    """Rotate image 180 degrees and run the standard transcription retry flow."""
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
        with Image.open(image_path) as img:
            rotated = img.rotate(180, expand=True)
            rotated.save(tmp_file.name, "JPEG", quality=80)
        return transcribe_with_retry(
            image_path=Path(tmp_file.name),
            model_id=model_id,
            client=client,
            validation_model_id=validation_model_id,
            temperature=0.1,
            profile_context=profile_context,
            max_retries=1,
            page_kind="text",
        )


def _should_mark_unsupported_visual(
    embedded_text: str,
    transcription: str,
    blocking_kind: str | None = None,
    chart_type: str | None = None,
) -> bool:
    """Heuristic for image-only pages that do not contain readable text."""
    if chart_type:
        return False
    if embedded_text.strip():
        return False

    body = transcription.strip()
    if blocking_kind in {
        "empty_output",
        "model_narration",
        "illegible_only",
        "low_signal_ocr",
    }:
        return True
    if not body:
        return True

    lowered = body.lower()
    words = re.findall(r"\w+", lowered)
    if lowered == build_no_readable_text_marker().lower():
        return True
    if body.count("[illegible]") >= 2 and len(words) <= 25:
        return True
    if (
        len(body) < 120
        and body.count("\n") >= 3
        and len(re.findall(r"[a-zà-ÿ]{4,}", lowered)) < 8
    ):
        return True
    return False


def extract_date_from_filename(filename: str) -> str | None:
    """Try to extract date from filename in YYYY-MM-DD format."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)

    match = re.search(r"(\d{4})_(\d{2})_(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return None


def select_most_frequent_date(
    exams: list[ExamRecord],
    exclude_dates: set[str] | None = None,
    filename_date: str | None = None,
) -> str | None:
    """Select the most frequent exam date across all pages using frequency-based voting."""
    from collections import Counter

    all_dates = []
    excluded_dates = exclude_dates or set()
    for exam in exams:
        if exam.transcription:
            page_dates = [
                date
                for date in extract_dates_from_text(exam.transcription)
                if date not in excluded_dates
            ]
            if page_dates:
                all_dates.append(min(page_dates))

    if not all_dates:
        all_dates = [exam.exam_date for exam in exams if exam.exam_date]

    if not all_dates:
        return None

    date_counts = Counter(all_dates)
    most_common_date, count = date_counts.most_common(1)[0]

    if len(date_counts) > 1:
        logger.info("Multi-era document detected. Date frequency: %s", dict(date_counts))
        logger.info(
            "Selected most frequent date: %s (%s/%s pages)",
            most_common_date,
            count,
            len(all_dates),
        )

    if (
        filename_date
        and most_common_date != filename_date
        and filename_date in date_counts
    ):
        logger.info(
            "Filename date override: %s (found in %s pages) overrides "
            "frequency winner %s (%s pages)",
            filename_date,
            date_counts[filename_date],
            most_common_date,
            count,
        )
        return filename_date

    return most_common_date


def process_single_pdf(
    pdf_path: Path,
    output_path: Path,
    config: ExtractionConfig,
    client: OpenAI,
    profile_context: str = "",
    birth_date: str | None = None,
    force_regenerate_images: bool = False,
) -> int | None | str:
    """Process a single PDF and return page count, None on failure, or 'skipped'."""
    doc_stem = pdf_path.stem
    logger.info("Processing: %s", pdf_path.name)
    doc_output_dir = output_path / doc_stem
    working_pdf_path = pdf_path
    existing_output_pdf = doc_output_dir / pdf_path.name
    if existing_output_pdf.exists():
        working_pdf_path = existing_output_pdf

    if config.dry_run:
        try:
            page_count = len(convert_pdf_to_images(working_pdf_path))
        except Exception as exc:
            logger.error("Failed to count pages in PDF: %s: %s", pdf_path.name, exc)
            return None
        logger.info("[DRY RUN] Would process %s pages: %s", page_count, pdf_path.name)
        return page_count

    existing_images = (
        sorted(doc_output_dir.glob(f"{doc_stem}.*.jpg"))
        if doc_output_dir.exists()
        else []
    )

    if force_regenerate_images and existing_images:
        logger.info("Force regenerating %s images", len(existing_images))
        for img_path in existing_images:
            img_path.unlink()
        existing_images = []

    if existing_images:
        try:
            expected_page_count = count_pdf_pages(working_pdf_path)
        except Exception as exc:
            logger.warning(
                "Could not validate existing images for %s: %s",
                pdf_path.name,
                exc,
            )
        else:
            if not reusable_existing_images(
                existing_images,
                doc_stem,
                expected_page_count,
            ):
                found_pages = _existing_image_page_numbers(existing_images, doc_stem)
                logger.info(
                    "Discarding %s stale existing images for %s "
                    "(found pages %s, expected pages 1-%s)",
                    len(existing_images),
                    pdf_path.name,
                    found_pages,
                    expected_page_count,
                )
                for img_path in existing_images:
                    img_path.unlink()
                existing_images = []

    temp_dir = None
    try:
        if existing_images:
            logger.info(
                "Reusing %s existing images from output directory", len(existing_images)
            )
            temp_image_paths = existing_images
        else:
            temp_dir, temp_image_paths = preprocess_pdf_images_to_temp(
                working_pdf_path, doc_stem
            )

        logger.debug(
            "Classifying document: %s (%s pages)",
            pdf_path.name,
            len(temp_image_paths),
        )
        try:
            classification = classify_document(
                temp_image_paths,
                config.extract_model_id,
                client,
                profile_context=profile_context,
            )
        except (APIError, RuntimeError, ValueError, TypeError) as exc:
            logger.error("Classification failed for %s: %s", pdf_path.name, exc)
            return None

        if not classification.is_exam:
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            purge_derived_outputs(
                doc_output_dir,
                doc_stem,
                remove_images=True,
            )
            copy_source_pdf(pdf_path, doc_output_dir)
            write_skip_marker(
                doc_output_dir,
                pdf_path.name,
                getattr(classification, "reason", None)
                or "classified as non-medical document",
            )
            logger.info("Skipped (not a medical exam): %s", pdf_path.name)
            return "skipped"

        doc_output_dir.mkdir(parents=True, exist_ok=True)
        purge_derived_outputs(
            doc_output_dir,
            doc_stem,
            remove_images=force_regenerate_images,
        )
        remove_skip_marker(doc_output_dir)

        copy_source_pdf(pdf_path, doc_output_dir)
        image_paths = existing_images or persist_temp_images(temp_image_paths, doc_output_dir)

        exam_name = classification.exam_name_raw or doc_stem
        exam_date = classification.exam_date or extract_date_from_filename(pdf_path.name)
        facility_name = classification.facility_name

        def process_page(page_num: int, image_path: Path) -> ExamRecord:
            embedded_text = extract_pdf_page_text(working_pdf_path, page_num)
            page_kind, chart_type, source_mode = determine_page_strategy(
                embedded_text,
                document_exam_name=exam_name,
            )
            confidence = None
            prompt_variant_used = "transcription_system"
            retry_attempts = 1
            transcription = ""
            page_exam_name = exam_name

            if chart_type == "sleep_summary_graph":
                transcription = build_non_discrete_chart_marker(
                    chart_type,
                    _extract_visible_chart_labels(embedded_text),
                )
                prompt_variant_used = "chart_marker"
            elif chart_type == "eeg_signal_trace":
                original_text, prompt_variant_used, retry_attempts = transcribe_with_retry(
                    image_path=image_path,
                    model_id=config.extract_model_id,
                    client=client,
                    validation_model_id=config.validation_model_id,
                    temperature=0.1,
                    profile_context=profile_context,
                    max_retries=1,
                    page_kind="text",
                )
                signal_labels = _extract_visible_signal_labels(
                    original_text,
                    document_date=exam_date,
                )
                if signal_labels:
                    transcription = build_non_discrete_chart_marker(
                        chart_type,
                        signal_labels,
                    )
                    prompt_variant_used = "eeg_signal_trace_marker"
                else:
                    rotated_text, rotated_variant, rotated_attempts = _transcribe_rotated_image(
                        image_path=image_path,
                        model_id=config.extract_model_id,
                        client=client,
                        validation_model_id=config.validation_model_id,
                        profile_context=profile_context,
                    )
                    retry_attempts += rotated_attempts
                    signal_labels = _extract_visible_signal_labels(
                        rotated_text,
                        document_date=exam_date,
                    )
                    if signal_labels:
                        transcription = build_non_discrete_chart_marker(
                            chart_type,
                            signal_labels,
                        )
                        prompt_variant_used = f"{rotated_variant}_rotated_marker"
                    else:
                        page_kind = "image_only"
                        chart_type = None
                        transcription = build_no_readable_text_marker()
                        prompt_variant_used = f"{rotated_variant}_rotated_no_text"
            elif chart_type == "audiogram":
                transcription, prompt_variant_used, retry_attempts = transcribe_with_retry(
                    image_path=image_path,
                    model_id=config.extract_model_id,
                    client=client,
                    validation_model_id=config.validation_model_id,
                    temperature=0.1,
                    profile_context=profile_context,
                    max_retries=1,
                    page_kind="chart",
                    chart_type=chart_type,
                    prompt_variants=[
                        "chart_transcription_system",
                        "chart_transcription_system_alt1",
                    ],
                    user_prompt_text=build_chart_user_prompt(chart_type, embedded_text),
                )
                page_exam_name = "Audiograma Vocal"
            elif chart_type == "tympanometry":
                transcription, prompt_variant_used, retry_attempts = transcribe_with_retry(
                    image_path=image_path,
                    model_id=config.extract_model_id,
                    client=client,
                    validation_model_id=config.validation_model_id,
                    temperature=0.1,
                    profile_context=profile_context,
                    max_retries=1,
                    page_kind="chart",
                    chart_type=chart_type,
                    prompt_variants=[
                        "chart_transcription_system",
                        "chart_transcription_system_alt1",
                    ],
                    user_prompt_text=build_chart_user_prompt(chart_type, embedded_text),
                )
                page_exam_name = "Timpanometria"
            elif source_mode == "embedded_text":
                transcription = embedded_text
                prompt_variant_used = "pdftotext_layout"
            elif config.n_extractions > 1:
                transcription, all_transcriptions = self_consistency(
                    transcribe_page,
                    config.self_consistency_model_id,
                    config.n_extractions,
                    image_path,
                    config.extract_model_id,
                    client,
                    client=client,
                    profile_context=profile_context,
                )
                confidence = score_transcription_confidence(
                    transcription,
                    all_transcriptions,
                    config.self_consistency_model_id,
                    client,
                )
                if confidence < 1.0:
                    logger.info(
                        "Self-consistency confidence: %.2f for %s",
                        confidence,
                        image_path.name,
                    )
            else:
                transcription, prompt_variant_used, retry_attempts = transcribe_with_retry(
                    image_path=image_path,
                    model_id=config.extract_model_id,
                    client=client,
                    validation_model_id=config.validation_model_id,
                    temperature=0.1,
                    profile_context=profile_context,
                    max_retries=3,
                    page_kind="text",
                )

            issues = validate_page_output(
                transcription,
                page_kind=page_kind,
                chart_type=chart_type,
                page=page_num,
            )
            blocking_issue = first_blocking_issue(issues)

            if blocking_issue and _should_mark_unsupported_visual(
                embedded_text,
                transcription,
                blocking_kind=blocking_issue.kind,
                chart_type=chart_type,
            ):
                page_kind = "image_only"
                chart_type = None
                source_mode = "vision"
                transcription = build_no_readable_text_marker()
                issues = []

            validation_meta = derive_validation_metadata(issues, page_kind, chart_type)

            if not transcription:
                logger.warning("Empty transcription for %s", image_path.name)

            if retry_attempts > 1:
                logger.info(
                    "Required %s prompt attempts for %s, final variant: %s",
                    retry_attempts,
                    image_path.name,
                    prompt_variant_used,
                )

            return ExamRecord(
                exam_name_raw=page_exam_name,
                exam_date=exam_date,
                transcription=transcription,
                page_number=page_num,
                source_file=pdf_path.name,
                transcription_confidence=confidence,
                prompt_variant=prompt_variant_used,
                retry_attempts=retry_attempts,
                physician_name=classification.physician_name,
                department=classification.department,
                facility_name=facility_name,
                page_kind=page_kind,
                source_mode=source_mode,
                chart_type=chart_type,
                **validation_meta,
            )

        all_exams: list[ExamRecord] = []
        page_errors: list[int] = []
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(process_page, page_num, image_path): page_num
                for page_num, image_path in enumerate(image_paths, start=1)
            }
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    exam = future.result()
                    all_exams.append(exam)
                except Exception as exc:
                    logger.error("Page %s processing failed: %s", page_num, exc)
                    page_errors.append(page_num)

        if page_errors:
            return None

        all_exams.sort(key=lambda exam: exam.page_number)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if all_exams:
        exclude_dates = {birth_date} if birth_date else None
        filename_date = extract_date_from_filename(pdf_path.name)
        corrected_date = select_most_frequent_date(
            all_exams,
            exclude_dates=exclude_dates,
            filename_date=filename_date,
        )
        if corrected_date:
            logger.debug("Selected document date by frequency: %s", corrected_date)
            for exam in all_exams:
                exam.exam_date = corrected_date

    if all_exams:
        blocking_pages = [
            exam
            for exam in all_exams
            if exam.validation_status in {"retryable_failure", "failed"}
        ]
        if blocking_pages:
            for exam in blocking_pages:
                logger.error(
                    "Blocking page validation failure in %s page %s: %s",
                    pdf_path.name,
                    exam.page_number,
                    exam.failure_type,
                )
            return None

        deterministic_mappings = {
            "audiogram": ("other", "Speech Audiogram"),
            "tympanometry": ("other", "Tympanometry"),
        }
        raw_names = list(
            {
                exam.exam_name_raw
                for exam in all_exams
                if exam.chart_type not in deterministic_mappings
            }
        )
        try:
            standardized = standardize_exam_types(raw_names, config.extract_model_id, client)
        except (APIError, RuntimeError, ValueError, TypeError) as exc:
            logger.error("Standardization failed for %s: %s", pdf_path.name, exc)
            return None

        for exam in all_exams:
            if exam.chart_type in deterministic_mappings:
                exam.exam_type, exam.exam_name_standardized = deterministic_mappings[
                    exam.chart_type
                ]
                continue

            if exam.exam_name_raw in standardized:
                exam.exam_type, exam.exam_name_standardized = standardized[
                    exam.exam_name_raw
                ]

        for exam in all_exams:
            save_transcription_file(
                [exam],
                doc_output_dir,
                doc_stem,
                exam.page_number,
            )

        try:
            document_summary = summarize_document(
                all_exams,
                config.summarize_model_id,
                client,
                max_input_tokens=config.summarize_max_input_tokens,
            )
        except (APIError, RuntimeError, ValueError, TypeError) as exc:
            logger.error("Summarization failed for %s: %s", pdf_path.name, exc)
            return None
        summary_issues = validate_summary_output(document_summary) if document_summary else []
        if first_blocking_issue(summary_issues):
            logger.error("Blocking summary validation failure in %s", pdf_path.name)
            return None
        save_document_summary(document_summary, doc_output_dir, doc_stem, all_exams)

    logger.info("Processed %s pages for: %s", len(all_exams), pdf_path.name)
    return len(all_exams)


def log_output_assertions_report(
    pdf_files: list[Path],
    output_path: Path,
    input_path: Path,
    label: str = "Post-run validation",
) -> bool:
    """Log grouped post-run assertions and return whether validation passed."""
    grouped_issues = collect_output_assertions(pdf_files, output_path, input_path)
    if not grouped_issues:
        logger.info("%s passed with no issues", label)
        return True

    total_issues = sum(len(items) for items in grouped_issues.values())
    logger.error("%s found %s issue(s):", label, total_issues)
    for category, issues in grouped_issues.items():
        logger.error("%s (%s):", category.capitalize(), len(issues))
        for issue in issues:
            logger.error("  %s", issue)
    return False


def run_profile(profile_name: str, args: Namespace) -> bool:
    """Run the pipeline for a single profile."""
    profile_path = ProfileConfig.find_profile(profile_name)
    if not profile_path:
        print(f"Error: Profile '{profile_name}' not found in {ProfileConfig.config_dir()}")
        return False

    profile = ProfileConfig.from_file(profile_path)

    try:
        config = ExtractionConfig.from_profile(profile)
    except ValueError as exc:
        print(f"Error: {exc}")
        return False

    if not config.input_path.exists():
        print(f"Error: Input path does not exist: {config.input_path}")
        return False

    config.output_path.mkdir(parents=True, exist_ok=True)

    if args.model:
        config.extract_model_id = config.self_consistency_model_id = args.model
        config.summarize_model_id = args.model
    if args.workers:
        config.max_workers = args.workers
    if args.pattern:
        config.input_file_regex = args.pattern
    config.dry_run = resolve_run_mode(args) == RunMode.DRY_RUN

    log_dir = config.output_path / "logs"
    setup_logging(log_dir, clear_logs=True)

    if config.dry_run:
        logger.info("=" * 60)
        logger.info("Medical Exams Parser - DRY RUN MODE")
        logger.info("=" * 60)
        logger.info("No LLM calls will be made. No files will be written.")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("Medical Exams Parser - Starting Pipeline")
        logger.info("=" * 60)
    logger.info("Profile: %s", profile.name)
    logger.info("Profile file: %s", profile.source_path)
    logger.info("Input path: %s", config.input_path)
    logger.info("Output path: %s", config.output_path)
    logger.info("Extract model: %s", config.extract_model_id)
    logger.info("Summarize model: %s", config.summarize_model_id)
    logger.info("N extractions: %s", config.n_extractions)
    logger.info("API base URL: %s", config.openrouter_base_url)

    client = OpenAI(
        base_url=config.openrouter_base_url,
        api_key=config.openrouter_api_key,
        http_client=httpx.Client(),
    )

    profile_context = build_profile_context(profile)
    if profile_context:
        logger.info("Profile context injected into extraction prompt")

    mode = resolve_run_mode(args)
    if mode in {RunMode.REGENERATE, RunMode.RESUMMARIZE}:
        doc_filter = args.document if mode == RunMode.RESUMMARIZE else None
        mode_label = "Resummarize" if mode == RunMode.RESUMMARIZE else "Regeneration"
        logger.info("%s mode%s", mode_label, f": {doc_filter}" if doc_filter else "")
        total_exams = regenerate_summaries(
            config.output_path,
            config,
            client,
            config.input_path,
            doc_filter=doc_filter,
        )
        logger.info("%s complete: %s exams", mode_label, total_exams)
        try:
            validation_pdfs = discover_pdf_files(
                config.input_path,
                config.input_file_regex,
                document=doc_filter,
            )
        except PermissionError as exc:
            logger.error(str(exc))
            return False
        return log_output_assertions_report(
            validation_pdfs,
            config.output_path,
            config.input_path,
            label=f"{mode_label} validation",
        )

    try:
        pdf_files = discover_pdf_files(
            config.input_path,
            config.input_file_regex,
            document=args.document,
        )
    except PermissionError as exc:
        logger.error(str(exc))
        return False
    logger.info("Found %s PDF files matching pattern", len(pdf_files))

    if not pdf_files:
        logger.warning("No PDF files found. Check INPUT_PATH and INPUT_FILE_REGEX.")
        return True

    if mode == RunMode.AUDIT:
        try:
            target_pdfs = (
                match_requested_document(pdf_files, args.document)
                if args.document
                else pdf_files
            )
        except ValueError as exc:
            logger.error(str(exc))
            return False

        return log_output_assertions_report(
            target_pdfs,
            config.output_path,
            config.input_path,
            label="Output audit",
        )

    try:
        to_process, already_processed = select_documents_to_process(
            pdf_files,
            config.output_path,
            document=args.document,
            reprocess_all=args.reprocess_all,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return False

    if args.document:
        logger.info("Force reprocessing: %s", to_process[0].name)
    elif args.reprocess_all:
        logger.info("Force reprocessing all %s documents", len(to_process))
    else:
        logger.info(
            "Processing %s new PDFs, %s already processed",
            len(to_process),
            already_processed,
        )

    total_pages = 0
    skipped_documents: list[str] = []
    processed_documents: list[Path] = []
    failed_count = 0

    for pdf_path in tqdm(to_process, desc="Processing PDFs"):
        try:
            result = process_single_pdf(
                pdf_path,
                config.output_path,
                config,
                client,
                profile_context=profile_context,
                birth_date=profile.birth_date,
                force_regenerate_images=bool(args.document),
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_path.name, exc)
            failed_count += 1
            continue

        if result == "skipped":
            skipped_documents.append(pdf_path.name)
        elif isinstance(result, int):
            total_pages += result
            processed_documents.append(pdf_path)
        else:
            failed_count += 1

    if config.dry_run:
        logger.info(
            "DRY RUN COMPLETE - Would process: %s documents (%s pages)",
            len(processed_documents),
            total_pages,
        )
        logger.info("Would skip (already processed): %s documents", already_processed)
        logger.info(
            "Would generate: %s .md files, %s summaries",
            total_pages,
            len(processed_documents),
        )
        if skipped_documents:
            logger.info(
                "Would classify as non-exam: %s documents", len(skipped_documents)
            )
        if failed_count > 0:
            logger.warning("Would fail: %s documents", failed_count)
        return True

    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(
        "Processed: %s documents (%s pages)",
        len(processed_documents),
        total_pages,
    )
    logger.info("Skipped (not medical exams): %s", len(skipped_documents))
    if failed_count > 0:
        logger.warning("Failed: %s", failed_count)

    if skipped_documents:
        logger.info("Skipped Documents (review for false negatives):")
        for doc_name in skipped_documents:
            logger.info("  - %s", doc_name)

    return log_output_assertions_report(
        pdf_files,
        config.output_path,
        config.input_path,
        label="Post-extraction validation",
    )
