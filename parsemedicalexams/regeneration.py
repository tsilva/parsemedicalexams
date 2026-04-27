"""Summary regeneration orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .document_io import (
    copy_source_pdf,
    frontmatter_to_exam,
    parse_frontmatter,
    save_document_summary,
    transcription_files,
)
from .models import ExamRecord
from .summarization import summarize_document
from .validation import first_blocking_issue, validate_summary_output

if TYPE_CHECKING:
    from openai import OpenAI

    from .config import ExtractionConfig

logger = logging.getLogger(__name__)


def regenerate_summaries(
    output_path: Path,
    config: "ExtractionConfig",
    client: "OpenAI",
    input_path: Path | None = None,
    doc_filter: str | None = None,
) -> int:
    """Regenerate document-level summary files from existing transcription markdown files."""
    doc_dirs = [
        doc_dir
        for doc_dir in output_path.iterdir()
        if doc_dir.is_dir() and doc_dir.name != "logs"
    ]

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
                "Skipping %s: page count mismatch - %s images but %s transcriptions",
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

        all_exams: list[ExamRecord] = []
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
                frontmatter_to_exam(
                    frontmatter,
                    transcription,
                    page_num,
                    f"{doc_stem}.pdf",
                )
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
