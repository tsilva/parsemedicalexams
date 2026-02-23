"""Main pipeline for medical exam extraction and summarization."""

import argparse
import re
import shutil
import sys
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from pdf2image import convert_from_path
from openai import OpenAI
from tqdm import tqdm

from config import ExtractionConfig, ProfileConfig
from extraction import (
    transcribe_with_retry,
    self_consistency,
    classify_document,
    transcribe_page,
    score_transcription_confidence,
    DocumentClassification,
)
from standardization import standardize_exam_types
from summarization import summarize_document
from utils import preprocess_page_image, setup_logging, load_dotenv_with_env, extract_dates_from_text

logger = logging.getLogger(__name__)


# exam dict key -> YAML frontmatter key
_FRONTMATTER_MAP = {
    "exam_date": "exam_date",
    "exam_name_raw": "exam_name_raw",
    "exam_name_standardized": "title",
    "exam_type": "category",
    "physician_name": "doctor",
    "facility_name": "facility",
    "department": "department",
}


def _transcription_files(doc_dir: Path, doc_stem: str) -> list[Path]:
    """Return all transcription .md files (excluding .summary.md) in doc_dir."""
    return [f for f in doc_dir.glob(f"{doc_stem}.*.md") if not f.name.endswith(".summary.md")]


def is_document_processed(pdf_path: Path, output_path: Path) -> bool:
    """Check if a PDF has already been processed by looking for transcription .md files."""
    doc_stem = pdf_path.stem
    doc_output_dir = output_path / doc_stem
    if not doc_output_dir.exists():
        return False
    return len(_transcription_files(doc_output_dir, doc_stem)) > 0


def build_exam_frontmatter(exam: dict, extra_fields: dict = None) -> dict:
    """Build YAML frontmatter dict from an exam dict."""
    fm = {fm_key: exam[exam_key] for exam_key, fm_key in _FRONTMATTER_MAP.items() if exam.get(exam_key)}
    if extra_fields:
        fm.update(extra_fields)
    return fm


def write_markdown_with_frontmatter(path: Path, frontmatter: dict, body: str) -> None:
    """Write a markdown file with YAML frontmatter."""
    with open(path, "w", encoding="utf-8") as f:
        if frontmatter:
            f.write("---\n")
            f.write(
                yaml.dump(
                    frontmatter,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            )
            f.write("---\n\n")
        f.write(body)


def save_transcription_file(
    exams: list[dict], doc_output_dir: Path, doc_stem: str, page_num: int
) -> None:
    """
    Save page transcription as markdown file with YAML frontmatter.
    - .md = YAML frontmatter + raw transcription verbatim

    Frontmatter includes all metadata (no separate JSON file needed).
    """
    md_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.md"

    frontmatter = {}
    if exams:
        exam = exams[0]
        extra = {k: exam[ek] for k, ek in [("page", "page_number"), ("source", "source_file"), ("prompt_variant", "prompt_variant")] if exam.get(ek)}
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
    summary: str, doc_output_dir: Path, doc_stem: str, exams: list[dict] = None
) -> None:
    """
    Save document-level summary as markdown file with YAML frontmatter.
    - .summary.md = YAML frontmatter + comprehensive clinical summary for the entire document
    """
    if summary:
        summary_path = doc_output_dir / f"{doc_stem}.summary.md"
        frontmatter = build_exam_frontmatter(exams[0]) if exams else {}
        write_markdown_with_frontmatter(summary_path, frontmatter, summary.strip() + "\n")


def extract_date_from_filename(filename: str) -> str | None:
    """Try to extract date from filename in YYYY-MM-DD format."""
    # Try YYYY-MM-DD pattern
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)

    # Try YYYY_MM_DD pattern
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # Try YYYYMMDD pattern
    match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return None


def select_most_frequent_date(
    exams: list[dict],
    exclude_dates: set[str] | None = None,
    filename_date: str | None = None,
) -> str | None:
    """Select the most frequent exam date across all pages using frequency-based voting.
    Handles multi-era documents; prefers filename_date as tiebreaker for DD/MM vs MM/DD ambiguity."""
    from collections import Counter

    # Extract dates from each page's transcription
    all_dates = []
    _exclude = exclude_dates or set()
    for exam in exams:
        transcription = exam.get("transcription", "")
        if transcription:
            page_dates = extract_dates_from_text(transcription)
            # Filter out excluded dates (e.g. patient birth date)
            page_dates = [d for d in page_dates if d not in _exclude]
            # Use the earliest date on each page as that page's representative date
            if page_dates:
                page_date = min(page_dates)  # Earliest = likely the exam date
                all_dates.append(page_date)

    # Fallback: use exam_date if no dates found in transcriptions
    if not all_dates:
        all_dates = [exam.get("exam_date") for exam in exams if exam.get("exam_date")]

    if not all_dates:
        return None

    # Count frequency of each date
    date_counts = Counter(all_dates)

    # Return most common date (mode)
    most_common_date, count = date_counts.most_common(1)[0]

    # Log if there's date variation (indicating multi-era document)
    if len(date_counts) > 1:
        logger.info(f"Multi-era document detected. Date frequency: {dict(date_counts)}")
        logger.info(
            f"Selected most frequent date: {most_common_date} ({count}/{len(all_dates)} pages)"
        )

    # If the most common date conflicts with the filename date, and the filename date
    # appears in at least one page, prefer the filename date. This handles DD/MM vs MM/DD
    # ambiguity where software timestamps use MM/DD but the actual exam date uses DD/MM.
    if (
        filename_date
        and most_common_date != filename_date
        and filename_date in date_counts
    ):
        logger.info(
            f"Filename date override: {filename_date} (found in {date_counts[filename_date]} pages) "
            f"overrides frequency winner {most_common_date} ({count} pages)"
        )
        return filename_date

    return most_common_date


def process_single_pdf(
    pdf_path: Path,
    output_path: Path,
    config: ExtractionConfig,
    client: OpenAI,
    page_filter: int | None = None,
    profile_context: str = "",
    birth_date: str | None = None,
    force_regenerate_images: bool = False,
) -> int | None | str:
    """Process a single PDF: classify, then transcribe all pages verbatim.
    Returns page count, None on failure, or "skipped" if not a medical exam."""
    doc_stem = pdf_path.stem
    logger.info(f"Processing: {pdf_path.name}")

    # DRY RUN: Skip all processing, just count pages
    if config.dry_run:
        try:
            pages = convert_from_path(str(pdf_path))
            page_count = len(pages)
        except Exception as e:
            logger.error(f"Failed to count pages in PDF: {pdf_path.name}: {e}")
            return None
        logger.info(f"[DRY RUN] Would process {page_count} pages: {pdf_path.name}")
        return page_count

    # Check if we can reuse existing images from output directory
    doc_output_dir = output_path / doc_stem
    existing_images = (
        sorted(doc_output_dir.glob(f"{doc_stem}.*.jpg"))
        if doc_output_dir.exists()
        else []
    )

    # Delete existing images if force regeneration requested
    if force_regenerate_images and existing_images:
        logger.info(f"Force regenerating {len(existing_images)} images")
        for img_path in existing_images:
            img_path.unlink()
        existing_images = []

    # Convert PDF to images (or reuse existing ones)
    if existing_images:
        logger.info(
            f"Reusing {len(existing_images)} existing images from output directory"
        )
        pages = None  # Not needed
    else:
        try:
            pages = convert_from_path(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {pdf_path.name}: {e}")
            return None

    # Create temp directory for classification images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_image_paths = []

        if existing_images:
            # Use existing images directly
            temp_image_paths = existing_images
        else:
            # Preprocess and save images to temp directory
            for page_num, page_image in enumerate(pages, start=1):
                processed_image = preprocess_page_image(page_image)
                temp_image_path = temp_path / f"{doc_stem}.{page_num:03d}.jpg"
                processed_image.save(str(temp_image_path), "JPEG", quality=80)
                temp_image_paths.append(temp_image_path)

        # PHASE 1: Classify document (skip when reprocessing a specific page)
        if page_filter is not None:
            # When reprocessing a specific page, skip classification
            logger.debug(
                f"Skipping classification for page-specific reprocessing: {pdf_path.name}"
            )
            classification = DocumentClassification(is_exam=True)
        else:
            logger.debug(
                f"Classifying document: {pdf_path.name} ({len(temp_image_paths)} pages)"
            )
            try:
                classification = classify_document(
                    temp_image_paths,
                    config.extract_model_id,
                    client,
                    profile_context=profile_context,
                )
            except Exception as e:
                logger.error(f"Classification failed for {pdf_path.name}: {e}")
                # Default to treating as exam to avoid missing content
                classification = DocumentClassification(is_exam=True)

            # If not an exam, skip this document
            if not classification.is_exam:
                logger.info(f"Skipped (not a medical exam): {pdf_path.name}")
                return "skipped"

        # PHASE 2: Document is an exam - create output directory and transcribe all pages
        doc_output_dir = output_path / doc_stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy source PDF to output directory
        dest_pdf = doc_output_dir / pdf_path.name
        if not dest_pdf.exists():
            try:
                shutil.copy2(pdf_path, dest_pdf)
            except PermissionError:
                logger.warning(
                    f"Could not copy PDF to output (permission denied): {pdf_path.name}"
                )

        # Move images from temp to output directory (skip if reusing existing)
        if existing_images:
            image_paths = existing_images
        else:
            image_paths = []
            for temp_image_path in temp_image_paths:
                final_image_path = doc_output_dir / temp_image_path.name
                shutil.copy2(temp_image_path, final_image_path)
                image_paths.append(final_image_path)

        # Get document-level metadata from classification
        exam_name = classification.exam_name_raw or doc_stem
        exam_date = classification.exam_date
        facility_name = classification.facility_name

        # Try to extract date from filename if not found in classification
        if not exam_date:
            exam_date = extract_date_from_filename(pdf_path.name)

        # Define page processing function for parallel execution
        def process_page(page_num: int, image_path: Path) -> dict | None:
            """Process a single page - returns exam dict or None if skipped."""
            # Skip pages if filter is active
            if page_filter is not None and page_num != page_filter:
                return None

            # Transcribe page with automatic retry on refusal using different prompt variants
            confidence = None
            prompt_variant_used = "transcription_system"
            retry_attempts = 1
            transcription = ""

            try:
                if config.n_extractions > 1:
                    # Self-consistency mode: use transcribe_page directly with retry wrapper
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
                    # Use LLM to assess semantic agreement for confidence
                    confidence = score_transcription_confidence(
                        transcription,
                        all_transcriptions,
                        config.self_consistency_model_id,
                        client,
                    )
                    if confidence < 1.0:
                        logger.info(
                            f"Self-consistency confidence: {confidence:.2f} for {image_path.name}"
                        )
                else:
                    # Single extraction with prompt variant retry on refusal
                    transcription, prompt_variant_used, retry_attempts = (
                        transcribe_with_retry(
                            image_path=image_path,
                            model_id=config.extract_model_id,
                            client=client,
                            validation_model_id=config.validation_model_id,
                            temperature=0.1,
                            profile_context=profile_context,
                            max_retries=3,
                        )
                    )
            except Exception as e:
                logger.error(f"Transcription failed for {image_path.name}: {e}")
                transcription = ""

            if not transcription:
                logger.warning(f"Empty transcription for {image_path.name}")

            # Log if alternative prompt was needed
            if retry_attempts > 1:
                logger.info(
                    f"Required {retry_attempts} prompt attempts for {image_path.name}, "
                    f"final variant: {prompt_variant_used}"
                )

            # Create exam entry for this page
            return {
                "exam_name_raw": exam_name,
                "exam_date": exam_date,
                "transcription": transcription,
                "page_number": page_num,
                "source_file": pdf_path.name,
                "transcription_confidence": confidence,
                "prompt_variant": prompt_variant_used,
                "retry_attempts": retry_attempts,
                "physician_name": classification.physician_name,
                "department": classification.department,
                "facility_name": facility_name,
            }

        # Process pages in parallel
        all_exams = []
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(process_page, page_num, image_path): page_num
                for page_num, image_path in enumerate(image_paths, start=1)
            }
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    exam = future.result()
                    if exam is not None:
                        all_exams.append(exam)
                except Exception as e:
                    logger.error(f"Page {page_num} processing failed: {e}")

        # Sort by page number for consistent ordering
        all_exams.sort(key=lambda x: x["page_number"])

    # Apply frequency-based date correction for multi-era documents
    if all_exams:
        exclude_dates = {birth_date} if birth_date else None
        filename_date = extract_date_from_filename(pdf_path.name)
        corrected_date = select_most_frequent_date(
            all_exams, exclude_dates=exclude_dates, filename_date=filename_date
        )
        if corrected_date:
            logger.debug(f"Selected document date by frequency: {corrected_date}")
            # Update all exams with the corrected date
            for exam in all_exams:
                exam["exam_date"] = corrected_date

    # Standardize exam types
    if all_exams:
        raw_names = list(set(exam.get("exam_name_raw", "") for exam in all_exams))
        standardized = standardize_exam_types(
            raw_names, config.extract_model_id, client
        )

        for exam in all_exams:
            raw_name = exam.get("exam_name_raw", "")
            if raw_name in standardized:
                exam_type, std_name = standardized[raw_name]
                exam["exam_type"] = exam_type
                exam["exam_name_standardized"] = std_name

        # Resave transcription files with YAML frontmatter (now includes standardized info)
        for exam in all_exams:
            page_num = exam.get("page_number", 1)
            save_transcription_file([exam], doc_output_dir, doc_stem, page_num)

        # Generate document-level summary
        document_summary = summarize_document(
            all_exams,
            config.summarize_model_id,
            client,
            max_input_tokens=config.summarize_max_input_tokens,
        )
        save_document_summary(document_summary, doc_output_dir, doc_stem, all_exams)

    logger.info(f"Processed {len(all_exams)} pages for: {pdf_path.name}")
    return len(all_exams)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown content.

    Returns:
        Tuple of (frontmatter_dict, transcription_content)
    """
    frontmatter = {}
    transcription = content.strip()

    if transcription.startswith("---"):
        # Find the end of frontmatter
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
    frontmatter: dict, transcription: str, page_num: int, source_file: str = None
) -> dict:
    """Convert frontmatter fields to internal exam dict format (inverse of _FRONTMATTER_MAP)."""
    _inv = {fm: ex for ex, fm in _FRONTMATTER_MAP.items()}
    result = {exam_key: frontmatter.get(fm_key) for fm_key, exam_key in _inv.items()}
    result["transcription_confidence"] = frontmatter.get("confidence")
    result["transcription"] = transcription
    result["page_number"] = frontmatter.get("page") or page_num
    result["source_file"] = frontmatter.get("source") or source_file
    return result


def regenerate_summaries(
    output_path: Path,
    config: ExtractionConfig,
    client: OpenAI,
    input_path: Path | None = None,
    doc_filter: str | None = None,
):
    """
    Regenerate document-level summary files from existing transcription (.md) files.

    Reads metadata from YAML frontmatter and transcription content from .md files,
    re-runs document-level summarization, and saves updated .summary.md files.
    If input_path is provided, also copies source PDFs to output directories if missing.
    """
    # Find all document directories
    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name != "logs"]

    # Apply document filter if specified
    if doc_filter:
        query = doc_filter.lower()
        query_stem = query[:-4] if query.endswith(".pdf") else query
        doc_dirs = [
            d
            for d in doc_dirs
            if d.name.lower() == query_stem or d.name.lower() == query
        ]
        if not doc_dirs:
            logger.error(f"No matching document directory found for: {doc_filter}")
            return 0

    logger.info(f"Found {len(doc_dirs)} document directories to regenerate")

    total_exams = 0
    for doc_dir in tqdm(doc_dirs, desc="Regenerating summaries"):
        doc_stem = doc_dir.name

        # Page completeness check: compare .jpg count to .md transcription count
        jpg_files = list(doc_dir.glob(f"{doc_stem}.*.jpg"))
        md_transcription_files = _transcription_files(doc_dir, doc_stem)
        if jpg_files and len(jpg_files) != len(md_transcription_files):
            logger.error(
                f"Skipping {doc_stem}: page count mismatch — "
                f"{len(jpg_files)} images but {len(md_transcription_files)} transcriptions"
            )
            continue

        # Copy source PDF if missing and input_path is provided
        if input_path:
            existing_pdfs = list(doc_dir.glob("*.pdf"))
            if not existing_pdfs:
                # Try to find source PDF in input directory
                source_pdfs = list(input_path.glob(f"**/{doc_stem}.pdf"))
                if source_pdfs:
                    try:
                        shutil.copy2(source_pdfs[0], doc_dir / source_pdfs[0].name)
                        logger.info(f"Copied source PDF: {source_pdfs[0].name}")
                    except PermissionError:
                        logger.warning(
                            f"Could not copy PDF (permission denied): {source_pdfs[0].name}"
                        )

        md_files = sorted(_transcription_files(doc_dir, doc_stem))
        if not md_files:
            logger.warning(f"No transcription files found in {doc_dir}")
            continue

        all_exams = []

        for md_path in md_files:
            # Extract page number from filename (e.g., "doc.001.md" -> 1)
            parts = md_path.stem.split(".")
            if len(parts) >= 2:
                try:
                    page_num = int(parts[-1])
                except ValueError:
                    continue
            else:
                continue

            # Read and parse the markdown file
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            frontmatter, transcription = parse_frontmatter(content)

            # Convert frontmatter to exam dict (use doc_stem.pdf as fallback source)
            exam = frontmatter_to_exam(
                frontmatter, transcription, page_num, f"{doc_stem}.pdf"
            )
            all_exams.append(exam)

        if not all_exams:
            logger.warning(f"No exams found in {doc_dir}")
            continue

        # Delete existing .summary.md files
        for old_summary in doc_dir.glob("*.summary.md"):
            old_summary.unlink()

        # Generate document-level summary
        document_summary = summarize_document(
            all_exams,
            config.summarize_model_id,
            client,
            max_input_tokens=config.summarize_max_input_tokens,
        )

        # Save one summary for the entire document
        save_document_summary(document_summary, doc_dir, doc_stem, all_exams)

        logger.info(f"Regenerated summary for {len(all_exams)} exams: {doc_stem}")
        total_exams += len(all_exams)

    return total_exams


def validate_pipeline_outputs(pdf_files: list[Path], output_path: Path) -> list[str]:
    """Validate expected output files exist for each processed PDF using jpg/md counts."""
    missing = []
    for pdf_path in pdf_files:
        doc_stem = pdf_path.stem
        doc_output_dir = output_path / doc_stem
        if not doc_output_dir.exists():
            missing.append(f"Missing target folder: {doc_output_dir}")
            continue
        jpg_count = len(list(doc_output_dir.glob(f"{doc_stem}.*.jpg")))
        md_count = len(_transcription_files(doc_output_dir, doc_stem))
        if not jpg_count:
            missing.append(f"Missing page images in: {doc_output_dir}")
        elif jpg_count != md_count:
            missing.append(f"Page count mismatch: {jpg_count} images, {md_count} transcriptions in {doc_stem}")
        summary_path = doc_output_dir / f"{doc_stem}.summary.md"
        if not summary_path.exists():
            missing.append(f"Missing document summary: {summary_path}")
    return missing


def validate_frontmatter(output_path: Path) -> list[str]:
    """Validate that all .md files have YAML frontmatter with required fields."""
    issues = []
    required_fields = {"exam_date", "category", "title"}
    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name != "logs"]
    for doc_dir in doc_dirs:
        doc_stem = doc_dir.name
        for md_path in doc_dir.glob(f"{doc_stem}.*.md"):
            try:
                frontmatter, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
                if not frontmatter:
                    issues.append(f"Missing frontmatter: {md_path.name}")
                    continue
                missing_fields = required_fields - set(frontmatter.keys())
                if missing_fields:
                    issues.append(f"Missing fields {missing_fields}: {md_path.name}")
            except Exception as e:
                issues.append(f"Error reading {md_path.name}: {e}")
    return issues


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and summarize medical exam reports from PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --profile tsilva              # Process all new PDFs
  python main.py --list-profiles               # List available profiles
  python main.py -p tsilva --regenerate        # Regenerate summaries only
  python main.py -p tsilva --resummarize       # Resummarize all documents
  python main.py -p tsilva --resummarize -d exam.pdf  # Resummarize one document
  python main.py -p tsilva --reprocess-all     # Force reprocess all documents
  python main.py -p tsilva -d exam.pdf         # Reprocess specific document
  python main.py -p tsilva -d exam.pdf --page 2  # Reprocess specific page
  python main.py -p tsilva --dry-run           # Preview what would be processed
        """,
    )
    parser.add_argument(
        "--profile", "-p", type=str, help="Profile name (without extension)"
    )
    parser.add_argument(
        "--list-profiles", action="store_true", help="List available profiles and exit"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate summaries from existing transcription files",
    )
    parser.add_argument(
        "--resummarize",
        action="store_true",
        help="Regenerate summaries only (use with -d to target a specific document)",
    )
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help="Force reprocessing of all documents (ignores already processed)",
    )
    parser.add_argument(
        "--document",
        "-d",
        type=str,
        help="Process only this document (filename or stem). Forces reprocessing.",
    )
    parser.add_argument(
        "--page", type=int, help="Process only this page number (requires --document)"
    )

    # Override arguments
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model ID for extraction (overrides profile/env)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of parallel workers (overrides profile/env)",
    )
    parser.add_argument(
        "--pattern", type=str, help="Regex pattern for input files (overrides profile)"
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name to load (loads .env.{name} instead of .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the pipeline without making LLM calls or writing files. Reports what would be processed.",
    )
    return parser.parse_args()


def run_profile(profile_name: str, args) -> bool:
    """
    Run the pipeline for a single profile.

    Args:
        profile_name: Name of the profile to run
        args: Parsed command line arguments

    Returns:
        True if successful, False otherwise
    """
    # Load profile
    profile_path = None
    for ext in (".yaml", ".yml", ".json"):
        p = Path(f"profiles/{profile_name}{ext}")
        if p.exists():
            profile_path = p
            break

    if not profile_path:
        print(f"Error: Profile '{profile_name}' not found")
        return False

    profile = ProfileConfig.from_file(profile_path)

    # Validate profile has required paths
    if not profile.input_path:
        print(f"Error: Profile '{profile_name}' has no input_path defined.")
        return False
    if not profile.output_path:
        print(f"Error: Profile '{profile_name}' has no output_path defined.")
        return False

    # Validate input path exists
    if not profile.input_path.exists():
        print(f"Error: Input path does not exist: {profile.input_path}")
        return False

    # Ensure output directory exists
    profile.output_path.mkdir(parents=True, exist_ok=True)

    # Load base config from environment (API keys and model settings)
    config = ExtractionConfig.from_env()

    # Apply profile paths to config
    config.input_path = profile.input_path
    config.output_path = profile.output_path
    config.input_file_regex = (
        profile.input_file_regex or config.input_file_regex or ".*\\.pdf"
    )

    # Apply profile overrides
    if profile.model:
        config.extract_model_id = config.self_consistency_model_id = config.summarize_model_id = profile.model
    if profile.workers:
        config.max_workers = profile.workers

    # Apply CLI overrides (highest priority)
    if args.model:
        config.extract_model_id = config.self_consistency_model_id = config.summarize_model_id = args.model
    if args.workers:
        config.max_workers = args.workers
    if args.pattern:
        config.input_file_regex = args.pattern
    config.dry_run = args.dry_run

    # Setup logging
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
    logger.info(f"Profile: {profile.name}")
    logger.info(f"Input path: {config.input_path}")
    logger.info(f"Output path: {config.output_path}")
    logger.info(f"Extract model: {config.extract_model_id}")
    logger.info(f"Summarize model: {config.summarize_model_id}")
    logger.info(f"N extractions: {config.n_extractions}")
    logger.info(f"API base URL: {config.openrouter_base_url}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=config.openrouter_base_url, api_key=config.openrouter_api_key
    )

    # Build profile context for extraction prompt
    profile_context = ""
    if profile.birth_date or profile.full_name:
        parts = []
        if profile.full_name:
            parts.append(f"Patient name: {profile.full_name}")
        if profile.birth_date:
            parts.append(f"Patient date of birth: {profile.birth_date}")
            parts.append(
                f"IMPORTANT: {profile.birth_date} is the patient's birth date — NEVER use it as exam_date"
            )
        if profile.locale:
            parts.append(
                f"Locale: {profile.locale} (dates in documents typically use DD/MM/YYYY format)"
            )
        profile_context = "PATIENT CONTEXT:\n" + "\n".join(parts)
        logger.info(f"Profile context injected into extraction prompt")

    # Handle --resummarize and --regenerate modes
    if args.resummarize or args.regenerate:
        doc_filter = (args.document or None) if args.resummarize else None
        mode = "Resummarize" if args.resummarize else "Regeneration"
        logger.info(f"{mode} mode{f': {doc_filter}' if doc_filter else ''}")
        total_exams = regenerate_summaries(
            config.output_path, config, client, config.input_path, doc_filter=doc_filter
        )
        logger.info(f"{mode} complete: {total_exams} exams")
        if args.regenerate:
            frontmatter_issues = validate_frontmatter(config.output_path)
            if frontmatter_issues:
                logger.warning(f"Frontmatter issues ({len(frontmatter_issues)}):")
                for issue in frontmatter_issues:
                    logger.warning(f"  {issue}")
            else:
                logger.info("All frontmatter validated successfully")
        return True

    # Find PDF files
    pdf_pattern = re.compile(config.input_file_regex)
    pdf_files = sorted(
        [f for f in config.input_path.glob("**/*.pdf") if pdf_pattern.match(f.name)]
    )

    logger.info(f"Found {len(pdf_files)} PDF files matching pattern")

    if not pdf_files:
        logger.warning("No PDF files found. Check INPUT_PATH and INPUT_FILE_REGEX.")
        return True

    # Select documents to process
    already_processed = 0  # Initialize for dry run stats
    if args.document:
        # Find the specific document (by filename or stem, case-insensitive)
        doc_query = args.document.lower()
        # Strip .pdf extension if present for stem matching
        doc_query_stem = doc_query[:-4] if doc_query.endswith(".pdf") else doc_query
        matches = [
            f
            for f in pdf_files
            if f.name.lower() == doc_query or f.stem.lower() == doc_query_stem
        ]
        if not matches:
            logger.error(f"Document not found: {args.document}")
            return False
        if len(matches) > 1:
            logger.error(
                f"Multiple matches for '{args.document}': {[m.name for m in matches]}"
            )
            return False
        to_process = matches
        page_info = f" (page {args.page})" if args.page else ""
        logger.info(f"Force reprocessing: {to_process[0].name}{page_info}")
    elif args.reprocess_all:
        # Force reprocess all documents
        to_process = pdf_files
        logger.info(f"Force reprocessing all {len(to_process)} documents")
    else:
        # Check for already processed files
        to_process = []
        for pdf_path in pdf_files:
            if is_document_processed(pdf_path, config.output_path):
                logger.info(f"Skipping (already processed): {pdf_path.name}")
                already_processed += 1
            else:
                to_process.append(pdf_path)

        logger.info(
            f"Processing {len(to_process)} new PDFs, {already_processed} already processed"
        )

    # Process PDFs (sequential for now to avoid rate limits)
    total_pages = 0
    skipped_documents = []
    processed_documents = []
    failed_count = 0

    for pdf_path in tqdm(to_process, desc="Processing PDFs"):
        try:
            result = process_single_pdf(
                pdf_path,
                config.output_path,
                config,
                client,
                page_filter=args.page,
                profile_context=profile_context,
                birth_date=profile.birth_date,
                force_regenerate_images=bool(args.document),
            )
            if result == "skipped":
                skipped_documents.append(pdf_path.name)
            elif isinstance(result, int):
                total_pages += result
                processed_documents.append(pdf_path)
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            failed_count += 1

    # Summary
    if config.dry_run:
        logger.info(f"DRY RUN COMPLETE - Would process: {len(processed_documents)} documents ({total_pages} pages)")
        logger.info(f"Would skip (already processed): {already_processed} documents")
        logger.info(f"Would generate: {total_pages} .md files, {len(processed_documents)} summaries")
        if skipped_documents:
            logger.info(f"Would classify as non-exam: {len(skipped_documents)} documents")
        if failed_count > 0:
            logger.warning(f"Would fail: {failed_count} documents")
        return True

    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Processed: {len(processed_documents)} documents ({total_pages} pages)")
    logger.info(f"Skipped (not medical exams): {len(skipped_documents)}")
    if failed_count > 0:
        logger.warning(f"Failed: {failed_count}")

    # Report skipped documents for false negative review
    if skipped_documents:
        logger.info("Skipped Documents (review for false negatives):")
        for doc_name in skipped_documents:
            logger.info(f"  - {doc_name}")

    # Validate outputs for processed documents only (not skipped ones)
    missing_outputs = validate_pipeline_outputs(processed_documents, config.output_path)
    if missing_outputs:
        logger.warning(f"Missing outputs ({len(missing_outputs)}):")
        for item in missing_outputs:
            logger.warning(f"  {item}")
    else:
        logger.info("All outputs validated successfully")

    # Validate frontmatter
    frontmatter_issues = validate_frontmatter(config.output_path)
    if frontmatter_issues:
        logger.warning(f"Frontmatter issues ({len(frontmatter_issues)}):")
        for issue in frontmatter_issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("All frontmatter validated successfully")

    return True


def main():
    """Main pipeline entry point."""
    env_name = load_dotenv_with_env()
    args = parse_args()

    if env_name:
        print(f"Using environment: {env_name} (.env.{env_name})")

    # Handle --list-profiles
    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()
        if profiles:
            print("Available profiles:")
            for p in profiles:
                print(f"  - {p}")
        else:
            print("No profiles found in profiles/ directory")
        return

    # --page requires --document
    if args.page and not args.document:
        print("Error: --page requires --document")
        sys.exit(1)

    # Determine which profiles to run
    if args.profile:
        profiles_to_run = [args.profile]
    else:
        # No profile specified - run all profiles
        profiles_to_run = ProfileConfig.list_profiles()
        if not profiles_to_run:
            print("No profiles found in profiles/ directory")
            print("Use --list-profiles to see available profiles or create a profile.")
            sys.exit(1)
        print(
            f"Running all {len(profiles_to_run)} profiles: {', '.join(profiles_to_run)}"
        )

    # Run each profile
    success_count = 0
    failed_profiles = []

    for profile_name in profiles_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running profile: {profile_name}")
        print("=" * 60)

        if run_profile(profile_name, args):
            success_count += 1
        else:
            failed_profiles.append(profile_name)

    # Summary when running multiple profiles
    if len(profiles_to_run) > 1:
        print(f"\n{'=' * 60}")
        print("All Profiles Summary")
        print("=" * 60)
        print(f"Successful: {success_count}/{len(profiles_to_run)}")
        if failed_profiles:
            print(f"Failed profiles: {', '.join(failed_profiles)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
