"""Main pipeline for medical exam extraction and summarization."""

import argparse
import json
import re
import shutil
import sys
import logging
from collections import defaultdict
from pathlib import Path

from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from config import ExtractionConfig, ProfileConfig
from extraction import extract_exams_from_page_image, self_consistency
from standardization import standardize_exam_types
from summarization import summarize_document
from utils import preprocess_page_image, setup_logging

logger = logging.getLogger(__name__)


def is_document_processed(pdf_path: Path, output_path: Path) -> bool:
    """Check if a PDF has already been processed by looking for JSON extraction files."""
    doc_stem = pdf_path.stem
    doc_output_dir = output_path / doc_stem
    # A document is considered processed if its output directory exists and has JSON files
    if not doc_output_dir.exists():
        return False
    json_files = list(doc_output_dir.glob(f"{doc_stem}.*.json"))
    return len(json_files) > 0


def save_transcription_file(
    exams: list[dict],
    doc_output_dir: Path,
    doc_stem: str,
    page_num: int
) -> None:
    """
    Save page transcription as markdown file.
    - .md = raw transcription verbatim
    """
    md_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.md"
    transcriptions = [exam.get("transcription", "") for exam in exams]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(transcriptions).strip() + "\n")


def save_document_summary(
    summary: str,
    doc_output_dir: Path,
    doc_stem: str
) -> None:
    """
    Save document-level summary as markdown file.
    - .summary.md = comprehensive clinical summary for the entire document
    """
    if summary:
        summary_path = doc_output_dir / f"{doc_stem}.summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary.strip() + "\n")


def save_metadata_json(
    exams: list[dict],
    doc_output_dir: Path,
    doc_stem: str,
    page_num: int
) -> Path:
    """
    Save exam metadata as JSON (no transcription - that goes in .md file).
    """
    json_filename = f"{doc_stem}.{page_num:03d}.json"
    json_path = doc_output_dir / json_filename

    # Strip transcription and summary from metadata (they have their own .md files)
    metadata_exams = []
    for exam in exams:
        meta = {k: v for k, v in exam.items() if k not in ("transcription", "summary")}
        metadata_exams.append(meta)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"exams": metadata_exams}, f, ensure_ascii=False, indent=2)

    return json_path


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


def process_single_pdf(
    pdf_path: Path,
    output_path: Path,
    config: ExtractionConfig,
    client: OpenAI,
    page_filter: int | None = None
) -> Path | None:
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_path: Base output directory
        config: Extraction configuration
        client: OpenAI client instance
        page_filter: If set, only process this specific page number

    Returns:
        Number of exams extracted, or None if processing failed
    """
    doc_stem = pdf_path.stem
    doc_output_dir = output_path / doc_stem
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy source PDF to output directory (skip if already exists or permission denied)
    dest_pdf = doc_output_dir / pdf_path.name
    if not dest_pdf.exists():
        try:
            shutil.copy2(pdf_path, dest_pdf)
        except PermissionError:
            logger.warning(f"Could not copy PDF to output (permission denied): {pdf_path.name}")

    logger.info(f"Processing: {pdf_path.name}")

    # Convert PDF to images
    try:
        pages = convert_from_path(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {pdf_path.name}: {e}")
        return None

    all_exams = []
    report_date = None

    for page_num, page_image in enumerate(pages, start=1):
        # Skip pages if filter is active
        if page_filter is not None and page_num != page_filter:
            continue

        # Preprocess and save page image
        processed_image = preprocess_page_image(page_image)
        image_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.jpg"
        processed_image.save(str(image_path), "JPEG", quality=95)

        # Extract exams from page using self-consistency
        try:
            extraction_result, _ = self_consistency(
                extract_exams_from_page_image,
                config.self_consistency_model_id,
                config.n_extractions,
                image_path,
                config.extract_model_id,
                client
            )
        except Exception as e:
            logger.error(f"Extraction failed for {image_path.name}: {e}")
            extraction_result = {"exams": [], "page_has_exam_data": None}

        # Save raw extraction JSON
        json_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_result, f, ensure_ascii=False, indent=2)

        # Capture report date from first page if available
        if page_num == 1 and extraction_result.get("report_date"):
            report_date = extraction_result["report_date"]

        # Process extracted exams
        for exam in extraction_result.get("exams", []):
            exam["source_file"] = pdf_path.name
            exam["page_number"] = page_num

            # Resolve exam date
            if not exam.get("exam_date"):
                exam["exam_date"] = report_date

            all_exams.append(exam)

    # If no exams extracted but we have pages, try to extract date from filename
    if not all_exams:
        logger.warning(f"No exams extracted from {pdf_path.name}")
        return None

    # Resolve dates from filename if missing
    filename_date = extract_date_from_filename(pdf_path.name)
    for exam in all_exams:
        if not exam.get("exam_date") and filename_date:
            exam["exam_date"] = filename_date

    # Standardize exam types
    raw_names = [exam.get("exam_name_raw", "") for exam in all_exams]
    standardized = standardize_exam_types(raw_names, config.extract_model_id, client)

    for exam in all_exams:
        raw_name = exam.get("exam_name_raw", "")
        if raw_name in standardized:
            exam_type, std_name = standardized[raw_name]
            exam["exam_type"] = exam_type
            exam["exam_name_standardized"] = std_name

    # Generate document-level summary from all transcriptions
    document_summary = summarize_document(all_exams, config.summarize_model_id, client)

    # Group exams by page and save transcription/metadata files per page
    exams_by_page = defaultdict(list)
    for exam in all_exams:
        page_num = exam.get("page_number", 1)
        exams_by_page[page_num].append(exam)

    for page_num, page_exams in sorted(exams_by_page.items()):
        save_transcription_file(page_exams, doc_output_dir, doc_stem, page_num)
        save_metadata_json(page_exams, doc_output_dir, doc_stem, page_num)

    # Save one summary for the entire document
    save_document_summary(document_summary, doc_output_dir, doc_stem)

    logger.info(f"Saved {len(exams_by_page)} pages ({len(all_exams)} exams) for: {pdf_path.name}")

    return len(all_exams)


def regenerate_summaries(output_path: Path, config: ExtractionConfig, client: OpenAI):
    """
    Regenerate document-level summary files from existing transcription (.md) and metadata (.json) files.

    Reads transcription from .md files and metadata from .json files,
    re-runs document-level summarization, and saves updated .summary.md files.
    """
    # Find all document directories
    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name != "logs"]

    logger.info(f"Found {len(doc_dirs)} document directories to regenerate")

    total_exams = 0
    for doc_dir in tqdm(doc_dirs, desc="Regenerating summaries"):
        doc_stem = doc_dir.name

        # Find all JSON metadata files (exclude .summary.md pattern)
        json_files = sorted([f for f in doc_dir.glob(f"{doc_stem}.*.json")])
        if not json_files:
            logger.warning(f"No JSON files found in {doc_dir}")
            continue

        all_exams = []

        for json_path in json_files:
            # Extract page number from filename
            parts = json_path.stem.split(".")
            if len(parts) >= 2:
                try:
                    page_num = int(parts[-1])
                except ValueError:
                    continue
            else:
                continue

            # Read metadata from JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Read transcription from corresponding .md file
            md_path = doc_dir / f"{doc_stem}.{page_num:03d}.md"
            if not md_path.exists():
                logger.warning(f"No transcription file found: {md_path}")
                continue

            with open(md_path, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()

            # Combine metadata with transcription
            for exam in metadata.get("exams", []):
                exam["transcription"] = transcription
                exam["page_number"] = page_num
                all_exams.append(exam)

        if not all_exams:
            logger.warning(f"No exams found in {doc_dir}")
            continue

        # Delete existing .summary.md files (both old page-level and document-level)
        for old_summary in doc_dir.glob("*.summary.md"):
            old_summary.unlink()

        # Generate document-level summary
        document_summary = summarize_document(all_exams, config.summarize_model_id, client)

        # Save one summary for the entire document
        save_document_summary(document_summary, doc_dir, doc_stem)

        logger.info(f"Regenerated summary for {len(all_exams)} exams: {doc_stem}")
        total_exams += len(all_exams)

    return total_exams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and summarize medical exam reports from PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --profile tsilva              # Process all new PDFs
  python main.py --list-profiles               # List available profiles
  python main.py -p tsilva --regenerate        # Regenerate summaries
  python main.py -p tsilva -d exam.pdf         # Reprocess specific document
  python main.py -p tsilva -d exam.pdf --page 2  # Reprocess specific page
        """
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        help="Profile name (without .json extension)"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate summaries from existing transcription files"
    )
    parser.add_argument(
        "--document", "-d",
        type=str,
        help="Process only this document (filename or stem). Forces reprocessing."
    )
    parser.add_argument(
        "--page",
        type=int,
        help="Process only this page number (requires --document)"
    )
    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    load_dotenv()
    args = parse_args()

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

    # Profile is required
    if not args.profile:
        print("Error: --profile is required.")
        print("Use --list-profiles to see available profiles.")
        sys.exit(1)

    # --page requires --document
    if args.page and not args.document:
        print("Error: --page requires --document")
        sys.exit(1)

    # Load configuration
    config = ExtractionConfig.from_env()

    # Apply profile overrides
    profile_path = Path("profiles") / f"{args.profile}.json"
    if not profile_path.exists():
        print(f"Profile not found: {profile_path}")
        print(f"Available profiles: {ProfileConfig.list_profiles()}")
        sys.exit(1)
    profile = ProfileConfig.from_file(profile_path, config)

    # Override config with profile paths
    if profile.input_path:
        config.input_path = profile.input_path
    if profile.output_path:
        config.output_path = profile.output_path
        config.output_path.mkdir(parents=True, exist_ok=True)
    if profile.input_file_regex:
        config.input_file_regex = profile.input_file_regex

    # Validate required paths after profile overrides
    if not config.input_path:
        print("Error: INPUT_PATH not set in .env or profile.")
        sys.exit(1)
    if not config.input_path.exists():
        print(f"Error: INPUT_PATH does not exist: {config.input_path}")
        sys.exit(1)
    if not config.output_path:
        print("Error: OUTPUT_PATH not set in .env or profile.")
        sys.exit(1)
    if not config.input_file_regex:
        print("Error: INPUT_FILE_REGEX not set in .env or profile.")
        sys.exit(1)

    # Setup logging
    log_dir = config.output_path / "logs"
    setup_logging(log_dir, clear_logs=True)

    logger.info("=" * 60)
    logger.info("Medical Exams Parser - Starting Pipeline")
    logger.info("=" * 60)
    if profile:
        logger.info(f"Profile: {profile.name}")
    logger.info(f"Input path: {config.input_path}")
    logger.info(f"Output path: {config.output_path}")
    logger.info(f"Extract model: {config.extract_model_id}")
    logger.info(f"Summarize model: {config.summarize_model_id}")
    logger.info(f"N extractions: {config.n_extractions}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.openrouter_api_key
    )

    # Handle --regenerate mode
    if args.regenerate:
        logger.info("Regeneration mode: re-summarizing from existing transcription files")
        total_exams = regenerate_summaries(config.output_path, config, client)
        logger.info("=" * 60)
        logger.info("Regeneration Complete")
        logger.info("=" * 60)
        logger.info(f"Regenerated summaries for {total_exams} exams")
        return

    # Find PDF files
    pdf_pattern = re.compile(config.input_file_regex)
    pdf_files = sorted([
        f for f in config.input_path.glob("**/*.pdf")
        if pdf_pattern.match(f.name)
    ])

    logger.info(f"Found {len(pdf_files)} PDF files matching pattern")

    if not pdf_files:
        logger.warning("No PDF files found. Check INPUT_PATH and INPUT_FILE_REGEX.")
        return

    # Select documents to process
    if args.document:
        # Find the specific document (by filename or stem, case-insensitive)
        doc_query = args.document.lower()
        # Strip .pdf extension if present for stem matching
        doc_query_stem = doc_query[:-4] if doc_query.endswith('.pdf') else doc_query
        matches = [f for f in pdf_files
                   if f.name.lower() == doc_query
                   or f.stem.lower() == doc_query_stem]
        if not matches:
            logger.error(f"Document not found: {args.document}")
            sys.exit(1)
        if len(matches) > 1:
            logger.error(f"Multiple matches for '{args.document}': {[m.name for m in matches]}")
            sys.exit(1)
        to_process = matches
        page_info = f" (page {args.page})" if args.page else ""
        logger.info(f"Force reprocessing: {to_process[0].name}{page_info}")
    else:
        # Check for already processed files
        to_process = []
        already_processed = 0
        for pdf_path in pdf_files:
            if is_document_processed(pdf_path, config.output_path):
                logger.info(f"Skipping (already processed): {pdf_path.name}")
                already_processed += 1
            else:
                to_process.append(pdf_path)

        logger.info(f"Processing {len(to_process)} new PDFs, {already_processed} already processed")

    # Process PDFs (sequential for now to avoid rate limits)
    total_exams = 0
    for pdf_path in tqdm(to_process, desc="Processing PDFs"):
        try:
            exam_count = process_single_pdf(
                pdf_path, config.output_path, config, client,
                page_filter=args.page
            )
            if exam_count:
                total_exams += exam_count
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Processed {len(to_process)} PDFs, extracted {total_exams} exams")


if __name__ == "__main__":
    main()
