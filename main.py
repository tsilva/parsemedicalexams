"""Main pipeline for medical exam extraction and summarization."""

import argparse
import json
import re
import shutil
import sys
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from config import ExtractionConfig, ProfileConfig
from extraction import (
    extract_exams_from_page_image,
    self_consistency,
    classify_document,
    transcribe_page,
    DocumentClassification
)
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
) -> int | None | str:
    """
    Process a single PDF file using two-phase approach:
    1. Classify document (is it a medical exam?)
    2. If yes, transcribe all pages verbatim

    Args:
        pdf_path: Path to the PDF file
        output_path: Base output directory
        config: Extraction configuration
        client: OpenAI client instance
        page_filter: If set, only process this specific page number

    Returns:
        Number of pages processed if success
        None if processing failed
        "skipped" if document is not a medical exam
    """
    doc_stem = pdf_path.stem
    logger.info(f"Processing: {pdf_path.name}")

    # Convert PDF to images in temp directory first (before classification)
    try:
        pages = convert_from_path(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {pdf_path.name}: {e}")
        return None

    # Create temp directory for classification images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_image_paths = []

        # Preprocess and save images to temp directory
        for page_num, page_image in enumerate(pages, start=1):
            processed_image = preprocess_page_image(page_image)
            temp_image_path = temp_path / f"{doc_stem}.{page_num:03d}.jpg"
            processed_image.save(str(temp_image_path), "JPEG", quality=95)
            temp_image_paths.append(temp_image_path)

        # PHASE 1: Classify document
        logger.debug(f"Classifying document: {pdf_path.name} ({len(temp_image_paths)} pages)")
        try:
            classification = classify_document(
                temp_image_paths,
                config.extract_model_id,
                client
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
                logger.warning(f"Could not copy PDF to output (permission denied): {pdf_path.name}")

        # Move images from temp to output directory
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

        # Transcribe all pages
        all_exams = []
        for page_num, image_path in enumerate(image_paths, start=1):
            # Skip pages if filter is active
            if page_filter is not None and page_num != page_filter:
                continue

            # Transcribe page (with optional self-consistency voting)
            confidence = None
            try:
                if config.n_extractions > 1:
                    transcription, all_transcriptions = self_consistency(
                        transcribe_page,
                        config.self_consistency_model_id,
                        config.n_extractions,
                        image_path,
                        config.extract_model_id,
                        client
                    )
                    # Calculate agreement level for confidence
                    agreement_count = sum(1 for t in all_transcriptions if t == transcription)
                    confidence = agreement_count / len(all_transcriptions)
                    if confidence < 1.0:
                        logger.info(f"Self-consistency: {agreement_count}/{len(all_transcriptions)} agreement for {image_path.name}")
                else:
                    transcription = transcribe_page(
                        image_path,
                        config.extract_model_id,
                        client
                    )
            except Exception as e:
                logger.error(f"Transcription failed for {image_path.name}: {e}")
                transcription = ""

            if not transcription:
                logger.warning(f"Empty transcription for {image_path.name}")

            # Create exam entry for this page
            exam = {
                "exam_name_raw": exam_name,
                "exam_date": exam_date,
                "transcription": transcription,
                "page_number": page_num,
                "source_file": pdf_path.name,
                "transcription_confidence": confidence
            }
            all_exams.append(exam)

            # Save transcription file
            save_transcription_file([exam], doc_output_dir, doc_stem, page_num)

            # Save metadata JSON (without transcription)
            json_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.json"
            metadata = {
                "exams": [{
                    "exam_date": exam_date,
                    "exam_name_raw": exam_name,
                    "exam_type": None,
                    "exam_name_standardized": None,
                    "page_number": page_num,
                    "source_file": pdf_path.name,
                    "transcription_confidence": confidence
                }]
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Standardize exam types
    if all_exams:
        raw_names = list(set(exam.get("exam_name_raw", "") for exam in all_exams))
        standardized = standardize_exam_types(raw_names, config.extract_model_id, client)

        for exam in all_exams:
            raw_name = exam.get("exam_name_raw", "")
            if raw_name in standardized:
                exam_type, std_name = standardized[raw_name]
                exam["exam_type"] = exam_type
                exam["exam_name_standardized"] = std_name

        # Update JSON files with standardized info
        for exam in all_exams:
            page_num = exam.get("page_number", 1)
            json_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.json"
            metadata = {
                "exams": [{
                    "exam_date": exam.get("exam_date"),
                    "exam_name_raw": exam.get("exam_name_raw"),
                    "exam_type": exam.get("exam_type"),
                    "exam_name_standardized": exam.get("exam_name_standardized"),
                    "page_number": page_num,
                    "source_file": pdf_path.name,
                    "transcription_confidence": exam.get("transcription_confidence")
                }]
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Generate document-level summary
        document_summary = summarize_document(all_exams, config.summarize_model_id, client)
        save_document_summary(document_summary, doc_output_dir, doc_stem)

    logger.info(f"Processed {len(all_exams)} pages for: {pdf_path.name}")
    return len(all_exams)


def regenerate_summaries(output_path: Path, config: ExtractionConfig, client: OpenAI, input_path: Path | None = None):
    """
    Regenerate document-level summary files from existing transcription (.md) and metadata (.json) files.

    Reads transcription from .md files and metadata from .json files,
    re-runs document-level summarization, and saves updated .summary.md files.
    If input_path is provided, also copies source PDFs to output directories if missing.
    """
    # Find all document directories
    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name != "logs"]

    logger.info(f"Found {len(doc_dirs)} document directories to regenerate")

    total_exams = 0
    for doc_dir in tqdm(doc_dirs, desc="Regenerating summaries"):
        doc_stem = doc_dir.name

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
                        logger.warning(f"Could not copy PDF (permission denied): {source_pdfs[0].name}")

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


def validate_pipeline_outputs(
    pdf_files: list[Path],
    output_path: Path
) -> list[str]:
    """
    Validate that all expected output files exist for each source PDF.

    Returns list of missing file paths (empty if all complete).
    """
    missing = []

    for pdf_path in pdf_files:
        doc_stem = pdf_path.stem
        doc_output_dir = output_path / doc_stem

        # Check target folder exists
        if not doc_output_dir.exists():
            missing.append(f"Missing target folder: {doc_output_dir}")
            continue  # Can't check other files if folder doesn't exist

        # Check source PDF copy
        pdf_copy = doc_output_dir / pdf_path.name
        if not pdf_copy.exists():
            missing.append(f"Missing source PDF copy: {pdf_copy}")

        # Get page count from source PDF
        try:
            pages = convert_from_path(str(pdf_path))
            page_count = len(pages)
        except Exception as e:
            missing.append(f"Could not read PDF to count pages: {pdf_path} ({e})")
            continue

        # Check per-page files
        for page_num in range(1, page_count + 1):
            # Image
            img_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.jpg"
            if not img_path.exists():
                missing.append(f"Missing page image: {img_path}")

            # Metadata
            json_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.json"
            if not json_path.exists():
                missing.append(f"Missing page metadata: {json_path}")

            # Transcription
            md_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.md"
            if not md_path.exists():
                missing.append(f"Missing page transcription: {md_path}")

        # Check document summary
        summary_path = doc_output_dir / f"{doc_stem}.summary.md"
        if not summary_path.exists():
            missing.append(f"Missing document summary: {summary_path}")

    return missing


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
  python main.py -p tsilva --reprocess-all     # Force reprocess all documents
  python main.py -p tsilva -d exam.pdf         # Reprocess specific document
  python main.py -p tsilva -d exam.pdf --page 2  # Reprocess specific page
        """
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        help="Profile name (without extension)"
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
        "--reprocess-all",
        action="store_true",
        help="Force reprocessing of all documents (ignores already processed)"
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

    # Override arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model ID for extraction (overrides profile/env)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="Number of parallel workers (overrides profile/env)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Regex pattern for input files (overrides profile)"
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

    # Load profile (required)
    profile_path = None
    for ext in ('.yaml', '.yml', '.json'):
        p = Path(f"profiles/{args.profile}{ext}")
        if p.exists():
            profile_path = p
            break

    if not profile_path:
        print(f"Error: Profile '{args.profile}' not found")
        print("Use --list-profiles to see available profiles.")
        sys.exit(1)

    profile = ProfileConfig.from_file(profile_path)

    # Validate profile has required paths
    if not profile.input_path:
        print(f"Error: Profile '{args.profile}' has no input_path defined.")
        sys.exit(1)
    if not profile.output_path:
        print(f"Error: Profile '{args.profile}' has no output_path defined.")
        sys.exit(1)

    # Validate input path exists
    if not profile.input_path.exists():
        print(f"Error: Input path does not exist: {profile.input_path}")
        sys.exit(1)

    # Ensure output directory exists
    profile.output_path.mkdir(parents=True, exist_ok=True)

    # Load base config from environment (API keys and model settings)
    config = ExtractionConfig.from_env()

    # Apply profile paths to config
    config.input_path = profile.input_path
    config.output_path = profile.output_path
    config.input_file_regex = profile.input_file_regex or config.input_file_regex or ".*\\.pdf"

    # Apply profile overrides
    if profile.model:
        config.extract_model_id = profile.model
        config.self_consistency_model_id = profile.model
        config.summarize_model_id = profile.model
    if profile.workers:
        config.max_workers = profile.workers

    # Apply CLI overrides (highest priority)
    if args.model:
        config.extract_model_id = args.model
        config.self_consistency_model_id = args.model
        config.summarize_model_id = args.model
    if args.workers:
        config.max_workers = args.workers
    if args.pattern:
        config.input_file_regex = args.pattern

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
        total_exams = regenerate_summaries(config.output_path, config, client, config.input_path)
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
    elif args.reprocess_all:
        # Force reprocess all documents
        to_process = pdf_files
        logger.info(f"Force reprocessing all {len(to_process)} documents")
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
    total_pages = 0
    skipped_documents = []
    processed_documents = []
    failed_count = 0

    for pdf_path in tqdm(to_process, desc="Processing PDFs"):
        try:
            result = process_single_pdf(
                pdf_path, config.output_path, config, client,
                page_filter=args.page
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
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Processed: {len(processed_documents)} documents ({total_pages} pages)")
    logger.info(f"Skipped (not medical exams): {len(skipped_documents)}")
    if failed_count > 0:
        logger.warning(f"Failed: {failed_count}")

    # Report skipped documents for false negative review
    if skipped_documents:
        logger.info("=" * 60)
        logger.info("Skipped Documents (review for false negatives):")
        logger.info("=" * 60)
        for doc_name in skipped_documents:
            logger.info(f"  - {doc_name}")

    # Validate outputs for processed documents only (not skipped ones)
    logger.info("Validating pipeline outputs...")
    missing_outputs = validate_pipeline_outputs(processed_documents, config.output_path)

    if missing_outputs:
        logger.warning("=" * 60)
        logger.warning("⚠️  Missing outputs detected:")
        logger.warning("=" * 60)
        for item in missing_outputs:
            logger.warning(f"  ⚠️  {item}")
        logger.warning(f"⚠️  Total missing: {len(missing_outputs)}")
    else:
        logger.info("All outputs validated successfully")


if __name__ == "__main__":
    main()
