"""Main pipeline for medical exam extraction and summarization."""

import argparse
import re
import shutil
import sys
import logging
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from pdf2image import convert_from_path
from openai import OpenAI
from tqdm import tqdm

from config import ExtractionConfig, ProfileConfig
from extraction import (
    extract_exams_from_page_image,
    self_consistency,
    classify_document,
    transcribe_page,
    score_transcription_confidence,
    DocumentClassification
)
from standardization import standardize_exam_types
from summarization import summarize_document
from utils import preprocess_page_image, setup_logging, load_dotenv_with_env

logger = logging.getLogger(__name__)


def is_document_processed(pdf_path: Path, output_path: Path) -> bool:
    """Check if a PDF has already been processed by looking for transcription .md files."""
    doc_stem = pdf_path.stem
    doc_output_dir = output_path / doc_stem
    # A document is considered processed if its output directory exists and has transcription .md files
    if not doc_output_dir.exists():
        return False
    # Find .md files excluding .summary.md
    md_files = [f for f in doc_output_dir.glob(f"{doc_stem}.*.md") if not f.name.endswith(".summary.md")]
    return len(md_files) > 0


def save_transcription_file(
    exams: list[dict],
    doc_output_dir: Path,
    doc_stem: str,
    page_num: int
) -> None:
    """
    Save page transcription as markdown file with YAML frontmatter.
    - .md = YAML frontmatter + raw transcription verbatim

    Frontmatter includes all metadata (no separate JSON file needed).
    """
    md_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.md"

    # Build frontmatter from first exam
    frontmatter = {}
    if exams:
        exam = exams[0]
        if exam.get("exam_date"):
            frontmatter["exam_date"] = exam["exam_date"]
        if exam.get("exam_name_raw"):
            frontmatter["exam_name_raw"] = exam["exam_name_raw"]
        if exam.get("exam_name_standardized"):
            frontmatter["title"] = exam["exam_name_standardized"]
        if exam.get("exam_type"):
            frontmatter["category"] = exam["exam_type"]
        if exam.get("physician_name"):
            frontmatter["doctor"] = exam["physician_name"]
        if exam.get("facility_name"):
            frontmatter["facility"] = exam["facility_name"]
        if exam.get("department"):
            frontmatter["department"] = exam["department"]
        if exam.get("transcription_confidence") is not None:
            frontmatter["confidence"] = exam["transcription_confidence"]
        if exam.get("page_number"):
            frontmatter["page"] = exam["page_number"]
        if exam.get("source_file"):
            frontmatter["source"] = exam["source_file"]

    # Write file
    transcriptions = [exam.get("transcription", "") for exam in exams]
    with open(md_path, 'w', encoding='utf-8') as f:
        if frontmatter:
            f.write("---\n")
            f.write(yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False))
            f.write("---\n\n")
        f.write("\n\n".join(transcriptions).strip() + "\n")


def save_document_summary(
    summary: str,
    doc_output_dir: Path,
    doc_stem: str,
    exams: list[dict] = None
) -> None:
    """
    Save document-level summary as markdown file with YAML frontmatter.
    - .summary.md = YAML frontmatter + comprehensive clinical summary for the entire document
    """
    if summary:
        summary_path = doc_output_dir / f"{doc_stem}.summary.md"

        # Build frontmatter from first exam
        frontmatter = {}
        if exams:
            exam = exams[0]
            if exam.get("exam_date"):
                frontmatter["exam_date"] = exam["exam_date"]
            if exam.get("exam_name_raw"):
                frontmatter["exam_name_raw"] = exam["exam_name_raw"]
            if exam.get("exam_name_standardized"):
                frontmatter["title"] = exam["exam_name_standardized"]
            if exam.get("exam_type"):
                frontmatter["category"] = exam["exam_type"]
            if exam.get("physician_name"):
                frontmatter["doctor"] = exam["physician_name"]
            if exam.get("facility_name"):
                frontmatter["facility"] = exam["facility_name"]
            if exam.get("department"):
                frontmatter["department"] = exam["department"]

        with open(summary_path, 'w', encoding='utf-8') as f:
            if frontmatter:
                f.write("---\n")
                f.write(yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False))
                f.write("---\n\n")
            f.write(summary.strip() + "\n")


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


def extract_dates_from_text(text: str) -> list[str]:
    """
    Extract all dates in YYYY-MM-DD format from text.

    Handles common Portuguese/European date formats:
    - DD/MM/YYYY (e.g., 20/11/2024)
    - DD-MM-YYYY (e.g., 20-11-2024)
    - YYYY-MM-DD (e.g., 2024-11-20)
    - DD de MMMM de YYYY (e.g., 20 de Novembro de 2024)

    Args:
        text: Text to extract dates from

    Returns:
        List of dates in YYYY-MM-DD format
    """
    dates = []

    # Pattern 1: YYYY-MM-DD (already correct format)
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        year, month, day = match.groups()
        # Validate ranges
        if 1900 <= int(year) <= 2100 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            dates.append(f"{year}-{month}-{day}")

    # Pattern 2: DD/MM/YYYY or DD-MM-YYYY
    for match in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text):
        day, month, year = match.groups()
        # Validate ranges
        day_int, month_int, year_int = int(day), int(month), int(year)
        if 1900 <= year_int <= 2100 and 1 <= month_int <= 12 and 1 <= day_int <= 31:
            dates.append(f"{year}-{month:0>2}-{day:0>2}")

    return dates


def select_most_frequent_date(exams: list[dict]) -> str | None:
    """
    Select the most frequent date across all pages using frequency-based voting.

    This handles multi-era documents where administrative pages (e.g., 2024 cover letter)
    may have different dates than the actual medical records (e.g., 1997 hospitalization).

    First extracts dates from each page's transcription, then votes on the most frequent.

    Args:
        exams: List of exam dictionaries with transcriptions

    Returns:
        The most frequently occurring date, or None if no dates found

    Example:
        - Page 1: 2024 (1 occurrence)
        - Pages 2-45: 1997 (44 occurrences)
        - Result: 1997 (most frequent = real document date)
    """
    from collections import Counter

    # Extract dates from each page's transcription
    all_dates = []
    for exam in exams:
        transcription = exam.get("transcription", "")
        if transcription:
            page_dates = extract_dates_from_text(transcription)
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
        logger.info(f"Selected most frequent date: {most_common_date} ({count}/{len(all_dates)} pages)")

    return most_common_date


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

        # Define page processing function for parallel execution
        def process_page(page_num: int, image_path: Path) -> dict | None:
            """Process a single page - returns exam dict or None if skipped."""
            # Skip pages if filter is active
            if page_filter is not None and page_num != page_filter:
                return None

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
                    # Use LLM to assess semantic agreement for confidence
                    confidence = score_transcription_confidence(
                        transcription,
                        all_transcriptions,
                        config.self_consistency_model_id,
                        client
                    )
                    if confidence < 1.0:
                        logger.info(f"Self-consistency confidence: {confidence:.2f} for {image_path.name}")
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
            return {
                "exam_name_raw": exam_name,
                "exam_date": exam_date,
                "transcription": transcription,
                "page_number": page_num,
                "source_file": pdf_path.name,
                "transcription_confidence": confidence,
                "physician_name": classification.physician_name,
                "department": classification.department,
                "facility_name": facility_name
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
                        # Save transcription file immediately
                        save_transcription_file([exam], doc_output_dir, doc_stem, page_num)
                except Exception as e:
                    logger.error(f"Page {page_num} processing failed: {e}")

        # Sort by page number for consistent ordering
        all_exams.sort(key=lambda x: x["page_number"])

    # Apply frequency-based date correction for multi-era documents
    if all_exams:
        corrected_date = select_most_frequent_date(all_exams)
        if corrected_date:
            logger.debug(f"Selected document date by frequency: {corrected_date}")
            # Update all exams with the corrected date
            for exam in all_exams:
                exam["exam_date"] = corrected_date

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

        # Resave transcription files with YAML frontmatter (now includes standardized info)
        for exam in all_exams:
            page_num = exam.get("page_number", 1)
            save_transcription_file([exam], doc_output_dir, doc_stem, page_num)

        # Generate document-level summary
        document_summary = summarize_document(all_exams, config.summarize_model_id, client)
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
            transcription = transcription[end_marker + 3:].strip()

    return frontmatter, transcription


def frontmatter_to_exam(frontmatter: dict, transcription: str, page_num: int, source_file: str = None) -> dict:
    """
    Convert frontmatter fields to internal exam dict format.

    Frontmatter fields -> Internal fields:
    - date -> exam_date
    - exam_name_raw -> exam_name_raw
    - title -> exam_name_standardized
    - category -> exam_type
    - doctor -> physician_name
    - facility -> facility_name
    - department -> department
    - confidence -> transcription_confidence
    - page -> page_number
    - source -> source_file
    """
    return {
        "exam_date": frontmatter.get("exam_date"),
        "exam_name_raw": frontmatter.get("exam_name_raw"),
        "exam_name_standardized": frontmatter.get("title"),
        "exam_type": frontmatter.get("category"),
        "physician_name": frontmatter.get("doctor"),
        "facility_name": frontmatter.get("facility"),
        "department": frontmatter.get("department"),
        "transcription_confidence": frontmatter.get("confidence"),
        "transcription": transcription,
        "page_number": frontmatter.get("page") or page_num,
        "source_file": frontmatter.get("source") or source_file
    }


def regenerate_summaries(output_path: Path, config: ExtractionConfig, client: OpenAI, input_path: Path | None = None):
    """
    Regenerate document-level summary files from existing transcription (.md) files.

    Reads metadata from YAML frontmatter and transcription content from .md files,
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

        # Find all transcription .md files (exclude .summary.md)
        md_files = sorted([
            f for f in doc_dir.glob(f"{doc_stem}.*.md")
            if not f.name.endswith(".summary.md")
        ])
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
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            frontmatter, transcription = parse_frontmatter(content)

            # Convert frontmatter to exam dict (use doc_stem.pdf as fallback source)
            exam = frontmatter_to_exam(frontmatter, transcription, page_num, f"{doc_stem}.pdf")
            all_exams.append(exam)

        if not all_exams:
            logger.warning(f"No exams found in {doc_dir}")
            continue

        # Delete existing .summary.md files
        for old_summary in doc_dir.glob("*.summary.md"):
            old_summary.unlink()

        # Generate document-level summary
        document_summary = summarize_document(all_exams, config.summarize_model_id, client)

        # Save one summary for the entire document
        save_document_summary(document_summary, doc_dir, doc_stem, all_exams)

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

            # Transcription (with frontmatter containing all metadata)
            md_path = doc_output_dir / f"{doc_stem}.{page_num:03d}.md"
            if not md_path.exists():
                missing.append(f"Missing page transcription: {md_path}")

        # Check document summary
        summary_path = doc_output_dir / f"{doc_stem}.summary.md"
        if not summary_path.exists():
            missing.append(f"Missing document summary: {summary_path}")

    return missing


def validate_frontmatter(output_path: Path) -> list[str]:
    """
    Validate that all .md files have YAML frontmatter with required fields.

    Returns list of files missing frontmatter or required fields.
    """
    issues = []
    required_fields = {"exam_date", "category", "title"}  # Minimum required fields

    # Find all document directories
    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name != "logs"]

    for doc_dir in doc_dirs:
        doc_stem = doc_dir.name

        # Check all .md files (both transcription and summary)
        md_files = list(doc_dir.glob(f"{doc_stem}.*.md"))

        for md_path in md_files:
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for frontmatter
                if not content.startswith("---"):
                    issues.append(f"Missing frontmatter: {md_path.name}")
                    continue

                # Parse frontmatter
                end_marker = content.find("---", 3)
                if end_marker == -1:
                    issues.append(f"Malformed frontmatter: {md_path.name}")
                    continue

                frontmatter_str = content[3:end_marker].strip()
                try:
                    frontmatter = yaml.safe_load(frontmatter_str)
                except yaml.YAMLError:
                    issues.append(f"Invalid YAML frontmatter: {md_path.name}")
                    continue

                if not frontmatter:
                    issues.append(f"Empty frontmatter: {md_path.name}")
                    continue

                # Check for required fields
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
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name to load (loads .env.{name} instead of .env)"
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
    for ext in ('.yaml', '.yml', '.json'):
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
    logger.info(f"Profile: {profile.name}")
    logger.info(f"Input path: {config.input_path}")
    logger.info(f"Output path: {config.output_path}")
    logger.info(f"Extract model: {config.extract_model_id}")
    logger.info(f"Summarize model: {config.summarize_model_id}")
    logger.info(f"N extractions: {config.n_extractions}")
    logger.info(f"API base URL: {config.openrouter_base_url}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=config.openrouter_base_url,
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

        # Validate frontmatter
        logger.info("Validating frontmatter...")
        frontmatter_issues = validate_frontmatter(config.output_path)
        if frontmatter_issues:
            logger.warning("=" * 60)
            logger.warning("Frontmatter issues detected:")
            logger.warning("=" * 60)
            for issue in frontmatter_issues:
                logger.warning(f"  {issue}")
            logger.warning(f"Total issues: {len(frontmatter_issues)}")
        else:
            logger.info("All frontmatter validated successfully")

        return True

    # Find PDF files
    pdf_pattern = re.compile(config.input_file_regex)
    pdf_files = sorted([
        f for f in config.input_path.glob("**/*.pdf")
        if pdf_pattern.match(f.name)
    ])

    logger.info(f"Found {len(pdf_files)} PDF files matching pattern")

    if not pdf_files:
        logger.warning("No PDF files found. Check INPUT_PATH and INPUT_FILE_REGEX.")
        return True

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
            return False
        if len(matches) > 1:
            logger.error(f"Multiple matches for '{args.document}': {[m.name for m in matches]}")
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
        logger.warning("Missing outputs detected:")
        logger.warning("=" * 60)
        for item in missing_outputs:
            logger.warning(f"  {item}")
        logger.warning(f"Total missing: {len(missing_outputs)}")
    else:
        logger.info("All outputs validated successfully")

    # Validate frontmatter
    logger.info("Validating frontmatter...")
    frontmatter_issues = validate_frontmatter(config.output_path)
    if frontmatter_issues:
        logger.warning("=" * 60)
        logger.warning("Frontmatter issues detected:")
        logger.warning("=" * 60)
        for issue in frontmatter_issues:
            logger.warning(f"  {issue}")
        logger.warning(f"Total issues: {len(frontmatter_issues)}")
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
        print(f"Running all {len(profiles_to_run)} profiles: {', '.join(profiles_to_run)}")

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
