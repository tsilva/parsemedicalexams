"""Main pipeline for medical exam extraction and summarization."""

import argparse
import json
import re
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from config import ExtractionConfig, ProfileConfig
from extraction import extract_exams_from_page_image, self_consistency
from standardization import standardize_exam_types
from summarization import batch_summarize_exams
from utils import preprocess_page_image, setup_logging

logger = logging.getLogger(__name__)

# Column schema for output CSV
COLUMN_SCHEMA = {
    "date": {"dtype": "datetime64[ns]", "excel_width": 13},
    "exam_type": {"dtype": "str", "excel_width": 12},
    "exam_name_raw": {"dtype": "str", "excel_width": 30},
    "exam_name_standardized": {"dtype": "str", "excel_width": 30},
    "transcription": {"dtype": "str", "excel_width": 80},
    "summary": {"dtype": "str", "excel_width": 60},
    "source_file": {"dtype": "str", "excel_width": 25},
    "page_number": {"dtype": "Int64", "excel_width": 8},
}


def get_csv_path(pdf_path: Path, output_path: Path) -> Path:
    """Get CSV output path for a PDF file."""
    doc_stem = pdf_path.stem
    doc_output_dir = output_path / doc_stem
    return doc_output_dir / f"{doc_stem}.csv"


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
    client: OpenAI
) -> Path | None:
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_path: Base output directory
        config: Extraction configuration
        client: OpenAI client instance

    Returns:
        Path to the generated CSV file, or None if processing failed
    """
    doc_stem = pdf_path.stem
    doc_output_dir = output_path / doc_stem
    doc_output_dir.mkdir(parents=True, exist_ok=True)

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

    # Generate summaries
    all_exams = batch_summarize_exams(all_exams, config.summarize_model_id, client)

    # Convert to DataFrame
    df = pd.DataFrame(all_exams)

    # Rename date column
    if "exam_date" in df.columns:
        df = df.rename(columns={"exam_date": "date"})

    # Ensure all required columns exist
    for col in COLUMN_SCHEMA.keys():
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    df = df[[col for col in COLUMN_SCHEMA.keys() if col in df.columns]]

    # Save per-document CSV
    csv_path = doc_output_dir / f"{doc_stem}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path} ({len(df)} exams)")

    return csv_path


def merge_csv_files(csv_paths: list[Path], output_path: Path) -> pd.DataFrame:
    """Merge all per-document CSVs into a single DataFrame."""
    dfs = []
    for csv_path in csv_paths:
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)

    if not dfs:
        logger.warning("No CSV files to merge")
        return pd.DataFrame(columns=COLUMN_SCHEMA.keys())

    merged = pd.concat(dfs, ignore_index=True)

    # Sort by date (newest first)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.sort_values("date", ascending=False)

    return merged


def export_excel(df: pd.DataFrame, output_path: Path):
    """Export DataFrame to Excel with formatting."""
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='AllData', index=False)

        workbook = writer.book
        worksheet = writer.sheets['AllData']

        # Set column widths
        for i, col in enumerate(df.columns):
            if col in COLUMN_SCHEMA:
                width = COLUMN_SCHEMA[col].get("excel_width", 15)
                worksheet.set_column(i, i, width)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and summarize medical exam reports from PDFs"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Profile name to use (from profiles/ directory)"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit"
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

    # Load configuration
    config = ExtractionConfig.from_env()

    # Apply profile overrides if specified
    profile = None
    if args.profile:
        profile_path = Path("profiles") / f"{args.profile}.json"
        if not profile_path.exists():
            print(f"Profile not found: {profile_path}")
            print(f"Available profiles: {ProfileConfig.list_profiles()}")
            return
        profile = ProfileConfig.from_file(profile_path, config)

        # Override config with profile paths
        if profile.input_path:
            config.input_path = profile.input_path
        if profile.output_path:
            config.output_path = profile.output_path
            config.output_path.mkdir(parents=True, exist_ok=True)
        if profile.input_file_regex:
            config.input_file_regex = profile.input_file_regex

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

    # Process PDFs
    csv_paths = []

    # Check for already processed files
    to_process = []
    for pdf_path in pdf_files:
        csv_path = get_csv_path(pdf_path, config.output_path)
        if csv_path.exists():
            logger.info(f"Skipping (already processed): {pdf_path.name}")
            csv_paths.append(csv_path)
        else:
            to_process.append(pdf_path)

    logger.info(f"Processing {len(to_process)} new PDFs, {len(csv_paths)} already processed")

    # Process PDFs (sequential for now to avoid rate limits)
    for pdf_path in tqdm(to_process, desc="Processing PDFs"):
        try:
            result = process_single_pdf(pdf_path, config.output_path, config, client)
            if result:
                csv_paths.append(result)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    # Merge all CSVs
    logger.info("Merging all CSVs...")
    merged_df = merge_csv_files(csv_paths, config.output_path)

    # Save merged outputs
    all_csv_path = config.output_path / "all.csv"
    merged_df.to_csv(all_csv_path, index=False)
    logger.info(f"Saved: {all_csv_path} ({len(merged_df)} total exams)")

    all_xlsx_path = config.output_path / "all.xlsx"
    export_excel(merged_df, all_xlsx_path)
    logger.info(f"Saved: {all_xlsx_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Total exams extracted: {len(merged_df)}")
    if "exam_type" in merged_df.columns:
        type_counts = merged_df["exam_type"].value_counts()
        for exam_type, count in type_counts.items():
            logger.info(f"  - {exam_type}: {count}")


if __name__ == "__main__":
    main()
