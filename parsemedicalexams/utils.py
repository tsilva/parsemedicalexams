"""Shared utility functions for the medical exams parser."""

import json
import logging
import re
from pathlib import Path
from typing import cast

from PIL import Image, ImageEnhance  # type: ignore[import-untyped]

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
logger = logging.getLogger(__name__)


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def preprocess_page_image(image: Image.Image) -> Image.Image:
    """Convert image to grayscale, resize, and enhance contrast."""
    gray_image = image.convert("L")
    MAX_LONG_SIDE = 1000
    long_side = max(gray_image.width, gray_image.height)
    if long_side > MAX_LONG_SIDE:
        ratio = MAX_LONG_SIDE / long_side
        new_width = int(gray_image.width * ratio)
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(gray_image).enhance(2.0)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_completion_text(completion: object, context: str) -> str:
    """Safely extract stripped text content from a chat completion."""
    if not completion:
        logger.error(f"Missing completion response for {context}")
        return ""

    choices = getattr(completion, "choices", None)
    if not choices:
        logger.error(f"Missing completion choices for {context}")
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        logger.warning(f"Missing completion message for {context}")
        return ""

    content = getattr(message, "content", None)
    if content is None:
        logger.warning(f"Missing completion content for {context}")
        return ""

    if not isinstance(content, str):
        logger.warning(f"Non-string completion content for {context}: {type(content).__name__}")
        return ""

    return content.strip()


def require_completion_text(completion: object, context: str) -> str:
    """Return completion text or raise when the response is empty or malformed."""
    content = extract_completion_text(completion, context)
    if not content:
        raise RuntimeError(f"Missing completion text for {context}")
    return content


def parse_json_mapping(text: str, context: str) -> dict[str, object]:
    """Parse a JSON object from model output."""
    raw = strip_markdown_fences(text)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(dict[str, object], parsed)


def extract_dates_from_text(text: str) -> list[str]:
    """Extract YYYY-MM-DD dates from ISO, DD/MM/YYYY, and DD-MM-YYYY input."""
    dates = []
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        year, month, day = match.groups()
        if 1900 <= int(year) <= 2100 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            dates.append(f"{year}-{month}-{day}")
    for match in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text):
        day, month, year = match.groups()
        day_int, month_int, year_int = int(day), int(month), int(year)
        if 1900 <= year_int <= 2100 and 1 <= month_int <= 12 and 1 <= day_int <= 31:
            dates.append(f"{year}-{month:0>2}-{day:0>2}")
    return dates


def setup_logging(log_dir: Path, clear_logs: bool = False) -> logging.Logger:
    """Configure file and console logging, optionally clearing existing logs."""
    log_dir.mkdir(exist_ok=True)
    info_log_path = log_dir / "info.log"
    error_log_path = log_dir / "error.log"

    if clear_logs:
        for log_file in (info_log_path, error_log_path):
            if log_file.exists():
                log_file.write_text("", encoding="utf-8")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    info_handler = logging.FileHandler(info_log_path, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_formatter)

    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(info_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)
