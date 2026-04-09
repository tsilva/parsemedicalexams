#!/usr/bin/env python3
"""
Scan extraction outputs for known sentinel failure patterns.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_PATTERNS = {
    "hard_failure": [
        r"Request too large\. Try with a smaller file\.",
    ],
    "model_narration": [
        r"\bThis image shows\b",
        r"\bThe following text elements are partially visible\b",
        r"\bThe remaining technical parameters\b",
        r"\bNo readable text is visible on this page\b",
        r"\bThe page consists of multiple medical ultrasound image frames\b",
    ],
    "placeholder_metadata": [
        r"\bUnknown Institution\b",
    ],
}


def compile_patterns(pattern_sets):
    return {
        label: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for label, patterns in pattern_sets.items()
    }


def scan_outputs(output_dir: Path, pattern_sets):
    compiled = compile_patterns(pattern_sets)
    results = defaultdict(list)

    for md_file in sorted(output_dir.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        for label, patterns in compiled.items():
            matches = []
            for pattern in patterns:
                for match in pattern.finditer(text):
                    snippet = match.group(0)
                    matches.append(snippet[:160])
            if matches:
                results[label].append(
                    {
                        "file": str(md_file),
                        "doc": md_file.parent.name,
                        "snippets": sorted(set(matches)),
                    }
                )
    return results


def print_report(results):
    if not results:
        print("No sentinel failures found.")
        return

    for label in sorted(results):
        entries = results[label]
        docs = sorted({entry["doc"] for entry in entries})
        print(f"[{label}] files={len(entries)} docs={len(docs)}")
        for entry in entries[:10]:
            print(f"  - {entry['file']}")
            for snippet in entry["snippets"][:3]:
                print(f"      snippet: {snippet}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory containing extraction outputs")
    parser.add_argument("--json-output", help="Optional JSON report path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    results = scan_outputs(output_dir, DEFAULT_PATTERNS)
    print_report(results)

    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
