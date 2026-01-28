#!/usr/bin/env python3
"""Generate document inventory for quality audit."""

import json
import re
import yaml
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("/Users/tsilva/Google Drive/My Drive/medicalexamsparser-tiago/")

def extract_frontmatter(file_path: Path) -> dict:
    """Extract YAML frontmatter from markdown file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                return yaml.safe_load(parts[1]) or {}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return {}

def get_era(date_str: str) -> str:
    """Determine era from date string."""
    if not date_str:
        return "unknown"
    year_match = re.search(r'(\d{4})', str(date_str))
    if year_match:
        year = int(year_match.group(1))
        if year < 2000:
            return "1990s"
        elif year < 2010:
            return "2000s"
        elif year < 2020:
            return "2010s"
        else:
            return "2020s"
    return "unknown"

def main():
    inventory = []
    doc_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name != 'logs']

    for doc_dir in sorted(doc_dirs):
        doc_stem = doc_dir.name
        md_files = list(doc_dir.glob("*.md"))

        # Separate page files from summary
        page_files = [f for f in md_files if not f.stem.endswith('.summary')]
        summary_files = [f for f in md_files if f.stem.endswith('.summary')]

        page_count = len(page_files)
        has_summary = len(summary_files) > 0

        # Extract metadata from first page
        metadata = {}
        dates = set()
        categories = set()
        doctors = set()
        facilities = set()

        # Check for empty content pages
        empty_pages = []

        for pf in page_files:
            fm = extract_frontmatter(pf)
            if fm:
                if fm.get('date'):
                    dates.add(str(fm['date']))
                if fm.get('category'):
                    categories.add(fm['category'])
                if fm.get('doctor'):
                    doctors.add(fm['doctor'])
                if fm.get('facility'):
                    facilities.add(fm['facility'])

                # Check content (after frontmatter)
                content = pf.read_text(encoding='utf-8')
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
                    if len(body) < 10:  # Very short/empty content
                        empty_pages.append(pf.name)

        # Get primary date from first page
        if page_files:
            first_fm = extract_frontmatter(sorted(page_files)[0])
            primary_date = first_fm.get('date', '')
            primary_category = first_fm.get('category', 'unknown')
            exam_name_raw = first_fm.get('exam_name_raw', '')
            title = first_fm.get('title', '')
        else:
            primary_date = ''
            primary_category = 'unknown'
            exam_name_raw = ''
            title = ''

        era = get_era(primary_date)

        # Build tags
        tags = []

        # Era tags
        if era in ['1990s', '2000s']:
            tags.append('OLD_DOC')

        # Complexity tags
        if page_count >= 10:
            tags.append('COMPLEX')
        elif page_count >= 3:
            tags.append('MULTI_PAGE')

        # Category tag
        tags.append(f'CAT_{primary_category.upper()}')

        # Empty content tag
        if empty_pages:
            tags.append('HAS_EMPTY_PAGES')

        # Check for date mismatch between folder and content
        folder_date_match = re.match(r'^(\d{4}-\d{2}-\d{2})', doc_stem)
        if folder_date_match and primary_date:
            folder_date = folder_date_match.group(1)
            if str(primary_date) != folder_date:
                tags.append('DATE_MISMATCH')

        doc_entry = {
            'profile': 'tiago',
            'doc_stem': doc_stem,
            'doc_dir': str(doc_dir),
            'page_count': page_count,
            'has_summary': has_summary,
            'date': str(primary_date) if primary_date else None,
            'era': era,
            'category': primary_category,
            'exam_name_raw': exam_name_raw,
            'title': title,
            'doctors': list(doctors),
            'facilities': list(facilities),
            'tags': tags,
            'empty_pages': empty_pages,
        }

        inventory.append(doc_entry)

    # Save inventory
    output_path = Path("/Users/tsilva/repos/tsilva/medical-exams-parser/audit_inventory.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"INVENTORY SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents: {len(inventory)}")
    print(f"Total pages: {sum(d['page_count'] for d in inventory)}")

    # By era
    era_counts = defaultdict(int)
    for d in inventory:
        era_counts[d['era']] += 1
    print(f"\nBy era:")
    for era, count in sorted(era_counts.items()):
        print(f"  {era}: {count}")

    # By category
    cat_counts = defaultdict(int)
    for d in inventory:
        cat_counts[d['category']] += 1
    print(f"\nBy category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # By tags
    tag_counts = defaultdict(int)
    for d in inventory:
        for tag in d['tags']:
            tag_counts[tag] += 1
    print(f"\nBy priority tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    # Documents with issues
    docs_with_empty = [d for d in inventory if d['empty_pages']]
    docs_with_mismatch = [d for d in inventory if 'DATE_MISMATCH' in d['tags']]

    print(f"\n{'='*60}")
    print(f"POTENTIAL ISSUES")
    print(f"{'='*60}")
    print(f"Documents with empty pages: {len(docs_with_empty)}")
    for d in docs_with_empty:
        print(f"  - {d['doc_stem']}: {d['empty_pages']}")

    print(f"\nDocuments with date mismatch: {len(docs_with_mismatch)}")
    for d in docs_with_mismatch:
        print(f"  - {d['doc_stem']}: folder vs content date '{d['date']}'")

    print(f"\nInventory saved to: {output_path}")

if __name__ == '__main__':
    main()
