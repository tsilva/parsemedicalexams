#!/usr/bin/env python3
"""
Apply stratified sampling to select representative documents for quality investigation.
Configurable priority areas and sampling criteria.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional


def load_inventory(inventory_file: Path) -> List[Dict]:
    """Load the document inventory."""
    with open(inventory_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_sampling_config(config_file: Optional[Path]) -> Dict:
    """Load sampling configuration or return defaults."""
    if config_file is not None:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Default configuration
    return {
        'priority_tags': {
            'LOW_CONF': {'coverage': 1.0, 'priority': 1000},
            'COMPLEX': {'coverage': 1.0, 'priority': 100},
            'OLD_DOC': {'coverage': 0.7, 'priority': 10},
            'MULTI_PAGE': {'coverage': 0.4, 'priority': 1},
        },
        'category_balance': {
            'imaging': 3,
            'ultrasound': 3,
            'endoscopy': 3,
            'other': 2,
        },
        'random_seed': 42
    }


def document_key(doc):
    """A document stem is unique only inside its explicitly supplied profile."""
    if not isinstance(doc, dict) or any(not isinstance(doc.get(key), str) or not doc[key] for key in ('profile', 'doc_stem')):
        raise ValueError('Inventory entries need nonempty profile and doc_stem fields')
    return (doc['profile'], doc['doc_stem'])


def mandatory_documents(inventory, sentinel_report):
    """Match the existing sentinel JSON by document directory; never open source documents."""
    if not isinstance(inventory, list):
        raise ValueError('Inventory must be a list')
    if not isinstance(sentinel_report, dict):
        raise ValueError('Sentinel report must be an object of match groups')
    entries = []
    for matches in sentinel_report.values():
        if not isinstance(matches, list):
            raise ValueError('Sentinel match groups must be lists')
        entries.extend(matches)
    if not entries:
        return set()
    directories = {}
    for doc in inventory:
        key = document_key(doc)
        if not isinstance(doc.get('doc_dir'), str) or not doc['doc_dir']:
            raise ValueError('Inventory doc_dir is required for matching a sentinel report')
        directories.setdefault(str(Path(doc['doc_dir']).resolve()), []).append(key)
    required = set()
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get('file'), str) or not entry['file']
                or not isinstance(entry.get('doc'), str) or Path(entry['file']).parent.name != entry['doc']):
            raise ValueError('Sentinel report has an unsupported document identity')
        matches = directories.get(str(Path(entry['file']).parent.resolve()), [])
        if len(matches) != 1:
            raise ValueError('Sentinel document is absent or ambiguous in the selected inventory')
        required.add(matches[0])
    return required


def protect_inputs(output, inputs):
    for source in (item for item in inputs if item is not None):
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError('Output must not overwrite supplied audit evidence')


def known_number(value, maximum):
    return value if (isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(value) and 0 <= value <= maximum) else None


def known_confidence(doc):
    return known_number(doc.get('min_confidence'), 1)


def stratified_sample(
    inventory: List[Dict],
    target_size: int,
    config: Dict,
    required_keys=()
) -> List[Dict]:
    """
    Apply stratified sampling based on priority criteria.

    Args:
        inventory: List of document entries
        target_size: Target sample size
        config: Sampling configuration

    Returns:
        List of sampled documents with sample_reason field
    """
    if not isinstance(inventory, list):
        raise ValueError('Inventory must be a list')
    keys = [document_key(doc) for doc in inventory]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate profile/document identities in inventory')
    if type(target_size) is not int or target_size < 0:
        raise ValueError('Target size must be a nonnegative integer')
    required_keys = set(required_keys)
    if not required_keys.issubset(keys):
        raise ValueError('Required documents are missing from the selected inventory')
    sample = []
    used_keys = set()
    priority_tags = config.get('priority_tags', {})
    category_balance = config.get('category_balance', {})

    def add_to_sample(docs: List[Dict], count: int, label: str):
        """Add documents to sample, avoiding duplicates."""
        available = [d for d in docs if document_key(d) not in used_keys]
        if not available:
            return []

        # Calculate actual count based on coverage
        actual_count = min(count, len(available))

        selected = random.sample(available, actual_count)
        for doc in selected:
            sample.append({**doc, 'sample_reason': label})
            used_keys.add(document_key(doc))
        print(f"  {label}: selected {len(selected)} documents")
        return selected

    print("\nStratified Sampling Process:")
    print("=" * 60)

    # Known failures are mandatory, even when they exceed the nominal target.
    required = [doc for doc in inventory if document_key(doc) in required_keys]
    add_to_sample(required, len(required), 'SENTINEL (all)')

    # Phase 1: Priority tags with specified coverage
    for tag, tag_config in sorted(
        priority_tags.items(),
        key=lambda x: -x[1].get('priority', 0)
    ):
        coverage = tag_config.get('coverage', 1.0)
        docs_with_tag = [d for d in inventory if tag in d.get('tags', [])]

        if docs_with_tag:
            count = int(len(docs_with_tag) * coverage)
            if coverage == 1.0:
                add_to_sample(docs_with_tag, count, f"{tag} (all)")
            else:
                add_to_sample(docs_with_tag, count, f"{tag} ({coverage*100:.0f}%)")

    # Track remaining target
    remaining = target_size - len(sample)
    print(f"\n  Currently selected: {len(sample)} documents")
    print(f"  Remaining target: {remaining} documents")

    # Phase 2: Ensure category diversity
    if category_balance and remaining > 0:
        for category, count in category_balance.items():
            cat_docs = [d for d in inventory if d.get('category') == category]
            if cat_docs:
                add_to_sample(cat_docs, count, f"CATEGORY_{category.upper()}")

        remaining = target_size - len(sample)
        print(f"\n  After category balancing: {len(sample)} documents")
        print(f"  Remaining target: {remaining} documents")

    # Phase 3: Ensure era diversity
    if remaining > 0:
        # Try to include some recent documents for comparison
        recent_docs = [d for d in inventory if d.get('era') == '2020s']
        if recent_docs:
            add_to_sample(recent_docs, min(3, remaining), "ERA_2020s (recent)")

        remaining = target_size - len(sample)

    # Phase 4: High confidence baseline
    if remaining > 0:
        high_conf = [
            d for d in inventory
            if known_confidence(d) is not None and known_confidence(d) >= 0.9
        ]
        if high_conf:
            add_to_sample(high_conf, min(3, remaining), "HIGH_CONF (baseline)")

        remaining = target_size - len(sample)

    # Phase 5: Random fill to reach target size
    if remaining > 0:
        add_to_sample(inventory, remaining, "RANDOM_FILL")

    print(f"\n  Final sample size: {len(sample)} documents")

    return sample


def print_sample_summary(sample: List[Dict], inventory: List[Dict]):
    """Print summary statistics about the sample."""
    print("\n" + "=" * 60)
    print("SAMPLE SUMMARY")
    print("=" * 60)

    # Profile distribution
    profiles = {}
    for doc in sample:
        profile = doc.get('profile', 'unknown')
        profiles[profile] = profiles.get(profile, 0) + 1

    if len(profiles) > 1:
        print(f"\nProfile Distribution:")
        for profile, count in sorted(profiles.items()):
            pct = count / len(sample) * 100
            print(f"  {profile}: {count} documents ({pct:.1f}%)")

    # Tag distribution in sample
    tag_counts = {}
    for doc in sample:
        for tag in doc.get('tags', []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\nPriority Tags in Sample:")
    priority_tags = ['LOW_CONF', 'COMPLEX', 'OLD_DOC', 'MULTI_PAGE']
    for tag in priority_tags:
        count = tag_counts.get(tag, 0)
        total = len([d for d in inventory if tag in d.get('tags', [])])
        if total > 0:
            coverage = count / total * 100
            print(f"  {tag}: {count} documents ({coverage:.0f}% of {total})")

    # Category distribution
    cat_counts = {}
    for doc in sample:
        cat = doc.get('category', 'unknown')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"\nCategory Distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} documents")

    # Era distribution
    era_counts = {}
    for doc in sample:
        era = doc.get('era', 'unknown')
        era_counts[era] = era_counts.get(era, 0) + 1
    print(f"\nEra Distribution:")
    for era, count in sorted(era_counts.items()):
        print(f"  {era}: {count} documents")

    # Confidence distribution
    confidence = [known_confidence(doc) for doc in sample]
    low = sum(value is not None and value < 0.7 for value in confidence)
    med = sum(value is not None and 0.7 <= value < 0.9 for value in confidence)
    high = sum(value is not None and value >= 0.9 for value in confidence)
    unknown = sum(value is None for value in confidence)
    print(f"\nConfidence Distribution:")
    print(f"  High (≥0.9): {high} documents")
    print(f"  Medium (0.7-0.9): {med} documents")
    print(f"  Low (<0.7): {low} documents")
    print(f"  Unknown/invalid: {unknown} documents")

    # Complexity distribution
    simple = len([d for d in sample if d['page_count'] <= 2])
    multi = len([d for d in sample if 3 <= d['page_count'] < 10])
    complex_docs = len([d for d in sample if d['page_count'] >= 10])
    print(f"\nComplexity Distribution:")
    print(f"  Simple (1-2 pages): {simple} documents")
    print(f"  Multi-page (3-9 pages): {multi} documents")
    print(f"  Complex (10+ pages): {complex_docs} documents")


def print_sample_list(sample: List[Dict]):
    """Print detailed list of selected documents."""
    print("\n" + "=" * 60)
    print("SELECTED DOCUMENTS")
    print("=" * 60)

    for i, doc in enumerate(sample, 1):
        tags_str = ', '.join(doc.get('tags', []))
        confidence = known_confidence(doc)
        conf = f'{confidence:.2f}' if confidence is not None else 'unknown'
        print(f"\n{i}. {doc['doc_stem'][:60]}")
        print(f"   Profile: {doc['profile']} | Pages: {doc['page_count']} | Conf: {conf}")
        print(f"   Category: {doc.get('category', 'unknown')} | Era: {doc.get('era', 'unknown')}")
        print(f"   Tags: {tags_str}")
        print(f"   Reason: {doc.get('sample_reason', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate stratified sample for quality investigation'
    )
    parser.add_argument(
        '--inventory',
        type=Path,
        default=Path('inventory.json'),
        help='Input inventory file'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=30,
        help='Target sample size'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Sampling configuration file (JSON)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('sample.json'),
        help='Output filename for sample'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='Print detailed list of selected documents'
    )

    parser.add_argument('--sentinel-report', type=Path,
                        help='Existing sentinel_scan.py JSON; every matched document is mandatory')
    args = parser.parse_args()

    try:
        protect_inputs(args.output, [args.inventory, args.config, args.sentinel_report])
        inventory = load_inventory(args.inventory)
        config = load_sampling_config(args.config)
        required = set()
        if args.sentinel_report:
            with args.sentinel_report.open(encoding='utf-8') as handle:
                required = mandatory_documents(inventory, json.load(handle))
    except (OSError, ValueError, TypeError, KeyError):
        parser.error('Invalid audit inputs or unsafe output path; inspect the selected files without exposing document contents')
    print(f"Loaded inventory: {len(inventory)} documents")

    # Set random seed
    seed = args.seed if args.seed is not None else config.get('random_seed', 42)
    random.seed(seed)
    print(f"Using random seed: {seed}")

    # Generate stratified sample
    try:
        sample = stratified_sample(inventory, args.target_size, config, required)
    except (ValueError, TypeError, KeyError):
        parser.error('Invalid inventory or sampling configuration; no sample has been written')

    # Sort by priority tags and profile
    def sort_key(doc):
        priority = 10000 if doc.get('sample_reason') == 'SENTINEL (all)' else 0
        if 'LOW_CONF' in doc.get('tags', []):
            priority += 1000
        if 'COMPLEX' in doc.get('tags', []):
            priority += 100
        if 'OLD_DOC' in doc.get('tags', []):
            priority += 10
        if 'MULTI_PAGE' in doc.get('tags', []):
            priority += 1
        return (-priority, doc['profile'], doc['doc_stem'])

    sample.sort(key=sort_key)

    # Recheck evidence inputs before writing the selected sample.
    try:
        protect_inputs(args.output, [args.inventory, args.config, args.sentinel_report])
    except (OSError, ValueError):
        parser.error('Sample output collides with audit evidence; no sample has been written')
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Sample saved to {args.output}")

    # Print summary statistics
    print_sample_summary(sample, inventory)

    # Print detailed list if requested
    if args.list:
        print_sample_list(sample)


if __name__ == '__main__':
    main()
