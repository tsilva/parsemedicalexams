"""CLI entrypoint for the medical exams parser."""

import argparse
from pathlib import Path

from .config import (
    LEGACY_CONFIG_DIR,
    ProfileConfig,
    ensure_config_dir,
    get_env_example_path,
    migrate_env_file,
    migrate_profiles,
    sync_example_file,
)
from .pipeline import run_profile


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and summarize medical exam reports from PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  medicalexamsparser --profile tsilva              # Process all new PDFs
  medicalexamsparser --list-profiles               # List available profiles
  medicalexamsparser -p tsilva --regenerate        # Regenerate summaries only
  medicalexamsparser -p tsilva --resummarize       # Resummarize all documents
  medicalexamsparser -p tsilva --resummarize -d exam.pdf  # Resummarize one document
  medicalexamsparser -p tsilva --reprocess-all     # Force reprocess all documents
  medicalexamsparser -p tsilva -d exam.pdf         # Reprocess specific document
  medicalexamsparser -p tsilva --audit-outputs     # Audit existing outputs
  medicalexamsparser -p tsilva --dry-run           # Preview what would be processed
        """,
    )
    parser.add_argument("--profile", "-p", type=str, help="Profile name (without extension)")
    parser.add_argument(
        "--list-profiles", action="store_true", help="List available profiles and exit"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate summaries from existing transcription markdown files",
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
        "--model",
        "-m",
        type=str,
        help="Model ID for extraction (overrides the profile)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of parallel workers (overrides the profile)",
    )
    parser.add_argument(
        "--pattern", type=str, help="Regex pattern for input files (overrides profile)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Simulate the pipeline without making LLM calls or writing files. "
            "Reports what would be processed."
        ),
    )
    parser.add_argument(
        "--audit-outputs",
        action="store_true",
        help="Audit existing output bundles for blocking content issues and exit.",
    )
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    config_dir = ensure_config_dir()
    repo_root = Path(__file__).resolve().parent.parent
    moved_profiles = migrate_profiles(LEGACY_CONFIG_DIR)
    migrated_env = migrate_env_file(LEGACY_CONFIG_DIR, repo_root)
    sync_example_file(repo_root / ".env.example", get_env_example_path())

    if moved_profiles:
        print(f"Moved {len(moved_profiles)} profile file(s) into {config_dir}")
    if migrated_env:
        print(f"Moved shared .env into {migrated_env}")

    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()
        if profiles:
            print(f"Available profiles in {config_dir}:")
            for profile_name in profiles:
                print(f"  - {profile_name}")
        else:
            print(f"No profiles found in {config_dir}")
        return

    if args.profile:
        profiles_to_run = [args.profile]
    else:
        profiles_to_run = ProfileConfig.list_profiles()
        if not profiles_to_run:
            print(f"No profiles found in {config_dir}")
            print("Create a YAML or JSON profile there, then rerun or use --list-profiles.")
            raise SystemExit(1)
        print(f"Running all {len(profiles_to_run)} profiles: {', '.join(profiles_to_run)}")

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

    if len(profiles_to_run) > 1:
        print(f"\n{'=' * 60}")
        print("All Profiles Summary")
        print("=" * 60)
        print(f"Successful: {success_count}/{len(profiles_to_run)}")
        if failed_profiles:
            print(f"Failed profiles: {', '.join(failed_profiles)}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
