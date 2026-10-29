"""Keep the documented shared-folder convention compatible with PDF discovery."""

from pathlib import Path

from parsemedicalexams.config import ProfileConfig
from parsemedicalexams.pipeline import discover_pdf_files


def test_profile_template_loads_from_a_private_runtime_copy(tmp_path):
    template = Path(__file__).parents[1] / "profiles/template.yaml.example"
    copied = tmp_path / "example.yaml"
    copied.write_bytes(template.read_bytes())
    profile = ProfileConfig.from_file(copied)
    assert profile.input_path.is_absolute()
    assert profile.output_path.is_absolute()


def test_shared_folder_regex_excludes_only_a_leading_lab_label(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    names = [
        "Analises.PDF", "2025-01-02 - analises - painel.pdf",
        "2025-01-02 - ultrasound.pdf", "ultrasound.PDF",
        "analises-extra.pdf", "analises.txt", "analisesomething.pdf",
    ]
    for name in names:
        (nested / name).touch()
    pattern = r"^(?!(?:\d{4}-\d{2}-\d{2}\s*-\s*)?analises(?:\s|[-.]|$)).*\.pdf$"

    assert {p.name for p in discover_pdf_files(tmp_path, pattern)} == {
        names[2], names[3], names[6]
    }
    # Explicit requests retain the supported filter bypass.
    assert discover_pdf_files(tmp_path, pattern, names[0]) == [nested / names[0]]
