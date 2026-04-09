import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsemedicalexams.validation import (
    build_no_readable_text_marker,
    build_non_discrete_chart_marker,
    determine_page_strategy,
    first_blocking_issue,
    validate_page_output,
    validate_summary_output,
)


def test_validate_page_output_detects_hard_failure():
    issues = validate_page_output("Request too large. Try with a smaller file.")

    assert first_blocking_issue(issues).kind == "hard_failure"


def test_validate_page_output_detects_model_narration():
    issues = validate_page_output(
        "No readable text is visible on this page. The page consists of multiple medical ultrasound image frames."
    )

    assert first_blocking_issue(issues).kind == "model_narration"


def test_validate_page_output_detects_alternate_ultrasound_narration():
    issues = validate_page_output(
        "The page consists of ultrasound/scan images with small white measurement text that is too low-resolution to read accurately."
    )

    assert first_blocking_issue(issues).kind == "model_narration"


def test_validate_page_output_detects_no_readable_text_narration():
    issues = validate_page_output(
        "[illegible]\n\n(There is no clearly readable text on this page image; all visible annotations are too small/blurred to transcribe exactly.)"
    )

    assert first_blocking_issue(issues).kind == "model_narration"


def test_validate_page_output_detects_ultrasound_image_only_narration():
    issues = validate_page_output(
        "[Image contains ultrasound scans only; no readable text is clearly visible.]"
    )

    assert first_blocking_issue(issues).kind == "model_narration"


def test_validate_page_output_treats_unknown_institution_as_warning():
    issues = validate_page_output("Unknown Institution\nPolysomnography Report")

    assert first_blocking_issue(issues) is None
    assert any(issue.kind == "placeholder_metadata" for issue in issues)


def test_validate_page_output_rejects_scaffold_only_audiogram():
    issues = validate_page_output(
        "Ouvido Direito\nHearing Level (dB HL)\n125 250 500 1K 2K 4K 8K\nAudiograma Vocal",
        page_kind="chart",
        chart_type="audiogram",
    )

    assert first_blocking_issue(issues).kind == "chart_scaffold_only"


def test_validate_page_output_accepts_structured_audiogram():
    text = """[AUDIOGRAM]
Right ear thresholds (dB HL):
- 125 Hz: unreadable
- 250 Hz: 20
- 500 Hz: 10
- 1K Hz: 10
- 2K Hz: 10
- 4K Hz: 25
- 8K Hz: 10
Left ear thresholds (dB HL):
- 125 Hz: unreadable
- 250 Hz: 10
- 500 Hz: 10
- 1K Hz: 10
- 2K Hz: 10
- 4K Hz: 15
- 8K Hz: 10
Speech audiogram:
- Right ear SRT: 10
- Left ear SRT: 10
- Right ear discrimination: 100
- Left ear discrimination: 90
"""

    issues = validate_page_output(text, page_kind="chart", chart_type="audiogram")

    assert first_blocking_issue(issues) is None


def test_validate_page_output_requires_sleep_chart_marker():
    issues = validate_page_output(
        "Summary Graph\nArousal\nSpO2\nHeart Rate",
        page_kind="chart",
        chart_type="sleep_summary_graph",
    )

    assert first_blocking_issue(issues).kind == "chart_scaffold_only"


def test_validate_page_output_accepts_image_only_marker():
    issues = validate_page_output(
        build_no_readable_text_marker(),
        page_kind="image_only",
    )

    assert first_blocking_issue(issues) is None


def test_validate_page_output_rejects_illegible_only_body():
    issues = validate_page_output(
        "\n\n".join(["[illegible] [illegible] [illegible] [illegible]"] * 4)
    )

    assert first_blocking_issue(issues).kind == "illegible_only"


def test_validate_page_output_rejects_short_illegible_only_body():
    issues = validate_page_output("[illegible]\n\n[illegible]")

    assert first_blocking_issue(issues).kind == "illegible_only"


def test_validate_page_output_accepts_structured_tympanometry():
    text = """[TYMPANOMETRY]
Right ear:
- Probe tone: 226 Hz
- Volume: 2,11 ml
- Pressure: -16 daPa
- Compliance: 1,01 ml
- Gradient: - ml
Left ear:
- Probe tone: 226 Hz
- Volume: 1,87 ml
- Pressure: -5 daPa
- Compliance: 0,79 ml
- Gradient: - ml
Audiologist: Maria Joao Lopes
"""

    issues = validate_page_output(text, page_kind="chart", chart_type="tympanometry")

    assert first_blocking_issue(issues) is None


def test_validate_page_output_requires_eeg_signal_marker():
    issues = validate_page_output(
        "Echo: ECG\nPatient: Rachael\nComputer analysis",
        page_kind="chart",
        chart_type="eeg_signal_trace",
    )

    assert first_blocking_issue(issues).kind == "signal_trace_requires_marker"


def test_validate_summary_output_detects_hard_failure():
    issues = validate_summary_output("Request too large. Try with a smaller file.")

    assert first_blocking_issue(issues).kind == "hard_failure"


def test_determine_page_strategy_prefers_embedded_text_for_digital_page():
    embedded_text = "Polysomnography Report\n" + ("Total Recording Time 492 minutes\n" * 40)

    assert determine_page_strategy(embedded_text) == ("text", None, "embedded_text")


def test_determine_page_strategy_routes_audiogram_chart():
    embedded_text = "Ouvido Direito\nOuvido Esquerdo\nAudiograma Vocal\nClínica Lusíadas Gaia"

    assert determine_page_strategy(embedded_text) == ("chart", "audiogram", "hybrid")


def test_determine_page_strategy_routes_tympanometry_chart():
    embedded_text = (
        "Tímpano 226 Hz Direito\nTímpano 226 Hz Esquerdo\n"
        "Volume: 2,11 ml\nPressão: -16 daPa\nComplacência: 1,01 ml"
    )

    assert determine_page_strategy(embedded_text) == ("chart", "tympanometry", "hybrid")


def test_determine_page_strategy_routes_empty_eeg_page_to_signal_trace():
    assert determine_page_strategy("", document_exam_name="Electroencefalografia (EEG)") == (
        "chart",
        "eeg_signal_trace",
        "hybrid",
    )


def test_build_non_discrete_chart_marker_is_valid():
    text = build_non_discrete_chart_marker("sleep_summary_graph", ["Arousal", "SpO2"])

    issues = validate_page_output(
        text,
        page_kind="chart",
        chart_type="sleep_summary_graph",
    )

    assert first_blocking_issue(issues) is None


def test_build_non_discrete_eeg_marker_is_valid():
    text = build_non_discrete_chart_marker(
        "eeg_signal_trace",
        ["FP2 - FZ", "Post HV 60 Sec"],
    )

    issues = validate_page_output(
        text,
        page_kind="chart",
        chart_type="eeg_signal_trace",
    )

    assert first_blocking_issue(issues) is None
