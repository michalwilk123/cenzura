"""Integration tests for video scanning and redaction."""

from pathlib import Path
import pytest
from cenzura import scan_video, redact_video, RapidOcrEngine


@pytest.fixture(scope="module")
def ocr_engine():
    """Create a shared OCR engine instance for all tests."""
    # Use config.yaml if it exists in project root
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        return RapidOcrEngine(str(config_path))
    return RapidOcrEngine()


@pytest.fixture(scope="module")
def test_video_path():
    """Return path to test_input.mp4 relative to project root."""
    return str(Path(__file__).parent.parent / "test_input.mp4")


def test_secret_found_produces_redacted_video(ocr_engine, test_video_path, tmp_path):
    """Test that searching for SECRET_KEY_123 finds matches and produces redacted video."""
    # Scan for the secret that exists in the video
    scan_result = scan_video(test_video_path, ["SECRET_KEY_123"], ocr_engine)

    # Assert that matches were found
    assert scan_result.bbox_lookup, "Expected to find SECRET_KEY_123 in video"
    assert len(scan_result.bbox_lookup) > 0, "Expected at least one frame with detections"

    # Create redacted video
    output_path = tmp_path / "redacted_output.mp4"
    redact_video(
        test_video_path,
        str(output_path),
        scan_result.bbox_lookup,
        scan_result.total_frames,
        scan_result.width,
        scan_result.height,
        scan_result.fps,
    )

    # Assert output file exists and has content
    assert output_path.exists(), "Redacted video should be created"
    assert output_path.stat().st_size > 0, "Redacted video should have non-zero size"


def test_nonexistent_secret_produces_no_video(ocr_engine, test_video_path):
    """Test that searching for non-existent secret finds nothing."""
    # Scan for a secret that doesn't exist in the video
    scan_result = scan_video(test_video_path, ["THIS_DOES_NOT_EXIST"], ocr_engine)

    # Assert that no matches were found
    assert not scan_result.bbox_lookup, "Expected no matches for non-existent secret"
    assert len(scan_result.bbox_lookup) == 0, "bbox_lookup should be empty"

    # No redact_video call - matching CLI behavior where no output is created when no matches
