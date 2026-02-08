"""CLI entry point for cenzura video text redaction tool."""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple

from cenzura.ocr import RapidOcrEngine
from cenzura.scanner import scan_video
from cenzura.redactor import redact_video


# Maps extension -> ffmpeg codec settings
SUPPORTED_FORMATS = {
    ".mp4": {"vcodec": "libx264", "preset": "medium", "crf": "23", "acodec": "copy"},
    ".mkv": {"vcodec": "libx264", "preset": "medium", "crf": "23", "acodec": "copy"},
    ".avi": {"vcodec": "libx264", "preset": "medium", "crf": "23", "acodec": "copy"},
}


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_video_extension(path: str, label: str) -> dict:
    """Validate file extension and return codec config. Raises SystemExit on invalid."""
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS)
        print(f"Error: unsupported {label} format '{ext}'. Supported: {supported}")
        sys.exit(1)
    return SUPPORTED_FORMATS[ext]


def parse_args() -> Tuple[str, List[str], str, dict, bool]:
    """
    Parse command line arguments.

    Returns:
        Tuple of (input_path, secrets, output_path, codec_config, verbose)
    """
    parser = argparse.ArgumentParser(
        description="Detect and redact specified text from videos using OCR.",
        epilog="""
cenzura scans videos for sensitive text (API keys, passwords, secrets) and produces
a redacted copy with those regions blacked out. It uses RapidOCR to detect text.
        """.strip(),
    )
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument(
        "secrets",
        help="Comma-separated list of secrets to redact (e.g., 'SECRET_KEY_123,password123')",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output video file (default: <input_stem>_redacted.<ext>)",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed timing statistics",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.isfile(args.input_video):
        print(f"Error: Input file '{args.input_video}' does not exist")
        sys.exit(1)

    # Validate input extension
    validate_video_extension(args.input_video, "input")

    # Parse secrets
    secrets = [s.strip() for s in args.secrets.split(",") if s.strip()]
    if not secrets:
        print("Error: No secrets provided")
        sys.exit(1)

    # Generate output path if not provided
    if args.output is None:
        input_file = Path(args.input_video)
        output_path = str(
            input_file.parent / f"{input_file.stem}_redacted{input_file.suffix}"
        )
    else:
        output_path = args.output

    # Validate output extension and get codec config
    codec_config = validate_video_extension(output_path, "output")

    return args.input_video, secrets, output_path, codec_config, args.verbose


def main():
    """Main entry point."""
    # Check ffmpeg dependency
    if not check_ffmpeg_available():
        print("Error: ffmpeg is not available. Please install ffmpeg.")
        sys.exit(1)

    # Parse arguments
    input_path, secrets, output_path, codec_config, verbose = parse_args()

    # Initialize OCR engine
    print("Initializing OCR engine...")
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        print(f"Using optimized config: {config_path}")
        ocr_engine = RapidOcrEngine(config_path=str(config_path))
    else:
        print("Warning: config.yaml not found, using defaults")
        ocr_engine = RapidOcrEngine()

    # Pass 1: Scan
    print(f"Scanning video for secrets: {', '.join(secrets)}")
    scan_result = scan_video(input_path, secrets, ocr_engine, verbose=verbose)

    if not scan_result.bbox_lookup:
        print("No occurrences found. No redaction needed.")
        print(f"Output file '{output_path}' will NOT be created.")
        sys.exit(0)

    # Calculate statistics
    num_frames_with_detections = len(scan_result.bbox_lookup)
    total_occurrences = sum(len(bboxes) for bboxes in scan_result.bbox_lookup.values())
    duration_with_detections = num_frames_with_detections / scan_result.fps

    print(
        f"Found {total_occurrences} occurrences across {duration_with_detections:.1f} seconds"
    )

    # Pass 2: Redact
    print(f"Redacting video...")
    redact_video(
        input_path,
        output_path,
        scan_result.bbox_lookup,
        scan_result.total_frames,
        scan_result.width,
        scan_result.height,
        scan_result.fps,
        codec_config,
    )

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
