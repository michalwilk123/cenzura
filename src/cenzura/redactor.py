"""Video redaction functionality (Pass 2)."""

import subprocess
from typing import Dict, List, Tuple
import numpy as np
import cv2
from tqdm import tqdm


# Default codec settings for output video
DEFAULT_CODEC = {
    "vcodec": "libx264",
    "preset": "medium",
    "crf": "23",
    "acodec": "copy",
}


def redact_video(
    input_path: str,
    output_path: str,
    bbox_lookup: Dict[int, List[Tuple[int, int, int, int]]],
    total_frames: int,
    width: int,
    height: int,
    fps: float,
    codec_config: dict | None = None,
):
    """
    Redact video by drawing black rectangles over detected text.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        bbox_lookup: Dictionary mapping frame index to list of (x, y, w, h) bounding boxes
        total_frames: Total number of frames in video
        width: Video frame width
        height: Video frame height
        fps: Video frames per second
        codec_config: Optional codec settings dict (uses DEFAULT_CODEC if None)

    Raises:
        ValueError: If FFmpeg processes fail
    """
    if codec_config is None:
        codec_config = DEFAULT_CODEC

    # FFmpeg decoder: extract raw BGR24 frames
    decoder_cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    # FFmpeg encoder: encode from raw BGR24, copy audio
    encoder_cmd = [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        codec_config["vcodec"],
        "-preset",
        codec_config["preset"],
        "-crf",
        codec_config["crf"],
        "-c:a",
        codec_config["acodec"],
        "-y",
        output_path,
    ]

    decoder = subprocess.Popen(
        decoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    encoder = subprocess.Popen(
        encoder_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    frame_size = width * height * 3
    frame_idx = 0
    redacted_count = 0

    try:
        with tqdm(
            total=total_frames, desc="Redacting", unit="frame", mininterval=0.1
        ) as pbar:
            while frame_idx < total_frames:
                # Read frame data
                frame_data = decoder.stdout.read(frame_size)
                if len(frame_data) != frame_size:
                    break

                # Check if this frame needs redaction
                if frame_idx in bbox_lookup:
                    # Convert bytes to numpy array (make it writable by copying)
                    frame = (
                        np.frombuffer(frame_data, dtype=np.uint8)
                        .reshape((height, width, 3))
                        .copy()
                    )

                    # Draw black rectangles
                    for x, y, w, h in bbox_lookup[frame_idx]:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

                    # Convert back to bytes
                    frame_data = frame.tobytes()
                    redacted_count += 1

                # Write to encoder
                encoder.stdin.write(frame_data)
                frame_idx += 1
                pbar.update(1)

        print(f"Redacted {redacted_count} frames total")

    finally:
        encoder.stdin.close()
        encoder.wait()
        decoder.stdout.close()
        decoder.wait()
