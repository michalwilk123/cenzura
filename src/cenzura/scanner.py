"""Video scanning for text detection (Pass 1)."""

import time
from typing import List, Dict, Tuple, NamedTuple
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from thefuzz import fuzz
from tqdm import tqdm

from cenzura.ocr import OcrEngine


class ScanResult(NamedTuple):
    """Result from video scanning."""

    bbox_lookup: Dict[int, List[Tuple[int, int, int, int]]]
    total_frames: int
    width: int
    height: int
    fps: float


def text_matches_secret(text: str, secrets: List[str]) -> bool:
    """
    Check if text contains or fuzzy matches any secret.

    Args:
        text: Text to check
        secrets: List of secret strings to match against

    Returns:
        True if text matches any secret
    """
    text_lower = text.lower()

    for secret in secrets:
        secret_lower = secret.lower()

        # Case-insensitive substring match
        if secret_lower in text_lower:
            return True

        # Fuzzy match with partial_ratio >= 85
        if fuzz.partial_ratio(text_lower, secret_lower) >= 85:
            return True

    return False


def has_substantial_change(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold_pct: float = 1.0,
    pixel_thresh: int = 25,
) -> bool:
    """
    Check if there's substantial change between two frames.

    Args:
        prev_frame: Previous frame (BGR)
        curr_frame: Current frame (BGR)
        threshold_pct: Percentage of pixels that must change (default 1.0%)
        pixel_thresh: Pixel difference threshold to consider changed (default 25)

    Returns:
        True if change exceeds threshold
    """
    diff = cv2.absdiff(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY),
    )
    changed_pixels = np.count_nonzero(diff > pixel_thresh)
    total_pixels = diff.shape[0] * diff.shape[1]
    pct_changed = (changed_pixels / total_pixels) * 100
    return pct_changed > threshold_pct


def scan_video(
    video_path: str, secrets: List[str], ocr_engine: OcrEngine, verbose: bool = False
) -> ScanResult:
    """
    Scan video for text occurrences.

    Args:
        video_path: Path to input video file
        secrets: List of secret strings to detect
        ocr_engine: OCR engine instance to use for text detection
        verbose: Whether to print detailed timing statistics

    Returns:
        ScanResult with bbox lookup and video metadata

    Raises:
        ValueError: If video cannot be opened or is invalid
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_interval = round(fps)  # Sample at ~1 fps
    detections = []  # List of (frame_index, timestamp_sec, bboxes)

    # Timing stats
    total_read_time = 0
    total_ocr_time = 0
    total_match_time = 0
    total_bbox_time = 0
    total_diff_time = 0
    ocr_calls = 0
    match_calls = 0
    diff_calls = 0

    # Frame diff tracking
    last_ocr_frame = None

    frame_idx = 0
    scan_start = time.time()

    # Phase A: Read all frames + submit OCR tasks in background
    ocr_tasks = []  # List of (frame_idx, future) tuples

    with ThreadPoolExecutor(max_workers=1) as executor:
        with tqdm(
            total=total_frames, desc="Scanning", unit="frame", mininterval=0.1
        ) as pbar:
            while True:
                read_start = time.time()
                ret, frame = cap.read()
                total_read_time += time.time() - read_start

                if not ret:
                    break

                pbar.update(1)

                # Sample every N frames
                if frame_idx % sample_interval == 0:
                    should_run_ocr = False

                    # First sampled frame: always run OCR
                    if last_ocr_frame is None:
                        should_run_ocr = True
                    else:
                        # Check if frame has changed substantially
                        diff_start = time.time()
                        has_change = has_substantial_change(last_ocr_frame, frame)
                        total_diff_time += time.time() - diff_start
                        diff_calls += 1

                        if has_change:
                            should_run_ocr = True

                    if should_run_ocr:
                        # Submit OCR to background thread (don't block)
                        future = executor.submit(ocr_engine.detect, frame.copy())
                        ocr_tasks.append((frame_idx, future))
                        ocr_calls += 1

                        # Update last OCR frame (for diffing purposes)
                        last_ocr_frame = frame.copy()

                frame_idx += 1

        # Phase B: Collect OCR results and build detections
        print("\nProcessing OCR results...")
        for task_frame_idx, future in ocr_tasks:
            ocr_start = time.time()
            result = future.result()
            total_ocr_time += time.time() - ocr_start

            if result:
                frame_bboxes = []
                for x, y, w, h, text in result:
                    match_start = time.time()
                    matches = text_matches_secret(text, secrets)
                    total_match_time += time.time() - match_start
                    match_calls += 1

                    if matches:
                        # Apply padding to bbox
                        bbox_start = time.time()
                        x_min = max(0, x - 10)
                        y_min = max(0, y - 10)
                        x_max = min(width, x + w + 10)
                        y_max = min(height, y + h + 10)

                        padded_w = x_max - x_min
                        padded_h = y_max - y_min

                        frame_bboxes.append((x_min, y_min, padded_w, padded_h))
                        total_bbox_time += time.time() - bbox_start

                if frame_bboxes:
                    timestamp = task_frame_idx / fps
                    detections.append((task_frame_idx, timestamp, frame_bboxes))
                elif detections:
                    # Frame changed but no secrets found - add stop boundary
                    timestamp = task_frame_idx / fps
                    detections.append((task_frame_idx, timestamp, []))
            else:
                # No text detected at all
                if detections:
                    # Add stop boundary
                    timestamp = task_frame_idx / fps
                    detections.append((task_frame_idx, timestamp, []))

    cap.release()
    scan_total = time.time() - scan_start

    # Calculate total sampled frames
    total_sampled = (frame_idx + sample_interval - 1) // sample_interval

    # Print timing stats if verbose
    if verbose:
        print(f"\n=== TIMING STATISTICS ===")
        print(f"Total scan time: {scan_total:.2f}s")
        print(
            f"Frame reading: {total_read_time:.2f}s ({total_read_time/scan_total*100:.1f}%)"
        )
        if ocr_calls > 0:
            print(
                f"OCR processing: {total_ocr_time:.2f}s ({total_ocr_time/scan_total*100:.1f}%) - "
                f"{ocr_calls} calls, {total_ocr_time/ocr_calls:.3f}s avg per call"
            )
        else:
            print(f"OCR processing: {total_ocr_time:.2f}s (0.0%) - 0 calls")
        if diff_calls > 0:
            print(
                f"Frame diffing: {total_diff_time:.4f}s ({total_diff_time/scan_total*100:.1f}%) - "
                f"{diff_calls} comparisons, {total_diff_time/diff_calls*1000:.1f}ms avg"
            )
        else:
            print(f"Frame diffing: 0.0000s (0.0%) - 0 comparisons")
        print(
            f"Text matching: {total_match_time:.4f}s ({total_match_time/scan_total*100:.1f}%) - "
            f"{match_calls} calls"
        )
        print(
            f"Bbox calculation: {total_bbox_time:.4f}s ({total_bbox_time/scan_total*100:.1f}%)"
        )
        other = (
            scan_total
            - total_read_time
            - total_ocr_time
            - total_match_time
            - total_bbox_time
            - total_diff_time
        )
        print(f"Other overhead: {other:.2f}s")
        print(f"\nOCR scans: {ocr_calls}/{total_sampled} sampled frames")
        print()

    # Interpolate between detected frames
    bbox_lookup = {}
    if detections:
        print(f"\nInterpolating detections: {len(detections)} detection points")
        for i in range(len(detections)):
            start_frame, _, start_bboxes = detections[i]

            if i + 1 < len(detections):
                end_frame, _, _ = detections[i + 1]
            else:
                # For the last detection, extend to end of video
                end_frame = total_frames - 1

            print(
                f"  Detection {i+1}: frames {start_frame} to {end_frame} "
                f"({end_frame - start_frame + 1} frames)"
            )

            # Apply bboxes to all frames in range
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num not in bbox_lookup:
                    bbox_lookup[frame_num] = []
                bbox_lookup[frame_num].extend(start_bboxes)

    print(f"Total frames with redactions: {len(bbox_lookup)}")
    return ScanResult(bbox_lookup, total_frames, width, height, fps)
