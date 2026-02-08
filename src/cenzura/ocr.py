"""OCR engine abstraction and implementations."""

from pathlib import Path
from typing import Protocol, List, Tuple
import numpy as np


class OcrEngine(Protocol):
    """Protocol defining the OCR engine interface."""

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        """
        Detect text in a frame.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            List of (x, y, w, h, text) tuples representing detected text bounding boxes
        """
        ...


class RapidOcrEngine:
    """OCR engine implementation using RapidOCR."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize RapidOCR engine.

        Args:
            config_path: Optional path to RapidOCR config YAML file
        """
        from rapidocr_onnxruntime import RapidOCR

        if config_path:
            self.engine = RapidOCR(config_path=config_path)
        else:
            self.engine = RapidOCR()

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        """
        Detect text using RapidOCR.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            List of (x, y, w, h, text) tuples
        """
        result, _ = self.engine(frame)

        if not result:
            return []

        detections = []
        for detection in result:
            bbox_points = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = detection[1]

            # Convert 4-corner polygon to bounding box (x, y, w, h)
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x_min = int(min(xs))
            y_min = int(min(ys))
            x_max = int(max(xs))
            y_max = int(max(ys))

            w = x_max - x_min
            h = y_max - y_min

            detections.append((x_min, y_min, w, h, text))

        return detections


