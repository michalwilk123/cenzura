# cenzura

Video text redaction tool - detect and redact sensitive text from videos using OCR.

## Installation

```bash
pip install git+https://github.com/michalwilk123/cenzura.git
```

Or install from source with uv:

```bash
uv pip install -e .
```

## Usage

```bash
cenzura input.mp4 "SECRET_KEY,password123" -o output.mp4
```

Run `cenzura --help` for all options.

## How it works

cenzura processes videos in two passes. In the scanning pass, it samples frames at ~1 fps
and only runs OCR when the frame content changes substantially (pixel diff threshold),
which avoids redundant processing on static scenes. Detected text is fuzzy-matched against
the provided secrets. In the redaction pass, it pipes raw frames through FFmpeg, draws
black rectangles over matched regions, and re-encodes with the original audio.

## Why not Tesseract?

Tesseract was evaluated as an alternative OCR backend. While ~3-4x faster than RapidOCR,
its detection accuracy was unusable - it frequently failed to detect target phrases correctly.
RapidOCR is the only supported engine.
