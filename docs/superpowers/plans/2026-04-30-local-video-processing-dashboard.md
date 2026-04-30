# Local Video Processing Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local private dashboard that can process client videos, transcribe Hebrew/English audio, export transcript/SRT files, answer transcript-grounded questions, optionally translate transcripts, and optionally blur faces using the existing YuNet pipeline.

**Architecture:** Refactor the current single script into a focused Python package with reusable media, face blur, transcription, transcript indexing, question answering, translation, subtitle, job store, and pipeline modules. Add a small FastAPI local dashboard on top of the backend while keeping the existing `blur_faces.py` command working. All processing stays local and writes artifacts to per-job folders.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, ffmpeg/ffprobe, SQLite, pytest, FastAPI/Uvicorn, optional `faster-whisper` for local speech-to-text, optional Argos Translate for offline translation, optional Ollama for local transcript-grounded answer generation.

---

## Context And Library Notes

- Current repo contains `blur_faces.py`, `requirements.txt`, `.gitignore`, and README.
- Current `blur_faces.py` downloads YuNet, detects faces with OpenCV `FaceDetectorYN`, applies oval Gaussian blur, and muxes original audio with `ffmpeg`.
- `faster-whisper` is a Python speech-to-text implementation using CTranslate2 and supports CPU/GPU execution. Its README says Python 3.9+ is required and GPU execution requires CUDA/cuBLAS/cuDNN on NVIDIA machines.
- Argos Translate is an offline Python translation library using installable `.argosmodel` language packages.
- Ollama exposes local HTTP APIs at `http://localhost:11434/api`, including generation and embeddings endpoints. Version 1 must not require Ollama to be installed.

References checked while writing the plan:

- https://github.com/SYSTRAN/faster-whisper
- https://github.com/argosopentech/argos-translate/
- https://docs.ollama.com/api/introduction
- https://docs.ollama.com/capabilities/embeddings
- https://github.com/fastapi/fastapi
- https://uvicorn.dev/installation/

## File Map

Create these package files:

- `face_blur_yunet/__init__.py`: package marker and version.
- `face_blur_yunet/models.py`: dataclasses and enums shared by the pipeline.
- `face_blur_yunet/face_blur.py`: YuNet model download, box clamping, blur mask, video blur, audio mux.
- `face_blur_yunet/media.py`: ffmpeg/ffprobe checks, media metadata, audio extraction.
- `face_blur_yunet/subtitles.py`: transcript text export and SRT formatting.
- `face_blur_yunet/transcription.py`: local transcription adapter interface plus `faster-whisper` implementation and test fake.
- `face_blur_yunet/transcript_index.py`: transcript chunking and local JSON index.
- `face_blur_yunet/question_answering.py`: transcript-grounded answer engine with deterministic retrieval and optional Ollama generation.
- `face_blur_yunet/translation.py`: translation adapter interface plus optional Argos implementation and unavailable fallback.
- `face_blur_yunet/jobs.py`: SQLite job store and question history.
- `face_blur_yunet/pipeline.py`: orchestration of selected job options.
- `face_blur_yunet/app.py`: FastAPI routes and static UI serving.
- `face_blur_yunet/static/index.html`: local dashboard UI.
- `face_blur_yunet/static/styles.css`: dashboard styling.
- `face_blur_yunet/static/app.js`: dashboard behavior.

Modify these existing files:

- `blur_faces.py`: compatibility CLI that calls `face_blur_yunet.face_blur`.
- `requirements.txt`: runtime dependencies.
- `.gitignore`: local data directories.
- `README.md`: usage instructions for dashboard and CLI.

Create these tests:

- `tests/test_face_blur.py`
- `tests/test_media.py`
- `tests/test_subtitles.py`
- `tests/test_transcript_index.py`
- `tests/test_question_answering.py`
- `tests/test_translation.py`
- `tests/test_jobs.py`
- `tests/test_pipeline.py`
- `tests/test_app.py`

---

### Task 1: Package Scaffold And Shared Models

**Files:**
- Create: `face_blur_yunet/__init__.py`
- Create: `face_blur_yunet/models.py`
- Create: `tests/test_models.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the shared model tests**

Create `tests/test_models.py`:

```python
from pathlib import Path

from face_blur_yunet.models import JobOptions, Language, TranscriptSegment


def test_job_options_enables_questions_when_transcript_is_enabled():
    options = JobOptions(transcript=True, questions=True)
    assert options.requires_audio is True


def test_face_blur_only_does_not_require_audio():
    options = JobOptions(transcript=False, questions=False, face_blur=True)
    assert options.requires_audio is False


def test_transcript_segment_formats_timestamp():
    segment = TranscriptSegment(id=1, start=65.2, end=67.8, text="hello", language=Language.ENGLISH)
    assert segment.start_srt == "00:01:05,200"
    assert segment.end_srt == "00:01:07,800"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'face_blur_yunet'`.

- [ ] **Step 3: Add the package and models**

Create `face_blur_yunet/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `face_blur_yunet/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Language(StrEnum):
    AUTO = "auto"
    HEBREW = "he"
    ENGLISH = "en"


class JobStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    INDEXING_TRANSCRIPT = "indexing_transcript"
    TRANSLATING = "translating"
    WRITING_SUBTITLES = "writing_subtitles"
    BLURRING_FACES = "blurring_faces"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class JobOptions:
    source_language: Language = Language.AUTO
    transcript: bool = True
    questions: bool = True
    translation_target: Language | None = None
    subtitles: bool = True
    face_blur: bool = False
    score_threshold: float = 0.75
    blur_strength: int = 31
    face_padding: float = 0.0

    @property
    def requires_audio(self) -> bool:
        return self.transcript or self.questions or self.translation_target is not None


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    language: Language = Language.AUTO
    confidence: float | None = None

    @property
    def start_srt(self) -> str:
        return seconds_to_srt_time(self.start)

    @property
    def end_srt(self) -> str:
        return seconds_to_srt_time(self.end)


@dataclass(frozen=True)
class TranscriptChunk:
    id: int
    text: str
    segment_ids: list[int]
    start: float
    end: float
    language: Language


@dataclass(frozen=True)
class QuestionAnswer:
    question: str
    answer: str
    language: Language
    timestamps: list[tuple[float, float]] = field(default_factory=list)
    excerpts: list[str] = field(default_factory=list)
    grounded: bool = True


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def seconds_to_srt_time(value: float) -> str:
    milliseconds = int(round(value * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
```

- [ ] **Step 4: Update runtime/test requirements**

Replace `requirements.txt` with:

```text
numpy
opencv-python-headless
fastapi
uvicorn
python-multipart
pytest
httpx
```

Do not add `faster-whisper`, `argostranslate`, or `ollama` as hard requirements yet. Those are optional local engines loaded dynamically.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt face_blur_yunet/__init__.py face_blur_yunet/models.py tests/test_models.py
git commit -m "feat: add shared processing models"
```

---

### Task 2: Refactor Existing Face Blur Into Package

**Files:**
- Create: `face_blur_yunet/face_blur.py`
- Modify: `blur_faces.py`
- Create: `tests/test_face_blur.py`

- [ ] **Step 1: Write tests for box clamping and padding**

Create `tests/test_face_blur.py`:

```python
import numpy as np

from face_blur_yunet.face_blur import BlurOptions, clamp_box, blur_face_only


def test_clamp_box_keeps_box_inside_frame():
    assert clamp_box((-10.2, 5.4, 50.1, 20.2), 100, 80) == (0, 5, 50, 20)
    assert clamp_box((90, 70, 30, 30), 100, 80) == (90, 70, 10, 10)


def test_clamp_box_applies_padding_inside_frame():
    options = BlurOptions(face_padding=0.25)
    assert clamp_box((20, 20, 40, 20), 100, 80, options) == (10, 15, 60, 30)


def test_blur_face_only_changes_pixels_inside_roi():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[20:60, 30:70] = 255
    original = frame.copy()
    blur_face_only(frame, (30, 20, 40, 40), BlurOptions(blur_strength=31))
    assert np.any(frame != original)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_face_blur.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing functions.

- [ ] **Step 3: Move face blur logic into the package**

Create `face_blur_yunet/face_blur.py` by moving the current logic from `blur_faces.py` and adding these public APIs:

```python
from __future__ import annotations

import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class BlurOptions:
    score_threshold: float = 0.75
    blur_strength: int = 31
    face_padding: float = 0.0
    oval_mask: bool = True


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def clamp_box(box, frame_w: int, frame_h: int, options: BlurOptions | None = None) -> tuple[int, int, int, int]:
    x, y, w, h = [int(round(v)) for v in box]
    if options and options.face_padding > 0:
        pad_x = int(round(w * options.face_padding))
        pad_y = int(round(h * options.face_padding))
        x -= pad_x
        y -= pad_y
        w += pad_x * 2
        h += pad_y * 2
    x = max(0, x)
    y = max(0, y)
    w = min(frame_w - x, max(1, w))
    h = min(frame_h - y, max(1, h))
    return x, y, w, h


def _odd_kernel(value: int) -> int:
    return max(3, value if value % 2 == 1 else value + 1)


def blur_face_only(frame, box, options: BlurOptions | None = None) -> None:
    options = options or BlurOptions()
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = clamp_box(box, frame_w, frame_h, options)
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return

    kernel = _odd_kernel(max(options.blur_strength, (max(w, h) // 2) * 2 + 1))
    blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)

    if not options.oval_mask:
        frame[y : y + h, x : x + w] = blurred
        return

    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(1, int(w * 0.48)), max(1, int(h * 0.53)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    frame[y : y + h, x : x + w] = (blurred * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def process_video(model_path: Path, input_path: Path, temp_video_path: Path, options: BlurOptions) -> dict[str, int]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.FaceDetectorYN.create(str(model_path), "", (width, height), options.score_threshold, 0.3, 5000)
    writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {temp_video_path}")

    frame_count = 0
    detected_frames = 0
    total_faces = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, faces = detector.detect(frame)
        if faces is not None:
            detected_frames += 1
            total_faces += len(faces)
            for face in faces:
                blur_face_only(frame, face[:4], options)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    return {"frames": frame_count, "detected_frames": detected_frames, "faces": total_faces}


def mux_audio(temp_video_path: Path, input_path: Path, output_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_video_path), "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy", "-shortest", str(output_path),
    ], check=True)


def blur_video(input_path: Path, output_path: Path, model_path: Path = DEFAULT_MODEL_PATH, options: BlurOptions | None = None) -> dict[str, int]:
    model_path = ensure_model(model_path)
    options = options or BlurOptions()
    with tempfile.TemporaryDirectory() as td:
        temp_video = Path(td) / "blurred_no_audio.mp4"
        report = process_video(model_path, input_path, temp_video, options)
        mux_audio(temp_video, input_path, output_path)
        return report
```

- [ ] **Step 4: Replace `blur_faces.py` with a compatibility CLI**

Replace `blur_faces.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from face_blur_yunet.face_blur import BlurOptions, blur_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Blur faces in a video using OpenCV YuNet.")
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("output", type=Path, help="Output video path")
    parser.add_argument("--model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--score-threshold", type=float, default=0.75)
    parser.add_argument("--blur-strength", type=int, default=31)
    parser.add_argument("--face-padding", type=float, default=0.0)
    parser.add_argument("--box-mask", action="store_true", help="Blur the full box instead of using an oval mask")
    args = parser.parse_args()

    report = blur_video(
        args.input,
        args.output,
        args.model,
        BlurOptions(
            score_threshold=args.score_threshold,
            blur_strength=args.blur_strength,
            face_padding=args.face_padding,
            oval_mask=not args.box_mask,
        ),
    )
    print(f"processed {report['frames']} frames; faces blurred: {report['faces']}")
    print(args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_face_blur.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add blur_faces.py face_blur_yunet/face_blur.py tests/test_face_blur.py
git commit -m "refactor: move face blur engine into package"
```

---

### Task 3: Transcript And Subtitle Export

**Files:**
- Create: `face_blur_yunet/subtitles.py`
- Create: `tests/test_subtitles.py`

- [ ] **Step 1: Write subtitle tests**

Create `tests/test_subtitles.py`:

```python
from face_blur_yunet.models import Language, TranscriptSegment
from face_blur_yunet.subtitles import segments_to_plain_text, segments_to_srt


def sample_segments():
    return [
        TranscriptSegment(id=1, start=0.0, end=1.5, text="Hello", language=Language.ENGLISH),
        TranscriptSegment(id=2, start=2.0, end=4.25, text="World", language=Language.ENGLISH),
    ]


def test_segments_to_plain_text():
    assert segments_to_plain_text(sample_segments()) == "Hello\nWorld\n"


def test_segments_to_srt():
    assert segments_to_srt(sample_segments()) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"
        "2\n00:00:02,000 --> 00:00:04,250\nWorld\n"
    )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_subtitles.py -v`

Expected: FAIL with missing module or functions.

- [ ] **Step 3: Implement subtitle helpers**

Create `face_blur_yunet/subtitles.py`:

```python
from __future__ import annotations

from pathlib import Path

from face_blur_yunet.models import TranscriptSegment


def segments_to_plain_text(segments: list[TranscriptSegment]) -> str:
    return "".join(f"{segment.text.strip()}\n" for segment in segments if segment.text.strip())


def segments_to_srt(segments: list[TranscriptSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(f"{index}\n{segment.start_srt} --> {segment.end_srt}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_transcript(path: Path, segments: list[TranscriptSegment]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(segments_to_plain_text(segments), encoding="utf-8")
    return path


def write_srt(path: Path, segments: list[TranscriptSegment]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(segments_to_srt(segments), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_subtitles.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/subtitles.py tests/test_subtitles.py
git commit -m "feat: add transcript and subtitle export"
```

---

### Task 4: Media Probe And Audio Extraction

**Files:**
- Create: `face_blur_yunet/media.py`
- Create: `tests/test_media.py`

- [ ] **Step 1: Write media command tests using monkeypatch**

Create `tests/test_media.py`:

```python
import json
import subprocess
from pathlib import Path

import pytest

from face_blur_yunet.media import extract_audio, probe_media


def test_probe_media_parses_ffprobe_json(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    def fake_run(cmd, check, capture_output, text):
        payload = {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "12.5"},
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = probe_media(video)
    assert info.width == 1920
    assert info.height == 1080
    assert round(info.fps, 2) == 29.97
    assert info.has_audio is True


def test_extract_audio_runs_ffmpeg(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = extract_audio(tmp_path / "input.mp4", tmp_path / "audio.wav")
    assert output.name == "audio.wav"
    assert calls[0][0] == "ffmpeg"
    assert "-vn" in calls[0]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_media.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement media helpers**

Create `face_blur_yunet/media.py`:

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from face_blur_yunet.models import MediaInfo


def _parse_fps(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value or 0.0)


def probe_media(path: Path) -> MediaInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)
    ], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    video_stream = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")
    has_audio = any(stream.get("codec_type") == "audio" for stream in payload.get("streams", []))
    return MediaInfo(
        path=path,
        duration=float(payload.get("format", {}).get("duration") or 0.0),
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=_parse_fps(video_stream.get("avg_frame_rate") or "0"),
        has_audio=has_audio,
    )


def extract_audio(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", str(output_path),
    ], check=True)
    return output_path
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_media.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/media.py tests/test_media.py
git commit -m "feat: add media probe and audio extraction"
```

---

### Task 5: Transcript Indexing

**Files:**
- Create: `face_blur_yunet/transcript_index.py`
- Create: `tests/test_transcript_index.py`

- [ ] **Step 1: Write transcript indexing tests**

Create `tests/test_transcript_index.py`:

```python
import json

from face_blur_yunet.models import Language, TranscriptSegment
from face_blur_yunet.transcript_index import build_chunks, index_transcript, load_index


def test_build_chunks_preserves_timestamp_range():
    segments = [
        TranscriptSegment(id=1, start=0, end=2, text="pricing starts today", language=Language.ENGLISH),
        TranscriptSegment(id=2, start=2, end=4, text="delivery is next week", language=Language.ENGLISH),
    ]
    chunks = build_chunks(segments, max_chars=80)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == 4
    assert chunks[0].segment_ids == [1, 2]


def test_index_transcript_writes_json(tmp_path):
    segments = [TranscriptSegment(id=1, start=0, end=1, text="hello", language=Language.ENGLISH)]
    path = index_transcript(segments, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["chunks"][0]["text"] == "hello"
    loaded = load_index(path)
    assert loaded[0].text == "hello"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_transcript_index.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement transcript indexing**

Create `face_blur_yunet/transcript_index.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from face_blur_yunet.models import Language, TranscriptChunk, TranscriptSegment


def build_chunks(segments: list[TranscriptSegment], max_chars: int = 1200) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_len = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if current and current_len + len(text) + 1 > max_chars:
            chunks.append(_make_chunk(len(chunks) + 1, current))
            current = []
            current_len = 0
        current.append(segment)
        current_len += len(text) + 1
    if current:
        chunks.append(_make_chunk(len(chunks) + 1, current))
    return chunks


def _make_chunk(chunk_id: int, segments: list[TranscriptSegment]) -> TranscriptChunk:
    return TranscriptChunk(
        id=chunk_id,
        text=" ".join(segment.text.strip() for segment in segments),
        segment_ids=[segment.id for segment in segments],
        start=segments[0].start,
        end=segments[-1].end,
        language=segments[0].language,
    )


def index_transcript(segments: list[TranscriptSegment], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(segments)
    path = output_dir / "transcript-index.json"
    payload = {"chunks": [asdict(chunk) for chunk in chunks]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_index(path: Path) -> list[TranscriptChunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        TranscriptChunk(
            id=int(item["id"]),
            text=item["text"],
            segment_ids=[int(value) for value in item["segment_ids"]],
            start=float(item["start"]),
            end=float(item["end"]),
            language=Language(item["language"]),
        )
        for item in payload.get("chunks", [])
    ]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_transcript_index.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/transcript_index.py tests/test_transcript_index.py
git commit -m "feat: add transcript indexing"
```

---

### Task 6: Transcript-Grounded Question Answering

**Files:**
- Create: `face_blur_yunet/question_answering.py`
- Create: `tests/test_question_answering.py`

- [ ] **Step 1: Write deterministic Q&A tests**

Create `tests/test_question_answering.py`:

```python
from face_blur_yunet.models import Language, TranscriptChunk
from face_blur_yunet.question_answering import answer_from_chunks


def chunks():
    return [
        TranscriptChunk(id=1, text="The price is 500 dollars and includes setup.", segment_ids=[1], start=10, end=20, language=Language.ENGLISH),
        TranscriptChunk(id=2, text="Delivery will happen next Monday.", segment_ids=[2], start=30, end=40, language=Language.ENGLISH),
    ]


def test_answer_from_chunks_returns_grounded_excerpt():
    answer = answer_from_chunks("What is the price?", chunks(), Language.ENGLISH)
    assert answer.grounded is True
    assert "500 dollars" in answer.answer
    assert answer.timestamps == [(10, 20)]


def test_answer_from_chunks_does_not_guess():
    answer = answer_from_chunks("What is the warranty?", chunks(), Language.ENGLISH)
    assert answer.grounded is False
    assert "I could not find" in answer.answer
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_question_answering.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement deterministic Q&A with optional Ollama hook**

Create `face_blur_yunet/question_answering.py`:

```python
from __future__ import annotations

import re
from collections import Counter

from face_blur_yunet.models import Language, QuestionAnswer, TranscriptChunk

STOPWORDS = {
    "a", "an", "and", "are", "about", "did", "do", "does", "for", "how", "in", "is", "it", "of", "on",
    "the", "they", "to", "what", "when", "where", "who", "why", "will", "with",
    "את", "על", "של", "מה", "מי", "מתי", "איפה", "איך", "האם", "זה", "זו", "עם", "לא", "כן",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[\w\u0590-\u05ff]+", text) if token.lower() not in STOPWORDS]


def rank_chunks(question: str, chunks: list[TranscriptChunk], limit: int = 3) -> list[TranscriptChunk]:
    question_terms = Counter(tokenize(question))
    scored: list[tuple[int, TranscriptChunk]] = []
    for chunk in chunks:
        chunk_terms = Counter(tokenize(chunk.text))
        score = sum(min(count, chunk_terms.get(term, 0)) for term, count in question_terms.items())
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].start))
    return [chunk for _, chunk in scored[:limit]]


def answer_from_chunks(question: str, chunks: list[TranscriptChunk], answer_language: Language) -> QuestionAnswer:
    matches = rank_chunks(question, chunks, limit=2)
    if not matches:
        fallback = "I could not find enough information in the transcript to answer that."
        if answer_language == Language.HEBREW:
            fallback = "לא מצאתי מספיק מידע בתמלול כדי לענות על השאלה."
        return QuestionAnswer(question=question, answer=fallback, language=answer_language, grounded=False)

    excerpts = [chunk.text for chunk in matches]
    timestamps = [(chunk.start, chunk.end) for chunk in matches]
    answer = "\n\n".join(excerpts)
    return QuestionAnswer(question=question, answer=answer, language=answer_language, timestamps=timestamps, excerpts=excerpts, grounded=True)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_question_answering.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/question_answering.py tests/test_question_answering.py
git commit -m "feat: add transcript grounded question answering"
```

---

### Task 7: Transcription Adapter

**Files:**
- Create: `face_blur_yunet/transcription.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 1: Write tests using a fake backend**

Create `tests/test_transcription.py`:

```python
from face_blur_yunet.models import Language
from face_blur_yunet.transcription import FakeTranscriber, transcribe_audio


def test_fake_transcriber_returns_segments(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    segments = transcribe_audio(audio, Language.ENGLISH, FakeTranscriber(["hello", "world"]))
    assert [segment.text for segment in segments] == ["hello", "world"]
    assert segments[0].start == 0.0
    assert segments[1].start == 2.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_transcription.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement adapter and optional faster-whisper backend**

Create `face_blur_yunet/transcription.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from face_blur_yunet.models import Language, TranscriptSegment


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, language: Language) -> list[TranscriptSegment]:
        ...


class FakeTranscriber:
    def __init__(self, texts: list[str]):
        self.texts = texts

    def transcribe(self, audio_path: Path, language: Language) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(id=index, start=(index - 1) * 2.0, end=index * 2.0, text=text, language=language)
            for index, text in enumerate(self.texts, start=1)
        ]


class FasterWhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "cpu", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to enable local transcription") from exc
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, language: Language) -> list[TranscriptSegment]:
        language_code = None if language == Language.AUTO else language.value
        raw_segments, info = self.model.transcribe(str(audio_path), language=language_code)
        detected = Language(info.language) if getattr(info, "language", None) in {"he", "en"} else language
        segments = []
        for index, segment in enumerate(raw_segments, start=1):
            segments.append(
                TranscriptSegment(
                    id=index,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    language=detected,
                    confidence=None,
                )
            )
        return segments


def transcribe_audio(audio_path: Path, language: Language, transcriber: Transcriber | None = None) -> list[TranscriptSegment]:
    engine = transcriber or FasterWhisperTranscriber()
    return engine.transcribe(audio_path, language)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_transcription.py -v`

Expected: PASS without installing `faster-whisper` because the test uses `FakeTranscriber`.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/transcription.py tests/test_transcription.py
git commit -m "feat: add local transcription adapter"
```

---

### Task 8: Optional Offline Translation Adapter

**Files:**
- Create: `face_blur_yunet/translation.py`
- Create: `tests/test_translation.py`

- [ ] **Step 1: Write translation tests with fake and unavailable engines**

Create `tests/test_translation.py`:

```python
import pytest

from face_blur_yunet.models import Language, TranscriptSegment
from face_blur_yunet.translation import FakeTranslator, UnavailableTranslator, translate_segments


def segments():
    return [TranscriptSegment(id=1, start=0, end=1, text="hello", language=Language.ENGLISH)]


def test_fake_translator_preserves_timestamps():
    translated = translate_segments(segments(), Language.HEBREW, FakeTranslator({"hello": "שלום"}))
    assert translated[0].text == "שלום"
    assert translated[0].start == 0
    assert translated[0].end == 1
    assert translated[0].language == Language.HEBREW


def test_unavailable_translator_raises_clear_error():
    with pytest.raises(RuntimeError, match="Translation backend is not configured"):
        translate_segments(segments(), Language.HEBREW, UnavailableTranslator())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_translation.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement translation adapters**

Create `face_blur_yunet/translation.py`:

```python
from __future__ import annotations

from typing import Protocol

from face_blur_yunet.models import Language, TranscriptSegment


class Translator(Protocol):
    def translate(self, text: str, source: Language, target: Language) -> str:
        ...


class UnavailableTranslator:
    def translate(self, text: str, source: Language, target: Language) -> str:
        raise RuntimeError("Translation backend is not configured")


class FakeTranslator:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def translate(self, text: str, source: Language, target: Language) -> str:
        return self.mapping.get(text, text)


class ArgosTranslator:
    def __init__(self):
        try:
            import argostranslate.translate
        except ImportError as exc:
            raise RuntimeError("Install argostranslate and language packages to enable offline translation") from exc
        self._translate = argostranslate.translate

    def translate(self, text: str, source: Language, target: Language) -> str:
        if source == Language.AUTO:
            raise RuntimeError("Choose a source language before translation")
        return self._translate.translate(text, source.value, target.value)


def translate_segments(
    segments: list[TranscriptSegment],
    target: Language,
    translator: Translator | None = None,
) -> list[TranscriptSegment]:
    engine = translator or UnavailableTranslator()
    translated = []
    for segment in segments:
        translated.append(
            TranscriptSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=engine.translate(segment.text, segment.language, target),
                language=target,
                confidence=segment.confidence,
            )
        )
    return translated
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_translation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/translation.py tests/test_translation.py
git commit -m "feat: add optional offline translation adapter"
```

---

### Task 9: SQLite Job Store

**Files:**
- Create: `face_blur_yunet/jobs.py`
- Create: `tests/test_jobs.py`

- [ ] **Step 1: Write job store tests**

Create `tests/test_jobs.py`:

```python
from face_blur_yunet.jobs import JobStore
from face_blur_yunet.models import JobOptions, JobStatus, Language, QuestionAnswer


def test_job_store_creates_and_updates_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(tmp_path / "input.mp4", tmp_path / "outputs", JobOptions(source_language=Language.ENGLISH))
    assert job.status == JobStatus.QUEUED
    store.update_status(job.id, JobStatus.COMPLETE)
    loaded = store.get_job(job.id)
    assert loaded.status == JobStatus.COMPLETE


def test_job_store_records_question_history(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(tmp_path / "input.mp4", tmp_path / "outputs", JobOptions())
    store.add_question_answer(job.id, QuestionAnswer(question="price?", answer="500", language=Language.ENGLISH, timestamps=[(1, 2)], excerpts=["price 500"]))
    history = store.list_question_history(job.id)
    assert history[0].answer == "500"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_jobs.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement SQLite job store**

Create `face_blur_yunet/jobs.py` with a small dataclass `StoredJob` and methods `create_job`, `get_job`, `update_status`, `set_artifact`, `add_question_answer`, and `list_question_history`. Store `JobOptions` and artifact paths as JSON strings. Use `sqlite3.connect(self.db_path)` per method to avoid long-lived connection issues in tests.

The schema must include:

```sql
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  input_path TEXT NOT NULL,
  output_dir TEXT NOT NULL,
  options_json TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  artifacts_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE TABLE IF NOT EXISTS question_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  language TEXT NOT NULL,
  timestamps_json TEXT NOT NULL,
  excerpts_json TEXT NOT NULL,
  grounded INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(job_id) REFERENCES jobs(id)
);
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_jobs.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/jobs.py tests/test_jobs.py
git commit -m "feat: add sqlite job store"
```

---

### Task 10: Processing Pipeline

**Files:**
- Create: `face_blur_yunet/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write pipeline tests with fake engines**

Create `tests/test_pipeline.py`:

```python
from face_blur_yunet.jobs import JobStore
from face_blur_yunet.models import JobOptions, JobStatus, Language
from face_blur_yunet.pipeline import Pipeline, PipelineEngines
from face_blur_yunet.transcription import FakeTranscriber
from face_blur_yunet.translation import FakeTranslator


def test_pipeline_writes_transcript_srt_index_and_answers(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(input_path, tmp_path / "outputs", JobOptions(source_language=Language.ENGLISH, translation_target=Language.HEBREW))

    monkeypatch.setattr("face_blur_yunet.pipeline.probe_media", lambda path: type("Info", (), {"has_audio": True})())
    monkeypatch.setattr("face_blur_yunet.pipeline.extract_audio", lambda src, dst: dst.write_bytes(b"audio") or dst)

    pipeline = Pipeline(store, PipelineEngines(transcriber=FakeTranscriber(["price is 500"]), translator=FakeTranslator({"price is 500": "המחיר הוא 500"})))
    pipeline.run(job.id)

    loaded = store.get_job(job.id)
    assert loaded.status == JobStatus.COMPLETE
    assert (tmp_path / "outputs" / f"job-{job.id}" / "transcript.en.txt").exists()
    assert (tmp_path / "outputs" / f"job-{job.id}" / "subtitles.en.srt").exists()
    assert (tmp_path / "outputs" / f"job-{job.id}" / "transcript-index.json").exists()
    assert (tmp_path / "outputs" / f"job-{job.id}" / "transcript.he.txt").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement pipeline orchestration**

Create `face_blur_yunet/pipeline.py` with:

- `PipelineEngines` dataclass containing optional `transcriber`, `translator`, and `face_blur_func`.
- `Pipeline.run(job_id)` that:
  - loads the job from `JobStore`;
  - validates media with `probe_media`;
  - extracts audio when `job.options.requires_audio`;
  - transcribes with `transcribe_audio`;
  - writes `transcript.<lang>.txt` and `subtitles.<lang>.srt`;
  - builds `transcript-index.json`;
  - translates if `translation_target` is set;
  - blurs faces if `face_blur` is true;
  - writes `processing-report.json`;
  - updates status after each step;
  - marks failed jobs with the exception message.

Use these output names exactly:

```text
job-{id}/audio.wav
job-{id}/transcript.<language>.txt
job-{id}/subtitles.<language>.srt
job-{id}/transcript-index.json
job-{id}/video.face-blurred.mp4
job-{id}/processing-report.json
```

- [ ] **Step 4: Run pipeline tests**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: Run all backend tests**

Run: `python -m pytest tests/test_models.py tests/test_face_blur.py tests/test_media.py tests/test_subtitles.py tests/test_transcript_index.py tests/test_question_answering.py tests/test_translation.py tests/test_jobs.py tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add face_blur_yunet/pipeline.py tests/test_pipeline.py
git commit -m "feat: add local video processing pipeline"
```

---

### Task 11: Local Dashboard API

**Files:**
- Create: `face_blur_yunet/app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write FastAPI route tests**

Create `tests/test_app.py`:

```python
from fastapi.testclient import TestClient

from face_blur_yunet.app import create_app
from face_blur_yunet.jobs import JobStore
from face_blur_yunet.models import JobOptions, Language


def test_health_route(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}


def test_ask_question_route_uses_job_index(tmp_path):
    app = create_app(tmp_path)
    store = JobStore(tmp_path / "jobs.sqlite")
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    job = store.create_job(input_path, tmp_path / "outputs", JobOptions(source_language=Language.ENGLISH))
    job_dir = tmp_path / "outputs" / f"job-{job.id}"
    job_dir.mkdir(parents=True)
    (job_dir / "transcript-index.json").write_text(
        '{"chunks":[{"id":1,"text":"price is 500","segment_ids":[1],"start":1,"end":2,"language":"en"}]}',
        encoding="utf-8",
    )
    store.set_artifact(job.id, "transcript_index", job_dir / "transcript-index.json")

    client = TestClient(app)
    response = client.post(f"/api/jobs/{job.id}/questions", json={"question": "price?", "answer_language": "en"})
    assert response.status_code == 200
    assert "500" in response.json()["answer"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_app.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement API app**

Create `face_blur_yunet/app.py` with `create_app(base_dir: Path) -> FastAPI` and these routes:

- `GET /api/health` returns `{"ok": true}`.
- `POST /api/jobs` accepts JSON with `input_path`, `source_language`, `translation_target`, `subtitles`, `face_blur`, and creates a queued job.
- `GET /api/jobs/{job_id}` returns job metadata.
- `POST /api/jobs/{job_id}/run` runs the pipeline synchronously for version 1.
- `POST /api/jobs/{job_id}/questions` loads the transcript index artifact, calls `answer_from_chunks`, stores the answer in `JobStore`, and returns the answer.
- `GET /` serves the static dashboard HTML after Task 12.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_app.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add face_blur_yunet/app.py tests/test_app.py
git commit -m "feat: add local dashboard api"
```

---

### Task 12: Static Dashboard UI

**Files:**
- Create: `face_blur_yunet/static/index.html`
- Create: `face_blur_yunet/static/styles.css`
- Create: `face_blur_yunet/static/app.js`
- Modify: `face_blur_yunet/app.py`

- [ ] **Step 1: Add static HTML**

Create `face_blur_yunet/static/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Processing Dashboard</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="shell">
    <section class="panel">
      <h1>Video Processing Dashboard</h1>
      <form id="job-form">
        <label>Video path <input id="input-path" name="input_path" required placeholder="/Users/me/video.mp4"></label>
        <label>Source language
          <select id="source-language"><option value="auto">Auto</option><option value="he">Hebrew</option><option value="en">English</option></select>
        </label>
        <label>Translate to
          <select id="translation-target"><option value="">None</option><option value="en">English</option><option value="he">Hebrew</option></select>
        </label>
        <label><input id="subtitles" type="checkbox" checked> Export SRT subtitles</label>
        <label><input id="face-blur" type="checkbox"> Blur faces</label>
        <button type="submit">Create Job</button>
      </form>
    </section>
    <section class="panel">
      <h2>Current Job</h2>
      <pre id="job-output">No job yet.</pre>
      <button id="run-job" disabled>Run Job</button>
    </section>
    <section class="panel">
      <h2>Ask the Video</h2>
      <form id="question-form">
        <input id="question" placeholder="Ask a question about the transcript">
        <select id="answer-language"><option value="en">English</option><option value="he">Hebrew</option></select>
        <button type="submit" disabled id="ask-button">Ask</button>
      </form>
      <pre id="answer-output">Process a video first.</pre>
    </section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Add restrained dashboard CSS**

Create `face_blur_yunet/static/styles.css`:

```css
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f7f8fa; color: #18202f; }
.shell { max-width: 1040px; margin: 0 auto; padding: 32px 20px; display: grid; gap: 16px; }
.panel { background: #ffffff; border: 1px solid #d8dee8; border-radius: 8px; padding: 18px; }
h1, h2 { margin: 0 0 14px; font-size: 20px; }
form { display: grid; gap: 12px; }
label { display: grid; gap: 6px; font-size: 14px; }
input, select, button { font: inherit; min-height: 40px; border-radius: 6px; border: 1px solid #b7c0ce; padding: 0 10px; }
button { background: #1f6feb; color: white; border-color: #1f6feb; cursor: pointer; }
button:disabled { background: #9aa7b8; border-color: #9aa7b8; cursor: not-allowed; }
pre { white-space: pre-wrap; overflow: auto; background: #f0f3f8; border-radius: 6px; padding: 12px; min-height: 72px; }
@media (max-width: 720px) { .shell { padding: 18px 12px; } }
```

- [ ] **Step 3: Add dashboard JavaScript**

Create `face_blur_yunet/static/app.js`:

```javascript
let currentJobId = null;

const jobOutput = document.querySelector('#job-output');
const runButton = document.querySelector('#run-job');
const askButton = document.querySelector('#ask-button');
const answerOutput = document.querySelector('#answer-output');

document.querySelector('#job-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const body = {
    input_path: document.querySelector('#input-path').value,
    source_language: document.querySelector('#source-language').value,
    translation_target: document.querySelector('#translation-target').value || null,
    subtitles: document.querySelector('#subtitles').checked,
    face_blur: document.querySelector('#face-blur').checked
  };
  const response = await fetch('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const payload = await response.json();
  currentJobId = payload.id;
  jobOutput.textContent = JSON.stringify(payload, null, 2);
  runButton.disabled = false;
});

runButton.addEventListener('click', async () => {
  runButton.disabled = true;
  jobOutput.textContent = 'Processing...';
  const response = await fetch(`/api/jobs/${currentJobId}/run`, { method: 'POST' });
  const payload = await response.json();
  jobOutput.textContent = JSON.stringify(payload, null, 2);
  askButton.disabled = false;
});

document.querySelector('#question-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const body = {
    question: document.querySelector('#question').value,
    answer_language: document.querySelector('#answer-language').value
  };
  const response = await fetch(`/api/jobs/${currentJobId}/questions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const payload = await response.json();
  answerOutput.textContent = JSON.stringify(payload, null, 2);
});
```

- [ ] **Step 4: Wire static files in FastAPI**

Modify `face_blur_yunet/app.py` so `/static` serves `face_blur_yunet/static` and `/` returns `index.html`.

- [ ] **Step 5: Smoke test the app imports**

Run: `python -c "from pathlib import Path; from face_blur_yunet.app import create_app; app = create_app(Path('data')); print(app.title)"`

Expected output contains the app title.

- [ ] **Step 6: Commit**

```bash
git add face_blur_yunet/app.py face_blur_yunet/static/index.html face_blur_yunet/static/styles.css face_blur_yunet/static/app.js
git commit -m "feat: add local dashboard ui"
```

---

### Task 13: Docs, Ignore Rules, And Final Verification

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Update `.gitignore`**

Add:

```text
data/
outputs/
work/
*.sqlite
*.wav
```

Keep existing video/model ignores.

- [ ] **Step 2: Update README usage**

Add sections for:

- Dashboard setup: `python3 -m venv .venv`, `pip install -r requirements.txt`, `python -m uvicorn face_blur_yunet.app:create_default_app --factory --reload` if `create_default_app` is added, or the exact command implemented in Task 11.
- Optional transcription backend: `pip install faster-whisper`.
- Optional translation backend: `pip install argostranslate` plus a note that Hebrew/English language packages must be installed locally.
- Existing face blur CLI compatibility: `python blur_faces.py input.mp4 output_blurred.mp4`.
- Privacy note: files stay local unless the user configures a future cloud provider.

- [ ] **Step 3: Run all tests**

Run: `python -m pytest -v`

Expected: PASS.

- [ ] **Step 4: Run import smoke checks**

Run:

```bash
python -c "from face_blur_yunet.models import JobOptions; print(JobOptions())"
python -c "from face_blur_yunet.face_blur import BlurOptions; print(BlurOptions())"
python -c "from face_blur_yunet.app import create_app; from pathlib import Path; print(create_app(Path('data')).title)"
```

Expected: each command exits 0 and prints object/app information.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md
git commit -m "docs: document local dashboard workflow"
```

---

## Self-Review

Spec coverage:

- Local private dashboard: Tasks 11 and 12.
- Hebrew/English transcription: Task 7 defines the adapter and faster-whisper implementation path.
- Transcript exports and SRT: Task 3 and Task 10.
- Transcript-grounded questions: Tasks 5, 6, 9, 10, 11, and 12.
- Optional translation: Task 8 and Task 10.
- Optional face blur: Task 2 and Task 10.
- SQLite job state and question history: Task 9.
- Processing reports and per-job output folders: Task 10.
- Dubbing: intentionally excluded from version 1 per design.
- Subtitle burn-in: intentionally reserved for milestone 1b per design.
- Windows NVIDIA worker: intentionally reserved for later per design.

Placeholder scan:

- The plan contains no `TBD` or `TODO` markers.
- Task 9 and Task 10 describe implementation details where full file contents would be long, but they define exact method names, schema, filenames, and test contracts. The implementing worker must satisfy the tests before committing.

Type consistency:

- `Language`, `JobOptions`, `TranscriptSegment`, `TranscriptChunk`, `QuestionAnswer`, and `JobStatus` originate in Task 1 and are reused consistently.
- The dashboard routes use `JobStore`, `Pipeline`, `answer_from_chunks`, and `load_index` from earlier tasks.
- The output file names are consistent between pipeline tests, dashboard design, and README instructions.
