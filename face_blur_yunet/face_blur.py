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


def clamp_box(
    box, frame_w: int, frame_h: int, options: BlurOptions | None = None
) -> tuple[int, int, int, int]:
    x, y, w, h = [int(round(v)) for v in box]
    if options and options.face_padding > 0:
        pad_x = int(round(w * options.face_padding))
        pad_y = int(round(h * options.face_padding))
        x -= pad_x
        y -= pad_y
        w += pad_x * 2
        h += pad_y * 2
    x1 = max(0, min(frame_w, x))
    y1 = max(0, min(frame_h, y))
    x2 = max(0, min(frame_w, x + max(1, w)))
    y2 = max(0, min(frame_h, y + max(1, h)))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _odd_kernel(value: int) -> int:
    return max(3, value if value % 2 == 1 else value + 1)


def blur_face_only(frame, box, options: BlurOptions | None = None) -> None:
    options = options or BlurOptions()
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = clamp_box(box, frame_w, frame_h, options)
    if w == 0 or h == 0:
        return
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
    frame[y : y + h, x : x + w] = (blurred * alpha + roi * (1.0 - alpha)).astype(
        np.uint8
    )


def process_video(
    model_path: Path, input_path: Path, temp_video_path: Path, options: BlurOptions
) -> dict[str, int]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (width, height), options.score_threshold, 0.3, 5000
        )
        writer = cv2.VideoWriter(
            str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
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

        return {
            "frames": frame_count,
            "detected_frames": detected_frames,
            "faces": total_faces,
        }
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def mux_audio(temp_video_path: Path, input_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temp_video_path),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def blur_video(
    input_path: Path,
    output_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    options: BlurOptions | None = None,
) -> dict[str, int]:
    model_path = ensure_model(model_path)
    options = options or BlurOptions()
    with tempfile.TemporaryDirectory() as td:
        temp_video = Path(td) / "blurred_no_audio.mp4"
        report = process_video(model_path, input_path, temp_video, options)
        mux_audio(temp_video, input_path, output_path)
        return report
