#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np


MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_model(model_path):
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet face detector to {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)


def clamp_box(box, frame_w, frame_h):
    x, y, w, h = [int(round(v)) for v in box]
    x = max(0, x)
    y = max(0, y)
    w = min(frame_w - x, max(1, w))
    h = min(frame_h - y, max(1, h))
    return x, y, w, h


def blur_face_only(frame, box):
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = clamp_box(box, frame_w, frame_h)
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return

    kernel = max(31, (max(w, h) // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(1, int(w * 0.48)), max(1, int(h * 0.53)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    frame[y : y + h, x : x + w] = (blurred * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def process_video(model_path, input_path, temp_video_path, score_threshold):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (width, height),
        score_threshold,
        0.3,
        5000,
    )

    writer = cv2.VideoWriter(
        str(temp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {temp_video_path}")

    frame_idx = 0
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
                blur_face_only(frame, face[:4])

        writer.write(frame)
        frame_idx += 1
        if total and frame_idx % 100 == 0:
            print(f"processed {frame_idx}/{total} frames")

    cap.release()
    writer.release()
    print(f"processed {frame_idx} frames; detections on {detected_frames} frames; faces blurred: {total_faces}")


def mux_audio(temp_video_path, input_path, output_path):
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


def main():
    parser = argparse.ArgumentParser(description="Blur faces in a video using OpenCV YuNet.")
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("output", type=Path, help="Output video path")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/face_detection_yunet_2023mar.onnx"),
        help="Path to YuNet ONNX model; downloaded automatically if missing",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.75,
        help="Face detection confidence threshold. Lower catches more faces; higher reduces false positives.",
    )
    args = parser.parse_args()

    ensure_model(args.model)
    with tempfile.TemporaryDirectory() as td:
        temp_video = Path(td) / "blurred_no_audio.mp4"
        process_video(args.model, args.input, temp_video, args.score_threshold)
        mux_audio(temp_video, args.input, args.output)
        print(args.output)


if __name__ == "__main__":
    main()
