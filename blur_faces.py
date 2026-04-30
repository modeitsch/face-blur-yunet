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
