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
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    video_stream = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), None)
    has_audio = any(stream.get("codec_type") == "audio" for stream in payload.get("streams", []))
    if video_stream is None and not has_audio:
        raise ValueError(f"No audio or video stream found in {path}")
    return MediaInfo(
        path=path,
        duration=float(payload.get("format", {}).get("duration") or 0.0),
        width=int(video_stream.get("width") or 0) if video_stream else 0,
        height=int(video_stream.get("height") or 0) if video_stream else 0,
        fps=_parse_fps(video_stream.get("avg_frame_rate") or "0") if video_stream else 0.0,
        has_audio=has_audio,
        has_video=video_stream is not None,
    )


def extract_audio(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ],
        check=True,
    )
    return output_path
