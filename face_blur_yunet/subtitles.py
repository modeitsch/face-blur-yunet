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
