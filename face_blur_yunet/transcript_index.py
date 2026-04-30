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
