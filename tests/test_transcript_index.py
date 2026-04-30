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
