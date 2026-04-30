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


def test_segments_to_srt_numbers_non_blank_segments_contiguously():
    segments = [
        TranscriptSegment(id=1, start=0.0, end=1.5, text="Hello", language=Language.ENGLISH),
        TranscriptSegment(id=2, start=2.0, end=3.0, text="   ", language=Language.ENGLISH),
        TranscriptSegment(id=3, start=4.0, end=5.25, text="World", language=Language.ENGLISH),
    ]

    assert segments_to_srt(segments) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"
        "2\n00:00:04,000 --> 00:00:05,250\nWorld\n"
    )
