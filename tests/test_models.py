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
