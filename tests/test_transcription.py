from face_blur_yunet.models import Language
from face_blur_yunet.transcription import FakeTranscriber, transcribe_audio


def test_fake_transcriber_returns_segments(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    segments = transcribe_audio(audio, Language.ENGLISH, FakeTranscriber(["hello", "world"]))
    assert [segment.text for segment in segments] == ["hello", "world"]
    assert segments[0].start == 0.0
    assert segments[1].start == 2.0
