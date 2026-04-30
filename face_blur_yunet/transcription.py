from __future__ import annotations

from pathlib import Path
from typing import Protocol

from face_blur_yunet.models import Language, TranscriptSegment


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, language: Language) -> list[TranscriptSegment]:
        ...


class FakeTranscriber:
    def __init__(self, texts: list[str]):
        self.texts = texts

    def transcribe(self, audio_path: Path, language: Language) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                id=index,
                start=(index - 1) * 2.0,
                end=index * 2.0,
                text=text,
                language=language,
            )
            for index, text in enumerate(self.texts, start=1)
        ]


class FasterWhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "cpu", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to enable local transcription") from exc
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, language: Language) -> list[TranscriptSegment]:
        language_code = None if language == Language.AUTO else language.value
        raw_segments, info = self.model.transcribe(str(audio_path), language=language_code)
        detected = Language(info.language) if getattr(info, "language", None) in {"he", "en"} else language
        segments = []
        for index, segment in enumerate(raw_segments, start=1):
            segments.append(
                TranscriptSegment(
                    id=index,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    language=detected,
                    confidence=None,
                )
            )
        return segments


def transcribe_audio(
    audio_path: Path,
    language: Language,
    transcriber: Transcriber | None = None,
) -> list[TranscriptSegment]:
    engine = transcriber or FasterWhisperTranscriber()
    return engine.transcribe(audio_path, language)
