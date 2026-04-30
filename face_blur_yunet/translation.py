from __future__ import annotations

from typing import Protocol

from face_blur_yunet.models import Language, TranscriptSegment


class Translator(Protocol):
    def translate(self, text: str, source: Language, target: Language) -> str:
        ...


class UnavailableTranslator:
    def translate(self, text: str, source: Language, target: Language) -> str:
        raise RuntimeError("Translation backend is not configured")


class FakeTranslator:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def translate(self, text: str, source: Language, target: Language) -> str:
        return self.mapping.get(text, text)


class ArgosTranslator:
    def __init__(self):
        try:
            import argostranslate.translate
        except ImportError as exc:
            raise RuntimeError("Install argostranslate and language packages to enable offline translation") from exc
        self._translate = argostranslate.translate

    def translate(self, text: str, source: Language, target: Language) -> str:
        if source == Language.AUTO:
            raise RuntimeError("Choose a source language before translation")
        return self._translate.translate(text, source.value, target.value)


def translate_segments(
    segments: list[TranscriptSegment],
    target: Language,
    translator: Translator | None = None,
) -> list[TranscriptSegment]:
    engine = translator or UnavailableTranslator()
    translated = []
    for segment in segments:
        translated.append(
            TranscriptSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=engine.translate(segment.text, segment.language, target),
                language=target,
                confidence=segment.confidence,
            )
        )
    return translated
