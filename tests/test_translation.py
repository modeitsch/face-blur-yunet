import pytest

from face_blur_yunet.models import Language, TranscriptSegment
from face_blur_yunet.translation import ArgosTranslator, FakeTranslator, UnavailableTranslator, translate_segments


def segments():
    return [TranscriptSegment(id=1, start=0, end=1, text="hello", language=Language.ENGLISH)]


def test_fake_translator_preserves_timestamps():
    translated = translate_segments(segments(), Language.HEBREW, FakeTranslator({"hello": "שלום"}))
    assert translated[0].text == "שלום"
    assert translated[0].start == 0
    assert translated[0].end == 1
    assert translated[0].language == Language.HEBREW


def test_unavailable_translator_raises_clear_error():
    with pytest.raises(RuntimeError, match="Translation backend is not configured"):
        translate_segments(segments(), Language.HEBREW, UnavailableTranslator())


def test_argos_translator_wraps_missing_language_package_error():
    class FakeArgos:
        def translate(self, text, source, target):
            raise ValueError("internal argos package error")

    translator = object.__new__(ArgosTranslator)
    translator._translate = FakeArgos()

    with pytest.raises(RuntimeError, match="Offline translation package is not available for this language pair") as exc:
        translator.translate("hello", Language.ENGLISH, Language.HEBREW)

    assert isinstance(exc.value.__cause__, ValueError)
