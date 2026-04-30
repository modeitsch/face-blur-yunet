from face_blur_yunet.models import Language, TranscriptChunk
from face_blur_yunet.question_answering import answer_from_chunks


def chunks():
    return [
        TranscriptChunk(
            id=1,
            text="The price is 500 dollars and includes setup.",
            segment_ids=[1],
            start=10,
            end=20,
            language=Language.ENGLISH,
        ),
        TranscriptChunk(
            id=2,
            text="Delivery will happen next Monday.",
            segment_ids=[2],
            start=30,
            end=40,
            language=Language.ENGLISH,
        ),
    ]


def test_answer_from_chunks_returns_grounded_excerpt():
    answer = answer_from_chunks("What is the price?", chunks(), Language.ENGLISH)
    assert answer.grounded is True
    assert "500 dollars" in answer.answer
    assert answer.timestamps == [(10, 20)]


def test_answer_from_chunks_does_not_guess():
    answer = answer_from_chunks("What is the warranty?", chunks(), Language.ENGLISH)
    assert answer.grounded is False
    assert "I could not find" in answer.answer


def test_answer_from_chunks_rejects_partial_token_overlap():
    answer = answer_from_chunks(
        "What is the setup warranty?", chunks(), Language.ENGLISH
    )
    assert answer.grounded is False
    assert "I could not find" in answer.answer


def test_answer_from_chunks_reports_excerpt_language():
    answer = answer_from_chunks("What is the price?", chunks(), Language.HEBREW)
    assert answer.grounded is True
    assert answer.language == Language.ENGLISH
