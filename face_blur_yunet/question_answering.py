from __future__ import annotations

import re
from collections import Counter

from face_blur_yunet.models import Language, QuestionAnswer, TranscriptChunk

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "did",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "they",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "את",
    "על",
    "של",
    "מה",
    "מי",
    "מתי",
    "איפה",
    "איך",
    "האם",
    "זה",
    "זו",
    "עם",
    "לא",
    "כן",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[\w\u0590-\u05ff]+", text)
        if token.lower() not in STOPWORDS
    ]


def rank_chunks(
    question: str, chunks: list[TranscriptChunk], limit: int = 3
) -> list[TranscriptChunk]:
    question_terms = Counter(tokenize(question))
    scored: list[tuple[int, TranscriptChunk]] = []
    for chunk in chunks:
        chunk_terms = Counter(tokenize(chunk.text))
        score = sum(
            min(count, chunk_terms.get(term, 0))
            for term, count in question_terms.items()
        )
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].start))
    return [chunk for _, chunk in scored[:limit]]


def answer_from_chunks(
    question: str, chunks: list[TranscriptChunk], answer_language: Language
) -> QuestionAnswer:
    matches = rank_chunks(question, chunks, limit=2)
    if not matches:
        fallback = "I could not find enough information in the transcript to answer that."
        if answer_language == Language.HEBREW:
            fallback = "לא מצאתי מספיק מידע בתמלול כדי לענות על השאלה."
        return QuestionAnswer(
            question=question,
            answer=fallback,
            language=answer_language,
            grounded=False,
        )

    excerpts = [chunk.text for chunk in matches]
    timestamps = [(chunk.start, chunk.end) for chunk in matches]
    answer = "\n\n".join(excerpts)
    return QuestionAnswer(
        question=question,
        answer=answer,
        language=answer_language,
        timestamps=timestamps,
        excerpts=excerpts,
        grounded=True,
    )
