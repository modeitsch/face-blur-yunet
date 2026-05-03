from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class StringEnum(str, Enum):
    __str__ = str.__str__


class Language(StringEnum):
    AUTO = "auto"
    HEBREW = "he"
    ENGLISH = "en"


class JobStatus(StringEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    DOWNLOADING_MEDIA = "downloading_media"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    INDEXING_TRANSCRIPT = "indexing_transcript"
    TRANSLATING = "translating"
    WRITING_SUBTITLES = "writing_subtitles"
    BLURRING_FACES = "blurring_faces"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class JobOptions:
    source_language: Language = Language.AUTO
    transcript: bool = True
    questions: bool = True
    translation_target: Language | None = None
    subtitles: bool = True
    face_blur: bool = False
    score_threshold: float = 0.75
    blur_strength: int = 31
    face_padding: float = 0.0

    @property
    def requires_audio(self) -> bool:
        return self.transcript or self.questions or self.translation_target is not None


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    language: Language = Language.AUTO
    confidence: float | None = None

    @property
    def start_srt(self) -> str:
        return seconds_to_srt_time(self.start)

    @property
    def end_srt(self) -> str:
        return seconds_to_srt_time(self.end)


@dataclass(frozen=True)
class TranscriptChunk:
    id: int
    text: str
    segment_ids: list[int]
    start: float
    end: float
    language: Language


@dataclass(frozen=True)
class QuestionAnswer:
    question: str
    answer: str
    language: Language
    timestamps: list[tuple[float, float]] = field(default_factory=list)
    excerpts: list[str] = field(default_factory=list)
    grounded: bool = True


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool = True


def seconds_to_srt_time(value: float) -> str:
    milliseconds = int(round(value * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
