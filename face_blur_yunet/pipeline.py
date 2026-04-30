from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from face_blur_yunet.face_blur import BlurOptions, blur_video
from face_blur_yunet.jobs import JobStore
from face_blur_yunet.media import extract_audio, probe_media
from face_blur_yunet.models import JobStatus, Language, TranscriptSegment
from face_blur_yunet.subtitles import write_srt, write_transcript
from face_blur_yunet.transcript_index import index_transcript
from face_blur_yunet.transcription import Transcriber, transcribe_audio
from face_blur_yunet.translation import Translator, translate_segments

FaceBlurFunc = Callable[[Path, Path, BlurOptions], dict[str, int]]


@dataclass(frozen=True)
class PipelineEngines:
    transcriber: Transcriber | None = None
    translator: Translator | None = None
    face_blur_func: FaceBlurFunc | None = None


class Pipeline:
    def __init__(self, store: JobStore, engines: PipelineEngines | None = None) -> None:
        self.store = store
        self.engines = engines or PipelineEngines()

    def run(self, job_id: int) -> None:
        job = self.store.get_job(job_id)
        output_dir = job.output_dir / f"job-{job.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        error: str | None = None

        try:
            self.store.update_status(job.id, JobStatus.VALIDATING)
            media_info = probe_media(job.input_path)
            if job.options.requires_audio and not media_info.has_audio:
                raise RuntimeError("Video has no audio stream")

            source_segments: list[TranscriptSegment] = []
            source_language = job.options.source_language
            if job.options.requires_audio:
                self.store.update_status(job.id, JobStatus.EXTRACTING_AUDIO)
                audio_path = extract_audio(job.input_path, output_dir / "audio.wav")
                self._set_artifact(job.id, "audio", audio_path)

                self.store.update_status(job.id, JobStatus.TRANSCRIBING)
                source_segments = transcribe_audio(audio_path, source_language, self.engines.transcriber)
                source_language = _segments_language(source_segments, source_language)

                transcript_path = write_transcript(
                    output_dir / f"transcript.{source_language.value}.txt",
                    source_segments,
                )
                self._set_artifact(job.id, "transcript", transcript_path)

                self.store.update_status(job.id, JobStatus.WRITING_SUBTITLES)
                srt_path = write_srt(
                    output_dir / f"subtitles.{source_language.value}.srt",
                    source_segments,
                )
                self._set_artifact(job.id, "subtitles", srt_path)

                self.store.update_status(job.id, JobStatus.INDEXING_TRANSCRIPT)
                index_path = index_transcript(source_segments, output_dir)
                self._set_artifact(job.id, "transcript_index", index_path)

                if job.options.translation_target is not None:
                    self.store.update_status(job.id, JobStatus.TRANSLATING)
                    translated_segments = translate_segments(
                        source_segments,
                        job.options.translation_target,
                        self.engines.translator,
                    )
                    translated_transcript_path = write_transcript(
                        output_dir / f"transcript.{job.options.translation_target.value}.txt",
                        translated_segments,
                    )
                    self._set_artifact(job.id, "translated_transcript", translated_transcript_path)
                    translated_srt_path = write_srt(
                        output_dir / f"subtitles.{job.options.translation_target.value}.srt",
                        translated_segments,
                    )
                    self._set_artifact(job.id, "translated_subtitles", translated_srt_path)

            if job.options.face_blur:
                self.store.update_status(job.id, JobStatus.BLURRING_FACES)
                blurred_video_path = output_dir / "video.face-blurred.mp4"
                face_blur_func = self.engines.face_blur_func or _blur_video
                face_blur_func(
                    job.input_path,
                    blurred_video_path,
                    BlurOptions(
                        score_threshold=job.options.score_threshold,
                        blur_strength=job.options.blur_strength,
                        face_padding=job.options.face_padding,
                    ),
                )
                self._set_artifact(job.id, "face_blurred_video", blurred_video_path)

            self.store.update_status(job.id, JobStatus.COMPLETE)
        except Exception as exc:
            error = str(exc)
            self.store.update_status(job.id, JobStatus.FAILED, error)
        finally:
            self._write_processing_report(job.id, output_dir, error)

    def _set_artifact(self, job_id: int, name: str, path: Path) -> None:
        self.store.set_artifact(job_id, name, path)

    def _write_processing_report(self, job_id: int, output_dir: Path, error: str | None) -> None:
        job = self.store.get_job(job_id)
        report_path = output_dir / "processing-report.json"
        payload = {
            "job_id": job.id,
            "status": job.status.value,
            "artifacts": job.artifacts,
            "error": job.error or error,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.set_artifact(job.id, "processing_report", report_path)


def _segments_language(segments: list[TranscriptSegment], fallback: Language) -> Language:
    for segment in segments:
        if segment.language != Language.AUTO:
            return segment.language
    return fallback


def _blur_video(input_path: Path, output_path: Path, options: BlurOptions) -> dict[str, int]:
    return blur_video(input_path, output_path, options=options)
