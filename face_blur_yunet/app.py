from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from face_blur_yunet.jobs import JobStore, StoredJob
from face_blur_yunet.models import JobOptions, Language, QuestionAnswer
from face_blur_yunet.pipeline import Pipeline
from face_blur_yunet.question_answering import answer_from_chunks
from face_blur_yunet.transcript_index import load_index


def create_app(base_dir: Path) -> FastAPI:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(base_dir / "jobs.sqlite")
    app = FastAPI(title="Local Video Processing Dashboard")

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/jobs")
    def create_job(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        input_path = payload.get("input_path")
        if not input_path:
            raise HTTPException(status_code=422, detail="input_path is required")

        options = JobOptions(
            source_language=_language_or_default(payload.get("source_language"), Language.AUTO),
            translation_target=_language_or_none(payload.get("translation_target")),
            subtitles=bool(payload.get("subtitles", True)),
            face_blur=bool(payload.get("face_blur", False)),
        )
        job = store.create_job(Path(input_path), base_dir / "outputs", options)
        return _job_to_dict(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int) -> dict[str, Any]:
        return _job_to_dict(_get_job_or_404(store, job_id))

    @app.post("/api/jobs/{job_id}/run")
    def run_job(job_id: int) -> dict[str, Any]:
        _get_job_or_404(store, job_id)
        Pipeline(store).run(job_id)
        return _job_to_dict(_get_job_or_404(store, job_id))

    @app.post("/api/jobs/{job_id}/questions")
    def ask_question(job_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        job = _get_job_or_404(store, job_id)
        index_path_value = job.artifacts.get("transcript_index")
        if index_path_value is None:
            raise HTTPException(status_code=404, detail="transcript index artifact not found")

        index_path = Path(index_path_value)
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="transcript index artifact not found")

        question = payload.get("question")
        if not question:
            raise HTTPException(status_code=422, detail="question is required")

        answer_language = _language_or_default(payload.get("answer_language"), job.options.source_language)
        answer = answer_from_chunks(str(question), load_index(index_path), answer_language)
        store.add_question_answer(job.id, answer)
        return _question_answer_to_dict(answer)

    @app.get("/")
    def dashboard() -> HTMLResponse:
        static_path = base_dir / "dashboard.html"
        if static_path.exists():
            return HTMLResponse(static_path.read_text(encoding="utf-8"))
        return HTMLResponse("<!doctype html><title>Local Video Dashboard</title><h1>Local Video Dashboard</h1>")

    return app


def create_default_app() -> FastAPI:
    return create_app(Path("data"))


def _get_job_or_404(store: JobStore, job_id: int) -> StoredJob:
    try:
        return store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


def _language_or_default(value: Any, default: Language) -> Language:
    if value is None:
        return default
    try:
        return Language(str(value))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unsupported language: {value}") from exc


def _language_or_none(value: Any) -> Language | None:
    if value is None:
        return None
    return _language_or_default(value, Language.AUTO)


def _job_options_to_dict(options: JobOptions) -> dict[str, Any]:
    data = asdict(options)
    data["source_language"] = options.source_language.value
    data["translation_target"] = options.translation_target.value if options.translation_target is not None else None
    return data


def _job_to_dict(job: StoredJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "input_path": str(job.input_path),
        "output_dir": str(job.output_dir),
        "options": _job_options_to_dict(job.options),
        "status": job.status.value,
        "error": job.error,
        "artifacts": {name: str(path) for name, path in job.artifacts.items()},
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


def _question_answer_to_dict(question_answer: QuestionAnswer) -> dict[str, Any]:
    return {
        "question": question_answer.question,
        "answer": question_answer.answer,
        "language": question_answer.language.value,
        "timestamps": question_answer.timestamps,
        "excerpts": question_answer.excerpts,
        "grounded": question_answer.grounded,
    }
