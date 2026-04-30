from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from face_blur_yunet.models import JobOptions, JobStatus, Language, QuestionAnswer


@dataclass(frozen=True)
class StoredJob:
    id: int
    input_path: Path
    output_dir: Path
    options: JobOptions
    status: JobStatus
    error: str | None
    artifacts: dict[str, str]
    created_at: str
    completed_at: str | None


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def create_job(self, input_path: str | Path, output_dir: str | Path, options: JobOptions) -> StoredJob:
        created_at = _now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO jobs (input_path, output_dir, options_json, status, artifacts_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(input_path),
                    str(output_dir),
                    _options_to_json(options),
                    JobStatus.QUEUED.value,
                    "{}",
                    created_at,
                ),
            )
            job_id = cursor.lastrowid
        if job_id is None:
            raise RuntimeError("SQLite did not return a job id")
        return self.get_job(job_id)

    def get_job(self, job_id: int) -> StoredJob:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _row_to_job(row)

    def update_status(self, job_id: int, status: JobStatus, error: str | None = None) -> None:
        completed_at = _now_iso() if status in {JobStatus.COMPLETE, JobStatus.FAILED} else None
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, completed_at = ? WHERE id = ?",
                (status.value, error, completed_at, job_id),
            )

    def set_artifact(self, job_id: int, name: str, path: str | Path) -> None:
        job = self.get_job(job_id)
        artifacts = dict(job.artifacts)
        artifacts[name] = str(path)
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET artifacts_json = ? WHERE id = ?", (json.dumps(artifacts), job_id))

    def add_question_answer(self, job_id: int, question_answer: QuestionAnswer) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO question_history (
                    job_id, question, answer, language, timestamps_json, excerpts_json, grounded, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    question_answer.question,
                    question_answer.answer,
                    question_answer.language.value,
                    json.dumps(question_answer.timestamps),
                    json.dumps(question_answer.excerpts),
                    int(question_answer.grounded),
                    _now_iso(),
                ),
            )

    def list_question_history(self, job_id: int) -> list[QuestionAnswer]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, answer, language, timestamps_json, excerpts_json, grounded
                FROM question_history
                WHERE job_id = ?
                ORDER BY id
                """,
                (job_id,),
            ).fetchall()
        return [_row_to_question_answer(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  input_path TEXT NOT NULL,
                  output_dir TEXT NOT NULL,
                  options_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  error TEXT,
                  artifacts_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS question_history (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id INTEGER NOT NULL,
                  question TEXT NOT NULL,
                  answer TEXT NOT NULL,
                  language TEXT NOT NULL,
                  timestamps_json TEXT NOT NULL,
                  excerpts_json TEXT NOT NULL,
                  grounded INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(job_id) REFERENCES jobs(id)
                );
                """
            )


def _options_to_json(options: JobOptions) -> str:
    data = asdict(options)
    data["source_language"] = options.source_language.value
    data["translation_target"] = options.translation_target.value if options.translation_target is not None else None
    return json.dumps(data)


def _options_from_json(value: str) -> JobOptions:
    data = json.loads(value)
    data["source_language"] = Language(data["source_language"])
    if data["translation_target"] is not None:
        data["translation_target"] = Language(data["translation_target"])
    return JobOptions(**data)


def _row_to_job(row: sqlite3.Row) -> StoredJob:
    return StoredJob(
        id=row["id"],
        input_path=Path(row["input_path"]),
        output_dir=Path(row["output_dir"]),
        options=_options_from_json(row["options_json"]),
        status=JobStatus(row["status"]),
        error=row["error"],
        artifacts=json.loads(row["artifacts_json"]),
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


def _row_to_question_answer(row: sqlite3.Row) -> QuestionAnswer:
    timestamps = [tuple(timestamp) for timestamp in json.loads(row["timestamps_json"])]
    return QuestionAnswer(
        question=row["question"],
        answer=row["answer"],
        language=Language(row["language"]),
        timestamps=timestamps,
        excerpts=json.loads(row["excerpts_json"]),
        grounded=bool(row["grounded"]),
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
