from fastapi.testclient import TestClient

from face_blur_yunet.app import create_app
from face_blur_yunet.jobs import JobStore
from face_blur_yunet.models import JobOptions, Language


def test_health_route(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}


def test_create_job_preserves_false_boolean_options(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")

    response = client.post(
        "/api/jobs",
        json={
            "input_path": str(input_path),
            "source_language": "en",
            "translation_target": None,
            "subtitles": False,
            "face_blur": False,
        },
    )

    assert response.status_code == 200
    options = response.json()["options"]
    assert options["subtitles"] is False
    assert options["face_blur"] is False


def test_create_job_strips_wrapping_quotes_from_input_path(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    input_path = tmp_path / "input with spaces.mp4"
    input_path.write_bytes(b"fake")

    response = client.post(
        "/api/jobs",
        json={
            "input_path": f' "{input_path}" ',
            "source_language": "en",
        },
    )

    assert response.status_code == 200
    assert response.json()["input_path"] == str(input_path)


def test_create_job_rejects_invalid_boolean_options(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")

    response = client.post(
        "/api/jobs",
        json={
            "input_path": str(input_path),
            "face_blur": {},
        },
    )

    assert response.status_code == 422


def test_ask_question_route_uses_job_index(tmp_path):
    app = create_app(tmp_path)
    store = JobStore(tmp_path / "jobs.sqlite")
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    job = store.create_job(input_path, tmp_path / "outputs", JobOptions(source_language=Language.ENGLISH))
    job_dir = tmp_path / "outputs" / f"job-{job.id}"
    job_dir.mkdir(parents=True)
    (job_dir / "transcript-index.json").write_text(
        '{"chunks":[{"id":1,"text":"price is 500","segment_ids":[1],"start":1,"end":2,"language":"en"}]}',
        encoding="utf-8",
    )
    store.set_artifact(job.id, "transcript_index", job_dir / "transcript-index.json")

    client = TestClient(app)
    response = client.post(f"/api/jobs/{job.id}/questions", json={"question": "price?", "answer_language": "en"})
    assert response.status_code == 200
    assert "500" in response.json()["answer"]
