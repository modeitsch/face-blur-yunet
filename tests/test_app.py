from fastapi.testclient import TestClient

from face_blur_yunet.app import create_app
from face_blur_yunet.jobs import JobStore
from face_blur_yunet.models import JobOptions, Language


def test_health_route(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}


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
