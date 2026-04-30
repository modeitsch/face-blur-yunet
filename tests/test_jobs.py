from face_blur_yunet.jobs import JobStore
from face_blur_yunet.models import JobOptions, JobStatus, Language, QuestionAnswer


def test_job_store_creates_and_updates_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(tmp_path / "input.mp4", tmp_path / "outputs", JobOptions(source_language=Language.ENGLISH))
    assert job.status == JobStatus.QUEUED
    store.update_status(job.id, JobStatus.COMPLETE)
    loaded = store.get_job(job.id)
    assert loaded.status == JobStatus.COMPLETE


def test_job_store_records_question_history(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(tmp_path / "input.mp4", tmp_path / "outputs", JobOptions())
    store.add_question_answer(
        job.id,
        QuestionAnswer(
            question="price?",
            answer="500",
            language=Language.ENGLISH,
            timestamps=[(1, 2)],
            excerpts=["price 500"],
        ),
    )
    history = store.list_question_history(job.id)
    assert history[0].answer == "500"
