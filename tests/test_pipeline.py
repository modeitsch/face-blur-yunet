from face_blur_yunet.jobs import JobStore
from face_blur_yunet.models import JobOptions, JobStatus, Language
from face_blur_yunet.pipeline import Pipeline, PipelineEngines
from face_blur_yunet.transcription import FakeTranscriber
from face_blur_yunet.translation import FakeTranslator


def test_pipeline_writes_transcript_srt_index_and_answers(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        input_path,
        tmp_path / "outputs",
        JobOptions(source_language=Language.ENGLISH, translation_target=Language.HEBREW),
    )

    monkeypatch.setattr("face_blur_yunet.pipeline.probe_media", lambda path: type("Info", (), {"has_audio": True})())
    monkeypatch.setattr("face_blur_yunet.pipeline.extract_audio", lambda src, dst: dst.write_bytes(b"audio") or dst)

    pipeline = Pipeline(
        store,
        PipelineEngines(
            transcriber=FakeTranscriber(["price is 500"]),
            translator=FakeTranslator({"price is 500": "המחיר הוא 500"}),
        ),
    )
    pipeline.run(job.id)

    loaded = store.get_job(job.id)
    assert loaded.status == JobStatus.COMPLETE
    assert (tmp_path / "outputs" / f"job-{job.id}" / "transcript.en.txt").exists()
    assert (tmp_path / "outputs" / f"job-{job.id}" / "subtitles.en.srt").exists()
    assert (tmp_path / "outputs" / f"job-{job.id}" / "transcript-index.json").exists()
    assert (tmp_path / "outputs" / f"job-{job.id}" / "transcript.he.txt").exists()
    assert (tmp_path / "outputs" / f"job-{job.id}" / "subtitles.he.srt").exists()


def test_pipeline_writes_source_artifacts_for_translation_only_job(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        input_path,
        tmp_path / "outputs",
        JobOptions(
            source_language=Language.ENGLISH,
            transcript=False,
            questions=False,
            subtitles=False,
            translation_target=Language.HEBREW,
        ),
    )

    monkeypatch.setattr("face_blur_yunet.pipeline.probe_media", lambda path: type("Info", (), {"has_audio": True})())
    monkeypatch.setattr("face_blur_yunet.pipeline.extract_audio", lambda src, dst: dst.write_bytes(b"audio") or dst)

    pipeline = Pipeline(
        store,
        PipelineEngines(
            transcriber=FakeTranscriber(["price is 500"]),
            translator=FakeTranslator({"price is 500": "המחיר הוא 500"}),
        ),
    )
    pipeline.run(job.id)

    loaded = store.get_job(job.id)
    output_dir = tmp_path / "outputs" / f"job-{job.id}"
    assert loaded.status == JobStatus.COMPLETE
    assert (output_dir / "transcript.en.txt").exists()
    assert (output_dir / "subtitles.en.srt").exists()
    assert (output_dir / "transcript-index.json").exists()
    assert (output_dir / "transcript.he.txt").exists()
    assert (output_dir / "subtitles.he.srt").exists()
