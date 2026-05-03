import json

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

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": True, "has_video": True})(),
    )
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
    assert loaded.artifacts["source_media"] == str(tmp_path / "outputs" / f"job-{job.id}" / "video.original.mp4")
    assert (tmp_path / "outputs" / f"job-{job.id}" / "video.original.mp4").exists()
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

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": True, "has_video": True})(),
    )
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


def test_pipeline_stores_artifact_contracts_for_translated_job(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        input_path,
        tmp_path / "outputs",
        JobOptions(source_language=Language.ENGLISH, translation_target=Language.HEBREW),
    )

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": True, "has_video": True})(),
    )
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
    assert set(loaded.artifacts) >= {
        "source_media",
        "source_video",
        "transcript",
        "subtitles",
        "transcript_index",
        "translated_transcript",
        "translated_subtitles",
        "processing_report",
    }


def test_pipeline_processing_report_includes_registered_report_artifact(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        input_path,
        tmp_path / "outputs",
        JobOptions(source_language=Language.ENGLISH, translation_target=Language.HEBREW),
    )

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": True, "has_video": True})(),
    )
    monkeypatch.setattr("face_blur_yunet.pipeline.extract_audio", lambda src, dst: dst.write_bytes(b"audio") or dst)

    pipeline = Pipeline(
        store,
        PipelineEngines(
            transcriber=FakeTranscriber(["price is 500"]),
            translator=FakeTranslator({"price is 500": "המחיר הוא 500"}),
        ),
    )
    pipeline.run(job.id)

    report_path = tmp_path / "outputs" / f"job-{job.id}" / "processing-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["job_id"] == job.id
    assert report["status"] == JobStatus.COMPLETE.value
    assert "processing_report" in report["artifacts"]


def test_pipeline_missing_audio_marks_job_failed(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(input_path, tmp_path / "outputs", JobOptions(source_language=Language.ENGLISH))

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": False, "has_video": True})(),
    )

    pipeline = Pipeline(store, PipelineEngines(transcriber=FakeTranscriber(["price is 500"])))
    pipeline.run(job.id)

    loaded = store.get_job(job.id)
    assert loaded.status == JobStatus.FAILED
    assert "Media has no audio stream" in loaded.error


def test_pipeline_face_blur_calls_injected_engine_and_stores_artifact(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        input_path,
        tmp_path / "outputs",
        JobOptions(transcript=False, questions=False, subtitles=False, face_blur=True),
    )
    calls = []

    def fake_face_blur(src, dst, options):
        calls.append((src, dst, options))
        dst.write_bytes(b"blurred")
        return {"frames": 1, "detected_frames": 1, "faces": 1}

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": False, "has_video": True})(),
    )

    pipeline = Pipeline(store, PipelineEngines(face_blur_func=fake_face_blur))
    pipeline.run(job.id)

    loaded = store.get_job(job.id)
    artifact_path = tmp_path / "outputs" / f"job-{job.id}" / "video.face-blurred.mp4"
    assert loaded.status == JobStatus.COMPLETE
    assert loaded.artifacts["source_video"] == str(tmp_path / "outputs" / f"job-{job.id}" / "video.original.mp4")
    assert calls and calls[0][0] == input_path
    assert calls[0][1] == artifact_path
    assert loaded.artifacts["face_blurred_video"] == str(artifact_path)
    assert artifact_path.read_bytes() == b"blurred"


def test_pipeline_copies_source_video_to_output_folder(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"original video")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        input_path,
        tmp_path / "outputs",
        JobOptions(transcript=False, questions=False, subtitles=False),
    )

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": False, "has_video": True})(),
    )

    Pipeline(store).run(job.id)

    loaded = store.get_job(job.id)
    output_video = tmp_path / "outputs" / f"job-{job.id}" / "video.original.mov"
    assert loaded.status == JobStatus.COMPLETE
    assert loaded.artifacts["source_video"] == str(output_video)
    assert loaded.artifacts["source_media"] == str(output_video)
    assert output_video.read_bytes() == b"original video"


def test_pipeline_processes_mp3_audio_input(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp3"
    input_path.write_bytes(b"original audio")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(input_path, tmp_path / "outputs", JobOptions(source_language=Language.ENGLISH))

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": True, "has_video": False})(),
    )
    monkeypatch.setattr("face_blur_yunet.pipeline.extract_audio", lambda src, dst: dst.write_bytes(b"audio") or dst)

    Pipeline(store, PipelineEngines(transcriber=FakeTranscriber(["audio only works"]))).run(job.id)

    loaded = store.get_job(job.id)
    output_audio = tmp_path / "outputs" / f"job-{job.id}" / "audio.original.mp3"
    assert loaded.status == JobStatus.COMPLETE
    assert loaded.artifacts["source_media"] == str(output_audio)
    assert loaded.artifacts["source_audio"] == str(output_audio)
    assert "source_video" not in loaded.artifacts
    assert output_audio.read_bytes() == b"original audio"
    assert (tmp_path / "outputs" / f"job-{job.id}" / "transcript.en.txt").exists()


def test_pipeline_downloads_youtube_url_before_processing(tmp_path, monkeypatch):
    downloaded_path = tmp_path / "downloaded.mp4"
    downloaded_path.write_bytes(b"downloaded video")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        "https://www.youtube.com/watch?v=abc123",
        tmp_path / "outputs",
        JobOptions(source_language=Language.ENGLISH),
    )
    downloads = []
    probed_paths = []
    extracted_sources = []

    def fake_downloader(url, output_dir):
        downloads.append((url, output_dir))
        return downloaded_path

    def fake_probe_media(path):
        probed_paths.append(path)
        return type("Info", (), {"has_audio": True, "has_video": True})()

    def fake_extract_audio(src, dst):
        extracted_sources.append(src)
        dst.write_bytes(b"audio")
        return dst

    monkeypatch.setattr("face_blur_yunet.pipeline.probe_media", fake_probe_media)
    monkeypatch.setattr("face_blur_yunet.pipeline.extract_audio", fake_extract_audio)

    Pipeline(
        store,
        PipelineEngines(
            transcriber=FakeTranscriber(["youtube works"]),
            media_downloader=fake_downloader,
        ),
    ).run(job.id)

    loaded = store.get_job(job.id)
    output_dir = tmp_path / "outputs" / f"job-{job.id}"
    assert loaded.status == JobStatus.COMPLETE
    assert downloads == [("https://www.youtube.com/watch?v=abc123", output_dir)]
    assert probed_paths == [downloaded_path]
    assert extracted_sources == [downloaded_path]
    assert loaded.artifacts["downloaded_media"] == str(downloaded_path)
    assert loaded.artifacts["source_media"] == str(output_dir / "video.original.mp4")
    assert (output_dir / "video.original.mp4").read_bytes() == b"downloaded video"


def test_pipeline_rejects_face_blur_for_audio_input(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp3"
    input_path.write_bytes(b"original audio")
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        input_path,
        tmp_path / "outputs",
        JobOptions(transcript=False, questions=False, subtitles=False, face_blur=True),
    )

    monkeypatch.setattr(
        "face_blur_yunet.pipeline.probe_media",
        lambda path: type("Info", (), {"has_audio": True, "has_video": False})(),
    )

    Pipeline(store).run(job.id)

    loaded = store.get_job(job.id)
    assert loaded.status == JobStatus.FAILED
    assert "Face blur requires a video file" in loaded.error
