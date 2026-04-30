import json
import subprocess
from pathlib import Path

import pytest

from face_blur_yunet.media import extract_audio, probe_media


def test_probe_media_parses_ffprobe_json(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    def fake_run(cmd, check, capture_output, text):
        payload = {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "12.5"},
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = probe_media(video)
    assert info.width == 1920
    assert info.height == 1080
    assert round(info.fps, 2) == 29.97
    assert info.has_audio is True
    assert info.has_video is True


def test_probe_media_accepts_audio_only_file(monkeypatch, tmp_path):
    audio = tmp_path / "input.mp3"
    audio.write_bytes(b"fake")

    def fake_run(cmd, check, capture_output, text):
        payload = {
            "streams": [{"codec_type": "audio"}],
            "format": {"duration": "45.25"},
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = probe_media(audio)
    assert info.duration == 45.25
    assert info.width == 0
    assert info.height == 0
    assert info.fps == 0.0
    assert info.has_audio is True
    assert info.has_video is False


def test_extract_audio_runs_ffmpeg(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = extract_audio(tmp_path / "input.mp4", tmp_path / "audio.wav")
    assert output.name == "audio.wav"
    assert calls[0][0] == "ffmpeg"
    assert "-vn" in calls[0]
