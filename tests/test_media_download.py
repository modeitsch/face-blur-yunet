from pathlib import Path

import pytest

from face_blur_yunet.media_download import download_youtube_media, is_youtube_url


def test_is_youtube_url_accepts_youtube_hosts():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert is_youtube_url(" 'https://music.youtube.com/watch?v=abc' ")


def test_is_youtube_url_rejects_local_paths_and_other_urls():
    assert not is_youtube_url("/Users/me/video.mp4")
    assert not is_youtube_url("https://example.com/video.mp4")


def test_download_youtube_media_uses_ytdlp_and_returns_downloaded_file(tmp_path, monkeypatch):
    calls = []

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            calls.append(options)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"downloaded")
            calls.append(urls)

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYoutubeDL)

    output_path = download_youtube_media("https://youtu.be/abc", tmp_path)

    assert output_path == tmp_path / "youtube.original.mp4"
    assert output_path.read_bytes() == b"downloaded"
    assert calls[0]["format"] == "bv*+ba/b"
    assert calls[1] == ["https://youtu.be/abc"]


def test_download_youtube_media_raises_when_ytdlp_creates_no_file(tmp_path, monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):
            return None

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYoutubeDL)

    with pytest.raises(RuntimeError, match="YouTube download did not produce a media file"):
        download_youtube_media("https://youtu.be/abc", tmp_path)
