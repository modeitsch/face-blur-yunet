from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


def is_youtube_url(value: str) -> bool:
    url = value.strip().strip("'\"")
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in YOUTUBE_HOSTS


def download_youtube_media(url: str, output_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Install yt-dlp to download YouTube media") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "youtube.original.%(ext)s")
    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url.strip().strip("'\"")])

    downloaded_files = sorted(output_dir.glob("youtube.original.*"))
    if not downloaded_files:
        raise RuntimeError("YouTube download did not produce a media file")
    return downloaded_files[0]
