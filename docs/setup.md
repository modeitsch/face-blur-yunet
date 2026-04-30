# Setup Guide

This project runs locally on macOS and Windows. The dashboard itself is a FastAPI app, and media processing depends on `ffmpeg`.

## Supported Inputs

- Video files with audio, such as `.mp4`, `.mov`, `.mkv`, `.avi`, and `.webm`
- MP3 audio files

Face blur only applies to video files. MP3 files can be transcribed, translated, exported as text/SRT, and used for transcript questions.

## macOS Apple Silicon

Recommended setup:

```bash
brew install ffmpeg
./scripts/setup-macos.sh
source .venv/bin/activate
python3 -m uvicorn face_blur_yunet.app:create_default_app --factory --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The setup script:

- checks Python 3
- checks `ffmpeg`
- creates `.venv`
- upgrades `pip`
- installs project dependencies
- optionally installs `faster-whisper`
- optionally installs `argostranslate`

## Windows With NVIDIA GPU

Install these first:

- Python 3.9 or newer
- `ffmpeg`, added to `PATH`
- NVIDIA drivers
- CUDA/cuDNN if you want GPU acceleration for transcription

Then run PowerShell from the repository folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1
.\.venv\Scripts\Activate.ps1
python -m uvicorn face_blur_yunet.app:create_default_app --factory --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The Windows script uses the same dependency path as macOS. It does not install NVIDIA drivers, CUDA, cuDNN, or `ffmpeg`; those are machine-level tools and should be installed deliberately.

## Optional Local AI Backends

Transcription:

```bash
pip install faster-whisper
```

Translation:

```bash
pip install argostranslate
```

Argos translation also needs Hebrew and English language packages installed locally.

## Development Tools

Install development tools:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run Ruff checks if installed:

```bash
ruff check .
```

Check the dashboard JavaScript:

```bash
node --check face_blur_yunet/static/app.js
```

## Privacy Checklist

Before pushing to GitHub:

- Do not commit real client videos.
- Do not commit real client audio.
- Do not commit generated transcripts.
- Do not commit secrets or API keys.
- Keep `data/`, `outputs/`, `work/`, `models/`, local databases, and media files ignored.

## Common Problems

### `ffmpeg` Not Found

Install `ffmpeg` and make sure the command works:

```bash
ffmpeg -version
```

### Translation Backend Is Not Configured

Install `argostranslate` and the required Hebrew/English language packages.

### Transcription Backend Is Not Configured

Install `faster-whisper`.

### Face Blur Fails On MP3

This is expected. Face blur requires a video stream.
