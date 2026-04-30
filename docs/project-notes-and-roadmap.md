# Project Notes And Roadmap

## Current Position

This repository is now a local-first video processing dashboard for private client videos. It can create processing jobs from a local video path, transcribe Hebrew or English with a local faster-whisper backend when installed, translate Hebrew to English or English to Hebrew with a local Argos backend when installed, export transcripts and SRT subtitles, answer transcript-grounded questions, optionally blur faces, and save outputs under `data/outputs/job-<id>/`.

The repo is suitable to keep public as a portfolio/demo project because the tracked files are code, tests, and documentation only. Client videos, generated outputs, local databases, downloaded models, and common video files are ignored by git. Before making the repo public, keep checking that no real client media, transcripts, secrets, API tokens, or private business notes are committed.

## What Is Already Working

- Local FastAPI dashboard at `http://127.0.0.1:8000`.
- Clean web client for creating jobs, running processing, viewing output paths, and asking questions.
- Local transcription integration through `faster-whisper` when available.
- Local translation integration through `argostranslate` when Hebrew/English packages are installed.
- Optional YuNet face blurring.
- Output folder per job with source video copy, transcripts, subtitles, indexes, processing report, and blurred video when requested.
- Tests for request validation, pipeline artifacts, subtitle export, translation, question answering, and face blur logic.

## Known Limits

- The dashboard has no login yet. It should only run on a trusted local machine or private network.
- Processing runs inside the web request, so long videos can keep the browser waiting until the job finishes.
- The question-answering system is deterministic and transcript-grounded, but it is not a full semantic search or LLM answer engine.
- Translation quality depends on the installed Argos language packages.
- Whisper model selection and hardware acceleration are not exposed in the UI yet.
- There is no built-in progress bar for long transcription or face blur jobs.
- The browser shows local output paths, but it does not open local files directly from the page.

## Recommended Improvements

1. Add a background job worker so the UI can show queued, running, complete, and failed jobs without waiting on one long HTTP request.
2. Add progress events for stages such as audio extraction, transcription, translation, subtitle export, and face blur.
3. Add a local jobs history page with filters by date, status, language, and client/project name.
4. Add configurable Whisper model size in the UI, such as `tiny`, `base`, `small`, `medium`, and `large-v3`, with clear speed/quality labels.
5. Add optional NVIDIA GPU settings for the Windows machine and Apple Silicon settings for the Mac.
6. Add direct download routes for output files instead of showing only local paths.
7. Add a local password or single-user auth gate before exposing the dashboard beyond `127.0.0.1`.
8. Improve question answering with embeddings or a local LLM while keeping the transcript excerpts visible for trust.
9. Add speaker diarization when the client video includes multiple speakers.
10. Add automatic subtitle burn-in as a separate option from plain SRT export.
11. Add a packaging script or Docker setup for repeatable install on Mac and Windows.
12. Add a public-facing README section with screenshots, privacy explanation, and example workflow.

## Public Repository Recommendation

Making this repository public can help attract people because it shows a clear practical tool: local transcription, translation, face privacy, and question answering for client videos. The privacy story is also a strong selling point because client media stays local and generated data is ignored by git.

Keep the repo public only if these rules stay true:

- No real client videos or transcripts are committed.
- No credentials, API keys, or private config files are committed.
- `data/`, `outputs/`, `work/`, `models/`, local databases, and media files stay ignored.
- The README says the dashboard is local-first and should not be exposed publicly without authentication.

If those rules are followed, public is a good default. If client-specific workflows, private datasets, or business-sensitive prompts are added later, move that work to a private branch or a private repo.
