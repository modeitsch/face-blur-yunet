# Local Video Processing Dashboard Design

Date: 2026-04-30
Repository: modeitsch/face-blur-yunet
Status: Draft for review

## Goal

Build the existing face blur script into a private, local-first video processing tool for client videos. The first version is for admin-only use by the repository owner. It should transcribe videos in Hebrew or English, optionally translate the transcript to the other language, optionally create subtitle outputs, and optionally blur faces in the final video.

The tool should keep client videos local by default. Cloud AI APIs should not be required for version 1. The design should still leave clean extension points for future cloud adapters or a second worker machine.

## Non-Goals For Version 1

- No public upload portal.
- No client accounts, authentication, or permissions.
- No payment or billing workflow.
- No guaranteed fully automatic broadcast-quality dubbing.
- No multi-user job management.
- No training or fine-tuning custom AI models.

## Recommended Product Shape

Version 1 should be a local private dashboard with a simple job workflow:

1. Add one or more videos.
2. Choose the requested outputs for each video.
3. Start processing.
4. Review generated files in an output folder.

The dashboard should run locally on the Mac Apple Silicon machine first. The Windows NVIDIA PC should be supported later as a worker by keeping processing logic separate from the UI.

## User Workflow

### Create Job

The admin selects one or more local video files. Each video becomes a processing job with its own options.

Required job options:

- Source language: auto-detect, Hebrew, or English.
- Transcript: on or off.
- Translation: none, Hebrew to English, English to Hebrew, or auto to other language.
- Subtitles: none, SRT only, or burned into video when subtitle burn-in is enabled.
- Face blur: off or on.
- Dubbing: hidden or disabled by default until a local voice engine is configured.

### Process Job

The app shows job status:

- Queued
- Extracting audio
- Transcribing
- Translating
- Rendering subtitles
- Blurring faces
- Dubbing audio, if enabled
- Muxing final video
- Complete or failed

### Review Output

Each processed job writes to a dedicated output folder containing the selected artifacts.

Example output folder:

```text
outputs/client-video-name/
  original-metadata.json
  transcript.he.txt
  transcript.en.txt
  subtitles.he.srt
  subtitles.en.srt
  video.face-blurred.mp4
  video.subtitled.en.mp4
  video.face-blurred.subtitled.en.mp4
  processing-report.json
```

## Architecture

Use a modular pipeline behind the dashboard:

```text
Dashboard UI
  -> Job Store
  -> Processing Orchestrator
      -> Media Probe
      -> Audio Extractor
      -> Transcription Engine
      -> Translation Engine
      -> Subtitle Renderer
      -> Face Blur Engine
      -> Dubbing Engine, optional
      -> Video Muxer
```

### Dashboard UI

The dashboard owns only interaction and job visibility. It should not contain model-specific logic. It should call backend job APIs such as:

- `create_job(video_path, options)`
- `start_job(job_id)`
- `get_job_status(job_id)`
- `open_output_folder(job_id)`

A lightweight local web app is preferred because it works on both Mac and Windows and can evolve into a worker-based setup later.

### Job Store

Use SQLite for version 1 so job history, retries, and status updates are reliable without adding a separate database service.

Stored data:

- Job id
- Input path
- Output path
- Selected options
- Current status
- Progress details
- Error message, if any
- Generated artifact paths
- Created and completed timestamps

### Processing Orchestrator

The orchestrator runs each selected step in order and skips steps that are not needed. It should write intermediate files to a job working directory so failed jobs can be inspected or retried.

The orchestrator should expose each processing step as a small module with a clear input and output. This keeps the current face blur code reusable and prevents one large script from owning every responsibility.

### Media Probe And Audio Extraction

Use `ffmpeg` and `ffprobe` for media inspection, audio extraction, subtitle burn-in, and final muxing. Version 1 should require `ffmpeg` on the machine, matching the current repository requirement.

The app should validate input files before starting expensive model work:

- File exists
- File is readable
- File contains a video stream
- Audio stream exists when transcription, translation, or dubbing is requested
- Output folder can be written

### Transcription Engine

Use a local speech-to-text engine with Hebrew and English support. The implementation should hide the specific model behind an adapter so the project can choose the best local engine for each machine.

Adapter interface:

```text
transcribe(audio_path, source_language, output_dir) -> transcript segments
```

Transcript segments should include:

- Start time
- End time
- Original text
- Detected language, when available
- Confidence or quality metadata, when available

The first implementation can support one local backend. Later implementations can add a faster Apple Silicon path and a CUDA/NVIDIA path without changing the rest of the app.

### Translation Engine

Use a local translation engine behind an adapter. Version 1 should support Hebrew to English and English to Hebrew if a local model is installed. If the local translation model is missing, the UI should explain that translation is unavailable instead of failing late.

Adapter interface:

```text
translate(segments, source_language, target_language, output_dir) -> translated segments
```

Translation should preserve timestamps from the transcription segments so subtitles remain aligned.

### Subtitle Outputs

The subtitle module should generate `.srt` files from transcript or translated segments. Burned-in subtitles should be rendered with `ffmpeg` into a new video file after the SRT path is working.

Subtitle options:

- Original language SRT
- Translated language SRT
- Original language burned into video, milestone 1b
- Translated language burned into video, milestone 1b

For version 1, prefer simple readable subtitles over advanced styling.

### Face Blur Engine

Reuse the existing YuNet-based face detection and oval blur behavior, but move it behind a reusable engine interface.

Adapter interface:

```text
blur_faces(video_path, output_path, options) -> blur report
```

Face blur options:

- Detection score threshold
- Blur strength
- Face padding
- Oval mask on or off
- Preview/report mode later

The face blur output should preserve the original audio unless another pipeline step replaces it.

### Dubbing Engine

Dubbing should be treated as optional and experimental in the first design. The app should support the workflow conceptually, but the version 1 implementation should ship with dubbing disabled unless a local TTS backend passes a readiness check.

Dubbing pipeline:

1. Transcribe source speech.
2. Translate segments.
3. Generate target-language speech for each segment.
4. Align generated audio to the video timeline.
5. Mux dubbed audio into a new video.

Dubbing risks:

- Generated speech may not match the original timing.
- Multiple speakers are hard to preserve locally.
- Voice quality varies by local model.
- Hebrew voice generation may be harder than English depending on the chosen local engine.

Version 1 should expose dubbing only when a configured local TTS backend passes a basic readiness check.

## Output Composition Rules

The pipeline should compose selected outputs instead of forcing one final artifact.

Examples:

- Transcript only: extract audio -> transcribe -> write transcript.
- Translated SRT: extract audio -> transcribe -> translate -> write SRT.
- Face blur only: blur video -> mux original audio.
- Face blur plus translated subtitles: transcribe -> translate -> write SRT -> blur video -> burn translated subtitles.
- Dubbed and blurred video: transcribe -> translate -> generate dubbed audio -> blur video -> mux dubbed audio.

When several video outputs are selected, the app should avoid duplicate work by reusing intermediate artifacts.

## Error Handling

The app should fail clearly and preserve partial outputs where useful.

Expected errors:

- Missing `ffmpeg` or `ffprobe`.
- Unsupported video file.
- Video has no audio when transcription is requested.
- Local transcription model is missing.
- Local translation model is missing.
- Local dubbing engine is not configured.
- Face detector model download fails.
- Output path is not writable.

Every failed job should write a `processing-report.json` file with the failed step and error message.

## Privacy And File Handling

Client videos should stay on local disk. The app should not upload video, audio, transcript, or face data anywhere unless a future cloud provider is explicitly configured.

Recommended local folders:

```text
input/          optional watched/input folder
outputs/        final artifacts
work/           temporary per-job files
models/         local model files
```

Temporary files should be kept until the job completes. The user can choose later whether completed working files are automatically cleaned up.

## Testing Strategy

Version 1 should include focused tests around pipeline behavior rather than expensive full model inference.

Test areas:

- Backend option validation.
- Job status transitions.
- Transcript segment to `.srt` formatting.
- Output filename generation.
- Pipeline composition for selected options.
- Face blur box clamping and mask behavior.
- Error handling for missing audio or missing models.

A small sample media fixture can be used for optional integration tests, but it should not be required for fast unit tests.

## Migration From Current Repository

Current code should be split gradually:

- Keep `blur_faces.py` working while new modules are introduced.
- Move YuNet model download and face blur logic into a reusable package module.
- Add pipeline modules around transcription, translation, subtitles, and job orchestration.
- Add a local dashboard after the backend job flow works.

Suggested package shape:

```text
face_blur_yunet/
  __init__.py
  face_blur.py
  media.py
  jobs.py
  pipeline.py
  transcription.py
  translation.py
  subtitles.py
  dubbing.py
  reports.py
blur_faces.py
app.py
```

`blur_faces.py` can remain as a compatibility CLI that calls the new package.

## First Implementation Milestone

The first useful milestone should focus on dependable daily use without dubbing:

- Local web dashboard.
- Add video job.
- Persist job state in SQLite.
- Transcribe Hebrew or English locally.
- Export original-language `.txt` and `.srt`.
- Translate Hebrew to English or English to Hebrew when a local translation backend is available.
- Export translated `.txt` and `.srt`.
- Optional YuNet face blur.
- Processing report per job.

## Milestone 1b

- Burn original or translated subtitles into video.
- Add simple progress bars.
- Add batch folder processing.

## Later Milestones

- Add progress estimates based on observed processing speed.
- Add Windows NVIDIA worker mode.
- Add local dubbing backend.
- Add speaker labeling if a reliable local option is selected.
- Add preview clips for quality review.
- Add cloud provider adapters as opt-in fallback, if desired.

## Open Decisions

These decisions should be finalized before implementation planning:

1. Which local transcription backend to use first on Apple Silicon.
2. Which local translation backend to use first for Hebrew and English.
3. Whether subtitle burn-in is included immediately after SRT export or reserved for milestone 1b.
4. Whether the Windows NVIDIA PC is configured in version 1 or reserved for a later worker milestone.

## Recommended Version 1 Decision

Given the intended admin-only workflow, the preferred design is:

- Build backend pipeline modules first.
- Keep the current face blur CLI working.
- Add a minimal local web dashboard on top of the backend.
- Ship transcription, transcript export, SRT export, translation if local backend readiness is good, and optional face blur.
- Treat subtitle burn-in as milestone 1b unless it is trivial after SRT generation.
- Treat dubbing and remote worker support as later milestones.
