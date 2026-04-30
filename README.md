# Face Blur YuNet

Blur faces in a video with OpenCV's small YuNet face detector.

The script detects faces frame by frame, blurs only the detected face area with an oval mask, and keeps the original audio track in the output video.

## Requirements

- Python 3.9+
- `ffmpeg` available on your `PATH`

## Dashboard Setup

Create a virtual environment and install the dashboard dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the local dashboard server:

```bash
python3 -m uvicorn face_blur_yunet.app:create_default_app --factory --reload
```

The dashboard stores working files locally under ignored project directories.

## Optional Transcription Backend

Install `faster-whisper` to enable local transcription support:

```bash
pip install faster-whisper
```

## Optional Translation Backend

Install `argostranslate` to enable local translation support:

```bash
pip install argostranslate
```

Hebrew and English language packages must also be installed locally for translation.

## Usage

The existing face blur CLI remains compatible:

```bash
python blur_faces.py input.mp4 output_blurred.mp4
```

The YuNet ONNX model is downloaded automatically on first run into `models/`.

For stricter detection, increase the threshold:

```bash
python blur_faces.py input.mp4 output_blurred.mp4 --score-threshold 0.85
```

For more sensitive detection, lower it:

```bash
python blur_faces.py input.mp4 output_blurred.mp4 --score-threshold 0.6
```

Lower thresholds may catch more side faces or small faces, but can also blur non-face areas.

## Notes

Always review the output before sharing. Automatic face detection can miss faces, especially if they are tiny, heavily rotated, covered, or motion blurred.

Files stay local unless you configure a future cloud provider.

## Model

This project uses OpenCV's YuNet face detector:

- OpenCV Zoo model: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- OpenCV FaceDetectorYN docs: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
