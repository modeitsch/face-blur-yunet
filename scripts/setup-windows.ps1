Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

function Test-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "python")) {
    Write-Error "python was not found. Install Python 3.9 or newer and add it to PATH."
}

if (-not (Test-Command "ffmpeg")) {
    Write-Error "ffmpeg was not found. Install ffmpeg and add it to PATH."
}

python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$installWhisper = Read-Host "Install local transcription backend faster-whisper? [y/N]"
if ($installWhisper -match "^[Yy]$") {
    python -m pip install faster-whisper
}

$installArgos = Read-Host "Install local translation backend argostranslate? [y/N]"
if ($installArgos -match "^[Yy]$") {
    python -m pip install argostranslate
}

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Start the dashboard:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m uvicorn face_blur_yunet.app:create_default_app --factory --host 127.0.0.1 --port 8000"
Write-Host ""
Write-Host "Open:"
Write-Host "  http://127.0.0.1:8000"
