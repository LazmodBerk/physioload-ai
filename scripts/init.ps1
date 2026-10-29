param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"

$Directories = @(
    "data/raw/fit3d", "data/raw/human36m", "data/raw/other",
    "data/metadata", "data/interim/static_frames", "data/processed", "data/splits",
    "models/checkpoints", "models/calibrators",
    "outputs/predictions", "outputs/metrics", "outputs/figures", "outputs/reports", "outputs/logs"
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $Directory) | Out-Null
}

if (-not (Test-Path (Join-Path $VenvDir "Scripts/python.exe"))) {
    py -3.11 -m venv $VenvDir
}

$Python = Join-Path $VenvDir "Scripts/python.exe"
if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    & $Python -m pip install -e $ProjectRoot
    & $Python -m ipykernel install --user --name physioload --display-name "Python (physioload)"
}

Write-Host "PhysioLoad environment is ready."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"

