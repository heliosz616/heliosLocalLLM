$ErrorActionPreference = 'Stop'
foreach ($Dir in @('reports','models','state')) { New-Item -ItemType Directory -Force "$PSScriptRoot\$Dir" | Out-Null }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv and rerun bootstrap.' }
if (-not (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe")) {
    uv venv "$PSScriptRoot\.venv" --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
$Python = "$PSScriptRoot\.venv\Scripts\python.exe"
uv pip install --python $Python -r "$PSScriptRoot\requirements.lock.txt"
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (-not (Test-Path "$PSScriptRoot\models\manifest.json")) {
    Copy-Item "$PSScriptRoot\model.source.json" "$PSScriptRoot\models\manifest.json"
}
# Download overlaps source preparation and compilation. Failed downloads remain resumable.
$Download = Start-Process $Python -ArgumentList ('"'+$PSScriptRoot+'\scripts\download_model.py"') -WindowStyle Hidden -PassThru -RedirectStandardOutput "$PSScriptRoot\reports\download.log" -RedirectStandardError "$PSScriptRoot\reports\download-error.log"
try {
    & $Python "$PSScriptRoot\scripts\prepare_runtime.py"
    if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency preparation failed' }
    & "$PSScriptRoot\scripts\build_runtime.ps1"
} finally {
    $Download.WaitForExit()
}
if ($Download.ExitCode -ne 0) { throw 'Model download failed; rerun to resume' }
if (-not (Test-Path "$PSScriptRoot\config.local.json")) {
    Copy-Item "$PSScriptRoot\config.example.json" "$PSScriptRoot\config.local.json"
}
Write-Host 'Prepared. Review config.local.json, then run qwen27b/start.ps1.'
