param([int]$Context = 0, [ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Model = '')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force "$PSScriptRoot\reports" | Out-Null
$Python = "$PSScriptRoot\.venv\Scripts\python.exe"
$ArgsList = @('"'+$PSScriptRoot+'\scripts\service.py"','run')
if ($Context -gt 0) { $ArgsList += @('--context', "$Context") }
if ($Model) { $ArgsList += @('--default-model', $Model) }
Start-Process -FilePath $Python -ArgumentList $ArgsList -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "$PSScriptRoot\reports\supervisor.log" -RedirectStandardError "$PSScriptRoot\reports\supervisor-error.log"
Write-Host 'Service starting. Inspect reports/runtime.log and reports/gateway.log for readiness.'
