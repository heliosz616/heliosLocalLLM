$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$State = Join-Path $Root 'state'
New-Item -ItemType Directory -Force $State | Out-Null
$Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
# Restrict only this project's secrets and conversation snapshots.
& icacls.exe $State /inheritance:r /grant:r "$($Identity):(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F'
if ($LASTEXITCODE -ne 0) { throw 'Could not restrict state directory ACL' }
