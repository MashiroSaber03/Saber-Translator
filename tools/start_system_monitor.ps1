$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $repoRoot "logs\system_monitor"
$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
$scriptPath = Join-Path $repoRoot "tools\system_monitor.py"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $outDir (Get-Date -Format "yyyyMMdd")
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $logDir "system_monitor.pid"

$argsList = @(
  $scriptPath,
  "--out-dir", $outDir
)

$p = Start-Process -FilePath $venvPython -ArgumentList $argsList -PassThru -WindowStyle Minimized
$p.Id | Out-File -FilePath $pidFile -Encoding ascii -Force

Write-Output "started pid=$($p.Id)"
Write-Output "pid_file=$pidFile"
