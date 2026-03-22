$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $repoRoot "logs\system_monitor"
$logDir = Join-Path $outDir (Get-Date -Format "yyyyMMdd")
$pidFile = Join-Path $logDir "system_monitor.pid"

if (!(Test-Path $pidFile)) {
  Write-Output "pid_file_not_found=$pidFile"
  exit 1
}

$pidText = Get-Content -Path $pidFile -Raw
$pid = [int]$pidText.Trim()

try {
  Stop-Process -Id $pid -Force
  Write-Output "stopped pid=$pid"
} catch {
  Write-Output "stop_failed pid=$pid error=$($_.Exception.Message)"
  exit 1
}
