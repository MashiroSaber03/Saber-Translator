param(
    [string]$InstallRoot = 'D:\Saber-Translator'
)

$ErrorActionPreference = 'Stop'
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$startScript = Join-Path $InstallRoot 'app\scripts\public\Start-Public.ps1'
$backupScript = Join-Path $InstallRoot 'app\scripts\public\Backup-Public.ps1'
$powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

$startAction = New-ScheduledTaskAction `
    -Execute $powerShell `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$startScript`""
$startTrigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
Register-ScheduledTask `
    -TaskName 'Saber Translator Public' `
    -Action $startAction `
    -Trigger $startTrigger `
    -Principal $principal `
    -Settings $settings `
    -Description 'Start Saber Translator public profile on localhost:5100.' `
    -Force | Out-Null

$backupAction = New-ScheduledTaskAction `
    -Execute $powerShell `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$backupScript`""
$backupTrigger = New-ScheduledTaskTrigger -Daily -At '04:15'
Register-ScheduledTask `
    -TaskName 'Saber Translator Public Backup' `
    -Action $backupAction `
    -Trigger $backupTrigger `
    -Principal $principal `
    -Settings $settings `
    -Description 'Keep seven SQLite snapshots and one current asset mirror.' `
    -Force | Out-Null
