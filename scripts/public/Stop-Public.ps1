param(
    [string]$InstallRoot = 'D:\Saber-Translator',
    [int]$Port = 5100
)

$ErrorActionPreference = 'Stop'
$dataRoot = [System.IO.Path]::GetFullPath((Join-Path $InstallRoot 'data-public'))
$taskName = 'Saber Translator Public'

$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $task -and $task.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $taskName
}

function Get-PublicLauncherProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe', 'pythonw.exe') -and
        $_.CommandLine -match '--role\s+launcher(?:\s|$)' -and
        $_.CommandLine.IndexOf($dataRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    })
}

$launchers = Get-PublicLauncherProcesses
if ($launchers.Count -gt 0) {
    $launcherIds = @($launchers | ForEach-Object { [int]$_.ProcessId })
    $runtimeLauncher = @(
        $launchers | Where-Object { $launcherIds -contains [int]$_.ParentProcessId }
    )
    if ($runtimeLauncher.Count -eq 0 -and $launchers.Count -eq 1) {
        $runtimeLauncher = @($launchers[0])
    }
    if ($runtimeLauncher.Count -ne 1) {
        throw 'Could not identify exactly one public Launcher process.'
    }
    Stop-Process -Id $runtimeLauncher[0].ProcessId -Force
}

$deadline = (Get-Date).AddSeconds(15)
do {
    Start-Sleep -Milliseconds 250
    $remaining = Get-PublicLauncherProcesses
} while ($remaining.Count -gt 0 -and (Get-Date) -lt $deadline)

# A copied Windows venv uses a short-lived redirector process. It should exit
# with the real Launcher; clean up only redirectors that still match this data root.
foreach ($process in $remaining) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw "Port $Port is still listening after the public Launcher stopped."
}

Write-Output 'Saber Translator public profile stopped.'
