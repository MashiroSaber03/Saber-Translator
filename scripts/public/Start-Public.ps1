param(
    [string]$InstallRoot = 'D:\Saber-Translator',
    [int]$Port = 5100
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $InstallRoot 'venv\Scripts\python.exe'
$appRoot = Join-Path $InstallRoot 'app'
$entrypoint = Join-Path $appRoot 'saber_v2.py'
$dataRoot = Join-Path $InstallRoot 'data-public'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
    throw "Application entrypoint not found: $entrypoint"
}

New-Item -ItemType Directory -Path $dataRoot -Force | Out-Null
Set-Location -LiteralPath $appRoot
& $python $entrypoint `
    --role launcher `
    --profile public `
    --data-dir $dataRoot `
    --host 127.0.0.1 `
    --port $Port `
    --no-browser `
    --log-level INFO
exit $LASTEXITCODE
