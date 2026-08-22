param(
    [string]$InstallRoot = 'D:\Saber-Translator'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $InstallRoot 'venv\Scripts\python.exe'
$appRoot = Join-Path $InstallRoot 'app'
$dataRoot = Join-Path $InstallRoot 'data-public'
$backupRoot = Join-Path $InstallRoot 'backups'
$databaseBackup = Join-Path $appRoot 'scripts\public\backup_database.py'

if (-not (Test-Path -LiteralPath $databaseBackup -PathType Leaf)) {
    throw "Backup helper not found: $databaseBackup"
}

New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
& $python $databaseBackup --data-dir $dataRoot --backup-dir $backupRoot --keep 7
if ($LASTEXITCODE -ne 0) {
    throw "SQLite backup failed with exit code $LASTEXITCODE"
}

$sourceObjects = Join-Path $dataRoot 'objects'
$backupObjects = Join-Path $backupRoot 'objects-current'
New-Item -ItemType Directory -Path $sourceObjects -Force | Out-Null
New-Item -ItemType Directory -Path $backupObjects -Force | Out-Null
& robocopy $sourceObjects $backupObjects /MIR /COPY:DAT /DCOPY:T /R:2 /W:2 /NP /NFL /NDL /NJH /NJS
$robocopyCode = $LASTEXITCODE
if ($robocopyCode -gt 7) {
    throw "Asset mirror failed with robocopy exit code $robocopyCode"
}
