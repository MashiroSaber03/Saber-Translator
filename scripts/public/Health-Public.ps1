param(
    [int]$Port = 5100
)

$ErrorActionPreference = 'Stop'
$response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v2/health" -TimeoutSec 5
if ($response.status -ne 'ok' -or $response.role -ne 'api') {
    throw "Unexpected health response: $($response | ConvertTo-Json -Compress)"
}
$response | ConvertTo-Json
