#Requires -Version 7.0
param([string]$Database)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$tokenPath = Join-Path $workspace '.world-research-token'
$token = (Get-Content -LiteralPath $tokenPath -Raw).Trim() | ConvertTo-SecureString
$snapshot = Invoke-RestMethod `
    -Uri 'https://escapethroughreality.com/api/world-research/active' `
    -Method Post -Authentication Bearer -Token $token `
    -MaximumRedirection 0 -TimeoutSec 30

$snapshotPath = Join-Path ([System.IO.Path]::GetTempPath()) ('vrchat-worlds-' + [guid]::NewGuid() + '.json')
try {
    $snapshot | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $snapshotPath -Encoding utf8
    $arguments = @((Join-Path $PSScriptRoot 'collector.py'))
    if ($Database) { $arguments += @('--db', $Database) }
    $arguments += @('import-active', $snapshotPath)
    python @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Remove-Item -LiteralPath $snapshotPath -ErrorAction SilentlyContinue
}
