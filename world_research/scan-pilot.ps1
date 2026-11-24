#Requires -Version 7.0
param(
    [string]$OutputPath = "world-pilot-$((Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')).ndjson",
    [ValidateRange(1, 1000000)][int]$ActivityFloor,
    [ValidateSet('active', 'games', 'hangouts', 'social', 'published')][string[]]$Pools = @('active', 'games', 'social', 'published')
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'hosted-origin.ps1')
$baseUri = Get-WorldResearchBaseUri
$workspace = Split-Path -Parent $PSScriptRoot
$token = (Get-Content -LiteralPath (Join-Path $workspace '.world-research-token') -Raw).Trim() | ConvertTo-SecureString
if (Test-Path -LiteralPath $OutputPath) { throw 'Choose a new output path for each pilot.' }
$queries = foreach ($pool in ($Pools | Select-Object -Unique)) {
    "${pool}:1"
    if ($pool -ne 'published') { "${pool}:2" }
}
$skipSecond = @{}
$nextRequest = [DateTimeOffset]::UtcNow
Write-Output "Writing live pages to $OutputPath"

foreach ($query in $queries) {
    $pool, $page = $query.Split(':')
    if ($page -eq '2' -and $skipSecond[$pool]) {
        Write-Output "Skipped $pool page 2 after evaluating page 1."
        continue
    }
    $uri = "$baseUri/page?pool=$pool&page=$page"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $delay = ($nextRequest - [DateTimeOffset]::UtcNow).TotalSeconds
        if ($delay -gt 0) { Start-Sleep -Seconds ([Math]::Ceiling($delay)) }
        $result = Invoke-RestMethod -Uri $uri -Method Post -Authentication Bearer -Token $token `
            -MaximumRedirection 0 -TimeoutSec 30 -SkipHttpErrorCheck -StatusCodeVariable status
        $nextRequest = [DateTimeOffset]::UtcNow.AddSeconds(63 + (Get-Random -Maximum 5))
        if ($status -eq 200) {
            $line = $result | ConvertTo-Json -Depth 10 -Compress
            Add-Content -LiteralPath $OutputPath -Value $line -Encoding utf8
            $size = [Text.Encoding]::UTF8.GetByteCount($line)
            Write-Output "$pool page $page`: $($result.worlds.Count) worlds, $size bytes"
            if ($ActivityFloor -and $page -eq '1' -and $pool -ne 'published') {
                $tail = @($result.worlds | Select-Object -Last 20)
                $known = @($tail | Where-Object { $null -ne $_.occupants })
                $above = @($known | Where-Object { $_.occupants -ge $ActivityFloor })
                $skipSecond[$pool] = $result.worlds.Count -lt $result.limit -or (
                    $tail.Count -eq 20 -and $known.Count -eq 20 -and $above.Count -lt 10
                )
            }
            break
        }
        if ($status -ne 429 -or $attempt -eq 3) {
            throw "VRChat scan stopped at $pool page $page (HTTP $status). $($result.error)"
        }
        Write-Output 'The shared Worker rate limit is busy. Retrying after the next minute.'
    }
}
