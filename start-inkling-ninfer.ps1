param(
    [ValidateRange(2048,1048576)][int]$Context = 32768,
    [ValidateSet('fp16', 'mxfp8', 'rk8v4', 'rk4v4')][string]$KvType = 'mxfp8',
    [ValidateSet('UD-Q2_K_XL', 'UD-IQ4_XS')][string]$Quantization = 'UD-Q2_K_XL',
    [ValidateSet('device', 'host')][string]$ExpertResidency = 'device',
    [ValidateSet('none', 'dspark')][string]$Speculation = 'dspark',
    [ValidateRange(1,7)][int]$DraftTokens = 2,
    [ValidateRange(0,12288)][int]$ExpertCacheMiB = 0,
    [ValidateRange(0,12288)][int]$HostCacheMiB = 2048,
    [switch]$NoPrefixReuse,
    [ValidateRange(0,8192)][int]$PrefillChunk = 0,
    [ValidateRange(1024,65535)][int]$Port = 8097
)

$ErrorActionPreference = 'Stop'
$directory = Join-Path $PSScriptRoot 'build/inkling-2-mini/ninfer'
$server = Join-Path $directory 'runtime/ninfer-windows-x64-cuda12.9-sm70-75-80-86-89-baef81615ada/ninfer-serve.exe'
$model = Join-Path $directory "Inkling-2-Mini-${Quantization}-ninfer-v3.ninfer"
$modelSizes = @{ 'UD-Q2_K_XL' = 10080361216; 'UD-IQ4_XS' = 16494910976 }
foreach ($asset in @($server, $model)) {
    if (!(Test-Path -LiteralPath $asset -PathType Leaf)) { throw "Missing NInfer asset: $asset" }
}
if ((Get-Item -LiteralPath $model).Length -ne $modelSizes[$Quantization]) { throw "Incomplete Inkling model: $model" }

$launchArguments = @(
    $model, '--host', '127.0.0.1', '--port', $Port, '--model-id', 'vr-agent',
    '--max-context', $Context, '--max-concurrency', '1', '--kv-dtype', $KvType,
    '--expert-residency', $ExpertResidency, '--vision', '--structured-output',
    '--spec', $Speculation
)
if ($Speculation -eq 'dspark') { $launchArguments += @('--draft-tokens', $DraftTokens) }
if ($ExpertCacheMiB) { $launchArguments += @('--expert-cache-mib', $ExpertCacheMiB) }
if ($PrefillChunk) { $launchArguments += @('--prefill-chunk', $PrefillChunk) }
if ($NoPrefixReuse) {
    $launchArguments += '--no-prefix-reuse'
} else {
    $launchArguments += @('--host-cache-mib', $HostCacheMiB)
}
& $server @launchArguments
exit $LASTEXITCODE
