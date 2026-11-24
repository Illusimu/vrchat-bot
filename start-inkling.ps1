param(
    [ValidateSet('none', 'ngram-mod', 'draft-dspark')][string]$Speculation = 'none',
    [ValidateRange(2048,1048576)][int]$Context = 4096,
    [ValidateRange(0,99)][int]$GpuLayers = 99,
    [ValidateRange(0,24)][int]$CpuMoeLayers = 0,
    [ValidateRange(0,8192)][int]$MoeCacheMiB = 0,
    [ValidateRange(32,512)][int]$MicroBatch = 128,
    [ValidateSet('q8_0', 'q4_0')][string]$KvType = 'q8_0',
    [ValidateRange(1024,65535)][int]$Port = 8097,
    [ValidateSet('mainline', 'unsloth', 'ada', 'ada-mixb')][string]$Runtime = 'mainline'
)

$ErrorActionPreference = 'Stop'
$inklingDirectory = Join-Path $PSScriptRoot 'build/inkling-2-mini'
$server = Join-Path $inklingDirectory 'runtime/llama-server.exe'
$weights = Join-Path $inklingDirectory 'downloads/Inkling-2-Mini-UD-IQ3_XXS.gguf'
$weightSize = 11663310784
$projector = Join-Path $inklingDirectory 'downloads/mmproj-F16.gguf'
$draftArguments = @()
$draft = $null
$useUnsloth = $Runtime -eq 'unsloth' -or ($Runtime -eq 'mainline' -and $Speculation -eq 'draft-dspark')
$useAda = $Runtime -in 'ada', 'ada-mixb'
if ($useAda) {
    $server = Join-Path $inklingDirectory 'ada/runtime/llama-server.exe'
}
if ($Runtime -eq 'ada-mixb') {
    $weights = Join-Path $inklingDirectory 'ada/Inkling-2-Mini-mixB.gguf'
    $weightSize = 11043480096
}
if ($useUnsloth) {
    $server = Get-ChildItem (Join-Path $inklingDirectory 'dspark') -Recurse -Filter llama-server.exe |
        Select-Object -First 1 -ExpandProperty FullName
    $weights = Join-Path $inklingDirectory 'dspark/downloads/Inkling-2-Mini-UD-IQ3_XXS.gguf'
    $weightSize = 11663309504
}
if ($Speculation -eq 'draft-dspark') {
    $draft = Join-Path $inklingDirectory 'dspark/downloads/dspark-Inkling-2-Mini-Q8_0.gguf'
    $draftArguments = @('--model-draft', $draft, '--spec-draft-n-max', '3', '--n-gpu-layers-draft', $GpuLayers)
}
foreach ($asset in @($server, $weights, $projector) + @($draft | Where-Object { $_ })) {
    if (!(Test-Path -LiteralPath $asset -PathType Leaf)) { throw "Missing Inkling asset: $asset" }
}
if ((Get-Item -LiteralPath $weights).Length -ne $weightSize) { throw "Inkling model download is incomplete: $weights" }

$serverArguments = @(
    '-m', $weights, '--mmproj', $projector, '--alias', 'vr-agent',
    '--host', '127.0.0.1', '--port', $Port, '-c', $Context, '-ngl', $GpuLayers,
    '--parallel', '1', '-fa', 'on', '-ctk', $KvType, '-ctv', $KvType,
    '-b', [Math]::Max(384, $MicroBatch), '-ub', $MicroBatch, '--spec-type', $Speculation
)
if ($CpuMoeLayers) { $serverArguments += @('--n-cpu-moe', $CpuMoeLayers) }
if ($MoeCacheMiB) {
    if (!$CpuMoeLayers -or $useUnsloth -or $Runtime -ne 'mainline') {
        throw 'This launcher uses the mainline runtime and CPU expert offload for GPU expert caching.'
    }
    $serverArguments += @('--moe-cache-mib', $MoeCacheMiB)
}
if ($useUnsloth -or $useAda) {
    $env:Path = (Join-Path $inklingDirectory 'runtime') + ';' + $env:Path
}
& $server @serverArguments @draftArguments
exit $LASTEXITCODE
