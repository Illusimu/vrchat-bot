param(
    [Parameter(Mandatory)]
    [ValidateSet('vision', 'decision')]
    [string]$Role,
    [ValidateRange(0,65535)]
    [int]$Port = 0,
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$directory = Join-Path $PSScriptRoot 'build/liquid-omni'
$runtime = if ($Cpu) { 'cpu' } else { 'cuda' }
$server = Join-Path $directory "$runtime/llama-server.exe"
if ($Role -eq 'vision') {
    $model = 'LFM3-Omni-2B-Q4_K_M.gguf'
    if (!$Port) { $Port = 8098 }
    $batch = 512
} else {
    $model = 'd1-Omni-2B-Q8_0.gguf'
    if (!$Port) { $Port = 8099 }
    $batch = 8192
}
$weights = Join-Path $directory $model
$projector = Join-Path $directory "mmproj-$model"
foreach ($asset in @($server, $weights, $projector)) {
    if (!(Test-Path -LiteralPath $asset -PathType Leaf)) { throw "Missing Liquid Omni asset: $asset" }
}
if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw 'FFmpeg must be on PATH for video input.'
}

$gpuLayers = if ($Cpu) { 0 } else { 99 }
$launchArguments = @(
    '-m', $weights, '--mmproj', $projector, '--host', '127.0.0.1', '--port', $Port,
    '--alias', $Role, '-c', 8192, '-np', 1, '-b', $batch, '-ub', $batch, '-ngl', $gpuLayers
)
if ($Cpu) { $launchArguments += '--no-mmproj-offload' }
& $server @launchArguments
exit $LASTEXITCODE
