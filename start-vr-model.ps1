$ErrorActionPreference = 'Stop'
$runtime = Join-Path $PSScriptRoot 'build/llama-cuda/llama-server.exe'
$models = Join-Path $PSScriptRoot 'build/agent-model'
$model = Get-Item (Join-Path $models '*MAX-MTP-Q4_K_M.gguf')
$projector = Get-Item (Join-Path $models 'mmproj-F16.gguf')
$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + $env:Path
$gpuRows = nvidia-smi --query-gpu=name,uuid --format=csv,noheader
$target = @($gpuRows | Where-Object { $_ -match 'Tesla V100S' })
if ($target.Count -ne 1) { throw 'Expected one Tesla V100S.' }
$env:CUDA_VISIBLE_DEVICES = ($target[0] -split ',')[-1].Trim()
& $runtime -m $model.FullName --mmproj $projector.FullName --alias vr-agent --host 127.0.0.1 --port 8097 -c 262144 -ngl 99 --parallel 1 -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 128
exit $LASTEXITCODE
