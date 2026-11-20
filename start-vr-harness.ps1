param([switch]$NoOpen)
$ErrorActionPreference = 'Stop'
$runtime = Join-Path $PSScriptRoot 'build/deepseek-harness'
$cli = Join-Path $runtime 'node_modules/@deepseek-ai/dsh/lib/bin.js'
if (!(Test-Path $cli)) {
    npm.cmd install --prefix $runtime --no-audit --no-fund --save-exact '@deepseek-ai/dsh@0.2.2-rc.5'
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
$env:DSH_HOME = Join-Path $runtime 'home'
$env:VR_AGENT_LOCAL_KEY = 'local'
$env:VR_PILOT_INPUTS = Join-Path $PSScriptRoot 'harness/pilot_inputs.py'
$options = @('web', '--patch', (Join-Path $PSScriptRoot 'harness/local.patch.yml'),
    '--patch', (Join-Path $PSScriptRoot 'harness/web.patch.yml'),
    '--patch', (Join-Path $PSScriptRoot 'harness/inputs.patch.yml'))
if ($NoOpen) { $options += '--no-open' }
& node $cli @options
exit $LASTEXITCODE
