param(
    [string]$Python = (Get-Command python.exe -ErrorAction Stop).Source,
    [string]$TaskName = 'Escape Through Reality World History Sync',
    [ValidatePattern('^https://[a-zA-Z0-9.-]+(?::[0-9]+)?/?$')][string]$Origin,
    [switch]$SkipInvestigations
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$pythonw = Join-Path (Split-Path -Parent $Python) 'pythonw.exe'
$program = (Resolve-Path -LiteralPath $pythonw).Path
$sync = Join-Path $PSScriptRoot 'hosted_sync.py'
$log = Join-Path $workspace 'world-history-sync.log'
$taskArguments = '"{0}" --log "{1}"' -f $sync, $log
if ($Origin) { $taskArguments += ' --origin "{0}"' -f $Origin }
if ($SkipInvestigations) { $taskArguments += ' --skip-investigations' }
$action = New-ScheduledTaskAction -Execute $program -Argument $taskArguments `
    -WorkingDirectory $workspace
$trigger = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $owner),
    (New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(2)) -RepetitionInterval (New-TimeSpan -Minutes 30))
)
$principal = New-ScheduledTaskPrincipal -UserId $owner -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName $TaskName -Description 'Import pending VRChat world scans into local SQLite.' `
    -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
