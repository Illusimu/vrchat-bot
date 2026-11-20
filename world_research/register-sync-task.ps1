param([string]$Python = (Get-Command python.exe -ErrorAction Stop).Source)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$name = 'Escape Through Reality World History Sync'
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$pythonw = Join-Path (Split-Path -Parent $Python) 'pythonw.exe'
$program = (Resolve-Path -LiteralPath $pythonw).Path
$sync = Join-Path $PSScriptRoot 'hosted_sync.py'
$log = Join-Path $workspace 'world-history-sync.log'
$action = New-ScheduledTaskAction -Execute $program -Argument ('"{0}" --log "{1}"' -f $sync, $log) `
    -WorkingDirectory $workspace
$trigger = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $owner),
    (New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(2)) -RepetitionInterval (New-TimeSpan -Minutes 30))
)
$principal = New-ScheduledTaskPrincipal -UserId $owner -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName $name -Description 'Import pending VRChat world scans into local SQLite.' `
    -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State
