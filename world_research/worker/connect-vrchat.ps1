param([Parameter(Mandatory)][string]$Config)
$ErrorActionPreference = 'Stop'
$configPath = (Resolve-Path -LiteralPath $Config).Path
$project = Split-Path -Parent $configPath
$wrangler = Join-Path $project 'node_modules/wrangler/bin/wrangler.js'
$researchConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($researchConfig.name -ne 'vrchat-world-research' -or !$researchConfig.vars.VRCHAT_USER_AGENT) {
    throw 'Select the configured standalone World Research Worker before signing in.'
}
$env:WRANGLER_SEND_METRICS = 'false'
Add-Type -AssemblyName System.Net.Http
$researchCookies = New-Object System.Net.CookieContainer
$researchHandler = New-Object System.Net.Http.HttpClientHandler
$researchHandler.CookieContainer = $researchCookies
$researchHandler.AllowAutoRedirect = $false
$researchClient = New-Object System.Net.Http.HttpClient($researchHandler)
$researchClient.Timeout = [TimeSpan]::FromSeconds(25)
$researchClient.DefaultRequestHeaders.TryAddWithoutValidation('User-Agent', $researchConfig.vars.VRCHAT_USER_AGENT) | Out-Null
$researchClient.DefaultRequestHeaders.TryAddWithoutValidation('Accept', 'application/json') | Out-Null
$researchBase = 'https://api.vrchat.cloud/api/1'
function Read-ResearchSecret([string]$Prompt) {
    $researchSecure = Read-Host -Prompt $Prompt -AsSecureString
    $researchPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($researchSecure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($researchPointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($researchPointer); $researchSecure.Dispose() }
}
function Invoke-VRChatRequest([string]$Path, $Payload = $null) {
    try {
        if ($null -eq $Payload) { $researchResponse = $researchClient.GetAsync($researchBase + $Path).GetAwaiter().GetResult() }
        else {
            $researchBody = New-Object System.Net.Http.StringContent(($Payload | ConvertTo-Json -Compress), [Text.Encoding]::UTF8, 'application/json')
            try { $researchResponse = $researchClient.PostAsync($researchBase + $Path, $researchBody).GetAwaiter().GetResult() }
            finally { $researchBody.Dispose() }
        }
    } catch { throw 'VRChat connection failed. No credentials were uploaded. Try again later.' }
    try {
        if (-not $researchResponse.IsSuccessStatusCode) {
            $status = [int]$researchResponse.StatusCode
            if ($Path.StartsWith('/auth/twofactorauth/') -and $status -in @(400, 401)) {
                throw "VRChat declined the verification attempt (HTTP $status). Use a fresh code from the source named in the prompt."
            }
            throw "VRChat returned HTTP $status. Check login details or retry later."
        }
        return ($researchResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json)
    } finally { $researchResponse.Dispose() }
}
Write-Host 'Connect the dedicated backend VRChat account. Use its LOGIN USERNAME (not display name).'
Write-Host 'Password and verification codes stay in this process and are sent only to VRChat.'
Write-Host 'The resulting VRChat session is sent to the standalone Worker after you confirm the account.'
try {
    $researchUsername = Read-Host 'VRChat login username'
    $researchPassword = Read-ResearchSecret 'VRChat password'
    if ([string]::IsNullOrWhiteSpace($researchUsername) -or [string]::IsNullOrEmpty($researchPassword)) { throw 'Username and password are required.' }
    $researchBasic = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(([Uri]::EscapeDataString($researchUsername) + ':' + [Uri]::EscapeDataString($researchPassword))))
    $researchClient.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Basic', $researchBasic)
    $researchUser = Invoke-VRChatRequest '/auth/user'
    $researchClient.DefaultRequestHeaders.Authorization = $null
    $researchPassword = $null; $researchBasic = $null
    if ($researchUser.requiresTwoFactorAuth) {
        if ($researchUser.requiresTwoFactorAuth -contains 'totp') {
            $researchMethod = 'totp'
            Write-Host 'Open the authenticator app linked to this VRChat account. The app generates the code.'
            $researchPrompt = 'Current 6-digit code from your authenticator app'
        }
        elseif ($researchUser.requiresTwoFactorAuth -contains 'emailOtp') {
            $researchMethod = 'emailotp'
            $researchPrompt = 'Code sent by VRChat to your account email'
        }
        else { throw 'This verification method is not supported by the helper. No secrets uploaded.' }
        $researchCode = (Read-ResearchSecret $researchPrompt).Trim()
        if (!$researchCode) { throw 'Enter the code shown by the verification method above.' }
        if ($researchMethod -eq 'totp' -and $researchCode -notmatch '^[0-9]{6}$') {
            throw 'Enter the six digits shown in your authenticator app.'
        }
        $researchVerified = Invoke-VRChatRequest ('/auth/twofactorauth/' + $researchMethod + '/verify') @{code=$researchCode}
        $researchCode = $null
        if (-not $researchVerified.verified) { throw 'VRChat did not verify the login. No secrets uploaded.' }
        $researchUser = Invoke-VRChatRequest '/auth/user'
    }
    if ($researchUser.id -notmatch '^usr_[a-fA-F0-9-]{36}$' -or $researchUser.requiresTwoFactorAuth) { throw 'VRChat login was not completed. No secrets uploaded.' }
    $researchSession = @($researchCookies.GetCookies([Uri]$researchBase) | Where-Object { $_.Name -in @('auth','twoFactorAuth') } | ForEach-Object { $_.Name + '=' + $_.Value }) -join '; '
    if ($researchSession -notmatch '(^|; )auth=') { throw 'VRChat did not provide a session cookie. No secrets uploaded.' }
    Write-Host ('Verified VRChat account: ' + $researchUser.displayName)
    if ((Read-Host 'Type CONNECT to authorize this account for vrchat-world-research') -cne 'CONNECT') {
        throw 'Connection cancelled.'
    }
    $researchPayload = @{VRCHAT_AUTH_COOKIE=$researchSession} | ConvertTo-Json -Compress
    $researchPayload | & node $wrangler secret bulk --config $configPath
    if ($LASTEXITCODE -ne 0) { throw 'Cloudflare could not save the authorized VRChat session.' }
    Write-Host 'VRChat authorization stored for the standalone Worker.'
} finally {
    $researchClient.Dispose(); $researchHandler.Dispose()
    $researchPassword=$null; $researchBasic=$null; $researchCode=$null; $researchSession=$null; $researchPayload=$null; $researchCookies=$null
}
