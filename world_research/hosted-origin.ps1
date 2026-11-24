function Get-WorldResearchBaseUri {
    $origin = $env:WORLD_RESEARCH_ORIGIN
    if (!$origin) { $origin = 'https://escapethroughreality.com' }
    $parsed = $null
    if ($origin -match '[\s\\]' -or
        ![uri]::TryCreate($origin, [UriKind]::Absolute, [ref]$parsed) -or
        $parsed.Scheme -ne 'https' -or !$parsed.Host -or $parsed.UserInfo -or
        $parsed.AbsolutePath -ne '/' -or $parsed.Query -or $parsed.Fragment) {
        throw 'WORLD_RESEARCH_ORIGIN must be an HTTPS origin without a path.'
    }
    return $parsed.GetLeftPart([UriPartial]::Authority) + '/api/world-research'
}
