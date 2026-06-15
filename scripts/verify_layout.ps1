param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [ValidateSet('open', 'full')]
    [string]$Profile = 'full',

    [switch]$RequireRestrictedAv,
    [switch]$SkipFnirsCheck
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path $Root).Path

$required = @('VERSION')

if ($Profile -eq 'open') {
    $required += @(
        'data/subject_data_inventory.csv',
        'data/README.md'
    )
    if (-not $SkipFnirsCheck) {
        $required += 'data/fnirs_signals/AEPO_001filt02'
    }
} else {
    $required += @(
        'README.md',
        'data/all_raw/subject_info.csv',
        'code/validation_traits.py',
        'experiments/main_exp1.py'
    )
    if (-not $SkipFnirsCheck) {
        $required += 'data/fnirs_signals/AEPO_001filt02'
    }
}

$missing = @()
foreach ($rel in $required) {
    $p = Join-Path $Root $rel
    if (-not (Test-Path $p)) { $missing += $rel }
}

if ($missing.Count -gt 0) {
    Write-Error "Missing required paths (profile=$Profile):`n$($missing -join "`n")"
}

$mp4 = @()
if (Test-Path -LiteralPath (Join-Path $Root 'data/all_raw')) {
    $mp4 = @(Get-ChildItem -Path (Join-Path $Root 'data/all_raw') -Filter '*.mp4' -Recurse -ErrorAction SilentlyContinue)
}
Write-Host "Profile: $Profile — layout OK. MP4 count under data/all_raw: $($mp4.Count)"

if ($RequireRestrictedAv -and $mp4.Count -lt 800) {
    Write-Warning "Expected ~810 MP4 after restricted merge; found $($mp4.Count)."
}

Write-Host "verify_layout.ps1 passed."
