param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [string]$RepoUrl = 'https://github.com/BAK262/mENACT',
    [string]$GitHubTag = 'v1.0.0'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path $Root).Path

if (-not (Test-Path -LiteralPath (Join-Path $Root 'VERSION'))) {
    Write-Warning "VERSION not found in $Root — expected R0 unzipped first."
}

$tempClone = Join-Path ([IO.Path]::GetTempPath()) ("menact_clone_{0}" -f [guid]::NewGuid().ToString('n'))
New-Item -ItemType Directory -Force -Path $tempClone | Out-Null

try {
    Write-Host "Cloning $RepoUrl (branch/tag $GitHubTag) ..."
    & git clone --depth 1 --branch $GitHubTag $RepoUrl $tempClone
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed (exit $LASTEXITCODE)"
    }

    $moveItems = @('code', 'experiments', 'environments', 'docs', 'scripts', 'results')
    foreach ($name in $moveItems) {
        $src = Join-Path $tempClone $name
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Missing in clone: $name"
        }
        $dest = Join-Path $Root $name
        if (Test-Path -LiteralPath $dest) {
            Write-Warning "Removing existing $dest"
            Remove-Item -LiteralPath $dest -Recurse -Force
        }
        Move-Item -LiteralPath $src -Destination $dest
        Write-Host "Installed $name -> $dest"
    }
} finally {
    if (Test-Path -LiteralPath $tempClone) {
        Remove-Item -LiteralPath $tempClone -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ''
Write-Host 'Engineering tree installed.'
Write-Host 'Optional: restricted zips -> scripts/restore_restricted.ps1'
Write-Host "Verify: powershell -ExecutionPolicy Bypass -File scripts/verify_layout.ps1 -Root `"$Root`" -Profile full"
Write-Host 'Guide: docs/assembly.md'
