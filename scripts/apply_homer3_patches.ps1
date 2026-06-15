param(
    [string]$Root = (Split-Path $PSScriptRoot -Parent)
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path $Root).Path

$homer = Join-Path $Root 'code\utils\Homer3-1.80.2'
$patches = Join-Path $Root 'code\utils\homer3_patches'

if (-not (Test-Path $homer)) {
    Write-Error "Homer3 not found at $homer. Clone v1.80.2 there first; see docs/install.md"
}

$targets = @(
    @{
        Source = Join-Path $patches 'FuncRegistry\UserFunctions\hmrR_PruneChannels.m'
        Dest   = Join-Path $homer 'FuncRegistry\UserFunctions\hmrR_PruneChannels.m'
    },
    @{
        Source = Join-Path $patches 'DataTree\AcquiredData\DataFiles\Hdf5\hdf5write_safe.m'
        Dest   = Join-Path $homer 'DataTree\AcquiredData\DataFiles\Hdf5\hdf5write_safe.m'
    }
)

foreach ($item in $targets) {
    if (-not (Test-Path $item.Source)) {
        Write-Error "Missing patch file: $($item.Source)"
    }
    Copy-Item $item.Source $item.Dest -Force
    Write-Host "Patched: $($item.Dest)"
}

Write-Host 'Homer3 patches applied.'
