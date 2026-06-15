param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [Parameter(Mandatory = $true)]
    [string[]]$Parts
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path $Root).Path

function Copy-TreeContents {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceDir,
        [Parameter(Mandatory = $true)]
        [string]$DestDir
    )
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Get-ChildItem -LiteralPath $SourceDir -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($SourceDir.Length).TrimStart('\', '/')
        $dest = Join-Path $DestDir ($rel -replace '/', [IO.Path]::DirectorySeparatorChar)
        New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
}

function Merge-ExtractedPart {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetRoot,
        [Parameter(Mandatory = $true)]
        [string]$ExtractRoot
    )
    $entries = @(Get-ChildItem -LiteralPath $ExtractRoot -Force)
    $dirs = @($entries | Where-Object { $_.PSIsContainer })

    $stimuli = $dirs | Where-Object { $_.Name -eq 'stimuli_exp3' } | Select-Object -First 1
    if ($stimuli) {
        $target = Join-Path $DatasetRoot 'experiments/stimuli_exp3'
        Write-Host "Merging stimuli_exp3/ -> $target"
        Copy-TreeContents -SourceDir $stimuli.FullName -DestDir $target
    }

    $experiments = $dirs | Where-Object { $_.Name -eq 'experiments' } | Select-Object -First 1
    if ($experiments) {
        $target = Join-Path $DatasetRoot 'experiments'
        Write-Host "Merging experiments/ -> $target"
        Copy-TreeContents -SourceDir $experiments.FullName -DestDir $target
    }

    foreach ($dir in $dirs) {
        if ($dir.Name -match '^\d+$') {
            $target = Join-Path $DatasetRoot ("data/all_raw/{0}" -f $dir.Name)
            Write-Host "Merging $($dir.Name)/ -> $target"
            Copy-TreeContents -SourceDir $dir.FullName -DestDir $target
        }
    }

    if ($dirs.Count -eq 1 -and $dirs[0].Name -match '^\d+$') {
        return
    }
    if (-not $stimuli -and -not $experiments -and $dirs.Count -eq 1) {
        $only = $dirs[0]
        if ($only.Name -eq 'stimuli_exp3') { return }
        if ($only.Name -eq 'experiments') { return }
    }
}

foreach ($part in $Parts) {
    $resolved = (Resolve-Path $part).Path
    if ($resolved -notmatch '\.zip$') {
        throw "Expected a .zip file: $resolved"
    }

    $tempDir = Join-Path ([IO.Path]::GetTempPath()) ("menact_restore_{0}" -f [guid]::NewGuid().ToString('n'))
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    try {
        Write-Host "Extracting $resolved"
        Expand-Archive -LiteralPath $resolved -DestinationPath $tempDir -Force
        Merge-ExtractedPart -DatasetRoot $Root -ExtractRoot $tempDir
    } finally {
        Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Done. Run scripts/verify_layout.ps1 -Root '$Root' to check layout."
