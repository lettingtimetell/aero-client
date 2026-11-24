# The guard that keeps a redistributable artefact free of Mojang's assets.
#
# Minecraft's own files are never distributed. `packaging/README.md`
# explains why that decides the whole packaging design. This script is the mechanical enforcement:
# it walks a staged payload directory and fails if it finds anything that looks like it came out
# of a Minecraft installation.
#
# It is deliberately conservative — it rejects on *shape* (extension, name, directory) rather than
# trying to identify content, because the cost of a false positive is a build failure and the cost
# of a false negative is redistributing copyrighted material.
#
# Usage:
#   pwsh -File verify-payload.ps1 -Payload <dir> [-ExpectBinary mc-client.exe] [-Quiet]
#        [-ExpectFiles <file>,<file>...] [-ForbidExtensions <.ext>,<.ext>...]
# -ExpectFiles names files the payload must hold (paths relative to it); -ForbidExtensions adds
# file types a package must not hold to Minecraft's. Lists may also be one comma-separated
# string (what `pwsh -File` passes).
# Exit codes: 0 clean, 1 a forbidden file was found, 2 the payload is not usable.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Payload,
    [string] $ExpectBinary = "",
    [string[]] $ExpectFiles = @(),
    [string[]] $ForbidExtensions = @(),
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"

function Write-Note([string] $Message) {
    if (-not $Quiet) { Write-Host $Message }
}

if (-not (Test-Path -LiteralPath $Payload -PathType Container)) {
    Write-Host "verify-payload: payload directory not found: $Payload"
    exit 2
}
$ExpectFiles = @($ExpectFiles | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$ForbidExtensions = @($ForbidExtensions | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim().ToLowerInvariant() } |
    Where-Object { $_ } | ForEach-Object { if ($_.StartsWith(".")) { $_ } else { ".$_" } })

# Anything with one of these extensions is a Minecraft asset in every layout that matters:
# textures, sounds, models/blockstates, language files, the version jar itself, world saves.
$forbiddenExtensions = @(
    ".jar", ".png", ".ogg", ".lang", ".mcmeta", ".dat", ".mus", ".nbt", ".mca", ".mcr", ".icns"
)
# Directory names that only ever appear inside a Minecraft installation.
$forbiddenDirectories = @(
    "assets", "versions", "libraries", "resourcepacks", "saves", "objects", "indexes",
    "blockstates", "models", "textures", "sounds", "shaders"
)
# Exact file names that are Mojang's regardless of where they sit.
$forbiddenNames = @(
    "1.8.9.jar", "1.8.9.json", "servers.dat", "options.txt", "glyph_sizes.bin", "pack.mcmeta"
)

$problems = New-Object System.Collections.Generic.List[string]
$fileCount = 0

foreach ($file in Get-ChildItem -LiteralPath $Payload -Recurse -File) {
    $fileCount++
    $relative = $file.FullName.Substring($Payload.Length).TrimStart('\', '/')
    $extension = $file.Extension.ToLowerInvariant()
    $name = $file.Name.ToLowerInvariant()

    if ($forbiddenExtensions -contains $extension) {
        $problems.Add("forbidden extension '$extension': $relative")
    }
    if ($ForbidExtensions -contains $extension) {
        $problems.Add("file type '$extension' not allowed in this package: $relative")
    }
    if ($forbiddenNames -contains $name) {
        $problems.Add("forbidden file name: $relative")
    }
    foreach ($segment in ($relative -split '[\\/]')) {
        if ($forbiddenDirectories -contains $segment.ToLowerInvariant()) {
            $problems.Add("forbidden directory segment '$segment': $relative")
            break
        }
    }
}

if ($fileCount -eq 0) {
    Write-Host "verify-payload: payload is empty: $Payload"
    exit 2
}

# What a package must hold to work at all, as the caller names it: the program players start,
# the files an installer or an updater relies on.
foreach ($expected in $ExpectFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $Payload $expected) -PathType Leaf)) {
        Write-Host "verify-payload: expected file not found in payload: $expected"
        exit 2
    }
}

# The offline login path (no sign-in) is dev-only And S21's
# handoff says explicitly that a release build must turn it off. `offline` is on by default, so
# forgetting `--no-default-features` is a one-word mistake with a real consequence.
#
# This is checked through the BUILD-INFO.txt the build script writes, pinned to the binary by its
# SHA-256, rather than by inspecting the compiled bytes. That was the first attempt and it does
# **not** work, which is worth recording so nobody tries it again: the obvious marker,
# `mc_auth::offline::OFFLINE_PREFIX = "OfflinePlayer:"`, is absent from a release binary built
# *with* the feature — it is only ever fed to an MD5 update, so the optimiser folds it away and
# no literal survives. A byte search would therefore have reported every build as clean, which is
# the worst possible failure mode for a guard.
#
# So this is honest about what it is: a check that the build script did what it claims and that
# the manifest belongs to this binary. It catches the realistic mistake — a hand-run `cargo build
# --release` copied into a payload, or a stale binary beside a fresh manifest. It is not a
# defence against someone who edits both.
if ($ExpectBinary -ne "") {
    $binary = Join-Path $Payload $ExpectBinary
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        Write-Host "verify-payload: expected binary not found in payload: $ExpectBinary"
        exit 2
    }
    $infoPath = Join-Path $Payload "BUILD-INFO.txt"
    if (-not (Test-Path -LiteralPath $infoPath -PathType Leaf)) {
        $problems.Add("no BUILD-INFO.txt beside $ExpectBinary; the payload was not staged by a build script")
    } else {
        $info = @{}
        foreach ($line in Get-Content -LiteralPath $infoPath) {
            if ($line -match '^\s*([a-z0-9-]+)\s*:\s*(.*?)\s*$') { $info[$Matches[1]] = $Matches[2] }
        }
        if ($info["default-features"] -ne "off") {
            $problems.Add("BUILD-INFO.txt says default-features: $($info['default-features']); a release artefact must build with --no-default-features so the dev-only 'offline' login is compiled out")
        }
        $actual = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
        $recorded = ($info["binary-sha256"] + "").ToLowerInvariant()
        if ($recorded -eq "") {
            $problems.Add("BUILD-INFO.txt records no binary-sha256")
        } elseif ($recorded -ne $actual) {
            $problems.Add("BUILD-INFO.txt describes a different binary (sha256 $recorded, payload has $actual)")
        }
    }
}

if ($problems.Count -gt 0) {
    Write-Host "PAYLOAD REJECTED ($($problems.Count) problem(s) across $fileCount file(s)):"
    foreach ($p in $problems) { Write-Host "  - $p" }
    Write-Host ""
    Write-Host "See CONTRIBUTING.md (Packaging): a redistributable artefact must carry none of Minecraft's"
    Write-Host "own files, and must be built with --no-default-features."
    exit 1
}

Write-Note "payload clean: $fileCount file(s) in $Payload, no Minecraft assets"
exit 0
