# Shared staging helpers for the per-platform build scripts.
#
# Kept in PowerShell rather than duplicated into each script so that the three platforms cannot
# drift in what they put in the payload; the Linux and macOS scripts call this through `pwsh`
# when it is available and fall back to inline shell when it is not (see their headers).

Set-StrictMode -Version Latest

function Get-WorkspaceVersion {
    <#
    .SYNOPSIS
    The workspace version from the root Cargo.toml's [workspace.package].
    #>
    param([Parameter(Mandatory = $true)][string] $Workspace)
    $manifest = Join-Path $Workspace "Cargo.toml"
    $inPackage = $false
    foreach ($line in Get-Content -LiteralPath $manifest) {
        $trimmed = $line.Trim()
        if ($trimmed -eq "[workspace.package]") { $inPackage = $true; continue }
        if ($inPackage -and $trimmed.StartsWith("[")) { break }
        if ($inPackage -and $trimmed -match '^version\s*=\s*"([^"]+)"') { return $Matches[1] }
    }
    throw "could not read [workspace.package] version from $manifest"
}

function New-StagedPayload {
    <#
    .SYNOPSIS
    Create an empty staging directory, removing any previous one so a stale file from an earlier
    build can never be shipped.
    #>
    param([Parameter(Mandatory = $true)][string] $Stage)
    if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
    New-Item -ItemType Directory -Path $Stage -Force | Out-Null
}

function Write-PayloadDocs {
    <#
    .SYNOPSIS
    Write the documentation every artefact carries: what it is, what it deliberately does not
    contain, and how to point it at the user's own Minecraft installation.
    #>
    param(
        [Parameter(Mandatory = $true)][string] $Stage,
        [Parameter(Mandatory = $true)][string] $Workspace,
        [Parameter(Mandatory = $true)][string] $Version,
        [Parameter(Mandatory = $true)][string] $Triple,
        [Parameter(Mandatory = $true)][string] $RunLine
    )

    $readme = @"
mc-rs $Version ($Triple)
========================

A native Rust client for Minecraft 1.8.9 servers (protocol 47).

WHAT IS IN THIS ARCHIVE
-----------------------
This project's own binary, this file, and a launcher script. That is all.

WHAT IS DELIBERATELY NOT IN IT
------------------------------
Any file belonging to Mojang: the 1.8.9 jar, block models, textures, sounds, language files,
the asset index, the asset object store. Redistributing them is not permitted, so this build
does not carry them and no configuration of it will.

WHAT YOU NEED
-------------
An existing Minecraft installation made by the official launcher, with the 1.8.9 profile
started at least once so its assets have been downloaded. The launcher script finds it in the
standard location for this platform, or wherever MC_RS_MINECRAFT_DIR points.

HOW TO RUN
----------
$RunLine

Nothing is copied out of your Minecraft directory; it is read in place.

LOGGING IN
----------
Sign in on the client's Accounts screen (a Microsoft device-code sign-in; it needs a build made
with AERO_MSA_CLIENT_ID), or start the client from a launcher that passes --username, --uuid and
--accessToken. The dev-only offline login is compiled out of release artefacts.

VERSION
-------
mc-rs $Version, built for $Triple.
"@
    Set-Content -LiteralPath (Join-Path $Stage "README.txt") -Value $readme -Encoding utf8

    $notice = @"
mc-rs — NOTICE

This software is an independent client implementation. It is not affiliated with, endorsed by,
or a product of Mojang Studios or Microsoft.

It contains no Minecraft game assets and no Minecraft code. It reads the assets of an existing,
separately obtained Minecraft installation from the machine it runs on, and does not copy or
redistribute them.

Minecraft is a trademark of Mojang AB.
"@
    Set-Content -LiteralPath (Join-Path $Stage "NOTICE.txt") -Value $notice -Encoding utf8

    # A licence file is copied when the repository has one, and its absence is reported rather
    # than silently producing an unlicensed artefact.
    $license = Join-Path $Workspace "LICENSE"
    if (Test-Path -LiteralPath $license) {
        Copy-Item -LiteralPath $license -Destination (Join-Path $Stage "LICENSE.txt")
    } else {
        Write-Host "note: no LICENSE at the workspace root; the artefact ships without one"
    }
}

function Write-BuildInfo {
    <#
    .SYNOPSIS
    Record what was built, how, and which bytes came out — the manifest `verify-payload.ps1`
    checks. The SHA-256 pins the manifest to this binary, so a stale manifest beside a
    hand-substituted binary is caught.
    #>
    param(
        [Parameter(Mandatory = $true)][string] $Stage,
        [Parameter(Mandatory = $true)][string] $Version,
        [Parameter(Mandatory = $true)][string] $Triple,
        [Parameter(Mandatory = $true)][string] $Profile,
        [Parameter(Mandatory = $true)][string] $Binary,
        [string] $DefaultFeatures = "off",
        [string] $Features = ""
    )
    $hash = (Get-FileHash -LiteralPath $Binary -Algorithm SHA256).Hash.ToLowerInvariant()
    $size = (Get-Item -LiteralPath $Binary).Length
    $lines = @(
        "# mc-rs build manifest, written by packaging/*/build.*",
        "version: $Version",
        "triple: $Triple",
        "profile: $Profile",
        "default-features: $DefaultFeatures",
        "features: $Features",
        "binary: $([System.IO.Path]::GetFileName($Binary))",
        "binary-bytes: $size",
        "binary-sha256: $hash",
        "built-utc: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    )
    Set-Content -LiteralPath (Join-Path $Stage "BUILD-INFO.txt") -Value $lines -Encoding utf8
}

Export-ModuleMember -Function Get-WorkspaceVersion, New-StagedPayload, Write-PayloadDocs, Write-BuildInfo
