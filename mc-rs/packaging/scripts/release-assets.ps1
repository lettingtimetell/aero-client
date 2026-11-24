#requires -Version 5.1
<#
.SYNOPSIS
Turn a packaged build into the files of a GitHub release: release\v<version>\.

.DESCRIPTION
Run it after package-aero-client.ps1, which writes release\Aero Client.zip and
release\Aero-Client-Setup.exe. It writes into release\v<version>\:

  Aero-Client-<version>-windows-x64.zip          the portable zip, under the name the in-game
  Aero-Client-<version>-windows-x64.zip.sha256   updater looks for, its SHA-256 and (unless
  Aero-Client-<version>-windows-x64.zip.sig      -Unsigned) its update signature, all three made
                                                 by aero-update-sign (crates\mc-update)
  Aero-Client-Setup.exe                          the installer
  Aero-Client-Portable.zip                       the portable zip again, byte for byte
  SHA256SUMS.txt                                 the SHA-256 of the zips and the installer

Aero-Client-Setup.exe and Aero-Client-Portable.zip keep these names in every release: the website
links https://github.com/<repository>/releases/latest/download/<name>, which always serves the
newest release's file of that name. Never rename them.

Then it prints the gh command that publishes the release; it never runs it.

.PARAMETER Key
The update signing key (PKCS#8, made by aero-update-sign keygen). Clients built with its public
key (AERO_UPDATE_PUBKEY) install only releases signed with it.

.PARAMETER Unsigned
No update signature: every client then announces the release, and none installs it.

.PARAMETER Version
The release's version, x.y.z. Default: the workspace version in mc-rs\Cargo.toml, which the
packaged zip's marker must name.

.PARAMETER Zip
Default: release\Aero Client.zip.

.PARAMETER Installer
Default: release\Aero-Client-Setup.exe.

.PARAMETER OutDir
Default: release\v<version>. It must not exist yet, unless -Force (which deletes it first).
#>
[CmdletBinding()]
param(
    [string]$Key = '',
    [switch]$Unsigned,
    [string]$Version = '',
    [string]$Zip = '',
    [string]$Installer = '',
    [string]$OutDir = '',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$rs = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = Split-Path -Parent $rs
Import-Module (Join-Path $PSScriptRoot 'common.psm1') -Force

if (($Key -eq '') -eq (-not $Unsigned)) { throw 'give exactly one of -Key <signing key file> and -Unsigned' }
if ($Key -ne '' -and -not (Test-Path -LiteralPath $Key -PathType Leaf)) { throw "no signing key at $Key" }
if ($Version -eq '') { $Version = Get-WorkspaceVersion -Workspace $rs }
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw "$Version is not a release version (x.y.z)" }
if ($Zip -eq '') { $Zip = Join-Path $repoRoot 'release\Aero Client.zip' }
if ($Installer -eq '') { $Installer = Join-Path $repoRoot 'release\Aero-Client-Setup.exe' }
if ($OutDir -eq '') { $OutDir = Join-Path $repoRoot "release\v$Version" }
foreach ($file in @($Zip, $Installer)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "missing $file (run package-aero-client.ps1 first; the installer needs Inno Setup 6)"
    }
}
$Zip = (Resolve-Path -LiteralPath $Zip).Path
$Installer = (Resolve-Path -LiteralPath $Installer).Path

# The installer must be this version's (Setup writes its version into Settings > Apps).
# (Inno Setup pads the version text with spaces.)
$setupVersion = ([string](Get-Item -LiteralPath $Installer).VersionInfo.ProductVersion) -replace '[\s\x00]+$', ''
if ($setupVersion -ne $Version) { throw "$Installer is version '$setupVersion', not $($Version): package again" }
# Portable players start Aero Client.exe. Installed copies point their shortcuts at
# app\mc-client.exe with the icon app\aero-client.ico, and an update replaces app\, so every
# release zip must carry them. (Windows PowerShell's Compress-Archive writes \ in names.)
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
try {
    $names = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
} finally {
    $archive.Dispose()
}
foreach ($required in @('Aero Client/Aero Client.exe', 'Aero Client/app/mc-client.exe', 'Aero Client/app/aero-client.ico')) {
    if ($names -notcontains $required) { throw "$Zip has no $($required): package again" }
}
# A release holds no script, PowerShell or batch: the client starts itself.
$scriptFiles = @($names | Where-Object { $_ -match '\.(ps1|psm1|psd1|cmd|bat)$' })
if ($scriptFiles.Count -gt 0) { throw "$Zip holds scripts ($($scriptFiles -join ', ')): package again" }

if (Test-Path -LiteralPath $OutDir) {
    if (-not $Force) { throw "$OutDir already exists; delete it or pass -Force" }
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
# Where the release goes: the repository the client's updater reads (mc-update's DEFAULT_REPO, or
# AERO_UPDATE_REPO when the build set it).
$repo = $env:AERO_UPDATE_REPO
if (-not $repo) {
    $line = Select-String -LiteralPath (Join-Path $rs 'crates\mc-update\src\lib.rs') `
        -Pattern '^pub const DEFAULT_REPO: &str = "([^"]+)";' | Select-Object -First 1
    if ($line) { $repo = $line.Matches[0].Groups[1].Value }
}
if ($repo -notmatch '^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$') { throw "cannot tell the release repository (got '$repo')" }

New-Item -ItemType Directory -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$assetName = "Aero-Client-$Version-windows-x64.zip"
$versioned = Join-Path $OutDir $assetName
$setup = Join-Path $OutDir 'Aero-Client-Setup.exe'
$portable = Join-Path $OutDir 'Aero-Client-Portable.zip'
$expected = @($assetName, "$assetName.sha256")
if (-not $Unsigned) { $expected += "$assetName.sig" }
try {
    # The updater's three files (crates\mc-update\src\bin\aero-update-sign.rs). It unpacks the
    # zip with the updater's own checks first: the layout, and a marker naming this version.
    $signArgs = @(if ($Unsigned) { '--unsigned' } else { '--key'; (Resolve-Path -LiteralPath $Key).Path })
    Push-Location -LiteralPath $rs
    try {
        & cargo run --quiet -p mc-update --bin aero-update-sign -- sign @signArgs --zip $Zip --version $Version --out $OutDir
        if ($LASTEXITCODE -ne 0) { throw "aero-update-sign sign failed (exit code $LASTEXITCODE)" }
        if (-not $Unsigned) {
            # Check the files the way a client built with this key's public half will.
            $publicKey = (& cargo run --quiet -p mc-update --bin aero-update-sign -- pubkey --key (Resolve-Path -LiteralPath $Key).Path | Select-Object -Last 1)
            if ($LASTEXITCODE -ne 0 -or -not $publicKey) { throw 'aero-update-sign pubkey failed' }
            & cargo run --quiet -p mc-update --bin aero-update-sign -- verify --pubkey $publicKey.Trim() --dir $OutDir --version $Version
            if ($LASTEXITCODE -ne 0) { throw "aero-update-sign verify failed (exit code $LASTEXITCODE)" }
        }
    } finally {
        Pop-Location
    }
    foreach ($name in $expected) {
        if (-not (Test-Path -LiteralPath (Join-Path $OutDir $name) -PathType Leaf)) { throw "aero-update-sign wrote no $name" }
    }

    # The two names the website links.
    Copy-Item -LiteralPath $Installer -Destination $setup
    Copy-Item -LiteralPath $versioned -Destination $portable

    $hash = @{}
    foreach ($file in @($setup, $portable, $versioned)) {
        $hash[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if ($hash[$portable] -ne $hash[$versioned]) { throw 'Aero-Client-Portable.zip differs from the versioned zip' }
    if ($hash[$versioned] -ne (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant()) {
        throw "the versioned zip differs from $Zip"
    }
    $sidecar = (Get-Content -LiteralPath "$versioned.sha256" -Raw).Trim()
    if ($sidecar -ne "$($hash[$versioned])  $assetName") { throw "the .sha256 file does not match the zip: $sidecar" }
    # sha256sum's format (two spaces, LF), so `sha256sum -c SHA256SUMS.txt` checks the downloads.
    $sums = (@($setup, $portable, $versioned) | ForEach-Object { "$($hash[$_])  $(Split-Path -Leaf $_)`n" }) -join ''
    [IO.File]::WriteAllText((Join-Path $OutDir 'SHA256SUMS.txt'), $sums, (New-Object Text.UTF8Encoding $false))
} catch {
    # Half a release folder must never be published by mistake.
    Remove-Item -LiteralPath $OutDir -Recurse -Force -ErrorAction SilentlyContinue
    throw
}

Write-Host ''
Write-Host "Release files for v$Version in $OutDir" -ForegroundColor Green
Get-ChildItem -LiteralPath $OutDir | Sort-Object Name | ForEach-Object {
    Write-Host ('  {0,-46} {1,12:N0} bytes' -f $_.Name, $_.Length)
}
$assets = @('Aero-Client-Setup.exe', 'Aero-Client-Portable.zip') + $expected + @('SHA256SUMS.txt')
$quoted = ($assets | ForEach-Object { '"' + (Join-Path $OutDir $_) + '"' }) -join ' '
Write-Host ''
Write-Host 'To publish (not a draft, not a prerelease: the updater and releases/latest skip those):'
Write-Host ("gh release create v$Version --repo $repo --title `"Aero Client $Version`" --latest " +
    "--notes `"Windows: Aero-Client-Setup.exe installs Aero Client with Start menu and desktop " +
    "shortcuts; Aero-Client-Portable.zip runs from any folder.`" $quoted")
