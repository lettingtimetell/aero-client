#requires -Version 5.1
<#
.SYNOPSIS
Start mc-rs (the installed client, or a development build) with your own Minecraft 1.8.9 files.
.DESCRIPTION
Finds the Minecraft 1.8.9 game files and starts the client with them; its game directory is
mc-rs\, so settings, logs and screenshots go there (all git-ignored).

The game files, first found wins: -JarPath and -AssetsDir when given; the copy that
setup-vanilla-files.ps1 made in MCP-919-main\jars; then -MinecraftDir, the official launcher's
folder by default (a folder that holds the 1.8.9 client jar under versions\1.8.9\ or
libraries\com\mojang\minecraft\1.8.9\, and an assets folder with the 1.8 index).

This script reads no account and passes no sign-in. Sign in inside the client: its Accounts
screen (the account corner at the top right of the title screen, or Options > Accounts...) adds
your Microsoft account and keeps that sign-in, sealed for your Windows user, for the next start.
.EXAMPLE
.\launch-mc-rs.ps1
.EXAMPLE
.\launch-mc-rs.ps1 -Server localhost
.EXAMPLE
.\launch-mc-rs.ps1 -Build debug -CheckOnly
.EXAMPLE
.\launch-mc-rs.ps1 -MinecraftDir 'D:\Games\Minecraft'
#>
[CmdletBinding()]
param(
    [string]$MinecraftDir = $(if ($env:APPDATA) { Join-Path $env:APPDATA '.minecraft' } else { '' }),
    [string]$JarPath,
    [string]$AssetsDir,
    [string]$ClientPath,
    [ValidateSet('release', 'debug')][string]$Build = 'release',
    [string]$Server,
    [ValidateRange(1, 65535)][int]$Port = 25565,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$clientPathGiven = $PSBoundParameters.ContainsKey('ClientPath')
$minecraftDirGiven = $PSBoundParameters.ContainsKey('MinecraftDir')

# Newest source file under mc-rs. Keep in step with the same function in install-mc-rs.ps1.
function Get-NewestSource([string]$McRs) {
    $files = @(Get-ChildItem -LiteralPath (Join-Path $McRs 'crates') -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in '.rs', '.wgsl', '.toml' })
    $files += @(Get-Item -LiteralPath (Join-Path $McRs 'Cargo.toml'), (Join-Path $McRs 'Cargo.lock'),
        (Join-Path $McRs '.cargo\config.toml') -ErrorAction SilentlyContinue)
    $newest = $files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    [pscustomobject]@{ Ticks = $newest.LastWriteTimeUtc.Ticks; File = $newest.FullName.Substring($McRs.Length + 1) }
}

# The installed client must be a PGO build of the current code, made by install-mc-rs.ps1 (a plain
# cargo build is ~8% slower in SkyWars, and a PGO profile only fits the code it was trained on).
# Warn on every launch when that is no longer true; never block the launch.
function Test-InstalledBuild([string]$McRs, [string]$Client) {
    $fix = 'Run install-mc-rs.ps1 (about 25 min) before relying on its performance.'
    $stampPath = Join-Path (Split-Path -Parent $Client) 'mc-client.install.json'
    if (-not (Test-Path -LiteralPath $stampPath -PathType Leaf)) {
        Write-Host "WARNING: this mc-rs client was not installed by install-mc-rs.ps1, so it is not a PGO build. $fix" -ForegroundColor Yellow
        return
    }
    $stamp = Get-Content -LiteralPath $stampPath -Raw | ConvertFrom-Json
    if ((Get-FileHash -LiteralPath $Client -Algorithm SHA256).Hash -ne $stamp.exe_sha256) {
        Write-Host "WARNING: the mc-rs client was replaced after install-mc-rs.ps1 installed it (a plain cargo build?). $fix" -ForegroundColor Yellow
        return
    }
    $source = Get-NewestSource $McRs
    if ($source.Ticks -gt [long]$stamp.source_newest_ticks) {
        Write-Host "WARNING: the mc-rs code changed after this client was built ($($source.File)). The installed client is out of date and its PGO profile no longer fits. $fix" -ForegroundColor Yellow
    }
}

# The Minecraft 1.8.9 files in a game folder: the 1.8.9 client jar (where the official launcher
# keeps it, or under libraries\) and the assets folder that holds the 1.8 index. Either may be
# $null. Keep in step with the same function in setup-vanilla-files.ps1.
function Find-VanillaFiles([string]$Dir) {
    $jar = @((Join-Path $Dir 'versions\1.8.9\1.8.9.jar'),
        (Join-Path $Dir 'libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar')) |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    $assets = Join-Path $Dir 'assets'
    if (-not (Test-Path -LiteralPath (Join-Path $assets 'indexes\1.8.json') -PathType Leaf)) { $assets = $null }
    [pscustomobject]@{ Jar = $jar; Assets = $assets }
}

try {
    $projectDir = Join-Path $PSScriptRoot 'mc-rs'
    if (-not $ClientPath) {
        $ClientPath = Join-Path $projectDir "target\$Build\mc-client.exe"
        if (-not (Test-Path -LiteralPath $ClientPath -PathType Leaf)) {
            $buildCommand = if ($Build -eq 'release') { 'install-mc-rs.ps1 (the PGO install)' } else { 'cargo build -p mc-client from mc-rs' }
            throw "The $Build client was not found. Run $buildCommand first."
        }
    }
    if (-not (Test-Path -LiteralPath $ClientPath -PathType Leaf)) {
        throw 'The specified mc-rs client executable does not exist.'
    }
    $ClientPath = (Resolve-Path -LiteralPath $ClientPath).Path
    if ($Build -eq 'release' -and -not $clientPathGiven) {
        Test-InstalledBuild $projectDir $ClientPath
    }

    # The game files: explicit paths first, then the folders in order.
    $folders = if ($minecraftDirGiven) { @($MinecraftDir) } else {
        @((Join-Path $PSScriptRoot 'MCP-919-main\jars'), $MinecraftDir)
    }
    $versionJar = if ($JarPath) { $JarPath } else { $null }
    $assets = if ($AssetsDir) { $AssetsDir } else { $null }
    foreach ($dir in @($folders | Where-Object { $_ })) {
        if ($versionJar -and $assets) { break }
        $found = Find-VanillaFiles $dir
        if (-not $versionJar) { $versionJar = $found.Jar }
        if (-not $assets) { $assets = $found.Assets }
    }
    if (-not $versionJar -or -not (Test-Path -LiteralPath $versionJar -PathType Leaf)) {
        throw ('The Minecraft 1.8.9 jar was not found. Start Minecraft 1.8.9 once in the official ' +
            'launcher, run .\setup-vanilla-files.ps1, or pass -MinecraftDir or -JarPath.')
    }
    if (-not $assets -or -not (Test-Path -LiteralPath (Join-Path $assets 'indexes\1.8.json') -PathType Leaf)) {
        throw ('The Minecraft 1.8 assets were not found. Start Minecraft 1.8.9 once in the official ' +
            'launcher, run .\setup-vanilla-files.ps1, or pass -MinecraftDir or -AssetsDir.')
    }

    if ($CheckOnly) {
        Write-Host "Ready to launch. Sign in on the client's Accounts screen if you haven't yet."
        Write-Host "Client: $ClientPath"
        Write-Host "Jar:    $versionJar"
        Write-Host "Assets: $assets"
        return
    }
    $clientArgs = @(
        '--gameDir', $projectDir,
        '--assetsDir', $assets,
        '--assetIndex', '1.8',
        '--versionJar', $versionJar
    )
    if ($Server) { $clientArgs += @('--server', $Server, '--port', [string]$Port) }
    Write-Host 'Launching mc-rs (sign in on its Accounts screen if you haven''t yet)...'
    Write-Host "Client: $ClientPath"
    # Pipelined submit with pacing: frame encoding, submit and present run on a second thread;
    # ~26% more frames in SkyWars at the same average input latency. Set MC_PIPELINED_SUBMIT=0
    # before running this script to use the serial loop instead. Only the client process sees the
    # value; this PowerShell session's environment is restored after.
    $previousPipelinedSubmit = $env:MC_PIPELINED_SUBMIT
    if (-not $previousPipelinedSubmit) { $env:MC_PIPELINED_SUBMIT = '1' }
    Push-Location -LiteralPath $projectDir
    try {
        & $ClientPath @clientArgs
        $clientExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $env:MC_PIPELINED_SUBMIT = $previousPipelinedSubmit
    }
    if ($clientExit -ne 0) { throw "mc-rs exited with code $clientExit." }
} catch {
    Write-Host ('mc-rs launcher: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
