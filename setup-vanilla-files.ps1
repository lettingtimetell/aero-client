#requires -Version 5.1
<#
.SYNOPSIS
Copy your own Minecraft 1.8.9 files into MCP-919-main\jars\ in this repository, where development
builds, the PGO training of install-mc-rs.ps1 and the test suite look for them.

.DESCRIPTION
This repository contains no Minecraft game files. These parts of the build expect them under
MCP-919-main\jars\ (the folder name is that of the local reference workspace the port was developed
against; git ignores the whole MCP-919-main\ folder, so nothing copied there is ever committed):

  - the client's development fallback: started without --versionJar/--assetsDir, a build of this
    repository looks there (mc_client::app::config::discover_jars_dir). The PGO training runs and
    the install smoke test start the client that way;
  - the integration tests that load real vanilla models, textures, sounds and NBT.

The script fills:
  versions\1.8.9\1.8.9.jar                    the vanilla 1.8.9 client jar (textures, models, blockstates)
  assets\indexes\1.8.json, assets\objects\    the 1.8 asset index and the objects it lists (sounds, languages)
  servers.dat, saves\MCP Development World\level.dat
                                              two tiny reference NBT files (60 and 416 bytes, embedded
                                              below) that the mc-nbt and mc-ui golden tests read

Sources, first found wins: -JarPath/-AssetsDir if given, then -MinecraftDir: the official
launcher's folder by default, or any game folder that holds the 1.8.9 client jar under
versions\1.8.9\ or libraries\com\mojang\minecraft\1.8.9\, and an assets folder with the 1.8 index.
Install and start Minecraft 1.8.9 once first. Files already present with the same size are
skipped, so running it again is quick.

Playing does not need this: launch-mc-rs.ps1 finds the same files itself.

.PARAMETER MinecraftDir
The game folder to copy from. Default: %APPDATA%\.minecraft (the official launcher's folder).

.PARAMETER JarPath
A specific Minecraft 1.8.9 client jar to copy.

.PARAMETER AssetsDir
A specific assets folder (the one holding indexes\1.8.json and objects\).

.PARAMETER Destination
Where to put the files. Default: MCP-919-main\jars next to this script.

.EXAMPLE
.\setup-vanilla-files.ps1
.EXAMPLE
.\setup-vanilla-files.ps1 -MinecraftDir 'D:\Games\Minecraft'
#>
[CmdletBinding()]
param(
    [string]$MinecraftDir = $(if ($env:APPDATA) { Join-Path $env:APPDATA '.minecraft' } else { '' }),
    [string]$JarPath,
    [string]$AssetsDir,
    [string]$Destination = (Join-Path $PSScriptRoot 'MCP-919-main\jars')
)

$ErrorActionPreference = 'Stop'
$Destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)

function Copy-IfChanged([string]$From, [string]$To) {
    if ((Test-Path -LiteralPath $To -PathType Leaf) -and
        (Get-Item -LiteralPath $To).Length -eq (Get-Item -LiteralPath $From).Length) {
        return $false
    }
    New-Item -ItemType Directory -Force (Split-Path -Parent $To) | Out-Null
    Copy-Item -LiteralPath $From -Destination $To -Force
    return $true
}

# The Minecraft 1.8.9 files in a game folder: the 1.8.9 client jar (where the official launcher
# keeps it, or under libraries\) and the assets folder that holds the 1.8 index. Either may be
# $null. Keep in step with the same function in launch-mc-rs.ps1.
function Find-VanillaFiles([string]$Dir) {
    $jar = @((Join-Path $Dir 'versions\1.8.9\1.8.9.jar'),
        (Join-Path $Dir 'libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar')) |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    $assets = Join-Path $Dir 'assets'
    if (-not (Test-Path -LiteralPath (Join-Path $assets 'indexes\1.8.json') -PathType Leaf)) { $assets = $null }
    [pscustomobject]@{ Jar = $jar; Assets = $assets }
}

$found = if ($MinecraftDir) { Find-VanillaFiles $MinecraftDir } else { [pscustomobject]@{ Jar = $null; Assets = $null } }

# 1. The client jar.
if (-not $JarPath) { $JarPath = $found.Jar }
if (-not $JarPath -or -not (Test-Path -LiteralPath $JarPath -PathType Leaf)) {
    throw ('Minecraft 1.8.9 jar not found. Install and start Minecraft 1.8.9 once in the official ' +
        'launcher, or pass -MinecraftDir (your game folder) or -JarPath.')
}

# 2. The asset store holding the 1.8 index.
if (-not $AssetsDir) { $AssetsDir = $found.Assets }
if (-not $AssetsDir -or -not (Test-Path -LiteralPath (Join-Path $AssetsDir 'indexes\1.8.json') -PathType Leaf)) {
    throw ('Asset index 1.8.json not found. Start Minecraft 1.8.9 once in your launcher (it downloads ' +
        'the assets), or pass -MinecraftDir or -AssetsDir.')
}

Write-Host "jar:    $JarPath"
Write-Host "assets: $AssetsDir"
Write-Host "into:   $Destination"

$copied = 0
if (Copy-IfChanged $JarPath (Join-Path $Destination 'versions\1.8.9\1.8.9.jar')) { $copied++ }
$indexPath = Join-Path $AssetsDir 'indexes\1.8.json'
if (Copy-IfChanged $indexPath (Join-Path $Destination 'assets\indexes\1.8.json')) { $copied++ }

# Only the objects the 1.8 index names: a launcher's shared store also holds every other version's.
$index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
$hashes = @($index.objects.PSObject.Properties | ForEach-Object { $_.Value.hash } | Sort-Object -Unique)
$missing = 0
foreach ($hash in $hashes) {
    $rel = Join-Path 'objects' (Join-Path $hash.Substring(0, 2) $hash)
    $from = Join-Path $AssetsDir $rel
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { $missing++; continue }
    if (Copy-IfChanged $from (Join-Path (Join-Path $Destination 'assets') $rel)) { $copied++ }
}

# 3. The two reference NBT files, byte for byte: a one-entry server list ("MCP Test Server",
#    127.0.0.1) and the level.dat of an empty development world. Not Minecraft code or assets.
$nbt = [ordered]@{
    'servers.dat' = 'CgAACQAHc2VydmVycwoAAAABCAAEbmFtZQAPTUNQIFRlc3QgU2VydmVyCAACaXAACTEyNy4wLjAuMQAA'
    'saves\MCP Development World\level.dat' = (
        'H4sIADS0rFMA/01RUW7TQBCdrUPj2opAcBG++UCojVJAKUGJVQFCSFPvNFllvWPtrhOZc3ABfjgFXIAjwAW4BYyduHQ+' +
        '38578/a9DCCD0RQjKsjipnGavHFrGEE2xxDfWmxJA6jZr6dfvo9gPMW2MBVBP8kfBblxJhq05jNpJawlOs3Vikj/lfn6' +
        '8eenBNJLrKho646mhHKF9YwwNp6CSmC8Ix8MO4DXPxRM0FreX3BViVAQxdH/e6eFgrFHuSgOE8iPfg8Lz58lcLqqce/e' +
        'AZy8VJBu0OuSPcHw8F40XqRwNqcd2TfiCdKqrPfsrR5WPsjKtxQercmRx8h+UUfxFrqYzrpvLBtLIYW84ptLb+i2Dyv6' +
        'hkRX8xXfzJnjgGSaZ8ZTYcrtAOWaC2Np6rkOAzbpif39e3KTLVH9yu3IiY0WHtyiDQI/cV1yaJd09NhFd6Q8Lg+5nVsu' +
        't4sm1s2dlYeapTtr1pt40ZaWDjjImbuv9onkFv2azg1XFKTOlfS6cFMTtnAcqbOroA/95Pp3ci+r66HIbv4BGLY+iFoC' +
        'AAA=')
}
foreach ($name in $nbt.Keys) {
    $to = Join-Path $Destination $name
    if (-not (Test-Path -LiteralPath $to -PathType Leaf)) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $to) | Out-Null
        [IO.File]::WriteAllBytes($to, [Convert]::FromBase64String($nbt[$name]))
        $copied++
    }
}

Write-Host "Done: $copied file(s) copied; the 1.8 index lists $($hashes.Count) asset objects." -ForegroundColor Green
if ($missing) {
    Write-Host ("$missing of them were not in $AssetsDir; start 1.8.9 once more in the launcher to " +
        'download them (some sounds or languages may be missing until then).') -ForegroundColor Yellow
}
Write-Host 'MCP-919-main\ is git-ignored: these files stay on this PC.'
