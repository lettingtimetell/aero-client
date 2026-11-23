#requires -Version 5.1
<#
.SYNOPSIS
Build the portable "Aero Client" zip to send to other people: release\Aero Client.zip.

.DESCRIPTION
The zip holds one "Aero Client" folder:
  Play Aero Client.cmd   double-click to play (runs app\launcher.ps1 hidden)
  README.txt             setup in plain words
  app\mc-client.exe      the client: PGO-optimised like the installed one
  app\launcher.ps1       finds the player's Minecraft 1.8.9 files and starts the game, friendly pop-ups
  app\LICENSE.txt        Aero Client's licence, the GPL-3.0 text (a copy of mc-rs\LICENSE)
  app\NOTICE.txt, app\BUILD-INFO.txt, app\licenses\ (the font licence and THIRD-PARTY-NOTICES.txt)
  app\aero-install.json  the updater's marker: a packaged portable install of this version
Each player's settings, logs and screenshots go in a "game" folder created on first start.

How the shared client differs from the one install-mc-rs.ps1 installs for you:
  - --no-default-features --features mimalloc: the dev-only "offline" login (no sign-in) is compiled
    out, as CONTRIBUTING.md ("Packaging") requires for anything distributed;
  - -Ctarget-feature=+crt-static: the C runtime is built in, so players don't need the Visual C++
    Redistributable.
Both builds map this PC's folders (user profile, cargo's home, this project, the build folders) to
neutral names with --remap-path-prefix, so a player's crash report names no folder of your PC.
It never touches target\release (your installed client) or your settings.

Steps (about 35 minutes; separate build folders, so your next install-mc-rs.ps1 stays fast):
  1. instrumented build           (optimization\scratch\round2-2026-11-05\tgt\pkg-gen)
  2. training on the benchmark scenes (as install-mc-rs.ps1)
  3. merge the profile            (...\pgo\package.profdata)
  4. optimised build              (...\tgt\pkg-use)
  5. checks: no Visual C++ runtime import, no offline login, a replay smoke run and a title-screen run
  6. stage, verify that no Minecraft file is inside (packaging\scripts\verify-payload.ps1), zip

.PARAMETER SkipBuild
Re-stage and re-zip the last build without compiling.

.PARAMETER SkipVanillaCheck
Build even though no Minecraft 1.8.9 jar is where the PGO training runs look for one (see
install-mc-rs.ps1). The profile then fits the real game less well.

.PARAMETER SignThumbprint
Optional Authenticode signing (see CONTRIBUTING.md, "Packaging"): the thumbprint of a code signing
certificate in Cert:\CurrentUser\My (hardware token, cloud signing service, any CA). Signs
app\mc-client.exe and app\launcher.ps1 before zipping. Also read from $env:AERO_SIGN_THUMBPRINT.
Off when empty.

.PARAMETER SignDlib
Optional Azure Artifact Signing: path to Microsoft's Azure.CodeSigning.Dlib.dll. Needs
-SignMetadata too. Signs app\mc-client.exe. Also read from $env:AERO_SIGN_DLIB.

.PARAMETER SignMetadata
Azure Artifact Signing metadata JSON (endpoint, account, certificate profile). Also read from
$env:AERO_SIGN_METADATA.

.PARAMETER TimestampUrl
RFC 3161 timestamp server for signing. Default: http://timestamp.digicert.com, or Microsoft's for
Artifact Signing.
#>
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipVanillaCheck,
    [string]$SignThumbprint = $env:AERO_SIGN_THUMBPRINT,
    [string]$SignDlib = $env:AERO_SIGN_DLIB,
    [string]$SignMetadata = $env:AERO_SIGN_METADATA,
    [string]$TimestampUrl = ''
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$rs = Join-Path $root 'mc-rs'
$round2 = Join-Path $rs 'optimization\scratch\round2-2026-11-05'
$templates = Join-Path $rs 'packaging\aero'
$scripts = Join-Path $rs 'packaging\scripts'
$outDir = Join-Path $root 'release'
$gen = Join-Path $round2 'tgt\pkg-gen'
$use = Join-Path $round2 'tgt\pkg-use'
$profraw = Join-Path $round2 'pgo\profraw-pkg'
$profileData = Join-Path $round2 'pgo\package.profdata'
$features = @('--no-default-features', '--features', 'mimalloc')
$unitSeparator = [string][char]0x1f

function Restore-Env([hashtable]$Saved) {
    foreach ($name in $Saved.Keys) {
        if ($null -eq $Saved[$name]) {
            Remove-Item -Path "env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item -Path "env:$name" -Value $Saved[$name]
        }
    }
}

function Invoke-Native([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Get-RemapPathFlags([string]$Workspace, [string[]]$TargetDirs) {
    # Keep in step with the same function in install-mc-rs.ps1.
    # A panic message names the source file, so without these flags a crash report shows the
    # folders of the PC that built the client (C:\Users\<account>\.cargo\registry\... for every
    # dependency). Each folder a source path can start with maps to a neutral name. The folders
    # are found here, on the PC that builds; none is typed in. rustc applies the last mapping
    # that matches, so the list runs from the shortest folder to the longest: cargo's home wins
    # over the user profile around it, the target folders over the workspace.
    $userProfile = if ($env:USERPROFILE) { $env:USERPROFILE } else { [Environment]::GetFolderPath('UserProfile') }
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $userProfile '.cargo' }
    $rustupHome = if ($env:RUSTUP_HOME) { $env:RUSTUP_HOME } else { Join-Path $userProfile '.rustup' }
    $map = @(
        @($userProfile, '~'), @((Split-Path -Parent $Workspace), 'project'), @($Workspace, 'mc-rs'),
        @($cargoHome, 'cargo'), @($rustupHome, 'rustup')
    ) + @($TargetDirs | ForEach-Object { , @($_, 'target') })
    $map | ForEach-Object {
        # cargo resolves a relative CARGO_HOME or RUSTUP_HOME from the workspace, where it runs.
        $from = if ([IO.Path]::IsPathRooted($_[0])) { $_[0] } else { Join-Path $Workspace $_[0] }
        [pscustomobject]@{ From = [IO.Path]::GetFullPath($from); To = $_[1] }
    } | Sort-Object { $_.From.Length }, From | ForEach-Object { "--remap-path-prefix=$($_.From)=$($_.To)" }
}

function Set-DefaultJarsDir {
    # Keep in step with the same function in install-mc-rs.ps1.
    # The training and smoke runs pass no --versionJar. An optimised client has no path to this
    # checkout's vanilla 1.8.9 files compiled in (it would name this PC's folders), so these runs
    # get MC_JARS_DIR, as the launcher passes --versionJar. One the caller set wins.
    if ($env:MC_JARS_DIR) { return }
    foreach ($jars in @((Join-Path $root 'MCP-919-main\jars'), (Join-Path $root 'jars'))) {
        if (Test-Path -LiteralPath $jars -PathType Container) { $env:MC_JARS_DIR = $jars; return }
    }
}

function Test-BytesContain([string]$Path, [string]$Ascii) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    $text = [Text.Encoding]::ASCII.GetString($bytes)
    return $text.IndexOf($Ascii, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Find-SignTool {
    $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    if (Test-Path -LiteralPath $kits) {
        $found = Get-ChildItem -LiteralPath $kits -Recurse -Filter 'signtool.exe' -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\' } | Sort-Object FullName -Descending | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    throw 'signtool.exe not found: install the Windows SDK signing tools (see CONTRIBUTING.md, "Packaging")'
}

# Optional Authenticode signing of the staged client (CONTRIBUTING.md, "Packaging"). A no-op unless a
# certificate thumbprint or the Artifact Signing dlib + metadata were given.
function Invoke-OptionalSigning([string]$AppDir) {
    $thumb = ($SignThumbprint -replace '\s', '')
    if ($thumb -eq '' -and $SignDlib -eq '') { return }
    $signtool = Find-SignTool
    $exePath = Join-Path $AppDir 'mc-client.exe'
    if ($thumb -ne '') {
        $tsa = if ($TimestampUrl -ne '') { $TimestampUrl } else { 'http://timestamp.digicert.com' }
        Write-Host "signing with certificate $thumb"
        Invoke-Native 'signing mc-client.exe' { & $signtool sign /fd SHA256 /sha1 $thumb /tr $tsa /td SHA256 $exePath }
        $cert = Get-ChildItem -LiteralPath "Cert:\CurrentUser\My\$thumb" -ErrorAction SilentlyContinue
        if ($cert) {
            $result = Set-AuthenticodeSignature -LiteralPath (Join-Path $AppDir 'launcher.ps1') -Certificate $cert `
                -HashAlgorithm SHA256 -TimestampServer $tsa
            if ($result.Status -ne 'Valid') { throw "signing launcher.ps1 failed: $($result.StatusMessage)" }
        } else {
            Write-Host 'note: the certificate is not in Cert:\CurrentUser\My, launcher.ps1 left unsigned' -ForegroundColor Yellow
        }
    } else {
        if ($SignMetadata -eq '' -or -not (Test-Path -LiteralPath $SignMetadata)) { throw '-SignDlib needs -SignMetadata (the Artifact Signing metadata JSON)' }
        if (-not (Test-Path -LiteralPath $SignDlib)) { throw "Artifact Signing dlib not found: $SignDlib" }
        $tsa = if ($TimestampUrl -ne '') { $TimestampUrl } else { 'http://timestamp.acs.microsoft.com' }
        Write-Host 'signing with Azure Artifact Signing'
        Invoke-Native 'signing mc-client.exe' { & $signtool sign /v /fd SHA256 /tr $tsa /td SHA256 /dlib $SignDlib /dmdf $SignMetadata $exePath }
    }
    $check = Get-AuthenticodeSignature -LiteralPath $exePath
    if ($check.Status -ne 'Valid') { throw "mc-client.exe signature check failed: $($check.Status) $($check.StatusMessage)" }
    Write-Host "signed: $($check.SignerCertificate.Subject)" -ForegroundColor Green
}

Import-Module (Join-Path $scripts 'common.psm1') -Force

# Same rule as install-mc-rs.ps1 (mc_client::app::config::discover_jars_dir); keep the two in step.
function Find-VanillaJar([string]$Root) {
    $dirs = if ($env:MC_JARS_DIR) { @($env:MC_JARS_DIR) } else {
        @((Join-Path $Root 'MCP-919-main\jars'), (Join-Path $Root 'jars'))
    }
    $dir = $dirs | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -First 1
    if (-not $dir) { return $null }
    foreach ($jar in @((Join-Path $dir 'versions\1.8.9\1.8.9.jar'), (Join-Path $dir 'versions\1.8.9.jar'))) {
        if (Test-Path -LiteralPath $jar -PathType Leaf) { return $jar }
    }
    return $null
}

if (-not $SkipBuild -and -not $SkipVanillaCheck -and -not (Find-VanillaJar $root)) {
    throw ('No Minecraft 1.8.9 jar where the PGO training runs look for it. Run .\setup-vanilla-files.ps1 ' +
        'once or set MC_JARS_DIR. -SkipVanillaCheck builds anyway, with a less representative profile.')
}

if (-not $SkipBuild) {
    $profdata = Join-Path (& rustc --print sysroot) 'lib\rustlib\x86_64-pc-windows-msvc\bin\llvm-profdata.exe'
    if (-not (Test-Path -LiteralPath $profdata)) {
        throw 'llvm-profdata is missing: run `rustup component add llvm-tools-preview`'
    }
    $saved = @{ CARGO_TARGET_DIR = $env:CARGO_TARGET_DIR; CARGO_ENCODED_RUSTFLAGS = $env:CARGO_ENCODED_RUSTFLAGS; MC_JARS_DIR = $env:MC_JARS_DIR }
    # One list for both builds, naming both build folders: a PGO profile matches functions by
    # name, so the instrumented and the optimised build must see exactly the same paths.
    $remap = @(Get-RemapPathFlags -Workspace $rs -TargetDirs @($gen, $use))
    Push-Location -LiteralPath $rs
    try {
        Write-Host "[1/6] instrumented build $(Get-Date -Format T)"
        # CARGO_ENCODED_RUSTFLAGS, not RUSTFLAGS: RUSTFLAGS splits a path that contains a space.
        # The instrumented build skips crt-static: the profile maps to the same Rust functions either
        # way, and the profiler runtime is not exercised against the static CRT.
        $env:CARGO_TARGET_DIR = $gen
        $env:CARGO_ENCODED_RUSTFLAGS = (@("-Cprofile-generate=$profraw") + $remap) -join $unitSeparator
        Invoke-Native 'instrumented build' { cargo build --profile dist -p mc-client @features --bin mc-client --example skywars_bench }

        Write-Host "[2/6] training $(Get-Date -Format T)"
        New-Item -ItemType Directory -Force $profraw | Out-Null
        Get-ChildItem -LiteralPath $profraw -Filter '*.profraw' | Remove-Item -Force
        $trainOut = Join-Path $round2 ('results\pkg-train-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        Set-DefaultJarsDir
        Invoke-Native 'training' {
            python -I (Join-Path $round2 'tools\pgo_train.py') (Join-Path $rs 'optimization') (Join-Path $gen 'dist') $trainOut
        }

        Write-Host "[3/6] merge $(Get-Date -Format T)"
        $raw = @(Get-ChildItem -LiteralPath $profraw -Filter '*.profraw' | ForEach-Object FullName)
        if ($raw.Count -eq 0) { throw 'training wrote no profiles' }
        Invoke-Native 'merge' { & $profdata merge -o $profileData @raw }

        Write-Host "[4/6] optimised build $(Get-Date -Format T)"
        $env:CARGO_TARGET_DIR = $use
        $env:CARGO_ENCODED_RUSTFLAGS = (@("-Cprofile-use=$profileData", '-Ctarget-feature=+crt-static') + $remap) -join $unitSeparator
        Invoke-Native 'optimised build' { cargo build --profile dist -p mc-client @features --bin mc-client }
    } finally {
        Pop-Location
        Restore-Env $saved
    }
}

$exe = Join-Path $use 'dist\mc-client.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "no built client at $exe" }

Write-Host "[5/6] checks $(Get-Date -Format T)"
foreach ($dll in @('VCRUNTIME140', 'MSVCP140', 'api-ms-win-crt-')) {
    if (Test-BytesContain $exe $dll) { throw "the client still imports $dll; the C runtime must be static" }
}
if (Test-BytesContain $exe 'OfflinePlayer:') { throw 'the client still contains the dev-only offline login' }
# Players see panic messages, and every string of the exe ships to them: nothing in it may name
# this PC or its owner (Get-RemapPathFlags maps the build folders). Checked: the profile, cargo's
# home and the projects folder in every separator form, the account, this PC's name and
# addresses, and the git e-mail, as ASCII/UTF-8 and as UTF-16LE at both byte alignments. The
# owner's player names, UUIDs and servers are not known here: the private gate checks the zip.
$cargoHomeDir = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
$needles = New-Object System.Collections.Generic.List[string]
foreach ($folder in @($env:USERPROFILE, $cargoHomeDir, (Split-Path -Parent $root)) | Where-Object { $_ }) {
    foreach ($form in @($folder, $folder.Replace('\', '/'), $folder.Replace('\', '\\'))) { $needles.Add($form) }
}
$gitEmail = if (Get-Command git -ErrorAction SilentlyContinue) { git config --global user.email } else { $null }
$addresses = @([System.Net.Dns]::GetHostAddresses($env:COMPUTERNAME) |
    Where-Object { -not [System.Net.IPAddress]::IsLoopback($_) -and -not $_.IsIPv6LinkLocal } |
    ForEach-Object { $_.IPAddressToString })
foreach ($value in @($env:USERNAME, $env:COMPUTERNAME, $gitEmail) + $addresses) {
    if ($value -and $value.Length -ge 4) { $needles.Add($value) }
}
$bytes = [IO.File]::ReadAllBytes($exe)
$texts = @([Text.Encoding]::ASCII.GetString($bytes), [Text.Encoding]::Unicode.GetString($bytes))
if ($bytes.Length -gt 1) { $texts += [Text.Encoding]::Unicode.GetString($bytes, 1, $bytes.Length - 1) }
foreach ($needle in $needles) {
    foreach ($text in $texts) {
        if ($text.IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            throw 'the client still names this PC or its owner (a folder, the account, the PC name or address, or the e-mail)'
        }
    }
}
# Players of the zip sign in only on the client's Accounts screen, which needs the Azure
# application ID compiled in (CONTRIBUTING.md, "Sign-in").
if (-not $env:AERO_MSA_CLIENT_ID) {
    Write-Host 'note: AERO_MSA_CLIENT_ID is not set, so the players of this zip cannot sign in (CONTRIBUTING.md, "Sign-in")' -ForegroundColor Yellow
} elseif (-not (Test-BytesContain $exe $env:AERO_MSA_CLIENT_ID.Trim())) {
    Write-Host 'note: the client was built without the AERO_MSA_CLIENT_ID set now; build again (without -SkipBuild) so its players can sign in' -ForegroundColor Yellow
}
# The zip's launcher: what it trusts, and exactly what it passes to the game (its folder and the
# 1.8.9 files, never an account or a session). A few seconds, in a temporary folder, with a
# stand-in game; Windows PowerShell, whose Add-Type compiles that stand-in.
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts 'test-launcher-trust.ps1')
if ($LASTEXITCODE -ne 0) { throw 'the launcher checks failed (packaging\scripts\test-launcher-trust.ps1)' }
# The title-screen smoke run needs the vanilla 1.8.9 files: the copy setup-vanilla-files.ps1 made
# (or MC_JARS_DIR), else the official launcher's folder.
$jar = $null
$assets = $null
foreach ($dir in @($env:MC_JARS_DIR, (Join-Path $root 'MCP-919-main\jars'), (Join-Path $root 'jars'),
        $(if ($env:APPDATA) { Join-Path $env:APPDATA '.minecraft' })) | Where-Object { $_ }) {
    $candidate = Join-Path $dir 'versions\1.8.9\1.8.9.jar'
    if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $dir 'assets\indexes\1.8.json') -PathType Leaf)) {
        $jar = $candidate
        $assets = Join-Path $dir 'assets'
        break
    }
}
$smoke = Join-Path ([IO.Path]::GetTempPath()) ('aero-package-smoke-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $smoke | Out-Null
$savedSmoke = @{ MC_PIPELINED_SUBMIT = $env:MC_PIPELINED_SUBMIT; MC_BACKGROUND_WINDOW = $env:MC_BACKGROUND_WINDOW; MC_NO_AUDIO = $env:MC_NO_AUDIO; MC_JARS_DIR = $env:MC_JARS_DIR }
try {
    $env:MC_PIPELINED_SUBMIT = '1'; $env:MC_BACKGROUND_WINDOW = '1'; $env:MC_NO_AUDIO = '1'
    Set-DefaultJarsDir
    $capture = Join-Path $rs 'crates\mc-parity\captures\spigot-1.8.8-perfworld-vd16.mccap'
    $proc = Start-Process -FilePath $exe -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $smoke `
        -RedirectStandardOutput (Join-Path $smoke 'replay.out.log') -RedirectStandardError (Join-Path $smoke 'replay.log') `
        -ArgumentList @('--replay', "`"$capture`"", '--replayCamera', 'orbit', '--width', '1280', '--height', '720',
            '--offline', 'BenchBot', '--frames', '600', '--gameDir', "`"$smoke`"")
    if ($proc.ExitCode -ne 0) { throw "replay smoke run failed (exit code $($proc.ExitCode)); logs in $smoke" }
    if ($jar) {
        # The title screen with the Aero menu scene, which bakes vanilla 1.8.9 from the jar.
        $title = Join-Path $smoke 'title'
        New-Item -ItemType Directory $title | Out-Null
        $proc = Start-Process -FilePath $exe -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $title `
            -RedirectStandardOutput (Join-Path $smoke 'title.out.log') -RedirectStandardError (Join-Path $smoke 'title.log') `
            -ArgumentList @('--offline', 'TitleCheck', '--frames', '600', '--width', '1280', '--height', '720',
                '--assetsDir', "`"$assets`"", '--assetIndex', '1.8', '--versionJar', "`"$jar`"", '--gameDir', "`"$title`"")
        if ($proc.ExitCode -ne 0) { throw "title-screen smoke run failed (exit code $($proc.ExitCode)); logs in $smoke" }
    } else {
        Write-Host 'note: no Minecraft 1.8.9 files found (run .\setup-vanilla-files.ps1), title-screen smoke run skipped' -ForegroundColor Yellow
    }
} finally {
    Restore-Env $savedSmoke
}
Remove-Item -LiteralPath $smoke -Recurse -Force

Write-Host "[6/6] stage and zip $(Get-Date -Format T)"
New-Item -ItemType Directory -Force $outDir | Out-Null
$stage = Join-Path $outDir 'Aero Client'
New-StagedPayload -Stage $stage
$appDir = Join-Path $stage 'app'
New-Item -ItemType Directory -Force (Join-Path $appDir 'licenses') | Out-Null
Copy-Item -LiteralPath $exe -Destination (Join-Path $appDir 'mc-client.exe')
Copy-Item -LiteralPath (Join-Path $templates 'launcher.ps1') -Destination (Join-Path $appDir 'launcher.ps1')
Copy-Item -LiteralPath (Join-Path $templates 'Play Aero Client.cmd') -Destination (Join-Path $stage 'Play Aero Client.cmd')
Copy-Item -LiteralPath (Join-Path $templates 'README.txt') -Destination (Join-Path $stage 'README.txt')
Copy-Item -LiteralPath (Join-Path $rs 'crates\mc-ui\assets\fonts\inter\OFL.txt') -Destination (Join-Path $appDir 'licenses\Inter-font-OFL.txt')
# Aero Client's own licence (GPL-3.0-or-later): the GPL version 3 text, next to NOTICE.txt.
Copy-Item -LiteralPath (Join-Path $rs 'LICENSE') -Destination (Join-Path $appDir 'LICENSE.txt')
# The licences of everything compiled into the client (its Rust libraries for exactly this feature
# set and target, the Inter typeface, the fdlibm notice), generated offline from the lock file, so
# it always matches the build.
Invoke-Native 'third-party notices' {
    python -I (Join-Path $scripts 'third_party_notices.py') (Join-Path $appDir 'licenses\THIRD-PARTY-NOTICES.txt') @features
}
# Before BUILD-INFO.txt, which records the shipped binary's hash.
Invoke-OptionalSigning -AppDir $appDir
# Where the source code is published: the repository this build's updater reads releases from
# (mc-update's DEFAULT_REPO, or AERO_UPDATE_REPO when the build set it), so the notice follows it.
$sourceRepo = $env:AERO_UPDATE_REPO
if (-not $sourceRepo) {
    $repoLine = Select-String -LiteralPath (Join-Path $rs 'crates\mc-update\src\lib.rs') `
        -Pattern '^pub const DEFAULT_REPO: &str = "([^"]+)";' | Select-Object -First 1
    if ($repoLine) { $sourceRepo = $repoLine.Matches[0].Groups[1].Value }
}
if ($sourceRepo -notmatch '^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$') {
    throw "cannot tell the source repository for NOTICE.txt (got '$sourceRepo')"
}
$noticeText = @'
Aero Client - NOTICE

Aero Client is an independent client. It is not an official Minecraft product and is not approved
by or associated with Mojang Studios or Microsoft. Minecraft is a trademark of Mojang AB.

Copyright (C) 2026 pvps and contributors.

Aero Client is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version (GPL-3.0-or-later). It comes WITHOUT ANY WARRANTY;
see the licence for details. The licence text is LICENSE.txt in this folder. The source code is
published at https://github.com/{REPO}.

It contains no Minecraft game files. It reads the Minecraft 1.8.9 files already on your own PC
(in the official launcher's folder, or a folder you choose), and does not copy or share them. You
need your own Minecraft Java Edition account.

Sign-in: you sign in inside the game, on its Accounts screen, with your Microsoft account. The
saved sign-in stays on this PC for your Windows user, outside the game folder. The launcher reads
no account and passes no login to the game. Aero Client sends your login only to Microsoft's,
Xbox Live's and Minecraft's own services, and never prints it or writes it to a log.

The title uses the Inter typeface, Copyright 2020 The Inter Project Authors, under the SIL Open
Font License 1.1 (licenses\Inter-font-OFL.txt).

The client also contains open-source Rust libraries. Their licences, notices and attributions are
in licenses\THIRD-PARTY-NOTICES.txt.
'@
Set-Content -LiteralPath (Join-Path $appDir 'NOTICE.txt') -Encoding UTF8 -Value $noticeText.Replace('{REPO}', $sourceRepo)
Write-BuildInfo -Stage $appDir -Version (Get-WorkspaceVersion -Workspace $rs) -Triple 'windows-x86_64' `
    -Profile 'dist (PGO, crt-static)' -Binary (Join-Path $appDir 'mc-client.exe') -DefaultFeatures 'off' -Features 'mimalloc'
# The updater's install marker (mc-update, install.rs): only a client next to it is a packaged
# portable install that checks GitHub releases and may update itself; development builds have
# none. UTF-8 without a byte order mark. Releases: see crates\mc-update\src\bin\aero-update-sign.rs.
$installMarker = [ordered]@{
    format = 1; product = 'Aero Client'; version = (Get-WorkspaceVersion -Workspace $rs); platform = 'windows-x64'
} | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $appDir 'aero-install.json'), $installMarker, (New-Object Text.UTF8Encoding $false))
if ($env:AERO_UPDATE_PUBKEY) {
    Write-Host 'updates: signed releases only (AERO_UPDATE_PUBKEY is compiled in)'
} else {
    Write-Host 'note: no AERO_UPDATE_PUBKEY, so this client announces updates but never installs them' -ForegroundColor Yellow
}

& (Join-Path $scripts 'verify-payload.ps1') -Payload $appDir -ExpectBinary 'mc-client.exe'
if ($LASTEXITCODE -ne 0) { throw 'payload verification failed (app)' }
& (Join-Path $scripts 'verify-payload.ps1') -Payload $stage
if ($LASTEXITCODE -ne 0) { throw 'payload verification failed (package)' }

$zip = Join-Path $outDir 'Aero Client.zip'
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
Write-Host ''
Write-Host ("Built {0} ({1:N1} MB)" -f $zip, ((Get-Item -LiteralPath $zip).Length / 1MB)) -ForegroundColor Green
