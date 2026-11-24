#requires -Version 5.1
<#
.SYNOPSIS
Build what other people download: the portable zip release\Aero Client.zip and, when Inno Setup 6
is installed, the installer release\Aero-Client-Setup.exe.

.DESCRIPTION
The zip holds one "Aero Client" folder:
  Aero Client.exe        double-click to play: starts app\mc-client.exe (mc-shell's aero-start)
  README.txt             setup in plain words
  app\mc-client.exe      the client: PGO-optimised like the installed one. Started with no arguments
                         it finds the player's Minecraft 1.8.9 files (asking once), keeps its logs in
                         game\logs and shows a plain message when something goes wrong
  app\LICENSE.txt        Aero Client's licence, the GPL-3.0 text (a copy of mc-rs\LICENSE)
  app\NOTICE.txt, app\BUILD-INFO.txt, app\licenses\ (the font licence and THIRD-PARTY-NOTICES.txt)
  app\aero-install.json  the updater's marker: a packaged portable install of this version
  app\aero-client.ico    the logo as a program icon (the installer's shortcuts and Settings > Apps)
Each player's settings, logs and screenshots go in a "game" folder created on first start. Both
programs are windowed (no console window opens), and the package holds no script of any kind
(no PowerShell, no .cmd or .bat).

The installer (mc-rs\packaging\aero\installer\aero-client.iss) installs exactly that folder for the
current Windows user, into %LOCALAPPDATA%\Programs\Aero Client or a folder the player chooses, with
Start menu and desktop shortcuts to app\mc-client.exe and an uninstaller, so the in-game updater
updates it like an unzipped copy. It is built with Inno Setup 6's ISCC.exe: $env:ISCC, else Inno
Setup 6 installed for this user or for everyone. Without it the script builds the zip only and
says so.

How the shared client differs from the one install-mc-rs.ps1 installs for you:
  - --no-default-features --features mimalloc,windowed: the dev-only "offline" login (no sign-in)
    is compiled out, as CONTRIBUTING.md ("Packaging") requires for anything distributed, and the
    client is a windowed program (Windows' GUI subsystem), so no console window opens beside it;
  - -Ctarget-feature=+crt-static: the C runtime is built in, so players don't need the Visual C++
    Redistributable.
Aero Client.exe is built the same way (crt-static, its own build folder), without PGO: it only
starts the client. Every build maps this PC's folders (user profile, cargo's home, this project,
the build folders) to neutral names with --remap-path-prefix, so a player's crash report names no
folder of your PC. It never touches target\release (your installed client) or your settings.

Steps (about 35 minutes; separate build folders, so your next install-mc-rs.ps1 stays fast):
  1. instrumented build           (optimization\scratch\round2-2026-11-05\tgt\pkg-gen)
  2. training on the benchmark scenes (as install-mc-rs.ps1)
  3. merge the profile            (...\pgo\package.profdata)
  4. optimised build              (...\tgt\pkg-use), then Aero Client.exe (...\tgt\pkg-starter)
  5. checks: windowed programs, no Visual C++ runtime import, no offline login, nothing that names
     this PC, the notices cover Aero Client.exe, a replay smoke run and a title-screen run
  6. stage, verify that the package is complete and holds no Minecraft file and no script
     (packaging\scripts\verify-payload.ps1), zip, and build the installer from the same staged
     folder
The release files for GitHub come from both afterwards: mc-rs\packaging\scripts\release-assets.ps1.

.PARAMETER SkipBuild
Re-stage and re-zip the last build (the client and Aero Client.exe) without compiling.

.PARAMETER SkipVanillaCheck
Build even though no Minecraft 1.8.9 jar is where the PGO training runs look for one (see
install-mc-rs.ps1). The profile then fits the real game less well.

.PARAMETER SignThumbprint
Optional Authenticode signing (see CONTRIBUTING.md, "Packaging"): the thumbprint of a code signing
certificate in Cert:\CurrentUser\My (hardware token, cloud signing service, any CA). Signs
Aero Client.exe and app\mc-client.exe before zipping, and the installer (Setup and its
uninstaller). Also read from $env:AERO_SIGN_THUMBPRINT. Off when empty.

.PARAMETER SignDlib
Optional Azure Artifact Signing: path to Microsoft's Azure.CodeSigning.Dlib.dll. Needs
-SignMetadata too. Signs Aero Client.exe, app\mc-client.exe and the installer. Also read from
$env:AERO_SIGN_DLIB.

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
$starterBuild = Join-Path $round2 'tgt\pkg-starter'
$profraw = Join-Path $round2 'pgo\profraw-pkg'
$profileData = Join-Path $round2 'pgo\package.profdata'
# The same features for the instrumented and the optimised build: the profile must fit.
$featureList = 'mimalloc,windowed'
$features = @('--no-default-features', '--features', $featureList)
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

# The Subsystem field of a Windows program's PE header: 2 for a windowed program, 3 for a console
# one (Windows opens a console window beside a console program started from Explorer or a shortcut).
function Get-PeSubsystem([string]$Path) {
    $head = New-Object byte[] 4096
    $stream = [IO.File]::OpenRead($Path)
    try { $read = $stream.Read($head, 0, $head.Length) } finally { $stream.Dispose() }
    $pe = if ($read -ge 64) { [BitConverter]::ToInt32($head, 0x3C) } else { -1 }
    if ($pe -lt 0 -or $pe + 94 -gt $read -or [Text.Encoding]::ASCII.GetString($head, $pe, 4) -ne "PE`0`0") {
        throw "$Path is not a Windows program"
    }
    return [BitConverter]::ToUInt16($head, $pe + 92)
}

# The crates.io libraries a package links into its program, as "name vX.Y.Z" lines (the workspace's
# own crates carry their folder and are left out): the same cargo tree third_party_notices.py reads.
function Get-LinkedCrates([string[]]$Selection) {
    Push-Location -LiteralPath $rs
    try {
        $lines = & cargo tree @Selection -e normal,no-proc-macro --target x86_64-pc-windows-msvc --prefix none --format '{p}' --offline --locked
        if ($LASTEXITCODE -ne 0) { throw "cargo tree $($Selection -join ' ') failed (exit code $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
    return @($lines | Where-Object { $_ -match '^[A-Za-z0-9_-]+ v[0-9][^ ]*$' } | Sort-Object -Unique)
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

# Optional Authenticode signing of the staged programs (CONTRIBUTING.md, "Packaging"). A no-op unless
# a certificate thumbprint or the Artifact Signing dlib + metadata were given.
function Invoke-OptionalSigning([string[]]$Programs) {
    $thumb = ($SignThumbprint -replace '\s', '')
    if ($thumb -eq '' -and $SignDlib -eq '') { return }
    $signtool = Find-SignTool
    if ($thumb -ne '') {
        $tsa = if ($TimestampUrl -ne '') { $TimestampUrl } else { 'http://timestamp.digicert.com' }
        Write-Host "signing with certificate $thumb"
    } else {
        if ($SignMetadata -eq '' -or -not (Test-Path -LiteralPath $SignMetadata)) { throw '-SignDlib needs -SignMetadata (the Artifact Signing metadata JSON)' }
        if (-not (Test-Path -LiteralPath $SignDlib)) { throw "Artifact Signing dlib not found: $SignDlib" }
        $tsa = if ($TimestampUrl -ne '') { $TimestampUrl } else { 'http://timestamp.acs.microsoft.com' }
        Write-Host 'signing with Azure Artifact Signing'
    }
    foreach ($program in $Programs) {
        $name = Split-Path -Leaf $program
        if ($thumb -ne '') {
            Invoke-Native "signing $name" { & $signtool sign /fd SHA256 /sha1 $thumb /tr $tsa /td SHA256 $program }
        } else {
            Invoke-Native "signing $name" { & $signtool sign /v /fd SHA256 /tr $tsa /td SHA256 /dlib $SignDlib /dmdf $SignMetadata $program }
        }
        $check = Get-AuthenticodeSignature -LiteralPath $program
        if ($check.Status -ne 'Valid') { throw "$name signature check failed: $($check.Status) $($check.StatusMessage)" }
        Write-Host "signed $($name): $($check.SignerCertificate.Subject)" -ForegroundColor Green
    }
}

# The command Inno Setup runs to sign Setup and its uninstaller, with the same certificate or
# Artifact Signing account as the client; empty when not signing. Inno Setup puts the file name in
# for $f and a quote for $q (a literal quote would not survive the trip through its command line).
function Get-InstallerSignCommand {
    $thumb = ($SignThumbprint -replace '\s', '')
    if ($thumb -eq '' -and $SignDlib -eq '') { return '' }
    $signtool = Find-SignTool
    if ($thumb -ne '') {
        $tsa = if ($TimestampUrl -ne '') { $TimestampUrl } else { 'http://timestamp.digicert.com' }
        return '$q' + $signtool + '$q sign /fd SHA256 /sha1 ' + $thumb + ' /tr ' + $tsa + ' /td SHA256 $f'
    }
    $tsa = if ($TimestampUrl -ne '') { $TimestampUrl } else { 'http://timestamp.acs.microsoft.com' }
    return '$q' + $signtool + '$q sign /v /fd SHA256 /tr ' + $tsa + ' /td SHA256 /dlib $q' + $SignDlib +
        '$q /dmdf $q' + $SignMetadata + '$q $f'
}

# Inno Setup 6's command-line compiler: $env:ISCC when set, else Inno Setup 6 installed for this
# user, for everyone (where GitHub's Windows runners have it) or on PATH; $null when there is none.
function Find-Iscc {
    if ($env:ISCC) {
        if (-not (Test-Path -LiteralPath $env:ISCC -PathType Leaf)) { throw "ISCC is set, but there is no $env:ISCC" }
        return $env:ISCC
    }
    $candidates = @()
    if ($env:LOCALAPPDATA) { $candidates += Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe' }
    if (${env:ProgramFiles(x86)}) { $candidates += Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe' }
    if ($env:ProgramFiles) { $candidates += Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe' }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    return $null
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

        # Aero Client.exe only starts the client, so no profile; but the same static C runtime and
        # neutral folder names, in its own build folder (the client's stays as the profile built it).
        Write-Host "      Aero Client.exe $(Get-Date -Format T)"
        $env:CARGO_TARGET_DIR = $starterBuild
        $env:CARGO_ENCODED_RUSTFLAGS = (@('-Ctarget-feature=+crt-static') +
            @(Get-RemapPathFlags -Workspace $rs -TargetDirs @($starterBuild))) -join $unitSeparator
        Invoke-Native 'Aero Client.exe build' { cargo build --profile dist -p mc-shell --bin aero-start }
    } finally {
        Pop-Location
        Restore-Env $saved
    }
}

$exe = Join-Path $use 'dist\mc-client.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "no built client at $exe" }
$starterExe = Join-Path $starterBuild 'dist\aero-start.exe'
if (-not (Test-Path -LiteralPath $starterExe -PathType Leaf)) { throw "no built Aero Client.exe at $starterExe" }
$programs = @(
    [pscustomobject]@{ Path = $exe; Name = 'the client' },
    [pscustomobject]@{ Path = $starterExe; Name = 'Aero Client.exe (aero-start)' }
)

Write-Host "[5/6] checks $(Get-Date -Format T)"
foreach ($program in $programs) {
    # A console program opens a console window next to the game. Both are built as windowed
    # programs (the client with the windowed feature).
    if ((Get-PeSubsystem $program.Path) -ne 2) { throw "$($program.Name) is not a windowed program (Windows GUI subsystem)" }
    # The logo as the file's icon comes in one resource with the version information, compiled by
    # the build scripts with the Windows SDK's rc.exe; without rc.exe cargo only warns.
    if (-not (Get-Item -LiteralPath $program.Path).VersionInfo.FileVersion) {
        throw "$($program.Name) was built without its icon and version information (the Windows SDK's rc.exe was not found)"
    }
    foreach ($dll in @('VCRUNTIME140', 'MSVCP140', 'api-ms-win-crt-')) {
        if (Test-BytesContain $program.Path $dll) { throw "$($program.Name) still imports $dll; the C runtime must be static" }
    }
}
if (Test-BytesContain $exe 'OfflinePlayer:') { throw 'the client still contains the dev-only offline login' }
# Players see panic messages, and every string of both programs ships to them: nothing in them may
# name this PC or its owner (Get-RemapPathFlags maps the build folders). Checked: the profile,
# cargo's home and the projects folder in every separator form, the account, this PC's name and
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
foreach ($program in $programs) {
    $bytes = [IO.File]::ReadAllBytes($program.Path)
    $texts = @([Text.Encoding]::ASCII.GetString($bytes), [Text.Encoding]::Unicode.GetString($bytes))
    if ($bytes.Length -gt 1) { $texts += [Text.Encoding]::Unicode.GetString($bytes, 1, $bytes.Length - 1) }
    foreach ($needle in $needles) {
        foreach ($text in $texts) {
            if ($text.IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                throw "$($program.Name) still names this PC or its owner (a folder, the account, the PC name or address, or the e-mail)"
            }
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
# THIRD-PARTY-NOTICES.txt is written for the client's libraries; Aero Client.exe ships beside it, so
# every library it links must be one the client links too.
$clientCrates = Get-LinkedCrates (@('-p', 'mc-client') + $features)
$starterOnly = @(Get-LinkedCrates @('-p', 'mc-shell') | Where-Object { $clientCrates -notcontains $_ })
if ($starterOnly.Count -gt 0) {
    throw "Aero Client.exe links libraries that THIRD-PARTY-NOTICES.txt would not list: $($starterOnly -join ', ')"
}
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
# Both runs pass the client its arguments, so it starts the way a developer starts it, from the build
# folder: the staged copy would see its updater marker and act as a player's install.
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

Write-Host "[6/6] stage, zip and installer $(Get-Date -Format T)"
New-Item -ItemType Directory -Force $outDir | Out-Null
$stage = Join-Path $outDir 'Aero Client'
New-StagedPayload -Stage $stage
$appDir = Join-Path $stage 'app'
New-Item -ItemType Directory -Force (Join-Path $appDir 'licenses') | Out-Null
Copy-Item -LiteralPath $exe -Destination (Join-Path $appDir 'mc-client.exe')
Copy-Item -LiteralPath $starterExe -Destination (Join-Path $stage 'Aero Client.exe')
Copy-Item -LiteralPath (Join-Path $templates 'README.txt') -Destination (Join-Path $stage 'README.txt')
# The logo as a program icon, drawn from the logo renders in crates\mc-client\assets\icon. It goes
# in app\, so the zip, the installer and every update carry it: the installer's shortcuts point at
# it, and an update replaces app\ as a whole.
$makeInstallerAssets = Join-Path $templates 'installer\make-installer-assets.ps1'
$iconOut = Join-Path ([IO.Path]::GetTempPath()) ('aero-icon-' + [guid]::NewGuid().ToString('N'))
try {
    & $makeInstallerAssets -OutDir $iconOut
    Copy-Item -LiteralPath (Join-Path $iconOut 'aero-client.ico') -Destination (Join-Path $appDir 'aero-client.ico')
} finally {
    Remove-Item -LiteralPath $iconOut -Recurse -Force -ErrorAction SilentlyContinue
}
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
Invoke-OptionalSigning -Programs @((Join-Path $stage 'Aero Client.exe'), (Join-Path $appDir 'mc-client.exe'))
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
saved sign-in stays on this PC for your Windows user, outside the game folder. Aero Client reads
no other program's accounts, sends your login only to Microsoft's, Xbox Live's and Minecraft's
own services, and never prints it or writes it to a log.

The title uses the Inter typeface, Copyright 2020 The Inter Project Authors, under the SIL Open
Font License 1.1 (licenses\Inter-font-OFL.txt).

The client also contains open-source Rust libraries. Their licences, notices and attributions are
in licenses\THIRD-PARTY-NOTICES.txt.
'@
Set-Content -LiteralPath (Join-Path $appDir 'NOTICE.txt') -Encoding UTF8 -Value $noticeText.Replace('{REPO}', $sourceRepo)
Write-BuildInfo -Stage $appDir -Version (Get-WorkspaceVersion -Workspace $rs) -Triple 'windows-x86_64' `
    -Profile 'dist (PGO, crt-static)' -Binary (Join-Path $appDir 'mc-client.exe') -DefaultFeatures 'off' -Features $featureList
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
# The whole package: what players start and what the updater and the installer need, and no script
# anywhere, PowerShell or batch (the client starts itself; Aero Client.exe only starts it).
& (Join-Path $scripts 'verify-payload.ps1') -Payload $stage -ExpectFiles @('Aero Client.exe', 'README.txt',
    'app\mc-client.exe', 'app\aero-install.json', 'app\aero-client.ico') -ForbidExtensions @('.ps1', '.psm1', '.psd1', '.cmd', '.bat')
if ($LASTEXITCODE -ne 0) { throw 'payload verification failed (package)' }

$zip = Join-Path $outDir 'Aero Client.zip'
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip

# The installer, from the same staged folder. An old one goes first, so a failed or skipped build
# never leaves last time's installer next to this zip.
$setup = Join-Path $outDir 'Aero-Client-Setup.exe'
if (Test-Path -LiteralPath $setup) { Remove-Item -LiteralPath $setup -Force }
$iscc = Find-Iscc
if ($iscc) {
    # The wizard's pictures, drawn from the logo like the icon; they stay outside the package.
    $wizardImages = Join-Path ([IO.Path]::GetTempPath()) ('aero-wizard-' + [guid]::NewGuid().ToString('N'))
    try {
        & $makeInstallerAssets -OutDir $wizardImages
        $isccArgs = @('/Qp', "/DAppVersion=$(Get-WorkspaceVersion -Workspace $rs)", "/DSourceDir=$stage",
            "/DOutputDir=$outDir", "/DAssetDir=$wizardImages")
        $signCommand = Get-InstallerSignCommand
        if ($signCommand -ne '') { $isccArgs += @('/DSign', "/Saerosign=$signCommand") }
        Invoke-Native 'building the installer' { & $iscc @isccArgs (Join-Path $templates 'installer\aero-client.iss') }
    } finally {
        Remove-Item -LiteralPath $wizardImages -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath $setup -PathType Leaf)) { throw "Inno Setup wrote no $setup" }
    if ($signCommand -ne '') {
        $check = Get-AuthenticodeSignature -LiteralPath $setup
        if ($check.Status -ne 'Valid') { throw "installer signature check failed: $($check.Status) $($check.StatusMessage)" }
    }
}

Write-Host ''
Write-Host ("Built {0} ({1:N1} MB)" -f $zip, ((Get-Item -LiteralPath $zip).Length / 1MB)) -ForegroundColor Green
if ($iscc) {
    Write-Host ("Built {0} ({1:N1} MB)" -f $setup, ((Get-Item -LiteralPath $setup).Length / 1MB)) -ForegroundColor Green
} else {
    Write-Host ('note: Inno Setup 6 was not found, so there is no installer (release\Aero-Client-Setup.exe). ' +
        'Install it from https://jrsoftware.org/isdl.php (or set ISCC to its ISCC.exe), then run this ' +
        'script again with -SkipBuild') -ForegroundColor Yellow
}
