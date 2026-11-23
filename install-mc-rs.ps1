#requires -Version 5.1
<#
.SYNOPSIS
Build mc-rs with profile-guided optimisation (PGO) from the current code and install it as the
client that launch-mc-rs.ps1 runs. This is THE way to install a new mc-rs build.

.DESCRIPTION
Why this script exists:
- A plain `cargo build --release` gives a slower client: PGO measured -7.8% frame time in SkyWars,
  -8.2% in the entity scene and lower input latency (2026-11-06).
- A PGO profile only fits the code it was trained on, so EVERY code change needs this whole script
  again. Nobody has to remember that: launch-mc-rs.ps1 checks the stamp this script writes
  (target\release\mc-client.install.json) and warns on every launch when the code has changed
  since, or when the client was replaced by a build that did not come from here.
- Both builds map this PC's folders (user profile, cargo's home, this project, the build folders)
  to neutral names with --remap-path-prefix, so the client's panic messages and crash reports
  name no folder of the PC that built it.

Needs (see CONTRIBUTING.md):Rust with `rustup component add llvm-tools-preview`, Python 3 on PATH as
`python`, and your own Minecraft 1.8.9 files where the client finds them (run
.\setup-vanilla-files.ps1 once): the training runs render the vanilla textures.

Steps (about 25 minutes; the PC does not need to be idle):
  1. instrumented build          (optimization\scratch\round2-2026-11-05\tgt\pgo-gen)
  2. training runs on the benchmark scenes (SkyWars pipelined + serial, dense terrain fixed +
     orbit, the e1-16 entity scene)
  3. merge the profile           (llvm-profdata from `rustup component add llvm-tools-preview`)
  4. optimised build             (...\tgt\pgo-use)
  5. smoke test of the new client (short replay, pipelined submit on)
  6. back up the old client to optimization\installed-backups\, install, write the stamp

.PARAMETER Prebuilt
Install an already PGO-built folder holding mc-client.exe (e.g. the output of
optimization\scratch\round2-2026-11-05\tools\pgo_build.sh) instead of building. Refused if any
source file is newer than that mc-client.exe.

.PARAMETER SkipVanillaCheck
Build even though no Minecraft 1.8.9 jar is where the client looks for one. The training runs then
render untextured scenes and the profile fits the real game less well.

.EXAMPLE
.\install-mc-rs.ps1
#>
[CmdletBinding()]
param([string]$Prebuilt, [switch]$SkipVanillaCheck)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$rs = Join-Path $root 'mc-rs'
$round2 = Join-Path $rs 'optimization\scratch\round2-2026-11-05'
$release = Join-Path $rs 'target\release'
$installed = Join-Path $release 'mc-client.exe'
$stampPath = Join-Path $release 'mc-client.install.json'
$unitSeparator = [string][char]0x1f

function Get-NewestSource([string]$McRs) {
    # Keep in step with the same function in launch-mc-rs.ps1.
    $files = @(Get-ChildItem -LiteralPath (Join-Path $McRs 'crates') -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in '.rs', '.wgsl', '.toml' })
    $files += @(Get-Item -LiteralPath (Join-Path $McRs 'Cargo.toml'), (Join-Path $McRs 'Cargo.lock'),
        (Join-Path $McRs '.cargo\config.toml') -ErrorAction SilentlyContinue)
    $newest = $files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    [pscustomobject]@{
        Ticks = $newest.LastWriteTimeUtc.Ticks
        File  = $newest.FullName.Substring($McRs.Length + 1)
        Count = $files.Count
    }
}

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

# Where the client finds the vanilla 1.8.9 jar when it is given no --versionJar, as the training
# runs are (mc_client::app::config::discover_jars_dir): MC_JARS_DIR, else the first of
# MCP-919-main\jars and jars\ in this folder that exists. Keep in step with package-aero-client.ps1.
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

function Get-RemapPathFlags([string]$Workspace, [string[]]$TargetDirs) {
    # Keep in step with the same function in package-aero-client.ps1.
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
    # Keep in step with the same function in package-aero-client.ps1.
    # The training and smoke runs pass no --versionJar. An optimised client has no path to this
    # checkout's vanilla 1.8.9 files compiled in (it would name this PC's folders), so these runs
    # get MC_JARS_DIR, as the launcher passes --versionJar. One the caller set wins.
    if ($env:MC_JARS_DIR) { return }
    foreach ($jars in @((Join-Path $root 'MCP-919-main\jars'), (Join-Path $root 'jars'))) {
        if (Test-Path -LiteralPath $jars -PathType Container) { $env:MC_JARS_DIR = $jars; return }
    }
}

if (Get-Process -Name mc-client -ErrorAction SilentlyContinue) {
    throw 'mc-client is running; close the game first.'
}

if (-not $Prebuilt -and -not $SkipVanillaCheck -and -not (Find-VanillaJar $root)) {
    throw ('No Minecraft 1.8.9 jar where the PGO training runs look for it. Run .\setup-vanilla-files.ps1 ' +
        'once (it copies your own 1.8.9 files from the official launcher''s folder, or one you name, into MCP-919-main\jars, ' +
        'which git ignores) or set MC_JARS_DIR. -SkipVanillaCheck builds anyway, with a less representative profile.')
}

if ($Prebuilt) {
    $newExe = Join-Path $Prebuilt 'mc-client.exe'
    if (-not (Test-Path -LiteralPath $newExe)) { throw "no mc-client.exe in $Prebuilt" }
    $source = Get-NewestSource $rs
    if ($source.Ticks -gt (Get-Item -LiteralPath $newExe).LastWriteTimeUtc.Ticks) {
        throw "the code changed after $newExe was built ($($source.File) is newer); run without -Prebuilt"
    }
    $profileData = Join-Path $round2 'pgo\merged.profdata'
    $builtBy = "prebuilt: $Prebuilt"
} else {
    $source = Get-NewestSource $rs
    $profdata = Join-Path (& rustc --print sysroot) 'lib\rustlib\x86_64-pc-windows-msvc\bin\llvm-profdata.exe'
    if (-not (Test-Path -LiteralPath $profdata)) {
        throw 'llvm-profdata is missing: run `rustup component add llvm-tools-preview`'
    }
    $profraw = Join-Path $round2 'pgo\profraw'
    $profileData = Join-Path $round2 'pgo\merged.profdata'
    $gen = Join-Path $round2 'tgt\pgo-gen'
    $use = Join-Path $round2 'tgt\pgo-use'
    $saved = @{ CARGO_TARGET_DIR = $env:CARGO_TARGET_DIR; CARGO_ENCODED_RUSTFLAGS = $env:CARGO_ENCODED_RUSTFLAGS; MC_JARS_DIR = $env:MC_JARS_DIR }
    # One list for both builds, naming both build folders: a PGO profile matches functions by
    # name, so the instrumented and the optimised build must see exactly the same flags.
    $remap = @(Get-RemapPathFlags -Workspace $rs -TargetDirs @($gen, $use))
    Push-Location -LiteralPath $rs
    try {
        Write-Host "[1/6] instrumented build $(Get-Date -Format T)"
        # CARGO_ENCODED_RUSTFLAGS, not RUSTFLAGS: RUSTFLAGS splits a path that contains a space.
        $env:CARGO_TARGET_DIR = $gen
        $env:CARGO_ENCODED_RUSTFLAGS = (@("-Cprofile-generate=$profraw") + $remap) -join $unitSeparator
        Invoke-Native 'instrumented build' { cargo build --profile dist -p mc-client --bin mc-client --example skywars_bench }

        Write-Host "[2/6] training $(Get-Date -Format T)"
        # Build scripts and proc-macros also wrote profiles during step 1; they are not the client.
        New-Item -ItemType Directory -Force $profraw | Out-Null
        Get-ChildItem -LiteralPath $profraw -Filter '*.profraw' | Remove-Item -Force
        $trainOut = Join-Path $round2 ('results\pgo-train-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
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
        $env:CARGO_ENCODED_RUSTFLAGS = (@("-Cprofile-use=$profileData") + $remap) -join $unitSeparator
        Invoke-Native 'optimised build' { cargo build --profile dist -p mc-client --bin mc-client --example skywars_bench }
    } finally {
        Pop-Location
        Restore-Env $saved
    }
    $newExe = Join-Path $use 'dist\mc-client.exe'
    $builtBy = 'install-mc-rs.ps1 (PGO build of the current code)'
}

Write-Host "[5/6] smoke test $(Get-Date -Format T)"
$smoke = Join-Path ([IO.Path]::GetTempPath()) ('mc-rs-install-smoke-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $smoke | Out-Null
Set-Content -LiteralPath (Join-Path $smoke 'options.txt') -Encoding ASCII -Value @(
    'renderDistance:8', 'pauseOnLostFocus:false', 'soundCategory_master:0.0', 'maxFps:260', 'enableVsync:false')
$savedSmoke = @{ MC_PIPELINED_SUBMIT = $env:MC_PIPELINED_SUBMIT; MC_BACKGROUND_WINDOW = $env:MC_BACKGROUND_WINDOW; MC_NO_AUDIO = $env:MC_NO_AUDIO; MC_JARS_DIR = $env:MC_JARS_DIR }
try {
    $env:MC_PIPELINED_SUBMIT = '1'; $env:MC_BACKGROUND_WINDOW = '1'; $env:MC_NO_AUDIO = '1'
    Set-DefaultJarsDir
    $capture = Join-Path $rs 'crates\mc-parity\captures\spigot-1.8.8-perfworld-vd16.mccap'
    $proc = Start-Process -FilePath $newExe -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $smoke `
        -RedirectStandardOutput (Join-Path $smoke 'stdout.log') -RedirectStandardError (Join-Path $smoke 'stderr.log') `
        -ArgumentList @('--replay', "`"$capture`"", '--replayCamera', 'orbit', '--width', '1280', '--height', '720',
            '--offline', 'BenchBot', '--frames', '600', '--gameDir', "`"$smoke`"")
    if ($proc.ExitCode -ne 0) { throw "smoke test failed (exit code $($proc.ExitCode)); logs in $smoke" }
} finally {
    Restore-Env $savedSmoke
}
Remove-Item -LiteralPath $smoke -Recurse -Force

Write-Host "[6/6] install $(Get-Date -Format T)"
if (Test-Path -LiteralPath $installed) {
    $backup = Join-Path $rs ('optimization\installed-backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    New-Item -ItemType Directory -Force $backup | Out-Null
    Copy-Item -LiteralPath $installed -Destination $backup
    if (Test-Path -LiteralPath $stampPath) { Copy-Item -LiteralPath $stampPath -Destination $backup }
    Write-Host "previous client backed up to $backup"
}
New-Item -ItemType Directory -Force $release | Out-Null
Copy-Item -LiteralPath $newExe -Destination $installed -Force
[pscustomobject]@{
    installed_at        = (Get-Date).ToString('o')
    installed_by        = $builtBy
    pgo                 = $true
    exe_sha256          = (Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash
    profile_sha256      = (Get-FileHash -LiteralPath $profileData -Algorithm SHA256).Hash
    source_newest_ticks = $source.Ticks
    source_newest_file  = $source.File
    source_files        = $source.Count
    note                = 'Written by install-mc-rs.ps1; launch-mc-rs.ps1 compares the exe hash and the newest source file against it.'
} | ConvertTo-Json | Set-Content -LiteralPath $stampPath -Encoding UTF8
Write-Host "Installed $installed (PGO). Launch with launch-mc-rs.ps1." -ForegroundColor Green
