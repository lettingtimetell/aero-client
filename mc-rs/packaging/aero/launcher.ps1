#requires -Version 5.1
<#
Aero Client launcher (portable package). Started hidden by "Play Aero Client.cmd".

What it does, in order:
  1. Finds the Minecraft 1.8.9 game files on this PC: the folder saved last time, then the
     official launcher's folder (%APPDATA%\.minecraft). If neither has them it asks once with a
     folder picker and remembers the answer for this Windows user
     (%LOCALAPPDATA%\Aero Client\launcher-settings.json, never inside the shareable "game" folder).
     Nothing from Minecraft is shipped in this package.
  2. Starts the game with its settings in the "game" folder, and shows a plain message if it fails.
It reads no account and passes no sign-in to the game. Players sign in inside the game, on its
Accounts screen, with the Microsoft account that owns Minecraft; the game keeps that sign-in for
their Windows user, outside this folder.
Every problem is reported with a pop-up that says exactly what to do next.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()

$root = Split-Path -Parent $PSScriptRoot          # the "Aero Client" folder
$app = $PSScriptRoot                               # ...\Aero Client\app
$game = Join-Path $root 'game'
$client = Join-Path $app 'mc-client.exe'
# The game files folder, remembered per Windows user and never in the "game" folder: players zip
# and send that folder, and a path saved there could point this script at files someone else
# chose. (A launcher-settings.json that older versions left in "game" is ignored.)
$settingsPath = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA 'Aero Client\launcher-settings.json'
} else {
    $null
}
$title = 'Aero Client'
# Test hooks (never set by "Play Aero Client.cmd"; packaging\scripts\test-launcher-trust.ps1 uses
# them): AERO_LAUNCHER_NOGUI=1 prints messages instead of showing pop-ups; AERO_LAUNCHER_FRAMES=N
# makes the game close itself after N frames.
$noGui = $env:AERO_LAUNCHER_NOGUI -eq '1'

function Show-Message([string]$Text, [string]$Icon = 'Information', [string]$Buttons = 'OK') {
    if ($noGui) {
        Write-Host "[$Icon] $Text"
        return [System.Windows.Forms.DialogResult]::Cancel
    }
    return [System.Windows.Forms.MessageBox]::Show($Text, $title,
        [System.Windows.Forms.MessageBoxButtons]::$Buttons,
        [System.Windows.Forms.MessageBoxIcon]::$Icon)
}

function Stop-WithMessage([string]$Text) {
    [void](Show-Message $Text 'Warning')
    exit 1
}

# Windows' "Start-Process -ArgumentList" joins arguments with spaces and does not quote them, so
# every argument that could contain a space or a quote (any path) is quoted here, the way the
# C runtime splits a command line: backslashes are literal except before a quote, so the ones
# before a quote (or before the closing quote) are doubled, and the quote itself is escaped.
function ConvertTo-Argument([string]$Value) {
    if ($Value -ne '' -and $Value -notmatch '[\s"]') { return $Value }
    $quoted = [regex]::Replace($Value, '(\\*)"', { param($m) ($m.Groups[1].Value * 2) + '\"' })
    $quoted = [regex]::Replace($quoted, '(\\+)$', { param($m) $m.Groups[1].Value * 2 })
    return '"' + $quoted + '"'
}

function Read-Settings {
    if ($settingsPath -and (Test-Path -LiteralPath $settingsPath -PathType Leaf)) {
        try { return Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json } catch { }
    }
    return $null
}

function Save-GameFilesDir([string]$Dir) {
    if (-not $settingsPath) { return }
    try {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $settingsPath) | Out-Null
        # Anything else saved in the file is kept.
        $data = Read-Settings
        if ($data -isnot [System.Management.Automation.PSCustomObject]) { $data = [pscustomobject]@{} }
        $data | Add-Member -NotePropertyName 'gameFilesDir' -NotePropertyValue $Dir -Force
        $data | ConvertTo-Json | Set-Content -LiteralPath $settingsPath -Encoding UTF8
    } catch {
        # Only a convenience: the folder is looked for again next time.
    }
}

# The Minecraft 1.8.9 files in a game folder, or $null: the assets folder that holds the 1.8
# index, and the 1.8.9 client jar, either where the official launcher keeps it
# (versions\1.8.9\1.8.9.jar) or under libraries\ (com\mojang\minecraft\1.8.9).
function Find-VanillaFiles([string]$Dir) {
    $assets = Join-Path $Dir 'assets'
    if (-not (Test-Path -LiteralPath (Join-Path $assets 'indexes\1.8.json') -PathType Leaf)) {
        return $null
    }
    foreach ($jar in @((Join-Path $Dir 'versions\1.8.9\1.8.9.jar'),
            (Join-Path $Dir 'libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar'))) {
        if (Test-Path -LiteralPath $jar -PathType Leaf) {
            return [pscustomobject]@{ Jar = $jar; Assets = $assets }
        }
    }
    return $null
}

# A game files folder: a folder on a local drive given by its full path (never a network path,
# which Windows would sign in to, nor a relative one), outside the Aero Client folder (which other
# players' files can come in with), that holds the Minecraft 1.8.9 files. The path is checked
# before anything on it is touched.
function Test-GameFilesDir([string]$Dir) {
    if (-not $Dir -or $Dir -notmatch '^[A-Za-z]:[\\/]') { return $false }
    try { $full = [System.IO.Path]::GetFullPath($Dir) } catch { return $false }
    if ($full -notmatch '^[A-Za-z]:\\') { return $false }
    $inside = [System.IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    if (($full.TrimEnd('\') + '\').StartsWith($inside, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    return $null -ne (Find-VanillaFiles $full)
}

function Find-GameFilesDir {
    $saved = Read-Settings
    if ($saved -and (Test-GameFilesDir ([string]$saved.gameFilesDir))) { return [string]$saved.gameFilesDir }

    # The official launcher keeps the game files here.
    if ($env:APPDATA) {
        $official = Join-Path $env:APPDATA '.minecraft'
        if (Test-GameFilesDir $official) { return $official }
    }

    $answer = Show-Message ("Aero Client uses your own Minecraft 1.8.9 game files, but couldn't find them.`n`n" +
        "If you haven't set them up yet: in the official Minecraft Launcher, add an installation " +
        "of version 1.8.9, press Play once and close the game when it opens. Then start Aero Client again.`n`n" +
        "If your game files are somewhere else, click OK and select their folder " +
        "(the one that contains the assets folder).") 'Question' 'OKCancel'
    if ($answer -ne [System.Windows.Forms.DialogResult]::OK) { exit 1 }
    $picker = New-Object System.Windows.Forms.FolderBrowserDialog
    $picker.Description = 'Select the folder with your Minecraft 1.8.9 game files (the one that contains the assets folder)'
    $picker.ShowNewFolderButton = $false
    if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 1 }
    if (-not (Test-GameFilesDir $picker.SelectedPath)) {
        Stop-WithMessage ("That folder doesn't have the Minecraft 1.8.9 game files.`n`n" +
            "Select the folder that contains the assets folder and the 1.8.9 game, or start version " +
            '1.8.9 once in your launcher so it downloads them, then start Aero Client again.')
    }
    return $picker.SelectedPath
}

try {
    # Files unzipped from the internet keep Windows' "downloaded" mark: the launcher changes no
    # file's mark, so Windows' own checks (README.txt explains the first-start questions) apply.

    if (-not [Environment]::Is64BitOperatingSystem) {
        Stop-WithMessage 'Aero Client needs 64-bit Windows 10 or 11.'
    }
    if (-not (Test-Path -LiteralPath $client -PathType Leaf)) {
        Stop-WithMessage ("The game file app\mc-client.exe is missing.`n`n" +
            'Your antivirus may have removed it. Extract the zip again, then start Aero Client.')
    }
    New-Item -ItemType Directory -Force -Path $game | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $game 'logs') | Out-Null

    # 1. Minecraft 1.8.9's own files
    $filesDir = Find-GameFilesDir
    Save-GameFilesDir $filesDir
    $files = Find-VanillaFiles ([System.IO.Path]::GetFullPath($filesDir))

    # 2. Start the game. It gets the game files and its folder, nothing else: the player signs in
    #    inside it (Accounts).
    $clientArgs = @(
        '--gameDir', $game,
        '--assetsDir', $files.Assets,
        '--assetIndex', '1.8',
        '--versionJar', $files.Jar
    )
    if ($env:AERO_LAUNCHER_FRAMES -match '^[0-9]+$') { $clientArgs += @('--frames', $env:AERO_LAUNCHER_FRAMES) }
    $clientArgs = @($clientArgs | ForEach-Object { ConvertTo-Argument $_ })
    # The game writes its log to standard output and crash messages to standard error.
    $outLog = Join-Path $game 'logs\latest-launch.log'
    $errLog = Join-Path $game 'logs\latest-launch.errors.log'
    # Frame encoding and presenting on a second thread, paced (the build's tuned default).
    $env:MC_PIPELINED_SUBMIT = '1'
    $proc = Start-Process -FilePath $client -ArgumentList $clientArgs -WorkingDirectory $game `
        -NoNewWindow -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    $null = $proc.Handle   # keep the handle so ExitCode is readable after the game closes
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
        $answer = Show-Message ("Aero Client closed because of an error (code $($proc.ExitCode)).`n`n" +
            "Logs were saved in the game\logs folder. If you ask for help, send latest-launch.log " +
            "and latest-launch.errors.log from there.`n`n" +
            'Open the logs folder now?') 'Error' 'YesNo'
        if ($answer -eq [System.Windows.Forms.DialogResult]::Yes) {
            # By full path: a bare name would also be looked for in the current folder.
            Start-Process -FilePath (Join-Path $env:SystemRoot 'explorer.exe') `
                -ArgumentList (ConvertTo-Argument (Join-Path $game 'logs')) | Out-Null
        }
        exit 1
    }
} catch {
    [void](Show-Message ("Aero Client couldn't start:`n`n" + $_.Exception.Message) 'Error')
    exit 1
}
