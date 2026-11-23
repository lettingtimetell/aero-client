#requires -Version 5.1
<#
Checks the portable launcher (packaging\aero\launcher.ps1): what it trusts about the folder with
the Minecraft 1.8.9 files, and exactly what it passes to the game.

Players zip and send their "game" folder to each other. A folder path the launcher remembered
there could point it at files someone else chose, and a network path (\\server\share) makes
Windows sign in to that server. So the launcher remembers the folder per Windows user
(%LOCALAPPDATA%), and accepts only a local folder given by its full path, outside the Aero Client
folder, that holds the 1.8.9 files.

1. The launcher's function definitions alone (loaded through the PowerShell parser, so none of its
   top-level code runs): the folder rules, where the choice is remembered, and how arguments are
   quoted, checked with a stand-in game that prints the arguments it receives.
2. A copy of the launcher, run in its test mode (AERO_LAUNCHER_NOGUI=1) in a temporary
   "Aero Client" folder whose game is that stand-in, with APPDATA and LOCALAPPDATA pointing into
   the temporary folder: it starts the game with the stand-in game files and its "game" folder
   and nothing else (no account, no sign-in of any kind), remembers the folder, and asks when
   there are no game files.
Everything happens in a temporary folder; no real game files, accounts or tokens are read. The
stand-in game is compiled from a few lines of C# by Add-Type (Windows PowerShell's compiler).

Run:  powershell -NoProfile -ExecutionPolicy Bypass -File packaging\scripts\test-launcher-trust.ps1
Exit code 0 when every check passes.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$launcher = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\aero\launcher.ps1')).Path
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($launcher, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw "launcher.ps1 does not parse: $($errors[0].Message)" }
$functions = $ast.FindAll({
        param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
    }, $false)
foreach ($function in $functions) { . ([scriptblock]::Create($function.Extent.Text)) }

$work = Join-Path ([System.IO.Path]::GetTempPath()) ('aero-launcher-trust-' + [guid]::NewGuid())
# What the launcher's functions read: its folders, where it remembers things, its test mode.
$root = Join-Path $work 'Aero Client'
$game = Join-Path $root 'game'
$settingsPath = Join-Path $work 'LocalAppData\Aero Client\launcher-settings.json'
$noGui = $true
$title = 'Aero Client'
$failures = New-Object System.Collections.Generic.List[string]

function Assert-That([bool]$Condition, [string]$What) {
    if ($Condition) { Write-Host "ok   $What" } else { Write-Host "FAIL $What"; $failures.Add($What) }
}

function Test-SameList([object[]]$Got, [object[]]$Want) {
    if ($Got.Count -ne $Want.Count) { return $false }
    for ($i = 0; $i -lt $Want.Count; $i++) {
        if (-not ([string]$Got[$i] -ceq [string]$Want[$i])) { return $false }
    }
    return $true
}

# A stand-in game folder: the two files the launcher looks for, with stand-in content.
function New-GameFiles([string]$Dir, [switch]$Libraries, [switch]$NoJar, [switch]$NoIndex) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    if (-not $NoIndex) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Dir 'assets\indexes') | Out-Null
        Set-Content -LiteralPath (Join-Path $Dir 'assets\indexes\1.8.json') -Value '{"objects":{}}'
    }
    if (-not $NoJar) {
        $jar = if ($Libraries) {
            Join-Path $Dir 'libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar'
        } else {
            Join-Path $Dir 'versions\1.8.9\1.8.9.jar'
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jar) | Out-Null
        Set-Content -LiteralPath $jar -Value 'a stand-in, not a jar'
    }
}

# The arguments a stand-in game printed, one "ARG<...>" line each.
function Read-PrintedArgs([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return @() }
    return @(Get-Content -LiteralPath $Path | Where-Object { $_ -like 'ARG<*>' } |
        ForEach-Object { $_.Substring(4, $_.Length - 5) })
}

try {
    # ---------------------------------------------------------------- 1. the functions alone
    New-Item -ItemType Directory -Force -Path $game | Out-Null
    # Game files a shared "game" folder could bring along.
    $planted = Join-Path $game 'evil'
    New-GameFiles $planted
    # Real-looking game folders elsewhere on the local disk.
    $official = Join-Path $work 'Roaming\.minecraft'
    New-GameFiles $official
    $libraries = Join-Path $work 'Games\Minecraft'
    New-GameFiles $libraries -Libraries
    $noJar = Join-Path $work 'NoJar'
    New-GameFiles $noJar -NoJar
    $noIndex = Join-Path $work 'NoIndex'
    New-GameFiles $noIndex -NoIndex

    Assert-That (Test-GameFilesDir $official) 'a local folder with the 1.8.9 jar under versions\ is accepted'
    Assert-That (Test-GameFilesDir $libraries) 'and one with it under libraries\'
    Assert-That (-not (Test-GameFilesDir $noJar)) 'a folder without the 1.8.9 jar is not'
    Assert-That (-not (Test-GameFilesDir $noIndex)) 'nor one without the 1.8 asset index'
    Assert-That (-not (Test-GameFilesDir (Join-Path $work 'missing'))) 'nor a missing folder'
    Assert-That (-not (Test-GameFilesDir $planted)) 'nor a folder inside the Aero Client folder'
    Assert-That (-not (Test-GameFilesDir (Join-Path $root 'game\..\game\evil'))) 'nor the same through ..'
    Assert-That (-not (Test-GameFilesDir 'game\evil')) 'nor a relative path'
    Assert-That (-not (Test-GameFilesDir '\\localhost\aero-no-such-share\games')) 'nor a network (UNC) path'
    Assert-That (-not (Test-GameFilesDir '\\?\C:\Windows')) 'nor a device path'
    Assert-That (-not (Test-GameFilesDir '')) 'nor an empty path'

    $files = Find-VanillaFiles $official
    Assert-That ($files.Jar -eq (Join-Path $official 'versions\1.8.9\1.8.9.jar') -and
        $files.Assets -eq (Join-Path $official 'assets')) 'the jar and the assets folder are found'
    Assert-That ((Find-VanillaFiles $libraries).Jar -eq
        (Join-Path $libraries 'libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar')) 'under libraries\ too'

    # Remembering: per user, never in the game folder; anything else in the file is kept.
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $settingsPath) | Out-Null
    Set-Content -LiteralPath $settingsPath -Value '{"keep":"me"}'
    Save-GameFilesDir $official
    Assert-That (Test-Path -LiteralPath $settingsPath -PathType Leaf) 'the choice is saved under LocalAppData'
    Assert-That (-not (Test-Path -LiteralPath (Join-Path $game 'launcher-settings.json'))) 'and not in the game folder'
    Assert-That ([string](Read-Settings).gameFilesDir -eq $official) 'and read back from there'
    Assert-That ([string](Read-Settings).keep -eq 'me') 'and the rest of the file is kept'

    # A settings file that came in with a shared game folder is never read.
    Set-Content -LiteralPath (Join-Path $game 'launcher-settings.json') `
        -Value (@{ gameFilesDir = $planted } | ConvertTo-Json)
    Assert-That ([string](Read-Settings).gameFilesDir -eq $official) 'a game\launcher-settings.json is ignored'

    # Quoting: the game receives exactly the arguments the launcher meant, one for one, even with
    # spaces, quotes and backslashes (the last value tries to smuggle in a second --gameDir).
    $printer = Join-Path $work 'printer\argprinter.exe'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $printer) | Out-Null
    Add-Type -OutputAssembly $printer -OutputType ConsoleApplication -TypeDefinition @'
public static class ArgPrinter {
    public static int Main(string[] args) {
        foreach (string a in args) { System.Console.WriteLine("ARG<" + a + ">"); }
        System.Console.WriteLine("ENV<MC_PIPELINED_SUBMIT=" +
            System.Environment.GetEnvironmentVariable("MC_PIPELINED_SUBMIT") + ">");
        return 0;
    }
}
'@
    $printed = Join-Path $work 'printer\args.txt'
    $passThrough = {
        param([string[]]$Values)
        try {
            Remove-Item -LiteralPath $printed -ErrorAction SilentlyContinue
            $proc = Start-Process -FilePath $printer -NoNewWindow -PassThru -Wait `
                -ArgumentList @($Values | ForEach-Object { ConvertTo-Argument $_ }) -RedirectStandardOutput $printed
            if ($proc.ExitCode -ne 0) { return @("the stand-in game exited with $($proc.ExitCode)") }
            return Read-PrintedArgs $printed
        } catch {
            return @("not started: $($_.Exception.Message)")
        }
    }
    $values = @('plain', 'two words', 'C:\Folder With Space\', 'ends\\', 'a "quoted" word',
        'back\"slash', '')
    Assert-That (Test-SameList (& $passThrough $values) $values) 'every argument arrives whole and unchanged'
    $smuggler = 'x\" --gameDir \\example-host\share --userType \"'
    Assert-That (Test-SameList (& $passThrough @('--example', $smuggler)) @('--example', $smuggler)) `
        'a value cannot smuggle in arguments of its own'

    # The launcher starts the game without an account: nothing in its code (comments aside) reads
    # an account file or passes a session.
    $code = ($tokens | Where-Object { $_.Kind -ne [System.Management.Automation.Language.TokenKind]::Comment } |
            ForEach-Object { $_.Text }) -join "`n"
    Assert-That ($code -notmatch '(?i)accessToken|--uuid|--username|--userType|accounts\.json|\btoken\b') `
        'launcher.ps1 reads no account and passes no session'
    $source = Get-Content -LiteralPath $launcher -Raw
    Assert-That ($source -notmatch "Join-Path \`$game 'launcher-settings.json'") 'launcher.ps1 keeps no settings in game'
    # Every program it starts is named by a path, never by a bare name that Windows would also
    # look for in the current folder.
    $bare = @($ast.FindAll({
                param($node) $node -is [System.Management.Automation.Language.CommandAst] -and
                $node.GetCommandName() -eq 'Start-Process'
            }, $true) | ForEach-Object {
            $elements = $_.CommandElements
            for ($i = 1; $i -lt $elements.Count - 1; $i++) {
                if ($elements[$i] -is [System.Management.Automation.Language.CommandParameterAst] -and
                    $elements[$i].ParameterName -eq 'FilePath' -and
                    $elements[$i + 1] -is [System.Management.Automation.Language.StringConstantExpressionAst] -and
                    $elements[$i + 1].Value -notmatch '[\\/]') { $elements[$i + 1].Value }
            }
        })
    Assert-That ($bare.Count -eq 0) "launcher.ps1 starts programs by path only (bare: $($bare -join ', '))"
    # Play Aero Client.cmd, which starts the launcher, names Windows PowerShell by full path too.
    $cmdFile = Join-Path $PSScriptRoot '..\aero\Play Aero Client.cmd'
    $cmdText = if (Test-Path -LiteralPath $cmdFile -PathType Leaf) { Get-Content -LiteralPath $cmdFile -Raw } else { '' }
    Assert-That ($cmdText.Contains('start "" "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" ') -and
        $cmdText -notmatch 'start "" powershell\.exe') 'Play Aero Client.cmd starts Windows PowerShell by full path'

    # ---------------------------------------------------------------- 2. the whole launcher
    $pkg = Join-Path $work 'package\Aero Client'
    $pkgApp = Join-Path $pkg 'app'
    New-Item -ItemType Directory -Force -Path $pkgApp, (Join-Path $pkg 'game\resourcepacks') | Out-Null
    Copy-Item -LiteralPath $launcher -Destination (Join-Path $pkgApp 'launcher.ps1')
    Copy-Item -LiteralPath $printer -Destination (Join-Path $pkgApp 'mc-client.exe')
    $roaming = Join-Path $work 'user\Roaming'
    $local = Join-Path $work 'user\Local'
    $gameFiles = Join-Path $roaming '.minecraft'
    New-GameFiles $gameFiles
    New-Item -ItemType Directory -Force -Path $local | Out-Null
    $log = Join-Path $pkg 'game\logs\latest-launch.log'
    $remembered = Join-Path $local 'Aero Client\launcher-settings.json'

    # Files with Windows' "downloaded from the internet" mark: the package's own, and one a player
    # put in the game folder.
    $ownFile = Join-Path $pkg 'README.txt'
    $playerFile = Join-Path $pkg 'game\resourcepacks\pack.zip'
    foreach ($file in $ownFile, $playerFile, (Join-Path $pkgApp 'NOTICE.txt')) {
        Set-Content -LiteralPath $file -Value 'stand-in'
        Set-Content -LiteralPath $file -Stream 'Zone.Identifier' -Value "[ZoneTransfer]`r`nZoneId=3"
    }

    function Invoke-Launcher {
        # Test mode, 5 frames, and this temporary folder as the per-user folders; restored after.
        $names = 'AERO_LAUNCHER_NOGUI', 'AERO_LAUNCHER_FRAMES', 'APPDATA', 'LOCALAPPDATA', 'MC_PIPELINED_SUBMIT'
        $saved = @{}
        foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
        try {
            [Environment]::SetEnvironmentVariable('AERO_LAUNCHER_NOGUI', '1')
            [Environment]::SetEnvironmentVariable('AERO_LAUNCHER_FRAMES', '5')
            [Environment]::SetEnvironmentVariable('APPDATA', $roaming)
            [Environment]::SetEnvironmentVariable('LOCALAPPDATA', $local)
            [Environment]::SetEnvironmentVariable('MC_PIPELINED_SUBMIT', $null)
            $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $pkgApp 'launcher.ps1')
            return [pscustomobject]@{ Exit = $LASTEXITCODE; Output = ($output | Out-String) }
        } finally {
            foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
        }
    }

    # a. First start: the official launcher's folder has the files.
    $run = Invoke-Launcher
    $got = Read-PrintedArgs $log
    $want = @('--gameDir', (Join-Path $pkg 'game'), '--assetsDir', (Join-Path $gameFiles 'assets'),
        '--assetIndex', '1.8', '--versionJar', (Join-Path $gameFiles 'versions\1.8.9\1.8.9.jar'),
        '--frames', '5')
    Assert-That ($run.Exit -eq 0) "the launcher starts the game and returns 0 (got $($run.Exit))"
    Assert-That (Test-SameList $got $want) "the game gets its folder, the game files and the frame count, nothing else: $($got -join ' | ')"
    Assert-That ((Get-Content -LiteralPath $log) -contains 'ENV<MC_PIPELINED_SUBMIT=1>') 'with pipelined submit on'
    Assert-That ((Test-Path -LiteralPath $remembered -PathType Leaf) -and
        ([string](Get-Content -LiteralPath $remembered -Raw | ConvertFrom-Json).gameFilesDir -eq $gameFiles)) `
        'the game files folder is remembered under LocalAppData'
    Assert-That (-not (Test-Path -LiteralPath (Join-Path $pkg 'game\launcher-settings.json'))) 'and nothing in the game folder'
    $marked = { param($Path) $null -ne (Get-Item -LiteralPath $Path -Stream 'Zone.Identifier' -ErrorAction SilentlyContinue) }
    Assert-That ((& $marked $ownFile) -and (& $marked (Join-Path $pkgApp 'NOTICE.txt'))) `
        "the launcher leaves the package's own download marks alone"
    Assert-That (& $marked $playerFile) 'and the marks of files a player put in the game folder'

    # b. The folder chosen last time wins while it still has the files.
    $other = Join-Path $work 'user\Games\Minecraft'
    New-GameFiles $other -Libraries
    Set-Content -LiteralPath $remembered -Value (@{ gameFilesDir = $other } | ConvertTo-Json)
    Remove-Item -LiteralPath $log
    $run = Invoke-Launcher
    $got = Read-PrintedArgs $log
    Assert-That ($run.Exit -eq 0 -and $got.Count -ge 8 -and
        $got[7] -eq (Join-Path $other 'libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar')) `
        'the remembered folder is used'

    # c. No game files anywhere: the launcher explains, asks for the folder and starts nothing (in
    #    test mode the question is answered with Cancel).
    Remove-Item -LiteralPath $remembered, $gameFiles, $other -Recurse -Force
    Remove-Item -LiteralPath $log
    $run = Invoke-Launcher
    Assert-That ($run.Exit -eq 1 -and -not (Test-Path -LiteralPath $log)) 'without game files the game is not started'
    Assert-That ($run.Output -match "couldn't find them") 'and the player is told what to do'
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host "$($failures.Count) check(s) failed"
    exit 1
}
Write-Host 'all checks passed'
exit 0
