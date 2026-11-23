# Regenerate the end-to-end movement differential fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-movement-diff.ps1 [-Work <dir>] [-Filter <name>]
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. Writes crates/mc-client/tests/fixtures/h1_movement/movement_diff.bin,
# which `src/app/h1_movement_oracle.rs` replays through the Rust client.
#
# The oracle runs the REAL `Minecraft.runTick()` headless. LWJGL's Display/Mouse/Keyboard/Sys are
# replaced by the stubs in javaref/h1stubs. The real LWJGL jar is `Sealed: true`, so it cannot
# share its packages with a directory of stubs: this script extracts it into the work directory
# (a directory is never sealed) and compiles the four stubs over the originals.
#
# Java runs from the work directory, never from inside `crates/`: log4j creates a `logs` folder in
# the working directory, and `crates/logs` breaks every cargo command.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and
# the vanilla jar LAST so it can never shadow the deobfuscated classes.

param(
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-h1-movement-diff'),
    # Only the scenarios whose name contains this text (use with -Out: a filtered run is not
    # the committed fixture). Set H1_DEBUG=1 to print every frame's packets and player state.
    [string]$Filter = '',
    [string]$Out = ''
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\entity\Entity.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$lwjglJar = Get-ChildItem (Join-Path $mcp 'jars\libraries\org\lwjgl\lwjgl\lwjgl') -Recurse -Filter 'lwjgl-*.jar' |
    Select-Object -First 1
if (-not $lwjglJar) { throw 'LWJGL jar not found under jars\libraries\org\lwjgl\lwjgl\lwjgl' }

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar |
    Where-Object { $_.FullName -ne $lwjglJar.FullName } |
    ForEach-Object { $_.FullName }) -join ';'

New-Item -ItemType Directory -Force -Path $Work | Out-Null
$lwjgl = Join-Path $Work 'lwjgl-unsealed'
$classes = Join-Path $Work 'classes'
if (Test-Path $lwjgl) { Remove-Item -Recurse -Force $lwjgl }
New-Item -ItemType Directory -Force -Path $lwjgl, $classes | Out-Null

Write-Host "== unsealed LWJGL copy -> $lwjgl =="
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($lwjglJar.FullName)
try {
    foreach ($entry in $zip.Entries) {
        if ($entry.FullName.EndsWith('/') -or $entry.FullName.StartsWith('META-INF/')) { continue }
        $dest = Join-Path $lwjgl ($entry.FullName -replace '/', '\')
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
    }
} finally {
    $zip.Dispose()
}

Write-Host '== javac stubs (over the extracted originals) =='
$stubs = Get-ChildItem (Join-Path $here 'h1stubs') -Recurse -Filter *.java | ForEach-Object { $_.FullName }
& javac -nowarn -encoding UTF-8 -d $lwjgl @stubs
if ($LASTEXITCODE -ne 0) { throw 'javac (stubs) failed' }

$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $lwjgl,
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

Write-Host '== javac MovementDiffRef =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'MovementDiffRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

$out = if ($Out) { $Out } else { Join-Path (Split-Path -Parent $here) 'tests\fixtures\h1_movement\movement_diff.bin' }
Write-Host "== MovementDiffRef -> $out =="
Push-Location $Work
try {
    if ($Filter) {
        & java -cp "$classes;$cp" MovementDiffRef $out $Filter
    } else {
        & java -cp "$classes;$cp" MovementDiffRef $out
    }
    if ($LASTEXITCODE -ne 0) { throw 'MovementDiffRef failed' }
} finally {
    Pop-Location
}
Write-Host 'ok'
