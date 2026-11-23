# Regenerate the interaction differential fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-interaction-diff.ps1 [-Work <dir>] [-Filter <name>] [-Out <path>]
#
# The same harness as generate-movement-diff.ps1 (the real `Minecraft.runTick()` headless, LWJGL
# stubbed by javaref/h1stubs over an unsealed copy of the LWJGL jar): InteractionDiffRef.java only
# adds scenarios and reuses MovementDiffRef's harness classes, so both files compile together.
# Writes crates/mc-client/tests/fixtures/h2_interaction/interaction_diff.bin, which
# `src/app/h2_interaction_oracle.rs` replays through the Rust client.
#
# Requires a JDK and the recompiled vanilla classes in MCP-919-main/bin/minecraft. Java runs from
# the work directory, never from inside `crates/` (log4j creates a `logs` folder in the working
# directory, and `crates/logs` breaks every cargo command). Set H1_DEBUG=1 to print every frame's
# packets and player state.

param(
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-h2-interaction-diff'),
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
if (Test-Path $classes) { Remove-Item -Recurse -Force $classes }
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

Write-Host '== javac MovementDiffRef + InteractionDiffRef =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'MovementDiffRef.java') (Join-Path $here 'InteractionDiffRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

$out = if ($Out) { $Out } else { Join-Path (Split-Path -Parent $here) 'tests\fixtures\h2_interaction\interaction_diff.bin' }
# Run the `java` of the JDK whose `javac` compiled the classes: a different `java` earlier on PATH
# (an old runtime) cannot load them.
$java = Join-Path (Split-Path -Parent (Get-Command javac).Source) 'java.exe'
if (-not (Test-Path $java)) { $java = 'java' }
Write-Host "== InteractionDiffRef -> $out ($java) =="
Push-Location $Work
try {
    if ($Filter) {
        & $java -cp "$classes;$cp" InteractionDiffRef $out $Filter
    } else {
        & $java -cp "$classes;$cp" InteractionDiffRef $out
    }
    if ($LASTEXITCODE -ne 0) { throw 'InteractionDiffRef failed' }
} finally {
    Pop-Location
}
Write-Host 'ok'
