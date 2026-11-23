# Regenerate the vanilla ender dragon crystal beam reference.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-render/javaref/generate-dragon-beam.ps1 [-Out <dir>]
#
# Requires a JDK, the recompiled vanilla classes in MCP-919-main/bin/minecraft, the 1.8.9 natives
# and a GL driver (an LWJGL 2 offscreen Pbuffer; no window is opened). Prints the `REFS` table of
# crates/mc-render/tests/dragon_beam_java_parity.rs and writes <Out>/gl-<case>.png (the real
# vanilla frames, for MC_DRAGON_BEAM_JAVA_DIR) and <Out>/cases.txt. The PNGs sample a vanilla
# texture, so they stay outside the repository.
#
# Run it from a scratch folder, never from inside crates/: log4j may create a `logs` folder in
# the working directory, and crates/logs breaks cargo.

param(
    [string]$Out = (Join-Path $env:TEMP 'mc-rs-dragon-beam-javaref-out')
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$jar = Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\entity\RenderDragon.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

# Classpath order: recompiled classes first, libraries next, the vanilla jar LAST so it can never
# shadow the deobfuscated classes (it still supplies the assets DefaultResourcePack reads).
$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    $jar
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'

# The `java` next to `javac`, so the runtime always matches the compiler.
$javac = (Get-Command javac).Source
$java = Join-Path (Split-Path -Parent $javac) 'java.exe'

$classes = Join-Path $env:TEMP 'mc-rs-dragon-beam-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Write-Host '== javac =='
& $javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'DragonBeamRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== DragonBeamRef (dragon_beam_java_parity.rs REFS) =='
& $java "-Djava.library.path=$natives" -cp "$classes;$cp" DragonBeamRef $Out
if ($LASTEXITCODE -ne 0) { throw 'DragonBeamRef failed' }
