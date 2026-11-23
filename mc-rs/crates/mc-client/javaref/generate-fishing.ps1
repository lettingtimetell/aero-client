# Regenerate the fishing-hook Java reference tables.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-fishing.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. Prints two Rust tables to the console:
#
# * FishHookRef   -> crates/mc-client/src/sync/entities/fishing_hook_parity.rs (`REFS`):
#                    real EntityFishHook + EntityTrackerEntry, server and client side.
# * RenderFishRef -> crates/mc-render/tests/fishing_line_parity.rs (`REFS`): the real
#                    RenderFish.doRender in an LWJGL 2 offscreen Pbuffer (needs the 1.8.9
#                    natives and a GL driver; no window is opened).
#
# Classpath order matches the physics generator: recompiled classes first, libraries next, and
# the vanilla jar LAST so it can never shadow the deobfuscated classes.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\entity\Entity.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'

$classes = Join-Path $env:TEMP 'mc-rs-fishing-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'FishHookRef.java') (Join-Path $here 'RenderFishRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== FishHookRef (fishing_hook_parity.rs REFS) =='
& java -cp "$classes;$cp" FishHookRef
if ($LASTEXITCODE -ne 0) { throw 'FishHookRef failed' }

Write-Host '== RenderFishRef (fishing_line_parity.rs REFS) =='
& java "-Djava.library.path=$natives" -cp "$classes;$cp" RenderFishRef
if ($LASTEXITCODE -ne 0) { throw 'RenderFishRef failed' }
