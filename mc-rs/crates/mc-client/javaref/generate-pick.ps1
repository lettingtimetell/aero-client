# Regenerate the interpolation and click-pick reference.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-pick.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. Runs real 1.8.9 `EntityOtherPlayerMP` interpolation and a verbatim
# `EntityRenderer.getMouseOver(1.0F)` over the real classes, and writes
# crates/mc-client/javaref/pick_ref.txt, which `tests/l7_pick_oracle.rs` reads. Headless: no LWJGL,
# no window.
#
# Java runs from a scratch directory, never from inside `crates/`: log4j creates a `logs` folder in
# the working directory, and `crates/logs` breaks every cargo command.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and
# the vanilla jar LAST so it can never shadow the deobfuscated classes.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\entity\EntityOtherPlayerMP.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $env:TEMP 'mc-rs-pick-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'PickRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

$out = Join-Path $here 'pick_ref.txt'
Write-Host "== PickRef -> $out =="
Push-Location $classes
try {
    & java -cp "$classes;$cp" PickRef $out
    if ($LASTEXITCODE -ne 0) { throw 'PickRef failed' }
} finally {
    Pop-Location
}
