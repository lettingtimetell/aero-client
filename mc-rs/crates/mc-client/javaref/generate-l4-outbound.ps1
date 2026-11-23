# Regenerate the outbound packet bytes and mouse-look trace.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-l4-outbound.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. Writes crates/mc-client/javaref/l4_outbound_ref.txt, which
# `src/outbound.rs`'s tests read. Headless: no LWJGL, no window, no world.
#
# Java runs from a temp directory, never from inside `crates/`: log4j creates a `logs` folder in
# the working directory, and `crates/logs` breaks every cargo command.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and
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

$work = Join-Path $env:TEMP 'mc-rs-l4-outbound-javaref'
$classes = Join-Path $work 'classes'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'L4OutboundRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

$out = Join-Path $here 'l4_outbound_ref.txt'
Write-Host "== L4OutboundRef -> $out =="
Push-Location $work
try {
    & java -cp "$classes;$cp" L4OutboundRef $out
    if ($LASTEXITCODE -ne 0) { throw 'L4OutboundRef failed' }
} finally {
    Pop-Location
}
