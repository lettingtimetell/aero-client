# Regenerate the "inside a block" (suffocation overlay) Java
# reference.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-suffocation.ps1 [-OutDir <dir>]
#
# Requires a JDK (developed against Temurin 21), the recompiled vanilla classes in
# MCP-919-main/bin/minecraft, the 1.8.9 natives and a GL driver (LWJGL 2 offscreen Pbuffer,
# 1920x1080; no window is opened).
#
# SuffocationRef calls the real EntityRenderer.renderHand (hand, then ItemRenderer.renderOverlays)
# with the eye inside a block, records the live GL state at the suffocation quad, and prints the
# tables of crates/mc-render/tests/suffocation_overlay_java_parity.rs to the console (splice them
# in place of the existing tables). It also writes one PNG per case into -OutDir.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes (it still supplies the assets).

param(
    [string]$OutDir = (Join-Path $env:TEMP 'mc-rs-suffocation-javaref-png')
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\ItemRenderer.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'

$classes = Join-Path $env:TEMP 'mc-rs-suffocation-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'SuffocationRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host "== SuffocationRef (suffocation_overlay_java_parity.rs tables; PNGs in $OutDir) =="
# log4j prints Minecraft's own INFO lines ("[hh:mm:ss] [main/INFO]: ...") to stdout as well;
# drop them so only the Rust table is printed.
& java "-Djava.library.path=$natives" -cp "$classes;$cp" SuffocationRef $OutDir |
    Where-Object { $_ -notmatch '^\[\d\d:\d\d:\d\d\] \[' }
if ($LASTEXITCODE -ne 0) { throw 'SuffocationRef failed' }
