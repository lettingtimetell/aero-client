# Regenerate the vignette Java reference.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-vignette.ps1
#
# Requires a JDK (developed against Temurin 21), the recompiled vanilla classes in
# MCP-919-main/bin/minecraft, the 1.8.9 natives and a GL driver (LWJGL 2 offscreen Pbuffer,
# 1920x1080; no window is opened).
#
# VignetteRef calls the real GuiIngame.renderVignette over a cleared framebuffer and prints the
# tables of crates/mc-render/tests/x7_vignette_java_parity.rs to the console (splice them in place
# of the existing tables).
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes (it still supplies the assets).

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\gui\GuiIngame.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'

$classes = Join-Path $env:TEMP 'mc-rs-vignette-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'VignetteRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== VignetteRef (x7_vignette_java_parity.rs tables) =='
& java "-Djava.library.path=$natives" -cp "$classes;$cp" VignetteRef |
    Where-Object { $_ -notmatch '^\[\d\d:\d\d:\d\d\] \[' }
if ($LASTEXITCODE -ne 0) { throw 'VignetteRef failed' }
