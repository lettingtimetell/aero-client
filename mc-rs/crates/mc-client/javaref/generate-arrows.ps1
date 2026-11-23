# Regenerate the vanilla arrow render reference.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-arrows.ps1 [-OutDir <dir>] [-PackDir <dir>]
#
# Requires a JDK (developed against Temurin 21), the recompiled vanilla classes in
# MCP-919-main/bin/minecraft, the 1.8.9 natives and a GL driver (LWJGL 2 offscreen Pbuffer,
# 1920x1080; no window is opened).
#
# ArrowRenderRef draws arrows through the real RenderManager.renderEntityStatic (RenderArrow,
# Render.renderEntityOnFire, RenderPlayer + LayerArrow) under the world pass's GL state, and
# prints the REFS tables of crates/mc-render/tests/arrow_java_parity.rs to the console (splice
# them in place of the existing tables). It also writes one PNG per case and cases.txt (every
# captured draw with its GL state) into -OutDir.
#
# -PackDir runs the arrow cases a second time with that folder resource pack above the vanilla
# jar (case names prefixed `pack_`). The pack is only read.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes (it still supplies the assets).

param(
    [string]$OutDir = (Join-Path $env:TEMP 'mc-rs-arrow-javaref-png'),
    [string]$PackDir = ''
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\entity\RenderArrow.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$jdk = 'C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot\bin'
$javac = if (Test-Path (Join-Path $jdk 'javac.exe')) { Join-Path $jdk 'javac.exe' } else { 'javac' }
$java = if (Test-Path (Join-Path $jdk 'java.exe')) { Join-Path $jdk 'java.exe' } else { 'java' }

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'

$classes = Join-Path $env:TEMP 'mc-rs-arrow-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host '== javac =='
& $javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'ArrowRenderRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host "== ArrowRenderRef, vanilla (arrow_java_parity.rs REFS; PNGs in $OutDir) =="
# log4j prints Minecraft's own INFO lines ("[hh:mm:ss] [main/INFO]: ...") to stdout as well;
# drop them so only the Rust tables are printed.
& $java "-Djava.library.path=$natives" -cp "$classes;$cp" ArrowRenderRef $OutDir |
    Where-Object { $_ -notmatch '^\[\d\d:\d\d:\d\d\] \[' }
if ($LASTEXITCODE -ne 0) { throw 'ArrowRenderRef failed' }

if ($PackDir -ne '') {
    Write-Host "== ArrowRenderRef, pack $PackDir =="
    & $java "-Djava.library.path=$natives" -cp "$classes;$cp" ArrowRenderRef $OutDir -pack $PackDir -prefix 'pack_' |
        Where-Object { $_ -notmatch '^\[\d\d:\d\d:\d\d\] \[' }
    if ($LASTEXITCODE -ne 0) { throw 'ArrowRenderRef (pack) failed' }
}
