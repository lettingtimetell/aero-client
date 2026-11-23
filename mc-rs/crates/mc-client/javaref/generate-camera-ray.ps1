# Regenerate the F5 camera collision reference.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-camera-ray.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. Runs the real 1.8.9 `World.rayTraceBlocks` through a verbatim copy
# of `EntityRenderer.orientCamera`'s third-person block and writes
# crates/mc-client/javaref/camera_ray_ref.txt, which `src/app/mods/w5_grass.rs`'s tests read.
# Headless: no LWJGL, no window.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and
# the vanilla jar LAST so it can never shadow the deobfuscated classes.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\world\World.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $env:TEMP 'mc-rs-camera-ray-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'CameraRayRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

$out = Join-Path $here 'camera_ray_ref.txt'
Write-Host "== CameraRayRef -> $out =="
& java -cp "$classes;$cp" CameraRayRef $out
if ($LASTEXITCODE -ne 0) { throw 'CameraRayRef failed' }
