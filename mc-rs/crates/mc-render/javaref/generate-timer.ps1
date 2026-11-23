# Regenerate tests/fixtures/timer_ref.bin, the reference trace of
# vanilla's real `net/minecraft/util/Timer.class` that `tests/timer_java_parity.rs` replays.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-render/javaref/generate-timer.ps1
#
# Requires a JDK (Temurin 21) and the recompiled vanilla classes in MCP-919-main/bin/minecraft.
# Java runs from a temporary working directory, never from inside crates/ (a stray `logs`
# folder there breaks cargo).

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$classesDir = Join-Path $mcp 'bin\minecraft'
if (-not (Test-Path (Join-Path $classesDir 'net\minecraft\util\Timer.class'))) {
    throw "Missing recompiled vanilla classes at $classesDir. Run recompile.bat in MCP-919-main first."
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) 'mc-rs-timer-ref'
$build = Join-Path $work 'build'
New-Item -ItemType Directory -Force -Path $build | Out-Null
$fixture = Join-Path (Split-Path -Parent $here) 'tests\fixtures\timer_ref.bin'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $fixture) | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -d $build (Join-Path $here 'TimerRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== generate =='
Push-Location $work
try {
    & java -cp "$build;$classesDir" TimerRef $fixture
    if ($LASTEXITCODE -ne 0) { throw 'TimerRef failed' }
} finally {
    Pop-Location
}
Write-Host 'ok'
