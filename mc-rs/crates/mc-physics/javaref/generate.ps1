# S13 - regenerate the Java physics-parity fixtures for `mc-physics`.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-physics/javaref/generate.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft (the same prerequisite S01's extractor has). Writes
# ../tests/fixtures/*.bin - one per scenario, raw big-endian IEEE-754 bit patterns.
#
# Classpath order matches S01's run.ps1: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes.

$ErrorActionPreference = 'Stop'

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

$classes = Join-Path $here 'build'
$fixtures = Join-Path (Split-Path -Parent $here) 'tests\fixtures'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'PhysicsRefGen.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== generate =='
& java -cp "$classes;$cp" PhysicsRefGen $fixtures
if ($LASTEXITCODE -ne 0) { throw 'PhysicsRefGen failed' }

Write-Host 'ok'
