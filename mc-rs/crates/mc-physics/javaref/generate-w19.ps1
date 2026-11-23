# Regenerate the combined-effect traces for `mc-physics`.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-physics/javaref/generate-w19.ps1 [-Out <dir>]
#
# Writes ../tests/fixtures/w19_*.bin (ComboRefGen -> tests/parity.rs), or into -Out. ComboRefGen
# reuses PhysicsRefGen's harness. Same prerequisites as generate.ps1: a JDK (Temurin 21; it must
# be the first `java`/`javac` on PATH) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft.
#
# Java runs with its working directory in %TEMP%: log4j creates a `logs/` folder in the working
# directory, and one inside `crates/` breaks every cargo command.

param(
    [string]$Out = ''
)

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

$work = Join-Path $env:TEMP 'mc-physics-javaref-w19'
$classes = Join-Path $work 'build'
$fixtures = if ($Out) { $Out } else { Join-Path (Split-Path -Parent $here) 'tests\fixtures' }
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes `
    (Join-Path $here 'PhysicsRefGen.java') (Join-Path $here 'ComboRefGen.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Push-Location $work
try {
    Write-Host '== combined-effect traces =='
    & java -cp "$classes;$cp" ComboRefGen $fixtures
    if ($LASTEXITCODE -ne 0) { throw 'ComboRefGen failed' }
}
finally {
    Pop-Location
}

Write-Host 'ok'
