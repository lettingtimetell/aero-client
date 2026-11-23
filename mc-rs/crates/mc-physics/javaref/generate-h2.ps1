# Regenerate the riding/flight/potion/combined-block traces for `mc-physics`.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-physics/javaref/generate-h2.ps1
#
# Writes ../tests/fixtures/h2_*.bin (H2EdgeRefGen -> tests/parity.rs). H2EdgeRefGen reuses
# PhysicsRefGen's harness, including its riding event kinds 14 (mount) and 15 (dismount). Same
# prerequisites as generate.ps1: a JDK and the recompiled vanilla classes in MCP-919-main/bin/minecraft.
#
# Java runs with its working directory in %TEMP%: log4j creates a `logs/` folder in the working
# directory, and one inside `crates/` breaks every cargo command.

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

$work = Join-Path $env:TEMP 'mc-physics-javaref-h2'
$classes = Join-Path $work 'build'
$fixtures = Join-Path (Split-Path -Parent $here) 'tests\fixtures'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes `
    (Join-Path $here 'PhysicsRefGen.java') (Join-Path $here 'H2EdgeRefGen.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Push-Location $work
try {
    Write-Host '== H2 edge traces =='
    & java -cp "$classes;$cp" H2EdgeRefGen $fixtures
    if ($LASTEXITCODE -ne 0) { throw 'H2EdgeRefGen failed' }
}
finally {
    Pop-Location
}

Write-Host 'ok'
