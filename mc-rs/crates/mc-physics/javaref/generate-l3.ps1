# Regenerate the block collision-box oracle and the moveEntity edge-case traces for
# `mc-physics`.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-physics/javaref/generate-l3.ps1
#
# Writes ../tests/fixtures/block_boxes.bin (BoxRefGen -> tests/block_boxes.rs) and
# ../tests/fixtures/l3_*.bin (MoveEdgeRefGen -> tests/parity.rs). Same prerequisites as
# generate.ps1: a JDK and the recompiled vanilla classes in MCP-919-main/bin/minecraft.
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

$work = Join-Path $env:TEMP 'mc-physics-javaref-l3'
$classes = Join-Path $work 'build'
$fixtures = Join-Path (Split-Path -Parent $here) 'tests\fixtures'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes `
    (Join-Path $here 'PhysicsRefGen.java') (Join-Path $here 'MoveEdgeRefGen.java') (Join-Path $here 'BoxRefGen.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Push-Location $work
try {
    Write-Host '== block boxes =='
    & java -Xmx2g -cp "$classes;$cp" BoxRefGen (Join-Path $fixtures 'block_boxes.bin') (Join-Path $work 'boxes-dump.txt')
    if ($LASTEXITCODE -ne 0) { throw 'BoxRefGen failed' }

    Write-Host '== moveEntity edge traces =='
    & java -cp "$classes;$cp" MoveEdgeRefGen $fixtures
    if ($LASTEXITCODE -ne 0) { throw 'MoveEdgeRefGen failed' }
}
finally {
    Pop-Location
}

Write-Host "ok (per-query box dump for MC_BOXES_DUMP diffs: $work\boxes-dump.txt)"
