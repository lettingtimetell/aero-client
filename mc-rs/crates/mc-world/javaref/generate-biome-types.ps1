# Regenerate the seed-0 biome fallback fixture for the packet world
# types (default, default_1_1, largeBiomes).
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-world/javaref/generate-biome-types.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft, like generate.ps1. Writes ../tests/fixtures/biome/world_types_seed0.txt,
# which `src/genlayer.rs`'s `packet_world_type_fallbacks_match_java_world_chunk_manager` reads.
#
# Classpath order matches generate.ps1: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes. `Bootstrap.register()` starts
# log4j, which creates a `logs` folder in the working directory, so Java runs from a temporary
# work directory, never from inside `crates/`.

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\world\biome\WorldChunkManager.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $here 'build'
$fixtures = Join-Path (Split-Path -Parent $here) 'tests\fixtures\biome'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'BiomeTypeRefGen.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== generate =='
$work = Join-Path $env:TEMP 'mc-rs-biome-types'
New-Item -ItemType Directory -Force -Path $work | Out-Null
Push-Location $work
try {
    & java -cp "$classes;$cp" BiomeTypeRefGen (Join-Path $fixtures 'world_types_seed0.txt')
    if ($LASTEXITCODE -ne 0) { throw 'BiomeTypeRefGen failed' }
} finally {
    Pop-Location
}

Write-Host 'ok'
