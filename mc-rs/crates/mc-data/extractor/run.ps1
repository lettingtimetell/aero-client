# S01 — build + run the mc-data registry extractor, then regenerate the Rust tables.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-data/extractor/run.ps1
#
# Requires a JDK (tested: Temurin 21) and Python 3 on PATH.
#
# Classpath, in order:
#   1. MCP-919-main/bin/minecraft        — recompiled, deobfuscated 1.8.9 classes (the real game)
#   2. MCP-919-main/jars/libraries/**    — guava, log4j, commons, lwjgl, ...
#   3. MCP-919-main/jars/versions/1.8.9/1.8.9.jar
#        vanilla jar, LAST so it never shadows (1); it is only there for resources —
#        `assets/minecraft/lang/en_US.lang` (StatList.init needs it) and `sounds.json`.

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp  = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\block\Block.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp   = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $here 'build'
$out     = Join-Path $here 'out'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $out     | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'McDataExtractor.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== extract =='
& java -cp "$classes;$cp" McDataExtractor $out
if ($LASTEXITCODE -ne 0) { throw 'extractor failed' }

Write-Host '== codegen =='
& python (Join-Path $here 'codegen.py') $out (Join-Path $here '..\src\generated')
if ($LASTEXITCODE -ne 0) { throw 'codegen failed' }

# CI runs `cargo fmt --check`, and rustfmt reflows the generated tables, so normalise them
# here. Without this a regenerate would leave the tree failing the format check.
Write-Host '== fmt =='
Push-Location (Join-Path $here '..\..\..')
try {
    & cargo fmt -p mc-data
    if ($LASTEXITCODE -ne 0) { throw 'cargo fmt failed' }
} finally {
    Pop-Location
}

Write-Host 'ok'
