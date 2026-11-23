# Regenerate the committed Java DEBUG_WORLD oracle TSV and compact Rust schema.
# Requires the same JDK/classpath as run.ps1; this does not launch a client/server.
$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\block\Block.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft"
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar |
    ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $here 'build'
$out = Join-Path $here 'out'
New-Item -ItemType Directory -Force -Path $classes,$out | Out-Null

Write-Host '== javac DEBUG_WORLD oracle =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'DebugGridProbe.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== Java registry dump =='
& java -cp "$classes;$cp" DebugGridProbe (Join-Path $out 'debug-grid.tsv')
if ($LASTEXITCODE -ne 0) { throw 'DEBUG_WORLD probe failed' }

Write-Host '== compact Rust schema =='
& python (Join-Path $here 'generate_debug_states.py')
if ($LASTEXITCODE -ne 0) { throw 'DEBUG_WORLD schema generation failed' }

Write-Host 'ok'
