# The Java side of the decode oracle.
#
#   $env:Z1_ORACLE_DIR = '<an empty scratch folder>'
#   cargo test -p mc-client --test z1_hostile_server -- --ignored dump_cases_for_the_java_decode_oracle
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-z1-decode.ps1 PLAY
#   cargo test -p mc-client --test z1_hostile_server -- --ignored compare_with_the_java_decode_oracle --nocapture
#
# The first step writes cases.txt (damaged real packets, one hex payload per line) and the
# client's verdicts. This script decodes every case with the real 1.8.9 classes
# (Z1DecodeOracle.java) into java.txt. The last step lists every payload vanilla accepts and the
# client refuses: each one is a disconnect vanilla would not have.
#
# Requires a JDK and the recompiled vanilla classes in MCP-919-main/bin/minecraft. Java runs from a
# temp directory, never from inside `crates/` (log4j creates a `logs` folder in the working
# directory, and `crates/logs` breaks every cargo command). The heap is capped so a hostile length
# fails fast with OutOfMemoryError, which vanilla's pipeline also turns into a disconnect.

param([string]$State = 'PLAY')

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$dir = $env:Z1_ORACLE_DIR
if (-not $dir -or -not (Test-Path (Join-Path $dir 'cases.txt'))) {
    throw 'Set Z1_ORACLE_DIR to the folder the dump test wrote cases.txt into.'
}
if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\network\PacketBuffer.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$work = Join-Path $env:TEMP 'mc-rs-z1-decode-javaref'
$classes = Join-Path $work 'classes'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
# `--release 8`: the class must also run on a Java 8 `java`, the runtime 1.8.9 ships with.
& javac -nowarn --release 8 -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'Z1DecodeOracle.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host "== Z1DecodeOracle $State -> $dir\java.txt =="
Push-Location $work
try {
    & java -Xmx512m -cp "$classes;$cp" Z1DecodeOracle $State (Join-Path $dir 'cases.txt') (Join-Path $dir 'java.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Z1DecodeOracle failed' }
} finally {
    Pop-Location
}
