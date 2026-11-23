# Regenerate the cape chase Java trace.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-cape-trace.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. No GL. CapeTraceRef drives a real EntityOtherPlayerMP through 80
# ticks of movement packets and prints the JAVA_TICKS / JAVA_FRAMES tables of
# crates/mc-client/tests/cape_chase_java_trace.rs (splice them in).

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\entity\EntityOtherPlayerMP.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $env:TEMP 'mc-rs-cape-trace'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'CapeTraceRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== CapeTraceRef (cape_chase_java_trace.rs JAVA_TICKS / JAVA_FRAMES) =='
# Run from the temp class folder: log4j creates `logs/latest.log` in the working directory, and a
# stray `logs` folder under `mc-rs/crates` breaks the cargo workspace (`members = crates/*`).
Push-Location $classes
try {
    & java -cp "$classes;$cp" CapeTraceRef |
        Where-Object { $_ -notmatch '^\[\d\d:\d\d:\d\d\] \[' }
    if ($LASTEXITCODE -ne 0) { throw 'CapeTraceRef failed' }
} finally {
    Pop-Location
}
