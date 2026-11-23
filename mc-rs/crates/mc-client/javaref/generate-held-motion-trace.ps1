# Regenerate the swing / arm-sway Java trace.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-held-motion-trace.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. No GL. HeldMotionTraceRef drives the real EntityPlayerSP swing and
# arm-sway state through a 12 CPS / 90 deg/s / flick timeline and prints the JAVA_FRAMES table of
# crates/mc-client/tests/first_person_motion_java_trace.rs (splice it in).

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\entity\EntityPlayerSP.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $env:TEMP 'mc-rs-held-motion-trace'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'HeldMotionTraceRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== HeldMotionTraceRef (first_person_motion_java_trace.rs JAVA_FRAMES) =='
& java -cp "$classes;$cp" HeldMotionTraceRef |
    Where-Object { $_ -notmatch '^\[\d\d:\d\d:\d\d\] \[' }
if ($LASTEXITCODE -ne 0) { throw 'HeldMotionTraceRef failed' }
