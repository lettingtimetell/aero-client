# Regenerate the held-item equip-animation Java trace.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-held-equip-trace.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft. No GL is used: HeldEquipTraceRef drives the real
# ItemRenderer.updateEquippedItem / resetEquippedProgress(2) over real ItemStack objects and prints
# the JAVA_TRACES table of crates/mc-client/src/app/first_person_equip.rs (splice it in place of the
# existing table).
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\ItemRenderer.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path $env:TEMP 'mc-rs-held-equip-trace'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'HeldEquipTraceRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== HeldEquipTraceRef (first_person_equip.rs JAVA_TRACES) =='
& java -cp "$classes;$cp" HeldEquipTraceRef |
    Where-Object { $_ -notmatch '^\[\d\d:\d\d:\d\d\] \[' }
if ($LASTEXITCODE -ne 0) { throw 'HeldEquipTraceRef failed' }
