# Regenerate the fixed-function primed-TNT flash reference table.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-tnt-flash.ps1
#
# Requires a JDK (developed against Temurin 21), the recompiled vanilla classes in
# MCP-919-main/bin/minecraft, the 1.8.9 natives and a GL driver (an LWJGL 2 offscreen Pbuffer;
# no window is opened). Prints the `FLASH_REFS` table used by
# crates/mc-render/tests/x6_tnt_flash.rs.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\Tessellator.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'

$classes = Join-Path $env:TEMP 'mc-rs-tnt-flash-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'TntFlashRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== TntFlashRef (x6_tnt_flash.rs FLASH_REFS) =='
& java "-Djava.library.path=$natives" -cp "$classes;$cp" TntFlashRef
if ($LASTEXITCODE -ne 0) { throw 'TntFlashRef failed' }
