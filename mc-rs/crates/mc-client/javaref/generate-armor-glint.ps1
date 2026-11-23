# Regenerate the fixed-function armour-foil reference frames.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-armor-glint.ps1
#
# Requires a JDK (developed against Temurin 21), the recompiled vanilla classes in
# MCP-919-main/bin/minecraft, the 1.8.9 natives and a GL driver (an LWJGL 2 offscreen Pbuffer;
# no window is opened). Writes crates/mc-render/tests/snapshots/x6_armor_glint_gl-<age*10>.png,
# read by crates/mc-render/tests/x6_armor_glint.rs.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$jar = Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\Tessellator.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    $jar
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'

$classes = Join-Path $env:TEMP 'mc-rs-armor-glint-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
$out = Join-Path $repo 'mc-rs\crates\mc-render\tests\snapshots\x6_armor_glint_gl'

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'ArmorGlintRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== ArmorGlintRef =='
& java "-Djava.library.path=$natives" -cp "$classes;$cp" ArmorGlintRef $jar $out
if ($LASTEXITCODE -ne 0) { throw 'ArmorGlintRef failed' }
