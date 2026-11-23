# Regenerate the FaceBakery reference fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-assets/javaref/generate-face-bakery.ps1
#
# Requires a JDK and the recompiled vanilla classes in MCP-919-main/bin/minecraft. CPU only: no GL
# context, no natives. Writes crates/mc-assets/tests/fixtures/face_bakery_ref.txt, read by
# crates/mc-assets/tests/face_bakery_java_parity.rs.
#
# Run it from a scratch folder, never from inside crates/: log4j may create a `logs` folder in
# the working directory, and crates/logs breaks cargo.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$jar = Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\block\model\FaceBakery.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    $jar
) -join ';'

$classes = Join-Path $env:TEMP 'mc-rs-face-bakery-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
$out = Join-Path $repo 'mc-rs\crates\mc-assets\tests\fixtures\face_bakery_ref.txt'

Write-Host '== javac =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'FaceBakeryRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== FaceBakeryRef =='
& java -cp "$classes;$cp" FaceBakeryRef $jar $out
if ($LASTEXITCODE -ne 0) { throw 'FaceBakeryRef failed' }
