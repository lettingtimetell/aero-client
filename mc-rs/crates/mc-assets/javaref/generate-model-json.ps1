# Regenerate the model/blockstate JSON acceptance fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-assets/javaref/generate-model-json.ps1
#
# Requires a JDK and the recompiled vanilla 1.8.9 classes in MCP-919-main/bin/minecraft. CPU only:
# no GL context, no natives. Writes crates/mc-assets/tests/fixtures/model_json_ref.tsv, read by
# crates/mc-assets/tests/model_json_java_parity.rs.
#
# Run it from a scratch folder, never from inside crates/: log4j may create a `logs` folder in
# the working directory, and crates/logs breaks cargo.

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$jar = Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\renderer\block\model\ModelBlock.class'))) {
    throw "Missing the recompiled vanilla classes at $mcp\bin\minecraft. Recompile the source tree first."
}

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    $jar
) -join ';'

$classes = Join-Path $env:TEMP 'mc-rs-model-json-javaref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
$out = Join-Path $repo 'mc-rs\crates\mc-assets\tests\fixtures\model_json_ref.tsv'

& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'ModelJsonRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }
& java -cp "$classes;$cp" ModelJsonRef $out
if ($LASTEXITCODE -ne 0) { throw 'ModelJsonRef failed' }
Write-Host "wrote $out"
