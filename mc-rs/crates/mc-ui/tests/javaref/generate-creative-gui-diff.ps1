# Regenerate the creative inventory click fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-ui/tests/javaref/generate-creative-gui-diff.ps1 [-Work <dir>] [-Out <path>]
#
# Compiles CreativeGuiDiffRef.java against the recompiled vanilla classes in MCP-919-main/bin/minecraft,
# with the headless LWJGL stubs of crates/mc-client/javaref/h1stubs ahead of an UNSEALED copy of the
# LWJGL jar (the real jar is `Sealed: true`), and writes
# crates/mc-ui/tests/fixtures/w21_ci_creative/creative_gui_diff.txt, which mc-ui's
# `screens/container/creative_oracle.rs` replays through the Rust creative screen.
#
# Java runs from the work directory, never from inside `crates/` (log4j creates a `logs` folder in
# the working directory). The `java` next to `javac` runs it: an older runtime earlier on PATH cannot
# load classes compiled by a newer JDK.

param(
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-creative-gui-diff'),
    [string]$Out = ''
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$stubDir = (Resolve-Path (Join-Path $here '..\..\..\mc-client\javaref\h1stubs')).Path

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\client\gui\inventory\GuiContainerCreative.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$lwjglJar = Get-ChildItem (Join-Path $mcp 'jars\libraries\org\lwjgl\lwjgl\lwjgl') -Recurse -Filter 'lwjgl-*.jar' |
    Select-Object -First 1
if (-not $lwjglJar) { throw 'LWJGL jar not found under jars\libraries\org\lwjgl\lwjgl\lwjgl' }

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar |
    Where-Object { $_.FullName -ne $lwjglJar.FullName } |
    ForEach-Object { $_.FullName }) -join ';'

New-Item -ItemType Directory -Force -Path $Work | Out-Null
$lwjgl = Join-Path $Work 'lwjgl-unsealed'
$classes = Join-Path $Work 'classes'
if (Test-Path $lwjgl) { Remove-Item -Recurse -Force $lwjgl }
if (Test-Path $classes) { Remove-Item -Recurse -Force $classes }
New-Item -ItemType Directory -Force -Path $lwjgl, $classes | Out-Null

Write-Host "== unsealed LWJGL copy -> $lwjgl =="
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($lwjglJar.FullName)
try {
    foreach ($entry in $zip.Entries) {
        if ($entry.FullName.EndsWith('/') -or $entry.FullName.StartsWith('META-INF/')) { continue }
        $dest = Join-Path $lwjgl ($entry.FullName -replace '/', '\')
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
    }
} finally {
    $zip.Dispose()
}

Write-Host '== javac stubs (over the extracted originals) =='
$stubs = Get-ChildItem $stubDir -Recurse -Filter *.java | ForEach-Object { $_.FullName }
& javac -nowarn -encoding UTF-8 -d $lwjgl @stubs
if ($LASTEXITCODE -ne 0) { throw 'javac (stubs) failed' }

$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $lwjgl,
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

Write-Host '== javac CreativeGuiDiffRef =='
& javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'CreativeGuiDiffRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

$out = if ($Out) { $Out } else { Join-Path (Split-Path -Parent $here) 'fixtures\w21_ci_creative\creative_gui_diff.txt' }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
$java = Join-Path (Split-Path -Parent (Get-Command javac).Source) 'java.exe'
if (-not (Test-Path $java)) { $java = 'java' }
Write-Host "== CreativeGuiDiffRef -> $out ($java) =="
Push-Location $Work
try {
    & $java -cp "$classes;$cp" CreativeGuiDiffRef $out
    if ($LASTEXITCODE -ne 0) { throw 'CreativeGuiDiffRef failed' }
} finally {
    Pop-Location
}
Write-Host 'ok'
