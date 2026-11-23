# Regenerate the Java projectile-parity fixtures for mc-client.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-projectiles.ps1
#
# Requires a JDK (developed against Temurin 21) and the recompiled vanilla classes in
# MCP-919-main/bin/minecraft (the same prerequisite as crates/mc-physics/javaref/generate.ps1).
# Writes ../tests/fixtures/projectiles/*.txt, read by tests/projectile_parity.rs.
#
# Classpath order matches the other generators: recompiled classes first, libraries next, and the
# vanilla jar LAST so it can never shadow the deobfuscated classes. Class files go to a temporary
# directory, never into the repository.

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'

if (-not (Test-Path (Join-Path $mcp 'bin\minecraft\net\minecraft\entity\Entity.class'))) {
    throw "Missing recompiled vanilla classes at $mcp\bin\minecraft. Run recompile.bat in MCP-919-main first."
}

$jdk = 'C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot\bin'
$javac = if (Test-Path (Join-Path $jdk 'javac.exe')) { Join-Path $jdk 'javac.exe' } else { 'javac' }
$java = if (Test-Path (Join-Path $jdk 'java.exe')) { Join-Path $jdk 'java.exe' } else { 'java' }

$libs = (Get-ChildItem (Join-Path $mcp 'jars\libraries') -Recurse -Filter *.jar | ForEach-Object { $_.FullName }) -join ';'
$cp = @(
    (Join-Path $mcp 'bin\minecraft'),
    $libs,
    (Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar')
) -join ';'

$classes = Join-Path ([System.IO.Path]::GetTempPath()) 'mc-client-projectile-refgen'
$fixtures = Join-Path (Split-Path -Parent $here) 'tests\fixtures\projectiles'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null

Write-Host '== javac =='
& $javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'ProjectileRefGen.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Write-Host '== generate =='
& $java -cp "$classes;$cp" ProjectileRefGen $fixtures
if ($LASTEXITCODE -ne 0) { throw 'ProjectileRefGen failed' }

Write-Host 'ok'
