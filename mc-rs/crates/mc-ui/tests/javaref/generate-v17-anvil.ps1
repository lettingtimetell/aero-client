# 1.7.10 support - regenerate the 1.7.10 anvil fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-ui/tests/javaref/generate-v17-anvil.ps1 `
#       -Classes17 <dir> -Libraries17 <classpath file> -Jar17 <1.7.10.jar> [-Jdk <dir>] [-Work <dir>]
#
# Compiles V17AnvilRef.java against the deobfuscated 1.7.10 client classes and runs it: the real
# `ContainerRepair.updateRepairOutput` on the cases listed in the Java file. The output is
# crates/mc-ui/tests/fixtures/v17_anvil/anvil_ref.txt, which mc-ui's 1.7.10 anvil test reads.
#
#   -Classes17    the deobfuscated 1.7.10 classes (javac output of the 1.7.10 client sources, as the
#                 1.7.10 movement oracle's generator builds them). Default: $env:V17_CLASSES.
#   -Libraries17  a file holding Mojang's 1.7.10 libraries as one `;`-separated classpath.
#                 Default: $env:V17_LIBRARIES.
#   -Jar17        Mojang's 1.7.10 client jar, for its language files only (item names).
#                 Default: $env:V17_JAR.
#   -Jdk          a JDK home (developed against Temurin 8); default $env:JAVA_HOME, else PATH.
#
# Java runs from the work directory, never from inside `crates/`.

param(
    [string]$Classes17 = $env:V17_CLASSES,
    [string]$Libraries17 = $env:V17_LIBRARIES,
    [string]$Jar17 = $env:V17_JAR,
    [string]$Jdk = $env:JAVA_HOME,
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-v17-anvil')
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$crate = Split-Path -Parent (Split-Path -Parent $here)
$out = Join-Path $crate 'tests\fixtures\v17_anvil\anvil_ref.txt'

function Find-Tool([string]$name) {
    if ($Jdk) {
        $tool = Join-Path $Jdk "bin\$name.exe"
        if (Test-Path $tool) { return $tool }
        throw "No $name in $Jdk\bin"
    }
    return (Get-Command $name).Source
}

foreach ($required in @($Classes17, $Libraries17, $Jar17)) {
    if (-not $required -or -not (Test-Path $required)) {
        throw "Missing input: '$required' (see the parameters at the top of this script)"
    }
}

$javac = Find-Tool 'javac'
$java = Find-Tool 'java'
$libraries = (Get-Content -Raw $Libraries17).Trim()
$classes = Join-Path $Work 'classes'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

$compileCp = "$Classes17;$libraries"
& $javac -nowarn -encoding UTF-8 -d $classes -cp $compileCp (Join-Path $here 'V17AnvilRef.java')
if ($LASTEXITCODE -ne 0) { throw "javac failed ($LASTEXITCODE)" }

$runCp = "$classes;$Classes17;$libraries;$Jar17"
Push-Location $Work
try {
    $lines = & $java -cp $runCp V17AnvilRef
    if ($LASTEXITCODE -ne 0) { throw "V17AnvilRef failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
[System.IO.File]::WriteAllLines($out, [string[]]$lines, (New-Object System.Text.UTF8Encoding $false))
Write-Host "wrote $($lines.Count) cases to $out"
