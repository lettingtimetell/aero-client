# Regenerate the creative inventory oracle fixtures (1.7.10 and 1.8.9).
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-ui/tests/javaref/generate-creative-tabs.ps1 `
#       -Classes17 <dir> -Libraries17 <classpath file> -Jar17 <1.7.10.jar> `
#       -Classes18 <dir> -Libraries18 <classpath file> -Jar18 <1.8.9.jar> [-Jdk <dir>] [-Work <dir>]
#
# Compiles CreativeTabsRef.java against each version's deobfuscated client classes and runs it:
# every creative tab's list and the search tab's pool, every stack in full. The output is
# crates/mc-ui/tests/fixtures/creative/creative_1_7_10.txt and creative_1_8_9.txt, which
# tests/creative_tabs_oracle.rs reads.
#
#   -Classes17, -Classes18      the deobfuscated client classes of 1.7.10 and 1.8.9 (javac output
#                               of each version's client sources). Defaults: $env:V17_CLASSES,
#                               $env:V18_CLASSES.
#   -Libraries17, -Libraries18  a file holding Mojang's libraries of that version as one
#                               `;`-separated classpath. Defaults: $env:V17_LIBRARIES,
#                               $env:V18_LIBRARIES.
#   -Jar17, -Jar18              Mojang's client jar of that version, last on the classpath, for
#                               its resources. Defaults: $env:V17_JAR, $env:V18_JAR.
#   -Jdk                        a JDK home (developed against Temurin 8); default
#                               $env:JAVA_HOME, else PATH.
#
# Java runs from the work directory, never from inside `crates/`.

param(
    [string]$Classes17 = $env:V17_CLASSES,
    [string]$Libraries17 = $env:V17_LIBRARIES,
    [string]$Jar17 = $env:V17_JAR,
    [string]$Classes18 = $env:V18_CLASSES,
    [string]$Libraries18 = $env:V18_LIBRARIES,
    [string]$Jar18 = $env:V18_JAR,
    [string]$Jdk = $env:JAVA_HOME,
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-creative-oracle')
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$crate = Split-Path -Parent (Split-Path -Parent $here)
$fixtures = Join-Path $crate 'tests\fixtures\creative'

function Find-Tool([string]$name) {
    if ($Jdk) {
        $tool = Join-Path $Jdk "bin\$name.exe"
        if (Test-Path $tool) { return $tool }
        throw "No $name in $Jdk\bin"
    }
    return (Get-Command $name).Source
}

foreach ($required in @($Classes17, $Libraries17, $Jar17, $Classes18, $Libraries18, $Jar18)) {
    if (-not $required -or -not (Test-Path $required)) {
        throw "Missing input: '$required' (see the parameters at the top of this script)"
    }
}

$javac = Find-Tool 'javac'
$java = Find-Tool 'java'
$versions = @(
    @{ Name = '1.7.10'; Key = '17'; Out = 'creative_1_7_10.txt'
        Classpath = @($Classes17, (Get-Content -Raw $Libraries17).Trim(), $Jar17) -join ';' },
    @{ Name = '1.8.9'; Key = '18'; Out = 'creative_1_8_9.txt'
        Classpath = @($Classes18, (Get-Content -Raw $Libraries18).Trim(), $Jar18) -join ';' }
)
New-Item -ItemType Directory -Force -Path $Work, $fixtures | Out-Null

Push-Location $Work
try {
    foreach ($v in $versions) {
        $classes = Join-Path $Work "classes$($v.Key)"
        New-Item -ItemType Directory -Force -Path $classes | Out-Null
        & $javac -nowarn -encoding UTF-8 -d $classes -cp $v.Classpath (Join-Path $here 'CreativeTabsRef.java')
        if ($LASTEXITCODE -ne 0) { throw "javac failed for $($v.Name) ($LASTEXITCODE)" }
        & $java -cp "$classes;$($v.Classpath)" CreativeTabsRef (Join-Path $fixtures $v.Out)
        if ($LASTEXITCODE -ne 0) { throw "CreativeTabsRef failed for $($v.Name) ($LASTEXITCODE)" }
    }
} finally {
    Pop-Location
}
Write-Host "wrote $fixtures"
