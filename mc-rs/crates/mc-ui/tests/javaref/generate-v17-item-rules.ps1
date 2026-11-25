# 1.7.10 support - regenerate the 1.7.10 item rules fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-ui/tests/javaref/generate-v17-item-rules.ps1 `
#       -Classes17 <dir> -Libraries17 <classpath file> -Jar17 <1.7.10.jar> [-Jdk <dir>] [-Work <dir>]
#
# Compiles V17ItemRulesRef.java against the deobfuscated 1.7.10 client classes and runs it: every
# item 1.7.10 registers, at eight damages, through the real `ItemStack` properties, slot rules and
# shift-clicks of the 1.7.10 containers. The output is
# crates/mc-ui/tests/fixtures/v17_item_rules/item_rules_ref.txt, which
# tests/v5_item_rules_oracle.rs reads.
#
#   -Classes17    the deobfuscated 1.7.10 client classes (javac output of the 1.7.10 client
#                 sources). Default: $env:V17_CLASSES.
#   -Libraries17  a file holding Mojang's 1.7.10 libraries as one `;`-separated classpath.
#                 Default: $env:V17_LIBRARIES.
#   -Jar17        Mojang's 1.7.10 client jar, last on the classpath, for its resources.
#                 Default: $env:V17_JAR.
#   -Jdk          a JDK home (developed against Temurin 8); default $env:JAVA_HOME, else PATH.
#
# Java runs from the work directory, never from inside `crates/`.

param(
    [string]$Classes17 = $env:V17_CLASSES,
    [string]$Libraries17 = $env:V17_LIBRARIES,
    [string]$Jar17 = $env:V17_JAR,
    [string]$Jdk = $env:JAVA_HOME,
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-v17-item-rules')
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$crate = Split-Path -Parent (Split-Path -Parent $here)
$out = Join-Path $crate 'tests\fixtures\v17_item_rules\item_rules_ref.txt'

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
$cp = @($Classes17, (Get-Content -Raw $Libraries17).Trim(), $Jar17) -join ';'
$classes = Join-Path $Work 'classes'
New-Item -ItemType Directory -Force -Path $classes, (Split-Path -Parent $out) | Out-Null

& $javac -nowarn -encoding UTF-8 -d $classes -cp $cp (Join-Path $here 'V17ItemRulesRef.java')
if ($LASTEXITCODE -ne 0) { throw "javac failed ($LASTEXITCODE)" }

Push-Location $Work
try {
    & $java -cp "$classes;$cp" V17ItemRulesRef $out
    if ($LASTEXITCODE -ne 0) { throw "V17ItemRulesRef failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}
Write-Host "wrote $out"
