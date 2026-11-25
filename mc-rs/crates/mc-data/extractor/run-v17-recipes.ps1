# 1.7.10 support - regenerate the 1.7.10 crafting recipe table.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-data/extractor/run-v17-recipes.ps1 `
#       -Classes17 <dir> -Libraries17 <classpath file> -Jar17 <1.7.10.jar> [-Jdk <dir>] [-Work <dir>]
#
# Compiles V17RecipeDump.java against the deobfuscated 1.7.10 client classes and runs it: the real
# `CraftingManager` builds and sorts its list, and the dump (out/recipes_v5.tsv) holds every
# recipe in that order. codegen_recipes_v5.py then writes src/v5/recipe_table.rs from it.
#
#   -Classes17    the deobfuscated 1.7.10 client classes (javac output of the 1.7.10 client
#                 sources). Default: $env:V17_CLASSES.
#   -Libraries17  a file holding Mojang's 1.7.10 libraries as one `;`-separated classpath.
#                 Default: $env:V17_LIBRARIES.
#   -Jar17        Mojang's 1.7.10 client jar, last on the classpath, for its resources.
#                 Default: $env:V17_JAR.
#   -Jdk          a JDK home (developed against Temurin 8); default $env:JAVA_HOME, else PATH.
#
# Requires Python 3 on PATH. Java runs from the work directory, never from inside `crates/`.

param(
    [string]$Classes17 = $env:V17_CLASSES,
    [string]$Libraries17 = $env:V17_LIBRARIES,
    [string]$Jar17 = $env:V17_JAR,
    [string]$Jdk = $env:JAVA_HOME,
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-v17-recipes')
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tsv = Join-Path $here 'out\recipes_v5.tsv'
$src = Join-Path $here '..\src\v5'

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
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& $javac -nowarn -encoding UTF-8 -d $classes -cp $cp (Join-Path $here 'V17RecipeDump.java')
if ($LASTEXITCODE -ne 0) { throw "javac failed ($LASTEXITCODE)" }

Write-Host '== dump =='
Push-Location $Work
try {
    & $java -cp "$classes;$cp" V17RecipeDump $tsv
    if ($LASTEXITCODE -ne 0) { throw "V17RecipeDump failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

Write-Host '== codegen =='
& python (Join-Path $here 'codegen_recipes_v5.py') $tsv $src
if ($LASTEXITCODE -ne 0) { throw 'codegen failed' }

Write-Host 'ok'
