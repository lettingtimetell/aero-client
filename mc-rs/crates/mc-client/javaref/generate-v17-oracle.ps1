# 1.7.10 support - regenerate the 1.7.10 differential oracle's fixture.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-v17-oracle.ps1 `
#       -Vanilla17 <dir> -Src17 <dir> [-Mcp189 <dir>] [-Log4jConfig <file>] [-Work <dir>]
#       [-Filter <name>] [-Out17 <dir>] [-No189] [-Rebuild] [-Jdk <dir>]
#
# The oracle runs the REAL 1.7.10 `Minecraft.runTick()` headless (V17OracleRef.java): the
# deobfuscated 1.7.10 client sources, compiled by this script against Mojang's 1.7.10 libraries, with
# LWJGL's Display/Mouse/Keyboard/Sys replaced by the stubs in javaref/v17stubs (compiled over an
# UNSEALED copy of the LWJGL jar: the real jar is `Sealed: true`). It plays every scenario in
# tests/fixtures/v17_oracle/scenarios and writes one result file per scenario, plus the shared
# inbound packet table, to tests/fixtures/v17_oracle/java17.
#
# Unless -No189 is given, the same scenarios then run on the 1.8.9 harness (V17Cross189Ref.java,
# reusing MovementDiffRef's harness and the recompiled classes in MCP-919-main/bin/minecraft, as
# generate-movement-diff.ps1 does), and V17CompareRef.java writes <Work>/compare.md: per scenario,
# every state field and packet that differs between the two versions. The 1.8.9 results stay in
# the work directory.
#
#   -Vanilla17   Mojang's 1.7.10 files in launcher layout (versions/1.7.10/1.7.10.jar and
#                libraries/), checked against the sha1s in 1.7.10.json. Required: this or
#                $env:V17_VANILLA.
#   -Src17       the deobfuscated 1.7.10 client sources: a folder holding src/minecraft/net/...
#                Required: this or $env:V17_SRC.
#   -Mcp189      the 1.8.9 workspace with recompiled classes (bin/minecraft) and jars/. Default:
#                MCP-919-main next to or above the repository.
#   -Log4jConfig Mojang's client logging config for 1.7 (client-1.7.xml, the `logging.client` file
#                of 1.7.10.json; its sha1 is checked). The 1.7.10 and 1.8.9 clients use log4j
#                2.0-beta9; without this file the script writes a config with the same `${...}`
#                message filter into the work directory.
#   -Filter      only the scenarios whose name contains this text (writes to <Work>/java17 unless
#                -Out17 is given: a filtered run is not the committed fixture).
#   -Rebuild     recompile the 1.7.10 sources (otherwise reused from the work directory).
#   -Jdk         a JDK home (developed against Temurin 8); default $env:JAVA_HOME, else PATH.
#
# Java runs from the work directory, never from inside `crates/`: log4j creates a `logs` folder in
# the working directory, and `crates/logs` breaks every cargo command.

param(
    [string]$Vanilla17 = $env:V17_VANILLA,
    [string]$Src17 = $env:V17_SRC,
    [string]$Mcp189 = '',
    [string]$Log4jConfig = $env:V17_LOG4J_CONFIG,
    [string]$Work = (Join-Path $env:TEMP 'mc-rs-v17-oracle'),
    [string]$Filter = '',
    [string]$Out17 = '',
    [switch]$No189,
    [switch]$Rebuild,
    [string]$Jdk = $env:JAVA_HOME
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$crate = Split-Path -Parent $here
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$scenarios = Join-Path $crate 'tests\fixtures\v17_oracle\scenarios'

function Find-Tool([string]$name) {
    if ($Jdk) {
        $tool = Join-Path $Jdk "bin\$name.exe"
        if (Test-Path $tool) { return $tool }
        throw "No $name in $Jdk\bin"
    }
    return (Get-Command $name).Source
}

function Find-Dir([string]$given, [string[]]$candidates, [string]$probe, [string]$what) {
    $all = @()
    if ($given) { $all += $given }
    $all += $candidates
    if ($all.Count -eq 0) {
        throw "Missing input: $what (see the parameters at the top of this script)"
    }
    foreach ($c in $all) {
        if ($c -and (Test-Path (Join-Path $c $probe))) { return (Resolve-Path $c).Path }
    }
    throw "Cannot find $what (looked for $probe under: $($all -join ', '))"
}

# javac reads @argfiles in the platform encoding; quoted, forward-slashed paths survive spaces.
function Write-ArgFile([string]$path, [string[]]$files) {
    $lines = $files | ForEach-Object { '"' + ($_ -replace '\\', '/') + '"' }
    [System.IO.File]::WriteAllLines($path, [string[]]$lines, (New-Object System.Text.ASCIIEncoding))
}

function Expand-Unsealed([string]$jar, [string]$dest) {
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($jar)
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName.EndsWith('/') -or $entry.FullName.StartsWith('META-INF/')) { continue }
            $target = Join-Path $dest ($entry.FullName -replace '/', '\')
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        }
    } finally {
        $zip.Dispose()
    }
}

function Invoke-Checked([string]$what, [scriptblock]$block) {
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

$javac = Find-Tool 'javac'
$java = Find-Tool 'java'
$Vanilla17 = Find-Dir $Vanilla17 @() 'versions\1.7.10\1.7.10.jar' `
    'the 1.7.10 vanilla files (-Vanilla17 <dir> or $env:V17_VANILLA)'
$Src17 = Find-Dir $Src17 @() 'src\minecraft\net\minecraft\client\Minecraft.java' `
    'the 1.7.10 client sources (-Src17 <dir> or $env:V17_SRC)'
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# ---------------------------------------------------------------- logging
if ($Log4jConfig) {
    $sha1 = (Get-FileHash -Algorithm SHA1 $Log4jConfig).Hash.ToLowerInvariant()
    if ($sha1 -ne '50c9cc4af6d853d9fc137c84bcd153e2bd3a9a82') {
        throw "$Log4jConfig is not Mojang's client-1.7.xml (sha1 $sha1)"
    }
} else {
    $Log4jConfig = Join-Path $Work 'log4j-filtered.xml'
    $xml = @'
<?xml version="1.0" encoding="UTF-8"?>
<Configuration status="WARN">
    <Appenders>
        <Console name="SysOut" target="SYSTEM_OUT">
            <PatternLayout pattern="[%d{HH:mm:ss}] [%t/%level]: %msg%n" />
        </Console>
    </Appenders>
    <Loggers>
        <Root level="info">
            <filters>
                <MarkerFilter marker="NETWORK_PACKETS" onMatch="DENY" onMismatch="NEUTRAL" />
                <RegexFilter regex="(?s).*\$\{[^}]*\}.*" onMatch="DENY" onMismatch="NEUTRAL"/>
            </filters>
            <AppenderRef ref="SysOut"/>
        </Root>
    </Loggers>
</Configuration>
'@
    [System.IO.File]::WriteAllText($Log4jConfig, $xml, (New-Object System.Text.ASCIIEncoding))
}
$log4j = "-Dlog4j.configurationFile=$($Log4jConfig -replace '\\', '/')"

# ---------------------------------------------------------------- 1.7.10
$libs17 = Get-ChildItem (Join-Path $Vanilla17 'libraries') -Recurse -Filter *.jar |
    Where-Object { $_.Name -notlike '*natives*' -and $_.Name -notlike 'lwjgl-2*' } |
    ForEach-Object { $_.FullName }
$lwjglJar17 = Get-ChildItem (Join-Path $Vanilla17 'libraries\org\lwjgl\lwjgl\lwjgl') -Recurse -Filter 'lwjgl-*.jar' |
    Select-Object -First 1
if (-not $lwjglJar17) { throw 'LWJGL jar not found in the 1.7.10 libraries' }
$jar17 = Join-Path $Vanilla17 'versions\1.7.10\1.7.10.jar'

$mc17 = Join-Path $Work 'mc17-classes'
if ($Rebuild -or -not (Test-Path (Join-Path $mc17 'net\minecraft\client\Minecraft.class'))) {
    Write-Host "== javac the 1.7.10 client sources -> $mc17 =="
    if (Test-Path $mc17) { Remove-Item -Recurse -Force $mc17 }
    New-Item -ItemType Directory -Force -Path $mc17 | Out-Null
    $sources = Get-ChildItem (Join-Path $Src17 'src\minecraft\net') -Recurse -Filter *.java |
        ForEach-Object { $_.FullName }
    $argfile = Join-Path $Work 'mc17-sources.txt'
    Write-ArgFile $argfile $sources
    $cp = (@($lwjglJar17.FullName) + $libs17) -join ';'
    Invoke-Checked 'javac (1.7.10 sources)' {
        & $javac -J-Xmx2g -nowarn -encoding UTF-8 -proc:none -cp $cp -d $mc17 "@$argfile"
    }
}

Write-Host '== unsealed LWJGL 2.9.1 + v17stubs =='
$lwjgl17 = Join-Path $Work 'lwjgl17-unsealed'
Expand-Unsealed $lwjglJar17.FullName $lwjgl17
$stubs = Get-ChildItem (Join-Path $here 'v17stubs') -Recurse -Filter *.java | ForEach-Object { $_.FullName }
Invoke-Checked 'javac (stubs, 1.7.10)' { & $javac -nowarn -encoding UTF-8 -d $lwjgl17 $stubs }

$cp17 = (@($mc17, $lwjgl17) + $libs17 + @($jar17)) -join ';'
$classes17 = Join-Path $Work 'classes17'
if (Test-Path $classes17) { Remove-Item -Recurse -Force $classes17 }
New-Item -ItemType Directory -Force -Path $classes17 | Out-Null
Write-Host '== javac V17Scenario + V17OracleRef + V17CompareRef =='
Invoke-Checked 'javac (V17OracleRef)' {
    & $javac -nowarn -encoding UTF-8 -cp $cp17 -d $classes17 (Join-Path $here 'V17Scenario.java') `
        (Join-Path $here 'V17OracleRef.java') (Join-Path $here 'V17CompareRef.java')
}

if (-not $Out17) {
    if ($Filter) {
        $Out17 = Join-Path $Work 'java17'
    } else {
        $Out17 = Join-Path $crate 'tests\fixtures\v17_oracle\java17'
    }
}
if (Test-Path $Out17) { Get-ChildItem $Out17 -Filter *.json | Remove-Item -Force }
Write-Host "== V17OracleRef -> $Out17 =="
Push-Location $Work
try {
    $args17 = @($scenarios, $Out17)
    if ($Filter) { $args17 += $Filter }
    Invoke-Checked 'V17OracleRef' { & $java -Xmx1g $log4j -cp "$classes17;$cp17" V17OracleRef @args17 }
} finally {
    Pop-Location
}

if ($No189) {
    Write-Host 'ok (1.7.10 only)'
    return
}

# ---------------------------------------------------------------- 1.8.9 and the comparison
$Mcp189 = Find-Dir $Mcp189 @((Join-Path $repo 'MCP-919-main'), (Join-Path $repo '..\MCP-919-main')) `
    'bin\minecraft\net\minecraft\entity\Entity.class' 'the recompiled 1.8.9 classes (-Mcp189; run recompile.bat)'
$lwjglJar189 = Get-ChildItem (Join-Path $Mcp189 'jars\libraries\org\lwjgl\lwjgl\lwjgl') -Recurse -Filter 'lwjgl-*.jar' |
    Select-Object -First 1
if (-not $lwjglJar189) { throw 'LWJGL jar not found under jars\libraries\org\lwjgl\lwjgl\lwjgl' }
$libs189 = Get-ChildItem (Join-Path $Mcp189 'jars\libraries') -Recurse -Filter *.jar |
    Where-Object { $_.FullName -ne $lwjglJar189.FullName } | ForEach-Object { $_.FullName }

Write-Host '== unsealed LWJGL (1.8.9) + v17stubs =='
$lwjgl189 = Join-Path $Work 'lwjgl189-unsealed'
Expand-Unsealed $lwjglJar189.FullName $lwjgl189
Invoke-Checked 'javac (stubs, 1.8.9)' { & $javac -nowarn -encoding UTF-8 -d $lwjgl189 $stubs }

$jar189 = Join-Path $Mcp189 'jars\versions\1.8.9\1.8.9.jar'
$cp189 = (@((Join-Path $Mcp189 'bin\minecraft'), $lwjgl189) + $libs189 + @($jar189)) -join ';'
$classes189 = Join-Path $Work 'classes189'
if (Test-Path $classes189) { Remove-Item -Recurse -Force $classes189 }
New-Item -ItemType Directory -Force -Path $classes189 | Out-Null
Write-Host '== javac MovementDiffRef + V17Scenario + V17Cross189Ref =='
Invoke-Checked 'javac (V17Cross189Ref)' {
    & $javac -nowarn -encoding UTF-8 -cp $cp189 -d $classes189 (Join-Path $here 'MovementDiffRef.java') `
        (Join-Path $here 'V17Scenario.java') (Join-Path $here 'V17Cross189Ref.java')
}

$out18 = Join-Path $Work 'java18'
if (Test-Path $out18) { Get-ChildItem $out18 -Filter *.json | Remove-Item -Force }
Write-Host "== V17Cross189Ref -> $out18 =="
Push-Location $Work
try {
    $args18 = @($scenarios, $out18)
    if ($Filter) { $args18 += $Filter }
    Invoke-Checked 'V17Cross189Ref' { & $java -Xmx1g $log4j -cp "$classes189;$cp189" V17Cross189Ref @args18 }
    $report = Join-Path $Work 'compare.md'
    Invoke-Checked 'V17CompareRef' { & $java -cp "$classes17;$cp17" V17CompareRef $Out17 $out18 $report }
} finally {
    Pop-Location
}
Write-Host "ok: $Out17, comparison in $(Join-Path $Work 'compare.md')"
