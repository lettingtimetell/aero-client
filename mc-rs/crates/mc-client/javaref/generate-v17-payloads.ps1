# 1.7.10 support - regenerate the 1.7.10 plugin-message payload references.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-v17-payloads.ps1 `
#       -Classes17 <dir> -Vanilla17 <dir> [-Jdk <dir>]
#
# Runs V17PayloadRef.java on the real 1.7.10 classes: the MC|TrSel, MC|Beacon, MC|AdvCdm,
# MC|BEdit and MC|BSign payloads that the 1.7.10 screens write into a Netty buffer (sent as the
# buffer's whole backing array), including the 16384-byte limit past which nothing is sent.
# Writes crates/mc-client/javaref/v17_payload_ref.txt, which `src/outbound_v5.rs`'s tests read.
#
#   -Classes17  the 1.7.10 client classes compiled from the deobfuscated sources (the
#               mc17-classes folder generate-v17-oracle.ps1 builds in its work directory).
#               Default: $env:V17_CLASSES, else that work directory's folder.
#   -Vanilla17  Mojang's 1.7.10 files in launcher layout (libraries/ holds Netty 4.0.10 and the
#               rest). Default: $env:V17_VANILLA.
#   -Jdk        a JDK home (developed against Temurin 8); default $env:JAVA_HOME, else PATH.
#
# Java runs from a temp directory, never from inside `crates/`: log4j creates a `logs` folder in
# the working directory, and `crates/logs` breaks every cargo command.

param(
    [string]$Classes17 = $env:V17_CLASSES,
    [string]$Vanilla17 = $env:V17_VANILLA,
    [string]$Jdk = $env:JAVA_HOME
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-Tool([string]$name) {
    if ($Jdk) {
        $tool = Join-Path $Jdk "bin\$name.exe"
        if (Test-Path $tool) { return $tool }
        throw "No $name in $Jdk\bin"
    }
    return (Get-Command $name).Source
}

if (-not $Classes17) {
    $Classes17 = Join-Path $env:TEMP 'mc-rs-v17-oracle\mc17-classes'
}
if (-not (Test-Path (Join-Path $Classes17 'net\minecraft\network\PacketBuffer.class'))) {
    throw "No compiled 1.7.10 classes in $Classes17 (-Classes17; generate-v17-oracle.ps1 builds them)"
}
if (-not $Vanilla17 -or -not (Test-Path (Join-Path $Vanilla17 'libraries'))) {
    throw "Cannot find the 1.7.10 libraries (-Vanilla17 <dir holding libraries\>)"
}

$javac = Find-Tool 'javac'
$java = Find-Tool 'java'

$libs = Get-ChildItem (Join-Path $Vanilla17 'libraries') -Recurse -Filter *.jar |
    Where-Object { $_.Name -notmatch 'natives' } | Sort-Object FullName | ForEach-Object { $_.FullName }
$cp = (@((Resolve-Path $Classes17).Path) + $libs) -join ';'

$work = Join-Path $env:TEMP 'mc-rs-v17-payloads'
$classes = Join-Path $work 'classes'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

Write-Host '== javac =='
& $javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'V17PayloadRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

$out = Join-Path $here 'v17_payload_ref.txt'
Write-Host "== V17PayloadRef -> $out =="
Push-Location $work
try {
    & $java -cp "$classes;$cp" V17PayloadRef $out
    if ($LASTEXITCODE -ne 0) { throw 'V17PayloadRef failed' }
} finally {
    Pop-Location
}
