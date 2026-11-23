# Tests for the payload guard.
#
# A guard that has never been shown to reject anything is not a guard. This plants each class of
# forbidden file into a fixture payload and asserts `verify-payload.ps1` fails on it, then asserts
# it passes on a clean payload — the control, without which "it exited 0" would prove nothing.
#
#   powershell -ExecutionPolicy Bypass -File packaging/scripts/verify-payload.tests.ps1
# Exit code 0 = all cases behaved as expected.

param([string] $Payload = "")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$verify = Join-Path $here "verify-payload.ps1"
$root = Join-Path ([System.IO.Path]::GetTempPath()) ("mc-rs-payload-tests-" + [System.Guid]::NewGuid().ToString("N"))
$failures = 0

function New-Payload {
    param([string[]] $Files)
    $dir = Join-Path $root ([System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    foreach ($relative in $Files) {
        $path = Join-Path $dir $relative
        $parent = Split-Path -Parent $path
        if (-not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Set-Content -LiteralPath $path -Value "fixture" -Encoding ascii
    }
    return $dir
}

function Check {
    param([string] $Name, [string[]] $Files, [int] $Expected)
    $dir = New-Payload -Files $Files
    & $verify -Payload $dir -Quiet | Out-Null
    $code = $LASTEXITCODE
    if ($code -eq $Expected) {
        Write-Host "  ok    $Name (exit $code)"
    } else {
        Write-Host "  FAIL  $Name (exit $code, expected $Expected)"
        $script:failures++
    }
}

Write-Host "verify-payload.ps1 cases:"

# The control. Everything below only means something because this one passes.
Check -Name "a clean payload is accepted" -Expected 0 -Files @(
    "mc-client.exe", "mc-rs.cmd", "README.txt", "NOTICE.txt"
)

Check -Name "the version jar is rejected" -Expected 1 -Files @("mc-client.exe", "1.8.9.jar")
Check -Name "a stray texture is rejected" -Expected 1 -Files @("mc-client.exe", "stone.png")
Check -Name "a bundled Minecraft icon is rejected" -Expected 1 -Files @("mc-client.exe", "minecraft.icns")
Check -Name "a sound file is rejected" -Expected 1 -Files @("mc-client.exe", "click.ogg")
Check -Name "a language file is rejected" -Expected 1 -Files @("mc-client.exe", "en_US.lang")
Check -Name "an extracted assets tree is rejected" -Expected 1 -Files @(
    "mc-client.exe", "assets/indexes/1.8.json"
)
Check -Name "a bundled versions dir is rejected" -Expected 1 -Files @(
    "mc-client.exe", "versions/1.8.9/whatever.txt"
)
Check -Name "a models dir is rejected even without assets/" -Expected 1 -Files @(
    "mc-client.exe", "models/block/stone.json"
)
Check -Name "servers.dat is rejected" -Expected 1 -Files @("mc-client.exe", "servers.dat")
Check -Name "a nested asset is found, not just a top-level one" -Expected 1 -Files @(
    "mc-client.exe", "data/inner/deeper/grass.png"
)
Check -Name "an empty payload is a usage error" -Expected 2 -Files @()

# A payload that looks clean but whose expected binary is missing is a usage error, not a pass:
# `-ExpectBinary` exists so a build script cannot ship an empty directory and call it verified.
$dir = New-Payload -Files @("README.txt")
& $verify -Payload $dir -ExpectBinary "mc-client.exe" -Quiet | Out-Null
if ($LASTEXITCODE -eq 2) {
    Write-Host "  ok    a missing expected binary is a usage error (exit 2)"
} else {
    Write-Host "  FAIL  a missing expected binary gave exit $LASTEXITCODE, expected 2"
    $failures++
}

# The BUILD-INFO checks. Each case is a real staged payload with a real binary and a real hash,
# so the "clean" case is a genuine control for the three rejections around it.
function New-BinaryPayload {
    param([string] $DefaultFeatures = "off", [switch] $NoInfo, [switch] $WrongHash)
    $dir = New-Payload -Files @("README.txt", "NOTICE.txt")
    $exe = Join-Path $dir "mc-client.exe"
    # Any file will do: the check is over its hash, not its contents.
    Set-Content -LiteralPath $exe -Value "MZ pretend binary" -Encoding ascii
    if (-not $NoInfo) {
        $hash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($WrongHash) { $hash = "0" * 64 }
        Set-Content -LiteralPath (Join-Path $dir "BUILD-INFO.txt") -Encoding utf8 -Value @(
            "version: 0.1.0",
            "triple: test",
            "profile: dist",
            "default-features: $DefaultFeatures",
            "binary: mc-client.exe",
            "binary-sha256: $hash"
        )
    }
    return $dir
}

function CheckBinary {
    param([string] $Name, [string] $Dir, [int] $Expected)
    & $verify -Payload $Dir -ExpectBinary "mc-client.exe" -Quiet | Out-Null
    if ($LASTEXITCODE -eq $Expected) {
        Write-Host "  ok    $Name (exit $LASTEXITCODE)"
    } else {
        Write-Host "  FAIL  $Name (exit $LASTEXITCODE, expected $Expected)"
        $script:failures++
    }
}

CheckBinary -Name "a correctly built payload passes" -Expected 0 -Dir (New-BinaryPayload)
CheckBinary -Name "default features left on is rejected" -Expected 1 `
    -Dir (New-BinaryPayload -DefaultFeatures "on")
CheckBinary -Name "no BUILD-INFO.txt is rejected" -Expected 1 -Dir (New-BinaryPayload -NoInfo)
CheckBinary -Name "a manifest describing another binary is rejected" -Expected 1 `
    -Dir (New-BinaryPayload -WrongHash)

# And the real artefact this machine produced, if it is there.
$workspace = Resolve-Path (Join-Path $here "..\..")
$real = if ($Payload -ne "") { $Payload } else {
    Join-Path $workspace "target\package\mc-rs-0.1.0-windows-x86_64"
}
if (Test-Path -LiteralPath $real) {
    CheckBinary -Name "the real staged Windows payload passes" -Expected 0 -Dir $real
} else {
    Write-Host "  skip  real payload: run packaging/windows/build.ps1 first"
}

$resolvedRoot = [System.IO.Path]::GetFullPath($root)
$resolvedTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/')
if (-not $resolvedRoot.StartsWith($resolvedTemp + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path -Leaf $resolvedRoot) -notmatch '^mc-rs-payload-tests-[0-9a-f]{32}$') {
    throw "Refusing to remove an unexpected test directory: $resolvedRoot"
}
Remove-Item -LiteralPath $resolvedRoot -Recurse -Force -ErrorAction SilentlyContinue
if ($failures -gt 0) {
    Write-Host ""
    Write-Host "$failures case(s) failed"
    exit 1
}
Write-Host ""
Write-Host "all cases behaved as expected"
exit 0
