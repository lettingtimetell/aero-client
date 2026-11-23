# Build the Windows artefact.
#
#   pwsh -File packaging/windows/build.ps1 [-Out <dir>] [-SkipBuild]
#
# Produces `<out>/mc-rs-<version>-windows-x86_64/` and a zip beside it. The payload carries this
# project's own binary, its licence files and a launcher; it carries **no Minecraft assets** —
# see `scripts/verify-payload.ps1`, which is run before the zip is made
# and fails the build rather than warning.

[CmdletBinding()]
param(
    [string] $Out = "",
    [switch] $SkipBuild
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Resolve-Path (Join-Path $here "..\..")
Import-Module (Join-Path $here "..\scripts\common.psm1") -Force

$version = Get-WorkspaceVersion -Workspace $workspace
$triple = "windows-x86_64"
$name = "mc-rs-$version-$triple"
if ($Out -eq "") { $Out = Join-Path $workspace "target\package" }
$stage = Join-Path $Out $name

if (-not $SkipBuild) {
    Write-Host "building $name (profile: dist, default features off)"
    Push-Location $workspace
    try {
        # `offline` is on by default and is dev-only (a
        # release build must turn it off). `--no-default-features` is the whole of that fix, and
        # the payload verifier re-checks it against the built bytes.
        & cargo build --profile dist -p mc-client --no-default-features
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    } finally { Pop-Location }
}

$binary = Join-Path $workspace "target\dist\mc-client.exe"
if (-not (Test-Path -LiteralPath $binary)) { throw "built binary not found: $binary" }

New-StagedPayload -Stage $stage
Copy-Item -LiteralPath $binary -Destination (Join-Path $stage "mc-client.exe")
Copy-Item -LiteralPath (Join-Path $here "mc-rs.cmd") -Destination (Join-Path $stage "mc-rs.cmd")
Write-PayloadDocs -Stage $stage -Workspace $workspace -Version $version -Triple $triple `
    -RunLine "Double-click mc-rs.cmd, or run it from a terminal with any vanilla client option."
Write-BuildInfo -Stage $stage -Version $version -Triple $triple -Profile "dist" `
    -Binary (Join-Path $stage "mc-client.exe") -DefaultFeatures "off"

& (Join-Path $here "..\scripts\verify-payload.ps1") -Payload $stage -ExpectBinary "mc-client.exe"
if ($LASTEXITCODE -ne 0) { throw "payload verification failed" }

$zip = Join-Path $Out "$name.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
$size = (Get-Item -LiteralPath $zip).Length

Write-Host ""
Write-Host "artefact: $zip"
Write-Host ("size:     {0:N1} MiB" -f ($size / 1MB))
Write-Host "payload:  $stage"
