# S05 - regenerate the bit-exact Java reference fixtures for `mc-math`.
#
#   powershell -ExecutionPolicy Bypass -File generate.ps1
#
# Requires a JDK on PATH (developed against Temurin 21). Writes:
#   ../src/tables/*.bin      - the SIN_TABLE and atan2 tables embedded in the crate
#   ../tests/fixtures/*.bin  - ~1e6 (input, output) records for the differential test
#
# Everything is raw IEEE-754 bit patterns, little-endian. Never compare decimal strings.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $here "out"

New-Item -ItemType Directory -Force -Path $out | Out-Null
& javac -d $out (Join-Path $here "MathRefGen.java")
if ($LASTEXITCODE -ne 0) { throw "javac failed" }

$tables = Join-Path (Split-Path -Parent $here) "src/tables"
$fixtures = Join-Path (Split-Path -Parent $here) "tests/fixtures"

& java -cp $out MathRefGen $tables $fixtures
if ($LASTEXITCODE -ne 0) { throw "MathRefGen failed" }
