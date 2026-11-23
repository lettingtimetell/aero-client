# The Java side of the chat-link syntax reference: which `open_url` values `new URI(value)`
# accepts (`GuiScreen.handleComponentClick` opens nothing when it throws).
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-client/javaref/generate-link-uri.ps1
#   cargo test -p mc-client --lib -- external_links
#
# Writes crates/mc-client/tests/fixtures/link_uri_ref.tsv. The test there checks that every value
# the client refuses, Java refuses too, so the client never refuses a link vanilla opens. Needs a
# Java 8 JDK (the runtime 1.8.9 runs on): set JAVA8_HOME, or have javac 1.8 on PATH. Java runs
# from a temp directory, never from inside `crates/`.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$fixture = Join-Path $here '..\tests\fixtures\link_uri_ref.tsv'

$javac = 'javac'
$java = 'java'
if ($env:JAVA8_HOME) {
    $javac = Join-Path $env:JAVA8_HOME 'bin\javac.exe'
    $java = Join-Path $env:JAVA8_HOME 'bin\java.exe'
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ('link-uri-ref-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $work | Out-Null
try {
    & $javac -encoding UTF-8 -d $work (Join-Path $here 'LinkUriRef.java')
    if ($LASTEXITCODE -ne 0) { throw 'javac failed' }
    Push-Location $work
    try {
        & $java -cp $work LinkUriRef (Join-Path $work 'link_uri_ref.tsv')
        if ($LASTEXITCODE -ne 0) { throw 'LinkUriRef failed' }
    } finally {
        Pop-Location
    }
    Copy-Item (Join-Path $work 'link_uri_ref.tsv') $fixture -Force
    Write-Output ("wrote " + (Get-Content $fixture).Count + " cases to " + (Resolve-Path $fixture))
} finally {
    Remove-Item -Recurse -Force $work
}
