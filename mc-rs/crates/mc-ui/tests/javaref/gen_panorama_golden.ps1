# Regenerate the real-OpenGL reference numbers in `crates/mc-ui/tests/menu_panorama.rs`.
#
#   powershell -ExecutionPolicy Bypass -File crates/mc-ui/tests/javaref/gen_panorama_golden.ps1 [outDir]
#
# Requires a JDK 8 (LWJGL 2.9.4, the 1.8.9 client's GL binding, predates the module system), the
# recompiled vanilla classes in MCP-919-main/bin/minecraft (for net.minecraft.util.MathHelper), and a
# GPU with an OpenGL driver. Renders offscreen into a Pbuffer; no window opens. Prints the grid
# means the Rust test compares against; with `outDir` it also writes the frames as PNGs.

param([string]$OutDir = '')

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$mcp = Join-Path $repo 'MCP-919-main'
$jdk = Get-ChildItem 'C:\Program Files\Eclipse Adoptium' -Directory -Filter 'jdk-8*' |
    Select-Object -First 1
if (-not $jdk) { throw 'No JDK 8 under C:\Program Files\Eclipse Adoptium.' }
$java = Join-Path $jdk.FullName 'bin\java.exe'
$javac = Join-Path $jdk.FullName 'bin\javac.exe'

$lwjgl = Join-Path $mcp 'jars\libraries\org\lwjgl\lwjgl'
$cp = @(
    (Join-Path $lwjgl 'lwjgl\2.9.4-nightly-20150209\lwjgl-2.9.4-nightly-20150209.jar'),
    (Join-Path $lwjgl 'lwjgl_util\2.9.4-nightly-20150209\lwjgl_util-2.9.4-nightly-20150209.jar'),
    (Join-Path $mcp 'bin\minecraft')
) -join ';'
$natives = Join-Path $mcp 'jars\versions\1.8.9\1.8.9-natives'
$jar = Join-Path $mcp 'jars\versions\1.8.9\1.8.9.jar'

$classes = Join-Path ([System.IO.Path]::GetTempPath()) 'mc-ui-panorama-ref'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
if ($OutDir) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }
& $javac -nowarn -encoding UTF-8 -cp $cp -d $classes (Join-Path $here 'PanoramaRef.java')
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

# timer, partial ticks, GUI width, GUI height, display width, display height (GUI scale 2).
foreach ($frame in @(
        @('100', '0.0', '427', '240', '854', '480'),
        @('1234', '0.37', '427', '240', '854', '480'),
        @('100', '0.0', '320', '240', '640', '480'))) {
    $javaArgs = @("-Djava.library.path=$natives", '-cp', "$classes;$cp", 'PanoramaRef', $jar) + $frame
    if ($OutDir) { $javaArgs += $OutDir }
    Write-Host "== timer $($frame[0]) + $($frame[1]), $($frame[4])x$($frame[5]) =="
    & $java @javaArgs
    if ($LASTEXITCODE -ne 0) { throw 'PanoramaRef failed' }
}
