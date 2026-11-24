#requires -Version 5.1
<#
.SYNOPSIS
Make the installer's pictures from the logo: the program icon and the wizard images.

.DESCRIPTION
Reads the logo renders in mc-rs\crates\mc-client\assets\icon (aero-<size>.png, each drawn at its
own size from aero-client-logo.svg) and writes into -OutDir:

  aero-client.ico            16, 20, 24, 32, 40, 48, 64, 128 and 256 px, each a PNG entry copied
                             byte for byte from its render. package-aero-client.ps1 puts it in
                             app\ of the zip and the installer: the shortcuts, the uninstall entry
                             in Settings > Apps, Setup itself and the uninstaller use it.
  wizard-small-<n>.png       the logo for the top right corner of the wizard, at the sizes that
                             corner has at 100 % to 250 % display scaling
  wizard-large-<w>x<h>.png   the logo in the middle of the left panel of the last page,
                             transparent around it, at that panel's sizes

Nothing here needs more than Windows PowerShell 5.1: the icon is packed by hand, and the wizard
images are scaled with System.Drawing. The PNG files never go into the zip (its guard,
verify-payload.ps1, refuses .png files); only aero-client.ico does.

.PARAMETER OutDir
The folder to write to (created if needed).

.PARAMETER IconDir
The folder with aero-<size>.png. Default: mc-rs\crates\mc-client\assets\icon.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$IconDir = ''
)

$ErrorActionPreference = 'Stop'
if ($IconDir -eq '') {
    $IconDir = Join-Path $PSScriptRoot '..\..\..\crates\mc-client\assets\icon'
}
$IconDir = (Resolve-Path -LiteralPath $IconDir).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

# The icon's sizes. 16 to 64 px are what Windows asks for at 100 % to 200 % display scaling (lists,
# shortcuts, Start, the taskbar), 128 and 256 px the large icon views and Start at high scaling.
$iconSizes = @(16, 20, 24, 32, 40, 48, 64, 128, 256)
# The wizard's top right picture area at 100, 125, 150, 175, 200, 225 and 250 % scaling (square).
# Setup picks the file that fits its area best.
$smallSizes = @(58, 77, 97, 116, 124, 143, 159)
# The last page's left panel at the same scalings (164:314).
$largeSizes = @(@(202, 386), @(269, 515), @(336, 643), @(403, 772), @(430, 824), @(498, 953), @(534, 1022))

# A logo render: a PNG file of exactly Side x Side pixels, 8-bit RGBA (what a PNG icon entry must
# be). Returns its bytes.
function Read-LogoPng([int]$Side) {
    $path = Join-Path $IconDir "aero-$Side.png"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "missing logo render: $path" }
    $bytes = [IO.File]::ReadAllBytes($path)
    $signature = [byte[]](0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
    if ($bytes.Length -lt 33) { throw "not a PNG file: $path" }
    for ($i = 0; $i -lt 8; $i++) {
        if ($bytes[$i] -ne $signature[$i]) { throw "not a PNG file: $path" }
    }
    if ([Text.Encoding]::ASCII.GetString($bytes, 12, 4) -ne 'IHDR') { throw "not a PNG file: $path" }
    $width = ([int]$bytes[16] -shl 24) -bor ([int]$bytes[17] -shl 16) -bor ([int]$bytes[18] -shl 8) -bor [int]$bytes[19]
    $height = ([int]$bytes[20] -shl 24) -bor ([int]$bytes[21] -shl 16) -bor ([int]$bytes[22] -shl 8) -bor [int]$bytes[23]
    if ($width -ne $Side -or $height -ne $Side) { throw "$path is $width x $height, not $Side x $Side" }
    if ($bytes[24] -ne 8 -or $bytes[25] -ne 6) { throw "$path is not 8-bit RGBA" }
    return , $bytes
}

# ---------------------------------------------------------------- the icon
# An .ico file: a 6-byte header (0, type 1 = icon, the number of pictures), one 16-byte entry per
# picture (width and height, 0 meaning 256; no palette; 1 plane; 32 bits per pixel; the picture's
# length and where it starts), then the pictures. Windows Vista and later read PNG pictures.
$pictures = @($iconSizes | ForEach-Object { , (Read-LogoPng $_) })
$stream = New-Object IO.MemoryStream
$writer = New-Object IO.BinaryWriter $stream
$writer.Write([uint16]0)
$writer.Write([uint16]1)
$writer.Write([uint16]$iconSizes.Count)
$offset = 6 + 16 * $iconSizes.Count
for ($i = 0; $i -lt $iconSizes.Count; $i++) {
    $side = $iconSizes[$i]
    $dimension = if ($side -ge 256) { 0 } else { $side }
    $writer.Write([byte]$dimension)
    $writer.Write([byte]$dimension)
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]32)
    $writer.Write([uint32]$pictures[$i].Length)
    $writer.Write([uint32]$offset)
    $offset += $pictures[$i].Length
}
foreach ($picture in $pictures) { $writer.Write($picture) }
$writer.Flush()
$icon = Join-Path $OutDir 'aero-client.ico'
[IO.File]::WriteAllBytes($icon, $stream.ToArray())
$writer.Dispose()
Write-Host ("wrote {0} ({1} sizes, {2:N0} bytes)" -f $icon, $iconSizes.Count, (Get-Item -LiteralPath $icon).Length)

# ---------------------------------------------------------------- the wizard images
Add-Type -AssemblyName System.Drawing

# Draw the logo render $Source scaled to $Side x $Side at ($X, $Y) on a new transparent
# $Width x $Height picture and save it as PNG.
function Save-Scaled([System.Drawing.Image]$Source, [int]$Width, [int]$Height, [int]$Side, [string]$Path) {
    $bitmap = New-Object System.Drawing.Bitmap($Width, $Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $attributes = New-Object System.Drawing.Imaging.ImageAttributes
    try {
        $graphics.Clear([System.Drawing.Color]::Transparent)
        $graphics.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceCopy
        $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        # Mirrored edges, so the scaler blends no transparent border into the logo's outline.
        $attributes.SetWrapMode([System.Drawing.Drawing2D.WrapMode]::TileFlipXY)
        # A little above the middle of a tall panel, where the eye rests.
        $x = [int](($Width - $Side) / 2)
        $y = if ($Height -gt $Width) { [int](($Height - $Side) * 0.42) } else { [int](($Height - $Side) / 2) }
        $target = New-Object System.Drawing.Rectangle($x, $y, $Side, $Side)
        $graphics.DrawImage($Source, $target, 0, 0, $Source.Width, $Source.Height,
            [System.Drawing.GraphicsUnit]::Pixel, $attributes)
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $attributes.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

# Scale down from the smallest render at least as big as the target, never up.
function Get-Source([int]$Side) {
    $from = @($iconSizes | Where-Object { $_ -ge $Side } | Select-Object -First 1)
    $pick = if ($from.Count -gt 0) { $from[0] } else { $iconSizes[-1] }
    $memory = New-Object IO.MemoryStream(, (Read-LogoPng $pick))
    return [System.Drawing.Image]::FromStream($memory)
}

foreach ($side in $smallSizes) {
    $source = Get-Source $side
    try {
        Save-Scaled $source $side $side $side (Join-Path $OutDir "wizard-small-$side.png")
    } finally {
        $source.Dispose()
    }
}
foreach ($size in $largeSizes) {
    $width = $size[0]
    $height = $size[1]
    # The logo at 45 % of the panel's width: 91 px at 100 %, 240 px at 250 %.
    $side = [int][Math]::Round($width * 0.45)
    $source = Get-Source $side
    try {
        Save-Scaled $source $width $height $side (Join-Path $OutDir "wizard-large-${width}x$height.png")
    } finally {
        $source.Dispose()
    }
}
Write-Host ("wrote {0} small and {1} large wizard images into {2}" -f $smallSizes.Count, $largeSizes.Count, $OutDir)
