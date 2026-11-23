@echo off
rem Launcher, Windows.
rem
rem This artefact ships no Minecraft assets. It locates the *user's own* installation and points the client at it. If there isn't
rem one, it says exactly what it looked for and stops, rather than starting a client that would
rem render magenta placeholders and look like a bug in the port.
setlocal enabledelayedexpansion

set "MCDIR=%MC_RS_MINECRAFT_DIR%"
if "%MCDIR%"=="" set "MCDIR=%APPDATA%\.minecraft"

set "INDEX=%MCDIR%\assets\indexes\1.8.json"
set "JAR=%MCDIR%\versions\1.8.9\1.8.9.jar"

if not exist "%MCDIR%" goto :nodir
if not exist "%INDEX%" goto :noindex
if not exist "%JAR%" goto :nojar

"%~dp0mc-client.exe" ^
  --assetsDir "%MCDIR%\assets" ^
  --assetIndex 1.8 ^
  --versionJar "%JAR%" ^
  %*
exit /b %ERRORLEVEL%

:nodir
echo mc-rs: no Minecraft installation found.
echo   looked in: %MCDIR%
echo.
echo This build deliberately ships no Minecraft assets. Install Minecraft with the official
echo launcher and run 1.8.9 once, or set MC_RS_MINECRAFT_DIR to an existing installation.
exit /b 3

:noindex
echo mc-rs: found %MCDIR% but no asset index for 1.8.
echo   expected: %INDEX%
echo Run 1.8.9 once in the official launcher so it downloads the assets.
exit /b 3

:nojar
echo mc-rs: found %MCDIR% but no 1.8.9 version jar.
echo   expected: %JAR%
echo Install the 1.8.9 profile in the official launcher and start it once.
exit /b 3
