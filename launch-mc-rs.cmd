@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-mc-rs.ps1" %*
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)
