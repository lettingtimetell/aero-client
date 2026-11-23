@echo off
rem Aero Client: double-click to play. The real work happens in app\launcher.ps1, hidden.
setlocal
if not exist "%~dp0app\launcher.ps1" goto :notextracted
start "" "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0app\launcher.ps1"
exit /b 0

:notextracted
echo.
echo   Aero Client can't start from inside the zip file.
echo.
echo   1. Close this window.
echo   2. Right-click the zip file and choose "Extract All...".
echo   3. Open the extracted "Aero Client" folder and double-click "Play Aero Client".
echo.
pause
exit /b 1
