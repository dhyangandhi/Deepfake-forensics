@echo off
title DFX Frontend Dashboard (Port 3000)
cd /d "%~dp0frontend\frontend"
echo ========================================================
echo Starting DFX Forensics AI Frontend Dashboard (Port 3000)...
echo ========================================================
npm.cmd run dev
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Frontend encountered an error or stopped.
    pause
)

