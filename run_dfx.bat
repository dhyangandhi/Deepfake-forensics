@echo off
title Deepfake Forensics AI (DFX) Launcher
echo ========================================================
echo Launching Deepfake Forensics AI (DFX)...
echo ========================================================

echo Starting Backend Server on http://localhost:8000 ...
start "DFX Backend" cmd /c "%~dp0start_backend.bat"

echo Starting Frontend Dashboard on http://localhost:3000 ...
start "DFX Frontend" cmd /c "%~dp0start_frontend.bat"

echo Waiting 4 seconds for servers to initialize...
timeout /t 4 /nobreak >nul

echo Opening browser at http://localhost:3000 ...
start http://localhost:3000

echo ========================================================
echo DFX is now running! Keep the two opened windows active.
echo ========================================================

