@echo off
title DFX Backend API Server (Port 8000)
cd /d "%~dp0backend"
echo ========================================================
echo Starting DFX Forensics AI Backend Server on Port 8000...
echo ========================================================
ai-env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Backend encountered an error or stopped.
    pause
)

