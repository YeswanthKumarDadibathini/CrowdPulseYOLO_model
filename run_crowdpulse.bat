@echo off
title CrowdPulse - Crowd Density Management System
color 0B
cd /d "%~dp0"

echo ==============================================================================
echo                 CROWDPULSE - CROWD DENSITY MANAGEMENT SYSTEM
echo                Autonomous Multi-Agent AI Monitoring & Dispatch
echo ==============================================================================
echo.
echo [1/3] Verifying environment...

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please ensure the project setup is complete before launching.
    echo Path checked: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)

echo [2/3] Activating virtual environment (.venv)...
call .venv\Scripts\activate.bat

echo [3/3] Launching CrowdPulse Dashboard...
echo       Local URL: http://localhost:8501
echo       Press Ctrl+C in this console to terminate the server.
echo.

streamlit run app.py --server.port 8501 --server.headless false

if errorlevel 1 (
    echo.
    echo [ERROR] CrowdPulse exited with an error code.
    pause
)
