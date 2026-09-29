@echo off
title Bubat AI - Local Autonomous System Assistant
cls

echo ======================================================================
echo          BUBAT AI - LOCAL AUTONOMOUS SYSTEM & CODING AGENT
echo ======================================================================
echo.

:: 1. Check if Ollama server is already running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Ollama server is not running. Starting background server...
    start /b "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >nul 2>&1
    
    :: Wait up to 10 seconds for Ollama to start
    set /a count=0
    :wait_ollama
    timeout /t 1 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% EQU 0 goto ollama_ready
    set /a count+=1
    if %count% LSS 10 goto wait_ollama
    echo [WARNING] Ollama server did not respond quickly. Launching agent anyway...
)

:ollama_ready
echo [INFO] Ollama server is ready.
echo [INFO] Launching Bubat Autonomous System Assistant...
echo.

cd /d "%~dp0"
python local_assistant.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Assistant exited with code %ERRORLEVEL%.
)

echo.
pause

