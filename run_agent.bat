@echo off
title Bubat AI - Autonomous Forex Trading Agent
cd /d "%~dp0forex_local_agent"

echo ====================================================================
echo             BUBAT AI - LOCAL AUTONOMOUS FOREX AGENT
echo ====================================================================
echo.

:: 1. Check if Ollama server is running on port 11434
echo [*] Checking Ollama server status...
netstat -ano | findstr 11434 | findstr LISTENING >nul
if %errorlevel% neq 0 (
    echo [!] Ollama is not running. Starting background Ollama server...
    start /b "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >nul 2>&1
    timeout /t 3 /nobreak >nul
) else (
    echo [OK] Ollama server is active and listening on port 11434.
)

:: 2. Launch the Master Agent
echo [*] Launching Master Trading Agent (main.py)...
echo [i] Press Ctrl+C in this window at any time to safely stop trading.
echo.
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [!] The agent terminated with an error code: %errorlevel%
    pause
)
