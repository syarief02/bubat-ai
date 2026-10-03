@echo off
title Bubat AI - Interactive Intelligence Chat
cd /d "%~dp0forex_local_agent"

echo ====================================================================
echo             BUBAT AI - INTERACTIVE AI AGENT CHAT
echo ====================================================================
echo.

:: Check if Ollama server is running on port 11434
netstat -ano | findstr 11434 | findstr LISTENING >nul
if %errorlevel% neq 0 (
    echo [*] Starting background Ollama server...
    powershell -NoProfile -Command "Start-Process -FilePath ($env:LOCALAPPDATA + '\Programs\Ollama\ollama.exe') -ArgumentList serve -WindowStyle Hidden"
    timeout /t 3 /nobreak >nul
)

:: Run the interactive chat CLI
python chat.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Chat ended with an issue.
    pause
)
