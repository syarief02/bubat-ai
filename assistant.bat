@echo off
title Bubat AI - Autonomous System Assistant
cd /d "%~dp0"

echo ======================================================================
echo          BUBAT AI - AUTONOMOUS SYSTEM AND CODING AGENT
echo ======================================================================
echo.

:: Start Ollama if it is not already listening on port 11434.
:: (No labels or %%vars%% inside this block: cmd parses a whole block at once, and the old
::  wait loop here made every start abort with "10 was unexpected at this time".)
netstat -ano | findstr 11434 | findstr LISTENING >nul
if %errorlevel% neq 0 (
    echo [*] Starting background Ollama server...
    start "" /B "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >nul 2>&1
    timeout /t 5 /nobreak >nul
)

echo [*] Launching the assistant (first answer can take a minute while the model loads)...
echo.
python local_assistant.py

if %errorlevel% neq 0 (
    echo.
    echo [!] The assistant stopped with error code %errorlevel%.
    pause
)
