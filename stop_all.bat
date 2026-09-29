@echo off
title Bubat AI — Stop All Services
cd /d "%~dp0"

echo ====================================================================
echo             BUBAT AI — STOPPING ALL SERVICES & SERVERS
echo ====================================================================
echo.

:: 1. Terminate Ollama server and app
echo [*] Stopping Ollama server...
taskkill /F /T /IM ollama.exe 2>nul
taskkill /F /T /IM "ollama app.exe" 2>nul
powershell -Command "Stop-Process -Name ollama, 'ollama app' -Force -ErrorAction SilentlyContinue"

:: 2. Terminate Python agent processes
echo [*] Stopping Python trading agents...
taskkill /F /FI "WINDOWTITLE eq Bubat AI*" /T 2>nul
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'main.py|chat.py|local_assistant.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo.
echo ====================================================================
echo [OK] All Bubat AI tasks, Python loops, and Ollama servers are OFF.
echo ====================================================================
echo.
timeout /t 3
