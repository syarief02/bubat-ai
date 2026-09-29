@echo off
title Bubat AI — Stop All Services
cd /d "%~dp0"

echo ====================================================================
echo             BUBAT AI — STOPPING ALL SERVICES & SERVERS
echo ====================================================================
echo.

:: 1. Terminate any running python agents
echo [*] Terminating Python agent processes...
taskkill /F /FI "WINDOWTITLE eq Bubat AI*" /T >nul 2>&1
wmic process where "commandline like '%%forex_local_agent%%'" call terminate >nul 2>&1

:: 2. Terminate Ollama server
echo [*] Terminating Ollama server (ollama.exe)...
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM "ollama app.exe" >nul 2>&1

echo.
echo [OK] All Bubat AI services and background Ollama servers have stopped.
echo.
pause
