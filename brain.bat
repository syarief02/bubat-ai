@echo off
title Bubat Brain
cd /d "%~dp0forex_local_agent"

:: No arguments: show the latest assessment and the proposals waiting for you.
:: Otherwise pass through, e.g.  brain.bat approve P3   /   brain.bat reject P4 too risky   /   brain.bat run --force
if "%~1"=="" (
    python -m brain status
    echo.
    echo Commands: brain.bat run ^| list ^| approve P3 ^| reject P3 [reason] ^| journal
    pause
) else (
    python -m brain %*
)
