@echo off
title Bubat AI - Health Check
cd /d "%~dp0forex_local_agent"

:: Is the bot WORKING? (not: is it profitable). Read-only; the chat shows the result too.
:: Scheduled runs (health_check.bat scheduled) close by themselves when everything is OK.
python maintenance\health_check.py
set "HC_RESULT=%errorlevel%"

if /i "%~1"=="scheduled" if "%HC_RESULT%"=="0" (
    timeout /t 20
    goto :eof
)
echo Results are saved in forex_local_agent\logs\health.log
pause
