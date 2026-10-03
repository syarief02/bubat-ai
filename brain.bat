@echo off
title Bubat Brain
cd /d "%~dp0forex_local_agent"

:: With arguments, run one command and exit, e.g.  brain.bat approve P5   /   brain.bat run --force
if not "%~1"=="" (
    python -m brain %*
    goto :eof
)

:: Double-clicked: show the inbox, then keep taking commands until Enter on an empty line
python -m brain status

:menu
echo.
echo ---------------------------------------------------------------------
echo  Type a command, or just press Enter to close:
echo    approve P5          apply a proposal (tests run first, rollback on failure)
echo    reject P3 reason    reject a proposal; the reason is saved for the brain
echo    list                proposals waiting for you      status   show the inbox again
echo    journal             what the brain did recently    run --force   run the brain now (~10-40 min)
echo ---------------------------------------------------------------------
set "BRAIN_CMD="
set /p "BRAIN_CMD=brain> "
if not defined BRAIN_CMD goto :eof
if /i "%BRAIN_CMD%"=="exit" goto :eof
if /i "%BRAIN_CMD%"=="quit" goto :eof
python -m brain %BRAIN_CMD%
goto menu
