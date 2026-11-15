@echo off
setlocal enabledelayedexpansion
title FAST v5.0-Research - 1-Click Privacy Anonymizer & Empirical Audit
color 0B

echo.
echo ================================================================================
echo       FAST v5.0-Research -- 1-Click Universal Privacy Anonymizer
echo       Elliptical Blur | Energy Telemetry | PAG Ground Truth | HITL Audit
echo ================================================================================
echo.

set "PY_EXE="
if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\venvs\blur_pipeline\Scripts\python.exe" (
    set "PY_EXE=%USERPROFILE%\venvs\blur_pipeline\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Python 3.10+ was not found on your system!
        echo Please install Python from: https://www.python.org/downloads/
        echo *IMPORTANT*: Check "Add python.exe to PATH" during installation.
        pause
        exit /b 1
    )
    set "PY_EXE=python"
)

if not exist "inputs\" mkdir inputs
if not exist "outputs\" mkdir outputs

set "HAS_FILES=0"
for /f "delims=" %%A in ('dir /b /a-d "inputs\*" 2^>nul') do set "HAS_FILES=1"
for /f "delims=" %%A in ('dir /b /ad "inputs\*" 2^>nul') do set "HAS_FILES=1"

if "%HAS_FILES%"=="0" (
    echo.
    echo [!] Notice: The 'inputs' folder is currently empty.
    echo ------------------------------------------------------------------------
    echo Please copy or drag-and-drop your camera frames into:
    echo     %CD%\inputs
    echo ------------------------------------------------------------------------
    echo.
    echo Press any key AFTER you have copied your files to begin anonymization...
    pause >nul
)

echo.
echo [*] Starting FAST v5.0-Research Master Pipeline...
echo [*] Python:  !PY_EXE!
echo [*] Inputs:  %CD%\inputs
echo [*] Outputs: %CD%\outputs
echo.

!PY_EXE! run_blur_v5.py --input-dir inputs --output-dir outputs --session-id R001 --source real_car

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================================================
    echo [SUCCESS] Anonymization and Research Audit complete!
    echo Results, telemetry, and HITL review packs are saved in:
    echo %CD%\outputs
    echo ================================================================================
    echo.
    echo Opening 'outputs' folder in File Explorer...
    start "" explorer "outputs"
) else (
    echo.
    echo [!] Pipeline exited with code %ERRORLEVEL%. Please check messages above.
)

echo.
echo Press any key to exit...
pause >nul
