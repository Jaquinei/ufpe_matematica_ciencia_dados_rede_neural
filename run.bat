@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "MIN_PYTHON_VERSION=3.10"

if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_BIN=%SCRIPT_DIR%venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_BIN=python"
    ) else (
        echo Python is not installed
        pause
        exit /b 1
    )
)

"%PYTHON_BIN%" -c "import sys; sys.exit(0 if sys.version_info >= tuple(map(int, '%MIN_PYTHON_VERSION%'.split('.'))) else 1)"
if errorlevel 1 (
    echo Erro: e necessario Python ^>= %MIN_PYTHON_VERSION% ^(veja o arquivo .python-version^).
    pause
    exit /b 1
)

"%PYTHON_BIN%" src\main.py
pause
