@echo off

:: Define virtual environment directory
set VENV_DIR=test

:: Uninstall existing virtual environment
if exist %VENV_DIR% (
    echo Existing virtual environment found. Deleting...
    rmdir /s /q %VENV_DIR%
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to delete the virtual environment. Ensure no files are in use.
        pause
        exit /b
    )
    echo Virtual environment deleted successfully.
) else (
    echo No existing virtual environment found.
)

setlocal enabledelayedexpansion
for /f "tokens=*" %%i in ('where python') do (
    "%%i" --version 2>nul | findstr "3.12" >nul
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_EXEC=%%i"
        goto FoundPython
    )
)

echo Python 3.12 is not installed or not found in PATH.
pause
exit /b

:FoundPython
echo Using Python 3.12 at "%PYTHON_EXEC%".
 
:: Create virtual environment using Python 3.12
%PYTHON_EXEC% -m venv %VENV_DIR%

:: Activate virtual environment
call %VENV_DIR%\Scripts\activate

:: Install PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

echo Setup complete.
pause