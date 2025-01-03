@echo off

:: Define virtual environment directory
set VENV_DIR=venv

:: Uninstall existing virtual environment
if not exist %VENV_DIR% (
    echo Virtual environment not found. First install venv via install.bat or install_without_cuda.bat
    pause 
    exit /b
) 

call %VENV_DIR%\Scripts\activate


set PYTHONPATH=%~dp0%
jupyter notebook