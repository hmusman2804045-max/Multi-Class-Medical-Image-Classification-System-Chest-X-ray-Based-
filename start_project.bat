@echo off
echo ==================================================
echo   Medical Image Classification System - Environment
echo ==================================================
echo Activating virtual environment...
call venv\Scripts\activate
echo Environment activated! (Type 'deactivate' to exit)
echo You can now run: python train.py
cmd /k
