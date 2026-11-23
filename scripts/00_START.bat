@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0.."
echo =====================================================
echo AI Slop Detector: setup and launch
 echo =====================================================
if not exist ".venv-local\Scripts\python.exe" (
  echo Creating Python environment...
  py -3 -m venv ".venv-local" 2>nul
  if errorlevel 1 python -m venv ".venv-local"
  if errorlevel 1 goto python_fail
)
".venv-local\Scripts\python.exe" -m pip install -r requirements-local-lock.txt
if errorlevel 1 goto pip_fail
".venv-local\Scripts\python.exe" quick_start.py
if errorlevel 1 (
  echo.
  echo See error above. Close this window after checking it.
)
goto finish
:python_fail
echo ERROR: Python 3 not found or virtual environment could not be created.
echo Install Python 3.13 and enable PATH, then rerun 00_START.bat.
goto finish
:pip_fail
echo ERROR: Dependency installation failed. Check internet access and pip.
:finish
pause
