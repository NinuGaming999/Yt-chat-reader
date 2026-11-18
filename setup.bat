@echo off
setlocal
cd /d "%~dp0"
echo ======================================================
echo   NinuGaming Live Assistant - Source Developer Setup
echo ======================================================
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3.12") else (
  where python >nul 2>nul
  if %errorlevel%==0 (set "PY=python") else (
    echo Python 3.12 x64 was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/ and enable Add Python to PATH.
    pause
    exit /b 1
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
echo Installing desktop dependencies. CUDA speech installs on first launch...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo.
echo Setup complete. Double-click run.bat to open the desktop app.
pause
exit /b 0
:fail
echo.
echo Setup failed. Check the error above.
pause
exit /b 1