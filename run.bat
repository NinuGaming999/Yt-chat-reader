@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\\Scripts\\python.exe" (
  echo First run setup.bat to install the application.
  pause
  exit /b 1
)
.venv\\Scripts\\python.exe app.py
if errorlevel 1 (
  echo.
  echo The app closed with an error. Copy the error above when asking for support.
  pause
)
