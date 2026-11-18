@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup.bat
if not exist ".venv\Scripts\python.exe" exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --windowed --name NinuGamingLiveAssistant --add-data "ui.html;." --add-data "obs.html;." --add-data "vibevoice_worker.py;." --collect-all webview app.py
if errorlevel 1 goto :fail
set "ISCC="
for %%P in ("C:\Program Files (x86)\Inno Setup 7\ISCC.exe" "C:\Program Files\Inno Setup 7\ISCC.exe") do if exist "%%~P" set "ISCC=%%~P"
if not defined ISCC (
  echo Install Inno Setup 7 to build the installer: https://jrsoftware.org/isdl.php
  goto :fail
)
"%ISCC%" setup.iss
if errorlevel 1 goto :fail
echo.
echo INSTALLER BUILD COMPLETE:
echo installer\NinuGamingLiveAssistant-Setup.exe
pause
exit /b 0
:fail
echo.
echo BUILD FAILED. Review the message above.
pause
exit /b 1