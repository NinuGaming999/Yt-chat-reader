@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup.bat
if not exist ".venv\Scripts\python.exe" exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m pip install onnxruntime-directml==1.24.4
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m pip install --no-deps kokoro-onnx==0.6.1
if errorlevel 1 goto :fail
if not exist "models\voices-v1.0.bin" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_models.ps1"
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --windowed --name NinuGamingLiveAssistant --add-data "ui.html;." --add-data "obs.html;." --collect-all webview --collect-all kokoro_onnx --collect-all onnxruntime --collect-all phonemizer --collect-all espeakng_loader --collect-all sounddevice app.py
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