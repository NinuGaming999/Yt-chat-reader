@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\\Scripts\\python.exe" call setup.bat
if not exist ".venv\\Scripts\\python.exe" exit /b 1
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
.venv\\Scripts\\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name "NinuGamingLiveAssistant" --add-data "ui.html;." --add-data "obs.html;." --collect-submodules webview app.py
if errorlevel 1 goto :fail
echo.
echo BUILD COMPLETE:
echo dist\\NinuGamingLiveAssistant.exe
echo.
pause
exit /b 0
:fail
echo.
echo BUILD FAILED. Check the error above.
pause
exit /b 1
