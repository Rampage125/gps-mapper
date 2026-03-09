@echo off
echo.
echo   GPS Mapper - Setup
echo   ------------------
echo.

where tesseract >nul 2>&1
if %errorlevel% neq 0 (
    echo   [!] Tesseract not found!
    echo.
    echo   Download and install from:
    echo   https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo   Make sure to check "Add to PATH" during install.
    echo.
    pause
    exit /b 1
)

echo   [OK] Tesseract found
echo   [->] Installing Python packages...
pip install -r requirements.txt -q

echo   [OK] Packages installed
echo.
echo   [->] Starting at http://localhost:5000
echo.

python app.py
pause
