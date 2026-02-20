@echo off
echo Starting PromoNex Shopify API...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo Installing dependencies...
    pip install -r requirements.txt
)


REM Start the server
echo Starting API server...
echo API will be available at: http://localhost:8050
echo.
python -m app.main

pause 