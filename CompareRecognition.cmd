@echo off
title Webcam Recognition Comparison
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The project Python environment is missing.
    echo Expected: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)
echo Opening recognition comparison. No training and no bounding boxes.
echo Focus the video window: 1 original, 2 detector, Tab to switch, Q to quit.
".venv\Scripts\python.exe" -B -m recognition.compare %*
if errorlevel 1 (
    echo.
    echo The webcam application could not start or stopped with an error.
    echo Read the error above. Close other camera apps and check Windows camera permissions.
    pause
    exit /b 1
)
