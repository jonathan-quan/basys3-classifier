@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe -B -m detection.webcam --checkpoint detection\best_detector.pth
pause
