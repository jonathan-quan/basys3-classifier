@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe -B -m detection.webcam --checkpoint detection\runs\source_box_pilot\best_detector.pth --allow-pilot
pause
