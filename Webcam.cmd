@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m classifier.webcam --checkpoint best_model.pth
if errorlevel 1 pause
