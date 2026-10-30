@echo off
cd /d "%~dp0.."
set PYTHONPATH=
set PYTHONHOME=
title Webcam box FIX pass - close this window when finished
.venv\Scripts\python.exe -B -m detection.webcam_review --fix
pause
