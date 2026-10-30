@echo off
cd /d "%~dp0.."
set PYTHONPATH=
set PYTHONHOME=
title Webcam box review - close this window when finished
.venv\Scripts\python.exe -B -m detection.webcam_review
pause
