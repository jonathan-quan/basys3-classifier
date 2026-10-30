@echo off
cd /d "%~dp0.."
set PYTHONPATH=
set PYTHONHOME=
title Live box detector (laptop software)
echo Laptop software detector - not FPGA. Close other webcam apps first. Q quits.
.venv\Scripts\python.exe -B -m detection.webcam_live %*
if errorlevel 1 pause
