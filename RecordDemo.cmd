@echo off
title Basys 3 FPGA box detector - recording view
cd /d "%~dp0"
echo Box detector + activation maps from the FPGA. After the board has been powered off, run ProgramDetectorDemo.cmd first.
echo Close other webcam apps.  R = record MP4   SPACE = slow replay   A = attention glow   H = help   Q = quit
".venv\Scripts\python.exe" -B -m fpga_viz.demo --model detector %*
if errorlevel 1 pause
