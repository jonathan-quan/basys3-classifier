@echo off
title Basys 3 FPGA classifier - recording view
cd /d "%~dp0"
echo 4-class classifier with activation maps. Run ProgramVizFPGA.cmd first (it replaces the detector bitstream).
echo Close other webcam apps.  R = record MP4   SPACE = slow replay   F = framing   H = help   Q = quit
".venv\Scripts\python.exe" -B -m fpga_viz.demo --model classifier %*
if errorlevel 1 pause
