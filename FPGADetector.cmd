@echo off
title Basys 3 FPGA Box Detector
cd /d "%~dp0"
echo Webcam capture and display run on the laptop. The box detector runs on the FPGA.
echo Load the detector bitstream first (ProgramDetector.cmd). Close other webcam apps. Q quits.
".venv\Scripts\python.exe" -B -m fpga_det.webcam %*
if errorlevel 1 pause
