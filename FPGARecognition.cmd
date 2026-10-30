@echo off
title Basys 3 FPGA Recognition
cd /d "%~dp0"
echo Webcam capture and display run on the laptop. Neural inference runs on the FPGA.
echo Close other webcam applications first. Q quits; F changes framing.
".venv\Scripts\python.exe" -B -m fpga.webcam %*
if errorlevel 1 pause
