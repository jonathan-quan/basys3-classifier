@echo off
setlocal
rem Vivado-launched shells carry its own Python settings; use the project venv cleanly.
set PYTHONPATH=
set PYTHONHOME=
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
title Basys3 hardware benchmark
echo RUNNING> fpga\logs\benchmark.status
.venv\Scripts\python.exe -B -m fpga.benchmark > fpga\benchmark.log 2>&1
set RC=%ERRORLEVEL%
(echo EXIT=%RC%)> fpga\logs\benchmark.status
exit /b %RC%
