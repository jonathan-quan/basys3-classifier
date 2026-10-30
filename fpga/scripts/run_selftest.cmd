@echo off
setlocal
rem Vivado-launched shells carry its own Python settings; use the project venv cleanly.
set PYTHONPATH=
set PYTHONHOME=
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
title Basys3 hardware self-test
echo RUNNING> fpga\logs\selftest.status
.venv\Scripts\python.exe -B -m fpga.client --self-test > fpga\selftest.log 2>&1
set RC=%ERRORLEVEL%
(echo EXIT=%RC%)> fpga\logs\selftest.status
exit /b %RC%
