@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
title Basys3 RTL simulations
echo RUNNING> fpga\logs\simulations.status
powershell -NoProfile -ExecutionPolicy Bypass -File fpga\scripts\simulate_core.ps1 > fpga\simulation_core.log 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" goto done
powershell -NoProfile -ExecutionPolicy Bypass -File fpga\scripts\simulate_uart.ps1 > fpga\simulation_uart.log 2>&1
set RC=%ERRORLEVEL%
:done
(echo EXIT=%RC%)> fpga\logs\simulations.status
exit /b %RC%
