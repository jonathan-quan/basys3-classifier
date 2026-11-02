@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga_viz\logs mkdir fpga_viz\logs
echo RUNNING> fpga_viz\logs\simulations.status
powershell -NoProfile -ExecutionPolicy Bypass -File fpga_viz\scripts\simulate_core.ps1 > fpga_viz\logs\simulation_core.log 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" goto done
powershell -NoProfile -ExecutionPolicy Bypass -File fpga_viz\scripts\simulate_uart.ps1 > fpga_viz\logs\simulation_uart.log 2>&1
set RC=%ERRORLEVEL%
:done
(echo EXIT=%RC%)> fpga_viz\logs\simulations.status
exit /b %RC%
