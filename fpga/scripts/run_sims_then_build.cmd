@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
echo RUNNING> fpga\logs\sims_build.status
call fpga\scripts\run_simulations.cmd
findstr /C:"EXIT=0" fpga\logs\simulations.status >nul || ((echo SIM_FAILED)> fpga\logs\sims_build.status & exit /b 1)
call fpga\scripts\run_build.cmd
findstr /C:"EXIT=0" fpga\logs\build.status >nul && ((echo BUILD_OK)> fpga\logs\sims_build.status) || ((echo BUILD_FAILED)> fpga\logs\sims_build.status)
