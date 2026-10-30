@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
title Basys3 JTAG program
echo RUNNING> fpga\logs\program.status
call "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" -mode batch -nojournal -nolog -source fpga/scripts/program.tcl > fpga\program.log 2>&1
set RC=1
findstr /C:"PROGRAMMING_COMPLETE" fpga\program.log >nul && if exist fpga\build\program_status.json set RC=0
(echo EXIT=%RC%)> fpga\logs\program.status
exit /b %RC%
