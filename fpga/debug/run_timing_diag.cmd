@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
echo RUNNING> fpga\logs\timing_diag.status
call "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" -mode batch -nojournal -nolog -source fpga/debug/timing_diag.tcl > fpga\logs\timing_diag.log 2>&1
(echo DONE)> fpga\logs\timing_diag.status
