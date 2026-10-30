@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
echo RUNNING> fpga\logs\debug_x.status
set T=C:\AMDDesignTools\2026.1\Vivado\bin
(
call "%T%\xvlog.bat" --sv -i fpga/export fpga/rtl/classifier_core.sv fpga/debug/tb_debug_x.sv
call "%T%\xelab.bat" tb_debug_x -s debug_x_sim -debug typical
call "%T%\xsim.bat" debug_x_sim -runall
) > fpga\logs\debug_x.log 2>&1
(echo EXIT=%ERRORLEVEL%)> fpga\logs\debug_x.status
