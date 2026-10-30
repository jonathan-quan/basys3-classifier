@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga\logs mkdir fpga\logs
echo RUNNING> fpga\logs\build.status
call "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" -mode batch -nojournal -nolog -source fpga/scripts/build.tcl > fpga\build.log 2>&1
set RC=1
findstr /C:"BITSTREAM_READY" fpga\build.log >nul && if exist fpga\build\build_status.json if exist fpga\build\basys3_classifier.bit set RC=0
(echo EXIT=%RC%)> fpga\logs\build.status
exit /b %RC%
