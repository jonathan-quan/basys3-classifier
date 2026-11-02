@echo off
setlocal
cd /d "%~dp0..\.."
if not exist fpga_viz\logs mkdir fpga_viz\logs
if not defined PROCESSOR_ARCHITECTURE set PROCESSOR_ARCHITECTURE=AMD64
echo RUNNING> fpga_viz\logs\build_det.status
call "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" -mode batch -nojournal -nolog -source fpga_viz/scripts/build_det.tcl > fpga_viz\logs\build_det.log 2>&1
set RC=1
findstr /C:"BITSTREAM_READY" fpga_viz\logs\build_det.log >nul && if exist fpga_viz\det_build\build_status.json if exist fpga_viz\det_build\basys3_detector_viz.bit set RC=0
(echo EXIT=%RC%)> fpga_viz\logs\build_det.status
exit /b %RC%
