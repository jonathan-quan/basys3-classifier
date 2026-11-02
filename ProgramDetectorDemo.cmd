@echo off
title Program Basys 3 - box detector + telemetry (recording demo)
cd /d "%~dp0"
if not defined PROCESSOR_ARCHITECTURE set PROCESSOR_ARCHITECTURE=AMD64
call "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" -mode batch -nojournal -nolog -source fpga_viz/scripts/program_det.tcl
if errorlevel 1 (
    echo Programming failed. Check the USB PROG/J4 cable and board power.
    pause
    exit /b 1
)
echo Detector + telemetry bitstream loaded (volatile). SPI flash was not changed.
echo FPGADetector.cmd also still works with this bitstream.
pause
