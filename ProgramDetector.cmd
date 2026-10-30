@echo off
title Program Basys 3 Box Detector
cd /d "%~dp0"
rem Vivado's loader silently exits when PROCESSOR_ARCHITECTURE is missing.
if not defined PROCESSOR_ARCHITECTURE set PROCESSOR_ARCHITECTURE=AMD64
call "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" -mode batch -nojournal -nolog -source fpga_det/scripts/program.tcl
if errorlevel 1 (
    echo Programming failed. Check USB PROG connection and board power.
    pause
    exit /b 1
)
echo Detector bitstream loaded into volatile FPGA configuration. SPI flash was not changed.
pause
