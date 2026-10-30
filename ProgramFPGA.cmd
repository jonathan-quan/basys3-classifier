@echo off
title Program Basys 3 Classifier
cd /d "%~dp0"
call "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" -mode batch -nojournal -nolog -source fpga/scripts/program.tcl
if errorlevel 1 (
    echo Programming failed. Check USB PROG connection and board power.
    pause
    exit /b 1
)
echo Bitstream loaded into volatile FPGA configuration. SPI flash was not changed.
pause
