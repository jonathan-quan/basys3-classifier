@echo off
setlocal
rem Vivado-launched shells carry its own Python settings; use the project venv cleanly.
set PYTHONPATH=
set PYTHONHOME=
cd /d "%~dp0.."
echo RUNNING> detection\improve_sweep.status
.venv\Scripts\python.exe -B -m detection.improve --sweep r96_fc,r96_sa,r128_sa,w128_sa --tag improved_v1 --epochs 150 > detection\improve_sweep.log 2>&1
(echo EXIT=%ERRORLEVEL%)> detection\improve_sweep.status
