@echo off
cd /d "%~dp0"
set PYTHONPATH=
set PYTHONHOME=
title Basys3 job runner - close this window to stop it
.venv\Scripts\python.exe -B jobs\runner.py
