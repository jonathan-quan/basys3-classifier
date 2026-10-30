@echo off
setlocal
set PYTHONPATH=
set PYTHONHOME=
cd /d "%~dp0.."
echo RUNNING> detection\webcam_finetune.status
.venv\Scripts\python.exe -B -m detection.improve --experiment w128_sa --init detection\runs\improved_v1_w128_sa\best_detector.pth --extra-manifest detection\data\webcam_s1_manifest.jsonl --select-source webcam --output detection\runs\webcam_ft_v1 --epochs 80 --lr 5e-4 > detection\webcam_finetune.log 2>&1
(echo EXIT=%ERRORLEVEL%)> detection\webcam_finetune.status
