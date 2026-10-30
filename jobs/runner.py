"""Background job runner for the Basys3Classifier project.

Runs ONLY the fixed project tasks listed in TASKS below. A job is a small JSON file
placed in jobs/queue/ naming one of those tasks, e.g. {"task": "fpga_program"}.
Nothing in a job file can add or change a command line, except the validated run
folder name for the detector final test.

Jobs in the same lane run one after another, in file-name order; the 'fpga' and
'detector' lanes run in parallel. Close this window (or Ctrl+C) to stop the runner;
jobs already started keep running in their own windows/logs.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
JOBS = PROJECT / "jobs"
QUEUE, RUNNING, DONE, LOGS = (JOBS / n for n in ("queue", "running", "done", "logs"))
PY = str(PROJECT / ".venv" / "Scripts" / "python.exe")
VIVADO = r"C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat"

TASKS = {
    # name: (lane, command)
    "fpga_detect_board": ("fpga", [PY, "-B", "-c",
        "from serial.tools import list_ports;import json;print(json.dumps([{'port':p.device,'vid':p.vid,'pid':p.pid,'serial':p.serial_number,'desc':p.description} for p in list_ports.comports()]))"]),
    "fpga_probe": ("fpga", ["cmd", "/c", VIVADO, "-mode", "batch", "-nojournal", "-nolog", "-source", "fpga/scripts/probe.tcl"]),
    "fpga_simulations": ("fpga", ["cmd", "/c", r"fpga\scripts\run_simulations.cmd"]),
    "fpga_build": ("fpga", ["cmd", "/c", r"fpga\scripts\run_build.cmd"]),
    "fpga_build_resume": ("fpga", ["cmd", "/c", r"fpga\scripts\run_build_resume.cmd"]),
    "fpga_program": ("fpga", ["cmd", "/c", r"fpga\scripts\run_program.cmd"]),
    "fpga_selftest": ("fpga", ["cmd", "/c", r"fpga\scripts\run_selftest.cmd"]),
    "fpga_benchmark": ("fpga", ["cmd", "/c", r"fpga\scripts\run_benchmark.cmd"]),
    "detector_smoke": ("detector", ["cmd", "/c", r"detection\run_improve_smoke.cmd"]),
    "detector_sweep": ("detector", ["cmd", "/c", r"detection\run_improve_sweep.cmd"]),
    "detector_final_test": ("detector", None),  # built below from a validated run name
    "detector_finetune_webcam": ("detector", ["cmd", "/c", r"detection\run_webcam_finetune.cmd"]),
    "verify_original_hashes": ("detector", [PY, "-B", str(Path(__file__).resolve().parent / "verify_originals.py")]),
    "git_prepare": ("detector", [PY, "-B", str(Path(__file__).resolve().parent / "git_repo.py"), "prepare"]),
    "git_commit": ("detector", [PY, "-B", str(Path(__file__).resolve().parent / "git_repo.py"), "commit"]),
}
RUN_NAME = re.compile(r"^(improved_v[0-9]+_[a-z0-9_]+|webcam_ft_v[0-9]+)$")


def now():
    return datetime.now(timezone.utc).isoformat()


def clean_env():
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    return env


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8"); tmp.replace(path)


def command_for(job):
    task = job.get("task")
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}")
    lane, cmd = TASKS[task]
    if task == "detector_final_test":
        run = str(job.get("run", ""))
        if not RUN_NAME.match(run) or not (PROJECT / "detection" / "runs" / run).is_dir():
            raise ValueError("detector_final_test needs a valid existing improved run folder name")
        cmd = [PY, "-B", "-m", "detection.improve", "--final-test", str(Path("detection") / "runs" / run)]
    return lane, cmd


def disable_quick_edit():
    """A click in a Windows console with QuickEdit on pauses this process; turn it off."""
    try:
        import ctypes
        kernel = ctypes.windll.kernel32; handle = kernel.GetStdHandle(-10); mode = ctypes.c_uint32()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel.SetConsoleMode(handle, (mode.value & ~0x0040) | 0x0080)  # clear QUICK_EDIT, keep EXTENDED_FLAGS
    except Exception:
        pass


def main():
    disable_quick_edit()
    for d in (QUEUE, RUNNING, DONE, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    active = {}  # lane -> (job_id, Popen, record, log handle)
    print(f"Job runner watching {QUEUE}. Allowed tasks: {', '.join(TASKS)}", flush=True)
    while True:
        # Finish completed jobs.
        for lane, (job_id, proc, record, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log.close(); record.update({"status": "finished", "exit_code": code, "finished": now()})
                write(DONE / f"{job_id}.json", record); (RUNNING / f"{job_id}.json").unlink(missing_ok=True)
                print(f"{record['finished']} done {job_id} ({record['task']}) exit {code}", flush=True)
                del active[lane]
        # Start the next queued job for each idle lane.
        for path in sorted(QUEUE.glob("*.job")):
            job_id = path.stem
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                lane, cmd = command_for(job)
            except Exception as error:  # reject, never execute
                write(DONE / f"{job_id}.json", {"job": job_id, "status": "rejected", "error": str(error), "finished": now()})
                path.unlink(missing_ok=True); print(f"rejected {job_id}: {error}", flush=True); continue
            if lane in active:
                continue
            log_path = LOGS / f"{job_id}.log"; log = open(log_path, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd, cwd=PROJECT, env=clean_env(), stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            record = {"job": job_id, "task": job["task"], "lane": lane, "status": "running", "started": now(),
                      "pid": proc.pid, "log": str(log_path.relative_to(PROJECT))}
            write(RUNNING / f"{job_id}.json", record); path.unlink(missing_ok=True)
            active[lane] = (job_id, proc, record, log)
            print(f"{record['started']} started {job_id} ({job['task']}) pid {proc.pid}", flush=True)
        write(JOBS / "runner_heartbeat.json", {"time": now(), "pid": os.getpid(),
                                               "active": {l: v[0] for l, v in active.items()}})
        time.sleep(3)


if __name__ == "__main__":
    main()
