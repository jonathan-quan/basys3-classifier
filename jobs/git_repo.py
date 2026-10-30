"""Prepare (dry run) or commit the public GitHub repo for this project.

  prepare : git init (if needed), apply local exclude rules, stage files, write the staged list. No commit.
  commit  : same staging, then commit.
The tracked .gitignore is an inventoried original file, so extra rules go to .git/info/exclude
and the few small model files are force-added explicitly.
"""
import json, shutil, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REPORT = PROJECT / "jobs" / "git_report.json"
EXCLUDE = """# --- added for the public repo (local rules; tracked .gitignore is left unchanged) ---
# Vivado / XSim byproducts
.Xil/
xsim.dir/
*.wdb
*.jou
xsim*.log
xelab.*
xvlog.*
webtalk*
*.dcp
/clockInfo.txt
/tight_setup_hold_pins.txt
# datasets, personal webcam media, local review data
/data/
/detection/data/
*.mp4
*.zip
/recognition/*.jpg
# job runner state and local notes
/jobs/queue/
/jobs/running/
/jobs/done/
/jobs/logs/
/jobs/q*/
/jobs/*.json
/jobs/*.status
/fpga/logs/
/HANDOFF_TO_CLAUDE.md
*.tmp
"""
FORCE = ["best_model.pth", "baseline_3252.pth"] + [
    f"detection/runs/{run}/{name}" for run in ("webcam_ft_v1", "improved_v1_w128_sa")
    for name in ("best_detector.pth", "config.json", "summary.json", "history.json", "test_metrics.json")]
MESSAGE = """Basys 3 INT8 CNN classifier with verified FPGA build and webcam detector

- 6,084-parameter 4-class CNN (cube, pen, book, mouse), INT8 export
- Artix-7 RTL: BRAM activation banks, 100 MHz timing met, bit-exact on 490 test images (75.92%)
- UART client, self-test, benchmark and live webcam demo
- Laptop single-object box detector fine-tuned on webcam clips

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019kty8aSx7UwZAvf2sS9uAL
"""


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def main(mode):
    report = {"mode": mode}
    try:
        if not shutil.which("git"):
            raise RuntimeError("git was not found on PATH")
        report["git_version"] = git("--version")
        if not (PROJECT / ".git").is_dir():
            git("init", "-b", "main")
        info = PROJECT / ".git" / "info"; info.mkdir(exist_ok=True)
        (info / "exclude").write_text(EXCLUDE, encoding="utf-8")
        git("add", "-A")
        git("add", "-f", *[f for f in FORCE if (PROJECT / f).exists()])
        files = [f for f in git("ls-files", "-z").split("\0") if f]
        sizes = {f: (PROJECT / f).stat().st_size for f in files if (PROJECT / f).exists()}
        report.update(files=len(sizes), total_bytes=sum(sizes.values()),
                      largest=sorted(sizes.items(), key=lambda kv: -kv[1])[:15], file_list=sorted(sizes),
                      user_name=git("config", "user.name", check=False), user_email=git("config", "user.email", check=False))
        too_big = [f for f, s in sizes.items() if s > 50 * 2**20]
        if too_big:
            raise RuntimeError(f"files over 50 MB: {too_big}")
        if mode == "commit":
            if not report["user_name"] or not report["user_email"]:
                raise RuntimeError("git user.name / user.email are not set")
            r = subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=PROJECT, input=MESSAGE, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"git commit failed: {r.stderr.strip() or r.stdout.strip()}")
            report["commit"] = git("log", "-1", "--format=%H %s")
            report["branch"] = git("branch", "--show-current")
        report["ok"] = True
    except Exception as error:
        report.update(ok=False, error=str(error))
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "file_list"}, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prepare")
