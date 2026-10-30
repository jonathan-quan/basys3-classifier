"""Re-hash every file in detection/original_files_sha256.json and report changes."""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
inventory = json.loads((PROJECT / "detection" / "original_files_sha256.json").read_text(encoding="utf-8"))
missing, changed = [], []
for rel, expected in inventory.items():
    path = PROJECT / rel
    if not path.exists():
        missing.append(rel); continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != expected:
        changed.append(rel)
report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "inventoried": len(inventory),
          "unchanged": len(inventory) - len(missing) - len(changed), "missing": missing, "changed": changed}
(PROJECT / "fpga" / "original_files_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}))
print("ALL ORIGINAL FILES UNCHANGED" if not missing and not changed else "ORIGINAL FILES CHANGED OR MISSING")
sys.exit(0 if not missing and not changed else 1)
