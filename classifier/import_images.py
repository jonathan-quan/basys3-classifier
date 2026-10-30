"""Import one manually labeled session while recording source and image rights."""
import argparse
import shutil
from pathlib import Path

from . import CLASSES
from .common import file_hash, write_json
from .prepare import EXTENSIONS, image_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", required=True, choices=CLASSES)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--raw", default="data/raw")
    args = parser.parse_args()
    for value in (args.instance, args.session):
        if any(ch in value for ch in '/\\:') or value in ("", ".", ".."):
            raise ValueError("Use simple instance/session folder names")
    source = Path(args.input).resolve()
    if not source.is_dir():
        raise ValueError("--input must be a labeled image folder")
    folder = Path(args.raw) / args.label / args.instance / args.session
    if folder.exists():
        raise FileExistsError("Choose a new session rather than overwrite existing data")
    paths = [path for path in source.rglob("*") if path.suffix.lower() in EXTENSIONS]
    if not paths:
        raise ValueError("No supported images found")
    for path in paths:
        image_fingerprints(path)
    folder.mkdir(parents=True)
    write_json(folder / "session.json", {"instance": args.label + ":" + args.instance,
               "session": args.session, "source": args.source, "license": args.license})
    for path in paths:
        shutil.copy2(path, folder / (file_hash(path)[:20] + path.suffix.lower()))
    print(f"Imported {len(paths)} input files into {folder.resolve()}; duplicates with identical names collapse")


if __name__ == "__main__":
    main()

