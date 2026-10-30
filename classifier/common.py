import hashlib
import json
import os
import platform
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Full reproducibility may still depend on hardware/library versions.
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested="auto"):
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cpu")
    try:
        # Availability alone does not prove this wheel supports the GPU architecture.
        x = torch.ones((8, 8), device="cuda")
        (x @ x).sum().item()
        return torch.device("cuda")
    except RuntimeError as error:
        if requested == "cuda":
            raise
        warnings.warn(f"CUDA execution failed; using CPU. Reason: {error}")
        return torch.device("cpu")


def environment(device=None):
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "selected_device": str(device) if device is not None else None}

