#!/usr/bin/env python3
"""Capture a machine-readable software/hardware environment snapshot."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch_geometric

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "test_environment"


def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pyg": torch_geometric.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nvidia_smi": command(["nvidia-smi"]),
        "pdflatex": command(["pdflatex", "--version"]).splitlines()[:3],
    }
    (OUT / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "nvidia-smi.txt").write_text(payload["nvidia_smi"] + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
