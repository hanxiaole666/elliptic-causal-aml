"""Acquire the public Elliptic dataset, or skip cleanly when it already exists.

The official Kaggle endpoint returns a ZIP without requiring an API token in
this environment. No dataset is committed to the repository because the
original data are released under CC BY-NC-ND 4.0.
"""
from __future__ import annotations

import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Iterable

import requests

from ..utils.io import sha256_file

DEFAULT_URL = "https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set"
REQUIRED = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)
CHUNK = 1024 * 1024


def _find_required(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in REQUIRED:
        matches = list(root.rglob(name))
        if matches:
            found[name] = matches[0]
    return found


def _write_sums(data_dir: Path, paths: dict[str, Path]) -> None:
    lines = [f"{sha256_file(path)}  {name}" for name, path in sorted(paths.items())]
    (data_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_dataset(data_dir: str | Path, url: str = DEFAULT_URL, max_retries: int = 5) -> dict[str, Path]:
    """Download with HTTP Range resume, extract, and return canonical CSV paths."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "archive.zip"
    part_path = data_dir / "archive.zip.part"

    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        expected = None
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            head.raise_for_status()
            expected = int(head.headers.get("Content-Length", 0)) or None
        except Exception:
            expected = None

        for attempt in range(1, max_retries + 1):
            existing = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            mode = "ab" if existing else "wb"
            try:
                with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                    response.raise_for_status()
                    if existing and response.status_code == 200:
                        mode = "wb"
                    total = int(response.headers.get("Content-Length", 0)) or expected
                    with open(part_path, mode) as handle:
                        for chunk in response.iter_content(chunk_size=CHUNK):
                            if chunk:
                                handle.write(chunk)
                    if total and part_path.stat().st_size < total:
                        raise IOError(
                            f"incomplete download: {part_path.stat().st_size}/{total} bytes"
                        )
                os.replace(part_path, zip_path)
                break
            except Exception as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Dataset download failed after {max_retries} attempts: {exc}. "
                        f"Manual alternative: kaggle datasets download -d ellipticco/elliptic-data-set "
                        f"and unzip into {data_dir}"
                    ) from exc
                time.sleep(2 * attempt)

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {zip_path}")

    extract_dir = data_dir / "extracted"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    found = _find_required(extract_dir)
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise RuntimeError(f"ZIP is missing required files: {missing}")

    paths: dict[str, Path] = {}
    for name, source in found.items():
        target = data_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        paths[name] = target
    _write_sums(data_dir, paths)
    return paths


def ensure_data(
    data_dir: str | Path,
    url: str = DEFAULT_URL,
    download: bool = True,
    allow_toy: bool = False,
) -> tuple[dict[str, Path] | None, str]:
    """Return (paths, dataset_version). Existing files always win over download."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: data_dir / name for name in REQUIRED}
    if all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        _write_sums(data_dir, paths)
        return paths, "elliptic"

    if not download:
        if allow_toy:
            return None, "toy"
        raise FileNotFoundError(
            f"Missing Elliptic CSV files in {data_dir}. "
            "Run with --download or place the three files there manually."
        )

    try:
        downloaded = download_dataset(data_dir, url)
        return downloaded, "elliptic"
    except Exception as exc:
        if allow_toy:
            print(f"[WARN] Dataset acquisition failed ({exc}); using TOY data. TOY results are not paper evidence.")
            return None, "toy"
        raise


def dataset_fingerprint(paths: Iterable[Path]) -> str:
    return ";".join(f"{p.name}:{sha256_file(p)}" for p in paths)
