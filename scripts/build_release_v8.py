#!/usr/bin/env python3
"""Create the TIFS LaTeX source archive and the GitHub open-source archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
TIFS = ROOT / "submission" / "tifs_v8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_manifest_to_tifs():
    items = []
    for path in sorted(TIFS.rglob("*")):
        if path.is_file() and path.name != "SOURCE_MANIFEST.json":
            items.append({
                "path": path.relative_to(TIFS).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    payload = {
        "journal": "IEEE Transactions on Dependable and Secure Computing",
        "entrypoint": "main.tex",
        "description": "description.txt",
        "files": items,
        "note": "Raw Elliptic CSV files are intentionally excluded.",
    }
    (TIFS / "SOURCE_MANIFEST.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_zip(target: Path, source_root: Path, include: list[Path] | None = None) -> None:
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if include is None:
            for path in sorted(source_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source_root).as_posix())
        else:
            for item in include:
                if item.is_file():
                    archive.write(item, item.relative_to(source_root).as_posix())
                elif item.is_dir():
                    for path in sorted(item.rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(source_root).as_posix())


def build_tifs_zip():
    target = RELEASE / "TIFS_v8-LaTeX_source_full.zip"
    make_zip(target, TIFS)
    return target


def build_open_source_zip():
    include = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "CONTRIBUTING.md",
        ROOT / "RUNBOOK.md",
        ROOT / ".gitignore",
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "paper",
        ROOT / "review-stage",
        ROOT / "artifacts",
        ROOT / "outputs" / "full" / "logs",
        ROOT / "outputs" / "full" / "tables",
        ROOT / "outputs" / "full" / "figures",
        ROOT / "outputs" / "full" / "results",
        ROOT / "outputs" / "full" / "manifest",
        ROOT / "release" / "test_logs",
    ]
    target = RELEASE / "elliptic-causal-aml-open-source.zip"
    make_zip(target, ROOT, include=[p for p in include if p.exists()])
    return target


def build_tests_logs_zip():
    include = [
        ROOT / "release" / "test_logs",
        ROOT / "outputs" / "full" / "logs",
        ROOT / "outputs" / "full" / "tables",
        ROOT / "outputs" / "full" / "figures",
    ]
    target = RELEASE / "elliptic-causal-aml-tests-logs.zip"
    make_zip(target, ROOT, include=[p for p in include if p.exists()])
    return target


def build_release_manifest(zips: list[Path]):
    payload = {
        "tifs_latex_source": zips[0].name,
        "github_open_source": zips[1].name,
        "tests_logs": zips[2].name,
        "sha256": {zip_path.name: sha256(zip_path) for zip_path in zips},
        "excluded_from_open_source": [
            "data/raw/*.csv",
            "data/raw/archive.zip",
            "outputs/full/checkpoints/*.pt",
            "outputs/full_unstandardized_archive/",
            "outputs/full/invalid_labelC/",
        ],
        "policy": "No raw Elliptic data are redistributed.",
    }
    (RELEASE / "GITHUB_RELEASE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="do not rerun pytest")
    args = parser.parse_args()

    RELEASE.mkdir(parents=True, exist_ok=True)
    logs = RELEASE / "test_logs"
    logs.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        import subprocess
        import sys

        with (logs / "pytest.log").open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            raise SystemExit("pytest failed; see release/test_logs/pytest.log")

    add_manifest_to_tifs()
    zips = [build_tifs_zip(), build_open_source_zip(), build_tests_logs_zip()]
    build_release_manifest(zips)
    for path in zips:
        print(f"created {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
