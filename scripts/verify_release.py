#!/usr/bin/env python3
"""Verify that the release archives are self-contained and leakage-free."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.namelist()


def verify_tifs():
    zip_path = RELEASE / "TIFS-LaTeX_source_full.zip"
    names = members(zip_path)
    required = [
        "main.tex", "main.pdf", "IEEEtran.cls", "references.bib", "numbers.tex",
        "description.txt", "SOURCE_MANIFEST.json",
        "figures/protocol_gap.png", "figures/edge_worlds.png",
        "figures/drift.png", "figures/pareto.png",
        "tables/Table1_main.tex", "tables/Table2_worlds.tex", "tables/Table3_stats.tex",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise AssertionError(f"TIFS zip missing: {missing}")
    if any(name.startswith("data/raw/") for name in names):
        raise AssertionError("TIFS zip must not contain raw dataset files")
    return len(names)


def verify_open_source():
    zip_path = RELEASE / "elliptic-causal-aml-open-source.zip"
    names = members(zip_path)
    required = [
        "README.md", "LICENSE", "CITATION.cff", "CONTRIBUTING.md", ".gitignore",
        "scripts/run_pipeline.py", "scripts/summarize_results.py",
        "scripts/render_manuscript.py", "scripts/build_tifs_package.py",
        "tests/test_pipeline.py",
        "outputs/full/tables/Table1_main.csv",
        "outputs/full/tables/statistical_tests.json",
        "outputs/full/figures/drift.png",
        "release/test_logs/pytest.log",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise AssertionError(f"open-source zip missing: {missing}")
    forbidden_prefixes = ("data/raw/", "outputs/full/checkpoints/", "outputs/full_unstandardized_archive/")
    forbidden_suffixes = (".csv.zip",)
    for name in names:
        if name.startswith(forbidden_prefixes):
            raise AssertionError(f"open-source zip contains forbidden path: {name}")
        if name.endswith(".pt"):
            raise AssertionError(f"open-source zip contains checkpoint: {name}")
    if any(name.endswith(forbidden_suffixes) for name in names):
        raise AssertionError("open-source zip contains dataset archive")
    return len(names)


def verify_tests_logs():
    zip_path = RELEASE / "elliptic-causal-aml-tests-logs.zip"
    names = members(zip_path)
    required = ["release/test_logs/pytest.log", "outputs/full/tables/Table1_main.csv"]
    missing = [name for name in required if name not in names]
    if missing:
        raise AssertionError(f"tests/logs zip missing: {missing}")
    if any(name.startswith("data/raw/") for name in names):
        raise AssertionError("tests/logs zip must not contain raw dataset files")
    return len(names)


def verify_manifest():
    manifest_path = RELEASE / "GITHUB_RELEASE_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in payload["sha256"].items():
        actual = sha256(RELEASE / name)
        if actual != expected:
            raise AssertionError(f"checksum mismatch for {name}")
    return len(payload["sha256"])


def main():
    counts = {
        "TIFS-LaTeX_source_full.zip": verify_tifs(),
        "elliptic-causal-aml-open-source.zip": verify_open_source(),
        "elliptic-causal-aml-tests-logs.zip": verify_tests_logs(),
        "release_manifest_checksums": verify_manifest(),
    }
    print(json.dumps({"status": "ok", **counts}, indent=2))


if __name__ == "__main__":
    main()
