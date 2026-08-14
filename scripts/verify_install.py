#!/usr/bin/env python
"""Verify the Python package, C++ rules engine, and bundled champion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import alpha_go_cpp
import torch


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = (
    ROOT
    / "local_data/checkpoints/2026-07-31_11-39-local-9x9-k7-pcr-001"
    / "iter1002-candidate.pt"
)
EXPECTED_SHA256 = "3bd995e19240b7d2e6511cbdba5a94b381c01a910ed974e97011704541b9467f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    board = alpha_go_cpp.GoBoard(9, 7.5)
    assert board.play(4, 4)
    assert CHECKPOINT.is_file(), f"missing checkpoint: {CHECKPOINT}"
    actual_sha256 = sha256_file(CHECKPOINT)
    assert actual_sha256 == EXPECTED_SHA256, actual_sha256

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["completed"] is True
    assert int(payload["iteration"]) == 1002
    assert payload["model_config"]["class"] == "SignedKomiGoResNet"

    print("AutoGo Conda verification passed")
    print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    print(f"Champion: iter1002; SHA-256: {actual_sha256}")


if __name__ == "__main__":
    main()
