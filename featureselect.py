#!/usr/bin/env python3
"""Fallback entrypoint for running the PatchTST feature selector as a script."""

from __future__ import annotations

import pathlib
import sys


def _ensure_src_on_path() -> None:
    base_dir = pathlib.Path(__file__).resolve().parent
    src_dir = base_dir / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from patchtst_featureselect.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
