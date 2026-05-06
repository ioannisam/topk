#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from src.profiler import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
